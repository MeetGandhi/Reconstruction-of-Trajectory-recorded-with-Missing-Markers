###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import lib.utils as utils
from lib.plotting import *

from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver
from mujoco_physics import HopperPhysics

from lib.utils import compute_loss_all_batches

from generate_timeseries import Periodic_1d
from torch.distributions import uniform

from torch.utils.data import DataLoader
from mujoco_physics import HopperPhysics
from physionet import PhysioNet, variable_time_collate_fn, get_data_min_max
from person_activity import PersonActivity, variable_time_collate_fn_activity

from sklearn import model_selection
import random

import tarfile
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from lib.utils import get_device

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=50)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='periodic', help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")


args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

###########################
###########################

"""
This is the main file of this project.

It contains the training function as well as the testing routine.

This file uses all the other files, such as AE.py, FlatAE.py and files from the folder utils

If you encounter any problems/bugs/issues please contact me on github
or by emailing me at tarask@kth.se for any bug reports/questions/suggestions.
"""
from __future__ import division
from __future__ import print_function

import time

from AE import use_existing_markers
from FlatAE import FlatAutoEncoder
from utils.data import *
from utils.flags import FLAGS

from tensorflow.core.protobuf import saver_pb2

SKIP = FLAGS.skip_duration      # skip first few seconds - to let motion begin
NO_GAP = FLAGS.no_gap_duration  # give all the markers for the first second

class DataInfo(object):
    """Information about the datasets

     Will be passed to the FlatAe for creating corresponding variables in the graph
    """

    def __init__(self, data_sigma, train_shape, eval_shape, max_val):
        """DataInfo initializer

        Args:
          data_sigma:   variance in the dataset
          train_shape:  dimensionality of the train dataset
          eval_shape:   dimensionality of the evaluation dataset
        """
        self._data_sigma = data_sigma
        self._train_shape = train_shape
        self._eval_shape = eval_shape
        self._max_val = max_val


def learning(data, max_val, learning_rate, batch_size, dropout):
    """ Training of the denoising autoencoder

    Returns:
      Autoencoder trained on a data provided by FLAGS from utils/flags.py
    """

    with tf.Graph().as_default():

        tf.set_random_seed(FLAGS.seed)

        start_time = time.time()

        # Read the flags
        variance = FLAGS.variance_of_noise
        num_hidden = FLAGS.num_hidden_layers
        ae_hidden_shapes = [FLAGS.network_width for j in range(num_hidden)]

        # Check if recurrency is set in the correct way
        if FLAGS.reccurent == False and FLAGS.chunk_length > 1:
            print("ERROR: Without recurrency chunk length should be 1!"
                  " Please, change flags accordingly")
            exit(1)

        # Check if the flags makes sence
        if dropout < 0 or variance < 0:
            print('ERROR! Have got negative values in the flags!')
            exit(1)

        # Get the information about the dataset
        data_info = DataInfo(data.train.sigma, data.train._sequences.shape,
                             data.test._sequences.shape, max_val)

        # Allow tensorflow to change device allocation when needed
        config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True)
        # Adjust configuration so that multiple executions are possible
        config.gpu_options.allow_growth = True

        # Start a session
        sess = tf.Session(config=config)

        # Create an autoencoder
        ae_shape = [FLAGS.frame_size * FLAGS.amount_of_frames_as_input] + ae_hidden_shapes + [
            FLAGS.frame_size * FLAGS.amount_of_frames_as_input]
        ae = FlatAutoEncoder(ae_shape, sess, batch_size, variance, data_info)
        print('\nFlat AE was created : ', ae_shape)

        # Initialize input_producer
        sess.run(tf.local_variables_initializer())

        with tf.variable_scope("Train"):

            ##############        DEFINE  Optimizer and training OPERATOR      ############

            # Define optimizers
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            # Do gradient clipping
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(ae._loss, tvars), 1e12)
            #with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients\
                       (zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())

            # Prepare for making a summary for TensorBoard
            train_error = tf.placeholder(dtype=tf.float32, shape=(), name='train_error')
            #gen_train_error = tf.placeholder(dtype=tf.float32, shape=(), name='gen_train_error')
            #dis_train_error = tf.placeholder(dtype=tf.float32, shape=(), name='dis_train_error')
            #total_train_error = tf.placeholder(dtype=tf.float32, shape=(), name='total_train_error')
            eval_error = tf.placeholder(dtype=tf.float32, shape=(), name='eval_error')
            tf.summary.scalar('Train_error', train_error)
            #tf.summary.scalar('Gen_Train_error', gen_train_error)
            #tf.summary.scalar('Dis_Train_error', dis_train_error)
            #tf.summary.scalar('Total_Train_error', total_train_error)
            train_summary_op = tf.summary.merge_all()
            eval_summary_op = tf.summary.scalar('Validation_error', eval_error)

            summary_dir = FLAGS.summary_dir
            summary_writer = tf.summary.FileWriter(summary_dir, graph=tf.get_default_graph())

            num_batches = int(data.train._num_sequences / batch_size)

            # Initialize the part of the graph with the input data
            sess.run(ae._train_data.initializer,
                     feed_dict={ae._train_data_initializer: data.train._sequences})
            sess.run(ae._valid_data.initializer,
                     feed_dict={ae._valid_data_initializer: data.test._sequences})

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Initialize variables
            sess.run(tf.global_variables_initializer())

            # Create a saver
            saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)

            # restore model, if needed
            if FLAGS.restore:
                chkpt_file = FLAGS.chkpt_dir + '/chkpt-' + str(FLAGS.chkpt_num)
                saver.restore(sess, chkpt_file)
                print("Model restored from the file " + str(chkpt_file) + '.')

            # A few initialization for the early stopping
            delta = FLAGS.delta_for_early_stopping  # error tolerance for early stopping
            best_error = 10000
            num_valid_batches = int(data.test.num_sequences / batch_size)

            try:  # running enqueue threads.

                # Train the whole network jointly
                step = 0
                print('\nWe train on ', num_batches, ' batches with ', batch_size,
                      ' training examples in each for', FLAGS.training_epochs, ' epochs...')
                print("")
                print(" ______________ ________ ")
                print("|     Epoch    |  RMSE  |")
                print("|------------  |--------|")

                while not coord.should_stop():

                    if FLAGS.continuos_gap:
                        loss_summary, loss_value = sess.run([train_op, ae._reconstruction_loss],
                                                            feed_dict={ae._mask: cont_gap_mask()})
                        #loss_summary, loss_value, gen_loss_value, dis_loss_value, total_loss_value = sess.run([train_op, ae._reconstruction_loss, ae._gen_loss, ae._dis_loss, ae._loss], feed_dict={ae._mask: cont_gap_mask()})
                    else:
                        loss_summary, loss_value = sess.run\
                            ([train_op, ae._reconstruction_loss],
                             feed_dict={ae._mask: ae._mask_generator.eval(session=ae.session)})
                        #loss_summary, loss_value, gen_loss_value, dis_loss_value, total_loss_value = sess.run([train_op, ae._reconstruction_loss, ae._gen_loss, ae._dis_loss, ae._loss], feed_dict={ae._mask: ae._mask_generator.eval(session=ae.session)})

                    train_error_ = loss_value

                    if step % num_batches == 0:
                        epoch = step * 1.0 / num_batches

                        train_summary = sess.run(train_summary_op,
                                                 feed_dict={train_error: train_error_})
                                                            #gen_train_error: gen_loss_value,
                                                            #dis_train_error: dis_loss_value,
                                                            #total_train_error: total_loss_value})

                        # Print results of screen
                        epoch_str = "| {0:3.0f} ".format(epoch)[:5]
                        percent_str = "({0:3.2f}".format(epoch * 100.0 / FLAGS.training_epochs)[:5]
                        error_str = "%) |{0:5.2f}".format(train_error_)[:13] + "|"
                        #gen_error_str = "{0:5.2f}".format(gen_loss_value)[:8] + "|"
                        #dis_error_str = "{0:5.2f}".format(dis_loss_value)[:8] + "|"
                        #total_error_str = "{0:5.2f}".format(total_loss_value)[:9] + "|"
                        print(epoch_str,percent_str,error_str)#,gen_error_str,dis_error_str,total_error_str)

                        if epoch % 5 == 0:

                            rmse = test(ae, FLAGS.data_dir + '/../test_seq/basketball.binary',
                                        max_val, mean_pose)

                            print("\nOur RMSE for basketball is : ", rmse)

                            rmse = test(ae, FLAGS.data_dir + '/../test_seq/boxing.binary',
                                        max_val, mean_pose)
                            print("\nOur RMSE for boxing is : ", rmse)

                            rmse = test(ae, FLAGS.data_dir + '/../test_seq/salto.binary',
                                        max_val, mean_pose)#, True)
                            print("\nOur RMSE for the jump turn is : ", rmse)

                        if epoch > 0:
                            summary_writer.add_summary(train_summary, step)

                            # Evaluate on the validation sequences
                            error_sum = 0
                            for valid_batch in range(num_valid_batches):
                                curr_err = sess.run\
                                    ([ae._valid_loss],
                                     feed_dict={ae._mask: ae._mask_generator.eval(session=sess)})
                                error_sum += curr_err[0]
                            new_error = error_sum / (num_valid_batches)
                            eval_sum = sess.run(eval_summary_op,
                                                feed_dict={eval_error: np.sqrt(new_error)})
                            summary_writer.add_summary(eval_sum, step)

                            # Early stopping
                            if FLAGS.Early_stopping and epoch > 20:
                                if (new_error - best_error) / best_error > delta:
                                    print('After ' + str(step) +
                                          ' steps the training started over-fitting ')
                                    break
                                if new_error < best_error:
                                    best_error = new_error

                                    # Saver for the model
                                    save_path = saver.save(sess, FLAGS.chkpt_dir + '/chkpt',
                                                           global_step=step)

                            if epoch % 5 == 0:
                                # Save for the model
                                save_path = saver.save(sess, FLAGS.chkpt_dir + '/chkpt',
                                                       global_step=step)
                                print('Done training for %d epochs, %d steps.' %
                                      (FLAGS.training_epochs, step))
                                print("The model was saved in file: %s" % save_path)

                    step += 1

            except tf.errors.OutOfRangeError:
                if not FLAGS.Early_stopping:
                    # Save the model
                    save_path = saver.save(sess, FLAGS.chkpt_dir + '/chkpt',
                                           global_step=step)
                print('Done training for %d epochs, %d steps.' % (FLAGS.training_epochs, step))
                print("The final model was saved in file: %s" % save_path)
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)

        duration = (time.time() - start_time) / 60  # in minutes, instead of seconds

        print("The training was running for %.3f  min" % (duration))

        # Save the results
        f = open(FLAGS.results_file, 'a')
        f.write('\nFor the data with ' + str(FLAGS.duration_of_a_gap) + ' gap ! '
                + ' the test error is ' + str.format("{0:.5f}", np.sqrt(new_error)))
        f.close()

        return ae

def test(ae, input_seq_file_name, max_val, mean_pose, write_skels_to_files=False):
    """
    Test our system on a particular sequence

    Args:
     ae:                    trained AE
     input_seq_file_name:   address of the binary file with a test sequence
     max_val:               max values in the dataset (for the normalization)
     mean_pose:             mean values in the dataset (for the normalization)
     write_skels_to_files:  weather we write the sequnces into a file (for further visualization)

    Returns:
     rmse                 root squared mean error
    """
    with ae.session.graph.as_default() as sess:
        sess = ae.session
        chunking_stride = FLAGS.chunking_stride


        #                    GET THE DATA

        # get input sequnce
        #print('\nRead a test sequence from the file',input_seq_file_name,'...')
        original_input = read_test_seq_from_binary(input_seq_file_name)

        visualizing = False
        if visualizing:
            visualize(original_input)

        if FLAGS.plot_error:
            # cut only interesting part of a sequence
            original_input = original_input[SKIP:SKIP +NO_GAP+FLAGS.duration_of_a_gap+NO_GAP]

        # Get a mask with very long gaps
        long_mask = cont_gap_mask(original_input.shape[0], NO_GAP, test=True)

        if long_mask.shape[1] < ae.sequence_length:
            print("ERROR! Your gap is too short for your sequence length")
            exit(0)

        mask_chunks = np.array([long_mask[0, i:i + ae.sequence_length, :] for i in
                                range(0, len(long_mask[0]) - ae.sequence_length + 1,
                                       chunking_stride)])

        # Pad with itself if it is too short
        if mask_chunks.shape[0] < ae.batch_size:
            mupliplication_factor = int(ae.batch_size / mask_chunks.shape[0]) + 1
            mask_chunks = np.tile(mask_chunks, (mupliplication_factor, 1, 1))

        # Batch those chunks
        mask_batches = np.array([mask_chunks[i:i + ae.batch_size, :] for i in
                                 range(0, len(mask_chunks) - ae.batch_size + 1, ae.batch_size)])

        if write_skels_to_files:

            # No Preprocessing!
            coords_normalized = original_input

            save_motion(original_input, input_seq_file_name + '_original.csv')

            if coords_normalized.shape[0] < ae.sequence_length:
                mupliplication_factor = int(ae.batch_size * ae.sequence_length
                                            / coords_normalized.shape[0]) + 1
                # Pad the sequence with itself in order to fill the batch completely
                coords_normalized = np.tile(coords_normalized, mupliplication_factor)
                print("Test sequence was way to short!")

            # Split it into chunks
            seq_chunks = np.array([coords_normalized[i:i + ae.sequence_length, :] for i in
                                   range(0, len(original_input) - ae.sequence_length + 1,
                                          chunking_stride)])  # Split sequence into chunks

            original_size = seq_chunks.shape[0]

            if original_size < ae.batch_size:
                mupliplication_factor = int(ae.batch_size / seq_chunks.shape[0]) + 1

                # Pad the sequence with itself in order to fill the batch completely
                seq_chunks = np.tile(seq_chunks, (mupliplication_factor, 1, 1))

            # Batch those chunks
            batches = np.array([seq_chunks[i:i + ae.batch_size, :] for i in
                                range(0, len(seq_chunks) - ae.batch_size + 1, ae.batch_size)])

            numb_of_batches = batches.shape[0]

            #                    MAKE A SEQUENCE WITH MISSING MARKERS

            output_batches = np.array([])

            # Go over all batches one by one
            for batch_numb in range(numb_of_batches):

                mask = mask_batches[batch_numb]

                # Simulate missing markers
                new_result = np.multiply(batches[batch_numb], mask)

                output_batches = np.append(output_batches, [new_result], axis=0) if \
                    output_batches.size else np.array([new_result])

            # No postprocessing
            output_sequence = reshape_from_batch_to_sequence(output_batches)

            noisy = output_sequence.reshape(-1, output_sequence.shape[-1])

            visualize(noisy)

            save_motion(noisy, input_seq_file_name + '_noisy.csv')


        #                    MAKE AN OUTPUT SEQUENCE

        # Preprocess...
        coords_minus_mean = original_input - mean_pose[np.newaxis, :]
        eps = 1e-15
        coords_normalized = np.divide(coords_minus_mean, max_val[np.newaxis, :] + eps)

        if coords_normalized.shape[0] < ae.sequence_length:
            mupliplication_factor = (ae.batch_size * ae.sequence_length /
                                     coords_normalized.shape[0]) + 1
            # Pad the sequence with itself in order to fill the batch completely
            coords_normalized = np.tile(coords_normalized, mupliplication_factor)
            print("Test sequence was way to short!")

        # Split it into chunks
        seq_chunks = np.array([coords_normalized[i:i + ae.sequence_length, :] for i in
                               range(0, len(original_input) - ae.sequence_length + 1,
                                      chunking_stride)])  # Split sequence into chunks

        # Pad with itself if it is too short
        if seq_chunks.shape[0] < ae.batch_size:
            mupliplication_factor = int(ae.batch_size / seq_chunks.shape[0]) + 1
            # Pad the sequence with itself in order to fill the batch completely
            seq_chunks = np.tile(seq_chunks, (mupliplication_factor, 1, 1))

        # Batch those chunks
        batches = np.array([seq_chunks[i:i + ae.batch_size, :] for i in
                            range(0, len(seq_chunks) - ae.batch_size + 1, ae.batch_size)])

        numb_of_batches = batches.shape[0]

        # Create an empty array for an output
        output_batches = np.array([])

        # Go over all batches one by one
        for batch_numb in range(numb_of_batches):
            if FLAGS.continuos_gap:
                output_batch, mask = sess.run([ae._valid_output, ae._mask],
                                              feed_dict={ae._valid_input_: batches[batch_numb],
                                                         ae._mask: mask_batches[batch_numb]})
            else:
                output_batch, mask = sess.run([ae._valid_output, ae._mask],
                                              feed_dict={ae._valid_input_: batches[batch_numb],
                                                         ae._mask:
                                                            ae._mask_generator.eval(session=sess)})

            # Take known values into account
            new_result = use_existing_markers(batches[batch_numb], output_batch, mask,
                                              FLAGS.defaul_value)

            output_batches = np.append(output_batches, [new_result], axis=0) if output_batches.size\
                else np.array([new_result])

        # Postprocess...
        output_sequence = reshape_from_batch_to_sequence(output_batches)

        reconstructed = convert_back_to_3d_coords(output_sequence, max_val, mean_pose)

        if write_skels_to_files:
            visualize(reconstructed, original_input)
            save_motion(reconstructed, input_seq_file_name + '_our_result.csv')

        #              CALCULATE the error for our network
        new_size = np.fmin(reconstructed.shape[0], original_input.shape[0])
        error = (reconstructed[0:new_size] - original_input[0:new_size]) * ae.scaling_factor
        # take into account only missing markers
        total_rmse = np.sqrt(((error[error > 0.000000001]) ** 2).mean())

        if FLAGS.plot_error:

            if not FLAGS.continuos_gap:
                print("ERROR! If you need to plot an error - you should have a continuosly "
                      "missing markers. Change flags.py accordingly")
                print("For example: set flag 'continuos_gap' to True")
                exit(0)

            assert FLAGS.duration_of_a_gap < error.shape[0] * FLAGS.amount_of_frames_as_input


            # Calculate error for every frame
            better_error = np.zeros([FLAGS.duration_of_a_gap + NO_GAP])
            for i in range(int(FLAGS.duration_of_a_gap/FLAGS.amount_of_frames_as_input)):

                # Convert from many frames at a time - to just one frame at at time
                if not FLAGS.reccurent:
                    new_error = error[i + int(NO_GAP/FLAGS.amount_of_frames_as_input)].\
                        reshape(-1, FLAGS.frame_size)

                    for time in range(FLAGS.amount_of_frames_as_input):
                        this_frame_err = new_error[time]
                        rmse = np.sqrt(((this_frame_err[this_frame_err > 0.000000001]) ** 2).mean())

                        if rmse > 0:
                            better_error[i * FLAGS.amount_of_frames_as_input + time+ NO_GAP] = rmse

                else:
                    this_frame_err = error[i+ NO_GAP]
                    rmse = np.sqrt(((this_frame_err[this_frame_err > 0.000000001]) ** 2).mean())
                    if rmse > 0:
                        better_error[i+ NO_GAP] = rmse

            with open(FLAGS.contin_test_file, 'w') as file_handler:
                for item in better_error:
                    file_handler.write("{}\n".format(item))
                file_handler.close()

        return total_rmse

def reshape_from_batch_to_sequence(input_batch):
    '''
    Reshape batch of overlapping sequences into 1 sequence

    Args:
         input_batch: batch of overlapping sequences
    Return:
         flat_sequence: one sequence with the same values

    '''

    # Get the data from the Flags
    chunking_stride = FLAGS.chunking_stride
    if FLAGS.reccurent:
        sequence_length = FLAGS.chunk_length
    else:
        sequence_length = FLAGS.amount_of_frames_as_input

    # Reshape batches
    input_chunks = input_batch.reshape(-1, input_batch.shape[2], input_batch.shape[3])
    numb_of_chunks = input_chunks.shape[0]

    if FLAGS.reccurent:
        # Map from overlapping windows to non-overlaping
        # Take first chunk as a whole and the last part of each other chunk

        input_non_overlaping = input_chunks[0]
        for i in range(1, numb_of_chunks, 1):

            input_non_overlaping = np.concatenate(
                (input_non_overlaping,
                 input_chunks[i][sequence_length - chunking_stride: sequence_length][:]), axis=0)

        input_non_overlaping = np.array(input_non_overlaping)

    else:
        input_non_overlaping = input_chunks.reshape(input_chunks.shape[0], 1,
                                                    sequence_length * FLAGS.frame_size)

    # Flaten it into a sequence
    flat_sequence = input_non_overlaping.reshape(-1, input_non_overlaping.shape[-1])

    return flat_sequence


def convert_back_to_3d_coords(sequence, max_val, mean_pose):
    '''
    Convert back from the normalized values between -1 and 1 to original 3d coordinates
    and unroll them into the sequence

    Args:
        sequence: sequence of the normalized values
        max_val: maximal value in the dataset
        mean_pose: mean value in the dataset

    Return:
        3d coordinates corresponding to the batch
    '''

    # Convert it back from the [-1,1] to original values
    reconstructed = np.multiply(sequence, max_val[np.newaxis, :] + 1e-15)

    # Add the mean pose back
    reconstructed = reconstructed + mean_pose[np.newaxis, :]

    # Unroll batches into the sequence
    reconstructed = reconstructed.reshape(-1, reconstructed.shape[-1])

    return reconstructed


def get_the_data():
    data, max_val, mean_pose = read_datasets_from_binary()

    # Check, if we have enough data
    if FLAGS.batch_size > data.train._num_chunks:
        print('ERROR! Cannot have less train sequences than a batch size!')
        exit(1)
    if FLAGS.batch_size > data.test._num_chunks:
        print('ERROR! Cannot have less test sequences than a batch size!')
        exit(1)

    return data, max_val, mean_pose

def cont_gap_mask(length=0, gap_begins=0, test=False):

    if not test:
        mask_size = [FLAGS.batch_size, FLAGS.chunk_length,
                     int(FLAGS.frame_size * FLAGS.amount_of_frames_as_input)]
        length = FLAGS.chunk_length
    else:
        mask_size = [1, length, int(FLAGS.frame_size * FLAGS.amount_of_frames_as_input)]

    mask = np.ones(mask_size)
    probabilities = [1.0 / (41) for marker in range(41)]

    for batch in range(mask_size[0]):

        start_fr = int(gap_begins/FLAGS.amount_of_frames_as_input)

        if test:
            if FLAGS.duration_of_a_gap:
                gap_length = FLAGS.duration_of_a_gap
            else:
                gap_length = int(length/FLAGS.amount_of_frames_as_input)
        else:
            gap_length = length

        time_fr = start_fr
        while time_fr < gap_length+start_fr:

            # choose random amount of time frames for a gap
            if FLAGS.duration_of_a_gap:
                gap_duration = FLAGS.duration_of_a_gap
            else:
                # between 0.1s and 1s (frame rate 60 fps)
                gap_duration = int(np.random.normal(120, 20))

            # choose random markers for the gap
            if FLAGS.amount_of_missing_markers < 21:
                random_markers = np.random.choice(41, FLAGS.amount_of_missing_markers,
                                                  replace=False, p=probabilities)
            else:
                random_markers = np.random.choice(41, FLAGS.amount_of_missing_markers,
                                                  replace=False)

            for gap_time in range(gap_duration):

                for muptipl_inputs in range(FLAGS.amount_of_frames_as_input):

                    for marker in random_markers:

                        mask[batch][time_fr][marker + 123*muptipl_inputs] = 0
                        mask[batch][time_fr][marker + 41+ 123*muptipl_inputs] = 0
                        mask[batch][time_fr][marker + 82+ 123*muptipl_inputs] = 0

                time_fr += 1
                if time_fr >= gap_length+start_fr:
                    break

            # Make sure not to use the same markers twice in a raw
            p = 1.0 / (41 - FLAGS.amount_of_missing_markers)
            probabilities = [0 if marker in random_markers else p for marker in range(41)]

    return mask


def save_motion(motion, file_name):
    """
    Save the motion into a csv file
    :param motion:     sequence of the motion 3d coordinates
    :param file_name:  file to write the motion into
    :return:           nothing
    """

    with open(file_name, 'w') as fp:

        if not FLAGS.reccurent:
            # Reshape input - to have just one frame at a time
            to_output = motion.reshape(-1, FLAGS.frame_size)
        else:
            to_output = motion

        np.savetxt(fp, to_output, delimiter=",")
        print("Motion was written to " + file_name)

if __name__ == '__main__':

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
	experimentID = int(SystemRandom().random()*100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')

    start = time.time()
    print("Sampling dataset of {} training examples".format(args.n))
	
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
	input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)

    utils.makedirs("results/")

    ##################################################################
    ##################################################################
    ##################################################################

    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    dropout = FLAGS.dropout  # keep probability value

    # Read the data
    data, max_val, mean_pose = read_datasets_from_binary()

    # Check, if we have enough data
    if FLAGS.batch_size > data.train.num_sequences:
        print('ERROR! Cannot have less train sequences than a batch size!')
        exit(1)
    if FLAGS.batch_size > data.test.num_sequences:
        print('ERROR! Cannot have less test sequences than a batch size!')
        exit(1)

    # Pad max values and the mean pose, if neeeded
    if FLAGS.amount_of_frames_as_input > 1:
        max_val = np.tile(max_val, FLAGS.amount_of_frames_as_input)
        mean_pose = np.tile(mean_pose, FLAGS.amount_of_frames_as_input)

    # Train the network
    ##ae = learning(data, max_val, learning_rate, batch_size, dropout)

    # TEST it
    ##rmse = test(ae, FLAGS.data_dir + '/../test_seq/boxing.binary', max_val, mean_pose, True)
    ##print("\nOur RMSE for boxing is : ", rmse)

    ##rmse = test(ae, FLAGS.data_dir + '/../test_seq/basketball.binary', max_val, mean_pose, True)
    ##print("\nOur RMSE for basketball is : ", rmse)

    # Close Tf session
    ##ae.session.close()

    ##################################################################
    ##################################################################
    ##################################################################

    # Process small datasets
    dataset = data
    dataset = dataset.to(device)
    time_steps_extrap = time_steps_extrap.to(device)

    train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

    n_samples = len(dataset)
    input_dim = dataset.size(-1)

    batch_size = min(args.batch_size, args.n)
    train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
    test_dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

	
    data_obj = {"dataset_obj": dataset,
        "train_dataloader": utils.inf_generator(train_dataloader),
        "test_dataloader": utils.inf_generator(test_dataloader),
	"input_dim": input_dim,
	"n_train_batches": len(train_dataloader),
	"n_test_batches": len(test_dataloader)}

	# Create the model
	obsrv_std = 0.01
	if args.dataset == "hopper":
		obsrv_std = 1e-3 

	obsrv_std = torch.Tensor([obsrv_std]).to(device)

	z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

	if args.rnn_vae:
		if args.poisson:
			print("Poisson process likelihood not implemented for RNN-VAE: ignoring --poisson")

		# Create RNN-VAE model
		model = RNN_VAE(input_dim, args.latents, 
			device = device, 
			rec_dims = args.rec_dims, 
			concat_mask = True, 
			obsrv_std = obsrv_std,
			z0_prior = z0_prior,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = args.linear_classif,
			n_units = args.units,
			input_space_decay = args.input_decay,
			cell = args.rnn_cell,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)


	elif args.classic_rnn:
		if args.poisson:
			print("Poisson process likelihood not implemented for RNN: ignoring --poisson")

		if args.extrap:
			raise Exception("Extrapolation for standard RNN not implemented")
		# Create RNN model
		model = Classic_RNN(input_dim, args.latents, device, 
			concat_mask = True, obsrv_std = obsrv_std,
			n_units = args.units,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = args.linear_classif,
			input_space_decay = args.input_decay,
			cell = args.rnn_cell,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)
	elif args.ode_rnn:
		# Create ODE-GRU model
		n_ode_gru_dims = args.latents
				
		if args.poisson:
			print("Poisson process likelihood not implemented for ODE-RNN: ignoring --poisson")

		if args.extrap:
			raise Exception("Extrapolation for ODE-RNN not implemented")

		ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
			n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)

		rec_ode_func = ODEFunc(
			input_dim = input_dim, 
			latent_dim = n_ode_gru_dims,
			ode_func_net = ode_func_net,
			device = device).to(device)

		z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, "euler", args.latents, 
			odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
	
		model = ODE_RNN(input_dim, n_ode_gru_dims, device = device, 
			z0_diffeq_solver = z0_diffeq_solver, n_gru_units = args.gru_units,
			concat_mask = True, obsrv_std = obsrv_std,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)
	elif args.latent_ode:
		model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
			classif_per_tp = classif_per_tp,
			n_labels = n_labels)
	else:
		raise Exception("Model not specified")

	##################################################################

	if args.viz:
		viz = Visualizations(device)

	##################################################################
	
	#Load checkpoint and evaluate the model
	if args.load is not None:
		utils.get_ckpt_model(ckpt_path, model, device)
		exit()

	##################################################################
	# Training

	log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
	logger.info(input_command)

	optimizer = optim.Adamax(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"]

	for itr in range(1, num_batches * (args.niters + 1)):
		optimizer.zero_grad()
		utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

		wait_until_kl_inc = 10
		if itr // num_batches < wait_until_kl_inc:
			kl_coef = 0.
		else:
			kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

		batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
		train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
		train_res["loss"].backward()
		optimizer.step()

		n_iters_to_viz = 1
		if itr % (n_iters_to_viz * num_batches) == 0:
			with torch.no_grad():

				test_res = compute_loss_all_batches(model, 
					data_obj["test_dataloader"], args,
					n_batches = data_obj["n_test_batches"],
					experimentID = experimentID,
					device = device,
					n_traj_samples = 3, kl_coef = kl_coef)

				message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
					itr//num_batches, 
					test_res["loss"].detach(), test_res["likelihood"].detach(), 
					test_res["kl_first_p"], test_res["std_first_p"])
		 	
				logger.info("Experiment " + str(experimentID))
				logger.info(message)
				logger.info("KL coef: {}".format(kl_coef))
				logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
				logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))
				
				if "auc" in test_res:
					logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))

				if "mse" in test_res:
					logger.info("Test MSE: {:.4f}".format(test_res["mse"]))

				if "accuracy" in train_res:
					logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))

				if "accuracy" in test_res:
					logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))

				if "pois_likelihood" in test_res:
					logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))

				if "ce_loss" in test_res:
					logger.info("CE loss: {}".format(test_res["ce_loss"]))

			torch.save({
				'args': args,
				'state_dict': model.state_dict(),
			}, ckpt_path)


			# Plotting
			if args.viz:
				with torch.no_grad():
					test_dict = utils.get_next_batch(data_obj["test_dataloader"])

					print("plotting....")
					if isinstance(model, LatentODE) and (args.dataset == "periodic"): #and not args.classic_rnn and not args.ode_rnn:
						plot_id = itr // num_batches // n_iters_to_viz
						viz.draw_all_plots_one_dim(test_dict, model, 
							plot_name = file_name + "_" + str(experimentID) + "_{:03d}".format(plot_id) + ".png",
						 	experimentID = experimentID, save=True)
						plt.pause(0.01)
	torch.save({
		'args': args,
		'state_dict': model.state_dict(),
	}, ckpt_path)
