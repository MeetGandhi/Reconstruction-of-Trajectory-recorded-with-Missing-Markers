"""
This file contains implementation of the core model: Denoising AutoEncoder
"""
from __future__ import division
from __future__ import print_function

from AE import AutoEncoder, simulate_missing_markets
from utils.data import add_noise, loss_reconstruction
from utils.flags import FLAGS

from typing import List, Tuple

#import keras.backend as K
#import keras.layers
#from keras import optimizers
#from keras.engine.topology import Layer
import tensorflow as tf
from tensorflow.python.keras.layers import Activation, Lambda
from tensorflow.python.keras.layers import Conv1D, SpatialDropout1D
from tensorflow.python.keras.layers import Dense, BatchNormalization
#from keras.models import Input, Model


def residual_block(x, dilation_rate, nb_filters, kernel_size, padding, activation='relu', dropout_rate=0,
                   kernel_initializer='he_normal', use_batch_norm=False):
    # type: (Layer, int, int, int, str, str, float, str, bool) -> Tuple[Layer, Layer]
    """Defines the residual block for the WaveNet TCN

    Args:
        x: The previous layer in the model
        dilation_rate: The dilation power of 2 we are using for this residual block
        nb_filters: The number of convolutional filters to use in this block
        kernel_size: The size of the convolutional kernel
        padding: The padding used in the convolutional layers, 'same' or 'causal'.
        activation: The final activation used in o = Activation(x + F(x))
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
        use_batch_norm: Whether to use batch normalization in the residual layers or not.
    Returns:
        A tuple where the first element is the residual model layer, and the second
        is the skip connection.
    """
    prev_x = x
    for k in range(2):
        x = Conv1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   padding=padding)(x)
        if use_batch_norm:
            x = BatchNormalization()(x)  # TODO should be WeightNorm here, but using batchNorm instead
        x = Activation('relu')(x)
        x = SpatialDropout1D(rate=dropout_rate)(x)

    # 1x1 conv to match the shapes (channel dimension).
    prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
    res_x = tf.keras.layers.add([prev_x, x])
    res_x = Activation(activation)(res_x)
    return res_x, x


def process_dilations(dilations):
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations

    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations


class TCN:
    """Creates a TCN layer.

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            name: Name of the model. Useful when having multiple TCN.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.

        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=[1, 2, 4, 8, 16, 32],
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='linear',
                 name='tcn',
                 kernel_initializer='he_normal',
                 use_batch_norm=False):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm

        dilations = process_dilations(dilations)

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(x, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(x)')
            print('The alternative is to downgrade to 2.1.2 (pip install keras-tcn==2.1.2).')
            raise Exception()

    def __call__(self, inputs):
        x = inputs
        # 1D FCN.
        x = Conv1D(self.nb_filters, 1, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for d in self.dilations:
                x, skip_out = residual_block(x,
                                             dilation_rate=d,
                                             nb_filters=self.nb_filters,
                                             kernel_size=self.kernel_size,
                                             padding=self.padding,
                                             activation=self.activation,
                                             dropout_rate=self.dropout_rate,
                                             kernel_initializer=self.kernel_initializer,
                                             use_batch_norm=self.use_batch_norm)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = tf.keras.layers.add(skip_connections)
        if not self.return_sequences:
            x = Lambda(lambda tt: tt[:, -1, :])(x)
        return x


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    #tf.summary.histogram('histogram', var)
    
##def build_generator(z_prior,keep_prob):
##    z_prior=tf.unstack(z_prior,FLAGS.chunk_length,1)
##    lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(FLAGS.n_hidden), output_keep_prob=keep_prob)for _ in range(FLAGS.g_num_layers)])
##    with tf.variable_scope("gen") as gen:
##      res, states = tf.contrib.rnn.static_rnn(lstm_cell, z_prior, dtype=tf.float32)
##      weights=tf.Variable(tf.random_normal([n_hidden, FLAGS.frame_size * FLAGS.amount_of_frames_as_input]))
##      biases=tf.Variable(tf.random_normal([FLAGS.frame_size * FLAGS.amount_of_frames_as_input]))
##      for i in range(len(res)):
##        res[i]=tf.nn.tanh(tf.matmul(res[i], weights) + biases)
##      g_params=[v for v in tf.global_variables() if v.name.startswith(gen.name)]
##    with tf.name_scope("gen_params"):
##      for param in g_params:
##        variable_summaries(param)
##    return res,g_params
##
##def build_discriminator(x_data, x_generated, keep_prob):
##    x_data=tf.unstack(x_data,seq_size,1)
##    x_generated=list(x_generated)
##    x_in = tf.concat([x_data, x_generated],1)
##    x_in=tf.unstack(x_in,seq_size,0)
##    lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(FLAGS.n_hidden), output_keep_prob=keep_prob) for _ in range(FLAGS.d_num_layers)])
##    with tf.variable_scope("dis") as dis:
##      weights=tf.Variable(tf.random_normal([FLAGS.n_hidden, 1]))
##      biases=tf.Variable(tf.random_normal([1]))
##      outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x_in, dtype=tf.float32)
##      res=tf.matmul(outputs[-1], weights) + biases
##      y_data = tf.nn.sigmoid(tf.slice(res, [0, 0], [batch_size, -1], name=None))
##      y_generated = tf.nn.sigmoid(tf.slice(res, [batch_size, 0], [-1, -1], name=None))
##      d_params=[v for v in tf.global_variables() if v.name.startswith(dis.name)]
##    with tf.name_scope("desc_params"):
##      for param in d_params:
##        variable_summaries(param)
##    return y_data, y_generated, d_params

class FlatAutoEncoder(AutoEncoder):
    """Flat autoencoder.
    It has all-to-all connections at each layer

    The user specifies the structure of the neural net
    by specifying number of inputs, the number of hidden
    units for each layer and the number of final outputs.
    All this information is set in the utils/flags.py file.
    """

    def __init__(self, shape, sess, batch_size, variance_coef, data_info):
        """Autoencoder initializer

        Args:
          shape:          list of ints specifying
                          num input, hidden1 units,...hidden_n units, num outputs
          sess:           tensorflow session object to use
          batch_size:     batch size
          varience_coef:  multiplicative factor for the variance of noise wrt the variance of data
          data_info:      key information about the dataset
        """

        AutoEncoder.__init__(self, len(shape) - 2, batch_size, FLAGS.chunk_length, sess, data_info)

        self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]

        self._gen_shape = FLAGS.gen_layers

        self._dis_shape = FLAGS.dis_layers

        self.__variables = {}


        with sess.graph.as_default():

            with tf.variable_scope("AE_Variables"):

                ##############        SETUP VARIABLES       #####################################

                #for i in range(len(self._gen_shape)-1):  # go over all layers

                    # create variables for matrices and biases for each layer
                    #self._gen_create_variables(i, FLAGS.Weight_decay)

                    
                #for i in range(len(self._dis_shape)-1):  # go over all layers

                    # create variables for matrices and biases for each layer
                    #self._dis_create_variables(i, FLAGS.Weight_decay)


                #if FLAGS.reccurent:

                    # Define LSTM cell
                    #lstm_sizes = self.__shape[1:]

                    #def lstm_cell(size):
                    #    basic_cell = tf.contrib.rnn.BasicLSTMCell(
                    #        size, forget_bias=1.0, state_is_tuple=True, reuse=tf.AUTO_REUSE)
                    #    # Apply dropout on the hidden layers
                    #    if size != self.__shape[-1]:
                    #        hidden_cell = tf.contrib.rnn.DropoutWrapper\
                    #            (cell=basic_cell, output_keep_prob=FLAGS.dropout)
                    #        return hidden_cell
                    #    else:
                    #        return basic_cell

                    #self._gen_lstm_cell = tf.contrib.rnn.MultiRNNCell(
                    #    [lstm_cell(sz) for sz in lstm_sizes], state_is_tuple=True)
                    #self._dis_lstm_cell = tf.contrib.rnn.MultiRNNCell(
                    #    [lstm_cell(sz) for sz in lstm_sizes], state_is_tuple=True)
                    

                ##############        DEFINE THE NETWORK     ###################################

                # Declare a mask for simulating missing_values
                self._mask = tf.placeholder(dtype=tf.float32,
                                            shape=[FLAGS.batch_size, FLAGS.chunk_length,
                                                   FLAGS.frame_size *
                                                   FLAGS.amount_of_frames_as_input],
                                            name='Mask_of_mis_markers')
                self._mask_generator = self.binary_random_matrix_generator(FLAGS.missing_rate)

                # Reminder: we use Denoising AE
                # (http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)

                ''' 1 - Setup network for TRAINing '''
                # Input noisy data and reconstruct the original one
                self._input_ = add_noise(self._train_batch, variance_coef, data_info._data_sigma)
                self._target_ = self._train_batch

                # Define output and loss for the training data
                # self._output = self.construct_graph(self._input_, FLAGS.dropout)
                # self._output, self._gen_params_ = self.construct_gen_graph(self._input_, FLAGS.dropout)
                #self._output, self._gen_params_ = self.construct_gen_graph(self._input_, FLAGS.dropout)
                self._output, self._tcn_params_ = self.construct_tcn_graph(self._input_, FLAGS.dropout)
                #self._y_data_, self._y_generated_, self._dis_params_ = self.construct_dis_graph(self._input_, self._output, FLAGS.dropout)
                #self._y_data_, self._y_generated_, self._dis_params_ = self.construct_dis_graph(self._target_, self._output, FLAGS.dropout)
                print("self._output=",self._output)
                #print("self._target_=",self._target_)
                print("self._tcn_params_=",self._tcn_params_)
                self._reconstruction_loss = loss_reconstruction(self._output, self._target_, self.max_val)
                # self._gen_loss = tf.reduce_mean(tf.square(- tf.log(self._y_generated_)))
                # self._dis_loss = tf.reduce_mean(tf.square(- (tf.log(self._y_data_) + tf.log(1 - self._y_generated_))))
                #self._gen_loss = - tf.reduce_mean(self._y_generated_)
                #self._dis_loss = - tf.reduce_mean(self._y_data_) + tf.reduce_mean(self._y_generated_)

                # Gradient Penalty
                #self.epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
                #self.epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=1.)
                #X_hat = self._target_ + self.epsilon * (self._output - self._target_)
                #D_X_hat, _, __ = self.construct_dis_graph(X_hat, self._output, FLAGS.dropout)
                #grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
                #red_idx = [i for i in range(1, X_hat.shape.ndims)]
                #slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
                #gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                #self._dis_loss = self._dis_loss + 10.0 * gradient_penalty
        
                #self.reg = tf.contrib.layers.apply_regularization(
                #    tf.contrib.layers.l1_regularizer(2.5e-5),
                #    weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
                #)
                #self._gen_loss = self._gen_loss + self.reg
                #self._dis_loss = self._dis_loss + self.reg
        
                tf.add_to_collection('losses', self._reconstruction_loss)
                #tf.add_to_collection('losses', self._gen_loss)
                #tf.add_to_collection('losses', self._dis_loss)
                self._loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

                ''' 2 - Setup network for TESTing '''
                self._valid_input_ = self._valid_batch
                self._valid_target_ = self._valid_batch

                # Define output
                #self._valid_output = self.construct_graph(self._valid_input_, 1)
                self._valid_output, self._valid_tcn_params_ = self.construct_tcn_graph(self._valid_input_, 1)

                # Define loss
                self._valid_loss = loss_reconstruction(self._valid_output, self._valid_target_,
                                                       self.max_val)

    def construct_tcn_graph(self, input_seq_pl, dropout):

        """ Contruct a tensorflow graph for the network

        Args:
          input_seq_pl:     tf placeholder for ae input data [batch_size, sequence_length, DoF]
          dropout:          how much of the input neurons will be activated, value in range [0,1]
        Returns:
          Tensor of output
        """

        network_input = simulate_missing_markets(input_seq_pl, self._mask, self.default_value)

        if FLAGS.reccurent is False:
            last_output = network_input[:, 0, :]

            numb_layers = self.num_hidden_layers + 1

            # Pass through the network
            for i in range(numb_layers):
                # First - Apply Dropout
                last_output = tf.nn.dropout(last_output, dropout)

                w = self._w(i + 1)
                b = self._b(i + 1)

                last_output = self._activate(last_output, w, b)

            output = tf.reshape(last_output, [self.batch_size, 1,
                                              FLAGS.frame_size * FLAGS.amount_of_frames_as_input])

        else:            
           # output, last_states = tf.nn.dynamic_rnn(
           #     cell=self._RNN_cell,
           #     dtype=tf.float32,
           #     inputs=network_input)

           z_prior = tf.convert_to_tensor(tf.unstack(network_input, FLAGS.chunk_length, 1))
           print("z_prior=", z_prior.shape)
           
           with tf.variable_scope("tcn") as tcn:
              #g_trainingState = tf.placeholder(tf.bool)
              #g_res, g_states = tf.contrib.rnn.static_rnn(self._gen_lstm_cell, z_prior, dtype=tf.float32)
              ##g_res, g_states = tf.nn.dynamic_rnn(self._gen_lstm_cell, z_prior, dtype=tf.float32)

              #dilations = process_dilations(FLAGS.dilations)
              #input_layer = Input(shape=(max_len, num_feat))
              t_res = TCN(nb_filters=123,
                          kernel_size=1,
                          nb_stacks=1,
                          dilations=[2 ** i for i in range(1)],
                          padding='causal', #same: ~40
                          use_skip_connections=False,
                          dropout_rate=0.0,
                          return_sequences=True,
                          activation='relu',
                          name='tcn',
                          kernel_initializer='he_normal',
                          use_batch_norm=True
                          )(z_prior)
              t_res = Dense(123)(t_res)
              t_res = Activation('relu')(t_res)
              print('t_res.shape=', t_res.shape)
              
              t_last_output = t_res

              #g_w_1 = self._w("g" + str(1))
              #g_b_1 = self._b("g" + str(1))
              #
              #g_w_2 = self._w("g" + str(2))
              #g_b_2 = self._b("g" + str(2))
              #
              #g_hidden_1 = tf.nn.bias_add(tf.matmul(g_last_output, g_w_1),g_b_1)
              #g_bnHidden_1 = tf.layers.batch_normalization(g_hidden_1, training = True)
              #g_last_output_1 = tf.nn.leaky_relu(g_bnHidden_1)
              #
              #g_hidden_2 = tf.nn.bias_add(tf.matmul(g_last_output_1, g_w_2),g_b_2)
              #g_bnHidden_2 = tf.layers.batch_normalization(g_hidden_2, training = True)
              #g_last_output_2 = tf.nn.leaky_relu(g_bnHidden_2)
              
              #for i in range(len(self._gen_shape)-1):
              #  g_w = self._w("g" + str(i + 1))
              #  g_b = self._b("g" + str(i + 1))
              #  g_hidden = tf.nn.bias_add(tf.matmul(g_last_output, g_w),g_b)
              #  g_hidden_mean, g_hidden_variance = tf.nn.moments(g_hidden, axes = [i for i in range(len(g_hidden.shape))], keep_dims = True)
              #  g_bnHidden = tf.nn.batch_normalization(g_hidden, mean = g_hidden_mean,
              #                                         variance = g_hidden_variance,
              #                                         variance_epsilon = FLAGS.variance_epsilon,
              #                                         offset = None,
              #                                         scale = None)
                #g_bnHidden = tf.layers.batch_normalization(g_hidden, training = True)
              #  g_last_output = tf.nn.leaky_relu(g_bnHidden)
              #  print(g_last_output)

              #g_output = tf.nn.tanh(g_last_output)
              t_output = t_last_output
          
              t_params = [v for v in tf.global_variables() if v.name.startswith(tcn.name)]
              
              t_output = tf.reshape(t_output, [self.batch_size, FLAGS.chunk_length,
                                              FLAGS.frame_size * FLAGS.amount_of_frames_as_input])
              
           with tf.name_scope("tcn_params"):
              for param in t_params:
                  variable_summaries(param)

            # Reuse variables
            # so that we can use the same LSTM both for training and testing
           tf.get_variable_scope().reuse_variables()

           return t_output, t_params

    def construct_dis_graph(self, input_seq_pl, generated_seq_pl, dropout):

        """ Contruct a tensorflow graph for the network

        Args:
          input_seq_pl:     tf placeholder for ae input data [batch_size, sequence_length, DoF]
          dropout:          how much of the input neurons will be activated, value in range [0,1]
        Returns:
          Tensor of output
        """

        #network_input = simulate_missing_markets(input_seq_pl, self._mask, self.default_value)

        if FLAGS.reccurent is False:
            last_output = network_input[:, 0, :]

            numb_layers = self.num_hidden_layers + 1

            # Pass through the network
            for i in range(numb_layers):
                # First - Apply Dropout
                last_output = tf.nn.dropout(last_output, dropout)

                w = self._w(i + 1)
                b = self._b(i + 1)

                last_output = self._activate(last_output, w, b)

            output = tf.reshape(last_output, [self.batch_size, 1,
                                              FLAGS.frame_size * FLAGS.amount_of_frames_as_input])

        else:            
           # output, last_states = tf.nn.dynamic_rnn(
           #     cell=self._RNN_cell,
           #     dtype=tf.float32,
           #     inputs=network_input)

            x_data = tf.unstack(input_seq_pl,FLAGS.chunk_length,1)
            x_data = tf.reshape(x_data, [self.batch_size, FLAGS.chunk_length,
                                              FLAGS.frame_size * FLAGS.amount_of_frames_as_input])
            
           # x_generated = list(generated_seq_pl)
            x_generated = generated_seq_pl
            x_in = tf.concat([x_data, x_generated], 1)
           # x_in=tf.unstack(x_in,FLAGS.chunk_length,0)
            x_in = tf.convert_to_tensor(tf.unstack(x_in, self.batch_size, 0))
            print(x_in)
            with tf.variable_scope("dis") as dis:
              d_trainingState = tf.placeholder(tf.bool)
              #d_res, d_states = tf.contrib.rnn.static_rnn(self._dis_lstm_cell, x_in, dtype=tf.float32)
              ##d_res, d_states = tf.nn.dynamic_rnn(self._dis_lstm_cell, x_in, dtype=tf.float32)

              #dilations = process_dilations(FLAGS.dilations)
              #input_layer = Input(shape=(max_len, num_feat))
              d_res = TCN(nb_filters=123,
                          kernel_size=2,
                          nb_stacks=1,
                          dilations=[2 ** i for i in range(6)],
                          padding='causal',
                          use_skip_connections=False,
                          dropout_rate=0.0,
                          return_sequences=True,
                          activation='relu',
                          name='dis_tcn',
                          kernel_initializer='he_normal',
                          use_batch_norm=True
                          )(x_in)
              d_res = Dense(123)(d_res)
              d_res = Activation('relu')(d_res)
              print('d_res.shape=', d_res.shape)
              
              d_last_output = d_res
              
              #d_w_1 = self._w("d" + str(1))
              #d_b_1 = self._b("d" + str(1))
              #
              #d_w_2 = self._w("d" + str(2))
              #d_b_2 = self._b("d" + str(2))
              # 
              #d_hidden_1 = tf.nn.bias_add(tf.matmul(d_last_output, d_w_1),d_b_1)
              #d_bnHidden_1 = tf.layers.batch_normalization(d_hidden_1, training = True)
              #d_last_output_1 = tf.nn.leaky_relu(d_bnHidden_1)
              #
              #d_hidden_2 = tf.nn.bias_add(tf.matmul(d_last_output_1, d_w_2),d_b_2)
              #d_bnHidden_2 = tf.layers.batch_normalization(d_hidden_2, training = True)
              #d_last_output_2 = tf.nn.leaky_relu(d_bnHidden_2)

              for i in range(len(self._dis_shape)-1):
                #d_gamma = tf.get_variable("gamma", initializer=tf.zeros(self._dis_shape[i + 1]))
                d_w = self._w("d" + str(i + 1))
                d_b = self._b("d" + str(i + 1))
                d_hidden = tf.nn.bias_add(tf.matmul(d_last_output, d_w),d_b)
                d_hidden_mean, d_hidden_variance = tf.nn.moments(d_hidden, axes = [i for i in range(len(d_hidden.shape))], keep_dims = True)
                d_bnHidden = tf.nn.batch_normalization(d_hidden, mean = d_hidden_mean,
                                                       variance = d_hidden_variance,
                                                       variance_epsilon = FLAGS.variance_epsilon,
                                                       offset = None,
                                                       scale = None)
                #d_bnHidden = tf.layers.batch_normalization(d_hidden, training = True)
                d_last_output = tf.nn.leaky_relu(d_bnHidden)

              d_last_output = tf.reshape(d_last_output, [2*FLAGS.chunk_length, self.batch_size])  
              print(d_last_output)
              #y_data = tf.nn.sigmoid(tf.slice(res, [0, 0], [self.batch_size, -1], name=None))
              #y_generated = tf.nn.sigmoid(tf.slice(res, [self.batch_size, 0], [-1, -1], name=None))
              y_data = tf.nn.tanh(tf.slice(d_last_output, [0, 0], [FLAGS.chunk_length, -1], name=None))
              y_generated = tf.nn.tanh(tf.slice(d_last_output, [FLAGS.chunk_length, 0], [-1, -1], name=None))
              
              d_params=[v for v in tf.global_variables() if v.name.startswith(dis.name)]
              
            with tf.name_scope("disc_params"):
              for param in d_params:
                variable_summaries(param)

            # Reuse variables
            # so that we can use the same LSTM both for training and testing
            tf.get_variable_scope().reuse_variables()

            return y_data, y_generated, d_params


    def construct_graph(self, input_seq_pl, dropout):

        """ Contruct a tensofrlow graph for the network

        Args:
          input_seq_pl:     tf placeholder for ae input data [batch_size, sequence_length, DoF]
          dropout:          how much of the input neurons will be activated, value in range [0,1]
        Returns:
          Tensor of output
        """

        network_input = simulate_missing_markets(input_seq_pl, self._mask, self.default_value)

        if FLAGS.reccurent is False:
            last_output = network_input[:, 0, :]

            numb_layers = self.num_hidden_layers + 1

            # Pass through the network
            for i in range(numb_layers):
                # First - Apply Dropout
                last_output = tf.nn.dropout(last_output, dropout)

                w = self._w(i + 1)
                b = self._b(i + 1)

                last_output = self._activate(last_output, w, b)

            output = tf.reshape(last_output, [self.batch_size, 1,
                                              FLAGS.frame_size * FLAGS.amount_of_frames_as_input])

        else:
            output, last_states = tf.nn.dynamic_rnn(
                cell=self._RNN_cell,
                dtype=tf.float32,
                inputs=network_input)

            # Reuse variables
            # so that we can use the same LSTM both for training and testing
            tf.get_variable_scope().reuse_variables()

        return output

    # Make more comfortable interface to the network weights

    def _w(self, n, suffix=""):
        return self[self._weights_str.format(n) + suffix]

    def _b(self, n, suffix=""):
        return self[self._biases_str.format(n) + suffix]

    @property
    def shape(self):
        return self.__shape

    @staticmethod
    def _activate(x, w, b, transpose_w=False):
        y = tf.tanh(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
        return y

    def __getitem__(self, item):
        """Get autoencoder tf variable

        Returns the specified variable created by this object.
        Names are weights#, biases#, biases#_out, weights#_fixed,
        biases#_fixed.

        Args:
         item: string, variables internal name
        Returns:
         Tensorflow variable
        """
        return self.__variables[item]

    def __setitem__(self, key, value):
        """Store a tensorflow variable

        NOTE: Don't call this explicitly. It should
        be used only internally when setting up
        variables.

        Args:
          key: string, name of variable
          value: tensorflow variable
        """
        self.__variables[key] = value

    def _gen_create_variables(self, i, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if 'wd' is specified.
        If 'wd' is None, weight decay is not added for this Variable.

        This function was taken from the web

        Args:
          i: number of hidden layer
          wd: add L2Loss weight decay multiplied by this float.
        Returns:
          Nothing
        """

        # Initialize Train weights
        w_shape = (self._gen_shape[i], self._gen_shape[i + 1])
        a = tf.multiply(2.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        name_w = self._weights_str.format("g" + str(i + 1))
        self[name_w] = tf.get_variable(name_w,
                                       initializer=tf.random_uniform(w_shape, -1 * a, a))

        # Add weight to the loss function for weight decay
        if wd is not None and FLAGS.reccurent == True:

            if i == 1:
                print('We apply weight decay')

            weight_decay = tf.multiply(tf.nn.l2_loss(self[name_w]), wd, name='w_'+str(i)+'_loss')
            tf.add_to_collection('losses', weight_decay)

        # Add the histogram summary
        tf.summary.histogram(name_w, self[name_w])

        # Initialize Train biases
        name_b = self._biases_str.format("g" + str(i + 1))
        b_shape = (self._gen_shape[i + 1],)
        self[name_b] = tf.get_variable(name_b, initializer=tf.zeros(b_shape))

        if i < len(self._gen_shape)-2:
            # Hidden layer fixed weights
            # which are used after pretraining before fine-tuning
            self[name_w + "_fixed"] = tf.get_variable\
                (name=name_w + "_fixed", initializer=tf.random_uniform(w_shape, -1 * a, a),
                 trainable=False)
            # Hidden layer fixed biases
            self[name_b + "_fixed"] = tf.get_variable\
                (name_b + "_fixed", initializer=tf.zeros(b_shape), trainable=False)

            # Pre-training output training biases
            name_b_out = self._biases_str.format("g" + str(i + 1)) + "_out"
            b_shape = (self._gen_shape[i],)
            b_init = tf.zeros(b_shape)
            self[name_b_out] = tf.get_variable(name=name_b_out, initializer=b_init,
                                               trainable=True)

    def _dis_create_variables(self, i, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if 'wd' is specified.
        If 'wd' is None, weight decay is not added for this Variable.

        This function was taken from the web

        Args:
          i: number of hidden layer
          wd: add L2Loss weight decay multiplied by this float.
        Returns:
          Nothing
        """

        # Initialize Train weights
        w_shape = (self._dis_shape[i], self._dis_shape[i + 1])
        a = tf.multiply(2.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        name_w = self._weights_str.format("d" + str(i + 1))
        self[name_w] = tf.get_variable(name_w,
                                       initializer=tf.random_uniform(w_shape, -1 * a, a))

        # Add weight to the loss function for weight decay
        if wd is not None and FLAGS.reccurent == True:

            if i == 1:
                print('We apply weight decay')

            weight_decay = tf.multiply(tf.nn.l2_loss(self[name_w]), wd, name='w_'+str(i)+'_loss')
            tf.add_to_collection('losses', weight_decay)

        # Add the histogram summary
        tf.summary.histogram(name_w, self[name_w])

        # Initialize Train biases
        name_b = self._biases_str.format("d" + str(i + 1))
        b_shape = (self._dis_shape[i + 1],)
        self[name_b] = tf.get_variable(name_b, initializer=tf.zeros(b_shape))

        if i < len(self._dis_shape)-2:
            # Hidden layer fixed weights
            # which are used after pretraining before fine-tuning
            self[name_w + "_fixed"] = tf.get_variable\
                (name=name_w + "_fixed", initializer=tf.random_uniform(w_shape, -1 * a, a),
                 trainable=False)
            # Hidden layer fixed biases
            self[name_b + "_fixed"] = tf.get_variable\
                (name_b + "_fixed", initializer=tf.zeros(b_shape), trainable=False)

            # Pre-training output training biases
            name_b_out = self._biases_str.format("d" + str(i + 1)) + "_out"
            b_shape = (self._gen_shape[i],)
            b_init = tf.zeros(b_shape)
            self[name_b_out] = tf.get_variable(name=name_b_out, initializer=b_init,
                                               trainable=True)

    def run_less_layers(self, input_pl, n, is_target=False):
        """Return result of a net after n layers or n-1 layer, if is_target is true
           This function will be used for the layer-wise pretraining of the AE

        Args:
          input_pl:  tensorflow placeholder of AE inputs
          n:         int specifying pretrain step
          is_target: bool specifying if required tensor
                      should be the target tensor
                     meaning if we should run n layers or n-1 (if is_target)
        Returns:
          Tensor giving pretraining net result or pretraining target
        """
        assert n > 0
        assert n <= self.num_hidden_layers

        last_output = input_pl[:, 0, :]  # reduce dimensionality

        for i in range(n - 1):
            w = self._w(i + 1, "_fixed")
            b = self._b(i + 1, "_fixed")

            last_output = self._activate(last_output, w, b)

        if is_target:
            return last_output

        last_output = self._activate(last_output, self._w(n), self._b(n))

        out = self._activate(last_output, self._w(n), self._b(n, "_out"),
                             transpose_w=True)

        return out
