# Reconstruction of Trajectory with Missing Markers

The project deals with implementing various Neural Network approaches to perform the reconstruction of missing markers in the human motion capture data and as a result finding the one which gives the most accurate recreation.

# Prerequisites

* Python 3
* Tensorflow >= 1.0
* keras-tcn
* numpy
* matplotlib
* torchdiffeq
* torch >= 1.0

# Human Motion Capture Data

The pre-processed human motion capture data as well as the base code were extracted from [A Neural Network Approach to Missing Marker Reconstruction](https://github.com/Svito-zar/NN-for-Missing-Marker-Reconstruction). In order to use your own Motion Capture data, follow the steps in the Data Preparation section of *A Neural Network Approach to Missing Marker Reconstruction* repository.

Once the data is available make sure that you put it in data folder ad change the path in code/utils/flags.py appropriately for every implementation.

# Implementations

### Temporal Convolutional Neural Network

See this [Jupyter Notebook](https://github.com/MeetGandhi/Reconstruction-of-Trajectory-with-Missing-Markers/blob/master/Temporal%20Convolutional%20Neural%20Network/TCN.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TvyKyYiPAc3wMLbw-e5CIYFcaj8KyWCg)

### Temporal Convolutional Neural Network + Wasserstein GAN

See this [Jupyter Notebook](https://github.com/MeetGandhi/Reconstruction-of-Trajectory-with-Missing-Markers/blob/master/Temporal%20CNN%20%2B%20Wasserstein%20GAN/TCN_WGAN.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jIAf_1XJLDRil7z6_NZomclrGp_4yGXw)

### Temporal Convolutional Neural Network + Wasserstein GAN with Gradient Penalty

See this [Jupyter Notebook](https://github.com/MeetGandhi/Reconstruction-of-Trajectory-with-Missing-Markers/blob/master/Temporal%20CNN%20%2B%20Wasserstein%20GAN%20with%20Gradient%20Penalty/TCN_WGAN_GP.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yUxODlBlonchPvQ44J9kfReDvwlJx8_s)

### [Long Short Term Network](https://github.com/Svito-zar/NN-for-Missing-Marker-Reconstruction)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Sx0xBeTOc-9zu8LKadMPy4zmLi7H-Jhz)

### Temporal Convolutional Neural Network + LSTM

See this [Jupyter Notebook](https://github.com/MeetGandhi/Reconstruction-of-Trajectory-with-Missing-Markers/blob/master/Temporal%20CNN%20%2B%20LSTM/TCN_LSTM.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_CUVh4rRRxq7d7TJKkG7fBFQ7jWwhPXs)

### Wasserstein GAN + LSTM

See this [Jupyter Notebook](https://github.com/MeetGandhi/Reconstruction-of-Trajectory-with-Missing-Markers/blob/master/Wasserstein%20GAN/WGAN.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q5iDNJBbk0wWcLbWPPplIIU6b6cqA1dV)

Note that WGAN + LSTM implementation was ran for 50 epochs instead of the default 100 epochs(for the rest of the networks) as WGAN was not showing any further improvement in test RMSE.

### [Latent ODEs for Irregularly-Sampled Time Series](https://github.com/YuliaRubanova/latent_ode)

See this [Jupyter Notebook](https://github.com/MeetGandhi/Reconstruction-of-Trajectory-with-Missing-Markers/blob/master/Latent%20ODEs%20for%20Irregularly-Sampled%20Time%20Series/Latent_ODEs_for_Irregularly_Sampled_Time_Series.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X42HCkgZh-CsHqtEUVj_EE-CmAdNXCGP)

# Conclusions

### Temporal Convolutional Neural Network

The training session for TCN started with a very low validation RMSE error (~40) compared to the other networks, however in the subsequent epochs the RMSE error did not improve but stayed the same. At the end of the training session, the test RMSE error was ~32 which was just a slight decrease from the starting validation error. According to our observations, the limitations in the receptive field of the Temporal Convolutional Neural Network is the reason for the above training performance.

### Temporal Convolutional Neural Network + Wasserstein GAN

To leverage the WGAN features with TCN which had a hard time training, an ensemble network was trained showing a starting validation RMSE error of ~ 117 and eventually a test RMSE error of ~24 after the training ended. As a result TCN seems to have increased WGAN's test RMSE error. Similar results were observed for TCN + WGAN with Gradient Penalty.

### Long Short Term Network

LSTM was trained with the same hyperparameters as mentioned in [A Neural Network Approach to Missing Marker Reconstruction](https://github.com/Svito-zar/NN-for-Missing-Marker-Reconstruction). The training session for LSTM started with exceptionally low validation RMSE error of ~14 (test RMSE error for other networks) and ended with test RMSE error of ~2. As expected LSTM is unmatched in time-series sequence to sequence generation task. However, in order to lower down the test RMSE error further, various other ensemble neural networks were devised with a combination of LSTM and another neural network.

### Long Short Term Network + Temporal Convolutional Neural Network

Similar to LSTM's cases, the training session here started with a validation RMSE error of ~14, however due to the non-training influence of TCN, the training ended with approximately equal test RMSE error of ~13.

### Long Short Term Network + Wasserstein GAN

WGAN+LSTM's training started with a very high validation RMSE error (~117) as expected, nevertheless in the next five training epochs WGAN+LSTM trained very quickly, thus bring down validation RMSE error to ~19. As WGAN+LSTM was not showing much improvements after epoch 5 all the way up to epoch 50, the training session was stopped at epoch 50. This suggests that the current WGAN+LSTM configuration had trained completely and could not go beyond ~15 test RMSE error.


# Customization

You can experiment with the hyperparameters of the network by changing the same in the file code/utils/flags.py present in each of the network's folder.
