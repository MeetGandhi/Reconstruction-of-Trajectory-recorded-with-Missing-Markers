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

See this [Jupyter Notebook]() or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

### Temporal Convolutional Neural Network + LSTM

See this [Jupyter Notebook](https://github.com/MeetGandhi/Reconstruction-of-Trajectory-with-Missing-Markers/blob/master/Temporal%20CNN%20%2B%20LSTM/TCN_LSTM.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_CUVh4rRRxq7d7TJKkG7fBFQ7jWwhPXs)

### [Latent ODEs for Irregularly-Sampled Time Series](https://github.com/YuliaRubanova/latent_ode)

See this [Jupyter Notebook](https://github.com/MeetGandhi/Reconstruction-of-Trajectory-with-Missing-Markers/blob/master/Latent%20ODEs%20for%20Irregularly-Sampled%20Time%20Series/Latent_ODEs_for_Irregularly_Sampled_Time_Series.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X42HCkgZh-CsHqtEUVj_EE-CmAdNXCGP)

# Conclusions

# Customization

You can experiment with the hyperparameters of the network by changing the same in the file code/utils/flags.py present in each of the network's folder.

