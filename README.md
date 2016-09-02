# Conviz

Conviz is an example of visualizing weights and output of convolutional layers. In this project I show how to add
convolution kernels and output of convolutional layers to a collection that fetched and visualized after training is completed. This approach can be easily generalized 
to arbitrary number of layers.

It is based on the following tutorials:
* [3_NeuralNetworks](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py)
* [02_Convolutional_Neural_Network](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb)

# Data set
MNIST data set that is shiped with Tensorflow. 

# Output
Plotted convolution kernels and output of convolutional layers are saved to disk.

### First convolutional layer
![con_weights_0](img/conv_weights_0.png)
![con_output_0](img/conv_output_0.png)

### Second convolutional layer
![con_weights_1](img/conv_weights_1.png)
![con_output_1](img/conv_output_1.png)

# Dependencies
* Numpy
* Matplotlib
* Tensorflow