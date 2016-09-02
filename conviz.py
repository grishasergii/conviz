from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


PLOT_DIR = './out/plots'


def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')


def plot_conv_output(conv_img, name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


def conv2d(x_, filter_size, filter_num, stride=1):
    """
    Wrapper of a convolutional layer
    :param x_: tensor, input to convolutional layer
    :param filter_size: int, size of a convolutional kernel
    :param filter_num: int, number of convolutional kernels
    :param stride: int, optional, stride
    :return: tensor
    """
    # get number of channels in input
    channels = x_.get_shape()[3].value

    # create weights tensor
    weights = tf.Variable(tf.random_normal([filter_size, filter_size, channels, filter_num]))

    # add weights tensor to collection
    tf.add_to_collection('conv_weights', weights)

    # create bias tensor
    bias = tf.Variable(tf.random_normal([filter_num]))

    # apply weights and biases
    preactivations = tf.nn.conv2d(x_, weights, strides=[1, stride, stride, 1], padding='SAME')
    preactivations = tf.nn.bias_add(preactivations, bias)

    # apply activation function, this is layer output
    activations = tf.nn.relu(preactivations)

    # add output to collection
    tf.add_to_collection('conv_output', activations)

    return activations


def fc(x_, nodes, keep_prob_=1, act=tf.nn.relu):
    """
    Wrapper for fully-connected layer
    :param x_: tensor, input to fully-connected alyer
    :param nodes: int, number of nodes in layer
    :param keep_prob_: float, optional, keep probability for dropout operation
    :param act: tf.nn method, optional, activation function
    :return: tensor
    """
    shape = x_.get_shape()

    # if rank of input tensor is greater than 2
    # we need to reshape it
    if shape.ndims > 2:
        n = 1
        for s in shape[1:]:
            n *= s.value
        x_ = tf.reshape(x_, tf.pack([-1, n]))
        x_.set_shape([None, n])

    # get number of column in input tensor
    n = x_.get_shape()[1].value

    # create weights
    weights = tf.Variable(tf.random_normal([n, nodes]))

    # create biases
    bias = tf.Variable(tf.random_normal([nodes]))

    # apply weights and bias
    preactivate = tf.add(tf.matmul(x_, weights), bias)
    out = preactivate

    # apply activation function if not None
    if act is not None:
        out = act(preactivate)

    # apply dropout
    out = tf.nn.dropout(out, keep_prob_)

    return out


def maxpool(x_, size, stride):
    """
    Wrapper for max-pooling layer
    :param x_: tensor, input to max-pooling layer
    :param size: int
    :param stride: int
    :return: tensor
    """
    return tf.nn.max_pool(x_,
                          ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')

# Reshape inputs
x_reshaped = tf.reshape(x, shape=[-1, 28, 28, 1])

# First convolutional layer
predictions = conv2d(x_reshaped, filter_size=5, filter_num=32)

# First max-pooling layer
predictions = maxpool(predictions, 2, 2)

# Second convolutional layer
predictions = conv2d(predictions, filter_size=5, filter_num=64)

# Second max-pooling layer
predictions = maxpool(predictions, 2, 2)

# First fully-connected layer
predictions = fc(predictions, 1024, keep_prob)

# Output layer, no activation function
# This layer returns logits
predictions = fc(predictions, n_classes, keep_prob, act=None)

# Define loss operation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, y))

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Define accuracy operation
correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("\rIter " + str(step*batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc), end='')
        step += 1
    print("\rOptimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                        y: mnist.test.labels[:256],
                                        keep_prob: 1.}))

    # get weights of all convolutional layers
    # no need for feed dictionary here
    conv_weights = sess.run([tf.get_collection('conv_weights')])
    for i, c in enumerate(conv_weights[0]):
        plot_conv_weights(c, 'conv{}'.format(i))

    # get output of all convolutional layers
    # here we need to provide an input image
    conv_out = sess.run([tf.get_collection('conv_output')], feed_dict={x: mnist.test.images[:1]})
    for i, c in enumerate(conv_out[0]):
        plot_conv_output(c, 'conv{}'.format(i))


