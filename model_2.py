from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from skimage import transform

import tensorflow as tf

import input

# load the train and test data
train_data, train_label = input.load_data('image/train.txt')
test_data, test_label = input.load_data('image/val.txt')

# resize rhe training and testing pictures
train_data_m = []
for image in train_data:
    img64 = transform.resize(image, (64, 64, 3))
    train_data_m.append(img64)

test_data_m = []
for image in test_data:
    img64 = transform.resize(image, (64, 64, 3))
    test_data_m.append(img64)


train_labels_ = []
for i in train_label:
    if i == 0:
        train_labels_.append(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    elif i == 1:
        train_labels_.append(np.array([0.0, 1.0, 0.0, 0.0, 0.0]))
    elif i == 2:
        train_labels_.append(np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
    elif i == 3:
        train_labels_.append(np.array([0.0, 0.0, 0.0, 1.0, 0.0]))
    elif i == 4:
        train_labels_.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))

test_labels_ = []
for i in test_label:
    if i == 0:
        test_labels_.append(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    elif i == 1:
        test_labels_.append(np.array([0.0, 1.0, 0.0, 0.0, 0.0]))
    elif i == 2:
        test_labels_.append(np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
    elif i == 3:
        test_labels_.append(np.array([0.0, 0.0, 0.0, 1.0, 0.0]))
    elif i == 4:
        test_labels_.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))

FLAGS = None


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is the
        number of pixels in a standard MNIST image.
    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the digit into one of 10 classes (the
        digits 0-9). keep_prob is a scalar placeholder for the probability of
        dropout.
    """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 3])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 5])
        b_fc2 = bias_variable([5])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


# Create a graph to hold the model.
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
    # Placeholders for inputs and labels.
    x = tf.placeholder(tf.float32, [None, 28, 28, 3])
    y_ = tf.placeholder(tf.float32, [None, 5])
    y = tf.placeholder(tf.float32, [None, 5])

    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    x_flat = tf.contrib.layers.flatten(x)

    y_conv, keep_prob = deepnn(x_flat)
    # ============-------------===============-------------=============

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.name_scope('test_accuracy'):
        prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        prediction = tf.cast(prediction, tf.float32)
    test_accuracy = tf.reduce_mean(prediction)

    init = tf.initialize_all_variables()


with tf.Session(graph=graph) as sess:
    sess.run(init)
    for i in range(20):
        train, loss_value, accu = sess.run([train_step,cross_entropy,accuracy], feed_dict={x: train_data_m, y_: train_label, keep_prob: 0.5})
#         train, loss_value, accu = session.run([train_step,cross_entropy,accuracy], feed_dict={x: test_images64, y_: test_labels_, keep_prob: 0.5})
        if i % 10 == 0:
            print('step ' + str(i) + ' Loss: ', loss_value)
            print(accu)
    predicted = sess.run(test_accuracy, feed_dict={x: test_data_m, y: test_label, keep_prob: 0.5})
