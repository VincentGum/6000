from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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


# Create a graph to hold the model.
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
    # Placeholders for inputs and labels.
    images_ph = tf.placeholder(tf.float32, [None, 64, 64, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)

    # Fully connected layer.
    # Generates logits of size [None, 62]
    logits = tf.contrib.layers.fully_connected(images_flat, 5, tf.nn.relu)

    # Convert logits to label indexes (int).
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, 1)

    # Define the loss function.
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # And, finally, an initialization op to execute before training.
    # TODO: rename to tf.global_variables_initializer() on TF 0.12.
    init = tf.initialize_all_variables()

# Create a session to run the graph we created.
session = tf.Session(graph=graph)

# First step is always to initialize all variables.
# We don't care about the return value, though. It's None.
_ = session.run([init])

for i in range(201):
    _, loss_value = session.run([train, loss], feed_dict={images_ph: train_data_m, labels_ph: train_label})
    if i % 10 == 0:
        print("Loss: ", loss_value)



