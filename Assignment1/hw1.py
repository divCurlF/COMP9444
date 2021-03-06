"""
All tensorflow objects, if not otherwise specified, should be explicity
created with tf.float32 datatypes. Not specifying this datatype for variables and
placeholders will cause your code to fail some tests.

You do not need to import any other libraries for this assignment.

Along with the provided functional prototypes, there is another file,
"train.py" which calls the functions listed in this file. It trains the
specified network on the MNIST dataset, and then optimizes the loss using a
standard gradient decent optimizer. You can run this code to check the models
you create in part II.
"""

import tensorflow as tf
import string
import random


def rand_string(chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(5))


""" PART I """


def add_consts():
    """
    EXAMPLE:
    Construct a TensorFlow graph that declares 3 constants, 5.1, 1.0 and 5.9
    and adds these together, returning the resulting tensor.
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.constant(5.9)
    a1 = tf.add(c1, c2)
    af = tf.add(a1, c3)
    return af


def add_consts_with_placeholder():
    """
    Construct a TensorFlow graph that constructs 2 constants, 5.1, 1.0 and one
    TensorFlow placeholder of type tf.float32 that accepts a scalar input,
    and adds these three values together, returning as a tuple, and in the
    following order:
    (the resulting tensor, the constructed placeholder).
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.placeholder(tf.float32)
    a1 = tf.add(c1, c2)
    af = tf.add(a1, c3)
    return af, c3


def my_relu(in_value):
    """
    Implement a ReLU activation function that takes a scalar tf.placeholder as input
    and returns the appropriate output. For more information see the assignment spec.
    """
    return tf.maximum(tf.constant(0, dtype=tf.float32), in_value)


def my_perceptron(x):
    """
    Implement a single perception that takes four inputs and produces one output,
    using the RelU activation function you defined previously.

    Specifically, implement a function that takes a list of 4 floats x, and
    creates a tf.placeholder the same length as x. Then create a trainable TF
    variable that for the weights w. Ensure this variable is
    set to be initialized as all ones.

    Multiply and sum the weights and inputs following the peceptron outlined in the
    lecture slides. Finally, call your relu activation function.
    hint: look at tf.get_variable() and the initalizer argument.
    return the placeholder and output in that order as a tuple

    Note: The code will be tested using the following init scheme
        # graph def (your code called)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # tests here

    """
    i = tf.placeholder(tf.float32, shape=(x))

    weights = tf.get_variable("weights",
                              initializer=tf.ones_initializer(),
                              shape=(x),
                              dtype=tf.float32,
                              trainable=True)

    output = tf.tensordot(i, weights, 1)

    out = my_relu(output)
    return i, out


""" PART II """
fc_count = 0  # count of fully connected layers. Do not remove.


def input_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 784],
                          name="image_input")


def target_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 10],
                          name="image_target_onehot")




def onelayer(X, Y, layersize=10, inputsize=784):
    # Added default parameter to the function to be used for the convolutional
    # function so I can use varying input sizes.
    """
    Create a Tensorflow model for logistic regression (i.e. single layer NN)

    :param X: The input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned  (variables in the
    python sense, not in the Tensorflow sense, although some may be
    Tensorflow variables). They must be returned in the following order.
        w: Connection weights
        b: Biases
        logits: The input to the activation function
        preds: The output of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """

    w = tf.get_variable(rand_string(),
                        initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),

                        shape=[inputsize, layersize])

    b = tf.get_variable(rand_string(),
                        initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                        shape=[layersize])

    logits = tf.matmul(X, w) + b

    preds = tf.nn.softmax(logits)

    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)

    batch_loss = tf.reduce_mean(batch_xentropy)

    return w, b, logits, preds, batch_xentropy, batch_loss


def twolayer(X, Y, hiddensize=30, outputsize=10):
    """
    Create a Tensorflow model for a Neural Network with one hidden layer

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        W1: Connection weights for the first layer
        b1: Biases for the first layer
        W2: Connection weights for the second layer
        b2: Biases for the second layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """

    # Use a small range of uniform random values for better train accuracy.

    w1 = tf.get_variable(rand_string(),
                         initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                         shape=[784, hiddensize])

    b1 = tf.get_variable(rand_string(),
                         initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                         shape=[hiddensize])

    first_layer_out = tf.matmul(X, w1) + b1

    first_layer_act = tf.nn.relu(first_layer_out)

    w2 = tf.get_variable(rand_string(),
                         initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                         shape=[hiddensize, outputsize])

    b2 = tf.get_variable(rand_string(),
                         initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                         shape=[outputsize])

    logits = tf.matmul(first_layer_act, w2) + b2

    preds = tf.nn.softmax(logits)

    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y,
                                                             logits=logits)

    batch_loss = tf.reduce_mean(batch_xentropy)

    return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss


def convnet(X, Y, convlayer_sizes=[10, 10],
            filter_shape=[3, 3], outputsize=10, padding="same", inputsize=784):
    """
    Create a Tensorflow model for a Convolutional Neural Network. The network
    should be of the following structure:
    conv_layer1 -> conv_layer2 -> fully-connected -> output

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        conv1: A convolutional layer of convlayer_sizes[0] filters of shape filter_shape
        conv2: A convolutional layer of convlayer_sizes[1] filters of shape filter_shape
        w: Connection weights for final layer
        b: biases for final layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch

    hints:
    1) consider tf.layer.conv2d
    2) the final layer is very similar to the onelayer network. Only the input
    will be from the conv2 layer. If you reshape the conv2 output using tf.reshape,
    you should be able to call onelayer() to get the final layer of your network
    """

    conv1 = tf.layers.conv2d(inputs=X,
                             filters=convlayer_sizes[0],
                             kernel_size=filter_shape[0],
                             activation=tf.nn.relu,
                             padding=padding
                             )

    conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=convlayer_sizes[1],
                             kernel_size=filter_shape[1],
                             activation=tf.nn.relu,
                             padding=padding)

    flattened_input = tf.contrib.layers.flatten(conv2)

    # Due to zero padding, the convolutional layer will be the same size as the image
    # So to get the size of the one layer vector we multiply by the number of filters
    # in the second convolutional layer.

    sz = inputsize*convlayer_sizes[1]

    # Pass the flattened input into one layer function with default parameter sz.
    w, b, logits, preds, batch_xentropy, batch_loss = onelayer(flattened_input, Y, outputsize, inputsize=sz)

    return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss


def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):
    """
    Run one step of training.

    :param sess: the current session
    :param batch: holds the inputs and target outputs for the current minibatch
    batch[0] - array of shape [minibatch_size, 784] with each row holding the
    input images
    batch[1] - array of shape [minibatch_size, 10] with each row holding the
    one-hot encoded targets
    :param X: the input placeholder
    :param Y: the output target placeholder
    :param train_op: the tensorflow operation that will run one step of training
    :param loss_op: the tensorflow operation that will return the loss of your
    model on the batch input/output

    :return: a 3-tuple: train_op_result, loss, summary
    which are the results of running the train_op, loss_op and summaries_op
    respectively.
    """
    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    return train_result, loss, summary
