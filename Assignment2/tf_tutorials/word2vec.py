import tensorflow as tf
import numpy as np
import math
import collections
import random

"""
GENERATING DATA FOR SKIP GRAM MODEL.
"""

data = list()

# Make it any number between 16 and 512

batch_size = 128


def generate_batch(batch_size, num_skips, skip_window):
    # batch_size: size of the embedding vector.
    # num_skips: ?
    # skip_window: [ skip_window | target | skip_window ] - the amount
    # of words around the target we will examine.

    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2*skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2*skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


"""
BUILDING AND TRAINING A SKIP GRAM MODEL.
"""

# The number of words in our dictionary.
vocabulary_size = 5000

# The size of each vector
embedding_size = 128

# Initialising an embedding matrix as random values.

embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
                )

# Estimation loss defined by logistic regression.
# Define weights and biases for each word in the vocabulary.
# These are output weights, rather than, input embeddings.

nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0/math.sqrt(embedding_size))
                )

nce_biases = tf.Variable(tf.zeros[vocabulary_size])

# Placeholders for inputs.
# Assume we have turned text into a format where each word is represented as an integer.
# THIS IS A todo.


# One row vector of our inputs
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])

# Vector tuple with labels.
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])


# Look up the vector for each of the source words in the batch.

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# NCE Loss: train through logitistic regression to discriminate between samples
# from the data distribution and samples from the noise distribution.

# Compute NCE loss, using a sample of the negative labels each time.

nce_loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size)
        )

softmax_loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=nce_weights,
                                   biases=nce_biases,
                                   labels=train_labels,
                                   inputs=embed,
                                   num_sampled=num_sampled,
                                   num_classes=vocabulary_size)
        )
# Use stochastic gradient descent to train the model.

optimiser = tf.train.GradientDescentOptimizer(learning_rate=1.0).minmize(nce_loss)
