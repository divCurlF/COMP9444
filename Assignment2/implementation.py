import tensorflow as tf
import re

# HYPERPARAMETERS

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
LSTM_SIZE = 128
NUM_LAYERS = 1
FC_UNITS = 256
LEARNING_RATE = 0.001

stop_words = {'ourselves', 'hers', 'between', 'yourself', 'again',
              'there', 'about', 'once', 'during', 'out', 'very', 'having',
              'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
              'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
              'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
              'each', 'the', 'themselves', 'below', 'are', 'we',
              'these', 'your', 'his', 'through', 'don', 'me', 'were',
              'her', 'more', 'himself', 'this', 'down', 'should', 'our',
              'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
              'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
              'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
              'yourselves', 'then', 'that', 'because', 'what', 'over',
              'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
              'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
              'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
              'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
              'how', 'further', 'was', 'here', 'than', 'ain\'t', 'it\'s',
              'i\'m', 'can\'t', 'won\'t'}

replacement_patterns = [
    (r'(\w+)\'ll', '\g<1>'),
    (r'(\w+)n\'t', '\g<1>'),
    (r'(\w+)\'ve', '\g<1>'),
    (r'(\w+)\'s', '\g<1>'),
    (r'(\w+)\'re', '\g<1>'),
    (r'(\w+)\'d', '\g<1>'),
]

post_patterns = [
    (r"<br />", " "),
    (r"[^a-z]", " ")
]


def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is
    manipulation at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    review = review.lower().split()

    review = [word for word in review if word not in stop_words]

    review = ' '.join(review)

    for pattern, replacement in replacement_patterns:
        review = re.sub(pattern, replacement, review)

    review = ' '.join(review.split())

    for pattern, replacement in post_patterns:
        review = re.sub(pattern, replacement, review)

    review = ' '.join(review.split())

    return review


def define_graph():

    with tf.name_scope("input_data"):
        input_data = tf.placeholder(
                            tf.float32,
                            [None, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],
                            name="inputs")

    with tf.name_scope("labels"):
        labels = tf.placeholder(tf.int32, name="labels")

    with tf.name_scope("dropout_keep_prob"):
        dropout_keep_prob = tf.placeholder_with_default(
                                0.6,
                                shape=(),
                                name="dropout_keep_prob")
    # BUILD AN LSTM WITH DROPOUT
    with tf.name_scope("RNN_layers"):
        lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
        drop = tf.contrib.rnn.DropoutWrapper(
                                    lstm)
                                    # output_keep_prob=dropout_keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * NUM_LAYERS)

    # INITIALISE THE STATE OF THE LSTM
    with tf.name_scope("RNN_init_state"):
        initial_state = cell.zero_state(BATCH_SIZE, tf.float32)

    # FEEDFORWARD LSTM
    with tf.name_scope("RNN_forward"):
        outputs, final_state = tf.nn.dynamic_rnn(
                                    cell,
                                    input_data,
                                    initial_state=initial_state
                                )

    with tf.name_scope("fully_connected"):
        weights = tf.contrib.layers.xavier_initializer(
                uniform=False,
                dtype=tf.float32)
        bias = tf.zeros_initializer()
        preds = tf.layers.dense(
                        outputs[:, -1],
                        units=2,
                        activation=tf.nn.softmax,
                        kernel_initializer=weights,
                        bias_initializer=bias)

    with tf.name_scope("loss"):

        # EXPERIMENT WITH THE LOSS FUNCTION
        loss = tf.losses.mean_squared_error(labels, preds)

    with tf.name_scope("train"):

        # EXPERIMENT THE OPTIMIZER FUNCTION
        optimizer = tf.train.AdamOptimizer(
                                        learning_rate=LEARNING_RATE
                ).minimize(loss)

    with tf.name_scope("accuracy"):
        accuracy = tf.contrib.metrics.accuracy(
                                    tf.cast(tf.round(preds),
                                    dtype=tf.int32),
                                    labels)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
