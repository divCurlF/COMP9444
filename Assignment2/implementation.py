import tensorflow as tf
import re
import string

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 150  # Maximum length of a review to consider
EMBEDDING_SIZE = 50	 # Dimensions for each word vector
LSTM_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 2
LEARNING_RATE = 0.001

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we', 'its',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he',
                  'you', 'herself', 'has', 'just', 'where', 'too', 'only',
                  'myself', 'which', 'those', 'i', 'after', 'few', 'whom',
                  't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
                  'doing', 'it', 'how', 'further', 'was', 'here', 'than'})


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here
    that is manipulation at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """

    review = decontracted(review.lower())
    processed_review = [word for word in review.lower().translate(
                                            str.maketrans('', '',
                                            string.punctuation)).split()]
    return processed_review


def define_graph():
    """
    Implement your model here. You will need to define placeholders,
    for the input and labels. Note that the input is not strings of
    words, but the strings after the embedding lookup has been applied
    (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py.
    You should read this file and ensure your code here is compatible.

    Consult the assignment specification for details of which
    parts of the TF API are permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    input_data = tf.placeholder(
            tf.float32,
            [None, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],
            name="input_data"
            )

    labels = tf.placeholder(
            tf.int32,
            [None, NUM_CLASSES],
            name="labels"
            )

    dropout_keep_prob = tf.placeholder_with_default(
            0.9,
            shape=(),
            name="dropout_keep_prob"
            )

    with tf.name_scope("RNN_LAYER"):
            lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
            drop = tf.contrib.rnn.DropoutWrapper(
                                    cell=lstm,
                                    output_keep_prob=dropout_keep_prob,
                    )

    initial_state = drop.zero_state(BATCH_SIZE, tf.float32)

    with tf.name_scope("RNN_DYNAMIC_CELL"):
        outputs, final_state = tf.nn.dynamic_rnn(
                                        drop,
                                        input_data,
                                        initial_state=initial_state,
                                        dtype=tf.float32
                                        )

    with tf.name_scope("fully_connected_1"):
        preds = tf.contrib.layers.fully_connected(
                outputs[:, -1],
                num_outputs=2,
                activation_fn=tf.nn.softmax,
                )
        preds = tf.contrib.layers.dropout(preds, dropout_keep_prob)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                        logits=preds,
                                        labels=labels),
                          name="loss")

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    accuracy = tf.reduce_mean(
                        tf.cast(
                            tf.equal(
                                tf.argmax(preds, 1),
                                tf.argmax(labels, 1)
                                    ),
                            dtype=tf.float32
                               ),
                        name="accuracy")

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
