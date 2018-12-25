import tensorflow as tf
import re
import string

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 180  # Maximum length of a review to consider
EMBEDDING_SIZE = 50	 # Dimensions for each word vector
LAYER1_SIZE = 128
LAYER2_SIZE = 64
LEARNING_RATE = 0.001


stop_words = {"i", "me", "my", "myself", "we", "our", "ours",
              "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself",
              "she", "her", "hers", "herself", "it", "its",
              "itself", "they", "them", "their", "theirs",
              "themselves", "what", "which", "who", "whom", "this", "that",
              "these", "those", "am", "is","are", "was", "were", "be",
              "been", "being", "have", "has", "had", "having", "do", "does",
              "did", "doing", "a", "an", "the", "and", "but",
              "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against",
              "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up",
              "down", "in", "out", "on", "off", "over", "under",
              "again", "further", "then", "once", "here", "there",
              "when", "where", "why", "how", "all", "any", "both",
              "each", "few", "more", "most", "other", "some",
              "such", "no", "nor", "came", "only", "own", "same",
              "so", "than", "too", "very", "s", "t", "can", "come",
              "will", "just", "don", "should", "now", "hey", "com", "ann",
              "los", "iii", "las",  "vii", "i'm", "anu",
              "rtd", "tng", "gft", "rgv", "mph", "nva", "gor", "zep",
              "mas", "viii", "hui", "lbs", "nah", "ahhh", "ishe",
              "whos", "ang", "fro", "sec", "rep", "thy", "cmi",
              "aots", "ist", "ltl", "lol", "shi", "ons", "yat",
              "rev", "aha", "isa", "ike", "coz", "tsa", "nri",
              "une", "kat", "cyf", "odc", "msr", "orr",  "bbfc",
              "vey", "istory", "aftab", "adr", "daa", "ther", "cid",
              "coo", "hep", "kam", "sai", "upn", "doi", "sao", "est",
              "irk", "nic", "suu", "kyi", "vid", "dio", "hhh", "jbl",
              "fop", "ola", "hao", "lds", "drc", "adjl", "pca", "stv",
              "pox", "fdr", "zey", "ll", "dah", "iwo", "ros", "dab",
              "und", "bci", "rao", "dat", "ped", "ahh",
              "doa", "uhf", "siu", "ans", "wen", "aww", "iti",
              "sik", "glo", "tis", "anl", "bwp", "eke", "ocp",
              "ric", "viz", "eds", "laz", "qui", "hoy", "mui", "fxs",
              "rin", "ryo", "teh", "usc",  "aye",
              "imam", "sas", "wah", "mmmm", "jun", "jae", "fav",
              "hwa", "ami", "isi", "aip", "rey", "hel", "flo",
              "jyo", "duc", "tah", "lop", "ook", "jou", "dbz", "imf",
              "mmt", "lyn", "oaf", "syn", "adi", "zis", "jog", "diy",
              "gai", "yah", "mit", "tel", "hll", "nyu", "ison",
              "vuh", "uno", "sxsw", "hex",
              "sig", "voz", "htm", "ely", "viv", "hof", "rds", 
              "sur", "asch", "yyz", "abu", "nel", "vcds", "rei", "eta",
              "ulf", "oyl", "sup", "ood", "oct", "bai", "mtm", "lom",
              "hob", "cit", "int", "wor", "awwww",
              "wtc", "caa", "adv", "iqs", "oli", "cho", "htv", "ilm",
              "hbk", "mcrd", "dci", "feb", "ame", "tbo", "eff", "ddl",
              "liek", "buu", "ddt", "mib", "fks", "vvv",
              "tox", "aks", "inu", "moh", "tlc", "plo", "dei", "osa",
              "nix", "tss", "pac", "prn", "sms", "hoi", "edo"
              "uhr", "yee", "itc", "cyd", "fei", "sate", "irma",
              "fob", "gta", "twtl", "ona", "foy", "och", "wga", "lmn",
              "whet", "dmn", "hon", "mme", "lei", "cot", "obi", "aus"
              "nuf", "lis", "zeki", "laa", "hea", "lok",
              "bel", "lax", "sar", "sne", "amrs", "lim",
              "mov", "gci", "jjl", "ism", "emi", "aya", "trw", "hwy",
              "goa", "epp", "dax", "tne", "jut", "gol", "slc",
              "tau", "rah", "cpl", "ous", "imx", "mts", "thr",
              "npr", "nab", "ers", "wmd", "nyt", "fcc", "cro",
              "tot", "ich", "ova", "apr", "jez", "ari", "cud", "wdr",
              "kes", "bbs", "rog", "xyz", "atm", "caw", "dts", "gli",
              "emt", "mus", "nos", "elo", "loo", "ifs", "voo", "rad",
              "amp", "rce", "toi", "mot", "wec", "reo", "moa", 
              "taw", "mr", "us", "go", "n", "ts", "im", "platforms", "sord",
              "tv", "alida", "galiena", "ss", "er", "oh", "co", "pochath"}


def decontracted(phrase):
    # specific

    # remove HTML break tag.
    phrase = re.sub(r"<br />", " ", phrase)

    # replace ellipses with a space.
    phrase = re.sub(r"[...]", " ", phrase)

    # Fix contractions.
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"[0-9]", "", phrase)

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

    # translator to remove punctuations
    translator = str.maketrans('', '', string.punctuation)

    # remove contractions and use regex to clean some words.
    review = decontracted(review.lower())

    # replace - and / with spaces to separate words, remove contraction
    # and spit into a list.
    # don't include any stop words and any words under a length of two.

    processed_review = [word for word in
            review.lower().replace('-', ' ').replace('/', ' ').translate(translator).split()
            if len(word) > 2 and word not in stop_words
                       ]
    return processed_review


# Builds a stack of lstm layers with dropout probability.
# lstm_sizes is a list of integers representing the size of each layer.
def build_lstm_layers(lstm_sizes, embed, keep_prob, batch_size):
    """
    Create the LSTM layers
    """
    lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in lstm_sizes]
    # Add dropout to the cell
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) for lstm in lstms]
    # Stack up multiple LSTM layers
    cell = tf.contrib.rnn.MultiRNNCell(drops)
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

    return lstm_outputs, final_state


def define_graph(param_list=None):
    input_data = tf.placeholder(
            tf.float32,
            [None, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],
            name="input_data"
            )

    labels = tf.placeholder(
            tf.int32,
            [None, 2],
            name="labels"
            )

    # leave default dropout as 1.0 for eval mode.
    dropout_keep_prob = tf.placeholder_with_default(
            1.0,
            shape=(),
            name="dropout_keep_prob"
            )

    # Build the lstm model.
    outputs, final_state = build_lstm_layers(
                                       [LAYER1_SIZE, LAYER2_SIZE],
                                       input_data,
                                       dropout_keep_prob,
                                       BATCH_SIZE)

    # fully connected layer with softmax activation.
    with tf.name_scope("fully_connected_1"):
        preds = tf.contrib.layers.fully_connected(
                outputs[:, -1],
                num_outputs=2,
                activation_fn=tf.nn.softmax,
                )
    # softmax loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                                        logits=preds,
                                        labels=labels),
                          name="loss")
    # ADAM optimizer.
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # accuracy: Find the greatest probability for all preds in the output vector,
    # Compare them  with the labels and cast it as a float, reducing the mean over
    # all the samples.

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
