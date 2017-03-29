'''
===RNN Network in TensorFlow

* This is a Recurrent Neural Network for sarcasm detection
* Author: DATX02-25
'''

# Importing tensorflow
import tensorflow as tf
# Importing rnn framework
from tensorflow.contrib import rnn
# Importing Settings
import settings

# Loading Settings
vocabulary_size = settings.vocabulary_size #should match actual dictionary
embedding_size = settings.embedding_size
epochs = settings.epochs
batch_size = settings.batch_size
partition_training = settings.partition_training
partition_validation = settings.partition_validation
partition_test = settings.partition_test
set_balance = settings.set_balance
max_sequence = settings.max_sequence
ascii_console = settings.ascii_console

# debug commands, will mess up the training:
random_labels = settings.random_labels
add_snitch = settings.add_snitch
random_embeddings = not settings.use_embeddings

samples_path = settings.samples_path
vocabulary_path = settings.vocabulary_path
rev_vocabulary_path = settings.rev_vocabulary_path
embeddings_path = settings.embeddings_path

# Network parameters
n_classes = 2
chunk_size = embedding_size
n_chunks = max_sequence
rnn_size = 128

# Defining the RNN network

def recurrent_neural_network(data):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    data = tf.transpose(data, [1,0,2])
    data = tf.reshape(data ,[-1,chunk_size])
    data = tf.split(data, n_chunks, 0)

    gru_cell = rnn.GRUCell(rnn_size)

    output, states = rnn.static_rnn(gru_cell, data, dtype=tf.float32)

    return output

# The method for training the neural network

#TODO: Finish this function
def train_neural_network(data):
    prediction = recurrent_neural_network(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels = y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0
        print("Hello")

    print("TODO: Finish this method")
