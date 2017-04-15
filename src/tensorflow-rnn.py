'''
===RNN Network in TensorFlow

* This is a Recurrent Neural Network for sarcasm detection
* Author: DATX02-25
'''
# Importing dependencies for file handling
import collections
import math
import random
import os
import json
import pickle
import csv
import sys

# Importing tensorflow
import tensorflow as tf
# Importing rnn framework
from tensorflow.contrib import rnn
# Importing NumPy
import numpy as np

# Importing Settings
from settings import *


n_classes = 2
chunk_size = embedding_size
n_chunks = max_sequence
rnn_size = 128

## Create the embedding variable
def init_embedding(vocabulary_size,embedding_size):
    W = tf.Variable(tf.constant(0.0, shape = [vocabulary_size, embedding_size]),
                    trainable = False,
                    name = "W")
    embedding_placeholder = tf.placeholder(dtype = tf.float32,
                                           shape = [vocabulary_size, embedding_size]
                                           )
    embedding_init = W.assign(embedding_placeholder)
    return embedding_init, W, embedding_placeholder


#Word embedding layer
def word_embedding_layer(word,embedding_tensor):
    embedding_layer = tf.nn.embedding_lookup(embedding_tensor,word)
    return embedding_layer #Not sure if this is done yet

#Defining and building the Neural Network
def recurrent_neural_network(data):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    data = tf.transpose(data, [1,0,2])
    data = tf.reshape(data ,[-1,chunk_size])
    data = tf.split(data, n_chunks, 0)

    gru_cell = rnn.GRUCell(rnn_size)

    outputs, states = rnn.static_rnn(gru_cell, data, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1],layer['weights']), layer['biases'])

    return output

# The method for training the neural network
# TODO: Finish this function

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

# Here starts the program
with open(samples_path, 'rb') as handle:
    pd = pickle.load( handle )

ps = pd.dataset #Processed Samples
print("Looking at the dataset:")
print(ps.train.xs)

if use_embeddings:
    emb = np.array(pd.embeddings[:pd.vocab_size], dtype=np.float32)
else:
    emb = np.random.randn(pd.vocab_size, pd.emb_size).astype(np.float32)

emb_init, W, emb_placeholder = init_embedding(pd.vocab_size, pd.emb_size)

print(emb_init)
print(W)
print(emb_placeholder)

print ("=== Code ran Successfully ===")
