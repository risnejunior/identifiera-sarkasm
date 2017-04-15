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


# Setting the word embeddings
def set_embedding(sess,init,placeholder,embeddings):
    sess.run(init, feed_dict={placeholder: embeddings})

#Word embedding layer
def word_embedding_layer(word,embedding_tensor):
    embedding_layer = tf.nn.embedding_lookup(embedding_tensor,word)
    return embedding_layer #Not sure if this is done yet

#Defining and building the Neural Network
def recurrent_neural_network(data):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    data = tf.transpose(data)
    data = tf.reshape(data,[-1,chunk_size])
    sequence = tf.split(data, n_chunks, 0)


    gru_cell = rnn.GRUCell(rnn_size)

    outputs, states = rnn.static_rnn(gru_cell, sequence, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1],layer['weights']), layer['biases'])

    return output

# The method for training the neural network
# TODO: Finish this function

def train_neural_network(ps,emb_init,W,emb_placeholder):
    n_samples,words = ps.train.xs.shape
    n_batches = n_samples/batch_size

    data_placeholder = tf.placeholder(dtype=tf.int32,shape=[max_sequence,batch_size])
    labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size,n_classes])
    embeddings = word_embedding_layer(data_placeholder,W)
    prediction = recurrent_neural_network(embeddings)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,
                                                                  labels = labels_placeholder
                                                                  ))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    sess = tf.Session()
    with sess.as_default():
        set_embedding(sess,emb_init,emb_placeholder,emb)
        for epoch in range(epochs):
            print("Hello")


# Here starts the program
with open(samples_path, 'rb') as handle:
    pd = pickle.load( handle )

ps = pd.dataset #Processed Samples

if use_embeddings:
    emb = np.array(pd.embeddings[:pd.vocab_size], dtype=np.float32)
else:
    emb = np.random.randn(pd.vocab_size, pd.emb_size).astype(np.float32)

emb_init, W, emb_placeholder = init_embedding(pd.vocab_size, pd.emb_size)
print(ps.valid.xs.shape)
train_neural_network(ps,emb_init,W,emb_placeholder)

print ("=== Code ran Successfully ===")
