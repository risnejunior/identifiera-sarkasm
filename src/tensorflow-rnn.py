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
from common_funs import *

sys.path.append("../identifiera-sarkasm/")
sys.path.append("../identifiera-sarkasm/tfnetworks/")

import tfnetworks

import time

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
rnn_size = 64
roundform = "{0:.5f}"

data_placeholder = tf.placeholder(dtype=tf.int32,shape=[None,max_sequence])
labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,n_classes])
keep_prob_placeholder = tf.placeholder('float')

train_call = 1
val_call = 2

## Create the embedding variable
def init_embedding(vocabulary_size,embedding_size):
    with tf.device("/cpu:0"):
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

def train_neural_network(ps,emb_init,W,emb_placeholder):

    embeddings = word_embedding_layer(data_placeholder,W)
    network = tfnetworks.fetch_network("little pony",n_classes,params = {'rnn_size': rnn_size})
    prediction = network.feed_network(embeddings,keep_prob_placeholder,chunk_size,n_chunks)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,
                                                                  labels = labels_placeholder
                                                                  ))

    l2_loss = network.calc_l2_loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost + 0.01 * l2_loss)
    sess = tf.Session()
    xs_split,ys_split = split_chunks(ps.train.xs, np.array(ps.train.ys),batch_size)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        set_embedding(sess,emb_init,emb_placeholder,emb)
        loops = len(xs_split)
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_i in range(loops):
                batch_x = xs_split[batch_i]
                batch_y = ys_split[batch_i]
                _, c,pred = sess.run([optimizer,cost,prediction], feed_dict = {data_placeholder: batch_x, labels_placeholder: batch_y, keep_prob_placeholder: 0.5})
                comp = np.equal(np.argmax(pred,1),np.argmax(batch_y,1))
                current_accuracy = np.mean(comp.astype(np.float32))
                epoch_loss += c
                print( batch_i + 1, "batches completed out of:", loops ,"| current loss:",roundform.format(epoch_loss),"| Accuracy :",roundform.format(current_accuracy), "",end=" \r",)

            saver = tf.train.Saver()
            print("")
            save_path = saver.save(sess, "../models/tfcheckpoint.ckpt")
            accuracy = test_network_run(ps.valid.xs,np.array(ps.valid.ys),prediction)
            print("Checkpoint file saved in %s" % save_path )
            print('Epoch', epoch+1, 'completed out of', epochs, 'loss:', roundform.format(epoch_loss), '| Accuracy:', roundform.format(accuracy))

        saver = tf.train.Saver()
        date = time.strftime("%m%d%y-%H%M%S")
        saver_path = saver.save(sess, "../models/tfrnn_model-%s.ckpt" % date)
        print("Model saved at %s" % saver_path )
        accuracy = test_network_run(ps.test.xs,np.array(ps.test.ys),prediction)
        print("Test accuracy of this network is: ", accuracy)
    sess.close()

def split_chunks(xs,ys,size):
    xh,xw = xs.shape
    yh,yw = ys.shape
    if xh != yh :
        raise ValueError("Sizes don't match")
    #print(xh)
    n_batches = xh//size
    #print(n_batches)
    overflow = xh - (n_batches * size)
    #print(overflow)
    xs_overflow = xs[-overflow:]
    #print(xs_overflow.shape)
    xs_notoverflow = xs[:xh - overflow]
    ys_overflow = ys[-overflow:]
    ys_notoverflow = ys[:yh - overflow]
    xs_split = np.split(xs_notoverflow,n_batches)
    ys_split = np.split(ys_notoverflow,n_batches)
    xs_split.append(np.array(xs_overflow))
    ys_split.append(np.array(ys_overflow))
    return xs_split,ys_split


# Method for validating network in training
def test_network_run(data,labels,network_op):
    prediction = tf.nn.log_softmax(network_op)
    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(labels_placeholder,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    return accuracy.eval(feed_dict={data_placeholder: data, labels_placeholder: labels, keep_prob_placeholder: 1.0})

def test_network(ps,network_name,path=None):
    if path == None :
        print("No model selected")
    else:
        run_test(ps,path,network_name)

def run_test(ps,path,network_name):
    network = tfnetworks.fetch_network(network_name,n_classes,params={'rnn_size': rnn_size})
    test_data = ps.test.xs
    test_labels = np.array(ps.test.ys)
    emb_init, W, emb_placeholder = init_embedding(pd.vocab_size, pd.emb_size)
    embeddings = word_embedding_layer(data_placeholder,W)
    output = network.feed_network(embeddings,keep_prob_placeholder,chunk_size,n_chunks)
    with tf.Session().as_default() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, path)
        accuracy = test_network_run(ps.train.xs,np.array(ps.train.ys),output)
        print("Test accuracy of this network is: ", accuracy)




# Here starts the program
with open(samples_path, 'rb') as handle:
    pd = pickle.load( handle )

ps = pd.dataset #Processed Samples

if use_embeddings:
    emb = np.array(pd.embeddings[:pd.vocab_size], dtype=np.float32)
else:
    emb = np.random.randn(pd.vocab_size, pd.emb_size).astype(np.float32)


emb_init, W, emb_placeholder = init_embedding(pd.vocab_size, pd.emb_size)
train_neural_network(ps,emb_init,W,emb_placeholder)



print ("=== Code ran Successfully ===")
