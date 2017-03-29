# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 00:18:02 2017

@author: JustAbanan
"""
import tflearn
import numpy as np
import tensorflow as tf
import os
import pickle
import common_funs

max_sequence = 30
vocabulary_size = 20000

# load pickled data
with open('train_X.pickle', 'rb') as handle:
    train_X = pickle.load( handle )
with open('train_Y.pickle', 'rb') as handle:
    train_Y = pickle.load( handle )
with open('test_X.pickle', 'rb') as handle:
    test_X = pickle.load( handle )
with open('test_Y.pickle', 'rb') as handle:
    test_Y = pickle.load( handle )

# Network building
net = tflearn.input_data([None, max_sequence], dtype=tf.int32) 
net = tflearn.embedding(net, input_dim=vocabulary_size, output_dim=256)
net = tflearn.lstm(net, 256 , dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')


model = tflearn.DNN(net, tensorboard_verbose=3)

# Training
this_run_id = '6'
this_model_id = '6'
model_file_path = os.path.join("models", this_model_id + ".tfl")
model.load(model_file_path)


# print confusion matrix for the different sets
print("\n   TRAINING SET \n")
predictions = model.predict(train_X)
common_funs.binary_confusion_matrix( [], predictions, train_Y)

#print("\n   VALISDATION SET \n")
#predictions = model.predict(validate_X)
#common_funs.binary_confusion_matrix( validate_ids, predictions, validate_Y)

print("\n   TEST SET \n")
predictions = model.predict(test_X)
common_funs.binary_confusion_matrix( [], predictions, test_Y)