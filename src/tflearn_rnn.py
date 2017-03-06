# -*- coding: utf-8 -*-

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import string
import numpy as np
import re
import collections
import math
import tensorflow as tf
import random
import os
import json
from copy import deepcopy
import sklearn as sk

# settings ############################################################################

vocabulary_size = 10000 #should match actual dictionary
partition_training = 0.7
partition_validation = 0.15
partition_test = 0.15
max_sequence = 230 #word length of sample, larger samples will be padded

#poria
rel_data_path = os.path.join(".","..", "datasets","poria")
samples_path = os.path.join(rel_data_path, "processed.json") 
vocabulary_path = os.path.join(rel_data_path, "vocabulary.json") 
rev_vocabulary_path = os.path.join(rel_data_path, "rev_vocabulary.json")

#imdb
rel_data_path = os.path.join(".","..", "datasets","imdb")
samples_path = file_name = os.path.join(rel_data_path, 'processed_imdb.json')
vocabulary_path = file_name = os.path.join(rel_data_path, 'vocabulary_imdb.json')
rev_vocabulary_path = file_name = os.path.join(rel_data_path, 'rev_vocabulary_imdb.json')

###################################################################################

#l = [1,2,3,4,5,6]
#k = ['a','b','c','d','e','f']
#m = [1,1,1,0,0,0]
#m = to_categorical(m, nb_classes=2)
#lkm = list( zip(l, k, m) )
#random.Random(1).shuffle(lkm)
#print(lkm)
#quit()

def reverse_lookup( index_vector, rev_vocabulary ):
	text = []
	for i in index_vector:
		text.append( rev_vocabulary[str(i)].encode('unicode-escape') )
	return text


# get data
with open( samples_path, 'r', encoding='utf8' ) as samples_file:
	samples_json = json.load( samples_file )

#get dictionary
with open( rev_vocabulary_path, 'r', encoding='utf8' ) as rev_vocab_file:
	rev_vocabulary = json.load( rev_vocab_file )

int_vectors = []
ids = []
labels = []

for key, val in samples_json.items():
	if val['sarcastic'] == True:
		labels.append( 1 )
	else:
		labels.append( 0 )
	int_vectors.append( val['int_vector'] )
	ids.append( key )




#zip the list shuffle them and unzip
#the seed should be kept the same so we 
# always get the same shuffle
labeld_samples =  list( zip(ids, int_vectors, labels) ) 
random.Random(1).shuffle( labeld_samples ) 
#ids, int_vectors, labels = zip(*labeld_samples)

#calculate  training and validation set size
sample_size = len( labeld_samples )
training_size = math.floor( partition_training * sample_size )
validation_size = math.floor( partition_validation * sample_size )

# slice data into training, validation & test sets
train_samples = labeld_samples[:training_size]
validation_samples = labeld_samples[training_size:( training_size + validation_size ) ]
test_samples = labeld_samples[training_size + validation_size:]

# transpose list: # [0]:id - [1] int_vector - [2] label
t_train_s = list(map(list, zip(*train_samples)))
t_validation_s = list(map(list, zip(*validation_samples)))
t_test_s = list(map(list, zip(*test_samples)))

# pad s to tweet max length
#Xs
t_train_s[1] = pad_sequences(t_train_s[1], maxlen=max_sequence, value=0.)
t_validation_s[1] = pad_sequences(t_validation_s[1], maxlen=max_sequence, value=0.)
t_test_s[1] = pad_sequences(t_test_s[1], maxlen=max_sequence, value=0.)
# Converting labels to binary vectors
#Ys
t_train_s[2] = to_categorical(t_train_s[2], nb_classes=2)
t_validation_s[2] = to_categorical(t_validation_s[2], nb_classes=2)
t_test_s[2] = to_categorical(t_test_s[2], nb_classes=2)

###############################
for s in range(10):
	print( t_train_s[0][s], end=" , " )
	print( t_train_s[1][s], end=" , " )
	print( t_train_s[2][s] )
	print( reverse_lookup(t_train_s[1][s], rev_vocabulary ) )
	print()

acc = tflearn.metrics.Accuracy

# Network building
net = tflearn.input_data([None, max_sequence]) #changed from 100
net = tflearn.embedding(net, input_dim=vocabulary_size, output_dim=100)
net = tflearn.lstm(net, 100, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

#%%

# Training
this_run_id = '1'
model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path='checkpoints/')
model.fit(t_train_s[1], t_train_s[2], validation_set=(t_validation_s[1], t_validation_s[2]), show_metric=True,
          batch_size=32, n_epoch=1, run_id=this_run_id, shuffle=True)

model.save("models/rnn.tfl")

print("training set>")
predictions = model.predict(t_train_s[1])
facit = list( zip( predictions, t_train_s[2] ) )
for p in facit[:15]:
	print( p )

print("test set>")
predictions = model.predict(t_test_s[1])
facit = list( zip( predictions, t_train_s[2] ) )
for p in facit[:15]:
	print( p )

print ("f1_score:" ) 
print( confusion_matrix(t_train_s[2], predictions) ) 




#1. unpack json
#2. ghetto accuracy calc
#3. sklearn f1