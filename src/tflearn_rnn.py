# -*- coding: utf-8 -*-

import tflearn
import numpy as np
import tensorflow as tf

import string
import re
import collections
import math
import random
import os
import json
import pickle
from collections import namedtuple
from copy import deepcopy

import common_funs
import settings

"""
If you get "UnicodeEncodeError: 'charmap' codec can't encode character" on windows,
 write this in console before running the code: chcp 65001 & cmd
"""
# settings ############################################################################

vocabulary_size = settings.vocabulary_size #should match actual dictionary
embedding_size = settings.embedding_size
epochs = settings.epochs
batch_size = settings.batch_size
max_sequence = settings.max_sequence
ascii_console = settings.ascii_console
random_embeddings = not settings.use_embeddings

# !! random_labels = settings.random_labels
# !! add_snitch = settings.add_snitch

#samples_path = settings.samples_path
#vocabulary_path = settings.vocabulary_path

neg_label = settings.neg_label
pos_label = settings.pos_label
Dataset = settings.Dataset
Setpart = settings.Setpart
###################################################################################

print("loading data...")

# reverse dictionary
with open( settings.rev_vocabulary_path, 'r', encoding='utf8' ) as rev_vocab_file:
	rev_vocabulary = json.load( rev_vocab_file )

# embeddings
with open( settings.embeddings_path, 'r', encoding='utf8' ) as embeddings_file:
	embeddings = json.load( embeddings_file )

# processed samples
with open(settings.samples_path, 'rb') as handle:
    ps = pickle.load( handle )


# should search replace this
train_ids = ps.train.ids
train_X = ps.train.xs
train_Y = ps.train.ys

validate_ids = ps.valid.ids
validate_X = ps.valid.xs
validate_Y = ps.valid.ys

test_ids = ps.test.ids
test_X = ps.test.xs
test_Y = ps.test.ys


# use random data (random tweets)
if settings.random_data:
	print("Using random data")
	tmp_X = []
	for _ in range( len(train_X) ):
		row = np.random.randint(
			1, (settings.vocabulary_size - 1), 
			size=max_sequence, 
			dtype=np.int32)
		tmp_X.append(row)
	train_X = np.array(tmp_X, dtype=np.int32)

# debug print tweets
for s in range(10):
	print("Sample id (Tweet id): %s, " %(train_ids[s]), end="")
	print("Positive (Sarcastic)" if np.array_equal(train_Y[s], pos_label)
								 else "Negative (normal)")
	print( train_X[s], end="\n" )
	print( " ".join( common_funs.reverse_lookup(train_X[s], 
												rev_vocabulary, 
												settings.ascii_console ) ) 
												+ "\n" )
	

# Network building
net = tflearn.input_data([None, max_sequence], dtype=tf.int32) 
net = tflearn.embedding(net, input_dim=vocabulary_size, 
						     output_dim=embedding_size, 
						     restore=False)							 
net = tflearn.lstm(net, 
				   128,
				   dropout=settings.dropout,
				   dynamic=True)
#net = time_distrubuted(net, tflearn.fully_connected, settings.embedding_size)
net = tflearn.fully_connected(net, 
							  64, 
							  activation='sigmoid',
							  regularizer='L2', 
							  weight_decay=0.01,
							  name="middle")
net = tflearn.fully_connected(net, 
							  2, 
							  activation='softmax', 
							  name="output")
net = tflearn.regression(net, 
	                     optimizer='adam', 
	                     learning_rate=0.001,
                         loss='categorical_crossentropy')

"""
net = tflearn.time_distrubuted(net,
							  fun
							  activation='softmax', 
							  name="theshizzle")
"""			

# add regulizer
"""
theshizzel_tns = tflearn.variables.get_layer_variables_by_name('theshizzel')[0]
tflearn.helpers.regularizer.add_weights_regularizer(theshizzel_tns, 
													loss='L2', 
													weight_decay=0.01)
<<<<<<< HEAD
=======

>>>>>>> 07a184c73bf243a590295e581134ddc5fe0fab92
tflearn.activations.sigmoid (theshizzel_tns)
"""

# create model
shared_name = common_funs.generate_name()
this_run_id = shared_name
this_model_id = shared_name
checkpoint_path = os.path.join("checkpoints")
if not (os.path.isdir(checkpoint_path)):
	os.makedirs(checkpoint_path)
checkpoint_path = os.path.join("checkpoints",this_run_id + "ckpt")
model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=checkpoint_path)

#set embeddings
if random_embeddings:
	emb = np.random.randn(vocabulary_size, embedding_size).astype(np.float32)
else:
	emb = np.array(embeddings[:vocabulary_size], dtype=np.float32)

new_emb_t = tf.convert_to_tensor(emb)
embeddings_tensor = tflearn.variables.get_layer_variables_by_name('Embedding')[0]
model.set_weights( embeddings_tensor, new_emb_t)
w = model.get_weights(embeddings_tensor)
print("embedding layer shape: {}".format(w.shape))

# Training
model.fit(X_inputs=train_X, 
		  Y_targets=train_Y, 
		  validation_set=(validate_X, validate_Y), 
		  show_metric=True,
          batch_size=batch_size, 
          n_epoch=epochs, 
          shuffle=False, 
          snapshot_step=settings.snapshot_steps,
          run_id=this_run_id)

# save model
models_path = os.path.join("models")
if not (os.path.isdir(models_path)):
	os.makedirs(models_path)

model_file_path = os.path.join(models_path,this_model_id + ".tfl")
model.save(model_file_path)

# print confusion matrix for the different sets
horiz_bar = "-" * (len(shared_name) + 9 )
print(horiz_bar)
print("runid: " + shared_name + ' |')
print(horiz_bar)

print("\n   TRAINING SET \n")
predictions = model.predict(train_X)
common_funs.binary_confusion_matrix( train_ids, predictions, train_Y)

print("\n   VALIDATION SET \n")
predictions = model.predict(validate_X)
common_funs.binary_confusion_matrix( validate_ids, predictions, validate_Y)