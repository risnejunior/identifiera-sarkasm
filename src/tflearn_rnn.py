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
import pickle
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
###################################################################################

print("loading data...")
# get data
with open( samples_path, 'r', encoding='utf8' ) as samples_file:
	samples_json = json.load( samples_file )

#get dictionary
with open( rev_vocabulary_path, 'r', encoding='utf8' ) as rev_vocab_file:
	rev_vocabulary = json.load( rev_vocab_file )

#get embeddings
with open( embeddings_path, 'r', encoding='utf8' ) as embeddings_file:
	embeddings = json.load( embeddings_file )

int_vectors = []
ids = []
labels = []
sample_count = len(samples_json)
sarcastic_label = np.array([0., 1.], dtype="float32")
normal_label = np.array([1., 0.], dtype="float32")

# assign category labels
print("Assigning category labels...")
positive_count = negative_count = 0
positive_max = math.ceil(set_balance * sample_count)
negative_max = sample_count - positive_max
for key, val in samples_json.items():
	pos = val['sarcastic']
	if pos:
		if positive_count > positive_max:
			continue
		else:
			positive_count += 1
	else:
		if negative_count > negative_max:
			continue
		else:
		 	negative_count += 1

	if random_labels:
		if random.randint(1,2) == 1:
			labels.append( sarcastic_label )
		else:
			labels.append( normal_label )
	elif val['sarcastic'] == True:
		labels.append( sarcastic_label )
		if add_snitch: val['int_vector'].extend( [vocabulary_size-1] )
	else:
		labels.append( normal_label )


	int_vectors.append( np.array( val['int_vector'], dtype="int32" ) )
	#print(val['int_vector'])
	#quit()
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

# friendlier names
train_ids = t_train_s[0]
validate_ids = t_validation_s[0]
test_ids = t_test_s[0]
# pad s to tweet max length
#Xs
train_X = pad_sequences( np.array( t_train_s[1] ), maxlen=max_sequence, value=0.)
validate_X = pad_sequences( np.array( t_validation_s[1] ), maxlen=max_sequence, value=0.)
test_X = pad_sequences( np.array( t_test_s[1] ), maxlen=max_sequence, value=0.)
# Converting labels to binary vectors
#Ys
#train_Y = to_categorical(t_train_s[2], nb_classes=2)
#validate_Y = to_categorical(t_validation_s[2], nb_classes=2)
#test_Y = to_categorical(t_test_s[2], nb_classes=2)
train_Y = t_train_s[2]
validate_Y = t_validation_s[2]
test_Y = t_test_s[2]

# pickle data to file
with open('train_X.pickle', 'wb') as handle:
    pickle.dump(train_X, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('train_Y.pickle', 'wb') as handle:
    pickle.dump(train_Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('test_X.pickle', 'wb') as handle:
    pickle.dump(test_X, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('test_Y.pickle', 'wb') as handle:
    pickle.dump(test_Y, handle, protocol=pickle.HIGHEST_PROTOCOL)

# use random data (random tweets)
if settings.random_data:
	tmp_X = []
	for _ in range( len(train_X) ):
		row = np.random.randint(
			1, (settings.vocabulary_size - 1), 
			size=max_sequence, 
			dtype=np.int32)
		tmp_X.append(row)
	train_X = np.array(tmp_X, dtype=np.int32)

for s in range(25):
	#header
	print("Sample id (Tweet id): %s, " %(train_ids[s]), end="")
	print("Positive (Sarcastic)" if train_Y[s] is sarcastic_label \
		else "Negative (normal)")
	# dictionary index vector
	print( train_X[s], end="\n" )
	# reverse lookup
	print( " ".join( common_funs.reverse_lookup(train_X[s], rev_vocabulary ) ) )
	print()

#os.system("pause")

# Network building
net = tflearn.input_data([None, max_sequence], dtype=tf.int32)
net = tflearn.embedding(net, input_dim=vocabulary_size, output_dim=embedding_size, restore=True)
net = tflearn.gru(net, embedding_size , dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# create model
this_run_id = '8'
this_model_id = '8'
checkpoint_path = os.path.join("checkpoints")
if not (os.path.isdir(checkpoint_path)):
	os.makedirs(checkpoint_path)
checkpoint_path = os.path.join("checkpoints",this_run_id + "ckpt")
model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=checkpoint_path)

#set embeddings
if random_embeddings:
	emb = np.random.randn(vocabulary_size, embedding_size).astype(np.float32)
	#train_X = a = np.random.uniform(
	#	-2, 2, (, settings.embedding_size)).astype(np.float32)
else:
	emb = np.array(embeddings[:vocabulary_size], dtype=np.float32)


new_emb_t = tf.convert_to_tensor(emb)
embeddings_tensor = tflearn.variables.get_layer_variables_by_name('Embedding')[0]
model.set_weights( embeddings_tensor, new_emb_t)
print("embedding layer weights:")
w = model.get_weights(embeddings_tensor)
print( w.shape )
	

# Training #run_id=this_run_id
model.fit(X_inputs=train_X, Y_targets=train_Y,
		  validation_set=(validate_X, validate_Y),
		  show_metric=True,
          batch_size=batch_size, n_epoch=epochs,
          shuffle=False,
          snapshot_step=settings.snapshot_steps)


# save model
models_path = os.path.join("models")
if not (os.path.isdir(models_path)):
	os.makedirs(models_path)

model_file_path = os.path.join(models_path,this_model_id + ".tfl")
model.save(model_file_path)


# print confusion matrix for the different sets
print("\n   TRAINING SET \n")
predictions = model.predict(train_X)
common_funs.binary_confusion_matrix( train_ids, predictions, train_Y)

print("\n   VALISDATION SET \n")
predictions = model.predict(validate_X)
common_funs.binary_confusion_matrix( validate_ids, predictions, validate_Y)

print("\n   TEST SET \n")
predictions = model.predict(test_X)
common_funs.binary_confusion_matrix( test_ids, predictions, test_Y)
