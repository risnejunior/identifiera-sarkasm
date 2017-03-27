# -*- coding: utf-8 -*-

# import tflearn
# from tflearn.data_utils import to_categorical, pad_sequences

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Activation
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
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
train_X = sequence.pad_sequences(np.array(t_train_s[1]), maxlen=max_sequence)
validate_X = sequence.pad_sequences(np.array(t_validation_s[1]), maxlen=max_sequence)
test_X = sequence.pad_sequences(np.array(t_test_s[1]), maxlen=max_sequence)
# Converting labels to binary vectors
#Ys

train_Y = [i[0] for i in t_train_s[2]]
validate_Y = [i[0] for i in t_validation_s[2]]
test_Y = [i[0] for i in t_test_s[2]]

train_Y = to_categorical(train_Y, 2)
validate_Y = to_categorical(validate_Y, 2)
test_Y = to_categorical(test_Y, 2)

# pickle data to file
samples = {
	"train_X": train_X,
	"train_Y": train_Y,
	"validate_X": validate_X,
	"validate_Y": validate_Y,
	"test_X": test_X,
	"test_Y": test_Y
}

with open('samples.pickle', 'wb') as handle:
    pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

# TensorBoard
tb_callback = TensorBoard(log_dir='/tmp/logs', histogram_freq=0, write_graph=True, write_images=False)

# Network building
model = Sequential()
model.add(Embedding(vocabulary_size+2, output_dim=256, mask_zero=True))
model.add(GRU(128, dropout=0.8))
model.add(Dense(2, activation='sigmoid'))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training
model.fit(train_X, train_Y,
          batch_size=batch_size,
          epochs=15,
          validation_data=(validate_X, validate_Y),
          callbacks=[tb_callback])
score, acc = model.evaluate(validate_X, validate_Y,
                            batch_size=batch_size)

# save model
models_path = os.path.join("models")
if not (os.path.isdir(models_path)):
	os.makedirs(models_path)

model_file_path = os.path.join(models_path,this_model_id + ".tfl")
# model.save(model_file_path)


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

print("\n   TEST SET \n")
predictions = model.predict(test_X)
common_funs.binary_confusion_matrix( test_ids, predictions, test_Y)
