# -*- coding: utf-8 -*-

#import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import string
import numpy as nm
import re
import collections
import math
#import tensorflow as tf
import random
import os
import pickle
import nltk
from nltk.corpus import stopwords
from collections import Counter, OrderedDict
import json

# settings ############################################################################
remove_punctuation = True
remove_stopwords = False
max_size_vocabulary = 10000
partition_training = 0.7
partition_validation = 0.15
partition_test = 0.15
sample_length = 25 #word length of sample, larger samples will be padded

rel_data_path = os.path.join(".","..", "datasets","poria_balanced_cleaned")
path_name_normal = os.path.join(rel_data_path, "normal") #39'967
path_name_sarcastic = os.path.join(rel_data_path, "sarcastic") #36'366 # sum: 76'333

vocabulary_file_name = 'balanced_vocabulary.pickle'
pos_data_file_name = 'balanced_vocabulary.pickle'
vocabulary_file_name = 'balanced_vocabulary.pickle'

########################################################################################

t_table = dict( ( ord(char), None) for char in string.punctuation ) #translation tabler  for puctuation
file_list_normal = os.listdir(path_name_normal)
file_list_sarcastic = os.listdir( path_name_sarcastic )
file_list_all = file_list_normal + file_list_sarcastic

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# input: 2-dim array of toeknized texts
# input: words to be put in vocabulary
# input: max size of vocabulary, uncommon words will be mapped to zero
# output: modified vocabulary dict
# output: modified vocabulary dict reversed
def build_vocabulary( words, max_size ):
	d = dict( collections.Counter(words) ) #.most_common(15) )
	vocabulary = OrderedDict( sorted(d.items(), key=lambda t: t[1],  reverse=True) )
	i = 1
	for key, value in vocabulary.items():
		if i < max_size:
			vocabulary[key] = i
			i += 1
		else:
			pass
			vocabulary[key] = 0

	rev_vocabulary = {v: k for k, v in vocabulary.items()}
	rev_vocabulary[0] = '_unknown_'
	return vocabulary, rev_vocabulary

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# imput: text
# inpuit: dictionary hash list
# output toeknized text
def tokenize_text( file_path ):
	words = []
	processed_text = []
	with open(file_path, 'r', encoding='utf8') as f:
		for line in f:
			if remove_punctuation:
				cleaned = line.lower().translate( t_table ) # remove punctuation and make lowercase
			
			tokens = nltk.word_tokenize( cleaned, language='english') #tokenize text
			if remove_stopwords:
				tokens = [w for w in tokens if not w in stopwords.words('english')] #remove stopwords

			processed_text.extend( tokens )
			
	return processed_text

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# input: tokenized texts
# input: vocabulary dict
# output: texts transfomed to token intger representation
def make_index_vectors( texts, vocabulary ):
	index_vectors = []
	for text in texts:
		index_vector = []
		for word in text:
			try:
				#print (word.encode('unicode-escape'))
				#print (vocabulary[word])
				index_vector.append (vocabulary[word])
			except KeyError:
				print ('Token not in vocabulary')
				quit()
		index_vectors.append( index_vector )
		#print (index_vector)
	#print (index_vectors)
	return index_vectors

def reverse_lookup( index_vector, rev_vocabulary ):
	text = []
	for i in index_vector:
		text.append( rev_vocabulary[i] )
	return text

# toekenize normal text
normal_texts = []
all_words = []
for file_name in file_list_normal[:1150]:
	file_path = os.path.join(path_name_normal, file_name)
	text_tokens = tokenize_text( file_path )
	all_words.extend(text_tokens)
	file_name, _ = file_name.split('.')
	normal_texts.append( {'id': file_name, 'sarcastic': False, 'text': text_tokens} )

# toekenize sarcastic text
sarcastic_texts = []
for file_name in file_list_sarcastic[:1150]:
	file_path = os.path.join(path_name_sarcastic, file_name)
	text_tokens = tokenize_text( file_path )
	file_name, _ = file_name.split('.')
	sarcastic_texts.append( {'id': file_name, 'sarcastic': True, 'text': text_tokens} )
	

all_texts = sarcastic_texts + normal_texts
json_text= json.dumps(all_texts, sort_keys=False, indent=4, separators=( ',',': '))
with open('test.json', 'w', encoding='utf8') as out_file:
	out_file.write(json_text)
#print(json_text)
quit()

# build vocabulary
vocabulary, rev_vocabulary = build_vocabulary(all_words, max_size_vocabulary)

# make index vectors
sarcastic_indexed = make_index_vectors( sarcastic_texts, vocabulary )
normal_indexed = make_index_vectors( normal_texts, vocabulary)

# make label vectors
sarcastic_labels = [1 for _ in range( len(sarcastic_indexed) )]
normal_labels = [0 for _ in range( len(normal_indexed) )]

# conatecate vectors to labels
sarcastic_labled = list( zip(sarcastic_indexed, sarcastic_labels) )
normal_labled = list( zip(normal_indexed, normal_labels) )

#zip the list shuffle them and unzip
#the seed should be kept the same so we 
# always get the same shuffle
all_labled = sarcastic_labled + normal_labled
random.Random(1337).shuffle( all_labled ) 
all_indexed, all_labels = zip(*all_labled)

	#for i in range( len(all_indexed) ):
	#	print ( all_indexed[i], end="")
	#	print ( " , ", end="")
	#	print ( all_labels[i])

#calculate  training and validation set size
sample_size = len( all_indexed )
training_size = math.floor( partition_training * sample_size )
validation_size = math.floor( partition_validation * sample_size )

# slice data into training, validation & test sets
trainX = all_indexed[:training_size]
validateX = all_indexed[training_size:( training_size + validation_size ) ] 
trainY = all_labels[:training_size]
validateY = all_labels[training_size:( training_size + validation_size ) ]

print("training set size " + str( len(trainX) ) )
print("validation set size " + str( len(validateX) ) )
print("training labels size " + str( len(trainY) ) )
print("validation label size " + str( len(validateY) ) )

# pad samples to tweet max length
trainX = pad_sequences(trainX, maxlen=sample_length, value=0.) 
validateX = pad_sequences(validateX, maxlen=sample_length, value=0.)

# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
validateY = to_categorical(validateY, nb_classes=2)


###############################
for s in range(15):
	print( trainX[s], end=" , " )
	print( validateY[s] )
	print( reverse_lookup(trainX[s], rev_vocabulary ) )
	print()
"""

# Network building
net = tflearn.input_data([None, sample_length]) #changed from 100
net = tflearn.embedding(net, input_dim=max_size_vocabulary, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
this_run_id = '1'
model = tflearn.DNN(net, tensorboard_verbose=3)
model.fit(trainX, trainY, validation_set=(validateX, validateY), show_metric=True,
          batch_size=32, n_epoch=10, run_id=this_run_id)

model.save("models/rnn.tfl")

predictions = model.predict(trainX)
facit = list( zip( predictions, trainY ) )

for p in facit[:15]:
	print( p )
"""