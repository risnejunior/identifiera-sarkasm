# -*- coding: utf-8 -*-

from collections import Counter, OrderedDict
import string
import re
import math
import random
import os
import pickle
import json
from sys import getsizeof
import sys

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

import common_funs
import importlib
import settings


# the nltk casual toeknizer, reduce_len keeps repeating chars to 3 max
tknzr = TweetTokenizer(reduce_len=True, preserve_case=False)

# json files will be written all in one row without indentation unless
#  debug_print is True
j_indent = 4 if settings.print_debug else None

# If you don't have the packages installed..
if not settings.print_debug: nltk.download("stopwords"); nltk.download("punkt")
print()

logger = common_funs.Logger()

#t_table = dict( ( ord(char), None) for char in string.punctuation ) #translation tabler  for puctuation
t_table = dict( ( ord(char), None) for char in ['.','_'] ) #translation tabler  for puctuation



print ("\n".join(sys.argv[1:]))
#### functions ###############################################################################


def build_vocabulary( words, max_size ):	
	vocab_instances = 0 									# vocabulary word instances in corpus
	d = dict( Counter(words) )
	pb = common_funs.Progress_bar(len(d) - 1) 
	vocabulary = OrderedDict( sorted(d.items(), key=lambda t: t[1],  reverse=True) )
	i = 2 													#leave room for padding & unknown
	for key, value in vocabulary.items():
		pb.tick()
		if i < max_size:
			vocab_instances += value
			vocabulary[key] = i
			i += 1
		else:			
			vocabulary[key] = 1		
		
	vocabulary[settings.padding_char] = 0
	vocabulary[settings.placeholder_char] = 1
	rev_vocabulary = {v: k for k, v in vocabulary.items()}
	return vocab_instances, vocabulary, rev_vocabulary

def tokenize_text( file_path ):
	global sequence_lengths	
	processed_text = []
	with open(file_path, 'r', encoding='utf8') as f:
		for line in f:
			
			if settings.remove_punctuation:
				cleaned = line.lower().translate( t_table ) 	
			else:
				cleaned = line

			if settings.use_casual_tokenizer:
				tokens = tknzr.tokenize( cleaned ) 
			else:
				tokens = nltk.word_tokenize( cleaned, language='english') 	
						
			if settings.remove_stopwords:
				tokens = [w for w in tokens if not w in stopwords.words('english')] 	

			sequence_lengths.append( len( tokens ) )
			processed_text.extend( tokens )		
	return processed_text

def tokenize_helper(path_name, file_list, samples, all_words, sarcastic):
	file_count = len(file_list)
	print("Tokenizing %i %s samples (tweets)" 
		%(file_count, "positive (sarcastic)" if sarcastic else "negative (normal)" ) )
	pb = common_funs.Progress_bar( file_count-1 )
	for file_name in file_list:
		file_path = os.path.join(path_name, file_name)
		text_tokens = tokenize_text( file_path )
		all_words.extend(text_tokens)
		file_name, _ = file_name.split('.')
		samples[file_name] = {'sarcastic': sarcastic, 'text': text_tokens, 'int_vector':[]}
		pb.tick()

	print()

# input: tokenized samples dict
# input: vocabulary dict
# 
# Adds int vector to sample dict
def make_index_vectors( samples, vocabulary ):
	for key, value in samples.items():
		int_vector = []
		for word in value['text']:
			try:
				int_vector.append( vocabulary[word] )
			except KeyError:
				print ('Token not in vocabulary')
				quit()
		samples[key]['int_vector'] = int_vector

def reverse_lookup( index_vector, rev_vocabulary ):
	text = []
	for i in index_vector:
		text.append( rev_vocabulary[i] )
	return text

def reshape_embedding(vocabulary, embeddings_voc):
	pb = common_funs.Progress_bar( len(vocabulary)-1 )
	rt = random.triangular
	minval = 0.0
	maxval = 0.0
	not_found = 0
	embeddings = []
	for word, i in vocabulary.items():
		if word in embeddings_voc:
			string_vector = embeddings_voc[word]
		else:
			# this should be calculated the same as below (min, max)
			string_vector = [rt(-6.0, 6.0) for _ in range(settings.embedding_size)]
			not_found += 1
			logger.log(word, "missing_embeddings", 100)

		float_vec = list(map(lambda x: float(x), string_vector)	)
		minval = min(float_vec) if min(float_vec) < minval else minval
		maxval = max(float_vec) if max(float_vec) > maxval else maxval

		embeddings.append(float_vec)
		
		logger.log(float_vec, logname="embeddings", step=1000)
		pb.tick()

	
	embeddings[0] = [0.0 for _ in range(settings.embedding_size)]
	embeddings[1] = [rt(minval, maxval) for _ in range(settings.embedding_size)]

	logger.log(minval, logname="min")
	logger.log(maxval, logname="max")
	logger.log(not_found, logname="not_found")
	logger.log(embeddings[0], logname="padding")
	logger.log(embeddings[1], logname="placeholder")

	return embeddings

###########################################################################################
# > log stats throughout
# 1. load samples from file list <id, text, label>
# 2. tokenize text
# 3. build vocabulary
# 4. build embeddings
# 5. shuffle samples
# 6. split into training set, etc.
# 6. 
# 5. print logs
# 6. save samples

neg_label = settings.neg_label
pos_label = settings.pos_label

file_list_normal = os.listdir(settings.path_name_neg)[:settings.sample_count]
file_list_sarcastic = os.listdir(settings.path_name_pos)[:settings.sample_count]
file_list_all = file_list_normal + file_list_sarcastic

samples = {}
all_words = []
sequence_lengths = []

print("{} sample files found (positive + negative)\n".format(len(file_list_all)))

tokenize_helper(settings.path_name_neg,
			    file_list_normal, 
			    samples, 
			    all_words,
			    False)
tokenize_helper(settings.path_name_pos,
				file_list_sarcastic, 
				samples, 
				all_words, 
				True)


# build vocabulary
print("Building vocabulary..")
vocab_instances, vocabulary, rev_vocabulary = \
	build_vocabulary(all_words, settings.vocabulary_size)

#load embeddings vocabulary
if settings.use_embeddings:
	print()
	wa = common_funs.working_animation("Loading embeddings vocabulary")
	embeddings_voc = {}
	with open(settings.emb_voc_path, encoding="utf8") as emb_file:
		for i, line in enumerate(emb_file):
			wa.tick("Embeddings loaded: " + str(i))
			(word, *vector) = line.split()
			embeddings_voc[word] = vector
		wa.done()

	# fit embeddings to vocabulary
	print()
	print("Reshaping embeddings...")
	logger.log(getsizeof(embeddings_voc), logname="embedding_bytes")
	embeddings = reshape_embedding(vocabulary, embeddings_voc)

	json_embedding= json.dumps(
		embeddings, ensure_ascii=False, indent=j_indent, separators=( ',',': '))
	with open(settings.embeddings_path, 'w', encoding='utf8') as out_file:
		out_file.write(json_embedding)	

# print word stats
print('\n')
print("Calculating statistics...")
seq_max = max( sequence_lengths )
seq_mean = round( np.mean( sequence_lengths ), 2 )
seq_std = round( np.std( sequence_lengths ), 2 )
print("Longest sqeuence (words): " + str( seq_max) , end =", ") 
print("mean: " + str( seq_mean), end =", ") 
print("std: " + str( seq_std), end =", ") 
print("3-sigma: " + str(math.ceil( seq_mean + 3 * seq_std) ) )
print("Words in corpus: {:0}, Unique words in corpus: {:1}" \
	.format( len(all_words), len(vocabulary) ) )
print("Vocabulary size: {:0}, Vocabulary coverage of corpus {:1.0%}" \
	.format(settings.vocabulary_size, vocab_instances / len(all_words) ) ) 
print('\n')

# make index vectors
print ("Making index vectors..")
make_index_vectors( samples, vocabulary )
print( str( len(samples) ) + " samples indexed")


if settings.print_debug:
	all_samples= json.dumps(
		samples, ensure_ascii=False, indent=j_indent, separators=( ',',': ')
	)
	with open(settings.debug_samples_path, 'w', encoding='utf8') as out_file:
		out_file.write(all_samples)	


int_vectors = []
ids = []
labels = []
sample_count = len(samples)

# assign category labels
print("Assigning category labels...")
positive_count = negative_count = 0
positive_max = math.ceil(settings.set_balance * settings.sample_count)
negative_max = settings.sample_count - positive_max
for key, val in samples.items():
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

	if settings.random_labels:
		if random.randint(1,2) == 1:
			labels.append( pos_label )
		else:
			labels.append( neg_label )
	elif val['sarcastic'] == True:
		labels.append( pos_label )
		if settings.add_snitch: val['int_vector'].extend( 
			[settings.vocabulary_size-1] )
	else:
		labels.append( neg_label )

	int_vectors.append( np.array( val['int_vector'], dtype="int32" ) )
	ids.append( key )


#zip the list shuffle them and unzip
#the seed should be kept the same so we 
# always get the same shuffle
labeld_samples =  list( zip(ids, int_vectors, labels) ) 
random.Random(1).shuffle( labeld_samples ) 
#ids, int_vectors, labels = zip(*labeld_samples)

#calculate  training and validation set size
sample_size = len( labeld_samples )
training_size = math.floor( settings.partition_training * sample_size )
validation_size = math.floor( settings.partition_validation * sample_size )

# slice data into training, validation & test sets
train_samples = labeld_samples[:training_size]
validation_samples = labeld_samples[
	training_size:( training_size + validation_size ) ]
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
train_Y = t_train_s[2]
validate_Y = t_validation_s[2]
test_Y = t_test_s[2]

train_X = common_funs.pad_sequences( np.array( t_train_s[1] ), 
	padding=settings.padding_pos, 
	maxlen=settings.max_sequence, value=0.)
validate_X = common_funs.pad_sequences( np.array( t_validation_s[1] ), 
	padding=settings.padding_pos, 
	maxlen=settings.max_sequence, value=0.)
test_X = common_funs.pad_sequences( np.array( t_test_s[1] ), 
	padding=settings.padding_pos,
	 maxlen=settings.max_sequence, value=0.)

# package it all in a named touple
spt_train = settings.Setpart('training set', len(train_ids), train_ids, train_X, train_Y)
spt_val = settings.Setpart('validation set', len(validate_ids), validate_ids, validate_X, validate_Y)
spt_test = settings.Setpart('test set', len(test_ids), test_ids, test_X, test_Y)
ds = settings.Dataset(spt_train, spt_val, spt_test)

for setpart in ds:
	for i in range(setpart.length - 1):
		name, _, ids, xs, ys = setpart
		line = "name: {} id:{} x:{} y:{}".format(name, ids[i], xs[i], ys[i])
		logger.log("processed", line, 1000, 5)

#save to json
print ("Saving to disk..")

with open(settings.samples_path, 'wb') as handle:
    pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

json_vocabulary= json.dumps(vocabulary, ensure_ascii=False, indent=j_indent, separators=( ',',': '))
with open(settings.vocabulary_path, 'w', encoding='utf8') as out_file:
	out_file.write(json_vocabulary)

json_rev_vocabulary= json.dumps(rev_vocabulary, ensure_ascii=False, indent=j_indent, separators=( ',',': '))
with open(settings.rev_vocabulary_path, 'w', encoding='utf8') as out_file:
	out_file.write(json_rev_vocabulary)

logger.save(file_name="preprocess.log")

# use random data (random tweets)
"""
if random_data:
	print("Using random data")
	tmp_X = []
	for _ in range( len(train_X) ):
		row = np.random.randint(
			1, (vocabulary_size - 1), 
			size=max_sequence, 
			dtype=np.int32)
		tmp_X.append(row)
	train_X = np.array(tmp_X, dtype=np.int32)
"""
