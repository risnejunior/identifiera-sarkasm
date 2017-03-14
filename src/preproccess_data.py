# -*- coding: utf-8 -*-

import string
import re
import collections
import math
import random
import os
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import Counter, OrderedDict
import json
import common_funs
import importlib

# settings ############################################################################
print_debug = False
remove_punctuation = True
remove_stopwords = False
max_size_vocabulary = 10000 #words that don't fit get indexed as 0
sample_count = 36366 #how many of both class3es of samples to use, to ensure they are 50/50
twitter = True #twitter or imdb

if twitter:
	#tweets
	rel_data_path = os.path.join(".","..", "datasets","poria")
	path_name_normal = os.path.join(rel_data_path, "en-balanced","cleaned","normal") #39'967
	path_name_sarcastic = os.path.join(rel_data_path, "en-balanced","cleaned","sarcastic") #36'366 # sum: 76'333
	proc_file_name = file_name = os.path.join(rel_data_path, 'processed.json')
	voc_file_name = file_name = os.path.join(rel_data_path, 'vocabulary.json')
	rev_voc_file_name = file_name = os.path.join(rel_data_path, 'rev_vocabulary.json')
else:
	#imdb
	rel_data_path = os.path.join(".","..", "datasets","imdb")
	path_name_normal = os.path.join(rel_data_path, "neg") #12'500
	path_name_sarcastic = os.path.join(rel_data_path, "pos") #12'500
	proc_file_name = file_name = os.path.join(rel_data_path, 'processed_imdb.json')
	voc_file_name = file_name = os.path.join(rel_data_path, 'vocabulary_imdb.json')
	rev_voc_file_name = file_name = os.path.join(rel_data_path, 'rev_vocabulary_imdb.json')

############################################################################################

# json files will be written all in one row without indentation unless
#  debug_print is True
j_indent = 4 if print_debug else None

# If you don't have the packages installed..
if not print_debug: nltk.download("stopwords"); print()

t_table = dict( ( ord(char), None) for char in string.punctuation ) #translation tabler  for puctuation
file_list_normal = os.listdir(path_name_normal)[:sample_count]
file_list_sarcastic = os.listdir( path_name_sarcastic )[:sample_count]
file_list_all = file_list_normal + file_list_sarcastic
#### functions ###############################################################################


# input: 2-dim array of toeknized texts
# input: words to be put in vocabulary
# input: max size of vocabulary, uncommon words will be mapped to zero
# output: modified vocabulary dict
# output: modified vocabulary dict reversed
def build_vocabulary( words, max_size ):
	vocab_instances = 0 # vocabulary word instances in corpus
	d = dict( collections.Counter(words) ) #.most_common(15) )
	vocabulary = OrderedDict( sorted(d.items(), key=lambda t: t[1],  reverse=True) )
	i = 2 #leave room for padding & unknown
	for key, value in vocabulary.items():
		if i < max_size:
			vocab_instances += value
			vocabulary[key] = i
			i += 1
		else:
			pass
			vocabulary[key] = 1

	vocabulary['.'] = 0
	vocabulary['_'] = 1
	rev_vocabulary = {v: k for k, v in vocabulary.items()}
	return vocab_instances, vocabulary, rev_vocabulary

# imput: text
# inpuit: dictionary hash list
# output toeknized text
def tokenize_text( file_path ):
	global sequence_length
	words = []
	processed_text = []
	with open(file_path, 'r', encoding='utf8') as f:
		for line in f:
			if remove_punctuation:
				cleaned = line.lower().translate( t_table ) # remove punctuation and make lowercase
			
			tokens = nltk.word_tokenize( cleaned, language='english') #tokenize text
			if remove_stopwords:
				tokens = [w for w in tokens if not w in stopwords.words('english')] #remove stopwords
			
			sequence_length.append( len( tokens ) )
			processed_text.extend( tokens )
			
	return processed_text

def tokenize_helper(path_name, file_list, samples, all_words, normal_texts, sarcastic):
	file_count = len(file_list)
	print("Tokenizing %i %s samples(tweets)" %(file_count, "positive(sarcastic)" if sarcastic else "negative(normal)" ) )
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

###########################################################################################

# tokenize  text
samples = {}
all_words = []
sequence_length = []
print( str( len(file_list_all) ) + " files selected")
if( print_debug ): print ("Tokenizing text..")
tokenize_helper( path_name_normal, file_list_normal, samples, all_words, samples, False )
tokenize_helper( path_name_sarcastic, file_list_sarcastic, samples, all_words, samples, True )
if( print_debug ): print ("Doing math..")
seq_max = max( sequence_length )
seq_mean = round( np.mean( sequence_length ), 2 )
seq_std = round( np.std( sequence_length ), 2 )
print("Longest sqeuence (words): " + str( seq_max) + " ", end ="") 
print("mean: " + str( seq_mean) + " ", end ="") 
print("std: " + str( seq_std) + " ", end ="") 
print("3-sigma: " + str(math.ceil( seq_mean + 3 * seq_std) ) )

# build vocabulary
if( print_debug ): print ("Building vocabulary..")
vocab_instances, vocabulary, rev_vocabulary = \
	build_vocabulary(all_words, max_size_vocabulary)

print("Words in corpus: {:0}, Unique words in corpus: {:1}" \
	.format( len(all_words), len(vocabulary) ) )
print("Vocabulary size: {:0}, Vocabulary coverage of corpus {:1.0%}" \
	.format(max_size_vocabulary, vocab_instances / len(all_words) ) ) 

# make index vectors
if( print_debug ): print ("Making index vectors..")
make_index_vectors( samples, vocabulary )
if (print_debug) : print( str( len(samples) ) + " samples indexed")

#save to json
print ("Saving to disk..")
json_samples = json.dumps(samples, ensure_ascii=False, sort_keys=False, indent=j_indent, separators=( ',',': '))
with open(proc_file_name, 'w', encoding='utf8') as out_file:
	out_file.write(json_samples)

json_vocabulary= json.dumps(vocabulary, ensure_ascii=False, sort_keys=False, indent=j_indent, separators=( ',',': '))
with open(voc_file_name, 'w', encoding='utf8') as out_file:
	out_file.write(json_vocabulary)

json_rev_vocabulary= json.dumps(rev_vocabulary, ensure_ascii=False, sort_keys=False, indent=j_indent, separators=( ',',': '))
with open(rev_voc_file_name, 'w', encoding='utf8') as out_file:
	out_file.write(json_rev_vocabulary)