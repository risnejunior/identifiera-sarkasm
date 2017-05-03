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
import importlib
import csv
from random import triangular as rt

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

import common_funs
from common_funs import Logger
from common_funs import MinMax
from common_funs import Working_animation
from common_funs import Progress_bar
from common_funs import DebugLoop
from common_funs import Arg_handler
from common_funs import Open_Dataset
from common_funs import balance
from common_funs import pad_sequences
from common_funs import interleave

from common_funs import ProcessedData
from common_funs import Dataset
from common_funs import Setpart
from common_funs import pos_label
from common_funs import neg_label

from config import Config

#### functions ###############################################################################

def _arg_callback_sp(train, cross, test):
	cfg.partition_training = float(train)
	cfg.partition_validation = float(cross)
	cfg.partition_test = float(test)

	if (cfg.partition_training + cfg.partition_validation + cfg.partition_test > 1):
		raise ValueException("Sum of partitions cannot exceed 1.")

def	_arg_callback_sb(set_balance):
	cfg.set_balance = float(set_balance)
	print("<Using set balance: {}>".format(set_balance))

def	_arg_callback_le(embeddings_count):
	cfg.embeddings_maxloop = int(embeddings_count)
	print("<Limiting embeddings: {}>".format(set_balance))

def _arg_callback_ds(dataset_name):
	cfg.dataset_name = dataset_name
	print("<Using dataset: {}>".format(dataset_name))

def _arg_callback_reverse():
	cfg.reverse_samples = True
	print("<Reversing samples..>")

def _arg_callback_scramble():
	cfg.scramble_samples = True
	print("<Scrambling samples..>")

def _arg_callback_sd():
	cfg.save_debug = True
	print("<Will save debug files..>")

def _arg_callback_nltk():
	cfg.nltk_dowload = True
	print("<checking nltk and downloading if needed...>")

def _arg_callback_ms():
	#Create a minisample for debugging; will run quickly
	cfg.limit_samples = 1000
	cfg.embeddings_maxloop = 10000
	cfg.vocabulary_size = 5000
	cfg.embedding_size = 25
	cfg.max_sequence = 30
	print("<using mini-sample>")

def _arg_callback_re():
	"""
	Skip fitting embeddings, all will be set to random
	"""
	cfg.embeddings_maxloop = 0
	print("<Using random-embeddings>")

def _arg_callback_ls(s_count=5000):
	"""
	Limit the samples used (how many teewts to preprocess)
	"""
	cfg.limit_samples = int(s_count)
	print("<Using limited sample count: {}>".format(s_count))

def _arg_callback_pf(file_name):
	"""
	Save preprocessed samples under a different file name
	"""
	cfg.ps_file_name = file_name
	print("<Saving processed samples as: {}>".format(file_name))

def build_vocabulary( words, max_size ):
	vocab_instances = 0
	unique_counts = Counter(words)
	d = dict(unique_counts.most_common(cfg.vocabulary_size-2) )
	vocabulary = OrderedDict( sorted(d.items(), key=lambda t: t[1],  reverse=True) )

	# start at 2 to leave room for padding & unknown
	pb = Progress_bar(len(d) - 1) 
	for i, (key, value) in enumerate(vocabulary.items(), start=2):		
		vocab_instances += value
		vocabulary[key] = i
		pb.tick()

	vocabulary[cfg.padding_char] = 0
	vocabulary[cfg.placeholder_char] = 1
	#reverse the vocbulary (for reverse lookup)
	rev_vocabulary = {v: k for k, v in vocabulary.items()}	
	vocab = (len(unique_counts), vocab_instances, vocabulary, rev_vocabulary)

	return vocab

def tokenize_text( sample_text ):
	global sequence_lengths
	processed_text = []

	if cfg.remove_punctuation:
		cleaned = sample_text.lower().translate( t_table )
	else:
		cleaned = sample_text

	if cfg.use_casual_tokenizer:
		tokens = tknzr.tokenize( cleaned )
	else:
		tokens = nltk.word_tokenize( cleaned, language='english')

	if cfg.remove_stopwords:
		tokens = [w for w in tokens if not w in stopwords.words('english')]

	sequence_lengths.append( len( tokens ) )
	processed_text.extend( tokens )

	return processed_text

def tokenize_helper(sample_list, all_words, sample_class):
	tokenized_samples =[]
	class_name = "positive" if sample_class == 1 else "negative"	
	list_length = len(sample_list)
	print("Tokenizing {}, {} samples".format(list_length, class_name))
	pb = common_funs.Progress_bar( list_length-1 )
	for sample in sample_list:
		text_tokens = tokenize_text( sample['sample_text'] )
		all_words.extend(text_tokens)
		tokenized_samples.append({
			'sample_id': sample['sample_id'], 
			'sample_class': sample_class, 
			'text_tokens': text_tokens, 
			'int_vector':[]
		})
		pb.tick()

	return tokenized_samples

def make_index_vectors( samples, vocabulary ):
	for sample in samples:
		int_vector = []
		for word in sample['text_tokens']:
			if word in vocabulary:
				int_vector.append( vocabulary[word] )
			else:
				int_vector.append( 1 ) # 1 - used for masking samples

		if cfg.reverse_samples:
			int_vector.reverse()

		if cfg.scramble_samples:
			random.shuffle(int_vector)

		sample['int_vector'] = int_vector

	return samples

		
def transpose_setpart(samples, set_name):
	ids = []; xs = []; ys = [];

	for sample in samples:	
		label = pos_label if sample['sample_class'] == 1 else neg_label

		xs.append(np.array( sample['int_vector'], dtype="int32"))
		ids.append(sample['sample_id'])
		ys.append(label)

	#pad in vectors
	xs = pad_sequences(xs, padding=cfg.padding_pos, maxlen=cfg.max_sequence, value=0.)
	
	return Setpart(set_name, len(ids), ids, xs, ys)

def modify_samples(samples, random_data, add_snitch, random_labels):
	minval = 1
	maxval = cfg.vocabulary_size - 1
	pb = Progress_bar(len(samples)-1)
	for sample in samples:

		int_vector = sample['int_vector']
		sample_class = sample['sample_class']

		if random_data:
			int_vector = [rt(minval, maxval) for _ in range(cfg.max_sequence)] 
		
		if add_snitch: 
			int_vector.extend([cfg.vocabulary_size-1])

		if random_labels:
			sample_class = random.randint(1,2)

		
		sample['int_vector'] = int_vector
		sample['sample_class'] = sample_class
		pb.tick()


def fit_embeddings(vocabulary, source_path):
	debug_logger = Logger(enable = False)

	found = notfound = masks = randvecs = nulls = 0
	rs_embeddings = [[] for _ in range(cfg.vocabulary_size)]
	minmax = MinMax()
	wa = Working_animation("Fit embeddings to vocabulary", 50)
	try:
		with open(source_path, 'r', encoding="utf8") as emb_file:
			dl = DebugLoop(maxloops = cfg.embeddings_maxloop)
			for i, line in enumerate(dl.loop(emb_file)):
				(word, *vector) = line.split()
				vector = list(map(float, vector))

				if word in vocabulary:
					found += 1
					voc_index = vocabulary[word]
					debug_logger.log(voc_index, "vocindexes", aslist=False)

					if voc_index == 1:
						masks += 1
						continue
					elif voc_index == 0:
						nulls += 1
						rs_embeddings[voc_index] = [0.0 for _ in range(cfg.embedding_size)]
					else:
						rs_embeddings[voc_index] = vector
						minmax.add(vector)
				else:
					notfound += 1

				wa.tick("Scanned: {:,}, matched to vocabulary: {:,}".format(i, found))
			wa.done()

		minval = minmax.get('min') if minmax.get('min') else -6.0
		maxval = minmax.get('max') if minmax.get('max') else 6.0
		print("\nGenerating random word vectors for missing words...")
		pb = Progress_bar(len(rs_embeddings) - 1)
		for i, vector in enumerate(rs_embeddings):
			if not vector:
				randvecs += 1
				rs_embeddings[i] = [
					round(rt(minval, maxval), 5) for _ in range(cfg.embedding_size)]
			pb.tick()

	except IndexError as e:
		print(e)
		debug_logger.log(e)
	finally:
		debug_logger.save()

	print()
	print("Words not found in vocabulary: {:,}".format(notfound))
	print("Words found in vocabulary: {:,}".format(found))
	print("Masked words skipped: {:,}".format(masks))
	print("Null vectors written to embedding: {:,}".format(nulls))
	print("Word vectors copied to embedding: {:,}".format(found - masks))
	print("Word vectors coverage of vocabulary: {:1.0%}"
		.format((found - masks) / cfg.vocabulary_size))
	print("Random word vectors written to embedding: {:,}".format(randvecs))
	print("Range for random embedding (min/max): {}".format((minval, maxval)))
	print("Total written: {:,}".format(randvecs + found - masks))

	return rs_embeddings

###########################################################################################

# affected by flags, need to be set before consume_flags()
cfg = Config()
cfg.embeddings_maxloop = None
cfg.nltk_dowload = False
cfg.save_debug = False
cfg.scramble_samples = False
cfg.reverse_samples = False

arghandler = Arg_handler()
arghandler.register_flag('ms', _arg_callback_ms, ['mini-sample'], "Minimal run, with few samples, small vocab, seq. length and few embeddings used.")
arghandler.register_flag('of', _arg_callback_pf, ['out-file', 'out'], "name of output file. Args: <filename>")
arghandler.register_flag('nltk', _arg_callback_nltk, [], "check for nltk, and download if missing")
arghandler.register_flag('sd', _arg_callback_sd, ['save-debug'], "save .json debugging files")
arghandler.register_flag('scramble', _arg_callback_scramble, [], "scramble the samples (tweets)")
arghandler.register_flag('rev', _arg_callback_reverse, ['reverse'], "reverese the samples (tweets)")
arghandler.register_flag('ls', _arg_callback_ls, ['limit', 'limit-samples'], "Limit how many samples to use (tweets) Args: <sample count>")
arghandler.register_flag('re', _arg_callback_re, ['random-embeddings'], "Use random embeddings")
arghandler.register_flag('ds', _arg_callback_ds, ['select-dataset', 'dataset'], "Which dataset to use. Args: <dataset-name>")
arghandler.register_flag('sp', _arg_callback_sp, ['set-partition'], "Set the partition sizes.")
arghandler.register_flag('sb', _arg_callback_sb, ['set-balance'], "Choose the set-balance. Args: <set-balance>")
arghandler.register_flag('le', _arg_callback_le, ['limit-embeddings'], "Limits how many embeddings are fitted. Args: <embedding count>")
arghandler.consume_flags()

logger = common_funs.Logger()
# the nltk casual toeknizer, reduce_len keeps repeating chars to 3 max
tknzr = TweetTokenizer(reduce_len=True, preserve_case=False)
# json files will be written all in one row without indentation unless..
j_indent = 4
#t_table = dict( ( ord(char), None) for char in string.punctuation ) #translation tabler  for puctuation
t_table = dict( ( ord(char), None) for char in ['.','_'] ) #translation tabler  for puctuation
# If you don't have the packages installed..
if cfg.nltk_dowload: nltk.download("stopwords"); nltk.download("punkt")


# get data and applay limit and set balance
negative_rows = Open_Dataset(cfg.dataset_name, 'cleaned', 'r', sample_class = 0).getRows()
positive_rows =  Open_Dataset(cfg.dataset_name, 'cleaned', 'r', sample_class = 1).getRows()

pos_samples_count = len(positive_rows)
neg_samples_count= len(negative_rows)
samples_count = pos_samples_count + neg_samples_count

print("Sample files found, total: {}, positive: {}, negative: {}"
	.format(samples_count, pos_samples_count, neg_samples_count))

if not cfg.limit_samples is None:
	pos_samples_count = cfg.limit_samples if cfg.limit_samples < pos_samples_count  else pos_samples_count
	neg_samples_count = cfg.limit_samples if cfg.limit_samples < neg_samples_count else neg_samples_count

if not cfg.set_balance is None:
	pos_samples_count, neg_samples_count = balance(pos_samples_count, neg_samples_count, cfg.set_balance)

# limit count
positive_rows = positive_rows[:pos_samples_count]
negative_rows = negative_rows[:neg_samples_count]

# actual count
pos_samples_count = len(positive_rows)
neg_samples_count = len(negative_rows)
samples_count = neg_samples_count + pos_samples_count

print("After set limiting, total: {}, positive: {}, negative: {}\n"
	.format(samples_count, pos_samples_count,neg_samples_count))


# tokenize samples
# samples are changed from litesql.row to list of dict
all_words = []
sequence_lengths = []
neg_samples = tokenize_helper(negative_rows, all_words, False)
pos_samples = tokenize_helper(positive_rows, all_words, True)


# build vocabulary
print("Building vocabulary..")
vocab = build_vocabulary(all_words, cfg.vocabulary_size)
unique_words, vocab_instances, vocabulary, rev_vocabulary = vocab

#load and fit embeddings to vocabulary
if cfg.use_embeddings:
	print("Fitting embeddings to vocabulary...")
	embeddings = fit_embeddings(vocabulary, cfg.raw_embeddings_path)

# print word stats
print()
print("Calculating vocabulary statistics...")
seq_max = max( sequence_lengths )
seq_mean = round( np.mean( sequence_lengths ), 2 )
seq_std = round( np.std( sequence_lengths ), 2 )
print("Longest sqeuence (words): " + str( seq_max) , end =", ")
print("mean: " + str( seq_mean), end =", ")
print("std: " + str( seq_std), end =", ")
print("3-sigma: " + str(math.ceil( seq_mean + 3 * seq_std) ) )
print("Words in corpus: {:0}, Unique words in corpus: {:1}" \
	.format( len(all_words), unique_words ) )
print("Vocabulary size: {:0}, Vocabulary coverage of corpus {:1.0%}" \
	.format(cfg.vocabulary_size, vocab_instances / len(all_words) ) ) 


# make index vectors
print ("Making index vectors..")
neg_samples = make_index_vectors(neg_samples, vocabulary)
pos_samples = make_index_vectors(pos_samples, vocabulary)
print("samples indexed, pos: {}, neg: {}".format(len(pos_samples), len(neg_samples)))


#mix the samples evenly
mixed_samples = interleave(pos_samples, neg_samples)


# modify samples
if cfg.random_data or cfg.add_snitch or cfg.random_labels:
	print("modifying samples..")
	mixed_samples = modify_samples(mixed_samples, 
								  cfg.random_data, 
								  cfg.add_snitch, 
								  cfg.random_labels)


#calculate how to partition samples in training, validation & test set
sample_count = len(mixed_samples)
training_index = math.floor(cfg.partition_training * sample_count)
validation_index = math.floor(cfg.partition_validation * sample_count) + training_index

# slice data into training, validation & test sets
train_samples = mixed_samples[:training_index]
validation_samples = mixed_samples[training_index:validation_index]
test_samples = mixed_samples[validation_index:]

#transpose the samples, change to numpy arrays, package in setpart
# package it all in a named touple
spt_train = transpose_setpart(train_samples, 'training samples')
spt_val = transpose_setpart(validation_samples, 'validation samples')
spt_test = transpose_setpart(test_samples, 'test samples')
ds = Dataset(spt_train, spt_val, spt_test)

# All processed data in one named touple
pd = ProcessedData(
	ds, 
	embeddings, 
	vocabulary, 
	rev_vocabulary, 
	cfg.embedding_size, 
	cfg.vocabulary_size, 
	cfg.max_sequence
)

print ("Samples partitioning; training: {}, validation: {}, test: {}"
	.format(ds.train.length, ds.valid.length, ds.test.length))



print ("Saving to disk at: {}".format(cfg.samples_path))
with open(cfg.samples_path, 'wb') as handle:
    pickle.dump(pd, handle, protocol=pickle.HIGHEST_PROTOCOL)

# saves readable versions of the data for debugging
if cfg.save_debug:
	debug_path = cfg.samples_path

	all_samples= json.dumps(mixed_samples, ensure_ascii=False, indent=j_indent, separators=( ',',': '))
	with open(debug_path + '.samples.json', 'w', encoding='utf8') as out_file:
		out_file.write(all_samples)	

	json_vocabulary= json.dumps(vocabulary, ensure_ascii=False, indent=j_indent, separators=( ',',': '))
	with open(debug_path + '.vocab.json', 'w', encoding='utf8') as out_file:
		out_file.write(json_vocabulary)

	json_rev_vocabulary= json.dumps(rev_vocabulary, ensure_ascii=False, indent=j_indent, separators=( ',',': '))
	with open(debug_path + '.rev_vocab.json', 'w', encoding='utf8') as out_file:
		out_file.write(json_rev_vocabulary)

	with open(debug_path + '.embeddings.json', 'w', encoding='utf8', newline='') as out_file:
		csv_w = csv.writer(out_file, delimiter=',')
		csv_w.writerows(embeddings)