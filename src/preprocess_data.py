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

from common_funs import ProcessedData
from common_funs import Dataset
from common_funs import Setpart
from common_funs import pos_label
from common_funs import neg_label

from config import Config

#### functions ###############################################################################

def _arg_callback_sp(train, cross, test):
	global partition_training, partition_validation, partition_test
	cfg.partition_training = float(train)
	cfg.partition_validation = float(cross)
	cfg.partition_test = float(test)

	if (cfg.partition_training + cfg.partition_validation + cfg.partition_test > 1):
		print("Sum of partitions cannot exceed 1.")
		sys.exit(0)

	#print("Train partition = {}, Evaluation partition = {:%}, Test partition = {}."
	#	.format(partition_training, partition_validation, partition_test))

def	_arg_callback_sb(set_balance):
	cfg.set_balance = float(set_balance)
	print("<Using set balance: {}>".format(set_balance))

def	_arg_callback_le(embeddings_count):
	cfg.embeddings_maxloop = int(embeddings_count)
	print("<Limiting embeddings: {}>".format(set_balance))

def _arg_callback_ds(ds_name):
	"""
	Select dataset
	"""
	cfg.dataset_name = ds_name
	print("<Using dataset: {}>".format(ds_name))

def _arg_callback_reverse():
	"""
	Reverse samples
	"""
	cfg.reverse_samples = True
	print("<Reversing samples..>")

def _arg_callback_scramble():
	"""
	Scramble the samples (tweets) to see if the network takes word order in account
	"""
	cfg.scramble_samples = True
	print("<Scrambling samples..>")

def _arg_callback_sd():
	"""
	save json files with vocabulary, samples etc. for debugging
	"""
	cfg.save_debug = True
	print("<Will save debug files..>")

def _arg_callback_nltk():
	"""
	Download nltk packages if missing
	"""
	cfg.nltk_dowload = True
	print("<Checking nltk..>")

def _arg_callback_ms():
	"""
	Create a minisample for debugging; will run quickly
	"""
	cfg.limit_samples = 1000
	cfg.embeddings_maxloop = 10000
	cfg.vocabulary_size = 5000
	cfg.embedding_size = 25
	cfg.max_sequence = 30

	print("<Using mini-sample>")

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
	pb = Progress_bar(len(d) - 1)
	vocabulary = OrderedDict( sorted(d.items(), key=lambda t: t[1],  reverse=True) )

	# start at 2 to leave room for padding & unknown
	for i, (key, value) in enumerate(vocabulary.items(), start=2):
		vocab_instances += value
		vocabulary[key] = i
		pb.tick()

	vocabulary[cfg.padding_char] = 0
	vocabulary[cfg.placeholder_char] = 1

	#reverse the vocbulary (for reverse lookup)
	rev_vocabulary = {v: k for k, v in vocabulary.items()}

	return len(unique_counts), vocab_instances, vocabulary, rev_vocabulary

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

def tokenize_helper(file_list, samples, all_words, sarcastic):
	file_count = len(file_list)
	print("Tokenizing %i %s samples (tweets)"
		%(file_count, "positive (sarcastic)" if sarcastic else "negative (normal)" ) )
	pb = common_funs.Progress_bar( file_count-1 )
	for file_name in file_list:
		text_tokens = tokenize_text( file_name['sample_text'] )
		all_words.extend(text_tokens)
		file_name = file_name['sample_id']
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
			if word in vocabulary:
				int_vector.append( vocabulary[word] )
			else:
				int_vector.append( 1 ) # 1 - used for masking samples

		if cfg.reverse_samples:
			int_vector.reverse()

		if cfg.scramble_samples:
			random.shuffle(int_vector)

		samples[key]['int_vector'] = int_vector


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

file_list_normal = Open_Dataset(cfg.dataset_name, 'cleaned', 'r', sample_class = 0).getRows()
file_list_sarcastic = Open_Dataset(cfg.dataset_name, 'cleaned', 'r', sample_class = 1).getRows()

pos_samples_count = len(file_list_sarcastic)
neg_samples_count= len(file_list_normal)
samples_count = pos_samples_count + neg_samples_count


print("Sample files found, total: {}, positive: {}, negative: {}"
	.format(samples_count, pos_samples_count, neg_samples_count))

if not cfg.limit_samples is None:
	pos_samples_count = cfg.limit_samples if cfg.limit_samples < pos_samples_count  else pos_samples_count
	neg_samples_count = cfg.limit_samples if cfg.limit_samples < neg_samples_count else neg_samples_count

if not cfg.set_balance is None:
	pos_samples_count, neg_samples_count = balance(pos_samples_count, neg_samples_count, cfg.set_balance)

# limit count
file_list_sarcastic = file_list_sarcastic[:pos_samples_count]
file_list_normal = file_list_normal[:neg_samples_count]

# actual count
pos_samples_count = len(file_list_sarcastic)
neg_samples_count = len(file_list_normal)

print("After set limiting, total: {}, positive: {}, negative: {}\n"
	.format(pos_samples_count +neg_samples_count, pos_samples_count,neg_samples_count))

samples = {}
all_words = []
sequence_lengths = []

tokenize_helper(file_list_normal, samples, all_words, False)
tokenize_helper(file_list_sarcastic, samples, all_words, True)

# build vocabulary
print("Building vocabulary..")
unique_words, vocab_instances, vocabulary, rev_vocabulary = \
	build_vocabulary(all_words, cfg.vocabulary_size)

#load and fit embeddings to vocabulary
if cfg.use_embeddings:
	print()
	print("Fitting embeddings to vocabulary...")
	embeddings = fit_embeddings(vocabulary, cfg.raw_embeddings_path)
	logger.log(getsizeof(embeddings), logname="embedding_bytes")

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
print()

# make index vectors
print ("Making index vectors..")
make_index_vectors( samples, vocabulary )
print( str( len(samples) ) + " samples indexed")

int_vectors = []
ids = []
labels = []
sample_count = len(samples)

# assign category labels
print("Assigning category labels...")
pb = Progress_bar(sample_count-1)
for i, (key, val) in enumerate(samples.items()):
	if cfg.random_labels:
		if random.randint(1,2) == 1:
			labels.append( pos_label )
		else:
			labels.append( neg_label )
	elif val['sarcastic'] == True:
		labels.append( pos_label )
		if cfg.add_snitch:
			val['int_vector'].extend( [cfg.vocabulary_size-1] )
	else:
		labels.append( neg_label )

	int_vectors.append( np.array( val['int_vector'], dtype="int32" ) )
	ids.append( key )
	pb.tick()


#zip the list shuffle them and unzip
#the seed should be kept the same so we
# always get the same shuffle
labeld_samples =  list( zip(ids, int_vectors, labels) )
#random.Random(1).shuffle( labeld_samples )
#ids, int_vectors, labels = zip(*labeld_samples)

#calculate  training and validation set size
sample_size = len( labeld_samples )
training_size = math.floor( cfg.partition_training * sample_size )
validation_size = math.floor( cfg.partition_validation * sample_size )

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

# use random data (random tweets)
if cfg.random_data:
	print("Using random data")
	tmp_X = []
	for _ in range( len(train_X) ):
		row = np.random.randint(
			1,
			(cfg.vocabulary_size - 1),
			size=cfg.max_sequence,
			dtype=np.int32)
		tmp_X.append(row)
	train_X = np.array(tmp_X, dtype=np.int32)

#pad the data
train_X = common_funs.pad_sequences(
	np.array( t_train_s[1] ),
	padding=cfg.padding_pos,
	maxlen=cfg.max_sequence,
	value=0.)
validate_X = common_funs.pad_sequences(
	np.array( t_validation_s[1] ),
	padding=cfg.padding_pos,
	maxlen=cfg.max_sequence,
	value=0.)
test_X = common_funs.pad_sequences(
	np.array( t_test_s[1] ),
	padding=cfg.padding_pos,
	maxlen=cfg.max_sequence,
	value=0.)

# package it all in a named touple
spt_train = Setpart('training set', len(train_ids), train_ids, train_X, train_Y)
spt_val = Setpart('validation set', len(validate_ids), validate_ids, validate_X, validate_Y)
spt_test = Setpart('test set', len(test_ids), test_ids, test_X, test_Y)
ds = Dataset(spt_train, spt_val, spt_test)
#logger.log(sample_size, "sample size", aslist=False)

for setpart in ds:
	for i in range(setpart.length - 1):
		name, _, ids, xs, ys = setpart
		line = "name: {} id:{} x:{} y:{}".format(name, ids[i], xs[i], ys[i])
		#logger.log(line, "processed", 1000, 5)

print ("Samples partitioning; training: {}, validation: {}, test: {}"
	.format(ds.train.length, ds.valid.length, ds.test.length))

# All processed data in one named touple
pd = ProcessedData(
	ds, embeddings, vocabulary, rev_vocabulary, cfg.embedding_size, cfg.vocabulary_size, cfg.max_sequence
)

print ("Saving to disk at: {}".format(cfg.samples_path))
with open(cfg.samples_path, 'wb') as handle:
    pickle.dump(pd, handle, protocol=pickle.HIGHEST_PROTOCOL)


# saves readable versions of the data for debugging
if cfg.save_debug:
	debug_path = os.path.join(cfg.dataset_path, "debug_")

	all_samples= json.dumps(samples, ensure_ascii=False, indent=j_indent, separators=( ',',': '))
	with open(debug_path + 'samples.json', 'w', encoding='utf8') as out_file:
		out_file.write(all_samples)

	json_vocabulary= json.dumps(vocabulary, ensure_ascii=False, indent=j_indent, separators=( ',',': '))
	with open(debug_path + 'vocab.json', 'w', encoding='utf8') as out_file:
		out_file.write(json_vocabulary)

	json_rev_vocabulary= json.dumps(rev_vocabulary, ensure_ascii=False, indent=j_indent, separators=( ',',': '))
	with open(debug_path + 'rev_vocab.json', 'w', encoding='utf8') as out_file:
		out_file.write(json_rev_vocabulary)

	with open(debug_path + 'embeddings.json', 'w', encoding='utf8', newline='') as out_file:
		csv_w = csv.writer(out_file, delimiter=',')
		csv_w.writerows(embeddings)

logger.save(file_name="preprocess.log")
