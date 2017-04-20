import json
from collections import Counter, OrderedDict
import math
import pickle
from pprint import pprint
import random
import os

import numpy as np
from scipy.stats import gmean
from scipy.stats import hmean
from numpy import mean

import common_funs
from common_funs import Binary_confusion_matrix
from common_funs import Arg_handler
from settings import *

class FubarException(Exception):
    pass

#####settings #########
max_len = None # None = take all
snitch = False # None = don't add
#######################

pos_label = np.array([0., 1.], dtype="float32")
neg_label = np.array([1., 0.], dtype="float32")
logger = common_funs.Logger(enable=use_logger)

def predict(int_vectors, word_dicts, max_len=None):
	predictions = []
	sure_thing = 0
	for i, vector in enumerate(int_vectors):
		vector = vector[:max_len] if max_len != None else vector
		prediction = predict_helper(vector,  word_dicts)
		if 0 in prediction:
			sure_thing += 1
		predictions.append(prediction)
	logger.log(sure_thing, 'sure_things')
	return predictions

# sentence vector -> score
def predict_helper(sentence, word_dicts, print_debug = False):
	pos_freqs = []
	neg_freqs = []
	all_freqs = []

	for word in sentence:

		# skip padding and placeholders
		if word == 0 or word == 1:
			continue

		# if the word wasn't in the training samples, skip it
		if word not in word_dicts[0]:
			continue

		freq_all = word_dicts[0][word]
		freq_neg = word_dicts[1][word] if word in word_dicts[1] else 0
		freq_pos = word_dicts[2][word] if word in word_dicts[2] else 0

		all_freqs.append(freq_all) #- freq_all)
		pos_freqs.append(freq_pos) #- freq_all)
		neg_freqs.append(freq_neg) #- freq_all)


	#logger.log(neg_loss, 'neg_loss', 30)
	#logger.log(pos_loss, 'pos_loss', 30)

	#return [(1 - neg_loss), (1 - pos_loss)]

	# if no words found of either class, chose at random, or the one with any
	if len (neg_freqs) == 0 and len (pos_freqs) == 0:
		rand = random.choice([[0, 1], [1, 0]])
		neg_freqs.append(rand[0])
		pos_freqs.append(rand[1])
	elif len (neg_freqs) == 0:
		neg_freqs.append(0)
	elif len (pos_freqs) == 0:
		pos_freqs.append(0)

	#rand = random.choice([[0, 1], [1, 0]])
	if 0 in pos_freqs:
		pos_mean = 0
	else:
		pos_mean = gmean(pos_freqs)
		#pos_mean = rand[1]
		#pos_mean = common_funs.squared_error(neg_freqs, pos_freqs)
		#pos_mean = ((np.array(freq_all) - np.array(pos_freqs)) ** 2).mean(axis=None)

	if 0 in neg_freqs:
		neg_mean = 0
	else:
		neg_mean = gmean(neg_freqs)
		#neg_mean = rand[0]
		#neg_mean = common_funs.squared_error(pos_freqs, neg_freqs)
		#neg_mean = ((np.array(freq_all) - np.array(neg_freqs)) ** 2).mean(axis=None)

	if np.isnan(neg_mean) or np.isnan(pos_mean):
		raise FubarException("Result from Mean should never be NaN")

	#return  common_funs.normalize([neg_mean, pos_mean])
	return [neg_mean, pos_mean]

def _arg_callback_in(file_name):
	"""
	Take preprocessed samples from the selected file
	"""
	global samples_path
	samples_path = os.path.join(rel_data_path, file_name)
	print("<Using processed samples from: {}>".format(samples_path))

##################################################################################

arghandler = Arg_handler()
arghandler.register_flag('in', _arg_callback_in, ['input', 'in-file'], "Which file to take samples from. args: <filename>")
arghandler.consume_flags()

# load processed samples
with open(samples_path, 'rb') as handle:
    ps = (pickle.load(handle)).dataset

all_words = []
pos_words = []
neg_words = []

print('Grouping words...')
training_samples = zip(ps.train.xs, ps.train.ys)
logger.log("Training set length: {:d}".format(len(ps.train.xs)))
pb = common_funs.Progress_bar(ps.train.xs.shape[0] - 1)
pos = neg = 0
for sentence, label in training_samples:
	all_words.extend(sentence)
	is_pos = np.array_equal(label,pos_label)

	if (is_pos):
		if snitch: sentence = np.append(sentence, snitch)

		pos_words.extend(sentence)
		pos += 1
	else:
		neg_words.extend(sentence)
		neg += 1
	pb.tick()
	#logger.log(np.array_str(sentence, max_line_width = 1000),
	#	       logname="sentences", step=10000, maxlogs=5)

logger.log ("pos samples {:d}".format(pos))
logger.log ("neg samples {:d}".format(neg))

print('Counting and sorting word frequencies...')
logger.log("Total words: {:d}".format(len(all_words)))
word_dicts = OrderedDict({0: {}, 1:{}, 2:{}})
for i_dict, words in enumerate([all_words, neg_words, pos_words]):
	c = Counter(words)
	largest = c.most_common(1)[0][1]
	logger.log(largest, logname="most common")
	d = dict(c)
	d = OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True) )
	logger.log("dictionery {:d} length: {:d}".format(i_dict, len(d)))
	for i, (word, count) in enumerate(d.items()):
		d[word] = count/len(words) #frequency
		#d[word] = count/largest #normalization

		logger.log("word: {}, freq: {}".format(word, d[word]), 
				   logname="freqs",
				   step=10000)
	word_dicts[i_dict] = d

print("Running prediction...\n")
cm = Binary_confusion_matrix()

# print confusion matrix for the different sets
predictions = predict(ps.train.xs, word_dicts, max_len)
cm.calc(ps.train.ids , predictions, ps.train.ys, 'training-set')

predictions = predict(ps.valid.xs, word_dicts, max_len)
cm.calc(ps.valid.ids , predictions, ps.valid.ys, 'validation-set')

cm.print_tables()
cm.save(content='metrics')
cm.save(content='table')
logger.save(file_name="frequencies.log")
