import json
from collections import Counter, OrderedDict
import math
import pickle
from pprint import pprint
import random

import numpy as np
from scipy.stats import gmean
from numpy import mean

import common_funs
import settings

class FubarException(Exception):
    pass

#####settings #########
max_len = None # None = take all
snitch = None # None = don't add
#######################

pos_label = np.array([0., 1.], dtype="float32")
neg_label = np.array([1., 0.], dtype="float32")
logger = common_funs.Logger(enable=settings.use_logger)

def predict(int_vectors, word_dicts, max_len=None):
	predictions = []
	for i, vector in enumerate(int_vectors):
		vector = vector[:max_len] if max_len != None else vector
		predictions.append(predict_helper(vector,  word_dicts))		
	return predictions
		
# sentence vector -> score
def predict_helper(sentence, word_dicts, print_debug = False):
	pos_freqs = []
	neg_freqs = []

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

		pos_freqs.append(freq_pos) #- freq_all)
		neg_freqs.append(freq_neg) #- freq_all)


	# if no words found of either class, chose at random, or the one with any
	if len (neg_freqs) == 0 and len (pos_freqs) == 0:
		rand = random.choice([[0, 1], [1, 0]])
		neg_freqs.append(rand[0])
		pos_freqs.append(rand[1])
	elif len (neg_freqs) == 0:
		neg_freqs.append(0)
	elif len (pos_freqs) == 0:
		pos_freqs.append(0)
	
	if 0 in pos_freqs:
		pos_mean = 0
	else:
		pos_mean = gmean(pos_freqs)
		#pos_mean = ((np.array(freq_all) - np.array(pos_freqs)) ** 2).mean(axis=None)
	
	if 0 in neg_freqs:
		neg_mean = 0
	else:
		neg_mean = gmean(neg_freqs) 
		#neg_mean = ((np.array(freq_all) - np.array(neg_freqs)) ** 2).mean(axis=None)

	if np.isnan(neg_mean) or np.isnan(pos_mean):
		raise FubarException("Result from Mean should never be NaN")

	#return  common_funs.normalize([neg_mean, pos_mean])
	return [neg_mean, pos_mean]

# load samples
print("Loading samples...")
with open('samples.pickle', 'rb') as handle:
    samples = pickle.load( handle )

train_X = samples["train_X"]
train_Y = samples["train_Y"]
test_X = samples["test_X"]
test_Y = samples["test_Y"]

all_words = []
pos_words = []
neg_words = []

print('grouping words...')
training_samples = zip(train_X, train_Y)
logger.log("Training set length: {:d}".format(len(train_X)))
pb = common_funs.Progress_bar(train_X.shape[0]-1)
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

print("running prediction...")

# print confusion matrix for the different sets
print("\n   TRAINING SET \n")
predictions = predict(train_X, word_dicts, max_len)
ids = [i for i in range(len(predictions))] #faske ids
common_funs.binary_confusion_matrix(ids , predictions, train_Y)

print("\n   TEST SET \n")
predictions = predict(test_X, word_dicts, max_len)
ids = [i for i in range(len(predictions))] #faske ids
common_funs.binary_confusion_matrix( ids, predictions, test_Y)

logger.save(file_name="frequencies.log")