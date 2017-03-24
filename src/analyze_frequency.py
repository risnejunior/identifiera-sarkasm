import json
from collections import Counter, OrderedDict
import math
import pickle
from pprint import pprint

import numpy as np
from scipy.stats import gmean

import common_funs
import settings

#####settings #########
max_len = None
#######################

pos_label = np.array([0., 1.], dtype="float32")
neg_label = np.array([1., 0.], dtype="float32")

def normalize(xs): 
	min_xs = min(xs)
	max_xs = max(xs)
	ys = []
	
	if min_xs == max_xs:
		print("max equals min, not possible")
		print(xs)
		quit()
	

	for x in xs:
		y = ( x - min_xs ) / ( max_xs - min_xs )
		ys.append(y)

	return ys

def predict(int_vectors, word_dicts, max_len=9999):
	predictions = []
	for vector in int_vectors:
		predictions.append(predict_helper(vector[:max_len],  word_dicts))
	return predictions
		
# sentence vector -> score
def predict_helper(sentence, word_dicts, print_debug = False):

	pos_score_v = []
	neg_score_v = []

	for word in sentence:		
		# neg score (normal)
		if word == 0:
			continue

		if word in word_dicts[1]:
			neg_score_v.append(word_dicts[1][word])
		else:
			#neg_score_v.append(0)
			pass
		# pos score
		if word in word_dicts[2]:
			pos_score_v.append(word_dicts[2][word])
		else:
			#pos_score_v.append(0)
			pass

	pos_mean = gmean(pos_score_v) 
	neg_mean = gmean(neg_score_v)

	return  normalize([neg_mean, pos_mean])

# load pickled data
print("Loading pickles...")
with open('train_X.pickle', 'rb') as handle:
    train_X = pickle.load( handle )
with open('train_Y.pickle', 'rb') as handle:
    train_Y = pickle.load( handle )
with open('test_X.pickle', 'rb') as handle:
    test_X = pickle.load( handle )
with open('test_Y.pickle', 'rb') as handle:
    test_Y = pickle.load( handle )

"""
print("loading samples...")
with open( settings.samples_path, 'r', encoding='utf8' ) as samples_file:
	samples = json.load( samples_file )

sample_count = len(samples)
"""

all_words = []
pos_words = []
neg_words = []

print('grouping words...')
samples = zip(train_X, train_Y)
print("Test set length: {:d}".format(len(train_X)))
pos = neg = 0
for sentence, label in samples:
	#padding_from = next(x[0] for x in enumerate(sentence) if x[1] == 0)
	all_words.extend(sentence)	
	if (np.array_equal(label,pos_label)):
		pos_words.extend(sentence)
		pos += 1
	else:
		neg_words.extend(sentence)
		neg += 1

print ("pos samples {:d}".format(pos))
print ("neg samples {:d}".format(neg))

"""
pb = common_funs.Progress_bar(sample_count)
for i,(key, val) in enumerate(samples.items()):
	
	all_words.extend(val["int_vector"])

	if val["sarcastic"]:
		pos_words.extend(val["int_vector"])
		pos_corp.append(val["int_vector"])
		pos_ids.append(key)
		pos_labels.append(pos_label)
	else:
		neg_words.extend(val["int_vector"])
		neg_corp.append(val["int_vector"])
		neg_ids.append(key)
		neg_labels.append(neg_label)
	pb.tick()
"""

print('Counting and sorting word frequencies...')
tot_len = len(all_words) + len(neg_words) + len(pos_words)
print("tot len: {:d}".format(tot_len))
word_dicts = OrderedDict({0: {}, 1:{}, 2:{}})
pb = common_funs.Progress_bar(tot_len)
for i_dict, words in enumerate([all_words, neg_words, pos_words]):
	print()
	d = dict(Counter(words))
	d = OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True) )
	print("counted length: {:d}".format(len(d)))
	for i, (word, count) in enumerate(d.items()):				
		#print("word: {}, count: {}".format(word, d[word]))
		d[word] = count/len(words)
		#print("word: {}, freq: {}".format(word, d[word]))

		continue

		"""
		#no need to compare all words to itself
		if id(all_words) == id(words):
			#print("all_words")
			d[word] = count/len(words)			
		else:
			#print("other words")
			d[word] = count/len(words) #- all_words[word]	
		#pb.tick()
		"""
	word_dicts[i_dict] = d

"""
print("len word dict 0 {:d}".format(len(word_dicts[0])))
quit()
for i, (word, freq) in enumerate(word_dicts[0].items()):
	print(i, end=" > ")
	print(word, end=" : ")
	print(freq)
quit()

"""



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
