import numpy as np

import pickle
from collections import Counter
import random
from scipy.stats import gmean

from common_funs import Progress_bar
from common_funs import Binary_confusion_matrix
from common_funs import Arg_handler
from common_funs import ProcessedData
from common_funs import Dataset
from common_funs import Setpart
from common_funs import pos_label
from common_funs import neg_label

from config import Config

def _arg_callback_in(file_name):
	cfg.ps_file_name = file_name
	print("<Using processed samples from: {}>".format(cfg.samples_path))

def _arg_callback_ds(ds_name):
	cfg.dataset_name = ds_name
	print("<Using dataset: {}>".format(ds_name))

def relative_frequency(ns, nu, ratio): 
	""" 
		returns -1 to +1, where +1 is most sarcastic, ratio is used to balance- 
		  out unbalanced datasets.
	"""
	return ((ns*ratio - nu) / (ns*ratio + nu))

def predict(samples, frequencies):
	predictions = []
	pb = Progress_bar(len(samples)-1)
	for i, words in enumerate(samples):
		
		if word == 0 or word == 1:
			continue

		if sum([frequencies[word] for word in words if word in frequencies]) > 0:
			label = pos_label
		else:
			label = neg_label		

		predictions.append(label)
		pb.tick()

	return predictions

##################################################################################

cfg = Config()
arghandler = Arg_handler()
arghandler.register_flag('ds', _arg_callback_ds, ['select-dataset', 'dataset'], "Which dataset to use. Args: <dataset-name>")
arghandler.register_flag('in', _arg_callback_in, ['input', 'in-file'], "Which file to take samples from. args: <filename>")
arghandler.consume_flags()
print("-" * 70)

# load processed samples
with open(cfg.samples_path, 'rb') as handle:
    ps = (pickle.load(handle)).dataset

# build list of words contained in samples from each class 
print("Separating words into each class..")
pos_words = []; neg_words = [];
for words, y in  zip(ps.train.xs, ps.train.ys):
	if np.array_equal(y, pos_label):
		pos_words.extend(words)
	else:
		neg_words.extend(words)
	
# count words in each class
print("Counting word occurrences for each class..")
pos_counts = Counter(pos_words)
neg_counts = Counter(neg_words)
ratio = len(neg_words) / len(pos_words)
print("Ratio between neutral and positive words: {}".format(ratio))

# make set of all words
all_words = set()
for word in list(pos_counts.keys()) + list(neg_counts.keys()):
	all_words.add(word)

# get the relative word frequency for every word
print("Calculating relative frequency for every words..")
frequencies = {}
for i, word in enumerate(all_words):
	frequencies[word] = relative_frequency(pos_counts[word], neg_counts[word], ratio)

# classify samples and print confusion matrix
print("Classifying samples..\n")
cm = Binary_confusion_matrix()

predictions = predict(ps.train.xs, frequencies)
cm.calc(ps.train.ids , predictions, ps.train.ys, 'training-set')

predictions = predict(ps.valid.xs, frequencies)
cm.calc(ps.valid.ids , predictions, ps.valid.ys, 'validation-set')

cm.print_tables()