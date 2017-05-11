import numpy as np

import pickle
from collections import Counter
import random
from scipy.stats import gmean
import time
from time import strftime
import json
import os

from common_funs import Progress_bar
from common_funs import Binary_confusion_matrix
from common_funs import Arg_handler
from common_funs import ProcessedData
from common_funs import Dataset
from common_funs import Setpart
from common_funs import pos_label
from common_funs import neg_label
from common_funs import DB_backed_log
from common_funs import generate_name
from common_funs import Bad_boys
from common_funs import file_selector

from config import Config

def _arg_callback_train(nr_epochs=1, count=1, batchsize=30):
		pass

def _arg_callback_net(name):
		pass

def _arg_callback_sm(name = None):
	cfg.save_the_model = True
	cfg.model_save_name = name

def _arg_callback_eval(pretrained_id = None):
	if pretrained_id is None:
		pretrained_id = file_selector(cfg.models_path, "Select model to evaluate")

	cfg.pretrained_id = pretrained_id
	cfg.training_mode = 'evaluate'
	print("<Evaluating pretrained model " + cfg.pretrained_id + " for results only.>")

def _arg_callback_st(gang_colors):
	cfg.trouble_gang = gang_colors
	cfg.trouble_type = 'save'
	print("<saving troublemakers after training>")

def _arg_callback_pt():
	cfg.print_test = True
	print("<Adding test set to prediction>")

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
		
		# sum up the frequencies over the sample ignoring padding and placeholders (0 and 1)
		if sum([frequencies[word] for word in words if word in frequencies and word != 0 and word != 1]) > 0:
			label = pos_label
		else:
			label = neg_label		

		predictions.append(label)
		pb.tick()

	return predictions

##################################################################################

cfg = Config()
cfg.trouble_type = None
cfg.trouble_gang = None
cfg.trouble_level = None
cfg.pretrained_id = None
cfg.training_mode = 'train'
cfg.save_the_model = False
cfg.model_save_name = None
arghandler = Arg_handler()
arghandler.register_flag('ds', _arg_callback_ds, ['select-dataset', 'dataset'], "Which dataset to use. Args: <dataset-name>")
arghandler.register_flag('in', _arg_callback_in, ['input', 'in-file'], "Which file to take samples from. args: <filename>")
arghandler.register_flag('pt', _arg_callback_pt, ['print-test'], "Produce results on test-partition of dataset.")
arghandler.register_flag('st', _arg_callback_st, ['save-trouble'], "save trouble maker prediction to DB under gang name. Args: <gang-name>")
arghandler.register_flag('eval', _arg_callback_eval, [], "Evaluate the network performance of a pre-trained model specified by run id. args: <run_id>")
arghandler.register_flag('sm', _arg_callback_sm, ['save-model'], "Save the trained model. Will be saved in dir with it's run id")
arghandler.register_flag('net', _arg_callback_net, [], "dummy flag")
arghandler.register_flag('train', _arg_callback_train, [], "dummy flag")
arghandler.consume_flags()
print("-" * 70)

this_run_id = 'RF_' + generate_name()

# get som gangtas' and log that ass
badboys = Bad_boys(cfg.sqlite_file, cfg.trouble_gang)
perflog = DB_backed_log(cfg.sqlite_file, 'training_performance')

# load processed samples
with open(cfg.samples_path, 'rb') as handle:
    pd = pickle.load(handle)
    ps = pd.dataset


if 'train' == cfg.training_mode:

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

elif 'evaluate' == cfg.training_mode:
	this_run_id = cfg.pretrained_id
	frequencies_path = os.path.join(cfg.models_path, cfg.pretrained_id, 'frequencies.json')
	print("Loading frequencies from: {}".format(frequencies_path))
	with open(frequencies_path, 'r', encoding='utf8') as f:		
		frequencies = json.load(f)
		# json want keys to be strings, convert back to int
		frequencies = {int(k): v for k,v in frequencies.items()}
else:
	raise Exception("Training mode not supported")


# classify samples and print confusion matrix
print("Classifying samples..\n")
cm = Binary_confusion_matrix()

train_predictions = predict(ps.train.xs, frequencies)
cm.calc(ps.train.ids , train_predictions, ps.train.ys, 'training-set')

predictions = predict(ps.valid.xs, frequencies)
cm.calc(ps.valid.ids , predictions, ps.valid.ys, 'validation-set')

# if writtern in the stars, log test set
if cfg.print_test:
	predictions = predict(ps.test.xs, frequencies)
	cm.calc(ps.test.ids , predictions, ps.test.ys, 'test-set')
	perflog.log(
		test_acc = cm.metrics['test-set']['accuracy'],
		test_f1 = cm.metrics['test-set']['f1_score']
)

cm.print_tables()

perflog.log(
	time = strftime('%Y-%m-%d %H:%M', time.localtime()),
	network = 'relative_frequency',
	dataset = cfg.dataset_name,
	samples_file = cfg.ps_file_name,
	val_acc = cm.metrics['validation-set']['accuracy'],
	val_f1 = cm.metrics['validation-set']['f1_score'],
	run_id = this_run_id
)

# let the bad boys run the streets
if cfg.trouble_type == 'save':
	print("Bad boys are runnning the sreets..")
	badboys.update(ps.train.ids , train_predictions, ps.train.ys)
	badboys.save()

if cfg.save_the_model:
	print("saving frequencies..")
	import shutil
	name = this_run_id if cfg.model_save_name is None else cfg.model_save_name
	model_folder = os.path.join(cfg.models_path, name)
	
	# delete folder if it exists
	if os.path.exists(model_folder):
		shutil.rmtree(model_folder)

	os.mkdir(model_folder)	
	frequencies_path = os.path.join(model_folder, 'frequencies.json')
	resolved_frequencies_path = os.path.join(model_folder, 'resolved_frequencies.txt')

	# keys need to be strings, sort by frequency for the resolved dict
	from collections import OrderedDict
	from operator import itemgetter
	sorted_frequncies = OrderedDict(sorted(frequencies.items(), key=itemgetter(1)))
	resolved_frequencies = [(pd.rev_vocab[k], v) for k,v in sorted_frequncies.items()]
	frequencies = {str(k): v for k,v in frequencies.items()}

	# save to file
	json_frequencies = json.dumps(frequencies, ensure_ascii=False, indent=4, separators=( ',',': '))
	with open(frequencies_path, 'w', encoding='utf8') as out_file:
		out_file.write(json_frequencies)	
	
	with open(resolved_frequencies_path, 'w', encoding='utf8') as out_file:
		for row in resolved_frequencies:
			out_file.write(str(row) + '\n')	

perflog.flush()