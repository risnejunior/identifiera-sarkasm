import os
import math
from collections import namedtuple

import numpy as np
################# settings ###############################################
##########################################################################

#mixed use
dataset_name = "detector" #"poria-ratio", "poria-balanced", "imdb", "detector"
use_embeddings = True
ascii_console = False #set to true if your console doesn't handle unicode
use_logger = True
allowed_emb_sizes = [25,50,100,200]
remove_punctuation = True
remove_stopwords = False
use_casual_tokenizer = True 	# doens't remove special chars
sample_count = 100000 # set to the smallest (27131) of the both classes to get an even nr of samples
partition_training = 0.7
partition_validation = 0.15
partition_test = 0.15
set_balance = None # proportion of sarcastic samples (0.0 - 0.1). Not used if None
placeholder_char = '_' # placeholder char for words not in vocabulary
padding_char = '.'
padding_pos = "post" #pad at the start or at the end of the sample (pre/post)

embedding_size = 200 #allowed: 25, 50, 100, 200
vocabulary_size = 20000
max_sequence = 75 # words to include from sample, smaller samples will be padded

#clean tweets
strict = False #option to chose strict cleansing of tweets or non-strict
includetags = True #option to include tags or not
tags = ["<user>", "<url>", "<hashtag>"]

#used in training
network_name = 'little_pony'
run_count = 1
epochs = 1
batch_size = 90
snapshot_steps = math.floor(sample_count / (1 * batch_size)) # n = checkpoints per epoch

#For loading and saving models
save_the_model = True # If true, save the model to path specified in tflearn_rnn
pretrained_model = False # If true, create_model will initialize the model specified in pretrained_path
training = True # If false, the modeled will not be trained. Useful for testing pretrained model

# debug commands, will mess up the training: ##########################
print_test = False
random_labels = False # Used for debugging. If true will assign ranom labels (Ys) to samples.
add_snitch = False # adds a word to all positive and another to all negative samples
random_data = False # sets random training data
##########################################################################
##########################################################################



#Specify path to pretrained model, if any
models_path = os.path.join("models")
if not (os.path.isdir(models_path)):
	os.makedirs(models_path)
pretrained_path = os.path.join(models_path, '6P8GFZ_FEVERISH_FOXBAT' + ".tfl")

# what data set to use
datasets = {
	"poria-balanced": {
		"rel_path": [".","..", "datasets","poria", "en-balanced"],
		"neg_source": "normal_tweets.csv",
		"pos_source": "sarcastic_tweets.csv",
		"ps_file_name": "processed.pickle",
		"source_format": {'sample_id':0, 'sample_text':2, 'unescape': False}
	},
	"poria-ratio": {
		"rel_path": [".","..", "datasets","poria", "en-ratio"],
		"neg_source": "normal_tweets.csv",
		"pos_source": "sarcastic_tweets.csv",
		"ps_file_name": "processed.pickle",
		"source_format": {'sample_id':0, 'sample_text':2, 'unescape': False}
	},
	"imdb" : {
		"rel_path": [".","..", "datasets","imdb"],
		"neg_source": "",
		"pos_source": "",
		"ps_file_name": "processed.pickle",
		"source_format": {'sample_id': 0, 'sample_text':1, 'unescape': False}
	},
	"detector" : {
		"rel_path": [".","..", "datasets","detector"],
		"neg_source": "normal_tweets.csv",
		"pos_source": "sarcastic_tweets.csv",
		"ps_file_name": "processed.pickle",
		"source_format": {'sample_id':None, 'sample_text':0, 'unescape': True}
	}
}

ProcessedData = namedtuple('ProcessedData',[
	'dataset',
	'embeddings',
	'vocab',
	'rev_vocab',
	'emb_size',
	'vocab_size',
	'max_sequence'])
Dataset = namedtuple('Dataset', ['train', 'valid', 'test'])
Setpart = namedtuple('Setpart', ['names', 'length', 'ids', 'xs','ys'])

pos_label = np.array([0., 1.], dtype="float32")
neg_label = np.array([1., 0.], dtype="float32")

def set_rel_paths(dataset_proto):
	dataset = {}
	dataset["rel_path"] = os.path.join(*dataset_proto["rel_path"])
	dataset["neg_source_path"] = os.path.join(dataset["rel_path"], dataset_proto["neg_source"])
	dataset["pos_source_path"] = os.path.join(dataset["rel_path"], dataset_proto["pos_source"])
	dataset["path_name_neg"] = os.path.join(dataset["rel_path"], "neg")
	dataset["path_name_pos"] = os.path.join(dataset["rel_path"], "pos")
	dataset["samples_path"] = os.path.join(dataset["rel_path"], dataset_proto['ps_file_name'])

	return dataset


def get_raw_embeddings_path(size):
	if size not in allowed_emb_sizes:
		print("Wrong embedding size provided, quiting.")
		print("Allowed sizes: {0:s}, provided: {1:d}".format(
			','.join(map(lambda x: str(x), allowed_emb_sizes)), embedding_size))
		quit()
	else:
		return os.path.join(
			".", "..","datasets","glove_twitter_embeddings",
			"glove.twitter.27B." + str(size) + "d.txt")

#############################################################
ds_paths = set_rel_paths(datasets[dataset_name])
rel_data_path = ds_paths["rel_path"]
path_neg = ds_paths["neg_source_path"]
path_pos = ds_paths["pos_source_path"]
path_name_neg = ds_paths["path_name_neg"]
path_name_pos = ds_paths["path_name_pos"]
samples_path = ds_paths["samples_path"]

sqlite_file = os.path.join(".","..", "datasets","datasets.sqlite")
