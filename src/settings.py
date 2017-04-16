import os
import math
from collections import namedtuple

import numpy as np
################# settings ###############################################
##########################################################################

#used in preprocess
dataset_name = "poria-balanced" #"poria-ratio" # "poria-balanced", "imdb"
remove_punctuation = True
remove_stopwords = False
use_casual_tokenizer = True 	# doens't remove special chars
sample_count = 50000 # set to the smallest (36366) of the both classes to get an even nr of samples
partition_training = 0.7
partition_validation = 0.15
partition_test = 0.15
set_balance = 0.5 # proportion of sarcastic samples.
placeholder_char = '_' # placeholder char for words not in vocabulary
padding_char = '.'
padding_pos = "post" #pad at the start or at the end of the sample (pre/post)

embedding_size = 200 #allowed: 25, 50, 100, 200
vocabulary_size = 20000
max_sequence = 45 # words to include from sample, smaller samples will be padded

#mixed use
use_embeddings = True
ascii_console = False #set to true if your console doesn't handle unicode
use_logger = True

#used in training
network_name = 'little_pony'
run_count = 1
epochs = 3
batch_size = 187
snapshot_steps = math.floor(sample_count / (1 * batch_size)) # n = checkpoints per epoch

#For loading and saving models
save_the_model = False # If true, save the model to path specified in tflearn_rnn
pretrained_model = False # If true, create_model will initialize the model specified in pretrained_path
training = True # If false, the modeled will not be trained. Useful for testing pretrained model

# debug commands, will mess up the training: ##########################
random_labels = False # Used for debugging. If true will assign ranom labels (Ys) to samples.
add_snitch = False # adds a word to all positive and another to all negative samples
random_data = False # sets random training data
##########################################################################
##########################################################################

allowed_emb_sizes = [25,50,100,200]

#Specify path to pretrained model, if any
models_path = os.path.join("models")
if not (os.path.isdir(models_path)):
	os.makedirs(models_path)
pretrained_path = os.path.join(models_path, '6P8GFZ_FEVERISH_FOXBAT' + ".tfl")


# what data set to use
datsets = {
	"poria-balanced": {
		"rel_path": os.path.join(".","..", "datasets","poria", "en-balanced"),
		"neg_source": "balanced_normal_tweets.csv",
		"pos_source": "balanced_sarcastic_tweets.csv"
	},
	"poria-ratio": {
		"rel_path": os.path.join(".","..", "datasets","poria", "en-ratio"),
		"neg_source": "normal_tweets.csv",
		"pos_source": "sarcastic_tweets.csv"
	},
	"imdb" : {
		"rel_path": os.path.join(".","..", "datasets","imdb"),
		"neg_source": "",
		"pos_source": ""
	}
}

dataset = datsets[dataset_name]
rel_data_path = dataset["rel_path"]
dataset["neg_source"] = os.path.join(rel_data_path, dataset["neg_source"])
dataset["pos_source"] = os.path.join(rel_data_path, dataset["pos_source"])
path_name_neg = os.path.join(rel_data_path, "neg")
path_name_pos = os.path.join(rel_data_path, "pos")
samples_path = os.path.join(rel_data_path, "processed.pickle")

################# validate settings ############################

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


################## shared classes & objects ####################

ProcessedData = namedtuple('ProcessedData',
	['dataset', 'embeddings', 'vocab', 'rev_vocab', 'emb_size', 'vocab_size', 'max_sequence'])
Dataset = namedtuple('Dataset', ['train', 'valid', 'test'])
Setpart = namedtuple('Setpart', ['names', 'length', 'ids', 'xs','ys'])

pos_label = np.array([0., 1.], dtype="float32")
neg_label = np.array([1., 0.], dtype="float32")
