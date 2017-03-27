import os
import math

################# settings ###############################################
##########################################################################

dataset_name = "poria-balanced" #"poria-ratio" # "poria-balanced", "imdb"
remove_punctuation = True
remove_stopwords = False
use_casual_tokenizer = True 	# doens't remove special chars
sample_count = 50000 # set to the smallest (36366) of the both classes to get an even nr of samples

use_embeddings = True
placeholder_char = '_' # placeholder char for words not in vocabulary
padding_char = '.'
embedding_size = 25 #allowed: 25, 50, 100, 200 (OBS! 100+ will use 8GB+ RAM)
vocabulary_size = 20000
ascii_console = False #set to true if your console doesn't handle unicode
print_debug = False
use_logger = True

padding_pos = "post" #pad at the start or at the end of the sample (pre/post)
dropout = 0.5
epochs = 2
batch_size = 128
partition_training = 0.7
partition_validation = 0.15
partition_test = 0.15
set_balance = 0.5 # proportion of sarcastic samples.
max_sequence = 75 # words to include from sample, smaller samples will be padded
snapshot_steps = math.floor(sample_count / (1 * batch_size)) # n = checkpoints per epoch

# debug commands, will mess up the training: ##########################
random_labels = False # Used for debugging. If true will assign ranom labels (Ys) to samples.
add_snitch = False # adds a word to all positive and another to all negative samples
random_data = False # sets random training data
##########################################################################
##########################################################################

allowed_emb_sizes = [25,50,100,200]

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
samples_path = os.path.join(rel_data_path, "processed.json")
vocabulary_path = os.path.join(rel_data_path, "vocabulary.json")
rev_vocabulary_path = os.path.join(rel_data_path, "rev_vocabulary.json")
embeddings_path = os.path.join(rel_data_path, 'embeddings.json')
emb_voc_path = os.path.join(
		".", "..","datasets","glove_twitter_embeddings",
		"glove.twitter.27B." + str(embedding_size) + "d.txt")

################# validate settings ############################
if embedding_size not in allowed_emb_sizes:
	print("Wrong embedding size provided, quiting.")
	print("Allowed sizes: {0:s}, provided: {1:d}".format(
		','.join(map(lambda x: str(x), allowed_emb_sizes)), embedding_size))
	quit()
