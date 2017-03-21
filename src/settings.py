import os

################# settings ###############################################
##########################################################################
dataset_name = "imdb" #poria-balanced, imdb
print_debug = False
remove_punctuation = True #unused atm
remove_stopwords = False
sample_count = 50000 #36366 #how many of both class3es of samples to use, to ensure they are 50/50
use_embeddings = True
placeholder_char = '_' # placeholder char for words not in dic
padding_char = '.'
embedding_size = 25 #allowed: 25, 50, 100, 200
vocabulary_size = 20000 #should match actual dictionary
epochs = 1
batch_size = 60
partition_training = 0.7 #0.7
partition_validation = 0.15
partition_test = 0.15
set_balance = 0.5 # proportion of sarcastic samples.
max_sequence = 30 #words to include from sample, smaller samples will be padded
ascii_console = False #set to true if your console doesn't handle unicode

# debug commands, will mess up the training: ##########################
random_labels = False # Used for debugging. If true will assign ranom labels (Ys) to samples.
add_snitch = False # adds a word to all positive and another to all negative samples
random_embeddings = False

##########################################################################
##########################################################################

allowed_emb_sizes = [25,50,100,200]

# what data set to use
datsets_paths = {
	"poria-balanced": os.path.join(".","..", "datasets","poria", "en-balanced"),
	"imdb" : os.path.join(".","..", "datasets","imdb")
}

rel_data_path = datsets_paths[dataset_name]
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