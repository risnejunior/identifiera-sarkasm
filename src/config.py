import os
import math

class Config:
	"""
	You can use your own settings from a file, by default it loads settings.json
	"""
	standard_settings = dict(
		#mixed use
		dataset_name = "detector", #"poria-ratio", "poria-balanced", "imdb", "detector"
		ps_file_name = 'processed.pickle',
		use_embeddings = True,
		ascii_console = False, #set to true if your console doesn't handle unicode
		use_logger = True,
		remove_punctuation = True,
		remove_stopwords = False,
		use_casual_tokenizer = True, 	# doens't remove special chars
		sample_count = None, 
		partition_training = 0.7,
		partition_validation = 0.15,
		partition_test = 0.15,

		limit_samples = None, # [int] if you want to limit the samples from each class
		set_balance = None, # [float] proportion of sarcastic samples, Not used if None
		placeholder_char = '_', # placeholder char for words not in vocabulary
		padding_char = '.',
		padding_pos = "post", #pad at the start or at the end of the sample (pre/post)

		embedding_size = 200, #allowed: 25, 50, 100, 200
		vocabulary_size = 20000,
		max_sequence = 75, # words to include from sample, smaller samples will be padded

		#clean tweets
		strict = False, #option to chose strict cleansing of tweets or non-strict
		includetags = False, #option to include tags or not
		
		#used in training
		network_name = 'little_pony',
		run_count = 1, # how many hyperparamter permutations to iterate over
		epochs = 1,
		batch_size = 90,
		snapshots_per_epoch = 1, #checkpoints per epoch

		#For loading and saving models
		save_the_model = False, # If true, save the model to path specified in tflearn_rnn
		pretrained_id = None, # if none then you will select the model from an interactive menu
		training_mode = 'training', #training, boost, evaluate

		# debug commands
		print_test = False,
		random_labels = False, # Used for debugging. If true will assign ranom labels (Ys) to samples.
		add_snitch = False, # adds a word to all positive and another to all negative samples
		random_data = False, # sets random training data
	)
	
	# constants
	tags = ["<user>", "<url>", "<hashtag>"]
	allowed_emb_sizes = [25,50,100,200]
	ensure_paths = {
		'models_path':'models', 
		'logs_path':'logs', 
	}

	datasets = {
		"poria-balanced": {
			"rel_path": ["poria", "en-balanced"],
			"neg_source": "normal_tweets.csv",
			"pos_source": "sarcastic_tweets.csv",
			"source_format": {'sample_id':0, 'sample_text':2, 'unescape': False}
		},
		"poria-ratio": {
			"rel_path": ["poria", "en-ratio"],
			"neg_source": "normal_tweets.csv",
			"pos_source": "sarcastic_tweets.csv",
			"source_format": {'sample_id':0, 'sample_text':2, 'unescape': False}
		},
		"imdb" : {
			"rel_path": ["imdb"],
			"neg_source": "",
			"pos_source": "",
			"source_format": {'sample_id': 0, 'sample_text':1, 'unescape': False}
		},
		"detector" : {
			"rel_path": ["detector"],
			"neg_source": "normal_tweets.csv",
			"pos_source": "sarcastic_tweets.csv",
			"source_format": {'sample_id':None, 'sample_text':0, 'unescape': True}
		}
	}

	def __init__(self, own_settings = 'settings.json'):
		self._root_datasets_path = os.path.join(".", "..","datasets")

		# set default settings
		self.__dict__.update(Config.standard_settings)		

		# make sure these paths exist		
		self.__dict__.update(Config.ensure_paths)		
		for path_name, path in Config.ensure_paths.items():
			if not (os.path.isdir(path)):
				os.makedirs(path)
	
		# load own config file
		if os.path.exists(own_settings):
			import json
			settings_file = open(own_settings, 'r', encoding='utf8')
			self.__dict__.update(json.load(settings_file))

	def get_raw_embeddings_path(self):
		if self.embedding_size not in Config.allowed_emb_sizes:
			raise ValueException("Wrong embedding size provided, allowed {0:s}"
				.format(Config.allowed_emb_sizes))
		else:
			return os.path.join(
				self._root_datasets_path,"glove_twitter_embeddings",
				"glove.twitter.27B." + str(self.embedding_size) + "d.txt")

	# dynmaic settings; getters and setters
	def get_sqlite_file_path(self):
		return os.path.join(self._root_datasets_path, "datasets.sqlite")

	def get_dataset_path(self):
		return os.path.join(self._root_datasets_path, 
							*Config.datasets[self.dataset_name]['rel_path'])

	def get_neg_source_path(self):
		return os.path.join(self.dataset_path, 
			                Config.datasets[self.dataset_name]['neg_source'])

	def get_pos_source_path(self):
		return os.path.join(self.dataset_path, 
			                Config.datasets[self.dataset_name]['pos_source'])

	def get_processed_path(self):
		path = os.path.join(self.dataset_path, 'processed')
		if not (os.path.isdir(path)):
				os.makedirs(path)
		
		return path

	def get_ps_path(self):
		return os.path.join(self.processed_path, self.ps_file_name)

	def get_source_format(self):
		return Config.datasets[self.dataset_name]['source_format']

	# def get_snapshot_steps(self):
		# return math.floor(self.sample_count / (self.snapshots_per_epoch * self.batch_size))

	# def get_pretrained_file_path(self):
	# 	from os import listdir
	# 	pretrained_files = [file.split('.')[0] for file in listdir(self.pretrained_path)]

	# 	if pretrained_files:
	# 		best_file = sorted(pretrained_files, reverse=True)[0]
	# 		return os.path.join(self.pretrained_path, best_file)
	# 	else:
	# 		raise Exception("path empty")


	# def get_pretrained_path(self):
	# 	return os.path.join(self.models_path, self.pretrained_id)

	# dynamic paths
	raw_embeddings_path = property(get_raw_embeddings_path)
	sqlite_file = property(get_sqlite_file_path)
	dataset_path = property(get_dataset_path)
	neg_source_path = property(get_neg_source_path)
	pos_source_path = property(get_pos_source_path)
	processed_path = property(get_processed_path)
	samples_path = property(get_ps_path)
	source_format = property(get_source_format)