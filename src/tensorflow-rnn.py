'''
===RNN Network in TensorFlow

* This is a Recurrent Neural Network for sarcasm detection
* Author: DATX02-25
'''

# Importing tensorflow
import tensorflow as tf
# Importing rnn framework
from tensorflow.contrib import rnn
# Importing Settings
import settings

# Loading Settings
vocabulary_size = settings.vocabulary_size #should match actual dictionary
embedding_size = settings.embedding_size
epochs = settings.epochs
batch_size = settings.batch_size
partition_training = settings.partition_training
partition_validation = settings.partition_validation
partition_test = settings.partition_test
set_balance = settings.set_balance
max_sequence = settings.max_sequence
ascii_console = settings.ascii_console

# debug commands, will mess up the training:
random_labels = settings.random_labels
add_snitch = settings.add_snitch
random_embeddings = not settings.use_embeddings

samples_path = settings.samples_path
vocabulary_path = settings.vocabulary_path
rev_vocabulary_path = settings.rev_vocabulary_path
embeddings_path = settings.embeddings_path
