# -*- coding: utf-8 -*-
import collections
import math
import random
import os
import json
import pickle
import csv
import sys

import tflearn
import numpy as np
import tensorflow as tf
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

import common_funs
from common_funs import Binary_confusion_matrix
from common_funs import Logger
from common_funs import reverse_lookup
from common_funs import Hyper
from common_funs import Arg_handler
from common_funs import FileBackedCSVBuffer
from settings import *

class EarlyStoppingError(StopIteration):
	pass

class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, api):
        self.monitor_api = api

    def on_epoch_end(self, training_state):
        self.monitor_api.send({
            'val_accuracy': training_state.val_acc,
            'val_loss': training_state.val_loss,
            'best_accuracy': training_state.best_accuracy,
            'accuracy': training_state.acc_value,
        })

class EarlyStoppingMonitor():
	"""
	Monitors the on_epoch_end callback and checks if validation loss is going up.
	If the validation loss gets worse over as many epochs as given by
	  maxlosingstreak, then it will abort the training.

	Keyword Arguments:
	avgOverNrEpochs -- how many epochs to average over
	avgLimitPercent -- percentage of the average that sets the limit
	"""

	def __init__(self, avgOverNrEpochs = 3, avgLimitPercent = 1.1):
		self.epoch = 0
		self.losses = []
		self.avgOverNrEpochs = avgOverNrEpochs
		self.avgLimitPercent = avgLimitPercent
		self._buff = FileBackedCSVBuffer(
			filename='earlystopping.csv',
			directory='logs', 
			header=['epoch', 'val loss', 'avg val loss', 'loss limit', 'status'],
			clearFile=True)

	def send(self, state):
		
		self.epoch += 1

		if state['val_loss']:
			val_loss = state['val_loss']
			val_loss = round(val_loss, 3)
		else:
			val_loss = 1

		if len(self.losses) < self.avgOverNrEpochs:
			avg_loss = val_loss
		else:
			avg_loss = sum(self.losses[-self.avgOverNrEpochs:]) / self.avgOverNrEpochs
			avg_loss = round(avg_loss, 3)

		avg_limit = self.avgLimitPercent * avg_loss

		self._buff.write([self.epoch, val_loss, avg_loss, avg_limit])
		
		if val_loss > avg_limit:
			self._buff.append(["Stopped due to loss average"])
			raise EarlyStoppingError("Early stopping due to loss average")
		else:
			m = "Loss delta to limit: {}, continuing...".format(avg_limit-val_loss)
			self._buff.append([m])
			self.losses.append(val_loss)
		
		self._buff.flush()

def _arg_callback_pf(file_name):
	"""
		Save preprocessed samples under a different file name
	"""
	global samples_path
	samples_path = os.path.join(rel_data_path, file_name)
	print()
	print("Using processed samples from: {}".format(samples_path))

def build_network(hyp):
	net = tflearn.input_data([None, max_sequence], dtype=tf.float32)
	net = tflearn.embedding(net, input_dim=vocabulary_size,
							     output_dim=embedding_size,
							     name="embedding",
							     restore=False)

	net = tflearn.lstm(net,
					   64,
					   dropout=hyp.lstm.dropout,
					   dynamic=True,
					   name="lstm",
					   restore=False)

	"""
	net = bidirectional_rnn(net,
							BasicLSTMCell(96),
							BasicLSTMCell(96),
					        dynamic=True)
	"""
	net = tflearn.fully_connected(net,
								  64,
								  activation='sigmoid',
								  regularizer='L2',
								  weight_decay=hyp.middle.weight_decay,
								  name="middle",
								  restore=False)

	net = tflearn.dropout(net, hyp.dropout.dropout, name="dropout")
	net = tflearn.fully_connected(net,
								  2,
								  activation='softmax',
								  regularizer='L2',
								  weight_decay=hyp.output.weight_decay,
								  name="output",
								  restore=True)
	net = tflearn.regression(net,
		                     optimizer='adam',
		                     learning_rate=hyp.regression.learning_rate,
	                         loss='categorical_crossentropy')
	return net

def create_model(net, hyp, this_run_id, log_run):
	checkpoint_path = os.path.join("checkpoints")
	if not (os.path.isdir(checkpoint_path)):
		os.makedirs(checkpoint_path)
	checkpoint_path = os.path.join("checkpoints",this_run_id + ".ckpt")
	model = tflearn.DNN(net,
					    tensorboard_verbose=3,
					    checkpoint_path=checkpoint_path)

	#Load pretrained model
	if pretrained_model:
		print("Attempting to load model")
		model.load(pretrained_path)
		print("Successfully loaded model")
		return model
	
	#set embeddings
	if use_embeddings:
		emb = np.array(embeddings[:vocabulary_size], dtype=np.float32)
	else:
		emb = np.random.randn(vocabulary_size, embedding_size).astype(np.float32)

	new_emb_t = tf.convert_to_tensor(emb)
	embeddings_tensor = tflearn.variables.get_layer_variables_by_name('embedding')[0]
	model.set_weights( embeddings_tensor, new_emb_t)
	w = model.get_weights(embeddings_tensor)
	debug_log.log(str(w.shape), "embedding layer shape", aslist=False)

	return model

def train_model(model, hyp, this_run_id, log_run):
	api = EarlyStoppingMonitor(avgOverNrEpochs = 3, avgLimitPercent = 1.05)
	monitorCallback = MonitorCallback(api)

	model.fit(X_inputs=ps.train.xs,
			  Y_targets=ps.train.ys,
			  validation_set=(ps.valid.xs, ps.valid.ys),
			  show_metric=True,
	          batch_size=batch_size,
	          n_epoch=epochs,
	          shuffle=False,
	          run_id=this_run_id,
			  snapshot_step=snapshot_steps,
			  snapshot_epoch=True,
	          callbacks=monitorCallback)

	# save model
	if save_the_model:
		models_path = os.path.join("models")
		if not (os.path.isdir(models_path)):
			os.makedirs(models_path)

		model_file_path = os.path.join(models_path,this_run_id + ".tfl")
		model.save(model_file_path)

	return model

def do_prediction(model, hyp, this_run_id, log_run):
	# print confusion matrix for the different sets
	print("running prediction...\n")
	cm = Binary_confusion_matrix()
	horiz_bar = "-" * (len(this_run_id) + 9 )
	print(horiz_bar)
	print("runid: " + this_run_id + ' |')
	print(horiz_bar)

	predictions = model.predict(ps.train.xs)
	cm.calc(ps.train.ids , predictions, ps.train.ys, 'training-set')

	predictions = model.predict(ps.valid.xs)
	cm.calc(ps.valid.ids , predictions, ps.valid.ys, 'validation-set')

	cm.print_tables()
	#cm.save(this_run_id + '.res', content='metrics')
	log_run.log(cm.metrics, logname="metrics", aslist = False)
	perflog.write([
		this_run_id,
		cm.metrics['validation-set']['accuracy'],
		cm.metrics['validation-set']['f1_score']
		]
	)

################################################################################

# Handles command arguments, usefull for debugging 
# usage: tflearn_rnn.py --pf debug_processed.pickle
#  will get samples from debug_processed.pickle
arghandler = Arg_handler()
arghandler.register_flag('pf', _arg_callback_pf, ['processed-file'], "Which file to take samples from")
arghandler.consume_flags()

run_count = 2
debug_log = Logger()
perflog = FileBackedCSVBuffer(
	"training_performance.csv", "logs", header=['Run id', 'Val acc', 'Val f1'])


# reverse dictionary
with open( rev_vocabulary_path, 'r', encoding='utf8' ) as rev_vocab_file:
	rev_vocabulary = json.load( rev_vocab_file )

# embeddings
embeddings = []
with open(embeddings_path, 'r', encoding='utf8', newline='') as in_file:
	csv_r = csv.reader(in_file, delimiter=',')
	for row in csv_r:
		embeddings.append(row)

# processed samples
with open(samples_path, 'rb') as handle:
    ps = pickle.load( handle )

# debug print tweets
if print_debug:
	for s_id, s_y, s_x in zip(ps.train.ids, ps.train.ys, ps.train.xs):
		ispos = np.array_equal(s_y, pos_label)
		label = "Positive (Sarcastic)" if ispos else "Negative (not sarcastic)"
		logstring = "Sample id: {}, {}: {:<5}".format(s_id, label, "\n")
		logstring += " ".join( reverse_lookup(s_x, rev_vocabulary, ascii_console ))
		debug_log.log(logstring, logname="reverse_lookup", maxlogs = 5, step = 2500)
	print("\nLogged sample values:\n")
	debug_log.print_log(logname="reverse_lookup")
	print('', end='\n\n')


# 'sönderhaxad' class that generates random hyperparamters in the range provided

hypers = Hyper(run_count,
	lstm = {'dropout': (0.4, 0.8)},
	middle= {'weight_decay': (0.01, 0.03)},
	dropout = {'dropout': (0.4, 0.8)},
	regression = {'learning_rate': (0.0005, 0.0015)},
	output = {'weight_decay': (0.01, 0.03)}
)

# training loop, every llop
for hyp in hypers:
	log_run = Logger()
	this_run_id = common_funs.generate_name()
	log_run.log(hyp.get_hypers(), logname='hypers', aslist = False)
	log_run.log(this_run_id, logname='run_id', aslist = False)

	tf.reset_default_graph()
	with tf.Graph().as_default(), tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		tflearn.config.init_training_mode()
		stop_reason = ["Other error"]

		try:
			net = build_network(hyp)
			model = create_model(net, hyp, this_run_id, log_run)
			if training:
				model = train_model(model, hyp, this_run_id, log_run)			
		except EarlyStoppingError as e:
			print(e)
			stop_reason = ["Stopping due to early stopping"]
		else:
			stop_reason = ["Stopping due to epoch limit"]			
		finally:
			do_prediction(model, hyp, this_run_id, log_run)
			perflog.append(stop_reason)			
			perflog.flush()
		
	log_run.save(this_run_id + '.log')
	
debug_log.save("training_debug.log")