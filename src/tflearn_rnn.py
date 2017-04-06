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
from networks import Networks
from networks import NetworkNotFoundError

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
	If the average validation loss gets worse than the limit it throws an
	  exception, later caught in the trining loop.

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
			m = "Loss delta to limit: {}, continuing...".format(round(avg_limit-val_loss,3))
			self._buff.append([m])
			self.losses.append(val_loss)
		
		self._buff.flush()

def _arg_callback_train(nr_epochs=1, count=1, batchsize=30):
	global epochs, run_count, batch_size, training
	epochs = int(nr_epochs)
	run_count = int(count)
	batch_size = int(batchsize)
	training = True
	print("<Training for, epochs: {}, runs:{}, batchsize: {}>".format(nr_epochs, count, batchsize))

def _arg_callback_net(name):
	global network_name
	network_name = name
	print("<Using network: {}>".format(name))

def _arg_callback_in(file_name):
	"""
		Save preprocessed samples under a different file name
	"""
	global samples_path
	samples_path = os.path.join(rel_data_path, file_name)
	print("<Using processed samples from: {}>".format(samples_path))

def build_network(name, hyp, pd):

	params = {
		'hyp': hypers,
		'pd': pd,
	}

	nets = Networks()
	net = nets.get_network(name=name, params=params)

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
		emb = np.array(pd.embeddings[:pd.vocab_size], dtype=np.float32)
	else:
		emb = np.random.randn(pd.vocab_size, pd.emb_size).astype(np.float32)

	new_emb_t = tf.convert_to_tensor(emb)
	embeddings_tensor = tflearn.variables.get_layer_variables_by_name('embedding')[0]
	model.set_weights( embeddings_tensor, new_emb_t)
	w = model.get_weights(embeddings_tensor)
	debug_log.log(str(w.shape), "embedding layer shape", aslist=False)

	return model

def train_model(model, hyp, this_run_id, log_run):
	api = EarlyStoppingMonitor(avgOverNrEpochs = 3, avgLimitPercent = 1.05)
	monitorCallback = MonitorCallback(api)

	perflog.write([
		this_run_id,
		network_name,
		os.path.basename(samples_path),
		'-',
		'-',
		'Starting training...',
		]
	)
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
	perflog.replace([
		this_run_id,
		network_name,
		os.path.basename(samples_path),
		cm.metrics['validation-set']['accuracy'],
		cm.metrics['validation-set']['f1_score']
		]
	)

################################################################################

print_debug = True
# Handles command arguments, usefull for debugging 
# usage: tflearn_rnn.py --pf debug_processed.pickle
#  will get samples from debug_processed.pickle
arghandler = Arg_handler()
arghandler.register_flag('in', _arg_callback_in, ['input', 'in-file'], "Which file to take samples from. args: <filename>")
arghandler.register_flag('net', _arg_callback_net, ['network'], "Which network to use. args: <network name>")
arghandler.register_flag('train', _arg_callback_train, helptext = "Use settings for training. Args: <epochs> <run_count> <batch size>")
arghandler.consume_flags()

debug_log = Logger()
perflog = FileBackedCSVBuffer(
	"training_performance.csv",
	"logs",
	header=['Run id', 'Network name', 'data file', 'Val acc', 'Val f1', 'Status'])

# Load processed data from file
with open(samples_path, 'rb') as handle:
    pd = pickle.load( handle )
ps = pd.dataset #processed samples

# debug print tweets
if print_debug:	
	for s_id, s_y, s_x in zip(ps.train.ids, ps.train.ys, ps.train.xs):		
		ispos = np.array_equal(s_y, pos_label)
		label = "Positive (Sarcastic)" if ispos else "Negative (not sarcastic)"
		logstring = "Sample id: {}, {}: {:<5}".format(s_id, label, "\n")
		logstring += " ".join( reverse_lookup(s_x, pd.rev_vocab, ascii_console ))
		debug_log.log(logstring, logname="reverse_lookup", maxlogs = 10, step = 2500)
	print("\nLogged sample values:\n")
	debug_log.print_log(logname="reverse_lookup")
	print('', end='\n\n')


# 'sönderhaxad' class that generates random hyperparamters in the range provided

hypers = Hyper(run_count,
	lstm = {'dropout': (0.4, 0.8)},
	middle= {'weight_decay': (0.01, 0.06)},
	dropout = {'dropout': (0.4, 0.8)},
	regression = {'learning_rate': (0.0005, 0.0015)},
	output = {'weight_decay': (0.01, 0.06)}
)


# training loop, every loop trains a network with different hyperparameters
for hyp in hypers:
	log_run = Logger()
	this_run_id = common_funs.generate_name()
	log_run.log(hyp.get_hypers(), logname='hypers', aslist = False)
	log_run.log(this_run_id, logname='run_id', aslist = False)
	log_run.log(network_name, logname='network_name', aslist = False)

	tf.reset_default_graph()
	with tf.Graph().as_default(), tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		tflearn.config.init_training_mode()
		stop_reason = ["Other error"]

		try:
			net = build_network(network_name, hyp, pd)
			model = create_model(net, hyp, this_run_id, log_run)
			if training:
				model = train_model(model, hyp, this_run_id, log_run)		
		except NetworkNotFoundError as e:
			print("The network name provided din't match any defined network")
		except EarlyStoppingError as e:			
			stop_reason = ["Stopping due to early stopping"]
			do_prediction(model, hyp, this_run_id, log_run)
		else:
			stop_reason = ["Stopping due to epoch limit"]
			do_prediction(model, hyp, this_run_id, log_run)		
		finally:
			#do_prediction(model, hyp, this_run_id, log_run)
			perflog.append(stop_reason)			
			perflog.flush()
		
	log_run.save(this_run_id + '.log')
	
debug_log.save("training_debug.log")