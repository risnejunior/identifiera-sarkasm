# -*- coding: utf-8 -*-
import sys
import collections
import math
import random
import os
import json
import pickle
import csv
import time
from time import strftime
import tempfile
from os import listdir
from operator import itemgetter

import common_funs
from common_funs import Binary_confusion_matrix
from common_funs import Logger
from common_funs import reverse_lookup
from common_funs import Hyper
from common_funs import Arg_handler
from common_funs import FileBackedCSVBuffer
from common_funs import boxString
from common_funs import DB_backed_log
from common_funs import file_selector
from common_funs import Bad_boys

from networks import Networks
from networks import NetworkNotFoundError

from common_funs import ProcessedData
from common_funs import Dataset
from common_funs import Setpart
from common_funs import pos_label
from common_funs import neg_label

from config import Config

#suppress tf logging
sys.stdout = tempfile.TemporaryFile(mode='w')
os.environ['TF_CCP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

#suppress tf logging
tf.logging.set_verbosity(tf.logging.FATAL)
sys.stdout = sys.__stdout__




class EarlyStoppingError(StopIteration):
	pass

class EpochLimitException(StopIteration):
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

	def __init__(self, perflog, avgOverNrEpochs = 3, avgLimitPercent = 1.1):
		self.epoch = 0
		self.losses = []
		self.avgOverNrEpochs = avgOverNrEpochs
		self.avgLimitPercent = avgLimitPercent
		self._buff = FileBackedCSVBuffer(
			filename='earlystopping.csv',
			directory='logs',
			header=['epoch', 'val loss', 'avg val loss', 'loss limit', 'status'],
			clearFile=True,
			padding=12)

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
			perflog.log(best_acc = round(state['best_accuracy'], 3))
			raise EarlyStoppingError("Early stopping, on epoch: {}".format(self.epoch))
		else:
			m = "Loss delta to limit: {}, continuing...".format(round(avg_limit-val_loss,3))
			self._buff.append([m])
			self.losses.append(val_loss)

		self._buff.flush()

def _arg_callback_device(device_name):
	if device_name == 'cpu':
		cfg.device = '/cpu:0'
	elif device_name == 'gpu':
		cfg.device = '/gpu:0'
	else:
		raise Exception("device not in list")

def _arg_callback_st(gang_colors):
	cfg.trouble_gang = gang_colors
	cfg.trouble_type = 'save'
	print("<saving troublemakers after training>")

def _arg_callback_tt(gang_colors, trouble):
	trouble = float(trouble)
	cfg.trouble_gang = gang_colors
	cfg.trouble_level = trouble
	cfg.trouble_type = 'filter'
	print("<filtering training set with badness > {0:.0%}, troublemakers from gang: {1:}>"
		.format(trouble, gang_colors))

def _arg_callback_pt():
	cfg.print_test = True

def _arg_callback_ds(ds_name):
	"""
	Select dataset
	"""
	cfg.dataset_name = ds_name
	print("<Using dataset: {}>".format(ds_name))

def _arg_callback_sm(name = None):
	cfg.save_the_model = True
	cfg.model_save_name = name

def _arg_callback_boost(pretrained_id = None):
	if pretrained_id is None:
		pretrained_id = file_selector(cfg.models_path, "Select model for boosting")

	cfg.pretrained_id = pretrained_id
	cfg.training_mode = 'boost'
	print("<Boosting with pretrained model: {}>".format(cfg.pretrained_id))

def _arg_callback_eval(pretrained_id = None):
	if pretrained_id is None:
		pretrained_id = file_selector(cfg.models_path, "Select model to evaluate")

	cfg.pretrained_id = pretrained_id
	cfg.training_mode = 'evaluate'
	print("<Evaluating pretrained model " + cfg.pretrained_id + " for results only.>")

def _arg_callback_train(nr_epochs=1, count=1, batchsize=30):
	cfg.epochs = int(nr_epochs)
	cfg.run_count = int(count)
	cfg.batch_size = int(batchsize)
	cfg.training_mode = 'training'
	print("<Training for, epochs: {}, runs:{}, batchsize: {}>".format(nr_epochs, count, batchsize))

def _arg_callback_net(name):
	cfg.network_name = name
	print("<Using network: {}>".format(name))

def _arg_callback_in(file_name = None):
	cfg.ps_file_name = file_name
	print("<Using processed samples from selected file>")

def _arg_callback_ss(s_step = None, s_epoch = False):
	"""
	Set the snapshot step
	"""
	if isinstance(s_step, str) and s_step.lower() == 'none':
		s_step = None
	cfg.snapshots_per_epoch = int(s_step) if s_step is not None else None
	cfg.snapshot_epoch = True if str(s_epoch).lower() == 'true' else False
	print("<Snapshot step: {}, Snaphot epoch end: {}>".format(s_step, s_epoch))

def build_network(name, hyp, pd):
	params = {'hyp': hypers, 'pd': pd}
	nets = Networks()
	net = nets.get_network(name=name, params=params)

	return net

def create_model(net):
	best_path = os.path.join(temp_dir_best.name, 'checkpoint-best-')
	chkpt_path = os.path.join(temp_dir_checkpoints.name, 'checkpoint-')

	model = tflearn.DNN(
		net,
		tensorboard_verbose=3,
		checkpoint_path=chkpt_path,
		best_checkpoint_path=best_path,
		best_val_accuracy=0.0
	)

	#set embeddings
	if cfg.use_embeddings:
		emb = np.array(pd.embeddings[:pd.vocab_size], dtype=np.float32)
	else:
		emb = np.random.randn(pd.vocab_size, pd.emb_size).astype(np.float32)

	new_emb_t = tf.convert_to_tensor(emb)
	embeddings_tensor = tflearn.variables.get_layer_variables_by_name('embedding')[0]
	model.set_weights( embeddings_tensor, new_emb_t)
	w = model.get_weights(embeddings_tensor)
	debug_log.log(str(w.shape), "embedding layer shape", aslist=False)

	#debug_embeddingd(model, "fresh", embeddings_log)

	return model

def debug_embeddingd(model, when, logger):	
	embeddings_tensor = tflearn.variables.get_layer_variables_by_name('embedding')[0]
	w = model.get_weights(embeddings_tensor)
	for line in w:
		logger.log(np.array_str(line), logname=when, maxlogs=10)

def train_model(model, hyp, this_run_id, log_run, perflog):
	api = EarlyStoppingMonitor(perflog, avgOverNrEpochs = 3, avgLimitPercent = 1.05)
	monitorCallback = MonitorCallback(api)

	if cfg.snapshots_per_epoch is not None:
		snapshot_step = math.floor(ps.train.length / (cfg.snapshots_per_epoch * cfg.batch_size))
	else:
		snapshot_step = None

	model.fit(X_inputs=ps.train.xs,
			  Y_targets=ps.train.ys,
			  validation_set=(ps.valid.xs, ps.valid.ys),
			  show_metric=True,
	          batch_size=cfg.batch_size,
	          n_epoch=cfg.epochs,
	          shuffle=False,
	          run_id=this_run_id,
			  snapshot_step=snapshot_step,
			  snapshot_epoch=cfg.snapshot_epoch,
	          callbacks=monitorCallback)

	raise EpochLimitException("epoch limit")

	return model

def do_prediction(model, hyp, this_run_id, log_run, perflog, net):

	#debug_embeddingd(model, "after_training", embeddings_log)

	
	# print confusion matrix for the different sets
	print("\nRunning prediction...")
	print(boxString("Run id: " + this_run_id))

	cm = Binary_confusion_matrix()
	# splite the sets in chunks so that predict doesn't use all the memmories
	from common_funs import chunks
	fun_chunks = lambda fun, parts: [fun(part) for part in chunks(parts, 1000)]
	flatten = lambda l: [x for xs in l for x in xs]

	train_predictions = flatten(fun_chunks(model.predict, ps.train.xs))
	cm.calc(ps.train.ids , train_predictions, ps.train.ys, 'training-set')

	predictions = flatten(fun_chunks(model.predict, ps.valid.xs))
	cm.calc(ps.valid.ids , predictions, ps.valid.ys, 'validation-set')

	if cfg.print_test:
		predictions = flatten(fun_chunks(model.predict, ps.test.xs))
		cm.calc(ps.test.ids , predictions, ps.test.ys, 'test-set')
		perflog.log(
			test_acc = cm.metrics['test-set']['accuracy'],
			test_f1 = cm.metrics['test-set']['f1_score']
		)

	cm.print_tables()
	badboys.update(ps.train.ids , train_predictions, ps.train.ys)

	log_run.log(cm.metrics, logname="metrics", aslist = False)
	perflog.log(
		time = strftime('%Y-%m-%d %H:%M', time.localtime()),
		network = cfg.network_name,
		dataset = cfg.dataset_name,
		samples_file = cfg.ps_file_name,
		val_acc = cm.metrics['validation-set']['accuracy'],
		val_f1 = cm.metrics['validation-set']['f1_score'],
		run_id = this_run_id
	)


def get_model_magic_path(path):
	"""
	Return the path to the file that is last when alphabeticly sorted
	Gets a list of touple of (name, file) where name is the filename
	 with magic nrs, but without extensions
	"""
	best_name_path = None
	names_files = [(file.split('.')[0], file) for file in listdir(path) if file != "checkpoint"]

	if names_files:
		best_name_file = sorted(names_files, reverse=True, key = itemgetter(0))[0]
		best_name = best_name_file[0]
		best_name_path = os.path.join(path, best_name)

	return best_name_path

def save_model(model, name):
	import shutil
	this_model_path = os.path.join(cfg.models_path, name)
	
	# delete folder if it exists
	if os.path.exists(this_model_path):
		shutil.rmtree(this_model_path)

	os.mkdir(this_model_path)	
	magic_path = os.path.join(this_model_path, 'model')
	model.save(magic_path)

################################################################################

# affected by flags, need to be before consume_flags()
cfg = Config()
cfg.print_debug = True
cfg.trouble_gang = None
cfg.trouble_level = None
cfg.trouble_type = None
cfg.model_save_name = None
cfg.device = '/cpu:0'

# Handles command arguments, usefull for debugging
# usage: tflearn_rnn.py --pf debug_processed.pickle
#  will get samples from debug_processed.pickle
arghandler = Arg_handler()
arghandler.register_flag('in', _arg_callback_in, ['input', 'in-file'], "Which file to take samples from. args: <filename>")
arghandler.register_flag('net', _arg_callback_net, ['network'], "Which network to use. args: <network name>")
arghandler.register_flag('train', _arg_callback_train, helptext = "Use settings for training. Args: <epochs> <run_count> <batch size>")
arghandler.register_flag('ss', _arg_callback_ss, ['snapshot'], helptext = "Set snapshots. No arguments means no snapshots. Args: <snapshot step> <epoch end>")
arghandler.register_flag('eval', _arg_callback_eval, [], "Evaluate the network performance of a pre-trained model specified by run id. args: <run_id>")
arghandler.register_flag('ds', _arg_callback_ds, ['select-dataset', 'dataset'], "Which dataset to use. Args: <dataset-name>")
arghandler.register_flag('pt', _arg_callback_pt, ['print-test'], "Produce results on test-partition of dataset.")
arghandler.register_flag('st', _arg_callback_st, ['save-trouble'], "save trouble maker prediction to DB under gang name. Args: <gang-name>")
arghandler.register_flag('tt', _arg_callback_tt, ['train-trouble'], "Filter training set on troublemakers with trouble level. Args: <gang-name> <float trouble>")
arghandler.register_flag('sm', _arg_callback_sm, ['save-model'], "Save the trained model. Will be saved in dir with it's run id")
arghandler.register_flag('boost', _arg_callback_boost, [], "Load a saved model and continue training. Args <model id>")
arghandler.register_flag('device', _arg_callback_device, [], "Train on gpu or cpu. Args: <'gpu'/'cpu'>")
arghandler.consume_flags()

debug_log = Logger()
embeddings_log = Logger(enable=False)
perflog = DB_backed_log(cfg.sqlite_file, 'training_performance')
badboys = Bad_boys(cfg.sqlite_file, cfg.trouble_gang)

# show select menu if no file name given
if cfg.ps_file_name is None:
	cfg.ps_file_name = file_selector(cfg.processed_path, "Select sample file")


# Load processed data from file
with open(cfg.samples_path, 'rb') as handle:
    pd = pickle.load( handle )
ps = pd.dataset #processed samples

if cfg.trouble_level is not None:
	print('creating new training set based on troublemakers that are wrong {0:.0%} of the time'
		.format(cfg.trouble_level))
	trouble_filter = badboys.find(cfg.trouble_level)				
	ids = []; xs = []; ys = [];
	for sid, x, y in zip(ps.train.ids, ps.train.xs, ps.train.ys):
		if sid in trouble_filter:
			xs.append(x)
			ys.append(y)
			ids.append(sid)

	new_train_set = Setpart('training set', len(ids), ids, xs, ys)
	ps = Dataset(new_train_set, ps.valid, ps.test)
	cfg.model_save_name = cfg.trouble_gang
	cfg.save_the_model = True


# debug print tweets
if cfg.print_debug:
	for s_id, s_y, s_x in zip(ps.train.ids, ps.train.ys, ps.train.xs):
		ispos = np.array_equal(s_y, pos_label)
		label = "Positive (Sarcastic)" if ispos else "Negative (Not sarcastic)"
		logstring = "Sample id: {}, {}: {:<5}".format(s_id, label, "\n")
		logstring += " ".join( reverse_lookup(s_x, pd.rev_vocab, cfg.ascii_console ))
		debug_log.log(logstring, logname="reverse_lookup", maxlogs = 10, step = 3003)
	print("\nLogged sample values:\n")
	debug_log.print_log(logname="reverse_lookup")
	print('', end='\n\n')


# 'sönderhaxad' class that generates random hyperparamters in the range provided

hypers = Hyper(cfg.run_count,
	lstm = {'dropout': (0.4, 0.8)},
	middle= {'weight_decay': (0.01, 0.06)},
	dropout = {'dropout': (0.4, 0.8)},
	regression = {'learning_rate': (0.0005, 0.0015)},
	output = {'weight_decay': (0.01, 0.06)}
)
# hypers = Hyper(cfg.run_count,
# 	lstm = {'dropout': (0.51, 0.51)},
# 	middle= {'weight_decay': (0.038, 0.038)},
# 	dropout = {'dropout': (0.63, 0.63)},
# 	regression = {'learning_rate': (0.001, 0.001)},
# 	output = {'weight_decay': (0.038, 0.038)}
# )

# training loop, every loop trains a network with different hyperparameters
for hyp in hypers:
	log_run = Logger()

	tf.reset_default_graph()
	with tf.Graph().as_default(), tf.Session() as sess: #tf.device(cfg.device)
		sess.run(tf.global_variables_initializer())
		tflearn.config.init_training_mode()
		stop_reason = "Other error"

		this_run_id = common_funs.generate_name()
		log_run.log(hyp.get_hypers(), logname='hypers', aslist = False)
		log_run.log(this_run_id, logname='run_id', aslist = False)
		log_run.log(cfg.network_name, logname='network_name', aslist = False)
		log_run.log(cfg.ps_file_name, logname='Dataset', aslist = False)

		try:
			temp_dir_checkpoints = tempfile.TemporaryDirectory()
			temp_dir_best = tempfile.TemporaryDirectory()

			net = build_network(cfg.network_name, hyp, pd)

			if cfg.training_mode == 'training':
				model = create_model(net)
				model = train_model(model, hyp, this_run_id, log_run, perflog)

			elif cfg.training_mode == 'evaluate':
				model = tflearn.DNN(net)
				this_run_id = cfg.pretrained_id
				path = os.path.join(cfg.models_path, cfg.pretrained_id)
				magic_path = get_model_magic_path(path)
				model.load(magic_path)
				do_prediction(model, hyp, this_run_id, log_run, perflog, net)
				stop_reason = "Evaluation"

			elif cfg.training_mode == 'boost':
				model = tflearn.DNN(net)
				path = os.path.join(cfg.models_path, cfg.pretrained_id)
				magic_path = get_model_magic_path(path)
				model.load(magic_path)
				model = train_model(model, hyp, this_run_id, log_run, perflog)

			else:
				raise Exception("training mode not recognized")

		except (EarlyStoppingError, EpochLimitException) as e:
			stop_reason = str(e)
			magic_path = get_model_magic_path(temp_dir_best.name)

			if magic_path:
				print("\nLoading best checkpoint...")
				model.load(magic_path)

			do_prediction(model, hyp, this_run_id, log_run, perflog, net)

			if cfg.save_the_model:
				name = this_run_id if cfg.model_save_name is None else cfg.model_save_name
				save_model(model, name)

		finally:
			temp_dir_best.cleanup()
			temp_dir_checkpoints.cleanup()
	

	log_run.save(this_run_id + '.log')
	if len(perflog.peek()) > 0:
		perflog.log(status = stop_reason)
		perflog.flush()

	embeddings_log.save("debug_embeddingd.log")

	if cfg.trouble_type == 'save':
		badboys.save()

debug_log.save("training_debug.log")

