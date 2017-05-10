'''
===RNN Network in TensorFlow

* This is a Recurrent Neural Network for sarcasm detection
* Author: DATX02-25
'''
# Importing dependencies for file handling
import collections
import math
import random
import os
import glob
import json
import pickle
import csv
import sys
import tempfile
from common_funs import *

sys.path.append("../identifiera-sarkasm/")
sys.path.append("../identifiera-sarkasm/tfnetworks/")

import tfnetworks
from tfnetworks.networks import calc_seqlenth

import time

import random
# Importing tensorflow
import tensorflow as tf
# Importing rnn framework
from tensorflow.contrib import rnn
from tensorflow.contrib import learn
# Importing NumPy
import numpy as np

# Importing Settings
from config import Config

#This is 'fulhack'
class EarlyStoppingHelper:

    def __init__(self, epoch_threshold = 3, avg_limit_percent = 1.1):
        self.epochs_since_best = 0
        self.last_avg_loss = 0
        self.last_best_acc = 0
        self.last_best_f1 = 0
        self.losses = []
        self.epoch_threshold = epoch_threshold
        self.avg_limit_percent = avg_limit_percent

    def test_scorings(self,loss,accuracy,f1_score = 0):
        flags = {'update': False , 'passing': False, 'early_stop': False}
        # First Check accuracy scoring, if better, than reset counter and set new scores
        if f1_score and f1_score > self.last_best_f1 :
            self.epochs_since_best = 0
            self.last_best_f1 = f1_score
            flags['update'] = True
            return flags

        if not f1_score and accuracy > self.last_best_acc :
            self.epochs_since_best = 0
            self.last_best_acc = accuracy
            flags['update'] = True
            return flags

        self.epochs_since_best += 1

        if self.epochs_since_best > self.epoch_threshold :
            flags['early_stop'] = True
            return flags
        if len(self.losses) < self.epoch_threshold:
            avg_loss = loss
        else:
            avg_loss = sum(self.losses[self.epoch_threshold:])/self.epoch_threshold
            avg_loss = round(avg_loss,3)

        loss_limit = self.avg_limit_percent * avg_loss
        if loss > loss_limit:
            flags['early_stop'] = True
        else:
            self.losses.append(loss)
            flags['passing'] = True
        return flags


cfg = Config()
cfg = Config()
cfg.print_debug = True
cfg.predictions_filename = 'predictions.pickle'
cfg.model_save_name = None
n_classes = 2
chunk_size = cfg.embedding_size
n_chunks = cfg.max_sequence
roundform = "{0:.5f}"
stop_reason = None

data_placeholder = tf.placeholder(dtype=tf.int32,shape=[None,cfg.max_sequence])
predict_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,None])
labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,n_classes])
keep_prob_placeholder = tf.placeholder('float')
f1_score_placeholder = tf.placeholder('float')

trainable_embeddings = False
logs_path = os.path.join(tempfile.gettempdir() ,"tfnetwork")
shuffle_training = False
batch_size = cfg.batch_size

network_name = cfg.network_name
date_stamp = time.strftime("%d%b-%H%M")
run_id = date_stamp + "-" + network_name
slizing = False
monitor_f1 = False
stop_reason = None

dynamic = True

def _arg_callback_pt():
	global print_test
	print_test = True

def _arg_callback_ds(ds_name):
    """
    Select dataset
    """
    cfg.dataset_name = ds_name
    print("<Using dataset: {}>".format(ds_name))

def _arg_callback_eval(model_name=None):
    cfg.save_the_model = False
    #cfg.pretrained_model = True
    cfg.training_mode = "evaluate"
    cfg.pretrained_file = model_name if model_name != None else None
    if model_name != None:
        print("<Using pretrained model " + model_name + " for results only >")

def _arg_callback_train(nr_epochs=1, count=1, batchsize=30):
    cfg.epochs = int(nr_epochs)
    cfg.run_count = int(count)
    cfg.batch_size = int(batchsize)
    cfg.training_mode = "training"
    print("<Training for, epochs: {}, runs:{}, batchsize: {}>".format(nr_epochs, count, batchsize))

def _arg_callback_net(name):
	cfg.network_name = name
	print("<Using network: {}>".format(name))

def _arg_callback_in(file_name):
    """
    Take preprocessed samples from the selected file
    """
    cfg.ps_file_name = file_name
    print("<Using processed samples from: {}>".format(cfg.samples_path))

def _arg_callback_trainemb(trainable = True):
    global trainable_embeddings
    trainable_embeddings = trainable

def _arg_callback_eshuffle(truth = True):
	global shuffle_training
	shuffle_training = truth

def _arg_callback_slicing(slicing = True):
    global slizing
    slizing = slicing

def _arg_callback_usef1(f1 = True):
    global monitor_f1
    monitor_f1 = f1

def _arg_callback_dynseq(dyn = False):
    global dynamic
    dynamic = dyn
#def _arg_callback_ss(s_step = None, s_epoch = 'False'):
#	"""
#	Set the snapshot step
#	"""
#	global snapshot_step, snapshot_epoch
#	if isinstance(s_step, str) and s_step.lower() == 'none':
#		s_step = None
#	snapshot_step = int(s_step) if s_step is not None else None
#	snapshot_epoch = True if s_epoch.lower() == 'true' else False
#	print("<Snapshot step: {}, Snaphot epoch end: {}>".format(s_step, s_epoch))

## Create the embedding variable
def init_embedding(vocabulary_size,embedding_size,trainable = False):
    with tf.device("/cpu:0"):
        W = tf.Variable(tf.constant(0.0, shape = [vocabulary_size, embedding_size]),
                        trainable = trainable,
                        name = "W")
        embedding_placeholder = tf.placeholder(dtype = tf.float32,
                                               shape = [vocabulary_size, embedding_size]
                                               )
        embedding_init = W.assign(embedding_placeholder)
        return embedding_init, W, embedding_placeholder


# Setting the word embeddings
def set_embedding(sess,init,placeholder,embeddings):
	with tf.device("/cpu:0"):
		sess.run(init, feed_dict={placeholder: embeddings})
#Word embedding layer
def word_embedding_layer(word,embedding_tensor):
	with tf.device("/cpu:0"):
		embedding_layer = tf.nn.embedding_lookup(embedding_tensor,word, validate_indices=False)
	#	shape = [-1] + embedding_layer.get_shape().as_list()[1:3] + [1]
	#	embedding_layer.seq_length = calc_seqlenth(tf.reshape(word,shape))
		return embedding_layer #Not sure if this is done yet

#Defining and building the Neural Network

def train_neural_network(ps,emb_init,W,emb_placeholder,network_name,log_run,perflog):
	# Defining all the operations
	es_handler = EarlyStoppingHelper(epoch_threshold = 3, avg_limit_percent = 1.05)
	embeddings = word_embedding_layer(data_placeholder,W)
	network = tfnetworks.fetch_network(network_name,n_classes)
	prediction = network.feed_network(embeddings,keep_prob_placeholder,chunk_size,n_chunks)
	accuracy = test_accuracy(tf.nn.softmax(prediction),labels_placeholder)
	val_accuracy_op = test_accuracy(predict_placeholder,labels_placeholder)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = labels_placeholder))

	val_cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_placeholder, labels = labels_placeholder))
	l2_loss = network.calc_l2_loss()
	if trainable_embeddings:
		l2_loss = l2_loss + tf.nn.l2_loss(W)

	optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost + 0.01 * l2_loss)

	#Defining the summaries
	train_acc_sum = tf.summary.scalar("Accuracy:", accuracy)
	#tf.summary.scalar("Optimizer", optimizer)
	train_loss_sum = tf.summary.scalar("Loss:", cost)
	val_acc_sum = tf.summary.scalar("val_accuracy:", val_accuracy_op)
	val_f1_sum = tf.summary.scalar("val_f1_score:", f1_score_placeholder)
	date = time.strftime("%b%d%y-%H%M%S")
	train_summary_op = tf.summary.merge([train_acc_sum,train_loss_sum])
	val_summary_op = tf.summary.merge([val_acc_sum,val_f1_sum])
	sess = tf.Session()
	width, _ = ps.train.xs.shape
	slice_size = width//cfg.epochs if cfg.epochs > 1 else width//10
	#xs_split,ys_split = split_chunks(ps.train.xs,cfg.batch_size, np.array(ps.train.ys))
	with sess.as_default():
		print("Current Network Model id: %s" % run_id )
		sess.run(tf.global_variables_initializer())
		set_embedding(sess,emb_init,emb_placeholder,emb)
		early_stop = False
		writer = tf.summary.FileWriter(os.path.join(logs_path,run_id),sess.graph)
		os.makedirs(os.path.join(".","models",run_id))
		print("Tensorboard log path:",logs_path)
		training_flags = {}
		for epoch in range(cfg.epochs):
			train_set,train_test = batch_slice(ps.train.xs,epoch*slice_size , slice_size, slizing)
			train_lab,train_labtest = batch_slice(np.array(ps.train.ys),epoch*slice_size,slice_size, slizing)
			if shuffle_training:
				train_set,train_lab = shuffle_data(train_set,train_lab)

			xs_split,ys_split = split_chunks(train_set,cfg.batch_size,train_lab)
			epoch_loss = 0
			loops = len(xs_split)
			#print(indices)
			print("\n=== BEGIN EPOCH",epoch+1, "===\n")
			for batch_i in range(loops):
				batch_x = xs_split[batch_i]
				batch_y = ys_split[batch_i]
				_, c,train_acc ,summary = sess.run([optimizer,cost,accuracy,train_summary_op], feed_dict = {data_placeholder: batch_x, labels_placeholder: batch_y, keep_prob_placeholder: 0.5})
				writer.add_summary(summary, epoch*loops + batch_i)
				epoch_loss += c

				print("Optimizer:",optimizer.name, "|", batch_i + 1, "batches completed out of:", loops)
				print("current loss:",roundform.format(epoch_loss),"| Accuracy :",roundform.format(train_acc), "",end=" \033[A\r",flush=True)

			print("\033[2B")
			print("VALIDATING TRAINING:...")
			val_labels = np.array(ps.valid.ys)
			val_predict = batchpredict(batch_size,ps.valid.xs,prediction)
			val_loss = val_cost_op.eval(feed_dict={predict_placeholder: val_predict, labels_placeholder: val_labels})
			val_f1_score = calc_f1_score(val_predict,val_labels)
			val_accuracy,summary = sess.run([val_accuracy_op,val_summary_op],feed_dict={predict_placeholder: val_predict,labels_placeholder: val_labels, keep_prob_placeholder: 1.0,f1_score_placeholder: val_f1_score})
			writer.add_summary(summary, epoch*loops)
			print('Epoch', epoch+1, 'completed out of', cfg.epochs, 'loss:', roundform.format(epoch_loss), '| Accuracy:', roundform.format(val_accuracy),'| F1-score:',roundform.format(val_f1_score))
			if monitor_f1:
				training_flags = es_handler.test_scorings(val_loss,val_accuracy,val_f1_score)
			else:
				training_flags = es_handler.test_scorings(val_loss,val_accuracy)
			if training_flags['update']:
				saver = tf.train.Saver()
				save_path = saver.save(sess, os.path.join(".","models",run_id,"Checkpoint.ckpt"))
				print("Checkpoint file saved in %s" % save_path )

			elif training_flags['early_stop']:
				print("Training will stop because of EarlyStopping")
				print("last save path is %s" % save_path)
				early_stop = True
				break

		stop_reason = "epochs_end" if not early_stop else "early_stop"
		saver = tf.train.Saver()
		if training_flags['update']:
			saver_path = saver.save(sess, os.path.join(".","models",run_id,"final.ckpt"))
			print("Model saved at %s" % saver_path )
		elif training_flags['passing']:
			print("Latest Checkpoint is the best")
		run_test_print_cm(ps,prediction,perflog,log_run)
		sess.close()

def split_chunks(xs,size,ys=None):
    xh,xw = xs.shape
    n_batches = xh//size
    ys_split = None
    if ys != None:
        yh,yw = ys.shape
        if xh != yh :
            raise ValueError("Sizes don't match")
        ys_split = split_chunks_helper(ys,n_batches,size)
    xs_split = split_chunks_helper(xs,n_batches,size)
    return xs_split,ys_split

def split_chunks_helper(xs,n_batches,size):
	xh, _ = xs.shape
	#print(n_batches)
	overflow = xh - (n_batches * size)
	#print(overflow)
	xs_overflow = xs[-overflow:]
	    #print(xs_overflow.shape)
	xs_notoverflow = xs[:xh - overflow]
	xs_split = np.split(xs_notoverflow,n_batches)
	xs_split.append(np.array(xs_overflow))
	return xs_split

# Method for validating network in training
def test_accuracy(prediction,labels):
    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    return accuracy

def test_network(ps,W,network_name,perflog,model=None):
    if model == None :
        print("No model selected")
        print("Your network %s have following matching models:" % network_name)
        print_matching_models(network_name)
    elif network_name not in model:
        print("Your selected model %s is not for your selected network: %s" % (path, network_name))
    else:
        path = os.path.join(".","models", model)
        run_test(ps,W,path,perflog,network_name)

def run_test(ps,W,model,perflog,network_name):
    network = tfnetworks.fetch_network(network_name,n_classes)
    test_data = ps.test.xs
    test_labels = np.array(ps.test.ys)
    embeddings = word_embedding_layer(data_placeholder,W)
    output = network.feed_network(embeddings,keep_prob_placeholder,chunk_size,n_chunks)
    path = determine_model(model)
    with tf.Session().as_default() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, path)
        run_test_print_cm(ps,output,perflog,log_run)

def determine_model(model_path):
    files = os.listdir(model_path)
    append = "final.ckpt" if "final.ckpt" in files else "Checkpoint.ckpt"
    return os.path.join(model_path,append)

def batch_slice(arr,start,size,diff = False):
    arr_slice = arr[start:start + size]
    if diff:
        diff_left = arr[:start]
        diff_right = arr[start+size:]
        arr = np.concatenate((diff_left,diff_right))

    return arr, arr_slice

def shuffle_data(xs,ys):
    if (len(xs) != len(ys)):
        raise ValueError("Dimensions must match")
    indices = list(range(len(xs)))
    random.shuffle(indices)
    shuffled_xs = []
    shuffled_ys = []
    for i in indices:
        shuffled_xs.append(xs[i])
        shuffled_ys.append(ys[i])

    return np.array(shuffled_xs),np.array(shuffled_ys)

def run_test_print_cm(ps,network_op,perflog,log_run):

    stop_reason = "Evaluation"
    sess = tf.get_default_session()
    cm = Binary_confusion_matrix()
    pred1 = batchpredict(90,ps.train.xs,network_op)
    cm.calc(ps.train.ids , pred1, ps.train.ys, 'training-set')
    pred2 = batchpredict(90,ps.valid.xs,network_op)
    cm.calc(ps.valid.ids , pred2, ps.valid.ys, 'validation-set')
    if cfg.print_test:
        pred3 = sess.run(network_op, feed_dict={data_placeholder: ps.test.xs, keep_prob_placeholder: 1.0})
        cm.calc(ps.test.ids , pred3, ps.test.ys, 'test-set')
        perflog.log(
        test_acc = cm.metrics['test-set']['accuracy'],
        test_f1 = cm.metrics['test-set']['f1_score']
        )

    cm.print_tables()
    cm.save_predictions(predictions_filename,
                        directory = 'logs',
                        sets=['training-set','validation-set','test-set'],
                        update = True)

    cm.save(run_id + '.res', content='metrics')
    log_run.log(cm.metrics, logname="metrics", aslist = False)

    perflog.log(
		time = time.strftime('%Y-%m-%d %H:%M', time.localtime()),
		network = cfg.network_name,
		dataset = cfg.dataset_name,
		samples_file = cfg.ps_file_name,
		val_acc = cm.metrics['validation-set']['accuracy'],
		val_f1 = cm.metrics['validation-set']['f1_score'],
		run_id = run_id
	)

	# for troublemaker analysis
    cm.save_predictions(cfg.predictions_filename,
                        directory = cfg.logs_path,
                        sets=['training-set','validation-set','test-set'],
                        update = True
                        )

def batchpredict(batch_size,data,network_op):

	sess = tf.get_default_session()
	data_batches, _ = split_chunks(data,batch_size)
	results = np.array([np.zeros(n_classes)])

	for batch in data_batches:
		pred = sess.run(network_op,feed_dict={data_placeholder: batch, keep_prob_placeholder: 1.0})
		results = np.concatenate((results,pred))
	results = np.delete(results, (0), axis = 0)
	return results

def print_matching_models(network_name):
    for file in os.listdir("models"):
        if network_name in file:
            print(file,end="\n")
# Here starts the program
def calc_f1_score(predictions,labels):
    predictions = np.argmax(predictions,1)
    labels = np.argmax(labels,1)
    tp = fp = tn = fn = 0
    for a,b in zip(predictions,labels):
        if a == 1:
            if a == b:
                tp += 1
            else:
                fp += 1
        else:
            if a == b:
                tn += 1
            else:
                fn += 1

    precision = ( tp / (tp + fp) ) if (tp + fp) > 0 else 0
    recall = ( tp / (tp + fn) ) if (tp + fn) > 0 else 0
    f1_score = 2*((precision * recall) / (precision + recall )) if (precision + recall) > 0 else 0
    return f1_score

#Argument handling, Copy paste from tflearn_rnn.py
arghandler = Arg_handler()
arghandler.register_flag('in', _arg_callback_in, ['input', 'in-file'], "Which file to take samples from. args: <filename>")
arghandler.register_flag('net', _arg_callback_net, ['network'], "Which network to use. args: <network name>")
arghandler.register_flag('train', _arg_callback_train, helptext = "Use settings for training. Args: <epochs> <run_count> <batch size>")
#arghandler.register_flag('ss', _arg_callback_ss, ['snapshot'], helptext = "Set snapshots. No arguments means no snapshots. Args: <snapshot step> <epoch end>")
arghandler.register_flag('eval', _arg_callback_eval, ['model_path'], "Evaluate the network performance of a pre-trained model specified by the name of the argument. args: <path>")
arghandler.register_flag('ds', _arg_callback_ds, ['select-dataset', 'dataset'], "Which dataset to use. Args: <dataset-name>")
arghandler.register_flag('pt', _arg_callback_pt, ['print-test'], "Produce results on test-partition of dataset.")
print("\n")
arghandler.register_flag('trainemb', _arg_callback_trainemb, ['trainable'], "Set trainable embeddings")
arghandler.register_flag('eshuffle', _arg_callback_eshuffle, ['truth'], "Want to shuffle per epoch?")
arghandler.register_flag('slicing', _arg_callback_slicing, ['slicing'], "Slice out the training accuracy set from the data")
arghandler.register_flag('usef1', _arg_callback_usef1, ['f1'], "Use F1 as validation matric instead of accuracy")
arghandler.register_flag('dynseq', _arg_callback_dynseq, ['dyn'], "Use dynamic sequencing. If setting this to false, recommend to run without --trainemb args: <dyn>")
arghandler.consume_flags()
predictions_filename = 'predictions.pickle'

perflog = DB_backed_log(cfg.sqlite_file, 'training_performance')

with open(cfg.samples_path, 'rb') as handle:
    pd = pickle.load( handle )

ps = pd.dataset #Processed Samples

if cfg.use_embeddings:
    emb = np.array(pd.embeddings[:pd.vocab_size], dtype=np.float32)
else:
    emb = np.random.randn(pd.vocab_size, pd.emb_size).astype(np.float32)


emb_init, W, emb_placeholder = init_embedding(pd.vocab_size, pd.emb_size, trainable_embeddings)

log_run = Logger()

log_run.log(network_name, logname='network_name', aslist = False)
log_run.log(cfg.ps_file_name, logname='Dataset', aslist = False)

if cfg.training_mode != "evaluate":
    train_neural_network(ps,emb_init,W,emb_placeholder,network_name,log_run,perflog)
else:
    test_network(ps,W,network_name,perflog,cfg.pretrained_file)

log_run.save(run_id + '.log')
if len(perflog.peek()) > 0:
    perflog.log(status = stop_reason)
    perflog.flush()

print ("=== Code ran Successfully ===")
