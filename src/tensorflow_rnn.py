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
import json
import pickle
import csv
import sys
import tempfile
from common_funs import *

sys.path.append("../identifiera-sarkasm/")
sys.path.append("../identifiera-sarkasm/tfnetworks/")

import tfnetworks

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

cfg = Config()
n_classes = 2
chunk_size = cfg.embedding_size
n_chunks = cfg.max_sequence
rnn_size = 64
roundform = "{0:.5f}"

data_placeholder = tf.placeholder(dtype=tf.int32,shape=[None,None])
labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,n_classes])
keep_prob_placeholder = tf.placeholder('float')

trainable_embeddings = False
logs_path = tempfile.gettempdir() + "/tfnetwork/"
shuffle_training = False
batch_size = cfg.batch_size

network_name = cfg.network_name
date_stamp = time.strftime("%d%b")
run_id = date_stamp + "-" + network_name
def _arg_callback_pt():
	global print_test
	print_test = True

def _arg_callback_ds(ds_name):
    """
    Select dataset
    """
    cfg.dataset_name = ds_name
    print("<Using dataset: {}>".format(ds_name))

def _arg_callback_pretrained(file_name):
    cfg.save_the_model = False
    cfg.pretrained_model = True
    cfg.training = False
    cfg.pretrained_file = file_name + ".tfl"
    print("<Using pretrained model " + pretrained_path + " for results only.")

def _arg_callback_train(nr_epochs=1, count=1, batchsize=30):
    cfg.epochs = int(nr_epochs)
    cfg.run_count = int(count)
    cfg.batch_size = int(batchsize)
    cfg.training = True
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
		embedding_layer = tf.nn.embedding_lookup(embedding_tensor,word)
		return embedding_layer #Not sure if this is done yet

#Defining and building the Neural Network

def train_neural_network(ps,emb_init,W,emb_placeholder,network_name,log_run):
	# Defining all the operations
	embeddings = word_embedding_layer(data_placeholder,W)
	network = tfnetworks.fetch_network(network_name,n_classes,params = {'rnn_size': rnn_size})
	prediction = network.feed_network(embeddings,keep_prob_placeholder,chunk_size,n_chunks)
	accuracy = test_accuracy(tf.nn.softmax(prediction),labels_placeholder)
	val_accuracy_op = test_accuracy(data_placeholder,labels_placeholder)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,
	labels = labels_placeholder
	))

	l2_loss = network.calc_l2_loss()
	if trainable_embeddings:
		l2_loss = l2_loss + tf.nn.l2_loss(W)

	optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost + 0.01 * l2_loss)

	#Defining the summaries
	tf.summary.scalar("Accuracy:", accuracy)
	#tf.summary.scalar("Optimizer", optimizer)
	tf.summary.scalar("Loss:", cost)
	date = time.strftime("%b%d%y-%H%M%S")
	summary_op = tf.summary.merge_all()
	sess = tf.Session()
	xs_split,ys_split = split_chunks(ps.train.xs,cfg.batch_size, np.array(ps.train.ys))
	with sess.as_default():
		sess.run(tf.global_variables_initializer())
		set_embedding(sess,emb_init,emb_placeholder,emb)
		loops = len(xs_split)
		writer = tf.summary.FileWriter(logs_path + "/" + network_name + "-" + date,sess.graph)
		print("Tensorboard log path:",logs_path)
		for epoch in range(cfg.epochs):
			epoch_loss = 0
			indices = list(range(loops))
			#print(indices)
			if shuffle_training:
				random.shuffle(indices)
				#print(indices)
			print("\n=== BEGIN EPOCH",epoch+1, "===\n")
			for it,batch_i in enumerate(indices):
				batch_x = xs_split[batch_i]
				batch_y = ys_split[batch_i]
				_, c,train_acc,summary = sess.run([optimizer,cost,accuracy,summary_op], feed_dict = {data_placeholder: batch_x, labels_placeholder: batch_y, keep_prob_placeholder: 0.5})
				epoch_loss += c
				writer.add_summary(summary, epoch*loops + it)
				print("Optimizer:",optimizer.name, "|", it + 1, "batches completed out of:", loops)
				print("current loss:",roundform.format(epoch_loss),"| Accuracy :",roundform.format(train_acc), "",end=" \033[A\r",flush=True)

			print("\033[2B")
			print("VALIDATING TRAINING:...")
			val_predict = batchpredict(batch_size,ps.valid.xs,prediction)
			val_accuracy = val_accuracy_op.eval(feed_dict={data_placeholder: val_predict,labels_placeholder: np.array(ps.valid.ys), keep_prob_placeholder: 1.0})
			print('Epoch', epoch+1, 'completed out of', cfg.epochs, 'loss:', roundform.format(epoch_loss), '| Accuracy:', roundform.format(val_accuracy))

			saver = tf.train.Saver()
			save_path = saver.save(sess, "./models/tfcheckpoint.ckpt")
			print("Checkpoint file saved in %s" % save_path )

		saver = tf.train.Saver()
		date = time.strftime("%m%d%y-%H%M%S")
		saver_path = saver.save(sess, "./models/tfrnn_model-%s.ckpt" % date)
		print("Model saved at %s" % saver_path )
		run_test_print_cm(ps,prediction,log_run)
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

def test_network(ps,network_name,path=None):
    if path == None :
        print("No model selected")
    else:
        run_test(ps,path,network_name)

def run_test(ps,path,network_name):
    network = tfnetworks.fetch_network(network_name,n_classes,params={'rnn_size': rnn_size})
    test_data = ps.test.xs
    test_labels = np.array(ps.test.ys)
    emb_init, W, emb_placeholder = init_embedding(pd.vocab_size, pd.emb_size)
    embeddings = word_embedding_layer(data_placeholder,W)
    output = network.feed_network(embeddings,keep_prob_placeholder,chunk_size,n_chunks)
    with tf.Session().as_default() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, path)
        accuracy = test_network_run(ps.train.xs,np.array(ps.train.ys),output)
        print("Test accuracy of this network is: ", accuracy)

def run_test_print_cm(ps,network_op,log_run):

    sess = tf.get_default_session()
    cm = Binary_confusion_matrix()
    pred1 = batchpredict(90,ps.train.xs,network_op)
    cm.calc(ps.train.ids , pred1, ps.train.ys, 'training-set')
    pred2 = batchpredict(90,ps.valid.xs,network_op)
    cm.calc(ps.valid.ids , pred2, ps.valid.ys, 'validation-set')
    if cfg.print_test:
        pred3 = sess.run(network_op, feed_dict={data_placeholder: ps.test.xs, keep_prob_placeholder: 1.0})
        cm.calc(ps.test.ids , pred3, ps.test.ys, 'test-set')
    cm.print_tables()
    cm.save_predictions(predictions_filename,
						directory = 'logs',
						sets=['training-set','validation-set','test-set'],
						update = True)

    cm.save(run_id + '.res', content='metrics')
    log_run.log(cm.metrics, logname="metrics", aslist = False)
    the_list = [
    time.strftime('%Y-%m-%d %H:%M', time.localtime()),
    network_name,
    os.path.basename(cfg.samples_path),
    cm.metrics['validation-set']['accuracy'],
    cm.metrics['validation-set']['f1_score'],
    run_id
    ]
    if cfg.print_test:
    	the_list.append(cm.metrics['test-set']['accuracy'])
    	the_list.append(cm.metrics['test-set']['f1_score'])
    perflog.replace(the_list)

def batchpredict(batch_size,data,network_op):

	sess = tf.get_default_session()
	data_batches, _ = split_chunks(data,batch_size)
	results = np.array([np.zeros(n_classes)])
	for batch in data_batches:
		pred = sess.run(network_op,feed_dict={data_placeholder: batch, keep_prob_placeholder: 1.0})
		results = np.concatenate((results,pred))

	results = np.delete(results, (0), axis = 0)
	return results
# Here starts the program

#Argument handling, Copy paste from tflearn_rnn.py
arghandler = Arg_handler()
arghandler.register_flag('in', _arg_callback_in, ['input', 'in-file'], "Which file to take samples from. args: <filename>")
arghandler.register_flag('net', _arg_callback_net, ['network'], "Which network to use. args: <network name>")
arghandler.register_flag('train', _arg_callback_train, helptext = "Use settings for training. Args: <epochs> <run_count> <batch size>")
#arghandler.register_flag('ss', _arg_callback_ss, ['snapshot'], helptext = "Set snapshots. No arguments means no snapshots. Args: <snapshot step> <epoch end>")
arghandler.register_flag('pretrained', _arg_callback_pretrained, [], "Evaluate the network performance of a pre-trained model specified by the name of the argument. args: <path>")
arghandler.register_flag('ds', _arg_callback_ds, ['select-dataset', 'dataset'], "Which dataset to use. Args: <dataset-name>")
arghandler.register_flag('pt', _arg_callback_pt, ['print-test'], "Produce results on test-partition of dataset.")
print("\n")
arghandler.register_flag('trainemb', _arg_callback_trainemb, ['trainable'], "Set trainable embeddings")
arghandler.register_flag('eshuffle', _arg_callback_eshuffle, ['truth'], "Want to shuffle per epoch?")

arghandler.consume_flags()
predictions_filename = 'predictions.pickle'

perflog = FileBackedCSVBuffer(
	"training_performance.csv",
	"logs",
	header=['Time', 'Network name', 'data file', 'Val acc', 'Val f1', 'Run id','Status'],
	padding=17)

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

train_neural_network(ps,emb_init,W,emb_placeholder,network_name,log_run)


print ("=== Code ran Successfully ===")
