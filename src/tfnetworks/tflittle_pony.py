import tensorflow as tf
from tensorflow.contrib import rnn
from networks import abstract_network

class little_pony_network(abstract_network):
    def __init__(self,n_classes,rnn_size = 256):
        self._name = "little_pony"
        self._layer_weights = tf.Variable(tf.random_uniform([rnn_size,n_classes]), name="Weights")
        self._layer_biases = tf.Variable(tf.random_uniform([n_classes]))
        self._GRU_cell = rnn.GRUCell(rnn_size)
