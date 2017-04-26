import tensorflow as tf
from tensorflow.contrib import rnn
from .networks import AbstractNetwork

class LittlePonyNetwork(AbstractNetwork):
    def __init__(self,n_classes,rnn_size = 256):
        self._name = "little_pony"
        self._layer_weights = tf.Variable(tf.random_uniform([rnn_size,n_classes]), name="weights")
        self._layer_biases = tf.Variable(tf.random_uniform([n_classes]), name="biases")
        self._GRU_cell = rnn.GRUCell(rnn_size)

    def feed_network(self,data,keep_prob,chunk_size,n_chunks):
        dimensions = data.get_shape().as_list()
        batch_size = dimensions[0]
        weight_dropout = tf.nn.dropout(self._layer_weights, keep_prob)
        rnn_dropout = rnn.core_rnn_cell.DropoutWrapper(self._GRU_cell,output_keep_prob=keep_prob)
        data = tf.transpose(data,[1,0,2])
        data = tf.reshape(data,[-1,chunk_size])
        sequence = tf.split(data, n_chunks, 0)
        outputs, states = rnn.static_rnn(rnn_dropout, sequence, dtype=tf.float32)
        output = tf.add(tf.matmul(outputs[-1],weight_dropout), self._layer_biases)
        return output

    def get_name(self):
        return self._name

    def calc_l2_loss(self):
        l2_loss = tf.nn.l2_loss(self._layer_weights)
        return l2_loss
