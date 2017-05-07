import tensorflow as tf
from tensorflow.contrib import rnn
from . import networks as net

class LittlePonyNetwork(net.AbstractNetwork):
    def __init__(self,n_classes,rnn_size = 256):
        self._name = "little_pony"
        self._layer_weights = tf.Variable(tf.random_uniform([rnn_size,n_classes]), name="weights")
        self._layer_biases = tf.Variable(tf.random_uniform([n_classes]), name="biases")
        self._lstm_cell = rnn.BasicLSTMCell(rnn_size)

    def feed_network(self,data,keep_prob,chunk_size,n_chunks):
        sequence_lengths = net.calc_seqlenth(data)
        dimensions = data.get_shape().as_list()
        batch_size = dimensions[0]
        weight_dropout = tf.nn.dropout(self._layer_weights, keep_prob)
        rnn_dropout = rnn.core_rnn_cell.DropoutWrapper(self._lstm_cell,output_keep_prob=keep_prob)
        data = tf.transpose(data,[1,0,2])
        data = tf.reshape(data,[-1,chunk_size])
        sequence = tf.split(data, n_chunks, 0)
        outputs, states = rnn.static_rnn(rnn_dropout, sequence, dtype=tf.float32, sequence_length = sequence_lengths)
        outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])
        output = net.advanced_indexing_op(outputs, sequence_lengths)
        output = tf.add(tf.matmul(output,weight_dropout), self._layer_biases)
        return output

    def get_name(self):
        return self._name

    def calc_l2_loss(self):
        l2_loss = tf.nn.l2_loss(self._layer_weights)
        return l2_loss
