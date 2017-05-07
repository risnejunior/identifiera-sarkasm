import tensorflow as tf
from tensorflow.contrib import rnn
from . import networks as net

class LittlePonyNetwork(net.AbstractNetwork):
    def __init__(self,n_classes,rnn_size = 256):
        self._name = "little_pony"
        self._layer_weights_1 = tf.Variable(tf.random_uniform([rnn_size,64]), name="weights")
        self._layer_biases_1 = tf.Variable(tf.random_uniform([64]), name="biases")
        self._layer_weights_2 = tf.Variable(tf.random_uniform([64,n_classes]), name="weights")
        self._layer_biases_2 = tf.Variable(tf.random_uniform([n_classes]), name="biases")

        self._lstm_cell = rnn.BasicLSTMCell(rnn_size)

    def feed_network(self,data,keep_prob,chunk_size,n_chunks):
        sequence_lengths = net.calc_seqlenth(data)
        dimensions = data.get_shape().as_list()
        batch_size = dimensions[0]
        weight_dropout_1 = tf.nn.dropout(self._layer_weights_1, keep_prob)
        weight_dropout_2 = tf.nn.dropout(self._layer_weifhts_2, keep_prob)
        rnn_dropout = rnn.core_rnn_cell.DropoutWrapper(self._lstm_cell,output_keep_prob=keep_prob)

        # Calculation Begin
        input_shape = data.get_shape().as_list()
        ndim = len(input_shape)
        axis = [1, 0] + list(range(2,ndim))
        data = tf.transpose(data,(axis))
        sequence = tf.unstack(data)
        outputs, states = rnn.static_rnn(rnn_dropout, sequence, dtype=tf.float32, sequence_length = sequence_lengths)
        outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])
        output1 = net.advanced_indexing_op(outputs, sequence_lengths)
        output1 = tf.add(tf.matmul(output,weight_dropout_1), self._layer_biases_1)
        input2 = tf.nn.relu(output1)
        output2 = tf.add(tf.matmul(output,weight_dropout_2), self._layer_biases_2)
        return output2

    def get_name(self):
        return self._name

    def calc_l2_loss(self):
        l2_loss = tf.nn.l2_loss(self._layer_weights_1)
        return l2_loss
