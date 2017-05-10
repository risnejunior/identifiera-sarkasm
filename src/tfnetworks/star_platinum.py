import tensorflow as tf
from tensorflow.contrib import rnn
from . import networks as net
gru_cell_units = 128

class StarPlatinumNetwork(net.AbstractNetwork):
    def __init__(self,n_classes,rnn_size = 256,n_chunks=75):
        global gru_cell_units
        self._name = "star_platinum"
        self._hidden_layer_1 = {'weights': tf.Variable(tf.random_uniform([rnn_size,1024]),name = "weight1"),
                                'biases': tf.Variable(tf.random_uniform([1024]),name = "biases1")}

        self._hidden_layer_2 = {'weights': tf.Variable(tf.random_uniform([1024,n_chunks * 10]),name = "weight2"),
                                'biases': tf.Variable(tf.random_uniform([n_chunks * 10]),name = "biases2")}

        self._lstm_cell = rnn.BasicLSTMCell(rnn_size)
        self._gru_cell = rnn.GRUCell(gru_cell_units)
        self._output = {'weights': tf.Variable(tf.random_uniform([gru_cell_units,n_classes]),name = "weight3"),
                        'biases': tf.Variable(tf.random_uniform([n_classes]),name = "biases3")}

    def feed_network(self,data,keep_prob,chunk_size,n_chunks,dynamic):

        sequence_lengths = None
        if dynamic:
            sequence_lengths = net.calc_seqlenth(data if isinstance(data, tf.Tensor) else tf.stack(data))
        weight_dropout1 = tf.nn.dropout(self._hidden_layer_1['weights'], keep_prob)
        weight_dropout2 = tf.nn.dropout(self._hidden_layer_2['weights'], keep_prob)
        output_dropout = tf.nn.dropout(self._output['weights'], keep_prob)
        rnn_dropout1 = rnn.core_rnn_cell.DropoutWrapper(self._lstm_cell,output_keep_prob=keep_prob)
        rnn_dropout2 = rnn.core_rnn_cell.DropoutWrapper(self._gru_cell,output_keep_prob=keep_prob)
        batch_size = tf.shape(data)[0]

        #begin Calculations
        input_shape = data.get_shape().as_list()
        ndim = len(input_shape)
        axis = [1, 0] + list(range(2,ndim))
        data = tf.transpose(data,(axis))
        sequence = tf.unstack(data)
        lstm_outputs, states = rnn.static_rnn(rnn_dropout1, sequence, dtype=tf.float32, sequence_length = sequence_lengths)
        if dynamic:
            lstm_outputs = tf.transpose(tf.stack(lstm_outputs), [1, 0, 2])
            lstm_output = net.advanced_indexing_op(lstm_outputs, sequence_lengths)
        else:
            lstm_output = lstm_outputs[-1]

        layer1 = tf.add(tf.matmul(lstm_output,weight_dropout1),self._hidden_layer_1['biases'])
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.add(tf.matmul(layer1,weight_dropout2),self._hidden_layer_2['biases'])
        layer2 = tf.nn.relu(layer2)
        input_to_gru = tf.reshape(layer2,[batch_size,n_chunks,10])
        ndim = input_to_gru.get_shape().as_list()
        chunk_size = ndim[-1]
        input_to_gru = tf.transpose(input_to_gru,[1,0,2])
        input_to_gru = tf.reshape(input_to_gru,[-1,chunk_size])
        input_to_gru = tf.split(input_to_gru,n_chunks,0)

        gru_outputs, state = rnn.static_rnn(rnn_dropout2, input_to_gru, dtype=tf.float32, sequence_length = sequence_lengths)

        if dynamic:
            gru_outputs = tf.transpose(tf.stack(gru_outputs), [1, 0, 2])
            gru_output = net.advanced_indexing_op(gru_outputs, sequence_lengths)
        else:
            gru_output = gru_outputs[-1]

        output = tf.add(tf.matmul(gru_output,output_dropout),self._output['biases'])
        return output


    def calc_l2_loss(self):
        l2_loss = tf.nn.l2_loss(self._hidden_layer_1['weights'])
        l2_loss += tf.nn.l2_loss(self._hidden_layer_2['weights'])
        l2_loss += tf.nn.l2_loss(self._output['weights'])
        return l2_loss

    def get_name(self):
        return self._name
