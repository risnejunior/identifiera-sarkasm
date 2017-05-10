import tensorflow as tf
import tflearn
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

class NetworkNotFoundError(ValueError):
	pass

class Networks:

	def __init__(self):
		"""
		Add a network by defining a method that returns 'net'.
		Get it by calling get_network with the name and whatever params
		  that delight your heart.
		"""

		callables = [method for method in dir(self) if callable(getattr(self, method))]
		self.callables = [method for method in callables if method[0] != '_' and method[:2] != "get"]

	def get_network(self, name, params):
			if name in self.callables:
				return getattr(self, name)(**params)
			else:
				raise NetworkNotFoundError("Network not found: {}".format(name))


	def big_boy(self, hyp, pd):
		restore = True
		net = tflearn.input_data([None, pd.max_sequence], dtype=tf.float32)
		net = tflearn.embedding(net, input_dim=pd.vocab_size,
								     output_dim=pd.emb_size,
								     name="embedding",
								     restore=restore)

		net = tflearn.lstm(net,
						   512,
						   dropout=hyp.lstm.dropout,
						   weights_init='uniform_scaling',
						   dynamic=True,
						   name="lstm",
						   restore=restore)

		net = tflearn.fully_connected(net,
									  128,
									  activation='sigmoid',
									  regularizer='L2',
									  weight_decay=hyp.middle.weight_decay,
									  weights_init='uniform_scaling',
									  name="middle",
									  restore=restore)

		net = tflearn.dropout(net, hyp.dropout.dropout, name="dropout")
		net = tflearn.fully_connected(net,
									  2,
									  activation='softmax',
									  regularizer='L2',
									  weight_decay=hyp.output.weight_decay,
									  weights_init='uniform_scaling',
									  name="output",
									  restore=restore)
		net = tflearn.regression(net,
			                     optimizer='adam',
			                     learning_rate=hyp.regression.learning_rate,
		                         loss='categorical_crossentropy')
		return net

	def basic_pony(self, hyp, pd):
		net = tflearn.input_data([None, pd.max_sequence], dtype=tf.float32)
		net = tflearn.embedding(net, input_dim=pd.vocab_size,
								     output_dim=pd.emb_size,
								     name="embedding")
		net = tflearn.lstm(net,
						   32,
						   dynamic=False,
						   name="lstm")
		net = tflearn.fully_connected(net,
									  2,
									  activation='softmax',
									  name="output",
									  restore=True)
		net = tflearn.regression(net,
			                     optimizer='adam',
			                     learning_rate=hyp.regression.learning_rate,
		                         loss='categorical_crossentropy')
		return net

	def little_pony(self, hyp, pd):
		net = tflearn.input_data([None, pd.max_sequence], dtype=tf.float32)
		net = tflearn.embedding(net, input_dim=pd.vocab_size,
								     output_dim=pd.emb_size,
								     name="embedding")
		net = tflearn.lstm(net,
						   256,
						   dynamic=True,
						   name="lstm")
		net = tflearn.fully_connected(net,
									  2,
									  activation='softmax',
									  name="output",
									  restore=True)
		net = tflearn.regression(net,
			                     optimizer='adam',
			                     learning_rate=hyp.regression.learning_rate,
		                         loss='categorical_crossentropy')
		return net

	def pony_express(self, hyp, pd):
		"like little pony + dropout + more lstm nodes + restore on all layers"

		net = tflearn.input_data([None, pd.max_sequence], dtype=tf.float32)
		net = tflearn.embedding(net, input_dim=pd.vocab_size,
								     output_dim=pd.emb_size,
								     name="embedding",
								     restore=True)
		net = tflearn.lstm(net,
						   512,
						   dynamic=True,
						   name="lstm",				   
						   dropout=hyp.lstm.dropout,
						   restore=True)
		net = tflearn.fully_connected(net,
									  2,
									  activation='softmax',
									  name="output",
									  restore=True)
		net = tflearn.regression(net,
			                     optimizer='adam',
			                     learning_rate=hyp.regression.learning_rate,
		                         loss='categorical_crossentropy')
		return net

	def little_gru(self, hyp, pd):
		net = tflearn.input_data([None, pd.max_sequence], dtype=tf.float32)
		net = tflearn.embedding(net, input_dim=pd.vocab_size,
								     output_dim=pd.emb_size,
								     name="embedding")
		net = tflearn.gru(net,
						   256,
						   dynamic=True,
						   name="gru")
		net = tflearn.fully_connected(net,
									  2,
									  activation='softmax',
									  name="output",
									  restore=True)
		net = tflearn.regression(net,
			                     optimizer='adam',
			                     learning_rate=hyp.regression.learning_rate,
		                         loss='categorical_crossentropy')
		return net

	def bidirectional(self, hyp, pd):
			restore = True
			net = tflearn.input_data([None, pd.max_sequence], dtype=tf.float32)
			net = tflearn.embedding(net, input_dim=pd.vocab_size,
									     output_dim=pd.emb_size,
									     name="embedding",
									     restore=restore)

			net = bidirectional_rnn(net,
									BasicLSTMCell(256),
									BasicLSTMCell(256),
							        dynamic=True)

			net = tflearn.fully_connected(net,
										  128,
										  activation='sigmoid',
										  regularizer='L2',
										  weight_decay=hyp.middle.weight_decay,
										  name="middle",
										  restore=restore)

			net = tflearn.dropout(net, hyp.dropout.dropout, name="dropout")
			net = tflearn.fully_connected(net,
										  2,
										  activation='softmax',
										  regularizer='L2',
										  weight_decay=hyp.output.weight_decay,
										  name="output",
										  restore=restore)
			net = tflearn.regression(net,
				                     optimizer='adam',
				                     learning_rate=hyp.regression.learning_rate,
			                         loss='categorical_crossentropy')
			return net

	def convolve_me(self, hyp, pd):
		network = input_data(shape=[None, pd.max_sequence], name='input')
		network = tflearn.embedding(network,
									input_dim=pd.vocab_size,
								    output_dim=pd.emb_size,
								    name="embedding")
		branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
		branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
		branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
		network = merge([branch1, branch2, branch3], mode='concat', axis=1)
		network = tf.expand_dims(network, 2)
		network = global_max_pool(network)
		network = dropout(network, 0.5)
		network = fully_connected(network, 2, activation='softmax')
		network = regression(network, optimizer='adam', learning_rate=0.001,
		                     loss='categorical_crossentropy', name='target')
		return network
