import tensorflow as tf
import tflearn
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

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
						   64,
						   dropout=hyp.lstm.dropout,
						   dynamic=True,
						   name="lstm",
						   restore=restore)

		net = tflearn.fully_connected(net,
									  64,
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

	def little_pony(self, hyp, pd):
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

	def bidirectional(self, hyp, pd):
			restore = True
			net = tflearn.input_data([None, pd.max_sequence], dtype=tf.float32)
			net = tflearn.embedding(net, input_dim=pd.vocab_size,
									     output_dim=pd.emb_size,
									     name="embedding",
									     restore=restore)

			net = bidirectional_rnn(net,
									BasicLSTMCell(96),
									BasicLSTMCell(96),
							        dynamic=True)

			net = tflearn.fully_connected(net,
										  64,
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