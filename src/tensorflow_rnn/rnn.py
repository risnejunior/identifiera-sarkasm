## The recurrent neural network implementation class

# importing tensorflow and rnn framework
import tensorflow as tf
from tensorflow.contrib import rnn

class RecurrentNeuralNetwork:
    # hidden_layers_number : Integer number for number of hidden Neuronsz
    def __init__(self, hidden_layers_number, neurons_in_hidden_list, input_neurons, output_neurons):
