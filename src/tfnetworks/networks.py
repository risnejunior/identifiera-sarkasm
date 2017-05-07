import tensorflow as tf

"""
This is the file containing the abstract class for the neural networks.
It also contains method for getting a specific network that is inheriting
of the abstract class
"""

class AbstractNetwork(object):

    # Method for feeding data through the network
    # Chunk_size and n_chunks is only necessary for Recurrent networks
    def feed_network(self,data,keep_prob,chunk_size=0,n_chunks=0):
        raise NotImplementedError("This method is not implemented")

    def calc_l2_loss(self):
        raise NotImplementedError("This method is not implemented")

    def get_name(self):
        raise NotImplementedError("This method is not implemented")

def calc_seqlenth(input):
    # this code is copied from TFLearn retrieve seqlenth method. Credited to it's creator @aymericdamien
    with tf.name_scope('GetLength'):
        used = tf.sign(tf.reduce_max(tf.abs(input), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
    return length
# This code is copied from TFLearn advanced_indexing_op() method. Credited to it's creator @aymericdamien
def advanced_indexing_op(input, index):
    batch_size = tf.shape(input)[0]
    max_length = int(input.get_shape()[1])
    dim_size = int(input.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (index - 1)
    flat = tf.reshape(input, [-1, dim_size])
    relevant = tf.gather(flat, index)
    return relevant
