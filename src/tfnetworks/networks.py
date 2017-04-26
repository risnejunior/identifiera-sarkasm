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
