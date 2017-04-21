"""
This is the file containing the abstract class for the neural networks.
It also contains method for getting a specific network that is inheriting
of the abstract class
"""

class abstract_network(object):
    def feed_network(data,keep_prob):
        raise NotImplementedError("This method is not implemented")

    def calc_l2_loss():
        raise NotImplementedError("This method is not implemented")

    def get_name():
        raise NotImplementedError("This method is not implemented")
