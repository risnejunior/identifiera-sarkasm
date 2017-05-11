import sys
sys.path.append("../tfnetworks")
from . import networks
from .tflittle_pony import LittlePonyNetwork
from .tflittle_gru import LittleGruNetwork
from .tfbig_boy import BigBoyNetwork
from .star_platinum import StarPlatinumNetwork

def fetch_network(name=None,n_classes=2,params={}):
    if name == "little_pony":
        if 'rnn_size' in params:
            return LittlePonyNetwork(n_classes,params['rnn_size'])
        else:
            return LittlePonyNetwork(n_classes)

    if name == "little_gru":
        if 'rnn_size' in params:
            return LittleGruNetwork(n_classes,params['rnn_size'])
        else:
            return LittleGruNetwork(n_classes)

    if name == "big_boy":
        return BigBoyNetwork(n_classes)

    if name == "star_platinum":
        if 'n_chunks' in params:
            return StarPlatinumNetwork(n_classes,n_chunks = params['n_chunks'])
        else:
            return StarPlatinumNetwork(n_classes)


    else:

        return network.AbstractNetwork()
