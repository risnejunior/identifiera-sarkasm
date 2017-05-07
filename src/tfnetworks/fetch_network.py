import sys
sys.path.append("../tfnetworks")
from . import networks
from .tflittle_pony import LittlePonyNetwork
from .tflittle_gru import LittleGruNetwork
from .tfbig_boy import BigBoyNetwork

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

    else:

        return network.AbstractNetwork()
