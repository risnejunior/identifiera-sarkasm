import sys
sys.path.append("../tfnetworks")
from . import networks
from . import tflittle_pony

def fetch_network(name=None,n_classes=2,params={}):
    if name == "little_pony":
        if 'rnn_size' in params:
            return tflittle_pony.LittlePonyNetwork(n_classes,params['rnn_size'])
        else:
            return tflittle_pony.LittlePonyNetwork(n_classes)

    else:

        return network.AbstractNetwork()
