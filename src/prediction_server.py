import socketserver
import os
import pickle
from os import listdir
from operator import itemgetter
import re
import json
from urllib.parse import parse_qsl

import numpy as np
from nltk.tokenize import TweetTokenizer
import tensorflow as tf
import tflearn

from common_funs import pad_sequences
from common_funs import file_selector
from common_funs import ProcessedData
from common_funs import Dataset
from common_funs import Setpart
from common_funs import pos_label
from common_funs import neg_label
from common_funs import clear_console
from common_funs import boxString

from networks import Networks
from networks import NetworkNotFoundError

from config import Config


class PredictionHandler(socketserver.BaseRequestHandler):
    """
    The request handler used by the server.
    Instantiated once per connection to the server
    """

    def handle(self):
        self.data = self.request.recv(1024).strip()
        request = self.data.decode('utf-8')
        requestee = "Request from {}".format(self.client_address[0])
        print(boxString(requestee))
        print(request + "\n")

        query_line = re_query_line.findall(request)[0]
        qsl = dict(parse_qsl(query_line))
        
        errors = []
        if 'callback' not in qsl:
            callback_id = "error"
            errors.append("no callback in query")
        elif 'question' not in qsl:
            callback_id = qsl['callback']
            errors.append("No/empty question in query")
        else:
            question = qsl['question']
            callback_id = qsl['callback']
            tokens, prediction = predict(question)


        if errors:
            data = {'type': 'error', 'errors': errors}
        else:
            data = {'type': 'prediction', 'sarcasm': prediction, 'tokens': tokens}

        answer = json.dumps(data, separators=( ',',': '))
        response_body = "/**/{}({});".format(callback_id, answer)
        response_headers = [
            "HTTP/1.1 200 OK",
            "Content-Type: text/plain",
            "Content-Length: {}".format(len(response_body)),
            "Connection: close",
            ""]
        response_rows = response_headers + [response_body]
        response_message = "\n".join(response_rows)
        response_bytes = bytes(response_message.encode('utf8'))
        self.request.sendall(response_bytes)

        print(">" * 70)
        print(response_message)
        print("<" * 70)
        print("\n" * 3)



def build_network(name, hyp, pd):
    params = {'hyp': None, 'pd': pd}
    nets = Networks()
    net = nets.get_network(name=name, params=params)

    return net

def get_model_magic_path(path):
    """
    Return the path to the file that is last when alphabeticly sorted
    Gets a list of touple of (name, file) where name is the filename
     with magic nrs, but without extensions
    """
    best_name_path = None
    names_files = [(file.split('.')[0], file) for file in listdir(path) if file != "checkpoint"]

    if names_files:
        best_name_file = sorted(names_files, reverse=True, key = itemgetter(0))[0]
        best_name = best_name_file[0]
        best_name_path = os.path.join(path, best_name)

    return best_name_path

def _predict(input_string):
    tokens = input_string.split()
    return  (tokens, 0.5)

def predict(input_string):
    mask = lambda w, v: 1 if w not in v else v[w] 
    tknzr = TweetTokenizer(reduce_len=True, preserve_case=False)

    words = tknzr.tokenize(input_string)
    vec = [[mask(w, pd.vocab) for w in words]]
    vec = np.array( vec, dtype="int32")
    vec = pad_sequences(vec, maxlen=pd.max_sequence)
    predictions = model.predict(vec)
    sarcasm = round(predictions[0][1], 2) * 100

    return (words, sarcasm)

##################################################################
cfg = Config()
cfg.pretrained_id = "8F41L3_BIG_COW"
cfg.network_name = "little_pony"
cfg.ps_file_name = "tags.pickle"
cfg.dataset_name = "poria-balanced"
HOST, PORT = "192.168.0.6", 1024
re_query_line = re.compile(r"\?(\S+)")

with open(cfg.samples_path, 'rb') as handle:
    pd = pickle.load( handle )

net = build_network(cfg.network_name, None, pd)
model = tflearn.DNN(net)
this_run_id = cfg.pretrained_id
path = os.path.join(cfg.models_path, cfg.pretrained_id)
magic_path = get_model_magic_path(path)
model.load(magic_path)
clear_console()

if __name__ == "__main__":
    server = socketserver.TCPServer((HOST, PORT), PredictionHandler)
    # will keep running until you interrupt the program with Ctrl-C
    print("Ready to serve dank tweets..\n")
    server.serve_forever()