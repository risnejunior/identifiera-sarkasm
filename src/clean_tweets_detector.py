"""  This functions cleans all the tweets. In two ways: Either strict or non-strict. """

""" The strict version discards all tweets that start with a @ (i.e mention), and also discard all tweets that contain an url. Thereafter it replaces the sarcasm and sarcastic hashtags (#) with blank. Friendtags (@) and regular hashtags (#) are replaced with either a <tag> or with blank, depending on how the includetags variable is set in settings.py. Lastly we remove all duplicates and check if the tweet, after all replacing, is longer than 2, to get rid of short tweets. """

""" The non-strict version replaces the sarcasm and sarcastic hashtags (#) with blank. Friendtags (@), regular hashtags (#), urls (http) are replaced with either a <tag> or with blank, depending on how the includetags variable is set in settings.py. Lastly we remove all duplicates and check if the tweet, after all replacing, is longer than 2, to get rid of short tweets. """

import os
import csv
import re
import settings
import json
from common_funs import Progress_bar
from common_funs import Arg_handler
from settings import *

def _arg_callback_ds(ds_name):
    """
    Select dataset
    """
    global dataset_name, dataset_proto
    dataset_name = ds_name
    dataset_proto['rel_path'] = datasets[ds_name]['rel_path']
    print("<Using dataset: {}>".format(ds_name))

def _arg_callback_strict():
    global strict, tags
    strict = True
    tags = ["<user>", "<url>", "<hashtag>"]
    print("<Using strict cleaning>")

def _arg_callback_poria():
    global strict, tags
    strict = False
    tags = [" ", " ", " "]
    print("<Using poria cleaning>")

def clean_tweets_detector(source_name):
    data=[]
    hashtags = re.compile(r'#\w+\s?')
    friendtag = re.compile(r'@\w+\s?')
    sarcasmtag = re.compile(r'#sarcasm\b', re.IGNORECASE)
    sarcastictag = re.compile(r'#sarcastic\b', re.IGNORECASE)
    #sarcasmtag = re.compile(re.escape('sarcasm'),re.IGNORECASE)
    #sarcastictag = re.compile(re.escape('sarcastic'),re.IGNORECASE)
    url = re.compile(r'\bhttp\b\S+')
    url2 = re.compile(r'\bhttps\b\S+')

    csv_file_object = csv.reader(open(source_name, 'r', encoding='utf8'),delimiter='\n')
    next(csv_file_object)


    if strict:
        for row in csv_file_object:
            if len(row[0:])==1:

                if dataset_name == "poria-balanced" or dataset_name == "poria-ratio":
                    temp=row[0:]
                    temp = (temp[0].split('\t'))[2]
                else:
                    temp=row[0:][0]

                try:
                    temp = json.loads('"'+temp+'"')
                except Exception:
                    pass

                if len(temp)>0 and temp[0]!='@' and 'http' not in temp and 'https' not in temp: #try with and without url

                    temp=friendtag.sub(settings.tags[0], temp)
                    temp=sarcasmtag.sub('', temp)
                    temp=sarcastictag.sub('', temp)
                    temp=hashtags.sub(settings.tags[2],temp)
                    temp=' '.join(temp.split()) #remove useless space

                    # Check that tweet contains more than 3 words
                    if len(temp.split())>2:
                        data.append(temp)

        data=list(set(data))
        return data
    else:
        for row in csv_file_object:

            if len(row[0:])==1:

                if dataset_name == "poria-balanced" or dataset_name == "poria-ratio":
                    temp=row[0:]
                    temp = (temp[0].split('\t'))[2]
                else:
                    temp=row[0:][0]

                try:
                    temp = json.loads('"'+temp+'"')
                except Exception:
                    pass

                if len(temp)>0:

                    temp=friendtag.sub(settings.tags[0], temp)
                    temp=sarcasmtag.sub('', temp)
                    temp=sarcastictag.sub('', temp)
                    temp=hashtags.sub(settings.tags[2],temp)
                    temp=url.sub(settings.tags[1], temp)
                    temp=url2.sub(settings.tags[1], temp)
                    temp=' '.join(temp.split()) #remove useless space

                    # Check that tweet contains more than 3 words
                    if len(temp.split())>2:
                        data.append(temp)

        data=list(set(data))
        return data

def create_tweets(data, target_folder, index):
    pb = Progress_bar( len(data)-1 )
    i = index

    for row in data:
       # print ("Index is: " + str(i))
        out_file_name = os.path.join(target_folder, str(i) + ".txt")
        out_file = open(out_file_name, 'w', encoding='utf8')
        out_file.write(row)
        out_file.close()
        i += 1
        pb.tick()

dataset_proto = datasets[dataset_name]

arghandler = Arg_handler()
arghandler.register_flag('ds', _arg_callback_ds, ['select-dataset', 'dataset'], "Which dataset to use. Args: <dataset-name>")
arghandler.register_flag('strict', _arg_callback_strict, [''], "If flag is set, clean the dataset with strict settings.")
arghandler.register_flag('poria', _arg_callback_poria, [''], "If flag is set, clean the dataset with poria settings.")
arghandler.consume_flags()

dataset = settings.set_rel_paths(dataset_proto)
samples_path = dataset["samples_path"]

#normal
target_folder = os.path.join(dataset["rel_path"], "neg")
nindex = 0
if not (os.path.isdir(target_folder)):
    os.makedirs(target_folder)
print( "Cleaning:" + dataset["rel_path"]+"/"+dataset_proto["neg_source"])
negdata = clean_tweets_detector(dataset["rel_path"]+"/"+dataset_proto["neg_source"])
print ("Normal tweets: " + str(len(negdata)))
create_tweets(negdata, target_folder, nindex)

#sarcastic
target_folder = os.path.join(dataset["rel_path"], "pos")
sindex = 1000000
if not (os.path.isdir(target_folder)):
    os.makedirs(target_folder)
print( "Cleaning: " + dataset["rel_path"]+"/"+dataset_proto["pos_source"])
posdata = clean_tweets_detector(dataset["rel_path"]+"/"+dataset_proto["pos_source"])
print ("Sarcastic tweets: " + str(len(posdata)))
create_tweets(posdata, target_folder, sindex)
