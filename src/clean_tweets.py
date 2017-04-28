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
from collections import OrderedDict
from common_funs import Logger

def _arg_callback_ds(ds_name):
    """
    Select dataset
    """
    global dataset_name, dataset_proto
    dataset_name = ds_name
    dataset_proto['rel_path'] = datasets[ds_name]['rel_path']
    print("<Using dataset: {}>".format(ds_name))

def _arg_callback_strict():
    global strict, includetags
    strict = True
    includetags = False
    print("<Using strict cleaning>")

def clean_tweets_detector(source_name, sid_i):    
    ordered_data = OrderedDict()    
    skipped = dict(empty = 0, url = 0, reply = 0, short = 0, duplicate=0)
    
    csv_file_object = csv.reader(open(source_name, 'r', encoding='utf8'), 
                                delimiter='\t', 
                                quoting=csv.QUOTE_NONE,
                                quotechar=None,
                                strict=True,
                                escapechar=None,
                                doublequote=None)
    for row in csv_file_object:
        #skip empty rows
        if not row:
            skipped['empty'] += 1
            continue

        #poria has tweets id, 6 columns and shouldn't be unescaped
        #detector has no id, one column and should be unescaped
        if len(row) == 6:
            sid = row[0]
            temp = row[2]
            unescape = False
        elif len(row) == 1:
            sid = sid_i 
            temp = row[0]
            unescape = True
        else:
            raise Exception("Wrong number of columns in file")

        logger.log((sid, temp), logname='before',  maxlogs=5)

        if unescape:
            temp = temp.encode('unicode-escape').decode("utf8")
            temp = temp.replace('\\\\u', '\\u')
            temp = temp.replace(r'"', r'\"')
            temp = decoder.raw_decode('"'+ temp +'"')[0]

        if 'skip_url' in restrictions and url.search(temp):
            skipped['url'] += 1
            continue

        if 'skip_replies' in restrictions and retweet.search(temp):
            skipped['reply'] += 1
            continue

        if 'remove_tags' in restrictions:
            temp=friendtag.sub('', temp)
            temp=url.sub('', temp)
            temp=hashtags.sub('',temp)
        else:
            temp=friendtag.sub(settings.tags[0], temp)
            temp=url.sub(settings.tags[1], temp)
            temp=hashtags.sub(settings.tags[2],temp)

        # always remove hashtags
        temp=sarcasmtag.sub('', temp)
        
        if 'skip_short' in restrictions and len(temp.split()) < 3:  
            skipped['short'] += 1
            continue

        #whitespace
        temp = ' '.join(temp.split()) #whitespace
        logger.log((sid, temp), logname='after',  maxlogs=5)

        # the hash of the tweets is used as the key to remove duplicates
        sample_hash = hash(temp)
        if sample_hash not in ordered_data:
            ordered_data[sample_hash] = (sid, temp)
        else:
            skipped['duplicate'] += 1            
        sid_i += 1

    print("Tweets skipped/reason: " + str(skipped))
    return ordered_data.values()

def create_tweets(data, target_folder):
    pb = Progress_bar( len(data)-1 )
    for i, row in data:
        out_file_name = os.path.join(target_folder, str(i) + ".txt")
        out_file = open(out_file_name, 'w', encoding='utf8')
        out_file.write(row)
        out_file.close()
        pb.tick()

logger = Logger()
decoder = json.JSONDecoder()

hashtags = re.compile(r'#[^\s.,;]*')
friendtag = re.compile(r'\S*@[^\s.,;]*')
retweet = re.compile(r'\A@|RT')
sarcasmtag = re.compile(r'#sarcasm|sarcastic\b', re.IGNORECASE)
url = re.compile(r'\bhttps?:\S+', re.IGNORECASE)
pattern_newline = re.compile(r"\n|\r")

dataset_proto = datasets[dataset_name]
arghandler = Arg_handler()
arghandler.register_flag('ds', _arg_callback_ds, ['select-dataset', 'dataset'], "Which dataset to use. Args: <dataset-name>")
arghandler.register_flag('strict', _arg_callback_strict, [''], "If flag is set, clean the dataset with strict settings.")
arghandler.consume_flags()
dataset = settings.set_rel_paths(dataset_proto)
samples_path = dataset["samples_path"]

sid_i = 1 # id used, if not in file
restrictions = []
if not includetags: restrictions.append('remove_tags')
if strict: restrictions.extend(['skip_url', 'skip_replies', 'skip_short'])

#normal
target_folder = os.path.join(dataset["rel_path"], "neg")
if not (os.path.isdir(target_folder)):
    os.makedirs(target_folder)
print( "Cleaning:" + dataset["rel_path"]+"/"+dataset_proto["neg_source"])
negdata = clean_tweets_detector(dataset["rel_path"]+"/"+dataset_proto["neg_source"], sid_i)
sid_i = len(negdata) + 1
print ("Normal tweets: " + str(len(negdata)))
create_tweets(negdata, target_folder)

#sarcastic
target_folder = os.path.join(dataset["rel_path"], "pos")
if not (os.path.isdir(target_folder)):
    os.makedirs(target_folder)
print( "Cleaning: " + dataset["rel_path"]+"/"+dataset_proto["pos_source"])
posdata = clean_tweets_detector(dataset["rel_path"]+"/"+dataset_proto["pos_source"], sid_i)
print ("Sarcastic tweets: " + str(len(posdata)))
create_tweets(posdata, target_folder)
