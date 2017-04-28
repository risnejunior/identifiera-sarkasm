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
from common_funs import Open_Dataset

def _arg_callback_ds(ds_name):
    global dataset_name
    dataset_name = ds_name
    print("<Using dataset: {}>".format(ds_name))

def _arg_callback_strict():
    global strict, includetags
    strict = True
    includetags = False
    print("<Using strict cleaning>")

def clean_tweets(ds_name, s_class, s_format):
    ordered_data = OrderedDict()    
    skipped = dict(empty = 0, url = 0, reply = 0, short = 0, duplicate=0)

    for row in Open_Dataset(ds_name, 'raw', 'r', sample_class=s_class):
        temp = row['sample_text']
        sid = row['sample_id']
        logger.log((sid, temp), logname='before',  maxlogs=5)

        if s_format['unescape']:
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

    print("Tweets skipped/reason: " + str(skipped))
    return ordered_data.values()

def write_clean(data, ds_name, s_class):
    pb = Progress_bar( len(data)-1 )
    with Open_Dataset(ds_name, 'cleaned', 'w', sample_class=s_class) as ds:
        for i, row in data:
            ds.write(i, row)
            pb.tick()


hashtags = re.compile(r'#[^\s.,;]*')
friendtag = re.compile(r'\S*@[^\s.,;]*')
retweet = re.compile(r'\A@|RT')
sarcasmtag = re.compile(r'#sarcasm|sarcastic\b', re.IGNORECASE)
url = re.compile(r'\bhttps?:\S+', re.IGNORECASE) 

logger = Logger()
decoder = json.JSONDecoder()

arghandler = Arg_handler()
arghandler.register_flag('ds', _arg_callback_ds, ['select-dataset', 'dataset'], "Which dataset to use. Args: <dataset-name>")
arghandler.register_flag('strict', _arg_callback_strict, [''], "If flag is set, clean the dataset with strict settings.")
arghandler.consume_flags()

#check if the database is initialized, if not, load the missing dataset(s)
Open_Dataset.check_init_db()

restrictions = []
if not includetags: 
    restrictions.append('remove_tags')
if strict: 
    restrictions.extend(['skip_url', 'skip_replies', 'skip_short'])

# Description in settings of how the source file looks, here it's used to 
#  determine if the text should be unicode unescaped (detector)
source_format = settings.datasets[dataset_name]['source_format']

#normal
print("Cleaning normal from: " + dataset_name)
negdata = clean_tweets(dataset_name, 0, source_format)
print ("Normal tweets: " + str(len(negdata)))
write_clean(negdata, dataset_name, 0)

#sarcastic
print("Cleaning positive from: " + dataset_name)
posdata = clean_tweets(dataset_name, 1, source_format)
print ("Sarcastic tweets: " + str(len(posdata)))
write_clean(posdata, dataset_name, 1)

logger.save()