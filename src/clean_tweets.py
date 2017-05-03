"""  This functions cleans all the tweets. In two ways: Either strict or non-strict. """

""" The strict version discards all tweets that start with a @ (i.e mention), and also discard all tweets that contain an url. Thereafter it replaces the sarcasm and sarcastic hashtags (#) with blank. Friendtags (@) and regular hashtags (#) are replaced with either a <tag> or with blank, depending on how the includetags variable is set in settings.py. Lastly we remove all duplicates and check if the tweet, after all replacing, is longer than 2, to get rid of short tweets. """

""" The non-strict version replaces the sarcasm and sarcastic hashtags (#) with blank. Friendtags (@), regular hashtags (#), urls (http) are replaced with either a <tag> or with blank, depending on how the includetags variable is set in settings.py. Lastly we remove all duplicates and check if the tweet, after all replacing, is longer than 2, to get rid of short tweets. """

import os
import csv
import re
import json
from collections import OrderedDict

from common_funs import Arg_handler
from common_funs import Progress_bar
from common_funs import Logger
from common_funs import Open_Dataset

from config import Config

def _arg_callback_ds(ds_name):
    cfg.dataset_name = ds_name
    print("<Using dataset: {}>".format(ds_name))

def _arg_callback_strict(strict = True):
    cfg.strict = True if str(strict).lower() == 'true' else False
    print("<Using strict: {}>".format(strict))

def _arg_callback_tags(include = True):
    cfg.includetags = True if str(include).lower() == 'true' else False
    print("<Including tags: {}>".format(include))

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

        # always remove hashtags
        temp=sarcasmtag.sub('', temp)

        if 'remove_tags' in restrictions:
            temp=friendtag.sub('', temp)
            temp=url.sub('', temp)
            temp=hashtags.sub('',temp)
        else:
            temp=friendtag.sub(cfg.tags[0], temp)
            temp=url.sub(cfg.tags[1], temp)
            temp=hashtags.sub(cfg.tags[2],temp)


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
    # write to table 'cleaned', 'wc' - flag means that samples of the same type
    #   will first be deleted (write-clean) so that old writes don't remain
    pb = Progress_bar( len(data)-1 )
    with Open_Dataset(ds_name, 'cleaned', 'wc', sample_class=s_class) as ds:
        for i, row in data:
            ds.write(i, row)
            pb.tick()


hashtags = re.compile(r'#[^\s.,;]*')
friendtag = re.compile(r'\S*@[^\s.,;]*')
retweet = re.compile(r'\A@|RT')
sarcasmtag = re.compile(r'#sarcasm|sarcastic\b', re.IGNORECASE)
url = re.compile(r'\bhttps?:\S+', re.IGNORECASE)
pattern_newline = re.compile(r"\n|\r")

logger = Logger()
decoder = json.JSONDecoder()

cfg = Config()
arghandler = Arg_handler()
arghandler.register_flag('ds', _arg_callback_ds, ['select-dataset', 'dataset'], "Which dataset to use. Args: <dataset-name>")
arghandler.register_flag('strict', _arg_callback_strict, [''], "If flag is set, clean the dataset with strict settings.")
arghandler.register_flag('tags', _arg_callback_tags, [''], "If flag is set, preserve tags")
arghandler.consume_flags()

#check if the database is initialized, if not, load the missing dataset
datasets_config = [
(cfg.dataset_name, cfg.pos_source_path, cfg.source_format, 1 ),
(cfg.dataset_name, cfg.neg_source_path, cfg.source_format, 0 )]
Open_Dataset.check_init_db(datasets_config, cfg.sqlite_file)

restrictions = []
if not cfg.includetags:
    restrictions.append('remove_tags')
if cfg.strict:
    restrictions.extend(['skip_url', 'skip_replies', 'skip_short'])

#normal
print("Cleaning normal tweets from: {}".format(cfg.dataset_name))
negdata = clean_tweets(cfg.dataset_name, 0, cfg.source_format)
print ("Normal tweets: " + str(len(negdata)))
write_clean(negdata, cfg.dataset_name, 0)

#sarcastic
print("Cleaning positive from: " + cfg.dataset_name)
posdata = clean_tweets(cfg.dataset_name, 1, cfg.source_format)
print ("Sarcastic tweets: " + str(len(posdata)))
write_clean(posdata, cfg.dataset_name, 1)

logger.save()
