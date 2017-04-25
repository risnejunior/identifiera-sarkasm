"""  This functions cleans all the tweets. In two ways: Either strict or non-strict. """ 

""" The strict version discards all tweets that start with a @ (i.e mention), and also discard all tweets that contain an url. Thereafter it replaces the sarcasm and sarcastic hashtags (#) with blank. Friendtags (@) and regular hashtags (#) are replaced with either a <tag> or with blank, depending on how the includetags variable is set in settings.py. Lastly we remove all duplicates and check if the tweet, after all replacing, is longer than 2, to get rid of short tweets. """

""" The non-strict version replaces the sarcasm and sarcastic hashtags (#) with blank. Friendtags (@), regular hashtags (#), urls (http) are replaced with either a <tag> or with blank, depending on how the includetags variable is set in settings.py. Lastly we remove all duplicates and check if the tweet, after all replacing, is longer than 2, to get rid of short tweets. """

import os
import csv
import re
import settings
import json
from common_funs import Progress_bar

def clean_tweets_detector(source_name):
    data=[]
    hashtags = re.compile(r'#\w+\s?')
    friendtag = re.compile(r'@\w+\s?')
    sarcasmtag = re.compile(re.escape('sarcasm'),re.IGNORECASE)
    sarcastictag = re.compile(re.escape('sarcastic'),re.IGNORECASE)
    url = re.compile(r'\bhttp\b\S+')
    url2 = re.compile(r'\bhttps\b\S+')

    csv_file_object = csv.reader(open(source_name, 'r'),delimiter='\n')
    next(csv_file_object)


    if settings.strict:
        for row in csv_file_object:

            if len(row[0:])==1:

                if settings.dataset_name == "poria-balanced" or settings.dataset_name == "poria-ratio":
                    temp=row[0:]
                    temp = (temp[0].split('\t'))[2]
                else:
                    temp=row[0:][0]

                try:
                    temp = json.loads('"'+temp+'"')
                except Exception:
                    pass

                temp=hashtags.sub(settings.tags[2],temp)

                if len(temp)>0 and temp[0]!='@' and 'http' not in temp and 'https' not in temp:

                    temp=friendtag.sub(settings.tags[0], temp)
                    temp=sarcasmtag.sub('', temp)
                    temp=sarcastictag.sub('', temp)
                    temp=' '.join(temp.split()) #remove useless space

                    # Check that tweet contains more than 3 words
                    if len(temp.split())>2:
                        data.append(temp)

        data=list(set(data))
        return data
    else:
        for row in csv_file_object:

            if len(row[0:])==1:

                if settings.dataset_name == "poria-balanced" or settings.dataset_name == "poria-ratio":
                    temp=row[0:]
                    temp = (temp[0].split('\t'))[2]
                else:
                    temp=row[0:][0]

                try:
                    temp = json.loads('"'+temp+'"')
                except Exception:
                    pass

                temp=hashtags.sub(settings.tags[2],temp)

                if len(temp)>0:

                    temp=friendtag.sub(settings.tags[0], temp)
                    temp=sarcasmtag.sub('', temp)
                    temp=sarcastictag.sub('', temp)
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

#normal
target_folder = os.path.join(settings.rel_data_path, "neg")
nindex = 0
if not (os.path.isdir(target_folder)):
    os.makedirs(target_folder)
print( "Cleaning:" + settings.path_neg)
negdata = clean_tweets_detector(settings.path_neg)
print ("Normal tweets: " + str(len(negdata)))
create_tweets(negdata, target_folder, nindex)

#sarcastic
target_folder = os.path.join(settings.rel_data_path, "pos")
sindex = 1000000
if not (os.path.isdir(target_folder)):
    os.makedirs(target_folder)
print( "Cleaning: " + settings.path_pos)
posdata = clean_tweets_detector(settings.path_pos)
print ("Sarcastic tweets: " + str(len(posdata)))
create_tweets(posdata, target_folder, sindex)
