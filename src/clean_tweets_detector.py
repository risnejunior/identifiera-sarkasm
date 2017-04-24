"""  This functions cleans all the tweets """

import numpy as np
import os
import csv
import re
import settings





def clean_tweets_detector(source_name):
    data=[]
    hashtags = re.compile(r'#\w+\s?')
    friendtag = re.compile(r'@\w+\s?')
    sarcasmtag = re.compile(re.escape('sarcasm'),re.IGNORECASE)
    sarcastictag = re.compile(re.escape('sarcastic'),re.IGNORECASE)
    url = re.compile(r'\bhttp\b\S+')
    url2 = re.compile(r'\bhttps\b\S+')

    csv_file_object = csv.reader(open(source_name, 'rU'),delimiter='\n')
    next(csv_file_object)
    for row in csv_file_object:

        if len(row[0:])==1:



            if settings.dataset_name == "poria-balanced":
                temp=row[0:]
                temp = (temp[0].split('\t'))[2]
            else:
                temp=row[0:][0]


            temp=hashtags.sub(settings.tags[2],temp)

            if len(temp)>0 and temp[0]!='@' and r'\u' not in temp:

                temp=friendtag.sub(settings.tags[0], temp)
                temp=sarcasmtag.sub('', temp)
                temp=sarcastictag.sub('', temp)
                temp = url.sub(settings.tags[1], temp)
                temp = url2.sub(settings.tags[1], temp)
                temp=' '.join(temp.split()) #remove useless space

                # Check that tweet contains more than 3 words
                if len(temp.split())>2:
                    data.append(temp)

    data=list(set(data))
    #print ("Data length is: " + str(len(data)))
    return data

def create_tweets(data, target_folder, index):
    i = index

    for row in data:
       # print ("Index is: " + str(i))
        out_file_name = os.path.join(target_folder, str(i) + ".txt")
        out_file = open(out_file_name, 'w')
        out_file.write(row)
        out_file.close()
        i += 1

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
