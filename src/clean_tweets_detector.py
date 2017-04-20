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

    for row in csv_file_object:  
            if len(row[0:])==1:
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

    return data

def create_tweets(data, target_folder, index):
    i = index

    for row in data:
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
print( "Cleaning:" + settings.dataset["neg_source"])
negdata = clean_tweets_detector(settings.dataset["neg_source"])
print ("Normal tweets: " + str(len(negdata)))
create_tweets(negdata, target_folder, nindex)

#sarcastic
target_folder = os.path.join(settings.rel_data_path, "pos") 
sindex = 1000000
if not (os.path.isdir(target_folder)):
    os.makedirs(target_folder)
print( "Cleaning: " + settings.dataset["pos_source"])
posdata = clean_tweets_detector(settings.dataset["pos_source"])
print ("Sarcastic tweets: " + str(len(posdata)))
create_tweets(posdata, target_folder, sindex)

#   return data, length

### POSITIVE DATA ####
#csv_file_object_pos = csv.reader(open('sarcastic_tweets.csv', 'rU'),delimiter='\n')
#pos_data, length_pos = preprocessing(csv_file_object_pos)


### NEGATIVE DATA ####
#csv_file_object_neg = csv.reader(open('normal_tweets.csv', 'rU'),delimiter='\n')
#neg_data, length_neg = preprocessing(csv_file_object_neg)

#print 'Number of  sarcastic tweets :', len(pos_data)
#print 'Average length of sarcastic tweets :', np.mean(length_pos)
#print 'Number of  non-sarcastic tweets :', len(neg_data)
#print 'Average length of non-sarcastic tweets :', np.mean(length_neg)

#np.save('posproc',pos_data)
#np.save('negproc',neg_data)



