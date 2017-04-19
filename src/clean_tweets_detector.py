"""  This functions cleans all the tweets.
It first removes all the #tags, then make sure the tweets
does not contain http links, non ASCII charaters or that the
first letter of the tweet is @ (to ensure that the tweet is not out of context).
Then it removes any @tagging and any mention of the word sarcasm or sarcastic."""

import numpy as np
import os
import csv
import re
import settings

def clean_tweets_detector(target_folder, source_name, index):
    i = index

    hashtags = re.compile(r'#\w+\s?')
    friendtag = re.compile(r'@\w+\s?')
    sarcasmtag = re.compile(re.escape('sarcasm'),re.IGNORECASE)
    sarcastictag = re.compile(re.escape('sarcastic'),re.IGNORECASE)   
    url = re.compile(r'\bhttp\b\S+')   
    csv_file_object = csv.reader(open(source_name, 'rU'),delimiter='\n')

    for row in csv_file_object:
        out_file_name = os.path.join(target_folder, str(i) + ".txt")

        if i == 2 or i == 100002:
            print (out_file_name)
            break            

        if len(row[0:])==1:
            temp=row[0:][0]
            temp=hashtags.sub('<hashtag>',temp)

            if len(temp)>0 and temp[0]!='@': 
                temp=friendtag.sub('<user>', temp)
                temp=sarcasmtag.sub('<hashtag>', temp)
                temp=sarcastictag.sub('<hashtag>', temp)
                temp = url.sub('<url>', temp)
                temp=' '.join(temp.split()) #remove useless space

                #if len(temp)>0:
                out_file = open(out_file_name, 'w')
                out_file.write(temp)  # python will convert \n to os.linesep
                out_file.close()  # you can omit in most cases as the destructor will call it 
                i += 1


#normal
target_folder = os.path.join(settings.rel_data_path, "neg") 
nindex = 0
if not (os.path.isdir(target_folder)):
    os.makedirs(target_folder)
print( "Cleaning:" + settings.dataset["neg_source"])
clean_tweets_detector(target_folder, settings.dataset["neg_source"], nindex)

#sarcastic
target_folder = os.path.join(settings.rel_data_path, "pos") 
sindex = 100000
if not (os.path.isdir(target_folder)):
    os.makedirs(target_folder)
print( "Cleaning: " + settings.dataset["pos_source"])
clean_tweets_detector(target_folder, settings.dataset["pos_source"], sindex)

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



