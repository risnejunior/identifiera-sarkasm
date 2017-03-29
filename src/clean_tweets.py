# -*- coding: utf-8 -*-

##############
# Removes mentions, hashtags, links and other media from tweets and creates
# a new file per tweet in target sub directore under 'cleanded' containing only the text.
#
# Usage example: python clean_tweets.py balanced_normal_tweets.csv normal
#

from pprint import pprint
import sys
import codecs
import re
import nltk
import string
from collections import Counter
from nltk.corpus import stopwords
import csv
import json
import os
import settings

# settings
####################
#tweets

if len(sys.argv) == 3:
	source_name = sys.argv[1]
	target_folder = os.path.join('cleaned', sys.argv[2])
	if not os.path.exists(target_folder):
		os.makedirs(target_folder)

# Downlaod nltk data
##########################
# Can also use nltk downlaod manager
# in python promt:
# import nltk
# nltk.download()
#
#nltk.download("stopwords")

exclude_tags = ["#sarcasm"]
pattern_indices = re.compile(r"(?:\[)([\d\s,]+?)(?:\])") #regex for finding indices
pattern_whitespace = re.compile(r"(\s{3}|\s{2})+") #regex for reducing whitespace
delete_this = str.maketrans(dict.fromkeys("1234567890:\?;@!&\#\,.()[]\'"))

def replace(original_text, indices_list ):
	if len(indices_list) == 0:		
		return original_text

	prev_last = 0
	list = []
	indices_list =sorted(indices_list,key=lambda l:l[1][0], reverse=False)
	for i in indices_list:
		if not i:
			continue
		label,toup = i
		first, last = toup
		list.append( original_text[prev_last:first] )
		#special case sarcastic hashtags
		if original_text[first:last].lower() not in exclude_tags:
			list.append( label )
			#print(label + ": " + original_text[first:last].lower())
		prev_last = last

	if prev_last != len(original_text):
		list.append( original_text[prev_last:] )
	return "".join(list)

def get_indices(list_string, label):
	indices_list = re.findall(pattern_indices, list_string)
	if len(indices_list) == 0:		
		return []
	converted = []
	for i in indices_list:
		a, b = map(int, i.split(","))
		converted.append([a, b])
	name_list = [label for _ in list_string]
	return list( zip(name_list, converted) )


def status_message( tweet_nr ):
	print("Tweets written: %i" %(tweet_nr) , end="\r")

# Index mapping
################
#x0: id	
#x1: screen_name
#x2: tweet_text	
#x3: hashtag_indices
#x4: user_mentions_indices
#x5: url_indices

def clean_tweets(target_folder, source_name):
	tweets = ""
	tweet_nr = 1
	with open(source_name, encoding='utf8') as f:
		next(f) #skip column header
		for line in f:
			col=line.split("\t")
			text = col[2]
			z = re.findall(pattern_indices, col[3] + col[4] + col[5])
			out_file_name = os.path.join(target_folder, col[0] + ".txt")
			out_file = open(out_file_name, 'w', encoding="utf8")
					
			indices_list = []
			indices_list += get_indices(col[3], "<hashtag>") #<hashtag> removed because to high accuracy
			indices_list += get_indices(col[4], "<user>") 
			indices_list += get_indices(col[5], "<url>") 
			text = replace(text, indices_list )

			tweets += (text)
			reduced = re.sub(pattern_whitespace,' ', text)
			out_file.write( reduced )
			out_file.close()
			status_message( tweet_nr )
			tweet_nr += 1


#normal
target_folder = os.path.join(settings.rel_data_path, "neg") 
if not (os.path.isdir(target_folder)):
	os.makedirs(target_folder)
print( "Cleaning:" + settings.dataset["neg_source"])
clean_tweets(target_folder, settings.dataset["neg_source"])
print()

#sarcastic
target_folder = os.path.join(settings.rel_data_path, "pos") 
if not (os.path.isdir(target_folder)):
	os.makedirs(target_folder)
print( "Cleaning: " + settings.dataset["pos_source"])
clean_tweets(target_folder, settings.dataset["pos_source"])