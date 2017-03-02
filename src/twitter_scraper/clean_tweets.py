# -*- coding: utf-8 -*-

##############
# Removes mentions, hashtags, links and other media from tweets and creates a
# new file containing only the text.
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

# settings
####################
debug = True
#
#

#sys.stdout = codecs.getwriter('utf8')(sys.stdout)

if len(sys.argv) == 2:
	source_name = sys.argv[1]
else:
	print ("Please provide a tweet file name argument")

# Downlaod nltk data
##########################
# Can also use nltk downlaod manager
# in python promt:
# import nltk
# nltk.download()
#
#nltk.download("stopwords")

pattern_indices = re.compile(r"(?:\[)([\d\s,]+?)(?:\])") #regex for finding indices
pattern_whitespace = re.compile(r"(\s{3}|\s{2})+") #regex for reducing whitespace
delete_this = str.maketrans(dict.fromkeys("1234567890:\?;@!&\#\,.()[]\'"))

def replace(index1, index2, mainstring, replacementstring):
    return mainstring.replace(mainstring[index1:index2], replacementstring)

# Index mapping
################
#x0: id	
#x1: screen_name
#x2: tweet_text	
#x3: hashtag_indices
#x4: user_mentions_indices
#x5: url_indices

dest_file = open('%s_cleaned.csv' % source_name.split(".")[0], 'a', encoding='utf8', newline='')
#writer = csv.writer(dest_file, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='', quotechar='' )
tweets = ""
if(debug): print( "Cleaning tweets..", end="", flush=True)
with open(source_name, encoding='utf8') as f:
	next(f) #skip column header
	for line in f:
		col=line.split("\t")
		text = col[2]
		z = re.findall(pattern_indices, col[3] + col[4] + col[5])
		for i in z:
			indices_tuple = i.split(",") 
			a = int ( indices_tuple[0] )
			b = int ( indices_tuple[1] )
			text = text[:a] + ' ' * len( text[a:b] ) + text[b:]

		#text = text.strip().translate(delete_this)
		#text = text.replace("  ", " ")
		#text = text.encode("utf8")
		#print( text)
		tweets += (text)
		reduced = re.sub(pattern_whitespace,' ', text)
		dest_file.write( reduced + "\n")

#lowers = tweets.lower()
#remove the punctuation using the character deletion step of translate

#pprint(json.dumps(t_table, sort_keys=True, indent=4, separators=(',', ': ')))
quit()
# tokenization etc
###########################

t_table = dict( (ord(char), None) for char in string.tweets )
no_punctuation = lowers.translate( t_table)

tokens = nltk.word_tokenize(no_punctuation)
filtered = [w for w in tokens if not w in stopwords.words('english')]

count = Counter(tokens)
print ( count.most_common(10) )
count = Counter(filtered)
print ( count.most_common(10) )
#print(filtered )
for word in filtered:
	dest_file.write( word + "\n")
	dest_file.flush() 	#flush every 1000 tweets


# TODO ##############
# Regenerate tweet files to get rid of unecessary escape chars
#