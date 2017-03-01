# -*- coding: utf-8 -*-

##############
# Removes mentions, hashtags, links and other media from tweets and creates a
# new file containing only the text.
#

import sys
import codecs
import re
import nltk
import string
from collections import Counter
from nltk.corpus import stopwords
import csv

#sys.stdout = codecs.getwriter('utf8')(sys.stdout)

if len(sys.argv) == 2:
	source_name = sys.argv[1]

pattern_indices = re.compile(r"(?:\[)([\d\s,]+?)(?:\])") #regex for finding indices
delete_this = str.maketrans(dict.fromkeys("1234567890:\?;@!&#\,.()[]\'"))

def replace(index1, index2, mainstring, replacementstring):
    return mainstring.replace(mainstring[index1:index2], replacementstring)

#x0: id	
#x1: screen_name
#x2: tweet_text	
#x3: hashtag_indices
#x4: user_mentions_indices
#x5: url_indices

dest_file = open('%s_cleaned.csv' % source_name.split(".")[0], 'w', encoding='utf8', newline='')
#writer = csv.writer(dest_file, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='', quotechar='' )
tweets = ""
with open(source_name, encoding='utf8') as f:
	next(f)
	for x in f:
		x=x.split("\t")
		text = x[2]
		z = re.findall(pattern_indices, x[3] + x[4] + x[5])
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

lowers = tweets.lower()
#remove the punctuation using the character deletion step of translate
t_table = dict( (ord(char), None) for char in string.punctuation )
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


# TODO ##############
# Regenerate tweet files to get rid of unecessary escape chars
#