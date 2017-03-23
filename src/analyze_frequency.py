import json
from collections import Counter, OrderedDict

import numpy as np

import common_funs
import settings

print("loading samples...")
with open( settings.samples_path, 'r', encoding='utf8' ) as samples_file:
	samples = json.load( samples_file )

sample_count = len(samples)

pos_label = np.array([0., 1.], dtype="float32")
neg_label = np.array([1., 0.], dtype="float32")

all_words = []

pos_words = []
neg_words = []

pos_corp = []
neg_corp = []

pos_ids = []
neg_ids = []

pos_labels = []
neg_labels = []

print('grouping data...')
pb = common_funs.Progress_bar(sample_count)
for i,(key, val) in enumerate(samples.items()):
	
	all_words.extend(val["int_vector"])

	if val["sarcastic"]:
		pos_words.extend(val["int_vector"])
		pos_corp.append(val["int_vector"])
		pos_ids.append(key)
		pos_labels.append(pos_label)
	else:
		neg_words.extend(val["int_vector"])
		neg_corp.append(val["int_vector"])
		neg_ids.append(key)
		neg_labels.append(neg_label)
	pb.tick()
	#print(key)
	#print(val["int_vector"])
	#print(val["text"])
	#print(val["sarcastic"])
	#print()
	#if i > 10: 
	#	break

all_w_count = len(all_words)
pos_w_count = len(pos_words)
pos_corp_count = len(pos_corp)

neg_w_count = len(neg_words)
neg_corp_count = len(neg_corp)


#all_words = dict(Counter(all_words))
#all_words = OrderedDict(sorted(all_words.items(), key=lambda t: t[1], reverse=True) )

#pos_words = dict(Counter(pos_words))
#pos_words = OrderedDict(sorted(pos_words.items(), key=lambda t: t[1], reverse=True) )

#neg_words = dict(Counter(neg_words))
#neg_words = OrderedDict(sorted(neg_words.items(), key=lambda t: t[1], reverse=True) )
print('Counting, sorting and normalizing word frequencies...')
tot_len = len(all_words) + len(neg_words) + len(pos_words)
pb = common_funs.Progress_bar(tot_len)
for words in [all_words, neg_words, pos_words]:
	print()
	d = dict(Counter(words))
	d = OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True) )
	for i, (word, count) in enumerate(d.items()):				
		#no need to compare all words to itself
		if all_words == words:
			d[word] = count/len(words)
		else:
			d[word] = count/len(words) - all_words[word]/len(all_words)
		words = d		
		print("word: {}, count: {}".format(word, d[word]))
		pb.tick()
		if i > 10: 
			break


#print(neg_labels)
#print()
#print(pos_labels)
