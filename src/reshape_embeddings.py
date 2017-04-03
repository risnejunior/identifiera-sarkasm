import json
from collections import Counter, OrderedDict
import csv

from settings import *
from common_funs import Logger
from common_funs import working_animation
from common_funs import Progress_bar
from common_funs import MinMax
from random import triangular as rt


logger = Logger()

with open(vocabulary_path, 'r', encoding="utf8") as voc_file:
	vocabulary = json.load( voc_file )

found = notfound = masks = randvecs = nulls = 0
notfound_indexes = []
rs_embeddings = [[] for _ in range(vocabulary_size)]
minmax = MinMax()
wa = working_animation("Loading embeddings vocabulary", 50)
try:
	with open(emb_voc_path, 'r', encoding="utf8") as emb_file:
		for i, line in enumerate(emb_file):		
			(word, *vector) = line.split()
			vector = list(map(float, vector))
		
			if word in vocabulary:
				found += 1
				voc_index = vocabulary[word]
				logger.log(voc_index, "vocindexes", aslist=False)

				if voc_index == 1:
					masks += 1
					continue
				elif voc_index == 0:
					nulls += 1
					rs_embeddings[voc_index] = [0.0 for _ in range(embedding_size)]
					#rs_embeddings[voc_index] = [0]
				else:				
					rs_embeddings[voc_index] = vector
					minmax.add(vector)
			else:
				notfound += 1

			wa.tick("Scanned: {:,}, matching vocabulary: {:,}".format(i, found))
		wa.done()
		
	minval = minmax.get('min')
	maxval = minmax.get('max')
	print("Generating random word vectors for missing words")
	pb = Progress_bar(len(rs_embeddings) - 1)
	for i, vector in enumerate(rs_embeddings):
		if not vector:
			randvecs += 1
			rs_embeddings[i] = [
				round(rt(minval, maxval), 5) for _ in range(embedding_size)]
		pb.tick()

except IndexError as e:
	print(e)
finally:
	logger.save()


print("Words not found in vocabulary: {:,}".format(notfound))
print("Words found in vocabulary: {:,}".format(found))
print("Masked words skipped: {:,}".format(masks))
print("Null vectors written to embedding: {:,}".format(nulls))
print("Word vectors copied to embedding: {:,}".format(found - masks))
print("Random word vectors written to embedding: {:,}".format(randvecs))
print("Range for random embedding (min/max): {}".format(minmax.get()))
print("Total written: {:,}".format(randvecs + found - masks))

print("\nSaving embeddings to file...")
with open(embeddings_path, 'w', encoding='utf8', newline='') as out_file:
	csv_w = csv.writer(out_file, delimiter=',')
	csv_w.writerows(rs_embeddings)
