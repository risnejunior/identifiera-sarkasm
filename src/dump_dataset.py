"""
Creates a shuffled csv from the dataset selected
"""
######################################
dataset = 'poria-balanced'
dest_file_name = 'poria-balanced.csv'
sample_count = 11181
#######################################
import csv
import random
from common_funs import Open_Dataset
from common_funs import interleave


i = 0
dest_file = open(dest_file_name, 'wb') # 
#writer = csv.writer(dest_file, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='', quotechar='') #

#writer.writerow(["dataset","dataset_id", "class", "sample_text"])
pos =  Open_Dataset(dataset, 'cleaned', 'r', sample_class = 1).getRows()
neg =  Open_Dataset(dataset, 'cleaned', 'r', sample_class = 0).getRows()


rows = interleave(pos[:sample_count], neg[:sample_count])
random.shuffle(rows)
random.shuffle(rows)
random.shuffle(rows)

for row in rows:
	dest_file.write(
		("\t".join([
			str(row['dataset']), 
			str(row['sample_id']), 
			str(row['sample_class']), 
			str(row['sample_text'])
		]) + '\r\n').encode("utf8")
	)

	if i % 1000 == 0:
		row_text = "row: {}".format(i)
		print(row_text, end="\r")
	i += 1