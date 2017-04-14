from settings import *
import os
import random

file_lists = { 
	'negative': (path_name_neg, os.listdir(path_name_neg)),
	'positive': (path_name_pos, os.listdir(path_name_pos))
}

out_path = os.path.join(rel_data_path, 'cleaned.csv')
shuffle_buffer = []

with open(out_path, 'w', encoding='utf8') as out_file:
	for file_class, (class_path, file_list) in file_lists.items():
		print("\nReading %s files:" %file_class)
		for i, file_name in enumerate(file_list):
			print("File %s" %i, end="\r")
			in_path = os.path.join(class_path, file_name)
			with open(in_path, 'r', encoding='utf8') as in_file:
				text = in_file.read()
				file_id = file_name.split('.')[0]
				row = [dataset_name, file_id, file_class, text]
				row_text = '\t'.join(row)
				shuffle_buffer.append(row_text)

	print("\nshuffling rows..")
	random.shuffle(shuffle_buffer)
	print("Writing to file %s" %out_path)
	for i,row in enumerate(shuffle_buffer):
		print("Writing row %s" %i, end='\r')
		out_file.write(row + '\n')