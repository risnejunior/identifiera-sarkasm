import math


"""
Prints a pretty progress bar on every call to progress, or tick
"""
class Progress_bar:
	# To support unicde output in windows console type (e.g for double dash support):
	# chcp 65001 & cmd
	# Provide from and to values for the range that will be iterated, as well as 
	#  the size of the progress bar in chars.
	def __init__(self, iter_to, iter_from = 0, bar_max_len = 50 ):
		self.iter_to = iter_to
		self.i = self.iter_from = iter_from
		self.bar_max_len = bar_max_len
		self.spinner = 0

	
	# Update the progress bar by excplicitly providing the iteration step
	def progress( self, iteration ):
		eol = "\r"
		self.i = iteration
		percent = ( self.i / self.iter_to ) 
		bar_len = min( math.floor( percent * self.bar_max_len ), self.bar_max_len)
		s = self.spinner = self.spinner % 8
		if s == 0 or s == 4: spin_char = '|'
		if s == 1 or s == 5: spin_char = '/'
		if s == 2 or s == 6: spin_char = '-' # unicode double dash '\u2014'
		if s == 3 or s == 7: spin_char = '\\'
		if self.i >= self.iter_to: 
			spin_char = ''; 
			self.spinner = 0
			#eol="\r\n"
			#self.i = iter_from
			#bar_len = self.bar_max_len
			
		bar_len += 1
		bar = ("#" * bar_len) + spin_char
		bar_filler = " "*( self.bar_max_len - bar_len )
		print("Progress: [" + bar + bar_filler + "]" + '<{:3d} %>'.format( math.floor(percent * 100) ), end = eol)
		self.spinner += 1
		#os.system("pause")

	# Update the progress bar by an implicit step of 1
	def tick( self ):
		self.progress( self.i )
		self.i += 1
"""
Prints a confusion matrix and some other metrics for a given 
  binary classification
"""
def binary_confusion_matrix( ids, predictions, Ys):
	# positive means sarcastic, negative means normal
	# fn: false negative, fp: false positive,tp: true positive,tn: true negative 	
	fn = fp = tp = tn = 0 

	facit = list( zip( ids, predictions, Ys ) )
	for sample in facit:
		sample_id, predicted, actual = sample
		if predicted[0] < predicted[1]: # e.g (0.33, 0.77) predicted positive
			if actual[0] < actual[1]: #actual positive
				tp += 1
			else:
				fp += 1
		else: # predicted negative
			if actual[0] < actual[1]: #actual positive
				fn += 1
			else:
				tn += 1

	# format the table
	rows = ['' for i in range(10)]
	rows[0] = '{0:9}{1:^11}|{1:^11}'.format(' ', 'Predicted')
	rows[1] = '{0:12}{1:^8}|{2:^8}{3:^10}'.format('', 'No','Yes', 'total:')
	rows[2] = (' ' * 9) + ('-' * 20)
	rows[3] = '{:<20}{}'.format('Actual:','|')
	rows[4] = '{:^10}{:>9}{:^3}{:<9d}{:>}'.format("No", tn,'|', fp, (tn + fp) )
	rows[5] = rows[2]
	rows[6] = rows[3]
	rows[7] = '{:^10}{:>9}{:^3}{:<9d}{:>}'.format('Yes',fn , '|', tp, (fn + tp) )
	rows[8] = rows[2]
	rows[9] = '{:^12}{:^8} {:^9d}'.format('Total:', (tn + fn), (fp + tp) )

	print('Confusion Matrix:\n')
	for row in rows:
		print(row)

	#avoid division by zero
	count = len(predictions)
	accuracy = ( (tp + tn) / count ) if count > 0 else 0
	precision = ( tp / (tp + fp) ) if (tp + fp) > 0 else 0
	recall = ( tp / (tp + fn) ) if (tp + fn) > 0 else 0
	f1_score = 2*( (precision * recall) / (precision + recall ) ) \
		if  (precision + recall) > 0 else 0 

	print()
	print("accuracy: {:^1}{:<.2f}".format("",accuracy))
	print("precision: {:^}{:<.2f}".format("",precision))
	print("recall: {:^3}{:<.2f}".format("",recall))
	print("f1_score: {:^1}{:<.2f}".format("",f1_score))
