import math
import time
import os 
import json
import inspect
import random

import numpy as np


class Logger:
	"""
	Convenience class that let's you log data and then print or save it to 
	  file. Has the option to only log every n:th input and only up to
	  a maximum count, in order to stave of overwhelm.
	"""
	
	save_config = False

	def __init__(self, enable = True):
		self.logs = {}
		self.config = {}
		self.freetext = ""
		Logger.enable = enable
		if not enable:
			self.write = self.log = self.save = self.print_log = self.do_nothing
	
	def do_nothing(*args, **kwargs):
		pass

	def log(self, text, logname = "default", maxlogs = None, step = 1):
		"""
		Allows logging into differently names dictoinaries that then can be
		  printed. Good for debugging loops as it allows limiting the output.
		"""

		if logname not in self.logs:
			self.logs[logname] = []
			self.config[logname] = {"maxlogs": maxlogs, 
									"step": step,
									"log_count": 0,
									"call_count": 0
									}
		
		c = self.config[logname]		
		if c["call_count"] % c["step"] == 0:
			if c["maxlogs"] is None or c["log_count"] < c["maxlogs"]:
				self.logs[logname].append( (c["call_count"], text) )	
				self.config[logname]["log_count"] += 1
		self.config[logname]["call_count"] += 1
	
	def items(self, logname = "default", count=None):
		"""returns an iterator over the logname"""

		for i, (index, text) in enumerate(self.logs[logname]):
			if count is not None and i >= count:
				 raise StopIteration
			yield "{}: {}".format(index, text)

	def write(self, text, newline=True):
		""" 
		Log free text
		"""
		
		self.freetext += text
		if newline:
			self.freetext += '\n'

	def save(self, file_name = None, directory="logs"):
		if file_name is None:
			caller = inspect.stack()[1][1]
			#caller = inspect.getmodule(frame[0])
			file_name = str(os.path.basename(caller)) + ".log"

		if not (os.path.isdir(directory)):
			os.makedirs(directory)
		file_path = os.path.join(directory, file_name)

		content = json.dumps(
			self.logs, 
			ensure_ascii=False, 
			indent=4, 
			separators=( ',',': '))
		with open(file_path, 'w', encoding='utf8') as out_file:
			out_file.write(content)

		if self.freetext != "":
			file_path = os.path.join(directory, "text_" + file_name)
			with open(file_path, 'w', encoding='utf8') as out_file:
				out_file.write(self.freetext)

		# Used for debugging the logger
		if Logger.save_config:
			content = json.dumps(
				self.config, 
				ensure_ascii=False, 
				indent=4, 
				separators=( ',',': '))
			with open('logs/config.log', 'w', encoding='utf8') as out_file:
				out_file.write(content)
	
	def print_log(self, logname = "default", count = None):
		for item in self.items(logname=logname, count=count):
			print(item)



def binary_confusion_matrix( ids, predictions, Ys):
	"""
	Prints a confusion matrix and some other metrics for a given binary classification
	"""
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

	logger = Logger()
	print('Confusion Matrix:\n')
	for row in rows:
		print(row)
		logger.write(row)


	#avoid division by zero
	count = len(predictions)
	accuracy = ( (tp + tn) / count ) if count > 0 else 0
	precision = ( tp / (tp + fp) ) if (tp + fp) > 0 else 0
	recall = ( tp / (tp + fn) ) if (tp + fn) > 0 else 0
	f1_score = 2*((precision * recall) / (precision + recall )) if (precision + recall) > 0 else 0 


	logger.log(accuracy, logname="accuracy")
	logger.log(recall, logname="recall")
	logger.log(precision, logname="precision")
	logger.log(f1_score, logname="f1_score")
	logger.save(file_name="matrix.log")

	#print additional metrics
	print()
	print("accuracy: {:^1}{:<.3f}".format("",accuracy))
	print("precision: {:^}{:<.3f}".format("",precision))
	print("recall: {:^3}{:<.3f}".format("",recall))
	print("f1_score: {:^1}{:<.3f}".format("",f1_score))
	print()


class Progress_bar:
	"""Prints a pretty progress bar on every call to progress, or tick.

	Keyword arguments:
	iter_to -- the first iteration value
	iter_from -- the last iterstion value
	bar_max_len -- the maximum length of the progressbar in chars
	update_freq -- max updates per second (prints to screen)
	
	Usage example:
	pb = Progress_bar(1000, 0, 50, 60)
	for i in range(1000)
		do_stuff()
		pb.progress(i)

	To support unicde output in windows console type (e.g for double dash support):
	 chcp 65001 & cmd
	"""

	# 1/8 parts expanding right
	bar_gradual = {	
		0: '\u258F',
		1: '\u258E',
		2: '\u258D',
		3: '\u258C',
		4: '\u258B',
		5: '\u258A',
		6: '\u2589',
		7: '\u2588'
		}

	# Fade in-out
	bar_blink = {	
		0: ' ',
		1: '\u2591',
		2: '\u2592',
		3: '\u2593',
		4: '\u2588',
		5: '\u2593',
		6: '\u2592',
		7: '\u2591'
		}	

	# Ascii
	bar_ascii = {	
		0: '|',
		1: '/',
		2: '-',
		3: '\\',
		4: '|',
		5: '/',
		6: '-',
		7: '\\'
		}

	def __init__(self, iter_to, iter_from = 0, bar_max_len = 56, update_freq = 25, bar_type = 'gradual' ):

		self.iter_to = iter_to
		self.i = self.iter_from = iter_from
		self.bar_max_len = bar_max_len
		self.spinner = 0
		self.last_update = time.time()
		self.update_freq = update_freq
		# select the bar to use
		if bar_type == 'blink':
			self.bar_parts = Progress_bar.bar_blink
		elif bar_type == 'gradual':
			self.bar_parts = Progress_bar.bar_gradual
		else:
			self.bar_parts = Progress_bar.bar_ascii

	def progress( self, iteration ):

		# Update the progress bar by excplicitly providing the iteration step
		self.i = iteration		
		if (self.last_update + ( 1 / self.update_freq ) > time.time() ) and self.i < self.iter_to: 
			return		
		percent = ( self.i / self.iter_to ) 
		#bar_len = min( math.floor( percent * self.bar_max_len ), self.bar_max_len)
		bar_len = min( percent * self.bar_max_len , self.bar_max_len )
		decpart = bar_len % 1
		bar_len = math.floor( bar_len )
		s = math.floor( decpart*8 )
		spin_char = self.bar_parts[s]

		if self.i >= self.iter_to: #last iteration step
			spin_char = ''; 
			self.spinner = 0
			eol="\r\n"
		else:
			eol = "\r"	
			
		bar = ("\u2588" * bar_len) + spin_char
		bar_filler = "\u2577"*( self.bar_max_len - bar_len - 1 ) 
		line = " {0:} Progress: {0:}{1:}{2:}{0:}{3:^8.1%}{0:}".format('\u2551', bar, bar_filler, percent)
		print( line, end=eol)
		self.spinner += 1
		self.last_update = time.time()

	# Update the progress bar by an implicit step of 1
	def tick( self ):
		self.progress( self.i )
		self.i += 1

class working_animation:

	def __init__(self, message):
		self.message = message
		self.step = 0
		self.toggle = False
		self.last_update = time.time()

	def tick(self, message = None):
		if (self.last_update + ( 1 / 25 )) > time.time(): 
			return
		else:
			self.last_update = time.time()		
			message = self.message if message == None else message
			 #toggles on every 20th iteration
			rem = (self.step % 10)
			if rem == 0:
				self.toggle = not self.toggle
			if self.toggle: 
				t = rem + 1
			else:
				t = 10 - rem

			print("{0:}: {1:<{width_b}}{2:}{3:>{width_a}}".format(
				message,"[",'.',"]", width_b=11-t, width_a=t), end='\r', flush=True)
			self.step += 1

	def done(self, message = None):
		message = self.message if message == None else message
		print(message + ": [Done!]" + ' ' * 10)

def reverse_lookup( index_vector, rev_vocabulary, ascii_console=False ):
	text = []
	for i in index_vector:
		word = rev_vocabulary[str(i)]
		if ascii_console: word = word.encode('unicode-escape')
		text.append( word )
	return text

def generate_name():
	t = time.localtime()
	a = random.choice(['blue', 'yellow', 'green', 'red', 'orange','pink','grey', 
		               'white', 'black', 'turkouse', 'fushia', 'beige','purple',
		               'rustic', 'idyllic', 'kind', 'turbo', 'feverish','horrid',
		               'master', 'correct', 'insane', 'relevant','chocolate',
		               'silk', 'big', 'short', 'cool', 'mighty', 'weak','candid',
		               'figting','flustered', 'perplexed', 'screaming','hip',
		               'glorious','magnificent', 'crazy', 'gyrating','sleeping'])
	b = random.choice(['battery', 'horse', 'stapler', 'giraff', 'tiger', 'snake', 
		               'cow', 'mouse', 'eagle', 'elephant', 'whale', 'shark',
		               'house', 'car', 'boat', 'bird', 'plane', 'sea','genius',
		               'leopard', 'clown', 'matador', 'bull', 'ant','starfish',
		               'falcon', 'eagle','warthog','fulcrum', 'tank', 'foxbat',
		               'flanker', 'fullback', 'archer', 'arrow', 'hound'])
	
	datestr = time.strftime("%m%d%H%M%S", t).encode('utf8')
	b32 = base36encode(int(datestr))
	name = "{}_{}_{}".format(b32,a,b)
	return name.upper()
 

def base36encode(integer):
    chars, encoded = '0123456789abcdefghijklmnopqrstuvwxyz', ''

    while integer > 0:
        integer, remainder = divmod(integer, 36)
        encoded = chars[remainder] + encoded

    return encoded

def normalize(xs): 
	min_xs = min(xs)
	max_xs = max(xs)
	ys = []
	
	if max_xs == min_xs:
		ys = np.random.randint(0, 1, size=2)
	else:
		for x in xs:
			y = ( x - min_xs ) / ( max_xs - min_xs )
			ys.append(y)

	return ys

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences: borrowed from tflearn

    Pad each sequence to the same length: the length of the longest sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the beginning or the end (default)
    of the sequence. Supports pre-padding and post-padding (default).

    Arguments:
        sequences: list of lists where each element is a sequence.
        maxlen: int, maximum length.
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    Returns:
        x: `numpy array` with dimensions (number_of_sequences, maxlen)

    Credits: From Keras `pad_sequences` function.
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x

