import math
import time
import os
import json
import inspect
import random
from random import triangular as rt
from prettytable import PrettyTable, ALL
from colorama import Fore, Back, Style
import atexit
import sys

import numpy as np

class FileBackedCSVBuffer:

	def __init__(self, filename, directory = "", header = [], delimiter='\t', clearFile=False):
		self._header = ["<" + head + ">" for head in header]
		self._filebacked = []
		self._buffer = []
		self._delimiter = delimiter
		self._file_path = os.path.join(directory, filename)

		if directory and not (os.path.isdir(directory)):
			os.makedirs(directory)

		if os.path.isfile(self._file_path):
			with open(self._file_path, 'r', encoding='utf8') as file:
				self._filebacked = file.readlines()

		self._rewrite(clearFile)
		atexit.register(self.flush)

	def write(self, cols):
		cols = [str(col) for col in cols]
		self._buffer.append(self._delimiter.join(cols))

	def flush(self):
		with open(self._file_path, 'a', encoding='utf8') as out_file:
			self._buffer = [line + "\n" for line in self._buffer]
			out_file.writelines(self._buffer)
			self._filebacked.extend(self._buffer)
			self._buffer.clear()

	def _rewrite(self, clearFile):
		with open(self._file_path, 'w', encoding='utf8') as file:
			if clearFile:
				self._filebacked.clear()
			self._buffer = [line + "\n" for line in self._buffer]
			self._filebacked.extend(self._buffer)

			if self._header:
				header = self._delimiter.join(self._header) + "\n"
				if not self._filebacked:
					self._filebacked.append(header)
				else:
					self._filebacked[0] = header

			file.writelines(self._filebacked)
			self._buffer.clear()

	def append(self, cols):

		#can't append to empty buffer, write instead
		if self._buffer:
			last = self._buffer.pop() + self._delimiter
			cols = self._delimiter.join([str(col) for col in cols])
			self._buffer.append( last + cols)
		else:
			self.write(cols)

	def replace(self, cols):

		#can't replace empty buffer, just write
		if self._buffer:
			_ = self._buffer.pop()

		self.write(cols)





class Arg_handler():
	"""
		class for handling command line arguments.
		Regsiter a flag and it's corresponding callback function.

	"""

	def __init__(self):
		self._args = sys.argv[1:]
		self._flags = {}
		self._aliases = {}
		self.register_flag('?', self._printHelp, ['help'], 'Display help')

	def register_flag(self, flag, callback, aliases = [], helptext = ""):
		self._flags[flag] = (callback, helptext)
		self._aliases[flag] = flag
		for alias in aliases:
			self._aliases[alias] = flag

		return self

	def consume_flags(self):
		params = []
		for arg in reversed(self._args):
			if arg[0:2] == '--':
				#print("isflag")
				arg = arg[2:]
				if arg not in self._aliases:
					print("Flag not registered: {}".format(arg))
					quit()
				else:
					flag = self._aliases[arg]
					callback, _ = self._flags[flag]
					argspecs = inspect.getargspec(callback)
					all_fpc = len(argspecs.args) if argspecs.args else 0 #hasattr(argspecs,'args')
					def_fpc = len(argspecs.defaults) if argspecs.defaults else 0 #hasattr(argspecs,'defaults')
					if flag == '?':
						all_fpc = 0 #print_help takes self, ignore that
					#print("params: {}, cb args {}, default: {}".format(len(params), all_fpc, def_fpc))
					if (all_fpc >= len(params) >= all_fpc - def_fpc):
						#print("correct params count")
						callback(*params)
					else:
						print("Wrong number of parmaters for flag: {}".format(arg))
						quit()
					del params[:]
			else:
				params.insert(0, arg)

		if len(params) > 0:
			print("Arguments passed but not consumed:")
			print(params)
			quit()

	def _printHelp(self, alias = None):
		print('Flags registered:')
		for flag, cb in self._flags.items():
			_, helptext = self._flags[flag]
			items = filter(lambda v: v[1] == flag, self._aliases.items())
			aliases = [k for k,v in items if k != flag]
			print('Flag: [--' + flag + ']', end=" ")
			print(',aliases: ' + str(aliases), end=" ")
			print(', description: ' + helptext)
		quit()

class Stack(list):
	def push(self, item):
		self.append(item)
	def isEmpty(self):
		return not self

class DebugLoop:
	"""
	Used to debug loops, when you want a loop to stop early.
	Instantiate with maxloops set to the max number of iterations,
	  and then feed the loop() method with the iterable.
	If maxloops is set to None it has no effect, loop() just returns the iterable.

	Example (will print 0, 1, 2, 3, 4):
		dl = DebugLoop(5)
		for x in dl.loop(range(10))
			print(x)
	"""

	def __init__(self, maxloops=None):

		if maxloops == None:
			self.loop = lambda itrble : itrble
		else:
			self.maxloops = maxloops

	def loop(self, itrble):
		loops = 0
		for x in itrble:
			if loops >= self.maxloops:
				raise StopIteration
			loops += 1
			yield x

class Hyper:
	"""
	All hail the Hyper! Any attempt to understand this code will be met
	 with the fiercest of resistance!

    Iterable of generators, values are acessed via dynmically generated attributes.
	Holds generators for all params provided, that yield a random value in the
	  range provided for the step count provided.
	 Values are accessed via attributes that hold structs to allow for double
	  dot notation.
	Values are updated on every call to next (or iteration step)
	"""

	def __init__(self, steps, **kwargs):

		self.gens = {} #holds the generators for every value

		new_attribs = {}
		for name, valdict in kwargs.items():
			self.gens[name] = {}
			structdict = {}

			for valname, (minval, maxval) in valdict.items():
				self.gens[name][valname] = self.generate(minval, maxval, steps)
				structdict[valname] = 0.0
			new_attribs[name] = Struct(**structdict)

		self.__dict__.update(new_attribs)
		#self.update_args()

	def _update_args(self):
		for name, valstruct in self.gens.items():
			for valname in valstruct.keys():
				setattr(self.__dict__[name], valname, next(self.gens[name][valname]))

	def generate(self, minval, maxval, steps):
		vals = [round(rt(minval, maxval), 5) for _ in range(steps)]
		for val in vals:
			yield val

	def get_hypers(self):
		hypers = {}
		for name, valstruct in self.gens.items():
			hypers[name] = {}
			for valname in valstruct.keys():
				hypers[name][valname] = getattr(self.__dict__[name], valname)
		return hypers

	def __iter__(self):
		return self

	def __next__(self):
		self._update_args()
		return self

class Struct:
	def __init__(self, **entries):
		self.__dict__.update(entries)

	def __eq__(self, other):
		return self.__dict__ == other.__dict__

class MinMax():

	def __init__(self):
		self.minval = None
		self.maxval = None

	def add(self, val):
		if not val:
			return
		elif hasattr(val, '__iter__'):
			cur_min = min(val)
			cur_max = max(val)
		else:
			cur_min = cur_max = val

		if self.minval == None:
			self.minval = self.maxval = cur_min
		else:
			self.minval =  cur_min if cur_min < self.minval else self.minval
			self.maxval =  cur_max if cur_max > self.maxval else self.maxval

	def get(self, which="touple"):

		if which == 'touple':
			return (self.minval, self.maxval)
		elif which == 'min':
			return (self.minval)
		elif which == 'max':
			return (self.maxval)
		else:
			raise ValueError("illegal parameter value")





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
		else:
			atexit.register(self.save)

	def do_nothing(*args, **kwargs):
		pass

	def log(self, text, logname = "default", maxlogs = None, step = 1, aslist=True):
		"""
		Allows logging into differently names dictoinaries that then can be
		  printed. Good for debugging loops as it allows limiting the output.
		If no logname is provided the 'default' log will be used.
		On the first use of a log the settings are set, i.e maxlogs and step,
		by default everything is logged.

		Keyword arguments:
		logname -- the name of the log to save to if you want to keep
	      separate logs, else evrything is logged to 'default'.
		maxlogs -- maximum kept in a given logname (or default), further entries
		  are diregarded.
		step -- if you only want to keep every n:th log entry
		list -- If true (default) logged data is saved in a numbered list, and
		  printed in order when calling print_log. When false data is saved as
		  an single overwriteable value, however maxlogs and step still affect
		  this value.
		"""

		if logname not in self.logs:
			self.logs[logname] = []
			self.config[logname] = {"maxlogs": maxlogs,
									"step": step,
									"aslist": aslist,
									"log_count": 0,
									"call_count": 0
									}

		c = self.config[logname]
		if c["call_count"] % c["step"] == 0:
			if c["maxlogs"] is None or c["log_count"] < c["maxlogs"]:
				if c['aslist']:
					self.logs[logname].append( (c["call_count"], text) )
				else:
					self.logs[logname] = text
				self.config[logname]["log_count"] += 1
		self.config[logname]["call_count"] += 1

	def items(self, logname = "default", count=None):
		"""returns an iterator over the logname"""
		if logname not in self.config:
			raise StopIteration
		elif not self.config[logname]['aslist']:
			yield self.logs[logname]
		else:
			for i, (index, text) in enumerate(self.logs[logname]):
				if count is not None and i >= count:
					 raise StopIteration
				yield "{}: {}".format(index, text)

	def save(self, file_name = None, directory="logs", append=False, log_name = None):
		"""
		Save the logs to a JSON-file. Freetext is saved to a separate file
		  with 'freetext_' prepended to the filename
		"""

		if file_name is None:
			callstack = inspect.stack()
			if len(callstack) < 2:
				caller = callstack[0][1]
			else:
				caller = inspect.stack()[1][1]
			file_name = str(os.path.basename(caller)) + ".log"

		if not (os.path.isdir(directory)):
			os.makedirs(directory)
		file_path = os.path.join(directory, file_name)

		open_for = 'a' if append else 'w'

		if log_name == None:
			logs = self.logs
		else:
			logs = self.logs[log_name]

		content = json.dumps(
			logs,
			ensure_ascii=False,
			indent=4,
			separators=( ',',': '))

		with open(file_path, open_for, encoding='utf8') as out_file:
			out_file.write(content)

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



class Binary_confusion_matrix:
	"""
	Prints a confusion matrix and some other metrics for a given binary classification
	"""
	# positive means sarcastic, negative means normal
	# fn: false negative, fp: false positive,tp: true positive,tn: true negative

	def __init__(self, ids = [], predictions = [], Ys = [], name=''):
		self.metrics = {}
		self.rows = []
		self.t = PrettyTable([Fore.GREEN + name + Style.RESET_ALL,'Predicted NO','Predicted YES','Total'])

	def calc(self, ids, predictions, Ys, name = None):

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

		#avoid division by zero
		count = len(predictions)
		accuracy = ( (tp + tn) / count ) if count > 0 else 0
		precision = ( tp / (tp + fp) ) if (tp + fp) > 0 else 0
		recall = ( tp / (tp + fn) ) if (tp + fn) > 0 else 0
		f1_score = 2*((precision * recall) / (precision + recall )) if (precision + recall) > 0 else 0

		if name == None:
			name = len(self.metrics.items())

		metrics = {
			"fn": fn,
			"fp": fp,
			"tp": tp,
			"tn": tn,
			"accuracy": round(accuracy, 3),
			"precision": round(precision, 3),
			"recall": round(recall, 3),
			"f1_score": round(f1_score, 3)
		}

		self.metrics[name] = metrics
		self._compile_table(name)

	def _compile_table(self, name):
		# for readability
		metrics = self.metrics[name]
		fn = metrics['fn']; fp = metrics['fp']; tp = metrics['tp']; tn = metrics['tn']

		self.t.add_row(['Actual NO',tn,fp,tn+fp])
		self.t.add_row(['Actual YES',fn,tp,fn+tp])
		self.t.add_row(['Total',tn+fn,fp+tp,''])
		self.t.hrules = ALL

		self.rows = ['' for i in range(5)]
		self.rows[0] = "Accuracy: {:^1}{:<.2f}".format("", metrics['accuracy'])
		self.rows[1] = "Precision: {:^}{:<.2f}".format("", metrics['precision'])
		self.rows[2] = "Recall: {:^3}{:<.2f}".format("",   metrics['recall'])
		self.rows[3] = "F1-score: {:^1}{:<.2f}".format("", metrics['f1_score'])
		self.rows[4] = ""

	def print_tables(self):
		print(self.t)
		for row in self.rows:
			print(row)


	def save(self,
			 filename = 'confusion.log',
			 directory = 'logs',
			 content = 'table'):

		if not (os.path.isdir(directory)):
			os.makedirs(directory)
		file_path = os.path.join('.', directory, filename)

		text = ""
		if content == 'table':
			text = "\n".join(self.rows)
		elif content == 'metrics':
			text = json.dumps(
				self.metrics,
				ensure_ascii=False,
				indent=4,
				separators=( ',',': ')
			)
		else:
			raise ValueError("invalid fileformat provided")

		with open(file_path, 'w', encoding='utf8') as out_file:
			out_file.write(text)


		"""
		file_path = os.path.join(directory, file_name)
			with open(file_path, 'w', encoding='utf8') as out_file:
				out_file.write(self.freetext)


		logger.log(accuracy, logname="accuracy")
		logger.log(recall, logname="recall")
		logger.log(precision, logname="precision")
		logger.log(f1_score, logname="f1_score")
		logger.save(file_name="matrix.log")
		"""



class Progress_bar:
	"""
	Prints a pretty progress bar on every call to progress, or tick.

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
		percent = ( self.i / self.iter_to ) if self.iter_to != 0 else 1
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

class Working_animation:

	def __init__(self, message, message_len = None):
		self.message = message
		self.step = 0
		self.toggle = False
		self.last_update = time.time()
		if message_len:
			self.init_len = message_len
		else:
			self.init_len = len(message)

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

			print("{0:<{m_len}} {1:<{width_b}}{2:}{3:>{width_a}}".format(
				message,"[",'.',"]", m_len=self.init_len, width_b=11-t, width_a=t)
				,end='\r', flush=True)
			self.step += 1

	def done(self, message = None):
		message = self.message if message == None else message
		print(message + ": [Done!]" + ' ' * (self.init_len + 10))

def reverse_lookup( index_vector, rev_vocabulary, ascii_console=False ):
	text = []
	for i in index_vector:
		word = rev_vocabulary[i]
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


def squared_error(ys, xs, fun = lambda x: x):
	"""
	Returns the mean squared error for function fun over values xs compared
	  to targets ys. If no function is provided xs are treated like
	  already computed values.
	"""
	err = 0.0
	n = len(xs)

	if n != len(ys):
		raise LookupError("xs and ys need to be the same length")

	for x,y in zip(xs, ys):
		err += (1.0 / n) * (y - fun(x)) ** 2.0
	return err

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
