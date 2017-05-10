import math
import time
import os
import json
import inspect
import random
import pickle
from random import triangular as rt
import atexit
import sys
from collections import namedtuple
from os import listdir

import sqlite3
from prettytable import PrettyTable, ALL
from colorama import Fore, Back, Style
import numpy as np

from config import Config

class Bad_boys:
	def __init__(self, sqlite_file, gang):
		self._table_name = "troublemakers"
		self._samples = {}
		self._gang = gang

		if gang is None:
			self.update = self.load = self.save = self.close = self.do_nothing
		else:
			self._db = DB_Handler(sqlite_file)
			self.check_init_db(self._table_name)

	def do_nothing(self, *args, **kwargs):
		pass

	def check_init_db(self, table_name):
		if not self._db.exists(table_name):
			print("creating table: {}".format(table_name))
			self._db._c.executescript("""
				CREATE TABLE IF NOT EXISTS {tn} (
					`id`	INTEGER PRIMARY KEY AUTOINCREMENT,
					`gang`	TEXT,
					`sample_id`	INTEGER,
					`total`	INTEGER,
					`correct`	INTEGER,
					`trouble`	REAL
				);

				CREATE UNIQUE INDEX `sampleid_gang` ON {tn} (`sample_id` ,`gang` );
			""".format(tn = table_name))			
			self._db.commit()

	def load(self):
		""" 
			copies all troublemakes to samples with sample id as key
		"""
		rows = self._db.getRows(self._table_name, gang=self._gang )		
		for row in rows:
			self._samples[row['sample_id']] = {
				'total': row['total'],
				'correct': row['correct'],
				'trouble': row['trouble'],
				'gang': row['gang']
			}


	def update(self, ids, predictions, ys):
		facit = zip(ids, predictions, ys)
		samples = self._samples
		for sid, x, y in facit:
			if sid not in samples:
				samples[sid] = {
					'total': 0,
					'correct': 0,
					'trouble': 0.0,
					'gang': self._gang
				}

			# true if prediction in x the same as answer key in y
			if (0.5 >= x[1]) == (0.5 >= y[1]):
				samples[sid]['correct'] += 1
			samples[sid]['total'] += 1
			samples[sid]['trouble'] = round(1 - (samples[sid]['correct'] / samples[sid]['total']), 1)
		
	def save(self):
		for sid, sample in self._samples.items():
			self._db.insertRow(
				table_name = self._table_name, 
				mode ='REPLACE', 
				sample_id = sid, 
				total = sample['total'], 
				correct = sample['correct'], 
				trouble = sample['trouble'], 
				gang = sample['gang']
			
			)

		self._db.commit()

	def close(self):
		self._db.close()
		self._samples = {}

	def find(self, trouble):
		"""
		Return troublemakers up to the trouble percentage

		"""
		self._db._c.execute("SELECT sample_id FROM {} WHERE trouble > {}"
			.format(self._table_name, trouble))
		rows = self._db._c.fetchall()
		ids_filter = set()
		for row in rows:
			ids_filter.add(row['sample_id'])
		
		return ids_filter

	def print(self, trouble, out_path):
		"""
			unlike the other functions this takes for granted that
			  that the table names are troublemakers and cleaned
		"""
		self._db._c.executescript("""	
			SELECT DISTINCT
			troublemakers.sample_id,
			cleaned.sample_text,
			cleaned.sample_class

			FROM troublemakers
			LEFT JOIN cleaned
			ON troublemakers.sample_id = cleaned.sample_id

			WHERE troublemakers.trouble > 0
			AND troublemakers.gang = 'hells-angels'
		""".format(trouble, self._gang))
		rows = self._db._c.fetchall()
		print(len(rows))
		with open(out_path, 'w', encoding="utf8") as f:
			for i, row in enumerate(rows):
				f.write(row)
				if i > 5:
					break

	@staticmethod
	def _unit_test_1(sqlite_file):
		""" should print the gang member count """
		gang = 'og'
		bb = Bad_boys(sqlite_file, gang)
		trouble_filter = bb.find(0.0)
		print(len(trouble_filter))

	@staticmethod
	def _unit_test_2(sqlite_file):
		""" 
			Saves and updates gang activity, then saves again. Results 
			  should be all correct except 3,4; with each having troble = 0.5
		"""
		gang = 'og'
		bb = Bad_boys(sqlite_file, gang)
		bb._db.deleteRows(bb._table_name, gang=gang)


		# 1
		ids = [1, 2, 3, 4, 5]
		xs = [(0.3, 0.7), (0.3, 0.7), (0.3, 0.7), (0.9, 0.1), (0.5, 0.5)]
		ys = [(0.3, 0.7), (0.3, 0.7), (0.3, 0.7), (0.9, 0.1), (0.5, 0.5)]
		bb.update(ids, xs, ys)
		bb.save()

		# 2
		ids = ids
		xs = [(0.3, 0.7), (0.3, 0.7), (0.7, 0.3), (0.3, 0.7), (0.5, 0.5)]
		ys = ys
		bb.load()
		bb.update(ids, xs, ys)
		bb.save()		



class DB_backed_log:
	def __init__(self, sqlite_file, table_name, **cols):
		self._db = DB_Handler(sqlite_file)
		self._table_name = table_name
		self._row = {}
		self.check_init_db(table_name, **cols)

	def log(self, **cols):
		self._row.update(cols)

	def peek(self):
		return self._row

	def flush(self):
		self._db.insertRow(self._table_name, **self._row)
		self._db.commit()

	def check_init_db(self, table_name, **cols):
		if not self._db.exists(table_name):
			print("creating table: {}".format(table_name))
			self._db._c.executescript("""
				CREATE TABLE IF NOT EXISTS {} (
					`id`	INTEGER PRIMARY KEY AUTOINCREMENT,
					`time`	TEXT,
					`run_id`	TEXT,
					`network`	TEXT,
					`dataset`	TEXT,
					`samples_file`	TEXT,
					`val_acc`	REAL,
					`val_f1`	REAL,
					`best_acc`	REAL,
					`test_acc`	REAL,
					`test_f1`	REAL,
					`status`	TEXT
				);
			""".format(table_name))			
			self._db.commit()

class Open_Dataset:
	modes = {'wr':'write-replace', 'wc':'write-clean', 'r':'read'}

	@staticmethod
	def check_init_db(datasets_config, sqlite_file):
		"""
		Check if DB file has tables and some data, else create and populate tables 
		"""		

		#check if DB has tables raw & cleaned
		db = DB_Handler(sqlite_file)
		db._c.execute("SELECT COUNT(*) AS count FROM sqlite_master WHERE type='table' AND name='raw' OR name='cleaned';")
		count = db._c.fetchone()
		if count and count['count'] < 2:
			print("Adding tables to database")
			db._c.executescript("""
				CREATE TABLE IF NOT EXISTS raw (
				  id            integer NOT NULL PRIMARY KEY AUTOINCREMENT,
				  sample_id     integer,
				  sample_text   text,
				  sample_class  integer,
				  dataset       text,
				  
				  /* Keys */
				  CONSTRAINT sid_dataset
				    UNIQUE (sample_id, dataset)
				);

				CREATE TABLE IF NOT EXISTS cleaned (
				  id            integer NOT NULL PRIMARY KEY AUTOINCREMENT,
				  sample_id     integer,
				  sample_text   text,
				  sample_class  integer,
				  dataset       text,
				  
				  /* Keys */
				  CONSTRAINT sid_dataset
				    UNIQUE (sample_id, dataset)
				);
			""")
		else:
			print("Database present")

		#check if raw has at least one row for each dataset			
		from common_funs import Logger
		logger = Logger()
		sid_i = 1
		for ds_name, path, file_format, s_class in datasets_config:			
			db._c.execute("SELECT COUNT(*) AS count FROM raw WHERE dataset = '%s' AND sample_class = %s" %(ds_name, s_class))
			count = db._c.fetchone()
			if count and count['count'] < 1:
				print("no data for dataset {}, class: {}, fetching..".format(ds_name, s_class))
				import csv
				with open(path, 'r', encoding="utf8") as f:
					for line in csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE):						
						if not line: 
							continue

						logger.log(logname='format', maxlogs=10, text=str(file_format))
						if file_format['sample_id'] is not None:
							try:
								sid = int(line[file_format['sample_id']])
							except ValueError:
								continue
						else:
							sid = sid_i
							sid_i += 1

						sample_text = line[file_format['sample_text']]

						logger.log(logname=ds_name, maxlogs = 25, text=(sid, sample_text, s_class))
						db.insertRow('raw', dataset=ds_name, 
											sample_id= sid, 
											sample_class= s_class,
											sample_text=sample_text)
						#if sid % 5000 == 0: print("Rows: {}".format(str(sid)), end='\r')
						
		logger.save()
		db.close()

	def __init__(self, dataset, table, mode, **where):

		if mode not in Open_Dataset.modes:
			raise ValueError("mode not recognized")

		cfg = Config()
		self._db = DB_Handler(cfg.sqlite_file)
		self._dataset = dataset
		self._table = table
		self._mode = mode
		self._where = where
		self._rows = ()
		self._i = 0

	def write(self, sample_id, text):
		self._db.insertRow(self._table,
						   dataset = self._dataset,
						   sample_text=text, 
						   sample_id=sample_id,
						   **self._where)

	def _enter_wc(self):
		self._db.deleteRows(self._table, dataset = self._dataset, **self._where)
		return self

	def _enter_wr(self):
		return self

	def _enter_r(self):
		for row in self.getRows():
			yield row

	def getRows(self):
		where = {}
		
		# don't include dataset in where if dataset = all
		if (self._dataset).lower() != 'all':
			where = dict(dataset = self._dataset)

		where.update(self._where)
		return self._db.getRows(self._table, **where)

	def __enter__(self):
		if self._mode == 'r':
			return self._enter_r()
		elif self._mode == 'wr':
			return self._enter_wr()
		elif self._mode == 'wc':
			return self._enter_wc()
		else:
			raise ValueError('Mode not implemented')

	def __exit__(self, type, value, trackback):
		self._db.close()

	def __next__(self):
		if self._i < len(self._rows):
			row = self._rows[self._i]
			self._i += 1
			return row 
		else:			
			raise StopIteration

	def __iter__(self):
		self._rows = self.getRows()
		return self

class DB_Handler:

	def __init__(self, sqlite_file, row_factory = 'sqlite3-row'):
		self._conn = sqlite3.connect(sqlite_file)
		if 'sqlite3-row' == row_factory:
			self._conn.row_factory = sqlite3.Row
		self._c = self._conn.cursor()
		#atexit.register(self.c

	def setRowFactory(self):
		self._conn.row_factory = lambda cursor, row: row

	def close(self):
		#commit and close
		self._conn.commit()
		self._conn.close()

	def exists(self, table_name):
		self._c.execute("SELECT COUNT(*) AS count FROM sqlite_master WHERE type='table' AND name=:tn;", dict(tn=table_name))
		return self._c.fetchone()['count'] > 0


	def createTable(self, table_name):
		try:
			self._c.execute('CREATE TABLE {tn} (id INTEGER PRIMARY KEY)'
				.format(tn=table_name))
		except sqlite3.OperationalError:
			return False
		else:
			return True

	def addColumn(self, table_name, column_name, column_type):
		try:
			self._c.execute('ALTER TABLE {tn} ADD COLUMN {cn} {ct}'
				.format(tn=table_name, cn=column_name, ct=column_type))
		except sqlite3.OperationalError:
			return False
		else:
			return True

	def commit(self):
		self._conn.commit()

	def deleteRows(self, table_name, **where):
		whereSql = ""
		for i, clause in enumerate(where.keys()):		
			cmd =  'WHERE' if i == 0 else 'AND'
			whereSql += " {0:} {1:} = :{1:}".format(cmd, clause)

		self._c.execute("DELETE FROM {}{}"
			.format(table_name, whereSql), where)

	def getRows(self, table_name, **where):

		whereSql = ""
		for i, clause in enumerate(where.keys()):		
			cmd =  'WHERE' if i == 0 else 'AND'
			whereSql += " {0:} {1:} = :{1:}".format(cmd, clause)

		self._c.execute("SELECT * FROM {}{}"
			.format(table_name, whereSql), where)
		return self._c.fetchall()

	def insertRows(self, table_name, cols = [], values = [()]):
		columns = ','.join(cols)
		vals_ph = ','.join(['?' for _ in range(len(values[0]))])
		self._c.executemany('INSERT OR REPLACE INTO {tn} ({cols}) VALUES ({v_ph})'
			.format(tn=table_name, cols=columns, v_ph=vals_ph), values)

	def insertRow(self, table_name, mode = '', **vals):
		mode = mode.lower()
		if 'replace' == mode:
			mode_text = 'OR REPLACE'
		elif 'ignore' == mode:
			mode_text = "OR IGNORE"
		else:
			mode_text = ''

		columns = ','.join(vals.keys())
		vals_ph = ','.join(['?' for _ in range(len(vals))])
		self._c.execute('INSERT {rp} INTO {tn} ({cols}) VALUES ({v_ph})'
			.format(rp=mode_text, tn=table_name, cols=columns, v_ph=vals_ph), list(vals.values()))

class TroubleMakers:
	def __init__(self, ids= None, ys = None, do_nothing = False):
		self.datapoints = {}

		if do_nothing:
			self.init_points = self.increment = self.addNew = self.update = self.merge = self.do_nothing

		if ids and ys:
			self.datapoints = self.init_points(ids, ys)

	def do_nothing(self):
		pass

	def init_points(self, ids, ys):
		return {sid: Datapoint(sid, y[0] < y[1]) for sid, y in zip(ids, ys)}

	def increment(self, sid, isTrue):
		if sid in self.datapoints:
			if isTrue:
				self.datapoints[sid].correct += 1
			self.datapoints[sid].total += 1
		else:
			raise ValueError("Can't increment none existant key")

	def addNew(self, sid, label, correct, total):		
		newPoint = Datapoint(sid, label, correct, total)
		self.datapoints[sid] = newPoint

	def update(self, sid, correct, total):
		self.datapoints[sid].correct += correct		
		self.datapoints[sid].total += total		

	def merge(self, other):
		for key, val in other.datapoints.items():
			if key in self.datapoints:
				self.update(key, val.correct, val.total)
			else:
				self.datapoints[key] = val

	def tallyPredictions(self):
		Group = namedtuple("group", ['correct','total','length', 'ids']) #!not used
		tally = {}
		correct = 0
		total = 0
		for key, val in self.datapoints.items():
			#true_total = (val.correct, val.total)
			true_total = round(val.correct / val.total, 2)

			if true_total in tally:
				tally[true_total] += 1
			else:
				tally[true_total] = 1

		return tally.items()

	def __str__(self):
		points = []
		for key, val in self.datapoints.items():
			points.append(str(val))

		return ('\n').join(points)

	def __len__(self):
		return len(self.datapoints)


class Datapoint:
	def __init__(self, sid, label, correct = 0, total = 0):		
		self.sid = sid
		self.label = label
		self.correct = correct
		self.total = total

	def __str__(self):
			return "id: {0:}, label: {1:}, correct: {2:<d}, total: {3:}".format(self.sid, self.label, self.correct, self.total)

	def	__hash__(self):
			return hash(self.sid)

	def	__eq__(self, other):
			return slef.id == other.sid


class FileBackedCSVBuffer:

	def __init__(self, filename, directory = "", header = [], delimiter='\t', clearFile=False, padding=0):
		self._header = ["[{:<{padding}}]".format(head, padding=padding) for head in header]
		self._filebacked = []
		self._buffer = []
		self._delimiter = delimiter
		self._file_path = os.path.join(directory, filename)
		self._padding = padding

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

	def _padRows(self, rows):
		padded_lines = []
		for line in self._buffer:
			padded_line = []
			for col in line.split(self._delimiter):
				padded_line.append("{0:<{padding}}".format(col, padding=self._padding))
			padded_lines.append(self._delimiter.join(padded_line))

		return padded_lines

	def flush(self):
		with open(self._file_path, 'a', encoding='utf8') as out_file:
			self._buffer = self._padRows(self._buffer)
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

	def consume_flags(self):
		params = []
		for arg in reversed(self._args):
			if arg[0:2] == '--':
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

					# default params don't have to be set
					if (all_fpc >= len(params) >= all_fpc - def_fpc):
						callback(*params)
					else:
						print("Wrong number of parmaters for flag: {}".format(arg))
						quit()
					del params[:]
			else:
				params.insert(0, arg)

		if len(params) > 0:
			print("Arguments passed but not consumed: {}".foramt(params))
			print("Make sure you use '--' instead of '-' to denote flags")
			quit()

	def _printHelp(self, alias = None):
		print('Flags registered:')
		for flag, cb in self._flags.items():
			_, helptext = self._flags[flag]
			items = filter(lambda v: v[1] == flag, self._aliases.items())
			aliases = [k for k,v in items if k != flag]
			print('Flag: [--' + flag + ']', end=" ")
			print(',aliases: ' + str(aliases), end=" ")
			print(', description: ' + helptext, end="\n\n")
		quit()


class Stack(list):
	"""
	pop already defined in list which we inherit from
	"""
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

	# for debuffing, saves the settings for each logname
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
	
	positive means sarcastic, negative means normal
	fn: false negative, fp: false positive,tp: true positive,tn: true negative	
	"""

	def __init__(self, save_predictions = True):
		
		self.metrics = {}
		self.rows = []
		self._datapoints = {}

	def calc(self, ids, predictions, Ys, name = None):		
		
		tm = TroubleMakers(ids, Ys)

		fn = fp = tp = tn = 0
		facit = list( zip( ids, predictions, Ys ) )
		for sample in facit:
			predicted_true = False
			sample_id, predicted, actual = sample
			if predicted[0] < predicted[1]: # e.g (0.33, 0.77) 
				#predicted positive
				if actual[0] < actual[1]: #actual positive
					tp += 1
					predicted_true = True					
				else:
					fp += 1
			else: 
				#predicted negative
				if actual[0] < actual[1]: #actual positive
					fn += 1
				else:
					tn += 1
					predicted_true = True

			tm.increment(sample_id, isTrue = predicted_true)

		#avoid division by zero
		count = len(predictions)
		accuracy = ( (tp + tn) / count ) if count > 0 else 0
		precision = ( tp / (tp + fp) ) if (tp + fp) > 0 else 0
		recall = ( tp / (tp + fn) ) if (tp + fn) > 0 else 0
		f1_score = 2*((precision * recall) / (precision + recall )) if (precision + recall) > 0 else 0

		if name == None:
			name = 'default'

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
		
		if name in self._datapoints:
			self._datapoints[name].merge(tm)
		else:
			self._datapoints[name] = tm

	def _compile_table(self, name):
		# for readability
		metrics = self.metrics[name]
		fn = metrics['fn']; fp = metrics['fp']; tp = metrics['tp']; tn = metrics['tn']

		pt = PrettyTable([Fore.GREEN + name + Style.RESET_ALL,'Predicted NO','Predicted YES','Total'])
		pt.add_row(['Actual NO',tn,fp,tn+fp])
		pt.add_row(['Actual YES',fn,tp,fn+tp])
		pt.add_row(['Total',tn+fn,fp+tp,''])
		pt.hrules = ALL

		rows = ['' for i in range(6)]
		rows[0] = pt.get_string(padding_width=5)
		rows[1] = "Accuracy: {:^1}{:<.2f}".format("", metrics['accuracy'])
		rows[2] = "Precision: {:^}{:<.2f}".format("", metrics['precision'])
		rows[3] = "Recall: {:^3}{:<.2f}".format("",   metrics['recall'])
		rows[4] = "F1-score: {:^1}{:<.2f}".format("", metrics['f1_score'])
		rows[5] = ""

		self.rows.extend(rows)

	def print_tables(self):
		for row in self.rows:
			print(row)

	def save_predictions(self,
		filename = 'predictions.pickle', 
		directory = 'logs',
		sets = ['default'],
		update = True):

		tm = TroubleMakers()
		for s in sets:
			if s in self._datapoints:
				tm.merge(self._datapoints[s])


		file_path = os.path.join('.', directory, filename)
		if not (os.path.isdir(directory)):
			os.makedirs(directory)

		if update and os.path.isfile(file_path):
			with open(file_path, 'rb') as handle:
				saved = pickle.load(handle)
				tm.merge(saved)
		
		with open(file_path, 'wb') as handle:
			pickle.dump(tm, handle)

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
	"""
	Provides a nice work in progress animation.
	Use instead of Progressbar when size of the work isn't know beforehand.
	"""

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
	b36 = base36encode(int(datestr))
	name = "{}_{}_{}".format(b36,a,b)
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

def boxString(bare_string):
	str_len = len(bare_string)
	horiz_bar = "-" * (str_len + 4)
	vert_bar = '|'
	boxed_string = "{1:}{3:}{0:} {2:} {0:}{3:}{1:}".format(
		vert_bar, horiz_bar, bare_string, "\n")

	return boxed_string


def balance(a, b, fa):
	"""
	Returns a and b such that: a = fa * total, and: b = (1 - fa) * total, 
	  while making sure that: total < a + b
	"""
	if fa > 1 or fa < 0:
		raise ValueError("fa should be a float between 0 and 1")

	from math import floor, ceil
	fb = (1.0 - fa)  #fraction b
	m = min(a, b)

	assert_tot = lambda a, b: a + b == tot
	assert_frac = lambda fa, fb: fa + fb == 1

	if a <= b:
		tot = floor(a/fa)
		ao = a
		bo = tot-ao
	else: 
		tot = floor(b/fb)
		bo = b
		ao = tot-bo

	if  ao > a:
		tot -= ceil(tot * 1-(a/fa))
		ao = floor(tot * fa)
		bo = ceil(tot * fb)

	if bo > b:
		tot -= ceil(tot * 1-(b/fb))
		ao = floor(tot * fa)
		bo = ceil(tot * fb)


	if not assert_tot(ao,bo):
		raise Exception("Total wrong")

	if not assert_frac(fa,fb):
		raise Exception("Fraction wrong")

	return ao, bo

ProcessedData = namedtuple('ProcessedData',[
	'dataset',
    'embeddings',
	'vocab',
	'rev_vocab',
	'emb_size',
	'vocab_size',
	'max_sequence',
	'vocab_instances'
	]
)
Dataset = namedtuple('Dataset', ['train', 'valid', 'test'])
Setpart = namedtuple('Setpart', ['names', 'length', 'ids', 'xs','ys'])
pos_label = np.array([0., 1.], dtype="float32")
neg_label = np.array([1., 0.], dtype="float32")

class Struct:
	def __init__(self, **entries):
		self.__dict__.update(entries)

	def __eq__(self, other):
		return self.__dict__ == other.__dict__

def smaller(xs, ys): 
	if len(xs) < len(ys): 
		return (xs, ys) 
	else:
		return (ys, xs)

def interleave (xs, ys):
	"""
	Copies elements from xs and ys to out while trying to spread the elements
	  from the smaller list with the elements from the larger list as evenly
	  as possible. It does this by checking the fraction between their lengths
	  and then tries to keep this fraction while growing the out list.
	"""
	out = []
	of = l = s = 1
	small, large = smaller(xs, ys)
	f = len(small) / len(large)
	
	while len(small) + len(large) != 0:
		if len(small) == 0 and len(large) != 0:
			l += 1
			out.append(large.pop())
		elif len(large) == 0 and len(small) != 0:
			s += 1
			out.append(small.pop())
		elif of >= f:
			l += 1
			out.append(large.pop())
		elif of < f:
			s += 1
			out.append(small.pop())
		
		of = s / l


	return out

def clear_console():
	os.system('cls' if os.name == 'nt' else 'clear')

def file_selector(path, menuText = None):

	if menuText is None:
		menuText = "Choose a file from the list above"

	clear_console()	
	files = list(sorted(listdir(path), reverse=True))
	max_i = len(files)-1
	file = ''

	for i, file in enumerate(files):
		print("{}: {}".format(i, file))

	print(boxString(menuText))
	while True:
		index = input("Select file by typing the index: ")
		try:			
			index = int(index)
			if index < 0 or index > max_i:
				raise ValueException("Index not in list")
			
			file = files[index]
		except:
			print("Must be a number in the list, try again.")
		else:
			break

	return file