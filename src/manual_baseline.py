import json
import html
import hashlib
from functools import partial

import numpy as np

import settings
import common_funs
from common_funs import chunks

######################### settings ###################################
names = ['adele', 'andreas','henry', 'jan', 'oscar', 'victor']
quiz_size = 10
tot_size = len(names) * quiz_size


########################### funs ######################################

def save_html(name, samples, ids_digest):
	quiz_key = "{"
	quiz_last = len(samples) - 1
	for i, sample in enumerate(samples):
		isSarc = 'true' if sample[2] else 'false'
		comma = ', ' if i < quiz_last else ''
		quiz_key += '"{}": {}{}'.format(sample[0], isSarc, comma)
		
	quiz_key += "}"

	html_header = """
		<html>
			<head>
				<meta charset='utf-8'>
				<title>Manual classification</title>
				<meta name='Manual tweet classification' content="sarcastic tweets">

				<script src='https://ajax.googleapis.com/ajax/libs/jquery/3.2.0/jquery.min.js'></script>
				<script>
					var ids_digest = '%s';
					var results;
					var quiz_key = JSON.parse('%s')
					if (ids_digest in localStorage) {
						results = JSON.parse(localStorage[ids_digest])	
						console.log("getting results from localstorage")	
					} else {
						results = {};
						console.log("quiz id not in localstorage")
					}
					
					$( document ).ready(function() {
						var keys = Object.keys(results)
						for (var key in keys) {
							key = keys[key]						
							var val = results[key]
							var name = (val == true) ? 'sarcastic' : 'neutral'						
							console.log("key: " + key + " name: " + name)
							$('#' + key).children('button[name =' + name + ']').addClass('selected')
						}
					});					
				</script>
				<style>
					.numb {
						display: inline-block;
						padding: 2px 2px;
						font-size: 24px;
						font-family: fantasy;
					}
					.tweet {
						font-size: 20px;
						font-family: sans-serif;
					}
					.button {
						background-color: #9e9e9e;
						border: none;
						color: white;
						padding: 10px 14px;
						text-align: center;
						text-decoration: none;
						display: inline-block;
						font-size: 16px;
						margin: 4px 2px;
						border-radius: 5px;
						cursor: pointer;
					}
					.button.selected {
						background-color: #9c4d4d;
					}
					#tally {
						padding: 10px 10px 10px 10px;
						margin: 10px;
						border-width: 2px;
						display: block;
					}
				</style>
			</head>
			<body>
		""" %(ids_digest, quiz_key)

	html_body = "<div>Your quiz id: {}</div></br>".format(ids_digest)

	for i,row in enumerate(samples):
		html_body += """
			<div class='numb'>{1:}. </div>
			<div id='{0:}' class='' class='tweet'>{2:}</br>
				<button class='button' name='sarcastic'>Sarcastic</button>
				<button class='button' name='neutral'>Neutral</button>
			</div>
			<hr>
			""".format(row[0], i, html.escape(" ".join(row[1])))
		

	html_body +="""
		<button id='tally'>Get Score</button>
		<a id='saveit' href='' download='results.json'>DOWNLOAD ANSWERS</a>		

		<script type='text/javascript'>
			$('.button').click(function() {
				var name = $(this).attr('name');
				var id = $(this).parent().attr('id');
				id = id.toString()

				if ( name == 'sarcastic' ) {
					results[id] = true;
				} else {
					results[id] = false;
				}

				localStorage[ids_digest] = JSON.stringify(results)
				console.log(id + ' ' + results[id])

				if ( !$(this).hasClass('selected') ) {
					$(this).addClass('selected');
				}

				$(this).siblings('.button:first').removeClass('selected');
			});

			$('a[download]').click(function(event) {
				if (Object.keys(results).length < $('.tweet').length) {
					alert("Please take note that you havn't answered all the questions!")
					//event.stopPropagation();
					//event.preventDefault()
				} 

				var answers = {
					name: '%s',
					quiz_id: '%s',
					answers: results,
					quiz_key: quiz_key
				}
				var output = JSON.stringify(answers, null, '\\n');
				var uri = 'data:application/csv;charset=UTF-8,' + encodeURIComponent(output);
				$(this).attr('href', uri);
				
			});

			$('#tally').click(function() {
				var total = 0
				var tp = 0, tn = 0, fp = 0, fn = 0

				for (var key in results) { 
					var actual = quiz_key[key]
					var answer = results[key]
					total += 1
					if ( actual == answer && actual) {
						tp += 1;
					} else if (actual == answer && !actual ) {
						tn += 1;
					} else if (actual != answer && actual) {
						fp += 1;
					} else if (actual != answer && !actual) {
						fn += 1;
					} else {
						throw "Should not fall through!";
					}
				}
				var accuracy = total > 0 ? ((tp + tn) / total) : 0
				var precision = (tp + fp) > 0 ? (tp / (tp + fp)) : 0
				var recall = (tp + fn) > 0 ? (tp / (tp + fn)) : 0
				var f1_score =  (precision + recall) > 0 ? 2*((precision * recall) / (precision + recall )) : 0 				
				accuracy = Math.round(accuracy * 100) / 100;
				f1_score = Math.round(f1_score * 100) / 100;

				answer_string = 'Accuracy: ' + accuracy + ', F1 Score: ' + f1_score + " "
				answer_string += ' ,Correct: ' + (tp + tn) + ' out of: ' + total

				console.log('total: '+ total)
				console.log(tn)
				alert(answer_string)


			});

		</script>
		""" %(name, ids_digest)

	html_footer = "</body></html>"
	html_doc = html_header + html_body + html_footer

	with open( name + '_brain_rnn.html', 'w', encoding='utf8' ) as f:
		f.write(html_doc)

############################### main ##########################################	
logger = common_funs.Logger()

with open( settings.debug_samples_path, 'r', encoding='utf8' ) as all_samples_file:
	samples = json.load( all_samples_file )

# create list of (id, [tokens]) touples for every tweet
samples_list = []
for key, val in samples.items():
	samples_list.append((key, val['text'], val['sarcastic']))


# create list of random row nubers from the samples list
random_rows = np.random.choice([x for x in range(0, len(samples_list) - 1)], tot_size)


# pick out the random samples
rand_samp = []
for row in random_rows:
	rand_samp.append(samples_list[row])
	logger.log(samples_list[row][0], 'sample_ids')

#iterator that yields list of sample-lists, of size: quiz_size
chunked = list(chunks(rand_samp, quiz_size))

# create a html document per human computer
print("Saving html for human computers: ")
for i, (name, pers_samps) in enumerate(zip(names, chunked)):
	m = hashlib.md5()
	#personal_samps = next(personal_itr) 

	# create a quiz id by hashing all the tweets ids used in the quiz
	sids = list(map(lambda ps: str(ps[0]) , pers_samps))
	callsaul = partial(bytes, encoding="utf8")
	bids = list(map(callsaul, sids))
	for bid in bids:
		m.update(bid)
	"""
	for ps in pers_samps:
		sid = str(ps[0])
		bid = bytes(sid, encoding="utf8")
		m.update(bid)
	"""
	digest = m.hexdigest()

	# log the quiz id for each human computer
	logger.log(digest, name, aslist=False)
	print(name, end=': ')
	print(digest, end='\n')
	save_html(name, pers_samps, digest)

logger.save("quiz_ids.json")




