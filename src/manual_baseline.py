import json
import html

import numpy as np

######################### settings ###################################
names = ['adele', 'andreas','henry', 'jan', 'oscar', 'viktor']
quiz_size = 10
tot_size = len(names) * quiz_size


########################### funs ######################################
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def save_html(name, samples):        
	html_header = """
		<html>
			<head>
				<script src='https://ajax.googleapis.com/ajax/libs/jquery/3.2.0/jquery.min.js'>
				</script>
				<script>
					$( document ).ready(function() {
						for (i = 0; i < localStorage.length; i++) {
							var key = localStorage.key(i)
							var val = localStorage[key]
							var name = (val == 'true') ? 'sarcastic' : 'neutral'						
							console.log(name)

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
				</style>
			</head>
			<body>
		"""
	html_body = ""

	for i,row in enumerate(samples):
		html_body += """
			<div class='numb'>{1:}. </div>
			<div id='{0:}' class='tweet'>{2:}</br>
				<button class='button' name='sarcastic'>Sarcastic</button>
				<button class='button' name='neutral'>Neutral</button>
			</div>
			<hr>
			""".format(row[0], i, html.escape(" ".join(row[1])))
		

	html_body +="""
		<a id='saveit' href='' download='results.json'>DOWNLOAD RESULTS</a>

		<script type='text/javascript'>
			$('.button').click(function() {
				var name = $(this).attr('name');
				var id = $(this).parent().attr('id');

				if ( name == 'sarcastic' ) {
					localStorage[id] = true;
				} else {
					localStorage[id] = false;
				}

				console.log(id + ' ' + localStorage[id])

				if ( !$(this).hasClass('selected') ) {
					$(this).addClass('selected');
				}

				$(this).siblings('.button:first').removeClass('selected');
			});

			$('a[download]').click(function(event) {
				if (localStorage.length < $('.tweet').length) {
					alert('Please answer all the questions before saving!')
					event.stopPropagation();
					event.preventDefault()
				} else {
					var output = JSON.stringify(localStorage, null, '\\n');
					var uri = 'data:application/csv;charset=UTF-8,' + encodeURIComponent(output);
					$(this).attr('href', uri);
				}
			});

		</script>
		"""

	html_footer = "</body></html>"
	html_doc = html_header + html_body + html_footer

	with open( name + '_brain_rnn.html', 'w', encoding='utf8' ) as f:
		f.write(html_doc)

############################### main ##########################################		
with open( 'all_samples.json', 'r', encoding='utf8' ) as all_samples_file:
	samples = json.load( all_samples_file )

samples_list = []
for key, val in samples.items():
	samples_list.append((key, val['text']))

random_rows = np.random.choice([x for x in range(0, len(samples_list) - 1)], tot_size)

rand_samp = []
for row in random_rows:
	rand_samp.append(samples_list[row])

#personal_samples = [l[i:i + quiz_size] for i in range(0, len(l), quiz_size)]
#personal_samples = list( chunks(random_samples, quiz_size))
from pprint import pprint

for i, name in enumerate(names):
	pers_samp = rand_samp[quiz_size*i:quiz_size*(i+1)]
	print(name, end=": ")
	print(len(pers_samp))
	print()
	save_html(name, pers_samp)



