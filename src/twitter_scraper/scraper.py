# -*- coding: utf-8 -*-

import tweepy
import sys
import os
import csv
import time
import json
import re
from pprint import pprint

############ usage #################################
# python scraper.py <input file>
#
# Input file is a text file with tweet ids, one id per row
# Will create a file called [input file name]_tweets.csv
# This file is tab delimited.
#
####################################################

### settings #######################################
batch_size = 100 #tweets per api call, tweepy limits this to 100
max_request_before_sleep = 75000 #requests before sleeping (for debugging)
max_requests = 75000 #max request before quiting (for debugging)
rate_timeout = 60*16 #(in seconds) how long to wait when rate limited (not used when tweepy handles it)
capacity_timeout = 10 #(in seconds) time before retry when twitter replys with capacity error
retry_limit = 4 #how many times to retry before failing
del_char = "\t" #delimiter used in csv-file
debug = False #turn on debugging prints

# twitter api keys
consumer_key = 'dB2uAZ8CwIqtM8tZtuxBhUzdO'
consumer_secret ='jQp9LokzjMF2UqvI4oMtdVofIgMXC7DbrANnHiEFb5FvLL7vAo'
access_token = '266787445-vDXRMXVdTAkmL4SsrVBADsxF5ulWyUvWo4Tq6iEW'
access_token_secret = 'tBRwFgl9Nn0tIIYbktv4YNnHsiMCkS0260o6oaN7ot9AQ'
####################################################

#message line
def status_message(total_read_ids, total_downloaded_tweets, status_string):
	print("Read ids: %i, Downloaded tweets: %i, Status: %s           " %(total_read_ids, total_downloaded_tweets, status_string) , end="\r")

if(debug): print (sys.version)

if len(sys.argv) == 2:
	source_name = sys.argv[1]
else:
	print("Please provide an input file as an argument")
	quit()

pattern_replace_text = re.compile(r"\n") #regex for removing newlines in tweets
total_downloaded_tweets = 0
total_read_ids = 0

#authenticate with twitter
if(debug): print( "Authenticating with twitter...", end="", flush=True)
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)
if(debug): print(" [Done]", flush=True)

#open files
source_file = open(source_name, 'r') #file containing tweet ids
dest_file = open('%s_tweets.csv' % source_name.split(".")[0], 'w', encoding='utf8', newline='')
writer = csv.writer(dest_file, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='', quotechar='' )
writer.writerow(["id","screen_name","tweet_text", "hashtag_indices", "user_mentions_indices","url_indices"])

while True:
	#Read from file containing tweet ids
	if(debug): print( "Reading tweet ids...", end="", flush=True)
	tweetids = source_file.readlines(18*batch_size) #tweet ids are 18 bytes
	tweetids = list(map (int, tweetids) ) #convert trings to ints
	nr_read_ids = len(tweetids)
	total_read_ids += nr_read_ids
	if(debug): print( " " + str( nr_read_ids ) + " read [Done]", flush=True)
	if nr_read_ids < 1:
		break

	#lookup tweets with tweet ids from text file
	if(debug): print( "Downloading tweets...", end="", flush=True)

	while True:
		retries = 0
		try:
			status_message(total_read_ids, total_downloaded_tweets, "Downloading tweets")
			downloaded_tweets = api.statuses_lookup(tweetids, True, False, False)
			break
		except tweepy.RateLimitError:
			status_message(total_read_ids, total_downloaded_tweets, "Rate limit exceeded, sleeping")
			time.sleep(rate_timeout)
		except tweepy.TweepError as e:
			if retries < retry_limit:
				status_message(total_read_ids, total_downloaded_tweets, "Unhandled exception, retrying")
				retries += 1
				time.sleep(capacity_timeout)
			else:
				status_message(total_read_ids, total_downloaded_tweets, "Retries expired, failing")
				raise

	total_downloaded_tweets += len(downloaded_tweets)
	status_message(total_read_ids, total_downloaded_tweets, "Writing to file")
	
	#extract relevant fields from tweet
	if(debug): print( "Extracting fields...", end="", flush=True)
	status_message(total_read_ids, total_downloaded_tweets, "Extracting fields")
	outtweets = []
	for tweet in downloaded_tweets:
		#replace newlines in text
		tweet_text = tweet.text
		tweet_text = re.sub(pattern_replace_text,' ', tweet_text)

		#extract hashtag indices
		entities = tweet.entities
		hashtag_list = (entities['hashtags']);		
		indices = []
		for obj in hashtag_list:
			indices.append(obj['indices'])
		hashtags = str(indices)
		
		#extract user mentions
		user_mentions_list = (entities['user_mentions']);
		indices = []
		for obj in user_mentions_list:
			indices.append(obj['indices'])
		user_mentions = str(indices)

		#extract url, media, symbols and extended_enteties indices
		urls_list = (entities['urls']);
		if "media" in entities: urls_list += entities['media']
		if "symbols" in entities: urls_list += entities['symbols']
		if 'extended_entities' in entities: urls_list += entities['extended_entities']
		indices = []
		for obj in urls_list:
			indices.append(obj['indices'])
		urls = str(indices)

		#build line to be written to csv
		outtweets.append([tweet.id, tweet.user.screen_name, tweet_text, hashtags, user_mentions, urls])

	if(debug): print(" [Done]", flush="True")

	#write tweets to file
	if(debug): print( "Writing tweets to csv...", end="", flush=True)
	status_message(total_read_ids, total_downloaded_tweets, "Writing to file")
	writer.writerows(outtweets)
	if ( total_downloaded_tweets % 1000 ): dest_file.flush() 	#flush every 1000 tweets
	if(debug): print(" [Done]", flush="True")

	if max_requests < total_downloaded_tweets: #for debugging
		break

source_file.close()
dest_file.close()
print("\n")
print("Finished. Out of %i tweet ids consumed, %i tweets downloaded" % (total_read_ids, total_downloaded_tweets))

# debugging cruff
#
##pprint(json.dumps(tweet.entities, sort_keys=True, indent=4, separators=(',', ': ')))
#screenName = tweet.user.screen_name #.encode("UTF-8", errors="strict")