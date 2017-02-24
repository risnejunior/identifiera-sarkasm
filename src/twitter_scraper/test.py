import csv
import tweepy
import time

for nr in range(1, 100):
  print("%i  KB downloaded!" %(nr) , end="\r")
  time.sleep(1)


try:
	raise tweepy.TweepError('A test')
except tweepy.TweepError as e:
	print(type(e))
	print( dir( e ));
	print (e.response)
	print (e.reason)
	print (e.args)
finally:
	quit()