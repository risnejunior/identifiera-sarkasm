run:
	python3 clean_tweets_detector.py
	python3 preprocess_data.py
	python3 tflearn_rnn.py --net little_pony

balanced:
	python3 clean_tweets_detector.py --ds poria-balanced
	python3 preprocess_data.py
	python3 tflearn_rnn.py --ds poria-balanced --net little_pony

ratio:
	python3 clean_tweets_detector.py --ds poria-ratio
	python3 preprocess_data.py
	python3 tflearn_rnn.py --ds poria-ratio --net little_pony

detector:
	python3 clean_tweets_detector.py --ds detector
	python3 preprocess_data.py
	python3 tflearn_rnn.py --ds detector --net little_pony
