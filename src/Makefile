train:
	python3 tflearn_rnn.py

test:
	python3 tflearn_rnn.py --ds poria-ratio --in ratio.pickle --pt --pretrained 71PVGG_BIG_CLOWN

# CLEAN NORMAL DATASET #
balanced:
	python3 clean_tweets_detector.py --ds poria-balanced
	python3 preprocess_data.py --ds poria-balanced --of balanced.pickle

ratio:
	python3 clean_tweets_detector.py --ds poria-ratio
	python3 preprocess_data.py --ds poria-ratio --of ratio.pickle

detector:
	python3 clean_tweets_detector.py --ds detector
	python3 preprocess_data.py --ds detector --of detector.pickle

# TRAIN ON NORMAL DATASET #
train-balanced:
	python3 tflearn_rnn.py --ds poria-balanced --in balanced.pickle

train-ratio:
	python3 tflearn_rnn.py --ds poria-ratio --in ratio.pickle

train-detector:
	python3 tflearn_rnn.py --ds detector --in detector.pickle

# SCRAMBLE THE DATASET #
scramble-balanced:
	python3 clean_tweets_detector.py --ds poria-balanced
	python3 preprocess_data.py --scramble --ds poria-balanced --of scrambled_balanced.pickle

scramble-ratio:
	python3 clean_tweets_detector.py --ds poria-ratio
	python3 preprocess_data.py --scramble --ds poria-ratio --of scrambled_ratio.pickle

scramble-detector:
	python3 clean_tweets_detector.py --ds detector
	python3 preprocess_data.py --scramble --ds detector --of scrambled_detector.pickle

# TRAIN SCRAMBLED DATASET "
train-scramble-balanced:
	python3 tflearn_rnn.py --ds poria-balanced --in scrambled_balanced.pickle

train-scramble-ratio:
	python3 tflearn_rnn.py --ds poria-ratio --in scrambled_detector.pickle

train-scramble-detector:
	python3 tflearn_rnn.py --ds detector --in scrambled_detector.pickle

# CLEAN DATASET STRICT AND SCRAMBLE #
strict-scramble-balanced:
	python3 clean_tweets_detector.py --ds poria-balanced --strict
	python3 preprocess_data.py --scramble --ds poria-balanced --of strict_scrambled_balanced.pickle

strict-scramble-ratio:
	python3 clean_tweets_detector.py --ds poria-ratio --strict
	python3 preprocess_data.py --scramble --ds poria-ratio --of strict_scrambled_ratio.pickle

strict-scramble-detector:
	python3 clean_tweets_detector.py --ds detector --strict
	python3 preprocess_data.py --scramble --ds detector --of strict_scrambled_detector.pickle

# TRAIN STRICT SCRAMBLED DATASET #
train-strict-scramble-balanced:
	python3 tflearn_rnn.py --ds poria-balanced --in strict_scrambled_balanced.pickle

train-strict-scramble-ratio:
	python3 tflearn_rnn.py --ds poria-ratio --in strict_scrambled_ratio.pickle

train-strict-scramble-detector:
	python3 tflearn_rnn.py --ds detector --in strict_scrambled_detector.pickle

	# STRICT VS PORIA #
strict-clean-detector:
	python3 clean_tweets_detector.py --ds detector --strict
	python3 preprocess_data.py --ds detector --of strict_detector.pickle
	python3 tflearn_rnn.py --ds detector --in strict_detector.pickle

poria-clean-detector:
	python3 clean_tweets_detector.py --ds detector --poria
	python3 preprocess_data.py --ds detector --of poriacleaned_detector.pickle
	python3 tflearn_rnn.py --ds detector --in poriacleaned_detector.pickle
