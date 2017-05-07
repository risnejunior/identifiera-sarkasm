train:
	python3 tflearn_rnn.py

test:
	python3 tflearn_rnn.py --ds poria-ratio --in ratio.pickle --pt --pretrained 71PVGG_BIG_CLOWN

# CLEAN NORMAL DATASET #
balanced:
	python3 clean_tweets.py --ds poria-balanced
	python3 preprocess_data.py --ds poria-balanced --of balanced.pickle

ratio:
	python3 clean_tweets.py --ds poria-ratio
	python3 preprocess_data.py --ds poria-ratio --of ratio.pickle

detector:
	python3 clean_tweets.py --ds detector
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
	python3 clean_tweets.py --ds poria-balanced
	python3 preprocess_data.py --scramble --ds poria-balanced --of scrambled_balanced.pickle

scramble-ratio:
	python3 clean_tweets.py --ds poria-ratio
	python3 preprocess_data.py --scramble --ds poria-ratio --of scrambled_ratio.pickle

scramble-detector:
	python3 clean_tweets.py --ds detector
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
	python3 clean_tweets.py --ds poria-balanced --strict
	python3 preprocess_data.py --scramble --ds poria-balanced --of strict_scrambled_balanced.pickle

strict-scramble-ratio:
	python3 clean_tweets.py --ds poria-ratio --strict
	python3 preprocess_data.py --scramble --ds poria-ratio --of strict_scrambled_ratio.pickle

strict-scramble-detector:
	python3 clean_tweets.py --ds detector --strict
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
	python3 clean_tweets.py --ds detector --strict
	python3 preprocess_data.py --ds detector --of strict_detector.pickle
	python3 tflearn_rnn.py --ds detector --in strict_detector.pickle

poria-clean-detector:
	python3 clean_tweets.py --ds detector --poria
	python3 preprocess_data.py --ds detector --of poria_detector.pickle
	python3 tflearn_rnn.py --ds detector --in poria_detector.pickle

#cleans all 3 datasets with the 4 different types of cleaning, balanced
compare-strictness:
	
	python3 clean_tweets.py --ds detector --strict false false
	python3 preprocess_data.py --ds detector --out all-notags-balanced.pickle  --le 400000
	python3 clean_tweets.py --ds detector --strict true false
	python3 preprocess_data.py --ds detector --out strict-notags-balanced.pickle --le 400000	
	python3 clean_tweets.py --ds detector --strict false true
	python3 preprocess_data.py --ds detector --out all-tags-balanced.pickle --le 400000
	python3 clean_tweets.py --ds detector --strict true true
	python3 preprocess_data.py --ds detector --out strict-tags-balanced.pickle --le 400000
	
	
	python3 clean_tweets.py --ds poria-balanced --strict false false
	python3 preprocess_data.py --ds poria-balanced --out all-notags-balanced.pickle --le 400000
	python3 clean_tweets.py --ds poria-balanced --strict true false
	python3 preprocess_data.py --ds poria-balanced --out strict-notags-balanced.pickle --le 400000	
	python3 clean_tweets.py --ds poria-balanced --strict false true
	python3 preprocess_data.py --ds poria-balanced --out all-tags-balanced.pickle --le 400000
	python3 clean_tweets.py --ds poria-balanced --strict true true
	python3 preprocess_data.py --ds poria-balanced --out strict-tags-balanced.pickle --le 400000

	
	python3 clean_tweets.py --ds poria-ratio --strict false false
	python3 preprocess_data.py --ds poria-ratio --out all-notags-balanced.pickle
	python3 clean_tweets.py --ds poria-ratio --strict true false
	python3 preprocess_data.py --ds poria-ratio --out strict-notags-balanced.pickle	
	python3 clean_tweets.py --ds poria-ratio --strict false true
	python3 preprocess_data.py --ds poria-ratio --out all-tags-balanced.pickle
	python3 clean_tweets.py --ds poria-ratio --strict true true
	python3 preprocess_data.py --ds poria-ratio --out strict-tags-balanced.pickle


	python3 tflearn_rnn.py --in all-notags-balanced.pickle --ds detector --train 20 1 1200
	python3 tflearn_rnn.py --in strict-tags-balanced.pickle --ds detector --train 20 1 1200
	python3 tflearn_rnn.py --in all-tags-balanced.pickle --ds detector --train 20 1 1200
	python3 tflearn_rnn.py --in strict-notags-balanced.pickle --ds detector --train 20 1 1200
	
	python3 tflearn_rnn.py --in all-notags-balanced.pickle --ds poria-balanced --train 20 1 1200
	python3 tflearn_rnn.py --in strict-tags-balanced.pickle --ds poria-balanced --train 20 1 1200
	python3 tflearn_rnn.py --in all-tags-balanced.pickle --ds poria-balanced --train 20 1 1200
	python3 tflearn_rnn.py --in strict-notags-balanced.pickle --ds poria-balanced --train 20 1 1200

	python3 tflearn_rnn.py --in all-notags-balanced.pickle --ds poria-ratio --train 20 1 1200
	python3 tflearn_rnn.py --in strict-tags-balanced.pickle --ds poria-ratio --train 20 1 1200
	python3 tflearn_rnn.py --in all-tags-balanced.pickle --ds poria-ratio --train 20 1 1200
	python3 tflearn_rnn.py --in strict-notags-balanced.pickle --ds poria-ratio --train 20 1 1200

clean-process-all:
	python3 clean_tweets.py --ds detector --strict false --tags false
	python3 preprocess_data.py --ds detector --out notags.pickle --sb 0.5 --le 400000
	python3 preprocess_data.py --ds detector --out notags-scrambled.pickle --scramble --sb 0.5 --le 400000
	python3 preprocess_data.py --ds detector --out tags-scrambled.pickle --scramble --sb 0.5 --le 400000
	python3 clean_tweets.py --ds detector --strict false  --tags true
	python3 preprocess_data.py --ds detector --out tags.pickle --sb 0.5 --le 400000

	python3 clean_tweets.py --ds poria-balanced --strict false --tags false
	python3 preprocess_data.py --ds poria-balanced --out notags.pickle --sb 0.5 --le 400000
	python3 preprocess_data.py --ds poria-balanced --out notags-scrambled.pickle --scramble --sb 0.5 --le 
	python3 preprocess_data.py --ds poria-balanced --out tags-scrambled.pickle --scramble --sb 0.5 --le 400000
	python3 clean_tweets.py --ds poria-balanced --strict false --tags true
	python3 preprocess_data.py --ds poria-balanced --out tags.pickle --sb 0.5 --le 400000

	python3 clean_tweets.py --ds poria-ratio --strict false --tags false
	python3 preprocess_data.py --ds poria-ratio --out notags.pickle --sb 0.25 --le 400000
	python3 preprocess_data.py --ds poria-ratio --out notags-scrambled.pickle --scramble --sb 0.25 --le 400000
	python3 preprocess_data.py --ds poria-ratio --out tags-scrambled.pickle --scramble --sb 0.25 --le 400000
	python3 clean_tweets.py --ds poria-ratio --strict  false --tags true
	python3 preprocess_data.py --ds poria-ratio --out tags.pickle --sb 0.25 --le 400000
	python3 beep.py

result-detector-notags:
	python3 tflearn_rnn.py --ds detector --in notags.pickle --net little_pony --train 50 1 1200 --sm little_pony_detector_notags
	python3 tflearn_rnn.py --ds poria-balanced --in notags.pickle --net little_pony --eval little_pony_detector_notags
	python3 tflearn_rnn.py --ds poria-ratio --in notags.pickle --net little_pony --eval little_pony_detector_notags
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net little_pony --eval little_pony_detector_notags
	python3 tflearn_rnn.py --ds poria-ratio --in tags.pickle --net little_pony --eval little_pony_detector_notags
	python3 tflearn_rnn.py --ds detector --in tags.pickle --net little_pony --eval little_pony_detector_notags
	python3 tflearn_rnn.py --ds poria-balanced --in notags-scrambled.pickle --net little_pony --eval little_pony_detector_notags
	python3 tflearn_rnn.py --ds poria-ratio --in notags-scrambled.pickle --net little_pony --eval little_pony_detector_notags
	python3 tflearn_rnn.py --ds detector --in notags-scrambled.pickle --net little_pony --eval little_pony_detector_notags

	python3 tflearn_rnn.py --ds detector --in notags.pickle --net little_gru --train 50 1 1200 --sm little_gru_detector_notags
	python3 tflearn_rnn.py --ds poria-balanced --in notags.pickle --net little_gru --eval little_gru_detector_notags
	python3 tflearn_rnn.py --ds poria-ratio --in notags.pickle --net little_gru --eval little_gru_detector_notags
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net little_gru --eval little_gru_detector_notags
	python3 tflearn_rnn.py --ds poria-ratio --in tags.pickle --net little_gru --eval little_gru_detector_notags
	python3 tflearn_rnn.py --ds detector --in tags.pickle --net little_gru --eval little_gru_detector_notags
	python3 tflearn_rnn.py --ds poria-balanced --in notags-scrambled.pickle --net little_gru --eval little_gru_detector_notags
	python3 tflearn_rnn.py --ds poria-ratio --in notags-scrambled.pickle --net little_gru --eval little_gru_detector_notags
	python3 tflearn_rnn.py --ds detector --in notags-scrambled.pickle --net little_gru --eval little_gru_detector_notags



# example of using the beep script to signal when a batch is done. Quit with ctrl + c
beep:
	python3 tflearn_rnn.py --ds detector --in notags.pickle --net convolve_me --train 50 1 1200 --sm convolve_me_detector_notags
	python3 tflearn_rnn.py --ds poria-balanced --in notags.pickle --net convolve_me --eval convolve_me_detector_notags
	python3 tflearn_rnn.py --ds poria-ratio --in notags.pickle --net convolve_me --eval convolve_me_detector_notags
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net convolve_me --eval convolve_me_detector_notags
	python3 tflearn_rnn.py --ds poria-ratio --in tags.pickle --net convolve_me --eval convolve_me_detector_notags
	python3 tflearn_rnn.py --ds detector --in tags.pickle --net convolve_me --eval convolve_me_detector_notags
	python3 tflearn_rnn.py --ds poria-balanced --in notags-scrambled.pickle --net convolve_me --eval convolve_me_detector_notags
	python3 tflearn_rnn.py --ds poria-ratio --in notags-scrambled.pickle --net convolve_me --eval convolve_me_detector_notags
	python3 tflearn_rnn.py --ds detector --in notags-scrambled.pickle --net convolve_me --eval convolve_me_detector_notags

# LITTTLE_PONY: 
# train balanced with tags 
# test on x with no tags + balanced scramble tags
lp_balanced-tags:
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net little_pony --train 10 1 128 --sm little_pony_balanced_tags
	python3 tflearn_rnn.py --ds poria-balanced --in notags.pickle --net little_pony --eval little_pony_balanced_tags --pt
	python3 tflearn_rnn.py --ds poria-ratio --in notags.pickle --net little_pony --eval little_pony_balanced_tags --pt
	python3 tflearn_rnn.py --ds detector --in notags.pickle --net little_pony --eval little_pony_balanced_tags --pt
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net little_pony --eval little_pony_balanced_tags --pt
	python3 tflearn_rnn.py --ds poria-ratio --in tags.pickle --net little_pony --eval little_pony_balanced_tags --pt
	python3 tflearn_rnn.py --ds detector --in tags.pickle --net little_pony --eval little_pony_balanced_tags --pt
	python3 tflearn_rnn.py --ds poria-balanced --in tags-scrambled.pickle --net little_pony --eval little_pony_balanced_tags --pt

# LITTTLE_GRU: 
# train balanced with tags 
# test on x with no tags + balanced scramble tags
lg_balanced-tags:
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net little_gru --train 10 1 128 --sm little_gru_balanced_tags
	python3 tflearn_rnn.py --ds poria-balanced --in notags.pickle --net little_gru --eval little_gru_balanced_tags --pt
	python3 tflearn_rnn.py --ds poria-ratio --in notags.pickle --net little_gru --eval little_gru_balanced_tags --pt
	python3 tflearn_rnn.py --ds detector --in notags.pickle --net little_gru --eval little_gru_balanced_tags --pt
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net little_gru --eval little_gru_balanced_tags --pt
	python3 tflearn_rnn.py --ds poria-ratio --in tags.pickle --net little_gru --eval little_gru_balanced_tags --pt
	python3 tflearn_rnn.py --ds detector --in tags.pickle --net little_gru --eval little_gru_balanced_tags --pt
	python3 tflearn_rnn.py --ds poria-balanced --in tags-scrambled.pickle --net little_gru --eval little_gru_balanced_tags --pt

# CONVOLVE_ME:
# train balanced with tags
# test on x with no tags + balanced scramble tags
cm_balanced-tags:
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net convolve_me --train 10 1 128 --sm convolve_me_balanced_tags
	python3 tflearn_rnn.py --ds poria-balanced --in notags.pickle --net convolve_me --eval convolve_me_balanced_tags --pt
	python3 tflearn_rnn.py --ds poria-ratio --in notags.pickle --net convolve_me --eval convolve_me_balanced_tags --pt
	python3 tflearn_rnn.py --ds detector --in notags.pickle --net convolve_me --eval convolve_me_balanced_tags --pt
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net convolve_me --eval convolve_me_balanced_tags --pt
	python3 tflearn_rnn.py --ds poria-ratio --in tags.pickle --net convolve_me --eval convolve_me_balanced_tags --pt
	python3 tflearn_rnn.py --ds detector --in tags.pickle --net convolve_me --eval convolve_me_balanced_tags --pt
	python3 tflearn_rnn.py --ds poria-balanced --in tags-scrambled.pickle --net convolve_me --eval convolve_me_balanced_tags --pt


