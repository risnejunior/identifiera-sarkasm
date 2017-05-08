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

# processes cleans
clean-process-all:
	python3 clean_tweets.py --ds detector --strict false --tags false
	python3 clean_tweets.py --ds poria-balanced --strict false --tags false
	python3 clean_tweets.py --ds poria-ratio --strict false --tags false

	python3 preprocess_data.py --ds all --out notags.pickle --le 800000

	python3 preprocess_data.py --ds poria-ratio --out notags.pickle --sb 0.25 --cv all notags.pickle
	python3 preprocess_data.py --ds poria-ratio --out notags-scrambled.pickle --scramble --sb 0.25  --cv all notags.pickle
	python3 preprocess_data.py --ds detector --out notags.pickle --sb 0.5  --cv all notags.pickle
	python3 preprocess_data.py --ds detector --out notags-scrambled.pickle --scramble --sb 0.5  --cv all notags.pickle
	python3 preprocess_data.py --ds poria-balanced --out notags.pickle --sb 0.5  --cv all notags.pickle
	python3 preprocess_data.py --ds poria-balanced --out notags-scrambled.pickle --scramble --sb 0.5  --cv all notags.pickle

	python3 clean_tweets.py --ds detector --strict false  --tags true
	python3 clean_tweets.py --ds poria-balanced --strict false --tags true
	python3 clean_tweets.py --ds poria-ratio --strict  false --tags true

	python3 preprocess_data.py --ds all --out tags.pickle --le 800000

	python3 preprocess_data.py --ds detector --out tags-scrambled.pickle --scramble --sb 0.5 --cv all tags.pickle
	python3 preprocess_data.py --ds detector --out tags.pickle --sb 0.5 --cv all tags.pickle
	python3 preprocess_data.py --ds poria-balanced --out tags-scrambled.pickle --scramble --sb 0.5 --cv all tags.pickle
	python3 preprocess_data.py --ds poria-balanced --out tags.pickle --sb 0.5 --cv all tags.pickle
	python3 preprocess_data.py --ds poria-ratio --out tags-scrambled.pickle --scramble --sb 0.25 --cv all tags.pickle
	python3 preprocess_data.py --ds poria-ratio --out tags.pickle --sb 0.25 --cv all tags.pickle

	python3 beep.py

basename = notags
network = little_pony
dataset = detector
results:
	python3 tflearn_rnn.py --ds $(dataset) 	   --in $(basename).pickle 		--net $(network)  --train 50 1 1200 --sm $(network)_$(dataset)_$(basename) --pt

	python3 tflearn_rnn.py --ds poria-balanced --in notags.pickle 			--net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-ratio    --in notags.pickle 		    --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds detector    --in notags.pickle 		    --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle             --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-ratio    --in tags.pickle             --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds detector       --in tags.pickle   		    --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-balanced --in tags-scrambled.pickle   --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-ratio    --in tags-scrambled.pickle	--net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds detector 	   --in tags-scrambled.pickle   --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-balanced --in notags-scrambled.pickle --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-ratio    --in notags-scrambled.pickle --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds detector 	   --in notags-scrambled.pickle --net $(network) --eval $(network)_$(dataset)_$(basename) --pt

	python3 beep.py

troublemakers:
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net little_pony --train 20 10 1200  --st hells-angels
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net little_pony --train 20 1 1200  --tt hells-angels 0.0
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net little_pony --train 20 1 1200  --boost hells-angels
	python3 beep.py

# example of using the beep script to signal when a batch is done. Quit with ctrl + c
beep:
	python3 beep.py

jan:
	make results network=little_pony dataset=poria-balanced basename=notags
	make results network=little_pony dataset=poria-ratio basename=notags
	make results network=little_pony dataset=detector basename=notags
	make results network=little_pony dataset=poria-balanced basename=tags
	make results network=little_pony dataset=poria-ratio basename=tags
	make results network=little_pony dataset=detector basename=tags
	python3 beep.py

risne:
	make results network=little_gru dataset=poria-balanced basename=notags
	make results network=little_gru dataset=poria-ratio basename=notags
	make results network=little_gru dataset=detector basename=notags
	make results network=little_gru dataset=poria-balanced basename=tags
	make results network=little_gru dataset=poria-ratio basename=tags
	make results network=little_gru dataset=detector basename=tags
	say 'i love bark'

adele:
	make results network=convolve_me dataset=poria-balanced basename=notags
	make results network=convolve_me dataset=poria-ratio basename=notags
	make results network=convolve_me dataset=detector basename=notags
	make results network=convolve_me dataset=poria-balanced basename=tags
	make results network=convolve_me dataset=poria-ratio basename=tags
	make results network=convolve_me dataset=detector basename=tags
	say 'i love bark'
