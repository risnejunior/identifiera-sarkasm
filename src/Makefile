#cleans all 3 datasets with the 4 different types of cleaning, balanced
compare-strictness:
	python3 beep.py

# example of using the beep script to signal when a batch is done. Quit with ctrl + c
beep:
	python3 beep.py

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

	python3 preprocess_data.py --ds poria-balanced --out tags-scrambled-special.pickle --sb 0.5 --cv all tags.pickle --scramble-special

	python3 beep.py

troublemakers:
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net little_pony --train 20 1 1200  --st hells-angels
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net little_pony --train 20 1 1200  --tt hells-angels 0.0
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle --net little_pony --train 20 1 1200  --boost hells-angels
	python3 beep.py


basename = notags
network = little_pony
dataset = detector
results:
	python3 tflearn_rnn.py --ds $(dataset) 	   --in $(basename).pickle 		--net $(network)  --train 50 1 1200 --sm $(network)_$(dataset)_$(basename) --pt

	python3 tflearn_rnn.py --ds poria-balanced --in notags.pickle 			--net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-ratio    --in notags.pickle 		    --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds detector       --in notags.pickle 		    --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-balanced --in tags.pickle             --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-ratio    --in tags.pickle             --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds detector       --in tags.pickle   		    --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-balanced --in tags-scrambled.pickle   --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-ratio    --in tags-scrambled.pickle	--net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds detector 	   --in tags-scrambled.pickle   --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-balanced --in notags-scrambled.pickle --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-ratio    --in notags-scrambled.pickle --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds detector 	   --in notags-scrambled.pickle --net $(network) --eval $(network)_$(dataset)_$(basename) --pt
	python3 tflearn_rnn.py --ds poria-balanced --in tags-scrambled-special.pickle --net $(network) --eval $(network)_$(dataset)_$(basename) --pt

special:
	python3 tflearn_rnn.py --ds poria-balanced --in tags-scrambled-special.pickle --net $(network) --eval $(network)_$(dataset)_$(basename) --pt

janus:
	make special network=little_pony dataset=poria-balanced basename=notags
	make special network=little_pony dataset=poria-ratio basename=notags
	make special network=little_pony dataset=detector basename=notags
	make special network=little_pony dataset=poria-balanced basename=tags
	make special network=little_pony dataset=poria-ratio basename=tags
	make special network=little_pony dataset=detector basename=tags
	make special network=little_pony dataset=poria-balanced basename=tags-scrambled

jan:
	make results network=little_pony dataset=poria-balanced basename=notags
	make results network=little_pony dataset=poria-ratio basename=notags
	make results network=little_pony dataset=detector basename=notags
	make results network=little_pony dataset=poria-balanced basename=tags
	make results network=little_pony dataset=poria-ratio basename=tags
	make results network=little_pony dataset=detector basename=tags
	make results network=little_pony dataset=poria-balanced basename=tags-scrambled

risne:
	make results network=little_gru dataset=poria-balanced basename=notags
	make results network=little_gru dataset=poria-ratio basename=notags
	make results network=little_gru dataset=detector basename=notags
	make results network=little_gru dataset=poria-balanced basename=tags
	make results network=little_gru dataset=poria-ratio basename=tags
	make results network=little_gru dataset=detector basename=tags
	make results network=little_gru dataset=poria-balanced basename=tags-scrambled
	say 'i love bark'

adele:
	make results network=convolve_me dataset=poria-balanced basename=notags
	make results network=convolve_me dataset=poria-ratio basename=notags
	make results network=convolve_me dataset=detector basename=notags
	make results network=convolve_me dataset=poria-balanced basename=tags
	make results network=convolve_me dataset=poria-ratio basename=tags
	make results network=convolve_me dataset=detector basename=tags
	make results network=convolve_me dataset=poria-balanced basename=tags-scrambled
	say 'i love bark'
