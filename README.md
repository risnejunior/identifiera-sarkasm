# Identifiera sarkasm i text
Kandidatarbete 2017

## Folder structure for datasets
```
.
├── datasets
│   ├── glove_twitter_embeddings
│   ├── imdb
│   │   ├── neg
│   │   └── pos
│   └── poria
│       ├── en-balanced
│       │   ├── neg
│       │   └── pos
│       └── en-ratio
│           ├── neg
│           └── pos
```

## Get the data

Right now the following data is used:
- Glove twitter embeddings
- The downloaded tweets
- doImdb reviews


1. Download the imdb & twitter data (datasets.zip in this folder)
2. Unzip the imdb & twitter data in: `identifiera-sarkasm\datasets`, overwrite any existing data. Make sure you **don’t** get this structure: `\identifiera-sarkasm\datasets\datasets\`
3. You should have these three folders inside the dataset directory (you might already have others):
  1. poria
  2. Imdb
  3. glove_twitter_embeddings
4. Download the glove twitter embeddings:http://nlp.stanford.edu/data/glove.twitter.27B.zip (might take a while so feel free to copy from friend)
5. Unzip the embeddings inside the ‘glove_twitter_embeddings’ directory (4 files).
6. Profit!

## Description of the source files
The following python files are in the source directory:

* **analyze.py**: just a stub right now, meant to load an already trained network
* **clean_tweets.py**: cleans the tweets from links, hashtags etc
* **scraper.py**: downloads tweets, you probably won’t need this as they already are downloaded.
* **common_funs.py**: library file containing common functions used in other files
* **preprocess_data.py**: tokenizes the tweets and turns the data into a form ready to be used in the network.
* **tflearn_rnn.py**: the code that runs the classifier
* **twokenize.py**: a twitter tokenization library, unused for now
## What the code does

1. The scraper reads the tweet id’s from normal.txt and sarcastic.txt, downloads these tweets and puts them in balanced_normal_tweets.csv and balanced_sarcastic_tweets.csv
2. The clean_tweets code takes the tweets from the aforementioned .csv files, tokenizes & cleans the tweets, and then puts them in the cleaned folder, one file per tweet.
3. Preprocess_data reads the cleaned tweets and creates 4 .json files:
  1. The **vocabulary**: it contains the n-most common words from the tweets, with the words mapped to an integer. ‘.’ maps to 0, and ‘_’ maps to 1. ‘_’ signifies words missing in the dictionary. ‘.’ signifies the padding at the end of tweets.
  2. The **reverse vocabulary**: it’s the same mapping as in the vocabulary just reversed
  3. The **preprocesses_data** file: it contains all tweets in the form, each with 3 fields: the tweet id, a list of the tokenized words from the tweet, and a list of the mappings of this words according to the dictionary
  4. The **embeddings_file**: it’s a list of all the world vectors, sorted in the same order as the vocabulary file. So if a word has index n in the vocabulary it’s corresponding word vector will be on row n in the embeddings file.
4. **Tflearn_rnn.py**: takes the .json files the preprocessed data and embeddings, and trains the network with these. In the end it prints a confusion matrix for the training, evaluation and test set.

## How to run the code the first time
1. If you already have the data downloaded you first need to run the clean_tweets.py file. Unless the cleaning strategy changes you only need to do this once.
2. Run preprocess_data.py, if you get an error message regarding missing stopwords, change print_debug to False. You can also try to write this in console to download the dependencies:
  1. Import nltk
  2. nltk.download()
3. Run **tflearn_rnn.py**

## Settings
Right now the settings are spread out and duplicated over preprocess_data and tflearn_rnn if you want to change the size of the embeddings or the size of the dictionary etc you will need to rerun preprocess data. Otherwise you only need to run preprocess_data once

## Unicode issues (probably only windows users)
* If you have problems writing utf-8 to console, i.e some error about unicode mapping, type this in console, ‘chcp 65001 & cmd’, just after starting the console (windows only)
* If your console displays ?-signs or has similar problems displaying certain chars try changing the font to one that has better utf-8 support, like ‘dejavu mono sans’.
* In ConEMu (3rd party console) you can set this up to work automatically by going to settings->main and change alternative fonts to: ‘DejaVu Sans Mono’. Then go to startup->environment and add ‘chcp utf8’ to the environment variables (under PATH).

## Tensorboard
To run Tensorboard write this in console: `tensorboard --logdir=/tmp/tflearn_logs/`

Tensorboard will start a web server and print out the address where you can find it using your web browser.

## Roadmap 🚞
- [ ] Skriva lite kod
- [ ] Identifiera sarkasm i text
- [ ] 🍺
