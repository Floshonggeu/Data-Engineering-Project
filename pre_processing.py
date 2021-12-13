# Import libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag, pos_tag_sents
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Download nltk features
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def lower_data(data):
	lowered_data = data.str.lower()
	return lowered_data

def remove_punctuation(data):
	removed_data = data.str.replace('[^\w\s]',' ')
	removed_data = removed_data.str.replace('_','')
	removed_data = removed_data.str.replace("  "," ")
	return removed_data

def remove_stopwords(data):
	stop_words = stopwords.words('english')
	removed_data = data.apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
	return removed_data

def tokenize_data(data, serie):
	data[serie] = data.apply(lambda row: word_tokenize(row[serie]), axis = 1)
	return data[serie]

def set_pos_tag(data):
	data = pos_tag_sents(data.tolist())
	return data

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_data(data):
	lemmatizer = WordNetLemmatizer()
	data = data.transform(lambda value: ' '.join([lemmatizer.lemmatize(a[0], pos = get_wordnet_pos(a[1])) if get_wordnet_pos(a[1]) else a[0] for a in value]))
	return data

def preprocessing(data):
	data[0] = lower_data(data[0])
	# Remove puncuation
	data[0] = remove_punctuation(data[0])
	# Remove stopwords
	data[0] = remove_stopwords(data[0])
	# Tokenize data
	data[0] = tokenize_data(data, 0)
	# Set pos tags
	data[0] = set_pos_tag(data[0])
	# Lemmatize data
	data[0] = lemmatize_data(data[0])
	# Define test matrix
	X_train = pd.read_csv('X_train.csv')
	vectorizer = CountVectorizer(token_pattern = r'\b\w+\b')
	train_matrix = vectorizer.fit_transform(X_train)
	test_matrix = vectorizer.fit_transform(data[0])
	return test_matrix