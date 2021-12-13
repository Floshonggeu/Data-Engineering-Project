# Import libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag, pos_tag_sents
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

def input_score(data):
	counter = 0
	for i in data:
		if (i < 3):
			data[counter] = - 1
		elif (i == 3):
			data[counter] = 0
		elif (i > 3):
			data[counter] = 1
		counter = counter + 1
	return data

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

# Import data
data = pd.read_csv('Reviews.csv', encoding = 'utf8', engine = 'python', error_bad_lines = False)

# Remove useless columns
data = data.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time', 'Summary'], axis = 1)

len_data = len(data)
to_drop = int(len_data * 0.999)
data = data.drop(data.tail(to_drop).index)

# Lower data
data['Text'] = lower_data(data['Text'])

# Remove puncuation
data['Text'] = remove_punctuation(data['Text'])

# Transform rating to variables
data['Score'] = input_score(data['Score'])

# Remove stopwords
data['Text'] = remove_stopwords(data['Text'])

# Tokenize data
data['Text'] = tokenize_data(data, 'Text')

# Set pos tags
data['Text'] = set_pos_tag(data['Text'])

# Lemmatize data
data['Text'] = lemmatize_data(data['Text'])

# Separating X and Y
X = data['Text']
Y = data['Score']

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

# Initialize vectorizer
vectorizer = CountVectorizer(token_pattern = r'\b\w+\b')

# Create matrixes
train_matrix = vectorizer.fit_transform(X_train)
test_matrix = vectorizer.transform(X_test)

# Initialize model
lr = LogisticRegression()

# Use matrixes for the model
X_train = train_matrix
X_test = test_matrix

# Fit model
lr.fit(X_train , Y_train)

# Predict
predictions = lr.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, predictions)

# Print accuracy score
print("Accuracy score is:" + accuracy)