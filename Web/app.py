from flask import Flask, request, render_template
import joblib
import pandas as pd
from pre_processing import preprocessing
from csv import reader

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
	filename = 'finalized_model.sav'
	loaded_model = joblib.load(filename)
	text = request.form['text']
	df = pd.DataFrame(list(reader(text)))
	t_matrix = preprocessing(df)
	results = loaded_model.predict(t_matrix)
	if results == 1:
		return "positive"
	elif results == -1:
		return "negative"
	elif results == 0:
		return "neutral"
	else:
		return results