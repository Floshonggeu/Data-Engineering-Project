from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    if text=='positive':
        return "positive"

    elif text=='negative':
        return "negative"

    else:
        return "neutre"




           



