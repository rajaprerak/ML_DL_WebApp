from flask import Flask,render_template,url_for,request
import pickle
from werkzeug.utils import secure_filename
import os
from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import logging
import sys
from summarizer import nltk_summarizer
import time
import spacy
from bs4 import BeautifulSoup
from urllib.request import urlopen
import nltk

nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
nltk.download('punkt')
app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


path  = os.path.join(os.getcwd(),"models","iris_model.pkl")
with open(path, 'rb') as f:
        iris_clf = pickle.load(f)
path_model  = os.path.join(os.getcwd(),"models","catdog_model.h5")
catdog_clf = load_model(path_model)


@app.route('/')
def home():
    return render_template('bootstrap.html')

@app.route('/iris')
def iris():
    return render_template('iris.html')

@app.route('/result_iris', methods=['POST'])
def result_iris():

    slength = request.form['slength']
    swidth = request.form['swidth']
    plength = request.form['plength']
    pwidth = request.form['pwidth']
    my_prediction = iris_clf.predict([[slength,swidth,plength,pwidth]])

    return render_template('result_iris.html',prediction = my_prediction)

@app.route('/image_classify')
def image_classify():
    return render_template('image_classify.html')

def predict(file):
    img  = load_img(file, target_size = (64, 64))
    img = img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)
    probs = catdog_clf.predict(img)[0]
    output = {'Cat': probs[0], 'Dog': probs[1]}
    return output

@app.route('/result_image_classify', methods=['POST'])
def result_image_classify():

    if request.method == 'POST':
        file = request.files['image']
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('static/img/', filename)
        file.save(file_path)
        output = predict(file_path)
        temp = 0
        for v in output.values():
            if v < 0.5:
                temp += 1
        if temp == len(output):
            final_output = "None"
        else:
            final_output = max(output, key= lambda x: output[x])
    return render_template('result_image_classify.html',label = final_output, imagesource=file_path)

@app.route('/analyze',methods=['GET','POST'])
def analyze():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary = nltk_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('summarizer_output.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

@app.route('/analyze_url',methods=['GET','POST'])
def analyze_url():
	start = time.time()
	if request.method == 'POST':
		raw_url = request.form['raw_url']
		rawtext = get_text(raw_url)
		final_reading_time = readingTime(rawtext)
		final_summary = nltk_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('summarizer_output.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

# Reading Time
def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

@app.route('/summarizer')
def summarizer():
    return render_template('summarizer.html')

if __name__ == '__main__':
    app.run(debug=True)
