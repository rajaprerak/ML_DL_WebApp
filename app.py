from flask import Flask,render_template,url_for,request
import pickle
from werkzeug.utils import secure_filename
import os
from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import logging
import sys

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


path  = os.path.join(os.getcwd(),"models","knn.pkl")
with open(path, 'rb') as f:
        iris_clf = pickle.load(f)
path_model  = os.path.join(os.getcwd(),"models","test.h5")
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
        final_output = max(output, key= lambda x: output[x])
    return render_template('result_image_classify.html',label = final_output, imagesource=file_path)


if __name__ == '__main__':
    app.run(debug=True)
