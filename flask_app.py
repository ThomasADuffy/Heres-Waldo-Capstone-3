from flask import Flask, render_template, flash, request, redirect, url_for
import sys
import os
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags
import numpy as np
from gevent.pywsgi import WSGIServer
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
import pandas as pd
import pickle

FILEPATH=os.path.realpath(__file__)
ROOTPATH=os.path.split(FILEPATH)[0]
SRCPATH=os.path.join(ROOTPATH,'src')
MODELPATH=os.path.join(ROOTPATH,'models')
UPLOADPATH=os.path.join(ROOTPATH,'uploads')
sys.path.append(SRCPATH)

def rotate_save(f, file_path):
    try:
        image=Image.open(f)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
        image.save(file_path)
        image.close()

    except (AttributeError, KeyError, IndexError):
        image.save(file_path)
        image.close()


app = Flask(__name__)

#home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/predictor', methods=["GET","POST"])
def upload():
    if request.method == 'POST':
        f = request.files["file"]
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        rotate_save(f, file_path)
        preds = model_predict(file_path, model)
        os.remove(file_path)
        return preds
    return None
    return render_template('waldo_finder.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,threaded=True)




