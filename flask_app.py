from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
import sys
import os
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags
import numpy as np
from gevent.pywsgi import WSGIServer
import pandas as pd

FILEPATH = os.path.realpath(__file__)
ROOTPATH = os.path.split(FILEPATH)[0]
SRCPATH = os.path.join(ROOTPATH, 'src')
MODELPATH = os.path.join(ROOTPATH, 'model')
CURRENTMODEL = os.path.join(MODELPATH, 'model_v4.h5')
UPLOADPATH = os.path.join(ROOTPATH, 'uploads')
sys.path.append(SRCPATH)

from waldo_finder_img_classification import *


def rotate_save(f, file_path):
    try:
        image = Image.open(f)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        image.save(file_path)
        image.close()

    except (AttributeError, KeyError, IndexError):
        image.save(file_path)
        image.close()


def model_predict(path, model):
    waldo_finder = WaldoFinder(path, flask=True)
    waldo_finder.load_model(model)
    waldo_finder.find_waldo(savedir=path)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/waldo_finder', methods=["GET", "POST"])
def upload():
    if request.method == 'POST':
        f = request.files["file"]
        file_path = os.path.join(UPLOADPATH, secure_filename(f.filename))
        f.save(file_path)
        rotate_save(f, file_path)
        model_predict(file_path, CURRENTMODEL)
        return redirect(url_for('uploaded_file',
                        filename=os.path.split(file_path)[1]))
    if len(os.listdir(UPLOADPATH)) != 0:
        for file in os.listdir(UPLOADPATH):
            os.remove(os.path.join(UPLOADPATH, file))
    return render_template('waldo_finder.html')


@app.route('/show/<filename>')
def uploaded_file(filename):
    return render_template('waldo_finder.html', filename=filename)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOADPATH, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
