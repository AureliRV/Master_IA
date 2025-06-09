from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
#import tensorflow as tf

import numpy as np
from PIL import Image


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo CNN
#model = tf.keras.models.load_model('ianudis.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def inicio():
    return render_template("index.html")

@app.route('/templates')
def resultado():
    return render_template("resultado.html")



if __name__ == '__main__':
    app.run(debug=True)




  

