import os
from flask import Flask, request, redirect, url_for, flash,render_template
from werkzeug.utils import secure_filename
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
from keras.layers import Flatten
import matplotlib.pyplot as plt

TEST_DIR = 'C:\\Users\\Prime Focus Systems\\Pictures\\real time testing'
IMG_SIZE = 80
LR = 1e-3  # 0.001

def fakeOrReal():


    MODEL_NAME = 'flatten-{}-{}.model'.format(LR,
                                               '6conv-basic')  # just so we remember which saved model is which, sizes must match

    tf.reset_default_graph()
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet, num_features = flatten_layer(convnet)

    convnet = fully_connected(convnet, 1024, activation='relu')

    convnet = fully_connected(convnet, 512, activation='relu')

    convnet = fully_connected(convnet, 256, activation='relu')
    convnet = dropout(convnet, 0.7)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    # model.save(MODEL_NAME)



    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded! finally')
    else:
        print("model not loaded")

    test_data = process_test_data()

    for num, data in enumerate(test_data[:8000]):
        # cat: [1,0]
        # dog: [0,1]

        img_num = data[1]
        img_data = data[0]

        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 0:
            return "real"
        else:
            return "fake"


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == '500real' or word_label == '2000real':
        return [1, 0]
    #                             [no cat, very doggo]
    elif word_label == '500fake' or word_label == '2000fake':
        return [0, 1]

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.medianBlur(img, 5)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = np.array(layer_shape[1:4], dtype=int).prod()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features




UPLOAD_FOLDER = 'C:\\Users\\Prime Focus Systems\\Pictures\\real time testing'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#project k functions


#init func

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print("hello")
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print("hi")
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            str=fakeOrReal()
            os.remove('C:\\Users\\Prime Focus Systems\\Pictures\\real time testing\\' + file.filename)
            return render_template('index.html',str=str)

    return render_template('onepage.html',str="anuthi")


if __name__ == '__main__':
    app.secret_key='super secret key'
    app.run('0.0.0.0')
