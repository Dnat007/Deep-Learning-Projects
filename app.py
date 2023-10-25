import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


from flask import Flask, render_template

app = Flask(__name__)
new_model = tf.keras.models.load_model('mri_prediction.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/page2', methods=['GET', 'POST'])
def page2():
    return render_template('page2.html')


@app.route('/page3', methods=['GET', 'POST'])
def page3():
    return render_template('page3.html')


@app.route('/brain_cancer', methods=['GET', 'POST'])
def page4():
    return render_template('brain_cancer.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        f = request.files['img']
        f.save('img.jpg')

    f = image.load_img('img.jpg', target_size=(150, 150))

    x = np.array(f)
    reshape = x.reshape(1, 150, 150, 3)

    print(reshape.shape)

    a1 = new_model.predict(reshape)
    indices = a1.argmax()

    print(type(indices))

    if indices == 0:
        indices = 'glioma_tumor'
    elif indices == 1:
        indices = 'meningioma_tumor'

    elif indices == 2:
        indices = 'no_tumor'
    elif indices == 3:
        indices = 'pituitary_tumor'
    print(indices)
    return render_template('prediction.html', data=indices, name='img.jpg')


if __name__ == '__main__':
    app.run(debug=True)
