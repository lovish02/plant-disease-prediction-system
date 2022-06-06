from io import BytesIO
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

@app.route("/")
def index():
    return render_template('index.html')

disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                         'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                         'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                         'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                         'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

@app.route("/predict", methods = ['POST'])
def predict():

    file = request.files['file']
    filename = file.filename
    img = Image.open(file)
    img = img.resize((64,64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # print(x)
    x /= 255
    preds = model.predict(x)
    # print("reached here")
    # print(preds)
    a = preds[0]
    # print(a)
    ind=np.argmax(a)
    # print(ind)
    predicted_disease = disease_class[ind]
    print('Prediction:', predicted_disease)
    return render_template('index.html', res = predicted_disease)

if __name__ == "__main__":
    app.run(debug = True)