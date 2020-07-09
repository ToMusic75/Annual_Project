import base64
import re
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
import os
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
modelsDirectory = "models/"
models = {}

def get_models():
    for model in os.listdir(modelsDirectory):
        models[model] = load_model(os.path.join(modelsDirectory, model))
        print(model + " loaded !")

def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.mean(image, -1)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)
    image = image / 255.0
    print(image.shape)
    return image


print("Loading models")
get_models()


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    model_id = message["model_id"]
    encoded = message["image"]
    encoded = re.sub('^data:image/.+;base64,', '', encoded)

    decoded = base64.b64decode(encoded)
    predict_image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(predict_image, (128, 128))

    prediction = models[model_id+".h5"].predict(processed_image).tolist()

    response = {
        'prediction': {
            'peugeot': prediction[0][0],
            'renault': prediction[0][1],
            'volkswagen': prediction[0][2],
            #'peugeot': 1,
            #'renault': 0,
            #'volkswagen': 0,
        }
    }
    return jsonify(response), 201
