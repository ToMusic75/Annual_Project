import base64
import numpy as np
import io
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
import os
from flask import Flask

app = Flask(__name__)
modelsDirectory = "./models/"
models = {}

def get_models():
    for model in os.listdir(modelsDirectory):
        models[model] = load_model(modelsDirectory + model)
        print(model + " loaded !")

def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image.mean(axis=-1)
    image = np.expand_dims(image, axis = 3)
    image = image / 255.0
    return image

print("Loading models")
get_models()

@app.route("/predict", methods=['POST'])
def predict():
    message = request.get_json(force=True)
    model_id = message["model_id"]
    encoded = message["image"]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(128,128))

    prediction = models[model_id].predict(processed_image).to_list()

    response = {
        'prediction': {
            'peugeot': prediction[0][0],
            'renault': prediction[0][1],
            'volkswagen': prediction[0][2],
        }
    }
    return jsonify(response)