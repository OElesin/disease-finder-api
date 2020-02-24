import os

from flask import Flask, json
import flask
import pickle
import torch
import numpy as np
import base64
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms

MODEL_PATH = './api/model_dir/plant-disease-model-cpu.pt'
# outline output classes
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
CLEANED_CLASS_NAMES = list(map(lambda x: " ".join(x.replace('_', ' ').split()), CLASS_NAMES))


# load
class PredictionService:
    """
    Singleton for holding the PyTorch model.
    It has a predict function that does inference based on the model and input data
    """
    model = None

    @classmethod
    def load_model(cls):
        """Load AutoGluon Tabular task model for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            cls.model = pickle.load(open(MODEL_PATH, 'rb'))
            print("Model Loaded")
        return cls.model

    @classmethod
    def predict(cls, prediction_input):
        """For the input, do the predictions and return them.
        Args:
            prediction_input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""

        print("Prediction Data: ")
        cls.load_model()
        output = cls.model(prediction_input)
        _, preds = torch.max(output, 1)
        class_index = int(preds.numpy())
        return CLEANED_CLASS_NAMES[class_index]


input_image_transform_fn = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_image_bytes(image_bytes):
    """
    :param image_bytes:
    :return:
    """
    image = Image.open(BytesIO(image_bytes))
    inputs = input_image_transform_fn(image)
    return inputs[None]


def is_base64(s):
    try:
        return base64.b64encode(base64.b64decode(s)) == s
    except Exception:
        return False


def error_response(message=None, error_code=415):
    return flask.Response(
        response=json.dumps({'errorMessage': message or 'Something Went wrong over the wire'}),
        status=error_code, mimetype='application/json'
    )


def get_prediction(image_bytes):
    """
    :param image_bytes:
    :return:
    """
    image_tensor = load_image_bytes(image_bytes)
    predicted_class = PredictionService.predict(image_tensor)
    return {'prediction': predicted_class}


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'app.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/ping')
    def ping():
        health = PredictionService.load_model() is not None
        status = 200 if health else 404
        return flask.Response(response='\n', status=status, mimetype='application/json')

    @app.route('/invocations', methods=['POST'])
    def transformation():
        """
        Inference on a single image. Accepts image URL
        :return:
        """
        prediction = 'hello world'
        request_input = flask.request.data.decode('utf-8')
        print(f'Request Content Type: {flask.request.content_type}')
        if flask.request.content_type == 'application/json':
            data = json.loads(request_input)
        else:
            return error_response('This predictor only supports Image URL or Image Bytes base64 string.')
        if 'ImageUrl' in data:
            print("Accepts Image Url")
            image_url = data['ImageUrl']
            print(f"Image URL: {image_url}")
            image_bytes = requests.get(image_url)
            prediction = get_prediction(image_bytes.content)
            return flask.Response(response=json.dumps(prediction), status=200, mimetype='application/json')
        elif 'ImageBytes' in data:
            print("Accepts Image Bytes")
            image_base64_string = data['ImageBytes']
            base64_check = is_base64(image_base64_string)
            if base64_check:
                image_bytes = base64.decodebytes(image_base64_string.encode())
                prediction = get_prediction(image_bytes)
                return flask.Response(response=json.dumps(prediction), status=200, mimetype='application/json')
            else:
                return error_response('Error decoding base64 image bytes.')
        else:
            return error_response('Missing ImageUrl key in request payload.')
    return app
