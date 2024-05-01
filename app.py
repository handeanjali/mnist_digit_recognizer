# app.py
import os
# You can use 'tensorflow', 'torch' or 'jax' as backend to keras. Make sure to set the environment variable before importing.
os.environ["KERAS_BACKEND"] = "tensorflow"
#TensorFlow is using oneDNN library for performance enhancements.(This worning will not pop up if you are using GPU)
#Need to put the below code on top of your code before "import keras" code line to suppress these warning messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# -----------------------------------------------------------------------------------------------------------------------
from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from PIL import Image
from flask import jsonify
import io
import csv


app = Flask(__name__)

# Load the trained model
model = load_model('mnist_model.keras')


# Define function to preprocess input image
def preprocess_image(image_stream):
    # Open the image using PIL
    image = Image.open(image_stream)
    # Resize image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert image to grayscale
    image = image.convert('L')
    # Convert image to numpy array
    image = np.array(image)
    # Normalize pixel values
    image = image.astype('float32') / 255
    # Reshape image for model input
    image = np.expand_dims(image, axis=(0, 3))
    return image


# Define route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get image file from POST request
    image_file = request.files['image']
    # Read image data
    image_data = image_file.read()
    # Convert image data to BytesIO object
    image_stream = io.BytesIO(image_data)
    # Preprocess the image
    image = preprocess_image(image_stream)
    # Perform inference
    predictions = model.predict(image)
    # Convert predictions to class label (Python integer)
    predicted_class = int(np.argmax(predictions))
    # Return prediction result as JSON
    return jsonify(prediction=predicted_class)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
