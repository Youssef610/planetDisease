import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np

import base64

# Load the model
model = tf.keras.models.load_model('plant.h5')
class_names=[
    {'PlanetName': 'Apple', 'PlanetDisease': 'Apple scab'},
    {'PlanetName': 'Apple', 'PlanetDisease': 'Black rot'},
    {'PlanetName': 'Apple', 'PlanetDisease': 'Cedar apple rust'},
    {'PlanetName': 'Apple', 'PlanetDisease': 'healthy'},
    {'PlanetName': 'Blueberry', 'PlanetDisease': 'healthy'},
    {'PlanetName': 'Cherry', 'PlanetDisease': 'Powdery mildew'},
    {'PlanetName': 'Cherry', 'PlanetDisease': 'healthy'},
    {'PlanetName': 'Corn', 'PlanetDisease': 'Cercospora leaf spot Gray leaf spot'},
    {'PlanetName': 'Corn', 'PlanetDisease': 'Common rust'},
    {'PlanetName': 'Corn', 'PlanetDisease': 'Northern Leaf Blight'},
    {'PlanetName': 'Corn', 'PlanetDisease': 'healthy'},
    {'PlanetName': 'Grape', 'PlanetDisease': 'Black rot'},
    {'PlanetName': 'Grape', 'PlanetDisease': 'Esca Black Measles'},
    {'PlanetName': 'Grape', 'PlanetDisease': 'Leaf blight Isariopsis Leaf Spot'},
    {'PlanetName': 'Grape', 'PlanetDisease': 'healthy'},
    {'PlanetName': 'Orange', 'PlanetDisease': 'Huanglongbing Citrus greening'},
    {'PlanetName': 'Peach', 'PlanetDisease': 'Bacterial spot'},
    {'PlanetName': 'Peach', 'PlanetDisease': 'healthy'},
    {'PlanetName': 'Pepper bell', 'PlanetDisease': 'Bacterial spot'},
    {'PlanetName': 'Pepper bell', 'PlanetDisease': 'healthy'},
    {'PlanetName': 'Potato', 'PlanetDisease': 'Early blight'},
    {'PlanetName': 'Potato', 'PlanetDisease': 'Late blight'},
    {'PlanetName': 'Potato', 'PlanetDisease': 'healthy'},
    {'PlanetName': 'Raspberry', 'PlanetDisease': 'healthy'},
    {'PlanetName': 'Soybean', 'PlanetDisease': 'healthy'},
    {'PlanetName': 'Squash', 'PlanetDisease': 'Powdery mildew'},
    {'PlanetName': 'Strawberry', 'PlanetDisease': 'Leaf scorch'},
    {'PlanetName': 'Strawberry', 'PlanetDisease': 'healthy'},
    {'PlanetName': 'Tomato', 'PlanetDisease': 'Bacterial spot'},
    {'PlanetName': 'Tomato', 'PlanetDisease': 'Early blight'},
    {'PlanetName': 'Tomato', 'PlanetDisease': 'Late blight'},
    {'PlanetName': 'Tomato', 'PlanetDisease': 'Leaf Mold'},
    {'PlanetName': 'Tomato', 'PlanetDisease': 'Septoria leaf spot'},
    {'PlanetName': 'Tomato', 'PlanetDisease': 'Spider mites Two-spotted spider mite'},
    {'PlanetName': 'Tomato', 'PlanetDisease': 'Target Spot'},
    {'PlanetName': 'Tomato', 'PlanetDisease': 'Tomato Yellow Leaf Curl Virus'},
    {'PlanetName': 'Tomato', 'PlanetDisease': 'Tomato mosaic virus'},
    {'PlanetName': 'Tomato', 'PlanetDisease': 'healthy'}
]

# Define the classification function
def classify_image(image_data):
    # Decode the base64 encoded image data
    image_data = base64.b64decode(image_data)

    # Create a NumPy array from the image data
    image_np = np.frombuffer(image_data, dtype=np.uint8)

    # Convert the NumPy array to an image
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0

    # Make the prediction
    predictions = model.predict(tf.convert_to_tensor([image]))
    predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_class_label = class_names[predicted_class_index]

    return predicted_class_label

# Define the Flask application
app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    # Get the image data from the request
    image_data = request.json['image_data']

    # Classify the image
    predicted_class_label = classify_image(image_data)

    # Return the classification result
    return jsonify({'predicted_class_label': predicted_class_label})

if __name__ == '__main__':
    app.run(host="0.0.0.0")
