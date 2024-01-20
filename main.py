from tensorflow import keras,argmax,convert_to_tensor
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import base64

# Load the model
model = keras.models.load_model('plant.h5')
class_names=[
{'PlanetName': 'Apple',       'PlanetDisease': 'Apple scab', 'Id': 1},
{'PlanetName': 'Apple',       'PlanetDisease': 'Black rot', 'Id': 2}, 
{'PlanetName': 'Apple',       'PlanetDisease': 'Cedar apple rust', 'Id': 3},
{'PlanetName': 'Apple',       'PlanetDisease': 'healthy', 'Id': 4},
{'PlanetName': 'Blueberry',   'PlanetDisease': 'healthy', 'Id': 5},
{'PlanetName': 'Cherry',      'PlanetDisease': 'Powdery mildew', 'Id': 6},
{'PlanetName': 'Cherry',      'PlanetDisease': 'healthy', 'Id': 7},
{'PlanetName': 'Corn',        'PlanetDisease': 'Cercospora leaf spot Gray leaf spot', 'Id': 8}, 
{'PlanetName': 'Corn',        'PlanetDisease': 'Common rust', 'Id': 9},
{'PlanetName': 'Corn',        'PlanetDisease': 'Northern Leaf Blight', 'Id': 10}, 
{'PlanetName': 'Corn',        'PlanetDisease': 'healthy', 'Id': 11},
{'PlanetName': 'Grape',       'PlanetDisease': 'Black rot', 'Id': 12},
{'PlanetName': 'Grape',       'PlanetDisease': 'Esca Black Measles', 'Id': 13},
{'PlanetName': 'Grape',       'PlanetDisease': 'Leaf blight Isariopsis Leaf Spot', 'Id': 14},
{'PlanetName': 'Grape',       'PlanetDisease': 'healthy', 'Id': 15}, 
{'PlanetName': 'Orange',      'PlanetDisease': 'Huanglongbing Citrus greening', 'Id': 16},
{'PlanetName': 'Peach',       'PlanetDisease': 'Bacterial spot', 'Id': 17},
{'PlanetName': 'Peach',       'PlanetDisease': 'healthy', 'Id': 18},
{'PlanetName': 'Pepper bell', 'PlanetDisease': 'Bacterial spot', 'Id': 19},
{'PlanetName': 'Pepper bell', 'PlanetDisease': 'healthy', 'Id': 20},
{'PlanetName': 'Potato',      'PlanetDisease': 'Early blight', 'Id': 21},
{'PlanetName': 'Potato',      'PlanetDisease': 'Late blight', 'Id': 22}, 
{'PlanetName': 'Potato',      'PlanetDisease': 'healthy', 'Id': 23},  
{'PlanetName': 'Raspberry',   'PlanetDisease': 'healthy', 'Id': 24},
{'PlanetName': 'Soybean',     'PlanetDisease': 'healthy', 'Id': 25},
{'PlanetName': 'Squash',      'PlanetDisease': 'Powdery mildew', 'Id': 26},
{'PlanetName': 'Strawberry',  'PlanetDisease': 'Leaf scorch', 'Id': 27},
{'PlanetName': 'Strawberry',  'PlanetDisease': 'healthy', 'Id': 28}, 
{'PlanetName': 'Tomato',      'PlanetDisease': 'Bacterial spot', 'Id': 29}, 
{'PlanetName': 'Tomato',      'PlanetDisease': 'Early blight', 'Id': 30},
{'PlanetName': 'Tomato',      'PlanetDisease': 'Late blight', 'Id': 31},
{'PlanetName': 'Tomato',      'PlanetDisease': 'Leaf Mold', 'Id': 32},
{'PlanetName': 'Tomato',      'PlanetDisease': 'Septoria leaf spot', 'Id': 33},
{'PlanetName': 'Tomato',      'PlanetDisease': 'Spider mites Two-spotted spider mite', 'Id': 34},
{'PlanetName': 'Tomato',      'PlanetDisease': 'Target Spot', 'Id': 35},  
{'PlanetName': 'Tomato',      'PlanetDisease': 'Tomato Yellow Leaf Curl Virus', 'Id': 36}, 
{'PlanetName': 'Tomato',      'PlanetDisease': 'Tomato mosaic virus', 'Id': 37},
{'PlanetName': 'Tomato',      'PlanetDisease': 'healthy', 'Id': 38}
]

# Define the classification function
def classify_image(image_data):
    # Decode the base64 encoded image data
    image_data = base64.b64decode(image_data)

    # Create a PIL Image from the image data
    image = Image.open(io.BytesIO(image_data))

    # Preprocess the image
    image = image.convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0

    # Make the prediction
    predictions = model.predict(convert_to_tensor([image]))
    predicted_class_index = argmax(predictions, axis=1).numpy()[0]
    predicted_class_label = class_names[predicted_class_index]

    return predicted_class_label

# Define the Flask application
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
   
    return "<h1>By: @usfnassar</h1>"

@app.route('/data', methods=['GET'])
def get_unhealthy_planets():
    unhealthy_planets = []

    for entry in class_names:
        if entry['PlanetDisease'] != 'healthy':
            unhealthy_planets.append(entry)

    return jsonify({'data':unhealthy_planets})

@app.route('/getid', methods=['GET'])
def get_planet_id():
    planet_disease=request.json['imageDisease']

    planet_id = [entry['Id'] for entry in class_names if entry['PlanetDisease'].lower() == planet_disease.lower()]

    return jsonify({'id': planet_id[0]})
    
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
