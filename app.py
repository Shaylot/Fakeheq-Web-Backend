from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from flask_cors import CORS
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Initialize MTCNN detector
detector = MTCNN()

# Load the pre-trained MobileNet model
def build_mobilenet_model():
    model1 = keras.applications.MobileNet(input_shape=(224, 224, 3), weights="imagenet")
    model1.trainable = True
    inputs = Input(shape=(224, 224, 3))
    model = Sequential([
        inputs,
        model1,
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(2, activation='softmax')
    ])
    return model

# Load the pre-trained InceptionNet model
def build_inception_model():
    inputs = Input(shape=(224, 224, 3))
    inception = keras.applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    model = Sequential([
        inputs,
        inception,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load the pre-trained ResNet model
def build_resnet_model():
    inputs = Input(shape=(224, 224, 3))
    resnet = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    model = Sequential([
        inputs,
        resnet,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize models
models = {
    "MobileNet V3": build_mobilenet_model(),
    "ResNet": build_resnet_model(),
    "InceptionNet": build_inception_model()
}

# Global variables for the selected model
current_model = None
current_model_name = None

# Function to load the selected model
def load_model(model_name):
    global current_model, current_model_name
    if model_name == "MobileNet V3":
        current_model = build_mobilenet_model()
        current_model.load_weights("mobilenet.h5")  # Update the path to your MobileNet weights file
    elif model_name == "ResNet":
        current_model = build_resnet_model()
        current_model.load_weights("model.weights.h5")  # Update the path to your ResNet weights file
    elif model_name == "InceptionNet":
        current_model = build_inception_model()
        current_model.load_weights("inception.weights.h5")  # Update the path to your InceptionNet weights file
    current_model_name = model_name


# Preprocess the image for model input
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Convert image to RGB to remove alpha channel
    img = img.resize((224, 224))  # Resize to match model input shape
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


app = Flask(__name__)
CORS(app)  # Enable CORS for all origins


# Inference endpoint
@app.route("/inference", methods=["POST"])
def inference():
    try:
        image_file = request.files["image"]
        model_name = request.form["model"]

        if model_name not in models:
            return jsonify({"error": "Invalid model name"}), 400
        load_model(model_name)
        img_array = preprocess_image(image_file)
        prediction = current_model.predict(img_array)
        print("yo")
        deepfake_prob = prediction[0][0]
        
        return jsonify({"results": f"Deepfake Probability: {deepfake_prob:.2%}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Inference endpoint
@app.route("/inferenceAll", methods=["POST"])
def inferenceAll():
    try:
        image_file = request.files["image"]
        model_name = request.form["model"]

        results = [["Model Name", "Deepfake Probability"]]

        for model_name in models:
            load_model(model_name)  # Load the current model
            if current_model:
                img_array = preprocess_image(image_file)
                prediction = current_model.predict(img_array)
                deepfake_prob = prediction[0][0]  # Assuming the 1st neuron indicates Deepfake
                results.append([model_name, f"{deepfake_prob:.2%}"])
                print(results)
        
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/", methods=["GET"])
def base():
    return jsonify({"message": "hi"}), 200


# Run the application
if __name__ == "__main__":
    app.run(debug=True)
