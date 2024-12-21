from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Placeholder functions to load models
def load_mobilenet_model():
    model = tf.keras.applications.MobileNetV3Small(weights=None)  # Adjust weights as needed
    model.build((None, 224, 224, 3))  # Ensure input shape is correct
    return model

def load_resnet_model():
    model = tf.keras.applications.ResNet50(weights=None)  # Adjust weights as needed
    model.build((None, 224, 224, 3))  # Ensure input shape is correct
    return model

def load_inception_model():
    model = tf.keras.applications.InceptionV3(weights=None)  # Adjust weights as needed
    model.build((None, 224, 224, 3))  # Ensure input shape is correct
    return model

# Initialize models
models = {
    "MobileNet V3": load_mobilenet_model(),
    "ResNet": load_resnet_model(),
    "InceptionNet": load_inception_model()
}

# Initialize MTCNN detector
detector = MTCNN()

# Preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Inference endpoint
@app.route("/inference", methods=["POST"])
def inference():
    try:
        image_file = request.files["image"]
        model_name = request.form["model"]
        if model_name not in models:
            return jsonify({"error": "Invalid model name"}), 400
        
        # Load and preprocess image
        image = Image.open(image_file).convert("RGB")
        img_array = preprocess_image(image)
        
        # Perform inference
        predictions = models[model_name].predict(img_array)
        deepfake_prob = float(predictions[0][0])
        
        return jsonify({"results": f"Deepfake Probability: {deepfake_prob:.2%}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the application
if __name__ == "__main__":
    app.run(debug=True)
