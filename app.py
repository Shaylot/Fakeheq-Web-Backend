from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from mtcnn import MTCNN
from flask_cors import CORS
import tensorflow as tf

# Initialize MTCNN detector
detector = MTCNN()

# Preload TFLite models
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load the TFLite models
tflite_models = {
    "MobileNet V3": load_tflite_model("mobilenet.tflite"),
    "ResNet": load_tflite_model("resnet.tflite"),
    "InceptionNet": load_tflite_model("inception.tflite")
}

# Preprocess the image for model input
def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")  # Convert to RGB to remove alpha channel
    img = img.resize((224, 224))  # Resize to match model input shape
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimension
    return img_array

# Perform inference using TFLite interpreter
def run_tflite_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

@app.route("/inference", methods=["POST"])
def inference():
    try:
        image_file = request.files["image"]
        model_name = request.form["model"]

        if model_name not in tflite_models:
            return jsonify({"error": "Invalid model name"}), 400
        
        # Get the selected TFLite model
        interpreter = tflite_models[model_name]
        
        # Preprocess image
        img_array = preprocess_image(image_file)
        
        # Perform inference
        prediction = run_tflite_inference(interpreter, img_array)
        deepfake_prob = prediction[0][0]  # Assuming the 1st output neuron indicates Deepfake
        
        return jsonify({"results": f"Deepfake Probability: {deepfake_prob:.2%}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/inferenceAll", methods=["POST"])
def inferenceAll():
    try:
        image_file = request.files["image"]

        results = [["Model Name", "Deepfake Probability"]]

        for model_name, interpreter in tflite_models.items():
            # Preprocess image
            img_array = preprocess_image(image_file)
            
            # Perform inference
            prediction = run_tflite_inference(interpreter, img_array)
            deepfake_prob = prediction[0][0]  # Assuming the 1st output neuron indicates Deepfake
            
            results.append([model_name, f"{deepfake_prob:.2%}"])
        
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def base():
    return jsonify({"message": "hi"}), 200

# Run the application
if __name__ == "__main__":
    app.run(debug=True)
