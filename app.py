from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf

app = Flask(__name__)

# Initialize models
models = {
    "MobileNet V3": load_mobilenet_model(),
    "ResNet": load_resnet_model(),
    "InceptionNet": load_inception_model()
}

detector = MTCNN()

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/inference", methods=["POST"])
def inference():
    image_file = request.files["image"]
    model_name = request.form["model"]
    image = Image.open(image_file).convert("RGB")
    img_array = preprocess_image(image)

    predictions = models[model_name].predict(img_array)
    deepfake_prob = float(predictions[0][0])

    return jsonify({"results": f"Deepfake Probability: {deepfake_prob:.2%}"})

if __name__ == "__main__":
    app.run(debug=True)
