from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = "terrain.h5"
model = tf.keras.models.load_model(MODEL_PATH)

TERRAIN_CLASSES = {
    0: "Grasslands",
    1: "Marshland",
    2: "Other",
    3: "Rocky Terrain",
    4: "Sandy Terrain",
    5: "Water"
}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join("static/uploads", filename)
        file.save(file_path)

        # Process image
        img = image.load_img(file_path, target_size=(255, 255))  # Ensure correct size
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the image
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        # Get terrain type
        terrain_type = TERRAIN_CLASSES.get(predicted_class, "Unknown Terrain")

        return jsonify({"prediction": terrain_type, "file_path": file_path})

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)
