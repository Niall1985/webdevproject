from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
import cv2  # OpenCV for video processing
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

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "avi", "mov"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

def extract_frames(video_path, frame_interval=30):
    """
    Extract frames from the video every `frame_interval` frames.
    Returns a list of extracted frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (255, 255))  # Resize to model input size
            frame = frame.astype("float32") / 255.0  # Normalize pixel values
            frames.append(frame)

        frame_count += 1

    cap.release()
    return np.array(frames)

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

        file_extension = filename.rsplit('.', 1)[1].lower()

        if file_extension in {"png", "jpg", "jpeg"}:
            # Process image
            img = image.load_img(file_path, target_size=(255, 255))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            terrain_type = TERRAIN_CLASSES.get(predicted_class, "Unknown Terrain")

            return jsonify({"prediction": terrain_type, "file_path": file_path})

        elif file_extension in {"mp4", "avi", "mov"}:
            # Process video frame by frame
            frames = extract_frames(file_path)
            
            if len(frames) == 0:
                return jsonify({"error": "No frames extracted from video"}), 400

            predictions = model.predict(frames)
            predicted_classes = np.argmax(predictions, axis=1)

            # Get the most frequent terrain type in the video
            unique, counts = np.unique(predicted_classes, return_counts=True)
            most_frequent_class = unique[np.argmax(counts)]
            terrain_type = TERRAIN_CLASSES.get(most_frequent_class, "Unknown Terrain")

            return jsonify({"prediction": terrain_type, "file_path": file_path})

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)
