from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# =====================================================
# APP SETUP
# =====================================================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "model/plant_disease_model.h5"
LABELS_PATH = "labels.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =====================================================
# LOAD MODEL & LABELS (ONCE AT STARTUP)
# =====================================================
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# =====================================================
# IMAGE PREPROCESSING (MUST MATCH TRAINING)
# =====================================================
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =====================================================
# PREDICTION ENDPOINT
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run inference
    img = preprocess_image(file_path)
    preds = model.predict(img)

    # ===============================
    # TOP-3 PREDICTIONS (NEW)
    # ===============================
    top_indices = preds[0].argsort()[-3:][::-1]

    top_predictions = []
    for i in top_indices:
        label = labels[str(i)]
        plant = label.split("___")[0]  # Extract plant name

        top_predictions.append({
            "plant": plant,
            "disease": label,
            "confidence": round(float(preds[0][i]) * 100, 2)
        })

    # ===============================
    # PRIMARY PREDICTION
    # ===============================
    primary = top_predictions[0]

    return jsonify({
        "plant": primary["plant"],
        "disease": primary["disease"],
        "confidence": primary["confidence"],
        "top_predictions": top_predictions
    })

# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
