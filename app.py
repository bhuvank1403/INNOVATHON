from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load model
print("Loading model...")
model = YOLO("best.pt")
print("Model loaded successfully!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API is running", "endpoint": "/predict"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    try:
        # Read image file
        file = request.files["image"].read()
        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Perform prediction
        results = model(img)
        probs = results[0].probs
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        label = model.names[class_id]
        
        return jsonify({
            "class": label,
            "confidence": round(confidence, 2)
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)