
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model_loader import load_glaucoma_model, predict_image

app = Flask(__name__)
CORS(app)

# Path to trained model
MODEL_PATH = os.path.join("trained_model", "VisionTransformer_glaucoma_model.pth")

# Load model once
model = load_glaucoma_model(MODEL_PATH)

@app.get("/")
def index():
    return jsonify({"message": "Glaucoma Detection API running!"})

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image_file = request.files["image"]
    save_path = "uploaded_image.jpg"
    image_file.save(save_path)

    predicted_class, probability = predict_image(model, save_path)

    return jsonify({
        "prediction": int(predicted_class),      # 0 or 1
        "confidence": float(probability) * 100   # Convert to %
    })

if __name__ == "__main__":
    app.run(debug=True)
