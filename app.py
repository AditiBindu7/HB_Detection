from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

model = tf.keras.models.load_model("hemoglobin_model.h5")
IMG_SIZE = 128
LABELS = ["moderate anemia", "mild anemia", "normal"]

def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    return img

@app.route('/')
def home():
    return render_template('index.html')  #  Loads index.html from templates/

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img = preprocess_image(file)
    prediction = model.predict(img)
    pred_idx = np.argmax(prediction)
    result = LABELS[pred_idx]
    confidence = float(prediction[0][pred_idx])

    return jsonify({
        "hb_level": result,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(debug=True)