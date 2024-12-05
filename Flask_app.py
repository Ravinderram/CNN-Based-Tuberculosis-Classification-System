import os
from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import cv2
import numpy as np

app = Flask(__name__)
loaded_model = keras.models.load_model('pretrained_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # Normalize pixel values to between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_tb_with_image(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = loaded_model.predict(preprocessed_img)
    confidence_score = float (prediction[0][0])

    if confidence_score > 0.5:
        result = "TB Detected"
    else:
        result = "No TB Detected"

    return result, confidence_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        try:
            # Save the uploaded file
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Preprocess the image and make prediction
            result, confidence_score = predict_tb_with_image(file_path)

            return jsonify({'result': result, 'confidence_score': confidence_score})

        except Exception as e:
            return jsonify({'error': str(e)})

    return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
