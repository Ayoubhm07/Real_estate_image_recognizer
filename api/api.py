from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras import models
from PIL import Image, ImageOps
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)
model = models.load_model('../model/my_model.keras')


def prepare_image(image):
    image_bytes = io.BytesIO(image.read())
    img = Image.open(image_bytes)  # Utiliser Image.open pour ouvrir l'image
    img = img.convert('RGB')  # Assurer le format correct

    # Redimensionner l'image en utilisant le resampling LANCZOS
    img = img.resize((150, 150), Image.Resampling.LANCZOS)

    # Convertir en array
    img_array = np.array(img)

    # Normalisation
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file and allowed_file(file.filename):
        img = prepare_image(file)
        result = model.predict(img)
        class_prediction = 'property' if result[0][0] > 0.5 else 'not a property'
        return jsonify({'prediction': class_prediction})
    else:
        return jsonify({'error': 'Invalid file format'}), 400


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


if __name__ == '__main__':
    app.run(debug=True)
