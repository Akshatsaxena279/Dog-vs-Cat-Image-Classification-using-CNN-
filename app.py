import os
import random
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
import requests

app = Flask(__name__)

# --- Model Loading ---
MODEL_URL = "https://drive.google.com/uc?export=download&id=1a4Rih30BD5QaZG44uzjAG0tMm7uLrEpP"  # <--- IMPORTANT: REPLACE THIS URL
MODEL_PATH = "catvsdog_fixed.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded successfully.")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            return False
    return True

model = None
if download_model():
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
# --- End Model Loading ---


# Preprocess the image
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded!'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            # Save the file to a temporary location
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            image_path = os.path.join('uploads', file.filename)
            file.save(image_path)

            # Preprocess the image and make a prediction
            processed_image = preprocess_image(image_path)
            if processed_image is not None:
                prediction = model.predict(processed_image)
                predicted_class = 'Dog' if prediction[0][0] > 0.5 else 'Cat'
                confidence = float(prediction[0][0]) if predicted_class == 'Dog' else 1.0 - float(prediction[0][0])
                
                os.remove(image_path)

                return jsonify({
                    'prediction': predicted_class,
                    'confidence': f'{confidence:.2f}'
                })
            else:
                return jsonify({'error': 'Failed to preprocess image'})
        except Exception as e:
            return jsonify({'error': str(e)})

    return jsonify({'error': 'Something went wrong'})

@app.route('/predict_random', methods=['GET'])
def predict_random():
    if model is None:
        return jsonify({'error': 'Model is not loaded!'})

    try:
        test_dir = r"dogs-vs-cats/test"
        cat_dir = os.path.join(test_dir, 'cats')
        dog_dir = os.path.join(test_dir, 'dogs')

        if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
            return jsonify({'error': 'Test directories not found.'})

        category = random.choice(['cat', 'dog'])
        img_dir = cat_dir if category == 'cat' else dog_dir
        
        if not os.listdir(img_dir):
            return jsonify({'error': f'No images found in {img_dir}'})

        img_name = random.choice(os.listdir(img_dir))
        image_path = os.path.join(img_dir, img_name)

        processed_image = preprocess_image(image_path)
        if processed_image is not None:
            prediction = model.predict(processed_image)
            predicted_class = 'Dog' if prediction[0][0] > 0.5 else 'Cat'
            confidence = float(prediction[0][0]) if predicted_class == 'Dog' else 1.0 - float(prediction[0][0])

            # Encode image to base64
            buffered = BytesIO()
            img = Image.open(image_path)
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data = f'data:image/jpeg;base64,{img_str}'

            return jsonify({
                'prediction': predicted_class,
                'confidence': f'{confidence:.2f}',
                'image_data': image_data,
                'actual': category
            })
        else:
            return jsonify({'error': 'Failed to preprocess image'})

    except Exception as e:
        return jsonify({'error': str(e)})