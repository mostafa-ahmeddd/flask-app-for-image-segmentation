from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import os
import rasterio

app = Flask(__name__)

model = load_model('C:/Users/mosta/Downloads/flask app/modell-46-0.9264.keras')

def normalize_band(band, min_val, max_val):
    if max_val != min_val:
        normalized_band = (band - min_val) / (max_val - min_val)
    else:
        normalized_band = np.zeros_like(band)
    return normalized_band

def replace_nan_with_value(array):
    array[np.isnan(array)] = 0  # Replace NaNs with 0
    return array

def preprocess_image(image_path):
    with rasterio.open(image_path) as img_src:
        image = img_src.read()  # Read all bands
        # Normalize each band individually
        normalized_image = np.zeros_like(image, dtype='float32')
        for band_idx in range(image.shape[0]):
            band = image[band_idx]
            min_val = np.nanmin(band)
            max_val = np.nanmax(band)
            band = replace_nan_with_value(band)
            normalized_image[band_idx] = normalize_band(band, min_val, max_val)
            
        normalized_image = np.transpose(normalized_image, (1, 2, 0))
    return np.expand_dims(normalized_image, axis=0) 

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Save the file temporarily and process it
    file_path = os.path.join('temp', file.filename)
    file.save(file_path)

    # Preprocess the image according to your specific normalization
    processed_img = preprocess_image(file_path)

    # Make prediction
    prediction = model.predict(processed_img)

    # Post-process the output as needed for your segmentation task
    output = np.squeeze(prediction, axis=0)
    output = (output > 0.5).astype(np.uint8)  # Example thresholding

    # Clean up the saved file
    os.remove(file_path)

    return jsonify({"segmentation_mask": output.tolist()})

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    app.run(debug=True)
