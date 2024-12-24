from flask import Flask, request, jsonify, send_file, render_template
from tensorflow.keras.models import load_model
import numpy as np
import os
import rasterio
from PIL import Image

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

    # Preprocess the image
    processed_img = preprocess_image(file_path)

    # Make prediction
    prediction = model.predict(processed_img)

    # Post-process the output
    output = np.squeeze(prediction, axis=0)  # Remove the batch dimension
    output = (output > 0.5).astype(np.uint8)  # Apply thresholding for binary mask

    # Check the shape of the output
    print(f"Output shape: {output.shape}")

    # Convert the output to a proper 2D array (128, 128) if necessary
    if len(output.shape) == 3:  # If the output is (128, 128, 1), squeeze the last dimension
        output = np.squeeze(output, axis=-1)

    # Load the original image to overlay the mask on it
    with rasterio.open(file_path) as img_src:
        original_image = img_src.read([1, 2, 3])  # Assuming RGB bands for visualization
        original_image = np.transpose(original_image, (1, 2, 0))  # Shape (height, width, 3)
    

    mask_image = Image.fromarray(output * 255)  # Convert to grayscale mask (0 or 255)
    mask_image = mask_image.resize((original_image.shape[1], original_image.shape[0]))

    # Overlay the mask on the original image with transparency
    original_image_with_mask = Image.fromarray(original_image.astype('uint8'))
    original_image_with_mask.paste(mask_image, (0, 0), mask_image)  # Paste with transparency

    # Save the final image with the overlay
    overlay_path = os.path.join('temp', 'overlay_image.png')
    original_image_with_mask.save(overlay_path)

    # Clean up the input file
    os.remove(file_path)

    # Return the overlay image as a response
    return send_file(overlay_path, mimetype='image/png')

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    app.run(debug=True)
