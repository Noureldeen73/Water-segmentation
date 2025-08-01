import os
import io
import base64
import numpy as np
from PIL import Image
import tifffile
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from Util.model import createDeepLabv3

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

RESULTS_FOLDER = 'results'
os.makedirs(RESULTS_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_EXTENSIONS = {'tif', 'tiff'}

SELECTED_BANDS = [i for i in range(12) if i not in (0, 7, 8, 9)] 

model = None

def load_model():
    """Load the trained model from pretrained_segmentation.pth"""
    global model
    try:
        model_path = 'Pretrained_segmentation.pth'
        
        if os.path.exists(model_path):
            try:
                model = createDeepLabv3(input_channels=8, output_channels=1)
                
                state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state_dict)
                
                model.to(DEVICE)
                model.eval()
                print(f"Your trained model loaded successfully from: {model_path}")
                return True
            except Exception as e:
                print(f"Failed to load from {model_path}: {e}")
        
        # If no .pth file found, create fresh architecture
        print("⚠️ No trained model found, creating fresh model architecture...")
        model = createDeepLabv3(input_channels=8, output_channels=1)
        model.to(DEVICE)
        model.eval()
        print("Model architecture created, but no pretrained weights loaded")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def normalize_image(image):
    """Normalize image bands"""
    for b in range(image.shape[0]):
        band = image[b]
        min_val = band.min()
        max_val = band.max()
        image[b] = (band - min_val) / (max_val - min_val + 1e-8)
    return image

def preprocess_image(image_path):
    """Preprocess TIF image for prediction"""
    try:
        image = tifffile.imread(image_path).astype(np.float32)
        
        if len(image.shape) == 3 and image.shape[2] >= 12:
            image = image[:, :, SELECTED_BANDS]
        else:
            raise ValueError(f"Image must have at least 12 bands, got shape: {image.shape}")
        
        image = np.transpose(image, (2, 0, 1))
    
        image = normalize_image(image)
        
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(image_tensor):
    """Make prediction on preprocessed image"""
    global model
    if model is None:
        return None
    
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(DEVICE)
            outputs = model(image_tensor)['out']
            prediction = torch.sigmoid(outputs).cpu().numpy()
            prediction = (prediction > 0.4).astype(np.uint8)
        return prediction[0, 0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_visualization(original_image, prediction, filename):
    """Create a visualization of the original image and prediction"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if original_image.shape[2] >= 3:
            rgb_image = original_image[:, :, :3]
            # Normalize for display
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
            axes[0].imshow(rgb_image)
        else:
            axes[0].imshow(original_image[:, :, 0], cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(prediction, cmap='gray')
        axes[1].set_title('Water Segmentation Prediction')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        result_path = os.path.join(RESULTS_FOLDER, f'result_{filename}.png')
        plt.savefig(result_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return result_path
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure the model file exists and is properly configured.'
            }), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            image_tensor = preprocess_image(filepath)
            if image_tensor is None:
                return jsonify({'error': 'Failed to preprocess image'}), 500
            
            prediction = predict_image(image_tensor)
            if prediction is None:
                return jsonify({'error': 'Failed to make prediction'}), 500
            
            original_image = tifffile.imread(filepath).astype(np.float32)
            
            result_path = create_visualization(original_image, prediction, filename)
            if result_path is None:
                return jsonify({'error': 'Failed to create visualization'}), 500
            
            total_pixels = prediction.size
            water_pixels = np.sum(prediction)
            water_percentage = (water_pixels / total_pixels) * 100
            
            with open(result_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'water_percentage': round(water_percentage, 2),
                'total_pixels': int(total_pixels),
                'water_pixels': int(water_pixels),
                'result_image': img_base64,
                'filename': filename
            })
        
        else:
            return jsonify({'error': 'Invalid file type. Please upload a TIF/TIFF file.'}), 400
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """Endpoint to load a specific model"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        
        if not model_path or not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 400
        
        success = load_model()
        
        if success:
            return jsonify({'success': True, 'message': f'Model loaded from {model_path}'})
        else:
            return jsonify({'error': 'Failed to load model'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Error loading model: {str(e)}'}), 500

@app.route('/model_status')
def model_status():
    """Check if model is loaded"""
    is_loaded = model is not None
    return jsonify({
        'model_loaded': is_loaded,
        'device': DEVICE,
        'selected_bands': SELECTED_BANDS
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)