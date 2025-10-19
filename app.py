"""
IngredientWale - Food Ingredient Prediction Flask Backend
This Flask app serves a web interface for predicting ingredients in food images
using a trained model.
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
try:
    from flask_cors import CORS
except Exception:
    CORS = None
from PIL import Image
import io
import shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # or "3" to hide warnings too
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: disables oneDNN notice
# Try to import tensorflow/keras for model loading
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.resnet50 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Using mock predictions.")
    TF_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
# NOTE (deploy): If your model file path changes in the server/container,
# update MODEL_PATH accordingly.
MODEL_PATH = 'model/Resnet50_finetuned_model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MODEL_URL_ENV = 'MODEL_URL'  # Optional: set in environment to auto-download model

# CORS configuration
# Local defaults allow common dev ports. 
# NOTE (deploy): Replace this list with your production frontend origin(s), e.g. ['https://your-domain.com']
FRONTEND_ORIGINS = [
    'http://localhost:5500',
    'http://127.0.0.1:5500',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'null',  # Allow file:// origins (browsers send Origin: null)
]

# Ensure model directory exists
os.makedirs('model', exist_ok=True)
# --- External asset download helpers (for large models on Render) ---
def download_file(url, dest_path):
    """Download a file from a URL (supports Google Drive via gdown if available)."""
    try:
        # Prefer gdown for Google Drive links
        if 'drive.google.com' in url:
            try:
                import gdown  # type: ignore
                gdown.download(url=url, output=dest_path, quiet=False, fuzzy=True)
                return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
            except Exception as e:
                print(f"gdown download failed: {e}. Falling back to requests.")
        # Generic HTTP(S) download
        import requests  # type: ignore
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
    except Exception as e:
        print(f"❌ Failed to download file from {url}: {e}")
        return False

def ensure_model_file():
    """Ensure the model file exists locally; download from MODEL_URL if provided."""
    if os.path.exists(MODEL_PATH):
        return True
    model_url = os.environ.get(MODEL_URL_ENV)
    if not model_url:
        print(f"⚠️  {MODEL_PATH} not found and {MODEL_URL_ENV} env var is not set. Using mock predictions.")
        return False
    print(f"⬇️  Downloading model from {model_url} -> {MODEL_PATH}")
    ok = download_file(model_url, MODEL_PATH)
    if ok:
        print("✅ Model downloaded successfully")
    else:
        print("❌ Model download failed; continuing without a model")
    return ok

# Enable CORS at import time so it works under Gunicorn/Render
if CORS is not None:
    try:
        CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGINS or "*"}}, supports_credentials=False)
        print(f"CORS enabled for origins: {FRONTEND_ORIGINS or '*'}")
    except Exception as e:
        print(f"Failed to enable CORS: {e}")
else:
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

# (Initialization moved to initialize_backend() at bottom to ensure definitions exist)

# Default ingredient classes (fallback when model classes can't be detected)
DEFAULT_INGREDIENT_CLASSES = [
    "Biryani", "Chole bhature", "dabeli", "Dal tadka", "Dhokla", "Dosa", "Jalebi", "Kathi Roll", "Kofta", "Naan",
    "Kadhi pakoda", "Paneer tikka masala", "Pani puri", "Pav Bhaji", "Vadapav",
]

# Global variables
model = None
ingredient_classes = DEFAULT_INGREDIENT_CLASSES.copy()
food_ingredients_df = None

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_classes_from_file():
    """Load classes from classes.txt file if it exists."""
    classes_file = 'model/classes.txt'
    if os.path.exists(classes_file):
        try:
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            if classes:
                print(f"✅ Loaded {len(classes)} classes from {classes_file}")
                return classes
        except Exception as e:
            print(f"⚠️  Error reading {classes_file}: {e}")
    return None

def load_food_ingredients():
    """Load food ingredients from CSV file."""
    global food_ingredients_df
    csv_file = 'food_ingredients.csv'
    
    if os.path.exists(csv_file):
        try:
            # Read CSV using csv module for better control
            import csv
            cleaned_data = []
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 2 and row[0].strip():  # Ensure we have dish name and at least one ingredient
                        dish_name = row[0].strip()
                        # All remaining columns are ingredients
                        ingredients = [ing.strip() for ing in row[1:] if ing.strip()]
                        
                        if ingredients:
                            cleaned_data.append({
                                'Dish Name': dish_name,
                                'Ingredients': ', '.join(ingredients)
                            })
            
            food_ingredients_df = pd.DataFrame(cleaned_data)
            
            print(f"✅ Loaded food ingredients from {csv_file}")
            print(f"   Found {len(food_ingredients_df)} dishes with ingredients")
            print(f"   Sample dishes: {list(food_ingredients_df['Dish Name'].head())}")
            return food_ingredients_df
        except Exception as e:
            print(f"⚠️  Error reading {csv_file}: {e}")
    else:
        print(f"⚠️  {csv_file} not found. Ingredient lookup will not be available.")
    
    return None

def get_dish_ingredients(dish_name):
    """Get ingredients for a specific dish from the CSV."""
    global food_ingredients_df
    
    if food_ingredients_df is None:
        return []
    
    try:
        # Clean dish name for matching - more comprehensive cleaning
        dish_name_clean = dish_name.lower().replace('_', ' ').strip()
        
        print(f"🔍 Looking for ingredients for dish: '{dish_name}' (cleaned: '{dish_name_clean}')")
        
        # Search for exact match first
        exact_match = food_ingredients_df[food_ingredients_df['Dish Name'].str.lower() == dish_name_clean]
        
        if not exact_match.empty:
            ingredients_str = exact_match.iloc[0]['Ingredients']
            ingredients = [ing.strip() for ing in ingredients_str.split(',') if ing.strip()]
            print(f"✅ Found exact match: {len(ingredients)} ingredients")
            return ingredients
        
        # If no exact match, try partial match (contains)
        partial_match = food_ingredients_df[food_ingredients_df['Dish Name'].str.lower().str.contains(dish_name_clean, na=False)]
        
        if not partial_match.empty:
            ingredients_str = partial_match.iloc[0]['Ingredients']
            ingredients = [ing.strip() for ing in ingredients_str.split(',') if ing.strip()]
            print(f"✅ Found partial match: {len(ingredients)} ingredients")
            return ingredients
        
        # Try reverse partial match (dish name contains CSV name)
        reverse_match = food_ingredients_df[food_ingredients_df['Dish Name'].str.lower().apply(
            lambda x: dish_name_clean in x if pd.notna(x) else False
        )]
        
        if not reverse_match.empty:
            ingredients_str = reverse_match.iloc[0]['Ingredients']
            ingredients = [ing.strip() for ing in ingredients_str.split(',') if ing.strip()]
            print(f"✅ Found reverse match: {len(ingredients)} ingredients")
            return ingredients
        
        # Debug: Show what dishes are available
        print(f"⚠️  No match found for '{dish_name_clean}'")
        print(f"Available dishes in CSV: {list(food_ingredients_df['Dish Name'].str.lower())}")
        
        return []
        
    except Exception as e:
        print(f"❌ Error getting ingredients for {dish_name}: {e}")
        return []

def detect_model_classes(model):
    """Detect ingredient classes from the loaded model."""
    global ingredient_classes
    
    try:
        # First, try to load classes from classes.txt file
        file_classes = load_classes_from_file()
        if file_classes:
            ingredient_classes = file_classes
            return ingredient_classes
        
        
        # Try to get output layer shape
        if hasattr(model, 'output_shape'):
            output_shape = model.output_shape
            if isinstance(output_shape, (list, tuple)) and len(output_shape) > 0:
                num_classes = output_shape[-1]
                print(f"✅ Detected {num_classes} classes from model output shape")
                
                # Generate generic class names if we can't get the actual ones
                if num_classes != len(DEFAULT_INGREDIENT_CLASSES):
                    ingredient_classes = [f"ingredient_{i+1}" for i in range(num_classes)]
                    print(f"⚠️  Using generic class names: {ingredient_classes[:5]}...")
                    print(f"💡 Tip: Create model/classes.txt with your actual class names for better results")
                else:
                    ingredient_classes = DEFAULT_INGREDIENT_CLASSES.copy()
                    print("✅ Using default ingredient classes")
                
                return ingredient_classes
        
        # Fallback: use default classes
        ingredient_classes = DEFAULT_INGREDIENT_CLASSES.copy()
        print("⚠️  Could not detect model classes, using defaults")
        return ingredient_classes
        
    except Exception as e:
        print(f"❌ Error detecting model classes: {e}")
        ingredient_classes = DEFAULT_INGREDIENT_CLASSES.copy()
        return ingredient_classes

def load_model_from_file():
    """Load the trained model with compatibility handling."""
    global model
    
    if not TF_AVAILABLE:
        print("TensorFlow not available - using mock model")
        return None
    
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}")
            
            # Try loading with custom objects to handle compatibility issues
            try:
                # First attempt: load with comprehensive custom objects for compatibility
                custom_objects = {
                    'InputLayer': tf.keras.layers.InputLayer,
                    'DTypePolicy': tf.keras.mixed_precision.Policy,
                    'mixed_float16': tf.keras.mixed_precision.Policy('mixed_float16')
                }
                model = load_model(MODEL_PATH, custom_objects=custom_objects)
                print("Model loaded successfully with comprehensive custom objects!")
            except Exception as e1:
                print(f"First loading attempt failed: {e1}")
                try:
                    # Second attempt: load with compile=False to avoid compilation issues
                    model = load_model(MODEL_PATH, compile=False)
                    print("Model loaded successfully with compile=False!")
                except Exception as e2:
                    print(f"Second loading attempt failed: {e2}")
                    try:
                        # Third attempt: load with custom_objects and compile=False
                        custom_objects = {
                            'InputLayer': tf.keras.layers.InputLayer,
                            'DTypePolicy': tf.keras.mixed_precision.Policy,
                            'mixed_float16': tf.keras.mixed_precision.Policy('mixed_float16')
                        }
                        model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
                        print("Model loaded successfully with custom objects and compile=False!")
                    except Exception as e3:
                        print(f"Third loading attempt failed: {e3}")
                        try:
                            # Fourth attempt: monkey patch InputLayer and handle dtype issues
                            original_from_config = tf.keras.layers.InputLayer.from_config
                            
                            def patched_from_config(config):
                                # Remove batch_shape if it exists and is causing issues
                                if 'batch_shape' in config:
                                    print("Removing batch_shape from InputLayer config for compatibility")
                                    config = config.copy()
                                    del config['batch_shape']
                                return original_from_config(config)
                            
                            tf.keras.layers.InputLayer.from_config = patched_from_config
                            
                            try:
                                model = load_model(MODEL_PATH, compile=False)
                                print("Model loaded successfully with patched InputLayer!")
                            finally:
                                # Restore original method
                                tf.keras.layers.InputLayer.from_config = original_from_config
                                
                        except Exception as e4:
                            print(f"Fourth loading attempt failed: {e4}")
                            try:
                                # Fifth attempt: comprehensive patching for both InputLayer and dtype issues
                                original_from_config = tf.keras.layers.InputLayer.from_config
                                
                                def patched_from_config(config):
                                    # Remove batch_shape if it exists and is causing issues
                                    if 'batch_shape' in config:
                                        print("Removing batch_shape from InputLayer config for compatibility")
                                        config = config.copy()
                                        del config['batch_shape']
                                    return original_from_config(config)
                                
                                # Patch InputLayer
                                tf.keras.layers.InputLayer.from_config = patched_from_config
                                
                                # Set up custom objects for dtype policy
                                custom_objects = {
                                    'DTypePolicy': tf.keras.mixed_precision.Policy,
                                    'mixed_float16': tf.keras.mixed_precision.Policy('mixed_float16')
                                }
                                
                                try:
                                    model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
                                    print("Model loaded successfully with comprehensive patching!")
                                finally:
                                    # Restore original method
                                    tf.keras.layers.InputLayer.from_config = original_from_config
                                    
                            except Exception as e5:
                                print(f"All loading attempts failed. Last error: {e5}")
                                raise e5
            
            # Detect and update ingredient classes from the model
            detect_model_classes(model)
            
            return model
        else:
            print(f"Error loading model: file not found at {MODEL_PATH}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_placeholder_model():
    """Placeholder model creation removed. Provide a real model at MODEL_PATH."""
    print("Placeholder model creation is disabled.")

def preprocess_image(image):
    """Preprocess image for model prediction."""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 160x160 (model input size)
        image = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess using preprocessing
        if TF_AVAILABLE:
            img_array = preprocess_input(img_array)
        else:
            # Simple normalization for mock predictions
            img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_ingredients_mock(image_array):
    """Generate mock predictions for demonstration."""
    # Generate random predictions that sum to 1
    np.random.seed(42)  # For consistent results
    predictions = np.random.random(len(ingredient_classes))
    predictions = predictions / np.sum(predictions)
    
    # Get top 5 predictions
    top_indices = np.argsort(predictions)[-5:][::-1]
    
    results = []
    for idx in top_indices:
        confidence = float(predictions[idx])
        if confidence > 0.05:  # Only show predictions above 5% confidence
            dish_name = ingredient_classes[idx]
            dish_ingredients = get_dish_ingredients(dish_name)
            
            print(f"🍽️  Mock prediction: {dish_name} ({confidence:.2%}) - {len(dish_ingredients)} ingredients")
            
            results.append({
                'ingredient': dish_name,
                'confidence': confidence,
                'ingredients': dish_ingredients
            })
    
    return results

def predict_ingredients(image_array):
    """Predict ingredients from preprocessed image."""
    global model
    
    if model is None:
        print("Model not loaded, using mock predictions")
        return predict_ingredients_mock(image_array)
    
    try:
        # Make prediction
        predictions = model.predict(image_array)
        
        # Get top predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        
        results = []
        for idx in top_indices:
            confidence = float(predictions[0][idx])
            if confidence > 0.05:  # Only show predictions above 5% confidence
                # Make sure index is within bounds
                if idx < len(ingredient_classes):
                    dish_name = ingredient_classes[idx]
                    
                    # Get ingredients for this dish
                    dish_ingredients = get_dish_ingredients(dish_name)
                    
                    print(f"🍽️  Model prediction: {dish_name} ({confidence:.2%}) - {len(dish_ingredients)} ingredients")
                    
                    result = {
                        'ingredient': dish_name,
                        'confidence': confidence,
                        'ingredients': dish_ingredients
                    }
                    results.append(result)
                else:
                    print(f"⚠️  Warning: Prediction index {idx} out of range for {len(ingredient_classes)} classes")
        
        return results
    except Exception as e:
        print(f"Error making prediction: {e}")
        return predict_ingredients_mock(image_array)

# Note: When serving a standalone static frontend (e.g., GitHub Pages),
# you may not need the '/' route at all. Keeping it for local dev.
@app.route('/')
def index():
    """Simple root message (frontend is served separately)."""
    return (
        "IngredientWale API is running. Use the standalone frontend to interact with /predict and /menu.",
        200,
        {"Content-Type": "text/plain; charset=utf-8"}
    )

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Handle image upload and prediction."""
    try:
        # Handle CORS preflight
        if request.method == 'OPTIONS':
            return (
                '',
                204,
                {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                }
            )

        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if file and allowed_file(file.filename):
            # Read image data
            image_data = file.read()
            
            # Open image with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            if processed_image is None:
                return jsonify({'error': 'Error preprocessing image'}), 500
            
            # Make prediction
            predictions = predict_ingredients(processed_image)
            
            # Return results
            resp = jsonify({'success': True, 'predictions': predictions})
            # Ensure CORS header present even without flask-cors
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/menu', methods=['GET'])
def get_menu():
    """Return list of all possible dish classes."""
    return jsonify({
        'ingredients': ingredient_classes
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tensorflow_available': TF_AVAILABLE
    })
def initialize_backend():
    print("Checking model file...")
    ensure_model_file()
    print("Loading model...")
    load_model_from_file()
    print("Loading food ingredients...")
    load_food_ingredients()

# Initialize at import time (works under Gunicorn/Render)
initialize_backend()

if __name__ == '__main__':
    # Local dev server
    print("Starting IngredientWale Flask App (dev mode)...")
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(debug=True, use_reloader=False, host=host, port=port)

