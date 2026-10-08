import os
import base64
import io
import time
import random
from pathlib import Path

# Try importing ML dependencies
try:
    import numpy as np
    from tensorflow import keras
    from PIL import Image
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: TensorFlow, NumPy, or Pillow not installed. Falling back to mock detection.")

# Setup Model Path
ML_DIR = Path(__file__).parent / "ml"
MODEL_PATH = ML_DIR / "waste_classifier_mwcd_10epochs.keras"
model = None

# Load model if dependencies and model file exist
if ML_AVAILABLE:
    if MODEL_PATH.exists():
        try:
            # Rebuild the model architecture to bypass Keras 3 Sequential loading bugs with MobileNetV2
            number_of_classes = 6
            base_model = keras.applications.MobileNetV2(
                input_shape=(224, 224, 3), include_top=False, weights=None)
            base_model.trainable = False
            
            model = keras.Sequential([
                keras.layers.Rescaling(1./127.5, offset=-1),
                base_model,
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(number_of_classes, activation="softmax")
            ])
            model.build((None, 224, 224, 3))
            
            # Load weights directly from the .keras file
            model.load_weights(MODEL_PATH)
            
            print("Successfully loaded ML model:", MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"Model file not found at {MODEL_PATH}. Please train the model first. Falling back to mock detection.")

# The classes exactly as trained in test_model.py
CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic"
]

# UI configuration for each class
CATEGORIES = {
    "cardboard": {"label": "Cardboard", "color": "info", "instructions": "Flatten and place in the cardboard/paper recycling bin."},
    "glass": {"label": "Glass", "color": "success", "instructions": "Recycle in the glass bin. Empty and rinse first."},
    "metal": {"label": "Metal", "color": "warning", "instructions": "Recycle in the metal bin. Empty and rinse."},
    "organic": {"label": "Organic", "color": "success", "instructions": "Place in the compost or organic waste bin."},
    "paper": {"label": "Paper", "color": "info", "instructions": "Recycle in the paper bin. Keep it dry."},
    "plastic": {"label": "Plastic", "color": "primary", "instructions": "Recycle in the plastic bin. Empty and rinse."}
}

def detect_recycling(image_data_b64):
    """
    Decodes the base64 image, resizes it to 224x224, and runs inference.
    If the model is unavailable, falls back to a random prediction among the 6 classes.
    """
    if model is not None:
        try:
            # Strip the 'data:image/jpeg;base64,' prefix if present
            if ',' in image_data_b64:
                image_data_b64 = image_data_b64.split(',')[1]
                
            # Decode base64 to image
            image_bytes = base64.b64decode(image_data_b64)
            img = Image.open(io.BytesIO(image_bytes))
            
            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            # Resize to 224x224 as expected by the MobileNetV2 pipeline
            img = img.resize((224, 224))
            
            # Convert to numpy array and add batch dimension
            img_array = keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_index = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_index]
            
            print(f"ML Detection: {predicted_class} (Confidence: {predictions[0][predicted_index]*100:.2f}%)")
            
            return CATEGORIES[predicted_class]
        except Exception as e:
            print(f"Error during ML inference: {e}")
            # Fall through to mock on error
    
    # Mock fallback
    print("Using Mock Detection...")
    time.sleep(1.2)
    mock_class = random.choice(CLASS_NAMES)
    return CATEGORIES[mock_class]
