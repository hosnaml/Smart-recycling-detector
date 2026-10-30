"""Raspberry Pi TFLite/LiteRT waste classifier with 80% confidence handling.

This replaces the Pi detector, NOT the laptop Keras detector.
"""
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

# Pi uses ai-edge-litert; retain TensorFlow Lite as an optional dev-machine fallback.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        Interpreter = None

ML_DIR = Path(__file__).parent / "ml"
MODEL_PATH = ML_DIR / "waste_classifier_mwcd_finetuned.tflite"
CONFIDENCE_THRESHOLD = 0.80

interpreter = None
input_details = None
output_details = None

try:
    if Interpreter is None:
        raise RuntimeError("Install ai-edge-litert on the Pi (or TensorFlow Lite on a laptop).")
    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Successfully loaded TFLite model")
except Exception as e:
    print(f"Error loading model: {e}")
    interpreter = None

# Keep the name expected by the existing Flask app's `/status` endpoint.
model = interpreter

CLASS_NAMES = ["cardboard", "glass", "metal", "organic", "paper", "plastic"]

CATEGORIES = {
    "cardboard": {"label": "Cardboard", "color": "info", "instructions": "Flatten and place in the cardboard/paper recycling bin."},
    "glass": {"label": "Glass", "color": "success", "instructions": "Recycle in the glass bin. Empty and rinse first."},
    "metal": {"label": "Metal", "color": "warning", "instructions": "Recycle in the metal bin. Empty and rinse."},
    "organic": {"label": "Organic", "color": "success", "instructions": "Place in the compost or organic waste bin."},
    "paper": {"label": "Paper", "color": "info", "instructions": "Recycle in the paper bin. Keep it dry."},
    "plastic": {"label": "Plastic", "color": "primary", "instructions": "Recycle in the plastic bin. Empty and rinse."},
}


def detect_recycling(image_data_b64):
    """Classify a browser-captured image and flag predictions below 80%."""
    if interpreter is None:
        return {
            "label": "Error", "color": "danger",
            "instructions": "The classification model is unavailable. Please check the Raspberry Pi application.",
            "confidence": None, "uncertain": True,
        }

    try:
        if "," in image_data_b64:
            image_data_b64 = image_data_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        # Same RGB float32 preprocessing as original Raspberry Pi detector.
        img_array = np.expand_dims(np.array(img), axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])  # 0..1, not percent.
        confidence_percent = round(confidence * 100, 2)
        print(f"ML Detection: {predicted_class} (Confidence: {confidence_percent:.2f}%)")

        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "label": "Uncertain",
                "color": "warning",
                "instructions": "The model is not confident enough to classify this item. Please try another photo with the item clearly visible.",
                "confidence": confidence_percent,
                "uncertain": True,
            }

        result = CATEGORIES[predicted_class].copy()
        result["confidence"] = confidence_percent
        result["uncertain"] = False
        return result

    except Exception as e:
        print(f"Error during ML inference: {e}")
        return {
            "label": "Error", "color": "danger",
            "instructions": "Classification failed. Please try again or check the Raspberry Pi logs.",
            "confidence": None, "uncertain": True,
        }
