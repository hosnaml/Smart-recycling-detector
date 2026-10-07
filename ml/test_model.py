import numpy as np
from tensorflow import keras
from pathlib import Path


# Folder where this Python file is located
ML_DIR = Path(__file__).parent

# Load the trained MWCD model
model = keras.models.load_model(
    ML_DIR / "waste_classifier_mwcd.keras"
)

# These must match the class order used during training
class_names = [
    "cardboard",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic"
]

# Folder containing test images
TEST_DATASET_PATH = ML_DIR / "test_dataset"


# Go through each folder that currently exists in test_dataset
for class_folder in TEST_DATASET_PATH.iterdir():

    # Skip anything that is not a folder
    if not class_folder.is_dir():
        continue

    actual_class = class_folder.name

    # Skip folders that are not one of our model classes
    if actual_class not in class_names:
        continue

    # Go through every file in the class folder
    for image_path in class_folder.iterdir():

        # Only process image files
        if image_path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:
            continue

        # Load and resize the image
        img = keras.utils.load_img(
            image_path,
            target_size=(224, 224)
        )

        # Convert image to numbers
        img_array = keras.utils.img_to_array(img)

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Make prediction
        predictions = model.predict(
            img_array,
            verbose=0
        )

        # Find the class with the highest probability
        predicted_index = np.argmax(
            predictions[0]
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = (
            predictions[0][predicted_index]
            * 100
        )

        print("\n----------------------------")
        print("Image:", image_path.name)
        print("Actual class:", actual_class)

        print("All probabilities:")

        for i, class_name in enumerate(class_names):

            probability = (
                predictions[0][i]
                * 100
            )

            print(
                f"{class_name}: "
                f"{probability:.2f}%"
            )

        print(
            "Prediction:",
            predicted_class
        )

        print(
            f"Confidence: "
            f"{confidence:.2f}%"
        )