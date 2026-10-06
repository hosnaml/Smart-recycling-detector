import os
import numpy as np
from tensorflow import keras


# Load the trained model
model = keras.models.load_model("waste_classifier.keras")

# These must match the class order used during training
class_names = ["metal", "paper", "plastic"]

# Folder containing all test images
TEST_DATASET_PATH = "test_dataset"


# Go through each class folder
for actual_class in class_names:

    class_folder = os.path.join(
        TEST_DATASET_PATH,
        actual_class
    )

    # Go through every file in that folder
    for filename in os.listdir(class_folder):

        # Only process image files
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            class_folder,
            filename
        )

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

        # Find highest probability
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
        print("Image:", filename)
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