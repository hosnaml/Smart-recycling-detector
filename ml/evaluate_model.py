import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path


# Folder where this Python file is located
ML_DIR = Path(__file__).parent

# Paths
DATASET_PATH = ML_DIR / "dataset_mwcd"
MODEL_PATH = ML_DIR / "waste_classifier_mwcd.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123


# Load validation data
validation_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = validation_dataset.class_names

print("Classes:", class_names)


# Load trained model
model = keras.models.load_model(
    MODEL_PATH
)


# Lists to store correct labels and model predictions
actual_labels = []
predicted_labels = []


# Go through validation images
for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predictions = np.argmax(
        predictions,
        axis=1
    )

    actual_labels.extend(
        labels.numpy()
    )

    predicted_labels.extend(
        predictions
    )


# Create confusion matrix
matrix = tf.math.confusion_matrix(
    actual_labels,
    predicted_labels,
    num_classes=len(class_names)
).numpy()


print("\nConfusion Matrix:")
print(matrix)


# Calculate accuracy for each class
print("\nPer-Class Accuracy:")

accuracies = []

for i in range(len(class_names)):

    correct = matrix[i][i]
    total = matrix[i].sum()

    accuracy = (
        correct / total
    ) * 100

    accuracies.append(accuracy)

    print(
        class_names[i],
        f"{accuracy:.2f}%"
    )


# Create folder for result images
RESULTS_PATH = ML_DIR / "results"
RESULTS_PATH.mkdir(exist_ok=True)


# Create and save per-class accuracy chart
plt.figure(figsize=(8, 5))

plt.bar(
    class_names,
    accuracies
)

plt.xlabel("Class")
plt.ylabel("Accuracy (%)")
plt.title("Per-Class Accuracy")

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    RESULTS_PATH / "per_class_accuracy.png"
)

plt.close()


# Create and save confusion matrix image
plt.figure(figsize=(7, 6))

plt.imshow(
    matrix,
    cmap="Blues"
)

plt.colorbar()

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)


# Change number color depending on background
threshold = matrix.max() / 2

for i in range(len(class_names)):

    for j in range(len(class_names)):

        if matrix[i][j] > threshold:
            text_color = "white"
        else:
            text_color = "black"

        plt.text(
            j,
            i,
            matrix[i][j],
            ha="center",
            va="center",
            color=text_color
        )


plt.tight_layout()

plt.savefig(
    RESULTS_PATH / "confusion_matrix.png"
)

plt.close()


print("\nResult images saved in:")
print(RESULTS_PATH)