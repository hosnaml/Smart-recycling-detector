
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# Paths
ML_DIR = Path(__file__).parent
TEST_DATASET_PATH = ML_DIR / "test_dataset"
MODEL_PATH = ML_DIR / "waste_classifier_mwcd_finetuned.keras"
RESULTS_PATH = ML_DIR / "results"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Class order must match the model's training classes
class_names = [
    "cardboard",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic",
]

# Load the phone-photo dataset.
# Folder names provide the actual labels for each image.
test_dataset = keras.utils.image_dataset_from_directory(
    TEST_DATASET_PATH,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Classes:", class_names)

# Collect actual labels and predictions
actual_labels = []
predicted_labels = []
confidence_scores = []

for images, labels in test_dataset:
    probabilities = model.predict(images, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    confidence_scores.extend(confidences.tolist())
    actual_labels.extend(labels.numpy().tolist())
    predicted_labels.extend(predictions.tolist())

actual_labels = np.array(actual_labels)
predicted_labels = np.array(predicted_labels)
confidence_scores = np.array(confidence_scores)

# Calculate overall metrics
accuracy = accuracy_score(actual_labels, predicted_labels)

print("\nEvaluation Results")
print("==================")
print(f"Number of images: {len(actual_labels)}")
print(f"Overall accuracy: {accuracy * 100:.2f}%")

print("\nPer-Class Precision, Recall, and F1-Score")
print("=========================================")
print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
)

# Create results folder
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Confusion matrix
matrix = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
)

print("\nConfusion Matrix")
print(matrix)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names,
)
display.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Phone Photo Test Set")
plt.tight_layout()
plt.savefig(RESULTS_PATH / "confusion_matrix.png")
plt.close()

# Per-class accuracy: correct predictions / actual images in each class
class_accuracies = []

for i, class_name in enumerate(class_names):
    total = matrix[i].sum()
    correct = matrix[i, i]
    class_accuracy = correct / total * 100 if total else 0.0
    class_accuracies.append(class_accuracy)
    print(f"{class_name}: {class_accuracy:.2f}% ({correct}/{total})")

plt.figure(figsize=(8, 5))
plt.bar(class_names, class_accuracies)
plt.xlabel("Class")
plt.ylabel("Accuracy (%)")
plt.title("Per-Class Accuracy - Phone Photo Test Set")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(RESULTS_PATH / "per_class_accuracy.png")
plt.close()

print(f"\nResult images saved in: {RESULTS_PATH}")

# Confidence Threshold Investigation
print("\nConfidence Threshold Analysis")
print("=============================")

thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

image_results_path = RESULTS_PATH / "confidence_predictions.csv"
threshold_results_path = RESULTS_PATH / "confidence_threshold_results.csv"

# Save each image's prediction and confidence
with open(image_results_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "actual_class",
        "predicted_class",
        "confidence_percent",
        "correct",
    ])

    for actual, predicted, confidence in zip(
        actual_labels, predicted_labels, confidence_scores
    ):
        writer.writerow([
            class_names[int(actual)],
            class_names[int(predicted)],
            round(float(confidence) * 100, 2),
            int(actual == predicted),
        ])

# Compare the results at each confidence threshold
with open(threshold_results_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "threshold",
        "total_images",
        "accepted_images",
        "coverage_percent",
        "accepted_accuracy_percent",
        "incorrect_accepted",
        "uncertain_images",
    ])

    for threshold in thresholds:
        accepted = [
            i for i, confidence in enumerate(confidence_scores)
            if confidence >= threshold
        ]

        accepted_count = len(accepted)
        incorrect_count = sum(
            actual_labels[i] != predicted_labels[i] for i in accepted
        )

        coverage = accepted_count / len(confidence_scores) * 100

        if accepted_count:
            accepted_accuracy = (
                accepted_count - incorrect_count
            ) / accepted_count * 100
        else:
            accepted_accuracy = 0.0

        uncertain_count = len(confidence_scores) - accepted_count

        writer.writerow([
            threshold,
            len(confidence_scores),
            accepted_count,
            round(coverage, 2),
            round(accepted_accuracy, 2),
            incorrect_count,
            uncertain_count,
        ])

        print(
            f"Threshold {threshold:.0%}: "
            f"{accepted_count}/{len(confidence_scores)} accepted, "
            f"coverage={coverage:.2f}%, "
            f"accepted accuracy={accepted_accuracy:.2f}%, "
            f"incorrect accepted={incorrect_count}"
        )

print(f"\nPer-image results saved to: {image_results_path}")
print(f"Threshold summary saved to: {threshold_results_path}")