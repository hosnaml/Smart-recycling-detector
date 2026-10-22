# Model Evaluation Results

## Overview

The evaluation script was run on 35 labelled images from the test dataset to assess the performance of the trained waste classification model.

- **Model:** Fine-tuned MobileNetV2
- **Number of test images:** 35
- **Number of classes:** 6
- **Image size:** 224 × 224 pixels
- **Evaluation script:** `ml/evaluate_model.py`

## Overall Evaluation Metrics

| Metric | Result |
|---|---:|
| Accuracy | 77.14% |
| Macro precision | 78.33% |
| Macro recall | 77.92% |
| Macro F1-score | 73.52% |
| Weighted precision | 83.62% |
| Weighted recall | 77.14% |
| Weighted F1-score | 77.52% |

The model correctly classified 27 out of 35 test images.

## Per-Class Results

| Class | Correct Predictions | Total Images | Accuracy |
|---|---:|---:|---:|
| Cardboard | 2 | 2 | 100.00% |
| Glass | 4 | 5 | 80.00% |
| Metal | 8 | 8 | 100.00% |
| Organic | 4 | 6 | 66.67% |
| Paper | 2 | 6 | 33.33% |
| Plastic | 7 | 8 | 87.50% |

## Confusion Matrix

Rows represent actual classes, and columns represent predicted classes. Class order: cardboard, glass, metal, organic, paper, plastic.

| Actual / Predicted | Cardboard | Glass | Metal | Organic | Paper | Plastic |
|---|---:|---:|---:|---:|---:|---:|
| Cardboard | 2 | 0 | 0 | 0 | 0 | 0 |
| Glass | 0 | 4 | 0 | 0 | 0 | 1 |
| Metal | 0 | 0 | 8 | 0 | 0 | 0 |
| Organic | 0 | 0 | 0 | 4 | 0 | 2 |
| Paper | 4 | 0 | 0 | 0 | 2 | 0 |
| Plastic | 0 | 0 | 0 | 0 | 1 | 7 |

## Observations

- Metal achieved 100% accuracy on 8 test images.
- Cardboard achieved 100% accuracy on 2 test images; the small sample size limits this result.
- Paper had the lowest accuracy at 33.33%. Four paper images were classified as cardboard.
- Organic achieved 66.67% accuracy, with two images classified as plastic.
- The results indicate that paper and organic classification need further investigation.

These results are preliminary because the test dataset contains only 35 images.

## Generated Result Files

- `confusion_matrix.png` — visual representation of correct and incorrect predictions.
- `per_class_accuracy.png` — accuracy for each of the six classes.

Both files are stored in the `ml/results/` directory.

## Evaluation Scope

These metrics describe performance on the test dataset. They are separate from validation results on the original dataset. Validation evaluation remains pending because the expected `dataset_mwcd` directory was not found.
