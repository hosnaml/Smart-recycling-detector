# ML Training Pipeline

This folder contains the machine learning training pipeline for the Smart Recycling Detector.

## Current Classes

The current model classifies waste into six categories:

- Cardboard
- Glass
- Metal
- Organic
- Paper
- Plastic

## Dataset

The current training pipeline uses selected classes from the Merged Waste Classification Dataset (MWCD).

The original MWCD dataset contains nine classes. For the current project, six classes were selected:

- Cardboard
- Glass
- Metal
- Organic
- Paper
- Plastic

Battery, textile, and trash were excluded from the current model. Battery requires more specialized disposal guidance, textile is outside the current project scope, and trash is a broad category that does not represent a specific recyclable material.

The selected dataset contains 20,695 images.

The dataset is organized locally as:

```text
dataset_mwcd/
├── cardboard/
├── glass/
├── metal/
├── organic/
├── paper/
└── plastic/
```

The dataset is split into:

- 80% training data
- 20% validation data

A fixed random seed is used so that the same train/validation split can be reproduced.

The dataset itself is not included in the Git repository.

## Model

The model uses MobileNetV2 with transfer learning.

MobileNetV2 is pretrained on ImageNet. The pretrained MobileNetV2 layers are frozen and used as a feature extractor. The original MobileNetV2 classification layer is removed and a new classification layer is used for the selected waste classes.

The number of output classes is automatically determined from the dataset folders.

## Preprocessing

Before being passed to the model:

- Images are resized to 224 × 224 pixels.
- Pixel values are rescaled to the range expected by MobileNetV2.
- Images are processed in batches of 32.

## Training

The current training configuration is:

- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Batch size: 32
- Epochs: 5
- Validation split: 20%

Run the training pipeline with:

```bash
python train_model.py
```

The pipeline performs the following steps:

```text
MWCD Dataset
     ↓
Train / Validation Split
     ↓
Image Preprocessing
     ↓
MobileNetV2 Feature Extraction
     ↓
Waste Classification
     ↓
Validation
     ↓
Saved Model
```

After training, the model is saved as:

```text
waste_classifier_mwcd.keras
```

The trained model file is not included in the Git repository.

## Testing

`test_model.py` loads the trained MWCD model and can be used to make predictions on test images.

For each image, the script displays:

- Actual class
- Predicted class
- Confidence
- Probability for each class

The test images are stored locally in the `test_dataset` folder and are not included in the Git repository.

## Evaluation

Model evaluation is handled separately using `evaluate_model.py`.

The evaluation includes:

- Overall model performance
- Per-class accuracy
- Confusion matrix

## Previous Prototype

An earlier version of the training pipeline used three classes from the TrashNet dataset:

- Metal
- Paper
- Plastic

This smaller dataset was used to verify that the initial MobileNetV2 training and prediction pipeline worked before moving to the larger MWCD dataset.

## Baseline Evaluation

Baseline model performance, per-class accuracy, and the confusion matrix are documented in:

[BASELINE_RESULTS.md](BASELINE_RESULTS.md)


## Fine-Tuned MobileNetV2 (Issue #13)

The baseline MobileNetV2 model was improved using a two-stage training process:

1. **Initial training:** Train the classification head for 10 epochs while keeping the pretrained MobileNetV2 layers frozen.
2. **Fine-tuning:** Unfreeze the last 30 MobileNetV2 layers and continue training for 5 epochs using a learning rate of 0.00001.

### Results

| Model | Validation Accuracy |
|---|---|
| Original baseline (5 epochs) | 88.98% |
| Extended training (10 epochs) | 89.80% |
| Fine-tuned MobileNetV2 | **93.16%** |

Fine-tuning improved validation accuracy by 4.18 percentage points compared with the original baseline.

### Reproduce Training

Run:

```bash
python ml/train_final_mobilenetv2.py
```

The script trains the classification head, saves and reloads the intermediate model, and performs fine-tuning.

The final model is saved as `ml/waste_classifier_mobilenetv2_final.keras`.

### Limitations

Validation results are based on MWCD. Independent real-world testing is still needed to assess performance under different lighting, backgrounds, and camera conditions.

Trained model files are not included in Git.
