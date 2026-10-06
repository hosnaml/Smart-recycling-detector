# ML Training Pipeline

This folder contains the initial machine learning training pipeline for the Smart Recycling Detector.

## Current Classes

The current model classifies waste into three categories:

- Metal
- Paper
- Plastic

## Dataset

The initial pipeline uses images from the TrashNet dataset.

The dataset is organized as:

```text
dataset/
├── metal/
├── paper/
└── plastic/
```

The dataset is split into:

- 80% training data
- 20% validation data

The dataset itself is not included in the Git repository.

## Model

The initial model uses MobileNetV2 with transfer learning.

MobileNetV2 is pretrained on ImageNet. The pretrained MobileNetV2 layers are frozen and used as a feature extractor. A new classification layer is added and trained to classify the three waste categories.

## Preprocessing

Before being passed to the model:

- Images are resized to 224 × 224 pixels.
- Pixel values are rescaled to the range expected by MobileNetV2.

## Training

The current training configuration is:

- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Batch size: 32
- Epochs: 5

Run the training pipeline with:

```bash
python train_model.py
```

The pipeline performs the following steps:

```text
Dataset
↓
Preprocessing
↓
Training
↓
Validation
↓
Saved Model
```

After training, the model is saved as:

```text
waste_classifier.keras
```

The trained model file is not included in the repository.

## Testing

`test_model.py` loads the saved model and can be used to make predictions on new images.

For each image, the script displays:

- Predicted class
- Confidence
- Probability for each class

## Initial Result

The first MobileNetV2 training run produced:

- Training accuracy: 96.69%
- Validation accuracy: 93.27%
- Validation loss: 0.1996

This model serves as the initial baseline. Further dataset evaluation, model comparison, and model tuning will be handled in later project tasks.