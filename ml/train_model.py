import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# Folder where this Python file is located
ML_DIR = Path(__file__).parent

# Path to the MWCD dataset
DATASET_PATH = ML_DIR / "dataset_mwcd"

# MobileNetV2 expected size
IMAGE_SIZE = (224, 224)

# Number of images processed at one time during training
BATCH_SIZE = 32

# Random seed so we get the same train/validation split each time
SEED = 123


# Load the training dataset
train_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,   #splits dataset into 20% validation rest for training
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Load the validation dataset
validation_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# TensorFlow automatically reads the class names from the folder names
class_names = train_dataset.class_names
print("Classes:", class_names)

# Automatically determine how many classes we have
number_of_classes = len(class_names)
print("Number of classes:", number_of_classes)


# Load a pretrained MobileNetV2 model
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,  #removing the classifying layer to add our own
    weights="imagenet"
)

# Freeze MobileNetV2 so we do not change its pretrained knowledge yet
base_model.trainable = False


# Build our waste-classification model
model = keras.Sequential([
    # Convert pixel values from 0-255 into the range expected by MobileNetV2
    keras.layers.Rescaling(1./127.5, offset=-1),

    # Pretrained image-feature extractor
    base_model,

    # Reduce the feature maps into a smaller representation
    layers.GlobalAveragePooling2D(),
    layers.Dense(number_of_classes, activation="softmax")
])


# Tell TensorFlow how the model should learn
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Print the structure of the model
model.summary()


# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5
)


# Save the trained model so we can use it later
model.save(ML_DIR / "waste_classifier_mwcd.keras")