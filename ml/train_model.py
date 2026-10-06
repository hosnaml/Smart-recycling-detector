import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Path to the folder containing our image classes
DATASET_PATH = "dataset"

# MobileNetV2 expected size
IMAGE_SIZE = (224, 224)

# Number of images processed at one time during training
BATCH_SIZE = 32


# Load the training dataset
train_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,   #splits dataset into 20% validation rest for testing
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Load the validation dataset
validation_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Show the class names TensorFlow found
print("Classes:", train_dataset.class_names)


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

    # Final classifier for our 3 classes:
    # metal, paper, plastic
    layers.Dense(3, activation="softmax")
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
model.save("waste_classifier.keras")