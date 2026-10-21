
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------

ML_DIR = Path(__file__).parent
DATASET_PATH = ML_DIR / "dataset_mwcd"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

INITIAL_EPOCHS = 10
FINETUNING_EPOCHS = 5
LAYERS_TO_UNFREEZE = 30

# Use new filenames to protect our existing trained models
INTERMEDIATE_MODEL_PATH = ML_DIR / "waste_classifier_mobilenetv2_stage1.keras"
FINAL_MODEL_PATH = ML_DIR / "waste_classifier_mobilenetv2_final.keras"


# ---------------------------------------
# 1. LOAD DATASET
# ---------------------------------------

train_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
number_of_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", number_of_classes)


# ---------------------------------------
# 2. LOAD PRETRAINED MOBILENETV2
# ---------------------------------------

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ---------------------------------------
# 3. BUILD CLASSIFICATION MODEL
# ---------------------------------------

model = keras.Sequential([
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(number_of_classes, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ---------------------------------------
# 4. INITIAL TRAINING
# ---------------------------------------

print("\nStage 1: Training classification head...")

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS
)

# Save the model after initial training
model.save(INTERMEDIATE_MODEL_PATH)

print(f"Intermediate model saved: {INTERMEDIATE_MODEL_PATH}")


# ---------------------------------------
# 5. RELOAD MODEL FOR FINE-TUNING
# ---------------------------------------

model = keras.models.load_model(INTERMEDIATE_MODEL_PATH)

# In our Sequential architecture, MobileNetV2 is layer 1
base_model = model.layers[1]

base_model.trainable = True

# Freeze everything except the last 30 layers
for layer in base_model.layers[:-LAYERS_TO_UNFREEZE]:
    layer.trainable = False


# ---------------------------------------
# 6. FINE-TUNING
# ---------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nStage 2: Fine-tuning MobileNetV2...")

model.summary()

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=FINETUNING_EPOCHS
)


# ---------------------------------------
# 7. FINAL EVALUATION
# ---------------------------------------

print("\nEvaluating fine-tuned model...")

loss, accuracy = model.evaluate(validation_dataset)

print(f"Validation accuracy: {accuracy:.2%}")
print(f"Validation loss: {loss:.4f}")


# ---------------------------------------
# 8. SAVE FINAL MODEL
# ---------------------------------------

model.save(FINAL_MODEL_PATH)

print(f"\nFinal model saved: {FINAL_MODEL_PATH}")
