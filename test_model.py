from tensorflow import keras

MODEL_PATH = "ml/waste_classifier_mwcd_finetuned.keras"

number_of_classes = 6

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

base_model.trainable = False

model = keras.Sequential([
    keras.layers.Rescaling(1./127.5, offset=-1),
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(number_of_classes, activation="softmax")
])

model.build((None, 224, 224, 3))

model.load_weights(MODEL_PATH)

print("MODEL LOADED SUCCESSFULLY")
