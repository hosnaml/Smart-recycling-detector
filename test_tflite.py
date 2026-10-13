import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="ml/waste_classifier_mwcd_finetuned.tflite"
)

interpreter.allocate_tensors()

print("TFLite model loaded successfully!")

print(interpreter.get_input_details())
print(interpreter.get_output_details())
