import sys
if len(sys.argv) < 3:
    print("usage: tf_to_tflite.py <tf_model_path> <output_tflite_model_path>")
    exit()

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
model_path = sys.argv[1]
tflite_model_path = sys.argv[2]

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
