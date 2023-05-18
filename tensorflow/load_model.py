import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

model = tf.saved_model.load('/workspace/model_profiling/onnx/mobilenetv2')
print('loaded model inputs = ', model.signatures['serving_default'].inputs)
print('loaded model outputs = ', model.signatures['serving_default'].outputs)