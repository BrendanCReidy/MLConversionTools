import sys
if len(sys.argv) < 3:
    print("usage: onnx_to_tensorflow.py <onnx_model_path> <out_model_path>")
    exit()

IN_PATH = sys.argv[1]
OUT_PATH = sys.argv[2]

import onnx
from onnx_tf.backend import prepare
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

model = onnx.load(IN_PATH)
prepare(model).export_graph(OUT_PATH)