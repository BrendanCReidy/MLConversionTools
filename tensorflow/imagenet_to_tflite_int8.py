import sys
if len(sys.argv) < 3:
    print("usage: imagenet_to_tflite_int8.py <tf_model_path> <output_tflite_model_path>")
    exit()

import tensorflow as tf
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

REPRESENTATIVE_DATASET_SIZE = 10
IMAGENET_PATH = "/data/ImageNet1k/ILSVRC/Data/CLS-LOC"
model_path = sys.argv[1]
tflite_model_path = sys.argv[2]

model_name = model_path.split("/")[-1]

mean = [0.485, 0.456, 0.406] # https://pytorch.org/vision/stable/models.html#wide-resnet
std  = [0.229, 0.224, 0.225]
normalization = [transforms.Normalize(mean=mean, std=std)]
transform_list = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()] + normalization)
TRAIN_PATH = os.path.join(IMAGENET_PATH, 'train') # contains subfolders 50k samples
train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(TRAIN_PATH, transform_list), batch_size=1, shuffle=True, num_workers=10, pin_memory=True)

train_subset = []
i = 0
for data, label in train_loader:
    train_subset.append(data.numpy())
    if i==REPRESENTATIVE_DATASET_SIZE:
        break
    i+=1

def representative_dataset():
  for data in train_subset:
    yield [data]

ptq_converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
ptq_converter.optimizations = [tf.lite.Optimize.DEFAULT]
ptq_converter.representative_dataset = representative_dataset
ptq_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#tflite_model = ptq_converter.convert()

interpreter = tf.lite.Interpreter("tflite/imagenet/resnet18.tflite")
interpreter.allocate_tensors()
all_layers_details = interpreter.get_tensor_details() 
signature_list = interpreter.get_signature_list()

deny_nodes = []
"""
deny_ops = ["Sqrt", "ReduceMean"]

for i, layer in enumerate(all_layers_details):
    opname = layer["name"].split("/")[-1]
    allowed=True
    for op in deny_ops:
        if op in opname:
            allowed=False
    if not allowed:
        deny_nodes.append(layer["name"])
"""
debug_options = tf.lite.experimental.QuantizationDebugOptions(
    denylisted_nodes=deny_nodes)
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=ptq_converter,
    debug_dataset=representative_dataset,
    debug_options=debug_options)
debugger.run()
quantized_tflite_model = debugger.get_nondebug_quantized_model()
with open(tflite_model_path, 'wb') as f:
    f.write(quantized_tflite_model)

RESULTS_FILE = 'debug/' + model_name + '.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)

layer_stats = pd.read_csv(RESULTS_FILE)
layer_stats['range'] = 255.0 * layer_stats['scale']
layer_stats['rmse/scale'] = layer_stats.apply(
    lambda row: np.sqrt(row['mean_squared_error']) / row['scale'], axis=1)
layer_stats = layer_stats.sort_values(by=["rmse/scale"], ascending=False)
layer_stats.to_csv('debug/' + model_name + "_formatted.csv")