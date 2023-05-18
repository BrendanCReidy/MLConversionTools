import sys
if len(sys.argv) < 2:
    print("usage:", sys.argv[0], "<model_path>")
    exit()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

model_path = sys.argv[1]
model_name = model_path[:-7]
if "/" in model_path:
    model_name = model_path.split("/")[-1][:-7]

num_threads=1
if len(sys.argv) > 2:
    num_threads = int(sys.argv[2])

interpreter = tf.lite.Interpreter(
 model_path, experimental_preserve_all_tensors=True, num_threads=num_threads
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(interpreter, image):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, image)

    # Run inference.
    interpreter.invoke()

    out = interpreter.get_tensor(output_index)
    return out


def clean_accuracy(interpreter, data_loader):
    clean_dataset = []; correct = 0; total = 0; i = 0
    acc = 0
    for images, labels in data_loader:
        images = images.numpy()
        labels = labels
        outputs =  predict(interpreter, images)
        outputs = torch.from_numpy(outputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        i = i + 1
        if i % 500 == 0:
            acc = (i, 100 * correct / total)
            print('INFO: Accuracy of the network on the test images: %d, %.2f %%' % acc)
    return acc[1]
        
mean = [0.485, 0.456, 0.406] # https://pytorch.org/vision/stable/models.html#wide-resnet
std  = [0.229, 0.224, 0.225]
normalization = [transforms.Normalize(mean=mean, std=std)]
transform_list = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()] + normalization)
IMAGENET_PATH  = "/data/ImageNet1k" # imagenet 2012
TEST_PATH      = os.path.join(IMAGENET_PATH, 'val') # contains subfolders 50k samples

test_loader   = torch.utils.data.DataLoader(datasets.ImageFolder(TEST_PATH, transform_list), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

acc = clean_accuracy(interpreter, test_loader)

model_size = os.path.getsize(model_path)
model_size = round(model_size/(pow(1024,2)), 2)

df = pd.DataFrame([{"MODEL_NAME":model_name, "ACCURACY":acc, "SIZE (MB)":model_size}])
df.to_csv("results/" + model_name + ".csv")