import sys
from word_piece_tokenizer import WordPieceTokenizer
import numpy as np
from datasets import load_dataset
import tensorflow as tf
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

REPRESENTATIVE_DATASET_SIZE = 5
model_path = "/workspace/model_profiling/onnx/mobilebert_modified"
tflite_model_path = "mobilebert_modified_int8.tflite"

SEQ_LEN = 384

torch_dataset = load_dataset("glue", "mrpc")
train_data = torch_dataset["train"]
tokenizer = WordPieceTokenizer()

def bert_input_processor(inputs):
    tok1 = tokenizer.tokenize(inputs['sentence1'])
    tok2 = tokenizer.tokenize(inputs['sentence2'])
    packed = np.array(tok1 + tok2[1:])
    ids = np.zeros((SEQ_LEN), dtype=np.int64)
    type_ids = np.zeros((SEQ_LEN), dtype=np.int64)
    mask = np.zeros((SEQ_LEN), dtype=np.int64)

    ids[:len(packed)] = packed
    type_ids[len(tok1):len(tok1)+len(tok2)-1] = 1
    mask[:len(tok1)+len(tok2)-1]=1
    ids = np.expand_dims(ids,0)
    type_ids = np.expand_dims(type_ids,0)
    mask = np.expand_dims(mask,0)
    return ids, type_ids, mask

train_subset = []
for i in range(len(train_data["sentence1"])):
    sentence1 = train_data["sentence1"][i]
    sentence2 = train_data["sentence2"][i]
    label = train_data["label"][i]
        
    word_ids, type_ids, mask = bert_input_processor({
        "sentence1":sentence1,
        "sentence2":sentence2})
    
    
    train_subset.append([mask, word_ids, type_ids])
    if i==REPRESENTATIVE_DATASET_SIZE:
        break

def representative_dataset():
  for data in train_subset:
    yield data

interpreter = tf.lite.Interpreter("/workspace/model_profiling/tensorflow/mobilebert_modified.tflite")
interpreter.allocate_tensors()
all_layers_details = interpreter.get_tensor_details() 
signature_list = interpreter.get_signature_list()

deny_nodes = []

allowed_ops = ["MatMul", "Gather", "GatherV2", "Concat", "Transpose", "Reshape", "Softmax", "Const_"]

for i, layer in enumerate(all_layers_details):
    opname = layer["name"].split("/")[-1]
    #print(layer["name"].split("/")[-1])
    #print(op.OpcodeIndex())
    allowed=False
    for op in allowed_ops:
        if op in opname:
            allowed=True
    if not allowed:
        #if i<1000:
        #    continue
        if len(layer["shape"])<=3 and "Add" in layer["name"]:
            continue
        if "Add" in layer["name"]:
            print(layer["name"], len(layer["shape"]), i)
        deny_nodes.append(layer["name"])
    elif "Softmax" in opname:
        deny_nodes.append(layer["name"])
#"""
#deny_nodes = []
print(len(all_layers_details))

ptq_converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
ptq_converter.optimizations = [tf.lite.Optimize.DEFAULT]
ptq_converter.representative_dataset = representative_dataset
ptq_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#tflite_model = ptq_converter.convert()

#with open(tflite_model_path, 'wb') as f:
#    f.write(tflite_model)

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

RESULTS_FILE = 'debug/mobilebert_modified_results.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)

layer_stats = pd.read_csv(RESULTS_FILE)
layer_stats['range'] = 255.0 * layer_stats['scale']
layer_stats['rmse/scale'] = layer_stats.apply(
    lambda row: np.sqrt(row['mean_squared_error']) / row['scale'], axis=1)
layer_stats = layer_stats.sort_values(by=["rmse/scale"], ascending=False)
layer_stats.to_csv(RESULTS_FILE[:-4] + "_formatted.csv")
#"""
"""
layer_stats = pd.read_csv(RESULTS_FILE)
deny_nodes = []
for tensor_name in pd.unique(layer_stats["tensor_name"]):
    nums_removed = ''.join([i for i in tensor_name if not i.isdigit()])
    if nums_removed=="model_/transformer/layer_/add_" or nums_removed=="model_/transformer/layer_/add" or nums_removed=="model_/transformer/mul;model_/transformer/ExpandDims;model_/transformer/mul/y":
        deny_nodes.append(tensor_name)
        print("IS MATCH", tensor_name)
    elif "layer_norm" in tensor_name and not ";" in tensor_name or "embedding" in tensor_name:
        deny_nodes.append(tensor_name)
        print("IS MATCH", tensor_name)
    else:
        print("no match", tensor_name)

debug_options = tf.lite.experimental.QuantizationDebugOptions(
    denylisted_nodes=deny_nodes)
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter,
    debug_dataset=representative_dataset,
    debug_options=debug_options)
debugger.run()
RESULTS_FILE = 'debugger_results/QAT_MIXED-PRECISION_' + OUT_NAME + str(UUID) + '.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)
quantized_tflite_model = debugger.get_nondebug_quantized_model()
"""