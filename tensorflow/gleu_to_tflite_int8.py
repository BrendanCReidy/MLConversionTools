import sys
if len(sys.argv) < 3:
    print("usage: imagenet_to_tflite_int8.py <tf_model_path> <output_tflite_model_path>")
    exit()

from word_piece_tokenizer import WordPieceTokenizer
import numpy as np
from datasets import load_dataset
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

REPRESENTATIVE_DATASET_SIZE = 1
model_path = sys.argv[1]
tflite_model_path = sys.argv[2]

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

ptq_converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
ptq_converter.optimizations = [tf.lite.Optimize.DEFAULT]
ptq_converter.representative_dataset = representative_dataset
ptq_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = ptq_converter.convert()

with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)