from word_piece_tokenizer import WordPieceTokenizer
import numpy as np
from datasets import load_dataset
import tensorflow as tf
import os

model_path = "/workspace/model_profiling/tensorflow/mobilebert_modified.tflite"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

SEQ_LEN = 384

torch_dataset = load_dataset("glue", "mrpc")
val_data = torch_dataset["validation"]
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

def evaluate_model(interpreter):
    output_index = interpreter.get_output_details()[0]["index"]

    print(interpreter.get_input_details()[1])

    input_mask_index = interpreter.get_input_details()[0]["index"]
    word_ids_index = interpreter.get_input_details()[1]["index"]
    input_type_ids_index = interpreter.get_input_details()[2]["index"]

    acc = 0
    samples = 0
    for i in range(len(val_data["sentence1"])):
        sentence1 = val_data["sentence1"][i]
        sentence2 = val_data["sentence2"][i]
        label = val_data["label"][i]
            
        word_ids, type_ids, mask = bert_input_processor({
            "sentence1":sentence1,
            "sentence2":sentence2})

        interpreter.set_tensor(word_ids_index, word_ids)
        interpreter.set_tensor(input_mask_index, mask)
        interpreter.set_tensor(input_type_ids_index, type_ids)

        # Run inference.
        interpreter.invoke()

        out = interpreter.get_tensor(output_index)
        y_pred = int(np.argmax(out))
        if y_pred == label:
            acc+=1
        samples+=1
        if i%10==0:
            print(i, "/", str(len(val_data["sentence1"])) + ":\tAccuracy:", acc / float(samples))
    print("Accuracy:", acc / float(samples))



interpreter = tf.lite.Interpreter(
 model_path, experimental_preserve_all_tensors=False
)
interpreter.allocate_tensors()
evaluate_model(interpreter)