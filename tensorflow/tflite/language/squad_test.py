from word_piece_tokenizer import WordPieceTokenizer
import numpy as np
from datasets import load_dataset
import tensorflow as tf
import os

model_path = "/workspace/model_profiling/tensorflow/tflite/language/mobilebert_squad_int8.tflite"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

sentence1 = """
Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL)
for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National
Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.
The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara,
California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various
gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with
Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently
feature the Arabic numerals 50.
"""

sentence2 = "Which NFL team represented the AFC at Super Bowl 50?"

SEQ_LEN = 384

torch_dataset = load_dataset("glue", "mrpc")
val_data = torch_dataset["validation"]
tokenizer = WordPieceTokenizer()

def bert_input_processor(inputs):
    tok1 = tokenizer.tokenize(inputs['sentence1'])
    tok2 = tokenizer.tokenize(inputs['sentence2'])
    packed = np.array(tok1 + tok2[1:])
    ids = np.zeros((SEQ_LEN), dtype=np.int32)
    type_ids = np.zeros((SEQ_LEN), dtype=np.int32)
    mask = np.zeros((SEQ_LEN), dtype=np.int32)

    ids[:len(packed)] = packed
    type_ids[len(tok1):len(tok1)+len(tok2)-1] = 1
    mask[:len(tok1)+len(tok2)-1]=1
    ids = np.expand_dims(ids,0)
    type_ids = np.expand_dims(type_ids,0)
    mask = np.expand_dims(mask,0)
    return ids, type_ids, mask

def evaluate_model(interpreter, sentence1, sentence2):
    end_index = interpreter.get_output_details()[0]["index"]
    start_index = interpreter.get_output_details()[1]["index"]

    input_mask_index = interpreter.get_input_details()[2]["index"]
    word_ids_index = interpreter.get_input_details()[0]["index"]
    input_type_ids_index = interpreter.get_input_details()[1]["index"]

    acc = 0
    samples = 0
    word_ids, type_ids, mask = bert_input_processor({
        "sentence1":sentence1,
        "sentence2":sentence2})

    interpreter.set_tensor(word_ids_index, word_ids)
    interpreter.set_tensor(input_mask_index, mask)
    interpreter.set_tensor(input_type_ids_index, type_ids)

    # Run inference.
    interpreter.invoke()

    start = np.argmax(interpreter.get_tensor(start_index))
    end = np.argmax(interpreter.get_tensor(end_index))
    tokens = tokenizer.convert_ids_to_tokens(word_ids[0][start:end+1])
    out = tokenizer.convert_tokens_to_string(tokens)
    #print(start, end)
    print("Answer:", out)


interpreter = tf.lite.Interpreter(
 model_path, experimental_preserve_all_tensors=False
)
interpreter.allocate_tensors()
evaluate_model(interpreter, sentence1, sentence2)
inp = ""
print("Sentence:")
print(sentence1)
while not inp=="q":
    inp = input("Question:")
    print("Sentence:")
    print(sentence1)
    print(inp)
    evaluate_model(interpreter, sentence1, inp)
