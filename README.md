# ConverstionTools

Typical pipeline for PyTorch to INT8 tflite for ImageNet

1. Save PyTorch model (this varies by model, see pytorch/models/imagenet/mobilenetv2_onnx.py for example)
2. Make shapes static if not already (e.g:
```python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param batch --dim_value 1 model.onnx model.fixed.onnx```
3. Fold constants: 
```polygraphy surgeon sanitize /workspace/model_profiling/pytorch/models/language/mobilebert_mrpc.onnx -o folded.onnx --fold-constants```
4. Convert to tensorflow saved model format (onnx/onnx_to_tensorflow.py)
```onnx_to_tensorflow.py <onnx_model_path> <out_model_path>```
5. Convert to FP32 TFLite model (tensorflow/tf_to_tflite.py)
```tf_to_tflite.py <tf_model_path> <output_tflite_model_path>```
6. Convert to INT8 TFLite model (tensorflow/imagenet_to_tflite_int8.py)
```imagenet_to_tflite_int8.py <tf_model_path> <output_tflite_model_path>```
7. Evaluate model on ImageNet (tensorflow/tflite/imagenet/evaluate_imagenet_tflite.py)
```evaluate_imagenet_tflite.py <model_path>```

Typical pipeline for PyTorch to INT8 tflite for Language Transformers

1. Save and train transformer model:
MobileBERT MRPC
```
python run_glue.py   --model_name_or_path google/mobilebert-uncased   --task_name mrpc    --do_train   --do_predict --do_eval   --max_seq_length 128   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 6   --output_dir mrpc --overwrite_output_dir
python -m transformers.onnx --model=mrpc ../bert-mrpc.onnx --feature=sequence-classification
mv ../bert-mrpc.onnx/model.onnx mobilebert.onnx
```
3. Make shapes static if not already (e.g:
```
python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param batch --dim_value 1 mobilebert.onnx mobilebert.onnx
python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param sequence --dim_value 128 mobilebert.onnx mobilebert.onnx
```
4. Fold constants: 
```polygraphy surgeon sanitize mobilebert.onnx -o mobilebert_folded.onnx --fold-constants```
4. Change mask to INT8 value
```
cp mobilebert_folded.onnx ../../../graph_surgeon/
python3 modify_mobilebert_2.py
```
6. Convert to tensorflow saved model format (onnx/onnx_to_tensorflow.py)
```onnx_to_tensorflow.py <onnx_model_path> <out_model_path>```
6. Convert to FP32 TFLite model (tensorflow/tf_to_tflite.py)
```tf_to_tflite.py <tf_model_path> <output_tflite_model_path>```
7. Convert to INT8 TFLite model (tensorflow/imagenet_to_tflite_int8.py)
```imagenet_to_tflite_int8.py <tf_model_path> <output_tflite_model_path>```
8. Evaluate model on ImageNet (tensorflow/tflite/imagenet/evaluate_imagenet_tflite.py)
```evaluate_imagenet_tflite.py <model_path>```
