import onnx_graphsurgeon as gs
import numpy as np
import onnx

# Though omitted in this example, in some cases, it may be useful to embed
# shape information in the graph. We can use ONNX shape inference to do this:
#
# from onnx import shape_inference
# model = shape_inference.infer_shapes(onnx.load("model.onnx"))
#
# IMPORTANT: In some cases, ONNX shape inference may not correctly infer shapes,
# which will result in an invalid subgraph. To avoid this, you can instead modify
# the tensors to include the shape information yourself.

model_name = "folded.onnx"
#model_name = "subgraph_bert.onnx"

model = onnx.load(model_name)
graph = gs.import_onnx(model)
tensors = graph.tensors()


@gs.Graph.register()
def fix_mask(self, mask):
    const = tensors["onnx::Mul_1122"]
    print(const.values)
    print(const.values.dtype)
    const.values = np.array(-127, dtype=np.float32)
    print(const.values)
    print(const.values.dtype)
    

tmap = graph.tensors()

atten = tensors["attention_scores"]#.to_variable(dtype=np.float32)

inp1 = tensors["onnx::Add_1269"]
inp2 = tensors["onnx::Add_1123"]
smax = tensors["input.8"]#.to_variable(dtype=np.float32)

graph.fix_mask("a")

graph.cleanup().toposort()

if "subgraph" in model_name:
    onnx.save(gs.export_onnx(graph), "subgraph_bert_modified.onnx")
else:
    onnx.save(gs.export_onnx(graph), "mobilebert_modified.onnx")