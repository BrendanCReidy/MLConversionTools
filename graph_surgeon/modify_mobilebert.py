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
def separate_softmax(self, inp1, inp2, outputs):
    # Disconnect output nodes of all input tensors
    #inp_shape = inp.outputs[0].shape
    const = tensors["onnx::Mul_1122"]
    print(const.values.shape)
    print(const.values.dtype)
    const.values = np.array(-127, dtype=np.float32)
    print(const.values.shape)
    print(const.values.dtype)
    inp = inp1
    inp_shape = inp.shape
    gather_shape = list(inp_shape)
    gather_shape[1] = 1
    inp1.outputs.clear()
    #inp2.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    #reshape = self.layer(op="Reshape", inputs=[inp, [1,4,384,384]], outputs=["reshape"])[0]
    #print(reshape)
    #reshape.shape = [1,4,384,384]
    #reshape.dtype=inp.dtype
    #print(reshape)
    #smax = self.layer(op="Softmax", inputs=[reshape], attrs={"axis": 3}, outputs=outputs)[0]

    # Insert the new node.
    gat1 = self.layer(op="Gather", inputs=[inp, [0]], attrs={"axis": 1}, name="Gather", outputs=["gather_out_gs"])[0]
    gat2 = self.layer(op="Gather", inputs=[inp, [1]], attrs={"axis": 1}, name="Gather", outputs=["gather_out_gs"])[0]
    gat3 = self.layer(op="Gather", inputs=[inp, [2]], attrs={"axis": 1}, name="Gather", outputs=["gather_out_gs"])[0]
    gat4 = self.layer(op="Gather", inputs=[inp, [3]], attrs={"axis": 1}, name="Gather", outputs=["gather_out_gs"])[0]
    gat1.shape = gather_shape
    gat1.dtype = inp.dtype
    gat2.shape = gather_shape
    gat2.dtype = inp.dtype
    gat3.shape = gather_shape
    gat3.dtype = inp.dtype
    gat4.shape = gather_shape
    gat4.dtype = inp.dtype

    add1 = self.layer(op="Add", inputs=[gat1, inp2], name="Add", outputs=["add"])[0]
    add2 = self.layer(op="Add", inputs=[gat2, inp2], name="Add", outputs=["add"])[0]
    add3 = self.layer(op="Add", inputs=[gat3, inp2], name="Add", outputs=["add"])[0]
    add4 = self.layer(op="Add", inputs=[gat4, inp2], name="Add", outputs=["add"])[0]
    add1.shape = gather_shape
    add1.dtype = inp.dtype
    add2.shape = gather_shape
    add2.dtype = inp.dtype
    add3.shape = gather_shape
    add3.dtype = inp.dtype
    add4.shape = gather_shape
    add4.dtype = inp.dtype
    #print(add1.inputs)

    smax1 = self.layer(op="Softmax", inputs=[add1], name="Softmax", attrs={"axis": 3}, outputs=["softmax"])[0]
    smax2 = self.layer(op="Softmax", inputs=[add2], name="Softmax", attrs={"axis": 3}, outputs=["softmax"])[0]
    smax3 = self.layer(op="Softmax", inputs=[add3], name="Softmax", attrs={"axis": 3}, outputs=["softmax"])[0]
    smax4 = self.layer(op="Softmax", inputs=[add4], name="Softmax", attrs={"axis": 3}, outputs=["softmax"])[0]
    smax1.shape = gather_shape
    smax1.dtype = inp.dtype
    smax2.shape = gather_shape
    smax2.dtype = inp.dtype
    smax3.shape = gather_shape
    smax3.dtype = inp.dtype
    smax4.shape = gather_shape
    smax4.dtype = inp.dtype
    concat = self.layer(op="Concat", inputs=[smax1,smax2,smax3,smax4], name="Concat", attrs={"axis": 1}, outputs=outputs)[0]
    concat.shape = inp_shape
    concat.dtype = inp.dtype

tmap = graph.tensors()

atten = tensors["attention_scores"]#.to_variable(dtype=np.float32)

inp1 = tensors["onnx::Add_1269"]
inp2 = tensors["onnx::Add_1123"]
smax = tensors["input.8"]#.to_variable(dtype=np.float32)

atten_names = []

for t in tensors.keys():
    if "attention_scores" in t:
        inp = tensors[t].inputs[0]
        out = tensors[t].outputs[0]
        inp_name = inp.inputs[0].name
        out_name = out.outputs[0].name
        atten_names.append((inp_name, out_name))

for i, (inp_name, out_name) in enumerate(atten_names):
    in_tensor = tensors[inp_name]
    out_tensor = tensors[out_name]
    graph.separate_softmax(in_tensor, inp2, [out_tensor])
graph.cleanup().toposort()

if "subgraph" in model_name:
    onnx.save(gs.export_onnx(graph), "subgraph_bert_modified.onnx")
else:
    onnx.save(gs.export_onnx(graph), "mobilebert_modified.onnx")