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

model = onnx.load("subgraph_bert.onnx")
graph = gs.import_onnx(model)

tensors = graph.tensors()

@gs.Graph.register()
def gather(self, data, indices):
    return self.layer(op="Gather", inputs=[data, indices], outputs=["gather_out_gs"])[0]


@gs.Graph.register()
def concat(self, inputs, axis=0):
    return self.layer(op="Concat", inputs=inputs, attrs={"axis": axis}, outputs=["concat_out_gs"])[0]

@gs.Graph.register()
def replace_with_gather(self, inputs, outputs):
    # Disconnect output nodes of all input tensors
    for inp in inputs:
        inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    gat1 = self.layer(op="Gather", inputs=[inputs[0], [0,1]], attrs={"axis": 1}, outputs=["gather_out_gs"])[0]
    gat2 = self.layer(op="Gather", inputs=[inputs[0], [2,3]], attrs={"axis": 1}, outputs=["gather_out_gs"])[0]
    concat = self.layer(op="Concat", inputs=[gat1,gat2], attrs={"axis": 1}, outputs="concat")[0]
    smax = self.layer(op="Softmax", inputs=[concat], attrs={"axis": 3}, outputs=outputs)[0]
    return smax

tmap = graph.tensors()

atten = tensors["attention_scores"].to_variable(dtype=np.float32)

gat1 = graph.gather(atten, indices=[0,1])
gat2 = graph.gather(atten, indices=[2,3])
flat = graph.concat([gat1, gat2])
smax = tensors["input.8"].to_variable(dtype=np.float32)

graph.replace_with_gather([atten], [smax])

print(gat1)

graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), "subgraph_bert_modified.onnx")