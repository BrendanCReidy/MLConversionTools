import onnx
mymodel = "/workspace/model_profiling/pytorch/models/language/mobilebert_mrpc_inp_shape.onnx"

def change_input_dim(model):
    # Use some symbolic name not used for any other dimension
    # or an actal value
    actual_batch_dim = 1
    input_size = 384

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    outputs = model.graph.output
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        dim2 = input.type.tensor_type.shape.dim[1]
        # update dim to be a symbolic value
        #dim1.dim_param = 1
        # or update it to be an actual value:
        dim1.dim_value = actual_batch_dim
        dim2.dim_value = input_size

    outputs[0].type.tensor_type.shape.dim[0].dim_value = actual_batch_dim


def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)

apply(change_input_dim, mymodel, "/workspace/model_profiling/pytorch/models/language/mobilebert_mrpc.onnx")