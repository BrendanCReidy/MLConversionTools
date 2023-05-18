from transformers import MobileBertConfig, MobileBertModel

# Initializing a MobileBERT configuration
configuration = MobileBertConfig()

# Initializing a model (with random weights) from the configuration above
model = MobileBertModel(configuration)
print(model)

# Accessing the model configuration
configuration = model.config
from datasets import load_dataset
torch_dataset = load_dataset("glue", "mrpc")