import torch
from transformers import AutoModel
from ipfs_transformers import AutoModel
#from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights
from ipfs_accelerate import load_checkpoint_and_dispatch

checkpoint = "bge-small-en-v1.5"
weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")
config = AutoConfig.from_pretrained(checkpoint)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model = load_checkpoint_and_dispatch(
    model, checkpoint=weights_location, device_map="auto"
)


#model = load_checkpoint_and_dispatch(
#    model, checkpoint=checkpoint_file, device_map="lilypad"
#)

#model = load_checkpoint_and_dispatch(
#    model, checkpoint=checkpoint_file, device_map="akash"
#)


