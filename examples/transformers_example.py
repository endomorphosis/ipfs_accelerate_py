import torch
import tempfile
import requests
import json
import os
import sys
import asyncio
import random
import hashlib
import time
import logging


# Get the absolute path to the external directory
# Assuming the script is being run from a location where "external" is a subdirectory
external_ipfs_model_manager_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external/ipfs_model_manager_py/ipfs_model_manager_py"))
print(f"external_ipfs_model_manager_path = {external_ipfs_model_manager_path}")
sys.path.append(external_ipfs_model_manager_path)

external_ipfs_kit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external/ipfs_kit_py/ipfs_kit_py"))
print(f"external_ipfs_kit_path = {external_ipfs_kit_path}")
sys.path.append(external_ipfs_kit_path)

external_ipfs_transformers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external/ipfs_transformers_py/ipfs_transformers_py"))
print(f"external_ipfs_transformers_path = {external_ipfs_transformers_path}")
sys.path.append(external_ipfs_transformers_path)


from ipfs_accelerate_py import ipfs_accelerate_py

# Now we can import modules from the external directory
import ipfs_kit_py
import ipfs_model_manager_py
#import libp2p_kit_py  # Commented out to avoid dependency issues


# Initialize with the correct parameter names (resources and metadata)
# This matches the actual __init__ signature in the class
resources = {
    # Initialize required dictionaries
    "queues": {},
    "queue": {},
    "batch_sizes": {},
    "endpoint_handler": {},
    "consumer_tasks": {},
    "queue_tasks": {},
    "caches": {},
    "tokenizer": {},

    # Add transformers configuration into resources
    "transformers": {
        "cache_dir": "./model_cache",
        "revision": "main"
    }
}

metadata = {
    "description": "Integrated accelerator with IPFS support",
    "role": "leecher",  # Role is part of metadata
    # IPFS kit configuration in metadata
    "ipfs_kit": {
        "async_backend": "asyncio",
        "num_workers": 4
    }
}

# Create the accelerator with the correct parameter names
accelerator = ipfs_accelerate_py(resources=resources, metadata=metadata)

# Run a model with automatic hardware selection
result = accelerator.run_model(
    "bert-base-uncased",
    {"input_ids": [[101, 2054, 2003, 2026, 2171, 2024, 2059, 2038, 102]]},
    "text_embedding"
)

# Check the result
if result.get("success", False):
    embeddings = result["outputs"]["last_hidden_state"]
    print(f"Generated embeddings with shape: {embeddings.shape}")
else:
    error = result.get("error", "Unknown error")
    print(f"Error: {error}")


def get_optimal_backend():
    """Determine the optimal hardware backend for a model."""
    # Implement hardware detection logic
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"


# Get the optimal backend
device = get_optimal_backend()

# Alternative: run by IPFS CID directly
# Replace QmCID_HERE with an actual CID
# See https://huggingface.co/datasets/buttercutter/models-metadata-dataset
cid = "5c3eb3fb2a3b61094328465ba61fcd4272090d67"
model_name = "bert-base-uncased" #cid,
input_ids = torch.tensor([[101, 2054, 2003, 2026, 2171, 2024, 2059, 2038, 102]]).to(device)
attention_mask = torch.ones_like(input_ids).to(device)

result_ipfs = accelerator.run_model(
    model_name,
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    },
    "text_embedding",
    device=device,
    use_ipfs=True,
    ipfs_cid=cid
)

# Check the IPFS result
if result_ipfs.get("success", False):
    embeddings_ipfs = result_ipfs["outputs"]["last_hidden_state"]
    print(f"Generated embeddings from IPFS with shape: {embeddings_ipfs.shape}")
else:
    error_ipfs = result_ipfs.get("error", "Unknown error")
    print(f"IPFS Error: {error_ipfs}")
