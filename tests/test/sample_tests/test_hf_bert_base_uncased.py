#!/usr/bin/env python3
"""Simple test for bert-base-uncased"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Test bert-base-uncased
def test_bert_base_uncased():
    print(f"Testing bert-base-uncased")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    
    # Test with a simple input
    text = "This is a test input for bert-base-uncased"
    inputs = tokenizer(text, return_tensors="pt")
    
    # Run model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Print results
        print(f"Input: {}}}}}}}}}}}}text}")
        print(f"Output shape: {}}}}}}}}}}}}outputs.last_hidden_state.shape}")
    
    # Model metadata with enhanced information
        model_info = {}}}}}}}}}}}}
        "model_name": "bert-base-uncased",
        "input_format": "text",
        "output_format": "embeddings",
        "model_type": "transformer",
        
        # Input/Output specifications
        "input": {}}}}}}}}}}}}
        "format": "text",
        "tensor_type": "int64",
        "uses_attention_mask": True,
        "typical_shapes": ["batch_size, sequence_length"],
        },
        "output": {}}}}}}}}}}}}
        "format": "embedding",
        "tensor_type": "float32",
        "typical_shapes": ["batch_size, sequence_length, hidden_size"],
        },
        
        # Detailed helper functions with arguments
        "helper_functions": {}}}}}}}}}}}}
        "tokenizer": {}}}}}}}}}}}}
        "description": "Tokenizes input text",
        "args": ["text", "max_length", "padding", "truncation"],
        "returns": "Dictionary with input_ids and attention_mask"
        },
        "model_loader": {}}}}}}}}}}}}
        "description": "Loads model from pretrained weights",
        "args": ["model_name", "cache_dir", "device"],
        "returns": "Loaded model instance"
        }
        },
        
        # Endpoint handler parameters
        "handler_params": {}}}}}}}}}}}}
        "text": {}}}}}}}}}}}}
        "description": "Input text to process",
        "type": "str or List[str]",
        "required": True
        },
        "max_length": {}}}}}}}}}}}}
        "description": "Maximum sequence length",
        "type": "int",
        "required": False,
        "default": 512
        }
        },
        
        # Dependencies
        "dependencies": {}}}}}}}}}}}}
        "python": ">=3.8,<3.11",
        "pip": ["torch>=1.12.0", "transformers>=4.26.0", "numpy>=1.20.0"],
        "optional": {}}}}}}}}}}}}
        "cuda": ["nvidia-cuda-toolkit>=11.6", "nvidia-cudnn>=8.3"],
        }
        },
        
        # Hardware requirements
        "hardware_requirements": {}}}}}}}}}}}}
        "cpu": True,
        "cuda": True,
        "minimum_memory": "2GB",
        "recommended_memory": "4GB"
        }
        }
    
        print("\nModel Information:")
    for k, v in model_info.items():
        print(f"  {}}}}}}}}}}}}k}: {}}}}}}}}}}}}v}")
    
        return True

if __name__ == "__main__":
    test_bert_base_uncased()