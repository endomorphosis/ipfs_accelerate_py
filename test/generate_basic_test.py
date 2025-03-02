#!/usr/bin/env python3
"""
Basic Test Generator

This is a very simple test generator that creates a basic test file for a Hugging Face model.
"""

import os
import sys
from pathlib import Path

# Configure paths
TEST_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = TEST_DIR / "sample_tests"

def main():
    """Simple test generator main function."""
    if len(sys.argv) < 2:
        print("Usage: python generate_basic_test.py MODEL_NAME [OUTPUT_DIR]")
        return 1
        
    model_name = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(SAMPLE_DIR)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a simple test file
    print(f"Generating test file for {model_name}...")
    
    # Create normalized model name for function name
    normalized_name = model_name.replace('-', '_').replace('.', '_').lower()
    
    # Basic template for a test file - simplified version
    template = f"""#!/usr/bin/env python3
\"\"\"Simple test for {model_name}\"\"\"

import os
import sys
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Test {model_name}
def test_{normalized_name}():
    print(f"Testing {model_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("{model_name}")
    model = AutoModel.from_pretrained("{model_name}")
    
    # Test with a simple input
    text = "This is a test input for {model_name}"
    inputs = tokenizer(text, return_tensors="pt")
    
    # Run model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Print results
    print(f"Input: {{text}}")
    print(f"Output shape: {{outputs.last_hidden_state.shape}}")
    
    # Model metadata with enhanced information
    model_info = {{
        "model_name": "{model_name}",
        "input_format": "text",
        "output_format": "embeddings",
        "model_type": "transformer",
        
        # Input/Output specifications
        "input": {{
            "format": "text",
            "tensor_type": "int64",
            "uses_attention_mask": True,
            "typical_shapes": ["batch_size, sequence_length"]
        }},
        "output": {{
            "format": "embedding",
            "tensor_type": "float32",
            "typical_shapes": ["batch_size, sequence_length, hidden_size"]
        }},
        
        # Detailed helper functions with arguments
        "helper_functions": {{
            "tokenizer": {{
                "description": "Tokenizes input text",
                "args": ["text", "max_length", "padding", "truncation"],
                "returns": "Dictionary with input_ids and attention_mask"
            }},
            "model_loader": {{
                "description": "Loads model from pretrained weights",
                "args": ["model_name", "cache_dir", "device"],
                "returns": "Loaded model instance"
            }}
        }},
        
        # Endpoint handler parameters
        "handler_params": {{
            "text": {{
                "description": "Input text to process",
                "type": "str or List[str]",
                "required": True
            }},
            "max_length": {{
                "description": "Maximum sequence length",
                "type": "int",
                "required": False,
                "default": 512
            }}
        }},
        
        # Dependencies
        "dependencies": {{
            "python": ">=3.8,<3.11",
            "pip": ["torch>=1.12.0", "transformers>=4.26.0", "numpy>=1.20.0"],
            "optional": {{
                "cuda": ["nvidia-cuda-toolkit>=11.6", "nvidia-cudnn>=8.3"]
            }}
        }},
        
        # Hardware requirements
        "hardware_requirements": {{
            "cpu": True,
            "cuda": True,
            "minimum_memory": "2GB",
            "recommended_memory": "4GB"
        }}
    }}
    
    print("\\nModel Information:")
    for k, v in model_info.items():
        print(f"  {{k}}: {{v}}")
    
    return True

if __name__ == "__main__":
    test_{normalized_name}()
"""
    
    # Write to output file
    output_file = output_dir / f"test_hf_{normalized_name}.py"
    with open(output_file, 'w') as f:
        f.write(template)
    
    print(f"Generated test file: {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())