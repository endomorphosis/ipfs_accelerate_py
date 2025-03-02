#!/usr/bin/env python3
'''
Simple script to generate basic test files for Hugging Face models.
'''

import os
import sys
import json
import datetime
import argparse
from pathlib import Path

# Constants for paths
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = CURRENT_DIR / "skills"

# Ensure skills directory exists
SKILLS_DIR.mkdir(exist_ok=True, parents=True)

def normalize_model_name(name):
    '''Normalize model name to match file naming conventions.'''
    return name.replace('-', '_').replace('.', '_').lower()

def generate_test_file(model, task="feature-extraction"):
    '''Generate a basic test file for a model.'''
    normalized_name = normalize_model_name(model)
    
    # Determine appropriate test models and inputs based on model
    model_to_test_model = {
        # Language models
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "t5": "t5-small",
        "gpt2": "distilgpt2",
        "llama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "bart": "facebook/bart-base",
        "albert": "albert-base-v2",
        "opt": "facebook/opt-125m",
        "phi": "microsoft/phi-1",
        "distilbert": "distilbert-base-uncased",
        "gemma": "google/gemma-2b",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "mixtral": "mistralai/Mixtral-8x7B-v0.1",
        "falcon": "tiiuae/falcon-7b",
        "mamba": "state-spaces/mamba-370m",
        "qwen2": "Qwen/Qwen2-7B-Instruct",
        "qwen3": "Qwen/Qwen2.5-7B-Instruct",
        
        # Vision models
        "vit": "google/vit-base-patch16-224",
        "clip": "openai/clip-vit-base-patch32",
        "deit": "facebook/deit-base-patch16-224",
        "detr": "facebook/detr-resnet-50",
        "sam": "facebook/sam-vit-base",
        "segformer": "nvidia/segformer-b0-finetuned-ade-512-512",
        "visual_bert": "uclanlp/visualbert-vqa-coco-pre",
        
        # Audio models
        "whisper": "openai/whisper-tiny",
        "wav2vec2": "facebook/wav2vec2-base-960h",
        "hubert": "facebook/hubert-base-ls960",
        
        # Multimodal models
        "blip": "Salesforce/blip-image-captioning-base",
        "llava": "llava-hf/llava-1.5-7b-hf",
        "layoutlmv2": "microsoft/layoutlmv2-base-uncased",
        
        # Specialized models
        "patchtsmixer": "huggingface/time-series-transformer-tourism-monthly",
        "zoedepth": "intel/dpt-large"
    }
    
    # Determine appropriate task based on model or use the provided task
    model_to_task = {
        # Language models
        "bert": "feature-extraction",
        "roberta": "feature-extraction",
        "t5": "text2text-generation",
        "gpt2": "text-generation",
        "llama": "text-generation",
        "bart": "text2text-generation",
        "albert": "feature-extraction",
        "opt": "text-generation",
        "phi": "text-generation",
        "distilbert": "feature-extraction",
        "gemma": "text-generation",
        "mistral": "text-generation",
        "mixtral": "text-generation",
        "falcon": "text-generation",
        "mamba": "text-generation",
        "qwen2": "text-generation",
        "qwen3": "text-generation",
        
        # Vision models
        "vit": "image-classification",
        "clip": "image-classification",
        "deit": "image-classification",
        "detr": "object-detection",
        "sam": "image-segmentation",
        "segformer": "image-segmentation",
        "visual_bert": "image-classification",
        
        # Audio models
        "whisper": "automatic-speech-recognition",
        "wav2vec2": "automatic-speech-recognition",
        "hubert": "audio-classification",
        
        # Multimodal models
        "blip": "image-to-text",
        "llava": "image-to-text",
        "layoutlmv2": "document-question-answering",
        
        # Specialized models
        "patchtsmixer": "time-series-prediction",
        "zoedepth": "depth-estimation" 
    }
    
    # Use appropriate test model based on model type or fall back to task-based selection
    test_model = model_to_test_model.get(normalized_name, model_to_test_model.get("bert"))
    
    # Override task if model-specific task is available
    model_task = model_to_task.get(normalized_name)
    if model_task and task == "feature-extraction":
        task = model_task
    
    template = f"""#!/usr/bin/env python3
'''Test implementation for {model}'''

import os
import sys
import json
import time
import datetime
import traceback
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np

# Try/except pattern for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = MagicMock()
    TORCH_AVAILABLE = False
    print("Warning: torch not available, using mock implementation")

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    transformers = MagicMock()
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, using mock implementation")

class test_hf_{normalized_name}:
    '''Test class for {model}'''
    
    def __init__(self, resources=None, metadata=None):
        # Initialize test class
        self.resources = resources if resources else {{
            "torch": torch,
            "numpy": np,
            "transformers": transformers
        }}
        self.metadata = metadata if metadata else {{}}
        
        # Initialize dependency status
        self.dependency_status = {{
            "torch": TORCH_AVAILABLE,
            "transformers": TRANSFORMERS_AVAILABLE,
            "numpy": True
        }}
        print(f"{normalized_name} initialization status: {{self.dependency_status}}")
        
        # Try to import the real implementation
        real_implementation = False
        try:
            from ipfs_accelerate_py.worker.skillset.hf_{normalized_name} import hf_{normalized_name}
            self.model = hf_{normalized_name}(resources=self.resources, metadata=self.metadata)
            real_implementation = True
        except ImportError:
            # Create mock model class
            class hf_{normalized_name}:
                def __init__(self, resources=None, metadata=None):
                    self.resources = resources or {{}}
                    self.metadata = metadata or {{}}
                    self.torch = resources.get("torch") if resources else None
                
                def init_cpu(self, model_name, model_type, device="cpu", **kwargs):
                    print(f"Loading {{model_name}} for CPU inference...")
                    mock_handler = lambda x: {{"output": f"Mock CPU output for {{model_name}}", 
                                         "implementation_type": "MOCK"}}
                    return None, None, mock_handler, None, 1
                
                def init_cuda(self, model_name, model_type, device_label="cuda:0", **kwargs):
                    print(f"Loading {{model_name}} for CUDA inference...")
                    mock_handler = lambda x: {{"output": f"Mock CUDA output for {{model_name}}", 
                                         "implementation_type": "MOCK"}}
                    return None, None, mock_handler, None, 1
                
                def init_openvino(self, model_name, model_type, device="CPU", **kwargs):
                    print(f"Loading {{model_name}} for OpenVINO inference...")
                    mock_handler = lambda x: {{"output": f"Mock OpenVINO output for {{model_name}}", 
                                         "implementation_type": "MOCK"}}
                    return None, None, mock_handler, None, 1
            
            self.model = hf_{normalized_name}(resources=self.resources, metadata=self.metadata)
            print(f"Warning: hf_{normalized_name} module not found, using mock implementation")
        
        # Check for specific model handler methods
        if real_implementation:
            handler_methods = dir(self.model)
            print(f"Creating minimal {normalized_name} model for testing")
        
        # Define test model and input based on task
        self.model_name = "{test_model}"
        
        # Select appropriate test input based on task
        if "{task}" == "text-generation" or "{task}" == "text2text-generation":
            self.test_input = "The quick brown fox jumps over the lazy dog"
        elif "{task}" == "feature-extraction" or "{task}" == "fill-mask":
            self.test_input = "The quick brown fox jumps over the lazy dog"
        elif "{task}" == "image-classification" or "{task}" == "object-detection" or "{task}" == "image-segmentation":
            self.test_input = "test.jpg"  # Path to test image
        elif "{task}" == "automatic-speech-recognition" or "{task}" == "audio-classification":
            self.test_input = "test.mp3"  # Path to test audio file
        elif "{task}" == "image-to-text" or "{task}" == "visual-question-answering":
            self.test_input = {{"image": "test.jpg", "prompt": "Describe this image."}}
        elif "{task}" == "document-question-answering":
            self.test_input = {{"image": "test.jpg", "question": "What is the title of this document?"}}
        elif "{task}" == "time-series-prediction":
            self.test_input = {{"past_values": [100, 120, 140, 160, 180],
                              "past_time_features": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
                              "future_time_features": [[5, 0], [6, 0], [7, 0]]}}
        else:
            self.test_input = "Test input for {normalized_name}"
            
        # Report model and task selection
        print(f"Using model {{self.model_name}} for {task} task")
        
        # Initialize collection arrays for examples and status
        self.examples = []
        self.status_messages = {{}}
    
    def test(self):
        '''Run tests for the model'''
        results = {{}}
        
        # Test basic initialization
        results["init"] = "Success" if self.model is not None else "Failed initialization"
        
        # CPU Tests
        try:
            # Initialize for CPU
            endpoint, processor, handler, queue, batch_size = self.model.init_cpu(
                self.model_name, "{task}", "cpu"
            )
            
            results["cpu_init"] = "Success" if endpoint is not None or processor is not None or handler is not None else "Failed initialization"
            
            # Safely run handler with appropriate error handling
            if handler is not None:
                try:
                    output = handler(self.test_input)
                    
                    # Verify output type - could be dict, tensor, or other types
                    if isinstance(output, dict):
                        impl_type = output.get("implementation_type", "UNKNOWN")
                    elif hasattr(output, 'real_implementation'):
                        impl_type = "REAL" if output.real_implementation else "MOCK"
                    else:
                        impl_type = "REAL" if output is not None else "MOCK"
                    
                    results["cpu_handler"] = f"Success ({{impl_type}})"
                    
                    # Record example with safe serialization
                    self.examples.append({{
                        "input": str(self.test_input),
                        "output": {{
                            "type": str(type(output)),
                            "implementation_type": impl_type
                        }},
                        "timestamp": datetime.datetime.now().isoformat(),
                        "platform": "CPU"
                    }})
                except Exception as handler_err:
                    results["cpu_handler_error"] = str(handler_err)
                    traceback.print_exc()
            else:
                results["cpu_handler"] = "Failed (handler is None)"
        except Exception as e:
            results["cpu_error"] = str(e)
            traceback.print_exc()
        
        # Return structured results
        return {{
            "status": results,
            "examples": self.examples,
            "metadata": {{
                "model_name": self.model_name,
                "model_type": "{model}",
                "test_timestamp": datetime.datetime.now().isoformat()
            }}
        }}
    
    def __test__(self):
        '''Run tests and save results'''
        test_results = {{}}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {{
                "status": {{"test_error": str(e)}},
                "examples": [],
                "metadata": {{
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }}
            }}
        
        # Create directories if needed
        base_dir = os.path.dirname(os.path.abspath(__file__))
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        if not os.path.exists(collected_dir):
            os.makedirs(collected_dir, mode=0o755, exist_ok=True)
        
        # Format the test results for JSON serialization
        safe_test_results = {{
            "status": test_results.get("status", {{}}),
            "examples": [
                {{
                    "input": ex.get("input", ""),
                    "output": {{
                        "type": ex.get("output", {{}}).get("type", "unknown"),
                        "implementation_type": ex.get("output", {{}}).get("implementation_type", "UNKNOWN")
                    }},
                    "timestamp": ex.get("timestamp", ""),
                    "platform": ex.get("platform", "")
                }}
                for ex in test_results.get("examples", [])
            ],
            "metadata": test_results.get("metadata", {{}})
        }}
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(collected_dir, f'hf_{normalized_name}_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(safe_test_results, f, indent=2)
        except Exception as save_err:
            print(f"Error saving results: {{save_err}}")
        
        return test_results

if __name__ == "__main__":
    try:
        print(f"Starting {normalized_name} test...")
        test_instance = test_hf_{normalized_name}()
        results = test_instance.__test__()
        print(f"{normalized_name} test completed")
        
        # Extract implementation status
        status_dict = results.get("status", {{}})
        
        # Print summary
        model_name = results.get("metadata", {{}}).get("model_type", "UNKNOWN")
        print(f"\\n{{model_name.upper()}} TEST RESULTS:")
        for key, value in status_dict.items():
            print(f"{{key}}: {{value}}")
        
    except KeyboardInterrupt:
        print("Test stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {{e}}")
        traceback.print_exc()
        sys.exit(1)
"""
    
    # Save the file
    file_path = SKILLS_DIR / f"test_hf_{normalized_name}.py"
    with open(file_path, "w") as f:
        f.write(template)
    os.chmod(file_path, 0o755)
    
    print(f"Generated test file for {model} at {file_path}")
    return file_path

def parse_args():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(description="Generate basic test files for Hugging Face models")
    parser.add_argument("models", nargs="+", help="Models to generate tests for")
    parser.add_argument("--task", default="feature-extraction", help="Primary task for the model")
    return parser.parse_args()

def main():
    '''Main function'''
    args = parse_args()
    
    for model in args.models:
        generate_test_file(model, args.task)

if __name__ == "__main__":
    main()