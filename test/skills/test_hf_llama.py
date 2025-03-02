#!/usr/bin/env python3
'''Test implementation for llama'''

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

class test_hf_llama:
    '''Test class for llama'''
    
    def __init__(self, resources=None, metadata=None):
        # Initialize test class
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np,
            "transformers": transformers
        }
        self.metadata = metadata if metadata else {}
        
        # Initialize dependency status
        self.dependency_status = {
            "torch": TORCH_AVAILABLE,
            "transformers": TRANSFORMERS_AVAILABLE,
            "numpy": True
        }
        print(f"llama initialization status: {self.dependency_status}")
        
        # Try to import the real implementation
        real_implementation = False
        try:
            from ipfs_accelerate_py.worker.skillset.hf_llama import hf_llama
            self.model = hf_llama(resources=self.resources, metadata=self.metadata)
            real_implementation = True
        except ImportError:
            # Create mock model class
            class hf_llama:
                def __init__(self, resources=None, metadata=None):
                    self.resources = resources or {}
                    self.metadata = metadata or {}
                    self.torch = resources.get("torch") if resources else None
                
                def init_cpu(self, model_name, model_type, device="cpu", **kwargs):
                    print(f"Loading {model_name} for CPU inference...")
                    mock_handler = lambda x: {"output": f"Mock CPU output for {model_name}", 
                                         "implementation_type": "MOCK"}
                    return None, None, mock_handler, None, 1
                
                def init_cuda(self, model_name, model_type, device_label="cuda:0", **kwargs):
                    print(f"Loading {model_name} for CUDA inference...")
                    mock_handler = lambda x: {"output": f"Mock CUDA output for {model_name}", 
                                         "implementation_type": "MOCK"}
                    return None, None, mock_handler, None, 1
                
                def init_openvino(self, model_name, model_type, device="CPU", **kwargs):
                    print(f"Loading {model_name} for OpenVINO inference...")
                    mock_handler = lambda x: {"output": f"Mock OpenVINO output for {model_name}", 
                                         "implementation_type": "MOCK"}
                    return None, None, mock_handler, None, 1
            
            self.model = hf_llama(resources=self.resources, metadata=self.metadata)
            print(f"Warning: hf_llama module not found, using mock implementation")
        
        # Check for specific model handler methods
        if real_implementation:
            handler_methods = dir(self.model)
            print(f"Creating minimal llama model for testing")
        
        # Define test model and input based on task
        self.model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
        
        # Select appropriate test input based on task
        if "text-generation" == "text-generation" or "text-generation" == "text2text-generation":
            self.test_input = "The quick brown fox jumps over the lazy dog"
        elif "text-generation" == "feature-extraction" or "text-generation" == "fill-mask":
            self.test_input = "The quick brown fox jumps over the lazy dog"
        elif "text-generation" == "image-classification" or "text-generation" == "object-detection" or "text-generation" == "image-segmentation":
            self.test_input = "test.jpg"  # Path to test image
        elif "text-generation" == "automatic-speech-recognition" or "text-generation" == "audio-classification":
            self.test_input = "test.mp3"  # Path to test audio file
        elif "text-generation" == "image-to-text" or "text-generation" == "visual-question-answering":
            self.test_input = {"image": "test.jpg", "prompt": "Describe this image."}
        elif "text-generation" == "document-question-answering":
            self.test_input = {"image": "test.jpg", "question": "What is the title of this document?"}
        elif "text-generation" == "time-series-prediction":
            self.test_input = {"past_values": [100, 120, 140, 160, 180],
                              "past_time_features": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
                              "future_time_features": [[5, 0], [6, 0], [7, 0]]}
        else:
            self.test_input = "Test input for llama"
            
        # Report model and task selection
        print(f"Using model {self.model_name} for text-generation task")
        
        # Initialize collection arrays for examples and status
        self.examples = []
        self.status_messages = {}
    
    def test(self):
        '''Run tests for the model'''
        results = {}
        
        # Test basic initialization
        results["init"] = "Success" if self.model is not None else "Failed initialization"
        
        # CPU Tests
        try:
            # Initialize for CPU
            endpoint, processor, handler, queue, batch_size = self.model.init_cpu(
                self.model_name, "text-generation", "cpu"
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
                    
                    results["cpu_handler"] = f"Success ({impl_type})"
                    
                    # Record example with safe serialization
                    self.examples.append({
                        "input": str(self.test_input),
                        "output": {
                            "type": str(type(output)),
                            "implementation_type": impl_type
                        },
                        "timestamp": datetime.datetime.now().isoformat(),
                        "platform": "CPU"
                    })
                except Exception as handler_err:
                    results["cpu_handler_error"] = str(handler_err)
                    traceback.print_exc()
            else:
                results["cpu_handler"] = "Failed (handler is None)"
        except Exception as e:
            results["cpu_error"] = str(e)
            traceback.print_exc()
        
        # Return structured results
        return {
            "status": results,
            "examples": self.examples,
            "metadata": {
                "model_name": self.model_name,
                "model_type": "llama",
                "test_timestamp": datetime.datetime.now().isoformat()
            }
        }
    
    def __test__(self):
        '''Run tests and save results'''
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {
                "status": {"test_error": str(e)},
                "examples": [],
                "metadata": {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            }
        
        # Create directories if needed
        base_dir = os.path.dirname(os.path.abspath(__file__))
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        if not os.path.exists(collected_dir):
            os.makedirs(collected_dir, mode=0o755, exist_ok=True)
        
        # Format the test results for JSON serialization
        safe_test_results = {
            "status": test_results.get("status", {}),
            "examples": [
                {
                    "input": ex.get("input", ""),
                    "output": {
                        "type": ex.get("output", {}).get("type", "unknown"),
                        "implementation_type": ex.get("output", {}).get("implementation_type", "UNKNOWN")
                    },
                    "timestamp": ex.get("timestamp", ""),
                    "platform": ex.get("platform", "")
                }
                for ex in test_results.get("examples", [])
            ],
            "metadata": test_results.get("metadata", {})
        }
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(collected_dir, f'hf_llama_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(safe_test_results, f, indent=2)
        except Exception as save_err:
            print(f"Error saving results: {save_err}")
        
        return test_results

if __name__ == "__main__":
    try:
        print(f"Starting llama test...")
        test_instance = test_hf_llama()
        results = test_instance.__test__()
        print(f"llama test completed")
        
        # Extract implementation status
        status_dict = results.get("status", {})
        
        # Print summary
        model_name = results.get("metadata", {}).get("model_type", "UNKNOWN")
        print(f"\n{model_name.upper()} TEST RESULTS:")
        for key, value in status_dict.items():
            print(f"{key}: {value}")
        
    except KeyboardInterrupt:
        print("Test stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
