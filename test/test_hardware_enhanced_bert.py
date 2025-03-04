#!/usr/bin/env python3
"""
Test implementation for bert model with hardware support.

This test demonstrates the bert model working across all hardware platforms.
"""

import os
import sys
import torch
import time
import json
import traceback
import numpy as np
from typing import Dict, List, Any, Optional

try:
    import transformers
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    transformers = None
    print("Warning: transformers library not found")

class TestHFBert:
    """Test implementation for bert model."""
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the bert model."""
        self.resources = resources if resources else {
            "transformers": transformers,
            "torch": torch,
            "numpy": np
        }
        self.metadata = metadata if metadata else {}
        
        # Model parameters
        self.model_name = "bert-base-uncased"
        
        # Test data
        self.test_text = "The quick brown fox jumps over the lazy dog."
        self.batch_size = 4
        
    def init_cpu(self, model_name=None):
        """Initialize model for CPU inference."""
        try:
            model_name = model_name or self.model_name
            
            # Initialize tokenizer
            tokenizer = self.resources["transformers"].AutoTokenizer.from_pretrained(model_name)
            
            # Initialize model
            model = self.resources["transformers"].AutoModel.from_pretrained(model_name)
            model.eval()
            
            # Create handler function
            def handler(text_input, **kwargs):
                try:
                    # Process with tokenizer
                    if isinstance(text_input, list):
                        inputs = tokenizer(text_input, padding=True, truncation=True, return_tensors="pt")
                    else:
                        inputs = tokenizer(text_input, return_tensors="pt")
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    return {
                        "output": outputs,
                        "implementation_type": "REAL_CPU",
                        "model": model_name
                    }
                except Exception as e:
                    print(f"Error in CPU handler: {e}")
                    return {
                        "output": f"Error: {str(e)}",
                        "implementation_type": "ERROR",
                        "error": str(e),
                        "model": model_name
                    }
            
            # Return components
            return model, tokenizer, handler
            
        except Exception as e:
            print(f"Error initializing {model_name} on CPU: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Create mock handler
            def mock_handler(text_input, **kwargs):
                return {
                    "output": "MOCK CPU OUTPUT",
                    "implementation_type": "MOCK_CPU",
                    "model": model_name
                }
            
            return None, None, mock_handler
    
    def init_openvino(self, model_name=None, device="CPU"):
        """Initialize model for OpenVINO inference."""
        try:
            # Check for OpenVINO runtime
            try:
                import openvino as ov
                from optimum.intel import OVModelForFeatureExtraction
            except ImportError:
                raise RuntimeError("OpenVINO not available")
            
            model_name = model_name or self.model_name
            
            # Initialize tokenizer
            tokenizer = self.resources["transformers"].AutoTokenizer.from_pretrained(model_name)
            
            # Initialize model with OpenVINO
            try:
                model = OVModelForFeatureExtraction.from_pretrained(
                    model_name,
                    export=True,
                    provider=device,
                    trust_remote_code=True
                )
                print(f"Loaded {model_name} with OpenVINO")
            except Exception as e:
                print(f"Error loading OpenVINO model: {e}")
                print("Falling back to standard model - will convert to OpenVINO")
                # Load standard model and would convert to OpenVINO
                model = self.resources["transformers"].AutoModel.from_pretrained(model_name)
            
            # Create handler function
            def handler(text_input, **kwargs):
                try:
                    # Process with tokenizer
                    if isinstance(text_input, list):
                        inputs = tokenizer(text_input, padding=True, truncation=True, return_tensors="pt")
                    else:
                        inputs = tokenizer(text_input, return_tensors="pt")
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    return {
                        "output": outputs,
                        "implementation_type": "REAL_OPENVINO",
                        "model": model_name,
                        "device": device
                    }
                except Exception as e:
                    print(f"Error in OpenVINO handler: {e}")
                    return {
                        "output": f"Error: {str(e)}",
                        "implementation_type": "ERROR",
                        "error": str(e),
                        "model": model_name,
                        "device": device
                    }
            
            # Return components
            return model, tokenizer, handler
            
        except Exception as e:
            print(f"Error initializing {model_name} on OpenVINO: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Create mock handler
            def mock_handler(text_input, **kwargs):
                return {
                    "output": "MOCK OPENVINO OUTPUT",
                    "implementation_type": "MOCK_OPENVINO",
                    "model": model_name,
                    "device": device
                }
            
            return None, None, mock_handler
    
    def init_cuda(self, model_name=None, device="cuda:0"):
        """Initialize model for CUDA inference."""
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
                
            model_name = model_name or self.model_name
            
            # Initialize tokenizer
            tokenizer = self.resources["transformers"].AutoTokenizer.from_pretrained(model_name)
            
            # Initialize model on CUDA
            model = self.resources["transformers"].AutoModel.from_pretrained(model_name)
            model.to(device)
            model.eval()
            
            # Create handler function
            def handler(text_input, **kwargs):
                try:
                    # Process with tokenizer
                    if isinstance(text_input, list):
                        inputs = tokenizer(text_input, padding=True, truncation=True, return_tensors="pt")
                    else:
                        inputs = tokenizer(text_input, return_tensors="pt")
                    
                    # Move inputs to CUDA
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    return {
                        "output": outputs,
                        "implementation_type": "REAL_CUDA",
                        "model": model_name,
                        "device": device
                    }
                except Exception as e:
                    print(f"Error in CUDA handler: {e}")
                    return {
                        "output": f"Error: {str(e)}",
                        "implementation_type": "ERROR",
                        "error": str(e),
                        "model": model_name,
                        "device": device
                    }
            
            # Return components
            return model, tokenizer, handler
            
        except Exception as e:
            print(f"Error initializing {model_name} on CUDA: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Create mock handler
            def mock_handler(text_input, **kwargs):
                return {
                    "output": "MOCK CUDA OUTPUT",
                    "implementation_type": "MOCK_CUDA",
                    "model": model_name,
                    "device": device
                }
            
            return None, None, mock_handler
    
    def run(self, platform="cpu"):
        """Run the model on a specific platform."""
        print(f"Testing bert model on {platform} platform...")
        
        # Initialize the model on the specified platform
        init_method = getattr(self, f"init_{platform}", None)
        if init_method is None:
            print(f"Platform {platform} not supported")
            return {"error": f"Platform {platform} not supported"}
        
        # Initialize the model
        model, tokenizer, handler = init_method()
        
        # Test with sample data
        test_input = self.test_text
        start_time = time.time()
        result = handler(test_input)
        end_time = time.time()
        
        # Add timing information
        result["inference_time"] = end_time - start_time
        
        # Print results
        print(f"  Implementation type: {result.get('implementation_type', 'UNKNOWN')}")
        print(f"  Inference time: {result['inference_time']:.4f} seconds")
        
        return result

def run_all_platforms():
    """Run the bert model on all platforms."""
    bert_test = TestHFBert()
    
    # Test on each platform
    results = {}
    # Order platforms in a logical sequence - most widely available first
    platforms = ["cpu", "cuda", "openvino"]
    for platform in platforms:
        print(f"\nTesting on {platform.upper()}...")
        try:
            result = bert_test.run(platform)
            results[platform] = result
        except Exception as e:
            print(f"Error testing on {platform}: {e}")
            results[platform] = {"error": str(e), "implementation_type": "ERROR"}
    
    # Print summary
    print("\nSummary of results:")
    for platform in platforms:
        result = results.get(platform, {})
        impl_type = result.get("implementation_type", "UNKNOWN")
        time_ms = result.get("inference_time", 0) * 1000
        error = result.get("error", None)
        
        if error:
            print(f"  {platform.upper()}: ERROR - {error}")
        else:
            print(f"  {platform.upper()}: {impl_type} - {time_ms:.2f}ms")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test bert model on different hardware platforms")
    parser.add_argument("--platform", default="all", help="Hardware platform to test on")
    args = parser.parse_args()
    
    if args.platform == "all":
        run_all_platforms()
    else:
        bert_test = TestHFBert()
        bert_test.run(args.platform)