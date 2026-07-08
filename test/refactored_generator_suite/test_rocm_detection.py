#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify ROCm (AMD GPU) detection and initialization.
This script tests whether the hardware detection properly identifies ROCm
and whether the initialization code can properly load models on AMD GPUs.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_rocm_detection")

def check_rocm_environment():
    """Check if ROCm environment is available.
    
    Returns:
        Tuple of (is_available, details)
    """
    details = {}
    
    # Check for environment variables
    env_vars = {
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "ROCM_PATH": os.environ.get("ROCM_PATH"),
        "HSA_OVERRIDE_GFX_VERSION": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
    }
    details["environment"] = env_vars
    
    # Try to import torch
    try:
        import torch
        details["torch_version"] = torch.__version__
        
        # Check for CUDA availability (ROCm uses CUDA API)
        if torch.cuda.is_available():
            details["cuda_available"] = True
            
            # Get device count
            device_count = torch.cuda.device_count()
            details["device_count"] = device_count
            
            # Get device properties
            devices = []
            for i in range(device_count):
                device_props = torch.cuda.get_device_properties(i)
                device_name = device_props.name
                total_memory = device_props.total_memory / (1024 * 1024 * 1024)  # GB
                
                devices.append({
                    "index": i,
                    "name": device_name,
                    "total_memory_gb": round(total_memory, 2),
                })
                
                # Check if this is an AMD GPU
                if "AMD" in device_name or "Radeon" in device_name:
                    details["has_amd_gpu"] = True
            
            details["devices"] = devices
            
            # Check if HIP API is available
            try:
                if hasattr(torch, "hip") and torch.hip.is_available():
                    details["hip_available"] = True
                else:
                    details["hip_available"] = False
            except:
                details["hip_available"] = False
            
            # Determine ROCm availability from all checks
            is_rocm_available = details.get("has_amd_gpu", False) or details.get("hip_available", False)
            details["rocm_available"] = is_rocm_available
            
            return is_rocm_available, details
        else:
            details["cuda_available"] = False
            details["rocm_available"] = False
            return False, details
    
    except ImportError:
        details["error"] = "PyTorch not installed"
        return False, details
    except Exception as e:
        details["error"] = str(e)
        return False, details

def test_bert_on_rocm():
    """Test loading BERT on ROCm.
    
    Returns:
        Tuple of (success, details)
    """
    details = {}
    
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        # Record start time
        start_time = time.time()
        
        # Define model name
        model_name = "bert-base-uncased"
        details["model_name"] = model_name
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            details["error"] = "CUDA not available"
            return False, details
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        details["tokenizer_loaded"] = True
        
        # Check if half precision is supported
        try:
            # Try a small tensor in half precision
            test_tensor = torch.ones((10, 10), dtype=torch.float16, device="cuda")
            del test_tensor
            details["half_precision_supported"] = True
            dtype = torch.float16
        except Exception as e:
            details["half_precision_supported"] = False
            details["half_precision_error"] = str(e)
            dtype = torch.float32
        
        # Load model
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto"
        )
        details["model_loaded"] = True
        
        # Check if model is on CUDA
        if hasattr(model, "device"):
            details["model_device"] = str(model.device)
        
        # Run inference
        input_ids = tokenizer("Hello, this is a test", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            outputs = model(input_ids)
        
        details["inference_successful"] = True
        details["output_shape"] = list(outputs.last_hidden_state.shape)
        
        # Record end time
        end_time = time.time()
        details["duration"] = end_time - start_time
        
        return True, details
    
    except Exception as e:
        details["error"] = str(e)
        return False, details

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test ROCm detection and initialization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show more details")
    parser.add_argument("--run-model", "-m", action="store_true", help="Run a model on ROCm")
    args = parser.parse_args()
    
    print("\n=== ROCm Environment Check ===\n")
    
    # Check ROCm environment
    is_rocm_available, env_details = check_rocm_environment()
    
    if is_rocm_available:
        print("✅ ROCm environment is available!")
        
        # Show device details
        if "devices" in env_details:
            for device in env_details["devices"]:
                print(f"  - GPU {device['index']}: {device['name']} ({device['total_memory_gb']} GB)")
        
        # Show environment variables
        if args.verbose and "environment" in env_details:
            print("\nEnvironment variables:")
            for var, value in env_details["environment"].items():
                if value:
                    print(f"  - {var}: {value}")
        
        # Test model if requested
        if args.run_model:
            print("\n=== Testing BERT Model on ROCm ===\n")
            success, model_details = test_bert_on_rocm()
            
            if success:
                print(f"✅ Successfully loaded and ran BERT on ROCm")
                print(f"  - Model: {model_details.get('model_name')}")
                print(f"  - Device: {model_details.get('model_device')}")
                print(f"  - Half precision: {'Yes' if model_details.get('half_precision_supported') else 'No'}")
                print(f"  - Output shape: {model_details.get('output_shape')}")
                print(f"  - Duration: {model_details.get('duration', 0):.2f} seconds")
            else:
                print(f"❌ Failed to run BERT on ROCm: {model_details.get('error')}")
        
        return 0
    else:
        print("❌ ROCm environment is not available")
        
        if "cuda_available" in env_details and not env_details["cuda_available"]:
            print("  - CUDA/ROCm is not available")
        elif "devices" in env_details:
            print("  - No AMD GPUs detected in the CUDA devices")
            for device in env_details["devices"]:
                print(f"  - GPU {device['index']}: {device['name']} ({device['total_memory_gb']} GB)")
        
        if "error" in env_details:
            print(f"  - Error: {env_details['error']}")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())