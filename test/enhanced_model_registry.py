#!/usr/bin/env python3
"""
Enhanced Model Registry with AMD support and precision types

This is a demonstration of the enhanced model registry for the ipfs_accelerate_py framework,
including support for AMD ROCm hardware and comprehensive precision types (fp32, fp16, bf16, int8, etc.).
"""

import os
import sys
import json
import datetime
from pathlib import Path

# Sample model registry with AMD and precision support
MODEL_REGISTRY = {
    # Default/standard model configuration
    "bert": {
        "description": "Default BERT model",
        
        # Model dimensions and capabilities
        "embedding_dim": 768,
        "sequence_length": 512,
        "model_precision": "float32", 
        "default_batch_size": 1,
        
        # Hardware compatibility
        "hardware_compatibility": {
            "cpu": True,
            "cuda": True,
            "openvino": True,
            "apple": True,
            "qualcomm": False,  # Usually false for complex models
            "amd": True  # AMD ROCm support
        },
        
        # Precision support by hardware
        "precision_compatibility": {
            "cpu": {
                "fp32": True,
                "fp16": False,
                "bf16": True,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "cuda": {
                "fp32": True,
                "fp16": True,
                "bf16": True,
                "int8": True,
                "int4": True,
                "uint4": True,
                "fp8": False,
                "fp4": False
            },
            "openvino": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "apple": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": False,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "amd": {
                "fp32": True,
                "fp16": True,
                "bf16": True,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "qualcomm": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }
        },
        
        # Input/Output specifications
        "input": {
            "format": "text",
            "tensor_type": "int64",
            "uses_attention_mask": True,
            "uses_position_ids": False,
            "typical_shapes": ["batch_size, 512"]
        },
        "output": {
            "format": "embedding",
            "tensor_type": "float32",
            "typical_shapes": ["batch_size, 768"]
        },
        
        # Dependencies
        "dependencies": {
            "python": ">=3.8,<3.11",
            "pip": [
                "torch>=1.12.0",
                "transformers>=4.26.0",
                "numpy>=1.20.0"
            ],
            "system": [],
            "optional": {
                "cuda": ["nvidia-cuda-toolkit>=11.6", "nvidia-cudnn>=8.3"],
                "openvino": ["openvino>=2022.1.0"],
                "apple": ["torch>=1.12.0"],
                "qualcomm": ["qti-aisw>=1.8.0"],
                "amd": ["rocm-smi>=5.0.0", "rccl>=2.0.0", "torch-rocm>=2.0.0"]
            },
            "precision": {
                "fp16": [],
                "bf16": ["torch>=1.12.0"],
                "int8": ["bitsandbytes>=0.41.0", "optimum>=1.12.0"],
                "int4": ["bitsandbytes>=0.41.0", "optimum>=1.12.0", "auto-gptq>=0.4.0"],
                "uint4": ["bitsandbytes>=0.41.0", "optimum>=1.12.0", "auto-gptq>=0.4.0"],
                "fp8": ["transformers-neuronx>=0.8.0", "torch-neuronx>=2.0.0"],
                "fp4": ["transformers-neuronx>=0.8.0", "torch-neuronx>=2.0.0"]
            }
        }
    }
}

def detect_hardware():
    """Detect available hardware and return capabilities dictionary."""
    capabilities = {
        "cpu": True,
        "cuda": False,
        "cuda_version": None,
        "cuda_devices": 0,
        "mps": False,
        "openvino": False,
        "qualcomm": False,
        "amd": False,
        "amd_version": None,
        "amd_devices": 0
    }
    
    # Check AMD ROCm support
    try:
        # Check for the presence of ROCm by importing rocm-specific modules or checking for devices
        import subprocess
        
        # Try to run rocm-smi to detect ROCm installation
        result = subprocess.run(['rocm-smi', '--showproductname'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              universal_newlines=True, check=False)
        
        if result.returncode == 0:
            capabilities["amd"] = True
            
            # Try to get version information
            version_result = subprocess.run(['rocm-smi', '--showversion'], 
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                         universal_newlines=True, check=False)
            
            if version_result.returncode == 0:
                # Extract version from output
                import re
                match = re.search(r'ROCm-SMI version:\s+(\d+\.\d+\.\d+)', version_result.stdout)
                if match:
                    capabilities["amd_version"] = match.group(1)
            
            # Try to count devices
            devices_result = subprocess.run(['rocm-smi', '--showalldevices'], 
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                         universal_newlines=True, check=False)
            
            if devices_result.returncode == 0:
                # Count device entries in output
                device_lines = [line for line in devices_result.stdout.split('\n') if 'GPU[' in line]
                capabilities["amd_devices"] = len(device_lines)
    except (ImportError, FileNotFoundError):
        pass
        
    return capabilities

def main():
    """Main function to demonstrate model registry."""
    print("Enhanced Model Registry with AMD support and precision types")
    print("-" * 50)
    
    # Print model information
    model_type = "bert"
    model_info = MODEL_REGISTRY[model_type]
    
    print(f"Model: {model_type}")
    print(f"Description: {model_info['description']}")
    print(f"Embedding Dimension: {model_info['embedding_dim']}")
    print(f"Sequence Length: {model_info['sequence_length']}")
    
    # Print hardware compatibility
    print("\nHardware Compatibility:")
    for hw, supported in model_info["hardware_compatibility"].items():
        status = "✓ Supported" if supported else "✗ Not Supported"
        print(f" - {hw.upper()}: {status}")
    
    # Print precision compatibility
    print("\nPrecision Support by Hardware:")
    for hw, precisions in model_info["precision_compatibility"].items():
        print(f" - {hw.upper()}:")
        for precision, supported in precisions.items():
            status = "✓" if supported else "✗"
            print(f"   - {precision.upper()}: {status}")
    
    # Detect available hardware
    print("\nDetected Hardware:")
    hardware_capabilities = detect_hardware()
    for hw, value in hardware_capabilities.items():
        if value and hw not in ["cuda_version", "cuda_devices", "amd_version", "amd_devices"]:
            print(f" - {hw.upper()}: Available")
            if hw == "cuda" and hardware_capabilities["cuda_devices"] > 0:
                print(f"   - Devices: {hardware_capabilities['cuda_devices']}")
                print(f"   - Version: {hardware_capabilities['cuda_version']}")
            elif hw == "amd" and hardware_capabilities["amd_devices"] > 0:
                print(f"   - Devices: {hardware_capabilities['amd_devices']}")
                print(f"   - Version: {hardware_capabilities['amd_version']}")
    
    # Print dependencies required for AMD and precision types
    print("\nDependencies:")
    
    print(" - Required:")
    for dep in model_info["dependencies"]["pip"]:
        print(f"   - {dep}")
    
    print(" - AMD-specific:")
    for dep in model_info["dependencies"]["optional"].get("amd", []):
        print(f"   - {dep}")
    
    print(" - Precision-specific:")
    for precision, deps in model_info["dependencies"]["precision"].items():
        if deps:
            print(f"   - {precision.upper()}:")
            for dep in deps:
                print(f"     - {dep}")

if __name__ == "__main__":
    main()