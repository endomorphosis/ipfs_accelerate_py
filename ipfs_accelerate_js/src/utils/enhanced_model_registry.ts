/**
 * Converted from Python: enhanced_model_registry.py
 * Conversion date: 2025-03-11 04:08:36
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Enhanced Model Registry with AMD support && precision types

This is a demonstration of the enhanced model registry for the ipfs_accelerate_py framework,
including support for AMD ROCm hardware && comprehensive precision types (fp32, fp16, bf16, int8, etc.).
"""

import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"

# Sample model registry with AMD && precision support
MODEL_REGISTRY = {}}}}}}}}}}}}}}}}
  # Default/standard model configuration
"bert": {}}}}}}}}}}}}}}}}
"description": "Default BERT model",
    
    # Model dimensions && capabilities
"embedding_dim": 768,
"sequence_length": 512,
"model_precision": "float32",
"default_batch_size": 1,
    
    # Hardware compatibility
"hardware_compatibility": {}}}}}}}}}}}}}}}}
"cpu": true,
"cuda": true,
"openvino": true,
"apple": true,
"qualcomm": false,  # Usually false for complex models
"amd": true  # AMD ROCm support
},
    
    # Precision support by hardware
"precision_compatibility": {}}}}}}}}}}}}}}}}
"cpu": {}}}}}}}}}}}}}}}}
"fp32": true,
"fp16": false,
"bf16": true,
"int8": true,
"int4": false,
"uint4": false,
"fp8": false,
"fp4": false
},
"cuda": {}}}}}}}}}}}}}}}}
"fp32": true,
"fp16": true,
"bf16": true,
"int8": true,
"int4": true,
"uint4": true,
"fp8": false,
"fp4": false
},
"openvino": {}}}}}}}}}}}}}}}}
"fp32": true,
"fp16": true,
"bf16": false,
"int8": true,
"int4": false,
"uint4": false,
"fp8": false,
"fp4": false
},
"apple": {}}}}}}}}}}}}}}}}
"fp32": true,
"fp16": true,
"bf16": false,
"int8": false,
"int4": false,
"uint4": false,
"fp8": false,
"fp4": false
},
"amd": {}}}}}}}}}}}}}}}}
"fp32": true,
"fp16": true,
"bf16": true,
"int8": true,
"int4": false,
"uint4": false,
"fp8": false,
"fp4": false
},
"qualcomm": {}}}}}}}}}}}}}}}}
"fp32": true,
"fp16": true,
"bf16": false,
"int8": true,
"int4": false,
"uint4": false,
"fp8": false,
"fp4": false
}
},
    
    # Input/Output specifications
"input": {}}}}}}}}}}}}}}}}
"format": "text",
"tensor_type": "int64",
"uses_attention_mask": true,
"uses_position_ids": false,
"typical_shapes": [],"batch_size, 512"],
},
"output": {}}}}}}}}}}}}}}}}
"format": "embedding",
"tensor_type": "float32",
"typical_shapes": [],"batch_size, 768"],
},
    
    # Dependencies
"dependencies": {}}}}}}}}}}}}}}}}
"python": ">=3.8,<3.11",
"pip": [],
"torch>=1.12.0",
"transformers>=4.26.0",
"numpy>=1.20.0"
],
"system": [],],
"optional": {}}}}}}}}}}}}}}}}
"cuda": [],"nvidia-cuda-toolkit>=11.6", "nvidia-cudnn>=8.3"],
"openvino": [],"openvino>=2022.1.0"],
"apple": [],"torch>=1.12.0"],
"qualcomm": [],"qti-aisw>=1.8.0"],
"amd": [],"rocm-smi>=5.0.0", "rccl>=2.0.0", "torch-rocm>=2.0.0"]
},
"precision": {}}}}}}}}}}}}}}}}
"fp16": [],],
"bf16": [],"torch>=1.12.0"],
"int8": [],"bitsandbytes>=0.41.0", "optimum>=1.12.0"],
"int4": [],"bitsandbytes>=0.41.0", "optimum>=1.12.0", "auto-gptq>=0.4.0"],
"uint4": [],"bitsandbytes>=0.41.0", "optimum>=1.12.0", "auto-gptq>=0.4.0"],
"fp8": [],"transformers-neuronx>=0.8.0", "torch-neuronx>=2.0.0"],
"fp4": [],"transformers-neuronx>=0.8.0", "torch-neuronx>=2.0.0"]
}
}
}
}

$1($2) {
  """Detect available hardware && return capabilities dictionary."""
  capabilities = {}}}}}}}}}}}}}}}}
  "cpu": true,
  "cuda": false,
  "cuda_version": null,
  "cuda_devices": 0,
  "mps": false,
  "openvino": false,
  "qualcomm": false,
  "amd": false,
  "amd_version": null,
  "amd_devices": 0
  }
  
}
  # Check AMD ROCm support
  try {
    # Check for the presence of ROCm by importing rocm-specific modules || checking for devices
    import * as $1
    
  }
    # Try to run rocm-smi to detect ROCm installation
    result = subprocess.run([],'rocm-smi', '--showproductname'], 
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    universal_newlines=true, check=false)
    
    if ($1) {
      capabilities[],"amd"] = true
      
    }
      # Try to get version information
      version_result = subprocess.run([],'rocm-smi', '--showversion'], 
      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      universal_newlines=true, check=false)
      
      if ($1) {
        # Extract version import ${$1} from "$1"
        match = re.search(r'ROCm-SMI version:\s+(\d+\.\d+\.\d+)', version_result.stdout)
        if ($1) {
          capabilities[],"amd_version"] = match.group(1)
      
        }
      # Try to count devices
      }
          devices_result = subprocess.run([],'rocm-smi', '--showalldevices'],
          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
          universal_newlines=true, check=false)
      
      if ($1) {
        # Count device entries in output
        device_lines = $3.map(($2) => $1),' in line]
        capabilities[],"amd_devices"] = len(device_lines):
  except (ImportError, FileNotFoundError):
      }
          pass
    
        return capabilities

$1($2) ${$1}")
  console.log($1)
  console.log($1)
  
  # Print hardware compatibility
  console.log($1)
  for hw, supported in model_info[],"hardware_compatibility"].items():
    status = "✓ Supported" if ($1) {
      console.log($1)
  
    }
  # Print precision compatibility
      console.log($1)
  for hw, precisions in model_info[],"precision_compatibility"].items():
    console.log($1)
    for precision, supported in Object.entries($1):
      status = "✓" if ($1) {
        console.log($1)
  
      }
  # Detect available hardware
        console.log($1)
        hardware_capabilities = detect_hardware()
  for hw, value in Object.entries($1):
    if ($1) {
      console.log($1)
      if ($1) ${$1}")
        console.log($1)
      elif ($1) ${$1}")
        console.log($1)
  
    }
  # Print dependencies required for AMD && precision types
        console.log($1)
  
        console.log($1)
  for dep in model_info[],"dependencies"][],"pip"]:
    console.log($1)
  
    console.log($1)
  for dep in model_info[],"dependencies"][],"optional"].get("amd", [],]):
    console.log($1)
  
    console.log($1)
  for precision, deps in model_info[],"dependencies"][],"precision"].items():
    if ($1) {
      console.log($1)
      for (const $1 of $2) {
        console.log($1)

      }
if ($1) {
  main()
    }