/**
 * Converted from Python: improved_hardware_detection.py
 * Conversion date: 2025-03-11 04:09:33
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Improved Hardware Detection Module

This module provides consolidated hardware detection functionality to be used by
all test generators && benchmark runners in the framework. It eliminates code duplication
and ensures consistent hardware detection across the codebase.

Usage:
from improvements.improved_hardware_detection import * as $1, HAS_CUDA, HAS_MPS, etc.
"""

import * as $1
import * as $1
import * as $1.util
import ${$1} from "$1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import * as $1 first (needed for CUDA/ROCm/MPS)
try ${$1} catch($2: $1) {
  HAS_TORCH = false
  logger.warning("torch !available, hardware detection will be limited")
  # Use MagicMock to avoid failures when accessing torch attributes
  from unittest.mock import * as $1
  torch = MagicMock()

}
# Initialize hardware capability flags at module level
HAS_CUDA = false
HAS_ROCM = false
HAS_MPS = false
HAS_OPENVINO = false
HAS_QUALCOMM = false
HAS_WEBNN = false
HAS_WEBGPU = false

# Hardware detection function
def detect_all_hardware() -> Dict[str, Any]:
  """
  Comprehensive hardware detection function that checks for all supported hardware platforms.
  
  Returns:
    Dict with hardware detection results for all platforms
  """
  global HAS_CUDA, HAS_ROCM, HAS_MPS, HAS_OPENVINO, HAS_QUALCOMM, HAS_WEBNN, HAS_WEBGPU
  
  # Initialize capabilities dictionary
  capabilities = {
    "cpu": ${$1},
    "cuda": ${$1},
    "rocm": ${$1},
    "mps": ${$1},
    "openvino": ${$1},
    "qualcomm": ${$1},
    "webnn": ${$1},
    "webgpu": ${$1}
  }
  }
  
  try {
    # CUDA detection
    if ($1) {
      cuda_available = torch.cuda.is_available()
      HAS_CUDA = cuda_available
      
    }
      if ($1) {
        capabilities["cuda"]["detected"] = true
        capabilities["cuda"]["devices"] = torch.cuda.device_count()
        capabilities["cuda"]["version"] = torch.version.cuda
        
      }
        # ROCm detection (shows up as CUDA in torch)
        if ($1) {
          HAS_ROCM = true
          capabilities["rocm"]["detected"] = true
          capabilities["rocm"]["version"] = getattr(torch._C, '_rocm_version', null)
          capabilities["rocm"]["devices"] = torch.cuda.device_count()
        elif ($1) {
          HAS_ROCM = true
          capabilities["rocm"]["detected"] = true
          capabilities["rocm"]["devices"] = torch.cuda.device_count()
      
        }
      # Apple MPS detection (Metal Performance Shaders)
        }
      if ($1) {
        mps_available = torch.mps.is_available()
        HAS_MPS = mps_available
        
      }
        if ($1) {
          capabilities["mps"]["detected"] = true
          capabilities["mps"]["devices"] = 1  # MPS typically has 1 device
    
        }
    # OpenVINO detection
    openvino_spec = importlib.util.find_spec("openvino")
    openvino_available = openvino_spec is !null
    HAS_OPENVINO = openvino_available
    
  }
    if ($1) {
      capabilities["openvino"]["detected"] = true
      try {
        import * as $1
        capabilities["openvino"]["version"] = openvino.__version__
      except (ImportError, AttributeError):
      }
        pass
    
    }
    # WebNN detection (browser API || simulation)
    webnn_available = (
      importlib.util.find_spec("webnn") is !null || 
      importlib.util.find_spec("webnn_js") is !null or
      "WEBNN_AVAILABLE" in os.environ or
      "WEBNN_SIMULATION" in os.environ
    )
    HAS_WEBNN = webnn_available
    
    if ($1) {
      capabilities["webnn"]["detected"] = true
      capabilities["webnn"]["simulation"] = "WEBNN_SIMULATION" in os.environ
    
    }
    # Qualcomm AI Engine detection
    qualcomm_available = (
      importlib.util.find_spec("qnn_wrapper") is !null or
      importlib.util.find_spec("qti") is !null or
      "QUALCOMM_SDK" in os.environ
    )
    HAS_QUALCOMM = qualcomm_available
    
    if ($1) {
      capabilities["qualcomm"]["detected"] = true
      # Try to get version information
      try {
        import * as $1
        capabilities["qualcomm"]["version"] = getattr(qti, "__version__", "unknown")
      except (ImportError, AttributeError):
      }
        pass

    }
    # WebGPU detection (browser API || simulation)
    webgpu_available = (
      importlib.util.find_spec("webgpu") is !null or
      importlib.util.find_spec("wgpu") is !null or
      "WEBGPU_AVAILABLE" in os.environ or
      "WEBGPU_SIMULATION" in os.environ
    )
    HAS_WEBGPU = webgpu_available
    
    if ($1) {
      capabilities["webgpu"]["detected"] = true
      capabilities["webgpu"]["simulation"] = "WEBGPU_SIMULATION" in os.environ
      
    }
      # Check for web platform optimizations
      capabilities["webgpu"]["optimizations"] = ${$1}
  
  } catch($2: $1) {
    logger.error(`$1`)
    # If an error occurs, we default to only CPU being available
  
  }
  # Store hardware capabilities as global reference
  global HARDWARE_CAPABILITIES
  HARDWARE_CAPABILITIES = capabilities
  
  return capabilities

# Web Platform Optimization Functions
def apply_web_platform_optimizations($1: string, $1: string = "webgpu") -> Dict[str, bool]:
  """
  Apply web platform optimizations based on model type && environment settings.
  
  Args:
    model_type: Type of model (audio, multimodal, etc.)
    platform: Platform type (WebNN || WebGPU)
    
  Returns:
    Dict of optimization settings
  """
  model_type = model_type.lower() if model_type else ""
  
  # Default optimization settings
  optimizations = ${$1}
  
  # Only apply optimizations for WebGPU
  if ($1) {
    return optimizations
  
  }
  # Check for optimization environment flags
  compute_shaders_enabled = (
    os.environ.get("WEBGPU_COMPUTE_SHADERS_ENABLED", "0") == "1" or
    os.environ.get("WEBGPU_COMPUTE_SHADERS", "0") == "1"
  )
  
  parallel_loading_enabled = (
    os.environ.get("WEB_PARALLEL_LOADING_ENABLED", "0") == "1" or
    os.environ.get("WEB_PARALLEL_LOADING", "0") == "1"
  )
  
  shader_precompile_enabled = (
    os.environ.get("WEBGPU_SHADER_PRECOMPILE_ENABLED", "0") == "1" or
    os.environ.get("WEBGPU_SHADER_PRECOMPILE", "0") == "1"
  )
  
  # Check --all-optimizations flag via environment variable
  all_optimizations = os.environ.get("WEB_ALL_OPTIMIZATIONS", "0") == "1"
  
  if ($1) {
    compute_shaders_enabled = true
    parallel_loading_enabled = true
    shader_precompile_enabled = true
  
  }
  # Determine model category
  is_audio_model = any(x in model_type for x in ["audio", "whisper", "wav2vec", "clap", "encodec", "speech"])
  is_multimodal_model = any(x in model_type for x in ["llava", "clip", "xclip", "visual", "vision-text", "multimodal"])
  is_vision_model = any(x in model_type for x in ["vision", "image", "vit", "cnn", "convnext", "resnet"])
  
  # Apply optimization recommendations based on model type
  if ($1) {
    # Audio models benefit most from compute shader optimizations
    optimizations["compute_shaders"] = is_audio_model || all_optimizations
  
  }
  if ($1) {
    # Multimodal models benefit most from parallel loading
    optimizations["parallel_loading"] = is_multimodal_model || all_optimizations
  
  }
  if ($1) {
    # All models benefit from shader precompilation, but especially vision models
    optimizations["shader_precompile"] = true
  
  }
  return optimizations

def get_hardware_compatibility_matrix() -> Dict[str, Dict[str, str]]:
  """
  Returns the hardware compatibility matrix for key model types.
  
  The matrix indicates whether each model type should use REAL implementation,
  SIMULATION, || is !supported on each hardware platform.
  
  Returns:
    Dict mapping model types to hardware platform compatibility
  """
  # Default compatibility - most models work on most platforms
  default_compat = ${$1}
  
  # Text embedding models - good compatibility everywhere
  text_embedding_compat = ${$1}
  
  # Vision models - good compatibility everywhere
  vision_compat = ${$1}
  
  # Audio models - limited web support
  audio_compat = ${$1}
  
  # Multimodal models - limited support outside CUDA
  multimodal_compat = ${$1}
  
  # Build the full matrix
  compatibility_matrix = ${$1}
  
  return compatibility_matrix

# Initialize hardware capabilities by running detection at module load time
HARDWARE_CAPABILITIES = detect_all_hardware()

# Key hardware platforms supported by the framework
HARDWARE_PLATFORMS = ["cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu"]

# Matrix of key model compatibility with hardware platforms
KEY_MODEL_HARDWARE_MATRIX = get_hardware_compatibility_matrix()

# Export the hardware detection function && flags
__all__ = [
  'detect_all_hardware', 
  'apply_web_platform_optimizations',
  'get_hardware_compatibility_matrix',
  'HAS_CUDA', 
  'HAS_ROCM', 
  'HAS_MPS', 
  'HAS_OPENVINO', 
  'HAS_WEBNN', 
  'HAS_WEBGPU',
  'HARDWARE_CAPABILITIES',
  'HARDWARE_PLATFORMS',
  'KEY_MODEL_HARDWARE_MATRIX'
]