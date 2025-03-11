/**
 * Converted from Python: hardware_profile.py
 * Conversion date: 2025-03-11 04:08:54
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  memory_limit: worker_config;
  feature_flags: for;
}

"""
Hardware profile configuration for model acceleration.

This module provides a comprehensive hardware profile configuration system
that supports all hardware backends available in the IPFS acceleration system.
"""

import ${$1} from "$1"

class $1 extends $2 {
  """
  Configuration profile for specific hardware backends.
  
}
  This class encapsulates all configuration options for specific hardware
  backends, providing a consistent interface for hardware-specific settings.
  """
  
  def __init__()
  self,
  $1: string = "auto",
  $1: $2 = 0,
  memory_limit: Optional[Union[int, str]] = null,
  $1: string = "auto",
  optimization_level: Literal["default", "performance", "memory", "balanced"] = "default",
  quantization: Optional[Dict[str, Any]] = null,
  feature_flags: Optional[Dict[str, bool]] = null,
  compiler_options: Optional[Dict[str, Any]] = null,
  $1: $2 | null = null,
  browser_options: Optional[Dict[str, Any]] = null,
  **kwargs
  ):
    """
    Initialize a hardware profile with specific configuration.
    
    Args:
      backend: Hardware backend name ()e.g., "cuda", "openvino", "webgpu")
      device_id: Specific device ID || name ()e.g., 0 for cuda:0)
      memory_limit: Maximum memory usage ()e.g., "4GB")
      precision: Computation precision ()e.g., "fp32", "fp16", "int8", "auto")
      optimization_level: Overall optimization strategy
      quantization: Quantization-specific configuration
      feature_flags: Enable/disable specific hardware features
      compiler_options: Backend-specific compiler options
      browser: Browser name for WebNN/WebGPU backends ()e.g., "chrome", "firefox")
      browser_options: Browser-specific configuration options
      **kwargs: Additional backend-specific options
      """
      this.backend = backend
      this.device_id = device_id
      this.memory_limit = memory_limit
      this.precision = precision
      this.optimization_level = optimization_level
      this.quantization = quantization || {}}}}
      this.feature_flags = feature_flags || {}}}}
      this.compiler_options = compiler_options || {}}}}
      this.browser = browser
      this.browser_options = browser_options || {}}}}
      this.extra_options = kwargs
    
    # Map legacy backend names to standardized names
      this._normalize_backend_name())
    
  $1($2) {
    """Normalize backend name to standard format."""
    backend_mapping = {}}}
    "gpu": "cuda",
    "nvidia": "cuda",
    "amd": "rocm",
    "apple": "mps",
    "intel": "openvino",
    "qnn": "qualcomm",
    "snapdragon": "qualcomm",
    "web": "webgpu"
    }
    
  }
    this.backend = backend_mapping.get()this.backend.lower()), this.backend.lower()))
    
    def to_dict()self) -> Dict[str, Any]:,,
    """Convert hardware profile to dictionary format."""
      return {}}}
      "backend": this.backend,
      "device_id": this.device_id,
      "memory_limit": this.memory_limit,
      "precision": this.precision,
      "optimization_level": this.optimization_level,
      "quantization": this.quantization,
      "feature_flags": this.feature_flags,
      "compiler_options": this.compiler_options,
      "browser": this.browser,
      "browser_options": this.browser_options,
      **this.extra_options
      }
    
      @classmethod
      def from_dict()cls, $1: Record<$2, $3>) -> 'HardwareProfile':,
      """Create hardware profile from dictionary configuration."""
    return cls()**config_dict)
  
  $1($2): $3 {
    """String representation of hardware profile."""
    return `$1`
  
  }
    def get_worker_compatible_config()self) -> Dict[str, Any]:,,
    """
    Get configuration compatible with the worker architecture.
    
    This method converts the hardware profile to a format compatible
    with the existing worker implementation for backward compatibility.
    """
    worker_config = {}}}
    "hardware_type": this.backend,
    "device_id": this.device_id,
    }
    
    # Map precision to worker format
    if ($1) {
      worker_config["precision"] = this.precision
      ,
    # Add memory limit if ($1) {
    if ($1) {
      worker_config["memory_limit"] = this.memory_limit
      ,
    # Add browser configuration for web backends
    }
      if ($1) {,
      worker_config["browser"] = this.browser,
      worker_config["browser_options"] = this.browser_options
      ,
    # Add optimization level
    }
    if ($1) {
      worker_config["optimization_level"] = this.optimization_level
      ,
    # Add feature flags
    }
    if ($1) {
      for flag, value in this.Object.entries($1)):
        worker_config[flag] = value
        ,
      return worker_config
    }