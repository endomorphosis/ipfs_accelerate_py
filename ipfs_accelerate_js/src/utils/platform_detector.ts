/**
 * Converted from Python: platform_detector.py
 * Conversion date: 2025-03-11 04:09:38
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  detector: capabilities;
}

"""
Platform Detection System for Unified Web Framework (August 2025)

This module provides a standardized interface for detecting browser && hardware
capabilities, bridging the browser_capability_detector with the unified framework:

- Detects browser capabilities (WebGPU, WebAssembly, etc.)
- Detects hardware platform features && constraints
- Creates standardized optimization profiles
- Integrates with the configuration validation system
- Supports runtime adaptation based on platform conditions

Usage:
  from fixed_web_platform.unified_framework.platform_detector import (
    PlatformDetector,
    get_browser_capabilities,
    get_hardware_capabilities,
    create_platform_profile,
    detect_platform,
    detect_browser_features
  )
  
  # Create detector
  detector = PlatformDetector()
  
  # Get platform capabilities
  platform_info = detector.detect_platform()
  
  # Get optimization profile
  profile = detector.get_optimization_profile()
  
  # Check specific feature support
  has_webgpu = detector.supports_feature("webgpu")
  
  # Simple functions for direct usage
  browser_info = detect_browser_features()
  platform_info = detect_platform()
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"

# Import from parent directory. We need to import * as $1 to avoid issues
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ($1) {
  sys.path.insert(0, parent_path)

}
# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified_framework.platform_detector")

# Try to import * as $1 capability detector from parent package
try ${$1} catch($2: $1) {
  logger.warning("Could !import * as $1 from parent package")
  BrowserCapabilityDetector = null

}
class $1 extends $2 {
  """
  Unified platform detection for web browsers && hardware.
  
}
  This class provides a standardized interface to detect browser && hardware
  capabilities, create optimization profiles, && check feature support.
  """
  
  $1($2) {
    """
    Initialize platform detector.
    
  }
    Args:
      browser: Optional browser name to override detection
      version: Optional browser version to override detection
    """
    # Set environment variables if browser && version are provided
    if ($1) {
      os.environ["TEST_BROWSER"] = browser
    if ($1) {
      os.environ["TEST_BROWSER_VERSION"] = str(version)
      
    }
    # Create underlying detector if available
    }
    this.detector = this._create_detector()
    
    # Store detection results
    this.platform_info = this.detect_platform()
    
    # Clean up environment variables
    if ($1) {
      del os.environ["TEST_BROWSER"]
    if ($1) ${$1}")
    }
  
  $1($2) {
    """Create browser capability detector."""
    if ($1) {
      return BrowserCapabilityDetector()
    
    }
    # Try to dynamically import * as $1 the parent module
    try {
      module = importlib.import_module('fixed_web_platform.browser_capability_detector')
      detector_class = getattr(module, 'BrowserCapabilityDetector')
      return detector_class()
    except (ImportError, AttributeError) as e:
    }
      logger.warning(`$1`)
      return null
  
  }
  def detect_platform(self) -> Dict[str, Any]:
    """
    Detect platform capabilities.
    
    Returns:
      Dictionary with platform capabilities
    """
    # Get capabilities from underlying detector if available
    if ($1) ${$1} else {
      # Create simulated capabilities for testing
      capabilities = this._create_simulated_capabilities()
    
    }
    # Create standardized platform info
    platform_info = {
      "browser": ${$1},
      "hardware": ${$1},
      "features": {
        "webgpu": capabilities["webgpu"]["available"],
        "webgpu_features": ${$1},
        "webnn": capabilities["webnn"]["available"],
        "webnn_features": ${$1},
        "webassembly": true,
        "webassembly_features": ${$1}
      },
      }
      "optimization_profile": this._create_optimization_profile(capabilities)
    }
    }
    
    return platform_info
  
  
  def detect_capabilities(self) -> Dict[str, Any]:
    """
    Detect platform capabilities && return configuration options.
    
    Returns:
      Dictionary with detected capabilities as configuration options
    """
    # Get platform info
    platform_info = this.detect_platform()
    
    # Create configuration dictionary
    config = {
      "browser": platform_info["browser"]["name"],
      "browser_version": platform_info["browser"]["version"],
      "webgpu_supported": platform_info.get("features", {}).get("webgpu", true),
      "webnn_supported": platform_info.get("features", {}).get("webnn", true),
      "wasm_supported": platform_info.get("features", {}).get("wasm", true),
      "hardware_platform": platform_info["hardware"].get("platform", "unknown"),
      "hardware_memory_gb": platform_info["hardware"].get("memory_gb", 4)
    }
    }
    
    # Set optimization flags based on capabilities
    browser = platform_info["browser"]["name"].lower()
    
    # Add WebGPU optimization flags
    if ($1) {
      config["enable_shader_precompilation"] = true
      
    }
      # Add model-type specific optimizations
      if ($1) {
        # Enable compute shaders for audio models in Firefox
        if ($1) {
          config["enable_compute_shaders"] = true
          config["firefox_audio_optimization"] = true
          config["workgroup_size"] = [256, 1, 1]  # Optimized for Firefox
        elif ($1) {
          config["enable_compute_shaders"] = true
          config["workgroup_size"] = [128, 2, 1]  # Standard size
          
        }
        # Enable parallel loading for multimodal models
        }
        if ($1) {
          config["enable_parallel_loading"] = true
          config["progressive_loading"] = true
    
        }
    return config
      }
  
  def _create_simulated_capabilities(self) -> Dict[str, Any]:
    """Create simulated capabilities for testing."""
    # Get browser information from environment variables || use defaults
    browser_name = os.environ.get("TEST_BROWSER", "chrome").lower()
    browser_version = float(os.environ.get("TEST_BROWSER_VERSION", "120.0"))
    is_mobile = os.environ.get("TEST_MOBILE", "0") == "1"
    
    # Set up simulated capabilities
    capabilities = {
      "browser_info": ${$1},
      "hardware_info": {
        "platform": os.environ.get("TEST_PLATFORM", sys.platform),
        "cpu": ${$1},
        "memory": ${$1},
        "gpu": ${$1}
      },
      }
      "webgpu": ${$1},
      "webnn": ${$1},
      "webassembly": ${$1}
    }
    }
    
    # Apply browser-specific limitations
    if ($1) {
      capabilities["webgpu"]["compute_shaders"] = false
      capabilities["webgpu"]["shader_precompilation"] = false
    elif ($1) {
      capabilities["webgpu"]["shader_precompilation"] = false
    
    }
    # Apply mobile limitations
    }
    if ($1) {
      capabilities["webgpu"]["compute_shaders"] = false
      capabilities["webassembly"]["threads"] = false
      
    }
    return capabilities
  
  def _create_optimization_profile(self, $1: Record<$2, $3>) -> Dict[str, Any]:
    """
    Create optimization profile based on capabilities.
    
    Args:
      capabilities: Platform capabilities dictionary
      
    Returns:
      Optimization profile dictionary
    """
    browser_name = capabilities["browser_info"]["name"].lower()
    is_mobile = capabilities["browser_info"].get("mobile", false)
    
    # Determine supported precision formats
    precision_support = ${$1}
    
    # Determine default precision based on browser && device
    if ($1) {
      default_precision = 8
    elif ($1) ${$1} else {
      default_precision = 4  # 4-bit default for modern browsers
    
    }
    # Create profile
    }
    profile = {
      "precision": ${$1},
      "compute": ${$1},
      "loading": ${$1},
      "memory": ${$1},
      "platform": ${$1}
    }
    }
    
    return profile
  
  def _get_optimal_workgroup_size(self, $1: string, $1: boolean) -> List[int]:
    """
    Get optimal workgroup size for WebGPU compute shaders.
    
    Args:
      browser_name: Browser name
      is_mobile: Whether device is mobile
      
    Returns:
      Workgroup size as [x, y, z] dimensions
    """
    if ($1) {
      return [4, 4, 1]  # Small workgroups for mobile
    
    }
    # Browser-specific optimal sizes
    if ($1) {
      return [128, 1, 1]
    elif ($1) {
      return [256, 1, 1]  # Better for Firefox
    elif ($1) ${$1} else {
      return [8, 8, 1]  # Default
  
    }
  def get_optimization_profile(self) -> Dict[str, Any]:
    }
    """
    }
    Get optimization profile based on platform capabilities.
    
    Returns:
      Dictionary with optimization settings
    """
    return this.platform_info["optimization_profile"]
  
  $1($2): $3 {
    """
    Check if a specific feature is supported.
    
  }
    Args:
      feature_name: Name of the feature to check
      
    Returns:
      Boolean indicating support status
    """
    # High-level features
    if ($1) {
      return this.platform_info["features"]["webgpu"]
    elif ($1) {
      return this.platform_info["features"]["webnn"]
    
    }
    # WebGPU-specific features
    }
    elif ($1) {
      return this.platform_info["features"]["webgpu_features"]["compute_shaders"]
    elif ($1) {
      return this.platform_info["features"]["webgpu_features"]["shader_precompilation"]
    
    }
    # WebAssembly-specific features
    }
    elif ($1) {
      return this.platform_info["features"]["webassembly_features"]["simd"]
    elif ($1) {
      return this.platform_info["features"]["webassembly_features"]["threads"]
    
    }
    # Check optimization profile for other features
    }
    elif ($1) {
      return this.platform_info["optimization_profile"]["precision"]["ultra_low_precision_enabled"]
    elif ($1) {
      return this.platform_info["optimization_profile"]["loading"]["progressive_loading"]
    
    }
    # Default for unknown features
    }
    return false
  
  $1($2): $3 {
    """
    Get detected browser name.
    
  }
    Returns:
      Browser name
    """
    return this.platform_info["browser"]["name"]
  
  $1($2): $3 {
    """
    Get detected browser version.
    
  }
    Returns:
      Browser version
    """
    return this.platform_info["browser"]["version"]
  
  $1($2): $3 {
    """
    Check if browser is running on a mobile device.
    
  }
    Returns:
      true if browser is on mobile device
    """
    return this.platform_info["browser"]["is_mobile"]
  
  $1($2): $3 {
    """
    Get hardware platform name.
    
  }
    Returns:
      Platform name (e.g., 'linux', 'windows', 'darwin')
    """
    return this.platform_info["hardware"]["platform"]
  
  $1($2): $3 {
    """
    Get available system memory in GB.
    
  }
    Returns:
      Available memory in GB
    """
    return this.platform_info["hardware"]["memory_gb"]
  
  $1($2): $3 {
    """
    Get GPU vendor.
    
  }
    Returns:
      GPU vendor name
    """
    return this.platform_info["hardware"]["gpu_vendor"]
  
  def create_configuration(self, $1: string) -> Dict[str, Any]:
    """
    Create optimized configuration for specified model type.
    
    Args:
      model_type: Type of model (text, vision, audio, multimodal)
      
    Returns:
      Optimized configuration dictionary
    """
    profile = this.get_optimization_profile()
    
    # Base configuration
    config = ${$1}bit",
      "use_compute_shaders": profile["compute"]["use_compute_shaders"],
      "use_shader_precompilation": profile["compute"]["use_shader_precompilation"],
      "enable_parallel_loading": profile["loading"]["parallel_loading"],
      "use_kv_cache": profile["memory"]["kv_cache_optimization"],
      "workgroup_size": profile["compute"]["workgroup_size"],
      "browser": this.get_browser_name(),
      "browser_version": this.get_browser_version()
    }
    
    # Apply model-specific optimizations
    if ($1) {
      config.update(${$1})
    elif ($1) {
      config.update(${$1})
    elif ($1) {
      config.update(${$1})
      # Special Firefox audio optimizations
      if ($1) {
        config["firefox_audio_optimization"] = true
    elif ($1) {
      config.update(${$1})
    
    }
    # Apply hardware-specific adjustments
      }
    if ($1) {
      # Low memory devices
      config["precision"] = "4bit"
      config["offload_weights"] = true
    
    }
    logger.info(`$1`)
    }
    return config
    }
  
    }
  $1($2): $3 {
    """
    Convert platform info to JSON.
    
  }
    Returns:
      JSON string with platform information
    """
    return json.dumps(this.platform_info, indent=2)

# Utility functions for simple access

def get_browser_capabilities() -> Dict[str, Any]:
  """
  Get current browser capabilities.
  
  Returns:
    Dictionary with browser capabilities
  """
  detector = PlatformDetector()
  return ${$1}


def get_hardware_capabilities() -> Dict[str, Any]:
  """
  Get current hardware capabilities.
  
  Returns:
    Dictionary with hardware capabilities
  """
  detector = PlatformDetector()
  return detector.platform_info["hardware"]


def create_platform_profile($1: string, $1: $2 | null = null, $1: $2 | null = null) -> Dict[str, Any]:
  """
  Create platform-specific configuration profile for a model type.
  
  Args:
    model_type: Type of model (text, vision, audio, multimodal)
    browser: Optional browser name to override detection
    version: Optional browser version to override detection
    
  Returns:
    Optimized configuration dictionary
  """
  detector = PlatformDetector(browser, version)
  return detector.create_configuration(model_type)


def detect_platform() -> Dict[str, Any]:
  """
  Detect platform capabilities.
  
  Returns:
    Dictionary with platform capabilities
  """
  detector = PlatformDetector()
  return detector.platform_info


def detect_browser_features() -> Dict[str, Any]:
  """
  Detect browser features.
  
  Returns:
    Dictionary with browser features
  """
  detector = PlatformDetector()
  return ${$1}


def get_feature_support_matrix() -> Dict[str, Dict[str, bool]]:
  """
  Get feature support matrix for major browsers.
  
  Returns:
    Dictionary mapping browser names to feature support status
  """
  browsers = ["chrome", "firefox", "safari", "edge"]
  features = [
    "webgpu", "compute_shaders", "shader_precompilation", 
    "2bit_precision", "3bit_precision", "4bit_precision", 
    "parallel_loading", "kv_cache", "model_sharding"
  ]
  
  matrix = {}
  
  for (const $1 of $2) {
    detector = PlatformDetector(browser=browser)
    browser_support = {}
    
  }
    # Check standard features
    browser_support["webgpu"] = detector.supports_feature("webgpu")
    browser_support["compute_shaders"] = detector.supports_feature("compute_shaders")
    browser_support["shader_precompilation"] = detector.supports_feature("shader_precompilation")
    browser_support["ultra_low_precision"] = detector.supports_feature("ultra_low_precision")
    
    # Check optimization profile for precision support
    profile = detector.get_optimization_profile()
    browser_support["2bit_precision"] = "2bit" in profile["precision"]["supported"]
    browser_support["3bit_precision"] = "3bit" in profile["precision"]["supported"]
    browser_support["4bit_precision"] = "4bit" in profile["precision"]["supported"]
    
    # Check other features
    browser_support["parallel_loading"] = profile["loading"]["parallel_loading"]
    browser_support["kv_cache"] = profile["memory"]["kv_cache_optimization"]
    
    matrix[browser] = browser_support
  
  return matrix