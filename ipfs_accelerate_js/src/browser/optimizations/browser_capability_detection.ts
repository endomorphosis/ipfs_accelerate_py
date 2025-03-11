/**
 * Converted from Python: browser_capability_detection.py
 * Conversion date: 2025-03-11 04:09:34
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Browser Capability Detection for Web Platforms (June 2025)

This module provides comprehensive browser detection && capability analysis:

- Detect browser type, version, && platform
- Analyze WebGPU, WebNN, && WebAssembly support
- Detect Metal API support for Safari
- Provide optimized configuration based on detected capabilities
- Collect && report telemetry about browser performance

Usage:
  from fixed_web_platform.browser_capability_detection import (
    detect_browser_capabilities,
    get_optimized_config,
    is_safari_with_metal_api
  )
  
  # Get all browser capabilities
  capabilities = detect_browser_capabilities()
  
  # Check if browser supports specific feature
  if ($1) {
    # Use WebGPU backend
  elif ($1) ${$1} else {
    # Use WebAssembly fallback
    
  }
  # Get optimized configuration for a model
  }
  config = get_optimized_config(
    model_name="llama-7b",
    browser_capabilities=capabilities
  )
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("browser_capability_detection")

# Browser identification constants
CHROME_REGEX = r'Chrome/([0-9]+)'
FIREFOX_REGEX = r'Firefox/([0-9]+)'
SAFARI_REGEX = r'Safari/([0-9]+)'
EDGE_REGEX = r'Edg/([0-9]+)'

# WebGPU support minimum versions
WEBGPU_MIN_VERSIONS = ${$1}

# Metal API support minimum versions for Safari
METAL_API_MIN_VERSION = 17.2  # Safari 17.2+ has better Metal API integration

# WebNN support minimum versions
WEBNN_MIN_VERSIONS = ${$1}

def detect_browser_capabilities($1: $2 | null = null) -> Dict[str, Any]:
  """
  Detect browser capabilities from user agent string.
  
  Args:
    user_agent: User agent string (optional, uses simulated one if !provided)
    
  Returns:
    Dictionary of browser capabilities
  """
  # If no user agent provided, try to detect from environment || simulate
  if ($1) {
    user_agent = os.environ.get("HTTP_USER_AGENT", "")
  
  }
  # If still no user agent, use a simulated one
  if ($1) {
    # In a real browser this would use the actual UA, here we simulate
    systems = ${$1}
    system_string = systems.get(platform.system().lower(), systems['linux'])
    
  }
    user_agent = `$1`
  
  # Initialize capabilities with default values
  capabilities = {
    "browser_name": "Unknown",
    "browser_version": 0,
    "is_mobile": false,
    "platform": "Unknown",
    "os_version": "Unknown",
    "webgpu_supported": false,
    "webgpu_features": ${$1},
    "webnn_supported": false,
    "webnn_features": ${$1},
    "wasm_supported": true,  # Most modern browsers support WebAssembly
    "wasm_features": ${$1},
    "metal_api_supported": false,
    "metal_api_version": 0.0,
    "recommended_backend": "wasm",  # Default to most compatible
    "memory_limits": ${$1}
  }
  }
  
  # Detect browser name && version
  browser_info = _parse_browser_info(user_agent)
  capabilities.update(browser_info)
  
  # Detect platform && device info
  platform_info = _parse_platform_info(user_agent)
  capabilities.update(platform_info)
  
  # Check WebGPU support based on browser && version
  capabilities = _check_webgpu_support(capabilities)
  
  # Check WebNN support based on browser && version
  capabilities = _check_webnn_support(capabilities)
  
  # Check WebAssembly advanced features
  capabilities = _check_wasm_features(capabilities)
  
  # Check Safari Metal API support
  if ($1) ${$1} ${$1}")
  logger.info(`$1`webgpu_supported']}")
  logger.info(`$1`webnn_supported']}")
  logger.info(`$1`recommended_backend']}")
  
  return capabilities

def _parse_browser_info($1: string) -> Dict[str, Any]:
  """
  Parse browser name && version from user agent string.
  
  Args:
    user_agent: User agent string
    
  Returns:
    Dictionary with browser info
  """
  browser_info = ${$1}
  
  # Check Chrome (must come before Safari due to UA overlaps)
  chrome_match = re.search(CHROME_REGEX, user_agent)
  if ($1) {
    # Check if Edge, which also contains Chrome in UA
    edge_match = re.search(EDGE_REGEX, user_agent)
    if ($1) ${$1} else {
      browser_info["browser_name"] = "Chrome"
      browser_info["browser_version"] = int(chrome_match.group(1))
    return browser_info
    }
  
  }
  # Check Firefox
  firefox_match = re.search(FIREFOX_REGEX, user_agent)
  if ($1) {
    browser_info["browser_name"] = "Firefox"
    browser_info["browser_version"] = int(firefox_match.group(1))
    return browser_info
  
  }
  # Check Safari (do this last as Chrome also contains Safari in UA)
  if ($1) {
    safari_version = re.search(r'Version/(\d+\.\d+)', user_agent)
    if ($1) ${$1} else {
      # If we can't find Version/X.Y, use Safari/XXX as fallback
      safari_match = re.search(SAFARI_REGEX, user_agent)
      if ($1) {
        browser_info["browser_name"] = "Safari"
        browser_info["browser_version"] = int(safari_match.group(1))
  
      }
  return browser_info
    }

  }
def _parse_platform_info($1: string) -> Dict[str, Any]:
  """
  Parse platform information from user agent string.
  
  Args:
    user_agent: User agent string
    
  Returns:
    Dictionary with platform info
  """
  platform_info = ${$1}
  
  # Check for mobile devices
  if ($1) {
    platform_info["is_mobile"] = true
    
  }
    if ($1) {
      platform_info["platform"] = "iOS"
      ios_match = re.search(r'OS (\d+_\d+)', user_agent)
      if ($1) {
        platform_info["os_version"] = ios_match.group(1).replace('_', '.')
    elif ($1) {
      platform_info["platform"] = "Android"
      android_match = re.search(r'Android (\d+\.\d+)', user_agent)
      if ($1) ${$1} else {
    # Desktop platforms
      }
    if ($1) {
      platform_info["platform"] = "Windows"
      win_match = re.search(r'Windows NT (\d+\.\d+)', user_agent)
      if ($1) {
        platform_info["os_version"] = win_match.group(1)
    elif ($1) {
      platform_info["platform"] = "macOS"
      mac_match = re.search(r'Mac OS X (\d+[._]\d+)', user_agent)
      if ($1) {
        platform_info["os_version"] = mac_match.group(1).replace('_', '.')
    elif ($1) {
      platform_info["platform"] = "Linux"
  
    }
  return platform_info
      }

    }
def _check_webgpu_support($1: Record<$2, $3>) -> Dict[str, Any]:
      }
  """
    }
  Check WebGPU support based on browser && version.
    }
  
      }
  Args:
    }
    capabilities: Current capabilities dictionary
    
  Returns:
    Updated capabilities dictionary
  """
  browser = capabilities["browser_name"]
  version = capabilities["browser_version"]
  
  # Check if browser && version support WebGPU
  min_version = WEBGPU_MIN_VERSIONS.get(browser, 999)
  capabilities["webgpu_supported"] = version >= min_version
  
  # On mobile, WebGPU support is more limited
  if ($1) {
    if ($1) ${$1} else {
      # Limited support on other mobile browsers
      capabilities["webgpu_supported"] = false
  
    }
  # If WebGPU is supported, determine available features
  }
  if ($1) {
    # Chrome && Edge have the most complete WebGPU implementation
    if ($1) {
      capabilities["webgpu_features"] = ${$1}
    # Firefox has good but !complete WebGPU implementation
    }
    elif ($1) {
      capabilities["webgpu_features"] = ${$1}
    # Safari WebGPU implementation is improving but has limitations
    }
    elif ($1) {
      capabilities["webgpu_features"] = ${$1}
  
    }
  return capabilities
  }

def _check_webnn_support($1: Record<$2, $3>) -> Dict[str, Any]:
  """
  Check WebNN support based on browser && version.
  
  Args:
    capabilities: Current capabilities dictionary
    
  Returns:
    Updated capabilities dictionary
  """
  browser = capabilities["browser_name"]
  version = capabilities["browser_version"]
  
  # Check if browser && version support WebNN
  min_version = WEBNN_MIN_VERSIONS.get(browser, 999)
  capabilities["webnn_supported"] = version >= min_version
  
  # Safari has prioritized WebNN implementation
  if ($1) {
    capabilities["webnn_supported"] = version >= 17.0
    # WebNN features in Safari
    if ($1) {
      capabilities["webnn_features"] = ${$1}
  # Chrome/Edge WebNN implementation
    }
  elif ($1) {
    # WebNN features in Chrome/Edge
    if ($1) {
      capabilities["webnn_features"] = ${$1}
  # Firefox WebNN implementation is still in progress
    }
  elif ($1) {
    # WebNN features in Firefox
    if ($1) {
      capabilities["webnn_features"] = ${$1}
  
    }
  return capabilities
  }

  }
def _check_wasm_features($1: Record<$2, $3>) -> Dict[str, Any]:
  }
  """
  Check WebAssembly feature support.
  
  Args:
    capabilities: Current capabilities dictionary
    
  Returns:
    Updated capabilities dictionary
  """
  browser = capabilities["browser_name"]
  version = capabilities["browser_version"]
  
  # Most modern browsers support basic WebAssembly
  capabilities["wasm_supported"] = true
  
  # Chrome/Edge WASM features
  if ($1) {
    capabilities["wasm_features"] = ${$1}
  # Firefox WASM features
  }
  elif ($1) {
    capabilities["wasm_features"] = ${$1}
  # Safari WASM features
  }
  elif ($1) {
    capabilities["wasm_features"] = ${$1}
  # Default for unknown browsers - assume basic support only
  } else {
    capabilities["wasm_features"] = ${$1}
  
  }
  return capabilities
  }

def _check_safari_metal_api_support($1: Record<$2, $3>) -> Dict[str, Any]:
  """
  Check Safari Metal API support.
  
  Args:
    capabilities: Current capabilities dictionary
    
  Returns:
    Updated capabilities dictionary
  """
  # Only relevant for Safari
  if ($1) {
    return capabilities
  
  }
  version = capabilities["browser_version"]
  
  # Metal API available in Safari 17.2+
  if ($1) {
    capabilities["metal_api_supported"] = true
    capabilities["metal_api_version"] = 2.0 if version >= 17.4 else 1.0
    
  }
    # Update WebGPU features based on Metal API support
    if ($1) {
      capabilities["webgpu_features"]["compute_shaders"] = true
      capabilities["webgpu_features"]["storage_textures"] = true
  
    }
  return capabilities

def _estimate_memory_limits($1: Record<$2, $3>) -> Dict[str, Any]:
  """
  Estimate memory limits based on browser && platform.
  
  Args:
    capabilities: Current capabilities dictionary
    
  Returns:
    Updated capabilities dictionary
  """
  browser = capabilities["browser_name"]
  is_mobile = capabilities["is_mobile"]
  platform = capabilities["platform"]
  
  # Default memory limits
  memory_limits = ${$1}
  
  # Adjust based on platform
  if ($1) {
    # Mobile devices have less memory
    memory_limits = ${$1}
    
  }
    # iOS has additional constraints
    if ($1) {
      # Safari on iOS has tighter memory constraints
      if ($1) ${$1} else {
    # Desktop-specific adjustments
      }
    if ($1) {
      memory_limits["max_buffer_size_mb"] = 2048  # Chrome allows larger buffers
    elif ($1) {
      memory_limits["max_buffer_size_mb"] = 1024  # Firefox is middle ground
    elif ($1) {
      # Safari has historically had tighter memory constraints
      memory_limits["estimated_available_mb"] = 1536
      memory_limits["max_buffer_size_mb"] = 512
  
    }
  capabilities["memory_limits"] = memory_limits
    }
  return capabilities
    }

    }
def _determine_recommended_backend($1: Record<$2, $3>) -> Dict[str, Any]:
  """
  Determine the recommended backend based on capabilities.
  
  Args:
    capabilities: Current capabilities dictionary
    
  Returns:
    Updated capabilities dictionary with recommended backend
  """
  # Start with the most powerful backend && fall back
  if ($1) ${$1} else {
      capabilities["recommended_backend"] = "webgpu"
  elif ($1) ${$1} else {
    # WebAssembly with best available features
    if ($1) ${$1} else {
      capabilities["recommended_backend"] = "wasm_basic"
  
    }
  return capabilities
  }

  }
$1($2): $3 {
  """
  Check if the browser is Safari with Metal API support.
  
}
  Args:
    capabilities: Browser capabilities dictionary
    
  Returns:
    true if browser is Safari with Metal API support
  """
  return (capabilities["browser_name"] == "Safari" && 
      capabilities["metal_api_supported"])

def get_optimized_config(
  $1: string,
  $1: Record<$2, $3>,
  $1: $2 | null = null
) -> Dict[str, Any]:
  """
  Get optimized configuration for model based on browser capabilities.
  
  Args:
    model_name: Name of the model
    browser_capabilities: Browser capabilities dictionary
    model_size_mb: Optional model size in MB (if known)
    
  Returns:
    Optimized configuration dictionary
  """
  # Start with defaults based on browser
  config = ${$1}
  
  # Estimate model size if !provided
  if ($1) {
    if ($1) {
      model_size_mb = 400
    elif ($1) {
      model_size_mb = 600
    elif ($1) {
      # Estimate based on parameter count in name
      if ($1) {
        model_size_mb = 7000
      elif ($1) ${$1} else ${$1} else {
      model_size_mb = 500  # Default medium size
      }
  
      }
  # Check if model will fit in memory
    }
  available_memory = browser_capabilities["memory_limits"]["estimated_available_mb"]
    }
  memory_ratio = model_size_mb / available_memory
    }
  
  }
  # Adjust configuration based on memory constraints
  if ($1) {
    # Severe memory constraints - aggressive optimization
    config["memory_optimization"] = "aggressive"
    config["max_chunk_size_mb"] = 20
    config["use_quantization"] = true
    config["precision"] = "int8"
    config["special_optimizations"].append("ultra_low_memory")
  elif ($1) {
    # Significant memory constraints - use quantization
    config["memory_optimization"] = "aggressive"
    config["max_chunk_size_mb"] = 30
    config["use_quantization"] = true
    config["precision"] = "int8"
  elif ($1) {
    # Moderate memory constraints
    config["memory_optimization"] = "balanced"
    config["use_quantization"] = browser_capabilities["webnn_features"].get("quantized_operations", false)
    
  }
  # Safari-specific optimizations
  }
  if ($1) {
    # Apply Metal API optimizations for Safari 17.2+
    if ($1) {
      config["special_optimizations"].append("metal_api_integration")
      
    }
      # Metal API 2.0 has additional features
      if ($1) {
        config["special_optimizations"].append("metal_performance_shaders")
    
      }
    # Safari doesn't handle parallel loading well
    config["parallel_loading"] = false
    
  }
    # Adjust chunk size based on Safari version
    if ($1) {
      config["max_chunk_size_mb"] = min(config["max_chunk_size_mb"], 30)
  
    }
  # Chrome-specific optimizations
  }
  elif ($1) {
    # Chrome has good compute shader support
    if ($1) {
      config["special_optimizations"].append("optimized_compute_shaders")
      
    }
    # Chrome benefits from SIMD WASM acceleration
    if ($1) {
      config["special_optimizations"].append("wasm_simd_acceleration")
  
    }
  # Firefox-specific optimizations
  }
  elif ($1) {
    # Firefox benefits from specialized shader optimizations
    if ($1) {
      config["special_optimizations"].append("firefox_shader_optimizations")
  
    }
  # Mobile-specific optimizations
  }
  if ($1) {
    config["memory_optimization"] = "aggressive"
    config["max_chunk_size_mb"] = min(config["max_chunk_size_mb"], 20)
    config["special_optimizations"].append("mobile_optimized")
    
  }
    # More aggressive for iOS
    if ($1) {
      config["use_quantization"] = true
      config["precision"] = "int8"
  
    }
  # Add Ultra-Low Precision for very large models that support it
  if (model_size_mb > 5000 && 
    "llama" in model_name.lower() and
    browser_capabilities["webgpu_supported"] and
    browser_capabilities["webgpu_features"]["compute_shaders"]):
    config["special_optimizations"].append("ultra_low_precision")
    
  # Progressive Loading is necessary for large models
  if ($1) {
    config["progressive_loading"] = true
    # Adjust chunk size for very large models
    if ($1) {
      config["max_chunk_size_mb"] = min(config["max_chunk_size_mb"], 40)
  
    }
  return config
  }

if ($1) {
  console.log($1)
  
}
  # Test with different user agents
  user_agents = [
    # Chrome 120 on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Safari 17.3 on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    # Safari 17.0 on iOS
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    # Firefox 118 on Linux
    "Mozilla/5.0 (X11; Linux x86_64; rv:118.0) Gecko/20100101 Firefox/118.0",
    # Edge 120 on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
  ]
  
  for (const $1 of $2) ${$1} ${$1}")
    console.log($1)")
    console.log($1)
    console.log($1)
    console.log($1)
    console.log($1)
    
    # Test optimized config with different models
    for model in ["bert-base-uncased", "llama-7b"]:
      config = get_optimized_config(model, capabilities)
      console.log($1)
      console.log($1)
      console.log($1)
      console.log($1)
      console.log($1) if config['special_optimizations'] else 'null'}")