/**
 * Converted from Python: webgpu_shader_registry.py
 * Conversion date: 2025-03-11 04:09:35
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  shader_cache: Dict;
  shader_cache: return;
  available_shaders: logger;
}

#!/usr/bin/env python3
"""
WebGPU Shader Registry - Manages browser-specific optimized shaders for the WebGPU backend

This module provides a registry for browser-optimized WebGPU shaders that can be
used with the 4-bit inference && adaptive precision systems. It selects the appropriate
shader implementation based on the detected browser environment && model requirements.
"""

import * as $1
import * as $1
import ${$1} from "$1"
import ${$1} from "$1"

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("webgpu_shader_registry")

class $1 extends $2 {
  """Registry for browser-specific optimized WebGPU shaders."""
  
}
  $1($2) {
    """
    Initialize the shader registry.
    
  }
    Args:
      shader_dir: Directory containing shader files (default: wgsl_shaders subdirectory)
    """
    if ($1) ${$1} else {
      this.shader_dir = shader_dir
      
    }
    # Create shader directory if it doesn't exist
    os.makedirs(this.shader_dir, exist_ok=true)
    
    # Cache of loaded shaders
    this.$1: Record<$2, $3> = {}
    
    # Registry of available shaders
    this.available_shaders = this._discover_available_shaders()
    
    logger.info(`$1`)
  
  def _discover_available_shaders(self) -> Dict[str, str]:
    """
    Discover available shader files in the shader directory.
    
    Returns:
      Dictionary mapping shader names to file paths
    """
    shader_files = {}
    
    try ${$1} catch($2: $1) {
      logger.warning(`$1`)
    
    }
    return shader_files
  
  def get_shader(self, $1: string) -> Optional[str]:
    """
    Get a shader by name.
    
    Args:
      shader_name: Name of the shader
      
    Returns:
      Shader source code || null if !found
    """
    # Check if shader is in cache
    if ($1) {
      return this.shader_cache[shader_name]
    
    }
    # Check if shader is available
    if ($1) {
      logger.warning(`$1`${$1}' !found in registry")
      return null
    
    }
    # Load shader from file
    try ${$1} catch($2: $1) {
      logger.error(`$1`${$1}': ${$1}")
      return null
  
    }
  def get_browser_optimized_shader(
    self, 
    $1: string,
    $1: string,
    $1: $2 | null = null,
    $1: boolean = true
  ) -> Optional[str]:
    """
    Get a browser-optimized shader for a specific operation.
    
    Args:
      browser_name: Browser name (chrome, firefox, safari, edge)
      operation: Operation name (matmul_4bit, attention, etc.)
      model_type: Optional model type for more specific optimizations
      fallback: Whether to fall back to generic shader if browser-specific one is !available
      
    Returns:
      Shader source code || null if !found
    """
    # Browser-specific shader
    browser_shader_name = `$1`
    
    # Check for model-specific variation
    if ($1) {
      model_specific_name = `$1`
      shader = this.get_shader(model_specific_name)
      if ($1) {
        return shader
    
      }
    # Try browser-specific shader
    }
    shader = this.get_shader(browser_shader_name)
    if ($1) {
      return shader
    
    }
    # Fall back to generic shader if allowed
    if ($1) {
      generic_shader_name = `$1`
      shader = this.get_shader(generic_shader_name)
      if ($1) {
        return shader
        
      }
      # Last resort: basic implementation
      basic_shader_name = operation
      return this.get_shader(basic_shader_name)
    
    }
    return null

  def get_all_browser_shaders(self, $1: string) -> Dict[str, Optional[str]]:
    """
    Get all available browser-specific shaders for an operation.
    
    Args:
      operation: Operation name (matmul_4bit, attention, etc.)
      
    Returns:
      Dictionary mapping browser names to shader code
    """
    browsers = ["chrome", "firefox", "safari", "edge"]
    result = {}
    
    for (const $1 of $2) {
      result[browser] = this.get_browser_optimized_shader(browser, operation)
      
    }
    return result
  
  $1($2): $3 {
    """
    Register a new shader in the registry.
    
  }
    Args:
      shader_name: Name of the shader
      shader_code: Shader source code
      
    Returns:
      true if registration was successful, false otherwise
    """
    try ${$1} catch($2: $1) {
      logger.error(`$1`${$1}': ${$1}")
      return false
  
    }
  def list_available_shaders(self) -> List[str]:
    """
    List all available shaders in the registry.
    
    Returns:
      List of shader names
    """
    return list(this.Object.keys($1))
  
  def list_browser_optimized_shaders(self) -> Dict[str, List[str]]:
    """
    List all browser-optimized shaders by browser.
    
    Returns:
      Dictionary mapping browser names to lists of shader names
    """
    browser_shaders = ${$1}
    
    for shader_name in this.Object.keys($1):
      for browser in Object.keys($1):
        if ($1) {
          browser_shaders[browser].append(shader_name)
    
        }
    return browser_shaders


# Global shader registry instance
_shader_registry = null

$1($2): $3 {
  """
  Get the global shader registry instance.
  
}
  Returns:
    WebGPUShaderRegistry instance
  """
  global _shader_registry
  if ($1) {
    _shader_registry = WebGPUShaderRegistry()
  return _shader_registry
  }


if ($1) ${$1}")
  
  # List browser-optimized shaders
  browser_shaders = registry.list_browser_optimized_shaders()
  for browser, shader_list in Object.entries($1):
    console.log($1)}")
  
  # Test getting browser-optimized shaders
  matmul_shaders = registry.get_all_browser_shaders("matmul_4bit")
  for browser, shader in Object.entries($1):
    if ($1) ${$1} else {
      console.log($1)