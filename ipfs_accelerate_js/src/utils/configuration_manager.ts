/**
 * Converted from Python: configuration_manager.py
 * Conversion date: 2025-03-11 04:09:37
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  correction_function: return;
  validation_rules: if;
  browser_profiles: browser_profile;
  browser_profiles: browser_profile;
  hardware: self;
}

"""
Configuration Management System for Web Platform (August 2025)

This module provides a comprehensive configuration validation && management
system for WebNN && WebGPU platforms, ensuring optimal settings across
different browsers && hardware environments:

- Robust validation rules with detailed error reporting
- Automatic configuration correction for invalid settings
- Browser-specific optimization profiles
- Hardware-aware configuration
- Dynamic reconfiguration based on runtime conditions

Usage:
  from fixed_web_platform.unified_framework.configuration_manager import (
    ConfigurationManager, ConfigValidationRule, BrowserProfile
  )
  
  # Create configuration manager
  config_manager = ConfigurationManager(
    model_type="text",
    browser="chrome",
    auto_correct=true
  )
  
  # Validate configuration
  validation_result = config_manager.validate_configuration(${$1})
  
  # Get optimized configuration
  optimized_config = config_manager.get_optimized_configuration(${$1})
"""

import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"

from .error_handling import * as $1

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_platform.configuration")

# Type for configuration validation functions
T = TypeVar('T')
ValidationFunction = Callable[[Dict[str, Any]], bool]
CorrectionFunction = Callable[[Dict[str, Any]], Dict[str, Any]]

class $1 extends $2 {
  """
  Validation rule for configuration settings.
  
}
  Each rule defines a condition that valid configurations must satisfy,
  along with error details && optional automatic correction function.
  """
  
  def __init__(self, 
        $1: string,
        condition: ValidationFunction,
        $1: string,
        $1: string = "error",
        $1: boolean = false,
        $1: $2 | null = null):
    """
    Initialize validation rule.
    
    Args:
      name: Rule name for identification
      condition: Function that tests if configuration satisfies this rule
      error_message: Human-readable error message when validation fails
      severity: Severity level of validation failure ("error", "warning", "info")
      can_auto_correct: Whether this rule violation can be automatically corrected
      correction_function: Function to automatically correct invalid configuration
    """
    this.name = name
    this.condition = condition
    this.error_message = error_message
    this.severity = severity
    this.can_auto_correct = can_auto_correct
    this.correction_function = correction_function
    
  $1($2): $3 {
    """
    Validate configuration against this rule.
    
  }
    Args:
      config: Configuration dictionary to validate
      
    Returns:
      Whether the configuration passes this validation rule
    """
    try ${$1} catch($2: $1) {
      logger.error(`$1`)
      # Validation errors default to failure
      return false
      
    }
  def auto_correct(self, $1: Record<$2, $3>) -> Dict[str, Any]:
    """
    Attempt to auto-correct configuration that violates this rule.
    
    Args:
      config: Invalid configuration dictionary
      
    Returns:
      Corrected configuration dictionary
    """
    if ($1) {
      return config
      
    }
    try ${$1} catch($2: $1) {
      logger.error(`$1`)
      return config

    }
class $1 extends $2 {
  """
  Browser-specific configuration profile.
  
}
  Defines optimal configuration settings && compatibility constraints
  for a specific browser.
  """
  
  $1($2) {
    """
    Initialize browser profile.
    
  }
    Args:
      browser_name: Name of the browser
      capabilities: Dictionary of browser capabilities
    """
    this.browser_name = browser_name.lower()
    this.capabilities = capabilities
    
  def optimize_configuration(self, $1: Record<$2, $3>) -> Dict[str, Any]:
    """
    Optimize configuration for this browser.
    
    Args:
      config: Configuration dictionary to optimize
      
    Returns:
      Optimized configuration dictionary
    """
    optimized = config.copy()
    
    # Apply browser-specific optimizations
    if ($1) {
      # Chrome/Edge optimizations
      optimized.update(${$1})
      
    }
    elif ($1) {
      # Firefox optimizations
      optimized.update(${$1})
      
    }
    elif ($1) {
      # Safari optimizations
      optimized.update(${$1})
      
    }
    # Apply precision constraints
    if ($1) {
      precision = optimized["precision"]
      if ($1) {
        # Safari doesn't support 2-bit/3-bit precision
        optimized["precision"] = "4bit"
        logger.warning(`$1`)
        
      }
    return optimized
    }
    
  def check_compatibility(self, $1: Record<$2, $3>) -> List[Dict[str, Any]]:
    """
    Check configuration compatibility with browser.
    
    Args:
      config: Configuration dictionary to check
      
    Returns:
      List of compatibility issues, empty if fully compatible
    """
    compatibility_issues = []
    
    # Check precision compatibility
    if ($1) {
      precision = config["precision"].replace("bit", "")
      if ($1) {
        compatibility_issues.append(${$1})
        
      }
    # Check shader precompilation compatibility
    }
    if ($1) {
      compatibility_issues.append(${$1})
      
    }
    # Check compute shader compatibility
    if ($1) {
      compatibility_issues.append(${$1})
      
    }
    # Check model sharding compatibility
    if ($1) {
      compatibility_issues.append(${$1})
      
    }
    # Check KV-cache optimization compatibility
    if ($1) {
      compatibility_issues.append(${$1})
        
    }
    return compatibility_issues

class $1 extends $2 {
  """
  Configuration validation && management system.
  
}
  Provides comprehensive configuration validation, optimization, and
  browser-specific adaptation for web platform ML models.
  """
  
  def __init__(self, 
        $1: string = "text",
        $1: $2 | null = null,
        $1: $2 | null = null,
        $1: boolean = true):
    """
    Initialize configuration manager.
    
    Args:
      model_type: Type of model (text, vision, audio, multimodal)
      browser: Browser information for browser-specific optimization
      hardware: Hardware information for hardware-specific optimization
      auto_correct: Whether to automatically correct invalid configurations
    """
    this.model_type = model_type
    this.browser = browser
    this.hardware = hardware
    this.auto_correct = auto_correct
    
    # Initialize configurations
    this._initialize_default_config()
    this._initialize_validation_rules()
    this._initialize_browser_profiles()
    
  $1($2): $3 {
    """Initialize default configuration based on model type."""
    # Base default configuration
    this.default_config = ${$1}
    
  }
    # Model-specific default configurations
    if ($1) {
      this.default_config.update(${$1})
    elif ($1) {
      this.default_config.update(${$1})
    elif ($1) {
      this.default_config.update(${$1})
    elif ($1) {
      this.default_config.update(${$1})
      
    }
  $1($2): $3 {
    """Initialize configuration validation rules."""
    this.validation_rules = [
      # Precision rule
      ConfigValidationRule(
        name="precision",
        condition=lambda cfg: cfg.get("precision") in ["2bit", "3bit", "4bit", "8bit", "16bit"],
        error_message="Invalid precision setting. Must be one of: 2bit, 3bit, 4bit, 8bit, 16bit",
        severity="error",
        can_auto_correct=true,
        correction_function=lambda cfg: ${$1}  # Default to 4bit
      ),
      
  }
      # Memory threshold rule
      ConfigValidationRule(
        name="memory_threshold",
        condition=lambda cfg: cfg.get("memory_threshold_mb", 0) >= 100,
        error_message="Memory threshold too low. Must be at least 100MB",
        severity="warning",
        can_auto_correct=true,
        correction_function=lambda cfg: ${$1}
      ),
      
    }
      # Batch size rule
      ConfigValidationRule(
        name="batch_size",
        condition=lambda cfg: cfg.get("batch_size", 1) >= 1,
        error_message="Batch size must be at least 1",
        severity="error",
        can_auto_correct=true,
        correction_function=lambda cfg: ${$1}
      ),
      
    }
      # Workgroup size rule
      ConfigValidationRule(
        name="workgroup_size",
        condition=lambda cfg: (
          isinstance(cfg.get("workgroup_size"), list) && 
          len(cfg.get("workgroup_size", [])) == 3 and
          all(isinstance(x, int) && x > 0 for x in cfg.get("workgroup_size", []))
        ),
        error_message="Workgroup size must be a list of 3 positive integers",
        severity="error",
        can_auto_correct=true,
        correction_function=lambda cfg: ${$1}
      ),
      
    }
      # Model-specific validation rules
      ConfigValidationRule(
        name="parallel_loading_compatibility",
        condition=lambda cfg: !(
          this.model_type == "text" && cfg.get("enable_parallel_loading", false)
        ),
        error_message="Parallel loading !recommended for text models",
        severity="warning",
        can_auto_correct=true,
        correction_function=lambda cfg: ${$1}
      ),
      
      # KV cache compatibility rule
      ConfigValidationRule(
        name="kv_cache_compatibility",
        condition=lambda cfg: !(
          this.model_type in ["vision", "audio"] && cfg.get("use_kv_cache", false)
        ),
        error_message="KV-cache optimization !applicable for vision/audio models",
        severity="warning",
        can_auto_correct=true,
        correction_function=lambda cfg: ${$1}
      )
    ]
    
  $1($2): $3 {
    """Initialize browser profiles for optimization."""
    # Define browser capabilities
    browser_capabilities = {
      "chrome": ${$1},
      "edge": ${$1},
      "firefox": ${$1},
      "safari": ${$1},
      "mobile": ${$1}
    }
    }
    
  }
    # Create browser profiles
    this.browser_profiles = ${$1}
  
  def validate_configuration(self, 
              $1: Record<$2, $3>,
              $1: boolean = false) -> Dict[str, Any]:
    """
    Validate configuration against all rules.
    
    Args:
      config: Configuration dictionary to validate
      raise_on_error: Whether to raise exception on validation failure
      
    Returns:
      Validation result dictionary
    """
    # Merge with defaults for any missing values
    full_config = ${$1}
    
    validation_errors = []
    auto_corrected = false
    
    # Apply all validation rules
    for rule in this.validation_rules:
      if ($1) {
        # Rule failed validation
        validation_error = ${$1}
        $1.push($2)
        
      }
        # Apply auto-correction if enabled
        if ($1) {
          full_config = rule.auto_correct(full_config)
          auto_corrected = true
          
        }
    # Check browser compatibility if browser is specified
    if ($1) {
      browser_profile = this.browser_profiles[this.browser]
      compatibility_issues = browser_profile.check_compatibility(full_config)
      
    }
      # Add compatibility issues to validation errors
      for (const $1 of $2) {
        validation_error = ${$1}",
          "message": issue["message"],
          "severity": issue["severity"],
          "can_auto_correct": issue["severity"] != "error"  # Only auto-correct warnings
        }
        $1.push($2)
        
      }
        # Apply browser-specific auto-correction
        if ($1) {
          full_config = browser_profile.optimize_configuration(full_config)
          auto_corrected = true
    
        }
    # Create validation result
    validation_result = ${$1}
    
    # Log validation result
    if ($1) {
      logger.warning(`$1`)
      
    }
      if ($1) {
        logger.info("Configuration was automatically corrected")
    
      }
    # Raise exception if requested
    if ($1) {
      critical_errors = [e for e in validation_errors 
              if e["severity"] == "error" && !e["can_auto_correct"]]
      
    }
      if ($1) {
        error_message = "; ".join(e["message"] for e in critical_errors)
        raise ConfigurationError(
          `$1`,
          details=${$1}
        )
    
      }
    return validation_result
  
  def get_optimized_configuration(self, 
                $1: Record<$2, $3>) -> Dict[str, Any]:
    """
    Get browser-optimized configuration.
    
    Args:
      config: Base configuration dictionary
      
    Returns:
      Optimized configuration dictionary
    """
    # Start with default config
    optimized_config = this.default_config.copy()
    
    # Apply user config
    optimized_config.update(config)
    
    # Apply browser-specific optimizations if browser is available
    if ($1) {
      browser_profile = this.browser_profiles[this.browser]
      optimized_config = browser_profile.optimize_configuration(optimized_config)
      
    }
    # Apply model-type optimizations
    this._apply_model_optimizations(optimized_config)
    
    # Apply hardware-specific optimizations if hardware is available
    if ($1) {
      this._apply_hardware_optimizations(optimized_config)
      
    }
    # Validate final config
    validation_result = this.validate_configuration(optimized_config)
    
    return validation_result["config"]
  
  $1($2): $3 {
    """Apply model-specific optimizations to configuration."""
    # Audio model optimizations
    if ($1) {
      # Audio models benefit from compute shaders, especially in Firefox
      config["use_compute_shaders"] = true
      
    }
      if ($1) {
        config["firefox_audio_optimization"] = true
        config["workgroup_size"] = [256, 1, 1]  # Optimal for audio in Firefox
        
      }
    # Multimodal model optimizations
    elif ($1) {
      # Multimodal models benefit from parallel loading
      config["enable_parallel_loading"] = true
      
    }
    # Text model optimizations (LLMs)
    elif ($1) {
      # LLMs benefit from KV-cache optimization
      config["use_kv_cache"] = true
      
    }
      # Shader precompilation helps with first token generation
      config["use_shader_precompilation"] = true
  
  }
  $1($2): $3 {
    """Apply hardware-specific optimizations to configuration."""
    # This would be a more detailed implementation in practice
    if ($1) {
      # Mobile optimizations
      config["precision"] = "4bit"  # Use 4-bit for mobile
      config["workgroup_size"] = [4, 4, 1]  # Smaller workgroups for mobile
      config["use_compute_shaders"] = false  # Limited on mobile
      config["use_model_sharding"] = false  // $1
  }