/**
 * Converted from Python: error_propagation.py
 * Conversion date: 2025-03-11 04:09:38
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  max_history: self;
  component_errors: self;
  error_peaks: self;
  component_errors: return;
  components: self;
  collect_telemetry: self;
  collect_telemetry: self;
  collect_telemetry: self;
  collect_telemetry: self;
  handlers: return;
  collect_telemetry: return;
  collect_telemetry: return;
}

"""
Cross-Component Error Propagation for Web Platform (August 2025)

This module implements standardized error propagation between different components
of the web platform framework, ensuring:
- Consistent error handling across components
- Error categorization && standardized telemetry
- Graceful degradation pathways for critical errors
- Cross-component communication for related errors

Usage:
  from fixed_web_platform.unified_framework.error_propagation import (
    ErrorPropagationManager, ErrorTelemetryCollector, register_handler
  )
  
  # Create error propagation manager
  error_manager = ErrorPropagationManager(
    components=["webgpu", "streaming", "quantization"],
    collect_telemetry=true
  )
  
  # Register component error handlers
  error_manager.register_handler("streaming", streaming_component.handle_error)
  
  # Propagate errors between components
  error_manager.propagate_error(error, source_component="webgpu")
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"

# Import error handling
from fixed_web_platform.unified_framework.error_handling import (
  ErrorHandler, WebPlatformError, RuntimeError, HardwareError
)

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_platform.error_propagation")

class $1 extends $2 {
  """Enumeration of standardized error categories."""
  MEMORY = "memory"
  TIMEOUT = "timeout"
  CONNECTION = "connection"
  BROWSER_COMPATIBILITY = "browser_compatibility"
  HARDWARE = "hardware"
  CONFIGURATION = "configuration"
  RUNTIME = "runtime"
  UNKNOWN = "unknown"

}
class $1 extends $2 {
  """
  Collects && aggregates error telemetry data across components.
  
}
  Features:
  - Standardized error category tracking
  - Component-specific error frequency analysis
  - Error recovery success rate tracking
  - Temporal error pattern detection
  """
  
  $1($2) {
    """
    Initialize telemetry collector.
    
  }
    Args:
      max_history: Maximum number of error records to retain
    """
    this.max_history = max_history
    this.error_history = []
    this.error_categories = {}
    this.component_errors = {}
    this.recovery_attempts = ${$1}
    this.error_peaks = {}
    
  $1($2): $3 {
    """
    Record an error in telemetry.
    
  }
    Args:
      error: Error data dictionary
    """
    # Add timestamp if !present
    if ($1) {
      error["timestamp"] = time.time()
      
    }
    # Add to history, maintaining max size
    this.$1.push($2)
    if ($1) {
      this.error_history = this.error_history[-this.max_history:]
      
    }
    # Track by category
    category = error.get("category", ErrorCategory.UNKNOWN)
    this.error_categories[category] = this.error_categories.get(category, 0) + 1
    
    # Track by component
    component = error.get("component", "unknown")
    if ($1) {
      this.component_errors[component] = {}
    
    }
    comp_category = `$1`
    this.component_errors[component][category] = this.component_errors[component].get(category, 0) + 1
    
    # Check for error peaks (multiple errors in short time window)
    current_time = error.get("timestamp", time.time())
    recent_window = [e for e in this.error_history 
            if e.get("category") == category && 
            current_time - e.get("timestamp", 0) < 60]  # 60 second window
    
    if ($1) {  # 3+ errors of same type in 60 seconds
      if ($1) {
        this.error_peaks[category] = []
      
      }
      this.error_peaks[category].append(${$1})
      
      # Log error peak detection
      logger.warning(`$1`timestamp'):.1f} seconds")
  
  $1($2): $3 {
    """
    Record a recovery attempt outcome.
    
  }
    Args:
      success: Whether recovery was successful
    """
    if ($1) ${$1} else {
      this.recovery_attempts["failure"] += 1
  
    }
  def get_summary(self) -> Dict[str, Any]:
    """
    Get a summary of telemetry data.
    
    Returns:
      Dictionary with telemetry summary
    """
    total_errors = len(this.error_history)
    total_recovery_attempts = this.recovery_attempts["success"] + this.recovery_attempts["failure"]
    recovery_success_rate = (this.recovery_attempts["success"] / total_recovery_attempts 
              if total_recovery_attempts > 0 else 0)
    
    return ${$1}
  
  def get_component_summary(self, $1: string) -> Dict[str, Any]:
    """
    Get error summary for a specific component.
    
    Args:
      component: Component name
      
    Returns:
      Dictionary with component error summary
    """
    if ($1) {
      return {"component": component, "errors": 0, "categories": {}}
      
    }
    component_history = $3.map(($2) => $1)
    
    return ${$1}
  
  $1($2): $3 {
    """Clear all telemetry data."""
    this.error_history = []
    this.error_categories = {}
    this.component_errors = {}
    this.recovery_attempts = ${$1}
    this.error_peaks = {}

  }

class $1 extends $2 {
  """
  Manages error propagation between components.
  
}
  Features:
  - Centralized error handling for multiple components
  - Standardized error propagation between components
  - Component-specific error handlers with prioritization
  - Error telemetry collection
  """
  
  def __init__(self, 
        $1: $2[] = null,
        $1: boolean = true):
    """
    Initialize error propagation manager.
    
    Args:
      components: List of component names
      collect_telemetry { Whether to collect error telemetry
    """
    this.components = components || []
    this.handlers = {}
    this.error_handler = ErrorHandler(recovery_strategy="auto")
    this.collect_telemetry = collect_telemetry
    
    if ($1) {
      this.telemetry = ErrorTelemetryCollector()
    
    }
    # Set up component dependencies
    this.dependencies = ${$1}
  
  $1($2): $3 {
    """
    Register an error handler for a component.
    
  }
    Args:
      component: Component name
      handler: Error handler function
    """
    if ($1) {
      this.$1.push($2)
      
    }
    this.handlers[component] = handler
    logger.debug(`$1`)
  
  $1($2): $3 {
    """
    Categorize an error based on its characteristics.
    
  }
    Args:
      error: Error object || dictionary
      
    Returns:
      Error category string
    """
    # Extract error message
    if ($1) ${$1} else {
      error_message = error.get("message", "").lower()
      error_type = error.get("type", "unknown")
    
    }
    # Categorize based on message && type
    if ($1) {
      return ErrorCategory.MEMORY
      
    }
    elif ($1) {
      return ErrorCategory.TIMEOUT
      
    }
    elif ($1) {
      return ErrorCategory.CONNECTION
      
    }
    elif ($1) {
      return ErrorCategory.BROWSER_COMPATIBILITY
      
    }
    elif ($1) {
      return ErrorCategory.HARDWARE
      
    }
    elif ($1) ${$1} else {
      return ErrorCategory.RUNTIME
  
    }
  def propagate_error(self, 
          $1: $2],
          $1: string,
          context: Optional[Dict[str, Any]] = null) -> Dict[str, Any]:
    """
    Propagate an error to affected components.
    
    Args:
      error: Error object || dictionary
      source_component: Component where error originated
      context: Optional context information
      
    Returns:
      Error handling result dictionary
    """
    context = context || {}
    
    # Create standardized error record
    if ($1) {
      # Convert to web platform error if needed
      if ($1) {
        error = this.error_handler._convert_exception(error, context)
        
      }
      error_record = {
        "type": error.__class__.__name__,
        "message": str(error),
        "details": getattr(error, "details", {}),
        "severity": getattr(error, "severity", "error"),
        "timestamp": time.time(),
        "component": source_component,
        "category": this.categorize_error(error),
        "traceback": traceback.format_exc(),
        "context": context
      }
    } else {
      # Already a dictionary
      error_record = error.copy()
      error_record.setdefault("timestamp", time.time())
      error_record.setdefault("component", source_component)
      error_record.setdefault("category", this.categorize_error(error))
      error_record.setdefault("context", context)
    
    }
    # Record in telemetry
      }
    if ($1) {
      this.telemetry.record_error(error_record)
    
    }
    # Determine affected components based on dependencies
    }
    affected_components = this._get_affected_components(source_component)
    
    # Handle in source component first
    source_result = this._handle_in_component(error_record, source_component)
    
    # If source component handled successfully, we're done
    if ($1) {
      if ($1) {
        this.telemetry.record_recovery_attempt(true)
      return ${$1}
      }
    
    }
    # Try handling in affected components
    for (const $1 of $2) {
      component_result = this._handle_in_component(error_record, component)
      if ($1) {
        if ($1) {
          this.telemetry.record_recovery_attempt(true)
        return ${$1}
        }
    
      }
    # If we got here, no component could handle the error
    }
    if ($1) {
      this.telemetry.record_recovery_attempt(false)
      
    }
    # For critical errors, implement graceful degradation
    if ($1) {
      degradation_result = this._implement_graceful_degradation(error_record)
      if ($1) {
        return ${$1}
    
      }
    return ${$1}
    }
  
  def _get_affected_components(self, $1: string) -> List[str]:
    """
    Get components affected by an error in the source component.
    
    Args:
      source_component: Component where error originated
      
    Returns:
      List of affected component names
    """
    affected = []
    
    # Add components that depend on the source component
    for component, dependencies in this.Object.entries($1):
      if ($1) {
        $1.push($2)
    
      }
    return affected
  
  def _handle_in_component(self, 
            $1: Record<$2, $3>,
            $1: string) -> Dict[str, Any]:
    """
    Handle error in a specific component.
    
    Args:
      error_record: Error record dictionary
      component: Component name
      
    Returns:
      Handling result dictionary
    """
    # Skip if component has no handler
    if ($1) {
      return ${$1}
    
    }
    # Create component-specific context
    component_context = ${$1}
    
    # Add original error details
    if ($1) {
      component_context["error_details"] = error_record["details"]
    
    }
    try {
      # Call component handler
      handler = this.handlers[component]
      result = handler(error_record, component_context)
      
    }
      # Return if handler provided result
      if ($1) {
        result.setdefault("handled", false)
        return result
        
      }
      # If handler returned true/false, construct default result
      if ($1) {
        return ${$1}
        
      }
      # Default to !handled
      return ${$1}
      
    } catch($2: $1) {
      # Handler raised an exception
      logger.error(`$1`)
      return ${$1}
  
    }
  def _implement_graceful_degradation(self, 
                  $1: Record<$2, $3>) -> Dict[str, Any]:
    """
    Implement graceful degradation for critical errors.
    
    Args:
      error_record: Error record dictionary
      
    Returns:
      Degradation result dictionary
    """
    category = error_record.get("category")
    source_component = error_record.get("component")
    
    # Choose degradation strategy based on error category
    if ($1) {
      return this._handle_memory_degradation(source_component)
      
    }
    elif ($1) {
      return this._handle_timeout_degradation(source_component)
      
    }
    elif ($1) {
      return this._handle_connection_degradation(source_component)
      
    }
    elif ($1) {
      return this._handle_compatibility_degradation(source_component)
      
    }
    elif ($1) {
      return this._handle_hardware_degradation(source_component)
      
    }
    # Default to no degradation for other categories
    return ${$1}
  
  def _handle_memory_degradation(self, $1: string) -> Dict[str, Any]:
    """
    Handle memory-related degradation.
    
    Args:
      component: Affected component
      
    Returns:
      Degradation result dictionary
    """
    if ($1) {
      # For streaming, reduce batch size && precision
      return ${$1}
    elif ($1) {
      # For WebGPU, fall back to WebNN || WASM
      return ${$1}
    } else {
      # Generic memory reduction
      return ${$1}
  
    }
  def _handle_timeout_degradation(self, $1: string) -> Dict[str, Any]:
    }
    """
    }
    Handle timeout-related degradation.
    
    Args:
      component: Affected component
      
    Returns:
      Degradation result dictionary
    """
    if ($1) {
      # For streaming, reduce generation parameters && optimizations
      return ${$1}
    } else {
      # Generic timeout handling
      return ${$1}
  
    }
  def _handle_connection_degradation(self, $1: string) -> Dict[str, Any]:
    }
    """
    Handle connection-related degradation.
    
    Args:
      component: Affected component
      
    Returns:
      Degradation result dictionary
    """
    if ($1) {
      # For streaming, switch to non-streaming mode
      return ${$1}
    } else {
      # Generic connection handling with retry && backoff
      return ${$1}
  
    }
  def _handle_compatibility_degradation(self, $1: string) -> Dict[str, Any]:
    }
    """
    Handle browser compatibility degradation.
    
    Args:
      component: Affected component
      
    Returns:
      Degradation result dictionary
    """
    # Fall back to most widely supported implementation
    return ${$1}
  
  def _handle_hardware_degradation(self, $1: string) -> Dict[str, Any]:
    """
    Handle hardware-related degradation.
    
    Args:
      component: Affected component
      
    Returns:
      Degradation result dictionary
    """
    if ($1) {
      # Fall back to CPU implementation
      return ${$1}
    } else {
      # Generic hardware degradation
      return ${$1}
  
    }
  def get_telemetry_summary(self) -> Dict[str, Any]:
    }
    """
    Get error telemetry summary.
    
    Returns:
      Dictionary with telemetry summary
    """
    if ($1) {
      return ${$1}
      
    }
    return this.telemetry.get_summary()
  
  def get_component_telemetry(self, $1: string) -> Dict[str, Any]:
    """
    Get telemetry for a specific component.
    
    Args:
      component: Component name
      
    Returns:
      Dictionary with component telemetry
    """
    if ($1) {
      return ${$1}
      
    }
    return this.telemetry.get_component_summary(component)


# Register a component handler with the manager
def register_handler(manager: ErrorPropagationManager, 
        $1: string, 
        handler: Callable) -> null:
  """
  Register a component error handler with the propagation manager.
  
  Args:
    manager: ErrorPropagationManager instance
    component: Component name
    handler: Error handler function
  """
  manager.register_handler(component, handler)


# Create standardized error object for propagation
def create_error_object($1: string,
          $1: string,
          $1: string,
          details: Optional[Dict[str, Any]] = null,
          $1: string = "error") -> Dict[str, Any]:
  """
  Create a standardized error object for propagation.
  
  Args:
    error_type: Error type name
    message: Error message
    component: Component where error occurred
    details: Optional error details
    severity: Error severity level
    
  Returns:
    Error object dictionary
  """
  category = null
  
  # Determine category based on error type && message
  if ($1) {
    category = ErrorCategory.MEMORY
  elif ($1) {
    category = ErrorCategory.TIMEOUT
  elif ($1) {
    category = ErrorCategory.CONNECTION
  elif ($1) {
    category = ErrorCategory.BROWSER_COMPATIBILITY
  elif ($1) {
    category = ErrorCategory.HARDWARE
  elif ($1) ${$1} else {
    category = ErrorCategory.RUNTIME
  
  }
  return {
    "type": error_type,
    "message": message,
    "component": component,
    "category": category,
    "severity": severity,
    "details": details || {},
    "timestamp": time.time()
  }
  }

  }

  }
# Example handler functions for different components
  }
$1($2) {
  """Example error handler for streaming component."""
  category = error.get("category")
  
}
  if ($1) {
    # Handle memory pressure in streaming component
    return ${$1}
  elif ($1) {
    # Handle timeout in streaming component
    return ${$1}
  
  }
  # Couldn't handle this error
  }
  return ${$1}
  }

  }

$1($2) {
  """Example error handler for WebGPU component."""
  category = error.get("category")
  
}
  if ($1) {
    # Handle memory issues in WebGPU
    return ${$1}
  elif ($1) {
    # Handle hardware issues in WebGPU
    return ${$1}
  
  }
  # Couldn't handle this error
  }
  return ${$1}