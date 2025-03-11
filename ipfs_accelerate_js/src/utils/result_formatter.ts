/**
 * Converted from Python: result_formatter.py
 * Conversion date: 2025-03-11 04:09:38
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  include_metadata: metadata;
  include_raw_output: formatted_result;
  include_raw_output: formatted;
  include_metadata: error_response;
  include_metadata: result;
}

"""
Result Formatter for Unified Web Framework (August 2025)

This module provides standardized formatting for inference results across
different model types && browsers:

- Common result structure across different models
- Detailed metadata for model inference
- Performance statistics collection
- Browser-specific result formatting
- Error handling integration

Usage:
  from fixed_web_platform.unified_framework.result_formatter import (
    ResultFormatter,
    format_inference_result,
    format_error_response
  )
  
  # Create formatter for specific model type
  formatter = ResultFormatter(model_type="text")
  
  # Format raw inference result
  formatted_result = formatter.format_result(raw_result)
  
  # Add performance metrics
  formatter.add_performance_metrics(formatted_result, ${$1})
  
  # Format error response
  error_response = formatter.format_error(
    error_type="configuration_error",
    message="Invalid precision setting"
  )
"""

import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified_framework.result_formatter")

class $1 extends $2 {
  """
  Standardized result formatting for web platform inference.
  
}
  This class provides consistent formatting for inference results
  across different model types, with detailed metadata && performance
  statistics.
  """
  
  def __init__(self, 
        $1: string = "text",
        $1: $2 | null = null,
        $1: boolean = true,
        $1: boolean = false):
    """
    Initialize result formatter.
    
    Args:
      model_type: Type of model (text, vision, audio, multimodal)
      browser: Browser information for browser-specific formatting
      include_metadata: Whether to include metadata in results
      include_raw_output: Whether to include raw model output
    """
    this.model_type = model_type
    this.browser = browser
    this.include_metadata = include_metadata
    this.include_raw_output = include_raw_output
  
  def format_result(self, 
          $1: Record<$2, $3>,
          $1: $2 | null = null,
          input_summary: Optional[Dict[str, Any]] = null) -> Dict[str, Any]:
    """
    Format inference result into standardized structure.
    
    Args:
      result: Raw inference result from model
      model_name: Name of the model used
      input_summary: Summary of input data
      
    Returns:
      Formatted result dictionary
    """
    # Start with base structure
    formatted_result = ${$1}
    
    # Add metadata if enabled
    if ($1) {
      metadata = ${$1}
      
    }
      # Add input summary if provided
      if ($1) {
        metadata["input_summary"] = input_summary
        
      }
      formatted_result["metadata"] = metadata
      
    # Add raw output if enabled
    if ($1) {
      formatted_result["raw_output"] = result
    
    }
    return formatted_result
  
  def _format_output_by_type(self, result: Any) -> Dict[str, Any]:
    """
    Format output based on model type.
    
    Args:
      result: Raw result from model
      
    Returns:
      Formatted result specific to model type
    """
    # Handle dictionary results
    if ($1) {
      # Process based on model type
      if ($1) {
        return this._format_text_result(result)
      elif ($1) {
        return this._format_vision_result(result)
      elif ($1) {
        return this._format_audio_result(result)
      elif ($1) ${$1} else {
        # Default formatting for unknown types
        return result
        
      }
    # Handle string results
      }
    elif ($1) {
      return ${$1}
      
    }
    # Handle list results
      }
    elif ($1) {
      return ${$1}
      
    }
    # Return as is for other types
      }
    return ${$1}
    }
  
  def _format_text_result(self, $1: Record<$2, $3>) -> Dict[str, Any]:
    """Format text model results."""
    # Extract common text output fields
    formatted = {}
    
    if ($1) {
      formatted["text"] = result["text"]
    elif ($1) {
      formatted["text"] = result["generated_text"]
    elif ($1) {
      formatted["text"] = result["output"]
    
    }
    # Extract token counts if available
    }
    if ($1) {
      formatted["token_count"] = result["token_count"]
    
    }
    # Extract embeddings if available
    }
    if ($1) {
      formatted["embeddings"] = ${$1}
      
    }
    return formatted
  
  def _format_vision_result(self, $1: Record<$2, $3>) -> Dict[str, Any]:
    """Format vision model results."""
    # Extract common vision output fields
    formatted = {}
    
    # Handle different vision outputs
    if ($1) {
      # Classification model
      formatted["classifications"] = result["classifications"]
      
    }
    elif ($1) {
      # Object detection model
      formatted["detections"] = result.get("bounding_boxes", result.get("detections", []))
      
    }
    elif ($1) {
      # Segmentation model
      formatted["segmentation"] = ${$1}
      
    }
      if ($1) {
        formatted["segmentation"]["map"] = result["segmentation_map"]
    
      }
    # Extract embeddings if available
    if ($1) {
      formatted["embeddings"] = ${$1}
      
    }
    return formatted
  
  def _format_audio_result(self, $1: Record<$2, $3>) -> Dict[str, Any]:
    """Format audio model results."""
    # Extract common audio output fields
    formatted = {}
    
    # Handle different audio outputs
    if ($1) {
      # Speech recognition model
      formatted["transcription"] = result["transcription"]
      
    }
    elif ($1) {
      # Audio classification model
      formatted["classifications"] = result["classification"]
      
    }
    elif ($1) {
      # Audio embedding model
      formatted["embeddings"] = ${$1}
    
    }
    return formatted
  
  def _format_multimodal_result(self, $1: Record<$2, $3>) -> Dict[str, Any]:
    """Format multimodal model results."""
    # Extract common multimodal output fields
    formatted = {}
    
    # Handle different multimodal outputs
    if ($1) {
      # Text output from multimodal model
      formatted["text"] = result.get("text", result.get("generated_text", ""))
      
    }
    if ($1) {
      # Multimodal embeddings
      formatted["embeddings"] = {
        "visual": ${$1},
        "text": ${$1}
      }
      }
    
    }
    return formatted
  
  def add_performance_metrics(self, $1: Record<$2, $3>, 
              $1: Record<$2, $3>) -> Dict[str, Any]:
    """
    Add performance metrics to formatted result.
    
    Args:
      result: Formatted result dictionary
      metrics: Performance metrics to add
      
    Returns:
      Updated result dictionary with performance metrics
    """
    # Create performance section if it doesn't exist
    if ($1) {
      result["performance"] = {}
      
    }
    # Process common metrics
    if ($1) {
      result["performance"]["inference_time_ms"] = metrics["inference_time_ms"]
      
    }
    if ($1) {
      result["performance"]["preprocessing_time_ms"] = metrics["preprocessing_time_ms"]
      
    }
    if ($1) {
      result["performance"]["postprocessing_time_ms"] = metrics["postprocessing_time_ms"]
      
    }
    # Calculate total time if components are available
    if ($1) {
      result["performance"]["total_time_ms"] = (
        result["performance"]["inference_time_ms"] +
        result["performance"]["preprocessing_time_ms"] +
        result["performance"]["postprocessing_time_ms"]
      )
      
    }
    # Add text generation metrics
    if ($1) {
      result["performance"]["tokens_per_second"] = metrics["tokens_per_second"]
      
    }
    # Add memory usage metrics
    if ($1) {
      result["performance"]["peak_memory_mb"] = metrics["peak_memory_mb"]
      
    }
    # Add browser-specific metrics
    if ($1) {
      result["performance"]["browser"] = metrics["browser_metrics"]
      
    }
    return result
    
  def format_error(self, 
          $1: string, 
          $1: string, 
          details: Optional[Dict[str, Any]] = null) -> Dict[str, Any]:
    """
    Format error response.
    
    Args:
      error_type: Type of error
      message: Error message
      details: Optional error details
      
    Returns:
      Formatted error response
    """
    error_response = {
      "success": false,
      "timestamp": time.time(),
      "error": ${$1}
    }
    }
    
    # Add details if provided
    if ($1) {
      error_response["error"]["details"] = details
      
    }
    # Add metadata if enabled
    if ($1) {
      error_response["metadata"] = ${$1}
      
    }
    return error_response
  
  def create_progressive_result(self) -> Dict[str, Any]:
    """
    Create an empty result structure for progressive updates.
    
    Returns:
      Empty result dictionary for progressive updates
    """
    result = {
      "success": true,
      "timestamp": time.time(),
      "result": {},
      "complete": false,
      "progress": 0.0
    }
    }
    
    # Add metadata if enabled
    if ($1) {
      result["metadata"] = ${$1}
      
    }
    return result
  
  def update_progressive_result(self, 
                $1: Record<$2, $3>,
                $1: Record<$2, $3>,
                $1: number) -> Dict[str, Any]:
    """
    Update progressive result with new data.
    
    Args:
      result: Progressive result to update
      update: New data to add
      progress: Current progress (0.0-1.0)
      
    Returns:
      Updated progressive result
    """
    # Update progress
    result["progress"] = progress
    result["timestamp"] = time.time()
    
    # Merge new data into result
    if ($1) ${$1} else {
      # Assume update is the result data directly
      result["result"].update(update)
      
    }
    # Mark as complete if progress is 100%
    if ($1) {
      result["complete"] = true
      
    }
    return result
  
  @classmethod
  def merge_results(cls, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple formatted results into a single result.
    
    Args:
      results: List of formatted results to merge
      
    Returns:
      Merged result dictionary
    """
    if ($1) {
      return {"success": false, "error": ${$1}}
      
    }
    # Start with the first result as base
    merged = results[0].copy()
    
    # Track if any result failed
    all_succeeded = all(result.get("success", false) for result in results)
    merged["success"] = all_succeeded
    
    # Merge result data
    for result in results[1:]:
      if ($1) {
        merged["result"].update(result["result"])
        
      }
    # Merge performance metrics
    if ($1) {
      for result in results[1:]:
        if ($1) {
          for key, value in result["performance"].items():
            if ($1) {
              # Average numeric values
              if ($1) ${$1} else {
              # Add new metrics
              }
              merged["performance"][key] = value
              
            }
    return merged
        }

    }

# Utility functions

def format_inference_result($1: Record<$2, $3>, 
            $1: string = "text",
            $1: $2 | null = null,
            $1: $2 | null = null,
            $1: boolean = true) -> Dict[str, Any]:
  """
  Format inference result with standard utility function.
  
  Args:
    result: Raw inference result
    model_type: Type of model
    model_name: Name of model
    browser: Browser information
    include_metadata: Whether to include metadata
    
  Returns:
    Formatted result dictionary
  """
  formatter = ResultFormatter(
    model_type=model_type,
    browser=browser,
    include_metadata=include_metadata
  )
  
  return formatter.format_result(result, model_name)


def format_error_response($1: string,
            $1: string,
            details: Optional[Dict[str, Any]] = null,
            $1: string = "text",
            $1: $2 | null = null) -> Dict[str, Any]:
  """
  Format error response with standard utility function.
  
  Args:
    error_type: Type of error
    message: Error message
    details: Optional error details
    model_type: Type of model
    browser: Browser information
    
  Returns:
    Formatted error response
  """
  formatter = ResultFormatter(
    model_type=model_type,
    browser=browser
  )
  
  return formatter.format_error(error_type, message, details)


def parse_raw_output(raw_output: Any, $1: string = "text") -> Dict[str, Any]:
  """
  Parse raw model output into structured format.
  
  Args:
    raw_output: Raw output from model inference
    model_type: Type of model
    
  Returns:
    Parsed && structured output
  """
  # Create appropriate formatter
  formatter = ResultFormatter(model_type=model_type)
  
  # Format the output
  formatted = formatter._format_output_by_type(raw_output)
  
  return formatted