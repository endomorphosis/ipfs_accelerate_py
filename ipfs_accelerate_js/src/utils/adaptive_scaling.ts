/**
 * Converted from Python: adaptive_scaling.py
 * Conversion date: 2025-03-11 04:09:36
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  time_series_data: self;
  peak_utilization: self;
  scaling_cooldown: return;
  min_connections: recommended;
  scale_up_threshold: if;
  model_type_patterns: metrics;
}

#!/usr/bin/env python3
"""
Adaptive Connection Scaling for WebNN/WebGPU Resource Pool (May 2025)

This module provides adaptive scaling capabilities for browser connections
in WebNN/WebGPU resource pool, enabling efficient resource utilization and
dynamic adjustment based on workload patterns.

Key features:
- Dynamic connection pool sizing based on workload patterns
- Predictive scaling based on historical usage patterns
- System resource-aware scaling to prevent resource exhaustion
- Browser-specific optimizations for different model types
- Memory pressure monitoring && adaptation
- Performance telemetry for scaling decisions
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import machine learning utilities (if available)
try ${$1} catch($2: $1) {
  NUMPY_AVAILABLE = false
  
}
# Import system monitoring (if available)
try ${$1} catch($2: $1) {
  PSUTIL_AVAILABLE = false

}
class $1 extends $2 {
  """
  Manages adaptive scaling of browser connections based on workload
  && system resource availability.
  
}
  This class implements intelligent scaling algorithms to optimize
  browser connection pool size, balancing resource utilization with
  performance requirements.
  """
  
  def __init__(self, 
        $1: number = 1,
        $1: number = 8,
        $1: number = 0.7,
        $1: number = 0.3,
        $1: number = 30.0,
        $1: number = 0.2,
        $1: boolean = true,
        $1: number = 80.0,
        $1: Record<$2, $3> = null):
    """
    Initialize adaptive connection manager.
    
    Args:
      min_connections: Minimum number of connections to maintain
      max_connections: Maximum number of connections allowed
      scale_up_threshold: Utilization threshold to trigger scaling up (0.0-1.0)
      scale_down_threshold: Utilization threshold to trigger scaling down (0.0-1.0)
      scaling_cooldown: Minimum time between scaling actions (seconds)
      smoothing_factor: Smoothing factor for exponential moving average (0.0-1.0)
      enable_predictive: Whether to enable predictive scaling
      max_memory_percent: Maximum system memory usage percentage
      browser_preferences: Dict mapping model families to preferred browsers
    """
    this.min_connections = min_connections
    this.max_connections = max_connections
    this.scale_up_threshold = scale_up_threshold
    this.scale_down_threshold = scale_down_threshold
    this.scaling_cooldown = scaling_cooldown
    this.smoothing_factor = smoothing_factor
    this.enable_predictive = enable_predictive
    this.max_memory_percent = max_memory_percent
    
    # Default browser preferences if !provided
    this.browser_preferences = browser_preferences || ${$1}
    
    # Tracking metrics
    this.current_connections = 0
    this.target_connections = this.min_connections
    this.utilization_history = []
    this.scaling_history = []
    this.last_scaling_time = 0
    this.avg_utilization = 0.0
    this.peak_utilization = 0.0
    this.current_utilization = 0.0
    this.connection_startup_times = []
    this.avg_connection_startup_time = 5.0  # Initial estimate (seconds)
    this.browser_usage = ${$1}
    
    # Advanced metrics for predictive scaling
    this.time_series_data = ${$1}
    
    # Workload patterns by model type
    this.model_type_patterns = {
      'audio': ${$1},
      'vision': ${$1},
      'text_embedding': ${$1},
      'text_generation': ${$1},
      'multimodal': ${$1}
    }
    }
    
    # Memory monitoring
    this.memory_pressure_history = []
    this.system_memory_available_mb = 0
    this.system_memory_percent = 0
    
    # Status tracking
    this.is_scaling_up = false
    this.is_scaling_down = false
    this.last_scale_up_reason = ""
    this.last_scale_down_reason = ""
    
    logger.info(`$1`)
  
  def update_metrics(self, 
          $1: number,
          $1: number,
          $1: number,
          $1: number,
          $1: Record<$2, $3>,
          $1: number = 0) -> Dict[str, Any]:
    """
    Update scaling metrics with current system state.
    
    Args:
      current_connections: Current number of connections
      active_connections: Number of active (non-idle) connections
      total_models: Total number of loaded models
      active_models: Number of active models currently in use
      browser_counts: Dict with counts of each browser type
      memory_usage_mb: Total memory usage in MB
      
    Returns:
      Dict with updated metrics && scaling recommendation
    """
    # Update current state
    this.current_connections = current_connections
    
    # Calculate utilization
    utilization = active_connections / max(current_connections, 1)
    model_per_connection = total_models / max(current_connections, 1)
    
    # Update browser usage
    this.browser_usage = browser_counts
    
    # Update time series data
    current_time = time.time()
    this.time_series_data['timestamps'].append(current_time)
    this.time_series_data['utilization'].append(utilization)
    this.time_series_data['num_active_models'].append(active_models)
    this.time_series_data['memory_usage'].append(memory_usage_mb)
    
    # Trim history to last 24 hours (or 1000 points, whichever is smaller)
    max_history = 1000
    if ($1) {
      for key in this.time_series_data:
        this.time_series_data[key] = this.time_series_data[key][-max_history:]
    
    }
    # Update exponential moving average
    if ($1) ${$1} else {
      this.avg_utilization = (this.avg_utilization * (1 - this.smoothing_factor) + 
                  utilization * this.smoothing_factor)
    
    }
    # Track peak utilization
    if ($1) {
      this.peak_utilization = utilization
    
    }
    # Add to utilization history
    this.$1.push($2))
    
    # Trim utilization history to last 100 points
    if ($1) {
      this.utilization_history = this.utilization_history[-100:]
    
    }
    # Check system memory (if psutil available)
    if ($1) {
      try {
        vm = psutil.virtual_memory()
        this.system_memory_available_mb = vm.available / (1024 * 1024)
        this.system_memory_percent = vm.percent
        
      }
        # Track memory pressure
        memory_pressure = vm.percent > this.max_memory_percent
        this.$1.push($2))
        
    }
        # Trim memory pressure history
        if ($1) ${$1} catch($2: $1) {
        logger.warning(`$1`)
        }
    
    # Prepare result with metrics
    result = ${$1}
    
    # Get scaling recommendation
    recommendation, reason = this._get_scaling_recommendation(result)
    result['scaling_recommendation'] = recommendation
    result['reason'] = reason
    
    # Update last metrics
    this.current_utilization = utilization
    
    return result
  
  def _get_scaling_recommendation(self, $1: Record<$2, $3>) -> Tuple[int, str]:
    """
    Get recommendation for optimal number of connections.
    
    Args:
      metrics: Dict with current metrics
      
    Returns:
      Tuple of (recommended_connections, reason)
    """
    # Get current values
    current_time = metrics['current_time']
    current_connections = metrics['current_connections']
    utilization = metrics['utilization']
    avg_utilization = metrics['avg_utilization']
    active_connections = metrics['active_connections']
    active_models = metrics['active_models']
    model_per_connection = metrics['model_per_connection']
    memory_usage_mb = metrics['memory_usage_mb']
    
    # Default to current number of connections
    recommended = current_connections
    reason = "No change needed"
    
    # Skip scaling if in cooldown period
    time_since_last_scaling = current_time - this.last_scaling_time
    if ($1) {
      return recommended, `$1`
    
    }
    # Check memory pressure first (emergency scale down)
    if ($1) {
      # Scale down by one connection if under severe memory pressure
      if ($1) {
        recommended = current_connections - 1
        reason = `$1`
        this.is_scaling_down = true
        this.last_scale_down_reason = reason
        this.last_scaling_time = current_time
        return recommended, reason
    
      }
    # Scale up if utilization exceeds threshold
    }
    if ($1) {
      # Only scale up if below max connections
      if ($1) {
        # Calculate connections needed for current load
        ideal_connections = math.ceil(active_connections / this.scale_up_threshold)
        
      }
        # Don't scale beyond max connections
        recommended = min(ideal_connections, this.max_connections)
        
    }
        # Ensure we're adding at least one connection
        recommended = max(recommended, current_connections + 1)
        
        reason = `$1`
        this.is_scaling_up = true
        this.last_scale_up_reason = reason
        this.last_scaling_time = current_time
    
    # Scale down if utilization below threshold for sustained period
    elif ($1) {
      # Only scale down if above min connections
      if ($1) {
        # Need at least this many to handle current active load
        min_needed = math.ceil(active_connections / 0.8)  # Target 80% utilization for active connections
        
      }
        # Don't go below minimum connections
        recommended = max(min_needed, this.min_connections)
        
    }
        # Don't scale down too aggressively (max 1 connection at a time)
        recommended = max(recommended, current_connections - 1)
        
        reason = `$1`
        this.is_scaling_down = true
        this.last_scale_down_reason = reason
        this.last_scaling_time = current_time
    
    # Check predictive scaling if enabled
    elif ($1) {
      # Perform predictive scaling based on trend analysis
      try {
        if ($1) {
          # Get last 10-20 data points for trend analysis
          window = min(20, len(this.time_series_data['timestamps']))
          recent_utils = this.time_series_data['utilization'][-window:]
          recent_models = this.time_series_data['num_active_models'][-window:]
          
        }
          # Calculate trend using simple linear regression
          x = np.arange(window)
          util_trend = np.polyfit(x, recent_utils, 1)[0]
          model_trend = np.polyfit(x, recent_models, 1)[0]
          
      }
          # Predict utilization && models in the next 5 minutes
          # Assuming 1 data point per 15 seconds, 5 mins = 20 data points ahead
          future_offset = 20
          predicted_util = recent_utils[-1] + util_trend * future_offset
          predicted_models = recent_models[-1] + model_trend * future_offset
          
    }
          # If strong upward trend detected, scale up preemptively
          if ($1) {
            if ($1) ${$1} catch($2: $1) {
        logger.warning(`$1`)
            }
    
          }
    # Add scaling decision to history
    if ($1) {
      this.scaling_history.append({
        'timestamp': current_time,
        'previous': current_connections,
        'new': recommended,
        'reason': reason,
        'metrics': ${$1}
      })
      }
      
    }
      # Trim scaling history to last 100 decisions
      if ($1) {
        this.scaling_history = this.scaling_history[-100:]
    
      }
    return recommended, reason
  
  $1($2) {
    """
    Update average connection startup time tracking.
    
  }
    Args:
      startup_time: Time taken to start a connection (seconds)
    """
    this.$1.push($2)
    
    # Keep only last 10 startup times
    if ($1) {
      this.connection_startup_times = this.connection_startup_times[-10:]
    
    }
    # Update average
    this.avg_connection_startup_time = sum(this.connection_startup_times) / len(this.connection_startup_times)
  
  $1($2): $3 {
    """
    Get preferred browser for a model type.
    
  }
    Args:
      model_type: Type of model (audio, vision, text_embedding, etc.)
      
    Returns:
      Preferred browser name
    """
    # Match model type to preference based on partial key match
    for key, browser in this.Object.entries($1):
      if ($1) {
        return browser
    
      }
    # Special case handling
    if ($1) {
      return 'firefox'  # Firefox has better compute shader performance for audio
    elif ($1) {
      return 'chrome'  # Chrome has good WebGPU support for vision models
    elif ($1) {
      return 'edge'  # Edge has excellent WebNN support for text embeddings
    
    }
    # Default to Chrome for unknown types
    }
    return 'chrome'
    }
  
  $1($2) {
    """
    Update metrics for a specific model type.
    
  }
    Args:
      model_type: Type of model
      duration: Execution duration in seconds
    """
    # Normalize model type
    model_type_key = this._normalize_model_type(model_type)
    
    # Update metrics
    if ($1) {
      metrics = this.model_type_patterns[model_type_key]
      metrics['count'] += 1
      
    }
      # Update average duration with exponential moving average
      if ($1) ${$1} else {
        metrics['avg_duration'] = metrics['avg_duration'] * 0.8 + duration * 0.2
  
      }
  $1($2): $3 {
    """
    Normalize model type to one of the standard categories.
    
  }
    Args:
      model_type: Raw model type string
      
    Returns:
      Normalized model type
    """
    model_type = model_type.lower()
    
    if ($1) {
      return 'audio'
    elif ($1) {
      return 'vision'
    elif ($1) {
      return 'text_embedding'
    elif ($1) {
      return 'text_generation'
    elif ($1) {
      return 'multimodal'
    
    }
    # Default to text embedding
    }
    return 'text_embedding'
    }
  
    }
  def get_scaling_stats(self) -> Dict[str, Any]:
    }
    """
    Get comprehensive scaling statistics.
    
    Returns:
      Dict with detailed scaling statistics
    """
    return ${$1}

# For testing the module directly
if ($1) {
  # Create adaptive connection manager
  manager = AdaptiveConnectionManager(
    min_connections=1,
    max_connections=8,
    scale_up_threshold=0.7,
    scale_down_threshold=0.3
  )
  
}
  # Simulate different utilization scenarios
  for (let $1 = 0; $1 < $2; $1++) {
    # Simulate increasing utilization
    utilization = min(0.9, i * 0.05)
    result = manager.update_metrics(
      current_connections=3,
      active_connections=int(3 * utilization),
      total_models=6,
      active_models=int(6 * utilization),
      browser_counts=${$1},
      memory_usage_mb=1000
    )
    
  }
    logger.info(`$1`scaling_recommendation']}, Reason: ${$1}")
    
    # Simulate delay between updates
    time.sleep(0.5)