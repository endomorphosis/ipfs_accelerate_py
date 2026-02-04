#!/usr/bin/env python3
"""
Resource Pool Bridge Recovery System

This module provides enhanced error recovery mechanisms for the WebNN/WebGPU
Resource Pool Bridge Integration, improving reliability when operating with real
browser instances. It offers graceful fallbacks, automatic reconnection, and
performance-based recovery strategies.

Key features:
- Automatic connection recovery after browser crashes
- Performance-aware recovery strategies
- Browser capability detection and fallback
- Adaptive error handling based on failure types
- Comprehensive logging and telemetry

Usage:
    from resource_pool_bridge_recovery import ResourcePoolBridgeRecovery

    # Create recovery manager
    recovery = ResourcePoolBridgeRecovery(integration=integration)
    
    # Use with existing code
    result = recovery.execute_safely(lambda: integration.run_inference(...))
"""

import os
import time
import json
import logging
import random
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types for handling different failure scenarios."""
    IMMEDIATE_RETRY = "immediate_retry"  # Immediately retry without changes
    DELAY_RETRY = "delay_retry"  # Retry after a delay
    BROWSER_RESTART = "browser_restart"  # Restart the browser and retry
    ALTERNATIVE_BROWSER = "alternative_browser"  # Try a different browser
    ALTERNATIVE_PLATFORM = "alternative_platform"  # Try a different platform (WebNN/WebGPU)
    SIMULATION_FALLBACK = "simulation_fallback"  # Fall back to CPU simulation
    REDUCE_MODEL_SIZE = "reduce_model_size"  # Try a smaller model variant
    REDUCE_PRECISION = "reduce_precision"  # Lower precision for better compatibility
    ALLOCATE_MORE_RESOURCES = "allocate_more_resources"  # Increase resource allocation


class ErrorCategory(Enum):
    """Categories of errors for better diagnosis and recovery."""
    CONNECTION = "connection"  # WebSocket or browser connection issues
    BROWSER_CRASH = "browser_crash"  # Browser process crashed
    OUT_OF_MEMORY = "out_of_memory"  # OOM errors
    TIMEOUT = "timeout"  # Operation timed out
    UNSUPPORTED_OPERATION = "unsupported_operation"  # Operation not supported
    BROWSER_CAPABILITY = "browser_capability"  # Browser lacks capability
    MODEL_INCOMPATIBLE = "model_incompatible"  # Model incompatible with backend
    INTERNAL_ERROR = "internal_error"  # Internal implementation errors
    UNKNOWN = "unknown"  # Unknown error types


class ResourcePoolBridgeRecovery:
    """
    Enhanced error recovery system for WebNN/WebGPU Resource Pool Bridge.
    
    This class provides fault-tolerance and resilient operation for the WebNN/WebGPU
    integration, particularly when working with real browser instances that may crash,
    have inconsistent behavior, or encounter capability limitations.
    """
    
    def __init__(self, 
                 integration: Optional[Any] = None,
                 max_retries: int = 3, 
                 retry_delay: float = 2.0,
                 connection_timeout: float = 30.0,
                 performance_threshold: float = 0.5,
                 fallback_to_simulation: bool = True,
                 browser_preferences: Optional[Dict[str, str]] = None):
        """
        Initialize recovery manager.
        
        Args:
            integration: Existing ResourcePoolBridgeIntegration instance or None
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (will use exponential backoff)
            connection_timeout: Connection timeout in seconds
            performance_threshold: Relative performance threshold to trigger recovery
            fallback_to_simulation: Whether to allow fallback to simulation
            browser_preferences: Browser preferences by model type
        """
        self.integration = integration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_timeout = connection_timeout
        self.performance_threshold = performance_threshold
        self.fallback_to_simulation = fallback_to_simulation
        
        # Set default browser preferences if not provided
        self.browser_preferences = browser_preferences or {
            "text": "edge",  # Edge works best for WebNN and text models
            "vision": "chrome",  # Chrome works well for WebGPU and vision models 
            "audio": "firefox",  # Firefox has optimized compute shaders for audio
            "multimodal": "chrome"  # Chrome works well for multimodal models
        }
        
        # Internal recovery state
        self._lock = threading.RLock()
        self._recovery_history = []
        self._browser_health = {
            "chrome": {"success": 0, "errors": 0, "health_score": 1.0},
            "firefox": {"success": 0, "errors": 0, "health_score": 1.0},
            "edge": {"success": 0, "errors": 0, "health_score": 1.0},
            "safari": {"success": 0, "errors": 0, "health_score": 1.0}
        }
        self._platform_health = {
            "webgpu": {"success": 0, "errors": 0, "health_score": 1.0},
            "webnn": {"success": 0, "errors": 0, "health_score": 1.0},
            "cpu": {"success": 0, "errors": 0, "health_score": 1.0}
        }
        
        # Recovery strategy mappings
        self._error_strategy_map = self._create_error_strategy_map()
        
        # Performance baselines for detecting degradation
        self._performance_baselines = {}
        
        logger.info(f"ResourcePoolBridgeRecovery initialized with max_retries={max_retries}")
    
    def _create_error_strategy_map(self) -> Dict[ErrorCategory, List[RecoveryStrategy]]:
        """Create mapping from error categories to recovery strategies in priority order."""
        return {
            ErrorCategory.CONNECTION: [
                RecoveryStrategy.DELAY_RETRY,
                RecoveryStrategy.BROWSER_RESTART,
                RecoveryStrategy.ALTERNATIVE_BROWSER,
                RecoveryStrategy.SIMULATION_FALLBACK
            ],
            ErrorCategory.BROWSER_CRASH: [
                RecoveryStrategy.BROWSER_RESTART,
                RecoveryStrategy.ALTERNATIVE_BROWSER,
                RecoveryStrategy.SIMULATION_FALLBACK
            ],
            ErrorCategory.OUT_OF_MEMORY: [
                RecoveryStrategy.REDUCE_MODEL_SIZE,
                RecoveryStrategy.REDUCE_PRECISION,
                RecoveryStrategy.ALTERNATIVE_BROWSER,
                RecoveryStrategy.ALTERNATIVE_PLATFORM,
                RecoveryStrategy.SIMULATION_FALLBACK
            ],
            ErrorCategory.TIMEOUT: [
                RecoveryStrategy.DELAY_RETRY,
                RecoveryStrategy.BROWSER_RESTART,
                RecoveryStrategy.ALTERNATIVE_BROWSER,
                RecoveryStrategy.SIMULATION_FALLBACK
            ],
            ErrorCategory.UNSUPPORTED_OPERATION: [
                RecoveryStrategy.ALTERNATIVE_PLATFORM,
                RecoveryStrategy.ALTERNATIVE_BROWSER,
                RecoveryStrategy.REDUCE_PRECISION,
                RecoveryStrategy.SIMULATION_FALLBACK
            ],
            ErrorCategory.BROWSER_CAPABILITY: [
                RecoveryStrategy.ALTERNATIVE_BROWSER,
                RecoveryStrategy.ALTERNATIVE_PLATFORM,
                RecoveryStrategy.SIMULATION_FALLBACK
            ],
            ErrorCategory.MODEL_INCOMPATIBLE: [
                RecoveryStrategy.ALTERNATIVE_PLATFORM,
                RecoveryStrategy.REDUCE_PRECISION,
                RecoveryStrategy.REDUCE_MODEL_SIZE,
                RecoveryStrategy.SIMULATION_FALLBACK
            ],
            ErrorCategory.INTERNAL_ERROR: [
                RecoveryStrategy.DELAY_RETRY,
                RecoveryStrategy.BROWSER_RESTART,
                RecoveryStrategy.SIMULATION_FALLBACK
            ],
            ErrorCategory.UNKNOWN: [
                RecoveryStrategy.DELAY_RETRY,
                RecoveryStrategy.BROWSER_RESTART,
                RecoveryStrategy.ALTERNATIVE_BROWSER,
                RecoveryStrategy.SIMULATION_FALLBACK
            ]
        }
    
    def categorize_error(self, error: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """
        Determine the category of an error for appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Context information about the operation
            
        Returns:
            ErrorCategory indicating error type
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Check for connection issues
        if any(term in error_str for term in ["connection", "websocket", "closed", "connect", "disconnect", "handshake"]):
            return ErrorCategory.CONNECTION
        
        # Check for browser crashes
        if any(term in error_str for term in ["crashed", "terminated", "killed", "browser closed", "not responding"]):
            return ErrorCategory.BROWSER_CRASH
        
        # Check for out of memory errors
        if any(term in error_str for term in ["out of memory", "oom", "memory limit", "insufficient memory"]):
            return ErrorCategory.OUT_OF_MEMORY
        
        # Check for timeouts
        if any(term in error_str for term in ["timeout", "timed out", "took too long"]):
            return ErrorCategory.TIMEOUT
        
        # Check for unsupported operations
        if any(term in error_str for term in ["unsupported", "not supported", "not implemented", "operation not allowed"]):
            return ErrorCategory.UNSUPPORTED_OPERATION
        
        # Check for browser capability issues
        if any(term in error_str for term in ["webgpu not supported", "webnn not supported", "feature not available"]):
            return ErrorCategory.BROWSER_CAPABILITY
        
        # Check for model incompatibility
        if any(term in error_str for term in ["model incompatible", "incompatible model", "not compatible"]):
            return ErrorCategory.MODEL_INCOMPATIBLE
        
        # Check for internal errors
        if any(term in error_str for term in ["internal error", "implementation error", "assertion", "unexpected"]):
            return ErrorCategory.INTERNAL_ERROR
        
        # Default to unknown
        logger.debug(f"Could not categorize error: {error_str} (type: {error_type})")
        return ErrorCategory.UNKNOWN
    
    def determine_recovery_strategy(self, 
                                   error_category: ErrorCategory, 
                                   context: Dict[str, Any],
                                   attempt: int) -> RecoveryStrategy:
        """
        Determine the best recovery strategy for an error.
        
        Args:
            error_category: The categorized error
            context: Context information about the operation
            attempt: Current retry attempt number
            
        Returns:
            RecoveryStrategy to apply
        """
        # Get strategies for this error category
        strategies = self._error_strategy_map.get(error_category, 
                                                 self._error_strategy_map[ErrorCategory.UNKNOWN])
        
        # Use more aggressive strategies as attempt count increases
        strategy_index = min(attempt, len(strategies) - 1)
        strategy = strategies[strategy_index]
        
        # Special cases based on context
        model_type = context.get("model_type")
        platform = context.get("platform")
        browser = context.get("browser")
        
        # Check platform and browser health
        if platform and platform in self._platform_health:
            platform_health = self._platform_health[platform]["health_score"]
            # If platform health is poor, prefer alternative platform
            if platform_health < 0.3 and strategy != RecoveryStrategy.ALTERNATIVE_PLATFORM:
                logger.info(f"Platform {platform} health score is low ({platform_health:.2f}), prioritizing platform switch")
                return RecoveryStrategy.ALTERNATIVE_PLATFORM
        
        if browser and browser in self._browser_health:
            browser_health = self._browser_health[browser]["health_score"]
            # If browser health is poor, prefer alternative browser
            if browser_health < 0.3 and strategy != RecoveryStrategy.ALTERNATIVE_BROWSER:
                logger.info(f"Browser {browser} health score is low ({browser_health:.2f}), prioritizing browser switch")
                return RecoveryStrategy.ALTERNATIVE_BROWSER
        
        # For large models, prioritize memory-related strategies for OOM errors
        if error_category == ErrorCategory.OUT_OF_MEMORY:
            model_size = context.get("model_size", "medium")
            if model_size in ["large", "xlarge"]:
                return RecoveryStrategy.REDUCE_MODEL_SIZE
            else:
                return RecoveryStrategy.REDUCE_PRECISION
        
        # For audio models on browsers other than Firefox, prefer Firefox
        if (model_type == "audio" and 
            browser and browser != "firefox" and 
            error_category in [ErrorCategory.TIMEOUT, ErrorCategory.UNSUPPORTED_OPERATION]):
            logger.info(f"Audio model on {browser} encountered {error_category}, switching to Firefox")
            return RecoveryStrategy.ALTERNATIVE_BROWSER
        
        return strategy
    
    def apply_recovery_strategy(self, 
                               strategy: RecoveryStrategy, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a recovery strategy to modify the context for retry.
        
        Args:
            strategy: The strategy to apply
            context: Original context that will be modified
            
        Returns:
            Modified context with recovery strategy applied
        """
        # Create a copy of the context to modify
        new_context = context.copy()
        
        # Apply the selected strategy
        if strategy == RecoveryStrategy.IMMEDIATE_RETRY:
            # No changes needed, just retry
            pass
            
        elif strategy == RecoveryStrategy.DELAY_RETRY:
            # No context changes, but will delay before retry
            # Delay is handled in the execute_safely method
            pass
            
        elif strategy == RecoveryStrategy.BROWSER_RESTART:
            # Signal to restart the browser
            new_context["restart_browser"] = True
            
        elif strategy == RecoveryStrategy.ALTERNATIVE_BROWSER:
            current_browser = context.get("browser", "chrome")
            model_type = context.get("model_type", "text")
            
            # Select alternative browser based on model type and health scores
            browsers = ["chrome", "firefox", "edge"]
            browsers.remove(current_browser) if current_browser in browsers else None
            
            # Sort remaining browsers by health score
            browsers.sort(key=lambda b: self._browser_health[b]["health_score"], reverse=True)
            
            # Adjust based on model type preferences
            if model_type == "audio" and "firefox" in browsers:
                # Firefox is best for audio models
                new_browser = "firefox"
            elif model_type == "text" and "edge" in browsers:
                # Edge is good for text models with WebNN
                new_browser = "edge"
            elif browsers:
                # Use healthiest remaining browser
                new_browser = browsers[0]
            else:
                # Fallback to original
                new_browser = current_browser
                
            new_context["browser"] = new_browser
            logger.info(f"Switching from {current_browser} to {new_browser}")
            
        elif strategy == RecoveryStrategy.ALTERNATIVE_PLATFORM:
            current_platform = context.get("platform", "webgpu")
            
            # Switch between WebGPU and WebNN
            if current_platform == "webgpu":
                new_platform = "webnn"
            elif current_platform == "webnn":
                new_platform = "webgpu"
            else:
                # If neither, default to webgpu
                new_platform = "webgpu"
                
            # Update priority list if it exists
            if "hardware_preferences" in new_context and "priority_list" in new_context["hardware_preferences"]:
                priority_list = new_context["hardware_preferences"]["priority_list"]
                if current_platform in priority_list:
                    # Move the new platform to the front of the list
                    priority_list = [p for p in priority_list if p != new_platform]
                    priority_list.insert(0, new_platform)
                    new_context["hardware_preferences"]["priority_list"] = priority_list
            
            new_context["platform"] = new_platform
            logger.info(f"Switching from {current_platform} to {new_platform}")
            
        elif strategy == RecoveryStrategy.SIMULATION_FALLBACK:
            if not self.fallback_to_simulation:
                logger.warning("Simulation fallback requested but not allowed by configuration")
                return new_context
                
            # Enable simulation flag
            new_context["simulation"] = True
            new_context["hardware_preferences"] = new_context.get("hardware_preferences", {})
            new_context["hardware_preferences"]["fallback_to_simulation"] = True
            
            # Set device to CPU
            if "hardware_preferences" in new_context:
                if "priority_list" in new_context["hardware_preferences"]:
                    new_context["hardware_preferences"]["priority_list"] = ["cpu"]
                new_context["hardware_preferences"]["device"] = "cpu"
            
            logger.info(f"Falling back to CPU simulation mode")
            
        elif strategy == RecoveryStrategy.REDUCE_MODEL_SIZE:
            model_name = context.get("model_name", "")
            
            # Check if there's a smaller model variant
            if "large" in model_name:
                new_model_name = model_name.replace("large", "base")
            elif "medium" in model_name:
                new_model_name = model_name.replace("medium", "small")
            elif "base" in model_name:
                new_model_name = model_name.replace("base", "small")
            elif "small" in model_name:
                new_model_name = model_name.replace("small", "tiny")
            elif "-7b" in model_name:
                new_model_name = model_name.replace("-7b", "-3b")
            elif "-13b" in model_name:
                new_model_name = model_name.replace("-13b", "-7b")
            elif "-70b" in model_name:
                new_model_name = model_name.replace("-70b", "-13b")
            else:
                # No obvious smaller variant, try to find a -tiny suffix
                parts = model_name.split("/")
                if len(parts) > 1:
                    base = parts[0]
                    model = parts[1]
                    if "-tiny" not in model:
                        new_model_name = f"{base}/{model}-tiny"
                    else:
                        # Already tiny, can't reduce further
                        new_model_name = model_name
                else:
                    # Single part name, try adding -tiny
                    if "-tiny" not in model_name:
                        new_model_name = f"{model_name}-tiny"
                    else:
                        # Already tiny, can't reduce further
                        new_model_name = model_name
            
            # Only update if we found a smaller variant
            if new_model_name != model_name:
                new_context["model_name"] = new_model_name
                logger.info(f"Reducing model size from {model_name} to {new_model_name}")
            else:
                logger.warning(f"Could not find smaller variant for {model_name}")
                
        elif strategy == RecoveryStrategy.REDUCE_PRECISION:
            # Get current precision settings
            hardware_prefs = new_context.get("hardware_preferences", {})
            current_precision = hardware_prefs.get("precision", 16)
            
            # Reduce precision if possible
            if current_precision == 16:
                new_precision = 8
            elif current_precision == 8:
                new_precision = 4
            else:
                # Already at or below 4-bit, don't reduce further
                new_precision = current_precision
            
            # Update precision
            if "hardware_preferences" not in new_context:
                new_context["hardware_preferences"] = {}
            new_context["hardware_preferences"]["precision"] = new_precision
            
            # Enable mixed precision if reducing to 4-bit
            if new_precision == 4:
                new_context["hardware_preferences"]["mixed_precision"] = True
                
            logger.info(f"Reducing precision from {current_precision}-bit to {new_precision}-bit")
            
        elif strategy == RecoveryStrategy.ALLOCATE_MORE_RESOURCES:
            # Increase resource allocation if possible
            # This might involve increasing connection count, browser instances, etc.
            logger.info("Allocating more resources for recovery")
            
            # In a real implementation, this would adjust resource allocation parameters
            # For now, we just set a flag to indicate this was attempted
            new_context["increased_resources"] = True
            
        return new_context
    
    def execute_safely(self, 
                      operation: Callable[[], Any], 
                      context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any, Dict[str, Any]]:
        """
        Execute an operation with automatic recovery on failure.
        
        Args:
            operation: Function to execute
            context: Context information about the operation
            
        Returns:
            Tuple of (success, result, final_context)
        """
        context = context or {}
        current_context = context.copy()
        attempt = 0
        start_time = time.time()
        result = None
        error = None
        
        while attempt < self.max_retries:
            try:
                # Apply restart if requested from previous recovery
                if current_context.get("restart_browser", False):
                    self._restart_browser(current_context)
                    # Clear the flag to avoid repeated restarts
                    current_context["restart_browser"] = False
                
                # Execute the operation
                logger.debug(f"Executing operation (attempt {attempt+1}/{self.max_retries})")
                result = operation()
                
                # Check for error in result if it's a dict
                if isinstance(result, dict) and not result.get("success", True):
                    error_msg = result.get("error", "Unknown error in result")
                    raise RuntimeError(f"Operation failed: {error_msg}")
                
                # Record success for health tracking
                self._record_success(current_context)
                
                # Update performance baseline if needed
                self._update_performance_baseline(result, current_context)
                
                # Operation succeeded
                return True, result, current_context
                
            except Exception as e:
                # Record failure
                self._record_failure(current_context, e)
                
                # Save the error for return value
                error = e
                
                # Determine error category and recovery strategy
                error_category = self.categorize_error(e, current_context)
                recovery_strategy = self.determine_recovery_strategy(error_category, current_context, attempt)
                
                # Log recovery attempt
                logger.warning(f"Operation failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                logger.info(f"Categorized as {error_category.value}, applying {recovery_strategy.value}")
                
                # Record recovery attempt
                self._record_recovery_attempt(error_category, recovery_strategy, current_context)
                
                # Apply recovery strategy
                current_context = self.apply_recovery_strategy(recovery_strategy, current_context)
                
                # Check if we should delay before retry
                if recovery_strategy == RecoveryStrategy.DELAY_RETRY:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Delaying {delay:.2f}s before retry")
                    time.sleep(delay)
                
                # Increment attempt counter
                attempt += 1
        
        # All retries failed
        elapsed = time.time() - start_time
        logger.error(f"Operation failed after {attempt} attempts ({elapsed:.2f}s): {str(error)}")
        
        # Final context includes error information
        final_context = current_context.copy()
        final_context["error"] = str(error)
        final_context["error_type"] = type(error).__name__
        final_context["attempts"] = attempt
        final_context["elapsed_time"] = elapsed
        
        # If simulation fallback was used, make that clear in the result
        result = result or {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "simulation_fallback_attempted": current_context.get("simulation", False),
            "recovery_attempted": True,
            "attempts": attempt
        }
        
        return False, result, final_context
    
    def _record_success(self, context: Dict[str, Any]):
        """Record successful operation for health tracking."""
        with self._lock:
            browser = context.get("browser")
            if browser and browser in self._browser_health:
                self._browser_health[browser]["success"] += 1
                # Update health score as success/(success+errors)
                success = self._browser_health[browser]["success"]
                errors = self._browser_health[browser]["errors"]
                self._browser_health[browser]["health_score"] = success / max(1, success + errors)
                
            platform = context.get("platform")
            if platform and platform in self._platform_health:
                self._platform_health[platform]["success"] += 1
                # Update health score
                success = self._platform_health[platform]["success"]
                errors = self._platform_health[platform]["errors"]
                self._platform_health[platform]["health_score"] = success / max(1, success + errors)
    
    def _record_failure(self, context: Dict[str, Any], error: Exception):
        """Record operation failure for health tracking."""
        with self._lock:
            browser = context.get("browser")
            if browser and browser in self._browser_health:
                self._browser_health[browser]["errors"] += 1
                # Update health score as success/(success+errors)
                success = self._browser_health[browser]["success"]
                errors = self._browser_health[browser]["errors"]
                self._browser_health[browser]["health_score"] = success / max(1, success + errors)
                
            platform = context.get("platform")
            if platform and platform in self._platform_health:
                self._platform_health[platform]["errors"] += 1
                # Update health score
                success = self._platform_health[platform]["success"]
                errors = self._platform_health[platform]["errors"]
                self._platform_health[platform]["health_score"] = success / max(1, success + errors)
    
    def _record_recovery_attempt(self, error_category: ErrorCategory, 
                                strategy: RecoveryStrategy, 
                                context: Dict[str, Any]):
        """Record recovery attempt for analysis."""
        with self._lock:
            # Keep track of all recovery attempts
            attempt_record = {
                "timestamp": time.time(),
                "error_category": error_category.value,
                "recovery_strategy": strategy.value,
                "browser": context.get("browser"),
                "platform": context.get("platform"),
                "model_type": context.get("model_type"),
                "model_name": context.get("model_name")
            }
            
            self._recovery_history.append(attempt_record)
            
            # Limit history length to avoid memory growth
            if len(self._recovery_history) > 1000:
                self._recovery_history = self._recovery_history[-1000:]
    
    def _update_performance_baseline(self, result: Any, context: Dict[str, Any]):
        """Update performance baseline if result contains metrics."""
        if not isinstance(result, dict) or "metrics" not in result:
            return
        
        metrics = result.get("metrics", {})
        if not metrics:
            return
            
        model_name = context.get("model_name")
        if not model_name:
            return
            
        # Get key performance metrics
        latency = metrics.get("latency_ms")
        throughput = metrics.get("throughput_items_per_sec")
        memory = metrics.get("memory_usage_mb")
        
        if not (latency or throughput):
            return
            
        with self._lock:
            # Initialize baseline if not exists
            if model_name not in self._performance_baselines:
                self._performance_baselines[model_name] = {
                    "count": 0,
                    "avg_latency_ms": 0,
                    "avg_throughput": 0,
                    "avg_memory_mb": 0
                }
                
            baseline = self._performance_baselines[model_name]
            count = baseline["count"]
            
            # Update using exponential moving average
            alpha = 1.0 / (count + 1) if count < 10 else 0.1
            
            if latency:
                baseline["avg_latency_ms"] = (1 - alpha) * baseline["avg_latency_ms"] + alpha * latency
                
            if throughput:
                baseline["avg_throughput"] = (1 - alpha) * baseline["avg_throughput"] + alpha * throughput
                
            if memory:
                baseline["avg_memory_mb"] = (1 - alpha) * baseline["avg_memory_mb"] + alpha * memory
                
            baseline["count"] += 1
    
    def _check_performance_degradation(self, result: Any, context: Dict[str, Any]) -> bool:
        """Check if performance has degraded significantly compared to baseline."""
        if not isinstance(result, dict) or "metrics" not in result:
            return False
            
        metrics = result.get("metrics", {})
        if not metrics:
            return False
            
        model_name = context.get("model_name")
        if not model_name or model_name not in self._performance_baselines:
            return False
            
        baseline = self._performance_baselines[model_name]
        if baseline["count"] < 5:  # Need enough samples for reliable baseline
            return False
            
        # Check latency degradation
        latency = metrics.get("latency_ms")
        if latency and baseline["avg_latency_ms"] > 0:
            latency_ratio = latency / baseline["avg_latency_ms"]
            if latency_ratio > 2.0:  # 2x slowdown
                logger.warning(f"Performance degradation detected: latency {latency}ms vs baseline {baseline['avg_latency_ms']:.2f}ms")
                return True
                
        # Check throughput degradation
        throughput = metrics.get("throughput_items_per_sec")
        if throughput and baseline["avg_throughput"] > 0:
            throughput_ratio = baseline["avg_throughput"] / throughput
            if throughput_ratio > 2.0:  # 2x slowdown
                logger.warning(f"Performance degradation detected: throughput {throughput} vs baseline {baseline['avg_throughput']:.2f}")
                return True
                
        return False
    
    def _restart_browser(self, context: Dict[str, Any]):
        """Restart the browser instance."""
        browser = context.get("browser", "chrome")
        logger.info(f"Restarting {browser} browser")
        
        # If integration has browser restart capability, use it
        if self.integration and hasattr(self.integration, "restart_browser"):
            try:
                self.integration.restart_browser(browser)
                logger.info(f"Successfully restarted {browser} browser")
                return
            except Exception as e:
                logger.warning(f"Error restarting browser through integration: {e}")
                # Continue with fallback approach
        
        # Fallback: try to close and reinitialize
        if self.integration:
            try:
                # Attempt to close gracefully
                if hasattr(self.integration, "close_browser"):
                    self.integration.close_browser(browser)
                elif hasattr(self.integration, "close"):
                    # Generic close might close all browsers
                    self.integration.close()
                    
                # Reinitialize
                if hasattr(self.integration, "initialize"):
                    self.integration.initialize()
                    logger.info(f"Reinitialized integration after browser restart")
                    
            except Exception as e:
                logger.warning(f"Error in fallback browser restart: {e}")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery attempts and success rates."""
        with self._lock:
            # Count recovery attempts by category and strategy
            category_counts = {}
            strategy_counts = {}
            browser_counts = {}
            platform_counts = {}
            total_attempts = len(self._recovery_history)
            
            for attempt in self._recovery_history:
                # Count by error category
                category = attempt["error_category"]
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Count by strategy
                strategy = attempt["recovery_strategy"]
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                
                # Count by browser
                browser = attempt.get("browser")
                if browser:
                    browser_counts[browser] = browser_counts.get(browser, 0) + 1
                    
                # Count by platform
                platform = attempt.get("platform")
                if platform:
                    platform_counts[platform] = platform_counts.get(platform, 0) + 1
            
            # Get health scores
            browser_health = {b: self._browser_health[b]["health_score"] for b in self._browser_health}
            platform_health = {p: self._platform_health[p]["health_score"] for p in self._platform_health}
            
            return {
                "total_recovery_attempts": total_attempts,
                "error_categories": category_counts,
                "recovery_strategies": strategy_counts,
                "browser_recovery_counts": browser_counts,
                "platform_recovery_counts": platform_counts,
                "browser_health_scores": browser_health,
                "platform_health_scores": platform_health,
                "performance_baselines_count": len(self._performance_baselines)
            }


class ResourcePoolBridgeWithRecovery:
    """
    Enhanced WebNN/WebGPU Resource Pool Bridge Integration with automatic recovery.
    
    This class wraps an existing or mock ResourcePoolBridgeIntegration with the
    recovery system, providing resilient operation for web browser-based inference.
    """
    
    def __init__(self, 
                 integration=None, 
                 max_connections=4, 
                 browser_preferences=None,
                 max_retries=3,
                 fallback_to_simulation=True):
        """
        Initialize integrated resource pool with recovery.
        
        Args:
            integration: Existing ResourcePoolBridgeIntegration or None to create mock
            max_connections: Maximum browser connections to maintain
            browser_preferences: Browser preferences by model type
            max_retries: Maximum retry attempts for operations
            fallback_to_simulation: Whether to allow fallback to CPU simulation
        """
        # Create mock integration if none provided
        self.mock_mode = integration is None
        
        if self.mock_mode:
            # Import locally to avoid circular imports
            try:
                from resource_pool_bridge_test import MockResourcePoolBridgeIntegration
                self.integration = MockResourcePoolBridgeIntegration(max_connections=max_connections)
                logger.info("Created mock ResourcePoolBridgeIntegration")
            except ImportError:
                # Create very basic mock if import fails
                self.integration = self._create_basic_mock(max_connections)
                logger.info("Created basic mock integration (MockResourcePoolBridgeIntegration not available)")
        else:
            # Use provided integration
            self.integration = integration
            logger.info("Using provided ResourcePoolBridgeIntegration")
        
        # Create recovery manager
        self.recovery = ResourcePoolBridgeRecovery(
            integration=self.integration,
            max_retries=max_retries,
            browser_preferences=browser_preferences,
            fallback_to_simulation=fallback_to_simulation
        )
        
        # Track loaded models
        self.loaded_models = {}
        
        logger.info(f"ResourcePoolBridgeWithRecovery initialized (mock_mode={self.mock_mode})")
    
    def _create_basic_mock(self, max_connections):
        """Create a basic mock integration if the test module is not available."""
        class BasicMockIntegration:
            def __init__(self, max_connections):
                self.max_connections = max_connections
                self.initialized = False
                self.models = {}
                
            def initialize(self):
                self.initialized = True
                return True
                
            def get_model(self, model_type, model_name, hardware_preferences=None):
                model_id = f"{model_type}:{model_name}"
                # Create simple callable mock
                model = lambda inputs: {
                    "status": "success", 
                    "success": True,
                    "model_id": model_id,
                    "result": {"output": [0.5] * 10},
                    "metrics": {
                        "latency_ms": 100.0,
                        "throughput_items_per_sec": 10.0
                    }
                }
                self.models[model_id] = model
                return model
                
            def execute_concurrent(self, models_and_inputs):
                results = []
                for model_id, inputs in models_and_inputs:
                    if model_id in self.models:
                        result = self.models[model_id](inputs)
                        results.append(result)
                    else:
                        results.append({"status": "error", "error": f"Model {model_id} not found"})
                return results
                
            def get_metrics(self):
                return {"aggregate": {"total_inferences": 0}}
                
            def close(self):
                self.initialized = False
                
        return BasicMockIntegration(max_connections)
    
    def initialize(self):
        """
        Initialize the resource pool with recovery handling.
        
        Returns:
            bool: Success status
        """
        context = {"operation": "initialize"}
        
        def init_op():
            return self.integration.initialize()
            
        success, result, _ = self.recovery.execute_safely(init_op, context)
        return success
    
    def get_model(self, model_type, model_name, hardware_preferences=None):
        """
        Get a model with resilient error handling.
        
        Args:
            model_type: Type of model (text, vision, audio, etc.)
            model_name: Name of the model
            hardware_preferences: Hardware preferences for model loading
            
        Returns:
            Model instance or None on failure
        """
        # Create context for recovery
        context = {
            "operation": "get_model",
            "model_type": model_type,
            "model_name": model_name,
            "hardware_preferences": hardware_preferences
        }
        
        # Add appropriate browser based on model type if not specified
        if hardware_preferences is None:
            hardware_preferences = {}
            
        if "browser" not in hardware_preferences and model_type:
            browser_prefs = self.recovery.browser_preferences
            if model_type in browser_prefs:
                preferred_browser = browser_prefs[model_type]
                hardware_preferences["browser"] = preferred_browser
                context["browser"] = preferred_browser
                logger.debug(f"Using preferred browser {preferred_browser} for {model_type} model")
        
        # Extract platform from hardware preferences
        if "priority_list" in hardware_preferences:
            priority_list = hardware_preferences["priority_list"]
            if priority_list and priority_list[0] in ["webgpu", "webnn", "cpu"]:
                platform = priority_list[0]
                context["platform"] = platform
        elif "platform" in hardware_preferences:
            context["platform"] = hardware_preferences["platform"]
        
        # Create operation function
        def get_model_op():
            return self.integration.get_model(
                model_type=model_type,
                model_name=model_name,
                hardware_preferences=hardware_preferences
            )
        
        # Execute with recovery
        success, model, final_context = self.recovery.execute_safely(get_model_op, context)
        
        if success and model:
            # Store model for tracking
            model_id = f"{model_type}:{model_name}"
            self.loaded_models[model_id] = model
            
            # Wrap model with recovery for inference
            return self._wrap_model_with_recovery(model, model_id, model_type, model_name, final_context)
        else:
            return None
    
    def _wrap_model_with_recovery(self, model, model_id, model_type, model_name, context):
        """Wrap model with recovery capabilities for inference calls."""
        # Store original model
        original_model = model
        recovery = self.recovery  # Keep reference to recovery to use in wrapper
        
        # Create wrapper function with same signature as original model
        def model_with_recovery(inputs):
            # Create inference context
            infer_context = context.copy()
            infer_context["operation"] = "inference"
            infer_context["model_id"] = model_id
            
            # Create inference operation
            def infer_op():
                return original_model(inputs)
                
            # Execute with recovery
            success, result, _ = recovery.execute_safely(infer_op, infer_context)
            return result
        
        # Add attributes to make it look like the original model
        if hasattr(original_model, "model_id"):
            model_with_recovery.model_id = original_model.model_id
        else:
            model_with_recovery.model_id = model_id
            
        if hasattr(original_model, "model_type"):
            model_with_recovery.model_type = original_model.model_type
        else:
            model_with_recovery.model_type = model_type
            
        if hasattr(original_model, "model_name"):
            model_with_recovery.model_name = original_model.model_name
        else:
            model_with_recovery.model_name = model_name
        
        return model_with_recovery
    
    def execute_concurrent(self, models_and_inputs):
        """
        Execute multiple models concurrently with recovery handling.
        
        Args:
            models_and_inputs: List of (model_id, inputs) tuples
            
        Returns:
            List of results
        """
        # Create context for recovery
        context = {
            "operation": "execute_concurrent",
            "model_count": len(models_and_inputs)
        }
        
        # Create operation function
        def execute_op():
            return self.integration.execute_concurrent(models_and_inputs)
            
        # Execute with recovery
        success, results, _ = self.recovery.execute_safely(execute_op, context)
        return results
    
    def get_metrics(self):
        """
        Get metrics with recovery statistics included.
        
        Returns:
            Dict containing metrics and recovery statistics
        """
        # Get base metrics
        try:
            metrics = self.integration.get_metrics()
        except Exception as e:
            logger.warning(f"Error getting base metrics: {e}")
            metrics = {"error": str(e)}
        
        # Get recovery statistics
        recovery_stats = self.recovery.get_recovery_statistics()
        
        # Combine metrics
        combined = {
            "base_metrics": metrics,
            "recovery_stats": recovery_stats,
            "recovery_enabled": True,
            "mock_mode": self.mock_mode,
            "loaded_models_count": len(self.loaded_models)
        }
        
        return combined
    
    def close(self):
        """Close the integration with recovery handling."""
        context = {"operation": "close"}
        
        def close_op():
            return self.integration.close()
            
        self.recovery.execute_safely(close_op, context)
        self.loaded_models.clear()


def run_example():
    """Run an example demonstrating the recovery capabilities."""
    logging.info("Starting ResourcePoolBridgeRecovery example")
    
    # Create the integrated resource pool with recovery
    bridge = ResourcePoolBridgeWithRecovery(
        max_connections=3,
        max_retries=3,
        fallback_to_simulation=True
    )
    
    # Initialize 
    bridge.initialize()
    
    # Load models
    text_model = bridge.get_model(
        model_type="text",
        model_name="bert-base-uncased",
        hardware_preferences={
            "priority_list": ["webgpu", "webnn", "cpu"],
            "browser": "chrome"
        }
    )
    
    vision_model = bridge.get_model(
        model_type="vision",
        model_name="vit-base-patch16-224",
        hardware_preferences={
            "priority_list": ["webgpu", "cpu"],
            "browser": "chrome"
        }
    )
    
    audio_model = bridge.get_model(
        model_type="audio",
        model_name="whisper-tiny",
        hardware_preferences={
            "priority_list": ["webgpu", "cpu"],
            "browser": "firefox"
        }
    )
    
    # Generate sample inputs
    text_input = {
        "input_ids": [101, 2023, 2003, 1037, 3231, 102],
        "attention_mask": [1, 1, 1, 1, 1, 1]
    }
    
    vision_input = {
        "pixel_values": [[[0.5 for _ in range(224)] for _ in range(224)] for _ in range(3)]
    }
    
    audio_input = {
        "input_features": [[0.1 for _ in range(80)] for _ in range(3000)]
    }
    
    # Run inference with resilient error handling
    logging.info("Running inference on text model")
    text_result = text_model(text_input)
    logging.info(f"Text result status: {text_result.get('status')}")
    
    logging.info("Running inference on vision model")
    vision_result = vision_model(vision_input)
    logging.info(f"Vision result status: {vision_result.get('status')}")
    
    logging.info("Running inference on audio model")
    audio_result = audio_model(audio_input)
    logging.info(f"Audio result status: {audio_result.get('status')}")
    
    # Run concurrent inference
    logging.info("Running concurrent inference")
    model_inputs = [
        (text_model.model_id, text_input),
        (vision_model.model_id, vision_input),
        (audio_model.model_id, audio_input)
    ]
    
    concurrent_results = bridge.execute_concurrent(model_inputs)
    logging.info(f"Concurrent results count: {len(concurrent_results)}")
    
    # Get metrics and recovery statistics
    metrics = bridge.get_metrics()
    logging.info("Metrics and recovery statistics:")
    logging.info(f"  - Recovery attempts: {metrics['recovery_stats']['total_recovery_attempts']}")
    
    if 'aggregate' in metrics['base_metrics']:
        aggregate = metrics['base_metrics']['aggregate']
        logging.info(f"  - Total inferences: {aggregate.get('total_inferences', 0)}")
        logging.info(f"  - Average inference time: {aggregate.get('avg_inference_time', 0):.4f}s")
    
    # Close the bridge
    bridge.close()
    logging.info("ResourcePoolBridgeRecovery example completed")


if __name__ == "__main__":
    # Set up console logging with more detail
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console)
    
    # Run the example
    run_example()