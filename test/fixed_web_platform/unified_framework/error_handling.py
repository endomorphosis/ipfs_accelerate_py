"""
Comprehensive Error Handling System for Web Platform (August 2025)

This module provides a standardized error handling system for WebNN and WebGPU
platforms with:
- Custom exception hierarchies for different error types
- Graceful degradation paths with recovery strategies
- Structured error responses with detailed metadata
- Browser-specific error handling
- Exception propagation and traceability

Usage:
    from fixed_web_platform.unified_framework.error_handling import (
        ErrorHandler, WebPlatformError, ConfigurationError
    )
    
    # Create error handler
    error_handler = ErrorHandler(
        recovery_strategy="auto",
        collect_debug_info=True,
        browser="chrome"
    )
    
    # Handle errors with try/except
    try:
        # Run code that might raise exceptions
        pass
    except Exception as e:
        response = error_handler.handle_exception(e)
"""

import os
import time
import logging
import traceback
import json
from typing import Dict, Any, Optional, List, Tuple, Callable, Type, Union

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_platform.error_handling")

# Base exception class for web platform errors
class WebPlatformError(Exception):
    """Base class for all web platform errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize error with message and optional details.
        
        Args:
            message: Error message
            details: Optional dictionary with error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()
        self.severity = "error"  # Default severity
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "severity": self.severity,
            "timestamp": self.timestamp
        }
        
    def set_severity(self, severity: str) -> "WebPlatformError":
        """Set error severity level."""
        self.severity = severity
        return self

# Configuration error types
class ConfigurationError(WebPlatformError):
    """Error raised when configuration is invalid."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 can_auto_correct: bool = False, correction_func: Optional[Callable] = None):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            details: Optional dictionary with error details
            can_auto_correct: Whether this error can be automatically corrected
            correction_func: Function to correct the configuration
        """
        super().__init__(message, details)
        self.can_auto_correct = can_auto_correct
        self.correction_func = correction_func
        self.severity = "warning" if can_auto_correct else "error"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        error_dict = super().to_dict()
        error_dict.update({
            "can_auto_correct": self.can_auto_correct
        })
        return error_dict
        
    def auto_correct(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to auto-correct the configuration."""
        if self.can_auto_correct and self.correction_func:
            try:
                return self.correction_func(config)
            except Exception as e:
                logger.error(f"Error auto-correcting configuration: {e}")
                return None
        return None

# Browser compatibility errors
class BrowserCompatibilityError(WebPlatformError):
    """Error raised when browser doesn't support required features."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None,
                 feature: Optional[str] = None, browser: Optional[str] = None):
        """
        Initialize browser compatibility error.
        
        Args:
            message: Error message
            details: Optional dictionary with error details
            feature: The feature that's not supported
            browser: The browser that doesn't support the feature
        """
        details = details or {}
        if feature:
            details["feature"] = feature
        if browser:
            details["browser"] = browser
            
        super().__init__(message, details)
        self.feature = feature
        self.browser = browser
        self.severity = "error"

# Hardware related errors
class HardwareError(WebPlatformError):
    """Error raised for hardware-related issues."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None,
                 hardware_type: Optional[str] = None):
        """
        Initialize hardware error.
        
        Args:
            message: Error message
            details: Optional dictionary with error details
            hardware_type: Type of hardware involved
        """
        details = details or {}
        if hardware_type:
            details["hardware_type"] = hardware_type
            
        super().__init__(message, details)
        self.hardware_type = hardware_type
        self.severity = "error"

# Model sharding errors
class ShardingError(WebPlatformError):
    """Error raised for model sharding issues."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None,
                 can_disable_sharding: bool = True):
        """
        Initialize sharding error.
        
        Args:
            message: Error message
            details: Optional dictionary with error details
            can_disable_sharding: Whether sharding can be disabled as fallback
        """
        super().__init__(message, details)
        self.can_disable_sharding = can_disable_sharding
        self.severity = "warning" if can_disable_sharding else "error"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        error_dict = super().to_dict()
        error_dict.update({
            "can_disable_sharding": self.can_disable_sharding
        })
        return error_dict

# Runtime errors
class RuntimeError(WebPlatformError):
    """Error raised during model runtime."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None,
                 operation: Optional[str] = None, recoverable: bool = False):
        """
        Initialize runtime error.
        
        Args:
            message: Error message
            details: Optional dictionary with error details
            operation: Operation that failed
            recoverable: Whether operation is recoverable
        """
        details = details or {}
        if operation:
            details["operation"] = operation
            
        super().__init__(message, details)
        self.operation = operation
        self.recoverable = recoverable
        self.severity = "warning" if recoverable else "error"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        error_dict = super().to_dict()
        error_dict.update({
            "operation": self.operation,
            "recoverable": self.recoverable
        })
        return error_dict

# Error handler class
class ErrorHandler:
    """
    Comprehensive error handler for web platforms.
    
    This class provides methods to handle different types of errors that may
    occur during web platform operations, with standardized error reporting,
    recovery strategies, and detailed logging.
    """
    
    def __init__(self, 
                 recovery_strategy: str = "auto",
                 collect_debug_info: bool = True,
                 browser: Optional[str] = None,
                 hardware: Optional[str] = None):
        """
        Initialize error handler.
        
        Args:
            recovery_strategy: Strategy for error recovery ("auto", "fallback", "retry", "abort")
            collect_debug_info: Whether to collect debug information
            browser: Browser information for browser-specific handling
            hardware: Hardware information for hardware-specific handling
        """
        self.recovery_strategy = recovery_strategy
        self.collect_debug_info = collect_debug_info
        self.browser = browser
        self.hardware = hardware
        
        # Map standard exceptions to web platform errors
        self.exception_mapping = {
            ValueError: ConfigurationError,
            TypeError: ConfigurationError,
            NotImplementedError: BrowserCompatibilityError,
            MemoryError: HardwareError
        }
        
    def handle_exception(self, exception: Exception, 
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an exception and generate a standardized error response.
        
        Args:
            exception: The exception to handle
            context: Optional context information
            
        Returns:
            Standardized error response dictionary
        """
        context = context or {}
        
        # Convert standard exceptions to web platform errors
        if not isinstance(exception, WebPlatformError):
            exception = self._convert_exception(exception, context)
            
        # Log the error
        self._log_error(exception, context)
        
        # Collect debug information if enabled
        debug_info = self._collect_debug_info(exception) if self.collect_debug_info else None
        
        # Determine recovery action
        recovery_action = self._determine_recovery_action(exception)
        
        # Create standardized error response
        return self._create_error_response(exception, recovery_action, debug_info, context)
    
    def _convert_exception(self, exception: Exception, 
                          context: Dict[str, Any]) -> WebPlatformError:
        """Convert standard exception to web platform error."""
        for exc_type, web_error_type in self.exception_mapping.items():
            if isinstance(exception, exc_type):
                return web_error_type(str(exception), context)
                
        # Default to generic web platform error
        return WebPlatformError(str(exception), context)
    
    def _log_error(self, exception: WebPlatformError, context: Dict[str, Any]) -> None:
        """Log error with appropriate severity level."""
        error_dict = exception.to_dict()
        
        # Add context to error message
        log_message = f"{error_dict['type']}: {error_dict['message']}"
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            log_message += f" (Context: {context_str})"
            
        # Log based on severity
        if exception.severity == "error":
            logger.error(log_message)
        elif exception.severity == "warning":
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _collect_debug_info(self, exception: WebPlatformError) -> Dict[str, Any]:
        """Collect debug information for troubleshooting."""
        debug_info = {
            "timestamp": time.time(),
            "environment": {
                "browser": self.browser,
                "hardware": self.hardware,
                "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
                "webgpu_enabled": "WEBGPU_ENABLED" in os.environ,
                "webnn_enabled": "WEBNN_ENABLED" in os.environ
            },
            "traceback": traceback.format_exc()
        }
        
        # Add browser-specific info if available
        if self.browser:
            debug_info["browser_info"] = self._get_browser_info(self.browser)
            
        return debug_info
    
    def _get_browser_info(self, browser: str) -> Dict[str, Any]:
        """Get browser-specific information."""
        # This would be more detailed in a real implementation
        return {
            "name": browser,
            "version": os.environ.get(f"{browser.upper()}_VERSION", "unknown"),
            "webgpu_supported": browser.lower() in ["chrome", "edge", "firefox", "safari"],
            "webnn_supported": browser.lower() in ["chrome", "edge", "safari"]
        }
    
    def _determine_recovery_action(self, exception: WebPlatformError) -> str:
        """Determine appropriate recovery action for an error."""
        # Use automatic strategy if configured
        if self.recovery_strategy == "auto":
            # Different recovery strategies based on error type
            if isinstance(exception, ConfigurationError) and exception.can_auto_correct:
                return "auto_correct"
            elif isinstance(exception, BrowserCompatibilityError):
                return "fallback"
            elif isinstance(exception, ShardingError) and exception.can_disable_sharding:
                return "disable_sharding"
            elif isinstance(exception, RuntimeError) and exception.recoverable:
                return "retry"
            else:
                return "abort"
        else:
            # Use configured recovery strategy
            return self.recovery_strategy
    
    def _create_error_response(self, exception: WebPlatformError, 
                              recovery_action: str,
                              debug_info: Optional[Dict[str, Any]] = None,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create standardized error response dictionary."""
        response = {
            "success": False,
            "error": {
                "type": exception.__class__.__name__,
                "message": str(exception),
                "severity": exception.severity,
                "recovery_action": recovery_action,
                "timestamp": exception.timestamp,
                "details": exception.details
            },
            "performance_metrics": context.get("performance_metrics", {}) if context else {}
        }
        
        # Add debug info if available
        if debug_info and (
            self.collect_debug_info or 
            os.environ.get("WEBPLATFORM_DEBUG", "0") == "1"
        ):
            response["error"]["debug_info"] = debug_info
        
        return response
        
    def create_error(self, error_type: str, message: str, 
                    details: Optional[Dict[str, Any]] = None,
                    severity: str = "error") -> WebPlatformError:
        """
        Create a web platform error of specified type.
        
        Args:
            error_type: Type of error to create
            message: Error message
            details: Optional dictionary with error details
            severity: Error severity ("error", "warning", "info")
            
        Returns:
            WebPlatformError instance
        """
        error_classes = {
            "configuration": ConfigurationError,
            "browser_compatibility": BrowserCompatibilityError,
            "hardware": HardwareError,
            "sharding": ShardingError,
            "runtime": RuntimeError,
            "general": WebPlatformError
        }
        
        # Get error class or default to generic error
        error_class = error_classes.get(error_type, WebPlatformError)
        
        # Create and return error instance
        error = error_class(message, details)
        error.set_severity(severity)
        return error