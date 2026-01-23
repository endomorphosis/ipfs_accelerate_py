#!/usr/bin/env python3
"""
Browser-Specific Recovery Strategies for Distributed Testing Framework

This module implements specialized recovery strategies for browser automation, providing
model-aware and browser-specific recovery approaches for different failure scenarios.

Key features:
- Browser-specific recovery strategies tailored to Chrome, Firefox, Edge, and Safari
- Model-aware recovery strategies optimized for different AI model types
- Progressive recovery with escalating interventions
- Resource-aware recovery for memory and performance optimizations
- Comprehensive logging and monitoring for strategy effectiveness
- Integration with existing circuit breaker pattern

Usage:
    Import this module in the BrowserAutomationBridge to enhance browser recovery capabilities.
"""

import anyio
import json
import logging
import os
import re
import time
import traceback
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("browser_recovery")

# Set more verbose logging if environment variable is set
if os.environ.get("SELENIUM_BRIDGE_LOG_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)


class BrowserType(Enum):
    """Browser types supported by the recovery strategies."""
    CHROME = "chrome"
    FIREFOX = "firefox"
    EDGE = "edge"
    SAFARI = "safari"
    UNKNOWN = "unknown"


class ModelType(Enum):
    """Model types for specialized recovery strategies."""
    TEXT = "text"               # Text models (BERT, T5, etc.)
    VISION = "vision"           # Vision models (ViT, etc.)
    AUDIO = "audio"             # Audio models (Whisper, etc.)
    MULTIMODAL = "multimodal"   # Multimodal models (CLIP, LLaVA, etc.)
    GENERIC = "generic"         # Generic models or unknown type


class FailureType(Enum):
    """Types of browser failures."""
    LAUNCH_FAILURE = "launch_failure"           # Browser failed to launch
    CONNECTION_FAILURE = "connection_failure"   # Failed to connect to browser
    TIMEOUT = "timeout"                         # Operation timed out
    CRASH = "crash"                             # Browser crashed
    RESOURCE_EXHAUSTION = "resource_exhaustion" # Out of memory or resources
    GPU_ERROR = "gpu_error"                     # GPU/WebGPU specific error
    API_ERROR = "api_error"                     # WebNN/WebGPU API error
    INTERNAL_ERROR = "internal_error"           # Internal browser error
    UNKNOWN = "unknown"                         # Unknown failure type


class RecoveryLevel(Enum):
    """Levels of recovery intervention."""
    MINIMAL = 1    # Simple retry, no browser restart
    MODERATE = 2   # Browser restart with same settings
    AGGRESSIVE = 3 # Browser restart with modified settings
    FALLBACK = 4   # Switch to different browser or mode
    SIMULATION = 5 # Fall back to simulation mode


class BrowserRecoveryStrategy:
    """Base class for browser recovery strategies."""
    
    def __init__(self, name: str, browser_type: BrowserType, model_type: ModelType):
        """
        Initialize the recovery strategy.
        
        Args:
            name: Strategy name
            browser_type: Type of browser this strategy is for
            model_type: Type of model this strategy is optimized for
        """
        self.name = name
        self.browser_type = browser_type
        self.model_type = model_type
        self.attempts = 0
        self.successes = 0
        self.success_rate = 1.0  # Start optimistic
        self.last_attempt_time = None
        self.last_success_time = None
        self.last_failure_time = None
        self.execution_times = []
        
        logger.debug(f"Initialized browser recovery strategy: {name} for {browser_type.value} and {model_type.value} models")
    
    async def execute(self, bridge: Any, failure_info: Dict[str, Any]) -> bool:
        """
        Execute the recovery strategy.
        
        Args:
            bridge: BrowserAutomationBridge instance
            failure_info: Information about the failure
            
        Returns:
            True if recovery was successful, False otherwise
        """
        self.attempts += 1
        self.last_attempt_time = datetime.now()
        
        start_time = time.time()
        try:
            success = await self._execute_impl(bridge, failure_info)
            
            # Record execution time
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            
            # Update success stats
            if success:
                self.successes += 1
                self.last_success_time = datetime.now()
            else:
                self.last_failure_time = datetime.now()
            
            # Update success rate
            self.success_rate = self.successes / self.attempts
            
            logger.info(f"Browser recovery strategy '{self.name}' completed with success={success} in {execution_time:.2f}s (success rate: {self.success_rate:.2f})")
            
            return success
        except Exception as e:
            # Record execution time even on exception
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            
            # Record failure
            self.last_failure_time = datetime.now()
            self.success_rate = self.successes / self.attempts
            
            logger.error(f"Error executing browser recovery strategy '{self.name}': {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    async def _execute_impl(self, bridge: Any, failure_info: Dict[str, Any]) -> bool:
        """
        Implementation of recovery strategy.
        
        Args:
            bridge: BrowserAutomationBridge instance
            failure_info: Information about the failure
            
        Returns:
            True if recovery was successful, False otherwise
        """
        raise NotImplementedError("Recovery strategy implementation not provided")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about this recovery strategy.
        
        Returns:
            Dictionary with strategy statistics
        """
        avg_execution_time = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0
        
        return {
            "name": self.name,
            "browser_type": self.browser_type.value,
            "model_type": self.model_type.value,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "avg_execution_time": avg_execution_time,
            "last_attempt": self.last_attempt_time.isoformat() if self.last_attempt_time else None,
            "last_success": self.last_success_time.isoformat() if self.last_success_time else None,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class SimpleRetryStrategy(BrowserRecoveryStrategy):
    """Simple retry strategy with delay."""
    
    def __init__(self, browser_type: BrowserType, model_type: ModelType = ModelType.GENERIC, 
                retry_delay: float = 2.0, max_retries: int = 3):
        """
        Initialize the simple retry strategy.
        
        Args:
            browser_type: Type of browser this strategy is for
            model_type: Type of model this strategy is optimized for
            retry_delay: Delay between retries in seconds
            max_retries: Maximum number of retry attempts
        """
        super().__init__(f"simple_retry_{browser_type.value}", browser_type, model_type)
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.level = RecoveryLevel.MINIMAL
    
    async def _execute_impl(self, bridge: Any, failure_info: Dict[str, Any]) -> bool:
        """
        Implementation of simple retry strategy.
        
        Args:
            bridge: BrowserAutomationBridge instance
            failure_info: Information about the failure
            
        Returns:
            True if recovery was successful, False otherwise
        """
        retry_count = failure_info.get("retry_count", 0)
        
        # Check if we've exceeded max retries
        if retry_count >= self.max_retries:
            logger.warning(f"Maximum retries ({self.max_retries}) exceeded for {self.browser_type.value}")
            return False
        
        # Log the retry attempt
        logger.info(f"Retry attempt {retry_count + 1}/{self.max_retries} for {self.browser_type.value}")
        
        # Wait before retry
        await anyio.sleep(self.retry_delay)
        
        # Get the operation to retry
        operation = failure_info.get("operation")
        if not operation or not callable(operation):
            logger.error("No operation provided for retry")
            return False
        
        # Get operation arguments
        args = failure_info.get("args", [])
        kwargs = failure_info.get("kwargs", {})
        
        try:
            # Execute the operation
            result = operation(*args, **kwargs)
            
            # Handle async operations
            if asyncio.iscoroutine(result):
                result = await result
            
            logger.info(f"Retry successful for {self.browser_type.value}")
            return True
        except Exception as e:
            logger.warning(f"Retry failed for {self.browser_type.value}: {str(e)}")
            
            # Update retry count in failure info for next attempt
            failure_info["retry_count"] = retry_count + 1
            
            return False


class BrowserRestartStrategy(BrowserRecoveryStrategy):
    """Strategy to restart the browser when it fails."""
    
    def __init__(self, browser_type: BrowserType, model_type: ModelType = ModelType.GENERIC,
                cooldown_period: float = 5.0, preserve_browser_args: bool = True):
        """
        Initialize the browser restart strategy.
        
        Args:
            browser_type: Type of browser this strategy is for
            model_type: Type of model this strategy is optimized for
            cooldown_period: Cooldown period before restart
            preserve_browser_args: Whether to preserve browser arguments
        """
        super().__init__(f"browser_restart_{browser_type.value}", browser_type, model_type)
        self.cooldown_period = cooldown_period
        self.preserve_browser_args = preserve_browser_args
        self.level = RecoveryLevel.MODERATE
    
    async def _execute_impl(self, bridge: Any, failure_info: Dict[str, Any]) -> bool:
        """
        Implementation of browser restart strategy.
        
        Args:
            bridge: BrowserAutomationBridge instance
            failure_info: Information about the failure
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Executing browser restart for {self.browser_type.value}")
        
        # Safely close the browser if it's running
        try:
            await bridge.close()
        except Exception as e:
            logger.warning(f"Error closing browser during restart: {str(e)}")
        
        # Cooldown period before restart
        await anyio.sleep(self.cooldown_period)
        
        # Preserve original browser arguments if requested
        if self.preserve_browser_args and hasattr(bridge, 'get_browser_args'):
            # Save original arguments
            original_args = bridge.get_browser_args()
            logger.debug(f"Preserving browser arguments: {original_args}")
        
        try:
            # Launch browser again
            logger.info(f"Relaunching {self.browser_type.value} browser")
            success = await bridge.launch(allow_simulation=False)
            
            if success:
                logger.info(f"Successfully relaunched {self.browser_type.value} browser")
                
                # Verify that browser is responsive
                if hasattr(bridge, 'check_browser_responsive'):
                    responsive = await bridge.check_browser_responsive()
                    if not responsive:
                        logger.warning(f"Browser restarted but is not responsive")
                        return False
                
                return True
            else:
                logger.warning(f"Failed to relaunch {self.browser_type.value} browser")
                return False
                
        except Exception as e:
            logger.error(f"Error relaunching browser: {str(e)}")
            return False


class SettingsAdjustmentStrategy(BrowserRecoveryStrategy):
    """Strategy to adjust browser settings to recover from failures."""
    
    def __init__(self, browser_type: BrowserType, model_type: ModelType = ModelType.GENERIC):
        """
        Initialize the settings adjustment strategy.
        
        Args:
            browser_type: Type of browser this strategy is for
            model_type: Type of model this strategy is optimized for
        """
        super().__init__(f"settings_adjustment_{browser_type.value}", browser_type, model_type)
        self.level = RecoveryLevel.AGGRESSIVE
        
        # Define browser-specific adjustment strategies
        self.browser_adjustments = {
            BrowserType.CHROME: self._adjust_chrome_settings,
            BrowserType.FIREFOX: self._adjust_firefox_settings,
            BrowserType.EDGE: self._adjust_edge_settings,
            BrowserType.SAFARI: self._adjust_safari_settings,
        }
        
        # Define model-specific adjustment strategies
        self.model_adjustments = {
            ModelType.TEXT: self._adjust_text_model_settings,
            ModelType.VISION: self._adjust_vision_model_settings,
            ModelType.AUDIO: self._adjust_audio_model_settings,
            ModelType.MULTIMODAL: self._adjust_multimodal_model_settings,
        }
    
    async def _execute_impl(self, bridge: Any, failure_info: Dict[str, Any]) -> bool:
        """
        Implementation of settings adjustment strategy.
        
        Args:
            bridge: BrowserAutomationBridge instance
            failure_info: Information about the failure
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Executing settings adjustment for {self.browser_type.value} with {self.model_type.value} models")
        
        # Get failure type
        failure_type = FailureType(failure_info.get("failure_type", FailureType.UNKNOWN.value))
        
        # Safely close the browser if it's running
        try:
            await bridge.close()
        except Exception as e:
            logger.warning(f"Error closing browser during settings adjustment: {str(e)}")
        
        try:
            # Apply browser-specific adjustments
            if self.browser_type in self.browser_adjustments:
                adjustment_fn = self.browser_adjustments[self.browser_type]
                await adjustment_fn(bridge, failure_type, failure_info)
            
            # Apply model-specific adjustments
            if self.model_type in self.model_adjustments:
                adjustment_fn = self.model_adjustments[self.model_type]
                await adjustment_fn(bridge, failure_type, failure_info)
            
            # Relaunch browser with adjusted settings
            logger.info(f"Relaunching {self.browser_type.value} with adjusted settings")
            success = await bridge.launch(allow_simulation=False)
            
            if success:
                logger.info(f"Successfully relaunched {self.browser_type.value} with adjusted settings")
                return True
            else:
                logger.warning(f"Failed to relaunch {self.browser_type.value} with adjusted settings")
                return False
                
        except Exception as e:
            logger.error(f"Error adjusting browser settings: {str(e)}")
            return False
    
    async def _adjust_chrome_settings(self, bridge: Any, failure_type: FailureType, failure_info: Dict[str, Any]):
        """Adjust Chrome-specific settings based on failure type."""
        if not hasattr(bridge, 'add_browser_arg'):
            logger.warning("Browser bridge does not support add_browser_arg")
            return
            
        # Different adjustments based on failure type
        if failure_type == FailureType.RESOURCE_EXHAUSTION:
            # Reduce memory usage
            bridge.add_browser_arg("--disable-gpu-rasterization")
            bridge.add_browser_arg("--disable-gpu-vsync")
            bridge.add_browser_arg("--enable-low-end-device-mode")
            
        elif failure_type == FailureType.GPU_ERROR:
            # Adjust GPU settings
            bridge.add_browser_arg("--disable-gpu-process-crash-limit")
            bridge.add_browser_arg("--disable-gpu-watchdog")
            
        elif failure_type == FailureType.CRASH:
            # Crash-specific adjustments
            bridge.add_browser_arg("--disable-crash-reporter")
            bridge.add_browser_arg("--disable-breakpad")
            
        # Common Chrome adjustments
        bridge.add_browser_arg("--disable-web-security")  # Disable security features for testing
        bridge.add_browser_arg("--disable-features=BackForwardCache")  # Disable caching that can cause issues
    
    async def _adjust_firefox_settings(self, bridge: Any, failure_type: FailureType, failure_info: Dict[str, Any]):
        """Adjust Firefox-specific settings based on failure type."""
        if not hasattr(bridge, 'add_browser_pref'):
            logger.warning("Browser bridge does not support add_browser_pref")
            return
        
        # Different adjustments based on failure type
        if failure_type == FailureType.RESOURCE_EXHAUSTION:
            # Reduce memory usage
            bridge.add_browser_pref("browser.cache.memory.capacity", 32768)  # 32MB cache
            bridge.add_browser_pref("browser.sessionhistory.max_entries", 10)  # Reduce session history
            
        elif failure_type == FailureType.GPU_ERROR:
            # Adjust WebGPU settings
            bridge.add_browser_pref("dom.webgpu.unsafe", True)  # Allow unsafe operations
            bridge.add_browser_pref("gfx.webrender.all", False)  # Disable WebRender
            
            # For compute shader failures, try different workgroup size
            if self.model_type == ModelType.AUDIO:
                bridge.add_browser_pref("dom.webgpu.workgroup_size", "128,2,1")
            
        elif failure_type == FailureType.CRASH:
            # Crash-specific adjustments
            bridge.add_browser_pref("dom.ipc.processCount", 1)  # Use single process
            bridge.add_browser_pref("browser.tabs.remote.autostart", False)  # Disable multiprocess
        
        # Common Firefox adjustments
        bridge.add_browser_pref("dom.webgpu.enabled", True)  # Ensure WebGPU is enabled
        bridge.add_browser_pref("security.sandbox.content.level", 0)  # Lower sandbox level
    
    async def _adjust_edge_settings(self, bridge: Any, failure_type: FailureType, failure_info: Dict[str, Any]):
        """Adjust Edge-specific settings based on failure type."""
        if not hasattr(bridge, 'add_browser_arg'):
            logger.warning("Browser bridge does not support add_browser_arg")
            return
            
        # Edge uses Chromium, so most Chrome settings apply
        await self._adjust_chrome_settings(bridge, failure_type, failure_info)
        
        # Edge-specific adjustments
        if failure_type == FailureType.API_ERROR and self.model_type == ModelType.TEXT:
            # For WebNN issues with text models on Edge
            bridge.add_browser_arg("--enable-features=WebNN,WebNNCompileOptions")
            bridge.add_browser_arg("--enable-dawn-features=enable_webnn_extension")
    
    async def _adjust_safari_settings(self, bridge: Any, failure_type: FailureType, failure_info: Dict[str, Any]):
        """Adjust Safari-specific settings based on failure type."""
        # Safari has limited customization options via automation
        logger.info("Safari settings adjustment is limited")
        
        # The best we can do is restart with default settings
        # Safari-specific code would go here if more options were available
        pass
    
    async def _adjust_text_model_settings(self, bridge: Any, failure_type: FailureType, failure_info: Dict[str, Any]):
        """Adjust settings for text models based on failure type."""
        if self.browser_type == BrowserType.EDGE:
            # Edge has best WebNN support for text models
            if hasattr(bridge, 'set_platform'):
                bridge.set_platform("webnn")  # Switch to WebNN for text models
            
            if hasattr(bridge, 'add_browser_arg'):
                bridge.add_browser_arg("--enable-features=WebNN")
                
        elif self.browser_type == BrowserType.CHROME:
            # Optimize Chrome for text models
            if hasattr(bridge, 'add_browser_arg'):
                bridge.add_browser_arg("--enable-features=WebGPU")
                
        # Reduce resource usage for text models
        if hasattr(bridge, 'set_resource_settings'):
            bridge.set_resource_settings(
                max_batch_size=1,  # Conservative batch size
                optimize_for="latency"  # Optimize for latency over throughput
            )
    
    async def _adjust_vision_model_settings(self, bridge: Any, failure_type: FailureType, failure_info: Dict[str, Any]):
        """Adjust settings for vision models based on failure type."""
        if self.browser_type == BrowserType.CHROME:
            # Chrome has good WebGPU support for vision models
            if hasattr(bridge, 'set_platform'):
                bridge.set_platform("webgpu")
            
            if hasattr(bridge, 'add_browser_arg'):
                bridge.add_browser_arg("--enable-zero-copy")
                bridge.add_browser_arg("--enable-gpu-memory-buffer-video-frames")
        
        # Enable shader precompilation for vision models
        if hasattr(bridge, 'set_shader_precompilation'):
            bridge.set_shader_precompilation(True)
    
    async def _adjust_audio_model_settings(self, bridge: Any, failure_type: FailureType, failure_info: Dict[str, Any]):
        """Adjust settings for audio models based on failure type."""
        if self.browser_type == BrowserType.FIREFOX:
            # Firefox has best compute shader support for audio models
            if hasattr(bridge, 'set_platform'):
                bridge.set_platform("webgpu")
            
            if hasattr(bridge, 'set_compute_shaders'):
                bridge.set_compute_shaders(True)
                
            if hasattr(bridge, 'add_browser_pref'):
                bridge.add_browser_pref("dom.webgpu.advanced-compute", True)
        
        # Adjust audio processing settings
        if hasattr(bridge, 'set_audio_settings'):
            bridge.set_audio_settings(
                optimize_for_firefox=self.browser_type == BrowserType.FIREFOX,
                webgpu_compute_shaders=True
            )
    
    async def _adjust_multimodal_model_settings(self, bridge: Any, failure_type: FailureType, failure_info: Dict[str, Any]):
        """Adjust settings for multimodal models based on failure type."""
        # Enable parallel loading for multimodal models
        if hasattr(bridge, 'set_parallel_loading'):
            bridge.set_parallel_loading(True)
        
        # Memory-focused adjustments for multimodal models
        if failure_type == FailureType.RESOURCE_EXHAUSTION:
            if hasattr(bridge, 'set_resource_settings'):
                bridge.set_resource_settings(
                    max_batch_size=1,  # Smallest batch size
                    progressive_loading=True,  # Load model components progressively
                    shared_tensors=True  # Enable tensor sharing
                )


class BrowserFallbackStrategy(BrowserRecoveryStrategy):
    """Strategy to fall back to a different browser when current one fails."""
    
    def __init__(self, browser_type: BrowserType, model_type: ModelType = ModelType.GENERIC,
                fallback_order: Optional[List[BrowserType]] = None):
        """
        Initialize the browser fallback strategy.
        
        Args:
            browser_type: Type of browser this strategy is for
            model_type: Type of model this strategy is optimized for
            fallback_order: Custom order of browsers to try
        """
        super().__init__(f"browser_fallback_{browser_type.value}", browser_type, model_type)
        self.level = RecoveryLevel.FALLBACK
        
        # Set browser fallback order if not provided
        if fallback_order is None:
            # Different default fallback orders based on model type
            if model_type == ModelType.TEXT:
                self.fallback_order = {
                    BrowserType.EDGE: [BrowserType.CHROME, BrowserType.FIREFOX],
                    BrowserType.CHROME: [BrowserType.EDGE, BrowserType.FIREFOX],
                    BrowserType.FIREFOX: [BrowserType.CHROME, BrowserType.EDGE],
                    BrowserType.SAFARI: [BrowserType.CHROME, BrowserType.EDGE, BrowserType.FIREFOX],
                }
            elif model_type == ModelType.AUDIO:
                self.fallback_order = {
                    BrowserType.FIREFOX: [BrowserType.CHROME, BrowserType.EDGE],
                    BrowserType.CHROME: [BrowserType.FIREFOX, BrowserType.EDGE],
                    BrowserType.EDGE: [BrowserType.FIREFOX, BrowserType.CHROME],
                    BrowserType.SAFARI: [BrowserType.FIREFOX, BrowserType.CHROME, BrowserType.EDGE],
                }
            elif model_type == ModelType.VISION:
                self.fallback_order = {
                    BrowserType.CHROME: [BrowserType.EDGE, BrowserType.FIREFOX],
                    BrowserType.EDGE: [BrowserType.CHROME, BrowserType.FIREFOX],
                    BrowserType.FIREFOX: [BrowserType.CHROME, BrowserType.EDGE],
                    BrowserType.SAFARI: [BrowserType.CHROME, BrowserType.EDGE, BrowserType.FIREFOX],
                }
            else:  # Default or multimodal
                self.fallback_order = {
                    BrowserType.CHROME: [BrowserType.EDGE, BrowserType.FIREFOX],
                    BrowserType.EDGE: [BrowserType.CHROME, BrowserType.FIREFOX],
                    BrowserType.FIREFOX: [BrowserType.CHROME, BrowserType.EDGE],
                    BrowserType.SAFARI: [BrowserType.CHROME, BrowserType.EDGE, BrowserType.FIREFOX],
                }
        else:
            # Use custom fallback order for all browser types
            self.fallback_order = {browser: fallback_order for browser in BrowserType}
    
    async def _execute_impl(self, bridge: Any, failure_info: Dict[str, Any]) -> bool:
        """
        Implementation of browser fallback strategy.
        
        Args:
            bridge: BrowserAutomationBridge instance
            failure_info: Information about the failure
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Get current browser type
        current_browser = self.browser_type
        
        # Get fallback browsers to try
        fallback_browsers = self.fallback_order.get(current_browser, [])
        
        if not fallback_browsers:
            logger.warning(f"No fallback browsers defined for {current_browser.value}")
            return False
        
        # Safely close current browser
        try:
            await bridge.close()
        except Exception as e:
            logger.warning(f"Error closing browser during fallback: {str(e)}")
        
        # Try each fallback browser
        for browser in fallback_browsers:
            logger.info(f"Attempting fallback to {browser.value} browser")
            
            try:
                # Change browser type
                if hasattr(bridge, 'set_browser'):
                    bridge.set_browser(browser.value)
                else:
                    # If bridge doesn't support direct browser changing
                    logger.warning("Browser bridge does not support set_browser")
                    
                    # Try to change browser through bridge's initialization
                    if hasattr(bridge, 'browser_name'):
                        bridge.browser_name = browser.value
                
                # Set appropriate additional settings based on model type
                # For text models
                if self.model_type == ModelType.TEXT and browser == BrowserType.EDGE:
                    # Use WebNN for text models on Edge
                    if hasattr(bridge, 'set_platform'):
                        bridge.set_platform("webnn")
                
                # For audio models
                elif self.model_type == ModelType.AUDIO and browser == BrowserType.FIREFOX:
                    # Use compute shaders for audio on Firefox
                    if hasattr(bridge, 'set_platform'):
                        bridge.set_platform("webgpu")
                    if hasattr(bridge, 'set_compute_shaders'):
                        bridge.set_compute_shaders(True)
                
                # Launch the fallback browser
                success = await bridge.launch(allow_simulation=False)
                
                if success:
                    logger.info(f"Successfully switched to fallback browser {browser.value}")
                    return True
                else:
                    logger.warning(f"Failed to launch fallback browser {browser.value}")
            except Exception as e:
                logger.error(f"Error switching to fallback browser {browser.value}: {str(e)}")
        
        logger.error(f"All fallback browsers failed")
        return False


class SimulationFallbackStrategy(BrowserRecoveryStrategy):
    """Strategy to fall back to simulation mode when all browsers fail."""
    
    def __init__(self, browser_type: BrowserType, model_type: ModelType = ModelType.GENERIC):
        """
        Initialize the simulation fallback strategy.
        
        Args:
            browser_type: Type of browser this strategy is for
            model_type: Type of model this strategy is optimized for
        """
        super().__init__(f"simulation_fallback_{browser_type.value}", browser_type, model_type)
        self.level = RecoveryLevel.SIMULATION
    
    async def _execute_impl(self, bridge: Any, failure_info: Dict[str, Any]) -> bool:
        """
        Implementation of simulation fallback strategy.
        
        Args:
            bridge: BrowserAutomationBridge instance
            failure_info: Information about the failure
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Executing simulation fallback strategy as last resort")
        
        # Safely close current browser
        try:
            await bridge.close()
        except Exception as e:
            logger.warning(f"Error closing browser during simulation fallback: {str(e)}")
        
        try:
            # Launch with simulation allowed
            logger.info("Launching browser with simulation mode allowed")
            success = await bridge.launch(allow_simulation=True)
            
            if success:
                # Verify that we're running in simulation mode
                if hasattr(bridge, 'simulation_mode'):
                    is_simulation = bridge.simulation_mode
                    
                    if is_simulation:
                        logger.info("Successfully switched to simulation mode")
                        return True
                    else:
                        logger.warning("Browser launched but not in simulation mode")
                        return False
                else:
                    # Assume success if bridge doesn't expose simulation_mode
                    logger.info("Browser launched with simulation allowed")
                    return True
            else:
                logger.error("Failed to launch browser even with simulation mode")
                return False
                
        except Exception as e:
            logger.error(f"Error falling back to simulation mode: {str(e)}")
            return False


class ModelSpecificRecoveryStrategy(BrowserRecoveryStrategy):
    """Strategy that applies model-specific optimizations for recovery."""
    
    def __init__(self, browser_type: BrowserType, model_type: ModelType):
        """
        Initialize the model-specific recovery strategy.
        
        Args:
            browser_type: Type of browser this strategy is for
            model_type: Type of model this strategy is optimized for
        """
        super().__init__(f"model_specific_{model_type.value}_{browser_type.value}", browser_type, model_type)
        self.level = RecoveryLevel.AGGRESSIVE
        
        # Map of model types to their specific recovery functions
        self.model_recovery_functions = {
            ModelType.TEXT: self._recover_text_model,
            ModelType.VISION: self._recover_vision_model,
            ModelType.AUDIO: self._recover_audio_model,
            ModelType.MULTIMODAL: self._recover_multimodal_model,
        }
    
    async def _execute_impl(self, bridge: Any, failure_info: Dict[str, Any]) -> bool:
        """
        Implementation of model-specific recovery strategy.
        
        Args:
            bridge: BrowserAutomationBridge instance
            failure_info: Information about the failure
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Check if we have a recovery function for this model type
        if self.model_type not in self.model_recovery_functions:
            logger.warning(f"No model-specific recovery for {self.model_type.value}")
            return False
        
        # Get the recovery function
        recovery_fn = self.model_recovery_functions[self.model_type]
        
        # Safely close current browser
        try:
            await bridge.close()
        except Exception as e:
            logger.warning(f"Error closing browser during model-specific recovery: {str(e)}")
        
        try:
            # Apply model-specific recovery
            logger.info(f"Applying {self.model_type.value}-specific recovery for {self.browser_type.value}")
            await recovery_fn(bridge, failure_info)
            
            # Launch browser with adjusted settings
            logger.info("Launching browser with model-specific optimizations")
            success = await bridge.launch(allow_simulation=False)
            
            if success:
                logger.info(f"Successfully launched browser with {self.model_type.value}-specific optimizations")
                return True
            else:
                logger.warning(f"Failed to launch browser with {self.model_type.value}-specific optimizations")
                return False
                
        except Exception as e:
            logger.error(f"Error applying model-specific recovery: {str(e)}")
            return False
    
    async def _recover_text_model(self, bridge: Any, failure_info: Dict[str, Any]):
        """Apply text model specific optimizations."""
        # Set platform based on browser (Edge and Chrome have best WebNN support)
        if self.browser_type in [BrowserType.EDGE, BrowserType.CHROME]:
            if hasattr(bridge, 'set_platform'):
                bridge.set_platform("webnn")
        else:
            if hasattr(bridge, 'set_platform'):
                bridge.set_platform("webgpu")
        
        # Adjust browser-specific settings
        if self.browser_type == BrowserType.EDGE:
            if hasattr(bridge, 'add_browser_arg'):
                bridge.add_browser_arg("--enable-features=WebNN")
                bridge.add_browser_arg("--enable-dawn-features=enable_webnn_extension")
        
        elif self.browser_type == BrowserType.CHROME:
            if hasattr(bridge, 'add_browser_arg'):
                bridge.add_browser_arg("--enable-features=WebNN")
        
        # Apply common text model optimizations
        if hasattr(bridge, 'set_shader_precompilation'):
            bridge.set_shader_precompilation(True)
        
        # Reduce memory usage for text models
        if hasattr(bridge, 'set_resource_settings'):
            bridge.set_resource_settings(
                max_batch_size=1,
                optimize_for="latency",
                progressive_loading=False
            )
    
    async def _recover_vision_model(self, bridge: Any, failure_info: Dict[str, Any]):
        """Apply vision model specific optimizations."""
        # Vision models perform best on WebGPU
        if hasattr(bridge, 'set_platform'):
            bridge.set_platform("webgpu")
        
        # Always enable shader precompilation for vision models
        if hasattr(bridge, 'set_shader_precompilation'):
            bridge.set_shader_precompilation(True)
        
        # Adjust browser-specific settings
        if self.browser_type == BrowserType.CHROME:
            if hasattr(bridge, 'add_browser_arg'):
                bridge.add_browser_arg("--enable-zero-copy")
                bridge.add_browser_arg("--enable-gpu-memory-buffer-video-frames")
                bridge.add_browser_arg("--enable-features=WebGPU")
        
        elif self.browser_type == BrowserType.FIREFOX:
            if hasattr(bridge, 'add_browser_pref'):
                bridge.add_browser_pref("gfx.webrender.all", True)
                bridge.add_browser_pref("dom.webgpu.enabled", True)
        
        # Apply common vision model optimizations
        if hasattr(bridge, 'set_resource_settings'):
            bridge.set_resource_settings(
                max_batch_size=4,  # Vision models can handle larger batches
                optimize_for="throughput",
                shared_tensors=True  # Enable tensor sharing for vision models
            )
    
    async def _recover_audio_model(self, bridge: Any, failure_info: Dict[str, Any]):
        """Apply audio model specific optimizations."""
        # Audio models work best on Firefox with compute shaders
        if hasattr(bridge, 'set_platform'):
            bridge.set_platform("webgpu")
        
        # Always enable compute shaders for audio models
        if hasattr(bridge, 'set_compute_shaders'):
            bridge.set_compute_shaders(True)
        
        # Adjust browser-specific settings
        if self.browser_type == BrowserType.FIREFOX:
            if hasattr(bridge, 'add_browser_pref'):
                bridge.add_browser_pref("dom.webgpu.advanced-compute", True)
                bridge.add_browser_pref("dom.webgpu.enabled", True)
        
        elif self.browser_type == BrowserType.CHROME:
            if hasattr(bridge, 'add_browser_arg'):
                bridge.add_browser_arg("--enable-dawn-features=compute_shaders")
                bridge.add_browser_arg("--enable-features=WebGPU")
        
        # Apply common audio model optimizations
        if hasattr(bridge, 'set_audio_settings'):
            bridge.set_audio_settings(
                optimize_for_firefox=self.browser_type == BrowserType.FIREFOX,
                webgpu_compute_shaders=True
            )
    
    async def _recover_multimodal_model(self, bridge: Any, failure_info: Dict[str, Any]):
        """Apply multimodal model specific optimizations."""
        # Multimodal models benefit from parallel loading
        if hasattr(bridge, 'set_parallel_loading'):
            bridge.set_parallel_loading(True)
        
        # Choose platform based on browser capabilities
        if hasattr(bridge, 'set_platform'):
            bridge.set_platform("webgpu")  # WebGPU generally better for multimodal
        
        # Enable shader precompilation
        if hasattr(bridge, 'set_shader_precompilation'):
            bridge.set_shader_precompilation(True)
        
        # Adjust browser-specific settings
        if self.browser_type == BrowserType.CHROME:
            if hasattr(bridge, 'add_browser_arg'):
                bridge.add_browser_arg("--enable-zero-copy")
                bridge.add_browser_arg("--enable-features=WebGPU,ParallelDownloading")
        
        # Memory optimizations for multimodal models
        if hasattr(bridge, 'set_resource_settings'):
            bridge.set_resource_settings(
                max_batch_size=1,  # Conservative for multimodal models
                progressive_loading=True,  # Load components progressively
                shared_tensors=True,  # Enable tensor sharing
                optimize_for="memory"  # Optimize for memory efficiency
            )


class ProgressiveRecoveryManager:
    """
    Manager for progressive recovery strategies that tries multiple strategies in sequence.
    
    This class tries increasingly aggressive recovery strategies, starting from simple retries
    and escalating to more invasive approaches like browser restarts, settings adjustments,
    browser fallbacks, and finally simulation mode.
    """
    
    def __init__(self):
        """Initialize the progressive recovery manager."""
        self.strategies_by_browser = {}
        self.strategies_by_model = {}
        self.strategy_history = []
        self.max_history_size = 100
        
        # Initialize with default strategies
        self._initialize_default_strategies()
        
        logger.info("Progressive recovery manager initialized")
    
    def _initialize_default_strategies(self):
        """Initialize default recovery strategies for all browsers and model types."""
        # Create strategies for each browser and model type combination
        for browser_type in BrowserType:
            if browser_type == BrowserType.UNKNOWN:
                continue
                
            browser_strategies = {}
            
            for model_type in ModelType:
                if model_type == ModelType.GENERIC:
                    continue
                    
                # Create a progression of strategies for this combination
                progression = []
                
                # Level 1: Simple retry (minimal intervention)
                progression.append(SimpleRetryStrategy(browser_type, model_type))
                
                # Level 2: Browser restart (moderate intervention)
                progression.append(BrowserRestartStrategy(browser_type, model_type))
                
                # Level 3: Settings adjustment (aggressive intervention)
                progression.append(SettingsAdjustmentStrategy(browser_type, model_type))
                
                # Level 3.5: Model-specific optimizations (aggressive intervention)
                progression.append(ModelSpecificRecoveryStrategy(browser_type, model_type))
                
                # Level 4: Browser fallback (fallback intervention)
                progression.append(BrowserFallbackStrategy(browser_type, model_type))
                
                # Level 5: Simulation fallback (last resort)
                progression.append(SimulationFallbackStrategy(browser_type, model_type))
                
                # Store progression for this model type
                browser_strategies[model_type] = progression
            
            # Store strategies for this browser
            self.strategies_by_browser[browser_type] = browser_strategies
    
    def get_strategies(self, browser_type: BrowserType, model_type: ModelType) -> List[BrowserRecoveryStrategy]:
        """
        Get the progression of recovery strategies for a browser and model type.
        
        Args:
            browser_type: Type of browser
            model_type: Type of model
            
        Returns:
            List of recovery strategies in progression order
        """
        browser_strategies = self.strategies_by_browser.get(browser_type)
        
        if not browser_strategies:
            logger.warning(f"No recovery strategies defined for {browser_type.value}")
            return []
        
        # Get strategies for this model type, fallback to GENERIC if not found
        strategies = browser_strategies.get(model_type, browser_strategies.get(ModelType.GENERIC, []))
        
        return strategies
    
    async def execute_progressive_recovery(self, bridge: Any, browser_type: BrowserType, 
                                        model_type: ModelType, failure_info: Dict[str, Any],
                                        start_level: RecoveryLevel = RecoveryLevel.MINIMAL) -> bool:
        """
        Execute progressive recovery strategies in sequence until one succeeds.
        
        Args:
            bridge: BrowserAutomationBridge instance
            browser_type: Type of browser
            model_type: Type of model
            failure_info: Information about the failure
            start_level: Recovery level to start from
            
        Returns:
            True if any recovery strategy succeeded, False otherwise
        """
        # Get the progression of strategies
        strategies = self.get_strategies(browser_type, model_type)
        
        if not strategies:
            logger.error(f"No recovery strategies available for {browser_type.value}/{model_type.value}")
            return False
        
        # Filter strategies based on start_level
        strategies = [s for s in strategies if getattr(s, 'level', RecoveryLevel.MINIMAL).value >= start_level.value]
        
        # Try each strategy in sequence
        for strategy in strategies:
            logger.info(f"Trying recovery strategy: {strategy.name} (level {getattr(strategy, 'level', 'unknown')})")
            
            # Execute the strategy
            start_time = time.time()
            success = await strategy.execute(bridge, failure_info)
            execution_time = time.time() - start_time
            
            # Record in history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "strategy_name": strategy.name,
                "browser_type": browser_type.value,
                "model_type": model_type.value,
                "success": success,
                "execution_time": execution_time,
                "failure_info": failure_info.copy()
            }
            self.strategy_history.append(history_entry)
            
            # Trim history if needed
            if len(self.strategy_history) > self.max_history_size:
                self.strategy_history = self.strategy_history[-self.max_history_size:]
            
            # If the strategy succeeded, we're done
            if success:
                logger.info(f"Recovery strategy {strategy.name} succeeded")
                return True
            
            logger.warning(f"Recovery strategy {strategy.name} failed, trying next strategy")
        
        # All strategies failed
        logger.error(f"All recovery strategies failed for {browser_type.value}/{model_type.value}")
        return False
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all recovery strategies.
        
        Returns:
            Dictionary with strategy statistics
        """
        stats = {
            "summary": {
                "total_browsers": len(self.strategies_by_browser),
                "total_models": len(ModelType) - 1,  # Exclude GENERIC
                "total_strategies": 0,
                "total_attempts": 0,
                "total_successes": 0,
                "overall_success_rate": 0.0
            },
            "browsers": {},
            "models": {},
            "strategies": {}
        }
        
        total_attempts = 0
        total_successes = 0
        all_strategies = set()
        
        # Collect statistics from all strategies
        for browser_type, browser_strategies in self.strategies_by_browser.items():
            browser_stats = {
                "attempts": 0,
                "successes": 0,
                "success_rate": 0.0,
                "models": {}
            }
            
            for model_type, strategies in browser_strategies.items():
                model_stats = {
                    "attempts": 0,
                    "successes": 0,
                    "success_rate": 0.0,
                    "strategies": {}
                }
                
                for strategy in strategies:
                    all_strategies.add(strategy.name)
                    
                    # Add stats for this strategy
                    model_stats["attempts"] += strategy.attempts
                    model_stats["successes"] += strategy.successes
                    
                    # Add to strategy-specific stats
                    if strategy.name not in stats["strategies"]:
                        stats["strategies"][strategy.name] = strategy.get_stats()
                    
                    # Add to model-specific stats
                    model_stats["strategies"][strategy.name] = {
                        "attempts": strategy.attempts,
                        "successes": strategy.successes,
                        "success_rate": strategy.success_rate
                    }
                
                # Calculate model success rate
                if model_stats["attempts"] > 0:
                    model_stats["success_rate"] = model_stats["successes"] / model_stats["attempts"]
                
                # Add to browser stats
                browser_stats["attempts"] += model_stats["attempts"]
                browser_stats["successes"] += model_stats["successes"]
                browser_stats["models"][model_type.value] = model_stats
                
                # Add to global stats
                total_attempts += model_stats["attempts"]
                total_successes += model_stats["successes"]
                
                # Add to model-specific summary
                if model_type.value not in stats["models"]:
                    stats["models"][model_type.value] = {
                        "attempts": 0,
                        "successes": 0,
                        "success_rate": 0.0,
                        "browsers": {}
                    }
                
                stats["models"][model_type.value]["attempts"] += model_stats["attempts"]
                stats["models"][model_type.value]["successes"] += model_stats["successes"]
                
                if stats["models"][model_type.value]["attempts"] > 0:
                    stats["models"][model_type.value]["success_rate"] = (
                        stats["models"][model_type.value]["successes"] / 
                        stats["models"][model_type.value]["attempts"]
                    )
                
                stats["models"][model_type.value]["browsers"][browser_type.value] = {
                    "attempts": model_stats["attempts"],
                    "successes": model_stats["successes"],
                    "success_rate": model_stats["success_rate"]
                }
            
            # Calculate browser success rate
            if browser_stats["attempts"] > 0:
                browser_stats["success_rate"] = browser_stats["successes"] / browser_stats["attempts"]
            
            # Add to global browser stats
            stats["browsers"][browser_type.value] = browser_stats
        
        # Update summary stats
        stats["summary"]["total_strategies"] = len(all_strategies)
        stats["summary"]["total_attempts"] = total_attempts
        stats["summary"]["total_successes"] = total_successes
        
        if total_attempts > 0:
            stats["summary"]["overall_success_rate"] = total_successes / total_attempts
        
        return stats
    
    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of recovery attempts.
        
        Returns:
            List of recovery history entries
        """
        return self.strategy_history.copy()
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze the performance of recovery strategies.
        
        Returns:
            Dictionary with performance analysis
        """
        # Get basic stats
        stats = self.get_strategy_stats()
        
        # Analyze history for trends
        history = self.get_recovery_history()
        
        # Group by day
        days = {}
        for entry in history:
            timestamp = entry.get("timestamp", "")
            if timestamp:
                day = timestamp.split("T")[0]  # Extract YYYY-MM-DD
                
                if day not in days:
                    days[day] = {
                        "attempts": 0,
                        "successes": 0,
                        "total_execution_time": 0.0,
                        "strategies": {}
                    }
                
                days[day]["attempts"] += 1
                if entry.get("success", False):
                    days[day]["successes"] += 1
                days[day]["total_execution_time"] += entry.get("execution_time", 0.0)
                
                # Track strategy-specific stats
                strategy_name = entry.get("strategy_name", "unknown")
                if strategy_name not in days[day]["strategies"]:
                    days[day]["strategies"][strategy_name] = {
                        "attempts": 0,
                        "successes": 0,
                        "total_execution_time": 0.0
                    }
                
                days[day]["strategies"][strategy_name]["attempts"] += 1
                if entry.get("success", False):
                    days[day]["strategies"][strategy_name]["successes"] += 1
                days[day]["strategies"][strategy_name]["total_execution_time"] += entry.get("execution_time", 0.0)
        
        # Convert to time series
        time_series = []
        for day, day_stats in sorted(days.items()):
            success_rate = day_stats["successes"] / day_stats["attempts"] if day_stats["attempts"] > 0 else 0.0
            avg_execution_time = day_stats["total_execution_time"] / day_stats["attempts"] if day_stats["attempts"] > 0 else 0.0
            
            time_series.append({
                "date": day,
                "attempts": day_stats["attempts"],
                "successes": day_stats["successes"],
                "success_rate": success_rate,
                "avg_execution_time": avg_execution_time,
                "strategies": day_stats["strategies"]
            })
        
        # Find best strategy for each browser/model combination
        best_strategies = {}
        
        for browser_type, browser_stats in stats["browsers"].items():
            best_strategies[browser_type] = {}
            
            for model_type, model_stats in browser_stats["models"].items():
                best_strategy = None
                best_score = -1.0
                
                for strategy_name, strategy_stats in model_stats["strategies"].items():
                    # Only consider strategies with at least 3 attempts
                    if strategy_stats["attempts"] >= 3:
                        # Calculate score based on success rate
                        score = strategy_stats["success_rate"]
                        
                        if score > best_score:
                            best_score = score
                            best_strategy = strategy_name
                
                best_strategies[browser_type][model_type] = {
                    "strategy": best_strategy,
                    "score": best_score
                }
        
        return {
            "stats": stats,
            "time_series": time_series,
            "best_strategies": best_strategies
        }

# Utility functions for browser recovery

def detect_browser_type(browser_name: str) -> BrowserType:
    """
    Detect browser type from browser name.
    
    Args:
        browser_name: Name of the browser
        
    Returns:
        BrowserType enum value
    """
    browser_name_lower = browser_name.lower()
    
    if "chrome" in browser_name_lower:
        return BrowserType.CHROME
    elif "firefox" in browser_name_lower:
        return BrowserType.FIREFOX
    elif "edge" in browser_name_lower:
        return BrowserType.EDGE
    elif "safari" in browser_name_lower:
        return BrowserType.SAFARI
    else:
        return BrowserType.UNKNOWN

def detect_model_type(model_name: str) -> ModelType:
    """
    Detect model type from model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelType enum value
    """
    model_name_lower = model_name.lower()
    
    # Text models
    if any(text_model in model_name_lower for text_model in 
           ["bert", "t5", "gpt", "llama", "opt", "falcon", "roberta", "xlnet", "bart"]):
        return ModelType.TEXT
    
    # Vision models
    elif any(vision_model in model_name_lower for vision_model in 
            ["vit", "resnet", "efficientnet", "yolo", "detr", "dino", "swin"]):
        return ModelType.VISION
    
    # Audio models
    elif any(audio_model in model_name_lower for audio_model in 
            ["whisper", "wav2vec", "hubert", "audioclip", "clap"]):
        return ModelType.AUDIO
    
    # Multimodal models
    elif any(multimodal_model in model_name_lower for multimodal_model in 
            ["clip", "llava", "blip", "xclip", "flamingo", "qwen-vl"]):
        return ModelType.MULTIMODAL
    
    # Default is generic
    return ModelType.GENERIC

def categorize_browser_failure(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Categorize a browser failure.
    
    Args:
        error: Exception object
        context: Additional context
        
    Returns:
        Dictionary with failure information
    """
    error_message = str(error)
    error_type = type(error).__name__
    
    # Default failure type
    failure_type = FailureType.UNKNOWN
    
    # Check error message to categorize failure
    if any(launch_error in error_message.lower() for launch_error in 
          ["failed to launch", "could not start", "executable not found"]):
        failure_type = FailureType.LAUNCH_FAILURE
    
    elif any(connection_error in error_message.lower() for connection_error in 
            ["connection refused", "failed to connect", "connection error", "websocket", "socket"]):
        failure_type = FailureType.CONNECTION_FAILURE
    
    elif any(timeout_error in error_message.lower() for timeout_error in 
            ["timeout", "timed out"]):
        failure_type = FailureType.TIMEOUT
    
    elif any(crash_error in error_message.lower() for crash_error in 
            ["crashed", "crash", "terminated", "killed"]):
        failure_type = FailureType.CRASH
    
    elif any(resource_error in error_message.lower() for resource_error in 
            ["out of memory", "memory exhausted", "insufficient resources", "resource"]):
        failure_type = FailureType.RESOURCE_EXHAUSTION
    
    elif any(gpu_error in error_message.lower() for gpu_error in 
            ["gpu", "webgpu", "graphics", "shader", "driver"]):
        failure_type = FailureType.GPU_ERROR
    
    elif any(api_error in error_message.lower() for api_error in 
            ["webnn", "webn", "api", "interface", "unsupported operation"]):
        failure_type = FailureType.API_ERROR
    
    elif any(internal_error in error_message.lower() for internal_error in 
            ["internal error", "internal failure", "unexpected error"]):
        failure_type = FailureType.INTERNAL_ERROR
    
    # Additional context-based categorization
    if context:
        if context.get("type") == "launch":
            failure_type = FailureType.LAUNCH_FAILURE
        elif context.get("type") == "connection":
            failure_type = FailureType.CONNECTION_FAILURE
        elif context.get("type") == "webgpu" or context.get("platform") == "webgpu":
            failure_type = FailureType.GPU_ERROR
        elif context.get("type") == "webnn" or context.get("platform") == "webnn":
            failure_type = FailureType.API_ERROR
    
    # Create failure info
    failure_info = {
        "error_type": error_type,
        "error_message": error_message,
        "failure_type": failure_type.value,
        "timestamp": datetime.now().isoformat(),
        "context": context or {},
        "stack_trace": traceback.format_exc()
    }
    
    return failure_info

async def recover_browser(bridge: Any, error: Exception, context: Dict[str, Any] = None) -> bool:
    """
    Recover from a browser failure using progressive recovery.
    
    Args:
        bridge: BrowserAutomationBridge instance
        error: Exception that caused the failure
        context: Additional context
        
    Returns:
        True if recovery was successful, False otherwise
    """
    # Get browser type and model type
    browser_type = BrowserType.UNKNOWN
    model_type = ModelType.GENERIC
    
    # Extract browser and model type from bridge if available
    if hasattr(bridge, 'browser_name'):
        browser_type = detect_browser_type(bridge.browser_name)
    
    if hasattr(bridge, 'model_type'):
        model_type = detect_model_type(bridge.model_type) if isinstance(bridge.model_type, str) else bridge.model_type
    
    # Extract from context if not available from bridge
    if browser_type == BrowserType.UNKNOWN and context and "browser" in context:
        browser_type = detect_browser_type(context["browser"])
    
    if model_type == ModelType.GENERIC and context and "model" in context:
        model_type = detect_model_type(context["model"])
    
    # Categorize failure
    failure_info = categorize_browser_failure(error, context)
    
    # Create recovery manager
    recovery_manager = ProgressiveRecoveryManager()
    
    # Determine starting recovery level based on failure type
    start_level = RecoveryLevel.MINIMAL
    
    failure_type = FailureType(failure_info["failure_type"])
    
    # Adjust starting level based on failure type
    if failure_type == FailureType.LAUNCH_FAILURE:
        start_level = RecoveryLevel.MODERATE  # Start with browser restart
    elif failure_type == FailureType.CRASH:
        start_level = RecoveryLevel.MODERATE  # Start with browser restart
    elif failure_type == FailureType.API_ERROR:
        start_level = RecoveryLevel.AGGRESSIVE  # Start with settings adjustment
    elif failure_type == FailureType.GPU_ERROR:
        start_level = RecoveryLevel.AGGRESSIVE  # Start with settings adjustment
    
    # Execute progressive recovery
    logger.info(f"Starting progressive recovery for {browser_type.value}/{model_type.value} from level {start_level.value}")
    success = await recovery_manager.execute_progressive_recovery(
        bridge, browser_type, model_type, failure_info, start_level
    )
    
    if success:
        logger.info(f"Browser recovery successful for {browser_type.value}/{model_type.value}")
    else:
        logger.error(f"All recovery strategies failed for {browser_type.value}/{model_type.value}")
    
    return success