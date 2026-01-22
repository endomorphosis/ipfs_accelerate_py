#!/usr/bin/env python3
"""
Selenium Browser Integration Test Suite

This test suite thoroughly tests the Selenium integration with browser recovery strategies,
using multiple browsers, model types, and test cases to ensure proper functionality.

Key features:
- Comprehensive test cases for different browser and model combinations
- Performance benchmarking for recovery strategies
- Cross-browser compatibility testing
- WebGPU/WebNN feature detection and validation
- Real-world recovery scenario testing
- Reporting and metrics collection

Usage:
    python test_selenium_browser_integration.py --browsers chrome,firefox,edge --models bert,whisper,vit,clip
"""

import os
import sys
import json
import time
import random
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("selenium_browser_integration_test")

# Set more verbose logging if environment variable is set
if os.environ.get("SELENIUM_BRIDGE_LOG_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

try:
    # Try to import from the module directly if we're inside the package
    from selenium_browser_bridge import (
        BrowserConfiguration, SeleniumBrowserBridge, SELENIUM_AVAILABLE
    )
except ImportError:
    try:
        # Try to import using relative path if we're importing from elsewhere
        from distributed_testing.selenium_browser_bridge import (
            BrowserConfiguration, SeleniumBrowserBridge, SELENIUM_AVAILABLE
        )
    except ImportError:
        logger.error("Failed to import SeleniumBrowserBridge. Make sure distributed_testing/selenium_browser_bridge.py exists")
        SELENIUM_AVAILABLE = False

try:
    # Try to import from the module directly if we're inside the package
    from browser_recovery_strategies import (
        BrowserType, ModelType, FailureType, RecoveryLevel, 
        detect_browser_type, detect_model_type
    )
    RECOVERY_AVAILABLE = True
except ImportError:
    try:
        # Try to import using relative path if we're importing from elsewhere
        from distributed_testing.browser_recovery_strategies import (
            BrowserType, ModelType, FailureType, RecoveryLevel, 
            detect_browser_type, detect_model_type
        )
        RECOVERY_AVAILABLE = True
    except ImportError:
        logger.error("Failed to import browser_recovery_strategies. Make sure distributed_testing/browser_recovery_strategies.py exists")
        RECOVERY_AVAILABLE = False
        
        # Define fallback classes and functions for testing
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
        
        def detect_browser_type(browser_name):
            """Fallback browser type detection."""
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
        
        def detect_model_type(model_name):
            """Fallback model type detection."""
            model_name_lower = model_name.lower()
            
            # Direct mapping for category names
            if model_name_lower == "text":
                return ModelType.TEXT
            elif model_name_lower == "vision":
                return ModelType.VISION
            elif model_name_lower == "audio":
                return ModelType.AUDIO
            elif model_name_lower == "multimodal":
                return ModelType.MULTIMODAL
                
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

class TestCase:
    """Test case for Selenium browser integration."""
    
    def __init__(self, name: str, browser_type: BrowserType, model_type: ModelType, 
                 platform: str = "auto", failure_injection: Optional[str] = None):
        """
        Initialize a test case.
        
        Args:
            name: Test case name
            browser_type: Browser type to test
            model_type: Model type to test
            platform: Platform to test (auto, webgpu, webnn)
            failure_injection: Type of failure to inject (None for no failure)
        """
        self.name = name
        self.browser_type = browser_type
        self.model_type = model_type
        self.platform = platform
        self.failure_injection = failure_injection
        self.start_time = None
        self.end_time = None
        self.success = False
        self.error = None
        self.results = {}
        self.metrics = {}
        self.recovery_attempts = 0
        self.recovery_success = False
        
    def get_model_name(self) -> str:
        """Get a representative model name for this model type."""
        if self.model_type == ModelType.TEXT:
            return "bert-base-uncased"
        elif self.model_type == ModelType.VISION:
            return "vit-base-patch16-224"
        elif self.model_type == ModelType.AUDIO:
            return "whisper-tiny"
        elif self.model_type == ModelType.MULTIMODAL:
            return "clip-vit-base-patch32"
        else:
            return "generic-model"
    
    def get_duration(self) -> Optional[float]:
        """Get the duration of the test case in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test case to a dictionary."""
        return {
            "name": self.name,
            "browser_type": self.browser_type.value,
            "model_type": self.model_type.value,
            "platform": self.platform,
            "failure_injection": self.failure_injection,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.get_duration(),
            "success": self.success,
            "error": str(self.error) if self.error else None,
            "results": self.results,
            "metrics": self.metrics,
            "recovery_attempts": self.recovery_attempts,
            "recovery_success": self.recovery_success
        }


class SeleniumBrowserIntegrationTest:
    """Test suite for Selenium browser integration."""
    
    def __init__(self, browsers: List[str], models: List[str], platforms: List[str] = None,
                 test_timeout: int = 60, retry_count: int = 2, report_path: Optional[str] = None):
        """
        Initialize the test suite.
        
        Args:
            browsers: List of browsers to test (chrome, firefox, edge, safari)
            models: List of models to test (text, vision, audio, multimodal)
            platforms: List of platforms to test (auto, webgpu, webnn)
            test_timeout: Timeout for each test in seconds
            retry_count: Number of retries for failed tests
            report_path: Path to save the test report (None for no report)
        """
        self.browsers = browsers
        self.models = models
        self.platforms = platforms or ["auto"]
        self.test_timeout = test_timeout
        self.retry_count = retry_count
        self.report_path = report_path
        self.test_cases = []
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Initialize browser types
        self.browser_types = {
            browser: detect_browser_type(browser) for browser in browsers
        }
        
        # Initialize model types
        self.model_types = {
            model: self._detect_model_type(model) for model in models
        }
        
        logger.info(f"Initialized test suite with browsers: {browsers}, models: {models}, platforms: {platforms}")
    
    def _detect_model_type(self, model: str) -> ModelType:
        """
        Detect model type from model identifier.
        
        Args:
            model: Model identifier (text, vision, audio, multimodal, or a specific model name)
            
        Returns:
            ModelType enum value
        """
        # Direct mapping for category names
        if model.lower() == "text":
            return ModelType.TEXT
        elif model.lower() == "vision":
            return ModelType.VISION
        elif model.lower() == "audio":
            return ModelType.AUDIO
        elif model.lower() == "multimodal":
            return ModelType.MULTIMODAL
        
        # Use detection function for specific model names
        return detect_model_type(model)
    
    def _generate_test_cases(self) -> None:
        """Generate test cases for all browser and model combinations."""
        test_id = 1
        
        # Generate standard test cases
        for browser in self.browsers:
            browser_type = self.browser_types[browser]
            
            for model in self.models:
                model_type = self.model_types[model]
                
                for platform in self.platforms:
                    # Create a standard test case (no failure)
                    test_case = TestCase(
                        name=f"test_{browser}_{model}_{platform}_{test_id}",
                        browser_type=browser_type,
                        model_type=model_type,
                        platform=platform
                    )
                    
                    self.test_cases.append(test_case)
                    test_id += 1
                    
                    # Create a test case with connection failure
                    test_case = TestCase(
                        name=f"test_{browser}_{model}_{platform}_connection_failure_{test_id}",
                        browser_type=browser_type,
                        model_type=model_type,
                        platform=platform,
                        failure_injection="connection_failure"
                    )
                    
                    self.test_cases.append(test_case)
                    test_id += 1
                    
                    # Create a test case with resource exhaustion
                    test_case = TestCase(
                        name=f"test_{browser}_{model}_{platform}_resource_exhaustion_{test_id}",
                        browser_type=browser_type,
                        model_type=model_type,
                        platform=platform,
                        failure_injection="resource_exhaustion"
                    )
                    
                    self.test_cases.append(test_case)
                    test_id += 1
        
        # Add special test cases for specific browser-model combinations
        
        # Test Firefox with audio models and compute shaders
        if "firefox" in self.browsers and any(self.model_types[m] == ModelType.AUDIO for m in self.models):
            test_case = TestCase(
                name=f"test_firefox_audio_compute_shaders_{test_id}",
                browser_type=BrowserType.FIREFOX,
                model_type=ModelType.AUDIO,
                platform="webgpu",
                failure_injection="gpu_error"  # Test GPU error recovery
            )
            self.test_cases.append(test_case)
            test_id += 1
        
        # Test Edge with text models and WebNN
        if "edge" in self.browsers and any(self.model_types[m] == ModelType.TEXT for m in self.models):
            test_case = TestCase(
                name=f"test_edge_text_webnn_{test_id}",
                browser_type=BrowserType.EDGE,
                model_type=ModelType.TEXT,
                platform="webnn",
                failure_injection="api_error"  # Test API error recovery
            )
            self.test_cases.append(test_case)
            test_id += 1
        
        # Test Chrome with vision models and shader precompilation
        if "chrome" in self.browsers and any(self.model_types[m] == ModelType.VISION for m in self.models):
            test_case = TestCase(
                name=f"test_chrome_vision_shader_precompile_{test_id}",
                browser_type=BrowserType.CHROME,
                model_type=ModelType.VISION,
                platform="webgpu",
                failure_injection=None  # No failure
            )
            self.test_cases.append(test_case)
            test_id += 1
        
        logger.info(f"Generated {len(self.test_cases)} test cases")
    
    def _generate_test_input(self, model_type: ModelType) -> Any:
        """
        Generate a test input for a given model type.
        
        Args:
            model_type: Model type
            
        Returns:
            Test input appropriate for the model type
        """
        if model_type == ModelType.TEXT:
            return "This is a test input for a text model."
        elif model_type == ModelType.VISION:
            # Return dimensions for a simple image input
            return {"width": 224, "height": 224, "channels": 3}
        elif model_type == ModelType.AUDIO:
            # Return audio duration and sample rate
            return {"duration_seconds": 3.0, "sample_rate": 16000}
        elif model_type == ModelType.MULTIMODAL:
            # Return both text and image dimensions
            return {
                "text": "This is a test input for a multimodal model.",
                "image": {"width": 224, "height": 224, "channels": 3}
            }
        else:
            return "Generic test input"
    
    async def _inject_failure(self, bridge: SeleniumBrowserBridge, failure_type: str) -> None:
        """
        Inject a specific failure for testing recovery.
        
        Args:
            bridge: SeleniumBrowserBridge instance
            failure_type: Type of failure to inject
        """
        logger.info(f"Injecting failure: {failure_type}")
        
        if failure_type == "connection_failure":
            # Simulate a connection failure by forcibly closing browser
            try:
                if bridge.driver:
                    bridge.driver.quit()
                    bridge.driver = None
                # Force a connection attempt that will fail
                await asyncio.sleep(0.5)
                if hasattr(bridge, 'check_browser_responsive'):
                    await bridge.check_browser_responsive()
            except Exception as e:
                logger.debug(f"Successfully injected connection_failure: {str(e)}")
        
        elif failure_type == "resource_exhaustion":
            # Simulate resource exhaustion
            try:
                if bridge.driver:
                    # Trigger memory pressure by running a memory-intensive script
                    script = """
                    function exhaustMemory() {
                        const arrays = [];
                        try {
                            // Allocate arrays until we run out of memory or hit a limit
                            for (let i = 0; i < 20; i++) {
                                const arr = new Uint8Array(100 * 1024 * 1024);  // 100MB chunks
                                arrays.push(arr);
                                for (let j = 0; j < arr.length; j++) {
                                    arr[j] = (j % 256);  // Fill with data to ensure allocation
                                }
                            }
                        } catch (e) {
                            return "Memory exhaustion triggered: " + e.message;
                        }
                        return "Allocated memory without errors";
                    }
                    return exhaustMemory();
                    """
                    result = bridge.driver.execute_script(script)
                    logger.debug(f"Memory exhaustion script result: {result}")
            except Exception as e:
                logger.debug(f"Successfully injected resource_exhaustion: {str(e)}")
        
        elif failure_type == "gpu_error":
            # Simulate a GPU error
            try:
                if bridge.driver:
                    # Trigger a GPU error by running an invalid WebGPU operation
                    script = """
                    async function triggerGpuError() {
                        try {
                            if (!navigator.gpu) {
                                return "WebGPU not available";
                            }
                            const adapter = await navigator.gpu.requestAdapter();
                            if (!adapter) {
                                return "WebGPU adapter not available";
                            }
                            const device = await adapter.requestDevice();
                            
                            // Create an invalid buffer (negative size)
                            try {
                                const buffer = device.createBuffer({
                                    size: -1,  // Invalid size
                                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                                });
                            } catch (e) {
                                return "GPU error triggered: " + e.message;
                            }
                            return "No GPU error triggered";
                        } catch (e) {
                            return "Error setting up WebGPU: " + e.message;
                        }
                    }
                    return triggerGpuError();
                    """
                    result = bridge.driver.execute_script(script)
                    logger.debug(f"GPU error script result: {result}")
            except Exception as e:
                logger.debug(f"Successfully injected gpu_error: {str(e)}")
        
        elif failure_type == "api_error":
            # Simulate an API error (for WebNN)
            try:
                if bridge.driver:
                    # Trigger an API error by running an invalid WebNN operation
                    script = """
                    async function triggerApiError() {
                        try {
                            if (typeof navigator.ml === 'undefined' || 
                                typeof navigator.ml.getNeuralNetworkContext === 'undefined') {
                                return "WebNN not available";
                            }
                            
                            const context = navigator.ml.getNeuralNetworkContext();
                            const builder = new MLGraphBuilder(context);
                            
                            // Create a tensor with invalid dimensions
                            try {
                                const invalidTensor = builder.input(
                                    'invalid_input',
                                    {dataType: 'float32', dimensions: [-1, -1]} // Invalid dimensions
                                );
                                const graph = await builder.build({});
                            } catch (e) {
                                return "API error triggered: " + e.message;
                            }
                            return "No API error triggered";
                        } catch (e) {
                            return "Error setting up WebNN: " + e.message;
                        }
                    }
                    return triggerApiError();
                    """
                    result = bridge.driver.execute_script(script)
                    logger.debug(f"API error script result: {result}")
            except Exception as e:
                logger.debug(f"Successfully injected api_error: {str(e)}")
    
    async def _configure_bridge_for_test(self, test_case: TestCase) -> SeleniumBrowserBridge:
        """
        Configure a SeleniumBrowserBridge for a specific test case.
        
        Args:
            test_case: Test case to configure for
            
        Returns:
            Configured SeleniumBrowserBridge instance
        """
        # Determine platform based on model and browser type
        platform = test_case.platform
        
        if platform == "auto":
            # Auto-select optimal platform based on model and browser
            if test_case.model_type == ModelType.TEXT and test_case.browser_type == BrowserType.EDGE:
                platform = "webnn"
            else:
                platform = "webgpu"
        
        # Create browser configuration
        config = BrowserConfiguration(
            browser_name=test_case.browser_type.value,
            platform=platform,
            headless=True,
            timeout=self.test_timeout
        )
        
        # Apply model-specific optimizations
        if test_case.model_type == ModelType.TEXT:
            # Text models - optimize for latency
            config.shader_precompilation = True
            config.max_batch_size = 1
            config.optimize_for = "latency"
            
        elif test_case.model_type == ModelType.VISION:
            # Vision models - optimize for throughput
            config.shader_precompilation = True
            config.max_batch_size = 4
            config.optimize_for = "throughput"
            
            # Add Chrome-specific optimizations for vision models
            if test_case.browser_type == BrowserType.CHROME:
                config.custom_args = [
                    "--enable-zero-copy",
                    "--enable-gpu-memory-buffer-video-frames"
                ]
            
        elif test_case.model_type == ModelType.AUDIO:
            # Audio models - enable compute shaders
            config.compute_shaders = True
            
            # Add Firefox-specific optimizations for audio models
            if test_case.browser_type == BrowserType.FIREFOX:
                config.custom_prefs = {
                    "dom.webgpu.workgroup_size": "256,1,1"
                }
            
        elif test_case.model_type == ModelType.MULTIMODAL:
            # Multimodal models - enable parallel loading
            config.parallel_loading = True
            config.optimize_for = "memory"
        
        # Create browser bridge
        bridge = SeleniumBrowserBridge(config)
        
        # Store model type in bridge for recovery strategies
        bridge.model_type = test_case.model_type
        
        return bridge
    
    async def _run_test_case(self, test_case: TestCase) -> Dict[str, Any]:
        """
        Run a single test case.
        
        Args:
            test_case: Test case to run
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running test case: {test_case.name}")
        
        test_case.start_time = datetime.now()
        
        # Check if Selenium is available
        if not SELENIUM_AVAILABLE:
            test_case.error = "Selenium not available"
            test_case.end_time = datetime.now()
            logger.error("Selenium not available, skipping test")
            return test_case.to_dict()
        
        # Check if recovery strategies are available
        if not RECOVERY_AVAILABLE:
            test_case.error = "Browser recovery strategies not available"
            test_case.end_time = datetime.now()
            logger.error("Browser recovery strategies not available, skipping test")
            return test_case.to_dict()
        
        # Configure the browser bridge
        bridge = await self._configure_bridge_for_test(test_case)
        
        try:
            # Launch browser
            success = await bridge.launch(allow_simulation=True)
            
            if not success:
                test_case.error = "Failed to launch browser"
                logger.error(f"Failed to launch browser for test case: {test_case.name}")
                test_case.end_time = datetime.now()
                return test_case.to_dict()
            
            # Get browser capabilities
            platform_support = {}
            try:
                platform_support = await bridge._check_platform_support()
                test_case.results["platform_support"] = platform_support
            except Exception as e:
                logger.warning(f"Failed to check platform support: {str(e)}")
            
            # Check if we're running in simulation mode
            test_case.results["simulation_mode"] = bridge.simulation_mode
            
            # Generate test input
            test_input = self._generate_test_input(test_case.model_type)
            
            # Inject failure if requested
            if test_case.failure_injection:
                await self._inject_failure(bridge, test_case.failure_injection)
            
            # Run the test
            model_name = test_case.get_model_name()
            result = await bridge.run_test(
                model_name=model_name,
                input_data=test_input
            )
            
            # Store test result
            test_case.results["test_output"] = result
            
            # Check for recovery attempts
            if hasattr(bridge, 'circuit_breaker') and bridge.circuit_breaker:
                metrics = bridge.circuit_breaker.get_metrics()
                test_case.recovery_attempts = metrics.get("failures", 0)
                test_case.recovery_success = metrics.get("failures", 0) > 0 and metrics.get("success_rate", 0) > 0
                test_case.metrics["circuit_breaker"] = metrics
            
            # Get bridge metrics
            test_case.metrics["bridge"] = bridge.get_metrics()
            
            # Mark test as successful
            test_case.success = True
            
        except Exception as e:
            test_case.error = e
            logger.error(f"Error running test case {test_case.name}: {str(e)}")
        finally:
            # Close the browser
            if bridge:
                try:
                    await bridge.close()
                except Exception as e:
                    logger.warning(f"Error closing browser: {str(e)}")
            
            test_case.end_time = datetime.now()
        
        return test_case.to_dict()
    
    async def run_tests(self) -> Dict[str, Any]:
        """
        Run all test cases.
        
        Returns:
            Dictionary with test results
        """
        # Generate test cases
        self._generate_test_cases()
        
        # Record start time
        self.start_time = datetime.now()
        
        # Run all test cases
        all_results = []
        
        for test_case in self.test_cases:
            retry_count = 0
            
            while retry_count <= self.retry_count:
                logger.info(f"Running test case {test_case.name} (attempt {retry_count + 1}/{self.retry_count + 1})")
                
                result = await self._run_test_case(test_case)
                all_results.append(result)
                
                # Break if test was successful
                if result.get("success", False):
                    break
                
                # Retry if test failed
                retry_count += 1
                
                if retry_count <= self.retry_count:
                    logger.info(f"Retrying test case {test_case.name}")
                    # Wait before retry
                    await asyncio.sleep(2)
        
        # Record end time
        self.end_time = datetime.now()
        
        # Generate summary
        summary = self._generate_summary(all_results)
        
        # Create final results
        self.results = {
            "summary": summary,
            "test_cases": all_results,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
        }
        
        # Save report if requested
        if self.report_path:
            self._save_report()
        
        return self.results
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of test results.
        
        Args:
            results: List of test results
            
        Returns:
            Dictionary with test summary
        """
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.get("success", False))
        failed_tests = total_tests - successful_tests
        
        # Count by browser
        browser_counts = {}
        browser_success_counts = {}
        
        for r in results:
            browser = r.get("browser_type", "unknown")
            
            if browser not in browser_counts:
                browser_counts[browser] = 0
                browser_success_counts[browser] = 0
            
            browser_counts[browser] += 1
            
            if r.get("success", False):
                browser_success_counts[browser] += 1
        
        browser_success_rates = {
            browser: browser_success_counts[browser] / count if count > 0 else 0
            for browser, count in browser_counts.items()
        }
        
        # Count by model
        model_counts = {}
        model_success_counts = {}
        
        for r in results:
            model = r.get("model_type", "unknown")
            
            if model not in model_counts:
                model_counts[model] = 0
                model_success_counts[model] = 0
            
            model_counts[model] += 1
            
            if r.get("success", False):
                model_success_counts[model] += 1
        
        model_success_rates = {
            model: model_success_counts[model] / count if count > 0 else 0
            for model, count in model_counts.items()
        }
        
        # Count by platform
        platform_counts = {}
        platform_success_counts = {}
        
        for r in results:
            platform = r.get("platform", "unknown")
            
            if platform not in platform_counts:
                platform_counts[platform] = 0
                platform_success_counts[platform] = 0
            
            platform_counts[platform] += 1
            
            if r.get("success", False):
                platform_success_counts[platform] += 1
        
        platform_success_rates = {
            platform: platform_success_counts[platform] / count if count > 0 else 0
            for platform, count in platform_counts.items()
        }
        
        # Count by failure injection
        failure_counts = {}
        failure_success_counts = {}
        
        for r in results:
            failure = r.get("failure_injection", "none")
            
            if failure not in failure_counts:
                failure_counts[failure] = 0
                failure_success_counts[failure] = 0
            
            failure_counts[failure] += 1
            
            if r.get("success", False):
                failure_success_counts[failure] += 1
        
        failure_success_rates = {
            failure: failure_success_counts[failure] / count if count > 0 else 0
            for failure, count in failure_counts.items()
        }
        
        # Count recovery attempts and successes
        total_recovery_attempts = sum(r.get("recovery_attempts", 0) for r in results)
        total_recovery_successes = sum(1 for r in results if r.get("recovery_success", False))
        
        recovery_success_rate = total_recovery_successes / total_recovery_attempts if total_recovery_attempts > 0 else 0
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "browser_counts": browser_counts,
            "browser_success_rates": browser_success_rates,
            "model_counts": model_counts,
            "model_success_rates": model_success_rates,
            "platform_counts": platform_counts,
            "platform_success_rates": platform_success_rates,
            "failure_counts": failure_counts,
            "failure_success_rates": failure_success_rates,
            "total_recovery_attempts": total_recovery_attempts,
            "total_recovery_successes": total_recovery_successes,
            "recovery_success_rate": recovery_success_rate
        }
    
    def _save_report(self) -> None:
        """Save the test report to a file."""
        try:
            with open(self.report_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Test report saved to {self.report_path}")
        except Exception as e:
            logger.error(f"Failed to save test report: {str(e)}")
    
    def print_summary(self) -> None:
        """Print a summary of test results to the console."""
        if not self.results or "summary" not in self.results:
            logger.warning("No test results available")
            return
        
        summary = self.results["summary"]
        
        print("\n" + "=" * 80)
        print("Selenium Browser Integration Test Summary")
        print("=" * 80)
        
        print(f"Total Tests:      {summary['total_tests']}")
        print(f"Successful Tests: {summary['successful_tests']}")
        print(f"Failed Tests:     {summary['failed_tests']}")
        print(f"Success Rate:     {summary['success_rate']:.2%}")
        
        print("-" * 80)
        print("Browser Success Rates:")
        for browser, rate in summary['browser_success_rates'].items():
            print(f"  {browser:10}: {rate:.2%}")
        
        print("-" * 80)
        print("Model Type Success Rates:")
        for model, rate in summary['model_success_rates'].items():
            print(f"  {model:10}: {rate:.2%}")
        
        print("-" * 80)
        print("Platform Success Rates:")
        for platform, rate in summary['platform_success_rates'].items():
            print(f"  {platform:10}: {rate:.2%}")
        
        print("-" * 80)
        print("Failure Injection Success Rates:")
        for failure, rate in summary['failure_success_rates'].items():
            failure_str = str(failure) if failure is not None else "none"
            print(f"  {failure_str:20}: {rate:.2%}")
        
        print("-" * 80)
        print("Recovery Statistics:")
        print(f"  Recovery Attempts:    {summary['total_recovery_attempts']}")
        print(f"  Recovery Successes:   {summary['total_recovery_successes']}")
        print(f"  Recovery Success Rate: {summary['recovery_success_rate']:.2%}")
        
        print("=" * 80)


async def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Selenium Browser Integration Test Suite")
    parser.add_argument("--browsers", default="chrome", 
                      help="Comma-separated list of browsers to test (chrome, firefox, edge, safari)")
    parser.add_argument("--models", default="text,vision,audio,multimodal", 
                      help="Comma-separated list of models to test (text, vision, audio, multimodal)")
    parser.add_argument("--platforms", default="auto", 
                      help="Comma-separated list of platforms to test (auto, webgpu, webnn)")
    parser.add_argument("--test-timeout", type=int, default=60, 
                      help="Timeout for each test in seconds")
    parser.add_argument("--retry-count", type=int, default=1, 
                      help="Number of retries for failed tests")
    parser.add_argument("--report-path", default=None, 
                      help="Path to save the test report (None for no report)")
    parser.add_argument("--simulate", action="store_true", 
                      help="Run tests in simulation mode even if Selenium is not available")
    args = parser.parse_args()
    
    # Convert comma-separated lists to Python lists
    browsers = [b.strip() for b in args.browsers.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    platforms = [p.strip() for p in args.platforms.split(",")]
    
    # Create test suite
    test_suite = SeleniumBrowserIntegrationTest(
        browsers=browsers,
        models=models,
        platforms=platforms,
        test_timeout=args.test_timeout,
        retry_count=args.retry_count,
        report_path=args.report_path
    )
    
    # Run tests
    await test_suite.run_tests()
    
    # Print summary
    test_suite.print_summary()

if __name__ == "__main__":
    asyncio.run(main())