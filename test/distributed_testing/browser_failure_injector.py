#!/usr/bin/env python3
"""
Browser Failure Injector

This module provides utilities for injecting artificial failures into Selenium browser
instances to test recovery strategies. It can simulate various types of failures, including
connection failures, resource exhaustion, GPU errors, and browser crashes.

The failure injector is designed to work seamlessly with the browser recovery strategies
and circuit breaker pattern to provide comprehensive fault tolerance testing.

Usage:
    from browser_failure_injector import BrowserFailureInjector
    
    # Create injector
    injector = BrowserFailureInjector(bridge)
    
    # Inject specific failure
    await injector.inject_failure(FailureType.CONNECTION_FAILURE)
    
    # Inject specific failure with intensity
    await injector.inject_failure(FailureType.RESOURCE_EXHAUSTION, intensity="severe")
    
    # Inject random failure
    await injector.inject_random_failure()
    
    # Get failure statistics
    stats = injector.get_failure_stats()
"""

import os
import sys
import time
import random
import logging
import anyio
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("browser_failure_injector")

# Ensure the test package root is on sys.path (for distributed_testing imports)
test_root = Path(__file__).resolve().parents[1]
if str(test_root) not in sys.path:
    sys.path.insert(0, str(test_root))

# Set more verbose logging if environment variable is set
if os.environ.get("BROWSER_FAILURE_INJECTOR_LOG_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

# Import recovery strategies if available
try:
    from distributed_testing.browser_recovery_strategies import (
        BrowserType, ModelType, FailureType, RecoveryLevel
    )
except ImportError:
    try:
        from browser_recovery_strategies import (
            BrowserType, ModelType, FailureType, RecoveryLevel
        )
    except ImportError:
        # Define fallback enums if recovery strategies not available
        class FailureType(Enum):
            """Types of browser failures."""
            CONNECTION_FAILURE = "connection_failure"
            RESOURCE_EXHAUSTION = "resource_exhaustion"
            GPU_ERROR = "gpu_error"
            API_ERROR = "api_error"
            TIMEOUT = "timeout"
            CRASH = "crash"
            INTERNAL_ERROR = "internal_error"
            UNKNOWN = "unknown"

# Import circuit breaker if available
try:
    from distributed_testing.circuit_breaker import CircuitBreaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    try:
        from circuit_breaker import CircuitBreaker
        CIRCUIT_BREAKER_AVAILABLE = True
    except ImportError:
        logger.warning("CircuitBreaker not available. Circuit breaker integration will be disabled.")
        CIRCUIT_BREAKER_AVAILABLE = False

class BrowserFailureInjector:
    """
    Utility class for injecting artificial failures into Selenium browser instances.
    
    This class provides methods to simulate various types of browser failures, allowing
    thorough testing of recovery strategies and fault tolerance mechanisms.
    """
    
    def __init__(self, bridge: Any, circuit_breaker: Optional[Any] = None, 
                 use_circuit_breaker: bool = True):
        """
        Initialize the failure injector.
        
        Args:
            bridge: SeleniumBrowserBridge instance to inject failures into
            circuit_breaker: Optional CircuitBreaker instance for failure tracking
            use_circuit_breaker: Whether to use circuit breaker integration
        """
        self.bridge = bridge
        self.use_circuit_breaker = use_circuit_breaker and CIRCUIT_BREAKER_AVAILABLE
        
        # Track injected failures
        self.injected_failures = []
        
        # Initialize circuit breaker or use provided one
        self.circuit_breaker = None
        if self.use_circuit_breaker:
            if circuit_breaker is not None:
                self.circuit_breaker = circuit_breaker
                logger.info("Using provided circuit breaker")
            elif CIRCUIT_BREAKER_AVAILABLE:
                # Create a default circuit breaker with reasonable defaults
                self.circuit_breaker = CircuitBreaker(
                    failure_threshold=5,
                    recovery_timeout=60,
                    half_open_after=30,
                    name="browser_failure_injector"
                )
                logger.info("Created new circuit breaker")
        
        logger.info("Browser failure injector initialized")
    
    async def inject_failure(self, failure_type: FailureType, intensity: str = "moderate") -> Dict[str, Any]:
        """
        Inject a specific type of failure into the browser.
        
        Args:
            failure_type: Type of failure to inject
            intensity: Failure intensity (mild, moderate, severe)
            
        Returns:
            Dictionary with failure information
        """
        # Check circuit breaker state if enabled
        if self.use_circuit_breaker and self.circuit_breaker:
            circuit_state = self.circuit_breaker.get_state()
            if circuit_state == "open":
                logger.warning(f"Circuit breaker is OPEN. Too many failures detected. Refusing to inject more failures.")
                return {
                    "timestamp": time.time(),
                    "failure_type": failure_type.value,
                    "intensity": intensity,
                    "browser": getattr(self.bridge, "browser_name", "unknown"),
                    "platform": getattr(self.bridge, "platform", "unknown"),
                    "success": False,
                    "circuit_breaker_open": True,
                    "error": "Circuit breaker is open due to too many failures"
                }
        
        # Record failure attempt
        failure_info = {
            "timestamp": time.time(),
            "failure_type": failure_type.value,
            "intensity": intensity,
            "browser": getattr(self.bridge, "browser_name", "unknown"),
            "platform": getattr(self.bridge, "platform", "unknown"),
            "success": False
        }
        
        try:
            # Inject the specific failure type
            if failure_type == FailureType.CONNECTION_FAILURE:
                success = await self._inject_connection_failure(intensity)
            elif failure_type == FailureType.RESOURCE_EXHAUSTION:
                success = await self._inject_resource_exhaustion(intensity)
            elif failure_type == FailureType.GPU_ERROR:
                success = await self._inject_gpu_error(intensity)
            elif failure_type == FailureType.API_ERROR:
                success = await self._inject_api_error(intensity)
            elif failure_type == FailureType.TIMEOUT:
                success = await self._inject_timeout(intensity)
            elif failure_type == FailureType.CRASH:
                success = await self._inject_crash(intensity)
            elif failure_type == FailureType.INTERNAL_ERROR:
                success = await self._inject_internal_error(intensity)
            else:
                success = await self._inject_unknown_failure(intensity)
            
            # Update failure information
            failure_info["success"] = success
            self.injected_failures.append(failure_info)
            
            # Notify circuit breaker if enabled and injection was successful
            if self.use_circuit_breaker and self.circuit_breaker and success:
                # For severe intensity, record as a failure in the circuit breaker
                if intensity == "severe":
                    self.circuit_breaker.record_failure()
                    failure_info["circuit_breaker_updated"] = True
                    logger.info(f"Recorded severe failure in circuit breaker")
                
                # For crash failures, always record in circuit breaker
                if failure_type == FailureType.CRASH:
                    self.circuit_breaker.record_failure()
                    failure_info["circuit_breaker_updated"] = True
                    logger.info(f"Recorded crash failure in circuit breaker")
            
            if success:
                logger.info(f"Successfully injected {failure_type.value} failure with {intensity} intensity")
            else:
                logger.warning(f"Failed to inject {failure_type.value} failure")
            
            return failure_info
            
        except Exception as e:
            logger.error(f"Error during failure injection: {str(e)}")
            
            # Record as failure in circuit breaker
            if self.use_circuit_breaker and self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            failure_info["success"] = False
            failure_info["error"] = str(e)
            self.injected_failures.append(failure_info)
            return failure_info
    
    async def inject_random_failure(self, excluded_types: Optional[List[FailureType]] = None,
                           exclude_severe: bool = False) -> Dict[str, Any]:
        """
        Inject a random failure type.
        
        Args:
            excluded_types: List of failure types to exclude
            exclude_severe: Whether to exclude severe intensity failures
            
        Returns:
            Dictionary with failure information
        """
        # Check circuit breaker state if enabled
        if self.use_circuit_breaker and self.circuit_breaker:
            circuit_state = self.circuit_breaker.get_state()
            if circuit_state == "open":
                logger.warning(f"Circuit breaker is OPEN. Too many failures detected. Refusing to inject more failures.")
                return {
                    "timestamp": time.time(),
                    "failure_type": "random",
                    "intensity": "unknown",
                    "browser": getattr(self.bridge, "browser_name", "unknown"),
                    "platform": getattr(self.bridge, "platform", "unknown"),
                    "success": False,
                    "circuit_breaker_open": True,
                    "error": "Circuit breaker is open due to too many failures"
                }
        
        # Get all failure types
        all_failure_types = list(FailureType)
        
        # Filter out excluded types
        if excluded_types:
            available_types = [ft for ft in all_failure_types if ft not in excluded_types]
        else:
            available_types = all_failure_types
        
        # If circuit breaker is in half-open state, avoid severe failures
        circuit_half_open = (self.use_circuit_breaker and self.circuit_breaker and 
                            self.circuit_breaker.get_state() == "half-open")
        
        if circuit_half_open or exclude_severe:
            # Avoid severe intensity when circuit breaker is recovering
            intensities = ["mild", "moderate"]
            logger.info("Circuit breaker in half-open state or exclude_severe=True, avoiding severe failures")
        else:
            intensities = ["mild", "moderate", "severe"]
        
        # Choose a random failure type
        failure_type = random.choice(available_types)
        
        # Choose a random intensity
        intensity = random.choice(intensities)
        
        # Inject the failure
        return await self.inject_failure(failure_type, intensity)
    
    async def _inject_connection_failure(self, intensity: str) -> bool:
        """
        Inject a connection failure.
        
        Args:
            intensity: Failure intensity
            
        Returns:
            True if failure was successfully injected
        """
        try:
            # Get driver
            driver = getattr(self.bridge, "driver", None)
            if not driver:
                logger.warning("No WebDriver available to inject connection failure")
                return False
            
            # Execute script to disrupt connection
            if intensity == "mild":
                # Temporarily block network access
                script = """
                    // Block network access temporarily
                    const origOpen = XMLHttpRequest.prototype.open;
                    XMLHttpRequest.prototype.open = function() {
                        throw new Error("XMLHttpRequest blocked by test");
                    };
                    
                    // Restore normal behavior after delay
                    setTimeout(() => {
                        XMLHttpRequest.prototype.open = origOpen;
                    }, 5000);
                    
                    return true;
                """
            elif intensity == "moderate":
                # Disrupt WebSockets
                script = """
                    // Override WebSocket constructor
                    const origWebSocket = WebSocket;
                    window.WebSocket = function() {
                        throw new Error("WebSocket connection blocked by test");
                    };
                    
                    // Restore after delay
                    setTimeout(() => {
                        window.WebSocket = origWebSocket;
                    }, 10000);
                    
                    return true;
                """
            else:  # severe
                # Major network disruption
                script = """
                    // Block all network requests
                    const origFetch = window.fetch;
                    const origXHR = XMLHttpRequest.prototype.open;
                    const origWS = WebSocket;
                    
                    window.fetch = function() { throw new Error("Fetch blocked by test"); };
                    XMLHttpRequest.prototype.open = function() { throw new Error("XHR blocked by test"); };
                    window.WebSocket = function() { throw new Error("WebSocket blocked by test"); };
                    
                    // Restore after longer delay
                    setTimeout(() => {
                        window.fetch = origFetch;
                        XMLHttpRequest.prototype.open = origXHR;
                        window.WebSocket = origWS;
                    }, 15000);
                    
                    return true;
                """
            
            # Execute the script
            driver.execute_script(script)
            
            # Additional artificial delay to ensure script executes
            await anyio.sleep(1)
            
            return True
        except Exception as e:
            logger.error(f"Error injecting connection failure: {str(e)}")
            return False
    
    async def _inject_resource_exhaustion(self, intensity: str) -> bool:
        """
        Inject a resource exhaustion failure.
        
        Args:
            intensity: Failure intensity
            
        Returns:
            True if failure was successfully injected
        """
        try:
            # Get driver
            driver = getattr(self.bridge, "driver", None)
            if not driver:
                logger.warning("No WebDriver available to inject resource exhaustion")
                return False
            
            # Execute script to consume resources
            if intensity == "mild":
                # Allocate a moderate amount of memory
                script = """
                    // Allocate 100MB of memory
                    const arrays = [];
                    for (let i = 0; i < 100; i++) {
                        arrays.push(new Uint8Array(1024 * 1024));  // 1MB each
                    }
                    
                    // Free memory after delay
                    setTimeout(() => {
                        while (arrays.length > 0) arrays.pop();
                    }, 5000);
                    
                    return true;
                """
            elif intensity == "moderate":
                # Allocate significant memory
                script = """
                    // Allocate 500MB of memory
                    const arrays = [];
                    for (let i = 0; i < 500; i++) {
                        arrays.push(new Uint8Array(1024 * 1024));  // 1MB each
                    }
                    
                    // Free memory after delay
                    setTimeout(() => {
                        while (arrays.length > 0) arrays.pop();
                    }, 8000);
                    
                    return true;
                """
            else:  # severe
                # Extreme memory allocation and CPU usage
                script = """
                    // Allocate memory aggressively
                    const arrays = [];
                    try {
                        for (let i = 0; i < 1000; i++) {
                            arrays.push(new Uint8Array(1024 * 1024));  // 1MB each
                        }
                    } catch (e) {
                        // Reached memory limit
                    }
                    
                    // CPU-intensive operation
                    const start = Date.now();
                    while (Date.now() - start < 2000) {
                        // Spin CPU for 2 seconds
                        Math.random() * Math.random();
                    }
                    
                    // Free memory after delay
                    setTimeout(() => {
                        while (arrays.length > 0) arrays.pop();
                    }, 10000);
                    
                    return true;
                """
            
            # Execute the script
            driver.execute_script(script)
            
            # Additional artificial delay to ensure script executes
            await anyio.sleep(1)
            
            return True
        except Exception as e:
            logger.error(f"Error injecting resource exhaustion: {str(e)}")
            return False
    
    async def _inject_gpu_error(self, intensity: str) -> bool:
        """
        Inject a GPU error.
        
        Args:
            intensity: Failure intensity
            
        Returns:
            True if failure was successfully injected
        """
        try:
            # Get driver
            driver = getattr(self.bridge, "driver", None)
            if not driver:
                logger.warning("No WebDriver available to inject GPU error")
                return False
            
            # Execute script to cause WebGPU/WebGL errors
            if intensity == "mild":
                # Create and immediately destroy context
                script = """
                    // Create and destroy WebGL context repeatedly
                    const canvas = document.createElement('canvas');
                    document.body.appendChild(canvas);
                    
                    for (let i = 0; i < 5; i++) {
                        const gl = canvas.getContext('webgl');
                        if (gl) gl.getExtension('WEBGL_lose_context').loseContext();
                    }
                    
                    document.body.removeChild(canvas);
                    return true;
                """
            elif intensity == "moderate":
                # More aggressive context handling
                script = """
                    // Attempt to create conflicting contexts
                    const canvases = [];
                    
                    for (let i = 0; i < 10; i++) {
                        const canvas = document.createElement('canvas');
                        canvas.width = 1024;
                        canvas.height = 1024;
                        document.body.appendChild(canvas);
                        canvases.push(canvas);
                        
                        // Try both webgl and webgl2
                        const gl = canvas.getContext('webgl');
                        const gl2 = canvas.getContext('webgl2');
                        
                        // Force some operations
                        if (gl) {
                            gl.clearColor(Math.random(), Math.random(), Math.random(), 1.0);
                            gl.clear(gl.COLOR_BUFFER_BIT);
                        }
                    }
                    
                    // Clean up after delay
                    setTimeout(() => {
                        for (const canvas of canvases) {
                            document.body.removeChild(canvas);
                        }
                    }, 5000);
                    
                    return true;
                """
            else:  # severe
                # Directly interfere with WebGPU if available
                script = """
                    // Attempt to disrupt WebGPU
                    try {
                        if (navigator.gpu) {
                            // Request adapter and then intentionally create problems
                            navigator.gpu.requestAdapter().then(adapter => {
                                if (adapter) {
                                    // Request device with invalid limits
                                    adapter.requestDevice({
                                        requiredLimits: {
                                            maxBindGroups: 999999999,  // Invalid value
                                            maxBindingsPerBindGroup: 999999999,  // Invalid value
                                        }
                                    }).catch(e => console.log('Expected error:', e));
                                }
                            });
                            
                            // Create multiple canvas contexts rapidly
                            const canvases = [];
                            for (let i = 0; i < 20; i++) {
                                const canvas = document.createElement('canvas');
                                canvas.width = 2048;
                                canvas.height = 2048;
                                document.body.appendChild(canvas);
                                canvases.push(canvas);
                                
                                // Try to get GPU context
                                if ('gpu' in navigator) {
                                    const context = canvas.getContext('webgpu');
                                }
                            }
                            
                            // Clean up after delay
                            setTimeout(() => {
                                for (const canvas of canvases) {
                                    document.body.removeChild(canvas);
                                }
                            }, 8000);
                        }
                    } catch (e) {
                        console.log('GPU error injection:', e);
                    }
                    
                    return true;
                """
            
            # Execute the script
            driver.execute_script(script)
            
            # Additional artificial delay to ensure script executes
            await anyio.sleep(1)
            
            return True
        except Exception as e:
            logger.error(f"Error injecting GPU error: {str(e)}")
            return False
    
    async def _inject_api_error(self, intensity: str) -> bool:
        """
        Inject an API error (WebNN/WebGPU API issues).
        
        Args:
            intensity: Failure intensity
            
        Returns:
            True if failure was successfully injected
        """
        try:
            # Get driver
            driver = getattr(self.bridge, "driver", None)
            if not driver:
                logger.warning("No WebDriver available to inject API error")
                return False
            
            # Execute script to cause API errors
            if intensity == "mild":
                # Cause minor API errors
                script = """
                    // Override WebGPU/WebNN API temporarily
                    const origGPU = navigator.gpu;
                    const origML = navigator.ml;
                    
                    if (origGPU) {
                        navigator.gpu = {
                            requestAdapter: function() {
                                return Promise.reject(new Error("Simulated WebGPU API error"));
                            }
                        };
                    }
                    
                    if (origML) {
                        navigator.ml = {
                            getNeuralNetworkContext: function() {
                                throw new Error("Simulated WebNN API error");
                            }
                        };
                    }
                    
                    // Restore after delay
                    setTimeout(() => {
                        if (origGPU) navigator.gpu = origGPU;
                        if (origML) navigator.ml = origML;
                    }, 5000);
                    
                    return true;
                """
            elif intensity == "moderate":
                # More disruptive API errors
                script = """
                    // Override WebGPU/WebNN APIs with partial functionality
                    const origGPU = navigator.gpu;
                    const origML = navigator.ml;
                    
                    if (origGPU) {
                        navigator.gpu = {
                            requestAdapter: function() {
                                return Promise.resolve({
                                    requestDevice: function() {
                                        return Promise.reject(new Error("Simulated WebGPU device error"));
                                    }
                                });
                            }
                        };
                    }
                    
                    if (origML) {
                        navigator.ml = {
                            getNeuralNetworkContext: function() {
                                return {
                                    createModel: function() {
                                        throw new Error("Simulated WebNN model creation error");
                                    }
                                };
                            }
                        };
                    }
                    
                    // Restore after delay
                    setTimeout(() => {
                        if (origGPU) navigator.gpu = origGPU;
                        if (origML) navigator.ml = origML;
                    }, 8000);
                    
                    return true;
                """
            else:  # severe
                # Complete API disruption
                script = """
                    // Override and replace WebGPU/WebNN APIs
                    if ('gpu' in navigator) delete navigator.gpu;
                    if ('ml' in navigator) delete navigator.ml;
                    
                    // Create proxies that always fail
                    Object.defineProperty(navigator, 'gpu', {
                        get: function() {
                            return {
                                requestAdapter: function() {
                                    return Promise.reject(new Error("Severe WebGPU API disruption"));
                                }
                            };
                        },
                        configurable: true
                    });
                    
                    Object.defineProperty(navigator, 'ml', {
                        get: function() {
                            return {
                                getNeuralNetworkContext: function() {
                                    throw new Error("Severe WebNN API disruption");
                                }
                            };
                        },
                        configurable: true
                    });
                    
                    // Force page reload after delay
                    setTimeout(() => {
                        location.reload();
                    }, 10000);
                    
                    return true;
                """
            
            # Execute the script
            driver.execute_script(script)
            
            # Additional artificial delay to ensure script executes
            await anyio.sleep(1)
            
            return True
        except Exception as e:
            logger.error(f"Error injecting API error: {str(e)}")
            return False
    
    async def _inject_timeout(self, intensity: str) -> bool:
        """
        Inject a timeout.
        
        Args:
            intensity: Failure intensity
            
        Returns:
            True if failure was successfully injected
        """
        try:
            # Get driver
            driver = getattr(self.bridge, "driver", None)
            if not driver:
                logger.warning("No WebDriver available to inject timeout")
                return False
            
            # Execute script to cause timeouts
            if intensity == "mild":
                # Short delay
                script = """
                    // Block main thread for 5 seconds
                    const start = Date.now();
                    while (Date.now() - start < 5000) {
                        // Busy wait
                        Math.random();
                    }
                    return true;
                """
                timeout_duration = 5  # seconds
            elif intensity == "moderate":
                # Longer delay
                script = """
                    // Block main thread for 10 seconds
                    const start = Date.now();
                    while (Date.now() - start < 10000) {
                        // Busy wait
                        Math.random();
                    }
                    return true;
                """
                timeout_duration = 10  # seconds
            else:  # severe
                # Very long delay
                script = """
                    // Block main thread for 30 seconds
                    const start = Date.now();
                    while (Date.now() - start < 30000) {
                        // Busy wait
                        Math.random();
                    }
                    return true;
                """
                timeout_duration = 30  # seconds
            
            # Execute the script with a timeout
            try:
                # Set a very short script timeout to force a timeout
                driver.set_script_timeout(timeout_duration / 2)
                driver.execute_script(script)
            except Exception as e:
                # This exception is expected (timeout)
                logger.info(f"Timeout injection successful: {str(e)}")
                
                # Reset script timeout
                driver.set_script_timeout(30)  # Reset to default
                return True
            
            # If we get here, the script completed without timing out
            logger.warning("Timeout injection did not cause a timeout as expected")
            return False
        except Exception as e:
            logger.error(f"Error injecting timeout: {str(e)}")
            return False
    
    async def _inject_crash(self, intensity: str) -> bool:
        """
        Inject a browser crash or severe error.
        
        Args:
            intensity: Failure intensity
            
        Returns:
            True if failure was successfully injected
        """
        try:
            # Get driver
            driver = getattr(self.bridge, "driver", None)
            if not driver:
                logger.warning("No WebDriver available to inject crash")
                return False
            
            # Execute script to cause crashes or severe errors
            if intensity == "mild":
                # Mild error, shouldn't crash but will cause errors
                script = """
                    // Cause JS errors
                    for (let i = 0; i < 10; i++) {
                        setTimeout(() => {
                            throw new Error("Simulated JS error #" + i);
                        }, i * 100);
                    }
                    return true;
                """
            elif intensity == "moderate":
                # More severe, may cause tab crashes in some browsers
                script = """
                    // Trigger recursive function to cause stack overflow
                    function recursiveFunction() {
                        recursiveFunction();
                    }
                    
                    try {
                        recursiveFunction();
                    } catch (e) {
                        // Stack overflow error caught
                    }
                    
                    // Create invalid CSS
                    const style = document.createElement('style');
                    style.textContent = 'body { ';
                    for (let i = 0; i < 10000; i++) {
                        style.textContent += `filter: blur(${i}px) `;
                    }
                    style.textContent += '}';
                    document.head.appendChild(style);
                    
                    return true;
                """
            else:  # severe
                # Severe crash attempt
                script = """
                    // Attempt to crash renderer process
                    
                    // Create large allocation
                    let crash = [];
                    try {
                        while (true) {
                            crash.push(new Uint8Array(1024 * 1024 * 10));  // 10MB chunks
                        }
                    } catch (e) {
                        // Allocation failed
                    }
                    
                    // Force layout thrashing
                    for (let i = 0; i < 5000; i++) {
                        document.body.style.width = (1000 + i % 500) + 'px';
                        document.body.offsetWidth; // Force layout
                    }
                    
                    // WebAssembly infinite loop (might crash some browsers)
                    try {
                        const bytes = new Uint8Array([
                            0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0,
                            7, 8, 1, 4, 108, 111, 111, 112, 0, 0, 10, 4, 1, 2, 0, 11
                        ]);
                        const module = new WebAssembly.Module(bytes);
                        const instance = new WebAssembly.Instance(module, {});
                        instance.exports.loop();
                    } catch (e) {
                        // WebAssembly error
                    }
                    
                    return true;
                """
            
            # Execute the script
            driver.execute_script(script)
            
            # Additional artificial delay to ensure script executes
            await anyio.sleep(1)
            
            # Check if browser is still responsive
            try:
                driver.execute_script("return document.title")
                logger.info("Browser still responsive after crash injection")
                return True
            except Exception:
                logger.info("Browser crashed or became unresponsive as expected")
                return True
        except Exception as e:
            logger.error(f"Error injecting crash: {str(e)}")
            return False
    
    async def _inject_internal_error(self, intensity: str) -> bool:
        """
        Inject an internal browser error.
        
        Args:
            intensity: Failure intensity
            
        Returns:
            True if failure was successfully injected
        """
        try:
            # Get driver
            driver = getattr(self.bridge, "driver", None)
            if not driver:
                logger.warning("No WebDriver available to inject internal error")
                return False
            
            # Execute script to cause internal errors
            if intensity == "mild":
                # Create DOM errors
                script = """
                    // Create DOM errors
                    for (let i = 0; i < 100; i++) {
                        const div = document.createElement('div');
                        div.id = 'error-div-' + i;
                        
                        // Create nested structure
                        let current = div;
                        for (let j = 0; j < 20; j++) {
                            const child = document.createElement('div');
                            child.style.margin = '10px';
                            child.style.padding = '10px';
                            child.textContent = 'Level ' + j;
                            current.appendChild(child);
                            current = child;
                        }
                        
                        document.body.appendChild(div);
                    }
                    
                    // Force layout calculations
                    for (let i = 0; i < 100; i++) {
                        document.getElementById('error-div-' + i).getBoundingClientRect();
                    }
                    
                    // Clean up after delay
                    setTimeout(() => {
                        for (let i = 0; i < 100; i++) {
                            const div = document.getElementById('error-div-' + i);
                            if (div) document.body.removeChild(div);
                        }
                    }, 5000);
                    
                    return true;
                """
            elif intensity == "moderate":
                # Create more severe internal errors
                script = """
                    // Create invalid markup
                    document.body.innerHTML += '<div id="bad-div">';
                    for (let i = 0; i < 1000; i++) {
                        document.body.innerHTML += '<span>Bad markup</span>';
                    }
                    document.body.innerHTML += '</div';
                    
                    // Force style recalculations
                    for (let i = 0; i < 100; i++) {
                        document.body.style.backgroundColor = i % 2 ? 'red' : 'blue';
                        document.body.getBoundingClientRect();
                    }
                    
                    // Create and delete iframe rapidly
                    for (let i = 0; i < 20; i++) {
                        const iframe = document.createElement('iframe');
                        iframe.src = 'about:blank';
                        document.body.appendChild(iframe);
                        setTimeout(() => {
                            document.body.removeChild(iframe);
                        }, i * 100);
                    }
                    
                    return true;
                """
            else:  # severe
                # Attempt to cause severe internal errors
                script = """
                    // Create document fragments with invalid nesting
                    const frag = document.createDocumentFragment();
                    for (let i = 0; i < 1000; i++) {
                        const div = document.createElement('div');
                        frag.appendChild(div);
                        
                        // Create invalid table structure
                        const table = document.createElement('table');
                        const row = document.createElement('tr');
                        
                        // Invalid nesting
                        row.innerHTML = '<div><td>Invalid</td></div>';
                        
                        table.appendChild(row);
                        div.appendChild(table);
                    }
                    document.body.appendChild(frag);
                    
                    // Force synchronous layout and style
                    for (let i = 0; i < 1000; i++) {
                        document.body.style.width = (500 + (i % 500)) + 'px';
                        document.body.style.height = (500 + (i % 500)) + 'px';
                        document.body.getBoundingClientRect();
                    }
                    
                    // Clear document after delay
                    setTimeout(() => {
                        document.body.innerHTML = '';
                    }, 8000);
                    
                    return true;
                """
            
            # Execute the script
            driver.execute_script(script)
            
            # Additional artificial delay to ensure script executes
            await anyio.sleep(1)
            
            return True
        except Exception as e:
            logger.error(f"Error injecting internal error: {str(e)}")
            return False
    
    async def _inject_unknown_failure(self, intensity: str) -> bool:
        """
        Inject an unknown/generic failure.
        
        Args:
            intensity: Failure intensity
            
        Returns:
            True if failure was successfully injected
        """
        # Choose a random failure type to inject
        failure_types = [
            self._inject_connection_failure,
            self._inject_resource_exhaustion,
            self._inject_gpu_error,
            self._inject_api_error,
            self._inject_timeout,
            self._inject_internal_error
        ]
        
        # Avoid crashes for unknown failures to prevent difficulty in recovery
        
        # Choose random failure type
        failure_func = random.choice(failure_types)
        
        # Inject the failure
        return await failure_func(intensity)
    
    def get_injected_failures(self) -> List[Dict[str, Any]]:
        """
        Get list of injected failures.
        
        Returns:
            List of failure information dictionaries
        """
        return self.injected_failures
    
    def get_failure_stats(self) -> Dict[str, Any]:
        """
        Get statistics about injected failures.
        
        Returns:
            Dictionary with failure statistics
        """
        # Count failures by type
        failure_counts = {}
        success_counts = {}
        intensity_counts = {"mild": 0, "moderate": 0, "severe": 0}
        
        for failure in self.injected_failures:
            failure_type = failure["failure_type"]
            success = failure["success"]
            intensity = failure.get("intensity", "unknown")
            
            if failure_type not in failure_counts:
                failure_counts[failure_type] = 0
                success_counts[failure_type] = 0
            
            failure_counts[failure_type] += 1
            if success:
                success_counts[failure_type] += 1
            
            # Count by intensity
            if intensity in intensity_counts:
                intensity_counts[intensity] += 1
        
        # Calculate success rates
        success_rates = {}
        for failure_type in failure_counts:
            if failure_counts[failure_type] > 0:
                success_rates[failure_type] = success_counts[failure_type] / failure_counts[failure_type]
            else:
                success_rates[failure_type] = 0.0
        
        # Basic stats
        stats = {
            "total_attempts": len(self.injected_failures),
            "total_successful": sum(success_counts.values()),
            "success_rate": sum(success_counts.values()) / len(self.injected_failures) if self.injected_failures else 0.0,
            "failure_counts": failure_counts,
            "success_counts": success_counts,
            "success_rates": success_rates,
            "intensity_counts": intensity_counts
        }
        
        # Add circuit breaker info if available
        if self.use_circuit_breaker and self.circuit_breaker:
            cb_state = self.circuit_breaker.get_state()
            cb_failures = self.circuit_breaker.get_failure_count()
            cb_threshold = self.circuit_breaker.failure_threshold
            
            stats["circuit_breaker"] = {
                "state": cb_state,
                "failure_count": cb_failures,
                "threshold": cb_threshold,
                "threshold_percent": (cb_failures / cb_threshold) * 100 if cb_threshold > 0 else 0,
                "recovery_timeout": self.circuit_breaker.recovery_timeout,
                "half_open_after": self.circuit_breaker.half_open_after
            }
        
        return stats

# Example usage
async def example_usage():
    """Example of using the browser failure injector."""
    try:
        # Import bridge for example
        from selenium_browser_bridge import BrowserConfiguration, SeleniumBrowserBridge
        
        # Import circuit breaker if available
        try:
            from circuit_breaker import CircuitBreaker
            # Create a circuit breaker with lower thresholds for demonstration
            circuit_breaker = CircuitBreaker(
                failure_threshold=3,  # Open circuit after 3 failures
                recovery_timeout=30,  # Stay open for 30 seconds
                half_open_after=15,   # Try half-open after 15 seconds
                name="demo_circuit_breaker"
            )
            print("Created circuit breaker for demonstration")
        except ImportError:
            circuit_breaker = None
            print("Circuit breaker not available for demonstration")
        
        # Create browser bridge
        config = BrowserConfiguration(
            browser_name="chrome",
            platform="webgpu",
            headless=True
        )
        bridge = SeleniumBrowserBridge(config)
        
        # Launch browser
        success = await bridge.launch(allow_simulation=True)
        if not success:
            print("Failed to launch browser")
            return
        
        # Create failure injector with circuit breaker
        injector = BrowserFailureInjector(
            bridge, 
            circuit_breaker=circuit_breaker,
            use_circuit_breaker=(circuit_breaker is not None)
        )
        
        # Inject various failures
        print("\nInjecting mild connection failure...")
        await injector.inject_failure(FailureType.CONNECTION_FAILURE, "mild")
        
        print("\nInjecting moderate resource exhaustion...")
        await injector.inject_failure(FailureType.RESOURCE_EXHAUSTION, "moderate")
        
        print("\nInjecting mild GPU error...")
        await injector.inject_failure(FailureType.GPU_ERROR, "mild")
        
        # Get circuit breaker status
        if circuit_breaker:
            print(f"\nCircuit breaker state: {circuit_breaker.get_state()}")
            print(f"Failure count: {circuit_breaker.get_failure_count()}")
        
        # Inject severe failures to trigger circuit breaker
        print("\nInjecting severe failures...")
        await injector.inject_failure(FailureType.CRASH, "severe")
        await injector.inject_failure(FailureType.CRASH, "severe")
        await injector.inject_failure(FailureType.CRASH, "severe")
        
        # Check circuit breaker status again
        if circuit_breaker:
            print(f"\nCircuit breaker state after severe failures: {circuit_breaker.get_state()}")
            print(f"Failure count: {circuit_breaker.get_failure_count()}")
        
        # Try injecting a failure when circuit is open
        print("\nTrying to inject failure with circuit open...")
        result = await injector.inject_random_failure()
        print(f"Result: {'Blocked by circuit breaker' if result.get('circuit_breaker_open') else 'Injected successfully'}")
        
        # Get detailed statistics
        stats = injector.get_failure_stats()
        print(f"\nFailure injection stats: {stats}")
        
        # Wait for circuit to transition to half-open
        if circuit_breaker and circuit_breaker.get_state() == "open":
            print(f"\nWaiting for circuit to transition to half-open state...")
            await anyio.sleep(circuit_breaker.half_open_after + 1)
            print(f"Circuit state after waiting: {circuit_breaker.get_state()}")
            
            # Try a safer failure with half-open circuit
            print("Injecting moderate failure with half-open circuit...")
            result = await injector.inject_random_failure(exclude_severe=True)
            print(f"Result: {result.get('success')}")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
    finally:
        # Close browser
        if 'bridge' in locals():
            await bridge.close()

if __name__ == "__main__":
    # Run the example
    anyio.run(example_usage())