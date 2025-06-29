#!/usr/bin/env python3
"""
Example usage of the Adaptive Circuit Breaker with ML optimization.

This script demonstrates how to use the Adaptive Circuit Breaker in various
scenarios, including basic usage, hardware-specific optimization, and
integration with browser testing.

Usage:
    python adaptive_circuit_breaker_example.py [--mode basic|advanced|browser|hardware]
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
import random
from typing import Any, Dict, List, Optional, Callable, Awaitable
from datetime import datetime

# Add parent directory to path to import modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("adaptive_circuit_breaker_example")

# Try to import required components
try:
    from adaptive_circuit_breaker import AdaptiveCircuitBreaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    logger.error("AdaptiveCircuitBreaker not available. Run this example from the distributed_testing directory.")
    CIRCUIT_BREAKER_AVAILABLE = False

# Try to import optional components for browser testing
try:
    from selenium_browser_bridge import SeleniumBrowserBridge, BrowserConfiguration
    from browser_failure_injector import BrowserFailureInjector, FailureType
    from browser_recovery_strategies import BrowserRecoveryManager, RecoveryLevel
    BROWSER_TESTING_AVAILABLE = True
except ImportError:
    logger.warning("Browser testing components not available. Browser example will be limited.")
    BROWSER_TESTING_AVAILABLE = False
    
# Check for visualization dependencies
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Visualization dependencies not available. Visualizations will be disabled.")
    VISUALIZATION_AVAILABLE = False


async def basic_example():
    """Basic example of using the Adaptive Circuit Breaker."""
    if not CIRCUIT_BREAKER_AVAILABLE:
        logger.error("AdaptiveCircuitBreaker not available. Skipping example.")
        return
    
    logger.info("Running basic Adaptive Circuit Breaker example")
    
    # Create a circuit breaker
    circuit_breaker = AdaptiveCircuitBreaker(
        name="basic_example",
        base_failure_threshold=3,
        base_recovery_timeout=5,
        base_half_open_timeout=1,
        optimization_enabled=False,  # Disable for simplicity in this example
        prediction_enabled=False     # Disable for simplicity in this example
    )
    
    # Define test operations
    async def successful_operation():
        await asyncio.sleep(0.1)
        logger.info("Operation succeeded")
        return "Success"
        
    async def failing_operation():
        await asyncio.sleep(0.1)
        logger.info("Operation failed")
        raise Exception("Simulated failure")
    
    # Run successful operations
    logger.info("Running successful operations...")
    for i in range(3):
        try:
            result = await circuit_breaker.execute(successful_operation)
            logger.info(f"Operation {i+1} result: {result}")
        except Exception as e:
            logger.error(f"Operation {i+1} failed: {str(e)}")
    
    # Run failing operations (should open the circuit after 3 failures)
    logger.info("\nRunning failing operations...")
    for i in range(5):
        try:
            result = await circuit_breaker.execute(failing_operation)
            logger.info(f"Operation {i+1} result: {result}")
        except Exception as e:
            logger.error(f"Operation {i+1} failed: {str(e)}")
    
    # Show circuit state
    state = circuit_breaker.get_state()
    logger.info(f"\nCircuit state: {state['state']}")
    
    # Wait for the circuit to transition to half-open state
    logger.info("\nWaiting for recovery timeout...")
    await asyncio.sleep(6)  # Wait longer than recovery_timeout (5s)
    
    # Try a successful operation which should close the circuit
    logger.info("\nTrying operation after recovery timeout...")
    try:
        result = await circuit_breaker.execute(successful_operation)
        logger.info(f"Operation result: {result}")
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
    
    # Show circuit state again
    state = circuit_breaker.get_state()
    logger.info(f"\nCircuit state after recovery: {state['state']}")
    
    # Cleanup
    circuit_breaker.close()
    logger.info("Basic example completed")


async def advanced_example():
    """Advanced example with ML optimization and prediction."""
    if not CIRCUIT_BREAKER_AVAILABLE:
        logger.error("AdaptiveCircuitBreaker not available. Skipping example.")
        return
    
    logger.info("Running advanced Adaptive Circuit Breaker example with ML optimization")
    
    # Create temporary directory for metrics and models
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="circuit_breaker_example_")
    model_path = os.path.join(temp_dir, "models/circuit_breaker")
    metrics_path = os.path.join(temp_dir, "metrics/circuit_breaker")
    
    # Create a circuit breaker with optimization and prediction
    circuit_breaker = AdaptiveCircuitBreaker(
        name="advanced_example",
        base_failure_threshold=3,
        base_recovery_timeout=5,
        base_half_open_timeout=1,
        optimization_enabled=True,
        prediction_enabled=True,
        model_path=model_path,
        metrics_path=metrics_path,
        min_data_points=5,         # Use a small value for the example
        learning_rate=0.5          # Higher learning rate for faster adaptation
    )
    
    # Define operations with different failure patterns
    async def operation_with_pattern(pattern_type: str, iteration: int):
        """
        Operation with different failure patterns.
        
        Args:
            pattern_type: Type of pattern ("normal", "timeout", "resource", "api")
            iteration: Current iteration number
        """
        # Determine success probability and error type based on pattern
        if pattern_type == "timeout":
            success_prob = 0.3  # High failure rate
            error_type = "timeout"
        elif pattern_type == "resource":
            success_prob = 0.4  # Medium failure rate
            error_type = "resource"
        elif pattern_type == "api":
            success_prob = 0.5  # Medium failure rate
            error_type = "api"
        else:  # normal
            success_prob = 0.7  # Low failure rate
            error_type = random.choice(["timeout", "connection", "resource", "api"])
        
        # Add pattern to make prediction easier
        if pattern_type == "api" and iteration > 3 and iteration <= 7:
            # After a few API errors, we'll have more serious failures
            if iteration > 5:
                error_type = "crash"
                success_prob = 0.1  # Very high failure rate
        
        # Determine if this operation will succeed or fail
        if random.random() < success_prob:
            # Successful operation
            await asyncio.sleep(0.05)
            return f"Success ({pattern_type})"
        else:
            # Failing operation
            await asyncio.sleep(0.05)
            
            # If in half-open state for some patterns, make recovery slower
            if pattern_type == "resource" and circuit_breaker.circuit_breaker.state == "half_open":
                await asyncio.sleep(0.5)  # Slower recovery for resource issues
                
            raise Exception(f"Simulated {error_type} failure")
    
    # Train the circuit breaker with different patterns
    patterns = ["normal", "timeout", "resource", "api"]
    pattern_cycles = 2  # Run each pattern multiple times to build up data
    
    for cycle in range(pattern_cycles):
        for pattern in patterns:
            logger.info(f"\nTraining with {pattern} pattern (cycle {cycle+1}/{pattern_cycles}):")
            
            # Reset circuit breaker for each pattern
            circuit_breaker.reset()
            
            # Run several operations with this pattern
            for i in range(10):
                try:
                    result = await circuit_breaker.execute(
                        lambda: operation_with_pattern(pattern, i)
                    )
                    logger.info(f"  Operation {i+1} succeeded")
                except Exception as e:
                    logger.info(f"  Operation {i+1} failed: {str(e)}")
                
                # Add a short delay between operations
                await asyncio.sleep(0.1)
            
            # Check parameters after this pattern
            state = circuit_breaker.get_state()
            logger.info(f"Parameters after {pattern} pattern:")
            logger.info(f"  Failure threshold: {state.get('current_failure_threshold', 'N/A')}")
            logger.info(f"  Recovery timeout: {state.get('current_recovery_timeout', 'N/A'):.2f}s")
            logger.info(f"  Half-open timeout: {state.get('current_half_open_timeout', 'N/A'):.2f}s")
    
    # Check if ML models were created
    if circuit_breaker.threshold_model is not None:
        logger.info("\nML optimization succeeded:")
        logger.info(f"  Threshold model created: Yes")
        logger.info(f"  Recovery model created: {circuit_breaker.recovery_timeout_model is not None}")
        logger.info(f"  Half-open model created: {circuit_breaker.half_open_timeout_model is not None}")
        logger.info(f"  Prediction model created: {circuit_breaker.prediction_model is not None}")
    else:
        logger.warning("\nML optimization did not complete (try running the example longer)")
    
    # Test early warning system (prediction)
    if circuit_breaker.prediction_model is not None:
        logger.info("\nTesting predictive circuit breaking:")
        
        # Reset circuit breaker
        circuit_breaker.reset()
        
        # Run API pattern which should eventually trigger prediction
        pattern = "api"
        opened_preemptively = False
        
        for i in range(10):
            try:
                result = await circuit_breaker.execute(
                    lambda: operation_with_pattern(pattern, i)
                )
                logger.info(f"  Operation {i+1} succeeded")
            except Exception as e:
                error_msg = str(e)
                if "preemptively opened" in error_msg:
                    logger.info(f"  Operation {i+1} detected early warning! Circuit opened preemptively.")
                    opened_preemptively = True
                else:
                    logger.info(f"  Operation {i+1} failed: {error_msg}")
            
            # Add a short delay between operations
            await asyncio.sleep(0.1)
        
        if opened_preemptively:
            logger.info("Predictive circuit breaking successfully demonstrated!")
        else:
            logger.info("Circuit was not opened preemptively (try running the example longer or adjusting patterns)")
    
    # Clean up
    circuit_breaker.close()
    import shutil
    shutil.rmtree(temp_dir)
    logger.info("Advanced example completed")


async def hardware_specific_example():
    """Example of hardware-specific circuit breakers."""
    if not CIRCUIT_BREAKER_AVAILABLE:
        logger.error("AdaptiveCircuitBreaker not available. Skipping example.")
        return
    
    logger.info("Running hardware-specific Adaptive Circuit Breaker example")
    
    # Create temporary directory for metrics and models
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="circuit_breaker_example_")
    
    # Create hardware-specific circuit breakers
    gpu_circuit_breaker = AdaptiveCircuitBreaker(
        name="gpu_example",
        base_failure_threshold=3,
        base_recovery_timeout=5,
        base_half_open_timeout=1,
        optimization_enabled=True,
        prediction_enabled=False,
        model_path=os.path.join(temp_dir, "models/gpu_circuit_breaker"),
        metrics_path=os.path.join(temp_dir, "metrics/gpu_circuit_breaker"),
        min_data_points=5,
        hardware_specific=True,
        hardware_type="gpu"
    )
    
    cpu_circuit_breaker = AdaptiveCircuitBreaker(
        name="cpu_example",
        base_failure_threshold=3,
        base_recovery_timeout=5,
        base_half_open_timeout=1,
        optimization_enabled=True,
        prediction_enabled=False,
        model_path=os.path.join(temp_dir, "models/cpu_circuit_breaker"),
        metrics_path=os.path.join(temp_dir, "metrics/cpu_circuit_breaker"),
        min_data_points=5,
        hardware_specific=True,
        hardware_type="cpu"
    )
    
    webgpu_circuit_breaker = AdaptiveCircuitBreaker(
        name="webgpu_example",
        base_failure_threshold=3,
        base_recovery_timeout=5,
        base_half_open_timeout=1,
        optimization_enabled=True,
        prediction_enabled=False,
        model_path=os.path.join(temp_dir, "models/webgpu_circuit_breaker"),
        metrics_path=os.path.join(temp_dir, "metrics/webgpu_circuit_breaker"),
        min_data_points=5,
        hardware_specific=True,
        hardware_type="webgpu"
    )
    
    # Define hardware-specific operations with different error patterns
    async def gpu_operation(success_rate: float = 0.7):
        """GPU operation with typical GPU-related errors."""
        if random.random() < success_rate:
            await asyncio.sleep(0.05)
            return "GPU Success"
        else:
            await asyncio.sleep(0.05)
            # GPU failures are often related to memory or resources
            error_type = random.choice(["memory", "resource", "gpu"])
            raise Exception(f"Simulated GPU {error_type} failure")
    
    async def cpu_operation(success_rate: float = 0.8):
        """CPU operation with typical CPU-related errors."""
        if random.random() < success_rate:
            await asyncio.sleep(0.05)
            return "CPU Success"
        else:
            await asyncio.sleep(0.05)
            # CPU failures are often related to timeouts or API issues
            error_type = random.choice(["timeout", "api", "connection"])
            raise Exception(f"Simulated CPU {error_type} failure")
    
    async def webgpu_operation(success_rate: float = 0.6):
        """WebGPU operation with WebGPU-specific errors."""
        if random.random() < success_rate:
            await asyncio.sleep(0.05)
            return "WebGPU Success"
        else:
            await asyncio.sleep(0.05)
            # WebGPU failures include browser-specific issues
            error_type = random.choice([
                "shader_compilation", "memory", "resource", "context_lost",
                "browser_crash", "timeout"
            ])
            raise Exception(f"Simulated WebGPU {error_type} failure")
    
    # Train the circuit breakers with hardware-specific patterns
    circuit_breakers = [
        (gpu_circuit_breaker, gpu_operation, 0.6, "GPU"),
        (cpu_circuit_breaker, cpu_operation, 0.7, "CPU"),
        (webgpu_circuit_breaker, webgpu_operation, 0.5, "WebGPU")
    ]
    
    # Train each circuit breaker with its corresponding operation
    for cb, operation, base_success_rate, name in circuit_breakers:
        logger.info(f"\nTraining {name} circuit breaker:")
        
        # Run multiple iterations with varying success rates to simulate different conditions
        success_rates = [
            base_success_rate,          # Normal operation
            base_success_rate * 0.5,    # High failure rate
            base_success_rate * 1.2,    # Low failure rate
        ]
        
        for i, success_rate in enumerate(success_rates):
            logger.info(f"  Phase {i+1}: success rate {success_rate:.2f}")
            
            # Reset circuit breaker for this phase
            cb.reset()
            
            # Run several operations
            for j in range(15):
                try:
                    result = await cb.execute(lambda: operation(success_rate))
                    if j % 5 == 0:  # Log less to reduce noise
                        logger.info(f"    Operation {j+1} succeeded")
                except Exception as e:
                    if j % 5 == 0:  # Log less to reduce noise
                        logger.info(f"    Operation {j+1} failed: {str(e)}")
                
                # Add a short delay between operations
                await asyncio.sleep(0.05)
            
            # Show circuit breaker state after this phase
            state = cb.get_state()
            logger.info(f"  {name} parameters after phase {i+1}:")
            logger.info(f"    Failure threshold: {state.get('current_failure_threshold', 'N/A')}")
            logger.info(f"    Recovery timeout: {state.get('current_recovery_timeout', 'N/A'):.2f}s")
    
    # Compare the optimized parameters between different hardware types
    logger.info("\nComparing hardware-specific parameters:")
    
    for cb, _, _, name in circuit_breakers:
        state = cb.get_state()
        logger.info(f"  {name}:")
        logger.info(f"    Failure threshold: {state.get('current_failure_threshold', 'N/A')}")
        logger.info(f"    Recovery timeout: {state.get('current_recovery_timeout', 'N/A'):.2f}s")
        logger.info(f"    Half-open timeout: {state.get('current_half_open_timeout', 'N/A'):.2f}s")
        
        # Show if ML models were created
        logger.info(f"    Threshold model created: {cb.threshold_model is not None}")
        logger.info(f"    Recovery model created: {cb.recovery_timeout_model is not None}")
        
        # Show feature importances if available
        if cb.feature_importances:
            logger.info(f"    Feature importances: {json.dumps(cb.feature_importances, indent=2)}")
    
    # Clean up
    for cb, _, _, _ in circuit_breakers:
        cb.close()
    
    import shutil
    shutil.rmtree(temp_dir)
    logger.info("Hardware-specific example completed")


async def browser_testing_example():
    """Example of using Adaptive Circuit Breaker with browser testing."""
    if not CIRCUIT_BREAKER_AVAILABLE:
        logger.error("AdaptiveCircuitBreaker not available. Skipping example.")
        return
    
    if not BROWSER_TESTING_AVAILABLE:
        logger.warning("Browser testing components not available. Using simulation mode.")
    
    logger.info("Running browser testing example with Adaptive Circuit Breaker")
    
    # Create temporary directory for metrics and models
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="circuit_breaker_example_")
    
    # Create circuit breaker for WebGPU browser testing
    circuit_breaker = AdaptiveCircuitBreaker(
        name="webgpu_browser_test",
        base_failure_threshold=3,
        base_recovery_timeout=10,
        base_half_open_timeout=3,
        optimization_enabled=True,
        prediction_enabled=True,
        model_path=os.path.join(temp_dir, "models/browser_circuit_breaker"),
        metrics_path=os.path.join(temp_dir, "metrics/browser_circuit_breaker"),
        min_data_points=5,
        hardware_specific=True,
        hardware_type="webgpu"
    )
    
    # Simulate browser testing if actual browser components aren't available
    if not BROWSER_TESTING_AVAILABLE:
        # Define simulated browser testing function
        async def simulated_browser_test(test_type: str, success_rate: float = 0.7):
            """Simulate a browser test with typical browser-related errors."""
            if random.random() < success_rate:
                await asyncio.sleep(0.1)
                return f"Browser test '{test_type}' succeeded"
            else:
                await asyncio.sleep(0.1)
                # Simulate different types of browser failures
                error_types = {
                    "load": ["connection_failure", "timeout", "resource_exhaustion"],
                    "render": ["gpu_error", "resource_exhaustion", "context_lost"],
                    "compute": ["shader_compilation", "gpu_error", "memory_error"],
                    "interactive": ["api_error", "timeout", "crash"]
                }
                
                # Select error type based on test type
                available_errors = error_types.get(test_type, ["unknown"])
                error_type = random.choice(available_errors)
                
                raise Exception(f"Simulated browser {error_type} during {test_type} test")
        
        # Run simulated browser tests with circuit breaker
        test_types = ["load", "render", "compute", "interactive"]
        
        for test_type in test_types:
            logger.info(f"\nRunning simulated {test_type} browser tests:")
            
            # Reset circuit breaker for each test type
            circuit_breaker.reset()
            
            # Success rate varies by test type
            success_rates = {
                "load": 0.8,
                "render": 0.7,
                "compute": 0.6,
                "interactive": 0.75
            }
            
            # Run multiple iterations
            for i in range(15):
                try:
                    result = await circuit_breaker.execute(
                        lambda: simulated_browser_test(test_type, success_rates[test_type])
                    )
                    logger.info(f"  Test {i+1} succeeded: {result}")
                except Exception as e:
                    logger.info(f"  Test {i+1} failed: {str(e)}")
                
                # Add a short delay between tests
                await asyncio.sleep(0.1)
            
            # Show optimized parameters
            state = circuit_breaker.get_state()
            logger.info(f"Parameters after {test_type} tests:")
            logger.info(f"  Failure threshold: {state.get('current_failure_threshold', 'N/A')}")
            logger.info(f"  Recovery timeout: {state.get('current_recovery_timeout', 'N/A'):.2f}s")
    
    else:
        # Use actual browser testing components
        # Create browser configuration
        browser_config = BrowserConfiguration(
            browser_name="chrome",
            platform="webgpu",
            headless=True
        )
        
        # Create browser bridge
        browser_bridge = SeleniumBrowserBridge(browser_config)
        
        # Create recovery manager with circuit breaker
        recovery_manager = BrowserRecoveryManager(circuit_breaker=circuit_breaker)
        
        # Create failure injector
        failure_injector = BrowserFailureInjector(
            browser_bridge,
            circuit_breaker=circuit_breaker
        )
        
        # Launch browser (with simulation if needed)
        logger.info("Launching browser (may use simulation if no real browser available)")
        await browser_bridge.launch(allow_simulation=True)
        
        # Define browser test function
        async def browser_test(test_type: str):
            """Run a browser test of specified type."""
            # In a real scenario, this would run actual browser interactions
            # For this example, we'll simulate different test behaviors
            await asyncio.sleep(0.1)
            
            # Simulate navigating to a test page
            test_url = f"https://example.com/webgpu_tests/{test_type}"
            
            if hasattr(browser_bridge, "driver") and browser_bridge.driver:
                # Use actual browser driver if available
                try:
                    browser_bridge.driver.get(test_url)
                    logger.info(f"Navigated to {test_url}")
                    # Simulate test execution
                    await asyncio.sleep(0.2)
                    return f"Browser test '{test_type}' completed"
                except Exception as e:
                    raise Exception(f"Browser test '{test_type}' failed: {str(e)}")
            else:
                # Simulate browser behavior
                await asyncio.sleep(0.1)
                return f"Simulated browser test '{test_type}' completed"
        
        # Inject failures to test circuit breaker
        failure_types = [
            FailureType.CONNECTION_FAILURE,
            FailureType.RESOURCE_EXHAUSTION,
            FailureType.GPU_ERROR,
            FailureType.TIMEOUT
        ]
        
        # Run tests with injected failures
        for i, failure_type in enumerate(failure_types):
            logger.info(f"\nTesting with {failure_type.value} failures:")
            
            # Reset circuit breaker
            circuit_breaker.reset()
            
            # Run multiple iterations
            for j in range(5):
                try:
                    # First run a normal test
                    test_type = random.choice(["load", "render", "compute", "interactive"])
                    result = await circuit_breaker.execute(lambda: browser_test(test_type))
                    logger.info(f"  Test {j*2+1} succeeded: {result}")
                    
                    # Then inject a failure
                    await failure_injector.inject_failure(failure_type, intensity=random.choice(["mild", "moderate", "severe"]))
                    
                    # Try to run another test (may fail due to injected failure)
                    try:
                        result = await circuit_breaker.execute(lambda: browser_test(test_type))
                        logger.info(f"  Test {j*2+2} succeeded despite injected failure")
                    except Exception as e:
                        logger.info(f"  Test {j*2+2} failed as expected after failure injection: {str(e)}")
                        
                except Exception as e:
                    logger.error(f"  Unexpected error: {str(e)}")
                
                # Add a short delay between tests
                await asyncio.sleep(0.3)
            
            # Show optimized parameters
            state = circuit_breaker.get_state()
            logger.info(f"Parameters after {failure_type.value} tests:")
            logger.info(f"  Failure threshold: {state.get('current_failure_threshold', 'N/A')}")
            logger.info(f"  Recovery timeout: {state.get('current_recovery_timeout', 'N/A'):.2f}s")
            logger.info(f"  State: {state.get('state', 'N/A')}")
        
        # Close browser
        await browser_bridge.close()
    
    # Clean up
    circuit_breaker.close()
    import shutil
    shutil.rmtree(temp_dir)
    logger.info("Browser testing example completed")


async def main():
    """Main entry point for the example script."""
    parser = argparse.ArgumentParser(description="Adaptive Circuit Breaker Examples")
    parser.add_argument("--mode", type=str, default="basic", 
                       choices=["basic", "advanced", "hardware", "browser", "all"],
                       help="Example mode to run")
    args = parser.parse_args()
    
    # Check if required modules are available
    if not CIRCUIT_BREAKER_AVAILABLE:
        logger.error("AdaptiveCircuitBreaker not available. Run this example from the distributed_testing directory.")
        return 1
    
    # Run the selected example
    if args.mode == "basic" or args.mode == "all":
        await basic_example()
        if args.mode == "all":
            logger.info("\n" + "="*80 + "\n")
    
    if args.mode == "advanced" or args.mode == "all":
        await advanced_example()
        if args.mode == "all":
            logger.info("\n" + "="*80 + "\n")
    
    if args.mode == "hardware" or args.mode == "all":
        await hardware_specific_example()
        if args.mode == "all":
            logger.info("\n" + "="*80 + "\n")
    
    if args.mode == "browser" or args.mode == "all":
        await browser_testing_example()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)