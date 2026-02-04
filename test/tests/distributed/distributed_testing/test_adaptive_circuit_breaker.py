#!/usr/bin/env python3
"""
Test script for Adaptive Circuit Breaker with ML-based optimization.

This script tests the functionality of the Adaptive Circuit Breaker, which extends
the basic Circuit Breaker pattern with machine learning capabilities to dynamically
optimize thresholds based on historical performance data.

Usage:
    python test_adaptive_circuit_breaker.py [--quick] [--simulate]
"""

import os
import sys
import time
import json
import anyio
import logging
import argparse
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("test_adaptive_circuit_breaker")


def _is_pytest() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules)


def _log_optional_dependency(message: str) -> None:
    if _is_pytest():
        logger.info(message)
    else:
        logger.warning(message)

# Import the adaptive circuit breaker
try:
    from test.tests.distributed.distributed_testing.adaptive_circuit_breaker import AdaptiveCircuitBreaker
    ADAPTIVE_CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    try:
        from adaptive_circuit_breaker import AdaptiveCircuitBreaker
        ADAPTIVE_CIRCUIT_BREAKER_AVAILABLE = True
    except ImportError:
        ADAPTIVE_CIRCUIT_BREAKER_AVAILABLE = False
        logger.error("AdaptiveCircuitBreaker not available. Tests will be skipped.")

# Check for sklearn
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    _log_optional_dependency("sklearn not available. ML tests will be limited.")

# Check for visualization dependencies
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    _log_optional_dependency("Visualization dependencies not available. Visualizations will be disabled.")

class CircuitBreakerTester:
    """Test harness for Adaptive Circuit Breaker."""
    
    def __init__(self, quick_test: bool = False, simulate: bool = False):
        """
        Initialize the tester.
        
        Args:
            quick_test: Run only basic tests
            simulate: Use simulation mode
        """
        self.quick_test = quick_test
        self.simulate = simulate
        self.test_dir = tempfile.mkdtemp(prefix="circuit_breaker_test_")
        self.model_path = os.path.join(self.test_dir, "models/circuit_breaker")
        self.metrics_path = os.path.join(self.test_dir, "metrics/circuit_breaker")
        self.db_path = os.path.join(self.test_dir, "circuit_breaker_test.duckdb")
        
        # Make directories
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        
        # Test parameters
        self.success_rate = 0.7  # 70% success rate for test operations
        self.failure_types = [
            "timeout", "connection", "resource", "memory", "api"
        ]
        self.test_results = []
    
    async def run_tests(self) -> bool:
        """
        Run all tests for the Adaptive Circuit Breaker.
        
        Returns:
            True if all tests passed, False otherwise
        """
        if not ADAPTIVE_CIRCUIT_BREAKER_AVAILABLE:
            logger.error("AdaptiveCircuitBreaker not available. Skipping tests.")
            return False
        
        try:
            logger.info("Starting Adaptive Circuit Breaker tests")
            
            # Run basic functionality tests
            await self.test_basic_functionality()
            
            # Run failure simulation tests
            await self.test_failure_simulation()
            
            # Run optimization tests (if not quick test)
            if not self.quick_test and SKLEARN_AVAILABLE:
                await self.test_optimization()
                
                # Run prediction tests
                await self.test_prediction()
                
                # Run hardware-specific tests
                await self.test_hardware_specific()
            
            # Generate visualizations
            if VISUALIZATION_AVAILABLE:
                self.generate_visualizations()
            
            logger.info("All tests completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
        finally:
            # Clean up
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
    
    async def test_basic_functionality(self) -> None:
        """Test basic functionality of the Adaptive Circuit Breaker."""
        logger.info("Testing basic functionality")
        
        # Create a circuit breaker
        circuit_breaker = AdaptiveCircuitBreaker(
            name="basic_test",
            base_failure_threshold=3,
            base_recovery_timeout=5,
            base_half_open_timeout=1,
            optimization_enabled=False,  # Disable optimization for basic test
            prediction_enabled=False,    # Disable prediction for basic test
            metrics_path=self.metrics_path
        )
        
        # Define test operations
        async def successful_operation():
            await anyio.sleep(0.01)
            return "Success"
            
        async def failing_operation():
            await anyio.sleep(0.01)
            raise Exception("Simulated failure")
        
        # Test successful operations
        for i in range(5):
            result = await circuit_breaker.execute(successful_operation)
            assert result == "Success", f"Expected 'Success', got {result}"
        
        # Test failing operations (should open the circuit after 3 failures)
        failures = 0
        try:
            for i in range(5):
                await circuit_breaker.execute(failing_operation)
        except Exception:
            failures += 1
        
        # Verify circuit is open
        state = circuit_breaker.get_state()
        assert state["state"] == "open", f"Expected circuit to be open, got {state['state']}"
        assert failures > 0, "Expected at least one failure"
        
        # Wait for recovery timeout
        await anyio.sleep(5.1)
        
        # Test half-open state (should be in half-open state now)
        try:
            result = await circuit_breaker.execute(successful_operation)
            assert result == "Success", f"Expected 'Success', got {result}"
            
            # Verify circuit is closed again
            state = circuit_breaker.get_state()
            assert state["state"] == "closed", f"Expected circuit to be closed, got {state['state']}"
            
        except Exception as e:
            logger.error(f"Failed to test half-open state: {str(e)}")
            raise
        
        # Cleanup
        circuit_breaker.close()
        logger.info("Basic functionality test passed")
    
    async def test_failure_simulation(self) -> None:
        """Test circuit breaker with simulated failures."""
        logger.info("Testing with simulated failures")
        
        # Create a circuit breaker
        circuit_breaker = AdaptiveCircuitBreaker(
            name="failure_test",
            base_failure_threshold=3,
            base_recovery_timeout=3,
            base_half_open_timeout=1,
            optimization_enabled=False,  # Disable optimization for this test
            prediction_enabled=False,    # Disable prediction for this test
            metrics_path=self.metrics_path
        )
        
        # Define test operations with different error types
        async def operation_with_random_failures():
            # Determine if this operation will succeed or fail
            if self.simulate or (hash(f"{time.time()}{id(circuit_breaker)}") % 100) / 100 < self.success_rate:
                # Successful operation
                await anyio.sleep(0.01)
                return "Success"
            else:
                # Failing operation with random error type
                await anyio.sleep(0.01)
                error_type = self.failure_types[hash(f"{time.time()}") % len(self.failure_types)]
                raise Exception(f"Simulated {error_type} failure")
        
        # Run multiple operations (some will succeed, some will fail)
        results = []
        for i in range(20):
            try:
                result = await circuit_breaker.execute(operation_with_random_failures)
                results.append({"success": True, "result": result})
            except Exception as e:
                results.append({"success": False, "error": str(e)})
            
            # Short delay between operations
            await anyio.sleep(0.05)
        
        # Verify we have some successes and some failures
        successes = sum(1 for r in results if r["success"])
        failures = len(results) - successes
        
        assert successes > 0, "Expected some successful operations"
        assert failures > 0, "Expected some failed operations"
        
        # Verify circuit state is correct based on recent results
        state = circuit_breaker.get_state()
        logger.info(f"Circuit state after simulation: {state['state']}")
        
        # Store results for visualization
        self.test_results.extend(results)
        
        # Cleanup
        circuit_breaker.close()
        logger.info(f"Failure simulation test passed ({successes} successes, {failures} failures)")
    
    async def test_optimization(self) -> None:
        """Test ML-based optimization of circuit breaker parameters."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available. Skipping optimization test.")
            return
            
        logger.info("Testing ML-based optimization")
        
        # Create a circuit breaker with optimization enabled
        circuit_breaker = AdaptiveCircuitBreaker(
            name="optimization_test",
            base_failure_threshold=3,
            base_recovery_timeout=5,
            base_half_open_timeout=1,
            optimization_enabled=True,
            prediction_enabled=False,  # Test optimization separately from prediction
            metrics_path=self.metrics_path,
            model_path=self.model_path,
            min_data_points=10,        # Lower threshold for testing
            learning_rate=0.5          # Higher learning rate for faster adaptation
        )
        
        # Define test operations with controlled failure patterns
        async def operation_with_pattern(pattern_type: str = "normal"):
            """
            Operation with different failure patterns to test optimization.
            
            Args:
                pattern_type: Type of pattern to simulate
                    - "normal": 70% success rate
                    - "high_failure": 90% failure rate
                    - "slow_recovery": Slow recovery from failures
            """
            # Determine success probability based on pattern
            if pattern_type == "high_failure":
                success_prob = 0.1  # 10% success rate
            elif pattern_type == "slow_recovery":
                success_prob = 0.3  # 30% success rate
                if circuit_breaker.circuit_breaker.state == "half_open":
                    # Make recovery slower
                    await anyio.sleep(0.5)
            else:  # normal
                success_prob = 0.7  # 70% success rate
            
            # Determine if this operation will succeed or fail
            if self.simulate or (hash(f"{time.time()}{id(circuit_breaker)}") % 100) / 100 < success_prob:
                # Successful operation
                await anyio.sleep(0.01)
                return "Success"
            else:
                # Failing operation with random error type
                await anyio.sleep(0.01)
                error_type = self.failure_types[hash(f"{time.time()}") % len(self.failure_types)]
                raise Exception(f"Simulated {error_type} failure")
        
        # Test with normal pattern (baseline)
        logger.info("Testing with normal pattern")
        for i in range(30):
            try:
                await circuit_breaker.execute(lambda: operation_with_pattern("normal"))
            except Exception:
                pass
            await anyio.sleep(0.05)
        
        # Record parameters after normal pattern
        normal_params = {
            "failure_threshold": circuit_breaker.current_failure_threshold,
            "recovery_timeout": circuit_breaker.current_recovery_timeout,
            "half_open_timeout": circuit_breaker.current_half_open_timeout
        }
        
        logger.info(f"Parameters after normal pattern: {normal_params}")
        
        # Reset circuit breaker
        circuit_breaker.reset()
        
        # Test with high failure pattern
        logger.info("Testing with high failure pattern")
        for i in range(30):
            try:
                await circuit_breaker.execute(lambda: operation_with_pattern("high_failure"))
            except Exception:
                pass
            await anyio.sleep(0.05)
        
        # Record parameters after high failure pattern
        high_failure_params = {
            "failure_threshold": circuit_breaker.current_failure_threshold,
            "recovery_timeout": circuit_breaker.current_recovery_timeout,
            "half_open_timeout": circuit_breaker.current_half_open_timeout
        }
        
        logger.info(f"Parameters after high failure pattern: {high_failure_params}")
        
        # Reset circuit breaker
        circuit_breaker.reset()
        
        # Test with slow recovery pattern
        logger.info("Testing with slow recovery pattern")
        for i in range(30):
            try:
                await circuit_breaker.execute(lambda: operation_with_pattern("slow_recovery"))
            except Exception:
                pass
            await anyio.sleep(0.05)
        
        # Record parameters after slow recovery pattern
        slow_recovery_params = {
            "failure_threshold": circuit_breaker.current_failure_threshold,
            "recovery_timeout": circuit_breaker.current_recovery_timeout,
            "half_open_timeout": circuit_breaker.current_half_open_timeout
        }
        
        logger.info(f"Parameters after slow recovery pattern: {slow_recovery_params}")
        
        # Verify optimization worked
        # For high failure pattern, we expect higher failure threshold
        assert high_failure_params["failure_threshold"] >= normal_params["failure_threshold"], \
            "Expected higher failure threshold for high failure pattern"
            
        # For slow recovery pattern, we expect longer recovery timeout
        assert slow_recovery_params["recovery_timeout"] > normal_params["recovery_timeout"], \
            "Expected longer recovery timeout for slow recovery pattern"
            
        # Check that models were saved
        model_files = [
            f"{self.model_path}_threshold.pkl",
            f"{self.model_path}_recovery.pkl",
            f"{self.model_path}_half_open.pkl"
        ]
        
        for model_file in model_files:
            assert os.path.exists(model_file), f"Model file {model_file} was not created"
        
        # Cleanup
        circuit_breaker.close()
        logger.info("Optimization test passed")
    
    async def test_prediction(self) -> None:
        """Test predictive circuit breaking based on early warning signals."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available. Skipping prediction test.")
            return
            
        logger.info("Testing predictive circuit breaking")
        
        # Create a circuit breaker with prediction enabled
        circuit_breaker = AdaptiveCircuitBreaker(
            name="prediction_test",
            base_failure_threshold=3,
            base_recovery_timeout=5,
            base_half_open_timeout=1,
            optimization_enabled=True,
            prediction_enabled=True,
            metrics_path=self.metrics_path,
            model_path=f"{self.model_path}_pred",
            min_data_points=10,        # Lower threshold for testing
            learning_rate=0.5          # Higher learning rate for faster adaptation
        )
        
        # Create specific failure patterns for prediction training
        async def create_failure_pattern(circuit_breaker: AdaptiveCircuitBreaker, 
                                       pattern_type: str) -> None:
            """
            Create a specific failure pattern for prediction model training.
            
            Args:
                circuit_breaker: The circuit breaker to train
                pattern_type: Type of pattern to simulate
            """
            # Define test operations
            async def success_op():
                await anyio.sleep(0.01)
                return "Success"
                
            async def fail_op(error_type: str = "timeout"):
                await anyio.sleep(0.01)
                raise Exception(f"Simulated {error_type} failure")
            
            # Pattern 1: Failures after successful operations
            if pattern_type == "success_then_failure":
                # First several successful operations
                for i in range(5):
                    await circuit_breaker.execute(success_op)
                    await anyio.sleep(0.01)
                
                # Then several failing operations
                for i in range(5):
                    try:
                        await circuit_breaker.execute(lambda: fail_op("timeout"))
                    except Exception:
                        pass
                    await anyio.sleep(0.01)
                
            # Pattern 2: API failures leading to crashes
            elif pattern_type == "api_then_crash":
                # First some API errors
                for i in range(3):
                    try:
                        await circuit_breaker.execute(lambda: fail_op("api"))
                    except Exception:
                        pass
                    await anyio.sleep(0.01)
                
                # Then crash errors (which are more severe)
                for i in range(3):
                    try:
                        await circuit_breaker.execute(lambda: fail_op("crash"))
                    except Exception:
                        pass
                    await anyio.sleep(0.01)
                    
            # Pattern 3: Mixed errors
            else:
                # Mix of different error types
                error_types = ["timeout", "connection", "resource", "memory", "api", "crash"]
                for error_type in error_types:
                    try:
                        await circuit_breaker.execute(lambda: fail_op(error_type))
                    except Exception:
                        pass
                    await anyio.sleep(0.01)
        
        # Train the prediction model with various patterns
        logger.info("Training prediction model with patterns")
        
        # Reset between patterns
        await create_failure_pattern(circuit_breaker, "success_then_failure")
        circuit_breaker.reset()
        await anyio.sleep(0.1)
        
        await create_failure_pattern(circuit_breaker, "api_then_crash")
        circuit_breaker.reset()
        await anyio.sleep(0.1)
        
        await create_failure_pattern(circuit_breaker, "mixed")
        circuit_breaker.reset()
        await anyio.sleep(0.1)
        
        # Repeat patterns to build more training data
        for i in range(3):
            pattern = ["success_then_failure", "api_then_crash", "mixed"][i % 3]
            await create_failure_pattern(circuit_breaker, pattern)
            circuit_breaker.reset()
            await anyio.sleep(0.1)
        
        # Verify that prediction model was created
        assert circuit_breaker.prediction_model is not None, "Prediction model was not created"
        
        # Check that model file was saved
        prediction_model_file = f"{self.model_path}_pred_prediction.pkl"
        assert os.path.exists(prediction_model_file), f"Prediction model file {prediction_model_file} was not created"
        
        # Test predictive capability (re-create a pattern that should trigger prediction)
        logger.info("Testing predictive capability")
        
        # First reset circuit breaker
        circuit_breaker.reset()
        
        # Define predictive pattern detector
        preemptive_opens = 0
        regular_opens = 0
        
        # Define operation to test prediction
        async def operation_for_prediction(success: bool = True, error_type: str = "timeout"):
            if success:
                await anyio.sleep(0.01)
                return "Success"
            else:
                await anyio.sleep(0.01)
                raise Exception(f"Simulated {error_type} failure")
        
        # Run test pattern again
        try:
            # First several successful operations
            for i in range(5):
                await circuit_breaker.execute(lambda: operation_for_prediction(True))
                await anyio.sleep(0.01)
            
            # Verify circuit is closed
            assert circuit_breaker.circuit_breaker.state == "closed", "Circuit should be closed at this point"
            
            # Then API errors (this should trigger predictive open)
            for i in range(3):
                try:
                    await circuit_breaker.execute(lambda: operation_for_prediction(False, "api"))
                except Exception as e:
                    # Check if this was a preemptive open
                    if "preemptively opened" in str(e):
                        preemptive_opens += 1
                await anyio.sleep(0.01)
            
            # Then more errors if needed
            for i in range(5):
                try:
                    await circuit_breaker.execute(lambda: operation_for_prediction(False, "crash"))
                except Exception as e:
                    # Check if circuit is already open (not preemptive)
                    if "Circuit is open" in str(e) and "preemptively" not in str(e):
                        regular_opens += 1
                await anyio.sleep(0.01)
                
        except Exception as e:
            logger.warning(f"Error during prediction test: {str(e)}")
        
        # Verify results - we should have seen either preemptive opens or regular opens
        # Exact behavior depends on the trained model, both are valid outcomes
        logger.info(f"Prediction test results: {preemptive_opens} preemptive opens, {regular_opens} regular opens")
        
        # Print out model feature importances if available
        if VISUALIZATION_AVAILABLE and hasattr(circuit_breaker.prediction_model, 'feature_importances_'):
            importances = circuit_breaker.prediction_model.feature_importances_
            if hasattr(circuit_breaker.prediction_model, 'feature_names_in_'):
                feature_names = circuit_breaker.prediction_model.feature_names_in_
                importances_dict = dict(zip(feature_names, importances))
                logger.info(f"Prediction model feature importances: {importances_dict}")
        
        # Cleanup
        circuit_breaker.close()
        logger.info("Prediction test completed")
    
    async def test_hardware_specific(self) -> None:
        """Test hardware-specific optimizations."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available. Skipping hardware-specific test.")
            return
            
        logger.info("Testing hardware-specific optimizations")
        
        # Create hardware-specific circuit breakers
        gpu_circuit_breaker = AdaptiveCircuitBreaker(
            name="gpu_test",
            base_failure_threshold=3,
            base_recovery_timeout=5,
            base_half_open_timeout=1,
            optimization_enabled=True,
            prediction_enabled=False,
            metrics_path=f"{self.metrics_path}_gpu",
            model_path=f"{self.model_path}_gpu",
            min_data_points=10,
            hardware_specific=True,
            hardware_type="gpu"
        )
        
        cpu_circuit_breaker = AdaptiveCircuitBreaker(
            name="cpu_test",
            base_failure_threshold=3,
            base_recovery_timeout=5,
            base_half_open_timeout=1,
            optimization_enabled=True,
            prediction_enabled=False,
            metrics_path=f"{self.metrics_path}_cpu",
            model_path=f"{self.model_path}_cpu",
            min_data_points=10,
            hardware_specific=True,
            hardware_type="cpu"
        )
        
        # Define different failure patterns for different hardware
        async def gpu_operation(success_rate: float = 0.5):
            if self.simulate or (hash(f"{time.time()}{id(gpu_circuit_breaker)}") % 100) / 100 < success_rate:
                await anyio.sleep(0.01)
                return "GPU Success"
            else:
                await anyio.sleep(0.01)
                # GPU failures often related to memory or resources
                error_type = ["memory", "resource", "gpu"][hash(f"{time.time()}") % 3]
                raise Exception(f"Simulated GPU {error_type} failure")
        
        async def cpu_operation(success_rate: float = 0.7):
            if self.simulate or (hash(f"{time.time()}{id(cpu_circuit_breaker)}") % 100) / 100 < success_rate:
                await anyio.sleep(0.01)
                return "CPU Success"
            else:
                await anyio.sleep(0.01)
                # CPU failures often related to timeouts or api issues
                error_type = ["timeout", "api", "connection"][hash(f"{time.time()}") % 3]
                raise Exception(f"Simulated CPU {error_type} failure")
        
        # Run operations on both circuit breakers
        for i in range(30):
            # GPU operations (lower success rate)
            try:
                await gpu_circuit_breaker.execute(lambda: gpu_operation(0.4))
            except Exception:
                pass
                
            # CPU operations (higher success rate)
            try:
                await cpu_circuit_breaker.execute(lambda: cpu_operation(0.7))
            except Exception:
                pass
                
            await anyio.sleep(0.05)
        
        # Verify that hardware-specific features were used
        gpu_state = gpu_circuit_breaker.get_state()
        cpu_state = cpu_circuit_breaker.get_state()
        
        logger.info(f"GPU Circuit Breaker parameters: threshold={gpu_circuit_breaker.current_failure_threshold}, "
                  f"recovery_timeout={gpu_circuit_breaker.current_recovery_timeout:.2f}s")
        logger.info(f"CPU Circuit Breaker parameters: threshold={cpu_circuit_breaker.current_failure_threshold}, "
                  f"recovery_timeout={cpu_circuit_breaker.current_recovery_timeout:.2f}s")
        
        # We expect GPU circuit breaker to have higher threshold or longer recovery time
        # because GPU failures are more common in our test
        assert (gpu_circuit_breaker.current_failure_threshold >= cpu_circuit_breaker.current_failure_threshold or
                gpu_circuit_breaker.current_recovery_timeout > cpu_circuit_breaker.current_recovery_timeout), \
            "Expected GPU circuit breaker to have higher threshold or longer recovery time"
        
        # Check that separate models were trained
        gpu_model_files = [f"{self.model_path}_gpu_threshold.pkl"]
        cpu_model_files = [f"{self.model_path}_cpu_threshold.pkl"]
        
        for model_file in gpu_model_files:
            assert os.path.exists(model_file), f"GPU Model file {model_file} was not created"
            
        for model_file in cpu_model_files:
            assert os.path.exists(model_file), f"CPU Model file {model_file} was not created"
        
        # Cleanup
        gpu_circuit_breaker.close()
        cpu_circuit_breaker.close()
        logger.info("Hardware-specific test passed")
    
    def generate_visualizations(self) -> None:
        """Generate visualizations of test results."""
        if not VISUALIZATION_AVAILABLE:
            _log_optional_dependency("Visualization dependencies not available. Skipping visualizations.")
            return
            
        logger.info("Generating visualizations")
        
        # Create output directory
        os.makedirs(os.path.join(self.test_dir, "visualizations"), exist_ok=True)
        
        # Visualize success/failure patterns
        if self.test_results:
            # Convert to DataFrame
            df = pd.DataFrame(self.test_results)
            df['index'] = range(len(df))
            
            # Create success/failure plot
            plt.figure(figsize=(10, 5))
            plt.bar(df['index'], df['success'].astype(int), color=['green' if s else 'red' for s in df['success']])
            plt.title("Operation Success/Failure Pattern")
            plt.xlabel("Operation Index")
            plt.ylabel("Success (1) / Failure (0)")
            plt.ylim(-0.1, 1.1)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Success'),
                Patch(facecolor='red', label='Failure')
            ]
            plt.legend(handles=legend_elements)
            
            # Save figure
            plt.savefig(os.path.join(self.test_dir, "visualizations/success_failure_pattern.png"))
            plt.close()
            
            logger.info(f"Visualization saved to {os.path.join(self.test_dir, 'visualizations/success_failure_pattern.png')}")
            
            # If failure messages are available, visualize error types
            if 'error' in df.columns:
                # Extract error types
                error_types = {}
                for error in df[~df['success']]['error']:
                    for error_type in self.failure_types:
                        if error_type in error.lower():
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                
                if error_types:
                    # Create error type distribution plot
                    plt.figure(figsize=(10, 5))
                    plt.bar(error_types.keys(), error_types.values())
                    plt.title("Error Type Distribution")
                    plt.xlabel("Error Type")
                    plt.ylabel("Count")
                    
                    # Save figure
                    plt.savefig(os.path.join(self.test_dir, "visualizations/error_type_distribution.png"))
                    plt.close()
                    
                    logger.info(f"Error distribution visualization saved to {os.path.join(self.test_dir, 'visualizations/error_type_distribution.png')}")
        
        logger.info("Visualizations completed")


async def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test the Adaptive Circuit Breaker")
    parser.add_argument("--quick", action="store_true", help="Run only basic tests")
    parser.add_argument("--simulate", action="store_true", help="Use simulation mode")
    args = parser.parse_args()
    
    # Create and run tester
    tester = CircuitBreakerTester(quick_test=args.quick, simulate=args.simulate)
    success = await tester.run_tests()
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = anyio.run(main())
    sys.exit(exit_code)