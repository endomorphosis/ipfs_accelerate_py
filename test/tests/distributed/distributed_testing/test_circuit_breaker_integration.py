#!/usr/bin/env python3
"""
Test Circuit Breaker Integration with Browser Failure Injector

This script tests the integration between the circuit breaker pattern and
the browser failure injector to ensure proper fault tolerance and failure management.

It validates that:
1. Circuit breaker transitions between states correctly based on failures
2. Failure injector adapts behavior based on circuit breaker state
3. System behavior is appropriate for each circuit state (closed, open, half-open)
4. Circuit breaker metrics are correctly reported and tracked

Usage:
    python test_circuit_breaker_integration.py [--browser chrome] [--headless]
"""

import os
import sys
import time
import json
import anyio
import logging
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("circuit_breaker_test")

# Set more verbose logging if environment variable is set
if os.environ.get("CIRCUIT_BREAKER_LOG_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

# Import required components
try:
    from test.tests.distributed.distributed_testing.selenium_browser_bridge import (
        BrowserConfiguration, SeleniumBrowserBridge, SELENIUM_AVAILABLE
    )
except ImportError:
    try:
        from selenium_browser_bridge import (
            BrowserConfiguration, SeleniumBrowserBridge, SELENIUM_AVAILABLE
        )
    except ImportError:
        logger.error("Error importing selenium_browser_bridge. Make sure it exists at the expected path.")
        SELENIUM_AVAILABLE = False

try:
    from test.tests.distributed.distributed_testing.browser_failure_injector import (
        BrowserFailureInjector, FailureType
    )
    INJECTOR_AVAILABLE = True
except ImportError:
    try:
        from browser_failure_injector import (
            BrowserFailureInjector, FailureType
        )
        INJECTOR_AVAILABLE = True
    except ImportError:
        logger.error("Error importing browser_failure_injector. Make sure it exists at the expected path.")
        INJECTOR_AVAILABLE = False
    
    # Define fallback FailureType for type checking
    from enum import Enum
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

try:
    from test.tests.distributed.distributed_testing.circuit_breaker import CircuitBreaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    try:
        from circuit_breaker import CircuitBreaker
        CIRCUIT_BREAKER_AVAILABLE = True
    except ImportError:
        logger.error("Error importing circuit_breaker. Make sure it exists at the expected path.")
        CIRCUIT_BREAKER_AVAILABLE = False

class CircuitBreakerIntegrationTest:
    """
    Test class for the integration between Circuit Breaker and Browser Failure Injector.
    
    This class provides a comprehensive test suite for verifying that the circuit breaker
    works correctly with the browser failure injector to provide fault tolerance.
    """
    
    def __init__(self, browser_name: str = "chrome", platform: str = "webgpu",
                 headless: bool = True, save_results: Optional[str] = None):
        """
        Initialize the circuit breaker integration test.
        
        Args:
            browser_name: Browser name to test (chrome, firefox, edge)
            platform: Platform to test (webgpu, webnn)
            headless: Whether to run in headless mode
            save_results: Path to save test results (or None)
        """
        self.browser_name = browser_name
        self.platform = platform
        self.headless = headless
        self.save_results = save_results
        
        # Test results
        self.results = {}
        
        # All supported failure types
        self.all_failure_types = [
            FailureType.CONNECTION_FAILURE,
            FailureType.RESOURCE_EXHAUSTION,
            FailureType.GPU_ERROR,
            FailureType.API_ERROR,
            FailureType.TIMEOUT,
            FailureType.INTERNAL_ERROR,
            FailureType.CRASH
        ]
        
        # All supported intensities
        self.all_intensities = ["mild", "moderate", "severe"]
        
        logger.info(f"Initialized circuit breaker integration test with browser={browser_name}, platform={platform}")
        
        # Create circuit breaker with test-friendly thresholds
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,     # Open after 3 failures
            recovery_timeout=10,     # Stay open for 10 seconds
            half_open_after=5,       # Try half-open after 5 seconds
            name="test_circuit_breaker"
        )
        
        logger.info(f"Created circuit breaker with threshold={self.circuit_breaker.failure_threshold}")
    
    async def test_circuit_closed_state(self) -> Dict[str, Any]:
        """
        Test that the circuit breaker starts in closed state and allows all operations.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing circuit breaker in CLOSED state")
        
        # Results for this test
        test_results = {
            "test_name": "circuit_closed_state",
            "success": True,
            "failures": []
        }
        
        # Verify initial state
        state = self.circuit_breaker.get_state()
        if not state == "closed":
            test_results["success"] = False
            test_results["failures"].append(f"Expected initial state to be 'closed', got '{state}'")
            return test_results
        
        # Create a new browser and injector for each test
        bridge, injector = await self._create_browser_and_injector()
        
        if not bridge or not injector:
            test_results["success"] = False
            test_results["failures"].append("Failed to create browser or injector")
            return test_results
        
        try:
            # Test that all intensity levels are allowed in closed state
            for intensity in self.all_intensities:
                # Use connection failure as it's usually reliable
                failure_type = FailureType.CONNECTION_FAILURE
                
                # Inject the failure
                logger.info(f"Injecting {failure_type.value} with {intensity} intensity in CLOSED state")
                result = await injector.inject_failure(failure_type, intensity)
                
                # Verify the failure was allowed (not blocked by circuit breaker)
                if not result.get("success", False):
                    test_results["success"] = False
                    test_results["failures"].append(f"Failure injection failed with {intensity} intensity in CLOSED state")
                
                # Verify failure was not blocked by circuit breaker
                if result.get("circuit_breaker_open", False):
                    test_results["success"] = False
                    test_results["failures"].append(f"Circuit breaker incorrectly blocked {intensity} intensity in CLOSED state")
                
                # Small delay between tests
                await anyio.sleep(0.5)
            
            # Verify circuit breaker is still closed after mild/moderate failures
            state = self.circuit_breaker.get_state()
            if not state == "closed":
                test_results["success"] = False
                test_results["failures"].append(f"Expected state to remain 'closed' after mild/moderate failures, got '{state}'")
        
        except Exception as e:
            test_results["success"] = False
            test_results["failures"].append(f"Exception during test: {str(e)}")
        
        finally:
            # Close browser
            if bridge:
                await bridge.close()
        
        return test_results
    
    async def test_circuit_open_transition(self) -> Dict[str, Any]:
        """
        Test that the circuit breaker transitions to open state after threshold failures.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing circuit breaker transition to OPEN state")
        
        # Results for this test
        test_results = {
            "test_name": "circuit_open_transition",
            "success": True,
            "failures": []
        }
        
        # Verify initial state or reset
        if self.circuit_breaker.get_state() != "closed":
            self.circuit_breaker.reset()
        
        # Create a new browser and injector for each test
        bridge, injector = await self._create_browser_and_injector()
        
        if not bridge or not injector:
            test_results["success"] = False
            test_results["failures"].append("Failed to create browser or injector")
            return test_results
        
        try:
            # Trigger circuit breaker by injecting severe failures
            failures_needed = self.circuit_breaker.failure_threshold
            logger.info(f"Injecting {failures_needed} severe failures to open circuit")
            
            for i in range(failures_needed):
                # Use crash failures which are most likely to trigger circuit breaker
                failure_type = FailureType.CRASH
                intensity = "severe"
                
                # Inject the failure
                logger.info(f"Injecting severe failure {i+1}/{failures_needed}")
                result = await injector.inject_failure(failure_type, intensity)
                
                # Verify the failure was injected
                if not result.get("success", False):
                    test_results["success"] = False
                    test_results["failures"].append(f"Failed to inject severe failure {i+1}")
                
                # Check if circuit breaker was updated
                if not result.get("circuit_breaker_updated", False):
                    test_results["success"] = False
                    test_results["failures"].append(f"Circuit breaker was not updated after severe failure {i+1}")
                
                # Create a new browser and injector if needed for next iteration
                if i < failures_needed - 1:
                    await bridge.close()
                    bridge, injector = await self._create_browser_and_injector()
                    
                    if not bridge or not injector:
                        test_results["success"] = False
                        test_results["failures"].append(f"Failed to create new browser after failure {i+1}")
                        return test_results
            
            # Verify circuit breaker is now open
            state = self.circuit_breaker.get_state()
            if not state == "open":
                test_results["success"] = False
                test_results["failures"].append(f"Expected state to be 'open' after {failures_needed} severe failures, got '{state}'")
                return test_results
            
            logger.info("Circuit breaker is now OPEN")
            
            # Test that all failures are blocked when circuit is open
            # Try to inject one more failure - should be blocked
            failure_type = FailureType.CONNECTION_FAILURE
            intensity = "mild"
            
            logger.info(f"Attempting to inject failure with circuit OPEN - should be blocked")
            result = await injector.inject_failure(failure_type, intensity)
            
            # Verify the failure was blocked by circuit breaker
            if not result.get("circuit_breaker_open", False) or result.get("success", False):
                test_results["success"] = False
                test_results["failures"].append("Failure was not blocked when circuit was open")
            else:
                logger.info("Failure was correctly blocked by open circuit breaker")
        
        except Exception as e:
            test_results["success"] = False
            test_results["failures"].append(f"Exception during test: {str(e)}")
        
        finally:
            # Close browser
            if bridge:
                await bridge.close()
        
        return test_results
    
    async def test_circuit_half_open_transition(self) -> Dict[str, Any]:
        """
        Test that the circuit breaker transitions to half-open state after recovery timeout.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing circuit breaker transition to HALF-OPEN state")
        
        # Results for this test
        test_results = {
            "test_name": "circuit_half_open_transition",
            "success": True,
            "failures": []
        }
        
        # Ensure circuit is open
        current_state = self.circuit_breaker.get_state()
        if current_state != "open":
            test_results["success"] = False
            test_results["failures"].append(f"Expected circuit to be 'open' at start of test, got '{current_state}'")
            return test_results
        
        try:
            # Wait for half-open transition
            logger.info(f"Waiting {self.circuit_breaker.half_open_after} seconds for half-open transition")
            await anyio.sleep(self.circuit_breaker.half_open_after)
            
            # Check if circuit is now half-open
            state = self.circuit_breaker.get_state()
            if not state == "half-open":
                test_results["success"] = False
                test_results["failures"].append(f"Expected state to be 'half-open' after timeout, got '{state}'")
                return test_results
            
            logger.info("Circuit breaker is now HALF-OPEN")
            
            # Create a new browser and injector for the half-open test
            bridge, injector = await self._create_browser_and_injector()
            
            if not bridge or not injector:
                test_results["success"] = False
                test_results["failures"].append("Failed to create browser or injector for half-open test")
                return test_results
            
            # Test that severe failures are disallowed in half-open state
            # while mild/moderate are allowed
            for intensity in self.all_intensities:
                failure_type = FailureType.CONNECTION_FAILURE
                
                logger.info(f"Testing {intensity} intensity in HALF-OPEN state")
                result = await injector.inject_random_failure(exclude_severe=(intensity == "severe"))
                
                # For severe intensity, random_failure should exclude it
                # For mild/moderate, it should include based on the exclude_severe parameter
                expected_allowed = (intensity != "severe")
                actual_allowed = not result.get("circuit_breaker_open", False) and result.get("success", False)
                
                if expected_allowed != actual_allowed:
                    test_results["success"] = False
                    test_results["failures"].append(f"{intensity} intensity was {'allowed' if actual_allowed else 'blocked'} in half-open state, expected {'allowed' if expected_allowed else 'blocked'}")
                
                # In half-open state, a severe failure should reopen the circuit
                if intensity == "severe" and actual_allowed:
                    # Check if circuit went back to open
                    state = self.circuit_breaker.get_state()
                    if not state == "open":
                        test_results["success"] = False
                        test_results["failures"].append(f"Circuit did not transition back to 'open' after severe failure in half-open state, got '{state}'")
                
                # Small delay between tests
                await anyio.sleep(0.5)
        
        except Exception as e:
            test_results["success"] = False
            test_results["failures"].append(f"Exception during test: {str(e)}")
        
        finally:
            # Close browser
            if bridge:
                await bridge.close()
        
        return test_results
    
    async def test_circuit_reclosing(self) -> Dict[str, Any]:
        """
        Test that the circuit breaker transitions back to closed state after successful operations.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing circuit breaker transition back to CLOSED state")
        
        # Results for this test
        test_results = {
            "test_name": "circuit_reclosing",
            "success": True,
            "failures": []
        }
        
        # Reset the circuit to ensure we start fresh
        self.circuit_breaker.reset()
        
        # Verify the circuit is closed
        state = self.circuit_breaker.get_state()
        if not state == "closed":
            test_results["success"] = False
            test_results["failures"].append(f"Failed to reset circuit breaker to closed state, got '{state}'")
            return test_results
        
        # First part: Transition to open and then to half-open
        try:
            # Open the circuit
            logger.info("Opening the circuit")
            for i in range(self.circuit_breaker.failure_threshold):
                self.circuit_breaker.record_failure()
            
            # Verify circuit is open
            state = self.circuit_breaker.get_state()
            if not state == "open":
                test_results["success"] = False
                test_results["failures"].append(f"Expected state to be 'open' after recording failures, got '{state}'")
                return test_results
            
            # Wait for half-open transition
            logger.info(f"Waiting {self.circuit_breaker.half_open_after} seconds for half-open transition")
            await anyio.sleep(self.circuit_breaker.half_open_after + 0.5)
            
            # Verify circuit is half-open
            state = self.circuit_breaker.get_state()
            if not state == "half-open":
                test_results["success"] = False
                test_results["failures"].append(f"Expected state to be 'half-open' after timeout, got '{state}'")
                return test_results
            
            # Record success to close the circuit
            logger.info("Recording success in half-open state")
            self.circuit_breaker.record_success()
            
            # Verify circuit is now closed
            state = self.circuit_breaker.get_state()
            if not state == "closed":
                test_results["success"] = False
                test_results["failures"].append(f"Expected state to be 'closed' after success in half-open state, got '{state}'")
                return test_results
            
            logger.info("Circuit breaker is now CLOSED again")
        
        except Exception as e:
            test_results["success"] = False
            test_results["failures"].append(f"Exception during state transition test: {str(e)}")
        
        # Second part: Test with actual browser and injector
        try:
            # Reset circuit
            self.circuit_breaker.reset()
            
            # Open the circuit by recording failures
            for i in range(self.circuit_breaker.failure_threshold):
                self.circuit_breaker.record_failure()
            
            # Verify circuit is open
            state = self.circuit_breaker.get_state()
            if not state == "open":
                test_results["success"] = False
                test_results["failures"].append(f"Expected state to be 'open' for browser test, got '{state}'")
                return test_results
            
            # Wait for half-open transition
            logger.info(f"Waiting {self.circuit_breaker.half_open_after} seconds for half-open transition for browser test")
            await anyio.sleep(self.circuit_breaker.half_open_after + 0.5)
            
            # Create a new browser and injector
            bridge, injector = await self._create_browser_and_injector()
            
            if not bridge or not injector:
                test_results["success"] = False
                test_results["failures"].append("Failed to create browser or injector for reclosing test")
                return test_results
            
            # Inject a mild failure - should be allowed in half-open state
            # and not trigger circuit reopening
            failure_type = FailureType.CONNECTION_FAILURE
            intensity = "mild"
            
            logger.info(f"Injecting mild failure in half-open state")
            result = await injector.inject_failure(failure_type, intensity)
            
            # Verify the failure was allowed
            if not result.get("success", False) or result.get("circuit_breaker_open", False):
                test_results["success"] = False
                test_results["failures"].append("Mild failure was unexpectedly blocked in half-open state")
            
            # Verify circuit is still half-open
            state = self.circuit_breaker.get_state()
            if not state == "half-open":
                test_results["success"] = False
                test_results["failures"].append(f"Expected state to remain 'half-open' after mild failure, got '{state}'")
            
            # Manually trigger circuit closing
            self.circuit_breaker.record_success()
            
            # Verify circuit is now closed
            state = self.circuit_breaker.get_state()
            if not state == "closed":
                test_results["success"] = False
                test_results["failures"].append(f"Expected state to be 'closed' after recording success, got '{state}'")
            
            logger.info("Circuit breaker is now CLOSED after successful browser test")
            
            # Test that all intensities are allowed again in closed state
            for intensity in self.all_intensities:
                if intensity == "severe":
                    # Skip severe to avoid reopening circuit
                    continue
                    
                logger.info(f"Testing {intensity} intensity is allowed in closed state")
                result = await injector.inject_failure(failure_type, intensity)
                
                if not result.get("success", False) or result.get("circuit_breaker_open", False):
                    test_results["success"] = False
                    test_results["failures"].append(f"{intensity} intensity was unexpectedly blocked in closed state")
        
        except Exception as e:
            test_results["success"] = False
            test_results["failures"].append(f"Exception during browser test for reclosing: {str(e)}")
        
        finally:
            # Close browser
            if bridge:
                await bridge.close()
        
        return test_results
    
    async def test_adaptive_failure_injection(self) -> Dict[str, Any]:
        """
        Test that the failure injector adapts its behavior based on circuit breaker state.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing adaptive failure injection based on circuit state")
        
        # Results for this test
        test_results = {
            "test_name": "adaptive_failure_injection",
            "success": True,
            "failures": [],
            "circuit_closed": {},
            "circuit_half_open": {},
            "circuit_open": {}
        }
        
        # Test in closed state
        logger.info("Testing injection in CLOSED state")
        self.circuit_breaker.reset()
        
        # Create browser and injector
        bridge, injector = await self._create_browser_and_injector()
        
        if not bridge or not injector:
            test_results["success"] = False
            test_results["failures"].append("Failed to create browser or injector for closed state test")
            return test_results
        
        try:
            # Test random failure in closed state
            logger.info("Injecting random failure in CLOSED state")
            result = await injector.inject_random_failure()
            
            # Track what intensity was selected
            intensity = result.get("intensity", "unknown")
            test_results["circuit_closed"] = {
                "allowed_intensities": ["mild", "moderate", "severe"],
                "selected_intensity": intensity,
                "was_allowed": result.get("success", False),
                "was_blocked": result.get("circuit_breaker_open", False)
            }
            
            # Close browser
            await bridge.close()
            
            # Open the circuit
            for i in range(self.circuit_breaker.failure_threshold):
                self.circuit_breaker.record_failure()
            
            # Verify circuit is open
            state = self.circuit_breaker.get_state()
            if not state == "open":
                test_results["success"] = False
                test_results["failures"].append(f"Expected state to be 'open', got '{state}'")
            
            # Create new browser and injector
            bridge, injector = await self._create_browser_and_injector()
            
            if not bridge or not injector:
                test_results["success"] = False
                test_results["failures"].append("Failed to create browser or injector for open state test")
                return test_results
            
            # Test random failure in open state
            logger.info("Injecting random failure in OPEN state")
            result = await injector.inject_random_failure()
            
            # Should be blocked
            test_results["circuit_open"] = {
                "was_allowed": result.get("success", False),
                "was_blocked": result.get("circuit_breaker_open", False)
            }
            
            if not result.get("circuit_breaker_open", False) or result.get("success", False):
                test_results["success"] = False
                test_results["failures"].append("Failure injection was not blocked in OPEN state")
            
            # Close browser
            await bridge.close()
            
            # Wait for half-open transition
            logger.info(f"Waiting {self.circuit_breaker.half_open_after} seconds for half-open transition")
            await anyio.sleep(self.circuit_breaker.half_open_after + 0.5)
            
            # Create new browser and injector
            bridge, injector = await self._create_browser_and_injector()
            
            if not bridge or not injector:
                test_results["success"] = False
                test_results["failures"].append("Failed to create browser or injector for half-open state test")
                return test_results
            
            # Test random failure in half-open state
            logger.info("Injecting random failure in HALF-OPEN state")
            result = await injector.inject_random_failure()
            
            # Track what intensity was selected
            intensity = result.get("intensity", "unknown")
            test_results["circuit_half_open"] = {
                "allowed_intensities": ["mild", "moderate"],
                "selected_intensity": intensity,
                "was_allowed": result.get("success", False),
                "was_blocked": result.get("circuit_breaker_open", False)
            }
            
            # In half-open state, severe intensity should never be selected
            if intensity == "severe":
                test_results["success"] = False
                test_results["failures"].append("inject_random_failure selected severe intensity in HALF-OPEN state")
        
        except Exception as e:
            test_results["success"] = False
            test_results["failures"].append(f"Exception during adaptive failure injection test: {str(e)}")
        
        finally:
            # Close browser
            if bridge:
                await bridge.close()
        
        return test_results
    
    async def test_failure_metrics(self) -> Dict[str, Any]:
        """
        Test that the circuit breaker metrics are correctly reported.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing circuit breaker metrics reporting")
        
        # Results for this test
        test_results = {
            "test_name": "failure_metrics",
            "success": True,
            "failures": []
        }
        
        # Reset circuit breaker
        self.circuit_breaker.reset()
        
        # Create browser and injector
        bridge, injector = await self._create_browser_and_injector()
        
        if not bridge or not injector:
            test_results["success"] = False
            test_results["failures"].append("Failed to create browser or injector for metrics test")
            return test_results
        
        try:
            # Test that failure metrics are reported by injector
            logger.info("Checking initial circuit breaker metrics")
            
            # Get injector statistics
            stats = injector.get_failure_stats()
            
            # Verify circuit breaker metrics are included
            if "circuit_breaker" not in stats:
                test_results["success"] = False
                test_results["failures"].append("Circuit breaker metrics not found in failure stats")
                return test_results
            
            # Check circuit breaker initial metrics
            cb_metrics = stats["circuit_breaker"]
            if cb_metrics["state"] != "closed":
                test_results["success"] = False
                test_results["failures"].append(f"Unexpected initial state in metrics: {cb_metrics['state']}")
            
            if cb_metrics["failure_count"] != 0:
                test_results["success"] = False
                test_results["failures"].append(f"Unexpected initial failure count: {cb_metrics['failure_count']}")
            
            # Inject a few failures to update metrics
            logger.info("Injecting failures to update metrics")
            
            # Inject failures but not enough to open circuit
            failure_type = FailureType.CRASH
            intensity = "severe"
            
            # Inject one less than the threshold to keep circuit closed
            for i in range(self.circuit_breaker.failure_threshold - 1):
                result = await injector.inject_failure(failure_type, intensity)
                
                if not result.get("success", False):
                    test_results["success"] = False
                    test_results["failures"].append(f"Failed to inject failure {i+1} for metrics test")
                
                # Create a new browser if needed
                if i < self.circuit_breaker.failure_threshold - 2:
                    await bridge.close()
                    bridge, injector = await self._create_browser_and_injector()
                    
                    if not bridge or not injector:
                        test_results["success"] = False
                        test_results["failures"].append(f"Failed to create new browser after failure {i+1}")
                        return test_results
            
            # Get updated metrics
            logger.info("Checking updated circuit breaker metrics")
            stats = injector.get_failure_stats()
            
            # Verify circuit breaker metrics are updated
            if "circuit_breaker" not in stats:
                test_results["success"] = False
                test_results["failures"].append("Circuit breaker metrics not found in updated stats")
                return test_results
            
            # Check circuit breaker updated metrics
            cb_metrics = stats["circuit_breaker"]
            
            # State should still be closed
            if cb_metrics["state"] != "closed":
                test_results["success"] = False
                test_results["failures"].append(f"Unexpected state in updated metrics: {cb_metrics['state']}")
            
            # Failure count should be threshold - 1
            expected_count = self.circuit_breaker.failure_threshold - 1
            if cb_metrics["failure_count"] != expected_count:
                test_results["success"] = False
                test_results["failures"].append(f"Unexpected failure count: {cb_metrics['failure_count']}, expected {expected_count}")
            
            # Threshold percent should be calculated correctly
            expected_percent = (expected_count / self.circuit_breaker.failure_threshold) * 100
            if abs(cb_metrics["threshold_percent"] - expected_percent) > 0.1:
                test_results["success"] = False
                test_results["failures"].append(f"Unexpected threshold percent: {cb_metrics['threshold_percent']}, expected {expected_percent}")
            
            # Test one more failure to open circuit
            logger.info("Injecting one more failure to open circuit")
            result = await injector.inject_failure(failure_type, intensity)
            
            # Get final metrics
            logger.info("Checking final circuit breaker metrics")
            stats = injector.get_failure_stats()
            cb_metrics = stats["circuit_breaker"]
            
            # State should now be open
            if cb_metrics["state"] != "open":
                test_results["success"] = False
                test_results["failures"].append(f"Unexpected final state in metrics: {cb_metrics['state']}, expected 'open'")
            
            # Failure count should be threshold
            if cb_metrics["failure_count"] != self.circuit_breaker.failure_threshold:
                test_results["success"] = False
                test_results["failures"].append(f"Unexpected final failure count: {cb_metrics['failure_count']}, expected {self.circuit_breaker.failure_threshold}")
            
            # Threshold percent should be 100%
            if abs(cb_metrics["threshold_percent"] - 100.0) > 0.1:
                test_results["success"] = False
                test_results["failures"].append(f"Unexpected final threshold percent: {cb_metrics['threshold_percent']}, expected 100.0")
        
        except Exception as e:
            test_results["success"] = False
            test_results["failures"].append(f"Exception during metrics test: {str(e)}")
        
        finally:
            # Close browser
            if bridge:
                await bridge.close()
        
        return test_results
    
    async def _create_browser_and_injector(self):
        """
        Helper method to create a browser and injector for testing.
        
        Returns:
            Tuple of (bridge, injector)
        """
        # Create browser configuration
        config = BrowserConfiguration(
            browser_name=self.browser_name,
            platform=self.platform,
            headless=self.headless,
            timeout=30
        )
        
        # Create browser bridge
        bridge = SeleniumBrowserBridge(config)
        
        try:
            # Launch browser
            launch_success = await bridge.launch(allow_simulation=True)
            
            if not launch_success:
                logger.error(f"Failed to launch {self.browser_name}")
                return None, None
            
            # Create failure injector with circuit breaker
            injector = BrowserFailureInjector(
                bridge, 
                circuit_breaker=self.circuit_breaker,
                use_circuit_breaker=True
            )
            
            return bridge, injector
            
        except Exception as e:
            logger.error(f"Error creating browser and injector: {str(e)}")
            if bridge:
                await bridge.close()
            return None, None
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all circuit breaker integration tests.
        
        Returns:
            Dictionary with all test results
        """
        logger.info("Running all circuit breaker integration tests")
        
        # Tests to run in sequence (order matters because of circuit breaker state)
        tests = [
            ("circuit_closed_state", self.test_circuit_closed_state),
            ("circuit_open_transition", self.test_circuit_open_transition),
            ("circuit_half_open_transition", self.test_circuit_half_open_transition),
            ("circuit_reclosing", self.test_circuit_reclosing),
            ("adaptive_failure_injection", self.test_adaptive_failure_injection),
            ("failure_metrics", self.test_failure_metrics)
        ]
        
        # Overall results
        all_results = {
            "browser": self.browser_name,
            "platform": self.platform,
            "headless": self.headless,
            "circuit_breaker": {
                "failure_threshold": self.circuit_breaker.failure_threshold,
                "recovery_timeout": self.circuit_breaker.recovery_timeout,
                "half_open_after": self.circuit_breaker.half_open_after
            },
            "start_time": time.time(),
            "tests": {},
            "end_time": None,
            "passed_tests": 0,
            "failed_tests": 0,
            "total_tests": len(tests)
        }
        
        # Run each test in sequence
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            
            # Run the test
            result = await test_func()
            
            # Store result
            all_results["tests"][test_name] = result
            
            # Count passed/failed tests
            if result.get("success", False):
                all_results["passed_tests"] += 1
            else:
                all_results["failed_tests"] += 1
            
            # Print test result
            success = result.get("success", False)
            status = "✅ PASSED" if success else "❌ FAILED"
            
            print(f"\nTest {test_name}: {status}")
            if not success and "failures" in result:
                for failure in result["failures"]:
                    print(f"  - {failure}")
        
        # Record final stats
        all_results["end_time"] = time.time()
        all_results["duration_seconds"] = all_results["end_time"] - all_results["start_time"]
        all_results["success_rate"] = all_results["passed_tests"] / all_results["total_tests"]
        
        # Store all results
        self.results = all_results
        
        # Save results if requested
        if self.save_results:
            self._save_results()
        
        return all_results
    
    def _save_results(self) -> None:
        """Save test results to a file."""
        if not self.save_results:
            return
            
        try:
            with open(self.save_results, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            print(f"\nResults saved to {self.save_results}")
            
            # Also generate a markdown summary
            markdown_path = self.save_results.replace('.json', '.md')
            if markdown_path == self.save_results:
                markdown_path += '.md'
                
            # Create markdown summary
            with open(markdown_path, 'w') as f:
                f.write(f"# Circuit Breaker Integration Test Results\n\n")
                
                f.write(f"## Configuration\n\n")
                f.write(f"- **Browser:** {self.results['browser']}\n")
                f.write(f"- **Platform:** {self.results['platform']}\n")
                f.write(f"- **Headless:** {self.results['headless']}\n")
                f.write(f"- **Duration:** {self.results['duration_seconds']:.2f} seconds\n\n")
                
                f.write(f"### Circuit Breaker Configuration\n\n")
                f.write(f"- **Failure Threshold:** {self.results['circuit_breaker']['failure_threshold']}\n")
                f.write(f"- **Recovery Timeout:** {self.results['circuit_breaker']['recovery_timeout']} seconds\n")
                f.write(f"- **Half-Open After:** {self.results['circuit_breaker']['half_open_after']} seconds\n\n")
                
                f.write(f"## Summary\n\n")
                f.write(f"- **Total Tests:** {self.results['total_tests']}\n")
                f.write(f"- **Passed Tests:** {self.results['passed_tests']}\n")
                f.write(f"- **Failed Tests:** {self.results['failed_tests']}\n")
                f.write(f"- **Success Rate:** {self.results['success_rate']:.2%}\n\n")
                
                f.write(f"## Test Results\n\n")
                f.write(f"| Test | Result |\n")
                f.write(f"|------|--------|\n")
                
                for test_name, result in self.results["tests"].items():
                    status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
                    f.write(f"| {test_name} | {status} |\n")
                
                f.write(f"\n## Detailed Results\n\n")
                
                for test_name, result in self.results["tests"].items():
                    f.write(f"### {test_name}\n\n")
                    status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
                    f.write(f"**Result:** {status}\n\n")
                    
                    if not result.get("success", False) and "failures" in result:
                        f.write("**Failures:**\n\n")
                        for failure in result["failures"]:
                            f.write(f"- {failure}\n")
                        f.write("\n")
                
                f.write(f"## Conclusion\n\n")
                
                if self.results['success_rate'] == 1.0:
                    f.write("The circuit breaker integration is working perfectly with all tests passing.\n")
                elif self.results['success_rate'] > 0.8:
                    f.write("The circuit breaker integration is working well with minor issues.\n")
                elif self.results['success_rate'] > 0.5:
                    f.write("The circuit breaker integration has significant issues that should be addressed.\n")
                else:
                    f.write("The circuit breaker integration is not working correctly and requires immediate attention.\n")
                
            print(f"Markdown summary saved to {markdown_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def print_summary(self) -> None:
        """Print a summary of the test results."""
        if not self.results:
            print("\nNo test results available")
            return
        
        print("\n" + "=" * 80)
        print("Circuit Breaker Integration Test Summary")
        print("=" * 80)
        
        print(f"Browser:      {self.results['browser']}")
        print(f"Platform:     {self.results['platform']}")
        print(f"Duration:     {self.results['duration_seconds']:.2f} seconds")
        
        print("\nCircuit Breaker Configuration:")
        print(f"  Failure Threshold: {self.results['circuit_breaker']['failure_threshold']}")
        print(f"  Recovery Timeout:  {self.results['circuit_breaker']['recovery_timeout']} seconds")
        print(f"  Half-Open After:   {self.results['circuit_breaker']['half_open_after']} seconds")
        
        print("\nTest Results:")
        print(f"  Total Tests:  {self.results['total_tests']}")
        print(f"  Passed Tests: {self.results['passed_tests']}")
        print(f"  Failed Tests: {self.results['failed_tests']}")
        print(f"  Success Rate: {self.results['success_rate']:.2%}")
        
        print("\nIndividual Test Results:")
        print("-" * 60)
        print(f"{'Test Name':<30} {'Result':<10}")
        print("-" * 60)
        
        for test_name, result in self.results["tests"].items():
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            print(f"{test_name:<30} {status:<10}")
        
        print("=" * 80)
        
        # Provide recommendations based on results
        if self.results['success_rate'] == 1.0:
            print("\nThe circuit breaker integration is working perfectly with all tests passing.")
        elif self.results['success_rate'] > 0.8:
            print("\nThe circuit breaker integration is working well with minor issues.")
            
            # Identify problematic tests
            problematic = []
            for test_name, result in self.results["tests"].items():
                if not result.get("success", False):
                    problematic.append(test_name)
            
            if problematic:
                print(f"Areas to address: {', '.join(problematic)}")
        elif self.results['success_rate'] > 0.5:
            print("\nThe circuit breaker integration has significant issues that should be addressed.")
        else:
            print("\nThe circuit breaker integration is not working correctly and requires immediate attention.")


async def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Circuit Breaker Integration with Browser Failure Injector")
    parser.add_argument("--browser", default="chrome", choices=["chrome", "firefox", "edge"], 
                       help="Browser to test (chrome, firefox, edge)")
    parser.add_argument("--platform", default="webgpu", choices=["webgpu", "webnn"], 
                       help="Platform to test (webgpu, webnn)")
    parser.add_argument("--no-headless", action="store_true", 
                       help="Run browser in visible mode (not headless)")
    parser.add_argument("--save-results", type=str, 
                       help="Path to save test results (JSON)")
    args = parser.parse_args()
    
    # Create default save path if not provided
    save_path = args.save_results
    if not save_path:
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        save_path = os.path.join(reports_dir, f"circuit_breaker_integration_test_{args.browser}_{timestamp}.json")
    
    # Check dependencies
    if not SELENIUM_AVAILABLE:
        logger.error("Selenium not available. Cannot run tests.")
        return 1
        
    if not INJECTOR_AVAILABLE:
        logger.error("Browser Failure Injector not available. Cannot run tests.")
        return 1
        
    if not CIRCUIT_BREAKER_AVAILABLE:
        logger.error("Circuit Breaker not available. Cannot run tests.")
        return 1
    
    # Create and run tests
    print("-" * 80)
    print(f"Running Circuit Breaker Integration tests with:")
    print(f"  Browser:      {args.browser}")
    print(f"  Platform:     {args.platform}")
    print(f"  Headless:     {not args.no_headless}")
    print("-" * 80)
    
    circuit_test = CircuitBreakerIntegrationTest(
        browser_name=args.browser,
        platform=args.platform,
        headless=not args.no_headless,
        save_results=save_path
    )
    
    # Run tests
    await circuit_test.run_all_tests()
    
    # Print summary
    circuit_test.print_summary()
    
    # Determine exit code based on results
    if circuit_test.results.get("passed_tests", 0) == circuit_test.results.get("total_tests", 0):
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = anyio.run(main())
    sys.exit(exit_code)