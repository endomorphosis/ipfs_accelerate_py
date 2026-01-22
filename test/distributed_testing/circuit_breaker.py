#!/usr/bin/env python3
"""
Circuit Breaker for Distributed Testing Framework

This module implements the circuit breaker pattern to prevent cascading failures
in distributed systems. It provides fault tolerance by automatically detecting failures
and preventing operations when the system is in a failed state.
"""

import time
import asyncio
import logging
from enum import Enum
from typing import Dict, Any, Callable, Awaitable, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CircuitState(str, Enum):
    """States for the circuit breaker"""
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"      # Failure detected, requests are blocked
    HALF_OPEN = "half_open"  # Testing if system has recovered

class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascading failures
    """
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30,
                 half_open_timeout: float = 5, name: str = "default"):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            half_open_timeout: Timeout for half-open state tests
            name: Name for this circuit breaker
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_timeout = half_open_timeout
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.last_success_time = time.time()
        
        # Create logger
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
        self.logger.info(f"Circuit breaker {name} initialized with threshold {failure_threshold}")
    
    async def execute(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Async function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        # Check circuit state
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if current_time - self.last_failure_time > self.recovery_timeout:
                self.logger.info(f"Recovery timeout elapsed, transitioning to half-open state")
                self.state = CircuitState.HALF_OPEN
            else:
                # Circuit is open, fail fast
                wait_time = self.recovery_timeout - (current_time - self.last_failure_time)
                self.logger.warning(f"Circuit is open, failing fast. Retry after {wait_time:.1f}s")
                raise Exception(f"Circuit {self.name} is open, failing fast")
        
        # Execute function
        try:
            if self.state == CircuitState.HALF_OPEN:
                # Set timeout for half-open test
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.half_open_timeout
                )
            else:
                # Normal execution
                result = await func(*args, **kwargs)
            
            # Success, record and possibly reset circuit
            self.record_success()
            return result
            
        except Exception as e:
            # Failure, record and possibly open circuit
            self.record_failure()
            
            # Add context to exception
            if self.state == CircuitState.OPEN:
                raise Exception(f"Circuit {self.name} failed in open state: {str(e)}")
            else:
                raise Exception(f"Circuit {self.name} operation failed: {str(e)}")
    
    def record_success(self) -> None:
        """Record successful operation"""
        self.last_success_time = time.time()
        
        # If half-open, close the circuit
        if self.state == CircuitState.HALF_OPEN:
            self.logger.info(f"Test succeeded in half-open state, closing circuit")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
        
        # In closed state, reset failure count after success
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record failed operation"""
        current_time = time.time()
        self.last_failure_time = current_time
        
        # In closed state, increment failures and possibly open circuit
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            
            if self.failure_count >= self.failure_threshold:
                self.logger.warning(
                    f"Failure threshold reached ({self.failure_count}/{self.failure_threshold}), "
                    f"opening circuit for {self.recovery_timeout}s"
                )
                self.state = CircuitState.OPEN
        
        # In half-open state, immediately open circuit
        elif self.state == CircuitState.HALF_OPEN:
            self.logger.warning(
                f"Failure in half-open state, reopening circuit for {self.recovery_timeout}s"
            )
            self.state = CircuitState.OPEN
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current circuit state
        
        Returns:
            Dictionary with state information
        """
        current_time = time.time()
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "time_since_last_failure": current_time - self.last_failure_time if self.last_failure_time > 0 else None,
            "time_since_last_success": current_time - self.last_success_time,
            "recovery_timeout": self.recovery_timeout,
            "half_open_timeout": self.half_open_timeout
        }
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_success_time = time.time()
        self.logger.info(f"Circuit breaker {self.name} reset to closed state")