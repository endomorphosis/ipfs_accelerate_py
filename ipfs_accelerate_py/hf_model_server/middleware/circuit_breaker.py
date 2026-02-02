"""
Circuit breaker pattern for fault tolerance.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Callable, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


@dataclass
class CircuitStats:
    """Statistics for a circuit."""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = field(default_factory=time.time)


class CircuitBreaker:
    """Circuit breaker for model inference."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: int = 60
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            success_threshold: Successes needed to close from half-open
            timeout_seconds: Time before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        
        self._circuits: Dict[str, CircuitState] = {}
        self._stats: Dict[str, CircuitStats] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
    
    def _get_lock(self, model_id: str) -> asyncio.Lock:
        """Get or create lock for model."""
        if model_id not in self._locks:
            self._locks[model_id] = asyncio.Lock()
        return self._locks[model_id]
    
    def _get_state(self, model_id: str) -> CircuitState:
        """Get circuit state for model."""
        return self._circuits.get(model_id, CircuitState.CLOSED)
    
    def _get_stats(self, model_id: str) -> CircuitStats:
        """Get or create stats for model."""
        if model_id not in self._stats:
            self._stats[model_id] = CircuitStats()
        return self._stats[model_id]
    
    async def call(self, model_id: str, fn: Callable) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            model_id: Model identifier
            fn: Function to execute
            
        Returns:
            Result from function
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        async with self._get_lock(model_id):
            state = self._get_state(model_id)
            stats = self._get_stats(model_id)
            
            # Check if should attempt reset
            if state == CircuitState.OPEN:
                if time.time() - stats.last_state_change >= self.timeout_seconds:
                    logger.info(f"Circuit half-open for {model_id}, attempting recovery")
                    self._circuits[model_id] = CircuitState.HALF_OPEN
                    stats.last_state_change = time.time()
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker open for {model_id}"
                    )
        
        # Execute function
        try:
            result = await fn()
            await self._on_success(model_id)
            return result
        except Exception as e:
            await self._on_failure(model_id)
            raise
    
    async def _on_success(self, model_id: str):
        """Handle successful call."""
        async with self._get_lock(model_id):
            state = self._get_state(model_id)
            stats = self._get_stats(model_id)
            
            stats.success_count += 1
            stats.failure_count = 0
            
            if state == CircuitState.HALF_OPEN:
                if stats.success_count >= self.success_threshold:
                    logger.info(f"Circuit closed for {model_id}")
                    self._circuits[model_id] = CircuitState.CLOSED
                    stats.last_state_change = time.time()
                    stats.success_count = 0
    
    async def _on_failure(self, model_id: str):
        """Handle failed call."""
        async with self._get_lock(model_id):
            state = self._get_state(model_id)
            stats = self._get_stats(model_id)
            
            stats.failure_count += 1
            stats.success_count = 0
            stats.last_failure_time = time.time()
            
            if state == CircuitState.CLOSED:
                if stats.failure_count >= self.failure_threshold:
                    logger.warning(f"Circuit opened for {model_id}")
                    self._circuits[model_id] = CircuitState.OPEN
                    stats.last_state_change = time.time()
            elif state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit re-opened for {model_id}")
                self._circuits[model_id] = CircuitState.OPEN
                stats.last_state_change = time.time()
    
    def get_state(self, model_id: str) -> Dict[str, Any]:
        """Get circuit state and stats."""
        state = self._get_state(model_id)
        stats = self._get_stats(model_id)
        
        return {
            "state": state.value,
            "failure_count": stats.failure_count,
            "success_count": stats.success_count,
            "last_failure_time": stats.last_failure_time,
            "last_state_change": stats.last_state_change,
        }
    
    async def reset(self, model_id: str):
        """Reset circuit breaker for model."""
        async with self._get_lock(model_id):
            self._circuits[model_id] = CircuitState.CLOSED
            self._stats[model_id] = CircuitStats()
            logger.info(f"Circuit reset for {model_id}")
