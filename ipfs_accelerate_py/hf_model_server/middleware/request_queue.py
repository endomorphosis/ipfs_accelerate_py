"""
Request queuing system for managing inference workload.
"""

import anyio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class QueuedRequest:
    """Container for a queued request."""
    request_id: str
    model_id: str
    data: Dict[str, Any]
    priority: RequestPriority = RequestPriority.NORMAL
    queued_at: datetime = field(default_factory=datetime.utcnow)
    timeout_seconds: float = 30.0
    callback: Optional[Callable] = None
    
    def is_expired(self) -> bool:
        """Check if request has expired."""
        elapsed = (datetime.utcnow() - self.queued_at).total_seconds()
        return elapsed > self.timeout_seconds
    
    def __lt__(self, other):
        """Compare by priority (higher priority first) then by queue time."""
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        return self.queued_at < other.queued_at


class RequestQueue:
    """Async request queue with priority and timeout support."""
    
    def __init__(
        self,
        max_size: int = 100,
        timeout_seconds: float = 30.0,
        enable_priority: bool = True
    ):
        """
        Initialize request queue.
        
        Args:
            max_size: Maximum queue size
            timeout_seconds: Default timeout for requests
            enable_priority: Enable priority queue
        """
        self.max_size = max_size
        self.timeout_seconds = timeout_seconds
        self.enable_priority = enable_priority
        
        # Use anyio Event for signaling
        self._queue: list[QueuedRequest] = []
        self._lock = anyio.Lock()
        self._not_empty = anyio.Event()
        self._not_full = anyio.Event()
        self._not_full.set()  # Initially not full
        
        # Statistics
        self._total_queued = 0
        self._total_processed = 0
        self._total_timeouts = 0
        self._total_rejected = 0
    
    async def enqueue(
        self,
        request_id: str,
        model_id: str,
        data: Dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout_seconds: Optional[float] = None
    ) -> bool:
        """
        Add request to queue.
        
        Args:
            request_id: Unique request ID
            model_id: Model to use for inference
            data: Request data
            priority: Request priority
            timeout_seconds: Override default timeout
            
        Returns:
            True if queued successfully, False if queue full
        """
        async with self._lock:
            # Check if queue is full
            if len(self._queue) >= self.max_size:
                self._total_rejected += 1
                logger.warning(f"Queue full, rejected request: {request_id}")
                return False
            
            # Create queued request
            queued_request = QueuedRequest(
                request_id=request_id,
                model_id=model_id,
                data=data,
                priority=priority,
                timeout_seconds=timeout_seconds or self.timeout_seconds
            )
            
            # Add to queue
            self._queue.append(queued_request)
            self._total_queued += 1
            
            # Sort by priority if enabled
            if self.enable_priority:
                self._queue.sort()
            
            # Signal that queue is not empty
            self._not_empty.set()
            
            # Check if queue is now full
            if len(self._queue) >= self.max_size:
                self._not_full = anyio.Event()  # Reset event
            
            logger.debug(f"Queued request: {request_id} (queue size: {len(self._queue)})")
            return True
    
    async def dequeue(self, timeout: Optional[float] = None) -> Optional[QueuedRequest]:
        """
        Remove and return next request from queue.
        
        Args:
            timeout: Maximum time to wait for a request
            
        Returns:
            QueuedRequest or None if timeout
        """
        deadline = time.time() + timeout if timeout else None
        
        while True:
            async with self._lock:
                # Remove expired requests
                self._remove_expired()
                
                # Check if queue has items
                if self._queue:
                    request = self._queue.pop(0)
                    self._total_processed += 1
                    
                    # Signal that queue is not full
                    if len(self._queue) < self.max_size:
                        self._not_full.set()
                    
                    # Check if queue is now empty
                    if not self._queue:
                        self._not_empty = anyio.Event()  # Reset event
                    
                    logger.debug(f"Dequeued request: {request.request_id} (queue size: {len(self._queue)})")
                    return request
            
            # Check timeout
            if deadline and time.time() >= deadline:
                return None
            
            # Wait for queue to have items
            try:
                with anyio.fail_after(1.0 if not deadline else min(1.0, deadline - time.time())):
                    await self._not_empty.wait()
            except TimeoutError:
                if deadline and time.time() >= deadline:
                    return None
                continue
    
    def _remove_expired(self):
        """Remove expired requests from queue (must hold lock)."""
        original_size = len(self._queue)
        self._queue = [r for r in self._queue if not r.is_expired()]
        expired_count = original_size - len(self._queue)
        if expired_count > 0:
            self._total_timeouts += expired_count
            logger.warning(f"Removed {expired_count} expired requests from queue")
    
    async def size(self) -> int:
        """Get current queue size."""
        async with self._lock:
            return len(self._queue)
    
    async def is_full(self) -> bool:
        """Check if queue is full."""
        async with self._lock:
            return len(self._queue) >= self.max_size
    
    async def is_empty(self) -> bool:
        """Check if queue is empty."""
        async with self._lock:
            return len(self._queue) == 0
    
    async def clear(self):
        """Clear all requests from queue."""
        async with self._lock:
            count = len(self._queue)
            self._queue.clear()
            self._not_full.set()
            self._not_empty = anyio.Event()
            logger.info(f"Cleared {count} requests from queue")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "current_size": len(self._queue),
            "max_size": self.max_size,
            "total_queued": self._total_queued,
            "total_processed": self._total_processed,
            "total_timeouts": self._total_timeouts,
            "total_rejected": self._total_rejected,
            "utilization": len(self._queue) / self.max_size if self.max_size > 0 else 0,
        }


class QueueManager:
    """Manages multiple request queues (e.g., per model or per priority)."""
    
    def __init__(
        self,
        default_max_size: int = 100,
        default_timeout: float = 30.0,
        enable_per_model_queues: bool = False
    ):
        """
        Initialize queue manager.
        
        Args:
            default_max_size: Default max size for queues
            default_timeout: Default timeout for requests
            enable_per_model_queues: Create separate queue per model
        """
        self.default_max_size = default_max_size
        self.default_timeout = default_timeout
        self.enable_per_model_queues = enable_per_model_queues
        
        # Global queue or per-model queues
        if enable_per_model_queues:
            self._queues: Dict[str, RequestQueue] = {}
        else:
            self._global_queue = RequestQueue(
                max_size=default_max_size,
                timeout_seconds=default_timeout
            )
    
    def _get_queue(self, model_id: Optional[str] = None) -> RequestQueue:
        """Get queue for model (or global queue)."""
        if not self.enable_per_model_queues:
            return self._global_queue
        
        if model_id not in self._queues:
            self._queues[model_id] = RequestQueue(
                max_size=self.default_max_size,
                timeout_seconds=self.default_timeout
            )
            logger.info(f"Created queue for model: {model_id}")
        
        return self._queues[model_id]
    
    async def enqueue(
        self,
        request_id: str,
        model_id: str,
        data: Dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout_seconds: Optional[float] = None
    ) -> bool:
        """Enqueue request."""
        queue = self._get_queue(model_id if self.enable_per_model_queues else None)
        return await queue.enqueue(
            request_id=request_id,
            model_id=model_id,
            data=data,
            priority=priority,
            timeout_seconds=timeout_seconds
        )
    
    async def dequeue(
        self,
        model_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Optional[QueuedRequest]:
        """Dequeue request."""
        queue = self._get_queue(model_id if self.enable_per_model_queues else None)
        return await queue.dequeue(timeout=timeout)
    
    async def get_stats(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get queue statistics."""
        if not self.enable_per_model_queues:
            return self._global_queue.get_stats()
        
        if model_id:
            queue = self._queues.get(model_id)
            if queue:
                return queue.get_stats()
            return {}
        
        # Aggregate stats for all queues
        stats = {
            "queues": {},
            "total_size": 0,
            "total_queued": 0,
            "total_processed": 0,
        }
        
        for mid, queue in self._queues.items():
            queue_stats = queue.get_stats()
            stats["queues"][mid] = queue_stats
            stats["total_size"] += queue_stats["current_size"]
            stats["total_queued"] += queue_stats["total_queued"]
            stats["total_processed"] += queue_stats["total_processed"]
        
        return stats
