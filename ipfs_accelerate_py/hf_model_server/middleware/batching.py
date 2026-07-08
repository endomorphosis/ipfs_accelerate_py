"""
Request batching middleware using anyio.
"""

import anyio
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class BatchResult:
    """Container for batch results."""
    def __init__(self):
        self.value = None
        self.exception = None
        self.event = anyio.Event()
    
    def set_result(self, value):
        self.value = value
        self.event.set()
    
    def set_exception(self, exc):
        self.exception = exc
        self.event.set()
    
    async def get(self):
        await self.event.wait()
        if self.exception:
            raise self.exception
        return self.value


class BatchingMiddleware:
    """Batches requests for improved throughput using anyio."""
    
    def __init__(self, max_batch_size: int = 32, max_wait_ms: int = 100):
        """
        Initialize batching middleware.
        
        Args:
            max_batch_size: Maximum batch size
            max_wait_ms: Maximum wait time in milliseconds
        """
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds
        self._batches: Dict[str, List[Tuple[Dict, BatchResult]]] = defaultdict(list)
        self._locks: Dict[str, anyio.Lock] = {}
        self._flush_events: Dict[str, anyio.Event] = {}
    
    def _get_lock(self, model_id: str) -> anyio.Lock:
        """Get or create lock for model."""
        if model_id not in self._locks:
            self._locks[model_id] = anyio.Lock()
        return self._locks[model_id]
    
    async def add_request(
        self,
        model_id: str,
        request_data: Dict[str, Any],
        inference_fn: Callable
    ) -> Any:
        """
        Add request to batch and wait for result.
        
        Args:
            model_id: Model identifier
            request_data: Request parameters
            inference_fn: Function to call for batch inference
            
        Returns:
            Inference result for this request
        """
        result = BatchResult()
        
        async with self._get_lock(model_id):
            # Add to batch
            self._batches[model_id].append((request_data, result))
            logger.debug(f"Added request to batch for {model_id}, batch size: {len(self._batches[model_id])}")
            
            # Check if batch is full
            if len(self._batches[model_id]) >= self.max_batch_size:
                logger.debug(f"Batch full for {model_id}, flushing immediately")
                # Create task group for flush
                async with anyio.create_task_group() as tg:
                    tg.start_soon(self._flush_batch, model_id, inference_fn)
            elif model_id not in self._flush_events:
                # Schedule flush
                logger.debug(f"Scheduling flush for {model_id} in {self.max_wait_ms}s")
                async with anyio.create_task_group() as tg:
                    tg.start_soon(self._schedule_flush, model_id, inference_fn)
        
        # Wait for result
        return await result.get()
    
    async def _schedule_flush(self, model_id: str, inference_fn: Callable):
        """Schedule a flush after max_wait_ms."""
        await anyio.sleep(self.max_wait_ms)
        await self._flush_batch(model_id, inference_fn)
    
    async def _flush_batch(self, model_id: str, inference_fn: Callable):
        """Flush pending batch."""
        async with self._get_lock(model_id):
            if model_id not in self._batches or len(self._batches[model_id]) == 0:
                return
            
            # Get batch
            batch = self._batches[model_id]
            self._batches[model_id] = []
            
            logger.info(f"Flushing batch for {model_id} with {len(batch)} requests")
        
        # Execute batch inference
        try:
            requests = [req for req, _ in batch]
            results = await inference_fn(requests)
            
            # Distribute results
            for i, (_, result) in enumerate(batch):
                result.set_result(results[i] if i < len(results) else None)
        except Exception as e:
            logger.error(f"Batch inference failed for {model_id}: {e}")
            # Set exception for all results
            for _, result in batch:
                result.set_exception(e)
