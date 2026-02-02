"""
Request batching middleware.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class BatchingMiddleware:
    """Batches requests for improved throughput."""
    
    def __init__(self, max_batch_size: int = 32, max_wait_ms: int = 100):
        """
        Initialize batching middleware.
        
        Args:
            max_batch_size: Maximum batch size
            max_wait_ms: Maximum wait time in milliseconds
        """
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds
        self._batches: Dict[str, List[Tuple[Dict, asyncio.Future]]] = defaultdict(list)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._flush_tasks: Dict[str, asyncio.Task] = {}
    
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
        future = asyncio.Future()
        
        async with self._locks[model_id]:
            # Add to batch
            self._batches[model_id].append((request_data, future))
            logger.debug(f"Added request to batch for {model_id}, batch size: {len(self._batches[model_id])}")
            
            # Check if batch is full
            if len(self._batches[model_id]) >= self.max_batch_size:
                logger.debug(f"Batch full for {model_id}, flushing immediately")
                asyncio.create_task(self._flush_batch(model_id, inference_fn))
            elif model_id not in self._flush_tasks:
                # Schedule flush
                logger.debug(f"Scheduling flush for {model_id} in {self.max_wait_ms}s")
                self._flush_tasks[model_id] = asyncio.create_task(
                    self._schedule_flush(model_id, inference_fn)
                )
        
        # Wait for result
        return await future
    
    async def _schedule_flush(self, model_id: str, inference_fn: Callable):
        """Schedule a flush after max_wait_ms."""
        await asyncio.sleep(self.max_wait_ms)
        await self._flush_batch(model_id, inference_fn)
    
    async def _flush_batch(self, model_id: str, inference_fn: Callable):
        """Flush pending batch."""
        async with self._locks[model_id]:
            if model_id not in self._batches or len(self._batches[model_id]) == 0:
                return
            
            # Get batch
            batch = self._batches[model_id]
            self._batches[model_id] = []
            
            # Cancel flush task if exists
            if model_id in self._flush_tasks:
                task = self._flush_tasks[model_id]
                if not task.done():
                    task.cancel()
                del self._flush_tasks[model_id]
            
            logger.info(f"Flushing batch for {model_id} with {len(batch)} requests")
        
        # Execute batch inference
        try:
            requests = [req for req, _ in batch]
            results = await inference_fn(requests)
            
            # Distribute results
            for i, (_, future) in enumerate(batch):
                if not future.done():
                    future.set_result(results[i] if i < len(results) else None)
        except Exception as e:
            logger.error(f"Batch inference failed for {model_id}: {e}")
            # Set exception for all futures
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)
