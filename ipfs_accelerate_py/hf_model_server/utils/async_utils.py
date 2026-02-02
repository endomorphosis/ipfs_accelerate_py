"""
Async utility functions using anyio.
"""

import anyio
import inspect
import logging
from typing import Callable, Any, List
from functools import wraps

logger = logging.getLogger(__name__)


async def timeout(coro, seconds: float):
    """
    Add timeout to a coroutine.
    
    Args:
        coro: Coroutine to execute
        seconds: Timeout in seconds
        
    Returns:
        Result from coroutine
        
    Raises:
        TimeoutError: If timeout exceeded
    """
    try:
        with anyio.fail_after(seconds):
            return await coro
    except TimeoutError:
        logger.warning(f"Operation timed out after {seconds}s")
        raise


async def retry_with_backoff(
    fn: Callable,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> Any:
    """
    Retry function with exponential backoff.
    
    Args:
        fn: Function to retry
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Backoff multiplier
        
    Returns:
        Result from function
        
    Raises:
        Exception: Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            if inspect.iscoroutinefunction(fn):
                return await fn()
            else:
                return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                await anyio.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"All {max_attempts} attempts failed")
    
    raise last_exception


async def gather_with_timeout(
    coros: List,
    timeout_seconds: float
) -> List[Any]:
    """
    Gather coroutines with overall timeout using anyio task groups.
    
    Args:
        coros: List of coroutines
        timeout_seconds: Overall timeout
        
    Returns:
        List of results
        
    Raises:
        TimeoutError: If timeout exceeded
    """
    results = []
    
    try:
        with anyio.fail_after(timeout_seconds):
            async with anyio.create_task_group() as tg:
                for coro in coros:
                    tg.start_soon(_gather_helper, coro, results)
    except TimeoutError:
        logger.warning(f"Gather timed out after {timeout_seconds}s")
        raise
    
    return results


async def _gather_helper(coro, results_list):
    """Helper function to gather results."""
    result = await coro
    results_list.append(result)
