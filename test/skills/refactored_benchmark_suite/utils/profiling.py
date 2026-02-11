"""
Profiling utilities for the benchmark suite.
"""

import time
import functools
import logging
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger("benchmark.utils.profiling")

def profile_time(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator to profile the time taken by a function.
    
    Args:
        func: Function to profile
        name: Name for the profile (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profile_name = name or func.__name__
            
            # Start time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Call function
            result = func(*args, **kwargs)
            
            # End time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Log time
            elapsed_time = end_time - start_time
            logger.info(f"Time for {profile_name}: {elapsed_time:.6f} seconds")
            
            return result
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)

def profile_memory(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator to profile the memory usage of a function.
    
    Args:
        func: Function to profile
        name: Name for the profile (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profile_name = name or func.__name__
            
            # Record memory before
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            # Call function
            result = func(*args, **kwargs)
            
            # Record memory after
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                
                # Log memory usage
                memory_used = end_memory - start_memory
                logger.info(f"Memory for {profile_name}: {memory_used / 1024**2:.2f} MB")
                logger.info(f"Peak memory for {profile_name}: {peak_memory / 1024**2:.2f} MB")
            
            return result
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)