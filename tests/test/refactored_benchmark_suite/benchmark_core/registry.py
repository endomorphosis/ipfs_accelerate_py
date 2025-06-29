"""
Benchmark Registry Module

This module provides a registry system for benchmark implementations.
"""

import logging
from typing import Dict, Any, Type, Optional, List, Callable

logger = logging.getLogger(__name__)

class BenchmarkRegistry:
    """
    Central registry for all benchmark implementations.
    
    This class serves as a registry for all benchmark implementations, allowing
    benchmarks to be registered with metadata and later retrieved by name.
    
    Usage:
        @BenchmarkRegistry.register(name="model_throughput", category="inference", 
                                   models=["bert", "vit"], hardware=["cpu", "cuda"])
        class ModelThroughputBenchmark(BenchmarkBase):
            # Implementation
            pass
    """
    
    _registry: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: Optional[str] = None, **kwargs) -> Callable:
        """
        Register a benchmark implementation.
        
        Args:
            name: Optional name for the benchmark. If not provided, class name is used.
            **kwargs: Metadata attributes for the benchmark.
            
        Returns:
            Decorator function that registers the benchmark class.
        """
        def decorator(benchmark_class: Type) -> Type:
            registry_name = name or benchmark_class.__name__
            cls._registry[registry_name] = {
                'class': benchmark_class,
                'metadata': kwargs
            }
            logger.debug(f"Registered benchmark: {registry_name}")
            return benchmark_class
        return decorator
        
    @classmethod
    def get_benchmark(cls, name: str) -> Optional[Type]:
        """
        Get a benchmark implementation by name.
        
        Args:
            name: Name of the benchmark to retrieve.
            
        Returns:
            Benchmark class or None if not found.
        """
        if name not in cls._registry:
            logger.warning(f"Benchmark not found: {name}")
            return None
            
        return cls._registry[name]['class']
        
    @classmethod
    def list_benchmarks(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all registered benchmarks with their metadata.
        
        Returns:
            Dictionary mapping benchmark names to metadata.
        """
        return {name: meta['metadata'] for name, meta in cls._registry.items()}
        
    @classmethod
    def get_benchmarks_by_category(cls, category: str) -> List[str]:
        """
        Get list of benchmark names in a specific category.
        
        Args:
            category: Category to filter by.
            
        Returns:
            List of benchmark names in the specified category.
        """
        return [
            name for name, data in cls._registry.items()
            if data['metadata'].get('category') == category
        ]
        
    @classmethod
    def get_benchmarks_by_hardware(cls, hardware: str) -> List[str]:
        """
        Get list of benchmark names supporting specific hardware.
        
        Args:
            hardware: Hardware type to filter by.
            
        Returns:
            List of benchmark names supporting the specified hardware.
        """
        return [
            name for name, data in cls._registry.items()
            if 'hardware' in data['metadata'] and hardware in data['metadata']['hardware']
        ]