"""
Refactored HuggingFace Model Benchmark Suite

A comprehensive benchmarking framework for HuggingFace models with enhanced features,
extensibility, and reporting capabilities.
"""

__version__ = "0.1.0"

from .benchmark import ModelBenchmark, BenchmarkResults, BenchmarkSuite, BenchmarkConfig
from .metrics import LatencyMetric, ThroughputMetric, MemoryMetric, FLOPsMetric

__all__ = [
    "ModelBenchmark",
    "BenchmarkResults",
    "BenchmarkSuite",
    "BenchmarkConfig",
    "LatencyMetric",
    "ThroughputMetric",
    "MemoryMetric",
    "FLOPsMetric",
]