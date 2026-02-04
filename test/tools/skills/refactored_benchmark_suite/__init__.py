"""
Refactored HuggingFace Model Benchmark Suite

A comprehensive benchmarking framework for HuggingFace models with enhanced features,
extensibility, and reporting capabilities.
"""

__version__ = "0.1.0"

from test.tools.skills.refactored_benchmark_suite.benchmark import ModelBenchmark, BenchmarkResults, BenchmarkSuite, BenchmarkConfig
from test.tools.skills.refactored_benchmark_suite.metrics import LatencyMetric, ThroughputMetric, MemoryMetric, FLOPsMetric

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