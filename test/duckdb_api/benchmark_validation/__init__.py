"""
Benchmark Validation System

A comprehensive framework for validating, certifying, and tracking benchmark results
across different hardware platforms, models, and test scenarios.

This package provides tools for:
- Statistical validation of benchmark data
- Outlier detection in performance metrics
- Reproducibility testing of benchmarks
- Certification of benchmark results
- Quality assessment and monitoring
"""

from duckdb_api.benchmark_validation.core.base import (
    ValidationLevel,
    BenchmarkType,
    ValidationStatus,
    ValidationScope,
    BenchmarkResult,
    ValidationResult,
    BenchmarkValidator,
    ReproducibilityValidator,
    OutlierDetector,
    BenchmarkCertifier,
    ValidationReporter,
    ValidationRepository,
    BenchmarkValidationFramework
)

__version__ = "0.1.0"