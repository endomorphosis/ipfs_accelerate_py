"""
Core Components for Benchmark Validation

This package provides the core components and interfaces for the Benchmark Validation System,
including data structures, validation protocols, and base classes.
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

from duckdb_api.benchmark_validation.core.schema import BenchmarkValidationSchema