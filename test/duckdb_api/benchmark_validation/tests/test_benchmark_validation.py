#!/usr/bin/env python3
"""
Tests for the Benchmark Validation System.

This module contains tests for the core functionality of the benchmark validation system.
"""

import os
import sys
import unittest
import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path for module imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from duckdb_api.benchmark_validation.core.base import (
    ValidationLevel,
    BenchmarkType,
    ValidationStatus,
    BenchmarkResult,
    ValidationResult,
    BenchmarkValidationFramework
)
from duckdb_api.benchmark_validation.validation_protocol import StandardBenchmarkValidator
from duckdb_api.benchmark_validation.outlier_detection import StatisticalOutlierDetector
from duckdb_api.benchmark_validation.reproducibility import ReproducibilityValidator
from duckdb_api.benchmark_validation.certification import BenchmarkCertificationSystem

class TestBenchmarkValidation(unittest.TestCase):
    """Test case for benchmark validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create validator
        self.validator = StandardBenchmarkValidator()
        
        # Create outlier detector
        self.outlier_detector = StatisticalOutlierDetector()
        
        # Create reproducibility validator
        self.reproducibility_validator = ReproducibilityValidator()
        
        # Create certifier
        self.certifier = BenchmarkCertificationSystem()
        
        # Create framework
        self.framework = BenchmarkValidationFramework(
            validator=self.validator,
            outlier_detector=self.outlier_detector,
            reproducibility_validator=self.reproducibility_validator,
            certifier=self.certifier,
        )
        
        # Create benchmark results
        self.benchmark_results = []
        for i in range(5):
            benchmark_result = BenchmarkResult(
                result_id=f"benchmark-{i}",
                benchmark_type=BenchmarkType.PERFORMANCE,
                model_id=1,  # BERT model
                hardware_id=2,  # NVIDIA GPU
                metrics={
                    "average_latency_ms": 15.3 + (0.2 * i),  # Slight variation
                    "throughput_items_per_second": 156.7 - (0.5 * i),
                    "memory_peak_mb": 3450.2 + (10 * i),
                    "total_time_seconds": 120.5 + (0.3 * i)
                },
                run_id=100 + i,
                timestamp=datetime.datetime.now() - datetime.timedelta(hours=i),
                metadata={
                    "test_environment": "cloud",
                    "software_versions": {"framework": "1.2.3"},
                    "test_parameters": {"batch_size": 32, "precision": "fp16"}
                }
            )
            self.benchmark_results.append(benchmark_result)
    
    def test_standard_validation(self):
        """Test standard validation of benchmark results."""
        # Validate a benchmark result
        validation_result = self.validator.validate(
            benchmark_result=self.benchmark_results[0],
            validation_level=ValidationLevel.STANDARD
        )
        
        # Check result
        self.assertEqual(validation_result.status, ValidationStatus.VALID)
        self.assertGreater(validation_result.confidence_score, 0.7)
        
        # Validate with missing metrics
        benchmark_result = BenchmarkResult(
            result_id="benchmark-missing-metrics",
            benchmark_type=BenchmarkType.PERFORMANCE,
            model_id=1,
            hardware_id=2,
            metrics={
                "average_latency_ms": 15.3,
                # Missing required metric: throughput_items_per_second
            },
            run_id=100,
            timestamp=datetime.datetime.now(),
            metadata={}
        )
        
        validation_result = self.validator.validate(
            benchmark_result=benchmark_result,
            validation_level=ValidationLevel.STANDARD
        )
        
        # Check result
        self.assertEqual(validation_result.status, ValidationStatus.INVALID)
        self.assertLess(validation_result.confidence_score, 0.7)
    
    def test_outlier_detection(self):
        """Test outlier detection."""
        # Create benchmark results with an outlier
        benchmark_results = self.benchmark_results.copy()
        outlier = BenchmarkResult(
            result_id="benchmark-outlier",
            benchmark_type=BenchmarkType.PERFORMANCE,
            model_id=1,
            hardware_id=2,
            metrics={
                "average_latency_ms": 50.0,  # Much higher than others
                "throughput_items_per_second": 50.0,  # Much lower than others
                "memory_peak_mb": 3500.0,
                "total_time_seconds": 150.0
            },
            run_id=200,
            timestamp=datetime.datetime.now(),
            metadata={}
        )
        benchmark_results.append(outlier)
        
        # Detect outliers
        outliers = self.outlier_detector.detect_outliers(
            benchmark_results=benchmark_results,
            metrics=["average_latency_ms", "throughput_items_per_second"],
            threshold=2.0
        )
        
        # Check results
        self.assertIn("average_latency_ms", outliers)
        self.assertIn("throughput_items_per_second", outliers)
        self.assertGreaterEqual(len(outliers["average_latency_ms"]), 1)
        self.assertGreaterEqual(len(outliers["throughput_items_per_second"]), 1)
        
        # Check that the outlier was detected
        outlier_result_ids = [result.result_id for result in outliers["average_latency_ms"]]
        self.assertIn("benchmark-outlier", outlier_result_ids)
    
    def test_reproducibility_validation(self):
        """Test reproducibility validation."""
        # Validate reproducibility
        reproducibility_result = self.reproducibility_validator.validate_reproducibility(
            benchmark_results=self.benchmark_results,
            validation_level=ValidationLevel.STANDARD
        )
        
        # Check result
        self.assertEqual(reproducibility_result.status, ValidationStatus.VALID)
        self.assertGreater(reproducibility_result.confidence_score, 0.7)
        self.assertIn("reproducibility", reproducibility_result.validation_metrics)
        self.assertIn("reproducibility_score", reproducibility_result.validation_metrics["reproducibility"])
        
        # Create benchmark results with poor reproducibility
        poor_reproducibility = self.benchmark_results.copy()
        poor_reproducibility.append(BenchmarkResult(
            result_id="benchmark-poor-repro",
            benchmark_type=BenchmarkType.PERFORMANCE,
            model_id=1,
            hardware_id=2,
            metrics={
                "average_latency_ms": 30.0,  # Much higher than others
                "throughput_items_per_second": 80.0,  # Much lower than others
                "memory_peak_mb": 4000.0,
                "total_time_seconds": 180.0
            },
            run_id=300,
            timestamp=datetime.datetime.now(),
            metadata={}
        ))
        
        # Validate reproducibility
        reproducibility_result = self.reproducibility_validator.validate_reproducibility(
            benchmark_results=poor_reproducibility,
            validation_level=ValidationLevel.STANDARD
        )
        
        # Check result - might be WARNING or VALID depending on CV threshold
        self.assertIn(reproducibility_result.status, [ValidationStatus.WARNING, ValidationStatus.VALID])
    
    def test_certification(self):
        """Test benchmark certification."""
        # Validate benchmark results
        validation_results = []
        for benchmark_result in self.benchmark_results:
            validation_result = self.validator.validate(
                benchmark_result=benchmark_result,
                validation_level=ValidationLevel.STANDARD
            )
            validation_results.append(validation_result)
        
        # Validate reproducibility
        reproducibility_result = self.reproducibility_validator.validate_reproducibility(
            benchmark_results=self.benchmark_results,
            validation_level=ValidationLevel.STANDARD
        )
        
        # Certify benchmark
        certification = self.certifier.certify(
            benchmark_result=self.benchmark_results[0],
            validation_results=validation_results + [reproducibility_result],
            certification_level="auto"
        )
        
        # Check certification
        self.assertIn(certification["certification_level"], ["basic", "standard", "advanced", "gold"])
        self.assertEqual(certification["benchmark_id"], self.benchmark_results[0].result_id)
        self.assertEqual(certification["model_id"], self.benchmark_results[0].model_id)
        self.assertEqual(certification["hardware_id"], self.benchmark_results[0].hardware_id)
        
        # Verify certification
        verification = self.certifier.verify_certification(
            certification=certification,
            benchmark_result=self.benchmark_results[0]
        )
        self.assertTrue(verification)
    
    def test_framework_validation(self):
        """Test framework validation."""
        # Validate a benchmark result
        validation_result = self.framework.validate(
            benchmark_result=self.benchmark_results[0],
            validation_level=ValidationLevel.STANDARD
        )
        
        # Check result
        self.assertEqual(validation_result.status, ValidationStatus.VALID)
        self.assertGreater(validation_result.confidence_score, 0.7)
        
        # Validate batch
        validation_results = self.framework.validate_batch(
            benchmark_results=self.benchmark_results,
            validation_level=ValidationLevel.STANDARD,
            detect_outliers=True
        )
        
        # Check results
        self.assertEqual(len(validation_results), len(self.benchmark_results))
        for result in validation_results:
            self.assertEqual(result.status, ValidationStatus.VALID)
            self.assertGreater(result.confidence_score, 0.7)
        
        # Validate reproducibility
        reproducibility_result = self.framework.validate_reproducibility(
            benchmark_results=self.benchmark_results,
            validation_level=ValidationLevel.STANDARD
        )
        
        # Check result
        self.assertEqual(reproducibility_result.status, ValidationStatus.VALID)
        self.assertGreater(reproducibility_result.confidence_score, 0.7)

if __name__ == "__main__":
    unittest.main()