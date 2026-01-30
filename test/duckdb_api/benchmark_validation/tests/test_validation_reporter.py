#!/usr/bin/env python3
"""
Tests for the Benchmark Validation Reporter

This module contains unit tests for the ValidationReporterImpl class which generates
reports and visualizations of validation results.
"""

import os
import sys
import unittest
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

# Import validation components
from data.duckdb.benchmark_validation.core.base import (
    ValidationLevel,
    BenchmarkType,
    ValidationStatus,
    BenchmarkResult,
    ValidationResult
)
from data.duckdb.benchmark_validation.visualization.reporter import ValidationReporterImpl

class TestValidationReporter(unittest.TestCase):
    """Test cases for the ValidationReporterImpl class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample reporter
        self.reporter = ValidationReporterImpl({
            "output_directory": "./output"
        })
        
        # Create sample validation results
        self.validation_results = self._create_sample_validation_results()
    
    def _create_sample_validation_results(self):
        """Create sample validation results for testing."""
        results = []
        
        # Define some benchmark types, model IDs, and hardware IDs
        benchmark_types = [BenchmarkType.PERFORMANCE, BenchmarkType.COMPATIBILITY]
        model_ids = ["bert-base-uncased", "vit-base-patch16-224", "whisper-tiny"]
        hardware_ids = ["cpu", "gpu", "webgpu"]
        validation_levels = [ValidationLevel.MINIMAL, ValidationLevel.STANDARD, ValidationLevel.STRICT]
        statuses = [ValidationStatus.VALID, ValidationStatus.WARNING, ValidationStatus.INVALID]
        
        # Create a variety of results
        for i in range(10):
            # Create a benchmark result
            benchmark_result = BenchmarkResult(
                result_id=f"benchmark-{i}",
                benchmark_type=benchmark_types[i % len(benchmark_types)],
                model_id=model_ids[i % len(model_ids)],
                hardware_id=hardware_ids[i % len(hardware_ids)],
                metrics={
                    "throughput_items_per_second": 100 + i * 10,
                    "average_latency_ms": 50 - i * 2,
                    "memory_peak_mb": 500 + i * 20
                },
                run_id=i,
                timestamp=datetime.now(),
                metadata={"batch_size": 1}
            )
            
            # Create validation metrics based on the benchmark result
            validation_metrics = {
                "throughput_score": 0.8 + (i % 3) * 0.1,
                "latency_score": 0.7 + (i % 3) * 0.1,
                "memory_score": 0.9 - (i % 3) * 0.1,
                "overall_score": 0.8 + (i % 5) * 0.05
            }
            
            # Create some issues for some results
            issues = []
            if i % 3 == 1:
                issues = [
                    {"description": "Minor throughput fluctuation", "severity": "low"},
                    {"description": "Memory usage above threshold", "severity": "medium"}
                ]
            elif i % 3 == 2:
                issues = [
                    {"description": "High latency variability", "severity": "high"},
                ]
            
            # Create some recommendations
            recommendations = []
            if issues:
                recommendations = [
                    "Consider running additional tests to confirm results",
                    "Check for system load during benchmark execution"
                ]
            
            # Create validation result
            validation_result = ValidationResult(
                benchmark_result=benchmark_result,
                status=statuses[i % len(statuses)],
                validation_level=validation_levels[i % len(validation_levels)],
                confidence_score=0.7 + (i % 4) * 0.1,
                validation_metrics=validation_metrics,
                issues=issues,
                recommendations=recommendations,
                validation_timestamp=datetime.now(),
                validator_id="test-validator"
            )
            
            results.append(validation_result)
        
        return results
    
    def test_report_generation_html(self):
        """Test HTML report generation."""
        # Generate report
        report = self.reporter.generate_report(
            self.validation_results,
            report_format="html",
            include_visualizations=True
        )
        
        # Check that report is not empty
        self.assertTrue(len(report) > 0)
        
        # Check for key elements in the report
        self.assertIn("<!DOCTYPE html>", report)
        self.assertIn("<title>", report)
        self.assertIn("Summary", report)
        self.assertIn("Detailed Results", report)
        
        # Check that it contains benchmark types
        self.assertIn("PERFORMANCE", report)
        self.assertIn("COMPATIBILITY", report)
        
        # Check that it contains validation levels
        self.assertIn("MINIMAL", report)
        self.assertIn("STANDARD", report)
        self.assertIn("STRICT", report)
    
    def test_report_generation_markdown(self):
        """Test Markdown report generation."""
        # Generate report
        report = self.reporter.generate_report(
            self.validation_results,
            report_format="markdown",
            include_visualizations=True
        )
        
        # Check that report is not empty
        self.assertTrue(len(report) > 0)
        
        # Check for key elements in the report
        self.assertIn("# ", report)  # Markdown title
        self.assertIn("## Summary", report)
        self.assertIn("## Detailed Results", report)
        
        # Check for table headers
        self.assertIn("| --- | --- |", report)
        
        # Check that it contains benchmark types
        self.assertIn("PERFORMANCE", report)
        self.assertIn("COMPATIBILITY", report)
    
    def test_report_generation_json(self):
        """Test JSON report generation."""
        # Generate report
        report = self.reporter.generate_report(
            self.validation_results,
            report_format="json"
        )
        
        # Check that report is not empty
        self.assertTrue(len(report) > 0)
        
        # Parse JSON to verify structure
        report_data = json.loads(report)
        
        # Check for key elements
        self.assertIn("title", report_data)
        self.assertIn("timestamp", report_data)
        self.assertIn("total_results", report_data)
        self.assertIn("benchmark_type_stats", report_data)
        self.assertIn("validation_level_stats", report_data)
        self.assertIn("detailed_results", report_data)
        
        # Check that the counts match
        self.assertEqual(report_data["total_results"], len(self.validation_results))
        self.assertLessEqual(len(report_data["detailed_results"]), len(self.validation_results))
    
    def test_report_export(self):
        """Test exporting report to a file."""
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export report to HTML
            output_path = os.path.join(temp_dir, "validation_report.html")
            result_path = self.reporter.export_report(
                self.validation_results,
                output_path=output_path,
                report_format="html"
            )
            
            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(result_path, output_path)
            
            # Check file contents
            with open(output_path, 'r') as f:
                content = f.read()
                self.assertIn("<!DOCTYPE html>", content)
                self.assertIn("<title>", content)
    
    def test_empty_results(self):
        """Test report generation with empty results."""
        # Generate report with empty results
        report = self.reporter.generate_report(
            [],
            report_format="html"
        )
        
        # Check that report indicates no results
        self.assertEqual(report, "No validation results to report")
    
    def test_visualization_creation(self):
        """Test visualization creation if dependencies are available."""
        # Skip test if visualization libraries are not available
        try:
            import plotly
            import pandas
        except ImportError:
            self.skipTest("Plotly or pandas not installed")
        
        # Create visualization
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "confidence_distribution.html")
            result = self.reporter.create_visualization(
                self.validation_results,
                visualization_type="confidence_distribution",
                output_path=output_path,
                title="Test Confidence Distribution"
            )
            
            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(result, output_path)

if __name__ == "__main__":
    unittest.main()