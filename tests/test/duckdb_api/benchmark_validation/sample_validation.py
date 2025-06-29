#!/usr/bin/env python3
"""
Sample Demonstration of the Benchmark Validation System

This script demonstrates the key features of the Benchmark Validation System,
including validation, outlier detection, reproducibility testing, and certification.
"""

import os
import sys
import datetime
import json
import logging
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark_validation_demo")

from duckdb_api.benchmark_validation import (
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
from duckdb_api.benchmark_validation.framework import ComprehensiveBenchmarkValidation

def create_sample_benchmark_results(num_results=5, add_outlier=False):
    """
    Create sample benchmark results for demonstration.
    
    Args:
        num_results: Number of benchmark results to create
        add_outlier: Whether to add an outlier result
        
    Returns:
        List of BenchmarkResult objects
    """
    # Create benchmark results
    benchmark_results = []
    for i in range(num_results):
        benchmark_result = BenchmarkResult(
            result_id=f"benchmark-{i}",
            benchmark_type=BenchmarkType.PERFORMANCE,
            model_id=1,  # BERT model
            hardware_id=2,  # NVIDIA GPU
            metrics={
                "average_latency_ms": 15.3 + (0.2 * i),  # Slight variation
                "throughput_items_per_second": 156.7 - (0.5 * i),
                "memory_peak_mb": 3450.2 + (10 * i),
                "total_time_seconds": 120.5 + (0.3 * i),
                "warmup_iterations": 10,
                "iterations": 100
            },
            run_id=100 + i,
            timestamp=datetime.datetime.now() - datetime.timedelta(hours=i),
            metadata={
                "test_environment": "cloud",
                "software_versions": {"framework": "1.2.3"},
                "test_parameters": {"batch_size": 32, "precision": "fp16"},
                "hardware_details": {"gpu": "NVIDIA A100", "memory": "40GB"},
                "model_details": {"type": "BERT", "size": "base"}
            }
        )
        benchmark_results.append(benchmark_result)
    
    # Add outlier if requested
    if add_outlier:
        outlier = BenchmarkResult(
            result_id="benchmark-outlier",
            benchmark_type=BenchmarkType.PERFORMANCE,
            model_id=1,  # BERT model
            hardware_id=2,  # NVIDIA GPU
            metrics={
                "average_latency_ms": 35.0,  # Much higher latency
                "throughput_items_per_second": 80.0,  # Much lower throughput
                "memory_peak_mb": 4500.0,  # Higher memory usage
                "total_time_seconds": 180.0,  # Longer execution time
                "warmup_iterations": 10,
                "iterations": 100
            },
            run_id=200,
            timestamp=datetime.datetime.now(),
            metadata={
                "test_environment": "cloud",
                "software_versions": {"framework": "1.2.3"},
                "test_parameters": {"batch_size": 32, "precision": "fp16"},
                "hardware_details": {"gpu": "NVIDIA A100", "memory": "40GB"},
                "model_details": {"type": "BERT", "size": "base"}
            }
        )
        benchmark_results.append(outlier)
    
    return benchmark_results

def main():
    """Main demonstration of the Benchmark Validation System."""
    logger.info("Benchmark Validation System Demonstration")
    
    # Create the framework
    framework = ComprehensiveBenchmarkValidation()
    
    # Create sample benchmark results
    logger.info("Creating sample benchmark results")
    benchmark_results = create_sample_benchmark_results(num_results=5, add_outlier=True)
    
    # Validate benchmark results
    logger.info("Validating benchmark results")
    validation_results = framework.validate_batch(
        benchmark_results=benchmark_results,
        validation_level=ValidationLevel.STANDARD,
        detect_outliers=True
    )
    
    # Print validation summary
    print("\n--- Validation Results ---")
    for result in validation_results:
        outlier_info = ""
        if "outlier_detection" in result.validation_metrics:
            outlier_metrics = result.validation_metrics["outlier_detection"]
            if outlier_metrics.get("is_outlier", False):
                outlier_info = " (OUTLIER)"
        
        print(f"Benchmark {result.benchmark_result.result_id}: {result.status.name}{outlier_info}")
        print(f"  Confidence score: {result.confidence_score:.2f}")
        
        if result.issues:
            print("  Issues:")
            for issue in result.issues:
                print(f"    {issue['type'].upper()}: {issue['message']}")
    
    # Validate reproducibility
    logger.info("Validating reproducibility")
    # Remove outlier for reproducibility validation
    regular_results = [r for r in benchmark_results if r.result_id != "benchmark-outlier"]
    reproducibility_result = framework.validate_reproducibility(
        benchmark_results=regular_results,
        validation_level=ValidationLevel.STANDARD
    )
    
    # Print reproducibility results
    print("\n--- Reproducibility Results ---")
    repro_metrics = reproducibility_result.validation_metrics["reproducibility"]
    print(f"Status: {reproducibility_result.status.name}")
    print(f"Reproducibility score: {repro_metrics['reproducibility_score']:.2f}")
    print(f"Sample size: {repro_metrics['sample_size']}")
    
    if "metrics" in repro_metrics:
        print("Metrics:")
        for metric, stats in repro_metrics["metrics"].items():
            print(f"  {metric}:")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    Std deviation: {stats['std_deviation']:.2f}")
            print(f"    CV: {stats['coefficient_of_variation']:.2f}%")
            print(f"    Is reproducible: {stats['is_reproducible']}")
    
    # Certify benchmark
    logger.info("Certifying benchmark")
    certification = framework.certify_benchmark(
        benchmark_result=regular_results[0],  # Use first result for certification
        validation_results=[reproducibility_result] + [vr for vr in validation_results 
                                                    if vr.benchmark_result.result_id != "benchmark-outlier"],
        certification_level="auto"  # Determine highest possible level
    )
    
    # Print certification details
    print("\n--- Certification Results ---")
    print(f"Certification level: {certification['certification_level']}")
    print(f"Certification ID: {certification['certification_id']}")
    print(f"Certification authority: {certification['certification_authority']}")
    print(f"Certification timestamp: {certification['certification_timestamp']}")
    
    # Track stability
    logger.info("Tracking benchmark stability")
    stability_analysis = framework.track_benchmark_stability(
        benchmark_results=regular_results,
        metric="average_latency_ms",
        time_window_days=30
    )
    
    # Print stability analysis
    print("\n--- Stability Analysis ---")
    print(f"Overall stability score: {stability_analysis['overall_stability_score']:.2f}")
    print("Model-hardware combinations:")
    
    for key, data in stability_analysis["model_hardware_combinations"].items():
        model_id = data["model_id"]
        hardware_id = data["hardware_id"]
        print(f"\nModel {model_id} on hardware {hardware_id}:")
        print(f"  Stability score: {data['stability_score']:.2f}")
        print(f"  Mean value: {data['mean_value']:.2f}")
        print(f"  Std deviation: {data['std_dev']:.2f}")
        print(f"  Coefficient of variation: {data['coefficient_of_variation']:.2f}%")
    
    # Detect data quality issues
    logger.info("Detecting data quality issues")
    quality_issues = framework.detect_data_quality_issues(
        benchmark_results=benchmark_results
    )
    
    # Print data quality issues
    print("\n--- Data Quality Issues ---")
    for issue_type, issues in quality_issues.items():
        if issues:
            print(f"\n{issue_type.replace('_', ' ').title()} ({len(issues)} issues):")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue.get('issue', 'Issue not specified')}")
                if "result_id" in issue:
                    print(f"     Result ID: {issue['result_id']}")
                if "result_ids" in issue:
                    print(f"     Result IDs: {', '.join(issue['result_ids'])}")
                if "values" in issue:
                    print(f"     Values: {issue['values']}")
    
    # Generate a report
    logger.info("Generating validation report")
    report_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "validation_report.html")
    
    report = framework.generate_report(
        validation_results=validation_results + [reproducibility_result],
        report_format="html",
        include_visualizations=True,
        output_path=report_path
    )
    
    print(f"\nValidation report generated at: {report_path}")
    print("\nDemonstration complete!")

if __name__ == "__main__":
    main()