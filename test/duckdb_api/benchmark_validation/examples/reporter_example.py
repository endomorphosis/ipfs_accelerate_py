#!/usr/bin/env python3
"""
Example script demonstrating the usage of the ValidationReporter component.

This script creates sample validation results and demonstrates how to generate
reports and visualizations using the ValidationReporter.
"""

import os
import sys
import datetime
import random
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

# Import validation components
from duckdb_api.benchmark_validation.core.base import (
    ValidationLevel,
    BenchmarkType,
    ValidationStatus,
    BenchmarkResult,
    ValidationResult
)
from duckdb_api.benchmark_validation.visualization.reporter import ValidationReporterImpl

def create_sample_validation_results(num_results=20):
    """Create sample validation results for demonstration."""
    results = []
    
    # Define some benchmark types, model IDs, and hardware IDs
    benchmark_types = [BenchmarkType.PERFORMANCE, BenchmarkType.COMPATIBILITY, 
                       BenchmarkType.INTEGRATION, BenchmarkType.WEB_PLATFORM]
    model_ids = ["bert-base-uncased", "vit-base-patch16-224", "whisper-tiny", 
                "t5-small", "llama-7b", "resnet50"]
    hardware_ids = ["cpu", "gpu", "webgpu", "webnn", "tpu", "mps"]
    validation_levels = [ValidationLevel.MINIMAL, ValidationLevel.STANDARD, 
                        ValidationLevel.STRICT, ValidationLevel.CERTIFICATION]
    statuses = [ValidationStatus.VALID, ValidationStatus.WARNING, 
               ValidationStatus.INVALID, ValidationStatus.ERROR, ValidationStatus.PENDING]
    
    # Create a variety of results
    for i in range(num_results):
        # Create a benchmark result
        benchmark_result = BenchmarkResult(
            result_id=f"benchmark-{i}",
            benchmark_type=random.choice(benchmark_types),
            model_id=random.choice(model_ids),
            hardware_id=random.choice(hardware_ids),
            metrics={
                "throughput_items_per_second": 100 + random.uniform(0, 200),
                "average_latency_ms": 50 - random.uniform(0, 30),
                "memory_peak_mb": 500 + random.uniform(0, 1000),
                "total_inference_time_ms": 100 + random.uniform(0, 100)
            },
            run_id=i,
            timestamp=datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 30)),
            metadata={
                "batch_size": random.choice([1, 4, 8, 16, 32]),
                "precision": random.choice(["fp32", "fp16", "int8"]),
                "environment": random.choice(["local", "cloud", "ci"])
            }
        )
        
        # Create validation metrics
        validation_metrics = {
            "throughput_score": random.uniform(0.5, 1.0),
            "latency_score": random.uniform(0.5, 1.0),
            "memory_score": random.uniform(0.5, 1.0),
            "overall_score": random.uniform(0.5, 1.0)
        }
        
        # Create issues (with varying probability)
        issues = []
        if random.random() < 0.3:  # 30% chance of having issues
            num_issues = random.randint(1, 3)
            possible_issues = [
                {"description": "Throughput below expected range", "severity": "medium"},
                {"description": "High latency variability", "severity": "high"},
                {"description": "Memory usage above threshold", "severity": "medium"},
                {"description": "Incomplete benchmark run", "severity": "high"},
                {"description": "Unexpected performance regression", "severity": "high"},
                {"description": "Minor throughput fluctuation", "severity": "low"},
                {"description": "Potential system load interference", "severity": "medium"}
            ]
            issues = random.sample(possible_issues, num_issues)
        
        # Create recommendations (if there are issues)
        recommendations = []
        if issues:
            possible_recommendations = [
                "Consider running additional tests to confirm results",
                "Check for system load during benchmark execution",
                "Verify hardware configuration matches expectations",
                "Compare against reference hardware results",
                "Review recent code changes that might impact performance",
                "Ensure test environment is isolated from other processes",
                "Run benchmark with increased sample size"
            ]
            num_recommendations = random.randint(1, 3)
            recommendations = random.sample(possible_recommendations, num_recommendations)
        
        # Determine validation status (weighted towards VALID)
        if issues:
            # If there are issues, weight towards WARNING or INVALID
            status_weights = [0.3, 0.4, 0.2, 0.05, 0.05]  # VALID, WARNING, INVALID, ERROR, PENDING
        else:
            # If no issues, weight towards VALID
            status_weights = [0.8, 0.1, 0.05, 0.0, 0.05]  # VALID, WARNING, INVALID, ERROR, PENDING
        
        status = random.choices(statuses, weights=status_weights, k=1)[0]
        
        # Create validation result
        validation_result = ValidationResult(
            benchmark_result=benchmark_result,
            status=status,
            validation_level=random.choice(validation_levels),
            confidence_score=random.uniform(0.6, 1.0),
            validation_metrics=validation_metrics,
            issues=issues,
            recommendations=recommendations,
            validation_timestamp=datetime.datetime.now(),
            validator_id="example-validator"
        )
        
        results.append(validation_result)
    
    return results

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "../output")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating sample validation results...")
    validation_results = create_sample_validation_results(30)
    print(f"Created {len(validation_results)} sample validation results")
    
    # Create reporter
    print(f"Initializing ValidationReporter...")
    reporter = ValidationReporterImpl({
        "output_directory": output_dir,
        "report_title_template": "Benchmark Validation Report - Example",
        "max_results_per_page": 15
    })
    
    # Generate HTML report
    print(f"Generating HTML report...")
    html_path = os.path.join(output_dir, "validation_report.html")
    reporter.export_report(
        validation_results=validation_results,
        output_path=html_path,
        report_format="html",
        include_visualizations=True
    )
    print(f"HTML report saved to: {html_path}")
    
    # Generate Markdown report
    print(f"Generating Markdown report...")
    md_path = os.path.join(output_dir, "validation_report.md")
    reporter.export_report(
        validation_results=validation_results,
        output_path=md_path,
        report_format="markdown"
    )
    print(f"Markdown report saved to: {md_path}")
    
    # Generate JSON report
    print(f"Generating JSON report...")
    json_path = os.path.join(output_dir, "validation_report.json")
    reporter.export_report(
        validation_results=validation_results,
        output_path=json_path,
        report_format="json"
    )
    print(f"JSON report saved to: {json_path}")
    
    # Create visualizations
    try:
        # Check if visualization libraries are available
        import plotly
        import pandas
        
        print(f"Creating visualizations...")
        
        # Create confidence distribution visualization
        print(f"Generating confidence distribution visualization...")
        confidence_vis_path = os.path.join(output_dir, "confidence_distribution.html")
        reporter.create_visualization(
            validation_results=validation_results,
            visualization_type="confidence_distribution",
            output_path=confidence_vis_path,
            title="Confidence Score Distribution"
        )
        print(f"Confidence distribution visualization saved to: {confidence_vis_path}")
        
        # Try to create a validation heatmap if advanced visualization system is available
        try:
            print(f"Generating validation heatmap...")
            heatmap_path = os.path.join(output_dir, "validation_heatmap.html")
            reporter.create_visualization(
                validation_results=validation_results,
                visualization_type="validation_heatmap",
                output_path=heatmap_path,
                title="Validation Results by Model and Hardware",
                metric="confidence_score"
            )
            print(f"Validation heatmap saved to: {heatmap_path}")
        except Exception as e:
            print(f"Could not create validation heatmap: {e}")
            print(f"Advanced visualization system may not be available")
            
    except ImportError:
        print(f"Skipping visualizations - required libraries not available")
        print(f"Install required packages with: pip install plotly pandas matplotlib")
    
    print("\nExample completed successfully!")
    print(f"All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()