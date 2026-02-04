#!/usr/bin/env python3
"""
Command-line interface for the Benchmark Validation System.

This module provides a command-line interface for running benchmark validation,
reproducibility testing, certification, and reporting.
"""

import os
import sys
import argparse
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark_validation_cli")

from data.duckdb.benchmark_validation.core.base import (
    ValidationLevel,
    BenchmarkType,
    ValidationStatus,
    BenchmarkResult,
    ValidationResult
)
from data.duckdb.benchmark_validation.framework import ComprehensiveBenchmarkValidation, get_validation_framework

def load_benchmark_result(input_path: str) -> BenchmarkResult:
    """
    Load a benchmark result from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        BenchmarkResult object
    """
    logger.info(f"Loading benchmark result from {input_path}")
    
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Convert benchmark_type string to enum
        if isinstance(data.get("benchmark_type"), str):
            data["benchmark_type"] = BenchmarkType[data["benchmark_type"]]
        
        # Convert timestamp string to datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.datetime.fromisoformat(data["timestamp"])
        
        return BenchmarkResult.from_dict(data)
        
    except Exception as e:
        logger.error(f"Error loading benchmark result: {e}")
        raise

def load_validation_result(input_path: str) -> ValidationResult:
    """
    Load a validation result from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        ValidationResult object
    """
    logger.info(f"Loading validation result from {input_path}")
    
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        return ValidationResult.from_dict(data)
        
    except Exception as e:
        logger.error(f"Error loading validation result: {e}")
        raise

def load_batch(input_path: str) -> List[BenchmarkResult]:
    """
    Load a batch of benchmark results from a directory or multiple files.
    
    Args:
        input_path: Path to directory or file pattern
        
    Returns:
        List of BenchmarkResult objects
    """
    logger.info(f"Loading batch of benchmark results from {input_path}")
    
    try:
        # Check if input_path is a directory
        if os.path.isdir(input_path):
            files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.json')]
        else:
            # Assume input_path is a file pattern
            import glob
            files = glob.glob(input_path)
        
        # Load benchmark results
        benchmark_results = []
        for file_path in files:
            try:
                benchmark_result = load_benchmark_result(file_path)
                benchmark_results.append(benchmark_result)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Error loading batch: {e}")
        raise

def validate_cmd(args):
    """
    Validate benchmark results.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Running benchmark validation")
    
    # Create validation framework
    framework = get_validation_framework(args.config)
    
    # Load benchmark results
    if os.path.isdir(args.input) or '*' in args.input:
        # Batch validation
        benchmark_results = load_batch(args.input)
        logger.info(f"Loaded {len(benchmark_results)} benchmark results")
        
        # Validate batch
        validation_level = ValidationLevel[args.level.upper()]
        validation_results = framework.validate_batch(
            benchmark_results=benchmark_results,
            validation_level=validation_level,
            detect_outliers=args.detect_outliers
        )
        
        # Print summary
        print(f"Validated {len(validation_results)} benchmark results")
        status_counts = {}
        for result in validation_results:
            status = result.status.name
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        # Save results if requested
        if args.output:
            logger.info(f"Saving validation results to {args.output}")
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            
            output_data = [result.to_dict() for result in validation_results]
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
    else:
        # Single validation
        benchmark_result = load_benchmark_result(args.input)
        
        # Validate
        validation_level = ValidationLevel[args.level.upper()]
        validation_result = framework.validate(
            benchmark_result=benchmark_result,
            validation_level=validation_level
        )
        
        # Print result
        print(f"Validation status: {validation_result.status.name}")
        print(f"Confidence score: {validation_result.confidence_score:.2f}")
        
        if validation_result.issues:
            print("Issues:")
            for issue in validation_result.issues:
                print(f"  {issue['type']}: {issue['message']}")
        
        if validation_result.recommendations:
            print("Recommendations:")
            for recommendation in validation_result.recommendations:
                print(f"  - {recommendation}")
        
        # Save result if requested
        if args.output:
            logger.info(f"Saving validation result to {args.output}")
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            
            with open(args.output, 'w') as f:
                json.dump(validation_result.to_dict(), f, indent=2)

def reproducibility_cmd(args):
    """
    Validate reproducibility of benchmark results.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Running reproducibility validation")
    
    # Create validation framework
    framework = get_validation_framework(args.config)
    
    # Load benchmark results
    benchmark_results = load_batch(args.input)
    logger.info(f"Loaded {len(benchmark_results)} benchmark results")
    
    # Group by model and hardware
    grouped_results = {}
    for result in benchmark_results:
        key = f"{result.model_id}_{result.hardware_id}"
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Print groups
    print(f"Found {len(grouped_results)} groups of benchmark results")
    for key, results in grouped_results.items():
        print(f"  Group {key}: {len(results)} results")
    
    # Validate reproducibility for each group
    all_results = []
    for key, results in grouped_results.items():
        if len(results) < 3:
            print(f"  Skipping group {key}: not enough results (have {len(results)}, need at least 3)")
            continue
        
        # Validate reproducibility
        validation_level = ValidationLevel[args.level.upper()]
        reproducibility_result = framework.validate_reproducibility(
            benchmark_results=results,
            validation_level=validation_level
        )
        
        # Print result
        model_id = results[0].model_id
        hardware_id = results[0].hardware_id
        repro_metrics = reproducibility_result.validation_metrics["reproducibility"]
        
        print(f"\nReproducibility for model {model_id} on hardware {hardware_id}:")
        print(f"  Status: {reproducibility_result.status.name}")
        print(f"  Score: {repro_metrics['reproducibility_score']:.2f}")
        print(f"  Sample size: {repro_metrics['sample_size']}")
        
        if "metrics" in repro_metrics:
            print("  Metrics:")
            for metric, stats in repro_metrics["metrics"].items():
                print(f"    {metric}:")
                print(f"      Mean: {stats['mean']:.2f}")
                print(f"      Std deviation: {stats['std_deviation']:.2f}")
                print(f"      CV: {stats['coefficient_of_variation']:.2f}%")
                print(f"      Is reproducible: {stats['is_reproducible']}")
        
        all_results.append(reproducibility_result)
    
    # Save results if requested
    if args.output and all_results:
        logger.info(f"Saving reproducibility results to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        output_data = [result.to_dict() for result in all_results]
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

def certify_cmd(args):
    """
    Certify benchmark results.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Running benchmark certification")
    
    # Create validation framework
    framework = get_validation_framework(args.config)
    
    # Load benchmark result
    benchmark_result = load_benchmark_result(args.input)
    
    # Load validation results if provided
    validation_results = []
    if args.validation_results:
        if os.path.isdir(args.validation_results) or '*' in args.validation_results:
            # Load multiple validation results
            import glob
            files = glob.glob(args.validation_results) if '*' in args.validation_results else [
                os.path.join(args.validation_results, f) 
                for f in os.listdir(args.validation_results) 
                if f.endswith('.json')
            ]
            
            for file_path in files:
                try:
                    validation_result = load_validation_result(file_path)
                    validation_results.append(validation_result)
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
        else:
            # Load single validation result
            validation_result = load_validation_result(args.validation_results)
            validation_results.append(validation_result)
    
    # Certify benchmark
    certification = framework.certify_benchmark(
        benchmark_result=benchmark_result,
        validation_results=validation_results,
        certification_level=args.level
    )
    
    if not certification:
        print("Certification failed. Benchmark does not meet requirements.")
        if "missing_requirements" in certification:
            print("Missing requirements:")
            for req in certification["missing_requirements"]:
                print(f"  {req['message']}")
                print(f"  Recommendation: {req['recommendation']}")
        return
    
    # Print certification
    print(f"Certification ID: {certification['certification_id']}")
    print(f"Certification level: {certification['certification_level']}")
    print(f"Certification authority: {certification['certification_authority']}")
    print(f"Certification timestamp: {certification['certification_timestamp']}")
    
    if "certification_requirements" in certification:
        print("Certification requirements:")
        for req, value in certification["certification_requirements"].items():
            print(f"  {req}: {value}")
    
    # Save certification if requested
    if args.output:
        logger.info(f"Saving certification to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        with open(args.output, 'w') as f:
            json.dump(certification, f, indent=2)

def report_cmd(args):
    """
    Generate validation reports.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Generating validation report")
    
    # Create validation framework
    framework = get_validation_framework(args.config)
    
    # Load validation results
    validation_results = []
    if os.path.isdir(args.input) or '*' in args.input:
        # Load multiple validation results
        import glob
        files = glob.glob(args.input) if '*' in args.input else [
            os.path.join(args.input, f) 
            for f in os.listdir(args.input) 
            if f.endswith('.json')
        ]
        
        for file_path in files:
            try:
                validation_result = load_validation_result(file_path)
                validation_results.append(validation_result)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
    else:
        # Load single validation result
        validation_result = load_validation_result(args.input)
        validation_results.append(validation_result)
    
    logger.info(f"Loaded {len(validation_results)} validation results")
    
    # Generate report
    report = framework.generate_report(
        validation_results=validation_results,
        report_format=args.format,
        include_visualizations=args.visualizations,
        output_path=args.output
    )
    
    if args.output:
        print(f"Report generated at: {report}")
    else:
        print(report)

def stability_cmd(args):
    """
    Track benchmark stability.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Tracking benchmark stability")
    
    # Create validation framework
    framework = get_validation_framework(args.config)
    
    # Load benchmark results
    benchmark_results = load_batch(args.input)
    logger.info(f"Loaded {len(benchmark_results)} benchmark results")
    
    # Track stability
    stability_analysis = framework.track_benchmark_stability(
        benchmark_results=benchmark_results,
        metric=args.metric,
        time_window_days=args.time_window
    )
    
    # Print stability analysis
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
        print(f"  Number of results: {data['num_results']}")
        if "date_range" in data:
            print(f"  Date range: {data['date_range']['start']} to {data['date_range']['end']}")
    
    # Save stability analysis if requested
    if args.output:
        logger.info(f"Saving stability analysis to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        with open(args.output, 'w') as f:
            json.dump(stability_analysis, f, indent=2)

def quality_cmd(args):
    """
    Analyze data quality.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Analyzing data quality")
    
    # Create validation framework
    framework = get_validation_framework(args.config)
    
    # Load benchmark results
    benchmark_results = load_batch(args.input)
    logger.info(f"Loaded {len(benchmark_results)} benchmark results")
    
    # Analyze data quality
    quality_issues = framework.detect_data_quality_issues(
        benchmark_results=benchmark_results
    )
    
    # Print quality issues
    print("Data quality issues:")
    for issue_type, issues in quality_issues.items():
        print(f"\n{issue_type.replace('_', ' ').title()} ({len(issues)} issues):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue.get('issue', 'Issue not specified')}")
            if "result_id" in issue:
                print(f"     Result ID: {issue['result_id']}")
            if "result_ids" in issue:
                print(f"     Result IDs: {', '.join(issue['result_ids'])}")
            if "values" in issue:
                print(f"     Values: {issue['values']}")
    
    # Save quality analysis if requested
    if args.output:
        logger.info(f"Saving quality analysis to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        with open(args.output, 'w') as f:
            json.dump(quality_issues, f, indent=2)

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Benchmark Validation System CLI")
    parser.add_argument('--config', type=str, help="Path to configuration file")
    
    subparsers = parser.add_subparsers(dest='command', help="Command to run")
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help="Validate benchmark results")
    validate_parser.add_argument('--input', type=str, required=True, help="Path to benchmark result JSON or directory")
    validate_parser.add_argument('--level', type=str, default="STANDARD", help="Validation level (MINIMAL, STANDARD, STRICT, CERTIFICATION)")
    validate_parser.add_argument('--detect-outliers', action='store_true', help="Detect outliers in batch validation")
    validate_parser.add_argument('--output', type=str, help="Path to save validation result")
    validate_parser.set_defaults(func=validate_cmd)
    
    # Reproducibility command
    reproducibility_parser = subparsers.add_parser('reproducibility', help="Validate reproducibility of benchmark results")
    reproducibility_parser.add_argument('--input', type=str, required=True, help="Path to directory of benchmark result JSONs")
    reproducibility_parser.add_argument('--level', type=str, default="STANDARD", help="Validation level (MINIMAL, STANDARD, STRICT, CERTIFICATION)")
    reproducibility_parser.add_argument('--output', type=str, help="Path to save reproducibility result")
    reproducibility_parser.set_defaults(func=reproducibility_cmd)
    
    # Certify command
    certify_parser = subparsers.add_parser('certify', help="Certify benchmark results")
    certify_parser.add_argument('--input', type=str, required=True, help="Path to benchmark result JSON")
    certify_parser.add_argument('--validation-results', type=str, help="Path to validation result JSON or directory")
    certify_parser.add_argument('--level', type=str, default="auto", help="Certification level (basic, standard, advanced, gold, auto)")
    certify_parser.add_argument('--output', type=str, help="Path to save certification")
    certify_parser.set_defaults(func=certify_cmd)
    
    # Report command
    report_parser = subparsers.add_parser('report', help="Generate validation reports")
    report_parser.add_argument('--input', type=str, required=True, help="Path to validation result JSON or directory")
    report_parser.add_argument('--format', type=str, default="html", help="Report format (html, markdown, json, text)")
    report_parser.add_argument('--visualizations', action='store_true', help="Include visualizations in report")
    report_parser.add_argument('--output', type=str, help="Path to save report")
    report_parser.set_defaults(func=report_cmd)
    
    # Stability command
    stability_parser = subparsers.add_parser('stability', help="Track benchmark stability")
    stability_parser.add_argument('--input', type=str, required=True, help="Path to benchmark result JSON or directory")
    stability_parser.add_argument('--metric', type=str, default="average_latency_ms", help="Metric to track stability for")
    stability_parser.add_argument('--time-window', type=int, default=30, help="Time window in days")
    stability_parser.add_argument('--output', type=str, help="Path to save stability analysis")
    stability_parser.set_defaults(func=stability_cmd)
    
    # Quality command
    quality_parser = subparsers.add_parser('quality', help="Analyze data quality")
    quality_parser.add_argument('--input', type=str, required=True, help="Path to benchmark result JSON or directory")
    quality_parser.add_argument('--output', type=str, help="Path to save quality analysis")
    quality_parser.set_defaults(func=quality_cmd)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)

if __name__ == "__main__":
    main()