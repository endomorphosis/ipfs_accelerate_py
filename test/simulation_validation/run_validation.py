#!/usr/bin/env python3
"""
Simulation Accuracy Validation Framework - Command Line Interface

This script provides a command-line interface for the simulation validation framework.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simulation_validation_cli")

# Add parent directory to path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # Import database API
    from data.duckdb.core.benchmark_db_api import BenchmarkDBAPI
    HAS_DB_API = True
except ImportError:
    logger.warning("DuckDB API not available. Some features may be limited.")
    HAS_DB_API = False

# Import simulation validation framework
from simulation_validation.core.validation_framework import SimulationValidator
from simulation_validation.analysis.statistical_analysis import StatisticalAnalyzer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulation Accuracy Validation Framework CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate simulation accuracy")
    validate_parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                              help="Path to the DuckDB database")
    validate_parser.add_argument("--output-dir", default="./simulation_validation_results",
                              help="Directory to save validation results")
    validate_parser.add_argument("--output-file",
                              help="Path to save validation results (defaults to output_dir/simulation_validation_YYYYMMDD_HHMMSS.json)")
    validate_parser.add_argument("--hardware-types", nargs="+",
                              help="List of hardware types to include")
    validate_parser.add_argument("--model-types", nargs="+",
                              help="List of model types to include")
    validate_parser.add_argument("--time-range", type=int, default=90,
                              help="Time range in days to include")
    validate_parser.add_argument("--metrics", nargs="+",
                              help="List of metrics to validate")
    validate_parser.add_argument("--confidence-threshold", type=float, default=0.9,
                              help="Threshold for overall simulation confidence")
    validate_parser.add_argument("--match-criteria", nargs="+",
                              default=["hardware_type", "model_type", "model_name", "batch_size", "precision"],
                              help="Criteria for matching simulated and real results")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze validation results")
    analyze_parser.add_argument("--input-file", required=True,
                               help="Path to the validation results file")
    analyze_parser.add_argument("--output-file",
                               help="Path to save analysis results (defaults to input_file_analysis.json)")
    analyze_parser.add_argument("--include-drift", action="store_true",
                               help="Include drift analysis")
    analyze_parser.add_argument("--db-path",
                               help="Path to the DuckDB database (required for drift analysis)")
    analyze_parser.add_argument("--time-window", type=int, default=90,
                               help="Time window in days for drift analysis")
    analyze_parser.add_argument("--by-hardware", action="store_true",
                               help="Analyze by hardware type")
    analyze_parser.add_argument("--by-model", action="store_true",
                               help="Analyze by model type")
    analyze_parser.add_argument("--by-metric", action="store_true",
                               help="Analyze by metric")
    analyze_parser.add_argument("--worst-configs", action="store_true",
                               help="Identify worst-performing configurations")
    analyze_parser.add_argument("--calibration-candidates", action="store_true",
                               help="Identify calibration candidates")
    analyze_parser.add_argument("--all", action="store_true",
                               help="Run all analyses")
    
    # Validate and analyze command
    full_parser = subparsers.add_parser("full", help="Validate and analyze in one step")
    full_parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                          help="Path to the DuckDB database")
    full_parser.add_argument("--output-dir", default="./simulation_validation_results",
                          help="Directory to save validation and analysis results")
    full_parser.add_argument("--validation-file",
                          help="Path to save validation results (defaults to output_dir/simulation_validation_YYYYMMDD_HHMMSS.json)")
    full_parser.add_argument("--analysis-file",
                          help="Path to save analysis results (defaults to output_dir/simulation_analysis_YYYYMMDD_HHMMSS.json)")
    full_parser.add_argument("--hardware-types", nargs="+",
                          help="List of hardware types to include")
    full_parser.add_argument("--model-types", nargs="+",
                          help="List of model types to include")
    full_parser.add_argument("--time-range", type=int, default=90,
                          help="Time range in days to include")
    full_parser.add_argument("--metrics", nargs="+",
                          help="List of metrics to validate")
    full_parser.add_argument("--confidence-threshold", type=float, default=0.9,
                          help="Threshold for overall simulation confidence")
    full_parser.add_argument("--match-criteria", nargs="+",
                          default=["hardware_type", "model_type", "model_name", "batch_size", "precision"],
                          help="Criteria for matching simulated and real results")
    full_parser.add_argument("--include-drift", action="store_true",
                          help="Include drift analysis")
    full_parser.add_argument("--all-analyses", action="store_true",
                          help="Run all analyses")
    
    return parser.parse_args()


def validate_simulation(args):
    """
    Validate simulation accuracy.
    
    Args:
        args: Command-line arguments
        
    Returns:
        0 if successful, non-zero otherwise
    """
    if not HAS_DB_API:
        logger.error("DuckDB API is required for validation.")
        return 1
    
    try:
        # Initialize database API
        db_api = BenchmarkDBAPI(db_path=args.db_path)
        
        # Initialize validator
        validator = SimulationValidator(
            db_api=db_api,
            metrics=args.metrics,
            output_dir=args.output_dir,
            confidence_threshold=args.confidence_threshold
        )
        
        # Run validation
        logger.info("Running validation...")
        result = validator.run_validation(
            hardware_types=args.hardware_types,
            model_types=args.model_types,
            time_range=args.time_range,
            match_criteria=args.match_criteria,
            output_file=args.output_file
        )
        
        # Print summary
        if "error" in result:
            logger.error(f"Validation error: {result['error']}")
            return 1
            
        logger.info("Validation Summary:")
        logger.info(f"Overall Accuracy: {result.get('overall_accuracy', 0.0):.4f}")
        logger.info(f"Configurations Validated: {result.get('configurations_validated', 0)}")
        logger.info(f"Configurations Passed: {result.get('configurations_passed', 0)}")
        logger.info(f"Pass Rate: {result.get('pass_rate', 0.0):.2%}")
        
        # Print metric accuracies
        logger.info("Metric Accuracies:")
        for metric, accuracy in result.get('metric_accuracies', {}).items():
            logger.info(f"  {metric}: {accuracy:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        return 1


def analyze_validation_results(args):
    """
    Analyze validation results.
    
    Args:
        args: Command-line arguments
        
    Returns:
        0 if successful, non-zero otherwise
    """
    try:
        # Initialize analyzer
        analyzer = StatisticalAnalyzer()
        
        # Load validation results
        logger.info(f"Loading validation results from {args.input_file}...")
        analyzer.load_validation_results(args.input_file)
        
        # Convert to DataFrame for analysis
        df = analyzer.convert_to_dataframe()
        
        results = {}
        
        # Determine which analyses to run
        run_all = args.all
        run_hardware = args.by_hardware or run_all
        run_model = args.by_model or run_all
        run_metric = args.by_metric or run_all
        run_worst = args.worst_configs or run_all
        run_calibration = args.calibration_candidates or run_all
        run_drift = args.include_drift or run_all
        
        # Run selected analyses
        if run_hardware:
            logger.info("Analyzing by hardware type...")
            results["by_hardware_type"] = analyzer.analyze_by_hardware_type(df)
            
        if run_model:
            logger.info("Analyzing by model type...")
            results["by_model_type"] = analyzer.analyze_by_model_type(df)
            
        if run_metric:
            logger.info("Analyzing by metric...")
            results["by_metric"] = analyzer.analyze_by_metric(df)
            
        if run_worst:
            logger.info("Identifying worst-performing configurations...")
            results["worst_configurations"] = analyzer.analyze_worst_configurations(df)
            
        if run_calibration:
            logger.info("Identifying calibration candidates...")
            results["calibration_candidates"] = analyzer.analyze_calibration_candidates(df)
            
        if run_drift:
            if not HAS_DB_API or not args.db_path:
                logger.warning("DuckDB API is required for drift analysis. Skipping...")
            else:
                logger.info("Analyzing drift over time...")
                db_api = BenchmarkDBAPI(db_path=args.db_path)
                results["drift_over_time"] = analyzer.analyze_drift_over_time(
                    time_window=args.time_window,
                    db_api=db_api
                )
        
        # Save results
        output_file = args.output_file
        if not output_file:
            output_file = args.input_file.replace(".json", "_analysis.json")
            if output_file == args.input_file:  # No replacement occurred
                output_file = args.input_file + "_analysis.json"
        
        logger.info(f"Saving analysis results to {output_file}...")
        analyzer.analysis_results = results
        analyzer.save_analysis_results(output_file)
        
        # Print summary
        logger.info("Analysis Summary:")
        
        if "by_hardware_type" in results:
            hw_count = len(results["by_hardware_type"])
            logger.info(f"Hardware Types Analyzed: {hw_count}")
            
        if "by_model_type" in results:
            model_count = len(results["by_model_type"])
            logger.info(f"Model Types Analyzed: {model_count}")
            
        if "by_metric" in results:
            metric_count = len(results["by_metric"])
            logger.info(f"Metrics Analyzed: {metric_count}")
            
        if "worst_configurations" in results:
            worst_count = len(results["worst_configurations"])
            logger.info(f"Worst Configurations Identified: {worst_count}")
            
        if "calibration_candidates" in results:
            cal_count = len(results["calibration_candidates"])
            logger.info(f"Calibration Candidates Identified: {cal_count}")
            
        if "drift_over_time" in results:
            drift_detected = results["drift_over_time"].get("drift_detected", False)
            logger.info(f"Drift Detected: {drift_detected}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return 1


def validate_and_analyze(args):
    """
    Validate and analyze in one step.
    
    Args:
        args: Command-line arguments
        
    Returns:
        0 if successful, non-zero otherwise
    """
    if not HAS_DB_API:
        logger.error("DuckDB API is required for validation and analysis.")
        return 1
    
    try:
        # Initialize database API
        db_api = BenchmarkDBAPI(db_path=args.db_path)
        
        # Initialize validator
        validator = SimulationValidator(
            db_api=db_api,
            metrics=args.metrics,
            output_dir=args.output_dir,
            confidence_threshold=args.confidence_threshold
        )
        
        # Run validation
        logger.info("Running validation...")
        result = validator.run_validation(
            hardware_types=args.hardware_types,
            model_types=args.model_types,
            time_range=args.time_range,
            match_criteria=args.match_criteria,
            output_file=args.validation_file
        )
        
        if "error" in result:
            logger.error(f"Validation error: {result['error']}")
            return 1
            
        # Print validation summary
        logger.info("Validation Summary:")
        logger.info(f"Overall Accuracy: {result.get('overall_accuracy', 0.0):.4f}")
        logger.info(f"Configurations Validated: {result.get('configurations_validated', 0)}")
        logger.info(f"Configurations Passed: {result.get('configurations_passed', 0)}")
        logger.info(f"Pass Rate: {result.get('pass_rate', 0.0):.2%}")
        
        # Get validation results file path
        validation_file = args.validation_file
        if not validation_file:
            validation_file = os.path.join(
                validator.output_dir,
                f"simulation_validation_{validator.validation_results['validation_timestamp'].replace(':', '_').replace('-', '_')}.json"
            )
        
        # Initialize analyzer
        analyzer = StatisticalAnalyzer()
        
        # Load validation results
        logger.info(f"Loading validation results for analysis...")
        analyzer.load_validation_results(validation_file)
        
        # Run comprehensive analysis
        logger.info("Running comprehensive analysis...")
        analysis_result = analyzer.run_comprehensive_analysis(
            include_drift=args.include_drift,
            db_api=db_api if args.include_drift else None,
            time_window=args.time_range
        )
        
        # Save analysis results
        analysis_file = args.analysis_file
        if not analysis_file:
            validation_basename = os.path.basename(validation_file)
            analysis_basename = validation_basename.replace("validation", "analysis")
            analysis_file = os.path.join(args.output_dir, analysis_basename)
        
        logger.info(f"Saving analysis results to {analysis_file}...")
        analyzer.save_analysis_results(analysis_file)
        
        # Print analysis summary
        logger.info("Analysis Summary:")
        
        if "by_hardware_type" in analysis_result:
            hw_count = len(analysis_result["by_hardware_type"])
            logger.info(f"Hardware Types Analyzed: {hw_count}")
            
        if "by_model_type" in analysis_result:
            model_count = len(analysis_result["by_model_type"])
            logger.info(f"Model Types Analyzed: {model_count}")
            
        if "by_metric" in analysis_result:
            metric_count = len(analysis_result["by_metric"])
            logger.info(f"Metrics Analyzed: {metric_count}")
            
        if "worst_configurations" in analysis_result:
            worst_count = len(analysis_result["worst_configurations"])
            logger.info(f"Worst Configurations Identified: {worst_count}")
            
        if "calibration_candidates" in analysis_result:
            cal_count = len(analysis_result["calibration_candidates"])
            logger.info(f"Calibration Candidates Identified: {cal_count}")
            
        if "drift_over_time" in analysis_result:
            drift_detected = analysis_result["drift_over_time"].get("drift_detected", False)
            logger.info(f"Drift Detected: {drift_detected}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during validation and analysis: {e}")
        return 1


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "validate":
        return validate_simulation(args)
    elif args.command == "analyze":
        return analyze_validation_results(args)
    elif args.command == "full":
        return validate_and_analyze(args)
    else:
        logger.error("No command specified. Use --help for usage information.")
        return 1


if __name__ == "__main__":
    sys.exit(main())