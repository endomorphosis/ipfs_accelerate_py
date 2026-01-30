#!/usr/bin/env python3
"""
Command-line tool for running the Predictive Performance Modeling System with DuckDB integration.

This tool provides a command-line interface for predicting hardware performance,
storing predictions in DuckDB, recording actual measurements, and analyzing prediction accuracy.
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to sys.path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from data.duckdb.predictive_performance.predictor_repository import DuckDBPredictorRepository
    from data.duckdb.predictive_performance.repository_adapter import (
        HardwareModelPredictorDuckDBAdapter,
        ModelPerformancePredictorDuckDBAdapter
    )
    
    # Try to import from the predictive_performance package
    try:
        from predictive_performance.hardware_model_predictor import HardwareModelPredictor
        HARDWARE_MODEL_PREDICTOR_AVAILABLE = True
    except ImportError:
        HARDWARE_MODEL_PREDICTOR_AVAILABLE = False
        logger.warning("HardwareModelPredictor not available, some features will be limited")
    
    DUCKDB_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import DuckDB components: {e}")
    DUCKDB_AVAILABLE = False

def predict_hardware(args):
    """
    Predict optimal hardware for a given model and configuration.
    
    Args:
        args: Command-line arguments
    """
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB integration not available")
        return
    
    try:
        # Initialize repository and adapter
        repository = DuckDBPredictorRepository(db_path=args.database)
        
        if HARDWARE_MODEL_PREDICTOR_AVAILABLE:
            predictor = HardwareModelPredictor(
                benchmark_dir=args.benchmark_dir,
                database_path=args.benchmark_database
            )
        else:
            predictor = None
        
        adapter = HardwareModelPredictorDuckDBAdapter(
            predictor=predictor,
            repository=repository,
            user_id=args.user_id
        )
        
        # Parse available hardware if provided
        available_hardware = args.hardware.split(',') if args.hardware else None
        
        # Call the adapter to predict optimal hardware
        result = adapter.predict_optimal_hardware(
            model_name=args.model,
            model_family=args.family,
            batch_size=args.batch_size,
            sequence_length=args.seq_length,
            mode=args.mode,
            precision=args.precision,
            available_hardware=available_hardware,
            store_recommendation=not args.no_store
        )
        
        # Pretty-print the result
        print(f"\nHardware Recommendation for {args.model}:")
        print(f"  Primary Recommendation: {result.get('primary_recommendation')}")
        print(f"  Fallback Options: {', '.join(result.get('fallback_options', []))}")
        print(f"  Compatible Hardware: {', '.join(result.get('compatible_hardware', []))}")
        print(f"  Model Family: {result.get('model_family')}")
        print(f"  Model Size: {result.get('model_size_category', 'unknown')} ({result.get('model_size', 'unknown')} parameters)")
        print(f"  Explanation: {result.get('explanation')}")
        
        if result.get('recommendation_id'):
            print(f"\nRecommendation stored with ID: {result.get('recommendation_id')}")
        
        # If detailed output is requested, predict performance on recommended hardware
        if args.predict_performance:
            print("\nPredicted Performance:")
            
            performance = adapter.predict_performance(
                model_name=args.model,
                model_family=result.get('model_family', args.family),
                hardware=result.get('primary_recommendation'),
                batch_size=args.batch_size,
                sequence_length=args.seq_length,
                mode=args.mode,
                precision=args.precision,
                store_prediction=not args.no_store
            )
            
            hw = result.get('primary_recommendation')
            if hw in performance.get('predictions', {}):
                pred = performance['predictions'][hw]
                print(f"  Throughput: {pred.get('throughput', 'N/A'):.2f} items/sec")
                print(f"  Latency: {pred.get('latency', 'N/A'):.2f} ms")
                print(f"  Memory Usage: {pred.get('memory_usage', 'N/A'):.2f} MB")
                print(f"  Prediction Source: {pred.get('source', 'N/A')}")
                
                if pred.get('prediction_id'):
                    print(f"  Prediction stored with ID: {pred.get('prediction_id')}")
        
    except Exception as e:
        logger.error(f"Error predicting hardware: {e}")
        import traceback
        logger.error(traceback.format_exc())

def predict_performance(args):
    """
    Predict performance for a model on specified hardware.
    
    Args:
        args: Command-line arguments
    """
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB integration not available")
        return
    
    try:
        # Initialize repository and adapter
        repository = DuckDBPredictorRepository(db_path=args.database)
        
        if HARDWARE_MODEL_PREDICTOR_AVAILABLE:
            predictor = HardwareModelPredictor(
                benchmark_dir=args.benchmark_dir,
                database_path=args.benchmark_database
            )
        else:
            predictor = None
        
        adapter = HardwareModelPredictorDuckDBAdapter(
            predictor=predictor,
            repository=repository,
            user_id=args.user_id
        )
        
        # Parse hardware platforms
        hardware_platforms = args.hardware.split(',')
        
        # Call the adapter to predict performance
        result = adapter.predict_performance(
            model_name=args.model,
            model_family=args.family,
            hardware=hardware_platforms,
            batch_size=args.batch_size,
            sequence_length=args.seq_length,
            mode=args.mode,
            precision=args.precision,
            store_prediction=not args.no_store
        )
        
        # Pretty-print the result
        print(f"\nPerformance Predictions for {args.model} on {args.hardware}:")
        print(f"  Model Family: {result.get('model_family')}")
        print(f"  Batch Size: {result.get('batch_size')}")
        print(f"  Sequence Length: {result.get('sequence_length')}")
        print(f"  Mode: {result.get('mode')}")
        print(f"  Precision: {result.get('precision')}")
        
        print("\nPredicted Metrics by Hardware Platform:")
        for hw, pred in result.get('predictions', {}).items():
            print(f"\n  {hw.upper()}:")
            print(f"    Throughput: {pred.get('throughput', 'N/A'):.2f} items/sec")
            print(f"    Latency: {pred.get('latency', 'N/A'):.2f} ms")
            print(f"    Memory Usage: {pred.get('memory_usage', 'N/A'):.2f} MB")
            print(f"    Prediction Source: {pred.get('source', 'N/A')}")
            
            if pred.get('prediction_id'):
                print(f"    Prediction stored with ID: {pred.get('prediction_id')}")
        
    except Exception as e:
        logger.error(f"Error predicting performance: {e}")
        import traceback
        logger.error(traceback.format_exc())

def record_measurement(args):
    """
    Record an actual performance measurement.
    
    Args:
        args: Command-line arguments
    """
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB integration not available")
        return
    
    try:
        # Initialize repository and adapter
        repository = DuckDBPredictorRepository(db_path=args.database)
        
        adapter = HardwareModelPredictorDuckDBAdapter(
            repository=repository,
            user_id=args.user_id
        )
        
        # Call the adapter to record measurement
        result = adapter.record_actual_performance(
            model_name=args.model,
            model_family=args.family,
            hardware_platform=args.hardware,
            batch_size=args.batch_size,
            sequence_length=args.seq_length,
            precision=args.precision,
            mode=args.mode,
            throughput=args.throughput,
            latency=args.latency,
            memory_usage=args.memory,
            prediction_id=args.prediction_id,
            measurement_source=args.source
        )
        
        # Pretty-print the result
        print(f"\nRecorded Measurement for {args.model} on {args.hardware}:")
        print(f"  Measurement ID: {result.get('measurement_id')}")
        
        # Print comparison with prediction if available
        if result.get('prediction'):
            print("\nComparison with Prediction:")
            print(f"  Prediction ID: {result.get('prediction_id')}")
            
            for error in result.get('errors', []):
                metric = error.get('metric')
                predicted = error.get('predicted_value')
                actual = error.get('actual_value')
                rel_error = error.get('relative_error', 0) * 100  # Convert to percentage
                
                print(f"\n  {metric.capitalize()}:")
                print(f"    Predicted: {predicted:.2f}")
                print(f"    Actual: {actual:.2f}")
                print(f"    Relative Error: {rel_error:.2f}%")
        
    except Exception as e:
        logger.error(f"Error recording measurement: {e}")
        import traceback
        logger.error(traceback.format_exc())

def analyze_predictions(args):
    """
    Analyze prediction accuracy.
    
    Args:
        args: Command-line arguments
    """
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB integration not available")
        return
    
    try:
        # Initialize repository
        repository = DuckDBPredictorRepository(db_path=args.database)
        
        # Parse time range if provided
        start_time = None
        end_time = None
        
        if args.days:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=args.days)
        
        # Get accuracy stats
        stats = repository.get_prediction_accuracy_stats(
            model_name=args.model,
            hardware_platform=args.hardware,
            metric=args.metric,
            start_time=start_time,
            end_time=end_time
        )
        
        # Pretty-print the stats
        print("\nPrediction Accuracy Statistics:")
        
        if not stats:
            print("  No prediction error data found for the specified filters")
            return
        
        # Print overall stats if available
        if 'overall' in stats:
            overall = stats['overall']
            print(f"\nOverall Statistics:")
            print(f"  Total Predictions: {overall.get('count')}")
            print(f"  Metrics Analyzed: {', '.join(overall.get('metrics', []))}")
            print(f"  Overall Mean Relative Error: {overall.get('overall_mean_relative_error', 0) * 100:.2f}%")
        
        # Print stats by metric
        for metric, metric_stats in stats.items():
            if metric == 'overall':
                continue
                
            print(f"\n{metric.capitalize()} Prediction Stats:")
            print(f"  Count: {metric_stats.get('count')}")
            print(f"  Mean Absolute Error: {metric_stats.get('mean_absolute_error'):.2f}")
            print(f"  Mean Relative Error: {metric_stats.get('mean_relative_error', 0) * 100:.2f}%")
            print(f"  Standard Deviation: {metric_stats.get('std_absolute_error', 0):.2f}")
            print(f"  Range: {metric_stats.get('min_absolute_error', 0):.2f} - {metric_stats.get('max_absolute_error', 0):.2f}")
            
            if metric_stats.get('r_squared') is not None:
                print(f"  R²: {metric_stats.get('r_squared'):.4f}")
            
            print(f"  Bias: {metric_stats.get('bias', 0):.2f}")
            print(f"  Mean Predicted: {metric_stats.get('mean_predicted', 0):.2f}")
            print(f"  Mean Actual: {metric_stats.get('mean_actual', 0):.2f}")
        
    except Exception as e:
        logger.error(f"Error analyzing predictions: {e}")
        import traceback
        logger.error(traceback.format_exc())

def list_models(args):
    """
    List stored models in the database.
    
    Args:
        args: Command-line arguments
    """
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB integration not available")
        return
    
    try:
        # Initialize repository
        repository = DuckDBPredictorRepository(db_path=args.database)
        
        # Get models based on filters
        models = repository.get_prediction_models(
            model_type=args.model_type,
            target_metric=args.target_metric,
            hardware_platform=args.hardware,
            model_family=args.family,
            limit=args.limit
        )
        
        # Pretty-print the models
        print(f"\nStored Prediction Models ({len(models)} found):")
        
        if not models:
            print("  No models found for the specified filters")
            return
        
        for model in models:
            print(f"\nModel ID: {model.get('model_id')}")
            print(f"  Type: {model.get('model_type')}")
            print(f"  Target Metric: {model.get('target_metric')}")
            print(f"  Hardware Platform: {model.get('hardware_platform')}")
            print(f"  Model Family: {model.get('model_family')}")
            print(f"  Scores:")
            print(f"    Training: {model.get('training_score'):.4f}")
            print(f"    Validation: {model.get('validation_score'):.4f}")
            print(f"    Test: {model.get('test_score'):.4f}")
            print(f"  Created: {model.get('timestamp')}")
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        import traceback
        logger.error(traceback.format_exc())

def list_recommendations(args):
    """
    List hardware recommendations in the database.
    
    Args:
        args: Command-line arguments
    """
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB integration not available")
        return
    
    try:
        # Initialize repository
        repository = DuckDBPredictorRepository(db_path=args.database)
        
        # Parse time range if provided
        start_time = None
        end_time = None
        
        if args.days:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=args.days)
        
        # Get recommendations based on filters
        recommendations = repository.get_recommendations(
            model_name=args.model,
            model_family=args.family,
            user_id=args.user_id,
            primary_recommendation=args.hardware,
            was_accepted=args.accepted,
            start_time=start_time,
            end_time=end_time,
            limit=args.limit
        )
        
        # Pretty-print the recommendations
        print(f"\nHardware Recommendations ({len(recommendations)} found):")
        
        if not recommendations:
            print("  No recommendations found for the specified filters")
            return
        
        for rec in recommendations:
            print(f"\nRecommendation ID: {rec.get('recommendation_id')}")
            print(f"  Model: {rec.get('model_name')}")
            print(f"  Primary Recommendation: {rec.get('primary_recommendation')}")
            print(f"  Fallback Options: {', '.join(rec.get('fallback_options', []))}")
            print(f"  Configuration:")
            print(f"    Batch Size: {rec.get('batch_size')}")
            print(f"    Mode: {rec.get('mode')}")
            print(f"    Precision: {rec.get('precision')}")
            print(f"  User: {rec.get('user_id')}")
            print(f"  Timestamp: {rec.get('timestamp')}")
            
            if rec.get('was_accepted') is not None:
                status = "Accepted" if rec.get('was_accepted') else "Rejected"
                print(f"  Status: {status}")
                if rec.get('user_feedback'):
                    print(f"  Feedback: {rec.get('user_feedback')}")
        
    except Exception as e:
        logger.error(f"Error listing recommendations: {e}")
        import traceback
        logger.error(traceback.format_exc())

def record_feedback(args):
    """
    Record feedback for a hardware recommendation.
    
    Args:
        args: Command-line arguments
    """
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB integration not available")
        return
    
    try:
        # Initialize repository and adapter
        repository = DuckDBPredictorRepository(db_path=args.database)
        
        adapter = HardwareModelPredictorDuckDBAdapter(
            repository=repository,
            user_id=args.user_id
        )
        
        # Parse acceptance flag
        was_accepted = args.accepted.lower() in ['yes', 'true', 'y', '1']
        
        # Call the adapter to record feedback
        result = adapter.record_recommendation_feedback(
            recommendation_id=args.recommendation_id,
            was_accepted=was_accepted,
            user_feedback=args.feedback
        )
        
        if result:
            print(f"\nFeedback recorded successfully for recommendation {args.recommendation_id}")
        else:
            print(f"\nFailed to record feedback for recommendation {args.recommendation_id}")
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        import traceback
        logger.error(traceback.format_exc())

def generate_sample_data(args):
    """
    Generate sample data for testing.
    
    Args:
        args: Command-line arguments
    """
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB integration not available")
        return
    
    try:
        # Initialize repository
        repository = DuckDBPredictorRepository(db_path=args.database)
        
        # Generate sample data
        repository.generate_sample_data(num_models=args.num_models)
        
        print(f"\nGenerated sample data for {args.num_models} models")
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Predictive Performance Modeling System with DuckDB Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument("--database", type=str, default="predictive_performance.duckdb",
                        help="Path to the DuckDB database")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_results",
                        help="Path to benchmark results directory")
    parser.add_argument("--benchmark-database", type=str, default=None,
                        help="Path to benchmark database")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # predict-hardware command
    predict_hw_parser = subparsers.add_parser("predict-hardware", 
                                             help="Predict optimal hardware for a model")
    predict_hw_parser.add_argument("--model", type=str, required=True,
                                  help="Model name")
    predict_hw_parser.add_argument("--family", type=str,
                                  help="Model family/category")
    predict_hw_parser.add_argument("--batch-size", type=int, default=1,
                                  help="Batch size")
    predict_hw_parser.add_argument("--seq-length", type=int, default=128,
                                  help="Sequence length")
    predict_hw_parser.add_argument("--mode", type=str, choices=["inference", "training"],
                                  default="inference", help="Mode")
    predict_hw_parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"],
                                  default="fp32", help="Precision")
    predict_hw_parser.add_argument("--hardware", type=str,
                                  help="Comma-separated list of available hardware platforms")
    predict_hw_parser.add_argument("--predict-performance", action="store_true",
                                  help="Also predict performance on recommended hardware")
    predict_hw_parser.add_argument("--user-id", type=str, default="cli-user",
                                  help="User ID for tracking recommendations")
    predict_hw_parser.add_argument("--no-store", action="store_true",
                                  help="Don't store prediction in database")
    predict_hw_parser.set_defaults(func=predict_hardware)
    
    # predict-performance command
    predict_perf_parser = subparsers.add_parser("predict-performance", 
                                              help="Predict performance for a model on specified hardware")
    predict_perf_parser.add_argument("--model", type=str, required=True,
                                    help="Model name")
    predict_perf_parser.add_argument("--family", type=str,
                                    help="Model family/category")
    predict_perf_parser.add_argument("--hardware", type=str, required=True,
                                    help="Comma-separated list of hardware platforms")
    predict_perf_parser.add_argument("--batch-size", type=int, default=1,
                                    help="Batch size")
    predict_perf_parser.add_argument("--seq-length", type=int, default=128,
                                    help="Sequence length")
    predict_perf_parser.add_argument("--mode", type=str, choices=["inference", "training"],
                                    default="inference", help="Mode")
    predict_perf_parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"],
                                    default="fp32", help="Precision")
    predict_perf_parser.add_argument("--user-id", type=str, default="cli-user",
                                    help="User ID for tracking predictions")
    predict_perf_parser.add_argument("--no-store", action="store_true",
                                    help="Don't store prediction in database")
    predict_perf_parser.set_defaults(func=predict_performance)
    
    # record-measurement command
    record_meas_parser = subparsers.add_parser("record-measurement", 
                                             help="Record an actual performance measurement")
    record_meas_parser.add_argument("--model", type=str, required=True,
                                   help="Model name")
    record_meas_parser.add_argument("--family", type=str,
                                   help="Model family/category")
    record_meas_parser.add_argument("--hardware", type=str, required=True,
                                   help="Hardware platform")
    record_meas_parser.add_argument("--batch-size", type=int, default=1,
                                   help="Batch size")
    record_meas_parser.add_argument("--seq-length", type=int, default=128,
                                   help="Sequence length")
    record_meas_parser.add_argument("--mode", type=str, choices=["inference", "training"],
                                   default="inference", help="Mode")
    record_meas_parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"],
                                   default="fp32", help="Precision")
    record_meas_parser.add_argument("--throughput", type=float,
                                   help="Throughput measurement")
    record_meas_parser.add_argument("--latency", type=float,
                                   help="Latency measurement")
    record_meas_parser.add_argument("--memory", type=float,
                                   help="Memory usage measurement")
    record_meas_parser.add_argument("--prediction-id", type=str,
                                   help="ID of a previous prediction to compare with")
    record_meas_parser.add_argument("--source", type=str, default="cli",
                                   help="Source of the measurement")
    record_meas_parser.add_argument("--user-id", type=str, default="cli-user",
                                   help="User ID for tracking measurements")
    record_meas_parser.set_defaults(func=record_measurement)
    
    # analyze-predictions command
    analyze_parser = subparsers.add_parser("analyze-predictions", 
                                         help="Analyze prediction accuracy")
    analyze_parser.add_argument("--model", type=str,
                               help="Filter by model name")
    analyze_parser.add_argument("--hardware", type=str,
                               help="Filter by hardware platform")
    analyze_parser.add_argument("--metric", type=str, choices=["throughput", "latency", "memory_usage"],
                               help="Filter by metric")
    analyze_parser.add_argument("--days", type=int,
                               help="Analyze predictions from the last N days")
    analyze_parser.set_defaults(func=analyze_predictions)
    
    # list-models command
    list_models_parser = subparsers.add_parser("list-models", 
                                             help="List stored prediction models")
    list_models_parser.add_argument("--model-type", type=str,
                                   help="Filter by model type")
    list_models_parser.add_argument("--target-metric", type=str,
                                   help="Filter by target metric")
    list_models_parser.add_argument("--hardware", type=str,
                                   help="Filter by hardware platform")
    list_models_parser.add_argument("--family", type=str,
                                   help="Filter by model family")
    list_models_parser.add_argument("--limit", type=int, default=10,
                                   help="Maximum number of models to list")
    list_models_parser.set_defaults(func=list_models)
    
    # list-recommendations command
    list_rec_parser = subparsers.add_parser("list-recommendations", 
                                          help="List hardware recommendations")
    list_rec_parser.add_argument("--model", type=str,
                                help="Filter by model name")
    list_rec_parser.add_argument("--family", type=str,
                                help="Filter by model family")
    list_rec_parser.add_argument("--hardware", type=str,
                                help="Filter by primary recommendation")
    list_rec_parser.add_argument("--user-id", type=str,
                                help="Filter by user ID")
    list_rec_parser.add_argument("--accepted", type=bool,
                                help="Filter by acceptance status")
    list_rec_parser.add_argument("--days", type=int,
                                help="List recommendations from the last N days")
    list_rec_parser.add_argument("--limit", type=int, default=10,
                                help="Maximum number of recommendations to list")
    list_rec_parser.set_defaults(func=list_recommendations)
    
    # record-feedback command
    feedback_parser = subparsers.add_parser("record-feedback", 
                                          help="Record feedback for a hardware recommendation")
    feedback_parser.add_argument("--recommendation-id", type=str, required=True,
                                help="Recommendation ID")
    feedback_parser.add_argument("--accepted", type=str, required=True, 
                                choices=["yes", "no", "true", "false", "y", "n", "1", "0"],
                                help="Whether the recommendation was accepted")
    feedback_parser.add_argument("--feedback", type=str,
                                help="Optional feedback text")
    feedback_parser.add_argument("--user-id", type=str, default="cli-user",
                                help="User ID")
    feedback_parser.set_defaults(func=record_feedback)
    
    # generate-sample-data command
    sample_parser = subparsers.add_parser("generate-sample-data", 
                                        help="Generate sample data for testing")
    sample_parser.add_argument("--num-models", type=int, default=5,
                              help="Number of sample models to generate")
    sample_parser.set_defaults(func=generate_sample_data)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Call the appropriate function
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()