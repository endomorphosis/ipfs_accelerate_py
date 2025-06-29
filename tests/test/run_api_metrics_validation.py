#!/usr/bin/env python3
"""
Run API Metrics Validation Tool

This script demonstrates how to use the API Metrics Validation tools
to validate API performance metrics quality, prediction accuracy,
anomaly detection effectiveness, and recommendation relevance.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try different import paths
try:
    from duckdb_api.simulation_validation.api_metrics import DuckDBAPIMetricsRepository, APIMetricsValidator
except ImportError:
    try:
        from ipfs_accelerate_py.duckdb_api.simulation_validation.api_metrics import DuckDBAPIMetricsRepository, APIMetricsValidator
    except ImportError:
        from test.duckdb_api.simulation_validation.api_metrics import DuckDBAPIMetricsRepository, APIMetricsValidator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run API Metrics Validation Tool')
    
    parser.add_argument('--db-path', type=str, default='api_metrics.duckdb',
                        help='Path to DuckDB database file')
    
    parser.add_argument('--generate-sample', action='store_true',
                        help='Generate sample data in the DuckDB database')
    
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of sample records to generate')
    
    parser.add_argument('--days-back', type=int, default=30,
                        help='Number of days back to generate data for')
    
    parser.add_argument('--endpoint', type=str,
                        help='Filter by specific API endpoint')
    
    parser.add_argument('--model', type=str,
                        help='Filter by specific model')
    
    parser.add_argument('--output', type=str,
                        help='Path to output JSON file for validation results')
    
    parser.add_argument('--report-type', type=str, default='full',
                        choices=['full', 'data-quality', 'prediction', 'anomaly', 'recommendation'],
                        help='Type of validation report to generate')
    
    return parser.parse_args()


def run_validation(args):
    """Run the API metrics validation tool with the specified arguments."""
    try:
        # Create repository instance
        logger.info(f"Connecting to database at {args.db_path}")
        repository = DuckDBAPIMetricsRepository(
            db_path=args.db_path,
            create_if_missing=True
        )
        
        # Generate sample data if requested
        if args.generate_sample:
            logger.info(f"Generating {args.num_samples} sample records over {args.days_back} days")
            repository.generate_sample_data(
                num_records=args.num_samples,
                days_back=args.days_back
            )
            logger.info("Sample data generation completed")
        
        # Create validator instance
        validator = APIMetricsValidator(repository=repository)
        
        # Determine time range for validation
        end_time = datetime.now()
        start_time = end_time - timedelta(days=args.days_back)
        
        # Run validation based on report type
        if args.report_type == 'full':
            logger.info("Generating full validation report")
            results = validator.generate_validation_report(
                start_time=start_time,
                end_time=end_time,
                endpoint=args.endpoint,
                model=args.model
            )
        elif args.report_type == 'data-quality':
            logger.info("Validating data quality")
            results = validator.validate_data_quality(
                start_time=start_time,
                end_time=end_time,
                endpoint=args.endpoint,
                model=args.model
            )
        elif args.report_type == 'prediction':
            logger.info("Validating prediction accuracy")
            results = validator.validate_prediction_accuracy(
                start_time=start_time,
                end_time=end_time,
                endpoint=args.endpoint,
                model=args.model
            )
        elif args.report_type == 'anomaly':
            logger.info("Validating anomaly detection")
            results = validator.validate_anomaly_detection(
                start_time=start_time,
                end_time=end_time,
                endpoint=args.endpoint,
                model=args.model
            )
        elif args.report_type == 'recommendation':
            logger.info("Validating recommendation relevance")
            results = validator.validate_recommendation_relevance(
                endpoint=args.endpoint,
                model=args.model
            )
        
        # Save results to file if output path provided
        if args.output:
            # Convert datetime objects to ISO format strings
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            with open(args.output, 'w') as f:
                json.dump(results, f, default=json_serializer, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Print summary to console
        print_summary(results, args.report_type)
        
        # Close the repository connection
        repository.close()
        
    except Exception as e:
        logger.error(f"Error running validation: {str(e)}", exc_info=True)
        return 1
    
    return 0


def print_summary(results, report_type):
    """Print a summary of validation results to the console."""
    print("\n" + "="*80)
    print(f"API METRICS VALIDATION SUMMARY ({report_type.upper()})")
    print("="*80)
    
    if results.get('status') == 'error':
        print(f"ERROR: {results.get('message', 'Unknown error')}")
        return
    
    if report_type == 'full':
        print(f"Overall Score: {results.get('overall_score', 0.0):.2f}")
        print(f"Status: {results.get('status', 'unknown')}")
        print("\nComponent Scores:")
        print(f"- Data Quality: {results.get('data_quality', {}).get('overall_quality', 0.0):.2f}")
        print(f"- Prediction Accuracy: {results.get('prediction_accuracy', {}).get('accuracy', 0.0):.2f}")
        print(f"- Anomaly Detection: {results.get('anomaly_detection', {}).get('effectiveness', 0.0):.2f}")
        print(f"- Recommendation Quality: {results.get('recommendation_relevance', {}).get('overall_quality', 0.0):.2f}")
        
        print("\nTop Recommendations:")
        for i, rec in enumerate(results.get('all_recommendations', [])[:5], 1):
            print(f"{i}. [{rec.get('category', 'General')} - {rec.get('priority', 'medium').upper()}] {rec.get('issue', '')}")
    
    elif report_type == 'data-quality':
        print(f"Overall Quality: {results.get('overall_quality', 0.0):.2f}")
        print(f"Threshold Met: {results.get('threshold_met', False)}")
        print("\nQuality Dimensions:")
        print(f"- Completeness: {results.get('completeness', 0.0):.2f}")
        print(f"- Consistency: {results.get('consistency', 0.0):.2f}")
        print(f"- Validity: {results.get('validity', 0.0):.2f}")
        print(f"- Timeliness: {results.get('timeliness', 0.0):.2f}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(results.get('recommendations', []), 1):
            print(f"{i}. [{rec.get('priority', 'medium').upper()}] {rec.get('issue', '')}: {rec.get('recommendation', '')}")
    
    elif report_type == 'prediction':
        print(f"Accuracy: {results.get('accuracy', 0.0):.2f}")
        print(f"Threshold Met: {results.get('threshold_met', False)}")
        print(f"MAE: {results.get('mae', 0.0):.4f}")
        print(f"RMSE: {results.get('rmse', 0.0):.4f}")
        print(f"R²: {results.get('r2', 0.0):.4f}")
        print(f"Sample Size: {results.get('sample_size', 0)}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(results.get('recommendations', []), 1):
            print(f"{i}. [{rec.get('priority', 'medium').upper()}] {rec.get('issue', '')}: {rec.get('recommendation', '')}")
    
    elif report_type == 'anomaly':
        print(f"Effectiveness: {results.get('effectiveness', 0.0):.2f}")
        print(f"Threshold Met: {results.get('threshold_met', False)}")
        print(f"Precision: {results.get('precision', 0.0):.2f}")
        print(f"Recall: {results.get('recall', 0.0):.2f}")
        print(f"F1 Score: {results.get('f1_score', 0.0):.2f}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(results.get('recommendations', []), 1):
            print(f"{i}. [{rec.get('priority', 'medium').upper()}] {rec.get('issue', '')}: {rec.get('recommendation', '')}")
    
    elif report_type == 'recommendation':
        print(f"Overall Quality: {results.get('overall_quality', 0.0):.2f}")
        print(f"Threshold Met: {results.get('threshold_met', False)}")
        print(f"Relevance Score: {results.get('relevance_score', 0.0):.2f}")
        print(f"Actionability Score: {results.get('actionability_score', 0.0):.2f}")
        print(f"Impact Coverage: {results.get('impact_coverage', 0.0):.2f}")
        
        print("\nRecommendations for Improvement:")
        for i, rec in enumerate(results.get('recommendations', []), 1):
            print(f"{i}. [{rec.get('priority', 'medium').upper()}] {rec.get('issue', '')}: {rec.get('recommendation', '')}")
    
    print("="*80)


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_validation(args))