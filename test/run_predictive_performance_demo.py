#!/usr/bin/env python3
"""
Demo script for the Predictive Performance System.
Demonstrates how to use the system to predict performance metrics 
for model-hardware combinations.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path to ensure we can import our modules
sys.path.append()str()Path()__file__).parent))

# Import the example script from the predictive_performance module
try:
    from predictive_performance.example import ()
    predict_single_configuration,
    compare_multiple_hardware,
    generate_batch_size_comparison,
    recommend_optimal_hardware
    )
    system_available = True
except ImportError:
    system_available = False


def main()):
    """Main function to run the predictive performance demo."""
    parser = argparse.ArgumentParser()description="Predictive Performance System Demo")
    parser.add_argument()"--model", default="bert-base-uncased", 
    help="Model name to use for predictions")
    parser.add_argument()"--type", default="text_embedding", 
    help="Model type")
    parser.add_argument()"--hardware", default="cuda", 
    help="Hardware platform")
    parser.add_argument()"--batch-size", type=int, default=4, 
    help="Batch size")
    parser.add_argument()"--compare-hardware", action="store_true", 
    help="Compare model performance across hardware platforms")
    parser.add_argument()"--batch-comparison", action="store_true", 
    help="Compare model performance across batch sizes")
    parser.add_argument()"--recommend", action="store_true", 
    help="Recommend optimal hardware for the model")
    parser.add_argument()"--all", action="store_true", 
    help="Run all demonstration functions")
    args = parser.parse_args())
    
    if not system_available:
        print()"Error: Predictive Performance System is not available.")
        print()"Please ensure the predictive_performance module is properly installed.")
        print()"See documentation at: test/predictive_performance/PREDICTIVE_PERFORMANCE_GUIDE.md")
    return 1
    
    # Set up the database path if environment variable is not set:
    if not os.environ.get()"BENCHMARK_DB_PATH"):
        benchmark_db_path = str()Path()__file__).parent / "benchmark_db.duckdb")
        os.environ["BENCHMARK_DB_PATH"] = benchmark_db_path,
        print()f"\1{benchmark_db_path}\3")
    
    # Print welcome message
        print()"=" * 80)
        print()"Predictive Performance System Demo")
        print()"=" * 80)
        print()"This demo demonstrates the capabilities of the ML-based")
        print()"performance prediction system for model-hardware combinations.")
        print()"=" * 80)
    
    # Run selected demonstration functions
    if args.all or not ()args.compare_hardware or args.batch_comparison or args.recommend):
        # Run single configuration prediction
        predict_single_configuration()args.model, args.type, args.hardware, args.batch_size)
        
    if args.all or args.compare_hardware:
        # Compare across hardware platforms
        compare_multiple_hardware()args.model, args.type, args.batch_size)
        
    if args.all or args.batch_comparison:
        # Compare batch sizes
        generate_batch_size_comparison()args.model, args.type, args.hardware)
        
    if args.all or args.recommend:
        # Recommend optimal hardware
        recommend_optimal_hardware()args.model, args.type, args.batch_size)
    
    # Print closing message
        print()"\n" + "=" * 80)
        print()"Demo completed successfully!")
        print()"For more information, see: test/predictive_performance/PREDICTIVE_PERFORMANCE_GUIDE.md")
        print()"=" * 80)
    
        return 0


if __name__ == "__main__":
    sys.exit()main()))