#!/usr/bin/env python3
"""
Comprehensive Test for the Predictive Performance System

This script performs a series of tests to validate that the Predictive Performance System
is working correctly. It tests:

    1. Basic prediction functionality
    2. Prediction accuracy validation against known benchmarks
    3. Hardware recommendation system
    4. Active learning pipeline
    5. Visualization generation
    6. Integration with benchmark scheduler

Usage:
    python test_predictive_performance_system.py --full-test
    python test_predictive_performance_system.py --quick-test
    python test_predictive_performance_system.py --test-component prediction
    """

    import os
    import sys
    import json
    import time
    import logging
    import argparse
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from pathlib import Path

# Configure logging
    logging.basicConfig())))))))))
    level=logging.INFO,
    format='%())))))))))asctime)s - %())))))))))levelname)s - %())))))))))message)s',
    handlers=[]]]]],,,,,
    logging.StreamHandler())))))))))),
    logging.FileHandler())))))))))f"predictive_performance_test_{}}}}datetime.now())))))))))).strftime())))))))))'%Y%m%d_%H%M%S')}.log")
    ]
    )
    logger = logging.getLogger())))))))))__name__)

# Add parent directory to path for imports
    sys.path.append())))))))))os.path.dirname())))))))))os.path.abspath())))))))))__file__)))

# Import the Predictive Performance System modules
try:
    from predictive_performance.predict import PerformancePredictor
    from predictive_performance.example import ())))))))))
    predict_single_configuration,
    compare_multiple_hardware,
    generate_batch_size_comparison,
    recommend_optimal_hardware
    )
    from predictive_performance.active_learning import ActiveLearningSystem
    PPS_AVAILABLE = True
except ImportError as e:
    logger.error())))))))))f"Failed to import Predictive Performance System modules: {}}}}e}")
    PPS_AVAILABLE = False

def test_basic_prediction())))))))))):
    """Test basic prediction functionality."""
    logger.info())))))))))"Testing basic prediction functionality...")
    test_cases = []]]]],,,,,
    {}}"model_name": "bert-base-uncased", "model_type": "text_embedding", "hardware": "cuda", "batch_size": 8},
    {}}"model_name": "t5-small", "model_type": "text_generation", "hardware": "cpu", "batch_size": 1},
    {}}"model_name": "whisper-tiny", "model_type": "audio", "hardware": "webgpu", "batch_size": 4}
    ]
    
    predictor = PerformancePredictor()))))))))))
    
    all_passed = True
    for case in test_cases:
        logger.info())))))))))f"Testing prediction for {}}}}case[]]]]],,,,,'model_name']} on {}}}}case[]]]]],,,,,'hardware']} with batch size {}}}}case[]]]]],,,,,'batch_size']}")
        try:
            prediction = predictor.predict())))))))))
            model_name=case[]]]]],,,,,"model_name"],
            model_type=case[]]]]],,,,,"model_type"],
            hardware_platform=case[]]]]],,,,,"hardware"],
            batch_size=case[]]]]],,,,,"batch_size"]
            )
            
            # Check if the prediction has the expected fields
            required_fields = []]]]],,,,,"throughput", "latency", "memory", "confidence"]
            missing_fields = []]]]],,,,,field for field in required_fields if field not in prediction]
            :
            if missing_fields:
                logger.error())))))))))f"Missing required fields in prediction: {}}}}missing_fields}")
                all_passed = False
                continue
                
            # Check if values are reasonable:
            if prediction[]]]]],,,,,"throughput"] <= 0 or prediction[]]]]],,,,,"latency"] <= 0 or prediction[]]]]],,,,,"memory"] <= 0:
                logger.error())))))))))f"Prediction contains invalid values: {}}}}prediction}")
                all_passed = False
                continue
                
            # Check if confidence values are in the expected range ())))))))))0-1):
            if not ())))))))))0 <= prediction[]]]]],,,,,"confidence"] <= 1):
                logger.error())))))))))f"Confidence score out of range: {}}}}prediction[]]]]],,,,,'confidence']}")
                all_passed = False
                continue
                
                logger.info())))))))))f"Prediction successful: {}}}}prediction}")
            
        except Exception as e:
            logger.error())))))))))f"Error during prediction: {}}}}e}")
            all_passed = False
    
                return all_passed

def test_prediction_accuracy())))))))))):
    """Test prediction accuracy against known benchmark results."""
    logger.info())))))))))"Testing prediction accuracy against known benchmarks...")
    
    # Define known benchmark results ())))))))))model, hardware, batch_size, actual_throughput, actual_latency, actual_memory)
    # For simulation mode, these values should match what the simulation produces ())))))))))approximately)
    benchmark_results = []]]]],,,,,
    {}}"model_name": "bert-base-uncased", "model_type": "text_embedding", "hardware": "cuda", "batch_size": 8,
    "actual_throughput": 6000, "actual_latency": 4.0, "actual_memory": 3000},
    {}}"model_name": "t5-small", "model_type": "text_generation", "hardware": "cpu", "batch_size": 1,
    "actual_throughput": 20, "actual_latency": 100, "actual_memory": 3000}
    ]
    
    predictor = PerformancePredictor()))))))))))
    
    accuracy_metrics = {}}
    "throughput": []]]]],,,,,],
    "latency": []]]]],,,,,],
    "memory": []]]]],,,,,]
    }
    
    for benchmark in benchmark_results:
        logger.info())))))))))f"Testing accuracy for {}}}}benchmark[]]]]],,,,,'model_name']} on {}}}}benchmark[]]]]],,,,,'hardware']}")
        try:
            prediction = predictor.predict())))))))))
            model_name=benchmark[]]]]],,,,,"model_name"],
            model_type=benchmark[]]]]],,,,,"model_type"],
            hardware_platform=benchmark[]]]]],,,,,"hardware"],
            batch_size=benchmark[]]]]],,,,,"batch_size"]
            )
            
            # Calculate accuracy for each metric
            throughput_accuracy = min())))))))))prediction[]]]]],,,,,"throughput"], benchmark[]]]]],,,,,"actual_throughput"]) / max())))))))))prediction[]]]]],,,,,"throughput"], benchmark[]]]]],,,,,"actual_throughput"])
            latency_accuracy = min())))))))))prediction[]]]]],,,,,"latency"], benchmark[]]]]],,,,,"actual_latency"]) / max())))))))))prediction[]]]]],,,,,"latency"], benchmark[]]]]],,,,,"actual_latency"])
            memory_accuracy = min())))))))))prediction[]]]]],,,,,"memory"], benchmark[]]]]],,,,,"actual_memory"]) / max())))))))))prediction[]]]]],,,,,"memory"], benchmark[]]]]],,,,,"actual_memory"])
            
            accuracy_metrics[]]]]],,,,,"throughput"].append())))))))))throughput_accuracy)
            accuracy_metrics[]]]]],,,,,"latency"].append())))))))))latency_accuracy)
            accuracy_metrics[]]]]],,,,,"memory"].append())))))))))memory_accuracy)
            
            logger.info())))))))))f"Throughput accuracy: {}}}}throughput_accuracy:.2f}, Latency accuracy: {}}}}latency_accuracy:.2f}, Memory accuracy: {}}}}memory_accuracy:.2f}")
            
        except Exception as e:
            logger.error())))))))))f"Error during accuracy test: {}}}}e}")
    
    # Calculate average accuracy for each metric
            avg_accuracy = {}}}
    for metric, values in accuracy_metrics.items())))))))))):
        if values:
            avg_accuracy[]]]]],,,,,metric] = sum())))))))))values) / len())))))))))values)
            logger.info())))))))))f"Average {}}}}metric} accuracy: {}}}}avg_accuracy[]]]]],,,,,metric]:.2f}")
        else:
            logger.warning())))))))))f"No data available for {}}}}metric} accuracy")
    
    # For simulation mode, use a lower threshold since the simulation doesn't have to match exactly
            accuracy_threshold = 0.70  # 70% accuracy for simulation mode
    return all())))))))))acc >= accuracy_threshold for acc in avg_accuracy.values()))))))))))):
def test_hardware_recommendation())))))))))):
    """Test the hardware recommendation system."""
    logger.info())))))))))"Testing hardware recommendation system...")
    
    test_cases = []]]]],,,,,
    {}}"model_type": "text_embedding", "optimization_goal": "throughput"},
    {}}"model_type": "text_generation", "optimization_goal": "latency"},
    {}}"model_type": "vision", "optimization_goal": "memory"}
    ]
    
    all_passed = True
    for case in test_cases:
        logger.info())))))))))f"Testing recommendation for {}}}}case[]]]]],,,,,'model_type']} models optimizing for {}}}}case[]]]]],,,,,'optimization_goal']}")
        try:
            recommendations = recommend_optimal_hardware())))))))))
            model_type=case[]]]]],,,,,"model_type"],
            optimize_for=case[]]]]],,,,,"optimization_goal"]
            )
            
            # Check if recommendations are returned:
            if not recommendations:
                logger.error())))))))))f"No recommendations returned for {}}}}case}")
                all_passed = False
            continue
                
            # Check if top recommendation has expected fields
            top_recommendation = recommendations[]]]]],,,,,0]
            required_fields = []]]]],,,,,"hardware", "score", "throughput", "latency", "memory"]
            missing_fields = []]]]],,,,,field for field in required_fields if field not in top_recommendation]
            :
            if missing_fields:
                logger.error())))))))))f"Missing required fields in recommendation: {}}}}missing_fields}")
                all_passed = False
                continue
                
                logger.info())))))))))f"Top recommendation: {}}}}top_recommendation}")
            
        except Exception as e:
            logger.error())))))))))f"Error during hardware recommendation: {}}}}e}")
            all_passed = False
    
                return all_passed

def test_active_learning())))))))))):
    """Test the active learning pipeline."""
    logger.info())))))))))"Testing active learning pipeline...")
    
    try:
        # Initialize active learning system
        active_learning = ActiveLearningSystem()))))))))))
        
        # Request high-value configurations with a budget of 5
        configurations = active_learning.recommend_configurations())))))))))budget=5)
        
        # Check if configurations are returned:
        if not configurations:
            logger.error())))))))))"No configurations returned by active learning")
        return False
            
        # Check if each configuration has the expected fields
        required_fields = []]]]],,,,,"model_name", "model_type", "hardware", "batch_size", "expected_information_gain"]
        :
        for config in configurations:
            missing_fields = []]]]],,,,,field for field in required_fields if field not in config]:
            if missing_fields:
                logger.error())))))))))f"Missing required fields in configuration: {}}}}missing_fields}")
                return False
        
                logger.info())))))))))f"Active learning returned {}}}}len())))))))))configurations)} configurations")
                logger.info())))))))))f"Top configuration: {}}}}configurations[]]]]],,,,,0]}")
        
        # Check if configurations are ranked by expected information gain:
        gains = []]]]],,,,,config[]]]]],,,,,"expected_information_gain"] for config in configurations]:
        if gains != sorted())))))))))gains, reverse=True):
            logger.error())))))))))"Configurations are not properly ranked by information gain")
            return False
            
                return True
        
    except Exception as e:
        logger.error())))))))))f"Error during active learning test: {}}}}e}")
                return False

def test_visualization())))))))))):
    """Test the visualization generation."""
    logger.info())))))))))"Testing visualization generation...")
    
    try:
        # Test batch size comparison visualization
        output_file = "test_batch_size_comparison.png"
        generate_batch_size_comparison())))))))))
        model_name="bert-base-uncased",
        model_type="text_embedding",
        hardware="cuda",
        batch_sizes=[]]]]],,,,,1, 2, 4, 8, 16, 32],
        output_file=output_file
        )
        
        # Check if the file was created:
        if not os.path.exists())))))))))output_file):
            logger.error())))))))))f"Visualization file {}}}}output_file} was not created")
        return False
            
        logger.info())))))))))f"Visualization file created: {}}}}output_file}")
        
        # Test hardware comparison visualization
        predictor = PerformancePredictor()))))))))))
        predictor.visualize_hardware_comparison())))))))))
        model_name="bert-base-uncased",
        model_type="text_embedding",
        batch_size=8,
        output_file="test_hardware_comparison.png"
        )
        
        if not os.path.exists())))))))))"test_hardware_comparison.png"):
            logger.error())))))))))"Hardware comparison visualization file was not created")
        return False
            
        logger.info())))))))))"Hardware comparison visualization file created")
        
    return True
        
    except Exception as e:
        logger.error())))))))))f"Error during visualization test: {}}}}e}")
    return False

def test_benchmark_scheduler_integration())))))))))):
    """Test integration with the benchmark scheduler."""
    logger.info())))))))))"Testing benchmark scheduler integration...")
    
    try:
        # Create a temporary recommendations file
        recommendations_file = "test_recommendations.json"
        
        # Use active learning to generate recommendations
        active_learning = ActiveLearningSystem()))))))))))
        configurations = active_learning.recommend_configurations())))))))))budget=3)
        
        # Save recommendations to file
        with open())))))))))recommendations_file, "w") as f:
            json.dump())))))))))configurations, f)
        
        # Test if the benchmark scheduler can read the recommendations
        # This is a simulated test as we don't want to actually run benchmarks
            from predictive_performance.benchmark_integration import BenchmarkScheduler
        
            scheduler = BenchmarkScheduler()))))))))))
            loaded_configs = scheduler.load_recommendations())))))))))recommendations_file)
        :
        if len())))))))))loaded_configs) != len())))))))))configurations):
            logger.error())))))))))f"Benchmark scheduler loaded {}}}}len())))))))))loaded_configs)} configs, expected {}}}}len())))))))))configurations)}")
            return False
            
        # Test converting recommendations to benchmark commands
            benchmark_commands = scheduler.generate_benchmark_commands())))))))))loaded_configs)
        
        if len())))))))))benchmark_commands) != len())))))))))configurations):
            logger.error())))))))))f"Generated {}}}}len())))))))))benchmark_commands)} commands, expected {}}}}len())))))))))configurations)}")
            return False
            
            logger.info())))))))))f"Successfully generated {}}}}len())))))))))benchmark_commands)} benchmark commands")
            logger.info())))))))))f"Sample command: {}}}}benchmark_commands[]]]]],,,,,0]}")
        
            return True
        
    except Exception as e:
        logger.error())))))))))f"Error during benchmark scheduler integration test: {}}}}e}")
            return False
    finally:
        # Clean up
        if os.path.exists())))))))))recommendations_file):
            os.remove())))))))))recommendations_file)

def run_all_tests())))))))))):
    """Run all tests and report results."""
    if not PPS_AVAILABLE:
        logger.error())))))))))"Predictive Performance System modules not available. Tests cannot run.")
    return False
        
    test_functions = []]]]],,,,,
    test_basic_prediction,
    test_prediction_accuracy,
    test_hardware_recommendation,
    test_active_learning,
    test_visualization,
    test_benchmark_scheduler_integration
    ]
    
    results = {}}}
    overall_result = True
    
    for test_func in test_functions:
        test_name = test_func.__name__
        logger.info())))))))))f"\n{}}}}'='*80}\nRunning test: {}}}}test_name}\n{}}}}'='*80}")
        
        try:
            start_time = time.time()))))))))))
            result = test_func()))))))))))
            elapsed_time = time.time())))))))))) - start_time
            
            results[]]]]],,,,,test_name] = result
            if not result:
                overall_result = False
                
                logger.info())))))))))f"Test {}}}}test_name} {}}}}'PASSED' if result else 'FAILED'} in {}}}}elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error())))))))))f"Test {}}}}test_name} FAILED with exception: {}}}}e}")
            results[]]]]],,,,,test_name] = False
            overall_result = False
    
    # Print summary
            logger.info())))))))))"\n" + "="*80)
            logger.info())))))))))"TEST SUMMARY")
            logger.info())))))))))"="*80)
    
    for test_name, result in results.items())))))))))):
        logger.info())))))))))f"{}}}}test_name}: {}}}}'PASSED' if result else 'FAILED'}")
        :
            logger.info())))))))))f"\nOVERALL RESULT: {}}}}'PASSED' if overall_result else 'FAILED'}")
    
        return overall_result
:
def run_quick_test())))))))))):
    """Run a quick test of basic functionality."""
    if not PPS_AVAILABLE:
        logger.error())))))))))"Predictive Performance System modules not available. Tests cannot run.")
    return False
        
    logger.info())))))))))"Running quick test...")
    
    try:
        # Test basic prediction
        predictor = PerformancePredictor()))))))))))
        prediction = predictor.predict())))))))))
        model_name="bert-base-uncased",
        model_type="text_embedding",
        hardware_platform="cuda",
        batch_size=8
        )
        
        logger.info())))))))))f"Prediction: {}}}}prediction}")
        
        # Test hardware recommendation
        recommendation = recommend_optimal_hardware())))))))))
        model_type="text_embedding",
        optimize_for="throughput"
        )
        
        logger.info())))))))))f"Recommendation: {}}}}recommendation[]]]]],,,,,0] if recommendation else None}")
        
        # Quick visualization test
        generate_batch_size_comparison())))))))))
        model_name="bert-base-uncased",
        model_type="text_embedding",
        hardware="cuda",
        batch_sizes=[]]]]],,,,,1, 4, 16],
        output_file="quick_test_visualization.png"
        )
        
        logger.info())))))))))"Quick test PASSED")
    return True
        :
    except Exception as e:
        logger.error())))))))))f"Quick test FAILED with exception: {}}}}e}")
            return False

def main())))))))))):
    parser = argparse.ArgumentParser())))))))))description="Test the Predictive Performance System")
    test_group = parser.add_mutually_exclusive_group())))))))))required=True)
    test_group.add_argument())))))))))"--full-test", action="store_true", help="Run all tests")
    test_group.add_argument())))))))))"--quick-test", action="store_true", help="Run a quick test of basic functionality")
    test_group.add_argument())))))))))"--test-component", type=str, choices=[]]]]],,,,,"prediction", "accuracy", "recommendation", "active_learning", "visualization", "scheduler"], 
    help="Test a specific component")
    
    args = parser.parse_args()))))))))))
    
    if args.full_test:
        success = run_all_tests()))))))))))
    elif args.quick_test:
        success = run_quick_test()))))))))))
    elif args.test_component:
        # Map component names to test functions
        component_tests = {}}
        "prediction": test_basic_prediction,
        "accuracy": test_prediction_accuracy,
        "recommendation": test_hardware_recommendation,
        "active_learning": test_active_learning,
        "visualization": test_visualization,
        "scheduler": test_benchmark_scheduler_integration
        }
        
        test_func = component_tests[]]]]],,,,,args.test_component]
        logger.info())))))))))f"Testing component: {}}}}args.test_component}")
        success = test_func()))))))))))
        logger.info())))))))))f"Test {}}}}'PASSED' if success else 'FAILED'}")
    
        return 0 if success else 1
:
if __name__ == "__main__":
    sys.exit())))))))))main())))))))))))