#!/usr/bin/env python3
"""
Example script showcasing the Predictive Performance System.

This script demonstrates how to use different components of the Predictive Performance
System together, including prediction, active learning, and hardware recommendation.

Usage:
    python example.py --mode predict --model bert-base-uncased --hardware cuda
    python example.py --mode active-learning --budget 10
    python example.py --mode recommend-hardware --model vit-base --batch-size 8
    python example.py --mode integrate --budget 5 --metric throughput
"""

import os
import sys
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path to import the modules
sys.path.append()))))))))))))str()))))))))))))Path()))))))))))))__file__).parent.parent))

# Import the modules
try:
    from predictive_performance.predict import PerformancePredictor
    from predictive_performance.active_learning import ActiveLearningSystem
    from predictive_performance.benchmark_integration import BenchmarkScheduler
except ImportError as e:
    print()))))))))))))f"Error importing predictive performance modules: {}}}e}")
    sys.exit()))))))))))))1)


def predict_single_configuration()))))))))))))model_name, model_type, hardware, batch_size, precision="fp32"):
    """Demonstrates prediction for a single configuration."""
    print()))))))))))))f"\n--- Predicting for {}}}model_name} on {}}}hardware} with batch size {}}}batch_size} ---")
    
    # Initialize the predictor
    predictor = PerformancePredictor())))))))))))))
    
    # Make a prediction for the specified configuration
    prediction = predictor.predict()))))))))))))
    model_name=model_name,
    model_type=model_type,
    hardware_platform=hardware,
    batch_size=batch_size,
    precision=precision
    )
    
    # Print the prediction results with confidence scores
    print()))))))))))))f"Predicted throughput: {}}}prediction['throughput']:.2f} items/sec ()))))))))))))confidence: {}}}prediction['confidence'],:.2f})"),
    print()))))))))))))f"Predicted latency: {}}}prediction['latency']:.2f} ms ()))))))))))))confidence: {}}}prediction['confidence_latency']:.2f})"),
    print()))))))))))))f"Predicted memory: {}}}prediction['memory']:.2f} MB ()))))))))))))confidence: {}}}prediction['confidence_memory']:.2f})")
    ,
    if 'power' in prediction:
        print()))))))))))))f"Predicted power: {}}}prediction['power']:.2f} W ()))))))))))))confidence: {}}}prediction['confidence_power']:.2f})")
        ,
    return prediction


def compare_multiple_hardware()))))))))))))model_name, model_type, batch_size, precision="fp32"):
    """Compares predictions across multiple hardware platforms."""
    print()))))))))))))f"\n--- Comparing hardware platforms for {}}}model_name} with batch size {}}}batch_size} ---")
    
    # List of hardware platforms to compare
    hardware_platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]
    ,,
    # Initialize the predictor
    predictor = PerformancePredictor())))))))))))))
    
    # Make predictions for each hardware platform
    results = [],,
    for hardware in hardware_platforms:
        prediction = predictor.predict()))))))))))))
        model_name=model_name,
        model_type=model_type,
        hardware_platform=hardware,
        batch_size=batch_size,
        precision=precision
        )
        
        results.append())))))))))))){}}}
        "Hardware": hardware,
        "Throughput": prediction['throughput'],
        "Latency": prediction['latency'],
        "Memory": prediction['memory'],
        "Confidence": prediction['confidence'],
        })
    
    # Create a DataFrame from the results
        df = pd.DataFrame()))))))))))))results)
    
    # Print the comparison table
        print()))))))))))))"\nHardware Performance Comparison:")
        print()))))))))))))df.to_string()))))))))))))index=False))
    
    # Create and save a bar chart for throughput
        plt.figure()))))))))))))figsize=()))))))))))))12, 6))
        plt.bar()))))))))))))df['Hardware'], df['Throughput'], color='skyblue'),
        plt.title()))))))))))))f'Predicted Throughput for {}}}model_name} ()))))))))))))Batch Size {}}}batch_size})')
        plt.xlabel()))))))))))))'Hardware Platform')
        plt.ylabel()))))))))))))'Throughput ()))))))))))))items/sec)')
        plt.grid()))))))))))))axis='y', linestyle='--', alpha=0.7)
        plt.xticks()))))))))))))rotation=45)
    
    # Save the chart
        output_file = f"{}}}model_name.replace()))))))))))))'/', '_')}_throughput_comparison.png"
        plt.tight_layout())))))))))))))
        plt.savefig()))))))))))))output_file)
        print()))))))))))))f"\nThroughput comparison chart saved to: {}}}output_file}")
    
    return df


def generate_batch_size_comparison()))))))))))))model_name, model_type, hardware, batch_sizes=None, precision="fp32", output_file=None):
    """Demonstrates batch size comparison for a model on specific hardware."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]
        ,
    if output_file is None:
        output_file = f"{}}}model_name.replace()))))))))))))'/', '_')}_{}}}hardware}_batch_comparison.png"
        
        print()))))))))))))f"\n--- Comparing batch sizes for {}}}model_name} on {}}}hardware} ---")
    
    # Initialize the predictor
        predictor = PerformancePredictor())))))))))))))
    
    # Make predictions for different batch sizes
        batch_results = {}}}}
    for batch_size in batch_sizes:
        batch_results[batch_size] = predictor.predict())))))))))))),
        model_name=model_name,
        model_type=model_type,
        hardware_platform=hardware,
        batch_size=batch_size,
        precision=precision
        )
        
    # Extract throughput and latency
        throughputs = [batch_results[bs]["throughput"] for bs in batch_sizes]:,
    latencies = [batch_results[bs]["latency"] for bs in batch_sizes]:
        ,
    # Print the throughput for each batch size
        print()))))))))))))"\nThroughput by Batch Size:")
    for i, batch_size in enumerate()))))))))))))batch_sizes):
        print()))))))))))))f"Batch Size {}}}batch_size}: {}}}throughputs[i]:.2f} items/sec")
        ,
    # Create visualization
        fig, ()))))))))))))ax1, ax2) = plt.subplots()))))))))))))1, 2, figsize=()))))))))))))12, 5))
    
    # Throughput vs. batch size
        ax1.plot()))))))))))))batch_sizes, throughputs, marker='o', linestyle='-', color='royalblue')
        ax1.set_title()))))))))))))f"Throughput vs. Batch Size - {}}}model_name} on {}}}hardware}")
        ax1.set_xlabel()))))))))))))"Batch Size")
        ax1.set_ylabel()))))))))))))"Throughput ()))))))))))))items/second)")
        ax1.set_xscale()))))))))))))'log', base=2)
        ax1.grid()))))))))))))True, linestyle='--', alpha=0.7)
    
    # Latency vs. batch size
        ax2.plot()))))))))))))batch_sizes, latencies, marker='o', linestyle='-', color='firebrick')
        ax2.set_title()))))))))))))f"Latency vs. Batch Size - {}}}model_name} on {}}}hardware}")
        ax2.set_xlabel()))))))))))))"Batch Size")
        ax2.set_ylabel()))))))))))))"Latency ()))))))))))))ms)")
        ax2.set_xscale()))))))))))))'log', base=2)
        ax2.grid()))))))))))))True, linestyle='--', alpha=0.7)
    
        plt.tight_layout())))))))))))))
        plt.savefig()))))))))))))output_file, dpi=300)
        print()))))))))))))f"Batch size comparison saved to {}}}output_file}")
    
        return output_file


def recommend_optimal_hardware()))))))))))))model_type, optimize_for="throughput", balance_factor=None):
    """Recommend optimal hardware platforms for a given model type and optimization goal."""
    print()))))))))))))f"\n--- Recommending hardware for {}}}model_type} models optimizing for {}}}optimize_for} ---")
    
    # Initialize the predictor
    predictor = PerformancePredictor())))))))))))))
    
    # Define hardware platforms to compare
    hardware_platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]
    ,,
    # Define batch sizes to test
    batch_sizes = [1, 8, 32]
    ,
    # Create a model name based on the model type
    model_name = f"example_{}}}model_type}_model"
    
    # Collect predictions for each hardware platform
    hardware_results = {}}}}
    
    for hardware in hardware_platforms:
        # Get predictions for multiple batch sizes
        batch_results = {}}}}
        for batch_size in batch_sizes:
            prediction = predictor.predict()))))))))))))
            model_name=model_name,
            model_type=model_type,
            hardware_platform=hardware,
            batch_size=batch_size
            )
            batch_results[batch_size] = prediction
            ,
        # Calculate average metrics across batch sizes
            avg_throughput = sum()))))))))))))batch_results[bs]["throughput"] for bs in batch_sizes) / len()))))))))))))batch_sizes),
            avg_latency = sum()))))))))))))batch_results[bs]["latency"] for bs in batch_sizes) / len()))))))))))))batch_sizes),
            avg_memory = sum()))))))))))))batch_results[bs]["memory"] for bs in batch_sizes) / len()))))))))))))batch_sizes)
            ,
        # Store results
            hardware_results[hardware] = {}}},
            "throughput": avg_throughput,
            "latency": avg_latency,
            "memory": avg_memory,
            "model_type": model_type
            }
    
    # Calculate scores based on optimization goal
            scores = {}}}}
    for hardware, metrics in hardware_results.items()))))))))))))):
        if optimize_for == "throughput":
            # Higher throughput is better
            scores[hardware] = metrics["throughput"],
        elif optimize_for == "latency":
            # Lower latency is better ()))))))))))))invert for scoring)
            scores[hardware] = 1.0 / metrics["latency"] if metrics["latency"] > 0 else float()))))))))))))'inf'):,
        elif optimize_for == "memory":
            # Lower memory is better ()))))))))))))invert for scoring)
            scores[hardware] = 1.0 / metrics["memory"] if metrics["memory"] > 0 else float()))))))))))))'inf'):,
        elif optimize_for == "balanced":
            # Balanced score considering all metrics
            throughput_score = metrics["throughput"] / max()))))))))))))hardware_results[h]["throughput"] for h in hardware_platforms):,
            latency_score = min()))))))))))))hardware_results[h]["latency"] for h in hardware_platforms): / metrics["latency"] if metrics["latency"] > 0 else 0:,
            memory_score = min()))))))))))))hardware_results[h]["memory"] for h in hardware_platforms): / metrics["memory"] if metrics["memory"] > 0 else 0
            ,
            # Default weights if not specified:
            if balance_factor is None:
                balance_factor = {}}}"throughput": 0.5, "latency": 0.3, "memory": 0.2}
                
                scores[hardware] = ())))))))))))),
                balance_factor["throughput"] * throughput_score +,
                balance_factor["latency"] * latency_score +,
                balance_factor["memory"] * memory_score,
                )
    
    # Sort hardware platforms by score ()))))))))))))descending)
                ranked_hardware = sorted()))))))))))))scores.keys()))))))))))))), key=lambda h: scores[h], reverse=True)
                ,
    # Create result list with scores and metrics
                recommendations = [],,
    for hardware in ranked_hardware:
        recommendations.append())))))))))))){}}}
        "hardware": hardware,
        "score": scores[hardware],
        "throughput": hardware_results[hardware]["throughput"],
        "latency": hardware_results[hardware]["latency"],
        "memory": hardware_results[hardware]["memory"],
        })
    
    # Print top recommendations
        print()))))))))))))"\nTop hardware recommendations:")
        for i, rec in enumerate()))))))))))))recommendations[:3], 1):,
        print()))))))))))))f"{}}}i}. {}}}rec['hardware']} ()))))))))))))score: {}}}rec['score']:.2f})"),
        print()))))))))))))f"   Throughput: {}}}rec['throughput']:.2f} items/s, Latency: {}}}rec['latency']:.2f} ms, Memory: {}}}rec['memory']:.2f} MB")
        ,
                return recommendations


def recommend_benchmark_configurations()))))))))))))budget=10, output_file=None):
    """Use active learning to recommend high-value benchmark configurations."""
    print()))))))))))))f"\n--- Recommending {}}}budget} high-value benchmark configurations ---")
    
    # Initialize the active learning system
    active_learning = ActiveLearningSystem())))))))))))))
    
    # Get recommendations
    recommendations = active_learning.recommend_configurations()))))))))))))budget=budget)
    
    # Print recommendations
    print()))))))))))))"\nRecommended configurations:")
    for i, config in enumerate()))))))))))))recommendations, 1):
        print()))))))))))))f"{}}}i}. Model: {}}}config['model_name']}, Hardware: {}}}config['hardware']}, Batch Size: {}}}config['batch_size']}"),
        print()))))))))))))f"   Expected Information Gain: {}}}config['expected_information_gain']:.4f}")
        ,
        # Print additional metrics if available:
        if "uncertainty" in config:
            print()))))))))))))f"   Uncertainty: {}}}config['uncertainty']:.4f}, Diversity: {}}}config['diversity']:.4f}"),
        if "selection_method" in config:
            print()))))))))))))f"   Selection Method: {}}}config['selection_method']}")
            ,
    # Save recommendations to file if specified:
    if output_file:
        # Create output directory if it doesn't exist
        os.makedirs()))))))))))))os.path.dirname()))))))))))))os.path.abspath()))))))))))))output_file)), exist_ok=True)
        
        # Save recommendations:
        with open()))))))))))))output_file, 'w') as f:
            json.dump()))))))))))))recommendations, f, indent=2)
            
            print()))))))))))))f"\nRecommendations saved to {}}}output_file}")
        
        # Explain how to use these recommendations
            print()))))))))))))"\nTo schedule benchmarks using these recommendations:")
            print()))))))))))))f"python -m predictive_performance.benchmark_integration --recommendations {}}}output_file} --execute")
    
        return recommendations


def schedule_benchmarks()))))))))))))recommendations_file, execute=False, db_path=None):
    """Schedule benchmarks based on recommendations."""
    print()))))))))))))f"\n--- Scheduling benchmarks from {}}}recommendations_file} ---")
    
    # Initialize the benchmark scheduler
    scheduler = BenchmarkScheduler()))))))))))))db_path=db_path)
    
    # Load recommendations
    recommendations = scheduler.load_recommendations()))))))))))))recommendations_file)
    
    if not recommendations:
        print()))))))))))))"No recommendations found")
    return
    
    # Print recommendations
    print()))))))))))))f"Loaded {}}}len()))))))))))))recommendations)} recommendations")
    
    # Generate benchmark commands
    commands = scheduler.generate_benchmark_commands()))))))))))))recommendations)
    
    # Print commands
    print()))))))))))))"\nBenchmark commands:")
    for i, command in enumerate()))))))))))))commands, 1):
        print()))))))))))))f"{}}}i}. {}}}command}")
    
    # Execute benchmarks if requested:
    if execute:
        print()))))))))))))"\nExecuting benchmarks...")
        result = scheduler.schedule_benchmarks()))))))))))))recommendations, execute=True)
        
        # Print results
        benchmark_results = scheduler.get_benchmark_results())))))))))))))
        if benchmark_results:
            print()))))))))))))f"\n{}}}len()))))))))))))benchmark_results)} benchmark results obtained")
            for result in benchmark_results:
                print()))))))))))))f"Model: {}}}result['model_name']}, Hardware: {}}}result['hardware']}, Batch Size: {}}}result['batch_size']}"),
                if "throughput" in result:
                    print()))))))))))))f"  Throughput: {}}}result['throughput']:.2f} items/sec"),
                if "latency" in result:
                    print()))))))))))))f"  Latency: {}}}result['latency']:.2f} ms"),
                if "memory" in result:
                    print()))))))))))))f"  Memory: {}}}result['memory']:.2f} MB")
                    ,
            # Save results
                    report_file = f"benchmark_report_{}}}datetime.now()))))))))))))).strftime()))))))))))))'%Y%m%d_%H%M%S')}.json"
                    scheduler.save_job_report()))))))))))))report_file)
                    print()))))))))))))f"\nBenchmark report saved to {}}}report_file}")
            
            # Save benchmark results
                    results_file = f"benchmark_results_{}}}datetime.now()))))))))))))).strftime()))))))))))))'%Y%m%d_%H%M%S')}.csv"
                    scheduler.save_benchmark_results()))))))))))))results_file)
                    print()))))))))))))f"Benchmark results saved to {}}}results_file}")
    
                    return commands


def main()))))))))))))):
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser()))))))))))))description="Predictive Performance System Example")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers()))))))))))))dest="mode", help="Operation mode")
    
    # Single prediction parser
    predict_parser = subparsers.add_parser()))))))))))))"predict", help="Predict performance for a single configuration")
    predict_parser.add_argument()))))))))))))"--model", default="bert-base-uncased", help="Model name")
    predict_parser.add_argument()))))))))))))"--type", default="text_embedding", help="Model type")
    predict_parser.add_argument()))))))))))))"--hardware", default="cuda", help="Hardware platform")
    predict_parser.add_argument()))))))))))))"--batch-size", type=int, default=4, help="Batch size")
    predict_parser.add_argument()))))))))))))"--precision", default="fp32", choices=["fp32", "fp16", "int8"], help="Precision format"),,
    ,,
    # Compare hardware parser
    compare_parser = subparsers.add_parser()))))))))))))"compare-hardware", help="Compare performance across hardware platforms")
    compare_parser.add_argument()))))))))))))"--model", default="bert-base-uncased", help="Model name")
    compare_parser.add_argument()))))))))))))"--type", default="text_embedding", help="Model type")
    compare_parser.add_argument()))))))))))))"--batch-size", type=int, default=4, help="Batch size")
    compare_parser.add_argument()))))))))))))"--precision", default="fp32", choices=["fp32", "fp16", "int8"], help="Precision format"),,
    ,,
    # Compare batch sizes parser
    batch_parser = subparsers.add_parser()))))))))))))"compare-batch-sizes", help="Compare performance across batch sizes")
    batch_parser.add_argument()))))))))))))"--model", default="bert-base-uncased", help="Model name")
    batch_parser.add_argument()))))))))))))"--type", default="text_embedding", help="Model type")
    batch_parser.add_argument()))))))))))))"--hardware", default="cuda", help="Hardware platform")
    batch_parser.add_argument()))))))))))))"--precision", default="fp32", choices=["fp32", "fp16", "int8"], help="Precision format"),,
    ,,batch_parser.add_argument()))))))))))))"--batch-sizes", default="1,2,4,8,16,32", help="Comma-separated list of batch sizes")
    
    # Recommend hardware parser
    recommend_hw_parser = subparsers.add_parser()))))))))))))"recommend-hardware", help="Recommend hardware for a model type")
    recommend_hw_parser.add_argument()))))))))))))"--type", default="text_embedding", help="Model type")
    recommend_hw_parser.add_argument()))))))))))))"--optimize-for", default="throughput", 
    choices=["throughput", "latency", "memory", "balanced"],
    help="Optimization goal for hardware recommendations")
    
    # Recommend benchmark configurations parser
    recommend_benchmark_parser = subparsers.add_parser()))))))))))))"recommend-benchmarks", 
    help="Recommend high-value benchmark configurations")
    recommend_benchmark_parser.add_argument()))))))))))))"--budget", type=int, default=10, 
    help="Number of configurations to recommend")
    recommend_benchmark_parser.add_argument()))))))))))))"--output", default="recommendations.json", 
    help="Output file for recommendations")
    
    # Integration parser (Active Learning + Hardware Recommender)
    integrate_parser = subparsers.add_parser()))))))))))))"integrate", 
    help="Integrate active learning with hardware recommender")
    integrate_parser.add_argument()))))))))))))"--budget", type=int, default=5, 
    help="Number of configurations to recommend")
    integrate_parser.add_argument()))))))))))))"--metric", default="throughput", 
    choices=["throughput", "latency", "memory"], 
    help="Metric to optimize for")
    integrate_parser.add_argument()))))))))))))"--output", default="integrated_recommendations.json", 
    help="Output file for integrated recommendations")
    
    # Schedule benchmarks parser
    schedule_parser = subparsers.add_parser()))))))))))))"schedule-benchmarks", 
    help="Schedule benchmarks based on recommendations")
    schedule_parser.add_argument()))))))))))))"--recommendations", required=True, 
    help="Recommendations file to use")
    schedule_parser.add_argument()))))))))))))"--execute", action="store_true", 
    help="Execute the benchmarks")
    schedule_parser.add_argument()))))))))))))"--db-path", 
    help="Path to benchmark database")
    
    # Demo parser
    demo_parser = subparsers.add_parser()))))))))))))"demo", help="Run a demonstration")
    demo_parser.add_argument()))))))))))))"--model", default="bert-base-uncased", help="Model name to use for predictions")
    demo_parser.add_argument()))))))))))))"--type", default="text_embedding", help="Model type")
    demo_parser.add_argument()))))))))))))"--hardware", default="cuda", help="Hardware platform")
    demo_parser.add_argument()))))))))))))"--batch-size", type=int, default=4, help="Batch size")
    demo_parser.add_argument()))))))))))))"--precision", default="fp32", choices=["fp32", "fp16", "int8"], help="Precision format"),,
    ,,demo_parser.add_argument()))))))))))))"--quick", action="store_true", help="Run a quick demonstration")
    
    # Parse arguments
    args = parser.parse_args())))))))))))))
    
    # Set up the database path if environment variable is not set:
    if not os.environ.get()))))))))))))"BENCHMARK_DB_PATH"):
        benchmark_db_path = str()))))))))))))Path()))))))))))))__file__).parent.parent / "benchmark_db.duckdb")
        os.environ["BENCHMARK_DB_PATH"] = benchmark_db_path,
        print()))))))))))))f"Setting BENCHMARK_DB_PATH to: {}}}benchmark_db_path}")
    
    # Execute the requested mode
    if args.mode == "predict":
        predict_single_configuration()))))))))))))
        args.model, args.type, args.hardware, args.batch_size, args.precision
        )
    
    elif args.mode == "compare-hardware":
        compare_multiple_hardware()))))))))))))
        args.model, args.type, args.batch_size, args.precision
        )
    
    elif args.mode == "compare-batch-sizes":
        batch_sizes = [int()))))))))))))bs.strip())))))))))))))) for bs in args.batch_sizes.split()))))))))))))",")]:,
        generate_batch_size_comparison()))))))))))))
        args.model, args.type, args.hardware, batch_sizes, args.precision
        )
    
    elif args.mode == "recommend-hardware":
        recommend_optimal_hardware()))))))))))))
        args.type, optimize_for=args.optimize_for
        )
    
    elif args.mode == "recommend-benchmarks":
        recommend_benchmark_configurations()))))))))))))
        budget=args.budget, output_file=args.output
        )
    
    elif args.mode == "schedule-benchmarks":
        schedule_benchmarks()))))))))))))
        args.recommendations, execute=args.execute, db_path=args.db_path
        
    elif args.mode == "integrate":
        # Run integration example with active learning and hardware recommender
        integration_example(args)
    
    elif args.mode == "demo" or not args.mode:
        # Run a demonstration of the predictive performance system
        print()))))))))))))"Running demonstration of the Predictive Performance System\n")
        
        if args.quick:
            # Quick demo - just show basic prediction
            predict_single_configuration()))))))))))))args.model, args.type, args.hardware, args.batch_size, args.precision)
        else:
            # Full demo - show all features
            predict_single_configuration()))))))))))))args.model, args.type, args.hardware, args.batch_size, args.precision)
            compare_multiple_hardware()))))))))))))args.model, args.type, args.batch_size, args.precision)
            
            batch_sizes = [1, 4, 16],
            generate_batch_size_comparison()))))))))))))
            args.model, args.type, args.hardware, batch_sizes, args.precision
            )
            
            recommend_optimal_hardware()))))))))))))args.type)
            
            # Generate recommendations
            recommendations = recommend_benchmark_configurations()))))))))))))
            budget=5, output_file="demo_recommendations.json"
            )
            
            # Run integration example
            print()))))))))))))"\nRunning integration example with active learning and hardware recommender...")
            try:
                # Set demo parameters
                args.budget = 3
                args.metric = "throughput"
                args.output = "demo_integrated_recommendations.json"
                integration_example(args)
            except Exception as e:
                print()))))))))))))f"Error running integration example: {)}))))))))))))e}")
                print())))))))))))))"You can run the integration example directly with:")
                print())))))))))))))"python example.py integrate --budget 5 --metric throughput")
            
            # Show how to schedule benchmarks
            print()))))))))))))"\nYou can schedule benchmarks with:")
            print()))))))))))))"python example.py schedule-benchmarks --recommendations demo_recommendations.json")
    
    else:
        parser.print_help())))))))))))))


def integration_example(args):
    """Run integration example to demonstrate active learning with hardware recommender."""
    print("\n===== Integration Example: Active Learning with Hardware Recommender =====")
    
    try:
        # Import the active learning system
        from active_learning import ActiveLearningSystem
        
        # Import the hardware recommender
        from hardware_recommender import HardwareRecommender
        
        # Import the prediction system for the hardware recommender
        from predict import PerformancePredictor
        
        # Initialize components
        predictor = PerformancePredictor()
        active_learner = ActiveLearningSystem()
        hw_recommender = HardwareRecommender(
            predictor=predictor,
            available_hardware=["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"],
            confidence_threshold=0.7
        )
        
        # Run integrated recommendation
        print(f"Generating integrated recommendations (budget: {args.budget}, metric: {args.metric})...")
        integrated_results = active_learner.integrate_with_hardware_recommender(
            hardware_recommender=hw_recommender,
            test_budget=args.budget,
            optimize_for=args.metric
        )
        
        # Print results
        print(f"\nGenerated {len(integrated_results['recommendations'])} integrated recommendations")
        print(f"Total Candidates: {integrated_results['total_candidates']}")
        print(f"Enhanced Candidates: {integrated_results['enhanced_candidates']}")
        
        print("\nRecommended Configurations:")
        for i, config in enumerate(integrated_results["recommendations"]):
            print(f"\nConfiguration #{i+1}:")
            print(f"  - Model: {config['model_name']}")
            print(f"  - Current Hardware: {config['hardware']}")
            print(f"  - Recommended Hardware: {config.get('recommended_hardware', 'N/A')}")
            print(f"  - Hardware Match: {config.get('hardware_match', False)}")
            print(f"  - Batch Size: {config['batch_size']}")
            print(f"  - Information Gain: {config.get('expected_information_gain', 0):.4f}")
            print(f"  - Combined Score: {config.get('combined_score', 0):.4f}")
        
        # Save results if output file specified
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(integrated_results, f, indent=2)
            print(f"\nSaved integrated results to {args.output}")
            
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure active_learning.py and hardware_recommender.py are in the same directory.")
    except Exception as e:
        print(f"Error running integration example: {e}")

if __name__ == "__main__":
    main())))))))))))))