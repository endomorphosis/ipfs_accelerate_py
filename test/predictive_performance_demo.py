#!/usr/bin/env python3
"""
Predictive Performance System Demo for the IPFS Accelerate framework.

This script demonstrates how to use the enhanced Predictive Performance System
to make hardware recommendations and predict performance metrics for various models
across different hardware platforms.

Usage:
    python predictive_performance_demo.py --train
    python predictive_performance_demo.py --predict-all
    python predictive_performance_demo.py --compare
    """

    import os
    import sys
    import json
    import time
    import argparse
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from typing import Dict, List, Any, Optional, Tuple, Union
    from datetime import datetime

# Import the core modules
    from model_performance_predictor import ())))))))
    load_benchmark_data,
    preprocess_data,
    train_prediction_models,
    save_prediction_models,
    load_prediction_models,
    predict_performance,
    generate_prediction_matrix,
    visualize_predictions,
    PREDICTION_METRICS,
    MODEL_CATEGORIES,
    HARDWARE_CATEGORIES
    )

# Configure paths
    PROJECT_ROOT = Path())))))))os.path.dirname())))))))os.path.abspath())))))))__file__)))
    BENCHMARK_DIR = PROJECT_ROOT / "benchmark_results"
    DEMO_OUTPUT_DIR = PROJECT_ROOT / "predictive_performance_demo_output"
    DEMO_OUTPUT_DIR.mkdir())))))))exist_ok=True, parents=True)

# Configure model and hardware test cases
    TEST_MODELS = []],,
    {}}}}"name": "bert-base-uncased", "category": "text_embedding"},
    {}}}}"name": "t5-small", "category": "text_generation"},
    {}}}}"name": "facebook/opt-125m", "category": "text_generation"},
    {}}}}"name": "openai/whisper-tiny", "category": "audio"},
    {}}}}"name": "google/vit-base-patch16-224", "category": "vision"},
    {}}}}"name": "openai/clip-vit-base-patch32", "category": "multimodal"}
    ]

    TEST_HARDWARE = []],,"cpu", "cuda", "mps", "openvino", "webgpu"]
    TEST_BATCH_SIZES = []],,1, 8, 32]
    TEST_PRECISIONS = []],,"fp32", "fp16"]

def print_header())))))))title: str):
    """Print a formatted header."""
    print())))))))"\n" + "=" * 80)
    print())))))))f" {}}}}title} ".center())))))))80, "="))
    print())))))))"=" * 80 + "\n")

def train_demo_models())))))))):
    """Train the predictive performance models for the demo."""
    print_header())))))))"Training Predictive Performance Models")
    
    # Load benchmark data
    db_path = os.environ.get())))))))"BENCHMARK_DB_PATH", str())))))))BENCHMARK_DIR / "benchmark_db.duckdb"))
    print())))))))f"Loading benchmark data from {}}}}db_path}...")
    df = load_benchmark_data())))))))db_path)
    
    if df.empty:
        print())))))))"No benchmark data available. Please run benchmarks first or check database path.")
        sys.exit())))))))1)
    
        print())))))))f"Loaded {}}}}len())))))))df)} benchmark records")
    
    # Preprocess data
        print())))))))"Preprocessing benchmark data...")
        df, preprocessing_info = preprocess_data())))))))df)
    
    if df.empty:
        print())))))))"Error preprocessing data")
        sys.exit())))))))1)
    
    # Train models with advanced features
        print())))))))"Training prediction models with advanced features...")
        start_time = time.time()))))))))
    
        models = train_prediction_models())))))))
        df, 
        preprocessing_info,
        test_size=0.2,
        random_state=42,
        hyperparameter_tuning=True,
        use_ensemble=True
        )
    
        training_time = time.time())))))))) - start_time
    
    if not models:
        print())))))))"Error training models")
        sys.exit())))))))1)
    
    # Print model metrics
        print())))))))f"\nTraining completed in {}}}}training_time:.2f} seconds")
    
    for target in PREDICTION_METRICS:
        if target in models:
            metrics = models[]],,target].get())))))))"metrics", {}}}}})
            print())))))))f"\nModel metrics for {}}}}target}:")
            print())))))))f"  Test R²: {}}}}metrics.get())))))))'test_r2', 'N/A'):.4f}")
            print())))))))f"  MAPE: {}}}}metrics.get())))))))'mape', 'N/A'):.2%}")
            print())))))))f"  RMSE: {}}}}metrics.get())))))))'rmse', 'N/A'):.4f}")
            
            # Print top feature importances if available:
            if "feature_importance" in metrics and isinstance())))))))metrics[]],,"feature_importance"], dict):
                importances = metrics[]],,"feature_importance"]
                print())))))))"  Top feature importances:")
                sorted_features = sorted())))))))importances.items())))))))), key=lambda x: x[]],,1], reverse=True)[]],,:5]
                for feature, importance in sorted_features:
                    print())))))))f"    {}}}}feature}: {}}}}importance:.4f}")
    
    # Save models
                    output_dir = DEMO_OUTPUT_DIR / "models"
                    model_dir = save_prediction_models())))))))models, str())))))))output_dir))
    
    if not model_dir:
        print())))))))"Error saving models")
        sys.exit())))))))1)
    
        print())))))))f"\nTrained prediction models saved to {}}}}model_dir}")
    
                    return models

def predict_all_combinations())))))))models: Dict[]],,str, Any]):
    """Predict performance for all test combinations."""
    print_header())))))))"Predicting Performance for All Test Combinations")
    
    # Create results container
    results = {}}}}}
    
    # Track time
    start_time = time.time()))))))))
    
    # Make predictions for all combinations
    total_combinations = len())))))))TEST_MODELS) * len())))))))TEST_HARDWARE) * len())))))))TEST_BATCH_SIZES) * len())))))))TEST_PRECISIONS)
    print())))))))f"Making {}}}}total_combinations} predictions...")
    
    for model_info in TEST_MODELS:
        model_name = model_info[]],,"name"]
        model_category = model_info[]],,"category"]
        model_short_name = model_name.split())))))))"/")[]],,-1]
        
        results[]],,model_short_name] = {}}}}}
        
        for hardware in TEST_HARDWARE:
            results[]],,model_short_name][]],,hardware] = {}}}}}
            
            for batch_size in TEST_BATCH_SIZES:
                results[]],,model_short_name][]],,hardware][]],,batch_size] = {}}}}}
                
                for precision in TEST_PRECISIONS:
                    # Skip incompatible combinations
                    if precision == "fp16" and hardware == "cpu":
                    continue
                    
                    # Make prediction
                    prediction = predict_performance())))))))
                    models=models,
                    model_name=model_name,
                    model_category=model_category,
                    hardware=hardware,
                    batch_size=batch_size,
                    precision=precision,
                    mode="inference",
                    calculate_uncertainty=True
                    )
                    
                    # Store prediction
                    if prediction:
                        results[]],,model_short_name][]],,hardware][]],,batch_size][]],,precision] = prediction
    
                        prediction_time = time.time())))))))) - start_time
                        print())))))))f"Completed predictions in {}}}}prediction_time:.2f} seconds")
    
    # Save results
                        output_file = DEMO_OUTPUT_DIR / "prediction_results.json"
    with open())))))))output_file, "w") as f:
        json.dump())))))))results, f, indent=2)
    
        print())))))))f"Saved predictions to {}}}}output_file}")
    
    # Print some sample results
        print())))))))"\nSample prediction results:")
    
    for model_short_name in list())))))))results.keys())))))))))[]],,:2]:
        for hardware in list())))))))results[]],,model_short_name].keys())))))))))[]],,:2]:
            batch_size = TEST_BATCH_SIZES[]],,0]
            precision = TEST_PRECISIONS[]],,0]
            
            if precision in results[]],,model_short_name][]],,hardware][]],,batch_size]:
                prediction = results[]],,model_short_name][]],,hardware][]],,batch_size][]],,precision]
                
                print())))))))f"\nModel: {}}}}model_short_name}, Hardware: {}}}}hardware}, Batch Size: {}}}}batch_size}, Precision: {}}}}precision}")
                
                for metric in PREDICTION_METRICS:
                    if metric in prediction.get())))))))"predictions", {}}}}}):
                        value = prediction[]],,"predictions"][]],,metric]
                        
                        if metric in prediction.get())))))))"uncertainties", {}}}}}):
                            uncertainty = prediction[]],,"uncertainties"][]],,metric]
                            confidence = uncertainty.get())))))))"confidence", 0.0) * 100
                            
                            print())))))))f"  {}}}}metric}: {}}}}value:.2f} ())))))))confidence: {}}}}confidence:.1f}%)")
                            print())))))))f"    Range: {}}}}uncertainty.get())))))))'lower_bound', 0.0):.2f} - {}}}}uncertainty.get())))))))'upper_bound', 0.0):.2f}")
                        else:
                            print())))))))f"  {}}}}metric}: {}}}}value:.2f}")
                
                            print())))))))f"  Overall Confidence: {}}}}prediction.get())))))))'confidence_score', 0.0) * 100:.1f}%")
    
                            return results

def generate_comparison_visuals())))))))results: Dict[]],,str, Any]):
    """Generate comparison visualizations from prediction results."""
    print_header())))))))"Generating Comparison Visualizations")
    
    # Create output directory
    vis_dir = DEMO_OUTPUT_DIR / "visualizations"
    vis_dir.mkdir())))))))exist_ok=True, parents=True)
    
    # Set plot style
    sns.set_style())))))))"whitegrid")
    plt.rcParams[]],,"figure.figsize"] = ())))))))12, 8)
    
    # 1. Hardware comparison for each model ())))))))throughput)
    print())))))))"Generating hardware comparison charts...")
    
    for model_short_name in results:
        data = []],,]
        
        for hardware in results[]],,model_short_name]:
            batch_size = 8  # Use fixed batch size for this comparison
            
            if batch_size in results[]],,model_short_name][]],,hardware]:
                for precision in results[]],,model_short_name][]],,hardware][]],,batch_size]:
                    prediction = results[]],,model_short_name][]],,hardware][]],,batch_size][]],,precision]
                    
                    if "predictions" in prediction and "throughput" in prediction[]],,"predictions"]:
                        throughput = prediction[]],,"predictions"][]],,"throughput"]
                        
                        # Get confidence if available:
                        confidence = 1.0
                        if "uncertainties" in prediction and "throughput" in prediction[]],,"uncertainties"]:
                            confidence = prediction[]],,"uncertainties"][]],,"throughput"].get())))))))"confidence", 1.0)
                        
                            data.append()))))))){}}}}
                            "hardware": hardware,
                            "precision": precision,
                            "throughput": throughput,
                            "confidence": confidence
                            })
        
        if data:
            # Create DataFrame
            df = pd.DataFrame())))))))data)
            
            # Create plot
            plt.figure()))))))))
            
            # Use confidence for alpha
            sns.barplot())))))))x="hardware", y="throughput", hue="precision", data=df,
            alpha=df[]],,"confidence"].values)
            
            plt.title())))))))f"Throughput Comparison for {}}}}model_short_name} ())))))))Batch Size = 8)")
            plt.xlabel())))))))"Hardware")
            plt.ylabel())))))))"Throughput ())))))))samples/sec)")
            plt.xticks())))))))rotation=45)
            plt.legend())))))))title="Precision")
            plt.tight_layout()))))))))
            
            # Save plot
            output_file = vis_dir / f"{}}}}model_short_name}_hardware_comparison.png"
            plt.savefig())))))))output_file)
            plt.close()))))))))
    
    # 2. Batch size scaling for each model and hardware ())))))))throughput)
            print())))))))"Generating batch size scaling charts...")
    
    for model_short_name in results:
        for hardware in results[]],,model_short_name]:
            data = []],,]
            
            for batch_size in results[]],,model_short_name][]],,hardware]:
                for precision in results[]],,model_short_name][]],,hardware][]],,batch_size]:
                    prediction = results[]],,model_short_name][]],,hardware][]],,batch_size][]],,precision]
                    
                    if "predictions" in prediction and "throughput" in prediction[]],,"predictions"]:
                        throughput = prediction[]],,"predictions"][]],,"throughput"]
                        
                        data.append()))))))){}}}}
                        "batch_size": batch_size,
                        "precision": precision,
                        "throughput": throughput
                        })
            
            if data:
                # Create DataFrame
                df = pd.DataFrame())))))))data)
                
                # Create plot
                plt.figure()))))))))
                
                # Create line plot
                for precision in df[]],,"precision"].unique())))))))):
                    df_precision = df[]],,df[]],,"precision"] == precision].sort_values())))))))"batch_size")
                    plt.plot())))))))df_precision[]],,"batch_size"], df_precision[]],,"throughput"], marker='o', label=precision)
                
                    plt.title())))))))f"Throughput Scaling with Batch Size for {}}}}model_short_name} on {}}}}hardware}")
                    plt.xlabel())))))))"Batch Size")
                    plt.ylabel())))))))"Throughput ())))))))samples/sec)")
                    plt.legend())))))))title="Precision")
                    plt.grid())))))))True)
                    plt.tight_layout()))))))))
                
                # Save plot
                    output_file = vis_dir / f"{}}}}model_short_name}_{}}}}hardware}_batch_scaling.png"
                    plt.savefig())))))))output_file)
                    plt.close()))))))))
    
    # 3. Model comparison across hardware ())))))))latency)
                    print())))))))"Generating model comparison charts...")
    
    for hardware in TEST_HARDWARE:
        data = []],,]
        
        for model_short_name in results:
            if hardware in results[]],,model_short_name]:
                batch_size = 1  # Use batch size 1 for latency comparison
                
                if batch_size in results[]],,model_short_name][]],,hardware]:
                    precision = "fp32"  # Use fp32 for consistent comparison
                    
                    if precision in results[]],,model_short_name][]],,hardware][]],,batch_size]:
                        prediction = results[]],,model_short_name][]],,hardware][]],,batch_size][]],,precision]
                        
                        if "predictions" in prediction and "latency_mean" in prediction[]],,"predictions"]:
                            latency = prediction[]],,"predictions"][]],,"latency_mean"]
                            
                            data.append()))))))){}}}}
                            "model": model_short_name,
                            "latency": latency
                            })
        
        if data:
            # Create DataFrame
            df = pd.DataFrame())))))))data)
            
            # Create plot
            plt.figure()))))))))
            
            # Sort by latency
            df = df.sort_values())))))))"latency")
            
            # Create bar plot
            sns.barplot())))))))x="model", y="latency", data=df)
            
            plt.title())))))))f"Latency Comparison for Models on {}}}}hardware}")
            plt.xlabel())))))))"Model")
            plt.ylabel())))))))"Latency ())))))))ms)")
            plt.xticks())))))))rotation=45, ha="right")
            plt.tight_layout()))))))))
            
            # Save plot
            output_file = vis_dir / f"{}}}}hardware}_model_latency_comparison.png"
            plt.savefig())))))))output_file)
            plt.close()))))))))
    
    # 4. Generate uncertainty visualization for one model
            print())))))))"Generating uncertainty visualization...")
    
            model_short_name = list())))))))results.keys())))))))))[]],,0]
            hardware = "cuda" if "cuda" in results[]],,model_short_name] else list())))))))results[]],,model_short_name].keys())))))))))[]],,0]
    
            data = []],,]
    :
    for batch_size in results[]],,model_short_name][]],,hardware]:
        for precision in results[]],,model_short_name][]],,hardware][]],,batch_size]:
            prediction = results[]],,model_short_name][]],,hardware][]],,batch_size][]],,precision]
            
            if "predictions" in prediction and "throughput" in prediction[]],,"predictions"]:
                throughput = prediction[]],,"predictions"][]],,"throughput"]
                
                # Get uncertainty if available:
                lower_bound = throughput * 0.85
                upper_bound = throughput * 1.15
                
                if "uncertainties" in prediction and "throughput" in prediction[]],,"uncertainties"]:
                    uncertainty = prediction[]],,"uncertainties"][]],,"throughput"]
                    lower_bound = uncertainty.get())))))))"lower_bound", lower_bound)
                    upper_bound = uncertainty.get())))))))"upper_bound", upper_bound)
                
                    data.append()))))))){}}}}
                    "batch_size": batch_size,
                    "precision": precision,
                    "throughput": throughput,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound
                    })
    
    if data:
        # Create DataFrame
        df = pd.DataFrame())))))))data)
        
        # Create plot
        plt.figure()))))))))
        
        # Plot with error bars for uncertainty
        for precision in df[]],,"precision"].unique())))))))):
            df_precision = df[]],,df[]],,"precision"] == precision].sort_values())))))))"batch_size")
            plt.errorbar())))))))
            df_precision[]],,"batch_size"],
            df_precision[]],,"throughput"],
            yerr=[]],,
            df_precision[]],,"throughput"] - df_precision[]],,"lower_bound"],
            df_precision[]],,"upper_bound"] - df_precision[]],,"throughput"]
            ],
            marker='o',
            label=precision,
            capsize=5
            )
        
            plt.title())))))))f"Throughput with Uncertainty for {}}}}model_short_name} on {}}}}hardware}")
            plt.xlabel())))))))"Batch Size")
            plt.ylabel())))))))"Throughput ())))))))samples/sec)")
            plt.legend())))))))title="Precision")
            plt.grid())))))))True)
            plt.tight_layout()))))))))
        
        # Save plot
            output_file = vis_dir / f"{}}}}model_short_name}_{}}}}hardware}_uncertainty.png"
            plt.savefig())))))))output_file)
            plt.close()))))))))
    
    # 5. Generate comprehensive matrix
            print())))))))"Generating comprehensive performance matrix...")
    
    # Create directory for full matrix file
            matrix_dir = DEMO_OUTPUT_DIR / "matrix"
            matrix_dir.mkdir())))))))exist_ok=True, parents=True)
    
    # Load models
            models_dir = DEMO_OUTPUT_DIR / "models" / "latest"
            models = load_prediction_models())))))))str())))))))models_dir))
    
    if not models:
        print())))))))"Error loading models for matrix generation")
            return
    
    # Generate matrix
            matrix = generate_prediction_matrix())))))))
            models=models,
            model_configs=TEST_MODELS,
            hardware_platforms=TEST_HARDWARE,
            batch_sizes=TEST_BATCH_SIZES,
            precision_options=TEST_PRECISIONS,
            mode="inference",
            output_file=str())))))))matrix_dir / "prediction_matrix.json")
            )
    
    if not matrix:
        print())))))))"Error generating prediction matrix")
            return
    
    # Generate visualizations
            visualization_files = visualize_predictions())))))))
            matrix=matrix,
            metric="throughput",
            output_dir=str())))))))vis_dir)
            )
    
            visualization_files.extend())))))))visualize_predictions())))))))
            matrix=matrix,
            metric="latency_mean",
            output_dir=str())))))))vis_dir)
            ))
    
            visualization_files.extend())))))))visualize_predictions())))))))
            matrix=matrix,
            metric="memory_usage",
            output_dir=str())))))))vis_dir)
            ))
    
            print())))))))f"Generated {}}}}len())))))))visualization_files)} visualizations")
            print())))))))"\nVisualization files:")
    for file in visualization_files:
        print())))))))f"  - {}}}}file}")
    
        print())))))))f"\nAll visualizations saved to {}}}}vis_dir}")

def main())))))))):
    """Main function."""
    parser = argparse.ArgumentParser())))))))description="Predictive Performance System Demo")
    
    group = parser.add_mutually_exclusive_group())))))))required=True)
    group.add_argument())))))))"--train", action="store_true", help="Train prediction models")
    group.add_argument())))))))"--predict-all", action="store_true", help="Make predictions for all test combinations")
    group.add_argument())))))))"--compare", action="store_true", help="Generate comparison visualizations")
    group.add_argument())))))))"--full-demo", action="store_true", help="Run full demonstration ())))))))train, predict, visualize)")
    
    parser.add_argument())))))))"--output-dir", help="Directory to save output files")
    parser.add_argument())))))))"--db-path", help="Path to benchmark database")
    
    args = parser.parse_args()))))))))
    
    # Set output directory if specified::
    if args.output_dir:
        global DEMO_OUTPUT_DIR
        DEMO_OUTPUT_DIR = Path())))))))args.output_dir)
        DEMO_OUTPUT_DIR.mkdir())))))))exist_ok=True, parents=True)
    
    # Set database path if specified::
    if args.db_path:
        os.environ[]],,"BENCHMARK_DB_PATH"] = args.db_path
    
    if args.train or args.full_demo:
        models = train_demo_models()))))))))
    else:
        # Load models
        models_dir = DEMO_OUTPUT_DIR / "models" / "latest"
        models = load_prediction_models())))))))str())))))))models_dir))
        
        if not models:
            print())))))))"Error loading models. Please run with --train first.")
            sys.exit())))))))1)
    
    if args.predict_all or args.full_demo:
        results = predict_all_combinations())))))))models)
    else:
        # Load results
        results_file = DEMO_OUTPUT_DIR / "prediction_results.json"
        
        if not results_file.exists())))))))):
            print())))))))"No prediction results found. Please run with --predict-all first.")
            sys.exit())))))))1)
            
        with open())))))))results_file, "r") as f:
            results = json.load())))))))f)
    
    if args.compare or args.full_demo:
        generate_comparison_visuals())))))))results)
    
    if args.full_demo:
        print_header())))))))"Full Demonstration Completed")
        print())))))))f"All demonstration files saved to {}}}}DEMO_OUTPUT_DIR}")
        print())))))))"\nTo explore the results, check:")
        print())))))))f"  - Models: {}}}}DEMO_OUTPUT_DIR / 'models'}")
        print())))))))f"  - Prediction results: {}}}}DEMO_OUTPUT_DIR / 'prediction_results.json'}")
        print())))))))f"  - Visualizations: {}}}}DEMO_OUTPUT_DIR / 'visualizations'}")
        print())))))))f"  - Performance matrix: {}}}}DEMO_OUTPUT_DIR / 'matrix' / 'prediction_matrix.json'}")

if __name__ == "__main__":
    main()))))))))