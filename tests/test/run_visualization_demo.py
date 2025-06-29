#!/usr/bin/env python3
"""
Visualization Demo for the Predictive Performance System.

This script demonstrates how to use the advanced visualization capabilities
of the Predictive Performance System to create comprehensive visualizations
for model performance data.

Usage:
    python run_visualization_demo.py --data prediction_results.json
    python run_visualization_demo.py --demo
    python run_visualization_demo.py --generate
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Import visualization module
from predictive_performance.visualization import AdvancedVisualization, create_visualization_report

# Import performance prediction
try:
    from predictive_performance.predict import PerformancePredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False

# Define constants
DEMO_OUTPUT_DIR = Path("./visualization_demo_output")
DEFAULT_METRICS = ["throughput", "latency_mean", "memory_usage"]
DEFAULT_TEST_MODELS = [
    {"name": "bert-base-uncased", "category": "text_embedding"},
    {"name": "t5-small", "category": "text_generation"},
    {"name": "facebook/opt-125m", "category": "text_generation"},
    {"name": "openai/whisper-tiny", "category": "audio"},
    {"name": "google/vit-base-patch16-224", "category": "vision"},
    {"name": "openai/clip-vit-base-patch32", "category": "multimodal"}
]
DEFAULT_TEST_HARDWARE = ["cpu", "cuda", "mps", "openvino", "webgpu"]
DEFAULT_TEST_BATCH_SIZES = [1, 4, 8, 16, 32]
DEFAULT_TEST_PRECISIONS = ["fp32", "fp16"]

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def generate_sample_data():
    """Generate sample performance data for visualization demos."""
    print_header("Generating Sample Performance Data")
    
    # Create output directory
    DEMO_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Generate sample data
    data = []
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate timestamps for time-series data (past 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(days=i) for i in range(31)]
    
    print(f"Generating data for {len(DEFAULT_TEST_MODELS)} models, {len(DEFAULT_TEST_HARDWARE)} hardware platforms, {len(DEFAULT_TEST_BATCH_SIZES)} batch sizes, and {len(DEFAULT_TEST_PRECISIONS)} precision formats...")
    
    # Generate data for each combination
    for model_info in DEFAULT_TEST_MODELS:
        model_name = model_info["name"]
        model_category = model_info["category"]
        model_short_name = model_name.split("/")[-1]
        
        for hardware in DEFAULT_TEST_HARDWARE:
            # Skip incompatible combinations
            if hardware == "webgpu" and model_category == "audio":
                continue
                
            for batch_size in DEFAULT_TEST_BATCH_SIZES:
                for precision in DEFAULT_TEST_PRECISIONS:
                    # Skip incompatible combinations
                    if precision == "fp16" and hardware == "cpu":
                        continue
                        
                    # Base performance values (realistic scales)
                    # These will be modified by hardware, batch size, precision, and model type
                    base_throughput = 100.0
                    base_latency = 10.0
                    base_memory = 1000.0
                    base_power = 50.0
                    
                    # Hardware factors
                    hw_factors = {
                        "cpu": {"throughput": 1.0, "latency": 2.0, "memory": 1.0, "power": 1.0},
                        "cuda": {"throughput": 5.0, "latency": 0.5, "memory": 1.2, "power": 3.0},
                        "mps": {"throughput": 3.0, "latency": 0.7, "memory": 1.1, "power": 2.0},
                        "openvino": {"throughput": 2.0, "latency": 1.0, "memory": 0.8, "power": 1.5},
                        "webgpu": {"throughput": 2.5, "latency": 0.8, "memory": 1.3, "power": 2.0}
                    }
                    
                    # Model category factors
                    category_factors = {
                        "text_embedding": {"throughput": 1.2, "latency": 0.8, "memory": 0.8, "power": 0.9},
                        "text_generation": {"throughput": 0.6, "latency": 1.5, "memory": 1.3, "power": 1.2},
                        "vision": {"throughput": 0.8, "latency": 1.2, "memory": 1.5, "power": 1.1},
                        "audio": {"throughput": 0.5, "latency": 1.8, "memory": 1.2, "power": 1.3},
                        "multimodal": {"throughput": 0.4, "latency": 2.0, "memory": 1.8, "power": 1.4}
                    }
                    
                    # Precision factors
                    precision_factors = {
                        "fp32": {"throughput": 1.0, "latency": 1.0, "memory": 1.0, "power": 1.0},
                        "fp16": {"throughput": 1.5, "latency": 0.7, "memory": 0.6, "power": 0.8}
                    }
                    
                    # Batch size scaling (non-linear)
                    # Throughput increases sub-linearly with batch size
                    # Latency increases slightly with batch size
                    # Memory increases linearly with batch size
                    throughput_batch_factor = np.sqrt(batch_size)
                    latency_batch_factor = 1.0 + np.log(batch_size) * 0.1
                    memory_batch_factor = batch_size
                    power_batch_factor = 1.0 + np.log(batch_size) * 0.2
                    
                    # Calculate performance metrics with some randomness
                    hw_factor = hw_factors[hardware]
                    cat_factor = category_factors[model_category]
                    prec_factor = precision_factors[precision]
                    
                    # Calculate throughput with batch effect and randomness
                    throughput = (
                        base_throughput *
                        hw_factor["throughput"] *
                        cat_factor["throughput"] *
                        prec_factor["throughput"] *
                        throughput_batch_factor *
                        (1.0 + np.random.normal(0, 0.1))  # Add 10% random noise
                    )
                    
                    # Calculate latency with batch effect and randomness
                    latency = (
                        base_latency *
                        hw_factor["latency"] *
                        cat_factor["latency"] *
                        prec_factor["latency"] *
                        latency_batch_factor *
                        (1.0 + np.random.normal(0, 0.1))  # Add 10% random noise
                    )
                    
                    # Calculate memory with batch effect and randomness
                    memory = (
                        base_memory *
                        hw_factor["memory"] *
                        cat_factor["memory"] *
                        prec_factor["memory"] *
                        memory_batch_factor *
                        (1.0 + np.random.normal(0, 0.05))  # Add 5% random noise
                    )
                    
                    # Calculate power consumption with batch effect and randomness
                    power = (
                        base_power *
                        hw_factor["power"] *
                        cat_factor["power"] *
                        prec_factor["power"] *
                        power_batch_factor *
                        (1.0 + np.random.normal(0, 0.1))  # Add 10% random noise
                    )
                    
                    # Calculate confidence scores (higher for common combinations)
                    confidence_base = 0.85
                    
                    # Adjust confidence based on hardware
                    hw_confidence = {
                        "cpu": 0.95,
                        "cuda": 0.9,
                        "mps": 0.85,
                        "openvino": 0.8,
                        "webgpu": 0.75
                    }
                    
                    # Adjust confidence based on model category
                    category_confidence = {
                        "text_embedding": 0.95,
                        "text_generation": 0.9,
                        "vision": 0.85,
                        "audio": 0.8,
                        "multimodal": 0.75
                    }
                    
                    # Calculate confidence
                    confidence = min(
                        0.98,
                        confidence_base *
                        hw_confidence[hardware] *
                        category_confidence[model_category] *
                        (1.0 + np.random.normal(0, 0.05))  # Add 5% random noise
                    )
                    
                    # Calculate bounds for uncertainty visualization
                    throughput_lower = throughput * (1.0 - (1.0 - confidence) * 2)
                    throughput_upper = throughput * (1.0 + (1.0 - confidence) * 2)
                    
                    latency_lower = latency * (1.0 - (1.0 - confidence) * 2)
                    latency_upper = latency * (1.0 + (1.0 - confidence) * 2)
                    
                    memory_lower = memory * (1.0 - (1.0 - confidence) * 2)
                    memory_upper = memory * (1.0 + (1.0 - confidence) * 2)
                    
                    # Generate time-series data for this combination
                    for timestamp in timestamps:
                        # Add time trend (+/- 20% over time with sine wave pattern)
                        time_position = timestamps.index(timestamp) / len(timestamps)
                        time_factor = 1.0 + 0.2 * np.sin(time_position * 2 * np.pi)
                        
                        # Add record for this timestamp
                        data.append({
                            "model_name": model_short_name,
                            "model_category": model_category,
                            "hardware": hardware,
                            "batch_size": batch_size,
                            "precision": precision,
                            "throughput": throughput * time_factor,
                            "latency_mean": latency * time_factor,
                            "memory_usage": memory * time_factor,
                            "power_consumption": power * time_factor,
                            "timestamp": timestamp.isoformat(),
                            "confidence": confidence,
                            "throughput_lower_bound": throughput_lower * time_factor,
                            "throughput_upper_bound": throughput_upper * time_factor,
                            "latency_lower_bound": latency_lower * time_factor,
                            "latency_upper_bound": latency_upper * time_factor,
                            "memory_lower_bound": memory_lower * time_factor,
                            "memory_upper_bound": memory_upper * time_factor
                        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV and JSON
    csv_path = DEMO_OUTPUT_DIR / "sample_performance_data.csv"
    json_path = DEMO_OUTPUT_DIR / "sample_performance_data.json"
    
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    
    print(f"Generated {len(df)} sample data points")
    print(f"Data saved to:")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {json_path}")
    
    return df, json_path

def run_visualization_demo(data_path=None, advanced_vis=False):
    """Run visualization demo using sample or provided data."""
    print_header("Running Advanced Visualization Demo")
    
    # Create output directory
    vis_dir = DEMO_OUTPUT_DIR / "visualizations"
    vis_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate sample data if not provided
    if data_path is None:
        df, data_path = generate_sample_data()
    else:
        # Load provided data
        data_path = Path(data_path)
        if not data_path.exists():
            print(f"Error: Data file {data_path} not found")
            sys.exit(1)
            
        if data_path.suffix.lower() == ".json":
            with open(data_path, "r") as f:
                df = pd.DataFrame(json.load(f))
        elif data_path.suffix.lower() == ".csv":
            df = pd.read_csv(data_path)
        else:
            print(f"Error: Unsupported file format {data_path.suffix}")
            sys.exit(1)
    
    print(f"Using data from {data_path}")
    print(f"Data contains {len(df)} records")
    
    # Create visualization system
    print("Initializing visualization system...")
    vis = AdvancedVisualization(
        output_dir=str(vis_dir),
        interactive=True
    )
    
    # Create batch visualizations
    print("Generating visualizations...")
    
    # Basic visualizations
    metrics = DEFAULT_METRICS + ["power_consumption"] if "power_consumption" in df.columns else DEFAULT_METRICS
    
    # Determine visualization options based on advanced_vis flag
    if advanced_vis:
        print("Enabling advanced visualization features...")
        visualization_files = vis.create_batch_visualizations(
            data=df,
            metrics=metrics,
            groupby=["model_category", "hardware"],
            include_3d=True,
            include_time_series=True,
            include_power_efficiency="power_consumption" in df.columns,
            include_dimension_reduction=True,
            include_confidence=True
        )
        
        # Generate additional 3D visualizations with different metric combinations
        print("Generating advanced 3D visualizations...")
        metric_combinations = [
            ("batch_size", "throughput", "memory_usage"),
            ("batch_size", "throughput", "latency_mean"),
            ("memory_usage", "latency_mean", "throughput")
        ]
        
        for x, y, z in metric_combinations:
            output_file = vis.create_3d_visualization(
                df,
                x_metric=x,
                y_metric=y,
                z_metric=z,
                color_metric="hardware",
                title=f"Performance Relationship: {x} vs {y} vs {z}"
            )
            visualization_files["3d"].append(output_file)
        
        # Generate dimension reduction visualizations for feature importance
        print("Generating dimension reduction visualizations...")
        for method in ["pca", "tsne"]:
            for metric in metrics:
                output_file = vis.create_dimension_reduction_visualization(
                    df,
                    features=[col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
                    target=metric,
                    method=method,
                    groupby="model_category",
                    title=f"{method.upper()} Analysis for {metric}"
                )
                visualization_files["dimension_reduction"].append(output_file)
        
        # Generate advanced dashboards
        print("Generating advanced performance dashboards...")
        groupby_combinations = [
            ["model_category", "hardware"],
            ["model_name", "hardware"],
            ["model_category", "batch_size"],
            ["hardware", "batch_size"]
        ]
        
        for groupby in groupby_combinations:
            for metric in metrics:
                output_file = vis.create_performance_dashboard(
                    df,
                    metrics=[metric],
                    groupby=groupby,
                    title=f"{metric} Performance by {' and '.join(groupby)}"
                )
                visualization_files["dashboard"].append(output_file)
    else:
        # Basic visualizations
        visualization_files = vis.create_batch_visualizations(
            data=df,
            metrics=metrics,
            groupby=["model_category", "hardware"],
            include_3d=True,
            include_time_series=True,
            include_power_efficiency="power_consumption" in df.columns,
            include_dimension_reduction=True,
            include_confidence=True
        )
    
    # Generate visualization report
    print("Creating visualization report...")
    report_title = "Predictive Performance System - Advanced Visualization Demo" if advanced_vis else "Predictive Performance System - Visualization Demo"
    report_path = create_visualization_report(
        visualization_files=visualization_files,
        title=report_title,
        output_file="visualization_report.html",
        output_dir=str(vis_dir)
    )
    
    # Print summary
    total_visualizations = sum(len(files) for files in visualization_files.values())
    print(f"\nGenerated {total_visualizations} visualizations in {vis_dir}")
    
    for vis_type, files in visualization_files.items():
        if files:
            print(f"  - {len(files)} {vis_type} visualizations")
    
    print(f"\nVisualization report: {report_path}")
    print(f"Open this file in a web browser to view all visualizations")
    
    return visualization_files, report_path

def generate_predictions_for_visualization(advanced_vis=False):
    """Generate predictions using the PerformancePredictor and visualize them."""
    print_header("Generating Predictions for Visualization")
    
    if not PREDICTOR_AVAILABLE:
        print("Error: PerformancePredictor not available")
        print("Please ensure the Predictive Performance System is properly installed")
        sys.exit(1)
    
    # Create output directory
    pred_dir = DEMO_OUTPUT_DIR / "predictions"
    pred_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize predictor
    print("Initializing performance predictor...")
    try:
        predictor = PerformancePredictor()
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        print("Using sample data instead")
        return run_visualization_demo()
    
    # Generate predictions for all combinations
    print("Generating predictions for all model-hardware combinations...")
    
    # Prepare list for predictions
    predictions = []
    
    # Generate predictions
    for model_info in DEFAULT_TEST_MODELS:
        model_name = model_info["name"]
        model_category = model_info["category"]
        model_short_name = model_name.split("/")[-1]
        
        for hardware in DEFAULT_TEST_HARDWARE:
            for batch_size in DEFAULT_TEST_BATCH_SIZES:
                for precision in DEFAULT_TEST_PRECISIONS:
                    # Skip incompatible combinations
                    if precision == "fp16" and hardware == "cpu":
                        continue
                    
                    # Make prediction
                    try:
                        prediction = predictor.predict(
                            model_name=model_name,
                            model_type=model_category,
                            hardware_platform=hardware,
                            batch_size=batch_size,
                            precision=precision,
                            calculate_uncertainty=True
                        )
                        
                        if prediction:
                            # Extract prediction values
                            pred_values = prediction.get("predictions", {})
                            uncertainties = prediction.get("uncertainties", {})
                            
                            # Create prediction record
                            pred_record = {
                                "model_name": model_short_name,
                                "model_category": model_category,
                                "hardware": hardware,
                                "batch_size": batch_size,
                                "precision": precision,
                                "confidence": prediction.get("confidence_score", 0.8)
                            }
                            
                            # Add predicted metrics
                            for metric in DEFAULT_METRICS:
                                if metric in pred_values:
                                    pred_record[metric] = pred_values[metric]
                                    
                                    # Add uncertainty if available
                                    if metric in uncertainties:
                                        uncertainty = uncertainties[metric]
                                        pred_record[f"{metric}_lower_bound"] = uncertainty.get("lower_bound", pred_values[metric] * 0.8)
                                        pred_record[f"{metric}_upper_bound"] = uncertainty.get("upper_bound", pred_values[metric] * 1.2)
                            
                            # Add to predictions list
                            predictions.append(pred_record)
                            
                    except Exception as e:
                        print(f"Error predicting {model_name} on {hardware}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(predictions)
    
    # Save to CSV and JSON
    csv_path = pred_dir / "prediction_results.csv"
    json_path = pred_dir / "prediction_results.json"
    
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    
    print(f"Generated {len(df)} predictions")
    print(f"Predictions saved to:")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {json_path}")
    
    # Run visualization demo with predictions
    print("\nVisualizing predictions...")
    return run_visualization_demo(json_path, advanced_vis=advanced_vis)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualization Demo for the Predictive Performance System")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", help="Path to performance data file (JSON or CSV)")
    group.add_argument("--demo", action="store_true", help="Run demo with sample data")
    group.add_argument("--generate", action="store_true", help="Generate and visualize predictions")
    
    parser.add_argument("--output-dir", help="Directory to save output files")
    parser.add_argument("--advanced-vis", action="store_true", help="Enable advanced visualization features")
    
    args = parser.parse_args()
    
    # Set output directory if specified
    if args.output_dir:
        global DEMO_OUTPUT_DIR
        DEMO_OUTPUT_DIR = Path(args.output_dir)
        DEMO_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Run appropriate demo
    if args.data:
        # Run visualization demo with provided data
        visualization_files, report_path = run_visualization_demo(args.data, advanced_vis=args.advanced_vis)
    elif args.generate:
        # Generate predictions and visualize them
        visualization_files, report_path = generate_predictions_for_visualization(advanced_vis=args.advanced_vis)
    else:
        # Run demo with sample data
        visualization_files, report_path = run_visualization_demo(advanced_vis=args.advanced_vis)
    
    # Final output
    print_header("Visualization Demo Completed")
    print(f"All output files are in: {DEMO_OUTPUT_DIR}")
    print(f"Visualization report: {report_path}")
    print("\nOpen the visualization report in a web browser to explore all visualizations")
    
    # Additional advanced visualizations
    print("\nNOTE: New advanced visualization features are now available (as of May 2025):")
    print("- 3D visualizations for exploring multi-dimensional performance relationships")
    print("- Power efficiency analysis with contour plots and efficiency isolines")
    print("- Dimension reduction visualizations for feature importance analysis")
    print("- Interactive dashboards with filtering capabilities")
    print("- Time-series visualizations with trend detection")
    print("- Uncertainty visualization with confidence intervals and reliability indicators")
    print("- Cross-browser model sharding visualization for optimal resource allocation")
    
    print("\nThese features support both interactive (HTML/Plotly) and static (PNG/PDF) outputs.")
    print("Use the --advanced-vis flag to enable all advanced visualization features.")

if __name__ == "__main__":
    main()