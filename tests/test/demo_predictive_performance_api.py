#!/usr/bin/env python3
"""
Demonstration of Predictive Performance API

This script demonstrates the usage of the Predictive Performance API
through the Unified API Server integration.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import client library
from test.api_client.predictive_performance_client import (
    PredictivePerformanceClient,
    HardwarePlatform,
    PrecisionType,
    ModelMode
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo_predictive_performance_api")

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_json(data):
    """Print JSON data in a readable format."""
    print(json.dumps(data, indent=2))

def setup_demo_data(client):
    """Set up demo data for testing."""
    print_section("Setting up demo data")
    
    result = client.generate_sample_data(num_models=5, wait=True)
    if "result" in result:
        print("Sample data generated successfully")
    else:
        print("Failed to generate sample data")
        print_json(result)

def demo_hardware_recommendations(client):
    """Demonstrate hardware recommendations."""
    print_section("Hardware Recommendations")
    
    # Demo models
    models = [
        {"name": "bert-base-uncased", "family": "embedding"},
        {"name": "gpt2", "family": "text_generation"},
        {"name": "vit-base-patch16-224", "family": "image_classification"}
    ]
    
    for model in models:
        print(f"\nModel: {model['name']} ({model['family']})")
        print("-" * 40)
        
        result = client.predict_hardware(
            model_name=model["name"],
            model_family=model["family"],
            batch_size=8,
            available_hardware=[
                HardwarePlatform.CPU,
                HardwarePlatform.CUDA,
                HardwarePlatform.ROCM,
                HardwarePlatform.WEBGPU
            ],
            predict_performance=True,
            wait=True
        )
        
        if "result" in result:
            primary = result["result"]["primary_recommendation"]
            print(f"Primary recommendation: {primary}")
            
            alternatives = result["result"].get("alternative_recommendations", [])
            if alternatives:
                print("Alternative recommendations:")
                for alt in alternatives:
                    print(f"- {alt}")
            
            if "performance" in result["result"]:
                perf = result["result"]["performance"]
                print(f"Predicted performance on {primary}:")
                print(f"- Throughput: {perf.get('throughput', 'N/A')} samples/sec")
                print(f"- Latency: {perf.get('latency', 'N/A')} ms")
                print(f"- Memory usage: {perf.get('memory_usage', 'N/A')} MB")
        else:
            print("Failed to get hardware recommendations")
            print_json(result)

def demo_performance_predictions(client):
    """Demonstrate performance predictions."""
    print_section("Performance Predictions")
    
    model_name = "bert-base-uncased"
    hardware_platforms = [
        HardwarePlatform.CPU,
        HardwarePlatform.CUDA,
        HardwarePlatform.WEBGPU
    ]
    
    print(f"Model: {model_name}")
    print(f"Hardware: {', '.join([h.value for h in hardware_platforms])}")
    print("-" * 40)
    
    result = client.predict_performance(
        model_name=model_name,
        hardware=hardware_platforms,
        batch_size=8,
        wait=True
    )
    
    if "result" in result and "predictions" in result["result"]:
        predictions = result["result"]["predictions"]
        print("Performance predictions:")
        
        for hw, metrics in predictions.items():
            print(f"\n{hw.upper()}:")
            for metric, value in metrics.items():
                print(f"- {metric}: {value}")
    else:
        print("Failed to get performance predictions")
        print_json(result)

def demo_batch_size_analysis(client):
    """Demonstrate batch size performance analysis."""
    print_section("Batch Size Analysis")
    
    model_name = "bert-base-uncased"
    hardware = HardwarePlatform.CUDA
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    print(f"Model: {model_name}")
    print(f"Hardware: {hardware.value}")
    print(f"Batch sizes: {batch_sizes}")
    print("-" * 40)
    
    results = {}
    
    for batch in batch_sizes:
        result = client.predict_performance(
            model_name=model_name,
            hardware=hardware,
            batch_size=batch,
            wait=True
        )
        
        if "result" in result and "predictions" in result["result"]:
            predictions = result["result"]["predictions"]
            if hardware.value in predictions and "throughput" in predictions[hardware.value]:
                results[batch] = predictions[hardware.value]["throughput"]
    
    if results:
        print("Throughput by batch size:")
        for batch, throughput in results.items():
            print(f"- Batch size {batch}: {throughput:.2f} samples/sec")
        
        # Find optimal batch size
        optimal_batch = max(results, key=results.get)
        print(f"\nOptimal batch size: {optimal_batch} ({results[optimal_batch]:.2f} samples/sec)")
    else:
        print("Failed to analyze batch sizes")

def demo_measurement_recording(client):
    """Demonstrate measurement recording."""
    print_section("Recording Measurements")
    
    model_name = "bert-base-uncased"
    hardware = HardwarePlatform.CUDA
    
    # First get a prediction
    print(f"Getting prediction for {model_name} on {hardware.value}...")
    pred_result = client.predict_performance(
        model_name=model_name,
        hardware=hardware,
        batch_size=8,
        wait=True
    )
    
    prediction_id = None
    predicted_throughput = None
    
    if "result" in pred_result and "predictions" in pred_result["result"]:
        if "prediction_id" in pred_result["result"]:
            prediction_id = pred_result["result"]["prediction_id"]
        
        predictions = pred_result["result"]["predictions"]
        if hardware.value in predictions and "throughput" in predictions[hardware.value]:
            predicted_throughput = predictions[hardware.value]["throughput"]
    
    print(f"Prediction ID: {prediction_id}")
    print(f"Predicted throughput: {predicted_throughput}")
    
    # Now record a simulated measurement
    if predicted_throughput:
        # Simulate a measurement that's within 10% of the prediction
        import random
        actual_throughput = predicted_throughput * random.uniform(0.9, 1.1)
        
        print(f"\nRecording measurement: {actual_throughput:.2f} samples/sec")
        result = client.record_measurement(
            model_name=model_name,
            hardware_platform=hardware,
            batch_size=8,
            throughput=actual_throughput,
            latency=8.3,
            memory_usage=1024.0,
            prediction_id=prediction_id,
            source="demo",
            wait=True
        )
        
        if "result" in result:
            print("Measurement recorded successfully")
            if "accuracy" in result["result"]:
                accuracy = result["result"]["accuracy"]
                print(f"Prediction accuracy: {accuracy:.2f}%")
        else:
            print("Failed to record measurement")
            print_json(result)
    else:
        print("Skipping measurement recording due to missing prediction data")

def demo_prediction_analysis(client):
    """Demonstrate prediction analysis."""
    print_section("Prediction Analysis")
    
    result = client.analyze_predictions(
        model_name="bert-base-uncased",
        hardware_platform=HardwarePlatform.CUDA,
        metric="throughput",
        wait=True
    )
    
    if "result" in result:
        stats = result["result"]
        print("Prediction accuracy statistics:")
        print(f"Total predictions: {stats.get('total_count', 0)}")
        print(f"Average accuracy: {stats.get('average_accuracy', 0):.2f}%")
        print(f"Median accuracy: {stats.get('median_accuracy', 0):.2f}%")
        
        if "accuracy_distribution" in stats:
            print("\nAccuracy distribution:")
            for range_str, count in stats["accuracy_distribution"].items():
                print(f"- {range_str}: {count}")
    else:
        print("Failed to analyze predictions")
        print_json(result)

def demo_list_resources(client):
    """Demonstrate listing resources."""
    print_section("Listing Resources")
    
    # List recommendations
    print("\nRecent hardware recommendations:")
    print("-" * 40)
    
    result = client.list_recommendations(limit=3)
    if "recommendations" in result:
        for rec in result["recommendations"]:
            print(f"Model: {rec.get('model_name')}")
            print(f"Primary recommendation: {rec.get('primary_recommendation')}")
            print(f"Timestamp: {rec.get('timestamp')}")
            print(f"Feedback: {rec.get('was_accepted', 'No feedback')}")
            print()
    else:
        print("Failed to list recommendations")
    
    # List measurements
    print("\nRecent performance measurements:")
    print("-" * 40)
    
    result = client.list_measurements(limit=3)
    if "measurements" in result:
        for meas in result["measurements"]:
            print(f"Model: {meas.get('model_name')}")
            print(f"Hardware: {meas.get('hardware_platform')}")
            print(f"Throughput: {meas.get('throughput')} samples/sec")
            print(f"Latency: {meas.get('latency')} ms")
            print(f"Memory usage: {meas.get('memory_usage')} MB")
            print(f"Timestamp: {meas.get('timestamp')}")
            print()
    else:
        print("Failed to list measurements")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Predictive Performance API Demo")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="Unified API Server URL")
    parser.add_argument("--setup", action="store_true", help="Set up demo data")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--hardware", action="store_true", help="Run hardware recommendations demo")
    parser.add_argument("--performance", action="store_true", help="Run performance predictions demo")
    parser.add_argument("--batch", action="store_true", help="Run batch size analysis demo")
    parser.add_argument("--measure", action="store_true", help="Run measurement recording demo")
    parser.add_argument("--analyze", action="store_true", help="Run prediction analysis demo")
    parser.add_argument("--list", action="store_true", help="Run resource listing demo")
    
    args = parser.parse_args()
    
    # Create client
    client = PredictivePerformanceClient(base_url=args.url)
    
    # Check if server is accessible
    try:
        # Make a simple request to check connectivity
        response = client.session.get(f"{args.url}/health")
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Error connecting to API server: {e}")
        print("\nError: Could not connect to the API server.")
        print(f"       Please make sure the server is running at {args.url}")
        print("\nYou can start the server with:")
        print("  python test/run_integrated_api_servers.py")
        return 1
    
    # Set up demo data if requested
    if args.setup:
        setup_demo_data(client)
    
    # Run demos
    if args.all or args.hardware:
        demo_hardware_recommendations(client)
    
    if args.all or args.performance:
        demo_performance_predictions(client)
    
    if args.all or args.batch:
        demo_batch_size_analysis(client)
    
    if args.all or args.measure:
        demo_measurement_recording(client)
    
    if args.all or args.analyze:
        demo_prediction_analysis(client)
    
    if args.all or args.list:
        demo_list_resources(client)
    
    # If no demo specified, print help
    if not any([args.setup, args.all, args.hardware, args.performance, 
                args.batch, args.measure, args.analyze, args.list]):
        parser.print_help()
        print("\nTip: Use --all to run all demos, or --setup to generate sample data")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())