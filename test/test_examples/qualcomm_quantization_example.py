#\!/usr/bin/env python3
"""
Qualcomm AI Engine Quantization Example

This example demonstrates how to use the Qualcomm quantization support
to optimize models for deployment on Qualcomm hardware.

Usage:
  python test_examples/qualcomm_quantization_example.py --model-path /path/to/model.onnx
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Qualcomm quantization support
try:
    from test.qualcomm_quantization_support import QualcommQuantization
except ImportError:
    print("Error: Could not import QualcommQuantization. Make sure you're running from the project root.")
    sys.exit(1)

def main():
    """Main entry point for the example."""
    parser = argparse.ArgumentParser(description="Qualcomm AI Engine Quantization Example")
    parser.add_argument("--model-path", required=True, help="Path to input model (ONNX or PyTorch)")
    parser.add_argument("--output-dir", default="./quantized_models", help="Directory for saving quantized models")
    parser.add_argument("--model-type", default="text", choices=["text", "vision", "audio", "llm"], 
                       help="Model type (text, vision, audio, llm)")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb", help="Path to DuckDB database")
    parser.add_argument("--mock", action="store_true", help="Use mock mode for testing without hardware")
    args = parser.parse_args()

    # Set environment variables if needed
    if args.mock:
        os.environ["QUALCOMM_MOCK"] = "1"
        print("Using mock mode for testing without hardware")
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Qualcomm quantization handler
    qquant = QualcommQuantization(db_path=args.db_path)
    
    # Check if Qualcomm quantization is available
    if not qquant.is_available():
        print("Error: Qualcomm AI Engine not available and mock mode disabled.")
        return 1
    
    # Get supported quantization methods
    supported_methods = qquant.get_supported_methods()
    supported_method_names = [m for m, s in supported_methods.items() if s]
    
    print(f"\nSupported quantization methods: {', '.join(supported_method_names)}")
    print(f"Model type: {args.model_type}")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Database path: {args.db_path}")
    print(f"Mock mode: {qquant.mock_mode}")
    
    # Example 1: Basic quantization with INT8
    print("\n=== Example 1: Basic INT8 Quantization ===")
    model_basename = os.path.basename(args.model_path)
    int8_output_path = os.path.join(args.output_dir, f"{model_basename}.int8.qnn")
    
    print(f"Quantizing model to INT8: {int8_output_path}")
    start_time = time.time()
    
    result = qquant.quantize_model(
        model_path=args.model_path,
        output_path=int8_output_path,
        method="int8",
        model_type=args.model_type
    )
    
    elapsed_time = time.time() - start_time
    
    if "error" in result:
        print(f"Error during quantization: {result['error']}")
    else:
        print(f"Quantization successful in {elapsed_time:.2f} seconds")
        print(f"Size reduction: {result.get('size_reduction_ratio', 0):.2f}x")
        print(f"Status: {result.get('status', 'Unknown')}")
        
        # Print power efficiency metrics
        if "power_efficiency_metrics" in result:
            metrics = result["power_efficiency_metrics"]
            print("\nEstimated Power Efficiency Metrics:")
            print(f"- Power Consumption: {metrics.get('power_consumption_mw', 0):.2f} mW")
            print(f"- Energy Efficiency: {metrics.get('energy_efficiency_items_per_joule', 0):.2f} items/joule")
            print(f"- Battery Impact: {metrics.get('battery_impact_percent_per_hour', 0):.2f}% per hour")
            print(f"- Power Reduction: {metrics.get('power_reduction_percent', 0):.2f}%")
    
    # Example 2: Quantization Parameter Customization
    print("\n=== Example 2: Customized Quantization ===")
    custom_output_path = os.path.join(args.output_dir, f"{model_basename}.custom.qnn")
    
    # Customize parameters based on model type
    custom_params = {}
    if args.model_type == "text":
        custom_params = {
            "dynamic_quantization": True,
            "optimize_attention": True
        }
    elif args.model_type == "vision":
        custom_params = {
            "input_layout": "NCHW",
            "optimize_vision_models": True
        }
    elif args.model_type == "audio":
        custom_params = {
            "optimize_audio_models": True,
            "enable_attention_fusion": True
        }
    elif args.model_type == "llm":
        custom_params = {
            "optimize_llm": True,
            "enable_kv_cache": True,
            "enable_attention_fusion": True
        }
    
    print(f"Quantizing model with custom parameters: {custom_output_path}")
    start_time = time.time()
    
    result = qquant.quantize_model(
        model_path=args.model_path,
        output_path=custom_output_path,
        method="dynamic",
        model_type=args.model_type,
        **custom_params
    )
    
    elapsed_time = time.time() - start_time
    
    if "error" in result:
        print(f"Error during quantization: {result['error']}")
    else:
        print(f"Quantization successful in {elapsed_time:.2f} seconds")
        print(f"Size reduction: {result.get('size_reduction_ratio', 0):.2f}x")
        print(f"Status: {result.get('status', 'Unknown')}")
    
    # Example 3: Benchmark Quantized Model
    print("\n=== Example 3: Benchmark Quantized Model ===")
    print(f"Benchmarking quantized model: {int8_output_path}")
    start_time = time.time()
    
    benchmark_result = qquant.benchmark_quantized_model(
        model_path=int8_output_path,
        model_type=args.model_type
    )
    
    elapsed_time = time.time() - start_time
    
    if "error" in benchmark_result:
        print(f"Error during benchmarking: {benchmark_result['error']}")
    else:
        print(f"Benchmarking successful in {elapsed_time:.2f} seconds")
        print(f"Status: {benchmark_result.get('status', 'Unknown')}")
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        print(f"- Latency: {benchmark_result.get('latency_ms', 0):.2f} ms")
        print(f"- Throughput: {benchmark_result.get('throughput', 0):.2f} {benchmark_result.get('throughput_units', 'items/second')}")
        
        # Print power metrics
        if "metrics" in benchmark_result:
            metrics = benchmark_result["metrics"]
            print("\nPower and Thermal Metrics:")
            print(f"- Power Consumption: {metrics.get('power_consumption_mw', 0):.2f} mW")
            print(f"- Energy Efficiency: {metrics.get('energy_efficiency_items_per_joule', 0):.2f} items/joule")
            print(f"- Battery Impact: {metrics.get('battery_impact_percent_per_hour', 0):.2f}% per hour")
            print(f"- Temperature: {metrics.get('temperature_celsius', 0):.2f}°C")
            print(f"- Thermal Throttling Detected: {metrics.get('thermal_throttling_detected', False)}")
    
    # Example 4: Compare Quantization Methods
    print("\n=== Example 4: Compare Quantization Methods (Simplified) ===")
    # In a full example, this would be a full comparison, but we'll limit to 2 methods for speed
    limited_methods = ["dynamic", "int8"]
    
    print(f"Comparing quantization methods for {args.model_path}: {', '.join(limited_methods)}")
    print("Note: Using limited methods for demo purposes. Full comparison would include all methods.")
    
    comparison_dir = os.path.join(args.output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    result = qquant.compare_quantization_methods(
        model_path=args.model_path,
        output_dir=comparison_dir,
        model_type=args.model_type,
        methods=limited_methods
    )
    
    if "error" in result:
        print(f"Error during comparison: {result['error']}")
    else:
        # Print summary
        summary = result.get("summary", {})
        recommendation = summary.get("overall_recommendation", {})
        
        print("\nComparison Summary:")
        print(f"- Overall Recommendation: {recommendation.get('final_recommendation', 'Unknown')}")
        print(f"- Rationale: {recommendation.get('rationale', 'Unknown')}")
        
        # Generate report
        report_path = os.path.join(comparison_dir, "quantization_comparison_report.md")
        report = qquant.generate_report(result, report_path)
        print(f"\nDetailed comparison report saved to: {report_path}")
    
    print("\nQuantization examples completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
