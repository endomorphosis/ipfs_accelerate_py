#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Android Test Harness - Simple Example

This is a minimal example of how to use the Android Test Harness to run 
models on Android devices. For a more comprehensive example, see 
real_execution_example.py.

Usage:
    python example.py --model path/to/model.onnx --serial device_serial

Date: April 2025
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules
from test.android_test_harness.android_test_harness import AndroidTestHarness


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Android Test Harness - Simple Example")
    
    parser.add_argument("--model", required=True, help="Path to model file (ONNX or TFLite)")
    parser.add_argument("--serial", help="Device serial number")
    parser.add_argument("--output-dir", default="./android_results", help="Output directory")
    parser.add_argument("--accelerator", default="auto", choices=["auto", "cpu", "gpu", "npu", "dsp"],
                      help="Hardware accelerator to use")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--generate-report", action="store_true", help="Generate a report")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine model type from file extension
    model_path = args.model
    model_type = "onnx"
    if model_path.lower().endswith(".tflite"):
        model_type = "tflite"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Android Test Harness - Simple Example")
    print(f"Model: {model_path}")
    print(f"Model type: {model_type}")
    print(f"Accelerator: {args.accelerator}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    try:
        # Initialize test harness
        harness = AndroidTestHarness(
            device_serial=args.serial,
            output_dir=args.output_dir
        )
        
        # Connect to device
        if not harness.connect_to_device():
            print("Failed to connect to Android device. Make sure:")
            print("1. The device is connected via USB")
            print("2. USB debugging is enabled on the device")
            print("3. You have authorized USB debugging on the device")
            return 1
        
        # Print device info
        device_info = harness.device.to_dict()
        print("Connected to Android device:")
        print(f"  Model: {device_info.get('model', 'Unknown')}")
        print(f"  Manufacturer: {device_info.get('manufacturer', 'Unknown')}")
        print(f"  Android version: {device_info.get('android_version', 'Unknown')}")
        print(f"  Chipset: {device_info.get('chipset', 'Unknown')}")
        print()
        
        # Prepare model
        print(f"Preparing model {os.path.basename(model_path)}...")
        remote_model_path = harness.prepare_model(model_path, model_type)
        
        if not remote_model_path:
            print("Failed to prepare model")
            return 1
        
        print(f"Model prepared at: {remote_model_path}")
        
        # Run model
        print(f"Running model with {args.accelerator} accelerator...")
        result = harness.model_runner.run_model(
            model_path=remote_model_path,
            iterations=args.iterations,
            batch_size=args.batch_size,
            accelerator=args.accelerator
        )
        
        # Print performance results
        print("\nPerformance Results:")
        
        if result.get("status") == "success":
            # Print latency
            latency = result.get("latency_ms", {})
            print(f"  Latency (ms):")
            print(f"    Min: {latency.get('min', 0):.2f}")
            print(f"    Mean: {latency.get('mean', 0):.2f}")
            print(f"    Median: {latency.get('median', 0):.2f}")
            print(f"    P90: {latency.get('p90', 0):.2f}")
            print(f"    P99: {latency.get('p99', 0):.2f}")
            print(f"    Max: {latency.get('max', 0):.2f}")
            
            # Print throughput
            throughput = result.get("throughput_items_per_second", 0)
            print(f"  Throughput: {throughput:.2f} items/second")
            
            # Print memory usage
            memory = result.get("memory_metrics", {})
            print(f"  Memory usage: {memory.get('peak_mb', 0):.2f} MB")
            
            # Print battery impact
            battery = result.get("battery_metrics", {})
            print(f"  Battery impact: {battery.get('impact_percentage', 0):.1f}%")
            print(f"  Battery temperature increase: {battery.get('temperature_delta', 0):.1f}°C")
            
            # Print thermal impact
            thermal = result.get("thermal_metrics", {})
            thermal_delta = thermal.get("delta", {})
            if thermal_delta:
                max_zone = max(thermal_delta.items(), key=lambda x: x[1], default=(None, 0))
                if max_zone[0]:
                    print(f"  Thermal impact: {max_zone[1]:.1f}°C in {max_zone[0]} zone")
            
            # Generate report if requested
            if args.generate_report:
                report_path = os.path.join(args.output_dir, f"{os.path.basename(model_path)}_report.md")
                report = harness.generate_report(results_data=result)
                
                with open(report_path, "w") as f:
                    f.write(report)
                
                print(f"\nReport saved to: {report_path}")
        else:
            print(f"  Error: {result.get('message', 'Unknown error')}")
        
        print("\nDone!")
        return 0
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())