#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Android Test Harness - Real Model Execution Example

This example script demonstrates how to use the Android Test Harness to run real
model execution on Android devices, comparing performance across different
hardware accelerators and model formats.

Usage:
    python real_execution_example.py --model <path_to_model_file> --serial <device_serial>

Features demonstrated:
    - Real model execution on Android devices
    - Hardware accelerator selection
    - Performance comparison across accelerators
    - Thermal monitoring during execution
    - Storage of results in the benchmark database
    - Generation of performance reports

Date: April 2025
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

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
from test.android_test_harness.android_thermal_monitor import AndroidThermalMonitor
try:
    from test.android_test_harness.android_model_executor import ModelFormat, AcceleratorType
    MODEL_EXECUTOR_AVAILABLE = True
except ImportError:
    logger.warning("AndroidModelExecutor not available, some functionality will be limited")
    MODEL_EXECUTOR_AVAILABLE = False


def run_model_comparison(
    harness: AndroidTestHarness,
    model_path: str,
    model_name: Optional[str] = None,
    model_type: str = "onnx",
    batch_sizes: List[int] = [1],
    accelerators: List[str] = ["auto", "cpu", "gpu"],
    iterations: int = 50,
    warmup_iterations: int = 10,
    save_to_db: bool = True,
    output_dir: str = "./android_results"
) -> Dict[str, Any]:
    """
    Run model execution across different accelerators and compare performance.
    
    Args:
        harness: Android test harness
        model_path: Path to model file
        model_name: Optional name of the model
        model_type: Type of model (onnx, tflite)
        batch_sizes: List of batch sizes to test
        accelerators: List of accelerators to test
        iterations: Number of iterations per test
        warmup_iterations: Number of warmup iterations
        save_to_db: Whether to save results to database
        output_dir: Directory to save results
        
    Returns:
        Dictionary with comparison results
    """
    if model_name is None:
        model_name = os.path.basename(model_path)
    
    logger.info(f"Running model comparison for {model_name}")
    
    # Run benchmark for each configuration
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"{output_dir}/{model_name}_comparison_{timestamp}.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run benchmark
    benchmark_results = harness.run_benchmark(
        model_path=model_path,
        model_name=model_name,
        model_type=model_type,
        batch_sizes=batch_sizes,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        accelerators=accelerators,
        thread_counts=[4],  # Fixed thread count for comparison
        save_to_db=save_to_db,
        collect_metrics=True
    )
    
    # Generate comparison plots
    if benchmark_results["status"] == "success":
        plot_filename = f"{output_dir}/{model_name}_comparison_{timestamp}.png"
        plot_comparison_results(benchmark_results, plot_filename)
        
        # Generate report
        report_filename = f"{output_dir}/{model_name}_report_{timestamp}.md"
        report = harness.generate_report(results_data=benchmark_results)
        
        with open(report_filename, "w") as f:
            f.write(report)
        
        logger.info(f"Comparison results saved to {results_filename}")
        logger.info(f"Comparison plot saved to {plot_filename}")
        logger.info(f"Comparison report saved to {report_filename}")
    
    # Save results to file
    with open(results_filename, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    return benchmark_results


def plot_comparison_results(results: Dict[str, Any], output_file: str) -> None:
    """
    Plot comparison results.
    
    Args:
        results: Benchmark results
        output_file: Output file path
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Extract configurations
        configs = results.get("configurations", [])
        if not configs:
            logger.error("No configurations found in results")
            return
        
        # Group by accelerator
        accelerator_results = {}
        for config in configs:
            config_info = config.get("configuration", {})
            accelerator = config_info.get("accelerator", "unknown")
            batch_size = config_info.get("batch_size", 1)
            
            if accelerator not in accelerator_results:
                accelerator_results[accelerator] = {}
            
            if batch_size not in accelerator_results[accelerator]:
                accelerator_results[accelerator][batch_size] = []
            
            accelerator_results[accelerator][batch_size].append(config)
        
        # Plot comparison
        colors = {
            "cpu": "blue",
            "gpu": "green",
            "npu": "red",
            "dsp": "purple",
            "auto": "orange",
            "unknown": "gray"
        }
        
        # Plot 1: Latency comparison
        plt.subplot(2, 2, 1)
        
        for accelerator, batch_results in accelerator_results.items():
            batch_sizes = []
            latencies = []
            
            for batch_size, configs in sorted(batch_results.items()):
                avg_latency = sum(
                    config.get("latency_ms", {}).get("mean", 0) 
                    for config in configs
                ) / max(1, len(configs))
                
                batch_sizes.append(batch_size)
                latencies.append(avg_latency)
            
            plt.plot(
                batch_sizes, 
                latencies, 
                'o-', 
                color=colors.get(accelerator, "gray"),
                label=f"{accelerator}"
            )
        
        plt.title("Inference Latency by Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("Latency (ms)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot 2: Throughput comparison
        plt.subplot(2, 2, 2)
        
        for accelerator, batch_results in accelerator_results.items():
            batch_sizes = []
            throughputs = []
            
            for batch_size, configs in sorted(batch_results.items()):
                avg_throughput = sum(
                    config.get("throughput_items_per_second", 0) 
                    for config in configs
                ) / max(1, len(configs))
                
                batch_sizes.append(batch_size)
                throughputs.append(avg_throughput)
            
            plt.plot(
                batch_sizes, 
                throughputs, 
                'o-', 
                color=colors.get(accelerator, "gray"),
                label=f"{accelerator}"
            )
        
        plt.title("Throughput by Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("Throughput (items/s)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot 3: Battery impact comparison
        plt.subplot(2, 2, 3)
        
        for accelerator, batch_results in accelerator_results.items():
            batch_sizes = []
            battery_impacts = []
            
            for batch_size, configs in sorted(batch_results.items()):
                avg_impact = sum(
                    config.get("battery_metrics", {}).get("impact_percentage", 0) 
                    for config in configs
                ) / max(1, len(configs))
                
                batch_sizes.append(batch_size)
                battery_impacts.append(avg_impact)
            
            plt.plot(
                batch_sizes, 
                battery_impacts, 
                'o-', 
                color=colors.get(accelerator, "gray"),
                label=f"{accelerator}"
            )
        
        plt.title("Battery Impact by Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("Battery Impact (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot 4: Thermal impact comparison
        plt.subplot(2, 2, 4)
        
        for accelerator, batch_results in accelerator_results.items():
            batch_sizes = []
            thermal_impacts = []
            
            for batch_size, configs in sorted(batch_results.items()):
                # Get maximum thermal impact across all zones
                avg_thermal_impact = 0
                for config in configs:
                    thermal_metrics = config.get("thermal_metrics", {})
                    thermal_delta = thermal_metrics.get("delta", {})
                    max_delta = max(thermal_delta.values()) if thermal_delta else 0
                    avg_thermal_impact += max_delta
                
                avg_thermal_impact /= max(1, len(configs))
                
                batch_sizes.append(batch_size)
                thermal_impacts.append(avg_thermal_impact)
            
            plt.plot(
                batch_sizes, 
                thermal_impacts, 
                'o-', 
                color=colors.get(accelerator, "gray"),
                label=f"{accelerator}"
            )
        
        plt.title("Maximum Thermal Impact by Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("Max Temperature Increase (°C)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add super title
        plt.suptitle(
            f"Model Performance Comparison: {results.get('model_name', 'Unknown Model')}",
            fontsize=16
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_file, dpi=300)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting comparison results: {e}")


def run_thermal_analysis(
    harness: AndroidTestHarness,
    model_path: str,
    model_name: Optional[str] = None,
    model_type: str = "onnx",
    duration_seconds: int = 300,
    accelerator: str = "auto",
    batch_size: int = 1,
    threads: int = 4,
    output_dir: str = "./android_results"
) -> Dict[str, Any]:
    """
    Run thermal analysis during model execution.
    
    Args:
        harness: Android test harness
        model_path: Path to model file
        model_name: Optional name of the model
        model_type: Type of model (onnx, tflite)
        duration_seconds: Duration of the analysis in seconds
        accelerator: Hardware accelerator to use
        batch_size: Batch size for inference
        threads: Number of threads for CPU execution
        output_dir: Directory to save results
        
    Returns:
        Dictionary with thermal analysis results
    """
    if model_name is None:
        model_name = os.path.basename(model_path)
    
    logger.info(f"Running thermal analysis for {model_name} for {duration_seconds} seconds")
    
    # Prepare model
    remote_model_path = harness.prepare_model(model_path, model_type)
    
    if not remote_model_path:
        logger.error("Failed to prepare model")
        return {"status": "error", "message": "Failed to prepare model"}
    
    # Initialize thermal monitor
    thermal_monitor = AndroidThermalMonitor(harness.device)
    thermal_monitor.start_monitoring()
    
    # Run model in a loop
    logger.info(f"Running model with {accelerator} accelerator at batch size {batch_size}")
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    results = []
    temperatures = []
    timestamps = []
    
    try:
        while time.time() < end_time:
            # Run model
            result = harness.model_runner.run_model(
                model_path=remote_model_path,
                iterations=10,  # Small number of iterations
                warmup_iterations=2,
                batch_size=batch_size,
                threads=threads,
                accelerator=accelerator
            )
            
            results.append(result)
            
            # Collect thermal data
            current_time = time.time()
            current_temps = thermal_monitor.get_current_temperatures()
            
            timestamps.append(current_time - start_time)
            temperatures.append(current_temps)
            
            # Sleep briefly to avoid overwhelming the device
            time.sleep(1)
    
    finally:
        # Stop thermal monitoring
        thermal_monitor.stop_monitoring()
    
    # Generate thermal report
    thermal_report = thermal_monitor.get_thermal_report()
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"{output_dir}/{model_name}_thermal_{timestamp}.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analysis results
    analysis_results = {
        "status": "success",
        "model_name": model_name,
        "duration_seconds": duration_seconds,
        "accelerator": accelerator,
        "batch_size": batch_size,
        "threads": threads,
        "thermal_report": thermal_report,
        "temperature_time_series": [
            {
                "timestamp": timestamp,
                "temperatures": temps
            }
            for timestamp, temps in zip(timestamps, temperatures)
        ],
        "throttling_stats": thermal_monitor.get_throttling_stats(),
        "recommendations": thermal_monitor._generate_recommendations()
    }
    
    # Save results to file
    with open(results_filename, "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    # Generate thermal plot
    plot_filename = f"{output_dir}/{model_name}_thermal_{timestamp}.png"
    plot_thermal_analysis(analysis_results, plot_filename)
    
    logger.info(f"Thermal analysis results saved to {results_filename}")
    logger.info(f"Thermal analysis plot saved to {plot_filename}")
    
    return analysis_results


def plot_thermal_analysis(results: Dict[str, Any], output_file: str) -> None:
    """
    Plot thermal analysis results.
    
    Args:
        results: Thermal analysis results
        output_file: Output file path
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Extract time series data
        time_series = results.get("temperature_time_series", [])
        if not time_series:
            logger.error("No time series data found in results")
            return
        
        # Extract timestamps and temperature data
        timestamps = [entry["timestamp"] for entry in time_series]
        
        # Collect all zone names
        all_zones = set()
        for entry in time_series:
            all_zones.update(entry["temperatures"].keys())
        
        # Filter to commonly important zones
        important_zones = [
            zone for zone in all_zones 
            if any(keyword in zone.lower() for keyword in ["cpu", "gpu", "soc", "battery"])
        ]
        
        if not important_zones:
            important_zones = list(all_zones)[:4]  # Use first 4 zones if no important ones found
        
        # Plot temperature over time
        plt.subplot(2, 1, 1)
        
        for zone in important_zones:
            zone_temps = [
                entry["temperatures"].get(zone, float('nan')) 
                for entry in time_series
            ]
            
            plt.plot(timestamps, zone_temps, '-', label=zone)
        
        plt.title("Temperature Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot throttling status
        plt.subplot(2, 1, 2)
        
        # Throttling events from thermal report
        events = results.get("thermal_report", {}).get("recent_events", [])
        throttling_events = [
            event for event in events 
            if event["event_type"] in ["THROTTLING", "CRITICAL", "EMERGENCY"]
        ]
        
        # Create throttling level plot
        throttling_timestamps = []
        throttling_levels = []
        
        # Simulate throttling levels from events and time series
        throttling_status = 0
        for i, timestamp in enumerate(timestamps):
            # Check if there was a throttling event near this time
            for event in throttling_events:
                event_time = event["timestamp"] - results.get("thermal_report", {}).get("timestamp", 0) + timestamps[0]
                if abs(timestamp - event_time) < 5:  # Within 5 seconds
                    if event["event_type"] == "THROTTLING":
                        throttling_status = 2
                    elif event["event_type"] == "CRITICAL":
                        throttling_status = 4
                    elif event["event_type"] == "EMERGENCY":
                        throttling_status = 5
            
            # Check temperatures for throttling indicators
            temps = time_series[i]["temperatures"]
            max_temp = max(temps.values()) if temps else 0
            
            if max_temp > 80:
                throttling_status = max(throttling_status, 3)
            elif max_temp > 70:
                throttling_status = max(throttling_status, 2)
            elif max_temp > 60:
                throttling_status = max(throttling_status, 1)
            else:
                # Gradually reduce throttling status
                throttling_status = max(0, throttling_status - 0.2)
            
            throttling_timestamps.append(timestamp)
            throttling_levels.append(throttling_status)
        
        plt.plot(throttling_timestamps, throttling_levels, 'r-', linewidth=2)
        
        plt.title("Throttling Level Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Throttling Level")
        plt.yticks([0, 1, 2, 3, 4, 5], [
            "None", "Mild", "Moderate", "Heavy", "Severe", "Emergency"
        ])
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations for throttling recommendations
        recommendations = results.get("recommendations", [])
        for i, rec in enumerate(recommendations):
            if "THROTTLING" in rec:
                plt.annotate(
                    rec,
                    xy=(timestamps[-1], 5 - i * 0.5),
                    xytext=(timestamps[-1] * 0.8, 5 - i * 0.5),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    color="red",
                    fontsize=8
                )
        
        # Add super title
        plt.suptitle(
            f"Thermal Analysis: {results.get('model_name', 'Unknown Model')} "
            f"({results.get('accelerator', 'auto')}, batch={results.get('batch_size', 1)})",
            fontsize=16
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_file, dpi=300)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting thermal analysis results: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Android Test Harness - Real Model Execution Example")
    
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--name", help="Model name (defaults to filename)")
    parser.add_argument("--type", default="onnx", choices=["onnx", "tflite", "tflite_quantized"],
                       help="Model type")
    parser.add_argument("--serial", help="Device serial number")
    parser.add_argument("--output-dir", default="./android_results", help="Output directory")
    parser.add_argument("--db-path", help="Path to benchmark database")
    
    # Analysis type
    parser.add_argument("--analysis", default="comparison", choices=["comparison", "thermal", "both"],
                       help="Type of analysis to perform")
    
    # Comparison options
    parser.add_argument("--batch-sizes", default="1,2,4", help="Comma-separated batch sizes for comparison")
    parser.add_argument("--accelerators", default="auto,cpu,gpu", help="Comma-separated accelerators for comparison")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations per test")
    
    # Thermal analysis options
    parser.add_argument("--duration", type=int, default=300, help="Duration of thermal analysis in seconds")
    
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse batch sizes and accelerators
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    accelerators = args.accelerators.split(",")
    
    # Initialize test harness
    harness = AndroidTestHarness(
        device_serial=args.serial,
        db_path=args.db_path,
        output_dir=args.output_dir
    )
    
    if not harness.connect_to_device():
        logger.error("Failed to connect to Android device")
        return 1
    
    logger.info(f"Connected to device: {harness.device.device_info.get('model', args.serial)}")
    
    # Run selected analysis
    if args.analysis == "comparison" or args.analysis == "both":
        comparison_results = run_model_comparison(
            harness=harness,
            model_path=args.model,
            model_name=args.name,
            model_type=args.type,
            batch_sizes=batch_sizes,
            accelerators=accelerators,
            iterations=args.iterations,
            save_to_db=args.db_path is not None,
            output_dir=args.output_dir
        )
        
        # Print summary
        if comparison_results.get("status") == "success":
            best_config = None
            best_throughput = 0
            
            for config in comparison_results.get("configurations", []):
                throughput = config.get("throughput_items_per_second", 0)
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = config
            
            if best_config:
                config_info = best_config.get("configuration", {})
                print("\nComparison Results Summary:")
                print(f"- Best configuration: batch_size={config_info.get('batch_size', 1)}, "
                      f"accelerator={config_info.get('accelerator', 'auto')}")
                print(f"- Best throughput: {best_throughput:.2f} items/second")
                
                # Print thermal and battery impact
                battery_impact = best_config.get("battery_metrics", {}).get("impact_percentage", 0)
                thermal_metrics = best_config.get("thermal_metrics", {})
                thermal_delta = thermal_metrics.get("delta", {})
                max_thermal_delta = max(thermal_delta.values()) if thermal_delta else 0
                
                print(f"- Battery impact: {battery_impact:.1f}%")
                print(f"- Max thermal impact: {max_thermal_delta:.1f}°C")
    
    if args.analysis == "thermal" or args.analysis == "both":
        # Use best accelerator from comparison if available
        accelerator = "auto"
        batch_size = 1
        
        if args.analysis == "both" and comparison_results.get("status") == "success":
            best_config = None
            best_throughput = 0
            
            for config in comparison_results.get("configurations", []):
                throughput = config.get("throughput_items_per_second", 0)
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = config
            
            if best_config:
                config_info = best_config.get("configuration", {})
                accelerator = config_info.get("accelerator", "auto")
                batch_size = config_info.get("batch_size", 1)
        
        thermal_results = run_thermal_analysis(
            harness=harness,
            model_path=args.model,
            model_name=args.name,
            model_type=args.type,
            duration_seconds=args.duration,
            accelerator=accelerator,
            batch_size=batch_size,
            output_dir=args.output_dir
        )
        
        # Print summary
        if thermal_results.get("status") == "success":
            throttling_stats = thermal_results.get("throttling_stats", {})
            print("\nThermal Analysis Summary:")
            print(f"- Throttling detected: {throttling_stats.get('throttling_detected', False)}")
            
            if throttling_stats.get("throttling_detected", False):
                print(f"- Throttling level: {throttling_stats.get('throttling_level', 0)} "
                      f"({throttling_stats.get('level_description', 'Unknown')})")
                print(f"- Throttling duration: {throttling_stats.get('throttling_time_seconds', 0):.1f} seconds")
                print(f"- Performance impact: {throttling_stats.get('performance_impact', 0)*100:.1f}%")
            
            # Print recommendations
            print("\nRecommendations:")
            for rec in thermal_results.get("recommendations", []):
                print(f"- {rec}")
    
    return 0


if __name__ == "__main__":
    exit(main())