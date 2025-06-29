#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Android Test Harness CI Benchmark Runner

This script runs automated benchmarks for the Android Test Harness as part of the CI/CD pipeline.
It automates the execution of benchmarks on Android devices or emulators, stores results in a
DuckDB database, and generates reports.

Usage:
    python run_ci_benchmarks.py --device-id DEVICE_ID --output-db OUTPUT_DB [--model-list MODEL_LIST]
    [--timeout TIMEOUT] [--verbose]

Examples:
    # Run on specific device with default models
    python run_ci_benchmarks.py --device-id emulator-5554 --output-db benchmark_results.duckdb

    # Run specific models with timeout
    python run_ci_benchmarks.py --device-id emulator-5554 --output-db benchmark_results.duckdb 
        --model-list models.json --timeout 3600

Date: April 2025
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Local imports
try:
    from test.android_test_harness.android_test_harness import AndroidTestHarness
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)


class CIBenchmarkRunner:
    """
    Runs automated benchmarks for the Android Test Harness in CI environment.
    
    This class handles device connection, model deployment, benchmark execution,
    result storage, and reporting for CI/CD integration.
    """
    
    def __init__(self, 
                 device_id: str, 
                 output_db: str,
                 model_list: Optional[str] = None,
                 timeout: int = 7200,
                 verbose: bool = False):
        """
        Initialize the CI benchmark runner.
        
        Args:
            device_id: Android device ID (emulator ID or physical device ID)
            output_db: Path to output DuckDB database
            model_list: Optional path to JSON file with models to test
            timeout: Timeout in seconds for the entire benchmark run
            verbose: Enable verbose logging
        """
        self.device_id = device_id
        self.output_db = output_db
        self.model_list_path = model_list
        self.timeout = timeout
        self.verbose = verbose
        
        # Set logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize database connection
        self.db_api = None
        try:
            self.db_api = BenchmarkDBAPI(output_db)
            logger.info(f"Connected to database: {output_db}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            logger.warning("Will continue without database connection")
        
        # Initialize Android test harness
        self.test_harness = None
        self.models_to_test = []
        self.results = {}
        self.start_time = time.time()
    
    def load_model_list(self) -> List[Dict[str, Any]]:
        """
        Load the list of models to test.
        
        Returns:
            List of model configurations
        """
        if self.model_list_path and os.path.exists(self.model_list_path):
            try:
                with open(self.model_list_path, 'r') as f:
                    models = json.load(f)
                logger.info(f"Loaded {len(models)} models from {self.model_list_path}")
                return models
            except Exception as e:
                logger.error(f"Error loading model list: {e}")
        
        # Default model list if not provided or error loading
        logger.info("Using default model list")
        return [
            {
                "name": "bert-base-uncased",
                "path": "models/bert-base-uncased.onnx",
                "type": "onnx",
                "batch_sizes": [1, 4],
                "iterations": 50,
                "priority": "high"
            },
            {
                "name": "mobilenet-v2",
                "path": "models/mobilenet-v2.onnx",
                "type": "onnx",
                "batch_sizes": [1, 4],
                "iterations": 50,
                "priority": "high"
            },
            {
                "name": "roberta-base",
                "path": "models/roberta-base.onnx",
                "type": "onnx",
                "batch_sizes": [1],
                "iterations": 30,
                "priority": "medium"
            },
            {
                "name": "whisper-tiny",
                "path": "models/whisper-tiny.onnx",
                "type": "onnx",
                "batch_sizes": [1],
                "iterations": 20,
                "priority": "medium"
            },
            {
                "name": "clip-vit-base-patch32",
                "path": "models/clip-vit-base-patch32.onnx",
                "type": "onnx",
                "batch_sizes": [1],
                "iterations": 20,
                "priority": "medium"
            }
        ]
    
    def connect_to_device(self) -> bool:
        """
        Connect to the Android device.
        
        Returns:
            Success status
        """
        try:
            logger.info(f"Connecting to Android device: {self.device_id}")
            self.test_harness = AndroidTestHarness(device_id=self.device_id, db_path=self.output_db)
            connected = self.test_harness.connect_to_device()
            
            if connected:
                device_info = self.test_harness.device.to_dict()
                logger.info(f"Connected to device: {device_info.get('model', 'Unknown')}")
                logger.info(f"Android version: {device_info.get('android_version', 'Unknown')}")
                logger.info(f"Device capabilities: {device_info.get('capabilities', {})}")
                return True
            else:
                logger.error(f"Failed to connect to device: {self.device_id}")
                return False
        
        except Exception as e:
            logger.error(f"Error connecting to device: {e}")
            return False
    
    def prepare_models(self) -> bool:
        """
        Prepare models for testing.
        
        Returns:
            Success status
        """
        if not self.test_harness:
            logger.error("Test harness not initialized")
            return False
        
        # Load model list
        self.models_to_test = self.load_model_list()
        
        # Check if model files exist
        for model in self.models_to_test:
            model_path = model.get("path")
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                model["skip"] = True
                continue
            
            logger.info(f"Model file found: {model_path}")
            model["skip"] = False
        
        # Count models to test
        models_to_run = [m for m in self.models_to_test if not m.get("skip", False)]
        logger.info(f"Models prepared for testing: {len(models_to_run)}/{len(self.models_to_test)}")
        
        return len(models_to_run) > 0
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """
        Run benchmarks on all models.
        
        Returns:
            Dictionary with benchmark results
        """
        if not self.test_harness:
            logger.error("Test harness not initialized")
            return {"status": "error", "message": "Test harness not initialized"}
        
        # Check if we have models to test
        if not self.models_to_test:
            logger.error("No models to test")
            return {"status": "error", "message": "No models to test"}
        
        # Run benchmarks for each model
        benchmark_results = {
            "status": "success",
            "device_info": self.test_harness.device.to_dict(),
            "start_time": datetime.datetime.now().isoformat(),
            "models": {}
        }
        
        for i, model in enumerate(self.models_to_test):
            # Check if we should skip this model
            if model.get("skip", False):
                logger.info(f"Skipping model: {model.get('name')}")
                continue
            
            # Check timeout
            if (time.time() - self.start_time) > self.timeout:
                logger.warning("Benchmark timeout reached")
                benchmark_results["status"] = "timeout"
                benchmark_results["message"] = "Benchmark timeout reached"
                break
            
            # Extract model information
            model_name = model.get("name", f"model_{i}")
            model_path = model.get("path")
            model_type = model.get("type", "onnx")
            batch_sizes = model.get("batch_sizes", [1])
            iterations = model.get("iterations", 50)
            
            logger.info(f"Running benchmark for model: {model_name} ({i+1}/{len(self.models_to_test)})")
            logger.info(f"  Path: {model_path}")
            logger.info(f"  Type: {model_type}")
            logger.info(f"  Batch sizes: {batch_sizes}")
            logger.info(f"  Iterations: {iterations}")
            
            try:
                # Run benchmark
                result = self.test_harness.run_benchmark(
                    model_path=model_path,
                    model_name=model_name,
                    model_type=model_type,
                    batch_sizes=batch_sizes,
                    iterations=iterations,
                    save_to_db=True,
                    collect_metrics=True
                )
                
                # Store results
                benchmark_results["models"][model_name] = result
                
                # Log summary
                if result.get("status") == "success":
                    logger.info(f"Benchmark successful for {model_name}")
                    for config in result.get("configurations", []):
                        batch_size = config.get("configuration", {}).get("batch_size", 1)
                        throughput = config.get("throughput_items_per_second", 0)
                        latency = config.get("latency_ms", {}).get("mean", 0)
                        logger.info(f"  Batch size {batch_size}: {throughput:.2f} items/s, {latency:.2f} ms")
                else:
                    logger.error(f"Benchmark failed for {model_name}: {result.get('message', 'Unknown error')}")
            
            except Exception as e:
                logger.error(f"Error running benchmark for {model_name}: {e}")
                benchmark_results["models"][model_name] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Add end time
        benchmark_results["end_time"] = datetime.datetime.now().isoformat()
        benchmark_results["duration_seconds"] = time.time() - self.start_time
        
        # Count successful models
        successful_models = 0
        for model_name, result in benchmark_results["models"].items():
            if result.get("status") == "success":
                successful_models += 1
        
        benchmark_results["successful_models"] = successful_models
        benchmark_results["total_models"] = len(self.models_to_test)
        
        return benchmark_results
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """
        Save benchmark results to file.
        
        Args:
            results: Benchmark results
            
        Returns:
            Path to results file
        """
        # Create results directory if needed
        os.makedirs("android_benchmark_results", exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        device_model = results.get("device_info", {}).get("model", "unknown").replace(" ", "_")
        filename = f"android_benchmark_results/android_benchmark_{device_model}_{timestamp}.json"
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        return filename
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a benchmark report.
        
        Args:
            results: Benchmark results
            
        Returns:
            Path to report file
        """
        # Create results directory if needed
        os.makedirs("android_benchmark_results", exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        device_model = results.get("device_info", {}).get("model", "unknown").replace(" ", "_")
        filename = f"android_benchmark_results/android_benchmark_report_{device_model}_{timestamp}.md"
        
        # Generate report
        report = "# Android Benchmark Report\n\n"
        report += f"Generated: {datetime.datetime.now().isoformat()}\n\n"
        
        # Device information
        device_info = results.get("device_info", {})
        report += "## Device Information\n\n"
        report += f"- **Model**: {device_info.get('model', 'Unknown')}\n"
        report += f"- **Android Version**: {device_info.get('android_version', 'Unknown')}\n"
        report += f"- **Chipset**: {device_info.get('chipset', 'Unknown')}\n"
        report += f"- **Device ID**: {device_info.get('device_id', 'Unknown')}\n\n"
        
        # Summary
        report += "## Summary\n\n"
        report += f"- **Start Time**: {results.get('start_time', 'Unknown')}\n"
        report += f"- **End Time**: {results.get('end_time', 'Unknown')}\n"
        report += f"- **Duration**: {results.get('duration_seconds', 0):.1f} seconds\n"
        report += f"- **Status**: {results.get('status', 'Unknown')}\n"
        report += f"- **Models Tested**: {results.get('successful_models', 0)}/{results.get('total_models', 0)}\n\n"
        
        # Model results
        report += "## Model Results\n\n"
        
        # Create summary table
        report += "### Performance Summary\n\n"
        report += "| Model | Batch Size | Throughput (items/s) | Latency (ms) | Memory (MB) | Battery Impact |\n"
        report += "|-------|------------|----------------------|--------------|-------------|----------------|\n"
        
        for model_name, model_results in results.get("models", {}).items():
            if model_results.get("status") != "success":
                continue
            
            for config in model_results.get("configurations", []):
                batch_size = config.get("configuration", {}).get("batch_size", 1)
                throughput = config.get("throughput_items_per_second", 0)
                latency = config.get("latency_ms", {}).get("mean", 0)
                memory = config.get("memory_metrics", {}).get("peak_mb", 0)
                battery = config.get("battery_metrics", {}).get("impact_percentage", 0)
                
                report += f"| {model_name} | {batch_size} | {throughput:.2f} | {latency:.2f} | {memory:.1f} | {battery:.1f}% |\n"
        
        # Detailed results
        report += "\n### Detailed Results\n\n"
        
        for model_name, model_results in results.get("models", {}).items():
            report += f"#### {model_name}\n\n"
            
            if model_results.get("status") != "success":
                report += f"**Status**: {model_results.get('status', 'Unknown')}\n"
                report += f"**Message**: {model_results.get('message', 'Unknown')}\n\n"
                continue
            
            for i, config in enumerate(model_results.get("configurations", [])):
                batch_size = config.get("configuration", {}).get("batch_size", 1)
                report += f"##### Configuration {i+1}: Batch Size {batch_size}\n\n"
                
                # Latency statistics
                latency = config.get("latency_ms", {})
                report += "**Latency (ms)**:\n\n"
                report += f"- Min: {latency.get('min', 0):.2f}\n"
                report += f"- Mean: {latency.get('mean', 0):.2f}\n"
                report += f"- Median: {latency.get('median', 0):.2f}\n"
                report += f"- P90: {latency.get('p90', 0):.2f}\n"
                report += f"- P95: {latency.get('p95', 0):.2f}\n"
                report += f"- P99: {latency.get('p99', 0):.2f}\n"
                report += f"- Max: {latency.get('max', 0):.2f}\n\n"
                
                # Throughput
                report += "**Throughput**:\n\n"
                report += f"- {config.get('throughput_items_per_second', 0):.2f} items/s\n\n"
                
                # Battery impact
                battery_metrics = config.get("battery_metrics", {})
                report += "**Battery Impact**:\n\n"
                report += f"- Impact: {battery_metrics.get('impact_percentage', 0):.1f}%\n"
                report += f"- Temperature: {battery_metrics.get('temperature_delta', 0):.1f}°C\n\n"
                
                # Memory usage
                memory_metrics = config.get("memory_metrics", {})
                report += "**Memory Usage**:\n\n"
                report += f"- Peak: {memory_metrics.get('peak_mb', 0):.1f} MB\n\n"
        
        # Save report
        with open(filename, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {filename}")
        return filename
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete CI benchmark process.
        
        Returns:
            Dictionary with results
        """
        # Track start time
        self.start_time = time.time()
        logger.info(f"Starting Android CI benchmark run (timeout: {self.timeout}s)")
        
        # Connect to device
        if not self.connect_to_device():
            return {"status": "error", "message": "Failed to connect to device"}
        
        # Prepare models
        if not self.prepare_models():
            return {"status": "error", "message": "Failed to prepare models"}
        
        # Run benchmarks
        results = self.run_benchmarks()
        
        # Save results to file
        results_path = self.save_results(results)
        results["results_path"] = results_path
        
        # Generate report
        report_path = self.generate_report(results)
        results["report_path"] = report_path
        
        # Log summary
        duration = time.time() - self.start_time
        logger.info(f"Benchmark run completed in {duration:.1f} seconds")
        logger.info(f"Status: {results.get('status', 'Unknown')}")
        logger.info(f"Models tested: {results.get('successful_models', 0)}/{results.get('total_models', 0)}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Report saved to: {report_path}")
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Android Test Harness CI Benchmark Runner")
    
    parser.add_argument("--device-id", required=True, help="Android device ID")
    parser.add_argument("--output-db", required=True, help="Path to output DuckDB database")
    parser.add_argument("--model-list", help="Path to JSON file with models to test")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout in seconds (default: 7200)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        # Run benchmarks
        runner = CIBenchmarkRunner(
            device_id=args.device_id,
            output_db=args.output_db,
            model_list=args.model_list,
            timeout=args.timeout,
            verbose=args.verbose
        )
        
        results = runner.run()
        
        # Return exit code
        if results.get("status") == "success":
            return 0
        elif results.get("status") == "timeout":
            logger.warning("Benchmark run timed out")
            return 2
        else:
            logger.error(f"Benchmark run failed: {results.get('message', 'Unknown error')}")
            return 1
    
    except Exception as e:
        logger.error(f"Error running benchmarks: {e}")
        return 1


if __name__ == "__main__":
    exit(main())