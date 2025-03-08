#!/usr/bin/env python3
"""
Test script for OpenVINO backend integration.

This script demonstrates the usage of the OpenVINO backend for model acceleration.
"""

import os
import sys
import logging
import time
import argparse
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_openvino")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the OpenVINO backend
try:
    from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend
    BACKEND_IMPORTED = True
except ImportError as e:
    logger.error(f"Failed to import OpenVINO backend: {e}")
    BACKEND_IMPORTED = False

def test_backend_initialization():
    """Test OpenVINO backend initialization."""
    if not BACKEND_IMPORTED:
        logger.error("OpenVINO backend not imported, skipping initialization test")
        return False
    
    logger.info("Testing OpenVINO backend initialization...")
    
    try:
        backend = OpenVINOBackend()
        available = backend.is_available()
        
        if available:
            logger.info("OpenVINO backend initialized successfully")
            
            # Get device information
            devices = backend.get_all_devices()
            logger.info(f"Available devices: {len(devices)}")
            
            for i, device_info in enumerate(devices):
                logger.info(f"Device {i+1}: {device_info.get('device_name', 'Unknown')} - {device_info.get('full_name', 'Unknown')}")
                logger.info(f"  Type: {device_info.get('device_type', 'Unknown')}")
                logger.info(f"  FP32: {device_info.get('supports_fp32', False)}")
                logger.info(f"  FP16: {device_info.get('supports_fp16', False)}")
                logger.info(f"  INT8: {device_info.get('supports_int8', False)}")
            
            # Check for optimum.intel integration
            optimum_info = backend.get_optimum_integration()
            if optimum_info.get("available", False):
                logger.info(f"optimum.intel is available (version: {optimum_info.get('version', 'Unknown')})")
                
                # Log available model types
                logger.info(f"  Sequence Classification: {optimum_info.get('sequence_classification_available', False)}")
                logger.info(f"  Causal LM: {optimum_info.get('causal_lm_available', False)}")
                logger.info(f"  Seq2Seq LM: {optimum_info.get('seq2seq_lm_available', False)}")
            else:
                logger.info("optimum.intel is not available")
            
            return True
        else:
            logger.warning("OpenVINO is not available on this system")
            return False
    except Exception as e:
        logger.error(f"Error initializing OpenVINO backend: {e}")
        return False

def test_model_operations(model_name="bert-base-uncased", device="CPU"):
    """Test model operations with OpenVINO backend."""
    if not BACKEND_IMPORTED:
        logger.error("OpenVINO backend not imported, skipping model operations test")
        return False
    
    logger.info(f"Testing model operations with {model_name} on device {device}...")
    
    try:
        backend = OpenVINOBackend()
        
        if not backend.is_available():
            logger.warning("OpenVINO is not available on this system, skipping test")
            return False
        
        # Test loading a model
        load_result = backend.load_model(model_name, {"device": device, "model_type": "text"})
        
        if load_result.get("status") != "success":
            logger.error(f"Failed to load model: {load_result.get('message', 'Unknown error')}")
            return False
        
        logger.info(f"Model {model_name} loaded successfully on {device}")
        
        # Test inference
        logger.info(f"Running inference with {model_name} on {device}...")
        
        # Sample input content (dummy data)
        input_content = {
            "input_ids": [101, 2054, 2154, 2003, 2026, 3793, 2080, 2339, 1029, 102],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        
        inference_result = backend.run_inference(
            model_name,
            input_content,
            {"device": device, "model_type": "text"}
        )
        
        if inference_result.get("status") != "success":
            logger.error(f"Inference failed: {inference_result.get('message', 'Unknown error')}")
            return False
        
        # Print inference metrics
        logger.info(f"Inference completed successfully")
        logger.info(f"  Latency: {inference_result.get('latency_ms', 0):.2f} ms")
        logger.info(f"  Throughput: {inference_result.get('throughput_items_per_sec', 0):.2f} items/sec")
        logger.info(f"  Memory usage: {inference_result.get('memory_usage_mb', 0):.2f} MB")
        
        # Test unloading the model
        logger.info(f"Unloading model {model_name} from {device}...")
        
        unload_result = backend.unload_model(model_name, device)
        
        if unload_result.get("status") != "success":
            logger.error(f"Failed to unload model: {unload_result.get('message', 'Unknown error')}")
            return False
        
        logger.info(f"Model {model_name} unloaded successfully from {device}")
        
        return True
    except Exception as e:
        logger.error(f"Error during model operations test: {e}")
        return False

def test_model_conversion(output_dir="./openvino_models"):
    """Test model conversion capabilities."""
    if not BACKEND_IMPORTED:
        logger.error("OpenVINO backend not imported, skipping model conversion test")
        return False
    
    logger.info("Testing model conversion capabilities...")
    
    try:
        backend = OpenVINOBackend()
        
        if not backend.is_available():
            logger.warning("OpenVINO is not available on this system, skipping test")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Test ONNX conversion
        onnx_path = "dummy_model.onnx"  # This is just a placeholder
        output_path = os.path.join(output_dir, "converted_model")
        
        logger.info(f"Testing ONNX to OpenVINO conversion (simulated)...")
        
        onnx_result = backend.convert_from_onnx(
            onnx_path,
            output_path,
            {"precision": "FP16"}
        )
        
        if onnx_result.get("status") != "success":
            logger.error(f"ONNX conversion failed: {onnx_result.get('message', 'Unknown error')}")
        else:
            logger.info(f"ONNX conversion completed successfully")
            
        return True
    except Exception as e:
        logger.error(f"Error during model conversion test: {e}")
        return False

def run_benchmarks(model_name="bert-base-uncased", device="CPU", iterations=5):
    """Run benchmarks with the OpenVINO backend."""
    if not BACKEND_IMPORTED:
        logger.error("OpenVINO backend not imported, skipping benchmarks")
        return None
    
    logger.info(f"Running benchmarks with {model_name} on {device} for {iterations} iterations...")
    
    try:
        backend = OpenVINOBackend()
        
        if not backend.is_available():
            logger.warning("OpenVINO is not available on this system, skipping benchmarks")
            return None
        
        # Test loading a model
        load_result = backend.load_model(model_name, {"device": device, "model_type": "text"})
        
        if load_result.get("status") != "success":
            logger.error(f"Failed to load model: {load_result.get('message', 'Unknown error')}")
            return None
        
        # Sample input content (dummy data)
        input_content = {
            "input_ids": [101, 2054, 2154, 2003, 2026, 3793, 2080, 2339, 1029, 102],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        
        # Run warmup iteration
        logger.info("Running warmup iteration...")
        backend.run_inference(
            model_name,
            input_content,
            {"device": device, "model_type": "text"}
        )
        
        # Run benchmark iterations
        latencies = []
        throughputs = []
        memory_usages = []
        
        logger.info(f"Running {iterations} benchmark iterations...")
        
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            inference_result = backend.run_inference(
                model_name,
                input_content,
                {"device": device, "model_type": "text"}
            )
            
            if inference_result.get("status") != "success":
                logger.error(f"Inference failed: {inference_result.get('message', 'Unknown error')}")
                continue
            
            latencies.append(inference_result.get("latency_ms", 0))
            throughputs.append(inference_result.get("throughput_items_per_sec", 0))
            memory_usages.append(inference_result.get("memory_usage_mb", 0))
        
        # Calculate statistics
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            avg_throughput = sum(throughputs) / len(throughputs)
            min_throughput = min(throughputs)
            max_throughput = max(throughputs)
            
            avg_memory = sum(memory_usages) / len(memory_usages)
            
            # Print benchmark results
            logger.info("Benchmark Results:")
            logger.info(f"  Average Latency: {avg_latency:.2f} ms")
            logger.info(f"  Min Latency: {min_latency:.2f} ms")
            logger.info(f"  Max Latency: {max_latency:.2f} ms")
            logger.info(f"  Average Throughput: {avg_throughput:.2f} items/sec")
            logger.info(f"  Average Memory Usage: {avg_memory:.2f} MB")
            
            # Return benchmark results
            return {
                "model": model_name,
                "device": device,
                "iterations": iterations,
                "avg_latency_ms": avg_latency,
                "min_latency_ms": min_latency,
                "max_latency_ms": max_latency,
                "avg_throughput_items_per_sec": avg_throughput,
                "min_throughput_items_per_sec": min_throughput,
                "max_throughput_items_per_sec": max_throughput,
                "avg_memory_usage_mb": avg_memory
            }
        else:
            logger.error("No valid benchmark results collected")
            return None
    except Exception as e:
        logger.error(f"Error during benchmarks: {e}")
        return None
    finally:
        # Try to unload the model
        try:
            backend.unload_model(model_name, device)
        except Exception:
            pass

def compare_with_cpu(model_name="bert-base-uncased", iterations=5):
    """Compare OpenVINO performance against CPU."""
    logger.info(f"Comparing OpenVINO performance against CPU for {model_name}...")
    
    if not BACKEND_IMPORTED:
        logger.error("OpenVINO backend not imported, skipping comparison")
        return
    
    try:
        # Import CPU backend
        try:
            from ipfs_accelerate_py.hardware.backends.cpu_backend import CPUBackend
            cpu_backend_available = True
        except ImportError as e:
            logger.error(f"Failed to import CPU backend: {e}")
            cpu_backend_available = False
            return
        
        openvino_backend = OpenVINOBackend()
        cpu_backend = CPUBackend()
        
        if not openvino_backend.is_available():
            logger.warning("OpenVINO is not available on this system, skipping comparison")
            return
        
        # Sample input content (dummy data)
        input_content = {
            "input_ids": [101, 2054, 2154, 2003, 2026, 3793, 2080, 2339, 1029, 102],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        
        # Load and run on OpenVINO CPU
        logger.info("Testing on OpenVINO CPU...")
        openvino_backend.load_model(model_name, {"device": "CPU", "model_type": "text"})
        
        # Warmup
        openvino_backend.run_inference(model_name, input_content, {"device": "CPU", "model_type": "text"})
        
        # Benchmark
        openvino_latencies = []
        
        for i in range(iterations):
            result = openvino_backend.run_inference(model_name, input_content, {"device": "CPU", "model_type": "text"})
            openvino_latencies.append(result.get("latency_ms", 0))
        
        openvino_avg_latency = sum(openvino_latencies) / len(openvino_latencies)
        
        # Load and run on pure CPU
        logger.info("Testing on pure CPU...")
        cpu_backend.load_model(model_name, {"model_type": "text"})
        
        # Warmup
        cpu_backend.run_inference(model_name, input_content, {"model_type": "text"})
        
        # Benchmark
        cpu_latencies = []
        
        for i in range(iterations):
            result = cpu_backend.run_inference(model_name, input_content, {"model_type": "text"})
            cpu_latencies.append(result.get("latency_ms", 0))
        
        cpu_avg_latency = sum(cpu_latencies) / len(cpu_latencies)
        
        # Calculate speedup
        speedup = cpu_avg_latency / openvino_avg_latency if openvino_avg_latency > 0 else 0
        
        # Print comparison results
        logger.info("Performance Comparison Results:")
        logger.info(f"  OpenVINO CPU Average Latency: {openvino_avg_latency:.2f} ms")
        logger.info(f"  Pure CPU Average Latency: {cpu_avg_latency:.2f} ms")
        logger.info(f"  Speedup with OpenVINO: {speedup:.2f}x")
        
        # Unload models
        openvino_backend.unload_model(model_name, "CPU")
        cpu_backend.unload_model(model_name)
        
    except Exception as e:
        logger.error(f"Error during performance comparison: {e}")

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test OpenVINO backend integration")
    
    # Test options
    parser.add_argument("--test-init", action="store_true", help="Test backend initialization")
    parser.add_argument("--test-model", action="store_true", help="Test model operations")
    parser.add_argument("--test-conversion", action="store_true", help="Test model conversion")
    parser.add_argument("--run-benchmarks", action="store_true", help="Run benchmarks")
    parser.add_argument("--compare-cpu", action="store_true", help="Compare with CPU performance")
    parser.add_argument("--run-all", action="store_true", help="Run all tests")
    
    # Configuration options
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model name to use for tests")
    parser.add_argument("--device", type=str, default="CPU", help="OpenVINO device to use (CPU, GPU, AUTO, etc.)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for benchmarks")
    parser.add_argument("--output-dir", type=str, default="./openvino_models", help="Output directory for model conversion")
    parser.add_argument("--output-json", type=str, help="Output JSON file for benchmark results")
    
    args = parser.parse_args()
    
    # If no specific test is selected, run backend initialization test
    if not (args.test_init or args.test_model or args.test_conversion or args.run_benchmarks or args.compare_cpu or args.run_all):
        args.test_init = True
    
    # Run tests based on arguments
    results = {}
    
    if args.test_init or args.run_all:
        results["initialization"] = test_backend_initialization()
    
    if args.test_model or args.run_all:
        results["model_operations"] = test_model_operations(args.model, args.device)
    
    if args.test_conversion or args.run_all:
        results["model_conversion"] = test_model_conversion(args.output_dir)
    
    if args.run_benchmarks or args.run_all:
        benchmark_results = run_benchmarks(args.model, args.device, args.iterations)
        results["benchmarks"] = benchmark_results
    
    if args.compare_cpu or args.run_all:
        compare_with_cpu(args.model, args.iterations)
    
    # Save benchmark results to JSON if requested
    if args.output_json and "benchmarks" in results and results["benchmarks"]:
        try:
            with open(args.output_json, "w") as f:
                json.dump(results["benchmarks"], f, indent=2)
            logger.info(f"Benchmark results saved to {args.output_json}")
        except Exception as e:
            logger.error(f"Failed to save benchmark results to JSON: {e}")
    
    # Print overall test results
    logger.info("\nOverall Test Results:")
    for test_name, result in results.items():
        if isinstance(result, bool):
            status = "PASSED" if result else "FAILED"
            logger.info(f"  {test_name}: {status}")

if __name__ == "__main__":
    main()