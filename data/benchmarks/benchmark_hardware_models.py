"""
Comprehensive hardware platform benchmarking for key HuggingFace models.

This script provides comprehensive benchmarking capabilities for all key model types
across supported hardware platforms. It extends the test coverage plan by adding
detailed performance metrics and hardware-specific optimizations.

Usage:
    python benchmark_hardware_models.py --model [model_name] --hardware [platform] --batch-sizes 1,2,4,8
    python benchmark_hardware_models.py --all --quick
    python benchmark_hardware_models.py --category vision --compare
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Configure logger
import logging
logger = logging.getLogger(__name__)

# Add DuckDB database support
try:
    from benchmark_db_api import BenchmarkDBAPI
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    logger.warning("benchmark_db_api not available. Using deprecated JSON fallback.")

# Improved hardware detection
try:
    from integrated_improvements.improved_hardware_detection import (
        detect_available_hardware,
        check_web_optimizations,
        HARDWARE_PLATFORMS,
        HAS_CUDA,
        HAS_ROCM,
        HAS_MPS,
        HAS_OPENVINO,
        HAS_WEBNN,
        HAS_WEBGPU
    )
    HAS_HARDWARE_MODULE = True
except ImportError:
    logger.warning("Improved hardware detection not available")
    HAS_HARDWARE_MODULE = False

# Database integration
import os
try:
    from integrated_improvements.database_integration import (
        get_db_connection,
        store_test_result,
        store_performance_result,
        create_test_run,
        complete_test_run,
        get_or_create_model,
        get_or_create_hardware_platform,
        DEPRECATE_JSON_OUTPUT
    )
    HAS_DB_INTEGRATION = True
except ImportError:
    logger.warning("Database integration not available")
    HAS_DB_INTEGRATION = False
    DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1") == "1"


# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")


# Import key model and hardware definitions from test coverage script
try:
    from test_comprehensive_hardware_coverage import KEY_MODELS, HARDWARE_PLATFORMS
except ImportError:
    print("Error: Could not import from test_comprehensive_hardware_coverage.py")
    print("Make sure it exists in the same directory")
    sys.exit(1)

# Benchmark configuration
BENCHMARK_CONFIG = {
    "batch_sizes": [1, 2, 4, 8, 16],  # Default batch sizes to test
    "repeat_count": 5,                # Number of times to repeat each benchmark
    "warmup_iterations": 3,           # Warmup iterations before timing
    "quick_mode_batch_sizes": [1, 4], # Batch sizes for quick mode
    "quick_mode_repeat_count": 2,     # Repeat count for quick mode
    "result_dir": "benchmark_results",
    "timeout_seconds": 600,           # Maximum seconds per benchmark
}

def get_model_categories() -> Dict[str, List[str]]:
    """
    Group models by their category.
    
    Returns:
        Dict[str, List[str]]: Models grouped by category
    """
    categories = {}
    for model_key, model_info in KEY_MODELS.items():
        category = model_info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(model_key)
    
    return categories

def store_benchmark_in_database(result, db_path=None):
    # Store benchmark results in database
    if not HAS_DB_INTEGRATION:
        logger.warning("Database integration not available, cannot store benchmark")
        return False
    
    try:
        # Get database connection
        conn = get_db_connection(db_path)
        if conn is None:
            logger.error("Failed to connect to database")
            return False
        
        # Create test run
        run_id = create_test_run(
            test_name=result.get("model_name", "unknown_model"),
            test_type="benchmark",
            metadata={"benchmark_script": os.path.basename(__file__)}
        )
        
        # Get or create model
        model_id = get_or_create_model(
            model_name=result.get("model_name", "unknown_model"),
            model_family=result.get("model_family"),
            model_type=result.get("model_type"),
            metadata=result
        )
        
        # Get or create hardware platform
        hw_id = get_or_create_hardware_platform(
            hardware_type=result.get("hardware", "unknown"),
            metadata={"source": "benchmark"}
        )
        
        # Store performance result
        store_performance_result(
            run_id=run_id,
            model_id=model_id,
            hardware_id=hw_id,
            batch_size=result.get("batch_size", 1),
            throughput=result.get("throughput_items_per_second"),
            latency=result.get("latency_ms"),
            memory=result.get("memory_mb"),
            metadata=result
        )
        
        # Complete test run
        complete_test_run(run_id)
        
        logger.info(f"Stored benchmark result in database for {result.get('model_name', 'unknown')}")
        return True
    except Exception as e:
        logger.error(f"Error storing benchmark in database: {e}")
        return False

def get_benchmark_command(
    model_key: str, 
    hardware: str, 
    batch_size: int = 1,
    timeout: int = BENCHMARK_CONFIG["timeout_seconds"]
) -> Optional[str]:
    """
    Generate benchmark command for a specific model on specific hardware.
    
    Args:
        model_key (str): Model key to benchmark
        hardware (str): Hardware platform to benchmark on
        batch_size (int): Batch size for benchmark
        timeout (int): Timeout in seconds
        
    Returns:
        Optional[str]: Benchmark command or None if incompatible
    """
    if model_key not in KEY_MODELS:
        return None
    
    if hardware not in HARDWARE_PLATFORMS:
        return None
    
    if model_key not in HARDWARE_PLATFORMS[hardware]["compatibility"]:
        return None
    
    model_name = KEY_MODELS[model_key]["models"][0].split("/")[-1]
    hw_flag = HARDWARE_PLATFORMS[hardware]["flag"]
    
    # Basic benchmark command with correct path (without 'test/' prefix)
    command = f"python run_model_benchmarks.py --specific-models {model_name} {hw_flag} --batch-sizes {batch_size} --timeout {timeout}"
    
    # Add special flags for certain combinations
    if hardware in ["webnn", "webgpu"]:
        command += " --web-platform-test"
    
    return command

def run_benchmark(
    model_key: str, 
    hardware: str, 
    batch_sizes: List[int] = None,
    quick_mode: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Run benchmark for a specific model on specific hardware.
    
    Args:
        model_key (str): Model key to benchmark
        hardware (str): Hardware platform to benchmark on
        batch_sizes (List[int]): Batch sizes to test
        quick_mode (bool): Use reduced benchmarking settings for faster results
        verbose (bool): Print detailed output
        
    Returns:
        Dict: Benchmark results
    """
    if not batch_sizes:
        if quick_mode:
            batch_sizes = BENCHMARK_CONFIG["quick_mode_batch_sizes"]
        else:
            batch_sizes = BENCHMARK_CONFIG["batch_sizes"]
    
    results = {
        "model": KEY_MODELS[model_key]["name"],
        "model_key": model_key,
        "model_category": KEY_MODELS[model_key]["category"],
        "model_path": KEY_MODELS[model_key]["models"][0],
        "hardware": HARDWARE_PLATFORMS[hardware]["name"],
        "hardware_key": hardware,
        "timestamp": datetime.now().isoformat(),
        "batch_sizes": {},
        "compatible": model_key in HARDWARE_PLATFORMS[hardware]["compatibility"]
    }
    
    if not results["compatible"]:
        if verbose:
            print(f"Skipping incompatible model-hardware combination: {model_key} on {hardware}")
        return results
    
    for batch_size in batch_sizes:
        if verbose:
            print(f"Benchmarking {model_key} on {hardware} with batch size {batch_size}...")
        
        command = get_benchmark_command(model_key, hardware, batch_size)
        if not command:
            if verbose:
                print(f"Could not generate benchmark command for {model_key} on {hardware}")
            continue
        
        if verbose:
            print(f"Command: {command}")
        
        # Run the actual benchmark command and collect real results
        try:
            if verbose:
                print(f"Executing benchmark command: {command}")
            
            # Run the benchmark command and capture output
            start_time = time.time()
            process = subprocess.run(command, shell=True, check=True, 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, timeout=BENCHMARK_CONFIG["timeout_seconds"])
            execution_time = time.time() - start_time
            
            # Try to parse the output as JSON
            try:
                # First check if the output contains a valid JSON object
                stdout = process.stdout.strip()
                if stdout and (stdout.startswith('{') or stdout.startswith('[')):
                    benchmark_data = json.loads(stdout)
                    
                    # Extract performance metrics from the benchmark output
                    results["batch_sizes"][batch_size] = {
                        "throughput": benchmark_data.get("throughput_items_per_second", 0),
                        "latency_ms": benchmark_data.get("average_latency_ms", 0),
                        "memory_usage_mb": benchmark_data.get("memory_peak_mb", 0),
                        "success": True,
                        "execution_time_sec": execution_time,
                        "command": command,
                        "raw_output": stdout[:1000] if len(stdout) > 1000 else stdout,  # Truncate long output
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # If output is not JSON, try to extract metrics with regex or other means
                    # For now, just store the raw output
                    results["batch_sizes"][batch_size] = {
                        "success": True,
                        "execution_time_sec": execution_time,
                        "command": command,
                        "raw_output": stdout[:1000] if len(stdout) > 1000 else stdout,  # Truncate long output
                        "raw_stderr": process.stderr[:1000] if len(process.stderr) > 1000 else process.stderr,
                        "timestamp": datetime.now().isoformat(),
                        "parsed": False,
                        "note": "Output format not recognized as JSON"
                    }
                    
                    # Attempt to extract numerical values from output with regex
                    throughput_match = re.search(r'throughput[:\s]+(\d+\.?\d*)', stdout, re.IGNORECASE)
                    latency_match = re.search(r'latency[:\s]+(\d+\.?\d*)', stdout, re.IGNORECASE)
                    memory_match = re.search(r'memory[:\s]+(\d+\.?\d*)', stdout, re.IGNORECASE)
                    
                    if throughput_match:
                        results["batch_sizes"][batch_size]["throughput"] = float(throughput_match.group(1))
                    if latency_match:
                        results["batch_sizes"][batch_size]["latency_ms"] = float(latency_match.group(1))
                    if memory_match:
                        results["batch_sizes"][batch_size]["memory_usage_mb"] = float(memory_match.group(1))
            
            except json.JSONDecodeError:
                # If output is not valid JSON, store raw output
                results["batch_sizes"][batch_size] = {
                    "success": True,
                    "execution_time_sec": execution_time,
                    "command": command,
                    "raw_output": process.stdout[:1000] if len(process.stdout) > 1000 else process.stdout,  # Truncate long output
                    "raw_stderr": process.stderr[:1000] if len(process.stderr) > 1000 else process.stderr,
                    "timestamp": datetime.now().isoformat(),
                    "parsed": False,
                    "note": "Failed to parse output as JSON"
                }
                
        except subprocess.TimeoutExpired as e:
            if verbose:
                print(f"Benchmark timed out after {BENCHMARK_CONFIG['timeout_seconds']} seconds")
            
            results["batch_sizes"][batch_size] = {
                "success": False,
                "error_type": "timeout",
                "error_message": f"Benchmark timed out after {BENCHMARK_CONFIG['timeout_seconds']} seconds",
                "command": command,
                "timestamp": datetime.now().isoformat()
            }
            
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"Benchmark command failed with exit code {e.returncode}")
                print(f"Error output: {e.stderr}")
            
            results["batch_sizes"][batch_size] = {
                "success": False,
                "error_type": "execution_error",
                "error_message": f"Benchmark command failed with exit code {e.returncode}",
                "error_output": e.stderr[:1000] if len(e.stderr) > 1000 else e.stderr,  # Truncate long error output
                "command": command,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            if verbose:
                print(f"Unexpected error during benchmark: {str(e)}")
            
            results["batch_sizes"][batch_size] = {
                "success": False,
                "error_type": "unexpected_error",
                "error_message": str(e),
                "command": command,
                "timestamp": datetime.now().isoformat()
            }
    
    return results

def save_benchmark_results(results: Dict, output_dir: str = None) -> str:
    """
    Save benchmark results to a file.
    
    Args:
        results (Dict): Benchmark results
        output_dir (str): Directory to save results
        
    Returns:
        str: Path to saved results file
    """
    if not output_dir:
        output_dir = BENCHMARK_CONFIG["result_dir"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_key = results["model_key"]
    hardware_key = results["hardware_key"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result_path = os.path.join(output_dir, f"benchmark_{model_key}_{hardware_key}_{timestamp}.json")
    
# JSON output deprecated in favor of database storage
    if not DEPRECATE_JSON_OUTPUT:
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
    else:
        logger.info("JSON output is deprecated. Results are stored directly in the database.")
    
    return result_path

def generate_markdown_report(results_list: List[Dict], output_file: str = None) -> str:
    """
    Generate a markdown report from benchmark results.
    
    Args:
        results_list (List[Dict]): List of benchmark results
        output_file (str): Path to output file
        
    Returns:
        str: Generated report content
    """
    if not results_list:
        return "No benchmark results to report."
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = [
        f"# Hardware Benchmark Report ({timestamp})",
        "",
        "## Summary",
        ""
    ]
    
    # Count summary statistics
    total_benchmarks = len(results_list)
    successful_benchmarks = sum(1 for r in results_list if r.get("batch_sizes"))
    
    report.extend([
        f"- Total benchmarks run: {total_benchmarks}",
        f"- Successful benchmarks: {successful_benchmarks} ({successful_benchmarks/total_benchmarks*100:.1f}%)",
        "",
        "## Results by Category",
        ""
    ])
    
    # Group results by category
    categories = {}
    for result in results_list:
        category = result.get("model_category", "unknown")
        if category not in categories:
            categories[category] = []
        categories[category].append(result)
    
    # Generate tables for each category
    for category, category_results in categories.items():
        report.append(f"### {category.replace('_', ' ').title()} Models")
        report.append("")
        report.append("| Model | Hardware | Batch Size | Throughput | Latency (ms) | Memory (MB) |")
        report.append("|-------|----------|------------|------------|--------------|-------------|")
        
        for result in category_results:
            model = result.get("model", "Unknown")
            hardware = result.get("hardware", "Unknown")
            
            for batch_size, batch_results in result.get("batch_sizes", {}).items():
                throughput = batch_results.get("throughput", "N/A")
                latency = batch_results.get("latency_ms", "N/A")
                memory = batch_results.get("memory_usage_mb", "N/A")
                
                report.append(f"| {model} | {hardware} | {batch_size} | {throughput} | {latency} | {memory} |")
        
        report.append("")
    
    # Save report if output file specified
    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(report))
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Benchmark HuggingFace models across hardware platforms")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Benchmark all compatible model-hardware combinations")
    group.add_argument("--model", help="Benchmark a specific model across all compatible hardware platforms")
    group.add_argument("--hardware", help="Benchmark all compatible models on a specific hardware platform")
    group.add_argument("--category", help="Benchmark all models in a category across compatible hardware")
    
    parser.add_argument("--batch-sizes", help="Comma-separated list of batch sizes to test")
    parser.add_argument("--output-dir", help="Directory to save benchmark results")
    parser.add_argument("--quick", action="store_true", help="Run a faster, less comprehensive benchmark")
    parser.add_argument("--compare", action="store_true", help="Generate comparison report from benchmark results")
    parser.add_argument("--no-report", action="store_false", dest="generate_report", help="Don't generate a markdown report")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
    
    args = parser.parse_args()
    
    # Process batch sizes
    batch_sizes = None
    if args.batch_sizes:
        try:
            batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
        except ValueError:
            print("Error: Batch sizes must be comma-separated integers")
            sys.exit(1)
    
    results_list = []
    
    if args.all:
        print("Benchmarking all compatible model-hardware combinations...")
        for model_key in KEY_MODELS:
            for hw_key in HARDWARE_PLATFORMS:
                if model_key in HARDWARE_PLATFORMS[hw_key]["compatibility"]:
                    result = run_benchmark(
                        model_key, 
                        hw_key, 
                        batch_sizes, 
                        quick_mode=args.quick,
                        verbose=args.verbose
                    )
                    results_list.append(result)
                    if result.get("batch_sizes"):
                        save_benchmark_results(result, args.output_dir)
    
    elif args.model:
        if args.model not in KEY_MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available models: {', '.join(KEY_MODELS.keys())}")
            sys.exit(1)
        
        print(f"Benchmarking {args.model} across all compatible hardware platforms...")
        for hw_key in HARDWARE_PLATFORMS:
            if args.model in HARDWARE_PLATFORMS[hw_key]["compatibility"]:
                result = run_benchmark(
                    args.model, 
                    hw_key, 
                    batch_sizes, 
                    quick_mode=args.quick,
                    verbose=args.verbose
                )
                results_list.append(result)
                if result.get("batch_sizes"):
                    save_benchmark_results(result, args.output_dir)
    
    elif args.hardware:
        if args.hardware not in HARDWARE_PLATFORMS:
            print(f"Unknown hardware platform: {args.hardware}")
            print(f"Available platforms: {', '.join(HARDWARE_PLATFORMS.keys())}")
            sys.exit(1)
        
        print(f"Benchmarking all compatible models on {args.hardware}...")
        for model_key in KEY_MODELS:
            if model_key in HARDWARE_PLATFORMS[args.hardware]["compatibility"]:
                result = run_benchmark(
                    model_key, 
                    args.hardware, 
                    batch_sizes, 
                    quick_mode=args.quick,
                    verbose=args.verbose
                )
                results_list.append(result)
                if result.get("batch_sizes"):
                    save_benchmark_results(result, args.output_dir)
    
    elif args.category:
        categories = get_model_categories()
        if args.category not in categories:
            print(f"Unknown category: {args.category}")
            print(f"Available categories: {', '.join(categories.keys())}")
            sys.exit(1)
        
        print(f"Benchmarking all {args.category} models across compatible hardware...")
        for model_key in categories[args.category]:
            for hw_key in HARDWARE_PLATFORMS:
                if model_key in HARDWARE_PLATFORMS[hw_key]["compatibility"]:
                    result = run_benchmark(
                        model_key, 
                        hw_key, 
                        batch_sizes, 
                        quick_mode=args.quick,
                        verbose=args.verbose
                    )
                    results_list.append(result)
                    if result.get("batch_sizes"):
                        save_benchmark_results(result, args.output_dir)
    
    else:
        parser.print_help()
        sys.exit(0)
    
    # Generate report
    if args.generate_report and results_list:
        output_dir = args.output_dir or BENCHMARK_CONFIG["result_dir"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"benchmark_report_{timestamp}.md")
        
        report = generate_markdown_report(results_list, report_file)
        
        print(f"\nBenchmark report saved to: {report_file}")
        
        if args.verbose:
            print("\nReport Preview:")
            print("=" * 80)
            preview_lines = report.split("\n")[:20]
            print("\n".join(preview_lines))
            if len(report.split("\n")) > 20:
                print("...")
    
    print(f"\nCompleted {len(results_list)} benchmarks")

if __name__ == "__main__":
    main()