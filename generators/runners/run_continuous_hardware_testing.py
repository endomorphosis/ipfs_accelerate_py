#!/usr/bin/env python
"""
Continuous Hardware Benchmarking and Optimization System

This script provides comprehensive testing, benchmarking, and optimization for models
across different hardware platforms. It integrates with CI systems and provides detailed
performance reports and compatibility matrices.

Key features:
- Hardware benchmarking across different model families and devices
- Model compression and optimization based on hardware characteristics
- Continuous monitoring and trend analysis
- Integration with CI systems for automated testing
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
import platform
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if running in CI
IN_CI = os.environ.get("CI", "false").lower() == "true"
CI_SYSTEM = os.environ.get("CI_SYSTEM", "unknown").lower()
CI_PLATFORM = ""

# Detect CI platform
if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
    CI_PLATFORM = "github"
elif os.environ.get("GITLAB_CI", "").lower() == "true":
    CI_PLATFORM = "gitlab" 
elif os.environ.get("JENKINS_URL", ""):
    CI_PLATFORM = "jenkins"
elif os.environ.get("TRAVIS", "").lower() == "true":
    CI_PLATFORM = "travis"
elif os.environ.get("CIRCLECI", "").lower() == "true":
    CI_PLATFORM = "circle"
else:
    CI_PLATFORM = "local"

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "hardware_benchmark_results")
OPTIMIZATION_DIR = os.path.join(SCRIPT_DIR, "optimized_models")
CACHE_DIR = os.path.join(SCRIPT_DIR, "hardware_cache")
HISTORY_FILE = os.path.join(RESULTS_DIR, "hardware_test_history.json")
BENCHMARK_DATABASE = os.path.join(RESULTS_DIR, "benchmark_database.json")

# Model families and representative models for testing
MODEL_FAMILIES = {
    "embedding": ["prajjwal1/bert-tiny"],
    "text_generation": ["gpt2"],
    "vision": ["google/vit-base-patch16-224"],
    "audio": ["openai/whisper-tiny"],
    "multimodal": ["openai/clip-vit-base-patch32"]
}

# Default hardware platforms to test (if available)
DEFAULT_PLATFORMS = ["cpu", "cuda", "mps", "rocm", "openvino", "webnn", "webgpu"]

# Try to import advanced components
try:
    from continuous_hardware_benchmarking import HardwareBenchmark
    from model_compression_optimizer import ModelCompressor
    from generators.utils.resource_pool import get_global_resource_pool
    HAS_ADVANCED_COMPONENTS = True
    logger.info("Advanced benchmarking and optimization components available")
except ImportError as e:
    logger.warning(f"Advanced components not available: {e}")
    logger.warning("Falling back to basic compatibility testing only")
    HAS_ADVANCED_COMPONENTS = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run hardware benchmarking, optimization, and compatibility testing")
    
    # Basic options
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR,
                      help="Output directory for test results")
    parser.add_argument("--optimization-dir", type=str, default=OPTIMIZATION_DIR,
                      help="Directory for optimized models")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR,
                      help="Directory for hardware detection cache")
    parser.add_argument("--history-file", type=str, default=HISTORY_FILE,
                      help="Path to test history JSON file")
    parser.add_argument("--benchmark-db", type=str, default=BENCHMARK_DATABASE,
                      help="Path to benchmark database JSON file")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["compatibility", "benchmark", "optimize", "all", "schedule", "trend"],
                      default="compatibility", help="Operation mode")
    
    # Model and platform selection
    parser.add_argument("--models", type=str, 
                      help="Comma-separated list of models to test")
    parser.add_argument("--platforms", type=str,
                      help="Comma-separated list of platforms to test (default: detect automatically)")
    parser.add_argument("--families", type=str,
                      help="Comma-separated list of model families to test")
    parser.add_argument("--all-families", action="store_true",
                      help="Test all model families (default: use representative models)")
    
    # Benchmark options
    parser.add_argument("--iterations", type=int, default=5,
                      help="Number of benchmark iterations")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                      help="Comma-separated list of batch sizes to test")
    
    # Optimization options
    parser.add_argument("--optimizations", type=str,
                      help="Comma-separated list of optimization techniques to apply")
    parser.add_argument("--target-device", type=str, default="cpu",
                      help="Target device for optimization")
    
    # Scheduling options
    parser.add_argument("--interval", type=float, default=24.0,
                      help="Hours between scheduled runs")
    parser.add_argument("--max-runs", type=int,
                      help="Maximum number of scheduled runs")
    
    # CI and reporting options
    parser.add_argument("--skip-report", action="store_true",
                      help="Skip generating detailed report")
    parser.add_argument("--ci-mode", action="store_true",
                      help="Run in CI mode with appropriate output format")
    parser.add_argument("--verify-install", action="store_true",
                      help="Verify installation of required packages before testing")
    parser.add_argument("--compare-to-baseline", action="store_true",
                      help="Compare results to baseline/previous results")
    parser.add_argument("--fail-on-regression", action="store_true",
                      help="Exit with error if results show regression from baseline")
    parser.add_argument("--max-workers", type=int, default=4,
                      help="Maximum number of workers for parallel testing")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    return parser.parse_args()

def ensure_directories(args):
    """Ensure all necessary directories exist"""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.optimization_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(args.output_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "benchmarks"), exist_ok=True)
    os.makedirs(os.path.join(args.optimization_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.optimization_dir, "reports"), exist_ok=True)
    
    # Create hardware cache path
    hw_cache_path = os.path.join(args.cache_dir, "hardware_detection_cache.json")
    
    return hw_cache_path

def verify_installation():
    """Verify that required packages are installed"""
    required_packages = [
        "torch",
        "transformers",
        "numpy",
        "psutil",
        "tqdm"
    ]
    
    # Add advanced packages if using advanced features
    if HAS_ADVANCED_COMPONENTS:
        required_packages.extend([
            "onnx",
            "onnxruntime",
        ])
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install the missing packages and try again")
        logger.error("You can install them with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("All required packages are installed")
    return True

def detect_system_info():
    """Detect system information to include in test results"""
    system_info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "ci_environment": IN_CI,
        "ci_platform": CI_PLATFORM,
        "date": datetime.now().isoformat()
    }
    
    # Try to get additional system information
    try:
        import psutil
        memory = psutil.virtual_memory()
        system_info["total_memory_gb"] = memory.total / (1024**3)
        system_info["available_memory_gb"] = memory.available / (1024**3)
        
        # CPU info
        system_info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        system_info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    
    # Try to get GPU information
    try:
        import torch
        if torch.cuda.is_available():
            system_info["cuda_available"] = True
            system_info["cuda_version"] = torch.version.cuda
            system_info["cuda_device_count"] = torch.cuda.device_count()
            system_info["cuda_devices"] = []
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                system_info["cuda_devices"].append({
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": device_props.total_memory / (1024**3),
                    "compute_capability": f"{device_props.major}.{device_props.minor}"
                })
        else:
            system_info["cuda_available"] = False
            
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            system_info["mps_available"] = True
            system_info["mps_built"] = torch.backends.mps.is_built()
        else:
            system_info["mps_available"] = False
    except ImportError:
        system_info["cuda_available"] = "unknown"
        system_info["mps_available"] = "unknown"
    
    return system_info

def run_hardware_compatibility_test(args, hw_cache_path):
    """Run the hardware compatibility test and return the results path"""
    # Construct command
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "test_automated_hardware_compatibility.py"),
        "--output-dir", args.output_dir,
        "--hw-cache", hw_cache_path,
        "--max-workers", str(args.max_workers)
    ]
    
    # Add optional arguments
    if args.models:
        cmd.extend(["--models", args.models])
    
    if args.platforms:
        cmd.extend(["--platforms", args.platforms])
    
    if args.debug:
        cmd.append("--debug")
    
    # Run the command
    logger.info(f"Running hardware compatibility test: {' '.join(cmd)}")
    
    try:
        start_time = datetime.now()
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = datetime.now()
        
        # Log output
        logger.debug(process.stdout)
        
        # Find the most recent results file
        results_files = list(Path(args.output_dir).glob("hardware_compatibility_results_*.json"))
        if not results_files:
            logger.error("No results file found")
            return None
        
        # Sort by modification time
        results_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_results_file = str(results_files[0])
        
        # Find corresponding report
        report_file = latest_results_file.replace("_results_", "_report_").replace(".json", ".md")
        
        logger.info(f"Test completed in {end_time - start_time}")
        logger.info(f"Results saved to {latest_results_file}")
        if os.path.exists(report_file):
            logger.info(f"Report saved to {report_file}")
        
        return latest_results_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running hardware compatibility test: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running hardware compatibility test: {e}")
        return None

def load_results(results_path):
    """Load results from a JSON file"""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return None

def update_test_history(history_file, results, system_info):
    """Update the test history with the latest results"""
    history = {}
    
    # Load existing history if available
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading history file: {e}")
            history = {}
    
    # Initialize history structure if needed
    if "tests" not in history:
        history["tests"] = []
    
    # Create entry for this test
    test_entry = {
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info,
        "compatibility_matrix": results.get("compatibility_matrix", {}),
        "model_family_compatibility": results.get("model_family_compatibility", {}),
        "hardware_platform_capabilities": results.get("hardware_platform_capabilities", {})
    }
    
    # Add to history
    history["tests"].append(test_entry)
    
    # Keep only the last 10 entries
    if len(history["tests"]) > 10:
        history["tests"] = history["tests"][-10:]
    
    # Save updated history
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Updated test history at {history_file}")
    except Exception as e:
        logger.error(f"Error saving history file: {e}")

def compare_to_baseline(current_results, history_file):
    """
    Compare current results to baseline/previous results.
    
    Returns:
        tuple: (has_regression, comparison_report)
    """
    if not os.path.exists(history_file):
        logger.warning("No history file found for comparison")
        return False, "No previous test results available for comparison"
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except Exception as e:
        logger.error(f"Error loading history file: {e}")
        return False, f"Error loading history file: {e}"
    
    if not history.get("tests"):
        logger.warning("No previous tests found in history")
        return False, "No previous test results available for comparison"
    
    # Get previous test (second most recent)
    previous_tests = history.get("tests", [])[:-1]  # Exclude most recent (current test)
    if not previous_tests:
        logger.warning("No previous tests found for comparison")
        return False, "This is the first test run, no baseline available"
    
    previous_test = previous_tests[-1]  # Most recent of the previous tests
    
    # Compare compatibility matrices
    current_matrix = current_results.get("compatibility_matrix", {})
    previous_matrix = previous_test.get("compatibility_matrix", {})
    
    # Initialize comparisons
    regressions = []
    improvements = []
    unchanged = []
    
    # Compare families and platforms
    for family, platforms in current_matrix.items():
        if family not in previous_matrix:
            # New family, consider as improvement
            improvements.append(f"New model family: {family}")
            continue
        
        for platform, status in platforms.items():
            if platform not in previous_matrix[family]:
                # New platform, consider as improvement
                improvements.append(f"New platform {platform} for {family}: {status}")
                continue
            
            previous_status = previous_matrix[family][platform]
            
            # Check for changes
            if status != previous_status:
                if (status == "incompatible" and previous_status == "compatible") or \
                   (status == "device_mismatch" and previous_status == "compatible"):
                    regressions.append(f"{family} on {platform}: {previous_status} → {status}")
                elif (status == "compatible" and previous_status != "compatible"):
                    improvements.append(f"{family} on {platform}: {previous_status} → {status}")
                else:
                    # Other changes (might be improvements or neutral changes)
                    if status == "compatible" or status == "device_mismatch":
                        improvements.append(f"{family} on {platform}: {previous_status} → {status}")
                    else:
                        unchanged.append(f"{family} on {platform}: {previous_status} → {status} (neutral change)")
            else:
                unchanged.append(f"{family} on {platform}: {status} (unchanged)")
    
    # Create comparison report
    has_regression = len(regressions) > 0
    
    # Format the report
    report = f"## Hardware Compatibility Comparison\n\n"
    report += f"Comparing results from {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
    report += f"to previous test from {previous_test.get('timestamp', 'unknown')}\n\n"
    
    if regressions:
        report += "### 🔴 Regressions\n\n"
        for regression in regressions:
            report += f"- {regression}\n"
        report += "\n"
    
    if improvements:
        report += "### 🟢 Improvements\n\n"
        for improvement in improvements:
            report += f"- {improvement}\n"
        report += "\n"
    
    if unchanged:
        report += "### ⚪ Unchanged\n\n"
        report += f"- {len(unchanged)} compatibility checks remained the same\n\n"
    
    # Summary
    report += "### Summary\n\n"
    report += f"- Regressions: {len(regressions)}\n"
    report += f"- Improvements: {len(improvements)}\n"
    report += f"- Unchanged: {len(unchanged)}\n"
    
    return has_regression, report

def generate_ci_artifacts(results_path, args):
    """
    Generate CI artifacts based on the CI platform.
    Returns the paths to the artifacts.
    """
    results = load_results(results_path)
    if not results:
        return []
    
    # Generate CI-specific artifacts
    artifacts = []
    
    if CI_PLATFORM == "github":
        # GitHub Actions
        try:
            # Create a summary markdown file for GitHub Actions
            summary_path = os.path.join(args.output_dir, "github_summary.md")
            
            # Find the report path
            report_path = results_path.replace("_results_", "_report_").replace(".json", ".md")
            
            if os.path.exists(report_path):
                # Copy report to summary
                shutil.copy(report_path, summary_path)
                artifacts.append(summary_path)
                
                # Set output for GitHub Actions
                with open(os.environ.get("GITHUB_OUTPUT", "/dev/null"), "a") as f:
                    f.write(f"hardware_compatibility_report={summary_path}\n")
                    f.write(f"hardware_compatibility_results={results_path}\n")
            
            # Create a compatibility badge
            platform_capabilities = results.get("hardware_platform_capabilities", {})
            if platform_capabilities:
                # Calculate overall score
                total_success = sum(p.get("success_count", 0) for p in platform_capabilities.values())
                total_tests = sum(p.get("total_count", 0) for p in platform_capabilities.values())
                
                if total_tests > 0:
                    score = total_success / total_tests
                    # Badge color based on score
                    color = "brightgreen" if score >= 0.8 else "yellow" if score >= 0.5 else "red"
                    
                    # Badge URL
                    badge_url = f"https://img.shields.io/badge/Hardware%20Compatibility-{int(score*100)}%25-{color}"
                    
                    # Set badge output
                    with open(os.environ.get("GITHUB_OUTPUT", "/dev/null"), "a") as f:
                        f.write(f"hardware_compatibility_badge={badge_url}\n")
                        f.write(f"hardware_compatibility_score={int(score*100)}\n")
        except Exception as e:
            logger.error(f"Error generating GitHub artifacts: {e}")
    
    return artifacts

def run_benchmarks(args):
    """Run benchmarks using the HardwareBenchmark class"""
    if not HAS_ADVANCED_COMPONENTS:
        logger.error("Advanced benchmarking components not available")
        return None
    
    # Parse batch sizes
    batch_sizes = [int(s) for s in args.batch_sizes.split(',')]
    
    # Parse model families if provided
    model_families = args.families.split(',') if args.families else None
    
    # Create benchmark instance
    benchmark = HardwareBenchmark(
        output_dir=args.output_dir,
        database_path=args.benchmark_db
    )
    
    # Determine device parameter
    device = args.platforms if args.platforms else "all"
    
    # Run benchmarks
    logger.info(f"Running benchmarks with device={device}, families={model_families}, iterations={args.iterations}")
    
    results = benchmark.run_comprehensive_benchmark(
        device=device,
        families=model_families,
        iterations=args.iterations,
        batch_sizes=batch_sizes
    )
    
    # Generate report
    report_path = benchmark.generate_report(results)
    logger.info(f"Benchmark report generated: {report_path}")
    
    # Generate compatibility matrix
    matrix_path = benchmark.generate_compatibility_report()
    logger.info(f"Compatibility matrix report generated: {matrix_path}")
    
    return results

def run_optimizations(args, benchmark_results=None):
    """Run model optimizations using the ModelCompressor class"""
    if not HAS_ADVANCED_COMPONENTS:
        logger.error("Advanced optimization components not available")
        return None
    
    # Parse models if provided
    models = args.models.split(',') if args.models else None
    
    # Parse optimizations if provided
    optimizations = args.optimizations.split(',') if args.optimizations else None
    
    # Create compressor instance
    compressor = ModelCompressor(
        output_dir=args.optimization_dir
    )
    
    # If specific models provided, optimize them directly
    if models:
        all_results = {}
        
        for model in models:
            logger.info(f"Optimizing model: {model}")
            
            # Run optimization pipeline
            results = compressor.run_optimization_pipeline(
                model_name_or_path=model,
                optimization_types=optimizations,
                target_device=args.target_device
            )
            
            # Generate report
            report_path = compressor.generate_report(results)
            logger.info(f"Optimization report generated: {report_path}")
            
            # Store results
            all_results[model] = results
        
        return all_results
    
    # If no specific models, use benchmark results to select models
    # This will use a more sophisticated approach
    elif HAS_ADVANCED_COMPONENTS and benchmark_results:
        # Create hardware benchmark instance to access database
        benchmark = HardwareBenchmark(
            output_dir=args.output_dir,
            database_path=args.benchmark_db
        )
        
        # Get benchmarked models from compatibility matrix
        compatibility_matrix = benchmark.get_compatibility_matrix()
        
        # Select models from each compatible family
        selected_models = []
        
        for family, platforms in compatibility_matrix.items():
            for platform, data in platforms.items():
                if data.get("compatible", False) and platform == args.target_device:
                    # Get the best model from this family
                    models_tested = data.get("models_tested", [])
                    if models_tested:
                        # Add first model from this family that's not already in our list
                        for model in models_tested:
                            if model not in selected_models:
                                selected_models.append(model)
                                break
        
        # If we didn't find any models, use defaults
        if not selected_models:
            selected_models = ["prajjwal1/bert-tiny", "gpt2", "google/vit-base-patch16-224"]
        
        # Limit to a reasonable number
        selected_models = selected_models[:3]
        
        # Optimize selected models
        all_results = {}
        for model in selected_models:
            logger.info(f"Optimizing model: {model}")
            
            # Run optimization pipeline
            results = compressor.run_optimization_pipeline(
                model_name_or_path=model,
                optimization_types=optimizations,
                target_device=args.target_device
            )
            
            # Generate report
            report_path = compressor.generate_report(results)
            logger.info(f"Optimization report generated: {report_path}")
            
            # Store results
            all_results[model] = results
        
        return all_results
    
    # Fallback to default models
    else:
        # Select representative models
        default_models = [
            "prajjwal1/bert-tiny",  # Embedding
            "gpt2",                 # Text generation
            "google/vit-base-patch16-224"  # Vision
        ]
        
        all_results = {}
        for model in default_models:
            logger.info(f"Optimizing model: {model}")
            
            # Run optimization pipeline
            results = compressor.run_optimization_pipeline(
                model_name_or_path=model,
                optimization_types=optimizations,
                target_device=args.target_device
            )
            
            # Generate report
            report_path = compressor.generate_report(results)
            logger.info(f"Optimization report generated: {report_path}")
            
            # Store results
            all_results[model] = results
        
        return all_results

def run_scheduled_benchmarks(args):
    """Run benchmarks and optimizations on a schedule"""
    if not HAS_ADVANCED_COMPONENTS:
        logger.error("Advanced components not available for scheduled benchmarks")
        return False
    
    # Parse interval
    interval_hours = args.interval
    
    # Parse max runs
    max_runs = args.max_runs
    
    logger.info(f"Starting scheduled benchmarking every {interval_hours} hours")
    if max_runs:
        logger.info(f"Will run a maximum of {max_runs} times")
    
    run_count = 0
    
    while max_runs is None or run_count < max_runs:
        # Log run information
        run_count += 1
        logger.info(f"Starting scheduled benchmark run #{run_count}")
        start_time = time.time()
        
        try:
            # Run benchmarks
            benchmark_results = run_benchmarks(args)
            
            # Run optimizations
            optimization_results = run_optimizations(args, benchmark_results)
            
            # Generate trends if we have multiple runs
            if run_count > 1 and HAS_ADVANCED_COMPONENTS:
                try:
                    # Create benchmark instance
                    benchmark = HardwareBenchmark(
                        output_dir=args.output_dir,
                        database_path=args.benchmark_db
                    )
                    
                    # Get performance trends
                    trends = benchmark.get_performance_trends()
                    
                    # Create a report
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    trend_filename = f"performance_trends_{timestamp}.md"
                    trend_path = os.path.join(args.output_dir, "reports", trend_filename)
                    
                    # Format report content
                    report_content = [
                        "# Hardware Performance Trends",
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        "",
                        "## Model-specific Trends",
                        "",
                        "| Model | Device | Latency Change | Throughput Change |",
                        "| --- | --- | --- | --- |"
                    ]
                    
                    # Add model-device trends
                    for key, data in trends.get("by_model_device", {}).items():
                        model, device = key.split(":")
                        latency_change = data.get("latency_change_percent", 0)
                        throughput_change = data.get("throughput_change_percent", 0)
                        
                        # Determine icons based on changes
                        latency_icon = "🟢" if latency_change < 0 else "🔴" if latency_change > 0 else "⚪"
                        throughput_icon = "🟢" if throughput_change > 0 else "🔴" if throughput_change < 0 else "⚪"
                        
                        row = f"| {model} | {device} | {latency_change:.2f}% {latency_icon} | {throughput_change:.2f}% {throughput_icon} |"
                        report_content.append(row)
                    
                    # Write report
                    with open(trend_path, 'w') as f:
                        f.write("\n".join(report_content))
                    
                    logger.info(f"Trend report generated: {trend_path}")
                    
                except Exception as e:
                    logger.error(f"Error generating trend report: {e}")
            
            # Record end time
            run_time = time.time() - start_time
            logger.info(f"Benchmark run #{run_count} completed in {run_time/60:.1f} minutes")
            
        except Exception as e:
            logger.error(f"Error during scheduled benchmark run #{run_count}: {e}")
        
        # Stop if we reached max runs
        if max_runs is not None and run_count >= max_runs:
            logger.info(f"Reached maximum number of runs ({max_runs}). Stopping.")
            break
        
        # Sleep until next run
        next_run_time = start_time + (interval_hours * 3600)
        sleep_seconds = max(0, next_run_time - time.time())
        
        if sleep_seconds > 0:
            logger.info(f"Sleeping for {sleep_seconds/3600:.1f} hours until next benchmark run")
            time.sleep(sleep_seconds)
    
    return True

def generate_trend_report(args):
    """Generate trend report from historical benchmark data"""
    if not HAS_ADVANCED_COMPONENTS:
        logger.error("Advanced components not available for trend analysis")
        return None
    
    try:
        # Create benchmark instance
        benchmark = HardwareBenchmark(
            output_dir=args.output_dir,
            database_path=args.benchmark_db
        )
        
        # Get performance trends
        trends = benchmark.get_performance_trends()
        
        if not trends or not trends.get("by_model_device"):
            logger.warning("No trend data available - need more benchmark runs")
            return None
        
        # Create a report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trend_filename = f"performance_trends_{timestamp}.md"
        trend_path = os.path.join(args.output_dir, "reports", trend_filename)
        
        # Format report content
        report_content = [
            "# Hardware Performance Trends",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model-specific Trends",
            "",
            "| Model | Device | Latency Change | Throughput Change |",
            "| --- | --- | --- | --- |"
        ]
        
        # Add model-device trends
        for key, data in trends.get("by_model_device", {}).items():
            model, device = key.split(":")
            latency_change = data.get("latency_change_percent", 0)
            throughput_change = data.get("throughput_change_percent", 0)
            
            # Determine icons based on changes
            latency_icon = "🟢" if latency_change < 0 else "🔴" if latency_change > 0 else "⚪"
            throughput_icon = "🟢" if throughput_change > 0 else "🔴" if throughput_change < 0 else "⚪"
            
            row = f"| {model} | {device} | {latency_change:.2f}% {latency_icon} | {throughput_change:.2f}% {throughput_icon} |"
            report_content.append(row)
        
        # Add family trends
        report_content.extend([
            "",
            "## Model Family Trends",
            "",
            "| Family | Device | Latency Change | Throughput Change |",
            "| --- | --- | --- | --- |"
        ])
        
        # Add family-device trends
        for key, data in trends.get("by_family", {}).items():
            family, device = key.split(":")
            latency_change = data.get("latency_change_percent", 0)
            throughput_change = data.get("throughput_change_percent", 0)
            
            # Determine icons based on changes
            latency_icon = "🟢" if latency_change < 0 else "🔴" if latency_change > 0 else "⚪"
            throughput_icon = "🟢" if throughput_change > 0 else "🔴" if throughput_change < 0 else "⚪"
            
            row = f"| {family} | {device} | {latency_change:.2f}% {latency_icon} | {throughput_change:.2f}% {throughput_icon} |"
            report_content.append(row)
        
        # Add interpretation guide
        report_content.extend([
            "",
            "## Interpretation",
            "",
            "- 🟢 in latency change: Improvement (lower latency)",
            "- 🔴 in latency change: Regression (higher latency)",
            "- 🟢 in throughput change: Improvement (higher throughput)",
            "- 🔴 in throughput change: Regression (lower throughput)",
            "- ⚪: No significant change",
            "",
            "Note: Changes are relative to the first recorded benchmark for each model-device combination."
        ])
        
        # Write report
        with open(trend_path, 'w') as f:
            f.write("\n".join(report_content))
        
        logger.info(f"Trend report generated: {trend_path}")
        return trend_path
        
    except Exception as e:
        logger.error(f"Error generating trend report: {e}")
        return None

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Set CI mode if detected
    if IN_CI:
        args.ci_mode = True
        logger.info(f"Running in CI environment: {CI_PLATFORM}")
    
    # Ensure directories exist
    hw_cache_path = ensure_directories(args)
    
    # Verify installation if requested
    if args.verify_install:
        if not verify_installation():
            return 1
    
    # Detect system information
    system_info = detect_system_info()
    logger.info(f"System: {system_info['platform']} {system_info['architecture']}")
    if system_info.get("cuda_available"):
        logger.info(f"CUDA: {system_info.get('cuda_version', 'unknown')}, "
                   f"{system_info.get('cuda_device_count', 0)} device(s)")
    
    # Run in selected mode
    if args.mode == "compatibility" or args.mode == "all":
        # Run hardware compatibility test
        results_path = run_hardware_compatibility_test(args, hw_cache_path)
        
        if not results_path:
            logger.error("Hardware compatibility test failed")
            return 1
        
        # Load results
        results = load_results(results_path)
        if not results:
            logger.error("Failed to load test results")
            return 1
        
        # Update test history
        update_test_history(args.history_file, results, system_info)
        
        # Compare to baseline if requested
        if args.compare_to_baseline:
            has_regression, comparison_report = compare_to_baseline(results, args.history_file)
            
            # Save comparison report
            comparison_path = os.path.join(args.output_dir, "hardware_compatibility_comparison.md")
            with open(comparison_path, 'w') as f:
                f.write(comparison_report)
            
            logger.info(f"Comparison report saved to {comparison_path}")
            
            if has_regression and args.fail_on_regression:
                logger.error("Regressions detected, failing the build")
                return 1
        
        # Generate CI-specific artifacts
        if args.ci_mode:
            generate_ci_artifacts(results_path, args)
        
        # Output summary
        print("\nHardware Compatibility Test Summary:\n")
        
        compatibility_matrix = results.get("compatibility_matrix", {})
        if compatibility_matrix:
            for family, platforms in compatibility_matrix.items():
                print(f"Family: {family}")
                for platform, status in platforms.items():
                    status_indicator = "✅" if status == "compatible" else "⚠️" if status == "device_mismatch" else "❌"
                    print(f"  {platform}: {status_indicator} {status}")
                print()
    
    # Run benchmarks if requested
    benchmark_results = None
    if args.mode == "benchmark" or args.mode == "all":
        if HAS_ADVANCED_COMPONENTS:
            logger.info("Running benchmarks")
            benchmark_results = run_benchmarks(args)
            if benchmark_results:
                logger.info("Benchmarks completed successfully")
            else:
                logger.error("Benchmark run failed")
        else:
            logger.error("Advanced benchmarking components not available")
            
    # Run optimizations if requested
    if args.mode == "optimize" or args.mode == "all":
        if HAS_ADVANCED_COMPONENTS:
            logger.info("Running optimizations")
            optimization_results = run_optimizations(args, benchmark_results)
            if optimization_results:
                logger.info("Optimizations completed successfully")
                
                # Print summary
                print("\nOptimization Summary:\n")
                for model, results in optimization_results.items():
                    print(f"Model: {model}")
                    print(f"- Original size: {results.get('original_size_mb', 0):.2f} MB")
                    print(f"- Optimized size: {results.get('optimized_size_mb', 0):.2f} MB")
                    print(f"- Size reduction: {results.get('size_reduction_percent', 0):.2f}%")
                    if "optimized_model_path" in results:
                        print(f"- Optimized model path: {results['optimized_model_path']}")
                    print()
            else:
                logger.error("Optimization run failed")
        else:
            logger.error("Advanced optimization components not available")
    
    # Run scheduled benchmarks if requested
    if args.mode == "schedule":
        if HAS_ADVANCED_COMPONENTS:
            logger.info("Starting scheduled benchmarking")
            run_scheduled_benchmarks(args)
        else:
            logger.error("Advanced components not available for scheduled benchmarks")
            return 1
    
    # Generate trend report if requested
    if args.mode == "trend":
        if HAS_ADVANCED_COMPONENTS:
            logger.info("Generating trend report")
            trend_path = generate_trend_report(args)
            if trend_path:
                print(f"Trend report generated: {trend_path}")
            else:
                logger.error("Trend report generation failed - need more benchmark data")
                return 1
        else:
            logger.error("Advanced components not available for trend analysis")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())