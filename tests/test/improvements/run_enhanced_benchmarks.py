#!/usr/bin/env python3
"""
Run Enhanced Benchmarks (March 2025)

This script runs benchmarks using the enhanced test files with full
cross-platform hardware support. It runs benchmarks for key models
across all hardware platforms and stores results in the benchmark
database.

Features:
- Runs benchmarks for key models with enhanced hardware support
- Tests against all hardware platforms (CPU, CUDA, OpenVINO, MPS, ROCm, WebNN, WebGPU)
- Stores results directly in the benchmark database
- Generates comprehensive compatibility matrix based on results
- Validates that all tests pass on their respective platforms
"""

import os
import sys
import time
import json
import datetime
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("enhanced_benchmarks")

# Import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.dirname(current_dir)
sys.path.append(test_dir)

# Constants
PROJECT_ROOT = Path(os.path.dirname(test_dir))
SKILLS_DIR = PROJECT_ROOT / "test" / "skills"
BENCHMARK_RESULTS_DIR = PROJECT_ROOT / "test" / "benchmark_results"
COMPATIBILITY_MATRIX_PATH = PROJECT_ROOT / "test" / "hardware_compatibility_matrix.json"

# Ensure database path is set
os.environ["BENCHMARK_DB_PATH"] = str(PROJECT_ROOT / "test" / "benchmark_db.duckdb")
# Disable JSON output in favor of direct database storage
os.environ["DEPRECATE_JSON_OUTPUT"] = "1"

# Key model types to benchmark
KEY_MODELS = [
    "bert", "t5", "llama", "vit", "clip", "clap", "whisper", 
    "wav2vec2", "llava", "xclip", "qwen2", "detr"
]

# Hardware platforms to test
HARDWARE_PLATFORMS = ["cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu"]

def setup_directories():
    """Create necessary directories."""
    BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created benchmark results directory: {BENCHMARK_RESULTS_DIR}")

def detect_available_hardware() -> Dict[str, bool]:
    """Detect which hardware platforms are available on this system."""
    available_hardware = {
        "cpu": True  # CPU is always available
    }
    
    # Try to import required libraries to detect hardware
    try:
        # Check CUDA availability
        try:
            import torch
            available_hardware["cuda"] = torch.cuda.is_available()
            
            # Check ROCm (AMD GPU) via special torch build
            rocm_available = torch.cuda.is_available() and hasattr(torch.version, "hip")
            available_hardware["rocm"] = rocm_available
            
            # Check MPS (Apple Silicon)
            mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            available_hardware["mps"] = mps_available
        except ImportError:
            available_hardware["cuda"] = False
            available_hardware["rocm"] = False
            available_hardware["mps"] = False
    except:
        available_hardware["cuda"] = False
        available_hardware["rocm"] = False
        available_hardware["mps"] = False
    
    # Check OpenVINO
    try:
        import openvino
        available_hardware["openvino"] = True
    except ImportError:
        available_hardware["openvino"] = False
    
    # Check WebNN and WebGPU (simulation for local environment)
    try:
        # Try to import the fixed_web_platform module
        import fixed_web_platform
        available_hardware["webnn"] = True
        available_hardware["webgpu"] = True
    except ImportError:
        available_hardware["webnn"] = False
        available_hardware["webgpu"] = False
    
    return available_hardware

def run_benchmark(model_name: str, hardware: str) -> Dict:
    """Run a benchmark for a specific model on a specific hardware platform."""
    # Convert model name to normalized form
    normalized_name = model_name.replace("-", "_").replace(".", "_").lower()
    
    # Find the test file
    test_file = SKILLS_DIR / f"test_hf_{normalized_name}.py"
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return {"success": False, "error": f"Test file not found: {test_file}"}
    
    # Build the benchmark command
    # We're using run_model_benchmarks.py which integrates with the database
    benchmark_script = PROJECT_ROOT / "test" / "run_model_benchmarks.py"
    if not benchmark_script.exists():
        logger.error(f"Benchmark script not found: {benchmark_script}")
        return {"success": False, "error": f"Benchmark script not found: {benchmark_script}"}
    
    # Make sure the test file is executable
    os.chmod(test_file, 0o755)
    
    # Build the command
    cmd = [
        sys.executable,
        str(benchmark_script),
        "--models", model_name,
        "--hardware", hardware,
        "--output-dir", str(BENCHMARK_RESULTS_DIR),
        "--small-models",  # Use small model variants for quicker testing
        "--db-path", os.environ["BENCHMARK_DB_PATH"]
    ]
    
    # Run the benchmark
    try:
        logger.info(f"Running benchmark for {model_name} on {hardware}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully benchmarked {model_name} on {hardware}")
            
            # Parse the output for benchmark results
            # The results should already be in the database
            return {
                "success": True,
                "model": model_name,
                "hardware": hardware,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            logger.error(f"Failed to benchmark {model_name} on {hardware}")
            logger.error(f"Stdout: {result.stdout}")
            logger.error(f"Stderr: {result.stderr}")
            return {
                "success": False,
                "model": model_name,
                "hardware": hardware,
                "error": "Benchmark process failed",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
    except Exception as e:
        logger.error(f"Error benchmarking {model_name} on {hardware}: {e}")
        return {
            "success": False,
            "model": model_name,
            "hardware": hardware,
            "error": str(e)
        }

def run_all_benchmarks(available_hardware: Dict[str, bool]) -> Dict[str, Dict[str, Dict]]:
    """Run benchmarks for all key models on all available hardware platforms."""
    results = {}
    
    for model in KEY_MODELS:
        results[model] = {}
        
        for hardware in HARDWARE_PLATFORMS:
            # Skip hardware that's not available
            if not available_hardware.get(hardware, False):
                logger.warning(f"Skipping {hardware} for {model} as it's not available")
                results[model][hardware] = {
                    "success": False,
                    "model": model,
                    "hardware": hardware,
                    "error": "Hardware not available"
                }
                continue
            
            # Run the benchmark
            result = run_benchmark(model, hardware)
            results[model][hardware] = result
    
    return results

def generate_compatibility_matrix(benchmark_results: Dict[str, Dict[str, Dict]]) -> Dict:
    """Generate a compatibility matrix based on benchmark results."""
    compatibility_matrix = {
        "models": {},
        "hardware": {hw: {"available": True} for hw in HARDWARE_PLATFORMS},
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Update hardware availability
    for hardware in HARDWARE_PLATFORMS:
        # Check if any benchmark was run for this hardware
        any_benchmark_run = any(
            benchmark_results.get(model, {}).get(hardware, {}).get("success", False)
            for model in KEY_MODELS
        )
        compatibility_matrix["hardware"][hardware]["available"] = any_benchmark_run
    
    # Build model compatibility info
    for model in KEY_MODELS:
        compatibility_matrix["models"][model] = {
            "hardware_compatibility": {}
        }
        
        for hardware in HARDWARE_PLATFORMS:
            result = benchmark_results.get(model, {}).get(hardware, {})
            compatibility_matrix["models"][model]["hardware_compatibility"][hardware] = {
                "supported": result.get("success", False),
                "error": result.get("error", None) if not result.get("success", False) else None
            }
    
    return compatibility_matrix

def save_compatibility_matrix(matrix: Dict):
    """Save the compatibility matrix to a file."""
    with open(COMPATIBILITY_MATRIX_PATH, "w") as f:
        json.dump(matrix, f, indent=2)
    logger.info(f"Saved compatibility matrix to {COMPATIBILITY_MATRIX_PATH}")

def main():
    """Main function."""
    logger.info("Starting enhanced benchmarks for all models and hardware platforms")
    
    # Set up directories
    setup_directories()
    
    # Detect available hardware
    available_hardware = detect_available_hardware()
    logger.info(f"Detected hardware: {available_hardware}")
    
    # Run benchmarks
    benchmark_results = run_all_benchmarks(available_hardware)
    
    # Generate compatibility matrix
    compatibility_matrix = generate_compatibility_matrix(benchmark_results)
    
    # Save compatibility matrix
    save_compatibility_matrix(compatibility_matrix)
    
    # Print summary
    logger.info("\nBenchmark Summary:")
    total_success = 0
    total_benchmarks = 0
    
    for model in KEY_MODELS:
        model_success = 0
        model_total = 0
        
        for hardware in HARDWARE_PLATFORMS:
            if available_hardware.get(hardware, False):
                result = benchmark_results.get(model, {}).get(hardware, {})
                success = result.get("success", False)
                
                if success:
                    model_success += 1
                    total_success += 1
                
                model_total += 1
                total_benchmarks += 1
        
        logger.info(f"{model}: {model_success}/{model_total} benchmarks passed")
    
    logger.info(f"Overall: {total_success}/{total_benchmarks} benchmarks passed")
    logger.info(f"Compatibility matrix saved to: {COMPATIBILITY_MATRIX_PATH}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())