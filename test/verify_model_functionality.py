#!/usr/bin/env python3
"""
Script to verify the functionality of key model test implementations.
This checks if models actually work across various hardware platforms.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the 13 key models
KEY_MODELS = [
    "bert",
    "clap",
    "clip",
    "detr",
    "llama",
    "llava",
    "llava_next",
    "qwen2",
    "t5",
    "vit",
    "wav2vec2",
    "whisper",
    "xclip"
]

# Define hardware platforms to test
HARDWARE_PLATFORMS = ["cpu"]  # Start with CPU only
# Add other platforms if available
try:
    import torch
    if torch.cuda.is_available():
        HARDWARE_PLATFORMS.append("cuda")
    # Check for Apple MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        HARDWARE_PLATFORMS.append("mps")
except ImportError:
    logger.warning("PyTorch not available, only testing on CPU")

# Add OpenVINO if available
try:
    import openvino
    HARDWARE_PLATFORMS.append("openvino")
except ImportError:
    logger.warning("OpenVINO not available")

# Check for AMD ROCm
if os.environ.get("ROCM_HOME"):
    HARDWARE_PLATFORMS.append("rocm")

def run_model_test(model_name, hardware="cpu", timeout=300):
    """Run a model test on a specific hardware platform."""
    test_file = f"skills/test_hf_{model_name}.py"
    
    if not Path(test_file).exists():
        logger.error(f"Test file not found: {test_file}")
        return {
            "model": model_name,
            "hardware": hardware,
            "success": False,
            "error": "Test file not found",
            "output": None
        }
    
    # Construct command based on hardware
    cmd = [sys.executable, test_file]
    if hardware == "cpu":
        cmd.append("--cpu-only")
        # Set CUDA_VISIBLE_DEVICES to empty to force CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif hardware == "cuda":
        # Use CUDA - don't set anything special
        pass
    elif hardware == "mps":
        # Use MPS (Apple silicon)
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    elif hardware == "openvino":
        # Use OpenVINO - add specific flag if needed
        pass
    elif hardware == "rocm":
        # Use ROCm - add specific flag if needed
        pass
    
    try:
        # Run the test with timeout
        output = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=timeout,
            check=False
        )
        
        # Determine if the test was successful
        success = output.returncode == 0
        
        return {
            "model": model_name,
            "hardware": hardware,
            "success": success,
            "return_code": output.returncode,
            "stdout": output.stdout,
            "stderr": output.stderr
        }
        
    except subprocess.TimeoutExpired:
        return {
            "model": model_name,
            "hardware": hardware,
            "success": False,
            "error": f"Test timed out after {timeout} seconds",
            "output": None
        }
    except Exception as e:
        return {
            "model": model_name,
            "hardware": hardware,
            "success": False,
            "error": str(e),
            "output": None
        }

def verify_all_models(models=None, hardware_platforms=None, max_workers=4, timeout=300):
    """Verify functionality of all specified models on all specified hardware platforms."""
    models = models or KEY_MODELS
    hardware_platforms = hardware_platforms or HARDWARE_PLATFORMS
    
    logger.info(f"Starting verification of {len(models)} models on {len(hardware_platforms)} hardware platforms")
    
    results = []
    
    # Use ThreadPoolExecutor to run tests in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_test = {}
        
        # Submit all test combinations
        for model in models:
            for hardware in hardware_platforms:
                future = executor.submit(run_model_test, model, hardware, timeout)
                future_to_test[future] = (model, hardware)
        
        # Process results as they complete
        for future in as_completed(future_to_test):
            model, hardware = future_to_test[future]
            try:
                result = future.result()
                results.append(result)
                
                status = "✅ PASSED" if result["success"] else "❌ FAILED"
                logger.info(f"{status} - {model} on {hardware}")
                
            except Exception as e:
                logger.error(f"Error processing {model} on {hardware}: {e}")
                results.append({
                    "model": model,
                    "hardware": hardware,
                    "success": False,
                    "error": str(e),
                    "output": None
                })
    
    return results

def generate_report(results, output_dir="./functionality_reports"):
    """Generate a report from the test results."""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = output_dir / f"model_functionality_{timestamp}.json"
    md_file = output_dir / f"model_functionality_report_{timestamp}.md"
    
    # Compile statistics
    stats = {
        "timestamp": timestamp,
        "total_tests": len(results),
        "successful_tests": sum(1 for r in results if r["success"]),
        "failed_tests": sum(1 for r in results if not r["success"]),
        "models_tested": len(set(r["model"] for r in results)),
        "hardware_platforms": list(set(r["hardware"] for r in results)),
        "success_rate": sum(1 for r in results if r["success"]) / len(results) * 100 if results else 0
    }
    
    # Group results by model and hardware
    model_results = {}
    for result in results:
        model = result["model"]
        hardware = result["hardware"]
        
        if model not in model_results:
            model_results[model] = {}
        
        model_results[model][hardware] = result["success"]
    
    # Write JSON report
    with open(json_file, "w") as f:
        json.dump({
            "stats": stats,
            "model_results": model_results,
            "detailed_results": results
        }, f, indent=2)
    
    # Write Markdown report
    with open(md_file, "w") as f:
        f.write("# Model Functionality Verification Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Tests**: {stats['total_tests']}\n")
        f.write(f"- **Successful Tests**: {stats['successful_tests']}\n")
        f.write(f"- **Failed Tests**: {stats['failed_tests']}\n")
        f.write(f"- **Models Tested**: {stats['models_tested']}\n")
        f.write(f"- **Hardware Platforms**: {', '.join(stats['hardware_platforms'])}\n")
        f.write(f"- **Overall Success Rate**: {stats['success_rate']:.1f}%\n\n")
        
        f.write("## Results by Model and Hardware\n\n")
        f.write("| Model | " + " | ".join(stats['hardware_platforms']) + " | Overall |\n")
        f.write("|-------|" + "|".join(["---------" for _ in stats['hardware_platforms']]) + "|--------|\n")
        
        for model in sorted(model_results.keys()):
            row = [model]
            model_success = 0
            model_total = 0
            
            for hardware in stats['hardware_platforms']:
                success = model_results[model].get(hardware, False)
                status = "✅" if success else "❌"
                row.append(status)
                
                model_success += 1 if success else 0
                model_total += 1
            
            # Add overall success rate for the model
            model_rate = (model_success / model_total * 100) if model_total > 0 else 0
            row.append(f"{model_rate:.1f}%")
            
            f.write("| " + " | ".join(row) + " |\n")
        
        f.write("\n## Detailed Results\n\n")
        for result in results:
            if not result["success"]:
                f.write(f"### {result['model']} on {result['hardware']} ❌\n\n")
                
                if "error" in result and result["error"]:
                    f.write(f"**Error**: {result['error']}\n\n")
                
                if "stderr" in result and result["stderr"]:
                    f.write("**Error Output**:\n")
                    f.write("```\n")
                    f.write(result["stderr"])
                    f.write("\n```\n\n")
        
        f.write("\n## Next Steps\n\n")
        if stats['failed_tests'] > 0:
            f.write("1. Investigate and fix the failing tests\n")
            f.write("2. Re-run the verification to confirm fixes\n")
            f.write("3. Run performance benchmarks on successfully verified models\n")
        else:
            f.write("1. All tests passed! Proceed with performance benchmarking\n")
            f.write("2. Implement advanced model compression techniques\n")
            f.write("3. Consider expanding to multi-node testing\n")
    
    logger.info(f"Report generated: {md_file}")
    logger.info(f"JSON data: {json_file}")
    
    return stats, md_file

def main():
    parser = argparse.ArgumentParser(description="Verify model functionality across hardware platforms")
    parser.add_argument("--models", nargs="+", help="Specific models to verify (default: all key models)")
    parser.add_argument("--hardware", nargs="+", help="Hardware platforms to test on (default: all available)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for each test")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel tests")
    parser.add_argument("--output-dir", default="./functionality_reports", help="Directory for output reports")
    
    args = parser.parse_args()
    
    # Use specified models or all key models
    models = args.models or KEY_MODELS
    
    # Use specified hardware or all available
    hardware_platforms = args.hardware or HARDWARE_PLATFORMS
    
    logger.info(f"Starting verification for models: {', '.join(models)}")
    logger.info(f"Testing on hardware platforms: {', '.join(hardware_platforms)}")
    
    results = verify_all_models(
        models=models,
        hardware_platforms=hardware_platforms,
        max_workers=args.max_workers,
        timeout=args.timeout
    )
    
    stats, report_file = generate_report(results, output_dir=args.output_dir)
    
    # Print summary to console
    print("\nVerification Summary:")
    print(f"Total Tests: {stats['total_tests']}")
    print(f"Successful Tests: {stats['successful_tests']}")
    print(f"Failed Tests: {stats['failed_tests']}")
    print(f"Overall Success Rate: {stats['success_rate']:.1f}%")
    print(f"Report generated: {report_file}")

if __name__ == "__main__":
    main()