#!/usr/bin/env python3
"""
Verify Key Model Tests Across Hardware Platforms

This script runs tests for the key model classes on all available hardware platforms
to verify the new implementations.
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = CURRENT_DIR / "skills"
RESULTS_DIR = CURRENT_DIR / "collected_results"

# Key models to verify, from CLAUDE.md
KEY_MODELS = [
    {"name": "bert", "priority": "high", "modality": "text"},
    {"name": "t5", "priority": "high", "modality": "text"},
    {"name": "llama", "priority": "high", "modality": "text"},
    {"name": "clip", "priority": "high", "modality": "vision"},
    {"name": "vit", "priority": "high", "modality": "vision"},
    {"name": "clap", "priority": "high", "modality": "audio"},
    {"name": "whisper", "priority": "high", "modality": "audio"},
    {"name": "wav2vec2", "priority": "high", "modality": "audio"},
    {"name": "llava", "priority": "high", "modality": "multimodal"},
    {"name": "llava_next", "priority": "high", "modality": "multimodal"},
    {"name": "xclip", "priority": "medium", "modality": "vision"},
    {"name": "qwen2", "priority": "medium", "modality": "text"},
    {"name": "detr", "priority": "medium", "modality": "vision"}
]

def check_hardware_availability():
    """Check for available hardware platforms."""
    available_platforms = {
        "cpu": True,
        "cuda": False,
        "openvino": False,
        "mps": False,
        "rocm": False
    }
    
    try:
        import torch
        
        # Check CUDA
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            available_platforms["cuda"] = True
            logger.info(f"CUDA available with {torch.cuda.device_count()} devices")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch, 'mps') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available_platforms["mps"] = True
            logger.info("Apple Silicon MPS available")
        
        # Check for ROCm/HIP (AMD)
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if any("hip" in d.lower() for d in cuda_devices):
                available_platforms["rocm"] = True
                logger.info("AMD ROCm/HIP available")
        
        # Check for OpenVINO
        try:
            import openvino
            available_platforms["openvino"] = True
            logger.info("OpenVINO available")
        except ImportError:
            pass
    
    except ImportError:
        logger.warning("PyTorch not available, defaulting to CPU only")
    
    return available_platforms

def run_test_for_model(model_name, platform=None):
    """
    Run a specific test for a model, optionally on a specific platform.
    
    Args:
        model_name: Name of the model to test
        platform: Specific platform to test on (optional)
    
    Returns:
        Tuple of (success, output)
    """
    test_file_path = SKILLS_DIR / f"test_hf_{model_name}.py"
    
    # Check if test file exists
    if not test_file_path.exists():
        logger.error(f"Test file does not exist for {model_name} at {test_file_path}")
        return False, f"Test file does not exist for {model_name}"
    
    # Build command
    command = [sys.executable, str(test_file_path)]
    
    # Add platform parameter if specified
    if platform:
        command.extend(["--platform", platform])
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Capture output to a log file
    log_file = RESULTS_DIR / f"{model_name}_{platform or 'all'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Run the test
    logger.info(f"Running test for {model_name} on {platform or 'all platforms'}")
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Process and log output in real-time
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                f.write(line)
                f.flush()
                logger.debug(line.strip())
            
            process.wait()
            
            # Check if test was successful
            success = process.returncode == 0
            output = "".join(output_lines)
            
            if success:
                logger.info(f"Test for {model_name} on {platform or 'all platforms'} completed successfully")
            else:
                logger.error(f"Test for {model_name} on {platform or 'all platforms'} failed with code {process.returncode}")
            
            return success, output
    
    except Exception as e:
        logger.error(f"Error running test for {model_name} on {platform or 'all platforms'}: {e}")
        return False, str(e)

def verify_all_models(platforms=None, high_priority_only=False):
    """
    Verify all key models, optionally on specific platforms.
    
    Args:
        platforms: List of platforms to test on (default: all available)
        high_priority_only: If True, only test high priority models
    
    Returns:
        Dictionary with test results
    """
    # Check available hardware
    available_hardware = check_hardware_availability()
    
    # Filter platforms if specified
    if platforms:
        platforms = [p for p in platforms if available_hardware.get(p, False)]
    else:
        platforms = [p for p, available in available_hardware.items() if available]
    
    logger.info(f"Testing on platforms: {', '.join(platforms)}")
    
    # Filter models if high priority only
    models_to_test = KEY_MODELS
    if high_priority_only:
        models_to_test = [m for m in KEY_MODELS if m["priority"] == "high"]
    
    # Initialize results
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "available_hardware": available_hardware,
        "platforms_tested": platforms,
        "models_tested": len(models_to_test),
        "results": {}
    }
    
    # Run tests for each model on each platform
    for model_data in models_to_test:
        model_name = model_data["name"]
        model_results = {
            "priority": model_data["priority"],
            "modality": model_data["modality"],
            "platforms": {}
        }
        
        for platform in platforms:
            success, output = run_test_for_model(model_name, platform)
            
            # Extract implementation type from output
            implementation_type = "UNKNOWN"
            if "REAL" in output:
                implementation_type = "REAL"
            elif "MOCK" in output:
                implementation_type = "MOCK"
            
            model_results["platforms"][platform] = {
                "success": success,
                "implementation_type": implementation_type
            }
        
        results["results"][model_name] = model_results
    
    # Calculate summary statistics
    total_tests = len(models_to_test) * len(platforms)
    successful_tests = sum(
        1 for model_data in results["results"].values()
        for platform_data in model_data["platforms"].values()
        if platform_data["success"]
    )
    real_implementations = sum(
        1 for model_data in results["results"].values()
        for platform_data in model_data["platforms"].values()
        if platform_data["implementation_type"] == "REAL"
    )
    
    results["summary"] = {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "real_implementations": real_implementations,
        "success_rate": f"{successful_tests / total_tests * 100:.1f}%" if total_tests > 0 else "N/A",
        "real_implementation_rate": f"{real_implementations / total_tests * 100:.1f}%" if total_tests > 0 else "N/A"
    }
    
    return results

def save_results(results, filename=None):
    """
    Save test results to a file.
    
    Args:
        results: Test results dictionary
        filename: Optional filename to save to
    
    Returns:
        Path to the saved file
    """
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"key_model_verification_{timestamp}.json"
    
    # Save results
    results_path = RESULTS_DIR / filename
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    return results_path

def print_results_summary(results):
    """
    Print a summary of test results.
    
    Args:
        results: Test results dictionary
    """
    print("\n" + "=" * 80)
    print(f"KEY MODEL VERIFICATION SUMMARY ({results['timestamp']})")
    print("=" * 80)
    
    print(f"\nHardware Availability:")
    for platform, available in results["available_hardware"].items():
        status = "Available" if available else "Not Available"
        print(f"  - {platform.upper()}: {status}")
    
    print(f"\nPlatforms Tested: {', '.join(results['platforms_tested'])}")
    print(f"Models Tested: {results['models_tested']}")
    
    print(f"\nSummary Statistics:")
    summary = results["summary"]
    print(f"  - Total Tests Run: {summary['total_tests']}")
    print(f"  - Successful Tests: {summary['successful_tests']} ({summary['success_rate']})")
    print(f"  - Real Implementations: {summary['real_implementations']} ({summary['real_implementation_rate']})")
    
    print("\nResults by Model:")
    for model_name, model_data in results["results"].items():
        print(f"\n  - {model_name} ({model_data['priority']}, {model_data['modality']}):")
        
        for platform, platform_data in model_data["platforms"].items():
            status = "✅ Success" if platform_data["success"] else "❌ Failed"
            impl_type = platform_data["implementation_type"]
            impl_indicator = "🔵 REAL" if impl_type == "REAL" else "🟠 MOCK"
            
            print(f"    - {platform.upper()}: {status} ({impl_indicator})")
    
    print("\n" + "=" * 80)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Verify Key Model Tests Across Hardware Platforms")
    parser.add_argument("--model", type=str, help="Verify a specific model")
    parser.add_argument("--platform", type=str, help="Test on a specific platform (cpu, cuda, openvino, mps, rocm)")
    parser.add_argument("--high-priority", action="store_true", help="Only test high priority models")
    parser.add_argument("--save", type=str, help="Save results to a specific file")
    
    args = parser.parse_args()
    
    # If a specific model is requested, just test that one
    if args.model:
        model_data = next((m for m in KEY_MODELS if m["name"] == args.model), None)
        if model_data:
            platform = args.platform
            success, output = run_test_for_model(args.model, platform)
            
            print(f"\nTest for {args.model} on {platform or 'default platform'}:")
            print(f"  - Success: {success}")
            print(f"  - Output: {output[:500]}..." if len(output) > 500 else f"  - Output: {output}")
            
            return
        else:
            print(f"Model {args.model} not found in the list of key models.")
            return
    
    # Verify all models
    start_time = time.time()
    platforms = [args.platform] if args.platform else None
    
    print(f"Starting verification of key models...")
    results = verify_all_models(platforms, args.high_priority)
    end_time = time.time()
    
    # Add execution time to results
    results["execution_time_seconds"] = end_time - start_time
    
    # Save results
    save_results(results, args.save)
    
    # Print summary
    print_results_summary(results)
    
    print(f"\nVerification completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()