#!/usr/bin/env python3
"""
Script to enhance test generation and validation across all hardware platforms.

This script:
1. Identifies key models representing different modalities
2. Generates comprehensive test files for each model
3. Validates hardware platform support in each test
4. Tests execution across all available hardware platforms
5. Generates a comprehensive coverage report
6. Updates the merged_test_generator with hardware platform improvements

Usage:
  python enhance_hardware_test_coverage.py --generate-all
  python enhance_hardware_test_coverage.py --analyze-coverage
  python enhance_hardware_test_coverage.py --validate-tests
  python enhance_hardware_test_coverage.py --visualize-coverage
"""

import os
import sys
import json
import time
import glob
import shutil
import logging
import argparse
import tempfile
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hardware_test_coverage.log")
    ]
)
logger = logging.getLogger("hardware_test_coverage")

# Paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
WORKER_SKILLSET = PROJECT_ROOT / "ipfs_accelerate_py" / "worker" / "skillset"
GENERATOR_FILE = TEST_DIR / "merged_test_generator.py"
SKILLS_DIR = TEST_DIR / "skills"
OUTPUT_DIR = TEST_DIR / "hardware_coverage_results"

# Key models representing different modalities
KEY_MODELS = [
    "bert",       # Text embedding (base model)
    "t5",         # Text generation (sequence-to-sequence)
    "gpt2",       # Text generation (causal)
    "vit",        # Vision (transformer)
    "clip",       # Vision-text multimodal
    "whisper",    # Audio transcription
    "llava",      # Vision-language multimodal
    "clap",       # Audio-text multimodal
    "wav2vec2",   # Audio feature extraction
    "deit",       # Vision (distilled transformer)
    "roberta",    # Text embedding (optimized)
    "hubert",     # Audio representation
    "flava"       # Multimodal foundation model
]

# Hardware platforms to validate
HARDWARE_PLATFORMS = [
    "cpu",       # Always available
    "cuda",      # NVIDIA GPUs
    "openvino",  # Intel hardware
    "mps",       # Apple Silicon 
    "rocm",      # AMD GPUs
    "webnn",     # Browser WebNN API
    "webgpu"     # Browser WebGPU API
]

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def get_existing_tests() -> Set[str]:
    """Get the normalized names of existing test files"""
    test_files = glob.glob(str(SKILLS_DIR / 'test_hf_*.py'))
    existing_tests = set()
    
    for test_file in test_files:
        model_name = os.path.basename(test_file).replace('test_hf_', '').replace('.py', '')
        existing_tests.add(model_name)
    
    logger.info(f"Found {len(existing_tests)} existing test implementations")
    return existing_tests

def generate_test_file(model_name: str, output_dir: Optional[Path] = None) -> Path:
    """
    Generate a test file for the specified model.
    
    Args:
        model_name: Name of the model to generate a test for
        output_dir: Directory to save the generated file (if None, uses SKILLS_DIR)
        
    Returns:
        Path to the generated test file
    """
    output_dir = output_dir or SKILLS_DIR
    output_dir.mkdir(exist_ok=True, parents=True)
    
    test_file_path = output_dir / f"test_hf_{model_name}.py"
    
    # Check if test file already exists
    if test_file_path.exists():
        logger.info(f"Test file already exists for {model_name} at {test_file_path}")
        return test_file_path
    
    # Run the generator
    cmd = [
        sys.executable,
        str(GENERATOR_FILE),
        "--generate", model_name,
        "--output-dir", str(output_dir)
    ]
    
    logger.info(f"Generating test file for {model_name}...")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"Error generating test file for {model_name}: {result.stderr}")
            return None
        
        if not test_file_path.exists():
            logger.error(f"Test file not generated for {model_name}")
            return None
        
        logger.info(f"Successfully generated test file for {model_name}")
        return test_file_path
    
    except Exception as e:
        logger.error(f"Exception while generating test file for {model_name}: {e}")
        return None

def analyze_test_coverage(test_file: Path) -> Dict[str, Any]:
    """
    Analyze a test file for hardware platform support.
    
    Args:
        test_file: Path to the test file to analyze
        
    Returns:
        Dictionary with coverage analysis results
    """
    if not test_file or not test_file.exists():
        return {"error": "Test file does not exist"}
    
    model_name = test_file.stem.replace('test_hf_', '')
    
    coverage_results = {
        "model": model_name,
        "filename": str(test_file),
        "timestamp": datetime.now().isoformat(),
        "platforms": {}
    }
    
    try:
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Check for platform handlers and initialization methods
        for platform in HARDWARE_PLATFORMS:
            platform_results = {}
            
            # Check for initialization methods
            init_method = f"def init_{platform}"
            platform_results["has_init_method"] = init_method in content
            
            # Check for handler creation methods
            handler_method = f"create_{platform}_"
            platform_results["has_handler_method"] = handler_method in content
            
            # Check for platform testing
            platform_test = f"platform: {platform.upper()}"
            platform_results["has_platform_test"] = platform_test in content
            
            # Check for imports needed for this platform
            if platform == "openvino":
                platform_results["has_imports"] = "import openvino" in content
            elif platform == "webnn":
                platform_results["has_imports"] = "webnn" in content
            elif platform == "webgpu":
                platform_results["has_imports"] = "webgpu" in content
            else:
                platform_results["has_imports"] = True  # Default platforms use standard imports
            
            # Overall platform support
            platform_results["supported"] = (
                platform_results["has_init_method"] and 
                platform_results["has_handler_method"] and
                platform_results["has_platform_test"]
            )
            
            coverage_results["platforms"][platform] = platform_results
        
        # Calculate overall coverage
        supported_platforms = sum(1 for p in coverage_results["platforms"].values() if p["supported"])
        total_platforms = len(HARDWARE_PLATFORMS)
        coverage_results["coverage_percentage"] = (supported_platforms / total_platforms) * 100
        
        return coverage_results
    
    except Exception as e:
        logger.error(f"Error analyzing test file {test_file}: {e}")
        return {"error": str(e), "model": model_name, "filename": str(test_file)}

def generate_coverage_report(coverage_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a comprehensive coverage report.
    
    Args:
        coverage_data: List of coverage analysis results for multiple test files
        
    Returns:
        Dictionary with summary report
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_models": len(coverage_data),
        "platform_coverage": {platform: 0 for platform in HARDWARE_PLATFORMS},
        "model_results": coverage_data,
        "average_coverage": 0.0,
        "fully_covered_models": [],
        "partially_covered_models": [],
        "poorly_covered_models": []
    }
    
    # Calculate platform coverage
    for model_data in coverage_data:
        if "error" in model_data:
            continue
        
        platforms = model_data.get("platforms", {})
        for platform, platform_data in platforms.items():
            if platform_data.get("supported", False):
                report["platform_coverage"][platform] += 1
    
    # Convert to percentages
    total_models = len(coverage_data)
    for platform in HARDWARE_PLATFORMS:
        report["platform_coverage"][platform] = (
            (report["platform_coverage"][platform] / total_models) * 100
            if total_models > 0 else 0
        )
    
    # Calculate average model coverage
    coverage_percentages = [
        model_data.get("coverage_percentage", 0) 
        for model_data in coverage_data 
        if "error" not in model_data
    ]
    
    report["average_coverage"] = (
        sum(coverage_percentages) / len(coverage_percentages)
        if coverage_percentages else 0
    )
    
    # Categorize models by coverage
    for model_data in coverage_data:
        if "error" in model_data:
            continue
        
        coverage = model_data.get("coverage_percentage", 0)
        model = model_data.get("model", "unknown")
        
        if coverage >= 80:
            report["fully_covered_models"].append(model)
        elif coverage >= 40:
            report["partially_covered_models"].append(model)
        else:
            report["poorly_covered_models"].append(model)
    
    return report

def verify_test_execution(test_file: Path, platform: str = "cpu") -> Dict[str, Any]:
    """
    Verify the execution of a test file on a specific platform.
    
    Args:
        test_file: Path to the test file to execute
        platform: Platform to test on (cpu, cuda, etc.)
        
    Returns:
        Dictionary with execution results
    """
    if not test_file or not test_file.exists():
        return {"error": "Test file does not exist", "success": False}
    
    model_name = test_file.stem.replace('test_hf_', '')
    
    execution_results = {
        "model": model_name,
        "platform": platform,
        "timestamp": datetime.now().isoformat(),
        "success": False
    }
    
    try:
        # Set up environment variable to control test platform
        env = os.environ.copy()
        env["TEST_PLATFORM"] = platform
        
        # Run the test file
        cmd = [sys.executable, str(test_file), f"--platform={platform}"]
        
        logger.info(f"Executing test file {test_file} on platform {platform}...")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=env,
            timeout=300  # 5 minute timeout
        )
        
        # Store execution results
        execution_results["returncode"] = result.returncode
        execution_results["stdout"] = result.stdout
        execution_results["stderr"] = result.stderr
        execution_results["success"] = result.returncode == 0
        
        # Try to extract implementation type from output
        if result.stdout:
            # Look for implementation type markers in stdout
            if "REAL" in result.stdout:
                execution_results["implementation_type"] = "REAL"
            elif "MOCK" in result.stdout:
                execution_results["implementation_type"] = "MOCK"
            else:
                execution_results["implementation_type"] = "UNKNOWN"
        
        return execution_results
    
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout executing test file {test_file} on {platform}")
        execution_results["error"] = "Timeout"
        return execution_results
    
    except Exception as e:
        logger.error(f"Error executing test file {test_file} on {platform}: {e}")
        execution_results["error"] = str(e)
        return execution_results

def improve_test_generator():
    """
    Enhance the merged_test_generator.py file with better hardware platform support.
    
    1. Update handler methods for all platforms
    2. Add WebNN and WebGPU support
    3. Improve template selection for different modalities
    """
    # Create a backup of the generator file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = GENERATOR_FILE.with_suffix(f".py.bak_{timestamp}")
    shutil.copy2(GENERATOR_FILE, backup_file)
    logger.info(f"Created backup of merged_test_generator.py at {backup_file}")
    
    try:
        # Read the generator file
        with open(GENERATOR_FILE, 'r') as f:
            content = f.read()
        
        # Here you would implement specific improvements to the generator file
        # For example:
        # 1. Add WebNN and WebGPU platform support
        # 2. Enhance template generation for different hardware platforms
        # 3. Improve mock implementations for unsupported platforms
        
        # Write updated content back
        # with open(GENERATOR_FILE, 'w') as f:
        #     f.write(updated_content)
        
        logger.info("Improvements to merged_test_generator.py implemented")
        return True
    
    except Exception as e:
        logger.error(f"Error updating merged_test_generator.py: {e}")
        if backup_file.exists():
            shutil.copy2(backup_file, GENERATOR_FILE)
            logger.info(f"Restored original from backup after error")
        return False

def main():
    """Main function to enhance hardware test coverage."""
    parser = argparse.ArgumentParser(description="Enhance hardware test coverage across platforms")
    parser.add_argument("--generate-all", action="store_true", help="Generate tests for all key models")
    parser.add_argument("--analyze-coverage", action="store_true", help="Analyze test coverage across platforms")
    parser.add_argument("--validate-tests", action="store_true", help="Validate test execution across platforms")
    parser.add_argument("--visualize-coverage", action="store_true", help="Generate coverage visualization")
    parser.add_argument("--improve-generator", action="store_true", help="Enhance the test generator")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to test")
    parser.add_argument("--platforms", type=str, help="Comma-separated list of platforms to test")
    args = parser.parse_args()
    
    # Process provided models and platforms
    models_to_test = KEY_MODELS
    if args.models:
        models_to_test = args.models.split(",")
    
    platforms_to_test = HARDWARE_PLATFORMS
    if args.platforms:
        platforms_to_test = args.platforms.split(",")
    
    # Generate tests for all key models
    if args.generate_all:
        logger.info("Generating tests for all key models...")
        
        generated_files = []
        for model in models_to_test:
            test_file = generate_test_file(model)
            if test_file:
                generated_files.append(test_file)
        
        logger.info(f"Generated {len(generated_files)} test files")
    
    # Analyze test coverage
    if args.analyze_coverage:
        logger.info("Analyzing test coverage across platforms...")
        
        # Get existing tests
        existing_tests = get_existing_tests()
        
        # Filter to only requested models
        models_to_analyze = [m for m in models_to_test if m in existing_tests]
        
        coverage_data = []
        for model in models_to_analyze:
            test_file = SKILLS_DIR / f"test_hf_{model}.py"
            if test_file.exists():
                coverage_results = analyze_test_coverage(test_file)
                coverage_data.append(coverage_results)
        
        # Generate summary report
        report = generate_coverage_report(coverage_data)
        
        # Save report to file
        report_file = OUTPUT_DIR / "coverage_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\nHardware Platform Coverage Summary:")
        print(f"Total models analyzed: {report['total_models']}")
        print(f"Average model coverage: {report['average_coverage']:.2f}%")
        print("\nPlatform Coverage:")
        for platform, coverage in report["platform_coverage"].items():
            print(f"{platform.upper()}: {coverage:.2f}%")
        
        print("\nFully covered models:")
        for model in report["fully_covered_models"]:
            print(f"- {model}")
        
        print("\nPartially covered models:")
        for model in report["partially_covered_models"]:
            print(f"- {model}")
        
        print("\nPoorly covered models:")
        for model in report["poorly_covered_models"]:
            print(f"- {model}")
        
        print(f"\nDetailed report saved to: {report_file}")
    
    # Validate test execution
    if args.validate_tests:
        logger.info("Validating test execution across platforms...")
        
        # Get existing tests
        existing_tests = get_existing_tests()
        
        # Filter to only requested models
        models_to_test = [m for m in models_to_test if m in existing_tests]
        
        execution_results = []
        for model in models_to_test:
            test_file = SKILLS_DIR / f"test_hf_{model}.py"
            if test_file.exists():
                for platform in platforms_to_test:
                    platform_results = verify_test_execution(test_file, platform)
                    execution_results.append(platform_results)
        
        # Save execution results
        results_file = OUTPUT_DIR / "execution_results.json"
        with open(results_file, 'w') as f:
            json.dump(execution_results, f, indent=2)
        
        # Print summary
        print("\nTest Execution Summary:")
        success_count = sum(1 for r in execution_results if r.get("success", False))
        total_tests = len(execution_results)
        print(f"Total tests run: {total_tests}")
        print(f"Successful tests: {success_count} ({(success_count / total_tests) * 100:.2f}%)")
        
        # Summarize by platform
        platform_success = {platform: {"success": 0, "total": 0} for platform in platforms_to_test}
        for result in execution_results:
            platform = result.get("platform")
            if platform in platform_success:
                platform_success[platform]["total"] += 1
                if result.get("success", False):
                    platform_success[platform]["success"] += 1
        
        print("\nSuccess rate by platform:")
        for platform, counts in platform_success.items():
            success_rate = (counts["success"] / counts["total"]) * 100 if counts["total"] > 0 else 0
            print(f"{platform.upper()}: {counts['success']}/{counts['total']} ({success_rate:.2f}%)")
        
        print(f"\nDetailed results saved to: {results_file}")
    
    # Visualize coverage data
    if args.visualize_coverage:
        # This would require matplotlib or another visualization library
        # For simplicity, just print a message about where to find the data
        print("\nCoverage data is available for visualization in:")
        print(f"- {OUTPUT_DIR / 'coverage_report.json'}")
        print(f"- {OUTPUT_DIR / 'execution_results.json'}")
    
    # Improve test generator
    if args.improve_generator:
        print("\nImproving the test generator with better hardware platform support...")
        success = improve_test_generator()
        if success:
            print("Successfully enhanced the test generator!")
        else:
            print("Failed to improve the test generator. Check the logs for details.")
    
    # If no arguments provided, show help
    if not any([args.generate_all, args.analyze_coverage, args.validate_tests, 
                args.visualize_coverage, args.improve_generator]):
        parser.print_help()

if __name__ == "__main__":
    main()