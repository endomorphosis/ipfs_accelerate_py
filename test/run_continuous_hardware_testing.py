#!/usr/bin/env python
"""
Continuous Integration Script for Hardware Compatibility Testing

This script runs automated hardware compatibility tests and integrates with CI systems
like GitHub Actions, GitLab CI, or Jenkins. It handles test execution, result reporting,
and error notification.
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import platform
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
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
RESULTS_DIR = os.path.join(SCRIPT_DIR, "hardware_compatibility_results")
CACHE_DIR = os.path.join(SCRIPT_DIR, "hardware_cache")
HISTORY_FILE = os.path.join(RESULTS_DIR, "hardware_test_history.json")

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run continuous hardware compatibility testing")
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR,
                      help="Output directory for test results")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR,
                      help="Directory for hardware detection cache")
    parser.add_argument("--history-file", type=str, default=HISTORY_FILE,
                      help="Path to test history JSON file")
    parser.add_argument("--models", type=str, 
                      help="Comma-separated list of models to test (default: use representative models)")
    parser.add_argument("--platforms", type=str,
                      help="Comma-separated list of platforms to test (default: detect automatically)")
    parser.add_argument("--all-families", action="store_true",
                      help="Test all model families (default: use representative models)")
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
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    return parser.parse_args()

def ensure_directories(args):
    """Ensure all necessary directories exist"""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
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
    
    return 0

if __name__ == "__main__":
    sys.exit(main())