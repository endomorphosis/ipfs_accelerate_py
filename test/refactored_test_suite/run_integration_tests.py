#!/usr/bin/env python3
"""
Run integration tests for generated model tests.

This script selects representative models from each architecture and runs
tests against them to verify the tests are correctly implemented.
"""

import os
import sys
import json
import time
import argparse
import logging
import datetime
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"integration_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Import required modules
try:
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from refactored_test_suite.generators.architecture_detector import ARCHITECTURE_TYPES
except ImportError:
    logger.error("Failed to import required modules")
    sys.exit(1)

# Define representative models for each architecture
REPRESENTATIVE_MODELS = {
    "encoder-only": [
        "bert-base-uncased",
        "roberta-base"
    ],
    "decoder-only": [
        "gpt2",
        "mistralai/Mistral-7B-v0.1"
    ],
    "encoder-decoder": [
        "t5-small",
        "facebook/bart-base"
    ],
    "vision": [
        "google/vit-base-patch16-224",
        "microsoft/swin-base-patch4-window7-224-in22k"
    ],
    "vision-encoder-text-decoder": [
        "openai/clip-vit-base-patch32",
        "Salesforce/blip-image-captioning-base"
    ],
    "speech": [
        "openai/whisper-tiny",
        "facebook/wav2vec2-base-960h"
    ],
    "multimodal": [
        "llava-hf/llava-1.5-7b-hf",
        "facebook/flava-full"
    ]
}

# Define mocked testing flag - in CI environments we mock dependencies
USE_MOCKS = os.environ.get("USE_MOCKS", "False").lower() == "true"

def run_test_for_model(test_file: str, model_id: str = None, device: str = None, save_results: bool = True) -> Dict[str, Any]:
    """
    Run a test for a specific model.
    
    Args:
        test_file: Path to test file
        model_id: Specific model ID to test, or None to use default
        device: Device to test on, or None to auto-detect
        save_results: Whether to save results to file
        
    Returns:
        Dict with test results
    """
    try:
        # Prepare command
        cmd = [sys.executable, test_file]
        
        if model_id:
            cmd.extend(["--model-id", model_id])
        
        if device:
            cmd.extend(["--device", device])
        
        if save_results:
            cmd.append("--save")
        
        # Set environment variables if using mocks
        env = os.environ.copy()
        if USE_MOCKS:
            env["MOCK_TORCH"] = "True"
            env["MOCK_TRANSFORMERS"] = "True"
            env["MOCK_TOKENIZERS"] = "True"
            env["MOCK_SENTENCEPIECE"] = "True"
        
        # Run command
        logger.info(f"Running test: {' '.join(cmd)}")
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        # Parse output
        success = result.returncode == 0
        
        # Extract model ID from output if possible
        tested_model = model_id
        if not tested_model:
            for line in result.stdout.splitlines():
                if "Successfully tested" in line:
                    parts = line.split("Successfully tested")[1].strip().split(" on ")
                    if len(parts) > 0:
                        tested_model = parts[0].strip()
                        break
        
        return {
            "test_file": test_file,
            "model_id": tested_model,
            "success": success,
            "elapsed_time": elapsed,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "mocked": USE_MOCKS
        }
    except Exception as e:
        logger.error(f"Error running test for {test_file}: {e}")
        return {
            "test_file": test_file,
            "model_id": model_id,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "mocked": USE_MOCKS
        }

def find_test_file_for_model(test_dir: str, model_name: str) -> Optional[str]:
    """
    Find the appropriate test file for a model.
    
    Args:
        test_dir: Directory containing test files
        model_name: Model name to find test for
        
    Returns:
        Path to test file, or None if not found
    """
    # Extract base model name without organization
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    
    # Clean up model name for matching
    model_name_clean = model_name.lower().replace("-", "_")
    
    # Try different patterns
    patterns = [
        f"test_hf_{model_name}.py",
        f"test_hf_{model_name_clean}.py",
        f"test_{model_name}.py",
        f"test_{model_name_clean}.py"
    ]
    
    # Extract base name without version or size
    base_name = model_name.split("-")[0].lower()
    patterns.append(f"test_hf_{base_name}.py")
    
    # Try to find a matching file
    for pattern in patterns:
        matches = list(Path(test_dir).glob(pattern))
        if matches:
            return str(matches[0])
    
    return None

def run_integration_tests(args):
    """
    Run integration tests for selected models.
    
    Args:
        args: Command-line arguments
    """
    results = []
    failed = []
    skipped = []
    
    # Determine which architectures to test
    architectures = args.architectures
    if "all" in architectures:
        architectures = list(REPRESENTATIVE_MODELS.keys())
    
    # Create results directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run tests for each architecture
    for arch in architectures:
        logger.info(f"\nTesting {arch} architecture:")
        
        # Get representative models for this architecture
        models = REPRESENTATIVE_MODELS.get(arch, [])
        if not models:
            logger.warning(f"No representative models defined for {arch}")
            continue
        
        # Test each model
        for model in models:
            logger.info(f"  Testing model: {model}")
            
            # Find test file
            test_file = find_test_file_for_model(args.test_dir, model)
            if not test_file:
                logger.warning(f"  ⚠️ No test file found for {model}, skipping")
                skipped.append({
                    "model": model,
                    "architecture": arch,
                    "reason": "No test file found"
                })
                continue
            
            # Run test
            result = run_test_for_model(
                test_file=test_file,
                model_id=model if args.use_specific_ids else None,
                device=args.device,
                save_results=args.save_results
            )
            
            # Log result
            if result["success"]:
                logger.info(f"  ✅ Test passed: {model}")
            else:
                logger.error(f"  ❌ Test failed: {model}")
                if "stderr" in result:
                    logger.error(f"  Error: {result['stderr']}")
                failed.append({
                    "model": model,
                    "architecture": arch,
                    "test_file": test_file,
                    "error": result.get("stderr", result.get("error", "Unknown error"))
                })
            
            results.append(result)
    
    # Generate report
    report_file = os.path.join(args.output_dir, f"integration_test_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    generate_report(results, failed, skipped, report_file, USE_MOCKS)
    
    # Print summary
    logger.info("\nTest Summary:")
    logger.info(f"- Total: {len(results)}")
    logger.info(f"- Passed: {sum(1 for r in results if r['success'])}")
    logger.info(f"- Failed: {len(failed)}")
    logger.info(f"- Skipped: {len(skipped)}")
    logger.info(f"- Report: {report_file}")
    
    return len(failed) == 0

def generate_report(results: List[Dict[str, Any]], failed: List[Dict[str, Any]], skipped: List[Dict[str, Any]], report_file: str, mocked: bool) -> None:
    """
    Generate an integration test report.
    
    Args:
        results: Test results
        failed: Failed tests
        skipped: Skipped tests
        report_file: Path to save report
        mocked: Whether tests were run with mocks
    """
    with open(report_file, "w") as f:
        f.write("# Model Test Integration Report\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write test mode
        f.write(f"**Test Mode**: {'MOCKED' if mocked else 'REAL'} dependencies\n\n")
        
        # Write summary
        total = len(results)
        passed = sum(1 for r in results if r["success"])
        
        f.write("## Summary\n\n")
        f.write(f"- **Total tests**: {total}\n")
        pass_percentage = (passed/total*100) if total > 0 else 0
        f.write(f"- **Passed**: {passed} ({pass_percentage:.1f}%)\n")
        f.write(f"- **Failed**: {len(failed)}\n")
        f.write(f"- **Skipped**: {len(skipped)}\n\n")
        
        # Write results by architecture
        f.write("## Results by Architecture\n\n")
        f.write("| Architecture | Total | Passed | Failed | Pass Rate |\n")
        f.write("|--------------|-------|--------|--------|----------|\n")
        
        arch_results = {}
        for result in results:
            # Determine architecture from test file
            arch = "unknown"
            for a, models in REPRESENTATIVE_MODELS.items():
                for model in models:
                    if model == result.get("model_id") or model in result.get("test_file", ""):
                        arch = a
                        break
            
            if arch not in arch_results:
                arch_results[arch] = {"total": 0, "passed": 0}
            
            arch_results[arch]["total"] += 1
            if result["success"]:
                arch_results[arch]["passed"] += 1
        
        for arch, stats in sorted(arch_results.items()):
            pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            f.write(f"| {arch} | {stats['total']} | {stats['passed']} | {stats['total'] - stats['passed']} | {pass_rate:.1f}% |\n")
        
        # Write detailed results
        f.write("\n## Test Details\n\n")
        f.write("| Model | Architecture | Result | Time (s) |\n")
        f.write("|-------|--------------|--------|----------|\n")
        
        for result in results:
            model_id = result.get("model_id", "Unknown")
            
            # Determine architecture
            arch = "unknown"
            for a, models in REPRESENTATIVE_MODELS.items():
                for model in models:
                    if model == model_id or model in result.get("test_file", ""):
                        arch = a
                        break
            
            status = "✅ Passed" if result["success"] else "❌ Failed"
            time_str = f"{result.get('elapsed_time', 0):.2f}"
            
            f.write(f"| {model_id} | {arch} | {status} | {time_str} |\n")
        
        # Write failed tests section
        if failed:
            f.write("\n## Failed Tests\n\n")
            
            for fail in failed:
                f.write(f"### {fail['model']} ({fail['architecture']})\n\n")
                f.write(f"- **Test file**: {fail['test_file']}\n")
                f.write(f"- **Error**:\n```\n{fail['error']}\n```\n\n")
        
        # Write skipped tests section
        if skipped:
            f.write("\n## Skipped Tests\n\n")
            
            for skip in skipped:
                f.write(f"- **{skip['model']}** ({skip['architecture']}): {skip['reason']}\n")
    
    logger.info(f"Report written to {report_file}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run integration tests for model tests")
    
    parser.add_argument(
        "--test-dir",
        default="./generated_tests",
        help="Directory containing test files"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./reports",
        help="Directory to save test results and reports"
    )
    
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=list(REPRESENTATIVE_MODELS.keys()) + ["all"],
        default=["all"],
        help="Specific architectures to test"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to test on (default: auto-detect)"
    )
    
    parser.add_argument(
        "--use-specific-ids",
        action="store_true",
        help="Use specific model IDs instead of test file defaults"
    )
    
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save test results to files"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mocked dependencies (useful for CI/CD)"
    )
    
    args = parser.parse_args()
    
    # Set mock flag from args
    global USE_MOCKS
    USE_MOCKS = USE_MOCKS or args.mock
    
    return args

def main():
    """Command-line entry point."""
    args = parse_args()
    success = run_integration_tests(args)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())