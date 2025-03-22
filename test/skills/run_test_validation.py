#!/usr/bin/env python3
"""
Run Test Validation

This script executes a subset of the generated model tests using small models to verify functionality.
It validates that the tests can initialize models, run inference, and handle errors gracefully.

Usage:
    python run_test_validation.py [--directory TESTS_DIR] [--max-tests MAX_TESTS] [--report REPORT_FILE] [--verbose]
"""

import os
import sys
import importlib.util
import argparse
import logging
import json
import time
import glob
import random
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Small models for testing different architectures - all under 500MB
SMALL_MODELS = {
    "encoder-only": [
        "distilbert-base-uncased",
        "prajjwal1/bert-tiny",
        "google/electra-small-discriminator",
        "distilroberta-base"
    ],
    "decoder-only": [
        "openai-community/gpt2",
        "gpt2-medium",
        "facebook/opt-125m",
        "mistralai/Mistral-7B-Instruct-v0.2"  # Larger but commonly used
    ],
    "encoder-decoder": [
        "t5-small",
        "google/flan-t5-small",
        "facebook/bart-base",
        "google/pegasus-xsum"
    ],
    "vision": [
        "google/vit-base-patch16-224",
        "microsoft/resnet-50",
        "facebook/deit-tiny-patch16-224",
        "facebook/convnext-tiny-224"
    ],
    "vision-text": [
        "openai/clip-vit-base-patch32",
        "Salesforce/blip-image-captioning-base",
        "valhalla/vit-bert-image-classification"
    ],
    "speech": [
        "facebook/wav2vec2-base",
        "openai/whisper-tiny",
        "facebook/hubert-base-ls960"
    ],
    "multimodal": [
        "openai/clip-vit-base-patch32",
        "Salesforce/blip-image-captioning-base"
    ]
}

class TestRunner:
    """Runs HuggingFace model tests to validate functionality."""
    
    def __init__(self, directory: str, max_tests: int, report_file: str, verbose: bool = False):
        """Initialize the test runner.
        
        Args:
            directory: Directory containing test files
            max_tests: Maximum number of tests to run
            report_file: Path to output report file
            verbose: Whether to print verbose output
        """
        self.directory = Path(directory)
        self.max_tests = max_tests
        self.report_file = Path(report_file)
        self.verbose = verbose
        
        self.results = {
            "passed": [],
            "failed": [],
            "stats": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "by_architecture": {},
                "by_error_type": {},
                "avg_execution_time": 0
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create architecture stats
        for arch in SMALL_MODELS.keys():
            self.results["stats"]["by_architecture"][arch] = {
                "total": 0, "passed": 0, "failed": 0
            }
    
    def find_test_files(self) -> Dict[str, List[Path]]:
        """Find test files grouped by architecture."""
        test_files_by_arch = {arch: [] for arch in SMALL_MODELS.keys()}
        
        all_test_files = list(self.directory.glob("test_hf_*.py"))
        
        # Group files by architecture
        for file_path in all_test_files:
            model_name = self._extract_model_name(file_path)
            architecture = self._get_model_architecture(model_name)
            
            if architecture in test_files_by_arch:
                test_files_by_arch[architecture].append(file_path)
        
        return test_files_by_arch
    
    def _extract_model_name(self, file_path: Path) -> str:
        """Extract the model name from a test file path."""
        return file_path.stem.replace("test_hf_", "")
    
    def _get_model_architecture(self, model_name: str) -> str:
        """Determine the model architecture from the model name."""
        # Architecture type definitions from validate_model_tests.py
        ARCHITECTURE_TYPES = {
            "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta"],
            "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt"],
            "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan"],
            "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2"],
            "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip"],
            "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
            "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava"]
        }
        
        model_name_lower = model_name.lower()
        
        for arch_type, models in ARCHITECTURE_TYPES.items():
            for model in models:
                if model.lower() in model_name_lower:
                    return arch_type
        
        return "unknown"
    
    def select_tests_to_run(self) -> List[Tuple[Path, str, str]]:
        """Select a subset of tests to run for each architecture."""
        test_files_by_arch = self.find_test_files()
        
        selected_tests = []
        
        # Calculate how many tests to run for each architecture
        total_arch_count = len([a for a in test_files_by_arch.keys() if test_files_by_arch[a]])
        tests_per_arch = max(1, min(5, self.max_tests // total_arch_count))
        
        # Select tests for each architecture
        for arch, tests in test_files_by_arch.items():
            if not tests:
                continue
                
            # Select random subset
            arch_tests = random.sample(tests, min(tests_per_arch, len(tests)))
            
            # Assign a small model for each test
            for test_path in arch_tests:
                model_name = self._extract_model_name(test_path)
                small_model = random.choice(SMALL_MODELS.get(arch, ["distilbert-base-uncased"]))
                selected_tests.append((test_path, small_model, arch))
        
        logger.info(f"Selected {len(selected_tests)} tests to run")
        return selected_tests
    
    def run_all_tests(self) -> Dict:
        """Run all selected tests and return results."""
        selected_tests = self.select_tests_to_run()
        self.results["stats"]["total"] = len(selected_tests)
        
        # Set up total execution time tracking
        total_execution_time = 0
        
        # Use ThreadPoolExecutor for parallel test execution
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 4)) as executor:
            future_to_test = {
                executor.submit(self.run_test, test_path, model_id, arch): (test_path, model_id, arch) 
                for test_path, model_id, arch in selected_tests
            }
            
            for future in as_completed(future_to_test):
                test_path, model_id, arch = future_to_test[future]
                try:
                    result = future.result()
                    total_execution_time += result.get("execution_time", 0)
                except Exception as exc:
                    logger.error(f"Exception running {test_path}: {exc}")
                    self._record_failure(test_path, model_id, arch, ["Exception during test execution", str(exc)])
        
        # Calculate average execution time
        if self.results["stats"]["passed"] > 0:
            self.results["stats"]["avg_execution_time"] = round(total_execution_time / self.results["stats"]["passed"], 2)
        
        # Generate and save report
        self.generate_report()
        
        return self.results
    
    def run_test(self, test_path: Path, model_id: str, arch: str) -> Dict:
        """Run a single test file as a subprocess."""
        logger.info(f"Running test: {test_path} with model {model_id}")
        
        start_time = time.time()
        
        try:
            # Run the test file as a subprocess
            cmd = [sys.executable, str(test_path), "--model", model_id]
            
            # Run with a timeout to prevent hanging tests
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                self._record_success(test_path, model_id, arch, execution_time)
                return {
                    "success": True, 
                    "output": result.stdout,
                    "execution_time": execution_time
                }
            else:
                errors = [
                    f"Test failed with exit code {result.returncode}",
                    f"STDOUT: {result.stdout[:500]}...",
                    f"STDERR: {result.stderr[:500]}..."
                ]
                self._record_failure(test_path, model_id, arch, errors)
                self._record_error("test_execution_failed")
                return {
                    "success": False, 
                    "errors": errors,
                    "output": result.stdout,
                    "stderr": result.stderr,
                    "execution_time": execution_time
                }
                
        except subprocess.TimeoutExpired:
            self._record_failure(test_path, model_id, arch, ["Test execution timed out after 5 minutes"])
            self._record_error("timeout")
            return {"success": False, "errors": ["Test execution timed out"]}
            
        except Exception as e:
            self._record_failure(test_path, model_id, arch, [f"Error running test: {str(e)}"])
            self._record_error("runtime_exception")
            return {"success": False, "errors": [str(e)]}
    
    def _record_success(self, test_path: Path, model_id: str, arch: str, execution_time: float):
        """Record a successful test run."""
        self.results["passed"].append({
            "file": str(test_path),
            "model_name": self._extract_model_name(test_path),
            "model_id": model_id,
            "architecture": arch,
            "execution_time": round(execution_time, 2)
        })
        self.results["stats"]["passed"] += 1
        
        # Update architecture stats
        if arch in self.results["stats"]["by_architecture"]:
            self.results["stats"]["by_architecture"][arch]["passed"] += 1
            self.results["stats"]["by_architecture"][arch]["total"] += 1
        
        if self.verbose:
            logger.info(f"✅ Passed: {test_path} with model {model_id} in {round(execution_time, 2)}s")
    
    def _record_failure(self, test_path: Path, model_id: str, arch: str, errors: List[str]):
        """Record a failed test run."""
        self.results["failed"].append({
            "file": str(test_path),
            "model_name": self._extract_model_name(test_path),
            "model_id": model_id,
            "architecture": arch,
            "errors": errors
        })
        self.results["stats"]["failed"] += 1
        
        # Update architecture stats
        if arch in self.results["stats"]["by_architecture"]:
            self.results["stats"]["by_architecture"][arch]["failed"] += 1
            self.results["stats"]["by_architecture"][arch]["total"] += 1
        
        if self.verbose:
            logger.error(f"❌ Failed: {test_path} with model {model_id}")
            for error in errors:
                logger.error(f"  - {error}")
    
    def _record_error(self, error_type: str):
        """Record an error type for statistics."""
        if error_type not in self.results["stats"]["by_error_type"]:
            self.results["stats"]["by_error_type"][error_type] = 0
        self.results["stats"]["by_error_type"][error_type] += 1
    
    def generate_report(self):
        """Generate and save a validation report."""
        # Calculate percentages
        total = self.results["stats"]["total"]
        if total > 0:
            self.results["stats"]["passed_percent"] = round(self.results["stats"]["passed"] / total * 100, 2)
            self.results["stats"]["failed_percent"] = round(self.results["stats"]["failed"] / total * 100, 2)
        else:
            self.results["stats"]["passed_percent"] = 0
            self.results["stats"]["failed_percent"] = 0
        
        # Save JSON report
        with open(self.report_file.with_suffix('.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate markdown report
        md_report = self._generate_markdown_report()
        with open(self.report_file, 'w') as f:
            f.write(md_report)
        
        logger.info(f"Test execution report saved to {self.report_file}")
        logger.info(f"JSON report saved to {self.report_file.with_suffix('.json')}")
        
        # Print summary
        logger.info(f"Test execution summary: {self.results['stats']['passed']} passed, "
                   f"{self.results['stats']['failed']} failed, "
                   f"out of {total} test files.")
        if self.results["stats"]["passed"] > 0:
            logger.info(f"Average execution time: {self.results['stats']['avg_execution_time']}s")
    
    def _generate_markdown_report(self) -> str:
        """Generate a markdown report."""
        total = self.results["stats"]["total"]
        passed = self.results["stats"]["passed"]
        failed = self.results["stats"]["failed"]
        passed_percent = self.results["stats"]["passed_percent"]
        
        report = [
            "# HuggingFace Model Test Execution Report",
            f"\nGenerated on: {self.results['timestamp']}",
            
            "\n## Summary",
            f"- **Total tests executed**: {total}",
            f"- **Passed**: {passed} ({passed_percent}%)",
            f"- **Failed**: {failed} ({self.results['stats']['failed_percent']}%)",
        ]
        
        if passed > 0:
            report.append(f"- **Average execution time**: {self.results['stats']['avg_execution_time']}s")
        
        report.extend([
            "\n## Results by Architecture",
            "| Architecture | Total | Passed | Failed | Pass Rate |",
            "|-------------|-------|--------|--------|-----------|",
        ])
        
        for arch, stats in sorted(self.results["stats"]["by_architecture"].items()):
            arch_total = stats["total"]
            if arch_total == 0:
                continue
                
            arch_passed = stats["passed"]
            arch_failed = stats["failed"]
            arch_pass_rate = round(arch_passed / arch_total * 100, 2) if arch_total > 0 else 0
            report.append(f"| {arch} | {arch_total} | {arch_passed} | {arch_failed} | {arch_pass_rate}% |")
        
        if self.results["stats"]["by_error_type"]:
            report.append("\n## Error Types")
            report.append("| Error Type | Count |")
            report.append("|------------|-------|")
            
            for error_type, count in sorted(self.results["stats"]["by_error_type"].items()):
                # Format the error type for display
                display_error = error_type.replace("_", " ").capitalize()
                report.append(f"| {display_error} | {count} |")
        
        if passed > 0:
            report.append("\n## Passed Tests")
            report.append("| Model | Architecture | Execution Time (s) |")
            report.append("|-------|--------------|-------------------|")
            
            # Sort by execution time (fastest first)
            sorted_passed = sorted(self.results["passed"], key=lambda x: x.get("execution_time", float("inf")))
            
            for success in sorted_passed[:10]:  # Show top 10 fastest
                model_name = success["model_name"]
                arch = success["architecture"]
                exec_time = success.get("execution_time", "N/A")
                report.append(f"| {model_name} | {arch} | {exec_time} |")
                
            if len(sorted_passed) > 10:
                report.append(f"... and {len(sorted_passed) - 10} more. See JSON report for details.")
        
        if failed > 0:
            report.append("\n## Failed Tests")
            for i, failure in enumerate(self.results["failed"]):
                if i > 9:  # Limit to 10 failures in the report
                    report.append(f"\n... and {len(self.results['failed']) - 10} more failures. See JSON report for details.")
                    break
                
                report.append(f"\n### {i+1}. {failure['model_name']}")
                report.append(f"- **Model ID**: {failure['model_id']}")
                report.append(f"- **Architecture**: {failure['architecture']}")
                report.append("- **Errors**:")
                for error in failure["errors"]:
                    report.append(f"  - {error}")
        
        return "\n".join(report)

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run HuggingFace model tests")
    parser.add_argument("--directory", "-d", type=str, default="fixed_tests",
                        help="Directory containing test files")
    parser.add_argument("--max-tests", "-m", type=int, default=20,
                        help="Maximum number of tests to run")
    parser.add_argument("--report", "-r", type=str, default="test_execution_report.md",
                        help="Path to output report file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Resolve directory path
    directory = args.directory
    if not os.path.isabs(directory):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    
    # Create test runner and run tests
    runner = TestRunner(directory, args.max_tests, args.report, args.verbose)
    results = runner.run_all_tests()
    
    # Set exit code based on test results
    if results["stats"]["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()