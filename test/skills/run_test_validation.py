#!/usr/bin/env python3
"""
Run Test Validation

This script executes a sample of HuggingFace model test files to verify they work correctly.
It focuses on basic functional testing, with an option to run tests with mocked or real models.

Usage:
    python run_test_validation.py --directory TESTS_DIR [--max-tests N] [--use-mocks] [--verbose]
"""

import os
import sys
import random
import subprocess
import argparse
import logging
from pathlib import Path
import datetime
import time
from typing import Dict, List, Optional, Tuple, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestRunner:
    """Executes HuggingFace model test files to verify functionality."""
    
    def __init__(self, directory: str, max_tests: int = 10, use_mocks: bool = True, verbose: bool = False):
        """Initialize the test runner.
        
        Args:
            directory: Directory containing test files
            max_tests: Maximum number of tests to run (0 for all)
            use_mocks: Whether to use mock objects instead of real models
            verbose: Whether to print verbose output
        """
        self.directory = Path(directory)
        self.max_tests = max_tests
        self.use_mocks = use_mocks
        self.verbose = verbose
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "tested_files": 0,
            "success_count": 0,
            "failure_count": 0,
            "by_architecture": {},
            "test_results": {}
        }
    
    def run(self):
        """Run a sample of test files."""
        # Find all test files
        test_files = list(self.directory.glob("test_hf_*.py"))
        self.stats["total_files"] = len(test_files)
        
        logger.info(f"Found {len(test_files)} test files")
        
        # Determine which files to test
        files_to_test = self._select_test_files(test_files)
        self.stats["tested_files"] = len(files_to_test)
        
        logger.info(f"Selected {len(files_to_test)} files to test")
        
        # Run each test file
        for i, file_path in enumerate(files_to_test):
            logger.info(f"[{i+1}/{len(files_to_test)}] Testing {file_path.name}...")
            success, result = self._run_test_file(file_path)
            
            # Update statistics
            self.stats["test_results"][str(file_path)] = result
            
            if success:
                self.stats["success_count"] += 1
                logger.info(f"✅ Test passed: {file_path.name}")
            else:
                self.stats["failure_count"] += 1
                logger.error(f"❌ Test failed: {file_path.name}")
        
        # Generate report
        self._generate_report()
        
        return self.stats
    
    def _select_test_files(self, test_files: List[Path]) -> List[Path]:
        """Select a sample of test files to run."""
        # If max_tests is 0, run all tests
        if self.max_tests <= 0:
            return test_files
        
        # If max_tests is greater than the number of files, run all tests
        if self.max_tests >= len(test_files):
            return test_files
        
        # Group files by architecture for balanced selection
        files_by_architecture = {}
        for file_path in test_files:
            # Extract model name from file name
            model_name = file_path.stem.replace("test_hf_", "")
            
            # Determine architecture (approximate)
            architecture = self._get_model_architecture(model_name)
            
            # Add to architecture group
            if architecture not in files_by_architecture:
                files_by_architecture[architecture] = []
                
                # Initialize architecture stats
                if architecture not in self.stats["by_architecture"]:
                    self.stats["by_architecture"][architecture] = {
                        "total": 0,
                        "tested": 0,
                        "success": 0,
                        "failure": 0
                    }
            
            files_by_architecture[architecture].append(file_path)
            self.stats["by_architecture"][architecture]["total"] += 1
        
        # Select files proportionally from each architecture
        selected_files = []
        remaining = self.max_tests
        architecture_counts = {}
        
        # First pass: calculate proportional counts
        total_files = len(test_files)
        for arch, files in files_by_architecture.items():
            arch_count = max(1, int(self.max_tests * len(files) / total_files))
            architecture_counts[arch] = min(arch_count, len(files))
            remaining -= architecture_counts[arch]
        
        # Second pass: distribute remaining tests
        if remaining > 0:
            for arch in sorted(architecture_counts.keys()):
                if architecture_counts[arch] < len(files_by_architecture[arch]):
                    architecture_counts[arch] += 1
                    remaining -= 1
                    if remaining == 0:
                        break
        
        # Select random files from each architecture
        for arch, count in architecture_counts.items():
            arch_files = random.sample(files_by_architecture[arch], count)
            selected_files.extend(arch_files)
            self.stats["by_architecture"][arch]["tested"] += count
        
        return selected_files
    
    def _run_test_file(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Run a single test file and return the results."""
        start_time = time.time()
        
        # Set up environment variables for mocking
        env = os.environ.copy()
        if self.use_mocks:
            env["MOCK_TORCH"] = "True"
            env["MOCK_TRANSFORMERS"] = "True"
            env["MOCK_TOKENIZERS"] = "True"
            env["MOCK_SENTENCEPIECE"] = "True"
        
        try:
            # Run the test file with subprocess
            result = subprocess.run(
                [sys.executable, str(file_path)],
                capture_output=True,
                text=True,
                env=env,
                timeout=60  # 1 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            # Check the return code
            success = result.returncode == 0
            
            # Extract architecture
            model_name = file_path.stem.replace("test_hf_", "")
            architecture = self._get_model_architecture(model_name)
            
            # Update architecture stats
            if success:
                self.stats["by_architecture"][architecture]["success"] += 1
            else:
                self.stats["by_architecture"][architecture]["failure"] += 1
            
            # Prepare result dictionary
            test_result = {
                "success": success,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "architecture": architecture,
                "model_name": model_name,
                "stdout": result.stdout[:1000] if self.verbose else "",  # Limit output size
                "stderr": result.stderr[:1000] if self.verbose else "",  # Limit output size
            }
            
            if self.verbose:
                if success:
                    logger.info(f"Test completed in {execution_time:.2f} seconds")
                else:
                    logger.error(f"Test failed with return code {result.returncode}")
                    logger.error(f"Stderr: {result.stderr[:500]}...")
            
            return success, test_result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            
            # Extract architecture
            model_name = file_path.stem.replace("test_hf_", "")
            architecture = self._get_model_architecture(model_name)
            
            # Update architecture stats
            self.stats["by_architecture"][architecture]["failure"] += 1
            
            logger.error(f"Test timed out after {execution_time:.2f} seconds")
            
            return False, {
                "success": False,
                "execution_time": execution_time,
                "return_code": None,
                "architecture": architecture,
                "model_name": model_name,
                "stdout": "",
                "stderr": "Test timed out after 60 seconds",
                "error": "Timeout"
            }
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Extract architecture
            model_name = file_path.stem.replace("test_hf_", "")
            architecture = self._get_model_architecture(model_name)
            
            # Update architecture stats
            self.stats["by_architecture"][architecture]["failure"] += 1
            
            logger.error(f"Error running test: {str(e)}")
            
            return False, {
                "success": False,
                "execution_time": execution_time,
                "return_code": None,
                "architecture": architecture,
                "model_name": model_name,
                "stdout": "",
                "stderr": str(e),
                "error": "Exception"
            }
    
    def _get_model_architecture(self, model_name: str) -> str:
        """Determine the model architecture from the model name (approximate)."""
        model_name_lower = model_name.lower()
        
        # Define simple architecture detection rules
        if "bert" in model_name_lower or "roberta" in model_name_lower or "electra" in model_name_lower:
            return "encoder-only"
        elif "gpt" in model_name_lower or "llama" in model_name_lower or "bloom" in model_name_lower:
            return "decoder-only"
        elif "t5" in model_name_lower or "bart" in model_name_lower or "pegasus" in model_name_lower:
            return "encoder-decoder"
        elif "vit" in model_name_lower or "swin" in model_name_lower or "deit" in model_name_lower:
            return "vision"
        elif "clip" in model_name_lower or "blip" in model_name_lower:
            return "vision-text"
        elif "wav2vec" in model_name_lower or "whisper" in model_name_lower:
            return "speech"
        elif "llava" in model_name_lower:
            return "multimodal"
        
        return "unknown"
    
    def _generate_report(self):
        """Generate a test execution report."""
        # Create reports directory if needed
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Create report file path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(reports_dir, f"test_execution_report_{timestamp}.md")
        
        # Calculate overall pass rate
        tested_files = self.stats["tested_files"]
        success_count = self.stats["success_count"]
        pass_rate = success_count / tested_files * 100 if tested_files > 0 else 0
        
        # Generate markdown report
        report = [
            "# HuggingFace Model Test Execution Report",
            "",
            f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Mode:** {'Mock Objects' if self.use_mocks else 'Real Models'}",
            "",
            "## Summary",
            "",
            f"- **Total files available:** {self.stats['total_files']}",
            f"- **Files tested:** {tested_files} ({tested_files/self.stats['total_files']*100:.1f}% sample)",
            f"- **Tests passed:** {success_count} ({pass_rate:.1f}%)",
            f"- **Tests failed:** {self.stats['failure_count']} ({100-pass_rate:.1f}%)",
            "",
            "## Results by Architecture",
            ""
        ]
        
        # Add architecture-specific stats
        for arch, stats in sorted(self.stats["by_architecture"].items()):
            if stats["tested"] > 0:
                arch_pass_rate = stats["success"] / stats["tested"] * 100
                report.extend([
                    f"### {arch.capitalize()} Models",
                    "",
                    f"- **Files tested:** {stats['tested']} (of {stats['total']} available)",
                    f"- **Tests passed:** {stats['success']} ({arch_pass_rate:.1f}%)",
                    f"- **Tests failed:** {stats['failure']} ({100-arch_pass_rate:.1f}%)",
                    ""
                ])
        
        # Add detailed results for failed tests
        failed_tests = [
            (path, result) for path, result in self.stats["test_results"].items()
            if not result["success"]
        ]
        
        if failed_tests:
            report.extend([
                "## Failed Tests",
                "",
                "The following tests failed:",
                ""
            ])
            
            for path, result in failed_tests:
                file_name = os.path.basename(path)
                error_message = result.get("stderr", "No error message")[:200]  # Limit size
                report.extend([
                    f"### {file_name}",
                    "",
                    f"- **Architecture:** {result['architecture']}",
                    f"- **Execution time:** {result['execution_time']:.2f} seconds",
                    f"- **Error type:** {result.get('error', 'Execution failure')}",
                    "",
                    "```",
                    f"{error_message}...",
                    "```",
                    ""
                ])
        
        # Write report to file
        with open(report_file, 'w') as f:
            f.write("\n".join(report))
        
        logger.info(f"Test execution report written to {report_file}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run functional tests on HuggingFace model test files")
    parser.add_argument("--directory", "-d", type=str, default="fixed_tests",
                        help="Directory containing test files")
    parser.add_argument("--max-tests", "-m", type=int, default=10,
                        help="Maximum number of tests to run (0 for all)")
    parser.add_argument("--use-mocks", "-M", action="store_true", default=True,
                        help="Use mock objects instead of real models")
    parser.add_argument("--real-models", "-r", action="store_true",
                        help="Use real models instead of mocks (overrides --use-mocks)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print verbose output")
    
    args = parser.parse_args()
    
    # Handle real-models flag (inverts use-mocks)
    if args.real_models:
        args.use_mocks = False
    
    # Resolve directory path
    directory = args.directory
    if not os.path.isabs(directory):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    
    # Create and run the test runner
    runner = TestRunner(directory, args.max_tests, args.use_mocks, args.verbose)
    stats = runner.run()
    
    # Print summary
    tested_files = stats["tested_files"]
    success_count = stats["success_count"]
    pass_rate = success_count / tested_files * 100 if tested_files > 0 else 0
    
    print("\nTEST EXECUTION SUMMARY")
    print("="*50)
    print(f"Mode: {'Mock Objects' if args.use_mocks else 'Real Models'}")
    print(f"Files tested: {tested_files} (of {stats['total_files']} available)")
    print(f"Tests passed: {success_count} ({pass_rate:.1f}%)")
    print(f"Tests failed: {stats['failure_count']} ({100-pass_rate:.1f}%)")
    
    # Determine exit code based on pass rate
    # For CI/CD integration: non-zero exit code if pass rate is below threshold
    pass_threshold = 80.0  # 80% pass rate threshold
    return 0 if pass_rate >= pass_threshold else 1

if __name__ == "__main__":
    sys.exit(main())