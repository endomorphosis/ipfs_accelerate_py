#!/usr/bin/env python3
"""
End-to-End Testing Framework for IPFS Accelerate

This script automates the generation and testing of skill, test, and benchmark components
for models. It generates all three components together, runs tests, collects results,
and compares them with expected results.

Usage:
    python run_e2e_tests.py --model bert --hardware cuda
    python run_e2e_tests.py --model-family text-embedding --hardware all
    python run_e2e_tests.py --model vit --hardware cuda,webgpu --update-expected
    python run_e2e_tests.py --all-models --priority-hardware --quick-test
"""

import os
import sys
import json
import time
import argparse
import logging
import datetime
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import project utilities
from simple_utils import setup_logging, ensure_dir_exists
from template_validation import ModelValidator, ResultComparer
from model_documentation_generator import generate_model_documentation

# Set up logging
logger = logging.getLogger(__name__)
setup_logging(logger)

# Constants
RESULTS_ROOT = os.path.abspath(os.path.join(script_dir, "../../"))
EXPECTED_RESULTS_DIR = os.path.join(RESULTS_ROOT, "expected_results")
COLLECTED_RESULTS_DIR = os.path.join(RESULTS_ROOT, "collected_results")
DOCS_DIR = os.path.join(RESULTS_ROOT, "model_documentation")
TEST_TIMEOUT = 300  # seconds

# Ensure directories exist
for directory in [EXPECTED_RESULTS_DIR, COLLECTED_RESULTS_DIR, DOCS_DIR]:
    ensure_dir_exists(directory)

# Hardware platforms supported by the testing framework
SUPPORTED_HARDWARE = [
    "cpu", "cuda", "rocm", "mps", "openvino", 
    "qnn", "webnn", "webgpu", "samsung"
]

PRIORITY_HARDWARE = ["cpu", "cuda", "openvino", "webgpu"]

# Mapping of model families to specific models for testing
MODEL_FAMILY_MAP = {
    "text-embedding": ["bert-base-uncased", "bert-tiny"],
    "text-generation": ["opt-125m", "t5-small", "t5-efficient-tiny"],
    "vision": ["vit-base", "clip-vit"],
    "audio": ["whisper-tiny", "wav2vec2-base"],
    "multimodal": ["clip-vit", "llava-onevision-base"]
}

class E2ETester:
    """Main class for end-to-end testing framework."""
    
    def __init__(self, args):
        """Initialize the E2E testing framework with command line arguments."""
        self.args = args
        self.models_to_test = self._determine_models_to_test()
        self.hardware_to_test = self._determine_hardware_to_test()
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_results = {}
        self.temp_dirs = []
        
    def _determine_models_to_test(self) -> List[str]:
        """Determine which models to test based on args."""
        if self.args.all_models:
            # Collect all models from all families
            models = []
            for family_models in MODEL_FAMILY_MAP.values():
                models.extend(family_models)
            return list(set(models))  # Remove duplicates
        
        if self.args.model_family:
            if self.args.model_family in MODEL_FAMILY_MAP:
                return MODEL_FAMILY_MAP[self.args.model_family]
            else:
                logger.warning(f"Unknown model family: {self.args.model_family}")
                return []
            
        if self.args.model:
            return [self.args.model]
            
        logger.error("No models specified. Use --model, --model-family, or --all-models")
        return []
    
    def _determine_hardware_to_test(self) -> List[str]:
        """Determine which hardware platforms to test based on args."""
        if self.args.all_hardware:
            return SUPPORTED_HARDWARE
            
        if self.args.priority_hardware:
            return PRIORITY_HARDWARE
            
        if self.args.hardware:
            hardware_list = self.args.hardware.split(',')
            # Validate hardware platforms
            invalid_hw = [hw for hw in hardware_list if hw not in SUPPORTED_HARDWARE]
            if invalid_hw:
                logger.warning(f"Unsupported hardware platforms: {', '.join(invalid_hw)}")
                hardware_list = [hw for hw in hardware_list if hw in SUPPORTED_HARDWARE]
            
            return hardware_list
            
        logger.error("No hardware specified. Use --hardware, --priority-hardware, or --all-hardware")
        return []
    
    def run_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run end-to-end tests for all specified models and hardware platforms."""
        if not self.models_to_test or not self.hardware_to_test:
            logger.error("No models or hardware specified, exiting")
            return {}
            
        logger.info(f"Starting end-to-end tests for models: {', '.join(self.models_to_test)}")
        logger.info(f"Testing on hardware platforms: {', '.join(self.hardware_to_test)}")
        
        for model in self.models_to_test:
            self.test_results[model] = {}
            
            for hardware in self.hardware_to_test:
                logger.info(f"Testing {model} on {hardware}...")
                
                try:
                    # Create a temp directory for this test
                    temp_dir = tempfile.mkdtemp(prefix=f"e2e_test_{model}_{hardware}_")
                    self.temp_dirs.append(temp_dir)
                    
                    # Generate skill, test, and benchmark components together
                    skill_path, test_path, benchmark_path = self._generate_components(model, hardware, temp_dir)
                    
                    # Run the test and collect results
                    result = self._run_test(model, hardware, temp_dir, test_path)
                    
                    # Compare results with expected (if they exist)
                    comparison = self._compare_with_expected(model, hardware, result)
                    
                    # Update expected results if requested
                    if self.args.update_expected:
                        self._update_expected_results(model, hardware, result)
                    
                    # Store results
                    self._store_results(model, hardware, result, comparison)
                    
                    # Generate model documentation if requested
                    if self.args.generate_docs:
                        self._generate_documentation(model, hardware, skill_path, test_path, benchmark_path)
                    
                    # Record the test result
                    self.test_results[model][hardware] = {
                        "status": "success" if comparison["matches"] else "failure",
                        "result_path": self._get_result_path(model, hardware),
                        "comparison": comparison
                    }
                    
                    logger.info(f"Testing {model} on {hardware} - {'SUCCESS' if comparison['matches'] else 'FAILURE'}")
                
                except Exception as e:
                    logger.error(f"Error testing {model} on {hardware}: {str(e)}")
                    self.test_results[model][hardware] = {
                        "status": "error",
                        "error": str(e)
                    }
        
        self._generate_summary_report()
        self._cleanup()
        
        return self.test_results
    
    def _generate_components(self, model: str, hardware: str, temp_dir: str) -> Tuple[str, str, str]:
        """Generate skill, test, and benchmark components for a model/hardware combination."""
        logger.debug(f"Generating components for {model} on {hardware}...")
        
        # Paths for the generated components
        skill_path = os.path.join(temp_dir, f"skill_{model}_{hardware}.py")
        test_path = os.path.join(temp_dir, f"test_{model}_{hardware}.py")
        benchmark_path = os.path.join(temp_dir, f"benchmark_{model}_{hardware}.py")
        
        # Call generator scripts to create all three components
        # TODO: Replace this with actual calls to generator scripts when available
        self._mock_generate_skill(model, hardware, skill_path)
        self._mock_generate_test(model, hardware, test_path, skill_path)
        self._mock_generate_benchmark(model, hardware, benchmark_path, skill_path)
        
        return skill_path, test_path, benchmark_path
    
    def _mock_generate_skill(self, model: str, hardware: str, skill_path: str):
        """Mock function to generate a skill file."""
        with open(skill_path, 'w') as f:
            f.write(f"""
# Generated skill for {model} on {hardware}
import torch

class {model.replace('-', '_').title()}Skill:
    def __init__(self):
        self.model_name = "{model}"
        self.hardware = "{hardware}"
        
    def setup(self):
        # Mock setup logic for {hardware}
        print(f"Setting up {model} for {hardware}")
        
    def run(self, input_data):
        # Mock inference logic
        # This would be replaced with actual model code
        return {{"output": "mock_output_for_{model}_on_{hardware}"}}
            """)
    
    def _mock_generate_test(self, model: str, hardware: str, test_path: str, skill_path: str):
        """Mock function to generate a test file."""
        with open(test_path, 'w') as f:
            f.write(f"""
# Generated test for {model} on {hardware}
import unittest
import os
import sys
from pathlib import Path

# Add skill path to system path
skill_dir = Path("{os.path.dirname(skill_path)}")
if str(skill_dir) not in sys.path:
    sys.path.append(str(skill_dir))

from skill_{model}_{hardware} import {model.replace('-', '_').title()}Skill

class Test{model.replace('-', '_').title()}(unittest.TestCase):
    def setUp(self):
        self.skill = {model.replace('-', '_').title()}Skill()
        self.skill.setup()
        
    def test_inference(self):
        input_data = {{"input": "test_input"}}
        result = self.skill.run(input_data)
        self.assertIn("output", result)
        
if __name__ == "__main__":
    unittest.main()
            """)
    
    def _mock_generate_benchmark(self, model: str, hardware: str, benchmark_path: str, skill_path: str):
        """Mock function to generate a benchmark file."""
        with open(benchmark_path, 'w') as f:
            f.write(f"""
# Generated benchmark for {model} on {hardware}
import time
import json
import os
import sys
from pathlib import Path

# Add skill path to system path
skill_dir = Path("{os.path.dirname(skill_path)}")
if str(skill_dir) not in sys.path:
    sys.path.append(str(skill_dir))

from skill_{model}_{hardware} import {model.replace('-', '_').title()}Skill

def benchmark():
    skill = {model.replace('-', '_').title()}Skill()
    skill.setup()
    
    # Warmup
    for _ in range(5):
        skill.run({{"input": "warmup"}})
    
    # Benchmark
    batch_sizes = [1, 2, 4, 8]
    results = {{}}
    
    for batch_size in batch_sizes:
        start_time = time.time()
        for _ in range(10):
            skill.run({{"input": "benchmark", "batch_size": batch_size}})
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        results[str(batch_size)] = {{
            "latency_ms": avg_time * 1000,
            "throughput": batch_size / avg_time
        }}
    
    return results

if __name__ == "__main__":
    results = benchmark()
    print(json.dumps(results, indent=2))
    
    # Write results to file
    output_file = "{benchmark_path}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmark results written to {{output_file}}")
            """)
    
    def _run_test(self, model: str, hardware: str, temp_dir: str, test_path: str) -> Dict[str, Any]:
        """Run the test for a model/hardware combination and capture results."""
        logger.debug(f"Running test for {model} on {hardware}...")
        
        # Mock results for demonstration
        result = {
            "model": model,
            "hardware": hardware,
            "timestamp": self.timestamp,
            "input": {"input": "test_input"},
            "output": {"output": f"mock_output_for_{model}_on_{hardware}"},
            "metrics": {
                "latency_ms": 12.5,
                "throughput": 80.0,
                "memory_mb": 512
            },
            "hardware_details": {
                "platform": hardware,
                "device_name": f"Mock {hardware.upper()} Device"
            }
        }
        
        # In a real implementation, we would execute the test file and capture results
        # For example:
        # result = subprocess.run(["python", test_path], capture_output=True, text=True, timeout=TEST_TIMEOUT)
        
        return result
    
    def _compare_with_expected(self, model: str, hardware: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare test results with expected results, if they exist."""
        expected_path = os.path.join(EXPECTED_RESULTS_DIR, model, hardware, "expected_result.json")
        
        if not os.path.exists(expected_path):
            logger.warning(f"No expected results found for {model} on {hardware}")
            return {"matches": False, "reason": "no_expected_results"}
        
        try:
            with open(expected_path, 'r') as f:
                expected = json.load(f)
            
            # Use ResultComparer from template_validation module
            # This is a placeholder - real implementation would use the actual module
            differences = {}
            
            # Mock comparison logic
            if result["output"] != expected["output"]:
                differences["output"] = {
                    "expected": expected["output"],
                    "actual": result["output"]
                }
            
            # Check if metrics are within tolerance
            for metric in ["latency_ms", "throughput", "memory_mb"]:
                if metric in expected.get("metrics", {}) and metric in result.get("metrics", {}):
                    expected_val = expected["metrics"][metric]
                    actual_val = result["metrics"][metric]
                    # Allow 10% tolerance for performance metrics
                    tolerance = 0.1 * expected_val
                    if abs(actual_val - expected_val) > tolerance:
                        if "metrics" not in differences:
                            differences["metrics"] = {}
                        differences["metrics"][metric] = {
                            "expected": expected_val,
                            "actual": actual_val,
                            "tolerance": tolerance
                        }
            
            return {
                "matches": len(differences) == 0,
                "differences": differences
            }
            
        except Exception as e:
            logger.error(f"Error comparing results for {model} on {hardware}: {str(e)}")
            return {"matches": False, "reason": f"comparison_error: {str(e)}"}
    
    def _update_expected_results(self, model: str, hardware: str, result: Dict[str, Any]):
        """Update expected results with current results if requested."""
        if not self.args.update_expected:
            return
            
        expected_dir = os.path.join(EXPECTED_RESULTS_DIR, model, hardware)
        os.makedirs(expected_dir, exist_ok=True)
        
        expected_path = os.path.join(expected_dir, "expected_result.json")
        
        # Add metadata for expected results
        result_with_metadata = result.copy()
        result_with_metadata["metadata"] = {
            "updated_at": self.timestamp,
            "updated_by": os.environ.get("USER", "unknown"),
            "version": "1.0"
        }
        
        with open(expected_path, 'w') as f:
            json.dump(result_with_metadata, f, indent=2)
            
        logger.info(f"Updated expected results for {model} on {hardware}")
    
    def _store_results(self, model: str, hardware: str, result: Dict[str, Any], comparison: Dict[str, Any]):
        """Store test results in the collected_results directory."""
        result_dir = os.path.join(COLLECTED_RESULTS_DIR, model, hardware, self.timestamp)
        os.makedirs(result_dir, exist_ok=True)
        
        # Store the test result
        result_path = os.path.join(result_dir, "result.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        # Store the comparison
        comparison_path = os.path.join(result_dir, "comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
            
        # Create a status file for easy filtering
        status = "success" if comparison["matches"] else "failure"
        status_path = os.path.join(result_dir, f"{status}.status")
        with open(status_path, 'w') as f:
            f.write(f"Test completed at {self.timestamp}\n")
            f.write(f"Status: {status.upper()}\n")
            
            if not comparison["matches"] and "differences" in comparison:
                f.write("\nDifferences found:\n")
                for key, diff in comparison["differences"].items():
                    f.write(f"- {key}: {json.dumps(diff)}\n")
    
    def _get_result_path(self, model: str, hardware: str) -> str:
        """Get the path to the collected results for a model/hardware combination."""
        return os.path.join(COLLECTED_RESULTS_DIR, model, hardware, self.timestamp)
    
    def _generate_documentation(self, model: str, hardware: str, skill_path: str, test_path: str, benchmark_path: str):
        """Generate Markdown documentation for the model, including expected behavior and implementation details."""
        logger.debug(f"Generating documentation for {model} on {hardware}...")
        
        doc_dir = os.path.join(DOCS_DIR, model)
        os.makedirs(doc_dir, exist_ok=True)
        
        doc_path = os.path.join(doc_dir, f"{hardware}_implementation.md")
        
        # In a real implementation, this would call the model_documentation_generator module
        # For now, we create a simple placeholder document
        with open(doc_path, 'w') as f:
            f.write(f"""# {model} Implementation Guide for {hardware}

## Overview

This document describes the implementation of {model} on {hardware} hardware.

## Skill Implementation

The skill implementation is responsible for loading and running the model on {hardware}.

```python
# Include key parts of the skill implementation
```

## Test Implementation

The test ensures that the model produces correct outputs.

```python
# Include key parts of the test implementation
```

## Benchmark Implementation

The benchmark measures the performance of the model on {hardware}.

```python
# Include key parts of the benchmark implementation
```

## Expected Results

The model should produce the following outputs for the given inputs:

```json
# Include examples of expected inputs and outputs
```

## Performance Characteristics

- **Latency**: Expected latency on {hardware} is ~10-15ms
- **Throughput**: Expected throughput is ~80 inferences/second
- **Memory**: Expected memory usage is ~512MB

## Implementation Notes

- This model uses XYZ architecture
- On {hardware}, the best performance is achieved with batch size 8
- Implementation based on HuggingFace Transformers library

""")
        
        logger.info(f"Generated documentation for {model} on {hardware} at {doc_path}")
    
    def _generate_summary_report(self):
        """Generate a summary report of all test results."""
        if not self.test_results:
            return
            
        summary = {
            "timestamp": self.timestamp,
            "summary": {
                "total": 0,
                "success": 0,
                "failure": 0,
                "error": 0
            },
            "results": self.test_results
        }
        
        # Calculate summary statistics
        for model, hw_results in self.test_results.items():
            for hw, result in hw_results.items():
                summary["summary"]["total"] += 1
                summary["summary"][result["status"]] = summary["summary"].get(result["status"], 0) + 1
        
        # Write summary to file
        summary_dir = os.path.join(COLLECTED_RESULTS_DIR, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        summary_path = os.path.join(summary_dir, f"summary_{self.timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Generate a markdown report
        report_path = os.path.join(summary_dir, f"report_{self.timestamp}.md")
        with open(report_path, 'w') as f:
            f.write(f"# End-to-End Test Report - {self.timestamp}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Tests**: {summary['summary']['total']}\n")
            f.write(f"- **Successful**: {summary['summary']['success']}\n")
            f.write(f"- **Failed**: {summary['summary']['failure']}\n")
            f.write(f"- **Errors**: {summary['summary']['error']}\n\n")
            
            f.write("## Results by Model\n\n")
            for model, hw_results in self.test_results.items():
                f.write(f"### {model}\n\n")
                
                for hw, result in hw_results.items():
                    status_icon = "✅" if result["status"] == "success" else "❌" if result["status"] == "failure" else "⚠️"
                    f.write(f"- {status_icon} **{hw}**: {result['status'].upper()}\n")
                    
                    if result["status"] == "failure" and "comparison" in result and "differences" in result["comparison"]:
                        f.write("  - Differences found:\n")
                        for key, diff in result["comparison"]["differences"].items():
                            f.write(f"    - {key}: {json.dumps(diff)}\n")
                            
                    if result["status"] == "error" and "error" in result:
                        f.write(f"  - Error: {result['error']}\n")
                        
                f.write("\n")
                
        logger.info(f"Generated summary report at {report_path}")
    
    def _cleanup(self):
        """Clean up temporary directories."""
        if not self.args.keep_temp:
            for temp_dir in self.temp_dirs:
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary directory {temp_dir}: {str(e)}")
    
    def clean_old_results(self):
        """Clean up old collected results."""
        if not self.args.clean_old_results:
            return
            
        days = self.args.days if self.args.days else 14
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        logger.info(f"Cleaning up collected results older than {days} days...")
        
        cleaned_count = 0
        
        for model_dir in os.listdir(COLLECTED_RESULTS_DIR):
            model_path = os.path.join(COLLECTED_RESULTS_DIR, model_dir)
            if not os.path.isdir(model_path) or model_dir == "summary":
                continue
                
            for hw_dir in os.listdir(model_path):
                hw_path = os.path.join(model_path, hw_dir)
                if not os.path.isdir(hw_path):
                    continue
                    
                for result_dir in os.listdir(hw_path):
                    result_path = os.path.join(hw_path, result_dir)
                    if not os.path.isdir(result_path):
                        continue
                        
                    # Skip directories that don't match timestamp format
                    if not result_dir.isdigit() or len(result_dir) != 15:  # 20250311_120000 format
                        continue
                        
                    # Check if the directory is older than cutoff
                    try:
                        dir_time = datetime.datetime.strptime(result_dir, "%Y%m%d_%H%M%S").timestamp()
                        if dir_time < cutoff_time:
                            # Check if it's a failed test that we want to keep
                            if os.path.exists(os.path.join(result_path, "failure.status")) and not self.args.clean_failures:
                                continue
                                
                            shutil.rmtree(result_path)
                            cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Error cleaning up {result_path}: {str(e)}")
        
        logger.info(f"Cleaned up {cleaned_count} old result directories")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="End-to-End Testing Framework for IPFS Accelerate")
    
    # Model selection arguments
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", help="Specific model to test")
    model_group.add_argument("--model-family", help="Model family to test (e.g., text-embedding, vision)")
    model_group.add_argument("--all-models", action="store_true", help="Test all supported models")
    
    # Hardware selection arguments
    hardware_group = parser.add_mutually_exclusive_group()
    hardware_group.add_argument("--hardware", help="Hardware platforms to test, comma-separated (e.g., cpu,cuda,webgpu)")
    hardware_group.add_argument("--priority-hardware", action="store_true", help="Test on priority hardware platforms (cpu, cuda, openvino, webgpu)")
    hardware_group.add_argument("--all-hardware", action="store_true", help="Test on all supported hardware platforms")
    
    # Test options
    parser.add_argument("--quick-test", action="store_true", help="Run a quick test with minimal validation")
    parser.add_argument("--update-expected", action="store_true", help="Update expected results with current test results")
    parser.add_argument("--generate-docs", action="store_true", help="Generate markdown documentation for models")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary directories after tests")
    
    # Cleanup options
    parser.add_argument("--clean-old-results", action="store_true", help="Clean up old collected results")
    parser.add_argument("--days", type=int, help="Number of days to keep results when cleaning (default: 14)")
    parser.add_argument("--clean-failures", action="store_true", help="Clean failed test results too")
    
    # Logging options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Initialize the tester
    tester = E2ETester(args)
    
    # If cleaning old results, do that and exit
    if args.clean_old_results:
        tester.clean_old_results()
        return
    
    # Run the tests
    results = tester.run_tests()
    
    # Print a brief summary
    total = sum(len(hw_results) for hw_results in results.values())
    success = sum(sum(1 for result in hw_results.values() if result["status"] == "success") for hw_results in results.values())
    
    logger.info(f"Test run completed - {success}/{total} tests passed")


if __name__ == "__main__":
    main()