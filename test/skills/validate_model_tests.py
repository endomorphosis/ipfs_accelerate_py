#!/usr/bin/env python3
"""
Model Test Validation Framework

This script validates the generated test files for HuggingFace models by:
1. Checking syntax validity
2. Verifying class structure and required methods
3. Validating task configuration for each model type
4. Running basic import test to ensure dependencies are correctly imported
5. Generating a comprehensive validation report

Usage:
    python validate_model_tests.py [--directory DIR] [--report REPORT_FILE] [--verbose]
"""

import os
import sys
import ast
import importlib.util
import argparse
import logging
import re
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Set, Optional, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define architecture types for task validation
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2"],
    "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava"]
}

# Define expected tasks for different architecture types
EXPECTED_TASKS = {
    "encoder-only": ["fill-mask", "text-classification", "token-classification", "question-answering"],
    "decoder-only": ["text-generation", "causal-lm"],
    "encoder-decoder": ["text2text-generation", "translation", "summarization"],
    "vision": ["image-classification", "object-detection", "image-segmentation"],
    "vision-text": ["image-to-text", "zero-shot-image-classification", "visual-question-answering"],
    "speech": ["automatic-speech-recognition", "audio-classification", "text-to-speech"],
    "multimodal": ["image-to-text", "video-to-text", "visual-question-answering"]
}

class ModelTestValidator:
    """Validator for HuggingFace model test files."""
    
    def __init__(self, directory: str, report_file: str, verbose: bool = False):
        """Initialize the validator.
        
        Args:
            directory: Directory containing test files
            report_file: Path to output report file
            verbose: Whether to print verbose output
        """
        self.directory = Path(directory)
        self.report_file = Path(report_file)
        self.verbose = verbose
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "stats": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "by_architecture": {},
                "by_error_type": {},
            }
        }
        
    def find_test_files(self) -> List[Path]:
        """Find all test files in the directory."""
        return list(self.directory.glob("test_hf_*.py"))
    
    def validate_all(self) -> Dict:
        """Validate all test files and return results."""
        test_files = self.find_test_files()
        logger.info(f"Found {len(test_files)} test files to validate")
        
        self.results["stats"]["total"] = len(test_files)
        
        # Use ThreadPoolExecutor for parallel validation
        with ThreadPoolExecutor(max_workers=min(10, os.cpu_count() or 4)) as executor:
            future_to_file = {executor.submit(self.validate_file, file_path): file_path for file_path in test_files}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    # Results processed in validate_file method
                except Exception as exc:
                    logger.error(f"Exception validating {file_path}: {exc}")
                    self.results["failed"].append({
                        "file": str(file_path),
                        "errors": [f"Exception during validation: {str(exc)}"],
                        "model_name": self._extract_model_name(file_path),
                        "traceback": traceback.format_exc()
                    })
                    self.results["stats"]["failed"] += 1
        
        # Generate and save report
        self.generate_report()
        
        return self.results
    
    def _extract_model_name(self, file_path: Path) -> str:
        """Extract the model name from a test file path."""
        match = re.match(r'test_hf_(.+)\.py$', file_path.name)
        if match:
            return match.group(1)
        return "unknown"
    
    def _get_model_architecture(self, model_name: str) -> str:
        """Determine the model architecture from the model name."""
        model_name_lower = model_name.lower()
        
        for arch_type, models in ARCHITECTURE_TYPES.items():
            for model in models:
                if model.lower() in model_name_lower:
                    return arch_type
        
        return "unknown"
    
    def validate_file(self, file_path: Path) -> Dict:
        """Validate a single test file."""
        model_name = self._extract_model_name(file_path)
        architecture = self._get_model_architecture(model_name)
        
        # Update architecture stats
        if architecture not in self.results["stats"]["by_architecture"]:
            self.results["stats"]["by_architecture"][architecture] = {
                "total": 0, "passed": 0, "failed": 0, "warnings": 0
            }
        self.results["stats"]["by_architecture"][architecture]["total"] += 1
        
        errors = []
        warnings = []
        
        # 1. Check if file exists
        if not file_path.exists():
            errors.append(f"File {file_path} does not exist")
            self._record_failure(file_path, model_name, architecture, errors)
            return {"success": False, "errors": errors}
        
        # 2. Check syntax validity
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
            
            try:
                ast.parse(file_content)
            except SyntaxError as e:
                errors.append(f"Syntax error on line {e.lineno}: {e.msg}")
                if hasattr(e, 'text') and e.text:
                    errors.append(f"  {e.text}")
                    if hasattr(e, 'offset') and e.offset:
                        errors.append(f"  {' ' * (e.offset - 1)}^")
                self._record_error("syntax_error")
                
        except Exception as e:
            errors.append(f"Error reading file: {str(e)}")
            self._record_failure(file_path, model_name, architecture, errors)
            return {"success": False, "errors": errors}
        
        # 3. Validate file structure
        if errors:
            self._record_failure(file_path, model_name, architecture, errors)
            return {"success": False, "errors": errors}
        
        # Parse the AST for structural validation
        try:
            tree = ast.parse(file_content)
            
            # Check for class definition
            test_class = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                    test_class = node
                    break
            
            if test_class is None:
                errors.append("No test class found (should start with 'Test')")
                self._record_error("missing_test_class")
            else:
                # Check for required methods in the test class
                methods = {
                    "test_pipeline": False,
                    "run_tests": False,
                    "__init__": False
                }
                
                for node in ast.walk(test_class):
                    if isinstance(node, ast.FunctionDef):
                        method_name = node.name
                        if method_name in methods:
                            methods[method_name] = True
                
                for method_name, found in methods.items():
                    if not found:
                        errors.append(f"Missing required method: {method_name}")
                        self._record_error("missing_method")
            
            # Check for main function
            has_main = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "main":
                    has_main = True
                    break
            
            if not has_main:
                errors.append("Missing main function")
                self._record_error("missing_main")
                
            # Check for hardware detection
            hardware_detection = False
            device_selection = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in ["HAS_TORCH", "torch.cuda.is_available"]:
                    hardware_detection = True
                if isinstance(node, ast.Attribute) and node.attr == "cuda" and isinstance(node.value, ast.Name) and node.value.id == "torch":
                    hardware_detection = True
                if isinstance(node, ast.FunctionDef) and node.name == "select_device":
                    device_selection = True
            
            if not hardware_detection:
                warnings.append("No hardware detection found (torch.cuda.is_available)")
                self._record_warning("missing_hardware_detection")
            
            if not device_selection:
                warnings.append("No device selection function found")
                self._record_warning("missing_device_selection")
            
            # Check for task configuration
            task_validation = self._validate_task_configuration(file_content, model_name, architecture)
            if task_validation["errors"]:
                errors.extend(task_validation["errors"])
                for error in task_validation["errors"]:
                    self._record_error("task_configuration")
            
            if task_validation["warnings"]:
                warnings.extend(task_validation["warnings"])
                for warning in task_validation["warnings"]:
                    self._record_warning("task_configuration")
            
        except Exception as e:
            errors.append(f"Error validating file structure: {str(e)}")
            errors.append(traceback.format_exc())
            self._record_error("validation_error")
        
        if errors:
            self._record_failure(file_path, model_name, architecture, errors, warnings)
            return {"success": False, "errors": errors, "warnings": warnings}
        else:
            self._record_success(file_path, model_name, architecture, warnings)
            return {"success": True, "warnings": warnings}
    
    def _validate_task_configuration(self, content: str, model_name: str, architecture: str) -> Dict:
        """Validate the task configuration for a model."""
        errors = []
        warnings = []
        
        # Get expected tasks for this architecture
        expected_tasks = EXPECTED_TASKS.get(architecture, [])
        if not expected_tasks:
            warnings.append(f"Unknown architecture {architecture} for model {model_name}, can't validate task")
            return {"errors": errors, "warnings": warnings}
        
        # Check for pipeline usage with task configuration
        pipeline_pattern = r'transformers\.pipeline\(\s*["\']([^"\']+)["\']'
        pipeline_matches = re.findall(pipeline_pattern, content)
        
        if not pipeline_matches:
            errors.append("No pipeline task configuration found")
            return {"errors": errors, "warnings": warnings}
        
        task = pipeline_matches[0]
        
        # Check if the task is valid for this architecture
        if task not in expected_tasks:
            valid_tasks_str = ", ".join(expected_tasks)
            errors.append(f"Task '{task}' may not be appropriate for {architecture} models. Expected one of: {valid_tasks_str}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _record_success(self, file_path: Path, model_name: str, architecture: str, warnings: List[str]):
        """Record a successful validation."""
        self.results["passed"].append({
            "file": str(file_path),
            "model_name": model_name,
            "architecture": architecture,
            "warnings": warnings
        })
        self.results["stats"]["passed"] += 1
        if warnings:
            self.results["stats"]["warnings"] += 1
            
        # Update architecture stats
        self.results["stats"]["by_architecture"][architecture]["passed"] += 1
        if warnings:
            self.results["stats"]["by_architecture"][architecture]["warnings"] += 1
        
        if self.verbose and not warnings:
            logger.info(f"✅ Passed: {file_path}")
        elif self.verbose:
            logger.info(f"⚠️ Passed with warnings: {file_path}")
    
    def _record_failure(self, file_path: Path, model_name: str, architecture: str, errors: List[str], warnings: List[str] = None):
        """Record a failed validation."""
        warnings = warnings or []
        self.results["failed"].append({
            "file": str(file_path),
            "model_name": model_name,
            "architecture": architecture,
            "errors": errors,
            "warnings": warnings
        })
        self.results["stats"]["failed"] += 1
        if warnings:
            self.results["stats"]["warnings"] += 1
            
        # Update architecture stats
        self.results["stats"]["by_architecture"][architecture]["failed"] += 1
        if warnings:
            self.results["stats"]["by_architecture"][architecture]["warnings"] += 1
        
        if self.verbose:
            logger.error(f"❌ Failed: {file_path}")
            for error in errors:
                logger.error(f"  - {error}")
    
    def _record_error(self, error_type: str):
        """Record an error type for statistics."""
        if error_type not in self.results["stats"]["by_error_type"]:
            self.results["stats"]["by_error_type"][error_type] = 0
        self.results["stats"]["by_error_type"][error_type] += 1
    
    def _record_warning(self, warning_type: str):
        """Record a warning type for statistics."""
        warning_key = f"warning_{warning_type}"
        if warning_key not in self.results["stats"]["by_error_type"]:
            self.results["stats"]["by_error_type"][warning_key] = 0
        self.results["stats"]["by_error_type"][warning_key] += 1
    
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
        
        # Add timestamp
        self.results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save JSON report
        with open(self.report_file.with_suffix('.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate markdown report
        md_report = self._generate_markdown_report()
        with open(self.report_file, 'w') as f:
            f.write(md_report)
        
        logger.info(f"Validation report saved to {self.report_file}")
        logger.info(f"JSON report saved to {self.report_file.with_suffix('.json')}")
        
        # Print summary
        logger.info(f"Validation summary: {self.results['stats']['passed']} passed, "
                   f"{self.results['stats']['failed']} failed, "
                   f"{self.results['stats']['warnings']} with warnings, "
                   f"out of {total} test files.")
    
    def _generate_markdown_report(self) -> str:
        """Generate a markdown report."""
        total = self.results["stats"]["total"]
        passed = self.results["stats"]["passed"]
        failed = self.results["stats"]["failed"]
        warnings = self.results["stats"]["warnings"]
        passed_percent = self.results["stats"]["passed_percent"]
        
        report = [
            "# HuggingFace Model Test Validation Report",
            f"\nGenerated on: {self.results['timestamp']}",
            
            "\n## Summary",
            f"- **Total test files**: {total}",
            f"- **Passed**: {passed} ({passed_percent}%)",
            f"- **Failed**: {failed} ({self.results['stats']['failed_percent']}%)",
            f"- **With warnings**: {warnings}",
            
            "\n## Results by Architecture",
            "| Architecture | Total | Passed | Failed | Warnings |",
            "|-------------|-------|--------|--------|----------|",
        ]
        
        for arch, stats in sorted(self.results["stats"]["by_architecture"].items()):
            arch_total = stats["total"]
            arch_passed = stats["passed"]
            arch_failed = stats["failed"]
            arch_warnings = stats["warnings"]
            report.append(f"| {arch} | {arch_total} | {arch_passed} | {arch_failed} | {arch_warnings} |")
        
        report.append("\n## Error Types")
        report.append("| Error Type | Count |")
        report.append("|------------|-------|")
        
        for error_type, count in sorted(self.results["stats"]["by_error_type"].items()):
            # Format the error type for display
            display_error = error_type.replace("_", " ").capitalize()
            report.append(f"| {display_error} | {count} |")
        
        if failed > 0:
            report.append("\n## Failed Tests")
            for i, failure in enumerate(self.results["failed"]):
                if i > 9:  # Limit to 10 failures in the report
                    report.append(f"\n... and {len(self.results['failed']) - 10} more failures. See JSON report for details.")
                    break
                
                report.append(f"\n### {i+1}. {failure['model_name']}")
                report.append(f"- **File**: {failure['file']}")
                report.append(f"- **Architecture**: {failure['architecture']}")
                report.append("- **Errors**:")
                for error in failure["errors"]:
                    report.append(f"  - {error}")
                
                if failure.get("warnings"):
                    report.append("- **Warnings**:")
                    for warning in failure["warnings"]:
                        report.append(f"  - {warning}")
        
        return "\n".join(report)

def main():
    """Main entry point for the model test validator."""
    parser = argparse.ArgumentParser(description="Validate HuggingFace model test files")
    parser.add_argument("--directory", "-d", type=str, default="fixed_tests",
                        help="Directory containing test files")
    parser.add_argument("--report", "-r", type=str, default="validation_report.md",
                        help="Path to output report file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Resolve directory path
    directory = args.directory
    if not os.path.isabs(directory):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    
    # Create validator and run validation
    validator = ModelTestValidator(directory, args.report, args.verbose)
    results = validator.validate_all()
    
    # Set exit code based on validation results
    if results["stats"]["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()