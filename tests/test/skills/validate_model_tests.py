#!/usr/bin/env python3
"""
Validate Model Tests

This script validates HuggingFace model test files for:
1. Syntax correctness (using AST parsing)
2. Structure validation (checking for required components)
3. Pipeline configuration validation (checking for appropriate tasks)
4. Task input validation (checking for appropriate inputs)

Usage:
    python validate_model_tests.py --directory TESTS_DIR [--report REPORT_FILE]
"""

import os
import sys
import re
import ast
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import architecture and task mappings from the standardization script
try:
    from standardize_task_configurations import (
        ARCHITECTURE_TYPES,
        RECOMMENDED_TASKS,
        SPECIAL_TASK_OVERRIDES,
        TEST_INPUTS
    )
except ImportError:
    # Define them here as fallback
    ARCHITECTURE_TYPES = {
        "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta", "albert"],
        "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "gpt-neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt"],
        "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan"],
        "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2", "resnet"],
        "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip"],
        "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
        "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava"]
    }
    
    RECOMMENDED_TASKS = {
        "encoder-only": "fill-mask",
        "decoder-only": "text-generation",
        "encoder-decoder": "text2text-generation",
        "vision": "image-classification",
        "vision-text": "image-to-text",
        "speech": "automatic-speech-recognition",
        "multimodal": "image-to-text"
    }
    
    SPECIAL_TASK_OVERRIDES = {
        "clip": "zero-shot-image-classification",
        "chinese-clip": "zero-shot-image-classification",
        "vision-text-dual-encoder": "zero-shot-image-classification",
        "wav2vec2-bert": "automatic-speech-recognition",
        "speech-to-text": "automatic-speech-recognition",
        "speech-to-text-2": "translation",
        "blip-2": "image-to-text",
        "video-llava": "image-to-text",
        "conditional-detr": "object-detection",
        "detr": "object-detection",
        "mask2former": "image-segmentation",
        "segformer": "image-segmentation",
        "sam": "image-segmentation"
    }
    
    TEST_INPUTS = {
        "fill-mask": '"The <mask> is a language model."',
        "text-generation": '"This model can"',
        "text2text-generation": '"translate English to French: Hello, how are you?"',
        "image-classification": '"An image of a cat."',
        "image-to-text": '"An image of a landscape."',
        "automatic-speech-recognition": '"A short audio clip."',
        "zero-shot-image-classification": '"An image with labels: dog, cat, bird."',
        "translation": '"Hello, how are you?"',
        "object-detection": '"An image of a street scene."',
        "image-segmentation": '"An image for segmentation."'
    }

class ModelTestValidator:
    """Validates HuggingFace model test files."""
    
    def __init__(self, directory: str, report_file: Optional[str] = None):
        """Initialize the validator.
        
        Args:
            directory: Directory containing test files
            report_file: Path to write validation report (optional)
        """
        self.directory = Path(directory)
        self.report_file = report_file
        self.stats = {
            "total_files": 0,
            "syntax_valid": 0,
            "syntax_invalid": 0,
            "structure_valid": 0,
            "structure_invalid": 0,
            "pipeline_valid": 0,
            "pipeline_invalid": 0,
            "pipeline_missing": 0,
            "task_valid": 0,
            "task_invalid": 0,
            "by_architecture": {}
        }
        self.validation_results = {}
    
    def run(self):
        """Run the validation process."""
        # Find all test files
        test_files = list(self.directory.glob("test_hf_*.py"))
        self.stats["total_files"] = len(test_files)
        
        logger.info(f"Found {len(test_files)} test files to validate")
        
        # Process each file
        for file_path in test_files:
            file_result = self._validate_file(file_path)
            self.validation_results[str(file_path)] = file_result
        
        # Generate validation report
        self._generate_report()
        
        return self.validation_results
    
    def _validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a single test file.
        
        Returns:
            Dict containing validation results
        """
        model_name = self._extract_model_name(file_path)
        architecture = self._get_model_architecture(model_name)
        
        # Update architecture stats
        if architecture not in self.stats["by_architecture"]:
            self.stats["by_architecture"][architecture] = {
                "total": 0,
                "syntax_valid": 0,
                "syntax_invalid": 0,
                "structure_valid": 0,
                "structure_invalid": 0,
                "pipeline_valid": 0,
                "pipeline_invalid": 0,
                "pipeline_missing": 0,
                "task_valid": 0,
                "task_invalid": 0
            }
        self.stats["by_architecture"][architecture]["total"] += 1
        
        # Read file content
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return {
                "model_name": model_name,
                "architecture": architecture,
                "syntax_valid": False,
                "structure_valid": False,
                "pipeline_valid": False,
                "task_valid": False,
                "errors": [f"File read error: {str(e)}"]
            }
        
        # Initialize result dict
        result = {
            "model_name": model_name,
            "architecture": architecture,
            "syntax_valid": False,
            "structure_valid": False,
            "pipeline_valid": False,
            "task_valid": False,
            "errors": []
        }
        
        # 1. Check syntax validity using AST parsing
        try:
            ast.parse(content)
            result["syntax_valid"] = True
            self.stats["syntax_valid"] += 1
            self.stats["by_architecture"][architecture]["syntax_valid"] += 1
        except SyntaxError as e:
            result["syntax_valid"] = False
            error_msg = f"Syntax error on line {e.lineno}: {e.msg}"
            result["errors"].append(error_msg)
            self.stats["syntax_invalid"] += 1
            self.stats["by_architecture"][architecture]["syntax_invalid"] += 1
            logger.warning(f"Syntax error in {file_path}: {error_msg}")
        
        # 2. Check structure (only if syntax is valid)
        if result["syntax_valid"]:
            structure_result = self._validate_structure(content, model_name)
            result.update(structure_result)
            if structure_result["structure_valid"]:
                self.stats["structure_valid"] += 1
                self.stats["by_architecture"][architecture]["structure_valid"] += 1
            else:
                self.stats["structure_invalid"] += 1
                self.stats["by_architecture"][architecture]["structure_invalid"] += 1
        
        # 3. Check pipeline configuration (only if syntax and structure are valid)
        if result["syntax_valid"] and result["structure_valid"]:
            pipeline_result = self._validate_pipeline(content, model_name, architecture)
            result.update(pipeline_result)
            
            if "has_pipeline" in pipeline_result and pipeline_result["has_pipeline"]:
                if pipeline_result["pipeline_valid"]:
                    self.stats["pipeline_valid"] += 1
                    self.stats["by_architecture"][architecture]["pipeline_valid"] += 1
                else:
                    self.stats["pipeline_invalid"] += 1
                    self.stats["by_architecture"][architecture]["pipeline_invalid"] += 1
            else:
                self.stats["pipeline_missing"] += 1
                self.stats["by_architecture"][architecture]["pipeline_missing"] += 1
            
            if pipeline_result.get("task_valid", False):
                self.stats["task_valid"] += 1
                self.stats["by_architecture"][architecture]["task_valid"] += 1
            elif "current_task" in pipeline_result:  # Only count if pipeline exists
                self.stats["task_invalid"] += 1
                self.stats["by_architecture"][architecture]["task_invalid"] += 1
        
        return result
    
    def _validate_structure(self, content: str, model_name: str) -> Dict[str, Any]:
        """Validate the structure of a test file."""
        result = {
            "structure_valid": False,
            "has_test_class": False,
            "has_test_pipeline": False,
            "has_run_tests": False
        }
        
        errors = []
        
        # Look for test class using regex (more reliable than AST for potentially invalid files)
        class_pattern = r'class\s+Test(\w+)Models'
        class_match = re.search(class_pattern, content)
        
        if class_match:
            result["has_test_class"] = True
            class_name = class_match.group(1)
            
            # Check if class name matches model name
            expected_class_name = "".join(part.capitalize() for part in model_name.replace("-", "_").split("_"))
            if expected_class_name.lower() != class_name.lower():
                errors.append(f"Test class name mismatch: expected Test{expected_class_name}Models, got Test{class_name}Models")
        else:
            errors.append("No test class found")
        
        # Check for test_pipeline method
        if result["has_test_class"]:
            pipeline_pattern = r'def\s+test_pipeline\s*\('
            pipeline_match = re.search(pipeline_pattern, content)
            
            if pipeline_match:
                result["has_test_pipeline"] = True
            else:
                errors.append("No test_pipeline method found")
        
        # Check for run_tests method
        if result["has_test_class"]:
            run_tests_pattern = r'def\s+run_tests\s*\('
            run_tests_match = re.search(run_tests_pattern, content)
            
            if run_tests_match:
                result["has_run_tests"] = True
            else:
                errors.append("No run_tests method found")
        
        # Structure is valid if it has all required components
        result["structure_valid"] = (
            result["has_test_class"] and
            result["has_test_pipeline"] and
            result["has_run_tests"]
        )
        
        if not result["structure_valid"]:
            result["errors"] = errors
        
        return result
    
    def _validate_pipeline(self, content: str, model_name: str, architecture: str) -> Dict[str, Any]:
        """Validate the pipeline configuration of a test file."""
        result = {
            "pipeline_valid": False,
            "has_pipeline": False,
            "task_valid": False
        }
        
        errors = []
        
        # Check for pipeline configuration
        pipeline_pattern = r'transformers\.pipeline\(\s*["\']([^"\']+)["\']'
        pipeline_match = re.search(pipeline_pattern, content)
        
        if pipeline_match:
            result["has_pipeline"] = True
            current_task = pipeline_match.group(1)
            result["current_task"] = current_task
            
            # Get recommended task for this model
            recommended_task = self._get_recommended_task(model_name, architecture)
            result["recommended_task"] = recommended_task
            
            # Check if task is valid
            if current_task == recommended_task:
                result["task_valid"] = True
            else:
                errors.append(f"Task mismatch: using '{current_task}', recommended '{recommended_task}'")
                
            # Check for test input
            test_input_pattern = r'test_input\s*=\s*["\']([^"\']*)["\']'
            test_input_match = re.search(test_input_pattern, content)
            
            if test_input_match:
                result["has_test_input"] = True
                current_input = test_input_match.group(1)
                result["current_input"] = current_input
                
                # Check if input is appropriate for task
                # This is a simple check, just ensuring input isn't empty
                if current_input.strip():
                    result["input_valid"] = True
                else:
                    result["input_valid"] = False
                    errors.append("Empty test input")
            else:
                result["has_test_input"] = False
                errors.append("No test input found")
            
            # Pipeline is valid if task and input are valid
            result["pipeline_valid"] = result["task_valid"] and result.get("input_valid", False)
        else:
            errors.append("No pipeline configuration found")
        
        if errors:
            result["errors"] = errors
        
        return result
    
    def _extract_model_name(self, file_path: Path) -> str:
        """Extract the model name from a test file path."""
        return file_path.stem.replace("test_hf_", "")
    
    def _get_model_architecture(self, model_name: str) -> str:
        """Determine the model architecture from the model name."""
        model_name_lower = model_name.lower()
        
        for arch_type, models in ARCHITECTURE_TYPES.items():
            for model in models:
                if model.lower() in model_name_lower:
                    return arch_type
        
        return "unknown"
    
    def _get_recommended_task(self, model_name: str, architecture: str) -> str:
        """Get the recommended task for a model."""
        # Check for special case overrides
        for special_model, task in SPECIAL_TASK_OVERRIDES.items():
            if special_model.lower() in model_name.lower():
                return task
        
        # Otherwise use the architecture default
        return RECOMMENDED_TASKS.get(architecture, "fill-mask")
    
    def _generate_report(self):
        """Generate a validation report."""
        if not self.report_file:
            return
        
        # Create report directory if needed
        report_dir = os.path.dirname(self.report_file)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        
        # Generate markdown report
        report = [
            "# HuggingFace Model Test Validation Report",
            "",
            f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Total files:** {self.stats['total_files']}",
            f"- **Syntax valid:** {self.stats['syntax_valid']} ({self.stats['syntax_valid']/self.stats['total_files']*100:.1f}%)",
            f"- **Structure valid:** {self.stats['structure_valid']} ({self.stats['structure_valid']/self.stats['total_files']*100:.1f}%)",
            f"- **Pipeline valid:** {self.stats['pipeline_valid']} ({self.stats['pipeline_valid']/self.stats['total_files']*100:.1f}%)",
            f"- **Task appropriate:** {self.stats['task_valid']} ({self.stats['task_valid']/self.stats['total_files']*100:.1f}%)",
            f"- **Pipeline missing:** {self.stats['pipeline_missing']} ({self.stats['pipeline_missing']/self.stats['total_files']*100:.1f}%)",
            "",
            "## Results by Architecture",
            ""
        ]
        
        # Add architecture-specific stats
        for arch, stats in sorted(self.stats["by_architecture"].items()):
            arch_total = stats["total"]
            report.extend([
                f"### {arch.capitalize()} ({arch_total} files)",
                "",
                f"- **Syntax valid:** {stats['syntax_valid']} ({stats['syntax_valid']/arch_total*100:.1f}%)",
                f"- **Structure valid:** {stats['structure_valid']} ({stats['structure_valid']/arch_total*100:.1f}%)",
                f"- **Pipeline valid:** {stats['pipeline_valid']} ({stats['pipeline_valid']/arch_total*100:.1f}%)",
                f"- **Task appropriate:** {stats['task_valid']} ({stats['task_valid']/arch_total*100:.1f}%)",
                f"- **Pipeline missing:** {stats['pipeline_missing']} ({stats['pipeline_missing']/arch_total*100:.1f}%)",
                ""
            ])
        
        # Add detailed results by file
        report.extend([
            "## Detailed Results",
            ""
        ])
        
        # Group files by status for better organization
        files_by_status = {
            "all_valid": [],
            "syntax_invalid": [],
            "structure_invalid": [],
            "pipeline_invalid": [],
            "pipeline_missing": []
        }
        
        for file_path, result in self.validation_results.items():
            if result["syntax_valid"] and result.get("structure_valid", False) and result.get("pipeline_valid", False):
                files_by_status["all_valid"].append((file_path, result))
            elif not result["syntax_valid"]:
                files_by_status["syntax_invalid"].append((file_path, result))
            elif not result.get("structure_valid", False):
                files_by_status["structure_invalid"].append((file_path, result))
            elif not result.get("has_pipeline", False):
                files_by_status["pipeline_missing"].append((file_path, result))
            else:
                files_by_status["pipeline_invalid"].append((file_path, result))
        
        # Add valid files
        report.extend([
            "### Valid Files",
            "",
            "These files passed all validation checks:",
            ""
        ])
        
        for file_path, result in sorted(files_by_status["all_valid"]):
            report.append(f"- `{os.path.basename(file_path)}` - {result['architecture']} - Task: {result.get('current_task', 'N/A')}")
        
        # Add files with issues
        for status, title, description in [
            ("syntax_invalid", "Syntax Errors", "These files have syntax errors that need to be fixed:"),
            ("structure_invalid", "Structure Issues", "These files have structural issues (missing class or methods):"),
            ("pipeline_missing", "Missing Pipeline", "These files are missing pipeline configuration:"),
            ("pipeline_invalid", "Incorrect Pipeline", "These files have pipeline configuration issues:")
        ]:
            if files_by_status[status]:
                report.extend([
                    "",
                    f"### {title}",
                    "",
                    description,
                    ""
                ])
                
                for file_path, result in sorted(files_by_status[status]):
                    report.append(f"- `{os.path.basename(file_path)}` - {result['architecture']}")
                    if "errors" in result:
                        for error in result["errors"]:
                            report.append(f"  - {error}")
        
        # Write report to file
        with open(self.report_file, 'w') as f:
            f.write("\n".join(report))
        
        logger.info(f"Validation report written to {self.report_file}")

def main():
    """Main entry point for the validator."""
    parser = argparse.ArgumentParser(description="Validate HuggingFace model test files")
    parser.add_argument("--directory", "-d", type=str, default="fixed_tests",
                        help="Directory containing test files")
    parser.add_argument("--report", "-r", type=str,
                        help="Path to write validation report")
    
    args = parser.parse_args()
    
    # Resolve directory path
    directory = args.directory
    if not os.path.isabs(directory):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    
    # Create and run the validator
    validator = ModelTestValidator(directory, args.report)
    results = validator.run()
    
    # Print a brief summary
    total = validator.stats["total_files"]
    syntax_valid = validator.stats["syntax_valid"]
    structure_valid = validator.stats["structure_valid"]
    pipeline_valid = validator.stats["pipeline_valid"]
    task_valid = validator.stats["task_valid"]
    
    print("\nVALIDATION SUMMARY")
    print("="*50)
    print(f"Total files processed: {total}")
    print(f"Syntax valid: {syntax_valid} ({syntax_valid/total*100:.1f}%)")
    print(f"Structure valid: {structure_valid} ({structure_valid/total*100:.1f}%)")
    print(f"Pipeline valid: {pipeline_valid} ({pipeline_valid/total*100:.1f}%)")
    print(f"Task appropriate: {task_valid} ({task_valid/total*100:.1f}%)")
    
    if args.report:
        print(f"\nDetailed report written to: {args.report}")
    
    # Return success if all files are valid
    return 0 if syntax_valid == total and structure_valid == total and pipeline_valid == total else 1

if __name__ == "__main__":
    sys.exit(main())