#!/usr/bin/env python3
"""
Model Documentation Generator for End-to-End Testing Framework

This module generates Markdown documentation for models, explaining the implementation
details, expected behavior, and usage patterns.
"""

import os
import sys
import re
import json
import logging
import inspect
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import project utilities (assuming they exist)
try:
    from simple_utils import setup_logging
except ImportError:
    # Define a simple setup_logging function if the import fails
    def setup_logging(logger, level=logging.INFO):
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

# Set up logging
logger = logging.getLogger(__name__)
setup_logging(logger)

class ModelDocGenerator:
    """Generates comprehensive documentation for model implementations."""
    
    def __init__(self, model_name: str, hardware: str, 
                 skill_path: str, test_path: str, benchmark_path: str,
                 expected_results_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize the model documentation generator.
        
        Args:
            model_name: Name of the model being documented
            hardware: Hardware platform the model is running on
            skill_path: Path to the generated skill file
            test_path: Path to the generated test file
            benchmark_path: Path to the generated benchmark file
            expected_results_path: Path to expected results file (optional)
            output_dir: Directory to save the documentation (optional)
            verbose: Whether to output verbose logs
        """
        self.model_name = model_name
        self.hardware = hardware
        self.skill_path = skill_path
        self.test_path = test_path
        self.benchmark_path = benchmark_path
        self.expected_results_path = expected_results_path
        
        if output_dir:
            self.output_dir = output_dir
        else:
            # Default to a 'docs' directory next to the script
            self.output_dir = os.path.join(os.path.dirname(script_dir), "model_documentation")
        
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    
    def extract_docstrings(self, file_path: str) -> Dict[str, str]:
        """
        Extract docstrings from Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary mapping function/class names to their docstrings
        """
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
            
            # Use regex to extract docstrings
            # This is a simple implementation - a real one would use AST or similar
            docstring_map = {}
            
            # Extract module docstring
            module_match = re.search(r'^"""(.*?)"""', file_content, re.DOTALL)
            if module_match:
                docstring_map["module"] = module_match.group(1).strip()
            
            # Extract class docstrings
            class_matches = re.finditer(r'class\s+(\w+).*?:(?:\s+"""(.*?)""")?', file_content, re.DOTALL)
            for match in class_matches:
                class_name = match.group(1)
                docstring = match.group(2)
                if docstring:
                    docstring_map[class_name] = docstring.strip()
            
            # Extract method docstrings
            method_matches = re.finditer(r'def\s+(\w+).*?:(?:\s+"""(.*?)""")?', file_content, re.DOTALL)
            for match in method_matches:
                method_name = match.group(1)
                docstring = match.group(2)
                if docstring:
                    docstring_map[method_name] = docstring.strip()
            
            return docstring_map
            
        except Exception as e:
            logger.error(f"Error extracting docstrings from {file_path}: {str(e)}")
            return {}
    
    def extract_key_code_snippets(self, file_path: str) -> Dict[str, str]:
        """
        Extract key code snippets from Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary mapping snippet names to code
        """
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
            
            # Extract snippets based on file type
            snippets = {}
            
            if "skill" in os.path.basename(file_path):
                # Extract relevant parts from skill file
                
                # Extract class definition
                class_match = re.search(r'class\s+\w+.*?(?=\n\n|\Z)', file_content, re.DOTALL)
                if class_match:
                    snippets["class_definition"] = class_match.group(0)
                
                # Extract setup method
                setup_match = re.search(r'def\s+setup.*?(?=\n    def|\Z)', file_content, re.DOTALL)
                if setup_match:
                    snippets["setup_method"] = setup_match.group(0)
                
                # Extract run method
                run_match = re.search(r'def\s+run.*?(?=\n    def|\Z)', file_content, re.DOTALL)
                if run_match:
                    snippets["run_method"] = run_match.group(0)
            
            elif "test" in os.path.basename(file_path):
                # Extract relevant parts from test file
                
                # Extract test class
                test_class_match = re.search(r'class\s+Test\w+.*?(?=\n\nif|\Z)', file_content, re.DOTALL)
                if test_class_match:
                    snippets["test_class"] = test_class_match.group(0)
                
                # Extract test methods
                test_methods = re.finditer(r'def\s+test_\w+.*?(?=\n    def|\n\n|\Z)', file_content, re.DOTALL)
                for i, match in enumerate(test_methods):
                    snippets[f"test_method_{i+1}"] = match.group(0)
            
            elif "benchmark" in os.path.basename(file_path):
                # Extract relevant parts from benchmark file
                
                # Extract benchmark function
                benchmark_match = re.search(r'def\s+benchmark.*?(?=\n\ndef|\n\nif|\Z)', file_content, re.DOTALL)
                if benchmark_match:
                    snippets["benchmark_function"] = benchmark_match.group(0)
                
                # Extract main execution block
                main_match = re.search(r'if\s+__name__\s*==\s*"__main__".*', file_content, re.DOTALL)
                if main_match:
                    snippets["main_execution"] = main_match.group(0)
            
            return snippets
            
        except Exception as e:
            logger.error(f"Error extracting code snippets from {file_path}: {str(e)}")
            return {}
    
    def load_expected_results(self) -> Dict[str, Any]:
        """
        Load expected results from file.
        
        Returns:
            Dictionary with expected results or empty dict if file not found
        """
        if not self.expected_results_path or not os.path.exists(self.expected_results_path):
            logger.warning(f"No expected results found at {self.expected_results_path}")
            return {}
        
        try:
            with open(self.expected_results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading expected results: {str(e)}")
            return {}
    
    def generate_documentation(self) -> str:
        """
        Generate comprehensive Markdown documentation for the model.
        
        Returns:
            Path to the generated documentation file
        """
        logger.info(f"Generating documentation for {self.model_name} on {self.hardware}...")
        
        # Extract information from files
        skill_docstrings = self.extract_docstrings(self.skill_path)
        test_docstrings = self.extract_docstrings(self.test_path)
        benchmark_docstrings = self.extract_docstrings(self.benchmark_path)
        
        skill_snippets = self.extract_key_code_snippets(self.skill_path)
        test_snippets = self.extract_key_code_snippets(self.test_path)
        benchmark_snippets = self.extract_key_code_snippets(self.benchmark_path)
        
        expected_results = self.load_expected_results()
        
        # Create output directory if it doesn't exist
        model_doc_dir = os.path.join(self.output_dir, self.model_name)
        os.makedirs(model_doc_dir, exist_ok=True)
        
        # Generate documentation file
        doc_path = os.path.join(model_doc_dir, f"{self.hardware}_implementation.md")
        
        with open(doc_path, 'w') as f:
            f.write(f"# {self.model_name} Implementation on {self.hardware.upper()}\n\n")
            
            # Overview section
            f.write("## Overview\n\n")
            f.write(f"This document describes the implementation and testing of the {self.model_name} model ")
            f.write(f"on {self.hardware} hardware. It includes details about the skill implementation, ")
            f.write("test cases, benchmarking methodology, and expected results.\n\n")
            
            # Model information
            f.write("## Model Information\n\n")
            f.write(f"- **Model**: {self.model_name}\n")
            f.write(f"- **Hardware**: {self.hardware}\n")
            
            if expected_results:
                # Add performance metrics if available
                if "metrics" in expected_results:
                    f.write("- **Performance Metrics**:\n")
                    metrics = expected_results["metrics"]
                    for metric_name, metric_value in metrics.items():
                        f.write(f"  - {metric_name}: {metric_value}\n")
            
            f.write("\n")
            
            # Skill implementation
            f.write("## Skill Implementation\n\n")
            f.write("The skill implementation is responsible for loading and running the model.\n\n")
            
            if "class_definition" in skill_snippets:
                f.write("### Class Definition\n\n")
                f.write("```python\n" + skill_snippets["class_definition"] + "\n```\n\n")
            
            if "setup_method" in skill_snippets:
                f.write("### Setup Method\n\n")
                f.write("```python\n" + skill_snippets["setup_method"] + "\n```\n\n")
            
            if "run_method" in skill_snippets:
                f.write("### Run Method\n\n")
                f.write("```python\n" + skill_snippets["run_method"] + "\n```\n\n")
            
            # Test implementation
            f.write("## Test Implementation\n\n")
            f.write("The test implementation validates that the model produces correct outputs.\n\n")
            
            if "test_class" in test_snippets:
                f.write("### Test Class\n\n")
                f.write("```python\n" + test_snippets["test_class"] + "\n```\n\n")
            
            # Find all test methods
            test_methods = [k for k in test_snippets.keys() if k.startswith("test_method_")]
            if test_methods:
                f.write("### Test Methods\n\n")
                for method_key in test_methods:
                    f.write("```python\n" + test_snippets[method_key] + "\n```\n\n")
            
            # Benchmark implementation
            f.write("## Benchmark Implementation\n\n")
            f.write("The benchmark measures the performance of the model on this hardware.\n\n")
            
            if "benchmark_function" in benchmark_snippets:
                f.write("### Benchmark Function\n\n")
                f.write("```python\n" + benchmark_snippets["benchmark_function"] + "\n```\n\n")
            
            if "main_execution" in benchmark_snippets:
                f.write("### Execution\n\n")
                f.write("```python\n" + benchmark_snippets["main_execution"] + "\n```\n\n")
            
            # Expected results
            f.write("## Expected Results\n\n")
            
            if expected_results:
                f.write("The model should produce outputs matching these expected results:\n\n")
                f.write("```json\n" + json.dumps(expected_results, indent=2) + "\n```\n\n")
                
                # Add specific input/output examples if available
                if "input" in expected_results and "output" in expected_results:
                    f.write("### Input/Output Example\n\n")
                    f.write("**Input:**\n")
                    f.write("```json\n" + json.dumps(expected_results["input"], indent=2) + "\n```\n\n")
                    f.write("**Expected Output:**\n")
                    f.write("```json\n" + json.dumps(expected_results["output"], indent=2) + "\n```\n\n")
            else:
                f.write("No expected results are available yet. Run the tests and update the expected results.\n\n")
            
            # Hardware-specific notes
            f.write("## Hardware-Specific Notes\n\n")
            
            if self.hardware == "cpu":
                f.write("- Standard CPU implementation with no special optimizations\n")
                f.write("- Uses PyTorch's default CPU backend\n")
                f.write("- Suitable for development and testing\n")
            elif self.hardware == "cuda":
                f.write("- Optimized for NVIDIA GPUs using CUDA\n")
                f.write("- Requires CUDA toolkit and compatible NVIDIA drivers\n")
                f.write("- Best performance with batch processing\n")
            elif self.hardware == "rocm":
                f.write("- Optimized for AMD GPUs using ROCm\n")
                f.write("- Requires ROCm installation and compatible AMD hardware\n")
                f.write("- May require specific environment variables for optimal performance\n")
            elif self.hardware == "mps":
                f.write("- Optimized for Apple Silicon using Metal Performance Shaders\n")
                f.write("- Requires macOS and Apple Silicon hardware\n")
                f.write("- Lower power consumption compared to discrete GPUs\n")
            elif self.hardware == "openvino":
                f.write("- Optimized using Intel OpenVINO\n")
                f.write("- Works on CPU, Intel GPUs, and other Intel hardware\n")
                f.write("- Requires OpenVINO Runtime installation\n")
            elif self.hardware == "qnn":
                f.write("- Optimized for Qualcomm processors using QNN\n")
                f.write("- Requires Qualcomm AI Engine SDK\n")
                f.write("- Best suited for mobile and edge devices\n")
            elif self.hardware == "webnn":
                f.write("- Optimized for web browsers using WebNN API\n")
                f.write("- Best performance on browsers with native WebNN support (Edge, Chrome)\n")
                f.write("- Falls back to WebAssembly on unsupported browsers\n")
            elif self.hardware == "webgpu":
                f.write("- Optimized for web browsers using WebGPU API\n")
                f.write("- Requires browsers with WebGPU support\n")
                f.write("- Uses compute shaders for accelerated processing\n")
            elif self.hardware == "samsung":
                f.write("- Optimized for Samsung NPU\n")
                f.write("- Requires One UI 5.0+ and compatible Samsung devices\n")
                f.write("- Low power consumption for on-device inference\n")
            
            f.write("\n")
            
            # Implementation history
            f.write("## Implementation History\n\n")
            f.write("- Initial implementation: AUTO-GENERATED\n")
            f.write(f"- Last updated: {os.environ.get('USER', 'unknown')}, {os.environ.get('DATE', 'auto-generated')}\n\n")
        
        logger.info(f"Documentation generated: {doc_path}")
        return doc_path


def generate_model_documentation(model_name: str, hardware: str, 
                                skill_path: str, test_path: str, benchmark_path: str,
                                expected_results_path: Optional[str] = None,
                                output_dir: Optional[str] = None) -> str:
    """
    Generate documentation for a model implementation.
    
    Args:
        model_name: Name of the model
        hardware: Hardware platform
        skill_path: Path to skill implementation
        test_path: Path to test implementation
        benchmark_path: Path to benchmark implementation
        expected_results_path: Path to expected results file (optional)
        output_dir: Output directory for documentation (optional)
        
    Returns:
        Path to the generated documentation file
    """
    generator = ModelDocGenerator(
        model_name=model_name,
        hardware=hardware,
        skill_path=skill_path,
        test_path=test_path,
        benchmark_path=benchmark_path,
        expected_results_path=expected_results_path,
        output_dir=output_dir
    )
    
    return generator.generate_documentation()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate model documentation")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--hardware", required=True, help="Hardware platform")
    parser.add_argument("--skill-path", required=True, help="Path to skill implementation")
    parser.add_argument("--test-path", required=True, help="Path to test implementation")
    parser.add_argument("--benchmark-path", required=True, help="Path to benchmark implementation")
    parser.add_argument("--expected-results", help="Path to expected results file")
    parser.add_argument("--output-dir", help="Output directory for documentation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    doc_path = generate_model_documentation(
        model_name=args.model,
        hardware=args.hardware,
        skill_path=args.skill_path,
        test_path=args.test_path,
        benchmark_path=args.benchmark_path,
        expected_results_path=args.expected_results,
        output_dir=args.output_dir
    )
    
    print(f"Documentation generated: {doc_path}")