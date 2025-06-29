#!/usr/bin/env python3
"""
Unified Component Tester for End-to-End Testing Framework

This script implements an enhanced framework for generating and testing skill, test, 
and benchmark components together for every model. It addresses the remaining
priorities for the Improved End-to-End Testing Framework:

1. Generation and testing of all components together for every model
2. Creation of "expected_results" and "collected_results" folders for verification
3. Markdown documentation of HuggingFace class skills to compare with templates
4. Focus on fixing generators rather than individual test files
5. Template-driven approach for maintenance efficiency

Usage:
    python unified_component_tester.py --model bert-base-uncased --hardware cuda
    python unified_component_tester.py --model-family text-embedding --hardware all
    python unified_component_tester.py --model vit --hardware cuda,webgpu --update-expected
    python unified_component_tester.py --all-models --priority-hardware --quick-test
"""

import os
import sys
import json
import time
import uuid
import argparse
import logging
import datetime
import tempfile
import shutil
import concurrent.futures
import subprocess
import numpy as np
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import existing utilities
from simple_utils import ensure_dir_exists
from template_validation import ModelValidator, ResultComparer
from model_documentation_generator import ModelDocGenerator, generate_model_documentation

# Use the new documentation system
try:
    from doc_template_fixer import monkey_patch_model_doc_generator, monkey_patch_template_renderer
    from integrate_documentation_system import integrate_enhanced_doc_generator
    
    # Apply enhancements to documentation system
    monkey_patch_model_doc_generator()
    monkey_patch_template_renderer()
    integrate_enhanced_doc_generator()
    
    HAS_ENHANCED_DOCS = True
    logger.info("Enhanced documentation system integrated successfully")
except ImportError:
    HAS_ENHANCED_DOCS = False
    logger.warning("Enhanced documentation system not available. Using basic documentation.")

# Try to import DB integration
try:
    sys.path.append(os.path.join(test_dir, "../duckdb_api"))
    from duckdb_api.core.benchmark_db_updater import store_test_result, initialize_db
    import duckdb
    HAS_DB_API = True
except ImportError:
    HAS_DB_API = False
    logger.warning("DuckDB API modules not available. Using basic file storage only.")

# Constants
RESULTS_ROOT = os.path.abspath(os.path.join(script_dir, "../../"))
EXPECTED_RESULTS_DIR = os.path.join(RESULTS_ROOT, "expected_results")
COLLECTED_RESULTS_DIR = os.path.join(RESULTS_ROOT, "collected_results")
DOCS_DIR = os.path.join(RESULTS_ROOT, "model_documentation")
TEST_TIMEOUT = 300  # seconds
DEFAULT_DB_PATH = os.environ.get("BENCHMARK_DB_PATH", os.path.join(test_dir, "benchmark_db.duckdb"))
TEMPLATE_DB_PATH = os.path.join(script_dir, "template_database.duckdb")

# Ensure directories exist
for directory in [EXPECTED_RESULTS_DIR, COLLECTED_RESULTS_DIR, DOCS_DIR]:
    ensure_dir_exists(directory)

# Define model families and hardware platforms
MODEL_FAMILIES = {
    "text-embedding": ["bert-base-uncased", "bert-large-uncased", "sentence-transformers/all-MiniLM-L6-v2"],
    "text-generation": ["facebook/opt-125m", "google/flan-t5-small", "tiiuae/falcon-7b", "gpt2"],
    "vision": ["google/vit-base-patch16-224", "facebook/detr-resnet-50", "openai/clip-vit-base-patch32"],
    "audio": ["openai/whisper-tiny", "facebook/wav2vec2-base", "laion/clap-htsat-unfused"],
    "multimodal": ["openai/clip-vit-base-patch32", "llava-hf/llava-1.5-7b-hf", "facebook/flava-full"]
}

SUPPORTED_HARDWARE = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]
PRIORITY_HARDWARE = ["cpu", "cuda", "openvino", "webgpu"]  # Hardware platforms to prioritize in testing


class UnifiedComponentTester:
    """
    Unified Component Tester for generating and testing skill, test, and benchmark components together.
    
    This class implements the enhanced end-to-end testing framework that:
    1. Generates all components together using template-driven approach
    2. Creates expected results and collected results folders
    3. Runs comprehensive tests and benchmarks
    4. Validates results against expected outputs
    5. Generates detailed documentation with enhanced templates
    """
    
    def __init__(self, 
                 model_name: str,
                 hardware: str,
                 db_path: Optional[str] = None,
                 template_db_path: Optional[str] = None,
                 update_expected: bool = False,
                 generate_docs: bool = False,
                 quick_test: bool = False,
                 keep_temp: bool = False,
                 verbose: bool = False,
                 tolerance: float = 0.01,
                 git_hash: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 compare_templates: bool = False):
        """
        Initialize the unified component tester.
        
        Args:
            model_name: Name of the model to test
            hardware: Hardware platform to test on
            db_path: Path to DuckDB database for storing results
            template_db_path: Path to template database
            update_expected: Whether to update expected results with current test results
            generate_docs: Whether to generate documentation for the model
            quick_test: Whether to run a quick test with minimal validation
            keep_temp: Whether to keep temporary directories after tests
            verbose: Whether to output verbose logs
            tolerance: Tolerance for numeric comparisons (e.g., 0.01 for 1%)
            git_hash: Current git commit hash (for versioning)
            output_dir: Custom output directory for results
            compare_templates: Whether to compare generated components with templates
        """
        self.model_name = model_name
        self.hardware = hardware
        self.db_path = db_path or DEFAULT_DB_PATH
        self.template_db_path = template_db_path or TEMPLATE_DB_PATH
        self.update_expected = update_expected
        self.generate_docs = generate_docs
        self.quick_test = quick_test
        self.keep_temp = keep_temp
        self.verbose = verbose
        self.tolerance = tolerance
        self.git_hash = git_hash or self._get_git_hash()
        self.output_dir = output_dir or script_dir
        self.compare_templates = compare_templates
        
        # Determine model family from model name
        self.model_family = self._determine_model_family()
        
        # Set up logging
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            
        # Model validator for checking templates
        self.model_validator = ModelValidator(model_name, hardware, verbose=verbose)
        
        # Result comparer for validation
        self.result_comparer = ResultComparer(tolerance=tolerance, verbose=verbose)
        
        # Initialize database if needed
        if HAS_DB_API and db_path:
            try:
                initialize_db(db_path)
                logger.debug(f"Initialized database at {db_path}")
            except Exception as e:
                logger.warning(f"Error initializing database: {e}")
    
    def _determine_model_family(self) -> str:
        """Determine the model family based on model name."""
        for family, models in MODEL_FAMILIES.items():
            if any(model in self.model_name for model in models) or self.model_name in models:
                return family
                
        # Generic classification based on model name patterns
        if any(keyword in self.model_name.lower() for keyword in ["bert", "roberta", "sentence", "embedding"]):
            return "text-embedding"
        elif any(keyword in self.model_name.lower() for keyword in ["gpt", "llama", "opt", "t5", "falcon"]):
            return "text-generation"
        elif any(keyword in self.model_name.lower() for keyword in ["vit", "resnet", "detr", "yolo"]):
            return "vision"
        elif any(keyword in self.model_name.lower() for keyword in ["whisper", "wav2vec", "clap", "audio"]):
            return "audio"
        elif any(keyword in self.model_name.lower() for keyword in ["clip", "llava", "flava", "blip"]):
            return "multimodal"
            
        # Default to text-embedding if we can't determine
        logger.warning(f"Could not determine model family for {self.model_name}. Defaulting to text-embedding.")
        return "text-embedding"
    
    def _get_git_hash(self) -> str:
        """Get the current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Failed to get git hash. Using timestamp instead.")
            return f"unknown-{int(time.time())}"
    
    def generate_components(self, temp_dir: str) -> Tuple[str, str, str]:
        """
        Generate skill, test, and benchmark components together.
        
        Args:
            temp_dir: Directory to store generated files
            
        Returns:
            Tuple of paths to generated (skill_file, test_file, benchmark_file)
        """
        logger.info(f"Generating components for {self.model_name} on {self.hardware}")
        
        # Create paths for generated files
        model_name_safe = self.model_name.replace('/', '_')
        skill_file = os.path.join(temp_dir, f"{model_name_safe}_{self.hardware}_skill.py")
        test_file = os.path.join(temp_dir, f"test_{model_name_safe}_{self.hardware}.py")
        benchmark_file = os.path.join(temp_dir, f"benchmark_{model_name_safe}_{self.hardware}.py")
        
        try:
            # Try to use the template database and renderer
            from template_database import TemplateDatabase, add_default_templates
            from template_renderer import TemplateRenderer
            
            # Initialize template database if it doesn't exist
            if not os.path.exists(self.template_db_path):
                logger.info(f"Initializing template database at {self.template_db_path}")
                add_default_templates(self.template_db_path)
            
            # Create renderer
            renderer = TemplateRenderer(db_path=self.template_db_path, verbose=self.verbose)
            
            # Basic batch size settings
            batch_sizes = [1] if self.quick_test else [1, 2, 4, 8]
            
            # Create custom variables
            variables = {
                "batch_size": batch_sizes[0],
                "batch_sizes": batch_sizes,
                "git_hash": self.git_hash,
                "test_id": str(uuid.uuid4()),
                "test_timestamp": datetime.datetime.now().isoformat()
            }
            
            # Generate all components at once
            logger.debug("Generating components using template renderer")
            generated_files = renderer.render_component_set(
                model_name=self.model_name,
                hardware_platform=self.hardware,
                variables=variables,
                output_dir=temp_dir
            )
            
            # Verify the generated files exist
            if "skill" in generated_files and os.path.exists(skill_file):
                logger.info(f"Generated skill file: {skill_file}")
            else:
                # Fall back to legacy template method
                logger.warning("Template renderer didn't generate skill file, falling back to legacy method")
                skill_template = self._get_template("skill", self.model_name, self.hardware)
                skill_content = self._render_template(skill_template, self.model_name, self.hardware)
                with open(skill_file, 'w') as f:
                    f.write(skill_content)
                
            if "test" in generated_files and os.path.exists(test_file):
                logger.info(f"Generated test file: {test_file}")
            else:
                # Fall back to legacy template method
                logger.warning("Template renderer didn't generate test file, falling back to legacy method")
                test_template = self._get_template("test", self.model_name, self.hardware)
                test_content = self._render_template(test_template, self.model_name, self.hardware)
                with open(test_file, 'w') as f:
                    f.write(test_content)
                
            if "benchmark" in generated_files and os.path.exists(benchmark_file):
                logger.info(f"Generated benchmark file: {benchmark_file}")
            else:
                # Fall back to legacy template method
                logger.warning("Template renderer didn't generate benchmark file, falling back to legacy method")
                benchmark_template = self._get_template("benchmark", self.model_name, self.hardware)
                benchmark_content = self._render_template(benchmark_template, self.model_name, self.hardware)
                with open(benchmark_file, 'w') as f:
                    f.write(benchmark_content)
            
        except Exception as e:
            # Fall back to legacy template generation if the new method fails
            logger.warning(f"Error using template renderer: {e}. Falling back to legacy template method.")
            
            # Legacy template method
            skill_template = self._get_template("skill", self.model_name, self.hardware)
            test_template = self._get_template("test", self.model_name, self.hardware)
            benchmark_template = self._get_template("benchmark", self.model_name, self.hardware)
            
            # Render templates with model and hardware information
            skill_content = self._render_template(skill_template, self.model_name, self.hardware)
            test_content = self._render_template(test_template, self.model_name, self.hardware)
            benchmark_content = self._render_template(benchmark_template, self.model_name, self.hardware)
            
            # Write rendered templates to files
            with open(skill_file, 'w') as f:
                f.write(skill_content)
                
            with open(test_file, 'w') as f:
                f.write(test_content)
                
            with open(benchmark_file, 'w') as f:
                f.write(benchmark_content)
        
        # Validate generated files
        logger.debug("Validating generated files")
        skill_validation = self.model_validator.validate_skill(skill_file)
        test_validation = self.model_validator.validate_test(test_file)
        benchmark_validation = self.model_validator.validate_benchmark(benchmark_file)
        
        if not all([skill_validation.get("valid", False),
                   test_validation.get("valid", False),
                   benchmark_validation.get("valid", False)]):
            logger.warning("Validation failed for some components:")
            if not skill_validation.get("valid", False):
                logger.warning(f"Skill validation: {skill_validation.get('error', 'Unknown error')}")
            if not test_validation.get("valid", False):
                logger.warning(f"Test validation: {test_validation.get('error', 'Unknown error')}")
            if not benchmark_validation.get("valid", False):
                logger.warning(f"Benchmark validation: {benchmark_validation.get('error', 'Unknown error')}")
        
        return skill_file, test_file, benchmark_file
    
    def _render_template(self, template_content: str, model_name: str, hardware: str) -> str:
        """
        Render a template with model and hardware information.
        
        Args:
            template_content: Template content with placeholders
            model_name: Model name to substitute
            hardware: Hardware platform to substitute
            
        Returns:
            Rendered template content
        """
        # Get timestamp for the template
        timestamp = datetime.datetime.now().isoformat()
        
        # Replace placeholders
        rendered = template_content
        rendered = rendered.replace("{model_name}", model_name)
        rendered = rendered.replace("{hardware}", hardware)
        rendered = rendered.replace("{timestamp}", timestamp)
        
        return rendered
    
    def _get_template(self, template_type: str, model_name: str, hardware: str) -> str:
        """
        Get a template from the database or template files.
        
        Args:
            template_type: Type of template (skill, test, benchmark)
            model_name: Model name
            hardware: Hardware platform
            
        Returns:
            Template content as string
        """
        # In a real implementation, this would query the template database
        # For this example, we'll return placeholder templates
        
        # Determine model type based on name
        model_type = self._determine_model_family()
            
        # Basic placeholder templates
        if template_type == "skill":
            return f"""#!/usr/bin/env python3
'''
Skill implementation for {model_name} on {hardware} hardware.
Generated by unified component tester.
'''

import torch
import numpy as np
from typing import Dict, Any, List, Union

class {model_name.replace('-', '_').replace('/', '_').title()}Skill:
    '''
    Model skill for {model_name} on {hardware} hardware.
    Model type: {model_type}
    '''
    
    def __init__(self):
        self.model_name = "{model_name}"
        self.hardware = "{hardware}"
        self.model_type = "{model_type}"
        self.model = None
        self.tokenizer = None
        
    def setup(self, **kwargs) -> bool:
        '''Set up the model and tokenizer.'''
        try:
            if self.hardware == "cpu":
                # CPU setup logic
                device = "cpu"
            elif self.hardware == "cuda":
                # CUDA setup logic
                device = "cuda" if torch.cuda.is_available() else "cpu"
            # Additional hardware platforms...
            
            # Common setup logic
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(device)
            self.model.eval()
            
            return True
        except Exception as e:
            print(f"Error setting up model: {{e}}")
            return False
    
    def run(self, inputs: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        '''Run the model on inputs.'''
        try:
            if isinstance(inputs, str):
                inputs = [inputs]
                
            # Tokenize inputs
            encoded_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
            encoded_inputs = {{k: v.to(self.model.device) for k, v in encoded_inputs.items()}}
            
            # Run model
            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                
            # Process outputs
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return {{
                "embeddings": embeddings,
                "shape": embeddings.shape,
                "norm": np.linalg.norm(embeddings, axis=1).tolist()
            }}
        except Exception as e:
            print(f"Error running model: {{e}}")
            return {{"error": str(e)}}
    
    def cleanup(self) -> bool:
        '''Clean up resources.'''
        try:
            self.model = None
            self.tokenizer = None
            return True
        except Exception as e:
            print(f"Error during cleanup: {{e}}")
            return False
"""
        elif template_type == "test":
            return f"""#!/usr/bin/env python3
'''
Test file for {model_name} on {hardware} hardware.
Generated by unified component tester.
'''

import unittest
import numpy as np
import os
import sys
from typing import Dict, Any

# Import the skill
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the skill dynamically
from {model_name.replace('-', '_').replace('/', '_')}_{hardware}_skill import {model_name.replace('-', '_').replace('/', '_').title()}Skill

class Test{model_name.replace('-', '_').replace('/', '_').title()}:
    '''
    Test class for {model_name} on {hardware} hardware.
    '''
    
    def setUp(self):
        '''Set up the test environment.'''
        self.skill = {model_name.replace('-', '_').replace('/', '_').title()}Skill()
        self.setup_success = self.skill.setup()
        
    def tearDown(self):
        '''Clean up after the test.'''
        self.skill.cleanup()
        
    def test_setup(self):
        '''Test that the model sets up correctly.'''
        self.assertTrue(self.setup_success)
        self.assertIsNotNone(self.skill.model)
        self.assertIsNotNone(self.skill.tokenizer)
        
    def test_run_single_input(self):
        '''Test running the model on a single input.'''
        if not self.setup_success:
            self.skipTest("Model setup failed, skipping test.")
            
        # Run the model
        result = self.skill.run("This is a test.")
        
        # Check the result
        self.assertIn("embeddings", result)
        self.assertIn("shape", result)
        self.assertIn("norm", result)
        self.assertEqual(len(result["shape"]), 2)
        self.assertEqual(result["shape"][0], 1)
        
    def test_run_batch_input(self):
        '''Test running the model on a batch of inputs.'''
        if not self.setup_success:
            self.skipTest("Model setup failed, skipping test.")
            
        # Create a batch of inputs
        inputs = ["This is the first input.", "This is the second input."]
        
        # Run the model
        result = self.skill.run(inputs)
        
        # Check the result
        self.assertIn("embeddings", result)
        self.assertIn("shape", result)
        self.assertIn("norm", result)
        self.assertEqual(len(result["shape"]), 2)
        self.assertEqual(result["shape"][0], 2)
        
    def test_output_format(self):
        '''Test that the output format is correct.'''
        if not self.setup_success:
            self.skipTest("Model setup failed, skipping test.")
            
        # Run the model
        result = self.skill.run("This is a test.")
        
        # Check that embeddings have the expected format
        self.assertIsInstance(result["embeddings"], np.ndarray)
        self.assertEqual(len(result["embeddings"].shape), 2)
        
    def test_cleanup(self):
        '''Test that cleanup works correctly.'''
        if not self.setup_success:
            self.skipTest("Model setup failed, skipping test.")
            
        # Clean up resources
        cleanup_success = self.skill.cleanup()
        
        # Check that cleanup was successful
        self.assertTrue(cleanup_success)
        self.assertIsNone(self.skill.model)
        self.assertIsNone(self.skill.tokenizer)

# Run the tests if the script is executed directly
if __name__ == "__main__":
    unittest.main()
"""
        elif template_type == "benchmark":
            return f"""#!/usr/bin/env python3
'''
Benchmark file for {model_name} on {hardware} hardware.
Generated by unified component tester.
'''

import time
import json
import numpy as np
import os
import sys
from typing import List, Dict, Any
import argparse

# Import the skill
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the skill dynamically
from {model_name.replace('-', '_').replace('/', '_')}_{hardware}_skill import {model_name.replace('-', '_').replace('/', '_').title()}Skill

def benchmark(batch_sizes: List[int] = [1, 2, 4, 8], 
              warmup_runs: int = 5, 
              test_runs: int = 10) -> Dict[str, Any]:
    '''
    Benchmark the model on different batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to test
        warmup_runs: Number of warmup runs before timing
        test_runs: Number of test runs to average timing over
        
    Returns:
        Dictionary with benchmark results
    '''
    # Create skill
    skill = {model_name.replace('-', '_').replace('/', '_').title()}Skill()
    setup_success = skill.setup()
    
    if not setup_success:
        return {{"error": "Failed to set up model."}}
    
    # Prepare benchmark results
    results = {{
        "model_name": "{model_name}",
        "hardware": "{hardware}",
        "timestamp": "{timestamp}",
        "results_by_batch": {{}}
    }}
    
    # Generate a test sentence
    test_sentence = "This is a test sentence for benchmarking the model."
    
    # Benchmark different batch sizes
    for batch_size in batch_sizes:
        print(f"Benchmarking batch size: {{batch_size}}")
        
        # Create a batch of the specified size
        batch = [test_sentence] * batch_size
        
        # Warmup runs
        for _ in range(warmup_runs):
            skill.run(batch)
        
        # Test runs
        latencies = []
        for _ in range(test_runs):
            start_time = time.time()
            skill.run(batch)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        throughput = batch_size * 1000 / mean_latency  # Items per second
        
        # Store results for this batch size
        results["results_by_batch"][str(batch_size)] = {{
            "average_latency_ms": mean_latency,
            "std_latency_ms": std_latency,
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "average_throughput_items_per_second": throughput
        }}
    
    # Clean up
    skill.cleanup()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Benchmark {model_name} on {hardware}")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8", help="Comma-separated list of batch sizes")
    parser.add_argument("--warmup-runs", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("--test-runs", type=int, default=10, help="Number of test runs")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    
    # Run benchmark
    results = benchmark(batch_sizes, args.warmup_runs, args.test_runs)
    
    # Save results to file
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Benchmark results saved to {{args.output}}")
    
    # Print summary
    print("\\nSummary:")
    for batch_size, batch_results in results.get("results_by_batch", {{}}).items():
        print(f"Batch size {{batch_size}}:")
        print(f"  Average latency: {{batch_results['average_latency_ms']:.2f}} ms")
        print(f"  Throughput: {{batch_results['average_throughput_items_per_second']:.2f}} items/second")
"""
        else:
            return f"""# Placeholder template for {template_type} for {model_name} on {hardware}"""
    
    def run_test(self, temp_dir: str) -> Dict[str, Any]:
        """
        Run tests for the model on the specified hardware.
        
        Args:
            temp_dir: Directory containing generated test files
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running tests for {self.model_name} on {self.hardware}")
        
        # Get file paths
        model_name_safe = self.model_name.replace('/', '_')
        test_file = os.path.join(temp_dir, f"test_{model_name_safe}_{self.hardware}.py")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            logger.error(f"Test file does not exist: {test_file}")
            return {"success": False, "error": "Test file does not exist"}
        
        try:
            # Run the test
            start_time = time.time()
            
            # Create command to run the test
            command = [sys.executable, test_file]
            
            # Use subprocess to run the test
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Extract test results
            success = process.returncode == 0
            stdout = process.stdout
            stderr = process.stderr
            
            # Count tests from unittest output
            test_count = stdout.count("... ok") + stdout.count("... FAIL") + stdout.count("... ERROR")
            
            # Create result dictionary
            test_result = {
                "success": success,
                "execution_time": execution_time,
                "stdout": stdout,
                "stderr": stderr,
                "test_count": test_count,
                "returncode": process.returncode,
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            logger.info(f"Test execution completed in {execution_time:.2f} seconds with result: {'SUCCESS' if success else 'FAILURE'}")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Test execution timed out after {TEST_TIMEOUT} seconds")
            return {
                "success": False,
                "error": f"Timeout after {TEST_TIMEOUT} seconds",
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error during test execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def run_benchmark(self, temp_dir: str) -> Dict[str, Any]:
        """
        Run benchmark for the model on the specified hardware.
        
        Args:
            temp_dir: Directory containing generated benchmark files
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Running benchmark for {self.model_name} on {self.hardware}")
        
        # Get file paths
        model_name_safe = self.model_name.replace('/', '_')
        benchmark_file = os.path.join(temp_dir, f"benchmark_{model_name_safe}_{self.hardware}.py")
        
        # Check if benchmark file exists
        if not os.path.exists(benchmark_file):
            logger.error(f"Benchmark file does not exist: {benchmark_file}")
            return {"success": False, "error": "Benchmark file does not exist"}
        
        # Temporary output file for benchmark results
        benchmark_output = os.path.join(temp_dir, f"benchmark_{model_name_safe}_{self.hardware}_results.json")
        
        try:
            # Create batch sizes list
            batch_sizes = [1] if self.quick_test else [1, 2, 4, 8]
            batch_sizes_str = ",".join(map(str, batch_sizes))
            
            # Set the number of runs
            warmup_runs = 2 if self.quick_test else 5
            test_runs = 3 if self.quick_test else 10
            
            # Create command to run the benchmark
            command = [
                sys.executable, 
                benchmark_file,
                "--batch-sizes", batch_sizes_str,
                "--warmup-runs", str(warmup_runs),
                "--test-runs", str(test_runs),
                "--output", benchmark_output
            ]
            
            # Run the benchmark
            logger.debug(f"Running benchmark command: {' '.join(command)}")
            
            # Start timing
            start_time = time.time()
            
            # Use subprocess to run the benchmark
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT * 3  # Give benchmarks more time
            )
            
            # End timing
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Check if benchmark was successful
            if process.returncode != 0:
                logger.error(f"Benchmark execution failed with code {process.returncode}")
                logger.error(f"Stdout: {process.stdout}")
                logger.error(f"Stderr: {process.stderr}")
                return {
                    "success": False,
                    "error": f"Benchmark execution failed with code {process.returncode}",
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "model_name": self.model_name,
                    "hardware": self.hardware,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            # Load benchmark results from output file
            if os.path.exists(benchmark_output):
                with open(benchmark_output, 'r') as f:
                    benchmark_results = json.load(f)
            else:
                logger.error(f"Benchmark output file does not exist: {benchmark_output}")
                return {
                    "success": False,
                    "error": "Benchmark output file does not exist",
                    "model_name": self.model_name,
                    "hardware": self.hardware,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            # Add metadata
            benchmark_results["execution_time"] = execution_time
            benchmark_results["stdout"] = process.stdout
            benchmark_results["stderr"] = process.stderr
            benchmark_results["success"] = True
            
            logger.info(f"Benchmark execution completed in {execution_time:.2f} seconds")
            
            return {
                "success": True,
                "benchmark_results": benchmark_results,
                "execution_time": execution_time,
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark execution timed out after {TEST_TIMEOUT * 3} seconds")
            return {
                "success": False,
                "error": f"Timeout after {TEST_TIMEOUT * 3} seconds",
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error during benchmark execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def generate_documentation(self, temp_dir: str, 
                             test_results: Optional[Dict[str, Any]] = None,
                             benchmark_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate documentation for the model on the specified hardware.
        
        Args:
            temp_dir: Directory containing generated files
            test_results: Test results to include in documentation
            benchmark_results: Benchmark results to include in documentation
            
        Returns:
            Dictionary with documentation generation results
        """
        if not self.generate_docs:
            logger.info("Documentation generation disabled")
            return {"success": False, "error": "Documentation generation disabled"}
            
        logger.info(f"Generating documentation for {self.model_name} on {self.hardware}")
        
        # Get file paths
        model_name_safe = self.model_name.replace('/', '_')
        skill_file = os.path.join(temp_dir, f"{model_name_safe}_{self.hardware}_skill.py")
        test_file = os.path.join(temp_dir, f"test_{model_name_safe}_{self.hardware}.py")
        benchmark_file = os.path.join(temp_dir, f"benchmark_{model_name_safe}_{self.hardware}.py")
        
        # Check if required files exist
        if not all(os.path.exists(f) for f in [skill_file, test_file, benchmark_file]):
            logger.error("One or more required files do not exist")
            return {"success": False, "error": "Required files do not exist"}
        
        try:
            # Create output directory
            model_doc_dir = os.path.join(DOCS_DIR, model_name_safe)
            os.makedirs(model_doc_dir, exist_ok=True)
            
            # Generate documentation
            doc_path = generate_model_documentation(
                model_name=self.model_name,
                hardware=self.hardware,
                skill_path=skill_file,
                test_path=test_file,
                benchmark_path=benchmark_file,
                expected_results_path=None,
                output_dir=model_doc_dir,
                template_db_path=self.template_db_path
            )
            
            logger.info(f"Documentation generated: {doc_path}")
            
            return {
                "success": True,
                "documentation_path": doc_path,
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during documentation generation: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def store_results(self, 
                     test_results: Dict[str, Any],
                     benchmark_results: Dict[str, Any],
                     doc_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store test, benchmark, and documentation results.
        
        Args:
            test_results: Results from test execution
            benchmark_results: Results from benchmark execution
            doc_results: Results from documentation generation
            
        Returns:
            Dictionary with storage results
        """
        logger.info(f"Storing results for {self.model_name} on {self.hardware}")
        
        # Create directory for this model and hardware combination
        model_name_safe = self.model_name.replace('/', '_')
        expected_dir = os.path.join(EXPECTED_RESULTS_DIR, model_name_safe, self.hardware)
        collected_dir = os.path.join(COLLECTED_RESULTS_DIR, model_name_safe, self.hardware)
        
        # Create timestamped directory for collected results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        collected_timestamp_dir = os.path.join(collected_dir, timestamp)
        
        # Ensure directories exist
        for directory in [expected_dir, collected_dir, collected_timestamp_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Create combined results dictionary
        combined_results = {
            "model_name": self.model_name,
            "hardware": self.hardware,
            "timestamp": datetime.datetime.now().isoformat(),
            "git_hash": self.git_hash,
            "test_results": test_results,
            "benchmark_results": benchmark_results
        }
        
        if doc_results:
            combined_results["documentation_results"] = doc_results
        
        # Define file paths for results
        collected_file = os.path.join(collected_timestamp_dir, "results.json")
        expected_file = os.path.join(expected_dir, "expected_results.json")
        
        # Store collected results
        with open(collected_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        # Update expected results if requested
        if self.update_expected and test_results.get("success", False):
            with open(expected_file, 'w') as f:
                json.dump(combined_results, f, indent=2)
            logger.info(f"Updated expected results: {expected_file}")
        
        # Store results in database if available
        db_results = {}
        if HAS_DB_API and self.db_path:
            try:
                # Determine model family
                model_family = self._determine_model_family()
                
                # Store test results in database
                db_result = store_test_result(
                    self.db_path,
                    model_name=self.model_name,
                    model_family=model_family,
                    hardware=self.hardware,
                    success=test_results.get("success", False),
                    execution_time=test_results.get("execution_time", 0),
                    test_count=test_results.get("test_count", 0),
                    error_message=test_results.get("error", ""),
                    git_hash=self.git_hash,
                    timestamp=datetime.datetime.now().isoformat(),
                    results_json=json.dumps(combined_results)
                )
                
                db_results["db_storage"] = db_result
                logger.info(f"Results stored in database: {self.db_path}")
            except Exception as e:
                logger.error(f"Error storing results in database: {e}")
                db_results["db_storage"] = {"success": False, "error": str(e)}
        
        return {
            "success": True,
            "collected_file": collected_file,
            "expected_file": expected_file if self.update_expected else None,
            "db_results": db_results,
            "model_name": self.model_name,
            "hardware": self.hardware,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def compare_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare test results with expected results.
        
        Args:
            test_results: Results from test execution
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing results for {self.model_name} on {self.hardware}")
        
        # Get expected results file path
        model_name_safe = self.model_name.replace('/', '_')
        expected_file = os.path.join(EXPECTED_RESULTS_DIR, model_name_safe, self.hardware, "expected_results.json")
        
        # Check if expected results exist
        if not os.path.exists(expected_file):
            logger.warning(f"No expected results found: {expected_file}")
            return {
                "success": False,
                "error": "No expected results found",
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        try:
            # Load expected results
            with open(expected_file, 'r') as f:
                expected_results = json.load(f)
            
            # Compare test results with expected results
            comparison = self.result_comparer.compare_results(
                test_results,
                expected_results.get("test_results", {}),
                tolerance=self.tolerance
            )
            
            logger.info(f"Comparison result: {'MATCH' if comparison.get('match', False) else 'MISMATCH'}")
            
            return {
                "success": True,
                "match": comparison.get("match", False),
                "differences": comparison.get("differences", []),
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def run_test_with_docs(self, temp_dir: str) -> Dict[str, Any]:
        """
        Run full test workflow with documentation generation.
        
        Args:
            temp_dir: Directory for generated files
            
        Returns:
            Dictionary with results from all steps
        """
        # Run tests
        test_results = self.run_test(temp_dir)
        
        # Run benchmarks
        benchmark_results = self.run_benchmark(temp_dir)
        
        # Generate documentation if requested
        doc_results = None
        if self.generate_docs:
            doc_results = self.generate_documentation(
                temp_dir,
                test_results=test_results,
                benchmark_results=benchmark_results.get("benchmark_results", {})
            )
        
        # Store results
        storage_results = self.store_results(
            test_results=test_results,
            benchmark_results=benchmark_results,
            doc_results=doc_results
        )
        
        # Compare results with expected results
        comparison_results = self.compare_results(test_results)
        
        # Return combined results
        return {
            "success": test_results.get("success", False) and benchmark_results.get("success", False),
            "test_results": test_results,
            "benchmark_results": benchmark_results,
            "doc_results": doc_results,
            "storage_results": storage_results,
            "comparison_results": comparison_results,
            "model_name": self.model_name,
            "hardware": self.hardware,
            "timestamp": datetime.datetime.now().isoformat(),
            "temp_dir": temp_dir if self.keep_temp else None
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete testing workflow:
        1. Generate components
        2. Run tests
        3. Run benchmarks
        4. Generate documentation
        5. Store results
        6. Compare with expected results
        
        Returns:
            Dictionary with results from all steps
        """
        logger.info(f"Running unified testing for {self.model_name} on {self.hardware}")
        
        # Create temporary directory for generated files
        temp_dir = tempfile.mkdtemp(prefix=f"unified_test_{self.model_name.replace('/', '_')}_{self.hardware}_")
        
        try:
            # Generate components
            skill_file, test_file, benchmark_file = self.generate_components(temp_dir)
            
            # Run tests with documentation
            result = self.run_test_with_docs(temp_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in unified testing: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat()
            }
        finally:
            # Clean up temporary directory
            if not self.keep_temp and temp_dir:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            elif self.keep_temp:
                logger.info(f"Keeping temporary directory: {temp_dir}")


def run_unified_test(model_name: str,
                    hardware: str,
                    db_path: Optional[str] = None,
                    template_db_path: Optional[str] = None,
                    update_expected: bool = False,
                    generate_docs: bool = False,
                    quick_test: bool = False,
                    keep_temp: bool = False,
                    verbose: bool = False,
                    tolerance: float = 0.01,
                    output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a unified test for a single model and hardware combination.
    
    Args:
        model_name: Name of the model to test
        hardware: Hardware platform to test on
        db_path: Path to DuckDB database for storing results
        template_db_path: Path to template database
        update_expected: Whether to update expected results with current test results
        generate_docs: Whether to generate documentation for the model
        quick_test: Whether to run a quick test with minimal validation
        keep_temp: Whether to keep temporary directories after tests
        verbose: Whether to output verbose logs
        tolerance: Tolerance for numeric comparisons (e.g., 0.01 for 1%)
        output_dir: Custom output directory for results
        
    Returns:
        Dictionary with test results
    """
    tester = UnifiedComponentTester(
        model_name=model_name,
        hardware=hardware,
        db_path=db_path,
        template_db_path=template_db_path,
        update_expected=update_expected,
        generate_docs=generate_docs,
        quick_test=quick_test,
        keep_temp=keep_temp,
        verbose=verbose,
        tolerance=tolerance,
        output_dir=output_dir
    )
    
    return tester.run()


def run_batch_tests(models: List[str],
                   hardware_platforms: List[str],
                   db_path: Optional[str] = None,
                   template_db_path: Optional[str] = None,
                   update_expected: bool = False,
                   generate_docs: bool = False,
                   quick_test: bool = False,
                   keep_temp: bool = False,
                   verbose: bool = False,
                   tolerance: float = 0.01,
                   max_workers: int = 1,
                   output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run batch tests for multiple models and hardware platforms.
    
    Args:
        models: List of model names to test
        hardware_platforms: List of hardware platforms to test on
        db_path: Path to DuckDB database for storing results
        template_db_path: Path to template database
        update_expected: Whether to update expected results with current test results
        generate_docs: Whether to generate documentation for the model
        quick_test: Whether to run a quick test with minimal validation
        keep_temp: Whether to keep temporary directories after tests
        verbose: Whether to output verbose logs
        tolerance: Tolerance for numeric comparisons (e.g., 0.01 for 1%)
        max_workers: Maximum number of parallel workers (1 for sequential execution)
        output_dir: Custom output directory for results
        
    Returns:
        Dictionary with summary of test results
    """
    all_results = []
    start_time = time.time()
    
    # Create combinations of model and hardware
    combinations = [(model, hw) for model in models for hw in hardware_platforms]
    logger.info(f"Running tests for {len(combinations)} model-hardware combinations")
    
    # Run tests in parallel or sequentially
    if max_workers > 1 and len(combinations) > 1:
        logger.info(f"Running tests in parallel with {max_workers} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_combo = {
                executor.submit(
                    run_unified_test,
                    model_name=model,
                    hardware=hw,
                    db_path=db_path,
                    template_db_path=template_db_path,
                    update_expected=update_expected,
                    generate_docs=generate_docs,
                    quick_test=quick_test,
                    keep_temp=keep_temp,
                    verbose=verbose,
                    tolerance=tolerance,
                    output_dir=output_dir
                ): (model, hw)
                for model, hw in combinations
            }
            
            for future in concurrent.futures.as_completed(future_to_combo):
                model, hw = future_to_combo[future]
                try:
                    result = future.result()
                    result["model_name"] = model
                    result["hardware"] = hw
                    all_results.append(result)
                    logger.info(f"Completed test for {model} on {hw}: {'SUCCESS' if result.get('success', False) else 'FAILURE'}")
                except Exception as e:
                    logger.error(f"Test for {model} on {hw} raised an exception: {e}")
                    all_results.append({
                        "success": False,
                        "error": str(e),
                        "model_name": model,
                        "hardware": hw,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
    else:
        logger.info("Running tests sequentially")
        
        for model, hw in combinations:
            try:
                logger.info(f"Testing {model} on {hw}")
                result = run_unified_test(
                    model_name=model,
                    hardware=hw,
                    db_path=db_path,
                    template_db_path=template_db_path,
                    update_expected=update_expected,
                    generate_docs=generate_docs,
                    quick_test=quick_test,
                    keep_temp=keep_temp,
                    verbose=verbose,
                    tolerance=tolerance,
                    output_dir=output_dir
                )
                all_results.append(result)
                logger.info(f"Completed test for {model} on {hw}: {'SUCCESS' if result.get('success', False) else 'FAILURE'}")
            except Exception as e:
                logger.error(f"Test for {model} on {hw} raised an exception: {e}")
                all_results.append({
                    "success": False,
                    "error": str(e),
                    "model_name": model,
                    "hardware": hw,
                    "timestamp": datetime.datetime.now().isoformat()
                })
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Count successes and failures
    success_count = sum(1 for r in all_results if r.get("success", False))
    failure_count = len(all_results) - success_count
    
    # Create summary report
    summary = {
        "total_tests": len(all_results),
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / len(all_results) if all_results else 0,
        "execution_time": execution_time,
        "timestamp": datetime.datetime.now().isoformat(),
        "results": all_results
    }
    
    # Save summary report
    summary_dir = os.path.join(COLLECTED_RESULTS_DIR, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.path.join(summary_dir, f"unified_test_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary report saved to {summary_file}")
    logger.info(f"Test execution completed in {execution_time:.2f} seconds")
    logger.info(f"Success rate: {success_count}/{len(all_results)} ({(success_count / len(all_results) * 100):.2f}%)")
    
    return summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified Component Tester for IPFS Accelerate")
    
    # Model and hardware selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Model name to test")
    model_group.add_argument("--model-family", choices=MODEL_FAMILIES.keys(), help="Model family to test")
    model_group.add_argument("--all-models", action="store_true", help="Test all models")
    
    hardware_group = parser.add_mutually_exclusive_group(required=True)
    hardware_group.add_argument("--hardware", help="Hardware platform(s) to test (comma-separated)")
    hardware_group.add_argument("--priority-hardware", action="store_true", help="Test on priority hardware platforms")
    hardware_group.add_argument("--all-hardware", action="store_true", help="Test on all hardware platforms")
    
    # Database options
    parser.add_argument("--db-path", help="Path to DuckDB database")
    parser.add_argument("--no-db", action="store_true", help="Disable database integration")
    parser.add_argument("--template-db-path", help="Path to template database")
    
    # Testing options
    parser.add_argument("--update-expected", action="store_true", help="Update expected results with current results")
    parser.add_argument("--generate-docs", action="store_true", help="Generate documentation for models")
    parser.add_argument("--quick-test", action="store_true", help="Run quick tests with minimal validation")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary directories after tests")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Tolerance for numeric comparisons")
    
    # Execution options
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument("--output-dir", help="Custom output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Clean up options
    parser.add_argument("--clean-old-results", action="store_true", help="Clean up old collected results")
    parser.add_argument("--days", type=int, default=14, help="Age of results to clean up in days")
    
    # Compare templates option
    parser.add_argument("--compare-templates", action="store_true", help="Compare generated components with templates")
    
    return parser.parse_args()


def clean_old_results(days: int) -> Dict[str, Any]:
    """
    Clean up old collected results.
    
    Args:
        days: Age of results to clean up in days
        
    Returns:
        Dictionary with cleanup results
    """
    logger.info(f"Cleaning up collected results older than {days} days")
    
    # Calculate cutoff time
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    
    # Walk through collected results directory
    total_count = 0
    removed_count = 0
    
    for root, dirs, files in os.walk(COLLECTED_RESULTS_DIR):
        for dir_name in dirs:
            # Skip if not a timestamp directory
            if not dir_name[0].isdigit():
                continue
                
            dir_path = os.path.join(root, dir_name)
            dir_time = os.path.getmtime(dir_path)
            
            total_count += 1
            
            # Check if directory is older than cutoff time
            if dir_time < cutoff_time:
                try:
                    shutil.rmtree(dir_path)
                    removed_count += 1
                    logger.debug(f"Removed old results directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Error removing directory {dir_path}: {e}")
    
    logger.info(f"Cleaned up {removed_count} of {total_count} results directories")
    
    return {
        "success": True,
        "total_count": total_count,
        "removed_count": removed_count,
        "cutoff_days": days,
        "timestamp": datetime.datetime.now().isoformat()
    }


def main():
    """Main function"""
    args = parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Clean up old results if requested
    if args.clean_old_results:
        clean_old_results(args.days)
        return
    
    # Set up database path
    db_path = None
    if not args.no_db:
        db_path = args.db_path or DEFAULT_DB_PATH
    
    # Determine models to test
    if args.model:
        models = [args.model]
    elif args.model_family:
        models = MODEL_FAMILIES.get(args.model_family, [])
    elif args.all_models:
        models = [model for models in MODEL_FAMILIES.values() for model in models]
    
    # Determine hardware platforms to test
    if args.hardware:
        hardware_platforms = args.hardware.split(',')
    elif args.priority_hardware:
        hardware_platforms = PRIORITY_HARDWARE
    elif args.all_hardware:
        hardware_platforms = SUPPORTED_HARDWARE
    
    # Run batch tests
    summary = run_batch_tests(
        models=models,
        hardware_platforms=hardware_platforms,
        db_path=db_path,
        template_db_path=args.template_db_path,
        update_expected=args.update_expected,
        generate_docs=args.generate_docs,
        quick_test=args.quick_test,
        keep_temp=args.keep_temp,
        verbose=args.verbose,
        tolerance=args.tolerance,
        max_workers=args.max_workers,
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\nTest Execution Summary:")
    print("-----------------------")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Successful tests: {summary['success_count']}")
    print(f"Failed tests: {summary['failure_count']}")
    print(f"Success rate: {summary['success_rate']*100:.2f}%")
    print(f"Execution time: {summary['execution_time']:.2f} seconds")
    
    # List any failures
    if summary['failure_count'] > 0:
        print("\nFailed tests:")
        for result in summary['results']:
            if not result.get('success', False):
                print(f"- {result['model_name']} on {result['hardware']}: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()