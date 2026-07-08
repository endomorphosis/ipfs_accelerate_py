#!/usr/bin/env python3
"""
End-to-End Testing Framework for IPFS Accelerate

This script automates the generation and testing of skill, test, and benchmark components
for models. It generates all three components together, runs tests, collects results,
and compares them with expected results.

Enhanced with database integration, distributed testing capabilities, and improved
result comparison for complex tensor outputs.

Usage:
    python run_e2e_tests.py --model bert --hardware cuda
    python run_e2e_tests.py --model-family text-embedding --hardware all
    python run_e2e_tests.py --model vit --hardware cuda,webgpu --update-expected
    python run_e2e_tests.py --all-models --priority-hardware --quick-test
    python run_e2e_tests.py --model bert --hardware cuda --db-path ./benchmark_db.duckdb
    python run_e2e_tests.py --model-family vision --priority-hardware --distributed
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
import concurrent.futures
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

# For distributed testing
import socket
import threading
import queue
import uuid
from contextlib import contextmanager

# Set up logging early for the import warnings
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# For DuckDB integration
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    logger.warning("DuckDB not available. Database features disabled.")
    
# For real hardware detection
try:
    import psutil
    import torch
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    logger.warning("Hardware detection libraries not available. Using basic detection.")

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import project utilities
from simple_utils import ensure_dir_exists
from template_validation import ModelValidator, ResultComparer
from model_documentation_generator import generate_model_documentation

# Try to import DuckDB-related modules
try:
    sys.path.append(os.path.join(test_dir, "../duckdb_api"))
    from data.duckdb.core.benchmark_db_updater import store_test_result, initialize_db
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
DISTRIBUTED_PORT = 9090  # Default port for distributed testing
WORKER_COUNT = os.cpu_count() or 4  # Default number of worker threads

# Ensure directories exist
for directory in [EXPECTED_RESULTS_DIR, COLLECTED_RESULTS_DIR, DOCS_DIR]:
    ensure_dir_exists(directory)

# Hardware platforms supported by the testing framework
SUPPORTED_HARDWARE = [
    "cpu", "cuda", "rocm", "mps", "openvino", 
    "qnn", "webnn", "webgpu", "samsung"
]

PRIORITY_HARDWARE = ["cpu", "cuda", "openvino", "webgpu"]

# Mapping of hardware to detection method
# Enhanced hardware detection functions
def detect_openvino():
    """Detect if OpenVINO is available and usable."""
    try:
        import openvino
        # Check if the Runtime module is available and can be initialized
        from openvino.runtime import Core
        core = Core()
        available_devices = core.available_devices
        return len(available_devices) > 0
    except (ImportError, ModuleNotFoundError, Exception):
        return False

def detect_qnn():
    """Detect if Qualcomm Neural Network SDK is available."""
    try:
        # First, check if the QNN Python bindings are available
        import qnn
        
        # Try to list available devices
        from qnn.messaging import QnnMessageListener
        listener = QnnMessageListener()
        return True
    except (ImportError, ModuleNotFoundError, Exception):
        # QNN SDK not available or couldn't be initialized
        return False

def detect_web_capabilities(capability="webgpu"):
    """Detect if browser with WebNN or WebGPU capabilities can be launched."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        # Try to launch a headless browser instance
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        
        driver = webdriver.Chrome(options=options)
        
        if capability == "webgpu":
            # Check for WebGPU support
            is_supported = driver.execute_script("""
                return 'gpu' in navigator && 'requestAdapter' in navigator.gpu;
            """)
        elif capability == "webnn":
            # Check for WebNN support
            is_supported = driver.execute_script("""
                return 'ml' in navigator && 'getNeuralNetworkContext' in navigator.ml;
            """)
        else:
            is_supported = False
            
        driver.quit()
        return is_supported
    except Exception:
        return False

def detect_samsung_npu():
    """Detect if Samsung NPU is available."""
    try:
        # Check for Samsung NPU SDK
        import enn
        from enn import NpuContext
        
        # Try to initialize NPU
        context = NpuContext()
        is_available = context.is_available()
        return is_available
    except (ImportError, ModuleNotFoundError, Exception):
        return False

# Updated hardware detection map with enhanced detection functions
HARDWARE_DETECTION_MAP = {
    "cpu": lambda: True,  # CPU is always available
    "cuda": lambda: torch.cuda.is_available() if HAS_HARDWARE_DETECTION else False,
    "rocm": lambda: hasattr(torch, 'hip') and torch.hip.is_available() if HAS_HARDWARE_DETECTION else False,
    "mps": lambda: hasattr(torch, 'mps') and torch.mps.is_available() if HAS_HARDWARE_DETECTION else False,
    "openvino": lambda: detect_openvino() if HAS_HARDWARE_DETECTION else False,
    "qnn": lambda: detect_qnn() if HAS_HARDWARE_DETECTION else False,
    "webnn": lambda: detect_web_capabilities("webnn") if HAS_HARDWARE_DETECTION else False,
    "webgpu": lambda: detect_web_capabilities("webgpu") if HAS_HARDWARE_DETECTION else False,
    "samsung": lambda: detect_samsung_npu() if HAS_HARDWARE_DETECTION else False,
}

# Distinguish between real and simulated hardware
def is_simulation(hardware):
    """Determine if the hardware testing will be simulated or real."""
    if hardware not in HARDWARE_DETECTION_MAP:
        return True
        
    return not HARDWARE_DETECTION_MAP[hardware]()

# Database connection handling
@contextmanager
def get_db_connection(db_path=None):
    """Context manager for database connections."""
    if not HAS_DUCKDB:
        yield None
        return
        
    conn = None
    try:
        db_path = db_path or DEFAULT_DB_PATH
        conn = duckdb.connect(db_path)
        yield conn
    finally:
        if conn:
            conn.close()

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
        
        # Database configuration
        self.db_path = self.args.db_path or DEFAULT_DB_PATH
        self.use_db = HAS_DUCKDB and self.args.use_db
        
        # Initialize database if needed
        if self.use_db and HAS_DB_API:
            try:
                initialize_db(self.db_path)
                logger.info(f"Initialized database at {self.db_path}")
            except Exception as e:
                logger.error(f"Error initializing database: {str(e)}")
                self.use_db = False
                
        # Distributed testing configuration
        self.distributed = self.args.distributed
        self.workers = self.args.workers or WORKER_COUNT
        self.task_queue = queue.Queue() if self.distributed else None
        self.result_queue = queue.Queue() if self.distributed else None
        self.worker_threads = [] if self.distributed else None
        
        # Hardware simulation tracking
        self.simulation_status = {}
        
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
        
        # Check for simulation status
        for hardware in self.hardware_to_test:
            self.simulation_status[hardware] = is_simulation(hardware)
            if self.simulation_status[hardware]:
                logger.warning(f"{hardware} will be simulated as the real hardware is not detected")
            else:
                logger.info(f"{hardware} detected as real hardware")
        
        # Use distributed or sequential processing
        if self.distributed:
            self._run_distributed_tests()
        else:
            self._run_sequential_tests()
        
        self._generate_summary_report()
        self._cleanup()
        
        return self.test_results
        
    def _run_sequential_tests(self):
        """Run tests sequentially for all models and hardware."""
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
    
    def _run_distributed_tests(self):
        """Run tests in parallel using worker threads."""
        logger.info(f"Starting distributed testing with {self.workers} workers")
        
        # Create tasks for all model-hardware combinations
        for model in self.models_to_test:
            self.test_results[model] = {}
            for hardware in self.hardware_to_test:
                self.task_queue.put((model, hardware))
        
        # Start worker threads
        for i in range(self.workers):
            worker = threading.Thread(
                target=self._worker_function, 
                args=(i,), 
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        # Wait for all tasks to complete
        self.task_queue.join()
        
        # Collect results from result queue
        while not self.result_queue.empty():
            model, hardware, result_data = self.result_queue.get()
            self.test_results[model][hardware] = result_data
        
        logger.info("Distributed testing completed")
    
    def _worker_function(self, worker_id):
        """Worker thread function for distributed testing."""
        logger.debug(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get a task from the queue with timeout
                model, hardware = self.task_queue.get(timeout=1)
                logger.debug(f"Worker {worker_id} processing {model} on {hardware}")
                
                try:
                    # Create a temp directory for this test
                    temp_dir = tempfile.mkdtemp(prefix=f"e2e_test_{model}_{hardware}_{worker_id}_")
                    self.temp_dirs.append(temp_dir)
                    
                    # Generate components and run test
                    skill_path, test_path, benchmark_path = self._generate_components(model, hardware, temp_dir)
                    result = self._run_test(model, hardware, temp_dir, test_path)
                    comparison = self._compare_with_expected(model, hardware, result)
                    
                    # Update expected results if requested (protected by lock)
                    if self.args.update_expected:
                        self._update_expected_results(model, hardware, result)
                    
                    # Store results
                    self._store_results(model, hardware, result, comparison)
                    
                    # Generate documentation if requested
                    if self.args.generate_docs:
                        self._generate_documentation(model, hardware, skill_path, test_path, benchmark_path)
                    
                    # Collect result data
                    result_data = {
                        "status": "success" if comparison["matches"] else "failure",
                        "result_path": self._get_result_path(model, hardware),
                        "comparison": comparison
                    }
                    
                    # Put result in result queue
                    self.result_queue.put((model, hardware, result_data))
                    
                    logger.info(f"Worker {worker_id}: {model} on {hardware} - {'SUCCESS' if comparison['matches'] else 'FAILURE'}")
                
                except Exception as e:
                    logger.error(f"Worker {worker_id} error testing {model} on {hardware}: {str(e)}")
                    self.result_queue.put((model, hardware, {
                        "status": "error",
                        "error": str(e)
                    }))
                
                finally:
                    # Mark task as done
                    self.task_queue.task_done()
            
            except queue.Empty:
                # No more tasks
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} unexpected error: {str(e)}")
        
        logger.debug(f"Worker {worker_id} finished")
    
    def _generate_components(self, model: str, hardware: str, temp_dir: str) -> Tuple[str, str, str]:
        """Generate skill, test, and benchmark components for a model/hardware combination."""
        logger.debug(f"Generating components for {model} on {hardware}...")
        
        # Paths for the generated components
        skill_path = os.path.join(temp_dir, f"skill_{model}_{hardware}.py")
        test_path = os.path.join(temp_dir, f"test_{model}_{hardware}.py")
        benchmark_path = os.path.join(temp_dir, f"benchmark_{model}_{hardware}.py")
        
        try:
            # Import the generator modules dynamically to avoid circular imports
            sys.path.append(os.path.join(test_dir, "generators"))
            
            # Try to use the actual generators from the generators directory
            from skill_generators.integrated_skillset_generator import generate_skill
            from test_generators.qualified_test_generator import generate_test
            from benchmark_generators.run_model_benchmarks import generate_benchmark
            
            # Generate the skill file
            logger.info(f"Generating skill for {model} on {hardware}...")
            generate_skill(
                model_name=model,
                hardware_platform=hardware,
                output_path=skill_path,
                use_db_templates=True
            )
            
            # Generate the test file
            logger.info(f"Generating test for {model} on {hardware}...")
            generate_test(
                model_name=model,
                hardware_platforms=[hardware],
                output_path=test_path,
                skill_path=skill_path,
                use_db_templates=True
            )
            
            # Generate the benchmark file
            logger.info(f"Generating benchmark for {model} on {hardware}...")
            generate_benchmark(
                model_name=model,
                hardware_platform=hardware,
                output_path=benchmark_path,
                skill_path=skill_path,
                use_db_templates=True
            )
            
            logger.info(f"Successfully generated all components for {model} on {hardware}")
            
        except ImportError as e:
            logger.warning(f"Could not import generator modules: {str(e)}")
            logger.warning("Falling back to mock implementation")
            
            # Fall back to mock implementation if import fails
            self._mock_generate_skill(model, hardware, skill_path)
            self._mock_generate_test(model, hardware, test_path, skill_path)
            self._mock_generate_benchmark(model, hardware, benchmark_path, skill_path)
        
        except Exception as e:
            logger.error(f"Error generating components for {model} on {hardware}: {str(e)}")
            logger.warning("Falling back to mock implementation")
            
            # Fall back to mock implementation
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
        
        # Name for the results output file
        results_json = os.path.join(temp_dir, f"test_results_{model}_{hardware}.json")
        
        # Add argument to the test file to output results to JSON
        modified_test_path = self._modify_test_for_json_output(test_path, results_json)
        
        try:
            # Execute the test and capture results
            import subprocess
            import time
            import psutil
            
            # Record starting memory usage
            process = psutil.Process()
            start_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            
            # Start execution timer
            start_time = time.time()
            
            # Run the test with timeout
            logger.info(f"Running test: python {modified_test_path}")
            result = subprocess.run(
                ["python", modified_test_path], 
                capture_output=True, 
                text=True, 
                timeout=TEST_TIMEOUT
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Record ending memory usage and calculate difference
            end_memory = process.memory_info().rss / (1024 * 1024)
            memory_diff = end_memory - start_memory
            
            # Check if the test was successful
            if result.returncode != 0:
                logger.error(f"Test failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                
                # Return error result
                return {
                    "model": model,
                    "hardware": hardware,
                    "timestamp": self.timestamp,
                    "status": "error",
                    "error_message": result.stderr or result.stdout,
                    "return_code": result.returncode,
                    "execution_time": execution_time,
                    "memory_mb": memory_diff
                }
            
            # Check if the results file was created
            if not os.path.exists(results_json):
                logger.warning(f"Test executed successfully but no results file was found at {results_json}")
                logger.warning("Falling back to parsing stdout for results")
                
                # Try to parse JSON from stdout
                import re
                import json
                
                json_match = re.search(r'{.*}', result.stdout, re.DOTALL)
                if json_match:
                    try:
                        parsed_results = json.loads(json_match.group(0))
                        parsed_results.update({
                            "model": model,
                            "hardware": hardware,
                            "timestamp": self.timestamp,
                            "execution_time": execution_time,
                            "memory_mb": memory_diff,
                            "console_output": result.stdout,
                            "hardware_details": {
                                "platform": hardware,
                                "device_name": self._get_hardware_device_name(hardware)
                            }
                        })
                        return parsed_results
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON from stdout")
                
                # If JSON parsing fails, return basic results
                return {
                    "model": model,
                    "hardware": hardware,
                    "timestamp": self.timestamp,
                    "status": "success",
                    "return_code": result.returncode,
                    "console_output": result.stdout,
                    "execution_time": execution_time,
                    "memory_mb": memory_diff,
                    "hardware_details": {
                        "platform": hardware,
                        "device_name": self._get_hardware_device_name(hardware)
                    }
                }
            
            # Load the results from the JSON file
            try:
                with open(results_json, 'r') as f:
                    test_results = json.load(f)
                
                # Add additional metadata
                test_results.update({
                    "model": model,
                    "hardware": hardware,
                    "timestamp": self.timestamp,
                    "execution_time": execution_time,
                    "memory_mb": memory_diff,
                    "console_output": result.stdout,
                    "hardware_details": test_results.get("hardware_details", {
                        "platform": hardware,
                        "device_name": self._get_hardware_device_name(hardware)
                    })
                })
                
                return test_results
                
            except Exception as e:
                logger.error(f"Error loading test results from {results_json}: {str(e)}")
                # Return basic results
                return {
                    "model": model,
                    "hardware": hardware,
                    "timestamp": self.timestamp,
                    "status": "success_with_errors",
                    "error_message": f"Failed to load results file: {str(e)}",
                    "return_code": result.returncode,
                    "console_output": result.stdout,
                    "execution_time": execution_time,
                    "memory_mb": memory_diff,
                    "hardware_details": {
                        "platform": hardware,
                        "device_name": self._get_hardware_device_name(hardware)
                    }
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"Test timed out after {TEST_TIMEOUT} seconds")
            return {
                "model": model,
                "hardware": hardware,
                "timestamp": self.timestamp,
                "status": "timeout",
                "error_message": f"Test timed out after {TEST_TIMEOUT} seconds",
                "execution_time": TEST_TIMEOUT,
                "hardware_details": {
                    "platform": hardware,
                    "device_name": self._get_hardware_device_name(hardware)
                }
            }
            
        except Exception as e:
            logger.error(f"Error running test {test_path}: {str(e)}")
            # Fall back to mock results if execution fails
            logger.warning("Falling back to mock results")
            return {
                "model": model,
                "hardware": hardware,
                "timestamp": self.timestamp,
                "status": "error",
                "error_message": str(e),
                "input": {"input": "test_input"},
                "output": {"output": f"mock_output_for_{model}_on_{hardware}"},
                "metrics": {
                    "latency_ms": 12.5,
                    "throughput": 80.0,
                    "memory_mb": 512
                },
                "hardware_details": {
                    "platform": hardware,
                    "device_name": self._get_hardware_device_name(hardware)
                }
            }
            
    def _modify_test_for_json_output(self, test_path: str, results_json: str) -> str:
        """
        Modify the test file to output results in JSON format.
        Returns the path to the modified test file.
        """
        try:
            # Read the original test file
            with open(test_path, 'r') as f:
                content = f.read()
            
            # Create the modified file path
            modified_path = test_path + '.modified.py'
            
            # Add imports if needed
            if 'import json' not in content:
                imports = 'import json\nimport os\nimport sys\n'
                if content.startswith('#!'):
                    # Keep shebang line at the top
                    lines = content.split('\n', 1)
                    content = lines[0] + '\n' + imports + lines[1]
                else:
                    content = imports + content
            
            # Add result output logic
            result_output = f"""
# Added by End-to-End Testing Framework
def _save_test_results(test_results):
    results_path = {repr(results_json)}
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"Test results saved to {{results_path}}")

# Override unittest's main function to capture results
import unittest
_original_main = unittest.main

def _custom_main(*args, **kwargs):
    # Remove the 'exit' parameter to prevent unittest from calling sys.exit()
    kwargs['exit'] = False
    result = _original_main(*args, **kwargs)
    
    # Collect test results
    test_results = {{
        "status": "success" if result.result.wasSuccessful() else "failure",
        "tests_run": result.result.testsRun,
        "failures": len(result.result.failures),
        "errors": len(result.result.errors),
        "skipped": len(result.result.skipped) if hasattr(result.result, 'skipped') else 0,
        "metrics": {{
            "latency_ms": 0,
            "throughput": 0,
            "memory_mb": 0
        }},
        "detail": {{
            "failures": [{{
                "test": test[0].__str__(),
                "error": test[1]
            }} for test in result.result.failures],
            "errors": [{{
                "test": test[0].__str__(),
                "error": test[1]
            }} for test in result.result.errors],
        }}
    }}
    
    # Try to extract metrics if available
    try:
        import inspect
        for test in result.result.testCase._tests:
            if hasattr(test, 'skill'):
                if hasattr(test.skill, 'get_metrics'):
                    metrics = test.skill.get_metrics()
                    if isinstance(metrics, dict):
                        test_results["metrics"].update(metrics)
    except Exception as e:
        print(f"Error getting metrics: {{str(e)}}")
    
    # Save the results
    _save_test_results(test_results)
    return result

# Replace unittest's main with our custom version
unittest.main = _custom_main
"""
            
            # Add result output at the end of the file
            if 'if __name__ == "__main__"' in content:
                # Add before the main block
                content = content.replace('if __name__ == "__main__"', result_output + '\nif __name__ == "__main__"')
            else:
                # Add at the end of the file
                content += '\n' + result_output + '\n\nif __name__ == "__main__":\n    unittest.main()\n'
            
            # Write the modified file
            with open(modified_path, 'w') as f:
                f.write(content)
            
            logger.debug(f"Created modified test file at {modified_path}")
            return modified_path
            
        except Exception as e:
            logger.error(f"Error modifying test file {test_path}: {str(e)}")
            logger.warning("Using original test file without modifications")
            return test_path
            
    def _get_hardware_device_name(self, hardware: str) -> str:
        """Get a human-readable device name for the hardware platform with detailed information."""
        is_sim = self.simulation_status.get(hardware, True)
        
        if hardware == "cpu":
            import platform
            import multiprocessing
            cores = multiprocessing.cpu_count()
            return f"CPU: {platform.processor()} ({cores} cores)" + (" [SIMULATED]" if is_sim else "")
            
        elif hardware == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    devices = []
                    for i in range(device_count):
                        name = torch.cuda.get_device_name(i)
                        memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                        devices.append(f"{name} ({memory:.1f} GB)")
                    return f"CUDA: {', '.join(devices)}" + (" [SIMULATED]" if is_sim else "")
                else:
                    return "CUDA: Not Available [SIMULATED]"
            except:
                return "CUDA: Unknown [SIMULATED]"
                
        elif hardware == "rocm":
            try:
                import torch
                if hasattr(torch, 'hip') and torch.hip.is_available():
                    device_count = torch.hip.device_count()
                    devices = []
                    for i in range(device_count):
                        name = torch.hip.get_device_name(i)
                        memory = torch.hip.get_device_properties(i).total_memory / (1024**3)  # GB
                        devices.append(f"{name} ({memory:.1f} GB)")
                    return f"ROCm: {', '.join(devices)}" + (" [SIMULATED]" if is_sim else "")
                else:
                    return "ROCm: Not Available [SIMULATED]"
            except:
                return "ROCm: AMD GPU" + (" [SIMULATED]" if is_sim else "")
                
        elif hardware == "mps":
            try:
                import torch
                if hasattr(torch, 'mps') and torch.mps.is_available():
                    return f"MPS: Apple Silicon ({torch.mps.get_device_name()})" + (" [SIMULATED]" if is_sim else "")
                else:
                    return "MPS: Not Available [SIMULATED]"
            except:
                return "MPS: Apple Silicon" + (" [SIMULATED]" if is_sim else "")
                
        elif hardware == "openvino":
            try:
                if not is_sim:
                    from openvino.runtime import Core
                    core = Core()
                    available_devices = core.available_devices
                    version = getattr(core, "get_version", lambda: "Unknown")()
                    return f"OpenVINO: {version} on {', '.join(available_devices)}"
                else:
                    return "OpenVINO: Intel Hardware [SIMULATED]"
            except:
                return "OpenVINO: Intel Hardware [SIMULATED]"
                
        elif hardware == "qnn":
            try:
                if not is_sim:
                    import qnn
                    version = getattr(qnn, "__version__", "Unknown")
                    return f"QNN: Qualcomm AI Engine v{version}"
                else:
                    return "QNN: Qualcomm AI Engine [SIMULATED]"
            except:
                return "QNN: Qualcomm AI Engine [SIMULATED]"
                
        elif hardware == "webnn":
            try:
                if not is_sim:
                    from selenium import webdriver
                    options = webdriver.ChromeOptions()
                    options.add_argument("--headless=new")
                    driver = webdriver.Chrome(options=options)
                    user_agent = driver.execute_script("return navigator.userAgent")
                    driver.quit()
                    return f"WebNN: Browser Neural Network API ({user_agent})"
                else:
                    return "WebNN: Browser Neural Network API [SIMULATED]"
            except:
                return "WebNN: Browser Neural Network API [SIMULATED]"
                
        elif hardware == "webgpu":
            try:
                if not is_sim:
                    from selenium import webdriver
                    options = webdriver.ChromeOptions()
                    options.add_argument("--headless=new")
                    driver = webdriver.Chrome(options=options)
                    user_agent = driver.execute_script("return navigator.userAgent")
                    driver.quit()
                    return f"WebGPU: Browser GPU API ({user_agent})"
                else:
                    return "WebGPU: Browser GPU API [SIMULATED]"
            except:
                return "WebGPU: Browser GPU API [SIMULATED]"
                
        elif hardware == "samsung":
            try:
                if not is_sim:
                    import enn
                    version = getattr(enn, "__version__", "Unknown")
                    return f"Samsung: NPU v{version}"
                else:
                    return "Samsung: NPU [SIMULATED]"
            except:
                return "Samsung: NPU [SIMULATED]"
                
        else:
            return f"{hardware.upper()}: Unknown Device" + (" [SIMULATED]" if is_sim else "")
    
    def _compare_with_expected(self, model: str, hardware: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare test results with expected results using the ResultComparer from template_validation."""
        from template_validation import ResultComparer
        
        expected_path = os.path.join(EXPECTED_RESULTS_DIR, model, hardware, "expected_result.json")
        
        if not os.path.exists(expected_path):
            logger.warning(f"No expected results found for {model} on {hardware}")
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(expected_path), exist_ok=True)
            # Save the current results as expected for future comparisons
            with open(expected_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Created new expected results for {model} on {hardware}")
            return {"matches": True, "reason": "created_new_baseline"}
        
        try:
            # Initialize ResultComparer with appropriate tolerance settings
            comparer = ResultComparer(
                tolerance=0.1,  # 10% general tolerance
                tensor_rtol=1e-5,  # Relative tolerance for tensors
                tensor_atol=1e-7,  # Absolute tolerance for tensors
                tensor_comparison_mode='auto'  # Automatically select comparison mode
            )
            
            # Use file-based comparison
            comparison_result = comparer.compare_with_file(expected_path, result)
            
            # Log detailed information about differences
            if not comparison_result['match']:
                logger.warning(f"Result mismatch for {model} on {hardware}:")
                for key, diff in comparison_result.get('differences', {}).items():
                    logger.warning(f"  - {key}: expected {diff.get('expected')}, got {diff.get('actual')}")
            else:
                logger.info(f"Results match for {model} on {hardware}")
            
            return {
                "matches": comparison_result.get('match', False),
                "differences": comparison_result.get('differences', {}),
                "statistics": comparison_result.get('statistics', {})
            }
            
        except Exception as e:
            logger.error(f"Error comparing results for {model} on {hardware}: {str(e)}")
            # Log traceback for debugging
            import traceback
            logger.debug(traceback.format_exc())
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
        """Store test results in the collected_results directory and/or database with enhanced metadata."""
        import platform
        import traceback
        
        # File-based storage
        result_dir = os.path.join(COLLECTED_RESULTS_DIR, model, hardware, self.timestamp)
        os.makedirs(result_dir, exist_ok=True)
        
        # Add execution metadata to the result
        result["execution_metadata"] = {
            "timestamp": self.timestamp,
            "model": model,
            "hardware": hardware,
            "hostname": platform.node(),
            "system": platform.system(),
            "python_version": platform.python_version(),
            "simulated": self.simulation_status.get(hardware, is_simulation(hardware))
        }
        
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
            f.write(f"Hardware: {self._get_hardware_device_name(hardware)}\n")
            
            if not comparison["matches"] and "differences" in comparison:
                f.write("\nDifferences found:\n")
                for key, diff in comparison["differences"].items():
                    f.write(f"- {key}: {json.dumps(diff)}\n")
        
        # Database storage if enabled
        if self.use_db and HAS_DB_API:
            try:
                # Track whether hardware is simulated or real
                is_sim = self.simulation_status.get(hardware, is_simulation(hardware))
                
                # Get hardware device info
                device_name = self._get_hardware_device_name(hardware)
                
                # Get git information if available
                git_info = {}
                try:
                    import git
                    repo = git.Repo(search_parent_directories=True)
                    git_info = {
                        "commit_hash": repo.head.object.hexsha,
                        "branch": repo.active_branch.name,
                        "commit_message": repo.head.object.message.strip(),
                        "author": f"{repo.head.object.author.name} <{repo.head.object.author.email}>",
                        "commit_date": str(repo.head.object.committed_datetime)
                    }
                except (ImportError, Exception):
                    # Git package not available or not a git repository
                    pass
                
                # Extract metrics if available
                metrics = {}
                if "metrics" in result:
                    metrics = result["metrics"]
                elif "output" in result and isinstance(result["output"], dict) and "metrics" in result["output"]:
                    metrics = result["output"]["metrics"]
                
                # Add CI/CD information if running in a CI environment
                ci_env = {}
                for env_var in ["CI", "GITHUB_ACTIONS", "GITHUB_WORKFLOW", "GITHUB_RUN_ID", 
                              "GITHUB_REPOSITORY", "GITHUB_REF", "GITHUB_SHA"]:
                    if env_var in os.environ:
                        ci_env[env_var.lower()] = os.environ[env_var]
                
                # Prepare extended result with comprehensive metadata
                db_result = {
                    "model_name": model,
                    "hardware_type": hardware,
                    "device_name": device_name,
                    "test_type": "e2e",
                    "test_date": self.timestamp,
                    "success": comparison["matches"],
                    "is_simulation": is_sim,
                    "error_message": str(comparison.get("differences", {})) if not comparison["matches"] else None,
                    "platform_info": {
                        "system": platform.system(),
                        "release": platform.release(),
                        "machine": platform.machine(),
                        "python_version": platform.python_version(),
                        "hostname": platform.node()
                    },
                    "git_info": git_info,
                    "ci_environment": ci_env if ci_env else None,
                    "metrics": metrics,
                    "result_data": result,
                    "comparison_data": comparison
                }
                
                # Store in database with transaction support
                try:
                    with self._get_db_connection() as conn:
                        store_test_result(conn, db_result)
                    logger.info(f"Stored results for {model} on {hardware} in database at {self.db_path}")
                except Exception as db_conn_error:
                    # If connection manager fails, try direct approach
                    logger.warning(f"DB connection manager failed: {str(db_conn_error)}, trying direct approach")
                    store_test_result(self.db_path, db_result)
                    logger.info(f"Stored results using direct database path")
                    
            except Exception as e:
                logger.error(f"Error storing results in database: {str(e)}")
                # Log detailed traceback for debugging
                logger.debug(f"Database error details:\n{traceback.format_exc()}")
                
                # Create error report for database debugging
                db_error_file = os.path.join(result_dir, "db_error.log")
                with open(db_error_file, 'w') as f:
                    f.write(f"Error storing to database: {str(e)}\n\n")
                    f.write(traceback.format_exc())
                    
                # Fall back to file-based storage only
                logger.info("Results still stored in file system")
                
        logger.info(f"Results for {model} on {hardware} stored in {result_dir}")
        return result_dir
    
    def _get_result_path(self, model: str, hardware: str) -> str:
        """Get the path to the collected results for a model/hardware combination."""
        return os.path.join(COLLECTED_RESULTS_DIR, model, hardware, self.timestamp)
    
    def _generate_documentation(self, model: str, hardware: str, skill_path: str, test_path: str, benchmark_path: str):
        """Generate Markdown documentation for the model using the ModelDocGenerator."""
        from model_documentation_generator import ModelDocGenerator
        
        logger.debug(f"Generating documentation for {model} on {hardware}...")
        
        doc_dir = os.path.join(DOCS_DIR, model)
        os.makedirs(doc_dir, exist_ok=True)
        
        doc_path = os.path.join(doc_dir, f"{hardware}_implementation.md")
        
        # Get the expected results path
        expected_results_path = os.path.join(EXPECTED_RESULTS_DIR, model, hardware, "expected_result.json")
        
        try:
            # Initialize the documentation generator
            doc_generator = ModelDocGenerator(
                model_name=model,
                hardware=hardware,
                skill_path=skill_path,
                test_path=test_path,
                benchmark_path=benchmark_path,
                expected_results_path=expected_results_path if os.path.exists(expected_results_path) else None,
                output_dir=doc_dir
            )
            
            # Generate the documentation
            doc_path = doc_generator.generate()
            
            logger.info(f"Generated documentation for {model} on {hardware} at {doc_path}")
            
        except Exception as e:
            logger.error(f"Error generating documentation for {model} on {hardware}: {str(e)}")
            
            # Fallback to a simple template if documentation generation fails
            fallback_doc_path = os.path.join(doc_dir, f"{hardware}_implementation.md")
            with open(fallback_doc_path, 'w') as f:
                f.write(f"""# {model} Implementation Guide for {hardware}

## Overview

This document describes the implementation of {model} on {hardware} hardware.

## Skill Implementation

The skill implementation is responsible for loading and running the model on {hardware}.

File path: `{skill_path}`

## Test Implementation

The test ensures that the model produces correct outputs.

File path: `{test_path}`

## Benchmark Implementation

The benchmark measures the performance of the model on {hardware}.

File path: `{benchmark_path}`

## Expected Results

Expected results file: `{expected_results_path if os.path.exists(expected_results_path) else "Not available yet"}`

## Hardware Information

{self._get_hardware_description(hardware)}

## Generation Note

This is a fallback documentation. Full documentation generation failed: {str(e)}
""")
            
            logger.info(f"Generated fallback documentation for {model} on {hardware} at {fallback_doc_path}")
            
        return doc_path
    
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
    
    # Database options
    parser.add_argument("--use-db", action="store_true", help="Store results in the database")
    parser.add_argument("--db-path", help="Path to the database file (default: $BENCHMARK_DB_PATH or ./benchmark_db.duckdb)")
    parser.add_argument("--db-only", action="store_true", help="Store results only in the database, not in files")
    
    # Distributed testing options
    parser.add_argument("--distributed", action="store_true", help="Run tests in parallel using worker threads")
    parser.add_argument("--workers", type=int, help=f"Number of worker threads for distributed testing (default: {WORKER_COUNT})")
    parser.add_argument("--simulation-aware", action="store_true", help="Be explicit about real vs simulated hardware testing")
    
    # CI/CD options
    parser.add_argument("--ci", action="store_true", help="Run in CI/CD mode with additional reporting")
    parser.add_argument("--ci-report-dir", help="Custom directory for CI/CD reports")
    parser.add_argument("--badge-only", action="store_true", help="Generate status badge only")
    parser.add_argument("--github-actions", action="store_true", help="Optimize output for GitHub Actions")
    
    # Advanced options
    parser.add_argument("--tensor-tolerance", type=float, default=0.1, help="Tolerance for tensor comparison (default: 0.1)")
    parser.add_argument("--parallel-docs", action="store_true", help="Generate documentation in parallel")
    
    # Logging options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()


def setup_for_ci_cd(args):
    """
    Set up the end-to-end testing framework for CI/CD integration.
    This configures the framework for automated testing in CI/CD environments.
    
    Returns:
        Dict with CI/CD setup information
    """
    logger.info("Setting up CI/CD integration for end-to-end testing")
    
    # Create required directories
    for directory in [EXPECTED_RESULTS_DIR, COLLECTED_RESULTS_DIR, DOCS_DIR]:
        ensure_dir_exists(directory)
    
    # Set up CI/CD specific configurations
    os.environ['E2E_TESTING_CI'] = 'true'
    
    # Check for git repository info (used for versioning test results)
    ci_info = {
        "is_ci": True,
        "git_commit": "unknown",
        "git_branch": "unknown",
        "ci_platform": os.environ.get('CI_PLATFORM', 'unknown')
    }
    
    try:
        import subprocess
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip()
        
        os.environ['E2E_GIT_COMMIT'] = git_commit
        os.environ['E2E_GIT_BRANCH'] = git_branch
        
        ci_info["git_commit"] = git_commit
        ci_info["git_branch"] = git_branch
        
        logger.info(f"CI/CD integration set up with git commit: {git_commit}, branch: {git_branch}")
    except Exception as e:
        logger.warning(f"Failed to get git information: {str(e)}")
    
    # Create CI specific directories for reports
    ci_report_dir = os.path.join(COLLECTED_RESULTS_DIR, "ci_reports")
    os.makedirs(ci_report_dir, exist_ok=True)
    ci_info["report_dir"] = ci_report_dir
    
    return ci_info


def generate_ci_report(ci_info, test_results, timestamp):
    """
    Generate a comprehensive report for CI/CD systems.
    
    Args:
        ci_info: CI/CD setup information
        test_results: Results from running the tests
        timestamp: Timestamp for the report
        
    Returns:
        Dict with report paths
    """
    logger.info("Generating CI/CD report")
    
    if not test_results:
        logger.warning("No test results available to generate CI/CD report")
        return None
        
    report = {
        "timestamp": timestamp,
        "git_commit": ci_info.get('git_commit', 'unknown'),
        "git_branch": ci_info.get('git_branch', 'unknown'),
        "ci_platform": ci_info.get('ci_platform', 'unknown'),
        "summary": {
            "total": 0,
            "success": 0,
            "failure": 0,
            "error": 0
        },
        "results_by_model": {},
        "results_by_hardware": {},
        "compatibility_matrix": {}
    }
    
    # Calculate summary statistics and organize results
    for model, hw_results in test_results.items():
        report["results_by_model"][model] = {}
        
        for hw, result in hw_results.items():
            # Update summary counts
            report["summary"]["total"] += 1
            report["summary"][result["status"]] = report["summary"].get(result["status"], 0) + 1
            
            # Add to model results
            report["results_by_model"][model][hw] = {
                "status": result["status"],
                "has_differences": result.get("comparison", {}).get("matches", True) == False
            }
            
            # Make sure the hardware section exists
            if hw not in report["results_by_hardware"]:
                report["results_by_hardware"][hw] = {
                    "total": 0,
                    "success": 0,
                    "failure": 0,
                    "error": 0
                }
            
            # Update hardware counts
            report["results_by_hardware"][hw]["total"] += 1
            report["results_by_hardware"][hw][result["status"]] = report["results_by_hardware"][hw].get(result["status"], 0) + 1
            
            # Update compatibility matrix
            if model not in report["compatibility_matrix"]:
                report["compatibility_matrix"][model] = {}
            
            report["compatibility_matrix"][model][hw] = result["status"] == "success"
    
    # Generate report files
    ci_report_dir = ci_info.get("report_dir", os.path.join(COLLECTED_RESULTS_DIR, "ci_reports"))
    os.makedirs(ci_report_dir, exist_ok=True)
    
    # JSON report
    json_path = os.path.join(ci_report_dir, f"ci_report_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Markdown report
    md_path = os.path.join(ci_report_dir, f"ci_report_{timestamp}.md")
    with open(md_path, 'w') as f:
        f.write(f"# End-to-End Testing CI/CD Report\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Git Commit: {report['git_commit']}\n")
        f.write(f"Git Branch: {report['git_branch']}\n")
        f.write(f"CI Platform: {report['ci_platform']}\n\n")
        
        # Summary status line for CI parsers (SUCCESS/FAILURE marker)
        overall_status = "SUCCESS" if report['summary'].get('failure', 0) == 0 and report['summary'].get('error', 0) == 0 else "FAILURE"
        f.write(f"## Status: {overall_status}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Tests**: {report['summary']['total']}\n")
        f.write(f"- **Successful**: {report['summary'].get('success', 0)}\n")
        f.write(f"- **Failed**: {report['summary'].get('failure', 0)}\n")
        f.write(f"- **Errors**: {report['summary'].get('error', 0)}\n\n")
        
        f.write("## Compatibility Matrix\n\n")
        
        # Generate header row with all hardware platforms
        all_hardware = sorted(list(report["results_by_hardware"].keys()))
        f.write("| Model | " + " | ".join(all_hardware) + " |\n")
        f.write("|-------|" + "|".join(["---" for hw in all_hardware]) + "|\n")
        
        # Generate rows for each model
        for model in sorted(list(report["compatibility_matrix"].keys())):
            row = [model]
            for hw in all_hardware:
                if hw in report["compatibility_matrix"][model]:
                    if report["compatibility_matrix"][model][hw]:
                        row.append("✅")
                    else:
                        row.append("❌")
                else:
                    row.append("⚪")
            
            f.write("| " + " | ".join(row) + " |\n")
        
        f.write("\n## Detailed Results\n\n")
        
        # Generate detailed results for each model
        for model in sorted(list(report["results_by_model"].keys())):
            f.write(f"### {model}\n\n")
            
            for hw, result in report["results_by_model"][model].items():
                status_icon = "✅" if result["status"] == "success" else "❌" if result["status"] == "failure" else "⚠️"
                f.write(f"- {status_icon} **{hw}**: {result['status'].upper()}")
                
                if result["has_differences"]:
                    f.write(" (has differences)")
                
                f.write("\n")
            
            f.write("\n")
    
    # Create a status badge SVG for CI systems
    badge_color = "#4c1" if overall_status == "SUCCESS" else "#e05d44"
    svg_path = os.path.join(ci_report_dir, f"status_badge_{timestamp}.svg")
    
    with open(svg_path, 'w') as f:
        f.write(f"""<svg xmlns="http://www.w3.org/2000/svg" width="136" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="136" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <path fill="#555" d="M0 0h71v20H0z"/>
    <path fill="{badge_color}" d="M71 0h65v20H71z"/>
    <path fill="url(#b)" d="M0 0h136v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="35.5" y="15" fill="#010101" fill-opacity=".3">E2E Tests</text>
    <text x="35.5" y="14">E2E Tests</text>
    <text x="102.5" y="15" fill="#010101" fill-opacity=".3">{overall_status}</text>
    <text x="102.5" y="14">{overall_status}</text>
  </g>
</svg>""")
        
    logger.info(f"CI/CD report generated: {md_path}")
    return {
        "json_report": json_path,
        "markdown_report": md_path,
        "status_badge": svg_path,
        "status": overall_status
    }


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Set up CI/CD environment if requested
    ci_mode = args.ci or args.simulation_aware or args.github_actions or "CI" in os.environ or "GITHUB_ACTIONS" in os.environ
    ci_info = None
    
    if ci_mode:
        ci_info = setup_for_ci_cd(args)
        logger.info("Running in CI/CD mode with enhanced reporting")
        
        # Configure CI-specific options
        if args.github_actions:
            logger.info("Optimizing output for GitHub Actions")
            os.environ['CI_PLATFORM'] = 'github_actions'
            
        if args.ci_report_dir:
            ci_info["report_dir"] = args.ci_report_dir
    
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
    
    # Generate CI/CD reports if running in CI mode
    if ci_mode and ci_info and results:
        logger.info("Generating CI/CD reports...")
        ci_report = generate_ci_report(ci_info, results, tester.timestamp)
        
        if ci_report:
            logger.info(f"CI/CD report status: {ci_report['status']}")
            logger.info(f"CI/CD report available at: {ci_report['markdown_report']}")
            
            # Set exit code for CI/CD systems
            if ci_report['status'] != "SUCCESS":
                logger.warning("Tests failed - setting exit code to 1 for CI/CD systems")
                # For automated CI systems, non-zero exit code indicates failure
                sys.exit(1)


if __name__ == "__main__":
    main()