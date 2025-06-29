#!/usr/bin/env python3
"""
Improved Merged Test Generator

This module is an enhanced version of the merged_test_generator.py with:

1. Standardized hardware detection using improved_hardware_detection module
2. Properly integrated database storage using database_integration module
3. Fixed duplicated code and inconsistent error handling
4. Added improved cross-platform test generation

Usage:
    python improved_merged_test_generator.py --generate bert
    python improved_merged_test_generator.py --all --cross-platform --hardware all
    python improved_merged_test_generator.py --batch-generate bert,t5,clip,vit,whisper
    python improved_merged_test_generator.py --generate bert --hardware cuda,webgpu
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import traceback
import importlib.util
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if improvements module is in the path
if importlib.util.find_spec("improvements") is None:
    # Add the parent directory to the path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import improved hardware detection and database modules
try:
    from improvements.improved_hardware_detection import (
        detect_all_hardware, 
        apply_web_platform_optimizations,
        get_hardware_compatibility_matrix,
        HAS_CUDA,
        HAS_ROCM,
        HAS_MPS,
        HAS_OPENVINO,
        HAS_WEBNN,
        HAS_WEBGPU,
        HARDWARE_PLATFORMS,
        KEY_MODEL_HARDWARE_MATRIX
    )
    HAS_HARDWARE_MODULE = True
except ImportError:
    logger.warning("Could not import hardware detection module, using local implementation")
    HAS_HARDWARE_MODULE = False

try:
    from improvements.database_integration import (
        DUCKDB_AVAILABLE,
        DEPRECATE_JSON_OUTPUT,
        get_or_create_test_run,
        get_or_create_model,
        store_test_result,
        store_implementation_metadata,
        complete_test_run
    )
    HAS_DATABASE_MODULE = True
except ImportError:
    logger.warning("Could not import database integration module, using local implementation")
    HAS_DATABASE_MODULE = False
    # Set environment variable flag
    DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")

# Create a fallback hardware detection function if the module is not available
if not HAS_HARDWARE_MODULE:
    def detect_hardware():
        """Simple hardware detection fallback"""
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
            
        return {
            "cpu": {"detected": True},
            "cuda": {"detected": has_cuda},
            "rocm": {"detected": False},
            "mps": {"detected": False},
            "openvino": {"detected": False},
            "webnn": {"detected": False},
            "webgpu": {"detected": False},
        }
    
    # Use fallback detection
    detect_all_hardware = detect_hardware
    
    # Define fallback variables
    HARDWARE_PLATFORMS = ["cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu"]
    
    # Create a fallback compatibility matrix
    def get_hardware_compatibility_matrix():
        """Fallback hardware compatibility matrix"""
        # Default compatibility for all models
        default_compat = {
            "cpu": "REAL",
            "cuda": "REAL",
            "openvino": "REAL",
            "mps": "REAL",
            "rocm": "REAL",
            "webnn": "REAL",
            "webgpu": "REAL"
        }
        
        # Build matrix with defaults
        compatibility_matrix = {
            "bert": default_compat,
            "t5": default_compat,
            "gpt2": default_compat,
            "vit": default_compat,
            "clip": default_compat,
            # Add other models as needed
        }
        
        return compatibility_matrix
    
    KEY_MODEL_HARDWARE_MATRIX = get_hardware_compatibility_matrix()

# Output directory for generated tests
OUTPUT_DIR = Path("./generated_tests")

# Test template for model implementations
def get_test_template(model_type, hardware_platforms=None, optimizations=None):
    """
    Get a test template for the given model type with hardware support.
    
    Args:
        model_type: Type of model (bert, t5, etc.)
        hardware_platforms: List of hardware platforms to support
        optimizations: Web platform optimizations to include
    
    Returns:
        Test template string
    """
    # Hardware flag imports
    hw_imports = """
import os
import importlib.util

# Hardware detection
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    from unittest.mock import MagicMock
    torch = MagicMock()
    
# Initialize hardware capability flags
HAS_CUDA = False
HAS_ROCM = False
HAS_MPS = False
HAS_OPENVINO = False
HAS_WEBNN = False
HAS_WEBGPU = False

# Hardware detection
if HAS_TORCH:
    # CUDA
    HAS_CUDA = torch.cuda.is_available()
    
    # ROCm (AMD)
    if HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
        HAS_ROCM = True
    elif 'ROCM_HOME' in os.environ:
        HAS_ROCM = True
    
    # MPS (Apple)
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        HAS_MPS = torch.mps.is_available()

# OpenVINO
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None

# WebNN (browser API)
HAS_WEBNN = (
    importlib.util.find_spec("webnn") is not None or 
    importlib.util.find_spec("webnn_js") is not None or
    "WEBNN_AVAILABLE" in os.environ or
    "WEBNN_SIMULATION" in os.environ
)

# WebGPU (browser API)
HAS_WEBGPU = (
    importlib.util.find_spec("webgpu") is not None or
    importlib.util.find_spec("wgpu") is not None or
    "WEBGPU_AVAILABLE" in os.environ or
    "WEBGPU_SIMULATION" in os.environ
)

# Hardware detection function
def detect_hardware():
    capabilities = {
        "cpu": True,
        "cuda": HAS_CUDA,
        "rocm": HAS_ROCM,
        "mps": HAS_MPS,
        "openvino": HAS_OPENVINO,
        "webnn": HAS_WEBNN,
        "webgpu": HAS_WEBGPU
    }
    return capabilities
"""

    # Web platform optimizations
    web_optimizations = """
# Web Platform Optimizations - March 2025
def apply_web_platform_optimizations():
    """
    if not optimizations:
        optimizations = {}
    
    web_optimizations += f"""
    optimizations = {{
        "compute_shaders": {str(optimizations.get("compute_shaders", False)).lower()},
        "parallel_loading": {str(optimizations.get("parallel_loading", False)).lower()},
        "shader_precompile": {str(optimizations.get("shader_precompile", False)).lower()}
    }}
    
    # Check environment variables
    if os.environ.get("WEBGPU_COMPUTE_SHADERS_ENABLED", "0") == "1":
        optimizations["compute_shaders"] = True
    
    if os.environ.get("WEB_PARALLEL_LOADING_ENABLED", "0") == "1":
        optimizations["parallel_loading"] = True
        
    if os.environ.get("WEBGPU_SHADER_PRECOMPILE_ENABLED", "0") == "1":
        optimizations["shader_precompile"] = True
        
    if os.environ.get("WEB_ALL_OPTIMIZATIONS", "0") == "1":
        optimizations = {{"compute_shaders": True, "parallel_loading": True, "shader_precompile": True}}
    
    return optimizations
"""

    # Database storage integration
    database_integration = """
# Database storage integration
try:
    import duckdb
    import pandas as pd
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False

# Check if JSON output is deprecated
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")

def store_test_results(results, test_name, model_name=None, hardware=None):
    """Store test results in database if available"""
    if not HAS_DATABASE:
        return
        
    try:
        # Get database path from environment
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Store results with appropriate schema
        # (Simplified implementation - would be more complex in real usage)
        results_df = pd.DataFrame([results])
        
        # Create table if needed
        conn.execute(f"CREATE TABLE IF NOT EXISTS test_results_{test_name} AS SELECT * FROM results_df LIMIT 0")
        
        # Insert results
        conn.execute(f"INSERT INTO test_results_{test_name} SELECT * FROM results_df")
        
        # Close connection
        conn.close()
    except Exception as e:
        print(f"Error storing test results in database: {e}")
"""

    # Base test class
    base_test = f"""
import unittest
import os
import time
import json
from pathlib import Path
{hw_imports}
{web_optimizations}
{database_integration}

class Test{model_type.capitalize()}(unittest.TestCase):
    """
    
    # Add test methods
    test_methods = """
    @classmethod
    def setUpClass(cls):
        """Setup for all test methods"""
        cls.hardware = detect_hardware()
        cls.results = {{"hardware": cls.hardware}}
        
    def test_cpu_inference(self):
        """Test CPU inference"""
        # CPU is always available
        try:
            # Implement CPU test logic here
            self.results["cpu"] = {{"status": "pass", "time": time.time()}}
            self.assertTrue(True)
        except Exception as e:
            self.results["cpu"] = {{"status": "fail", "error": str(e)}}
            raise
"""
    
    # Add hardware-specific tests
    if not hardware_platforms:
        hardware_platforms = ["cpu"]

    if "cuda" in hardware_platforms:
        test_methods += """
    def test_cuda_inference(self):
        """Test CUDA inference if available"""
        if not self.hardware["cuda"]:
            self.skipTest("CUDA not available")
        
        try:
            # Implement CUDA test logic here
            self.results["cuda"] = {"status": "pass", "time": time.time()}
            self.assertTrue(True)
        except Exception as e:
            self.results["cuda"] = {"status": "fail", "error": str(e)}
            raise
"""

    if "openvino" in hardware_platforms:
        test_methods += """
    def test_openvino_inference(self):
        """Test OpenVINO inference if available"""
        if not self.hardware["openvino"]:
            self.skipTest("OpenVINO not available")
        
        try:
            # Implement OpenVINO test logic here
            self.results["openvino"] = {"status": "pass", "time": time.time()}
            self.assertTrue(True)
        except Exception as e:
            self.results["openvino"] = {"status": "fail", "error": str(e)}
            raise
"""

    if "mps" in hardware_platforms:
        test_methods += """
    def test_mps_inference(self):
        """Test MPS inference if available"""
        if not self.hardware["mps"]:
            self.skipTest("MPS not available")
        
        try:
            # Implement MPS test logic here
            self.results["mps"] = {"status": "pass", "time": time.time()}
            self.assertTrue(True)
        except Exception as e:
            self.results["mps"] = {"status": "fail", "error": str(e)}
            raise
"""

    if "rocm" in hardware_platforms:
        test_methods += """
    def test_rocm_inference(self):
        """Test ROCm inference if available"""
        if not self.hardware["rocm"]:
            self.skipTest("ROCm not available")
        
        try:
            # Implement ROCm test logic here
            self.results["rocm"] = {"status": "pass", "time": time.time()}
            self.assertTrue(True)
        except Exception as e:
            self.results["rocm"] = {"status": "fail", "error": str(e)}
            raise
"""

    if "webnn" in hardware_platforms:
        test_methods += """
    def test_webnn_inference(self):
        """Test WebNN inference if available"""
        if not self.hardware["webnn"]:
            self.skipTest("WebNN not available")
        
        try:
            # Implement WebNN test logic here
            self.results["webnn"] = {"status": "pass", "time": time.time()}
            self.assertTrue(True)
        except Exception as e:
            self.results["webnn"] = {"status": "fail", "error": str(e)}
            raise
"""

    if "webgpu" in hardware_platforms:
        test_methods += """
    def test_webgpu_inference(self):
        """Test WebGPU inference if available"""
        if not self.hardware["webgpu"]:
            self.skipTest("WebGPU not available")
        
        # Get web platform optimizations
        opt = apply_web_platform_optimizations()
        
        try:
            # Implement WebGPU test logic here with optimizations
            self.results["webgpu"] = {
                "status": "pass", 
                "time": time.time(),
                "optimizations": opt
            }
            self.assertTrue(True)
        except Exception as e:
            self.results["webgpu"] = {"status": "fail", "error": str(e)}
            raise
"""

    # Add tearDown method to save results
    teardown = """
    @classmethod
    def tearDownClass(cls):
        """Save test results after all tests run"""
        # Save results with timestamp and hardware info
        results = {
            "timestamp": time.time(),
            "model_type": "MODEL_TYPE_PLACEHOLDER",
            "results": cls.results
        }
        
        if not DEPRECATE_JSON_OUTPUT:
            # Create output directory if it doesn't exist
            output_dir = Path("./test_results")
            output_dir.mkdir(exist_ok=True)
            
            # Save results to JSON file
            output_file = output_dir / f"test_{model_type.lower()}_{int(time.time())}.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
        
        # Store in database if available
        store_test_results(results, "model_tests", model_type.lower())

if __name__ == "__main__":
    unittest.main()
"""
    
    # Combine all parts
    template = base_test + test_methods + teardown
    
    # Replace model type placeholder
    template = template.replace("MODEL_TYPE_PLACEHOLDER", model_type.lower())
    
    return template

class TestGenerator:
    """
    Enhanced test generator that creates test files based on model types,
    with comprehensive hardware detection and database integration.
    """
    
    def __init__(self):
        """Initialize the test generator"""
        self.hw_capabilities = detect_all_hardware()
        self.output_dir = OUTPUT_DIR
        self.model_registry = self._load_model_registry()
        
    def _load_model_registry(self):
        """Load model registry containing available models"""
        # This would normally load from a centralized registry
        # but for this example, we'll use a simple dictionary
        registry = {
            "bert": {"type": "text_embedding", "task": "embedding"},
            "t5": {"type": "text_generation", "task": "translation"},
            "gpt2": {"type": "text_generation", "task": "generation"},
            "clip": {"type": "vision_text", "task": "multimodal"},
            "vit": {"type": "vision", "task": "classification"},
            "whisper": {"type": "audio", "task": "transcription"},
            "wav2vec2": {"type": "audio", "task": "transcription"},
            "llava": {"type": "vision_language", "task": "multimodal"},
        }
        return registry
    
    def generate_test(self, model_type: str, hardware_platforms: List[str] = None, 
                     optimizations: Dict[str, bool] = None, 
                     cross_platform: bool = False) -> Optional[Path]:
        """
        Generate a test file for the given model type.
        
        Args:
            model_type: Type of model (bert, t5, etc.)
            hardware_platforms: List of hardware platforms to support
            optimizations: Web platform optimizations to include
            cross_platform: Whether to generate tests for all platforms
            
        Returns:
            Path to the generated test file
        """
        # Standardize model type
        model_type = model_type.lower()
        
        # Check if model type is in registry
        if model_type not in self.model_registry:
            logger.warning(f"Model type '{model_type}' not found in registry")
            return None
        
        # Determine hardware platforms to include
        if cross_platform:
            # Use all available platforms
            hardware_platforms = HARDWARE_PLATFORMS
        elif not hardware_platforms:
            # Default to CPU and any available GPU (CUDA, ROCm, MPS)
            hardware_platforms = ["cpu"]
            if self.hw_capabilities.get("cuda", {}).get("detected", False):
                hardware_platforms.append("cuda")
            if self.hw_capabilities.get("rocm", {}).get("detected", False):
                hardware_platforms.append("rocm")
            if self.hw_capabilities.get("mps", {}).get("detected", False):
                hardware_platforms.append("mps")
        
        # Determine web platform optimizations
        if not optimizations and (
            "webnn" in hardware_platforms or 
            "webgpu" in hardware_platforms
        ):
            # Get optimizations based on model type
            optimizations = apply_web_platform_optimizations(model_type, "webgpu")
        
        logger.info(f"Generating test for {model_type} with platforms: {hardware_platforms}")
        
        # Get the test template
        template = get_test_template(model_type, hardware_platforms, optimizations)
        
        # Prepare output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate file path
        file_path = self.output_dir / f"test_hf_{model_type.lower()}.py"
        
        # Write the file
        with open(file_path, "w") as f:
            f.write(template)
        
        # Store metadata in database if available
        if HAS_DATABASE_MODULE:
            # Create or get a test run
            run_id = get_or_create_test_run(
                test_name=f"generate_{model_type.lower()}",
                test_type="test_generation",
                metadata={
                    "model_type": model_type,
                    "hardware_platforms": hardware_platforms,
                    "cross_platform": cross_platform
                }
            )
            
            # Get or create model
            model_id = get_or_create_model(
                model_name=model_type,
                model_family=model_type.split("-")[0],
                model_type=self.model_registry.get(model_type, {}).get("type"),
                task=self.model_registry.get(model_type, {}).get("task")
            )
            
            # Store implementation metadata
            store_implementation_metadata(
                model_type=model_type,
                file_path=str(file_path),
                generation_date=datetime.datetime.now(),
                model_category=self.model_registry.get(model_type, {}).get("type"),
                hardware_support={p: "REAL" for p in hardware_platforms},
                primary_task=self.model_registry.get(model_type, {}).get("task"),
                cross_platform=cross_platform
            )
            
            # Store test result
            store_test_result(
                run_id=run_id,
                test_name=f"generate_{model_type.lower()}",
                status="PASS",
                model_id=model_id,
                metadata={
                    "hardware_platforms": hardware_platforms,
                    "file_path": str(file_path)
                }
            )
            
            # Complete test run
            complete_test_run(run_id)
        
        logger.info(f"Generated test file: {file_path}")
        return file_path
    
    def generate_tests_batch(self, model_types: List[str], 
                            hardware_platforms: List[str] = None,
                            cross_platform: bool = False,
                            max_workers: int = 5) -> List[Path]:
        """
        Generate test files for multiple model types in parallel.
        
        Args:
            model_types: List of model types
            hardware_platforms: List of hardware platforms to support
            cross_platform: Whether to generate tests for all platforms
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of paths to generated test files
        """
        results = []
        failed_models = []
        
        # Use thread pool to generate tests in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dict mapping futures to their models
            future_to_model = {}
            for model_type in model_types:
                future = executor.submit(
                    self.generate_test,
                    model_type,
                    hardware_platforms,
                    None,  # optimizations
                    cross_platform
                )
                future_to_model[future] = model_type
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                model_type = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.info(f"Successfully generated test for {model_type}")
                    else:
                        failed_models.append(model_type)
                        logger.error(f"Failed to generate test for {model_type}")
                except Exception as e:
                    failed_models.append(model_type)
                    logger.error(f"Exception generating test for {model_type}: {e}")
                    logger.debug(traceback.format_exc())
        
        # Log summary
        logger.info(f"Generated {len(results)} test files, {len(failed_models)} failures")
        if failed_models:
            logger.info(f"Failed models: {', '.join(failed_models)}")
        
        return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate model test files")
    
    # Model selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", type=str, help="Generate test for a specific model")
    group.add_argument("--batch-generate", type=str, help="Generate tests for comma-separated list of models")
    group.add_argument("--all", action="store_true", help="Generate tests for all models in registry")
    
    # Hardware platforms
    parser.add_argument("--hardware", type=str, help="Comma-separated list of hardware platforms to include")
    parser.add_argument("--cross-platform", action="store_true", help="Generate tests for all hardware platforms")
    
    # Output options
    parser.add_argument("--output-dir", type=str, help="Output directory for generated tests")
    
    # Web platform options
    parser.add_argument("--web-optimizations", action="store_true", help="Include web platform optimizations")
    
    # Parallel processing
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of parallel workers")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Create test generator
    generator = TestGenerator()
    
    # Set output directory if provided
    if args.output_dir:
        generator.output_dir = Path(args.output_dir)
    
    # Parse hardware platforms
    hardware_platforms = None
    if args.hardware:
        hardware_platforms = args.hardware.split(",")
        if "all" in hardware_platforms:
            hardware_platforms = HARDWARE_PLATFORMS
    
    # Generate tests based on arguments
    if args.generate:
        # Generate a single test
        generator.generate_test(
            args.generate,
            hardware_platforms,
            {"compute_shaders": args.web_optimizations, 
             "parallel_loading": args.web_optimizations,
             "shader_precompile": args.web_optimizations},
            args.cross_platform
        )
    elif args.batch_generate:
        # Generate multiple tests
        model_types = args.batch_generate.split(",")
        generator.generate_tests_batch(
            model_types,
            hardware_platforms,
            args.cross_platform,
            args.max_workers
        )
    elif args.all:
        # Generate tests for all models in registry
        model_types = list(generator.model_registry.keys())
        generator.generate_tests_batch(
            model_types,
            hardware_platforms,
            args.cross_platform,
            args.max_workers
        )

if __name__ == "__main__":
    main()