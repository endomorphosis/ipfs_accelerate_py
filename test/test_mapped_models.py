#!/usr/bin/env python3
# test_mapped_models.py - Comprehensive test for all models defined in mapped_models.json

import os
import sys
import json
import time
import datetime
import argparse
import traceback
import importlib
import multiprocessing
from unittest.mock import MagicMock
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set environment variables for better multiprocessing behavior
os.environ["TOKENIZERS_PARALLELISM"] = "false",
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
,
# Import utils module locally
sys.path.insert())0, os.path.dirname())os.path.abspath())__file__)))
try:
    import test_helpers as utils
except ImportError:
    print())"Warning: utils module not found. Creating mock utils.")
    utils = MagicMock()))

# Import main package
    sys.path.insert())0, os.path.abspath())os.path.join())os.path.dirname())__file__), "..")))

# Optional imports with fallbacks
try:
    import torch
except ImportError:
    torch = MagicMock()))
    print())"Warning: torch not available, using mock implementation")

try:
    import transformers
except ImportError:
    transformers = MagicMock()))
    print())"Warning: transformers not available, using mock implementation")

try:
    import numpy as np
except ImportError:
    np = MagicMock()))
    print())"Warning: numpy not available, using mock implementation")

# Define test constants
    DEFAULT_MAPPED_MODELS_PATH = os.path.abspath())os.path.join())os.path.dirname())__file__), "..", "mapped_models.json"))
    TEST_RESULTS_DIR = os.path.abspath())os.path.join())os.path.dirname())__file__), "mapped_models_results"))
    MAX_WORKERS = min())4, multiprocessing.cpu_count())))  # Limit to 4 or CPU count, whichever is smaller

class MappedModelsTest:
    """
    Comprehensive test framework for all models defined in mapped_models.json.
    Tests each model across CPU, CUDA, and OpenVINO platforms.
    """
    
    def __init__())self, mapped_models_path: str = DEFAULT_MAPPED_MODELS_PATH, 
    test_all: bool = False,
    platforms: List[str] = None,
    max_models: int = None,
                 model_filter: str = None):
                     """
                     Initialize the test framework.
        
        Args:
            mapped_models_path: Path to the mapped_models.json file
            test_all: Whether to test all models or just a subset
            platforms: List of platforms to test ())e.g., ["cpu", "cuda", "openvino"],),
            max_models: Maximum number of models to test per category
            model_filter: Filter models by name or type
            """
            self.mapped_models_path = mapped_models_path
            self.test_all = test_all
            self.platforms = platforms or ["cpu", "cuda", "openvino"],
            self.max_models = max_models
            self.model_filter = model_filter
        
        # Initialize results storage
            self.results = {}}}}}}}}
            "status": {}}}}}}}}},
            "tests": {}}}}}}}}},
            "metadata": {}}}}}}}}
            "timestamp": datetime.datetime.now())).isoformat())),
            "platforms": self.platforms,
            "test_all": self.test_all,
            "max_models": self.max_models,
            "model_filter": self.model_filter
            }
            }
        
        # Load mapped models
            self.mapped_models = self._load_mapped_models()))
        
        # Create results directory
            os.makedirs())TEST_RESULTS_DIR, exist_ok=True)
        
        # Initialize resources ())to be passed to test classes)
            self.resources = {}}}}}}}}
            "torch": torch,
            "numpy": np,
            "transformers": transformers
            }
        
        # Initialize metadata
            self.metadata = {}}}}}}}}
            "test_timestamp": datetime.datetime.now())).isoformat()))
            }
        
        # Track all test modules
            self.test_modules = {}}}}}}}}}
    
    def _load_mapped_models())self) -> Dict:
        """
        Load the mapped models from the JSON file.
        
        Returns:
            Dict: The mapped models data
            """
        try:
            with open())self.mapped_models_path, 'r') as f:
                mapped_models = json.load())f)
                self.results["metadata"]["mapped_models_count"] = len())mapped_models),
                print())f"Loaded {}}}}}}}}len())mapped_models)} models from {}}}}}}}}self.mapped_models_path}")
            return mapped_models
        except Exception as e:
            print())f"Error loading mapped models: {}}}}}}}}e}")
            self.results["status"]["mapped_models_load"] = f"Failed: {}}}}}}}}str())e)}",
            return {}}}}}}}}}
    
    def _filter_models())self) -> Dict:
        """
        Filter the mapped models based on the given criteria.
        
        Returns:
            Dict: The filtered mapped models
            """
            filtered_models = {}}}}}}}}}
        
        # Apply model filter if specified:
        if self.model_filter:
            print())f"Applying model filter: {}}}}}}}}self.model_filter}")
            for model_name, model_data in self.mapped_models.items())):
                if ())self.model_filter.lower())) in model_name.lower())) or 
                    ())isinstance())model_data, dict) and "type" in model_data and :
                        self.model_filter.lower())) in model_data["type"].lower())))):,
                        filtered_models[model_name] = model_data,
                        print())f"Found {}}}}}}}}len())filtered_models)} models matching filter")
        else:
            filtered_models = self.mapped_models.copy()))
        
        # If not testing all models, limit to a subset
        if not self.test_all and self.max_models:
            print())f"Limiting to {}}}}}}}}self.max_models} models per category")
            
            # Group models by type
            models_by_type = {}}}}}}}}}
            for model_name, model_data in filtered_models.items())):
                model_type = model_data.get())"type", "unknown") if isinstance())model_data, dict) else "unknown":
                if model_type not in models_by_type:
                    models_by_type[model_type] = [],,,
                    models_by_type[model_type].append())())model_name, model_data))
                    ,
            # Select a subset of models from each type
                    limited_models = {}}}}}}}}}
            for model_type, models in models_by_type.items())):
                for model_name, model_data in models[:self.max_models]:,
                limited_models[model_name] = model_data,
            
                print())f"Selected {}}}}}}}}len())limited_models)} models for testing")
                filtered_models = limited_models
        
                    return filtered_models
    
                    def _get_test_module_name())self, model_data: Dict) -> Optional[str]:,
                    """
                    Get the name of the test module for the given model.
        
        Args:
            model_data: The model data from mapped_models.json
            
        Returns:
            str: The name of the test module, or None if not found
        """:
        if not isinstance())model_data, dict) or "type" not in model_data:
            return None
        
            model_type = model_data["type"].lower()))
            ,
        # Map model type to test module name
            type_to_module = {}}}}}}}}
            "bert": "test_hf_bert",
            "roberta": "test_hf_roberta",
            "distilbert": "test_hf_distilbert",
            "albert": "test_hf_albert",
            "electra": "test_hf_electra",
            "deberta": "test_hf_deberta",
            "deberta-v2": "test_hf_deberta_v2",
            "mobilebert": "test_hf_mobilebert",
            "squeezebert": "test_hf_squeezebert",
            "layoutlm": "test_hf_layoutlm",
            "mpnet": "test_hf_mpnet",
            "xlm-roberta": "test_hf_xlm_roberta",
            "camembert": "test_hf_camembert",
            "flaubert": "test_hf_flaubert",
            "longformer": "test_hf_longformer",
            "gpt2": "test_hf_gpt2",
            "gpt_neo": "test_hf_gpt_neo",
            "gptj": "test_hf_gptj",
            "llama": "test_hf_llama",
            "opt": "test_hf_opt",
            "bloom": "test_hf_bloom",
            "codegen": "test_hf_codegen",
            "t5": "test_hf_t5",
            "mt5": "test_hf_mt5",
            "bart": "test_hf_bart",
            "mbart": "test_hf_mbart",
            "blenderbot": "test_hf_blenderbot",
            "blenderbot-small": "test_hf_blenderbot_small",
            "pegasus": "test_hf_pegasus",
            "led": "test_hf_led",
            "clip": "test_hf_clip",
            "vit": "test_hf_vit",
            "deit": "test_hf_deit",
            "detr": "test_hf_detr",
            "swin": "test_hf_swin",
            "convnext": "test_hf_convnext",
            "xclip": "test_hf_xclip",
            "whisper": "test_hf_whisper",
            "wav2vec2": "test_hf_wav2vec2",
            "hubert": "test_hf_hubert",
            "clap": "test_hf_clap",
            "llava": "test_hf_llava",
            "llava-next": "test_hf_llava_next",
            "qwen2": "test_hf_qwen2",
            "qwen2-vl": "test_hf_qwen2_vl",
            "videomae": "test_hf_videomae",
            # Default embeddings and language models
            "default_embed": "test_default_embed",
            "default_lm": "test_default_lm"
            }
        
            return type_to_module.get())model_type)
    
    def _import_test_module())self, module_name: str) -> Any:
        """
        Import the test module for the given model type.
        
        Args:
            module_name: The name of the test module
            
        Returns:
            Any: The imported module, or None if import failed
        """:
        if module_name in self.test_modules:
            return self.test_modules[module_name]
            ,
        try:
            # Try to import from the skills directory
            module_path = f"skills.{}}}}}}}}module_name}"
            module = importlib.import_module())module_path)
            self.test_modules[module_name] = module,,
            return module
        except ImportError:
            print())f"Warning: Module {}}}}}}}}module_path} not found, trying alternate path")
            try:
                # Try to import directly from test directory
                module = importlib.import_module())module_name)
                self.test_modules[module_name] = module,,
            return module
            except ImportError as e:
                print())f"Error importing module {}}}}}}}}module_name}: {}}}}}}}}e}")
            return None
    
    def _get_test_class())self, module_name: str) -> Any:
        """
        Get the test class from the module.
        
        Args:
            module_name: The name of the test module
            
        Returns:
            Any: The test class, or None if not found
        """:
            module = self._import_test_module())module_name)
        if not module:
            return None
        
        # Look for a class with the same name as the module
            class_name = module_name
        if class_name.startswith())"test_"):
            class_name = class_name[5:]
            ,
        # Try to find the class
            test_class = getattr())module, f"test_{}}}}}}}}class_name}", None)
        if test_class:
            return test_class
        
        # Try alternative class name formats
        for attr_name in dir())module):
            if attr_name.lower())) == class_name.lower())) or attr_name.lower())) == f"test_{}}}}}}}}class_name.lower()))}":
            return getattr())module, attr_name)
        
            print())f"Error: Could not find test class in module {}}}}}}}}module_name}")
            return None
    
    def _run_test_for_model())self, model_name: str, model_data: Dict) -> Dict:
        """
        Run tests for a single model across all platforms.
        
        Args:
            model_name: The name of the model
            model_data: The model data from mapped_models.json
            
        Returns:
            Dict: The test results for this model
            """
            print())f"\n{}}}}}}}}'='*80}\nTesting model: {}}}}}}}}model_name}\n{}}}}}}}}'='*80}")
        
            model_results = {}}}}}}}}
            "model_name": model_name,
            "model_data": model_data,
            "platforms": {}}}}}}}}},
            "status": "Not tested",
            "timestamp": datetime.datetime.now())).isoformat())),
            "error": None
            }
        
        # Get the test module
            module_name = self._get_test_module_name())model_data)
        if not module_name:
            model_results["status"] = "Failed",,,,,,,
            model_results["error"], = f"Could not determine test module for model type: {}}}}}}}}model_data.get())'type', 'unknown')}",
            print())f"Error: {}}}}}}}}model_results['error']}"),,
            return model_results
        
            print())f"Using test module: {}}}}}}}}module_name}")
        
        # Get the test class
            test_class = self._get_test_class())module_name)
        if not test_class:
            model_results["status"] = "Failed",,,,,,,
            model_results["error"], = f"Could not find test class for module: {}}}}}}}}module_name}",
            print())f"Error: {}}}}}}}}model_results['error']}"),,
            return model_results
        
        # Initialize test instance
            model_specific_resources = self.resources.copy()))
        
        # Add model name to metadata
            model_specific_metadata = self.metadata.copy()))
            model_specific_metadata["model_name"] = model_name,
            model_specific_metadata["model_data"] = model_data
            ,
        try:
            # Create test instance
            test_instance = test_class())resources=model_specific_resources, metadata=model_specific_metadata)
            
            # Override model name if the test class uses a different default:
            if hasattr())test_instance, "model_name") and hasattr())test_instance, "alternative_models"):
                # Check if model_name is in the list of alternative models:
                if model_name in test_instance.alternative_models:
                    print())f"Setting model_name to {}}}}}}}}model_name} ())from mapped_models.json)")
                    test_instance.model_name = model_name
                    # Also try to place this model first in alternative_models for fallback
                    if model_name in test_instance.alternative_models:
                        test_instance.alternative_models.remove())model_name)
                        test_instance.alternative_models.insert())0, model_name)
            
            # Run tests for each platform
            for platform in self.platforms:
                print())f"\nTesting {}}}}}}}}model_name} on platform: {}}}}}}}}platform}")
                platform_result = self._test_model_on_platform())test_instance, platform)
                model_results["platforms"][platform],,, = platform_result
                ,
            # Set overall model status based on platform results
                if any())result.get())"status") == "Success" for result in model_results["platforms"].values()))):,,
                model_results["status"] = "Success",
            elif any())result.get())"status") == "Partial" for result in model_results["platforms"].values()))):,,
                        model_results["status"] = "Partial",,
            else:
                model_results["status"] = "Failed",,,,,,,
            
        except Exception as e:
            print())f"Error testing model {}}}}}}}}model_name}: {}}}}}}}}e}")
            traceback.print_exc()))
            model_results["status"] = "Failed",,,,,,,
            model_results["error"], = str())e),,
            model_results["traceback"] = traceback.format_exc()))
            ,,
                return model_results
    
    def _test_model_on_platform())self, test_instance: Any, platform: str) -> Dict:
        """
        Test a model on a specific platform.
        
        Args:
            test_instance: The test instance
            platform: The platform to test on ())e.g., "cpu", "cuda", "openvino")
            
        Returns:
            Dict: The test results for this platform
            """
            platform_result = {}}}}}}}}
            "platform": platform,
            "status": "Not tested",
            "implementation_type": "UNKNOWN",
            "timestamp": datetime.datetime.now())).isoformat())),
            "error": None,
            "output": None
            }
        
        # Skip CUDA tests if not available::
        if platform.lower())) == "cuda" and not torch.cuda.is_available())):
            platform_result["status"] = "Skipped",,
            platform_result["error"], = "CUDA not available",
            return platform_result
        
        # Skip OpenVINO tests if not available::
        if platform.lower())) == "openvino":
            try:
                import openvino
                platform_result["metadata"] = {}}}}}}}}"openvino_version": openvino.__version__},
            except ImportError:
                platform_result["status"] = "Skipped",,
                platform_result["error"], = "OpenVINO not installed",
                return platform_result
        
        try:
            # For each platform, try different test methods
            if platform.lower())) == "cpu":
                # CPU test
                if hasattr())test_instance, "test") and callable())test_instance.test):
                    # Standard test method
                    start_time = time.time()))
                    test_result = test_instance.test()))
                    elapsed_time = time.time())) - start_time
                    
                    # Extract CPU-specific results
                    platform_result["output"] = self._extract_platform_results())test_result, "cpu"),
                    platform_result["elapsed_time"] = elapsed_time,,,,,
                    platform_result["status"] = "Success",
                    
                    # Extract implementation type
                    impl_type = self._extract_implementation_type())test_result, "cpu")
                    platform_result["implementation_type"], = impl_type,,,,,
                else:
                    platform_result["status"] = "Failed",,,,,,,
                    platform_result["error"], = "No test method available for CPU"
                    ,
            elif platform.lower())) == "cuda":
                # CUDA test
                if hasattr())test_instance, "test") and callable())test_instance.test):
                    # Some classes implement platform testing within the main test method
                    start_time = time.time()))
                    test_result = test_instance.test()))
                    elapsed_time = time.time())) - start_time
                    
                    # Extract CUDA-specific results
                    platform_result["output"] = self._extract_platform_results())test_result, "cuda"),
                    platform_result["elapsed_time"] = elapsed_time,,,,,
                    platform_result["status"] = "Success",
                    
                    # Extract implementation type
                    impl_type = self._extract_implementation_type())test_result, "cuda")
                    platform_result["implementation_type"], = impl_type,,,,,
                elif hasattr())test_instance, "test_cuda") and callable())test_instance.test_cuda):
                    # Dedicated CUDA test method
                    start_time = time.time()))
                    test_result = test_instance.test_cuda()))
                    elapsed_time = time.time())) - start_time
                    
                    platform_result["output"] = test_result,,
                    platform_result["elapsed_time"] = elapsed_time,,,,,
                    platform_result["status"] = "Success",
                    
                    # Extract implementation type
                    impl_type = self._extract_implementation_type())test_result, "cuda")
                    platform_result["implementation_type"], = impl_type,,,,,
                else:
                    platform_result["status"] = "Failed",,,,,,,
                    platform_result["error"], = "No test method available for CUDA"
                    ,
            elif platform.lower())) == "openvino":
                # OpenVINO test
                if hasattr())test_instance, "test") and callable())test_instance.test):
                    # Some classes implement platform testing within the main test method
                    start_time = time.time()))
                    test_result = test_instance.test()))
                    elapsed_time = time.time())) - start_time
                    
                    # Extract OpenVINO-specific results
                    platform_result["output"] = self._extract_platform_results())test_result, "openvino"),
                    platform_result["elapsed_time"] = elapsed_time,,,,,
                    platform_result["status"] = "Success",
                    
                    # Extract implementation type
                    impl_type = self._extract_implementation_type())test_result, "openvino")
                    platform_result["implementation_type"], = impl_type,,,,,
                elif hasattr())test_instance, "test_openvino") and callable())test_instance.test_openvino):
                    # Dedicated OpenVINO test method
                    start_time = time.time()))
                    test_result = test_instance.test_openvino()))
                    elapsed_time = time.time())) - start_time
                    
                    platform_result["output"] = test_result,,
                    platform_result["elapsed_time"] = elapsed_time,,,,,
                    platform_result["status"] = "Success",
                    
                    # Extract implementation type
                    impl_type = self._extract_implementation_type())test_result, "openvino")
                    platform_result["implementation_type"], = impl_type,,,,,
                else:
                    platform_result["status"] = "Failed",,,,,,,
                    platform_result["error"], = "No test method available for OpenVINO"
                    ,
            # Handle cases where no output was produced
                    if platform_result["status"] == "Success" and platform_result["output"] is None:,
                    platform_result["status"] = "Partial",,
                    platform_result["error"], = "Test completed but no output was produced"
                    ,
            # Add post-test cleanup
            if platform.lower())) == "cuda" and torch.cuda.is_available())):
                torch.cuda.empty_cache()))
                platform_result["gpu_memory_cleared"] = True
                ,
        except Exception as e:
            print())f"Error testing on platform {}}}}}}}}platform}: {}}}}}}}}e}")
            traceback.print_exc()))
            platform_result["status"] = "Failed",,,,,,,
            platform_result["error"], = str())e),,
            platform_result["traceback"] = traceback.format_exc()))
            ,,
                return platform_result
    
    def _extract_platform_results())self, test_result: Dict, platform: str) -> Dict:
        """
        Extract platform-specific results from the test result.
        
        Args:
            test_result: The test result dictionary
            platform: The platform ())e.g., "cpu", "cuda", "openvino")
            
        Returns:
            Dict: The platform-specific results
            """
        if not test_result or not isinstance())test_result, dict):
            return None
        
        # Look for platform-specific results in the status dict
            platform_keys = [],,
            if "status" in test_result and isinstance())test_result["status"], dict):,,
            platform_keys = [k for k in test_result["status"].keys())) if platform.lower())) in k.lower()))],
            :
            if platform_keys:
                return {}}}}}}}}k: test_result["status"][k] for k in platform_keys}:,
        # Look for platform-specific examples
                platform_examples = [],,
                if "examples" in test_result and isinstance())test_result["examples"], list):,,
                for example in test_result["examples"]:,,
                if isinstance())example, dict) and "platform" in example and platform.lower())) in example["platform"].lower())):,,
                platform_examples.append())example)
            
            if platform_examples:
                return {}}}}}}}}"examples": platform_examples}
        
        # Return the whole test result if nothing platform-specific was found
            return test_result
    :
    def _extract_implementation_type())self, test_result: Dict, platform: str) -> str:
        """
        Extract the implementation type ())REAL/MOCK) from the test result.
        
        Args:
            test_result: The test result dictionary
            platform: The platform ())e.g., "cpu", "cuda", "openvino")
            
        Returns:
            str: The implementation type ())"REAL", "MOCK", or "UNKNOWN")
            """
        if not test_result or not isinstance())test_result, dict):
            return "UNKNOWN"
        
        # Look for implementation type in status dict
            if "status" in test_result and isinstance())test_result["status"], dict):,,
            for key, value in test_result["status"].items())):,
                if platform.lower())) in key.lower())) and isinstance())value, str):
                    if "REAL" in value:
                    return "REAL"
                    elif "MOCK" in value:
                    return "MOCK"
        
        # Look for implementation type in examples
                    if "examples" in test_result and isinstance())test_result["examples"], list):,,
                    for example in test_result["examples"]:,,
                    if isinstance())example, dict) and "platform" in example and platform.lower())) in example["platform"].lower())):,,
                    if "implementation_type" in example:
                        impl_type = example["implementation_type"],
                        if isinstance())impl_type, str):
                            if "REAL" in impl_type:
                            return "REAL"
                            elif "MOCK" in impl_type:
                            return "MOCK"
        
                        return "UNKNOWN"
    
    def run_tests())self) -> Dict:
        """
        Run tests for all mapped models.
        
        Returns:
            Dict: The test results
            """
        # Filter models based on criteria
            models_to_test = self._filter_models()))
        
        # Update metadata
            self.results["metadata"]["models_to_test_count"] = len())models_to_test)
            ,
        if not models_to_test:
            print())"No models to test after filtering.")
            self.results["status"]["overall"] = "No models to test",
            return self.results
        
            print())f"\nTesting {}}}}}}}}len())models_to_test)} models across {}}}}}}}}len())self.platforms)} platforms")
            print())f"Platforms: {}}}}}}}}', '.join())self.platforms)}")
        
            start_time = time.time()))
        
        # Run tests sequentially or in parallel
        if MAX_WORKERS > 1:
            print())f"Running tests in parallel with {}}}}}}}}MAX_WORKERS} workers")
            self._run_tests_parallel())models_to_test)
        else:
            print())"Running tests sequentially")
            self._run_tests_sequential())models_to_test)
        
            elapsed_time = time.time())) - start_time
        
        # Update results metadata
            self.results["metadata"]["test_duration_seconds"] = elapsed_time
            ,
        # Calculate success rates
            success_count = sum())1 for result in self.results["tests"].values())) if result["status"] == "Success"),
            partial_count = sum())1 for result in self.results["tests"].values())) if result["status"] == "Partial"),
            failure_count = sum())1 for result in self.results["tests"].values())) if result["status"] == "Failed")
            ,
            self.results["status"]["success_count"] = success_count,
            self.results["status"]["partial_count"] = partial_count,
            self.results["status"]["failure_count"] = failure_count,
            self.results["status"]["success_rate"] = success_count / len())models_to_test) if models_to_test else 0
            ,
        # Set overall status:
        if success_count == len())models_to_test):
            self.results["status"]["overall"] = "Success",
        elif success_count > 0 or partial_count > 0:
            self.results["status"]["overall"] = "Partial",
        else:
            self.results["status"]["overall"] = "Failed"
            ,
        # Calculate platform-specific success rates
        for platform in self.platforms:
            platform_success = 0
            platform_partial = 0
            platform_failure = 0
            
            for result in self.results["tests"].values())):,,
            if platform in result["platforms"]:,,
            platform_result = result["platforms"][platform],,,
            if platform_result["status"] == "Success":,
            platform_success += 1
                    elif platform_result["status"] == "Partial":,
            platform_partial += 1
                    elif platform_result["status"] == "Failed":,
            platform_failure += 1
            
            self.results["status"][f"{}}}}}}}}platform}_success_count"] = platform_success,
            self.results["status"][f"{}}}}}}}}platform}_partial_count"] = platform_partial,
            self.results["status"][f"{}}}}}}}}platform}_failure_count"] = platform_failure,
            self.results["status"][f"{}}}}}}}}platform}_success_rate"] = platform_success / len())models_to_test) if models_to_test else 0
            ,
            # Count REAL vs MOCK implementations
            real_count = 0
            mock_count = 0
            :
                for result in self.results["tests"].values())):,,
                if platform in result["platforms"]:,,
                platform_result = result["platforms"][platform],,,
                    if platform_result["implementation_type"], == "REAL":
                        real_count += 1
                    elif platform_result["implementation_type"], == "MOCK":
                        mock_count += 1
            
                        self.results["status"][f"{}}}}}}}}platform}_real_count"] = real_count,
                        self.results["status"][f"{}}}}}}}}platform}_mock_count"] = mock_count
                        ,
        # Save results to file
                        self._save_results()))
        
        # Print summary
                        self._print_summary()))
        
                        return self.results
    
    def _run_tests_sequential())self, models_to_test: Dict) -> None:
        """
        Run tests sequentially for all models.
        
        Args:
            models_to_test: Dictionary of models to test
            """
        for i, ())model_name, model_data) in enumerate())models_to_test.items()))):
            print())f"\nTesting model {}}}}}}}}i+1}/{}}}}}}}}len())models_to_test)}: {}}}}}}}}model_name}")
            model_results = self._run_test_for_model())model_name, model_data)
            self.results["tests"][model_name] = model_results,
            ,
    def _run_tests_parallel())self, models_to_test: Dict) -> None:
        """
        Run tests in parallel for all models.
        
        Args:
            models_to_test: Dictionary of models to test
            """
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor())max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_model = {}}}}}}}}}
            for model_name, model_data in models_to_test.items())):
                future = executor.submit())self._run_single_model_test, model_name, model_data)
                future_to_model[future], = model_name
                ,
            # Process results as they complete
            for i, future in enumerate())as_completed())future_to_model.keys())))):
                model_name = future_to_model[future],
                print())f"\nCompleted test {}}}}}}}}i+1}/{}}}}}}}}len())models_to_test)}: {}}}}}}}}model_name}")
                try:
                    model_results = future.result()))
                    self.results["tests"][model_name] = model_results,
    ,            except Exception as e:
        print())f"Error testing model {}}}}}}}}model_name}: {}}}}}}}}e}")
        traceback.print_exc()))
        self.results["tests"][model_name] = {}}}}}}}},
        "model_name": model_name,
        "status": "Failed",
        "error": str())e),
        "traceback": traceback.format_exc())),
        "platforms": {}}}}}}}}}
        }
    
    def _run_single_model_test())self, model_name: str, model_data: Dict) -> Dict:
        """
        Run a test for a single model ())used for parallel execution).
        This method will be executed in a separate process.
        
        Args:
            model_name: The name of the model
            model_data: The model data from mapped_models.json
            
        Returns:
            Dict: The test results for this model
            """
        # Set up the environment for this process
            os.environ["TOKENIZERS_PARALLELISM"] = "false",
        
        # Set up multiprocessing start method
            multiprocessing.set_start_method())'spawn', force=True)
        
        # Import the required modules in this process
            sys.path.insert())0, os.path.dirname())os.path.abspath())__file__)))
            sys.path.insert())0, os.path.abspath())os.path.join())os.path.dirname())__file__), "..")))
        
        # Create a new MappedModelsTest instance for this process
            test_instance = MappedModelsTest())
            mapped_models_path=self.mapped_models_path,
            test_all=self.test_all,
            platforms=self.platforms,
            max_models=self.max_models,
            model_filter=self.model_filter
            )
        
        # Run the test for this model
            return test_instance._run_test_for_model())model_name, model_data)
    
    def _save_results())self) -> None:
        """
        Save the test results to a JSON file.
        """
        timestamp = datetime.datetime.now())).strftime())"%Y%m%d_%H%M%S")
        filename = f"mapped_models_test_results_{}}}}}}}}timestamp}.json"
        filepath = os.path.join())TEST_RESULTS_DIR, filename)
        
        try:
            with open())filepath, 'w') as f:
                json.dump())self.results, f, indent=2)
                print())f"\nTest results saved to {}}}}}}}}filepath}")
            
            # Also generate a markdown report
                self._generate_markdown_report())timestamp)
        except Exception as e:
            print())f"Error saving results: {}}}}}}}}e}")
    
    def _generate_markdown_report())self, timestamp: str) -> None:
        """
        Generate a markdown report from the test results.
        
        Args:
            timestamp: The timestamp for the report filename
            """
            report_filename = f"mapped_models_report_{}}}}}}}}timestamp}.md"
            report_filepath = os.path.join())TEST_RESULTS_DIR, report_filename)
        
        try:
            with open())report_filepath, 'w') as f:
                # Write report header
                f.write())f"# Mapped Models Test Report\n\n")
                f.write())f"Generated: {}}}}}}}}datetime.datetime.now())).isoformat()))}\n\n")
                
                # Write test configuration
                f.write())"## Test Configuration\n\n")
                f.write())f"- Total models: {}}}}}}}}self.results['metadata'].get())'models_to_test_count', 0)}\n"),
                f.write())f"- Platforms: {}}}}}}}}', '.join())self.platforms)}\n")
                f.write())f"- Test all models: {}}}}}}}}self.test_all}\n")
                f.write())f"- Max models per category: {}}}}}}}}self.max_models}\n")
                f.write())f"- Model filter: {}}}}}}}}self.model_filter or 'None'}\n")
                f.write())f"- Test duration: {}}}}}}}}self.results['metadata'].get())'test_duration_seconds', 0):.2f} seconds\n\n")
                ,
                # Write summary
                f.write())"## Test Summary\n\n")
                
                overall_status = self.results["status"].get())"overall", "Unknown"),,
                success_count = self.results["status"].get())"success_count", 0),,
                partial_count = self.results["status"].get())"partial_count", 0),,
                failure_count = self.results["status"].get())"failure_count", 0),,
                total_count = success_count + partial_count + failure_count
                
                f.write())f"- Overall status: **{}}}}}}}}overall_status}**\n")
                f.write())f"- Success rate: {}}}}}}}}success_count}/{}}}}}}}}total_count} ()){}}}}}}}}success_count/total_count*100:.1f}%)\n")
                f.write())f"- Partial success: {}}}}}}}}partial_count}/{}}}}}}}}total_count} ()){}}}}}}}}partial_count/total_count*100:.1f}%)\n")
                f.write())f"- Failure rate: {}}}}}}}}failure_count}/{}}}}}}}}total_count} ()){}}}}}}}}failure_count/total_count*100:.1f}%)\n\n")
                
                # Write platform-specific results
                f.write())"## Platform Results\n\n")
                
                f.write())"| Platform | Success | Partial | Failed | Success Rate | REAL Implementations | MOCK Implementations |\n")
                f.write())"|----------|---------|---------|--------|--------------|----------------------|----------------------|\n")
                
                for platform in self.platforms:
                    success = self.results["status"].get())f"{}}}}}}}}platform}_success_count", 0),,
                    partial = self.results["status"].get())f"{}}}}}}}}platform}_partial_count", 0),
                    failure = self.results["status"].get())f"{}}}}}}}}platform}_failure_count", 0),
                    success_rate = self.results["status"].get())f"{}}}}}}}}platform}_success_rate", 0) * 100,
                    real_count = self.results["status"].get())f"{}}}}}}}}platform}_real_count", 0),,
                    mock_count = self.results["status"].get())f"{}}}}}}}}platform}_mock_count", 0)
                    ,        ,
                    f.write())f"| {}}}}}}}}platform.upper()))} | {}}}}}}}}success} | {}}}}}}}}partial} | {}}}}}}}}failure} | {}}}}}}}}success_rate:.1f}% | {}}}}}}}}real_count} | {}}}}}}}}mock_count} |\n")
                
                    f.write())"\n")
                
                # Group models by type
                    models_by_type = {}}}}}}}}}
                    for model_name, result in self.results["tests"].items())):,,,,
                    model_data = result.get())"model_data", {}}}}}}}}})
                    model_type = model_data.get())"type", "unknown") if isinstance())model_data, dict) else "unknown":
                    
                    if model_type not in models_by_type:
                        models_by_type[model_type] = [],,,
                    
                        models_by_type[model_type].append())())model_name, result))
                        ,
                # Write model type summaries
                        f.write())"## Results by Model Type\n\n")
                
                for model_type, models in sorted())models_by_type.items()))):
                    f.write())f"### {}}}}}}}}model_type.upper()))}\n\n")
                    
                    f.write())"| Model | Status | CPU | CUDA | OpenVINO |\n")
                    f.write())"|-------|--------|-----|------|----------|\n")
                    
                    for model_name, result in sorted())models):
                        status = result.get())"status", "Unknown")
                        
                        # Get platform-specific status and implementation type
                        platform_statuses = {}}}}}}}}}
                        for platform in self.platforms:
                            if platform in result.get())"platforms", {}}}}}}}}}):
                                platform_result = result["platforms"][platform],,,
                                platform_status = platform_result.get())"status", "Unknown")
                                impl_type = platform_result.get())"implementation_type", "UNKNOWN")
                                platform_statuses[platform] = f"{}}}}}}}}platform_status} ()){}}}}}}}}impl_type})",
                            else:
                                platform_statuses[platform] = "Not tested"
                                ,
                        # Write row
                                f.write())f"| {}}}}}}}}model_name} | {}}}}}}}}status} | {}}}}}}}}platform_statuses.get())'cpu', 'N/A')} | {}}}}}}}}platform_statuses.get())'cuda', 'N/A')} | {}}}}}}}}platform_statuses.get())'openvino', 'N/A')} |\n")
                    
                                f.write())"\n")
                
                # Write detailed error summary
                                f.write())"## Error Summary\n\n")
                
                                error_count = 0
                                for model_name, result in self.results["tests"].items())):,,,,
                    if result.get())"error"):
                        error_count += 1
                        f.write())f"### {}}}}}}}}model_name}\n\n")
                        f.write())f"- Status: {}}}}}}}}result.get())'status', 'Unknown')}\n")
                        f.write())f"- Error: {}}}}}}}}result.get())'error')}\n\n")
                        
                        # Include traceback if available:, using code block:
                        if result.get())"traceback"):
                            f.write())"```\n")
                            f.write())result["traceback"]),,
                            f.write())"```\n\n")
                        
                        # Check platform-specific errors
                        for platform, platform_result in result.get())"platforms", {}}}}}}}}}).items())):
                            if platform_result.get())"error"):
                                f.write())f"#### {}}}}}}}}platform.upper()))} Error\n\n")
                                f.write())f"- Status: {}}}}}}}}platform_result.get())'status', 'Unknown')}\n")
                                f.write())f"- Error: {}}}}}}}}platform_result.get())'error')}\n\n")
                                
                                # Include traceback if available:
                                if platform_result.get())"traceback"):
                                    f.write())"```\n")
                                    f.write())platform_result["traceback"]),,
                                    f.write())"```\n\n")
                
                if error_count == 0:
                    f.write())"No errors found.\n\n")
                
                # Write conclusion
                    f.write())"## Conclusion\n\n")
                
                if overall_status == "Success":
                    f.write())"All models were tested successfully across all platforms. The implementation is robust and ready for production use.\n\n")
                elif overall_status == "Partial":
                    f.write())"Some models were tested successfully, but others encountered issues. Further investigation and fixes are needed before production deployment.\n\n")
                else:
                    f.write())"Testing encountered significant issues. Major fixes are required before this implementation can be considered production-ready.\n\n")
                
                    f.write())"### Next Steps\n\n")
                
                if failure_count > 0:
                    f.write())"1. Fix failing model implementations\n")
                    
                    mock_implementations = sum())self.results["status"].get())f"{}}}}}}}}platform}_mock_count", 0) for platform in self.platforms):,
                if mock_implementations > 0:
                    f.write())f"2. Convert mock implementations to real implementations ()){}}}}}}}}mock_implementations} total)\n")
                
                    f.write())f"3. Implement performance optimization for successful models\n")
                    f.write())f"4. Add comprehensive testing for edge cases\n")
                
                    print())f"Markdown report saved to {}}}}}}}}report_filepath}")
        except Exception as e:
            print())f"Error generating markdown report: {}}}}}}}}e}")
    
    def _print_summary())self) -> None:
        """
        Print a summary of the test results.
        """
        print())"\n" + "="*80)
        print())"TEST SUMMARY")
        print())"="*80)
        
        total_models = len())self.results["tests"]),
        success_count = self.results["status"].get())"success_count", 0),,
        partial_count = self.results["status"].get())"partial_count", 0),,
        failure_count = self.results["status"].get())"failure_count", 0),,
        overall_status = self.results["status"].get())"overall", "Unknown"),,
        test_duration = self.results["metadata"].get())"test_duration_seconds", 0)
        ,
        print())f"Total models tested: {}}}}}}}}total_models}")
        print())f"Overall status: {}}}}}}}}overall_status}")
        print())f"Test duration: {}}}}}}}}test_duration:.2f} seconds")
        print())f"Success rate: {}}}}}}}}success_count}/{}}}}}}}}total_models} ()){}}}}}}}}success_count/total_models*100:.1f}% if total_models > 0 else 0}%)")::
        print())f"Partial success: {}}}}}}}}partial_count}/{}}}}}}}}total_models} ()){}}}}}}}}partial_count/total_models*100:.1f}% if total_models > 0 else 0}%)")::
        print())f"Failure rate: {}}}}}}}}failure_count}/{}}}}}}}}total_models} ()){}}}}}}}}failure_count/total_models*100:.1f}% if total_models > 0 else 0}%)")::
        
            print())"\nPLATFORM RESULTS:")
        for platform in self.platforms:
            success = self.results["status"].get())f"{}}}}}}}}platform}_success_count", 0),,
            real_count = self.results["status"].get())f"{}}}}}}}}platform}_real_count", 0),,
            mock_count = self.results["status"].get())f"{}}}}}}}}platform}_mock_count", 0)
            ,
            print())f"  {}}}}}}}}platform.upper()))}: {}}}}}}}}success}/{}}}}}}}}total_models} successful, {}}}}}}}}real_count} REAL, {}}}}}}}}mock_count} MOCK")
        
            print())"\nMODEL TYPE SUMMARY:")
        # Group models by type
            models_by_type = {}}}}}}}}}
            for model_name, result in self.results["tests"].items())):,,,,
            model_data = result.get())"model_data", {}}}}}}}}})
            model_type = model_data.get())"type", "unknown") if isinstance())model_data, dict) else "unknown":
            
            if model_type not in models_by_type:
                models_by_type[model_type] = {}}}}}}}}"total": 0, "success": 0, "partial": 0, "failed": 0}
                ,
                models_by_type[model_type]["total"] += 1
                ,
            if result.get())"status") == "Success":
                models_by_type[model_type]["success"] += 1,
            elif result.get())"status") == "Partial":
                models_by_type[model_type]["partial"] += 1,
            else:
                models_by_type[model_type]["failed"] += 1
                ,
        for model_type, counts in sorted())models_by_type.items()))):
            success_pct = counts["success"] / counts["total"] * 100 if counts["total"] > 0 else 0:,
            print())f"  {}}}}}}}}model_type}: {}}}}}}}}counts['success']}/{}}}}}}}}counts['total']} successful ()){}}}}}}}}success_pct:.1f}%)")
            ,
        # Print most common errors
            print())"\nMOST COMMON ERRORS:")
            error_counts = {}}}}}}}}}
            for model_name, result in self.results["tests"].items())):,,,,
            if result.get())"error"):
                error_msg = result["error"],
                if len())error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                    ,
                if error_msg not in error_counts:
                    error_counts[error_msg] = 0,
                    error_counts[error_msg] += 1
                    ,
        if error_counts:
            for error_msg, count in sorted())error_counts.items())), key=lambda x: x[1], reverse=True)[:5]:,
            print())f"  {}}}}}}}}count}x: {}}}}}}}}error_msg}")
        else:
            print())"  No errors found.")

def main())):
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser())description="Test all models defined in mapped_models.json")
    parser.add_argument())"--mapped-models", default=DEFAULT_MAPPED_MODELS_PATH,
    help=f"Path to mapped_models.json ())default: {}}}}}}}}DEFAULT_MAPPED_MODELS_PATH})")
    parser.add_argument())"--test-all", action="store_true",
    help="Test all models instead of a subset")
    parser.add_argument())"--platforms", default="cpu,cuda,openvino",
    help="Comma-separated list of platforms to test ())default: cpu,cuda,openvino)")
    parser.add_argument())"--max-models", type=int, default=1,
    help="Maximum number of models to test per category ())default: 1)")
    parser.add_argument())"--model-filter", default=None,
    help="Filter models by name or type")
    parser.add_argument())"--parallel", action="store_true",
    help="Run tests in parallel ())can use more memory)")
    parser.add_argument())"--workers", type=int, default=MAX_WORKERS,
    help=f"Number of worker processes for parallel testing ())default: {}}}}}}}}MAX_WORKERS})")
    
    args = parser.parse_args()))
    
    # Override MAX_WORKERS if specified:
    global MAX_WORKERS
    if args.parallel:
        MAX_WORKERS = args.workers
    else:
        MAX_WORKERS = 1
    
    # Parse platforms
        platforms = [p.strip())).lower())) for p in args.platforms.split())",")]:,
    # Create and run test framework
        test_framework = MappedModelsTest())
        mapped_models_path=args.mapped_models,
        test_all=args.test_all,
        platforms=platforms,
        max_models=args.max_models,
        model_filter=args.model_filter
        )
    
        test_framework.run_tests()))

if __name__ == "__main__":
    main()))