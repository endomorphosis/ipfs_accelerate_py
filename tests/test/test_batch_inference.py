#!/usr/bin/env python3
# test_batch_inference.py - Test batch inference capabilities across model types

import os
import sys
import json
import time
import datetime
import argparse
import traceback
import importlib
from unittest.mock import MagicMock
from typing import Dict, List, Tuple, Any, Optional

# Set environment variables for better multiprocessing behavior
os.environ["TOKENIZERS_PARALLELISM"] = "false",
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
,
# Import utils module locally
sys.path.insert()))))0, os.path.dirname()))))os.path.abspath()))))__file__)))
try:
    import utils
except ImportError:
    print()))))"Warning: utils module not found. Creating mock utils.")
    utils = MagicMock())))))

# Import main package
    sys.path.insert()))))0, os.path.abspath()))))os.path.join()))))os.path.dirname()))))__file__), "..")))

# Optional imports with fallbacks
try:
    import torch
except ImportError:
    torch = MagicMock())))))
    print()))))"Warning: torch not available, using mock implementation")

try:
    import transformers
except ImportError:
    transformers = MagicMock())))))
    print()))))"Warning: transformers not available, using mock implementation")

try:
    import numpy as np
except ImportError:
    np = MagicMock())))))
    print()))))"Warning: numpy not available, using mock implementation")

# Define test constants
    TEST_RESULTS_DIR = os.path.abspath()))))os.path.join()))))os.path.dirname()))))__file__), "batch_inference_results"))
    DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16]
    ,
class BatchInferenceTest:
    """
    Tests batch inference capabilities for different model types.
    Measures throughput, memory usage, and latency across different batch sizes.
    """
    
    def __init__()))))self, model_types: List[str] = None, 
    batch_sizes: List[int] = None,
    specific_models: Dict[str, str] = None,
    platforms: List[str] = None,
                 use_fp16: bool = False):
                     """
                     Initialize the batch inference test framework.
        
        Args:
            model_types: List of model types to test ()))))e.g., ["bert", "t5", "clip"]),
            batch_sizes: List of batch sizes to test
            specific_models: Dict mapping model types to specific model names
            platforms: List of platforms to test ()))))e.g., ["cpu", "cuda", "openvino"]),
            use_fp16: Whether to use FP16 precision for CUDA tests
            """
            self.model_types = model_types or ["bert", "t5", "clip", "llama", "whisper", "wav2vec2"],
            self.batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
            self.specific_models = specific_models or {}}}}}}}}}}}}}}}}}}}}}}
            self.platforms = platforms or ["cpu", "cuda"],
            self.use_fp16 = use_fp16
        
        # Initialize resources to be passed to test classes
            self.resources = {}}}}}}}}}}}}}}}}}}}}}
            "torch": torch,
            "numpy": np,
            "transformers": transformers
            }
        
        # Initialize metadata
            self.metadata = {}}}}}}}}}}}}}}}}}}}}}
            "test_timestamp": datetime.datetime.now()))))).isoformat()))))),
            "batch_sizes": self.batch_sizes,
            "platforms": self.platforms,
            "use_fp16": self.use_fp16
            }
        
        # Initialize results
            self.results = {}}}}}}}}}}}}}}}}}}}}}
            "metadata": self.metadata,
            "model_results": {}}}}}}}}}}}}}}}}}}}}}},
            "summary": {}}}}}}}}}}}}}}}}}}}}}}
            }
        
        # Create results directory
            os.makedirs()))))TEST_RESULTS_DIR, exist_ok=True)
        
        # Map model types to test modules and test data generators
            self.model_type_mapping = {}}}}}}}}}}}}}}}}}}}}}
            "bert": {}}}}}}}}}}}}}}}}}}}}}
            "module": "test_hf_bert",
            "data_generator": self._generate_text_batch,
            "category": "embeddings"
            },
            "t5": {}}}}}}}}}}}}}}}}}}}}}
            "module": "test_hf_t5",
            "data_generator": self._generate_text_batch,
            "category": "text-generation"
            },
            "llama": {}}}}}}}}}}}}}}}}}}}}}
            "module": "test_hf_llama",
            "data_generator": self._generate_text_batch,
            "category": "text-generation"
            },
            "clip": {}}}}}}}}}}}}}}}}}}}}}
            "module": "test_hf_clip",
            "data_generator": self._generate_image_batch,
            "category": "vision"
            },
            "whisper": {}}}}}}}}}}}}}}}}}}}}}
            "module": "test_hf_whisper",
            "data_generator": self._generate_audio_batch,
            "category": "audio"
            },
            "wav2vec2": {}}}}}}}}}}}}}}}}}}}}}
            "module": "test_hf_wav2vec2",
            "data_generator": self._generate_audio_batch,
            "category": "audio"
            }
            }
        
        # Initialize test data path
            self.test_data_dir = os.path.abspath()))))os.path.join()))))os.path.dirname()))))__file__), "test_data"))
            os.makedirs()))))self.test_data_dir, exist_ok=True)
        
        # Default test data
            self.test_text = "The quick brown fox jumps over the lazy dog"
            self.test_image_path = os.path.abspath()))))os.path.join()))))os.path.dirname()))))__file__), "test.jpg"))
            self.test_audio_path = os.path.abspath()))))os.path.join()))))os.path.dirname()))))__file__), "test.mp3"))
        
        # Track test modules
            self.test_modules = {}}}}}}}}}}}}}}}}}}}}}}
    
    def _import_test_module()))))self, module_name: str) -> Any:
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
            module_path = f"skills.{}}}}}}}}}}}}}}}}}}}}}module_name}"
            module = importlib.import_module()))))module_path)
            self.test_modules[module_name] = module,,
            return module
        except ImportError:
            print()))))f"Warning: Module {}}}}}}}}}}}}}}}}}}}}}module_path} not found, trying alternate path")
            try:
                # Try to import directly from test directory
                module = importlib.import_module()))))module_name)
                self.test_modules[module_name] = module,,
            return module
            except ImportError as e:
                print()))))f"Error importing module {}}}}}}}}}}}}}}}}}}}}}module_name}: {}}}}}}}}}}}}}}}}}}}}}e}")
            return None
    
    def _get_test_class()))))self, module_name: str) -> Any:
        """
        Get the test class from the module.
        
        Args:
            module_name: The name of the test module
            
        Returns:
            Any: The test class, or None if not found
            """
        module = self._import_test_module()))))module_name):
        if not module:
            return None
        
        # Look for a class with the same name as the module
            class_name = module_name
        if class_name.startswith()))))"test_"):
            class_name = class_name[5:]
            ,
        # Try to find the class
            test_class = getattr()))))module, f"test_{}}}}}}}}}}}}}}}}}}}}}class_name}", None)
        if test_class:
            return test_class
        
        # Try alternative class name formats
        for attr_name in dir()))))module):
            if attr_name.lower()))))) == class_name.lower()))))) or attr_name.lower()))))) == f"test_{}}}}}}}}}}}}}}}}}}}}}class_name.lower())))))}":
            return getattr()))))module, attr_name)
        
            print()))))f"Error: Could not find test class in module {}}}}}}}}}}}}}}}}}}}}}module_name}")
            return None
    
            def _generate_text_batch()))))self, batch_size: int) -> List[str]:,,,
            """
            Generate a batch of text inputs for testing.
        
        Args:
            batch_size: The batch size
            
        Returns:
            List[str]:,,, A batch of text inputs
            """
        # Create variations of the test text to simulate real batch inputs
            return [f"{}}}}}}}}}}}}}}}}}}}}}self.test_text} ()))))batch item {}}}}}}}}}}}}}}}}}}}}}i})" for i in range()))))batch_size)]:,
            def _generate_image_batch()))))self, batch_size: int) -> List[str]:,,,
            """
            Generate a batch of image inputs for testing.
        
        Args:
            batch_size: The batch size
            
        Returns:
            List[str]:,,, A batch of image file paths
            """
        # For simplicity, just return the same image path multiple times
            return [self.test_image_path for _ in range()))))batch_size)]:,
            def _generate_audio_batch()))))self, batch_size: int) -> List[str]:,,,
            """
            Generate a batch of audio inputs for testing.
        
        Args:
            batch_size: The batch size
            
        Returns:
            List[str]:,,, A batch of audio file paths
            """
        # For simplicity, just return the same audio path multiple times
            return [self.test_audio_path for _ in range()))))batch_size)]:,
    def _create_batch_handler()))))self, handler, model_type: str, batch_size: int) -> callable:
        """
        Create a batch handler function that processes inputs in batches.
        
        Args:
            handler: The original handler function
            model_type: The type of model
            batch_size: The batch size
            
        Returns:
            callable: A batch handler function
            """
        # Define the batch handler function
        def batch_handler()))))inputs):
            # If inputs is not a list, convert it to a list
            if not isinstance()))))inputs, list):
                inputs = [inputs]
                ,
            # Check if we need to pad or truncate the batch:
            if len()))))inputs) < batch_size:
                # Pad with copies of the first input
                inputs = inputs + [inputs[0]] * ()))))batch_size - len()))))inputs)),
            elif len()))))inputs) > batch_size:
                # Truncate to the batch size
                inputs = inputs[:batch_size]
                ,
            # Process the entire batch at once
                if model_type in ["bert", "t5", "llama"]:,,
                # For text models, we can often batch at the handler level
                try:
                return handler()))))inputs)
                except Exception as e:
                    print()))))f"Error processing batch at handler level: {}}}}}}}}}}}}}}}}}}}}}e}")
                    # Fall back to individual processing
                return [handler()))))input_item) for input_item in inputs]:,
            else:
                # For other models, process inputs individually and collect results
                return [handler()))))input_item) for input_item in inputs]:,
        
                return batch_handler
    
    def _modify_handler_for_batch()))))self, handler, model_instance, model_type: str, platform: str, batch_size: int) -> callable:
        """
        Modify the handler to support batched processing.
        
        Args:
            handler: The original handler function
            model_instance: The model instance
            model_type: The type of model
            platform: The platform
            batch_size: The batch size
            
        Returns:
            callable: A modified handler function that supports batched processing
            """
        # Get the method that creates handlers for this platform
        if platform == "cuda":
            handler_creator = getattr()))))model_instance, "init_cuda", None)
        elif platform == "openvino":
            handler_creator = getattr()))))model_instance, "init_openvino", None)
        else:  # Default to CPU
            handler_creator = getattr()))))model_instance, "init_cpu", None)
        
        if not handler_creator:
            print()))))f"No handler creator found for platform {}}}}}}}}}}}}}}}}}}}}}platform}")
            return handler
        
        # Get the model name from the model instance
            model_name = getattr()))))model_instance, "model_name", None)
        if not model_name:
            print()))))"No model name found, cannot create batch handler")
            return handler
        
        # Try to create a new handler with batch support
        try:
            if model_type in ["bert", "t5", "llama"]:,,
                # For these model types, the handler initialization already supports batch_size
                # Try to initialize with explicit batch size
            endpoint, tokenizer, batch_handler, queue, actual_batch_size = handler_creator()))))
            model_name,
            f"{}}}}}}}}}}}}}}}}}}}}}platform}",
            f"{}}}}}}}}}}}}}}}}}}}}}platform}:0",
            batch_size=batch_size
            )
                
                # Check if the batch size was respected:
                if actual_batch_size == batch_size:
                    print()))))f"Successfully created batch handler with batch size {}}}}}}}}}}}}}}}}}}}}}batch_size}")
            return batch_handler
                else:
                    print()))))f"Batch size not respected: requested {}}}}}}}}}}}}}}}}}}}}}batch_size}, got {}}}}}}}}}}}}}}}}}}}}}actual_batch_size}")
            
            # Fall back to creating a wrapper around the existing handler
            return self._create_batch_handler()))))handler, model_type, batch_size)
            
        except Exception as e:
            print()))))f"Error creating batch handler: {}}}}}}}}}}}}}}}}}}}}}e}")
            # Fall back to creating a wrapper around the existing handler
            return self._create_batch_handler()))))handler, model_type, batch_size)
    
    def _measure_batch_performance()))))self, handler, inputs, batch_size: int, platform: str, use_fp16: bool = False) -> Dict:
        """
        Measure the performance of batch inference.
        
        Args:
            handler: The handler function
            inputs: The batch inputs
            batch_size: The batch size
            platform: The platform
            use_fp16: Whether to use FP16 precision
            
        Returns:
            Dict: Performance metrics
            """
        # Initialize performance metrics
            metrics = {}}}}}}}}}}}}}}}}}}}}}
            "batch_size": batch_size,
            "platform": platform,
            "use_fp16": use_fp16,
            "input_count": len()))))inputs),
            "timestamp": datetime.datetime.now()))))).isoformat())))))
            }
        
        # Measure CUDA memory usage before inference
            cuda_memory_before = 0
        if platform == "cuda" and torch.cuda.is_available()))))):
            torch.cuda.synchronize())))))
            cuda_memory_before = torch.cuda.memory_allocated()))))) / ()))))1024 * 1024)  # MB
        
        # Run inference with timing
        try:
            start_time = time.time())))))
            
            # Perform inference
            outputs = handler()))))inputs)
            
            # Ensure everything is finished ()))))important for CUDA)
            if platform == "cuda" and torch.cuda.is_available()))))):
                torch.cuda.synchronize())))))
            
                end_time = time.time())))))
                inference_time = end_time - start_time
            
            # Calculate metric: inputs per second
                inputs_per_second = len()))))inputs) / inference_time if inference_time > 0 else 0
            
            # Record basic metrics
                metrics["status"] = "Success",,
                metrics["inference_time_seconds"] = inference_time,
                metrics["inputs_per_second"] = inputs_per_second,
                metrics["average_latency_seconds"] = inference_time / len()))))inputs) if len()))))inputs) > 0 else 0
                ,
            # Measure CUDA memory usage after inference:
            if platform == "cuda" and torch.cuda.is_available()))))):
                torch.cuda.synchronize())))))
                cuda_memory_after = torch.cuda.memory_allocated()))))) / ()))))1024 * 1024)  # MB
                metrics["cuda_memory_before_mb"] = cuda_memory_before,
                metrics["cuda_memory_after_mb"] = cuda_memory_after,
                metrics["cuda_memory_used_mb"] = cuda_memory_after - cuda_memory_before
                ,
            # Check outputs
            if isinstance()))))outputs, list):
                metrics["output_count"] = len()))))outputs)
                ,
                # Get output shapes if possible
                output_shapes = []:,
                for output in outputs:
                    if isinstance()))))output, torch.Tensor):
                        output_shapes.append()))))list()))))output.shape))
                    elif isinstance()))))output, np.ndarray):
                        output_shapes.append()))))list()))))output.shape))
                    elif isinstance()))))output, dict) and "embedding" in output and hasattr()))))output["embedding"], "shape"):,,
                        output_shapes.append()))))list()))))output["embedding"].shape)),
                    elif isinstance()))))output, dict) and "logits" in output and hasattr()))))output["logits"], "shape"):,,
                    output_shapes.append()))))list()))))output["logits"].shape))
                    ,
                if output_shapes:
                    metrics["output_shapes"] = output_shapes,
            else:
                # Single output for the whole batch
                if isinstance()))))outputs, torch.Tensor):
                    metrics["output_shape"] = list()))))outputs.shape),,
                elif isinstance()))))outputs, np.ndarray):
                    metrics["output_shape"] = list()))))outputs.shape),,
                elif isinstance()))))outputs, dict) and "embedding" in outputs and hasattr()))))outputs["embedding"], "shape"):,,
                    metrics["output_shape"] = list()))))outputs["embedding"].shape),
                elif isinstance()))))outputs, dict) and "logits" in outputs and hasattr()))))outputs["logits"], "shape"):,,
                metrics["output_shape"] = list()))))outputs["logits"].shape)
                ,
        except Exception as e:
            metrics["status"] = "Failed",
            metrics["error"] = str()))))e),
            metrics["traceback"] = traceback.format_exc())))))
            ,
                return metrics
    
    def test_model_type()))))self, model_type: str) -> Dict:
        """
        Test batch inference for a specific model type across platforms and batch sizes.
        
        Args:
            model_type: The type of model to test
            
        Returns:
            Dict: Test results for this model type
            """
            print()))))f"\n{}}}}}}}}}}}}}}}}}}}}}'='*80}\nTesting batch inference for model type: {}}}}}}}}}}}}}}}}}}}}}model_type}\n{}}}}}}}}}}}}}}}}}}}}}'='*80}")
        
        # Check if model type is supported:
        if model_type not in self.model_type_mapping:
            return {}}}}}}}}}}}}}}}}}}}}}
            "model_type": model_type,
            "status": "Failed",
            "error": f"Unsupported model type: {}}}}}}}}}}}}}}}}}}}}}model_type}"
            }
        
        # Get model mapping
            model_mapping = self.model_type_mapping[model_type],
            module_name = model_mapping["module"],
            data_generator = model_mapping["data_generator"],
            category = model_mapping["category"]
            ,
        # Get test class
            test_class = self._get_test_class()))))module_name)
        if not test_class:
            return {}}}}}}}}}}}}}}}}}}}}}
            "model_type": model_type,
            "status": "Failed",
            "error": f"Could not find test class for module: {}}}}}}}}}}}}}}}}}}}}}module_name}"
            }
        
        # Initialize test instance
        try:
            # Check if we have a specific model for this type
            model_name = self.specific_models.get()))))model_type)
            
            # Create a copy of resources and metadata for this test
            test_resources = self.resources.copy())))))
            test_metadata = self.metadata.copy())))))
            
            # Add model_name to metadata if specified::
            if model_name:
                test_metadata["model_name"] = model_name
                ,
            # Create test instance
                test_instance = test_class()))))resources=test_resources, metadata=test_metadata)
            
            # Override model name if specified:
            if model_name and hasattr()))))test_instance, "model_name"):
                print()))))f"Using specified model: {}}}}}}}}}}}}}}}}}}}}}model_name}")
                test_instance.model_name = model_name
            
                model_results = {}}}}}}}}}}}}}}}}}}}}}
                "model_type": model_type,
                "model_name": getattr()))))test_instance, "model_name", "Unknown"),
                "category": category,
                "platforms": {}}}}}}}}}}}}}}}}}}}}}},
                "status": "Not tested",
                "timestamp": datetime.datetime.now()))))).isoformat())))))
                }
            
            # Test each platform
            for platform in self.platforms:
                print()))))f"\nTesting {}}}}}}}}}}}}}}}}}}}}}model_type} on platform: {}}}}}}}}}}}}}}}}}}}}}platform}")
                
                # Skip CUDA tests if not available::
                if platform == "cuda" and not torch.cuda.is_available()))))):
                    model_results["platforms"][platform] = {}}}}}}}}}}}}}}}}}}}}},,,,
                    "status": "Skipped",
                    "error": "CUDA not available"
                    }
                continue
                
                # Skip OpenVINO tests if not available::
                if platform == "openvino":
                    try:
                        import openvino
                    except ImportError:
                        model_results["platforms"][platform] = {}}}}}}}}}}}}}}}}}}}}},,,,
                        "status": "Skipped",
                        "error": "OpenVINO not installed"
                        }
                        continue
                
                # Run the test method to get handlers
                if hasattr()))))test_instance, "test") and callable()))))test_instance.test):
                    test_result = test_instance.test())))))
                else:
                    model_results["platforms"][platform] = {}}}}}}}}}}}}}}}}}}}}},,,,
                    "status": "Failed",
                    "error": "No test method available"
                    }
                    continue
                
                # Extract handler for this platform
                    handler = None
                
                # Look for platform-specific handler in examples
                if isinstance()))))test_result, dict) and "examples" in test_result:
                    for example in test_result["examples"]:,
                    if isinstance()))))example, dict) and "platform" in example and platform.lower()))))) in example["platform"].lower()))))):,
                            # Get the handler used to produce this example
                            if "handler" in example:
                                handler = example["handler"],
                    break
                
                # If no handler found in examples, try alternative approaches
                if not handler:
                    # Try calling the test_instance's method directly
                    platform_method = getattr()))))test_instance, f"test_{}}}}}}}}}}}}}}}}}}}}}platform}", None)
                    if platform_method and callable()))))platform_method):
                        print()))))f"Using test_{}}}}}}}}}}}}}}}}}}}}}platform} method")
                        platform_result = platform_method())))))
                        if "handler" in platform_result:
                            handler = platform_result["handler"],
                
                # If still no handler, check if the test class has a direct handler creation method:
                if not handler:
                    handler_method = None
                    if platform == "cuda":
                        handler_method = getattr()))))test_instance, "init_cuda", None)
                    elif platform == "openvino":
                        handler_method = getattr()))))test_instance, "init_openvino", None)
                    else:
                        handler_method = getattr()))))test_instance, "init_cpu", None)
                    
                    if handler_method and callable()))))handler_method):
                        try:
                            print()))))f"Using init_{}}}}}}}}}}}}}}}}}}}}}platform} method")
                            model_name = getattr()))))test_instance, "model_name", None)
                            if model_name:
                                # Call handler method
                                _, _, handler, _, _ = handler_method()))))
                                model_name,
                                f"{}}}}}}}}}}}}}}}}}}}}}platform}",
                                f"{}}}}}}}}}}}}}}}}}}}}}platform}:0"
                                )
                        except Exception as e:
                            print()))))f"Error creating handler: {}}}}}}}}}}}}}}}}}}}}}e}")
                
                # If no handler found, report failure
                if not handler:
                    model_results["platforms"][platform] = {}}}}}}}}}}}}}}}}}}}}},,,,
                    "status": "Failed",
                    "error": f"Could not find handler for platform {}}}}}}}}}}}}}}}}}}}}}platform}"
                    }
                            continue
                
                # Initialize platform results
                            platform_results = {}}}}}}}}}}}}}}}}}}}}}
                            "status": "Success",
                            "batch_results": {}}}}}}}}}}}}}}}}}}}}}}
                            }
                
                # Test each batch size
                for batch_size in self.batch_sizes:
                    print()))))f"  Testing batch size: {}}}}}}}}}}}}}}}}}}}}}batch_size}")
                    
                    # Generate batch inputs
                    batch_inputs = data_generator()))))batch_size)
                    
                    # Create a handler with batch support
                    batch_handler = self._modify_handler_for_batch()))))
                    handler, test_instance, model_type, platform, batch_size
                    )
                    
                    # Measure batch performance
                    batch_metrics = self._measure_batch_performance()))))
                    batch_handler, batch_inputs, batch_size, platform, self.use_fp16
                    )
                    
                    # Store batch results
                    platform_results["batch_results"][batch_size],,,, = batch_metrics
                    ,
                    # Clean up CUDA memory after each batch test
                    if platform == "cuda" and torch.cuda.is_available()))))):
                        torch.cuda.empty_cache())))))
                
                # Calculate performance scaling across batch sizes
                        batch_throughputs = {}}}}}}}}}}}}}}}}}}}}}}
                        for batch_size, metrics in platform_results["batch_results"].items()))))):,
                        if metrics["status"] == "Success":,,,,,,
                        batch_throughputs[batch_size],, = metrics.get()))))"inputs_per_second", 0)
                        ,
                if batch_throughputs:
                    # Calculate speedup relative to batch size 1
                    base_throughput = batch_throughputs.get()))))1, None)
                    if base_throughput:
                        scaling = {}}}}}}}}}}}}}}}}}}}}}}
                        for batch_size, throughput in batch_throughputs.items()))))):
                            scaling[batch_size],, = throughput / base_throughput,
                            platform_results["throughput_scaling"] = scaling
                            ,
                        # Calculate efficiency ()))))scaling / batch_size)
                            efficiency = {}}}}}}}}}}}}}}}}}}}}}}
                        for batch_size, scale_factor in scaling.items()))))):
                            efficiency[batch_size],, = scale_factor / batch_size,
                            platform_results["batch_efficiency"] = efficiency
                            ,
                # Store platform results
                            model_results["platforms"][platform] = platform_results
                            ,
            # Set overall model status
                            if any()))))platform["status"] == "Success" for platform in model_results["platforms"].values())))))):,
                            model_results["status"] = "Success",,
            else:
                model_results["status"] = "Failed",
            
                            return model_results
            
        except Exception as e:
            print()))))f"Error testing model type {}}}}}}}}}}}}}}}}}}}}}model_type}: {}}}}}}}}}}}}}}}}}}}}}e}")
            traceback.print_exc())))))
                            return {}}}}}}}}}}}}}}}}}}}}}
                            "model_type": model_type,
                            "status": "Failed",
                            "error": str()))))e),
                            "traceback": traceback.format_exc())))))
                            }
    
    def run_tests()))))self) -> Dict:
        """
        Run batch inference tests for all specified model types.
        
        Returns:
            Dict: Test results
            """
        # Set up test environment
            print()))))f"Running batch inference tests for {}}}}}}}}}}}}}}}}}}}}}len()))))self.model_types)} model types")
            print()))))f"Batch sizes: {}}}}}}}}}}}}}}}}}}}}}self.batch_sizes}")
            print()))))f"Platforms: {}}}}}}}}}}}}}}}}}}}}}self.platforms}")
        
            start_time = time.time())))))
        
        # Test each model type
        for model_type in self.model_types:
            model_results = self.test_model_type()))))model_type)
            self.results["model_results"][model_type], = model_results
        
            elapsed_time = time.time()))))) - start_time
        
        # Update metadata
            self.results["metadata"]["test_duration_seconds"] = elapsed_time
            ,
        # Calculate summary statistics
            self.results["summary"], = self._calculate_summary())))))
            ,
        # Save results
            self._save_results())))))
        
        # Print summary
            self._print_summary())))))
        
            return self.results
    
    def _calculate_summary()))))self) -> Dict:
        """
        Calculate summary statistics from test results.
        
        Returns:
            Dict: Summary statistics
            """
            summary = {}}}}}}}}}}}}}}}}}}}}}
            "total_models": len()))))self.results["model_results"]),
            "successful_models": 0,
            "platforms": {}}}}}}}}}}}}}}}}}}}}}},
            "categories": {}}}}}}}}}}}}}}}}}}}}}},
            "batch_sizes": {}}}}}}}}}}}}}}}}}}}}}}
            }
        
        # Count successful models
            for model_type, model_result in self.results["model_results"].items()))))):,,,,,,,
            if model_result["status"] == "Success":,,,,,,
            summary["successful_models"] += 1
            ,
                # Update category statistics
            category = model_result.get()))))"category", "unknown")
            if category not in summary["categories"]:,
            summary["categories"][category] = {}}}}}}}}}}}}}}}}}}}}}"successful": 0, "total": 0},
            summary["categories"][category]["total"] += 1,
            if model_result["status"] == "Success":,,,,,,
            summary["categories"][category]["successful"] += 1
            ,
                # Update platform statistics
            for platform, platform_result in model_result["platforms"].items()))))):,,,,,,,,,,
            if platform not in summary["platforms"]:,
            summary["platforms"][platform] = {}}}}}}}}}}}}}}}}}}}}},,,,"successful": 0, "total": 0}
            summary["platforms"][platform]["total"] += 1,
            if platform_result["status"] == "Success":,,,,,,
            summary["platforms"][platform]["successful"] += 1
            ,
                        # Update batch size statistics
                        for batch_size, batch_result in platform_result.get()))))"batch_results", {}}}}}}}}}}}}}}}}}}}}}}).items()))))):
                            if str()))))batch_size) not in summary["batch_sizes"]:,
                            summary["batch_sizes"][str()))))batch_size)] = {}}}}}}}}}}}}}}}}}}}}}"successful": 0, "total": 0},
                            summary["batch_sizes"][str()))))batch_size)]["total"] += 1,
                            if batch_result["status"] == "Success":,,,,,,
                            summary["batch_sizes"][str()))))batch_size)]["successful"] += 1
                            ,
        # Calculate average throughput scaling across models and platforms
                            scaling_data = {}}}}}}}}}}}}}}}}}}}}}}
                            for model_type, model_result in self.results["model_results"].items()))))):,,,,,,,
                            for platform, platform_result in model_result["platforms"].items()))))):,,,,,,,,,,
                if "throughput_scaling" in platform_result:
                    for batch_size, scale_factor in platform_result["throughput_scaling"].items()))))):,
                    key = f"{}}}}}}}}}}}}}}}}}}}}}platform}_{}}}}}}}}}}}}}}}}}}}}}batch_size}"
                        if key not in scaling_data:
                            scaling_data[key] = [],
                            scaling_data[key].append()))))scale_factor)
                            ,
        # Calculate averages
                            avg_scaling = {}}}}}}}}}}}}}}}}}}}}}}
        for key, values in scaling_data.items()))))):
            if values:
                avg_scaling[key] = sum()))))values) / len()))))values)
                ,
                summary["average_throughput_scaling"] = avg_scaling
                ,
        # Calculate success rate
                summary["success_rate"], = summary["successful_models"] / summary["total_models"] if summary["total_models"] > 0 else 0
                ,
        # Calculate platform success rates:
                for platform, stats in summary["platforms"].items()))))):,,,,,,,,,,
                stats["success_rate"], = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
                ,,,
        # Calculate category success rates:
                for category, stats in summary["categories"].items()))))):,,,
                stats["success_rate"], = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
                ,,,
        # Calculate batch size success rates:
                for batch_size, stats in summary["batch_sizes"].items()))))):,,
                stats["success_rate"], = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
                ,,,
            return summary
    :
    def _save_results()))))self) -> None:
        """
        Save the test results to a JSON file.
        """
        timestamp = datetime.datetime.now()))))).strftime()))))"%Y%m%d_%H%M%S")
        filename = f"batch_inference_results_{}}}}}}}}}}}}}}}}}}}}}timestamp}.json"
        filepath = os.path.join()))))TEST_RESULTS_DIR, filename)
        
        try:
            with open()))))filepath, 'w') as f:
                json.dump()))))self.results, f, indent=2)
                print()))))f"\nTest results saved to {}}}}}}}}}}}}}}}}}}}}}filepath}")
            
            # Also generate a markdown report
                self._generate_markdown_report()))))timestamp)
        except Exception as e:
            print()))))f"Error saving results: {}}}}}}}}}}}}}}}}}}}}}e}")
    
    def _generate_markdown_report()))))self, timestamp: str) -> None:
        """
        Generate a markdown report from the test results.
        
        Args:
            timestamp: The timestamp for the report filename
            """
            report_filename = f"batch_inference_report_{}}}}}}}}}}}}}}}}}}}}}timestamp}.md"
            report_filepath = os.path.join()))))TEST_RESULTS_DIR, report_filename)
        
        try:
            with open()))))report_filepath, 'w') as f:
                # Write report header
                f.write()))))f"# Batch Inference Test Report\n\n")
                f.write()))))f"Generated: {}}}}}}}}}}}}}}}}}}}}}datetime.datetime.now()))))).isoformat())))))}\n\n")
                
                # Write test configuration
                f.write()))))"## Test Configuration\n\n")
                f.write()))))f"- Model types: {}}}}}}}}}}}}}}}}}}}}}', '.join()))))self.model_types)}\n")
                f.write()))))f"- Batch sizes: {}}}}}}}}}}}}}}}}}}}}}self.batch_sizes}\n")
                f.write()))))f"- Platforms: {}}}}}}}}}}}}}}}}}}}}}self.platforms}\n")
                f.write()))))f"- FP16 precision: {}}}}}}}}}}}}}}}}}}}}}self.use_fp16}\n")
                f.write()))))f"- Test duration: {}}}}}}}}}}}}}}}}}}}}}self.results['metadata'].get()))))'test_duration_seconds', 0):.2f} seconds\n\n")
                ,
                # Write summary
                summary = self.results["summary"],
                f.write()))))"## Test Summary\n\n")
                f.write()))))f"- Total models tested: {}}}}}}}}}}}}}}}}}}}}}summary['total_models']}\n"),
                f.write()))))f"- Successful models: {}}}}}}}}}}}}}}}}}}}}}summary['successful_models']}\n"),
                f.write()))))f"- Success rate: {}}}}}}}}}}}}}}}}}}}}}summary['success_rate']*100:.1f}%\n\n")
                ,
                # Write platform summary
                f.write()))))"### Platform Results\n\n")
                f.write()))))"| Platform | Success | Total | Success Rate |\n")
                f.write()))))"|----------|---------|-------|-------------|\n")
                
                for platform, stats in summary["platforms"].items()))))):,,,,,,,,,,
                f.write()))))f"| {}}}}}}}}}}}}}}}}}}}}}platform} | {}}}}}}}}}}}}}}}}}}}}}stats['successful']} | {}}}}}}}}}}}}}}}}}}}}}stats['total']} | {}}}}}}}}}}}}}}}}}}}}}stats['success_rate']*100:.1f}% |\n")
                ,,,
                f.write()))))"\n")
                
                # Write category summary
                f.write()))))"### Category Results\n\n")
                f.write()))))"| Category | Success | Total | Success Rate |\n")
                f.write()))))"|----------|---------|-------|-------------|\n")
                
                for category, stats in summary["categories"].items()))))):,,,
                f.write()))))f"| {}}}}}}}}}}}}}}}}}}}}}category} | {}}}}}}}}}}}}}}}}}}}}}stats['successful']} | {}}}}}}}}}}}}}}}}}}}}}stats['total']} | {}}}}}}}}}}}}}}}}}}}}}stats['success_rate']*100:.1f}% |\n")
                ,,,
                f.write()))))"\n")
                
                # Write batch size summary
                f.write()))))"### Batch Size Results\n\n")
                f.write()))))"| Batch Size | Success | Total | Success Rate |\n")
                f.write()))))"|------------|---------|-------|-------------|\n")
                
                for batch_size, stats in summary["batch_sizes"].items()))))):,,
                f.write()))))f"| {}}}}}}}}}}}}}}}}}}}}}batch_size} | {}}}}}}}}}}}}}}}}}}}}}stats['successful']} | {}}}}}}}}}}}}}}}}}}}}}stats['total']} | {}}}}}}}}}}}}}}}}}}}}}stats['success_rate']*100:.1f}% |\n")
                ,,,
                f.write()))))"\n")
                
                # Write throughput scaling data
                if "average_throughput_scaling" in summary:
                    f.write()))))"## Throughput Scaling\n\n")
                    f.write()))))"Average speedup factor relative to batch size 1:\n\n")
                    
                    # Group by platform
                    platform_scaling = {}}}}}}}}}}}}}}}}}}}}}}
                    for key, value in summary["average_throughput_scaling"].items()))))):,,
                    platform, batch_size = key.split()))))"_")
                        if platform not in platform_scaling:
                            platform_scaling[platform] = {}}}}}}}}}}}}}}}}}}}}}},,
                            platform_scaling[platform][int()))))batch_size)] = value
                            ,        ,
                    # Write scaling table for each platform
                    for platform, scaling in platform_scaling.items()))))):
                        f.write()))))f"### {}}}}}}}}}}}}}}}}}}}}}platform.upper())))))}\n\n")
                        f.write()))))"| Batch Size | Speedup Factor | Efficiency |\n")
                        f.write()))))"|------------|---------------|------------|\n")
                        
                        for batch_size in sorted()))))scaling.keys())))))):
                            speedup = scaling[batch_size],,
                            efficiency = speedup / batch_size
                            f.write()))))f"| {}}}}}}}}}}}}}}}}}}}}}batch_size} | {}}}}}}}}}}}}}}}}}}}}}speedup:.2f}x | {}}}}}}}}}}}}}}}}}}}}}efficiency*100:.1f}% |\n")
                        
                            f.write()))))"\n")
                
                # Write detailed model results
                            f.write()))))"## Detailed Model Results\n\n")
                
                            for model_type, model_result in self.results["model_results"].items()))))):,,,,,,,
                            f.write()))))f"### {}}}}}}}}}}}}}}}}}}}}}model_type}\n\n")
                            f.write()))))f"- Model name: {}}}}}}}}}}}}}}}}}}}}}model_result.get()))))'model_name', 'Unknown')}\n")
                            f.write()))))f"- Category: {}}}}}}}}}}}}}}}}}}}}}model_result.get()))))'category', 'Unknown')}\n")
                            f.write()))))f"- Status: {}}}}}}}}}}}}}}}}}}}}}model_result.get()))))'status', 'Unknown')}\n\n")
                    
                    if "error" in model_result:
                        f.write()))))f"- Error: {}}}}}}}}}}}}}}}}}}}}}model_result['error']}\n\n"),,
                            continue
                    
                    # Write platform-specific results
                            for platform, platform_result in model_result["platforms"].items()))))):,,,,,,,,,,
                            f.write()))))f"#### {}}}}}}}}}}}}}}}}}}}}}platform.upper())))))}\n\n")
                            f.write()))))f"- Status: {}}}}}}}}}}}}}}}}}}}}}platform_result.get()))))'status', 'Unknown')}\n")
                        
                        if platform_result.get()))))"status") != "Success":
                            if "error" in platform_result:
                                f.write()))))f"- Error: {}}}}}}}}}}}}}}}}}}}}}platform_result['error']}\n\n"),,
                            continue
                        
                        # Write batch results
                            f.write()))))"\nBatch performance:\n\n")
                            f.write()))))"| Batch Size | Inputs/Sec | Avg Latency ()))))ms) | Status |\n")
                            f.write()))))"|------------|------------|------------------|--------|\n")
                        
                        for batch_size in sorted()))))platform_result.get()))))"batch_results", {}}}}}}}}}}}}}}}}}}}}}}).keys())))))):
                            batch_result = platform_result["batch_results"][batch_size],,,,
                            inputs_per_sec = batch_result.get()))))"inputs_per_second", 0)
                            latency_ms = batch_result.get()))))"average_latency_seconds", 0) * 1000  # Convert to ms
                            status = batch_result.get()))))"status", "Unknown")
                            
                            f.write()))))f"| {}}}}}}}}}}}}}}}}}}}}}batch_size} | {}}}}}}}}}}}}}}}}}}}}}inputs_per_sec:.2f} | {}}}}}}}}}}}}}}}}}}}}}latency_ms:.2f} | {}}}}}}}}}}}}}}}}}}}}}status} |\n")
                        
                            f.write()))))"\n")
                        
                        # Write memory usage if available: ()))))CUDA only):
                        if platform == "cuda":
                            f.write()))))"\nMemory usage:\n\n")
                            f.write()))))"| Batch Size | GPU Memory ()))))MB) |\n")
                            f.write()))))"|------------|----------------|\n")
                            
                            for batch_size in sorted()))))platform_result.get()))))"batch_results", {}}}}}}}}}}}}}}}}}}}}}}).keys())))))):
                                batch_result = platform_result["batch_results"][batch_size],,,,
                                memory_mb = batch_result.get()))))"cuda_memory_used_mb", 0)
                                
                                f.write()))))f"| {}}}}}}}}}}}}}}}}}}}}}batch_size} | {}}}}}}}}}}}}}}}}}}}}}memory_mb:.2f} |\n")
                            
                                f.write()))))"\n")
                        
                        # Write throughput scaling if available:
                        if "throughput_scaling" in platform_result:
                            f.write()))))"\nThroughput scaling:\n\n")
                            f.write()))))"| Batch Size | Speedup | Efficiency |\n")
                            f.write()))))"|------------|---------|------------|\n")
                            
                            for batch_size in sorted()))))platform_result["throughput_scaling"].keys())))))),:,
                            speedup = platform_result["throughput_scaling"][batch_size],,,
                            efficiency = platform_result["batch_efficiency"].get()))))batch_size, 0)
                            ,
                            f.write()))))f"| {}}}}}}}}}}}}}}}}}}}}}batch_size} | {}}}}}}}}}}}}}}}}}}}}}speedup:.2f}x | {}}}}}}}}}}}}}}}}}}}}}efficiency*100:.1f}% |\n")
                            
                            f.write()))))"\n")
                
                # Write conclusion
                            f.write()))))"## Conclusion\n\n")
                
                            success_rate = summary["success_rate"],
                if success_rate > 0.8:
                    f.write()))))"The batch inference testing shows strong support across most model types. The implementation effectively scales with batch size, although with varying efficiency between model types.\n\n")
                elif success_rate > 0.5:
                    f.write()))))"The batch inference testing shows good support for many model types, but some limitations exist. Further optimization is needed to improve scaling efficiency and ensure consistent support across all models.\n\n")
                else:
                    f.write()))))"The batch inference testing revealed significant limitations in batch support. Major improvements are needed before batch processing can be reliably used in production.\n\n")
                
                # Write recommendations
                    f.write()))))"### Recommendations\n\n")
                
                # Find models with best scaling
                    best_scaling = {}}}}}}}}}}}}}}}}}}}}}}
                    for model_type, model_result in self.results["model_results"].items()))))):,,,,,,,
                    for platform, platform_result in model_result["platforms"].items()))))):,,,,,,,,,,
                    if "throughput_scaling" in platform_result and platform_result["throughput_scaling"]:,
                            # Get scaling at highest batch size
                    max_batch_size = max()))))platform_result["throughput_scaling"].keys())))))),
                    scaling = platform_result["throughput_scaling"][max_batch_size],
                    best_scaling[f"{}}}}}}}}}}}}}}}}}}}}}model_type}_{}}}}}}}}}}}}}}}}}}}}}platform}"] = ()))))scaling, max_batch_size)
                    ,
                if best_scaling:
                    best_model, ()))))best_scale, best_batch) = max()))))best_scaling.items()))))), key=lambda x: x[1][0]),,
                    model_type, platform = best_model.split()))))"_")
                    f.write()))))f"1. The {}}}}}}}}}}}}}}}}}}}}}model_type} model on {}}}}}}}}}}}}}}}}}}}}}platform} showed the best scaling, with a {}}}}}}}}}}}}}}}}}}}}}best_scale:.2f}x speedup at batch size {}}}}}}}}}}}}}}}}}}}}}best_batch}. Consider prioritizing optimization efforts for other models based on this architecture.\n")
                
                # Identify inefficient models
                    poor_scaling = {}}}}}}}}}}}}}}}}}}}}}}
                    for model_type, model_result in self.results["model_results"].items()))))):,,,,,,,
                    for platform, platform_result in model_result["platforms"].items()))))):,,,,,,,,,,
                    if "batch_efficiency" in platform_result and platform_result["batch_efficiency"]:,
                            # Get efficiency at highest batch size
                    max_batch_size = max()))))platform_result["batch_efficiency"].keys())))))),
                    efficiency = platform_result["batch_efficiency"][max_batch_size],
                    if efficiency < 0.7:  # Less than 70% efficient
                    poor_scaling[f"{}}}}}}}}}}}}}}}}}}}}}model_type}_{}}}}}}}}}}}}}}}}}}}}}platform}"] = ()))))efficiency, max_batch_size)
                    ,
                if poor_scaling:
                    worst_model, ()))))worst_eff, worst_batch) = min()))))poor_scaling.items()))))), key=lambda x: x[1][0]),,
                    model_type, platform = worst_model.split()))))"_")
                    f.write()))))f"2. The {}}}}}}}}}}}}}}}}}}}}}model_type} model on {}}}}}}}}}}}}}}}}}}}}}platform} showed poor scaling efficiency ())))){}}}}}}}}}}}}}}}}}}}}}worst_eff*100:.1f}%) at batch size {}}}}}}}}}}}}}}}}}}}}}worst_batch}. Investigate potential bottlenecks in the data processing pipeline.\n")
                
                    f.write()))))"3. For optimal performance, models should adapt batch size dynamically based on available resources.\n")
                    f.write()))))"4. Implement proper error handling and fallback mechanisms for batch processing to ensure robustness.\n")
                    f.write()))))"5. Additional testing with larger batch sizes is recommended for high-throughput production use cases.\n")
            
                    print()))))f"Markdown report saved to {}}}}}}}}}}}}}}}}}}}}}report_filepath}")
        except Exception as e:
            print()))))f"Error generating markdown report: {}}}}}}}}}}}}}}}}}}}}}e}")
    
    def _print_summary()))))self) -> None:
        """
        Print a summary of the test results.
        """
        print()))))"\n" + "="*80)
        print()))))"BATCH INFERENCE TEST SUMMARY")
        print()))))"="*80)
        
        summary = self.results["summary"],
        
        print()))))f"Total models tested: {}}}}}}}}}}}}}}}}}}}}}summary['total_models']}"),
        print()))))f"Successful models: {}}}}}}}}}}}}}}}}}}}}}summary['successful_models']}"),
        print()))))f"Success rate: {}}}}}}}}}}}}}}}}}}}}}summary['success_rate']*100:.1f}%")
        ,
        print()))))"\nPLATFORM RESULTS:")
        for platform, stats in summary["platforms"].items()))))):,,,,,,,,,,
        print()))))f"  {}}}}}}}}}}}}}}}}}}}}}platform.upper())))))}: {}}}}}}}}}}}}}}}}}}}}}stats['successful']}/{}}}}}}}}}}}}}}}}}}}}}stats['total']} successful ())))){}}}}}}}}}}}}}}}}}}}}}stats['success_rate']*100:.1f}%)")
        ,,,
        print()))))"\nCATEGORY RESULTS:")
        for category, stats in summary["categories"].items()))))):,,,
        print()))))f"  {}}}}}}}}}}}}}}}}}}}}}category}: {}}}}}}}}}}}}}}}}}}}}}stats['successful']}/{}}}}}}}}}}}}}}}}}}}}}stats['total']} successful ())))){}}}}}}}}}}}}}}}}}}}}}stats['success_rate']*100:.1f}%)")
        ,,,
        print()))))"\nBATCH SIZE RESULTS:")
        for batch_size, stats in sorted()))))summary["batch_sizes"].items()))))), key=lambda x: int()))))x[0])):,
        print()))))f"  Batch size {}}}}}}}}}}}}}}}}}}}}}batch_size}: {}}}}}}}}}}}}}}}}}}}}}stats['successful']}/{}}}}}}}}}}}}}}}}}}}}}stats['total']} successful ())))){}}}}}}}}}}}}}}}}}}}}}stats['success_rate']*100:.1f}%)")
        ,,,
        print()))))"\nTHROUGHPUT SCALING:")
        if "average_throughput_scaling" in summary:
            # Group by platform
            platform_scaling = {}}}}}}}}}}}}}}}}}}}}}}
            for key, value in summary["average_throughput_scaling"].items()))))):,,
            platform, batch_size = key.split()))))"_")
                if platform not in platform_scaling:
                    platform_scaling[platform] = {}}}}}}}}}}}}}}}}}}}}}},,
                    platform_scaling[platform][int()))))batch_size)] = value
                    ,
            for platform, scaling in platform_scaling.items()))))):
                print()))))f"  {}}}}}}}}}}}}}}}}}}}}}platform.upper())))))} average speedup:")
                for batch_size in sorted()))))scaling.keys())))))):
                    speedup = scaling[batch_size],,
                    efficiency = speedup / batch_size
                    print()))))f"    Batch size {}}}}}}}}}}}}}}}}}}}}}batch_size}: {}}}}}}}}}}}}}}}}}}}}}speedup:.2f}x speedup ())))){}}}}}}}}}}}}}}}}}}}}}efficiency*100:.1f}% efficiency)")
        
                    print()))))"\nKEY FINDINGS:")
        # Find highest throughput model/platform/batch size combination
                    highest_throughput = 0
                    highest_config = None
        
                    for model_type, model_result in self.results["model_results"].items()))))):,,,,,,,
                    for platform, platform_result in model_result["platforms"].items()))))):,,,,,,,,,,
                for batch_size, batch_result in platform_result.get()))))"batch_results", {}}}}}}}}}}}}}}}}}}}}}}).items()))))):
                    if batch_result["status"] == "Success":,,,,,,
                    throughput = batch_result.get()))))"inputs_per_second", 0)
                        if throughput > highest_throughput:
                            highest_throughput = throughput
                            highest_config = ()))))model_type, platform, batch_size)
        
        if highest_config:
            model_type, platform, batch_size = highest_config
            print()))))f"  Highest throughput: {}}}}}}}}}}}}}}}}}}}}}model_type} on {}}}}}}}}}}}}}}}}}}}}}platform} with batch size {}}}}}}}}}}}}}}}}}}}}}batch_size} ())))){}}}}}}}}}}}}}}}}}}}}}highest_throughput:.2f} inputs/sec)")
        
        # Find models with best and worst scaling efficiency
            best_efficiency = 0
            best_config = None
            worst_efficiency = 1.0
            worst_config = None
        
            for model_type, model_result in self.results["model_results"].items()))))):,,,,,,,
            for platform, platform_result in model_result["platforms"].items()))))):,,,,,,,,,,
                if "batch_efficiency" in platform_result:
                    for batch_size, efficiency in platform_result["batch_efficiency"].items()))))):,
                        if efficiency > best_efficiency:
                            best_efficiency = efficiency
                            best_config = ()))))model_type, platform, batch_size)
                            if efficiency < worst_efficiency and platform_result["batch_results"].get()))))batch_size, {}}}}}}}}}}}}}}}}}}}}}}).get()))))"status") == "Success":,
                            worst_efficiency = efficiency
                            worst_config = ()))))model_type, platform, batch_size)
        
        if best_config:
            model_type, platform, batch_size = best_config
            print()))))f"  Best scaling efficiency: {}}}}}}}}}}}}}}}}}}}}}model_type} on {}}}}}}}}}}}}}}}}}}}}}platform} with batch size {}}}}}}}}}}}}}}}}}}}}}batch_size} ())))){}}}}}}}}}}}}}}}}}}}}}best_efficiency*100:.1f}%)")
        
        if worst_config:
            model_type, platform, batch_size = worst_config
            print()))))f"  Worst scaling efficiency: {}}}}}}}}}}}}}}}}}}}}}model_type} on {}}}}}}}}}}}}}}}}}}}}}platform} with batch size {}}}}}}}}}}}}}}}}}}}}}batch_size} ())))){}}}}}}}}}}}}}}}}}}}}}worst_efficiency*100:.1f}%)")

def main()))))):
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser()))))description="Test batch inference capabilities")
    parser.add_argument()))))"--model-types", default="bert,t5,clip,llama,whisper,wav2vec2",
    help="Comma-separated list of model types to test ()))))default: bert,t5,clip,llama,whisper,wav2vec2)")
    parser.add_argument()))))"--batch-sizes", default="1,2,4,8,16",
    help="Comma-separated list of batch sizes to test ()))))default: 1,2,4,8,16)")
    parser.add_argument()))))"--platforms", default="cpu,cuda",
    help="Comma-separated list of platforms to test ()))))default: cpu,cuda)")
    parser.add_argument()))))"--specific-model", action="append", default=[],
    help="Specify a model to use for a given type ()))))format: type:model_name)")
    parser.add_argument()))))"--fp16", action="store_true",
    help="Use FP16 precision for CUDA tests")
    
    args = parser.parse_args())))))
    
    # Parse model types
    model_types = [m.strip()))))) for m in args.model_types.split()))))",")]:,
    # Parse batch sizes
    batch_sizes = [int()))))b.strip())))))) for b in args.batch_sizes.split()))))",")]:,
    # Parse platforms
    platforms = [p.strip()))))).lower()))))) for p in args.platforms.split()))))",")]:,
    # Parse specific models
    specific_models = {}}}}}}}}}}}}}}}}}}}}}}
    for spec in args.specific_model:
        if ":" in spec:
            model_type, model_name = spec.split()))))":", 1)
            specific_models[model_type.strip())))))] = model_name.strip())))))
            ,
    # Create test directories
            os.makedirs()))))TEST_RESULTS_DIR, exist_ok=True)
    
    # Create and run test framework
            test_framework = BatchInferenceTest()))))
            model_types=model_types,
            batch_sizes=batch_sizes,
            specific_models=specific_models,
            platforms=platforms,
            use_fp16=args.fp16
            )
    
            test_framework.run_tests())))))

if __name__ == "__main__":
    main())))))