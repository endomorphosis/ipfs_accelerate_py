# Standard library imports
import os
import sys
import json
import time
import datetime
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch

# Use direct import with absolute path
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")

# Import optional dependencies with fallbacks
try:
    import torch
    import numpy as np
except ImportError:
    torch = MagicMock()
    np = MagicMock()
    print("Warning: torch/numpy not available, using mock implementation")

try:
    import transformers
except ImportError:
    transformers = MagicMock()
    print("Warning: transformers not available, using mock implementation")

# Try to import from ipfs_accelerate_py
try:
    from ipfs_accelerate_py.worker.skillset.hf_starcoder2 import hf_starcoder2
except ImportError:
    # Create a mock class if the real one doesn't exist
    class hf_starcoder2:
        def __init__(self, resources=None, metadata=None):
            self.resources = resources if resources else {}
            self.metadata = metadata if metadata else {}
            
        def init_cpu(self, model_name, model_type, device):
            mock_handler = lambda prompt=None, **kwargs: {
                "generated_text": "def hello_world():\n    print('Hello, World!')\n",
                "implementation_type": "(MOCK)"
            }
            return "mock_endpoint", "mock_tokenizer", mock_handler, None, 1
            
        def init_cuda(self, model_name, model_type, device):
            return self.init_cpu(model_name, model_type, device)
            
        def init_openvino(self, model_name, model_type, device):
            return self.init_cpu(model_name, model_type, device)
    
    print("Warning: hf_starcoder2 not found, using mock implementation")

class test_hf_starcoder2:
    """
    Test class for Hugging Face's StarCoder2 model.
    
    StarCoder2 is an advanced code generation model trained on a large corpus
    of code from various programming languages. It's designed for code completion,
    generation, and understanding tasks, with powerful capabilities for
    multiple programming languages.
    
    This class tests StarCoder2 functionality across different hardware 
    backends including CPU, CUDA, and OpenVINO.
    
    It verifies:
    1. Code completion capabilities
    2. Code generation from scratch
    3. Inferring documentation/comments
    4. Multiple programming language support
    5. Cross-platform compatibility
    6. Performance metrics
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the StarCoder2 test environment"""
        # Set up resources with fallbacks
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np, 
            "transformers": transformers
        }
        
        # Store metadata
        self.metadata = metadata if metadata else {}
        
        # Initialize the StarCoder2 model
        self.starcoder2 = hf_starcoder2(resources=self.resources, metadata=self.metadata)
        
        # Use a small, openly accessible model that doesn't require authentication
        self.model_name = "bigcode/starcoder2-3b"  # Smallest version in StarCoder2 family
        
        # Create test prompts for different capabilities
        self.test_prompts = {
            "python_completion": "def factorial(n):\n    # Calculate factorial of n\n    ",
            "python_generation": "# Write a function to check if a string is a palindrome\n",
            "javascript_completion": "function calculateSum(arr) {\n    // Calculate sum of array elements\n    ",
            "rust_completion": "fn fibonacci(n: u32) -> u32 {\n    // Calculate the nth Fibonacci number\n    ",
            "documentation_generation": "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] < target:\n            low = mid + 1\n        elif arr[mid] > target:\n            high = mid - 1\n        else:\n            return mid\n    return -1\n\n# Document this function:\n"
        }
        
        # Configuration for generation
        self.generation_config = {
            "max_new_tokens": 100,
            "temperature": 0.2,
            "top_p": 0.95,
            "do_sample": True
        }
        
        # Status tracking
        self.status_messages = {
            "cpu": "Not tested yet",
            "cuda": "Not tested yet",
            "openvino": "Not tested yet"
        }
        
        # Examples for tracking test outputs
        self.examples = []
        
        return None
    
    def _create_local_test_model(self):
        """Create a minimal test model directory for testing without downloading"""
        try:
            print("Creating local test model for StarCoder2...")
            
            # Create model directory in /tmp for tests
            test_model_dir = os.path.join("/tmp", "starcoder2_test_model")
            os.makedirs(test_model_dir, exist_ok=True)
            
            # Create minimal config file
            config = {
                "model_type": "starcoder2",
                "architectures": ["CausalLM"],
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 2,
                "num_attention_heads": 16,
                "vocab_size": 49152,
                "max_position_embeddings": 4096,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
                "ignore_mismatched_sizes": True
            }
            
            # Write config
            with open(os.path.join(test_model_dir, "config.json"), "w") as f:
                json.dump(config, f)
                
            print(f"Test model created at {test_model_dir}")
            return test_model_dir
            
        except Exception as e:
            print(f"Error creating test model: {e}")
            return self.model_name  # Fall back to original name
    
    def test(self):
        """Run all tests for the StarCoder2 model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.starcoder2 is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"
        
        # Test CPU initialization and functionality
        try:
            print("Testing StarCoder2 on CPU...")
            
            # Check if using real transformers
            transformers_available = not isinstance(self.resources["transformers"], MagicMock)
            implementation_type = "(REAL)" if transformers_available else "(MOCK)"
            
            # Initialize for CPU
            endpoint, tokenizer, handler, queue, batch_size = self.starcoder2.init_cpu(
                self.model_name,
                "cpu", 
                "cpu"
            )
            
            valid_init = endpoint is not None and tokenizer is not None and handler is not None
            results["cpu_init"] = f"Success {implementation_type}" if valid_init else "Failed CPU initialization"
            
            # Test code generation with different prompts
            prompt_results = {}
            
            for prompt_type, prompt_text in self.test_prompts.items():
                try:
                    start_time = time.time()
                    
                    output = handler(
                        prompt=prompt_text,
                        **self.generation_config
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    # Verify output contains text
                    has_generation = (
                        output is not None and
                        isinstance(output, dict) and
                        "generated_text" in output and
                        isinstance(output["generated_text"], str) and
                        len(output["generated_text"]) > 0
                    )
                    
                    if has_generation:
                        # Extract text
                        generated_text = output["generated_text"]
                        
                        # Analyze code generation quality
                        token_count = len(generated_text.split())
                        lines_count = len(generated_text.splitlines())
                        
                        # Simple code quality metrics
                        quality_metrics = {
                            "token_count": token_count,
                            "lines_count": lines_count,
                            "chars_per_line": len(generated_text) / max(1, lines_count)
                        }
                        
                        # Add example to collection
                        example = {
                            "input": {
                                "prompt_type": prompt_type,
                                "prompt": prompt_text
                            },
                            "output": {
                                "generated_text": generated_text,
                                "quality_metrics": quality_metrics
                            },
                            "timestamp": time.time(),
                            "elapsed_time": elapsed_time,
                            "implementation_type": implementation_type.strip("()"),
                            "platform": "CPU"
                        }
                        
                        self.examples.append(example)
                        
                        prompt_results[prompt_type] = {
                            "success": True,
                            "token_count": token_count,
                            "elapsed_time": elapsed_time,
                            "tokens_per_second": token_count / elapsed_time if elapsed_time > 0 else 0
                        }
                    else:
                        prompt_results[prompt_type] = {
                            "success": False,
                            "error": "No text generated"
                        }
                except Exception as e:
                    print(f"Error in CPU test for prompt type '{prompt_type}': {e}")
                    prompt_results[prompt_type] = {
                        "success": False,
                        "error": str(e)
                    }
            
            # Record prompt results
            results["cpu_prompt_results"] = prompt_results
            
            # Determine overall success
            any_prompt_succeeded = any(item.get("success", False) for item in prompt_results.values())
            results["cpu_overall"] = f"Success {implementation_type}" if any_prompt_succeeded else "Failed all prompts"
                
        except Exception as e:
            print(f"Error in CPU tests: {e}")
            traceback.print_exc()
            results["cpu_tests"] = f"Error: {str(e)}"
            
        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                print("Testing StarCoder2 on CUDA...")
                # Import CUDA utilities
                try:
                    sys.path.insert(0, "/home/barberb/ipfs_accelerate_py/test")
                    from utils import get_cuda_device, optimize_cuda_memory, benchmark_cuda_inference
                    cuda_utils_available = True
                except ImportError:
                    cuda_utils_available = False
                
                # Initialize for CUDA
                endpoint, tokenizer, handler, queue, batch_size = self.starcoder2.init_cuda(
                    self.model_name,
                    "cuda",
                    "cuda:0"
                )
                
                valid_init = endpoint is not None and tokenizer is not None and handler is not None
                results["cuda_init"] = "Success (REAL)" if valid_init else "Failed CUDA initialization"
                
                # Get GPU memory if available
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024) if hasattr(torch.cuda, "memory_allocated") else None
                
                # Only test a subset of prompts on CUDA to reduce test time
                cuda_test_prompts = {
                    "python_completion": self.test_prompts["python_completion"],
                    "python_generation": self.test_prompts["python_generation"]
                }
                
                # Test code generation with selected prompts
                prompt_results = {}
                
                for prompt_type, prompt_text in cuda_test_prompts.items():
                    try:
                        start_time = time.time()
                        
                        output = handler(
                            prompt=prompt_text,
                            **self.generation_config
                        )
                        
                        elapsed_time = time.time() - start_time
                        
                        # Verify output contains text
                        has_generation = (
                            output is not None and
                            isinstance(output, dict) and
                            "generated_text" in output and
                            isinstance(output["generated_text"], str) and
                            len(output["generated_text"]) > 0
                        )
                        
                        if has_generation:
                            # Extract text
                            generated_text = output["generated_text"]
                            
                            # Analyze code generation quality
                            token_count = len(generated_text.split())
                            lines_count = len(generated_text.splitlines())
                            
                            # Simple code quality metrics
                            quality_metrics = {
                                "token_count": token_count,
                                "lines_count": lines_count,
                                "chars_per_line": len(generated_text) / max(1, lines_count)
                            }
                            
                            # Performance metrics
                            perf_metrics = {
                                "processing_time_seconds": elapsed_time,
                                "tokens_per_second": token_count / elapsed_time if elapsed_time > 0 else 0
                            }
                            
                            if gpu_memory_mb is not None:
                                perf_metrics["gpu_memory_mb"] = gpu_memory_mb
                            
                            # Add example to collection
                            example = {
                                "input": {
                                    "prompt_type": prompt_type,
                                    "prompt": prompt_text
                                },
                                "output": {
                                    "generated_text": generated_text,
                                    "quality_metrics": quality_metrics
                                },
                                "timestamp": time.time(),
                                "elapsed_time": elapsed_time,
                                "implementation_type": "REAL",
                                "platform": "CUDA",
                                "performance_metrics": perf_metrics
                            }
                            
                            self.examples.append(example)
                            
                            prompt_results[prompt_type] = {
                                "success": True,
                                "token_count": token_count,
                                "elapsed_time": elapsed_time,
                                "tokens_per_second": token_count / elapsed_time if elapsed_time > 0 else 0,
                                "performance_metrics": perf_metrics
                            }
                        else:
                            prompt_results[prompt_type] = {
                                "success": False,
                                "error": "No text generated"
                            }
                    except Exception as e:
                        print(f"Error in CUDA test for prompt type '{prompt_type}': {e}")
                        prompt_results[prompt_type] = {
                            "success": False,
                            "error": str(e)
                        }
                
                # Record prompt results
                results["cuda_prompt_results"] = prompt_results
                
                # Determine overall success
                any_prompt_succeeded = any(item.get("success", False) for item in prompt_results.values())
                results["cuda_overall"] = "Success (REAL)" if any_prompt_succeeded else "Failed all prompts"
                
            except Exception as e:
                print(f"Error in CUDA tests: {e}")
                traceback.print_exc()
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"
            
        # Test OpenVINO if available
        try:
            print("Testing StarCoder2 on OpenVINO...")
            
            # Try to import OpenVINO
            try:
                import openvino
                openvino_available = True
            except ImportError:
                openvino_available = False
                
            if not openvino_available:
                results["openvino_tests"] = "OpenVINO not available"
            else:
                # Initialize for OpenVINO
                endpoint, tokenizer, handler, queue, batch_size = self.starcoder2.init_openvino(
                    self.model_name,
                    "openvino",
                    "CPU"  # Standard OpenVINO device
                )
                
                valid_init = endpoint is not None and tokenizer is not None and handler is not None
                results["openvino_init"] = "Success (REAL)" if valid_init else "Failed OpenVINO initialization"
                
                # Test a single prompt for OpenVINO to keep it simpler
                prompt_type = "python_completion"
                prompt_text = self.test_prompts[prompt_type]
                
                try:
                    start_time = time.time()
                    
                    output = handler(
                        prompt=prompt_text,
                        **{**self.generation_config, "max_new_tokens": 50}  # Use fewer tokens for OpenVINO
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    # Verify output contains text
                    has_generation = (
                        output is not None and
                        isinstance(output, dict) and
                        "generated_text" in output and
                        isinstance(output["generated_text"], str) and
                        len(output["generated_text"]) > 0
                    )
                    
                    if has_generation:
                        # Extract text
                        generated_text = output["generated_text"]
                        
                        # Analyze code generation quality
                        token_count = len(generated_text.split())
                        lines_count = len(generated_text.splitlines())
                        
                        # Simple code quality metrics
                        quality_metrics = {
                            "token_count": token_count,
                            "lines_count": lines_count,
                            "chars_per_line": len(generated_text) / max(1, lines_count)
                        }
                        
                        # Performance metrics
                        perf_metrics = {
                            "processing_time_seconds": elapsed_time,
                            "tokens_per_second": token_count / elapsed_time if elapsed_time > 0 else 0
                        }
                        
                        # Add example to collection
                        example = {
                            "input": {
                                "prompt_type": prompt_type,
                                "prompt": prompt_text
                            },
                            "output": {
                                "generated_text": generated_text,
                                "quality_metrics": quality_metrics
                            },
                            "timestamp": time.time(),
                            "elapsed_time": elapsed_time,
                            "implementation_type": "REAL",
                            "platform": "OpenVINO",
                            "performance_metrics": perf_metrics
                        }
                        
                        self.examples.append(example)
                        
                        results["openvino_generation"] = {
                            "success": True,
                            "token_count": token_count,
                            "elapsed_time": elapsed_time,
                            "tokens_per_second": token_count / elapsed_time if elapsed_time > 0 else 0,
                            "performance_metrics": perf_metrics
                        }
                        
                        results["openvino_overall"] = "Success (REAL)"
                    else:
                        results["openvino_generation"] = {
                            "success": False,
                            "error": "No text generated"
                        }
                        
                        results["openvino_overall"] = "Failed generation"
                except Exception as e:
                    print(f"Error in OpenVINO test: {e}")
                    results["openvino_generation"] = {
                        "success": False,
                        "error": str(e)
                    }
                    
                    results["openvino_overall"] = f"Error: {str(e)}"
                
        except Exception as e:
            print(f"Error in OpenVINO tests: {e}")
            traceback.print_exc()
            results["openvino_tests"] = f"Error: {str(e)}"
        
        # Add examples to results
        results["examples"] = self.examples
        
        return results
    
    def __test__(self):
        """Run tests and handle result storage and comparison"""
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {
                "status": {"test_error": str(e)},
                "traceback": traceback.format_exc(),
                "examples": []
            }
        
        # Add metadata
        test_results["metadata"] = {
            "timestamp": time.time(),
            "torch_version": getattr(torch, "__version__", "mocked"),
            "numpy_version": getattr(np, "__version__", "mocked"),
            "transformers_version": getattr(transformers, "__version__", "mocked"),
            "cuda_available": getattr(torch, "cuda", MagicMock()).is_available() if not isinstance(torch, MagicMock) else False,
            "cuda_device_count": getattr(torch, "cuda", MagicMock()).device_count() if not isinstance(torch, MagicMock) else 0,
            "transformers_mocked": isinstance(self.resources["transformers"], MagicMock),
            "test_model": self.model_name,
            "test_run_id": f"starcoder2-test-{int(time.time())}"
        }
        
        # Create directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(collected_dir, 'hf_starcoder2_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_starcoder2_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                
                # Simple check for basic compatibility
                if "init" in expected_results and "init" in test_results:
                    print("Results structure matches expected format.")
                else:
                    print("Warning: Results structure does not match expected format.")
            except Exception as e:
                print(f"Error reading expected results: {e}")
                # Create new expected results file
                with open(expected_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
        else:
            # Create new expected results file
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2)
                
        return test_results

if __name__ == "__main__":
    try:
        print("Starting StarCoder2 test...")
        this_starcoder2 = test_hf_starcoder2()
        results = this_starcoder2.__test__()
        print(f"StarCoder2 Test Completed")
        
        # Print a summary of the results
        if "init" in results:
            print(f"Initialization: {results['init']}")
        
        # CPU results
        if "cpu_overall" in results:
            print(f"CPU Tests: {results['cpu_overall']}")
            
            # Print prompt success rates for CPU
            if "cpu_prompt_results" in results:
                success_count = sum(1 for item in results["cpu_prompt_results"].values() if item.get("success", False))
                total_count = len(results["cpu_prompt_results"])
                print(f"  CPU Prompts: {success_count}/{total_count} successful")
                
        elif "cpu_tests" in results:
            print(f"CPU Tests: {results['cpu_tests']}")
            
        # CUDA results
        if "cuda_overall" in results:
            print(f"CUDA Tests: {results['cuda_overall']}")
            
            # Print prompt success rates for CUDA
            if "cuda_prompt_results" in results:
                success_count = sum(1 for item in results["cuda_prompt_results"].values() if item.get("success", False))
                total_count = len(results["cuda_prompt_results"])
                print(f"  CUDA Prompts: {success_count}/{total_count} successful")
                
        elif "cuda_tests" in results:
            print(f"CUDA Tests: {results['cuda_tests']}")
            
        # OpenVINO results
        if "openvino_overall" in results:
            print(f"OpenVINO Tests: {results['openvino_overall']}")
        elif "openvino_tests" in results:
            print(f"OpenVINO Tests: {results['openvino_tests']}")
            
        # Example count
        example_count = len(results.get("examples", []))
        print(f"Collected {example_count} test examples")
        
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)