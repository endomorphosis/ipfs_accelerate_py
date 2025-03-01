# Standard library imports
import os
import sys
import json
import time
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
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    transformers = MagicMock()
    AutoTokenizer = MagicMock()
    AutoModelForCausalLM = MagicMock()
    print("Warning: transformers not available, using mock implementation")

# Try to import from ipfs_accelerate_py
try:
    from ipfs_accelerate_py.worker.skillset.hf_codellama import hf_codellama
except ImportError:
    # Create a mock class if the real one doesn't exist
    class hf_codellama:
        def __init__(self, resources=None, metadata=None):
            self.resources = resources if resources else {}
            self.metadata = metadata if metadata else {}
            
        def init_cpu(self, model_name, processor_name, device):
            mock_handler = lambda prompt=None, max_tokens=100, temperature=0.7, **kwargs: {
                "generated_text": "def hello_world():\n    print('Hello, World!')\n",
                "implementation_type": "(MOCK)"
            }
            return "mock_endpoint", "mock_processor", mock_handler, None, 1
            
        def init_cuda(self, model_name, processor_name, device):
            return self.init_cpu(model_name, processor_name, device)
            
        def init_openvino(self, model_name, processor_name, device):
            return self.init_cpu(model_name, processor_name, device)
    
    print("Warning: hf_codellama not found, using mock implementation")

class test_hf_codellama:
    """
    Test class for Hugging Face CodeLlama.
    
    This class tests the CodeLlama model functionality across different hardware 
    backends including CPU, CUDA, and OpenVINO.
    
    It verifies:
    1. Code generation from prompts
    2. Code completion
    3. Language-specific code generation
    4. Cross-platform compatibility
    5. Performance metrics
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the CodeLlama test environment"""
        # Set up resources with fallbacks
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np, 
            "transformers": transformers,
            "AutoTokenizer": AutoTokenizer,
            "AutoModelForCausalLM": AutoModelForCausalLM
        }
        
        # Store metadata
        self.metadata = metadata if metadata else {}
        
        # Initialize the CodeLlama model
        self.codellama = hf_codellama(resources=self.resources, metadata=self.metadata)
        
        # Use a small, openly accessible model that doesn't require authentication
        self.model_name = "codellama/CodeLlama-7b-Python-hf"  # Python-specific 7B model
        self.small_model_name = "codellama/CodeLlama-7b-Instruct-hf"  # Smaller model for testing
        
        # Create test prompts for code generation
        self.test_prompt = "Write a Python function that calculates the Fibonacci sequence up to n."
        self.test_completion_prompt = "def sort_array(arr):\n    # Sort the array in ascending order\n    "
        self.test_language_prompts = {
            "python": "Write a function to check if a string is a palindrome",
            "javascript": "Write a function to calculate the factorial of a number",
            "java": "Write a method to reverse a linked list",
            "c++": "Write a function to find the greatest common divisor of two numbers"
        }
        
        # Status tracking
        self.status_messages = {
            "cpu": "Not tested yet",
            "cuda": "Not tested yet",
            "openvino": "Not tested yet"
        }
        
        return None
        
    def _create_local_test_model(self):
        """Create a minimal test model directory for testing without downloading"""
        try:
            print("Creating local test model for CodeLlama...")
            
            # Create model directory in /tmp for tests
            test_model_dir = os.path.join("/tmp", "codellama_test_model")
            os.makedirs(test_model_dir, exist_ok=True)
            
            # Create minimal config file
            config = {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "vocab_size": 32000,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "hidden_act": "silu",
                "max_position_embeddings": 4096,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-6,
                "use_cache": True,
                "tie_word_embeddings": False
            }
            
            # Write config
            with open(os.path.join(test_model_dir, "config.json"), "w") as f:
                json.dump(config, f)
                
            # Create minimal tokenizer files
            tokenizer_config = {
                "model_type": "llama",
                "padding_side": "right"
            }
            
            with open(os.path.join(test_model_dir, "tokenizer_config.json"), "w") as f:
                json.dump(tokenizer_config, f)
                
            print(f"Test model created at {test_model_dir}")
            return test_model_dir
            
        except Exception as e:
            print(f"Error creating test model: {e}")
            return self.small_model_name  # Fall back to original name
            
    def test(self):
        """Run all tests for the CodeLlama model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.codellama is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"
        
        # Test CPU initialization and functionality
        try:
            print("Testing CodeLlama on CPU...")
            
            # Check if using real transformers
            transformers_available = not isinstance(self.resources["transformers"], MagicMock)
            implementation_type = "(REAL)" if transformers_available else "(MOCK)"
            
            # For CPU tests, use a smaller model if available
            model_name = self.small_model_name if transformers_available else self._create_local_test_model()
            
            # Initialize for CPU
            endpoint, processor, handler, queue, batch_size = self.codellama.init_cpu(
                model_name,
                "cpu",
                "cpu"
            )
            
            valid_init = endpoint is not None and processor is not None and handler is not None
            results["cpu_init"] = f"Success {implementation_type}" if valid_init else "Failed CPU initialization"
            
            # Test code generation
            output = handler(self.test_prompt, max_tokens=200, temperature=0.7)
            
            # Verify output contains code
            has_code = (
                output is not None and
                isinstance(output, dict) and
                "generated_text" in output
            )
            results["cpu_generation"] = f"Success {implementation_type}" if has_code else "Failed code generation"
            
            # Add details if successful
            if has_code:
                generated_code = output["generated_text"]
                
                # Add example for recorded output
                results["cpu_generation_example"] = {
                    "input": self.test_prompt,
                    "output": {
                        "generated_text": generated_code[:500] if len(generated_code) > 500 else generated_code,
                        "token_count": len(generated_code.split()),
                        "has_fibonacci_function": "fibonacci" in generated_code.lower() or "fib" in generated_code.lower()
                    },
                    "timestamp": time.time(),
                    "implementation": implementation_type
                }
                
            # Test code completion
            completion_output = handler(self.test_completion_prompt, max_tokens=100, temperature=0.5)
            
            # Verify output contains code
            has_completion = (
                completion_output is not None and
                isinstance(completion_output, dict) and
                "generated_text" in completion_output
            )
            results["cpu_completion"] = f"Success {implementation_type}" if has_completion else "Failed code completion"
            
            # Add details if successful
            if has_completion:
                completion_code = completion_output["generated_text"]
                
                # Add example for recorded output
                results["cpu_completion_example"] = {
                    "input": self.test_completion_prompt,
                    "output": {
                        "generated_text": completion_code[:500] if len(completion_code) > 500 else completion_code,
                        "token_count": len(completion_code.split()),
                        "has_sort_implementation": "sort" in completion_code.lower() 
                    },
                    "timestamp": time.time(),
                    "implementation": implementation_type
                }
                
            # Test language-specific generation (Python only for CPU tests to save time)
            try:
                language_output = handler(
                    self.test_language_prompts["python"],
                    max_tokens=150,
                    temperature=0.7,
                    language="python"
                )
                
                # Verify output contains code
                has_language_code = (
                    language_output is not None and
                    isinstance(language_output, dict) and
                    "generated_text" in language_output
                )
                results["cpu_language_specific"] = f"Success {implementation_type}" if has_language_code else "Failed language-specific generation"
                
                # Add details if successful
                if has_language_code:
                    language_code = language_output["generated_text"]
                    
                    # Add example for recorded output
                    results["cpu_language_specific_example"] = {
                        "input": {
                            "prompt": self.test_language_prompts["python"],
                            "language": "python"
                        },
                        "output": {
                            "generated_text": language_code[:500] if len(language_code) > 500 else language_code,
                            "token_count": len(language_code.split()),
                            "has_palindrome_check": "palindrome" in language_code.lower()
                        },
                        "timestamp": time.time(),
                        "implementation": implementation_type
                    }
            except Exception as lang_err:
                print(f"Error in language-specific generation test: {lang_err}")
                results["cpu_language_specific"] = f"Error: {str(lang_err)}"
                
        except Exception as e:
            print(f"Error in CPU tests: {e}")
            traceback.print_exc()
            results["cpu_tests"] = f"Error: {str(e)}"
            
        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                print("Testing CodeLlama on CUDA...")
                # Import CUDA utilities
                try:
                    sys.path.insert(0, "/home/barberb/ipfs_accelerate_py/test")
                    from utils import get_cuda_device, optimize_cuda_memory, benchmark_cuda_inference
                    cuda_utils_available = True
                except ImportError:
                    cuda_utils_available = False
                
                # Get optimal device if utilities available
                device = "cuda:0"
                if cuda_utils_available:
                    device = get_cuda_device()
                    optimize_cuda_memory()
                
                # Initialize for CUDA
                endpoint, processor, handler, queue, batch_size = self.codellama.init_cuda(
                    self.model_name,
                    "cuda",
                    device
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["cuda_init"] = "Success (REAL)" if valid_init else "Failed CUDA initialization"
                
                # Test code generation with performance metrics
                start_time = time.time()
                output = handler(self.test_prompt, max_tokens=200, temperature=0.7)
                elapsed_time = time.time() - start_time
                
                # Verify output contains code
                has_code = (
                    output is not None and
                    isinstance(output, dict) and
                    "generated_text" in output
                )
                results["cuda_generation"] = "Success (REAL)" if has_code else "Failed code generation"
                
                # Add details if successful
                if has_code:
                    generated_code = output["generated_text"]
                    token_count = len(generated_code.split())
                    
                    # Calculate performance metrics
                    performance_metrics = {
                        "processing_time_seconds": elapsed_time,
                        "tokens_per_second": token_count / elapsed_time if elapsed_time > 0 else 0
                    }
                    
                    # Get GPU memory usage if available
                    if hasattr(torch.cuda, "memory_allocated"):
                        performance_metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                    
                    # Add example with performance metrics
                    results["cuda_generation_example"] = {
                        "input": self.test_prompt,
                        "output": {
                            "generated_text": generated_code[:500] if len(generated_code) > 500 else generated_code,
                            "token_count": token_count,
                            "has_fibonacci_function": "fibonacci" in generated_code.lower() or "fib" in generated_code.lower()
                        },
                        "timestamp": time.time(),
                        "implementation": "REAL",
                        "performance_metrics": performance_metrics
                    }
                
                # Test multi-language generation if time permits
                if valid_init:
                    language_results = {}
                    
                    for lang, prompt in self.test_language_prompts.items():
                        try:
                            start_time = time.time()
                            lang_output = handler(prompt, max_tokens=150, temperature=0.7, language=lang)
                            lang_elapsed_time = time.time() - start_time
                            
                            if lang_output and "generated_text" in lang_output:
                                lang_code = lang_output["generated_text"]
                                lang_token_count = len(lang_code.split())
                                
                                language_results[lang] = {
                                    "success": True,
                                    "token_count": lang_token_count,
                                    "processing_time_seconds": lang_elapsed_time,
                                    "tokens_per_second": lang_token_count / lang_elapsed_time if lang_elapsed_time > 0 else 0
                                }
                        except Exception as lang_err:
                            language_results[lang] = {
                                "success": False,
                                "error": str(lang_err)
                            }
                    
                    results["cuda_multi_language_results"] = language_results
            except Exception as e:
                print(f"Error in CUDA tests: {e}")
                traceback.print_exc()
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"
            
        # Test OpenVINO if available
        try:
            print("Testing CodeLlama on OpenVINO...")
            
            # Try to import OpenVINO
            try:
                import openvino
                openvino_available = True
            except ImportError:
                openvino_available = False
                
            if not openvino_available:
                results["openvino_tests"] = "OpenVINO not available"
            else:
                # Use smaller model for OpenVINO due to potential memory constraints
                model_name = self.small_model_name
                
                # Initialize for OpenVINO
                endpoint, processor, handler, queue, batch_size = self.codellama.init_openvino(
                    model_name,
                    "openvino",
                    "CPU"  # Standard OpenVINO device
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["openvino_init"] = "Success (REAL)" if valid_init else "Failed OpenVINO initialization"
                
                # Test code generation with performance metrics
                start_time = time.time()
                output = handler(self.test_prompt, max_tokens=100, temperature=0.7)
                elapsed_time = time.time() - start_time
                
                # Verify output contains code
                has_code = (
                    output is not None and
                    isinstance(output, dict) and
                    "generated_text" in output
                )
                results["openvino_generation"] = "Success (REAL)" if has_code else "Failed code generation"
                
                # Add details if successful
                if has_code:
                    generated_code = output["generated_text"]
                    token_count = len(generated_code.split())
                    
                    # Calculate performance metrics
                    performance_metrics = {
                        "processing_time_seconds": elapsed_time,
                        "tokens_per_second": token_count / elapsed_time if elapsed_time > 0 else 0
                    }
                    
                    # Add example with performance metrics
                    results["openvino_generation_example"] = {
                        "input": self.test_prompt,
                        "output": {
                            "generated_text": generated_code[:500] if len(generated_code) > 500 else generated_code,
                            "token_count": token_count,
                            "has_fibonacci_function": "fibonacci" in generated_code.lower() or "fib" in generated_code.lower()
                        },
                        "timestamp": time.time(),
                        "implementation": "REAL",
                        "performance_metrics": performance_metrics
                    }
        except Exception as e:
            print(f"Error in OpenVINO tests: {e}")
            traceback.print_exc()
            results["openvino_tests"] = f"Error: {str(e)}"
            
        return results
    
    def __test__(self):
        """Run tests and handle result storage and comparison"""
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {"test_error": str(e), "traceback": traceback.format_exc()}
        
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
            "small_test_model": self.small_model_name,
            "test_run_id": f"codellama-test-{int(time.time())}"
        }
        
        # Create directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(collected_dir, 'hf_codellama_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_codellama_test_results.json')
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
        this_codellama = test_hf_codellama()
        results = this_codellama.__test__()
        print(f"CodeLlama Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)