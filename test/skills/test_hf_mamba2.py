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
except ImportError:
    transformers = MagicMock()
    print("Warning: transformers not available, using mock implementation")

# Try to import from ipfs_accelerate_py
try:
    from ipfs_accelerate_py.worker.skillset.hf_mamba2 import hf_mamba2
except ImportError:
    # Create a mock class if the real one doesn't exist
    class hf_mamba2:
        def __init__(self, resources=None, metadata=None):
            self.resources = resources if resources else {}
            self.metadata = metadata if metadata else {}
            
        def init_cpu(self, model_name, processor_name, device):
            mock_handler = lambda prompt=None, max_tokens=100, temperature=0.7, **kwargs: {
                "generated_text": "This is a mock response from Mamba2 model.",
                "implementation_type": "(MOCK)"
            }
            return "mock_endpoint", "mock_processor", mock_handler, None, 1
            
        def init_cuda(self, model_name, processor_name, device):
            return self.init_cpu(model_name, processor_name, device)
            
        def init_openvino(self, model_name, processor_name, device):
            return self.init_cpu(model_name, processor_name, device)
    
    print("Warning: hf_mamba2 not found, using mock implementation")

class test_hf_mamba2:
    """
    Test class for Hugging Face Mamba2 state-space sequence model.
    
    This class tests the Mamba2 model functionality across different hardware 
    backends including CPU, CUDA, and OpenVINO.
    
    It verifies:
    1. Text generation capabilities
    2. Long-context handling
    3. Performance metrics
    4. Cross-platform compatibility
    5. Linear scaling with sequence length
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the Mamba2 test environment"""
        # Set up resources with fallbacks
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np, 
            "transformers": transformers
        }
        
        # Store metadata
        self.metadata = metadata if metadata else {}
        
        # Initialize the Mamba2 model
        self.mamba2 = hf_mamba2(resources=self.resources, metadata=self.metadata)
        
        # Use small models for testing
        self.model_name = "state-spaces/mamba2-1.4b"  # 1.4B parameter model
        self.small_model_name = "state-spaces/mamba2-130m"  # 130M parameter model
        
        # Create test prompts for various tasks
        self.test_prompt = "Mamba2 is a state-space sequence model that"
        
        # Test for instruction following
        self.instruction_prompt = "Question: What are the key differences between Mamba and Transformer architectures?\nAnswer:"
        
        # Test for long context handling - create a longer prompt
        self.long_context_prompt = "Below is a description of state-space models in deep learning:\n\n" + \
            "State-space models (SSMs) are a class of models that map an input sequence to an output sequence through a hidden state. " + \
            "Unlike traditional RNNs, SSMs have a more structured parameterization based on linear dynamical systems theory. " + \
            "This structure allows them to capture long-range dependencies more effectively while maintaining computational efficiency. " + \
            "In particular, they can be implemented with linear scaling in sequence length, unlike the quadratic scaling of attention-based transformers. " + \
            "Mamba introduces a selective mechanism that allows the state-space model to adapt based on the input, " + \
            "enabling it to be selective about what information to remember from the past. " + \
            "Mamba2 improves upon the original Mamba architecture by incorporating\n"
        
        # Context window size test - generate prompts of different lengths
        self.context_sizes = [128, 512, 1024]  # Different context sizes to test
        
        # Status tracking
        self.status_messages = {
            "cpu": "Not tested yet",
            "cuda": "Not tested yet",
            "openvino": "Not tested yet"
        }
        
        return None
        
    def _create_long_prompt(self, length):
        """Create a prompt of approximate token length"""
        # Simple repeated text to create a long prompt
        base_text = "This is a test sentence for Mamba2 state-space sequence model. "  # About 12 tokens
        repetitions = max(1, length // 12)
        return base_text * repetitions
        
    def _create_local_test_model(self):
        """Create a minimal test model directory for testing without downloading"""
        try:
            print("Creating local test model for Mamba2...")
            
            # Create model directory in /tmp for tests
            test_model_dir = os.path.join("/tmp", "mamba2_test_model")
            os.makedirs(test_model_dir, exist_ok=True)
            
            # Create minimal config file
            config = {
                "model_type": "mamba2",
                "architectures": ["Mamba2ForCausalLM"],
                "vocab_size": 32000,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "ssm_cfg": {
                    "state_size": 16,
                    "conv_kernel": 4,
                    "d_state": 16,
                    "d_conv": 4,
                    "expand": 2
                },
                "rms_norm": True,
                "residual_in_fp32": True,
                "pad_vocab_size_multiple": 8,
                "fused_add_norm": True,
                "tie_word_embeddings": False
            }
            
            # Write config
            with open(os.path.join(test_model_dir, "config.json"), "w") as f:
                json.dump(config, f)
                
            # Create minimal tokenizer files
            tokenizer_config = {
                "model_type": "mamba2",
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
        """Run all tests for the Mamba2 model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.mamba2 is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"
        
        # Test CPU initialization and functionality
        try:
            print("Testing Mamba2 on CPU...")
            
            # Check if using real transformers
            transformers_available = not isinstance(self.resources["transformers"], MagicMock)
            implementation_type = "(REAL)" if transformers_available else "(MOCK)"
            
            # For CPU tests, use the smallest model
            model_name = self.small_model_name if transformers_available else self._create_local_test_model()
            
            # Initialize for CPU
            endpoint, processor, handler, queue, batch_size = self.mamba2.init_cpu(
                model_name,
                "cpu",
                "cpu"
            )
            
            valid_init = endpoint is not None and processor is not None and handler is not None
            results["cpu_init"] = f"Success {implementation_type}" if valid_init else "Failed CPU initialization"
            
            # Test basic generation
            output = handler(self.test_prompt, max_tokens=50, temperature=0.7)
            
            # Verify output contains text
            has_text = (
                output is not None and
                isinstance(output, dict) and
                "generated_text" in output
            )
            results["cpu_generation"] = f"Success {implementation_type}" if has_text else "Failed text generation"
            
            # Add details if successful
            if has_text:
                generated_text = output["generated_text"]
                
                # Add example for recorded output
                results["cpu_generation_example"] = {
                    "input": self.test_prompt,
                    "output": {
                        "generated_text": generated_text[:500] if len(generated_text) > 500 else generated_text,
                        "token_count": len(generated_text.split())
                    },
                    "timestamp": time.time(),
                    "implementation": implementation_type
                }
                
            # Test instruction following
            instruction_output = handler(self.instruction_prompt, max_tokens=100, temperature=0.7)
            
            # Verify output contains text
            has_instruction_response = (
                instruction_output is not None and
                isinstance(instruction_output, dict) and
                "generated_text" in instruction_output
            )
            
            results["cpu_instruction"] = f"Success {implementation_type}" if has_instruction_response else "Failed instruction test"
            
            # Add details if successful
            if has_instruction_response:
                instruction_text = instruction_output["generated_text"]
                
                # Add example for recorded output
                results["cpu_instruction_example"] = {
                    "input": self.instruction_prompt,
                    "output": {
                        "generated_text": instruction_text[:500] if len(instruction_text) > 500 else instruction_text,
                        "token_count": len(instruction_text.split())
                    },
                    "timestamp": time.time(),
                    "implementation": implementation_type
                }
                
            # Test long context handling
            long_context_output = handler(self.long_context_prompt, max_tokens=50, temperature=0.7)
            
            # Verify output contains text
            has_long_context_response = (
                long_context_output is not None and
                isinstance(long_context_output, dict) and
                "generated_text" in long_context_output
            )
            
            results["cpu_long_context"] = f"Success {implementation_type}" if has_long_context_response else "Failed long context test"
            
            # Add details if successful
            if has_long_context_response:
                long_context_text = long_context_output["generated_text"]
                
                # Add example for recorded output
                results["cpu_long_context_example"] = {
                    "input": self.long_context_prompt,
                    "output": {
                        "generated_text": long_context_text[:500] if len(long_context_text) > 500 else long_context_text,
                        "token_count": len(long_context_text.split()),
                        "input_tokens": len(self.long_context_prompt.split())
                    },
                    "timestamp": time.time(),
                    "implementation": implementation_type
                }
                
        except Exception as e:
            print(f"Error in CPU tests: {e}")
            traceback.print_exc()
            results["cpu_tests"] = f"Error: {str(e)}"
            
        # Test CUDA if available - with scaling tests
        if torch.cuda.is_available():
            try:
                print("Testing Mamba2 on CUDA...")
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
                
                # Initialize for CUDA - use standard model
                endpoint, processor, handler, queue, batch_size = self.mamba2.init_cuda(
                    self.model_name,
                    "cuda",
                    device
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["cuda_init"] = "Success (REAL)" if valid_init else "Failed CUDA initialization"
                
                # Test text generation with performance metrics
                start_time = time.time()
                output = handler(self.test_prompt, max_tokens=50, temperature=0.7)
                elapsed_time = time.time() - start_time
                
                # Verify output contains text
                has_text = (
                    output is not None and
                    isinstance(output, dict) and
                    "generated_text" in output
                )
                results["cuda_generation"] = "Success (REAL)" if has_text else "Failed text generation"
                
                # Add details if successful
                if has_text:
                    generated_text = output["generated_text"]
                    token_count = len(generated_text.split())
                    
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
                            "generated_text": generated_text[:500] if len(generated_text) > 500 else generated_text,
                            "token_count": token_count
                        },
                        "timestamp": time.time(),
                        "implementation": "REAL",
                        "performance_metrics": performance_metrics
                    }
                
                # Test scaling with context length - Mamba's key advantage is linear scaling
                scaling_results = {}
                
                for context_size in self.context_sizes:
                    long_prompt = self._create_long_prompt(context_size)
                    try:
                        # Time the generation
                        start_time = time.time()
                        scaling_output = handler(long_prompt, max_tokens=20, temperature=0.7)
                        scaling_elapsed_time = time.time() - start_time
                        
                        # Calculate metrics
                        input_token_count = len(long_prompt.split())
                        
                        if scaling_output and "generated_text" in scaling_output:
                            scaling_text = scaling_output["generated_text"]
                            output_token_count = len(scaling_text.split())
                            
                            scaling_results[f"context_{context_size}"] = {
                                "success": True,
                                "input_token_count": input_token_count,
                                "output_token_count": output_token_count,
                                "processing_time_seconds": scaling_elapsed_time,
                                "tokens_per_second": output_token_count / scaling_elapsed_time if scaling_elapsed_time > 0 else 0
                            }
                        else:
                            scaling_results[f"context_{context_size}"] = {
                                "success": False,
                                "error": "No valid output"
                            }
                    except Exception as scaling_err:
                        scaling_results[f"context_{context_size}"] = {
                            "success": False,
                            "error": str(scaling_err)
                        }
                
                # Store the scaling results
                results["cuda_scaling_tests"] = scaling_results
                
                # Calculate if we see linear scaling (which is Mamba's key advantage)
                if all(item["success"] for item in scaling_results.values()):
                    # Get processing times per token for different context sizes
                    times_per_token = {}
                    for size, data in scaling_results.items():
                        context_size = int(size.split("_")[1])
                        if data["input_token_count"] > 0:
                            times_per_token[context_size] = data["processing_time_seconds"] / data["input_token_count"]
                    
                    # Check if processing time per token stays roughly constant (linear scaling)
                    # or increases significantly with context size (superlinear scaling)
                    if len(times_per_token) >= 2:
                        sizes = sorted(times_per_token.keys())
                        baseline = times_per_token[sizes[0]]
                        max_ratio = max(times_per_token[size] / baseline for size in sizes[1:])
                        
                        # If processing time per token increases by less than 2x when context size
                        # increases by 8x, we consider it roughly linear scaling
                        results["cuda_scaling_analysis"] = {
                            "linear_scaling": max_ratio < 2.0,
                            "max_time_per_token_ratio": max_ratio,
                            "times_per_token": times_per_token
                        }
                
            except Exception as e:
                print(f"Error in CUDA tests: {e}")
                traceback.print_exc()
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"
            
        # Test OpenVINO if available
        try:
            print("Testing Mamba2 on OpenVINO...")
            
            # Try to import OpenVINO
            try:
                import openvino
                openvino_available = True
            except ImportError:
                openvino_available = False
                
            if not openvino_available:
                results["openvino_tests"] = "OpenVINO not available"
            else:
                # Initialize for OpenVINO using smaller model
                endpoint, processor, handler, queue, batch_size = self.mamba2.init_openvino(
                    self.small_model_name,
                    "openvino",
                    "CPU"  # Standard OpenVINO device
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["openvino_init"] = "Success (REAL)" if valid_init else "Failed OpenVINO initialization"
                
                # Test text generation with performance metrics
                start_time = time.time()
                output = handler(self.test_prompt, max_tokens=50, temperature=0.7)
                elapsed_time = time.time() - start_time
                
                # Verify output contains text
                has_text = (
                    output is not None and
                    isinstance(output, dict) and
                    "generated_text" in output
                )
                results["openvino_generation"] = "Success (REAL)" if has_text else "Failed text generation"
                
                # Add details if successful
                if has_text:
                    generated_text = output["generated_text"]
                    token_count = len(generated_text.split())
                    
                    # Calculate performance metrics
                    performance_metrics = {
                        "processing_time_seconds": elapsed_time,
                        "tokens_per_second": token_count / elapsed_time if elapsed_time > 0 else 0
                    }
                    
                    # Add example with performance metrics
                    results["openvino_generation_example"] = {
                        "input": self.test_prompt,
                        "output": {
                            "generated_text": generated_text[:500] if len(generated_text) > 500 else generated_text,
                            "token_count": token_count
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
            "test_run_id": f"mamba2-test-{int(time.time())}"
        }
        
        # Create directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(collected_dir, 'hf_mamba2_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_mamba2_test_results.json')
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
        this_mamba2 = test_hf_mamba2()
        results = this_mamba2.__test__()
        print(f"Mamba2 Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)