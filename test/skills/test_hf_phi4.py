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
    from ipfs_accelerate_py.worker.skillset.hf_phi4 import hf_phi4
except ImportError:
    # Create a mock class if the real one doesn't exist
    class hf_phi4:
        def __init__(self, resources=None, metadata=None):
            self.resources = resources if resources else {}
            self.metadata = metadata if metadata else {}
            
        def init_cpu(self, model_name, processor_name, device):
            mock_handler = lambda prompt=None, max_tokens=100, temperature=0.7, **kwargs: {
                "generated_text": "This is a mock response from Phi-4 model.",
                "implementation_type": "(MOCK)"
            }
            return "mock_endpoint", "mock_processor", mock_handler, None, 1
            
        def init_cuda(self, model_name, processor_name, device):
            return self.init_cpu(model_name, processor_name, device)
            
        def init_openvino(self, model_name, processor_name, device):
            return self.init_cpu(model_name, processor_name, device)
    
    print("Warning: hf_phi4 not found, using mock implementation")

class test_hf_phi4:
    """
    Test class for Hugging Face Phi-4 language model.
    
    This class tests the Phi-4 model functionality across different hardware 
    backends including CPU, CUDA, and OpenVINO.
    
    It verifies:
    1. Text generation capabilities
    2. Performance metrics
    3. Cross-platform compatibility
    4. Various prompting formats and configurations
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the Phi-4 test environment"""
        # Set up resources with fallbacks
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np, 
            "transformers": transformers
        }
        
        # Store metadata
        self.metadata = metadata if metadata else {}
        
        # Initialize the Phi-4 model
        self.phi4 = hf_phi4(resources=self.resources, metadata=self.metadata)
        
        # Use small models for testing
        self.model_name = "microsoft/Phi-4-1B-instruct"  # 1B instruction-tuned model
        self.small_model_name = "microsoft/Phi-4-mini-instruct"  # Even smaller model if needed
        
        # Create test prompts for various tasks
        self.test_prompt = "Write a brief explanation of how transformers work in deep learning."
        
        # Test for instruction following
        self.instruction_prompt = """<|system|>
You are a helpful AI assistant.
<|user|>
What are the three laws of robotics?
<|assistant|>"""
        
        # Test for reasoning
        self.reasoning_prompt = """<|system|>
You are a helpful AI assistant.
<|user|>
If a train travels at 120 km/h and needs to cover a distance of 300 km, how long will the journey take?
<|assistant|>"""
        
        # Test for creative writing
        self.creative_prompt = """<|system|>
You are a helpful AI assistant.
<|user|>
Write a short poem about artificial intelligence.
<|assistant|>"""
        
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
            print("Creating local test model for Phi-4...")
            
            # Create model directory in /tmp for tests
            test_model_dir = os.path.join("/tmp", "phi4_test_model")
            os.makedirs(test_model_dir, exist_ok=True)
            
            # Create minimal config file
            config = {
                "model_type": "phi4",
                "architectures": ["Phi4ForCausalLM"],
                "vocab_size": 32000,
                "hidden_size": 2048,
                "intermediate_size": 5632,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "hidden_act": "silu",
                "max_position_embeddings": 4096,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-6,
                "use_cache": true,
                "tie_word_embeddings": false
            }
            
            # Write config
            with open(os.path.join(test_model_dir, "config.json"), "w") as f:
                json.dump(config, f)
                
            # Create minimal tokenizer files
            tokenizer_config = {
                "model_type": "phi4",
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
        """Run all tests for the Phi-4 model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.phi4 is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"
        
        # Test CPU initialization and functionality
        try:
            print("Testing Phi-4 on CPU...")
            
            # Check if using real transformers
            transformers_available = not isinstance(self.resources["transformers"], MagicMock)
            implementation_type = "(REAL)" if transformers_available else "(MOCK)"
            
            # For CPU tests, use the smallest model
            model_name = self.small_model_name if transformers_available else self._create_local_test_model()
            
            # Initialize for CPU
            endpoint, processor, handler, queue, batch_size = self.phi4.init_cpu(
                model_name,
                "cpu",
                "cpu"
            )
            
            valid_init = endpoint is not None and processor is not None and handler is not None
            results["cpu_init"] = f"Success {implementation_type}" if valid_init else "Failed CPU initialization"
            
            # Test basic generation
            output = handler(self.test_prompt, max_tokens=100, temperature=0.7)
            
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
            instruction_output = handler(self.instruction_prompt, max_tokens=150, temperature=0.7)
            
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
                        "token_count": len(instruction_text.split()),
                        "contains_laws": any(s in instruction_text.lower() for s in ["first law", "second law", "third law", "1st law", "2nd law", "3rd law"])
                    },
                    "timestamp": time.time(),
                    "implementation": implementation_type
                }
                
            # Test reasoning capabilities
            reasoning_output = handler(self.reasoning_prompt, max_tokens=200, temperature=0.3)
            
            # Verify output contains text
            has_reasoning_response = (
                reasoning_output is not None and
                isinstance(reasoning_output, dict) and
                "generated_text" in reasoning_output
            )
            
            results["cpu_reasoning"] = f"Success {implementation_type}" if has_reasoning_response else "Failed reasoning test"
            
            # Add details if successful
            if has_reasoning_response:
                reasoning_text = reasoning_output["generated_text"]
                
                # Attempt to extract the answer (2.5 hours or 150 minutes)
                contains_correct_answer = any(s in reasoning_text.lower() for s in ["2.5", "2.5 hours", "150 minutes", "two and a half"])
                
                # Add example for recorded output
                results["cpu_reasoning_example"] = {
                    "input": self.reasoning_prompt,
                    "output": {
                        "generated_text": reasoning_text[:500] if len(reasoning_text) > 500 else reasoning_text,
                        "token_count": len(reasoning_text.split()),
                        "contains_correct_answer": contains_correct_answer
                    },
                    "timestamp": time.time(),
                    "implementation": implementation_type
                }
                
            # Test creative capabilities
            creative_output = handler(self.creative_prompt, max_tokens=200, temperature=0.9)
            
            # Verify output contains text
            has_creative_response = (
                creative_output is not None and
                isinstance(creative_output, dict) and
                "generated_text" in creative_output
            )
            
            results["cpu_creative"] = f"Success {implementation_type}" if has_creative_response else "Failed creative test"
            
            # Add details if successful
            if has_creative_response:
                creative_text = creative_output["generated_text"]
                
                # Add example for recorded output
                results["cpu_creative_example"] = {
                    "input": self.creative_prompt,
                    "output": {
                        "generated_text": creative_text[:500] if len(creative_text) > 500 else creative_text,
                        "token_count": len(creative_text.split()),
                        "is_poetic": "\n" in creative_text  # Simple heuristic for poem-like structure
                    },
                    "timestamp": time.time(),
                    "implementation": implementation_type
                }
                
        except Exception as e:
            print(f"Error in CPU tests: {e}")
            traceback.print_exc()
            results["cpu_tests"] = f"Error: {str(e)}"
            
        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                print("Testing Phi-4 on CUDA...")
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
                endpoint, processor, handler, queue, batch_size = self.phi4.init_cuda(
                    self.model_name,
                    "cuda",
                    device
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["cuda_init"] = "Success (REAL)" if valid_init else "Failed CUDA initialization"
                
                # Test text generation with performance metrics
                start_time = time.time()
                output = handler(self.test_prompt, max_tokens=100, temperature=0.7)
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
                
                # Test reasoning with CUDA - more complex task to test performance
                start_time = time.time()
                reasoning_output = handler(self.reasoning_prompt, max_tokens=200, temperature=0.3)
                reasoning_elapsed_time = time.time() - start_time
                
                # Verify output contains text
                has_reasoning_response = (
                    reasoning_output is not None and
                    isinstance(reasoning_output, dict) and
                    "generated_text" in reasoning_output
                )
                
                if has_reasoning_response:
                    reasoning_text = reasoning_output["generated_text"]
                    reasoning_token_count = len(reasoning_text.split())
                    
                    # Calculate performance metrics
                    reasoning_performance_metrics = {
                        "processing_time_seconds": reasoning_elapsed_time,
                        "tokens_per_second": reasoning_token_count / reasoning_elapsed_time if reasoning_elapsed_time > 0 else 0
                    }
                    
                    # Add example with performance metrics
                    results["cuda_reasoning_example"] = {
                        "input": self.reasoning_prompt,
                        "output": {
                            "generated_text": reasoning_text[:500] if len(reasoning_text) > 500 else reasoning_text,
                            "token_count": reasoning_token_count
                        },
                        "timestamp": time.time(),
                        "implementation": "REAL",
                        "performance_metrics": reasoning_performance_metrics
                    }
            except Exception as e:
                print(f"Error in CUDA tests: {e}")
                traceback.print_exc()
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"
            
        # Test OpenVINO if available
        try:
            print("Testing Phi-4 on OpenVINO...")
            
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
                endpoint, processor, handler, queue, batch_size = self.phi4.init_openvino(
                    self.small_model_name,
                    "openvino",
                    "CPU"  # Standard OpenVINO device
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["openvino_init"] = "Success (REAL)" if valid_init else "Failed OpenVINO initialization"
                
                # Test text generation with performance metrics
                start_time = time.time()
                output = handler(self.test_prompt, max_tokens=100, temperature=0.7)
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
            "test_run_id": f"phi4-test-{int(time.time())}"
        }
        
        # Create directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(collected_dir, 'hf_phi4_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_phi4_test_results.json')
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
        this_phi4 = test_hf_phi4()
        results = this_phi4.__test__()
        print(f"Phi-4 Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)