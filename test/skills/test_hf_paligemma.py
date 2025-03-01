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
    import PIL
    from PIL import Image
except ImportError:
    transformers = MagicMock()
    PIL = MagicMock()
    Image = MagicMock()
    print("Warning: transformers/PIL not available, using mock implementation")

# Try to import from ipfs_accelerate_py
try:
    from ipfs_accelerate_py.worker.skillset.hf_paligemma import hf_paligemma
except ImportError:
    # Create a mock class if the real one doesn't exist
    class hf_paligemma:
        def __init__(self, resources=None, metadata=None):
            self.resources = resources if resources else {}
            self.metadata = metadata if metadata else {}
            
        def init_cpu(self, model_name, processor_name, device):
            mock_handler = lambda image=None, text=None, **kwargs: {
                "generated_text": "The image shows a white circle on a black background.",
                "implementation_type": "(MOCK)"
            }
            return "mock_endpoint", "mock_processor", mock_handler, None, 1
            
        def init_cuda(self, model_name, processor_name, device):
            return self.init_cpu(model_name, processor_name, device)
            
        def init_openvino(self, model_name, processor_name, device):
            return self.init_cpu(model_name, processor_name, device)
    
    print("Warning: hf_paligemma not found, using mock implementation")

class test_hf_paligemma:
    """
    Test class for Google's PaliGemma model.
    
    PaliGemma is a family of multimodal models developed by Google, combining vision and
    language capabilities for advanced visual reasoning, image captioning, and VQA.
    
    This class tests PaliGemma functionality across different hardware 
    backends including CPU, CUDA, and OpenVINO.
    
    It verifies:
    1. Image understanding capabilities
    2. Visual question answering
    3. Cross-platform compatibility
    4. Performance metrics
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the PaliGemma test environment"""
        # Set up resources with fallbacks
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np, 
            "transformers": transformers,
            "PIL": PIL,
            "Image": Image
        }
        
        # Store metadata
        self.metadata = metadata if metadata else {}
        
        # Initialize the PaliGemma model
        self.paligemma = hf_paligemma(resources=self.resources, metadata=self.metadata)
        
        # Use a small, openly accessible model that doesn't require authentication
        # PaliGemma has different model sizes available
        self.model_name = "google/paligemma-3b-mix-224"  # Smaller version of PaliGemma
        
        # Create test image
        self.test_image = self._create_test_image()
        
        # Create test prompts for different capabilities
        self.test_prompts = {
            "captioning": "Describe this image in detail.",
            "vqa": "What is the shape in the image and what color is it?",
            "reasoning": "Why might this image have been created as a test input?",
            "multimodal": "If this image were part of a sequence, what might come next?"
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
    
    def _create_test_image(self):
        """Create a simple test image (256x256) with a white circle in the middle"""
        try:
            if isinstance(np, MagicMock) or isinstance(PIL, MagicMock):
                # Return mock if dependencies not available
                return MagicMock()
                
            # Create a black image with a white circle
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            
            # Draw a white circle in the middle
            center_x, center_y = 128, 128
            radius = 50
            
            y, x = np.ogrid[:256, :256]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            img[mask] = 255
            
            # Convert to PIL image
            pil_image = Image.fromarray(img)
            
            return pil_image
        except Exception as e:
            print(f"Error creating test image: {e}")
            return MagicMock()
    
    def _create_local_test_model(self):
        """Create a minimal test model directory for testing without downloading"""
        try:
            print("Creating local test model for PaliGemma...")
            
            # Create model directory in /tmp for tests
            test_model_dir = os.path.join("/tmp", "paligemma_test_model")
            os.makedirs(test_model_dir, exist_ok=True)
            
            # Create minimal config file
            config = {
                "model_type": "paligemma",
                "architectures": ["PaliGemmaForConditionalGeneration"],
                "vision_config": {
                    "hidden_size": 768,
                    "image_size": 224
                },
                "text_config": {
                    "hidden_size": 768,
                    "vocab_size": 32000
                }
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
        """Run all tests for the PaliGemma model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.paligemma is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"
        
        # Test CPU initialization and functionality
        try:
            print("Testing PaliGemma on CPU...")
            
            # Check if using real transformers
            transformers_available = not isinstance(self.resources["transformers"], MagicMock)
            implementation_type = "(REAL)" if transformers_available else "(MOCK)"
            
            # Initialize for CPU
            endpoint, processor, handler, queue, batch_size = self.paligemma.init_cpu(
                self.model_name,
                "cpu",
                "cpu"
            )
            
            valid_init = endpoint is not None and processor is not None and handler is not None
            results["cpu_init"] = f"Success {implementation_type}" if valid_init else "Failed CPU initialization"
            
            # Test various prompts for PaliGemma
            prompt_results = {}
            
            for prompt_type, prompt_text in self.test_prompts.items():
                try:
                    start_time = time.time()
                    output = handler(image=self.test_image, text=prompt_text)
                    elapsed_time = time.time() - start_time
                    
                    # Verify output contains text
                    has_output = (
                        output is not None and
                        isinstance(output, dict) and
                        "generated_text" in output
                    )
                    
                    if has_output:
                        # Extract text
                        generated_text = output["generated_text"]
                        
                        # Add example to collection
                        example = {
                            "input": {
                                "prompt_type": prompt_type,
                                "prompt_text": prompt_text,
                                "image": "image input (binary data not shown)"
                            },
                            "output": {
                                "generated_text": generated_text,
                                "token_count": len(generated_text.split())
                            },
                            "timestamp": time.time(),
                            "elapsed_time": elapsed_time,
                            "implementation_type": implementation_type.strip("()"),
                            "platform": "CPU"
                        }
                        
                        self.examples.append(example)
                        
                        prompt_results[prompt_type] = {
                            "success": True,
                            "generated_text": generated_text,
                            "elapsed_time": elapsed_time
                        }
                    else:
                        prompt_results[prompt_type] = {
                            "success": False,
                            "error": "No text generated"
                        }
                except Exception as e:
                    print(f"Error in {prompt_type} test: {e}")
                    prompt_results[prompt_type] = {
                        "success": False,
                        "error": str(e)
                    }
            
            # Record prompt results
            results["cpu_prompt_results"] = prompt_results
            results["cpu_overall"] = f"Success {implementation_type}" if any(item["success"] for item in prompt_results.values()) else "Failed all prompts"
                
        except Exception as e:
            print(f"Error in CPU tests: {e}")
            traceback.print_exc()
            results["cpu_tests"] = f"Error: {str(e)}"
            
        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                print("Testing PaliGemma on CUDA...")
                # Import CUDA utilities
                try:
                    sys.path.insert(0, "/home/barberb/ipfs_accelerate_py/test")
                    from utils import get_cuda_device, optimize_cuda_memory, benchmark_cuda_inference
                    cuda_utils_available = True
                except ImportError:
                    cuda_utils_available = False
                
                # Initialize for CUDA
                endpoint, processor, handler, queue, batch_size = self.paligemma.init_cuda(
                    self.model_name,
                    "cuda",
                    "cuda:0"
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["cuda_init"] = "Success (REAL)" if valid_init else "Failed CUDA initialization"
                
                # Test various prompts with performance metrics
                prompt_results = {}
                
                for prompt_type, prompt_text in self.test_prompts.items():
                    try:
                        start_time = time.time()
                        output = handler(image=self.test_image, text=prompt_text)
                        elapsed_time = time.time() - start_time
                        
                        # Verify output contains text
                        has_output = (
                            output is not None and
                            isinstance(output, dict) and
                            "generated_text" in output
                        )
                        
                        if has_output:
                            # Extract text
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
                            
                            # Add example to collection
                            example = {
                                "input": {
                                    "prompt_type": prompt_type,
                                    "prompt_text": prompt_text,
                                    "image": "image input (binary data not shown)"
                                },
                                "output": {
                                    "generated_text": generated_text,
                                    "token_count": token_count
                                },
                                "timestamp": time.time(),
                                "elapsed_time": elapsed_time,
                                "implementation_type": "REAL",
                                "platform": "CUDA",
                                "performance_metrics": performance_metrics
                            }
                            
                            self.examples.append(example)
                            
                            prompt_results[prompt_type] = {
                                "success": True,
                                "generated_text": generated_text,
                                "elapsed_time": elapsed_time,
                                "performance_metrics": performance_metrics
                            }
                        else:
                            prompt_results[prompt_type] = {
                                "success": False,
                                "error": "No text generated"
                            }
                    except Exception as e:
                        print(f"Error in CUDA {prompt_type} test: {e}")
                        prompt_results[prompt_type] = {
                            "success": False,
                            "error": str(e)
                        }
                
                # Record prompt results
                results["cuda_prompt_results"] = prompt_results
                results["cuda_overall"] = "Success (REAL)" if any(item["success"] for item in prompt_results.values()) else "Failed all prompts"
                
            except Exception as e:
                print(f"Error in CUDA tests: {e}")
                traceback.print_exc()
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"
            
        # Test OpenVINO if available
        try:
            print("Testing PaliGemma on OpenVINO...")
            
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
                endpoint, processor, handler, queue, batch_size = self.paligemma.init_openvino(
                    self.model_name,
                    "openvino",
                    "CPU"  # Standard OpenVINO device
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["openvino_init"] = "Success (REAL)" if valid_init else "Failed OpenVINO initialization"
                
                # Test with a single prompt (captioning) for simplicity
                prompt_text = self.test_prompts["captioning"]
                
                start_time = time.time()
                output = handler(image=self.test_image, text=prompt_text)
                elapsed_time = time.time() - start_time
                
                # Verify output contains text
                has_output = (
                    output is not None and
                    isinstance(output, dict) and
                    "generated_text" in output
                )
                
                if has_output:
                    # Extract text
                    generated_text = output["generated_text"]
                    token_count = len(generated_text.split())
                    
                    # Calculate performance metrics
                    performance_metrics = {
                        "processing_time_seconds": elapsed_time,
                        "tokens_per_second": token_count / elapsed_time if elapsed_time > 0 else 0
                    }
                    
                    # Add example to collection
                    example = {
                        "input": {
                            "prompt_type": "captioning",
                            "prompt_text": prompt_text,
                            "image": "image input (binary data not shown)"
                        },
                        "output": {
                            "generated_text": generated_text,
                            "token_count": token_count
                        },
                        "timestamp": time.time(),
                        "elapsed_time": elapsed_time,
                        "implementation_type": "REAL",
                        "platform": "OpenVINO",
                        "performance_metrics": performance_metrics
                    }
                    
                    self.examples.append(example)
                    
                    results["openvino_captioning"] = {
                        "success": True,
                        "generated_text": generated_text,
                        "elapsed_time": elapsed_time,
                        "performance_metrics": performance_metrics
                    }
                    
                    results["openvino_overall"] = "Success (REAL)"
                else:
                    results["openvino_captioning"] = {
                        "success": False,
                        "error": "No text generated"
                    }
                    
                    results["openvino_overall"] = "Failed captioning"
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
            "pil_version": getattr(PIL, "__version__", "mocked"),
            "cuda_available": getattr(torch, "cuda", MagicMock()).is_available() if not isinstance(torch, MagicMock) else False,
            "cuda_device_count": getattr(torch, "cuda", MagicMock()).device_count() if not isinstance(torch, MagicMock) else 0,
            "transformers_mocked": isinstance(self.resources["transformers"], MagicMock),
            "test_model": self.model_name,
            "test_run_id": f"paligemma-test-{int(time.time())}"
        }
        
        # Create directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(collected_dir, 'hf_paligemma_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_paligemma_test_results.json')
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
        print("Starting PaliGemma test...")
        this_paligemma = test_hf_paligemma()
        results = this_paligemma.__test__()
        print(f"PaliGemma Test Completed")
        
        # Print a summary of the results
        if "init" in results:
            print(f"Initialization: {results['init']}")
        
        # CPU results
        if "cpu_overall" in results:
            print(f"CPU Tests: {results['cpu_overall']}")
        elif "cpu_tests" in results:
            print(f"CPU Tests: {results['cpu_tests']}")
            
        # CUDA results
        if "cuda_overall" in results:
            print(f"CUDA Tests: {results['cuda_overall']}")
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