# Standard library imports first
import os
import sys
import json
import time
import datetime
import traceback
from unittest.mock import MagicMock, patch

# Third-party imports next
import numpy as np

# Use absolute path setup
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")

# Try/except pattern for importing optional dependencies
try:
    import torch
except ImportError:
    torch = MagicMock()
    print("Warning: torch not available, using mock implementation")

try:
    import transformers
except ImportError:
    transformers = MagicMock()
    print("Warning: transformers not available, using mock implementation")

# Import image processing libraries with proper error handling
try:
    from PIL import Image
except ImportError:
    Image = MagicMock()
    print("Warning: PIL not available, using mock implementation")

# Try to import the required modules from ipfs_accelerate_py
try:
    from ipfs_accelerate_py.worker.skillset.hf_instructblip import hf_instructblip
except ImportError:
    # If the real module doesn't exist, create a mock implementation
    class hf_instructblip:
        def __init__(self, resources=None, metadata=None):
            self.resources = resources if resources else {}
            self.metadata = metadata if metadata else {}
            
        def init_cpu(self, model_name, processor_name, device):
            """Mock CPU initialization"""
            mock_handler = lambda image, text=None, **kwargs: {
                "generated_text": "This is a mock response from InstructBLIP. The image shows a scene that I'm pretending to understand.",
                "implementation_type": "(MOCK)"
            }
            return MagicMock(), MagicMock(), mock_handler, None, 1
            
        def init_cuda(self, model_name, processor_name, device):
            """Mock CUDA initialization"""
            return self.init_cpu(model_name, processor_name, device)
            
        def init_openvino(self, model_name, processor_name, device):
            """Mock OpenVINO initialization"""
            return self.init_cpu(model_name, processor_name, device)
    
    print("Warning: hf_instructblip not found, using mock implementation")

class test_hf_instructblip:
    """
    Test class for Hugging Face InstructBLIP models.
    
    InstructBLIP is an instruction-tuned vision-language model that extends
    BLIP-2 with instruction-following capabilities, enabling it to follow
    natural language instructions for various vision-language tasks.
    """
    
    def __init__(self, resources=None, metadata=None):
        """
        Initialize the InstructBLIP test class.
        
        Args:
            resources (dict, optional): Resources dictionary
            metadata (dict, optional): Metadata dictionary
        """
        # Try to import transformers directly if available
        try:
            import transformers
            transformers_module = transformers
        except ImportError:
            transformers_module = MagicMock()
            
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np,
            "transformers": transformers_module,
            "PIL": Image
        }
        self.metadata = metadata if metadata else {}
        
        # Create InstructBLIP instance
        self.instructblip = hf_instructblip(resources=self.resources, metadata=self.metadata)
        
        # Define model variants
        self.primary_model = "Salesforce/instructblip-vicuna-7b"  # Primary model
        self.alternative_models = [
            "Salesforce/instructblip-flan-t5-xl",    # Smaller model
            "Salesforce/instructblip-flan-t5-xxl"    # Larger model
        ]
        
        # Initialize with primary model
        self.model_name = self.primary_model
        
        try:
            print(f"Attempting to use primary model: {self.model_name}")
            
            # Try to import transformers for validation
            if not isinstance(self.resources["transformers"], MagicMock):
                from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor
                try:
                    # Try to validate model
                    if "instructblip" in self.model_name.lower():
                        # For InstructBLIP, use the processor directly
                        try:
                            # Validate processor exists
                            BlipProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                            print(f"Successfully validated primary model processor: {self.model_name}")
                        except Exception as proc_error:
                            print(f"Warning: Could not validate processor: {proc_error}")
                    else:
                        # Generic check for other model types
                        from transformers import AutoConfig
                        AutoConfig.from_pretrained(self.model_name)
                        print(f"Successfully validated primary model: {self.model_name}")
                        
                except Exception as config_error:
                    print(f"Primary model validation failed: {config_error}")
                    
                    # Try alternatives one by one
                    for alt_model in self.alternative_models:
                        try:
                            print(f"Trying alternative model: {alt_model}")
                            if "instructblip" in alt_model.lower():
                                try:
                                    # Validate processor exists
                                    BlipProcessor.from_pretrained(alt_model, trust_remote_code=True)
                                    print(f"Successfully validated alternative model processor: {alt_model}")
                                    self.model_name = alt_model
                                    break
                                except Exception as proc_error:
                                    print(f"Warning: Could not validate processor: {proc_error}")
                            else:
                                from transformers import AutoConfig
                                AutoConfig.from_pretrained(alt_model)
                                print(f"Successfully validated alternative model: {alt_model}")
                                self.model_name = alt_model
                                break
                        except Exception as alt_error:
                            print(f"Alternative model validation failed: {alt_error}")
                    
                    # If all alternatives failed, check local cache
                    if self.model_name == self.primary_model:
                        # Try to find cached models
                        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models")
                        if os.path.exists(cache_dir):
                            # Look for any instructblip model in cache
                            instructblip_models = [name for name in os.listdir(cache_dir) if "instructblip" in name.lower()]
                            
                            if instructblip_models:
                                # Use the first model found
                                instructblip_model_name = instructblip_models[0].replace("--", "/")
                                print(f"Found local InstructBLIP model: {instructblip_model_name}")
                                self.model_name = instructblip_model_name
                            else:
                                # Use the main model name anyway for mocked testing
                                print("No suitable InstructBLIP models found in cache, using primary model for mocked tests")
                        else:
                            # Use primary model anyway for mocked testing
                            print("No cache directory found, using primary model for mocked tests")
            else:
                # If transformers is mocked, use local test model
                print("Transformers is mocked, using primary model for tests")
                
        except Exception as e:
            print(f"Error finding model: {e}")
            # Keep original models in case of error
            
        # The processor name is usually the same as the model name for InstructBLIP
        self.processor_name = self.model_name
            
        print(f"Using model: {self.model_name}")
        
        # Create test image and prompts
        self.test_image_path = "test.jpg"
        self.test_image = self._create_test_image()
        
        # Test prompts for different InstructBLIP tasks
        self.test_prompts = {
            "caption": "Describe this image in detail.",
            "vqa": "What objects are visible in this image?",
            "reasoning": "Why might this scene be important?",
            "classification": "What category does this image belong to?",
        }
        
        # Initialize collection arrays for examples and status
        self.examples = []
        self.status_messages = {}
        return None
        
    def _create_test_image(self):
        """Create a test image if one doesn't exist"""
        try:
            # Check if test image exists
            if os.path.exists(self.test_image_path):
                # Load existing image
                if not isinstance(Image, MagicMock):
                    return Image.open(self.test_image_path)
                return "test.jpg"  # Mock case
                
            # Create a new test image if PIL is available
            if not isinstance(Image, MagicMock):
                # Create a simple test image (100x100 with gradients)
                width, height = 224, 224  # Standard size
                img = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Create a gradient
                for y in range(height):
                    for x in range(width):
                        img[y, x, 0] = int(255 * y / height)  # R
                        img[y, x, 1] = int(255 * x / width)   # G
                        img[y, x, 2] = int(255 * (x + y) / (width + height))  # B
                
                # Draw a white circle in the middle
                center_x, center_y = width // 2, height // 2
                radius = min(width, height) // 4
                for y in range(height):
                    for x in range(width):
                        if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                            img[y, x] = [255, 255, 255]
                
                # Convert to PIL image
                image = Image.fromarray(img)
                
                # Save the image
                image.save(self.test_image_path)
                print(f"Created test image at {self.test_image_path}")
                
                return image
            else:
                return "test.jpg"  # Mock case
        except Exception as e:
            print(f"Error creating test image: {e}")
            return "test.jpg"  # Fall back to path string

    def init_cpu(self):
        """Initialize InstructBLIP model with CPU"""
        print(f"Initializing InstructBLIP on CPU with model {self.model_name}")
        
        endpoint, processor, handler, queue, batch_size = self.instructblip.init_cpu(
            self.model_name,
            self.processor_name,
            "cpu"
        )
        
        return endpoint, processor, handler
    
    def init_cuda(self):
        """Initialize InstructBLIP model with CUDA"""
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available() if not isinstance(torch, MagicMock) else False
        
        if not cuda_available:
            print("CUDA not available, falling back to CPU")
            return self.init_cpu()
            
        print(f"Initializing InstructBLIP on CUDA with model {self.model_name}")
        
        endpoint, processor, handler, queue, batch_size = self.instructblip.init_cuda(
            self.model_name,
            self.processor_name,
            "cuda:0"
        )
        
        return endpoint, processor, handler

    def test(self):
        """
        Run all tests for the InstructBLIP model, organized by hardware platform.
        Tests CPU and CUDA implementations.
        
        Returns:
            dict: Structured test results with status, examples and metadata
        """
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.instructblip is not None else "Failed initialization"
            
            # Check test image
            if isinstance(self.test_image, str):
                results["test_image"] = "Path only"
            elif hasattr(self.test_image, "size") and callable(getattr(self.test_image, "size")):
                results["test_image"] = f"PIL Image: {self.test_image.size}"
            else:
                results["test_image"] = "Unknown image type"
                
        except Exception as e:
            results["init"] = f"Error: {str(e)}"

        # ====== CPU TESTS ======
        try:
            print("Testing InstructBLIP on CPU...")
            
            # Initialize CPU model
            endpoint, processor, handler = self.init_cpu()
            
            valid_init = endpoint is not None and processor is not None and handler is not None
            results["cpu_init"] = "Success" if valid_init else "Failed CPU initialization"
            
            if valid_init:
                # Test with each prompt type
                for task, prompt in self.test_prompts.items():
                    try:
                        # Run the handler with image and prompt
                        start_time = time.time()
                        output = handler(self.test_image, prompt)
                        elapsed_time = time.time() - start_time
                        
                        if output is not None and isinstance(output, dict):
                            has_text = "generated_text" in output
                            implementation_type = output.get("implementation_type", "Unknown")
                            
                            results[f"cpu_{task}"] = f"Success ({implementation_type})" if has_text else "No text generated"
                            
                            # Record example
                            if has_text:
                                generated_text = output["generated_text"]
                                self.examples.append({
                                    "input": {
                                        "image": "test.jpg",
                                        "prompt": prompt
                                    },
                                    "output": {
                                        "generated_text": generated_text
                                    },
                                    "timestamp": datetime.datetime.now().isoformat(),
                                    "elapsed_time": elapsed_time,
                                    "implementation_type": implementation_type,
                                    "platform": "CPU",
                                    "task": task
                                })
                                
                                # Store sample in results
                                results[f"cpu_{task}_sample"] = generated_text[:150] + "..." if len(generated_text) > 150 else generated_text
                        else:
                            results[f"cpu_{task}"] = "Failed - Invalid output format"
                    except Exception as task_error:
                        print(f"Error in CPU {task} test: {task_error}")
                        results[f"cpu_{task}"] = f"Error: {str(task_error)}"
            
        except Exception as e:
            print(f"Error in CPU tests: {e}")
            traceback.print_exc()
            results["cpu_tests"] = f"Error: {str(e)}"
            self.status_messages["cpu"] = f"Failed: {str(e)}"

        # ====== CUDA TESTS ======
        cuda_available = torch.cuda.is_available() if not isinstance(torch, MagicMock) else False
        print(f"CUDA availability check result: {cuda_available}")
        
        if cuda_available:
            try:
                print("Testing InstructBLIP on CUDA...")
                
                # Initialize CUDA model
                endpoint, processor, handler = self.init_cuda()
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["cuda_init"] = "Success" if valid_init else "Failed CUDA initialization"
                
                if valid_init:
                    # Test with each prompt type
                    for task, prompt in self.test_prompts.items():
                        try:
                            # Run the handler with image and prompt
                            start_time = time.time()
                            output = handler(self.test_image, prompt)
                            elapsed_time = time.time() - start_time
                            
                            if output is not None and isinstance(output, dict):
                                has_text = "generated_text" in output
                                implementation_type = output.get("implementation_type", "Unknown")
                                
                                results[f"cuda_{task}"] = f"Success ({implementation_type})" if has_text else "No text generated"
                                
                                # Record example with performance metrics
                                if has_text:
                                    generated_text = output["generated_text"]
                                    example_output = {
                                        "generated_text": generated_text
                                    }
                                    
                                    # Include performance metrics if available
                                    if "gpu_memory_used_mb" in output:
                                        example_output["gpu_memory_used_mb"] = output["gpu_memory_used_mb"]
                                        results[f"cuda_{task}_gpu_memory"] = output["gpu_memory_used_mb"]
                                        
                                    if "generation_time" in output:
                                        example_output["generation_time"] = output["generation_time"]
                                        results[f"cuda_{task}_generation_time"] = output["generation_time"]
                                    
                                    self.examples.append({
                                        "input": {
                                            "image": "test.jpg",
                                            "prompt": prompt
                                        },
                                        "output": example_output,
                                        "timestamp": datetime.datetime.now().isoformat(),
                                        "elapsed_time": elapsed_time,
                                        "implementation_type": implementation_type,
                                        "platform": "CUDA",
                                        "task": task
                                    })
                                    
                                    # Store sample in results
                                    results[f"cuda_{task}_sample"] = generated_text[:150] + "..." if len(generated_text) > 150 else generated_text
                            else:
                                results[f"cuda_{task}"] = "Failed - Invalid output format"
                        except Exception as task_error:
                            print(f"Error in CUDA {task} test: {task_error}")
                            results[f"cuda_{task}"] = f"Error: {str(task_error)}"
                
            except Exception as e:
                print(f"Error in CUDA tests: {e}")
                traceback.print_exc()
                results["cuda_tests"] = f"Error: {str(e)}"
                self.status_messages["cuda"] = f"Failed: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"
            self.status_messages["cuda"] = "CUDA not available"

        # Create structured results
        structured_results = {
            "status": results,
            "examples": self.examples,
            "metadata": {
                "model_name": self.model_name,
                "processor_name": self.processor_name,
                "test_timestamp": datetime.datetime.now().isoformat(),
                "python_version": sys.version,
                "torch_version": torch.__version__ if hasattr(torch, "__version__") else "Unknown",
                "transformers_version": transformers.__version__ if hasattr(transformers, "__version__") else "Unknown",
                "platform_status": self.status_messages,
                "cuda_available": cuda_available,
                "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
                "transformers_mocked": isinstance(self.resources["transformers"], MagicMock)
            }
        }

        return structured_results

    def __test__(self):
        """
        Run tests and compare/save results.
        Handles result collection, comparison with expected results, and storage.
        
        Returns:
            dict: Test results
        """
        # Run tests
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {
                "status": {"test_error": str(e)},
                "examples": [],
                "metadata": {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            }
        
        # Create directories if they don't exist
        expected_dir = os.path.join(os.path.dirname(__file__), 'expected_results')
        collected_dir = os.path.join(os.path.dirname(__file__), 'collected_results')
        
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save collected results
        collected_file = os.path.join(collected_dir, 'hf_instructblip_test_results.json')
        with open(collected_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)  # Use default=str for non-serializable objects
            print(f"Saved results to {collected_file}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_instructblip_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                    
                # Basic structure check
                if "status" in expected_results and "status" in test_results:
                    print("Results structure matches expected format.")
                else:
                    print("Warning: Results structure does not match expected format.")
            except Exception as e:
                print(f"Error comparing with expected results: {str(e)}")
                # Create expected results file if there's an error
                with open(expected_file, 'w') as f:
                    json.dump(test_results, f, indent=2, default=str)
                    print(f"Created new expected results file: {expected_file}")
        else:
            # Create expected results file if it doesn't exist
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
                print(f"Created new expected results file: {expected_file}")

        return test_results

if __name__ == "__main__":
    try:
        print("Starting InstructBLIP test...")
        this_instructblip = test_hf_instructblip()
        results = this_instructblip.__test__()
        print("InstructBLIP test completed")
        
        # Print test results summary
        status_dict = results.get("status", {})
        examples = results.get("examples", [])
        metadata = results.get("metadata", {})
        
        # Print platform status
        print("\nINSTRUCTBLIP TEST RESULTS SUMMARY")
        print(f"MODEL: {metadata.get('model_name', 'Unknown')}")
        
        # Extract CPU task status
        cpu_tasks = {}
        for key, value in status_dict.items():
            if key.startswith("cpu_") and not key.endswith("_sample") and key != "cpu_init" and key != "cpu_tests":
                task = key[4:]  # Remove "cpu_" prefix
                cpu_tasks[task] = value
                
        print("\nCPU Task Results:")
        for task, status in cpu_tasks.items():
            print(f"  {task}: {status}")
            
        # Extract CUDA task status if available
        if metadata.get("cuda_available", False):
            cuda_tasks = {}
            for key, value in status_dict.items():
                if key.startswith("cuda_") and not key.endswith("_sample") and key != "cuda_init" and key != "cuda_tests":
                    task = key[5:]  # Remove "cuda_" prefix
                    cuda_tasks[task] = value
                    
            print("\nCUDA Task Results:")
            for task, status in cuda_tasks.items():
                print(f"  {task}: {status}")
        else:
            print("\nCUDA: Not available")
            
        # Print example outputs by task
        task_examples = {}
        
        # Group examples by task
        for example in examples:
            task = example.get("task", "unknown")
            
            if task not in task_examples:
                task_examples[task] = []
                
            task_examples[task].append(example)
        
        # Print one example per task
        print("\nExample Outputs:")
        for task, example_list in task_examples.items():
            if example_list:
                example = example_list[0]
                platform = example.get("platform", "Unknown")
                
                print(f"\n  {task.upper()} ({platform}):")
                print(f"    Prompt: {example.get('input', {}).get('prompt', 'No prompt')}")
                print(f"    Output: {example.get('output', {}).get('generated_text', 'No text')}")
                print(f"    Time: {example.get('elapsed_time', 0):.3f}s")
                
                # Print additional metrics if available
                if "gpu_memory_used_mb" in example.get("output", {}):
                    print(f"    GPU Memory: {example['output']['gpu_memory_used_mb']:.2f} MB")
        
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during testing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)