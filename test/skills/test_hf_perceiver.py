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

# Import hardware detection capabilities if available
try:
    from hardware_detection import (
        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
        detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback
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

# Try to import the Perceiver module from ipfs_accelerate_py
try:
    from ipfs_accelerate_py.worker.skillset.hf_perceiver import hf_perceiver
except ImportError:
    # Create a mock class if the real one doesn't exist
    class hf_perceiver:
        def __init__(self, resources=None, metadata=None):
            self.resources = resources if resources else {}
            self.metadata = metadata if metadata else {}
            
        def init_cpu(self, model_name, processor_name, device_label):
            """Mock CPU initialization for Perceiver models"""
            mock_handler = lambda inputs, **kwargs: {
                "logits": np.random.randn(1, 10),
                "predicted_class": "mock_class",
                "implementation_type": "(MOCK)"
            }
            return MagicMock(), MagicMock(), mock_handler, None, 1
            
        def init_cuda(self, model_name, processor_name, device_label):
            """Mock CUDA initialization for Perceiver models"""
            return self.init_cpu(model_name, processor_name, device_label)
    
    print("Warning: hf_perceiver not found, using mock implementation")

class test_hf_perceiver:
    """
    Test class for Hugging Face Perceiver IO models.
    
    The Perceiver IO architecture is a general-purpose encoder-decoder that can handle
    multiple modalities including text, images, audio, video, and multimodal data.
    """
    
    def __init__(self, resources=None, metadata=None):
        """
        Initialize the Perceiver test class.
        
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
        
        # Create Perceiver instance
        self.perceiver = hf_perceiver(resources=self.resources, metadata=self.metadata)
        
        # Define model variants for different tasks
        self.models = {
            "image_classification": "deepmind/vision-perceiver-conv",
            "text_classification": "deepmind/language-perceiver",
            "multimodal": "deepmind/multimodal-perceiver",
            "masked_language_modeling": "deepmind/language-perceiver-mlm"
        }
        
        # Default to image classification model
        self.default_task = "image_classification"
        self.model_name = self.models[self.default_task]
        self.processor_name = self.model_name  # Usually the same as model name
        
        # Try to validate models
        self._validate_models()
        
        # Create test inputs for different modalities
        self.test_inputs = self._create_test_inputs()
        
        # Initialize collection arrays for examples and status
        self.examples = []
        self.status_messages = {}
        return None
        
    def _validate_models(self):
        """Validate that models exist and fall back if needed"""
        try:
            # Check if we're using real transformers
            if not isinstance(self.resources["transformers"], MagicMock):
                from transformers import AutoConfig
                
                # Try to validate each model
                validated_models = {}
                for task, model in self.models.items():
                    try:
                        print(f"Validating {task} model: {model}")
                        AutoConfig.from_pretrained(model)
                        validated_models[task] = model
                        print(f"✓ Successfully validated {task} model")
                    except Exception as e:
                        print(f"✗ Failed to validate {task} model: {e}")
                        
                # Update models dict with only validated models
                if validated_models:
                    self.models = validated_models
                    self.default_task = list(validated_models.keys())[0]
                    self.model_name = validated_models[self.default_task]
                    self.processor_name = self.model_name
                    print(f"Selected default model: {self.model_name} for {self.default_task}")
                else:
                    print("No models could be validated, using original models")
        except Exception as e:
            print(f"Error validating models: {e}")
            # Keep original models in case of error
    
    def _create_test_inputs(self):
        """Create test inputs for different modalities"""
        test_inputs = {}
        
        # Text input
        test_inputs["text"] = "This is a sample text for testing the Perceiver model."
        
        # Image input
        try:
            # Try to create a test image if PIL is available
            if not isinstance(Image, MagicMock):
                # Create a simple test image (100x100 with gradients)
                width, height = 100, 100
                img = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Create a gradient
                for y in range(height):
                    for x in range(width):
                        img[y, x, 0] = int(255 * y / height)  # R
                        img[y, x, 1] = int(255 * x / width)   # G
                        img[y, x, 2] = int(255 * (x + y) / (width + height))  # B
                
                image = Image.fromarray(img)
                test_inputs["image"] = image
            else:
                test_inputs["image"] = MagicMock()
        except Exception as e:
            print(f"Error creating test image: {e}")
            test_inputs["image"] = MagicMock()
        
        # Multimodal input (text + image)
        test_inputs["multimodal"] = {
            "text": test_inputs["text"],
            "image": test_inputs["image"]
        }
        
        # Audio input (mock)
        test_inputs["audio"] = np.zeros((16000,), dtype=np.float32)  # 1 second at 16kHz
        
        return test_inputs
    
    def _get_test_input_for_task(self, task):
        """Get appropriate test input based on task"""
        if task == "image_classification":
            return self.test_inputs["image"]
        elif task in ["text_classification", "masked_language_modeling"]:
            return self.test_inputs["text"]
        elif task == "multimodal":
            return self.test_inputs["multimodal"]
        else:
            # Default to text
            return self.test_inputs["text"]

    def init_cpu(self, task=None):
        """Initialize Perceiver model on CPU for a specific task"""
        if task is None:
            task = self.default_task
            
        if task not in self.models:
            print(f"Unknown task: {task}, falling back to {self.default_task}")
            task = self.default_task
            
        model_name = self.models[task]
        processor_name = model_name  # Usually the same
        
        print(f"Initializing Perceiver for {task} on CPU with model {model_name}")
        
        # Initialize with CPU
        endpoint, processor, handler, queue, batch_size = self.perceiver.init_cpu(
            model_name,
            processor_name,
            "cpu"
        )
        
        return endpoint, processor, handler, task
    
    def init_cuda(self, task=None):
        """Initialize Perceiver model on CUDA for a specific task"""
        if task is None:
            task = self.default_task
            
        if task not in self.models:
            print(f"Unknown task: {task}, falling back to {self.default_task}")
            task = self.default_task
            
        model_name = self.models[task]
        processor_name = model_name  # Usually the same
        
        print(f"Initializing Perceiver for {task} on CUDA with model {model_name}")
        
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available() if not isinstance(torch, MagicMock) else False
        if not cuda_available:
            print("CUDA not available, falling back to CPU")
            return self.init_cpu(task)
        
        # Initialize with CUDA
        endpoint, processor, handler, queue, batch_size = self.perceiver.init_cuda(
            model_name,
            processor_name,
            "cuda:0"
        )
        
        return endpoint, processor, handler, task
    
    def test_task(self, platform, task=None):
        """Test a specific task on a specific platform"""
        result = {
            "platform": platform,
            "task": task if task is not None else self.default_task,
            "status": "Not run",
            "error": None
        }
        
        try:
            # Initialize model for task
            if platform == "CPU":
                endpoint, processor, handler, task = self.init_cpu(task)
            elif platform == "CUDA":
                endpoint, processor, handler, task = self.init_cuda(task)
            else:
                result["status"] = "Invalid platform"
                result["error"] = f"Unknown platform: {platform}"
                return result
                
            # Get appropriate test input
            test_input = self._get_test_input_for_task(task)
            
            # Test handler
            start_time = time.time()
            output = handler(test_input)
            elapsed_time = time.time() - start_time
            
            # Check if output is valid
            result["output"] = output
            result["elapsed_time"] = elapsed_time
            
            if output is not None:
                result["status"] = "Success"
                
                # Record example
                implementation_type = output.get("implementation_type", "Unknown")
                
                example = {
                    "input": str(test_input)[:100] + "..." if isinstance(test_input, str) and len(str(test_input)) > 100 else str(test_input),
                    "output": output,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "elapsed_time": elapsed_time,
                    "implementation_type": implementation_type,
                    "platform": platform,
                    "task": task
                }
                
                self.examples.append(example)
            else:
                result["status"] = "Failed - No output"
                
        except Exception as e:
            result["status"] = "Error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            
        return result

    def test(self):
        """
        Run tests for the Perceiver model across different tasks and platforms.
        
        Returns:
            dict: Structured test results with status, examples, and metadata
        """
        results = {}
        tasks_results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.perceiver is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"
        
        # Track tested tasks and platforms
        tested_tasks = set()
        tested_platforms = set()
        
        # Run CPU tests for all tasks
        for task in self.models.keys():
            task_result = self.test_task("CPU", task)
            tasks_results[f"cpu_{task}"] = task_result
            tested_tasks.add(task)
            tested_platforms.add("CPU")
            
            # Update status messages
            if task_result["status"] == "Success":
                self.status_messages[f"cpu_{task}"] = "Success"
            else:
                self.status_messages[f"cpu_{task}"] = task_result.get("error", "Failed")
        
        # Run CUDA tests if available
        cuda_available = torch.cuda.is_available() if not isinstance(torch, MagicMock) else False
        if cuda_available:
            for task in self.models.keys():
                task_result = self.test_task("CUDA", task)
                tasks_results[f"cuda_{task}"] = task_result
                tested_platforms.add("CUDA")
                
                # Update status messages
                if task_result["status"] == "Success":
                    self.status_messages[f"cuda_{task}"] = "Success"
                else:
                    self.status_messages[f"cuda_{task}"] = task_result.get("error", "Failed")
        else:
            results["cuda_tests"] = "CUDA not available"
            self.status_messages["cuda"] = "Not available"
        
        # Summarize task results
        for task in tested_tasks:
            cpu_success = tasks_results.get(f"cpu_{task}", {}).get("status") == "Success"
            cuda_success = tasks_results.get(f"cuda_{task}", {}).get("status") == "Success" if cuda_available else False
            
            results[f"task_{task}"] = {
                "cpu": "Success" if cpu_success else "Failed",
                "cuda": "Success" if cuda_success else "Not available" if not cuda_available else "Failed",
                "platforms_success": [p for p in tested_platforms if tasks_results.get(f"{p.lower()}_{task}", {}).get("status") == "Success"]
            }
        
        # Create structured results with tasks details
        structured_results = {
            "status": results,
            "task_results": tasks_results,
            "examples": self.examples,
            "metadata": {
                "models": self.models,
                "default_task": self.default_task,
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
        # Run actual tests
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
        collected_file = os.path.join(collected_dir, 'hf_perceiver_test_results.json')
        with open(collected_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)  # Use default=str to handle non-serializable objects
            print(f"Saved results to {collected_file}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_perceiver_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                    
                # Simple check to verify structure
                if "status" in expected_results and "status" in test_results:
                    print("Test completed with expected results structure.")
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
        print("Starting Perceiver test...")
        this_perceiver = test_hf_perceiver()
        results = this_perceiver.__test__()
        print("Perceiver test completed")
        
        # Print test results in detailed format for better parsing
        status_dict = results.get("status", {})
        task_results = results.get("task_results", {})
        examples = results.get("examples", [])
        metadata = results.get("metadata", {})
        
        # Print summary in a parser-friendly format
        print("\nPERCEIVER TEST RESULTS SUMMARY")
        print(f"Default task: {metadata.get('default_task', 'Unknown')}")
        
        # Print task results summary
        print("\nTask Status:")
        for key, value in status_dict.items():
            if key.startswith("task_"):
                task_name = key[5:]  # Remove "task_" prefix
                print(f"  {task_name}:")
                if isinstance(value, dict):
                    for platform, status in value.items():
                        if platform != "platforms_success":
                            print(f"    {platform}: {status}")
                else:
                    print(f"    {value}")
        
        # Print example outputs by task and platform
        task_platform_examples = {}
        
        # Group examples by task and platform
        for example in examples:
            task = example.get("task", "unknown")
            platform = example.get("platform", "Unknown")
            key = f"{task}_{platform}"
            
            if key not in task_platform_examples:
                task_platform_examples[key] = []
                
            task_platform_examples[key].append(example)
        
        # Print one example per task/platform
        print("\nExample Outputs:")
        for key, example_list in task_platform_examples.items():
            if example_list:
                example = example_list[0]
                task = example.get("task", "unknown")
                platform = example.get("platform", "Unknown")
                
                print(f"  {task} - {platform}:")
                print(f"    Input: {example.get('input', 'None')}")
                
                # Format output nicely based on content
                output = example.get("output", {})
                if isinstance(output, dict):
                    # Show only key fields to keep it readable
                    if "logits" in output:
                        print(f"    Output: {output.get('implementation_type', 'Unknown')} - Contains logits")
                    elif "predicted_class" in output:
                        print(f"    Output: Predicted class: {output.get('predicted_class', 'None')}")
                    elif "text" in output:
                        print(f"    Output: {output.get('text', 'None')[:50]}...")
                    else:
                        # Print a short summary
                        keys = list(output.keys())
                        print(f"    Output: Contains {len(keys)} fields: {', '.join(keys)}")
                else:
                    print(f"    Output: {str(output)[:50]}...")
                
                print(f"    Time: {example.get('elapsed_time', 0):.3f}s")
        
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during testing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)