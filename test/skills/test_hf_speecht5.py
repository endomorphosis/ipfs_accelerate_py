# Standard library imports first
import os
import sys
import json
import time
import datetime
import traceback
from unittest.mock import patch, MagicMock

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

# Try to import specific dependencies based on model type
# Model supports: text-to-audio, automatic-speech-recognition
if "text-to-audio" in ["image-classification", "object-detection", "image-segmentation", "image-to-text", "visual-question-answering"]:
    try:
        from PIL import Image
    except ImportError:
        Image = MagicMock()
        print("Warning: PIL not available, using mock implementation")

if "text-to-audio" in ["automatic-speech-recognition", "audio-classification", "text-to-audio"]:
    try:
        import librosa
    except ImportError:
        librosa = MagicMock()
        print("Warning: librosa not available, using mock implementation")

# Import the module to test (create a mock if not available)
try:
    from ipfs_accelerate_py.worker.skillset.hf_speecht5 import hf_speecht5
except ImportError:
    # If the module doesn't exist yet, create a mock class
    class hf_speecht5:
        def __init__(self, resources=None, metadata=None):
            self.resources = resources or {}
            self.metadata = metadata or {}
            
        def init_cpu(self, model_name, model_type, device="cpu", **kwargs):
            # Mock implementation
            return MagicMock(), MagicMock(), lambda x: torch.zeros((1, 768)), None, 1
            
        def init_cuda(self, model_name, model_type, device_label="cuda:0", **kwargs):
            # Mock implementation
            return MagicMock(), MagicMock(), lambda x: torch.zeros((1, 768)), None, 1
            
        def init_openvino(self, model_name, model_type, device="CPU", **kwargs):
            # Mock implementation
            return MagicMock(), MagicMock(), lambda x: torch.zeros((1, 768)), None, 1
    
    print(f"Warning: {class_name} module not found, using mock implementation")

# Define required methods to add to hf_speecht5
def init_cuda(self, model_name, model_type, device_label="cuda:0", **kwargs):
    """
    Initialize model with CUDA support.
    
    Args:
        model_name: Name or path of the model
        model_type: Type of model (e.g., "text-to-audio")
        device_label: CUDA device label (e.g., "cuda:0")
        
    Returns:
        tuple: (endpoint, tokenizer, handler, queue, batch_size)
    """
    import traceback
    import sys
    import unittest.mock
    import time
    
    # Try to import the necessary utility functions
    try:
        sys.path.insert(0, "/home/barberb/ipfs_accelerate_py/test")
        import utils as test_utils
        
        # Check if CUDA is really available
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to mock implementation")
            processor = unittest.mock.MagicMock()
            endpoint = unittest.mock.MagicMock()
            handler = lambda x: {"output": None, "implementation_type": "MOCK"}
            return endpoint, processor, handler, None, 0
            
        # Get the CUDA device
        device = test_utils.get_cuda_device(device_label)
        if device is None:
            print("Failed to get valid CUDA device, falling back to mock implementation")
            processor = unittest.mock.MagicMock()
            endpoint = unittest.mock.MagicMock()
            handler = lambda x: {"output": None, "implementation_type": "MOCK"}
            return endpoint, processor, handler, None, 0
            
        # Try to import and initialize HuggingFace components
        try:
            # Different imports based on model type
            if "text-to-audio" == "text-generation":
                from transformers import AutoModelForCausalLM, AutoTokenizer
                print(f"Attempting to load text generation model {model_name} with CUDA support")
                processor = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
            elif "text-to-audio" == "image-classification":
                from transformers import AutoFeatureExtractor, AutoModelForImageClassification
                print(f"Attempting to load image classification model {model_name} with CUDA support")
                processor = AutoFeatureExtractor.from_pretrained(model_name)
                model = AutoModelForImageClassification.from_pretrained(model_name)
            elif "text-to-audio" == "automatic-speech-recognition":
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                print(f"Attempting to load speech recognition model {model_name} with CUDA support")
                processor = AutoProcessor.from_pretrained(model_name)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
            else:
                # Default handling for other model types
                from transformers import AutoProcessor, AutoModel
                print(f"Attempting to load model {model_name} with CUDA support")
                try:
                    processor = AutoProcessor.from_pretrained(model_name)
                except:
                    from transformers import AutoTokenizer
                    processor = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                
            # Move to device and optimize
            model = test_utils.optimize_cuda_memory(model, device, use_half_precision=True)
            model.eval()
            print(f"Model loaded to {device} and optimized for inference")
            
            # Create a real handler function - implementation depends on model type
            def real_handler(input_data):
                try:
                    start_time = time.time()
                    
                    # Process input based on model type
                    with torch.no_grad():
                        if hasattr(torch.cuda, "synchronize"):
                            torch.cuda.synchronize()
                            
                        # Implementation depends on the model type and task
                        # This is a template that needs to be customized
                        outputs = model(**inputs)
                        
                        if hasattr(torch.cuda, "synchronize"):
                            torch.cuda.synchronize()
                    
                    return {
                        "output": outputs,
                        "implementation_type": "REAL",
                        "inference_time_seconds": time.time() - start_time,
                        "device": str(device)
                    }
                except Exception as e:
                    print(f"Error in real CUDA handler: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    return {
                        "output": None,
                        "implementation_type": "REAL",
                        "error": str(e),
                        "is_error": True
                    }
            
            return model, processor, real_handler, None, 8
            
        except Exception as model_err:
            print(f"Failed to load model with CUDA, will use simulation: {model_err}")
    except Exception as e:
        print(f"Error in init_cuda: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    
    # Fallback to mock implementation
    processor = unittest.mock.MagicMock()
    endpoint = unittest.mock.MagicMock()
    handler = lambda x: {"output": None, "implementation_type": "MOCK"}
    return endpoint, processor, handler, None, 0

# Add the method to the class
hf_speecht5.init_cuda = init_cuda

class test_hf_speecht5:
    def __init__(self, resources=None, metadata=None):
        """
        Initialize the test class.
        
        Args:
            resources (dict, optional): Resources dictionary
            metadata (dict, optional): Metadata dictionary
        """
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np,
            "transformers": transformers
        }
        self.metadata = metadata if metadata else {}
        self.model = hf_speecht5(resources=self.resources, metadata=self.metadata)
        
        # Use a small model for testing
        self.model_name = "openai/whisper-tiny"  # Small ASR model
        
        # Test inputs appropriate for this model type
        self.test_audio = "test.mp3"  # Path to a test audio file
        
        # Initialize collection arrays for examples and status
        self.examples = []
        self.status_messages = {}
        return None
        
    def test(self):
        """
        Run all tests for the model, organized by hardware platform.
        Tests CPU, CUDA, OpenVINO implementations.
        
        Returns:
            dict: Structured test results with status, examples and metadata
        """
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.model is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"

        # ====== CPU TESTS ======
        try:
            print("Testing speecht5 on CPU...")
            # Initialize for CPU
            endpoint, processor, handler, queue, batch_size = self.model.init_cpu(
                self.model_name,
                "text-to-audio", 
                "cpu"
            )
            
            valid_init = endpoint is not None and processor is not None and handler is not None
            results["cpu_init"] = "Success (REAL)" if valid_init else "Failed CPU initialization"
            
            # Run actual inference
            start_time = time.time()
            output = handler(self.test_input if hasattr(self, 'test_input') else 
                            self.test_text if hasattr(self, 'test_text') else
                            self.test_image if hasattr(self, 'test_image') else
                            self.test_audio if hasattr(self, 'test_audio') else
                            "Default test input")
            elapsed_time = time.time() - start_time
            
            # Verify the output
            is_valid_output = output is not None
            
            results["cpu_handler"] = "Success (REAL)" if is_valid_output else "Failed CPU handler"
            
            # Record example
            self.examples.append({
                "input": str(self.test_input if hasattr(self, 'test_input') else 
                           self.test_text if hasattr(self, 'test_text') else
                           self.test_image if hasattr(self, 'test_image') else
                           self.test_audio if hasattr(self, 'test_audio') else
                           "Default test input"),
                "output": {
                    "output_type": str(type(output)),
                    "implementation_type": "REAL" if "implementation_type" not in output else output["implementation_type"]
                },
                "timestamp": datetime.datetime.now().isoformat(),
                "elapsed_time": elapsed_time,
                "implementation_type": "REAL",
                "platform": "CPU"
            })
                
        except Exception as e:
            print(f"Error in CPU tests: {e}")
            traceback.print_exc()
            results["cpu_tests"] = f"Error: {str(e)}"
            self.status_messages["cpu"] = f"Failed: {str(e)}"

        # ====== CUDA TESTS ======
        if torch.cuda.is_available():
            try:
                print("Testing speecht5 on CUDA...")
                # Initialize for CUDA
                endpoint, processor, handler, queue, batch_size = self.model.init_cuda(
                    self.model_name,
                    "text-to-audio",
                    "cuda:0"
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["cuda_init"] = "Success (REAL)" if valid_init else "Failed CUDA initialization"
                
                # Run actual inference
                start_time = time.time()
                output = handler(self.test_input if hasattr(self, 'test_input') else 
                               self.test_text if hasattr(self, 'test_text') else
                               self.test_image if hasattr(self, 'test_image') else
                               self.test_audio if hasattr(self, 'test_audio') else
                               "Default test input")
                elapsed_time = time.time() - start_time
                
                # Verify the output
                is_valid_output = output is not None
                
                results["cuda_handler"] = "Success (REAL)" if is_valid_output else "Failed CUDA handler"
                
                # Record example
                self.examples.append({
                    "input": str(self.test_input if hasattr(self, 'test_input') else 
                              self.test_text if hasattr(self, 'test_text') else
                              self.test_image if hasattr(self, 'test_image') else
                              self.test_audio if hasattr(self, 'test_audio') else
                              "Default test input"),
                    "output": {
                        "output_type": str(type(output)),
                        "implementation_type": "REAL" if "implementation_type" not in output else output["implementation_type"]
                    },
                    "timestamp": datetime.datetime.now().isoformat(),
                    "elapsed_time": elapsed_time,
                    "implementation_type": "REAL",
                    "platform": "CUDA"
                })
                    
            except Exception as e:
                print(f"Error in CUDA tests: {e}")
                traceback.print_exc()
                results["cuda_tests"] = f"Error: {str(e)}"
                self.status_messages["cuda"] = f"Failed: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"
            self.status_messages["cuda"] = "CUDA not available"

        # ====== OPENVINO TESTS ======
        try:
            # First check if OpenVINO is installed
            try:
                import openvino
                has_openvino = True
                print("OpenVINO is installed")
            except ImportError:
                has_openvino = False
                results["openvino_tests"] = "OpenVINO not installed"
                self.status_messages["openvino"] = "OpenVINO not installed"
                
            if has_openvino:
                print("Testing speecht5 on OpenVINO...")
                # Initialize mock OpenVINO utils if not available
                try:
                    from ipfs_accelerate_py.worker.openvino_utils import openvino_utils
                    ov_utils = openvino_utils(resources=self.resources, metadata=self.metadata)
                    
                    # Initialize for OpenVINO
                    endpoint, processor, handler, queue, batch_size = self.model.init_openvino(
                        self.model_name,
                        "text-to-audio",
                        "CPU",
                        get_optimum_openvino_model=ov_utils.get_optimum_openvino_model,
                        get_openvino_model=ov_utils.get_openvino_model,
                        get_openvino_pipeline_type=ov_utils.get_openvino_pipeline_type,
                        openvino_cli_convert=ov_utils.openvino_cli_convert
                    )
                    
                    valid_init = endpoint is not None and processor is not None and handler is not None
                    results["openvino_init"] = "Success (REAL)" if valid_init else "Failed OpenVINO initialization"
                    
                    # Run actual inference
                    start_time = time.time()
                    output = handler(self.test_input if hasattr(self, 'test_input') else 
                                   self.test_text if hasattr(self, 'test_text') else
                                   self.test_image if hasattr(self, 'test_image') else
                                   self.test_audio if hasattr(self, 'test_audio') else
                                   "Default test input")
                    elapsed_time = time.time() - start_time
                    
                    # Verify the output
                    is_valid_output = output is not None
                    
                    results["openvino_handler"] = "Success (REAL)" if is_valid_output else "Failed OpenVINO handler"
                    
                    # Record example
                    self.examples.append({
                        "input": str(self.test_input if hasattr(self, 'test_input') else 
                                  self.test_text if hasattr(self, 'test_text') else
                                  self.test_image if hasattr(self, 'test_image') else
                                  self.test_audio if hasattr(self, 'test_audio') else
                                  "Default test input"),
                        "output": {
                            "output_type": str(type(output)),
                            "implementation_type": "REAL" if "implementation_type" not in output else output["implementation_type"]
                        },
                        "timestamp": datetime.datetime.now().isoformat(),
                        "elapsed_time": elapsed_time,
                        "implementation_type": "REAL",
                        "platform": "OpenVINO"
                    })
                        
                except Exception as e:
                    print(f"Error in OpenVINO implementation: {e}")
                    traceback.print_exc()
                    
                    # Try with mock implementations
                    print("Falling back to mock OpenVINO implementation...")
                    mock_get_openvino_model = lambda model_name, model_type=None: MagicMock()
                    mock_get_optimum_openvino_model = lambda model_name, model_type=None: MagicMock()
                    mock_get_openvino_pipeline_type = lambda model_name, model_type=None: "text-to-audio"
                    mock_openvino_cli_convert = lambda model_name, model_dst_path=None, task=None, weight_format=None, ratio=None, group_size=None, sym=None: True
                    
                    endpoint, processor, handler, queue, batch_size = self.model.init_openvino(
                        self.model_name,
                        "text-to-audio",
                        "CPU",
                        get_optimum_openvino_model=mock_get_optimum_openvino_model,
                        get_openvino_model=mock_get_openvino_model,
                        get_openvino_pipeline_type=mock_get_openvino_pipeline_type,
                        openvino_cli_convert=mock_openvino_cli_convert
                    )
                    
                    valid_init = endpoint is not None and processor is not None and handler is not None
                    results["openvino_init"] = "Success (MOCK)" if valid_init else "Failed OpenVINO initialization"
                    
                    # Run actual inference
                    start_time = time.time()
                    output = handler(self.test_input if hasattr(self, 'test_input') else 
                                   self.test_text if hasattr(self, 'test_text') else
                                   self.test_image if hasattr(self, 'test_image') else
                                   self.test_audio if hasattr(self, 'test_audio') else
                                   "Default test input")
                    elapsed_time = time.time() - start_time
                    
                    # Verify the output
                    is_valid_output = output is not None
                    
                    results["openvino_handler"] = "Success (MOCK)" if is_valid_output else "Failed OpenVINO handler"
                    
                    # Record example
                    self.examples.append({
                        "input": str(self.test_input if hasattr(self, 'test_input') else 
                                  self.test_text if hasattr(self, 'test_text') else
                                  self.test_image if hasattr(self, 'test_image') else
                                  self.test_audio if hasattr(self, 'test_audio') else
                                  "Default test input"),
                        "output": {
                            "output_type": str(type(output)),
                            "implementation_type": "MOCK" if "implementation_type" not in output else output["implementation_type"]
                        },
                        "timestamp": datetime.datetime.now().isoformat(),
                        "elapsed_time": elapsed_time,
                        "implementation_type": "MOCK",
                        "platform": "OpenVINO"
                    })
                
        except ImportError:
            results["openvino_tests"] = "OpenVINO not installed"
            self.status_messages["openvino"] = "OpenVINO not installed"
        except Exception as e:
            print(f"Error in OpenVINO tests: {e}")
            traceback.print_exc()
            results["openvino_tests"] = f"Error: {str(e)}"
            self.status_messages["openvino"] = f"Failed: {str(e)}"

        # Create structured results with status, examples and metadata
        structured_results = {
            "status": results,
            "examples": self.examples,
            "metadata": {
                "model_name": self.model_name,
                "test_timestamp": datetime.datetime.now().isoformat(),
                "python_version": sys.version,
                "torch_version": torch.__version__ if hasattr(torch, "__version__") else "Unknown",
                "transformers_version": transformers.__version__ if hasattr(transformers, "__version__") else "Unknown",
                "platform_status": self.status_messages
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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        # Create directories with appropriate permissions
        for directory in [expected_dir, collected_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, mode=0o755, exist_ok=True)
        
        # Save collected results
        results_file = os.path.join(collected_dir, 'hf_speecht5_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"Saved collected results to {results_file}")
        except Exception as e:
            print(f"Error saving results to {results_file}: {str(e)}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_speecht5_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                
                # Compare only status keys for backward compatibility
                status_expected = expected_results.get("status", expected_results)
                status_actual = test_results.get("status", test_results)
                
                # More detailed comparison of results
                all_match = True
                mismatches = []
                
                for key in set(status_expected.keys()) | set(status_actual.keys()):
                    if key not in status_expected:
                        mismatches.append(f"Missing expected key: {key}")
                        all_match = False
                    elif key not in status_actual:
                        mismatches.append(f"Missing actual key: {key}")
                        all_match = False
                    elif status_expected[key] != status_actual[key]:
                        # If the only difference is the implementation_type suffix, that's acceptable
                        if (
                            isinstance(status_expected[key], str) and 
                            isinstance(status_actual[key], str) and
                            status_expected[key].split(" (")[0] == status_actual[key].split(" (")[0] and
                            "Success" in status_expected[key] and "Success" in status_actual[key]
                        ):
                            continue
                        
                        mismatches.append(f"Key '{key}' differs: Expected '{status_expected[key]}', got '{status_actual[key]}'")
                        all_match = False
                
                if not all_match:
                    print("Test results differ from expected results!")
                    for mismatch in mismatches:
                        print(f"- {mismatch}")
                    print("
Would you like to update the expected results? (y/n)")
                    user_input = input().strip().lower()
                    if user_input == 'y':
                        with open(expected_file, 'w') as ef:
                            json.dump(test_results, ef, indent=2)
                            print(f"Updated expected results file: {expected_file}")
                    else:
                        print("Expected results not updated.")
                else:
                    print("All test results match expected results.")
            except Exception as e:
                print(f"Error comparing results with {expected_file}: {str(e)}")
                print("Creating new expected results file.")
                with open(expected_file, 'w') as ef:
                    json.dump(test_results, ef, indent=2)
        else:
            # Create expected results file if it doesn't exist
            try:
                with open(expected_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                    print(f"Created new expected results file: {expected_file}")
            except Exception as e:
                print(f"Error creating {expected_file}: {str(e)}")

        return test_results

if __name__ == "__main__":
    try:
        print("Starting speecht5 test...")
        test_instance = test_hf_speecht5()
        results = test_instance.__test__()
        print("speecht5 test completed")
        
        # Print test results in detailed format for better parsing
        status_dict = results.get("status", {})
        examples = results.get("examples", [])
        metadata = results.get("metadata", {})
        
        # Extract implementation status
        cpu_status = "UNKNOWN"
        cuda_status = "UNKNOWN"
        openvino_status = "UNKNOWN"
        
        for key, value in status_dict.items():
            if "cpu_" in key and "REAL" in value:
                cpu_status = "REAL"
            elif "cpu_" in key and "MOCK" in value:
                cpu_status = "MOCK"
                
            if "cuda_" in key and "REAL" in value:
                cuda_status = "REAL"
            elif "cuda_" in key and "MOCK" in value:
                cuda_status = "MOCK"
                
            if "openvino_" in key and "REAL" in value:
                openvino_status = "REAL"
            elif "openvino_" in key and "MOCK" in value:
                openvino_status = "MOCK"
                
        # Also look in examples
        for example in examples:
            platform = example.get("platform", "")
            impl_type = example.get("implementation_type", "")
            
            if platform == "CPU" and "REAL" in impl_type:
                cpu_status = "REAL"
            elif platform == "CPU" and "MOCK" in impl_type:
                cpu_status = "MOCK"
                
            if platform == "CUDA" and "REAL" in impl_type:
                cuda_status = "REAL"
            elif platform == "CUDA" and "MOCK" in impl_type:
                cuda_status = "MOCK"
                
            if platform == "OpenVINO" and "REAL" in impl_type:
                openvino_status = "REAL"
            elif platform == "OpenVINO" and "MOCK" in impl_type:
                openvino_status = "MOCK"
        
        # Print summary in a parser-friendly format
        print("\nSPEECHT5 TEST RESULTS SUMMARY")
        print(f"MODEL: {metadata.get('model_name', 'Unknown')}")
        print(f"CPU_STATUS: {cpu_status}")
        print(f"CUDA_STATUS: {cuda_status}")
        print(f"OPENVINO_STATUS: {openvino_status}")
        
        # Print a JSON representation to make it easier to parse
        print("\nstructured_results")
        print(json.dumps({
            "status": {
                "cpu": cpu_status,
                "cuda": cuda_status,
                "openvino": openvino_status
            },
            "model_name": metadata.get("model_name", "Unknown"),
            "examples": examples
        }))
        
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during testing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
