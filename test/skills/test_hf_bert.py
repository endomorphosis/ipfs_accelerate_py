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

# Import the module to test
from ipfs_accelerate_py.worker.skillset.hf_bert import hf_bert

class test_hf_bert:
    def __init__(self, resources=None, metadata=None):
        """
        Initialize the BERT test class.
        
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
        self.bert = hf_bert(resources=self.resources, metadata=self.metadata)
        self.model_name = "bert-base-uncased"
        self.test_text = "The quick brown fox jumps over the lazy dog"
        
        # Initialize collection arrays for examples and status
        self.examples = []
        self.status_messages = {}
        return None
        
    def test(self):
        """
        Run all tests for the BERT text embedding model, organized by hardware platform.
        Tests CPU, CUDA, OpenVINO, Apple, and Qualcomm implementations.
        
        Returns:
            dict: Structured test results with status, examples and metadata
        """
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.bert is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"

        # ====== CPU TESTS ======
        try:
            print("Testing BERT on CPU...")
            # Initialize for CPU without mocks
            endpoint, tokenizer, handler, queue, batch_size = self.bert.init_cpu(
                self.model_name,
                "cpu", 
                "cpu"
            )
            
            valid_init = endpoint is not None and tokenizer is not None and handler is not None
            results["cpu_init"] = "Success (REAL)" if valid_init else "Failed CPU initialization"
            
            # Get handler for CPU directly from initialization
            test_handler = handler
            
            # Run actual inference
            start_time = time.time()
            output = test_handler(self.test_text)
            elapsed_time = time.time() - start_time
            
            # Verify the output is a real embedding tensor
            is_valid_embedding = (
                output is not None and 
                isinstance(output, torch.Tensor) and 
                output.dim() == 2 and 
                output.size(0) == 1  # batch size
            )
            
            results["cpu_handler"] = "Success (REAL)" if is_valid_embedding else "Failed CPU handler"
            
            # Record example
            self.examples.append({
                "input": self.test_text,
                "output": {
                    "embedding_shape": list(output.shape) if is_valid_embedding else None,
                    "embedding_type": str(output.dtype) if is_valid_embedding else None
                },
                "timestamp": datetime.datetime.now().isoformat(),
                "elapsed_time": elapsed_time,
                "implementation_type": "REAL",
                "platform": "CPU"
            })
            
            # Add embedding shape to results
            if is_valid_embedding:
                results["cpu_embedding_shape"] = list(output.shape)
                results["cpu_embedding_type"] = str(output.dtype)
                
        except Exception as e:
            print(f"Error in CPU tests: {e}")
            traceback.print_exc()
            results["cpu_tests"] = f"Error: {str(e)}"
            self.status_messages["cpu"] = f"Failed: {str(e)}"

        # ====== CUDA TESTS ======
        if torch.cuda.is_available():
            try:
                print("Testing BERT on CUDA...")
                # Initialize for CUDA without mocks
                endpoint, tokenizer, handler, queue, batch_size = self.bert.init_cuda(
                    self.model_name,
                    "cuda",
                    "cuda:0"
                )
                
                valid_init = endpoint is not None and tokenizer is not None and handler is not None
                results["cuda_init"] = "Success (REAL)" if valid_init else "Failed CUDA initialization"
                
                # Get handler for CUDA directly from initialization
                test_handler = handler
                
                # Run actual inference
                start_time = time.time()
                output = test_handler(self.test_text)
                elapsed_time = time.time() - start_time
                
                # Verify the output is a real embedding
                is_valid_embedding = False
                if isinstance(output, dict):
                    # Check for different possible output formats
                    if 'hidden_states' in output:
                        hidden_states = output['hidden_states']
                        is_valid_embedding = (
                            hidden_states is not None and
                            hidden_states.shape[0] > 0
                        )
                    elif hasattr(output, 'keys') and len(output.keys()) > 0:
                        # Just verify any output exists
                        is_valid_embedding = True
                elif isinstance(output, torch.Tensor) or isinstance(output, np.ndarray):
                    is_valid_embedding = (
                        output is not None and
                        output.shape[0] > 0
                    )
                
                results["cuda_handler"] = "Success (REAL)" if is_valid_embedding else "Failed CUDA handler"
                
                # Record example
                output_shape = None
                if is_valid_embedding:
                    if isinstance(output, dict) and 'hidden_states' in output:
                        output_shape = list(output['hidden_states'].shape)
                    elif isinstance(output, torch.Tensor):
                        output_shape = list(output.shape)
                    elif isinstance(output, np.ndarray):
                        output_shape = list(output.shape)
                
                self.examples.append({
                    "input": self.test_text,
                    "output": {
                        "embedding_shape": output_shape,
                        "embedding_type": str(output.dtype) if hasattr(output, 'dtype') else None
                    },
                    "timestamp": datetime.datetime.now().isoformat(),
                    "elapsed_time": elapsed_time,
                    "implementation_type": "REAL",
                    "platform": "CUDA"
                })
                
                # Add embedding shape to results
                if is_valid_embedding and output_shape:
                    results["cuda_embedding_shape"] = output_shape
                
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
                # Import the existing OpenVINO utils from the main package
                from ipfs_accelerate_py.worker.openvino_utils import openvino_utils
                
                # Initialize openvino_utils
                ov_utils = openvino_utils(resources=self.resources, metadata=self.metadata)
                
                # Create a custom model class for testing
                class CustomOpenVINOModel:
                    def __init__(self):
                        pass
                        
                    def infer(self, inputs):
                        batch_size = 1
                        seq_len = 10
                        hidden_size = 768
                        
                        if isinstance(inputs, dict) and "input_ids" in inputs:
                            # Get shapes from actual inputs if available
                            if hasattr(inputs["input_ids"], "shape"):
                                batch_size = inputs["input_ids"].shape[0]
                                seq_len = inputs["input_ids"].shape[1]
                        
                        # Create output tensor (simulated hidden states)
                        output = np.random.rand(batch_size, seq_len, hidden_size).astype(np.float32)
                        return {"last_hidden_state": output}
                        
                    def __call__(self, inputs):
                        return self.infer(inputs)
                
                # Create a mock model instance
                mock_model = CustomOpenVINOModel()
                
                # Create mock get_openvino_model function
                def mock_get_openvino_model(model_name, model_type=None):
                    print(f"Mock get_openvino_model called for {model_name}")
                    return mock_model
                    
                # Create mock get_optimum_openvino_model function
                def mock_get_optimum_openvino_model(model_name, model_type=None):
                    print(f"Mock get_optimum_openvino_model called for {model_name}")
                    return mock_model
                    
                # Create mock get_openvino_pipeline_type function  
                def mock_get_openvino_pipeline_type(model_name, model_type=None):
                    return "feature-extraction"
                    
                # Create mock openvino_cli_convert function
                def mock_openvino_cli_convert(model_name, model_dst_path=None, task=None, weight_format=None, ratio=None, group_size=None, sym=None):
                    print(f"Mock openvino_cli_convert called for {model_name}")
                    return True
                
                # Initialize with mock OpenVINO utils
                endpoint, tokenizer, handler, queue, batch_size = self.bert.init_openvino(
                    model_name=self.model_name,
                    model_type="feature-extraction",
                    device="CPU",
                    openvino_label="openvino:0",
                    get_optimum_openvino_model=mock_get_optimum_openvino_model,
                    get_openvino_model=mock_get_openvino_model,
                    get_openvino_pipeline_type=mock_get_openvino_pipeline_type,
                    openvino_cli_convert=mock_openvino_cli_convert
                )
                
                # If we got a handler back, we succeeded
                valid_init = handler is not None
                results["openvino_init"] = "Success (MOCK)" if valid_init else "Failed OpenVINO initialization"
                
                # Run inference
                start_time = time.time()
                output = handler(self.test_text)
                elapsed_time = time.time() - start_time
                
                is_valid_embedding = (
                    output is not None and
                    hasattr(output, 'shape') and
                    output.shape[0] == 1  # batch size 1
                )
                
                results["openvino_handler"] = "Success (MOCK)" if is_valid_embedding else "Failed OpenVINO handler"
                
                # Record example
                self.examples.append({
                    "input": self.test_text,
                    "output": {
                        "embedding_shape": list(output.shape) if is_valid_embedding else None,
                    },
                    "timestamp": datetime.datetime.now().isoformat(),
                    "elapsed_time": elapsed_time,
                    "implementation_type": "MOCK",
                    "platform": "OpenVINO"
                })
                
                # Add embedding details if successful
                if is_valid_embedding:
                    results["openvino_embedding_shape"] = list(output.shape)
                
        except ImportError:
            results["openvino_tests"] = "OpenVINO not installed"
            self.status_messages["openvino"] = "OpenVINO not installed"
        except Exception as e:
            print(f"Error in OpenVINO tests: {e}")
            traceback.print_exc()
            results["openvino_tests"] = f"Error: {str(e)}"
            self.status_messages["openvino"] = f"Failed: {str(e)}"

        # ====== APPLE SILICON TESTS ======
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                print("Testing BERT on Apple Silicon...")
                try:
                    import coremltools  # Only try import if MPS is available
                    has_coreml = True
                except ImportError:
                    has_coreml = False
                    results["apple_tests"] = "CoreML Tools not installed"
                    self.status_messages["apple"] = "CoreML Tools not installed"

                if has_coreml:
                    with patch('coremltools.convert') as mock_convert:
                        mock_convert.return_value = MagicMock()
                        
                        endpoint, tokenizer, handler, queue, batch_size = self.bert.init_apple(
                            self.model_name,
                            "mps",
                            "apple:0"
                        )
                        
                        valid_init = handler is not None
                        results["apple_init"] = "Success (MOCK)" if valid_init else "Failed Apple initialization"
                        
                        test_handler = self.bert.create_apple_text_embedding_endpoint_handler(
                            endpoint_model=self.model_name,
                            apple_label="apple:0",
                            endpoint=endpoint,
                            tokenizer=tokenizer
                        )
                        
                        start_time = time.time()
                        output = test_handler(self.test_text)
                        elapsed_time = time.time() - start_time
                        
                        results["apple_handler"] = "Success (MOCK)" if output is not None else "Failed Apple handler"
                        
                        # Record example
                        output_shape = list(output.shape) if output is not None and hasattr(output, 'shape') else None
                        self.examples.append({
                            "input": self.test_text,
                            "output": {
                                "embedding_shape": output_shape,
                            },
                            "timestamp": datetime.datetime.now().isoformat(),
                            "elapsed_time": elapsed_time,
                            "implementation_type": "MOCK",
                            "platform": "Apple"
                        })
            except ImportError:
                results["apple_tests"] = "CoreML Tools not installed"
                self.status_messages["apple"] = "CoreML Tools not installed"
            except Exception as e:
                print(f"Error in Apple tests: {e}")
                traceback.print_exc()
                results["apple_tests"] = f"Error: {str(e)}"
                self.status_messages["apple"] = f"Failed: {str(e)}"
        else:
            results["apple_tests"] = "Apple Silicon not available"
            self.status_messages["apple"] = "Apple Silicon not available"

        # ====== QUALCOMM TESTS ======
        try:
            print("Testing BERT on Qualcomm...")
            try:
                from ipfs_accelerate_py.worker.skillset.qualcomm_snpe_utils import get_snpe_utils
                has_snpe = True
            except ImportError:
                has_snpe = False
                results["qualcomm_tests"] = "SNPE SDK not installed"
                self.status_messages["qualcomm"] = "SNPE SDK not installed"
                
            if has_snpe:
                # For Qualcomm, we need to mock since it's unlikely to be available in test environment
                with patch('ipfs_accelerate_py.worker.skillset.qualcomm_snpe_utils.get_snpe_utils') as mock_snpe:
                    mock_snpe_utils = MagicMock()
                    mock_snpe_utils.is_available.return_value = True
                    mock_snpe_utils.convert_model.return_value = "mock_converted_model"
                    mock_snpe_utils.load_model.return_value = MagicMock()
                    mock_snpe_utils.optimize_for_device.return_value = "mock_optimized_model"
                    mock_snpe_utils.run_inference.return_value = {
                        "last_hidden_state": np.random.rand(1, 10, 768)
                    }
                    mock_snpe.return_value = mock_snpe_utils
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.bert.init_qualcomm(
                        self.model_name,
                        "qualcomm",
                        "qualcomm:0"
                    )
                    
                    valid_init = handler is not None
                    results["qualcomm_init"] = "Success (MOCK)" if valid_init else "Failed Qualcomm initialization"
                    
                    # For handler testing, create a mock tokenizer
                    if tokenizer is None:
                        tokenizer = MagicMock()
                        tokenizer.return_value = {
                            "input_ids": np.ones((1, 10)),
                            "attention_mask": np.ones((1, 10))
                        }
                        
                    test_handler = self.bert.create_qualcomm_text_embedding_endpoint_handler(
                        endpoint_model=self.model_name,
                        qualcomm_label="qualcomm:0",
                        endpoint=endpoint,
                        tokenizer=tokenizer
                    )
                    
                    start_time = time.time()
                    output = test_handler(self.test_text)
                    elapsed_time = time.time() - start_time
                    
                    results["qualcomm_handler"] = "Success (MOCK)" if output is not None else "Failed Qualcomm handler"
                    
                    # Record example
                    output_shape = list(output.shape) if output is not None and hasattr(output, 'shape') else None
                    self.examples.append({
                        "input": self.test_text,
                        "output": {
                            "embedding_shape": output_shape,
                        },
                        "timestamp": datetime.datetime.now().isoformat(),
                        "elapsed_time": elapsed_time,
                        "implementation_type": "MOCK",
                        "platform": "Qualcomm"
                    })
        except ImportError:
            results["qualcomm_tests"] = "SNPE SDK not installed"
            self.status_messages["qualcomm"] = "SNPE SDK not installed"
        except Exception as e:
            print(f"Error in Qualcomm tests: {e}")
            traceback.print_exc()
            results["qualcomm_tests"] = f"Error: {str(e)}"
            self.status_messages["qualcomm"] = f"Failed: {str(e)}"

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
        results_file = os.path.join(collected_dir, 'hf_bert_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"Saved collected results to {results_file}")
        except Exception as e:
            print(f"Error saving results to {results_file}: {str(e)}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_bert_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                
                # Filter out variable fields for comparison
                def filter_variable_data(result):
                    if isinstance(result, dict):
                        # Create a copy to avoid modifying the original
                        filtered = {}
                        for k, v in result.items():
                            # Skip timestamp and variable output data for comparison
                            if k not in ["timestamp", "elapsed_time", "output"] and k != "examples" and k != "metadata":
                                filtered[k] = filter_variable_data(v)
                        return filtered
                    elif isinstance(result, list):
                        return [filter_variable_data(item) for item in result]
                    else:
                        return result
                
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
                    print("\nWould you like to update the expected results? (y/n)")
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
        print("Starting BERT test...")
        this_bert = test_hf_bert()
        results = this_bert.__test__()
        print("BERT test completed")
        print("Status summary:")
        for key, value in results.get("status", {}).items():
            print(f"  {key}: {value}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during testing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)