import os
import sys
import json
import time
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import transformers

# Use direct import with the absolute path
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")
from ipfs_accelerate_py.worker.skillset.hf_t5 import hf_t5

class test_hf_t5:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np,
            "transformers": transformers  # Use real transformers if available
        }
        self.metadata = metadata if metadata else {}
        self.t5 = hf_t5(resources=self.resources, metadata=self.metadata)
        self.model_name = "t5-small"
        self.test_input = "translate English to French: The quick brown fox jumps over the lazy dog"
        return None

    def test(self):
        """Run all tests for the T5 language model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.t5 is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"

        # Test CPU initialization and handler with real inference
        try:
            print("Initializing T5 for CPU...")
            
            # Check if we're using real transformers
            transformers_available = "transformers" in sys.modules
            implementation_type = "(REAL)" if transformers_available else "(MOCK)"
            
            # Initialize for CPU without mocks
            endpoint, tokenizer, handler, queue, batch_size = self.t5.init_cpu(
                self.model_name,
                "cpu",
                "cpu"
            )
            
            valid_init = endpoint is not None and tokenizer is not None and handler is not None
            results["cpu_init"] = f"Success {implementation_type}" if valid_init else "Failed CPU initialization"
            
            # Use handler directly from initialization
            test_handler = handler
            
            # Test text generation
            print(f"Testing T5 generation with input: '{self.test_input}'")
            output = test_handler(self.test_input)
            
            # Verify output
            is_valid_output = output is not None
            results["cpu_handler"] = f"Success {implementation_type}" if is_valid_output else "Failed CPU handler"
            
            # Add output information if available
            if is_valid_output:
                if isinstance(output, str):
                    # Truncate long outputs for readability
                    if len(output) > 100:
                        results["cpu_output"] = output[:100] + "..."
                    else:
                        results["cpu_output"] = output
                        
                    results["cpu_output_length"] = len(output)
                    results["cpu_output_timestamp"] = time.time()
                    results["cpu_output_implementation"] = implementation_type
                else:
                    results["cpu_output_type"] = str(type(output))
                    if hasattr(output, "__len__"):
                        results["cpu_output_length"] = len(output)
                
                # Save result to demonstrate working implementation
                results["cpu_output_example"] = {
                    "input": self.test_input,
                    "output": output[:100] + "..." if isinstance(output, str) and len(output) > 100 else output,
                    "timestamp": time.time(),
                    "implementation": implementation_type
                }
                
        except Exception as e:
            results["cpu_tests"] = f"Error: {str(e)}"

        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                with patch('transformers.T5Tokenizer.from_pretrained') as mock_tokenizer, \
                     patch('transformers.T5ForConditionalGeneration.from_pretrained') as mock_model:
                    
                    mock_tokenizer.return_value = MagicMock()
                    mock_model.return_value = MagicMock()
                    mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3]])
                    mock_tokenizer.batch_decode.return_value = ["Le renard brun rapide saute par-dessus le chien paresseux"]
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.t5.init_cuda(
                        self.model_name,
                        "cuda",
                        "cuda:0"
                    )
                    
                    valid_init = endpoint is not None and tokenizer is not None and handler is not None
                    results["cuda_init"] = "Success (MOCK)" if valid_init else "Failed CUDA initialization"
                    
                    test_handler = self.t5.create_cuda_t5_endpoint_handler(
                        tokenizer,
                        self.model_name,
                        "cuda:0",
                        endpoint
                    )
                    
                    output = test_handler(self.test_input)
                    results["cuda_handler"] = "Success (MOCK)" if output is not None else "Failed CUDA handler"
                    
                    # Include sample output for verification
                    if output is not None:
                        if isinstance(output, str):
                            if len(output) > 100:
                                results["cuda_output"] = output[:100] + "..."
                            else:
                                results["cuda_output"] = output
                            results["cuda_output_length"] = len(output)
                            results["cuda_timestamp"] = time.time()
                        
                        # Save result to demonstrate working implementation
                        results["cuda_output_example"] = {
                            "input": self.test_input,
                            "output": output[:100] + "..." if isinstance(output, str) and len(output) > 100 else output,
                            "timestamp": time.time(),
                            "implementation": "(MOCK)"
                        }
            except Exception as e:
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"

        # Test OpenVINO if installed
        try:
            import openvino
            # Import the existing OpenVINO utils from the main package
            from ipfs_accelerate_py.worker.openvino_utils import openvino_utils
            
            # Initialize openvino_utils
            ov_utils = openvino_utils(resources=self.resources, metadata=self.metadata)
            
            # Use a patched version for testing
            with patch('openvino.runtime.Core' if hasattr(openvino, 'runtime') and hasattr(openvino.runtime, 'Core') else 'openvino.Core'):
                endpoint, tokenizer, handler, queue, batch_size = self.t5.init_openvino(
                    self.model_name,
                    "text2text-generation",
                    "CPU",
                    "openvino:0",
                    ov_utils.get_optimum_openvino_model,
                    ov_utils.get_openvino_model,
                    ov_utils.get_openvino_pipeline_type,
                    ov_utils.openvino_cli_convert
                )
                
                valid_init = handler is not None
                results["openvino_init"] = "Success (MOCK)" if valid_init else "Failed OpenVINO initialization"
                
                test_handler = self.t5.create_openvino_t5_endpoint_handler(
                    endpoint,
                    tokenizer,
                    self.model_name,
                    "openvino:0"
                )
                
                output = test_handler(self.test_input)
                results["openvino_handler"] = "Success (MOCK)" if output is not None else "Failed OpenVINO handler"
                
                # Include sample output for verification
                if output is not None:
                    if isinstance(output, str):
                        if len(output) > 100:
                            results["openvino_output"] = output[:100] + "..."
                        else:
                            results["openvino_output"] = output
                        results["openvino_output_length"] = len(output)
                        results["openvino_timestamp"] = time.time()
                    
                    # Save result to demonstrate working implementation
                    results["openvino_output_example"] = {
                        "input": self.test_input,
                        "output": output[:100] + "..." if isinstance(output, str) and len(output) > 100 else output,
                        "timestamp": time.time(),
                        "implementation": "(MOCK)"
                    }
        except ImportError:
            results["openvino_tests"] = "OpenVINO not installed"
        except Exception as e:
            results["openvino_tests"] = f"Error: {str(e)}"

        # Test Apple Silicon if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                import coremltools
                with patch('coremltools.convert') as mock_convert:
                    mock_convert.return_value = MagicMock()
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.t5.init_apple(
                        self.model_name,
                        "mps",
                        "apple:0"
                    )
                    
                    valid_init = handler is not None
                    results["apple_init"] = "Success (MOCK)" if valid_init else "Failed Apple initialization"
                    
                    test_handler = self.t5.create_apple_t5_endpoint_handler(
                        endpoint,
                        tokenizer,
                        self.model_name,
                        "apple:0"
                    )
                    
                    output = test_handler(self.test_input)
                    results["apple_handler"] = "Success (MOCK)" if output is not None else "Failed Apple handler"
                    
                    # Include sample output for verification
                    if output is not None:
                        if isinstance(output, str):
                            if len(output) > 100:
                                results["apple_output"] = output[:100] + "..."
                            else:
                                results["apple_output"] = output
                            results["apple_output_length"] = len(output)
                            results["apple_timestamp"] = time.time()
                        
                        # Save result to demonstrate working implementation
                        results["apple_output_example"] = {
                            "input": self.test_input,
                            "output": output[:100] + "..." if isinstance(output, str) and len(output) > 100 else output,
                            "timestamp": time.time(),
                            "implementation": "(MOCK)"
                        }
            except ImportError:
                results["apple_tests"] = "CoreML Tools not installed"
            except Exception as e:
                results["apple_tests"] = f"Error: {str(e)}"
        else:
            results["apple_tests"] = "Apple Silicon not available"

        # Test Qualcomm if available
        try:
            with patch('ipfs_accelerate_py.worker.skillset.qualcomm_snpe_utils.get_snpe_utils') as mock_snpe:
                mock_snpe.return_value = MagicMock()
                
                # Initialize Qualcomm backend
                endpoint, tokenizer, handler, queue, batch_size = self.t5.init_qualcomm(
                    self.model_name,
                    "qualcomm",
                    "qualcomm:0"
                )
                
                # Check if initialization succeeded
                valid_init = endpoint is not None and tokenizer is not None and handler is not None
                results["qualcomm_init"] = "Success (MOCK)" if valid_init else "Failed Qualcomm initialization"
                
                # Only proceed with testing the handler if initialization was successful
                if valid_init:
                    # Create the handler if it wasn't returned from init
                    if handler is None:
                        test_handler = self.t5.create_qualcomm_t5_endpoint_handler(
                            tokenizer,
                            self.model_name,
                            "qualcomm:0",
                            endpoint
                        )
                    else:
                        test_handler = handler
                    
                    # Test the handler
                    try:
                        output = test_handler(self.test_input)
                        results["qualcomm_handler"] = "Success (MOCK)" if output is not None else "Failed Qualcomm handler"
                        
                        # Include sample output for verification
                        if output is not None:
                            if isinstance(output, str):
                                if len(output) > 100:
                                    results["qualcomm_output"] = output[:100] + "..."
                                else:
                                    results["qualcomm_output"] = output
                                results["qualcomm_output_length"] = len(output)
                                results["qualcomm_timestamp"] = time.time()
                            
                            # Save result to demonstrate working implementation
                            results["qualcomm_output_example"] = {
                                "input": self.test_input,
                                "output": output[:100] + "..." if isinstance(output, str) and len(output) > 100 else output,
                                "timestamp": time.time(),
                                "implementation": "(MOCK)"
                            }
                    except Exception as e:
                        results["qualcomm_handler"] = f"Failed (MOCK): {str(e)}"
                        results["qualcomm_output_example"] = {
                            "input": self.test_input,
                            "error": str(e),
                            "timestamp": time.time(),
                            "implementation": "(MOCK)"
                        }
                else:
                    # If initialization failed, don't try to test the handler
                    results["qualcomm_handler"] = "Skipped due to failed initialization"
        except ImportError:
            results["qualcomm_tests"] = "SNPE SDK not installed"
        except Exception as e:
            results["qualcomm_tests"] = f"Error: {str(e)}"

        return results

    def __test__(self):
        """Run tests and compare/save results"""
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {"test_error": str(e)}
        
        # Create directories if they don't exist
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        # Create directories with appropriate permissions
        for directory in [expected_dir, collected_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, mode=0o755, exist_ok=True)
        
        # Add metadata about the environment to the results
        test_results["metadata"] = {
            "timestamp": time.time(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "transformers_version": transformers.__version__ if hasattr(transformers, "__version__") else "mocked",
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "transformers_mocked": isinstance(self.resources["transformers"], MagicMock),
            "test_model": self.model_name,
            "test_run_id": f"t5-test-{int(time.time())}"
        }
        
        # Save collected results
        results_file = os.path.join(collected_dir, 'hf_t5_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"Saved test results to {results_file}")
        except Exception as e:
            print(f"Error saving results to {results_file}: {str(e)}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_t5_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                    
                    # Only compare the non-variable parts 
                    excluded_keys = ["metadata", "cpu_output", "cuda_output", "openvino_output", 
                                    "apple_output", "qualcomm_output", "cpu_output_example",
                                    "cuda_output_example", "openvino_output_example", 
                                    "apple_output_example", "qualcomm_output_example"]
                    
                    # Also exclude timestamp fields
                    timestamp_keys = [k for k in test_results.keys() if "timestamp" in k]
                    excluded_keys.extend(timestamp_keys)
                    
                    expected_copy = {k: v for k, v in expected_results.items() if k not in excluded_keys}
                    results_copy = {k: v for k, v in test_results.items() if k not in excluded_keys}
                    
                    mismatches = []
                    for key in set(expected_copy.keys()) | set(results_copy.keys()):
                        if key not in expected_copy:
                            mismatches.append(f"Key '{key}' missing from expected results")
                        elif key not in results_copy:
                            mismatches.append(f"Key '{key}' missing from current results")
                        elif expected_copy[key] != results_copy[key]:
                            mismatches.append(f"Key '{key}' differs: Expected '{expected_copy[key]}', got '{results_copy[key]}'")
                    
                    if mismatches:
                        print("Test results differ from expected results!")
                        for mismatch in mismatches:
                            print(f"- {mismatch}")
                        
                        print("\nConsider updating the expected results file if these differences are intentional.")
                        
                        # Option to update expected results
                        if input("Update expected results? (y/n): ").lower() == 'y':
                            with open(expected_file, 'w') as f:
                                json.dump(test_results, f, indent=2)
                                print(f"Updated expected results file: {expected_file}")
                    else:
                        print("Core test results match expected results (excluding variable outputs)")
            except Exception as e:
                print(f"Error comparing results with {expected_file}: {str(e)}")
                if input("Create/update expected results? (y/n): ").lower() == 'y':
                    with open(expected_file, 'w') as f:
                        json.dump(test_results, f, indent=2)
                        print(f"Updated expected results file: {expected_file}")
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
        this_t5 = test_hf_t5()
        results = this_t5.__test__()
        print(f"T5 Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)