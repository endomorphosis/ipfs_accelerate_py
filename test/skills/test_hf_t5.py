import os
import sys
import json
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
            
            # Initialize for CPU without mocks
            endpoint, tokenizer, handler, queue, batch_size = self.t5.init_cpu(
                self.model_name,
                "cpu",
                "cpu"
            )
            
            valid_init = endpoint is not None and tokenizer is not None and handler is not None
            results["cpu_init"] = "Success" if valid_init else "Failed CPU initialization"
            
            # Use handler directly from initialization
            test_handler = handler
            
            # Test text generation
            print(f"Testing T5 generation with input: '{self.test_input}'")
            output = test_handler(self.test_input)
            
            # Verify output
            is_valid_output = output is not None
            results["cpu_handler"] = "Success" if is_valid_output else "Failed CPU handler"
            
            # Add output information if available
            if is_valid_output:
                if isinstance(output, str):
                    # Truncate long outputs for readability
                    if len(output) > 100:
                        results["cpu_output"] = output[:100] + "..."
                    else:
                        results["cpu_output"] = output
                        
                    results["cpu_output_length"] = len(output)
                else:
                    results["cpu_output_type"] = str(type(output))
                    if hasattr(output, "__len__"):
                        results["cpu_output_length"] = len(output)
                
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
                    results["cuda_init"] = "Success" if valid_init else "Failed CUDA initialization"
                    
                    test_handler = self.t5.create_cuda_t5_endpoint_handler(
                        tokenizer,
                        self.model_name,
                        "cuda:0",
                        endpoint
                    )
                    
                    output = test_handler(self.test_input)
                    results["cuda_handler"] = "Success" if output is not None else "Failed CUDA handler"
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
                results["openvino_init"] = "Success" if valid_init else "Failed OpenVINO initialization"
                
                test_handler = self.t5.create_openvino_t5_endpoint_handler(
                    endpoint,
                    tokenizer,
                    self.model_name,
                    "openvino:0"
                )
                
                output = test_handler(self.test_input)
                results["openvino_handler"] = "Success" if output is not None else "Failed OpenVINO handler"
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
                    results["apple_init"] = "Success" if valid_init else "Failed Apple initialization"
                    
                    test_handler = self.t5.create_apple_t5_endpoint_handler(
                        endpoint,
                        tokenizer,
                        self.model_name,
                        "apple:0"
                    )
                    
                    output = test_handler(self.test_input)
                    results["apple_handler"] = "Success" if output is not None else "Failed Apple handler"
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
                
                endpoint, tokenizer, handler, queue, batch_size = self.t5.init_qualcomm(
                    self.model_name,
                    "qualcomm",
                    "qualcomm:0"
                )
                
                valid_init = handler is not None
                results["qualcomm_init"] = "Success" if valid_init else "Failed Qualcomm initialization"
                
                test_handler = self.t5.create_qualcomm_t5_endpoint_handler(
                    tokenizer,
                    self.model_name,
                    "qualcomm:0",
                    endpoint
                )
                
                output = test_handler(self.test_input)
                results["qualcomm_handler"] = "Success" if output is not None else "Failed Qualcomm handler"
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
        
        # Save collected results
        results_file = os.path.join(collected_dir, 'hf_t5_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
        except Exception as e:
            print(f"Error saving results to {results_file}: {str(e)}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_t5_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                    if expected_results != test_results:
                        print("Test results differ from expected results!")
                        print(f"Expected: {json.dumps(expected_results, indent=2)}")
                        print(f"Got: {json.dumps(test_results, indent=2)}")
            except Exception as e:
                print(f"Error comparing results with {expected_file}: {str(e)}")
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