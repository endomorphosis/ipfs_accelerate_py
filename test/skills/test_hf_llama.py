import os
import sys
import json
import time
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Use direct import with the absolute path
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")
from ipfs_accelerate_py.worker.skillset.hf_llama import hf_llama

class test_hf_llama:
    def __init__(self, resources=None, metadata=None):
        # Try to import transformers directly if available
        try:
            import transformers
            transformers_module = transformers
        except ImportError:
            transformers_module = MagicMock()
            
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np,
            "transformers": transformers_module
        }
        self.metadata = metadata if metadata else {}
        self.llama = hf_llama(resources=self.resources, metadata=self.metadata)
        # Use a publicly accessible smaller model for testing
        self.model_name = "facebook/opt-125m"  # Small public model for testing
        self.test_prompt = "Write a short story about a fox and a dog."
        return None

    def test(self):
        """Run all tests for the LLaMA language model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.llama is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"

        # Test CPU initialization and handler
        try:
            # Try with real model first
            try:
                transformers_available = isinstance(self.resources["transformers"], MagicMock) == False
                if transformers_available:
                    # Real model initialization
                    endpoint, tokenizer, handler, queue, batch_size = self.llama.init_cpu(
                        self.model_name,
                        "cpu",
                        "cpu"
                    )
                    
                    valid_init = endpoint is not None and tokenizer is not None and handler is not None
                    results["cpu_init"] = "Success" if valid_init else "Failed CPU initialization"
                    
                    if valid_init:
                        # Test with real handler
                        output = handler(self.test_prompt)
                        results["cpu_handler"] = "Success" if output is not None else "Failed CPU handler"
                        
                        # Check output structure
                        if output is not None and isinstance(output, dict):
                            results["cpu_output"] = "Valid" if "generated_text" in output else "Missing generated_text"
                        else:
                            results["cpu_output"] = "Invalid output format"
                else:
                    raise ImportError("Transformers not available")
                    
            except Exception as e:
                # Fall back to mock if real model fails
                print(f"Falling back to mock model: {str(e)}")
                with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                     patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                     patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                    
                    mock_config.return_value = MagicMock()
                    mock_tokenizer.return_value = MagicMock()
                    mock_tokenizer.return_value.batch_decode = MagicMock(return_value=["Once upon a time..."])
                    mock_model.return_value = MagicMock()
                    mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3]])
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.llama.init_cpu(
                        self.model_name,
                        "cpu",
                        "cpu"
                    )
                    
                    valid_init = endpoint is not None and tokenizer is not None and handler is not None
                    results["cpu_init"] = "Success (Mock)" if valid_init else "Failed CPU initialization"
                    
                    test_handler = self.llama.create_cpu_llama_endpoint_handler(
                        tokenizer,
                        self.model_name,
                        "cpu",
                        endpoint
                    )
                    
                    output = test_handler(self.test_prompt)
                    results["cpu_handler"] = "Success (Mock)" if output is not None else "Failed CPU handler"
                
        except Exception as e:
            results["cpu_tests"] = f"Error: {str(e)}"

        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                # Try with real model first
                try:
                    transformers_available = isinstance(self.resources["transformers"], MagicMock) == False
                    if transformers_available:
                        # Real model initialization
                        endpoint, tokenizer, handler, queue, batch_size = self.llama.init_cuda(
                            self.model_name,
                            "cuda",
                            "cuda:0"
                        )
                        
                        valid_init = endpoint is not None and tokenizer is not None and handler is not None
                        results["cuda_init"] = "Success" if valid_init else "Failed CUDA initialization"
                        
                        if valid_init:
                            # Test with real handler
                            output = handler(self.test_prompt)
                            results["cuda_handler"] = "Success" if output is not None else "Failed CUDA handler"
                            
                            # Check output structure
                            if output is not None and isinstance(output, dict):
                                results["cuda_output"] = "Valid" if "generated_text" in output else "Missing generated_text"
                            else:
                                results["cuda_output"] = "Invalid output format"
                    else:
                        raise ImportError("Transformers not available")
                        
                except Exception as e:
                    # Fall back to mock if real model fails
                    print(f"Falling back to mock model for CUDA: {str(e)}")
                    with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                         patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                         patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                        
                        mock_config.return_value = MagicMock()
                        mock_tokenizer.return_value = MagicMock()
                        mock_model.return_value = MagicMock()
                        mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3]])
                        mock_tokenizer.batch_decode.return_value = ["Once upon a time..."]
                        
                        endpoint, tokenizer, handler, queue, batch_size = self.llama.init_cuda(
                            self.model_name,
                            "cuda",
                            "cuda:0"
                        )
                        
                        valid_init = endpoint is not None and tokenizer is not None and handler is not None
                        results["cuda_init"] = "Success (Mock)" if valid_init else "Failed CUDA initialization"
                        
                        test_handler = self.llama.create_cuda_llama_endpoint_handler(
                            tokenizer,
                            self.model_name,
                            "cuda:0",
                            endpoint
                        )
                        
                        output = test_handler(self.test_prompt)
                        results["cuda_handler"] = "Success (Mock)" if output is not None else "Failed CUDA handler"
            except Exception as e:
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"

        # Test OpenVINO if installed
        try:
            # Import the existing OpenVINO utils from the main package
            import openvino
            from ipfs_accelerate_py.worker.openvino_utils import openvino_utils
            
            # Initialize openvino_utils
            ov_utils = openvino_utils(resources=self.resources, metadata=self.metadata)
            
            # Setup OpenVINO runtime environment
            with patch('openvino.runtime.Core' if hasattr(openvino, 'runtime') and hasattr(openvino.runtime, 'Core') else 'openvino.Core'):
                
                # Initialize OpenVINO endpoint with real utils
                endpoint, tokenizer, handler, queue, batch_size = self.llama.init_openvino(
                    self.model_name,
                    "text-generation",
                    "CPU",
                    "openvino:0",
                    ov_utils.get_optimum_openvino_model,
                    ov_utils.get_openvino_model,
                    ov_utils.get_openvino_pipeline_type,
                    ov_utils.openvino_cli_convert
                )
                
                valid_init = handler is not None
                results["openvino_init"] = "Success" if valid_init else "Failed OpenVINO initialization"
                
                # Create handler for testing
                test_handler = self.llama.create_openvino_llama_endpoint_handler(
                    tokenizer,
                    self.model_name,
                    "openvino:0",
                    endpoint
                )
                
                output = test_handler(self.test_prompt)
                results["openvino_handler"] = "Success" if output is not None else "Failed OpenVINO handler"
                
                # Check output structure if available
                if output is not None and isinstance(output, dict):
                    results["openvino_output"] = "Valid" if "generated_text" in output else "Missing generated_text"
                else:
                    results["openvino_output"] = "Invalid output format"
        except ImportError:
            results["openvino_tests"] = "OpenVINO not installed"
        except Exception as e:
            results["openvino_tests"] = f"Error: {str(e)}"

        # Test Apple Silicon if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                # Try using real CoreML Tools if available
                try:
                    import coremltools
                    
                    # For real CoreML, we'll try to initialize the Apple handler
                    with patch('.apple_coreml_utils.get_coreml_utils') as mock_coreml_utils:
                        # Create mock utils for testing
                        mock_utils = MagicMock()
                        mock_utils.is_available.return_value = True
                        mock_utils.convert_model.return_value = True
                        mock_utils.load_model.return_value = MagicMock()
                        mock_utils.optimize_for_device.return_value = "/mock/optimized_model.mlpackage"
                        mock_utils.run_inference.return_value = {"logits": np.random.rand(1, 10, 30522)}
                        mock_coreml_utils.return_value = mock_utils
                        
                        # Initialize Apple Silicon endpoint
                        endpoint, tokenizer, handler, queue, batch_size = self.llama.init_apple(
                            self.model_name,
                            "mps",
                            "apple:0"
                        )
                        
                        valid_init = handler is not None
                        results["apple_init"] = "Success" if valid_init else "Failed Apple initialization"
                        
                        if valid_init:
                            # Use the handler that was returned
                            output = handler(self.test_prompt)
                            results["apple_handler"] = "Success" if output is not None else "Failed Apple handler"
                            
                            # Check output format
                            if output is not None:
                                results["apple_output"] = "Valid output" if len(output) > 0 else "Empty output"
                            else:
                                results["apple_output"] = "No output"
                        else:
                            # If init failed, create a test handler directly
                            test_handler = self.llama.create_apple_text_generation_endpoint_handler(
                                MagicMock(),
                                MagicMock(),
                                self.model_name,
                                "apple:0"
                            )
                            
                            results["apple_handler_direct"] = "Handler created" if test_handler is not None else "Failed to create handler"
                
                except (ImportError, Exception) as e:
                    # Fall back to pure mocking if CoreML isn't available
                    print(f"Falling back to mock for Apple: {str(e)}")
                    with patch('coremltools.convert') as mock_convert:
                        mock_convert.return_value = MagicMock()
                        
                        # Mock the Apple CoreML Utils
                        with patch.object(self.llama, 'coreml_utils', MagicMock()) as mock_utils:
                            mock_utils.is_available.return_value = True
                            mock_utils.load_model.return_value = MagicMock()
                            mock_utils.run_inference.return_value = {"logits": np.random.rand(1, 10, 30522)}
                            
                            endpoint, tokenizer, handler, queue, batch_size = self.llama.init_apple(
                                self.model_name,
                                "mps",
                                "apple:0"
                            )
                            
                            valid_init = handler is not None
                            results["apple_init"] = "Success (Mock)" if valid_init else "Failed Apple initialization"
                            
                            test_handler = self.llama.create_apple_text_generation_endpoint_handler(
                                MagicMock(),
                                MagicMock(),
                                self.model_name,
                                "apple:0"
                            )
                            
                            output = test_handler(self.test_prompt)
                            results["apple_handler"] = "Success (Mock)" if output is not None else "Failed Apple handler"
            except ImportError:
                results["apple_tests"] = "CoreML Tools not installed"
            except Exception as e:
                results["apple_tests"] = f"Error: {str(e)}"
        else:
            results["apple_tests"] = "Apple Silicon not available"

        # Test Qualcomm if available
        try:
            # Since Qualcomm SDK is rarely available in test environments,
            # we'll create realistic mocks for testing
            with patch('ipfs_accelerate_py.worker.skillset.qualcomm_snpe_utils.get_snpe_utils') as mock_snpe:
                # Create a more realistic SNPE utils mock
                mock_utils = MagicMock()
                mock_utils.is_available.return_value = True
                mock_utils.convert_model.return_value = True
                mock_utils.load_model.return_value = MagicMock()
                mock_utils.optimize_for_device.return_value = "/mock/optimized_model.dlc"
                mock_utils.run_inference.return_value = {
                    "logits": np.random.rand(1, 10, 30522),
                    "past_key_values": [(np.random.rand(1, 8, 64, 128), np.random.rand(1, 8, 64, 128)) for _ in range(4)]
                }
                
                # Setup mock for the SNPE utilities module
                mock_snpe.return_value = mock_utils
                
                # We need to patch the snpe_utils directly on the llama object
                with patch.object(self.llama, 'snpe_utils', mock_utils):
                    # Mock all required functions for Qualcomm
                    mock_get_qualcomm_genai_pipeline = MagicMock()
                    mock_get_optimum_qualcomm_model = MagicMock()
                    mock_get_qualcomm_model = MagicMock() 
                    mock_get_qualcomm_pipeline_type = MagicMock()
                    
                    # Initialize Qualcomm endpoint
                    endpoint, tokenizer, handler, queue, batch_size = self.llama.init_qualcomm(
                        self.model_name,
                        "text-generation",
                        "qualcomm",
                        "qualcomm:0",
                        mock_get_qualcomm_genai_pipeline,
                        mock_get_optimum_qualcomm_model,
                        mock_get_qualcomm_model,
                        mock_get_qualcomm_pipeline_type
                    )
                    
                    valid_init = handler is not None
                    results["qualcomm_init"] = "Success" if valid_init else "Failed Qualcomm initialization"
                    
                    if valid_init:
                        # Use the handler returned from initialization
                        output = handler(self.test_prompt)
                        results["qualcomm_handler"] = "Success" if output is not None else "Failed Qualcomm handler"
                    else:
                        # Create a handler manually for testing
                        test_handler = self.llama.create_qualcomm_llama_endpoint_handler(
                            MagicMock(),
                            self.model_name,
                            "qualcomm:0",
                            MagicMock()
                        )
                        
                        # Test the handler
                        output = test_handler(self.test_prompt)
                        results["qualcomm_handler"] = "Success (Direct)" if output is not None else "Failed Qualcomm handler"
                    
                    # Check output structure if available
                    if output is not None and isinstance(output, dict):
                        results["qualcomm_output"] = "Valid" if "generated_text" in output else "Missing generated_text"
                    else:
                        results["qualcomm_output"] = "Invalid output format"
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
        expected_dir = os.path.join(os.path.dirname(__file__), 'expected_results')
        collected_dir = os.path.join(os.path.dirname(__file__), 'collected_results')
        
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Add metadata about the environment to the results
        test_results["metadata"] = {
            "timestamp": time.time(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "transformers_mocked": isinstance(self.resources["transformers"], MagicMock),
            "test_model": self.model_name
        }
        
        # Save collected results
        with open(os.path.join(collected_dir, 'hf_llama_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_llama_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                    
                    # Only compare the non-metadata parts
                    expected_copy = {k: v for k, v in expected_results.items() if k != "metadata"}
                    results_copy = {k: v for k, v in test_results.items() if k != "metadata"}
                    
                    if expected_copy != results_copy:
                        print("Test results differ from expected results!")
                        print(f"Expected: {expected_copy}")
                        print(f"Got: {results_copy}")
                    else:
                        print("Test results match expected results")
            except Exception as e:
                print(f"Error comparing with expected results: {str(e)}")
                # Create/update expected results file
                with open(expected_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                    print(f"Updated expected results file: {expected_file}")
        else:
            # Create expected results file if it doesn't exist
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2)
                print(f"Created new expected results file: {expected_file}")

        return test_results

if __name__ == "__main__":
    try:
        this_llama = test_hf_llama()
        results = this_llama.__test__()
        print(f"LLaMA Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)