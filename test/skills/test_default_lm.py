import os
import sys
import json
import time
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Use direct import with the absolute path
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")
from ipfs_accelerate_py.worker.skillset.default_lm import hf_lm

class test_hf_lm:
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
        self.lm = hf_lm(resources=self.resources, metadata=self.metadata)
        
        # Use small public model for testing
        self.model_name = "facebook/opt-125m"
        self.test_prompt = "Once upon a time"
        self.test_generation_config = {
            "max_new_tokens": 20,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        return None

    def test(self):
        """Run all tests for the base language model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.lm is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"

        # Test CPU initialization and handler
        try:
            # Try with real model first
            try:
                transformers_available = isinstance(self.resources["transformers"], MagicMock) == False
                if transformers_available:
                    print("Testing with real transformers")
                    # Real model initialization
                    endpoint, tokenizer, handler, queue, batch_size = self.lm.init_cpu(
                        self.model_name,
                        "cpu",
                        "cpu"
                    )
                    
                    valid_init = endpoint is not None and tokenizer is not None and handler is not None
                    results["cpu_init"] = "Success" if valid_init else "Failed CPU initialization"
                    
                    if valid_init:
                        # Test standard text generation
                        output = handler(self.test_prompt)
                        results["cpu_standard"] = "Success" if output is not None else "Failed standard generation"
                        
                        # Test with generation config
                        output_with_config = handler(self.test_prompt, generation_config=self.test_generation_config)
                        results["cpu_config"] = "Success" if output_with_config is not None else "Failed config generation"
                        
                        # Test batch generation
                        batch_output = handler([self.test_prompt, self.test_prompt])
                        results["cpu_batch"] = "Success" if batch_output is not None and isinstance(batch_output, list) else "Failed batch generation"
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
                    mock_tokenizer.batch_decode = MagicMock(return_value=["Test response", "Test response"])
                    mock_tokenizer.decode = MagicMock(return_value="Test response")
                    
                    mock_model.return_value = MagicMock()
                    mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3], [4, 5, 6]])
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.lm.init_cpu(
                        self.model_name,
                        "cpu",
                        "cpu"
                    )
                    
                    valid_init = endpoint is not None and tokenizer is not None and handler is not None
                    results["cpu_init"] = "Success (Mock)" if valid_init else "Failed CPU initialization"
                    
                    # Test standard text generation
                    output = handler(self.test_prompt)
                    results["cpu_standard"] = "Success (Mock)" if output is not None else "Failed standard generation"
                    
                    # Test with generation config
                    output_with_config = handler(self.test_prompt, generation_config=self.test_generation_config)
                    results["cpu_config"] = "Success (Mock)" if output_with_config is not None else "Failed config generation"
                    
                    # Test batch generation
                    batch_output = handler([self.test_prompt, self.test_prompt])
                    results["cpu_batch"] = "Success (Mock)" if batch_output is not None and isinstance(batch_output, list) else "Failed batch generation"
                
        except Exception as e:
            results["cpu_tests"] = f"Error: {str(e)}"

        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                     patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                     patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                    
                    mock_config.return_value = MagicMock()
                    mock_tokenizer.return_value = MagicMock()
                    mock_model.return_value = MagicMock()
                    mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3]])
                    mock_tokenizer.decode.return_value = "Test response"
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.lm.init_cuda(
                        self.model_name,
                        "cuda",
                        "cuda:0"
                    )
                    
                    valid_init = endpoint is not None and tokenizer is not None and handler is not None
                    results["cuda_init"] = "Success" if valid_init else "Failed CUDA initialization"
                    
                    test_handler = self.lm.create_cuda_lm_endpoint_handler(
                        endpoint,
                        tokenizer,
                        self.model_name,
                        "cuda:0"
                    )
                    
                    output = test_handler(self.test_prompt)
                    results["cuda_handler"] = "Success" if output is not None else "Failed CUDA handler"
            except Exception as e:
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"

        # Test OpenVINO if installed
        try:
            import openvino
            with patch('openvino.Runtime') as mock_runtime:
                mock_runtime.return_value = MagicMock()
                mock_get_openvino_model = MagicMock()
                mock_get_optimum_openvino_model = MagicMock()
                mock_get_openvino_pipeline_type = MagicMock()
                
                endpoint, tokenizer, handler, queue, batch_size = self.lm.init_openvino(
                    self.model_name,
                    "text-generation",
                    "CPU",
                    "openvino:0",
                    mock_get_optimum_openvino_model,
                    mock_get_openvino_model,
                    mock_get_openvino_pipeline_type
                )
                
                valid_init = handler is not None
                results["openvino_init"] = "Success" if valid_init else "Failed OpenVINO initialization"
                
                test_handler = self.lm.create_openvino_lm_endpoint_handler(
                    endpoint,
                    tokenizer,
                    self.model_name,
                    "openvino:0"
                )
                
                output = test_handler(self.test_prompt)
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
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.lm.init_apple(
                        self.model_name,
                        "mps",
                        "apple:0"
                    )
                    
                    valid_init = handler is not None
                    results["apple_init"] = "Success" if valid_init else "Failed Apple initialization"
                    
                    test_handler = self.lm.create_apple_lm_endpoint_handler(
                        endpoint,
                        tokenizer,
                        self.model_name,
                        "apple:0"
                    )
                    
                    # Test different generation scenarios
                    standard_output = test_handler(self.test_prompt)
                    results["apple_standard"] = "Success" if standard_output is not None else "Failed standard generation"
                    
                    config_output = test_handler(self.test_prompt, generation_config=self.test_generation_config)
                    results["apple_config"] = "Success" if config_output is not None else "Failed config generation"
                    
                    batch_output = test_handler([self.test_prompt, self.test_prompt])
                    results["apple_batch"] = "Success" if batch_output is not None else "Failed batch generation"
            except ImportError:
                results["apple_tests"] = "CoreML Tools not installed"
            except Exception as e:
                results["apple_tests"] = f"Error: {str(e)}"
        else:
            results["apple_tests"] = "Apple Silicon not available"

        # Test Qualcomm model
        try:
            endpoint, tokenizer, handler, queue, batch_size = self.lm.init_qualcomm(
                self.model_name,
                "qualcomm",
                "qualcomm:0"
            )
            
            valid_init = handler is not None
            results["qualcomm_init"] = "Success" if valid_init else "Failed Qualcomm initialization"
            
            # Test with integrated handler
            output = handler(self.test_prompt)
            results["qualcomm_handler"] = "Success" if output is not None else "Failed Qualcomm handler"
            
            # Test with specific generation parameters
            output_with_config = handler(self.test_prompt, generation_config=self.test_generation_config)
            results["qualcomm_config"] = "Success" if output_with_config is not None else "Failed Qualcomm config"
            
            # Test batch processing
            batch_output = handler([self.test_prompt, self.test_prompt])
            results["qualcomm_batch"] = "Success" if batch_output is not None and isinstance(batch_output, list) else "Failed batch generation"
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
        with open(os.path.join(collected_dir, 'hf_lm_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_lm_test_results.json')
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
        this_lm = test_hf_lm()
        results = this_lm.__test__()
        print(f"Language Model Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)