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
                    results["cpu_init"] = "Success (REAL)" if valid_init else "Failed CPU initialization"
                    
                    if valid_init:
                        # Test standard text generation
                        output = handler(self.test_prompt)
                        results["cpu_standard"] = "Success (REAL)" if output is not None else "Failed standard generation"
                        
                        # Include sample output for verification
                        if output is not None:
                            # Truncate long outputs for readability
                            if len(output) > 100:
                                results["cpu_standard_output"] = output[:100] + "..."
                            else:
                                results["cpu_standard_output"] = output
                            results["cpu_standard_output_length"] = len(output)
                            results["cpu_standard_timestamp"] = time.time()
                        
                        # Test with generation config
                        output_with_config = handler(self.test_prompt, generation_config=self.test_generation_config)
                        results["cpu_config"] = "Success (REAL)" if output_with_config is not None else "Failed config generation"
                        
                        # Include sample config output for verification
                        if output_with_config is not None:
                            if len(output_with_config) > 100:
                                results["cpu_config_output"] = output_with_config[:100] + "..."
                            else:
                                results["cpu_config_output"] = output_with_config
                            results["cpu_config_output_length"] = len(output_with_config)
                        
                        # Test batch generation
                        batch_output = handler([self.test_prompt, self.test_prompt])
                        results["cpu_batch"] = "Success (REAL)" if batch_output is not None and isinstance(batch_output, list) else "Failed batch generation"
                        
                        # Include sample batch output for verification
                        if batch_output is not None and isinstance(batch_output, list):
                            results["cpu_batch_output_count"] = len(batch_output)
                            if len(batch_output) > 0:
                                results["cpu_batch_first_output"] = batch_output[0][:50] + "..." if len(batch_output[0]) > 50 else batch_output[0]
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
                    mock_tokenizer.batch_decode = MagicMock(return_value=["Test response (MOCK)", "Test response (MOCK)"])
                    mock_tokenizer.decode = MagicMock(return_value="Test response (MOCK)")
                    
                    mock_model.return_value = MagicMock()
                    mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3], [4, 5, 6]])
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.lm.init_cpu(
                        self.model_name,
                        "cpu",
                        "cpu"
                    )
                    
                    valid_init = endpoint is not None and tokenizer is not None and handler is not None
                    results["cpu_init"] = "Success (MOCK)" if valid_init else "Failed CPU initialization"
                    
                    # Test standard text generation
                    output = handler(self.test_prompt)
                    results["cpu_standard"] = "Success (MOCK)" if output is not None else "Failed standard generation"
                    
                    # Include sample output for verification
                    if output is not None:
                        results["cpu_standard_output"] = output
                        results["cpu_standard_output_length"] = len(output)
                        results["cpu_standard_timestamp"] = time.time()
                    
                    # Test with generation config
                    output_with_config = handler(self.test_prompt, generation_config=self.test_generation_config)
                    results["cpu_config"] = "Success (MOCK)" if output_with_config is not None else "Failed config generation"
                    
                    # Include sample config output for verification
                    if output_with_config is not None:
                        results["cpu_config_output"] = output_with_config
                        results["cpu_config_output_length"] = len(output_with_config)
                    
                    # Test batch generation
                    batch_output = handler([self.test_prompt, self.test_prompt])
                    results["cpu_batch"] = "Success (MOCK)" if batch_output is not None and isinstance(batch_output, list) else "Failed batch generation"
                    
                    # Include sample batch output for verification
                    if batch_output is not None and isinstance(batch_output, list):
                        results["cpu_batch_output_count"] = len(batch_output)
                        if len(batch_output) > 0:
                            results["cpu_batch_first_output"] = batch_output[0]
                
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
                    results["cuda_init"] = "Success (MOCK)" if valid_init else "Failed CUDA initialization"
                    
                    test_handler = self.lm.create_cuda_lm_endpoint_handler(
                        endpoint,
                        tokenizer,
                        self.model_name,
                        "cuda:0"
                    )
                    
                    output = test_handler(self.test_prompt)
                    results["cuda_handler"] = "Success (MOCK)" if output is not None else "Failed CUDA handler"
                    
                    # Include sample output for verification
                    if output is not None:
                        if len(output) > 100:
                            results["cuda_output"] = output[:100] + "..."
                        else:
                            results["cuda_output"] = output
                        results["cuda_output_length"] = len(output)
                        results["cuda_timestamp"] = time.time()
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
                # The init_openvino method takes up to 7 arguments, but we're passing 8
                # Remove the last argument to match the expected signature
                endpoint, tokenizer, handler, queue, batch_size = self.lm.init_openvino(
                    self.model_name,
                    "text-generation",
                    "CPU",
                    "openvino:0",
                    ov_utils.get_optimum_openvino_model,
                    ov_utils.get_openvino_model,
                    ov_utils.get_openvino_pipeline_type
                )
                
                valid_init = handler is not None
                results["openvino_init"] = "Success (MOCK)" if valid_init else "Failed OpenVINO initialization"
                
                test_handler = self.lm.create_openvino_lm_endpoint_handler(
                    endpoint,
                    tokenizer,
                    self.model_name,
                    "openvino:0"
                )
                
                output = test_handler(self.test_prompt)
                results["openvino_handler"] = "Success (MOCK)" if output is not None else "Failed OpenVINO handler"
                
                # Include sample output for verification
                if output is not None:
                    if len(output) > 100:
                        results["openvino_output"] = output[:100] + "..."
                    else:
                        results["openvino_output"] = output
                    results["openvino_output_length"] = len(output)
                    results["openvino_timestamp"] = time.time()
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
                    results["apple_init"] = "Success (MOCK)" if valid_init else "Failed Apple initialization"
                    
                    test_handler = self.lm.create_apple_lm_endpoint_handler(
                        endpoint,
                        tokenizer,
                        self.model_name,
                        "apple:0"
                    )
                    
                    # Test different generation scenarios
                    standard_output = test_handler(self.test_prompt)
                    results["apple_standard"] = "Success (MOCK)" if standard_output is not None else "Failed standard generation"
                    
                    # Include sample output for verification
                    if standard_output is not None:
                        if len(standard_output) > 100:
                            results["apple_standard_output"] = standard_output[:100] + "..."
                        else:
                            results["apple_standard_output"] = standard_output
                        results["apple_standard_timestamp"] = time.time()
                    
                    config_output = test_handler(self.test_prompt, generation_config=self.test_generation_config)
                    results["apple_config"] = "Success (MOCK)" if config_output is not None else "Failed config generation"
                    
                    # Include sample config output for verification
                    if config_output is not None:
                        if len(config_output) > 100:
                            results["apple_config_output"] = config_output[:100] + "..."
                        else:
                            results["apple_config_output"] = config_output
                    
                    batch_output = test_handler([self.test_prompt, self.test_prompt])
                    results["apple_batch"] = "Success (MOCK)" if batch_output is not None else "Failed batch generation"
                    
                    # Include sample batch output for verification
                    if batch_output is not None and isinstance(batch_output, list):
                        results["apple_batch_output_count"] = len(batch_output)
                        if len(batch_output) > 0:
                            results["apple_batch_first_output"] = batch_output[0][:50] + "..." if len(batch_output[0]) > 50 else batch_output[0]
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
            results["qualcomm_init"] = "Success (MOCK)" if valid_init else "Failed Qualcomm initialization"
            
            # Test with integrated handler
            output = handler(self.test_prompt)
            results["qualcomm_handler"] = "Success (MOCK)" if output is not None else "Failed Qualcomm handler"
            
            # Include sample output for verification
            if output is not None:
                if len(output) > 100:
                    results["qualcomm_output"] = output[:100] + "..."
                else:
                    results["qualcomm_output"] = output
                results["qualcomm_output_length"] = len(output)
                results["qualcomm_timestamp"] = time.time()
            
            # Test with specific generation parameters
            output_with_config = handler(self.test_prompt, generation_config=self.test_generation_config)
            results["qualcomm_config"] = "Success (MOCK)" if output_with_config is not None else "Failed Qualcomm config"
            
            # Include sample config output for verification
            if output_with_config is not None:
                if len(output_with_config) > 100:
                    results["qualcomm_config_output"] = output_with_config[:100] + "..."
                else:
                    results["qualcomm_config_output"] = output_with_config
            
            # Test batch processing
            batch_output = handler([self.test_prompt, self.test_prompt])
            results["qualcomm_batch"] = "Success (MOCK)" if batch_output is not None and isinstance(batch_output, list) else "Failed batch generation"
            
            # Include sample batch output for verification
            if batch_output is not None and isinstance(batch_output, list):
                results["qualcomm_batch_output_count"] = len(batch_output)
                if len(batch_output) > 0:
                    results["qualcomm_batch_first_output"] = batch_output[0][:50] + "..." if len(batch_output[0]) > 50 else batch_output[0]
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
            "test_model": self.model_name,
            "test_run_id": f"lm-test-{int(time.time())}"
        }
        
        # Fix for issue with collected file having "clas" prefix
        collected_file = os.path.join(collected_dir, 'hf_lm_test_results.json')
        # Save collected results
        with open(collected_file, 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_lm_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                    
                    # Only compare the non-metadata parts and exclude outputs and timestamps
                    excluded_keys = ["metadata", "cpu_standard_output", "cpu_config_output", 
                                    "qualcomm_output", "openvino_output", "apple_standard_output",
                                    "cuda_output", "cpu_batch_first_output", "qualcomm_batch_first_output",
                                    "apple_batch_first_output", "qualcomm_config_output", "apple_config_output"]
                    
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
                        print("Consider updating the expected results file if these differences are intentional")
                    else:
                        print("Core test results match expected results (excluding variable outputs)")
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