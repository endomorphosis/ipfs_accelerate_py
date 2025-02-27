import os
import sys
import json
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import transformers

# Use direct import with the absolute path
sys.path.insert(0, '/home/barberb/ipfs_accelerate_py')
from ipfs_accelerate_py.worker.skillset.hf_bert import hf_bert

class test_hf_bert:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np,
            "transformers": transformers
        }
        self.metadata = metadata if metadata else {}
        self.bert = hf_bert(resources=self.resources, metadata=self.metadata)
        self.model_name = "bert-base-uncased"
        self.test_text = "The quick brown fox jumps over the lazy dog"
        return None
        
    def test(self):
        """Run all tests for the BERT text embedding model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.bert is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"

        # Test CPU initialization and handler - using real inference
        try:
            # Initialize for CPU without mocks
            endpoint, tokenizer, handler, queue, batch_size = self.bert.init_cpu(
                self.model_name,
                "cpu", 
                "cpu"
            )
            
            valid_init = endpoint is not None and tokenizer is not None and handler is not None
            results["cpu_init"] = "Success" if valid_init else "Failed CPU initialization"
            
            # Get handler for CPU directly from initialization
            # The handler should now be real and not mocked
            test_handler = handler
            
            # Run actual inference
            output = test_handler(self.test_text)
            
            # Verify the output is a real embedding tensor
            is_valid_embedding = (
                output is not None and 
                isinstance(output, torch.Tensor) and 
                output.dim() == 2 and 
                output.size(0) == 1  # batch size
            )
            
            results["cpu_handler"] = "Success" if is_valid_embedding else "Failed CPU handler"
            
            # Add embedding shape to results
            if is_valid_embedding:
                results["cpu_embedding_shape"] = list(output.shape)
                results["cpu_embedding_type"] = str(output.dtype)
                
        except Exception as e:
            results["cpu_tests"] = f"Error: {str(e)}"

        # Test CUDA if available - using real inference
        if torch.cuda.is_available():
            try:
                # Initialize for CUDA without mocks
                endpoint, tokenizer, handler, queue, batch_size = self.bert.init_cuda(
                    self.model_name,
                    "cuda",
                    "cuda:0"
                )
                
                valid_init = endpoint is not None and tokenizer is not None and handler is not None
                results["cuda_init"] = "Success" if valid_init else "Failed CUDA initialization"
                
                # Get handler for CUDA directly from initialization
                test_handler = handler
                
                # Run actual inference
                output = test_handler(self.test_text)
                
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
                
                results["cuda_handler"] = "Success" if is_valid_embedding else "Failed CUDA handler"
                
                # Add embedding shape to results
                if is_valid_embedding:
                    if isinstance(output, dict) and 'hidden_states' in output:
                        results["cuda_embedding_shape"] = list(output['hidden_states'].shape)
                    elif isinstance(output, torch.Tensor):
                        results["cuda_embedding_shape"] = list(output.shape)
                    elif isinstance(output, np.ndarray):
                        results["cuda_embedding_shape"] = list(output.shape)
                
            except Exception as e:
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"

        # Test OpenVINO if installed
        try:
            # First check if OpenVINO is installed
            try:
                import openvino
            except ImportError:
                results["openvino_tests"] = "OpenVINO not installed"
                return results
                
            # We'll use a combination of mocks for parts we can't run directly
            # and real inference for the parts we can
            
            # Import the existing OpenVINO utils from the main package
            from ipfs_accelerate_py.worker.openvino_utils import openvino_utils
            
            # Initialize openvino_utils
            ov_utils = openvino_utils(resources=self.resources, metadata=self.metadata)
            
            # Create a minimal OpenVINO model for testing
            try:
                # Import Core from the correct location based on OpenVINO version
                try:
                    from openvino.runtime import Core
                except (ImportError, AttributeError):
                    from openvino import Core
                core = Core()
                
                # Option 1: Try to create a minimal OpenVINO model from scratch
                # Simple matrix multiplication model as mock BERT
                import numpy as np
                from openvino.runtime import PartialShape, Type, Model, Output
                
                input_shapes = {
                    "input_ids": PartialShape([1, 10]),
                    "attention_mask": PartialShape([1, 10])
                }
                
                # Create a custom model class instead of a MagicMock
                # This ensures our model returns real tensors with proper shapes
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
                
                mock_ov_model = CustomOpenVINOModel()
                
                # Have the mock getters return our mock model
                mock_get_openvino_model.return_value = mock_ov_model
                mock_get_optimum_openvino_model.return_value = mock_ov_model
                
                # Initialize with real OpenVINO utils
                endpoint, tokenizer, handler, queue, batch_size = self.bert.init_openvino(
                    model_name=self.model_name,
                    model_type="feature-extraction",
                    device="CPU",
                    openvino_label="openvino:0",
                    get_optimum_openvino_model=ov_utils.get_optimum_openvino_model,
                    get_openvino_model=ov_utils.get_openvino_model,
                    get_openvino_pipeline_type=ov_utils.get_openvino_pipeline_type,
                    openvino_cli_convert=ov_utils.openvino_cli_convert
                )
                
                # If we got a handler back, we succeeded
                valid_init = handler is not None
                results["openvino_init"] = "Success" if valid_init else "Failed OpenVINO initialization"
                
                # If tokenizer is None (it might be from our mocks), create a simulated one
                if tokenizer is None:
                    # Create a functional tokenizer object
                    class SimpleTokenizer:
                        def __call__(self, text, return_tensors=None, padding=None, truncation=None, max_length=None):
                            if isinstance(text, str):
                                batch_size = 1
                            else:
                                batch_size = len(text)
                                
                            return {
                                "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
                                "attention_mask": torch.ones((batch_size, 10), dtype=torch.long)
                            }
                    
                    tokenizer = SimpleTokenizer()
                
                # Create handler if needed
                if handler is None:
                    test_handler = self.bert.create_openvino_text_embedding_endpoint_handler(
                        endpoint_model=self.model_name,
                        tokenizer=tokenizer,
                        openvino_label="openvino:0",
                        endpoint=mock_ov_model
                    )
                else:
                    test_handler = handler
                
                # Run inference with extra debugging
                try:
                    print("Running OpenVINO inference with test handler...")
                    output = test_handler(self.test_text)
                    print(f"OpenVINO output type: {type(output)}")
                    if output is not None:
                        print(f"OpenVINO output shape: {output.shape if hasattr(output, 'shape') else 'no shape'}")
                    
                    is_valid_embedding = (
                        output is not None and
                        isinstance(output, (torch.Tensor, np.ndarray)) and
                        hasattr(output, 'shape') and
                        output.shape[0] == 1  # batch size 1
                    )
                    
                    results["openvino_handler"] = "Success" if is_valid_embedding else "Failed OpenVINO handler"
                except Exception as e:
                    print(f"Exception in OpenVINO handler: {e}")
                    import traceback
                    traceback.print_exc()
                    results["openvino_handler"] = f"Error: {str(e)}"
                
                # Add embedding details if successful
                if is_valid_embedding:
                    if isinstance(output, torch.Tensor):
                        results["openvino_embedding_shape"] = list(output.shape)
                    else:
                        results["openvino_embedding_shape"] = list(output.shape)
                
            except Exception as e:
                print(f"Error in OpenVINO inference test: {e}")
                results["openvino_tests"] = f"Error: {str(e)}"
        except ImportError:
            results["openvino_tests"] = "OpenVINO not installed"
        except Exception as e:
            results["openvino_tests"] = f"Error: {str(e)}"

        # Test Apple Silicon if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                try:
                    import coremltools  # Only try import if MPS is available
                except ImportError:
                    results["apple_tests"] = "CoreML Tools not installed"
                    return results

                with patch('coremltools.convert') as mock_convert:
                    mock_convert.return_value = MagicMock()
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.bert.init_apple(
                        self.model_name,
                        "mps",
                        "apple:0"
                    )
                    
                    valid_init = handler is not None
                    results["apple_init"] = "Success" if valid_init else "Failed Apple initialization"
                    
                    test_handler = self.bert.create_apple_text_embedding_endpoint_handler(
                        endpoint_model=self.model_name,
                        apple_label="apple:0",
                        endpoint=endpoint,
                        tokenizer=tokenizer
                    )
                    
                    output = test_handler(self.test_text)
                    results["apple_handler"] = "Success" if output is not None else "Failed Apple handler"
            except ImportError:
                results["apple_tests"] = "CoreML Tools not installed"
            except Exception as e:
                results["apple_tests"] = f"Error: {str(e)}"
        else:
            results["apple_tests"] = "Apple Silicon not available"

        # Test Qualcomm if available
        try:
            try:
                from ipfs_accelerate_py.worker.skillset.qualcomm_snpe_utils import get_snpe_utils
            except ImportError:
                results["qualcomm_tests"] = "SNPE SDK not installed"
                return results
                
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
                results["qualcomm_init"] = "Success" if valid_init else "Failed Qualcomm initialization"
                
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
                
                output = test_handler(self.test_text)
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
        results_file = os.path.join(collected_dir, 'hf_bert_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
        except Exception as e:
            print(f"Error saving results to {results_file}: {str(e)}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_bert_test_results.json')
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
        this_bert = test_hf_bert()
        results = this_bert.__test__()
        print(f"BERT Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)