import os
import sys
import json
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from ...worker.skillset.hf_bert import hf_bert

class test_hf_bert:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np,
            "transformers": MagicMock()
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

        # Test CPU initialization and handler
        try:
            with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                 patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                 patch('transformers.AutoModel.from_pretrained') as mock_model:
                
                mock_config.return_value = MagicMock()
                mock_tokenizer.return_value = MagicMock()
                mock_model.return_value = MagicMock()
                
                endpoint, tokenizer, handler, queue, batch_size = self.bert.init_cpu(
                    self.model_name,
                    "cpu",
                    "cpu"
                )
                
                valid_init = endpoint is not None and tokenizer is not None and handler is not None
                results["cpu_init"] = "Success" if valid_init else "Failed CPU initialization"
                
                test_handler = self.bert.create_cpu_text_embedding_endpoint_handler(
                    endpoint,
                    self.model_name,
                    "cpu",
                    tokenizer
                )
                
                output = test_handler(self.test_text)
                results["cpu_handler"] = "Success" if output is not None else "Failed CPU handler"
                
        except Exception as e:
            results["cpu_tests"] = f"Error: {str(e)}"

        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                     patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                     patch('transformers.AutoModel.from_pretrained') as mock_model:
                    
                    mock_config.return_value = MagicMock()
                    mock_tokenizer.return_value = MagicMock()
                    mock_model.return_value = MagicMock()
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.bert.init_cuda(
                        self.model_name,
                        "cuda",
                        "cuda:0"
                    )
                    
                    valid_init = endpoint is not None and tokenizer is not None and handler is not None
                    results["cuda_init"] = "Success" if valid_init else "Failed CUDA initialization"
                    
                    test_handler = self.bert.create_cuda_text_embedding_endpoint_handler(
                        endpoint_model=self.model_name,
                        cuda_label="cuda:0",
                        endpoint=endpoint,
                        tokenizer=tokenizer
                    )
                    
                    output = test_handler(self.test_text)
                    results["cuda_handler"] = "Success" if output is not None else "Failed CUDA handler"
            except Exception as e:
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"

        # Test OpenVINO if installed
        try:
            try:
                import openvino
            except ImportError:
                results["openvino_tests"] = "OpenVINO not installed"
                return results
                
            with patch('openvino.Runtime') as mock_runtime:
                mock_runtime.return_value = MagicMock()
                mock_get_openvino_model = MagicMock()
                mock_get_optimum_openvino_model = MagicMock()
                mock_get_openvino_pipeline_type = MagicMock()
                mock_openvino_cli_convert = MagicMock()
                
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
                
                valid_init = handler is not None
                results["openvino_init"] = "Success" if valid_init else "Failed OpenVINO initialization"
                
                test_handler = self.bert.create_openvino_text_embedding_endpoint_handler(
                    endpoint_model=self.model_name,
                    tokenizer=tokenizer,
                    openvino_label="openvino:0",
                    endpoint=endpoint
                )
                
                output = test_handler(self.test_text)
                results["openvino_handler"] = "Success" if output is not None else "Failed OpenVINO handler"
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
                
            with patch('ipfs_accelerate_py.worker.skillset.qualcomm_snpe_utils.get_snpe_utils') as mock_snpe:
                mock_snpe.return_value = MagicMock()
                
                endpoint, tokenizer, handler, queue, batch_size = self.bert.init_qualcomm(
                    self.model_name,
                    "qualcomm",
                    "qualcomm:0"
                )
                
                valid_init = handler is not None
                results["qualcomm_init"] = "Success" if valid_init else "Failed Qualcomm initialization"
                
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