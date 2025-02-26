import os
import sys
import json
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from ...worker.skillset.default_embed import hf_embed

class test_hf_embed:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np,
            "transformers": MagicMock()
        }
        self.metadata = metadata if metadata else {}
        self.embed = hf_embed(resources=self.resources, metadata=self.metadata)
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.test_texts = [
            "The quick brown fox jumps over the lazy dog",
            "A fast auburn canine leaps above the sleepy hound"
        ]
        return None

    def test(self):
        """Run all tests for the text embedding model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.embed is not None else "Failed initialization"
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
                mock_model.return_value.last_hidden_state = torch.randn(1, 10, 384)
                
                endpoint, tokenizer, handler, queue, batch_size = self.embed.init_cpu(
                    self.model_name,
                    "cpu",
                    "cpu"
                )
                
                valid_init = endpoint is not None and tokenizer is not None and handler is not None
                results["cpu_init"] = "Success" if valid_init else "Failed CPU initialization"
                
                test_handler = self.embed.create_cpu_text_embedding_endpoint_handler(
                    endpoint,
                    "cpu",
                    endpoint,
                    tokenizer
                )
                
                # Test single text embedding
                single_output = test_handler(self.test_texts[0])
                results["cpu_single"] = "Success" if single_output is not None and len(single_output.shape) == 2 else "Failed single embedding"
                
                # Test batch text embedding
                batch_output = test_handler(self.test_texts)
                results["cpu_batch"] = "Success" if batch_output is not None and len(batch_output.shape) == 2 else "Failed batch embedding"
                
                # Test embedding similarity
                if single_output is not None and batch_output is not None:
                    similarity = torch.nn.functional.cosine_similarity(single_output, batch_output[0].unsqueeze(0))
                    results["cpu_similarity"] = "Success" if similarity is not None else "Failed similarity computation"
                
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
                    mock_model.return_value.last_hidden_state = torch.randn(1, 10, 384)
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.embed.init_cuda(
                        self.model_name,
                        "cuda",
                        "cuda:0"
                    )
                    
                    valid_init = endpoint is not None and tokenizer is not None and handler is not None
                    results["cuda_init"] = "Success" if valid_init else "Failed CUDA initialization"
                    
                    test_handler = self.embed.create_cuda_text_embedding_endpoint_handler(
                        endpoint,
                        "cuda:0",
                        endpoint,
                        tokenizer
                    )
                    
                    output = test_handler(self.test_texts)
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
                mock_openvino_cli_convert = MagicMock()
                
                endpoint, tokenizer, handler, queue, batch_size = self.embed.init_openvino(
                    self.model_name,
                    "feature-extraction",
                    "CPU",
                    "openvino:0",
                    mock_get_optimum_openvino_model,
                    mock_get_openvino_model,
                    mock_get_openvino_pipeline_type,
                    mock_openvino_cli_convert
                )
                
                valid_init = handler is not None
                results["openvino_init"] = "Success" if valid_init else "Failed OpenVINO initialization"
                
                test_handler = self.embed.create_openvino_text_embedding_endpoint_handler(
                    endpoint,
                    tokenizer,
                    "openvino:0",
                    endpoint
                )
                
                output = test_handler(self.test_texts)
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
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.embed.init_apple(
                        self.model_name,
                        "mps",
                        "apple:0"
                    )
                    
                    valid_init = handler is not None
                    results["apple_init"] = "Success" if valid_init else "Failed Apple initialization"
                    
                    test_handler = self.embed.create_apple_text_embedding_endpoint_handler(
                        endpoint,
                        tokenizer,
                        "apple:0",
                        endpoint
                    )
                    
                    # Test single and batch inputs
                    single_output = test_handler(self.test_texts[0])
                    results["apple_single"] = "Success" if single_output is not None else "Failed single text"
                    
                    batch_output = test_handler(self.test_texts)
                    results["apple_batch"] = "Success" if batch_output is not None else "Failed batch texts"
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
                
                endpoint, tokenizer, handler, queue, batch_size = self.embed.init_qualcomm(
                    self.model_name,
                    "qualcomm",
                    "qualcomm:0"
                )
                
                valid_init = handler is not None
                results["qualcomm_init"] = "Success" if valid_init else "Failed Qualcomm initialization"
                
                test_handler = self.embed.create_qualcomm_text_embedding_endpoint_handler(
                    endpoint,
                    tokenizer,
                    "qualcomm:0",
                    endpoint
                )
                
                output = test_handler(self.test_texts)
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
        expected_dir = os.path.join(os.path.dirname(__file__), 'expected_results')
        collected_dir = os.path.join(os.path.dirname(__file__), 'collected_results')
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save collected results
        with open(os.path.join(collected_dir, 'hf_embed_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_embed_test_results.json')
        if os.path.exists(expected_file):
            with open(expected_file, 'r') as f:
                expected_results = json.load(f)
                if expected_results != test_results:
                    print("Test results differ from expected results!")
                    print(f"Expected: {expected_results}")
                    print(f"Got: {test_results}")
        else:
            # Create expected results file if it doesn't exist
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2)
                print(f"Created new expected results file: {expected_file}")

        return test_results

if __name__ == "__main__":
    try:
        this_embed = test_hf_embed()
        results = this_embed.__test__()
        print(f"Embedding Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)