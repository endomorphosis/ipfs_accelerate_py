import os
import sys
import json
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image

# Add a patch for missing functions
def mock_load_audio(audio_file):
    return np.zeros(16000, dtype=np.float32), 16000

# Use direct import with the absolute path
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")
from ipfs_accelerate_py.worker.skillset.hf_clap import hf_clap

# Patch the class method after importing
with patch('ipfs_accelerate_py.worker.skillset.hf_clap.load_audio', mock_load_audio):
    pass

class test_hf_clap:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np,
            "transformers": MagicMock(),
            "soundfile": MagicMock()
        }
        self.metadata = metadata if metadata else {}
        self.clap = hf_clap(resources=self.resources, metadata=self.metadata)
        self.model_name = "laion/clap-htsat-unfused"
        self.test_audio_url = "https://calamitymod.wiki.gg/images/2/29/Bees3.wav"
        self.test_text = "buzzing bees"
        return None

    def test(self):
        """Run all tests for the CLAP audio-language model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.clap is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"

        # Test audio loading utilities
        try:
            with patch('soundfile.read') as mock_sf_read, \
                 patch('requests.get') as mock_get:
                mock_response = MagicMock()
                mock_response.content = b"fake_audio_data"
                mock_get.return_value = mock_response
                mock_sf_read.return_value = (np.random.randn(16000), 16000)
                
                audio_data, sr = load_audio(self.test_audio_url)
                results["load_audio"] = "Success" if audio_data is not None and sr == 16000 else "Failed audio loading"
                
                audio_tensor = load_audio_tensor(self.test_audio_url)
                results["load_audio_tensor"] = "Success" if audio_tensor is not None else "Failed tensor conversion"
        except Exception as e:
            results["audio_utils"] = f"Error: {str(e)}"

        # Test CPU initialization and handler
        try:
            with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                 patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                 patch('transformers.ClapProcessor.from_pretrained') as mock_processor, \
                 patch('transformers.ClapModel.from_pretrained') as mock_model:
                
                mock_config.return_value = MagicMock()
                mock_tokenizer.return_value = MagicMock()
                mock_processor.return_value = MagicMock()
                mock_model.return_value = MagicMock()
                
                endpoint, processor, handler, queue, batch_size = self.clap.init_cpu(
                    self.model_name,
                    "cpu",
                    "cpu"
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["cpu_init"] = "Success" if valid_init else "Failed CPU initialization"
                
                test_handler = self.clap.create_cpu_audio_embedding_endpoint_handler(
                    endpoint,
                    tokenizer,
                    self.model_name,
                    "cpu"
                )
                
                # Test with mock audio input
                with patch('soundfile.read') as mock_sf_read:
                    mock_sf_read.return_value = (np.random.randn(16000), 16000)
                    
                    # Test audio embedding
                    audio_embedding = test_handler(self.test_audio_url)
                    results["cpu_audio_embedding"] = "Success" if audio_embedding is not None else "Failed audio embedding"
                    
                    # Test text embedding
                    text_embedding = test_handler(text=self.test_text)
                    results["cpu_text_embedding"] = "Success" if text_embedding is not None else "Failed text embedding"
                    
                    # Test audio-text similarity
                    similarity = test_handler(self.test_audio_url, self.test_text)
                    results["cpu_similarity"] = "Success" if similarity is not None else "Failed similarity computation"
        except Exception as e:
            results["cpu_tests"] = f"Error: {str(e)}"

        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                     patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                     patch('transformers.ClapProcessor.from_pretrained') as mock_processor, \
                     patch('transformers.ClapModel.from_pretrained') as mock_model:
                    
                    mock_config.return_value = MagicMock()
                    mock_tokenizer.return_value = MagicMock()
                    mock_processor.return_value = MagicMock()
                    mock_model.return_value = MagicMock()
                    
                    endpoint, processor, handler, queue, batch_size = self.clap.init_cuda(
                        self.model_name,
                        "cuda",
                        "cuda:0"
                    )
                    
                    valid_init = endpoint is not None and processor is not None and handler is not None
                    results["cuda_init"] = "Success" if valid_init else "Failed CUDA initialization"
                    
                    test_handler = self.clap.create_cuda_audio_embedding_endpoint_handler(
                        endpoint,
                        tokenizer,
                        self.model_name,
                        "cuda:0"
                    )
                    
                    with patch('soundfile.read') as mock_sf_read:
                        mock_sf_read.return_value = (np.random.randn(16000), 16000)
                        output = test_handler(self.test_audio_url, self.test_text)
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
                
            # Import the existing OpenVINO utils from the main package
            from ipfs_accelerate_py.worker.openvino_utils import openvino_utils
            
            # Initialize openvino_utils
            ov_utils = openvino_utils(resources=self.resources, metadata=self.metadata)
            
            # Use a patched version for testing
            with patch('openvino.runtime.Core' if hasattr(openvino, 'runtime') and hasattr(openvino.runtime, 'Core') else 'openvino.Core'):
                
                endpoint, processor, handler, queue, batch_size = self.clap.init_openvino(
                    self.model_name,
                    "audio-classification",
                    "CPU",
                    "openvino:0",
                    ov_utils.get_optimum_openvino_model,
                    ov_utils.get_openvino_model,
                    ov_utils.get_openvino_pipeline_type,
                    ov_utils.openvino_cli_convert
                )
                
                valid_init = handler is not None
                results["openvino_init"] = "Success" if valid_init else "Failed OpenVINO initialization"
                
                test_handler = self.clap.create_openvino_audio_embedding_endpoint_handler(
                    endpoint,
                    tokenizer,
                    self.model_name,
                    "openvino:0"
                )
                
                with patch('soundfile.read') as mock_sf_read:
                    mock_sf_read.return_value = (np.random.randn(16000), 16000)
                    output = test_handler(self.test_audio_url, self.test_text)
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
                    
                    endpoint, processor, handler, queue, batch_size = self.clap.init_apple(
                        self.model_name,
                        "mps",
                        "apple:0"
                    )
                    
                    valid_init = handler is not None
                    results["apple_init"] = "Success" if valid_init else "Failed Apple initialization"
                    
                    test_handler = self.clap.create_apple_audio_embedding_endpoint_handler(
                        endpoint,
                        tokenizer,
                        self.model_name,
                        "apple:0"
                    )
                    
                    with patch('soundfile.read') as mock_sf_read:
                        mock_sf_read.return_value = (np.random.randn(16000), 16000)
                        # Test different input types
                        audio_output = test_handler(self.test_audio_url)
                        results["apple_audio"] = "Success" if audio_output is not None else "Failed audio input"
                        
                        text_output = test_handler(text=self.test_text)
                        results["apple_text"] = "Success" if text_output is not None else "Failed text input"
                        
                        similarity = test_handler(self.test_audio_url, self.test_text)
                        results["apple_similarity"] = "Success" if similarity is not None else "Failed similarity computation"
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
                
                endpoint, processor, handler, queue, batch_size = self.clap.init_qualcomm(
                    self.model_name,
                    "qualcomm",
                    "qualcomm:0"
                )
                
                valid_init = handler is not None
                results["qualcomm_init"] = "Success" if valid_init else "Failed Qualcomm initialization"
                
                test_handler = self.clap.create_qualcomm_audio_embedding_endpoint_handler(
                    endpoint,
                    tokenizer,
                    self.model_name,
                    "qualcomm:0"
                )
                
                with patch('soundfile.read') as mock_sf_read:
                    mock_sf_read.return_value = (np.random.randn(16000), 16000)
                    output = test_handler(self.test_audio_url, self.test_text)
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
        with open(os.path.join(collected_dir, 'hf_clap_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_clap_test_results.json')
        if os.path.exists(expected_file):
            with open(expected_file, 'r') as f:
                expected_results = json.load(f)
                
                # More detailed comparison of results
                all_match = True
                mismatches = []
                
                for key in set(expected_results.keys()) | set(test_results.keys()):
                    if key not in expected_results:
                        mismatches.append(f"Missing expected key: {key}")
                        all_match = False
                    elif key not in test_results:
                        mismatches.append(f"Missing actual key: {key}")
                        all_match = False
                    elif expected_results[key] != test_results[key]:
                        mismatches.append(f"Key '{key}' differs: Expected '{expected_results[key]}', got '{test_results[key]}'")
                        all_match = False
                
                if not all_match:
                    print("Test results differ from expected results!")
                    for mismatch in mismatches:
                        print(f"- {mismatch}")
                    print(f"\nComplete expected results: {json.dumps(expected_results, indent=2)}")
                    print(f"\nComplete actual results: {json.dumps(test_results, indent=2)}")
                else:
                    print("All test results match expected results.")
        else:
            # Create expected results file if it doesn't exist
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2)
                print(f"Created new expected results file: {expected_file}")

        return test_results

if __name__ == "__main__":
    try:
        this_clap = test_hf_clap()
        results = this_clap.__test__()
        print(f"CLAP Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)