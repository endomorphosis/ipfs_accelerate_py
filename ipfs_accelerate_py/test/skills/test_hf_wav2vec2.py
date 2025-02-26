import os
import sys
import json
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from ...worker.skillset.hf_wav2vec2 import hf_wav2vec2, load_audio, load_audio_16khz

class test_hf_wav2vec2:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np,
            "transformers": MagicMock(),
            "librosa": MagicMock(),
            "soundfile": MagicMock()
        }
        self.metadata = metadata if metadata else {}
        self.wav2vec2 = hf_wav2vec2(resources=self.resources, metadata=self.metadata)
        self.model_name = "facebook/wav2vec2-base-960h"
        self.test_audio_url = "https://calamitymod.wiki.gg/images/2/29/Bees3.wav"
        return None

    def test(self):
        """Run all tests for the WAV2Vec2 ASR model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.wav2vec2 is not None else "Failed initialization"
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
                
                audio_16k, sr_16k = load_audio_16khz(self.test_audio_url)
                results["load_audio_16khz"] = "Success" if audio_16k is not None and sr_16k == 16000 else "Failed 16kHz conversion"
        except Exception as e:
            results["audio_utils"] = f"Error: {str(e)}"

        # Test CPU initialization and handler
        try:
            with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                 patch('transformers.AutoProcessor.from_pretrained') as mock_processor, \
                 patch('transformers.AutoModelForCTC.from_pretrained') as mock_model:
                
                mock_config.return_value = MagicMock()
                mock_processor.return_value = MagicMock()
                mock_model.return_value = MagicMock()
                
                endpoint, processor, handler, queue, batch_size = self.wav2vec2.init_cpu(
                    self.model_name,
                    "cpu",
                    "cpu"
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["cpu_init"] = "Success" if valid_init else "Failed CPU initialization"
                
                test_handler = self.wav2vec2.create_cpu_asr_endpoint_handler(
                    endpoint,
                    processor,
                    self.model_name,
                    "cpu"
                )
                
                # Test with audio input
                with patch('soundfile.read') as mock_sf_read:
                    mock_sf_read.return_value = (np.random.randn(16000), 16000)
                    output = test_handler(self.test_audio_url)
                    results["cpu_handler"] = "Success" if output is not None else "Failed CPU handler"
        except Exception as e:
            results["cpu_tests"] = f"Error: {str(e)}"

        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                     patch('transformers.AutoProcessor.from_pretrained') as mock_processor, \
                     patch('transformers.AutoModelForCTC.from_pretrained') as mock_model:
                    
                    mock_config.return_value = MagicMock()
                    mock_processor.return_value = MagicMock()
                    mock_model.return_value = MagicMock()
                    
                    endpoint, processor, handler, queue, batch_size = self.wav2vec2.init_cuda(
                        self.model_name,
                        "cuda",
                        "cuda:0"
                    )
                    
                    valid_init = endpoint is not None and processor is not None and handler is not None
                    results["cuda_init"] = "Success" if valid_init else "Failed CUDA initialization"
                    
                    test_handler = self.wav2vec2.create_cuda_asr_endpoint_handler(
                        endpoint,
                        processor,
                        self.model_name,
                        "cuda:0"
                    )
                    
                    with patch('soundfile.read') as mock_sf_read:
                        mock_sf_read.return_value = (np.random.randn(16000), 16000)
                        output = test_handler(self.test_audio_url)
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
                
                endpoint, processor, handler, queue, batch_size = self.wav2vec2.init_openvino(
                    self.model_name,
                    "automatic-speech-recognition",
                    "CPU",
                    "openvino:0",
                    mock_get_optimum_openvino_model,
                    mock_get_openvino_model,
                    mock_get_openvino_pipeline_type
                )
                
                valid_init = handler is not None
                results["openvino_init"] = "Success" if valid_init else "Failed OpenVINO initialization"
                
                test_handler = self.wav2vec2.create_openvino_asr_endpoint_handler(
                    endpoint,
                    processor,
                    self.model_name,
                    "openvino:0"
                )
                
                with patch('soundfile.read') as mock_sf_read:
                    mock_sf_read.return_value = (np.random.randn(16000), 16000)
                    output = test_handler(self.test_audio_url)
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
                    
                    endpoint, processor, handler, queue, batch_size = self.wav2vec2.init_apple(
                        self.model_name,
                        "mps",
                        "apple:0"
                    )
                    
                    valid_init = handler is not None
                    results["apple_init"] = "Success" if valid_init else "Failed Apple initialization"
                    
                    test_handler = self.wav2vec2.create_apple_asr_endpoint_handler(
                        endpoint,
                        processor,
                        self.model_name,
                        "apple:0"
                    )
                    
                    with patch('soundfile.read') as mock_sf_read:
                        mock_sf_read.return_value = (np.random.randn(16000), 16000)
                        output = test_handler(self.test_audio_url)
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
                
                endpoint, processor, handler, queue, batch_size = self.wav2vec2.init_qualcomm(
                    self.model_name,
                    "qualcomm",
                    "qualcomm:0"
                )
                
                valid_init = handler is not None
                results["qualcomm_init"] = "Success" if valid_init else "Failed Qualcomm initialization"
                
                test_handler = self.wav2vec2.create_qualcomm_asr_endpoint_handler(
                    endpoint,
                    processor,
                    self.model_name,
                    "qualcomm:0"
                )
                
                with patch('soundfile.read') as mock_sf_read:
                    mock_sf_read.return_value = (np.random.randn(16000), 16000)
                    output = test_handler(self.test_audio_url)
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
        with open(os.path.join(collected_dir, 'hf_wav2vec2_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_wav2vec2_test_results.json')
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
        this_wav2vec2 = test_hf_wav2vec2()
        results = this_wav2vec2.__test__()
        print(f"WAV2Vec2 Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)