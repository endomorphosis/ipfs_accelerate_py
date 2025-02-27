import os
import sys
import json
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image

# Use direct import with the absolute path
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")
from ipfs_accelerate_py.worker.skillset.hf_xclip import hf_xclip

class test_hf_xclip:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np, 
            "transformers": MagicMock(),
            "decord": MagicMock()
        }
        self.metadata = metadata if metadata else {}
        self.xclip = hf_xclip(resources=self.resources, metadata=self.metadata)
        self.model_name = "microsoft/xclip-base-patch32"
        self.test_text = "A person dancing"
        # Create a dummy video as a sequence of frames
        self.frames = [Image.new('RGB', (224, 224), color='red') for _ in range(8)]
        self.test_video_url = "http://example.com/test.mp4"
        return None

    def test(self):
        """Run all tests for the XClip video-language model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.xclip is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"

        # Test video loading utilities
        try:
            with patch('decord.VideoReader') as mock_video_reader, \
                 patch('requests.get') as mock_get:
                mock_response = MagicMock()
                mock_response.content = b"fake_video_data"
                mock_get.return_value = mock_response
                mock_video_reader.return_value = MagicMock()
                mock_video_reader.return_value.__len__.return_value = 30
                mock_video_reader.return_value.__getitem__.return_value = np.random.randn(224, 224, 3)
                
                frames = load_video_frames(self.test_video_url)
                results["load_video"] = "Success" if len(frames) > 0 else "Failed video loading"
        except Exception as e:
            results["video_utils"] = f"Error: {str(e)}"

        # Test CPU initialization and handler
        try:
            with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                 patch('transformers.AutoProcessor.from_pretrained') as mock_processor, \
                 patch('transformers.AutoModelForVideoTextRetrieval.from_pretrained') as mock_model:
                
                mock_config.return_value = MagicMock()
                mock_processor.return_value = MagicMock()
                mock_model.return_value = MagicMock()
                
                endpoint, processor, handler, queue, batch_size = self.xclip.init_cpu(
                    self.model_name,
                    "cpu",
                    "cpu"
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["cpu_init"] = "Success" if valid_init else "Failed CPU initialization"
                
                test_handler = self.xclip.create_cpu_video_embedding_endpoint_handler(
                    endpoint,
                    processor,
                    self.model_name,
                    "cpu"
                )
                
                # Test text embedding
                text_embedding = test_handler(text=self.test_text)
                results["cpu_text_embedding"] = "Success" if text_embedding is not None else "Failed text embedding"
                
                # Test video embedding
                video_embedding = test_handler(frames=self.frames)
                results["cpu_video_embedding"] = "Success" if video_embedding is not None else "Failed video embedding"
                
        except Exception as e:
            results["cpu_tests"] = f"Error: {str(e)}"

        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                     patch('transformers.AutoProcessor.from_pretrained') as mock_processor, \
                     patch('transformers.AutoModelForVideoTextRetrieval.from_pretrained') as mock_model:
                    
                    mock_config.return_value = MagicMock()
                    mock_processor.return_value = MagicMock()
                    mock_model.return_value = MagicMock()
                    
                    endpoint, processor, handler, queue, batch_size = self.xclip.init_cuda(
                        self.model_name,
                        "cuda",
                        "cuda:0"
                    )
                    
                    valid_init = endpoint is not None and processor is not None and handler is not None
                    results["cuda_init"] = "Success" if valid_init else "Failed CUDA initialization"
                    
                    test_handler = self.xclip.create_cuda_video_embedding_endpoint_handler(
                        endpoint,
                        processor,
                        self.model_name,
                        "cuda:0"
                    )
                    
                    # Test different input formats
                    text_output = test_handler(text=self.test_text)
                    results["cuda_text"] = "Success" if text_output is not None else "Failed text input"
                    
                    video_output = test_handler(frames=self.frames)
                    results["cuda_video"] = "Success" if video_output is not None else "Failed video input"
                    
                    similarity = test_handler(self.frames, self.test_text)
                    results["cuda_similarity"] = "Success" if similarity is not None else "Failed similarity computation"
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
                
                endpoint, processor, handler, queue, batch_size = self.xclip.init_openvino(
                    self.model_name,
                    "video-classification",
                    "CPU",
                    "openvino:0",
                    ov_utils.get_optimum_openvino_model,
                    ov_utils.get_openvino_model,
                    ov_utils.get_openvino_pipeline_type,
                    ov_utils.openvino_cli_convert
                )
                
                valid_init = handler is not None
                results["openvino_init"] = "Success" if valid_init else "Failed OpenVINO initialization"
                
                test_handler = self.xclip.create_openvino_video_embedding_endpoint_handler(
                    endpoint,
                    processor,
                    self.model_name,
                    "openvino:0"
                )
                
                output = test_handler(self.frames, self.test_text)
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
                    
                    endpoint, processor, handler, queue, batch_size = self.xclip.init_apple(
                        self.model_name,
                        "mps",
                        "apple:0"
                    )
                    
                    valid_init = handler is not None
                    results["apple_init"] = "Success" if valid_init else "Failed Apple initialization"
                    
                    test_handler = self.xclip.create_apple_video_embedding_endpoint_handler(
                        endpoint,
                        processor,
                        self.model_name,
                        "apple:0"
                    )
                    
                    # Test different input formats
                    text_output = test_handler(text=self.test_text)
                    results["apple_text"] = "Success" if text_output is not None else "Failed text input"
                    
                    similarity = test_handler(self.frames, self.test_text)
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
                
                endpoint, processor, handler, queue, batch_size = self.xclip.init_qualcomm(
                    self.model_name,
                    "qualcomm",
                    "qualcomm:0"
                )
                
                valid_init = handler is not None
                results["qualcomm_init"] = "Success" if valid_init else "Failed Qualcomm initialization"
                
                test_handler = self.xclip.create_qualcomm_video_embedding_endpoint_handler(
                    endpoint,
                    processor,
                    self.model_name,
                    "qualcomm:0"
                )
                
                output = test_handler(self.frames, self.test_text)
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
        with open(os.path.join(collected_dir, 'hf_xclip_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_xclip_test_results.json')
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
        this_xclip = test_hf_xclip()
        results = this_xclip.__test__()
        print(f"XClip Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)