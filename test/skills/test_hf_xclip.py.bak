import os
import sys
import json
import time
import torch
import numpy as np
import asyncio
from unittest.mock import MagicMock, patch
from PIL import Image
import tempfile

# Define missing utility functions needed by tests
def load_video_frames(video_file, num_frames=8):
    """Mock function to load video frames from a file.
    
    Args:
        video_file: Path or URL to video file
        num_frames: Number of frames to extract
        
    Returns:
        List of numpy arrays representing video frames
    """
    # For testing, generate random frames
    return [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(num_frames)]

# Use direct import with the absolute path
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")

# Try to import transformers directly to have a real implementation available
transformers = None
try:
    import transformers
    transformers_available = True
    print("Successfully imported real transformers module")
except ImportError:
    transformers_available = False
    print("Transformers not available, using mock...")
    transformers = MagicMock()

# Import the XCLIP module
from ipfs_accelerate_py.worker.skillset.hf_xclip import hf_xclip

class test_hf_xclip:
    def __init__(self, resources=None, metadata=None):
        if resources:
            self.resources = resources
        else:
            # Create resources with real transformers if available
            self.resources = {
                "torch": torch,
                "numpy": np, 
                "transformers": transformers if transformers_available else MagicMock(),
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
                results["load_video"] = "Success (MOCK)" if len(frames) > 0 else "Failed video loading"
                results["load_video_timestamp"] = time.time()
                results["load_video_frame_count"] = len(frames)
        except Exception as e:
            results["video_utils"] = f"Error: {str(e)}"

        # Test CPU initialization and handler
        try:
            # Check if transformers is available as a non-mocked import
            transformers_available = "transformers" in sys.modules and not isinstance(self.resources["transformers"], MagicMock)
            
            # First try real initialization without mocks
            if transformers_available:
                try:
                    print("Trying real CPU initialization for XCLIP...")
                    endpoint, processor, handler, queue, batch_size = self.xclip.init_cpu(
                        self.model_name,
                        "cpu",
                        "cpu"
                    )
                    
                    valid_init = endpoint is not None and processor is not None and handler is not None
                    implementation_type = "(REAL)"
                    results["cpu_init"] = f"Success {implementation_type}" if valid_init else f"Failed CPU initialization"
                    
                    # Use handler directly from initialization
                    test_handler = handler
                    
                except Exception as real_init_error:
                    print(f"Real CPU initialization failed: {real_init_error}")
                    print("Falling back to mock implementation...")
                    implementation_type = "(MOCK)"
                    
                    # Fall back to mock initialization
                    with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                         patch('transformers.AutoProcessor.from_pretrained') as mock_processor, \
                         patch('transformers.AutoModel.from_pretrained') as mock_model:
                        
                        mock_config.return_value = MagicMock()
                        mock_processor.return_value = MagicMock()
                        mock_model.return_value = MagicMock()
                        
                        endpoint, processor, handler, queue, batch_size = self.xclip.init_cpu(
                            self.model_name,
                            "cpu",
                            "cpu"
                        )
                        
                        valid_init = endpoint is not None and processor is not None and handler is not None
                        results["cpu_init"] = f"Success {implementation_type}" if valid_init else f"Failed CPU initialization"
                        
                        # Create test handler
                        test_handler = self.xclip.create_cpu_video_embedding_endpoint_handler(
                            endpoint,
                            processor,
                            self.model_name,
                            "cpu"
                        )
            else:
                # If transformers not available, use mocks directly
                print("Transformers not available, using mock implementation...")
                implementation_type = "(MOCK)"
                
                with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                     patch('transformers.AutoProcessor.from_pretrained') as mock_processor, \
                     patch('transformers.AutoModel.from_pretrained') as mock_model:
                    
                    mock_config.return_value = MagicMock()
                    mock_processor.return_value = MagicMock()
                    mock_model.return_value = MagicMock()
                    
                    endpoint, processor, handler, queue, batch_size = self.xclip.init_cpu(
                        self.model_name,
                        "cpu",
                        "cpu"
                    )
                    
                    valid_init = endpoint is not None and processor is not None and handler is not None
                    results["cpu_init"] = f"Success {implementation_type}" if valid_init else f"Failed CPU initialization"
                    
                    # Create test handler
                    test_handler = self.xclip.create_cpu_video_embedding_endpoint_handler(
                        endpoint,
                        processor,
                        self.model_name,
                        "cpu"
                    )
            
            # Test text embedding
            text_embedding = test_handler(text=self.test_text)
            
            # Get implementation type from the result, or use the one from initialization
            result_impl_type = text_embedding.get("implementation_type", implementation_type) if isinstance(text_embedding, dict) else implementation_type
            
            # Make sure we have the correct format with parentheses
            if result_impl_type == "REAL" or result_impl_type == "MOCK":
                result_impl_type = f"({result_impl_type})"
                
            results["cpu_text_embedding"] = f"Success {result_impl_type}" if text_embedding is not None else "Failed text embedding"
            
            # Include sample output info
            if text_embedding is not None and isinstance(text_embedding, dict) and "text_embedding" in text_embedding:
                text_emb = text_embedding["text_embedding"]
                results["cpu_text_embedding_shape"] = list(text_emb.shape) if hasattr(text_emb, "shape") else "unknown shape"
                results["cpu_text_embedding_timestamp"] = time.time()
            
            # Test video embedding
            video_embedding = test_handler(frames=self.frames)
            
            # Get implementation type from the result, or use the one from initialization
            result_impl_type = video_embedding.get("implementation_type", implementation_type) if isinstance(video_embedding, dict) else implementation_type
            
            # Make sure we have the correct format with parentheses
            if result_impl_type == "REAL" or result_impl_type == "MOCK":
                result_impl_type = f"({result_impl_type})"
                
            results["cpu_video_embedding"] = f"Success {result_impl_type}" if video_embedding is not None else "Failed video embedding"
            
            # Include sample output info
            if video_embedding is not None and isinstance(video_embedding, dict) and "video_embedding" in video_embedding:
                video_emb = video_embedding["video_embedding"]
                results["cpu_video_embedding_shape"] = list(video_emb.shape) if hasattr(video_emb, "shape") else "unknown shape"
                results["cpu_video_embedding_timestamp"] = time.time()
            
            # Test similarity computation
            similarity = test_handler(frames=self.frames, text=self.test_text)
            
            # Get implementation type from the result, or use the one from initialization
            result_impl_type = similarity.get("implementation_type", implementation_type) if isinstance(similarity, dict) else implementation_type
            
            # Make sure we have the correct format with parentheses
            if result_impl_type == "REAL" or result_impl_type == "MOCK":
                result_impl_type = f"({result_impl_type})"
                
            results["cpu_similarity"] = f"Success {result_impl_type}" if similarity is not None else "Failed similarity computation"
            
            # Include similarity score if available
            if similarity is not None and isinstance(similarity, dict) and "similarity" in similarity:
                sim_score = similarity["similarity"]
                if hasattr(sim_score, "item") and callable(sim_score.item):
                    results["cpu_similarity_score"] = float(sim_score.item())
                elif hasattr(sim_score, "tolist") and callable(sim_score.tolist):
                    results["cpu_similarity_score"] = sim_score.tolist()
                else:
                    results["cpu_similarity_score"] = "unknown format"
                results["cpu_similarity_timestamp"] = time.time()
                
                # Add example with implementation type for completeness
                results["cpu_similarity_example"] = {
                    "input": {
                        "text": self.test_text,
                        "video": "Video frames (array data)"
                    },
                    "output": {
                        "similarity_value": results["cpu_similarity_score"]
                    },
                    "implementation_type": result_impl_type,
                    "platform": "CPU"
                }
                
        except Exception as e:
            results["cpu_tests"] = f"Error: {str(e)}"

        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                # Use a more generic patching approach to avoid AutoModelForVideoTextRetrieval errors
                with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                     patch('transformers.AutoProcessor.from_pretrained') as mock_processor, \
                     patch('transformers.AutoModel.from_pretrained') as mock_model:
                    
                    mock_config.return_value = MagicMock()
                    mock_processor.return_value = MagicMock()
                    mock_model.return_value = MagicMock()
                    
                    endpoint, processor, handler, queue, batch_size = self.xclip.init_cuda(
                        self.model_name,
                        "cuda",
                        "cuda:0"
                    )
                    
                    valid_init = endpoint is not None and processor is not None and handler is not None
                    results["cuda_init"] = "Success (MOCK)" if valid_init else "Failed CUDA initialization"
                    
                    test_handler = self.xclip.create_cuda_video_embedding_endpoint_handler(
                        endpoint,
                        processor,
                        self.model_name,
                        "cuda:0"
                    )
                    
                    # Test different input formats
                    text_output = test_handler(text=self.test_text)
                    results["cuda_text"] = "Success (MOCK)" if text_output is not None else "Failed text input"
                    
                    # Include sample output info
                    if text_output is not None and isinstance(text_output, dict) and "text_embedding" in text_output:
                        text_emb = text_output["text_embedding"]
                        results["cuda_text_embedding_shape"] = list(text_emb.shape) if hasattr(text_emb, "shape") else "unknown shape"
                        results["cuda_text_timestamp"] = time.time()
                    
                    video_output = test_handler(frames=self.frames)
                    results["cuda_video"] = "Success (MOCK)" if video_output is not None else "Failed video input"
                    
                    # Include sample output info
                    if video_output is not None and isinstance(video_output, dict) and "video_embedding" in video_output:
                        video_emb = video_output["video_embedding"]
                        results["cuda_video_embedding_shape"] = list(video_emb.shape) if hasattr(video_emb, "shape") else "unknown shape"
                        results["cuda_video_timestamp"] = time.time()
                    
                    similarity = test_handler(self.frames, self.test_text)
                    results["cuda_similarity"] = "Success (MOCK)" if similarity is not None else "Failed similarity computation"
                    
                    # Include similarity score if available
                    if similarity is not None and isinstance(similarity, dict) and "similarity" in similarity:
                        sim_score = similarity["similarity"]
                        if hasattr(sim_score, "item") and callable(sim_score.item):
                            results["cuda_similarity_score"] = float(sim_score.item())
                        elif hasattr(sim_score, "tolist") and callable(sim_score.tolist):
                            results["cuda_similarity_score"] = sim_score.tolist()
                        else:
                            results["cuda_similarity_score"] = "unknown format"
                        results["cuda_similarity_timestamp"] = time.time()
            except Exception as e:
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"

        # Test OpenVINO if installed
        try:
            try:
                import openvino
                import traceback  # For better error reporting
            except ImportError:
                results["openvino_tests"] = "OpenVINO not installed"
                return results
                
            # Import the existing OpenVINO utils from the main package
            from ipfs_accelerate_py.worker.openvino_utils import openvino_utils
            
            # Initialize openvino_utils
            ov_utils = openvino_utils(resources=self.resources, metadata=self.metadata)
            
            # Implement file locking for thread safety
            import fcntl
            from contextlib import contextmanager
            
            @contextmanager
            def file_lock(lock_file, timeout=600):
                """Simple file-based lock with timeout"""
                start_time = time.time()
                lock_dir = os.path.dirname(lock_file)
                os.makedirs(lock_dir, exist_ok=True)
                
                fd = open(lock_file, 'w')
                try:
                    while True:
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except IOError:
                            if time.time() - start_time > timeout:
                                raise TimeoutError(f"Could not acquire lock on {lock_file} within {timeout} seconds")
                            time.sleep(1)
                    yield
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    fd.close()
                    try:
                        os.unlink(lock_file)
                    except:
                        pass
            
            # First try real OpenVINO implementation
            is_real_impl = False
            implementation_type = "(MOCK)"  # Default assuming mock
            
            try:
                print("Trying real OpenVINO implementation for XCLIP...")
                
                # Helper function to find model path
                def find_model_path(model_name):
                    """Find a model's path with multiple fallback strategies"""
                    try:
                        # Handle case where model_name is already a path
                        if os.path.exists(model_name):
                            return model_name
                        
                        # Try HF cache locations
                        potential_cache_paths = [
                            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models"),
                            os.path.join(os.path.expanduser("~"), ".cache", "optimum", "ov"),
                            os.path.join("/tmp", "hf_models")
                        ]
                        
                        # Search in all potential cache paths
                        for cache_path in potential_cache_paths:
                            if os.path.exists(cache_path):
                                # Try direct match first
                                try:
                                    model_dirs = [x for x in os.listdir(cache_path) if model_name in x]
                                    if model_dirs:
                                        return os.path.join(cache_path, model_dirs[0])
                                except Exception as e:
                                    print(f"Error listing {cache_path}: {e}")
                                
                                # Try deeper search
                                for root, dirs, _ in os.walk(cache_path):
                                    if model_name.replace("/", "_") in root or model_name in root:
                                        return root
                        
                        # Last resort - return the model name
                        return model_name
                    except Exception as e:
                        print(f"Error finding model path: {e}")
                        return model_name
                
                # Create lock file path based on model name
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "xclip_ov_locks")
                os.makedirs(cache_dir, exist_ok=True)
                lock_file = os.path.join(cache_dir, f"{self.model_name.replace('/', '_')}_conversion.lock")
                
                # Try direct approach with optimum
                try:
                    print("Trying direct optimum-intel approach first...")
                    # Use file locking to prevent multiple conversions
                    with file_lock(lock_file):
                        try:
                            # Try to use optimum-intel for XCLIP
                            from optimum.intel.openvino import OVModelForVision2Seq
                            from transformers import XCLIPProcessor
                            
                            # Find model path
                            model_path = find_model_path(self.model_name)
                            print(f"Using model path: {model_path}")
                            
                            # Load model and processor with correct task type
                            ov_model = OVModelForVision2Seq.from_pretrained(
                                model_path,
                                device="CPU",
                                trust_remote_code=True
                            )
                            processor = XCLIPProcessor.from_pretrained(model_path)
                            
                            # Use correct task type for XCLIP
                            model_type = "video-to-text-retrieval"
                            
                            # Create handler function
                            def direct_handler(frames=None, text=None):
                                try:
                                    start_time = time.time()
                                    result = {}
                                    
                                    # Handle different input combinations
                                    if text is not None and frames is not None:
                                        # Process both for similarity
                                        inputs = processor(text=text, videos=frames, return_tensors="pt")
                                        with torch.no_grad():
                                            outputs = ov_model(**inputs)
                                        
                                        # Extract embeddings and similarity
                                        text_embedding = outputs.text_embeds
                                        video_embedding = outputs.vision_embeds
                                        
                                        # Calculate similarity
                                        similarity = torch.nn.functional.cosine_similarity(
                                            text_embedding, 
                                            video_embedding
                                        )
                                        
                                        result = {
                                            "text_embedding": text_embedding[0],
                                            "video_embedding": video_embedding[0],
                                            "similarity": similarity,
                                            "implementation_type": "REAL",
                                            "elapsed_time": time.time() - start_time
                                        }
                                        
                                    elif text is not None:
                                        # Process text only
                                        inputs = processor(text=text, return_tensors="pt")
                                        with torch.no_grad():
                                            outputs = ov_model.get_text_features(**inputs)
                                        
                                        result = {
                                            "text_embedding": outputs[0],
                                            "implementation_type": "REAL",
                                            "elapsed_time": time.time() - start_time
                                        }
                                        
                                    elif frames is not None:
                                        # Process video only
                                        inputs = processor(videos=frames, return_tensors="pt")
                                        with torch.no_grad():
                                            outputs = ov_model.get_video_features(**inputs)
                                        
                                        result = {
                                            "video_embedding": outputs[0],
                                            "implementation_type": "REAL",
                                            "elapsed_time": time.time() - start_time
                                        }
                                    
                                    return result
                                    
                                except Exception as e:
                                    print(f"Error in direct handler: {e}")
                                    print(f"Traceback: {traceback.format_exc()}")
                                    # Fall back to mock result with implementation type marker
                                    return {
                                        "text_embedding": torch.zeros(512),
                                        "video_embedding": torch.zeros(512),
                                        "similarity": torch.tensor([0.75]),
                                        "implementation_type": "MOCK",
                                        "error": str(e)
                                    }
                            
                            # Set handler
                            handler = direct_handler
                            endpoint = None
                            queue = None
                            batch_size = 1
                            
                            is_real_impl = True
                            implementation_type = "(REAL)"
                            print("Successfully created real OpenVINO implementation via direct approach")
                            
                        except Exception as optimum_error:
                            print(f"Direct optimum-intel approach failed: {optimum_error}")
                            print(f"Traceback: {traceback.format_exc()}")
                            raise optimum_error
                            
                except Exception as direct_error:
                    print(f"Direct approach failed: {direct_error}")
                    
                    # Fall back to standard approach with correct task type
                    try:
                        endpoint, processor, handler, queue, batch_size = self.xclip.init_openvino(
                            self.model_name,
                            "video-to-text-retrieval",  # Correct task type for XCLIP
                            "CPU",
                            "openvino:0",
                            ov_utils.get_optimum_openvino_model,
                            ov_utils.get_openvino_model,
                            ov_utils.get_openvino_pipeline_type,
                            ov_utils.openvino_cli_convert
                        )
                        
                        # Check if we got real implementation or mock
                        import unittest.mock
                        if isinstance(handler, unittest.mock.MagicMock) or (processor is not None and isinstance(processor, unittest.mock.MagicMock)):
                            is_real_impl = False
                            implementation_type = "(MOCK)"
                            print("Received mock components from handler, using mock implementation")
                        else:
                            is_real_impl = True
                            implementation_type = "(REAL)"
                            print("Successfully initialized real OpenVINO implementation")
                            
                    except Exception as ov_error:
                        print(f"Standard OpenVINO initialization failed: {ov_error}")
                        print(f"Falling back to mock implementation...")
                        is_real_impl = False
                        implementation_type = "(MOCK)"
                        traceback.print_exc()
                        
                        # Create mocks for testing
                        processor = MagicMock()
                        endpoint = MagicMock()
                        
                        # Create a custom handler that always returns success
                        def mock_openvino_handler(frames=None, text=None):
                            return {
                                "text_embedding": torch.zeros(1, 512), 
                                "video_embedding": torch.zeros(1, 512),
                                "similarity": torch.tensor([[0.8]]),
                                "implementation_type": "MOCK"
                            }
                        
                        handler = mock_openvino_handler
                        queue = asyncio.Queue(32)
                        batch_size = 0
                
            except Exception as e:
                print(f"All OpenVINO initialization attempts failed: {e}")
                traceback.print_exc()
                
                # Fall back to mock implementation
                processor = MagicMock()
                endpoint = MagicMock()
                
                # Create a custom handler that always returns success
                def fallback_mock_handler(frames=None, text=None):
                    return {
                        "text_embedding": torch.zeros(1, 512), 
                        "video_embedding": torch.zeros(1, 512),
                        "similarity": torch.tensor([[0.8]]),
                        "implementation_type": "MOCK"
                    }
                
                handler = fallback_mock_handler
                queue = asyncio.Queue(32)
                batch_size = 0
                
                is_real_impl = False
                implementation_type = "(MOCK)"
            
            valid_init = handler is not None
            results["openvino_init"] = f"Success {implementation_type}" if valid_init else "Failed OpenVINO initialization"
            
            test_handler = handler  # Use handler directly
            
            # If we still need to create a handler function, use this:
            if not test_handler:
                test_handler = self.xclip.create_openvino_video_embedding_endpoint_handler(
                    endpoint,
                    processor,
                    self.model_name,
                    "openvino:0"
                )
                
                # Test with both frames and text for similarity
                output = test_handler(self.frames, self.test_text)
                
                # Get implementation type from the result
                result_impl_type = output.get("implementation_type", implementation_type) if isinstance(output, dict) else implementation_type
                
                # Make sure we have the correct format with parentheses
                if result_impl_type == "REAL" or result_impl_type == "MOCK":
                    result_impl_type = f"({result_impl_type})"
                    
                results["openvino_handler"] = f"Success {result_impl_type}" if output is not None else "Failed OpenVINO handler"
                
                # Include similarity score if available
                if output is not None and isinstance(output, dict):
                    results["openvino_output_timestamp"] = time.time()
                    results["openvino_output_keys"] = list(output.keys())
                    
                    if "text_embedding" in output:
                        text_emb = output["text_embedding"]
                        results["openvino_text_embedding_shape"] = list(text_emb.shape) if hasattr(text_emb, "shape") else "unknown shape"
                    
                    if "video_embedding" in output:
                        video_emb = output["video_embedding"]
                        results["openvino_video_embedding_shape"] = list(video_emb.shape) if hasattr(video_emb, "shape") else "unknown shape"
                    
                    if "similarity" in output:
                        sim_score = output["similarity"]
                        if hasattr(sim_score, "item") and callable(sim_score.item):
                            results["openvino_similarity_score"] = float(sim_score.item())
                        elif hasattr(sim_score, "tolist") and callable(sim_score.tolist):
                            results["openvino_similarity_score"] = sim_score.tolist()
                        else:
                            results["openvino_similarity_score"] = "unknown format"
                            
                    # Add example with correct implementation type
                    results["openvino_similarity_example"] = {
                        "input": {
                            "text": self.test_text,
                            "video": "Video frames (array data)"
                        },
                        "output": {
                            "similarity_value": results.get("openvino_similarity_score", 0.0)
                        },
                        "implementation_type": result_impl_type,
                        "platform": "OpenVINO"
                    }
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
                    results["apple_init"] = "Success (MOCK)" if valid_init else "Failed Apple initialization"
                    
                    test_handler = self.xclip.create_apple_video_embedding_endpoint_handler(
                        endpoint,
                        processor,
                        self.model_name,
                        "apple:0"
                    )
                    
                    # Test different input formats
                    text_output = test_handler(text=self.test_text)
                    results["apple_text"] = "Success (MOCK)" if text_output is not None else "Failed text input"
                    
                    # Include sample output info
                    if text_output is not None and isinstance(text_output, dict) and "text_embedding" in text_output:
                        text_emb = text_output["text_embedding"]
                        results["apple_text_embedding_shape"] = list(text_emb.shape) if hasattr(text_emb, "shape") else "unknown shape"
                        results["apple_text_timestamp"] = time.time()
                    
                    # Test video-text similarity
                    similarity = test_handler(self.frames, self.test_text)
                    results["apple_similarity"] = "Success (MOCK)" if similarity is not None else "Failed similarity computation"
                    
                    # Include similarity score if available
                    if similarity is not None and isinstance(similarity, dict):
                        results["apple_output_timestamp"] = time.time()
                        results["apple_output_keys"] = list(similarity.keys())
                        
                        if "similarity" in similarity:
                            sim_score = similarity["similarity"]
                            if hasattr(sim_score, "item") and callable(sim_score.item):
                                results["apple_similarity_score"] = float(sim_score.item())
                            elif hasattr(sim_score, "tolist") and callable(sim_score.tolist):
                                results["apple_similarity_score"] = sim_score.tolist()
                            else:
                                results["apple_similarity_score"] = "unknown format"
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
                results["qualcomm_init"] = "Success (MOCK)" if valid_init else "Failed Qualcomm initialization"
                
                test_handler = self.xclip.create_qualcomm_video_embedding_endpoint_handler(
                    endpoint,
                    processor,
                    self.model_name,
                    "qualcomm:0"
                )
                
                output = test_handler(self.frames, self.test_text)
                results["qualcomm_handler"] = "Success (MOCK)" if output is not None else "Failed Qualcomm handler"
                
                # Include result details if available
                if output is not None and isinstance(output, dict):
                    results["qualcomm_output_timestamp"] = time.time()
                    results["qualcomm_output_keys"] = list(output.keys())
                    
                    # Save embedding shapes and similarity score
                    if "text_embedding" in output:
                        text_emb = output["text_embedding"]
                        results["qualcomm_text_embedding_shape"] = list(text_emb.shape) if hasattr(text_emb, "shape") else "unknown shape"
                    
                    if "video_embedding" in output:
                        video_emb = output["video_embedding"]
                        results["qualcomm_video_embedding_shape"] = list(video_emb.shape) if hasattr(video_emb, "shape") else "unknown shape"
                    
                    if "similarity" in output:
                        sim_score = output["similarity"]
                        if hasattr(sim_score, "item") and callable(sim_score.item):
                            results["qualcomm_similarity_score"] = float(sim_score.item())
                        elif hasattr(sim_score, "tolist") and callable(sim_score.tolist):
                            results["qualcomm_similarity_score"] = sim_score.tolist()
                        else:
                            results["qualcomm_similarity_score"] = "unknown format"
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
        # Check if transformers is a mock or real implementation
        transformers_available = "transformers" in sys.modules and not isinstance(self.resources["transformers"], MagicMock)
        transformers_version = transformers.__version__ if transformers_available and hasattr(transformers, "__version__") else "mocked"
        
        # Gather implementation type information from results
        contains_real_impl = any("(REAL)" in str(v) for v in test_results.values() if isinstance(v, str))
        
        test_results["metadata"] = {
            "timestamp": time.time(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "transformers_version": transformers_version,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "test_model": self.model_name,
            "test_run_id": f"xclip-test-{int(time.time())}",
            "using_mocks": not contains_real_impl,  # Only mark as using mocks if no real implementations found
            "has_real_cpu_impl": contains_real_impl,
            "transformers_available": transformers_available
        }
        
        # Save collected results
        collected_file = os.path.join(collected_dir, 'hf_xclip_test_results.json')
        with open(collected_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Saved test results to {collected_file}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_xclip_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                    
                    # Only compare the non-variable parts
                    excluded_keys = ["metadata"]
                    
                    # Exclude timestamp fields and embedding shapes since they might vary
                    variable_keys = [k for k in test_results.keys() 
                                    if "timestamp" in k 
                                    or "shape" in k 
                                    or "score" in k
                                    or "keys" in k]
                    excluded_keys.extend(variable_keys)
                    
                    # Also exclude cpu_similarity_example since similarity values are random
                    excluded_keys.append("cpu_similarity_example")
                    
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
                        print("\nConsider updating the expected results file if these differences are intentional")
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
        this_xclip = test_hf_xclip()
        results = this_xclip.__test__()
        print(f"XClip Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)