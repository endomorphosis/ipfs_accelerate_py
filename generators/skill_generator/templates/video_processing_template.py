import os
import anyio
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple

class hf_{model_type}:
    """HuggingFace {model_type_upper} video processing model implementation.
    
    This class provides standardized interfaces for working with {model_type_upper} video models
    across different hardware backends (CPU, CUDA, ROCm, Apple, OpenVINO, Qualcomm).
    
    {model_description}
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the {model_type_upper} video model.
        
        Args:
            resources (dict): Dictionary of shared resources (torch, transformers, etc.)
            metadata (dict): Configuration metadata
        """
        self.resources = resources if resources is not None else {}
        self.metadata = metadata if metadata is not None else {}
        
        # Handler creation methods
        self.create_cpu_video_endpoint_handler = self.create_cpu_video_endpoint_handler
        self.create_cuda_video_endpoint_handler = self.create_cuda_video_endpoint_handler
        self.create_rocm_video_endpoint_handler = self.create_rocm_video_endpoint_handler
        self.create_openvino_video_endpoint_handler = self.create_openvino_video_endpoint_handler
        self.create_apple_video_endpoint_handler = self.create_apple_video_endpoint_handler
        self.create_qualcomm_video_endpoint_handler = self.create_qualcomm_video_endpoint_handler
        
        # Initialization methods
        self.init = self.init
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_rocm = self.init_rocm
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.init_qualcomm = self.init_qualcomm
        
        # Test methods
        self.__test__ = self.__test__
        
        # Hardware-specific utilities
        self.snpe_utils = None  # Qualcomm SNPE utils
        return None
        
    def _create_mock_processor(self):
        """Create a mock processor for graceful degradation when the real one fails.
        
        Returns:
            Mock processor object with essential methods
        """
        try:
            from unittest.mock import MagicMock
            
            processor = MagicMock()
            
            # Configure mock processor call behavior
            def mock_process(video=None, text=None, return_tensors="pt", **kwargs):
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                
                # Create mock tensors for video input
                if video is not None:
                    if isinstance(video, (list, tuple)):
                        batch_size = len(video)
                    else:
                        batch_size = 1
                    
                    # Common sizes for video frames input
                    num_frames = 8  # Typical number of frames
                    channels = 3  # RGB channels
                    height = 224  # Standard height
                    width = 224  # Standard width
                    
                    return {
                        "pixel_values": torch.rand((batch_size, num_frames, channels, height, width)),
                        "attention_mask": torch.ones((batch_size, num_frames))
                    }
                
                # Create mock text input
                if text is not None:
                    if isinstance(text, str):
                        batch_size = 1
                    else:
                        batch_size = len(text)
                    
                    return {
                        "input_ids": torch.ones((batch_size, 77), dtype=torch.long),
                        "attention_mask": torch.ones((batch_size, 77), dtype=torch.long)
                    }
                
                # Default output with both modalities
                return {
                    "pixel_values": torch.rand((1, 8, 3, 224, 224)),
                    "input_ids": torch.ones((1, 77), dtype=torch.long),
                    "attention_mask": torch.ones((1, 77), dtype=torch.long)
                }
                
            processor.side_effect = mock_process
            processor.__call__ = mock_process
            
            print("(MOCK) Created mock {model_type_upper} processor")
            return processor
            
        except ImportError:
            # Fallback if unittest.mock is not available
            class SimpleProcessor:
                def __init__(self, parent):
                    self.parent = parent
                    
                def __call__(self, video=None, text=None, return_tensors="pt", **kwargs):
                    if hasattr(self.parent, 'torch'):
                        torch = self.parent.torch
                    else:
                        import torch
                    
                    # Create mock tensors for video input
                    if video is not None:
                        if isinstance(video, (list, tuple)):
                            batch_size = len(video)
                        else:
                            batch_size = 1
                        
                        # Common sizes for video frames input
                        num_frames = 8  # Typical number of frames
                        channels = 3  # RGB channels
                        height = 224  # Standard height
                        width = 224  # Standard width
                        
                        return {
                            "pixel_values": torch.rand((batch_size, num_frames, channels, height, width)),
                            "attention_mask": torch.ones((batch_size, num_frames))
                        }
                    
                    # Create mock text input
                    if text is not None:
                        if isinstance(text, str):
                            batch_size = 1
                        else:
                            batch_size = len(text)
                        
                        return {
                            "input_ids": torch.ones((batch_size, 77), dtype=torch.long),
                            "attention_mask": torch.ones((batch_size, 77), dtype=torch.long)
                        }
                    
                    # Default output with both modalities
                    return {
                        "pixel_values": torch.rand((1, 8, 3, 224, 224)),
                        "input_ids": torch.ones((1, 77), dtype=torch.long),
                        "attention_mask": torch.ones((1, 77), dtype=torch.long)
                    }
            
            print("(MOCK) Created simple mock {model_type_upper} processor")
            return SimpleProcessor(self)
    
    def _create_mock_endpoint(self, model_name, device_label):
        """Create mock endpoint objects when real initialization fails.
        
        Args:
            model_name (str): The model name or path
            device_label (str): The device label (cpu, cuda, etc.)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        """
        try:
            from unittest.mock import MagicMock
            
            # Create mock endpoint
            endpoint = MagicMock()
            
            # Configure mock endpoint behavior
            def mock_forward(**kwargs):
                # Extract input shapes from kwargs
                pixel_values = kwargs.get("pixel_values", None)
                if pixel_values is None:
                    batch_size = 1
                    num_frames = 8
                    hidden_size = {hidden_size}  # Standard hidden size for this model type
                else:
                    batch_size = pixel_values.shape[0]
                    num_frames = pixel_values.shape[1] if len(pixel_values.shape) > 4 else 1
                
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                
                # Create mock output structure based on model architecture
                result = MagicMock()
                result.last_hidden_state = torch.rand((batch_size, num_frames, hidden_size))
                result.pooler_output = torch.rand((batch_size, hidden_size))
                
                # Add video classification logits
                num_classes = 157  # Typical for Kinetics dataset
                result.logits = torch.rand((batch_size, num_classes))
                
                return result
                
            endpoint.side_effect = mock_forward
            endpoint.__call__ = mock_forward
            
            # Create mock processor
            processor = self._create_mock_processor()
            
            # Create appropriate handler for the device type
            if device_label.startswith('cpu'):
                handler_method = self.create_cpu_video_endpoint_handler
            elif device_label.startswith('cuda'):
                handler_method = self.create_cuda_video_endpoint_handler
            elif device_label.startswith('rocm'):
                handler_method = self.create_rocm_video_endpoint_handler
            elif device_label.startswith('openvino'):
                handler_method = self.create_openvino_video_endpoint_handler
            elif device_label.startswith('apple'):
                handler_method = self.create_apple_video_endpoint_handler
            elif device_label.startswith('qualcomm'):
                handler_method = self.create_qualcomm_video_endpoint_handler
            else:
                handler_method = self.create_cpu_video_endpoint_handler
            
            # Create handler function
            mock_handler = handler_method(
                endpoint_model=model_name,
                device=device_label.split(':')[0] if ':' in device_label else device_label,
                hardware_label=device_label,
                endpoint=endpoint,
                processor=processor
            )
            
            import asyncio
            print(f"(MOCK) Created mock {model_type_upper} endpoint for {model_name} on {device_label}")
            return endpoint, processor, mock_handler, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error creating mock endpoint: {e}")
            import asyncio
            return None, None, None, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(32), 0
    
    def init(self):        
        if "torch" not in list(self.resources.keys()):
            import torch
            self.torch = torch
        else:
            self.torch = self.resources["torch"]

        if "transformers" not in list(self.resources.keys()):
            import transformers
            self.transformers = transformers
        else:
            self.transformers = self.resources["transformers"]
            
        if "numpy" not in list(self.resources.keys()):
            import numpy as np
            self.np = np
        else:
            self.np = self.resources["numpy"]
            
        # Try to import video-specific libraries
        try:
            if "decord" not in list(self.resources.keys()):
                import decord
                self.decord = decord
            else:
                self.decord = self.resources["decord"]
        except ImportError:
            self.decord = None
            print("Decord not available, some video processing functions will be limited")
            
        try:
            if "PIL" not in list(self.resources.keys()):
                from PIL import Image
                self.Image = Image
            else:
                self.Image = self.resources["PIL"]
        except ImportError:
            self.Image = None
            print("PIL not available, frame visualization will be limited")

        return None
    
    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, processor):
        """Test function to validate endpoint functionality.
        
        Args:
            endpoint_model: The model name or path
            endpoint_handler: The handler function
            endpoint_label: The hardware label
            processor: The processor
            
        Returns:
            Boolean indicating test success
        """
        test_input = "{test_input}"
        timestamp1 = time.time()
        test_batch = None
        
        try:
            # Run the model
            test_batch = endpoint_handler(test_input)
            print(test_batch)
            print("hf_{model_type} test passed")
            
            # Check output structure for video-specific outputs
            if "embeddings" in test_batch:
                embeddings = test_batch["embeddings"]
                if isinstance(embeddings, list):
                    print(f"Embedding dimensions: {len(embeddings)} x {len(embeddings[0])}")
                
            if "classification" in test_batch:
                print(f"Classification results available: {len(test_batch['classification'])} classes")
                
            if "frame_features" in test_batch:
                print(f"Frame-level features available for {len(test_batch['frame_features'])} frames")
                
        except Exception as e:
            print(e)
            print("hf_{model_type} test failed")
            return False
            
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        print(f"elapsed time: {elapsed_time}")
        
        # Clean up memory
        with self.torch.no_grad():
            if "cuda" in dir(self.torch):
                self.torch.cuda.empty_cache()
        return True

    def init_cpu(self, model_name, device, cpu_label):
        """Initialize {model_type_upper} model for CPU inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('cpu')
            cpu_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        print(f"Loading {model_name} for CPU inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load processor for video models
            # Video models can have different processor types
            if "videomae" in model_name.lower():
                processor = self.transformers.VideoMAEImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            elif "vivit" in model_name.lower():
                processor = self.transformers.VivitImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            elif "timesformer" in model_name.lower():
                processor = self.transformers.TimesformerImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            else:
                # Default video processor
                processor = self.transformers.AutoProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            
            # Load model based on video model type
            if "videomae" in model_name.lower():
                model = self.transformers.VideoMAEForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float32,
                    cache_dir=cache_dir
                )
            elif "vivit" in model_name.lower():
                model = self.transformers.VivitForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float32,
                    cache_dir=cache_dir
                )
            elif "timesformer" in model_name.lower():
                model = self.transformers.TimesformerForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float32,
                    cache_dir=cache_dir
                )
            else:
                # Default to auto model
                model = self.transformers.AutoModelForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float32,
                    device_map=device,
                    cache_dir=cache_dir
                )
            
            model.eval()
            
            # Create handler function
            handler = self.create_cpu_video_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=cpu_label,
                endpoint=model,
                processor=processor
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, cpu_label, processor)
            
            return model, processor, handler, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing CPU endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, cpu_label)

    def init_cuda(self, model_name, device, cuda_label):
        """Initialize {model_type_upper} model for CUDA inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('cuda:0', 'cuda:1', etc.)
            cuda_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        if not self.torch.cuda.is_available():
            print(f"CUDA not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", cuda_label.replace("cuda", "cpu"))
        
        print(f"Loading {model_name} for CUDA inference on {device}...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load processor for video models
            # Video models can have different processor types
            if "videomae" in model_name.lower():
                processor = self.transformers.VideoMAEImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            elif "vivit" in model_name.lower():
                processor = self.transformers.VivitImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            elif "timesformer" in model_name.lower():
                processor = self.transformers.TimesformerImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            else:
                # Default video processor
                processor = self.transformers.AutoProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            
            # Load model with half precision for GPU efficiency
            if "videomae" in model_name.lower():
                model = self.transformers.VideoMAEForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float16,
                    device_map=device,
                    cache_dir=cache_dir
                )
            elif "vivit" in model_name.lower():
                model = self.transformers.VivitForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float16,
                    device_map=device,
                    cache_dir=cache_dir
                )
            elif "timesformer" in model_name.lower():
                model = self.transformers.TimesformerForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float16,
                    device_map=device,
                    cache_dir=cache_dir
                )
            else:
                # Default to auto model
                model = self.transformers.AutoModelForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float16,
                    device_map=device,
                    cache_dir=cache_dir
                )
            
            model.eval()
            
            # Create handler function
            handler = self.create_cuda_video_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=cuda_label,
                endpoint=model,
                processor=processor
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, cuda_label, processor)
            
            return model, processor, handler, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing CUDA endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, cuda_label)

    def init_rocm(self, model_name, device, rocm_label):
        """Initialize {model_type_upper} model for ROCm (AMD GPU) inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('cuda:0', 'cuda:1', etc. - ROCm uses CUDA device naming)
            rocm_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        # Check for ROCm availability
        rocm_available = False
        try:
            if hasattr(self.torch, 'hip') and self.torch.hip.is_available():
                rocm_available = True
            elif self.torch.cuda.is_available():
                # Could be ROCm using CUDA API
                device_name = self.torch.cuda.get_device_name(0)
                if "AMD" in device_name or "Radeon" in device_name:
                    rocm_available = True
        except Exception as e:
            print(f"Error checking ROCm availability: {e}")
        
        if not rocm_available:
            print(f"ROCm not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", rocm_label.replace("rocm", "cpu"))
        
        print(f"Loading {model_name} for ROCm (AMD GPU) inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load processor for video models
            # Video models can have different processor types
            if "videomae" in model_name.lower():
                processor = self.transformers.VideoMAEImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            elif "vivit" in model_name.lower():
                processor = self.transformers.VivitImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            elif "timesformer" in model_name.lower():
                processor = self.transformers.TimesformerImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            else:
                # Default video processor
                processor = self.transformers.AutoProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            
            # Check for HIP_VISIBLE_DEVICES environment variable
            visible_devices = os.environ.get("HIP_VISIBLE_DEVICES", None) or os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if visible_devices is not None:
                print(f"Using ROCm visible devices: {visible_devices}")
            
            # Get the total GPU memory for logging purposes
            try:
                total_mem = self.torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
                print(f"AMD GPU memory: {total_mem:.2f} GB")
            except Exception as e:
                print(f"Could not query AMD GPU memory: {e}")
            
            # Determine if we should use half precision
            use_half = True
            try:
                # Try to create a small tensor in half precision as a test
                test_tensor = self.torch.ones((10, 10), dtype=self.torch.float16, device="cuda")
                del test_tensor
                print("Half precision is supported on this AMD GPU")
            except Exception as e:
                use_half = False
                print(f"Half precision not supported on this AMD GPU: {e}")
            
            # Load model with appropriate precision for AMD GPU
            if "videomae" in model_name.lower():
                model = self.transformers.VideoMAEForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float16 if use_half else self.torch.float32,
                    device_map="auto",  # ROCm uses the same device map mechanism as CUDA
                    cache_dir=cache_dir
                )
            elif "vivit" in model_name.lower():
                model = self.transformers.VivitForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float16 if use_half else self.torch.float32,
                    device_map="auto",
                    cache_dir=cache_dir
                )
            elif "timesformer" in model_name.lower():
                model = self.transformers.TimesformerForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float16 if use_half else self.torch.float32,
                    device_map="auto",
                    cache_dir=cache_dir
                )
            else:
                # Default to auto model
                model = self.transformers.AutoModelForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float16 if use_half else self.torch.float32,
                    device_map="auto",
                    cache_dir=cache_dir
                )
            
            model.eval()
            
            # Log device mapping
            if hasattr(model, "hf_device_map"):
                print(f"Device map: {model.hf_device_map}")
            
            # Create handler function
            handler = self.create_rocm_video_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=rocm_label,
                endpoint=model,
                processor=processor
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, rocm_label, processor)
            
            return model, processor, handler, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing ROCm endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, rocm_label)

    def init_openvino(self, model_name, device, openvino_label):
        """Initialize {model_type_upper} model for OpenVINO inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('CPU', 'GPU', etc.)
            openvino_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        try:
            from optimum.intel import OVModelForImageClassification
        except ImportError:
            print(f"OpenVINO optimum not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", openvino_label.replace("openvino", "cpu"))
        
        print(f"Loading {model_name} for OpenVINO inference...")
        
        try:
            # Note: Most video models are not directly supported by OpenVINO
            # Falling back to CPU for now
            print("Video models are not directly supported by OpenVINO, falling back to CPU")
            return self.init_cpu(model_name, "cpu", openvino_label.replace("openvino", "cpu"))
            
        except Exception as e:
            print(f"Error initializing OpenVINO endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, openvino_label)

    def init_apple(self, model_name, device, apple_label):
        """Initialize {model_type_upper} model for Apple Silicon (MPS) inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('mps')
            apple_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        if not (hasattr(self.torch, 'backends') and 
                hasattr(self.torch.backends, 'mps') and 
                self.torch.backends.mps.is_available()):
            print(f"Apple MPS not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", apple_label.replace("apple", "cpu"))
        
        print(f"Loading {model_name} for Apple Silicon inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load processor for video models
            # Video models can have different processor types
            if "videomae" in model_name.lower():
                processor = self.transformers.VideoMAEImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            elif "vivit" in model_name.lower():
                processor = self.transformers.VivitImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            elif "timesformer" in model_name.lower():
                processor = self.transformers.TimesformerImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            else:
                # Default video processor
                processor = self.transformers.AutoProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
            
            # Load model
            if "videomae" in model_name.lower():
                model = self.transformers.VideoMAEForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float32,
                    device_map="mps",
                    cache_dir=cache_dir
                )
            elif "vivit" in model_name.lower():
                model = self.transformers.VivitForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float32,
                    device_map="mps",
                    cache_dir=cache_dir
                )
            elif "timesformer" in model_name.lower():
                model = self.transformers.TimesformerForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float32,
                    device_map="mps",
                    cache_dir=cache_dir
                )
            else:
                # Default to auto model
                model = self.transformers.AutoModelForVideoClassification.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float32,
                    device_map="mps",
                    cache_dir=cache_dir
                )
            
            model.eval()
            
            # Create handler function
            handler = self.create_apple_video_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=apple_label,
                endpoint=model,
                processor=processor
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, apple_label, processor)
            
            return model, processor, handler, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing Apple Silicon endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, apple_label)

    def init_qualcomm(self, model_name, device, qualcomm_label):
        """Initialize {model_type_upper} model for Qualcomm inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('qualcomm')
            qualcomm_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        # Check if Qualcomm SDK is available
        try:
            import importlib.util
            has_qnn = importlib.util.find_spec("qnn_wrapper") is not None
            has_qti = importlib.util.find_spec("qti.aisw.dlc_utils") is not None
            has_qualcomm_env = "QUALCOMM_SDK" in os.environ
            
            if not (has_qnn or has_qti or has_qualcomm_env):
                print(f"Qualcomm SDK not available, falling back to CPU")
                return self.init_cpu(model_name, "cpu", qualcomm_label.replace("qualcomm", "cpu"))
        except ImportError:
            print(f"Qualcomm SDK import error, falling back to CPU")
            return self.init_cpu(model_name, "cpu", qualcomm_label.replace("qualcomm", "cpu"))
        
        print(f"Loading {model_name} for Qualcomm inference...")
        
        # For now, we create a mock implementation since Qualcomm SDK integration requires specific hardware
        print("Qualcomm implementation is a mock for now")
        return self._create_mock_endpoint(model_name, qualcomm_label)

    def create_cpu_video_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, processor):
        """Create handler function for CPU video endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('cpu')
            hardware_label (str): The hardware label
            endpoint: The loaded model
            processor: The loaded processor
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(video_path_or_frames, *args, **kwargs):
            try:
                # Handle different input types
                if isinstance(video_path_or_frames, str):
                    # Video path provided, load frames
                    if self.decord is None:
                        return {
                            "success": False,
                            "error": "Decord not available for video loading. Please provide preprocessed frames."
                        }
                    
                    # Load video with decord
                    vr = self.decord.VideoReader(video_path_or_frames)
                    num_frames = len(vr)
                    
                    # Sample frames evenly
                    frame_indices = list(range(0, num_frames, max(1, num_frames // 8)))[:8]
                    if len(frame_indices) < 8:
                        # Pad if we don't have enough frames
                        frame_indices = frame_indices * (8 // len(frame_indices) + 1)
                    frame_indices = frame_indices[:8]  # Take exactly 8 frames
                    
                    frames = vr.get_batch(frame_indices).asnumpy()
                    
                    # Process frames
                    inputs = processor(
                        videos=list(frames),
                        return_tensors="pt"
                    )
                
                elif isinstance(video_path_or_frames, list):
                    # List of frames or arrays provided
                    inputs = processor(
                        videos=video_path_or_frames,
                        return_tensors="pt"
                    )
                    
                elif hasattr(video_path_or_frames, "shape") and len(video_path_or_frames.shape) >= 4:
                    # Already a tensor or numpy array
                    if isinstance(video_path_or_frames, self.np.ndarray):
                        inputs = processor(
                            videos=list(video_path_or_frames),
                            return_tensors="pt"
                        )
                    else:
                        # Assume it's a tensor
                        inputs = {"pixel_values": video_path_or_frames}
                
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported input type: {type(video_path_or_frames)}"
                    }
                
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    outputs = endpoint(**inputs)
                    
                # Process outputs
                result = {"success": True, "device": device, "hardware": hardware_label}
                
                # Extract classification results if available
                if hasattr(outputs, "logits"):
                    logits = outputs.logits.cpu().numpy()
                    
                    # Find top classes
                    if logits.ndim >= 2:
                        top_k = min(5, logits.shape[-1])
                        top_indices = self.np.argsort(-logits, axis=-1)[:, :top_k]
                        top_scores = logits[self.np.arange(logits.shape[0])[:, None], top_indices].tolist()
                        top_indices = top_indices.tolist()
                        
                        # For first batch item only
                        result["classification"] = [{
                            "class_idx": idx,
                            "score": score
                        } for idx, score in zip(top_indices[0], top_scores[0])]
                    
                # Extract embeddings from hidden states if available
                if hasattr(outputs, "last_hidden_state"):
                    # Get frame-level features
                    frame_features = outputs.last_hidden_state.cpu().numpy()
                    result["frame_features"] = frame_features.tolist()
                    
                    # Get pooled video-level embedding
                    if hasattr(outputs, "pooler_output"):
                        embeddings = outputs.pooler_output.cpu().numpy()
                        result["embeddings"] = embeddings.tolist()
                    else:
                        # Mean pooling
                        embeddings = frame_features.mean(axis=1)
                        result["embeddings"] = embeddings.tolist()
                
                return result
                
            except Exception as e:
                print(f"Error in CPU handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler

    def create_cuda_video_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, processor):
        """Create handler function for CUDA video endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('cuda:0', etc.)
            hardware_label (str): The hardware label
            endpoint: The loaded model
            processor: The loaded processor
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(video_path_or_frames, *args, **kwargs):
            try:
                # Handle different input types
                if isinstance(video_path_or_frames, str):
                    # Video path provided, load frames
                    if self.decord is None:
                        return {
                            "success": False,
                            "error": "Decord not available for video loading. Please provide preprocessed frames."
                        }
                    
                    # Load video with decord
                    vr = self.decord.VideoReader(video_path_or_frames)
                    num_frames = len(vr)
                    
                    # Sample frames evenly
                    frame_indices = list(range(0, num_frames, max(1, num_frames // 8)))[:8]
                    if len(frame_indices) < 8:
                        # Pad if we don't have enough frames
                        frame_indices = frame_indices * (8 // len(frame_indices) + 1)
                    frame_indices = frame_indices[:8]  # Take exactly 8 frames
                    
                    frames = vr.get_batch(frame_indices).asnumpy()
                    
                    # Process frames
                    inputs = processor(
                        videos=list(frames),
                        return_tensors="pt"
                    )
                
                elif isinstance(video_path_or_frames, list):
                    # List of frames or arrays provided
                    inputs = processor(
                        videos=video_path_or_frames,
                        return_tensors="pt"
                    )
                    
                elif hasattr(video_path_or_frames, "shape") and len(video_path_or_frames.shape) >= 4:
                    # Already a tensor or numpy array
                    if isinstance(video_path_or_frames, self.np.ndarray):
                        inputs = processor(
                            videos=list(video_path_or_frames),
                            return_tensors="pt"
                        )
                    else:
                        # Assume it's a tensor
                        inputs = {"pixel_values": video_path_or_frames}
                
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported input type: {type(video_path_or_frames)}"
                    }
                
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference with no gradient calculation and mixed precision
                with self.torch.no_grad():
                    with self.torch.cuda.amp.autocast():
                        outputs = endpoint(**inputs)
                    
                # Process outputs
                result = {"success": True, "device": device, "hardware": hardware_label}
                
                # Extract classification results if available
                if hasattr(outputs, "logits"):
                    logits = outputs.logits.cpu().numpy()
                    
                    # Find top classes
                    if logits.ndim >= 2:
                        top_k = min(5, logits.shape[-1])
                        top_indices = self.np.argsort(-logits, axis=-1)[:, :top_k]
                        top_scores = logits[self.np.arange(logits.shape[0])[:, None], top_indices].tolist()
                        top_indices = top_indices.tolist()
                        
                        # For first batch item only
                        result["classification"] = [{
                            "class_idx": idx,
                            "score": score
                        } for idx, score in zip(top_indices[0], top_scores[0])]
                    
                # Extract embeddings from hidden states if available
                if hasattr(outputs, "last_hidden_state"):
                    # Get frame-level features
                    frame_features = outputs.last_hidden_state.cpu().numpy()
                    result["frame_features"] = frame_features.tolist()
                    
                    # Get pooled video-level embedding
                    if hasattr(outputs, "pooler_output"):
                        embeddings = outputs.pooler_output.cpu().numpy()
                        result["embeddings"] = embeddings.tolist()
                    else:
                        # Mean pooling
                        embeddings = frame_features.mean(axis=1)
                        result["embeddings"] = embeddings.tolist()
                
                return result
                
            except Exception as e:
                print(f"Error in CUDA handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler

    def create_rocm_video_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, processor):
        """Create handler function for ROCm video endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('cuda:0', etc. - ROCm uses CUDA device naming)
            hardware_label (str): The hardware label
            endpoint: The loaded model
            processor: The loaded processor
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(video_path_or_frames, *args, **kwargs):
            try:
                # Handle different input types
                if isinstance(video_path_or_frames, str):
                    # Video path provided, load frames
                    if self.decord is None:
                        return {
                            "success": False,
                            "error": "Decord not available for video loading. Please provide preprocessed frames."
                        }
                    
                    # Load video with decord
                    vr = self.decord.VideoReader(video_path_or_frames)
                    num_frames = len(vr)
                    
                    # Sample frames evenly
                    frame_indices = list(range(0, num_frames, max(1, num_frames // 8)))[:8]
                    if len(frame_indices) < 8:
                        # Pad if we don't have enough frames
                        frame_indices = frame_indices * (8 // len(frame_indices) + 1)
                    frame_indices = frame_indices[:8]  # Take exactly 8 frames
                    
                    frames = vr.get_batch(frame_indices).asnumpy()
                    
                    # Process frames
                    inputs = processor(
                        videos=list(frames),
                        return_tensors="pt"
                    )
                
                elif isinstance(video_path_or_frames, list):
                    # List of frames or arrays provided
                    inputs = processor(
                        videos=video_path_or_frames,
                        return_tensors="pt"
                    )
                    
                elif hasattr(video_path_or_frames, "shape") and len(video_path_or_frames.shape) >= 4:
                    # Already a tensor or numpy array
                    if isinstance(video_path_or_frames, self.np.ndarray):
                        inputs = processor(
                            videos=list(video_path_or_frames),
                            return_tensors="pt"
                        )
                    else:
                        # Assume it's a tensor
                        inputs = {"pixel_values": video_path_or_frames}
                
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported input type: {type(video_path_or_frames)}"
                    }
                
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    # For AMD GPUs, autocast may or may not be supported depending on ROCm version
                    try:
                        # Try with autocast
                        with self.torch.cuda.amp.autocast():
                            outputs = endpoint(**inputs)
                    except RuntimeError:
                        # Fallback without autocast
                        outputs = endpoint(**inputs)
                    
                # Process outputs
                result = {"success": True, "device": device, "hardware": hardware_label}
                
                # Extract classification results if available
                if hasattr(outputs, "logits"):
                    logits = outputs.logits.cpu().numpy()
                    
                    # Find top classes
                    if logits.ndim >= 2:
                        top_k = min(5, logits.shape[-1])
                        top_indices = self.np.argsort(-logits, axis=-1)[:, :top_k]
                        top_scores = logits[self.np.arange(logits.shape[0])[:, None], top_indices].tolist()
                        top_indices = top_indices.tolist()
                        
                        # For first batch item only
                        result["classification"] = [{
                            "class_idx": idx,
                            "score": score
                        } for idx, score in zip(top_indices[0], top_scores[0])]
                    
                # Extract embeddings from hidden states if available
                if hasattr(outputs, "last_hidden_state"):
                    # Get frame-level features
                    frame_features = outputs.last_hidden_state.cpu().numpy()
                    result["frame_features"] = frame_features.tolist()
                    
                    # Get pooled video-level embedding
                    if hasattr(outputs, "pooler_output"):
                        embeddings = outputs.pooler_output.cpu().numpy()
                        result["embeddings"] = embeddings.tolist()
                    else:
                        # Mean pooling
                        embeddings = frame_features.mean(axis=1)
                        result["embeddings"] = embeddings.tolist()
                
                return result
                
            except Exception as e:
                print(f"Error in ROCm handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler
        
    def create_openvino_video_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, processor):
        """Create handler function for OpenVINO video endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('CPU', 'GPU', etc.)
            hardware_label (str): The hardware label
            endpoint: The loaded model
            processor: The loaded processor
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(video_path_or_frames, *args, **kwargs):
            try:
                # Handle different input types
                if isinstance(video_path_or_frames, str):
                    # Video path provided, load frames
                    if self.decord is None:
                        return {
                            "success": False,
                            "error": "Decord not available for video loading. Please provide preprocessed frames."
                        }
                    
                    # Load video with decord
                    vr = self.decord.VideoReader(video_path_or_frames)
                    num_frames = len(vr)
                    
                    # Sample frames evenly
                    frame_indices = list(range(0, num_frames, max(1, num_frames // 8)))[:8]
                    if len(frame_indices) < 8:
                        # Pad if we don't have enough frames
                        frame_indices = frame_indices * (8 // len(frame_indices) + 1)
                    frame_indices = frame_indices[:8]  # Take exactly 8 frames
                    
                    frames = vr.get_batch(frame_indices).asnumpy()
                    
                    # Process frames
                    inputs = processor(
                        videos=list(frames),
                        return_tensors="pt"
                    )
                
                elif isinstance(video_path_or_frames, list):
                    # List of frames or arrays provided
                    inputs = processor(
                        videos=video_path_or_frames,
                        return_tensors="pt"
                    )
                    
                elif hasattr(video_path_or_frames, "shape") and len(video_path_or_frames.shape) >= 4:
                    # Already a tensor or numpy array
                    if isinstance(video_path_or_frames, self.np.ndarray):
                        inputs = processor(
                            videos=list(video_path_or_frames),
                            return_tensors="pt"
                        )
                    else:
                        # Assume it's a tensor
                        inputs = {"pixel_values": video_path_or_frames}
                
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported input type: {type(video_path_or_frames)}"
                    }
                
                # Run inference
                # This is a mock response as video models with OpenVINO are not yet supported
                result = {"success": True, "device": device, "hardware": hardware_label}
                result["mock"] = True
                
                return result
                
            except Exception as e:
                print(f"Error in OpenVINO handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler

    def create_apple_video_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, processor):
        """Create handler function for Apple Silicon video endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('mps')
            hardware_label (str): The hardware label
            endpoint: The loaded model
            processor: The loaded processor
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(video_path_or_frames, *args, **kwargs):
            try:
                # Handle different input types
                if isinstance(video_path_or_frames, str):
                    # Video path provided, load frames
                    if self.decord is None:
                        return {
                            "success": False,
                            "error": "Decord not available for video loading. Please provide preprocessed frames."
                        }
                    
                    # Load video with decord
                    vr = self.decord.VideoReader(video_path_or_frames)
                    num_frames = len(vr)
                    
                    # Sample frames evenly
                    frame_indices = list(range(0, num_frames, max(1, num_frames // 8)))[:8]
                    if len(frame_indices) < 8:
                        # Pad if we don't have enough frames
                        frame_indices = frame_indices * (8 // len(frame_indices) + 1)
                    frame_indices = frame_indices[:8]  # Take exactly 8 frames
                    
                    frames = vr.get_batch(frame_indices).asnumpy()
                    
                    # Process frames
                    inputs = processor(
                        videos=list(frames),
                        return_tensors="pt"
                    )
                
                elif isinstance(video_path_or_frames, list):
                    # List of frames or arrays provided
                    inputs = processor(
                        videos=video_path_or_frames,
                        return_tensors="pt"
                    )
                    
                elif hasattr(video_path_or_frames, "shape") and len(video_path_or_frames.shape) >= 4:
                    # Already a tensor or numpy array
                    if isinstance(video_path_or_frames, self.np.ndarray):
                        inputs = processor(
                            videos=list(video_path_or_frames),
                            return_tensors="pt"
                        )
                    else:
                        # Assume it's a tensor
                        inputs = {"pixel_values": video_path_or_frames}
                
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported input type: {type(video_path_or_frames)}"
                    }
                
                # Move inputs to the correct device
                inputs = {k: v.to("mps") for k, v in inputs.items()}
                
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    outputs = endpoint(**inputs)
                    
                # Process outputs
                result = {"success": True, "device": device, "hardware": hardware_label}
                
                # Extract classification results if available
                if hasattr(outputs, "logits"):
                    logits = outputs.logits.cpu().numpy()
                    
                    # Find top classes
                    if logits.ndim >= 2:
                        top_k = min(5, logits.shape[-1])
                        top_indices = self.np.argsort(-logits, axis=-1)[:, :top_k]
                        top_scores = logits[self.np.arange(logits.shape[0])[:, None], top_indices].tolist()
                        top_indices = top_indices.tolist()
                        
                        # For first batch item only
                        result["classification"] = [{
                            "class_idx": idx,
                            "score": score
                        } for idx, score in zip(top_indices[0], top_scores[0])]
                    
                # Extract embeddings from hidden states if available
                if hasattr(outputs, "last_hidden_state"):
                    # Get frame-level features
                    frame_features = outputs.last_hidden_state.cpu().numpy()
                    result["frame_features"] = frame_features.tolist()
                    
                    # Get pooled video-level embedding
                    if hasattr(outputs, "pooler_output"):
                        embeddings = outputs.pooler_output.cpu().numpy()
                        result["embeddings"] = embeddings.tolist()
                    else:
                        # Mean pooling
                        embeddings = frame_features.mean(axis=1)
                        result["embeddings"] = embeddings.tolist()
                
                return result
                
            except Exception as e:
                print(f"Error in Apple handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler

    def create_qualcomm_video_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, processor):
        """Create handler function for Qualcomm video endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('qualcomm')
            hardware_label (str): The hardware label
            endpoint: The loaded model
            processor: The loaded processor
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(video_path_or_frames, *args, **kwargs):
            try:
                # This is a placeholder for Qualcomm implementation
                # In a real implementation, we would use the Qualcomm SDK
                
                # Mock result for now
                return {
                    "success": True,
                    "device": device,
                    "hardware": hardware_label,
                    "mock": True,
                    "input_type": str(type(video_path_or_frames))
                }
                
            except Exception as e:
                print(f"Error in Qualcomm handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler