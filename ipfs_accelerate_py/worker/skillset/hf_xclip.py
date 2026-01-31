import time
import anyio
from ..anyio_queue import AnyioQueue
import os
from PIL import Image
import requests
from io import BytesIO
import os
import tempfile
import numpy as np

try:
    from ...common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        try:
            from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
        except ImportError:
            HAVE_STORAGE_WRAPPER = False

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def load_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_image_tensor(image_file):
    import openvino as ov
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return ov.Tensor(image_data)

class hf_xclip:
    """Handles XCLIP model operations across different hardware backends.
    
    This class provides methods to initialize and create handlers for xclip models
    on various hardware backends including CPU, CUDA, OpenVINO, Apple Silicon, and Qualcomm.
    It supports both text/video embedding extraction and similarity calculation.
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the hf_xclip class.
        
        Args:
            resources: Dictionary of resource modules (torch, transformers, etc.)
            metadata: Additional metadata for the model
        """
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper(auto_detect_ci=True)
            except Exception:
                self._storage = None
        else:
            self._storage = None
        
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        self.torch = None
        self.transformers = None
        self.np = None
        self.decord = None
        self.snpe_utils = None
        self.coreml_utils = None
        self.init()  # Initialize core modules
        
        # Create method aliases for backward compatibility
        self._create_method_aliases()
        
    def _create_method_aliases(self):
        """Create method aliases for backward compatibility.
        
        This ensures that both old and new method naming patterns work properly.
        """
        # Map between handler methods to ensure all naming patterns work
        handler_mappings = {
            'create_video_embedding_endpoint_handler': 'create_cuda_video_embedding_endpoint_handler',
            'create_apple_multimodal_endpoint_handler': 'create_apple_video_embedding_endpoint_handler',
            'create_qualcomm_xclip_endpoint_handler': 'create_qualcomm_video_embedding_endpoint_handler'
        }
        
        # Create aliases in both directions to ensure all naming patterns work
        for method1, method2 in handler_mappings.items():
            # If first method exists but second doesn't, create an alias
            if hasattr(self, method1) and not hasattr(self, method2):
                setattr(self, method2, getattr(self, method1))
            # If second method exists but first doesn't, create an alias
            elif hasattr(self, method2) and not hasattr(self, method1):
                setattr(self, method1, getattr(self, method2))

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

        if "decord" not in list(self.resources.keys()):
            import decord
            self.decord = decord
        else:
            self.decord = self.resources["decord"]
        self.np.random.seed(0)
        return None
    
    def init_qualcomm(self, model, device, qualcomm_label):
        """Initialize XClip model for Qualcomm hardware.
        
        Args:
            model: HuggingFace model name or path
            device: Device to run inference on
            qualcomm_label: Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        
        # Import SNPE utilities
        try:
            from .qualcomm_snpe_utils import get_snpe_utils
            self.snpe_utils = get_snpe_utils()
        except ImportError:
            print("Failed to import Qualcomm SNPE utilities")
            return None, None, None, None, 0
            
        if not self.snpe_utils.is_available():
            print("Qualcomm SNPE is not available on this system")
            return None, None, None, None, 0
            
        try:
            # Initialize processor directly from HuggingFace
            processor = self.transformers.AutoProcessor.from_pretrained(model)
            
            # Convert model path to be compatible with SNPE
            model_name = model.replace("/", "--")
            dlc_path = f"~/snpe_models/{model_name}_xclip.dlc"
            dlc_path = os.path.expanduser(dlc_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(dlc_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(dlc_path):
                print(f"Converting {model} to SNPE format...")
                self.snpe_utils.convert_model(model, "vision_text_dual", str(dlc_path))
            
            # Load the SNPE model
            endpoint = self.snpe_utils.load_model(str(dlc_path))
            
            # Optimize for the specific Qualcomm device if possible
            if ":" in qualcomm_label:
                device_type = qualcomm_label.split(":")[1]
                optimized_path = self.snpe_utils.optimize_for_device(dlc_path, device_type)
                if optimized_path != dlc_path:
                    endpoint = self.snpe_utils.load_model(optimized_path)
            
            # Create endpoint handler
            endpoint_handler = self.create_qualcomm_xclip_endpoint_handler(processor, model, qualcomm_label, endpoint)
            
            return endpoint, processor, endpoint_handler, AnyioQueue(32), 0
        except Exception as e:
            print(f"Error initializing Qualcomm XClip model: {e}")
            return None, None, None, None, 0

    def init_apple(self, model, device, apple_label):
        """Initialize XClip model for Apple Silicon hardware."""
        self.init()
        
        try:
            from .apple_coreml_utils import get_coreml_utils
            self.coreml_utils = get_coreml_utils()
        except ImportError:
            print("Failed to import CoreML utilities")
            return None, None, None, None, 0
            
        if not self.coreml_utils.is_available():
            print("CoreML is not available on this system")
            return None, None, None, None, 0
            
        try:
            # Load processor from HuggingFace
            processor = self.transformers.XCLIPProcessor.from_pretrained(model, trust_remote_code=True)
            
            # Convert model path to be compatible with CoreML
            model_name = model.replace("/", "--")
            mlmodel_path = f"~/coreml_models/{model_name}_xclip.mlpackage"
            mlmodel_path = os.path.expanduser(mlmodel_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(mlmodel_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(mlmodel_path):
                print(f"Converting {model} to CoreML format...")
                self.coreml_utils.convert_model(model, "vision_text_dual", str(mlmodel_path))
            
            # Load the CoreML model
            endpoint = self.coreml_utils.load_model(str(mlmodel_path))
            
            # Optimize for Apple Silicon if possible
            if ":" in apple_label:
                compute_units = apple_label.split(":")[1]
                optimized_path = self.coreml_utils.optimize_for_device(mlmodel_path, compute_units)
                if optimized_path != mlmodel_path:
                    endpoint = self.coreml_utils.load_model(optimized_path)
            
            endpoint_handler = self.create_apple_multimodal_endpoint_handler(endpoint, processor, model, apple_label)
            
            return endpoint, processor, endpoint_handler, AnyioQueue(32), 0
        except Exception as e:
            print(f"Error initializing Apple Silicon XClip model: {e}")
            return None, None, None, None, 0

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, video_url)
            print(test_batch)
            print("hf_xclip test passed")
        except Exception as e:
            print(e)
            print("hf_xclip test failed")
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        len_tokens = 1
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"samples: {len_tokens}")
        print(f"samples per second: {tokens_per_second}")
        # test_batch_sizes = await self.test_batch_sizes(metadata['models'], ipfs_accelerate_init)
        if "openvino" not in endpoint_label:
            with self.torch.no_grad():
                if "cuda" in dir(self.torch):
                    self.torch.cuda.empty_cache()
        print("hf_xclip test")
        return None
    
    def init_cpu(self, model, device, cpu_label):
        """Initialize XCLIP model for CPU.
        
        Args:
            model: HuggingFace model name or path
            device: Device to run inference on
            cpu_label: Label for this CPU endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        
        # Check if transformers is available as a real module (not a mock)
        transformers_available = False
        try:
            # More robust check for transformers availability
            if self.transformers is not None:
                if not isinstance(self.transformers, type):
                    # Make sure we have a key class available
                    if hasattr(self.transformers, 'AutoProcessor'):
                        transformers_available = True
                        print("Transformers module available with AutoProcessor")
        except Exception as check_error:
            print(f"Error checking transformers availability: {check_error}")
        
        # Variable to track which implementation we're using
        is_real_impl = False
        
        if transformers_available:
            try:
                print(f"Trying to load real XCLIP model: {model}")
                
                # First attempt with AutoConfig for model information
                try:
                    config = self.transformers.AutoConfig.from_pretrained(
                        model, 
                        trust_remote_code=True
                    )
                    print(f"Successfully loaded config for {model}")
                except Exception as config_error:
                    print(f"Error loading AutoConfig: {config_error}")
                    config = None
                
                # Try to load processor - critical component for XCLIP
                processor = None
                try:
                    # Try with different processor classes in order of likelihood
                    processor_candidates = [
                        lambda: self.transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True),
                        lambda: self.transformers.CLIPProcessor.from_pretrained(model, trust_remote_code=True),
                        lambda: self.transformers.XCLIPProcessor.from_pretrained(model, trust_remote_code=True)
                    ]
                    
                    for proc_loader in processor_candidates:
                        try:
                            processor = proc_loader()
                            if processor is not None:
                                print(f"Successfully loaded processor for {model}")
                                break
                        except Exception as proc_error:
                            print(f"Error with processor candidate: {proc_error}")
                    
                    if processor is None:
                        print("All processor loading attempts failed")
                except Exception as processor_error:
                    print(f"Error loading processor: {processor_error}")
                
                # Try to load model
                endpoint = None
                try:
                    # Try with different model classes in order of likelihood
                    model_candidates = [
                        lambda: self.transformers.AutoModel.from_pretrained(model, trust_remote_code=True),
                        lambda: self.transformers.CLIPModel.from_pretrained(model, trust_remote_code=True),
                        lambda: self.transformers.VisionTextDualEncoderModel.from_pretrained(model, trust_remote_code=True)
                    ]
                    
                    # Try all model classes until one works
                    for model_loader in model_candidates:
                        try:
                            endpoint = model_loader()
                            if endpoint is not None:
                                print(f"Successfully loaded model for {model}")
                                break
                        except Exception as model_class_error:
                            print(f"Error with model candidate: {model_class_error}")
                    
                    if endpoint is None:
                        print("All model loading attempts failed")
                except Exception as model_error:
                    print(f"Error loading model: {model_error}")
                
                # If we have both processor and endpoint, we can create a real handler
                if processor is not None and endpoint is not None:
                    # Set flag that we're using real implementation
                    is_real_impl = True
                    
                    # Create the handler with the endpoint and processor
                    endpoint_handler = self.create_cpu_video_embedding_endpoint_handler(
                        tokenizer=processor,
                        endpoint_model=model,
                        cpu_label=cpu_label,
                        endpoint=endpoint
                    )
                    
                    # Return all components needed for inference
                    print(f"Successfully initialized real XCLIP model for CPU")
                    return endpoint, processor, endpoint_handler, AnyioQueue(64), 0
                else:
                    print("Missing processor or endpoint for real implementation")
                    # Continue to mock implementation
            except Exception as e:
                print(f"Error initializing real CPU model: {e}")
                print("Falling back to mock implementation...")
        else:
            print("Transformers not available, using mock implementation")
        
        # Create mock versions for testing if real initialization fails
        try:
            # Create mock processor and endpoint
            from unittest.mock import MagicMock
            
            # Try to get processor if transformers is available
            if transformers_available:
                try:
                    if processor is None:  # Only try again if we don't already have one
                        processor = self.transformers.AutoProcessor.from_pretrained(
                            model, 
                            trust_remote_code=True
                        )
                except Exception:
                    processor = MagicMock()
            else:
                processor = MagicMock()
            
            # Mock the endpoint
            mock_endpoint = MagicMock()
            
            # Define some basic behavior for mocks
            if isinstance(processor, MagicMock):
                processor.side_effect = lambda **kwargs: {
                    "input_ids": self.torch.ones((1, 10), dtype=self.torch.long),
                    "attention_mask": self.torch.ones((1, 10), dtype=self.torch.long),
                    "pixel_values": self.torch.zeros((1, 3, 224, 224), dtype=self.torch.float32)
                }
            
            if isinstance(mock_endpoint, MagicMock):
                # Create a mock output class that mimics the real model output
                class MockOutput:
                    def __init__(self):
                        import torch
                        self.text_embeds = torch.randn(1, 512)
                        self.image_embeds = torch.randn(1, 512)
                
                mock_endpoint.return_value = MockOutput()
            
            # Create handler with mocks
            endpoint_handler = self.create_cpu_video_embedding_endpoint_handler(
                tokenizer=processor, 
                endpoint_model=model,
                cpu_label=cpu_label,
                endpoint=mock_endpoint
            )
            
            print("Successfully initialized mock XCLIP model for CPU")
            return mock_endpoint, processor, endpoint_handler, AnyioQueue(64), 0
            
        except Exception as mock_error:
            print(f"Error creating mock implementation: {mock_error}")
            
            # Create absolute minimal mocks as a last resort
            processor = MagicMock()
            mock_endpoint = MagicMock()
            
            # Create handler with minimal mocks
            endpoint_handler = self.create_cpu_video_embedding_endpoint_handler(
                tokenizer=processor, 
                endpoint_model=model,
                cpu_label=cpu_label,
                endpoint=mock_endpoint
            )
            
            return mock_endpoint, processor, endpoint_handler, AnyioQueue(64), 0
    
    def init_cuda(self, model, device, cuda_label):
        """Initialize XCLIP model for CUDA/GPU.
        
        Args:
            model: HuggingFace model name or path
            device: CUDA device to use (e.g., 'cuda:0')
            cuda_label: Label for this CUDA endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        
        # Helper function to create dummy components that are JSON serializable
        def create_dummy_components():
            # Create a dummy processor
            class DummyProcessor:
                def __call__(self, *args, **kwargs):
                    return {"input_ids": self.torch.zeros((1, 10), dtype=self.torch.long),
                            "attention_mask": self.torch.ones((1, 10), dtype=self.torch.long),
                            "pixel_values": self.torch.zeros((1, 3, 224, 224), dtype=self.torch.float32)}
            
            # Create a dummy model
            class DummyModel:
                def __call__(self, *args, **kwargs):
                    return None
                def eval(self):
                    pass
                def to(self, device):
                    self.device = device
                    return self
                @property
                def device(self):
                    return device
            
            return DummyProcessor(), DummyModel()
        
        try:
            # Check if CUDA is available
            if not self.torch.cuda.is_available():
                print(f"CUDA not available. Using dummy components instead.")
                processor, endpoint = create_dummy_components()
                endpoint_handler = self.create_cuda_video_embedding_endpoint_handler(
                    endpoint=endpoint,
                    tokenizer=processor,
                    endpoint_model=model,
                    cuda_label=cuda_label
                )
                return endpoint, processor, endpoint_handler, AnyioQueue(64), 0
            
            # Try to load the model components
            try:    
                config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)
                processor = self.transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
            except Exception as config_error:
                print(f"Failed to load config/processor, trying alternatives: {config_error}")
                try:
                    processor = self.transformers.CLIPProcessor.from_pretrained(model, trust_remote_code=True)
                except Exception:
                    print("Creating a minimal processor")
                    processor, _ = create_dummy_components()
            
            # Load the model
            try:
                endpoint = self.transformers.AutoModel.from_pretrained(
                    model, 
                    torch_dtype=self.torch.float16, 
                    trust_remote_code=True
                ).to(device)
            except Exception as model_error:
                print(f"Failed to load AutoModel, trying specific model class: {model_error}")
                try:
                    endpoint = self.transformers.CLIPModel.from_pretrained(
                        model, 
                        torch_dtype=self.torch.float16, 
                        trust_remote_code=True
                    ).to(device)
                except Exception:
                    print("Creating a minimal model")
                    _, endpoint = create_dummy_components()
                    endpoint = endpoint.to(device)
            
            # Create the handler
            endpoint_handler = self.create_cuda_video_embedding_endpoint_handler(
                endpoint=endpoint,
                tokenizer=processor,
                endpoint_model=model,
                cuda_label=cuda_label
            )
            
            # Clean up GPU memory
            if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
            
            return endpoint, processor, endpoint_handler, AnyioQueue(64), 0
            
        except Exception as e:
            print(f"Error initializing CUDA model: {e}")
            processor, endpoint = create_dummy_components()
            endpoint_handler = self.create_cuda_video_embedding_endpoint_handler(
                endpoint=endpoint,
                tokenizer=processor,
                endpoint_model=model,
                cuda_label=cuda_label
            )
            return endpoint, processor, endpoint_handler, AnyioQueue(64), 0

    def init_openvino(self, model=None , model_type=None, device=None, openvino_label=None, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None):
        """Initialize XClip model for OpenVINO.
        
        Args:
            model: HuggingFace model name or path
            model_type: Type of model for OpenVINO
            device: Device to run inference on (typically 'CPU')
            openvino_label: Label for this OpenVINO endpoint
            get_optimum_openvino_model: Function to get optimum OpenVINO model
            get_openvino_model: Function to get OpenVINO model
            get_openvino_pipeline_type: Function to get pipeline type
            openvino_cli_convert: Function to convert model using CLI
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        
        # Helper function to create dummy components that are JSON serializable
        def create_dummy_components():
            # Create a dummy processor
            class DummyProcessor:
                def __call__(self, *args, **kwargs):
                    import numpy as np
                    return {"input_ids": np.zeros((1, 10), dtype=np.int64),
                            "attention_mask": np.ones((1, 10), dtype=np.int64),
                            "pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)}
            
            # Create a dummy model
            class DummyModel:
                def __call__(self, *args, **kwargs):
                    import numpy as np
                    return {
                        "text_embeds": np.random.randn(1, 512).astype(np.float32),
                        "image_embeds": np.random.randn(1, 512).astype(np.float32)
                    }
            
            return DummyProcessor(), DummyModel()
        
        # Initialize OpenVINO if available
        try:
            if "openvino" in self.resources:
                self.ov = self.resources["openvino"]
            else:
                try:
                    import openvino as ov
                    self.ov = ov
                except ImportError as e:
                    print(f"Error importing OpenVINO: {e}")
                    processor, endpoint = create_dummy_components()
                    endpoint_handler = self.create_openvino_video_embedding_endpoint_handler(
                        endpoint=endpoint,
                        tokenizer=processor,
                        endpoint_model=model,
                        openvino_label=openvino_label
                    )
                    return endpoint, processor, endpoint_handler, AnyioQueue(64), 0
        except Exception as e:
            print(f"Error setting up OpenVINO: {e}")
            processor, endpoint = create_dummy_components()
            endpoint_handler = self.create_openvino_video_embedding_endpoint_handler(
                endpoint=endpoint,
                tokenizer=processor,
                endpoint_model=model,
                openvino_label=openvino_label
            )
            return endpoint, processor, endpoint_handler, AnyioQueue(64), 0
        
        # Create dummy components that we'll use if any part of initialization fails
        dummy_processor, dummy_endpoint = create_dummy_components()
        
        try:
            # Safe handling of HuggingFace cache paths
            try:
                homedir = os.path.expanduser("~")
                model_name_convert = model.replace("/", "--")
                huggingface_cache = os.path.join(homedir, ".cache/huggingface")
                huggingface_cache_models = os.path.join(huggingface_cache, "hub")
                
                # Check if cache directory exists
                if os.path.exists(huggingface_cache_models):
                    huggingface_cache_models_files = os.listdir(huggingface_cache_models)
                    huggingface_cache_models_files_dirs = [
                        os.path.join(huggingface_cache_models, file) 
                        for file in huggingface_cache_models_files 
                        if os.path.isdir(os.path.join(huggingface_cache_models, file))
                    ]
                    huggingface_cache_models_files_dirs_models = [
                        x for x in huggingface_cache_models_files_dirs if "model" in x
                    ]
                    
                    # Safely get model directory
                    model_src_path = None
                    model_matches = [
                        x for x in huggingface_cache_models_files_dirs_models if model_name_convert in x
                    ]
                    if model_matches and len(model_matches) > 0:  # Safe list indexing
                        model_src_path = model_matches[0]
                    else:
                        print(f"Model {model} not found in HuggingFace cache")
                        model_src_path = os.path.join(huggingface_cache_models, f"models--{model_name_convert}")
                else:
                    print(f"HuggingFace cache directory not found at {huggingface_cache_models}")
                    model_src_path = os.path.join(homedir, "openvino_models", model_name_convert)
                
                # Create destination path
                model_dst_path = os.path.join(model_src_path, "openvino") if model_src_path else None
            except Exception as cache_error:
                print(f"Error accessing HuggingFace cache: {cache_error}")
                model_src_path = os.path.join(homedir, "openvino_models", model_name_convert)
                model_dst_path = os.path.join(model_src_path, "openvino")
            
            # Get task type safely
            task = "vision_text_dual"  # Default task for XCLIP
            if get_openvino_pipeline_type:
                try:
                    task = get_openvino_pipeline_type(model, model_type)
                except Exception as e:
                    print(f"Error getting OpenVINO pipeline type: {e}")
            
            # Get weight format safely
            weight_format = "int8"  # Default to int8
            try:
                if openvino_label and ":" in openvino_label:
                    openvino_index = int(openvino_label.split(":")[1])
                    if openvino_index == 0:
                        weight_format = "int8"  # CPU
                    elif openvino_index == 1:
                        weight_format = "int4"  # GPU
                    elif openvino_index == 2:
                        weight_format = "int4"  # NPU
            except Exception as e:
                print(f"Error parsing OpenVINO label: {e}")
                
            # Update model destination path
            if model_dst_path:
                model_dst_path = f"{model_dst_path}_{weight_format}"
                
                # Create directory if it doesn't exist
                if not os.path.exists(model_dst_path):
                    os.makedirs(model_dst_path, exist_ok=True)
                    
                    # Try using openvino_skill_convert if available
                    if hasattr(self, 'openvino_skill_convert'):
                        try:
                            convert = self.openvino_skill_convert(model, model_dst_path, task, weight_format)
                            print(f"Model converted with openvino_skill_convert: {convert}")
                        except Exception as e:
                            print(f"Error using openvino_skill_convert: {e}")
                        
                        # Fall back to openvino_cli_convert
                        if openvino_cli_convert is not None:
                            try:
                                convert = openvino_cli_convert(
                                    model, 
                                    model_dst_path=model_dst_path, 
                                    task=task, 
                                    weight_format=weight_format, 
                                    ratio="1.0", 
                                    group_size=128, 
                                    sym=True
                                )
                                print(f"Successfully converted model using OpenVINO CLI: {convert}")
                            except Exception as e:
                                print(f"Error using openvino_cli_convert: {e}")
            
            # Try to get processor
            processor = dummy_processor  # Default to dummy processor
            try:
                processor_result = self.transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
                if processor_result is not None:
                    processor = processor_result
            except Exception as e:
                print(f"Error loading AutoProcessor: {e}")
                try:
                    if model_src_path:
                        processor_result = self.transformers.AutoProcessor.from_pretrained(model_src_path, trust_remote_code=True)
                        if processor_result is not None:
                            processor = processor_result
                except Exception as e:
                    print(f"Error loading processor from cache: {e}")
                    try:
                        # Try with CLIPProcessor
                        processor_result = self.transformers.CLIPProcessor.from_pretrained(model, trust_remote_code=True)
                        if processor_result is not None:
                            processor = processor_result
                    except Exception as e:
                        print(f"Error loading CLIPProcessor: {e}")
                        # Will use our dummy processor
            
            # Try to get model
            endpoint = dummy_endpoint  # Default to dummy endpoint if initialization fails
            if get_openvino_model is not None:
                try:
                    model_result = get_openvino_model(model, model_type, openvino_label)
                    if model_result is not None:
                        endpoint = model_result
                        print(f"Successfully loaded OpenVINO model directly")
                except Exception as e:
                    print(f"Error with get_openvino_model: {e}")
            
            # Try optimum model if direct model loading failed
            if endpoint == dummy_endpoint and get_optimum_openvino_model is not None:
                try:
                    optimum_model_result = get_optimum_openvino_model(model, model_type, openvino_label)
                    if optimum_model_result is not None:
                        endpoint = optimum_model_result
                        print(f"Successfully loaded optimum OpenVINO model")
                except Exception as e:
                    print(f"Error with get_optimum_openvino_model: {e}")
            
            # Create endpoint handler
            endpoint_handler = self.create_openvino_video_embedding_endpoint_handler(
                endpoint=endpoint,
                tokenizer=processor,
                endpoint_model=model,
                openvino_label=openvino_label
            )
            
            # Return initialized components - always return success even with dummy components
            return endpoint, processor, endpoint_handler, AnyioQueue(64), 0
            
        except Exception as e:
            print(f"Error in OpenVINO initialization: {e}")
            # Create endpoint handler with dummy components
            endpoint_handler = self.create_openvino_video_embedding_endpoint_handler(
                endpoint=dummy_endpoint,
                tokenizer=dummy_processor,
                endpoint_model=model,
                openvino_label=openvino_label
            )
            return dummy_endpoint, dummy_processor, endpoint_handler, AnyioQueue(64), 0
    
    def create_cpu_video_embedding_endpoint_handler(self, tokenizer, endpoint_model, cpu_label, endpoint=None):
        def handler(text=None, frames=None, tokenizer=tokenizer, endpoint_model=endpoint_model, cpu_label=cpu_label, endpoint=endpoint):
            """CPU handler for video embedding and text-video similarity.
            
            Args:
                text: Optional text input
                frames: Optional video frames/images
                
            Returns:
                Dictionary with embeddings and/or similarity scores with implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = False  # Start with assuming real implementation
            
            # Check if we have valid model components
            has_valid_model = (
                endpoint is not None and 
                hasattr(endpoint, "eval") and 
                tokenizer is not None and
                hasattr(tokenizer, "__call__")
            )
            
            # If we don't have valid components, we'll need to use mock implementation
            if not has_valid_model:
                is_mock = True
                print("Invalid model components detected, will use mock implementation")
            
            # Create result dictionary for embeddings
            result = {}
            
            # Try to use real implementation first
            if not is_mock:
                try:
                    # Initialize model in evaluation mode
                    endpoint.eval()
                    
                    # Process with torch.no_grad to save memory
                    with self.torch.no_grad():
                        # Process text input if provided
                        if text is not None:
                            try:
                                # Process input through tokenizer
                                # Try different input formats based on what the tokenizer accepts
                                try:
                                    # Standard format first
                                    text_inputs = tokenizer(
                                        text=text,
                                        return_tensors="pt",
                                        padding=True
                                    )
                                except Exception as standard_error:
                                    print(f"Standard tokenizer format failed: {standard_error}")
                                    try:
                                        # Try alternative format
                                        text_inputs = tokenizer(
                                            text,
                                            return_tensors="pt",
                                            padding=True
                                        )
                                    except Exception as alt_error:
                                        print(f"Alternative tokenizer format failed: {alt_error}")
                                        raise
                                
                                # Run model inference with text inputs only
                                text_outputs = endpoint(**text_inputs)
                                
                                # Extract text embeddings - Try multiple possible output formats
                                if hasattr(text_outputs, "text_embeds"):
                                    result["text_embedding"] = text_outputs.text_embeds
                                    print("Found text embeddings in text_embeds attribute")
                                elif hasattr(text_outputs, "text_model_output") and hasattr(text_outputs.text_model_output, "pooler_output"):
                                    result["text_embedding"] = text_outputs.text_model_output.pooler_output
                                    print("Found text embeddings in text_model_output.pooler_output")
                                elif hasattr(text_outputs, "pooler_output"):
                                    result["text_embedding"] = text_outputs.pooler_output
                                    print("Found text embeddings in pooler_output")
                                elif hasattr(text_outputs, "last_hidden_state"):
                                    # Use mean pooling as a fallback for models that return hidden states
                                    print("Using mean pooling on last_hidden_state for text embeddings")
                                    # Apply attention mask if available
                                    if "attention_mask" in text_inputs:
                                        mask = text_inputs["attention_mask"].unsqueeze(-1)
                                        embeddings = text_outputs.last_hidden_state * mask
                                        result["text_embedding"] = embeddings.sum(1) / mask.sum(1)
                                    else:
                                        # Simple mean if no mask available
                                        result["text_embedding"] = text_outputs.last_hidden_state.mean(1)
                                else:
                                    # Try to find any attribute that might contain embeddings
                                    found_embedding = False
                                    for attr_name in dir(text_outputs):
                                        if "embed" in attr_name.lower() and not attr_name.startswith("_"):
                                            try:
                                                embed_attr = getattr(text_outputs, attr_name)
                                                if hasattr(embed_attr, "shape") and len(embed_attr.shape) >= 2:
                                                    result["text_embedding"] = embed_attr
                                                    print(f"Found text embeddings in {attr_name}")
                                                    found_embedding = True
                                                    break
                                            except:
                                                continue
                                    
                                    # Fallback if text embeddings not directly accessible
                                    if not found_embedding:
                                        is_mock = True
                                        print("Text embeddings not found in model output, using mock implementation")
                            except Exception as text_error:
                                print(f"Error processing text input: {text_error}")
                                is_mock = True
                        
                        # Process video frames if provided
                        if frames is not None and not is_mock:
                            try:
                                # Process video frames through tokenizer
                                if isinstance(frames, list) and len(frames) > 0:
                                    try:
                                        # Standard format first
                                        video_inputs = tokenizer(
                                            images=frames,
                                            return_tensors="pt",
                                            padding=True
                                        )
                                    except Exception as standard_error:
                                        print(f"Standard image tokenizer format failed: {standard_error}")
                                        try:
                                            # Try alternative format without named parameter
                                            video_inputs = tokenizer(
                                                frames,
                                                return_tensors="pt",
                                                padding=True
                                            )
                                        except Exception as alt_error:
                                            print(f"Alternative image tokenizer format failed: {alt_error}")
                                            raise
                                    
                                    # Run model inference with video inputs
                                    video_outputs = endpoint(**video_inputs)
                                    
                                    # Extract video embeddings - Try multiple possible output formats
                                    if hasattr(video_outputs, "image_embeds"):
                                        result["video_embedding"] = video_outputs.image_embeds
                                        print("Found video embeddings in image_embeds attribute")
                                    elif hasattr(video_outputs, "vision_model_output") and hasattr(video_outputs.vision_model_output, "pooler_output"):
                                        result["video_embedding"] = video_outputs.vision_model_output.pooler_output
                                        print("Found video embeddings in vision_model_output.pooler_output")
                                    elif hasattr(video_outputs, "pooler_output"):
                                        result["video_embedding"] = video_outputs.pooler_output
                                        print("Found video embeddings in pooler_output")
                                    elif hasattr(video_outputs, "last_hidden_state"):
                                        # Use mean pooling as a fallback for models that return hidden states
                                        print("Using mean pooling on last_hidden_state for video embeddings")
                                        result["video_embedding"] = video_outputs.last_hidden_state.mean(1)
                                    else:
                                        # Try to find any attribute that might contain embeddings
                                        found_embedding = False
                                        for attr_name in dir(video_outputs):
                                            if ("embed" in attr_name.lower() or "visual" in attr_name.lower() or "vision" in attr_name.lower()) and not attr_name.startswith("_"):
                                                try:
                                                    embed_attr = getattr(video_outputs, attr_name)
                                                    if hasattr(embed_attr, "shape") and len(embed_attr.shape) >= 2:
                                                        result["video_embedding"] = embed_attr
                                                        print(f"Found video embeddings in {attr_name}")
                                                        found_embedding = True
                                                        break
                                                except:
                                                    continue
                                        
                                        # Fallback if video embeddings not directly accessible
                                        if not found_embedding:
                                            is_mock = True
                                            print("Video embeddings not found in model output, using mock implementation")
                                else:
                                    # Input not in expected format
                                    is_mock = True
                                    print("Video frames not in expected format")
                            except Exception as video_error:
                                print(f"Error processing video frames: {video_error}")
                                is_mock = True
                        
                        # If we have both text and video embeddings, calculate similarity
                        if "text_embedding" in result and "video_embedding" in result and not is_mock:
                            try:
                                text_emb = result["text_embedding"]
                                video_emb = result["video_embedding"]
                                
                                # Normalize embeddings with careful handling of dimensions
                                # Make sure embeddings are of proper shape for matrix multiplication
                                if len(text_emb.shape) == 1:
                                    text_emb = text_emb.unsqueeze(0)  # Add batch dimension if missing
                                if len(video_emb.shape) == 1:
                                    video_emb = video_emb.unsqueeze(0)  # Add batch dimension if missing
                                
                                # Normalize embeddings
                                text_norm = text_emb.norm(dim=-1, keepdim=True)
                                video_norm = video_emb.norm(dim=-1, keepdim=True)
                                
                                # Check for zero norms to avoid NaN issues
                                if self.torch.all(text_norm > 0) and self.torch.all(video_norm > 0):
                                    text_emb_norm = text_emb / text_norm
                                    video_emb_norm = video_emb / video_norm
                                    
                                    # Calculate similarity score with proper transposition
                                    if video_emb_norm.shape[0] == 1:
                                        # Single video embedding
                                        result["similarity"] = self.torch.matmul(text_emb_norm, video_emb_norm.transpose(0, 1))
                                    else:
                                        # Multiple video embeddings
                                        result["similarity"] = self.torch.matmul(text_emb_norm, video_emb_norm.T)
                                    
                                    print(f"Successfully calculated similarity: {result['similarity'].item() if result['similarity'].numel() == 1 else result['similarity'].shape}")
                                else:
                                    # Handle zero norm case
                                    is_mock = True
                                    print("Zero norm detected in embeddings")
                            except Exception as sim_error:
                                print(f"Error calculating similarity: {sim_error}")
                                is_mock = True
                
                except Exception as e:
                    print(f"Error in real implementation: {e}")
                    is_mock = True
            
            # If real implementation failed or wasn't available, use mock implementation
            if is_mock:
                print("Using mock implementation for XCLIP handler")
                # Create mock embeddings
                if text is not None and "text_embedding" not in result:
                    result["text_embedding"] = self.torch.randn(1, 512)
                
                if frames is not None and "video_embedding" not in result:
                    result["video_embedding"] = self.torch.randn(1, 512)
                
                if text is not None and frames is not None and "similarity" not in result:
                    result["similarity"] = self.torch.tensor([[0.8]])  # Mock similarity score
            
            # Add implementation type to result
            result["implementation_type"] = "REAL" if not is_mock else "MOCK"
            
            # Return the appropriate result
            return result
        return handler
    
    def create_qualcomm_video_embedding_endpoint_handler(self, tokenizer, endpoint_model, qualcomm_label, endpoint=None):
        def handler(text=None, frames=None, tokenizer=tokenizer, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint):
            """Qualcomm handler for video embedding and text-video similarity.
            
            Args:
                text: Optional text input
                frames: Optional video frames/images
                
            Returns:
                Dictionary with embeddings and/or similarity scores with implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = True  # Default to mock for Qualcomm implementation (will update to REAL if successful)
            
            # Initialize model if available
            if endpoint is not None and "eval" in dir(endpoint):
                try:
                    endpoint.eval()
                except Exception as e:
                    print(f"Error putting model in eval mode: {e}")
            
            # Check if we have self.snpe_utils available for inference
            has_snpe = hasattr(self, 'snpe_utils') and self.snpe_utils is not None and hasattr(self.snpe_utils, 'run_inference')
            
            # Try to use real implementation if available
            if has_snpe and endpoint is not None and tokenizer is not None:
                try:
                    # Process inputs
                    inputs = {}
                    
                    # Process text if provided
                    if text is not None:
                        text_inputs = tokenizer(text=text, return_tensors="np")
                        for key, value in text_inputs.items():
                            inputs[key] = value
                    
                    # Process frames if provided
                    if frames is not None:
                        if isinstance(frames, list) and len(frames) > 0:
                            frame_inputs = tokenizer(images=frames, return_tensors="np")
                            for key, value in frame_inputs.items():
                                inputs[key] = value
                    
                    # Run inference if we have inputs
                    if inputs:
                        inference_results = self.snpe_utils.run_inference(endpoint, inputs)
                        
                        # If we got results, we're using real implementation
                        if inference_results:
                            is_mock = False
                            
                            # Extract results
                            result = {}
                            
                            # Map outputs to standard format
                            if "text_embeds" in inference_results and text is not None:
                                result["text_embedding"] = self.torch.tensor(inference_results["text_embeds"])
                            
                            if "image_embeds" in inference_results and frames is not None:
                                result["video_embedding"] = self.torch.tensor(inference_results["image_embeds"])
                            
                            # Calculate similarity if we have both embeddings
                            if "text_embeds" in inference_results and "image_embeds" in inference_results:
                                text_embeds = self.torch.tensor(inference_results["text_embeds"])
                                image_embeds = self.torch.tensor(inference_results["image_embeds"])
                                
                                # Normalize embeddings
                                text_norm = text_embeds.norm(dim=-1, keepdim=True)
                                image_norm = image_embeds.norm(dim=-1, keepdim=True)
                                
                                # Check for zero norms to avoid NaN issues
                                if torch.all(text_norm > 0) and torch.all(image_norm > 0):
                                    text_embeds_norm = text_embeds / text_norm
                                    image_embeds_norm = image_embeds / image_norm
                                    
                                    # Calculate similarity
                                    result["similarity"] = self.torch.matmul(text_embeds_norm, image_embeds_norm.T)
                                else:
                                    # Fallback for zero norm case
                                    result["similarity"] = self.torch.tensor([[0.8]])
                                    is_mock = True
                            
                            # Add implementation type and return if successful
                            if result:
                                result["implementation_type"] = "REAL"
                                return result
                except Exception as e:
                    print(f"Error in Qualcomm real implementation: {e}")
                    is_mock = True
            
            # Create mock embeddings for Qualcomm implementation as fallback
            result = {}
            
            # Create text embedding if text input is provided
            if text is not None:
                text_embedding = self.torch.randn(1, 512)
                result["text_embedding"] = text_embedding
            
            # Create video embedding if frames are provided
            if frames is not None:
                video_embedding = self.torch.randn(1, 512)
                result["video_embedding"] = video_embedding
            
            # If both inputs are provided, calculate similarity
            if text is not None and frames is not None:
                similarity = self.torch.tensor([[0.8]])  # Mock similarity score
                result["similarity"] = similarity
            
            # Add implementation type
            result["implementation_type"] = "MOCK"
            
            # Return the appropriate result
            return result
        return handler
    
    def create_apple_video_embedding_endpoint_handler(self, tokenizer, endpoint_model, apple_label, endpoint=None):
        def handler(text=None, frames=None, tokenizer=tokenizer, endpoint_model=endpoint_model, apple_label=apple_label, endpoint=endpoint):
            """Apple Silicon (CoreML) handler for video embedding and text-video similarity.
            
            Args:
                text: Optional text input
                frames: Optional video frames/images
                
            Returns:
                Dictionary with embeddings and/or similarity scores with implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = True  # Default to mock for Apple implementation (will update to REAL if successful)
            
            # Initialize model if available
            if endpoint is not None and "eval" in dir(endpoint):
                try:
                    endpoint.eval()
                except Exception as e:
                    print(f"Error putting model in eval mode: {e}")
            
            # Check if we have CoreML utilities available
            has_coreml = hasattr(self, 'coreml_utils') and self.coreml_utils is not None and hasattr(self.coreml_utils, 'run_inference')
            
            # Try to use real implementation if available
            if has_coreml and endpoint is not None and tokenizer is not None:
                try:
                    # Process inputs
                    inputs = {}
                    
                    # Handle text input
                    if text is not None:
                        if isinstance(text, str):
                            text_inputs = tokenizer(text=text, return_tensors="np")
                            for key, value in text_inputs.items():
                                inputs[key] = value
                    
                    # Handle frame/image input
                    if frames is not None:
                        if isinstance(frames, list) and len(frames) > 0:
                            frame_inputs = tokenizer(images=frames, return_tensors="np")
                            for key, value in frame_inputs.items():
                                inputs[key] = value
                    
                    # If we have valid inputs, run inference
                    if inputs:
                        outputs = self.coreml_utils.run_inference(endpoint, inputs)
                        
                        # If we have outputs, we're using real implementation
                        if outputs:
                            is_mock = False
                            result = {}
                            
                            # Extract text embeddings
                            if 'text_embeds' in outputs and text is not None:
                                result['text_embedding'] = self.torch.tensor(outputs['text_embeds'])
                                
                            # Extract image/video embeddings
                            if 'image_embeds' in outputs and frames is not None:
                                result['video_embedding'] = self.torch.tensor(outputs['image_embeds'])
                                
                            # Calculate similarity if we have both embeddings
                            if 'text_embeds' in outputs and 'image_embeds' in outputs:
                                text_emb = self.torch.tensor(outputs['text_embeds'])
                                image_emb = self.torch.tensor(outputs['image_embeds'])
                                
                                # Normalize embeddings with safe operations
                                text_norm = text_emb.norm(dim=-1, keepdim=True)
                                image_norm = image_emb.norm(dim=-1, keepdim=True)
                                
                                # Check for zero norms to avoid NaN issues
                                if self.torch.all(text_norm > 0) and self.torch.all(image_norm > 0):
                                    text_emb_norm = text_emb / text_norm
                                    image_emb_norm = image_emb / image_norm
                                    
                                    # Calculate similarity
                                    result['similarity'] = self.torch.matmul(text_emb_norm, image_emb_norm.T)
                                else:
                                    # Fallback for zero norm case
                                    result['similarity'] = self.torch.tensor([[0.8]])
                                    is_mock = True
                            
                            # Return with REAL implementation type if successful
                            if result:
                                result["implementation_type"] = "REAL"
                                return result
                except Exception as e:
                    print(f"Error in Apple Silicon real implementation: {e}")
                    is_mock = True
            
            # Create mock embeddings for Apple Silicon implementation as fallback
            result = {}
            
            # Create text embedding if text input is provided
            if text is not None:
                text_embedding = self.torch.randn(1, 512)
                result["text_embedding"] = text_embedding
            
            # Create video embedding if frames are provided
            if frames is not None:
                video_embedding = self.torch.randn(1, 512)
                result["video_embedding"] = video_embedding
            
            # If both inputs are provided, calculate similarity
            if text is not None and frames is not None:
                similarity = self.torch.tensor([[0.8]])  # Mock similarity score
                result["similarity"] = similarity
            
            # Add implementation type
            result["implementation_type"] = "MOCK"
            
            # Return the appropriate result
            return result
        return handler
    
    def create_cuda_video_embedding_endpoint_handler(self, endpoint=None, tokenizer=None, endpoint_model=None, cuda_label=None):
        """Creates a CUDA handler for XClip video and text embedding extraction.
        
        Args:
            endpoint: The model endpoint
            tokenizer: The text/image processor
            endpoint_model: The model name or path
            cuda_label: Label to identify this endpoint
            
        Returns:
            A handler function for CUDA XClip endpoint
        """
        def handler(text=None, frames=None, endpoint=endpoint, tokenizer=tokenizer, endpoint_model=endpoint_model, cuda_label=cuda_label):
            """CUDA handler for video embedding and text-video similarity.
            
            Args:
                text: Optional text input
                frames: Optional video frames/images
                
            Returns:
                Dictionary with embeddings and/or similarity scores with implementation type
            """
            # Import torch directly inside the handler
            import torch
            import numpy as np
            
            # Flag to track if we're using real implementation or mock
            is_mock = False
            
            # Initialize model if available
            if endpoint is not None and hasattr(endpoint, "eval"):
                try:
                    endpoint.eval()
                except Exception as e:
                    print(f"Error putting model in eval mode: {e}")
                    is_mock = True
            else:
                is_mock = True
            
            # Check if CUDA is truly available for inference
            cuda_available = (
                hasattr(torch, 'cuda') and 
                torch.cuda.is_available() and 
                endpoint is not None and
                hasattr(endpoint, "device") and
                "cuda" in str(endpoint.device)
            )
            
            # If CUDA isn't available, we'll have to use mock implementation
            if not cuda_available:
                is_mock = True
                
            try:
                with torch.no_grad():
                    # Clean GPU cache if available
                    if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    
                    result = {}
                    
                    # Process text if provided
                    if text is not None and tokenizer is not None and endpoint is not None and not is_mock:
                        try:
                            # Process text input
                            text_inputs = tokenizer(
                                text=text,
                                return_tensors="pt",
                                padding=True
                            )
                            
                            # Move inputs to GPU
                            for key in list(text_inputs.keys()):  # Use list to avoid dict size change issues
                                if isinstance(text_inputs[key], torch.Tensor):
                                    text_inputs[key] = text_inputs[key].to(endpoint.device)
                            
                            # Run model for text embedding
                            text_output = endpoint(**text_inputs)
                            
                            # Extract text embedding
                            if hasattr(text_output, "text_embeds"):
                                text_embedding = text_output.text_embeds
                                result["text_embedding"] = text_embedding.cpu().detach()
                            else:
                                # Create mock embedding if structure unexpected
                                print("Model output doesn't contain text_embeds attribute, using mock embedding")
                                text_embedding = torch.randn(1, 512, device=endpoint.device)
                                result["text_embedding"] = text_embedding.cpu()
                                is_mock = True
                        except Exception as text_error:
                            print(f"Error processing text input: {text_error}")
                            # Create mock text embedding on fallback
                            text_embedding = torch.randn(1, 512)
                            result["text_embedding"] = text_embedding
                            is_mock = True
                    elif text is not None:
                        # Create mock text embedding if no model/tokenizer
                        text_embedding = torch.randn(1, 512)
                        result["text_embedding"] = text_embedding
                        is_mock = True
                    
                    # Process video frames if provided
                    if frames is not None and tokenizer is not None and endpoint is not None and not is_mock:
                        try:
                            # Process video/image input
                            if isinstance(frames, list) and len(frames) > 0:
                                frame_inputs = tokenizer(
                                    images=frames,
                                    return_tensors="pt",
                                    padding=True
                                )
                                
                                # Move inputs to GPU
                                for key in list(frame_inputs.keys()):  # Use list to avoid dict size change issues
                                    if isinstance(frame_inputs[key], torch.Tensor):
                                        frame_inputs[key] = frame_inputs[key].to(endpoint.device)
                                
                                # Run model for video embedding
                                video_output = endpoint(**frame_inputs)
                                
                                # Extract video embedding
                                if hasattr(video_output, "image_embeds"):
                                    video_embedding = video_output.image_embeds
                                    result["video_embedding"] = video_embedding.cpu().detach()
                                else:
                                    # Create mock embedding if structure unexpected
                                    print("Model output doesn't contain image_embeds attribute, using mock embedding")
                                    video_embedding = torch.randn(1, 512, device=endpoint.device)
                                    result["video_embedding"] = video_embedding.cpu()
                                    is_mock = True
                            else:
                                # Create mock embedding if frames not in expected format
                                print("Frames not in expected format, using mock embedding")
                                video_embedding = torch.randn(1, 512)
                                result["video_embedding"] = video_embedding
                                is_mock = True
                        except Exception as video_error:
                            print(f"Error processing video input: {video_error}")
                            # Create mock video embedding on fallback
                            video_embedding = torch.randn(1, 512)
                            result["video_embedding"] = video_embedding
                            is_mock = True
                    elif frames is not None:
                        # Create mock video embedding if no model/tokenizer
                        video_embedding = torch.randn(1, 512)
                        result["video_embedding"] = video_embedding
                        is_mock = True
                    
                    # Calculate similarity if both embeddings are available
                    if "text_embedding" in result and "video_embedding" in result:
                        try:
                            text_emb = result["text_embedding"]
                            video_emb = result["video_embedding"]
                            
                            # Normalize embeddings with safe operations
                            text_norm = text_emb.norm(dim=-1, keepdim=True)
                            video_norm = video_emb.norm(dim=-1, keepdim=True)
                            
                            # Check for zero norms to avoid NaN issues
                            if torch.all(text_norm > 0) and torch.all(video_norm > 0):
                                text_emb_norm = text_emb / text_norm
                                video_emb_norm = video_emb / video_norm
                                
                                # Calculate similarity
                                similarity = torch.matmul(text_emb_norm, video_emb_norm.transpose(0, 1))
                                result["similarity"] = similarity
                            else:
                                print("Zero norm detected in embeddings, using mock similarity")
                                result["similarity"] = torch.tensor([[0.8]])
                                is_mock = True
                        except Exception as sim_error:
                            print(f"Error calculating similarity: {sim_error}")
                            # Create mock similarity on fallback
                            result["similarity"] = torch.tensor([[0.8]])
                            is_mock = True
                    
                    # Clean GPU memory
                    if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    
                    # Add implementation type to result
                    result["implementation_type"] = "REAL" if not is_mock else "MOCK"
                    return result
            except Exception as e:
                print(f"CUDA video/text embedding error: {e}")
                # Return mock result on complete failure
                result = {}
                if text is not None:
                    result["text_embedding"] = torch.randn(1, 512)
                if frames is not None:
                    result["video_embedding"] = torch.randn(1, 512)
                if text is not None and frames is not None:
                    result["similarity"] = torch.tensor([[0.8]])
                result["implementation_type"] = "MOCK"
                return result
        return handler

    def create_openvino_video_embedding_endpoint_handler(self, endpoint=None, tokenizer=None, endpoint_model=None, openvino_label=None):
        """Creates an OpenVINO handler for XClip video and text embedding extraction.
        
        Args:
            endpoint: The OpenVINO model endpoint
            tokenizer: The text/image processor
            endpoint_model: The model name or path
            openvino_label: Label to identify this endpoint
            
        Returns:
            A handler function for OpenVINO XClip endpoint
        """
        def handler(text=None, frames=None, endpoint=endpoint, tokenizer=tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label):
            """OpenVINO handler for video embedding and text-video similarity.
            
            Args:
                text: Optional text input
                frames: Optional video frames input (could be a list of images or video path)
                
            Returns:
                Dictionary with embeddings and/or similarity scores
            """
            # Flag to track if we're using real implementation or mock
            is_mock = False
            self.np.random.seed(0)
            video_frames = None
            
            # Process video input if provided
            if frames is not None:
                try:
                    # Handle different types of video input
                    if isinstance(frames, str):
                        # It's a path to a video
                        videoreader = None
                        if os.path.exists(frames):
                            videoreader = self.decord.VideoReader(frames, num_threads=1, ctx=self.decord.cpu(0))
                        elif "http" in frames:
                            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                                f.write(requests.get(frames).content)
                                f.flush()
                                videoreader = self.decord.VideoReader(f.name, num_threads=1, ctx=self.decord.cpu(0))
                        
                        if videoreader is not None:
                            videoreader.seek(0)
                            indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=len(videoreader))
                            video_frames = videoreader.get_batch(indices).asnumpy()
                    
                    elif isinstance(frames, list):
                        # It's already a list of frames
                        video_frames = self.np.stack([self.np.array(frame) for frame in frames])
                except Exception as e:
                    print(f"Error processing video frames: {e}")
                    video_frames = None
            
            # Fall back to mock data if we couldn't process the input
            if video_frames is None and frames is not None:
                # Create mock video frames
                video_frames = self.np.random.randint(0, 255, (32, 224, 224, 3), dtype=self.np.uint8)
                is_mock = True
            
            # Process text if provided
            text_input = ""
            if text is not None:
                text_input = text if isinstance(text, str) else ""
            
            # Check if we need to use mock implementation
            if endpoint is None or not hasattr(endpoint, "__call__"):
                is_mock = True
                result = {}
                if text is not None:
                    result["text_embedding"] = self.torch.randn(1, 512)
                if frames is not None:
                    result["video_embedding"] = self.torch.randn(1, 512)
                if text is not None and frames is not None:
                    result["similarity"] = self.torch.tensor([[0.8]])
                
                # Add implementation type
                result["implementation_type"] = "MOCK"
                return result
            
            # Try to use real implementation for inference
            try:
                # Only proceed if we have the necessary inputs
                if (text_input or video_frames) and tokenizer is not None and hasattr(tokenizer, "__call__"):
                    # Process the inputs with tokenizer
                    try:
                        processed_data = tokenizer(
                            text=text_input,
                            videos=list(video_frames) if video_frames is not None else None,
                            return_tensors="pt",
                            padding=True,
                        )
                        
                        # Use a copy of keys to avoid "dictionary changed size during iteration" error
                        keys_to_check = list(processed_data.keys())
                        
                        # Create a standardized processed data dict with fallbacks
                        new_processed_data = {
                            'input_ids': processed_data.get("input_ids", self.torch.ones((1, 10), dtype=self.torch.long)),
                            'attention_mask': processed_data.get("attention_mask", self.torch.ones((1, 10), dtype=self.torch.long)),
                            'pixel_values': processed_data.get("pixel_values", self.torch.zeros((1, 3, 224, 224), dtype=self.torch.float32))
                        }
                        
                        # Try to run inference with OpenVINO model
                        try:
                            inference_results = endpoint_model(dict(new_processed_data))
                            
                            # Safely extract output values
                            if inference_results and hasattr(inference_results, "values"):
                                results_list = list(inference_results.values())
                                
                                # Safe list indexing with length check
                                if len(results_list) >= 6:
                                    text_embeddings = results_list[3]
                                    video_embeddings = results_list[5]
                                    
                                    # Return appropriate embeddings based on inputs
                                    if text is not None and frames is not None:
                                        return {
                                            'video_embedding': video_embeddings,
                                            'text_embedding': text_embeddings,
                                            'similarity': self.torch.matmul(
                                                text_embeddings / text_embeddings.norm(dim=-1, keepdim=True),
                                                (video_embeddings / video_embeddings.norm(dim=-1, keepdim=True)).T
                                            ),
                                            'implementation_type': 'REAL'
                                        }
                                    elif text is not None:
                                        return {
                                            'text_embedding': text_embeddings,
                                            'implementation_type': 'REAL'
                                        }
                                    elif frames is not None:
                                        return {
                                            'video_embedding': video_embeddings,
                                            'implementation_type': 'REAL'
                                        }
                                else:
                                    print(f"OpenVINO inference results list length insufficient: {len(results_list)}")
                                    is_mock = True
                            else:
                                print("OpenVINO inference results invalid format")
                                is_mock = True
                        except Exception as e:
                            print(f"Error in OpenVINO inference: {e}")
                            is_mock = True
                    except Exception as e:
                        print(f"Error processing inputs with tokenizer: {e}")
                        is_mock = True
                else:
                    is_mock = True
            except Exception as e:
                print(f"Error in OpenVINO handler: {e}")
                is_mock = True
            
            # Fall back to mock data if real implementation failed
            if is_mock:
                result = {}
                if text is not None:
                    result["text_embedding"] = self.torch.randn(1, 512)
                if frames is not None:
                    result["video_embedding"] = self.torch.randn(1, 512)
                if text is not None and frames is not None:
                    result["similarity"] = self.torch.tensor([[0.8]])
                
                # Add implementation type
                result["implementation_type"] = "MOCK"
                return result
            
            # This should never be reached, but just in case
            return {"error": "Unexpected execution path", "implementation_type": "MOCK"}
        return handler


    def openvino_skill_convert(self, model_name, model_dst_path, task, weight_format, hfmodel=None, hfprocessor=None):
        if hfmodel is None:
            hfmodel = self.transformers.AutoModel.from_pretrained(model_name, torch_dtype=self.torch.float16)
    
        if hfprocessor is None:
            hfprocessor = self.transformers.AutoProcessor.from_pretrained(model_name)

        if hfprocessor is not None:
            text = "Replace me by any text you'd like."
            ##xclip processor
            video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
            self.np.random.seed(0)
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(requests.get(video_url).content)
                f.flush()
                videoreader = self.decord.VideoReader(f.name, num_threads=1, ctx=self.decord.cpu(0))
                videoreader.seek(0)
                indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=len(videoreader))
                video = videoreader.get_batch(indices).asnumpy()
                processed_data = hfprocessor(
                    text=text,
                    videos=list(video),
                    return_tensors="pt",
                    padding=True,
                )
                results = hfmodel(**processed_data)
                hfmodel.config.torchscript = True
                ov_model = self.ov.convert_model(hfmodel,  example_input=dict(processed_data))
                if not os.path.exists(model_dst_path):
                    os.mkdir(model_dst_path)
                self.ov.save_model(ov_model, os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
                ov_model = self.ov.compile_model(ov_model)
                hfmodel = None
        return ov_model

    def create_qualcomm_xclip_endpoint_handler(self, processor, endpoint_model, qualcomm_label, endpoint):
        """Creates an endpoint handler for Qualcomm hardware.
        
        Args:
            processor: The processor for text and image inputs
            endpoint_model: The model name or path
            qualcomm_label: Label to identify this endpoint
            endpoint: The SNPE model endpoint
            
        Returns:
            A handler function for the Qualcomm endpoint
        """
        def handler(text_input=None, image_input=None, processor=processor, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint):
            try:
                inputs = {}
                
                # Process text input if provided
                if text_input is not None:
                    if isinstance(text_input, str):
                        text_inputs = processor(text=text_input, return_tensors="np")
                    else:
                        # Assume it's a batch of texts
                        text_inputs = processor(text=text_input, return_tensors="np", padding=True)
                        
                    for key, value in text_inputs.items():
                        inputs[key] = value
                
                # Process image input if provided
                if image_input is not None:
                    if isinstance(image_input, str):
                        # Load image from URL or file
                        image = load_image(image_input)
                        image_inputs = processor(images=image, return_tensors="np")
                    elif isinstance(image_input, list):
                        # Process a batch of images
                        images = [load_image(img) for img in image_input]
                        image_inputs = processor(images=images, return_tensors="np", padding=True)
                    else:
                        # Assume it's already a PIL Image
                        image_inputs = processor(images=image_input, return_tensors="np")
                    
                    for key, value in image_inputs.items():
                        inputs[key] = value
                
                # Run inference with SNPE
                results = self.snpe_utils.run_inference(endpoint, inputs)
                
                # Process results
                output = {}
                
                # Convert numpy arrays to torch tensors
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        output[key] = self.torch.tensor(value)
                    else:
                        output[key] = value
                
                # Calculate similarity if both text and image embeddings are available
                if "text_embeds" in results and "image_embeds" in results:
                    text_embeds = self.torch.tensor(results["text_embeds"])
                    image_embeds = self.torch.tensor(results["image_embeds"])
                    
                    # Normalize embeddings
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity
                    similarity = self.torch.matmul(text_embeds, image_embeds.T)
                    output["similarity"] = similarity
                
                return output
                
            except Exception as e:
                print(f"Error in Qualcomm XClip endpoint handler: {e}")
                return {"error": str(e)}
                
        return handler

    def create_apple_multimodal_endpoint_handler(self, endpoint, processor, model_name, apple_label):
        """Creates an Apple Silicon optimized handler for XClip multimodal processing."""
        def handler(x, y=None, endpoint=endpoint, processor=processor, model_name=model_name, apple_label=apple_label):
            try:
                # Process inputs
                if isinstance(x, str) and y is not None:
                    # Handle image + text input
                    if isinstance(y, str):
                        # Load image
                        image = load_image(y)
                        inputs = processor(
                            text=x,
                            images=image,
                            return_tensors="np",
                            padding=True
                        )
                    elif isinstance(y, list):
                        # Handle multiple images
                        images = [load_image(img_path) for img_path in y]
                        inputs = processor(
                            text=[x] * len(images),
                            images=images,
                            return_tensors="np",
                            padding=True
                        )
                else:
                    inputs = x
                
                # Convert inputs to CoreML format
                input_dict = {}
                for key, value in inputs.items():
                    if hasattr(value, 'numpy'):
                        input_dict[key] = value.numpy()
                    else:
                        input_dict[key] = value
                
                # Run inference
                outputs = self.coreml_utils.run_inference(endpoint, input_dict)
                
                # Process outputs
                result = {}
                
                # Extract text embeddings
                if 'text_embeds' in outputs:
                    text_embeddings = self.torch.tensor(outputs['text_embeds'])
                    result['text_embedding'] = text_embeddings
                    
                # Extract image embeddings
                if 'image_embeds' in outputs:
                    image_embeddings = self.torch.tensor(outputs['image_embeds'])
                    result['image_embedding'] = image_embeddings
                    
                # If we have both embeddings, compute similarity
                if 'text_embeds' in outputs and 'image_embeds' in outputs:
                    text_emb = self.torch.tensor(outputs['text_embeds'])
                    image_emb = self.torch.tensor(outputs['image_embeds'])
                    
                    # Normalize embeddings
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity
                    similarity = self.torch.matmul(text_emb, image_emb.T)
                    result['similarity'] = similarity
                
                # Return single embedding if that's all we have
                if len(result) == 1 and list(result.keys())[0] in ['text_embedding', 'image_embedding']:
                    return {'embedding': list(result.values())[0]}
                    
                return result if result else None
                
            except Exception as e:
                print(f"Error in Apple Silicon XClip handler: {e}")
                return None
                
        return handler