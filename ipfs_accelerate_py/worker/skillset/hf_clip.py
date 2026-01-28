import time
import anyio
from ..anyio_queue import AnyioQueue
from PIL import Image
import requests
from io import BytesIO
import os
import numpy as np

try:
    from ...common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except (ImportError, ValueError):
    try:
        from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except (ImportError, ValueError):
        try:
            from common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
        except ImportError:
            HAVE_STORAGE_WRAPPER = False

def load_image(image_file):
    """
    Load an image from a file path or URL and convert to RGB format.
    
    Args:
        image_file: Path to image file or URL
        
    Returns:
        PIL.Image: Loaded RGB image
    """
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return image

def load_image_tensor(image_file):
    """
    Load an image from a file path or URL and convert to OpenVINO tensor format.
    
    Args:
        image_file: Path to image file or URL
        
    Returns:
        OpenVINO Tensor: Image data as tensor
    """
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return ov.Tensor(image_data)

class hf_clip:
    """
    Hugging Face CLIP model implementation for various hardware backends.
    
    This class provides a standardized interface for running CLIP (Contrastive Language-Image 
    Pretraining) models across different hardware backends including CPU, CUDA, OpenVINO,
    Apple Silicon, and Qualcomm. It supports text-image similarity, image embedding, 
    and text embedding capabilities.
    
    The implementation provides both real model inference and mock functionality when
    hardware or dependencies are unavailable.
    """
    def __init__(self, resources=None, metadata=None):
        """
        Initialize the CLIP model handler.
        
        Args:
            resources: Dictionary of resources (torch, transformers, numpy)
            metadata: Dictionary of metadata for initialization
        
        Returns:
            None
        """
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper(auto_detect_ci=True)
            except Exception:
                self._storage = None
        else:
            self._storage = None
        
        self.resources = resources
        self.metadata = metadata if metadata else {}
        # Initialize backend-specific utilities
        self.snpe_utils = None
        self.coreml_utils = None
        self.ov = None
        # These redundant self-assignments are kept for backward compatibility
        self.create_openvino_image_embedding_endpoint_handler = self.create_openvino_image_embedding_endpoint_handler
        self.create_cuda_image_embedding_endpoint_handler = self.create_cuda_image_embedding_endpoint_handler
        self.create_cpu_image_embedding_endpoint_handler = self.create_cpu_image_embedding_endpoint_handler
        self.create_apple_image_embedding_endpoint_handler = self.create_apple_image_embedding_endpoint_handler
        self.create_qualcomm_image_embedding_endpoint_handler = self.create_qualcomm_image_embedding_endpoint_handler
        self.init_cpu = self.init_cpu
        self.init_qualcomm = self.init_qualcomm
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.init = self.init
        self.__test__ = self.__test__
        # Not auto-initializing to allow explicit initialization
        return None


    def init(self):
        """
        Initialize required resources for CLIP model.
        
        Loads torch, transformers, and numpy either from provided resources
        or by importing them directly. This method must be called before
        using any other methods.
        
        Returns:
            None
        """
        # Initialize PyTorch
        if "torch" not in list(self.resources.keys()):
            try:
                import torch
                self.torch = torch
            except ImportError:
                print("Failed to import torch. Some functionality will be limited.")
                self.torch = None
        else:
            self.torch = self.resources["torch"]

        # Initialize Transformers
        if "transformers" not in list(self.resources.keys()):
            try:
                import transformers
                self.transformers = transformers
            except ImportError:
                print("Failed to import transformers. Will use mock implementations.")
                self.transformers = None
        else:
            self.transformers = self.resources["transformers"]
            
        # Initialize NumPy
        if "numpy" not in list(self.resources.keys()):
            try:
                import numpy as np
                self.np = np
            except ImportError:
                print("Failed to import numpy. Some functionality will be limited.")
                self.np = None
        else:
            self.np = self.resources["numpy"]
            
        # Check if we have all required resources
        initialization_status = {
            "torch": self.torch is not None,
            "transformers": self.transformers is not None,
            "numpy": self.np is not None
        }
        
        print(f"CLIP initialization status: {initialization_status}")
        return None

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        """
        Test CLIP model with a simple text-image pair.
        
        Args:
            endpoint_model: Model name or path
            endpoint_handler: Handler function to test
            endpoint_label: Label for the endpoint (cpu, cuda, openvino, etc.)
            tokenizer: Tokenizer or processor for the model
            
        Returns:
            None
        """
        # Standard test inputs
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        image_1 = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        
        # Measure performance
        timestamp1 = time.time()
        test_result = None
        try:
            # Run inference through the handler
            test_batch = endpoint_handler(sentence_1, image_1)
            
            # Check if we got valid results
            if test_batch is not None and isinstance(test_batch, dict):
                if "similarity" in test_batch:
                    test_status = "PASSED - Similarity score computed"
                elif "text_embedding" in test_batch and "image_embedding" in test_batch:
                    test_status = "PASSED - Both embeddings computed"
                else:
                    test_status = "PARTIAL - Incomplete results"
            else:
                test_status = "FAILED - Invalid results"
                
            # Print results
            print(f"CLIP test status: {test_status}")
            print(f"Result type: {type(test_batch)}")
            
            # Determine if the result was from a real model or mock
            implementation_type = "REAL"
            if test_batch and any(k.endswith("_status") and test_batch[k] == "MOCK" for k in test_batch):
                implementation_type = "MOCK"
            print(f"Implementation type: {implementation_type}")
            
            test_result = test_batch
            
        except Exception as e:
            print(f"CLIP test error: {str(e)}")
            test_result = {"error": str(e)}
        
        # Calculate and print performance metrics
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        len_tokens = 1  # We're processing one sample
        tokens_per_second = len_tokens / elapsed_time
        
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        print(f"Samples processed: {len_tokens}")
        print(f"Samples per second: {tokens_per_second:.4f}")
        
        # Clean up resources based on backend
        if "openvino" not in endpoint_label and self.torch is not None:
            try:
                with self.torch.no_grad():
                    if hasattr(self.torch, "cuda") and hasattr(self.torch.cuda, "empty_cache"):
                        self.torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error cleaning up resources: {str(e)}")
                
        return test_result
    
    def init_cpu(self, model, device, cpu_label):
        """
        Initialize CLIP model for CPU inference
        
        Args:
            model: Model name or path (e.g., 'openai/clip-vit-base-patch32')
            device: Device to run on ('cpu')
            cpu_label: Label for CPU endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        print(f"Loading {model} for CPU inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Define a fallback function to create a simple test model
            def create_test_model():
                print("Creating minimal CLIP model for testing")
                torch_module = self.torch  # Store reference to avoid name lookup issues
                
                # Create simple model objects
                class SimpleProcessor:
                    def __init__(self):
                        self.torch = torch_module  # Use the class's torch reference
                        self.image_processor = self
                        
                    def __call__(self, images=None, text=None, return_tensors="pt", padding=True, **kwargs):
                        """Process images or text for CLIP input"""
                        batch_size = 1
                        result = {}
                        
                        if images is not None:
                            if isinstance(images, list):
                                batch_size = len(images)
                            # Create random pixel values tensor
                            result["pixel_values"] = self.torch.rand((batch_size, 3, 224, 224))
                            
                        if text is not None:
                            if isinstance(text, list):
                                batch_size = len(text)
                            # Create dummy text tensors
                            result["input_ids"] = self.torch.ones((batch_size, 77), dtype=self.torch.long)
                            result["attention_mask"] = self.torch.ones((batch_size, 77), dtype=self.torch.long)
                            
                        return result
                
                class SimpleModel:
                    def __init__(self):
                        self.config = SimpleConfig()
                        self.torch = torch_module  # Use the class's torch reference
                        
                    def __call__(self, **kwargs):
                        batch_size = 1
                        
                        # Determine batch size from inputs
                        if "pixel_values" in kwargs:
                            batch_size = kwargs["pixel_values"].shape[0]
                        elif "input_ids" in kwargs:
                            batch_size = kwargs["input_ids"].shape[0]
                            
                        embed_dim = 512
                        
                        # Create an output object that mimics the CLIPOutput structure
                        class CLIPOutput:
                            def __init__(self, batch_size, dim):
                                self.text_embeds = torch_module.randn(batch_size, dim)
                                self.image_embeds = torch_module.randn(batch_size, dim)
                                self.last_hidden_state = torch_module.randn(batch_size, 77, dim)
                                
                        return CLIPOutput(batch_size, embed_dim)
                        
                    def get_text_features(self, **kwargs):
                        """Return text embeddings"""
                        batch_size = kwargs["input_ids"].shape[0] if "input_ids" in kwargs else 1
                        return torch_module.randn(batch_size, 512)
                        
                    def get_image_features(self, **kwargs):
                        """Return image embeddings"""
                        batch_size = kwargs["pixel_values"].shape[0] if "pixel_values" in kwargs else 1
                        return torch_module.randn(batch_size, 512)
                
                class SimpleConfig:
                    def __init__(self):
                        self.hidden_size = 512
                        self.vocab_size = 49408
                        self.max_position_embeddings = 77
                        self.model_type = "clip"
                        
                # Create and return our simple processor and model
                return SimpleProcessor(), SimpleModel()
            
            # Try to load the real model if possible
            if isinstance(self.transformers, type):
                try:
                    # Try to load configuration
                    config = self.transformers.AutoConfig.from_pretrained(
                        model, 
                        trust_remote_code=True,
                        cache_dir=cache_dir
                    )
                    
                    # Try to load tokenizer and processor
                    tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                        model,
                        cache_dir=cache_dir,
                        trust_remote_code=True
                    )
                    
                    processor = self.transformers.CLIPProcessor.from_pretrained(
                        model, 
                        cache_dir=cache_dir,
                        trust_remote_code=True
                    )
                    
                    # Try to load model
                    endpoint = self.transformers.CLIPModel.from_pretrained(
                        model, 
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                        low_cpu_mem_usage=True
                    )
                    
                    print(f"Successfully loaded CLIP model: {model}")
                    
                except Exception as e:
                    print(f"Failed to load real CLIP model: {e}")
                    print("Creating test CLIP model instead")
                    processor, endpoint = create_test_model()
                    tokenizer = processor  # Use processor as tokenizer for simplicity
            else:
                # Create a test model if transformers is mocked
                processor, endpoint = create_test_model()
                tokenizer = processor  # Use processor as tokenizer
                
            # Create the handler
            endpoint_handler = self.create_cpu_image_embedding_endpoint_handler(
                tokenizer, 
                model, 
                cpu_label, 
                endpoint
            )
            
            return endpoint, tokenizer, endpoint_handler, AnyioQueue(64), 0
            
        except Exception as e:
            print(f"Error initializing CPU model: {e}")
            return None, None, None, None, 0
    
    def init_qualcomm(self, model, device, qualcomm_label):
        """
        Initialize CLIP model for Qualcomm hardware with SNPE support.
        
        Args:
            model: Model name or path for CLIP model
            device: Device to use ('qualcomm')
            qualcomm_label: Label for the Qualcomm hardware (e.g., 'qualcomm:0')
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, AnyioQueue, batch_size)
            If initialization fails, returns (None, None, None, None, 0)
        """
        # Ensure base dependencies are loaded
        self.init()
        
        # Helper function to create mock components for testing
        def _create_mock_processor():
            """Create a mock processor that returns valid tensor shapes"""
            class MockProcessor:
                def __init__(self):
                    self.image_processor = self
                
                def __call__(self, images=None, text=None, return_tensors="np", **kwargs):
                    """Process images and text for CLIP input"""
                    import numpy as np
                    result = {}
                    batch_size = 1
                    
                    if images is not None:
                        if isinstance(images, list):
                            batch_size = len(images)
                        # Create mock pixel values
                        result["pixel_values"] = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
                        
                    if text is not None:
                        if isinstance(text, list):
                            batch_size = len(text)
                        # Create mock input tensors
                        result["input_ids"] = np.zeros((batch_size, 77), dtype=np.int32)
                        result["attention_mask"] = np.ones((batch_size, 77), dtype=np.int32)
                        
                    return result
            
            return MockProcessor()
        
        # Helper function to create mock endpoint
        def _create_mock_endpoint():
            """Create a mock endpoint that returns valid tensor shapes"""
            class MockEndpoint:
                def __init__(self):
                    pass
                    
                def __call__(self, inputs):
                    """Return mock embeddings"""
                    import numpy as np
                    batch_size = 1
                    embed_dim = 512
                    
                    return {
                        "text_embeds": np.random.randn(batch_size, embed_dim).astype(np.float32),
                        "image_embeds": np.random.randn(batch_size, embed_dim).astype(np.float32)
                    }
            
            return MockEndpoint()
        
        # Initialize with both real and mock capabilities
        try:
            # Import SNPE utilities
            try:
                from .qualcomm_snpe_utils import get_snpe_utils
                self.snpe_utils = get_snpe_utils()
            except ImportError:
                print("(MOCK) Failed to import Qualcomm SNPE utilities")
                self.snpe_utils = None
                
            # Check if SNPE is available
            if self.snpe_utils is None or not self.snpe_utils.is_available():
                print("(MOCK) Qualcomm SNPE is not available, using mock implementation")
                processor = _create_mock_processor()
                endpoint = _create_mock_endpoint()
                endpoint_handler = self.create_qualcomm_image_embedding_endpoint_handler(
                    processor, processor, model, qualcomm_label, endpoint
                )
                return endpoint, processor, endpoint_handler, AnyioQueue(64), 0
            
            # Try to initialize real model
            real_processor = None
            try:
                if self.transformers is not None:
                    # Initialize tokenizer and processor
                    config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
                    tokenizer = self.transformers.AutoTokenizer.from_pretrained(model)
                    processor = self.transformers.CLIPProcessor.from_pretrained(model, trust_remote_code=True)
                    real_processor = processor
                else:
                    print("(MOCK) Transformers not available, using mock processor")
                    tokenizer = _create_mock_processor()
                    processor = tokenizer
            except Exception as e:
                print(f"(MOCK) Error loading real tokenizer/processor: {e}")
                tokenizer = _create_mock_processor()
                processor = tokenizer
            
            # Convert and load model for SNPE
            endpoint = None
            initialization_type = "MOCK"
            try:
                # Convert model path to be compatible with SNPE
                model_name = model.replace("/", "--")
                dlc_path = f"~/snpe_models/{model_name}_clip.dlc"
                dlc_path = os.path.expanduser(dlc_path)
                
                # Create directory if needed
                os.makedirs(os.path.dirname(dlc_path), exist_ok=True)
                
                # Convert or load the model
                if not os.path.exists(dlc_path):
                    print(f"(REAL) Converting {model} to SNPE format...")
                    self.snpe_utils.convert_model(model, "vision_text_dual", str(dlc_path))
                
                # Load the SNPE model
                endpoint = self.snpe_utils.load_model(str(dlc_path))
                
                # Optimize for the specific Qualcomm device if possible
                if ":" in qualcomm_label:
                    device_type = qualcomm_label.split(":")[1]
                    optimized_path = self.snpe_utils.optimize_for_device(dlc_path, device_type)
                    if optimized_path != dlc_path:
                        endpoint = self.snpe_utils.load_model(optimized_path)
                
                initialization_type = "REAL"
            except Exception as e:
                print(f"(MOCK) Error loading real model, using mock: {e}")
                endpoint = _create_mock_endpoint()
            
            # Create handler function
            endpoint_handler = self.create_qualcomm_image_embedding_endpoint_handler(
                tokenizer, 
                processor if real_processor is not None else processor, 
                model, 
                qualcomm_label, 
                endpoint
            )
            
            print(f"Initialized Qualcomm CLIP model ({initialization_type})")
            return endpoint, tokenizer, endpoint_handler, AnyioQueue(64), 0
        except Exception as e:
            print(f"Error initializing Qualcomm model: {e}")
            # Create mock components as fallback
            processor = _create_mock_processor()
            endpoint = _create_mock_endpoint()
            endpoint_handler = self.create_qualcomm_image_embedding_endpoint_handler(
                processor, processor, model, qualcomm_label, endpoint
            )
            print("(MOCK) Initialized Qualcomm CLIP model with mock components")
            return endpoint, processor, endpoint_handler, AnyioQueue(64), 0
            
    def init_apple(self, model, device, apple_label):
        """Initialize CLIP model for Apple Silicon hardware."""
        self.init()
        
        # Import CoreML utilities
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
            # Load processor directly from HuggingFace
            processor = self.transformers.CLIPProcessor.from_pretrained(model, trust_remote_code=True)
            
            # Convert model path to be compatible with CoreML
            model_name = model.replace("/", "--")
            mlmodel_path = f"~/coreml_models/{model_name}_clip.mlpackage"
            mlmodel_path = os.path.expanduser(mlmodel_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(mlmodel_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(mlmodel_path):
                print(f"Converting {model} to CoreML format...")
                self.coreml_utils.convert_model(model, "vision", str(mlmodel_path))
            
            # Load the CoreML model
            endpoint = self.coreml_utils.load_model(str(mlmodel_path))
            
            # Optimize for Apple Silicon if possible
            if ":" in apple_label:
                compute_units = apple_label.split(":")[1]
                optimized_path = self.coreml_utils.optimize_for_device(mlmodel_path, compute_units)
                if optimized_path != mlmodel_path:
                    endpoint = self.coreml_utils.load_model(optimized_path)
            
            endpoint_handler = self.create_apple_image_embedding_endpoint_handler(endpoint, processor, model, apple_label)
            
            return endpoint, processor, endpoint_handler, AnyioQueue(32), 0
        except Exception as e:
            print(f"Error initializing Apple Silicon CLIP model: {e}")
            return None, None, None, None, 0

    def init_cuda(self, model, device, cuda_label):
        """
        Initialize CLIP model for CUDA inference
        
        Args:
            model: Model name or path (e.g., 'openai/clip-vit-base-patch32')
            device: Device to run on ('cuda')
            cuda_label: CUDA device label (e.g., 'cuda:0')
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        print(f"Initializing CLIP model {model} for CUDA inference...")
        
        # Create batch_size and implementation type flags for result tracking
        batch_size = 0
        implementation_type = "MOCK"
        
        # First validate CUDA is actually available
        if not self.torch.cuda.is_available():
            print("CUDA requested but not available, using mock implementation")
            # Create empty endpoint and mock objects
            endpoint = None
            tokenizer = None
            endpoint_handler = self.create_cuda_image_embedding_endpoint_handler(
                tokenizer, endpoint_model=model, cuda_label=cuda_label, endpoint=endpoint
            )
            return endpoint, tokenizer, endpoint_handler, AnyioQueue(64), batch_size
            
        # Get device from cuda_label
        try:
            # Parse device ID from label
            device_parts = cuda_label.split(":")
            device_id = int(device_parts[1]) if len(device_parts) > 1 else 0
            
            # Validate device ID
            if device_id >= self.torch.cuda.device_count():
                print(f"Warning: CUDA device index {device_id} exceeds available devices ({self.torch.cuda.device_count()})")
                device_id = 0
                
            # Create device
            cuda_device = f"cuda:{device_id}"
            
            # Print device information
            print(f"Using CUDA device: {self.torch.cuda.get_device_name(device_id)} (index {device_id})")
        except Exception as e:
            print(f"Error parsing CUDA device: {e}, using default cuda:0")
            cuda_device = "cuda:0"
        
        try:
            # First attempt to load the model configuration
            try:
                config = self.transformers.AutoConfig.from_pretrained(
                    model, 
                    trust_remote_code=True, 
                    cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models")
                )
            except Exception as e:
                print(f"Error loading model configuration: {e}")
                
            # Then load tokenizer and processor
            try:
                tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                    model, 
                    trust_remote_code=True
                )
                processor = self.transformers.CLIPProcessor.from_pretrained(
                    model, 
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Error loading tokenizer/processor: {e}")
                tokenizer = None
                processor = None
            
            # Try to load the actual model with CUDA support
            try:
                # Empty cache before loading model
                self.torch.cuda.empty_cache()
                
                # Create model with half precision for better CUDA performance
                print(f"Loading CLIP model {model} to {cuda_device} with FP16 precision...")
                endpoint = self.transformers.CLIPModel.from_pretrained(
                    model, 
                    torch_dtype=self.torch.float16,  # Use half precision for CUDA
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Move model to the specified CUDA device
                endpoint = endpoint.to(cuda_device)
                
                # Set model to evaluation mode
                endpoint.eval()
                
                # Print memory usage info
                if hasattr(self.torch.cuda, "memory_allocated"):
                    allocated_mem = self.torch.cuda.memory_allocated(device_id) / (1024**2)
                    print(f"CUDA memory allocated: {allocated_mem:.2f} MB")
                    
                # Determine reasonable batch size based on available memory
                total_mem = self.torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
                # Set batch size to roughly scale with available VRAM (conservative estimate)
                batch_size = max(1, min(16, int(total_mem * 2)))
                print(f"Using batch size: {batch_size}")
                
                # Set implementation type to REAL
                implementation_type = "REAL"
                
            except Exception as e:
                print(f"Error loading model to CUDA: {e}")
                endpoint = None
                
        except Exception as e:
            print(f"Error in CUDA initialization: {e}")
            import traceback
            traceback.print_exc()
            endpoint = None
            tokenizer = None
            
        # Create the handler function (will use mock if endpoint is None)
        endpoint_handler = self.create_cuda_image_embedding_endpoint_handler(
            tokenizer if tokenizer is not None else processor, 
            endpoint_model=model, 
            cuda_label=cuda_device,  # Use corrected/validated device
            endpoint=endpoint,
            implementation_type=implementation_type
        )
        
        # Clean up memory
        self.torch.cuda.empty_cache()
        
        print(f"CLIP CUDA initialization complete: {implementation_type}")
        return endpoint, tokenizer if tokenizer is not None else processor, endpoint_handler, AnyioQueue(64), batch_size

    def init_openvino(self, model=None, model_type=None, device=None, openvino_label=None, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None):
        """Initialize CLIP model for OpenVINO.
        
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
                    return {"input_ids": np.zeros((1, 77), dtype=np.int64),
                            "attention_mask": np.ones((1, 77), dtype=np.int64),
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
                    import openvino
                    self.ov = openvino
                except ImportError as e:
                    print(f"Error importing OpenVINO: {e}")
                    processor, endpoint = create_dummy_components()
                    endpoint_handler = self.create_openvino_image_embedding_endpoint_handler(
                        endpoint=endpoint,
                        tokenizer=processor,
                        endpoint_model=model,
                        openvino_label=openvino_label
                    )
                    return endpoint, processor, endpoint_handler, AnyioQueue(64), 0
        except Exception as e:
            print(f"Error setting up OpenVINO: {e}")
            processor, endpoint = create_dummy_components()
            endpoint_handler = self.create_openvino_image_embedding_endpoint_handler(
                endpoint=endpoint,
                tokenizer=processor,
                endpoint_model=model,
                openvino_label=openvino_label
            )
            return endpoint, processor, endpoint_handler, AnyioQueue(64), 0
        
        # Create dummy components that we'll use if any part of initialization fails
        dummy_processor, dummy_endpoint = create_dummy_components()
        
        # Flag to track if we're using real implementation
        using_real_implementation = False
        
        try:
            # Safe handling of HuggingFace cache paths
            try:
                # Setup cached model paths
                homedir = os.path.expanduser("~")
                model_name_convert = model.replace("/", "--")
                huggingface_cache = os.path.join(homedir, ".cache/huggingface")
                huggingface_cache_models = os.path.join(huggingface_cache, "hub")
                
                # Check if OpenVINO and optimum are available
                try:
                    from optimum.intel import OVModelForFeatureExtraction
                    optimum_available = True
                except ImportError:
                    optimum_available = False
                    print("Optimum Intel not available, will use direct OpenVINO conversion")
                
                # Create model destination directory if needed
                model_dir = os.path.join(homedir, "openvino_models", model_name_convert)
                os.makedirs(model_dir, exist_ok=True)
                
                # Create CLIP processor
                try:
                    processor = self.transformers.CLIPProcessor.from_pretrained(model, trust_remote_code=True)
                    print(f"Successfully loaded CLIP processor from model: {model}")
                except Exception as e:
                    print(f"Error loading CLIP processor: {e}")
                    try:
                        processor = self.transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
                        print(f"Successfully loaded AutoProcessor from model: {model}")
                    except Exception as e:
                        print(f"Error loading AutoProcessor: {e}")
                        processor = dummy_processor
                
                # Try to load model directly through OpenVINO
                ov_model = None
                
                # Process device information
                if openvino_label and ":" in openvino_label:
                    try:
                        device_parts = openvino_label.split(":")
                        device_type = device_parts[0]
                        device_index = int(device_parts[1]) if len(device_parts) > 1 else 0
                    except Exception as e:
                        print(f"Error parsing device label: {e}")
                        device_type = "CPU"
                        device_index = 0
                else:
                    device_type = "CPU"
                    device_index = 0
                
                # Get the task type for CLIP (feature-extraction)
                task_type = "feature-extraction"
                if get_openvino_pipeline_type:
                    try:
                        task_type = get_openvino_pipeline_type(model, model_type)
                        print(f"Using task type: {task_type} for model {model}")
                    except Exception as e:
                        print(f"Error getting pipeline type: {e}, using default: {task_type}")
                
                # Try multiple approaches to initialize the model
                
                # Approach 1: Try optimum-intel integration first if available
                if optimum_available:
                    try:
                        print(f"Trying to initialize CLIP model with optimum-intel OVModelForFeatureExtraction")
                        ov_model = OVModelForFeatureExtraction.from_pretrained(
                            model, 
                            export=True,
                            compile=False,
                            trust_remote_code=True
                        )
                        # Compile the model for the target device
                        ov_model.compile()
                        print("Successfully loaded CLIP model using optimum-intel")
                        using_real_implementation = True
                    except Exception as e:
                        print(f"Error loading model with optimum-intel: {e}")
                        ov_model = None
                
                # Approach 2: Try direct OpenVINO integration with CLI convert
                if ov_model is None and openvino_cli_convert is not None:
                    try:
                        print(f"Converting CLIP model using OpenVINO CLI")
                        model_dst_path = os.path.join(model_dir, "openvino_model")
                        os.makedirs(model_dst_path, exist_ok=True)
                        
                        # Convert the model using CLI
                        convert_result = openvino_cli_convert(
                            model, 
                            model_dst_path=model_dst_path, 
                            task=task_type, 
                            weight_format="int8", 
                            ratio="1.0", 
                            group_size=128, 
                            sym=True
                        )
                        
                        if convert_result:
                            print(f"Successfully converted model using CLI: {convert_result}")
                            
                            # Try to load the converted model
                            model_xml_path = os.path.join(model_dst_path, f"{model_name_convert}.xml")
                            if os.path.exists(model_xml_path):
                                # Create a Core object and read the model
                                core = self.ov.Core()
                                ov_model = core.read_model(model_xml_path)
                                ov_model = core.compile_model(ov_model, device_type)
                                print(f"Successfully loaded converted CLIP model from {model_xml_path}")
                                using_real_implementation = True
                    except Exception as e:
                        print(f"Error with OpenVINO CLI conversion: {e}")
                        ov_model = None
                
                # Approach 3: Try using get_openvino_model directly
                if ov_model is None and get_openvino_model is not None:
                    try:
                        print(f"Trying to get OpenVINO model directly with get_openvino_model")
                        ov_model = get_openvino_model(model, task_type, openvino_label)
                        if ov_model is not None:
                            print(f"Successfully loaded OpenVINO model directly")
                            using_real_implementation = True
                    except Exception as e:
                        print(f"Error with get_openvino_model: {e}")
                        ov_model = None
                
                # Approach 4: Try using our own conversion if available
                if ov_model is None and hasattr(self, 'openvino_skill_convert'):
                    try:
                        print(f"Trying CLIP model conversion with openvino_skill_convert")
                        model_dst_path = os.path.join(model_dir, "openvino_converted")
                        os.makedirs(model_dst_path, exist_ok=True)
                        ov_model = self.openvino_skill_convert(model, model_dst_path, task_type, "int8")
                        if ov_model is not None:
                            print(f"Successfully converted and loaded CLIP model with openvino_skill_convert")
                            using_real_implementation = True
                    except Exception as e:
                        print(f"Error with openvino_skill_convert: {e}")
                        ov_model = None
                
                # Fall back to dummy implementation if all approaches failed
                if ov_model is None:
                    print("All initialization approaches failed, using dummy implementation")
                    ov_model = dummy_endpoint
                
                # Create the endpoint handler
                endpoint_handler = self.create_openvino_image_embedding_endpoint_handler(
                    endpoint=ov_model,
                    tokenizer=processor,
                    endpoint_model=model,
                    openvino_label=openvino_label,
                    implementation_real=using_real_implementation
                )
                
                # Return the model components
                implementation_type = "REAL" if using_real_implementation else "MOCK"
                print(f"Initialized OpenVINO CLIP model: {implementation_type}")
                return ov_model, processor, endpoint_handler, AnyioQueue(64), 0
                
            except Exception as cache_error:
                print(f"Error in CLIP model setup: {cache_error}")
                processor, endpoint = create_dummy_components()
        except Exception as e:
            print(f"Error in OpenVINO initialization: {e}")
        
        # Create endpoint handler with dummy components as fallback
        endpoint_handler = self.create_openvino_image_embedding_endpoint_handler(
            endpoint=dummy_endpoint,
            tokenizer=dummy_processor,
            endpoint_model=model,
            openvino_label=openvino_label,
            implementation_real=False
        )
        return dummy_endpoint, dummy_processor, endpoint_handler, AnyioQueue(64), 0
    
    def create_cpu_image_embedding_endpoint_handler(self, tokenizer, endpoint_model, cpu_label, endpoint=None):
        """
        Create a handler for CLIP that can process text, images, or both
        
        Args:
            tokenizer: The tokenizer or processor
            endpoint_model: The model name or path
            cpu_label: The label for the CPU endpoint
            endpoint: The model endpoint
            
        Returns:
            A handler function
        """
        def handler(x=None, y=None, tokenizer=tokenizer, endpoint_model=endpoint_model, cpu_label=cpu_label, endpoint=endpoint):
            """
            Process text and/or image inputs with CLIP
            
            Args:
                x: Text input (str or list of str) or image if y is None
                y: Image input (str path, PIL Image, or list of either)
                
            Returns:
                Dict containing embeddings and/or similarity scores
            """
            # Ensure model is in eval mode
            if hasattr(endpoint, 'eval'):
                endpoint.eval()
            
            # Mark if we're using a mock
            using_mock = False
            
            try:
                result = {}
                
                # Check what kind of inputs we have
                text_input = None
                image_input = None
                
                # If only x is provided, determine if it's text or image
                if x is not None and y is None:
                    if isinstance(x, str) and (x.startswith('http') or os.path.exists(x)):
                        # x is an image path
                        image_input = x
                    elif isinstance(x, Image.Image):
                        # x is a PIL image
                        image_input = x
                    else:
                        # Assume x is text
                        text_input = x
                else:
                    # Both x and y are provided or both are None
                    text_input = x
                    image_input = y
                
                # Process image if provided
                if image_input is not None:
                    try:
                        # Load and process image(s)
                        if isinstance(image_input, str):
                            # Single image path
                            image = load_image(image_input)
                            image_inputs = tokenizer(images=[image], return_tensors='pt', padding=True)
                        elif isinstance(image_input, Image.Image):
                            # Single PIL image
                            image_inputs = tokenizer(images=[image_input], return_tensors='pt', padding=True)
                        elif isinstance(image_input, list):
                            # List of images
                            images = [
                                img if isinstance(img, Image.Image) else load_image(img)
                                for img in image_input
                            ]
                            image_inputs = tokenizer(images=images, return_tensors='pt', padding=True)
                        else:
                            raise ValueError(f"Unsupported image input type: {type(image_input)}")
                        
                        # Get image embeddings
                        with self.torch.no_grad():
                            if endpoint is not None and (hasattr(endpoint, 'get_image_features') or hasattr(endpoint, '__call__')):
                                try:
                                    if hasattr(endpoint, 'get_image_features'):
                                        image_features = endpoint.get_image_features(**image_inputs)
                                    else:
                                        # For processors that handle both image and text
                                        outputs = endpoint(**image_inputs)
                                        image_features = outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs
                                    
                                    result["image_embedding"] = image_features
                                except Exception as e:
                                    print(f"Error getting image features: {e}")
                                    # Mark as mock if real inference fails
                                    using_mock = True
                                    batch_size = 1 if not isinstance(image_input, list) else len(image_input)
                                    result["image_embedding"] = self.torch.rand((batch_size, 512))
                            else:
                                # No valid endpoint - use mock
                                using_mock = True
                                batch_size = 1 if not isinstance(image_input, list) else len(image_input)
                                result["image_embedding"] = self.torch.rand((batch_size, 512))
                    except Exception as e:
                        print(f"Error processing image input: {e}")
                        # Create fallback image embedding
                        using_mock = True
                        batch_size = 1 if not isinstance(image_input, list) else len(image_input)
                        result["image_embedding"] = self.torch.rand((batch_size, 512))
                
                # Process text if provided
                if text_input is not None:
                    try:
                        # Process text input(s)
                        if isinstance(text_input, str):
                            # Single text
                            text_inputs = tokenizer(text=[text_input], return_tensors='pt', padding=True)
                        elif isinstance(text_input, list):
                            # List of texts
                            text_inputs = tokenizer(text=text_input, return_tensors='pt', padding=True)
                        else:
                            raise ValueError(f"Unsupported text input type: {type(text_input)}")
                        
                        # Get text embeddings
                        with self.torch.no_grad():
                            if endpoint is not None and (hasattr(endpoint, 'get_text_features') or hasattr(endpoint, '__call__')):
                                try:
                                    if hasattr(endpoint, 'get_text_features'):
                                        text_features = endpoint.get_text_features(**text_inputs)
                                    else:
                                        # For processors that handle both image and text
                                        outputs = endpoint(**text_inputs)
                                        text_features = outputs.text_embeds if hasattr(outputs, 'text_embeds') else outputs
                                    
                                    result["text_embedding"] = text_features
                                except Exception as e:
                                    print(f"Error getting text features: {e}")
                                    # Mark as mock if real inference fails
                                    using_mock = True
                                    batch_size = 1 if not isinstance(text_input, list) else len(text_input)
                                    result["text_embedding"] = self.torch.rand((batch_size, 512))
                            else:
                                # No valid endpoint - use mock
                                using_mock = True
                                batch_size = 1 if not isinstance(text_input, list) else len(text_input)
                                result["text_embedding"] = self.torch.rand((batch_size, 512))
                    except Exception as e:
                        print(f"Error processing text input: {e}")
                        # Create fallback text embedding
                        using_mock = True
                        batch_size = 1 if not isinstance(text_input, list) else len(text_input)
                        result["text_embedding"] = self.torch.rand((batch_size, 512))
                
                # Calculate similarity if we have both embeddings
                if "image_embedding" in result and "text_embedding" in result:
                    try:
                        # Normalize embeddings
                        image_norm = result["image_embedding"] / result["image_embedding"].norm(dim=-1, keepdim=True)
                        text_norm = result["text_embedding"] / result["text_embedding"].norm(dim=-1, keepdim=True)
                        
                        # Calculate cosine similarity
                        similarity = (text_norm @ image_norm.T)
                        result["similarity"] = similarity
                    except Exception as e:
                        print(f"Error calculating similarity: {e}")
                        # Create a mock similarity
                        using_mock = True
                        result["similarity"] = self.torch.tensor([[0.5]])
                
                # No valid inputs
                if not result:
                    return {"message": "No valid input provided"}
                
                # Add MOCK/REAL indicator to results - create a copy of keys first to avoid dictionary size change during iteration
                keys_to_process = list(result.keys())
                for key in keys_to_process:
                    if isinstance(result[key], (self.torch.Tensor, list, tuple)) and key != "similarity":
                        result[key + "_status"] = "MOCK" if using_mock else "REAL"
                
                # Return single embedding if that's all that was requested
                if len(result) == 1 and (
                    "image_embedding" in result or 
                    "text_embedding" in result
                ):
                    embedding_key = list(result.keys())[0]
                    return {
                        embedding_key: result[embedding_key],
                        embedding_key + "_status": "MOCK" if using_mock else "REAL"
                    }
                
                return result
                
            except Exception as e:
                print(f"Error in CPU CLIP handler: {e}")
                return {
                    "error": str(e),
                    "status": "MOCK"
                }
                
        return handler
    
    def create_qualcomm_image_embedding_endpoint_handler(self, tokenizer, processor, endpoint_model, qualcomm_label, endpoint=None):
        """
        Create a handler for CLIP image/text embeddings and similarity using Qualcomm hardware.
        
        Args:
            tokenizer: Tokenizer for processing text
            processor: Processor for processing images
            endpoint_model: Model name or path
            qualcomm_label: Label for Qualcomm endpoint
            endpoint: The model endpoint (or None to use mock)
            
        Returns:
            Handler function for CLIP inference on Qualcomm hardware
        """
        def handler(x=None, y=None, tokenizer=tokenizer, processor=processor, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint):
            """
            Process text and/or image inputs with CLIP on Qualcomm hardware.
            
            Args:
                x: Text input (str or list of str)
                y: Image input (str path, PIL Image, or list of either)
                
            Returns:
                Dict containing embeddings and/or similarity scores
            """
            # Track whether we're using mock functionality
            using_mock = False
            
            try:
                inputs = {}
                
                # Process text input (if provided)
                if x is not None:
                    try:
                        if isinstance(x, str):
                            text_inputs = tokenizer(text=[x], return_tensors='np')
                        elif isinstance(x, list):
                            text_inputs = tokenizer(text=x, return_tensors='np')
                        else:
                            raise ValueError(f"Unsupported text input type: {type(x)}")
                        
                        # Add to combined inputs
                        if "input_ids" in text_inputs and "attention_mask" in text_inputs:
                            inputs["input_ids"] = text_inputs["input_ids"]
                            inputs["attention_mask"] = text_inputs["attention_mask"]
                    except Exception as e:
                        print(f"Error processing text input: {e}")
                        using_mock = True
                        # Create dummy text inputs
                        batch_size = 1 if not isinstance(x, list) else len(x)
                        inputs["input_ids"] = self.np.zeros((batch_size, 77), dtype=self.np.int32)
                        inputs["attention_mask"] = self.np.ones((batch_size, 77), dtype=self.np.int32)
                
                # Process image input (if provided)
                if y is not None:
                    try:
                        if isinstance(y, str):
                            # Load single image
                            image = load_image(y)
                            # Convert to proper format for CLIP
                            if hasattr(processor, "image_processor"):
                                image_inputs = processor.image_processor(images=[image], return_tensors='np')
                            else:
                                # Fallback to basic processing
                                image = image.resize((224, 224))  # Standard size for most vision models
                                img_array = self.np.array(image)
                                img_array = img_array.transpose(2, 0, 1)  # Convert to CHW format
                                img_array = img_array / 255.0  # Normalize
                                image_inputs = {"pixel_values": self.np.expand_dims(img_array, axis=0)}
                                
                            inputs["pixel_values"] = image_inputs["pixel_values"]
                            
                        elif isinstance(y, list):
                            # Process multiple images
                            images = [img if isinstance(img, Image.Image) else load_image(img) for img in y]
                            if hasattr(processor, "image_processor"):
                                image_inputs = processor.image_processor(images=images, return_tensors='np')
                            else:
                                # Fallback processing for multiple images
                                processed_images = []
                                for img in images:
                                    img = img.resize((224, 224))
                                    img_array = self.np.array(img)
                                    img_array = img_array.transpose(2, 0, 1)
                                    img_array = img_array / 255.0
                                    processed_images.append(img_array)
                                image_inputs = {"pixel_values": self.np.stack(processed_images)}
                                
                            inputs["pixel_values"] = image_inputs["pixel_values"]
                            
                        elif isinstance(y, Image.Image):
                            # Process a PIL Image directly
                            if hasattr(processor, "image_processor"):
                                image_inputs = processor.image_processor(images=[y], return_tensors='np')
                            else:
                                # Basic processing
                                img_resized = y.resize((224, 224))
                                img_array = self.np.array(img_resized)
                                img_array = img_array.transpose(2, 0, 1)
                                img_array = img_array / 255.0
                                image_inputs = {"pixel_values": self.np.expand_dims(img_array, axis=0)}
                                
                            inputs["pixel_values"] = image_inputs["pixel_values"]
                            
                        else:
                            raise ValueError(f"Unsupported image input type: {type(y)}")
                            
                    except Exception as e:
                        print(f"Error processing image input: {e}")
                        using_mock = True
                        # Create dummy image inputs
                        batch_size = 1
                        if isinstance(y, list):
                            batch_size = len(y)
                        inputs["pixel_values"] = self.np.zeros((batch_size, 3, 224, 224), dtype=self.np.float32)
                
                # Run inference with SNPE if available
                outputs = {}
                try:
                    if endpoint is not None and self.snpe_utils is not None:
                        outputs = self.snpe_utils.run_inference(endpoint, inputs)
                    else:
                        using_mock = True
                        # Create mock outputs
                        batch_size = 1
                        if x is not None and isinstance(x, list):
                            batch_size = len(x)
                        elif y is not None and isinstance(y, list):
                            batch_size = len(y)
                            
                        outputs = {
                            "text_embeds": self.np.random.randn(batch_size, 512).astype(self.np.float32),
                            "image_embeds": self.np.random.randn(batch_size, 512).astype(self.np.float32)
                        }
                except Exception as e:
                    print(f"Error in SNPE inference: {e}")
                    using_mock = True
                    # Create mock outputs
                    batch_size = 1
                    if x is not None and isinstance(x, list):
                        batch_size = len(x)
                    elif y is not None and isinstance(y, list):
                        batch_size = len(y)
                        
                    outputs = {
                        "text_embeds": self.np.random.randn(batch_size, 512).astype(self.np.float32),
                        "image_embeds": self.np.random.randn(batch_size, 512).astype(self.np.float32)
                    }
                
                # Process results based on what inputs were provided
                result = {}
                
                # Add text embeddings if text was provided
                if x is not None and "text_embeds" in outputs:
                    text_embeddings = self.torch.tensor(outputs["text_embeds"]) if self.torch is not None else outputs["text_embeds"]
                    result["text_embedding"] = text_embeddings
                    result["text_embedding_status"] = "MOCK" if using_mock else "REAL"
                    
                # Add image embeddings if image was provided
                if y is not None and "image_embeds" in outputs:
                    image_embeddings = self.torch.tensor(outputs["image_embeds"]) if self.torch is not None else outputs["image_embeds"]
                    result["image_embedding"] = image_embeddings
                    result["image_embedding_status"] = "MOCK" if using_mock else "REAL"
                
                # Calculate similarity if we have both text and image
                if x is not None and y is not None and "text_embeds" in outputs and "image_embeds" in outputs:
                    if self.torch is not None:
                        try:
                            # Convert to PyTorch tensors if needed
                            text_embeds = self.torch.tensor(outputs["text_embeds"]) if not isinstance(outputs["text_embeds"], self.torch.Tensor) else outputs["text_embeds"]
                            image_embeds = self.torch.tensor(outputs["image_embeds"]) if not isinstance(outputs["image_embeds"], self.torch.Tensor) else outputs["image_embeds"]
                            
                            # Normalize embeddings
                            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                            
                            # Calculate similarity
                            similarity = self.torch.matmul(text_embeds, image_embeds.T)
                            result["similarity"] = similarity
                        except Exception as e:
                            print(f"Error calculating similarity: {e}")
                            # Create mock similarity
                            if self.torch is not None:
                                result["similarity"] = self.torch.tensor([[0.5]])
                            else:
                                result["similarity"] = 0.5
                    else:
                        # Create mock similarity without torch
                        result["similarity"] = 0.5
                
                # Add overall status
                result["implementation_status"] = "MOCK" if using_mock else "REAL"
                
                # Check for empty result
                if len(result) == 0 or (len(result) == 1 and "implementation_status" in result):
                    return {"message": "No valid embeddings generated", "implementation_status": "MOCK"}
                
                return result
                
            except Exception as e:
                print(f"Error in Qualcomm CLIP endpoint handler: {e}")
                return {
                    "error": str(e),
                    "implementation_status": "MOCK" 
                }
                
        return handler
        
    def create_apple_image_embedding_endpoint_handler(self, endpoint, processor, endpoint_model, apple_label):
        """Creates an Apple Silicon optimized handler for CLIP image/text embedding models."""
        def handler(x, y=None, endpoint=endpoint, processor=processor, endpoint_model=endpoint_model, apple_label=apple_label):
            try:
                inputs = {}
                
                # Handle text input
                if x is not None:
                    if type(x) == str:
                        text_inputs = processor(
                            text=x,
                            return_tensors='np',
                            padding=True
                        )
                    elif type(x) == list:
                        text_inputs = processor(text=[text for text in x], return_tensors='np', padding=True)
                    
                    for key, value in text_inputs.items():
                        inputs[key] = value
                
                # Handle image input
                if y is not None:
                    if type(y) == str:
                        image = load_image(y)
                        image_inputs = processor(
                            images=[image], 
                            return_tensors='np', 
                            padding=True
                        )
                    elif type(y) == list:
                        images = [load_image(image_file) for image_file in y]
                        image_inputs = processor(
                            images=images,
                            return_tensors='np',
                            padding=True
                        )
                    
                    # Add image inputs
                    for key, value in image_inputs.items():
                        if key.startswith('pixel_values'):
                            inputs[key] = value
                
                # Run inference with CoreML
                results = self.coreml_utils.run_inference(endpoint, inputs)
                
                # Process results
                output = {}
                
                # Extract text embeddings
                if x is not None and "text_embeds" in results:
                    text_embeddings = self.torch.tensor(results["text_embeds"])
                    output["text_embedding"] = text_embeddings
                
                # Extract image embeddings
                if y is not None and "image_embeds" in results:
                    image_embeddings = self.torch.tensor(results["image_embeds"])
                    output["image_embedding"] = image_embeddings
                
                # If we have both text and image, compute similarity
                if x is not None and y is not None and "text_embeds" in results and "image_embeds" in results:
                    text_emb = self.torch.tensor(results["text_embeds"])
                    image_emb = self.torch.tensor(results["image_embeds"])
                    
                    # Normalize embeddings
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity
                    similarity = self.torch.matmul(text_emb, image_emb.T)
                    output["similarity"] = similarity
                
                # Return single embedding if that's all we have
                if len(output) == 1 and list(output.keys())[0] in ["text_embedding", "image_embedding"]:
                    return {"embedding": list(output.values())[0]}
                    
                return output if output else None
                
            except Exception as e:
                print(f"Error in Apple Silicon image embedding handler: {e}")
                return None
                
        return handler
    
    def create_cuda_image_embedding_endpoint_handler(self, tokenizer, endpoint_model, cuda_label, endpoint=None, implementation_type="MOCK"):
        """
        Create an enhanced handler for CLIP that processes text and/or images using CUDA acceleration
        with optimized memory management and detailed performance metrics.
        
        Args:
            tokenizer: The tokenizer or processor for the model
            endpoint_model: The model name or path
            cuda_label: The CUDA device label (e.g., 'cuda:0')
            endpoint: The model endpoint (optional)
            implementation_type: Flag indicating if this is a real or mock implementation
            
        Returns:
            A handler function for CUDA-accelerated CLIP with comprehensive performance metrics
        """
        # Import necessary modules
        import traceback
        
        # Validate and create device object
        cuda_device = None
        device_id = 0
        try:
            if ":" in cuda_label:
                device_parts = cuda_label.split(":")
                device_id = int(device_parts[1]) if len(device_parts) > 1 else 0
                
                # Validate device index against available devices
                if self.torch.cuda.is_available() and device_id < self.torch.cuda.device_count():
                    cuda_device = self.torch.device(cuda_label)
                else:
                    # Use default device if specified one is invalid
                    print(f"Warning: CUDA device {device_id} not available, using device 0")
                    device_id = 0
                    cuda_device = self.torch.device("cuda:0")
            else:
                cuda_device = self.torch.device("cuda:0")
                
        except Exception as e:
            print(f"Error creating CUDA device object: {e}")
            cuda_device = None
            
        def handler(x=None, y=None, tokenizer=tokenizer, endpoint_model=endpoint_model, 
                    cuda_label=cuda_label, endpoint=endpoint, batch_size=None):
            """
            Process text and/or image inputs with CLIP using CUDA with optimized memory management
            and detailed performance metrics.
            
            Args:
                x: Text input (str or list of str) or image if y is None
                y: Image input (str path, PIL Image, or list of either)
                batch_size: Optional custom batch size for processing (default: auto-determine)
                
            Returns:
                Dict containing embeddings and/or similarity scores with comprehensive performance metrics
            """
            # Start performance tracking
            start_time = time.time()
            
            # Track if we're using mocks
            using_mock = (implementation_type != "REAL" or endpoint is None or 
                         not self.torch.cuda.is_available() or cuda_device is None)

            # Initialize performance trackers
            preprocessing_time = None
            embedding_time = None
            similarity_time = None
            gpu_memory_before = None
            gpu_memory_after = None
            gpu_memory_used = None
            
            # Initialize result dict
            result = {
                "implementation_type": "MOCK" if using_mock else "REAL",
                "device": str(cuda_device) if cuda_device else cuda_label,
                "model_name": endpoint_model
            }
            
            # Return mock early if we know we're using mocks
            if using_mock:
                # Create a basic mock response
                time.sleep(0.1)  # Small delay to simulate processing
                if x is not None:
                    batch_size_text = 1 if not isinstance(x, list) else len(x)
                    result["text_embedding"] = self.torch.rand((batch_size_text, 512))
                if y is not None:
                    batch_size_img = 1 if not isinstance(y, list) else len(y)
                    result["image_embedding"] = self.torch.rand((batch_size_img, 512))
                if x is not None and y is not None:
                    result["similarity"] = self.torch.tensor([[0.75]])
                
                # Add timing information
                result["total_time"] = time.time() - start_time
                return result
            
            # Set eval mode if available
            if endpoint is not None and hasattr(endpoint, "eval"):
                endpoint.eval()
            
            # Run inference within torch.no_grad() context for efficiency
            with self.torch.no_grad():
                try:
                    # Clear GPU memory before starting
                    if hasattr(self.torch.cuda, "empty_cache"):
                        self.torch.cuda.empty_cache()
                    
                    # Get initial memory usage for tracking
                    if hasattr(self.torch.cuda, 'mem_get_info'):
                        try:
                            free_memory_start, total_memory = self.torch.cuda.mem_get_info(device_id)
                            gpu_memory_before = {
                                "free_gb": free_memory_start / (1024**3),
                                "total_gb": total_memory / (1024**3),
                                "used_gb": (total_memory - free_memory_start) / (1024**3)
                            }
                        except Exception as mem_error:
                            print(f"Error getting initial GPU memory info: {mem_error}")
                    
                    # Start preprocessing timer
                    preprocessing_start = time.time()
                    
                    # Check what kind of inputs we have
                    text_input = None
                    image_input = None
                    
                    # If only x is provided, determine if it's text or image
                    if x is not None and y is None:
                        if isinstance(x, str) and (x.startswith('http') or os.path.exists(x)):
                            # x is an image path
                            image_input = x
                        elif isinstance(x, Image.Image):
                            # x is a PIL image
                            image_input = x
                        else:
                            # Assume x is text
                            text_input = x
                    else:
                        # Both x and y are provided or both are None
                        text_input = x
                        image_input = y
                    
                    # Process image if provided
                    image_embeddings = None
                    if image_input is not None:
                        try:
                            # Load and process image(s)
                            if isinstance(image_input, str):
                                # Single image path
                                image = load_image(image_input)
                                image_inputs = tokenizer(images=[image], return_tensors='pt', padding=True)
                            elif isinstance(image_input, Image.Image):
                                # Single PIL image
                                image_inputs = tokenizer(images=[image_input], return_tensors='pt', padding=True)
                            elif isinstance(image_input, list):
                                # List of images
                                images = [
                                    img if isinstance(img, Image.Image) else load_image(img)
                                    for img in image_input
                                ]
                                image_inputs = tokenizer(images=images, return_tensors='pt', padding=True)
                            else:
                                raise ValueError(f"Unsupported image input type: {type(image_input)}")
                            
                            # Move inputs to CUDA
                            cuda_inputs = {}
                            for k, v in image_inputs.items():
                                if hasattr(v, 'to') and callable(v.to):
                                    cuda_inputs[k] = v.to(cuda_device)
                                else:
                                    cuda_inputs[k] = v
                            
                            # Record preprocessing time
                            preprocessing_time = time.time() - preprocessing_start
                            
                            # Start embedding timer
                            embedding_start = time.time()
                            
                            # Handle batch processing for multiple images
                            is_batch = isinstance(image_input, list) and len(image_input) > 1
                            
                            # Determine optimal batch size if needed
                            if is_batch and batch_size is None:
                                # Calculate based on available memory if possible
                                if gpu_memory_before is not None:
                                    # Rough heuristic: 1 GB can handle ~16 images at 224x224
                                    available_gb = gpu_memory_before["free_gb"]
                                    batch_size = max(1, min(32, int(available_gb * 16)))
                                else:
                                    # Default batch size if memory info not available
                                    batch_size = 8
                                print(f"Using auto-determined batch size for images: {batch_size}")
                            
                            # Get image features
                            if endpoint is not None and hasattr(endpoint, 'get_image_features'):
                                if is_batch and batch_size is not None:
                                    # Process in batches to avoid OOM errors
                                    all_embeddings = []
                                    batch_count = 0
                                    for i in range(0, len(image_input), batch_size):
                                        batch_count += 1
                                        batch_end = min(i + batch_size, len(image_input))
                                        # Extract batch inputs
                                        batch_inputs = {
                                            k: v[i:batch_end] if hasattr(v, '__getitem__') else v
                                            for k, v in cuda_inputs.items()
                                        }
                                        
                                        # Process batch
                                        batch_features = endpoint.get_image_features(**batch_inputs)
                                        # Move to CPU and detach immediately to save CUDA memory
                                        all_embeddings.append(batch_features.detach().cpu())
                                        
                                        # Clean cache between batches
                                        if hasattr(self.torch.cuda, "empty_cache"):
                                            self.torch.cuda.empty_cache()
                                    
                                    # Combine all batches
                                    image_embeddings = self.torch.cat(all_embeddings, dim=0)
                                    result["image_embedding"] = image_embeddings
                                    result["image_batch_count"] = batch_count
                                else:
                                    # Single image or small batch processing
                                    image_features = endpoint.get_image_features(**cuda_inputs)
                                    # Move back to CPU and detach from graph
                                    image_embeddings = image_features.detach().cpu()
                                    result["image_embedding"] = image_embeddings
                            else:
                                # Fall back to mock
                                using_mock = True
                                batch_size_img = 1 if not isinstance(image_input, list) else len(image_input)
                                result["image_embedding"] = self.torch.rand((batch_size_img, 512))
                            
                            # Record embedding time
                            embedding_image_time = time.time() - embedding_start
                            result["image_embedding_time"] = embedding_image_time
                            
                        except Exception as e:
                            print(f"Error processing image input on CUDA: {e}")
                            print(f"Traceback: {traceback.format_exc()}")
                            
                            # Create fallback image embedding
                            using_mock = True
                            batch_size_img = 1 if not isinstance(image_input, list) else len(image_input)
                            result["image_embedding"] = self.torch.rand((batch_size_img, 512))
                    
                    # Process text if provided
                    text_embeddings = None
                    if text_input is not None:
                        try:
                            # Process text input(s)
                            if isinstance(text_input, str):
                                # Single text
                                text_inputs = tokenizer(text=[text_input], return_tensors='pt', padding=True)
                            elif isinstance(text_input, list):
                                # List of texts
                                text_inputs = tokenizer(text=text_input, return_tensors='pt', padding=True)
                            else:
                                raise ValueError(f"Unsupported text input type: {type(text_input)}")
                            
                            # Move to CUDA
                            cuda_inputs = {}
                            for k, v in text_inputs.items():
                                if hasattr(v, 'to') and callable(v.to):
                                    cuda_inputs[k] = v.to(cuda_device)
                                else:
                                    cuda_inputs[k] = v
                            
                            # Record preprocessing time if not already set
                            if preprocessing_time is None:
                                preprocessing_time = time.time() - preprocessing_start
                            
                            # Start embedding timer
                            embedding_start = time.time()
                            
                            # Handle batch processing for multiple texts
                            is_batch = isinstance(text_input, list) and len(text_input) > 1
                            
                            # Determine optimal batch size if needed
                            if is_batch and batch_size is None:
                                # Calculate based on available memory if possible
                                if gpu_memory_before is not None:
                                    # Text uses less memory than images
                                    available_gb = gpu_memory_before["free_gb"]
                                    batch_size = max(1, min(64, int(available_gb * 32)))
                                else:
                                    # Default batch size if memory info not available
                                    batch_size = 16
                                print(f"Using auto-determined batch size for texts: {batch_size}")
                            
                            # Get text features
                            if endpoint is not None and hasattr(endpoint, 'get_text_features'):
                                if is_batch and batch_size is not None:
                                    # Process in batches to avoid OOM errors
                                    all_embeddings = []
                                    batch_count = 0
                                    for i in range(0, len(text_input), batch_size):
                                        batch_count += 1
                                        batch_end = min(i + batch_size, len(text_input))
                                        # Extract batch inputs
                                        batch_inputs = {
                                            k: v[i:batch_end] if hasattr(v, '__getitem__') else v
                                            for k, v in cuda_inputs.items()
                                        }
                                        
                                        # Process batch
                                        batch_features = endpoint.get_text_features(**batch_inputs)
                                        # Move to CPU and detach immediately to save CUDA memory
                                        all_embeddings.append(batch_features.detach().cpu())
                                        
                                        # Clean cache between batches
                                        if hasattr(self.torch.cuda, "empty_cache"):
                                            self.torch.cuda.empty_cache()
                                    
                                    # Combine all batches
                                    text_embeddings = self.torch.cat(all_embeddings, dim=0)
                                    result["text_embedding"] = text_embeddings
                                    result["text_batch_count"] = batch_count
                                else:
                                    # Single text or small batch processing
                                    text_features = endpoint.get_text_features(**cuda_inputs)
                                    # Move back to CPU and detach from graph
                                    text_embeddings = text_features.detach().cpu()
                                    result["text_embedding"] = text_embeddings
                            else:
                                # Fall back to mock
                                using_mock = True
                                batch_size_text = 1 if not isinstance(text_input, list) else len(text_input)
                                result["text_embedding"] = self.torch.rand((batch_size_text, 512))
                            
                            # Record embedding time
                            embedding_text_time = time.time() - embedding_start
                            result["text_embedding_time"] = embedding_text_time
                            
                        except Exception as e:
                            print(f"Error processing text input on CUDA: {e}")
                            print(f"Traceback: {traceback.format_exc()}")
                            
                            # Create fallback text embedding
                            using_mock = True
                            batch_size_text = 1 if not isinstance(text_input, list) else len(text_input)
                            result["text_embedding"] = self.torch.rand((batch_size_text, 512))
                    
                    # Calculate similarity if we have both embeddings 
                    if "image_embedding" in result and "text_embedding" in result:
                        try:
                            # Start similarity timer
                            similarity_start = time.time()
                            
                            # Get embeddings from result
                            image_emb = result["image_embedding"]
                            text_emb = result["text_embedding"]
                            
                            # Calculate on CPU to avoid transferring back to CUDA
                            image_norm = image_emb / image_emb.norm(dim=-1, keepdim=True)
                            text_norm = text_emb / text_emb.norm(dim=-1, keepdim=True)
                            
                            # Calculate cosine similarity with better batch handling
                            if image_norm.dim() == 2 and text_norm.dim() == 2:
                                # Standard similarity calculation
                                similarity = (text_norm @ image_norm.T)
                            elif image_norm.dim() == 1 and text_norm.dim() == 1:
                                # Single examples
                                similarity = (text_norm @ image_norm).unsqueeze(0).unsqueeze(0)
                            elif image_norm.dim() == 1 and text_norm.dim() == 2:
                                # Multiple texts, single image
                                image_norm = image_norm.unsqueeze(0)
                                similarity = (text_norm @ image_norm.T)
                            elif image_norm.dim() == 2 and text_norm.dim() == 1:
                                # Single text, multiple images
                                text_norm = text_norm.unsqueeze(0)
                                similarity = (text_norm @ image_norm.T)
                            else:
                                # Unexpected case, fallback
                                similarity = self.torch.tensor([[0.75]])
                                using_mock = True
                            
                            result["similarity"] = similarity
                            
                            # Record similarity time
                            similarity_time = time.time() - similarity_start
                            result["similarity_time"] = similarity_time
                            
                        except Exception as e:
                            print(f"Error calculating similarity: {e}")
                            print(f"Traceback: {traceback.format_exc()}")
                            
                            # Create a mock similarity
                            using_mock = True
                            result["similarity"] = self.torch.tensor([[0.75]])
                    
                    # Get GPU memory usage after processing
                    if hasattr(self.torch.cuda, 'mem_get_info'):
                        try:
                            free_memory_after, total_memory = self.torch.cuda.mem_get_info(device_id)
                            gpu_memory_after = {
                                "free_gb": free_memory_after / (1024**3),
                                "total_gb": total_memory / (1024**3),
                                "used_gb": (total_memory - free_memory_after) / (1024**3)
                            }
                            
                            # Calculate memory used for this operation
                            if gpu_memory_before is not None:
                                gpu_memory_used = (free_memory_start - free_memory_after) / (1024**3)  # in GB
                        except Exception as mem_error:
                            print(f"Error getting final GPU memory info: {mem_error}")
                    
                    # No valid inputs
                    if not any(k in result for k in ["image_embedding", "text_embedding", "similarity"]):
                        result["message"] = "No valid input provided"
                    
                    # Update implementation type in result
                    result["implementation_type"] = "MOCK" if using_mock else "REAL"
                    
                    # Add all performance information
                    total_time = time.time() - start_time
                    result["total_time"] = total_time
                    if preprocessing_time is not None:
                        result["preprocessing_time"] = preprocessing_time
                    if embedding_time is not None:
                        result["embedding_time"] = embedding_time
                    
                    # Add detailed memory information
                    if gpu_memory_before is not None:
                        result["gpu_memory_before"] = gpu_memory_before
                    if gpu_memory_after is not None:
                        result["gpu_memory_after"] = gpu_memory_after
                    if gpu_memory_used is not None:
                        result["gpu_memory_used_gb"] = gpu_memory_used
                    
                    # Add embedding dimensions for reference
                    if "image_embedding" in result and isinstance(result["image_embedding"], self.torch.Tensor):
                        result["image_embedding_shape"] = list(result["image_embedding"].shape)
                    if "text_embedding" in result and isinstance(result["text_embedding"], self.torch.Tensor):
                        result["text_embedding_shape"] = list(result["text_embedding"].shape)
                    
                    # Clean up CUDA memory before returning
                    if hasattr(self.torch.cuda, "empty_cache"):
                        self.torch.cuda.empty_cache()
                    
                    return result
                
                except Exception as e:
                    # Clean up memory on error
                    if hasattr(self.torch.cuda, "empty_cache"):
                        self.torch.cuda.empty_cache()
                        
                    print(f"Error in CUDA CLIP handler: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    
                    # Return error result with full details
                    error_result = {
                        "error": str(e),
                        "implementation_type": "MOCK",
                        "device": str(cuda_device) if cuda_device else cuda_label,
                        "total_time": time.time() - start_time
                    }
                    
                    # Add preprocessing time if available
                    if preprocessing_time is not None:
                        error_result["preprocessing_time"] = preprocessing_time
                    
                    return error_result
        
        return handler

    def create_openvino_image_embedding_endpoint_handler(self, endpoint=None, tokenizer=None, endpoint_model=None, openvino_label=None, implementation_real=False):
        """Creates an OpenVINO handler for CLIP image and text embedding extraction.
        
        Args:
            endpoint: The OpenVINO model endpoint
            tokenizer: The text/image processor
            endpoint_model: The model name or path
            openvino_label: Label to identify this endpoint
            implementation_real: Flag indicating if this is a real implementation
            
        Returns:
            A handler function for OpenVINO CLIP endpoint
        """
        def handler(x=None, y=None, tokenizer=tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=endpoint):
            """OpenVINO handler for image embedding and text-image similarity.
            
            Args:
                x: Text input (str or list of str) or image if y is None
                y: Image input (str path, PIL Image, or list of either)
                
            Returns:
                Dictionary with embeddings and/or similarity scores
            """
            # Mark if we're using a mock - initialize with the passed implementation flag
            using_mock = not implementation_real
            
            try:
                result = {}
                
                # Check what kind of inputs we have
                text_input = None
                image_input = None
                
                # If only x is provided, determine if it's text or image
                if x is not None and y is None:
                    if isinstance(x, str) and (x.startswith('http') or os.path.exists(x)):
                        # x is an image path
                        image_input = x
                    elif isinstance(x, Image.Image):
                        # x is a PIL image
                        image_input = x
                    else:
                        # Assume x is text
                        text_input = x
                else:
                    # Both x and y are provided or both are None
                    text_input = x
                    image_input = y
                
                # Process image if provided
                if image_input is not None:
                    try:
                        # Load and process image(s)
                        if isinstance(image_input, str):
                            # Single image path
                            image = load_image(image_input)
                            image_inputs = tokenizer(images=[image], return_tensors='pt', padding=True)
                        elif isinstance(image_input, Image.Image):
                            # Single PIL image
                            image_inputs = tokenizer(images=[image_input], return_tensors='pt', padding=True)
                        elif isinstance(image_input, list):
                            # List of images
                            images = [
                                img if isinstance(img, Image.Image) else load_image(img)
                                for img in image_input
                            ]
                            image_inputs = tokenizer(images=images, return_tensors='pt', padding=True)
                        else:
                            raise ValueError(f"Unsupported image input type: {type(image_input)}")
                        
                        # Check if we have real endpoint and it's callable
                        if endpoint is not None and callable(endpoint) and not using_mock:
                            # Run inference with OpenVINO
                            try:
                                # Convert to dictionary first if not already
                                image_inputs_dict = dict(image_inputs) if not isinstance(image_inputs, dict) else image_inputs
                                
                                # For OpenVINO optimum models
                                if hasattr(endpoint, 'forward') and callable(endpoint.forward):
                                    image_features = endpoint.forward(**image_inputs_dict)
                                else:
                                    # For direct OpenVINO models
                                    image_features = endpoint(image_inputs_dict)
                                
                                # Extract image embedding from output based on model output format
                                if hasattr(image_features, 'image_embeds'):
                                    # For optimum models that return an object
                                    image_embeddings = self.torch.tensor(image_features.image_embeds)
                                    result["image_embedding"] = image_embeddings
                                elif isinstance(image_features, dict) and "image_embeds" in image_features:
                                    # For models that return a dict with image_embeds
                                    image_embeddings = self.torch.tensor(image_features["image_embeds"])
                                    result["image_embedding"] = image_embeddings
                                elif isinstance(image_features, dict) and "last_hidden_state" in image_features:
                                    # For BERT-like outputs
                                    image_embeddings = self.torch.tensor(image_features["last_hidden_state"])
                                    # Pool if needed
                                    if image_embeddings.dim() > 2:
                                        image_embeddings = image_embeddings.mean(dim=1)
                                    result["image_embedding"] = image_embeddings
                                elif isinstance(image_features, dict) and len(image_features) > 0:
                                    # Try to get from first output if named keys aren't available
                                    output_values = list(image_features.values())
                                    if len(output_values) > 0:
                                        image_embeddings = self.torch.tensor(output_values[0])
                                        result["image_embedding"] = image_embeddings
                                    else:
                                        # Create a fallback embedding
                                        using_mock = True
                                        batch_size = 1 if not isinstance(image_input, list) else len(image_input)
                                        result["image_embedding"] = self.torch.rand((batch_size, 512))
                                elif isinstance(image_features, (list, tuple)) and len(image_features) > 0:
                                    # For list-like outputs, use the first element
                                    image_embeddings = self.torch.tensor(image_features[0])
                                    result["image_embedding"] = image_embeddings
                                else:
                                    # Create a fallback embedding if output format is unknown
                                    using_mock = True
                                    batch_size = 1 if not isinstance(image_input, list) else len(image_input)
                                    result["image_embedding"] = self.torch.rand((batch_size, 512))
                            except Exception as e:
                                print(f"Error in OpenVINO image inference: {e}")
                                # Create a fallback embedding
                                using_mock = True
                                batch_size = 1 if not isinstance(image_input, list) else len(image_input)
                                result["image_embedding"] = self.torch.rand((batch_size, 512))
                        else:
                            # Create a fallback embedding if no endpoint or we're already using mock
                            using_mock = True
                            batch_size = 1 if not isinstance(image_input, list) else len(image_input)
                            result["image_embedding"] = self.torch.rand((batch_size, 512))
                    except Exception as e:
                        print(f"Error processing image input: {e}")
                        # Create fallback image embedding
                        using_mock = True
                        batch_size = 1 if not isinstance(image_input, list) else len(image_input)
                        result["image_embedding"] = self.torch.rand((batch_size, 512))
                
                # Process text if provided
                if text_input is not None:
                    try:
                        # Process text input(s)
                        if isinstance(text_input, str):
                            # Single text
                            text_inputs = tokenizer(text=[text_input], return_tensors='pt', padding=True)
                        elif isinstance(text_input, list):
                            # List of texts
                            text_inputs = tokenizer(text=text_input, return_tensors='pt', padding=True)
                        else:
                            raise ValueError(f"Unsupported text input type: {type(text_input)}")
                        
                        # Check if we have real endpoint and it's callable
                        if endpoint is not None and callable(endpoint) and not using_mock:
                            # Run inference with OpenVINO
                            try:
                                # Convert to dictionary first if not already
                                text_inputs_dict = dict(text_inputs) if not isinstance(text_inputs, dict) else text_inputs
                                
                                # For OpenVINO optimum models
                                if hasattr(endpoint, 'forward') and callable(endpoint.forward):
                                    text_features = endpoint.forward(**text_inputs_dict)
                                else:
                                    # For direct OpenVINO models
                                    text_features = endpoint(text_inputs_dict)
                                
                                # Extract text embedding from output based on model output format
                                if hasattr(text_features, 'text_embeds'):
                                    # For optimum models that return an object
                                    text_embeddings = self.torch.tensor(text_features.text_embeds)
                                    result["text_embedding"] = text_embeddings
                                elif isinstance(text_features, dict) and "text_embeds" in text_features:
                                    # For models that return a dict with text_embeds
                                    text_embeddings = self.torch.tensor(text_features["text_embeds"])
                                    result["text_embedding"] = text_embeddings
                                elif isinstance(text_features, dict) and "last_hidden_state" in text_features:
                                    # For BERT-like outputs
                                    text_embeddings = self.torch.tensor(text_features["last_hidden_state"])
                                    # Pool to get sentence embedding - average across sequence dimension
                                    if text_embeddings.dim() > 2:
                                        text_embeddings = text_embeddings.mean(dim=1)
                                    result["text_embedding"] = text_embeddings
                                elif isinstance(text_features, dict) and len(text_features) > 0:
                                    # Try to get from first output if named keys aren't available
                                    output_values = list(text_features.values())
                                    if len(output_values) > 0:
                                        text_embeddings = self.torch.tensor(output_values[0])
                                        result["text_embedding"] = text_embeddings
                                    else:
                                        # Create a fallback embedding
                                        using_mock = True
                                        batch_size = 1 if not isinstance(text_input, list) else len(text_input)
                                        result["text_embedding"] = self.torch.rand((batch_size, 512))
                                elif isinstance(text_features, (list, tuple)) and len(text_features) > 0:
                                    # For list-like outputs, use the first element
                                    text_embeddings = self.torch.tensor(text_features[0])
                                    result["text_embedding"] = text_embeddings
                                else:
                                    # Create a fallback embedding if output format is unknown
                                    using_mock = True
                                    batch_size = 1 if not isinstance(text_input, list) else len(text_input)
                                    result["text_embedding"] = self.torch.rand((batch_size, 512))
                            except Exception as e:
                                print(f"Error in OpenVINO text inference: {e}")
                                # Create a fallback embedding
                                using_mock = True
                                batch_size = 1 if not isinstance(text_input, list) else len(text_input)
                                result["text_embedding"] = self.torch.rand((batch_size, 512))
                        else:
                            # Create a fallback embedding if no endpoint or we're already using mock
                            using_mock = True
                            batch_size = 1 if not isinstance(text_input, list) else len(text_input)
                            result["text_embedding"] = self.torch.rand((batch_size, 512))
                    except Exception as e:
                        print(f"Error processing text input: {e}")
                        # Create fallback text embedding
                        using_mock = True
                        batch_size = 1 if not isinstance(text_input, list) else len(text_input)
                        result["text_embedding"] = self.torch.rand((batch_size, 512))
                
                # Calculate similarity if we have both embeddings
                if "image_embedding" in result and "text_embedding" in result:
                    try:
                        # Normalize embeddings
                        image_norm = result["image_embedding"] / result["image_embedding"].norm(dim=-1, keepdim=True)
                        text_norm = result["text_embedding"] / result["text_embedding"].norm(dim=-1, keepdim=True)
                        
                        # Calculate cosine similarity
                        similarity = (text_norm @ image_norm.T)
                        result["similarity"] = similarity
                    except Exception as e:
                        print(f"Error calculating similarity: {e}")
                        # Create a mock similarity
                        using_mock = True
                        result["similarity"] = self.torch.tensor([[0.5]])
                
                # No valid inputs
                if not result:
                    return {"message": "No valid input provided", "implementation_type": "MOCK" if using_mock else "REAL"}
                
                # Add implementation type information
                result["implementation_type"] = "MOCK" if using_mock else "REAL"
                
                # Add MOCK/REAL indicator to results - create a copy of keys first to avoid dictionary size change during iteration
                keys_to_process = list(result.keys())
                for key in keys_to_process:
                    if isinstance(result[key], (self.torch.Tensor, list, tuple)) and key != "similarity" and key != "implementation_type":
                        result[key + "_status"] = "MOCK" if using_mock else "REAL"
                
                # Return single embedding if that's all that was requested
                if len(result) == 2 and (
                    "image_embedding" in result or 
                    "text_embedding" in result
                ) and "implementation_type" in result:
                    embedding_key = next(k for k in result.keys() if k != "implementation_type")
                    return {
                        embedding_key: result[embedding_key],
                        embedding_key + "_status": "MOCK" if using_mock else "REAL",
                        "implementation_type": "MOCK" if using_mock else "REAL"
                    }
                
                return result
                
            except Exception as e:
                print(f"Error in OpenVINO CLIP handler: {e}")
                # Return a minimal result with mock data on complete failure
                return {
                    "error": str(e),
                    "status": "MOCK",
                    "implementation_type": "MOCK"
                }
                
        return handler

    def openvino_skill_convert(self, model_name, model_dst_path, task, weight_format, hfmodel=None, hfprocessor=None):
        """Convert a CLIP model to OpenVINO format.
        
        Args:
            model_name: Name or path of the HuggingFace model
            model_dst_path: Destination path for the converted model
            task: Model task type (usually 'feature-extraction' for CLIP)
            weight_format: Weight format for quantization (int8, int4, etc.)
            hfmodel: Pre-loaded model (optional)
            hfprocessor: Pre-loaded processor (optional)
            
        Returns:
            Compiled OpenVINO model or None if conversion fails
        """
        print(f"Converting {model_name} to OpenVINO format with task={task}, format={weight_format}")
        
        try:
            # Create the destination directory if it doesn't exist
            os.makedirs(model_dst_path, exist_ok=True)
            model_xml_path = os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml")
            
            # If model is already converted, just load and return it
            if os.path.exists(model_xml_path):
                print(f"Found existing OpenVINO CLIP model at {model_xml_path}, loading it")
                core = self.ov.Core()
                ov_model = core.read_model(model_xml_path)
                ov_model = core.compile_model(ov_model)
                return ov_model
                
            # Load model if not provided
            if hfmodel is None:
                try:
                    print("Loading CLIP model...")
                    hfmodel = self.transformers.CLIPModel.from_pretrained(model_name, torch_dtype=self.torch.float16)
                except Exception as e:
                    print(f"Error loading CLIPModel: {e}, trying AutoModel")
                    hfmodel = self.transformers.AutoModel.from_pretrained(model_name, torch_dtype=self.torch.float16)
        
            # Load processor if not provided
            if hfprocessor is None:
                try:
                    print("Loading CLIP processor...")
                    hfprocessor = self.transformers.CLIPProcessor.from_pretrained(model_name)
                except Exception as e:
                    print(f"Error loading CLIPProcessor: {e}, trying AutoProcessor")
                    hfprocessor = self.transformers.AutoProcessor.from_pretrained(model_name)
                    
            # Set up inputs for tracing the model
            if hfprocessor is not None:
                # Prepare sample inputs for conversion
                try:
                    # Use a sample image and text for conversion
                    text = "Example text for conversion."
                    # First try with a local image if available
                    local_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.jpg")
                    if os.path.exists(local_image_path):
                        image = load_image(local_image_path)
                    else:
                        # Use a remote image if no local image is available
                        image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
                        image = load_image(image_url)
                        
                    # Process the data
                    processed_data = hfprocessor(
                        text=text,
                        images=[image],
                        return_tensors="pt", 
                        padding=True
                    )
                    
                    # Run a forward pass to make sure everything works
                    print("Running forward pass on CLIP model...")
                    with self.torch.no_grad():
                        results = hfmodel(**processed_data)
                    
                    # Prepare model for conversion
                    print("Converting CLIP model to OpenVINO format...")
                    hfmodel.config.torchscript = True
                    
                    # Convert to OpenVINO model
                    ov_model = self.ov.convert_model(hfmodel, example_input=dict(processed_data))
                    
                    # Save model to disk
                    print(f"Saving OpenVINO CLIP model to {model_xml_path}")
                    self.ov.save_model(ov_model, model_xml_path)
                    
                    # Compile and return the model
                    ov_model = self.ov.compile_model(ov_model)
                    
                    # Clean up to reduce memory usage
                    del hfmodel
                    
                    return ov_model
                    
                except Exception as conversion_error:
                    print(f"Error during CLIP model conversion: {conversion_error}")
                    import traceback
                    traceback.print_exc()
                    return None
            else:
                print("No processor available, cannot convert CLIP model")
                return None
                
        except Exception as e:
            print(f"Error in openvino_skill_convert: {e}")
            import traceback
            traceback.print_exc()
            return None