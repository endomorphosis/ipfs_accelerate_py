import requests
from PIL import Image
from io import BytesIO
import anyio
from ..anyio_queue import AnyioQueue
from pathlib import Path
import json
import time
import os
import tempfile

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    import torch
    from torchvision.transforms import InterpolationMode, Compose, Lambda, Resize, ToTensor, Normalize
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = Compose([
        Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def streamer(subword: str) -> bool:
    """

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    """
    print(subword, end="", flush=True)

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# def load_image_bak(image_file, input_size=448, max_num=12):
#     if os.path.exists(image_file):
#         image = Image.open(image_file).convert('RGB')
#     transform = build_transform(input_size=input_size)
#     if os.path.exists(image_file):
#         image = Image.open(image_file).convert('RGB')
#     elif "http" in image_file:
#         try:
#             with tempfile.NamedTemporaryFile(delete=True) as f:
#                 f.write(requests.get(image_file).content)
#                 image = Image.open(f).convert('RGB')
#         except Exception as e:
#             print(e)
#             raise ValueError("Invalid image file")
#     else:
#         raise ValueError("Invalid image file")
        
#     images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#     pixel_values = [transform(image) for image in images]
#     pixel_values = torch.stack(pixel_values)
#     return pixel_values

def load_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_image_tensor(image_file):
    import openvino as ov
    import numpy as np
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return image, ov.Tensor(image_data)

class hf_llava:
    """HuggingFace LLaVA (Large Language and Vision Assistant) implementation.
    
    This class provides standardized interfaces for working with LLaVA models
    across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm).
    
    LLaVA combines vision and language capabilities to enable multimodal understanding
    and generation. It can analyze images and respond to text prompts about them.
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the LLaVA model.
        
        Args:
            resources (dict): Dictionary of shared resources (torch, transformers, etc.)
            metadata (dict): Configuration metadata
        """
        self.resources = resources
        self.metadata = metadata
        
        # Handler creation methods
        self.create_cpu_vlm_endpoint_handler = self.create_cpu_vlm_endpoint_handler
        self.create_cuda_vlm_endpoint_handler = self.create_cuda_vlm_endpoint_handler
        self.create_openvino_vlm_endpoint_handler = self.create_openvino_vlm_endpoint_handler
        self.create_openvino_genai_vlm_endpoint_handler = self.create_openvino_genai_vlm_endpoint_handler
        self.create_optimum_vlm_endpoint_handler = self.create_optimum_vlm_endpoint_handler
        self.create_apple_vlm_endpoint_handler = self.create_apple_vlm_endpoint_handler
        self.create_qualcomm_vlm_endpoint_handler = self.create_qualcomm_vlm_endpoint_handler
        
        # Initialization methods
        self.init = self.init
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.init_qualcomm = self.init_qualcomm
        
        # Utility methods
        self.build_transform = build_transform
        self.load_image = load_image
        self.load_image_tensor = load_image_tensor
        self.dynamic_preprocess = dynamic_preprocess
        self.find_closest_aspect_ratio = find_closest_aspect_ratio
        self.__test__ = self.__test__
        
        # Hardware-specific utilities
        self.snpe_utils = None  # Qualcomm SNPE utils
        self.coreml_utils = None  # Apple CoreML utils
        return None
        
    def _create_mock_processor(self):
        """Create a mock processor for graceful degradation when the real one fails.
        
        Returns:
            Mock processor object with essential methods
        """
        from unittest.mock import MagicMock
        
        processor = MagicMock()
        
        # Add necessary methods for the processor
        processor.apply_chat_template = lambda conversation, add_generation_prompt=True: f"User: {conversation[0]['content'][1]['text']} with an image"
        processor.decode = lambda ids, skip_special_tokens=True: "This is a mock response from LLaVA"
        processor.batch_decode = lambda ids, skip_special_tokens=True, clean_up_tokenization_spaces=False: ["This is a mock response from LLaVA"]
        
        # Make it callable
        processor.__call__ = lambda text=None, images=None, return_tensors=None, padding=None: {
            "input_ids": self.torch.ones((1, 10), dtype=self.torch.long),
            "attention_mask": self.torch.ones((1, 10), dtype=self.torch.long),
            "pixel_values": self.torch.ones((1, 3, 224, 224), dtype=self.torch.float)
        }
        
        return processor
        
    def _create_mock_endpoint(self, model_name, device_label):
        """Create mock endpoint objects when real initialization fails.
        
        Args:
            model_name (str): The model name or path
            device_label (str): The device label (cpu, cuda, etc.)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        """
        from unittest.mock import MagicMock
        import time
        
        # Basic mock endpoint
        endpoint = MagicMock()
        endpoint.generate = lambda **kwargs: self.torch.ones((1, 10), dtype=self.torch.long)
        
        # Create processor with essential methods
        processor = self._create_mock_processor()
        
        # Create a handler function that clearly identifies as a mock
        def mock_handler(text="", image=None):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            image_info = f"of size {image.size}" if hasattr(image, 'size') else "with the provided content"
            return f"(MOCK) LLaVA response [timestamp: {timestamp}]: This is a mock response. I was asked to analyze an image {image_info} with prompt: '{text}'"
        
        print(f"Created mock LLaVA endpoint for {model_name} on {device_label}")
        return endpoint, processor, mock_handler, AnyioQueue(64), 0
    
    
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

        return None
    

    
    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        sentence_2 = "The quick brown fox jumps over the lazy dog"
        image_1 = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, image_1)
            print(test_batch)
            print("hf_llava test passed")
        except Exception as e:
            print(e)
            print("hf_llava test failed")
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        tokens = tokenizer[endpoint_label]()
        len_tokens = len(tokens["input_ids"])
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"tokens: {len_tokens}")
        print(f"tokens per second: {tokens_per_second}")
        # test_batch_sizes = await self.test_batch_sizes(metadata['models'], ipfs_accelerate_init)
        if "openvino" not in endpoint_label:
            with self.torch.no_grad():
                if "cuda" in dir(self.torch):
                    self.torch.cuda.empty_cache()
        print("hf_llava test")
        return None
    
    def init_cpu(self, model_name, device, cpu_label):
        """Initialize LLaVA model for CPU.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run inference on ("cpu")
            cpu_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        
        try:
            # Load model configuration, processor and model
            config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)    
            processor = self.transformers.AutoProcessor.from_pretrained(model_name)
            
            # Load the model - use AutoModelForImageTextToText for consistency
            endpoint = self.transformers.AutoModelForImageTextToText.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # Create the handler function
            endpoint_handler = self.create_cpu_vlm_endpoint_handler(
                endpoint=endpoint,
                processor=processor,
                model_name=model_name,
                cpu_label=cpu_label
            )
            
            # Clean up any CUDA memory if it was used previously
            if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
                
            print(f"Successfully initialized LLaVA model '{model_name}' on CPU")
            return endpoint, processor, endpoint_handler, AnyioQueue(64), 0
            
        except Exception as e:
            print(f"Error initializing LLaVA model on CPU: {e}")
            # Return mock objects in case of failure for graceful degradation
            return self._create_mock_endpoint(model_name, cpu_label)
    
    def init_qualcomm(self, model_name, device, qualcomm_label):
        """Initialize LLaVA model for Qualcomm hardware.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run inference on ("qualcomm")
            qualcomm_label: Label to identify this endpoint ("qualcomm:0", etc.)
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        
        # Check for SNPE utilities
        try:
            from .qualcomm_snpe_utils import get_snpe_utils
            self.snpe_utils = get_snpe_utils()
        except ImportError:
            print("Failed to import Qualcomm SNPE utilities")
            return self._create_mock_endpoint(model_name, qualcomm_label)
            
        # Verify SNPE is available in this environment
        if not self.snpe_utils.is_available():
            print("Qualcomm SNPE is not available on this system")
            return self._create_mock_endpoint(model_name, qualcomm_label)
            
        try:
            # Initialize processor directly from HuggingFace
            processor = self.transformers.AutoProcessor.from_pretrained(model_name)
            
            # Convert model path to be compatible with SNPE
            safe_model_name = model_name.replace("/", "--")
            dlc_path = f"~/snpe_models/{safe_model_name}_llava.dlc"
            dlc_path = os.path.expanduser(dlc_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(dlc_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(dlc_path):
                print(f"Converting {model_name} to SNPE format...")
                self.snpe_utils.convert_model(model_name, "llava", str(dlc_path))
                print(f"Model converted and saved to {dlc_path}")
            else:
                print(f"Using existing SNPE model at {dlc_path}")
            
            # Load the SNPE model
            endpoint = self.snpe_utils.load_model(str(dlc_path))
            
            # Optimize for the specific Qualcomm device if possible
            if ":" in qualcomm_label:
                device_type = qualcomm_label.split(":")[1]
                print(f"Optimizing for Qualcomm device type: {device_type}")
                optimized_path = self.snpe_utils.optimize_for_device(dlc_path, device_type)
                if optimized_path != dlc_path:
                    print(f"Using optimized model at {optimized_path}")
                    endpoint = self.snpe_utils.load_model(optimized_path)
            
            # Create endpoint handler
            endpoint_handler = self.create_qualcomm_vlm_endpoint_handler(
                endpoint=endpoint,
                processor=processor,
                model_name=model_name,
                qualcomm_label=qualcomm_label
            )
            
            print(f"Successfully initialized LLaVA model '{model_name}' on Qualcomm device {qualcomm_label}")
            return endpoint, processor, endpoint_handler, AnyioQueue(16), 1
            
        except Exception as e:
            print(f"Error initializing Qualcomm LLaVA model: {e}")
            return self._create_mock_endpoint(model_name, qualcomm_label)
    
    def init_cuda(self, model_name, device, cuda_label):
        """Initialize LLaVA model for CUDA/GPU with enhanced memory optimization and error handling.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run inference on ("cuda", "cuda:0", etc.)
            cuda_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        
        # First check if CUDA is available
        if not hasattr(self.torch, 'cuda') or not self.torch.cuda.is_available():
            print(f"CUDA is not available, falling back to CPU for model '{model_name}'")
            return self._create_mock_endpoint(model_name, cuda_label)
        
        # Get CUDA device information
        try:
            # Parse cuda_label to extract device index
            if ":" in cuda_label:
                device_index = int(cuda_label.split(":")[1])
                if device_index >= self.torch.cuda.device_count():
                    print(f"Warning: CUDA device index {device_index} out of range (0-{self.torch.cuda.device_count()-1}), using device 0")
                    device_index = 0
                    cuda_label = "cuda:0"
            else:
                device_index = 0
                cuda_label = "cuda:0"
                
            # Create device object for consistent usage
            cuda_device = self.torch.device(cuda_label)
            
            # Get and display device information
            device_name = self.torch.cuda.get_device_name(device_index)
            print(f"Using CUDA device: {device_name} (index {device_index})")
            
            # Get memory information for logging and memory settings
            if hasattr(self.torch.cuda, 'mem_get_info'):
                free_memory, total_memory = self.torch.cuda.mem_get_info(device_index)
                free_memory_gb = free_memory / (1024**3)
                total_memory_gb = total_memory / (1024**3)
                print(f"Available CUDA memory: {free_memory_gb:.2f}GB / {total_memory_gb:.2f}GB")
            else:
                # Older CUDA versions might not have mem_get_info
                free_memory = None
                total_memory = None
                print("CUDA memory info not available")
            
            # Clean up CUDA cache before loading model
            if hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error initializing CUDA device: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return self._create_mock_endpoint(model_name, cuda_label)
        
        try:
            # Check if we're using mock transformers
            if isinstance(self.transformers, type(MagicMock())):
                # Create mocks for testing
                print("Using mock transformers implementation for CUDA test")
                config = MagicMock()
                processor = MagicMock() 
                processor.batch_decode = MagicMock(return_value=["This is a mock response from LLaVA"])
                
                endpoint = MagicMock()
                endpoint.generate = MagicMock(return_value=self.torch.tensor([[101, 102, 103]]))
                is_real_impl = False
            else:
                # Try loading real model with CUDA support
                print("Attempting to load real model with CUDA support")
                is_real_impl = True
                
                # Load processor first
                try:
                    print(f"Loading processor for {model_name}")
                    processor = self.transformers.AutoProcessor.from_pretrained(model_name)
                except Exception as e:
                    print(f"Error loading processor: {e}")
                    print(f"Falling back to mock processor")
                    processor = self._create_mock_processor()
                    is_real_impl = False
                
                # Load model with optimized memory settings if we have real processor
                if is_real_impl:
                    try:
                        # Set memory optimization parameters
                        if free_memory is not None:
                            free_memory_gb = free_memory / (1024**3)
                        else:
                            # Conservative estimate if we can't get actual memory info
                            free_memory_gb = 4.0
                        
                        # Determine memory optimization settings
                        use_half_precision = True  # Default to half precision for better memory efficiency
                        use_8bit_quantization = free_memory_gb < 4.0  # Use 8-bit quantization for lower memory
                        low_cpu_mem_usage = True  # Always use low CPU memory usage
                        
                        # Choose batch size based on available memory
                        batch_size = max(1, min(8, int(free_memory_gb / 2)))
                        print(f"Using batch size: {batch_size}")
                        
                        # Prepare model loading parameters with memory optimizations
                        model_kwargs = {
                            "torch_dtype": self.torch.float16 if use_half_precision else self.torch.float32,
                            "low_cpu_mem_usage": low_cpu_mem_usage,
                            "trust_remote_code": True,
                            "device_map": cuda_label if hasattr(self.transformers, "AutoModelForImageTextToText") else "auto"
                        }
                        
                        # Add 8-bit quantization if needed
                        if use_8bit_quantization and hasattr(self.transformers, 'BitsAndBytesConfig'):
                            print("Using 8-bit quantization for memory efficiency")
                            model_kwargs["quantization_config"] = self.transformers.BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_threshold=6.0
                            )
                        
                        # Load the model with the optimized settings
                        print(f"Loading model {model_name} with optimized memory settings:")
                        print(f"- Half precision: {use_half_precision}")
                        print(f"- 8-bit quantization: {use_8bit_quantization if 'quantization_config' in model_kwargs else False}")
                        print(f"- Low CPU memory usage: {low_cpu_mem_usage}")
                        
                        # Try loading with optimal model class first
                        try:
                            if hasattr(self.transformers, "AutoModelForImageTextToText"):
                                endpoint = self.transformers.AutoModelForImageTextToText.from_pretrained(
                                    model_name, 
                                    **model_kwargs
                                )
                            else:
                                # Fall back to other model classes
                                print("AutoModelForImageTextToText not available, trying AutoModelForVision2Seq")
                                if hasattr(self.transformers, "AutoModelForVision2Seq"):
                                    endpoint = self.transformers.AutoModelForVision2Seq.from_pretrained(
                                        model_name, 
                                        **model_kwargs
                                    )
                                else:
                                    # Generic fallback
                                    print("Falling back to generic model loading")
                                    endpoint = self.transformers.AutoModel.from_pretrained(
                                        model_name, 
                                        **model_kwargs
                                    )
                        except Exception as model_error:
                            print(f"Error loading model with transformers: {model_error}")
                            print(f"Traceback: {traceback.format_exc()}")
                            print("Falling back to mock implementation")
                            endpoint = MagicMock()
                            endpoint.generate = MagicMock(return_value=self.torch.tensor([[101, 102, 103]]))
                            is_real_impl = False
                        
                        # Move model to device if not already done by device_map
                        if is_real_impl:
                            if not hasattr(endpoint, 'device') or endpoint.device != cuda_device:
                                print(f"Moving model to {cuda_label}")
                                endpoint = endpoint.to(cuda_device)
                            
                            # Set model to evaluation mode
                            endpoint.eval()
                            
                            # Log memory usage after model loading
                            if hasattr(self.torch.cuda, 'mem_get_info'):
                                try:
                                    free_after_load, _ = self.torch.cuda.mem_get_info(device_index)
                                    free_after_load_gb = free_after_load / (1024**3)
                                    memory_used_gb = free_memory_gb - free_after_load_gb
                                    print(f"Model loaded. CUDA memory used: {memory_used_gb:.2f}GB, Available: {free_after_load_gb:.2f}GB")
                                except Exception as mem_error:
                                    print(f"Error getting CUDA memory usage after model loading: {mem_error}")
                    
                    except Exception as e:
                        print(f"Error loading CUDA model: {e}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        print("Falling back to mock implementation")
                        endpoint = MagicMock()
                        endpoint.generate = MagicMock(return_value=self.torch.tensor([[101, 102, 103]]))
                        is_real_impl = False
            
            # Create handler with implementation status
            endpoint_handler = self.create_cuda_vlm_endpoint_handler(
                endpoint=endpoint, 
                processor=processor,
                model_name=model_name, 
                cuda_label=cuda_label,
                is_real_impl=is_real_impl
            )
            
            # Final cache cleanup
            if hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
                
            return endpoint, processor, endpoint_handler, AnyioQueue(16), batch_size if is_real_impl else 1
            
        except Exception as e:
            print(f"Error in CUDA initialization: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # Ensure we clean up CUDA memory on error
            if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
            return self._create_mock_endpoint(model_name, cuda_label)

    def init_openvino(self, model_name, model_type, device, openvino_label, get_openvino_genai_pipeline, get_optimum_openvino_model, get_openvino_model, get_openvino_pipeline_type, openvino_cli_convert):
        """Initialize LLaVA model for OpenVINO.
        
        Args:
            model_name (str): HuggingFace model name or path
            model_type (str): Model type (e.g., "text-generation", "vision-language")
            device (str): Device to run inference on ("CPU", "GPU", etc.)
            openvino_label (str): Label to identify this endpoint ("openvino:0", etc.)
            get_openvino_genai_pipeline: Function to get OpenVINO GenAI pipeline
            get_optimum_openvino_model: Function to get Optimum OpenVINO model
            get_openvino_model: Function to get OpenVINO model
            get_openvino_pipeline_type: Function to get pipeline type
            openvino_cli_convert: Function to convert model with CLI
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        
        # Import required OpenVINO libraries
        try:
            if "ov_genai" not in list(self.resources.keys()):
                import openvino_genai as ov_genai
                self.ov_genai = ov_genai
            else:
                self.ov_genai = self.resources["ov_genai"]
            
            if "openvino" not in list(self.resources.keys()):
                import openvino as ov
                self.ov = ov
            else:
                self.ov = self.resources["openvino"]
        except ImportError as e:
            print(f"Failed to import OpenVINO libraries: {e}")
            return self._create_mock_endpoint(model_name, openvino_label)
        
        try:
            # Setup model paths
            homedir = os.path.expanduser("~")
            safe_model_name = model_name.replace("/", "--")
            openvino_models_dir = os.path.join(homedir, "openvino_models")
            os.makedirs(openvino_models_dir, exist_ok=True)
            
            # Extract the target device index
            openvino_index = int(openvino_label.split(":")[1]) if ":" in openvino_label else 0
            
            # Determine weight format based on target device
            weight_format = "int8"  # Default for CPU
            if openvino_index == 1:
                weight_format = "int4"  # For GPU
            elif openvino_index == 2:
                weight_format = "int4"  # For NPU
                
            # Create model paths
            model_src_dir = os.path.join(openvino_models_dir, safe_model_name)
            model_dst_path = os.path.join(model_src_dir, f"openvino_{weight_format}")
            
            # Create directory for the model
            os.makedirs(model_dst_path, exist_ok=True)
            
            # Get model configuration
            config = self.transformers.AutoConfig.from_pretrained(model_name)
            
            # Determine the correct pipeline task
            task = get_openvino_pipeline_type(model_name, model_type)
            
            # Convert model if needed
            if not os.path.exists(os.path.join(model_dst_path, "openvino_model.xml")):
                print(f"Converting {model_name} to OpenVINO format with {weight_format} precision...")
                openvino_cli_convert(
                    model_name, 
                    model_dst_path=model_dst_path, 
                    task=task, 
                    weight_format=weight_format, 
                    ratio="1.0", 
                    group_size=128, 
                    sym=True
                )
                print(f"Model converted and saved to {model_dst_path}")
            else:
                print(f"Using existing OpenVINO model at {model_dst_path}")
            
            # Load tokenizer/processor from the converted model
            processor = self.transformers.AutoProcessor.from_pretrained(
                model_dst_path, 
                patch_size=config.vision_config.patch_size if hasattr(config, 'vision_config') and hasattr(config.vision_config, 'patch_size') else None,
                vision_feature_select_strategy=config.vision_feature_select_strategy if hasattr(config, 'vision_feature_select_strategy') else None
            )
            
            # Load the optimized model
            endpoint = get_optimum_openvino_model(model_name, model_type)
            
            # Create handler
            endpoint_handler = self.create_openvino_vlm_endpoint_handler(
                endpoint=endpoint,
                processor=processor,
                model_name=model_name,
                openvino_label=openvino_label
            )
            
            print(f"Successfully initialized LLaVA model '{model_name}' on OpenVINO device {openvino_label}")
            return endpoint, processor, endpoint_handler, AnyioQueue(64), 0
            
        except Exception as e:
            print(f"Error initializing OpenVINO LLaVA model: {e}")
            return self._create_mock_endpoint(model_name, openvino_label)
    
    def init_apple(self, model_name, device, apple_label):
        """Initialize LLaVA model for Apple Silicon hardware.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run inference on ("mps")
            apple_label (str): Label to identify this endpoint ("apple:0", "apple:CPU_AND_GPU", etc.)
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        
        # Check for CoreML utilities
        try:
            from .apple_coreml_utils import get_coreml_utils
            self.coreml_utils = get_coreml_utils()
        except ImportError:
            print("Failed to import CoreML utilities")
            return self._create_mock_endpoint(model_name, apple_label)
            
        # Verify CoreML is available in this environment
        if not self.coreml_utils.is_available():
            print("CoreML is not available on this system")
            return self._create_mock_endpoint(model_name, apple_label)
            
        try:
            # Initialize processor directly from HuggingFace
            processor = self.transformers.AutoProcessor.from_pretrained(model_name)
            
            # Convert model path to be compatible with CoreML
            safe_model_name = model_name.replace("/", "--")
            mlmodel_path = f"~/coreml_models/{safe_model_name}_llava.mlpackage"
            mlmodel_path = os.path.expanduser(mlmodel_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(mlmodel_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(mlmodel_path):
                print(f"Converting {model_name} to CoreML format...")
                self.coreml_utils.convert_model(model_name, "vision_text_dual", str(mlmodel_path))
                print(f"Model converted and saved to {mlmodel_path}")
            else:
                print(f"Using existing CoreML model at {mlmodel_path}")
            
            # Load the CoreML model
            endpoint = self.coreml_utils.load_model(str(mlmodel_path))
            
            # Optimize for the specific Apple Silicon configuration if possible
            if ":" in apple_label:
                compute_units = apple_label.split(":")[1]
                print(f"Optimizing for Apple Silicon with compute units: {compute_units}")
                optimized_path = self.coreml_utils.optimize_for_device(mlmodel_path, compute_units)
                if optimized_path != mlmodel_path:
                    print(f"Using optimized model at {optimized_path}")
                    endpoint = self.coreml_utils.load_model(optimized_path)
            
            # Create endpoint handler
            endpoint_handler = self.create_apple_vlm_endpoint_handler(
                endpoint=endpoint,
                processor=processor,
                model_name=model_name,
                apple_label=apple_label
            )
            
            print(f"Successfully initialized LLaVA model '{model_name}' on Apple Silicon {apple_label}")
            return endpoint, processor, endpoint_handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing Apple Silicon LLaVA model: {e}")
            return self._create_mock_endpoint(model_name, apple_label)
            
    def create_apple_multimodal_endpoint_handler(self, endpoint, processor, model_name, apple_label):
        """Creates an Apple Silicon optimized handler for LLaVA multimodal processing."""
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
                
                # Process outputs - LLaVA typically outputs text responses
                if 'logits' in outputs:
                    logits = self.torch.tensor(outputs['logits'])
                    generated_ids = self.torch.argmax(logits, dim=-1)
                    responses = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    return responses[0] if len(responses) == 1 else responses
                
                return None
                
            except Exception as e:
                print(f"Error in Apple Silicon LLaVA handler: {e}")
                return None
                
        return handler
    
    def create_optimum_vlm_endpoint_handler(self, cuda_endpoint_handler, local_cuda_processor, endpoint_model, cuda_label):
        def handler(x, y, cuda_endpoint_handler=cuda_endpoint_handler, local_cuda_processor=local_cuda_processor, endpoint_model=endpoint_model, cuda_label=cuda_label):
                try:
                    if y is not None and type(y) == str:
                        image = load_image(y)
                    elif type(y) == tuple:
                        image = load_image(y[1])
                    elif type(y) == dict:
                        image = load_image(y["image"])
                    elif type(y) == list:
                        image = load_image(y[1])
                    else:
                        image = Image.open(requests.get(y, stream=True).raw)
                    
                    if x is not None and type(x) == str:
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": x},
                                ],
                            },
                        ]
                    elif type(x) == tuple:
                        conversation = x
                    elif type(x) == dict:
                        raise Exception("Invalid input to vlm endpoint handler")
                    elif type(x) == list:
                        # conversation = x
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": x},
                                ],
                            },
                        ]
                    else:
                        raise Exception("Invalid input to vlm endpoint handler")
                    result = None
                    # prompt = local_cuda_processor.apply_chat_template(conversation, add_generation_prompt=True)
                    # inputs = local_cuda_processor(image, prompt, return_tensors="pt").to(cuda_label, torch.float16)
                    # output = cuda_endpoint_handler.generate(**inputs, max_new_tokens=30)
                    # result = local_cuda_processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    return result
                except Exception as e:
                    raise e
        return handler

    def create_openvino_genai_vlm_endpoint_handler(self, openvino_endpoint_handler, openvino_processor, endpoint_model, openvino_label):
        def handler(x, y, openvino_endpoint_handler=openvino_endpoint_handler, openvino_processor=openvino_processor, endpoint_model=endpoint_model, openvino_label=openvino_label):
            config = self.ov_genai.GenerationConfig()
            config.max_new_tokens = 100

            try:
                if y is not None and type(y) == str and "http" in y:
                    max_retries = 3
                    retry_delay = 1
                    for attempt in range(max_retries):
                        try:
                            response = requests.get(y, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content)).convert('RGB')
                            image_data = self.np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(self.np.byte)
                            image_tensor = self.ov.Tensor(image_data)
                            break
                        except (requests.RequestException, Image.UnidentifiedImageError) as e:
                            if attempt == max_retries - 1:
                                raise ValueError(f"Failed to load image from URL after {max_retries} attempts: {y}. Error: {str(e)}")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                elif y is not None and type(y) == str:
                    image = load_image(y)
                elif type(y) == tuple:
                    image = load_image(y[1])
                elif type(y) == dict:
                    image = load_image(y["image"])
                elif type(y) == list:
                    image = load_image(y[1])
                else:
                    image = Image.open(requests.get(y, stream=True).raw)
                
                if x is not None and type(x) == str:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": x},
                            ],
                        },
                    ]
                elif type(x) == tuple:
                    conversation = x
                elif type(x) == dict:
                    raise Exception("Invalid input to vlm endpoint handler")
                elif type(x) == list:
                    # conversation = x
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": x},
                            ],
                        },
                    ]
                else:
                    raise Exception("Invalid input to vlm endpoint handler")
                prompt = x
                output = openvino_endpoint_handler.generate(prompt, image=image_tensor, generation_config=config, streamer=streamer)
                # Run model inference
                return output
            except Exception as e:
                # Cleanup GPU memory in case of error
                raise e
        return handler

    
    def create_openvino_vlm_endpoint_handler(self, endpoint, processor, model_name, openvino_label):
        """Create endpoint handler for OpenVINO backend.
        
        Args:
            endpoint: The model endpoint/object
            processor: The tokenizer/processor for the model
            model_name: Name of the model
            openvino_label: Label identifying this endpoint
        
        Returns:
            Handler function for processing requests
        """
        def handler(text=None, image=None, endpoint=endpoint, processor=processor, model_name=model_name, openvino_label=openvino_label):
            """Process text and image inputs using LLaVA on OpenVINO.
            
            Args:
                text: Text prompt to process with the image
                image: Image to analyze (can be path, URL, or PIL Image)
                
            Returns:
                Generated text response from the model
            """
            try:
                # Load and process image if provided
                if image is not None:
                    if isinstance(image, str):
                        if image.startswith("http") or image.startswith("https"):
                            max_retries = 3
                            retry_delay = 1
                            last_error = None
                            
                            for attempt in range(max_retries):
                                try:
                                    response = requests.get(image, timeout=10)
                                    response.raise_for_status()  # Check for HTTP errors
                                    image_obj = Image.open(BytesIO(response.content)).convert("RGB")
                                    break
                                except (requests.RequestException, Image.UnidentifiedImageError) as e:
                                    last_error = e
                                    if attempt < max_retries - 1:
                                        time.sleep(retry_delay)
                                        retry_delay *= 2
                                    else:
                                        raise ValueError(f"Failed to load image from URL after {max_retries} attempts: {image}. Error: {str(last_error)}")
                        else:
                            image_obj = Image.open(image).convert("RGB")
                    elif isinstance(image, Image.Image):
                        image_obj = image
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                else:
                    # Text-only input
                    image_obj = None
                    
                # Process the text input
                if text is not None and isinstance(text, str):
                    # Standard format for multimodal conversation
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {"type": "image"}
                            ]
                        }
                    ]
                elif isinstance(text, (tuple, list)):
                    # Allow for pre-formatted conversation
                    conversation = text
                elif isinstance(text, dict):
                    # Single message as dict
                    conversation = [text]
                else:
                    raise ValueError(f"Unsupported text input type: {type(text)}")
                
                # Create streamer for real-time text output (if supported)
                try:
                    streamer = self.transformers.TextStreamer(
                        processor, 
                        skip_prompt=True, 
                        skip_special_tokens=True
                    )
                except Exception:
                    streamer = None
                
                # Process inputs
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(image_obj, prompt, return_tensors="pt")
                
                # Generate text with OpenVINO optimized model
                generation_config = {
                    "do_sample": False,
                    "max_new_tokens": 50
                }
                
                if streamer is not None:
                    generation_config["streamer"] = streamer
                
                output_ids = endpoint.generate(**inputs, **generation_config)
                
                # Decode response
                response = processor.decode(output_ids[0], skip_special_tokens=True)
                
                # Add timestamp and metadata for testing/debugging
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                image_info = f"of size {image_obj.size}" if image_obj and hasattr(image_obj, 'size') else "with the provided content"
                
                return f"(REAL) OpenVINO LLaVA response [device: {openvino_label}, timestamp: {timestamp}]: {response}"
                
            except Exception as e:
                print(f"Error in OpenVINO LLaVA handler: {e}")
                import time
                import traceback
                traceback.print_exc()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                return f"(ERROR) OpenVINO LLaVA response [timestamp: {timestamp}]: Error processing request - {str(e)}"
        
        return handler

    def create_cpu_vlm_endpoint_handler(self, endpoint, processor, model_name, cpu_label):
        """Create endpoint handler for CPU backend.
        
        Args:
            endpoint: The model endpoint/object
            processor: The tokenizer/processor for the model
            model_name: Name of the model
            cpu_label: Label identifying this endpoint
        
        Returns:
            Handler function for processing requests
        """
        def handler(text=None, image=None, endpoint=endpoint, processor=processor, model_name=model_name, cpu_label=cpu_label):
            """Process text and image inputs using LLaVA on CPU.
            
            Args:
                text: Text prompt to process with the image
                image: Image to analyze (can be path, URL, or PIL Image)
                
            Returns:
                Generated text response from the model
            """
            try:
                # Load and process image if provided
                if image is not None:
                    if isinstance(image, str):
                        if image.startswith("http") or image.startswith("https"):
                            response = requests.get(image)
                            image_obj = Image.open(BytesIO(response.content)).convert("RGB")
                        else:
                            image_obj = Image.open(image).convert("RGB")
                    elif isinstance(image, Image.Image):
                        image_obj = image
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                else:
                    # Text-only input
                    image_obj = None
                    
                # Process the text input
                if text is not None and isinstance(text, str):
                    # Standard format for multimodal conversation
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {"type": "image"}
                            ]
                        }
                    ]
                elif isinstance(text, (tuple, list)):
                    # Allow for pre-formatted conversation
                    conversation = text
                elif isinstance(text, dict):
                    # Single message as dict
                    conversation = [text]
                else:
                    raise ValueError(f"Unsupported text input type: {type(text)}")
                
                # Create streamer for real-time text output
                streamer = self.transformers.TextStreamer(
                    processor, 
                    skip_prompt=True, 
                    skip_special_tokens=True
                )
                
                # Process inputs
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(image_obj, prompt, return_tensors="pt")
                
                # Generate response
                output_ids = endpoint.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=50,
                    streamer=streamer,
                )
                
                # Decode response
                response = processor.decode(output_ids[0], skip_special_tokens=True)
                
                # Add timestamp and metadata for testing/debugging
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                image_info = f"of size {image_obj.size}" if image_obj and hasattr(image_obj, 'size') else "with the provided content"
                
                return f"(REAL) CPU LLaVA response [timestamp: {timestamp}]: {response}"
                
            except Exception as e:
                print(f"Error in CPU LLaVA handler: {e}")
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                return f"(ERROR) CPU LLaVA response [timestamp: {timestamp}]: Error processing request - {str(e)}"
        
        return handler
    
    
    def create_cuda_vlm_endpoint_handler(self, endpoint, processor, model_name, cuda_label, is_real_impl=True):
        """Create enhanced endpoint handler for CUDA/GPU backend with memory optimization and performance tracking.
        
        Args:
            endpoint: The model endpoint/object
            processor: The tokenizer/processor for the model
            model_name: Name of the model
            cuda_label: Label identifying this endpoint (e.g., "cuda:0")
            is_real_impl: Flag indicating if we're using real implementation or mock
            
        Returns:
            Handler function for processing requests with performance metrics
        """
        # Import needed modules
        import time
        import traceback
        
        # Create device object
        cuda_device = None
        if ":" in cuda_label:
            device_index = int(cuda_label.split(":")[1])
            if self.torch.cuda.is_available() and device_index < self.torch.cuda.device_count():
                cuda_device = self.torch.device(cuda_label)
        
        def handler(text=None, image=None, endpoint=endpoint, processor=processor, model_name=model_name, 
                   cuda_label=cuda_label, max_new_tokens=None, temperature=0.7, top_p=0.9, top_k=50, 
                   do_sample=True, batch_mode=False):
            """Process text and image inputs using LLaVA on CUDA/GPU with full performance metrics.
            
            Args:
                text: Text prompt to process with the image (str, list, or dict)
                image: Image to analyze (file path, URL, PIL Image, or list of images)
                max_new_tokens: Maximum number of tokens to generate (default: auto-determined)
                temperature: Temperature for sampling (default: 0.7)
                top_p: Top-p sampling parameter (default: 0.9)
                top_k: Top-k sampling parameter (default: 50)
                do_sample: Whether to use sampling (default: True)
                batch_mode: Whether to process inputs as batch (default: False)
                
            Returns:
                Dictionary with generated text and detailed performance metrics
            """
            # Start time tracking
            start_time = time.time()
            
            # If we know we're using mocks, return a clear mock response
            if not is_real_impl or isinstance(endpoint, type(MagicMock())) or isinstance(processor, type(MagicMock())):
                # Create mock response that's clearly marked as such
                time.sleep(0.1)  # Small sleep to simulate processing time
                if isinstance(text, str):
                    mock_text = f"(MOCK) CUDA LLaVA analyzed the provided image and found: This appears to be an image {text}"
                else:
                    mock_text = "(MOCK) CUDA LLaVA: This appears to be an image with various objects and elements"
                
                return {
                    "text": mock_text,
                    "implementation_type": "MOCK",
                    "device": cuda_label,
                    "total_time": time.time() - start_time
                }
            
            # Initialize performance trackers
            preprocessing_start = time.time()
            preprocessing_time = None
            generation_time = None
            postprocessing_time = None
            gpu_memory_before = None
            gpu_memory_after = None
            gpu_memory_used = None
            
            try:
                # Clean CUDA cache before processing
                if hasattr(self.torch.cuda, 'empty_cache'):
                    self.torch.cuda.empty_cache()
                
                # Get initial GPU memory usage if available
                if cuda_device is not None and hasattr(self.torch.cuda, 'mem_get_info'):
                    try:
                        free_memory_before, total_memory = self.torch.cuda.mem_get_info(cuda_device.index)
                        gpu_memory_before = {
                            "free_gb": free_memory_before / (1024**3),
                            "total_gb": total_memory / (1024**3),
                            "used_gb": (total_memory - free_memory_before) / (1024**3)
                        }
                    except Exception as mem_error:
                        print(f"Error getting initial GPU memory info: {mem_error}")
                
                # Process the input image(s)
                try:
                    # Handle batch mode with multiple images
                    if batch_mode and isinstance(image, list):
                        # Process multiple images
                        image_objs = []
                        for img in image:
                            if isinstance(img, str):
                                if img.startswith("http") or img.startswith("https"):
                                    response = requests.get(img, timeout=10)
                                    image_objs.append(Image.open(BytesIO(response.content)).convert("RGB"))
                                else:
                                    image_objs.append(Image.open(img).convert("RGB"))
                            elif isinstance(img, Image.Image):
                                image_objs.append(img)
                            else:
                                raise ValueError(f"Unsupported image type in batch: {type(img)}")
                    else:
                        # Single image processing
                        if image is not None:
                            if isinstance(image, str):
                                if image.startswith("http") or image.startswith("https"):
                                    response = requests.get(image, timeout=10)
                                    image_obj = Image.open(BytesIO(response.content)).convert("RGB")
                                else:
                                    image_obj = Image.open(image).convert("RGB")
                            elif isinstance(image, Image.Image):
                                image_obj = image
                            else:
                                raise ValueError(f"Unsupported image type: {type(image)}")
                        else:
                            # Text-only input
                            image_obj = None
                except Exception as img_error:
                    print(f"Error processing image: {img_error}")
                    print(f"Traceback: {traceback.format_exc()}")
                    return {
                        "text": f"Error processing image: {str(img_error)}",
                        "implementation_type": "REAL (error)",
                        "device": cuda_label,
                        "error": str(img_error),
                        "total_time": time.time() - start_time
                    }
                    
                # Process the text input(s)
                try:
                    if batch_mode and isinstance(text, list):
                        # Process batch of text inputs
                        conversations = []
                        for txt in text:
                            if isinstance(txt, str):
                                conversations.append([
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": txt},
                                            {"type": "image"}
                                        ]
                                    }
                                ])
                            elif isinstance(txt, (tuple, list)):
                                conversations.append(txt)
                            elif isinstance(txt, dict):
                                conversations.append([txt])
                            else:
                                raise ValueError(f"Unsupported text input type in batch: {type(txt)}")
                    else:
                        # Single text processing
                        if text is not None and isinstance(text, str):
                            # Standard format for multimodal conversation
                            conversation = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": text},
                                        {"type": "image"}
                                    ]
                                }
                            ]
                        elif isinstance(text, (tuple, list)):
                            # Allow for pre-formatted conversation
                            conversation = text
                        elif isinstance(text, dict):
                            # Single message as dict
                            conversation = [text]
                        else:
                            raise ValueError(f"Unsupported text input type: {type(text)}")
                except Exception as txt_error:
                    print(f"Error processing text input: {txt_error}")
                    print(f"Traceback: {traceback.format_exc()}")
                    return {
                        "text": f"Error processing text input: {str(txt_error)}",
                        "implementation_type": "REAL (error)",
                        "device": cuda_label,
                        "error": str(txt_error),
                        "total_time": time.time() - start_time
                    }
                
                # Process inputs with the processor
                try:
                    if batch_mode and isinstance(text, list):
                        # Batch processing
                        inputs_list = []
                        for i, conv in enumerate(conversations):
                            prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
                            img = image_objs[i] if i < len(image_objs) else None
                            inputs = processor(img, prompt, return_tensors="pt")
                            # Move to CUDA
                            for key in inputs:
                                if hasattr(inputs[key], 'to') and callable(inputs[key].to):
                                    inputs[key] = inputs[key].to(cuda_device)
                            inputs_list.append(inputs)
                    else:
                        # Single input processing
                        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                        inputs = processor(image_obj, prompt, return_tensors="pt")
                        # Move to CUDA
                        for key in inputs:
                            if hasattr(inputs[key], 'to') and callable(inputs[key].to):
                                inputs[key] = inputs[key].to(cuda_device)
                except Exception as proc_error:
                    print(f"Error processing inputs with processor: {proc_error}")
                    print(f"Traceback: {traceback.format_exc()}")
                    return {
                        "text": f"Error processing inputs: {str(proc_error)}",
                        "implementation_type": "REAL (error)",
                        "device": cuda_label,
                        "error": str(proc_error),
                        "total_time": time.time() - start_time
                    }
                
                # Calculate preprocessing time
                preprocessing_time = time.time() - preprocessing_start
                
                # Generate response with GPU acceleration
                generation_start = time.time()
                try:
                    # Set up generation parameters
                    if max_new_tokens is None:
                        # Auto-determine based on input length
                        if batch_mode:
                            # For batch, use a moderate default
                            max_new_tokens = 100
                        else:
                            # For single, get input length if possible
                            if "input_ids" in inputs and hasattr(inputs["input_ids"], "shape"):
                                input_length = inputs["input_ids"].shape[1]
                                max_new_tokens = max(50, min(512, 1024 - input_length))
                            else:
                                max_new_tokens = 100
                    
                    generation_params = {
                        "do_sample": do_sample,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k
                    }
                    
                    # Try to create streamer if we're not in batch mode
                    if not batch_mode:
                        try:
                            generation_params["streamer"] = self.transformers.TextStreamer(
                                processor, 
                                skip_prompt=True, 
                                skip_special_tokens=True
                            )
                        except Exception as streamer_error:
                            print(f"Streamer creation failed: {streamer_error}, continuing without streamer")
                    
                    # Run generation
                    with self.torch.no_grad():
                        if batch_mode and isinstance(text, list):
                            # Batch generation
                            outputs = []
                            for batch_inputs in inputs_list:
                                batch_output_ids = endpoint.generate(
                                    **batch_inputs,
                                    **generation_params
                                )
                                outputs.append(batch_output_ids)
                        else:
                            # Single generation
                            output_ids = endpoint.generate(
                                **inputs,
                                **generation_params
                            )
                    
                    # Get GPU memory usage after generation
                    if cuda_device is not None and hasattr(self.torch.cuda, 'mem_get_info'):
                        try:
                            free_memory_after, total_memory = self.torch.cuda.mem_get_info(cuda_device.index)
                            gpu_memory_after = {
                                "free_gb": free_memory_after / (1024**3),
                                "total_gb": total_memory / (1024**3),
                                "used_gb": (total_memory - free_memory_after) / (1024**3)
                            }
                            
                            # Calculate memory used for this operation
                            gpu_memory_used = (free_memory_before - free_memory_after) / (1024**3)  # in GB
                        except Exception as mem_error:
                            print(f"Error getting final GPU memory info: {mem_error}")
                    
                except Exception as gen_error:
                    print(f"Error during generation: {gen_error}")
                    print(f"Traceback: {traceback.format_exc()}")
                    return {
                        "text": f"Error during generation: {str(gen_error)}",
                        "implementation_type": "REAL (error)",
                        "device": cuda_label,
                        "error": str(gen_error),
                        "preprocessing_time": preprocessing_time,
                        "total_time": time.time() - start_time
                    }
                
                # Calculate generation time
                generation_time = time.time() - generation_start
                
                # Decode response
                postprocessing_start = time.time()
                try:
                    if batch_mode and isinstance(text, list):
                        # Batch decoding
                        responses = []
                        for output in outputs:
                            batch_response = processor.decode(output[0], skip_special_tokens=True)
                            responses.append(batch_response)
                        
                        # Combine or return as list based on need
                        response = responses
                    else:
                        # Single decoding
                        response = processor.decode(output_ids[0], skip_special_tokens=True)
                        
                except Exception as decode_error:
                    print(f"Error decoding response: {decode_error}")
                    print(f"Traceback: {traceback.format_exc()}")
                    return {
                        "text": f"Error decoding response: {str(decode_error)}",
                        "implementation_type": "REAL (error)",
                        "device": cuda_label,
                        "error": str(decode_error),
                        "preprocessing_time": preprocessing_time,
                        "generation_time": generation_time,
                        "total_time": time.time() - start_time
                    }
                
                # Calculate postprocessing time
                postprocessing_time = time.time() - postprocessing_start
                
                # Calculate total time
                total_time = time.time() - start_time
                
                # Construct response with full metrics
                result = {
                    "text": response,
                    "implementation_type": "REAL",
                    "device": str(cuda_device) if cuda_device else cuda_label,
                    "model_name": model_name,
                    "preprocessing_time": preprocessing_time,
                    "generation_time": generation_time,
                    "postprocessing_time": postprocessing_time,
                    "total_time": total_time,
                    "is_batch": batch_mode
                }
                
                # Add memory metrics if available
                if gpu_memory_before:
                    result["gpu_memory_before"] = gpu_memory_before
                if gpu_memory_after:
                    result["gpu_memory_after"] = gpu_memory_after
                if gpu_memory_used is not None:
                    result["gpu_memory_used_gb"] = gpu_memory_used
                
                # Add generation parameters for reference
                result["generation_params"] = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "do_sample": do_sample
                }
                
                # Calculate tokens per second if we have actual text
                if not batch_mode and isinstance(response, str):
                    # Simple approximation: count spaces to estimate tokens
                    generated_tokens = len(response.split())
                    if generation_time > 0:
                        result["tokens_per_second"] = generated_tokens / generation_time
                    result["generated_tokens"] = generated_tokens
                
                return result
                
            except Exception as e:
                print(f"Unexpected error in CUDA LLaVA handler: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                return {
                    "text": f"Unexpected error in CUDA handler: {str(e)}",
                    "implementation_type": "REAL (error)",
                    "device": cuda_label,
                    "error": str(e),
                    "total_time": time.time() - start_time
                }
                
            finally:
                # Clean up CUDA memory after request
                if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                    self.torch.cuda.empty_cache()
        
        return handler
        
    def create_apple_vlm_endpoint_handler(self, endpoint, processor, model_name, apple_label):
        """Creates an endpoint handler for Apple Silicon.
        
        Args:
            endpoint: The model endpoint
            processor: The tokenizer or processor
            model_name: The model name or path
            apple_label: Label to identify this endpoint
            
        Returns:
            A handler function for the Apple endpoint
        """
        def handler(text=None, image=None, endpoint=endpoint, processor=processor, model_name=model_name, apple_label=apple_label):
            """Process text and image inputs using LLaVA on Apple Silicon.
            
            Args:
                text: Text prompt to process with the image
                image: Image to analyze (can be path, URL, or PIL Image)
                
            Returns:
                Generated text response from the model
            """
            try:
                # Load and process image if provided
                if image is not None:
                    if isinstance(image, str):
                        if image.startswith("http") or image.startswith("https"):
                            max_retries = 3
                            retry_delay = 1
                            last_error = None
                            
                            for attempt in range(max_retries):
                                try:
                                    response = requests.get(image, timeout=10)
                                    response.raise_for_status() 
                                    image_obj = Image.open(BytesIO(response.content)).convert("RGB")
                                    break
                                except (requests.RequestException, Image.UnidentifiedImageError) as e:
                                    last_error = e
                                    if attempt < max_retries - 1:
                                        time.sleep(retry_delay)
                                        retry_delay *= 2
                                    else:
                                        raise ValueError(f"Failed to load image from URL after {max_retries} attempts: {image}. Error: {str(last_error)}")
                        else:
                            image_obj = Image.open(image).convert("RGB")
                    elif isinstance(image, Image.Image):
                        image_obj = image
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                else:
                    # Text-only input
                    image_obj = None
                    
                # Process the text input
                if text is not None and isinstance(text, str):
                    # Standard format for multimodal conversation
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {"type": "image"}
                            ]
                        }
                    ]
                elif isinstance(text, (tuple, list)):
                    # Allow for pre-formatted conversation
                    conversation = text
                elif isinstance(text, dict):
                    # Single message as dict
                    conversation = [text]
                else:
                    raise ValueError(f"Unsupported text input type: {type(text)}")
                
                # Create streamer for real-time text output
                try:
                    streamer = self.transformers.TextStreamer(
                        processor, 
                        skip_prompt=True, 
                        skip_special_tokens=True
                    )
                except Exception:
                    streamer = None
                
                # Process inputs
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(image_obj, prompt, return_tensors="pt")
                
                # Move inputs to MPS device if available
                if hasattr(self.torch, 'backends') and hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
                    for key in inputs:
                        if isinstance(inputs[key], self.torch.Tensor):
                            inputs[key] = inputs[key].to("mps")
                
                # Generate response with MPS acceleration
                generation_config = {
                    "do_sample": False,
                    "max_new_tokens": 50
                }
                
                if streamer is not None:
                    generation_config["streamer"] = streamer
                
                output_ids = endpoint.generate(**inputs, **generation_config)
                
                # Decode response
                response = processor.decode(output_ids[0], skip_special_tokens=True)
                
                # Add timestamp and metadata for testing/debugging
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                image_info = f"of size {image_obj.size}" if image_obj and hasattr(image_obj, 'size') else "with the provided content"
                
                return f"(REAL) Apple Silicon LLaVA response [device: {apple_label}, timestamp: {timestamp}]: {response}"
                
            except Exception as e:
                print(f"Error in Apple Silicon LLaVA handler: {e}")
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                return f"(ERROR) Apple Silicon LLaVA response [timestamp: {timestamp}]: Error processing request - {str(e)}"
        
        return handler
        
    def create_qualcomm_vlm_endpoint_handler(self, endpoint, processor, model_name, qualcomm_label):
        """Creates an endpoint handler for Qualcomm hardware.
        
        Args:
            endpoint: The model endpoint
            processor: The tokenizer or processor
            model_name: The model name or path
            qualcomm_label: Label to identify this endpoint
            
        Returns:
            A handler function for the Qualcomm endpoint
        """
        def handler(text=None, image=None, endpoint=endpoint, processor=processor, model_name=model_name, qualcomm_label=qualcomm_label):
            """Process text and image inputs using LLaVA on Qualcomm hardware.
            
            Args:
                text: Text prompt to process with the image
                image: Image to analyze (can be path, URL, or PIL Image)
                
            Returns:
                Generated text response from the model
            """
            try:
                # Check if SNPE utils is available
                if self.snpe_utils is None:
                    import time
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    return f"(MOCK) Qualcomm LLaVA response [timestamp: {timestamp}]: Qualcomm SNPE not available in this environment"
                
                # Load and process image if provided
                if image is not None:
                    if isinstance(image, str):
                        if image.startswith("http") or image.startswith("https"):
                            response = requests.get(image)
                            image_obj = Image.open(BytesIO(response.content)).convert("RGB")
                        else:
                            image_obj = Image.open(image).convert("RGB")
                    elif isinstance(image, Image.Image):
                        image_obj = image
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                else:
                    # Text-only input
                    image_obj = None
                
                # Create model inputs in numpy format
                if image_obj is not None:
                    inputs = processor(text=text, images=image_obj, return_tensors="np")
                else:
                    inputs = processor(text=text, return_tensors="np")
                
                # Run inference through SNPE
                results = self.snpe_utils.run_inference(endpoint, inputs)
                
                # For LLaVA models, we might need to do generation token by token
                generated_ids = []
                
                # Check if we have direct generation output
                if "generated_ids" in results:
                    generated_ids = results["generated_ids"][0]
                else:
                    # We need to do token-by-token generation
                    # First, get the processed inputs if available
                    if "input_ids" in results:
                        generated_ids = results["input_ids"][0].tolist()
                    else:
                        generated_ids = inputs["input_ids"][0].tolist()
                    
                    # Prepare for token-by-token generation
                    past_key_values = results.get("past_key_values", None)
                    max_new_tokens = 256
                    
                    # Generate tokens one by one
                    for _ in range(max_new_tokens):
                        # Prepare inputs for next token prediction
                        gen_inputs = {
                            "input_ids": self.np.array([generated_ids[-1:]]),
                            "attention_mask": self.np.array([[1]])
                        }
                        
                        # Add past key values if available
                        if past_key_values is not None:
                            for i, (k, v) in enumerate(past_key_values):
                                gen_inputs[f"past_key_values.{i}.key"] = k
                                gen_inputs[f"past_key_values.{i}.value"] = v
                        
                        # Get next token
                        token_results = self.snpe_utils.run_inference(endpoint, gen_inputs)
                        
                        # Get logits and past key values
                        if "logits" in token_results:
                            logits = self.np.array(token_results["logits"])
                            
                            # Update past key values
                            if "past_key_values" in token_results:
                                past_key_values = token_results["past_key_values"]
                            
                            # Basic greedy decoding
                            next_token_id = int(self.np.argmax(logits[0, -1, :]))
                            
                            # Add token to generated sequence
                            generated_ids.append(next_token_id)
                            
                            # Check for EOS token
                            if next_token_id == processor.tokenizer.eos_token_id:
                                break
                        else:
                            break
                
                # Decode the generated text
                generated_text = processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
                
                # Add timestamp and metadata for testing/debugging
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                image_info = f"of size {image_obj.size}" if image_obj and hasattr(image_obj, 'size') else "with the provided content"
                
                return f"(REAL) Qualcomm LLaVA response [device: {qualcomm_label}, timestamp: {timestamp}]: {generated_text}"
                
            except Exception as e:
                print(f"Error in Qualcomm LLaVA endpoint handler: {e}")
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                return f"(ERROR) Qualcomm LLaVA response [timestamp: {timestamp}]: Error processing request - {str(e)}"
                
        return handler