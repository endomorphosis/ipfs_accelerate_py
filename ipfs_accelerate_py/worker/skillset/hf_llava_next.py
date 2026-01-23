import requests
from PIL import Image
from io import BytesIO
import anyio
from pathlib import Path
import json
import time
import os
import tempfile
import numpy as np
import torch
from torchvision.transforms import InterpolationMode, Compose, Lambda, Resize, ToTensor, Normalize

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = Compose([
        Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def streamer(subword: str) -> bool:
    """
    Stream tokens as they are generated
    
    Args:
        subword: The subword/token to stream
        
    Returns:
        Boolean indicating whether to continue streaming
    """
    print(subword, end="", flush=True)
    return True

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Find the closest aspect ratio to the target to minimize distortion
    
    Args:
        aspect_ratio: Original aspect ratio
        target_ratios: List of target aspect ratios to choose from
        width: Original image width
        height: Original image height
        image_size: Target size for the image
        
    Returns:
        Tuple of (width, height) for the resized image
    """
    closest_ratio = min(target_ratios, key=lambda r: abs(r - aspect_ratio))
    
    if closest_ratio > 1:  # width > height
        new_width = image_size
        new_height = int(image_size / closest_ratio)
    else:  # height > width
        new_height = image_size
        new_width = int(image_size * closest_ratio)
        
    return new_width, new_height

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Dynamically preprocess image based on its properties
    
    Args:
        image: PIL Image to process
        min_num: Minimum number of image patches
        max_num: Maximum number of image patches
        image_size: Target image size
        use_thumbnail: Whether to use image thumbnail
        
    Returns:
        Processed image ready for model input
    """
    width, height = image.size
    aspect_ratio = width / height
    
    # Common target aspect ratios
    target_ratios = [0.5, 0.75, 1.0, 1.33, 1.5, 2.0]
    
    new_width, new_height = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    )
    
    # Resize the image
    if use_thumbnail:
        image = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        image = image.resize((new_width, new_height), Image.BICUBIC)
    
    # Additional preprocessing for model input would go here
    transform = build_transform(image_size)
    tensor = transform(image)
    
    return tensor

def load_image(image_file):
    """
    Load image from file path or URL
    
    Args:
        image_file: Path or URL to image
        
    Returns:
        PIL Image object
    """
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        import requests
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_image_tensor(image_file):
    """
    Load image and convert directly to tensor
    
    Args:
        image_file: Path or URL to image
        
    Returns:
        Image as tensor ready for model input
    """
    image = load_image(image_file)
    transform = build_transform(448)  # Default size
    return transform(image).unsqueeze(0)  # Add batch dimension

class hf_llava_next:
    """HuggingFace LLaVA-Next (Large Language and Vision Assistant, Next Generation) implementation.
    
    This class provides standardized interfaces for working with LLaVA-Next models
    across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm).
    
    LLaVA-Next extends LLaVA with improved multimodal capabilities, including
    multi-image understanding, better visual reasoning, and enhanced vision-language alignment.
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the LLaVA-Next model.
        
        Args:
            resources (dict): Dictionary of shared resources (torch, transformers, etc.)
            metadata (dict): Configuration metadata
        """
        self.resources = resources
        self.metadata = metadata
        
        # Handler creation methods
        self.create_cpu_multimodal_endpoint_handler = self.create_cpu_multimodal_endpoint_handler if hasattr(self, 'create_cpu_multimodal_endpoint_handler') else None
        self.create_cuda_multimodal_endpoint_handler = self.create_cuda_multimodal_endpoint_handler if hasattr(self, 'create_cuda_multimodal_endpoint_handler') else None
        self.create_openvino_multimodal_endpoint_handler = self.create_openvino_multimodal_endpoint_handler if hasattr(self, 'create_openvino_multimodal_endpoint_handler') else None
        self.create_apple_multimodal_endpoint_handler = self.create_apple_multimodal_endpoint_handler if hasattr(self, 'create_apple_multimodal_endpoint_handler') else None
        self.create_qualcomm_multimodal_endpoint_handler = self.create_qualcomm_multimodal_endpoint_handler if hasattr(self, 'create_qualcomm_multimodal_endpoint_handler') else None
        
        # Initialization methods
        self.init = self.init
        
        # Hardware-specific utilities
        self.coreml_utils = None  # Apple CoreML utils

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
            
    def get_cuda_device(self, device_label="cuda:0"):
        """
        Get a valid CUDA device from label with proper error handling
        
        Args:
            device_label: String like "cuda:0" or "cuda:1"
            
        Returns:
            torch.device: CUDA device object, or None if not available
        """
        try:
            # Make sure torch is initialized
            self.init()
            import torch
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                print("CUDA is not available on this system")
                return None
                
            # Parse device parts
            parts = device_label.split(":")
            device_type = parts[0].lower()
            device_index = int(parts[1]) if len(parts) > 1 else 0
            
            # Validate device type
            if device_type != "cuda":
                print(f"Warning: Device type '{device_type}' is not CUDA, defaulting to 'cuda'")
                device_type = "cuda"
                
            # Validate device index
            cuda_device_count = torch.cuda.device_count()
            if device_index >= cuda_device_count:
                print(f"Warning: CUDA device index {device_index} out of range (0-{cuda_device_count-1}), using device 0")
                device_index = 0
                
            # Create device object
            device = torch.device(f"{device_type}:{device_index}")
            
            # Print device info
            device_name = torch.cuda.get_device_name(device_index)
            print(f"Using CUDA device: {device_name} (index {device_index})")
            
            return device
        except Exception as e:
            print(f"Error setting up CUDA device: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def init_cuda(self, model_name, model_type, device_label="cuda:0", **kwargs):
        """
        Initialize LLaVA-Next model with CUDA support
        
        Args:
            model_name: Name or path of the model to load
            model_type: Type of model (typically "vision2seq")
            device_label: CUDA device to use (e.g., "cuda:0", "cuda:1")
            **kwargs: Additional model-specific parameters
            
        Returns:
            tuple: (endpoint, processor, handler, queue, batch_size)
        """
        self.init()
        import torch  # Direct import to ensure availability
        
        # Validate CUDA availability
        device = self.get_cuda_device(device_label)
        if device is None:
            print("CUDA initialization failed: no valid device available")
            return None, None, None, None, 0
            
        try:
            # Clear CUDA cache before loading model
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                
            print(f"Loading LLaVA-Next model '{model_name}' on {device}...")
            
            # Load processor and model configuration
            try:
                processor = self.transformers.AutoProcessor.from_pretrained(model_name)
                model_config = self.transformers.AutoConfig.from_pretrained(model_name)
                print(f"Loaded processor and config for {model_name}")
            except Exception as e:
                print(f"Error loading processor or config: {e}")
                return None, None, None, None, 0
                
            # Determine the correct model class to use based on config
            try:
                model_class = self.transformers.AutoModelForVision2Seq
                if hasattr(self.transformers, "LlavaForConditionalGeneration"):
                    if hasattr(model_config, "model_type") and model_config.model_type == "llava":
                        model_class = self.transformers.LlavaForConditionalGeneration
                        print("Using LlavaForConditionalGeneration model class")
                    else:
                        print("Using AutoModelForVision2Seq model class")
                else:
                    print("LlavaForConditionalGeneration not available, using AutoModelForVision2Seq")
                    
                # Load model with optimizations for CUDA
                print(f"Loading model on {device} with half precision...")
                model = model_class.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Use FP16 precision for memory efficiency
                    device_map=device.index,    # Use the selected device
                    **kwargs
                )
                
                # Optimize the model for inference
                model.eval()  # Set to evaluation mode
                
                # Report memory usage
                if hasattr(torch.cuda, 'memory_allocated'):
                    allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB
                    print(f"CUDA memory allocated: {allocated_memory:.2f} GB")
                    
                # Create the endpoint handler with the model
                handler = self.create_cuda_multimodal_endpoint_handler(
                    model, 
                    processor, 
                    model_name, 
                    device_label
                )
                
                # Use a queue for batch processing
                batch_size = 1  # LLaVA models are memory-intensive, start with small batch
                queue = # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(32)
                
                return model, processor, handler, queue, batch_size
                
            except Exception as e:
                print(f"Error loading model: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                
                # Clear cache after failed initialization
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    
                return None, None, None, None, 0
                
        except Exception as e:
            print(f"Error in CUDA initialization: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None, None, None, None, 0
            
    def init_apple(self, model, device, apple_label):
        """Initialize LLaVA-Next model for Apple Silicon hardware."""
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
            processor = self.transformers.AutoProcessor.from_pretrained(model)
            
            # Convert model path to be compatible with CoreML
            model_name = model.replace("/", "--")
            mlmodel_path = f"~/coreml_models/{model_name}_llava_next.mlpackage"
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
            
            return endpoint, processor, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Apple Silicon LLaVA-Next model: {e}")
            return None, None, None, None, 0
            
    def create_cuda_multimodal_endpoint_handler(self, model, processor, model_name, device_label):
        """
        Creates a CUDA-accelerated handler for LLaVA-Next multimodal processing
        
        Args:
            model: The PyTorch LLaVA-Next model
            processor: The model processor for inputs
            model_name: Name of the model being used
            device_label: CUDA device label (e.g., "cuda:0")
            
        Returns:
            function: Handler function that processes inputs using the CUDA model
        """
        # Get the actual device object
        device = self.get_cuda_device(device_label)
        if device is None:
            print(f"Warning: Invalid CUDA device {device_label}, using model's device")
            if hasattr(model, 'device'):
                device = model.device
            else:
                # Last resort - try to create a default CUDA device
                try:
                    import torch
                    device = torch.device("cuda:0")
                except:
                    import torch
                    device = torch.device("cpu")
                    
        # Create the handler function with captured parameters
        def handler(text=None, image=None, max_new_tokens=256, temperature=0.7, top_p=0.9):
            try:
                import torch  # Import directly inside handler for access
                start_time = time.time()
                
                # Determine input type and process accordingly
                if image is None and text is None:
                    print("Error: Both text and image are None")
                    return {
                        "text": "Error: No input provided",
                        "implementation_type": "REAL", 
                        "platform": "CUDA",
                        "error": "Both text and image inputs are None"
                    }
                
                # Preprocess images
                if image is not None:
                    # Check if it's a list of images
                    if isinstance(image, list):
                        # Handle multiple images
                        try:
                            if all(isinstance(img, str) for img in image):
                                # Load images from paths/URLs
                                pil_images = [load_image(img_path) for img_path in image]
                            else:
                                # Assume they're already PIL images
                                pil_images = image
                                
                            # Track preprocessing time
                            preprocess_start = time.time()
                            
                            # Process inputs with the processor
                            inputs = processor(
                                text=[text] * len(pil_images) if text is not None else [""] * len(pil_images),
                                images=pil_images,
                                return_tensors="pt",
                                padding=True
                            )
                            
                            preprocess_time = time.time() - preprocess_start
                            print(f"Preprocessed {len(pil_images)} images in {preprocess_time:.2f}s")
                            
                        except Exception as e:
                            print(f"Error processing multiple images: {e}")
                            import traceback
                            print(f"Traceback: {traceback.format_exc()}")
                            return {
                                "text": f"Error processing multiple images: {str(e)}",
                                "implementation_type": "REAL", 
                                "platform": "CUDA",
                                "error": str(e)
                            }
                    else:
                        # Single image case
                        try:
                            if isinstance(image, str):
                                # Load image from path/URL
                                pil_image = load_image(image)
                            else:
                                # Assume it's already a PIL image
                                pil_image = image
                                
                            # Track preprocessing time
                            preprocess_start = time.time()
                            
                            # Process inputs with the processor
                            inputs = processor(
                                text=text if text is not None else "",
                                images=pil_image,
                                return_tensors="pt",
                                padding=True
                            )
                            
                            preprocess_time = time.time() - preprocess_start
                            print(f"Preprocessed image in {preprocess_time:.2f}s")
                            
                        except Exception as e:
                            print(f"Error processing image: {e}")
                            import traceback
                            print(f"Traceback: {traceback.format_exc()}")
                            return {
                                "text": f"Error processing image: {str(e)}",
                                "implementation_type": "REAL", 
                                "platform": "CUDA",
                                "error": str(e)
                            }
                else:
                    # Text-only input
                    try:
                        # Track preprocessing time
                        preprocess_start = time.time()
                        
                        # Process text input with the processor
                        inputs = processor(
                            text=text,
                            return_tensors="pt",
                            padding=True
                        )
                        
                        preprocess_time = time.time() - preprocess_start
                        print(f"Preprocessed text in {preprocess_time:.2f}s")
                        
                    except Exception as e:
                        print(f"Error processing text: {e}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        return {
                            "text": f"Error processing text: {str(e)}",
                            "implementation_type": "REAL", 
                            "platform": "CUDA",
                            "error": str(e)
                        }
                
                try:
                    # Move inputs to the correct device
                    for key in inputs:
                        if hasattr(inputs[key], 'to') and callable(inputs[key].to):
                            inputs[key] = inputs[key].to(device)
                    
                    # Report memory usage before generation
                    if hasattr(torch.cuda, 'memory_allocated'):
                        before_memory = torch.cuda.memory_allocated(device) / (1024**2)  # MB
                        print(f"CUDA memory before generation: {before_memory:.2f} MB")
                    
                    # Track generation time
                    generate_start = time.time()
                    
                    # Use no_grad for inference to reduce memory usage
                    with torch.no_grad():
                        # Configure generation parameters
                        generation_config = {
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                            "do_sample": temperature > 0,
                        }
                        
                        # Generate output
                        outputs = model.generate(
                            **inputs,
                            **generation_config
                        )
                        
                    generate_time = time.time() - generate_start
                    print(f"Generated response in {generate_time:.2f}s")
                    
                    # Report memory usage after generation
                    if hasattr(torch.cuda, 'memory_allocated'):
                        after_memory = torch.cuda.memory_allocated(device) / (1024**2)  # MB
                        print(f"CUDA memory after generation: {after_memory:.2f} MB")
                    
                    # Decode outputs
                    responses = processor.batch_decode(
                        outputs,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Calculate total time
                    total_time = time.time() - start_time
                    
                    # Process response to trim instruction prefix if needed
                    if responses and len(responses) > 0:
                        response_text = responses[0]
                        
                        # Some models include the prompt in the response - remove it if detected
                        if text is not None and response_text.startswith(text):
                            response_text = response_text[len(text):].strip()
                    else:
                        response_text = "No response generated"
                    
                    # Prepare result with metrics
                    result = {
                        "text": response_text,
                        "implementation_type": "REAL",
                        "platform": "CUDA",
                        "device": str(device),
                        "model": model_name,
                        "timing": {
                            "preprocess_time": preprocess_time,
                            "generate_time": generate_time,
                            "total_time": total_time
                        },
                        "metrics": {
                            "tokens_per_second": max_new_tokens / generate_time if generate_time > 0 else 0,
                        }
                    }
                    
                    # Add memory usage if available
                    if hasattr(torch.cuda, 'memory_allocated'):
                        result["metrics"]["memory_used_mb"] = after_memory
                        result["metrics"]["memory_change_mb"] = after_memory - before_memory
                    
                    # Clean up CUDA cache
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        
                    return result
                    
                except Exception as e:
                    print(f"Error in CUDA generation: {e}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    
                    # Try to clean up CUDA cache even after error
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    
                    # Provide error information
                    return {
                        "text": f"Error during CUDA generation: {str(e)}",
                        "implementation_type": "REAL", 
                        "platform": "CUDA",
                        "error": str(e)
                    }
                    
            except Exception as e:
                # Catch-all for any other errors
                print(f"Unexpected error in CUDA handler: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                
                # Try to clean up CUDA cache even after error
                try:
                    import torch
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                except:
                    pass
                
                return {
                    "text": f"Unexpected error in CUDA handler: {str(e)}",
                    "implementation_type": "REAL", 
                    "platform": "CUDA",
                    "error": str(e)
                }
                
        return handler
    
    def create_apple_multimodal_endpoint_handler(self, endpoint, processor, model_name, apple_label):
        """Creates an Apple Silicon optimized handler for LLaVA-Next multimodal processing."""
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
                print(f"Error in Apple Silicon LLaVA-Next handler: {e}")
                return None
                
        return handler