import requests
from PIL import Image
from io import BytesIO
import asyncio
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
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.init = self.init
        self.coreml_utils = None

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
            
            return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Apple Silicon LLaVA-Next model: {e}")
            return None, None, None, None, 0
            
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