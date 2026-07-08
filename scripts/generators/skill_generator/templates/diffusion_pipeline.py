#!/usr/bin/env python3
"""
Diffusion Pipeline Template for IPFS Accelerate Python.

This module implements a pipeline template for diffusion models like Stable Diffusion,
Kandinsky, and SAM (Segment Anything Model). It handles diffusion-specific parameters
and image generation/manipulation tasks.
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipelineTemplate


class DiffusionPipelineTemplate(BasePipelineTemplate):
    """Template for diffusion-based model pipelines."""
    
    def __init__(self):
        """Initialize the diffusion pipeline template."""
        super().__init__()
        self.pipeline_type = "diffusion"
        self.input_type = "multimodal"
        self.output_type = "image"
        self.requires_preprocessing = True
        self.requires_postprocessing = True
        self.supports_batching = True
        self.max_batch_size = 1  # Most diffusion models work best with batch size 1
    
    def get_import_statements(self) -> str:
        """Get diffusion pipeline import statements."""
        return """
# Diffusion pipeline imports
import os
import json
import base64
import numpy as np
from typing import List, Dict, Union, Any, Optional, Tuple
from PIL import Image
import io
import tempfile
"""
    
    def get_preprocessing_code(self, task_type: str) -> str:
        """Get diffusion preprocessing code for specific task types."""
        if task_type == "image_generation":
            return """
# Preprocess for image generation (text-to-image)
# Handle different input formats for the prompt
if isinstance(text, dict):
    # Advanced input with parameters
    prompt = text.get("prompt", "")
    negative_prompt = text.get("negative_prompt", None)
    width = text.get("width", 512)
    height = text.get("height", 512)
    num_inference_steps = text.get("num_inference_steps", 50)
    guidance_scale = text.get("guidance_scale", 7.5)
    seed = text.get("seed", None)
elif isinstance(text, str):
    # Simple prompt
    prompt = text
    negative_prompt = None
    width = 512
    height = 512
    num_inference_steps = 50
    guidance_scale = 7.5
    seed = None
else:
    # Default
    prompt = "A photo of a beautiful landscape"
    negative_prompt = None
    width = 512
    height = 512
    num_inference_steps = 50
    guidance_scale = 7.5
    seed = None

# Setup diffusion parameters dictionary
generation_params = {
    "prompt": prompt,
    "height": height,
    "width": width,
    "num_inference_steps": num_inference_steps,
    "guidance_scale": guidance_scale,
}

# Add optional parameters if provided
if negative_prompt is not None:
    generation_params["negative_prompt"] = negative_prompt

# Set random seed if provided for reproducibility
if seed is not None:
    generator = self.torch.Generator(device=device)
    generator.manual_seed(seed)
    generation_params["generator"] = generator

# Extract any additional parameters from kwargs
for param_name, param_value in kwargs.items():
    if param_name not in generation_params:
        generation_params[param_name] = param_value
"""
        elif task_type == "image_to_image":
            return """
# Preprocess for image-to-image generation
# Parse input
init_image = None
strength = 0.75
prompt = ""

if isinstance(text, dict):
    # Advanced input with parameters
    if "image" in text:
        init_image = text["image"]
    if "prompt" in text:
        prompt = text["prompt"]
    if "strength" in text:
        strength = float(text["strength"])
    
    # Get other parameters
    negative_prompt = text.get("negative_prompt", None)
    num_inference_steps = text.get("num_inference_steps", 50)
    guidance_scale = text.get("guidance_scale", 7.5)
    seed = text.get("seed", None)
elif isinstance(text, tuple) and len(text) >= 2:
    # Tuple of (image, prompt)
    init_image = text[0]
    prompt = text[1]
    
    # Optional strength parameter
    if len(text) >= 3 and isinstance(text[2], (int, float)):
        strength = float(text[2])
    
    negative_prompt = None
    num_inference_steps = 50
    guidance_scale = 7.5
    seed = None
elif isinstance(text, str) and os.path.exists(text):
    # Path to an image file (use default prompt)
    init_image = text
    prompt = "Enhance this image"
    negative_prompt = None
    num_inference_steps = 50
    guidance_scale = 7.5
    seed = None
else:
    # Default fallback
    prompt = "Please provide an image to transform"
    negative_prompt = None
    num_inference_steps = 50
    guidance_scale = 7.5
    seed = None

# Process the initial image if provided
if init_image is not None:
    # Handle different image input types
    if isinstance(init_image, str):
        if os.path.exists(init_image):
            # It's a file path
            init_image = Image.open(init_image).convert('RGB')
        elif init_image.startswith(('data:image', 'http://', 'https://')):
            # It's a URL or data URI
            if init_image.startswith('data:image'):
                # Base64 encoded image
                image_data = init_image.split(',')[1]
                init_image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
            else:
                # URL (would require requests)
                raise ValueError("URL images not implemented yet")
        else:
            # Try to decode as base64
            try:
                init_image = Image.open(io.BytesIO(base64.b64decode(init_image))).convert('RGB')
            except:
                raise ValueError(f"Could not parse input as image: {init_image[:30]}...")
    elif isinstance(init_image, Image.Image):
        # Already a PIL Image
        pass
    elif isinstance(init_image, bytes):
        # Raw bytes
        init_image = Image.open(io.BytesIO(init_image)).convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(init_image)}")

# Setup diffusion parameters dictionary
generation_params = {
    "prompt": prompt,
    "image": init_image,
    "strength": strength,
    "num_inference_steps": num_inference_steps,
    "guidance_scale": guidance_scale,
}

# Add optional parameters if provided
if negative_prompt is not None:
    generation_params["negative_prompt"] = negative_prompt

# Set random seed if provided for reproducibility
if seed is not None:
    generator = self.torch.Generator(device=device)
    generator.manual_seed(seed)
    generation_params["generator"] = generator

# Extract any additional parameters from kwargs
for param_name, param_value in kwargs.items():
    if param_name not in generation_params:
        generation_params[param_name] = param_value
"""
        elif task_type == "inpainting":
            return """
# Preprocess for inpainting
# Parse input
init_image = None
mask_image = None
prompt = ""

if isinstance(text, dict):
    # Advanced input with parameters
    if "image" in text:
        init_image = text["image"]
    if "mask" in text:
        mask_image = text["mask"]
    if "prompt" in text:
        prompt = text["prompt"]
    
    # Get other parameters
    negative_prompt = text.get("negative_prompt", None)
    num_inference_steps = text.get("num_inference_steps", 50)
    guidance_scale = text.get("guidance_scale", 7.5)
    seed = text.get("seed", None)
elif isinstance(text, tuple) and len(text) >= 3:
    # Tuple of (image, mask, prompt)
    init_image = text[0]
    mask_image = text[1]
    prompt = text[2]
    
    negative_prompt = None
    num_inference_steps = 50
    guidance_scale = 7.5
    seed = None
else:
    # Default fallback (can't really do inpainting without image and mask)
    prompt = "Please provide an image and mask for inpainting"
    negative_prompt = None
    num_inference_steps = 50
    guidance_scale = 7.5
    seed = None

# Process the initial image if provided
if init_image is not None:
    # Handle different image input types
    if isinstance(init_image, str):
        if os.path.exists(init_image):
            # It's a file path
            init_image = Image.open(init_image).convert('RGB')
        elif init_image.startswith(('data:image', 'http://', 'https://')):
            # It's a URL or data URI
            if init_image.startswith('data:image'):
                # Base64 encoded image
                image_data = init_image.split(',')[1]
                init_image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
            else:
                # URL (would require requests)
                raise ValueError("URL images not implemented yet")
        else:
            # Try to decode as base64
            try:
                init_image = Image.open(io.BytesIO(base64.b64decode(init_image))).convert('RGB')
            except:
                raise ValueError(f"Could not parse input as image: {init_image[:30]}...")
    elif isinstance(init_image, Image.Image):
        # Already a PIL Image
        pass
    elif isinstance(init_image, bytes):
        # Raw bytes
        init_image = Image.open(io.BytesIO(init_image)).convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(init_image)}")

# Process the mask image if provided
if mask_image is not None:
    # Handle different mask input types
    if isinstance(mask_image, str):
        if os.path.exists(mask_image):
            # It's a file path
            mask_image = Image.open(mask_image).convert('L')  # Convert to grayscale
        elif mask_image.startswith(('data:image', 'http://', 'https://')):
            # It's a URL or data URI
            if mask_image.startswith('data:image'):
                # Base64 encoded image
                mask_data = mask_image.split(',')[1]
                mask_image = Image.open(io.BytesIO(base64.b64decode(mask_data))).convert('L')
            else:
                # URL (would require requests)
                raise ValueError("URL images not implemented yet")
        else:
            # Try to decode as base64
            try:
                mask_image = Image.open(io.BytesIO(base64.b64decode(mask_image))).convert('L')
            except:
                raise ValueError(f"Could not parse mask as image: {mask_image[:30]}...")
    elif isinstance(mask_image, Image.Image):
        # Already a PIL Image, convert to grayscale
        mask_image = mask_image.convert('L')
    elif isinstance(mask_image, bytes):
        # Raw bytes
        mask_image = Image.open(io.BytesIO(mask_image)).convert('L')
    else:
        raise ValueError(f"Unsupported mask type: {type(mask_image)}")

# Setup diffusion parameters dictionary
generation_params = {
    "prompt": prompt,
    "image": init_image,
    "mask_image": mask_image,
    "num_inference_steps": num_inference_steps,
    "guidance_scale": guidance_scale,
}

# Add optional parameters if provided
if negative_prompt is not None:
    generation_params["negative_prompt"] = negative_prompt

# Set random seed if provided for reproducibility
if seed is not None:
    generator = self.torch.Generator(device=device)
    generator.manual_seed(seed)
    generation_params["generator"] = generator

# Extract any additional parameters from kwargs
for param_name, param_value in kwargs.items():
    if param_name not in generation_params:
        generation_params[param_name] = param_value
"""
        elif task_type == "image_segmentation":
            return """
# Preprocess for image segmentation (SAM-like)
# Parse input
input_image = None
points = []
point_labels = []
boxes = []
multimask_output = True

if isinstance(text, dict):
    # Advanced input with parameters
    if "image" in text:
        input_image = text["image"]
    if "points" in text:
        points = text["points"]  # List of [x, y] coordinates
    if "point_labels" in text:
        point_labels = text["point_labels"]  # 1 for foreground, 0 for background
    if "boxes" in text:
        boxes = text["boxes"]  # [x1, y1, x2, y2] format
    if "multimask_output" in text:
        multimask_output = text["multimask_output"]
elif isinstance(text, tuple) and len(text) >= 1:
    # Tuple with at least an image
    input_image = text[0]
    
    # Optional points
    if len(text) >= 2:
        points = text[1]
    
    # Optional point labels
    if len(text) >= 3:
        point_labels = text[2]
elif isinstance(text, str) and os.path.exists(text):
    # Path to an image file
    input_image = text
else:
    raise ValueError("Image input required for segmentation")

# Process the input image if provided
if input_image is not None:
    # Handle different image input types
    if isinstance(input_image, str):
        if os.path.exists(input_image):
            # It's a file path
            input_image = Image.open(input_image).convert('RGB')
        elif input_image.startswith(('data:image', 'http://', 'https://')):
            # It's a URL or data URI
            if input_image.startswith('data:image'):
                # Base64 encoded image
                image_data = input_image.split(',')[1]
                input_image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
            else:
                # URL (would require requests)
                raise ValueError("URL images not implemented yet")
        else:
            # Try to decode as base64
            try:
                input_image = Image.open(io.BytesIO(base64.b64decode(input_image))).convert('RGB')
            except:
                raise ValueError(f"Could not parse input as image: {input_image[:30]}...")
    elif isinstance(input_image, Image.Image):
        # Already a PIL Image
        pass
    elif isinstance(input_image, bytes):
        # Raw bytes
        input_image = Image.open(io.BytesIO(input_image)).convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(input_image)}")

# Convert PIL image to numpy array if needed
input_array = np.array(input_image)

# Setup segmentation parameters dictionary
segmentation_params = {
    "image": input_array,
}

# Add optional parameters if provided
if points and len(points) > 0:
    segmentation_params["point_coords"] = np.array(points)
    
    # If point_labels is provided, use it, otherwise default to all foreground points
    if point_labels and len(point_labels) == len(points):
        segmentation_params["point_labels"] = np.array(point_labels)
    else:
        segmentation_params["point_labels"] = np.ones(len(points))

if boxes and len(boxes) > 0:
    # Convert to the format expected by the model
    if len(boxes) == 4 and all(isinstance(coord, (int, float)) for coord in boxes):
        # Single box as [x1, y1, x2, y2]
        segmentation_params["box"] = np.array(boxes)
    else:
        # Multiple boxes or different format
        segmentation_params["boxes"] = np.array(boxes)

segmentation_params["multimask_output"] = multimask_output

# Extract any additional parameters from kwargs
for param_name, param_value in kwargs.items():
    if param_name not in segmentation_params:
        segmentation_params[param_name] = param_value
"""
        else:
            # Default preprocessing for other diffusion tasks
            return """
# Default preprocessing for diffusion models
# Parse input
if isinstance(text, dict):
    # Use parameters from the dictionary
    params = text
elif isinstance(text, str):
    # Simple prompt
    params = {"prompt": text}
else:
    # Default
    params = {"prompt": "A sample prompt for diffusion model"}

# Extract diffusion parameters
prompt = params.get("prompt", "")
guidance_scale = params.get("guidance_scale", 7.5)
num_inference_steps = params.get("num_inference_steps", 50)
width = params.get("width", 512)
height = params.get("height", 512)
seed = params.get("seed", None)

# Setup parameters dictionary
generation_params = {
    "prompt": prompt,
    "guidance_scale": guidance_scale,
    "num_inference_steps": num_inference_steps,
    "width": width, 
    "height": height
}

# Set random seed if provided
if seed is not None:
    generator = self.torch.Generator(device=device)
    generator.manual_seed(seed)
    generation_params["generator"] = generator

# Extract any additional parameters from kwargs
for param_name, param_value in kwargs.items():
    if param_name not in generation_params:
        generation_params[param_name] = param_value
"""
    
    def get_postprocessing_code(self, task_type: str) -> str:
        """Get diffusion postprocessing code for specific task types."""
        if task_type == "image_generation" or task_type == "image_to_image" or task_type == "inpainting":
            return """
# Process outputs from diffusion model
results = {}

# Run inference
with self.torch.no_grad():
    output = endpoint(**generation_params)

# Extract generated images
if hasattr(output, "images"):
    # Most diffusion pipelines return images
    generated_images = output.images
    results["images"] = generated_images
elif hasattr(output, "image"):
    # Some pipelines return a single image
    generated_images = [output.image]
    results["images"] = generated_images
else:
    # Fallback for other return types
    generated_images = []
    results["raw_output"] = str(output)

# Encode images to base64 for response
if generated_images:
    base64_images = []
    for img in generated_images:
        if isinstance(img, np.ndarray):
            # Convert numpy array to PIL Image
            img = Image.fromarray(np.uint8(img * 255) if img.max() <= 1.0 else np.uint8(img))
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(img_str)
    
    results["base64_images"] = base64_images

# Add parameters used for generation
results["parameters"] = {
    "prompt": generation_params.get("prompt", ""),
    "guidance_scale": generation_params.get("guidance_scale", 7.5),
    "num_inference_steps": generation_params.get("num_inference_steps", 50),
}

# Add negative prompt if it was used
if "negative_prompt" in generation_params:
    results["parameters"]["negative_prompt"] = generation_params["negative_prompt"]

# If seed was used, add it to response
if "generator" in generation_params:
    results["parameters"]["seed"] = generation_params["generator"].initial_seed()
"""
        elif task_type == "image_segmentation":
            return """
# Process outputs from segmentation model
results = {}

# Run inference
with self.torch.no_grad():
    output = endpoint(**segmentation_params)

# Extract segmentation masks
if hasattr(output, "masks"):
    # Standard format for segmentation models
    masks = output.masks.numpy()
    results["masks"] = masks
    results["mask_count"] = len(masks)
elif isinstance(output, tuple) and len(output) >= 1:
    # SAM-like format (masks, scores, logits)
    masks = output[0].numpy()
    results["masks"] = masks
    results["mask_count"] = len(masks)
    
    if len(output) >= 2:
        scores = output[1].numpy().tolist()
        results["scores"] = scores
    
    if len(output) >= 3:
        logits = output[2].numpy()
        results["logits"] = True  # Just indicate presence, tensor too large
else:
    # Fallback
    results["raw_output"] = str(output)

# Convert masks to visualizable format if available
if "masks" in results:
    # Create colored mask visualizations
    mask_images = []
    
    for i, mask in enumerate(results["masks"]):
        # Convert binary mask to RGB
        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        # Use different colors for different masks
        color = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
        ][i % 6]
        
        # Apply color to mask
        mask_rgb[mask] = color
        
        # Convert to PIL and encode
        mask_image = Image.fromarray(mask_rgb)
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        mask_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        mask_images.append(mask_str)
    
    results["base64_masks"] = mask_images

# Add parameters used for segmentation
results["parameters"] = {
    "multimask_output": segmentation_params.get("multimask_output", True),
}

# Add points if they were used
if "point_coords" in segmentation_params:
    results["parameters"]["points"] = segmentation_params["point_coords"].tolist()
    results["parameters"]["point_labels"] = segmentation_params["point_labels"].tolist()

# Add box if it was used
if "box" in segmentation_params:
    results["parameters"]["box"] = segmentation_params["box"].tolist()
elif "boxes" in segmentation_params:
    results["parameters"]["boxes"] = segmentation_params["boxes"].tolist()
"""
        else:
            # Default postprocessing for other diffusion tasks
            return """
# Default postprocessing for diffusion models
results = {}

# Run inference
with self.torch.no_grad():
    output = endpoint(**generation_params)

# Extract generated content
if hasattr(output, "images"):
    # Most diffusion pipelines return images
    generated_images = output.images
    results["images"] = generated_images
    
    # Encode images to base64 for response
    base64_images = []
    for img in generated_images:
        if isinstance(img, np.ndarray):
            # Convert numpy array to PIL Image
            img = Image.fromarray(np.uint8(img * 255) if img.max() <= 1.0 else np.uint8(img))
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(img_str)
    
    results["base64_images"] = base64_images
else:
    # Generic output capture
    results["raw_output"] = str(output)

# Add parameters used for generation
results["parameters"] = {key: value for key, value in generation_params.items() 
                         if key not in ["generator", "image", "mask_image"]}

# If seed was used, add it to response
if "generator" in generation_params:
    results["parameters"]["seed"] = generation_params["generator"].initial_seed()
"""
    
    def get_result_formatting_code(self, task_type: str) -> str:
        """Get diffusion result formatting code for specific task types."""
        if task_type == "image_generation" or task_type == "image_to_image" or task_type == "inpainting":
            return """
# Format results for diffusion-based image generation
return {
    "success": True,
    "diffusion_output": {
        "images": results.get("base64_images", []),
        "count": len(results.get("base64_images", [])),
        "parameters": results.get("parameters", {}),
        "task_type": "{0}".format(task_type)
    },
    "device": device,
    "hardware": hardware_label
}
"""
        elif task_type == "image_segmentation":
            return """
# Format results for image segmentation
return {
    "success": True,
    "segmentation_output": {
        "masks": results.get("base64_masks", []),
        "count": results.get("mask_count", 0),
        "parameters": results.get("parameters", {}),
        "scores": results.get("scores", []),
    },
    "device": device,
    "hardware": hardware_label
}
"""
        else:
            # Default result formatting for diffusion tasks
            return """
# Default format for diffusion model results
return {
    "success": True,
    "diffusion_output": results,
    "device": device,
    "hardware": hardware_label
}
"""
    
    def get_mock_input_code(self) -> str:
        """Get diffusion mock input code."""
        return """
# Mock diffusion input
from PIL import Image
import numpy as np
import tempfile
import os

# Create a mock prompt
mock_prompt = "A beautiful landscape with mountains"

# For image-to-image, create a mock image
mock_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
mock_image_path = os.path.join(tempfile.gettempdir(), "mock_image.jpg")
mock_image.save(mock_image_path)

# For inpainting, create a mock mask
mock_mask = Image.fromarray(np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255)
mock_mask_path = os.path.join(tempfile.gettempdir(), "mock_mask.jpg")
mock_mask.save(mock_mask_path)

# For segmentation, create mock points and box
mock_points = [[256, 256], [300, 300]]
mock_box = [100, 100, 400, 400]

# Create combined diffusion input
mock_input = {
    "prompt": mock_prompt,
    "image": mock_image,
    "mask": mock_mask,
    "points": mock_points,
    "box": mock_box,
    "guidance_scale": 7.5,
    "num_inference_steps": 30
}
"""
    
    def get_mock_output_code(self) -> str:
        """Get diffusion mock output code."""
        return """
# Mock diffusion output
from PIL import Image
import numpy as np
import base64
import io

# Create a mock image result
mock_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
buffered = io.BytesIO()
mock_image.save(buffered, format="PNG")
mock_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

# For segmentation, create a mock mask
mock_mask = np.zeros((512, 512), dtype=np.bool_)
mock_mask[100:400, 100:400] = True

mock_output = type('MockDiffusionOutput', (), {})()
mock_output.images = [mock_image]
mock_output.masks = [mock_mask]

return mock_output
"""
    
    def get_pipeline_utilities(self) -> str:
        """Get diffusion utility functions."""
        return """
# Diffusion pipeline utilities
def encode_image_base64(image):
    # Encode an image to base64 string
    if isinstance(image, str) and os.path.exists(image):
        with open(image, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    # Assume it's a PIL Image
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_image_base64(base64_string):
    # Decode a base64 string to a PIL Image
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def create_mask_from_points(image_size, points, radius=10):
    # Create a binary mask from a list of points
    from PIL import Image, ImageDraw
    
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for point in points:
        x, y = point
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
    
    return mask
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check diffusion pipeline compatibility with architecture type."""
        # Diffusion pipeline is compatible with diffusion-based architectures
        return arch_type in [
            "diffusion",
            "vae",  # Variational Autoencoders can be components in diffusion models
            "sam"   # Segment Anything Model can use similar preprocessing
        ]
    
    def is_compatible_with_task(self, task_type: str) -> bool:
        """Check diffusion pipeline compatibility with task type."""
        # Diffusion pipeline is compatible with these tasks
        return task_type in [
            "image_generation",
            "text_to_image",
            "image_to_image",
            "inpainting",
            "image_segmentation",
            "image_variation"
        ]