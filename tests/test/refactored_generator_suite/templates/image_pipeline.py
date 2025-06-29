#!/usr/bin/env python3
"""
Image pipeline template for IPFS Accelerate Python.

This module implements the pipeline template for image processing operations.
"""

from typing import Dict, Any, Callable, Tuple, Optional, List, Union
from templates.base_pipeline import BasePipelineTemplate


class ImagePipelineTemplate(BasePipelineTemplate):
    """Image pipeline template implementation."""
    
    def __init__(self):
        """Initialize the image pipeline template."""
        super().__init__()
        self.pipeline_type = "image"
        self.input_type = "image"
        self.output_type = "text"  # Can be 'text' or 'class' or 'embeddings' depending on task
        self.requires_preprocessing = True
        self.requires_postprocessing = True
        self.supports_batching = True
        self.max_batch_size = 16  # Images typically need more memory
    
    def get_import_statements(self) -> str:
        """Get image-specific import statements."""
        return """
# Image-specific imports
import os
import json
import numpy as np
import base64
from typing import List, Dict, Union, Any
from PIL import Image
import io
"""
    
    def get_preprocessing_code(self, task_type: str) -> str:
        """Get image-specific preprocessing code."""
        if task_type == "image_classification":
            return """
# Preprocess images for classification
# Handle both file paths, PIL images, and base64 encoded images
if isinstance(text, str):
    # Check if it's a file path
    if os.path.exists(text):
        # It's a file path
        image = Image.open(text).convert('RGB')
    elif text.startswith(('data:image', 'http://', 'https://')):
        # It's a URL or data URI
        if text.startswith('data:image'):
            # Base64 encoded image
            image_data = text.split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        else:
            # URL (this would require requests in a real implementation)
            raise ValueError("URL images not implemented yet")
    else:
        # Assume it's a base64 string directly
        try:
            image = Image.open(io.BytesIO(base64.b64decode(text))).convert('RGB')
        except:
            raise ValueError(f"Could not parse input as image: {text[:30]}...")
else:
    # Assume it's already a PIL Image
    image = text

# Use image processor for preprocessing 
if hasattr(tokenizer, 'preprocess'):
    # Some models use a processor instead of a tokenizer
    inputs = tokenizer.preprocess(image, return_tensors="pt")
else:
    # For compatibility with vision models using processors
    try:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(endpoint_model)
        inputs = processor(image, return_tensors="pt")
    except Exception as e:
        raise ValueError(f"Could not process image: {e}")

# Move inputs to the correct device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "image_to_text":
            return """
# Preprocess images for image-to-text generation
# Handle both file paths, PIL images, and base64 encoded images
if isinstance(text, str):
    # Check if it's a file path
    if os.path.exists(text):
        # It's a file path
        image = Image.open(text).convert('RGB')
    elif text.startswith(('data:image', 'http://', 'https://')):
        # It's a URL or data URI
        if text.startswith('data:image'):
            # Base64 encoded image
            image_data = text.split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        else:
            # URL (this would require requests in a real implementation)
            raise ValueError("URL images not implemented yet")
    else:
        # Assume it's a base64 string directly
        try:
            image = Image.open(io.BytesIO(base64.b64decode(text))).convert('RGB')
        except:
            raise ValueError(f"Could not parse input as image: {text[:30]}...")
else:
    # Assume it's already a PIL Image
    image = text

# For multimodal models, use the appropriate processor
if hasattr(tokenizer, 'image_processor') and hasattr(tokenizer, 'tokenizer'):
    # This is a multimodal processor like for BLIP, etc.
    inputs = tokenizer(image, return_tensors="pt")
elif hasattr(tokenizer, 'processor'):
    # For models like VisionEncoderDecoderModel
    inputs = tokenizer.processor(image, return_tensors="pt")
else:
    # Generic approach - try to find an image processor
    try:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(endpoint_model)
        inputs = processor(image, return_tensors="pt")
    except Exception as e:
        raise ValueError(f"Could not process image: {e}")

# Move inputs to the correct device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "vision_embedding":
            return """
# Preprocess images for embedding extraction
# Handle both file paths, PIL images, and base64 encoded images
if isinstance(text, str):
    # Check if it's a file path
    if os.path.exists(text):
        # It's a file path
        image = Image.open(text).convert('RGB')
    elif text.startswith(('data:image', 'http://', 'https://')):
        # It's a URL or data URI
        if text.startswith('data:image'):
            # Base64 encoded image
            image_data = text.split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        else:
            # URL (this would require requests in a real implementation)
            raise ValueError("URL images not implemented yet")
    else:
        # Assume it's a base64 string directly
        try:
            image = Image.open(io.BytesIO(base64.b64decode(text))).convert('RGB')
        except:
            raise ValueError(f"Could not parse input as image: {text[:30]}...")
else:
    # Assume it's already a PIL Image
    image = text

# For embedding models like CLIP, use the appropriate processor
if hasattr(tokenizer, 'image_processor'):
    inputs = {"pixel_values": tokenizer.image_processor(image, return_tensors="pt")["pixel_values"]}
else:
    # Generic approach
    try:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(endpoint_model)
        inputs = processor(image, return_tensors="pt")
    except Exception as e:
        raise ValueError(f"Could not process image: {e}")

# Move inputs to the correct device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        else:
            return f"""
# Generic image preprocessing for {task_type}
# Handle both file paths and PIL images
if isinstance(text, str) and os.path.exists(text):
    image = Image.open(text).convert('RGB')
else:
    # Assume it's already a PIL Image or try to convert from base64
    try:
        if isinstance(text, str):
            image = Image.open(io.BytesIO(base64.b64decode(text))).convert('RGB')
        else:
            image = text
    except:
        raise ValueError(f"Could not parse input as image")

# Generic image processing
try:
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(endpoint_model)
    inputs = processor(image, return_tensors="pt")
except Exception as e:
    raise ValueError(f"Could not process image: {e}")

# Move inputs to the correct device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
    
    def get_postprocessing_code(self, task_type: str) -> str:
        """Get image-specific postprocessing code."""
        if task_type == "image_classification":
            return """
# Postprocess classification results
logits = outputs.logits
predictions = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

# Get class labels if they exist in the model
id2label = getattr(endpoint.config, 'id2label', None)
if id2label:
    # Sort predictions and get top 5
    top_indices = predictions[0].argsort()[-5:][::-1]
    labels = [id2label[str(idx)] if str(idx) in id2label else f"LABEL_{idx}" for idx in top_indices]
    scores = predictions[0][top_indices].tolist()
    results = [{"label": label, "score": score} for label, score in zip(labels, scores)]
else:
    # If no labels, just return indices and scores
    top_indices = predictions[0].argsort()[-5:][::-1]
    scores = predictions[0][top_indices].tolist()
    results = [{"class": int(idx), "score": score} for idx, score in zip(top_indices, scores)]
"""
        elif task_type == "image_to_text":
            return """
# Postprocess image to text results
if hasattr(outputs, 'sequences'):
    # For models that return sequences directly
    output_ids = outputs.sequences
else:
    # For models where we used generate()
    output_ids = output_ids  # Already set by the generate call

# Decode the output tokens to text
generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
"""
        elif task_type == "vision_embedding":
            return """
# Postprocess vision embeddings
if hasattr(outputs, 'image_embeds'):
    # For models like CLIP that return image_embeds
    embeddings = outputs.image_embeds.cpu().numpy().tolist()
elif hasattr(outputs, 'last_hidden_state'):
    # For encoder models that return hidden states
    # Take the CLS token embedding or average all token embeddings
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()  # CLS token
else:
    # Generic approach - treat outputs as embeddings
    embeddings = outputs.cpu().detach().numpy().tolist()
"""
        else:
            return f"""
# Generic postprocessing for {task_type}
# Just convert outputs to Python types for JSON serialization
result = outputs
if hasattr(result, 'cpu'):
    result = result.cpu().detach().numpy()
"""
    
    def get_result_formatting_code(self, task_type: str) -> str:
        """Get image-specific result formatting code."""
        if task_type == "image_classification":
            return """
return {"success": True,
        "classifications": results,
        "device": device,
        "hardware": hardware_label}
"""
        elif task_type == "image_to_text":
            return """
return {"success": True,
        "generated_text": generated_texts[0] if len(generated_texts) > 0 else "",
        "all_texts": generated_texts,
        "device": device,
        "hardware": hardware_label}
"""
        elif task_type == "vision_embedding":
            return """
return {"success": True,
        "embeddings": embeddings,
        "device": device,
        "hardware": hardware_label}
"""
        else:
            return f"""
return {{"success": True,
        "result": result,
        "device": device,
        "hardware": hardware_label}}
"""
    
    def get_mock_input_code(self) -> str:
        """Get image-specific mock input code."""
        return """
# Mock image input (create a small black image)
from PIL import Image
import numpy as np
import io
import base64

mock_image = Image.new('RGB', (224, 224), color='black')
mock_buffer = io.BytesIO()
mock_image.save(mock_buffer, format='PNG')
mock_image_base64 = base64.b64encode(mock_buffer.getvalue()).decode('utf-8')
"""
    
    def get_mock_output_code(self) -> str:
        """Get image-specific mock output code."""
        return """
# Mock image output based on the task
if "image_classification" in task_type:
    # Mock classification output
    output_obj = type('obj', (object,), {
        'logits': torch.zeros((batch_size, 1000))
    })
    return output_obj
elif "image_to_text" in task_type:
    # Mock image to text output
    return torch.ones((batch_size, 5), dtype=torch.long)  # Mock token IDs
elif "vision_embedding" in task_type:
    # Mock embedding output
    output_obj = type('obj', (object,), {
        'image_embeds': torch.zeros((batch_size, hidden_size))
    })
    return output_obj
else:
    # Generic mock output
    return torch.zeros((batch_size, hidden_size))
"""
    
    def get_pipeline_utilities(self) -> str:
        """Get image-specific utility functions."""
        return """
# Image pipeline utilities
def resize_image(image, target_size=(224, 224)):
    \"\"\"Resize an image to the target size.\"\"\"
    if isinstance(image, str) and os.path.exists(image):
        image = Image.open(image)
    
    if image.size != target_size:
        return image.resize(target_size, Image.LANCZOS)
    return image

def encode_image_base64(image):
    \"\"\"Encode an image to base64 string.\"\"\"
    if isinstance(image, str) and os.path.exists(image):
        with open(image, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    # Assume it's a PIL Image
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check image pipeline compatibility with architecture type."""
        # Image pipeline is compatible with vision-based architectures
        return arch_type in [
            "vision",
            "vision-encoder-text-decoder"
        ]
    
    def is_compatible_with_task(self, task_type: str) -> bool:
        """Check image pipeline compatibility with task type."""
        # Image pipeline is compatible with vision-based tasks
        return task_type in [
            "image_classification",
            "object_detection",
            "image_segmentation",
            "image_to_text",
            "vision_embedding"
        ]