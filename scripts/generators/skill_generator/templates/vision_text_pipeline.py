#!/usr/bin/env python3
"""
Vision-Text Pipeline Template for IPFS Accelerate Python.

This module implements a pipeline template for vision-text models like CLIP, BLIP, etc.
It handles multimodal inputs (images and text) and implements task-specific processing
for image-text matching, visual question answering, and image captioning.
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipelineTemplate


class VisionTextPipelineTemplate(BasePipelineTemplate):
    """Template for vision-text pipelines."""
    
    def __init__(self):
        """Initialize the vision-text pipeline template."""
        super().__init__()
        self.pipeline_type = "vision-text"
        self.input_type = "multimodal"
        self.output_type = "multimodal"
        self.requires_preprocessing = True
        self.requires_postprocessing = True
        self.supports_batching = True
        self.max_batch_size = 8  # Smaller batch size due to memory requirements
    
    def get_import_statements(self) -> str:
        """Get vision-text pipeline import statements."""
        return """
# Vision-Text pipeline imports
import os
import json
import numpy as np
import base64
from typing import List, Dict, Union, Any
from PIL import Image
import io
"""
    
    def get_preprocessing_code(self, task_type: str) -> str:
        """Get vision-text preprocessing code for specific task types."""
        if task_type == "image_text_matching":
            return """
# Preprocess for image-text matching (CLIP-like)
from PIL import Image
import base64
import io

# Get image and text inputs
if isinstance(text, dict) and "image" in text and "text" in text:
    # Input is a dictionary with both image and text
    image_input = text["image"]
    text_input = text["text"]
elif isinstance(text, dict) and "image" in text:
    # Input is a dictionary with only image, use default text prompts
    image_input = text["image"]
    text_input = [
        "a photo of a person",
        "a photo of an animal",
        "a photo of a landscape",
        "a photo of food",
        "a photo of an object"
    ]
elif isinstance(text, str) and os.path.exists(text) and text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
    # Input is a path to an image, use default text prompts
    image_input = text
    text_input = [
        "a photo of a person",
        "a photo of an animal",
        "a photo of a landscape",
        "a photo of food",
        "a photo of an object"
    ]
elif isinstance(text, tuple) and len(text) == 2:
    # Input is a tuple of (image, text)
    image_input, text_input = text
else:
    # Try to find a test image
    test_paths = [
        "test.jpg",
        os.path.join(os.path.dirname(__file__), "test.jpg"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")
    ]
    
    image_input = None
    for path in test_paths:
        if os.path.exists(path):
            image_input = path
            break
    
    if image_input is None:
        # Create a simple test image if no test image found
        try:
            image = Image.new("RGB", (224, 224), color="white")
            test_path = os.path.join(os.path.dirname(__file__), "temp_test_image.jpg")
            image.save(test_path)
            image_input = test_path
        except Exception as e:
            raise ValueError(f"Could not create test image: {e}")
    
    # Default text prompts
    text_input = [
        "a photo of a person",
        "a photo of an animal",
        "a photo of a landscape",
        "a photo of food",
        "a photo of an object"
    ]

# Process image input
if isinstance(image_input, str):
    # Check if it's a file path
    if os.path.exists(image_input):
        # It's a file path
        image = Image.open(image_input).convert('RGB')
    elif image_input.startswith(('data:image', 'http://', 'https://')):
        # It's a URL or data URI
        if image_input.startswith('data:image'):
            # Base64 encoded image
            image_data = image_input.split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        else:
            # URL (this would require requests in a real implementation)
            raise ValueError("URL images not implemented yet")
    else:
        # Assume it's a base64 string directly
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_input))).convert('RGB')
        except:
            raise ValueError(f"Could not parse input as image: {image_input[:30]}...")
elif isinstance(image_input, Image.Image):
    # Already a PIL Image
    image = image_input
elif isinstance(image_input, bytes):
    # Raw bytes
    image = Image.open(io.BytesIO(image_input)).convert('RGB')
else:
    raise ValueError(f"Invalid image input type: {type(image_input)}")

# Process text input
if not isinstance(text_input, list):
    text_input = [text_input]

# Prepare inputs for the model
inputs = tokenizer(
    text=text_input,
    images=image,
    return_tensors="pt",
    padding=True
)

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "visual_question_answering":
            return """
# Preprocess for visual question answering (BLIP-like)
from PIL import Image
import base64
import io

# Get image and question inputs
if isinstance(text, dict) and "image" in text and "question" in text:
    # Input is a dictionary with both image and question
    image_input = text["image"]
    question = text["question"]
elif isinstance(text, tuple) and len(text) == 2:
    # Input is a tuple of (image, question)
    image_input, question = text
elif isinstance(text, str) and os.path.exists(text) and text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
    # Input is a path to an image, use default question
    image_input = text
    question = "What is in this image?"
else:
    # Try to find a test image
    test_paths = [
        "test.jpg",
        os.path.join(os.path.dirname(__file__), "test.jpg"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")
    ]
    
    image_input = None
    for path in test_paths:
        if os.path.exists(path):
            image_input = path
            break
    
    if image_input is None:
        # Create a simple test image if no test image found
        try:
            image = Image.new("RGB", (224, 224), color="white")
            test_path = os.path.join(os.path.dirname(__file__), "temp_test_image.jpg")
            image.save(test_path)
            image_input = test_path
        except Exception as e:
            raise ValueError(f"Could not create test image: {e}")
    
    # Default question
    question = "What is in this image?"

# Process image input
if isinstance(image_input, str):
    # Check if it's a file path
    if os.path.exists(image_input):
        # It's a file path
        image = Image.open(image_input).convert('RGB')
    elif image_input.startswith(('data:image', 'http://', 'https://')):
        # It's a URL or data URI
        if image_input.startswith('data:image'):
            # Base64 encoded image
            image_data = image_input.split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        else:
            # URL (this would require requests in a real implementation)
            raise ValueError("URL images not implemented yet")
    else:
        # Assume it's a base64 string directly
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_input))).convert('RGB')
        except:
            raise ValueError(f"Could not parse input as image: {image_input[:30]}...")
elif isinstance(image_input, Image.Image):
    # Already a PIL Image
    image = image_input
elif isinstance(image_input, bytes):
    # Raw bytes
    image = Image.open(io.BytesIO(image_input)).convert('RGB')
else:
    raise ValueError(f"Invalid image input type: {type(image_input)}")

# Prepare inputs for the model
inputs = tokenizer(
    images=image,
    text=question,
    return_tensors="pt",
    padding=True
)

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "image_captioning":
            return """
# Preprocess for image captioning
from PIL import Image
import base64
import io

# Get image input
if isinstance(text, dict) and "image" in text:
    # Input is a dictionary with image
    image_input = text["image"]
elif isinstance(text, str) and os.path.exists(text) and text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
    # Input is a path to an image
    image_input = text
elif isinstance(text, Image.Image):
    # Input is already a PIL Image
    image_input = text
else:
    # Try to find a test image
    test_paths = [
        "test.jpg",
        os.path.join(os.path.dirname(__file__), "test.jpg"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")
    ]
    
    image_input = None
    for path in test_paths:
        if os.path.exists(path):
            image_input = path
            break
    
    if image_input is None:
        # Create a simple test image if no test image found
        try:
            image = Image.new("RGB", (224, 224), color="white")
            test_path = os.path.join(os.path.dirname(__file__), "temp_test_image.jpg")
            image.save(test_path)
            image_input = test_path
        except Exception as e:
            raise ValueError(f"Could not create test image: {e}")

# Process image input
if isinstance(image_input, str):
    # Check if it's a file path
    if os.path.exists(image_input):
        # It's a file path
        image = Image.open(image_input).convert('RGB')
    elif image_input.startswith(('data:image', 'http://', 'https://')):
        # It's a URL or data URI
        if image_input.startswith('data:image'):
            # Base64 encoded image
            image_data = image_input.split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        else:
            # URL (this would require requests in a real implementation)
            raise ValueError("URL images not implemented yet")
    else:
        # Assume it's a base64 string directly
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_input))).convert('RGB')
        except:
            raise ValueError(f"Could not parse input as image: {image_input[:30]}...")
elif isinstance(image_input, Image.Image):
    # Already a PIL Image
    image = image_input
elif isinstance(image_input, bytes):
    # Raw bytes
    image = Image.open(io.BytesIO(image_input)).convert('RGB')
else:
    raise ValueError(f"Invalid image input type: {type(image_input)}")

# Prepare inputs for the model
inputs = tokenizer(images=image, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        else:
            # Default preprocessing for other tasks
            return """
# Default preprocessing for vision-text tasks
from PIL import Image
import base64
import io

# Try to get image input
if isinstance(text, dict) and "image" in text:
    # Input is a dictionary with image
    image_input = text["image"]
elif isinstance(text, str) and os.path.exists(text) and text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
    # Input is a path to an image
    image_input = text
elif isinstance(text, Image.Image):
    # Input is already a PIL Image
    image_input = text
else:
    # Try to find a test image
    test_paths = [
        "test.jpg",
        os.path.join(os.path.dirname(__file__), "test.jpg"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")
    ]
    
    image_input = None
    for path in test_paths:
        if os.path.exists(path):
            image_input = path
            break
    
    if image_input is None:
        # Create a simple test image if no test image found
        try:
            image = Image.new("RGB", (224, 224), color="white")
            test_path = os.path.join(os.path.dirname(__file__), "temp_test_image.jpg")
            image.save(test_path)
            image_input = test_path
        except Exception as e:
            raise ValueError(f"Could not create test image: {e}")

# Process image input
if isinstance(image_input, str):
    # Check if it's a file path
    if os.path.exists(image_input):
        # It's a file path
        image = Image.open(image_input).convert('RGB')
    elif image_input.startswith(('data:image', 'http://', 'https://')):
        # It's a URL or data URI
        if image_input.startswith('data:image'):
            # Base64 encoded image
            image_data = image_input.split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        else:
            # URL (this would require requests in a real implementation)
            raise ValueError("URL images not implemented yet")
    else:
        # Assume it's a base64 string directly
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_input))).convert('RGB')
        except:
            raise ValueError(f"Could not parse input as image: {image_input[:30]}...")
elif isinstance(image_input, Image.Image):
    # Already a PIL Image
    image = image_input
elif isinstance(image_input, bytes):
    # Raw bytes
    image = Image.open(io.BytesIO(image_input)).convert('RGB')
else:
    raise ValueError(f"Invalid image input type: {type(image_input)}")

# Check if we have text input alongside the image
text_input = None
if isinstance(text, dict) and "text" in text:
    text_input = text["text"]
elif isinstance(text, tuple) and len(text) >= 2:
    text_input = text[1]

# Prepare inputs for the model
if text_input is not None:
    # Multimodal input (both image and text)
    inputs = tokenizer(
        text=text_input,
        images=image,
        return_tensors="pt",
        padding=True
    )
else:
    # Image-only input
    inputs = tokenizer(images=image, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
    
    def get_postprocessing_code(self, task_type: str) -> str:
        """Get vision-text postprocessing code for specific task types."""
        if task_type == "image_text_matching":
            return """
# Run inference for image-text matching
with self.torch.no_grad():
    outputs = model(**inputs)

# Process outputs
logits_per_image = outputs.logits_per_image  # image-text similarity score
probs = self.torch.nn.functional.softmax(logits_per_image, dim=1)
"""
        elif task_type == "visual_question_answering":
            return """
# Run inference for visual question answering
with self.torch.no_grad():
    outputs = model.generate(**inputs)

# Process outputs
answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
"""
        elif task_type == "image_captioning":
            return """
# Run inference for image captioning
with self.torch.no_grad():
    outputs = model.generate(**inputs)

# Process outputs
caption = tokenizer.batch_decode(outputs, skip_special_tokens=True)
"""
        else:
            # Default postprocessing for other tasks
            return """
# Default inference for vision-text tasks
with self.torch.no_grad():
    outputs = model(**inputs)

# Generic output processing
if hasattr(outputs, "logits_per_image"):
    # CLIP-like output (image-text similarity)
    logits_per_image = outputs.logits_per_image
    probs = self.torch.nn.functional.softmax(logits_per_image, dim=1)
    result = {
        "scores": probs[0].cpu().tolist(),
        "type": "image_text_similarity"
    }
elif hasattr(outputs, "loss"):
    # Vision-language modeling loss
    result = {
        "loss": outputs.loss.item(),
        "type": "vision_language_modeling"
    }
else:
    # Generic output
    result = {
        "type": "vision_text_generic",
        "data": str(outputs)
    }
"""
    
    def get_result_formatting_code(self, task_type: str) -> str:
        """Get vision-text result formatting code for specific task types."""
        if task_type == "image_text_matching":
            return """
# Format results for image-text matching
similarity_scores = probs[0].cpu().tolist() if hasattr(probs[0], "tolist") else probs[0]
highest_similarity_idx = int(probs[0].argmax().item()) if hasattr(probs[0], "argmax") else 0

# Extract text labels if available
text_labels = text_input if isinstance(text_input, list) else ["Unknown"]

# Create result dictionary
result = {
    "success": True,
    "similarity": {
        "scores": similarity_scores,
        "highest_idx": highest_similarity_idx,
        "highest_match": text_labels[highest_similarity_idx] if highest_similarity_idx < len(text_labels) else None,
        "labels": text_labels if len(text_labels) == len(similarity_scores) else None
    },
    "device": device,
    "hardware": hardware_label
}

return result
"""
        elif task_type == "visual_question_answering":
            return """
# Format results for visual question answering
return {
    "success": True,
    "visual_qa": {
        "question": question,
        "answer": answer[0] if len(answer) > 0 else "",
        "answers": answer
    },
    "device": device,
    "hardware": hardware_label
}
"""
        elif task_type == "image_captioning":
            return """
# Format results for image captioning
return {
    "success": True,
    "caption": {
        "text": caption[0] if len(caption) > 0 else "",
        "all_captions": caption
    },
    "device": device,
    "hardware": hardware_label
}
"""
        else:
            # Default result formatting for other tasks
            return """
# Default result formatting for vision-text tasks
return {
    "success": True,
    "vision_text_results": result,
    "device": device,
    "hardware": hardware_label
}
"""
    
    def get_mock_input_code(self) -> str:
        """Get vision-text mock input code."""
        return """
# Mock vision-text input
from PIL import Image
import numpy as np

# Create a mock image
mock_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

# Create mock text input
mock_text = ["a photo of a cat", "a photo of a dog", "a photo of a landscape"]

# Create mock combined input
mock_input = {"image": mock_image, "text": mock_text}
"""
    
    def get_mock_output_code(self) -> str:
        """Get vision-text mock output code."""
        return """
# Mock vision-text output
mock_output = {
    "success": True,
    "similarity": {
        "scores": [0.8, 0.1, 0.1],
        "highest_idx": 0,
        "highest_match": "a photo of a cat",
        "labels": ["a photo of a cat", "a photo of a dog", "a photo of a landscape"]
    },
    "device": "cpu",
    "hardware": "mock"
}
"""
    
    def get_pipeline_utilities(self) -> str:
        """Get vision-text utility functions."""
        return """
# Vision-Text pipeline utilities
def resize_image(image, target_size=(224, 224)):
    # Resize an image to the target size
    if isinstance(image, str) and os.path.exists(image):
        image = Image.open(image)
    
    if image.size != target_size:
        return image.resize(target_size, Image.LANCZOS)
    return image

def encode_image_base64(image):
    # Encode an image to base64 string
    if isinstance(image, str) and os.path.exists(image):
        with open(image, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    # Assume it's a PIL Image
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check vision-text pipeline compatibility with architecture type."""
        # Vision-text pipeline is compatible with vision-text architectures
        return arch_type in [
            "vision-encoder-text-decoder",
            "multimodal"
        ]
    
    def is_compatible_with_task(self, task_type: str) -> bool:
        """Check vision-text pipeline compatibility with task type."""
        # Vision-text pipeline is compatible with vision-text tasks
        return task_type in [
            "image_text_matching",
            "visual_question_answering",
            "image_captioning",
            "multimodal_embedding"
        ]