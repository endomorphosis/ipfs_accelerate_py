#!/usr/bin/env python3
"""
Vision-Text Architecture Template

This module provides the architecture template for vision-text models like CLIP, BLIP, etc.
"""

from typing import Dict, Any, List
from .base_architecture import BaseArchitectureTemplate

class VisionTextArchitectureTemplate(BaseArchitectureTemplate):
    """Template for vision-text architecture models like CLIP, BLIP, etc."""
    
    def __init__(self):
        """Initialize the vision-text architecture template."""
        super().__init__()
        self.architecture_type = "vision-encoder-text-decoder"
        self.architecture_name = "Vision-Text Architecture"
        self.supported_task_types = ["image_text_matching", "visual_question_answering", "image_captioning"]
        self.default_task_type = "image_text_matching"
        self.model_description = "This is a multimodal model that processes both vision and text inputs."
        self.hidden_size = 768  # Default hidden size, varies by model
        self.test_input = "test.jpg"  # Default test file for images
        
    def get_model_class(self, task_type: str) -> str:
        """Get vision-text model class for task type."""
        if task_type == "image_text_matching":
            return "self.transformers.CLIPModel"
        elif task_type == "visual_question_answering":
            return "self.transformers.VisionEncoderDecoderModel"
        elif task_type == "image_captioning":
            return "self.transformers.VisionEncoderDecoderModel"
        else:
            return "self.transformers.AutoModel"
    
    def get_processor_class(self, task_type: str) -> str:
        """Get vision-text processor class for task type."""
        if task_type == "image_text_matching":
            return "self.transformers.CLIPProcessor"
        else:
            return "self.transformers.AutoProcessor"
    
    def get_input_processing_code(self, task_type: str) -> str:
        """Get vision-text input processing code."""
        if task_type == "image_text_matching":
            return """
# Load image
import os
from PIL import Image

# If text is a path to an image, use it directly
if os.path.exists(text) and text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
    image_path = text
else:
    # If not, search for a default test image
    test_paths = [
        "test.jpg",
        os.path.join(os.path.dirname(__file__), "test.jpg"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")
    ]
    
    image_path = None
    for path in test_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path is None:
        # Create a simple test image
        import numpy as np
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        image_path = "temp_test_image.jpg"
        img.save(image_path)

# Load image
image = Image.open(image_path).convert("RGB")

# Prepare text prompts
text_prompts = [
    "a photo of a cat",
    "a photo of a dog",
    "a landscape",
    "a portrait",
    "a painting"
]

# Process inputs for image-text matching
inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "visual_question_answering":
            return """
# Load image
import os
from PIL import Image

# If text starts with "question:", extract the question part
if text.lower().startswith("question:"):
    question = text[len("question:"):].strip()
    image_path = None
else:
    # If text is a path to an image, use it directly
    if os.path.exists(text) and text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        image_path = text
        question = "What is in this image?"
    else:
        # If not, search for a default test image
        test_paths = [
            "test.jpg",
            os.path.join(os.path.dirname(__file__), "test.jpg"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")
        ]
        
        image_path = None
        for path in test_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        question = text

# If we still don't have an image path, create a simple test image
if image_path is None:
    import numpy as np
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    image_path = "temp_test_image.jpg"
    img.save(image_path)

# Load image
image = Image.open(image_path).convert("RGB")

# Process inputs for visual question answering
inputs = processor(image, question, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "image_captioning":
            return """
# Load image
import os
from PIL import Image

# If text is a path to an image, use it directly
if os.path.exists(text) and text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
    image_path = text
else:
    # If not, search for a default test image
    test_paths = [
        "test.jpg",
        os.path.join(os.path.dirname(__file__), "test.jpg"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")
    ]
    
    image_path = None
    for path in test_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path is None:
        # Create a simple test image
        import numpy as np
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        image_path = "temp_test_image.jpg"
        img.save(image_path)

# Load image
image = Image.open(image_path).convert("RGB")

# Process inputs for image captioning
inputs = processor(images=image, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        else:
            return """
# Load image
import os
from PIL import Image

# If text is a path to an image, use it directly
if os.path.exists(text) and text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
    image_path = text
else:
    # If not, search for a default test image
    test_paths = [
        "test.jpg",
        os.path.join(os.path.dirname(__file__), "test.jpg"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")
    ]
    
    image_path = None
    for path in test_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path is None:
        # Create a simple test image
        import numpy as np
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        image_path = "temp_test_image.jpg"
        img.save(image_path)

# Load image
image = Image.open(image_path).convert("RGB")

# Process image
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Process text if provided
if text != image_path:
    text_inputs = processor(text=text, return_tensors="pt")
    inputs = {**text_inputs, "pixel_values": pixel_values}
else:
    inputs = {"pixel_values": pixel_values}

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
    
    def get_output_processing_code(self, task_type: str) -> str:
        """Get vision-text output processing code."""
        if task_type == "image_text_matching":
            return """
# Process output for image-text matching
with self.torch.no_grad():
    outputs = model(**inputs)

# Get similarity scores
if hasattr(outputs, "logits_per_image"):
    # CLIP-like output
    logits_per_image = outputs.logits_per_image
    probs = self.torch.nn.functional.softmax(logits_per_image, dim=1)
    
    # Format results
    result = []
    for i, text_prompt in enumerate(text_prompts):
        result.append({
            "text": text_prompt,
            "score": probs[0, i].item()
        })
    
    # Sort by score (highest first)
    result = sorted(result, key=lambda x: x["score"], reverse=True)
else:
    # Generic fallback
    result = outputs
"""
        elif task_type == "visual_question_answering":
            return """
# Process output for visual question answering
with self.torch.no_grad():
    outputs = model.generate(**inputs)

# Decode the generated answer
result = processor.decode(outputs[0], skip_special_tokens=True)
"""
        elif task_type == "image_captioning":
            return """
# Process output for image captioning
with self.torch.no_grad():
    outputs = model.generate(**inputs)

# Decode the generated caption
result = processor.decode(outputs[0], skip_special_tokens=True)
"""
        else:
            return """
# Generic output processing
with self.torch.no_grad():
    outputs = model(**inputs)

# Extract relevant information
if hasattr(outputs, "logits_per_image"):
    result = outputs.logits_per_image
elif hasattr(outputs, "text_embeds") and hasattr(outputs, "image_embeds"):
    # Get similarity between text and image embeddings
    import torch.nn.functional as F
    text_embeds = outputs.text_embeds
    image_embeds = outputs.image_embeds
    similarity = F.cosine_similarity(text_embeds, image_embeds)
    result = similarity.item()
else:
    # Fallback to returning the full output
    result = outputs
"""
    
    def get_mock_processor_code(self) -> str:
        """Get vision-text mock processor code."""
        return """
def mock_tokenize(text=None, images=None, return_tensors="pt", padding=None, truncation=None, max_length=None, **kwargs):
    if hasattr(self, 'torch'):
        torch = self.torch
    else:
        import torch
    
    # Determine batch size
    if images is not None:
        if hasattr(images, "shape"):
            # Single image tensor
            batch_size = 1
        elif hasattr(images, "__len__"):
            # List of images
            batch_size = len(images)
        else:
            # Single PIL image
            batch_size = 1
    elif text is not None:
        if isinstance(text, str):
            batch_size = 1
        else:
            batch_size = len(text)
    else:
        batch_size = 1
    
    # Create mock vision inputs
    pixel_values = torch.rand(batch_size, 3, 224, 224)
    
    # Create mock text inputs if text is provided
    result = {"pixel_values": pixel_values}
    
    if text is not None:
        result.update({
            "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 10), dtype=torch.long)
        })
    
    return result
"""
    
    def get_mock_output_code(self) -> str:
        """Get vision-text mock output code."""
        return """
result = MagicMock()
result.logits_per_image = torch.rand((batch_size, batch_size))
result.logits_per_text = torch.rand((batch_size, batch_size))
result.text_embeds = torch.rand((batch_size, hidden_size))
result.image_embeds = torch.rand((batch_size, hidden_size))
return result
"""
    
    def get_compatibility_matrix(self) -> Dict[str, bool]:
        """Get vision-text hardware compatibility matrix."""
        return {
            "cpu": True,
            "cuda": True,
            "rocm": True,
            "mps": True,
            "openvino": True,
            "qnn": False  # Limited support for multimodal models in Qualcomm QNN
        }