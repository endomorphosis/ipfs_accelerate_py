#!/usr/bin/env python3
"""
Multimodal Pipeline Template for IPFS Accelerate Python.

This module implements a pipeline template for multimodal models like FLAVA, LLaVA, 
ImageBind, IDEFICS, PaliGemma, etc. It handles multiple modality inputs (text, 
images, audio) and implements task-specific processing for multimodal understanding
and generation tasks.
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipelineTemplate


class MultimodalPipelineTemplate(BasePipelineTemplate):
    """Template for multimodal pipelines."""
    
    def __init__(self):
        """Initialize the multimodal pipeline template."""
        super().__init__()
        self.pipeline_type = "multimodal"
        self.input_type = "multimodal"
        self.output_type = "multimodal"
        self.requires_preprocessing = True
        self.requires_postprocessing = True
        self.supports_batching = True
        self.max_batch_size = 4  # Smaller batch size due to memory requirements
    
    def get_import_statements(self) -> str:
        """Get multimodal pipeline import statements."""
        return """
# Multimodal pipeline imports
import os
import json
import numpy as np
import base64
from typing import List, Dict, Union, Any, Optional, Tuple
from PIL import Image
import io
import tempfile
"""
    
    def get_preprocessing_code(self, task_type: str) -> str:
        """Get multimodal preprocessing code for specific task types."""
        if task_type == "multimodal_classification":
            return """
# Preprocess for multimodal classification (FLAVA-like)
from PIL import Image
import base64
import io
import tempfile

# Initialize containers for different modality inputs
image_input = None
text_input = None
audio_input = None

# Handle different input types
if isinstance(text, dict):
    # Input is a dictionary with different modalities
    if "image" in text:
        image_input = text["image"]
    if "text" in text:
        text_input = text["text"]
    if "audio" in text:
        audio_input = text["audio"]
elif isinstance(text, tuple) and len(text) >= 2:
    # Input is a tuple of multiple modalities
    # Assume (image, text) or (image, text, audio)
    image_input = text[0]
    text_input = text[1]
    if len(text) >= 3:
        audio_input = text[2]
elif isinstance(text, str):
    # Check if it's a file path
    if os.path.exists(text):
        if text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            # It's an image file
            image_input = text
        elif text.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            # It's an audio file
            audio_input = text
        else:
            # Assume it's a text file
            with open(text, 'r') as f:
                text_input = f.read()
    else:
        # Assume it's a text string
        text_input = text

# If no inputs were provided, use defaults
if image_input is None and text_input is None and audio_input is None:
    # Default text input
    text_input = "This is a test input for multimodal classification."
    
    # Try to find a test image
    test_image_paths = [
        "test.jpg",
        os.path.join(os.path.dirname(__file__), "test.jpg"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")
    ]
    
    for path in test_image_paths:
        if os.path.exists(path):
            image_input = path
            break

# Process image input if available
image = None
if image_input is not None:
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

# Process audio input if available
audio_path = None
if audio_input is not None:
    if isinstance(audio_input, str):
        # Check if it's a file path
        if os.path.exists(audio_input) and audio_input.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            # It's an audio file path
            audio_path = audio_input
        else:
            # Try to decode as base64
            try:
                audio_data = base64.b64decode(audio_input)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    audio_path = temp_file.name
            except:
                raise ValueError("Could not process audio input as base64")
    elif isinstance(audio_input, bytes):
        # Raw audio bytes
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_input)
            audio_path = temp_file.name

# Prepare inputs for the model based on what's available
if image is not None and text_input is not None and audio_path is None:
    # Image and text input
    inputs = tokenizer(
        text=text_input,
        images=image,
        return_tensors="pt",
        padding=True
    )
elif image is not None and text_input is None and audio_path is None:
    # Image-only input
    inputs = tokenizer(images=image, return_tensors="pt")
elif image is None and text_input is not None and audio_path is None:
    # Text-only input
    inputs = tokenizer(text=text_input, return_tensors="pt", padding=True)
elif image is None and text_input is None and audio_path is not None:
    # Audio-only input
    inputs = tokenizer(audio_path, return_tensors="pt")
elif image is not None and text_input is not None and audio_path is not None:
    # All modalities
    # Note: This is model-specific and may need customization
    inputs = tokenizer(
        text=text_input,
        images=image,
        audio=audio_path,
        return_tensors="pt",
        padding=True
    )
else:
    # Fallback to empty inputs
    inputs = {}

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "multimodal_generation":
            return """
# Preprocess for multimodal generation (LLaVA-like)
from PIL import Image
import base64
import io
import tempfile

# Initialize containers for different modality inputs
image_input = None
text_input = None
audio_input = None

# Handle different input types
if isinstance(text, dict):
    # Input is a dictionary with different modalities
    if "image" in text:
        image_input = text["image"]
    if "text" in text:
        text_input = text["text"]
    if "audio" in text:
        audio_input = text["audio"]
    if "prompt" in text:
        # Alternative key for text
        text_input = text["prompt"]
elif isinstance(text, tuple) and len(text) >= 2:
    # Input is a tuple of multiple modalities
    # Assume (image, text) or (image, text, audio)
    image_input = text[0]
    text_input = text[1]
    if len(text) >= 3:
        audio_input = text[2]
elif isinstance(text, str):
    # Check if it's a file path
    if os.path.exists(text):
        if text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            # It's an image file
            image_input = text
            text_input = "What can you tell me about this image?"  # Default prompt
        elif text.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            # It's an audio file
            audio_input = text
            text_input = "What can you tell me about this audio?"  # Default prompt
        else:
            # Assume it's a text file
            with open(text, 'r') as f:
                text_input = f.read()
    else:
        # Assume it's a text string
        text_input = text

# If no inputs were provided, use defaults
if image_input is None and text_input is None and audio_input is None:
    # Default text input
    text_input = "Generate a creative response."
    
    # Try to find a test image
    test_image_paths = [
        "test.jpg",
        os.path.join(os.path.dirname(__file__), "test.jpg"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")
    ]
    
    for path in test_image_paths:
        if os.path.exists(path):
            image_input = path
            break

# Process image input if available
image = None
if image_input is not None:
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

# Process audio input if available
audio_path = None
if audio_input is not None:
    if isinstance(audio_input, str):
        # Check if it's a file path
        if os.path.exists(audio_input) and audio_input.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            # It's an audio file path
            audio_path = audio_input
        else:
            # Try to decode as base64
            try:
                audio_data = base64.b64decode(audio_input)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    audio_path = temp_file.name
            except:
                raise ValueError("Could not process audio input as base64")
    elif isinstance(audio_input, bytes):
        # Raw audio bytes
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_input)
            audio_path = temp_file.name

# Create generation config
generation_config = {}
if isinstance(text, dict):
    # Extract generation parameters if provided
    generation_params = ["max_length", "min_length", "top_k", "top_p", 
                         "temperature", "repetition_penalty", "do_sample"]
    for param in generation_params:
        if param in text:
            generation_config[param] = text[param]

# Prepare inputs for the model based on what's available
if image is not None and text_input is not None and audio_path is None:
    # Image and text input
    inputs = tokenizer(
        text=text_input,
        images=image,
        return_tensors="pt",
        padding=True
    )
elif image is not None and text_input is None and audio_path is None:
    # Image-only input with default prompt
    inputs = tokenizer(
        text="What can you tell me about this image?",
        images=image,
        return_tensors="pt"
    )
elif image is None and text_input is not None and audio_path is None:
    # Text-only input
    inputs = tokenizer(text=text_input, return_tensors="pt", padding=True)
elif image is None and text_input is None and audio_path is not None:
    # Audio-only input with default prompt
    inputs = tokenizer(
        text="What can you tell me about this audio?",
        audio=audio_path,
        return_tensors="pt"
    )
elif image is not None and text_input is not None and audio_path is not None:
    # All modalities
    # Note: This is model-specific and may need customization
    inputs = tokenizer(
        text=text_input,
        images=image,
        audio=audio_path,
        return_tensors="pt",
        padding=True
    )
else:
    # Fallback to empty inputs with default prompt
    inputs = tokenizer(text="Generate a creative response.", return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "multimodal_question_answering":
            return """
# Preprocess for multimodal question answering (PaliGemma-like)
from PIL import Image
import base64
import io
import tempfile

# Initialize containers for different modality inputs
image_input = None
question = None
audio_input = None

# Handle different input types
if isinstance(text, dict):
    # Input is a dictionary with image and question
    if "image" in text:
        image_input = text["image"]
    if "question" in text:
        question = text["question"]
    if "audio" in text:
        audio_input = text["audio"]
elif isinstance(text, tuple) and len(text) >= 2:
    # Input is a tuple of (image, question) or (audio, question)
    if isinstance(text[0], (str, Image.Image, bytes)) and isinstance(text[1], str):
        # Check if first element is likely an image or audio
        if isinstance(text[0], str):
            if os.path.exists(text[0]):
                if text[0].lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    image_input = text[0]
                elif text[0].lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_input = text[0]
        else:
            # Assume it's an image
            image_input = text[0]
        
        # Second element is the question
        question = text[1]
elif isinstance(text, str):
    # Check if it's a file path to an image
    if os.path.exists(text) and text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        # It's an image file, use default question
        image_input = text
        question = "What can you see in this image?"
    elif os.path.exists(text) and text.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        # It's an audio file, use default question
        audio_input = text
        question = "What can you hear in this audio?"
    else:
        # Assume it's a question without any media
        question = text

# If no inputs were provided, use defaults
if image_input is None and question is None and audio_input is None:
    # Default question
    question = "What is shown in the image?"
    
    # Try to find a test image
    test_image_paths = [
        "test.jpg",
        os.path.join(os.path.dirname(__file__), "test.jpg"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")
    ]
    
    for path in test_image_paths:
        if os.path.exists(path):
            image_input = path
            break

# Process image input if available
image = None
if image_input is not None:
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

# Process audio input if available
audio_path = None
if audio_input is not None:
    if isinstance(audio_input, str):
        # Check if it's a file path
        if os.path.exists(audio_input) and audio_input.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            # It's an audio file path
            audio_path = audio_input
        else:
            # Try to decode as base64
            try:
                audio_data = base64.b64decode(audio_input)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    audio_path = temp_file.name
            except:
                raise ValueError("Could not process audio input as base64")
    elif isinstance(audio_input, bytes):
        # Raw audio bytes
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_input)
            audio_path = temp_file.name

# Prepare inputs for the model based on what's available
if image is not None and question is not None and audio_path is None:
    # Image and question
    inputs = tokenizer(
        text=question,
        images=image,
        return_tensors="pt",
        padding=True
    )
elif image is None and question is not None and audio_path is not None:
    # Audio and question
    inputs = tokenizer(
        text=question,
        audio=audio_path,
        return_tensors="pt",
        padding=True
    )
elif image is None and question is not None and audio_path is None:
    # Question only
    inputs = tokenizer(text=question, return_tensors="pt", padding=True)
elif image is not None and question is None and audio_path is None:
    # Image only with default question
    inputs = tokenizer(
        text="What can you see in this image?",
        images=image,
        return_tensors="pt"
    )
elif image is None and question is None and audio_path is not None:
    # Audio only with default question
    inputs = tokenizer(
        text="What can you hear in this audio?",
        audio=audio_path,
        return_tensors="pt"
    )
elif image is not None and audio_path is not None:
    # Both image and audio with question
    q = question if question is not None else "What can you tell me about this media?"
    # Note: This is model-specific and may need customization
    inputs = tokenizer(
        text=q,
        images=image,
        audio=audio_path,
        return_tensors="pt",
        padding=True
    )
else:
    # Fallback to empty inputs
    inputs = tokenizer(text="No input provided. Please describe what you see.", return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "multimodal_retrieval":
            return """
# Preprocess for multimodal retrieval (FLAVA/ImageBind-like)
from PIL import Image
import base64
import io
import tempfile

# Initialize containers for different modality inputs
image_input = None
text_input = None
audio_input = None

# Handle different input types
if isinstance(text, dict):
    # Input is a dictionary with different modalities
    if "image" in text:
        image_input = text["image"]
    if "text" in text:
        text_input = text["text"]
    if "audio" in text:
        audio_input = text["audio"]
    if "query" in text:
        # Alternative key for text
        text_input = text["query"]
elif isinstance(text, tuple) and len(text) >= 2:
    # Input is a tuple of multiple modalities
    # Could be (image, text), (audio, text), etc.
    if isinstance(text[0], (str, Image.Image, bytes)):
        # First element could be image or audio
        if isinstance(text[0], str):
            if os.path.exists(text[0]):
                if text[0].lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    image_input = text[0]
                elif text[0].lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_input = text[0]
            else:
                # Assume it's text
                text_input = text[0]
        else:
            # Assume it's an image
            image_input = text[0]
    
    # Second element could be another modality
    if len(text) >= 2:
        if isinstance(text[1], str):
            if text_input is None:
                text_input = text[1]
elif isinstance(text, str):
    # Check if it's a file path
    if os.path.exists(text):
        if text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            # It's an image file
            image_input = text
        elif text.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            # It's an audio file
            audio_input = text
        else:
            # Assume it's a text file
            with open(text, 'r') as f:
                text_input = f.read()
    else:
        # Assume it's a text string
        text_input = text

# If no inputs were provided, use defaults
if image_input is None and text_input is None and audio_input is None:
    # Default text input
    text_input = "This is a test query for multimodal retrieval."

# Process image input if available
image = None
if image_input is not None:
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

# Process audio input if available
audio_path = None
if audio_input is not None:
    if isinstance(audio_input, str):
        # Check if it's a file path
        if os.path.exists(audio_input) and audio_input.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            # It's an audio file path
            audio_path = audio_input
        else:
            # Try to decode as base64
            try:
                audio_data = base64.b64decode(audio_input)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    audio_path = temp_file.name
            except:
                raise ValueError("Could not process audio input as base64")
    elif isinstance(audio_input, bytes):
        # Raw audio bytes
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_input)
            audio_path = temp_file.name

# Prepare inputs for the model based on what's available
if image is not None and text_input is None and audio_path is None:
    # Image-only input (for embedding extraction)
    inputs = tokenizer(images=image, return_tensors="pt")
elif image is None and text_input is not None and audio_path is None:
    # Text-only input (for embedding extraction)
    inputs = tokenizer(text=text_input, return_tensors="pt", padding=True)
elif image is None and text_input is None and audio_path is not None:
    # Audio-only input (for embedding extraction)
    inputs = tokenizer(audio=audio_path, return_tensors="pt")
elif image is not None and text_input is not None:
    # Image and text input (for similarity)
    inputs = tokenizer(
        text=text_input,
        images=image,
        return_tensors="pt",
        padding=True
    )
elif audio_path is not None and text_input is not None:
    # Audio and text input (for similarity)
    inputs = tokenizer(
        text=text_input,
        audio=audio_path,
        return_tensors="pt",
        padding=True
    )
elif image is not None and audio_path is not None:
    # Image and audio input (for cross-modal similarity)
    inputs = {
        "image": tokenizer(images=image, return_tensors="pt"),
        "audio": tokenizer(audio=audio_path, return_tensors="pt")
    }
    # Flatten the dictionary for device placement
    flat_inputs = {}
    for modality, modality_inputs in inputs.items():
        for k, v in modality_inputs.items():
            flat_inputs[f"{modality}_{k}"] = v
    inputs = flat_inputs
elif image is not None and text_input is not None and audio_path is not None:
    # All modalities (for cross-modal fusion)
    inputs = tokenizer(
        text=text_input,
        images=image,
        audio=audio_path,
        return_tensors="pt",
        padding=True
    )
else:
    # Fallback to empty inputs
    inputs = {}

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        else:
            # Default preprocessing for other multimodal tasks
            return """
# Default preprocessing for multimodal tasks
from PIL import Image
import base64
import io
import tempfile

# Initialize containers for different modality inputs
image_input = None
text_input = None
audio_input = None

# Handle different input types
if isinstance(text, dict):
    # Input is a dictionary with different modalities
    if "image" in text:
        image_input = text["image"]
    if "text" in text:
        text_input = text["text"]
    if "audio" in text:
        audio_input = text["audio"]
elif isinstance(text, tuple) and len(text) >= 2:
    # Input is a tuple of multiple modalities
    # Assume (image, text) or (image, text, audio)
    image_input = text[0]
    text_input = text[1]
    if len(text) >= 3:
        audio_input = text[2]
elif isinstance(text, str):
    # Check if it's a file path
    if os.path.exists(text):
        if text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            # It's an image file
            image_input = text
        elif text.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            # It's an audio file
            audio_input = text
        else:
            # Assume it's a text file
            with open(text, 'r') as f:
                text_input = f.read()
    else:
        # Assume it's a text string
        text_input = text

# If no inputs were provided, use defaults
if image_input is None and text_input is None and audio_input is None:
    # Default text input
    text_input = "This is a test input for the multimodal model."

# Process image input if available
image = None
if image_input is not None:
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

# Process audio input if available
audio_path = None
if audio_input is not None:
    if isinstance(audio_input, str):
        # Check if it's a file path
        if os.path.exists(audio_input) and audio_input.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            # It's an audio file path
            audio_path = audio_input
        else:
            # Try to decode as base64
            try:
                audio_data = base64.b64decode(audio_input)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    audio_path = temp_file.name
            except:
                raise ValueError("Could not process audio input as base64")
    elif isinstance(audio_input, bytes):
        # Raw audio bytes
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_input)
            audio_path = temp_file.name

# Prepare inputs for the model based on what's available
try:
    # Try to detect what modalities the model supports by inspecting tokenizer
    if hasattr(tokenizer, 'image_processor') and hasattr(tokenizer, 'tokenizer'):
        # Likely a vision-language model
        if image is not None and text_input is not None:
            inputs = tokenizer(
                text=text_input,
                images=image,
                return_tensors="pt", 
                padding=True
            )
        elif image is not None:
            inputs = tokenizer(images=image, return_tensors="pt")
        elif text_input is not None:
            inputs = tokenizer(text=text_input, return_tensors="pt", padding=True)
    elif hasattr(tokenizer, 'feature_extractor') and audio_path is not None:
        # Likely an audio model
        inputs = tokenizer(audio_path, return_tensors="pt")
    else:
        # Generic fallback
        if text_input is not None:
            inputs = tokenizer(text=text_input, return_tensors="pt", padding=True)
        else:
            # Empty inputs (model will likely error)
            inputs = {}
except Exception as e:
    print(f"Error preparing inputs: {e}")
    # Fallback to text-only
    if text_input is not None:
        inputs = tokenizer(text=text_input, return_tensors="pt", padding=True)
    else:
        inputs = tokenizer("No valid input provided", return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
    
    def get_postprocessing_code(self, task_type: str) -> str:
        """Get multimodal postprocessing code for specific task types."""
        if task_type == "multimodal_classification":
            return """
# Run inference for multimodal classification
with self.torch.no_grad():
    outputs = model(**inputs)

# Process classification outputs
if hasattr(outputs, "logits"):
    logits = outputs.logits
    probabilities = self.torch.nn.functional.softmax(logits, dim=-1)
    predictions = probabilities[0].cpu().tolist()
    
    # Get class labels if available
    id2label = getattr(model.config, 'id2label', None)
    if id2label:
        # Convert to more readable format
        top_indices = probabilities[0].cpu().argsort(descending=True)[:5].tolist()
        results = []
        for idx in top_indices:
            label = id2label.get(str(idx), f"CLASS_{idx}")
            score = probabilities[0][idx].item()
            results.append({"label": label, "score": score})
    else:
        # Just return raw probabilities
        results = {"probabilities": predictions}
elif hasattr(outputs, "image_embeds") and hasattr(outputs, "text_embeds"):
    # CLIP-like model with different embeddings
    # Calculate similarity between image and text embeddings
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    
    # Normalize embeddings
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Calculate similarity scores
    similarity = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)
    results = {"multimodal_similarity": similarity[0].cpu().tolist()}
else:
    # Generic handling
    results = {"outputs": str(outputs)}
"""
        elif task_type == "multimodal_generation":
            return """
# Run inference for multimodal generation
# Check if we have generation config parameters
generation_params = {}
if isinstance(text, dict):
    for param in ["max_length", "min_length", "top_k", "top_p", "temperature", "num_return_sequences"]:
        if param in text:
            generation_params[param] = text[param]

# Set default generation parameters if not provided
if "max_length" not in generation_params:
    generation_params["max_length"] = 256
if "temperature" not in generation_params:
    generation_params["temperature"] = 0.7
if "top_p" not in generation_params:
    generation_params["top_p"] = 0.9

with self.torch.no_grad():
    # Check if the model has a generate method
    if hasattr(model, "generate"):
        # Add extra args for generation
        generate_args = {**inputs, **generation_params}
        output_ids = model.generate(**generate_args)
        
        # Decode the generated output
        generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    else:
        # Fall back to regular forward pass
        outputs = model(**inputs)
        
        if hasattr(outputs, "logits"):
            # Try to generate from logits
            logits = outputs.logits
            output_ids = self.torch.argmax(logits, dim=-1)
            generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        else:
            # Generic handling
            generated_text = [str(outputs)]
"""
        elif task_type == "multimodal_question_answering":
            return """
# Run inference for multimodal question answering
with self.torch.no_grad():
    # Check if the model has a generate method for question answering
    if hasattr(model, "generate"):
        # Add generation parameters
        generation_params = {"max_length": 256, "min_length": 1, "temperature": 0.7}
        
        # Override with user parameters if provided
        if isinstance(text, dict):
            for param in ["max_length", "min_length", "temperature", "top_p", "num_beams"]:
                if param in text:
                    generation_params[param] = text[param]
        
        # Add generation parameters to inputs
        generate_args = {**inputs, **generation_params}
        
        # Generate answer
        output_ids = model.generate(**generate_args)
        
        # Decode the generated answer
        answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    else:
        # Fall back to regular forward pass
        outputs = model(**inputs)
        
        if hasattr(outputs, "logits"):
            # Try to extract answer from logits
            logits = outputs.logits
            output_ids = self.torch.argmax(logits, dim=-1)
            answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        else:
            # Generic handling
            answers = [str(outputs)]
"""
        elif task_type == "multimodal_retrieval":
            return """
# Run inference for multimodal retrieval
with self.torch.no_grad():
    outputs = model(**inputs)

# Process embeddings based on what's available
embeddings = {}

if hasattr(outputs, "image_embeds"):
    # Extract image embeddings
    embeddings["image"] = outputs.image_embeds.cpu().numpy().tolist()

if hasattr(outputs, "text_embeds"):
    # Extract text embeddings
    embeddings["text"] = outputs.text_embeds.cpu().numpy().tolist()

if hasattr(outputs, "audio_embeds"):
    # Extract audio embeddings
    embeddings["audio"] = outputs.audio_embeds.cpu().numpy().tolist()

if hasattr(outputs, "multimodal_embeds"):
    # Extract fused multimodal embeddings
    embeddings["multimodal"] = outputs.multimodal_embeds.cpu().numpy().tolist()

# If no structured embeddings found, try to extract from generic outputs
if not embeddings and hasattr(outputs, "last_hidden_state"):
    # Use mean pooling as a fallback
    embeddings["generic"] = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()

# Check if similarity scores can be calculated
if "image" in embeddings and "text" in embeddings:
    # Calculate cosine similarity
    import numpy as np
    from scipy.spatial.distance import cosine
    
    img_emb = np.array(embeddings["image"][0] if isinstance(embeddings["image"], list) else embeddings["image"])
    txt_emb = np.array(embeddings["text"][0] if isinstance(embeddings["text"], list) else embeddings["text"])
    
    # Normalize embeddings
    img_emb = img_emb / np.linalg.norm(img_emb)
    txt_emb = txt_emb / np.linalg.norm(txt_emb)
    
    # Compute similarity (1 - cosine distance)
    similarity = 1 - cosine(img_emb, txt_emb)
    embeddings["image_text_similarity"] = float(similarity)
"""
        else:
            # Default postprocessing for multimodal tasks
            return """
# Default inference for multimodal tasks
with self.torch.no_grad():
    # Try to run the model with the inputs
    try:
        outputs = model(**inputs)
        
        # Try to identify the type of outputs
        if hasattr(outputs, "logits"):
            # Classification-like output
            logits = outputs.logits
            
            if hasattr(model.config, 'id2label') and hasattr(model.config, 'label2id'):
                # It's likely a classification model
                probabilities = self.torch.nn.functional.softmax(logits, dim=-1)
                predictions = probabilities[0].cpu().tolist()
                
                # Get class labels
                id2label = model.config.id2label
                top_indices = probabilities[0].cpu().argsort(descending=True)[:5].tolist()
                results = []
                for idx in top_indices:
                    label = id2label.get(str(idx), f"CLASS_{idx}")
                    score = probabilities[0][idx].item()
                    results.append({"label": label, "score": score})
                
                result = {
                    "type": "classification",
                    "predictions": results
                }
            else:
                # Generic logits
                result = {
                    "type": "logits",
                    "values": logits[0].cpu().tolist()
                }
        elif hasattr(outputs, "last_hidden_state"):
            # Embedding-like output
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
            result = {
                "type": "embeddings",
                "values": embeddings[0] if embeddings else None
            }
        elif hasattr(outputs, "image_embeds") or hasattr(outputs, "text_embeds"):
            # Multimodal embeddings
            result = {
                "type": "multimodal_embeddings",
                "modalities": []
            }
            
            if hasattr(outputs, "image_embeds"):
                result["modalities"].append("image")
            if hasattr(outputs, "text_embeds"):
                result["modalities"].append("text")
            if hasattr(outputs, "audio_embeds"):
                result["modalities"].append("audio")
        else:
            # Generic output
            result = {
                "type": "generic",
                "data": str(outputs)
            }
    except Exception as e:
        # Handle errors gracefully
        result = {
            "type": "error",
            "error": str(e)
        }
"""
    
    def get_result_formatting_code(self, task_type: str) -> str:
        """Get multimodal result formatting code for specific task types."""
        if task_type == "multimodal_classification":
            return """
# Format results for multimodal classification
if "multimodal_similarity" in results:
    # CLIP-like similarity results
    return {
        "success": True,
        "multimodal_classification": {
            "similarity": results["multimodal_similarity"],
            "type": "cross_modal_similarity"
        },
        "device": device,
        "hardware": hardware_label
    }
elif "label" in results.get("results", [{}])[0]:
    # Classification with labels
    return {
        "success": True,
        "multimodal_classification": {
            "predictions": results["results"],
            "top_class": results["results"][0]["label"],
            "top_score": results["results"][0]["score"],
            "type": "labeled_classification"
        },
        "device": device,
        "hardware": hardware_label
    }
else:
    # Generic classification result
    return {
        "success": True,
        "multimodal_classification": results,
        "device": device,
        "hardware": hardware_label
    }
"""
        elif task_type == "multimodal_generation":
            return """
# Format results for multimodal generation
return {
    "success": True,
    "multimodal_generation": {
        "generated_text": generated_text[0] if generated_text else "",
        "all_generated_texts": generated_text,
        "input_modalities": [
            "image" if "image" in locals() and image is not None else None,
            "text" if "text_input" in locals() and text_input is not None else None,
            "audio" if "audio_path" in locals() and audio_path is not None else None
        ]
    },
    "device": device,
    "hardware": hardware_label
}
"""
        elif task_type == "multimodal_question_answering":
            return """
# Format results for multimodal question answering
return {
    "success": True,
    "multimodal_qa": {
        "question": question if question else "No question provided",
        "answer": answers[0] if answers else "",
        "all_answers": answers,
        "input_modalities": [
            "image" if "image" in locals() and image is not None else None,
            "audio" if "audio_path" in locals() and audio_path is not None else None
        ]
    },
    "device": device,
    "hardware": hardware_label
}
"""
        elif task_type == "multimodal_retrieval":
            return """
# Format results for multimodal retrieval
return {
    "success": True,
    "multimodal_retrieval": {
        "embeddings": embeddings,
        "modalities_present": list(embeddings.keys()),
        "dimensionality": {
            modality: len(embedding) if isinstance(embedding, list) else (
                len(embedding[0]) if isinstance(embedding, list) and embedding else "unknown"
            ) for modality, embedding in embeddings.items() if modality not in ["image_text_similarity"]
        }
    },
    "device": device,
    "hardware": hardware_label
}
"""
        else:
            # Default result formatting for multimodal tasks
            return """
# Default result formatting for multimodal tasks
return {
    "success": True,
    "multimodal_results": result,
    "device": device,
    "hardware": hardware_label
}
"""
    
    def get_mock_input_code(self) -> str:
        """Get multimodal mock input code."""
        return """
# Mock multimodal input
from PIL import Image
import numpy as np
import tempfile
import os

# Create a mock image
mock_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
mock_image_path = os.path.join(tempfile.gettempdir(), "mock_image.jpg")
mock_image.save(mock_image_path)

# Create mock text input
mock_text = "This is a mock multimodal input for testing."

# Create mock audio input
mock_audio_path = os.path.join(tempfile.gettempdir(), "mock_audio.wav")
sample_rate = 16000
duration = 2  # seconds
sample_count = sample_rate * duration
mock_audio_data = np.sin(2 * np.pi * 440 * np.arange(sample_count) / sample_rate).astype(np.float32)

# Save mock audio to WAV file
import wave
with wave.open(mock_audio_path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes((mock_audio_data * 32767).astype(np.int16).tobytes())

# Create combined multimodal input
mock_input = {
    "image": mock_image,
    "text": mock_text,
    "audio": mock_audio_path
}
"""
    
    def get_mock_output_code(self) -> str:
        """Get multimodal mock output code."""
        return """
# Mock multimodal output
mock_output = {
    "success": True,
    "multimodal_results": {
        "type": "multimodal_classification",
        "predictions": [
            {"label": "person", "score": 0.85},
            {"label": "building", "score": 0.10},
            {"label": "nature", "score": 0.05}
        ],
        "embeddings": {
            "image": [0.1, 0.2, 0.3, 0.4, 0.5],
            "text": [0.2, 0.3, 0.4, 0.5, 0.6],
            "multimodal": [0.15, 0.25, 0.35, 0.45, 0.55]
        }
    },
    "device": "cpu",
    "hardware": "mock"
}
"""
    
    def get_pipeline_utilities(self) -> str:
        """Get multimodal utility functions."""
        return """
# Multimodal pipeline utilities
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

def encode_audio_base64(audio_path):
    # Encode an audio file to base64 string
    if not os.path.exists(audio_path):
        return None
        
    with open(audio_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def normalize_embedding(embedding):
    # Normalize an embedding vector to unit length
    import numpy as np
    if isinstance(embedding, list):
        embedding = np.array(embedding)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return (embedding / norm).tolist()
    return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

def compute_similarity(embedding1, embedding2):
    # Compute cosine similarity between two embeddings
    import numpy as np
    from scipy.spatial.distance import cosine
    
    if isinstance(embedding1, list):
        embedding1 = np.array(embedding1)
    if isinstance(embedding2, list):
        embedding2 = np.array(embedding2)
        
    # Normalize embeddings
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    # Compute similarity (1 - cosine distance)
    return 1 - cosine(embedding1, embedding2)
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check multimodal pipeline compatibility with architecture type."""
        # Multimodal pipeline is compatible with multimodal architectures
        return arch_type in [
            "multimodal",
            "vision-encoder-text-decoder",  # Some vision-text models can work with this pipeline too
            "speech"  # Some speech models can be treated as multimodal
        ]
    
    def is_compatible_with_task(self, task_type: str) -> bool:
        """Check multimodal pipeline compatibility with task type."""
        # Multimodal pipeline is compatible with multimodal tasks
        return task_type in [
            "multimodal_classification",
            "multimodal_generation",
            "multimodal_question_answering",
            "multimodal_retrieval",
            "image_text_matching",  # For backward compatibility
            "visual_question_answering",  # For backward compatibility
            "audio_classification",  # For some multimodal models
            "speech_recognition"  # For some multimodal models
        ]