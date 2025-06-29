#!/usr/bin/env python3
"""
Model helper utilities for IPFS Accelerate tests.

This module provides utilities for loading and preparing models and inputs.
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_name: str, device: str = 'cpu') -> Tuple[Any, Any]:
    """
    Load a model and tokenizer/processor from HuggingFace.
    
    Args:
        model_name: Name of the model
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer/processor)
    """
    try:
        # Determine model type
        if any(kw in model_name.lower() for kw in ['bert', 't5', 'gpt', 'llama']):
            return load_text_model(model_name, device)
        elif any(kw in model_name.lower() for kw in ['vit', 'resnet', 'deit']):
            return load_vision_model(model_name, device)
        elif any(kw in model_name.lower() for kw in ['whisper', 'wav2vec']):
            return load_audio_model(model_name, device)
        elif any(kw in model_name.lower() for kw in ['clip']):
            return load_multimodal_model(model_name, device)
        else:
            # Default to text model
            logger.warning(f"Unknown model type for {model_name}, defaulting to text model")
            return load_text_model(model_name, device)
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return None, None


def load_text_model(model_name: str, device: str = 'cpu') -> Tuple[Any, Any]:
    """
    Load a text model and tokenizer from HuggingFace.
    
    Args:
        model_name: Name of the model
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        
        logger.info(f"Loading text model {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        
        logger.info(f"Successfully loaded {model_name}")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading text model {model_name}: {e}")
        return None, None


def load_vision_model(model_name: str, device: str = 'cpu') -> Tuple[Any, Any]:
    """
    Load a vision model and feature extractor from HuggingFace.
    
    Args:
        model_name: Name of the model
        device: Device to load the model on
        
    Returns:
        Tuple of (model, feature_extractor)
    """
    try:
        from transformers import AutoImageProcessor, AutoModel
        
        logger.info(f"Loading vision model {model_name}")
        
        # Load feature extractor
        feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        
        # Load model
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        
        logger.info(f"Successfully loaded {model_name}")
        
        return model, feature_extractor
    except Exception as e:
        logger.error(f"Error loading vision model {model_name}: {e}")
        return None, None


def load_audio_model(model_name: str, device: str = 'cpu') -> Tuple[Any, Any]:
    """
    Load an audio model and processor from HuggingFace.
    
    Args:
        model_name: Name of the model
        device: Device to load the model on
        
    Returns:
        Tuple of (model, processor)
    """
    try:
        from transformers import AutoProcessor, AutoModel
        
        logger.info(f"Loading audio model {model_name}")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Load model
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        
        logger.info(f"Successfully loaded {model_name}")
        
        return model, processor
    except Exception as e:
        logger.error(f"Error loading audio model {model_name}: {e}")
        return None, None


def load_multimodal_model(model_name: str, device: str = 'cpu') -> Tuple[Any, Any]:
    """
    Load a multimodal model and processor from HuggingFace.
    
    Args:
        model_name: Name of the model
        device: Device to load the model on
        
    Returns:
        Tuple of (model, processor)
    """
    try:
        from transformers import AutoProcessor, AutoModel
        
        logger.info(f"Loading multimodal model {model_name}")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Load model
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        
        logger.info(f"Successfully loaded {model_name}")
        
        return model, processor
    except Exception as e:
        logger.error(f"Error loading multimodal model {model_name}: {e}")
        return None, None


def get_sample_inputs_for_model(model_name: str, tokenizer_or_processor: Any) -> Dict[str, Any]:
    """
    Get sample inputs for a model.
    
    Args:
        model_name: Name of the model
        tokenizer_or_processor: Tokenizer or processor for the model
        
    Returns:
        Dictionary with model inputs
    """
    try:
        # Determine model type
        if any(kw in model_name.lower() for kw in ['bert', 't5', 'gpt', 'llama']):
            return get_sample_text_inputs(tokenizer_or_processor)
        elif any(kw in model_name.lower() for kw in ['vit', 'resnet', 'deit']):
            return get_sample_vision_inputs(tokenizer_or_processor)
        elif any(kw in model_name.lower() for kw in ['whisper', 'wav2vec']):
            return get_sample_audio_inputs(tokenizer_or_processor)
        elif any(kw in model_name.lower() for kw in ['clip']):
            return get_sample_multimodal_inputs(tokenizer_or_processor)
        else:
            # Default to text model
            logger.warning(f"Unknown model type for {model_name}, defaulting to text inputs")
            return get_sample_text_inputs(tokenizer_or_processor)
    except Exception as e:
        logger.error(f"Error getting sample inputs for {model_name}: {e}")
        return {}


def get_sample_text_inputs(tokenizer) -> Dict[str, Any]:
    """
    Get sample inputs for a text model.
    
    Args:
        tokenizer: Tokenizer for the model
        
    Returns:
        Dictionary with model inputs
    """
    try:
        # Sample text
        text = "This is a sample input for testing the model."
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        
        return inputs
    except Exception as e:
        logger.error(f"Error getting sample text inputs: {e}")
        return {}


def get_sample_vision_inputs(processor) -> Dict[str, Any]:
    """
    Get sample inputs for a vision model.
    
    Args:
        processor: Feature extractor for the model
        
    Returns:
        Dictionary with model inputs
    """
    try:
        from PIL import Image
        import numpy as np
        
        # Create a sample image if none is available
        if os.path.exists("test.jpg"):
            image = Image.open("test.jpg")
        else:
            # Create a dummy image
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        
        return inputs
    except Exception as e:
        logger.error(f"Error getting sample vision inputs: {e}")
        return {}


def get_sample_audio_inputs(processor) -> Dict[str, Any]:
    """
    Get sample inputs for an audio model.
    
    Args:
        processor: Processor for the model
        
    Returns:
        Dictionary with model inputs
    """
    try:
        import numpy as np
        
        # Create a sample audio if none is available
        if os.path.exists("test.wav"):
            import librosa
            audio, _ = librosa.load("test.wav", sr=16000)
        elif os.path.exists("test.mp3"):
            import librosa
            audio, _ = librosa.load("test.mp3", sr=16000)
        else:
            # Create a dummy audio
            duration = 3  # seconds
            sampling_rate = 16000
            t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Process audio
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        
        return inputs
    except Exception as e:
        logger.error(f"Error getting sample audio inputs: {e}")
        return {}


def get_sample_multimodal_inputs(processor) -> Dict[str, Any]:
    """
    Get sample inputs for a multimodal model.
    
    Args:
        processor: Processor for the model
        
    Returns:
        Dictionary with model inputs
    """
    try:
        from PIL import Image
        import numpy as np
        
        # Create a sample image if none is available
        if os.path.exists("test.jpg"):
            image = Image.open("test.jpg")
        else:
            # Create a dummy image
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Sample text
        text = "A picture of a cat"
        
        # Process inputs
        inputs = processor(text=text, images=image, return_tensors="pt")
        
        return inputs
    except Exception as e:
        logger.error(f"Error getting sample multimodal inputs: {e}")
        return {}