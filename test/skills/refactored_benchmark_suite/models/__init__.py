"""
Model-specific adapters for benchmarking.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("benchmark.models")

class ModelAdapter:
    """Base class for model-specific adapters."""
    
    def __init__(self, model_id: str, task: Optional[str] = None):
        """
        Initialize a model adapter.
        
        Args:
            model_id: HuggingFace model ID
            task: Model task
        """
        self.model_id = model_id
        self.task = task
    
    def load_model(self, device: Any = None) -> Any:
        """
        Load the model on the specified device.
        
        Args:
            device: Device to load the model on
            
        Returns:
            Loaded model
        """
        raise NotImplementedError("Subclasses must implement load_model")
    
    def prepare_inputs(self, batch_size: int, sequence_length: int) -> Any:
        """
        Prepare inputs for the model.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            
        Returns:
            Model inputs
        """
        raise NotImplementedError("Subclasses must implement prepare_inputs")

def get_model_adapter(model_id: str, task: Optional[str] = None) -> ModelAdapter:
    """
    Get the appropriate model adapter for a given model ID and task.
    
    Args:
        model_id: HuggingFace model ID
        task: Model task
        
    Returns:
        ModelAdapter instance
    """
    # Import here to avoid circular imports
    from .text_models import TextModelAdapter
    from .vision_models import VisionModelAdapter
    from .speech_models import SpeechModelAdapter
    from .multimodal_models import MultimodalModelAdapter
    
    # Determine model type based on model ID and task
    if not task:
        # Try to infer task from model ID
        task = _infer_task_from_model_id(model_id)
    
    # Select appropriate adapter based on task
    if task in ["text-generation", "fill-mask", "text-classification", "token-classification", "question-answering"]:
        return TextModelAdapter(model_id, task)
    elif task in ["image-classification", "object-detection", "semantic-segmentation", "instance-segmentation"]:
        return VisionModelAdapter(model_id, task)
    elif task in ["automatic-speech-recognition", "audio-classification", "audio-to-audio", "text-to-speech"]:
        return SpeechModelAdapter(model_id, task)
    elif task in ["image-to-text", "text-to-image", "visual-question-answering"]:
        return MultimodalModelAdapter(model_id, task)
    else:
        # Default to text model adapter
        logger.warning(f"Unknown task '{task}' for model {model_id}, using default text model adapter")
        return TextModelAdapter(model_id, task)

def _infer_task_from_model_id(model_id: str) -> str:
    """
    Infer the task from model ID.
    
    Args:
        model_id: HuggingFace model ID
        
    Returns:
        Inferred task
    """
    model_id_lower = model_id.lower()
    
    # Vision models
    if any(name in model_id_lower for name in ["vit", "resnet", "efficientnet", "convnext", "swin", "beit", "deit"]):
        return "image-classification"
    
    # Text generation models
    if any(name in model_id_lower for name in ["gpt", "llama", "bloom", "opt", "falcon", "mistral", "llm"]):
        return "text-generation"
    
    # Masked language models
    if any(name in model_id_lower for name in ["bert", "roberta", "albert", "electra", "deberta"]):
        return "fill-mask"
    
    # Seq2seq models
    if any(name in model_id_lower for name in ["t5", "bart", "pegasus", "marian"]):
        return "text2text-generation"
    
    # Speech models
    if any(name in model_id_lower for name in ["wav2vec", "hubert", "whisper", "speecht5"]):
        return "automatic-speech-recognition"
    
    # Multimodal models
    if any(name in model_id_lower for name in ["clip", "blip", "vilt", "flava"]):
        return "image-to-text"
    
    # Default to text generation
    return "text-generation"