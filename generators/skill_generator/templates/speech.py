#!/usr/bin/env python3
"""
Speech Architecture Template

This module provides the architecture template for speech models like Whisper, Wav2Vec2, etc.
"""

from typing import Dict, Any, List
from .base_architecture import BaseArchitectureTemplate

class SpeechArchitectureTemplate(BaseArchitectureTemplate):
    """Template for speech architecture models like Whisper, Wav2Vec2, etc."""
    
    def __init__(self):
        """Initialize the speech architecture template."""
        super().__init__()
        self.architecture_type = "speech"
        self.architecture_name = "Speech Architecture"
        self.supported_task_types = ["speech_recognition", "audio_classification", "text_to_speech"]
        self.default_task_type = "speech_recognition"
        self.model_description = "This is a model designed for speech and audio processing tasks."
        self.hidden_size = 768  # Default hidden size, varies by model
        self.test_input = "test.wav"  # Default test file
    
    def get_model_class(self, task_type: str) -> str:
        """Get speech model class for task type."""
        if task_type == "speech_recognition":
            return "self.transformers.AutoModelForSpeechSeq2Seq"
        elif task_type == "audio_classification":
            return "self.transformers.AutoModelForAudioClassification"
        elif task_type == "text_to_speech":
            return "self.transformers.AutoModelForTextToSpeech"
        else:
            return "self.transformers.AutoModel"
    
    def get_processor_class(self, task_type: str) -> str:
        """Get speech processor class for task type."""
        return "self.transformers.AutoProcessor"
    
    def get_input_processing_code(self, task_type: str) -> str:
        """Get speech input processing code."""
        return """
# Load audio file
import os
if not os.path.exists(text):
    # If 'text' is not a valid audio file path, assume it's a test file name
    # Check for test audio in common locations
    test_paths = [
        text,
        os.path.join(os.path.dirname(__file__), text),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), text),
        "test.wav", 
        "test.mp3",
        os.path.join(os.path.dirname(__file__), "test.wav"),
        os.path.join(os.path.dirname(__file__), "test.mp3"),
    ]
    
    audio_path = None
    for path in test_paths:
        if os.path.exists(path):
            audio_path = path
            break
    
    if audio_path is None:
        raise ValueError(f"Audio file not found: {text}")
    
    text = audio_path

# Process audio for the speech model
try:
    import librosa
    audio, sr = librosa.load(text, sr=16000)
except:
    # Fallback if librosa is not available
    try:
        import soundfile as sf
        audio, sr = sf.read(text)
    except:
        # Last resort fallback
        import numpy as np
        audio = np.zeros(16000)  # 1 second of silence
        sr = 16000

# Process audio with processor
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
    
    def get_output_processing_code(self, task_type: str) -> str:
        """Get speech output processing code."""
        if task_type == "speech_recognition":
            return """
# Process output for speech recognition
with self.torch.no_grad():
    outputs = model.generate(inputs["input_features"])

# Decode the generated IDs
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
"""
        elif task_type == "audio_classification":
            return """
# Process output for audio classification
with self.torch.no_grad():
    outputs = model(**inputs)

# Get predicted class
logits = outputs.logits
predicted_class_id = self.torch.argmax(logits, dim=-1).item()

# Convert ID to label
if hasattr(model.config, "id2label"):
    result = model.config.id2label[predicted_class_id]
else:
    result = f"CLASS_{predicted_class_id}"
"""
        elif task_type == "text_to_speech":
            return """
# Process output for text-to-speech
with self.torch.no_grad():
    outputs = model(**inputs)

# Extract audio array
if hasattr(outputs, "waveform"):
    audio_array = outputs.waveform[0].cpu().numpy()
    result = {"audio": audio_array, "sampling_rate": model.config.sampling_rate}
else:
    # Generic fallback
    result = outputs
"""
        else:
            return """
# Generic output processing
with self.torch.no_grad():
    outputs = model(**inputs)

# Extract relevant information from outputs
result = outputs
"""
    
    def get_mock_processor_code(self) -> str:
        """Get speech mock processor code."""
        return """
def mock_tokenize(audio, sampling_rate=16000, return_tensors="pt", **kwargs):
    if hasattr(self, 'torch'):
        torch = self.torch
    else:
        import torch
    
    # Mock audio input format
    if isinstance(audio, str):
        # Audio path was provided
        batch_size = 1
    elif hasattr(audio, "shape"):
        # Audio array was provided
        batch_size = 1
    else:
        # Assume batch of audio paths
        batch_size = len(audio)
    
    # Create mock input features
    input_features = torch.rand(batch_size, 80, 3000)  # Common mel spectrogram size
    attention_mask = torch.ones(batch_size, 3000)
    
    return {
        "input_features": input_features,
        "attention_mask": attention_mask
    }
"""
    
    def get_mock_output_code(self) -> str:
        """Get speech mock output code."""
        return """
result = MagicMock()
result.logits = torch.rand((batch_size, sequence_length, hidden_size))
result.generation_outputs = torch.randint(0, 1000, (batch_size, 20))
return result
"""
    
    def get_compatibility_matrix(self) -> Dict[str, bool]:
        """Get speech hardware compatibility matrix."""
        return {
            "cpu": True,
            "cuda": True,
            "rocm": True,
            "mps": True,
            "openvino": True,
            "qnn": False  # Limited support for speech models in Qualcomm QNN
        }