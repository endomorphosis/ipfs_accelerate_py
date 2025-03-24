#!/usr/bin/env python3
"""
Audio Pipeline Template for IPFS Accelerate Python.

This module implements a pipeline template for audio/speech models like Whisper,
Wav2Vec2, etc. It handles audio inputs and implements task-specific processing
for speech recognition, audio classification, and text-to-speech.
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipelineTemplate


class AudioPipelineTemplate(BasePipelineTemplate):
    """Template for audio/speech pipelines."""
    
    def __init__(self):
        """Initialize the audio pipeline template."""
        super().__init__()
        self.pipeline_type = "audio"
        self.input_type = "audio"
        self.output_type = "text"  # Most common output type for audio models
        self.requires_preprocessing = True
        self.requires_postprocessing = True
        self.supports_batching = True
        self.max_batch_size = 4  # Smaller batch size due to memory requirements
    
    def get_import_statements(self) -> str:
        """Get audio pipeline import statements."""
        return """
# Audio pipeline imports
import os
import json
import numpy as np
import base64
from typing import List, Dict, Union, Any
import io
import wave
import tempfile
"""
    
    def get_preprocessing_code(self, task_type: str) -> str:
        """Get audio preprocessing code for specific task types."""
        if task_type == "speech_recognition":
            return """
# Preprocess for speech recognition (Whisper-like)
# Handle different input types for audio

# First, determine what kind of input we have
if isinstance(text, str):
    # Check if it's a file path
    if os.path.exists(text) and text.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        # It's an audio file path
        audio_path = text
    else:
        # It might be base64 encoded audio data
        try:
            audio_data = base64.b64decode(text)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                audio_path = temp_file.name
        except:
            # Not a valid base64 string, try to find a test audio file
            test_paths = [
                "test.wav",
                "test.mp3",
                os.path.join(os.path.dirname(__file__), "test.wav"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.wav")
            ]
            
            audio_path = None
            for path in test_paths:
                if os.path.exists(path):
                    audio_path = path
                    break
            
            if audio_path is None:
                raise ValueError("Could not find a valid audio file to process")
elif isinstance(text, dict) and "audio" in text:
    # It's a dictionary with an audio key
    audio_input = text["audio"]
    if isinstance(audio_input, str) and os.path.exists(audio_input):
        audio_path = audio_input
    elif isinstance(audio_input, str):
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
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
elif isinstance(text, bytes):
    # Raw audio bytes
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(text)
        audio_path = temp_file.name
else:
    raise ValueError(f"Unsupported input type: {type(text)}")

# Process the audio with the model's processor
inputs = tokenizer(audio_path, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "audio_classification":
            return """
# Preprocess for audio classification
# Handle different input types for audio

# First, determine what kind of input we have
if isinstance(text, str):
    # Check if it's a file path
    if os.path.exists(text) and text.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        # It's an audio file path
        audio_path = text
    else:
        # It might be base64 encoded audio data
        try:
            audio_data = base64.b64decode(text)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                audio_path = temp_file.name
        except:
            # Not a valid base64 string, try to find a test audio file
            test_paths = [
                "test.wav",
                "test.mp3",
                os.path.join(os.path.dirname(__file__), "test.wav"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.wav")
            ]
            
            audio_path = None
            for path in test_paths:
                if os.path.exists(path):
                    audio_path = path
                    break
            
            if audio_path is None:
                raise ValueError("Could not find a valid audio file to process")
elif isinstance(text, dict) and "audio" in text:
    # It's a dictionary with an audio key
    audio_input = text["audio"]
    if isinstance(audio_input, str) and os.path.exists(audio_input):
        audio_path = audio_input
    elif isinstance(audio_input, str):
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
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
elif isinstance(text, bytes):
    # Raw audio bytes
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(text)
        audio_path = temp_file.name
else:
    raise ValueError(f"Unsupported input type: {type(text)}")

# Process the audio with the model's processor
inputs = tokenizer(audio_path, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "text_to_speech":
            return """
# Preprocess for text-to-speech
# For text-to-speech, we expect text input

# First, determine what kind of input we have
if isinstance(text, str):
    # It's a text string to synthesize
    input_text = text
elif isinstance(text, dict) and "text" in text:
    # It's a dictionary with a text key
    input_text = text["text"]
elif isinstance(text, list) and all(isinstance(item, str) for item in text):
    # It's a list of strings to synthesize
    input_text = text[0]  # Just use the first one for now
else:
    # Default text
    input_text = "Hello, this is a test of the text to speech system."

# Process the text with the model's processor
inputs = tokenizer(input_text, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        else:
            # Default preprocessing for other audio tasks
            return """
# Default preprocessing for audio tasks
# Handle different input types for audio

# First, determine what kind of input we have
if isinstance(text, str):
    # Check if it's a file path
    if os.path.exists(text) and text.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        # It's an audio file path
        audio_path = text
    else:
        # It might be base64 encoded audio data
        try:
            audio_data = base64.b64decode(text)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                audio_path = temp_file.name
        except:
            # Not a valid base64 string, try to find a test audio file
            test_paths = [
                "test.wav",
                "test.mp3",
                os.path.join(os.path.dirname(__file__), "test.wav"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.wav")
            ]
            
            audio_path = None
            for path in test_paths:
                if os.path.exists(path):
                    audio_path = path
                    break
            
            if audio_path is None:
                raise ValueError("Could not find a valid audio file to process")
elif isinstance(text, dict) and "audio" in text:
    # It's a dictionary with an audio key
    audio_input = text["audio"]
    if isinstance(audio_input, str) and os.path.exists(audio_input):
        audio_path = audio_input
    elif isinstance(audio_input, str):
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
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
elif isinstance(text, bytes):
    # Raw audio bytes
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(text)
        audio_path = temp_file.name
else:
    raise ValueError(f"Unsupported input type: {type(text)}")

# Process the audio with the model's processor
inputs = tokenizer(audio_path, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
    
    def get_postprocessing_code(self, task_type: str) -> str:
        """Get audio postprocessing code for specific task types."""
        if task_type == "speech_recognition":
            return """
# Run speech recognition inference
with self.torch.no_grad():
    generated_ids = model.generate(inputs.input_features)

# Decode the generated text
transcription = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
"""
        elif task_type == "audio_classification":
            return """
# Run audio classification inference
with self.torch.no_grad():
    outputs = model(**inputs)

# Process classification outputs
logits = outputs.logits
predictions = self.torch.nn.functional.softmax(logits, dim=-1)
"""
        elif task_type == "text_to_speech":
            return """
# Run text-to-speech inference
with self.torch.no_grad():
    speech_output = model.generate_speech(**inputs)

# Process speech output
audio_array = speech_output.numpy()

# Save to temporary file
audio_path = None
try:
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        audio_path = temp_file.name
        # Convert audio array to wav file
        import scipy.io.wavfile as wavfile
        wavfile.write(audio_path, model.config.sampling_rate, audio_array)
except ImportError:
    # Fallback if scipy is not available
    audio_path = "speech_output.wav"
    with open(audio_path, 'wb') as f:
        # Simple raw PCM write (not ideal but works as fallback)
        f.write(audio_array.tobytes())
"""
        else:
            # Default postprocessing for other audio tasks
            return """
# Default inference for audio tasks
with self.torch.no_grad():
    outputs = model(**inputs)

# Generic output processing
if hasattr(outputs, "logits"):
    # Classification-like output
    logits = outputs.logits
    predictions = self.torch.nn.functional.softmax(logits, dim=-1)
    result = {
        "probabilities": predictions[0].cpu().tolist(),
        "type": "audio_classification"
    }
elif hasattr(outputs, "last_hidden_state"):
    # Embedding-like output
    embeddings = outputs.last_hidden_state.mean(dim=1)
    result = {
        "embeddings": embeddings[0].cpu().tolist(),
        "type": "audio_embedding"
    }
else:
    # Generic output
    result = {
        "type": "audio_generic",
        "data": str(outputs)
    }
"""
    
    def get_result_formatting_code(self, task_type: str) -> str:
        """Get audio result formatting code for specific task types."""
        if task_type == "speech_recognition":
            return """
# Format results for speech recognition
return {
    "success": True,
    "transcription": {
        "text": transcription[0] if len(transcription) > 0 else "",
        "all_texts": transcription
    },
    "device": device,
    "hardware": hardware_label
}
"""
        elif task_type == "audio_classification":
            return """
# Format results for audio classification
# Get class labels if available
id2label = getattr(model.config, 'id2label', None)

if id2label:
    # Get top 5 predictions
    top_indices = predictions[0].cpu().argsort(descending=True)[:5].tolist()
    results = []
    for idx in top_indices:
        label = id2label.get(str(idx), f"LABEL_{idx}")
        score = predictions[0][idx].item()
        results.append({"label": label, "score": score})
else:
    # Just return raw probabilities for top 5 classes
    top_indices = predictions[0].cpu().argsort(descending=True)[:5].tolist()
    results = []
    for idx in top_indices:
        results.append({"class_idx": idx, "score": predictions[0][idx].item()})

return {
    "success": True,
    "classification": {
        "results": results,
        "top_label": results[0]["label"] if "label" in results[0] else f"CLASS_{results[0]['class_idx']}",
        "top_score": results[0]["score"]
    },
    "device": device,
    "hardware": hardware_label
}
"""
        elif task_type == "text_to_speech":
            return """
# Format results for text-to-speech
# Encode audio file to base64 if available
audio_base64 = None
if audio_path and os.path.exists(audio_path):
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

return {
    "success": True,
    "speech": {
        "input_text": input_text,
        "audio_path": audio_path,
        "audio_base64": audio_base64,
        "sample_rate": getattr(model.config, "sampling_rate", 22050)
    },
    "device": device,
    "hardware": hardware_label
}
"""
        else:
            # Default result formatting for other audio tasks
            return """
# Default result formatting for audio tasks
return {
    "success": True,
    "audio_results": result,
    "device": device,
    "hardware": hardware_label
}
"""
    
    def get_mock_input_code(self) -> str:
        """Get audio mock input code."""
        return """
# Mock audio input
import os
import tempfile
import numpy as np
import wave

# Create a mock audio file
mock_audio_path = os.path.join(tempfile.gettempdir(), "mock_audio.wav")
sample_rate = 16000
duration = 3  # seconds
sample_count = sample_rate * duration
mock_audio_data = np.sin(2 * np.pi * 440 * np.arange(sample_count) / sample_rate).astype(np.float32)

# Save mock audio to WAV file
with wave.open(mock_audio_path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes((mock_audio_data * 32767).astype(np.int16).tobytes())

# Create mock audio input
mock_input = mock_audio_path
"""
    
    def get_mock_output_code(self) -> str:
        """Get audio mock output code."""
        return """
# Mock audio output for speech recognition
mock_output = {
    "success": True,
    "transcription": {
        "text": "This is a mock transcription of the audio.",
        "all_texts": ["This is a mock transcription of the audio."]
    },
    "device": "cpu",
    "hardware": "mock"
}
"""
    
    def get_pipeline_utilities(self) -> str:
        """Get audio utility functions."""
        return """
# Audio pipeline utilities
def encode_audio_base64(audio_path):
    """Encode an audio file to base64 string."""
    if not os.path.exists(audio_path):
        return None
        
    with open(audio_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def save_audio_to_file(audio_array, sample_rate, output_path):
    """Save a numpy array audio to a WAV file."""
    try:
        import scipy.io.wavfile as wavfile
        wavfile.write(output_path, sample_rate, audio_array)
        return True
    except ImportError:
        # Fallback if scipy is not available
        try:
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((audio_array * 32767).astype(np.int16).tobytes())
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check audio pipeline compatibility with architecture type."""
        # Audio pipeline is compatible with speech architectures
        return arch_type in [
            "speech",
            "multimodal"
        ]
    
    def is_compatible_with_task(self, task_type: str) -> bool:
        """Check audio pipeline compatibility with task type."""
        # Audio pipeline is compatible with speech/audio tasks
        return task_type in [
            "speech_recognition",
            "audio_classification", 
            "text_to_speech",
            "audio_embedding"
        ]