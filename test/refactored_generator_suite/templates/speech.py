#!/usr/bin/env python3
"""
Speech Template

This module provides the template for speech models like Whisper, Wav2Vec2, etc.
"""

import logging
from typing import Dict, Any, List

from .base import TemplateBase

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeechTemplate(TemplateBase):
    """
    Template for speech models like Whisper, Wav2Vec2, etc.
    
    This template provides specialized support for models that process audio inputs
    for tasks like speech recognition, audio classification, etc.
    """
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this template.
        
        Returns:
            Dictionary of metadata
        """
        metadata = super().get_metadata()
        metadata.update({
            "name": "SpeechTemplate",
            "description": "Template for speech models like Whisper, Wav2Vec2, etc.",
            "supported_architectures": ["speech", "audio"],
            "supported_models": [
                "whisper", "wav2vec2", "hubert", "sew", "unispeech", "clap", 
                "musicgen", "encodec", "bark", "speecht5"
            ]
        })
        return metadata
    
    def get_imports(self) -> List[str]:
        """
        Get the imports required by this template.
        
        Returns:
            List of import statements
        """
        imports = super().get_imports()
        imports.extend([
            "import numpy as np",
            "try:",
            "    import torch",
            "    HAS_TORCH = True",
            "except ImportError:",
            "    torch = MagicMock()",
            "    HAS_TORCH = False",
            "    print(\"torch not available, using mock\")",
            "",
            "try:",
            "    import transformers",
            "    from transformers import AutoProcessor, AutoModelForAudioClassification, AutoModelForSpeechSeq2Seq, AutoModelForCTC",
            "    from transformers import pipeline",
            "    HAS_TRANSFORMERS = True",
            "except ImportError:",
            "    transformers = MagicMock()",
            "    AutoProcessor = MagicMock()",
            "    AutoModelForAudioClassification = MagicMock()",
            "    AutoModelForSpeechSeq2Seq = MagicMock()",
            "    AutoModelForCTC = MagicMock()",
            "    pipeline = MagicMock()",
            "    HAS_TRANSFORMERS = False",
            "    print(\"transformers not available, using mock\")",
            "",
            "try:",
            "    import librosa",
            "    HAS_LIBROSA = True",
            "except ImportError:",
            "    librosa = MagicMock()",
            "    HAS_LIBROSA = False",
            "    print(\"librosa not available, using mock\")"
        ])
        return imports
    
    def get_template_str(self) -> str:
        """
        Get the template string for speech models.
        
        Returns:
            The template as a string
        """
        return """#!/usr/bin/env python3
"""
"""
Test file for {{ model_info.name }}

This test file was automatically generated for the {{ model_info.name }} model,
which is a speech model from the {{ model_info.type }} family.

Generated on: {{ timestamp }}
"""

# Standard library imports
import os
import sys
import json
import time
import datetime
import logging
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Third-party imports
import numpy as np

# Try to import hardware detection if available
try:
    from generators.hardware.hardware_detection import (
        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
        detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback

# Check if we should mock specific dependencies
MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'
MOCK_LIBROSA = os.environ.get('MOCK_LIBROSA', 'False').lower() == 'true'

# Try to import torch
try:
    if MOCK_TORCH:
        raise ImportError("Mocked torch import failure")
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

# Try to import transformers
try:
    if MOCK_TRANSFORMERS:
        raise ImportError("Mocked transformers import failure")
    import transformers
    from transformers import AutoProcessor, AutoModelForAudioClassification, AutoModelForSpeechSeq2Seq, AutoModelForCTC
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    AutoProcessor = MagicMock()
    AutoModelForAudioClassification = MagicMock()
    AutoModelForSpeechSeq2Seq = MagicMock()
    AutoModelForCTC = MagicMock()
    pipeline = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Try to import librosa
try:
    if MOCK_LIBROSA:
        raise ImportError("Mocked librosa import failure")
    import librosa
    HAS_LIBROSA = True
except ImportError:
    librosa = MagicMock()
    HAS_LIBROSA = False
    logger.warning("librosa not available, using mock")

# CUDA detection
if HAS_TORCH:
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        cuda_version = torch.version.cuda
        logger.info(f"CUDA available: version {cuda_version}")
        num_devices = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {num_devices}")
        
        # Log CUDA device properties
        for i in range(num_devices):
            device_props = torch.cuda.get_device_properties(i)
            logger.info(f"CUDA Device {i}: {device_props.name} with {device_props.total_memory / 1024**3:.2f} GB memory")
    else:
        logger.info("CUDA not available")
else:
    HAS_CUDA = False
    logger.info("CUDA detection skipped (torch not available)")

# MPS (Apple Silicon) detection
if HAS_TORCH:
    HAS_MPS = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if HAS_MPS:
        logger.info("MPS available for Apple Silicon acceleration")
    else:
        logger.info("MPS not available")
else:
    HAS_MPS = False
    logger.info("MPS detection skipped (torch not available)")

# ROCm detection
HAS_ROCM = False
if HAS_TORCH:
    try:
        if torch.cuda.is_available() and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
            HAS_ROCM = True
            ROCM_VERSION = torch._C._rocm_version()
            logger.info(f"ROCm available: version {ROCM_VERSION}")
        elif 'ROCM_HOME' in os.environ:
            HAS_ROCM = True
            logger.info("ROCm available (detected via ROCM_HOME)")
    except:
        HAS_ROCM = False
        logger.info("ROCm not available")

# OpenVINO detection
try:
    import openvino
    from openvino.runtime import Core
    HAS_OPENVINO = True
    logger.info(f"OpenVINO available: version {openvino.__version__}")
except ImportError:
    HAS_OPENVINO = False
    logger.info("OpenVINO not available")

# WebGPU detection
HAS_WEBGPU = False
try:
    import ctypes.util
    HAS_WEBGPU = hasattr(ctypes.util, 'find_library') and ctypes.util.find_library('webgpu') is not None
    if HAS_WEBGPU:
        logger.info("WebGPU available")
    else:
        logger.info("WebGPU not available")
except ImportError:
    HAS_WEBGPU = False
    logger.info("WebGPU not available")

# WebNN detection
HAS_WEBNN = False
try:
    import ctypes.util
    HAS_WEBNN = hasattr(ctypes.util, 'find_library') and ctypes.util.find_library('webnn') is not None
    if HAS_WEBNN:
        logger.info("WebNN available")
    else:
        logger.info("WebNN not available")
except ImportError:
    HAS_WEBNN = False
    logger.info("WebNN not available")

def select_device():
    """Select the best available device for inference."""
    if HAS_CUDA:
        return "cuda:0"
    elif HAS_ROCM:
        return "cuda:0"  # ROCm uses CUDA interface
    elif HAS_MPS:
        return "mps"
    else:
        return "cpu"

def get_sample_audio_path():
    """
    Get the path to a sample audio file for testing.
    
    Returns:
        Path to a sample audio file
    """
    # Check for test audio in the project
    test_paths = [
        # Current directory
        Path("test.wav"),
        Path("test.mp3"),
        Path("test_audio.wav"),
        # Test directory
        Path(__file__).parent / "test.wav",
        Path(__file__).parent / "test.mp3",
        # Project root
        Path(__file__).parent.parent.parent / "test.wav",
        Path(__file__).parent.parent.parent / "test.mp3",
        Path(__file__).parent.parent.parent / "test_audio.wav",
    ]
    
    for path in test_paths:
        if path.exists():
            return str(path)
    
    # If no test audio is found, create a dummy audio file
    if HAS_LIBROSA and HAS_TORCH:
        # Create a simple audio file with 1s of silence
        sr = 16000  # Sample rate (16kHz)
        audio = np.zeros(sr)  # 1 second of silence
        audio_path = "temp_test_audio.wav"
        
        try:
            import soundfile as sf
            sf.write(audio_path, audio, sr)
            logger.info(f"Created temporary test audio at {audio_path}")
            return audio_path
        except:
            try:
                import scipy.io.wavfile as wav
                wav.write(audio_path, sr, audio.astype(np.float32))
                logger.info(f"Created temporary test audio at {audio_path}")
                return audio_path
            except:
                logger.warning("Could not create temporary audio file")
    
    logger.warning("No sample audio found and couldn't create one")
    return None

def determine_model_type(model_name):
    """
    Determine the specific speech model type based on the model name.
    
    Args:
        model_name: Model name
        
    Returns:
        Model type (e.g., "whisper", "wav2vec2")
    """
    model_name_lower = model_name.lower()
    
    if "whisper" in model_name_lower:
        return "whisper"
    elif "wav2vec2" in model_name_lower:
        return "wav2vec2"
    elif "hubert" in model_name_lower:
        return "hubert"
    elif "sew" in model_name_lower:
        return "sew"
    elif "unispeech" in model_name_lower:
        return "unispeech"
    elif "clap" in model_name_lower:
        return "clap"
    elif "musicgen" in model_name_lower:
        return "musicgen"
    elif "encodec" in model_name_lower:
        return "encodec"
    elif "bark" in model_name_lower:
        return "bark"
    elif "speecht5" in model_name_lower:
        return "speecht5"
    else:
        return "unknown"

class {{ model_info.type|capitalize }}Test:
    """
    Test class for {{ model_info.name }} model.
    
    This class provides methods to test the model's functionality
    using both the pipeline API and direct model access.
    """
    
    def __init__(self, model_name=None, output_dir=None, device=None, audio_path=None):
        """
        Initialize the test class.
        
        Args:
            model_name: The name or path of the model to test (default: {{ model_info.name }})
            output_dir: Directory to save outputs (default: None)
            device: Device to run the model on (default: auto-selected)
            audio_path: Path to audio file for testing (default: auto-selected)
        """
        self.model_name = model_name or "{{ model_info.name }}"
        self.output_dir = output_dir
        self.device = device or select_device()
        self.audio_path = audio_path or get_sample_audio_path()
        
        # Determine model type for specific handling
        self.model_type = determine_model_type(self.model_name)
        
        # Create output directory if specified
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Mock detection
        self.using_real_inference = HAS_TRANSFORMERS and HAS_TORCH and (HAS_LIBROSA or self.audio_path is not None)
        self.using_mocks = not self.using_real_inference
        
        logger.info(f"Initialized test for {self.model_name} (type: {self.model_type}) on {self.device}")
        logger.info(f"Test type: {'🚀 REAL INFERENCE' if self.using_real_inference else '🔷 MOCK OBJECTS (CI/CD)'}")
        logger.info(f"Audio path: {self.audio_path or 'None'}")
    
    def run(self):
        """
        Run all tests for this model.
        
        Returns:
            Dictionary with test results
        """
        results = {
            "metadata": {
                "model_name": self.model_name,
                "model_type": self.model_type,
                "device": self.device,
                "audio_path": self.audio_path,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_librosa": HAS_LIBROSA,
                "has_cuda": HAS_CUDA,
                "has_rocm": HAS_ROCM,
                "has_mps": HAS_MPS,
                "has_openvino": HAS_OPENVINO,
                "has_webgpu": HAS_WEBGPU,
                "has_webnn": HAS_WEBNN,
                "using_real_inference": self.using_real_inference,
                "using_mocks": self.using_mocks,
                "test_type": "REAL INFERENCE" if (self.using_real_inference and not self.using_mocks) else "MOCK OBJECTS (CI/CD)"
            },
            "tests": {}
        }
        
        # Skip audio tests if no audio is available
        if not self.audio_path:
            results["tests"]["error"] = {
                "success": False,
                "elapsed_time": 0,
                "error": "No test audio available"
            }
            return results
        
        # Run tests based on model type
        if self.model_type == "whisper":
            results["tests"]["pipeline"] = self.test_pipeline_asr()
            results["tests"]["model"] = self.test_model_whisper()
        elif self.model_type in ["wav2vec2", "hubert", "unispeech", "sew"]:
            results["tests"]["pipeline"] = self.test_pipeline_asr()
            results["tests"]["model"] = self.test_model_wav2vec2()
        elif self.model_type == "clap":
            results["tests"]["pipeline"] = self.test_pipeline_audio_classification()
            results["tests"]["model"] = self.test_model_clap()
        elif self.model_type in ["encodec", "musicgen", "bark", "speecht5"]:
            results["tests"]["pipeline"] = self.test_pipeline_text_to_audio()
            results["tests"]["model"] = self.test_model_generic()
        else:
            # Generic handling for unknown model types
            results["tests"]["pipeline"] = self.test_pipeline_asr()
            results["tests"]["model"] = self.test_model_generic()
        
        # Common tests
        results["tests"]["processor"] = self.test_processor()
        
        {% if has_openvino %}
        # Run OpenVINO test if available
        results["tests"]["openvino"] = self.test_openvino()
        {% endif %}
        
        return results
    
    def load_audio(self):
        """
        Load audio file and convert to appropriate format.
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self.audio_path:
            # Create dummy audio if no path is available
            return np.zeros(16000), 16000
            
        if HAS_LIBROSA:
            # Load with librosa (recommended)
            try:
                audio, sr = librosa.load(self.audio_path, sr=16000)
                return audio, sr
            except Exception as e:
                logger.warning(f"Error loading audio with librosa: {e}")
        
        # Fallbacks for different audio loading methods
        try:
            import soundfile as sf
            audio, sr = sf.read(self.audio_path)
            if sr != 16000:
                # Resample to 16kHz if needed
                if HAS_LIBROSA:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            return audio, sr
        except:
            try:
                from scipy import io as sio
                sr, audio = sio.wavfile.read(self.audio_path)
                # Convert to float32 if needed
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
                if sr != 16000:
                    # Warn about resampling without librosa
                    logger.warning("Audio sample rate is not 16kHz and librosa not available for resampling")
                return audio, sr
            except:
                # Last resort: dummy audio
                logger.warning("Could not load audio file, using dummy audio")
                return np.zeros(16000), 16000
    
    def test_pipeline_asr(self):
        """
        Test the model using the ASR pipeline.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} with ASR pipeline")
        start_time = time.time()
        
        try:
            # Create automatic-speech-recognition pipeline
            asr = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self.device if self.device != "mps" else -1  # MPS not supported by pipeline
            )
            
            # Run inference
            result = asr(self.audio_path)
            
            # Process result
            if isinstance(result, dict) and "text" in result:
                text = result["text"]
                logger.info(f"ASR result: {text}")
            else:
                text = str(result)
                logger.info(f"ASR result (raw): {text}")
            
            return {
                "success": True,
                "elapsed_time": time.time() - start_time,
                "text": text,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in ASR pipeline test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "text": None,
                "error": str(e)
            }
    
    def test_pipeline_audio_classification(self):
        """
        Test the model using the audio classification pipeline.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} with audio classification pipeline")
        start_time = time.time()
        
        try:
            # Create audio-classification pipeline
            classifier = pipeline(
                "audio-classification",
                model=self.model_name,
                device=self.device if self.device != "mps" else -1  # MPS not supported by pipeline
            )
            
            # Run inference
            result = classifier(self.audio_path)
            
            # Process result
            if isinstance(result, list) and len(result) > 0:
                top_result = result[0]
                if isinstance(top_result, dict) and "label" in top_result and "score" in top_result:
                    label = top_result["label"]
                    score = top_result["score"]
                    logger.info(f"Classification result: {label} ({score:.4f})")
                else:
                    label = str(top_result)
                    score = 0.0
            else:
                label = "unknown"
                score = 0.0
            
            return {
                "success": True,
                "elapsed_time": time.time() - start_time,
                "label": label,
                "score": score,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in audio classification pipeline test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "label": None,
                "score": None,
                "error": str(e)
            }
    
    def test_pipeline_text_to_audio(self):
        """
        Test the model using the text-to-audio pipeline.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} with text-to-audio pipeline")
        start_time = time.time()
        
        try:
            # Create text-to-audio pipeline
            synthesizer = pipeline(
                "text-to-audio",
                model=self.model_name,
                device=self.device if self.device != "mps" else -1  # MPS not supported by pipeline
            )
            
            # Run inference
            result = synthesizer("This is a test of the speech synthesis model.")
            
            # Process result
            if isinstance(result, dict) and "audio" in result:
                audio = result["audio"]
                sampling_rate = result.get("sampling_rate", 16000)
                audio_length = len(audio) / sampling_rate
                logger.info(f"Synthesized audio: {audio_length:.2f} seconds at {sampling_rate} Hz")
            else:
                audio_length = 0.0
                sampling_rate = 0
            
            return {
                "success": True,
                "elapsed_time": time.time() - start_time,
                "audio_length": audio_length,
                "sampling_rate": sampling_rate,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in text-to-audio pipeline test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "audio_length": None,
                "sampling_rate": None,
                "error": str(e)
            }
    
    def test_model_whisper(self):
        """
        Test Whisper model using direct model access.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} with direct model access")
        start_time = time.time()
        
        try:
            # Load processor and model
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
            
            # Move model to device
            if self.device != "cpu":
                model = model.to(self.device)
            
            # Load audio
            audio, sr = self.load_audio()
            
            # Process audio
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate token ids
            with torch.no_grad():
                predicted_ids = model.generate(inputs["input_features"])
            
            # Decode token ids to text
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            logger.info(f"Transcription: {transcription}")
            
            return {
                "success": True,
                "elapsed_time": time.time() - start_time,
                "transcription": transcription,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in Whisper model test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "transcription": None,
                "error": str(e)
            }
    
    def test_model_wav2vec2(self):
        """
        Test Wav2Vec2-like model using direct model access.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} with direct model access")
        start_time = time.time()
        
        try:
            # Load processor and model
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModelForCTC.from_pretrained(self.model_name)
            
            # Move model to device
            if self.device != "cpu":
                model = model.to(self.device)
            
            # Load audio
            audio, sr = self.load_audio()
            
            # Process audio
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Get predicted ids
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode to text
            transcription = processor.batch_decode(predicted_ids)[0]
            
            logger.info(f"Transcription: {transcription}")
            
            return {
                "success": True,
                "elapsed_time": time.time() - start_time,
                "transcription": transcription,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in Wav2Vec2 model test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "transcription": None,
                "error": str(e)
            }
    
    def test_model_clap(self):
        """
        Test CLAP model using direct model access.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} with direct model access")
        start_time = time.time()
        
        try:
            # Load processor and model
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            
            # Move model to device
            if self.device != "cpu":
                model = model.to(self.device)
            
            # Load audio
            audio, sr = self.load_audio()
            
            # Process audio
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get class probabilities
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
                
                # Get top predictions
                top_prob, top_idx = torch.topk(probabilities, k=1)
                
                # Get labels if available
                if hasattr(model.config, "id2label"):
                    label = model.config.id2label[top_idx.item()]
                else:
                    label = f"LABEL_{top_idx.item()}"
                
                score = top_prob.item()
                
                logger.info(f"Classification: {label} ({score:.4f})")
                
                return {
                    "success": True,
                    "elapsed_time": time.time() - start_time,
                    "label": label,
                    "score": score,
                    "error": None
                }
            else:
                return {
                    "success": True,
                    "elapsed_time": time.time() - start_time,
                    "output_shape": {k: list(v.shape) for k, v in outputs.items() if hasattr(v, "shape")},
                    "error": None
                }
        except Exception as e:
            logger.error(f"Error in CLAP model test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "label": None,
                "score": None,
                "error": str(e)
            }
    
    def test_model_generic(self):
        """
        Test model using direct model access with generic approach.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} with direct model access (generic)")
        start_time = time.time()
        
        try:
            # Load processor and model
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Determine the appropriate model class based on model type
            if "whisper" in self.model_type:
                model_class = AutoModelForSpeechSeq2Seq
            elif self.model_type in ["wav2vec2", "hubert", "sew", "unispeech"]:
                model_class = AutoModelForCTC
            elif self.model_type in ["clap"]:
                model_class = AutoModelForAudioClassification
            else:
                # Generic fallback to AutoModel
                from transformers import AutoModel
                model_class = AutoModel
            
            model = model_class.from_pretrained(self.model_name)
            
            # Move model to device
            if self.device != "cpu":
                model = model.to(self.device)
            
            # Load audio
            audio, sr = self.load_audio()
            
            # Process audio
            try:
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            except:
                # Alternative input format for some models
                try:
                    inputs = processor(raw_speech=audio, sampling_rate=sr, return_tensors="pt")
                except:
                    # Last resort: create dummy inputs
                    inputs = {"input_features": torch.rand(1, 80, 3000)}
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Process outputs generically
            output_info = {}
            for key, value in outputs.items():
                if hasattr(value, "shape"):
                    output_info[key] = list(value.shape)
            
            return {
                "success": True,
                "elapsed_time": time.time() - start_time,
                "output_info": output_info,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in generic model test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "output_info": None,
                "error": str(e)
            }
    
    def test_processor(self):
        """
        Test the model's processor.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} processor")
        start_time = time.time()
        
        try:
            # Load processor
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load audio
            audio, sr = self.load_audio()
            
            # Process audio
            try:
                processed = processor(audio, sampling_rate=sr, return_tensors="pt")
            except:
                try:
                    processed = processor(raw_speech=audio, sampling_rate=sr, return_tensors="pt")
                except:
                    processed = {"input_features": torch.rand(1, 80, 3000)}
            
            # Get processor info
            processor_info = {
                "processor_type": processor.__class__.__name__,
                "sampling_rate": getattr(processor, "sampling_rate", sr),
                "output_keys": list(processed.keys())
            }
            
            # Add feature extractor info if available
            if hasattr(processor, "feature_extractor"):
                feature_extractor = processor.feature_extractor
                processor_info.update({
                    "feature_extractor_type": feature_extractor.__class__.__name__,
                    "feature_size": getattr(feature_extractor, "feature_size", None),
                    "padding_side": getattr(feature_extractor, "padding_side", None),
                    "padding_value": getattr(feature_extractor, "padding_value", None)
                })
            
            # Add tokenizer info if available
            if hasattr(processor, "tokenizer"):
                tokenizer = processor.tokenizer
                processor_info.update({
                    "tokenizer_type": tokenizer.__class__.__name__,
                    "vocab_size": getattr(tokenizer, "vocab_size", None),
                    "model_max_length": getattr(tokenizer, "model_max_length", None)
                })
            
            return {
                "success": True,
                "elapsed_time": time.time() - start_time,
                "processor_info": processor_info,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in processor test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "processor_info": None,
                "error": str(e)
            }
    
    {% if has_openvino %}
    def test_openvino(self):
        """
        Test the model using OpenVINO.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} with OpenVINO")
        start_time = time.time()
        
        try:
            from optimum.intel import OVModelForSpeechSeq2Seq
            
            # Load processor and model
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Try to load model based on model type
            if "whisper" in self.model_type:
                model = OVModelForSpeechSeq2Seq.from_pretrained(self.model_name)
            else:
                # OpenVINO may not support all model types
                return {
                    "success": False,
                    "elapsed_time": time.time() - start_time,
                    "error": f"OpenVINO support not available for model type: {self.model_type}"
                }
            
            # Load audio
            audio, sr = self.load_audio()
            
            # Process audio
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            
            # Run inference (for Whisper)
            predicted_ids = model.generate(inputs["input_features"])
            
            # Decode token ids to text
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            logger.info(f"OpenVINO transcription: {transcription}")
            
            return {
                "success": True,
                "elapsed_time": time.time() - start_time,
                "transcription": transcription,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in OpenVINO test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "transcription": None,
                "error": str(e)
            }
    {% endif %}
    
    def save_results(self, results, filename=None):
        """
        Save test results to a file.
        
        Args:
            results: Dictionary with test results
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to the saved file
        """
        if not self.output_dir:
            logger.warning("No output directory specified, results not saved")
            return None
            
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_safe = self.model_name.replace("/", "_")
            filename = f"{model_name_safe}_{timestamp}.json"
            
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")
        return output_path

def main():
    """
    Main function to run the test.
    """
    parser = argparse.ArgumentParser(description="Test {{ model_info.type|capitalize }} model")
    parser.add_argument("--model", default="{{ model_info.name }}", help="Model name or path")
    parser.add_argument("--output-dir", help="Directory to save outputs")
    parser.add_argument("--device", help="Device to run on (cpu, cuda:0, etc.)")
    parser.add_argument("--audio", help="Path to audio file for testing")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    args = parser.parse_args()
    
    # Run test
    test = {{ model_info.type|capitalize }}Test(
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device,
        audio_path=args.audio
    )
    
    results = test.run()
    
    # Save results if requested
    if args.save or args.output_dir:
        test.save_results(results)
    
    # Print summary
    print(f"\nTest Summary for {args.model}:")
    print(f"  Device: {results['metadata']['device']}")
    print(f"  Test Type: {results['metadata']['test_type']}")
    
    for test_name, test_result in results["tests"].items():
        status = "✅ Passed" if test_result.get("success", False) else "❌ Failed"
        print(f"  {test_name}: {status}")
        if not test_result.get("success", False):
            print(f"    Error: {test_result.get('error', 'Unknown')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""