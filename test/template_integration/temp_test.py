#!/usr/bin/env python3
"""
Test file for speech models using the refactored test suite structure.
"""

import os
import sys
import json
import time
import logging
import torch
import numpy as np
from pathlib import Path
from refactored_test_suite.model_test import ModelTest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestSpeechModel(ModelTest):
    """Test class for speech/audio models like Whisper, Wav2Vec2, etc."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        
        # Initialize model-specific attributes
        self.model_id = "facebook/wav2vec2-base-960h"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define model parameters
        self.task = "automatic-speech-recognition"
        self.audio_sampling_rate = 16000
        
        # Define test audio path
        self.test_audio_path = "test_audio.wav"
    
    def tearDown(self):
        """Clean up resources after the test."""
        super().tearDown()
    
    def create_test_audio(self):
        """Create a test audio file if it doesn't exist."""
        if not os.path.exists(self.test_audio_path):
            try:
                # Generate a simple sine wave
                import scipy.io.wavfile as wav
                sample_rate = self.audio_sampling_rate
                duration = 3  # seconds
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
                wav.write(self.test_audio_path, sample_rate, audio.astype(np.float32))
                return True
            except Exception as e:
                logger.error(f"Error creating test audio: {e}")
                return False
        return True
    
    def load_audio(self):
        """Load audio data from file."""
        # Ensure test audio exists
        self.create_test_audio()
        
        try:
            # Try to use soundfile
            import soundfile as sf
            audio, sample_rate = sf.read(self.test_audio_path)
        except ImportError:
            # Fallback to scipy
            import scipy.io.wavfile as wav
            sample_rate, audio = wav.read(self.test_audio_path)
            # Convert to float if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        
        return audio, sample_rate
    
    def load_model(self):
        """Load the model for testing."""
        try:
            if "whisper" in self.model_id.lower():
                # For Whisper models
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                
                processor = WhisperProcessor.from_pretrained(self.model_id)
                model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
            elif "wav2vec2" in self.model_id.lower():
                # For Wav2Vec2 models
                from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
                
                processor = Wav2Vec2Processor.from_pretrained(self.model_id)
                model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
            else:
                # For other speech models
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                
                processor = AutoProcessor.from_pretrained(self.model_id)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id)
            
            # Move to device
            model = model.to(self.device)
            
            return {"model": model, "processor": processor}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def test_model_loading(self):
        """Test that the model loads correctly."""
        model_components = self.load_model()
        
        # Verify model and processor
        self.assertIsNotNone(model_components["model"])
        self.assertIsNotNone(model_components["processor"])
        
        logger.info("Model loaded successfully")
    
    def test_basic_inference(self):
        """Test basic inference with the model."""
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        processor = model_components["processor"]
        
        # Load audio
        audio, sample_rate = self.load_audio()
        
        # Process audio
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        
        # Check output shape
        if hasattr(outputs, "logits"):
            logger.info(f"Output shape: {outputs.logits.shape}")
        
        logger.info("Basic inference successful")
    
    def test_transcription(self):
        """Test transcription with the model."""
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        processor = model_components["processor"]
        
        # Load audio
        audio, sample_rate = self.load_audio()
        
        # Process audio
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Model-specific transcription
        if "whisper" in self.model_id.lower():
            # Whisper model
            with torch.no_grad():
                generated_ids = model.generate(inputs["input_features"], max_length=100)
                
            # Decode the output
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            # CTC-based models like Wav2Vec2
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Get the predicted ids
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode the output
            transcription = processor.batch_decode(predicted_ids)[0]
        
        logger.info(f"Transcription: {transcription}")
        logger.info("Transcription successful")
    
    def test_hardware_compatibility(self):
        """Test model compatibility across hardware platforms."""
        available_devices = ["cpu"]
        if torch.cuda.is_available():
            available_devices.append("cuda")
        
        results = {}
        original_device = self.device
        
        for device in available_devices:
            try:
                self.device = device
                model_components = self.load_model()
                model = model_components["model"]
                
                # Basic verification
                self.assertIsNotNone(model)
                results[device] = True
                logger.info(f"Model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"Failed on {device}: {e}")
                results[device] = False
            finally:
                self.device = original_device
        
        # Verify at least one device works
        self.assertTrue(any(results.values()))
