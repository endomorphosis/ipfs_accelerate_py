#!/usr/bin/env python3
"""
Test file for CLAP (Contrastive Language-Audio Pretraining) models.

This file has been migrated to the refactored test suite.
Generated: 2025-03-21 from key_models_hardware_fixes/test_hf_clap.py
"""

import os
import sys
import json
import logging
import unittest
from unittest.mock import patch, MagicMock
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from refactored_test_suite.model_test import ModelTest

# Try to import required packages with fallbacks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False

try:
    import transformers
    from transformers import AutoProcessor, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    AutoProcessor = MagicMock()
    AutoModel = MagicMock()
    HAS_TRANSFORMERS = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    librosa = MagicMock()
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    sf = MagicMock()
    HAS_SOUNDFILE = False

try:
    import openvino
    HAS_OPENVINO = True
except ImportError:
    openvino = MagicMock()
    HAS_OPENVINO = False

# Define utility functions needed for tests
def load_audio(audio_file, target_sr=48000):
    """Load audio from file and return audio data and sample rate."""
    if not os.path.exists(audio_file):
        # Return mock audio data if file doesn't exist
        return np.zeros(target_sr * 3, dtype=np.float32), target_sr
        
    try:
        if HAS_LIBROSA:
            # Use librosa if available
            audio_data, sample_rate = librosa.load(audio_file, sr=target_sr)
            return audio_data, sample_rate
        elif HAS_SOUNDFILE:
            # Use soundfile as fallback
            audio_data, sample_rate = sf.read(audio_file)
            return audio_data, sample_rate
        else:
            # Return mock audio data if no library is available
            return np.zeros(target_sr * 3, dtype=np.float32), target_sr
    except Exception as e:
        logging.error(f"Error loading audio: {e}")
        # Return mock audio data on error
        return np.zeros(target_sr * 3, dtype=np.float32), target_sr

def create_test_audio(file_path, duration=3, sample_rate=48000):
    """Create a test audio file with a simple tone."""
    # Generate a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Save to file if soundfile is available
    if HAS_SOUNDFILE:
        sf.write(file_path, audio, sample_rate)
    
    return audio, sample_rate

class TestClapModels(ModelTest):
    """Test class for CLAP (Contrastive Language-Audio Pretraining) models."""
    
    def setUp(self):
        """Initialize the test with model details and hardware detection."""
        super().setUp()
        self.model_id = "laion/clap-htsat-unfused"  # Default model ID
        self.test_text = "This is a dog barking"
        self.test_audio_path = os.path.join(self.model_dir, "test_audio.wav")
        self.audio_duration = 3  # seconds
        self.sample_rate = 48000  # CLAP model requires 48kHz sample rate
        
        # Create a test audio file
        self.test_audio, _ = create_test_audio(
            self.test_audio_path, 
            self.audio_duration, 
            self.sample_rate
        )
        
        # Hardware detection
        self.setup_hardware()
        
        # Results storage
        self.results = {}
    
    def setup_hardware(self):
        """Set up hardware detection for the test."""
        # Skip test if torch is not available
        if not HAS_TORCH:
            self.skipTest("PyTorch not available")
            
        # CUDA support
        self.has_cuda = torch.cuda.is_available()
        
        # MPS support (Apple Silicon)
        try:
            self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except AttributeError:
            self.has_mps = False
            
        # ROCm support (AMD)
        self.has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None
        
        # OpenVINO support
        self.has_openvino = HAS_OPENVINO
        
        # WebNN/WebGPU support (always false for server-side tests)
        self.has_webnn = False
        self.has_webgpu = False
        
        # Set default device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
            
        self.logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_id=None):
        """Load CLAP model from HuggingFace."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
            
        model_id = model_id or self.model_id
            
        try:
            # Load processor
            self.logger.info(f"Loading CLAP processor: {model_id}")
            processor = AutoProcessor.from_pretrained(model_id)
            
            # Load model
            self.logger.info(f"Loading CLAP model: {model_id} on {self.device}")
            model = AutoModel.from_pretrained(model_id).to(self.device)
            
            # Store processor for later use
            self.processor = processor
            
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.skipTest(f"Could not load model: {str(e)}")
    
    def test_basic_inference(self):
        """Test basic inference with the CLAP model."""
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            self.skipTest("Required dependencies not available")
        
        self.logger.info(f"Testing {self.model_id} with basic inference on {self.device}...")
        
        try:
            # Load model and processor
            model = self.load_model()
            processor = self.processor
            
            # Load audio
            audio_data, sample_rate = load_audio(self.test_audio_path)
            
            # Process inputs
            inputs = processor(
                text=[self.test_text],
                audios=[audio_data],
                sampling_rate=sample_rate,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move inputs to the right device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                start_time = time.time()
                outputs = model(**inputs)
                inference_time = time.time() - start_time
            
            # Verify outputs
            self.assertIsNotNone(outputs, "Model outputs should not be None")
            self.assertIn("audio_embeds", outputs, "Model should output audio embeddings")
            self.assertIn("text_embeds", outputs, "Model should output text embeddings")
            
            # Check embedding shapes
            audio_embeds = outputs.audio_embeds
            text_embeds = outputs.text_embeds
            
            self.assertEqual(audio_embeds.shape[0], 1, "Batch size should be 1")
            self.assertEqual(text_embeds.shape[0], 1, "Batch size should be 1")
            self.assertEqual(audio_embeds.shape[-1], text_embeds.shape[-1], 
                             "Audio and text embedding dimensions should match")
            
            # Calculate similarity
            audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            similarity = torch.matmul(audio_embeds, text_embeds.T)
            
            self.logger.info(f"Audio-text similarity: {similarity.item():.4f}")
            self.logger.info(f"Inference time: {inference_time:.4f} seconds")
            self.logger.info(f"Audio embedding shape: {audio_embeds.shape}")
            self.logger.info(f"Text embedding shape: {text_embeds.shape}")
            
            # Store results
            self.results['basic_inference'] = {
                'similarity': similarity.item(),
                'inference_time': inference_time,
                'audio_embed_dim': audio_embeds.shape[-1],
                'text_embed_dim': text_embeds.shape[-1],
            }
            
            self.logger.info("Basic inference test passed")
            
        except Exception as e:
            self.logger.error(f"Error in basic inference: {e}")
            self.fail(f"Basic inference test failed: {str(e)}")
    
    def test_audio_text_similarity(self):
        """Test the audio-text similarity calculation."""
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            self.skipTest("Required dependencies not available")
        
        self.logger.info(f"Testing {self.model_id} audio-text similarity...")
        
        try:
            # Load model and processor
            model = self.load_model()
            processor = self.processor
            
            # Load audio
            audio_data, sample_rate = load_audio(self.test_audio_path)
            
            # Create multiple text prompts
            text_prompts = [
                "A dog barking loudly",
                "A cat meowing",
                "A person speaking",
                "Background music playing"
            ]
            
            # Process inputs
            inputs = processor(
                text=text_prompts,
                audios=[audio_data] * len(text_prompts),  # Repeat the same audio for each text
                sampling_rate=sample_rate,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move inputs to the right device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get embeddings
            audio_embeds = outputs.audio_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize embeddings
            audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Calculate similarity matrix
            similarity_matrix = torch.matmul(audio_embeds, text_embeds.T)
            
            # Verify the similarity matrix shape
            self.assertEqual(similarity_matrix.shape, (len(text_prompts), len(text_prompts)), 
                            f"Expected similarity matrix shape {(len(text_prompts), len(text_prompts))}, but got {similarity_matrix.shape}")
            
            # Log the similarity matrix
            self.logger.info("Similarity matrix:")
            for i, text in enumerate(text_prompts):
                similarities = [f"{similarity_matrix[i, j].item():.4f}" for j in range(len(text_prompts))]
                self.logger.info(f"  {text}: {', '.join(similarities)}")
            
            # Store results
            self.results['audio_text_similarity'] = {
                'similarity_matrix': similarity_matrix.cpu().numpy().tolist(),
                'text_prompts': text_prompts
            }
            
            self.logger.info("Audio-text similarity test passed")
            
        except Exception as e:
            self.logger.error(f"Error in audio-text similarity test: {e}")
            self.fail(f"Audio-text similarity test failed: {str(e)}")
    
    def test_hardware_compatibility(self):
        """Test model compatibility with different hardware platforms."""
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            self.skipTest("Required dependencies not available")
            
        devices_to_test = []
        
        # Always test CPU
        devices_to_test.append('cpu')
        
        # Only test available hardware
        if self.has_cuda:
            devices_to_test.append('cuda')
        if self.has_mps:
            devices_to_test.append('mps')
            
        # Test each device
        for device in devices_to_test:
            original_device = self.device
            try:
                self.logger.info(f"Testing on {device}...")
                self.device = device
                
                # Load model and processor
                model = self.load_model()
                processor = self.processor
                
                # Load audio
                audio_data, sample_rate = load_audio(self.test_audio_path)
                
                # Process inputs
                inputs = processor(
                    text=[self.test_text],
                    audios=[audio_data],
                    sampling_rate=sample_rate,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Move inputs to the right device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Run inference with timing
                with torch.no_grad():
                    start_time = time.time()
                    outputs = model(**inputs)
                    inference_time = time.time() - start_time
                
                # Verify outputs
                self.assertIsNotNone(outputs, f"Outputs on {device} should not be None")
                self.assertIn("audio_embeds", outputs, f"Model on {device} should output audio embeddings")
                self.assertIn("text_embeds", outputs, f"Model on {device} should output text embeddings")
                
                # Store performance results
                self.results[f'performance_{device}'] = {
                    'inference_time': inference_time,
                    'embedding_dim': outputs.audio_embeds.shape[-1]
                }
                
                self.logger.info(f"Test on {device} passed (inference time: {inference_time:.4f}s)")
            except Exception as e:
                self.logger.error(f"Error testing on {device}: {e}")
                self.fail(f"Test on {device} failed: {str(e)}")
            finally:
                # Restore original device
                self.device = original_device



    def test_model_loading(self):
        # Test basic model loading
        if not hasattr(self, 'model_id') or not self.model_id:
            self.skipTest("No model_id specified")
        
        try:
            # Import the appropriate library
            if 'bert' in self.model_id.lower() or 'gpt' in self.model_id.lower() or 't5' in self.model_id.lower():
                import transformers
                model = transformers.AutoModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'clip' in self.model_id.lower():
                import transformers
                model = transformers.CLIPModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'whisper' in self.model_id.lower():
                import transformers
                model = transformers.WhisperModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'wav2vec2' in self.model_id.lower():
                import transformers
                model = transformers.Wav2Vec2Model.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            else:
                # Generic loading
                try:
                    import transformers
                    model = transformers.AutoModel.from_pretrained(self.model_id)
                    self.assertIsNotNone(model, "Model loading failed")
                except:
                    self.skipTest(f"Could not load model {self.model_id} with AutoModel")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")



    def detect_preferred_device(self):
        # Detect available hardware and choose the preferred device
        try:
            import torch
        
            # Check for CUDA
            if torch.cuda.is_available():
                return "cuda"
        
            # Check for MPS (Apple Silicon)
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
                return "mps"
        
            # Fallback to CPU
            return "cpu"
        except ImportError:
            return "cpu"


if __name__ == "__main__":
    unittest.main()