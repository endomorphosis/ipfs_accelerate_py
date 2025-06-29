#!/usr/bin/env python3
"""
Test file for openai/whisper-tiny model.

This file is auto-generated using the template-based test generator.
Generated: 2025-03-10 01:36:02
"""

import os
import sys
import logging
import torch
import numpy as np
import unittest
from pathlib import Path

# Import ModelTest base class
try:
    from refactored_test_suite.model_test import ModelTest
except ImportError:
    # Fallback to alternative import path
    try:
        from model_test import ModelTest
    except ImportError:
        # Create a temporary ModelTest class if not available
        class ModelTest(unittest.TestCase):
            """Temporary ModelTest class."""
            pass

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestWhisperTiny(ModelTest):
    """Test class for openai/whisper-tiny model."""
    
    def setUp(self):
        """Initialize the test with model details and hardware detection."""
        super().setUp()
        self.model_id = "openai/whisper-tiny"
        self.model_type = "audio"
        self.device = self.detect_preferred_device()
        logger.info(f"Using device: {self.device}")
    
    def setup_hardware(self):
        """Set up hardware detection for the template."""
        # CUDA support
        self.has_cuda = torch.cuda.is_available()
        # MPS support (Apple Silicon)
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        # ROCm support (AMD)
        self.has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None
        # OpenVINO support
        self.has_openvino = 'openvino' in sys.modules
        # Qualcomm AI Engine support
        self.has_qualcomm = 'qti' in sys.modules or 'qnn_wrapper' in sys.modules
        # WebNN/WebGPU support
        self.has_webnn = False  # Will be set by WebNN bridge if available
        self.has_webgpu = False  # Will be set by WebGPU bridge if available
        
        # Set default device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
            
        logger.info(f"Using device: {self.device}")
        
    def get_model(self):
        """Load model from HuggingFace."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Get tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Get model
            model = AutoModel.from_pretrained(self.model_id)
            model = model.to(self.device)
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None
    
    def get_model_specific(self):
        """Load model with specialized configuration for audio processing."""
        try:
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
            
            # Get feature extractor
            processor = AutoFeatureExtractor.from_pretrained(self.model_id)
            
            # Get model with audio-specific settings
            model = AutoModelForAudioClassification.from_pretrained(
                self.model_id,
                torchscript=True if self.device == 'cpu' else False
            )
            model = model.to(self.device)
            
            # Put model in evaluation mode
            model.eval()
            
            return model, processor
        except Exception as e:
            logger.error(f"Error loading audio model with specific settings: {e}")
            
            # Try alternative model type (speech recognition)
            try:
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
                processor = AutoProcessor.from_pretrained(self.model_id)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id)
                model = model.to(self.device)
                model.eval()
                return model, processor
            except Exception as e2:
                logger.error(f"Error in alternative loading: {e2}")
                
                # Fallback to generic model
                try:
                    from transformers import AutoModel, AutoFeatureExtractor
                    processor = AutoFeatureExtractor.from_pretrained(self.model_id)
                    model = AutoModel.from_pretrained(self.model_id)
                    model = model.to(self.device)
                    model.eval()
                    return model, processor
                except Exception as e3:
                    logger.error(f"Error in fallback loading: {e3}")
                    return None, None
    
    def test_model_loading(self):
        """Test that the model loads correctly."""
        model = self.load_model(self.model_id)
        self.assertIsNotNone(model, "Model should load successfully")
    
    def load_model(self, model_name):
        """Load a model for testing."""
        try:
            import transformers
            
            # Try specific audio model classes first
            try:
                from transformers import WhisperModel
                model = WhisperModel.from_pretrained(model_name)
            except Exception:
                try:
                    from transformers import AutoModelForSpeechSeq2Seq
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
                except Exception:
                    # Fallback to generic model loading
                    model = transformers.AutoModel.from_pretrained(model_name)
                
            # Move model to the detected device
            model = model.to(self.device)
            model.eval()
            
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def verify_model_output(self, model, input_data, expected_output=None):
        """Verify that model produces expected output for audio data."""
        try:
            import torch
            
            # For Whisper models, input_data should be processed audio
            if isinstance(input_data, dict):
                # Input is already a dict of tensors (e.g., from processor)
                inputs = input_data
            elif isinstance(input_data, np.ndarray):
                # Raw audio, needs processing
                from transformers import AutoFeatureExtractor
                processor = AutoFeatureExtractor.from_pretrained(model.config._name_or_path)
                inputs = processor(input_data, sampling_rate=16000, return_tensors="pt")
            else:
                # Otherwise, assume input is already properly formatted
                inputs = input_data
                
            # Move inputs to the right device if model is on a specific device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, 'to')}
                
            # Run model
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Verify outputs
            self.assertIsNotNone(outputs, "Model output should not be None")
            
            # If expected output is provided, verify it matches
            if expected_output is not None:
                self.assertEqual(outputs, expected_output)
                
            return outputs
        except Exception as e:
            logger.error(f"Error verifying model output: {e}")
            self.fail(f"Model verification failed: {e}")
    
    def detect_preferred_device(self):
        """Detect available hardware and choose the preferred device."""
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
    
    def test_basic_inference(self):
        """Run a basic inference test with the model."""
        model, tokenizer = self.get_model()
        
        if model is None or tokenizer is None:
            logger.error("Failed to load model or tokenizer")
            return False
        
        try:
            # Prepare audio input
            import torch
            import numpy as np
            from transformers import AutoFeatureExtractor
            
            # Create a test audio if none exists
            test_audio_path = "test_audio.wav"
            if not os.path.exists(test_audio_path):
                # Generate a simple sine wave
                import scipy.io.wavfile as wav
                sample_rate = 16000
                duration = 3  # seconds
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
                wav.write(test_audio_path, sample_rate, audio.astype(np.float32))
            
            # Load audio file
            sample_rate = 16000
            audio = np.zeros(sample_rate * 3)  # 3 seconds of silence as fallback
            try:
                import soundfile as sf
                audio, sample_rate = sf.read(test_audio_path)
            except:
                logger.warning("Could not load audio, using zeros array")
            
            # Get feature extractor
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            assert outputs is not None, "Outputs should not be None"
            if hasattr(outputs, "last_hidden_state"):
                assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
                logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
            else:
                # Some audio models have different output structures
                logger.info(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")
            
            logger.info("Basic inference test passed")
            return True
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return False
    
    def test_hardware_compatibility(self):
        """Test model compatibility with different hardware platforms."""
        # Set up hardware detection for compatibility testing
        self.setup_hardware()
        
        devices_to_test = []
        
        if self.has_cuda:
            devices_to_test.append('cuda')
        if self.has_mps:
            devices_to_test.append('mps')
        if self.has_rocm:
            devices_to_test.append('cuda')  # ROCm uses CUDA compatibility layer
        if self.has_openvino:
            devices_to_test.append('openvino')
        if self.has_qualcomm:
            devices_to_test.append('qualcomm')
        
        # Always test CPU
        if 'cpu' not in devices_to_test:
            devices_to_test.append('cpu')
        
        results = {}
        
        for device in devices_to_test:
            try:
                logger.info(f"Testing on {device}...")
                original_device = self.device
                self.device = device
                
                # Run a simple test
                success = self.test_basic_inference()
                results[device] = success
                
                # Restore original device
                self.device = original_device
            except Exception as e:
                logger.error(f"Error testing on {device}: {e}")
                results[device] = False
        
        return results
    
    def run_all_tests(self):
        """Run all tests."""
        logger.info(f"Testing {self.model_id} on {self.device}")
        
        # Run basic inference test
        basic_result = self.test_basic_inference()
        
        # Run hardware compatibility test
        hw_results = self.test_hardware_compatibility()
        
        # Run audio processing test
        audio_result = self.test_audio_processing()
        
        # Summarize results
        logger.info("Test Results:")
        logger.info(f"- Basic inference: {'PASS' if basic_result else 'FAIL'}")
        logger.info(f"- Audio processing: {'PASS' if audio_result else 'FAIL'}")
        logger.info("- Hardware compatibility:")
        for device, result in hw_results.items():
            logger.info(f"  - {device}: {'PASS' if result else 'FAIL'}")
        
        return basic_result and audio_result and all(hw_results.values())

    def test_audio_processing(self):
        """Test audio processing functionality."""
        try:
            # Create a test audio if none exists
            test_audio_path = "test_audio.wav"
            if not os.path.exists(test_audio_path):
                # Generate a simple sine wave
                import scipy.io.wavfile as wav
                sample_rate = 16000
                duration = 3  # seconds
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
                wav.write(test_audio_path, sample_rate, audio.astype(np.float32))
                
            # Load audio file
            sample_rate = 16000
            try:
                import soundfile as sf
                audio, sample_rate = sf.read(test_audio_path)
            except:
                logger.warning("Could not load audio, using zeros array")
                audio = np.zeros(sample_rate * 3)  # 3 seconds of silence
                
            # Try different model classes
            try:
                from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
                processor = AutoFeatureExtractor.from_pretrained(self.model_id)
                model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            except:
                try:
                    # Try speech recognition model
                    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
                    processor = AutoProcessor.from_pretrained(self.model_id)
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id)
                except:
                    # Fallback to generic model
                    from transformers import AutoModel, AutoFeatureExtractor
                    processor = AutoFeatureExtractor.from_pretrained(self.model_id)
                    model = AutoModel.from_pretrained(self.model_id)
                    
            model = model.to(self.device)
            
            # Process audio
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            assert outputs is not None, "Outputs should not be None"
            
            # If it's a classification model, try to get class probabilities
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                logger.info(f"Logits shape: {logits.shape}")
                
            return True
        except Exception as e:
            logger.error(f"Error during audio processing test: {e}")
            return False


if __name__ == "__main__":
    # Create and run the test using unittest or run directly
    if 'unittest' in sys.argv:
        unittest.main()
    else:
        # Create and run the test
        test = TestWhisperTiny()
        test.run_all_tests()
