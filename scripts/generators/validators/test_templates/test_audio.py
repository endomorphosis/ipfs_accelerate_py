#!/usr/bin/env python3
"""
Test file for openai/whisper-tiny model.

This file is auto-generated using the template-based test generator.
Generated: 2025-03-10 01:31:40
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Testwhispertiny:
    """Test class for openai/whisper-tiny model."""
    
    def __init__(self):
        """Initialize the test with model details and hardware detection."""
        self.model_name = "openai/whisper-tiny"
        self.model_type = "audio"
        self.setup_hardware()
    
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
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Get model
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(self.device)
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None
    
    def custom_method(self):
        pass
    
    def test_basic_inference(self):
        """Run a basic inference test with the model."""
        model, tokenizer = self.get_model()
        
        if model is None or tokenizer is None:
            logger.error("Failed to load model or tokenizer")
            return False
        
        try:
            # Prepare input
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
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
                        # Check output shape and values
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
    
    def run(self):
        """Run all tests."""
        logger.info(f"Testing {self.model_name} on {self.device}")
        
        # Run basic inference test
        basic_result = self.test_basic_inference()
        
        # Run hardware compatibility test
        hw_results = self.test_hardware_compatibility()
        
        # Summarize results
        logger.info("Test Results:")
        logger.info(f"- Basic inference: {'PASS' if basic_result else 'FAIL'}")
        logger.info("- Hardware compatibility:")
        for device, result in hw_results.items():
            logger.info(f"  - {device}: {'PASS' if result else 'FAIL'}")
        
        return basic_result and all(hw_results.values())


# No model specific code


if __name__ == "__main__":
    # Create and run the test
    test = Testwhispertiny()
    test.run()
