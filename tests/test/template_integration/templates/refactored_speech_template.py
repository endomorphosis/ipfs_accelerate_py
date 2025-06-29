#!/usr/bin/env python3
"""
Class-based test file for speech/audio models compatible with the refactored test suite.

This template provides a unified testing interface for speech models like Whisper and Wav2Vec2
within the refactored test suite architecture, inheriting from ModelTest.
"""

import os
import sys
import json
import time
import datetime
import logging
import numpy as np
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from unittest.mock import patch, MagicMock, Mock

# Import from the refactored test suite
from refactored_test_suite.model_test import ModelTest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Models registry - Maps model IDs to their specific configurations
SPEECH_MODELS_REGISTRY = {
    "openai/whisper-tiny": {
        "full_name": "Whisper Tiny",
        "architecture": "encoder-decoder",
        "description": "Whisper Tiny model",
        "model_type": "whisper",
        "parameters": "39M",
        "audio_sampling_rate": 16000,
        "embedding_dim": 384,
        "attention_heads": 6,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "model_class": "WhisperForConditionalGeneration",
        "processor_class": "WhisperProcessor",
        "recommended_tasks": ["automatic-speech-recognition", "audio-classification"]
    },
    "openai/whisper-base": {
        "full_name": "Whisper Base",
        "architecture": "encoder-decoder",
        "description": "Whisper Base model",
        "model_type": "whisper",
        "parameters": "74M",
        "audio_sampling_rate": 16000,
        "embedding_dim": 512,
        "attention_heads": 8,
        "encoder_layers": 6,
        "decoder_layers": 6,
        "model_class": "WhisperForConditionalGeneration",
        "processor_class": "WhisperProcessor",
        "recommended_tasks": ["automatic-speech-recognition", "audio-classification"]
    },
    "facebook/wav2vec2-base-960h": {
        "full_name": "Wav2Vec2 Base (960h)",
        "architecture": "encoder-only",
        "description": "Wav2Vec2 Base model fine-tuned on 960h of Librispeech",
        "model_type": "wav2vec2",
        "parameters": "95M",
        "audio_sampling_rate": 16000,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "model_class": "Wav2Vec2ForCTC",
        "processor_class": "Wav2Vec2Processor",
        "recommended_tasks": ["automatic-speech-recognition", "audio-classification"]
    },
    "facebook/hubert-base-ls960": {
        "full_name": "HuBERT Base (LS960)",
        "architecture": "encoder-only",
        "description": "HuBERT Base model trained on LibriSpeech 960h",
        "model_type": "hubert",
        "parameters": "95M",
        "audio_sampling_rate": 16000,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "model_class": "HubertForCTC",
        "processor_class": "Wav2Vec2Processor",
        "recommended_tasks": ["automatic-speech-recognition", "audio-classification"]
    }
}

class TestSpeechModel(ModelTest):
    """Test class for speech/audio models like Whisper, Wav2Vec2, etc."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        
        # Initialize model-specific attributes
        self.model_id = "openai/whisper-tiny"
        
        # Verify model exists in registry
        if self.model_id not in SPEECH_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = SPEECH_MODELS_REGISTRY["openai/whisper-tiny"]
        else:
            self.model_info = SPEECH_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "automatic-speech-recognition"
        self.model_type = self.model_info["model_type"]
        self.model_class = self.model_info["model_class"]
        self.processor_class = self.model_info["processor_class"]
        self.description = self.model_info["description"]
        self.audio_sampling_rate = self.model_info["audio_sampling_rate"]
        
        # Define test inputs
        self.test_audio_path = "test_audio.wav"
        self.test_audio_duration = 3  # seconds
        
        # Setup hardware detection
        self.setup_hardware()
    
    def setup_hardware(self):
        """Set up hardware detection."""
        try:
            # Try to import hardware detection capabilities
            from generators.hardware.hardware_detection import (
                HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
                detect_all_hardware
            )
            hardware_info = detect_all_hardware()
        except ImportError:
            # Fallback to manual detection
            import torch
            
            # Basic hardware detection
            self.has_cuda = torch.cuda.is_available()
            self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            self.has_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            
            # Check for OpenVINO
            try:
                import openvino
                self.has_openvino = True
            except ImportError:
                self.has_openvino = False
            
            # WebNN/WebGPU are not directly accessible in Python
            self.has_webnn = False
            self.has_webgpu = False
        
        # Configure preferred device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
        
        logger.info(f"Using device: {self.device}")
    
    def tearDown(self):
        """Clean up resources after the test."""
        # Release any resources that need cleanup
        super().tearDown()
    
    def create_test_audio(self):
        """Create a test audio file if it doesn't exist."""
        if not os.path.exists(self.test_audio_path):
            try:
                # Generate a simple sine wave
                import scipy.io.wavfile as wav
                sample_rate = self.audio_sampling_rate
                duration = self.test_audio_duration
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
                wav.write(self.test_audio_path, sample_rate, audio.astype(np.float32))
                logger.info(f"Created test audio file: {self.test_audio_path}")
                return True
            except Exception as e:
                logger.error(f"Error creating test audio: {e}")
                return False
        return True
    
    def load_audio(self):
        """Load the audio data from file."""
        try:
            # Ensure test audio exists
            if not os.path.exists(self.test_audio_path):
                self.create_test_audio()
            
            # Try to load with soundfile (preferred)
            try:
                import soundfile as sf
                audio, sample_rate = sf.read(self.test_audio_path)
                return audio, sample_rate
            except ImportError:
                # Fallback to scipy
                try:
                    import scipy.io.wavfile as wav
                    sample_rate, audio = wav.read(self.test_audio_path)
                    # Convert to float if needed
                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
                    return audio, sample_rate
                except ImportError:
                    # Last resort: create a dummy audio array
                    logger.warning("Could not load audio libraries, using dummy audio")
                    dummy_audio = np.zeros(self.audio_sampling_rate * self.test_audio_duration, dtype=np.float32)
                    return dummy_audio, self.audio_sampling_rate
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            # Create a dummy audio array as fallback
            dummy_audio = np.zeros(self.audio_sampling_rate * self.test_audio_duration, dtype=np.float32)
            return dummy_audio, self.audio_sampling_rate
    
    def load_model(self, model_id=None):
        """Load the model for testing."""
        model_id = model_id or self.model_id
        
        try:
            import torch
            import transformers
            
            # Get model and processor classes
            model_class = getattr(transformers, self.model_class)
            processor_class = getattr(transformers, self.processor_class, transformers.AutoProcessor)
            
            # Load the processor
            processor = processor_class.from_pretrained(model_id)
            
            # Load the model
            model = model_class.from_pretrained(model_id)
            
            # Move to appropriate device
            model = model.to(self.device)
            
            return {"model": model, "processor": processor}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_input(self):
        """Prepare input for the model."""
        # Load audio data
        audio, sample_rate = self.load_audio()
        
        return {
            "audio": audio,
            "sample_rate": sample_rate
        }
    
    def test_model_loading(self):
        """Test that the model loads correctly."""
        model_components = self.load_model()
        
        # Verify that model and processor were loaded
        self.assertIsNotNone(model_components["model"])
        self.assertIsNotNone(model_components["processor"])
        
        logger.info("Model loaded successfully")
    
    def test_basic_inference(self):
        """Test basic inference with the model."""
        import torch
        
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        processor = model_components["processor"]
        
        # Prepare input
        input_data = self.prepare_input()
        audio, sample_rate = input_data["audio"], input_data["sample_rate"]
        
        # Process the audio based on model type
        if self.model_type == "whisper":
            # For Whisper models
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        else:
            # For Wav2Vec2, HuBERT, etc.
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        
        # Move inputs to device if needed
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        
        # Model-specific output checks
        if self.model_type == "whisper":
            # Check for logits in output
            self.assertTrue(hasattr(outputs, "logits"))
        else:
            # For encoder-only models like Wav2Vec2, HuBERT
            if hasattr(outputs, "logits"):
                self.assertGreater(outputs.logits.shape[0], 0)
            else:
                # Some models might have different output structures
                logger.info(f"Output structure: {type(outputs)}")
        
        logger.info("Basic inference successful")
    
    def test_transcription(self):
        """Test transcription with the model."""
        import torch
        
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        processor = model_components["processor"]
        
        # Prepare input
        input_data = self.prepare_input()
        audio, sample_rate = input_data["audio"], input_data["sample_rate"]
        
        # Process audio
        if self.model_type == "whisper":
            # For Whisper models
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = model.generate(inputs["input_features"], max_length=100)
            
            # Decode the outputs
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            # For Wav2Vec2, HuBERT, etc.
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Decode the outputs
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
        
        # Verify transcription
        self.assertIsNotNone(transcription)
        self.assertIsInstance(transcription, str)
        
        logger.info(f"Transcription: {transcription}")
        logger.info("Transcription successful")
    
    def test_hardware_compatibility(self):
        """Test the model's compatibility with different hardware platforms."""
        devices_to_test = []
        
        # Add available devices
        if self.has_cuda:
            devices_to_test.append('cuda')
        if self.has_mps:
            devices_to_test.append('mps')
        
        # Always test CPU
        if 'cpu' not in devices_to_test:
            devices_to_test.append('cpu')
        
        results = {}
        
        # Test on each device
        for device in devices_to_test:
            original_device = self.device
            try:
                logger.info(f"Testing on {device}...")
                self.device = device
                
                # Load model and prepare input
                model_components = self.load_model()
                model = model_components["model"]
                processor = model_components["processor"]
                
                input_data = self.prepare_input()
                audio, sample_rate = input_data["audio"], input_data["sample_rate"]
                
                # Process audio
                inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Run inference (simple forward pass only for hardware test)
                import torch
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Verify results
                results[device] = True
                logger.info(f"Test on {device} successful")
                
            except Exception as e:
                logger.error(f"Error testing on {device}: {e}")
                results[device] = False
            finally:
                # Restore original device
                self.device = original_device
        
        # Verify at least one device works
        self.assertTrue(any(results.values()), "Model should work on at least one device")
        
        # Log results
        for device, success in results.items():
            logger.info(f"Device {device}: {'Success' if success else 'Failed'}")
    
    def test_openvino_compatibility(self):
        """Test compatibility with OpenVINO, if available."""
        if not self.has_openvino:
            logger.info("OpenVINO not available, skipping test")
            self.skipTest("OpenVINO not available")
        
        try:
            # Try to import OpenVINO integration
            try:
                from optimum.intel import OVModelForSpeechSeq2Seq, OVModelForCTC
                optimum_available = True
            except ImportError:
                logger.warning("optimum-intel not available, using direct OpenVINO conversion")
                optimum_available = False
            
            # Load processor
            processor = self.load_model()["processor"]
            
            # Load model with OpenVINO
            if optimum_available:
                # Use appropriate model class based on model type
                if self.model_type == "whisper":
                    model = OVModelForSpeechSeq2Seq.from_pretrained(
                        self.model_id,
                        export=True,
                        provider="CPU"
                    )
                else:
                    # For Wav2Vec2, HuBERT
                    model = OVModelForCTC.from_pretrained(
                        self.model_id,
                        export=True,
                        provider="CPU"
                    )
            else:
                # Direct OpenVINO conversion (fallback)
                import torch
                from openvino.runtime import Core
                
                # Load PyTorch model
                pytorch_model = self.load_model()["model"].to("cpu")
                
                # Prepare input for tracing
                input_data = self.prepare_input()
                processor = self.load_model()["processor"]
                inputs = processor(input_data["audio"], sampling_rate=input_data["sample_rate"], return_tensors="pt")
                
                # Convert model to ONNX format
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
                    onnx_path = tmp.name
                    torch.onnx.export(
                        pytorch_model,
                        tuple(inputs.values()),
                        onnx_path,
                        input_names=list(inputs.keys()),
                        output_names=["logits"],
                        dynamic_axes={
                            key: {0: "batch_size", 1: "sequence_length"} 
                            for key in inputs.keys()
                        }
                    )
                    
                    # Load the model with OpenVINO
                    core = Core()
                    ov_model = core.read_model(onnx_path)
                    model = core.compile_model(ov_model, "CPU")
            
            # Prepare input
            input_data = self.prepare_input()
            audio, sample_rate = input_data["audio"], input_data["sample_rate"]
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            
            # Run inference
            if optimum_available:
                outputs = model(**inputs)
            else:
                # For direct OpenVINO conversion, we need to convert inputs to numpy
                inputs_np = {k: v.numpy() for k, v in inputs.items()}
                outputs = model(inputs_np)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            
            logger.info("OpenVINO compatibility test successful")
        except ImportError:
            logger.warning("optimum-intel not available, skipping detailed test")
            self.skipTest("optimum-intel not available")
        except Exception as e:
            logger.error(f"Error in OpenVINO test: {e}")
            raise
    
    def test_pipeline_inference(self):
        """Test the model using HuggingFace pipeline API."""
        try:
            import transformers
            
            # Initialize the pipeline with appropriate task
            pipe = transformers.pipeline(
                self.task,
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Load audio
            input_data = self.prepare_input()
            audio, sample_rate = input_data["audio"], input_data["sample_rate"]
            
            # Run inference
            outputs = pipe(audio, sampling_rate=sample_rate)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            
            # Check for model-specific outputs
            if self.task == "automatic-speech-recognition":
                if isinstance(outputs, dict) and "text" in outputs:
                    transcription = outputs["text"]
                    logger.info(f"Transcription: {transcription}")
                else:
                    logger.info(f"Output structure: {outputs}")
            else:
                logger.info(f"Pipeline output: {outputs}")
            
            logger.info("Pipeline inference successful")
        except Exception as e:
            logger.error(f"Error in pipeline inference: {e}")
            self.fail(f"Pipeline inference failed: {e}")
    
    def run_all_tests(self):
        """Run all tests for this model."""
        test_methods = [method for method in dir(self) if method.startswith('test_')]
        results = {}
        
        for method in test_methods:
            try:
                logger.info(f"Running {method}...")
                getattr(self, method)()
                results[method] = "PASS"
            except Exception as e:
                logger.error(f"Error in {method}: {e}")
                results[method] = f"FAIL: {str(e)}"
        
        return results


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test speech models with refactored test suite")
    parser.add_argument("--model", type=str, default="openai/whisper-tiny", 
                       help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to test on (cpu, cuda, etc.)")
    parser.add_argument("--audio", type=str, help="Path to audio file for testing")
    parser.add_argument("--task", type=str, help="Task to test (automatic-speech-recognition, audio-classification)")
    parser.add_argument("--save-results", action="store_true", help="Save test results to file")
    
    args = parser.parse_args()
    
    # Create test instance
    test = TestSpeechModel()
    
    # Override settings if specified
    if args.model:
        test.model_id = args.model
    if args.device:
        test.device = args.device
    if args.audio:
        test.test_audio_path = args.audio
    if args.task:
        test.task = args.task
    
    # Run tests
    test.setUp()
    results = test.run_all_tests()
    test.tearDown()
    
    # Print results
    print("\nTest Results:")
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    # Save results if requested
    if args.save_results:
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{args.model.replace('/', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "w") as f:
            json.dump({
                "model": args.model,
                "device": test.device,
                "task": test.task,
                "results": results,
                "timestamp": datetime.datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()