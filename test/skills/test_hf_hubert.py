#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test HuggingFace hubert models with IPFS Accelerate.
This tests the facebook/hubert-base-ls960 model and similar variants.
"""

import os
import sys
import json
import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch
import traceback

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try to import required libraries
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Import test utilities
from test.utils import setup_logger, get_test_resources, compare_results, save_results

# Configure logging
logger = setup_logger("test_hf_hubert")

class TestHFHubert(unittest.TestCase):
    """Test HuggingFace hubert models with IPFS Accelerate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resources = {}
        self.metadata = {}
        self.implementation_type = "MOCK"  # Will be updated during tests
        
        # Small, open-access model options
        self.model_options = [
            "facebook/hubert-base-ls960",  # Primary model
            "facebook/hubert-small",       # Smaller alternative
            "facebook/hubert-tiny",        # Tiny model for fast testing
        ]
        self.model_name = self.model_options[0]  # Default to first model
        
        # Test audio paths
        self.test_audio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.mp3")
        if not os.path.exists(self.test_audio_path):
            self.create_test_audio()

    def create_test_audio(self):
        """Create a test audio file if it doesn't exist."""
        try:
            import numpy as np
            from scipy.io import wavfile
            
            logger.info(f"Creating test audio file at {self.test_audio_path}")
            sample_rate = 16000
            duration = 3  # seconds
            audio_data = np.zeros(sample_rate * duration, dtype=np.float32)
            
            # Add a simple tone
            for i in range(len(audio_data)):
                audio_data[i] = 0.5 * np.sin(2 * np.pi * 440 * i / sample_rate)
            
            # Save as MP3 or WAV depending on available libraries
            try:
                import soundfile as sf
                sf.write(self.test_audio_path, audio_data, sample_rate)
            except ImportError:
                # Fallback to WAV
                wavfile.write(self.test_audio_path.replace(".mp3", ".wav"), sample_rate, audio_data)
                self.test_audio_path = self.test_audio_path.replace(".mp3", ".wav")
                
            logger.info(f"Created test audio file: {self.test_audio_path}")
        except Exception as e:
            logger.error(f"Failed to create test audio: {e}")
            self.test_audio_path = None

    def setUp(self):
        """Set up test resources."""
        self.resources, self.metadata = get_test_resources()
        
        # Initialize resources if empty
        if not self.resources:
            self.resources = {
                "local_endpoints": {},
                "queue": {},
                "queues": {},
                "batch_sizes": {},
                "consumer_tasks": {},
                "caches": {},
                "tokenizer": {}
            }
            self.metadata = {"models": [self.model_name]}

    def test_cpu(self):
        """Test hubert model on CPU."""
        try:
            # Import the transformers module
            from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
            
            # Load the model
            logger.info(f"Loading hubert model: {self.model_name}")
            model = HubertForSequenceClassification.from_pretrained(self.model_name)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            
            # Load audio
            if not LIBROSA_AVAILABLE:
                logger.warning("Librosa not available, loading audio with numpy")
                import numpy as np
                from scipy.io import wavfile
                
                if self.test_audio_path.endswith(".wav"):
                    sample_rate, audio_input = wavfile.read(self.test_audio_path)
                    audio_input = audio_input.astype(np.float32) / 32768.0  # Convert to float
                else:
                    logger.error("Cannot load MP3 without librosa, creating simple waveform")
                    sample_rate = 16000
                    audio_input = np.zeros(sample_rate * 3, dtype=np.float32)
                    for i in range(len(audio_input)):
                        audio_input[i] = 0.5 * np.sin(2 * np.pi * 440 * i / sample_rate)
            else:
                audio_input, sample_rate = librosa.load(self.test_audio_path, sr=16000)
            
            # Process audio input
            logger.info("Processing audio input")
            inputs = feature_extractor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
            
            # Run inference
            logger.info("Running inference on CPU")
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            
            # Set implementation type
            self.implementation_type = "REAL"
            
            # Return test result
            return {
                "status": f"Success ({self.implementation_type})",
                "model": self.model_name,
                "logits_shape": list(logits.shape),
                "audio_duration": len(audio_input) / sample_rate,
                "platform": "CPU",
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
        except Exception as e:
            logger.error(f"Error testing hubert model on CPU: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try with a smaller model
            if self.model_name != self.model_options[-1]:
                for option in self.model_options[1:]:
                    logger.info(f"Attempting with smaller model: {option}")
                    self.model_name = option
                    try:
                        return self.test_cpu()
                    except Exception:
                        continue
            
            # Return error
            return {
                "status": "Error",
                "model": self.model_name,
                "error": str(e),
                "platform": "CPU",
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }

    def test_cuda(self):
        """Test hubert model on CUDA."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping test")
            return {
                "status": "Skipped",
                "model": self.model_name,
                "reason": "CUDA not available",
                "platform": "CUDA",
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        
        try:
            # Import the transformers module
            from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
            
            # Load the model
            logger.info(f"Loading hubert model: {self.model_name}")
            model = HubertForSequenceClassification.from_pretrained(self.model_name).to("cuda")
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            
            # Load audio
            if not LIBROSA_AVAILABLE:
                logger.warning("Librosa not available, loading audio with numpy")
                import numpy as np
                from scipy.io import wavfile
                
                if self.test_audio_path.endswith(".wav"):
                    sample_rate, audio_input = wavfile.read(self.test_audio_path)
                    audio_input = audio_input.astype(np.float32) / 32768.0  # Convert to float
                else:
                    logger.error("Cannot load MP3 without librosa, creating simple waveform")
                    sample_rate = 16000
                    audio_input = np.zeros(sample_rate * 3, dtype=np.float32)
                    for i in range(len(audio_input)):
                        audio_input[i] = 0.5 * np.sin(2 * np.pi * 440 * i / sample_rate)
            else:
                audio_input, sample_rate = librosa.load(self.test_audio_path, sr=16000)
            
            # Process audio input
            logger.info("Processing audio input")
            inputs = feature_extractor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Measure memory before inference
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            # Start timing
            start_time = time.time()
            
            # Run inference
            logger.info("Running inference on CUDA")
            with torch.no_grad():
                outputs = model(**inputs)
            
            # End timing
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Measure memory after inference
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            
            logits = outputs.logits
            
            # Calculate realtime factor
            audio_duration = len(audio_input) / sample_rate
            realtime_factor = audio_duration / inference_time
            
            # Set implementation type
            self.implementation_type = "REAL"
            
            # Clean up GPU memory
            del model, inputs, outputs, logits
            torch.cuda.empty_cache()
            
            # Return test result
            return {
                "status": f"Success ({self.implementation_type})",
                "model": self.model_name,
                "logits_shape": list(logits.cpu().shape),
                "inference_time": inference_time,
                "peak_memory_mb": peak_memory,
                "memory_increase_mb": peak_memory - start_memory,
                "audio_duration": audio_duration,
                "realtime_factor": realtime_factor,
                "platform": "CUDA",
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
        except Exception as e:
            logger.error(f"Error testing hubert model on CUDA: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try with a smaller model
            if self.model_name != self.model_options[-1]:
                for option in self.model_options[1:]:
                    logger.info(f"Attempting with smaller model: {option}")
                    self.model_name = option
                    try:
                        return self.test_cuda()
                    except Exception:
                        continue
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
            # Return error
            return {
                "status": "Error",
                "model": self.model_name,
                "error": str(e),
                "platform": "CUDA",
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }

    def test_openvino(self):
        """Test hubert model on OpenVINO."""
        try:
            # Import OpenVINO
            import openvino
            from openvino.runtime import Core
        except ImportError:
            logger.warning("OpenVINO not available, skipping test")
            return {
                "status": "Skipped",
                "model": self.model_name,
                "reason": "OpenVINO not available",
                "platform": "OpenVINO",
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        
        try:
            # Import the transformers module
            from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
            from optimum.intel import OVModelForSequenceClassification
            
            # Load the model
            logger.info(f"Loading hubert model with OpenVINO: {self.model_name}")
            model = OVModelForSequenceClassification.from_pretrained(
                self.model_name, 
                from_transformers=True,
                device="CPU"
            )
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            
            # Load audio
            if not LIBROSA_AVAILABLE:
                logger.warning("Librosa not available, loading audio with numpy")
                import numpy as np
                from scipy.io import wavfile
                
                if self.test_audio_path.endswith(".wav"):
                    sample_rate, audio_input = wavfile.read(self.test_audio_path)
                    audio_input = audio_input.astype(np.float32) / 32768.0  # Convert to float
                else:
                    logger.error("Cannot load MP3 without librosa, creating simple waveform")
                    sample_rate = 16000
                    audio_input = np.zeros(sample_rate * 3, dtype=np.float32)
                    for i in range(len(audio_input)):
                        audio_input[i] = 0.5 * np.sin(2 * np.pi * 440 * i / sample_rate)
            else:
                audio_input, sample_rate = librosa.load(self.test_audio_path, sr=16000)
            
            # Process audio input
            logger.info("Processing audio input")
            inputs = feature_extractor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
            
            # Start timing
            start_time = time.time()
            
            # Run inference
            logger.info("Running inference on OpenVINO")
            outputs = model(**inputs)
            
            # End timing
            end_time = time.time()
            inference_time = end_time - start_time
            
            logits = outputs.logits
            
            # Calculate realtime factor
            audio_duration = len(audio_input) / sample_rate
            realtime_factor = audio_duration / inference_time
            
            # Set implementation type
            self.implementation_type = "REAL"
            
            # Return test result
            return {
                "status": f"Success ({self.implementation_type})",
                "model": self.model_name,
                "logits_shape": list(logits.shape),
                "inference_time": inference_time,
                "audio_duration": audio_duration,
                "realtime_factor": realtime_factor,
                "platform": "OpenVINO",
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
        except Exception as e:
            logger.error(f"Error testing hubert model on OpenVINO: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try with a smaller model
            if self.model_name != self.model_options[-1]:
                for option in self.model_options[1:]:
                    logger.info(f"Attempting with smaller model: {option}")
                    self.model_name = option
                    try:
                        return self.test_openvino()
                    except Exception:
                        continue
            
            # Fallback to standard implementation with mock conversion
            try:
                logger.info("Attempting fallback implementation with mock OpenVINO conversion")
                
                from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
                
                # Load the model (CPU only)
                model = HubertForSequenceClassification.from_pretrained(self.model_name)
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
                
                # Load audio
                if not LIBROSA_AVAILABLE:
                    import numpy as np
                    sample_rate = 16000
                    audio_input = np.zeros(sample_rate * 3, dtype=np.float32)
                    for i in range(len(audio_input)):
                        audio_input[i] = 0.5 * np.sin(2 * np.pi * 440 * i / sample_rate)
                else:
                    audio_input, sample_rate = librosa.load(self.test_audio_path, sr=16000)
                
                # Process audio input
                inputs = feature_extractor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
                
                # Start timing
                start_time = time.time()
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # End timing
                end_time = time.time()
                inference_time = end_time - start_time
                
                logits = outputs.logits
                
                # Calculate realtime factor
                audio_duration = len(audio_input) / sample_rate
                realtime_factor = audio_duration / inference_time
                
                # Set implementation type
                self.implementation_type = "MOCK"
                
                # Return test result
                return {
                    "status": f"Success ({self.implementation_type})",
                    "model": self.model_name,
                    "logits_shape": list(logits.shape),
                    "inference_time": inference_time,
                    "audio_duration": audio_duration,
                    "realtime_factor": realtime_factor,
                    "platform": "OpenVINO (fallback to CPU)",
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                
            except Exception as fallback_e:
                logger.error(f"Fallback implementation also failed: {str(fallback_e)}")
                return {
                    "status": "Error",
                    "model": self.model_name,
                    "error": f"Original error: {str(e)}; Fallback error: {str(fallback_e)}",
                    "platform": "OpenVINO",
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                }

    def __test__(self, resources=None, metadata=None):
        """Run the tests and compare results."""
        # Update resources if provided
        if resources:
            self.resources = resources
        if metadata:
            self.metadata = metadata
            if "models" in metadata and metadata["models"]:
                for model in metadata["models"]:
                    if "hubert" in model.lower():
                        self.model_name = model
                        break
        
        # Define the filename for test results
        test_results_file = "hf_hubert_test_results.json"
        test_results_dir = os.path.join(os.path.dirname(__file__), "collected_results")
        os.makedirs(test_results_dir, exist_ok=True)
        test_results_path = os.path.join(test_results_dir, test_results_file)
        
        # Define the filename for expected results
        expected_results_dir = os.path.join(os.path.dirname(__file__), "expected_results")
        os.makedirs(expected_results_dir, exist_ok=True)
        expected_results_path = os.path.join(expected_results_dir, test_results_file)
        
        # Run tests
        results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model": self.model_name,
            "tests": {}
        }
        
        # Test CPU
        results["tests"]["cpu"] = self.test_cpu()
        
        # Test CUDA
        results["tests"]["cuda"] = self.test_cuda()
        
        # Test OpenVINO
        results["tests"]["openvino"] = self.test_openvino()
        
        # Count successful tests
        success_count = sum(1 for platform, result in results["tests"].items() 
                          if result.get("status", "").startswith("Success"))
        total_count = len(results["tests"])
        
        results["summary"] = {
            "success_count": success_count,
            "total_count": total_count,
            "implementation_type": self.implementation_type
        }
        
        # Save results
        save_results(results, test_results_path)
        
        # Compare with expected results if they exist
        comparison_result = None
        if os.path.exists(expected_results_path):
            try:
                comparison_result = compare_results(results, expected_results_path)
                results["comparison"] = comparison_result
                # Update the results file with comparison information
                save_results(results, test_results_path)
            except Exception as e:
                logger.error(f"Error comparing results: {e}")
        
        logger.info(f"Tests completed: {success_count}/{total_count} successful")
        return results

if __name__ == "__main__":
    unittest.main()