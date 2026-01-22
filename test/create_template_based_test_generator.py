#!/usr/bin/env python3
"""
Template-Based Test Generator

This script generates test files from templates stored in a database.
It supports generating test files for specific models, hardware platforms,
and model families.

Usage:
    python create_template_based_test_generator.py --model MODEL_NAME [--output OUTPUT_FILE]
    python create_template_based_test_generator.py --family MODEL_FAMILY [--output OUTPUT_DIR]
    python create_template_based_test_generator.py --list-models
    python create_template_based_test_generator.py --list-families
"""

import os
import sys
import argparse
import json
import logging
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import template validator
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from generators.validators.template_validator_integration import (
        validate_template_for_generator,
        validate_template_file_for_generator
    )
    HAS_VALIDATOR = True
    logger.info("Template validator loaded successfully")
except ImportError:
    HAS_VALIDATOR = False
    logger.warning("Template validator not found. Templates will not be validated.")
    
    # Define minimal validation function
    def validate_template_for_generator(template_content, generator_type, **kwargs):
        return True, []
        
    def validate_template_file_for_generator(file_path, generator_type, **kwargs):
        return True, []

# Check for DuckDB availability
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    logger.warning("DuckDB not available. Will use JSON-based storage.")

# Model family mapping
MODEL_FAMILIES = {
    "text_embedding": ["bert", "sentence-transformers", "distilbert", "roberta", "mpnet"],
    "text_generation": ["gpt2", "llama", "opt", "t5", "bloom", "mistral", "qwen", "falcon"],
    "vision": ["vit", "resnet", "detr", "deit", "convnext", "beit"],
    "audio": ["whisper", "wav2vec2", "hubert", "speecht5", "clap"],
    "multimodal": ["clip", "llava", "xclip", "blip", "flava"]
}

# Reverse mapping from model name to family
MODEL_TO_FAMILY = {}
for family, models in MODEL_FAMILIES.items():
    for model in models:
        MODEL_TO_FAMILY[model] = family

# Standard template for a test file with hardware support
STANDARD_TEMPLATE = '''#!/usr/bin/env python3
"""
Test file for {{model_name}} model.

This file is auto-generated using the template-based test generator.
Generated: {{generation_date}}
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

class Test{{model_class_name}}:
    """Test class for {{model_name}} model."""
    
    def __init__(self):
        """Initialize the test with model details and hardware detection."""
        self.model_name = "{{model_name}}"
        self.model_type = "{{model_type}}"
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
    
    {{custom_model_loading}}
    
    def test_basic_inference(self):
        """Run a basic inference test with the model."""
        model, tokenizer = self.get_model()
        
        if model is None or tokenizer is None:
            logger.error("Failed to load model or tokenizer")
            return False
        
        try:
            # Prepare input
            {{model_input_code}}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            {{output_check_code}}
            
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


{{model_specific_code}}


if __name__ == "__main__":
    # Create and run the test
    test = Test{{model_class_name}}()
    test.run()
'''

# Custom model input code by model type
MODEL_INPUT_TEMPLATES = {
    "text_embedding": '''            # Prepare text input
            text = "This is a sample text for testing the {{model_name}} model."
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}''',
    
    "text_generation": '''            # Prepare text input for generation
            text = "Generate a short explanation of machine learning:"
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}''',
    
    "vision": '''            # Prepare image input
            from PIL import Image
            import requests
            from io import BytesIO
            from transformers import AutoImageProcessor

            # Create a test image if none exists
            test_image_path = "test_image.jpg"
            if not os.path.exists(test_image_path):
                # Create a simple test image (black and white gradient)
                import numpy as np
                from PIL import Image
                size = 224
                img_array = np.zeros((size, size, 3), dtype=np.uint8)
                for i in range(size):
                    for j in range(size):
                        img_array[i, j, :] = (i + j) % 256
                img = Image.fromarray(img_array)
                img.save(test_image_path)

            # Load the image
            image = Image.open(test_image_path)

            # Get image processor
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}''',
    
    "audio": '''            # Prepare audio input
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
            inputs = {k: v.to(self.device) for k, v in inputs.items()}''',
    
    "multimodal": '''            # Prepare multimodal input (text and image)
            from PIL import Image
            from transformers import AutoProcessor

            # Create a test image if none exists
            test_image_path = "test_image.jpg"
            if not os.path.exists(test_image_path):
                # Create a simple test image
                import numpy as np
                from PIL import Image
                size = 224
                img_array = np.zeros((size, size, 3), dtype=np.uint8)
                for i in range(size):
                    for j in range(size):
                        img_array[i, j, :] = (i + j) % 256
                img = Image.fromarray(img_array)
                img.save(test_image_path)

            # Load the image
            image = Image.open(test_image_path)

            # Prepare text
            text = "What's in this image?"

            # Get processor
            processor = AutoProcessor.from_pretrained(self.model_name)
            inputs = processor(text=text, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}'''
}

# Custom output check code by model type
OUTPUT_CHECK_TEMPLATES = {
    "text_embedding": '''            # Check output shape and values
            assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
            assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
            assert outputs.last_hidden_state.shape[1] > 0, "Sequence length should be positive"
            logger.info(f"Output shape: {outputs.last_hidden_state.shape}")''',
    
    "text_generation": '''            # For generation models, just check that we have valid output tensors
            assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
            assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
            assert outputs.last_hidden_state.shape[1] > 0, "Sequence length should be positive"
            logger.info(f"Output shape: {outputs.last_hidden_state.shape}")''',
    
    "vision": '''            # Check output shape and values
            assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
            assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
            logger.info(f"Output shape: {outputs.last_hidden_state.shape}")''',
    
    "audio": '''            # Check output shape and values
            assert outputs is not None, "Outputs should not be None"
            if hasattr(outputs, "last_hidden_state"):
                assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
                logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
            else:
                # Some audio models have different output structures
                logger.info(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")''',
    
    "multimodal": '''            # Check output shape and values
            assert outputs is not None, "Outputs should not be None"
            if hasattr(outputs, "last_hidden_state"):
                assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
                logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
            elif hasattr(outputs, "logits"):
                assert outputs.logits.shape[0] == 1, "Batch size should be 1"
                logger.info(f"Logits shape: {outputs.logits.shape}")
            else:
                # Some multimodal models have different output structures
                logger.info(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")'''
}

# Custom model loading code by model type
CUSTOM_MODEL_LOADING_TEMPLATES = {
    "text_embedding": '''def get_model_specific(self):
        """Load model with specialized configuration."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Get tokenizer with specific settings
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                truncation_side="right",
                use_fast=True
            )
            
            # Get model with specific settings
            model = AutoModel.from_pretrained(
                self.model_name,
                torchscript=True if self.device == 'cpu' else False
            )
            model = model.to(self.device)
            
            # Put model in evaluation mode
            model.eval()
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model with specific settings: {e}")
            return None, None''',
    
    "text_generation": '''def get_model_specific(self):
        """Load model with specialized configuration for text generation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Get tokenizer with specific settings
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                truncation_side="left",
                use_fast=True
            )
            
            # Get model with specific settings for generation
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == 'cuda' else None
            )
            model = model.to(self.device)
            
            # Put model in evaluation mode
            model.eval()
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model with specific settings: {e}")
            return None, None''',
    
    "vision": '''def get_model_specific(self):
        """Load model with specialized configuration for vision tasks."""
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            
            # Get image processor
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            # Get model with vision-specific settings
            model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                torchscript=True if self.device == 'cpu' else False
            )
            model = model.to(self.device)
            
            # Put model in evaluation mode
            model.eval()
            
            return model, processor
        except Exception as e:
            logger.error(f"Error loading vision model with specific settings: {e}")
            
            # Fallback to generic model
            from transformers import AutoModel, AutoFeatureExtractor
            try:
                processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)
                model = model.to(self.device)
                model.eval()
                return model, processor
            except Exception as e2:
                logger.error(f"Error in fallback loading: {e2}")
                return None, None''',
    
    "audio": '''def get_model_specific(self):
        """Load model with specialized configuration for audio processing."""
        try:
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
            
            # Get feature extractor
            processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            
            # Get model with audio-specific settings
            model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
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
                processor = AutoProcessor.from_pretrained(self.model_name)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
                model = model.to(self.device)
                model.eval()
                return model, processor
            except Exception as e2:
                logger.error(f"Error in alternative loading: {e2}")
                
                # Fallback to generic model
                try:
                    from transformers import AutoModel, AutoFeatureExtractor
                    processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                    model = AutoModel.from_pretrained(self.model_name)
                    model = model.to(self.device)
                    model.eval()
                    return model, processor
                except Exception as e3:
                    logger.error(f"Error in fallback loading: {e3}")
                    return None, None''',
    
    "multimodal": '''def get_model_specific(self):
        """Load model with specialized configuration for multimodal tasks."""
        try:
            from transformers import AutoProcessor, AutoModel
            
            # Get processor for multimodal inputs
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Get model with multimodal-specific settings
            model = AutoModel.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == 'cuda' else None
            )
            model = model.to(self.device)
            
            # Put model in evaluation mode
            model.eval()
            
            return model, processor
        except Exception as e:
            logger.error(f"Error loading multimodal model with specific settings: {e}")
            
            # Try alternative model class
            try:
                from transformers import CLIPModel, CLIPProcessor
                processor = CLIPProcessor.from_pretrained(self.model_name)
                model = CLIPModel.from_pretrained(self.model_name)
                model = model.to(self.device)
                model.eval()
                return model, processor
            except Exception as e2:
                logger.error(f"Error in alternative loading: {e2}")
                return None, None'''
}

# Model-specific code by model type
MODEL_SPECIFIC_CODE_TEMPLATES = {
    "text_embedding": '''# Additional methods for text embedding models
def test_embedding_similarity(self):
    """Test embedding similarity functionality."""
    model, tokenizer = self.get_model()
    
    if model is None or tokenizer is None:
        logger.error("Failed to load model or tokenizer")
        return False
    
    try:
        # Prepare input texts
        texts = [
            "This is a sample text for testing embeddings.",
            "Another example text that is somewhat similar.",
            "This text is completely different from the others."
        ]
        
        # Get embeddings
        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Use mean pooling to get sentence embedding
            embedding = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(embedding)
        
        # Calculate similarities
        import torch.nn.functional as F
        
        sim_0_1 = F.cosine_similarity(embeddings[0], embeddings[1])
        sim_0_2 = F.cosine_similarity(embeddings[0], embeddings[2])
        
        logger.info(f"Similarity between text 0 and 1: {sim_0_1.item():.4f}")
        logger.info(f"Similarity between text 0 and 2: {sim_0_2.item():.4f}")
        
        # First two should be more similar than first and third
        assert sim_0_1 > sim_0_2, "Expected similarity between similar texts to be higher"
        
        return True
    except Exception as e:
        logger.error(f"Error during embedding similarity test: {e}")
        return False''',
    
    "text_generation": '''# Additional methods for text generation models
def test_text_generation(self):
    """Test text generation functionality."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    try:
        # Use the specialized model class for generation
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model = model.to(self.device)
        
        # Prepare input
        prompt = "Once upon a time, there was a"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        
        logger.info(f"Generated text: {generated_text}")
        
        # Basic validation
        assert len(generated_text) > len(prompt), "Generated text should be longer than prompt"
        
        return True
    except Exception as e:
        logger.error(f"Error during text generation test: {e}")
        return False''',
    
    "vision": '''# Additional methods for vision models
def test_image_classification(self):
    """Test image classification functionality."""
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    
    try:
        # Create a test image if none exists
        test_image_path = "test_image.jpg"
        if not os.path.exists(test_image_path):
            # Create a simple test image
            import numpy as np
            from PIL import Image
            size = 224
            img_array = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(size):
                for j in range(size):
                    img_array[i, j, :] = (i + j) % 256
            img = Image.fromarray(img_array)
            img.save(test_image_path)
            
        # Load specialized model and processor
        try:
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = AutoModelForImageClassification.from_pretrained(self.model_name)
        except:
            # Fallback to general model
            from transformers import AutoModel, AutoFeatureExtractor
            processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            
        model = model.to(self.device)
        
        # Load and process the image
        from PIL import Image
        image = Image.open(test_image_path)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Check outputs
        assert outputs is not None, "Outputs should not be None"
        
        # If it's a classification model, try to get class probabilities
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            logger.info(f"Top probability: {probabilities.max().item():.4f}")
            
        return True
    except Exception as e:
        logger.error(f"Error during image classification test: {e}")
        return False''',
    
    "audio": '''# Additional methods for audio models
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
            processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        except:
            try:
                # Try speech recognition model
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
                processor = AutoProcessor.from_pretrained(self.model_name)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
            except:
                # Fallback to generic model
                from transformers import AutoModel, AutoFeatureExtractor
                processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)
                
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
        return False''',
    
    "multimodal": '''# Additional methods for multimodal models
def test_multimodal_processing(self):
    """Test multimodal processing functionality."""
    try:
        # Create a test image if none exists
        test_image_path = "test_image.jpg"
        if not os.path.exists(test_image_path):
            # Create a simple test image
            import numpy as np
            from PIL import Image
            size = 224
            img_array = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(size):
                for j in range(size):
                    img_array[i, j, :] = (i + j) % 256
            img = Image.fromarray(img_array)
            img.save(test_image_path)
            
        # Prepare text
        text = "What's in this image?"
            
        # Try different model classes
        try:
            from transformers import AutoProcessor, AutoModel
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
        except:
            try:
                # Try CLIP model
                from transformers import CLIPProcessor, CLIPModel
                processor = CLIPProcessor.from_pretrained(self.model_name)
                model = CLIPModel.from_pretrained(self.model_name)
            except:
                # Fallback
                from transformers import AutoProcessor, AutoModel
                processor = AutoProcessor.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)
                
        model = model.to(self.device)
        
        # Load and process the inputs
        from PIL import Image
        image = Image.open(test_image_path)
        
        # Process multimodal input
        try:
            inputs = processor(text=text, images=image, return_tensors="pt")
        except:
            try:
                # Try CLIP-style
                inputs = processor(text=[text], images=image, return_tensors="pt")
            except:
                # Try another method
                text_inputs = processor.tokenizer(text, return_tensors="pt")
                image_inputs = processor.image_processor(image, return_tensors="pt")
                inputs = {**text_inputs, **image_inputs}
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Check outputs
        assert outputs is not None, "Outputs should not be None"
        
        # If it's a classification/similarity model, check for specific outputs
        if hasattr(outputs, "logits_per_image") or hasattr(outputs, "logits_per_text"):
            logger.info("Model produces image-text similarity scores")
            
        return True
    except Exception as e:
        logger.error(f"Error during multimodal processing test: {e}")
        return False'''
}

class TemplateBasedTestGenerator:
    """
    Generator for test files from templates.
    """
    
    def __init__(self, db_path: str = "../generators/templates/template_db.json", args=None):
        """
        Initialize the generator with database connection.
        
        Args:
            db_path: Path to the database file
            args: Command line arguments
        """
        self.db_path = db_path
        self.templates = {}
        self.args = args or argparse.Namespace()  # Default empty args
        
        # Set default validation behavior if not specified
        if not hasattr(self.args, "validate"):
            self.args.validate = HAS_VALIDATOR
        if not hasattr(self.args, "skip_validation"):
            self.args.skip_validation = False
        if not hasattr(self.args, "strict_validation"):
            self.args.strict_validation = False
            
        self.load_templates()
    
    def load_templates(self):
        """Load templates from the database."""
        if not HAS_DUCKDB or self.db_path.endswith('.json'):
            # Use JSON-based storage
            json_db_path = self.db_path if self.db_path.endswith('.json') else self.db_path.replace('.duckdb', '.json')
            
            if not os.path.exists(json_db_path):
                logger.error(f"JSON database file not found: {json_db_path}")
                return
            
            try:
                # Load the JSON database
                with open(json_db_path, 'r') as f:
                    template_db = json.load(f)
                
                if 'templates' not in template_db:
                    logger.error("No templates found in JSON database")
                    return
                
                self.templates = template_db['templates']
                logger.info(f"Loaded {len(self.templates)} templates from JSON database")
                
                # Check how many templates have valid syntax
                valid_count = 0
                for template_id, template_data in self.templates.items():
                    try:
                        content = template_data.get('template', '')
                        ast.parse(content)
                        valid_count += 1
                    except SyntaxError:
                        pass
                
                logger.info(f"Found {valid_count}/{len(self.templates)} templates with valid syntax")
                
            except Exception as e:
                logger.error(f"Error loading templates from JSON database: {str(e)}")
        else:
            # Use DuckDB
            try:
                import duckdb
                
                if not os.path.exists(self.db_path):
                    logger.error(f"Database file not found: {self.db_path}")
                    return
                
                # Connect to the database
                conn = duckdb.connect(self.db_path)
                
                # Check if templates table exists
                table_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='templates'").fetchall()
                if not table_check:
                    logger.error("No 'templates' table found in database")
                    return
                
                # Get all templates
                templates = conn.execute("SELECT id, model_type, template_type, platform, template FROM templates").fetchall()
                if not templates:
                    logger.error("No templates found in database")
                    return
                
                # Convert to dictionary
                for template_id, model_type, template_type, platform, content in templates:
                    template_key = f"{model_type}_{template_type}"
                    if platform:
                        template_key += f"_{platform}"
                    
                    self.templates[template_key] = {
                        'id': template_id,
                        'model_type': model_type,
                        'template_type': template_type,
                        'platform': platform,
                        'template': content
                    }
                
                conn.close()
                logger.info(f"Loaded {len(self.templates)} templates from DuckDB database")
            except Exception as e:
                logger.error(f"Error loading templates from DuckDB database: {str(e)}")
    
    def get_model_family(self, model_name: str) -> str:
        """
        Determine the model family for a given model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model family name
        """
        # Check direct mapping
        model_prefix = model_name.split('/')[0] if '/' in model_name else model_name
        model_prefix = model_prefix.split('-')[0] if '-' in model_prefix else model_prefix
        
        if model_prefix in MODEL_TO_FAMILY:
            return MODEL_TO_FAMILY[model_prefix]
        
        # Try pattern matching
        for family, models in MODEL_FAMILIES.items():
            for model in models:
                if model in model_name.lower():
                    return family
        
        # Default to text_embedding if unknown
        return "text_embedding"
    
    def generate_test_file(self, model_name: str, output_file: Optional[str] = None, model_type: Optional[str] = None) -> str:
        """
        Generate a test file for a specific model.
        
        Args:
            model_name: Name of the model
            output_file: Path to output file (optional)
            model_type: Model type/family (optional)
            
        Returns:
            Generated test file content
        """
        if not model_type:
            model_type = self.get_model_family(model_name)
        
        logger.info(f"Generating test file for model {model_name} of type {model_type}")
        
        # Get model class name from model name
        model_class_name = model_name.split('/')[-1] if '/' in model_name else model_name
        model_class_name = ''.join(part.capitalize() for part in re.sub(r'[^a-zA-Z0-9]', ' ', model_class_name).split())
        
        # Get appropriate templates for this model type
        model_input_code = MODEL_INPUT_TEMPLATES.get(model_type, MODEL_INPUT_TEMPLATES["text_embedding"])
        output_check_code = OUTPUT_CHECK_TEMPLATES.get(model_type, OUTPUT_CHECK_TEMPLATES["text_embedding"])
        custom_model_loading = CUSTOM_MODEL_LOADING_TEMPLATES.get(model_type, CUSTOM_MODEL_LOADING_TEMPLATES["text_embedding"])
        model_specific_code = MODEL_SPECIFIC_CODE_TEMPLATES.get(model_type, MODEL_SPECIFIC_CODE_TEMPLATES["text_embedding"])
        
        # Create test file content
        content = STANDARD_TEMPLATE
        content = content.replace("{{model_name}}", model_name)
        content = content.replace("{{model_class_name}}", model_class_name)
        content = content.replace("{{model_type}}", model_type)
        content = content.replace("{{generation_date}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        content = content.replace("{{model_input_code}}", model_input_code)
        content = content.replace("{{output_check_code}}", output_check_code)
        content = content.replace("{{custom_model_loading}}", custom_model_loading)
        content = content.replace("{{model_specific_code}}", model_specific_code)
        
        # Validate the generated template content
        should_validate = HAS_VALIDATOR and (getattr(self.args, "validate", True) and not getattr(self.args, "skip_validation", False))
        
        if should_validate:
            logger.info(f"Validating template for {model_name}...")
            is_valid, validation_errors = validate_template_for_generator(
                content, 
                "merged_test_generator",
                validate_hardware=True,
                check_resource_pool=True,
                strict_indentation=False  # Be lenient with template indentation
            )
            
            if not is_valid:
                logger.warning(f"Generated template has validation errors:")
                for error in validation_errors:
                    logger.warning(f"  - {error}")
                
                if getattr(self.args, "strict_validation", False):
                    raise ValueError(f"Template validation failed for {model_name}")
                else:
                    logger.warning("Continuing despite validation errors (use --strict-validation to fail on errors)")
            else:
                logger.info(f"Template validation passed for {model_name}")
        elif getattr(self.args, "validate", False) and not HAS_VALIDATOR:
            logger.warning("Template validation requested but validator not available. Skipping validation.")
        
        # Write to file if requested
        if output_file:
            output_path = Path(output_file)
            os.makedirs(output_path.parent, exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Generated test file saved to {output_file}")
            
            # Make file executable
            os.chmod(output_file, 0o755)
        
        return content
    
    def generate_family_tests(self, family: str, output_dir: str):
        """
        Generate test files for all models in a family.
        
        Args:
            family: Model family name
            output_dir: Directory to save test files
        """
        if family not in MODEL_FAMILIES:
            logger.error(f"Unknown model family: {family}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_prefix in MODEL_FAMILIES[family]:
            # Use a standard model for each prefix
            if model_prefix == "bert":
                model_name = "bert-base-uncased"
            elif model_prefix == "sentence-transformers":
                model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
            elif model_prefix == "distilbert":
                model_name = "distilbert-base-uncased"
            elif model_prefix == "roberta":
                model_name = "roberta-base"
            elif model_prefix == "gpt2":
                model_name = "gpt2"
            elif model_prefix == "llama":
                model_name = "meta-llama/Llama-2-7b-hf"
            elif model_prefix == "t5":
                model_name = "t5-small"
            elif model_prefix == "vit":
                model_name = "google/vit-base-patch16-224"
            elif model_prefix == "whisper":
                model_name = "openai/whisper-tiny"
            elif model_prefix == "wav2vec2":
                model_name = "facebook/wav2vec2-base-960h"
            elif model_prefix == "clip":
                model_name = "openai/clip-vit-base-patch32"
            else:
                model_name = f"{model_prefix}-base"
            
            output_file = os.path.join(output_dir, f"test_{model_prefix}.py")
            self.generate_test_file(model_name, output_file, family)
    
    def list_models(self):
        """
        List all model types/families.
        """
        print("Available model families:")
        for family, models in MODEL_FAMILIES.items():
            print(f"- {family} ({len(models)} models)")
            for model in models[:3]:  # Show first 3 models
                print(f"  - {model}")
            if len(models) > 3:
                print(f"  - ... ({len(models) - 3} more)")
    
    def list_families(self):
        """
        List all model families.
        """
        print("Available model families:")
        for family in MODEL_FAMILIES:
            print(f"- {family}")

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Template-Based Test Generator")
    parser.add_argument("--model", type=str, help="Generate test file for specific model")
    parser.add_argument("--family", type=str, help="Generate test files for specific model family")
    parser.add_argument("--output", type=str, help="Output file or directory (depends on mode)")
    parser.add_argument("--db-path", type=str, default="../generators/templates/template_db.json", 
                      help="Path to the template database")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-families", action="store_true", help="List available model families")
    parser.add_argument("--list-valid-templates", action="store_true", help="List templates with valid syntax")
    parser.add_argument("--use-valid-only", action="store_true", help="Only use templates with valid syntax")
    # Validation options
    parser.add_argument("--validate", action="store_true", 
                     help="Validate templates before generation (default if validator available)")
    parser.add_argument("--skip-validation", action="store_true",
                     help="Skip template validation even if validator is available")
    parser.add_argument("--strict-validation", action="store_true",
                     help="Fail on validation errors")
    
    args = parser.parse_args()
    
    # Create generator
    generator = TemplateBasedTestGenerator(args.db_path, args)
    
    if args.list_models:
        generator.list_models()
    elif args.list_families:
        generator.list_families()
    elif args.list_valid_templates:
        # List templates with valid syntax
        print("Templates with valid syntax:")
        valid_count = 0
        for template_id, template_data in generator.templates.items():
            try:
                content = template_data.get('template', '')
                ast.parse(content)
                model_type = template_data.get('model_type', 'unknown')
                template_type = template_data.get('template_type', 'unknown')
                platform = template_data.get('platform', 'generic')
                key = f"{model_type}/{template_type}"
                if platform and platform != 'generic':
                    key += f"/{platform}"
                print(f"- {template_id}: {key}")
                valid_count += 1
            except SyntaxError:
                continue
        print(f"\nFound {valid_count}/{len(generator.templates)} templates with valid syntax ({valid_count/len(generator.templates)*100:.1f}%)")
    elif args.model:
        # Generate test file for specific model
        output_file = args.output if args.output else f"test_{args.model.split('/')[-1]}.py"
        content = generator.generate_test_file(args.model, output_file)
        if not args.output:
            print(content)
    elif args.family:
        # Generate test files for family
        output_dir = args.output if args.output else f"tests_{args.family}"
        generator.generate_family_tests(args.family, output_dir)
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())