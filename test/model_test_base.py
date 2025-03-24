#!/usr/bin/env python3
"""
ModelTest base class for standardized HuggingFace model testing.

This module defines the base class for all model tests, providing common functionality
and requiring implementation of architecture-specific methods.
"""

import os
import sys
import time
import json
import logging
import datetime
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if we should mock specific dependencies
MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'
MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'
MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'

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
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Try to import tokenizers
try:
    if MOCK_TOKENIZERS:
        raise ImportError("Mocked tokenizers import failure")
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

# Try to import sentencepiece
try:
    if MOCK_SENTENCEPIECE:
        raise ImportError("Mocked sentencepiece import failure")
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")

# Import numpy (usually available)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = MagicMock()
    HAS_NUMPY = False
    logger.warning("numpy not available, using mock")

# Create mock implementations if needed
if not HAS_TOKENIZERS:
    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            self.vocab_size = 32000
            self.pad_token = "[PAD]"
            self.eos_token = "</s>"
            self.bos_token = "<s>"
            self.mask_token = "[MASK]"
            
        def encode(self, text, **kwargs):
            return {"ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]}
            
        def decode(self, ids, **kwargs):
            return "Decoded text from mock"
            
        @staticmethod
        def from_file(vocab_filename):
            return MockTokenizer()
    
    tokenizers.Tokenizer = MockTokenizer

class ModelTest:
    """
    Base class for all model tests.
    
    This class defines the interface and common functionality for all model tests.
    Subclasses must implement architecture-specific methods.
    """
    
    def __init__(self, model_id=None, device=None):
        """
        Initialize the model test.
        
        Args:
            model_id: ID of the model to test, or None to use the default
            device: Device to run the test on, or None to auto-detect
        """
        # Set model type from class name if not explicitly set
        if not hasattr(self, 'model_type'):
            # Extract model type from class name (e.g., TestBertModel -> bert)
            class_name = self.__class__.__name__
            if class_name.startswith('Test') and class_name.endswith('Model'):
                self.model_type = class_name[4:-5].lower()
            else:
                self.model_type = class_name.lower()
        
        # Set model ID
        self.model_id = model_id or self.get_default_model_id()
        
        # Set device
        self.device = device or self.detect_preferred_device()
        
        # Initialize results storage
        self.results = {}
        self.examples = []
        self.performance_stats = {}
        
        # Set task based on architecture type if not set
        if not hasattr(self, 'task'):
            self.task = self.get_default_task()
        
        # Log initialization
        logger.info(f"Initialized {self.__class__.__name__} for {self.model_id} on {self.device}")
    
    def get_default_model_id(self) -> str:
        """
        Get the default model ID for this model type.
        
        Returns:
            Default model ID as a string
        """
        # This should be overridden by subclasses
        return f"{self.model_type}-base"
    
    def get_default_task(self) -> str:
        """
        Get the default task for this model type.
        
        Returns:
            Default task as a string
        """
        # This should be overridden by subclasses
        return "text-classification"
    
    def detect_preferred_device(self) -> str:
        """
        Detect the best available device for inference.
        
        Returns:
            Device string (cuda, mps, cpu, etc.)
        """
        # Check for CUDA
        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
            
        # Check for MPS (Apple Silicon)
        if HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            return "mps"
            
        # Default to CPU
        return "cpu"
    
    def get_hardware_capabilities(self) -> Dict[str, Any]:
        """
        Get detailed hardware capabilities.
        
        Returns:
            Dict with hardware capabilities
        """
        capabilities = {
            "cpu": True,
            "cuda": False,
            "cuda_version": None,
            "cuda_devices": 0,
            "mps": False,
            "openvino": False,
            "webnn": False,
            "webgpu": False
        }
        
        # Check CUDA
        if HAS_TORCH:
            capabilities["cuda"] = torch.cuda.is_available()
            if capabilities["cuda"]:
                capabilities["cuda_devices"] = torch.cuda.device_count()
                capabilities["cuda_version"] = torch.version.cuda
        
        # Check MPS (Apple Silicon)
        if HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
            capabilities["mps"] = torch.mps.is_available()
        
        # Check OpenVINO
        try:
            import openvino
            capabilities["openvino"] = True
        except ImportError:
            pass
        
        # Check WebNN/WebGPU (placeholder)
        # These would typically be checked in browser environments
        
        return capabilities
    
    def load_model(self, model_id=None) -> Any:
        """
        Load a model with the given ID.
        
        Args:
            model_id: ID of the model to load, or None to use the default
            
        Returns:
            Loaded model
        """
        # This should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement load_model()")
    
    def test_model_loading(self) -> Dict[str, Any]:
        """
        Test basic model loading functionality.
        
        Returns:
            Dict with test results
        """
        # This should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement test_model_loading()")
    
    def verify_model_output(self, model, input_data, expected_output=None) -> Dict[str, Any]:
        """
        Verify model outputs against expected values or sanity checks.
        
        Args:
            model: Model to test
            input_data: Input data for the model
            expected_output: Expected output (optional)
            
        Returns:
            Dict with verification results
        """
        # This should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement verify_model_output()")
    
    def run_tests(self, all_hardware=False) -> Dict[str, Any]:
        """
        Run all tests for this model.
        
        Args:
            all_hardware: If True, test on all available hardware
            
        Returns:
            Dict with all test results
        """
        # Test model loading
        self.results["model_loading"] = self.test_model_loading()
        
        # Add metadata
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
        self.results["metadata"] = {
            "model": self.model_id,
            "model_type": self.model_type,
            "task": self.task,
            "device": self.device,
            "timestamp": datetime.datetime.now().isoformat(),
            "dependencies": {
                "transformers": getattr(transformers, "__version__", None) if HAS_TRANSFORMERS else None,
                "torch": getattr(torch, "__version__", None) if HAS_TORCH else None,
                "numpy": getattr(np, "__version__", None) if HAS_NUMPY else None
            },
            "hardware_capabilities": self.get_hardware_capabilities(),
            "has_transformers": HAS_TRANSFORMERS,
            "has_torch": HAS_TORCH,
            "has_tokenizers": HAS_TOKENIZERS,
            "has_sentencepiece": HAS_SENTENCEPIECE,
            "using_real_inference": using_real_inference,
            "using_mocks": using_mocks,
            "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
        }
        
        return self.results
    
    def save_results(self, output_dir="collected_results") -> Optional[str]:
        """
        Save test results to a JSON file.
        
        Args:
            output_dir: Directory to save results in
            
        Returns:
            Path to saved file, or None if save failed
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate a filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id_safe = self.model_id.replace("/", "__")
            filename = f"model_test_{model_id_safe}_{timestamp}.json"
            file_path = os.path.join(output_dir, filename)
            
            # Save results to file
            with open(file_path, "w") as f:
                json.dump(self.results, f, indent=2)
                
            logger.info(f"Results saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None


class EncoderOnlyModelTest(ModelTest):
    """
    Base class for encoder-only models like BERT, RoBERTa, etc.
    """
    
    def __init__(self, model_id=None, device=None):
        """Initialize with encoder-only defaults."""
        # Set task and architecture type
        self.task = "fill-mask"
        self.architecture_type = "encoder-only"
        super().__init__(model_id, device)
    
    def get_default_model_id(self) -> str:
        """Get default model ID."""
        return f"{self.model_type}-base-uncased"
    
    def get_default_task(self) -> str:
        """Get default task."""
        return "fill-mask"
    
    def load_model(self, model_id=None) -> Any:
        """Load an encoder-only model."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required to load models")
            
        model_id = model_id or self.model_id
        
        # Load tokenizer first
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        
        # Load model with proper class
        model = transformers.AutoModelForMaskedLM.from_pretrained(model_id)
        
        # Move to device if needed
        if self.device != "cpu" and HAS_TORCH:
            model = model.to(self.device)
            
        return {"model": model, "tokenizer": tokenizer}
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test encoder model loading."""
        try:
            # Record start time
            start_time = time.time()
            
            # Load model
            loaded = self.load_model()
            
            # Record loading time
            load_time = time.time() - start_time
            
            # Add to performance stats
            self.performance_stats["model_loading"] = {
                "load_time": load_time
            }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "load_time": load_time
            }
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {
                "success": False,
                "model_id": self.model_id,
                "device": self.device,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def verify_model_output(self, model_data, input_text="The [MASK] runs quickly.", expected_output=None) -> Dict[str, Any]:
        """Verify encoder model outputs."""
        try:
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            # Ensure we have a mask token in the input
            if not "[MASK]" in input_text and hasattr(tokenizer, "mask_token"):
                input_text = input_text.replace("[MASK]", tokenizer.mask_token)
                
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt")
            
            # Move to device if needed
            if self.device != "cpu" and HAS_TORCH:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            # Run model
            start_time = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            inference_time = time.time() - start_time
            
            # Get prediction at masked position
            mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
            
            if len(mask_token_index) > 0:
                mask_token_index = mask_token_index.item()
                logits = outputs.logits
                mask_token_logits = logits[0, mask_token_index, :]
                
                # Get top predictions
                top_k = torch.topk(mask_token_logits, 5, dim=0)
                top_tokens = []
                
                for i, token_id in enumerate(top_k.indices):
                    token = tokenizer.decode([token_id])
                    score = top_k.values[i].item()
                    top_tokens.append({"token": token, "score": score})
                    
                # Add to performance stats
                self.performance_stats["model_inference"] = {
                    "inference_time": inference_time
                }
                
                # Add to examples
                self.examples.append({
                    "input": input_text,
                    "top_predictions": top_tokens
                })
                
                return {
                    "success": True,
                    "input": input_text,
                    "predictions": top_tokens,
                    "inference_time": inference_time
                }
            else:
                return {
                    "success": False,
                    "input": input_text,
                    "error": "No mask token found in input"
                }
                
        except Exception as e:
            logger.error(f"Error verifying model output: {e}")
            return {
                "success": False,
                "input": input_text,
                "error": str(e),
                "traceback": traceback.format_exc()
            }


class DecoderOnlyModelTest(ModelTest):
    """
    Base class for decoder-only models like GPT-2, LLaMA, etc.
    """
    
    def __init__(self, model_id=None, device=None):
        """Initialize with decoder-only defaults."""
        # Set task and architecture type
        self.task = "text-generation"
        self.architecture_type = "decoder-only"
        super().__init__(model_id, device)
    
    def get_default_model_id(self) -> str:
        """Get default model ID."""
        return self.model_type
    
    def get_default_task(self) -> str:
        """Get default task."""
        return "text-generation"
    
    def load_model(self, model_id=None) -> Any:
        """Load a decoder-only model."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required to load models")
            
        model_id = model_id or self.model_id
        
        # Load tokenizer first
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        
        # Fix padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model with proper class
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
        
        # Move to device if needed
        if self.device != "cpu" and HAS_TORCH:
            model = model.to(self.device)
            
        return {"model": model, "tokenizer": tokenizer}
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test decoder model loading."""
        try:
            # Record start time
            start_time = time.time()
            
            # Load model
            loaded = self.load_model()
            
            # Record loading time
            load_time = time.time() - start_time
            
            # Add to performance stats
            self.performance_stats["model_loading"] = {
                "load_time": load_time
            }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "load_time": load_time
            }
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {
                "success": False,
                "model_id": self.model_id,
                "device": self.device,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def verify_model_output(self, model_data, input_text="Once upon a time", expected_output=None) -> Dict[str, Any]:
        """Verify decoder model outputs."""
        try:
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
                
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt")
            
            # Move to device if needed
            if self.device != "cpu" and HAS_TORCH:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            # Run model to generate text
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9
                )
            inference_time = time.time() - start_time
            
            # Decode generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            # Add to performance stats
            self.performance_stats["model_inference"] = {
                "inference_time": inference_time
            }
            
            # Add to examples
            self.examples.append({
                "input": input_text,
                "generated_text": generated_text
            })
            
            return {
                "success": True,
                "input": input_text,
                "generated_text": generated_text,
                "inference_time": inference_time
            }
                
        except Exception as e:
            logger.error(f"Error verifying model output: {e}")
            return {
                "success": False,
                "input": input_text,
                "error": str(e),
                "traceback": traceback.format_exc()
            }


class EncoderDecoderModelTest(ModelTest):
    """
    Base class for encoder-decoder models like T5, BART, etc.
    """
    
    def __init__(self, model_id=None, device=None):
        """Initialize with encoder-decoder defaults."""
        # Set task and architecture type
        self.task = "text2text-generation"
        self.architecture_type = "encoder-decoder"
        super().__init__(model_id, device)
    
    def get_default_model_id(self) -> str:
        """Get default model ID."""
        return f"{self.model_type}-base"
    
    def get_default_task(self) -> str:
        """Get default task."""
        return "text2text-generation"
    
    def load_model(self, model_id=None) -> Any:
        """Load an encoder-decoder model."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required to load models")
            
        model_id = model_id or self.model_id
        
        # Load tokenizer first
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        
        # Load model with proper class
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        # Move to device if needed
        if self.device != "cpu" and HAS_TORCH:
            model = model.to(self.device)
            
        return {"model": model, "tokenizer": tokenizer}
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test encoder-decoder model loading."""
        try:
            # Record start time
            start_time = time.time()
            
            # Load model
            loaded = self.load_model()
            
            # Record loading time
            load_time = time.time() - start_time
            
            # Add to performance stats
            self.performance_stats["model_loading"] = {
                "load_time": load_time
            }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "load_time": load_time
            }
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {
                "success": False,
                "model_id": self.model_id,
                "device": self.device,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def verify_model_output(self, model_data, input_text="translate English to French: Hello, how are you?", expected_output=None) -> Dict[str, Any]:
        """Verify encoder-decoder model outputs."""
        try:
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
                
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt")
            
            # Move to device if needed
            if self.device != "cpu" and HAS_TORCH:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            # Run model to generate text
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs)
            inference_time = time.time() - start_time
            
            # Decode generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            # Add to performance stats
            self.performance_stats["model_inference"] = {
                "inference_time": inference_time
            }
            
            # Add to examples
            self.examples.append({
                "input": input_text,
                "generated_text": generated_text
            })
            
            return {
                "success": True,
                "input": input_text,
                "generated_text": generated_text,
                "inference_time": inference_time
            }
                
        except Exception as e:
            logger.error(f"Error verifying model output: {e}")
            return {
                "success": False,
                "input": input_text,
                "error": str(e),
                "traceback": traceback.format_exc()
            }


class VisionModelTest(ModelTest):
    """
    Base class for vision models like ViT, Swin, etc.
    """
    
    def __init__(self, model_id=None, device=None):
        """Initialize with vision model defaults."""
        # Set task and architecture type
        self.task = "image-classification"
        self.architecture_type = "vision"
        super().__init__(model_id, device)
    
    def get_default_model_id(self) -> str:
        """Get default model ID."""
        return f"google/{self.model_type}-base-patch16-224"
    
    def get_default_task(self) -> str:
        """Get default task."""
        return "image-classification"
    
    def load_model(self, model_id=None) -> Any:
        """Load a vision model."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required to load models")
            
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL is required to load vision models")
            
        model_id = model_id or self.model_id
        
        # Load processor first
        processor = transformers.AutoImageProcessor.from_pretrained(model_id)
        
        # Load model with proper class
        model = transformers.AutoModelForImageClassification.from_pretrained(model_id)
        
        # Move to device if needed
        if self.device != "cpu" and HAS_TORCH:
            model = model.to(self.device)
            
        return {"model": model, "processor": processor}
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test vision model loading."""
        try:
            # Record start time
            start_time = time.time()
            
            # Load model
            loaded = self.load_model()
            
            # Record loading time
            load_time = time.time() - start_time
            
            # Add to performance stats
            self.performance_stats["model_loading"] = {
                "load_time": load_time
            }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "load_time": load_time
            }
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {
                "success": False,
                "model_id": self.model_id,
                "device": self.device,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def verify_model_output(self, model_data, input_image=None, expected_output=None) -> Dict[str, Any]:
        """Verify vision model outputs."""
        try:
            from PIL import Image
            
            model = model_data["model"]
            processor = model_data["processor"]
            
            # Create a dummy image if none provided
            if input_image is None:
                input_image = Image.new('RGB', (224, 224), color='red')
                
            # Process image
            inputs = processor(images=input_image, return_tensors="pt")
            
            # Move to device if needed
            if self.device != "cpu" and HAS_TORCH:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            # Run model
            start_time = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            inference_time = time.time() - start_time
            
            # Get class prediction
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                
                # If we have id2label mapping, get the class name
                if hasattr(model.config, "id2label"):
                    predicted_class = model.config.id2label[predicted_class_idx]
                else:
                    predicted_class = f"Class {predicted_class_idx}"
                    
                # Add to performance stats
                self.performance_stats["model_inference"] = {
                    "inference_time": inference_time
                }
                
                # Add to examples
                self.examples.append({
                    "input_type": "image",
                    "top_prediction": predicted_class
                })
                
                return {
                    "success": True,
                    "predicted_class": predicted_class,
                    "predicted_class_id": predicted_class_idx,
                    "inference_time": inference_time
                }
            else:
                return {
                    "success": False,
                    "error": "No logits found in model output"
                }
                
        except Exception as e:
            logger.error(f"Error verifying model output: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }


# Additional base class implementations can be added for:
# - SpeechModelTest
# - VisionTextModelTest
# - MultimodalModelTest


def get_model_test_class(model_type: str) -> type:
    """
    Get the appropriate ModelTest subclass for a model type.
    
    Args:
        model_type: Type of model to get a test class for
        
    Returns:
        ModelTest subclass appropriate for the model type
    """
    # Architecture detection mapping
    architecture_mapping = {
        # Encoder-only models
        "bert": EncoderOnlyModelTest,
        "roberta": EncoderOnlyModelTest,
        "distilbert": EncoderOnlyModelTest,
        "albert": EncoderOnlyModelTest,
        "electra": EncoderOnlyModelTest,
        
        # Decoder-only models
        "gpt2": DecoderOnlyModelTest,
        "llama": DecoderOnlyModelTest,
        "mistral": DecoderOnlyModelTest,
        "gpt_neo": DecoderOnlyModelTest,
        "bloom": DecoderOnlyModelTest,
        "falcon": DecoderOnlyModelTest,
        "gemma": DecoderOnlyModelTest,
        
        # Encoder-decoder models
        "t5": EncoderDecoderModelTest,
        "bart": EncoderDecoderModelTest,
        "pegasus": EncoderDecoderModelTest,
        "mbart": EncoderDecoderModelTest,
        
        # Vision models
        "vit": VisionModelTest,
        "swin": VisionModelTest,
        "deit": VisionModelTest,
        "beit": VisionModelTest,
        "convnext": VisionModelTest
    }
    
    # Convert to lowercase for case-insensitive matching
    model_type_lower = model_type.lower()
    
    # Replace hyphens with underscores
    model_type_lower = model_type_lower.replace("-", "_")
    
    # Return the appropriate class or default to EncoderOnlyModelTest
    return architecture_mapping.get(model_type_lower, EncoderOnlyModelTest)


def create_model_test(model_type: str, model_id: str = None, device: str = None) -> ModelTest:
    """
    Factory function to create an appropriate ModelTest instance.
    
    Args:
        model_type: Type of model to create a test for
        model_id: Optional specific model ID
        device: Optional device to run on
        
    Returns:
        ModelTest instance appropriate for the model type
    """
    test_class = get_model_test_class(model_type)
    
    # Create test instance with the right model type
    test_instance = test_class(model_id, device)
    
    # Set model type if not already set
    if not hasattr(test_instance, 'model_type') or test_instance.model_type != model_type:
        test_instance.model_type = model_type
    
    return test_instance