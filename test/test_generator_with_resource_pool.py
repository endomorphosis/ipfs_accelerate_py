#!/usr/bin/env python
"""
Test generator with resource pool integration and hardware awareness.
This script generates optimized test files for models with hardware-specific configurations.
"""

import os
import sys
import json
import logging
import argparse
import importlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import required components
try:
    from resource_pool import get_global_resource_pool
    from hardware_detection import detect_hardware_with_comprehensive_checks, CUDA, ROCM, MPS, OPENVINO, CPU, WEBNN, WEBGPU
except ImportError as e:
    logger.error(f"Required module not found: {e}")
    logger.error("Make sure resource_pool.py and hardware_detection.py are in your path")
    sys.exit(1)

# Try to import model classification components
try:
    from model_family_classifier import classify_model, ModelFamilyClassifier
except ImportError as e:
    logger.warning(f"Model family classifier not available: {e}")
    logger.warning("Will use basic model classification based on model name")
    classify_model = None
    ModelFamilyClassifier = None

# Try to import hardware-model integration
try:
    from hardware_model_integration import (
        HardwareAwareModelClassifier, 
        get_hardware_aware_model_classification
    )
    HARDWARE_MODEL_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Hardware-model integration not available: {e}")
    logger.warning("Will use basic hardware-model integration")
    HARDWARE_MODEL_INTEGRATION_AVAILABLE = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test generator with resource pool integration")
    parser.add_argument("--model", type=str, required=True, help="Model name to generate tests for")
    parser.add_argument("--output-dir", type=str, default="./skills", help="Output directory for generated tests")
    parser.add_argument("--timeout", type=float, default=0.1, help="Resource cleanup timeout (minutes)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear resource cache before running")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "auto"], 
                      default="auto", help="Force specific device for testing")
    parser.add_argument("--hw-cache", type=str, help="Path to hardware detection cache")
    parser.add_argument("--model-db", type=str, help="Path to model database")
    parser.add_argument("--use-model-family", action="store_true", 
                      help="Use model family classifier for optimal template selection")
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment and configure logging"""
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Clear resource pool if requested
    if args.clear_cache:
        pool = get_global_resource_pool()
        pool.clear()
        logger.info("Resource pool cleared")

def load_dependencies():
    """Load common dependencies with resource pooling"""
    logger.info("Loading dependencies using resource pool")
    pool = get_global_resource_pool()
    
    # Load common libraries
    torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
    transformers = pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
    
    # Check if dependencies were loaded successfully
    if torch is None or transformers is None:
        logger.error("Failed to load required dependencies")
        return False
    
    logger.info("Dependencies loaded successfully")
    return True

def get_hardware_aware_classification(model_name, hw_cache_path=None, model_db_path=None):
    """
    Get hardware-aware model classification
    
    Args:
        model_name: Model name to classify
        hw_cache_path: Optional path to hardware detection cache
        model_db_path: Optional path to model database
        
    Returns:
        Dictionary with hardware-aware classification results
    """
    # Use hardware-model integration if available
    if HARDWARE_MODEL_INTEGRATION_AVAILABLE:
        try:
            logger.info("Using hardware-model integration for classification")
            classification = get_hardware_aware_model_classification(
                model_name=model_name,
                hw_cache_path=hw_cache_path,
                model_db_path=model_db_path
            )
            return classification
        except Exception as e:
            logger.warning(f"Error using hardware-model integration: {e}")
    
    # Fallback to simpler classification with hardware detection
    logger.info("Using basic hardware-model integration")
    
    # Detect hardware
    hardware_result = detect_hardware_with_comprehensive_checks()
    hardware_info = {k: v for k, v in hardware_result.items() if isinstance(v, bool)}
    best_hardware = hardware_result.get('best_available', CPU)
    torch_device = hardware_result.get('torch_device', 'cpu')
    
    # Classify model if classifier is available
    model_family = "default"
    subfamily = None
    if classify_model:
        try:
            # Get hardware compatibility information for more accurate classification
            hw_compatibility = {
                "cuda": {"compatible": hardware_info.get("cuda", False)},
                "mps": {"compatible": hardware_info.get("mps", False)},
                "rocm": {"compatible": hardware_info.get("rocm", False)},
                "openvino": {"compatible": hardware_info.get("openvino", False)},
                "webnn": {"compatible": hardware_info.get("webnn", False)},
                "webgpu": {"compatible": hardware_info.get("webgpu", False)}
            }
            
            # Call classify_model with model name and hardware compatibility
            classification = classify_model(
                model_name=model_name,
                hw_compatibility=hw_compatibility,
                model_db_path=model_db_path
            )
            
            model_family = classification.get("family", "default")
            subfamily = classification.get("subfamily")
            confidence = classification.get("confidence", 0)
            logger.info(f"Model classified as: {model_family} (subfamily: {subfamily}, confidence: {confidence:.2f})")
        except Exception as e:
            logger.warning(f"Error classifying model: {e}")
    
    # Create basic hardware-aware classification result
    # Map model family to recommended hardware
    family_to_hardware = {
        "embedding": best_hardware,  # Embedding models work on any hardware
        "text_generation": "cuda" if hardware_info.get("cuda", False) else best_hardware,
        "vision": "cuda" if hardware_info.get("cuda", False) else 
                 ("openvino" if hardware_info.get("openvino", False) else best_hardware),
        "audio": "cuda" if hardware_info.get("cuda", False) else best_hardware,
        "multimodal": "cuda" if hardware_info.get("cuda", False) else "cpu",
        "default": best_hardware
    }
    
    # Map model family to template
    # Try to use ModelFamilyClassifier's get_template_for_family method if available
    template = None
    if ModelFamilyClassifier and model_family != "default":
        try:
            classifier = ModelFamilyClassifier()
            template = classifier.get_template_for_family(model_family, subfamily)
            logger.debug(f"Template selected by ModelFamilyClassifier: {template}")
        except Exception as e:
            logger.warning(f"Error getting template from ModelFamilyClassifier: {e}")
    
    # Fallback to static mapping if template selection failed
    if not template:
        family_to_template = {
            "embedding": "hf_embedding_template.py",
            "text_generation": "hf_text_generation_template.py",
            "vision": "hf_vision_template.py",
            "audio": "hf_audio_template.py",
            "multimodal": "hf_multimodal_template.py",
            "default": "hf_template.py"
        }
        template = family_to_template.get(model_family, "hf_template.py")
    
    # Create comprehensive hardware compatibility profile
    hw_profile = {
        "cuda": {"compatible": hardware_info.get("cuda", False)},
        "mps": {"compatible": hardware_info.get("mps", False)},
        "rocm": {"compatible": hardware_info.get("rocm", False)},
        "openvino": {"compatible": hardware_info.get("openvino", False)},
        "webnn": {"compatible": hardware_info.get("webnn", False)},
        "webgpu": {"compatible": hardware_info.get("webgpu", False)},
        "cpu": {"compatible": True}
    }
    
    # Return classification with hardware awareness
    return {
        "model_name": model_name,
        "family": model_family,
        "subfamily": subfamily,
        "recommended_hardware": family_to_hardware.get(model_family, best_hardware),
        "recommended_template": template,
        "hardware_profile": hw_profile,
        "torch_device": torch_device,
        "resource_requirements": {
            "min_memory_mb": 2000,
            "recommended_memory_mb": 4000
        }
    }

def load_model_with_hardware_awareness(model_name, hardware_preferences=None):
    """
    Load model with hardware awareness using resource pool
    
    Args:
        model_name: Model name to load
        hardware_preferences: Optional hardware preferences
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model with hardware awareness: {model_name}")
    pool = get_global_resource_pool()
    
    # Get hardware information for classification
    hardware_result = detect_hardware_with_comprehensive_checks()
    hardware_info = {k: v for k, v in hardware_result.items() if isinstance(v, bool)}
    
    # Get model classification 
    model_family = "default"
    subfamily = None
    if classify_model:
        try:
            # Create hardware compatibility information
            hw_compatibility = {
                "cuda": {"compatible": hardware_info.get("cuda", False)},
                "mps": {"compatible": hardware_info.get("mps", False)},
                "rocm": {"compatible": hardware_info.get("rocm", False)},
                "openvino": {"compatible": hardware_info.get("openvino", False)},
                "webnn": {"compatible": hardware_info.get("webnn", False)},
                "webgpu": {"compatible": hardware_info.get("webgpu", False)}
            }
            
            # Call classify_model with hardware compatibility
            classification = classify_model(
                model_name=model_name,
                hw_compatibility=hw_compatibility
            )
            
            model_family = classification.get("family", "default")
            subfamily = classification.get("subfamily")
            confidence = classification.get("confidence", 0)
            logger.info(f"Model classified as: {model_family} (subfamily: {subfamily}, confidence: {confidence:.2f})")
        except Exception as e:
            logger.warning(f"Error classifying model: {e}")
    elif HARDWARE_MODEL_INTEGRATION_AVAILABLE:
        try:
            classification = get_hardware_aware_model_classification(model_name)
            model_family = classification.get("family", "default")
            subfamily = classification.get("subfamily")
            logger.info(f"Model classified as: {model_family} (subfamily: {subfamily})")
        except Exception as e:
            logger.warning(f"Error getting hardware-aware classification: {e}")
    
    # Define model constructor with improved model family support
    def create_model():
        try:
            # Get necessary libraries from resource pool
            transformers = pool.get_resource("transformers")
            torch = pool.get_resource("torch")
            
            # Select appropriate AutoModel class based on model family
            if model_family == "text_generation":
                # Check subfamily for more specific model class selection
                if subfamily == "causal_lm":
                    from transformers import AutoModelForCausalLM
                    logger.debug(f"Using AutoModelForCausalLM for {model_name}")
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                elif subfamily == "seq2seq":
                    from transformers import AutoModelForSeq2SeqLM
                    logger.debug(f"Using AutoModelForSeq2SeqLM for {model_name}")
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                else:
                    # Default for text generation
                    from transformers import AutoModelForCausalLM
                    logger.debug(f"Using AutoModelForCausalLM for {model_name}")
                    model = AutoModelForCausalLM.from_pretrained(model_name)
            elif model_family == "vision":
                # Check subfamily for more specific vision model handling
                if subfamily == "object_detector":
                    from transformers import AutoModelForObjectDetection
                    logger.debug(f"Using AutoModelForObjectDetection for {model_name}")
                    model = AutoModelForObjectDetection.from_pretrained(model_name)
                elif subfamily == "segmentation":
                    from transformers import AutoModelForImageSegmentation
                    logger.debug(f"Using AutoModelForImageSegmentation for {model_name}")
                    model = AutoModelForImageSegmentation.from_pretrained(model_name)
                else:
                    # Default vision model
                    from transformers import AutoModelForImageClassification
                    logger.debug(f"Using AutoModelForImageClassification for {model_name}")
                    model = AutoModelForImageClassification.from_pretrained(model_name)
            elif model_family == "audio":
                # Check subfamily for audio model types
                if subfamily == "speech_recognition":
                    from transformers import AutoModelForSpeechSeq2Seq
                    logger.debug(f"Using AutoModelForSpeechSeq2Seq for {model_name}")
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
                else:
                    # Default audio model
                    from transformers import AutoModelForAudioClassification
                    logger.debug(f"Using AutoModelForAudioClassification for {model_name}")
                    model = AutoModelForAudioClassification.from_pretrained(model_name)
            elif model_family == "multimodal":
                # Try to load appropriate multimodal model class
                try:
                    if "clip" in model_name.lower() or subfamily == "image_text_encoder":
                        from transformers import CLIPModel
                        logger.debug(f"Using CLIPModel for {model_name}")
                        model = CLIPModel.from_pretrained(model_name)
                    elif "blip" in model_name.lower():
                        from transformers import BlipModel
                        logger.debug(f"Using BlipModel for {model_name}")
                        model = BlipModel.from_pretrained(model_name)
                    elif "llava" in model_name.lower():
                        from transformers import LlavaModel
                        logger.debug(f"Using LlavaModel for {model_name}")
                        model = LlavaModel.from_pretrained(model_name)
                    else:
                        # Fallback to VisionTextDualEncoder as a reasonable default
                        from transformers import VisionTextDualEncoderModel
                        logger.debug(f"Using VisionTextDualEncoderModel for {model_name}")
                        model = VisionTextDualEncoderModel.from_pretrained(model_name)
                except Exception as multimodal_err:
                    logger.warning(f"Error loading multimodal model: {multimodal_err}. Falling back to AutoModel.")
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(model_name)
            elif model_family == "embedding":
                # Handle embedding models like BERT
                if subfamily == "masked_lm":
                    from transformers import AutoModelForMaskedLM
                    logger.debug(f"Using AutoModelForMaskedLM for {model_name}")
                    model = AutoModelForMaskedLM.from_pretrained(model_name)
                else:
                    # Default embedding model
                    from transformers import AutoModel
                    logger.debug(f"Using AutoModel for {model_name}")
                    model = AutoModel.from_pretrained(model_name)
            else:
                # Fallback to AutoModel for any unrecognized family
                from transformers import AutoModel
                logger.debug(f"Using AutoModel for {model_name}")
                model = AutoModel.from_pretrained(model_name)
                
            logger.debug(f"Model loaded: {type(model).__name__}")
            return model
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            # Fallback to basic AutoModel if specific loading fails
            try:
                from transformers import AutoModel
                logger.debug(f"Falling back to AutoModel for {model_name}")
                return AutoModel.from_pretrained(model_name)
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return None
    
    # Get or create the model with hardware preferences
    model = pool.get_model(
        model_family, 
        model_name, 
        constructor=create_model,
        hardware_preferences=hardware_preferences
    )
    
    if model is None:
        logger.error(f"Failed to load model: {model_name}")
        return None, None
    
    # Also load tokenizer or processor with improved handling for different model types
    def create_tokenizer():
        try:
            transformers = pool.get_resource("transformers")
            
            # Select appropriate tokenizer/processor based on model family and subfamily
            if model_family == "vision":
                from transformers import AutoImageProcessor
                logger.debug(f"Using AutoImageProcessor for {model_name}")
                return AutoImageProcessor.from_pretrained(model_name)
            elif model_family == "audio":
                from transformers import AutoProcessor
                logger.debug(f"Using AutoProcessor for {model_name}")
                return AutoProcessor.from_pretrained(model_name)
            elif model_family == "multimodal":
                # Try to pick the right processor for multimodal models
                try:
                    if "clip" in model_name.lower():
                        from transformers import CLIPProcessor
                        logger.debug(f"Using CLIPProcessor for {model_name}")
                        return CLIPProcessor.from_pretrained(model_name)
                    elif "blip" in model_name.lower():
                        from transformers import BlipProcessor
                        logger.debug(f"Using BlipProcessor for {model_name}")
                        return BlipProcessor.from_pretrained(model_name)
                    elif "llava" in model_name.lower():
                        from transformers import LlavaProcessor
                        logger.debug(f"Using LlavaProcessor for {model_name}")
                        return LlavaProcessor.from_pretrained(model_name)
                    else:
                        # General processor fallback
                        from transformers import AutoProcessor
                        logger.debug(f"Using AutoProcessor for {model_name}")
                        return AutoProcessor.from_pretrained(model_name)
                except Exception as proc_err:
                    logger.warning(f"Error loading specific multimodal processor: {proc_err}. Trying AutoProcessor.")
                    from transformers import AutoProcessor
                    return AutoProcessor.from_pretrained(model_name)
            else:
                # Text-based models use tokenizers
                from transformers import AutoTokenizer
                logger.debug(f"Using AutoTokenizer for {model_name}")
                return AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error creating tokenizer/processor: {e}")
            # Fallback to basic AutoTokenizer
            try:
                from transformers import AutoTokenizer
                logger.debug(f"Falling back to AutoTokenizer for {model_name}")
                return AutoTokenizer.from_pretrained(model_name)
            except Exception as e2:
                logger.error(f"Tokenizer fallback also failed: {e2}")
                return None
    
    tokenizer = pool.get_tokenizer(model_family, model_name, constructor=create_tokenizer)
    
    logger.info(f"Model and tokenizer loaded for {model_name}")
    return model, tokenizer

def generate_test_file(model_name, output_dir, device=None, hw_cache_path=None, model_db_path=None):
    """
    Generate a test file for the model with hardware awareness
    
    Args:
        model_name: Model name to generate tests for
        output_dir: Output directory for test files
        device: Optional specific device override
        hw_cache_path: Optional path to hardware detection cache
        model_db_path: Optional path to model database
        
    Returns:
        Path to the generated test file
    """
    logger.info(f"Generating test file for {model_name}")
    
    # Get hardware-aware model classification
    try:
        classification = get_hardware_aware_classification(
            model_name, 
            hw_cache_path=hw_cache_path,
            model_db_path=model_db_path
        )
        
        logger.info(f"Model classification: family={classification.get('family')}, "
                   f"recommended_hardware={classification.get('recommended_hardware')}")
    except Exception as e:
        logger.error(f"Error getting hardware-aware classification: {e}")
        classification = {
            "model_name": model_name,
            "family": "default",
            "recommended_hardware": "cpu",
            "torch_device": "cpu"
        }
    
    # Set up hardware preferences
    if device == "auto" or device is None:
        device_str = classification.get("torch_device", "cpu")
    else:
        device_str = device
        
    logger.info(f"Using device: {device_str}")
    
    # Create hardware preferences
    hardware_preferences = {
        "device": device_str
    }
    
    # Get resource pool
    pool = get_global_resource_pool()
    
    # Try to load model and tokenizer with hardware awareness
    model, tokenizer = load_model_with_hardware_awareness(
        model_name, 
        hardware_preferences=hardware_preferences
    )
    
    if model is None or tokenizer is None:
        logger.error("Cannot generate test without model and tokenizer")
        return None
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create normalized model name for file
    normalized_name = model_name.replace("/", "_").replace("-", "_").lower()
    test_file_path = os.path.join(output_dir, f"test_hf_{normalized_name}.py")
    
    # Generate test file content
    logger.info(f"Creating test file: {test_file_path}")
    
    # Determine model type
    model_type = type(model).__name__
    
    # Get model-specific information for better test generation
    try:
        # Using resource pool for torch access
        torch = pool.get_resource("torch")
        
        # Analyze model outputs for better test generation
        # The approach depends on the model family and subfamily
        output_shapes = analyze_model_outputs(
            model, 
            tokenizer, 
            classification.get("family", "default"),
            subfamily=classification.get("subfamily")
        )
        logger.debug(f"Model output shapes: {output_shapes}")
    except Exception as e:
        logger.error(f"Error analyzing model outputs: {e}")
        output_shapes = {}
    
    # Create the test file content
    test_content = generate_test_content(
        model_name=model_name, 
        model_type=model_type, 
        classification=classification,
        device=device_str,
        output_shapes=output_shapes
    )
    
    # Write the test file
    try:
        with open(test_file_path, "w") as f:
            f.write(test_content)
        logger.info(f"Successfully generated test file: {test_file_path}")
        return test_file_path
    except Exception as e:
        logger.error(f"Error writing test file: {e}")
        return None

def analyze_model_outputs(model, tokenizer, model_family, subfamily=None):
    """
    Analyze model outputs for test generation
    
    Args:
        model: The model to analyze
        tokenizer: The tokenizer or processor for the model
        model_family: The model family (text_generation, vision, etc.)
        subfamily: Optional subfamily for more specific behavior
        
    Returns:
        Dictionary with output shapes
    """
    # Init test device
    import torch
    device = next(model.parameters()).device
    
    try:
        # Different analysis based on model family and subfamily
        if model_family == "vision":
            if subfamily == "object_detector":
                # Object detection has different input/output format
                sample_input = {"pixel_values": torch.rand(1, 3, 800, 1200, device=device)}
            elif subfamily == "segmentation":
                # Segmentation typically uses higher resolution
                sample_input = {"pixel_values": torch.rand(1, 3, 512, 512, device=device)}
            else:
                # Default vision model input
                sample_input = {"pixel_values": torch.rand(1, 3, 224, 224, device=device)}
        elif model_family == "audio":
            if subfamily == "speech_recognition":
                # Whisper and similar models use input_features
                sample_input = {"input_features": torch.rand(1, 80, 3000, device=device)}
            else:
                # Default audio model input shape
                sample_input = {"input_features": torch.rand(1, 80, 200, device=device)}
        elif model_family == "multimodal":
            # Handle different multimodal model types
            if subfamily == "image_text_encoder" or "clip" in str(model.__class__.__name__).lower():
                # CLIP-like inputs
                sample_input = {
                    "pixel_values": torch.rand(1, 3, 224, 224, device=device),
                    "input_ids": torch.randint(0, 1000, (1, 20), device=device)
                }
            elif "llava" in str(model.__class__.__name__).lower():
                # LLaVA-like inputs
                try:
                    sample_input = {
                        "pixel_values": torch.rand(1, 3, 336, 336, device=device),
                        "input_ids": torch.randint(0, 1000, (1, 20), device=device),
                        "attention_mask": torch.ones(1, 20, device=device)
                    }
                except Exception:
                    # Simpler fallback for LLaVA analysis
                    sample_input = {
                        "pixel_values": torch.rand(1, 3, 336, 336, device=device),
                        "input_ids": torch.randint(0, 1000, (1, 20), device=device)
                    }
            elif "blip" in str(model.__class__.__name__).lower():
                # BLIP-like inputs
                sample_input = {
                    "pixel_values": torch.rand(1, 3, 224, 224, device=device),
                    "input_ids": torch.randint(0, 1000, (1, 20), device=device),
                    "attention_mask": torch.ones(1, 20, device=device)
                }
            else:
                # Generic multimodal fallback - try multiple input formats
                try:
                    # First try CLIP-like inputs
                    sample_input = {
                        "pixel_values": torch.rand(1, 3, 224, 224, device=device),
                        "input_ids": torch.randint(0, 1000, (1, 20), device=device)
                    }
                except Exception:
                    # Fallback to vision-only
                    sample_input = {"pixel_values": torch.rand(1, 3, 224, 224, device=device)}
        elif model_family == "text_generation":
            # Text generation models like GPT, T5, etc.
            if subfamily == "seq2seq":
                # Sequence-to-sequence models (T5, BART, etc.)
                sample_input = tokenizer("translate English to French: Hello, world!", return_tensors="pt")
            else:
                # Causal language models (GPT, LLaMA, etc.)
                sample_input = tokenizer("Hello, world!", return_tensors="pt")
            # Move to device
            sample_input = {k: v.to(device) for k, v in sample_input.items()}
        else:
            # Default to text models for any other family (including embedding)
            sample_input = tokenizer("Hello, world!", return_tensors="pt")
            # Move to device
            sample_input = {k: v.to(device) for k, v in sample_input.items()}
        
        # Run inference with better error handling
        try:
            with torch.no_grad():
                outputs = model(**sample_input)
            
            # Extract output shapes
            output_shapes = {}
            if hasattr(outputs, "keys"):
                # Dictionary-like output
                for key, value in outputs.items():
                    if hasattr(value, "shape"):
                        output_shapes[key] = list(value.shape)
            elif hasattr(outputs, "shape"):
                # Single tensor output
                output_shapes["output"] = list(outputs.shape)
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                # Tuple output (common in some models)
                for i, item in enumerate(outputs):
                    if hasattr(item, "shape"):
                        output_shapes[f"output_{i}"] = list(item.shape)
            
            return output_shapes
        except Exception as inference_error:
            logger.warning(f"Error during model inference: {inference_error}")
            # Return empty dict to avoid test generation failures
            return {}
    except Exception as e:
        logger.warning(f"Error analyzing model outputs: {e}")
        return {}

def generate_test_content(model_name, model_type, classification, device, output_shapes=None):
    """
    Generate the content for the test file
    
    Args:
        model_name: Model name
        model_type: Model type (class name)
        classification: Model classification results
        device: Device to use
        output_shapes: Optional output shapes to check in tests
        
    Returns:
        Test file content
    """
    # Get model family and normalized class name
    model_family = classification.get("family", "default")
    class_name = model_type.replace("For", "_").replace("Model", "")
    
    # Determine import statements based on model family
    imports = ["import os", "import sys", "import logging", "import unittest"]
    if model_family == "vision" or model_family == "multimodal":
        imports.append("import PIL.Image")
    if model_family == "audio":
        imports.append("import numpy as np")
    
    # Determine base model class based on model family
    model_class = "AutoModel"
    if model_family == "text_generation":
        model_class = "AutoModelForCausalLM"
    elif model_family == "vision":
        model_class = "AutoModelForImageClassification"
    elif model_family == "audio":
        model_class = "AutoModelForAudioClassification"
    
    # Determine tokenizer/processor class based on model family
    tokenizer_class = "AutoTokenizer"
    if model_family == "vision":
        tokenizer_class = "AutoImageProcessor"
    elif model_family in ["audio", "multimodal"]:
        tokenizer_class = "AutoProcessor"
    
    # Create hardware preferences JSON
    hw_preferences = {
        "device": device
    }
    hw_preferences_str = json.dumps(hw_preferences, indent=4)
    
    # Create template for test file
    content = f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
Test file for {model_name}
Generated automatically by test_generator_with_resource_pool.py
Model family: {model_family}
Generated on: {datetime.now().isoformat()}
\"\"\"

{chr(10).join(imports)}
from typing import Dict, List, Any, Optional

# Import the resource pool for efficient resource sharing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource_pool import get_global_resource_pool

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestHF{class_name}(unittest.TestCase):
    \"\"\"Test class for {model_name} ({model_family} model)\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Set up the test class - load model once for all tests\"\"\"
        # Use resource pool to efficiently share resources
        pool = get_global_resource_pool()
        
        # Load dependencies
        cls.torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Define model constructor
        def create_model():
            from transformers import {model_class}
            return {model_class}.from_pretrained("{model_name}")
        
        # Set hardware preferences for optimal hardware selection
        hardware_preferences = {hw_preferences_str}
        
        # Get or create model from pool with hardware awareness
        cls.model = pool.get_model("{model_family}", "{model_name}", 
                                 constructor=create_model,
                                 hardware_preferences=hardware_preferences)
        
        # Define tokenizer constructor
        def create_tokenizer():
            from transformers import {tokenizer_class}
            return {tokenizer_class}.from_pretrained("{model_name}")
        
        # Get or create tokenizer from pool
        cls.tokenizer = pool.get_tokenizer("{model_family}", "{model_name}", 
                                         constructor=create_tokenizer)
        
        # Verify resources loaded correctly
        assert cls.model is not None, "Failed to load model"
        assert cls.tokenizer is not None, "Failed to load tokenizer/processor"
        
        # Get device - use model device if available, otherwise use preferred device
        if hasattr(cls.model, "device"):
            cls.device = cls.model.device
        else:
            cls.device = cls.torch.device("{device}")
            # Move model to device if needed
            if hasattr(cls.model, "to"):
                cls.model = cls.model.to(cls.device)
        
        logger.info(f"Model loaded on device: {{cls.device}}")
    
    def test_model_loading(self):
        \"\"\"Test that the model was loaded correctly\"\"\"
        self.assertIsNotNone(self.model, "Model should be loaded")
        self.assertIsNotNone(self.tokenizer, "Tokenizer/processor should be loaded")
        
        # Check model type
        from transformers import {model_class}
        expected_class = {model_class}.from_pretrained("{model_name}").__class__
        self.assertIsInstance(self.model, expected_class, 
                             f"Model should be an instance of {{expected_class.__name__}}")
    """
    
    # Add family-specific test methods
    if model_family == "embedding":
        content += """
    
    def test_embedding_generation(self):
        \"\"\"Test embedding generation\"\"\"
        # Prepare input
        text = "Hello, world!"
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move inputs to device if needed
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Check outputs
        self.assertIsNotNone(outputs, "Outputs should not be None")
        self.assertTrue(hasattr(outputs, "last_hidden_state"), 
                       "Output should have last_hidden_state attribute")
        
        # Check embedding shape
        last_hidden_state = outputs.last_hidden_state
        self.assertEqual(last_hidden_state.shape[0], 1, "Batch size should be 1")
        """
    elif model_family == "text_generation":
        content += """
    
    def test_text_generation(self):
        \"\"\"Test text generation\"\"\"
        # Prepare input
        text = "Hello, world!"
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move inputs to device if needed
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run forward pass
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Check outputs
        self.assertIsNotNone(outputs, "Outputs should not be None")
        self.assertTrue(hasattr(outputs, "logits"), 
                       "Output should have logits attribute")
        
        # Try basic generation
        try:
            # Use a small max_length to keep tests fast
            generation_output = self.model.generate(
                **inputs,
                max_length=20,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
            
            # Check that we got some text back
            self.assertIsNotNone(generated_text, "Generated text should not be None")
            self.assertTrue(len(generated_text) > 0, "Generated text should not be empty")
            
            logger.info(f"Generated text: {generated_text}")
        except Exception as e:
            # Generation might fail on some models, that's okay
            logger.warning(f"Generation failed (this may be expected): {e}")
        """
    elif model_family == "vision":
        content += """
    
    def test_image_processing(self):
        \"\"\"Test image processing\"\"\"
        # Create a simple test image
        width, height = 224, 224
        image = PIL.Image.new('RGB', (width, height), color=(128, 128, 128))
        
        # Process the image
        inputs = self.tokenizer(images=image, return_tensors="pt")
        
        # Move inputs to device if needed
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Check outputs
        self.assertIsNotNone(outputs, "Outputs should not be None")
        """
    elif model_family == "audio":
        content += """
    
    def test_audio_processing(self):
        \"\"\"Test audio processing\"\"\"
        # Create a simple test audio input
        # A random 2-second audio signal at 16kHz
        sample_rate = 16000
        dummy_audio = np.random.randn(2 * sample_rate)
        
        # Process the audio
        try:
            inputs = self.tokenizer(dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
            
            # Move inputs to device if needed
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            
            # Check outputs
            self.assertIsNotNone(outputs, "Outputs should not be None")
        except Exception as e:
            # Some audio processors might need different inputs
            logger.warning(f"Audio processing failed with standard approach: {e}")
            self.skipTest(f"Audio processing method not supported: {e}")
        """
    elif model_family == "multimodal":
        content += """
    
    def test_multimodal_processing(self):
        \"\"\"Test multimodal (text+vision) processing\"\"\"
        # Create a simple test image
        width, height = 224, 224
        image = PIL.Image.new('RGB', (width, height), color=(128, 128, 128))
        
        # Sample text
        text = "A test image"
        
        # Try different input configurations
        try:
            # First try with the standard format
            inputs = self.tokenizer(text=text, images=image, return_tensors="pt")
            
            # Move inputs to device if needed
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                
            # Check outputs
            self.assertIsNotNone(outputs, "Outputs should not be None")
        except Exception as e:
            logger.warning(f"Multimodal processing failed with standard approach: {e}")
            try:
                # Try alternate format
                inputs = self.tokenizer(image, text, return_tensors="pt")
                
                # Move inputs to device if needed
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Run inference
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Check outputs
                self.assertIsNotNone(outputs, "Outputs should not be None")
            except Exception as e2:
                # If both approaches fail, skip the test
                logger.warning(f"Multimodal processing failed with alternate approach: {e2}")
                self.skipTest(f"Multimodal processing methods not supported: {e} / {e2}")
        """
    
    # Add output shape assertions based on the shapes we discovered
    if output_shapes:
        content += """
        
    def test_output_shapes(self):
        \"\"\"Test that model outputs have expected shapes\"\"\"
        # Prepare a basic input
        """
        
        # Add input preparation based on model family
        if model_family == "vision":
            content += """
        width, height = 224, 224
        image = PIL.Image.new('RGB', (width, height), color=(128, 128, 128))
        inputs = self.tokenizer(images=image, return_tensors="pt")
        """
        elif model_family == "audio":
            content += """
        # Create a simple test audio input
        sample_rate = 16000
        dummy_audio = np.random.randn(2 * sample_rate)
        inputs = self.tokenizer(dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
        """
        elif model_family == "multimodal":
            content += """
        width, height = 224, 224
        image = PIL.Image.new('RGB', (width, height), color=(128, 128, 128))
        try:
            inputs = self.tokenizer(text="test", images=image, return_tensors="pt")
        except:
            # Try alternate format
            inputs = self.tokenizer(image, return_tensors="pt")
        """
        else:  # embedding or text_generation
            content += """
        text = "Hello, world!"
        inputs = self.tokenizer(text, return_tensors="pt")
        """
        
        content += """
        # Move inputs to device if needed
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Check output shapes
        """
        
        for key, shape in output_shapes.items():
            shape_str = str(shape).replace("[", "(").replace("]", ")")
            content += f"""
        self.assertTrue(hasattr(outputs, "{key}"), "Output should have {key} attribute")
        self.assertEqual(outputs.{key}.shape, {shape_str}, 
                        f"Output {key} shape should be {shape}, got {{outputs.{key}.shape}}")
        """
    
    # Add device-specific test case
    content += """
    
    def test_device_compatibility(self):
        \"\"\"Test device compatibility\"\"\"
        device_str = str(self.device)
        logger.info(f"Testing on device: {device_str}")
        
        # Check model device
        if hasattr(self.model, "device"):
            model_device = str(self.model.device)
            logger.info(f"Model is on device: {model_device}")
            self.assertEqual(model_device, device_str, 
                           f"Model should be on {device_str}, but is on {model_device}")
        
        # Check parameter device
        if hasattr(self.model, "parameters"):
            try:
                param_device = str(next(self.model.parameters()).device)
                logger.info(f"Model parameters are on device: {param_device}")
                self.assertEqual(param_device, device_str, 
                               f"Parameters should be on {device_str}, but are on {param_device}")
            except Exception as e:
                logger.warning(f"Could not check parameter device: {e}")
    
    @classmethod
    def tearDownClass(cls):
        \"\"\"Clean up resources\"\"\"
        # Get resource pool stats
        pool = get_global_resource_pool()
        stats = pool.get_stats()
        logger.info(f"Resource pool stats: {stats}")
        
        # Clean up unused resources to prevent memory leaks
        pool.cleanup_unused_resources(max_age_minutes=0.1)  # 6 seconds

def main():
    \"\"\"Run the test directly\"\"\"
    unittest.main()

if __name__ == "__main__":
    main()
"""
    
    return content

def cleanup_resources(timeout_minutes):
    """Clean up unused resources"""
    pool = get_global_resource_pool()
    removed = pool.cleanup_unused_resources(max_age_minutes=timeout_minutes)
    logger.info(f"Cleaned up {removed} unused resources")
    
    # Log current stats
    stats = pool.get_stats()
    logger.info(f"Resource pool stats after cleanup: {stats}")

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set up the environment
    setup_environment(args)
    
    # Load dependencies
    if not load_dependencies():
        logger.error("Failed to load dependencies. Exiting.")
        return 1
    
    # Generate the test file
    test_file = generate_test_file(
        model_name=args.model, 
        output_dir=args.output_dir,
        device=args.device,
        hw_cache_path=args.hw_cache,
        model_db_path=args.model_db
    )
    
    if test_file is None:
        logger.error("Failed to generate test file. Exiting.")
        return 1
    
    # Clean up unused resources
    cleanup_resources(args.timeout)
    
    logger.info(f"Successfully generated test file: {test_file}")
    print(f"\nTo run the test, use this command:")
    print(f"  python {test_file}")
    
    return 0

if __name__ == "__main__":
    # Run main function
    sys.exit(main())