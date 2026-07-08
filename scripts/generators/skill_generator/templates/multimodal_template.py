#!/usr/bin/env python3
"""
Template for multimodal models (FLAVA, LLaVA, Fuyu, etc.)

This template is designed for models that can process multiple modalities,
such as images, text, and sometimes audio inputs.
"""

import os
import time
import json
import torch
import logging
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    from transformers import {model_class_name}, {processor_class_name}
except ImportError:
    raise ImportError(
        "The transformers package is required to use this model. "
        "Please install it with `pip install transformers`."
    )

logger = logging.getLogger(__name__)

class {skillset_class_name}:
    """
    Skillset for {model_type_upper} - a multimodal model that can process 
    combinations of text, images, and other modalities.
    """
    
    def __init__(self, model_id: str = "{default_model_id}", device: str = "cpu", **kwargs):
        """
        Initialize the {model_type_upper} model.
        
        Args:
            model_id: HuggingFace model ID or path
            device: Device to run the model on ('cpu', 'cuda', 'mps', etc.)
            **kwargs: Additional arguments to pass to the model
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.is_initialized = False
        
        # Track hardware info for reporting
        self.hardware_info = {
            "device": device,
            "device_name": None,
            "memory_available": None,
            "supports_half_precision": False
        }
        
        # Initialize the model if auto_init is True
        auto_init = kwargs.get("auto_init", True)
        if auto_init:
            self.initialize()
    
    def initialize(self):
        """Initialize the model and processor."""
        if self.is_initialized:
            return
        
        logger.info(f"Initializing {self.model_id} on {self.device}")
        start_time = time.time()
        
        try:
            # Check if CUDA is available when device is cuda
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Check if MPS is available when device is mps
            if self.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Check if ROCm/HIP is available when device is rocm
            if self.device == "rocm":
                rocm_available = False
                try:
                    if hasattr(torch, 'hip') and torch.hip.is_available():
                        rocm_available = True
                    elif torch.cuda.is_available():
                        # Could be ROCm using CUDA API
                        device_name = torch.cuda.get_device_name(0)
                        if "AMD" in device_name or "Radeon" in device_name:
                            rocm_available = True
                            self.device = "cuda"  # ROCm uses CUDA device map
                except:
                    pass
                
                if not rocm_available:
                    logger.warning("ROCm requested but not available, falling back to CPU")
                    self.device = "cpu"
            
            # Record hardware info
            if self.device.startswith("cuda"):
                self.hardware_info["device_name"] = torch.cuda.get_device_name(0)
                self.hardware_info["memory_available"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.hardware_info["supports_half_precision"] = True
            
            # Device-specific initialization
            {device_init_code}
            
            # Initialize processor
            self.processor = {processor_class_name}.from_pretrained(self.model_id)
            
            self.is_initialized = True
            logger.info(f"Initialized {self.model_id} in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def process_multimodal(self, text_inputs=None, image_inputs=None, audio_inputs=None, **kwargs):
        """
        Process multimodal inputs (text, images, audio).
        
        Args:
            text_inputs: Text input or list of text inputs
            image_inputs: Image input (PIL Image, file path, or URL) or list of images
            audio_inputs: Audio input or list of audio inputs
            **kwargs: Additional arguments for the model's forward pass
            
        Returns:
            Dictionary containing model outputs
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Prepare text inputs if provided
            if text_inputs is not None:
                if isinstance(text_inputs, str):
                    text_inputs = [text_inputs]
            
            # Prepare image inputs if provided
            if image_inputs is not None:
                # Handle string inputs (file paths or URLs)
                if isinstance(image_inputs, str):
                    try:
                        image_inputs = Image.open(image_inputs).convert('RGB')
                    except Exception as e:
                        logger.error(f"Error loading image from {image_inputs}: {e}")
                        raise
                
                # Handle list of string inputs
                if isinstance(image_inputs, list) and all(isinstance(img, str) for img in image_inputs):
                    try:
                        image_inputs = [Image.open(img).convert('RGB') for img in image_inputs]
                    except Exception as e:
                        logger.error(f"Error loading images: {e}")
                        raise
            
            # Process inputs with the processor
            inputs = self.processor(
                text=text_inputs,
                images=image_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            
            # Move inputs to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, **kwargs)
            
            # Process outputs based on model type and task
            result = self._process_outputs(outputs, text_inputs, image_inputs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            return {"error": str(e)}
    
    def _process_outputs(self, outputs, text_inputs=None, image_inputs=None):
        """
        Process model outputs based on model type and task.
        
        Args:
            outputs: Model outputs
            text_inputs: Original text inputs
            image_inputs: Original image inputs
            
        Returns:
            Processed results
        """
        # Different multimodal models have different output formats
        # Handle each case appropriately
        
        # Case 1: Image-to-text models (like BLIP, GIT)
        if hasattr(outputs, "sequences") and text_inputs is None:
            # This might be an image captioning model
            generated_text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
            return {
                "generated_text": generated_text,
                "model_id": self.model_id,
                "device": self.device
            }
        
        # Case 2: Visual question answering (like LLaVA)
        elif hasattr(outputs, "sequences") and text_inputs is not None:
            # This might be a visual QA model
            generated_text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
            return {
                "generated_text": generated_text,
                "model_id": self.model_id,
                "device": self.device
            }
        
        # Case 3: Multimodal embeddings (like FLAVA)
        elif hasattr(outputs, "multimodal_embeddings"):
            # This is a multimodal embedding model
            embeddings = outputs.multimodal_embeddings.cpu().numpy().tolist()
            return {
                "embeddings": embeddings,
                "model_id": self.model_id,
                "device": self.device
            }
            
        # Case 4: Visual reasoning models that return logits
        elif hasattr(outputs, "logits"):
            # This might be a classification or reasoning model
            logits = outputs.logits.cpu().numpy().tolist()
            return {
                "logits": logits,
                "model_id": self.model_id,
                "device": self.device
            }
        
        # Default case: return the raw outputs converted to Python data types
        else:
            result = {}
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.cpu().numpy().tolist()
                else:
                    result[key] = value
            
            result["model_id"] = self.model_id
            result["device"] = self.device
            
            return result
    
    def generate(self, prompt=None, image=None, **kwargs):
        """
        Generate text from a prompt and/or image.
        
        Args:
            prompt: Text prompt
            image: Image input (PIL Image, file path, or URL)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Process inputs
            inputs = self.process_multimodal(text_inputs=prompt, image_inputs=image, output_hidden_states=False)
            
            # Set generation parameters
            generation_kwargs = {
                "max_length": kwargs.get("max_length", 100),
                "num_beams": kwargs.get("num_beams", 1),
                "temperature": kwargs.get("temperature", 1.0),
                "top_p": kwargs.get("top_p", 1.0),
                "top_k": kwargs.get("top_k", 50),
                "do_sample": kwargs.get("do_sample", False),
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
            }
            
            # Generate text
            output_ids = self.model.generate(**inputs, **generation_kwargs)
            generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)
            
            return {
                "generated_text": generated_text,
                "model_id": self.model_id,
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            return {"error": str(e)}
    
    def embed(self, text=None, image=None, **kwargs):
        """
        Get embeddings for text and/or image inputs.
        
        Args:
            text: Text input or list of text inputs
            image: Image input or list of image inputs
            **kwargs: Additional embedding parameters
            
        Returns:
            Embeddings
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Process inputs
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, **kwargs)
            
            # Extract embeddings based on model architecture
            if hasattr(outputs, "multimodal_embeddings"):
                embeddings = outputs.multimodal_embeddings.cpu().numpy().tolist()
            elif hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output.cpu().numpy().tolist()
            elif hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
            else:
                raise ValueError("Could not extract embeddings from model outputs")
            
            return {
                "embeddings": embeddings,
                "model_id": self.model_id,
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            return {"error": str(e)}
    
    def get_hardware_info(self):
        """Get information about the hardware being used."""
        return self.hardware_info


# For testing
class {test_class_name}:
    """Test class for {model_type_upper} model."""
    
    @staticmethod
    def run_tests():
        """Run basic tests for the model."""
        model_id = "{default_model_id}"
        skillset = {skillset_class_name}(model_id=model_id, device="cpu")
        
        # Test with dummy inputs
        try:
            # Text-only test
            text_result = skillset.process_multimodal(text_inputs="This is a test")
            print(f"Text test complete. Result keys: {text_result.keys()}")
            
            # Create a simple test image (red square)
            try:
                import numpy as np
                from PIL import Image
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)
                img_array[:100, :100] = [255, 0, 0]  # Red square
                test_image = Image.fromarray(img_array)
                
                # Image-only test
                img_result = skillset.process_multimodal(image_inputs=test_image)
                print(f"Image test complete. Result keys: {img_result.keys()}")
                
                # Multimodal test
                mm_result = skillset.process_multimodal(
                    text_inputs="What's in this image?", 
                    image_inputs=test_image
                )
                print(f"Multimodal test complete. Result keys: {mm_result.keys()}")
                
            except ImportError:
                print("PIL not available, skipping image tests")
            
            print("All tests completed successfully!")
            return True
            
        except Exception as e:
            print(f"Test failed: {e}")
            return False


if __name__ == "__main__":
    # Run tests when the script is executed directly
    {test_class_name}.run_tests()