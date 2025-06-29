#!/usr/bin/env python3
"""
Template for diffusion models (Stable Diffusion, Kandinsky, etc.)

This template is designed for generative image models based on diffusion,
which are used for text-to-image generation, image-to-image transformation,
inpainting, and other image generation tasks.
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
    from diffusers import {model_class_name}, {processor_class_name}
except ImportError:
    raise ImportError(
        "The diffusers package is required to use this model. "
        "Please install it with `pip install diffusers`."
    )

logger = logging.getLogger(__name__)

class {skillset_class_name}:
    """
    Skillset for {model_type_upper} - a diffusion-based generative model
    for image synthesis from text prompts or other inputs.
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
        self.pipeline = None
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
        """Initialize the diffusion pipeline."""
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
            
            self.is_initialized = True
            logger.info(f"Initialized {self.model_id} in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    def generate_image(self, prompt, negative_prompt=None, guidance_scale=7.5, num_inference_steps=50, **kwargs):
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt to guide generation away from certain concepts
            guidance_scale: Scale for classifier-free guidance
            num_inference_steps: Number of denoising steps (more = higher quality but slower)
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            Dictionary containing generated images
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Set up generation parameters
            generation_kwargs = {
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            }
            
            # Add optional parameters if provided
            if negative_prompt is not None:
                generation_kwargs["negative_prompt"] = negative_prompt
            
            # Add height and width if provided
            if "height" in kwargs:
                generation_kwargs["height"] = kwargs.pop("height")
            if "width" in kwargs:
                generation_kwargs["width"] = kwargs.pop("width")
                
            # Add any other parameters
            generation_kwargs.update(kwargs)
            
            # Generate image
            start_time = time.time()
            result = self.pipeline(prompt, **generation_kwargs)
            generation_time = time.time() - start_time
            
            # Process output
            images = result.images if hasattr(result, "images") else [result[0]]
            
            # Return results with metadata
            return {
                "images": images,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "generation_time": generation_time,
                "model_id": self.model_id,
                "parameters": generation_kwargs,
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return {"error": str(e)}
    
    def image_to_image(self, prompt, init_image, strength=0.8, **kwargs):
        """
        Generate a new image based on an input image and text prompt.
        
        Args:
            prompt: Text prompt to guide the transformation
            init_image: Initial image to transform (PIL Image or path)
            strength: How much to transform the image (0-1)
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            Dictionary containing generated images
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Load image if a path is provided
            if isinstance(init_image, str):
                try:
                    init_image = Image.open(init_image).convert("RGB")
                except Exception as e:
                    logger.error(f"Error loading image from {init_image}: {e}")
                    raise
            
            # Make sure we're using the img2img pipeline
            if not hasattr(self.pipeline, "image_processor"):
                logger.warning("Switching to image-to-image pipeline")
                from diffusers import StableDiffusionImg2ImgPipeline
                self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.hardware_info["supports_half_precision"] else torch.float32
                ).to(self.device)
            
            # Set up generation parameters
            generation_kwargs = {
                "image": init_image,
                "strength": strength,
            }
            generation_kwargs.update(kwargs)
            
            # Generate image
            start_time = time.time()
            result = self.pipeline(prompt, **generation_kwargs)
            generation_time = time.time() - start_time
            
            # Process output
            images = result.images if hasattr(result, "images") else [result[0]]
            
            # Return results with metadata
            return {
                "images": images,
                "prompt": prompt,
                "generation_time": generation_time,
                "model_id": self.model_id,
                "parameters": generation_kwargs,
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Error in image-to-image generation: {e}")
            return {"error": str(e)}
    
    def inpaint(self, prompt, image, mask_image, **kwargs):
        """
        Inpaint parts of an image based on a mask.
        
        Args:
            prompt: Text prompt to guide the inpainting
            image: Input image (PIL Image or path)
            mask_image: Mask defining areas to inpaint (PIL Image or path)
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            Dictionary containing generated images
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Load images if paths are provided
            if isinstance(image, str):
                try:
                    image = Image.open(image).convert("RGB")
                except Exception as e:
                    logger.error(f"Error loading image from {image}: {e}")
                    raise
            
            if isinstance(mask_image, str):
                try:
                    mask_image = Image.open(mask_image).convert("RGB")
                except Exception as e:
                    logger.error(f"Error loading mask from {mask_image}: {e}")
                    raise
            
            # Make sure we're using the inpainting pipeline
            if not hasattr(self.pipeline, "mask_processor"):
                logger.warning("Switching to inpainting pipeline")
                from diffusers import StableDiffusionInpaintPipeline
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.hardware_info["supports_half_precision"] else torch.float32
                ).to(self.device)
            
            # Set up generation parameters
            generation_kwargs = {
                "image": image,
                "mask_image": mask_image,
            }
            generation_kwargs.update(kwargs)
            
            # Generate image
            start_time = time.time()
            result = self.pipeline(prompt, **generation_kwargs)
            generation_time = time.time() - start_time
            
            # Process output
            images = result.images if hasattr(result, "images") else [result[0]]
            
            # Return results with metadata
            return {
                "images": images,
                "prompt": prompt,
                "generation_time": generation_time,
                "model_id": self.model_id,
                "parameters": generation_kwargs,
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Error in inpainting: {e}")
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
            # Text-to-image test (with minimal steps for speed)
            result = skillset.generate_image(
                "A beautiful mountain landscape with a lake",
                num_inference_steps=2  # Very few steps for quick testing
            )
            
            if "error" in result:
                print(f"Test failed: {result['error']}")
                return False
                
            print(f"Image generation test complete. Generated {len(result['images'])} images")
            print(f"Generation time: {result['generation_time']:.2f}s")
            
            # Try to display the first image
            try:
                result["images"][0].show()
            except:
                print("Could not display image (likely running in a headless environment)")
            
            print("All tests completed successfully!")
            return True
            
        except Exception as e:
            print(f"Test failed: {e}")
            return False


if __name__ == "__main__":
    # Run tests when the script is executed directly
    {test_class_name}.run_tests()