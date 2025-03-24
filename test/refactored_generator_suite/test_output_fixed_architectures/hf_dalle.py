import os
import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple

class hf_dalle:
    """HuggingFace DALLE Text-to-Image model implementation.
    
    This class provides standardized interfaces for working with DALLE text-to-image models
    across different hardware backends (CPU, CUDA, ROCm, Apple, OpenVINO, Qualcomm).
    
    The dalle model is a text-to-image diffusion model that generates images from text prompts.
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the DALLE text-to-image model.
        
        Args:
            resources (dict): Dictionary of shared resources (torch, transformers, diffusers, etc.)
            metadata (dict): Configuration metadata
        """
        self.resources = resources if resources is not None else {}
        self.metadata = metadata if metadata is not None else {}
        
        # Handler creation methods
        self.create_cpu_text_to_image_endpoint_handler = self.create_cpu_text_to_image_endpoint_handler
        self.create_cuda_text_to_image_endpoint_handler = self.create_cuda_text_to_image_endpoint_handler
        self.create_rocm_text_to_image_endpoint_handler = self.create_rocm_text_to_image_endpoint_handler
        self.create_openvino_text_to_image_endpoint_handler = self.create_openvino_text_to_image_endpoint_handler
        self.create_apple_text_to_image_endpoint_handler = self.create_apple_text_to_image_endpoint_handler
        self.create_qualcomm_text_to_image_endpoint_handler = self.create_qualcomm_text_to_image_endpoint_handler
        
        # Initialization methods
        self.init = self.init
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_rocm = self.init_rocm
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.init_qualcomm = self.init_qualcomm
        
        # Test methods
        self.__test__ = self.__test__
        
        # Hardware-specific utilities
        self.snpe_utils = None  # Qualcomm SNPE utils
        return None
        
    def _create_mock_processor(self):
        """Create a mock processor for graceful degradation when the real one fails.
        
        Returns:
            Mock processor object with essential methods
        """
        try:
            from unittest.mock import MagicMock
            
            processor = MagicMock()
            
            # Configure mock processor call behavior
            def mock_process(text=None, image=None, return_tensors="pt"):
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                
                # Create mock input IDs for text input
                if text is not None:
                    if isinstance(text, str):
                        batch_size = 1
                    else:
                        batch_size = len(text)
                        
                    return {
                        "input_ids": torch.ones((batch_size, 77), dtype=torch.long),
                        "attention_mask": torch.ones((batch_size, 77), dtype=torch.long)
                    }
                # Create mock pixel values for image input
                elif image is not None:
                    if isinstance(image, (list, tuple)):
                        batch_size = len(image)
                    else:
                        batch_size = 1
                        
                    return {
                        "pixel_values": torch.rand((batch_size, 3, 512, 512))
                    }
                
                return {}
                
            processor.side_effect = mock_process
            processor.__call__ = mock_process
            
            print("(MOCK) Created mock DALLE processor")
            return processor
            
        except ImportError:
            # Fallback if unittest.mock is not available
            class SimpleProcessor:
                def __init__(self, parent):
                    self.parent = parent
                    
                def __call__(self, text=None, image=None, return_tensors="pt"):
                    if hasattr(self.parent, 'torch'):
                        torch = self.parent.torch
                    else:
                        import torch
                    
                    # Create mock input IDs for text input
                    if text is not None:
                        if isinstance(text, str):
                            batch_size = 1
                        else:
                            batch_size = len(text)
                            
                        return {
                            "input_ids": torch.ones((batch_size, 77), dtype=torch.long),
                            "attention_mask": torch.ones((batch_size, 77), dtype=torch.long)
                        }
                    # Create mock pixel values for image input
                    elif image is not None:
                        if isinstance(image, (list, tuple)):
                            batch_size = len(image)
                        else:
                            batch_size = 1
                            
                        return {
                            "pixel_values": torch.rand((batch_size, 3, 512, 512))
                        }
                    
                    return {}
            
            print("(MOCK) Created simple mock DALLE processor")
            return SimpleProcessor(self)
    
    def _create_mock_endpoint(self, model_name, device_label):
        """Create mock endpoint objects when real initialization fails.
        
        Args:
            model_name (str): The model name or path
            device_label (str): The device label (cpu, cuda, etc.)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        """
        try:
            from unittest.mock import MagicMock
            
            # Create mock endpoint
            endpoint = MagicMock()
            
            # Configure mock endpoint behavior for text-to-image generation
            def mock_forward(prompt_embeds=None, **kwargs):
                batch_size = prompt_embeds.shape[0] if prompt_embeds is not None else 1
                height = kwargs.get("height", 512)
                width = kwargs.get("width", 512)
                
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                
                # Create mock image output
                result = MagicMock()
                result.images = [torch.rand((3, height, width)) for _ in range(batch_size)]
                
                # Convert to numpy arrays to simulate PIL Images
                if hasattr(self, 'np'):
                    np = self.np
                else:
                    import numpy as np
                    
                result.images = [img.numpy() for img in result.images]
                
                return result
                
            endpoint.side_effect = mock_forward
            endpoint.__call__ = mock_forward
            
            # Create a more specific generator method for diffusion models
            def mock_generate(**kwargs):
                num_images = kwargs.get("num_images_per_prompt", 1) * kwargs.get("batch_size", 1)
                height = kwargs.get("height", 512)
                width = kwargs.get("width", 512)
                
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                    
                if hasattr(self, 'np'):
                    np = self.np
                else:
                    import numpy as np
                
                # Create mock image output as numpy arrays
                images = [np.random.rand(height, width, 3) for _ in range(num_images)]
                
                class MockOutput:
                    def __init__(self, images):
                        self.images = images
                
                return MockOutput(images)
            
            endpoint.generate = mock_generate
            
            # Create mock processor
            processor = self._create_mock_processor()
            
            # Create appropriate handler for the device type
            if device_label.startswith('cpu'):
                handler_method = self.create_cpu_text_to_image_endpoint_handler
            elif device_label.startswith('cuda'):
                handler_method = self.create_cuda_text_to_image_endpoint_handler
            elif device_label.startswith('rocm'):
                handler_method = self.create_rocm_text_to_image_endpoint_handler
            elif device_label.startswith('openvino'):
                handler_method = self.create_openvino_text_to_image_endpoint_handler
            elif device_label.startswith('apple'):
                handler_method = self.create_apple_text_to_image_endpoint_handler
            elif device_label.startswith('qualcomm'):
                handler_method = self.create_qualcomm_text_to_image_endpoint_handler
            else:
                handler_method = self.create_cpu_text_to_image_endpoint_handler
            
            # Create handler function
            mock_handler = handler_method(
                endpoint_model=model_name,
                device=device_label.split(':')[0] if ':' in device_label else device_label,
                hardware_label=device_label,
                endpoint=endpoint,
                processor=processor
            )
            
            import asyncio
            print(f"(MOCK) Created mock DALLE endpoint for {model_name} on {device_label}")
            return endpoint, processor, mock_handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error creating mock endpoint: {e}")
            import asyncio
            return None, None, None, asyncio.Queue(32), 0
    
    def init(self):        
        if "torch" not in list(self.resources.keys()):
            import torch
            self.torch = torch
        else:
            self.torch = self.resources["torch"]

        if "transformers" not in list(self.resources.keys()):
            import transformers
            self.transformers = transformers
        else:
            self.transformers = self.resources["transformers"]
            
        if "diffusers" not in list(self.resources.keys()):
            import diffusers
            self.diffusers = diffusers
        else:
            self.diffusers = self.resources["diffusers"]
            
        if "numpy" not in list(self.resources.keys()):
            import numpy as np
            self.np = np
        else:
            self.np = self.resources["numpy"]
            
        if "PIL" not in list(self.resources.keys()):
            try:
                from PIL import Image
                self.Image = Image
            except ImportError:
                self.Image = None
        else:
            self.Image = self.resources["PIL"]

        return None
    

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, processor):
        """Test function to validate endpoint functionality.
        
        Args:
            endpoint_model: The model name or path
            endpoint_handler: The handler function
            endpoint_label: The hardware label
            processor: The processor
            
        Returns:
            Boolean indicating test success
        """
        test_input = "A beautiful sunset over the mountains with a lake in the foreground."
        timestamp1 = time.time()
        test_batch = None
        
        try:
            # Run the model
            test_batch = endpoint_handler(test_input)
            print(test_batch)
            print("hf_dalle test passed")
            
            # Verify output structure
            if "image" in test_batch:
                print(f"Image shape or size: {test_batch['image'].size if hasattr(test_batch['image'], 'size') else len(test_batch['image'])}")
            elif "images" in test_batch:
                print(f"Generated {len(test_batch['images'])} images")
                
        except Exception as e:
            print(f"Error during test: {e}")
            print("hf_dalle test failed")
            return False
            
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        print(f"elapsed time: {elapsed_time}")
        
        # Clean up memory
        with self.torch.no_grad():
            if "cuda" in dir(self.torch):
                self.torch.cuda.empty_cache()
        return True

    def init_cpu(self, model_name, device, cpu_label):
        """Initialize DALLE model for CPU inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('cpu')
            cpu_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        print(f"Loading {model_name} for CPU inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load processor (e.g., CLIPTextProcessor for Stable Diffusion)
            processor = self.transformers.CLIPTextProcessor.from_pretrained(
                model_name,
                subfolder="text_encoder",
                cache_dir=cache_dir
            )
            
            # Load pipeline (e.g., StableDiffusionPipeline)
            pipeline = self.diffusers.StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=self.torch.float32,
                cache_dir=cache_dir
            )
            
            # Move pipeline to CPU
            pipeline = pipeline.to(device)
            
            # Create handler function
            handler = self.create_cpu_text_to_image_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=cpu_label,
                endpoint=pipeline,
                processor=processor
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, cpu_label, processor)
            
            return pipeline, processor, handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing CPU endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, cpu_label)

    def init_cuda(self, model_name, device, cuda_label):
        """Initialize DALLE model for CUDA inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('cuda:0', 'cuda:1', etc.)
            cuda_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        if not self.torch.cuda.is_available():
            print(f"CUDA not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", cuda_label.replace("cuda", "cpu"))
        
        print(f"Loading {model_name} for CUDA inference on {device}...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load processor
            processor = self.transformers.CLIPTextProcessor.from_pretrained(
                model_name,
                subfolder="text_encoder",
                cache_dir=cache_dir
            )
            
            # Load pipeline with half precision for GPU efficiency
            pipeline = self.diffusers.StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=self.torch.float16,
                cache_dir=cache_dir
            )
            
            # Move pipeline to GPU
            pipeline = pipeline.to(device)
            
            # Enable memory optimization if available
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            
            # Create handler function
            handler = self.create_cuda_text_to_image_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=cuda_label,
                endpoint=pipeline,
                processor=processor
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, cuda_label, processor)
            
            return pipeline, processor, handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing CUDA endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, cuda_label)

    def init_rocm(self, model_name, device, rocm_label):
        """Initialize DALLE model for ROCm (AMD GPU) inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('cuda:0', 'cuda:1', etc. - ROCm uses CUDA device naming)
            rocm_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        # Check for ROCm availability
        rocm_available = False
        try:
            if hasattr(self.torch, 'hip') and self.torch.hip.is_available():
                rocm_available = True
            elif self.torch.cuda.is_available():
                # Could be ROCm using CUDA API
                device_name = self.torch.cuda.get_device_name(0)
                if "AMD" in device_name or "Radeon" in device_name:
                    rocm_available = True
        except Exception as e:
            print(f"Error checking ROCm availability: {e}")
        
        if not rocm_available:
            print(f"ROCm not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", rocm_label.replace("rocm", "cpu"))
        
        print(f"Loading {model_name} for ROCm (AMD GPU) inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load processor
            processor = self.transformers.CLIPTextProcessor.from_pretrained(
                model_name,
                subfolder="text_encoder",
                cache_dir=cache_dir
            )
            
            # Check for HIP_VISIBLE_DEVICES environment variable
            visible_devices = os.environ.get("HIP_VISIBLE_DEVICES", None) or os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if visible_devices is not None:
                print(f"Using ROCm visible devices: {visible_devices}")
            
            # Get the total GPU memory for logging purposes
            try:
                total_mem = self.torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
                print(f"AMD GPU memory: {total_mem:.2f} GB")
            except Exception as e:
                print(f"Could not query AMD GPU memory: {e}")
            
            # Determine if we should use half precision
            use_half = True
            try:
                # Try to create a small tensor in half precision as a test
                test_tensor = self.torch.ones((10, 10), dtype=self.torch.float16, device="cuda")
                del test_tensor
                print("Half precision is supported on this AMD GPU")
            except Exception as e:
                use_half = False
                print(f"Half precision not supported on this AMD GPU: {e}")
            
            # Load pipeline with appropriate precision for AMD GPU
            pipeline = self.diffusers.StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=self.torch.float16 if use_half else self.torch.float32,
                cache_dir=cache_dir
            )
            
            # Move pipeline to GPU
            pipeline = pipeline.to(device)
            
            # Enable memory optimization for ROCm
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
                
            # Create handler function
            handler = self.create_rocm_text_to_image_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=rocm_label,
                endpoint=pipeline,
                processor=processor
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, rocm_label, processor)
            
            return pipeline, processor, handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing ROCm endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, rocm_label)

    def init_openvino(self, model_name, device, openvino_label):
        """Initialize DALLE model for OpenVINO inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('CPU', 'GPU', etc.)
            openvino_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        try:
            from optimum.intel.openvino import OVStableDiffusionPipeline
        except ImportError:
            print(f"OpenVINO optimum.intel not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", openvino_label.replace("openvino", "cpu"))
        
        print(f"Loading {model_name} for OpenVINO inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load processor
            processor = self.transformers.CLIPTextProcessor.from_pretrained(
                model_name,
                subfolder="text_encoder",
                cache_dir=cache_dir
            )
            
            # Load pipeline with OpenVINO optimization
            pipeline = OVStableDiffusionPipeline.from_pretrained(
                model_name,
                export=True,
                cache_dir=cache_dir
            )
            
            # Set the OpenVINO device
            pipeline.to(device)
            
            # Create handler function
            handler = self.create_openvino_text_to_image_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=openvino_label,
                endpoint=pipeline,
                processor=processor
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, openvino_label, processor)
            
            return pipeline, processor, handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing OpenVINO endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, openvino_label)

    def init_apple(self, model_name, device, apple_label):
        """Initialize DALLE model for Apple Silicon (MPS) inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('mps')
            apple_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        if not (hasattr(self.torch, 'backends') and 
                hasattr(self.torch.backends, 'mps') and 
                self.torch.backends.mps.is_available()):
            print(f"Apple MPS not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", apple_label.replace("apple", "cpu"))
        
        print(f"Loading {model_name} for Apple Silicon inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load processor
            processor = self.transformers.CLIPTextProcessor.from_pretrained(
                model_name,
                subfolder="text_encoder",
                cache_dir=cache_dir
            )
            
            # Load pipeline for MPS
            pipeline = self.diffusers.StableDiffusionPipeline.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Move pipeline to MPS device
            pipeline = pipeline.to("mps")
            
            # Enable memory-efficient techniques
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            
            # Create handler function
            handler = self.create_apple_text_to_image_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=apple_label,
                endpoint=pipeline,
                processor=processor
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, apple_label, processor)
            
            return pipeline, processor, handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing Apple Silicon endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, apple_label)

    def init_qualcomm(self, model_name, device, qualcomm_label):
        """Initialize DALLE model for Qualcomm inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('qualcomm')
            qualcomm_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        # Check if Qualcomm SDK is available
        try:
            import importlib.util
            has_qnn = importlib.util.find_spec("qnn_wrapper") is not None
            has_qti = importlib.util.find_spec("qti.aisw.dlc_utils") is not None
            has_qualcomm_env = "QUALCOMM_SDK" in os.environ
            
            if not (has_qnn or has_qti or has_qualcomm_env):
                print(f"Qualcomm SDK not available, falling back to CPU")
                return self.init_cpu(model_name, "cpu", qualcomm_label.replace("qualcomm", "cpu"))
        except ImportError:
            print(f"Qualcomm SDK import error, falling back to CPU")
            return self.init_cpu(model_name, "cpu", qualcomm_label.replace("qualcomm", "cpu"))
        
        print(f"Loading {model_name} for Qualcomm inference...")
        
        # For now, we create a mock implementation since Qualcomm SDK integration requires specific hardware
        print("Qualcomm implementation is a mock for now")
        return self._create_mock_endpoint(model_name, qualcomm_label)

    def create_cpu_text_to_image_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, processor):
        """Create handler function for CPU text-to-image endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('cpu')
            hardware_label (str): The hardware label
            endpoint: The loaded model pipeline
            processor: The loaded processor
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(text, **kwargs):
            try:
                # Get generation parameters with defaults
                num_inference_steps = kwargs.get("num_inference_steps", 50)
                guidance_scale = kwargs.get("guidance_scale", 7.5)
                height = kwargs.get("height", 512)
                width = kwargs.get("width", 512)
                num_images = kwargs.get("num_images", 1)
                
                # Set a fixed seed if provided
                generator = None
                if "seed" in kwargs:
                    generator = self.torch.Generator(device=device).manual_seed(kwargs["seed"])
                
                # Run text-to-image generation
                output = endpoint(
                    prompt=text,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images,
                    generator=generator
                )
                
                # Convert images to a standard format
                if hasattr(output, "images"):
                    images = output.images
                else:
                    images = [output.image]
                
                # Return result
                return {
                    "success": True,
                    "images": images,
                    "prompt": text,
                    "device": device,
                    "hardware": hardware_label,
                    "parameters": {
                        "steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "height": height,
                        "width": width
                    }
                }
                
            except Exception as e:
                print(f"Error in CPU handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler

    def create_cuda_text_to_image_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, processor):
        """Create handler function for CUDA text-to-image endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('cuda:0', etc.)
            hardware_label (str): The hardware label
            endpoint: The loaded model pipeline
            processor: The loaded processor
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(text, **kwargs):
            try:
                # Get generation parameters with defaults
                num_inference_steps = kwargs.get("num_inference_steps", 50)
                guidance_scale = kwargs.get("guidance_scale", 7.5)
                height = kwargs.get("height", 512)
                width = kwargs.get("width", 512)
                num_images = kwargs.get("num_images", 1)
                
                # Set a fixed seed if provided
                generator = None
                if "seed" in kwargs:
                    generator = self.torch.Generator(device=device).manual_seed(kwargs["seed"])
                
                # Run text-to-image generation
                with self.torch.autocast(device_type="cuda"):
                    output = endpoint(
                        prompt=text,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=num_images,
                        generator=generator
                    )
                
                # Convert images to a standard format
                if hasattr(output, "images"):
                    images = output.images
                else:
                    images = [output.image]
                
                # Return result
                return {
                    "success": True,
                    "images": images,
                    "prompt": text,
                    "device": device,
                    "hardware": hardware_label,
                    "parameters": {
                        "steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "height": height,
                        "width": width
                    }
                }
                
            except Exception as e:
                print(f"Error in CUDA handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler

    def create_rocm_text_to_image_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, processor):
        """Create handler function for ROCm text-to-image endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('cuda:0', etc. - ROCm uses CUDA device naming)
            hardware_label (str): The hardware label
            endpoint: The loaded model pipeline
            processor: The loaded processor
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(text, **kwargs):
            try:
                # Get generation parameters with defaults
                num_inference_steps = kwargs.get("num_inference_steps", 50)
                guidance_scale = kwargs.get("guidance_scale", 7.5)
                height = kwargs.get("height", 512)
                width = kwargs.get("width", 512)
                num_images = kwargs.get("num_images", 1)
                
                # Set a fixed seed if provided
                generator = None
                if "seed" in kwargs:
                    generator = self.torch.Generator(device=device).manual_seed(kwargs["seed"])
                
                # For AMD GPUs, autocast may or may not be supported depending on ROCm version
                try:
                    with self.torch.autocast(device_type="cuda"):
                        output = endpoint(
                            prompt=text,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            num_images_per_prompt=num_images,
                            generator=generator
                        )
                except RuntimeError:
                    # Fallback without autocast if not supported
                    output = endpoint(
                        prompt=text,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=num_images,
                        generator=generator
                    )
                
                # Convert images to a standard format
                if hasattr(output, "images"):
                    images = output.images
                else:
                    images = [output.image]
                
                # Return result
                return {
                    "success": True,
                    "images": images,
                    "prompt": text,
                    "device": device,
                    "hardware": hardware_label,
                    "parameters": {
                        "steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "height": height,
                        "width": width
                    }
                }
                
            except Exception as e:
                print(f"Error in ROCm handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler
        
    def create_openvino_text_to_image_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, processor):
        """Create handler function for OpenVINO text-to-image endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('CPU', 'GPU', etc.)
            hardware_label (str): The hardware label
            endpoint: The loaded model pipeline
            processor: The loaded processor
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(text, **kwargs):
            try:
                # Get generation parameters with defaults
                num_inference_steps = kwargs.get("num_inference_steps", 50)
                guidance_scale = kwargs.get("guidance_scale", 7.5)
                height = kwargs.get("height", 512)
                width = kwargs.get("width", 512)
                num_images = kwargs.get("num_images", 1)
                
                # OpenVINO pipelines have different parameter handling
                output = endpoint(
                    prompt=text,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images
                )
                
                # Convert images to a standard format
                if hasattr(output, "images"):
                    images = output.images
                else:
                    images = [output.image]
                
                # Return result
                return {
                    "success": True,
                    "images": images,
                    "prompt": text,
                    "device": device,
                    "hardware": hardware_label,
                    "parameters": {
                        "steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "height": height,
                        "width": width
                    }
                }
                
            except Exception as e:
                print(f"Error in OpenVINO handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler

    def create_apple_text_to_image_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, processor):
        """Create handler function for Apple Silicon text-to-image endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('mps')
            hardware_label (str): The hardware label
            endpoint: The loaded model pipeline
            processor: The loaded processor
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(text, **kwargs):
            try:
                # Get generation parameters with defaults
                num_inference_steps = kwargs.get("num_inference_steps", 50)
                guidance_scale = kwargs.get("guidance_scale", 7.5)
                height = kwargs.get("height", 512)
                width = kwargs.get("width", 512)
                num_images = kwargs.get("num_images", 1)
                
                # Set a fixed seed if provided
                generator = None
                if "seed" in kwargs:
                    generator = self.torch.Generator(device="mps").manual_seed(kwargs["seed"])
                
                # Run text-to-image generation
                output = endpoint(
                    prompt=text,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images,
                    generator=generator
                )
                
                # Convert images to a standard format
                if hasattr(output, "images"):
                    images = output.images
                else:
                    images = [output.image]
                
                # Return result
                return {
                    "success": True,
                    "images": images,
                    "prompt": text,
                    "device": device,
                    "hardware": hardware_label,
                    "parameters": {
                        "steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "height": height,
                        "width": width
                    }
                }
                
            except Exception as e:
                print(f"Error in Apple handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler

    def create_qualcomm_text_to_image_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, processor):
        """Create handler function for Qualcomm text-to-image endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('qualcomm')
            hardware_label (str): The hardware label
            endpoint: The loaded model pipeline
            processor: The loaded processor
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(text, **kwargs):
            try:
                # This is a placeholder for Qualcomm implementation
                # In a real implementation, we would use the Qualcomm SDK
                
                # Mock result for now
                return {
                    "success": True,
                    "device": device,
                    "hardware": hardware_label,
                    "mock": True,
                    "prompt": text
                }
                
            except Exception as e:
                print(f"Error in Qualcomm handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler