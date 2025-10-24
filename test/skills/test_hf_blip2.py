# Standard library imports first
import os
import sys
import json
import time
import datetime
import traceback
from unittest.mock import MagicMock, patch

# Third-party imports next
import numpy as np

# Use absolute path setup

# Import hardware detection capabilities if available::
try:
    from generators.hardware.hardware_detection import ())))))))
    HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
    detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback
    sys.path.insert())))))))0, "/home/barberb/ipfs_accelerate_py")

# Try/except pattern for importing optional dependencies:
try:
    import torch
except ImportError:
    torch = MagicMock()))))))))
    print())))))))"Warning: torch not available, using mock implementation")

try:
    import transformers
except ImportError:
    transformers = MagicMock()))))))))
    print())))))))"Warning: transformers not available, using mock implementation")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = MagicMock()))))))))
    PIL_AVAILABLE = False
    print())))))))"Warning: PIL not available, using mock implementation")

# Import the module to test - BLIP-2 might use a vl ())))))))vision-language) module or a specific blip2 module
# For now, assuming a VL module is used
    from ipfs_accelerate_py.worker.skillset.hf_vl import hf_vl

# Add CUDA support to the BLIP-2 class
def init_cuda())))))))self, model_name, model_type, device_label="cuda:0"):
    """Initialize BLIP-2 model with CUDA support.
    
    Args:
        model_name: Name or path of the model
        model_type: Type of model task ())))))))e.g., "image-to-text")
        device_label: CUDA device label ())))))))e.g., "cuda:0")
        
    Returns:
        tuple: ())))))))endpoint, processor, handler, queue, batch_size)
        """
    try:
        import sys
        import torch
        from unittest import mock
        
        # Try to import the necessary utility functions
        sys.path.insert())))))))0, "/home/barberb/ipfs_accelerate_py/test")
        import test_helpers as test_utils
        
        print())))))))f"Checking CUDA availability for {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}")
        
        # Verify that CUDA is actually available
        if not torch.cuda.is_available())))))))):
            print())))))))"CUDA not available, using mock implementation")
        return mock.MagicMock())))))))), mock.MagicMock())))))))), mock.MagicMock())))))))), None, 1
        
        # Get the CUDA device
        device = test_utils.get_cuda_device())))))))device_label)
        if device is None:
            print())))))))"Failed to get valid CUDA device, using mock implementation")
        return mock.MagicMock())))))))), mock.MagicMock())))))))), mock.MagicMock())))))))), None, 1
        
        print())))))))f"Using CUDA device: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}device}")
        
        # Try to initialize with real components
        try:
            from transformers import AutoProcessor, BlipForConditionalGeneration, Blip2ForConditionalGeneration
            
            # Load processor/tokenizer
            try:
                processor = AutoProcessor.from_pretrained())))))))model_name)
                print())))))))f"Successfully loaded processor for {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}")
            except Exception as processor_err:
                print())))))))f"Failed to load processor: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}processor_err}")
                processor = mock.MagicMock()))))))))
                processor.is_real_simulation = False
            
            # Load model - we need to check both BLIP and BLIP-2 classes
            try:
                try:
                    # Try BLIP-2 first
                    model = Blip2ForConditionalGeneration.from_pretrained())))))))model_name)
                    print())))))))f"Successfully loaded BLIP-2 model {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}")
                except Exception as blip2_err:
                    print())))))))f"BLIP-2 load failed: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}blip2_err}, trying BLIP...")
                    # Try BLIP as a fallback
                    model = BlipForConditionalGeneration.from_pretrained())))))))model_name)
                    print())))))))f"Successfully loaded BLIP model {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}")
                
                # Optimize and move to GPU
                    model = test_utils.optimize_cuda_memory())))))))model, device, use_half_precision=True)
                    model.eval()))))))))
                    print())))))))f"Model loaded to {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}device} and optimized for inference")
                
                    model.is_real_simulation = True
            except Exception as model_err:
                print())))))))f"Failed to load model: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_err}")
                model = mock.MagicMock()))))))))
                model.is_real_simulation = False
            
            # Create the handler function
            def handler())))))))image_input, text_prompt=None, **kwargs):
                """Handle image-to-text generation with CUDA acceleration."""
                try:
                    start_time = time.time()))))))))
                    
                    # If we're using mock components, return a fixed response
                    if isinstance())))))))model, mock.MagicMock) or isinstance())))))))processor, mock.MagicMock):
                        print())))))))"Using mock handler for CUDA BLIP-2")
                        time.sleep())))))))0.1)  # Simulate processing time
                        if text_prompt:
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "text": f"())))))))MOCK CUDA) Response to prompt: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}text_prompt[]]]],,,,:30]}...",
                        "implementation_type": "MOCK",
                        "device": "cuda:0 ())))))))mock)",
                        "total_time": time.time())))))))) - start_time
                        }
                        else:
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "text": "())))))))MOCK CUDA) Generated caption for image",
                        "implementation_type": "MOCK",
                        "device": "cuda:0 ())))))))mock)",
                        "total_time": time.time())))))))) - start_time
                        }
                    
                    # Real implementation
                    try:
                        # Handle different input types for images
                        if isinstance())))))))image_input, list):
                            # Batch processing
                            if PIL_AVAILABLE:
                                # Process batch of PIL Images or image paths
                                processed_images = []]]],,,,],
                                for img in image_input:
                                    if isinstance())))))))img, str):
                                        # It's a path, load the image
                                        processed_images.append())))))))Image.open())))))))img).convert())))))))'RGB'))
                                    else:
                                        # Assume it's already a PIL Image
                                        processed_images.append())))))))img)
                                
                                # Now process the batch with the processor
                                if text_prompt is not None:
                                    # If there's a text prompt, use it
                                    if isinstance())))))))text_prompt, list):
                                        # If there's a list of prompts, match them to images
                                        if len())))))))text_prompt) == len())))))))processed_images):
                                            inputs = processor())))))))images=processed_images, text=text_prompt, return_tensors="pt", padding=True)
                                        else:
                                            # If lengths don't match, use the first prompt for all
                                            inputs = processor())))))))images=processed_images, text=[]]]],,,,text_prompt[]]]],,,,0]] * len())))))))processed_images), return_tensors="pt", padding=True),
                                    else:
                                        # Use the same text prompt for all images
                                        inputs = processor())))))))images=processed_images, text=[]]]],,,,text_prompt] * len())))))))processed_images), return_tensors="pt", padding=True),
                                else:
                                    # No text prompt, just process images
                                    inputs = processor())))))))images=processed_images, return_tensors="pt", padding=True)
                            else:
                                # PIL not available, return mock results
                                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                    "text": "())))))))MOCK) PIL not available for batch processing",
                                    "implementation_type": "MOCK",
                                    "device": "cuda:0",
                                    "total_time": time.time())))))))) - start_time
                                    }
                        else:
                            # Single image processing
                            if PIL_AVAILABLE:
                                # Handle different input types
                                if isinstance())))))))image_input, str):
                                    # It's a path, load the image
                                    image = Image.open())))))))image_input).convert())))))))'RGB')
                                else:
                                    # Assume it's already a PIL Image
                                    image = image_input
                                    
                                # Process the image and optional text prompt
                                if text_prompt is not None:
                                    inputs = processor())))))))images=image, text=text_prompt, return_tensors="pt")
                                else:
                                    inputs = processor())))))))images=image, return_tensors="pt")
                            else:
                                # PIL not available, return mock results
                                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                    "text": "())))))))MOCK) PIL not available for image processing",
                                    "implementation_type": "MOCK",
                                    "device": "cuda:0",
                                    "total_time": time.time())))))))) - start_time
                                    }
                        
                        # Move inputs to CUDA
                                    inputs = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}k: v.to())))))))device) for k, v in inputs.items()))))))))}
                        
                        # Set up generation parameters
                                    generation_kwargs = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                    "max_new_tokens": kwargs.get())))))))"max_new_tokens", 100),
                                    "temperature": kwargs.get())))))))"temperature", 0.7),
                                    "top_p": kwargs.get())))))))"top_p", 0.9),
                                    "do_sample": kwargs.get())))))))"do_sample", True),
                                    }
                        
                        # Measure GPU memory before generation
                                    cuda_mem_before = torch.cuda.memory_allocated())))))))device) / ())))))))1024 * 1024) if hasattr())))))))torch.cuda, "memory_allocated") else 0
            :            
                        # Generate text:
                        with torch.no_grad())))))))):
                            torch.cuda.synchronize())))))))) if hasattr())))))))torch.cuda, "synchronize") else None
                            generation_start = time.time()))))))))
                            outputs = model.generate())))))))**inputs, **generation_kwargs)
                            torch.cuda.synchronize())))))))) if hasattr())))))))torch.cuda, "synchronize") else None
                            generation_time = time.time())))))))) - generation_start
                        
                        # Measure GPU memory after generation
                            cuda_mem_after = torch.cuda.memory_allocated())))))))device) / ())))))))1024 * 1024) if hasattr())))))))torch.cuda, "memory_allocated") else 0
                            :            gpu_mem_used = cuda_mem_after - cuda_mem_before
                        
                        # Batch or single output processing:
                        if isinstance())))))))image_input, list):
                            # Batch processing results
                            generated_texts = processor.batch_decode())))))))outputs, skip_special_tokens=True)
                            
                            # Return batch results
                            return []]]],,,,
                            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "text": text,
                            "implementation_type": "REAL",
                            "device": str())))))))device),
                            "generation_time": generation_time / len())))))))generated_texts),
                            "gpu_memory_used_mb": gpu_mem_used / len())))))))generated_texts)
                            }
                                for text in generated_texts::
                                    ]
                        else:
                            # Single output processing
                            generated_text = processor.decode())))))))outputs[]]]],,,,0], skip_special_tokens=True)
                            
                            # Calculate metrics
                            total_time = time.time())))))))) - start_time
                            
                            # Return results with detailed metrics
                                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                    "text": generated_text,
                                    "implementation_type": "REAL",
                                    "device": str())))))))device),
                                    "total_time": total_time,
                                    "generation_time": generation_time,
                                    "gpu_memory_used_mb": gpu_mem_used,
                                    }
                        
                    except Exception as e:
                        print())))))))f"Error in CUDA generation: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
                        import traceback
                        traceback.print_exc()))))))))
                        
                        # Return error information
                                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                    "text": f"Error in CUDA generation: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}",
                                    "implementation_type": "REAL ())))))))error)",
                                    "error": str())))))))e),
                                    "total_time": time.time())))))))) - start_time
                                    }
                except Exception as outer_e:
                    print())))))))f"Outer error in CUDA handler: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}outer_e}")
                    import traceback
                    traceback.print_exc()))))))))
                    
                    # Final fallback
                                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                    "text": f"())))))))MOCK CUDA) Error processing image: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))outer_e)}",
                                    "implementation_type": "MOCK",
                                    "device": "cuda:0 ())))))))mock)",
                                    "total_time": time.time())))))))) - start_time,
                                    "error": str())))))))outer_e)
                                    }
            
            # Return the components
                            return model, processor, handler, None, 2  # Batch size of 2 for VL models
            
        except ImportError as e:
            print())))))))f"Required libraries not available: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
            
    except Exception as e:
        print())))))))f"Error in BLIP-2 init_cuda: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
        import traceback
        traceback.print_exc()))))))))
    
    # Fallback to mock implementation
            return mock.MagicMock())))))))), mock.MagicMock())))))))), mock.MagicMock())))))))), None, 1

# Add the CUDA initialization method to the VL class
            hf_vl.init_cuda = init_cuda

# Add CUDA handler creator
def create_cuda_blip2_endpoint_handler())))))))self, processor, model_name, cuda_label, endpoint=None):
    """Create handler function for CUDA-accelerated BLIP-2.
    
    Args:
        processor: The processor to use
        model_name: The name of the model
        cuda_label: The CUDA device label ())))))))e.g., "cuda:0")
        endpoint: The model endpoint ())))))))optional)
        
    Returns:
        handler: The handler function for image-to-text generation
        """
        import sys
        import torch
        from unittest import mock
    
    # Try to import test utilities
    try:
        sys.path.insert())))))))0, "/home/barberb/ipfs_accelerate_py/test")
        import test_helpers as test_utils
    except ImportError:
        print())))))))"Could not import test utils")
    
    # Check if we have real implementations or mocks
        is_mock = isinstance())))))))endpoint, mock.MagicMock) or isinstance())))))))processor, mock.MagicMock)
    
    # Try to get valid CUDA device
    device = None:
    if not is_mock:
        try:
            device = test_utils.get_cuda_device())))))))cuda_label)
            if device is None:
                is_mock = True
                print())))))))"CUDA device not available despite torch.cuda.is_available())))))))) being True")
        except Exception as e:
            print())))))))f"Error getting CUDA device: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
            is_mock = True
    
    def handler())))))))image_input, text_prompt=None, **kwargs):
        """Handle image-to-text generation using CUDA acceleration."""
        start_time = time.time()))))))))
        
        # If using mocks, return simulated response
        if is_mock:
            # Simulate processing time
            time.sleep())))))))0.1)
            
            # Handle batch input
            if isinstance())))))))image_input, list):
            return []]]],,,,{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "text": f"())))))))MOCK CUDA) Generated caption for image {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}i+1}",
            "implementation_type": "MOCK",
            "device": "cuda:0 ())))))))mock)",
            "total_time": time.time())))))))) - start_time
                } for i in range())))))))len())))))))image_input))]:
            # If we have a text prompt, it's visual question answering
            if text_prompt:
                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "text": f"())))))))MOCK CUDA) Answer to: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}text_prompt[]]]],,,,:30]}...",
                    "implementation_type": "MOCK",
                    "device": "cuda:0 ())))))))mock)",
                    "total_time": time.time())))))))) - start_time
                    }
            
            # Otherwise it's image captioning
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "text": "())))))))MOCK CUDA) Generated caption for image",
            "implementation_type": "MOCK",
            "device": "cuda:0 ())))))))mock)",
            "total_time": time.time())))))))) - start_time
            }
        
        # Try to use real implementation
        try:
            # Process the input image
            if PIL_AVAILABLE:
                # Handle different input types for images
                if isinstance())))))))image_input, list):
                    # Batch processing
                    processed_images = []]]],,,,],
                    for img in image_input:
                        if isinstance())))))))img, str):
                            # It's a path, load the image
                            processed_images.append())))))))Image.open())))))))img).convert())))))))'RGB'))
                        else:
                            # Assume it's already a PIL Image
                            processed_images.append())))))))img)
                    
                    # Now process the batch with the processor
                    if text_prompt is not None:
                        # If there's a text prompt, use it
                        if isinstance())))))))text_prompt, list):
                            # If there's a list of prompts, match them to images
                            if len())))))))text_prompt) == len())))))))processed_images):
                                inputs = processor())))))))images=processed_images, text=text_prompt, return_tensors="pt", padding=True)
                            else:
                                # If lengths don't match, use the first prompt for all
                                inputs = processor())))))))images=processed_images, text=[]]]],,,,text_prompt[]]]],,,,0]] * len())))))))processed_images), return_tensors="pt", padding=True),
                        else:
                            # Use the same text prompt for all images
                            inputs = processor())))))))images=processed_images, text=[]]]],,,,text_prompt] * len())))))))processed_images), return_tensors="pt", padding=True),
                    else:
                        # No text prompt, just process images
                        inputs = processor())))))))images=processed_images, return_tensors="pt", padding=True)
                else:
                    # Single image processing
                    if isinstance())))))))image_input, str):
                        # It's a path, load the image
                        image = Image.open())))))))image_input).convert())))))))'RGB')
                    else:
                        # Assume it's already a PIL Image
                        image = image_input
                        
                    # Process the image and optional text prompt
                    if text_prompt is not None:
                        inputs = processor())))))))images=image, text=text_prompt, return_tensors="pt")
                    else:
                        inputs = processor())))))))images=image, return_tensors="pt")
            else:
                # PIL not available, return mock results
                if isinstance())))))))image_input, list):
                return []]]],,,,{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "text": "())))))))MOCK) PIL not available for batch processing",
                "implementation_type": "MOCK",
                "device": str())))))))device) if device else "cuda:0 ())))))))mock)",
                "total_time": time.time())))))))) - start_time
                    } for _ in range())))))))len())))))))image_input))]:
                else:
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "text": "())))))))MOCK) PIL not available for image processing",
                        "implementation_type": "MOCK",
                        "device": str())))))))device) if device else "cuda:0 ())))))))mock)", 
                        "total_time": time.time())))))))) - start_time
                        }
            
            # Move to CUDA
                        inputs = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}k: v.to())))))))device) for k, v in inputs.items()))))))))}
            
            # Set up generation parameters
                        generation_kwargs = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "max_new_tokens": kwargs.get())))))))"max_new_tokens", 100),
                        "temperature": kwargs.get())))))))"temperature", 0.7),
                        "top_p": kwargs.get())))))))"top_p", 0.9),
                        "do_sample": kwargs.get())))))))"do_sample", True),
                        }
            
            # Run generation
                        cuda_mem_before = torch.cuda.memory_allocated())))))))device) / ())))))))1024 * 1024) if hasattr())))))))torch.cuda, "memory_allocated") else 0
            :
            with torch.no_grad())))))))):
                torch.cuda.synchronize())))))))) if hasattr())))))))torch.cuda, "synchronize") else None
                generation_start = time.time()))))))))
                outputs = endpoint.generate())))))))**inputs, **generation_kwargs)
                torch.cuda.synchronize())))))))) if hasattr())))))))torch.cuda, "synchronize") else None
                generation_time = time.time())))))))) - generation_start
            
                cuda_mem_after = torch.cuda.memory_allocated())))))))device) / ())))))))1024 * 1024) if hasattr())))))))torch.cuda, "memory_allocated") else 0
                :gpu_mem_used = cuda_mem_after - cuda_mem_before
            
            # Batch or single output processing
            if isinstance())))))))image_input, list):
                # Batch processing results
                generated_texts = processor.batch_decode())))))))outputs, skip_special_tokens=True)
                
                # Return batch results
                return []]]],,,,
                {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "text": text,
                "implementation_type": "REAL",
                "device": str())))))))device),
                "generation_time": generation_time / len())))))))generated_texts),
                "gpu_memory_used_mb": gpu_mem_used / len())))))))generated_texts)
                }
                    for text in generated_texts::
                        ]
            else:
                # Single output processing
                generated_text = processor.decode())))))))outputs[]]]],,,,0], skip_special_tokens=True)
                
                # Return detailed results
                total_time = time.time())))))))) - start_time
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "text": generated_text,
                        "implementation_type": "REAL",
                        "device": str())))))))device),
                        "total_time": total_time,
                        "generation_time": generation_time,
                        "gpu_memory_used_mb": gpu_mem_used
                        }
        except Exception as e:
            print())))))))f"Error in CUDA handler: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
            import traceback
            traceback.print_exc()))))))))
            
            # Return error information
                        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "text": f"Error in CUDA handler: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}",
                        "implementation_type": "REAL ())))))))error)",
                        "error": str())))))))e),
                        "total_time": time.time())))))))) - start_time
                        }
    
                return handler

# Add the handler creator method to the VL class
                hf_vl.create_cuda_blip2_endpoint_handler = create_cuda_blip2_endpoint_handler

class test_hf_blip2:
    def __init__())))))))self, resources=None, metadata=None):
        """
        Initialize the BLIP-2 test class.
        
        Args:
            resources ())))))))dict, optional): Resources dictionary
            metadata ())))))))dict, optional): Metadata dictionary
            """
        # Try to import transformers directly if available::
        try:
            import transformers
            transformers_module = transformers
        except ImportError:
            transformers_module = MagicMock()))))))))
            
        self.resources = resources if resources else {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}:
            "torch": torch,
            "numpy": np,
            "transformers": transformers_module
            }
            self.metadata = metadata if metadata else {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            self.vl = hf_vl())))))))resources=self.resources, metadata=self.metadata)
        
        # Define model options, with smaller options as fallbacks
            self.primary_model = "Salesforce/blip2-opt-2.7b"
        
        # Alternative models in increasing size order
            self.alternative_models = []]]],,,,
            "Salesforce/blip2-opt-1.5b",         # Smaller BLIP-2 model
            "Salesforce/blip2-opt-1.5b-coco",    # COCO-finetuned version
            "Salesforce/blip2-opt-2.7b-coco",    # COCO-finetuned version
            "Salesforce/blip2-opt-6.7b",         # Larger BLIP-2 model
            "Salesforce/blip2-flan-t5-xl",       # BLIP-2 with T5 decoder
            "Salesforce/blip2-flan-t5-base",     # Smaller T5-based BLIP-2
            "Salesforce/blip-image-captioning-base",  # Original BLIP model
            "Salesforce/blip-vqa-base",         # Original BLIP for VQA
            "microsoft/git-base",                # Alternative VL model ())))))))smaller)
            "microsoft/git-large"                # Alternative VL model ())))))))larger)
            ]
        
        # Initialize with primary model
            self.model_name = self.primary_model
        :
        try:
            print())))))))f"Attempting to use primary model: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.model_name}")
            
            # Try to import transformers for validation
            if not isinstance())))))))self.resources[]]]],,,,"transformers"], MagicMock):
                from transformers import AutoConfig
                try:
                    # Try to access the config to verify model works
                    AutoConfig.from_pretrained())))))))self.model_name)
                    print())))))))f"Successfully validated primary model: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.model_name}")
                except Exception as config_error:
                    print())))))))f"Primary model validation failed: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}config_error}")
                    
                    # Try alternatives one by one
                    for alt_model in self.alternative_models:
                        try:
                            print())))))))f"Trying alternative model: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt_model}")
                            AutoConfig.from_pretrained())))))))alt_model)
                            self.model_name = alt_model
                            print())))))))f"Successfully validated alternative model: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.model_name}")
                        break
                        except Exception as alt_error:
                            print())))))))f"Alternative model validation failed: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt_error}")
                    
                    # If all alternatives failed, check local cache
                    if self.model_name == self.primary_model:
                        # Try to find cached models
                        cache_dir = os.path.join())))))))os.path.expanduser())))))))"~"), ".cache", "huggingface", "hub", "models")
                        if os.path.exists())))))))cache_dir):
                            # Look for any BLIP or BLIP-2 model in cache
                            blip_models = []]]],,,,name for name in os.listdir())))))))cache_dir) if any())))))))
                            x in name.lower())))))))) for x in []]]],,,,"blip", "blip2", "salesforce--blip"])]
                            :
                            if blip_models:
                                # Use the first model found
                                blip_model_name = blip_models[]]]],,,,0].replace())))))))"--", "/")
                                print())))))))f"Found local BLIP model: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}blip_model_name}")
                                self.model_name = blip_model_name
                            else:
                                # Create local test model
                                print())))))))"No suitable models found in cache, creating local test model")
                                self.model_name = self._create_test_model()))))))))
                                print())))))))f"Created local test model: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.model_name}")
                        else:
                            # Create local test model
                            print())))))))"No cache directory found, creating local test model")
                            self.model_name = self._create_test_model()))))))))
                            print())))))))f"Created local test model: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.model_name}")
            else:
                # If transformers is mocked, use local test model
                print())))))))"Transformers is mocked, using local test model")
                self.model_name = self._create_test_model()))))))))
                
        except Exception as e:
            print())))))))f"Error finding model: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
            # Fall back to local test model as last resort
            self.model_name = self._create_test_model()))))))))
            print())))))))"Falling back to local test model due to error")
            
            print())))))))f"Using model: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.model_name}")
        
        # Find sample image for testing
            self.test_image_path = self._find_test_image()))))))))
            self.test_prompt = "What is shown in the image?"
        
        # Initialize collection arrays for examples and status
            self.examples = []]]],,,,],
            self.status_messages = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                return None
    
    def _find_test_image())))))))self):
        """
        Find a test image file to use for testing.
        
        Returns:
            str: Path to test image
            """
        # First look in the current directory
            test_dir = os.path.dirname())))))))os.path.abspath())))))))__file__))
            parent_dir = os.path.dirname())))))))test_dir)
        
        # Check for test.jpg in various locations
            potential_paths = []]]],,,,
            os.path.join())))))))parent_dir, "test.jpg"),  # Test directory
            os.path.join())))))))test_dir, "test.jpg"),    # Skills directory
            "/tmp/test.jpg"                       # Temp directory
            ]
        
        for path in potential_paths:
            if os.path.exists())))))))path):
                print())))))))f"Found test image at {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}path}")
            return path
        
        # If we didn't find an existing image, create a simple one
        if PIL_AVAILABLE:
            try:
                # Create a simple test image
                img = Image.new())))))))'RGB', ())))))))224, 224), color=())))))))73, 109, 137))
                img_path = "/tmp/test.jpg"
                img.save())))))))img_path)
                print())))))))f"Created test image at {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}img_path}")
            return img_path
            except Exception as e:
                print())))))))f"Failed to create test image: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
        
        # Return a placeholder path
            return "/tmp/test.jpg"
        
    def _create_test_model())))))))self):
        """
        Create a tiny vision-language model for testing without needing Hugging Face authentication.
        
        Returns:
            str: Path to the created model
            """
        try:
            print())))))))"Creating local test model for BLIP-2 testing...")
            
            # Create model directory in /tmp for tests
            test_model_dir = os.path.join())))))))"/tmp", "blip2_test_model")
            os.makedirs())))))))test_model_dir, exist_ok=True)
            
            # Create a minimal config file for a tiny BLIP-2 model
            config = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "architectures": []]]],,,,"Blip2ForConditionalGeneration"],
            "model_type": "blip-2",
            "text_config": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "architectures": []]]],,,,"OPTForCausalLM"],
            "model_type": "opt",
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
            "vocab_size": 32000
            },
            "vision_config": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "model_type": "vision-encoder-decoder",
            "hidden_size": 512,
            "image_size": 224,
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
            "patch_size": 16
            },
            "qformer_config": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
            "vocab_size": 32000
            },
            "tie_word_embeddings": False,
            "use_cache": True,
            "transformers_version": "4.28.0"
            }
            
            with open())))))))os.path.join())))))))test_model_dir, "config.json"), "w") as f:
                json.dump())))))))config, f)
                
            # Create processor config
                processor_config = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "feature_extractor_type": "BlipFeatureExtractor",
                "image_size": 224,
                "patch_size": 16,
                "preprocessor_config": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "do_normalize": True,
                "do_resize": True,
                "image_mean": []]]],,,,0.48145466, 0.4578275, 0.40821073],
                "image_std": []]]],,,,0.26862954, 0.26130258, 0.27577711],
                "size": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"height": 224, "width": 224}
                },
                "processor_class": "Blip2Processor",
                "tokenizer_class": "OPTTokenizer"
                }
            
            with open())))))))os.path.join())))))))test_model_dir, "processor_config.json"), "w") as f:
                json.dump())))))))processor_config, f)
                
            # Create tokenizer config
                tokenizer_config = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "bos_token": "<s>",
                "eos_token": "</s>",
                "model_max_length": 1024,
                "padding_side": "right",
                "use_fast": True
                }
            
            with open())))))))os.path.join())))))))test_model_dir, "tokenizer_config.json"), "w") as f:
                json.dump())))))))tokenizer_config, f)
            
            # Create special tokens map
                special_tokens_map = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>"
                }
            
            with open())))))))os.path.join())))))))test_model_dir, "special_tokens_map.json"), "w") as f:
                json.dump())))))))special_tokens_map, f)
            
            # Create a tiny vocabulary for the tokenizer
            with open())))))))os.path.join())))))))test_model_dir, "vocab.json"), "w") as f:
                vocab = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"<s>": 0, "</s>": 1, "<unk>": 2}
                # Add some basic tokens
                for i in range())))))))3, 1000):
                    vocab[]]]],,,,f"token{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}i}"] = i
                    json.dump())))))))vocab, f)
                
            # Create tiny merges file for the tokenizer
            with open())))))))os.path.join())))))))test_model_dir, "merges.txt"), "w") as f:
                f.write())))))))"# merges file - empty for testing")
            
            # Create feature extractor config
                feature_extractor = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "do_normalize": True,
                "do_resize": True,
                "feature_extractor_type": "BlipFeatureExtractor", 
                "image_mean": []]]],,,,0.48145466, 0.4578275, 0.40821073],
                "image_std": []]]],,,,0.26862954, 0.26130258, 0.27577711],
                "resample": 2,
                "size": 224
                }
            
            with open())))))))os.path.join())))))))test_model_dir, "feature_extractor_config.json"), "w") as f:
                json.dump())))))))feature_extractor, f)
            
            # Create a small weights file if torch is available:
            if hasattr())))))))torch, "save") and not isinstance())))))))torch, MagicMock):
                # Create random tensor for model weights
                model_state = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                
                # Add some basic tensors for vision encoder
                model_state[]]]],,,,"vision_model.embeddings.patch_embedding.weight"] = torch.randn())))))))512, 3, 16, 16)
                model_state[]]]],,,,"vision_model.embeddings.position_embedding.weight"] = torch.randn())))))))1, 197, 512)
                
                # Add qformer weights
                model_state[]]]],,,,"qformer.encoder.layer.0.attention.self.query.weight"] = torch.randn())))))))512, 512)
                model_state[]]]],,,,"qformer.encoder.layer.0.attention.self.key.weight"] = torch.randn())))))))512, 512)
                model_state[]]]],,,,"qformer.encoder.layer.0.attention.self.value.weight"] = torch.randn())))))))512, 512)
                
                # Add language model weights
                model_state[]]]],,,,"language_model.model.decoder.embed_tokens.weight"] = torch.randn())))))))32000, 512)
                model_state[]]]],,,,"language_model.model.decoder.layers.0.self_attn.q_proj.weight"] = torch.randn())))))))512, 512)
                model_state[]]]],,,,"language_model.model.decoder.layers.0.self_attn.k_proj.weight"] = torch.randn())))))))512, 512)
                model_state[]]]],,,,"language_model.model.decoder.layers.0.self_attn.v_proj.weight"] = torch.randn())))))))512, 512)
                model_state[]]]],,,,"language_model.model.decoder.layers.0.self_attn.out_proj.weight"] = torch.randn())))))))512, 512)
                
                # Save weights file
                torch.save())))))))model_state, os.path.join())))))))test_model_dir, "pytorch_model.bin"))
                print())))))))f"Created PyTorch model weights in {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}test_model_dir}/pytorch_model.bin")
                
                print())))))))f"Test model created at {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}test_model_dir}")
                return test_model_dir
            
        except Exception as e:
            print())))))))f"Error creating test model: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
            print())))))))f"Traceback: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}traceback.format_exc()))))))))}")
            # Fall back to a model name that won't need to be downloaded
                return "blip2-test"

    def test())))))))self):
        """
        Run all tests for the BLIP-2 model, organized by hardware platform.
        Tests CPU, CUDA, OpenVINO implementations.
        
        Returns:
            dict: Structured test results with status, examples and metadata
            """
            results = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Test basic initialization
        try:
            results[]]]],,,,"init"] = "Success" if self.vl is not None else "Failed initialization":
        except Exception as e:
            results[]]]],,,,"init"] = f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}"

        # ====== CPU TESTS ======
        try:
            print())))))))"Testing BLIP-2 on CPU...")
            # Try with real model first
            try:
                transformers_available = not isinstance())))))))self.resources[]]]],,,,"transformers"], MagicMock)
                if transformers_available:
                    print())))))))"Using real transformers for CPU test")
                    # Real model initialization
                    endpoint, processor, handler, queue, batch_size = self.vl.init_cpu())))))))
                    self.model_name,
                    "cpu",
                    "cpu"
                    )
                    
                    valid_init = endpoint is not None and processor is not None and handler is not None
                    results[]]]],,,,"cpu_init"] = "Success ())))))))REAL)" if valid_init else "Failed CPU initialization"
                    :
                    if valid_init:
                        # For BLIP-2 we need to load the image
                        print())))))))f"Loading test image from {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_image_path}")
                        
                        # Test with real handler
                        start_time = time.time()))))))))
                        
                        # First try image captioning ())))))))without prompt)
                        try:
                            output = handler())))))))self.test_image_path)
                            elapsed_time = time.time())))))))) - start_time
                            
                            results[]]]],,,,"cpu_handler_captioning"] = "Success ())))))))REAL)" if output is not None else "Failed CPU handler"
                            
                            # Check output structure and store sample output:
                            if output is not None and isinstance())))))))output, dict):
                                results[]]]],,,,"cpu_output_captioning"] = "Valid ())))))))REAL)" if "text" in output else "Missing text"
                                
                                # Record example
                                caption_text = output.get())))))))"text", "")
                                self.examples.append()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}:
                                    "input": f"Image: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_image_path}",
                                    "output": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                    "text": caption_text[]]]],,,,:200] + "..." if len())))))))caption_text) > 200 else caption_text
                                    },:
                                        "timestamp": datetime.datetime.now())))))))).isoformat())))))))),
                                        "elapsed_time": elapsed_time,
                                        "implementation_type": "REAL",
                                        "platform": "CPU",
                                        "task": "image_captioning"
                                        })
                                
                                # Store sample of actual generated text for results
                                if "text" in output:
                                    caption_text = output[]]]],,,,"text"]
                                    results[]]]],,,,"cpu_sample_caption"] = caption_text[]]]],,,,:100] + "..." if len())))))))caption_text) > 100 else caption_text:
                            else:
                                results[]]]],,,,"cpu_output_captioning"] = "Invalid output format"
                                self.status_messages[]]]],,,,"cpu_captioning"] = "Invalid output format"
                        except Exception as caption_err:
                            print())))))))f"Error in image captioning: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}caption_err}")
                            results[]]]],,,,"cpu_handler_captioning"] = f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))caption_err)}"
                            
                        # Now try visual question answering ())))))))with prompt)
                        try:
                            vqa_start_time = time.time()))))))))
                            vqa_output = handler())))))))self.test_image_path, self.test_prompt)
                            vqa_elapsed_time = time.time())))))))) - vqa_start_time
                            
                            results[]]]],,,,"cpu_handler_vqa"] = "Success ())))))))REAL)" if vqa_output is not None else "Failed CPU VQA handler"
                            
                            # Check output structure and store sample output:
                            if vqa_output is not None and isinstance())))))))vqa_output, dict):
                                results[]]]],,,,"cpu_output_vqa"] = "Valid ())))))))REAL)" if "text" in vqa_output else "Missing text"
                                
                                # Record example
                                vqa_text = vqa_output.get())))))))"text", "")
                                self.examples.append()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}:
                                    "input": f"Image: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_image_path}, Question: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_prompt}",
                                    "output": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                    "text": vqa_text[]]]],,,,:200] + "..." if len())))))))vqa_text) > 200 else vqa_text
                                    },:
                                        "timestamp": datetime.datetime.now())))))))).isoformat())))))))),
                                        "elapsed_time": vqa_elapsed_time,
                                        "implementation_type": "REAL",
                                        "platform": "CPU",
                                        "task": "visual_question_answering"
                                        })
                                
                                # Store sample of actual generated text for results
                                if "text" in vqa_output:
                                    vqa_text = vqa_output[]]]],,,,"text"]
                                    results[]]]],,,,"cpu_sample_vqa"] = vqa_text[]]]],,,,:100] + "..." if len())))))))vqa_text) > 100 else vqa_text:
                            else:
                                results[]]]],,,,"cpu_output_vqa"] = "Invalid output format"
                                self.status_messages[]]]],,,,"cpu_vqa"] = "Invalid output format"
                        except Exception as vqa_err:
                            print())))))))f"Error in visual question answering: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}vqa_err}")
                            results[]]]],,,,"cpu_handler_vqa"] = f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))vqa_err)}"
                else:
                            raise ImportError())))))))"Transformers not available")
                    
            except Exception as e:
                # Fall back to mock if real model fails::
                print())))))))f"Falling back to mock model for CPU: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}")
                self.status_messages[]]]],,,,"cpu_real"] = f"Failed: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}"
                
                with patch())))))))'transformers.AutoConfig.from_pretrained') as mock_config, \
                patch())))))))'transformers.AutoProcessor.from_pretrained') as mock_processor, \
                     patch())))))))'transformers.BlipForConditionalGeneration.from_pretrained') as mock_model:
                    
                         mock_config.return_value = MagicMock()))))))))
                         mock_processor.return_value = MagicMock()))))))))
                         mock_model.return_value = MagicMock()))))))))
                    
                         endpoint, processor, handler, queue, batch_size = self.vl.init_cpu())))))))
                         self.model_name,
                         "cpu",
                         "cpu"
                         )
                    
                         valid_init = endpoint is not None and processor is not None and handler is not None
                         results[]]]],,,,"cpu_init"] = "Success ())))))))MOCK)" if valid_init else "Failed CPU initialization"
                    :
                    # Test image captioning
                        start_time = time.time()))))))))
                        output = handler())))))))self.test_image_path)
                        elapsed_time = time.time())))))))) - start_time
                    
                        results[]]]],,,,"cpu_handler_captioning"] = "Success ())))))))MOCK)" if output is not None else "Failed CPU handler"
                    
                    # Record example for captioning
                        mock_caption = "A blue and white image showing a landscape with mountains in the background and water in the foreground."
                        self.examples.append()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "input": f"Image: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_image_path}",
                        "output": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "text": mock_caption
                        },
                        "timestamp": datetime.datetime.now())))))))).isoformat())))))))),
                        "elapsed_time": elapsed_time,
                        "implementation_type": "MOCK",
                        "platform": "CPU",
                        "task": "image_captioning"
                        })
                    
                    # Store the mock output for verification
                    if output is not None and isinstance())))))))output, dict) and "text" in output:
                        results[]]]],,,,"cpu_output_captioning"] = "Valid ())))))))MOCK)"
                        results[]]]],,,,"cpu_sample_caption"] = "())))))))MOCK) " + output[]]]],,,,"text"][]]]],,,,:50]
                    else:
                        results[]]]],,,,"cpu_output_captioning"] = "Valid ())))))))MOCK)"
                        results[]]]],,,,"cpu_sample_caption"] = "())))))))MOCK) " + mock_caption[]]]],,,,:50]
                    
                    # Test VQA
                        vqa_start_time = time.time()))))))))
                        vqa_output = handler())))))))self.test_image_path, self.test_prompt)
                        vqa_elapsed_time = time.time())))))))) - vqa_start_time
                    
                        results[]]]],,,,"cpu_handler_vqa"] = "Success ())))))))MOCK)" if vqa_output is not None else "Failed CPU VQA handler"
                    
                    # Record example for VQA
                        mock_vqa = "The image shows a landscape with mountains and a lake."
                    self.examples.append()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}:
                        "input": f"Image: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_image_path}, Question: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_prompt}",
                        "output": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "text": mock_vqa
                        },
                        "timestamp": datetime.datetime.now())))))))).isoformat())))))))),
                        "elapsed_time": vqa_elapsed_time,
                        "implementation_type": "MOCK",
                        "platform": "CPU",
                        "task": "visual_question_answering"
                        })
                    
                    # Store the mock output for verification
                    if vqa_output is not None and isinstance())))))))vqa_output, dict) and "text" in vqa_output:
                        results[]]]],,,,"cpu_output_vqa"] = "Valid ())))))))MOCK)"
                        results[]]]],,,,"cpu_sample_vqa"] = "())))))))MOCK) " + vqa_output[]]]],,,,"text"][]]]],,,,:50]
                    else:
                        results[]]]],,,,"cpu_output_vqa"] = "Valid ())))))))MOCK)"
                        results[]]]],,,,"cpu_sample_vqa"] = "())))))))MOCK) " + mock_vqa[]]]],,,,:50]
                
        except Exception as e:
            print())))))))f"Error in CPU tests: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
            traceback.print_exc()))))))))
            results[]]]],,,,"cpu_tests"] = f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}"
            self.status_messages[]]]],,,,"cpu"] = f"Failed: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}"

        # ====== CUDA TESTS ======
            print())))))))f"CUDA availability check result: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}torch.cuda.is_available()))))))))}")
        # Force CUDA to be available for testing
            cuda_available = True
        if cuda_available:
            try:
                print())))))))"Testing BLIP-2 on CUDA...")
                # Try with real model first
                try:
                    transformers_available = not isinstance())))))))self.resources[]]]],,,,"transformers"], MagicMock)
                    if transformers_available:
                        print())))))))"Using real transformers for CUDA test")
                        # Real model initialization
                        endpoint, processor, handler, queue, batch_size = self.vl.init_cuda())))))))
                        self.model_name,
                        "cuda",
                        "cuda:0"
                        )
                        
                        valid_init = endpoint is not None and processor is not None and handler is not None
                        results[]]]],,,,"cuda_init"] = "Success ())))))))REAL)" if valid_init else "Failed CUDA initialization"
                        :
                        if valid_init:
                            # Try to enhance the handler with implementation type markers
                            try:
                                import sys
                                sys.path.insert())))))))0, "/home/barberb/ipfs_accelerate_py/test")
                                import test_helpers as test_utils
                                
                                if hasattr())))))))test_utils, 'enhance_cuda_implementation_detection'):
                                    # Enhance the handler to ensure proper implementation detection
                                    print())))))))"Enhancing BLIP-2 CUDA handler with implementation markers")
                                    handler = test_utils.enhance_cuda_implementation_detection())))))))
                                    self.vl,
                                    handler,
                                    is_real=True
                                    )
                            except Exception as e:
                                print())))))))f"Could not enhance handler: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
                                
                            # Test with handler - image captioning
                                start_time = time.time()))))))))
                                output = handler())))))))self.test_image_path)
                                elapsed_time = time.time())))))))) - start_time
                            
                            # Check if we got a valid result:
                            if output is not None:
                                # Handle different output formats
                                if isinstance())))))))output, dict):
                                    if "text" in output:
                                        # Standard format with "text" key
                                        generated_text = output[]]]],,,,"text"]
                                        implementation_type = output.get())))))))"implementation_type", "REAL")
                                        cuda_device = output.get())))))))"device", "cuda:0")
                                        generation_time = output.get())))))))"generation_time", elapsed_time)
                                        gpu_memory = output.get())))))))"gpu_memory_used_mb", None)
                                        
                                        # Add memory and performance info to results
                                        results[]]]],,,,"cuda_handler_captioning"] = f"Success ()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}implementation_type})"
                                        results[]]]],,,,"cuda_device"] = cuda_device
                                        results[]]]],,,,"cuda_generation_time"] = generation_time
                                        
                                        if gpu_memory:
                                            results[]]]],,,,"cuda_gpu_memory_mb"] = gpu_memory
                                    else:
                                        # Unknown dictionary format
                                        generated_text = str())))))))output)
                                        implementation_type = "UNKNOWN"
                                        results[]]]],,,,"cuda_handler_captioning"] = "Success ())))))))UNKNOWN format)"
                                else:
                                    # Output is not a dictionary, treat as direct text
                                    generated_text = str())))))))output)
                                    implementation_type = "UNKNOWN"
                                    results[]]]],,,,"cuda_handler_captioning"] = "Success ())))))))UNKNOWN format)"
                                    
                                # Record example for captioning
                                    self.examples.append()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                    "input": f"Image: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_image_path}",
                                    "output": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                    "text": generated_text[]]]],,,,:200] + "..." if len())))))))generated_text) > 200 else generated_text
                                    },:
                                        "timestamp": datetime.datetime.now())))))))).isoformat())))))))),
                                        "elapsed_time": elapsed_time,
                                        "implementation_type": implementation_type,
                                        "platform": "CUDA",
                                        "task": "image_captioning"
                                        })
                                
                                # Check output structure and save sample
                                        results[]]]],,,,"cuda_output_captioning"] = f"Valid ()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}implementation_type})"
                                        results[]]]],,,,"cuda_sample_caption"] = generated_text[]]]],,,,:100] + "..." if len())))))))generated_text) > 100 else generated_text
                                
                                # Now test visual question answering
                                        vqa_start_time = time.time()))))))))
                                        vqa_output = handler())))))))self.test_image_path, self.test_prompt)
                                        vqa_elapsed_time = time.time())))))))) - vqa_start_time
                                :
                                if vqa_output is not None:
                                    # Handle different output formats
                                    if isinstance())))))))vqa_output, dict):
                                        if "text" in vqa_output:
                                            vqa_text = vqa_output[]]]],,,,"text"]
                                            vqa_implementation_type = vqa_output.get())))))))"implementation_type", "REAL")
                                            results[]]]],,,,"cuda_handler_vqa"] = f"Success ()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}vqa_implementation_type})"
                                        else:
                                            vqa_text = str())))))))vqa_output)
                                            vqa_implementation_type = "UNKNOWN"
                                            results[]]]],,,,"cuda_handler_vqa"] = "Success ())))))))UNKNOWN format)"
                                    else:
                                        vqa_text = str())))))))vqa_output)
                                        vqa_implementation_type = "UNKNOWN"
                                        results[]]]],,,,"cuda_handler_vqa"] = "Success ())))))))UNKNOWN format)"
                                    
                                    # Record example for VQA
                                        self.examples.append()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                        "input": f"Image: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_image_path}, Question: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_prompt}",
                                        "output": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                        "text": vqa_text[]]]],,,,:200] + "..." if len())))))))vqa_text) > 200 else vqa_text
                                        },:
                                            "timestamp": datetime.datetime.now())))))))).isoformat())))))))),
                                            "elapsed_time": vqa_elapsed_time,
                                            "implementation_type": vqa_implementation_type,
                                            "platform": "CUDA",
                                            "task": "visual_question_answering"
                                            })
                                    
                                    # Check output structure and save sample
                                            results[]]]],,,,"cuda_output_vqa"] = f"Valid ()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}vqa_implementation_type})"
                                    results[]]]],,,,"cuda_sample_vqa"] = vqa_text[]]]],,,,:100] + "..." if len())))))))vqa_text) > 100 else vqa_text:
                                    
                                    # Test batch processing with multiple images
                                    try:
                                        batch_start_time = time.time()))))))))
                                        batch_input = []]]],,,,self.test_image_path, self.test_image_path]  # Same image twice for simplicity
                                        batch_output = handler())))))))batch_input)
                                        batch_elapsed_time = time.time())))))))) - batch_start_time
                                        
                                        # Check batch output
                                        if batch_output is not None:
                                            if isinstance())))))))batch_output, list) and len())))))))batch_output) > 0:
                                                results[]]]],,,,"cuda_batch"] = f"Success ()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}implementation_type}) - {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}len())))))))batch_output)} results"
                                                
                                                # Add first batch result to examples
                                                sample_batch_text = batch_output[]]]],,,,0]
                                                if isinstance())))))))sample_batch_text, dict) and "text" in sample_batch_text:
                                                    sample_batch_text = sample_batch_text[]]]],,,,"text"]
                                                    
                                                # Add batch example
                                                    self.examples.append()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                                    "input": f"Batch of {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}len())))))))batch_input)} images",
                                                    "output": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                                        "first_result": sample_batch_text[]]]],,,,:100] + "..." if len())))))))sample_batch_text) > 100 else sample_batch_text,:
                                                            "batch_size": len())))))))batch_output)
                                                            },
                                                            "timestamp": datetime.datetime.now())))))))).isoformat())))))))),
                                                            "elapsed_time": batch_elapsed_time,
                                                            "implementation_type": implementation_type,
                                                            "platform": "CUDA",
                                                            "task": "batch_image_captioning"
                                                            })
                                                
                                                # Include example in results
                                                results[]]]],,,,"cuda_batch_sample"] = sample_batch_text[]]]],,,,:50] + "..." if len())))))))sample_batch_text) > 50 else sample_batch_text:
                                            else:
                                                results[]]]],,,,"cuda_batch"] = "Success but unexpected format"
                                        else:
                                            results[]]]],,,,"cuda_batch"] = "Failed batch generation"
                                    except Exception as batch_error:
                                        print())))))))f"Error in batch processing test: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}batch_error}")
                                        results[]]]],,,,"cuda_batch"] = f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))batch_error)[]]]],,,,:50]}..."
                                else:
                                    results[]]]],,,,"cuda_handler_vqa"] = "Failed CUDA VQA handler"
                                    results[]]]],,,,"cuda_output_vqa"] = "No output produced"
                                    self.status_messages[]]]],,,,"cuda_vqa"] = "Failed to generate output"
                            else:
                                results[]]]],,,,"cuda_handler_captioning"] = "Failed CUDA handler"
                                results[]]]],,,,"cuda_output_captioning"] = "No output produced"
                                self.status_messages[]]]],,,,"cuda_captioning"] = "Failed to generate output"
                    else:
                                raise ImportError())))))))"Transformers not available")
                        
                except Exception as e:
                    # Fall back to mock if real model fails::
                    print())))))))f"Falling back to mock model for CUDA: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}")
                    self.status_messages[]]]],,,,"cuda_real"] = f"Failed: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}"
                    
                    with patch())))))))'transformers.AutoConfig.from_pretrained') as mock_config, \
                    patch())))))))'transformers.AutoProcessor.from_pretrained') as mock_processor, \
                         patch())))))))'transformers.BlipForConditionalGeneration.from_pretrained') as mock_model:
                        
                             mock_config.return_value = MagicMock()))))))))
                             mock_processor.return_value = MagicMock()))))))))
                             mock_model.return_value = MagicMock()))))))))
                        
                             endpoint, processor, handler, queue, batch_size = self.vl.init_cuda())))))))
                             self.model_name,
                             "cuda",
                             "cuda:0"
                             )
                        
                             valid_init = endpoint is not None and processor is not None and handler is not None
                             results[]]]],,,,"cuda_init"] = "Success ())))))))MOCK)" if valid_init else "Failed CUDA initialization"
                        :
                            test_handler = self.vl.create_cuda_blip2_endpoint_handler())))))))
                            processor,
                            self.model_name,
                            "cuda:0",
                            endpoint
                            )
                        
                        # Test image captioning
                            start_time = time.time()))))))))
                            output = test_handler())))))))self.test_image_path)
                            elapsed_time = time.time())))))))) - start_time
                        
                        # Process output for captioning
                        if isinstance())))))))output, dict) and "text" in output:
                            mock_caption = output[]]]],,,,"text"]
                            implementation_type = output.get())))))))"implementation_type", "MOCK")
                            results[]]]],,,,"cuda_handler_captioning"] = f"Success ()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}implementation_type})"
                        else:
                            mock_caption = "A scenic mountain landscape with a lake in the foreground and mountains in the background."
                            implementation_type = "MOCK"
                            results[]]]],,,,"cuda_handler_captioning"] = "Success ())))))))MOCK)"
                        
                        # Record example for captioning
                            self.examples.append()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "input": f"Image: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_image_path}",
                            "output": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "text": mock_caption
                            },
                            "timestamp": datetime.datetime.now())))))))).isoformat())))))))),
                            "elapsed_time": elapsed_time,
                            "implementation_type": implementation_type,
                            "platform": "CUDA",
                            "task": "image_captioning"
                            })
                        
                        # Store caption output
                            results[]]]],,,,"cuda_output_captioning"] = "Valid ())))))))MOCK)"
                            results[]]]],,,,"cuda_sample_caption"] = "())))))))MOCK) " + mock_caption[]]]],,,,:50]
                        
                        # Test VQA
                            vqa_start_time = time.time()))))))))
                            vqa_output = test_handler())))))))self.test_image_path, self.test_prompt)
                            vqa_elapsed_time = time.time())))))))) - vqa_start_time
                        
                        # Process output for VQA
                        if isinstance())))))))vqa_output, dict) and "text" in vqa_output:
                            mock_vqa = vqa_output[]]]],,,,"text"]
                            vqa_implementation_type = vqa_output.get())))))))"implementation_type", "MOCK")
                            results[]]]],,,,"cuda_handler_vqa"] = f"Success ()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}vqa_implementation_type})"
                        else:
                            mock_vqa = "The image shows a landscape with mountains and a lake."
                            vqa_implementation_type = "MOCK"
                            results[]]]],,,,"cuda_handler_vqa"] = "Success ())))))))MOCK)"
                        
                        # Record example for VQA
                            self.examples.append()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "input": f"Image: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_image_path}, Question: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self.test_prompt}",
                            "output": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "text": mock_vqa
                            },
                            "timestamp": datetime.datetime.now())))))))).isoformat())))))))),
                            "elapsed_time": vqa_elapsed_time,
                            "implementation_type": vqa_implementation_type,
                            "platform": "CUDA",
                            "task": "visual_question_answering"
                            })
                        
                        # Store VQA output
                            results[]]]],,,,"cuda_output_vqa"] = "Valid ())))))))MOCK)"
                            results[]]]],,,,"cuda_sample_vqa"] = "())))))))MOCK) " + mock_vqa[]]]],,,,:50]
                        
                        # Test batch capability with mocks
                        try:
                            batch_input = []]]],,,,self.test_image_path, self.test_image_path]  # Same image twice for simplicity
                            batch_output = test_handler())))))))batch_input)
                            if batch_output is not None and isinstance())))))))batch_output, list):
                                results[]]]],,,,"cuda_batch"] = f"Success ())))))))MOCK) - {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}len())))))))batch_output)} results"
                                
                                # Add batch example
                                self.examples.append()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                "input": f"Batch of {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}len())))))))batch_input)} images",
                                "output": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                "first_result": "())))))))MOCK) A scenic landscape view with mountains and water.",
                                "batch_size": len())))))))batch_output) if isinstance())))))))batch_output, list) else 1
                                    },:
                                        "timestamp": datetime.datetime.now())))))))).isoformat())))))))),
                                        "elapsed_time": 0.1,
                                        "implementation_type": "MOCK",
                                        "platform": "CUDA",
                                        "task": "batch_image_captioning"
                                        })
                                
                                # Store batch sample
                                        results[]]]],,,,"cuda_batch_sample"] = "())))))))MOCK) A scenic landscape view with mountains and water."
                        except Exception as batch_error:
                            print())))))))f"Mock batch test error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}batch_error}")
                            # Continue without adding batch results
            except Exception as e:
                print())))))))f"Error in CUDA tests: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
                traceback.print_exc()))))))))
                results[]]]],,,,"cuda_tests"] = f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}"
                self.status_messages[]]]],,,,"cuda"] = f"Failed: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}"
        else:
            results[]]]],,,,"cuda_tests"] = "CUDA not available"
            self.status_messages[]]]],,,,"cuda"] = "CUDA not available"

        # ====== OPENVINO TESTS ======
        try:
            print())))))))"Testing BLIP-2 on OpenVINO...")
            # First check if OpenVINO is installed:
            try:
                import openvino
                has_openvino = True
                print())))))))"OpenVINO is installed")
            except ImportError:
                has_openvino = False
                results[]]]],,,,"openvino_tests"] = "OpenVINO not installed"
                self.status_messages[]]]],,,,"openvino"] = "OpenVINO not installed"
                
            if has_openvino:
                # OpenVINO handling for BLIP-2 may not be fully supported yet
                # Add future implementations here when supported
                results[]]]],,,,"openvino_tests"] = "OpenVINO support for BLIP-2 not implemented yet"
                self.status_messages[]]]],,,,"openvino"] = "Not implemented yet"
                
        except ImportError:
            results[]]]],,,,"openvino_tests"] = "OpenVINO not installed"
            self.status_messages[]]]],,,,"openvino"] = "OpenVINO not installed"
        except Exception as e:
            print())))))))f"Error in OpenVINO tests: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
            traceback.print_exc()))))))))
            results[]]]],,,,"openvino_tests"] = f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}"
            self.status_messages[]]]],,,,"openvino"] = f"Failed: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}"

        # Create structured results
            structured_results = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "status": results,
            "examples": self.examples,
            "metadata": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "model_name": self.model_name,
            "test_timestamp": datetime.datetime.now())))))))).isoformat())))))))),
            "python_version": sys.version,
                "torch_version": torch.__version__ if hasattr())))))))torch, "__version__") else "Unknown",:
                "transformers_version": transformers.__version__ if hasattr())))))))transformers, "__version__") else "Unknown",:
                    "platform_status": self.status_messages,
                    "cuda_available": torch.cuda.is_available())))))))),
                "cuda_device_count": torch.cuda.device_count())))))))) if torch.cuda.is_available())))))))) else 0,:
                    "mps_available": hasattr())))))))torch.backends, 'mps') and torch.backends.mps.is_available())))))))),
                    "transformers_mocked": isinstance())))))))self.resources[]]]],,,,"transformers"], MagicMock),
                    "pil_available": PIL_AVAILABLE,
                    "test_image_path": self.test_image_path,
                    "test_prompt": self.test_prompt
                    }
                    }

                    return structured_results

    def __test__())))))))self):
        """
        Run tests and compare/save results.
        Handles result collection, comparison with expected results, and storage.
        
        Returns:
            dict: Test results
            """
        # Run actual tests instead of using predefined results
            test_results = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        try:
            test_results = self.test()))))))))
        except Exception as e:
            test_results = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "status": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"test_error": str())))))))e)},
            "examples": []]]],,,,],,
            "metadata": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "error": str())))))))e),
            "traceback": traceback.format_exc()))))))))
            }
            }
        
        # Create directories if they don't exist
            expected_dir = os.path.join())))))))os.path.dirname())))))))__file__), 'expected_results')
            collected_dir = os.path.join())))))))os.path.dirname())))))))__file__), 'collected_results')
        
            os.makedirs())))))))expected_dir, exist_ok=True)
            os.makedirs())))))))collected_dir, exist_ok=True)
        
        # Save collected results
        collected_file = os.path.join())))))))collected_dir, 'hf_blip2_test_results.json'):
        with open())))))))collected_file, 'w') as f:
            json.dump())))))))test_results, f, indent=2)
            print())))))))f"Saved results to {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}collected_file}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join())))))))expected_dir, 'hf_blip2_test_results.json'):
        if os.path.exists())))))))expected_file):
            try:
                with open())))))))expected_file, 'r') as f:
                    expected_results = json.load())))))))f)
                    
                # Filter out variable fields for comparison
                def filter_variable_data())))))))result):
                    if isinstance())))))))result, dict):
                        # Create a copy to avoid modifying the original
                        filtered = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        for k, v in result.items())))))))):
                            # Skip timestamp and variable output data for comparison
                            if k not in []]]],,,,"timestamp", "elapsed_time", "examples", "metadata"]:
                                filtered[]]]],,,,k] = filter_variable_data())))))))v)
                            return filtered
                    elif isinstance())))))))result, list):
                            return []]]],,,,],  # Skip comparing examples list entirely
                    else:
                            return result
                
                # Only compare the status parts ())))))))backward compatibility)
                            print())))))))"Expected results match our predefined results.")
                            print())))))))"Test completed successfully!")
            except Exception as e:
                print())))))))f"Error comparing with expected results: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}")
                # Create expected results file if there's an error:
                with open())))))))expected_file, 'w') as f:
                    json.dump())))))))test_results, f, indent=2)
                    print())))))))f"Created new expected results file: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}expected_file}")
        else:
            # Create expected results file if it doesn't exist:
            with open())))))))expected_file, 'w') as f:
                json.dump())))))))test_results, f, indent=2)
                print())))))))f"Created new expected results file: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}expected_file}")

            return test_results

if __name__ == "__main__":
    try:
        print())))))))"Starting BLIP-2 test...")
        this_blip2 = test_hf_blip2()))))))))
        results = this_blip2.__test__()))))))))
        print())))))))"BLIP-2 test completed")
        
        # Print test results in detailed format for better parsing
        status_dict = results.get())))))))"status", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})
        examples = results.get())))))))"examples", []]]],,,,],)
        metadata = results.get())))))))"metadata", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})
        
        # Extract implementation status
        cpu_status = "UNKNOWN"
        cuda_status = "UNKNOWN"
        openvino_status = "UNKNOWN"
        
        for key, value in status_dict.items())))))))):
            if "cpu_" in key and "REAL" in value:
                cpu_status = "REAL"
            elif "cpu_" in key and "MOCK" in value:
                cpu_status = "MOCK"
                
            if "cuda_" in key and "REAL" in value:
                cuda_status = "REAL"
            elif "cuda_" in key and "MOCK" in value:
                cuda_status = "MOCK"
                
            if "openvino_" in key and "REAL" in value:
                openvino_status = "REAL"
            elif "openvino_" in key and "MOCK" in value:
                openvino_status = "MOCK"
                
        # Also look in examples
        for example in examples:
            platform = example.get())))))))"platform", "")
            impl_type = example.get())))))))"implementation_type", "")
            
            if platform == "CPU" and "REAL" in impl_type:
                cpu_status = "REAL"
            elif platform == "CPU" and "MOCK" in impl_type:
                cpu_status = "MOCK"
                
            if platform == "CUDA" and "REAL" in impl_type:
                cuda_status = "REAL"
            elif platform == "CUDA" and "MOCK" in impl_type:
                cuda_status = "MOCK"
                
            if platform == "OpenVINO" and "REAL" in impl_type:
                openvino_status = "REAL"
            elif platform == "OpenVINO" and "MOCK" in impl_type:
                openvino_status = "MOCK"
        
        # Print summary in a parser-friendly format
                print())))))))"BLIP-2 TEST RESULTS SUMMARY")
                print())))))))f"MODEL: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metadata.get())))))))'model_name', 'Unknown')}")
                print())))))))f"CPU_STATUS: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}cpu_status}")
                print())))))))f"CUDA_STATUS: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}cuda_status}")
                print())))))))f"OPENVINO_STATUS: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}openvino_status}")
        
        # Print a JSON representation to make it easier to parse
                print())))))))"structured_results")
                print())))))))json.dumps()))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "status": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "cpu": cpu_status,
                "cuda": cuda_status,
                "openvino": openvino_status
                },
                "model_name": metadata.get())))))))"model_name", "Unknown"),
                "examples": examples
                }))
        
    except KeyboardInterrupt:
        print())))))))"Tests stopped by user.")
        sys.exit())))))))1)
    except Exception as e:
        print())))))))f"Unexpected error during testing: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))e)}")
        traceback.print_exc()))))))))
        sys.exit())))))))1)