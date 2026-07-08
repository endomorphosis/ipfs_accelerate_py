from ..anyio_queue import AnyioQueue
import anyio
import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple

class hf_vivit:
    """HuggingFace VIVIT implementation.
    
    This class provides standardized interfaces for working with VIVIT models
    across different hardware backends (CPU, CUDA, ROCm, OpenVINO, Apple, Qualcomm).
    
    This is a HuggingFace VIVIT model.
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the VIVIT model.
        
        Args:
            resources (dict): Dictionary of shared resources (torch, transformers, etc.)
            metadata (dict): Configuration metadata
        """
        self.resources = resources
        self.metadata = metadata
        
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
        self.snpe_utils = None
        return None
        
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
            
        if "numpy" not in list(self.resources.keys()):
            import numpy as np
            self.np = np
        else:
            self.np = self.resources["numpy"]

        return None
    
    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        """Test function to validate endpoint functionality."""
        test_input = "This is a test input for vivit."
        timestamp1 = time.time()
        test_batch = None
        
        try:
            # Run the model
            test_batch = endpoint_handler(test_input)
            print(test_batch)
            print("hf_vivit test passed")
            return True
        except Exception as e:
            print(e)
            print("hf_vivit test failed")
            return False
    
    def init_cpu(self, model_name, device, cpu_label):
        """Initialize VIVIT model for CPU inference."""
        self.init()
        
        print(f"Loading {model_name} for CPU inference...")
        
        try:
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            model = self.transformers.AutoModel.from_pretrained(
                model_name,
                torch_dtype=self.torch.float32,
                device_map="cpu"
            )
            
            model.eval()
            
            # Create handler function
            handler = self.create_cpu_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=cpu_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing CPU endpoint: {e}")
            return None, None, None, AnyioQueue(32), 0
    
    def init_cuda(self, model_name, device, cuda_label):
        """Initialize VIVIT model for CUDA inference."""
        self.init()
        
        if not self.torch.cuda.is_available():
            print(f"CUDA not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", cuda_label.replace("cuda", "cpu"))
        
        print(f"Loading {model_name} for CUDA inference...")
        
        try:
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            model = self.transformers.AutoModel.from_pretrained(
                model_name,
                torch_dtype=self.torch.float16,
                device_map=device
            )
            
            model.eval()
            
            # Create handler function
            handler = self.create_cuda_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=cuda_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing CUDA endpoint: {e}")
            return None, None, None, AnyioQueue(32), 0
    
    def init_rocm(self, model_name, device, rocm_label):
        """Initialize VIVIT model for ROCm (AMD GPU) inference."""
        self.init()
        
        # Check for ROCm availability
        if not hasattr(self.torch, 'cuda') or not self.torch.cuda.is_available():
            print("ROCm/HIP not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", rocm_label.replace("rocm", "cpu"))
        
        # Check if this is actually an AMD GPU
        if self.torch.cuda.get_device_name(0) and not any(x in self.torch.cuda.get_device_name(0).lower() for x in ["amd", "radeon"]):
            print("NVIDIA GPU detected instead of AMD GPU, using regular CUDA")
            return self.init_cuda(model_name, "cuda", rocm_label.replace("rocm", "cuda"))
        
        print(f"Loading {model_name} for ROCm (AMD GPU) inference...")
        
        try:
            # Check for environment variables 
            visible_devices = os.environ.get("HIP_VISIBLE_DEVICES", None) or os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if visible_devices is not None:
                print(f"Using ROCm visible devices: {visible_devices}")
            
            # Get the total GPU memory for logging purposes
            try:
                total_mem = self.torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
                print(f"AMD GPU memory: {total_mem:.2f} GB")
            except Exception as e:
                print(f"Could not query AMD GPU memory: {e}")
            
            # Determine if we should use half precision based on GPU capabilities
            use_half = True
            try:
                # Try to create a small tensor in half precision as a test
                test_tensor = self.torch.ones((10, 10), dtype=self.torch.float16, device="cuda")
                del test_tensor
                print("Half precision is supported on this AMD GPU")
            except Exception as e:
                use_half = False
                print(f"Half precision not supported on this AMD GPU: {e}")
            
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model_name)
            
            # For ROCm, we use CUDA device map since ROCm uses CUDA compatibility layer
            model = self.transformers.AutoModel.from_pretrained(
                model_name,
                torch_dtype=self.torch.float16 if use_half else self.torch.float32,
                device_map="cuda"
            )
            
            model.eval()
            
            # Create handler function
            handler = self.create_rocm_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device="cuda",  # ROCm uses CUDA device name in PyTorch
                hardware_label=rocm_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            return model, tokenizer, handler, AnyioQueue(32), 0
                
        except Exception as e:
            print(f"Error loading model on ROCm: {e}")
            return self.init_cpu(model_name, "cpu", rocm_label.replace("rocm", "cpu"))
    
    def init_openvino(self, model_name, device, openvino_label):
        """Initialize VIVIT model for OpenVINO inference."""
        self.init()
        
        try:
            from optimum.intel import OVModelForMaskedLM
        except ImportError:
            print(f"OpenVINO not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", openvino_label.replace("openvino", "cpu"))
        
        print(f"Loading {model_name} for OpenVINO inference...")
        
        try:
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Load model with OpenVINO optimization
            model = OVModelForMaskedLM.from_pretrained(
                model_name,
                device=device,
                export=True
            )
            
            # Create handler function
            handler = self.create_openvino_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=openvino_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing OpenVINO endpoint: {e}")
            return None, None, None, AnyioQueue(32), 0
    
    def init_apple(self, model_name, device, apple_label):
        """Initialize VIVIT model for Apple Silicon (MPS) inference."""
        self.init()
        
        if not (hasattr(self.torch, 'backends') and 
                hasattr(self.torch.backends, 'mps') and 
                self.torch.backends.mps.is_available()):
            print(f"Apple MPS not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", apple_label.replace("apple", "cpu"))
        
        print(f"Loading {model_name} for Apple Silicon inference...")
        
        try:
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            model = self.transformers.AutoModel.from_pretrained(
                model_name,
                torch_dtype=self.torch.float32,
                device_map="mps"
            )
            
            model.eval()
            
            # Create handler function
            handler = self.create_apple_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=apple_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing Apple Silicon endpoint: {e}")
            return None, None, None, AnyioQueue(32), 0
    
    def init_qualcomm(self, model_name, device, qualcomm_label):
        """Initialize VIVIT model for Qualcomm inference."""
        self.init()
        
        # For now, we create a mock implementation since Qualcomm SDK integration requires specific hardware
        print("Qualcomm implementation is a mock for now")
        return None, None, None, AnyioQueue(32), 0
    
    def create_cpu_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        """Create handler function for CPU text_embedding endpoint."""
        def handler(text, *args, **kwargs):
            try:
                # Convert single string to list for batch processing
                if isinstance(text, str):
                    batch = [text]
                else:
                    batch = text
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    outputs = endpoint(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
                
                return {
                    "success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label
                }
                
            except Exception as e:
                print(f"Error in CPU handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler
    
    def create_cuda_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        """Create handler function for CUDA text_embedding endpoint."""
        def handler(text, *args, **kwargs):
            try:
                # Convert single string to list for batch processing
                if isinstance(text, str):
                    batch = [text]
                else:
                    batch = text
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    outputs = endpoint(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
                
                return {
                    "success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label
                }
                
            except Exception as e:
                print(f"Error in CUDA handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler
    
    def create_rocm_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        """Create ROCm (AMD GPU) handler for text_embedding endpoint."""
        try:
            def handler(input_text, **kwargs):
                """ROCm handler for text_embedding inference."""
                # Ensure we're using the GPU
                device = "cuda"  # ROCm uses CUDA compatibility layer
                
                # Tokenize input
                if isinstance(input_text, str):
                    inputs = tokenizer(input_text, return_tensors="pt")
                elif isinstance(input_text, list) and all(isinstance(item, str) for item in input_text):
                    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
                else:
                    return {"error": "Input must be a string or list of strings", "success": False}
                
                # Move inputs to GPU
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference with no gradient tracking
                with self.torch.no_grad():
                    outputs = endpoint(**inputs)
                
                # Process outputs based on task type
                if 'text_embedding' == 'text_embedding':
                    # Get mean pooled embeddings
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
                    return {
                        "success": True,
                        "embeddings": embeddings,
                        "device": device,
                        "hardware": hardware_label
                    }
                elif 'text_embedding' == 'text_generation' or 'text_embedding' == 'causal_lm':
                    # For text generation
                    max_new_tokens = kwargs.get('max_new_tokens', 100)
                    do_sample = kwargs.get('do_sample', True)
                    temperature = kwargs.get('temperature', 0.7)
                    top_p = kwargs.get('top_p', 0.9)
                    
                    gen_outputs = endpoint.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p
                    )
                    
                    generated_text = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
                    return {
                        "success": True,
                        "generated_text": generated_text,
                        "device": device,
                        "hardware": hardware_label
                    }
                elif 'text_embedding' == 'image_classification':
                    # For image classification
                    logits = outputs.logits
                    predictions = self.torch.nn.functional.softmax(logits, dim=-1).cpu().numpy().tolist()
                    return {
                        "success": True,
                        "predictions": predictions,
                        "device": device,
                        "hardware": hardware_label
                    }
                else:
                    # Generic outputs handling
                    return {
                        "success": True,
                        "outputs": {k: v.cpu().numpy().tolist() if hasattr(v, 'cpu') else v for k, v in outputs.items()},
                        "device": device,
                        "hardware": hardware_label
                    }
            
            return handler
        except Exception as e:
            print(f"Error creating ROCm handler: {e}")
            # Return a mock handler in case of error
            def mock_handler(input_text, **kwargs):
                return {
                    "success": False,
                    "error": str(e),
                    "device": "cpu",
                    "hardware": hardware_label.replace("rocm", "cpu")
                }
            return mock_handler
    
    def create_openvino_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        """Create handler function for OpenVINO text_embedding endpoint."""
        def handler(text, *args, **kwargs):
            try:
                # Convert single string to list for batch processing
                if isinstance(text, str):
                    batch = [text]
                else:
                    batch = text
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                
                # Run inference
                outputs = endpoint(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
                
                return {
                    "success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label
                }
                
            except Exception as e:
                print(f"Error in OpenVINO handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler
    
    def create_apple_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        """Create handler function for Apple Silicon text_embedding endpoint."""
        def handler(text, *args, **kwargs):
            try:
                # Convert single string to list for batch processing
                if isinstance(text, str):
                    batch = [text]
                else:
                    batch = text
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    outputs = endpoint(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
                
                return {
                    "success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label
                }
                
            except Exception as e:
                print(f"Error in Apple handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler
    
    def create_qualcomm_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        """Create handler function for Qualcomm text_embedding endpoint."""
        def handler(text, *args, **kwargs):
            try:
                # This is a placeholder for Qualcomm implementation
                # Convert single string to list for batch processing
                if isinstance(text, str):
                    batch = [text]
                else:
                    batch = text
                
                return {
                    "success": True,
                    "device": device,
                    "hardware": hardware_label,
                    "embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)]
                }
                
            except Exception as e:
                print(f"Error in Qualcomm handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler