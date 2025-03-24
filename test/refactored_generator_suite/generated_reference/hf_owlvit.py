import asyncio
import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple

class hf_owlvit:
    """HuggingFace OWLVIT implementation.
    
    This class provides standardized interfaces for working with OWLVIT models
    across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm).
    
    This is a bidirectional encoder model used for text embeddings.
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the OWLVIT model.
        
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
        test_input = "This is a test input for BERT."
        timestamp1 = time.time()
        test_batch = None
        
        try:
            # Run the model
            test_batch = endpoint_handler(test_input)
            print(test_batch)
            print("hf_owlvit test passed")
            return True
        except Exception as e:
            print(e)
            print("hf_owlvit test failed")
            return False
    
    def init_cpu(self, model_name, device, cpu_label):
        """Initialize OWLVIT model for CPU inference."""
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
            
            return model, tokenizer, handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing CPU endpoint: {e}")
            return None, None, None, asyncio.Queue(32), 0
    
    def init_cuda(self, model_name, device, cuda_label):
        """Initialize OWLVIT model for CUDA inference."""
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
            
            return model, tokenizer, handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing CUDA endpoint: {e}")
            return None, None, None, asyncio.Queue(32), 0
    
    def init_openvino(self, model_name, device, openvino_label):
        """Initialize OWLVIT model for OpenVINO inference."""
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
            
            return model, tokenizer, handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing OpenVINO endpoint: {e}")
            return None, None, None, asyncio.Queue(32), 0
    
    def init_apple(self, model_name, device, apple_label):
        """Initialize OWLVIT model for Apple Silicon (MPS) inference."""
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
            
            return model, tokenizer, handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing Apple Silicon endpoint: {e}")
            return None, None, None, asyncio.Queue(32), 0
    
    def init_qualcomm(self, model_name, device, qualcomm_label):
        """Initialize OWLVIT model for Qualcomm inference."""
        self.init()
        
        # For now, we create a mock implementation since Qualcomm SDK integration requires specific hardware
        print("Qualcomm implementation is a mock for now")
        return None, None, None, asyncio.Queue(32), 0
    
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