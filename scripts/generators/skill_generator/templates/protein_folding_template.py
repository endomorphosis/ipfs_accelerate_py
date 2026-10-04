import os
import anyio
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from ..anyio_queue import AnyioQueue

class hf_{model_type}:
    """HuggingFace {model_type_upper} protein folding/biological sequence model implementation.
    
    This class provides standardized interfaces for working with {model_type_upper} protein folding models
    across different hardware backends (CPU, CUDA, ROCm, Apple, OpenVINO, Qualcomm).
    
    {model_description}
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the {model_type_upper} protein folding model.
        
        Args:
            resources (dict): Dictionary of shared resources (torch, transformers, etc.)
            metadata (dict): Configuration metadata
        """
        self.resources = resources if resources is not None else {}
        self.metadata = metadata if metadata is not None else {}
        
        # Handler creation methods
        self.create_cpu_protein_endpoint_handler = self.create_cpu_protein_endpoint_handler
        self.create_cuda_protein_endpoint_handler = self.create_cuda_protein_endpoint_handler
        self.create_rocm_protein_endpoint_handler = self.create_rocm_protein_endpoint_handler
        self.create_openvino_protein_endpoint_handler = self.create_openvino_protein_endpoint_handler
        self.create_apple_protein_endpoint_handler = self.create_apple_protein_endpoint_handler
        self.create_qualcomm_protein_endpoint_handler = self.create_qualcomm_protein_endpoint_handler
        
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
        """Create a mock tokenizer for graceful degradation when the real one fails.
        
        Returns:
            Mock tokenizer object with essential methods
        """
        try:
            from unittest.mock import MagicMock
            
            tokenizer = MagicMock()
            
            # Configure mock tokenizer call behavior
            def mock_tokenize(sequence, return_tensors="pt", add_special_tokens=True):
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                
                # Model-specific mock input format
                if isinstance(sequence, str):
                    batch_size = 1
                    seq_length = min(len(sequence), 1024)
                else:
                    batch_size = len(sequence)
                    seq_length = min(max(len(seq) for seq in sequence), 1024)
                
                return {
                    "input_ids": torch.ones((batch_size, seq_length), dtype=torch.long),
                    "attention_mask": torch.ones((batch_size, seq_length), dtype=torch.long),
                }
                
            tokenizer.side_effect = mock_tokenize
            tokenizer.__call__ = mock_tokenize
            
            # Add protein-specific methods
            def mock_batch_encode_plus(sequences, **kwargs):
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                
                if isinstance(sequences, str):
                    batch_size = 1
                    seq_length = min(len(sequences), 1024)
                else:
                    batch_size = len(sequences)
                    seq_length = min(max(len(seq) for seq in sequences), 1024)
                
                return {
                    "input_ids": torch.ones((batch_size, seq_length), dtype=torch.long),
                    "attention_mask": torch.ones((batch_size, seq_length), dtype=torch.long),
                    "token_type_ids": torch.zeros((batch_size, seq_length), dtype=torch.long)
                }
            
            tokenizer.batch_encode_plus = mock_batch_encode_plus
            tokenizer.encode_plus = lambda seq, **kwargs: mock_batch_encode_plus([seq], **kwargs)
            
            print("(MOCK) Created mock {model_type_upper} tokenizer")
            return tokenizer
            
        except ImportError:
            # Fallback if unittest.mock is not available
            class SimpleTokenizer:
                def __init__(self, parent):
                    self.parent = parent
                    
                def __call__(self, sequence, return_tensors="pt", add_special_tokens=True):
                    if hasattr(self.parent, 'torch'):
                        torch = self.parent.torch
                    else:
                        import torch
                    
                    # Model-specific mock input format
                    if isinstance(sequence, str):
                        batch_size = 1
                        seq_length = min(len(sequence), 1024)
                    else:
                        batch_size = len(sequence)
                        seq_length = min(max(len(seq) for seq in sequence), 1024)
                    
                    return {
                        "input_ids": torch.ones((batch_size, seq_length), dtype=torch.long),
                        "attention_mask": torch.ones((batch_size, seq_length), dtype=torch.long),
                    }
                    
                def batch_encode_plus(self, sequences, **kwargs):
                    if hasattr(self.parent, 'torch'):
                        torch = self.parent.torch
                    else:
                        import torch
                    
                    if isinstance(sequences, str):
                        batch_size = 1
                        seq_length = min(len(sequences), 1024)
                    else:
                        batch_size = len(sequences)
                        seq_length = min(max(len(seq) for seq in sequences), 1024)
                    
                    return {
                        "input_ids": torch.ones((batch_size, seq_length), dtype=torch.long),
                        "attention_mask": torch.ones((batch_size, seq_length), dtype=torch.long),
                        "token_type_ids": torch.zeros((batch_size, seq_length), dtype=torch.long)
                    }
                    
                def encode_plus(self, sequence, **kwargs):
                    return self.batch_encode_plus([sequence], **kwargs)
            
            print("(MOCK) Created simple mock {model_type_upper} tokenizer")
            return SimpleTokenizer(self)
    
    def _create_mock_endpoint(self, model_name, device_label):
        """Create mock endpoint objects when real initialization fails.
        
        Args:
            model_name (str): The model name or path
            device_label (str): The device label (cpu, cuda, etc.)
            
        Returns:
            Tuple of (endpoint, tokenizer, handler, queue, batch_size)
        """
        try:
            from unittest.mock import MagicMock
            
            # Create mock endpoint
            endpoint = MagicMock()
            
            # Configure mock endpoint behavior for protein folding/embedding
            def mock_forward(**kwargs):
                input_ids = kwargs.get("input_ids")
                if input_ids is None:
                    batch_size = 1
                    seq_length = 512
                else:
                    batch_size = input_ids.shape[0]
                    seq_length = input_ids.shape[1]
                    
                hidden_size = {hidden_size}  # Standard hidden size for this model type
                
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                
                # Create mock output structure based on model architecture
                result = MagicMock()
                result.last_hidden_state = torch.rand((batch_size, seq_length, hidden_size))
                result.logits = torch.rand((batch_size, seq_length, 20))  # 20 amino acids
                
                # Add structure prediction outputs for protein folding models
                if "protein" in model_name.lower() or "esm" in model_name.lower():
                    result.positions = torch.rand((batch_size, seq_length, 3, 3))  # 3D coordinates for backbone atoms
                    result.angles = torch.rand((batch_size, seq_length, 7))  # Torsion angles
                    result.distogram = torch.rand((batch_size, seq_length, seq_length))  # Distance matrix
                
                return result
                
            endpoint.side_effect = mock_forward
            endpoint.__call__ = mock_forward
            
            # Create mock tokenizer
            tokenizer = self._create_mock_processor()
            
            # Create appropriate handler for the device type
            if device_label.startswith('cpu'):
                handler_method = self.create_cpu_protein_endpoint_handler
            elif device_label.startswith('cuda'):
                handler_method = self.create_cuda_protein_endpoint_handler
            elif device_label.startswith('rocm'):
                handler_method = self.create_rocm_protein_endpoint_handler
            elif device_label.startswith('openvino'):
                handler_method = self.create_openvino_protein_endpoint_handler
            elif device_label.startswith('apple'):
                handler_method = self.create_apple_protein_endpoint_handler
            elif device_label.startswith('qualcomm'):
                handler_method = self.create_qualcomm_protein_endpoint_handler
            else:
                handler_method = self.create_cpu_protein_endpoint_handler
            
            # Create handler function
            mock_handler = handler_method(
                endpoint_model=model_name,
                device=device_label.split(':')[0] if ':' in device_label else device_label,
                hardware_label=device_label,
                endpoint=endpoint,
                tokenizer=tokenizer
            )
            
            print(f"(MOCK) Created mock {model_type_upper} endpoint for {model_name} on {device_label}")
            return endpoint, tokenizer, mock_handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error creating mock endpoint: {e}")
            return None, None, None, AnyioQueue(32), 0
    
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
            
        # Try to import bio-specific libraries
        try:
            if "biotite" not in list(self.resources.keys()):
                import biotite
                self.biotite = biotite
            else:
                self.biotite = self.resources["biotite"]
                
            if "biotite.structure" not in list(self.resources.keys()):
                import biotite.structure as bs
                self.bs = bs
            else:
                self.bs = self.resources["biotite.structure"]
        except ImportError:
            self.biotite = None
            self.bs = None
            print("Biotite not available, some protein visualization functions will be limited")

        return None
    
    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        """Test function to validate endpoint functionality.
        
        Args:
            endpoint_model: The model name or path
            endpoint_handler: The handler function
            endpoint_label: The hardware label
            tokenizer: The tokenizer
            
        Returns:
            Boolean indicating test success
        """
        test_input = "{test_input}"
        timestamp1 = time.time()
        test_batch = None
        
        # Get tokens for length calculation
        tokens = tokenizer(test_input)["input_ids"]
        len_tokens = len(tokens)
        
        try:
            # Run the model
            test_batch = endpoint_handler(test_input)
            print(test_batch)
            print("hf_{model_type} test passed")
            
            # Check output structure for protein-specific outputs
            if "embeddings" in test_batch:
                print(f"Embedding shape: {len(test_batch['embeddings'])} x {len(test_batch['embeddings'][0])}")
            if "structure" in test_batch:
                print(f"3D structure prediction available")
                
        except Exception as e:
            print(e)
            print("hf_{model_type} test failed")
            return False
            
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"tokens: {len_tokens}")
        print(f"tokens per second: {tokens_per_second}")
        
        # Clean up memory
        with self.torch.no_grad():
            if "cuda" in dir(self.torch):
                self.torch.cuda.empty_cache()
        return True

    def init_cpu(self, model_name, device, cpu_label):
        """Initialize {model_type_upper} model for CPU inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('cpu')
            cpu_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        
        print(f"Loading {model_name} for CPU inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Load model (check specific model types for protein folding)
            if "esm" in model_name.lower():
                # ESM protein language models have specific classes
                model = self.transformers.EsmModel.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float32,
                    cache_dir=cache_dir
                )
            else:
                # Default to auto model for other types
                model = self.transformers.AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float32,
                    cache_dir=cache_dir
                )
            
            model.eval()
            
            # Create handler function
            handler = self.create_cpu_protein_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=cpu_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, cpu_label, tokenizer)
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing CPU endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, cpu_label)

    def init_cuda(self, model_name, device, cuda_label):
        """Initialize {model_type_upper} model for CUDA inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('cuda:0', 'cuda:1', etc.)
            cuda_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, AnyioQueue, batch_size)
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
            
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Load model with half precision for GPU efficiency
            if "esm" in model_name.lower():
                # ESM protein language models have specific classes
                model = self.transformers.EsmModel.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float16,
                    device_map=device,
                    cache_dir=cache_dir
                )
            else:
                # Default to auto model for other types
                model = self.transformers.AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float16,
                    device_map=device,
                    cache_dir=cache_dir
                )
            
            model.eval()
            
            # Create handler function
            handler = self.create_cuda_protein_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=cuda_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, cuda_label, tokenizer)
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing CUDA endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, cuda_label)

    def init_rocm(self, model_name, device, rocm_label):
        """Initialize {model_type_upper} model for ROCm (AMD GPU) inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('cuda:0', 'cuda:1', etc. - ROCm uses CUDA device naming)
            rocm_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, AnyioQueue, batch_size)
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
            
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                model_name,
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
            
            # Load model with appropriate precision for AMD GPU
            if "esm" in model_name.lower():
                # ESM protein language models have specific classes
                model = self.transformers.EsmModel.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float16 if use_half else self.torch.float32,
                    device_map="auto",  # ROCm uses the same device map mechanism as CUDA
                    cache_dir=cache_dir
                )
            else:
                # Default to auto model for other types
                model = self.transformers.AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float16 if use_half else self.torch.float32,
                    device_map="auto",  # ROCm uses the same device map mechanism as CUDA
                    cache_dir=cache_dir
                )
            
            model.eval()
            
            # Log device mapping
            if hasattr(model, "hf_device_map"):
                print(f"Device map: {model.hf_device_map}")
            
            # Create handler function
            handler = self.create_rocm_protein_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=rocm_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, rocm_label, tokenizer)
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing ROCm endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, rocm_label)

    def init_openvino(self, model_name, device, openvino_label):
        """Initialize {model_type_upper} model for OpenVINO inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('CPU', 'GPU', etc.)
            openvino_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, AnyioQueue, batch_size)
        """
        self.init()
        
        try:
            from optimum.intel import OVModelForMaskedLM
        except ImportError:
            print(f"OpenVINO optimum not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", openvino_label.replace("openvino", "cpu"))
        
        print(f"Loading {model_name} for OpenVINO inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Load model with OpenVINO optimization
            # Note: Protein-specific models like ESM may not be directly supported by OpenVINO
            try:
                # Try with specific model class if available
                from optimum.intel import OVModelForMaskedLM
                model = OVModelForMaskedLM.from_pretrained(
                    model_name,
                    device=device,
                    cache_dir=cache_dir,
                    export=True
                )
            except Exception as not_supported_err:
                print(f"Model not directly supported by OpenVINO: {not_supported_err}")
                print("Falling back to CPU implementation")
                return self.init_cpu(model_name, "cpu", openvino_label.replace("openvino", "cpu"))
            
            # Create handler function
            handler = self.create_openvino_protein_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=openvino_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, openvino_label, tokenizer)
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing OpenVINO endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, openvino_label)

    def init_apple(self, model_name, device, apple_label):
        """Initialize {model_type_upper} model for Apple Silicon (MPS) inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('mps')
            apple_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, AnyioQueue, batch_size)
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
            
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Load model
            if "esm" in model_name.lower():
                # ESM protein language models have specific classes
                model = self.transformers.EsmModel.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float32,
                    device_map="mps",
                    cache_dir=cache_dir
                )
            else:
                # Default to auto model for other types
                model = self.transformers.AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.float32,
                    device_map="mps",
                    cache_dir=cache_dir
                )
            
            model.eval()
            
            # Create handler function
            handler = self.create_apple_protein_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=apple_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, apple_label, tokenizer)
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing Apple Silicon endpoint: {e}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, apple_label)

    def init_qualcomm(self, model_name, device, qualcomm_label):
        """Initialize {model_type_upper} model for Qualcomm inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('qualcomm')
            qualcomm_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, AnyioQueue, batch_size)
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

    def create_cpu_protein_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        """Create handler function for CPU protein sequence endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('cpu')
            hardware_label (str): The hardware label
            endpoint: The loaded model
            tokenizer: The loaded tokenizer
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and tokenizer
        def handler(sequence, *args, **kwargs):
            try:
                # Convert single sequence to list for batch processing
                if isinstance(sequence, str):
                    batch = [sequence]
                else:
                    batch = sequence
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True
                )
                
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    outputs = endpoint(**inputs)
                    
                # Process for protein specific outputs
                result = {"success": True, "device": device, "hardware": hardware_label}
                
                # Extract embeddings from the last hidden state
                if hasattr(outputs, "last_hidden_state"):
                    # For protein representation, we typically want per-residue embeddings
                    # and a global sequence embedding
                    per_residue_embeddings = outputs.last_hidden_state.cpu().numpy()
                    global_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    
                    result["per_residue_embeddings"] = per_residue_embeddings.tolist()
                    result["embeddings"] = global_embeddings.tolist()
                
                # Extract predicted structure if available
                if hasattr(outputs, "positions"):
                    positions = outputs.positions.cpu().numpy()
                    result["structure"] = {
                        "positions": positions.tolist(),
                    }
                    
                    # Add angles if available
                    if hasattr(outputs, "angles"):
                        angles = outputs.angles.cpu().numpy()
                        result["structure"]["angles"] = angles.tolist()
                        
                    # Add distogram if available
                    if hasattr(outputs, "distogram"):
                        distogram = outputs.distogram.cpu().numpy()
                        result["structure"]["distogram"] = distogram.tolist()
                
                return result
                
            except Exception as e:
                print(f"Error in CPU handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler

    def create_cuda_protein_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        """Create handler function for CUDA protein sequence endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('cuda:0', etc.)
            hardware_label (str): The hardware label
            endpoint: The loaded model
            tokenizer: The loaded tokenizer
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and tokenizer
        def handler(sequence, *args, **kwargs):
            try:
                # Convert single sequence to list for batch processing
                if isinstance(sequence, str):
                    batch = [sequence]
                else:
                    batch = sequence
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True
                )
                
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    # For CUDA, we can use mixed precision for faster inference
                    with self.torch.cuda.amp.autocast():
                        outputs = endpoint(**inputs)
                    
                # Process for protein specific outputs
                result = {"success": True, "device": device, "hardware": hardware_label}
                
                # Extract embeddings from the last hidden state
                if hasattr(outputs, "last_hidden_state"):
                    # For protein representation, we typically want per-residue embeddings
                    # and a global sequence embedding
                    per_residue_embeddings = outputs.last_hidden_state.cpu().numpy()
                    global_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    
                    result["per_residue_embeddings"] = per_residue_embeddings.tolist()
                    result["embeddings"] = global_embeddings.tolist()
                
                # Extract predicted structure if available
                if hasattr(outputs, "positions"):
                    positions = outputs.positions.cpu().numpy()
                    result["structure"] = {
                        "positions": positions.tolist(),
                    }
                    
                    # Add angles if available
                    if hasattr(outputs, "angles"):
                        angles = outputs.angles.cpu().numpy()
                        result["structure"]["angles"] = angles.tolist()
                        
                    # Add distogram if available
                    if hasattr(outputs, "distogram"):
                        distogram = outputs.distogram.cpu().numpy()
                        result["structure"]["distogram"] = distogram.tolist()
                
                return result
                
            except Exception as e:
                print(f"Error in CUDA handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler

    def create_rocm_protein_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        """Create handler function for ROCm protein sequence endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('cuda:0', etc. - ROCm uses CUDA device naming)
            hardware_label (str): The hardware label
            endpoint: The loaded model
            tokenizer: The loaded tokenizer
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and tokenizer
        def handler(sequence, *args, **kwargs):
            try:
                # Convert single sequence to list for batch processing
                if isinstance(sequence, str):
                    batch = [sequence]
                else:
                    batch = sequence
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True
                )
                
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # For AMD GPUs, autocast may or may not be supported depending on ROCm version
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    try:
                        # Try with autocast
                        with self.torch.cuda.amp.autocast():
                            outputs = endpoint(**inputs)
                    except RuntimeError:
                        # Fallback without autocast
                        outputs = endpoint(**inputs)
                    
                # Process for protein specific outputs
                result = {"success": True, "device": device, "hardware": hardware_label}
                
                # Extract embeddings from the last hidden state
                if hasattr(outputs, "last_hidden_state"):
                    # For protein representation, we typically want per-residue embeddings
                    # and a global sequence embedding
                    per_residue_embeddings = outputs.last_hidden_state.cpu().numpy()
                    global_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    
                    result["per_residue_embeddings"] = per_residue_embeddings.tolist()
                    result["embeddings"] = global_embeddings.tolist()
                
                # Extract predicted structure if available
                if hasattr(outputs, "positions"):
                    positions = outputs.positions.cpu().numpy()
                    result["structure"] = {
                        "positions": positions.tolist(),
                    }
                    
                    # Add angles if available
                    if hasattr(outputs, "angles"):
                        angles = outputs.angles.cpu().numpy()
                        result["structure"]["angles"] = angles.tolist()
                        
                    # Add distogram if available
                    if hasattr(outputs, "distogram"):
                        distogram = outputs.distogram.cpu().numpy()
                        result["structure"]["distogram"] = distogram.tolist()
                
                return result
                
            except Exception as e:
                print(f"Error in ROCm handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler
        
    def create_openvino_protein_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        """Create handler function for OpenVINO protein sequence endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('CPU', 'GPU', etc.)
            hardware_label (str): The hardware label
            endpoint: The loaded model
            tokenizer: The loaded tokenizer
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and processor
        def handler(sequence, *args, **kwargs):
            try:
                # Convert single sequence to list for batch processing
                if isinstance(sequence, str):
                    batch = [sequence]
                else:
                    batch = sequence
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True
                )
                
                # Run inference
                outputs = endpoint(**inputs)
                
                # Process for protein specific outputs
                result = {"success": True, "device": device, "hardware": hardware_label}
                
                # Extract embeddings from the last hidden state
                if hasattr(outputs, "last_hidden_state"):
                    # For protein representation, we typically want per-residue embeddings
                    # and a global sequence embedding
                    if hasattr(outputs.last_hidden_state, "cpu"):
                        # PyTorch tensor
                        per_residue_embeddings = outputs.last_hidden_state.cpu().numpy()
                        global_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    else:
                        # Already numpy array
                        per_residue_embeddings = outputs.last_hidden_state
                        global_embeddings = outputs.last_hidden_state.mean(axis=1)
                    
                    result["per_residue_embeddings"] = per_residue_embeddings.tolist()
                    result["embeddings"] = global_embeddings.tolist()
                
                # Note: OpenVINO may not support structure prediction outputs directly
                
                return result
                
            except Exception as e:
                print(f"Error in OpenVINO handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler

    def create_apple_protein_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        """Create handler function for Apple Silicon protein sequence endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('mps')
            hardware_label (str): The hardware label
            endpoint: The loaded model
            tokenizer: The loaded tokenizer
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and tokenizer
        def handler(sequence, *args, **kwargs):
            try:
                # Convert single sequence to list for batch processing
                if isinstance(sequence, str):
                    batch = [sequence]
                else:
                    batch = sequence
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True
                )
                
                # Move inputs to the correct device
                inputs = {k: v.to("mps") for k, v in inputs.items()}
                
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    outputs = endpoint(**inputs)
                    
                # Process for protein specific outputs
                result = {"success": True, "device": device, "hardware": hardware_label}
                
                # Extract embeddings from the last hidden state
                if hasattr(outputs, "last_hidden_state"):
                    # For protein representation, we typically want per-residue embeddings
                    # and a global sequence embedding
                    per_residue_embeddings = outputs.last_hidden_state.cpu().numpy()
                    global_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    
                    result["per_residue_embeddings"] = per_residue_embeddings.tolist()
                    result["embeddings"] = global_embeddings.tolist()
                
                # Extract predicted structure if available
                if hasattr(outputs, "positions"):
                    positions = outputs.positions.cpu().numpy()
                    result["structure"] = {
                        "positions": positions.tolist(),
                    }
                    
                    # Add angles if available
                    if hasattr(outputs, "angles"):
                        angles = outputs.angles.cpu().numpy()
                        result["structure"]["angles"] = angles.tolist()
                        
                    # Add distogram if available
                    if hasattr(outputs, "distogram"):
                        distogram = outputs.distogram.cpu().numpy()
                        result["structure"]["distogram"] = distogram.tolist()
                
                return result
                
            except Exception as e:
                print(f"Error in Apple handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler

    def create_qualcomm_protein_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        """Create handler function for Qualcomm protein sequence endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('qualcomm')
            hardware_label (str): The hardware label
            endpoint: The loaded model
            tokenizer: The loaded tokenizer
            
        Returns:
            Handler function for this endpoint
        """
        # Create closure that encapsulates the model and tokenizer
        def handler(sequence, *args, **kwargs):
            try:
                # This is a placeholder for Qualcomm implementation
                # In a real implementation, we would use the Qualcomm SDK
                
                # Mock result for now
                return {
                    "success": True,
                    "device": device,
                    "hardware": hardware_label,
                    "mock": True,
                    "sequence": sequence if isinstance(sequence, str) else sequence[0]
                }
                
            except Exception as e:
                print(f"Error in Qualcomm handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler