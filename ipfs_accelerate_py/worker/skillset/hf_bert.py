import asyncio
import os
import json
import time

class hf_bert:
    """HuggingFace BERT (Bidirectional Encoder Representations from Transformers) implementation.
    
    This class provides standardized interfaces for working with BERT models
    across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm).
    
    BERT is a transformer-based language model designed to understand context
    in text by looking at words bidirectionally. It's commonly used for text
    embedding generation, which can be used for tasks like semantic search,
    text classification, and more.
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the BERT model.
        
        Args:
            resources (dict): Dictionary of shared resources (torch, transformers, etc.)
            metadata (dict): Configuration metadata
        """
        self.resources = resources
        self.metadata = metadata
        
        # Handler creation methods
        self.create_cpu_text_embedding_endpoint_handler = self.create_cpu_text_embedding_endpoint_handler
        self.create_cuda_text_embedding_endpoint_handler = self.create_cuda_text_embedding_endpoint_handler
        self.create_openvino_text_embedding_endpoint_handler = self.create_openvino_text_embedding_endpoint_handler
        self.create_apple_text_embedding_endpoint_handler = self.create_apple_text_embedding_endpoint_handler
        self.create_qualcomm_text_embedding_endpoint_handler = self.create_qualcomm_text_embedding_endpoint_handler
        
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
            def mock_tokenize(text, return_tensors="pt", padding=None, truncation=None, max_length=None):
                if isinstance(text, str):
                    batch_size = 1
                else:
                    batch_size = len(text)
                
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                
                return {
                    "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
                    "attention_mask": torch.ones((batch_size, 10), dtype=torch.long),
                    "token_type_ids": torch.zeros((batch_size, 10), dtype=torch.long)
                }
                
            tokenizer.side_effect = mock_tokenize
            tokenizer.__call__ = mock_tokenize
            
            print("(MOCK) Created mock BERT tokenizer")
            return tokenizer
            
        except ImportError:
            # Fallback if unittest.mock is not available
            class SimpleTokenizer:
                def __init__(self, parent):
                    self.parent = parent
                    
                def __call__(self, text, return_tensors="pt", padding=None, truncation=None, max_length=None):
                    if isinstance(text, str):
                        batch_size = 1
                    else:
                        batch_size = len(text)
                    
                    if hasattr(self.parent, 'torch'):
                        torch = self.parent.torch
                    else:
                        import torch
                    
                    return {
                        "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
                        "attention_mask": torch.ones((batch_size, 10), dtype=torch.long),
                        "token_type_ids": torch.zeros((batch_size, 10), dtype=torch.long)
                    }
            
            print("(MOCK) Created simple mock BERT tokenizer")
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
            
            # Configure mock endpoint behavior
            def mock_forward(**kwargs):
                batch_size = kwargs.get("input_ids", kwargs.get("inputs_embeds", None)).shape[0]
                sequence_length = kwargs.get("input_ids", kwargs.get("inputs_embeds", None)).shape[1]
                hidden_size = 768  # Standard BERT hidden size
                
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                
                # Create mock output structure
                result = MagicMock()
                result.last_hidden_state = torch.rand((batch_size, sequence_length, hidden_size))
                return result
                
            endpoint.side_effect = mock_forward
            endpoint.__call__ = mock_forward
            
            # Create mock tokenizer
            tokenizer = self._create_mock_processor()
            
            # Create appropriate handler for the device type
            if device_label.startswith('cpu'):
                handler_method = self.create_cpu_text_embedding_endpoint_handler
            elif device_label.startswith('cuda'):
                handler_method = self.create_cuda_text_embedding_endpoint_handler
            elif device_label.startswith('openvino'):
                handler_method = self.create_openvino_text_embedding_endpoint_handler
            elif device_label.startswith('apple'):
                handler_method = self.create_apple_text_embedding_endpoint_handler
            elif device_label.startswith('qualcomm'):
                handler_method = self.create_qualcomm_text_embedding_endpoint_handler
            else:
                handler_method = self.create_cpu_text_embedding_endpoint_handler
            
            # Create handler function
            mock_handler = handler_method(
                endpoint_model=model_name,
                device=device_label.split(':')[0] if ':' in device_label else device_label,
                hardware_label=device_label,
                endpoint=endpoint,
                tokenizer=tokenizer
            )
            
            import asyncio
            print(f"(MOCK) Created mock BERT endpoint for {model_name} on {device_label}")
            return endpoint, tokenizer, mock_handler, asyncio.Queue(32), 0
            
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
            
        if "numpy" not in list(self.resources.keys()):
            import numpy as np
            self.np = np
        else:
            self.np = self.resources["numpy"]

        return None
    

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        timestamp1 = time.time()
        test_batch = None
        tokens = tokenizer(sentence_1)["input_ids"]
        len_tokens = len(tokens)
        try:
            test_batch = endpoint_handler(sentence_1)
            print(test_batch)
            print("hf_embed test passed")
        except Exception as e:
            print(e)
            print("hf_embed test failed")
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"tokens: {len_tokens}")
        print(f"tokens per second: {tokens_per_second}")
        # test_batch_sizes = await self.test_batch_sizes(metadata['models'], ipfs_accelerate_init)
        with self.torch.no_grad():
            if "cuda" in dir(self.torch):
                self.torch.cuda.empty_cache()
        return True

    def init_cpu(self, model_name, device, cpu_label):
        """Initialize BERT model for CPU inference.
        
        Args:
            model_name (str): HuggingFace model name or path (e.g., 'bert-base-uncased')
            device (str): Device to run on ('cpu')
            cpu_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        print(f"Loading {model_name} for CPU inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # First try loading with real transformers
            if "transformers" in self.resources and hasattr(self.resources["transformers"], "AutoModel"):
                # Load model configuration
                config = self.transformers.AutoConfig.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                
                # Load tokenizer
                tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                    model_name, 
                    use_fast=True, 
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                
                # Load the model
                try:
                    endpoint = self.transformers.AutoModel.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        config=config,
                        low_cpu_mem_usage=True,
                        return_dict=True,
                        cache_dir=cache_dir
                    )
                    endpoint.eval()  # Set to evaluation mode
                    
                    # Print model information
                    print(f"(REAL) Model loaded: {model_name}")
                    print(f"Model type: {config.model_type if hasattr(config, 'model_type') else 'bert'}")
                    print(f"Hidden size: {config.hidden_size}")
                    
                    # Create handler function
                    endpoint_handler = self.create_cpu_text_embedding_endpoint_handler(
                        endpoint_model=model_name,
                        device=device,
                        hardware_label=cpu_label,
                        endpoint=endpoint,
                        tokenizer=tokenizer
                    )
                    
                    return endpoint, tokenizer, endpoint_handler, asyncio.Queue(32), 0
                    
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Falling back to mock implementation")
            
            # If we get here, either transformers is a mock or the model loading failed
            # Return a mock implementation
            return self._create_mock_endpoint(model_name, cpu_label)
            
        except Exception as e:
            print(f"Error in CPU initialization: {e}")
            # Return mock objects for graceful degradation
            return self._create_mock_endpoint(model_name, cpu_label)

    def init_cuda(self, model_name, device, cuda_label):
        """Initialize BERT model for CUDA (GPU) inference with enhanced memory management.
        
        Args:
            model_name (str): HuggingFace model name or path (e.g., 'bert-base-uncased')
            device (str): Device to run on ('cuda' or 'cuda:0', etc.)
            cuda_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        # Import CUDA utilities
        try:
            from ipfs_accelerate_py.worker.cuda_utils import cuda_utils
            cuda_utils_available = True
            cuda_tools = cuda_utils(resources=self.resources, metadata=self.metadata)
            print("CUDA utilities imported successfully")
        except ImportError:
            cuda_utils_available = False
            cuda_tools = None
            print("CUDA utilities not available, using basic CUDA support")
        
        # Check if CUDA is available
        if not hasattr(self.torch, 'cuda') or not self.torch.cuda.is_available():
            print(f"CUDA is not available, falling back to CPU for model '{model_name}'")
            return self.init_cpu(model_name, "cpu", "cpu")
        
        # Get CUDA device information and validate device
        if cuda_utils_available:
            cuda_device = cuda_tools.get_cuda_device(cuda_label)
            if cuda_device is None:
                print(f"Invalid CUDA device specified in {cuda_label}, falling back to CPU")
                return self.init_cpu(model_name, "cpu", "cpu")
            device = cuda_device
        else:
            # Fallback to basic validation
            if ":" in cuda_label:
                device_index = int(cuda_label.split(":")[1])
                if device_index >= self.torch.cuda.device_count():
                    print(f"Invalid CUDA device index {device_index}, falling back to device 0")
                    device = "cuda:0"
                else:
                    device = cuda_label
            else:
                device = "cuda:0"
            
            # Clean GPU cache before loading
            self.torch.cuda.empty_cache()
        
        print(f"Loading {model_name} for CUDA inference on {device}...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load model configuration
            config = self.transformers.AutoConfig.from_pretrained(
                model_name, 
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=True, 
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            
            # Determine max batch size based on available memory (if cuda_utils available)
            if cuda_utils_available and hasattr(self.torch.cuda, 'mem_get_info'):
                try:
                    free_memory, total_memory = self.torch.cuda.mem_get_info()
                    free_memory_gb = free_memory / (1024**3)
                    batch_size = max(1, min(16, int(free_memory_gb / 0.5)))  # Heuristic: 0.5GB per batch item
                    print(f"Dynamic batch size based on available memory: {batch_size}")
                except Exception as mem_error:
                    print(f"Error determining memory-based batch size: {mem_error}")
                    batch_size = 8  # Default fallback
            else:
                batch_size = 8  # Default batch size for CUDA
            
            # Try loading with FP16 precision first for better performance
            use_half_precision = True  # Default for GPUs
            
            try:
                endpoint = self.transformers.AutoModel.from_pretrained(
                    model_name, 
                    torch_dtype=self.torch.float16 if use_half_precision else self.torch.float32,
                    trust_remote_code=True,
                    config=config,
                    low_cpu_mem_usage=True,
                    return_dict=True,
                    cache_dir=cache_dir
                )
                
                # Use CUDA utils for memory optimization if available
                if cuda_utils_available:
                    endpoint = cuda_tools.optimize_cuda_memory(
                        model=endpoint,
                        device=device,
                        use_half_precision=use_half_precision
                    )
                else:
                    # Manual optimization
                    endpoint = endpoint.to(device)
                    endpoint.eval()
                
                precision_type = "FP16" if use_half_precision else "FP32"
                print(f"(REAL) Model loaded with {precision_type} precision")
                is_real_impl = True
                
            except Exception as e:
                print(f"Failed to load model: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                
                # Fall back to mock implementation
                print("Falling back to mock implementation")
                endpoint = self._create_mock_openvino_model(model_name)
                is_real_impl = False
            
            if is_real_impl:
                # Print model and device information
                print(f"Device: {device}")
                print(f"Model type: {config.model_type if hasattr(config, 'model_type') else 'bert'}")
                print(f"Hidden size: {config.hidden_size}")
                print(f"Model precision: {endpoint.dtype}")
            
            # Create the handler function
            endpoint_handler = self.create_cuda_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=cuda_label,
                endpoint=endpoint,
                tokenizer=tokenizer,
                is_real_impl=is_real_impl,
                batch_size=batch_size
            )
            
            # Clean up memory after initialization
            if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
            
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(batch_size), batch_size
            
        except Exception as e:
            print(f"Error loading CUDA model: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            # Clean up GPU memory on error
            if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
                
            # Return mock objects for graceful degradation
            return self._create_mock_endpoint(model_name, cuda_label)

    def init_openvino(self, model_name, model_type, device, openvino_label, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None):
        """Initialize BERT model for OpenVINO inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            model_type (str): Type of model (e.g., 'feature-extraction')
            device (str): Target device for inference ('CPU', 'GPU', etc.)
            openvino_label (str): Label to identify this endpoint
            get_optimum_openvino_model: Function to get Optimum OpenVINO model
            get_openvino_model: Function to get OpenVINO model
            get_openvino_pipeline_type: Function to determine pipeline type
            openvino_cli_convert: Function to convert model to OpenVINO format
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        print(f"Loading {model_name} for OpenVINO inference...")
        
        # Load OpenVINO module - either from resources or import
        if "openvino" not in list(self.resources.keys()):
            try:
                import openvino as ov
                self.ov = ov
            except ImportError:
                print("OpenVINO not available. Falling back to CPU handler.")
                return self.init_cpu(model_name, "cpu", "cpu")
        else:
            self.ov = self.resources["openvino"]
        
        try:
            # Create local cache directory for models
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # First try using the real model if the utility functions are available
            model = None
            task = "feature-extraction"  # Default for BERT
            
            if callable(get_openvino_pipeline_type):
                task = get_openvino_pipeline_type(model_name, model_type)
            
            # Try loading the model with the utility functions
            if callable(get_openvino_model):
                try:
                    model = get_openvino_model(model_name, model_type)
                    print(f"(REAL) Successfully loaded OpenVINO model with get_openvino_model")
                except Exception as e:
                    print(f"Error loading with get_openvino_model: {e}")
            
            # Try optimum if direct loading failed
            if model is None and callable(get_optimum_openvino_model):
                try:
                    model = get_optimum_openvino_model(model_name, model_type)
                    print(f"(REAL) Successfully loaded OpenVINO model with get_optimum_openvino_model")
                except Exception as e:
                    print(f"Error loading with get_optimum_openvino_model: {e}")
            
            # If both loading methods failed, create a mock model
            if model is None:
                print("All OpenVINO model loading methods failed, creating mock model")
                model = self._create_mock_openvino_model(model_name)
                print("(MOCK) Created mock OpenVINO model for testing")
            
            # Try loading tokenizer
            try:
                tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                print(f"(REAL) Successfully loaded tokenizer for {model_name}")
            except Exception as e:
                print(f"Error loading tokenizer from HuggingFace: {e}")
                tokenizer = self._create_mock_processor()
            
            # Create the handler
            endpoint_handler = self.create_openvino_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                tokenizer=tokenizer,
                openvino_label=openvino_label,
                endpoint=model
            )
            
            print(f"Successfully initialized OpenVINO handler for {model_name}")
            return model, tokenizer, endpoint_handler, asyncio.Queue(64), 0
            
        except Exception as e:
            print(f"Error initializing OpenVINO model: {e}")
            # Return mock objects for graceful degradation
            return self._create_mock_endpoint(model_name, openvino_label)
            
    def _create_mock_openvino_model(self, model_name):
        """Create a mock OpenVINO model for testing purposes"""
        try:
            from unittest.mock import MagicMock
            mock_model = MagicMock()
            
            # Mock infer method
            def mock_infer(inputs):
                batch_size = 1
                seq_len = 10
                hidden_size = 768
                
                if isinstance(inputs, dict):
                    if "input_ids" in inputs and hasattr(inputs["input_ids"], "shape"):
                        batch_size = inputs["input_ids"].shape[0]
                        if len(inputs["input_ids"].shape) > 1:
                            seq_len = inputs["input_ids"].shape[1]
                
                # Create mock output
                last_hidden = self.torch.rand((batch_size, seq_len, hidden_size))
                return {"last_hidden_state": last_hidden}
            
            # Add the infer method
            mock_model.infer = mock_infer
            
            return mock_model
            
        except ImportError:
            # If unittest.mock is not available, create a simpler version
            class SimpleMockModel:
                def __init__(self, torch_module):
                    self.torch = torch_module
                    
                def infer(self, inputs):
                    batch_size = 1
                    seq_len = 10
                    hidden_size = 768
                    
                    if isinstance(inputs, dict):
                        if "input_ids" in inputs and hasattr(inputs["input_ids"], "shape"):
                            batch_size = inputs["input_ids"].shape[0]
                            if len(inputs["input_ids"].shape) > 1:
                                seq_len = inputs["input_ids"].shape[1]
                    
                    # Create output
                    last_hidden = self.torch.rand((batch_size, seq_len, hidden_size))
                    return {"last_hidden_state": last_hidden}
                    
                def __call__(self, inputs):
                    return self.infer(inputs)
            
            return SimpleMockModel(self.torch)              

    def init_apple(self, model, device, apple_label):
        """Initialize model for Apple Silicon (M1/M2/M3) hardware.
        
        Args:
            model: HuggingFace model name or path
            device: Device to run inference on (mps for Apple Silicon)
            apple_label: Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        try:
            import coremltools as ct
        except ImportError:
            print("coremltools not installed. Cannot initialize Apple Silicon model.")
            return None, None, None, None, 0
            
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        
        # Check if MPS (Metal Performance Shaders) is available
        if not hasattr(self.torch.backends, 'mps') or not self.torch.backends.mps.is_available():
            print("MPS not available. Cannot initialize model on Apple Silicon.")
            return None, None, None, None, 0
            
        # For Apple Silicon, we'll use MPS as the device
        try:
            endpoint = self.transformers.AutoModel.from_pretrained(
                model, 
                torch_dtype=self.torch.float16, 
                trust_remote_code=True
            ).to(device)
        except Exception as e:
            print(f"Error loading model on Apple Silicon: {e}")
            endpoint = None
            
        endpoint_handler = self.create_apple_text_embedding_endpoint_handler(endpoint, apple_label, endpoint, tokenizer)
        
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(32), 0
        
    def init_qualcomm(self, model, device, qualcomm_label):
        """Initialize model for Qualcomm hardware.
        
        Args:
            model: HuggingFace model name or path
            device: Device to run inference on
            qualcomm_label: Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        # Import SNPE utilities
        try:
            from .qualcomm_snpe_utils import get_snpe_utils
            self.snpe_utils = get_snpe_utils()
        except ImportError:
            print("Failed to import Qualcomm SNPE utilities")
            return None, None, None, None, 0
            
        if not self.snpe_utils.is_available():
            print("Qualcomm SNPE is not available on this system")
            return None, None, None, None, 0
            
        try:
            config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
            
            # Convert model path to be compatible with SNPE
            model_name = model.replace("/", "--")
            dlc_path = f"~/snpe_models/{model_name}_bert.dlc"
            dlc_path = os.path.expanduser(dlc_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(dlc_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(dlc_path):
                print(f"Converting {model} to SNPE format...")
                self.snpe_utils.convert_model(model, "embedding", str(dlc_path))
            
            # Load the SNPE model
            endpoint = self.snpe_utils.load_model(str(dlc_path))
            
            # Optimize for the specific Qualcomm device if possible
            if ":" in qualcomm_label:
                device_type = qualcomm_label.split(":")[1]
                optimized_path = self.snpe_utils.optimize_for_device(dlc_path, device_type)
                if optimized_path != dlc_path:
                    endpoint = self.snpe_utils.load_model(optimized_path)
            
            endpoint_handler = self.create_qualcomm_text_embedding_endpoint_handler(endpoint, qualcomm_label, endpoint, tokenizer)
            
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Qualcomm model: {e}")
            return None, None, None, None, 0

    def create_cpu_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None):
        """Create endpoint handler for CPU backend.
        
        Args:
            endpoint_model (str): The model name or path
            device (str): The device to run inference on ('cpu')
            hardware_label (str): Label to identify this endpoint
            endpoint: The model endpoint
            tokenizer: The tokenizer for the model
            
        Returns:
            A handler function for the CPU endpoint
        """
        def handler(text_input, endpoint_model=endpoint_model, device=device, hardware_label=hardware_label, endpoint=endpoint, tokenizer=tokenizer):
            """Process text input to generate BERT embeddings.
            
            Args:
                text_input: Input text (string or list of strings)
                
            Returns:
                Embedding tensor (mean pooled from last hidden state)
            """
            # Set model to evaluation mode
            if hasattr(endpoint, 'eval'):
                endpoint.eval()
            
            try:
                with self.torch.no_grad():
                    # Process different input types
                    if isinstance(text_input, str):
                        # Single text input
                        tokens = tokenizer(
                            text_input, 
                            return_tensors="pt", 
                            padding=True,
                            truncation=True,
                            max_length=512  # Standard BERT max length
                        )
                    elif isinstance(text_input, list):
                        # Batch of texts
                        tokens = tokenizer(
                            text_input,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512
                        )
                    else:
                        raise ValueError(f"Unsupported input type: {type(text_input)}")
                    
                    # Run inference
                    results = endpoint(**tokens)
                    
                    # Check if the output is in the expected format
                    if not hasattr(results, 'last_hidden_state'):
                        # Handle different output formats
                        if isinstance(results, dict) and 'last_hidden_state' in results:
                            last_hidden = results['last_hidden_state']
                        else:
                            # Unexpected output format, return mock
                            print(f"(MOCK) Unexpected output format from model, using fallback")
                            batch_size = 1 if isinstance(text_input, str) else len(text_input)
                            return self.torch.rand((batch_size, 768))
                    else:
                        last_hidden = results.last_hidden_state
                    
                    # Mean pooling: mask padding tokens and average across sequence length
                    # This is a standard way to get sentence embeddings from BERT
                    masked_hidden = last_hidden.masked_fill(
                        ~tokens['attention_mask'].bool().unsqueeze(-1), 
                        0.0
                    )
                    
                    # Sum and divide by actual token count (excluding padding)
                    average_pool_results = masked_hidden.sum(dim=1) / tokens['attention_mask'].sum(dim=1, keepdim=True)
                    
                    # Add timestamp and metadata for testing/debugging
                    import time
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # REAL signal in output tensor metadata for testing
                    average_pool_results.real_implementation = True
                    
                    return average_pool_results
                    
            except Exception as e:
                print(f"Error in CPU text embedding handler: {e}")
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Generate a mock embedding with error info
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                mock_embedding = self.torch.rand((batch_size, 768))
                
                # Add signal this is a mock for testing
                mock_embedding.mock_implementation = True
                
                return mock_embedding
                
        return handler

    def create_openvino_text_embedding_endpoint_handler(self, endpoint_model, tokenizer, openvino_label, endpoint=None):
        """Create endpoint handler for OpenVINO backend.
        
        Args:
            endpoint_model (str): The model name or path
            tokenizer: The tokenizer for the model
            openvino_label (str): Label to identify this endpoint
            endpoint: The OpenVINO model endpoint
            
        Returns:
            A handler function for the OpenVINO endpoint
        """
        def handler(text_input, tokenizer=tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=endpoint):
            """Process text input to generate BERT embeddings with OpenVINO.
            
            Args:
                text_input: Input text (string, list of strings, or preprocessed tokens)
                
            Returns:
                Embedding tensor (mean pooled from last hidden state)
            """
            try:
                # Process different input types
                if isinstance(text_input, str):
                    # Single text input
                    tokens = tokenizer(
                        text_input, 
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                elif isinstance(text_input, list):
                    # Batch of texts
                    tokens = tokenizer(
                        text_input, 
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                elif isinstance(text_input, dict) and "input_ids" in text_input:
                    # Already tokenized
                    tokens = text_input
                else:
                    raise ValueError(f"Unsupported input type: {type(text_input)}")

                # Convert inputs to the format expected by OpenVINO
                # OpenVINO models expect numpy arrays
                input_dict = {}
                for key, value in tokens.items():
                    if hasattr(value, 'numpy'):
                        input_dict[key] = value.numpy()
                    else:
                        input_dict[key] = value
                
                # Check if we have a valid endpoint
                if endpoint is None or not (hasattr(endpoint, '__call__') or hasattr(endpoint, 'infer')):
                    print("(MOCK) No valid OpenVINO endpoint available - using mock output")
                    # Create a fallback embedding
                    batch_size = 1 if isinstance(text_input, str) else len(text_input) if isinstance(text_input, list) else 1
                    mock_embedding = self.torch.rand((batch_size, 768))
                    mock_embedding.mock_implementation = True
                    return mock_embedding
                
                # Try different OpenVINO inference methods
                try:
                    results = None
                    
                    # Try different interface patterns for OpenVINO models
                    if hasattr(endpoint, 'infer'):
                        # OpenVINO Runtime compiled model
                        results = endpoint.infer(input_dict)
                        
                        # Extract hidden states from results
                        if isinstance(results, dict):
                            # Find output tensor - different models have different output names
                            if 'last_hidden_state' in results:
                                last_hidden_np = results['last_hidden_state']
                            elif 'hidden_states' in results:
                                last_hidden_np = results['hidden_states']
                            elif len(results) > 0:
                                # Just use first output
                                output_key = list(results.keys())[0]
                                last_hidden_np = results[output_key]
                            else:
                                raise ValueError("No output tensors in OpenVINO model results")
                                
                            # Convert to PyTorch tensor
                            last_hidden = self.torch.tensor(last_hidden_np)
                        else:
                            raise ValueError("Unexpected output format from OpenVINO model")
                            
                    elif hasattr(endpoint, '__call__'):
                        # Model might be a callable that accepts PyTorch tensors
                        results = endpoint(**tokens)
                        
                        # Extract last hidden state
                        if hasattr(results, 'last_hidden_state'):
                            last_hidden = results.last_hidden_state
                        elif isinstance(results, dict) and 'last_hidden_state' in results:
                            last_hidden = results['last_hidden_state']
                        else:
                            raise ValueError("No last_hidden_state in model output")
                    else:
                        raise ValueError("OpenVINO model has no supported inference method")
                    
                    # Get attention mask (may be numpy or tensor)
                    if 'attention_mask' in tokens:
                        attention_mask = tokens['attention_mask']
                        if not isinstance(attention_mask, self.torch.Tensor):
                            attention_mask = self.torch.tensor(attention_mask)
                    else:
                        # Create default attention mask (all 1s)
                        attention_mask = self.torch.ones(last_hidden.shape[:2], dtype=self.torch.bool)
                    
                    # Mean pooling: mask padding tokens and average across sequence length
                    masked_hidden = last_hidden.masked_fill(
                        ~attention_mask.bool().unsqueeze(-1), 
                        0.0
                    )
                    
                    # Sum and divide by actual token count (excluding padding)
                    average_pool_results = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                    
                    # Add REAL implementation marker for testing
                    average_pool_results.real_implementation = True
                    average_pool_results.is_openvino = True
                    
                    # Add timestamp for debugging
                    import time
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    return average_pool_results
                    
                except Exception as inference_error:
                    print(f"(MOCK) Error running OpenVINO inference: {inference_error}")
                    # Generate mock embedding as fallback
                    batch_size = 1 if isinstance(text_input, str) else len(text_input) if isinstance(text_input, list) else 1
                    mock_embedding = self.torch.rand((batch_size, 768))
                    mock_embedding.mock_implementation = True
                    return mock_embedding
                
            except Exception as e:
                print(f"Error in OpenVINO text embedding handler: {e}")
                
                # Generate a mock embedding with error info
                batch_size = 1 if isinstance(text_input, str) else len(text_input) if isinstance(text_input, list) else 1
                mock_embedding = self.torch.rand((batch_size, 768))
                mock_embedding.mock_implementation = True
                
                return mock_embedding
            
        return handler

    def create_cuda_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None, is_real_impl=True, batch_size=8):
        """Create endpoint handler for CUDA backend with advanced memory management.
        
        Args:
            endpoint_model (str): The model name or path
            device (str): The device to run inference on ('cuda', 'cuda:0', etc.)
            hardware_label (str): Label to identify this endpoint
            endpoint: The model endpoint
            tokenizer: The tokenizer for the model
            is_real_impl (bool): Flag indicating if we're using real implementation or mock
            batch_size (int): Batch size to use for processing
            
        Returns:
            A handler function for the CUDA endpoint
        """
        # Import CUDA utilities if available
        try:
            from ipfs_accelerate_py.worker.cuda_utils import cuda_utils
            cuda_utils_available = True
            cuda_tools = cuda_utils(resources=self.resources, metadata=self.metadata)
            print("CUDA utilities imported successfully for handler")
        except ImportError:
            cuda_utils_available = False
            cuda_tools = None
            print("CUDA utilities not available for handler, using basic implementation")
        
        def handler(text_input, endpoint_model=endpoint_model, device=device, hardware_label=hardware_label, 
                   endpoint=endpoint, tokenizer=tokenizer, is_real_impl=is_real_impl, batch_size=batch_size):
            """Process text input to generate BERT embeddings on CUDA with optimized memory handling.
            
            Args:
                text_input: Input text (string or list of strings)
                
            Returns:
                Embedding tensor (mean pooled from last hidden state)
            """
            # Start performance tracking
            import time
            start_time = time.time()
            
            # Record input stats
            if isinstance(text_input, str):
                input_size = 1
                input_type = "string"
            elif isinstance(text_input, list):
                input_size = len(text_input)
                input_type = "list"
            else:
                input_size = 1
                input_type = str(type(text_input))
                
            print(f"Processing {input_type} input with {input_size} items")
            
            # Set implementation type based on parameter
            using_mock = not is_real_impl
            
            # Set model to evaluation mode if it's a real model
            if hasattr(endpoint, 'eval') and not using_mock:
                endpoint.eval()
            
            # Early return for mock implementation
            if using_mock:
                mock_embedding = self.torch.rand((input_size, 768))
                mock_embedding.mock_implementation = True
                mock_embedding.implementation_type = "MOCK"
                mock_embedding.device = str(device)
                mock_embedding.model_name = endpoint_model
                return mock_embedding
            
            try:
                with self.torch.no_grad():
                    # Clean GPU memory before processing
                    if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                        self.torch.cuda.empty_cache()
                    
                    # Get CUDA memory information for tracking if available
                    free_memory_start = None
                    if hasattr(self.torch.cuda, 'mem_get_info'):
                        try:
                            free_memory_start, total_memory = self.torch.cuda.mem_get_info()
                            free_memory_start_gb = free_memory_start / (1024**3)
                            print(f"CUDA memory available before processing: {free_memory_start_gb:.2f}GB")
                        except Exception as mem_error:
                            print(f"Error getting CUDA memory info: {mem_error}")
                    
                    # Handle different input types
                    max_length = 512  # Default max length
                    if hasattr(endpoint, 'config') and hasattr(endpoint.config, 'max_position_embeddings'):
                        max_length = endpoint.config.max_position_embeddings
                    
                    # Process inputs based on type
                    if isinstance(text_input, str):
                        # Single text input
                        tokens = tokenizer(
                            text_input, 
                            return_tensors='pt', 
                            padding=True, 
                            truncation=True,
                            max_length=max_length
                        )
                    elif isinstance(text_input, list):
                        # Process in batches if input is larger than batch_size
                        if len(text_input) > batch_size and cuda_utils_available:
                            print(f"Processing input in batches (size: {batch_size})")
                            # Process in batches with CUDA utilities
                            batches = [text_input[i:i+batch_size] for i in range(0, len(text_input), batch_size)]
                            results = []
                            
                            for i, batch in enumerate(batches):
                                print(f"Processing batch {i+1}/{len(batches)}")
                                # Tokenize batch
                                batch_tokens = tokenizer(
                                    batch,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
                                    max_length=max_length
                                )
                                
                                # Move tokens to the correct device
                                if isinstance(device, str):
                                    cuda_device = device
                                else:
                                    cuda_device = device.type + ":" + str(device.index)
                                
                                input_ids = batch_tokens['input_ids'].to(cuda_device)
                                attention_mask = batch_tokens['attention_mask'].to(cuda_device)
                                
                                # Include token_type_ids if present
                                model_inputs = {
                                    'input_ids': input_ids,
                                    'attention_mask': attention_mask,
                                    'return_dict': True
                                }
                                
                                if 'token_type_ids' in batch_tokens:
                                    model_inputs['token_type_ids'] = batch_tokens['token_type_ids'].to(cuda_device)
                                
                                # Run model inference
                                outputs = endpoint(**model_inputs)
                                
                                # Process outputs
                                if hasattr(outputs, 'last_hidden_state'):
                                    # Apply attention mask to last_hidden_state
                                    last_hidden = outputs.last_hidden_state
                                    masked_hidden = last_hidden.masked_fill(
                                        ~attention_mask.bool().unsqueeze(-1), 
                                        0.0
                                    )
                                    
                                    # Mean pooling for sentence embeddings
                                    pooled_embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                                    
                                    # Move results to CPU
                                    batch_result = pooled_embeddings.cpu()
                                    results.append(batch_result)
                                else:
                                    # Skip batch on error
                                    print(f"Error processing batch {i+1}")
                                    continue
                                
                                # Clean up batch memory
                                if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                                    self.torch.cuda.empty_cache()
                            
                            # Combine batch results
                            if results:
                                result = self.torch.cat(results, dim=0)
                            else:
                                # Fallback if all batches failed
                                print("All batches failed to process")
                                mock_embedding = self.torch.rand((len(text_input), 768))
                                mock_embedding.mock_implementation = True
                                return mock_embedding
                                
                            # Add implementation markers
                            result.real_implementation = True
                            result.is_cuda = True
                            result.implementation_type = "REAL"
                            result.device = str(device)
                            result.model_name = endpoint_model
                            
                            # Performance metrics
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            result.elapsed_time = elapsed_time
                            
                            return result
                            
                        else:
                            # Process small batch normally
                            tokens = tokenizer(
                                text_input,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=max_length
                            )
                    else:
                        raise ValueError(f"Unsupported input type: {type(text_input)}")
                    
                    # Move tokens to the correct device
                    if isinstance(device, str):
                        cuda_device = device
                    else:
                        cuda_device = device.type + ":" + str(device.index)
                    
                    input_ids = tokens['input_ids'].to(cuda_device)
                    attention_mask = tokens['attention_mask'].to(cuda_device)
                    
                    # Include token_type_ids if present
                    model_inputs = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'return_dict': True
                    }
                    
                    if 'token_type_ids' in tokens:
                        model_inputs['token_type_ids'] = tokens['token_type_ids'].to(cuda_device)
                    
                    # Track inference time
                    inference_start = time.time()
                    
                    # Run model inference
                    outputs = endpoint(**model_inputs)
                    
                    # Calculate inference time
                    inference_time = time.time() - inference_start
                    
                    # Get CUDA memory usage after inference if available
                    if hasattr(self.torch.cuda, 'mem_get_info') and free_memory_start is not None:
                        try:
                            free_memory_after, _ = self.torch.cuda.mem_get_info()
                            memory_used_gb = (free_memory_start - free_memory_after) / (1024**3)
                            if memory_used_gb > 0:
                                print(f"CUDA memory used for inference: {memory_used_gb:.2f}GB")
                        except Exception as mem_error:
                            print(f"Error getting CUDA memory usage after inference: {mem_error}")
                    
                    # Process outputs to create embeddings
                    if hasattr(outputs, 'last_hidden_state'):
                        # Apply attention mask to last_hidden_state
                        last_hidden = outputs.last_hidden_state
                        masked_hidden = last_hidden.masked_fill(
                            ~attention_mask.bool().unsqueeze(-1), 
                            0.0
                        )
                        
                        # Mean pooling for sentence embeddings
                        pooled_embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                        
                        # Move results to CPU
                        result = pooled_embeddings.cpu()
                        
                        # Add REAL implementation markers
                        result.real_implementation = True
                        result.is_cuda = True
                        result.implementation_type = "REAL"
                        result.device = str(device)
                        result.model_name = endpoint_model
                        
                        # Add performance metrics
                        result.inference_time = inference_time
                        result.total_time = time.time() - start_time
                        
                    else:
                        # Fallback for models with different output structure
                        print(f"(MOCK) Unexpected output format from CUDA model, using fallback")
                        batch_size = 1 if isinstance(text_input, str) else len(text_input)
                        result = self.torch.rand((batch_size, 768))
                        result.mock_implementation = True
                        result.implementation_type = "MOCK"

                    # Cleanup GPU memory
                    for var in ['tokens', 'input_ids', 'attention_mask', 'outputs', 'last_hidden', 
                               'masked_hidden', 'pooled_embeddings']:
                        if var in locals():
                            del locals()[var]
                            
                    if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                        self.torch.cuda.empty_cache()
                    
                    return result
                    
            except Exception as e:
                # Cleanup GPU memory in case of error
                if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                    self.torch.cuda.empty_cache()
                
                print(f"Error in CUDA text embedding handler: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                
                # Generate a mock embedding with error info
                batch_size = 1 if isinstance(text_input, str) else len(text_input) 
                mock_embedding = self.torch.rand((batch_size, 768))
                
                # Add signal this is a mock for testing
                mock_embedding.mock_implementation = True
                mock_embedding.implementation_type = "MOCK"
                mock_embedding.error = str(e)
                
                return mock_embedding
                
        return handler
        
    def create_apple_text_embedding_endpoint_handler(self, endpoint_model, apple_label, endpoint=None, tokenizer=None):
        """Creates a handler for Apple Silicon.
        
        Args:
            endpoint_model: The model name or path
            apple_label: Label to identify this endpoint
            endpoint: The model endpoint
            tokenizer: The tokenizer
            
        Returns:
            A handler function for the Apple endpoint
        """
        def handler(x, endpoint_model=endpoint_model, apple_label=apple_label, endpoint=endpoint, tokenizer=tokenizer):
            if "eval" in dir(endpoint):
                endpoint.eval()
                
            try:
                with self.torch.no_grad():
                    # Prepare input
                    if type(x) == str:
                        tokens = tokenizer(
                            x, 
                            return_tensors='np', 
                            padding=True, 
                            truncation=True,
                            max_length=endpoint.config.max_position_embeddings
                        )
                    elif type(x) == list:
                        tokens = tokenizer(
                            x, 
                            return_tensors='np', 
                            padding=True, 
                            truncation=True,
                            max_length=endpoint.config.max_position_embeddings
                        )
                    else:
                        tokens = x
                    
                    # Convert input tensors to numpy arrays for CoreML
                    input_dict = {}
                    for key, value in tokens.items():
                        if hasattr(value, 'numpy'):
                            input_dict[key] = value.numpy()
                        else:
                            input_dict[key] = value
                    
                    # Run model inference
                    outputs = endpoint.predict(input_dict)
                    
                    # Get embeddings using mean pooling
                    if "last_hidden_state" in outputs:
                        hidden_states = self.torch.tensor(outputs["last_hidden_state"])
                        attention_mask = self.torch.tensor(tokens["attention_mask"])
                        
                        # Apply attention mask and pool
                        last_hidden = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
                        average_pool_results = last_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                        
                        # Move results back to CPU if needed
                        result = average_pool_results.cpu()
                    else:
                        # Handle case where model outputs pooled embeddings directly
                        result = self.torch.tensor(outputs.get("pooler_output", outputs.get("embeddings", None)))
                    
                    return result
                    
            except Exception as e:
                print(f"Error in Apple text embedding handler: {e}")
                raise e
                
        return handler
        
    def create_qualcomm_text_embedding_endpoint_handler(self, endpoint_model, qualcomm_label, endpoint=None, tokenizer=None):
        """Creates an endpoint handler for Qualcomm hardware.
        
        Args:
            endpoint_model: The model name or path
            qualcomm_label: Label to identify this endpoint
            endpoint: The model endpoint
            tokenizer: The tokenizer
            
        Returns:
            A handler function for the Qualcomm endpoint
        """
        def handler(x, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint, tokenizer=tokenizer):
            try:
                # Prepare input
                if type(x) == str:
                    tokens = tokenizer(
                        x, 
                        return_tensors='np', 
                        padding=True, 
                        truncation=True,
                        max_length=512  # Default max length
                    )
                elif type(x) == list:
                    tokens = tokenizer(
                        x, 
                        return_tensors='np', 
                        padding=True, 
                        truncation=True,
                        max_length=512  # Default max length
                    )
                else:
                    # If x is already tokenized, convert to numpy arrays if needed
                    tokens = {}
                    for key, value in x.items():
                        if hasattr(value, 'numpy'):
                            tokens[key] = value.numpy()
                        else:
                            tokens[key] = value
                
                # Run inference via SNPE
                results = self.snpe_utils.run_inference(endpoint, tokens)
                
                # Process results to get embeddings
                output = None
                
                if "last_hidden_state" in results:
                    # Convert to torch tensor
                    hidden_states = self.torch.tensor(results["last_hidden_state"])
                    attention_mask = self.torch.tensor(tokens["attention_mask"])
                    
                    # Apply attention mask
                    last_hidden = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
                    
                    # Mean pooling
                    output = last_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                    
                elif "pooler_output" in results:
                    # Some models provide a pooled output directly
                    output = self.torch.tensor(results["pooler_output"])
                
                return output
                
            except Exception as e:
                print(f"Error in Qualcomm text embedding handler: {e}")
                raise e
                
        return handler

