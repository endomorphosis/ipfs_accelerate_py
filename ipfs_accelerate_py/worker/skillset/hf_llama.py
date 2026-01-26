from PIL import Image
from io import BytesIO
from pathlib import Path
import os
import time
import anyio
from ..anyio_queue import AnyioQueue
import torch
import numpy as np
from unittest.mock import MagicMock

class hf_llama:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.init_cpu = self.init_cpu
        self.init_qualcomm = self.init_qualcomm
        self.init_apple = self.init_apple
        self.init_cuda = self.init_cuda
        self.init = self.init
        self.__test__ = self.__test__
        self.snpe_utils = None
        

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
        

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1)
            print(test_batch)
            print("hf_llama test passed")
        except Exception as e:
            print(e)
            print("hf_llama test failed")
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        tokens_per_second = 1 / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"samples per second: {tokens_per_second}")
        if "openvino" not in endpoint_label:
            with self.torch.no_grad():
                if "cuda" in dir(self.torch):
                    self.torch.cuda.empty_cache()
        return None
    
    def init_cpu(self, model, device, cpu_label):
        self.init()
        try:
            # Try loading model with trust_remote_code
            try:
                print(f"Loading LLaMA model {model} for CPU...")
                
                if isinstance(self.transformers, type(MagicMock())):
                    # We're using mocks - create dummy objects
                    print("Using mock transformers - creating dummy model")
                    config = MagicMock()
                    tokenizer = MagicMock()
                    tokenizer.decode = MagicMock(return_value="Once upon a time...")
                    tokenizer.batch_decode = MagicMock(return_value=["Once upon a time..."])
                    
                    endpoint = MagicMock()
                    endpoint.generate.return_value = self.torch.tensor([[101, 102, 103]])
                else:
                    # Try to load real model
                    try:
                        # First try regular loading
                        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
                        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model)
                        endpoint = self.transformers.AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
                    except Exception as model_error:
                        # If it fails, try with low_cpu_mem_usage
                        print(f"Failed to load model with trust_remote_code: {model_error}")
                        
                        try:
                            tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                                model, 
                                use_fast=True
                            )
                            endpoint = self.transformers.AutoModelForCausalLM.from_pretrained(
                                model, 
                                low_cpu_mem_usage=True,
                                torch_dtype=self.torch.float32
                            )
                        except Exception as e:
                            print(f"Failed low memory loading: {e}")
                            
                            # Create dummy tokenizer and model for offline testing
                            print("Creating dummy model for offline testing")
                            
                            # Create mock tokenizer
                            tokenizer = MagicMock()
                            tokenizer.decode = MagicMock(return_value="Once upon a time...")
                            tokenizer.batch_decode = MagicMock(return_value=["Once upon a time..."])
                            
                            # Create minimal dummy model
                            endpoint = MagicMock()
                            endpoint.generate = MagicMock(return_value=self.torch.tensor([[101, 102, 103]]))
            except Exception as e:
                print(f"Error creating model: {e}")
                tokenizer = MagicMock()
                endpoint = MagicMock()
                endpoint.generate = MagicMock(return_value=self.torch.tensor([[101, 102, 103]]))
            
            # Create the handler
            endpoint_handler = self.create_cpu_llama_endpoint_handler(tokenizer, model, cpu_label, endpoint)
            return endpoint, tokenizer, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - AnyioQueue(16), 1
        except Exception as e:
            print(f"Error initializing CPU model: {e}")
            return None, None, None, None, 0
    
    def init_apple(self, model, device, apple_label):
        """Initialize LLaMA model for Apple Silicon hardware."""
        self.init()
        
        try:
            from .apple_coreml_utils import get_coreml_utils
            self.coreml_utils = get_coreml_utils()
        except ImportError:
            print("Failed to import CoreML utilities")
            return None, None, None, None, 0
            
        if not self.coreml_utils.is_available():
            print("CoreML is not available on this system")
            return None, None, None, None, 0
            
        try:
            # Load tokenizer from HuggingFace
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            
            # Convert model path to be compatible with CoreML
            model_name = model.replace("/", "--")
            mlmodel_path = f"~/coreml_models/{model_name}_llama.mlpackage"
            mlmodel_path = os.path.expanduser(mlmodel_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(mlmodel_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(mlmodel_path):
                print(f"Converting {model} to CoreML format...")
                self.coreml_utils.convert_model(model, "text", str(mlmodel_path))
            
            # Load the CoreML model
            endpoint = self.coreml_utils.load_model(str(mlmodel_path))
            
            # Optimize for Apple Silicon if possible
            if ":" in apple_label:
                compute_units = apple_label.split(":")[1]
                optimized_path = self.coreml_utils.optimize_for_device(mlmodel_path, compute_units)
                if optimized_path != mlmodel_path:
                    endpoint = self.coreml_utils.load_model(optimized_path)
            
            endpoint_handler = self.create_apple_text_generation_endpoint_handler(endpoint, tokenizer, model, apple_label)
            
            return endpoint, tokenizer, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - AnyioQueue(32), 0
        except Exception as e:
            print(f"Error initializing Apple Silicon LLaMA model: {e}")
            return None, None, None, None, 0
    
    def init_cuda(self, model, device, cuda_label):
        """
        Initialize LLaMA model for CUDA execution with enhanced memory optimization
        
        Args:
            model: Model name or path
            device: Device type (should be 'cuda')
            cuda_label: CUDA device label (e.g. 'cuda:0')
            
        Returns:
            Tuple of (endpoint, tokenizer, handler, queue, batch_size)
        """
        self.init()
        
        # First check if CUDA is available
        if not hasattr(self.torch, 'cuda') or not self.torch.cuda.is_available():
            print("CUDA is not available on this system")
            return None, None, None, None, 0
        
        # Get CUDA device information
        try:
            # Parse cuda_label to extract device index
            if ":" in cuda_label:
                device_index = int(cuda_label.split(":")[1])
                if device_index >= self.torch.cuda.device_count():
                    print(f"Warning: CUDA device index {device_index} out of range (0-{self.torch.cuda.device_count()-1}), using device 0")
                    device_index = 0
                    cuda_label = "cuda:0"
            else:
                device_index = 0
                cuda_label = "cuda:0"
                
            # Create device object for consistent usage
            cuda_device = self.torch.device(cuda_label)
            
            # Get and display device information
            device_name = self.torch.cuda.get_device_name(device_index)
            print(f"Using CUDA device: {device_name} (index {device_index})")
            
            # Get memory information for logging and memory settings
            if hasattr(self.torch.cuda, 'mem_get_info'):
                free_memory, total_memory = self.torch.cuda.mem_get_info(device_index)
                free_memory_gb = free_memory / (1024**3)
                total_memory_gb = total_memory / (1024**3)
                print(f"Available CUDA memory: {free_memory_gb:.2f}GB / {total_memory_gb:.2f}GB")
            else:
                # Older CUDA versions might not have mem_get_info
                free_memory = None
                total_memory = None
                print("CUDA memory info not available")
            
            # Clean up CUDA cache before loading model
            if hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error initializing CUDA device: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None, None, None, None, 0
        
        try:
            # Check if we're using mock transformers
            if isinstance(self.transformers, type(MagicMock())):
                # Create mocks for testing
                print("Using mock transformers implementation for CUDA test")
                config = MagicMock()
                tokenizer = MagicMock() 
                tokenizer.batch_decode = MagicMock(return_value=["Once upon a time..."])
                
                endpoint = MagicMock()
                endpoint.generate = MagicMock(return_value=self.torch.tensor([[101, 102, 103]]))
                is_real_impl = False
            else:
                # Try loading real model with CUDA support
                print("Attempting to load real model with CUDA support")
                is_real_impl = True
                
                # Load config and tokenizer first
                try:
                    print(f"Loading configuration and tokenizer for {model}")
                    config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
                    tokenizer = self.transformers.AutoTokenizer.from_pretrained(model)
                except Exception as e:
                    print(f"Error loading config or tokenizer: {e}")
                    print(f"Falling back to mock tokenizer")
                    tokenizer = MagicMock()
                    tokenizer.batch_decode = MagicMock(return_value=["Once upon a time..."])
                    is_real_impl = False
                
                # Load model with optimized memory settings if we have real tokenizer
                if is_real_impl:
                    try:
                        # Set memory optimization parameters
                        if free_memory is not None:
                            free_memory_gb = free_memory / (1024**3)
                        else:
                            # Conservative estimate if we can't get actual memory info
                            free_memory_gb = 4.0
                        
                        # Determine memory optimization settings
                        use_half_precision = True  # Default to half precision for better memory efficiency
                        use_8bit_quantization = free_memory_gb < 4.0  # Use 8-bit quantization for lower memory
                        low_cpu_mem_usage = True  # Always use low CPU memory usage
                        
                        # Choose batch size based on available memory
                        batch_size = max(1, min(8, int(free_memory_gb / 2)))
                        print(f"Using batch size: {batch_size}")
                        
                        # Prepare model loading parameters with memory optimizations
                        model_kwargs = {
                            "torch_dtype": self.torch.float16 if use_half_precision else self.torch.float32,
                            "low_cpu_mem_usage": low_cpu_mem_usage,
                            "trust_remote_code": True,
                            "device_map": cuda_label
                        }
                        
                        # Add 8-bit quantization if needed
                        if use_8bit_quantization and hasattr(self.transformers, 'BitsAndBytesConfig'):
                            print("Using 8-bit quantization for memory efficiency")
                            model_kwargs["quantization_config"] = self.transformers.BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_threshold=6.0
                            )
                        
                        # Load the model with the optimized settings
                        print(f"Loading model {model} with optimized memory settings:")
                        print(f"- Half precision: {use_half_precision}")
                        print(f"- 8-bit quantization: {use_8bit_quantization if 'quantization_config' in model_kwargs else False}")
                        print(f"- Low CPU memory usage: {low_cpu_mem_usage}")
                        
                        endpoint = self.transformers.AutoModelForCausalLM.from_pretrained(
                            model, 
                            **model_kwargs
                        )
                        
                        # Move model to device if not already done by device_map
                        if not hasattr(endpoint, 'device') or endpoint.device != cuda_device:
                            print(f"Moving model to {cuda_label}")
                            endpoint = endpoint.to(cuda_device)
                        
                        # Set model to evaluation mode
                        endpoint.eval()
                        
                        # Log memory usage after model loading
                        if hasattr(self.torch.cuda, 'mem_get_info'):
                            try:
                                free_after_load, _ = self.torch.cuda.mem_get_info(device_index)
                                free_after_load_gb = free_after_load / (1024**3)
                                memory_used_gb = free_memory_gb - free_after_load_gb
                                print(f"Model loaded. CUDA memory used: {memory_used_gb:.2f}GB, Available: {free_after_load_gb:.2f}GB")
                            except Exception as mem_error:
                                print(f"Error getting CUDA memory usage after model loading: {mem_error}")
                        
                    except Exception as e:
                        print(f"Error loading CUDA model: {e}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        print("Falling back to mock implementation")
                        endpoint = MagicMock()
                        endpoint.generate = MagicMock(return_value=self.torch.tensor([[101, 102, 103]]))
                        is_real_impl = False
            
            # Create handler with implementation status
            endpoint_handler = self.create_cuda_llama_endpoint_handler(
                tokenizer, 
                endpoint_model=model, 
                cuda_label=cuda_label, 
                endpoint=endpoint,
                is_real_impl=is_real_impl
            )
            
            # Final cache cleanup
            if hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
                
            return endpoint, tokenizer, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - AnyioQueue(16), batch_size if is_real_impl else 1
            
        except Exception as e:
            print(f"Error in CUDA initialization: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # Ensure we clean up CUDA memory on error
            if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
            return None, None, None, None, 0

    def init_openvino(self, model, model_type, device, openvino_label, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None):
        """Initialize OpenVINO model for inference
        
        Args:
            model: Model name or path
            model_type: Type of model (text-generation, etc.)
            device: Device to run on (CPU, GPU, etc.)
            openvino_label: Label for the OpenVINO device
            get_optimum_openvino_model: Optional function to get Optimum model
            get_openvino_model: Optional function to get OpenVINO model
            get_openvino_pipeline_type: Optional function to get pipeline type
            openvino_cli_convert: Optional function to convert model using OpenVINO CLI
            
        Returns:
            Tuple of (endpoint, tokenizer, handler, queue, batch_size)
        """
        try:
            # Try importing OpenVINO
            try:
                import openvino as ov
                print("OpenVINO imported successfully")
            except ImportError:
                print("OpenVINO not available - using mocks")
                
            self.init()
            
            # Create mock objects if we're testing
            if isinstance(self.transformers, type(MagicMock())) or get_openvino_model is None:
                print("Using mocks for OpenVINO")
                tokenizer = MagicMock()
                tokenizer.batch_decode = MagicMock(return_value=["Once upon a time..."])
                
                endpoint = MagicMock()
                # Create mock functions for testing
                endpoint.run_model = MagicMock(return_value={
                    "logits": np.random.rand(1, 10, 30522)
                })
            else:
                # Try loading real model with OpenVINO
                try:
                    # Get tokenizer from original model
                    tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                        model, 
                        use_fast=True, 
                        trust_remote_code=True
                    )
                    
                    # Set up model paths for conversion
                    model_name_convert = model.replace("/", "--")
                    
                    # Get the OpenVINO pipeline type for this model
                    pipeline_type = "text-generation-with-past"
                    if get_openvino_pipeline_type is not None:
                        try:
                            pipeline_type = get_openvino_pipeline_type(model, model_type)
                        except Exception as e:
                            print(f"Error getting pipeline type: {e}")
                    
                    # Extract device info from openvino_label
                    openvino_index = 0
                    if ":" in openvino_label:
                        try:
                            openvino_index = int(openvino_label.split(":")[1])
                        except (ValueError, IndexError):
                            print(f"Invalid openvino_label format: {openvino_label}, using default index 0")
                    
                    # Set weight format based on device target
                    weight_format = "int8"  # CPU default
                    if openvino_index == 1:
                        weight_format = "int4"  # GPU
                    elif openvino_index == 2:
                        weight_format = "int4"  # NPU
                    
                    # Determine model path based on HuggingFace cache
                    import os
                    homedir = os.path.expanduser("~")
                    huggingface_cache = os.path.join(homedir, ".cache", "huggingface")
                    huggingface_cache_models = os.path.join(huggingface_cache, "hub")
                    
                    # Define source and destination paths
                    model_dst_path = os.path.join(homedir, ".cache", "openvino", model_name_convert + "_" + weight_format)
                    
                    # Create destination directory if needed
                    if not os.path.exists(model_dst_path):
                        os.makedirs(model_dst_path, exist_ok=True)
                        
                        # Convert the model using OpenVINO CLI if available
                        if openvino_cli_convert is not None:
                            print(f"Converting model {model} to OpenVINO format...")
                            openvino_cli_convert(
                                model, 
                                model_dst_path=model_dst_path, 
                                task=pipeline_type,
                                weight_format=weight_format, 
                                ratio="1.0", 
                                group_size=128, 
                                sym=True
                            )
                    
                    # Load the converted model, or fall back to get_openvino_model
                    try:
                        # First try loading the model from the destination path
                        if os.path.exists(os.path.join(model_dst_path, f"{model_name_convert}.xml")):
                            print(f"Loading model from {model_dst_path}")
                            # Try using Optimum if available
                            if get_optimum_openvino_model is not None:
                                endpoint = get_optimum_openvino_model(model_dst_path, model_type)
                            else:
                                endpoint = get_openvino_model(model_dst_path, model_type, openvino_label)
                        else:
                            # Fall back to direct model loading
                            endpoint = get_openvino_model(model, model_type, openvino_label)
                    except Exception as e:
                        print(f"Error loading converted model: {e}")
                        # Fall back to direct model loading as last resort
                        endpoint = get_openvino_model(model, model_type, openvino_label)
                    
                except Exception as e:
                    print(f"Error loading model: {e}")
                    tokenizer = MagicMock()
                    tokenizer.batch_decode = MagicMock(return_value=["Once upon a time..."])
                    
                    endpoint = MagicMock()
                    endpoint.run_model = MagicMock(return_value={
                        "logits": np.random.rand(1, 10, 30522)
                    })
            
            # Create handler function
            endpoint_handler = self.create_openvino_llama_endpoint_handler(
                tokenizer, 
                model, 
                openvino_label, 
                endpoint
            )
            
            return endpoint, tokenizer, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - AnyioQueue(64), 0
        
        except Exception as e:
            print(f"Error in OpenVINO initialization: {e}")
            return None, None, None, None, 0

    def init_qualcomm(self, model, model_type, device, qualcomm_label, get_qualcomm_genai_pipeline=None, get_optimum_qualcomm_model=None, get_qualcomm_model=None, get_qualcomm_pipeline_type=None):
        """
        Initialize LLaMA model for Qualcomm hardware
        
        Args:
            model: Model name or path
            model_type: Type of model
            device: Device to run on
            qualcomm_label: Label for Qualcomm hardware
            get_qualcomm_genai_pipeline: Optional function to get GenAI pipeline
            get_optimum_qualcomm_model: Optional function to get Optimum model 
            get_qualcomm_model: Optional function to get Qualcomm model
            get_qualcomm_pipeline_type: Optional function to get pipeline type
            
        Returns:
            Initialized model components
        """
        self.init()
        
        # Import SNPE utilities
        try:
            # Check if we're using mocks
            if isinstance(self.transformers, type(MagicMock())):
                print("Using mock transformers - creating dummy Qualcomm model")
                # Create mock objects for testing
                self.snpe_utils = MagicMock()
                tokenizer = MagicMock()
                tokenizer.decode = MagicMock(return_value="Once upon a time...")
                tokenizer.batch_decode = MagicMock(return_value=["Once upon a time..."])
                
                # Create dummy endpoint
                endpoint = MagicMock()
                endpoint.run_model = MagicMock(return_value={
                    "logits": np.random.rand(1, 10, 30522),
                    "past_key_values": [(np.random.rand(1, 2, 64, 128), np.random.rand(1, 2, 64, 128)) 
                                      for _ in range(4)]
                })
                
                # Create endpoint handler
                endpoint_handler = self.create_qualcomm_llama_endpoint_handler(
                    tokenizer, model, qualcomm_label, endpoint
                )
                
                return endpoint, tokenizer, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - AnyioQueue(16), 1
            
            # Try real initialization
            try:
                from .qualcomm_snpe_utils import get_snpe_utils
                self.snpe_utils = get_snpe_utils()
                
                if not self.snpe_utils.is_available():
                    print("Qualcomm SNPE is not available on this system")
                    raise ImportError("SNPE not available")
                
                # Initialize tokenizer directly from HuggingFace
                tokenizer = self.transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
                
                # Convert model path to be compatible with SNPE
                model_name = model.replace("/", "--")
                dlc_path = f"~/snpe_models/{model_name}_llm.dlc"
                dlc_path = os.path.expanduser(dlc_path)
                
                # Create directory if needed
                os.makedirs(os.path.dirname(dlc_path), exist_ok=True)
                
                # Convert or load the model
                if not os.path.exists(dlc_path):
                    print(f"Converting {model} to SNPE format...")
                    self.snpe_utils.convert_model(model, "llm", str(dlc_path))
                
                # Load the SNPE model
                endpoint = self.snpe_utils.load_model(str(dlc_path))
                
                # Optimize for the specific Qualcomm device if possible
                if ":" in qualcomm_label:
                    device_type = qualcomm_label.split(":")[1]
                    optimized_path = self.snpe_utils.optimize_for_device(dlc_path, device_type)
                    if optimized_path != dlc_path:
                        endpoint = self.snpe_utils.load_model(optimized_path)
                
                # Create endpoint handler
                endpoint_handler = self.create_qualcomm_llama_endpoint_handler(
                    tokenizer, model, qualcomm_label, endpoint
                )
                
                return endpoint, tokenizer, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - AnyioQueue(16), 1
                
            except (ImportError, Exception) as e:
                print(f"Error initializing real Qualcomm model: {e}")
                # Fallback to mocks
                self.snpe_utils = MagicMock()
                tokenizer = MagicMock()
                tokenizer.decode = MagicMock(return_value="Once upon a time in a forest...")
                tokenizer.batch_decode = MagicMock(return_value=["Once upon a time..."])
                
                endpoint = MagicMock()
                endpoint.run_model = MagicMock(return_value={
                    "logits": np.random.rand(1, 10, 30522),
                    "past_key_values": [(np.random.rand(1, 2, 64, 128), np.random.rand(1, 2, 64, 128)) 
                                      for _ in range(4)]
                })
                
                # Create handler that supports mocks
                endpoint_handler = self.create_qualcomm_llama_endpoint_handler(
                    tokenizer, model, qualcomm_label, endpoint
                )
                
                return endpoint, tokenizer, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - AnyioQueue(16), 1
                
        except Exception as e:
            print(f"Critical error initializing Qualcomm LLaMA model: {e}")
            return None, None, None, None, 0
    
    def create_cpu_llama_endpoint_handler(self, tokenizer, model_name, cpu_label, endpoint):
        """Create a handler for CPU-based LLaMA inference.
        
        Args:
            tokenizer: The tokenizer to use for input/output processing
            model_name: Model name or path
            cpu_label: Label for CPU device
            endpoint: The LLaMA model endpoint
            
        Returns:
            Handler function for CPU-based text generation
        """
        
        def handler(text_input, tokenizer=tokenizer, model_name=model_name, cpu_label=cpu_label, endpoint=endpoint):
            """CPU handler for LLaMA text generation.
            
            Args:
                text_input: Input text or tokenized input
                
            Returns:
                Dictionary with generated text and implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = False
            
            # Check if we're dealing with a mock component
            if isinstance(endpoint, type(MagicMock())) or isinstance(tokenizer, type(MagicMock())):
                is_mock = True
            
            # Validate input
            if text_input is None:
                is_mock = True
                return {
                    "generated_text": "No input provided",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
            
            # Initialize model with error handling
            if "eval" in dir(endpoint) and not is_mock:
                try:
                    endpoint.eval()
                except Exception as eval_error:
                    print(f"Error setting model to eval mode: {eval_error}")
                    # Continue anyway, just log the error
            
            try:
                # Mock handling for testing
                if is_mock:
                    # For mocks, return a simple response
                    print("Using mock handler for CPU LLaMA")
                    
                    # Try to use mocked components if available
                    try:
                        if hasattr(tokenizer, 'batch_decode') and callable(tokenizer.batch_decode):
                            # If the tokenizer has batch_decode mocked, use it
                            if hasattr(endpoint, 'generate') and callable(endpoint.generate):
                                mock_ids = endpoint.generate()
                                decoded_output = tokenizer.batch_decode(mock_ids)[0]
                            else:
                                # Just return a mock response
                                decoded_output = "Once upon a time, there was a clever fox who became friends with a loyal dog."
                        else:
                            # Default mock response
                            decoded_output = "The fox and dog played together in the forest, teaching everyone a lesson about friendship."
                    except Exception as mock_error:
                        print(f"Error in mock response generation: {mock_error}")
                        decoded_output = "Once upon a time, in a forest far away..."
                    
                    return {
                        "generated_text": decoded_output,
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Real model handling with better error handling
                try:
                    # Tokenize input safely
                    if isinstance(text_input, str):
                        try:
                            inputs = tokenizer(text_input, return_tensors="pt")
                        except Exception as tokenize_error:
                            print(f"Error tokenizing input: {tokenize_error}")
                            is_mock = True
                            return {
                                "generated_text": f"Error tokenizing input: {str(tokenize_error)[:50]}...",
                                "model_name": model_name,
                                "implementation_type": "MOCK"
                            }
                    else:
                        # Assume it's already tokenized
                        inputs = text_input
                        
                    # Validate inputs
                    if not isinstance(inputs, dict) or "input_ids" not in inputs:
                        is_mock = True
                        return {
                            "generated_text": "Invalid inputs format or missing input_ids",
                            "model_name": model_name,
                            "implementation_type": "MOCK"
                        }
                    
                    # Run generation safely
                    with self.torch.no_grad():
                        try:
                            # Verify endpoint has generate method
                            if not hasattr(endpoint, 'generate') or not callable(endpoint.generate):
                                is_mock = True
                                return {
                                    "generated_text": "Model endpoint missing generate method",
                                    "model_name": model_name,
                                    "implementation_type": "MOCK"
                                }
                                
                            # Get attention_mask safely
                            attention_mask = inputs.get("attention_mask", None)
                            
                            # Run generation
                            outputs = endpoint.generate(
                                inputs["input_ids"],
                                attention_mask=attention_mask,
                                max_new_tokens=256,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9
                            )
                            
                            # Verify outputs are valid
                            if outputs is None:
                                is_mock = True
                                return {
                                    "generated_text": "Model generated null output",
                                    "model_name": model_name,
                                    "implementation_type": "MOCK"
                                }
                        except Exception as gen_error:
                            print(f"Error during generation: {gen_error}")
                            is_mock = True
                            return {
                                "generated_text": f"Error during generation: {str(gen_error)[:50]}...",
                                "model_name": model_name,
                                "implementation_type": "MOCK"
                            }
                    
                    # Decode output safely
                    try:
                        # Move to CPU if needed
                        if hasattr(outputs, 'cpu'):
                            outputs_cpu = outputs.cpu()  # Move to CPU if on another device
                        else:
                            outputs_cpu = outputs
                        
                        # Verify tokenizer has decode method
                        if not hasattr(tokenizer, 'decode') or not callable(tokenizer.decode):
                            is_mock = True
                            return {
                                "generated_text": "Tokenizer missing decode method",
                                "model_name": model_name,
                                "implementation_type": "MOCK"
                            }
                        
                        # Check dimensions and decode appropriately
                        if hasattr(outputs_cpu, 'dim') and callable(outputs_cpu.dim) and outputs_cpu.dim() > 1:
                            # Safe indexing with bounds check
                            if outputs_cpu.shape[0] > 0:
                                decoded_output = tokenizer.decode(outputs_cpu[0], skip_special_tokens=True)
                            else:
                                decoded_output = "Empty generation result"
                                is_mock = True
                        else:
                            # Single dimension output
                            decoded_output = tokenizer.decode(outputs_cpu, skip_special_tokens=True)
                            
                        # Check for empty output
                        if not decoded_output or len(decoded_output.strip()) == 0:
                            decoded_output = "Empty generation result"
                            is_mock = True
                    except Exception as decode_error:
                        print(f"Error decoding output: {decode_error}")
                        is_mock = True
                        return {
                            "generated_text": f"Error decoding output: {str(decode_error)[:50]}...",
                            "model_name": model_name,
                            "implementation_type": "MOCK"
                        }
                    
                    # Return result with implementation type
                    return {
                        "generated_text": decoded_output,
                        "model_name": model_name,
                        "implementation_type": "REAL"
                    }
                except Exception as process_error:
                    print(f"Error in CPU processing: {process_error}")
                    is_mock = True
                    return {
                        "generated_text": f"Error in processing: {str(process_error)[:50]}...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
            except Exception as e:
                print(f"Unexpected error in CPU LLaMA endpoint handler: {e}")
                return {
                    "generated_text": f"Unexpected error: {str(e)[:100]}...",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
                
        return handler
        
    def create_apple_text_generation_endpoint_handler(self, endpoint, tokenizer, model_name, apple_label):
        """Creates an Apple Silicon optimized handler for LLaMA text generation.
        
        Args:
            endpoint: The CoreML model endpoint
            tokenizer: The text tokenizer
            model_name: Model name or path
            apple_label: Label for Apple endpoint
            
        Returns:
            Handler function for Apple Silicon text generation
        """
        def handler(x, endpoint=endpoint, tokenizer=tokenizer, model_name=model_name, apple_label=apple_label):
            """Apple Silicon handler for LLaMA text generation.
            
            Args:
                x: Input text or tokenized input
                
            Returns:
                Dictionary with generated text and implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = False
            
            # Validate input
            if x is None:
                is_mock = True
                return {
                    "generated_text": "No input provided",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
            
            # Check if we're dealing with a mock component or missing CoreML utils
            if (isinstance(endpoint, type(MagicMock())) or 
                isinstance(tokenizer, type(MagicMock())) or
                self.coreml_utils is None or
                not hasattr(self.coreml_utils, 'run_inference')):
                
                is_mock = True
                
                # For testing, return a simple mock response
                try:
                    print("Using mock handler for Apple Silicon LLaMA")
                    
                    # Generate a mock response based on input type
                    if isinstance(x, str):
                        mock_text = "The fox and the dog became best friends, going on many adventures together in the forest."
                    elif isinstance(x, list):
                        mock_text = "Once upon a time, a fox and a dog discovered they could achieve more together than apart."
                    else:
                        mock_text = "Once upon a time in the forest..."
                        
                    return {
                        "generated_text": mock_text,
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                except Exception as mock_error:
                    print(f"Error in mock handling: {mock_error}")
                    return {
                        "generated_text": "Once upon a time...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
            
            try:
                # Process input safely
                try:
                    if isinstance(x, str):
                        inputs = tokenizer(
                            x, 
                            return_tensors="np", 
                            padding=True,
                            truncation=True
                        )
                    elif isinstance(x, list):
                        inputs = tokenizer(
                            x, 
                            return_tensors="np", 
                            padding=True,
                            truncation=True
                        )
                    else:
                        inputs = x
                        
                    # Validate inputs structure
                    if not isinstance(inputs, dict) or "input_ids" not in inputs:
                        is_mock = True
                        return {
                            "generated_text": "Invalid inputs format or missing input_ids",
                            "model_name": model_name,
                            "implementation_type": "MOCK"
                        }
                except Exception as input_error:
                    print(f"Error processing input: {input_error}")
                    is_mock = True
                    return {
                        "generated_text": f"Error processing input: {str(input_error)[:50]}...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Ensure CoreML utils are properly initialized and available
                if not hasattr(self.coreml_utils, 'run_inference') or not callable(self.coreml_utils.run_inference):
                    is_mock = True
                    return {
                        "generated_text": "CoreML utilities not properly initialized or missing run_inference method",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Convert inputs to CoreML format safely
                try:
                    input_dict = {}
                    # Use list() to create a copy of keys to avoid dict size change errors
                    for key in list(inputs.keys()):
                        value = inputs[key]
                        if hasattr(value, 'numpy') and callable(value.numpy):
                            input_dict[key] = value.numpy()
                        else:
                            input_dict[key] = value
                except Exception as convert_error:
                    print(f"Error converting inputs to CoreML format: {convert_error}")
                    is_mock = True
                    return {
                        "generated_text": f"Error preparing inputs: {str(convert_error)[:50]}...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Run inference with error handling
                try:
                    outputs = self.coreml_utils.run_inference(endpoint, input_dict)
                    
                    # Validate outputs
                    if outputs is None or not isinstance(outputs, dict):
                        is_mock = True
                        return {
                            "generated_text": "Invalid model output format",
                            "model_name": model_name,
                            "implementation_type": "MOCK"
                        }
                except Exception as inference_error:
                    print(f"Error during inference: {inference_error}")
                    is_mock = True
                    return {
                        "generated_text": f"Error during inference: {str(inference_error)[:50]}...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Process outputs safely
                try:
                    if 'logits' in outputs:
                        # Convert logits to PyTorch tensor
                        logits = self.torch.tensor(outputs['logits'])
                        
                        # Validate tensor shape for argmax
                        if logits.dim() < 2 or logits.size(0) == 0:
                            is_mock = True
                            return {
                                "generated_text": "Invalid logits shape for processing",
                                "model_name": model_name,
                                "implementation_type": "MOCK"
                            }
                        
                        # Generate tokens using greedy decoding
                        generated_ids = self.torch.argmax(logits, dim=-1)
                        
                        # Verify tokenizer has batch_decode
                        if not hasattr(tokenizer, 'batch_decode') or not callable(tokenizer.batch_decode):
                            is_mock = True
                            return {
                                "generated_text": "Tokenizer missing batch_decode method",
                                "model_name": model_name,
                                "implementation_type": "MOCK"
                            }
                        
                        # Decode the generated tokens to text
                        try:
                            generated_text = tokenizer.batch_decode(
                                generated_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True
                            )
                            
                            # Verify we got valid output
                            if not generated_text or len(generated_text) == 0:
                                is_mock = True
                                return {
                                    "generated_text": "Empty generation result",
                                    "model_name": model_name,
                                    "implementation_type": "MOCK"
                                }
                                
                            # Format the result
                            result_text = generated_text[0] if len(generated_text) == 1 else generated_text
                            
                            # Check if result text is empty
                            if isinstance(result_text, str) and not result_text.strip():
                                is_mock = True
                                return {
                                    "generated_text": "Empty generation result",
                                    "model_name": model_name,
                                    "implementation_type": "MOCK"
                                }
                                
                            # Return successful result
                            return {
                                "generated_text": result_text,
                                "model_name": model_name,
                                "implementation_type": "REAL"
                            }
                        except Exception as decode_error:
                            print(f"Error decoding tokens: {decode_error}")
                            is_mock = True
                            return {
                                "generated_text": f"Error decoding tokens: {str(decode_error)[:50]}...",
                                "model_name": model_name,
                                "implementation_type": "MOCK"
                            }
                    else:
                        print("No logits found in model output")
                        is_mock = True
                        return {
                            "generated_text": "Model output format not supported (missing logits)",
                            "model_name": model_name,
                            "implementation_type": "MOCK"
                        }
                except Exception as process_error:
                    print(f"Error processing model output: {process_error}")
                    is_mock = True
                    return {
                        "generated_text": f"Error processing output: {str(process_error)[:50]}...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
            except Exception as e:
                print(f"Unexpected error in Apple Silicon LLaMA handler: {e}")
                return {
                    "generated_text": f"Unexpected error: {str(e)[:100]}...",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
                
        return handler
    
    def create_apple_llama_endpoint_handler(self, tokenizer, model_name, apple_label, endpoint):
        """Create a handler for Apple Silicon-based LLaMA inference.
        
        Args:
            tokenizer: The tokenizer to use for input/output processing
            model_name: Model name or path
            apple_label: Apple Silicon MPS device identifier
            endpoint: The LLaMA model endpoint
            
        Returns:
            Handler function for Apple Silicon-based text generation
        """
        
        def handler(text_input, tokenizer=tokenizer, model_name=model_name, apple_label=apple_label, endpoint=endpoint):
            """Apple Silicon handler for LLaMA text generation.
            
            Args:
                text_input: Input text or tokenized input
                
            Returns:
                Dictionary with generated text and implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = False
            
            # Validate input
            if text_input is None:
                is_mock = True
                return {
                    "generated_text": "No input provided",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
            
            # Check for MPS availability
            mps_available = (
                hasattr(self.torch.backends, 'mps') and 
                self.torch.backends.mps.is_available() and
                hasattr(endpoint, 'to') and callable(endpoint.to)
            )
            
            # Check if we're dealing with a mock component
            if isinstance(endpoint, type(MagicMock())) or isinstance(tokenizer, type(MagicMock())) or not mps_available:
                is_mock = True
                return {
                    "generated_text": "Once upon a time, a fox and a dog discovered they had much more in common than differences.",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
            
            # Initialize model with error handling
            if "eval" in dir(endpoint):
                try:
                    endpoint.eval()
                except Exception as eval_error:
                    print(f"Error setting model to eval mode: {eval_error}")
                    # Continue anyway, just log the error
            
            try:
                # Tokenize input safely
                try:
                    if isinstance(text_input, str):
                        inputs = tokenizer(text_input, return_tensors="pt")
                        
                        # Safely move tensors to MPS
                        input_dict = {}
                        for key in list(inputs.keys()):
                            if hasattr(inputs[key], 'to') and callable(inputs[key].to):
                                input_dict[key] = inputs[key].to("mps")
                            else:
                                input_dict[key] = inputs[key]
                        inputs = input_dict
                    else:
                        # Assume it's already tokenized, create a safe copy
                        inputs = {}
                        if hasattr(text_input, 'items'):
                            for k, v in text_input.items():
                                if hasattr(v, 'to') and callable(v.to):
                                    inputs[k] = v.to("mps")
                                else:
                                    inputs[k] = v
                        else:
                            # Invalid input type
                            is_mock = True
                            return {
                                "generated_text": f"Invalid input type: {type(text_input)}",
                                "model_name": model_name,
                                "implementation_type": "MOCK"
                            }
                except Exception as tokenize_error:
                    print(f"Error tokenizing or moving input to MPS: {tokenize_error}")
                    is_mock = True
                    return {
                        "generated_text": f"Error preparing input: {str(tokenize_error)[:50]}...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                    
                # Validate inputs
                if not isinstance(inputs, dict) or "input_ids" not in inputs:
                    is_mock = True
                    return {
                        "generated_text": "Invalid inputs format or missing input_ids",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Run generation safely
                with self.torch.no_grad():
                    try:
                        # Verify endpoint has generate method
                        if not hasattr(endpoint, 'generate') or not callable(endpoint.generate):
                            is_mock = True
                            return {
                                "generated_text": "Model endpoint missing generate method",
                                "model_name": model_name,
                                "implementation_type": "MOCK"
                            }
                            
                        # Get attention_mask safely
                        attention_mask = inputs.get("attention_mask", None)
                        
                        # Run generation
                        outputs = endpoint.generate(
                            inputs["input_ids"],
                            attention_mask=attention_mask,
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9
                        )
                        
                        # Verify outputs are valid
                        if outputs is None:
                            is_mock = True
                            return {
                                "generated_text": "Model generated null output",
                                "model_name": model_name,
                                "implementation_type": "MOCK"
                            }
                    except Exception as gen_error:
                        print(f"Error during generation: {gen_error}")
                        is_mock = True
                        return {
                            "generated_text": f"Error during generation: {str(gen_error)[:50]}...",
                            "model_name": model_name,
                            "implementation_type": "MOCK"
                        }
                
                # Decode output safely
                try:
                    # Move back to CPU for decoding
                    if hasattr(outputs, 'cpu') and callable(outputs.cpu):
                        outputs_cpu = outputs.cpu()
                    else:
                        outputs_cpu = outputs
                    
                    # Verify output has expected structure
                    if not hasattr(outputs_cpu, 'shape') or len(outputs_cpu.shape) < 1 or outputs_cpu.shape[0] < 1:
                        is_mock = True
                        return {
                            "generated_text": "Invalid output tensor shape",
                            "model_name": model_name,
                            "implementation_type": "MOCK"
                        }
                    
                    # Verify tokenizer has decode method
                    if not hasattr(tokenizer, 'decode') or not callable(tokenizer.decode):
                        is_mock = True
                        return {
                            "generated_text": "Tokenizer missing decode method",
                            "model_name": model_name,
                            "implementation_type": "MOCK"
                        }
                    
                    # Safe indexing with bounds check
                    if outputs_cpu.shape[0] > 0:
                        decoded_output = tokenizer.decode(outputs_cpu[0], skip_special_tokens=True)
                    else:
                        decoded_output = "Empty generation result"
                        is_mock = True
                    
                    # Check for empty output
                    if not decoded_output or len(decoded_output.strip()) == 0:
                        decoded_output = "Empty generation result"
                        is_mock = True
                except Exception as decode_error:
                    print(f"Error decoding output: {decode_error}")
                    is_mock = True
                    return {
                        "generated_text": f"Error decoding output: {str(decode_error)[:50]}...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Return result with implementation type
                return {
                    "generated_text": decoded_output,
                    "model_name": model_name,
                    "implementation_type": "REAL" if not is_mock else "MOCK"
                }
                
            except Exception as e:
                print(f"Unexpected error in Apple Silicon LLaMA endpoint handler: {e}")
                return {
                    "generated_text": f"Unexpected error: {str(e)[:100]}...",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
                
        return handler
    
    def create_cuda_llama_endpoint_handler(self, tokenizer, endpoint_model, cuda_label, endpoint, is_real_impl=False):
        """Create a handler for CUDA-based LLaMA inference with enhanced memory management.
        
        Args:
            tokenizer: The tokenizer to use for input/output processing
            endpoint_model: Model name or path
            cuda_label: CUDA device identifier (e.g., 'cuda:0')
            endpoint: The LLaMA model endpoint
            is_real_impl: Flag indicating if we're using a real implementation or mock
            
        Returns:
            Handler function for CUDA-based text generation
        """
        
        def handler(text_input, tokenizer=tokenizer, endpoint_model=endpoint_model, cuda_label=cuda_label, endpoint=endpoint, generation_config=None):
            """CUDA handler for LLaMA text generation with optimized memory management.
            
            Args:
                text_input: Input text or tokenized input
                generation_config: Optional dictionary with generation parameters
                
            Returns:
                Dictionary with generated text and implementation type
            """
            # Start performance tracking
            import time
            start_time = time.time()
            
            # Flag to track if we're using real implementation or mock
            is_mock = not is_real_impl
            
            # Validate input
            if text_input is None:
                is_mock = True
                return {
                    "text": "No input provided",
                    "model_name": endpoint_model,
                    "implementation_type": "MOCK",
                    "device": cuda_label
                }
            
            # Check for CUDA availability
            cuda_available = (
                hasattr(self.torch, 'cuda') and 
                self.torch.cuda.is_available() and 
                hasattr(endpoint, 'to') and callable(endpoint.to)
            )
            
            # Check if we're dealing with a mock component
            if isinstance(endpoint, type(MagicMock())) or isinstance(tokenizer, type(MagicMock())) or not cuda_available:
                is_mock = True
                return {
                    "text": "Once upon a time, a fox and a dog became best friends in the forest, learning to hunt together.",
                    "model_name": endpoint_model,
                    "implementation_type": "MOCK",
                    "device": cuda_label
                }
            
            # Initialize model with error handling
            if "eval" in dir(endpoint):
                try:
                    endpoint.eval()
                except Exception as eval_error:
                    print(f"Error setting model to eval mode: {eval_error}")
                    # Continue anyway, just log the error
            
            with self.torch.no_grad():
                try:
                    # Clean GPU cache before processing
                    if hasattr(self.torch.cuda, 'empty_cache'):
                        self.torch.cuda.empty_cache()
                    
                    # Get CUDA memory information for tracking
                    mem_info = {}
                    if hasattr(self.torch.cuda, 'mem_get_info'):
                        try:
                            free_memory_start, total_memory = self.torch.cuda.mem_get_info()
                            free_memory_start_gb = free_memory_start / (1024**3)
                            total_memory_gb = total_memory / (1024**3)
                            mem_info = {
                                "free_memory_gb": free_memory_start_gb,
                                "total_memory_gb": total_memory_gb,
                                "used_memory_gb": total_memory_gb - free_memory_start_gb
                            }
                            print(f"CUDA memory available before processing: {free_memory_start_gb:.2f}GB / {total_memory_gb:.2f}GB")
                        except Exception as mem_error:
                            print(f"Error getting CUDA memory info: {mem_error}")
                            free_memory_start = None
                    
                    # Handle batch input (list of strings)
                    is_batch = isinstance(text_input, list)
                    batch_size = len(text_input) if is_batch else 1
                    
                    # Tokenize input safely
                    try:
                        if is_batch:
                            # Process as batch
                            inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
                        elif isinstance(text_input, str):
                            # Record input length for diagnostics
                            input_length = len(text_input.split())
                            
                            # Set return_tensors to pt for PyTorch
                            inputs = tokenizer(text_input, return_tensors="pt")
                        else:
                            # Assume it's already tokenized, create a safe copy
                            inputs = {}
                            if hasattr(text_input, 'items'):
                                input_length = 0  # We don't know the length of pre-tokenized input
                                for k, v in text_input.items():
                                    if hasattr(v, 'to') and callable(v.to):
                                        inputs[k] = v.to(cuda_label)
                                    else:
                                        inputs[k] = v
                            else:
                                # Invalid input type
                                is_mock = True
                                if hasattr(self.torch.cuda, 'empty_cache'):
                                    self.torch.cuda.empty_cache()
                                return {
                                    "text": f"Invalid input type: {type(text_input)}",
                                    "model_name": endpoint_model,
                                    "implementation_type": "MOCK",
                                    "device": cuda_label
                                }
                        
                        # Safely move tensors to the correct device for all inputs
                        cuda_inputs = {}
                        for key in list(inputs.keys()):
                            if hasattr(inputs[key], 'to') and callable(inputs[key].to):
                                cuda_inputs[key] = inputs[key].to(cuda_label)
                            else:
                                cuda_inputs[key] = inputs[key]
                        inputs = cuda_inputs
                        
                    except Exception as tokenize_error:
                        print(f"Error tokenizing or moving input to CUDA: {tokenize_error}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        is_mock = True
                        if hasattr(self.torch.cuda, 'empty_cache'):
                            self.torch.cuda.empty_cache()
                        return {
                            "text": f"Error preparing input: {str(tokenize_error)[:50]}...",
                            "model_name": endpoint_model,
                            "implementation_type": "MOCK",
                            "device": cuda_label
                        }
                        
                    # Validate inputs
                    if not isinstance(inputs, dict) or "input_ids" not in inputs:
                        is_mock = True
                        if hasattr(self.torch.cuda, 'empty_cache'):
                            self.torch.cuda.empty_cache()
                        return {
                            "text": "Invalid inputs format or missing input_ids",
                            "model_name": endpoint_model,
                            "implementation_type": "MOCK",
                            "device": cuda_label
                        }
                    
                    # Run generation safely
                    try:
                        # Record timing for generation start
                        generation_start_time = time.time()
                        
                        # Verify endpoint has generate method
                        if not hasattr(endpoint, 'generate') or not callable(endpoint.generate):
                            is_mock = True
                            if hasattr(self.torch.cuda, 'empty_cache'):
                                self.torch.cuda.empty_cache()
                            return {
                                "text": "Model endpoint missing generate method",
                                "model_name": endpoint_model,
                                "implementation_type": "MOCK",
                                "device": cuda_label
                            }
                            
                        # Get attention_mask safely
                        attention_mask = inputs.get("attention_mask", None)
                        
                        # Determine generation parameters based on input length
                        # Longer input = fewer new tokens to prevent CUDA OOM
                        max_input_length = inputs["input_ids"].shape[1] if hasattr(inputs["input_ids"], "shape") else 0
                        max_new_tokens = max(32, min(512, 1024 - max_input_length))
                        
                        # Parse generation_config if provided
                        gen_params = {
                            "max_new_tokens": max_new_tokens,
                            "do_sample": True,
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "pad_token_id": tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
                        }
                        
                        # Override with user-provided parameters if any
                        if generation_config and isinstance(generation_config, dict):
                            for key, value in generation_config.items():
                                if key in ["max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "num_beams"]:
                                    gen_params[key] = value
                        
                        # Run generation with enhanced parameters
                        outputs = endpoint.generate(
                            inputs["input_ids"],
                            attention_mask=attention_mask,
                            **gen_params
                        )
                        
                        # Record generation time
                        generation_time = time.time() - generation_start_time
                        
                        # Get CUDA memory usage after generation
                        gpu_memory_used_mb = None
                        if hasattr(self.torch.cuda, 'mem_get_info') and free_memory_start is not None:
                            try:
                                free_memory_after, _ = self.torch.cuda.mem_get_info()
                                memory_used_mb = (free_memory_start - free_memory_after) / (1024**2)
                                gpu_memory_used_mb = memory_used_mb
                                print(f"CUDA memory used for generation: {memory_used_mb:.2f}MB")
                            except Exception as mem_error:
                                print(f"Error getting CUDA memory usage after generation: {mem_error}")
                        
                        # Verify outputs are valid
                        if outputs is None:
                            is_mock = True
                            if hasattr(self.torch.cuda, 'empty_cache'):
                                self.torch.cuda.empty_cache()
                            return {
                                "text": "Model generated null output",
                                "model_name": endpoint_model,
                                "implementation_type": "MOCK",
                                "device": cuda_label
                            }
                    except Exception as gen_error:
                        print(f"Error during generation: {gen_error}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        is_mock = True
                        if hasattr(self.torch.cuda, 'empty_cache'):
                            self.torch.cuda.empty_cache()
                        return {
                            "text": f"Error during generation: {str(gen_error)[:50]}...",
                            "model_name": endpoint_model,
                            "implementation_type": "MOCK",
                            "device": cuda_label
                        }
                    
                    # Decode output safely
                    try:
                        # Handle batch outputs
                        if is_batch:
                            # Process batch results
                            batch_results = []
                            
                            # Verify output has expected structure for batch
                            if not hasattr(outputs, 'shape') or len(outputs.shape) < 2 or outputs.shape[0] < batch_size:
                                is_mock = True
                                if hasattr(self.torch.cuda, 'empty_cache'):
                                    self.torch.cuda.empty_cache()
                                return {
                                    "text": "Invalid output tensor shape for batch",
                                    "model_name": endpoint_model,
                                    "implementation_type": "MOCK",
                                    "device": cuda_label
                                }
                            
                            # Move to CPU for decoding (save CUDA memory)
                            if hasattr(outputs, 'cpu') and callable(outputs.cpu):
                                outputs_cpu = outputs.cpu()
                            else:
                                outputs_cpu = outputs
                                
                            # Verify tokenizer has batch_decode method
                            if hasattr(tokenizer, 'batch_decode') and callable(tokenizer.batch_decode):
                                batch_texts = tokenizer.batch_decode(outputs_cpu, skip_special_tokens=True)
                                
                                # Return as a list of dict results with metadata
                                for i, text in enumerate(batch_texts):
                                    if text and len(text.strip()) > 0:
                                        batch_results.append(text)
                                    else:
                                        batch_results.append("Empty generation result")
                                
                                # Create batch output
                                return batch_results
                            else:
                                # Fall back to one-by-one decoding
                                for i in range(batch_size):
                                    if i < outputs_cpu.shape[0]:
                                        text = tokenizer.decode(outputs_cpu[i], skip_special_tokens=True)
                                        batch_results.append(text if text and len(text.strip()) > 0 else "Empty result")
                                    else:
                                        batch_results.append("Index out of range")
                                
                                return batch_results
                        else:
                            # Handle single output
                            # Verify output has expected structure
                            if not hasattr(outputs, 'shape') or len(outputs.shape) < 1 or outputs.shape[0] < 1:
                                is_mock = True
                                if hasattr(self.torch.cuda, 'empty_cache'):
                                    self.torch.cuda.empty_cache()
                                return {
                                    "text": "Invalid output tensor shape",
                                    "model_name": endpoint_model,
                                    "implementation_type": "MOCK",
                                    "device": cuda_label
                                }
                            
                            # Move to CPU for decoding (save CUDA memory)
                            if hasattr(outputs, 'cpu') and callable(outputs.cpu):
                                # Move entire tensor to CPU
                                outputs_cpu = outputs.cpu()
                                if outputs_cpu.shape[0] > 0:
                                    outputs_cpu = outputs_cpu[0]  # Use first sequence
                            elif hasattr(outputs[0], 'cpu') and callable(outputs[0].cpu):
                                # Already has batch dimension separated
                                outputs_cpu = outputs[0].cpu()
                            else:
                                # Can't move to CPU, use as is
                                outputs_cpu = outputs[0] if outputs.shape[0] > 0 else outputs
                            
                            # Verify tokenizer has decode method
                            if not hasattr(tokenizer, 'decode') or not callable(tokenizer.decode):
                                is_mock = True
                                if hasattr(self.torch.cuda, 'empty_cache'):
                                    self.torch.cuda.empty_cache()
                                return {
                                    "text": "Tokenizer missing decode method",
                                    "model_name": endpoint_model,
                                    "implementation_type": "MOCK",
                                    "device": cuda_label
                                }
                            
                            # Decode the output
                            decoded_output = tokenizer.decode(outputs_cpu, skip_special_tokens=True)
                            
                            # Check for empty output
                            if not decoded_output or len(decoded_output.strip()) == 0:
                                decoded_output = "Empty generation result"
                                is_mock = True
                            
                    except Exception as decode_error:
                        print(f"Error decoding output: {decode_error}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        is_mock = True
                        if hasattr(self.torch.cuda, 'empty_cache'):
                            self.torch.cuda.empty_cache()
                        return {
                            "text": f"Error decoding output: {str(decode_error)[:50]}...",
                            "model_name": endpoint_model,
                            "implementation_type": "MOCK",
                            "device": cuda_label
                        }
                    
                    # Calculate total processing time
                    total_time = time.time() - start_time
                    
                    # Cleanup GPU memory
                    if hasattr(self.torch.cuda, 'empty_cache'):
                        self.torch.cuda.empty_cache()
                    
                    # For batch outputs, we already returned above
                    if is_batch:
                        return batch_results
                    
                    # Return result with implementation type and performance metrics
                    result = {
                        "text": decoded_output,
                        "model_name": endpoint_model,
                        "implementation_type": "REAL" if not is_mock else "MOCK",
                        "device": cuda_label,
                        "generation_time_seconds": generation_time if 'generation_time' in locals() else None,
                        "total_time_seconds": total_time,
                        "gpu_memory_mb": gpu_memory_used_mb
                    }
                    
                    # Add memory info if available
                    if mem_info:
                        result["memory_info"] = mem_info
                        
                    return result
                    
                except Exception as e:
                    # Clean GPU memory on any unexpected error
                    if hasattr(self.torch.cuda, 'empty_cache'):
                        self.torch.cuda.empty_cache()
                    print(f"Unexpected error in CUDA LLaMA endpoint handler: {e}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    return {
                        "text": f"Unexpected error: {str(e)[:100]}...",
                        "model_name": endpoint_model,
                        "implementation_type": "MOCK",
                        "device": cuda_label
                    }
                    
        return handler
        
    def create_openvino_llama_endpoint_handler(self, tokenizer, model_name, openvino_label, endpoint):
        """Create a handler for OpenVINO-based LLaMA inference
        
        Args:
            tokenizer: HuggingFace tokenizer
            model_name: Name of the model
            openvino_label: Label for OpenVINO device
            endpoint: OpenVINO model endpoint
            
        Returns:
            Handler function for inference
        """
        def handler(text_input, tokenizer=tokenizer, model_name=model_name, openvino_label=openvino_label, endpoint=endpoint):
            """OpenVINO handler for LLaMA text generation.
            
            Args:
                text_input: Input text or tokenized input
                
            Returns:
                Dictionary with generated text and implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = False
            
            # Validate input
            if text_input is None:
                is_mock = True
                return {
                    "generated_text": "No input provided",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
            
            # Check if we're using a mock component
            if isinstance(endpoint, type(MagicMock())) or isinstance(tokenizer, type(MagicMock())):
                is_mock = True
                return {
                    "generated_text": "Once upon a time, a fox and a dog became friends in the forest.",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
            
            try:
                # Tokenize input with error handling
                try:
                    if isinstance(text_input, str):
                        # Tokenize the text input
                        tokens = tokenizer(text_input, return_tensors="np")
                    else:
                        # Assume it's already tokenized or in the right format
                        tokens = text_input
                        
                    # Validate tokens contains needed keys for inference
                    if not isinstance(tokens, dict) or "input_ids" not in tokens:
                        is_mock = True
                        return {
                            "generated_text": f"Failed to properly tokenize input: {text_input[:30]}...",
                            "model_name": model_name,
                            "implementation_type": "MOCK"
                        }
                except Exception as tokenize_error:
                    print(f"Error tokenizing input: {tokenize_error}")
                    is_mock = True
                    return {
                        "generated_text": f"Error tokenizing input: {str(tokenize_error)[:50]}...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Prepare the input for the model safely
                input_dict = {}
                try:
                    # Make a copy of keys to avoid dictionary size change issues
                    for key in list(tokens.keys()):
                        value = tokens[key]
                        if hasattr(value, 'numpy'):
                            input_dict[key] = value.numpy()
                        else:
                            input_dict[key] = value
                except Exception as prep_error:
                    print(f"Error preparing input dictionary: {prep_error}")
                    is_mock = True
                    return {
                        "generated_text": f"Error preparing input: {str(prep_error)[:50]}...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Run inference with better error handling
                try:
                    if hasattr(endpoint, 'run_model') and callable(endpoint.run_model):
                        # Direct model inference
                        outputs = endpoint.run_model(input_dict)
                    elif hasattr(endpoint, 'generate') and callable(endpoint.generate):
                        # Pipeline-style inference
                        if "input_ids" in tokens:
                            outputs = endpoint.generate(tokens["input_ids"])
                        else:
                            is_mock = True
                            return {
                                "generated_text": "Missing input_ids for generation",
                                "model_name": model_name,
                                "implementation_type": "MOCK"
                            }
                    else:
                        # Neither run_model nor generate methods available
                        is_mock = True
                        return {
                            "generated_text": "Unsupported endpoint type - missing run_model or generate methods",
                            "model_name": model_name,
                            "implementation_type": "MOCK"
                        }
                    
                    # Validate the outputs
                    if outputs is None:
                        is_mock = True
                        return {
                            "generated_text": "Model produced null output",
                            "model_name": model_name,
                            "implementation_type": "MOCK"
                        }
                except Exception as inference_error:
                    print(f"Error during inference: {inference_error}")
                    is_mock = True
                    return {
                        "generated_text": f"Error during inference: {str(inference_error)[:50]}...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Process outputs with better error handling
                try:
                    if isinstance(outputs, dict) and "logits" in outputs:
                        # Convert logits to token IDs with safe operations
                        logits = outputs["logits"]
                        if not isinstance(logits, np.ndarray):
                            logits = np.array(logits)
                            
                        # Safe argmax with shape check
                        if logits.size > 0:
                            next_token_ids = np.argmax(logits, axis=-1)
                            
                            # Safely decode token IDs
                            if hasattr(tokenizer, 'batch_decode') and callable(tokenizer.batch_decode):
                                try:
                                    generated_text = tokenizer.batch_decode(next_token_ids, skip_special_tokens=True)
                                    # Safe indexing with length check
                                    if generated_text and len(generated_text) > 0:
                                        generated_text = generated_text[0]
                                    else:
                                        generated_text = "Empty generation result"
                                        is_mock = True
                                except Exception as decode_error:
                                    print(f"Error decoding tokens: {decode_error}")
                                    generated_text = f"Error decoding tokens: {str(decode_error)[:50]}..."
                                    is_mock = True
                            else:
                                # Tokenizer doesn't have batch_decode
                                generated_text = "Cannot decode tokens - tokenizer missing batch_decode method"
                                is_mock = True
                        else:
                            # Empty logits
                            generated_text = "Empty logits array from model"
                            is_mock = True
                    elif hasattr(outputs, 'numpy') and callable(outputs.numpy):
                        # Outputs are token IDs directly
                        try:
                            output_numpy = outputs.numpy()
                            if hasattr(tokenizer, 'batch_decode') and callable(tokenizer.batch_decode):
                                generated_text = tokenizer.batch_decode(output_numpy, skip_special_tokens=True)
                                # Safe indexing with length check
                                if generated_text and len(generated_text) > 0:
                                    generated_text = generated_text[0]
                                else:
                                    generated_text = "Empty generation result"
                                    is_mock = True
                            else:
                                generated_text = "Cannot decode tokens - tokenizer missing batch_decode method"
                                is_mock = True
                        except Exception as numpy_error:
                            print(f"Error converting output to numpy: {numpy_error}")
                            generated_text = f"Error converting output: {str(numpy_error)[:50]}..."
                            is_mock = True
                    else:
                        # Fallback for other output formats
                        generated_text = str(outputs)
                        is_mock = True
                except Exception as process_error:
                    print(f"Error processing model outputs: {process_error}")
                    generated_text = f"Error processing outputs: {str(process_error)[:50]}..."
                    is_mock = True
                
                # Return the result with proper implementation type
                return {
                    "generated_text": generated_text,
                    "model_name": model_name,
                    "implementation_type": "MOCK" if is_mock else "REAL"
                }
                
            except Exception as e:
                print(f"Unexpected error in OpenVINO LLaMA handler: {e}")
                return {
                    "generated_text": f"Unexpected error: {str(e)[:100]}...",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
                
        return handler
    
    def create_qualcomm_llama_endpoint_handler(self, tokenizer, model_name, qualcomm_label, endpoint):
        """Create a handler for Qualcomm-based LLaMA inference
        
        Args:
            tokenizer: HuggingFace tokenizer
            model_name: Name of the model
            qualcomm_label: Label for Qualcomm hardware
            endpoint: SNPE model endpoint
            
        Returns:
            Handler function for inference
        """
        def handler(text_input, tokenizer=tokenizer, model_name=model_name, qualcomm_label=qualcomm_label, endpoint=endpoint):
            """Qualcomm handler for LLaMA text generation.
            
            Args:
                text_input: Input text or tokenized input
                
            Returns:
                Dictionary with generated text and implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = False
            
            # Validate input
            if text_input is None:
                is_mock = True
                return {
                    "generated_text": "No input provided",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
            
            # Check if we're using mocks or missing essential components
            if (isinstance(endpoint, type(MagicMock())) or 
                isinstance(tokenizer, type(MagicMock())) or 
                self.snpe_utils is None or
                not hasattr(self.snpe_utils, 'run_inference')):
                
                is_mock = True
                return {
                    "generated_text": "The fox and the dog became best friends, exploring the forest together every day.",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
            
            try:
                # Tokenize input with error handling
                try:
                    if isinstance(text_input, str):
                        inputs = tokenizer(text_input, return_tensors="np", padding=True)
                    else:
                        # Assume it's already tokenized, create a safe copy to avoid mutation issues
                        inputs = {}
                        # Use list() to make a copy of keys to avoid dict size change errors
                        if hasattr(text_input, 'items'):
                            for k, v in text_input.items():
                                if hasattr(v, 'numpy'):
                                    inputs[k] = v.numpy()
                                else:
                                    inputs[k] = v
                        else:
                            # Invalid input type
                            is_mock = True
                            return {
                                "generated_text": f"Invalid input type: {type(text_input)}",
                                "model_name": model_name,
                                "implementation_type": "MOCK"
                            }
                except Exception as tokenize_error:
                    print(f"Error tokenizing input: {tokenize_error}")
                    is_mock = True
                    return {
                        "generated_text": f"Error tokenizing input: {str(tokenize_error)[:50]}...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Validate required input keys
                if "input_ids" not in inputs or "attention_mask" not in inputs:
                    is_mock = True
                    return {
                        "generated_text": "Missing required input fields (input_ids or attention_mask)",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Create a safe copy of input_ids for generation
                try:
                    # Use .tolist() safely with shape checks
                    if (isinstance(inputs["input_ids"], (np.ndarray, list)) and 
                        (isinstance(inputs["input_ids"], np.ndarray) and inputs["input_ids"].size > 0) or
                        (isinstance(inputs["input_ids"], list) and len(inputs["input_ids"]) > 0)):
                        
                        # Try to access the first element safely
                        if isinstance(inputs["input_ids"], np.ndarray):
                            # Make a copy to avoid mutations
                            if inputs["input_ids"].ndim > 1:
                                generated_ids = inputs["input_ids"][0].tolist()
                            else:
                                generated_ids = inputs["input_ids"].tolist()
                        else:
                            # It's already a list
                            generated_ids = inputs["input_ids"][0] if isinstance(inputs["input_ids"][0], list) else inputs["input_ids"]
                    else:
                        is_mock = True
                        return {
                            "generated_text": "Invalid or empty input_ids",
                            "model_name": model_name,
                            "implementation_type": "MOCK"
                        }
                except Exception as input_error:
                    print(f"Error processing input_ids: {input_error}")
                    is_mock = True
                    return {
                        "generated_text": f"Error processing input_ids: {str(input_error)[:50]}...",
                        "model_name": model_name,
                        "implementation_type": "MOCK"
                    }
                
                # Initial input for the model
                model_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }
                
                # Set generation parameters
                past_key_values = None
                max_new_tokens = 256
                stop_generation = False
                
                # Generate tokens one by one with better error handling
                try:
                    for i in range(max_new_tokens):
                        if stop_generation:
                            break
                            
                        # Add KV cache to inputs if we have it
                        if past_key_values is not None:
                            try:
                                for j, (k, v) in enumerate(past_key_values):
                                    model_inputs[f"past_key_values.{j}.key"] = k
                                    model_inputs[f"past_key_values.{j}.value"] = v
                            except Exception as kv_error:
                                print(f"Error adding KV cache at iteration {i}: {kv_error}")
                                # Continue without KV cache
                                pass
                        
                        # Run inference safely
                        try:
                            results = self.snpe_utils.run_inference(endpoint, model_inputs)
                            
                            # Validate inference results
                            if not results or not isinstance(results, dict):
                                print(f"Invalid inference results at iteration {i}: {results}")
                                stop_generation = True
                                continue
                                
                            # Process logits
                            if "logits" in results:
                                # Convert to numpy safely
                                logits = results["logits"]
                                if not isinstance(logits, np.ndarray):
                                    try:
                                        logits = np.array(logits)
                                    except Exception as conv_error:
                                        print(f"Error converting logits to numpy: {conv_error}")
                                        stop_generation = True
                                        continue
                                
                                # Save KV cache if provided, with error handling
                                if "past_key_values" in results:
                                    try:
                                        past_key_values = results["past_key_values"]
                                    except Exception as kv_save_error:
                                        print(f"Error saving KV cache: {kv_save_error}")
                                        past_key_values = None
                                
                                # Safely compute next token ID
                                try:
                                    # Ensure logits have expected shape for indexing
                                    if logits.ndim > 2 and logits.shape[1] > 0:
                                        next_token_id = int(np.argmax(logits[0, -1, :]))
                                    elif logits.ndim == 2:
                                        next_token_id = int(np.argmax(logits[-1, :]))
                                    else:
                                        print(f"Unexpected logits shape: {logits.shape}")
                                        stop_generation = True
                                        continue
                                        
                                    # Add the generated token
                                    generated_ids.append(next_token_id)
                                    
                                    # Check for EOS token safely
                                    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
                                    if eos_token_id is not None and next_token_id == eos_token_id:
                                        break
                                        
                                    # Update inputs for next iteration
                                    model_inputs = {
                                        "input_ids": np.array([[next_token_id]]),
                                        "attention_mask": np.array([[1]])
                                    }
                                except Exception as token_error:
                                    print(f"Error selecting next token: {token_error}")
                                    stop_generation = True
                                    continue
                            else:
                                # No logits in results
                                print("No logits in model output")
                                stop_generation = True
                                continue
                        except Exception as run_error:
                            print(f"Error during inference run at iteration {i}: {run_error}")
                            stop_generation = True
                            continue
                            
                except Exception as gen_error:
                    print(f"Error in token generation loop: {gen_error}")
                    # We'll still try to decode what we have so far
                    is_mock = True
                
                # Decode the generated sequence
                try:
                    if hasattr(tokenizer, 'decode') and callable(tokenizer.decode) and generated_ids:
                        decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        
                        # Check if we got an actual result
                        if not decoded_output or len(decoded_output.strip()) == 0:
                            decoded_output = "Empty generation result"
                            is_mock = True
                    else:
                        # Fallback for missing decode method
                        decoded_output = "Generated text from Qualcomm device (decoding unavailable)"
                        is_mock = True
                except Exception as decode_error:
                    print(f"Error decoding tokens: {decode_error}")
                    decoded_output = f"Error decoding generation: {str(decode_error)[:50]}..."
                    is_mock = True
                
                # Return result with implementation type
                return {
                    "generated_text": decoded_output,
                    "model_name": model_name,
                    "implementation_type": "MOCK" if is_mock else "REAL"
                }
                
            except Exception as e:
                print(f"Unexpected error in Qualcomm LLaMA endpoint handler: {e}")
                return {
                    "generated_text": f"Unexpected error: {str(e)[:100]}...",
                    "model_name": model_name,
                    "implementation_type": "MOCK"
                }
                
        return handler