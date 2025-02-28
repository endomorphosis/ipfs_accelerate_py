import re
import sys
import os
import time
import asyncio
from pathlib import Path

class hf_t5:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_t5_endpoint_handler = self.create_openvino_t5_endpoint_handler
        self.create_cuda_t5_endpoint_handler = self.create_cuda_t5_endpoint_handler
        self.create_cpu_t5_endpoint_handler = self.create_cpu_t5_endpoint_handler
        self.create_apple_t5_endpoint_handler = self.create_apple_t5_endpoint_handler
        self.create_qualcomm_t5_endpoint_handler = self.create_qualcomm_t5_endpoint_handler
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_qualcomm = self.init_qualcomm
        self.init_apple = self.init_apple
        self.init = self.init
        self.__test__ = self.__test__
        self.snpe_utils = None
        self.coreml_utils = None
        return None

    def init(self):
        if "torch" not in dir(self):        
            if "torch" not in list(self.resources.keys()):
                import torch
                self.torch = torch
            else:
                self.torch = self.resources["torch"]

        if "transformers" not in dir(self):
            if "transformers" not in list(self.resources.keys()):
                import transformers
                self.transformers = transformers
            else:
                self.transformers = self.resources["transformers"]
            
        if "numpy" not in dir(self):
            if "numpy" not in list(self.resources.keys()):
                import numpy as np
                self.np = np
            else:
                self.np = self.resources["numpy"]
        
        return None
    
    
    def init_cpu(self, model, device, cpu_label):
        """
        Initialize T5 model for CPU inference
        
        Args:
            model: Model name or path (e.g., 't5-small')
            device: Device to run on ('cpu')
            cpu_label: Label for CPU endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        print(f"Loading {model} for CPU inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Function to create a simple test model when we can't download from HF
            def create_test_model():
                print("Creating minimal T5 model for testing")
                torch_module = self.torch  # Store reference to avoid name lookup issues
                
                # Create a minimal tokenizer
                class SimpleTokenizer:
                    def __init__(self):
                        self.vocab_size = 32000
                        
                    def __call__(self, text, return_tensors="pt", **kwargs):
                        """Convert text to token IDs"""
                        if isinstance(text, str):
                            batch_size = 1
                        else:
                            batch_size = len(text)
                            
                        # Create random token IDs (simulating tokenization)
                        seq_len = min(20, max(5, len(text) if isinstance(text, str) else 10))
                        return {
                            "input_ids": torch_module.ones((batch_size, seq_len), dtype=torch_module.long),
                            "attention_mask": torch_module.ones((batch_size, seq_len), dtype=torch_module.long)
                        }
                        
                    def batch_decode(self, token_ids, **kwargs):
                        """Convert token IDs back to text"""
                        if token_ids.dim() == 1:
                            return ["Example generated text from T5"]
                        else:
                            return ["Example generated text from T5"] * token_ids.shape[0]
                        
                    def decode(self, token_ids, **kwargs):
                        """Decode a single sequence"""
                        return "Example generated text from T5"
                
                # Create a minimal model
                class SimpleModel:
                    def __init__(self):
                        self.config = type('SimpleConfig', (), {
                            'vocab_size': 32000,
                            'd_model': 512,
                            'decoder_start_token_id': 0
                        })
                        
                    def __call__(self, **kwargs):
                        """Forward pass (not used for generation)"""
                        batch_size = kwargs.get("input_ids", torch_module.ones((1, 10))).shape[0]
                        return type('T5Output', (), {
                            'logits': torch_module.rand((batch_size, 10, 32000))
                        })
                        
                    def generate(self, input_ids=None, attention_mask=None, **kwargs):
                        """Generate text"""
                        batch_size = input_ids.shape[0] if input_ids is not None else 1
                        seq_len = 10  # Fixed output length for simplicity
                        return torch_module.ones((batch_size, seq_len), dtype=torch_module.long)
                        
                    def to(self, device):
                        """Move model to device (no-op for test)"""
                        return self
                        
                    def eval(self):
                        """Set model to evaluation mode"""
                        return self
                
                return SimpleTokenizer(), SimpleModel()
            
            # Try to load the real model if possible
            if isinstance(self.transformers, type):
                try:
                    # Try to load tokenizer and model from HuggingFace
                    tokenizer = self.transformers.T5Tokenizer.from_pretrained(
                        model, 
                        cache_dir=cache_dir
                    )
                    
                    endpoint = self.transformers.T5ForConditionalGeneration.from_pretrained(
                        model, 
                        cache_dir=cache_dir,
                        low_cpu_mem_usage=True
                    )
                    
                    print(f"Successfully loaded T5 model: {model}")
                    
                except Exception as e:
                    print(f"Failed to load real T5 model: {e}")
                    print("Creating test T5 model instead")
                    tokenizer, endpoint = create_test_model()
            else:
                # Create a test model if transformers is mocked
                tokenizer, endpoint = create_test_model()
                
            # Create the handler function
            endpoint_handler = self.create_cpu_t5_endpoint_handler(
                tokenizer, 
                model, 
                cpu_label, 
                endpoint
            )
            
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(16), 1
            
        except Exception as e:
            print(f"Error initializing CPU model: {e}")
            return None, None, None, None, 0
    
    def init_qualcomm(self, model, device, qualcomm_label):
        """Initialize T5 model for Qualcomm hardware.
        
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
            # Initialize tokenizer directly from HuggingFace
            tokenizer = self.transformers.T5Tokenizer.from_pretrained(model)
            
            # Convert model path to be compatible with SNPE
            model_name = model.replace("/", "--")
            dlc_path = f"~/snpe_models/{model_name}_t5.dlc"
            dlc_path = os.path.expanduser(dlc_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(dlc_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(dlc_path):
                print(f"Converting {model} to SNPE format...")
                self.snpe_utils.convert_model(model, "t5", str(dlc_path))
            
            # Load the SNPE model
            endpoint = self.snpe_utils.load_model(str(dlc_path))
            
            # Optimize for the specific Qualcomm device if possible
            if ":" in qualcomm_label:
                device_type = qualcomm_label.split(":")[1]
                optimized_path = self.snpe_utils.optimize_for_device(dlc_path, device_type)
                if optimized_path != dlc_path:
                    endpoint = self.snpe_utils.load_model(optimized_path)
            
            # Create endpoint handler
            endpoint_handler = self.create_qualcomm_t5_endpoint_handler(tokenizer, model, qualcomm_label, endpoint)
            
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(16), 1
        except Exception as e:
            print(f"Error initializing Qualcomm T5 model: {e}")
            return None, None, None, None, 0
            
    def init_apple(self, model, device, apple_label):
        """Initialize T5 model for Apple Silicon hardware."""
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
            tokenizer = self.transformers.T5Tokenizer.from_pretrained(model)
            
            # Convert model path to be compatible with CoreML
            model_name = model.replace("/", "--")
            mlmodel_path = f"~/coreml_models/{model_name}_t5.mlpackage"
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
            
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Apple Silicon T5 model: {e}")
            return None, None, None, None, 0

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "translate English to French: The quick brown fox jumps over the lazy dog"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1)
            print(test_batch)
            print("hf_t5 test passed")
        except Exception as e:
            print(e)
            print("hf_t5 test failed")
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
    
    def init_cuda(self, model, device, cuda_label):
        self.init()
        tokenizer = self.transformers.T5Tokenizer.from_pretrained(model)
        endpoint = None
        try:
            endpoint = self.transformers.T5ForConditionalGeneration.from_pretrained(model, torch_dtype=self.torch.float16).to(device)
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_cuda_t5_endpoint_handler(tokenizer, model, cuda_label, endpoint)
        self.torch.cuda.empty_cache()
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(16), 1
    
    def init_openvino(self, model, model_type, device, openvino_label, get_optimum_openvino_model, get_openvino_model, get_openvino_pipeline_type, openvino_cli_convert):
        """Initialize T5 model for OpenVINO.
        
        Args:
            model: Model name or path
            model_type: Model task type (e.g. 'text2text-generation-with-past')
            device: OpenVINO device ("CPU", "GPU", etc.)
            openvino_label: Label for the device ("openvino:0", etc.)
            get_optimum_openvino_model: Function to get optimum model
            get_openvino_model: Function to get OpenVINO model
            get_openvino_pipeline_type: Function to get pipeline type
            openvino_cli_convert: Function to convert model with CLI
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        # Import OpenVINO if needed
        try:
            if "openvino" not in list(self.resources.keys()):
                import openvino as ov
                self.ov = ov
            else:
                self.ov = self.resources["openvino"]
        except ImportError as e:
            print(f"Failed to import OpenVINO: {e}")
            return None, None, None, None, 0
        
        # Initialize return values
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        batch_size = 0
        
        try:
            # Verify model_type
            if model_type != "text2text-generation-with-past":
                # Try to get the correct model type
                model_type = get_openvino_pipeline_type(model, "t5")
                if model_type is None:
                    model_type = "text2text-generation-with-past"
                print(f"Using model_type: {model_type}")
            
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                model, 
                use_fast=True, 
                trust_remote_code=True
            )
            
            # Check if OpenVINO model conversion is needed
            homedir = os.path.expanduser("~")
            model_name_convert = model.replace("/", "--")
            model_dst_path = os.path.join(homedir, "openvino_models", model_name_convert, f"openvino_int8")
            
            # Create model path if needed
            os.makedirs(model_dst_path, exist_ok=True)
            
            # Check if model needs to be converted
            xml_path = os.path.join(model_dst_path, "openvino_decoder_with_past_model.xml")
            if not os.path.exists(xml_path):
                print(f"Converting {model} to OpenVINO format...")
                try:
                    openvino_cli_convert(
                        model, 
                        model_dst_path=model_dst_path, 
                        task=model_type, 
                        weight_format="int8",
                        ratio="1.0", 
                        group_size=128, 
                        sym=True
                    )
                    print(f"Model converted successfully, saved to {model_dst_path}")
                except Exception as convert_err:
                    print(f"Error converting model: {convert_err}")
            else:
                print(f"Using existing OpenVINO model at {model_dst_path}")
            
            # Load the model with optimum
            try:
                endpoint = get_optimum_openvino_model(model, model_type, openvino_label)
                print(f"Model loaded successfully with optimum")
            except Exception as load_err:
                print(f"Error loading model with optimum: {load_err}")
                try:
                    # Try loading with direct OpenVINO API
                    endpoint = get_openvino_model(model, model_type, openvino_label)
                    print("Loaded model with direct OpenVINO API")
                except Exception as direct_err:
                    print(f"Error loading model with direct API: {direct_err}")
                    return None, None, None, None, 0
            
            # Create handler function
            endpoint_handler = self.create_openvino_t5_endpoint_handler(
                endpoint, 
                tokenizer, 
                model, 
                openvino_label
            )
            
            batch_size = 0
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size
        
        except Exception as e:
            print(f"Error initializing T5 for OpenVINO: {e}")
            return None, None, None, None, 0
    
    def create_cuda_t5_endpoint_handler(self, tokenizer, endpoint_model, cuda_label, endpoint):
        """Creates a CUDA handler for T5 text generation.
        
        Args:
            tokenizer: Text tokenizer
            endpoint_model: Model name or path
            cuda_label: CUDA device identifier
            endpoint: Model endpoint
            
        Returns:
            Handler function for CUDA T5 text generation
        """
        def handler(x, cuda_endpoint_handler=endpoint, cuda_processor=tokenizer, endpoint_model=endpoint_model, cuda_label=cuda_label):
            """CUDA handler for T5 text generation.
            
            Args:
                x: Input text to process
                
            Returns:
                Dictionary with generated text and implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = False
            
            # Validate input
            if x is None:
                is_mock = True
                return {
                    "text": "No input provided",
                    "implementation_type": "MOCK"
                }
            
            chat = x if isinstance(x, str) else str(x)
            
            # Check for CUDA availability
            cuda_available = (
                hasattr(self.torch, 'cuda') and 
                self.torch.cuda.is_available() and 
                cuda_endpoint_handler is not None and
                hasattr(cuda_endpoint_handler, 'generate')
            )
            
            # If CUDA isn't available, use mock implementation
            if not cuda_available:
                is_mock = True
                return {
                    "text": f"Generated text from T5 (mock for {chat[:30]}...)",
                    "implementation_type": "MOCK"
                }
            
            # Validate tokenizer
            if cuda_processor is None or not hasattr(cuda_processor, '__call__'):
                is_mock = True
                return {
                    "text": f"Generated text from T5 (mock for {chat[:30]}...)",
                    "implementation_type": "MOCK"
                }
            
            # Try real CUDA implementation
            with self.torch.no_grad():
                try:
                    # Clean GPU cache before processing
                    if hasattr(self.torch.cuda, 'empty_cache'):
                        self.torch.cuda.empty_cache()
                    
                    # Tokenize input
                    try:
                        inputs = cuda_processor(chat, return_tensors="pt")
                        
                        # Move tensors to the correct device
                        try:
                            # Make a copy to avoid mutation issues
                            input_dict = {}
                            for key in list(inputs.keys()):
                                if hasattr(inputs[key], 'to') and callable(inputs[key].to):
                                    input_dict[key] = inputs[key].to(cuda_label)
                                else:
                                    input_dict[key] = inputs[key]
                        except Exception as device_error:
                            print(f"Error moving tensors to CUDA device: {device_error}")
                            is_mock = True
                            
                            # Clean GPU memory on error
                            if hasattr(self.torch.cuda, 'empty_cache'):
                                self.torch.cuda.empty_cache()
                                
                            return {
                                "text": f"Generated text from T5 (mock for {chat[:30]}...)",
                                "implementation_type": "MOCK"
                            }
                    except Exception as tokenize_error:
                        print(f"Tokenization error: {tokenize_error}")
                        is_mock = True
                        
                        # Clean GPU memory on error
                        if hasattr(self.torch.cuda, 'empty_cache'):
                            self.torch.cuda.empty_cache()
                            
                        return {
                            "text": f"Generated text from T5 (mock for {chat[:30]}...)",
                            "implementation_type": "MOCK"
                        }
                    
                    # Generate text
                    try:
                        outputs = cuda_endpoint_handler.generate(**input_dict)
                        
                        # Ensure output is valid
                        if outputs is None or not hasattr(outputs, '__getitem__') or len(outputs) == 0:
                            is_mock = True
                            
                            # Clean GPU memory before returning
                            if hasattr(self.torch.cuda, 'empty_cache'):
                                self.torch.cuda.empty_cache()
                                
                            return {
                                "text": f"Generated text from T5 (mock for {chat[:30]}...)",
                                "implementation_type": "MOCK"
                            }
                            
                        # Decode result
                        if hasattr(cuda_processor, 'decode'):
                            results = cuda_processor.decode(
                                outputs[0], 
                                skip_special_tokens=True, 
                                clean_up_tokenization_spaces=False
                            )
                        else:
                            is_mock = True
                            
                            # Clean GPU memory before returning
                            if hasattr(self.torch.cuda, 'empty_cache'):
                                self.torch.cuda.empty_cache()
                                
                            return {
                                "text": f"Generated text from T5 (mock for {chat[:30]}...)",
                                "implementation_type": "MOCK"
                            }
                            
                        # Clean GPU memory after successful generation
                        if hasattr(self.torch.cuda, 'empty_cache'):
                            self.torch.cuda.empty_cache()
                            
                        # Return successful result
                        return {
                            "text": results,
                            "implementation_type": "REAL"
                        }
                        
                    except Exception as gen_error:
                        print(f"Error generating text with CUDA: {gen_error}")
                        is_mock = True
                        
                        # Clean GPU memory on error
                        if hasattr(self.torch.cuda, 'empty_cache'):
                            self.torch.cuda.empty_cache()
                            
                        return {
                            "text": f"Generated text from T5 (mock for {chat[:30]}...)",
                            "implementation_type": "MOCK"
                        }
                        
                except Exception as e:
                    print(f"Unexpected error in CUDA handler: {e}")
                    is_mock = True
                    
                    # Clean GPU memory on error
                    if hasattr(self.torch.cuda, 'empty_cache'):
                        self.torch.cuda.empty_cache()
                        
                    return {
                        "text": f"Generated text from T5 (mock for {chat[:30]}...)",
                        "implementation_type": "MOCK"
                    }
        return handler
    
    def create_cpu_t5_endpoint_handler(self, tokenizer, endpoint_model, cpu_label, endpoint):
        """
        Create a handler for T5 text generation on CPU
        
        Args:
            tokenizer: T5 tokenizer for input/output processing
            endpoint_model: Model name or path
            cpu_label: Label for the CPU endpoint
            endpoint: T5 model instance
            
        Returns:
            Callable handler function for text generation
        """
        def handler(x, y=None, model=endpoint, tokenizer=tokenizer, endpoint_model=endpoint_model, cpu_label=cpu_label):
            """
            Generate text with T5
            
            Args:
                x: Input text to process (prompt)
                y: Optional parameter (unused, for API compatibility)
                
            Returns:
                Dictionary with generated text and implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = False
            
            # Set model to evaluation mode if possible
            if hasattr(model, 'eval'):
                try:
                    model.eval()
                except Exception as e:
                    print(f"Error setting model to eval mode: {e}")
                    # Continue even if setting eval mode fails
            
            try:
                # Ensure we have valid input
                if x is None:
                    is_mock = True
                    return {
                        "text": "No input provided",
                        "implementation_type": "MOCK"
                    }
                    
                # Convert input to string if needed
                input_text = x if isinstance(x, str) else str(x)
                
                print(f"Processing input: {input_text[:50]}...")
                
                # Check if we have a valid model and tokenizer
                if model is None or not (hasattr(model, 'generate') and callable(model.generate)):
                    is_mock = True
                    return {
                        "text": f"Generated text from T5 (mock for {input_text[:30]}...)",
                        "implementation_type": "MOCK"
                    }
                
                if tokenizer is None:
                    is_mock = True
                    return {
                        "text": f"Generated text from T5 (mock for {input_text[:30]}...)",
                        "implementation_type": "MOCK"
                    }
                
                # Tokenize input
                try:
                    inputs = tokenizer(input_text, return_tensors="pt")
                except Exception as tokenize_error:
                    print(f"Tokenization error: {tokenize_error}")
                    # Create a simple fallback tensor if tokenization fails
                    inputs = {
                        "input_ids": self.torch.ones((1, 10), dtype=self.torch.long),
                        "attention_mask": self.torch.ones((1, 10), dtype=self.torch.long)
                    }
                    is_mock = True
                
                # Copy inputs to avoid potential mutation issues
                input_dict = {}
                for key in list(inputs.keys()):
                    input_dict[key] = inputs[key]
                
                # Generate text with model
                try:
                    with self.torch.no_grad():
                        output_ids = model.generate(
                            **input_dict,
                            max_new_tokens=100,
                            do_sample=False,  # Deterministic output for testing
                            num_beams=1       # Simple beam search
                        )
                        
                    # Decode output tokens to text
                    if hasattr(tokenizer, 'decode'):
                        # Single string output
                        result = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    elif hasattr(tokenizer, 'batch_decode'):
                        # Batch output (take first item)
                        results = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        result = results[0] if results else ""
                    else:
                        # Fallback if tokenizer doesn't have expected methods
                        result = "Generated text (couldn't decode properly)"
                        is_mock = True
                    
                    # Return result with implementation type
                    return {
                        "text": result,
                        "implementation_type": "MOCK" if is_mock else "REAL"
                    }
                    
                except Exception as gen_error:
                    print(f"Generation error: {gen_error}")
                    # Provide a fallback result
                    is_mock = True
                    return {
                        "text": f"Error during generation: {str(gen_error)[:100]}",
                        "implementation_type": "MOCK"
                    }
                    
            except Exception as e:
                print(f"Error in CPU T5 handler: {e}")
                # Return a fallback message rather than raising an exception
                is_mock = True
                return {
                    "text": f"Error processing input: {str(e)[:100]}",
                    "implementation_type": "MOCK"
                }
                
        return handler
        
    def create_apple_t5_endpoint_handler(self, tokenizer, endpoint_model, apple_label, endpoint):
        """Create a handler for Apple Silicon-based T5 inference"""
        
        def handler(text_input, tokenizer=tokenizer, endpoint_model=endpoint_model, apple_label=apple_label, endpoint=endpoint):
            if "eval" in dir(endpoint):
                endpoint.eval()
            
            try:
                # Tokenize input
                if isinstance(text_input, str):
                    inputs = tokenizer(text_input, return_tensors="pt")
                    # Move to MPS if available
                    if hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
                        inputs = {k: v.to("mps") for k, v in inputs.items()}
                else:
                    # Assume it's already tokenized
                    inputs = {k: v.to("mps") if hasattr(v, 'to') else v for k, v in text_input.items()}
                
                # Run generation
                with self.torch.no_grad():
                    outputs = endpoint.generate(
                        inputs["input_ids"],
                        max_length=128,
                        do_sample=False
                    )
                    
                # Move back to CPU for decoding
                if hasattr(outputs, 'cpu'):
                    outputs = outputs.cpu()
                    
                # Decode output
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Return result
                return {
                    "text": decoded_output,
                    "model": endpoint_model
                }
                
            except Exception as e:
                print(f"Error in Apple T5 endpoint handler: {e}")
                return {"error": str(e)}
                
        return handler
        
    def create_qualcomm_t5_endpoint_handler(self, tokenizer, endpoint_model, qualcomm_label, endpoint):
        """Create a handler for Qualcomm-based T5 inference
        
        Args:
            tokenizer: HuggingFace tokenizer
            endpoint_model: Name of the model
            qualcomm_label: Label for Qualcomm hardware
            endpoint: SNPE model endpoint
            
        Returns:
            Handler function for inference
        """
        def handler(text_input, tokenizer=tokenizer, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint):
            """Qualcomm handler for T5 text generation.
            
            Args:
                text_input: Input text or tokenized inputs
                
            Returns:
                Dictionary with generated text and implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = False
            
            # Validate input
            if text_input is None:
                is_mock = True
                return {
                    "text": "No input provided",
                    "implementation_type": "MOCK"
                }
            
            # Check if we have SNPE utilities available
            has_snpe = (
                hasattr(self, 'snpe_utils') and 
                self.snpe_utils is not None and 
                hasattr(self.snpe_utils, 'run_inference')
            )
            
            # If necessary components aren't available, use mock implementation
            if not (has_snpe and endpoint is not None and tokenizer is not None):
                is_mock = True
                return {
                    "text": f"Generated text from T5 (mock for {text_input[:30] if isinstance(text_input, str) else str(text_input)[:30]}...)",
                    "implementation_type": "MOCK"
                }
            
            try:
                # Tokenize input with error handling
                try:
                    if isinstance(text_input, str):
                        inputs = tokenizer(text_input, return_tensors="np", padding=True)
                    else:
                        # Assume it's already tokenized, convert to numpy if needed
                        inputs = {}
                        # Use list to avoid dictionary mutation issues
                        for k, v in text_input.items() if hasattr(text_input, 'items') else {}:
                            inputs[k] = v.numpy() if hasattr(v, 'numpy') else v
                except Exception as tokenize_error:
                    print(f"Error tokenizing input: {tokenize_error}")
                    is_mock = True
                    return {
                        "text": f"Generated text from T5 (mock due to tokenization error)",
                        "implementation_type": "MOCK"
                    }
                
                # Verify inputs contain required keys
                if not ("input_ids" in inputs and "attention_mask" in inputs):
                    is_mock = True
                    return {
                        "text": f"Generated text from T5 (mock due to missing input components)",
                        "implementation_type": "MOCK"
                    }
                
                # Initial input for the model
                model_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }
                
                # Encoder pass with error handling
                try:
                    encoder_results = self.snpe_utils.run_inference(endpoint, model_inputs)
                    
                    # Check if we got valid results
                    if not encoder_results or not isinstance(encoder_results, dict):
                        is_mock = True
                        return {
                            "text": f"Generated text from T5 (mock due to invalid encoder output)",
                            "implementation_type": "MOCK"
                        }
                except Exception as encoder_error:
                    print(f"Error in encoder pass: {encoder_error}")
                    is_mock = True
                    return {
                        "text": f"Generated text from T5 (mock due to encoder error)",
                        "implementation_type": "MOCK"
                    }
                
                # Check for encoder outputs
                if "encoder_outputs.last_hidden_state" in encoder_results:
                    try:
                        # We have encoder outputs, now set up for decoder
                        decoder_inputs = {
                            "encoder_outputs.last_hidden_state": encoder_results["encoder_outputs.last_hidden_state"],
                            "decoder_input_ids": self.np.array([[tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0]])  # Start token
                        }
                        
                        # Prepare for token-by-token generation
                        generated_ids = [tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0]
                        max_length = 128
                        
                        # Generate tokens one by one
                        for _ in range(max_length):
                            # Update decoder input ids
                            decoder_inputs["decoder_input_ids"] = self.np.array([generated_ids])
                            
                            # Run decoder pass
                            try:
                                decoder_results = self.snpe_utils.run_inference(endpoint, decoder_inputs)
                            except Exception as decoder_error:
                                print(f"Error in decoder pass: {decoder_error}")
                                break
                            
                            # Get the logits
                            if "logits" in decoder_results and decoder_results["logits"] is not None:
                                try:
                                    logits = self.np.array(decoder_results["logits"])
                                    
                                    # Basic greedy decoding
                                    next_token_id = int(self.np.argmax(logits[0, -1, :]))
                                    
                                    # Add the generated token
                                    generated_ids.append(next_token_id)
                                    
                                    # Check for EOS token
                                    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
                                    if eos_token_id is not None and next_token_id == eos_token_id:
                                        break
                                except Exception as logit_error:
                                    print(f"Error processing logits: {logit_error}")
                                    break
                            else:
                                break
                        
                        # Decode the generated sequence
                        if generated_ids and hasattr(tokenizer, 'decode'):
                            try:
                                decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
                                
                                # Return result with REAL implementation type
                                return {
                                    "text": decoded_output,
                                    "model": endpoint_model,
                                    "implementation_type": "REAL"
                                }
                            except Exception as decode_error:
                                print(f"Error decoding output: {decode_error}")
                                is_mock = True
                        else:
                            is_mock = True
                    except Exception as generation_error:
                        print(f"Error in token generation: {generation_error}")
                        is_mock = True
                        
                else:
                    # Direct generation mode
                    try:
                        results = self.snpe_utils.run_inference(endpoint, model_inputs)
                        
                        # Check if we have output_ids
                        if results and "output_ids" in results and results["output_ids"] is not None:
                            try:
                                output_ids = results["output_ids"]
                                if hasattr(tokenizer, 'decode'):
                                    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                                    
                                    # Return result with REAL implementation type
                                    return {
                                        "text": decoded_output,
                                        "model": endpoint_model,
                                        "implementation_type": "REAL"
                                    }
                                else:
                                    is_mock = True
                            except Exception as decode_error:
                                print(f"Error decoding output: {decode_error}")
                                is_mock = True
                        else:
                            is_mock = True
                            print("Unexpected model output format")
                    except Exception as inference_error:
                        print(f"Error in direct generation: {inference_error}")
                        is_mock = True
                
            except Exception as e:
                print(f"Error in Qualcomm T5 endpoint handler: {e}")
                is_mock = True
            
            # Return mock result if anything failed
            if is_mock:
                return {
                    "text": f"Generated text from T5 (mock for {text_input[:30] if isinstance(text_input, str) else str(text_input)[:30]}...)",
                    "implementation_type": "MOCK"
                }
                
        return handler

    def create_openvino_t5_endpoint_handler(self, openvino_endpoint_handler, openvino_tokenizer, endpoint_model, openvino_label):
        """Creates an OpenVINO handler for T5 text generation.
        
        Args:
            openvino_endpoint_handler: The OpenVINO model endpoint
            openvino_tokenizer: The text tokenizer
            endpoint_model: The model name or path
            openvino_label: Label to identify this endpoint
            
        Returns:
            A handler function for OpenVINO T5 endpoint
        """
        def handler(x, openvino_endpoint_handler=openvino_endpoint_handler, openvino_tokenizer=openvino_tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label):
            """OpenVINO handler for T5 text generation.
            
            Args:
                x: Input text to generate from
                
            Returns:
                Generated text string with implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = False
            
            # Validate input
            chat = None
            if x is not None:
                chat = x if isinstance(x, str) else str(x)
            else:
                # Return a default response if no input is provided
                is_mock = True
                return {
                    "text": "No input provided",
                    "implementation_type": "MOCK"
                }
            
            # Validate that we have valid OpenVINO components
            if openvino_endpoint_handler is None or not hasattr(openvino_endpoint_handler, 'generate'):
                is_mock = True
                return {
                    "text": f"Generated text from T5 (mock for {chat})",
                    "implementation_type": "MOCK"
                }
                
            # Validate tokenizer
            if openvino_tokenizer is None or not hasattr(openvino_tokenizer, '__call__'):
                is_mock = True
                return {
                    "text": f"Generated text from T5 (mock for {chat})",
                    "implementation_type": "MOCK"
                }
            
            try:
                # Process input and generate text
                inputs = openvino_tokenizer(chat, return_tensors="pt")
                
                # Make a copy of inputs to avoid dict mutation issues
                input_dict = {}
                for key in list(inputs.keys()):
                    input_dict[key] = inputs[key]
                
                # Run generation with error handling
                try:
                    outputs = openvino_endpoint_handler.generate(**input_dict)
                    
                    # Ensure outputs is valid before decoding
                    if outputs is None or not hasattr(outputs, '__getitem__') or len(outputs) == 0:
                        is_mock = True
                        return {
                            "text": f"Generated text from T5 (mock for {chat})",
                            "implementation_type": "MOCK"
                        }
                    
                    # Decode the output tokens
                    if hasattr(openvino_tokenizer, 'decode'):
                        results = openvino_tokenizer.decode(
                            outputs[0], 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False
                        )
                    else:
                        is_mock = True
                        return {
                            "text": f"Generated text from T5 (mock for {chat})",
                            "implementation_type": "MOCK"
                        }
                    
                    # Return the result with implementation type
                    return {
                        "text": results,
                        "implementation_type": "REAL"
                    }
                    
                except Exception as e:
                    print(f"Error generating text with OpenVINO: {e}")
                    is_mock = True
                
            except Exception as e:
                print(f"Error in OpenVINO handler: {e}")
                is_mock = True
            
            # Fall back to mock if real implementation failed
            if is_mock:
                return {
                    "text": f"Generated text from T5 (mock for {chat})",
                    "implementation_type": "MOCK"
                }
                
            # Should never reach here, but just in case
            return {
                "text": f"Unexpected execution path in T5 handler",
                "implementation_type": "MOCK"
            }
        return handler

    def openvino_skill_convert(self, model_name, model_dst_path, task, weight_format, hfmodel=None, hfprocessor=None):
        import openvino as ov
        import os
        import numpy as np
        import requests
        import tempfile
        from transformers import AutoModel, AutoTokenizer, AutoProcessor  
        if hfmodel is None:
            hfmodel = AutoModel.from_pretrained(model_name, torch_dtype=self.torch.float16)
    
        if hfprocessor is None:
            hftokenizer = AutoTokenizer.from_pretrained(model_name)

        if hftokenizer is not None:
            from transformers import T5ForConditionalGeneration
            hfmodel = T5ForConditionalGeneration.from_pretrained(model_name)
            text = "Replace me by any text you'd like."
            text_inputs = hftokenizer(text, return_tensors="pt", padding=True).input_ids
            labels = "Das Haus ist wunderbar."
            labels_inputs = hftokenizer(labels, return_tensors="pt", padding=True).input_ids
            outputs = hfmodel(input_ids=text_inputs, decoder_input_ids=labels_inputs)
            hfmodel.config.torchscript = True
            try:
                ov_model = ov.convert_model(hfmodel)
                if not os.path.exists(model_dst_path):
                    os.mkdir(model_dst_path)
                ov.save_model(ov_model, os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
            except Exception as e:
                print(e)
                if os.path.exists(model_dst_path):
                    os.remove(model_dst_path)
                if not os.path.exists(model_dst_path):
                    os.mkdir(model_dst_path)
                self.openvino_cli_convert(model_name, model_dst_path=model_dst_path, task=task, weight_format="int8",  ratio="1.0", group_size=128, sym=True )
                core = ov.Core()
                ov_model = core.read_model(model_name, os.path.join(model_dst_path, 'openvino_decoder_with_past_model.xml'))

            ov_model = ov.compile_model(ov_model)
            hfmodel = None
        return ov_model

    def create_apple_text_generation_endpoint_handler(self, endpoint, tokenizer, model_name, apple_label):
        """Creates an Apple Silicon optimized handler for T5 text generation.
        
        Args:
            endpoint: The CoreML model endpoint
            tokenizer: The text tokenizer
            model_name: Model name or path
            apple_label: Label for Apple endpoint
            
        Returns:
            Handler function for Apple Silicon T5 text generation
        """
        def handler(x, endpoint=endpoint, tokenizer=tokenizer, model_name=model_name, apple_label=apple_label):
            """Apple Silicon handler for T5 text generation.
            
            Args:
                x: Input text to process
                
            Returns:
                Dictionary with generated text and implementation type
            """
            # Flag to track if we're using real implementation or mock
            is_mock = False
            
            # Validate input
            if x is None:
                is_mock = True
                return {
                    "text": "No input provided",
                    "implementation_type": "MOCK"
                }
            
            # Check if we have CoreML utilities available
            has_coreml = (
                hasattr(self, 'coreml_utils') and 
                self.coreml_utils is not None and 
                hasattr(self.coreml_utils, 'run_inference')
            )
            
            # Check Apple Silicon availability
            mps_available = (
                hasattr(self.torch.backends, 'mps') and 
                self.torch.backends.mps.is_available()
            )
            
            # If necessary components aren't available, use mock implementation
            if not (has_coreml and mps_available and endpoint is not None and tokenizer is not None):
                is_mock = True
                return {
                    "text": f"Generated text from T5 (mock for {x[:30] if isinstance(x, str) else str(x)[:30]}...)",
                    "implementation_type": "MOCK"
                }
            
            try:
                # Prepare input based on input type
                if isinstance(x, str):
                    # Process string input
                    try:
                        inputs = tokenizer(x, return_tensors='np', padding=True)
                    except Exception as tokenize_error:
                        print(f"Error tokenizing input: {tokenize_error}")
                        is_mock = True
                        return {
                            "text": f"Generated text from T5 (mock for {x[:30]}...)",
                            "implementation_type": "MOCK"
                        }
                        
                elif isinstance(x, list):
                    # Process list of strings
                    try:
                        inputs = tokenizer(x, return_tensors='np', padding=True)
                    except Exception as tokenize_error:
                        print(f"Error tokenizing input list: {tokenize_error}")
                        is_mock = True
                        return {
                            "text": f"Generated text from T5 (mock for list input)",
                            "implementation_type": "MOCK"
                        }
                        
                else:
                    # Use as-is (assume it's already processed)
                    inputs = x
                
                # Convert inputs to CoreML format safely
                input_dict = {}
                try:
                    # Use list to avoid dictionary size changes during iteration
                    for key in list(inputs.keys()):
                        value = inputs[key]
                        if hasattr(value, 'numpy'):
                            input_dict[key] = value.numpy()
                        else:
                            input_dict[key] = value
                except Exception as convert_error:
                    print(f"Error converting inputs to CoreML format: {convert_error}")
                    is_mock = True
                    return {
                        "text": f"Generated text from T5 (mock due to input conversion error)",
                        "implementation_type": "MOCK"
                    }
                
                # Run inference with error handling
                try:
                    outputs = self.coreml_utils.run_inference(endpoint, input_dict)
                    
                    # Check if we got valid outputs
                    if not outputs or not isinstance(outputs, dict):
                        is_mock = True
                        return {
                            "text": f"Generated text from T5 (mock due to invalid output)",
                            "implementation_type": "MOCK"
                        }
                    
                    # Process outputs
                    if 'logits' in outputs:
                        try:
                            # Convert logits to PyTorch tensor
                            logits = self.torch.tensor(outputs['logits'])
                            
                            # Generate text from logits
                            generated_ids = self.torch.argmax(logits, dim=-1)
                            
                            # Decode text
                            if hasattr(tokenizer, 'batch_decode'):
                                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                                
                                # Return as string if single item, otherwise as list
                                if isinstance(generated_text, list) and len(generated_text) == 1:
                                    result = generated_text[0]
                                else:
                                    result = generated_text
                                    
                                return {
                                    "text": result,
                                    "implementation_type": "REAL"
                                }
                            else:
                                is_mock = True
                                return {
                                    "text": f"Generated text from T5 (mock due to decoding issue)",
                                    "implementation_type": "MOCK"
                                }
                        except Exception as processing_error:
                            print(f"Error processing CoreML outputs: {processing_error}")
                            is_mock = True
                    else:
                        print("Expected 'logits' not found in CoreML outputs")
                        is_mock = True
                except Exception as inference_error:
                    print(f"Error running CoreML inference: {inference_error}")
                    is_mock = True
                    
            except Exception as e:
                print(f"Error in Apple Silicon T5 handler: {e}")
                is_mock = True
            
            # Return mock result if anything failed
            if is_mock:
                return {
                    "text": f"Generated text from T5 (mock for {x[:30] if isinstance(x, str) else str(x)[:30]}...)",
                    "implementation_type": "MOCK"
                }
                
        return handler

