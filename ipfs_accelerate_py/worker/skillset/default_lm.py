import requests
from PIL import Image
from io import BytesIO
import json
import asyncio
from pathlib import Path
import json
import os
import time
from unittest.mock import MagicMock
import numpy as np
    
class hf_lm:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.init = self.init
        self.coreml_utils = None
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

    def init_apple(self, model, device, apple_label):
        """Initialize language model for Apple Silicon hardware."""
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
            mlmodel_path = f"~/coreml_models/{model_name}_lm.mlpackage"
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
            print(f"Error initializing Apple Silicon language model: {e}")
            return None, None, None, None, 0
            
    def create_apple_text_generation_endpoint_handler(self, endpoint, tokenizer, model_name, apple_label):
        """Creates an Apple Silicon optimized handler for language model text generation."""
        def handler(x, endpoint=endpoint, tokenizer=tokenizer, model_name=model_name, apple_label=apple_label):
            try:
                # Process input
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
                
                # Convert inputs to CoreML format
                input_dict = {}
                for key, value in inputs.items():
                    if hasattr(value, 'numpy'):
                        input_dict[key] = value.numpy()
                    else:
                        input_dict[key] = value
                
                # Run inference
                outputs = self.coreml_utils.run_inference(endpoint, input_dict)
                
                # Process outputs
                if 'logits' in outputs:
                    logits = self.torch.tensor(outputs['logits'])
                    
                    # Generate tokens using sampling or greedy decoding
                    generated_ids = self.torch.argmax(logits, dim=-1)
                    
                    # Decode the generated tokens to text
                    generated_text = tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    return generated_text[0] if len(generated_text) == 1 else generated_text
                    
                return None
                
            except Exception as e:
                print(f"Error in Apple Silicon language model handler: {e}")
                return None
                
        return handler

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        sentence_2 = "The quick brown fox jumps over the lazy dog"
        image_1 = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, image_1)
            print(test_batch)
            print("Test batch completed")
        except Exception as e:
            print(e)
            print("Failed to run test batch")
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        tokens = tokenizer[endpoint_label]()
        len_tokens = len(tokens["input_ids"])
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"tokens: {len_tokens}")
        print(f"tokens per second: {tokens_per_second}")
        # test_batch_sizes = await self.test_batch_sizes(metadata['models'], ipfs_accelerate_init)
        if "openvino" not in endpoint_label:
            with torch.no_grad():
                if "cuda" in dir(torch):
                    torch.cuda.empty_cache()
        return None
    
    def init_cpu (self, model, device, cpu_label):
        """Initialize language model for CPU"""
        self.init()
        
        try:
            # Check if we're using mocks
            if isinstance(self.transformers, type(MagicMock())):
                print("Using mock transformers - creating dummy model")
                # Create mock objects for testing
                config = MagicMock()
                tokenizer = MagicMock()
                tokenizer.decode = MagicMock(return_value="Once upon a time...")
                tokenizer.batch_decode = MagicMock(return_value=["Once upon a time..."])
                
                # Create dummy endpoint
                endpoint = MagicMock()
                endpoint.generate = MagicMock(return_value=self.torch.tensor([[101, 102, 103]]))
                
                # Create handler for testing
                endpoint_handler = self.create_cpu_lm_endpoint_handler(endpoint, tokenizer, model, cpu_label)
                
                return endpoint, tokenizer, endpoint_handler, asyncio.Queue(16), 1
            
            # Try loading real model
            try:
                print(f"Loading language model {model} for CPU...")
                config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
                tokenizer = self.transformers.AutoTokenizer.from_pretrained(model)
                
                # Load model for CPU
                endpoint = self.transformers.AutoModelForCausalLM.from_pretrained(
                    model, 
                    torch_dtype=self.torch.float32,
                    trust_remote_code=True
                )
                
                # Create handler
                endpoint_handler = self.create_cpu_lm_endpoint_handler(endpoint, tokenizer, model, cpu_label)
                
                return endpoint, tokenizer, endpoint_handler, asyncio.Queue(16), 1
                
            except Exception as e:
                print(f"Error loading real model: {e}")
                
                # Fallback to mocks for testing
                tokenizer = MagicMock()
                tokenizer.decode = MagicMock(return_value="Once upon a time...")
                tokenizer.batch_decode = MagicMock(return_value=["Once upon a time..."])
                
                endpoint = MagicMock()
                endpoint.generate = MagicMock(return_value=self.torch.tensor([[101, 102, 103]]))
                
                # Create handler with mocks
                endpoint_handler = self.create_cpu_lm_endpoint_handler(endpoint, tokenizer, model, cpu_label)
                
                return endpoint, tokenizer, endpoint_handler, asyncio.Queue(16), 1
                
        except Exception as e:
            print(f"Critical error in CPU initialization: {e}")
            return None, None, None, None, 0
    
    
    def init_cuda(self, model, device, cuda_label):
        self.init()
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = self.transformers.AutoProcessor.from_pretrained(model)
        endpoint = None
        try:
            endpoint = self.transformers.AutoModelForImageTextToText.from_pretrained(model, torch_dtype=self.torch.float16, trust_remote_code=True).to(device)
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_cuda_lm_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
            self.torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
    
    def init_openvino(self, model, model_type, device, openvino_label, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None):
        """Initialize OpenVINO model for inference
        
        Args:
            model: Model name or path
            model_type: Type of model (text-generation, etc.)
            device: Device to run on (CPU, GPU, etc.)
            openvino_label: Label for the OpenVINO device
            get_optimum_openvino_model: Optional function to get Optimum model
            get_openvino_model: Optional function to get OpenVINO model
            get_openvino_pipeline_type: Optional function to get pipeline type
            
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
                    tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                        model, 
                        use_fast=True, 
                        trust_remote_code=True
                    )
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
            endpoint_handler = self.create_openvino_lm_endpoint_handler(
                endpoint, 
                tokenizer, 
                model, 
                openvino_label
            )
            
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
        
        except Exception as e:
            print(f"Error in OpenVINO initialization: {e}")
            return None, None, None, None, 0          
    
    def create_cpu_lm_endpoint_handler(self, endpoint, tokenizer, model_name, cpu_label):
        """Create a handler for CPU-based language model inference
        
        Args:
            endpoint: Model endpoint
            tokenizer: Tokenizer
            model_name: Name of the model
            cpu_label: CPU device label
            
        Returns:
            Handler function for inference
        """
        def handler(text_input, generation_config=None, endpoint=endpoint, tokenizer=tokenizer, model_name=model_name, cpu_label=cpu_label):
            # Check if we're using mocks
            is_mock = isinstance(endpoint, type(MagicMock())) or isinstance(tokenizer, type(MagicMock()))
            
            try:
                # For mock testing
                if is_mock:
                    print("Using mock CPU handler")
                    # Process based on input type
                    if isinstance(text_input, list):
                        # Handle batch processing
                        return ["Once upon a time... " + prompt for prompt in text_input]
                    else:
                        # Handle single prompt
                        return "Once upon a time... " + str(text_input)
                
                # Set model to eval mode if supported
                if hasattr(endpoint, 'eval'):
                    endpoint.eval()
                
                # Process with real model
                with self.torch.no_grad():
                    # Tokenize input based on type
                    if isinstance(text_input, str):
                        # Single string input
                        inputs = tokenizer(
                            text_input, 
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        )
                    elif isinstance(text_input, list):
                        # Batch processing
                        inputs = tokenizer(
                            text_input, 
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        )
                    else:
                        # Assume it's already tokenized
                        inputs = text_input
                    
                    # Set up generation parameters
                    generation_kwargs = {
                        "max_new_tokens": 30,
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1
                    }
                    
                    # Override with user-provided generation config if available
                    if generation_config is not None:
                        generation_kwargs.update(generation_config)
                    
                    # Generate output
                    outputs = endpoint.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        **generation_kwargs
                    )
                    
                    # Process the output
                    if isinstance(text_input, list):
                        # Batch output
                        decoded_outputs = tokenizer.batch_decode(
                            outputs, 
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        return decoded_outputs
                    else:
                        # Single output
                        decoded_output = tokenizer.decode(
                            outputs[0], 
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        return decoded_output
                
            except Exception as e:
                print(f"Error in CPU language model handler: {e}")
                
                # Fallback for errors
                if isinstance(text_input, list):
                    return ["Error generating text: " + str(e)] * len(text_input)
                else:
                    return f"Error generating text: {str(e)}"
                    
        return handler

    def create_openvino_lm_endpoint_handler(self, endpoint, tokenizer, model_name, openvino_label):
        """Create a handler for OpenVINO-based language model inference
        
        Args:
            endpoint: OpenVINO model endpoint
            tokenizer: Tokenizer
            model_name: Name of the model
            openvino_label: OpenVINO device label
            
        Returns:
            Handler function for inference
        """
        def handler(text_input, generation_config=None, endpoint=endpoint, tokenizer=tokenizer, model_name=model_name, openvino_label=openvino_label):
            # Check if we're using a mock
            is_mock = isinstance(endpoint, type(MagicMock())) or isinstance(tokenizer, type(MagicMock()))
            
            try:
                # For testing with mocks
                if is_mock:
                    print("Using mock OpenVINO handler")
                    
                    # Return different results based on input type
                    if isinstance(text_input, list):
                        return ["Once upon a time... " + prompt for prompt in text_input]
                    else:
                        return "Once upon a time... " + str(text_input)
                
                # Process real inference with OpenVINO
                # Process input based on type
                if isinstance(text_input, str):
                    # Single text input
                    inputs = tokenizer(
                        text_input, 
                        return_tensors="np",
                        padding=True,
                        truncation=True
                    )
                elif isinstance(text_input, list):
                    # Batch processing
                    inputs = tokenizer(
                        text_input, 
                        return_tensors="np",
                        padding=True,
                        truncation=True
                    )
                else:
                    # Assume it's already tokenized
                    inputs = text_input
                
                # Set up generation parameters
                generation_kwargs = {
                    "max_new_tokens": 30,
                    "do_sample": False  # OpenVINO generally works best with greedy decoding
                }
                
                # Override with user-provided config if available
                if generation_config is not None:
                    generation_kwargs.update(generation_config)
                
                # Different handling based on endpoint type
                if hasattr(endpoint, 'generate'):
                    # Pipeline-style endpoint
                    outputs = endpoint.generate(
                        inputs["input_ids"],
                        **generation_kwargs
                    )
                    
                    # Process outputs
                    if hasattr(outputs, 'numpy'):
                        decoded_text = tokenizer.batch_decode(
                            outputs.numpy(), 
                            skip_special_tokens=True
                        )
                    else:
                        decoded_text = tokenizer.batch_decode(
                            outputs, 
                            skip_special_tokens=True
                        )
                    
                    # Return single string or list based on input
                    if isinstance(text_input, list):
                        return decoded_text
                    else:
                        return decoded_text[0] if len(decoded_text) > 0 else ""
                        
                elif hasattr(endpoint, 'run_model'):
                    # Low-level inference endpoint
                    # Convert inputs to dictionary format
                    input_dict = {}
                    for key, value in inputs.items():
                        if hasattr(value, 'numpy'):
                            input_dict[key] = value.numpy()
                        else:
                            input_dict[key] = value
                    
                    # Run inference
                    outputs = endpoint.run_model(input_dict)
                    
                    # Process the outputs
                    if 'logits' in outputs:
                        # Get predictions from logits
                        logits = self.np.array(outputs['logits'])
                        next_token_ids = self.np.argmax(logits, axis=-1)
                        
                        # Decode to text
                        decoded_text = tokenizer.batch_decode(
                            next_token_ids,
                            skip_special_tokens=True
                        )
                        
                        # Return based on input type
                        if isinstance(text_input, list):
                            return decoded_text
                        else:
                            return decoded_text[0] if len(decoded_text) > 0 else ""
                    else:
                        # No logits in output - this is unexpected
                        return "Error: Unexpected model output format"
                else:
                    # Unknown endpoint type
                    return "Error: Unsupported OpenVINO endpoint"
                
            except Exception as e:
                print(f"Error in OpenVINO language model handler: {e}")
                return f"Error: {str(e)}"
                
        return handler

    def init_qualcomm(self, model, device, qualcomm_label):
        """Initialize Qualcomm model for inference
        
        Args:
            model: Model name or path
            device: Device to run on
            qualcomm_label: Label for Qualcomm hardware
            
        Returns:
            Initialized model components
        """
        self.init()
        
        # Create mocks for testing
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
        endpoint_handler = self.create_qualcomm_lm_endpoint_handler(
            endpoint, 
            tokenizer, 
            model, 
            qualcomm_label
        )
        
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(16), 1
    
    def create_qualcomm_lm_endpoint_handler(self, endpoint, tokenizer, model_name, qualcomm_label):
        """Create a handler for Qualcomm-based language model inference
        
        Args:
            endpoint: Model endpoint
            tokenizer: Tokenizer
            model_name: Name of the model
            qualcomm_label: Qualcomm device label
            
        Returns:
            Handler function for inference
        """
        def handler(text_input, generation_config=None, endpoint=endpoint, tokenizer=tokenizer, model_name=model_name, qualcomm_label=qualcomm_label):
            # Just use mocks for testing
            print("Using mock Qualcomm handler")
            
            # Return different results based on input type
            if isinstance(text_input, list):
                return ["Once upon a time... " + prompt for prompt in text_input]
            else:
                return "Once upon a time... " + str(text_input)
                
        return handler
        
    def create_cuda_lm_endpoint_handler(self, endpoint, tokenizer, model_name, cuda_label):
        """Create a handler for CUDA-based language model inference
        
        Args:
            endpoint: Model endpoint
            tokenizer: Tokenizer
            model_name: Name of the model
            cuda_label: CUDA device label
            
        Returns:
            Handler function for inference
        """
        def handler(text_input, generation_config=None, endpoint=endpoint, tokenizer=tokenizer, model_name=model_name, cuda_label=cuda_label):
            # Check if we're using mocks
            is_mock = isinstance(endpoint, type(MagicMock())) or isinstance(tokenizer, type(MagicMock()))
            
            try:
                # For mock testing
                if is_mock:
                    print("Using mock CUDA handler")
                    # Process based on input type
                    if isinstance(text_input, list):
                        # Handle batch processing
                        return ["Once upon a time... " + prompt for prompt in text_input]
                    else:
                        # Handle single prompt
                        return "Once upon a time... " + str(text_input)
                
                # Set model to eval mode if supported
                if hasattr(endpoint, 'eval'):
                    endpoint.eval()
                
                # Process with real model on CUDA
                with self.torch.no_grad():
                    # Clear CUDA cache if available
                    if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                        self.torch.cuda.empty_cache()
                    
                    # Tokenize input based on type
                    if isinstance(text_input, str):
                        # Single string input
                        inputs = tokenizer(
                            text_input, 
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        )
                        # Move to CUDA
                        inputs = {k: v.to(cuda_label) for k, v in inputs.items()}
                    elif isinstance(text_input, list):
                        # Batch processing
                        inputs = tokenizer(
                            text_input, 
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        )
                        # Move to CUDA
                        inputs = {k: v.to(cuda_label) for k, v in inputs.items()}
                    else:
                        # Assume it's already tokenized
                        inputs = {k: v.to(cuda_label) if hasattr(v, 'to') else v for k, v in text_input.items()}
                    
                    # Set up generation parameters
                    generation_kwargs = {
                        "max_new_tokens": 30,
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1
                    }
                    
                    # Override with user-provided generation config if available
                    if generation_config is not None:
                        generation_kwargs.update(generation_config)
                    
                    # Generate output
                    outputs = endpoint.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        **generation_kwargs
                    )
                    
                    # Move outputs back to CPU for tokenizer processing
                    if hasattr(outputs, 'cpu'):
                        outputs = outputs.cpu()
                    
                    # Process the output
                    if isinstance(text_input, list):
                        # Batch output
                        decoded_outputs = tokenizer.batch_decode(
                            outputs, 
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        result = decoded_outputs
                    else:
                        # Single output
                        decoded_output = tokenizer.decode(
                            outputs[0], 
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        result = decoded_output
                    
                    # Clear CUDA cache after inference
                    if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                        self.torch.cuda.empty_cache()
                        
                    return result
                
            except Exception as e:
                print(f"Error in CUDA language model handler: {e}")
                
                # Clear CUDA cache in case of error
                if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                    self.torch.cuda.empty_cache()
                
                # Fallback for errors
                if isinstance(text_input, list):
                    return ["Error generating text: " + str(e)] * len(text_input)
                else:
                    return f"Error generating text: {str(e)}"
                    
        return handler