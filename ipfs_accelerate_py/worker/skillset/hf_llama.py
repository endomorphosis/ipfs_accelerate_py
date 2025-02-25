import requests
from PIL import Image
from io import BytesIO
import json
import asyncio
from pathlib import Path
import json
import os
import time
    
class hf_llama:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.init = self.init
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_qualcomm = self.init_qualcomm
        self.init_cpu = self.init_cpu
        self.init_apple = self.init_apple
        self.__test__ = self.__test__
        self.create_openvino_llm_endpoint_handler = self.create_openvino_llm_endpoint_handler
        self.create_cpu_llm_endpoint_handler = self.create_cpu_llm_endpoint_handler
        self.create_cuda_llm_endpoint_handler = self.create_cuda_llm_endpoint_handler
        self.create_qualcomm_llm_endpoint_handler = self.create_qualcomm_llm_endpoint_handler
        self.create_apple_llm_endpoint_handler = self.create_apple_llm_endpoint_handler
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

    def init_qualcomm(self, model, model_type, device, qualcomm_label, get_qualcomm_genai_pipeline, get_optimum_qualcomm_model, get_qualcomm_model, get_qualcomm_pipeline_type):
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        batch_size = 0
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        endpoint = get_qualcomm_model(model, model_type, qualcomm_label)
        endpoint_handler = self.create_qualcomm_llm_endpoint_handler(endpoint,tokenizer, model, qualcomm_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size
    
    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        sentence_2 = "The quick brown fox jumps over the lazy dog"
        image_1 = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, image_1)
            print(test_batch)
            print("llama Test batch completed")
        except Exception as e:
            print(e)
            print("Failed to run llama test batch")
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
            with self.torch.no_grad():
                if "cuda" in dir(self.torch):
                    self.torch.cuda.empty_cache()
        return None
    
    def init_cpu (self, model, device, cpu_label):
        self.init()
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model, use_fast=True)
        try:
            endpoint = self.transformers.AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading CPU model: {e}")
            endpoint = None
            
        endpoint_handler = self.create_cpu_llm_endpoint_handler(endpoint, tokenizer, model, cpu_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(32), 0
    
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
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model, use_fast=True)
        
        # Check if MPS (Metal Performance Shaders) is available
        if not hasattr(self.torch.backends, 'mps') or not self.torch.backends.mps.is_available():
            print("MPS not available. Cannot initialize model on Apple Silicon.")
            return None, None, None, None, 0
            
        # For Apple Silicon, we'll use MPS as the device
        try:
            endpoint = self.transformers.AutoModelForCausalLM.from_pretrained(
                model, 
                torch_dtype=self.torch.float16, 
                trust_remote_code=True
            ).to(device)
        except Exception as e:
            print(f"Error loading model on Apple Silicon: {e}")
            endpoint = None
            
        endpoint_handler = self.create_apple_llm_endpoint_handler(endpoint, tokenizer, model, apple_label)
        
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(32), 0
    
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
        endpoint_handler = self.create_llm_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        self.torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
    
    def init_openvino(self, model, model_type, device, openvino_label, get_openvino_genai_pipeline, get_optimum_openvino_model, get_openvino_model, get_openvino_pipeline_type):
        self.init()
        if "openvino" not in list(self.resources.keys()):
            import openvino as ov
            self.ov = ov
        else:
            self.ov = self.resources["openvino"]
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        batch_size = 0                
        tokenizer =  self.transformers.AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        endpoint = get_openvino_model(model, model_type, openvino_label)
        endpoint_handler = self.create_openvino_llm_endpoint_handler(endpoint,tokenizer, model, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size          
    
    def create_cpu_llm_endpoint_handler(self, local_cuda_endpoint, local_cuda_processor, endpoint_model, cuda_label):
        def handler(x, y=None, local_cuda_endpoint=local_cuda_endpoint, local_cuda_processor=local_cuda_processor, endpoint_model=endpoint_model, cuda_label=cuda_label):
            # if "eval" in dir(self.local_endpoints[endpoint_model][cuda_label]):
            #       self.local_endpoints[endpoint_model][cuda_label].eval()
            if "eval" in dir(local_cuda_endpoint):
                local_cuda_endpoint.eval()
            else:
                pass
            with self.torch.no_grad():
                try:
                    self.torch.cuda.empty_cache()
                    config = self.transformers.AutoConfig.from_pretrained(endpoint_model, trust_remote_code=True)
                    
                    if x is not None and type(x) == str:
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": x},
                                ],
                            },
                        ]
                    elif type(x) == tuple:
                        conversation = x
                    elif type(x) == dict:
                        raise Exception("Invalid input to vlm endpoint handler")
                    elif type(x) == list:
                        # conversation = x
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": x},
                                ],
                            },
                        ]
                    else:
                        raise Exception("Invalid input to vlm endpoint handler")
                  
                    prompt = local_cuda_processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = local_cuda_processor(prompt, return_tensors="pt").to(cuda_label, self.torch.float16)
                    output = local_cuda_endpoint.generate(**inputs, max_new_tokens=30)
                    result = local_cuda_processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    # Run model inference
                    self.torch.cuda.empty_cache()
                    return result
                except Exception as e:
                    # Cleanup GPU memory in case of error
                    self.torch.cuda.empty_cache()
                    raise e
        return handler

    def create_openvino_llm_endpoint_handler(self, openvino_endpoint_handler, openvino_tokenizer, endpoint_model, openvino_label):
        def handler(x, y=None, openvino_endpoint_handler=openvino_endpoint_handler, openvino_tokenizer=openvino_tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label):
            chat = None
            if y is not None and x is not None:
                chat = x
            elif x is not None:
                if type(x) == tuple:
                    chat, image_file = x
                elif type(x) == list:
                    chat = x[0]
                    image_file = x[1]
                elif type(x) == dict:
                    chat = x["chat"]
                    image_file = x["image"]
                elif type(x) == str:
                    chat = x
                else:
                    pass

            pipeline_config = { "MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 512 ,  "NPUW_CACHE_DIR": ".npucache" }
            results = openvino_endpoint_handler.generate(x, max_new_tokens=100, do_sample=False)
            # prompt = openvino_endpoint_handler.apply_chat_template(chat, add_generation_prompt=True)
            # inputs = openvino_endpoint_handler(text=prompt, return_tensors="pt")
            # streamer = TextStreamer(openvino_tokenizer, skip_prompt=True, skip_special_tokens=True)
            # output_ids = openvino_endpoint_handler.generate(
            #     **inputs,
            #     do_sample=False,
            #     max_new_tokens=50,
            #     streamer=streamer,
            return results
        return handler

    def create_cuda_llm_endpoint_handler(self, local_cuda_endpoint, local_cuda_processor, endpoint_model, cuda_label):
        def handler(x, y=None, local_cuda_endpoint=local_cuda_endpoint, local_cuda_processor=local_cuda_processor, endpoint_model=endpoint_model, cuda_label=cuda_label):
            # if "eval" in dir(self.local_endpoints[endpoint_model][cuda_label]):
            #       self.local_endpoints[endpoint_model][cuda_label].eval()
            if "eval" in dir(local_cuda_endpoint):
                local_cuda_endpoint.eval()
            else:
                pass
            with self.torch.no_grad():
                try:
                    self.torch.cuda.empty_cache()
                    config = self.transformers.AutoConfig.from_pretrained(endpoint_model, trust_remote_code=True)
                    
                    if x is not None and type(x) == str:
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": x},
                                ],
                            },
                        ]
                    elif type(x) == tuple:
                        conversation = x
                    elif type(x) == dict:
                        raise Exception("Invalid input to vlm endpoint handler")
                    elif type(x) == list:
                        # conversation = x
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": x},
                                ],
                            },
                        ]
                    else:
                        raise Exception("Invalid input to vlm endpoint handler")
                  
                    prompt = local_cuda_processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = local_cuda_processor(prompt, return_tensors="pt").to(cuda_label, self.torch.float16)
                    output = local_cuda_endpoint.generate(**inputs, max_new_tokens=30)
                    result = local_cuda_processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    # Run model inference
                    self.torch.cuda.empty_cache()
                    return result
                except Exception as e:
                    # Cleanup GPU memory in case of error
                    self.torch.cuda.empty_cache()
                    raise e
        return handler
        
    def create_apple_llm_endpoint_handler(self, local_apple_endpoint, local_apple_tokenizer, endpoint_model, apple_label):
        """Creates an endpoint handler for Apple Silicon.
        
        Args:
            local_apple_endpoint: The model endpoint
            local_apple_tokenizer: The tokenizer
            endpoint_model: The model name or path
            apple_label: Label to identify this endpoint
            
        Returns:
            A handler function for the Apple endpoint
        """
        def handler(x, y=None, local_apple_endpoint=local_apple_endpoint, local_apple_tokenizer=local_apple_tokenizer, endpoint_model=endpoint_model, apple_label=apple_label):
            if "eval" in dir(local_apple_endpoint):
                local_apple_endpoint.eval()
            
            try:
                # Check if we're handling text-only or text with optional image
                if y is not None:
                    # Handle multimodal input if needed
                    pass
                
                if x is not None and type(x) == str:
                    conversation = [
                        {
                            "role": "user",
                            "content": x
                        }
                    ]
                elif type(x) == tuple:
                    conversation = x
                elif type(x) == dict:
                    conversation = [x]
                elif type(x) == list:
                    conversation = x
                else:
                    raise Exception("Invalid input to llm endpoint handler")
                
                # Apply chat template and generate
                prompt = local_apple_tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = local_apple_tokenizer(prompt, return_tensors="pt")
                
                # Move to MPS device if available
                if hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
                    for key in inputs:
                        if isinstance(inputs[key], self.torch.Tensor):
                            inputs[key] = inputs[key].to("mps")
                
                output = local_apple_endpoint.generate(**inputs, max_new_tokens=100)
                result = local_apple_tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                return result
            except Exception as e:
                raise e
                
        return handler
        
    def create_qualcomm_llm_endpoint_handler(self, qualcomm_endpoint, qualcomm_tokenizer, endpoint_model, qualcomm_label):
        """Creates an endpoint handler for Qualcomm hardware.
        
        Args:
            qualcomm_endpoint: The model endpoint
            qualcomm_tokenizer: The tokenizer
            endpoint_model: The model name or path
            qualcomm_label: Label to identify this endpoint
            
        Returns:
            A handler function for the Qualcomm endpoint
        """
        def handler(x, y=None, qualcomm_endpoint=qualcomm_endpoint, qualcomm_tokenizer=qualcomm_tokenizer, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label):
            try:
                # Process input
                if x is not None and type(x) == str:
                    input_text = x
                elif type(x) == list:
                    input_text = x[0] if len(x) > 0 else ""
                elif type(x) == dict:
                    input_text = x.get("text", "")
                else:
                    input_text = str(x)
                
                # Qualcomm implementation would use QNN (Qualcomm Neural Network)
                # or SNPE (Snapdragon Neural Processing Engine)
                # This is a placeholder for the actual implementation
                
                pipeline_config = { "MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 512 }
                
                # Actual implementation would use the Qualcomm-specific model and APIs
                return f"Qualcomm would process: {input_text[:20]}..."
            except Exception as e:
                raise e
                
        return handler