import requests
from PIL import Image
from io import BytesIO
import json
import asyncio
from pathlib import Path
import json
import os
import time
    
class hf_qwen2:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.init = self.init
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_cpu = self.init_cpu
        self.init_apple = self.init_apple
        self.init_qualcomm = self.init_qualcomm
        self.__test__ = self.__test__
        self.create_openvino_llm_endpoint_handler = self.create_openvino_llm_endpoint_handler
        self.create_cpu_llm_endpoint_handler = self.create_cpu_llm_endpoint_handler
        self.create_cuda_llm_endpoint_handler = self.create_cuda_llm_endpoint_handler
        self.create_apple_llm_endpoint_handler = self.create_apple_llm_endpoint_handler
        self.create_qualcomm_llm_endpoint_handler = self.create_qualcomm_llm_endpoint_handler
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
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        sentence_2 = "The quick brown fox jumps over the lazy dog"
        image_1 = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, image_1)
        except Exception as e:
            print(e)
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
        try:
            config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
            tokenizer = self.transformers.AutoProcessor.from_pretrained(model)
            endpoint = self.transformers.AutoModelForImageTextToText.from_pretrained(model, trust_remote_code=True)
            endpoint_handler = self.create_cpu_llm_endpoint_handler(endpoint, tokenizer, model, cpu_label)
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
        except Exception as e:
            print(f"Error initializing CPU model: {e}")
            return None, None, None, None, 0
    
    def init_apple(self, model, device, apple_label):
        self.init()
        try:
            if "coremltools" not in list(self.resources.keys()):
                import coremltools as ct
                self.ct = ct
            else:
                self.ct = self.resources["coremltools"]
                
            config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
            tokenizer = self.transformers.AutoProcessor.from_pretrained(model)
            
            # In a real implementation, we would convert and load a CoreML model
            # For now, we'll load the standard model as a placeholder
            endpoint = self.transformers.AutoModelForImageTextToText.from_pretrained(model, trust_remote_code=True)
            
            endpoint_handler = self.create_apple_llm_endpoint_handler(endpoint, tokenizer, model, apple_label)
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
        except ImportError:
            print("coremltools not installed. Can't initialize Apple backend.")
            return None, None, None, None, 0
        except Exception as e:
            print(f"Error initializing Apple model: {e}")
            return None, None, None, None, 0
    
    def init_qualcomm(self, model, model_type, device, qualcomm_label):
        self.init()
        try:
            config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
            tokenizer = self.transformers.AutoProcessor.from_pretrained(model)
            
            # Here we would initialize the model using Qualcomm's SDK
            # This is a placeholder for actual Qualcomm initialization
            endpoint = None
            
            endpoint_handler = self.create_qualcomm_llm_endpoint_handler(endpoint, tokenizer, model, qualcomm_label)
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
        except Exception as e:
            print(f"Error initializing Qualcomm model: {e}")
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
        endpoint_handler = self.create_cuda_llm_endpoint_handler(endpoint, tokenizer, model, cuda_label)
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
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        endpoint = get_openvino_model(model, model_type, openvino_label)
        endpoint_handler = self.create_openvino_llm_endpoint_handler(endpoint, tokenizer, model, openvino_label)
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
                    inputs = local_cuda_processor(prompt, return_tensors="pt")
                    output = local_cuda_endpoint.generate(**inputs, max_new_tokens=30)
                    result = local_cuda_processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    return result
                except Exception as e:
                    print(f"Error in CPU endpoint handler: {e}")
                    raise e
        return handler
        
    def create_apple_llm_endpoint_handler(self, local_endpoint, local_processor, endpoint_model, apple_label):
        def handler(x, y=None, local_endpoint=local_endpoint, local_processor=local_processor, endpoint_model=endpoint_model, apple_label=apple_label):
            if "eval" in dir(local_endpoint):
                local_endpoint.eval()
                
            try:
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
                
                # In a real implementation, this would use the CoreML model for inference
                # This is a placeholder implementation
                prompt = local_processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = local_processor(prompt, return_tensors="pt")
                output = local_endpoint.generate(**inputs, max_new_tokens=30)
                result = local_processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                return result
            except Exception as e:
                print(f"Error in Apple endpoint handler: {e}")
                raise e
        return handler
        
    def create_qualcomm_llm_endpoint_handler(self, local_endpoint, local_processor, endpoint_model, qualcomm_label):
        def handler(x, y=None, local_endpoint=local_endpoint, local_processor=local_processor, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label):
            # This is a placeholder for Qualcomm-specific implementation
            try:
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
                
                # Placeholder for Qualcomm-specific inference
                return {"message": "Qualcomm inference not fully implemented for Qwen2"}
            except Exception as e:
                print(f"Error in Qualcomm endpoint handler: {e}")
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

    def init_qualcomm(self, model, device, qualcomm_label):
        """Initialize Qwen2 model for Qualcomm hardware.
        
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
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            
            # Convert model path to be compatible with SNPE
            model_name = model.replace("/", "--")
            dlc_path = f"~/snpe_models/{model_name}_qwen2.dlc"
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
            endpoint_handler = self.create_qualcomm_qwen2_endpoint_handler(tokenizer, model, qualcomm_label, endpoint)
            
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(16), 1
        except Exception as e:
            print(f"Error initializing Qualcomm Qwen2 model: {e}")
            return None, None, None, None, 0

    def create_qualcomm_qwen2_endpoint_handler(self, tokenizer, endpoint_model, qualcomm_label, endpoint):
        """Create a handler for Qualcomm-based Qwen2 inference
        
        Args:
            tokenizer: HuggingFace tokenizer
            endpoint_model: Name of the model
            qualcomm_label: Label for Qualcomm hardware
            endpoint: SNPE model endpoint
            
        Returns:
            Handler function for inference
        """
        def handler(text_input, tokenizer=tokenizer, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint):
            try:
                # Tokenize input
                if isinstance(text_input, str):
                    inputs = tokenizer(text_input, return_tensors="np", padding=True)
                else:
                    # Assume it's already tokenized, convert to numpy if needed
                    inputs = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in text_input.items()}
                
                # Initial input for the model
                model_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }
                
                # Prepare for token-by-token generation
                generated_ids = inputs["input_ids"].tolist()[0]
                past_key_values = None
                max_new_tokens = 256
                
                # Generate tokens one by one
                for _ in range(max_new_tokens):
                    # Add KV cache to inputs if we have it
                    if past_key_values is not None:
                        for i, (k, v) in enumerate(past_key_values):
                            model_inputs[f"past_key_values.{i}.key"] = k
                            model_inputs[f"past_key_values.{i}.value"] = v
                    
                    # Get the next token logits from the model
                    results = self.snpe_utils.run_inference(endpoint, model_inputs)
                    
                    # Get the logits
                    if "logits" in results:
                        logits = self.np.array(results["logits"])
                        
                        # Save KV cache if provided
                        if "past_key_values" in results:
                            past_key_values = results["past_key_values"]
                        
                        # Basic greedy decoding
                        next_token_id = int(self.np.argmax(logits[0, -1, :]))
                        
                        # Add the generated token
                        generated_ids.append(next_token_id)
                        
                        # Check for EOS token
                        if next_token_id == tokenizer.eos_token_id:
                            break
                            
                        # Update inputs for next iteration
                        model_inputs = {
                            "input_ids": self.np.array([[next_token_id]]),
                            "attention_mask": self.np.array([[1]])
                        }
                    else:
                        break
                
                # Decode the generated sequence
                decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Return result
                return {
                    "generated_text": decoded_output,
                    "model": endpoint_model
                }
                
            except Exception as e:
                print(f"Error in Qualcomm Qwen2 endpoint handler: {e}")
                return {"error": str(e)}
                
        return handler