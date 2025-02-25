from PIL import Image
from io import BytesIO
from pathlib import Path
import os
import time
import asyncio
import torch
import numpy as np

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
            config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model)
            
            # Initialize model for CPU
            endpoint = self.transformers.AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
            
            endpoint_handler = self.create_cpu_llama_endpoint_handler(tokenizer, model, cpu_label, endpoint)
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(16), 1
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
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model)
            
            # Check if MPS is available
            if not hasattr(self.torch.backends, 'mps') or not self.torch.backends.mps.is_available():
                print("MPS not available. Cannot initialize model on Apple Silicon.")
                return None, None, None, None, 0
            
            # Initialize model for Apple Silicon
            endpoint = self.transformers.AutoModelForCausalLM.from_pretrained(
                model, 
                torch_dtype=self.torch.float16 if self.torch.backends.mps.is_available() else None,
                trust_remote_code=True
            ).to(device)
            
            endpoint_handler = self.create_apple_llama_endpoint_handler(tokenizer, model, apple_label, endpoint)
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(16), 1
        except Exception as e:
            print(f"Error initializing Apple model: {e}")
            return None, None, None, None, 0
    
    def init_cuda(self, model, device, cuda_label):
        self.init()
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model)
        endpoint = None
        try:
            endpoint = self.transformers.AutoModelForCausalLM.from_pretrained(
                model, 
                torch_dtype=self.torch.float16, 
                trust_remote_code=True
            ).to(device)
        except Exception as e:
            print(e)
            pass
            
        endpoint_handler = self.create_cuda_llama_endpoint_handler(tokenizer, endpoint_model=model, cuda_label=cuda_label, endpoint=endpoint)
        self.torch.cuda.empty_cache()
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(16), 1

    def init_qualcomm(self, model, model_type, device, qualcomm_label, get_qualcomm_genai_pipeline, get_optimum_qualcomm_model, get_qualcomm_model, get_qualcomm_pipeline_type):
        """
        Initialize LLaMA model for Qualcomm hardware
        
        Args:
            model: Model name or path
            model_type: Type of model
            device: Device to run on
            qualcomm_label: Label for Qualcomm hardware
            
        Returns:
            Initialized model components
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
            
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(16), 1
            
        except Exception as e:
            print(f"Error initializing Qualcomm LLaMA model: {e}")
            return None, None, None, None, 0
    
    def create_cpu_llama_endpoint_handler(self, tokenizer, model_name, cpu_label, endpoint):
        """Create a handler for CPU-based LLaMA inference"""
        
        def handler(text_input, tokenizer=tokenizer, model_name=model_name, cpu_label=cpu_label, endpoint=endpoint):
            if "eval" in dir(endpoint):
                endpoint.eval()
            
            try:
                # Tokenize input
                if isinstance(text_input, str):
                    inputs = tokenizer(text_input, return_tensors="pt")
                else:
                    # Assume it's already tokenized
                    inputs = text_input
                
                # Run generation
                with self.torch.no_grad():
                    outputs = endpoint.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                # Decode output
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Return result
                return {
                    "generated_text": decoded_output,
                    "model_name": model_name
                }
                
            except Exception as e:
                print(f"Error in CPU LLaMA endpoint handler: {e}")
                return {"error": str(e)}
                
        return handler
        
    def create_apple_llama_endpoint_handler(self, tokenizer, model_name, apple_label, endpoint):
        """Create a handler for Apple Silicon-based LLaMA inference"""
        
        def handler(text_input, tokenizer=tokenizer, model_name=model_name, apple_label=apple_label, endpoint=endpoint):
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
                        attention_mask=inputs.get("attention_mask", None),
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                # Move back to CPU for decoding
                if hasattr(outputs, 'cpu'):
                    outputs = outputs.cpu()
                    
                # Decode output
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Return result
                return {
                    "generated_text": decoded_output,
                    "model_name": model_name
                }
                
            except Exception as e:
                print(f"Error in Apple LLaMA endpoint handler: {e}")
                return {"error": str(e)}
                
        return handler
    
    def create_cuda_llama_endpoint_handler(self, tokenizer, endpoint_model, cuda_label, endpoint):
        """Create a handler for CUDA-based LLaMA inference"""
        
        def handler(text_input, tokenizer=tokenizer, endpoint_model=endpoint_model, cuda_label=cuda_label, endpoint=endpoint):
            if "eval" in dir(endpoint):
                endpoint.eval()
                
            with self.torch.no_grad():
                try:
                    self.torch.cuda.empty_cache()
                    
                    # Tokenize input
                    if isinstance(text_input, str):
                        inputs = tokenizer(text_input, return_tensors="pt").to(cuda_label)
                    else:
                        # Assume it's already tokenized
                        inputs = {k: v.to(cuda_label) if hasattr(v, 'to') else v for k, v in text_input.items()}
                    
                    # Run generation
                    outputs = endpoint.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                    # Decode output
                    decoded_output = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
                    
                    # Cleanup
                    self.torch.cuda.empty_cache()
                    
                    # Return result
                    return {
                        "generated_text": decoded_output,
                        "model_name": endpoint_model
                    }
                    
                except Exception as e:
                    self.torch.cuda.empty_cache()
                    print(f"Error in CUDA LLaMA endpoint handler: {e}")
                    return {"error": str(e)}
                    
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
                    "model_name": model_name
                }
                
            except Exception as e:
                print(f"Error in Qualcomm LLaMA endpoint handler: {e}")
                return {"error": str(e)}
                
        return handler