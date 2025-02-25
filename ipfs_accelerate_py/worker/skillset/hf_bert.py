import asyncio
import os
import json
import time

class hf_bert:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_text_embedding_endpoint_handler = self.create_openvino_text_embedding_endpoint_handler
        self.create_cuda_text_embedding_endpoint_handler = self.create_cuda_text_embedding_endpoint_handler
        self.create_cpu_text_embedding_endpoint_handler = self.create_cpu_text_embedding_endpoint_handler
        self.create_apple_text_embedding_endpoint_handler = self.create_apple_text_embedding_endpoint_handler
        self.create_qualcomm_text_embedding_endpoint_handler = self.create_qualcomm_text_embedding_endpoint_handler
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_qualcomm = self.init_qualcomm
        self.init_apple = self.init_apple
        self.init = self.init
        self.__test__ = self.__test__
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

    def init_cpu(self, model, device, cpu_label):
        self.init()
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        try:
            endpoint = self.transformers.AutoModel.from_pretrained(model, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading CPU model: {e}")
            endpoint = None
            
        endpoint_handler = self.create_cpu_text_embedding_endpoint_handler(endpoint, cpu_label, endpoint, tokenizer)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(32), 0

    def init_cuda(self, model, device, cuda_label):
        self.init()
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model, device=device, use_fast=True, trust_remote_code=True)
        try:
            endpoint = self.transformers.AutoModel.from_pretrained(model, torch_dtype=self.torch.float16, trust_remote_code=True).to(device)
        except Exception as e:
            try:
                endpoint = self.transformers.AutoModel.from_pretrained(model, trust_remote_code=True, device=device)
            except Exception as e:
                print(e)
                pass
        endpoint_handler = self.create_cuda_text_embedding_endpoint_handler(endpoint, cuda_label)
        self.torch.cuda.empty_cache()
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size

    def init_openvino(self, model_name=None , model_type=None, device=None, openvino_label=None, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None ):
        self.init()
        if "openvino" not in list(self.resources.keys()):
            import openvino as ov
            self.ov = ov
        else:
            self.ov = self.resources["openvino"]
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        homedir = os.path.expanduser("~")
        model_name_convert = model_name.replace("/", "--")
        huggingface_cache = os.path.join(homedir, ".cache/huggingface")
        huggingface_cache_models = os.path.join(huggingface_cache, "hub")
        huggingface_cache_models_files = os.listdir(huggingface_cache_models)
        huggingface_cache_models_files_dirs = [os.path.join(huggingface_cache_models, file) for file in huggingface_cache_models_files if os.path.isdir(os.path.join(huggingface_cache_models, file))]
        huggingface_cache_models_files_dirs_models = [ x for x in huggingface_cache_models_files_dirs if "model" in x ]
        huggingface_cache_models_files_dirs_models_model_name = [ x for x in huggingface_cache_models_files_dirs_models if model_name_convert in x ]
        model_src_path = os.path.join(huggingface_cache_models, huggingface_cache_models_files_dirs_models_model_name[0])
        model_dst_path = os.path.join(model_src_path, "openvino")
        # config = AutoConfig.from_pretrained(model)
        task = get_openvino_pipeline_type(model_name, model_type)
        openvino_index = int(openvino_label.split(":")[1])
        weight_format = ""
        if openvino_index is not None:
            if openvino_index == 0:
                weight_format = "int8" ## CPU
            if openvino_index == 1:
                weight_format = "int4" ## gpu
            if openvino_index == 2:
                weight_format = "int4" ## npu
        model_dst_path = model_dst_path+"_"+weight_format
        if not os.path.exists(model_dst_path):
            os.makedirs(model_dst_path)
            openvino_cli_convert(model_name, model_dst_path=model_dst_path, task=task, weight_format=weight_format, ratio="1.0", group_size=128, sym=True )
        try:
            tokenizer =  self.transformers.AutoTokenizer.from_pretrained(
                model_src_path
            )
        except Exception as e:
            print(e)
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                model_name
            )
            pass
        # genai_model = get_openvino_genai_pipeline(model, model_type, openvino_label)
        model = get_optimum_openvino_model(model_name, model_type)
        endpoint_handler = self.create_openvino_text_embedding_endpoint_handler(model_name, tokenizer, openvino_label, model)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size              

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
        
        # Qualcomm initialization would use SNPE (Snapdragon Neural Processing Engine)
        # This is a placeholder implementation
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        
        # SNPE would typically convert the PyTorch model to a Qualcomm-specific format
        endpoint = None
            
        endpoint_handler = self.create_qualcomm_text_embedding_endpoint_handler(endpoint, qualcomm_label, endpoint, tokenizer)
        
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(32), 0

    def create_cpu_text_embedding_endpoint_handler(self, endpoint_model, cpu_label, endpoint=None, tokenizer=None):
        def handler(x, endpoint_model=endpoint_model, cpu_label=cpu_label, endpoint=endpoint, tokenizer=tokenizer):
            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            with self.torch.no_grad():
                try:
                    tokens = tokenizer(x, return_tensors="pt")
                    results =  endpoint(**tokens)
                    # average_pool_results = self.average_pool(results.last_hidden_state, tokens['attention_mask'])
                    last_hidden = results.last_hidden_state.masked_fill(~tokens['attention_mask'].bool(), 0.0)
                    average_pool_results =  last_hidden.sum(dim=1) / tokens['attention_mask'].sum(dim=1)[..., None]

                    return average_pool_results
                except Exception as e:
                    print(e)
                    pass
        return handler

    def create_openvino_text_embedding_endpoint_handler(self, endpoint_model, tokenizer,  openvino_label, endpoint=None):
        def handler(x, tokenizer=tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=endpoint):
            text = None
            tokens = None
            if type(x) == str:
                text = x
                tokens = tokenizer(text, return_tensors="pt")
            elif type(x) == list:
                if "input_ids" in x[0].keys():
                    tokens = x
                else:
                    text = x
                    tokens = tokenizer(text, return_tensors="pt")
            elif type(x) == dict:
                if "input_ids" in x.keys():
                    tokens = x
                else:
                    text = x
                    tokens = tokenizer(text, return_tensors="pt")

            try:
                results =  endpoint(**tokens)
            except Exception as e:
                print(e)
                pass
            
            # average_pool_results = self.average_pool(results.last_hidden_state, tokens['attention_mask'])
            
            last_hidden = results.last_hidden_state.masked_fill(~tokens['attention_mask'].bool(), 0.0)
            average_pool_results =  last_hidden.sum(dim=1) / tokens['attention_mask'].sum(dim=1)[..., None]

            return average_pool_results
        
        return handler

    def create_cuda_text_embedding_endpoint_handler(self, endpoint_model, cuda_label, endpoint=None, tokenizer=None):
        def handler(x, endpoint_model=endpoint_model, cuda_label=cuda_label, endpoint=endpoint, tokenizer=tokenizer):
            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            with self.torch.no_grad():
                try:
                    self.torch.cuda.empty_cache()
                    # Tokenize input with truncation and padding
                    tokens = tokenizer[endpoint_model][cuda_label](
                        x, 
                        return_tensors='pt', 
                        padding=True, 
                        truncation=True,
                        max_length=endpoint.config.max_position_embeddings
                    )
                    
                    # Move tokens to the correct device
                    input_ids = tokens['input_ids'].to(endpoint.device)
                    attention_mask = tokens['attention_mask'].to(endpoint.device)
                    
                    # Run model inference
                    outputs = endpoint(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                        
                    # Process and prepare outputs
                    if hasattr(outputs, 'last_hidden_state'):
                        hidden_states = outputs.last_hidden_state.cpu().numpy()
                        attention_mask_np = attention_mask.cpu().numpy()
                        result = {
                            'hidden_states': hidden_states,
                            'attention_mask': attention_mask_np
                        }
                    else:
                        result = outputs.to('cpu').detach().numpy()

                    # Cleanup GPU memory
                    del tokens, input_ids, attention_mask, outputs
                    if 'hidden_states' in locals(): del hidden_states
                    if 'attention_mask_np' in locals(): del attention_mask_np
                    self.torch.cuda.empty_cache()
                    return result
                except Exception as e:
                    # Cleanup GPU memory in case of error
                    if 'tokens' in locals(): del tokens
                    if 'input_ids' in locals(): del input_ids
                    if 'attention_mask' in locals(): del attention_mask
                    if 'outputs' in locals(): del outputs
                    if 'hidden_states' in locals(): del hidden_states
                    if 'attention_mask_np' in locals(): del attention_mask_np
                    self.torch.cuda.empty_cache()
                    raise e
        return handler
        
    def create_apple_text_embedding_endpoint_handler(self, endpoint_model, apple_label, endpoint=None, tokenizer=None):
        """Creates an endpoint handler for Apple Silicon.
        
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
                            return_tensors='pt', 
                            padding=True, 
                            truncation=True,
                            max_length=endpoint.config.max_position_embeddings
                        )
                    elif type(x) == list:
                        tokens = tokenizer(
                            x, 
                            return_tensors='pt', 
                            padding=True, 
                            truncation=True,
                            max_length=endpoint.config.max_position_embeddings
                        )
                    else:
                        tokens = x
                    
                    # Move tokens to MPS device
                    if hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
                        input_ids = tokens['input_ids'].to("mps")
                        attention_mask = tokens['attention_mask'].to("mps")
                    else:
                        input_ids = tokens['input_ids']
                        attention_mask = tokens['attention_mask']
                    
                    # Run model inference
                    outputs = endpoint(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    
                    # Get embeddings using mean pooling
                    last_hidden = outputs.last_hidden_state.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
                    average_pool_results = last_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                    
                    # Move results back to CPU
                    result = average_pool_results.cpu()
                    
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
                        return_tensors='pt', 
                        padding=True, 
                        truncation=True,
                        max_length=512  # Default max length
                    )
                elif type(x) == list:
                    tokens = tokenizer(
                        x, 
                        return_tensors='pt', 
                        padding=True, 
                        truncation=True,
                        max_length=512  # Default max length
                    )
                else:
                    tokens = x
                
                # This is a placeholder for Qualcomm-specific implementation
                # Actual implementation would use SNPE (Snapdragon Neural Processing Engine)
                
                # Create dummy output for placeholder
                result = self.np.zeros((tokens['input_ids'].shape[0], 768))  # Default embedding size
                
                return result
            except Exception as e:
                print(f"Error in Qualcomm text embedding handler: {e}")
                raise e
                
        return handler

