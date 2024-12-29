import asyncio
import os
import torch
import torch.nn.functional as F
from torch import inference_mode, float16, Tensor
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, StoppingCriteriaList, pipeline
from transformers.generation.streamers import TextStreamer
from ipfs_transformers_py import AutoModel
import json
import time

class hf_embed:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_text_embedding_endpoint_handler = self.create_openvino_text_embedding_endpoint_handler
        self.create_cuda_text_embedding_endpoint_handler = self.create_cuda_text_embedding_endpoint_handler
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init = self.init
        self.__test__ = self.__test__
        return None
    
    def init(self):
        return None

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        timestamp1 = time.time()
        test_batch = None
        tokens = tokenizer(sentence_1)["input_ids"]
        len_tokens = len(tokens)
        try:
            test_batch = endpoint_handler(sentence_1)
        except Exception as e:
            print(e)
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"tokens: {len_tokens}")
        print(f"tokens per second: {tokens_per_second}")
        # test_batch_sizes = await self.test_batch_sizes(metadata['models'], ipfs_accelerate_init)
        with torch.no_grad():
            if "cuda" in dir(torch):
                torch.cuda.empty_cache()
        print("hf_embed test")
        return True

    def init_cpu():
        return None

    def init_cuda(self, model, device, cuda_label):
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model, device=device, use_fast=True, trust_remote_code=True)
        try:
            endpoint = AutoModel.from_pretrained(model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
        except Exception as e:
            try:
                endpoint = AutoModel.from_pretrained(model, trust_remote_code=True, device=device)
            except Exception as e:
                print(e)
                pass
        endpoint_handler = self.create_cuda_text_embedding_endpoint_handler(endpoint, cuda_label)
        torch.cuda.empty_cache()
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size

    def init_openvino(self, model_name=None , model_type=None, device=None, openvino_label=None, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None ):
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
        tokenizer =  AutoTokenizer.from_pretrained(
            model_dst_path
        )
        # genai_model = get_openvino_genai_pipeline(model, model_type, openvino_label)
        model = get_optimum_openvino_model(model_name, model_type)
        endpoint_handler = self.create_openvino_text_embedding_endpoint_handler(model_name, tokenizer, openvino_label, model)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size              

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
                results = endpoint.generate(**tokens, max_new_tokens=30)
            except Exception as e:
                print(e)
                try:
                    results = endpoint.generate(text, max_new_tokens=30)
                except Exception as e:
                    print(e)
                pass
            
            return results
        return handler

    def create_cuda_text_embedding_endpoint_handler(self, endpoint_model, cuda_label, endpoint=None, tokenizer=None):
        def handler(x, endpoint_model=endpoint_model, cuda_label=cuda_label, endpoint=endpoint, tokenizer=tokenizer):
            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            with torch.no_grad():
                try:
                    torch.cuda.empty_cache()
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
                    torch.cuda.empty_cache()
                    return result
                except Exception as e:
                    # Cleanup GPU memory in case of error
                    if 'tokens' in locals(): del tokens
                    if 'input_ids' in locals(): del input_ids
                    if 'attention_mask' in locals(): del attention_mask
                    if 'outputs' in locals(): del outputs
                    if 'hidden_states' in locals(): del hidden_states
                    if 'attention_mask_np' in locals(): del attention_mask_np
                    torch.cuda.empty_cache()
                    raise e
        return handler

    # def embed_bak(self, instruction, text , **kwargs):
    # 	self.input = text
    # 	self.method = 'embed'
    # 	embeddings = None
    # 	if "instructor" in self.modelName:
    # 		embeddings = self.model.encode([[instruction,self.input]])
    # 		print(embeddings)
    # 	if "gte" in self.modelName:
    # 		embeddings = self.model.encode([self.input])
    # 		print(embeddings)
    # 	if "bge" in self.modelName:
    # 		if self.model == None:
    # 			self.model = FlagModel(
    # 				'BAAI/'+self.modelName, query_instruction_for_retrieval=instruction,
    # 				use_fp16=True
    # 			)
    # 		embeddings = self.model.encode(str(self.input))
    # 		print(embeddings)

    # 	if type(embeddings) != str:
    # 		embeddings = json.dumps(embeddings.tolist())

    # 	return {
    # 		'text': embeddings, 
    # 		'done': True
    # 	}
        
    def average_pool_bak(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

export = hf_embed
