import re
from torch import inference_mode, float16
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig, AutoProcessor
# from ipfs_transformers_py import AutoModel
import sys
import os
import worker
import time
import asyncio

class hf_t5:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_mlm_endpoint_handler = self.create_openvino_mlm_endpoint_handler
        self.create_cuda_mlm_endpoint_handler = self.create_cuda_mlm_endpoint_handler
        self.create_cpu_mlm_endpoint_handler = self.create_cpu_mlm_endpoint_handler
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init = self.init
        self.__test__ = self.__test__
        return None
    
    def init(self):
        return None

    def init_cpu (self, model, device, cpu_label):
        return None
    

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        text1  = "The quick brown fox jumps over the lazy dog"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(text1)
        except Exception as e:
            print(e)
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        len_tokens = 1
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"samples: {len_tokens}")
        print(f"samples per second: {tokens_per_second}")
        # test_batch_sizes = await self.test_batch_sizes(metadata['models'], ipfs_accelerate_init)
        if "openvino" not in endpoint_label:
            with torch.no_grad():
                if "cuda" in dir(torch):
                    torch.cuda.empty_cache()
        print("hf_t5 test")
        return None
    
    def init_cuda(self, model, device, cuda_label):
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = AutoProcessor.from_pretrained(model)
        endpoint = None
        try:
            endpoint = T5ForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_cuda_mlm_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
    
    def init_openvino(self, model, model_type, device, openvino_label, get_optimum_openvino_model, get_openvino_model, get_openvino_pipeline_type, openvino_cli_convert):
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        batch_size = 0                
        tokenizer =  AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        endpoint = get_openvino_model(model, model_type, openvino_label)
        endpoint_handler = self.create_openvino_mlm_endpoint_handler( endpoint, tokenizer, model, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size          
    
    def create_cuda_mlm_endpoint_handler(self, local_cuda_endpoint, local_cuda_processor, endpoint_model, cuda_label):
        def handler(x, y=None, local_cuda_endpoint=local_cuda_endpoint, local_cuda_processor=local_cuda_processor, endpoint_model=endpoint_model, cuda_label=cuda_label):
            # if "eval" in dir(self.local_endpoints[endpoint_model][cuda_label]):
            #       self.local_endpoints[endpoint_model][cuda_label].eval()
            if "eval" in dir(local_cuda_endpoint):
                local_cuda_endpoint.eval()
            else:
                pass
            with torch.no_grad():
                try:
                    torch.cuda.empty_cache()
                    config = AutoConfig.from_pretrained(endpoint_model, trust_remote_code=True)
                    
                    # Run model inference
                    torch.cuda.empty_cache()
                except Exception as e:
                    # Cleanup GPU memory in case of error
                    torch.cuda.empty_cache()
                    raise e
        return handler
    
    def create_cpu_mlm_endpoint_handler(self, local_cuda_endpoint, local_cuda_processor, endpoint_model, cuda_label):
        def handler(x, y=None, local_cuda_endpoint=local_cuda_endpoint, local_cuda_processor=local_cuda_processor, endpoint_model=endpoint_model, cuda_label=cuda_label):
            # if "eval" in dir(self.local_endpoints[endpoint_model][cuda_label]):
            #       self.local_endpoints[endpoint_model][cuda_label].eval()
            if "eval" in dir(local_cuda_endpoint):
                local_cuda_endpoint.eval()
            else:
                pass
            with torch.no_grad():
                try:
                    torch.cuda.empty_cache()
                    config = AutoConfig.from_pretrained(endpoint_model, trust_remote_code=True)
                    
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
                    inputs = local_cuda_processor(prompt, return_tensors="pt").to(cuda_label, torch.float16)
                    output = local_cuda_endpoint.generate(**inputs, max_new_tokens=30)
                    result = local_cuda_processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    # Run model inference
                    torch.cuda.empty_cache()
                    return result
                except Exception as e:
                    # Cleanup GPU memory in case of error
                    torch.cuda.empty_cache()
                    raise e
        return handler

    def create_openvino_mlm_endpoint_handler(self, openvino_endpoint_handler, openvino_tokenizer, endpoint_model, openvino_label):
        def handler(x, openvino_endpoint_handler=openvino_endpoint_handler, openvino_tokenizer=openvino_tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label):
            chat = None
            if x is not None and x is not None:
                chat = x
            inputs = openvino_tokenizer(chat, return_tensors="pt")
            # prompt = openvino_endpoint_handler.apply_chat_template(chat, add_generation_prompt=True)
            outputs = openvino_endpoint_handler(**inputs)
            results = openvino_tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # streamer = TextStreamer(openvino_tokenizer, skip_prompt=True, skip_special_tokens=True)
            return results
        return handler
    
    # def __init__(self, resources, meta=None):
    # 	self.tokenizer = T5Tokenizer.from_pretrained(
    # 		resources['checkpoint'], 
    # 		local_files_only=True,
    # 		device_map='auto',
    # 		legacy=False
    # 	)
    # 	self.model = T5ForConditionalGeneration.from_pretrained(
    # 		resources['checkpoint'], 
    # 		local_files_only=True, 
    # 		low_cpu_mem_usage=True,
    # 		device_map='auto',
    # 		torch_dtype=float16,
    # 	).eval()
    # 	self.worker = worker
    # 	self.TaskAbortion = self.worker.TaskAbortion
    # 	self.should_abort = self.worker.should_abort

    # def __call__(self, method, **kwargs):
    # 	if method == 'instruct_t5':
    # 		return self.instruct_t5(**kwargs)
    # 	elif method == 'unmask_t5':
    # 		return self.unmask_t5(**kwargs)
    # 	else:
    # 		raise Exception('unknown method: %s' % method)

    # def instruct_t5(self, instruction, input, max_tokens,  **kwargs):

    # 	input_ids = self.tokenizer(instruction + input , return_tensors="pt").input_ids.to(self.model.device)
    # 	outputs = self.model.generate(input_ids, max_length=max_tokens)
    # 	#print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

    # 	return {
    #         'text': self.tokenizer.decode(outputs[0], skip_special_tokens=True), 
    #         'done': True
    #     }

    # def unmask_t5(self, masked_words, input, max_tokens, **kwargs):
    # 	if masked_words is None:
    # 		masked_words = []
    # 	if isinstance(masked_words, str):
    # 		masked_words = [masked_words]
    # 	if not isinstance(masked_words, list):
    # 		raise Exception('masked_words must be a list of strings')
    # 	else:
    # 		masked_words_len = len(masked_words)
    # 		for i in range(masked_words_len):
    # 			input = input.replace(masked_words[i], "<extra_id_"+str(i)+">")
    # 		pass
                
    # 	input_ids = self.tokenizer(input, return_tensors="pt").input_ids.to(self.model.device)
    # 	sequence_ids = self.model.generate(input_ids, max_length=max_tokens )
    # 	sequence = self.tokenizer.decode(sequence_ids[0])
    # 	print(sequence)
    # 	sequences = []
    # 	for i in range(len(sequence_ids)):
    # 		sequences.append(self.tokenizer.decode(sequence_ids[i]))
    # 	print(sequences)
    # 	return {
    #         'text': sequences, 
    #         'done': True
    #     }
    
#if __name__ == '__main__':
#    test_t5 = HF_T5({'checkpoint': '/storage/cloudkit-models/flan-t5-small@hf/'})
#    print(test_t5.task("translate English to German: ", "How old are you?", 100))