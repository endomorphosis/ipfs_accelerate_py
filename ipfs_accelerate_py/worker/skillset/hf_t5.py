import re
import sys
import os
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
        self.init_qualcomm = self.init_qualcomm
        self.init = self.init
        self.__test__ = self.__test__
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
    
    
    def init_cpu (self, model, device, cpu_label):
        self.init() 
        return None
    
    def init_qualcomm(self, model, device, qualcomm_label):
        return None

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        text1  = "The quick brown fox jumps over the lazy dog"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(text1)
            print(test_batch)
            print("hf_t5 test passed")
        except Exception as e:
            print(e)
            print("hf_t5 test failed")
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
            with self.torch.no_grad():
                if "cuda" in dir(self.torch):
                    self.torch.cuda.empty_cache()
        print("hf_t5 test")
        return None
    
    def init_cuda(self, model, device, cuda_label):
        self.init()
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = self.transformers.AutoProcessor.from_pretrained(model)
        endpoint = None
        try:
            endpoint = self.transformers.T5ForConditionalGeneration.from_pretrained(model, torch_dtype=self.torch.float16, trust_remote_code=True).to(device)
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_cuda_mlm_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        self.torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
    
    def init_openvino(self, model, model_type, device, openvino_label, get_optimum_openvino_model, get_openvino_model, get_openvino_pipeline_type, openvino_cli_convert):
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
        endpoint = get_optimum_openvino_model(model, model_type, openvino_label)
        endpoint_handler = self.create_openvino_mlm_endpoint_handler( endpoint, tokenizer, model, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size          
    
    def create_cuda_mlm_endpoint_handler(self, cuda_endpoint_handler, cuda_processor, endpoint_model, cuda_label):
        def handler(x, y=None, cuda_endpoint_handler=cuda_endpoint_handler, cuda_processor=cuda_processor, endpoint_model=endpoint_model, cuda_label=cuda_label):
            results = None
            chat = None
            if x is not None and x is not None:
                chat = x
                pass
            with self.torch.no_grad():
                try:
                    self.torch.cuda.empty_cache()
                    config = self.transformers.AutoConfig.from_pretrained(endpoint_model, trust_remote_code=True)
                    inputs = cuda_processor(chat, return_tensors="pt")
                    outputs = cuda_endpoint_handler.generate(**inputs)
                    results = cuda_processor.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    # Run model inference
                    self.torch.cuda.empty_cache()
                    return results
                except Exception as e:
                    # Cleanup GPU memory in case of error
                    self.torch.cuda.empty_cache()
                    raise e
        return handler
    
    def create_cuda_mlm_endpoint_handler(self, openvino_endpoint_handler, openvino_tokenizer, endpoint_model, openvino_label):
        def handler(x, openvino_endpoint_handler=openvino_endpoint_handler, openvino_tokenizer=openvino_tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label):
            results = None
            chat = None
            if x is not None and x is not None:
                chat = x
            inputs = openvino_tokenizer(chat, return_tensors="pt")
            outputs = openvino_endpoint_handler.generate(**inputs)
            results = openvino_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # streamer = TextStreamer(openvino_tokenizer, skip_prompt=True, skip_special_tokens=True)
            return results
        return handler
    
    
    def create_cpu_mlm_endpoint_handler(self, local_cuda_endpoint, local_cuda_processor, endpoint_model, cuda_label):
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

    def create_openvino_mlm_endpoint_handler(self, openvino_endpoint_handler, openvino_tokenizer, endpoint_model, openvino_label):
        def handler(x, openvino_endpoint_handler=openvino_endpoint_handler, openvino_tokenizer=openvino_tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label):
            results = None
            chat = None
            if x is not None and x is not None:
                chat = x
            
            inputs = openvino_tokenizer(chat, return_tensors="pt")
            outputs = openvino_endpoint_handler.generate(**inputs)
            results = openvino_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # streamer = TextStreamer(openvino_tokenizer, skip_prompt=True, skip_special_tokens=True)
            return results
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
    
