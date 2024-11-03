import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_ipfs_accelerate import test_ipfs_accelerate
from install_depends import install_depends_py
import optimum
import torch 
import asyncio
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, pipeline
import ipfs_transformers_py
from pathlib import Path
import numpy as np
import torch

class worker_py:
    def __init__(self, metadata, resources):
        self.metadata = metadata
        self.resources = resources
        self.tokenizer = {}
        self.queues = {}
        self.batch_sizes = {}
        self.endpoint_handler = {}
        self.endpoints = {}
        self.endpoint_types = ["local_endpoints"]
        self.local_endpoints = {}
        self.hardware_backends = ["llama_cpp", "cpu", "gpu", "openvino", "optimum", "optimum_intel", "optimum_openvino", "optimum_ipex", "optimum_neural_compressor"]
        if "test_ipfs_accelerate" not in globals():
            self.test_ipfs_accelerate = test_ipfs_accelerate(resources, metadata)
            self.hwtest = self.test_ipfs_accelerate
        if "install_depends" not in globals():
            self.install_depends = install_depends_py(resources, metadata)
        if "ipfs_transformers_py" not in globals() and "ipfs_transformers_py" not in list(self.resources.keys()):
            from ipfs_transformers_py import ipfs_transformers
            self.ipfs_transformers = { 
                # "AutoDownloadModel" : ipfs_transformers.AutoDownloadModel()
            }        
        elif "ipfs_transformers_py" in list(self.resources.keys()):
            self.ipfs_transformers_py = self.resources["ipfs_transformers_py"]
        elif "ipfs_transformers_py" in globals():
            from ipfs_transformers_py import ipfs_transformers
            self.ipfs_transformers = { 
                # "AutoDownloadModel" : ipfs_transformers.AutoDownloadModel()
            }
        if "transformers" not in globals() and "transformers" not in list(self.resources.keys()):
            self.transformers = transformers   
        elif "transformers" in list(self.resources.keys()):
            self.transformers = self.resources["transformers"]
        elif "transformers" in globals():
            self.transformers = transformers
        if "torch" not in globals() and "torch" not in list(self.resources.keys()):
            self.torch = torch
        elif "torch" in list(self.resources.keys()):
            self.torch = self.resources["torch"]
        elif "torch" in globals():
            self.torch = torch
        else:
            pass
        for endpoint in self.endpoint_types:
            if endpoint not in dir(self):
                self.__dict__[endpoint] = {}        
            for backend in self.hardware_backends:
                if backend not in list(self.__dict__[endpoint].keys()):
                    self.__dict__[endpoint][backend] = {}
        print(self.__dict__)
        return None
    
    async def test_hardware(self):
        return await self.install_depends.test_hardware()
    
    async def get_model_type(self, model_name, model_type=None):
        if model_type is None:
            config = AutoConfig.from_pretrained(model_name)
            model_type = config.__class__.model_type
        return model_type
    
    async def get_openvino_model(self, model_name, model_type=None):
        if model_type is None:
            config = AutoConfig.from_pretrained(model_name)
            model_type = config.__class__.model_type
        model_mapping_list = ["text-classification", "token-classification", "question-answering", "audio-classification", "image-classification", "feature-extraction", "fill-mask", "text-generation-with-past", "text2text-generation-with-past", "automatic-speech-recognition", "image-to-text", "bert"]
        if model_type not in model_mapping_list:
            return None
        if model_type == "bert":
            model_type = "text-classification"
            from optimum.intel import OVModelForSequenceClassification
            results = OVModelForSequenceClassification.from_pretrained(model_name, compile=False)
        elif model_type == "text-classification":
            from optimum.intel import OVModelForSequenceClassification
            results = OVModelForSequenceClassification.from_pretrained(model_name, compile=False)
        elif model_type == "token-classification":
            from optimum.intel import OVModelForTokenClassification
            results = OVModelForTokenClassification.from_pretrained(model_name, compile=False)
        elif model_type == "question-answering":
            from optimum.intel import OVModelForQuestionAnswering
            results = OVModelForQuestionAnswering.from_pretrained(model_name, compile=False)
        elif model_type == "audio-classification":
            from optimum.intel import OVModelForAudioClassification
            results = OVModelForAudioClassification.from_pretrained(model_name,  compile=False)
        elif model_type == "image-classification":
            from optimum.intel import OVModelForImageClassification
            results = OVModelForImageClassification.from_pretrained(model_name, compile=False) 
        elif model_type == "feature-extraction":
            from optimum.intel import OVModelForFeatureExtraction
            results = OVModelForFeatureExtraction.from_pretrained(model_name, compile=False)
        elif model_type == "fill-mask":
            from optimum.intel import OVModelForMaskedLM
            results = OVModelForMaskedLM.from_pretrained(model_name, compile=False)
        elif model_type == "text-generation-with-past":
            from optimum.intel import OVModelForCausalLM
            results = OVModelForCausalLM.from_pretrained(model_name, compile=False)
        elif model_type == "text2text-generation-with-past":
            from optimum.intel import OVModelForSeq2SeqLM
            results = OVModelForSeq2SeqLM.from_pretrained(model_name, compile=False)
        elif model_type == "automatic-speech-recognition":
            from optimum.intel import OVModelForSpeechSeq2Seq
            results = OVModelForSpeechSeq2Seq.from_pretrained(model_name, compile=False)
        elif model_type == "image-to-text":
            from optimum.intel import OVModelForVision2Seq
            results = OVModelForVision2Seq.from_pretrained(model_name, compile=False)
        else:
            return None
        # await results.compile()
        return results
    
    async def get_openvino_pipeline_type(self, model_name, model_type=None):
        model_mapping_list = ["text-classification", "token-classification", "question-answering", "audio-classification", "image-classification", "feature-extraction", "fill-mask", "text-generation-with-past", "text2text-generation-with-past", "automatic-speech-recognition", "image-to-text"]
        if model_type is None:
            config = AutoConfig.from_pretrained(model_name)
            model_type = config.__class__.model_type
        if model_type not in model_mapping_list:
            if model_type == "bert":
                model_type = "text-classification"
            else:
                return None
        if model_type == None:
            return None
        return model_type

    async def init_worker(self, models, local_endpoints, hwtest):
        if local_endpoints is None or len(local_endpoints) == 0:
            if "local_endpoints" in list(self.__dict__.keys()):
                local_endpoints = self.local_endpoints
            elif "local_endpoints" in list(self.resources.keys()):
                local_endpoints = self.resources["local_endpoints"]
            else:
                local_endpoints = []
        else:
            pass
        self.local_endpoints  = local_endpoints
        self.local_endpoint_types = [ x[1] for x in local_endpoints if x[0] in models]
        self.local_endpoint_models = [ x[0] for x in local_endpoints if x[0] in models]
        self.local_endpoints = { model: { endpoint[1]: endpoint[2] for endpoint in local_endpoints if endpoint[0] == model} for model in models}
        local = len(local_endpoints) > 0 if isinstance(self.local_endpoints, dict) and len(list(self.local_endpoints.keys())) > 0 else False
        if hwtest is None:
            if "hwtest" in list(self.__dict__.keys()):
                hwtest = self.hwtest
            elif "hwtest" in list(self.resources.keys()):
                hwtest = self.resources["hwtest"]
            else:
                hwtest = await self.hwtest()
        self.hwtest = hwtest
        cuda_test = self.hwtest["cuda"]
        openvino_test = self.hwtest["openvino"]
        llama_cpp_test = self.hwtest["llama_cpp"]
        ipex_test = self.hwtest["ipex"]
        cpus = os.cpu_count()
        cuda = torch.cuda.is_available()
        gpus = torch.cuda.device_count()
        
        for model in models:
            model_type = await self.get_model_type(model)
            if model not in self.tokenizer:
                self.tokenizer[model] = {}
            if model not in self.local_endpoints:
                self.local_endpoints[model] = {}
            if model not in self.queues:
                self.queues[model] = {}
            if model not in self.batch_sizes:
                self.batch_sizes[model] = {}
            if cuda and gpus > 0:
                if cuda_test and type(cuda_test) != ValueError:
                    for gpu in range(gpus):
                        cuda_index = self.local_endpoint_types.index("cuda:"+str(gpu))
                        endpoint_model = self.local_endpoint_models[cuda_index]
                        cuda_label = self.local_endpoint_types[cuda_index]
                        self.tokenizer[endpoint_model][cuda_label] = AutoTokenizer.from_pretrained(model, device='cuda:' + str(gpu), use_fast=True)
                        self.local_endpoints[endpoint_model][cuda_label] = AutoModel.from_pretrained(model).to("cuda:" + str(gpu))
                        self.endpoint_handler[(endpoint_model, cuda_label)] = self.local_endpoints[endpoint_model][cuda_label]
                        torch.cuda.empty_cache()
                        self.queues[endpoint_model][cuda_label] = asyncio.Queue(64)
                        batch_size = await self.max_batch_size(endpoint_model, cuda_label)
                        self.batch_sizes[endpoint_model][cuda_label] = batch_size
                        # consumer_tasks[(model, "cuda:" + str(gpu))] = asyncio.create_task(self.chunk_consumer(batch_size, model, "cuda:" + str(gpu)))
            if local > 0 and cpus > 0:
                all_test_types = [ type(openvino_test), type(llama_cpp_test), type(ipex_test)]
                all_tests_ValueError = all(x is ValueError for x in all_test_types)
                all_tests_none = all(x is None for x in all_test_types)
                if (all_tests_ValueError or all_tests_none) and model_type != "llama_cpp":  
                    self.local_endpoints[model]["cpu"] = AutoModel.from_pretrained(model).to("cpu")
                    self.queues[model]["cpu"] = asyncio.Queue(4)
                    self.endpoint_handler[(model, "cpu")] = ""
                    # consumer_tasks[(model, "cpu")] = asyncio.create_task(self.chunk_consumer( 1, model, "cpu"))
                elif openvino_test and type(openvino_test) != ValueError and model_type != "llama_cpp":
                    ov_count = 0
                    openvino_index = self.local_endpoint_types.index("openvino")
                    openvino_model = self.local_endpoint_models[openvino_index]
                    openvino_label = self.local_endpoint_types[openvino_index]
                    save_model_path = Path("./models/model.xml")
                    ## use openvino to call huggingface transformers
                    if self.hwtest["optimum-openvino"] == True: 
                            self.tokenizer[openvino_model][openvino_label] = AutoTokenizer.from_pretrained(model, use_fast=True)
                            model_type =  str(await self.get_openvino_pipeline_type(model))
                            self.local_endpoints[openvino_model][openvino_label] = pipe = pipeline(model_type, model= await self.get_openvino_model(model, model_type), tokenizer=self.tokenizer[openvino_model][openvino_label])
                            self.endpoint_handler[(openvino_model, openvino_label)] = pipe
                    elif self.hwtest["openvino"] == True:
                            import openvino as ov
                            from ov import Core
                            self.endpoint_handler[(openvino_model, openvino_label)] = pipeline( model=self.local_endpoints[openvino_model][openvino_label], tokenizer=self.tokenizer[openvino_model][openvino_label])
                            if not save_model_path.exists():
                                ov_model = Core().compile_model(self.local_endpoints[openvino_model][openvino_label], "CPU")
                                ov_model.export_model(save_model_path)
                            core = Core()
                            compiled_model = core.compile_model(save_model_path, "CPU")
                            encoded_input = self.tokenizer[openvino_model][openvino_label]("Hello, this one sentence!", return_tensors="pt")
                            scores_ov = compiled_model(encoded_input.data)[0]
                            scores_ov = torch.softmax(torch.tensor(scores_ov[0]), dim=0).detach().numpy()
                            print(scores_ov)
                    else:
                        # self.tokenizer[openvino_model][openvino_label] = OpenVinoTokenizer.from_pretrained(model, use_fast=True)
                        # self.local_endpoints[openvino_model][openvino_label] = OpenVinoModel.from_pretrained(model).to("cpu")
                        # self.queues[openvino_model][openvino_label] = asyncio.Queue(64)
                        # self.tokenizer[openvino_model][openvino_label] = AutoTokenizer.from_pretrained(model, use_fast=True)
                        # self.local_endpoints[openvino_model][openvino_label] = AutoModel.from_pretrained(model).to("cpu")
                        # self.queues[openvino_model][openvino_label] = asyncio.Queue(64)
                        # self.endpoint_handler[(openvino_model, openvino_label)] = ""
                        # batch_size = await self.max_batch_size(openvino_model, openvino_label)
                        # self.batch_sizes[openvino_model][openvino_label] = batch_size
                        # consumer_tasks[(model, "openvino")] = asyncio.create_task(self.chunk_consumer(batch_size, model, "openvino"))
                        pass
                        
                    # if endpoint not in list(self.batch_sizes[model].keys()):
                    #     batch_size = await self.max_batch_size(model, endpoint)
                    #     self.batch_sizes[model][endpoint_name] = batch_size
                    # if self.batch_sizes[model][endpoint_name] > 0:
                    #     self.queues[model][endpoint_name] = asyncio.Queue(64)
                    #     self.endpoint_handler[(model, endpoint_name)] = ""
                    #     consumer_tasks[(model, endpoint_name )] = asyncio.create_task(self.chunk_consumer(batch_size, model, endpoint_name))
                    ov_count = ov_count + 1
                elif llama_cpp_test and type(llama_cpp_test) != ValueError and model_type == "llama_cpp":
                    llama_count = 0
                    for endpoint in local:
                        if "llama_cpp" in endpoint:
                            endpoint_name = "llama:"+str(ov_count)
                            batch_size = 0                            
                            if model not in self.batch_sizes:
                                self.batch_sizes[model] = {}
                            if model not in self.queues:
                                self.queues[model] = {}
                            if endpoint not in list(self.batch_sizes[model].keys()):
                                batch_size = await self.max_batch_size(model, endpoint)
                                self.batch_sizes[model][endpoint] = batch_size
                            if self.batch_sizes[model][endpoint] > 0:
                                self.queues[model][endpoint] = asyncio.Queue(64)
                                self.endpoint_handler[(model, endpoint_name)] = ""
                                # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(batch_size, model, endpoint_name))
                            llama_count = llama_count + 1
                elif ipex_test and type(ipex_test) != ValueError:
                    ipex_count = 0
                    for endpoint in local:
                        if "ipex" in endpoint:
                            endpoint_name = "ipex:"+str(ipex_count)
                            batch_size = 0
                            if model not in self.batch_sizes:
                                self.batch_sizes[model] = {}
                            if model not in self.queues:
                                self.queues[model] = {}
                            if endpoint not in list(self.batch_sizes[model].keys()):
                                batch_size = await self.max_batch_size(model, endpoint)
                                self.batch_sizes[model][endpoint] = batch_size
                            if self.batch_sizes[model][endpoint] > 0:
                                self.queues[model][endpoint] = asyncio.Queue(64)
                                self.endpoint_handler[(model, endpoint_name)] = ""
                                # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
                            ipex_count = ipex_count + 1
        metadata = {"local_endpoints": self.local_endpoints, "local_endpoint_types": self.local_endpoint_types, "local_endpoint_models": self.local_endpoint_models, "tokenizer": self.tokenizer, "queues": self.queues, "batch_sizes": self.batch_sizes, "endpoint_handler": self.endpoint_handler}
        return metadata    
    
    async def max_batch_size(self, model, endpoint):
        
        return None
    
    def __test__(self):
        return self 
    
export = worker_py
    
if __name__ == '__main__':
    # run(skillset=os.path.join(os.path.dirname(__file__), 'skillset'))
    resources = {}
    metadata = {}
    try:
        this_worker = worker_py(resources, metadata)
        this_test = this_worker.__test__(resources, metadata)
    except Exception as e:
        print(e)
        pass
    
    
