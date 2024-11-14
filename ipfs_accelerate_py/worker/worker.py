import os
import sys
import subprocess
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_ipfs_accelerate import test_ipfs_accelerate
from install_depends import install_depends_py
import optimum
import torch 
import asyncio
import transformers
try:
    from ipfs_multiformats import ipfs_multiformats_py
except:
    from .ipfs_multiformats import ipfs_multiformats_py
from transformers import AutoTokenizer, AutoModel, AutoConfig, pipeline
import ipfs_transformers_py
from pathlib import Path
import numpy as np
import torch
import json
import hashlib

class should_abort:
    def __init__(self, resources, metadata):
        self.abort = False
        return None
    
    async def should_abort(self):
        return self.abort

class TaskAbortion:
    def __init__(self, resources, metadata):
        self.abort = False
        return None
    
    async def TaskAbortion(self):
        self.abort = True
        return self.abort

class dispatch_result:
    def __init__(self, resources, metadata):        
        self.inbox = {}
        self.outbox = {}
        self.queue = {}
        return None
    
    async def dispatch_result(self, result):
        return None    
    
class worker_py:
    def __init__(self, metadata, resources):
        self.metadata = metadata
        self.resources = resources
        self.inbox = {}
        self.outbox = {}
        self.tokenizer = {}
        self.queues = {}
        self.batch_sizes = {}
        self.endpoint_handler = {}
        self.endpoints = {}
        self.endpoint_types = ["local_endpoints"]
        self.local_endpoints = {}
        self.dispatch_result = self.dispatch_result
        self.hardware_backends = ["llama_cpp", "cpu", "gpu", "openvino", "optimum", "optimum_intel", "optimum_openvino", "optimum_ipex", "optimum_neural_compressor"]
        if "test_ipfs_accelerate" not in globals():
            self.test_ipfs_accelerate = test_ipfs_accelerate(resources, metadata)
            # self.hwtest = self.test_ipfs_accelerate
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
        if "ipfs_multiformats_py" not in globals() and "ipfs_multiformats_py" not in list(self.resources.keys()):
            ipfs_multiformats = ipfs_multiformats_py(resources, metadata)
            self.ipfs_multiformats = ipfs_multiformats
        elif "ipfs_multiformats_py" in list(self.resources.keys()):
            self.ipfs_multiformats = self.resources["ipfs_multiformats_py"]
        elif "ipfs_multiformats_py" in globals():
            ipfs_multiformats = ipfs_multiformats_py(resources, metadata)
            self.ipfs_multiformats = ipfs_multiformats

        if "dispatch_result" not in globals() and "dispatch_result" not in list(self.resources.keys()):
            self.dispatch_result = dispatch_result
        elif "dispatch_result" in list(self.resources.keys()):
            self.dispatch_result = self.resources["dispatch_result"]
        else:
            self.dispatch_result = dispatch_result
            pass

        for endpoint in self.endpoint_types:
            if endpoint not in dir(self):
                self.__dict__[endpoint] = {}        
            for backend in self.hardware_backends:
                if backend not in list(self.__dict__[endpoint].keys()):
                    self.__dict__[endpoint][backend] = {}
        return None
    
    async def dispatch_result(self, result):

        return None
    
    
    async def test_hardware(self):
        install_file_hash = None
        test_results_file = None
        install_depends_filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "install_depends", "install_depends.py")
        if os.path.exists(install_depends_filename):
            ## get the sha256 hash of the file
            sha256 = hashlib.sha256()
            with open(install_depends_filename, "rb") as f:
                for byte_block in iter(lambda: f.read(4096),b""):
                    sha256.update(byte_block)
            install_file_hash = sha256.hexdigest()
            test_results_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"test", install_file_hash + ".json")
            if os.path.exists(test_results_file):
                try:
                    with open(test_results_file, "r") as f:
                        test_results = json.load(f)
                        return test_results
                except Exception as e:
                    try:
                        test_results = await self.install_depends.test_hardware()
                        with open(test_results_file, "w") as f:
                            json.dump(test_results, f)
                        return test_results
                    except Exception as e:
                        print(e)
                        return e
            else:
                try:
                    test_results = await self.install_depends.test_hardware()
                    with open(test_results_file, "w") as f:
                        json.dump(test_results, f)
                    return test_results
                except Exception as e:
                    print(e)
                    return e
        else: 
            raise ValueError("install_depends.py not found")
        return test_results
    
    async def get_model_type(self, model_name, model_type=None):
        if model_type is None:
            config = AutoConfig.from_pretrained(model_name)
            model_type = config.__class__.model_type
        return model_type
    
    async def get_model_format(self, model_name, model_type=None):
        if model_type is None:
            config = AutoConfig.from_pretrained(model_name)
            model_type = config.__class__.model_type
        return model_type
    
    async def get_openvino_model(self, model_name, model_type=None):
        import openvino as ov                                
        core = ov.Core()
        homedir = os.path.expanduser("~")
        model_name_convert = model_name.replace("/", "--")
        huggingface_cache = os.path.join(homedir, ".cache/huggingface")
        huggingface_cache_models = os.path.join(huggingface_cache, "hub")
        huggingface_cache_models_files = os.listdir(huggingface_cache_models)
        huggingface_cache_models_files_dirs = [os.path.join(huggingface_cache_models, file) for file in huggingface_cache_models_files if os.path.isdir(os.path.join(huggingface_cache_models, file))]
        huggingface_cache_models_files_dirs_models = [ x for x in huggingface_cache_models_files_dirs if "model" in x ]
        huggingface_cache_models_files_dirs_models_model_name = [ x for x in huggingface_cache_models_files_dirs_models if model_name_convert in x ]
        model_src_path = os.path.join(huggingface_cache_models, huggingface_cache_models_files_dirs_models_model_name[0])
        model_dst_path = os.path.join(model_src_path, "model.xml")
        if model_type is None:
            config = AutoConfig.from_pretrained(model_src_path)
            model_type = config.__class__.model_type
        hftokenizer = AutoTokenizer.from_pretrained(model_name)
        hfmodel = AutoModel.from_pretrained(model_name)
        text = "Replace me by any text you'd like."
        encoded_input = hftokenizer(text, return_tensors='pt')
        import openvino as ov
        ov_model = ov.convert_model(hfmodel, example_input={**encoded_input})
        ov.save_model(ov_model, model_dst_path)
        return ov.compile_model(ov_model)
    
    async def get_optimum_openvino_model(self, model_name, model_type=None):
        if model_type is None:
            config = AutoConfig.from_pretrained(model_name)
            model_type = config.__class__.model_type
        model_mapping_list = ["text-classification", "token-classification", "question-answering", "audio-classification", "image-classification", "feature-extraction", "fill-mask", "text-generation-with-past", "text2text-generation-with-past", "automatic-speech-recognition", "image-to-text", "bert"]
        if model_type not in model_mapping_list:
            return None
        if model_type == "bert":
            model_type = "feature-extraction"
            from optimum.intel import OVModelForFeatureExtraction
            results = OVModelForFeatureExtraction.from_pretrained(model_name, compile=False)
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
                model_type = "feature-extraction"
            else:
                return None
        if model_type == None:
            return None
        return model_type
    
    def create_endpoint_handler(self, endpoint_model, cuda_label):
        def handler(x):
            self.local_endpoints[endpoint_model][cuda_label].eval()
            tokens = self.tokenizer[endpoint_model][cuda_label](x, return_tensors='pt').to(self.local_endpoints[endpoint_model][cuda_label].device)
            results = self.local_endpoints[endpoint_model][cuda_label](**tokens)
            return results
        return handler
    
    def create_openvino_endpoint_handler(self, endpoint_model, openvino_label):
        def handler(x):
            return self.local_endpoints[endpoint_model][openvino_label](x)
        return handler

    async def init_worker(self, models, local_endpoints, hwtest):
        if local_endpoints is None or len(local_endpoints) == 0:
            if "local_endpoints" in list(self.__dict__.keys()):
                local_endpoints = self.local_endpoints
            elif "local_endpoints" in list(self.resources.keys()):
                local_endpoints = self.resources["local_endpoints"]
            else:
                local_endpoints = {}
        else:
            pass
        self.local_endpoints  = local_endpoints
        # local_endpoint_types = [ x[1] for x in local_endpoints if x[0] in models]
        # local_endpoint_models = [ x[0] for x in local_endpoints if x[0] in models]
        local_endpoint_models = list(local_endpoints.keys())
        local_endpoint_types = []
        for model in local_endpoint_models:
            for endpoints in local_endpoints[model]:
                endpoint_type = endpoints[1]
                local_endpoint_types.append(endpoint_type)
        new_endpoints_list = {}
        for model in list(local_endpoints.keys()):
            if model not in list(new_endpoints_list.keys()):
                new_endpoints_list[model] = {}
            for endpoint in local_endpoints[model]:
                endpoint_type = endpoint[1]
                if endpoint_type not in list(new_endpoints_list[model].keys()):
                    new_endpoints_list[model][endpoint_type] = endpoint[2]
        
        print(new_endpoints_list)
        self.local_endpoints = new_endpoints_list

        # local_endpoints = { model: { endpoint[1]: endpoint[2] for endpoint in local_endpoints if endpoint[0] == model} for model in models}
        self.local_endpoint_types = local_endpoint_types
        self.local_endpoint_models = local_endpoint_models
        local = len(local_endpoints) > 0 if isinstance(self.local_endpoints, dict) and len(list(self.local_endpoints.keys())) > 0 else False
        if hwtest is None:
            if "hwtest" in list(self.__dict__.keys()):
                hwtest = await self.test_hardware()
            elif "hwtest" in list(self.resources.keys()):
                hwtest = self.resources["hwtest"]
            else:
                hwtest = await self.test_hardware()
        self.hwtest = hwtest
        self.resources["hwtest"] = hwtest
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
            if model not in self.endpoint_handler:
                self.endpoint_handler[model] = {}
            if model not in self.batch_sizes:
                self.batch_sizes[model] = {}
            if model_type != "llama_cpp":
                if cuda and gpus > 0:
                    if cuda_test and type(cuda_test) != ValueError:
                        for gpu in range(gpus):
                            device = 'cuda:' + str(gpu)
                            cuda_index = self.local_endpoint_types.index(device)
                            if "model" in list(self.local_endpoints.keys()):
                                endpoint_model = model
                            else:
                                for this_model in list(self.local_endpoints.keys()):
                                    if device in list(self.local_endpoints[this_model].keys()):
                                        endpoint_model = this_model
                            cuda_label = self.local_endpoint_types[cuda_index]
                            self.tokenizer[endpoint_model][cuda_label] = AutoTokenizer.from_pretrained(model, device=device, use_fast=True)
                            self.local_endpoints[endpoint_model][cuda_label] = AutoModel.from_pretrained(model).to(device)
                            self.endpoint_handler[endpoint_model][cuda_label] = self.create_endpoint_handler(endpoint_model, cuda_label)
                            torch.cuda.empty_cache()
                            self.queues[endpoint_model][cuda_label] = asyncio.Queue(64)
                            # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
                            self.batch_sizes[endpoint_model][cuda_label] = 0
                if local > 0 and cpus > 0:
                    all_test_types = [ type(openvino_test), type(llama_cpp_test), type(ipex_test)]
                    all_tests_ValueError = all(x is ValueError for x in all_test_types)
                    all_tests_none = all(x is None for x in all_test_types)
                    if (all_tests_ValueError or all_tests_none) and model_type != "llama_cpp":  
                        self.local_endpoints[model]["cpu"] = AutoModel.from_pretrained(model).to("cpu")
                        self.queues[model]["cpu"] = asyncio.Queue(4)
                        self.endpoint_handler[model]["cpu"] = ""
                    elif openvino_test and type(openvino_test) != ValueError and model_type != "llama_cpp":
                        ov_count = 0
                        openvino_endpoints = []
                        device = "openvino:" + str(ov_count)
                        openvino_index = self.local_endpoint_types.index(device)
                        if "model" in list(self.local_endpoints.keys()):
                                openvino_model = model
                        else:
                            for this_model in list(self.local_endpoints.keys()):
                                if device in list(self.local_endpoints[this_model].keys()):
                                    openvino_model = this_model
                        for item in self.local_endpoint_types:
                            if device in item:
                                openvino_endpoints.append(item)
                                
                        openvino_label = self.local_endpoint_types[openvino_index]
                        # to disable openvino to calling huggingface transformers uncomment
                        # self.hwtest["optimum-openvino"] = False
                        # if self.hwtest["optimum-openvino"] == True: 
                        try:
                            self.tokenizer[openvino_model][openvino_label] = AutoTokenizer.from_pretrained(model, use_fast=True)
                            model_type =  str(await self.get_openvino_pipeline_type(model))
                            self.local_endpoints[openvino_model][openvino_label] = pipe = pipeline(model_type, model= await self.get_optimum_openvino_model(model, model_type), tokenizer=self.tokenizer[openvino_model][openvino_label])
                            self.endpoint_handler[openvino_model][openvino_label] = self.create_openvino_endpoint_handler(openvino_model, openvino_label)
                            self.batch_sizes[openvino_model][openvino_label] = 0
                        # elif self.hwtest["openvino"] == True:                            
                        except Exception as e:
                            try:
                                self.tokenizer[openvino_model][openvino_label] =  AutoTokenizer.from_pretrained(model, use_fast=True)
                                self.local_endpoints[openvino_model][openvino_label] = await self.get_openvino_model(model, model_type)
                                self.endpoint_handler[openvino_model][openvino_label] = lambda x: self.local_endpoints[openvino_model][openvino_label]({**self.tokenizer[openvino_model][openvino_label](x, return_tensors='pt')})
                                self.batch_sizes[openvino_model][openvino_label] = 0
                            except Exception as e:
                                print(e)
                                pass
                        ov_count = ov_count + 1
                else:
                    pass
        worker_endpoint_types = []
        worker_model_types = []
        for endpoint in self.endpoint_handler:
            worker_model_types.append(endpoint)
        worker_endpoint_types = []
        for this_model in worker_model_types:
            worker_endpoint_types = worker_endpoint_types + list(self.endpoint_handler[this_model].keys())
        local_endpoint_keys = list(self.local_endpoints.keys())
        for model in local_endpoint_keys:
            if model not in worker_model_types:
                del self.local_endpoints[model]
            else:
                model_endpoint_types = set(list(self.local_endpoints[model].keys()))
                for endpoint_type in model_endpoint_types:
                    if endpoint_type not in worker_endpoint_types:
                        del self.local_endpoints[model][endpoint_type]
                    else:
                        pass
                pass

        resources = {"local_endpoints": self.local_endpoints, "tokenizer": self.tokenizer, "queues": self.queues, "batch_sizes": self.batch_sizes, "endpoint_handler": self.endpoint_handler , "local_endpoint_types": list(worker_endpoint_types), "local_endpoint_models": list(worker_model_types), "hwtest": self.hwtest}
        return resources    
    
    # def __test__(self):
    #     return self 
    
export = worker_py
    
# if __name__ == '__main__':
#     # run(skillset=os.path.join(os.path.dirname(__file__), 'skillset'))
#     resources = {}
#     metadata = {}
#     try:
#         this_worker = worker_py(resources, metadata)
#         this_test = this_worker.__test__(resources, metadata)
#     except Exception as e:
#         print(e)
#         pass
    
    
