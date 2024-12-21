import os
import sys
import subprocess
import requests
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_ipfs_accelerate import test_ipfs_accelerate
from PIL import Image
import tempfile
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from install_depends import install_depends_py
import transformers
from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoConfig, TextStreamer, AutoModelForImageTextToText
import optimum
import torch 
import asyncio
import openvino as ov
from optimum.intel.openvino import OVModelForVisualCausalLM
import platform
from io import BytesIO
import os
from PIL import Image
import requests

try:
    from ipfs_multiformats import ipfs_multiformats_py
except:
    from .ipfs_multiformats import ipfs_multiformats_py
    
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'skillset'))
from ipfs_accelerate_py.worker.skillset import hf_llava
from ipfs_accelerate_py.worker.skillset import default
from ipfs_accelerate_py.worker.skillset import hf_embed
from ipfs_accelerate_py.worker.skillset import hf_lm

try:
    from openvino_utils import openvino_utils
except:
    from .openvino_utils import openvino_utils

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
        # self.hwtest = self.test_ipfs_accelerate
        # if "install_depends" not in globals():
        #     self.install_depends = install_depends_py(resources, metadata)
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
            
        if "openvino_utils" not in globals() and "openvino_utils" not in list(self.resources.keys()):
            self.openvino_utils = openvino_utils(resources, metadata)
        elif "openvino_utils" in list(self.resources.keys()):
            self.openvino_utils = self.resources["openvino_utils"]
        elif "openvino_utils" in globals():
            self.openvino_utils = openvino_utils(resources, metadata)

        if "dispatch_result" not in globals() and "dispatch_result" not in list(self.resources.keys()):
            self.dispatch_result = dispatch_result
        elif "dispatch_result" in list(self.resources.keys()):
            self.dispatch_result = self.resources["dispatch_result"]
        else:
            self.dispatch_result = dispatch_result
            pass
        
        if "hf_llava" not in globals() and "hf_llava" not in list(self.resources.keys()):
            self.hf_llava = hf_llava
        elif "hf_llava" in list(self.resources.keys()):
            self.hf_llava = self.resources["hf_llava"]
        elif "hf_llava" in globals():
            self.hf_llava = hf_llava
        
        if "hf_embed" not in globals() and "default" not in list(self.resources.keys()):
            self.hf_embed = hf_embed
        elif "hf_embed" in list(self.resources.keys()):
            self.hf_embed = self.resources["hf_embed"]
        elif "hf_embed" in globals():
            self.hf_embed = hf_embed

        if "default" not in globals() and "default" not in list(self.resources.keys()):
            self.default = default
        elif "default" in list(self.resources.keys()):
            self.default = self.resources["default"]
        elif "default" in globals():
            self.default = default

            
        if "hf_lm" not in globals() and "hf_lm" not in list(self.resources.keys()):
            self.hf_lm = hf_lm
        elif "hf_lm" in list(self.resources.keys()):
            self.hf_lm = self.resources["hf_lm"]
        elif "hf_lm" in globals():
            self.hf_lm = hf_lm
        
        self.create_openvino_vlm_endpoint_handler = self.hf_llava.create_openvino_vlm_endpoint_handler
        self.create_vlm_endpoint_handler = self.hf_llava.create_vlm_endpoint_handler
        self.create_openvino_endpoint_handler = self.default.create_openvino_endpoint_handler
        self.create_endpoint_handler = self.default.create_endpoint_handler
        self.get_openvino_model = self.openvino_utils.get_openvino_model
        self.get_optimum_openvino_model = self.openvino_utils.get_optimum_openvino_model
        self.get_openvino_pipeline_type = self.openvino_utils.get_openvino_pipeline_type
        self.get_model_type = self.openvino_utils.get_model_type
        
        
        # if "hf_embed" not in globals() and "hf_embed" not in list(self.resources.keys()):
        #     self.hf_embed = hf_embed
        # elif "hf_embed" in list(self.resources.keys()):
        #     self.hf_embed = self.resources["hf_embed"]
        # elif "hf_embed" in globals():
        #     self.hf_embed = hf_embed
        
        # if "should_abort" not in globals() and "should_abort" not in list(self.resources.keys()):
        #     self.should_abort = should_abort(resources, metadata)
        # elif "should_abort" in list(self.resources.keys()):
        #     self.should_abort = self.resources["should_abort"]
        # elif "should_abort" in globals():
        #     self.should_abort = should_abort(resources, metadata)
        
        # if "TaskAbortion" not in globals() and "TaskAbortion" not in list(self.resources.keys()):
        #     self.TaskAbortion = TaskAbortion(resources, metadata)
        # elif "TaskAbortion" in list(self.resources.keys()):
        #     self.TaskAbortion = self.resources["TaskAbortion"]
        # elif "TaskAbortion" in globals():
        #     self.TaskAbortion = TaskAbortion(resources, metadata)

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
            test_results_file = os.path.join(tempfile.gettempdir(), install_file_hash + ".json")
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
    
    def get_model_type(self, model_name, model_type=None):
        if model_type is None:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_type = config.__class__.model_type
        return model_type
                
    def get_openvino_model(self, model_name, model_type=None, device_name=None ):
        return self.openvino_utils.get_openvino_model(model_name, model_type, device_name)
        
    def get_optimum_openvino_model(self, model_name, model_type=None, device_name=None ):
        results = self.openvino_utils.get_optimum_openvino_model(model_name, model_type, device_name)
        return results
    
    def get_openvino_pipeline_type(self, model_name, model_type=None):
        results = self.openvino_utils.get_openvino_pipeline_type(model_name, model_type)
        return results
    
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
            model_type = self.get_model_type(model)
            if model not in list(self.tokenizer.keys()):
                self.tokenizer[model] = {}
            if model not in list(self.local_endpoints.keys()):
                self.local_endpoints[model] = {}
            if model not in list(self.queues.keys()):
                self.queues[model] = {}
            if model not in list(self.endpoint_handler.keys()):
                self.endpoint_handler[model] = {}
            if model not in list(self.batch_sizes.keys()):
                self.batch_sizes[model] = {}
                
            vlm_model_types = ["llava", "llava_next"]
            llm_model_types = ["qwen2", "llama"]
            text_embedding_types = ["bert"]
            custom_types = vlm_model_types + text_embedding_types + llm_model_types
            if model_type != "llama_cpp" and model_type not in custom_types:
                if cuda and gpus > 0:
                    if cuda_test and type(cuda_test) != ValueError:
                        for gpu in range(gpus):
                            device = 'cuda:' + str(gpu)
                            cuda_label = 'cuda:' + str(gpu)
                            self.local_endpoints[model][cuda_label], self.tokenizer[model][cuda_label], self.endpoint_handler[model][cuda_label], self.queues[model][cuda_label], self.batch_sizes[model][cuda_label] = self.default.init_cuda(model, device, cuda_label)
                if local > 0 and cpus > 0:
                    if openvino_test and type(openvino_test) != ValueError and model_type != "llama_cpp":
                        openvino_local_endpont_types = [ x for x in local_endpoint_types if "openvino" in x]
                        for openvino_endpoint in openvino_local_endpont_types:
                            ov_count = openvino_endpoint.split(":")[1]
                            openvino_label = "openvino:" + str(ov_count)
                            device = "openvino:" + str(ov_count)
                            self.local_endpoints[model][openvino_label], self.tokenizer[model][openvino_label], self.endpoint_handler[model][openvino_label], self.queues[model][openvino_label], self.batch_sizes[model][openvino_label] = self.hf_embed.init_openvino(model, model_type, device, openvino_label, self.openvino_utils.get_openvino_model, self.openvino_utils.get_openvino_pipeline_type)
            elif model_type in vlm_model_types:
                if cuda and gpus > 0:
                    if cuda_test and type(cuda_test) != ValueError:
                        for gpu in range(gpus):
                            device = 'cuda:' + str(gpu)
                            cuda_label = device
                            self.local_endpoints[model][cuda_label], self.tokenizer[model][cuda_label], self.endpoint_handler[model][cuda_label], self.queues[model][cuda_label], self.batch_sizes[model][cuda_label] = self.hf_llava.init_cuda( model, device, cuda_label)
                            torch.cuda.empty_cache()
                if local > 0 and cpus > 0:
                    openvino_local_endpont_types = [ x for x in local_endpoint_types if "openvino" in x]
                    for openvino_endpoint in openvino_local_endpont_types:
                        ov_count = openvino_endpoint.split(":")[1]
                        openvino_label = "openvino:" + str(ov_count)
                        device = "openvino:" + str(ov_count)
                        if openvino_test and type(openvino_test) != ValueError and model_type != "llama_cpp":
                            self.local_endpoints[model][openvino_label], self.tokenizer[model][openvino_label], self.endpoint_handler[model][openvino_label], self.queues[model][openvino_label], self.batch_sizes[model][openvino_label] = self.hf_llava.init_openvino(model, model_type, device, openvino_label, self.get_openvino_model, self.get_openvino_pipeline_type)
                            torch.cuda.empty_cache()
            elif model_type in text_embedding_types:
                if cuda and gpus > 0:
                    if cuda_test and type(cuda_test) != ValueError:
                        for gpu in range(gpus):
                            device = 'cuda:' + str(gpu)
                            cuda_label = device
                            self.local_endpoints[model][cuda_label], self.tokenizer[model][cuda_label], self.endpoint_handler[model][cuda_label], self.queues[model][cuda_label], self.batch_sizes[model][cuda_label] = self.hf_embed.init_cuda( model, device, cuda_label)
                            torch.cuda.empty_cache()
                if local > 0 and cpus > 0:
                    if openvino_test and type(openvino_test) != ValueError and model_type != "llama_cpp":
                        openvino_local_endpont_types = [ x for x in local_endpoint_types if "openvino" in x]
                        for openvino_endpoint in openvino_local_endpont_types:
                            ov_count = openvino_endpoint.split(":")[1]
                            openvino_label = "openvino:" + str(ov_count)
                            device = "openvino:" + str(ov_count)
                            self.local_endpoints[model][openvino_label], self.tokenizer[model][openvino_label], self.endpoint_handler[model][openvino_label], self.queues[model][openvino_label], self.batch_sizes[model][openvino_label] = self.hf_embed.init_openvino(model, model_type, device, openvino_label, self.get_openvino_model, self.get_openvino_pipeline_type)
                            torch.cuda.empty_cache()
            elif model_type in llm_model_types:
                if cuda and gpus > 0:
                    if cuda_test and type(cuda_test) != ValueError:
                        for gpu in range(gpus):
                            device = 'cuda:' + str(gpu)
                            cuda_label = device
                            self.local_endpoints[model][cuda_label], self.tokenizer[model][cuda_label], self.endpoint_handler[model][cuda_label], self.queues[model][cuda_label], self.batch_sizes[model][cuda_label] = self.hf_lm.init_cuda( model, device, cuda_label)
                            torch.cuda.empty_cache()
                if local > 0 and cpus > 0:
                    if openvino_test and type(openvino_test) != ValueError and model_type != "llama_cpp":
                        openvino_local_endpont_types = [ x for x in local_endpoint_types if "openvino" in x]
                        for openvino_endpoint in openvino_local_endpont_types:
                            ov_count = openvino_endpoint.split(":")[1]
                            openvino_label = "openvino:" + str(ov_count)
                            device = "openvino:" + str(ov_count)
                            self.local_endpoints[model][openvino_label], self.tokenizer[model][openvino_label], self.endpoint_handler[model][openvino_label], self.queues[model][openvino_label], self.batch_sizes[model][openvino_label] = self.hf_lm.init_openvino(model, model_type, device, openvino_label, self.get_openvino_model, self.get_openvino_pipeline_type)
                            torch.cuda.empty_cache()

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
    
    
