import os
import sys
# import subprocess
import requests
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from PIL import Image
import tempfile
from install_depends import install_depends_py
import asyncio
import platform
from io import BytesIO
import os
#from PIL import Image
import requests
from pathlib import Path
import json
import hashlib

try:
    from ipfs_multiformats import ipfs_multiformats_py
except:
    from .ipfs_multiformats import ipfs_multiformats_py
    
sys.path.append(os.path.join(os.path.dirname(__file__)))
skillset_folder_files = os.listdir(os.path.join(os.path.dirname(__file__), 'skillset'))
filter_skillset_folder_files = [ x for x in skillset_folder_files if x.endswith(".py") and "hf_" in x]
for file in filter_skillset_folder_files:
    if file.endswith(".py"):
        file_name = file.split(".")[0]
        this_file = os.path.join(os.path.dirname(__file__), 'skillset', file)
        absolute_path = os.path.abspath(this_file)
        if file_name not in globals():
            try:
                with open(absolute_path, encoding='utf-8') as f:
                    exec(f.read())
            except Exception as e:
                print(e)
                pass
            with open(absolute_path, encoding='utf-8') as f:
                globals()[file_name] = exec(f.read())
        else:
            pass

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
    def __init__(self, resources=None, metadata=None):
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
        self.get_model_type = self.get_model_type

        self.hardware_backends = ["llama_cpp", "qualcomm", "apple", "cpu", "gpu", "openvino", "optimum", "optimum_intel", "optimum_openvino", "optimum_ipex", "optimum_neural_compressor", "webnn"]
        # self.hwtest = self.test_ipfs_accelerate
        # if "install_depends" not in globals():
        #     self.install_depends = install_depends_py(resources, metadata)
        
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
        
        # if "default" not in globals() and "default" not in list(self.resources.keys()):
        #     self.default = default
        # elif "default" in list(self.resources.keys()):
        #     self.default = self.resources["default"]
        # elif "default" in globals():
        #     self.default = default
        
        # self.create_cuda_default_endpoint_handler = self.default.create_cuda_default_endpoint_handler
        # self.create_openvino_default_endpoint_handler = self.default.create_openvino_default_endpoint_handler
        # self.create_cpu_default_endpoint_handler = self.default.create_cpu_default_endpoint_handler
        
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
    
    def init(self):
        if "transformers" not in globals() and "transformers" not in list(self.resources.keys()):
            import transformers
            self.transformers = transformers
            self.resources["transformers"] = self.transformers   
        elif "transformers" in list(self.resources.keys()):
            self.transformers = self.resources["transformers"]
        elif "transformers" in globals():
            import transformers
            self.transformers = transformers
            self.resources["transformers"] = self.transformers
            
        self.resources["transformers"] = self.transformers
            
        if "torch" not in globals() and "torch" not in list(self.resources.keys()):
            import torch
            self.torch = torch
            self.resources["torch"] = self.torch
        elif "torch" in list(self.resources.keys()):
            self.torch = self.resources["torch"]
        elif "torch" in globals():
            import torch
            self.torch = torch
            self.resources["torch"] = self.torch
                    
        if "np" not in globals() and "np" not in list(self.resources.keys()):
            import numpy as np
            self.np = np
            self.resources["np"] = self.np
        elif "np" in list(self.resources.keys()):
            self.np = self.resources["np"]
        elif "np" in globals():
            import numpy as np 
            self.np = np
            self.resources["np"] = self.np
        
        import importlib.util
        classes_to_load = [ self.get_model_type(x) for x in self.metadata["models"]]
        files_in_skills_folder = os.listdir(os.path.join(os.path.dirname(__file__), 'skillset'))
        filter_files_for_hf_prefix = [x for x in files_in_skills_folder if x.startswith("hf_")]
        filer_files_remove_suffix = [x.split(".")[0] for x in filter_files_for_hf_prefix]
        filter_files_remove_prefix = [x.replace("hf_", "") for x in filer_files_remove_suffix]
        filter_files_classes_to_load = [x for x in filter_files_remove_prefix if x in classes_to_load]   
        for class_name in filter_files_classes_to_load:
            file = "hf_" + class_name + ".py"
            file_name = "hf_" + class_name
            if file_name not in sys.modules and file_name not in list(self.resources.keys()):
                this_file = os.path.join(os.path.dirname(__file__), 'skillset', file)
                this_file = os.path.abspath(this_file)
                try:
                    spec = importlib.util.spec_from_file_location(file_name, this_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    this_class = getattr(module, file_name)
                    self.resources[file_name] = this_class(self.resources, self.metadata)
                    self.__dict__[file_name] =  self.resources[file_name]
                except Exception as e:
                    print(e)
                    pass
            elif file_name in list(self.resources.keys()):
                self.__dict__[file_name] = self.resources[file_name]
            elif file_name in sys.modules:
                module = sys.modules[file_name]
                this_class = getattr(module, file_name)
                self.resources[file_name] = this_class(self.resources, self.metadata)
            else:
                pass
            
        return None
    
    def init_cuda(self):
        # self.init()
        return None
    
    def init_qualcomm(self):
        # self.init()
        return None
    
    def init_qualcomm(self):
        # self.init()
        return None
    
    def init_cpu(self):
        # self.init()
        return None
    
    def init_networking(self, model, device, networking_label):
        if "ipfs_transformers_py" not in globals() and "ipfs_transformers_py" not in list(self.resources.keys()):
            from ipfs_transformers_py import ipfs_transformers
            self.resources["ipfs_transformers"] = ipfs_transformers
            self.ipfs_transformers = { 
                # "AutoDownloadModel" : ipfs_transformers.AutoDownloadModel()
            }        
        elif "ipfs_transformers" in list(self.resources.keys()):
            self.ipfs_transformers = self.resources["ipfs_transformers"]
        elif "ipfs_transformers" in globals():
            from ipfs_transformers_py import ipfs_transformers
            self.resources["ipfs_transformers"] = ipfs_transformers
            self.ipfs_transformers = { 
                # "AutoDownloadModel" : ipfs_transformers.AutoDownloadModel()
            }
        return None
    
    def init_openvino(self):
        self.init()
        import optimum
        self.resources["optimum"] = optimum
        self.optimum = self.resources["optimum"]
        import openvino as ov
        self.resources["ov"] = ov
        self.ov = self.resources["ov"]
        try:
            from openvino_utils import openvino_utils
        except:
            from .openvino_utils import openvino_utils
        self.resources["openvino_utils"] = openvino_utils
        self.openvino_utils = openvino_utils(self.resources, self.metadata)
        self.resources["openvino_utils"] = self.openvino_utils
        
        return None
    
    async def dispatch_result(self, result):

        return None
    
    def get_model_type(self, model_name=None, model_type=None):
        if model_name is not None:
            if os.path.exists(model_name):
                config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                model_type = config.__class__.model_type
            else:
                config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                model_type = config.__class__.model_type
        return model_type
    
    # def get_openvino_genai_pipeline(self, model_name, model_type=None):
    #     return self.openvino_utils.get_openvino_genai_pipeline(model_name, model_type)

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
            test_results = {"cuda": True, "openvino" : True, "llama_cpp": False, "ipex": False}
            if os.path.exists(test_results_file):
                try:
                    with open(test_results_file, "r") as f:
                        test_results = json.load(f)
                        test_results = {"cuda": True, "openvino" : True, "llama_cpp": False, "ipex": False}
                        return test_results
                except Exception as e:
                    try:
                        test_results = {"cuda": True, "openvino" : True, "llama_cpp": False, "ipex": False}
                        # test_results = await self.install_depends.test_hardware()
                        with open(test_results_file, "w") as f:
                            json.dump(test_results, f)
                        return test_results
                    except Exception as e:
                        print(e)
                        return e
            else:
                try:
                    test_results = {"cuda": True, "openvino" : True, "llama_cpp": False, "ipex": False}
                    # test_results = await self.install_depends.test_hardware()
                    with open(test_results_file, "w") as f:
                        json.dump(test_results, f)
                    return test_results
                except Exception as e:
                    print(e)
                    return e
        else: 
            raise ValueError("install_depends.py not found")
        return test_results
    
    async def get_huggingface_model_types(self):
        if "transformers" not in dir(self):
            if "transformers" not in list(self.resources.keys()):
                import transformers
                self.resources["transformers"] = transformers
                self.transformers = self.resources["transformers"]
            else:
                self.transformers = self.resources["transformers"]

        # Get all model types from the MODEL_MAPPING
        model_types = []
        for config in self.transformers.MODEL_MAPPING.keys():
            if hasattr(config, 'model_type'):
                model_types.append(config.model_type)

        # Add model types from the AutoModel registry
        model_types.extend(list(self.transformers.MODEL_MAPPING._model_mapping.keys()))
        
        # Remove duplicates and sort
        model_types = sorted(list(set(model_types)))
        return model_types
    
    async def generate_hf_skill(model_type):
        ## search hf documentation for the model type
        ## get the data from the class in huggingface transformers
        ## optional ** get examples using k nearest neighbors
        ## run LM inference on the examples
        ## generate the skill
        ## write to disk
        return None
    
    async def init_worker(self, models, local_endpoints, hwtest):
        if "torch" not in dir(self):
            if "torch" not in list(self.resources.keys()):
                import torch
                self.resources["torch"] = torch
                self.torch = self.resources["torch"]
            else:
                self.torch = self.resources["torch"]
        
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
        
        if "cuda" in list(self.hwtest.keys()) and self.hwtest["cuda"] is True:
            try:
                self.init_cuda()
            except Exception as e:
                print(e)
                pass
        if "openvino" in list(self.hwtest.keys()) and self.hwtest["openvino"] is True:
            try:
                self.init_openvino()
            except Exception as e:
                print(e)
                pass
        if "qualcomm" in list(self.hwtest.keys()) and self.hwtest["qualcomm"] is True:
            try:
                self.init_qualcomm()
            except Exception as e:
                print(e)
                pass
        if "apple" in list(self.hwtest.keys()) and self.hwtest["apple"] is True:
            try:
                self.init_apple()
            except Exception as e:
                print(e)
                pass
        if "webnn" in list(self.hwtest.keys()) and self.hwtest["webnn"] is True:
            try:
                self.init_webnn()
            except Exception as e:
                print(e)
                pass        
        cuda_test = self.hwtest["cuda"]
        openvino_test = self.hwtest["openvino"]
        llama_cpp_test = self.hwtest["llama_cpp"]
        ipex_test = self.hwtest["ipex"]
        cuda = cuda_test
        cpus = os.cpu_count()
        torch_gpus = 0
        if cuda == True and self.torch.cuda.is_available():
            torch_gpus = self.torch.cuda.device_count()
        else:    
            torch_gpus = 0
        if openvino_test == True:
            from openvino import Core
            openvino_gpus = 1 if "GPU" in Core().available_devices else 0
            del Core
        gpus = torch_gpus if cuda == True else openvino_gpus if openvino_test == True else 0        
        # cpus = os.cpu_count()
        # cuda = torch.cuda.is_available()
        # gpus = torch.cuda.device_count()
        huggingface_model_types = await self.get_huggingface_model_types()
        for model in models:
            model_type = self.get_model_type(model)
            if model_type not in huggingface_model_types:
                print("model type not found in huggingface model types")
                continue 
            if "hf_" + model_type not in list(dir(self)):
                print("model type not found in worker skills but it is a huggingface model type")
                print("generate the skill for this model type")
                generated = None
                while generated is not None and generated is not True:
                    generated = True
                    # generated = self.generate_hf_skill(model_type)
                continue
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
            optimum_model_types = ["clip", "wav2vec", "wav2vec2", "bert", "t5", "xclip", "llava", "llava_next", "qwen2", "llama", "clap", "whisper" ]
            openvino_model_types = ["clip", "wav2vec", "wav2vec2", "bert", "t5", "xclip", "llava", "llava_next", "qwen2", "llama", "clap", "whisper" ]
            openvino_genai_model_types = ["llava", "llava_next"]
            custom_types = []
            method_name = "hf_" + model_type
            this_method = None
            try:
                this_method = getattr(self, method_name, None)
            except Exception as e:
                print(e)
                pass
            if model_type != "llama_cpp" and model_type not in custom_types:
                if cuda and torch_gpus > 0:
                    if cuda_test and type(cuda_test) != ValueError:
                        for gpu in range(torch_gpus):
                            device = 'cuda:' + str(gpu)
                            cuda_label = device
                            self.local_endpoints[model][cuda_label], self.tokenizer[model][cuda_label], self.endpoint_handler[model][cuda_label], self.queues[model][cuda_label], self.batch_sizes[model][cuda_label] = this_method.init_cuda(
                                model,
                                device,
                                cuda_label
                            )
                if local > 0 and cpus > 0:
                    if model_type in openvino_genai_model_types or model_type in openvino_model_types or model_type in optimum_model_types:
                        if openvino_test and type(openvino_test) != ValueError and model_type != "llama_cpp":
                            openvino_local_endpont_types = [ x for x in local_endpoint_types if "openvino" in x]
                            for openvino_endpoint in openvino_local_endpont_types:
                                ov_count = openvino_endpoint.split(":")[1]
                                openvino_label = "openvino:" + str(ov_count)
                                device = "openvino:" + str(ov_count)
                                if model_type in openvino_genai_model_types:
                                    self.local_endpoints[model][openvino_label], self.tokenizer[model][openvino_label], self.endpoint_handler[model][openvino_label], self.queues[model][openvino_label], self.batch_sizes[model][openvino_label] = this_method.init_openvino(
                                        model,
                                        model_type,
                                        device,
                                        openvino_label,
                                        self.openvino_utils.get_openvino_genai_pipeline,
                                        self.openvino_utils.get_optimum_openvino_model,
                                        self.openvino_utils.get_openvino_model,
                                        self.openvino_utils.get_openvino_pipeline_type,
                                        self.openvino_utils.openvino_cli_convert,
                                    )
                                elif model_type in openvino_model_types:
                                    self.local_endpoints[model][openvino_label], self.tokenizer[model][openvino_label], self.endpoint_handler[model][openvino_label], self.queues[model][openvino_label], self.batch_sizes[model][openvino_label] = this_method.init_openvino(
                                        model,
                                        model_type,
                                        device,
                                        openvino_label,
                                        self.openvino_utils.get_optimum_openvino_model,
                                        self.openvino_utils.get_openvino_model,
                                        self.openvino_utils.get_openvino_pipeline_type,
                                        self.openvino_utils.openvino_cli_convert,
                                    )
                                elif model_type in optimum_model_types:
                                    self.local_endpoints[model][openvino_label], self.tokenizer[model][openvino_label], self.endpoint_handler[model][openvino_label], self.queues[model][openvino_label], self.batch_sizes[model][openvino_label] = this_method.init_openvino(
                                        model,
                                        model_type,
                                        device,
                                        openvino_label,
                                        self.openvino_utils.get_optimum_openvino_model,
                                        self.openvino_utils.get_openvino_model,
                                        self.openvino_utils.get_openvino_pipeline_type,
                                        self.openvino_utils.openvino_cli_convert,
                                    )
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
    
    
