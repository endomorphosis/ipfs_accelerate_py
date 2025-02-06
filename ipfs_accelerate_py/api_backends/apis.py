from .ovms import ovms
from .groq import groq
from .hf_tei import hf_tei
from .hf_tgi import hf_tgi
from .llvm import llvm
from .s3_kit import s3_kit
from .openai_api import openai_api
from .ollama import ollama
from api_models_registry import api_models
import asyncio
import os
import json

class apis:
    def __init__(self, resources, metadata):
        if resources is None:
            resources = {}
        if metadata is None:
            metadata = {}     
        self.resources = resources
        self.metadata = metadata
        self.api_models = api_models(self.resources, self.metadata)
        self.endpoints = self.metadata["endpoints"]
        
        if "hf_tei" not in self.resources:
            self.resources["hf_tei"] = hf_tei(self.resources, self.metadata)
            self.hf_tei = self.resources["hf_tei"]
        else:
            self.hf_tei = self.resources["hf_tei"]
            
        if "hf_tgi" not in self.resources:
            self.resources["hf_tgi"] = hf_tgi(self.resources, self.metadata)
            self.hf_tgi = self.resources["hf_tgi"]
        else:
            self.hf_tgi = self.resources["hf_tgi"]
            
        if "groq" not in self.resources:
            self.resources["groq"] = groq(self.resources, self.metadata)
            self.groq = self.resources["groq"]
        else:
            self.groq = self.resources["groq"]
            
        if "llvm" not in self.resources:
            self.resources["llvm"] = llvm(self.resources, self.metadata)
            self.llvm = self.resources["llvm"]
        else:
            self.llvm = self.resources["llvm"]
            
        if "ollama" not in self.resources:
            self.resources["ollama"] = ollama(self.resources, self.metadata)
            self.ollama = self.resources["ollama"]
        else:
            self.ollama = self.resources["ollama"]
            
        if "openai_api" not in self.resources:
            self.resources["openai_api"] = openai_api(self.resources, self.metadata)
            self.openai_api = self.resources["openai_api"]
        else:
            self.openai_api = self.resources["openai_api"]
        
        if "s3_kit" not in self.resources:
            self.resources["s3_kit"] = s3_kit(self.resources, self.metadata)
            self.s3_kit = self.resources["s3_kit"]
        else:
            self.s3_kit = self.resources["s3_kit"]
        
        self.tgi_endpoints = {}
        self.ollama_endpoints = {}
        self.tei_endpoints = {}
        self.groq_endpoints = {}
        self.llvm_endpoints = {}
        self.openai_endpoints = {}
        self.s3_endpoints = {}
        self.ovms_endpoints = {}
        
        self.request_tgi_endpoint = self.hf_tgi.request_tgi_endpoint
        self.request_tei_endpoint = self.hf_tei.request_tei_endpoint
        self.request_groq_endpoint = self.groq.request_groq_endpoint
        self.request_llvm_endpoint = self.llvm.request_llvm_endpoint
        self.request_ollama_endpoint = self.ollama.request_ollama_endpoint
        self.request_openai_endpoint = self.openai_api.request_openai_endpoint
        self.request_s3_endpoint = self.s3_kit.request_s3_endpoint
        self.request_ovms_endpoint = self.ovms.request_ovms_endpoint
        
        self.test_groq_endpoint = self.groq.test_groq_endpoint
        self.test_llvm_endpoint = self.llvm.test_llvm_endpoint
        self.test_ollama_endpoint = self.ollama.test_ollama_endpoint
        self.test_ovms_endpoint = self.ovms.test_ovms_endpoint
        self.test_tei_endpoint = self.hf_tei.test_tei_endpoint
        self.test_tgi_endpoint = self.hf_tgi.test_tgi_endpoint
        self.test_openai_endpoint = self.openai_api.test_openai_endpoint
        self.test_s3_endpoint = self.s3_kit.test_s3_endpoint

        self.add_groq_endpoint = self.groq.add_groq_endpoint
        self.add_llvm_endpoint = self.llvm.add_llvm_endpoint
        self.add_tei_endpoint = self.hf_tei.add_tei_endpoint
        self.add_tgi_endpoint = self.hf_tgi.add_tgi_endpoint
        self.add_ollama_endpoint = self.ollama.add_ollama_endpoint
        self.add_ovms_endpoint = self.ovms.add_ovms_endpoint
        self.add_libp2p_endpoint = self.ovms.add_libp2p_endpoint
        self.add_openai_endpoint = self.openai_api.add_openai_endpoint
        self.add_s3_endpoint = self.s3_kit.add_s3_endpoint
        
        self.rm_groq_endpoint = self.groq.rm_groq_endpoint
        self.rm_llvm_endpoint = self.llvm.rm_llvm_endpoint
        self.rm_tei_endpoint = self.hf_tei.rm_tei_endpoint
        self.rm_tgi_endpoint = self.hf_tgi.rm_tgi_endpoint
        self.rm_ollama_endpoint = self.ollama.rm_ollama_endpoint
        self.rm_ovms_endpoint = self.ovms.rm_ovms_endpoint
        self.rm_openai_endpoint = self.openai_api.rm_openai_endpoint
        self.rm_s3_endpoint = self.s3_kit.rm_s3_endpoint
        
        self.init()
        return None
    
    def init(self, models=None):
        test_models = [ self.api_models.test_model(model) for model in models ]
        models_dict = { model: test_models[i] for i, model in enumerate(models) }
        api_types = ["groq", "hf_tei", "hf_tgi", "llvm", "ollama", "openai", "s3_kit", "ovms"]
        
        for api_type in api_types:
            if api_type not in dir(self):
                init_method = getattr(self,"init_" + api_type)
                init_method(models)
            if api_type not in globals():
                absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), api_type + ".py"))
                try:
                    with open(absolute_path, encoding='utf-8') as f:
                        exec(f.read())
                except Exception as e:
                    print(e)
                    pass
                with open(absolute_path, encoding='utf-8') as f:
                    globals()[api_type] = exec(f.read())
            else:
                pass
            if api_type not in list(self.resources.keys()):
                # from .api_types[api_type] import api_type
                ## import the api_type class from the api_types module
                self.resources[api_type] = api_type(self.resources, self.metadata)
                setattr(self, api_type, self.resources[api_type])
            else:
                setattr(self, api_type, self.resources[api_type])
            setattr(self, "create_" + api_type + "_endpoint_handler", getattr(self, api_type).create_endpoint_handler)
            setattr(self, "make_post_request_" + api_type, getattr(self, api_type).make_post_request)   
                    
        for model in list(models_dict.keys()):
            for api_type in api_types:
                if api_type + "_endpoints"in list(self.endpoints.keys()):
                    if len(self.endpoints[api_type + "_endpoints"]) > 0:
                        for endpoint_model in list(self.endpoints[api_type + "_endpoints"].keys()):
                            for endpoint in self.endpoints[api_type + "_endpoints"][endpoint_model]:
                                if len(endpoint) == 3:
                                    this_model = endpoint[0]
                                    this_endpoint = endpoint[1]
                                    context_length = endpoint[2]
                                    if model == this_model:
                                        if endpoint not in list(self.batch_sizes[model].keys()):
                                            self.resources["batch_sizes"][model][this_endpoint] = 0
                                        self.resources["queues"][model][this_endpoint] = asyncio.Queue(64)
                                        self.resources["endpoint_handler"][model][this_endpoint] = self.create_endpoint_handler(model, this_endpoint, context_length)
                                        # self.resources["consumer_tasks"][model][this_endpoint] = asyncio.create_task(self.endpoint_consumer(self.resources["queues"][model][this_endpoint], 64, model, this_endpoint))
        return None
    
    def init_groq(self, models=None):
        return None
    
    def init_llvm(self, models=None):
        return None
    
    def init_tei(self, models=None):
        return None
    
    def init_tgi(self, models=None):
        return None
    
    def init_ollama(self, models=None):
        return None
    
    def init_openai(self, models=None):
        return None
    
    def init_s3(self, models=None):
        return None
    
    def init_ovms(self, models=None):
        return None
    
    
    async def add_ovms_endpoint(self, model, endpoint):
        return None

    async def add_llvm_endpoint(self, model, endpoint):
        return None
    
    async def add_tei_endpoint(self, model, endpoint):
        return None
    
    async def add_groq_endpoint(self, model, endpoint):
        return None

    async def add_ollama_endpoint(self, model, endpoint):
        return None
    
    async def add_tgi_endpoint(self, model, endpoint):
        return None
    
    async def add_openvino_endpoint(self, model, endpoint):
        return None
    
    async def add_libp2p_endpoint(self, model, endpoint):
        return None
    
    async def add_openai_endpoint(self, model, endpoint):
        return None
    
    async def add_s3_endpoint(self, model, endpoint):
        return None
    
    async def rm_ovms_endpoint(self, model, endpoint):
        return None
    
    async def rm_llvm_endpoint(self, model, endpoint):
        return None
    
    async def rm_tei_endpoint(self, model, endpoint):
        return None
    
    async def rm_groq_endpoint(self, model, endpoint):
        return None
    
    async def rm_ollama_endpoint(self, model, endpoint):
        return None
    
    async def rm_tgi_endpoint(self, model, endpoint):
        return None
    
    async def rm_openvino_endpoint(self, model, endpoint):
        return None
    
    async def rm_libp2p_endpoint(self, model, endpoint):
        return None
    
    async def rm_openai_endpoint(self, model, endpoint):
        return None
    
    async def rm_s3_endpoint(self, model, endpoint):
        return None
    
    async def test_ovms_endpoint(self, model, endpoint):
        return None

    async def test_llvm_endpoint(self, model, endpoint):
        return None
    
    async def test_tei_endpoint(self, model, endpoint):
        return None
    
    async def test_groq_endpoint(self, model, endpoint):
        return None
    
    async def test_ollama_endpoint(self, model, endpoint):
        return None
    
    async def test_tgi_endpoint(self, model, endpoint):
        return None
    
    async def test_openvino_endpoint(self, model, endpoint):
        return None

    async def test_libp2p_endpoint(self, model, endpoint):
        return None

    async def test_openai_endpoint(self, model, endpoint):
        return None
    
    async def test_s3_endpoint(self, model, endpoint):
        return None
    
    
    
    async def request_ovms_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
        batch_size = len(batch)
        if model in self.ovms_endpoints:
            for endpoint in self.ovms_endpoints[model]:
                if self.batch_sizes[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_ollama_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
        batch_size = len(batch)
        if model in self.local_endpoints:
            for endpoint in self.local_endpoints[model]:
                if self.batch_sizes[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_libp2p_endpoint(self, model, endpoint, endpoint_type, batch):
        batch_size = len(batch)
        if model in self.libp2p_endpoints:
            for endpoint in self.libp2p_endpoints[model]:
                if self.batch_sizes[endpoint] >= batch_size:
                    return endpoint
        return None
    
    
    def request_tei_endpoint(self, model, endpoint=None, endpoint_type=None, batch=None):
        incoming_batch_size = len(batch)
        endpoint_batch_size = 0
        if endpoint in self.endpoint_status:
            endpoint_batch_size = self.endpoint_status[endpoint]
        elif endpoint_type == None:
            for endpoint_type in self.endpoint_types:
                if endpoint_type in self.__dict__.keys():
                    if model in self.__dict__[endpoint_type]:
                        for endpoint in self.__dict__[endpoint_type][model]:
                            endpoint_batch_size = self.endpoint_status[endpoint]
                            if self.endpoint_status[endpoint] >= incoming_batch_size:
                                return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
                else:
                    pass
        else:
            if model in self.__dict__[endpoint_type]:
                for endpoint in self.__dict__[endpoint_type][model]:
                    endpoint_batch_size = self.endpoint_status[endpoint]
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
            else:
                return None
                
        if incoming_batch_size > endpoint_batch_size:
            return ValueError("Batch size too large")
        else:
            if model in self.endpoints:
                for endpoint in self.tei_endpoints[model]:
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
            return None
    
    def request_tgi_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
        incoming_batch_size = len(batch)
        endpoint_batch_size = 0
        if endpoint in self.endpoint_status:
            endpoint_batch_size = self.endpoint_status[endpoint]
        elif endpoint_type == None:
            for endpoint_type in self.endpoint_types:
                if endpoint_type in self.__dict__.keys():
                    if model in self.__dict__[endpoint_type]:
                        for endpoint in self.__dict__[endpoint_type][model]:
                            endpoint_batch_size = self.endpoint_status[endpoint]
                            if self.endpoint_status[endpoint] >= incoming_batch_size:
                                return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
                else:
                    pass
        else:
            if model in self.__dict__[endpoint_type]:
                for endpoint in self.__dict__[endpoint_type][model]:
                    endpoint_batch_size = self.endpoint_status[endpoint]
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
            else:
                return None
                
        if incoming_batch_size > endpoint_batch_size:
            return ValueError("Batch size too large")
        else:
            if model in self.endpoints:
                for endpoint in self.tei_endpoints[model]:
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
            return None
              
    
    async def request_ovms_endpoint(self, model, batch_size):
        if model in self.openvino_endpoints:
            for endpoint in self.openvino_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_tei_endpoint(self, model, batch_size):
        if model in self.tei_endpoints:
            for endpoint in self.tei_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None

    async def request_groq_endpoint(self, model, batch_size):
        if model in self.groq_endpoints:
            for endpoint in self.groq_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_llvm_endpoint(self, model, batch_size):
        if model in self.llvm_endpoints:
            for endpoint in self.llvm_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_openai_endpoint(self, model, batch_size):
        if model in self.openai_endpoints:
            for endpoint in self.openai_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_s3_endpoint(self, model, batch_size):
        if model in self.s3_endpoints:
            for endpoint in self.s3_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_ollama_endpoint(self, model, batch_size):
        if model in self.local_endpoints:
            for endpoint in self.local_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    def __test__(self):
        return None
