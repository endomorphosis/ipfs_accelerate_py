from .ovms import ovms
from .groq import groq
from .hf_tei import hf_tei
from .hf_tgi import hf_tgi
from .llvm import llvm
from .s3_kit import s3_kit
from .openai_api import openai_api
from .ollama import ollama
import asyncio

class apis:
    def __init__(self, resources, metadata):
        if resources is None:
            resources = {}
        if metadata is None:
            metadata = {}     
        self.resources = resources
        self.metadata = metadata
        self.init()
        self.make_post_request_openvino = self.ovms.make_post_request_openvino
        self.test_openvino_endpoint = self.ovms.test_openvino_endpoint
        self.add_groq_endpoint = self.groq.add_groq_endpoint
        self.add_llvm_endpoint = self.llvm.add_llvm_endpoint
        self.add_tei_endpoint = self.hf_tei.add_tei_endpoint
        self.add_tgi_endpoint = self.hf_tgi.add_tgi_endpoint
        self.add_ollama_endpoint = self.ollama.add_ollama_endpoint
        self.add_ovms_endpoint = self.ovms.add_ovms_endpoint
        self.add_libp2p_endpoint = self.ovms.add_libp2p_endpoint
        self.add_openai_endpoint = self.openai_api.add_openai_endpoint
        self.add_s3_endpoint = self.s3_kit.add_s3_endpoint
        return None
    
    def init(self, models=None):
        if "groq" not in dir(self):
            if "groq" not in list(self.resources.keys()):
                from .groq import groq
                self.resources["groq"] = groq(self.resources, self.metadata)
                self.groq = self.resources["groq"]
            else:
                self.groq = self.resources["groq"]
        
        if "hf_tei" not in dir(self):
            if "hf_tei" not in list(self.resources.keys()):
                from .hf_tei import hf_tei
                self.resources["hf_tei"] = hf_tei(self.resources, self.metadata)
                self.hf_tei = self.resources["hf_tei"]
            else:
                self.hf_tei = self.resources["hf_tei"]
                
        if "hf_tgi" not in dir(self):
            if "hf_tgi" not in list(self.resources.keys()):
                from .hf_tgi import hf_tgi
                self.resources["hf_tgi"] = hf_tgi(self.resources, self.metadata)
                self.hf_tgi = self.resources["hf_tgi"]
            else:
                self.hf_tgi = self.resources["hf_tgi"]
                
        if "llvm" not in dir(self):
            if "llvm" not in list(self.resources.keys()):
                from .llvm import llvm
                self.resources["llvm"] = llvm(self.resources, self.metadata)
                self.llvm = self.resources["llvm"]
            else:
                self.llvm = self.resources["llvm"]
                
        if "ollama" not in dir(self):
            if "ollama" not in list(self.resources.keys()):
                from .ollama import ollama
                self.resources["ollama"] = ollama(self.resources, self.metadata)
                self.ollama = self.resources["ollama"]
            else:
                self.ollama = self.resources["ollama"]
                
        if "ovms" not in dir(self):
            if "ovms" not in list(self.resources.keys()):
                from .ovms import ovms
                self.resources["ovms"] = ovms(self.resources, self.metadata)
                self.ovms = self.resources["ovms"]
            else:
                self.ovms = self.resources["ovms"]
                
        if "s3_kit" not in dir(self):
            if "s3_kit" not in list(self.resources.keys()):
                from .s3_kit import s3_kit
                self.resources["s3_kit"] = s3_kit(self.resources, self.metadata)
                self.s3_kit = self.resources["s3_kit"]
            else:
                self.s3_kit = self.resources["s3_kit"]
        
        if "openai_api" not in dir(self):
            if "openai_api" not in list(self.resources.keys()):
                from .openai_api import openai_api
                self.resources["openai_api"] = openai_api(self.resources, self.metadata)
                self.openai_api = self.resources["openai_api"]
            else:
                self.openai_api = self.resources["openai_api"]
                
        for model in models:

            if "ovms_endpoints" in list(self.endpoints.keys()):
                if len(self.endpoints["ovms_endpoints"]) > 0 :
                    for endpoint in self.endpoints["ovms_endpoints"][model]:
                        if len(endpoint) == 3:
                            this_model = endpoint[0]
                            this_endpoint = endpoint[1]
                            context_length = endpoint[2]
                            if model == this_model:
                                if endpoint not in list(self.batch_sizes[model].keys()):
                                    # self.batch_sizes[model][this_endpoint] = 0
                                    self.resources["batch_sizes"][model][this_endpoint] = 0
                                # self.queues[model][endpoint] = asyncio.Queue(64)  # Unbounded queue
                                self.resources["queues"][model][this_endpoint] = None
                                self.resources["queues"][model][this_endpoint] = asyncio.Queue(1)  # Unbounded queue
                                # self.endpoint_handler[(model, endpoint)] = self.make_post_request(self.request_openvino_endpoint(model))
                                self.resources["endpoint_handler"][model][this_endpoint] = self.create_openvino_endpoint_handler(model, this_endpoint, context_length)
                                # self.resources["consumer_tasks"][model][this_endpoint] = asyncio.create_task(self.endpoint_consumer(self.resources["queues"][model][this_endpoint], 64, model, this_endpoint))
            if "tei_endpoints" in list(self.endpoints.keys()):
                if len(self.endpoints["tei_endpoints"]) > 0:
                    for endpoint_model in list(self.endpoints["tei_endpoints"].keys()):
                        for endpoint in self.endpoints["tei_endpoints"][endpoint_model]:                            
                            if len(endpoint) == 3:
                                this_model = endpoint[0]
                                this_endpoint = endpoint[1]
                                context_length = endpoint[2]
                                if model == this_model:
                                    if endpoint not in list(self.batch_sizes[model].keys()):
                                        self.resources["batch_sizes"][model][this_endpoint] = 0
                                    self.resources["queues"][model][this_endpoint] = asyncio.Queue(64)  # Unbounded queue
                                    self.resources["endpoint_handler"][model][this_endpoint] = self.create_tei_endpoint_handler(model, this_endpoint, context_length)
                                    # self.resources["consumer_tasks"][model][this_endpoint] = asyncio.create_task(self.endpoint_consumer(self.resources["queues"][model][this_endpoint], 64, model, this_endpoint))
            if "libp2p_endpoints" in list(self.endpoints.keys()):
                if len(self.endpoints["libp2p_endpoints"]) > 0:
                    for endpoint in self.endpoints["libp2p_endpoints"]:
                        if len(endpoint) == 3:
                            this_model = endpoint[0]
                            this_endpoint = endpoint[1]
                            context_length = endpoint[2]
                            if model == this_model:
                                if endpoint not in list(self.batch_sizes[model].keys()):
                                    # self.batch_sizes[model][this_endpoint] =  0
                                    self.resources["batch_sizes"][model][this_endpoint] = 0
                                self.resources["queues"][model][endpoint] = asyncio.Queue(64)  # Unbounded queue
                                self.resources["endpoint_handler"][model][endpoint] = self.create_libp2p_endpoint_handler(model, this_endpoint, context_length)
                                # self.resources["consumer_tasks"][model][this_endpoint] = asyncio.create_task(self.endpoint_consumer(self.resources["queues"][model][this_endpoint], 64, model, this_endpoint))    
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
    
    async def request_ollama_endpoint(self, model, batch_size):
        if model in self.local_endpoints:
            for endpoint in self.local_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_libp2p_endpoint(self, model, batch_size):
        if model in self.libp2p_endpoints:
            for endpoint in self.libp2p_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None          
              
    def __test__(self):
        return None
