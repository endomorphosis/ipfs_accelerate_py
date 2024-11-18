import torch
import requests
import json
import os
import sys
import asyncio
import random
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from transformers import AutoTokenizer
from transformers import AutoModel
import hashlib
import time
from torch import Tensor

class ipfs_accelerate_py:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        if self.resources is None:
            self.resources = {}
        if self.metadata is None:
            self.metadata = {}
        if resources is None:
            resources = {}
        if metadata is None:
            metadata = {}
        if "queues" not in list(self.resources.keys()):
            self.resources["queues"] = {}
        if "queue" not in list(self.resources.keys()):
            self.resources["queue"] = {}
        if "batch_sizes" not in list(self.resources.keys()):
            self.resources["batch_sizes"] = {}
        if "endpoint_handler" not in list(self.resources.keys()):
            self.resources["endpoint_handler"] = {}
        if "consumer_tasks" not in list(self.resources.keys()):
            self.resources["consumer_tasks"] = {}
        if "queue_tasks" not in list(self.resources.keys()):
            self.resources["queue_tasks"] = {}
        if "caches" not in list(self.resources.keys()):
            self.resources["caches"] = {}
        if "tokenizer" not in list(self.resources.keys()):
            self.resources["tokenizer"] = {}
        self.resources["ipfs_accelerate_py"] = self
        if "test_ipfs_accelerate_py" not in globals() and "test_ipfs_accelerate" not in list(self.resources.keys()):
            try:
                from .test_ipfs_accelerate import test_ipfs_accelerate
            except:
                from test_ipfs_accelerate import test_ipfs_accelerate
            self.test_ipfs_accelerate = test_ipfs_accelerate(self.resources, self.metadata)
            resources["test_ipfs_accelerate"] = self.test_ipfs_accelerate
        elif "test_ipfs_accelerate" in list(self.resources.keys()):
            self.test_ipfs_accelerate = self.resources["test_ipfs_accelerate"]
        elif "test_ipfs_accelerate" in globals():
            self.test_ipfs_accelerate = test_ipfs_accelerate(self.resources, self.metadata) 
            resources["test_ipfs_accelerate"] = self.test_ipfs_accelerate
        if "install_depends_py" not in globals():
            try:
                from .install_depends import install_depends_py
            except:
                from install_depends import install_depends_py
            self.install_depends = install_depends_py(resources, metadata)
            resources["install_depends"] = self.install_depends 
        else:
            self.install_depends = install_depends_py(resources, metadata)
        if "worker" not in globals():
            try:
                from .worker import worker
            except:
                from worker import worker
            self.worker = worker.worker_py(resources, metadata)
            resources["worker"] = self.worker
        if "ipfs_multiformats" not in globals():
            try:
                from .ipfs_multiformats import ipfs_multiformats_py
            except:
                from ipfs_multiformats import ipfs_multiformats_py
            self.ipfs_multiformats = ipfs_multiformats_py(resources, metadata)
            resources["ipfs_multiformats"] = self.ipfs_multiformats
        self.endpoint_status = {}
        self.endpoint_handler = {}
        self.endpoints = {}
        self.batch_sizes = {}
        self.inbox = {}
        self.outbox = {}
        self.local_queues = {}
        self.tokenizer = {}
        self.local_queues = {}
        self.queues = {}
        self.request = {}
        self.local_endpoints = {}
        self.tei_endpoints = {}
        self.openvino_endpoints = {}
        self.libp2p_endpoints = {}
        self.caches = {}
        self.endpoint_types = ["tei_endpoints", "openvino_endpoints", "libp2p_endpoints", "local_endpoints"]
        self.add_endpoint = self.add_endpoint
        self.rm_endpoint = self.rm_endpoint
        self.get_endpoints = self.get_endpoints
        self.init_endpoints = self.init_endpoints
        self.get_https_endpoint = self.get_https_endpoint
        self.get_libp2p_endpoint = self.get_libp2p_endpoint
        self.request_tei_endpoint = self.request_tei_endpoint
        self.request_libp2p_endpoint = self.request_libp2p_endpoint
        self.request_openvino_endpoint = self.request_openvino_endpoint
        self.request_local_endpoint = self.request_local_endpoint
        self.request_llama_cpp_endpoint = self.request_llama_cpp_endpoint
        self.request_local_endpoint = self.request_local_endpoint
        self.make_local_request = self.make_local_request
        self.add_endpoint = self.add_endpoint
        self.rm_endpoint = self.rm_endpoint
        self.get_endpoints = self.get_endpoints
        self.get_https_endpoint = self.get_https_endpoint
        self.get_libp2p_endpoint = self.get_libp2p_endpoint
        self.init_endpoints = self.init_endpoints

        return None
    
    async def test_hardware(self):
        install_file_hash = None
        test_results_file = None
        install_depends_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "install_depends.py")
        if os.path.exists(install_depends_filename):
            ## get the sha256 hash of the file
            sha256 = hashlib.sha256()
            with open(install_depends_filename, "rb") as f:
                for byte_block in iter(lambda: f.read(4096),b""):
                    sha256.update(byte_block)
            install_file_hash = sha256.hexdigest()
            test_results_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"test", install_file_hash + ".json")
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

    async def query_endpoints(self, model):
        endpoints = None
        local = None
        openvino = None
        tei = None
        libp2p = None
        try:
            endpoints = await self.get_endpoints(model)
        except Exception as e:
            endpoints = e
        try:
            local = await self.get_endpoints(model, "local")
        except Exception as e:
            local = e
        try:
            openvino = await self.get_endpoints(model, "openvino")
        except Exception as e:
            openvino = e
        try:
            libp2p = await self.get_endpoints(model, "libp2p")
        except Exception as e:
            libp2p = e
        try:
            tei = await self.get_endpoints(model, "tei")
        except Exception as e:
            tei = e
        if type(tei) == list:
            tei_set = set(endpoints["tei"])
        else:
            tei_set = set()
        if type(local) == list:
            local_set = set(endpoints["local"])
        else:
            local_set = set()
        if type(libp2p) == list:
            libp2p_set = set(endpoints["libp2p"])
        else:
            libp2p_set = set()
        if type(openvino) == list:
            openvino_set = set(endpoints["openvino"])
        else:
            openvino_set = set()

        endpoints_set = set.union(tei_set, local_set, openvino_set, libp2p_set)
        endpoints =  { "tei" : tei , "local" : local , "openvino": openvino , "libp2p": libp2p }

        # endpoints_set = set(set(endpoints["tei"]),set(endpoints["local"]),set(endpoints["openvino"]),set(endpoints["libp2p"]))
        # self.endpoints = endpoints
        # self.endpoints_list = list(endpoints.keys())
        # self.endpoints_set = endpoints_set
        return {
            "endpoints": endpoints,
            "endpoints_set": endpoints_set
        }

    def create_tei_endpoint_handler(self, model, endpoint, context_length):
        async def handler(x):
            remote_endpoint = await self.make_post_request_tei(endpoint, x)
            return remote_endpoint
        return handler
    
    def create_openvino_endpoint_handler(self, model, endpoint, context_length):
        async def handler(x):
            tokenizer = None
            tokens = None
            if model not in list(self.resources["tokenizer"].keys()):
                self.resources["tokenizer"][model] = {}
            tokenizers = list(self.resources["tokenizer"][model].keys())
            if len(tokenizers) == 0:
                self.resources["tokenizer"][model]["cpu"] = AutoTokenizer.from_pretrained(model, device='cpu')
                tokens = await self.resources["tokenizer"][model]["cpu"](x, return_tensors="pt", padding=True, truncation=True)
            else:
                for tokenizer in tokenizers:
                    try:
                        this_tokenizer = self.resources["tokenizer"][model][tokenizer]
                        tokens = await this_tokenizer[model][endpoint](x, return_tensors="pt", padding=True, truncation=True)
                    except Exception as e:
                        pass
            if tokens is None:
                raise ValueError("No tokenizer found for model " + model)            
            tokens = await self.tokenizer[model][endpoint](x, return_tensors="pt", padding=True, truncation=True)
            remote_endpoint = await self.make_post_request_openvino(tokens, x)
            return remote_endpoint
        return handler
    
    def create_libp2p_endpoint_handler(self, model, endpoint, context_length):
        def handler(x):
            remote_endpoint = self.endpoint_handler[model][endpoint]
            request_results = self.request_libp2p_endpoint(model, endpoint, "libp2p_endpoints", x)
            return request_results
        return handler

    async def init_endpoints(self, models=None, endpoint_list=None):

        endpoint_set = set(endpoint_list)
        for endpoint_type in self.endpoint_types:
            if endpoint_type in endpoint_set:
                for endpoint_info in endpoint_list[endpoint_type]:
                    model, endpoint, context_length = endpoint_info
                    if model not in list(self.resources["batch_sizes"].keys()):
                        self.resources["batch_sizes"][model] = {}
                    if model not in list(self.resources["queues"].keys()):
                        self.resources["queues"][model] = {}
                    if endpoint not in list(self.resources["batch_sizes"][model].keys()):
                        self.resources["batch_sizes"][model][endpoint] = 0
                    await self.add_endpoint(model, endpoint_type, endpoint_info)
                else:
                    pass    
        for model in models:
            if model not in self.queues:
                self.resources["queue"][model] = asyncio.Queue(128)
            if model not in list(self.resources["consumer_tasks"].keys()):
                self.resources["consumer_tasks"][model] = {}
        if type(endpoint_list) == list:
            self.endpoints = { k : v for k, v in enumerate(endpoint_list) if endpoint_list[v] in self.endpoint_types or endpoint_list[k] in self.endpoint_types }
            self.endpoint_list = new_endpoints_list
            endpoints_set = set(new_endpoints_list)
            self.endpoint_set = endpoints_set
        if type(endpoint_list) == dict:
            query_endpoints = await self.query_endpoints(model)                
            new_endpoints_list = [ k for k in endpoint_list.keys() if k in self.endpoint_types or endpoint_list[k] in self.endpoint_types ]
            new_endpoints = {}
            endpoints_set = query_endpoints["endpoints_set"]
            for endpoint_type in new_endpoints_list:
                if endpoint_type in list(endpoint_list.keys()):
                    if endpoint_type not in list(new_endpoints.keys()):
                        new_endpoints[endpoint_type] = {}
            for endpoint_type in new_endpoints_list:
                for model in models:
                    if model not in new_endpoints[endpoint_type]:
                        new_endpoints[endpoint_type][model] = []
            for endpoint_type in new_endpoints_list:
                if endpoint_type in endpoint_list:
                    this_list = endpoint_list[endpoint_type]
                    for item in this_list:
                        this_model = item[0]
                        this_endpoint = item[1]
                        this_context_length = item[2]
                        endpoints_set.add(this_endpoint)
                        if this_model in list(new_endpoints[endpoint_type].keys()):
                            new_endpoints[endpoint_type][model].append(item)                
            self.endpoints = new_endpoints
            self.endpoints_list = new_endpoints_list
            self.endpoint_set = endpoints_set
        if endpoint_list is None:    
            query_endpoints = self.query_endpoints(model)
            endpoints = query_endpoints["endpoints"]
            endpoints_set = query_endpoints["endpoints_set"]
            endpoints_list = [ k for k in endpoints.keys() ]
            self.endpoints = endpoints
            self.endpoint_list = endpoints_list
            self.endpoint_set = endpoints_set
        else:
            pass
        if not endpoints_set:
            raise ValueError("No endpoints available for model " + model)
        else:
            local = [ endpoint for endpoint in self.endpoints["local_endpoints"] if "local" in endpoint or "cpu" in endpoint or "cuda" in endpoint or "openvino" in endpoint or "llama_cpp" in endpoint or "ipex" in endpoint]
            libp2p = []
            if "openvino_endpoints" in list(self.endpoints.keys()):
                openvino = [endpoint for endpoint in self.endpoints["openvino_endpoints"]]
            else:
                openvino = []
            
            if "tei_endpoints" in list(self.endpoints.keys()):
                tei = [endpoint for endpoint in self.endpoints["tei_endpoints"]]
            else:
                tei = []
                 
        for model in models:
            if model not in self.tokenizer:
                self.tokenizer[model] = {}
            if model not in self.local_endpoints:
                self.local_endpoints[model] = {}
            if model not in self.queues:    
                self.queues[model] = {}
            if model not in self.caches:
                self.caches[model] = {"items": {}}
            if model not in self.batch_sizes:
                self.batch_sizes[model] = {}
            if model not in self.resources["batch_sizes"]:
                self.resources["batch_sizes"][model] = {}
            if model not in self.resources["queues"]:
                self.resources["queues"][model] = {}
            if model not in self.resources["queue"]:
                self.resources["queue"][model] = {}
            if model not in self.resources["caches"]:
                self.resources["caches"][model] = {"items": {}}
            if model not in self.resources["tokenizer"]:
                self.resources["tokenizer"][model] = {}
            if model not in self.resources["endpoint_handler"]:
                self.resources["endpoint_handler"][model] = {}
            if "cpu" not in self.local_endpoints[model]:
                self.local_endpoints[model]["cpu"] = ""
            if "cpu" not in self.queues[model]:
                self.queues[model]["cpu"] = ""
            if "cpu" not in self.batch_sizes[model]:
                self.batch_sizes[model]["cpu"] = 1
        new_resources = {}    
        try:
            self.worker_resources = await self.worker.init_worker(models, self.endpoints["local_endpoints"], None)
        except Exception as e:
            print(e)
            self.worker_resources = e
            pass
        
        if type(self.worker_resources) is not ValueError:
            resource_list = list(self.worker_resources.keys())
            for resource in resource_list:
                if resource not in list(self.resources.keys()):
                    self.resources[resource] = self.worker_resources[resource]
                    pass
                else:
                    self.resources[resource] = self.worker_resources[resource]
                    pass
            pass
                
        if "openvino_endpoints" in list(self.endpoints.keys()):
            if len(self.endpoints["openvino_endpoints"]) > 0 :
                for endpoint in self.endpoints["openvino_endpoints"][model]:
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
        new_resources = {}
        for resource in resource_list:
            new_resources[resource] = self.resources[resource]
        new_resources["endpoints"] = self.endpoints
        return new_resources

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
        
    async def request_openvino_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
        batch_size = len(batch)
        if model in self.openvino_endpoints:
            for endpoint in self.openvino_endpoints[model]:
                if self.batch_sizes[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_llama_cpp_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
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
    
    async def request_local_endpoint(self, model, endpoint, endpoint_type, batch):
        batch_size = len(batch)
        if model in self.local_endpoints:
            for endpoint in self.local_endpoints[model]:
                if self.batch_sizes[endpoint] >= batch_size:
                    return endpoint
        return None

    async def make_local_request(self, model, endpoint, endpoint_type, data):
        device = torch.device(endpoint)
        inputs = self.tokenizer[model][endpoint](data, return_tensors="pt", padding=True, truncation=True).to(device)
        self.local_endpoints[model][endpoint].to(device).eval()
        with torch.no_grad():
            outputs = self.local_endpoints[model][endpoint](**inputs)
            query_response = outputs.last_hidden_state.mean(dim=1).tolist()  # Use mean of token embeddings
            results = query_response  # Return the entire batch of results
            del inputs, outputs  # Unallocate inputs and outputs
            torch.cuda.synchronize()  # Ensure all operations are complete
            torch.cuda.empty_cache()  # Free up GPU memory
        # self.local_endpoints[model][endpoint].to('cpu')  # Move model back to CPU
        torch.cuda.empty_cache()  # Free up GPU memory again
        return results

    async def add_endpoint(self, model, endpoint_type, endpoint):
        this_model = endpoint[0]
        backend = endpoint[1]
        context_length = endpoint[2]
        if endpoint_type in self.endpoint_types:
            success = False
            try:
                if endpoint_type not in list(dir(self)):
                    self.__dict__[endpoint_type]= {}
                if model not in list(self.__dict__[endpoint_type].keys()):
                    self.__dict__[endpoint_type][model] = {}
                if endpoint not in list(self.__dict__[endpoint_type][model].keys()):
                    self.__dict__[endpoint_type][model][backend] = context_length
                self.endpoint_status[endpoint] = context_length
                success = True
            except Exception as e:
                print(e)
                pass
            return success        
        return None
    
    async def rm_endpoint(self, model, endpoint_type, backend):
        if endpoint_type in self.endpoint_types:
            success = False
            try:
                if model in self.__dict__[endpoint_type] and backend in self.__dict__[endpoint_type][model]:
                    del self.__dict__[endpoint_type][model][backend]
                if backend in self.resources["batch_sizes"][model]:
                    del self.resources["batch_sizes"][model][backend]
                if backend in self.resources["queues"][model]:
                    del self.resources["queues"][model][backend]
                if backend in self.resources["endpoint_handler"][model]:
                    del self.resources["endpoint_handler"][model][backend]
                success = True
            except Exception as e:
                print(e)
                pass
            return success
        return None
    
    async def create_background_tasks(self):
        if "endpoint_handler" in list(self.resources.keys()):
            models = list(self.resources["endpoint_handler"].keys())
            for model in models:
                for endpoint in self.endpoints_list:
                    if model in list(self.resources["endpoint_handler"].keys()):
                        if model not in list(self.resources["consumer_tasks"].keys()):
                            self.resources["consumer_tasks"][model] = {}
                        if model not in list(self.resources["queue_tasks"].keys()):
                            self.resources["queue_tasks"][model] = {}
                        if model not in list(self.resources["queue"].keys()):
                            self.resources["queue"][model] = {}  
                        backends = list(self.resources["queues"][model].keys())
                        queues = list(self.resources["queues"][model].keys())
                        for backend in backends:
                            if model in list(self.resources["endpoint_handler"].keys()) and backend in list(self.resources["endpoint_handler"][model].keys())and backend not in list(self.resources["consumer_tasks"][model].keys()):
                                self.resources["consumer_tasks"][model][endpoint] = asyncio.create_task(self.endpoint_consumer(self.resources["queues"][model][backend], 64, model, self.resources["endpoint_handler"][model][backend]))
                            if model in list(self.resources["endpoint_handler"].keys()):
                                self.resources["queue_tasks"][model][backend] = asyncio.create_task(self.model_consumer(self.resources["queue"][model], 64, model))
        return None
    
    async def model_consumer(self, queue, batch_size, model):
        # print("consumer started for model " + model_name )
        batch = []
        results = None
        filtered_results = []
        while True:
            try:
                if queue.empty():
                    if len(batch) == 0:
                        await asyncio.sleep(0.1)
                else:
                    queue_length = queue.qsize()
                    endpoint_queue_lengths = {}
                    endpoint_queue_remaining = {}
                    for model in list(self.resources["queues"].keys()):
                        endpoint_queue_lengths[model] = {}
                        endpoint_queue_remaining[model] = {}
                        for endpoint in list(self.resources["queues"][model].keys()):
                            endpoint_queue_lengths[model][endpoint] = self.resources["queues"][model][endpoint].qsize()
                            endpoint_queue_remaining[model][endpoint] = self.resources["queues"][model][endpoint]._maxsize - self.resources["queues"][model][endpoint].qsize()
                    most_empty_endpoint = max(endpoint_queue_remaining[model], key=endpoint_queue_remaining[model].get)
                    most_full_endpoint = min(endpoint_queue_remaining[model], key=endpoint_queue_remaining[model].get)
                    if queue_length <= endpoint_queue_remaining[model][most_empty_endpoint]:
                        num_added = 0
                        while not queue.empty() and num_added < endpoint_queue_remaining[model][most_empty_endpoint]:
                            item = await queue.get()
                            self.resources["queues"][model][most_empty_endpoint].put_nowait(item)                        
                            queue.task_done()
                            num_added += 1
                    elif queue_length > endpoint_queue_remaining[model][most_empty_endpoint]:
                        while not queue.empty():
                            num_added = 0
                            while not queue.empty() and num_added < endpoint_queue_remaining[model][most_empty_endpoint]:
                                item = await queue.get()
                                self.resources["queues"][model][most_empty_endpoint].put_nowait(item)                        
                                num_added += 1
                                del item
                                queue.task_done()
                            num_added = 0
                            for endpoint in list(self.resources["queues"][model].keys()):
                                endpoint_queue_lengths[model][endpoint] = self.resources["queues"][model][endpoint].qsize()
                                endpoint_queue_remaining[model][endpoint] = self.resources["batch_sizes"][model][endpoint] - endpoint_queue_lengths[model][endpoint]
                            most_full_endpoint = max(endpoint_queue_remaining, key=endpoint_queue_remaining.get)
                            most_empty_endpoint = min(endpoint_queue_remaining, key=endpoint_queue_remaining.get)
            except Exception as e:
                print("error in model_consumer")
                print(e)
        return None
    
    async def queue(self, models, batch_data):
        for model in models:
            for item in range(len(batch_data)):
                ipfs_cid = self.ipfs_multiformats.get_cid(batch_data[item])
                cid_value = batch_data[item]
                queue_insert = {ipfs_cid: cid_value}
                if model in list(self.resources["queues"].keys()):
                    self.resources["queue"][model].put_nowait(queue_insert)
                    
        return None
    
    async def fetch(self, models, batch_data):
        return_results  = []
        for model in models:
            while len(batch_data) != len(return_results):
                while len(self.resources["caches"][model]["items"]) == 0:
                    if model in list(self.resources["queue"].keys()):
                        if self.resources["queue"][model].empty():
                            await asyncio.sleep(0.1)
                        else:
                            print("queue not empty")
                            await asyncio.sleep(0.1)       
                for item in range(len(batch_data)):
                    ipfs_cid = self.ipfs_multiformats.get_cid(batch_data[item])
                    if ipfs_cid in list(self.resources["caches"][model]["items"].keys()):
                        return_results.append({ipfs_cid: self.resources["caches"][model]["items"][ipfs_cid]})
                        del self.resources["caches"][model]["items"][ipfs_cid]
                await asyncio.sleep(0.1)
        return return_results
    
    async def endpoint_consumer(self, queue, batch_size, model_name, endpoint):
        # print("consumer started for model " + model_name + " at endpoint " + endpoint)
        batch = []
        results = None
        filtered_results = {}
        items = []
        cids = []
        while True:
            try:
                if queue.empty():
                    if len(batch) == 0:
                        await asyncio.sleep(0.1)
                    else:
                        # Process batch
                        items = []
                        cids = []
                        for i in range(len(batch)):
                            item = batch[i]
                            key = list(item.keys())[0]
                            value = item[key]
                            items.append(value)
                            cids.append(key)
                        try:
                            results = await endpoint(items)
                        except Exception as e:
                            results = e
                        if type(results) == ValueError or type(results) == Exception or type(results) or TypeError:
                            try:
                                results = endpoint(items)
                            except Exception as e:
                                results = e
                                pass
                        if type(results) != ValueError and results is not None:
                            print(results)
                            filtered_results = {}
                            for key in list(results.keys()):
                                if type(results[key]) == Tensor:
                                    tensor_list = results[key].tolist()
                                    filtered_results[key] = tensor_list
                                else:
                                    filtered_results[key] = results[key]
                            filtered_results = [filtered_results]
                            if len(cids) <= 1:
                                self.resources["caches"][model_name]["items"][cids[0]] = filtered_results[0]
                            else:
                                for i in range(len(cids)):
                                    self.resources["caches"][model_name]["items"][cids[i]] = filtered_results[i]
                            batch = []
                else:
                    item = await queue.get()  # Wait for item
                    batch.append(item)
                    if len(batch) >= batch_size:
                        for i in range(len(batch)):
                            if i <= batch_size:
                                item = batch[i]
                                key = list(item.keys())[0]
                                value = item[key]
                                items.append(value)
                        try:
                            results = await endpoint(items)
                        except Exception as e:
                            results = e
                        try:
                            results = endpoint(items)
                        except Exception as e:
                            results = e
                            pass
                        if type(results) != ValueError:
                            print(results)
                            self.resources["caches"][model_name]["items"] + results
                            batch = []
                        if type(results) != ValueError and results is not None:
                            print(results)
                            filtered_results = {}
                            for key in list(results.keys()):
                                if type(results[key]) == Tensor:
                                    tensor_list = results[key].tolist()
                                    filtered_results[key] = tensor_list
                                else:
                                    filtered_results[key] = results[key]
                            filtered_results = [filtered_results]
                            if len(cids) <= 1:
                                self.resources["caches"][model_name]["items"][cids[0]] = filtered_results[0]
                            else:
                                for i in range(len(cids)):
                                    self.resources["caches"][model_name]["items"][cids[i]] = filtered_results[i]
                            batch = []
            except Exception as e:
                print(e)
                pass
        return None
    
    async def max_batch_size(self, model, endpoint, endpoint_handler):
        import psutil
        process = psutil.Process(os.getpid())
        embed_fail = False
        context_length = None
        exponent = int(0)
        batch = []
        token_length_size = 0
        batch_size = 2**exponent
        test_tokens = []
        endpoint_types = list(self.endpoints.keys())
        find_token_str = str("z")
        this_tokenizer = self.resources["tokenizer"][model][endpoint]
        for endpoint_type in endpoint_types:
            endpoints = self.endpoints[endpoint_type]
            if model in list(endpoints.keys()):
                for this_endpoint in endpoints[model]:
                    if model == this_endpoint[0] and ( endpoint == this_endpoint[1] or this_endpoint[1] in list(self.resources["endpoint_handler"][model].keys()) ):
                        context_length = this_endpoint[2]
                        token_length_size = round(int(context_length) * 0.99)
                        break
            if token_length_size > 0:
                break
                        
        find_token_int = this_tokenizer.encode(find_token_str)
        if len(find_token_int) == 3:
            find_token_int = find_token_int[1]
        elif len(find_token_int) == 2:
            find_token_int = find_token_int[1]
        elif len(find_token_int) == 1:
            find_token_int = find_token_int[0]
        for i in range(token_length_size):
            test_tokens.append(find_token_int)
        test_text = this_tokenizer.decode(test_tokens)
        memory_increase = None
        while not embed_fail:
            test_batch = []
            exponent += 1
            for i in range(2**(exponent)):
                test_batch.append(test_text)
            parsed_knn_embeddings = None
            embeddings = None
            request_knn_results = None
            start_time = time.time()
            start_mem = process.memory_info().rss
            free_memory = psutil.virtual_memory().free
            if memory_increase is not None:
                if free_memory < (memory_increase * 2):
                    embed_fail = True
                    break
                    raise(ValueError("the system does not free system memory for batch size " + str(2**(exponent-1))))
            try:
                if "cuda" not in endpoint and "cpu" not in endpoint and "openvino:" not in endpoint:
                    request_knn_results = await endpoint_handler({"inputs": test_batch})
                    end_memory = process.memory_info().rss

                elif "cuda" in endpoint or "cpu" in endpoint or "openvino:" in endpoint:
                    try:
                        request_knn_results = await endpoint_handler(test_batch)
                        end_memory = process.memory_info().rss
                    except Exception as e:
                        pass
                    if request_knn_results == None:
                        try:
                            request_knn_results = endpoint_handler(test_batch)
                            end_memory = process.memory_info().rss
                        except Exception as e:
                            request_knn_results = e
                            embed_fail = True
                            end_memory = process.memory_info().rss
                            pass
                        pass
            except Exception as e:
                request_knn_results = e
                embed_fail = True
                end_memory = process.memory_info().rss
                pass
            if request_knn_results is None or type(request_knn_results) is None or type(request_knn_results) is ValueError or type(request_knn_results) is Exception or type(request_knn_results) is str or type(request_knn_results) is int:
                embed_fail = True
            end_time = time.time()
            batch_size = 2**(exponent-1)
            elapsed_time = end_time - start_time
            memory_increase = end_memory - start_mem
            free_memory = psutil.virtual_memory().free
            log = {
            "batch size": batch_size,
            "elapsed_time": elapsed_time,
            "memory_increase": memory_increase,
            "free_memory": free_memory
            }
            print(log)
            self.resources["batch_sizes"][model][endpoint] = int(2**(exponent-1))
            if batch_size >= 4096:
                embed_fail = True
                pass
        if exponent == 0:
            return 1
        else:
            return 2**(exponent-1)

    async def max_batch_size_bak(self, model, endpoint=None, endpoint_type=None ):
        embed_fail = False
        exponent = 0
        batch = []
        token_length_size = 0
        batch_size = 2**exponent
        if endpoint_type is None:
            this_model = None
            this_endpoint = None
            this_context_length = None
            if "/embed" in endpoint:
                endpoint_type = "tei_endpoints"
            elif "/infer" in endpoint:
                endpoint_type = "openvino_endpoints"
            elif "http" in endpoint:
                endpoint_type = "tei_endpoints"
            elif "cuda" in endpoint or "cpu" in endpoint or "local" in endpoint:
                endpoint_type = "local_endpoints"
            elif "libp2p" in endpoint:
                endpoint_type = "libp2p_endpoints"
            if endpoint_type is None:
                print('Endpoint not found')
                return 0
            else:
                pass
                  
            for this_endpoint in self.endpoints[endpoint_type]:
                if "cuda" in this_endpoint[1] or "cpu" in this_endpoint[1] or "local" in this_endpoint[1]:
                    this_endpoint_index = self.endpoints[endpoint_type].index(this_endpoint)
                    token_length_size = round(self.endpoints["local_endpoints"][this_endpoint_index][2] * 0.99)
                elif model is this_endpoint[0]:
                    this_endpoint_index = self.endpoints[endpoint_type].index(this_endpoint)
                    token_length_size = round(self.endpoints[endpoint_type][this_endpoint_index][2] * 0.99) 
            
            test_tokens = []
            if model not in self.tokenizer.keys():
                self.tokenizer[model] = {}
            if "cpu" not in self.tokenizer[model].keys():
                self.tokenizer[model]["cpu"] = AutoTokenizer.from_pretrained(model, device='cpu')
            find_token_str = str("z")
            find_token_int = self.tokenizer[model]["cpu"].encode(find_token_str)
            if len(find_token_int) == 3:
                find_token_int = find_token_int[1]
            elif len(find_token_int) == 2:
                find_token_int = find_token_int[1]
            elif len(find_token_int) == 1:
                find_token_int = find_token_int[0]
            for i in range(token_length_size):
                test_tokens.append(find_token_int)
            test_text = self.tokenizer[model]["cpu"].decode(test_tokens)
            if endpoint is None:
                endpoint = self.choose_endpoint(model)
            while not embed_fail:
                test_batch = []
                for i in range(batch_size):
                    test_batch.append(test_text)
                parsed_knn_embeddings = None
                embeddings = None
                request_knn_results = None
                try:
                    request_knn_results = await self.request_knn(test_batch, model, endpoint, endpoint_type)
                except Exception as e:
                    try:
                        embeddings = await self.index_knn(test_batch, model, endpoint)
                    except Exception as e:
                            pass
                if request_knn_results != None and parsed_knn_embeddings == None:
                    parsed_knn_embeddings = await self.parse_knn(request_knn_results, model, endpoint, endpoint_type)
                if parsed_knn_embeddings is not None:
                    embeddings = parsed_knn_embeddings
                embed_fail = True
                
        self.endpoint_status[endpoint] = 2**(exponent-1)
        if exponent == 0:
            return 1
        else:
            return 2**(exponent-1)
    
    async def request_openvino_endpoint(self, model, batch_size):
        if model in self.openvino_endpoints:
            for endpoint in self.openvino_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_llama_cpp_endpoint(self, model, batch_size):
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
    
    async def request_local_endpoint(self, model, batch_size):
        if model in self.local_endpoints:
            for endpoint in self.local_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None

    async def test_endpoints(self, models, endpoint_handler_object=None):
        test_results = {}
        for model in models:
            if model not in list(test_results.keys()):
                test_results[model] = {}
            try: 
                test_results[model]["local_endpoint"] = await self.test_local_endpoint(model)
            except Exception as e:
                test_results[model]["local_endpoint"] = e
            try:
                test_results[model]["libp2p_endpoint"] = await self.test_libp2p_endpoint(model)
            except Exception as e:
                test_results[model]["libp2p_endpoint"] = e
            try:
                test_results[model]["openvino_endpoint"] = await self.test_openvino_endpoint(model)
            except Exception as e:
                test_results[model]["openvino_endpoint"] = e
            try:
                test_results[model]["tei_endpoint"] = await self.test_tei_endpoint(model)
            except Exception as e:
                test_results[model]["tei_endpoint"] = e
            try:
                test_results[model]["webnn_endpoint"] = "not implemented"
            except Exception as e:
                test_results[model]["webnn_endpoint"] = e
        try:
            test_results[model]["endpoint_handler_resources"] = endpoint_handler_object
        except Exception as e:
            test_results[model]["endpoint_handler_resources"] = e
            test_results["batch_sizes"] = {}
            test_results["endpoint_handler"] = {}            
        try:    
            batch_sizes = self.resources["batch_sizes"]
            endpoint_handler = self.resources["endpoint_handler"]
            endpoint_tests = {}
            batch_sizes = {}
            for endpoint in endpoint_handler:
                this_model = endpoint
                if this_model not in list(endpoint_tests.keys()):
                    endpoint_tests[this_model] = {}
                if this_model not in list(batch_sizes.keys()):
                    batch_sizes[this_model] = {}
                endpoints_by_model = endpoint_handler[this_model]
                for endpoint_type in list(endpoints_by_model.keys()):
                    if endpoint_type not in list(endpoint_tests[this_model].keys()):
                        endpoint_tests[this_model][endpoint_type] = {}
                    if endpoint_type not in list(batch_sizes[this_model].keys()):
                        batch_sizes[this_model][endpoint_type] = {}
                    this_endpoint = endpoints_by_model[endpoint_type]
                    batch_size = batch_sizes[this_model][endpoint_type]
                    test_batch_size = await self.max_batch_size(this_model, endpoint_type, this_endpoint)
                    self.resources["batch_sizes"][this_model][endpoint_type] = test_batch_size
                    batch_sizes[this_model][endpoint_type] = test_batch_size
                    endpoint_tests[this_model][endpoint_type] = this_endpoint
            test_results["batch_sizes"] = batch_sizes
            test_results["endpoint_handler"] = endpoint_tests
        except Exception as e:
            test_results["batch_sizes"] = e
            test_results["endpoint_handler"] = e
            pass
        return test_results
    
    async def test_tei_endpoint(self, model, endpoint_list=None):
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        tei_endpoints = self.endpoints["tei_endpoints"]
        tei_endpoints_by_model = list(self.tei_endpoints.keys())
        endpoint_handlers_by_model = list(self.resources["endpoint_handler"][model].keys())
        if endpoint_list is not None:            
            for endpoint in tei_endpoints:
                    this_model = endpoint[0]
                    this_endpoint = endpoint[1]
                    this_data = endpoint[2]
                    if this_model == model and this_endpoint == endpoint:
                        filtered_list[this_endpoint] = endpoint
        if this_endpoint is not None:
            endpoint_list = [model,endpoint,""]
            for endpoint in tei_endpoints:
                this_model = endpoint[0]
                this_endpoint = endpoint[1]
                this_data = endpoint[2]
                if this_model == endpoint_list[0] and this_endpoint == endpoint_list[1]:
                    filtered_list[this_endpoint] = endpoint
        else:
            endpoint_list = [model,"",""]
            for endpoint in tei_endpoints:
                print(endpoint)
                this_model = endpoint[0]
                this_endpoint = endpoint[1]
                this_data = endpoint[2]
                if this_model == model:
                    filtered_list[this_endpoint] = endpoint
        endpoint_handlers = {}
        if len(filtered_list.keys()) > 0:
            for endpoint in list(filtered_list.keys()):
                if endpoint in endpoint_handlers_by_model:
                    endpoint_handlers[endpoint] = self.resources["endpoint_handler"][model][endpoint]
        else:
            return ValueError("No endpoints found")
        if len(endpoint_handlers) > 0:
            for endpoint in list(endpoint_handlers.keys()):
                try:
                    endpoint_handler = endpoint_handlers[endpoint]
                    test = await endpoint_handlers[endpoint]({"inputs": "hello world"})
                    test_results[endpoint] = test
                except Exception as e:
                    test_results[endpoint] = e
                    pass
        else:
            return ValueError("No endpoint_handlers found")
        return test_results
    
    async def test_libp2p_endpoint(self, model, endpoint=None):
        return ValueError("Not implemented")

    async def test_openvino_endpoint(self, model, endpoint=None):
        this_endpoint = None
        filtered_list = []
        test_results = {}
        if type(model) is str:
            model = [model]
        if endpoint is not None:
            for endpoint in self.resources["openvino_endpoints"]:
                this_model = endpoint[0]
                this_endpoint = endpoint[1]
                this_data = endpoint[2]
                if this_model[0] == model and this_endpoint == endpoint:
                    filtered_list.append(endpoint)
        if this_endpoint is not None:
            endpoint_list = [model[0],endpoint,""]
            for endpoint in self.resources["openvino_endpoints"]:
                this_model = endpoint[0]
                this_endpoint = endpoint[1]
                this_data = endpoint[2]
                if this_model[0] == endpoint_list[0] and this_endpoint == endpoint_list[1]:
                    filtered_list.append(endpoint)            
        else:
            endpoint_list = [model[0],"",""]
            for endpoint in self.resources["openvino_endpoints"]:
                print(endpoint)
                this_model = endpoint[0]
                this_endpoint = endpoint[1]
                this_data = endpoint[2]
                if this_model == endpoint_list[0]:
                    filtered_list.append(endpoint)
        endpoint_handlers = []
        if len(filtered_list) > 0:
            for endpoint in filtered_list:
                this_model = endpoint[0]
                this_endpoint = endpoint[1]
                this_data = endpoint[2]
                endpoint_handler = self.endpoint_handler[this_model][this_endpoint]
                endpoint_handlers.append((this_model, this_endpoint, endpoint_handler))
        else:
            return ValueError("No endpoints found")
        if len(endpoint_handlers) > 0:
            for i in endpoint_handlers:
                model = i[0]
                endpoint = i[1]
                endpoint_handler = i[2]
                test_endpoint = False
                try:
                    test_endpoint = endpoint_handler("hello world")
                    test_results[endpoint] = test_endpoint
                except Exception as e:
                    test_results[endpoint] = e
                    pass
        else:
            return ValueError("No endpoint_handlers found")
        return test_results

    async def test_local_endpoint(self, model, endpoint_list=None):
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        local_endpoints = self.resources["local_endpoints"]
        local_endpoints_by_model = self.resources["local_endpoints"][model]
        endpoint_handlers_by_model = self.resources["endpoint_handler"][model]
        local_endpoints_by_model_by_endpoint = list(local_endpoints_by_model.keys())
        if endpoint_list is not None:
            for endpoint in local_endpoints:
                this_model = endpoint[0]
                this_endpoint = endpoint[1]
                this_data = endpoint[2]
                if this_model[0] == model and this_endpoint == endpoint:
                    filtered_list[this_endpoint] = self.endpoint_handler[this_model][this_endpoint]
        if endpoint_list is not None:
            endpoint_list = [model,endpoint_list,""]
            for endpoint in list(local_endpoints.keys()):
                if endpoint in local_endpoints_by_model_by_endpoint:
                    this_endpoint = endpoint_handlers_by_model[endpoint]
                    filtered_list[endpoint] = local_endpoints[endpoint]            
        else:
            endpoint_list = [model,"",""]
            for endpoint in local_endpoints_by_model:
                if endpoint in local_endpoints_by_model_by_endpoint:
                    this_endpoint = endpoint_handlers_by_model[endpoint]        
                    filtered_list[endpoint] = this_endpoint
        endpoint_handlers = {}
        if len(filtered_list.keys()) > 0:
            for endpoint in filtered_list:
                endpoint_handlers[endpoint] = filtered_list[endpoint]
        else:
            return ValueError("No endpoints found")
        if len(endpoint_handlers) > 0:
            for endpoint in list(endpoint_handlers.keys()):
                try:
                    endpoint_handler = endpoint_handlers[endpoint]
                    test = endpoint_handler("hello world")
                    test_results[endpoint] = test
                except Exception as e:
                    test_results[endpoint] = e
                    pass
        else:
            return ValueError("No endpoint_handlers found")
        return test_results

    async def get_https_endpoint(self, model):
        if model in self.tei_endpoints:
            return self.tei_endpoints[model]
        return None

    async def get_libp2p_endpoint(self, model):
        if model in self.libp2p_endpoints:
            return self.libp2p_endpoints[model]
        return None

    async def request_tei_endpoint(self, model, endpoint=None, endpoint_type=None, batch=None):
        if batch == None:
            incoming_batch_size = 0
        else:
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
    
    async def make_post_request_tei(self, endpoint, data=None):
        if data is None:
            return None
        else:
            pass
        if type(data) is dict:
            if "inputs" not in list(data.keys()):
                data = {"inputs": data}
        if type(data) is list:
            data = {"inputs": data}
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=300) 
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        return ValueError(response)
                    return await response.json()
            except Exception as e:
                print(str(e))
                if "Can not write request body" in str(e):
                    print( "endpoint " + endpoint + " is not accepting requests")
                    return ValueError(e)
                if "Timeout" in str(e):
                    print("Timeout error")
                    return ValueError(e)
                if "Payload is not completed" in str(e):
                    print("Payload is not completed")
                    return ValueError(e)
                if "Can not write request body" in str(e):
                    return ValueError(e)
                pass
            except aiohttp.ClientPayloadError as e:
                print(f"ClientPayloadError: {str(e)}")
                return ValueError(f"ClientPayloadError: {str(e)}")
            except asyncio.TimeoutError as e:
                print(f"Timeout error: {str(e)}")
                return ValueError(f"Timeout error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return ValueError(f"Unexpected error: {str(e)}")

    async def make_post_request_libp2p(self, endpoint, data):
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=300) 
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        return ValueError(response)
                    return await response.json()
            except Exception as e:
                print(str(e))
                if "Can not write request body" in str(e):
                    print( "endpoint " + endpoint + " is not accepting requests")
                    return ValueError(e)
                if "Timeout" in str(e):
                    print("Timeout error")
                    return ValueError(e)
                if "Payload is not completed" in str(e):
                    print("Payload is not completed")
                    return ValueError(e)
                if "Can not write request body" in str(e):
                    return ValueError(e)
                pass
            except aiohttp.ClientPayloadError as e:
                print(f"ClientPayloadError: {str(e)}")
                return ValueError(f"ClientPayloadError: {str(e)}")
            except asyncio.TimeoutError as e:
                print(f"Timeout error: {str(e)}")
                return ValueError(f"Timeout error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return ValueError(f"Unexpected error: {str(e)}")
                    
    async def make_post_request_openvino(self, endpoint, data):
        if type(data) is dict:
            raise ValueError("Data must be a string")
        if type(data) is list:
            if len(data) > 1:
                raise ValueError("batch size must be 1")
            data = data[0]
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=300) 
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        return ValueError(response)
                    return await response.json()
            except Exception as e:
                print(str(e))
                if "Can not write request body" in str(e):
                    print( "endpoint " + endpoint + " is not accepting requests")
                    return ValueError(e)
                if "Timeout" in str(e):
                    print("Timeout error")
                    return ValueError(e)
                if "Payload is not completed" in str(e):
                    print("Payload is not completed")
                    return ValueError(e)
                if "Can not write request body" in str(e):
                    return ValueError(e)
                pass
            except aiohttp.ClientPayloadError as e:
                print(f"ClientPayloadError: {str(e)}")
                return ValueError(f"ClientPayloadError: {str(e)}")
            except asyncio.TimeoutError as e:
                print(f"Timeout error: {str(e)}")
                return ValueError(f"Timeout error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return ValueError(f"Unexpected error: {str(e)}")
 
    async def choose_endpoint(self, model, endpoint_type=None):
        if type(model) is list:
            model = model[0]
        if endpoint_type != None:
            this_endpoints = await self.get_endpoints(model, endpoint_type)
        else:
            tei_endpoints = await self.get_endpoints(model, endpoint_type="tei")
            libp2p_endpoints = await self.get_endpoints(model, endpoint_type="libp2p")
            openvino_endpoints = await self.get_endpoints(model, endpoint_type="openvino")
            local_endpoints = await self.get_endpoints(model, endpoint_type="local")
            filtered_libp2p_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and libp2p_endpoints is not None and k in list(libp2p_endpoints.keys())}
            filtered_tei_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and tei_endpoints is not None and k in list(tei_endpoints.keys())}
            filtered_openvino_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and openvino_endpoints is not None and k in list(openvino_endpoints.keys())}
            filtered_local_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and local_endpoints is not None and k in list(local_endpoints.keys())}
            if not filtered_tei_endpoints and not filtered_libp2p_endpoints and not filtered_openvino_endpoints and not filtered_local_endpoints:
                return None
            else:
                this_endpoint = None
                if len(list(filtered_local_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_local_endpoints.keys()))
                if len(list(filtered_tei_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_tei_endpoints.keys()))
                if len(list(filtered_openvino_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_openvino_endpoints.keys()))
                elif len(list(filtered_libp2p_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_libp2p_endpoints.keys()))
                print("chosen endpoint for " + model + " is " + this_endpoint)
                return this_endpoint

    async def choose_endpoint_new(self, model, endpoint_type=None):
        if type(model) is list:
            model = model[0]
        if endpoint_type != None:
            this_endpoints = await self.get_endpoints_new(model, endpoint_type)
        else:
            tei_endpoints = await self.get_endpoints_new(model, endpoint_type="tei")
            libp2p_endpoints = await self.get_endpoints_new(model, endpoint_type="libp2p")
            openvino_endpoints = await self.get_endpoints_new(model, endpoint_type="openvino")
            local_endpoints = await self.get_endpoints_new(model, endpoint_type="local")
            
            
            filtered_libp2p_endpoints = [x for x in libp2p_endpoints if x[1] in list(self.resources["endpoint_handler"][model].keys())]
            filtered_tei_endpoints = [x for x in tei_endpoints if x[1] in list(self.resources["endpoint_handler"][model].keys())]
            filtered_openvino_endpoints = [x for x in openvino_endpoints if x[1] in list(self.resources["endpoint_handler"][model].keys())]
            filtered_local_endpoints = [x for x in local_endpoints if x[1] in list(self.resources["endpoint_handler"][model].keys())]
            if not filtered_tei_endpoints and not filtered_libp2p_endpoints and not filtered_openvino_endpoints and not filtered_local_endpoints:
                return None
            else:
                this_endpoint = None
                combined_endpoints = filtered_tei_endpoints + filtered_libp2p_endpoints + filtered_openvino_endpoints + filtered_local_endpoints
                random_endpoint = random.choice(combined_endpoints)
                random_endpoint_model = random_endpoint[0]
                random_endpoint_type = random_endpoint[1]
                random_endpoint_handler = self.resources["endpoint_handler"][random_endpoint_model][random_endpoint_type]
                return random_endpoint_handler

    async def status(self):
        new_resources = {}
        included_resources = ["endpoint_handler", "batch_sizes", "queues","hwtest"]
        for resource in included_resources:
            new_resources[resource] = self.resources[resource]
        new_resources["endpoints"] = self.endpoints
        return new_resources
    
    async def infer(self, model, data, endpoint=None, endpoint_type=None):
        infer_results = {}
        if endpoint_type is None:        
            if endpoint is None:
                endpoint = await self.choose_endpoint_new(model, endpoint_type)
                if endpoint is None:
                    return ValueError("No endpoint found")
                else:
                    try:
                        infer_results["infer"] = await endpoint(data)
                    except Exception as e:
                        infer_results["infer"] = endpoint(data)
                        return infer_results
                    return infer_results
            if "cuda" in endpoint or "cpu" in endpoint:
                return await self.make_local_request(model, endpoint, endpoint_type, data)
            elif "openvino" in endpoint:
                return await self.make_post_request_openvino(endpoint, data)
            elif "libp2p" in endpoint:
                return await self.make_post_request_libp2p(endpoint, data)
            elif "http" in endpoint:
                return await self.make_post_request_tei(endpoint, data)
            else:
                return self.endpoint_handler[model][endpoint](data)
        elif endpoint_type == "tei":
            if endpoint is None:
                endpoint = await self.choose_endpoint(model, endpoint_type)
            return await self.make_post_request_tei(endpoint, data)
        elif endpoint_type == "openvino":
            if endpoint is None:
                endpoint = await self.choose_endpoint(model, endpoint_type)
            return await self.make_post_request_openvino(endpoint, data)
        elif endpoint_type == "libp2p":
            if endpoint is None:
                endpoint = await self.choose_endpoint(model, endpoint_type)
            return await self.make_post_request_libp2p(endpoint, data)
        elif endpoint_type == "local":
            if endpoint is None:
                endpoint = await self.choose_endpoint(model, endpoint_type)
            return await self.make_local_request(model, endpoint, endpoint_type, data)
        else:
            endpoint = await self.choose_endpoint(model, endpoint_type)
            if endpoint is None:
                return ValueError("No endpoint found")
            else:
                try:
                    infer_results["infer"] = await endpoint(data)
                except Exception as e:
                    infer_results["infer"] = endpoint(data)
                    return infer_results
                return infer_results

    async def get_endpoints(self, model, endpoint_type=None):
        if endpoint_type == "tei":
            endpoints_dict = self.tei_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "openvino":
            endpoints_dict = self.openvino_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "libp2p":
            endpoints_dict = self.libp2p_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "local":
            endpoints_dict = self.local_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "cuda":
            endpoint_dict = self.local_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoint_dict if "cuda" in endpoint and self.endpoint_status.get(endpoint, 0) >= 1]
        else:
            all_endpoints_dict = self.tei_endpoints.get(model, {}) + self.libp2p_endpoints.get(model, {}) + self.openvino_endpoints.get(model, {}) + self.local_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in all_endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        return filtered_endpoints
    
    
    async def get_endpoints_new(self, model, endpoint_type=None):
        filtered_endpoints = []
        endpoints_keys = list(self.endpoints.keys())
        if endpoint_type == "tei" and "tei_endpoints" in endpoints_keys:
            endpoints_dict = self.endpoints["tei_endpoints"].get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict ]
        elif endpoint_type == "openvino" and "openvino_endpoints" in list(self.endpoints.keys()):
            endpoints_dict = self.endpoints["openvino_endpoints"].get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict ]
        elif endpoint_type == "libp2p" and "libp2p_endpoints" in list(self.endpoints.keys()):
            endpoints_dict = self.libp2p_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict ]
        elif endpoint_type == "local" and "local_endpoints" in list(self.endpoints.keys()):
            endpoints_dict = self.endpoints["local_endpoints"].get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict ]
        elif endpoint_type == "cuda" and "local_endpoints" in list(self.endpoints.keys()):
            endpoint_dict = self.endpoints["local_endpoints"].get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoint_dict if "cuda" in endpoint and self.endpoint_status.get(endpoint, 0) >= 1]
        elif endpoint_type == "all" or endpoint_type == None:
            all_endpoints = []
            if "tei_endpoints" in list(self.endpoints.keys()):
                tei_endpoints = self.endpoints["tei_endpoints"].get(model, {})
                all_endpoints = all_endpoints + tei_endpoints
            if "openvino_endpoints" in list(self.endpoints.keys()):
                openvino_endpoints = self.endpoints["openvino_endpoints"].get(model, {})
                all_endpoints = all_endpoints + openvino_endpoints 
            if "libp2p_endpoints" in list(self.endpoints.keys()):
                libp2p_endpoints = self.libp2p_endpoints.get(model, {})
                all_endpoints = all_endpoints + libp2p_endpoints
            if "local_endpoints" in list(self.endpoints.keys()):
                local_endpoints = self.endpoints["local_endpoints"].get(model, {})
                all_endpoints = all_endpoints + local_endpoints
            filtered_endpoints = [endpoint for endpoint in all_endpoints]
        return filtered_endpoints
    
    async def async_generator(self, iterable):
        for item in iterable:
            yield item
    
    async def __test__(self, resources, metadata):
        results = {}
        ipfs_accelerate_init = await self.init_endpoints( metadata['models'], resources)
        test_endpoints = await self.test_endpoints(metadata['models'], ipfs_accelerate_init)
        results = {"ipfs_accelerate_init": ipfs_accelerate_init,"test_endpoints": test_endpoints}
        return results

ipfs_accelerate_py = ipfs_accelerate_py

if __name__ == "__main__":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "namespace": "TeraflopAI/Caselaw_Access_Project",
        "column": "text",
        "split": "train",
        "models": [
            "thenlper/gte-small",
            # "Alibaba-NLP/gte-large-en-v1.5",
            # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        ],
        "chunk_settings": {
            "chunk_size": 512,
            "n_sentences": 8,
            "step_size": 256,
            "method": "fixed",
            "embed_model": "thenlper/gte-small",
            "tokenizer": None
        },
        "dst_path": "/storage/teraflopai/tmp",
    }
    resources = {
        "local_endpoints": [
            ["thenlper/gte-small", "cpu", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "cpu", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cpu", 32768],
            ["thenlper/gte-small", "cuda:0", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "cuda:0", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cuda:0", 32768],
            ["thenlper/gte-small", "cuda:1", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "cuda:1", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cuda:1", 32768],
            ["thenlper/gte-small", "openvino", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "openvino", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "openvino", 32768],
            ["thenlper/gte-small", "llama_cpp", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "llama_cpp", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "llama_cpp", 32768],
            ["thenlper/gte-small", "ipex", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "ipex", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "ipex", 32768],
        ],
        "openvino_endpoints": [
            # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
            # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
            # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
            # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx0-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx0/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx1-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx1/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx2-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx2/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx3-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx3/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx4-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx4/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx5-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx5/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx6-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx6/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx7-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx7/infer", 1024]
        ],
        "tei_endpoints": [
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8080/embed-medium", 32768],
            ["thenlper/gte-small", "http://62.146.169.111:8080/embed-tiny", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8081/embed-small", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8081/embed-medium", 32768],
            ["thenlper/gte-small", "http://62.146.169.111:8081/embed-tiny", 512],
            # ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8082/embed-small", 8192],
            # ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8082/embed-medium", 32768],
            # ["thenlper/gte-small", "http://62.146.169.111:8082/embed-tiny", 512],
            # ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8083/embed-small", 8192],
            # ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8083/embed-medium", 32768],
            # ["thenlper/gte-small", "http://62.146.169.111:8083/embed-tiny", 512]
        ]
    }
    ipfs_accelerate_py = ipfs_accelerate_py(resources, metadata)
    asyncio.run(ipfs_accelerate_py.__test__(resources, metadata))
    print("test complete")
