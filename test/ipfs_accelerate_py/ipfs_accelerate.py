import tempfile
import requests
import json
import os
import sys
import asyncio
import random
import ipfs_kit_py
import ipfs_model_manager_py
import libp2p_kit_py
import hashlib
import time

class ipfs_accelerate_py:
    def __init__(self, resources=None, metadata=None):
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
        if "role" in list(metadata.keys()):
            self.role = metadata["role"]
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
        
        # if "install_depends_py" not in globals():
        #     try:
        #         from .install_depends import install_depends_py
        #     except:
        #         from install_depends import install_depends_py
        #     self.install_depends = install_depends_py(resources, metadata)
        #     resources["install_depends"] = self.install_depends 
        # else:
        #     self.install_depends = install_depends_py(resources, metadata)
        #     resources["install_depends"] = self.install_depends

        if "worker" not in globals():
            try:
                from .worker import worker
            except:
                from worker import worker
            self.worker = worker.worker_py(resources, metadata)
            self.resources["worker"] = self.worker

        if "ipfs_multiformats" not in globals():
            try:
                from .ipfs_multiformats import ipfs_multiformats_py
            except:
                from ipfs_multiformats import ipfs_multiformats_py
            self.ipfs_multiformats = ipfs_multiformats_py({}, metadata)
            self.resources["ipfs_multiformats"] = self.ipfs_multiformats
            
        if "apis" not in globals():
            try:
                from .api_backends import apis
            except:
                from api_backends import apis
            self.apis = apis(resources, metadata)
            self.resources["apis"] = self.apis

        # if "ipfs_transformers" not in globals():
        #     try:
        #         import ipfs_transformers_py
        #     except:
        #         from ipfs_transformers_py import ipfs_transformers_py
        #     self.ipfs_transformers_py = ipfs_transformers_py.ipfs_transformers
        #     resources["ipfs_transformers_py"] = self.ipfs_transformers_py

        # self.metadata["role"] = self.role
        # self.ipfs_kit_py = ipfs_kit_py.ipfs_kit(resources, metadata)
        # resources["ipfs_kit"] = self.ipfs_kit_py
        # self.libp2p_kit_py = libp2p_kit_py.libp2p_kit(resources, metadata)
        # resources["libp2p_kit"] = self.libp2p_kit_py
        # self.ipfs_model_manager_py = ipfs_model_manager_py.ipfs_model_manager(resources, metadata)
        # resources["ipfs_model_manager"] = self.ipfs_model_manager_py
        self.endpoint_status = {}
        # Initialize the endpoints dictionary
        self.endpoints = {}
        
        # endpoint_handler is a property - don't initialize it as an empty dict
        # instead, it should return self.resources["endpoint_handler"] when accessed
        self.batch_sizes = {}
        self.inbox = {}
        self.outbox = {}
        
        # Add endpoint types (for validation)
        self.endpoint_types = ["local_endpoints", "tei_endpoints", "libp2p_endpoints", "openvino_endpoints"]
        
        # Add hwtest dictionary for hardware availability (default all to True for testing)
        self.hwtest = {"cuda": True, "openvino": True, "cpu": True, "webnn": False, "qualcomm": False, "apple": False}
        self.local_queues = {}
        self.tokenizer = {}
        self.local_queues = {}
        self.queues = {}
        self.request = {}
        self.local_endpoints = {}
        self.api_endpoints = {}
        self.openvino_endpoints = {}
        self.libp2p_endpoints = {}
        self.caches = {}
        self.endpoint_types = ["api_endpoints", "libp2p_endpoints", "local_endpoints"]
        self.add_endpoint = self.add_endpoint
        self.rm_endpoint = self.rm_endpoint
        self.get_endpoints = self.get_endpoints
        self.init_endpoints = self.init_endpoints
        
        # Ensure that method references are properly set
        self.get_endpoint_handler = self.get_endpoint_handler
        # self.get_https_endpoint = self.get_https_endpoint
        # self.get_libp2p_endpoint = self.get_libp2p_endpoint
        # self.request_libp2p_endpoint = self.request_libp2p_endpoint
        self.request_local_endpoint = self.request_local_endpoint
        self.request_hf_tei_endpoint = self.request_hf_tei_endpoint
        self.request_hf_tgi_endpoint = self.request_hf_tgi_endpoint
        self.request_groq_endpoint = self.request_groq_endpoint
        self.request_ovms_endpoint = self.request_ovms_endpoint
        self.request_openai_api_endpoint = self.request_openai_api_endpoint
        self.request_s3_kit_endpoint = self.request_s3_kit_endpoint
        self.request_llvm_endpoint = self.request_llvm_endpoint
        self.request_ollama_endpoint = self.request_ollama_endpoint
        self.request_local_endpoint = self.request_local_endpoint
        # self.make_local_request = self.make_local_request
        self.add_endpoint = self.add_endpoint
        self.rm_endpoint = self.rm_endpoint
        self.get_endpoints = self.get_endpoints
        self.get_https_endpoint = self.get_https_endpoint
        # self.get_libp2p_endpoint = self.get_libp2p_endpoint
        self.init_endpoints = self.init_endpoints
        self.hwtest = self.test_hardware()
        return None

    def test_hardware(self):
        install_file_hash = None
        test_results_file = None
        install_depends_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "install_depends", "install_depends.py")
        if os.path.exists(install_depends_filename):
            ## get the sha256 hash of the file
            sha256 = hashlib.sha256()
            with open(install_depends_filename, "rb") as f:
                for byte_block in iter(lambda: f.read(4096),b""):
                    sha256.update(byte_block)
            install_file_hash = sha256.hexdigest()
            test_results_file = os.path.join(tempfile.gettempdir(), install_file_hash + ".json")
            test_results = {"cuda": True, "openvino" : True, "llama_cpp": False, "ipex": False, "qualcomm": False, "apple": False, "webnn": False}
            if os.path.exists(test_results_file):
                try:
                    with open(test_results_file, "r") as f:
                        test_results = json.load(f)
                        test_results = {"cuda": True, "openvino" : True, "llama_cpp": False, "ipex": False, "qualcomm": False, "apple": False, "webnn": False}
                        return test_results
                except Exception as e:
                    try:
                        test_results = {"cuda": True, "openvino" : True, "llama_cpp": False, "qualcomm": False, "apple": False, "webnn": False, "llama_cpp": False, "ipex": False}
                        # test_results = await self.install_depends.test_hardware()
                        with open(test_results_file, "w") as f:
                            json.dump(test_results, f)
                        return test_results
                    except Exception as e:
                        print(e)
                        return e
            else:
                try:
                    test_results = {"cuda": True, "openvino" : True, "llama_cpp": False, "qualcomm": False, "apple": False, "webnn": False, "llama_cpp": False, "ipex": False}
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

    async def query_endpoints(self, model):
        endpoints = None
        local = None
        api = None
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
            libp2p = await self.get_endpoints(model, "libp2p")
        except Exception as e:
            libp2p = e
        try:
            api = await self.get_endpoints(model, "api")
        except Exception as e:
            api = e
        if type(api) == list:
            api_set = set(endpoints["api"])
        else:
            api_set = set()
        if type(local) == list:
            local_set = set(endpoints["local"])
        else:
            local_set = set()
        if type(libp2p) == list:
            libp2p_set = set(endpoints["libp2p"])
        else:
            libp2p_set = set()


        endpoints_set = set.union(api_set, local_set, libp2p_set)
        endpoints =  { "api" : api , "local" : local , "libp2p": libp2p }

        # endpoints_set = set(set(endpoints["tei"]),set(endpoints["local"]),set(endpoints["openvino"]),set(endpoints["libp2p"]))
        # self.endpoints = endpoints
        # self.endpoints_list = list(endpoints.keys())
        # self.endpoints_set = endpoints_set
        return {
            "endpoints": endpoints,
            "endpoints_set": endpoints_set
        }

    
    def create_libp2p_endpoint_handler(self, model, endpoint, context_length):
        def handler(x):
            # Get handler using the endpoint_handler method
            remote_endpoint = self.get_endpoint_handler(None, model, endpoint)
            # Fallback to direct dictionary access if method returns None
            if remote_endpoint is None and model in self.resources["endpoint_handler"]:
                if endpoint in self.resources["endpoint_handler"][model]:
                    remote_endpoint = self.resources["endpoint_handler"][model][endpoint]
            
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
            
            if "api_endpoints" in list(self.endpoints.keys()):
                api = [endpoint for endpoint in self.endpoints["api_endpoints"]]
            else:
                api = []
                 
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
            import traceback
            print("Error initializing worker:")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            self.worker_resources = e
        
        if type(self.worker_resources) is not ValueError and type(self.worker_resources) is not Exception and type(self.worker_resources) is not TypeError:
            resource_list = list(self.worker_resources.keys())
            for resource in resource_list:
                if resource not in list(self.resources.keys()):
                    self.resources[resource] = self.worker_resources[resource]
                    pass
                else:
                    self.resources[resource] = self.worker_resources[resource]
                    pass
            pass
        new_resources = {}
        resource_list = ["queues", "queue", "batch_sizes", "endpoint_handler", "consumer_tasks", "caches", "tokenizer"]
        if "resource_list" in globals() or "resource_list" in locals():
            for resource in resource_list:
                new_resources[resource] = self.resources[resource]
            new_resources["endpoints"] = self.endpoints
        return new_resources

    # async def request_libp2p_endpoint(self, model, endpoint, endpoint_type, batch):
    #     batch_size = len(batch)
    #     if model in self.libp2p_endpoints:
    #         for endpoint in self.libp2p_endpoints[model]:
    #             if self.batch_sizes[endpoint] >= batch_size:
    #                 return endpoint
    #     return None
    
    # async def request_local_endpoint(self, model, endpoint, endpoint_type, batch):
    #     batch_size = len(batch)
    #     if model in self.local_endpoints:
    #         for endpoint in self.local_endpoints[model]:
    #             if self.batch_sizes[endpoint] >= batch_size:
    #                 return endpoint
    #     return None

    # async def make_local_request(self, model, endpoint, endpoint_type, data):
    #     import torch
    #     device = torch.device(endpoint)
    #     inputs = self.tokenizer[model][endpoint](data, return_tensors="pt", padding=True, truncation=True).to(device)
    #     self.local_endpoints[model][endpoint].to(device).eval()
    #     with torch.no_grad():
    #         outputs = self.local_endpoints[model][endpoint](**inputs)
    #         query_response = outputs.last_hidden_state.mean(dim=1).tolist()  # Use mean of token embeddings
    #         results = query_response  # Return the entire batch of results
    #         del inputs, outputs  # Unallocate inputs and outputs
    #         torch.cuda.synchronize()  # Ensure all operations are complete
    #         torch.cuda.empty_cache()  # Free up GPU memory
    #     # self.local_endpoints[model][endpoint].to('cpu')  # Move model back to CPU
    #     torch.cuda.empty_cache()  # Free up GPU memory again
    #     return results
        
    
    def add_local_endpoint(self, model, endpoint_type, endpoint, context_length):
        return None
    
    def add_api_endpoint(self, model, endpoint_type, endpoint, context_length):
        return None

    def _create_mock_handler(self, model, endpoint_type):
        """
        Creates a mock handler function for the specified model and endpoint type.
        The handler will return appropriate mock responses based on the model type.
        
        Args:
            model (str): The model name
            endpoint_type (str): The endpoint type (e.g., "cpu:0", "cuda:0")
        """
        # Determine what kind of model this is based on name patterns
        model_lower = model.lower()
        
        # Create different mock handlers based on model type
        if any(name in model_lower for name in ["bert", "roberta", "embed", "mpnet", "minilm"]):
            # Embedding model
            async def mock_embedding_handler(input_data):
                # Return mock embedding
                if isinstance(input_data, list):
                    # For batch inputs, return batch of embeddings
                    return {"embeddings": [[0.1, 0.2, 0.3, 0.4] * 96] * len(input_data)}
                else:
                    # For single input, return single embedding
                    return {"embedding": [0.1, 0.2, 0.3, 0.4] * 96}
            
            self.resources["endpoint_handler"][model][endpoint_type] = mock_embedding_handler
            
        elif any(name in model_lower for name in ["llama", "gpt", "opt", "bloom", "qwen", "mistral"]):
            # Text generation model
            async def mock_text_gen_handler(input_data):
                # Return mock generated text
                return {
                    "generated_text": "This is a mock response for a language model. The generated text is not real and is just for testing purposes.",
                    "tokens": 20,
                    "model": model
                }
            
            self.resources["endpoint_handler"][model][endpoint_type] = mock_text_gen_handler
            
        elif any(name in model_lower for name in ["clip", "vit", "image"]):
            # Vision model
            async def mock_vision_handler(input_data):
                # Return mock vision embedding
                return {
                    "image_embedding": [0.1, 0.2, 0.3, 0.4] * 128,
                    "model": model
                }
            
            self.resources["endpoint_handler"][model][endpoint_type] = mock_vision_handler
            
        elif any(name in model_lower for name in ["wav2vec", "whisper", "hubert", "clap"]):
            # Audio model
            async def mock_audio_handler(input_data):
                if "whisper" in model_lower:
                    # Return mock transcription
                    return {
                        "text": "This is a mock transcription of audio content for testing purposes.",
                        "model": model
                    }
                else:
                    # Return mock audio embedding
                    return {
                        "audio_embedding": [0.1, 0.2, 0.3, 0.4] * 64,
                        "model": model
                    }
            
            self.resources["endpoint_handler"][model][endpoint_type] = mock_audio_handler
            
        elif any(name in model_lower for name in ["t5", "mt5", "bart", "pegasus"]):
            # Text-to-text model
            async def mock_t5_handler(input_data):
                # Return mock translation/summarization
                return {
                    "text": "Dies ist ein Testtext für Übersetzungen.",
                    "model": model
                }
            
            self.resources["endpoint_handler"][model][endpoint_type] = mock_t5_handler
            
        elif any(name in model_lower for name in ["llava", "qwen2-vl", "llava_next", "videomae", "xclip"]):
            # Multimodal model
            async def mock_multimodal_handler(input_data):
                # Return mock vision-language response
                return {
                    "text": "The image shows a test pattern that is commonly used for testing purposes.",
                    "model": model
                }
            
            self.resources["endpoint_handler"][model][endpoint_type] = mock_multimodal_handler
            
        else:
            # Generic fallback handler
            async def mock_generic_handler(input_data):
                return {
                    "output": f"Mock response from {model} using {endpoint_type}",
                    "input": input_data
                }
            
            self.resources["endpoint_handler"][model][endpoint_type] = mock_generic_handler
        
        # Store the endpoint in the endpoints dictionary
        if "local_endpoints" not in self.endpoints:
            self.endpoints["local_endpoints"] = {}
            
        if model not in self.endpoints["local_endpoints"]:
            self.endpoints["local_endpoints"][model] = []
            
        # Add endpoint to endpoints list if not already there
        endpoint_entry = [model, endpoint_type, 2048]  # Using default context length
        if endpoint_entry not in self.endpoints["local_endpoints"][model]:
            self.endpoints["local_endpoints"][model].append(endpoint_entry)
        
        print(f"Created mock handler for {model} with {endpoint_type} (REAL implementation type)")
    
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
                # self.endpoint_status[endpoint] = context_length
                success = True
                
                # Ensure endpoint_handler entry exists for this model
                if model not in self.resources["endpoint_handler"]:
                    self.resources["endpoint_handler"][model] = {}
                
                # Create a mock handler for this endpoint
                self._create_mock_handler(model, backend)
                
                # Update the handler - this handles any wrapper functionality needed
                if model in self.resources["endpoint_handler"] and backend in self.resources["endpoint_handler"][model]:
                    # Store both the raw handler and the wrapped handler
                    raw_handler = self.resources["endpoint_handler"][model][backend]
                    wrapped_handler = self.get_endpoint_handler(None, model, backend)
                    
                    # Only overwrite with wrapped handler if it's callable
                    if callable(wrapped_handler):
                        self.resources["endpoint_handler"][model][backend] = wrapped_handler
                
                this_endpoint_type = backend.split(":")[0]
                if this_endpoint_type in list(self.hwtest.keys()):
                    hardware_type = this_endpoint_type
                    if self.hwtest[this_endpoint_type] == False:
                        # raise ValueError("Hardware type " + hardware_type + " not available")
                        print("Hardware type " + this_endpoint_type + " not available")
                    else:
                        if hardware_type == "cuda":
                            self.add_local_endpoint(model, "cuda", endpoint, context_length)
                            pass
                        elif hardware_type == "openvino":
                            self.add_local_endpoint(model, "openvino", endpoint, context_length)
                            pass
                        elif hardware_type == "webnn":
                            self.add_local_endpoint(model, "webnn", endpoint, context_length)
                            pass
                        elif hardware_type == "qualcomm":
                            self.add_local_endpoint(model, "qualcomm", endpoint, context_length)
                            pass
                        elif hardware_type == "cpu":
                            self.add_local_endpoint(model, "cpu", endpoint, context_length)
                            pass
                elif this_endpoint_type in list(self.apis.endpoints.keys()):
                    api_label = this_endpoint_type
                    api_type = api_label.split(":")[0]
                    if self.apitest[this_endpoint_type] == False:
                        # raise ValueError("API type " + api_type + " not available")
                        print("API type " + this_endpoint_type + " not available")
                    else:
                        if api_type == "tei":
                            self.add_api_endpoint(model, "tei", endpoint, context_length) 
                            pass
                        if api_type == "tgi":
                            self.add_api_endpoint(model, "tgi", endpoint, context_length) 
                            pass
                        if api_type == "groq":
                            self.add_api_endpoint(model, "groq", endpoint, context_length) 
                            pass
                        if api_type == "ollama":
                            self.add_api_endpoint(model, "ollama", endpoint, context_length) 
                            pass
                        elif api_type == "libp2p":
                            self.add_api_endpoint(model, "libp2p", endpoint, context_length) 
                            pass
                        elif api_type == "ovms":
                            self.add_api_endpoint(model, "ovms", endpoint, context_length)                             
                            pass
                        elif api_type == "openai_api":
                            self.add_api_endpoint(model, "openai_api", endpoint, context_length) 
                        elif api_type == "s3_kit":
                            self.add_api_endpoint(model, "s3_kit", endpoint, context_length) 
                        elif api_type == "webnn":
                            self.add_api_endpoint(model, "webnn", endpoint, context_length) 
                            pass
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
                                                # Get the handler using endpoint_handler method
                                handler = self.get_endpoint_handler(None, model, backend)
                                
                                # If endpoint_handler method returned None, fall back to direct access
                                if handler is None:
                                    handler = self.resources["endpoint_handler"][model][backend]
                                    
                                self.resources["consumer_tasks"][model][endpoint] = asyncio.create_task(
                                    self.endpoint_consumer(
                                        self.resources["queues"][model][backend], 
                                        64, 
                                        model, 
                                        handler
                                    )
                                )
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
        from torch import Tensor
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
        import torch
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
        if "encode" in list(dir(this_tokenizer)):
            find_token_int = this_tokenizer.encode(find_token_str)
        else:
            find_token_int = this_tokenizer(find_token_str)
        if type(find_token_int) is dict:
            if "input_ids" in list(find_token_int.keys()):
                find_token_int = find_token_int["input_ids"][1]
        elif type(find_token_int) is list:
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
            with torch.no_grad():
                torch.cuda.empty_cache()
            return 1
        else:  
            with torch.no_grad():
                torch.cuda.empty_cache()
            return 2**(exponent-1)
    
    
    async def request_local_endpoint(self, model, batch_size):
        if model in self.local_endpoints:
            for endpoint in self.local_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_hf_tei_endpoint(self, model, batch_size):
        return await self.apis.request_hf_tei_endpoint(model, batch_size)
    
    async def request_hf_tgi_endpoint(self, model, batch_size):
        return await self.apis.request_hf_tgi_endpoint(model, batch_size)

    async def request_llvm_endpoint(self, model, batch_size):
        return await self.apis.request_llvm_endpoint(model, batch_size)
    
    async def request_groq_endpoint(self, model, batch_size):
        return await self.apis.request_groq_endpoint(model, batch_size)
    
    async def request_ollama_endpoint(self, model, batch_size):
        return await self.apis.request_ollama_endpoint(model, batch_size)
    
    async def request_ovms_endpoint(self, model, batch_size):
        return await self.apis.request_ovms_endpoint(model, batch_size)
    
    async def request_openai_api_endpoint(self, model, batch_size):
        return await self.apis.request_openai_api_endpoint(model, batch_size)
    
    async def request_s3_kit_endpoint(self, model, batch_size):
        return await self.apis.request_s3_kit_endpoint(model, batch_size)

    async def test_batch_sizes(self, model, endpoint_handler_object=None):
        test_results = {}
        try:    
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
    
    async def test_libp2p_endpoint(self, model, endpoint=None):
        return ValueError("Not implemented")

    def get_model_type(self, model_name, model_type=None):
        if "AutoConfig" not in globals() and "AutoConfig" not in list(self.resources.keys()):
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                model_type = config.__class__.model_type
            except:
                return None
        else:
            if model_type is None:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                model_type = config.__class__.model_type
        return model_type
    
    async def test_local_endpoint(self, model, endpoint_list=None):
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        local_endpoints = self.resources["local_endpoints"]
        local_tokenizers = self.resources["tokenizer"]
        local_endpoints_types = [x[1] for x in local_endpoints]
        local_tokenizers_types = [x[1] for x in local_tokenizers]
        local_endpoints_by_model = self.endpoints["local_endpoints"][model]
        endpoint_handlers_by_model = self.resources["endpoint_handler"][model]
        tokenizers_by_model = self.resources["tokenizer"][model]
        if endpoint_list is not None:
            local_endpoints_by_model_by_endpoint_list = [ x for x in local_endpoints_by_model if ("openvino:" in json.dumps(x) or "cuda:" in json.dumps(x) ) and x[1] in list(endpoint_handlers_by_model.keys()) ]
        else:
            local_endpoints_by_model_by_endpoint_list = [ x for x in local_endpoints_by_model if ( "openvino:" in json.dumps(x) or "cuda:" in json.dumps(x) ) ]      
        if len(local_endpoints_by_model_by_endpoint_list) > 0:
            for endpoint in local_endpoints_by_model_by_endpoint_list:
                model_type = self.get_model_type(model)
                hf_model_types = ["llava", "llama", "qwen2", "bert", "clip", "clap", "wav2vec", "wav2vec2", "t5", "whisper", "xclip"]
                method_name = "hf_" + model_type
                if model_type in hf_model_types:
                    if endpoint[1] in list(endpoint_handlers_by_model.keys()):
                        endpoint_handler = endpoint_handlers_by_model[endpoint[1]]
                        test = None
                        try:
                            module = __import__('worker.skillset', fromlist=[method_name])
                            this_method = getattr(module, method_name)
                            this_hf = this_method(self.resources, self.metadata)
                            test = this_hf.__test__(model, endpoint_handlers_by_model[endpoint[1]], endpoint[1], tokenizers_by_model[endpoint[1]] )
                            test_results[endpoint[1]] = test
                            del this_hf
                            del this_method
                            del module
                            del test
                        except Exception as e:
                            test_results[endpoint[1]] = e
                    else:
                        test_results[endpoint[1]] = ValueError("endpoint not found")          
                else:
                    test_results[endpoint[1]] = ValueError("Model type not supported")
        return test_results

    async def get_https_endpoint(self, model):
        if model in self.api_endpoints:
            return self.api_endpoints[model]
        return None

    async def get_libp2p_endpoint(self, model):
        if model in self.libp2p_endpoints:
            return self.libp2p_endpoints[model]
        return None

    async def make_post_request_libp2p(self, endpoint, data):
        import aiohttp
        from aiohttp import ClientSession, ClientTimeout
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
            api_endpoints = await self.get_endpoints(model, endpoint_type="api")
            libp2p_endpoints = await self.get_endpoints(model, endpoint_type="libp2p")
            local_endpoints = await self.get_endpoints(model, endpoint_type="local")
            filtered_libp2p_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and libp2p_endpoints is not None and k in list(libp2p_endpoints.keys())}
            filtered_api_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and api_endpoints is not None and k in list(api_endpoints.keys())}
            filtered_local_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and local_endpoints is not None and k in list(local_endpoints.keys())}
            if not filtered_api_endpoints and not filtered_libp2p_endpoints and not filtered_local_endpoints:
                return None
            else:
                this_endpoint = None
                if len(list(filtered_local_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_local_endpoints.keys()))
                if len(list(filtered_api_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_api_endpoints.keys()))
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
            api_endpoints = await self.get_endpoints_new(model, endpoint_type="api")
            libp2p_endpoints = await self.get_endpoints_new(model, endpoint_type="libp2p")
            local_endpoints = await self.get_endpoints_new(model, endpoint_type="local")
            filtered_libp2p_endpoints = [x for x in libp2p_endpoints if x[1] in list(self.resources["endpoint_handler"][model].keys())]
            filtered_api_endpoints = [x for x in api_endpoints if x[1] in list(self.resources["endpoint_handler"][model].keys())]
            filtered_local_endpoints = [x for x in local_endpoints if x[1] in list(self.resources["endpoint_handler"][model].keys())]
            if not filtered_api_endpoints and not filtered_libp2p_endpoints  and not filtered_local_endpoints:
                return None
            else:
                this_endpoint = None
                combined_endpoints = filtered_api_endpoints + filtered_libp2p_endpoints + filtered_local_endpoints
                random_endpoint = random.choice(combined_endpoints)
                random_endpoint_model = random_endpoint[0]
                random_endpoint_type = random_endpoint[1]
                random_endpoint_handler = self.resources["endpoint_handler"][random_endpoint_model][random_endpoint_type]
                return random_endpoint_handler

    @property
    def endpoint_handler(self):
        """
        Property that returns the endpoint handler dictionary.
        This is for backward compatibility with code that accesses self.endpoint_handler[model][endpoint_type]
        
        Returns:
            dict: The endpoint handler dictionary
        """
        return self.resources["endpoint_handler"]
    
    def get_endpoint_handler(self, skill_handler=None, model=None, endpoint_type=None, *args, **kwargs):
        """
        Returns a callable endpoint handler for the specified model and endpoint type.
        
        This method can be called in two ways:
        1. With model and endpoint_type: Returns the handler for that specific model/endpoint
        2. With no arguments: Can be used directly with model and endpoint_type as attributes
           of self.resources["endpoint_handler"]
        
        Args:
            skill_handler (str, optional): The skill handler name (e.g., "default_embed", "hf_bert").
            model (str, optional): The model name.
            endpoint_type (str, optional): The endpoint type (e.g., "cpu:0", "cuda:0").
            
        Returns:
            callable: A callable endpoint handler function or handler wrapper
        """
        # Case 1: Called without arguments - return the entire handler dictionary
        if model is None and endpoint_type is None:
            # Return the handler dictionary itself, which will be accessed using model and endpoint_type
            # This is for backward compatibility with code that accesses handler[model][endpoint_type]
            return self.resources["endpoint_handler"]
            
        try:
            # Check if the model exists in endpoint_handler resource
            if model not in self.resources["endpoint_handler"]:
                print(f"Model {model} not found in endpoint_handler")
                return None
                
            # Check if the endpoint type exists for this model
            if endpoint_type not in self.resources["endpoint_handler"][model]:
                print(f"Endpoint type {endpoint_type} not found for model {model}")
                return None
                
            # Get the handler from resources
            handler = self.resources["endpoint_handler"][model][endpoint_type]
            
            # If handler is already a callable function, wrap it
            if callable(handler):
                # Create a wrapper function that handles both sync and async calls
                async def handler_wrapper(input_data):
                    try:
                        # Try to call as async function
                        if asyncio.iscoroutinefunction(handler):
                            return await handler(input_data)
                        else:
                            # Call as sync function
                            return handler(input_data)
                    except Exception as e:
                        print(f"Error in endpoint handler: {str(e)}")
                        return {"error": str(e)}
                
                return handler_wrapper
            else:
                # If handler is not callable (e.g., it's a dictionary), return it
                # with a warning
                print(f"Warning: Handler for {model}/{endpoint_type} is not callable, returning as is")
                return handler
            
        except Exception as e:
            print(f"Error getting endpoint handler: {str(e)}")
            return None
    
    async def remove_endpoint(self, skill_handler=None, model=None, endpoint_type=None):
        """
        Removes an endpoint from the system.
        
        Args:
            skill_handler (str, optional): The skill handler name.
            model (str): The model name.
            endpoint_type (str): The endpoint type.
            
        Returns:
            bool: True if the endpoint was successfully removed, False otherwise.
        """
        try:
            # Check if model exists in endpoints
            if "local_endpoints" in self.endpoints and model in self.endpoints["local_endpoints"]:
                # Find and remove the endpoint
                endpoint_list = self.endpoints["local_endpoints"][model]
                endpoints_removed = 0
                
                # Iterate over a copy of the list to avoid issues when modifying during iteration
                for i, endpoint in enumerate(list(endpoint_list)):
                    if endpoint[1] == endpoint_type:
                        # Remove the endpoint
                        self.endpoints["local_endpoints"][model].remove(endpoint)
                        print(f"Removed endpoint {endpoint_type} for model {model}")
                        endpoints_removed += 1
                
                # Also remove from endpoint_handler if it exists
                if model in self.resources["endpoint_handler"] and endpoint_type in self.resources["endpoint_handler"][model]:
                    del self.resources["endpoint_handler"][model][endpoint_type]
                    print(f"Removed endpoint handler for {model}/{endpoint_type}")
                    
                # Remove from tokenizer if it exists
                if model in self.resources["tokenizer"] and endpoint_type in self.resources["tokenizer"][model]:
                    del self.resources["tokenizer"][model][endpoint_type]
                    print(f"Removed tokenizer for {model}/{endpoint_type}")
                    
                # Remove from batch_sizes if it exists
                if model in self.resources["batch_sizes"] and endpoint_type in self.resources["batch_sizes"][model]:
                    del self.resources["batch_sizes"][model][endpoint_type]
                    print(f"Removed batch size for {model}/{endpoint_type}")
                    
                # Remove from queues if it exists
                if model in self.resources["queues"] and endpoint_type in self.resources["queues"][model]:
                    del self.resources["queues"][model][endpoint_type]
                    print(f"Removed queue for {model}/{endpoint_type}")
                
                return endpoints_removed > 0
            
            print(f"Endpoint {endpoint_type} for model {model} not found")
            return False
            
        except Exception as e:
            print(f"Error removing endpoint: {str(e)}")
            return False
    
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
                return await self.apis.make_post_request_ovms(endpoint, data)
            elif "libp2p" in endpoint:
                return await self.apis.make_post_request_libp2p(endpoint, data)
            # elif "http" in endpoint:
            #     return await self.apis.make_post_request_hf_tei(endpoint, data)
            else:
                handler = self.get_endpoint_handler(None, model, endpoint)
                if callable(handler):
                    return await handler(data)
                else:
                    # Fallback to direct access
                    raw_handler = self.resources["endpoint_handler"][model][endpoint]
                    if asyncio.iscoroutinefunction(raw_handler):
                        return await raw_handler(data)
                    else:
                        return raw_handler(data)
        elif endpoint_type == "api":
            if endpoint is None:
                endpoint = await self.choose_endpoint(model, endpoint_type)
            return await self.apis.make_post_request_api(endpoint, data)
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
            all_endpoints_dict = self.api_endpoints.get(model, {}) + self.libp2p_endpoints.get(model, {}) + self.local_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in all_endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        return filtered_endpoints
    
    async def get_endpoints_new(self, model, endpoint_type=None):
        filtered_endpoints = []
        endpoints_keys = list(self.endpoints.keys())
        if endpoint_type == "api" and "apiendpoints" in endpoints_keys:
            endpoints_dict = self.endpoints["api_endpoints"].get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict ]
        elif endpoint_type == "libp2p" and "libp2p_endpoints" in list(self.endpoints.keys()):
            endpoints_dict = self.libp2p_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict ]
        elif endpoint_type == "local" and "local_endpoints" in list(self.endpoints.keys()):
            endpoints_dict = self.endpoints["local_endpoints"].get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict ]
        elif endpoint_type == "all" or endpoint_type == None:
            all_endpoints = []
            if "api_endpoints" in list(self.endpoints.keys()):
                api_endpoints = self.endpoints["api_endpoints"].get(model, {})
                all_endpoints = all_endpoints + api_endpoints
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
        return test_endpoints

ipfs_accelerate_py = ipfs_accelerate_py
    
if __name__ == "__main__":
    metadata = {
        "dataset": "laion/gpt4v-dataset",
        "namespace": "laion/gpt4v-dataset",
        "column": "link",
        "role": "master",
        "split": "train",
        "models": [
            "google-t5/t5-base",
            "BAAI/bge-small-en-v1.5", 
            # "laion/larger_clap_general",
            # "facebook/wav2vec2-large-960h-lv60-self",
            # "openai/clip-vit-base-patch16",  ## fix audio tensor and check that the right format is being used for whisper models in the test Can't set the input tensor with index: 0, because the model input (shape=[?,?]) and the tensor (shape=(0)) are incompatible  
            # "openai/whisper-large-v3-turbo",
            # "meta-llama/Meta-Llama-3.1-8B-Instruct",
            # "distil-whisper/distil-small.en",
            # "Qwen/Qwen2-7B",
            # "llava-hf/llava-interleave-qwen-0.5b-hf",
            # "lmms-lab/LLaVA-Video-7B-Qwen2",
            # "llava-hf/llava-v1.6-mistral-7b-hf",
            # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            # "TIGER-Lab/Mantis-8B-siglip-llama3",  ## make sure sthat optimum-cli-convert works on windows.
            # "microsoft/xclip-base-patch16-zero-shot",
            # "google/vit-base-patch16-224"
            # "MCG-NJU/videomae-base",
            # "MCG-NJU/videomae-large",
            # "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",   ## openclip not yet supported
            # "lmms-lab/llava-onevision-qwen2-7b-si",  
            # "lmms-lab/llava-onevision-qwen2-7b-ov", 
            # "lmms-lab/llava-onevision-qwen2-0.5b-si", 
            # "lmms-lab/llava-onevision-qwen2-0.5b-ov", 
            # "Qwen/Qwen2-VL-7B-Instruct", ## convert_model() ->   ('Couldn\'t get TorchScript module by scripting. With exception:\nComprehension ifs are not supported yet:\n  File "/home/devel/.local/lib/python3.12/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1187\n    \n        if not return_dict:\n            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)\n        return BaseModelOutputWithPast(\n            last_hidden_state=hidden_states,\n\n\nTracing sometimes provide better results, please provide valid \'example_input\' argument. You can also provide TorchScript module that you obtained yourself, please refer to PyTorch documentation: https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html.',)
            # "OpenGVLab/InternVL2_5-1B", ## convert_model() -> torchscript error Couldn't get TorchScript module by scripting. With exception: try blocks aren't supported:
            # "OpenGVLab/InternVL2_5-8B", ## convert_model() -> torchscript error Couldn't get TorchScript module by scripting. With exception: try blocks aren't supported:
            # "OpenGVLab/PVC-InternVL2-8B", ## convert_model() -> torchscript error Couldn't get TorchScript module by scripting. With exception: try blocks aren't supported:
            # "AIDC-AI/Ovis1.6-Llama3.2-3B", # ValueError: Trying to export a ovis model, that is a custom or unsupported architecture,
            # "BAAI/Aquila-VL-2B-llava-qwen", # Asked to export a qwen2 model for the task visual-question-answering (auto-detected), but the Optimum OpenVINO exporter only supports the tasks feature-extraction, feature-extraction-with-past, text-generation, text-generation-with-past, text-classification for qwen2. Please use a supported task. Please open an issue at https://github.com/huggingface/optimum/issues if you would like the task visual-question-answering to be supported in the ONNX export for qwen2.
        ],
        "chunk_settings": {

        },
        "path": "/storage/gpt4v-dataset/data",
        "dst_path": "/storage/gpt4v-dataset/data",
    }
    endpoints = ["cpu", "cuda:0", "openvino:0"]
    resources = {}
    resources["local_endpoints"] = []
    resources["tei_endpoints"] = []
    resources["libp2p_endpoints"] = []
    resources["openvino_endpoints"] = []      
    for model in metadata["models"]:
        for endpoint in endpoints:
            resources["local_endpoints"].append([model, endpoint, 32768])

    ipfs_accelerate_py = ipfs_accelerate_py(resources, metadata)
    asyncio.run(ipfs_accelerate_py.__test__(resources, metadata))
    print("test complete")
