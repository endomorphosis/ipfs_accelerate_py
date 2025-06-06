# Try to import modules but set defaults if not available
import logging
logger = logging.getLogger(__name__)

try:
    from . import ovms
except ImportError:
    logger.debug("Failed to import ovms")
    ovms = None

try:
    from . import opea
except ImportError:
    logger.debug("Failed to import opea")
    opea = None

try:
    from . import groq
except ImportError:
    logger.debug("Failed to import groq")
    groq = None

try:
    from . import hf_tei
except ImportError:
    logger.debug("Failed to import hf_tei")
    hf_tei = None

try:
    from . import hf_tgi
except ImportError:
    logger.debug("Failed to import hf_tgi")
    hf_tgi = None

try:
    from . import llvm
except ImportError:
    logger.debug("Failed to import llvm")
    llvm = None

try:
    from . import s3_kit
except ImportError:
    logger.debug("Failed to import s3_kit")
    s3_kit = None

try:
    from . import openai_api
except ImportError:
    logger.debug("Failed to import openai_api")
    openai_api = None

try:
    from . import ollama
except ImportError:
    logger.debug("Failed to import ollama")
    ollama = None

# Create mock classes for each API implementation
def create_mock_class(name):
    attrs = {
        # Add standard method names
        f'request_{name}_endpoint': lambda *args, **kwargs: None,
        f'test_{name}_endpoint': lambda *args, **kwargs: None,
        f'create_{name}_endpoint_handler': lambda *args, **kwargs: None,
        f'make_post_request_{name}': lambda *args, **kwargs: None,
        
        # Add additional method variations used in the code
        'request_tgi_endpoint': lambda *args, **kwargs: None,
        'request_hf_tei_endpoint': lambda *args, **kwargs: None,
        'request_hf_tgi_endpoint': lambda *args, **kwargs: None,
        'test_tgi_endpoint': lambda *args, **kwargs: None,
        
        # Constructor
        '__init__': lambda self, *args, **kwargs: None,
    }
    return type(f'Mock{name.capitalize()}', (), attrs)

# Initialize class references, using real classes if available, mock classes if not
hf_tei_class = getattr(hf_tei, 'hf_tei', None) if hf_tei else create_mock_class('hf_tei')
hf_tgi_class = getattr(hf_tgi, 'hf_tgi', None) if hf_tgi else create_mock_class('hf_tgi')
groq_class = getattr(groq, 'groq', None) if groq else create_mock_class('groq')
ollama_class = getattr(ollama, 'ollama', None) if ollama else create_mock_class('ollama')
ovms_class = getattr(ovms, 'ovms', None) if ovms else create_mock_class('ovms')
opea_class = getattr(opea, 'opea', None) if opea else create_mock_class('opea')
llvm_class = getattr(llvm, 'llvm', None) if llvm else create_mock_class('llvm')
openai_api_class = getattr(openai_api, 'openai_api', None) if openai_api else create_mock_class('openai_api')
s3_kit_class = getattr(s3_kit, 's3_kit', None) if s3_kit else create_mock_class('s3_kit')
try:
    from .api_models_registry import api_models
except ImportError:
    # Create a mock api_models class
    class api_models:
        def __init__(self, resources=None, metadata=None):
            self.resources = resources or {}
            self.metadata = metadata or {}
            
        def test_model(self, model_name):
            return {"supported": True, "name": model_name}
import asyncio
import os
import json

class apis:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        self.inbox = {}
        self.outbox = {}
        self.tokenizer = {}
        self.queues = {}
        self.batch_sizes = {}
        self.endpoint_handler = {}
        self.endpoints = {}
        self.endpoint_types = ["api_endpoints"]
        self.api_endpoints = {}
        
        self.api_models = api_models(self.resources, self.metadata)
        # Initialize endpoints to empty dictionary
        self.endpoints = {}
        if "endpoints" in list(self.metadata.keys()):
            self.endpoints = self.metadata["endpoints"]
        
        if "hf_tei" not in self.resources:
            if hf_tei_class is not None:
                self.resources["hf_tei"] = hf_tei_class(self.resources, self.metadata)
            else:
                # Create a mock implementation
                self.resources["hf_tei"] = type('MockHFTEI', (), {'request_hf_tei_endpoint': lambda *args: None, 'test_hf_tei_endpoint': lambda *args: None})()
            self.hf_tei = self.resources["hf_tei"]
        else:
            self.hf_tei = self.resources["hf_tei"]
            
        if "hf_tgi" not in self.resources:
            if hf_tgi_class is not None:
                self.resources["hf_tgi"] = hf_tgi_class(self.resources, self.metadata)
            else:
                # Create a mock implementation
                self.resources["hf_tgi"] = type('MockHFTGI', (), {'request_tgi_endpoint': lambda *args: None, 'test_tgi_endpoint': lambda *args: None})()
            self.hf_tgi = self.resources["hf_tgi"]
        else:
            self.hf_tgi = self.resources["hf_tgi"]
            
        if "groq" not in self.resources:
            if groq_class is not None:
                self.resources["groq"] = groq_class(self.resources, self.metadata)
            else:
                # Create a mock implementation
                self.resources["groq"] = type('MockGROQ', (), {'request_groq_endpoint': lambda *args: None, 'test_groq_endpoint': lambda *args: None})()
            self.groq = self.resources["groq"]
        else:
            self.groq = self.resources["groq"]
            
        if "llvm" not in self.resources:
            if llvm_class is not None:
                self.resources["llvm"] = llvm_class(self.resources, self.metadata)
            else:
                # Create a mock implementation
                self.resources["llvm"] = type('MockLLVM', (), {'request_llvm_endpoint': lambda *args: None, 'test_llvm_endpoint': lambda *args: None})()
            self.llvm = self.resources["llvm"]
        else:
            self.llvm = self.resources["llvm"]
            
        if "ollama" not in self.resources:
            if ollama_class is not None:
                self.resources["ollama"] = ollama_class(self.resources, self.metadata)
            else:
                # Create a mock implementation
                self.resources["ollama"] = type('MockOllama', (), {'request_ollama_endpoint': lambda *args: None, 'test_ollama_endpoint': lambda *args: None})()
            self.ollama = self.resources["ollama"]
        else:
            self.ollama = self.resources["ollama"]
            
        if "opea" not in self.resources:
            if opea_class is not None:
                self.resources["opea"] = opea_class(self.resources, self.metadata)
            else:
                # Create a mock implementation
                self.resources["opea"] = type('MockOPEA', (), {'request_opea_endpoint': lambda *args: None, 'test_opea_endpoint': lambda *args: None})()
            self.opea = self.resources["opea"]
        else:
            self.opea = self.resources["opea"]
            
        if "ovms" not in self.resources:
            if ovms_class is not None:
                self.resources["ovms"] = ovms_class(self.resources, self.metadata)
            else:
                # Create a mock implementation
                self.resources["ovms"] = type('MockOVMS', (), {'request_ovms_endpoint': lambda *args: None, 'test_ovms_endpoint': lambda *args: None})()
            self.ovms = self.resources["ovms"]
        else:
            self.ovms = self.resources["ovms"]
            
        if "openai_api" not in self.resources:
            if openai_api_class is not None:
                self.resources["openai_api"] = openai_api_class(self.resources, self.metadata)
            else:
                # Create a mock implementation
                self.resources["openai_api"] = type('MockOpenAI', (), {'request_openai_api_endpoint': lambda *args: None, 'test_openai_api_endpoint': lambda *args: None})()
            self.openai_api = self.resources["openai_api"]
        else:
            self.openai_api = self.resources["openai_api"]
        
        if "s3_kit" not in self.resources:
            if s3_kit_class is not None:
                self.resources["s3_kit"] = s3_kit_class(self.resources, self.metadata)
            else:
                # Create a mock implementation
                self.resources["s3_kit"] = type('MockS3Kit', (), {'request_s3_kit_endpoint': lambda *args: None, 'test_s3_kit_endpoint': lambda *args: None})()
            self.s3_kit = self.resources["s3_kit"]
        else:
            self.s3_kit = self.resources["s3_kit"]
        
        self.tgi_endpoints = {}
        self.ollama_endpoints = {}
        self.tei_endpoints = {}
        self.groq_endpoints = {}
        self.llvm_endpoints = {}
        self.opea_endpoints = {}
        self.openai_endpoints = {}
        self.s3_endpoints = {}
        self.ovms_endpoints = {}
        
        self.request_tgi_endpoint = self.hf_tgi.request_tgi_endpoint
        self.request_hf_tei_endpoint = self.hf_tei.request_hf_tei_endpoint
        self.request_groq_endpoint = self.groq.request_groq_endpoint
        self.request_llvm_endpoint = self.llvm.request_llvm_endpoint
        self.request_ollama_endpoint = self.ollama.request_ollama_endpoint
        self.request_openai_api_endpoint = self.openai_api.request_openai_api_endpoint
        self.request_s3_kit_endpoint = self.s3_kit.request_s3_kit_endpoint
        self.request_ovms_endpoint = self.ovms.request_ovms_endpoint
        self.request_opea_endpoint = self.opea.request_opea_endpoint
        
        self.test_groq_endpoint = self.groq.test_groq_endpoint
        self.test_llvm_endpoint = self.llvm.test_llvm_endpoint
        self.test_ollama_endpoint = self.ollama.test_ollama_endpoint
        self.test_ovms_endpoint = self.ovms.test_ovms_endpoint
        self.test_hf_tei_endpoint = self.hf_tei.test_hf_tei_endpoint
        self.test_hf_tgi_endpoint = self.hf_tgi.test_tgi_endpoint
        self.test_openai_api_endpoint = self.openai_api.test_openai_api_endpoint
        self.test_s3_kit_endpoint = self.s3_kit.test_s3_kit_endpoint
        self.test_opea_endpoint = self.opea.test_opea_endpoint

        self.add_groq_endpoint = self.add_groq_endpoint
        self.add_llvm_endpoint = self.add_llvm_endpoint
        self.add_hf_tei_endpoint = self.add_hf_tei_endpoint
        self.add_hf_tgi_endpoint = self.add_hf_tgi_endpoint
        self.add_ollama_endpoint = self.add_ollama_endpoint
        self.add_ovms_endpoint = self.add_ovms_endpoint
        self.add_openai_api_endpoint = self.add_openai_api_endpoint
        self.add_s3_endpoint = self.add_s3_endpoint
        self.add_opea_endpoint = self.add_opea_endpoint
        
        self.rm_groq_endpoint = self.rm_groq_endpoint
        self.rm_llvm_endpoint = self.rm_llvm_endpoint
        self.rm_hf_tei_endpoint = self.rm_hf_tei_endpoint
        self.rm_hf_tgi_endpoint = self.rm_hf_tgi_endpoint
        self.rm_ollama_endpoint = self.rm_ollama_endpoint
        self.rm_ovms_endpoint = self.rm_ovms_endpoint
        self.rm_openai_api_endpoint = self.rm_openai_api_endpoint
        self.rm_s3_endpoint = self.rm_s3_endpoint
        self.rm_opea_endpoint = self.rm_opea_endpoint
        
        self.init()
        return None
    
    def init(self, models=None):
        if type(models) == type(list()):
            test_models = [ self.api_models.test_model(model) for model in models ]
            models_dict = { model: test_models[i] for i, model in enumerate(models) }
        else:
            test_models = []
            models_dict = {}
            
        api_types = ["groq", "hf_tei", "hf_tgi", "llvm", "ollama", "openai_api", "s3_kit", "ovms"]
        
        for api_type in api_types:
            if api_type not in dir(self):
                init_method = getattr(self,"init_" + api_type)
                init_method(models)
            if api_type not in globals():
                this_file_path = os.path.relpath(__file__)
                this_folder_path = os.path.dirname(this_file_path)
                skill_path  = os.path.join(this_folder_path, api_type + ".py")
                # absolute_path = os.path.normpath(absolute_path)
                try:
                    with open(skill_path, encoding='utf-8') as f:
                        exec(f.read())
                except Exception as e:
                    print(e)
                    print(os.listdir("."))
                    print(os.listdir(this_folder_path))
                    pass
                with open(skill_path, encoding='utf-8') as f:
                    globals()[api_type] = exec(f.read())
            else:
                pass
            if api_type not in list(self.resources.keys()):
                # from .api_types[api_type] import api_type
                ## import the api_type class from the api_types module
                
                self.resources[api_type] =  globals()[api_type](self.resources, self.metadata)
                setattr(self, api_type, self.resources[api_type])
            else:
                setattr(self, api_type, self.resources[api_type])
            try:
                if hasattr(self.resources[api_type], "create_" + api_type + "_endpoint_handler"):
                    get_remote_endpoint_handler = getattr(self.resources[api_type], "create_" + api_type + "_endpoint_handler")
                    setattr(self, "create_" + api_type + "_endpoint_handler", get_remote_endpoint_handler)
                else:
                    # Create a dummy handler
                    setattr(self, "create_" + api_type + "_endpoint_handler", lambda *args: None)
                
                if hasattr(self.resources[api_type], "make_post_request_" + api_type):
                    get_remote_post_request = getattr(self.resources[api_type], "make_post_request_" + api_type)
                    setattr(self, "make_post_request_" + api_type, get_remote_post_request)
                else:
                    # Create a dummy handler
                    setattr(self, "make_post_request_" + api_type, lambda *args: None)
            except Exception as e:
                print(f"Error setting up API {api_type}: {e}")
                # Create dummy handlers as fallback
                setattr(self, "create_" + api_type + "_endpoint_handler", lambda *args: None)
                setattr(self, "make_post_request_" + api_type, lambda *args: None)
            # setattr(self, "create_" + api_type + "_endpoint_handler", getattr(self, api_type).create_endpoint_handler)
            # setattr(self, "make_post_request_" + api_type, getattr(self, api_type).make_post_request)   
                    
        for model in list(models_dict.keys()) if models_dict else []:
            for api_type in api_types:
                endpoint_key = api_type + "_endpoints"
                if endpoint_key in list(self.endpoints.keys()):
                    try:
                        if len(self.endpoints[endpoint_key]) > 0:
                            for endpoint_model in list(self.endpoints[endpoint_key].keys()):
                                for endpoint in self.endpoints[endpoint_key][endpoint_model]:
                                    if len(endpoint) == 3:
                                        this_model = endpoint[0]
                                        this_endpoint = endpoint[1]
                                        context_length = endpoint[2]
                                        if model == this_model:
                                            # Initialize resources for this endpoint
                                            if model not in self.resources.get("batch_sizes", {}):
                                                self.resources.setdefault("batch_sizes", {})[model] = {}
                                            if model not in self.resources.get("queues", {}):
                                                self.resources.setdefault("queues", {})[model] = {}
                                            if model not in self.resources.get("endpoint_handler", {}):
                                                self.resources.setdefault("endpoint_handler", {})[model] = {}
                                                
                                            # Set up the endpoint
                                            self.resources["batch_sizes"][model][this_endpoint] = 0
                                            self.resources["queues"][model][this_endpoint] = asyncio.Queue(64)
                                            
                                            # Try to create an endpoint handler, but provide a fallback
                                            try:
                                                if hasattr(self, 'create_endpoint_handler'):
                                                    self.resources["endpoint_handler"][model][this_endpoint] = self.create_endpoint_handler(model, this_endpoint, context_length)
                                                else:
                                                    # Default dummy handler
                                                    self.resources["endpoint_handler"][model][this_endpoint] = lambda *args: None
                                            except Exception as e:
                                                print(f"Error creating endpoint handler: {e}")
                                                self.resources["endpoint_handler"][model][this_endpoint] = lambda *args: None
                    except Exception as e:
                        print(f"Error setting up endpoints for {api_type}: {e}")
        return None
    
    def init_groq(self, models=None):
        return None
    
    def init_llvm(self, models=None):
        return None
    
    def init_hf_tei(self, models=None):
        return None
    
    def init_hf_tgi(self, models=None):
        return None
    
    def init_ollama(self, models=None):
        return None
    
    def init_opea(self, models=None):
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
    
    async def add_hf_tei_endpoint(self, model, endpoint):
        return None
    
    async def add_groq_endpoint(self, model, endpoint):
        return None

    async def add_ollama_endpoint(self, model, endpoint):
        return None
    
    async def add_opea_endpoint(self, model, endpoint):
        return None
    
    async def add_hf_tgi_endpoint(self, model, endpoiointsnt):
        return None
    
    async def add_openvino_endpoint(self, model, endpoint):
        return None
    
    async def add_libp2p_endpoint(self, model, endpoint):
        return None
    
    async def add_openai_api_endpoint(self, model, endpoint):
        return None
    
    async def add_s3_endpoint(self, model, endpoint):
        return None
    
    async def rm_ovms_endpoint(self, model, endpoint):
        return None
    
    async def rm_llvm_endpoint(self, model, endpoint):
        return None
    
    async def rm_hf_tei_endpoint(self, model, endpoint):
        return None
    
    async def rm_groq_endpoint(self, model, endpoint):
        return None
    
    async def rm_ollama_endpoint(self, model, endpoint):
        return None
    
    async def rm_opea_endpoint(self, model, endpoint):
        return None
    
    async def rm_hf_tgi_endpoint(self, model, endpoint):
        return None
    
    async def rm_openvino_endpoint(self, model, endpoint):
        return None
    
    async def rm_libp2p_endpoint(self, model, endpoint):
        return None
    
    async def rm_openai_api_endpoint(self, model, endpoint):
        return None
    
    async def rm_s3_endpoint(self, model, endpoint):
        return None
    
    async def test_ovms_endpoint(self, model, endpoint):
        return None

    async def test_llvm_endpoint(self, model, endpoint):
        return None
    
    async def test_hf_tei_endpoint(self, model, endpoint):
        return None
    
    async def test_groq_endpoint(self, model, endpoint):
        return None
    
    async def test_ollama_endpoint(self, model, endpoint):
        return None
    
    async def test_opea_endpoint(self, model, endpoint):
        return None
    
    async def test_hf_tgi_endpoint(self, model, endpoint):
        return None
    
    async def test_openvino_endpoint(self, model, endpoint):
        return None

    async def test_libp2p_endpoint(self, model, endpoint):
        return None

    async def test_openai_api_endpoint(self, model, endpoint):
        return None
    
    async def test_s3_kit_endpoint(self, model, endpoint):
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
    
    
    def request_hf_tei_endpoint(self, model, endpoint=None, endpoint_type=None, batch=None):
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
    
    def request_hf_tgi_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
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
    
    async def request_hf_tei_endpoint(self, model, batch_size):
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
    
    async def request_openai_api_endpoint(self, model, batch_size):
        if model in self.openai_endpoints:
            for endpoint in self.openai_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_s3_kit_endpoint(self, model, batch_size):
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

    async def request_opea_endpoint(self, model, batch_size):
        if model in self.local_endpoints:
            for endpoint in self.local_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None

    
    def __test__(self):
        return None
