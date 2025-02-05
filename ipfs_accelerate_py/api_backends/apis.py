from .ovms import ovms
from .groq import groq
from .hf_tei import hf_tei
from .hf_tgi import hf_tgi
from .llvm import llvm
from .s3_kit import s3_kit
from .openai_api import openai_api
from .ollama import ollama

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
    
    def init(self):
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
    
    def __test__(self):
        return None
