import os 
import sys
import importlib.util
import importlib

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import from apis directory within the test directory
from apis import (
    test_claude, test_groq, test_hf_tgi, test_hf_tei, test_llvm,
    test_openai_api, test_ovms, test_ollama, test_s3_kit,
    test_gemini, test_opea
)

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ipfs_accelerate_py'))
import api_backends


class test_api_backend:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.test_hf_tgi = self.test_hf_tgi
        self.test_openai_api = self.test_openai_api
        self.test_groq = self.test_groq
        self.test_s3_kit = self.test_s3_kit
        self.test_hf_tei = self.test_hf_tei
        self.test_llvm = self.test_llvm
        self.test_ovms = self.test_ovms
        self.test_ollama = self.test_ollama
        self.test_gemini = self.test_gemini
        self.test_opea = self.test_opea
        
        if "apis" not in self.resources:
            from ipfs_accelerate_py.api_backends import apis
            self.resources["apis"] = apis(resources=self.resources, metadata=self.metadata)
            self.apis = self.resources["apis"]
        else:
            self.apis = self.resources["apis"]
            
        return None

    def test_ovms(self):
        try:
            ovms_test = test_ovms.test_ovms(resources=self.resources, metadata=self.metadata)
            return ovms_test.__test__()
        except Exception as e:
            return e
    
    def test_ollama(self):
        try:
            ollama_test = test_ollama.test_ollama(resources=self.resources, metadata=self.metadata)
            return ollama_test.__test__()
        except Exception as e:
            return e

    def test_hf_tgi(self):
        try:
            hf_tgi_test = test_hf_tgi.test_hf_tgi(resources=self.resources, metadata=self.metadata)
            return hf_tgi_test.__test__()
        except Exception as e:
            return e
    
    def test_openai_api(self):
        try:
            openai_test = test_openai_api.test_openai_api(resources=self.resources, metadata=self.metadata)
            return openai_test.__test__()
        except Exception as e:
            return e
    
    def test_groq(self):
        try:
            groq_test = test_groq.test_groq(resources=self.resources, metadata=self.metadata)
            return groq_test.__test__()
        except Exception as e:
            return e
        
    def test_s3_kit(self):
        try:
            s3_kit_test = test_s3_kit.test_s3_kit(resources=self.resources, metadata=self.metadata)
            return s3_kit_test.__test__()
        except Exception as e:
            return e
        
    def test_hf_tei(self):
        try:
            hf_tei_test = test_hf_tei.test_hf_tei(resources=self.resources, metadata=self.metadata)
            return hf_tei_test.__test__()
        except Exception as e:
            return e
    
    def test_llvm(self):
        try:
            llvm_test = test_llvm.test_llvm(resources=self.resources, metadata=self.metadata)
            return llvm_test.__test__()
        except Exception as e:
            return e
    
    def test_gemini(self):
        try:
            gemini_test = test_gemini.test_gemini(resources=self.resources, metadata=self.metadata)
            return gemini_test.__test__()
        except Exception as e:
            return e
    
    def test_opea(self):
        try:
            opea_test = test_opea.test_opea(resources=self.resources, metadata=self.metadata)
            return opea_test.__test__()
        except Exception as e:
            return e
    
    def __test__(self):
        results = {}
        
        try:
            results["openai_api"] =  self.test_openai_api()
        except Exception as e:
            results["openai_api"] = str(e)

        try:
            results["groq"] =  self.test_groq()
        except Exception as e:
            results["groq"] = str(e)

        try:
            results["ovms"] =  self.test_ovms()
        except Exception as e:
            results["ovms"] = str(e)
        
        try:
            results["ollama"] =  self.test_ollama()
        except Exception as e:
            results["ollama"] = str(e)
        
        try:
            results["hf_tgi"] =  self.test_hf_tgi()
        except Exception as e:
            results["hf_tgi"] = str(e)
        
        try:
            results["s3_kit"] =  self.test_s3_kit()
        except Exception as e:
            results["s3_kit"] = str(e)
        
        try:
            results["hf_tei"] =  self.test_hf_tei()
        except Exception as e:
            results["hf_tei"] = str(e)
        
        try:
            results["llvm"] =  self.test_llvm()
        except Exception as e:
            results["llvm"] = str(e)
        
        try:
            results["gemini"] =  self.test_gemini()
        except Exception as e:
            results["gemini"] = str(e)
        
        try:
            results["opea"] =  self.test_opea()
        except Exception as e:
            results["opea"] = str(e)
        
        return results
        
if __name__ == "__main__":
    this_api_backend = test_api_backend(resources={}, metadata={})
    results = this_api_backend.__test__()
    print("test_api_backend passed")