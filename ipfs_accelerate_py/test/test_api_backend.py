import os 
import sys
from ipfs_accelerate_py.api_backends import apis


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
        
        if "apis" not in self.resources:
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
            from api_backends import apis
            self.resources["apis"] = apis(resources=self.resources, metadata=self.metadata)
            self.apis = self.resources["apis"]
        else:
            self.apis = self.resources["apis"]
            
        return None

    def test_ovms(self):
        try:
            return self.apis.ovms.__test__()
        except Exception as e:
            return e
    
    def test_ollama(self):
        try:
            return self.apis.ollama.__test__()
        except Exception as e:
            return e

    def test_hf_tgi(self):
        try:
            return self.apis.hf_tgi.__test__()
        except Exception as e:
            return e
    
    def test_openai_api(self):
        try:
            return self.apis.openai_api.__test__()
        except Exception as e:
            return e
    
    def test_groq(self):
        try:
            return self.apis.groq.__test__()
        except Exception as e:
            return e
        
    def test_s3_kit(self):
        try:
            return self.apis.s3_kit.__test__()
        except Exception as e:
            return e
        
    def test_hf_tei(self):
        try:
            return self.apis.hf_tei.__test__()
        except Exception as e:
            return e
    
    def test_llvm(self):
        try:
            return self.apis.llvm.__test__()
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
        
        return results
        
if __name__ == "__main__":
    test_api_backend(resources={}, metadata={}).__test__()
    print("test_api_backend passed")