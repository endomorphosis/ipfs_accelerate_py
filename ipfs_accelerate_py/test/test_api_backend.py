import os 
import sys

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
        self.apis.ovms.__test__()
        return None
    
    def test_ollama(self):
        self.apis.ollama.__test__()
        return None

    def test_hf_tgi(self):
        self.apis.hf_tgi.__test__()
        return None
    
    def test_openai_api(self):
        self.apis.openai_api.__test__()
        return None
    
    def test_groq(self):
        self.apis.groq.__test__()
        return None
    
    def test_s3_kit(self):
        self.apis.s3_kit.__test__()
        return None
    
    def test_hf_tei(self):
        self.apis.hf_tei.__test__()
        return None
    
    def test_llvm(self):
        self.apis.llvm.__test__()
        return None
    
    def __test__(self):
        results = {}
        
        try:
            results["ovms"] =  self.test_ovms()
        except Exception as e:
            results["ovms"] = e
        
        try:
            results["ollama"] =  self.test_ollama()
        except Exception as e:
            results["ollama"] = e
        
        try:
            results["hf_tgi"] =  self.test_hf_tgi()
        except Exception as e:
            results["hf_tgi"] = e
        
        try:
            results["openai_api"] =  self.test_openai_api()
        except Exception as e:
            results["openai_api"] = e
        
        try:
            results["groq"] =  self.test_groq()
        except Exception as e:
            results["groq"] = e
        
        try:
            results["s3_kit"] =  self.test_s3_kit()
        except Exception as e:
            results["s3_kit"] = e
        
        try:
            results["hf_tei"] =  self.test_hf_tei()
        except Exception as e:
            results["hf_tei"] = e
        
        try:
            results["llvm"] =  self.test_llvm()
        except Exception as e:
            results["llvm"] = e
        
        return results
        
if __name__ == "__main__":
    test_api_backend(resources={}, metadata={}).__test__()
    print("test_api_backend passed")