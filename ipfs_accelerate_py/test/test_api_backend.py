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
            from ..api_backends.apis import apis
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
        self.test_ovms()
        self.test_ollama()
        self.test_hf_tgi()
        self.test_openai_api()
        self.test_groq()
        self.test_s3_kit()
        self.test_hf_tei()
        self.test_llvm()
        return None