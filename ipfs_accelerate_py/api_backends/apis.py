class apis:
    def __init__(self, resources, metadata):
        if resources is None:
            resources = {}
        if metadata is None:
            metadata = {}
            
        self.resources = resources
        self.metadata = metadata
        if "groq" not in dir(self):
            if "groq" not in list(self.resources.keys()):
                from .groq import groq
                self.resources["groq"] = groq()
                self.groq = self.resources["groq"]
            else:
                self.groq = self.resources["groq"]
        
        if "hf_tei" not in dir(self):
            if "hf_tei" not in list(self.resources.keys()):
                from .hf_tei import hf_tei
                self.resources["hf_tei"] = hf_tei()
                self.hf_tei = self.resources["hf_tei"]
            else:
                self.hf_tei = self.resources["hf_tei"]
                
        if "hf_tgi" not in dir(self):
            if hf_tgi not in list(self.resources.keys()):
                from .hf_tgi import hf_tgi
                self.resources["hf_tgi"] = hf_tgi()
                self.hf_tgi = self.resources["hf_tgi"]
            else:
                self.hf_tgi = self.resources["hf_tgi"]
                
        if "llvm" not in dir(self):
            if "llvm" not in list(self.resources.keys()):
                from .llvm import llvm
                self.resources["llvm"] = llvm()
                self.llvm = self.resources["llvm"]
            else:
                self.llvm = self.resources["llvm"]
                
        if "ovms" not in dir(self):
            if "ovms" not in list(self.resources.keys()):
                from .ovms import ovms
                self.resources["ovms"] = ovms()
                self.ovms = self.resources["ovms"]
            else:
                self.ovms = self.resources["ovms"]
                
        if "s3_kit" not in dir(self):
            if "s3_kit" not in list(self.resources.keys()):
                from .s3_kit import s3_kit
                self.resources["s3_kit"] = s3_kit()
                self.s3_kit = self.resources["s3_kit"]
            else:
                self.s3_kit = self.resources["s3_kit"]
                
        return None
    
    def init(self):
        return None
    
    def __test__(self):
        return None
