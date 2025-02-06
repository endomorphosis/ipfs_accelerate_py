## this is where models that are normally in the "provider/modelname" format are registered to an api skillset
class api_models:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources  
        self.metadata = metadata
        return None
    
    def openai_models(self):
        return None
    
    def huggingface_models(self):
        return None
    
    def llvm_models(self):
        return None
    
    def ollama_models(self):
        return None
    
    def groq_models(self):
        return None
    
    def ovms_models(self):
        return None