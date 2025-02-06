## this is where models that are normally in the "provider/modelname" format are registered to an api skillset
import json
import os

class api_models:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources  
        self.metadata = metadata
        return None
    
    def openai_models(self, model=None):
        models = []
        with open('openai_models.json') as f:
            models = json.load(f)
        if model in models:
            return True
        else:
            return False
    
    def huggingface_models(self, models=None):
        return True
    
    def llvm_models(self):
        return True
    
    def ollama_models(self):
        return False
    
    def groq_models(self):
        return False
    
    def ovms_models(self):
        return True
    
    def test_model(self, model):
        results = {}
        try:
            results["openai"] = self.openai_models(model)
        except Exception as e:
            print(e)
            results["openai"] = False
        try:
            results["huggingface"] = self.huggingface_models(model)
        except Exception as e:
            print(e)
            results["huggingface"] = False
        try:
            results["llvm"] = self.llvm_models(model)
        except Exception as e:
            print(e)
            results["llvm"] = False
        try:
            results["ollama"] = self.ollama_models(model)
        except Exception as e:
            print(e)
            results["ollama"] = False
        try:
            results["groq"] = self.groq_models(model)
        except Exception as e:
            print(e)
            results["groq"] = False
        try:
            results["ovms"] = self.ovms_models(model)
        except Exception as e:
            print(e)
            results["ovms"] = False
        
        return results
    