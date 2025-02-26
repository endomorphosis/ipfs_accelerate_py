import os
from pydantic import BaseModel

# https://console.groq.com/docs/models
PRODUCTION_MODELS = {
    "distil-whisper-large-v3-en": {
        "context_window": None
    },
    "gemma2-9b-it": {
        "context_window": 8192
    },
    "llama-3.3-70b-versatile": {
        "context_window": 128000
    },
    "llama-3.1-8b-instant": {
        "context_window": 128000
    },
    "llama-guard-3-8b": {
        "context_window": 8192
    },
    "llama3-70b-8192": {
        "context_window": 8192
    },
    "llama3-8b-8192": {
        "context_window": 8192
    },
    "mixtral-8x7b-32768": {
        "context_window": 32768
    },
    "whisper-large-v3": {
        "context_window": None
    },
    "whisper-large-v3-turbo": {
        "context_window": None
    },
}


PREVIEW_MODELS = {
    "qwen-2.5-32b": {
        "context_window": 128000
    },
    "deepseek-r1-distill-qwen-32b": {
        "context_window": 128000
    },
    "deepseek-r1-distill-llama-70b-specdec": {
        "context_window": 128000
    },
    "deepseek-r1-distill-llama-70b": {
        "context_window": 128000
    },
    "llama-3.3-70b-specdec": {
        "context_window": 8192
    },
    "llama-3.2-1b-preview": {
        "context_window": 128000
    },
    "llama-3.2-3b-preview": {
        "context_window": 128000
    },
    "llama-3.2-11b-vision-preview": {
        "context_window": 128000
    },
    "llama-3.2-90b-vision-preview": {
        "context_window": 128000
    },
}


class groq:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.create_groq_endpoint_handler = self.create_groq_endpoint_handler()
        self.request_groq_endpoint = self.request_groq_endpoint
        self.test_groq_endpoint = self.test_groq_endpoint
        self.make_post_request_groq = self.make_post_request_groq

        # api_key = os.getenv("GROQ_API_KEY")
        # if api_key is None:
        #     raise ValueError("GROQ_API_KEY environment variable not set")

        return None
    
    def make_post_request_groq(self, model, endpoint, endpoint_type, batch):
        
        return None
    
    def request_groq_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
        
        return None
    
    def test_groq_endpoint(self):
        ## test dependencies
        return None
    
    def _test_groq_endpoint(self):
        ## test dependencies
        return None
    
    def create_groq_endpoint_handler(self):
        def handler(request):
            return None
        return handler

    def __test___(self):
        # Test to see if we can load groq's dependencies
        self._test_groq_endpoint()
        # Test to see if we can make requests to the groq endpoint
        self._test_make_post_request_groq()

        # Test the production models
        self._test_llama_models()
        self._test_gemma2_model()
        self._test_mixtral_model()
        self._test_whisper_models()
        
        
        self._test_request_groq_endpoint()



        self._test_groq_endpoint_handler()
        return None