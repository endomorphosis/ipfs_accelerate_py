class opea:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.create_ollama_endpoint_handler = self.create_ollama_endpoint_handler
        self.request_ollama_endpoint = self.request_ollama_endpoint
        self.test_ollama_endpoint = self.test_ollama_endpoint
        self.make_post_request_ollama = self.make_post_request_ollama
        return None
    
    def make_post_request_ollama (self, model, endpoint, endpoint_type, batch):
        
        return None
    
    def test_ollama_endpoint(self):
        ## test dependencies
        return None
    
    def request_ollama_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
        
        return None
    
    def create_ollama_endpoint_handler(self):
        def handler(request):
            return None
        return handler
    
    