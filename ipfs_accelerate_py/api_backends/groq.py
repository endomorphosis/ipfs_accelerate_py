class groq:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.create_groq_endpoint_handler = self.create_groq_endpoint_handler()
        self.request_groq_endpoint = self.request_groq_endpoint
        self.test_groq_endpoint = self.test_groq_endpoint
        self.make_post_request_groq = self.make_post_request_groq
        return None
    
    def make_post_request_groq(self, model, endpoint, endpoint_type, batch):
        
        return None
    
    def request_groq_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
        
        return None
    
    def test_groq_endpoint(self):
        ## test dependencies
        return None
    
    def create_groq_endpoint_handler(self):
        def handler(request):
            return None
        return handler