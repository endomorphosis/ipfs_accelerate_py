class opea:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.create_opea_endpoint_handler = self.create_opea_endpoint_handler
        self.request_opea_endpoint = self.request_opea_endpoint
        self.test__endpoint = self.test_opea_endpoint
        self.make_post_request_opea = self.make_post_request_opea
        return None
    
    def make_post_request_opea (self, model, endpoint, endpoint_type, batch):
        
        return None
    
    def request_opea_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
        
        return None
    
    def test_opea_endpoint(self):
        ## test dependencies
        return None

    def create_opea_endpoint_handler(self):
        def handler(request):
            return None
        return handler
    
    