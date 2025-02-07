class llvm:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.create_llvm_endpoint_handler   = self.create_llvm_endpoint_handler
        self.request_llvm_endpoint         = self.request_llvm_endpoint
        self.test_llvm_endpoint            = self.test_llvm_endpoint
        self.make_post_request_llvm = self.make_post_request_llvm
        return None
    
    def make_post_request_llvm(self, model, endpoint, endpoint_type, batch):
        
        return None
    
    def test_llvm_endpoint(self):
        ## test dependencies
        return None
    
    def request_llvm_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
        
        return None
    
    def create_llvm_endpoint_handler(self):
        def handler(request):
            return None
        return handler