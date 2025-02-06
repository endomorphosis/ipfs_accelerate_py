class llvm:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.create_llvm_endpoint_handler   = self.create_llvm_endpoint_handler
        return None
    
    def create_llvm_endpoint_handler(self):
        def handler(request):
            return None
        return handler