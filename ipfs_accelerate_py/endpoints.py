class endpoints_py:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        
        
    def __call__(self):
        return self
    
    def __test__(self):
        return self
    
    