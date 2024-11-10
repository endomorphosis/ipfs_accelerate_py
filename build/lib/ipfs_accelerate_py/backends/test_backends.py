class test_backends_py:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        
    def __test__(self):
        return True
    
    def __call__(self):
        return self

    def get(self, key):
        return None

    def put(self, key, value):
        return None

    def delete(self, key):
        return None

    def list(self):
        return []