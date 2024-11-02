import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_ipfs_accelerate import test_ipfs_accelerate
class worker:
    def __init__(self, metadata, resources):
        self.metadata = metadata
        self.resources = resources
        self.endpoint_types = ["local_endpoints"]
        self.hardware_backends = ["llama_cpp", "cpu", "gpu", "openvino", "optimum", "optimum_cpp", "optimum_intel","ipex"]
        if "test_ipfs_accelerate" not in globals():
            self.test_ipfs_accelerate = test_ipfs_accelerate(resources, metadata)
            self.hwtest = self.test_ipfs_accelerate
        for endpoint in self.endpoint_types:
            if endpoint not in dir(self):
                self.__dict__[endpoint] = {}        
            for backend in self.hardware_backends:
                if backend not in list(self.__dict__[endpoint].keys()):
                    self.__dict__[endpoint][backend] = {}
        print(self.__dict__)
        return None
            
    def __call__(self):
        return self
    
    def __test__(self):
        return self 
    
if __name__ == '__main__':
    # run(skillset=os.path.join(os.path.dirname(__file__), 'skillset'))
    resources = {}
    metadata = {}
    try:
        this_worker = worker(resources, metadata)
        this_test = this_worker.__test__(resources, metadata)
    except Exception as e:
        print(e)
        pass