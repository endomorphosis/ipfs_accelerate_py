class backends:
    def __init__(self, model, checkpoint):
        self.model = model
        self.checkpoint = checkpoint
        self.backends = {
            "lilypad": None,
            "akash": None,
            "libp2p": None,
            "vast": None,
            "hallucinate": None
        }
        self.endpoint_types = ["tei_endpoints", "openvino_endpoints", "libp2p_endpoints", "local_endpoints"]


class marketplace:
    def __init__(self, model, checkpoint):
        self.gpu_models = []
        self.cpu_models = []
        self.markets = {
            "lilypad": {},
            "akash": {},
            "vast": {},
            "coreweave": {},
            "huggingface": {}
        }
        self.marketplace = {}

    def __call__(self):
        return self
    
    def define_market(self):
        for model in self.gpu_models:
            self.markets["lilypad"][model] = []
            self.markets["akash"][model] = []

    def query_marketplace(self):
        lilypad_instances = lilypad()
        akash_instances = akash()
        vast_instances = vast()
        coreweave_instances = coreweave()
        return lilypad_instances, akash_instances , vast_instances, coreweave_instances
    
    def marketplace_by_gpu_model(self):
        marketplace = self.query_marketplace()
        lilypad_instances = marketplace[0]
        akash_instances = marketplace[1]
        vast_instances = marketplace[2]
        coreweave_instances = marketplace[3]

        for instance in lilypad_instances:
            this_model = instance.gpu_model
            self.market["lilypad"][this_model]
        for instance in akash_instances:
            model = instance.gpu_model

        return self.gpu_models


class start_container:
    def __init__(self, model, checkpoint):
        self.model = model
        self.checkpoint = checkpoint

class stop_container:
    def __init__(self, model, checkpoint):
        self.model = model
        self.checkpoint = checkpoint

class docker_tunnel:
    def __init__(self, model, checkpoint):
        self.model = model
        self.checkpoint = checkpoint

    def __call__(self):
        import torch
        self.model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"))
        return self.model
    
class load_model_and_dispatch:
    def __init__(self, model, endpoint, backend):
        self.model = model
        self.endpoint = endpoint
        self.backend = backend
        self.load_model = self.load_model
        return None
    
