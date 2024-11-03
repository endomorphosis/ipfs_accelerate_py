class backends:
    def __init__(self,resources, metadata):
        self.models = metadata["models"]
        self.endpoint_types = [
            "tei_endpoints",
            "openvino_endpoints",
            "libp2p_endpoints",
            "local_endpoints"
        ]
        self.backends = {
            "lilypad": None,
            "akash": None,
            "libp2p": None,
            "hallucinate": None
        }
    
    def __call__(self):
        return self

class marketplace:
    def __init__(self, models):
        self.gpu_models = []
        self.cpu_models = []
        self.markets = {
            "lilypad": {},
            "akash": {},
            "libp2p": {},
            "vast": {},
        }
        self.marketplace = {}

    def __call__(self):
        return self
    
    def define_market(self):
        for model in self.gpu_models:
            self.markets["lilypad"][model] = []
            self.markets["akash"][model] = []
            self.markets["vast"][model] = []
            self.markets["coreweave"][model] = []
            
    def query_marketplace(self):
        lilypad_instances = self.backends.lilypad()
        akash_instances = self.backends.akash()
        vast_instances = self.backends.vast()
        coreweave_instances = self.backends.coreweave()
        return lilypad_instances, akash_instances, vast_instances, coreweave_instances
    
    def marketplace_by_gpu_model(self):
        marketplace = self.query_marketplace()
        lilypad_instances = marketplace[0]
        akash_instances = marketplace[1]

        for instance in lilypad_instances:
            this_model = instance.gpu_model
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