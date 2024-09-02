from .backends import backends

class load_checkpoint_and_dispatch:
    def __init__(self, model, checkpoint, device_map):
        self.model = model
        self.checkpoint = checkpoint
        self.device_map = device_map

    def __call__(self):

        if self.device_map == "auto":
            import torch
            self.model.load_state_dict(torch.load(self.checkpoint))
        elif self.device_map == "lilypad":
            from .backends import backends
            docker_tunnel = backends.lilypad(self.model, self.checkpoint)
            self.model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"))
        elif self.device_map == "akash":
            from .backends import backends
            docker_tunnel = backends.akash(self.model, self.checkpoint)
            self.model.load_state_dict(torch.load(self.checkpoint, map_location="cuda:0"))
        elif self.device_map == "vast":
            from .backends import backends
            docker_tunnel = backends.vast(self.model, self.checkpoint)
            self.model.load_state_dict(torch.load(self.checkpoint, map_location="cuda:0"))
        elif self.device_map == "hallucinate":
            from .backends import backends
            docker_tunnel = backends.hallucinate(self.model, self.checkpoint)
            self.model.load_state_dict(torch.load(self.checkpoint, map_location="cuda:0"))
        else:
            raise ValueError("Invalid device_map")

        return self.model