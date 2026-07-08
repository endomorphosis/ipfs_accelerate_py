import time

from ipfs_accelerate_py.mcp.tools import inference


class _FakeStorage:
    def __init__(self):
        self.items = []
        self.using_fallback = True

    def store(self, data, filename=None, pin=False):
        cid = f"cid-{len(self.items) + 1}"
        self.items.append({"cid": cid, "data": data, "filename": filename, "pin": pin})
        return cid


class _FakeDatasetsManager:
    def __init__(self):
        self.events = []
        self.provenance = []

    def log_event(self, event_type, data, level="INFO", category="GENERAL"):
        self.events.append(
            {
                "event_type": event_type,
                "data": data,
                "level": level,
                "category": category,
            }
        )
        return True

    def track_provenance(self, operation, data, record_type="TRANSFORMATION"):
        self.provenance.append(
            {
                "operation": operation,
                "data": data,
                "record_type": record_type,
            }
        )
        return "prov-cid-1"


class _FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator

    def access_resource(self, name, **kwargs):
        if name == "get_model_info":
            return {"type": "generation"}
        if name == "models_config":
            return {"generation": ["demo-model"]}
        return None

    def use_tool(self, name, **kwargs):
        return {"ok": True, "name": name, "kwargs": kwargs}


def test_run_inference_persists_outputs_and_provenance(monkeypatch):
    fake_storage = _FakeStorage()
    fake_datasets = _FakeDatasetsManager()
    fake_mcp = _FakeMCP()

    monkeypatch.setattr(inference, "_get_storage_client", lambda: fake_storage)
    monkeypatch.setattr(inference, "_datasets_manager", fake_datasets)
    monkeypatch.setattr(inference, "_provenance_logger", None)
    monkeypatch.setattr(inference.time, "sleep", lambda _: None)

    inference.register_tools(fake_mcp)
    run_inference = fake_mcp.tools["run_inference"]

    result = run_inference(model="demo-model", inputs=["hello world"], device="cpu")

    assert result["model"] == "demo-model"
    assert result["output_cid"] == "cid-2"
    assert result["input_cid"] == "cid-1"
    assert result["provenance_cid"] == "prov-cid-1"
    assert result["audit_logged"] is True
    assert result["storage"]["attempted"] is True
    assert result["storage"]["success"] is True

    assert len(fake_storage.items) == 2
    assert fake_storage.items[0]["filename"] == "demo-model_input.json"
    assert fake_storage.items[1]["filename"] == "demo-model_output.json"

    assert fake_datasets.events[0]["event_type"] == "inference_completed"
    assert fake_datasets.provenance[0]["operation"] == "inference"
    assert fake_datasets.provenance[0]["data"]["output_cid"] == "cid-2"
