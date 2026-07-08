from __future__ import annotations

import types
from types import SimpleNamespace


class _CapturedOutput:
    def __init__(self) -> None:
        self.payload = None
        self.as_json = None

    def emit(self, payload, as_json: bool) -> None:
        self.payload = payload
        self.as_json = as_json


def _install_fake_model_manager(monkeypatch):
    root_mod = types.ModuleType("ipfs_datasets_py")
    utils_mod = types.ModuleType("ipfs_datasets_py.utils")
    manager_mod = types.ModuleType("ipfs_datasets_py.utils.model_manager")

    def build_hf_inference_ipld_document(*, model_kind=None, include_generated_at=True):
        doc = {
            "kind": "ipfs_datasets_py.hf_inference_model_registry",
            "schema_version": "1.0",
            "model_kind": model_kind,
            "models": [{"model_id": "org/model-a"}],
            "count": 1,
        }
        if include_generated_at:
            doc["generated_at"] = "2026-03-02T00:00:00+00:00"
        return doc

    def get_hf_inference_ipld_cid(*, model_kind=None, base="base32", codec="raw", mh_type="sha2-256"):
        _ = (model_kind, base, codec, mh_type)
        return "bafkreifakecid"

    def publish_hf_inference_ipld_to_ipfs(*, model_kind=None, pin=True, backend=None, backend_instance=None):
        _ = backend_instance
        return {
            "status": "success",
            "local_cid": "bafkreifakelocal",
            "ipfs_cid": "bafkreiipfs",
            "bytes": 123,
            "model_kind": model_kind,
            "count": 1,
            "pin": pin,
            "backend": backend,
        }

    def load_hf_inference_ipld_from_ipfs(cid, *, backend=None, backend_instance=None):
        _ = (backend_instance,)
        if cid == "bad":
            raise ValueError("IPFS object is not an HF inference model registry IPLD document")
        return {
            "kind": "ipfs_datasets_py.hf_inference_model_registry",
            "schema_version": "1.0",
            "model_kind": "llm",
            "models": [],
            "count": 0,
            "cid": cid,
            "backend": backend,
        }

    manager_mod.build_hf_inference_ipld_document = build_hf_inference_ipld_document
    manager_mod.get_hf_inference_ipld_cid = get_hf_inference_ipld_cid
    manager_mod.publish_hf_inference_ipld_to_ipfs = publish_hf_inference_ipld_to_ipfs
    manager_mod.load_hf_inference_ipld_from_ipfs = load_hf_inference_ipld_from_ipfs

    monkeypatch.setitem(__import__("sys").modules, "ipfs_datasets_py", root_mod)
    monkeypatch.setitem(__import__("sys").modules, "ipfs_datasets_py.utils", utils_mod)
    monkeypatch.setitem(__import__("sys").modules, "ipfs_datasets_py.utils.model_manager", manager_mod)


def test_cli_models_ipld_commands_smoke(monkeypatch):
    _install_fake_model_manager(monkeypatch)

    import ipfs_accelerate_py.cli as cli_module

    monkeypatch.setattr(cli_module, "_load_heavy_imports", lambda: None)

    cli = cli_module.IPFSAccelerateCLI()
    captured = _CapturedOutput()
    monkeypatch.setattr(cli, "_print_output", captured.emit)

    rc = cli.run_models_ipld_document(
        SimpleNamespace(kind="llm", deterministic=True, output_json=False)
    )
    assert rc == 0
    assert captured.payload["status"] == "success"
    assert captured.payload["deterministic"] is True
    assert captured.payload["document"]["kind"] == "ipfs_datasets_py.hf_inference_model_registry"

    rc = cli.run_models_ipld_cid(
        SimpleNamespace(kind="llm", base="base32", codec="raw", mh_type="sha2-256", output_json=False)
    )
    assert rc == 0
    assert captured.payload["status"] == "success"
    assert captured.payload["cid"] == "bafkreifakecid"

    rc = cli.run_models_ipld_publish(
        SimpleNamespace(kind="llm", backend="helia", no_pin=False, output_json=False)
    )
    assert rc == 0
    assert captured.payload["status"] == "success"
    assert captured.payload["ipfs_cid"] == "bafkreiipfs"

    rc = cli.run_models_ipld_load(
        SimpleNamespace(cid="bafyok", backend="helia", output_json=False)
    )
    assert rc == 0
    assert captured.payload["status"] == "success"
    assert captured.payload["document"]["kind"] == "ipfs_datasets_py.hf_inference_model_registry"


def test_cli_models_ipld_load_error_path(monkeypatch):
    _install_fake_model_manager(monkeypatch)

    import ipfs_accelerate_py.cli as cli_module

    monkeypatch.setattr(cli_module, "_load_heavy_imports", lambda: None)

    cli = cli_module.IPFSAccelerateCLI()
    captured = _CapturedOutput()
    monkeypatch.setattr(cli, "_print_output", captured.emit)

    rc = cli.run_models_ipld_load(
        SimpleNamespace(cid="bad", backend=None, output_json=True)
    )
    assert rc == 1
    assert captured.payload["status"] == "error"
    assert captured.payload["cid"] == "bad"


def test_mcp_tools_hf_ipld_functions_and_registration(monkeypatch):
    _install_fake_model_manager(monkeypatch)

    from ipfs_accelerate_py.mcp.tools import models as model_tools

    doc = model_tools.build_hf_inference_ipld_document_tool(model_kind="embedding", include_generated_at=False)
    assert doc["status"] == "success"
    assert doc["document"]["kind"] == "ipfs_datasets_py.hf_inference_model_registry"

    cid = model_tools.get_hf_inference_ipld_cid_tool(model_kind="embedding")
    assert cid["status"] == "success"
    assert cid["cid"] == "bafkreifakecid"

    pub = model_tools.publish_hf_inference_ipld_to_ipfs_tool(model_kind="embedding", pin=True, backend="helia")
    assert pub["status"] == "success"
    assert pub["ipfs_cid"] == "bafkreiipfs"

    loaded = model_tools.load_hf_inference_ipld_from_ipfs_tool("bafyok", backend="helia")
    assert loaded["status"] == "success"
    assert loaded["document"]["kind"] == "ipfs_datasets_py.hf_inference_model_registry"

    class _FakeMCP:
        def __init__(self):
            self.tools = {}

        def register_tool(self, *, name, function, description, input_schema, execution_context):
            self.tools[name] = {
                "function": function,
                "description": description,
                "input_schema": input_schema,
                "execution_context": execution_context,
            }

    fake = _FakeMCP()
    model_tools.register_model_tools(fake)

    for tool_name in [
        "build_hf_inference_ipld_document",
        "get_hf_inference_ipld_cid",
        "publish_hf_inference_ipld_to_ipfs",
        "load_hf_inference_ipld_from_ipfs",
    ]:
        assert tool_name in fake.tools
