"""The embedded probe releases bulk data while keeping cold and warm checks."""
import builtins
import gc
import hashlib
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import weakref

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as probe
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_resource_diagnostics as diagnostics
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as owner


@pytest.mark.parametrize("fault", [None, "digest", "model_loads", "authority", "warm_replay"])
def test_probe_releases_bulk_inference_before_independent_warm_replay(
        tmp_path, monkeypatch, capsys, fault):
    # Native/Docker calls are explicit doubles; the actual generated probe runs.
    def path(value):
        value = str(value)
        if value.startswith("/opt/ipfs-supervisor/") or value == "/app":
            return tmp_path / value.lstrip("/")
        return Path(value)

    path(probe.RESULT_PATH).parent.mkdir(parents=True)
    native = dict(native_worker_executed=True, inference_executed=True,
                  report=dict(output=dict(model_loads=2 if fault == "model_loads" else 1,
                                          rows=[{"bulk": "authored candidate"}]),
                              preparation={"bulk": "authored source map"},
                              worker_receipt={"input_sha256": "input"}, key={"source_head": "head"}))
    encoded = json.dumps(native).encode()
    receipt = dict(schema="terminal-source384-repository-context@1", output=str(tmp_path / "inference"),
                   inference_sha256="wrong" if fault == "digest" else hashlib.sha256(encoded).hexdigest(),
                   config_path=str(path("/opt/ipfs-supervisor/models/source384/config.json")),
                   checkpoint_sha256="checkpoint", config_sha256="config", source_head={"head": 1},
                   source_hashes={"authored.py": "source"}, summary={"coverage": {"rows": 1}},
                   resource_profile={"authored": True}, seconds=1,
                   proof_authority=fault == "authority", execution_authority=False,
                   completion_authority=False, formalization_authority=False)
    context = dict(source384_context=receipt, seconds=2, nonoverlapping_seconds={"context": 2})
    released, roots, warm = [], [], []

    class BulkBytes(bytes):
        def __del__(self):
            released.append("raw")

    class BulkInference(dict):
        pass

    original_loads = json.loads

    def loads(value, *args, **kwargs):
        decoded = original_loads(value, *args, **kwargs)
        if isinstance(value, BulkBytes):
            decoded = BulkInference(decoded)
            roots.append(weakref.ref(decoded))
        return decoded

    def validate(**kwargs):
        warm.append(kwargs)
        gc.collect()
        assert released == ["raw"] and roots[0]() is None
        assert kwargs["expected_receipt"] is receipt
        if fault == "warm_replay":
            raise ValueError("authored warm source drift")

    monkeypatch.setattr(prep, "prepare", lambda **kw: {})
    monkeypatch.setattr(prep, "initial_context", lambda **kw: context)
    monkeypatch.setattr(owner, "_pins", lambda: {"authored": True})
    monkeypatch.setattr(owner, "_read", lambda *args: BulkBytes(encoded))
    monkeypatch.setattr(owner, "validate_source384_context", validate)
    monkeypatch.setattr(diagnostics, "collect_failure_scheduler", lambda: {"status": "authored_control"})
    monkeypatch.setattr(json, "loads", loads)
    monkeypatch.setattr(sys, "argv", ["authored-probe"])
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(signal, "setitimer", lambda *args: None)
    original_import = builtins.__import__

    def imports(name, *args, **kwargs):
        if name == "pathlib":
            return SimpleNamespace(Path=path)
        return original_import(name, *args, **kwargs)

    namespace = {"__builtins__": {**vars(builtins), "__import__": imports}}
    with pytest.raises(SystemExit) as stopped:
        exec(compile(probe.CONTEXT_PROBE, "source384-probe", "exec"), namespace)
    report = original_loads(path(probe.RESULT_PATH).read_bytes())
    assert stopped.value.code == (0 if fault is None else 1)
    assert report["qualified"] is (fault is None)
    assert report["provider_calls"] == 0 and report["benchmark_result"] is False
    assert len(warm) == int(fault in {None, "warm_replay"})
    if fault is None:
        assert report["native_worker_receipt"] == native["report"]["worker_receipt"]
        assert report["native_inference_key"] == native["report"]["key"]
        assert report["inference_sha256"] == receipt["inference_sha256"]
        assert report["neural_inference_replayed"] is False
    else:
        assert report["error_phase"] == ("warm_observation" if fault == "warm_replay" else "initial_context")
    capsys.readouterr()
