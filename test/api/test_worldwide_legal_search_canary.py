"""Canary worker scope checks; execution evidence requires the real run CLI."""
import json
import fcntl
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_canary as canary
from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_runtime as native
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_canary import _worker, run_canary


def test_canary_refuses_non_temporary_parent():
    with pytest.raises(ValueError, match="temporary directory"):
        run_canary(parent_dir=Path.home())


def test_provider_stub_refuses_non_canary_worktree(tmp_path, monkeypatch):
    spec = tmp_path / "stub.json"
    spec.write_text(json.dumps({"schema": "wwl.repair-canary-provider-stub@1", "fixture_root": str(tmp_path / "fixture"), "state_root": str(tmp_path / "state")}))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="non-canary native worktree"):
        _worker(spec)


def refresh_fixture(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    destination = tmp_path / "repair-canary.json"
    config_path.write_text(json.dumps({"repair_canary_receipt": str(destination)}))
    config = SimpleNamespace(state_root=tmp_path, config_path=config_path)
    monkeypatch.setattr(native, "config_from_args", lambda _args: config)
    return config_path, destination


def test_refresh_refuses_pipeline_drift_without_replacing_receipt(tmp_path, monkeypatch):
    config_path, destination = refresh_fixture(tmp_path, monkeypatch)
    destination.write_text('{"existing": "preserve"}\n')
    monkeypatch.setattr(native, "verified_repair_canary", lambda _config: None)
    monkeypatch.setattr(native, "repair_pipeline_binding", lambda _config: {"generation": "current"})
    monkeypatch.setattr(canary, "run_canary", lambda **_kwargs: {"pipeline_binding": {"generation": "old"}})
    with pytest.raises(RuntimeError, match="changed before"):
        canary.refresh_canary(config_path)
    assert destination.read_text() == '{"existing": "preserve"}\n'


def test_refresh_refuses_duplicate_owner_before_any_canary(tmp_path, monkeypatch):
    config_path, destination = refresh_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(canary, "run_canary", lambda **_kwargs: pytest.fail("duplicate owner executed canary"))
    with (tmp_path / ".repair-canary-refresh.lock").open("a+") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already has an owner"):
            canary.refresh_canary(config_path)
    assert not destination.exists()


def test_runtime_refresh_never_recurses_inside_canary_fixture(monkeypatch):
    runtime = object.__new__(native.CampaignRuntime)
    runtime.config = SimpleNamespace(repair_command=())
    monkeypatch.setattr(native, "verified_repair_canary", lambda _config: pytest.fail("disabled fixture inspected refresh state"))
    runtime._maybe_refresh_repair_canary()


def test_runtime_refresh_waits_for_loaded_revision_without_acknowledging_reload(tmp_path, monkeypatch):
    runtime = object.__new__(native.CampaignRuntime)
    runtime.config = SimpleNamespace(repair_command=("repair",), framework_root=tmp_path)
    runtime.stop, runtime.cancel = threading.Event(), threading.Event()
    runtime._canary_refresh_lock = threading.Lock()
    runtime._canary_refresh_stopping = False
    runtime._canary_refresh_next_at = 0
    runtime._canary_refresh_thread = None
    runtime._loaded_framework_revision = "old-loaded-revision"
    monkeypatch.setattr(native, "git", lambda *_args: "promoted-revision-requiring-reload")
    monkeypatch.setattr(native, "verified_repair_canary", lambda _config: pytest.fail("unreloaded owner attempted canary"))
    runtime._maybe_refresh_repair_canary()
    assert runtime._loaded_framework_revision == "old-loaded-revision"
    assert runtime._canary_refresh_thread is None


def test_optional_refresh_shutdown_never_waits_for_subprocess(monkeypatch):
    runtime = object.__new__(native.CampaignRuntime)
    runtime._canary_refresh_lock = threading.Lock()
    runtime._canary_refresh_process = SimpleNamespace(pid=12345, poll=lambda: None)
    signals = []
    monkeypatch.setattr(native.os, "killpg", lambda pid, signal: signals.append((pid, signal)))
    runtime._stop_repair_canary_refresh()
    assert runtime._canary_refresh_stopping is True
    assert signals == [(12345, native.signal.SIGKILL)]


def test_canary_restores_host_catalog_bindings_after_failure(monkeypatch):
    keys = ("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", "IPFS_ACCELERATE_AGENT_HOME")
    for key in keys:
        monkeypatch.setenv(key, "/host-owned/" + key.lower())
    before = {key: os.environ[key] for key in keys}

    def failed_fixture(**_kwargs):
        for key in keys:
            os.environ[key] = "/tmp/isolated-fixture/" + key.lower()
        raise RuntimeError("fixture failed")

    monkeypatch.setattr(canary, "_run_canary", failed_fixture)
    with pytest.raises(RuntimeError, match="fixture failed"):
        canary.run_canary()
    assert {key: os.environ[key] for key in keys} == before
