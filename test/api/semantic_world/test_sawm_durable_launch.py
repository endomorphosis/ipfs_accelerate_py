"""Native launch delegation happens before configuration or credential effects."""

import importlib.util
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import durable_launch


def test_detached_native_launch_delegates_before_config(monkeypatch):
    script = Path(__file__).resolve().parents[3] / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py"
    spec = importlib.util.spec_from_file_location("sawm_durable_launch_test", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    commands = []

    def delegate(command):
        commands.append(command)
        return 78

    monkeypatch.setattr(durable_launch, "delegate_repair_service_launch", delegate)
    monkeypatch.setattr(module, "_config", lambda *args: pytest.fail("native admission before scope delegation"))
    before = list(sys.path)
    arguments = ["--config", "config with spaces.json", "launch", "--duration-seconds", "120"]
    assert module.main(arguments) == 78
    assert commands == [[sys.executable, str(script), *arguments]]
    assert sys.path == before
