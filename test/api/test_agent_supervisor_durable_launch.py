from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import durable_launch


@pytest.mark.parametrize("cgroup,expected", [
    ("0::/user.slice/app.slice/ipfs-taskboard-repair-job.service\n", True),
    ("1:name=systemd:/user.slice/ipfs-taskboard-repair-job-test.service\n", True),
    ("0::/user.slice/app.slice/ipfs-agent-runtime-123.scope\n", False),
    ("0::/user.slice/app.slice/ipfs-taskboard-repair.service\n", False),
    ("0::/user.slice/app.slice/not-ipfs-taskboard-repair-job.service\n", False),
])
def test_exact_temporary_service_detection(monkeypatch, cgroup, expected):
    monkeypatch.setattr(durable_launch.sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_text", lambda self: cgroup)
    assert durable_launch.in_temporary_repair_service() is expected


def test_scope_preserves_native_command_and_propagates_failure(monkeypatch):
    monkeypatch.setattr(durable_launch, "in_temporary_repair_service", lambda: True)
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=78)
    monkeypatch.setattr(durable_launch.subprocess, "run", run)
    command = ["/usr/bin/python3", "/repo with spaces/operator.py", "launch"]
    assert durable_launch.delegate_repair_service_launch(command) == 78
    argv, kwargs = calls[0]
    assert argv[0] == "/usr/bin/systemd-run"
    assert "--scope" in argv and "--collect" in argv
    assert argv[argv.index("--") + 1:] == command
    assert kwargs == {"check": False}


def test_existing_runtime_scope_is_not_redelegated(monkeypatch):
    monkeypatch.setattr(durable_launch, "in_temporary_repair_service", lambda: False)
    monkeypatch.setattr(durable_launch.subprocess, "run", lambda *a, **kw: pytest.fail("duplicate launch"))
    assert durable_launch.delegate_repair_service_launch(["native", "launch"]) is None


def test_unavailable_cgroup_does_not_launch(monkeypatch):
    monkeypatch.setattr(durable_launch.sys, "platform", "linux")
    def unavailable(self):
        raise OSError("unavailable")
    monkeypatch.setattr(Path, "read_text", unavailable)
    with pytest.raises(RuntimeError, match="service lifetime"):
        durable_launch.delegate_repair_service_launch(["native", "launch"])
