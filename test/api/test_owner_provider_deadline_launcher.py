"""Actual stream/launcher transport with inert children and original deadlines.

The executable fixture authors only the installed wrapper path. These tests
use the production stream, launcher, sealed transport and actual Popen; no
provider, target code, native database or scheduler construction is involved.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_provider_deadline as deadline_module
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime as runtime
from test.api.test_owner_provider_deadline import _closed, _sealed, executable  # noqa: F401


def test_real_stream_and_launcher_deliver_original_deadline_to_target_cwd_and_close_only_owned_fd(executable, monkeypatch, tmp_path):
    target = tmp_path / "actual-linked-target"
    target.mkdir(mode=0o700)
    parent_cwd = Path.cwd()
    original_deadline = time.monotonic() + 30
    borrowed = _sealed(b"Authored retained archive descriptor.")
    # This inert wrapper additionally reports only whether its explicit archive
    # descriptor survived the native launcher; no descriptor contents are logged.
    script = executable.read_text()
    script = script.replace("print(json.dumps(result,sort_keys=True))",
        "result['borrowed_fd_usable']=os.pread(int(sys.argv[2]),128,0)==b'Authored retained archive descriptor.'\n"
        "print(json.dumps(result,sort_keys=True))")
    executable.write_text(script)
    issued, children = [], []
    actual_popen = subprocess.Popen
    def popen(command, **kwargs):
        assert kwargs["cwd"] == target
        descriptor = int(kwargs["env"][deadline_module.DEADLINE_FD_ENV])
        assert set(kwargs["pass_fds"]) == {borrowed, descriptor}
        assert not _closed(descriptor)
        issued.append(descriptor)
        child = actual_popen(command, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, "Popen", popen)
    def started(child):
        assert child is children[0]
        assert _closed(issued[0]) and not _closed(borrowed)
    output = tmp_path / "actual-stream-output.json"
    try:
        with output.open("wb") as stream:
            completed = runtime.run_process_group_stream([str(executable), "success", str(borrowed)],
                cwd=target, stdout=stream, env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"},
                inherit_environment=False, timeout_seconds=30, absolute_deadline=original_deadline,
                pass_fds=(borrowed,), on_started=started)
        assert completed.returncode == 0 and len(children) == len(issued) == 1
        numerator, denominator = original_deadline.as_integer_ratio()
        assert json.loads(output.read_bytes()) == dict(ok=True,
            deadline_ns=numerator * 1_000_000_000 // denominator, fd_closed=True,
            private_environment_removed=True, second_consume=None, borrowed_fd_usable=True)
        assert _closed(issued[0]) and not _closed(borrowed)
        assert Path.cwd() == parent_cwd
        with pytest.raises(ProcessLookupError): os.killpg(children[0].pid, 0)
    finally:
        for child in children:
            if child.poll() is None: child.kill()
            child.wait(timeout=5)
        os.close(borrowed)


@pytest.mark.parametrize("inherited_deadline", [False, True])
def test_real_generic_stream_keeps_no_private_descriptor_or_environment(monkeypatch, tmp_path, inherited_deadline):
    actual_popen = subprocess.Popen
    observed = []
    def popen(command, **kwargs):
        assert deadline_module.DEADLINE_FD_ENV not in kwargs["env"]
        assert deadline_module.DEADLINE_SHA256_ENV not in kwargs["env"]
        assert "pass_fds" not in kwargs and "absolute_deadline" not in kwargs
        observed.append(command)
        return actual_popen(command, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", popen)
    code = ("import json,os;from pathlib import Path;"
        "private=any(key.startswith('IPFS_ACCELERATE_PROVIDER_DEADLINE_') for key in os.environ);"
        "targets=[];"
        "\nfor path in Path('/proc/self/fd').iterdir():"
        "\n try:targets.append(os.readlink(path))"
        "\n except FileNotFoundError:pass"
        "\nprint(json.dumps(dict(private_environment_present=private,"
        "deadline_descriptor_present=any('memfd:owner-provider-deadline' in value for value in targets))))")
    output = tmp_path / "actual-generic-stream.json"
    with output.open("wb") as stream:
        completed = runtime.run_process_group_stream([sys.executable, "-I", "-c", code],
            cwd=tmp_path, stdout=stream, env={"PATH": os.defpath}, inherit_environment=False,
            timeout_seconds=10, absolute_deadline=time.monotonic() + 10 if inherited_deadline else None)
    assert completed.returncode == 0 and len(observed) == 1
    assert json.loads(output.read_bytes()) == dict(private_environment_present=False,
        deadline_descriptor_present=False)


@pytest.mark.parametrize("collision", ["fd", "digest"])
def test_launcher_refuses_caller_private_environment_before_actual_popen(executable, monkeypatch, collision):
    key = deadline_module.DEADLINE_FD_ENV if collision == "fd" else deadline_module.DEADLINE_SHA256_ENV
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs:
        pytest.fail("private deadline environment collision reached Popen"))
    with pytest.raises(ValueError, match="created by the launcher"):
        runtime.launch_process_child([str(executable), "success"], cwd=Path.cwd(),
            env={key: "authored foreign value"}, inherit_environment=False,
            absolute_deadline=time.monotonic() + 30)


def test_launcher_construction_failure_closes_issued_descriptor_and_preserves_original_error(executable, monkeypatch):
    descriptors = []
    original_error = OSError("authored Popen construction refusal")
    def refuse(command, **kwargs):
        descriptor = int(kwargs["env"][deadline_module.DEADLINE_FD_ENV])
        descriptors.append(descriptor)
        assert not _closed(descriptor) and kwargs["pass_fds"] == (descriptor,)
        raise original_error
    monkeypatch.setattr(subprocess, "Popen", refuse)
    with pytest.raises(OSError) as raised:
        runtime.launch_process_child([str(executable), "success"], cwd=Path.cwd(),
            env={"PATH": os.defpath}, inherit_environment=False,
            absolute_deadline=time.monotonic() + 30)
    assert raised.value is original_error
    assert len(descriptors) == 1 and _closed(descriptors[0])
