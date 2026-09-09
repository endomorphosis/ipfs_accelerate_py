"""Resource-only supervisor backport; preserve hashing and capsule authority."""

from __future__ import annotations

import hashlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py import _hash_resources as resources
from ipfs_accelerate_py import agent_implementation_route as route
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runner


@pytest.fixture
def isolated_lock(tmp_path, monkeypatch):
    lock = tmp_path / "resource.lock"
    monkeypatch.setattr(resources, "hash_lock_path", lambda kind="file-hash": lock)
    monkeypatch.setattr(resources, "host_hash_pressure", lambda: (2, "admitted"))
    monkeypatch.setenv("IPFS_HASH_LOCK_TIMEOUT_SECONDS", "0.1")
    return lock


def test_fingerprint_keeps_legacy_path_mode_bytes_and_links(tmp_path, isolated_lock):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    directory = workspace / "directory"
    directory.mkdir()
    payload = b"x" * (1024 * 1024 + 7)
    binary = directory / "binary"
    binary.write_bytes(payload)
    link = workspace / "link"
    link.symlink_to("directory/binary")
    expected = hashlib.sha256()
    for path, kind, content in (
        (directory, b"D", b""),
        (link, b"L", b"directory/binary"),
        (binary, b"F", payload),
    ):
        expected.update(path.relative_to(workspace).as_posix().encode())
        expected.update(b"\0")
        expected.update(str(path.lstat().st_mode).encode("ascii"))
        expected.update(b"\0")
        expected.update(kind + content + b"\0")
    assert runner._workspace_content_fingerprint(workspace) == expected.hexdigest()
    binary.write_bytes(b"y" + payload[1:])
    assert runner._workspace_content_fingerprint(workspace) != expected.hexdigest()


def test_fingerprint_enters_global_gate_before_reading(monkeypatch, tmp_path):
    events = []

    @contextmanager
    def admission(**kwargs):
        assert kwargs == {"kind": "workspace-fingerprint", "exclusive": True}
        events.append("entered")
        try:
            yield 1
        finally:
            events.append("released")

    def fingerprint(workspace):
        assert workspace == tmp_path
        assert events == ["entered"]
        return "digest"

    monkeypatch.setattr(runner, "hashing_lock", admission)
    monkeypatch.setattr(runner, "_workspace_content_fingerprint_unlocked", fingerprint)
    assert runner._workspace_content_fingerprint(tmp_path) == "digest"
    assert events == ["entered", "released"]


def test_busy_gate_does_not_fallback_to_unadmitted_hashing(monkeypatch, tmp_path):
    @contextmanager
    def admission(**kwargs):
        raise resources.HashingResourceTimeout("busy")
        yield 1

    monkeypatch.setattr(runner, "hashing_lock", admission)
    monkeypatch.setattr(
        runner, "_workspace_content_fingerprint_unlocked",
        lambda workspace: pytest.fail("hashing escaped admission"),
    )
    with pytest.raises(resources.HashingResourceTimeout, match="busy"):
        runner._workspace_content_fingerprint(tmp_path)


def test_capsule_requires_stdlib_resource_module():
    assert "ipfs_accelerate_py/_hash_resources.py" in route._AGENT_CONTROL_PLANE_RELATIVE_FILES


def test_isolated_capsule_contains_and_loads_resource_guard(tmp_path):
    import importlib.util
    import subprocess

    helper_path = Path(__file__).with_name(
        "test_agent_supervisor_control_plane_capsule_identity.py"
    )
    spec = importlib.util.spec_from_file_location("_hash_capsule_test_helpers", helper_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    source = helpers._clean_control_plane_source(tmp_path / "source")
    pin = helpers._real_materialized_pin(source, tmp_path / "capsules")
    sealed = helpers.llm_router.seal_agent_implementation_control_plane_capsule(pin)
    bootstrap = """
import importlib
import importlib.machinery
import sys
import types
archive = sys.argv[1]
sys.path.insert(0, archive)
import ipfs_accelerate_py
name = "ipfs_accelerate_py.agent_supervisor"
package = types.ModuleType(name)
package.__file__ = archive + "/ipfs_accelerate_py/agent_supervisor/__init__.py"
package.__package__ = name
package.__path__ = [archive + "/ipfs_accelerate_py/agent_supervisor"]
package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
sys.modules[name] = package
ipfs_accelerate_py.agent_supervisor = package
resources = importlib.import_module("ipfs_accelerate_py._hash_resources")
pressure = importlib.import_module(name + ".runtime.hash_pressure")
assert pressure.hashing_lock is resources.hashing_lock
assert resources.hash_lock_path().parent.as_posix() == "/tmp"
for module in (resources, pressure):
    assert module.__file__.startswith(archive + "/"), module.__file__
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", bootstrap, sealed.executable_path],
            cwd=tmp_path, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "/untrusted"},
            pass_fds=(sealed.descriptor,), capture_output=True, text=True, timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
    finally:
        os.close(sealed.descriptor)


def test_capsule_rejects_resource_module_from_another_root(monkeypatch, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("# unrelated source\n")
    monkeypatch.setitem(
        sys.modules, "ipfs_accelerate_py._hash_resources",
        SimpleNamespace(__file__=str(outside)),
    )
    root = Path(route.__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="crossed capsule roots"):
        route._agent_control_plane_source_files(root)


def test_resource_controls_are_bound_in_positive_lane_environment():
    from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner

    controls = {
        "IPFS_HASH_MAX_WORKERS": "1",
        "IPFS_HASH_LOCK_TIMEOUT_SECONDS": "20",
    }
    ambient = {**controls, "UNRELATED_SECRET": "do-not-forward"}
    profile = dict(multi_supervisor_runner._plan_bound_profile_environment(ambient))
    child = multi_supervisor_runner._plan_bound_positive_child_environment(ambient)
    assert profile == controls
    assert all(child.get(name) == value for name, value in controls.items())
    assert "UNRELATED_SECRET" not in child
