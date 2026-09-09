"""The resource gate must preserve complete mutable-workspace verification."""

from __future__ import annotations

import ast
import base64
import hashlib
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner, multi_supervisor_runner

from ipfs_accelerate_py import _hash_resources, llm_router


def test_workspace_fingerprint_preserves_legacy_byte_stream(tmp_path: Path) -> None:
    directory = tmp_path / "d"
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    file = directory / "a"
    file.write_bytes(b"payload")
    file.chmod(0o644)
    (tmp_path / "link").symlink_to("d/a")
    # Fixed legacy encoding: sorted directories, sorted files, then descendants;
    # all path names, types, modes, content bytes and link targets are committed.
    preimage = (
        b"d\0" + b"16877\0D\0"
        b"link\0" + b"41471\0Ld/a\0"
        b"d/a\0" + b"33188\0Fpayload\0"
    )
    assert grok_cli_runner._workspace_content_fingerprint(tmp_path) == (
        hashlib.sha256(preimage).hexdigest()
    )


def test_same_size_same_mtime_edit_is_still_fully_verified(tmp_path: Path) -> None:
    file = tmp_path / "source.py"
    file.write_bytes(b"a = 1\n")
    metadata = file.stat()
    before = grok_cli_runner._workspace_content_fingerprint(tmp_path)
    file.write_bytes(b"a = 2\n")
    os.utime(file, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    assert file.stat().st_size == metadata.st_size
    assert file.stat().st_mtime_ns == metadata.st_mtime_ns
    assert grok_cli_runner._workspace_content_fingerprint(tmp_path) != before


def test_hidden_files_and_modes_remain_in_fingerprint(tmp_path: Path) -> None:
    file = tmp_path / ".hidden"
    file.write_bytes(b"data")
    file.chmod(0o600)
    before = grok_cli_runner._workspace_content_fingerprint(tmp_path)
    file.chmod(0o700)
    assert grok_cli_runner._workspace_content_fingerprint(tmp_path) != before
    file.chmod(0o600)
    assert grok_cli_runner._workspace_content_fingerprint(tmp_path) == before
    file.unlink()
    assert grok_cli_runner._workspace_content_fingerprint(tmp_path) != before


@pytest.mark.parametrize("payload", [b"", b"hello", bytes(range(256))])
def test_router_emits_raw_cid_without_optional_packages(monkeypatch, payload) -> None:
    monkeypatch.setitem(sys.modules, "multiformats", None)
    cid = llm_router._cid_for_bytes(payload)
    assert cid.startswith("b")
    encoded = cid[1:].upper()
    binary = base64.b32decode(encoded + "=" * (-len(encoded) % 8))
    assert binary == b"\x01\x55\x12\x20" + hashlib.sha256(payload).digest()
    if not payload:
        assert cid == "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"


def test_bootstrap_and_runtime_serialize_on_the_same_lock() -> None:
    # Run the real pre-import bootstrap gate in an isolated process. It cannot
    # import project modules until it has verified the capsule containing them.
    tree = ast.parse(multi_supervisor_runner.SEALED_CONTROL_PLANE_BOOTSTRAP)
    prefix = ast.Module(
        body=[
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            or isinstance(node, ast.FunctionDef) and node.name == "_hash_budget"
        ],
        type_ignores=[],
    )
    code = ast.unparse(prefix) + '''
print("ready", flush=True)
try:
    with _hash_budget():
        print("admitted", flush=True)
        raise RuntimeError("exercise cleanup")
except RuntimeError:
    pass
'''
    child = None
    try:
        with _hash_resources.hashing_lock(kind="workspace-fingerprint"):
            child = subprocess.Popen(
                [sys.executable, "-I", "-S", "-u", "-c", code],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            assert select.select([child.stdout], [], [], 5)[0]
            assert child.stdout.readline() == b"ready\n"
            assert not select.select([child.stdout], [], [], 0.15)[0]
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == 0, stderr.decode(errors="replace")
        assert stdout == b"admitted\n"
        # An exception inside the bootstrap gate must release it for successors.
        with _hash_resources.hashing_lock(timeout=0.5):
            pass
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate()


def test_hash_policy_survives_positive_supervisor_environment_projection() -> None:
    policy = {
        "IPFS_HASH_CACHE_TTL_SECONDS": "86400",
        "IPFS_HASH_MAX_WORKERS": "2",
        "IPFS_HASH_LOCK_TIMEOUT_SECONDS": "60",
    }
    projected = multi_supervisor_runner._plan_bound_positive_child_environment({
        **policy, "LD_PRELOAD": "/hostile.so", "UNDECLARED_HASH_SETTING": "bad",
    })
    assert all(projected.get(name) == value for name, value in policy.items())
    assert "LD_PRELOAD" not in projected
    assert "UNDECLARED_HASH_SETTING" not in projected


def test_trusted_git_uses_shared_observation_but_strict_receipts_read_bytes(
    monkeypatch, tmp_path: Path,
) -> None:
    import threading

    from ipfs_accelerate_py.agent_supervisor.runtime import shared_hashing
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_duckdb_connection
    from ipfs_accelerate_py.agent_supervisor.task_sources.hash_observations import (
        HashObservationStore,
    )
    from test.api.causal_federation.test_typed_state_owner import _install
    from test.api.test_agent_supervisor_configured_typed_grant_handoff import aseh_operator

    database = tmp_path / "hashes.duckdb"
    _install(database)
    connection = open_duckdb_connection(database)
    try:
        store = HashObservationStore(connection, generation="test-owner:1")
        mutex = threading.Lock()

        class Client:
            def __init__(self, principal):
                self.principal = principal

            def hash_observation(self, request):
                with mutex:
                    return store.handle(request, principal=self.principal)

        # Another supervisor's completed observation is enough for routine Git
        # checks; this process must still perform all executable admission checks.
        first = shared_hashing.hash_file(aseh_operator.TRUSTED_GIT, connection=Client("other"))
        assert not first.cache_hit
        monkeypatch.setattr(shared_hashing, "default_hash_connection", lambda: Client("this"))
        monkeypatch.setattr(aseh_operator, "_TRUSTED_GIT_IDENTITY", None)
        real_pread = shared_hashing.os.pread
        reads = []

        def counted(descriptor, count, offset):
            payload = real_pread(descriptor, count, offset)
            reads.append(len(payload))
            return payload

        monkeypatch.setattr(shared_hashing.os, "pread", counted)
        assert aseh_operator._trusted_git_executable() == str(aseh_operator.TRUSTED_GIT)
        assert reads == []
        assert aseh_operator._trusted_git_executable(strict=True) == str(aseh_operator.TRUSTED_GIT)
        assert sum(reads) == aseh_operator.TRUSTED_GIT.stat().st_size

        def forbidden_cache(*args, **kwargs):
            raise AssertionError("receipt verification must not use the metadata cache")

        monkeypatch.setattr(aseh_operator, "hash_descriptor", forbidden_cache)
        monkeypatch.setattr(aseh_operator, "_ASEH_RECEIPT_VALIDATION_PYTHON_IDENTITY", None)
        monkeypatch.setattr(aseh_operator.sys, "executable", "/usr/bin/python3.12")
        assert aseh_operator._trusted_receipt_validation_python() == "/usr/bin/python3"
    finally:
        connection.close()
