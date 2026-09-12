"""Large launch references stay bounded without weakening observation admission."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading

import pytest

from scripts import run_agent_supervisor_efficiency_state_hardening as operator
from ipfs_accelerate_py.agent_supervisor.runtime import owner_observation_request as observation


@pytest.fixture
def launch_reference(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    directory = evidence / "control-plane"
    directory.mkdir(parents=True, mode=0o700)
    evidence.chmod(0o700)
    path = directory / "owner-launch.json"
    admission = {"source_transition_history": [{"retained_evidence": "x" * 2_500_000}]}
    identity = {"store_id": "disposable-launch-reader", "generation": 1}
    scope = observation.observation_scope(
        program_id=operator.PROGRAM, owner_identity=identity,
        source_head="a" * 40, source_tree="b" * 40,
        launch_admission_id=operator._identity(admission),
    )
    launch = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-owner-launch@1",
        "identity": identity, "materialized_launch_admission": admission,
        "observation_request": {"scope": scope, "authenticated_observation": False,
                                "completion_authority": False, "mutation_authority": False},
    }
    def write():
        launch.pop("receipt_cid", None)
        launch["receipt_cid"] = operator._identity(launch)
        path.write_text(json.dumps(launch, sort_keys=True, indent=2))
        path.chmod(0o600)
    write()
    assert path.stat().st_size > operator.STATUS_RECEIPT_MAX_BYTES
    monkeypatch.setattr(operator, "_load", lambda _: (object(), {}))
    monkeypatch.setattr(operator, "_paths", lambda _: {"evidence": evidence})
    return path, launch, write


def no_request(monkeypatch):
    def refuse(_):
        pytest.fail("rejected launch reference reached the transport")
    monkeypatch.setattr(observation, "request_observation", refuse)


def test_large_launch_reference_reaches_exact_native_observation_peer(launch_reference):
    path, launch, _ = launch_reference
    listener = observation.OwnerObservationRequests(launch["observation_request"]["scope"])
    stop = threading.Event()
    def poll():
        while not stop.wait(0.005):
            listener.poll(observer_available=True)
    worker = threading.Thread(target=poll)
    worker.start()
    try:
        result, acknowledgment = operator.request_status_refresh(Path("fixture-config"))
        assert result == 0 and acknowledgment["accepted"]
        assert listener.pending.is_set()
        assert all(acknowledgment[key] is False for key in
                   ("authenticated_observation", "completion_authority", "mutation_authority"))
        assert path.stat().st_size > 2_481_538
    finally:
        stop.set()
        worker.join(timeout=2)
        listener.close()
    assert not worker.is_alive()


def test_large_legacy_launch_still_has_no_observation_route(launch_reference, monkeypatch):
    _, launch, write = launch_reference
    del launch["observation_request"]
    write()
    no_request(monkeypatch)
    with pytest.raises(operator.OperatorError, match="no observation request route"):
        operator.request_status_refresh(Path("fixture-config"))


@pytest.mark.parametrize("mutation,match", [
    ("schema", "launch reference differs"),
    ("receipt", "launch reference differs"),
    ("authority", "no observation request route"),
    ("scope_identity", "owner scope differs"),
    ("scope_admission", "owner scope differs"),
])
def test_large_launch_preserves_content_and_scope_gates(launch_reference, monkeypatch, mutation, match):
    path, launch, write = launch_reference
    if mutation == "schema":
        launch["schema"] = "forged"
    elif mutation == "receipt":
        launch["identity"]["generation"] += 1
    elif mutation == "authority":
        launch["observation_request"]["mutation_authority"] = True
    elif mutation == "scope_identity":
        launch["observation_request"]["scope"]["owner_identity"] = {"generation": 99}
    else:
        launch["observation_request"]["scope"]["launch_admission_id"] = "sha256:wrong"
    if mutation == "receipt":
        path.write_text(json.dumps(launch))
    else:
        write()
    no_request(monkeypatch)
    with pytest.raises(operator.OperatorError, match=match):
        operator.request_status_refresh(Path("fixture-config"))


@pytest.mark.parametrize("kind", ["mode", "hardlink", "symlink", "fifo"])
def test_large_launch_preserves_file_identity_gates(launch_reference, monkeypatch, kind):
    path, _, _ = launch_reference
    if kind == "mode":
        path.chmod(0o640)
    elif kind == "hardlink":
        os.link(path, path.with_name("external-link.json"))
    else:
        retained = path.with_name("retained.json")
        path.rename(retained)
        if kind == "symlink":
            path.symlink_to(retained)
        else:
            os.mkfifo(path, mode=0o600)
    no_request(monkeypatch)
    with pytest.raises((operator.OperatorError, OSError)):
        operator.request_status_refresh(Path("fixture-config"))


def test_large_launch_replacement_during_read_is_rejected(launch_reference, monkeypatch):
    path, _, _ = launch_reference
    retained = path.stat()
    replacement = path.with_name("replacement.json")
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o600)
    real_read = os.read
    changed = False
    def replace_after_read(fd, count):
        nonlocal changed
        data = real_read(fd, count)
        current = os.fstat(fd)
        if not changed and (current.st_dev, current.st_ino) == (retained.st_dev, retained.st_ino):
            changed = True
            os.replace(replacement, path)
        return data
    monkeypatch.setattr(operator.os, "read", replace_after_read)
    no_request(monkeypatch)
    with pytest.raises(operator.OperatorError, match="changed during read"):
        operator.request_status_refresh(Path("fixture-config"))
    assert changed


def test_ordinary_status_reader_retains_its_smaller_bound(launch_reference):
    path, _, _ = launch_reference
    assert operator.STATUS_RECEIPT_MAX_BYTES == 1_048_576
    with pytest.raises(operator.OperatorError, match="identity is unsafe"):
        operator._secure_runtime_json(path, max_bytes=operator.STATUS_RECEIPT_MAX_BYTES)


def test_launch_reference_rejects_file_above_its_separate_bound(launch_reference, monkeypatch):
    path, _, _ = launch_reference
    with path.open("r+b") as handle:
        handle.truncate(operator.OWNER_LAUNCH_REFERENCE_MAX_BYTES + 1)
    no_request(monkeypatch)
    with pytest.raises(operator.OperatorError, match="identity is unsafe"):
        operator.request_status_refresh(Path("fixture-config"))
