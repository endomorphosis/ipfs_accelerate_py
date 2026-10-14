"""Private world input capture stays bounded and cannot change dispatch outcomes."""
import hashlib
import errno
import fcntl
import json
import os
from pathlib import Path
import stat

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_artifact_store as store
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet_replay import freeze_repository_world_packet
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_invocation_replay import replay_repository_world_invocation
from test.api.test_repository_world_dispatch import (
    TASK, dispatch_case, invoke, invocation_receipt, required, scheduler, world_session,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_capture(directory, receipt):
    reference = receipt["world_packet_replay_artifact"]
    assert set(reference) == {"schema", "invocation_id", "artifact_sha256",
                              "artifact_bytes", "packet_artifact_sha256"}
    assert reference["schema"] == "repository-world-invocation-artifact-reference@1"
    assert reference["invocation_id"] == receipt["invocation_id"]
    artifact = (directory / (reference["invocation_id"] + ".json")).read_bytes()
    assert len(artifact) == reference["artifact_bytes"]
    assert sha(artifact) == reference["artifact_sha256"]
    result = replay_repository_world_invocation(artifact,
        expected_sha256=reference["artifact_sha256"])
    return artifact, result


def test_runner_captures_before_edit_and_retains_exact_input_after_cleanup(dispatch_case, tmp_path):
    directory = tmp_path / "world-invocations"
    source, session, packet, state = dispatch_case
    def edit(_prompt, _options):
        path = source / "calc.py"
        path.write_text(path.read_text().replace("return value + 1", "return value + 2"))
    state["callback"] = edit
    _, receipt = invoke(dispatch_case, world_packet_audit_directory=directory,
                        repository_source_policy="repository-source-compact@1")
    assert receipt["repository_source_policy"]["usage"]["store_cleanup_complete"] is True
    with pytest.raises(ValueError):
        freeze_repository_world_packet(packet)
    session.close()
    artifact, restored = read_capture(directory, receipt)
    assert restored.model_prompt == state["seen"][0][0]
    assert restored.receipt()["provider_delivery_verified"] is False
    assert restored.receipt()["live_packet_issued"] is False
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    retained = directory / (receipt["invocation_id"] + ".json")
    assert stat.S_IMODE(retained.stat().st_mode) == 0o600
    assert TASK not in json.dumps(receipt["world_packet_replay_artifact"])
    assert b"Authored provider returned" not in artifact


@pytest.mark.parametrize("failure", [TimeoutError("authored timeout"), RuntimeError("429 rate limit exceeded")])
def test_failed_provider_attempt_retains_input_without_retry(dispatch_case, tmp_path, capsys, failure):
    directory = tmp_path / "world-invocations"
    def fail(_prompt, _options):
        raise failure
    dispatch_case[3]["callback"] = fail
    with pytest.raises(type(failure)):
        invoke(dispatch_case, world_packet_audit_directory=directory)
    receipt = invocation_receipt(capsys)
    assert receipt["status"] == "failed"
    assert receipt["world_task_packet"]["provider_attempted"] is True
    assert len(dispatch_case[3]["seen"]) == 1
    _, restored = read_capture(directory, receipt)
    assert restored.model_prompt == dispatch_case[3]["seen"][0][0]
    assert restored.receipt()["provider_delivery_verified"] is False


def test_late_source_drift_records_undispatched_input(dispatch_case, tmp_path, monkeypatch, capsys):
    from ipfs_accelerate_py import llm_router
    directory = tmp_path / "world-invocations"
    source = dispatch_case[0]
    def initialize(*args, **kwargs):
        (source / "calc.py").write_text("def add(value):\n    return value + 44\n")
        return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    with pytest.raises(ValueError):
        invoke(dispatch_case, world_packet_audit_directory=directory)
    receipt = invocation_receipt(capsys)
    assert receipt["world_task_packet"]["provider_attempted"] is False
    assert receipt["world_task_packet"]["dispatch_rechecked"] is False
    assert not dispatch_case[3]["seen"]
    _, restored = read_capture(directory, receipt)
    assert restored.receipt()["provider_delivery_verified"] is False


@pytest.mark.parametrize("provider_fails", [False, True])
def test_capture_failure_cannot_mask_provider_result(dispatch_case, tmp_path, monkeypatch, capsys, provider_fails):
    def storage_failure(**kwargs):
        raise OSError("private path and diagnostic must stay private")
    monkeypatch.setattr(store, "store_repository_world_invocation_artifact", storage_failure)
    if provider_fails:
        def fail(_prompt, _options):
            raise TimeoutError("original provider failure")
        dispatch_case[3]["callback"] = fail
        with pytest.raises(TimeoutError, match="original provider failure"):
            invoke(dispatch_case, world_packet_audit_directory=tmp_path / "world-invocations")
        receipt = invocation_receipt(capsys)
    else:
        output, receipt = invoke(dispatch_case, world_packet_audit_directory=tmp_path / "world-invocations")
        assert output.startswith("Authored provider")
    assert len(dispatch_case[3]["seen"]) == 1
    assert "world_packet_replay_artifact" not in receipt
    assert type(receipt.get("world_packet_replay_capture")) is dict
    assert "private path" not in json.dumps(receipt)


def test_capture_requires_world_route_before_provider(dispatch_case, tmp_path):
    with pytest.raises(ValueError):
        invoke(dispatch_case, world_task_packet=None,
               world_packet_audit_directory=tmp_path / "world-invocations")
    assert not dispatch_case[3]["seen"]


@pytest.mark.parametrize("case", ["inside", "symlink", "parent_symlink", "file", "fifo", "public_mode"])
def test_private_capture_directory_refuses_unsafe_locations(tmp_path, case):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    directory = tmp_path / "capture"
    if case == "inside":
        directory = workspace / "capture"
    elif case == "symlink":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        directory.symlink_to(target, target_is_directory=True)
    elif case == "parent_symlink":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        link = tmp_path / "parent-link"
        link.symlink_to(target, target_is_directory=True)
        directory = link / "capture"
    elif case == "file":
        directory.write_bytes(b"keep")
    elif case == "fifo":
        os.mkfifo(directory)
    elif case == "public_mode":
        directory.mkdir(mode=0o755)
        directory.chmod(0o755)
    with pytest.raises((ValueError, OSError)):
        store.prepare_repository_world_artifact_directory(directory, workspace=workspace)
    if case == "file":
        assert directory.read_bytes() == b"keep"


@pytest.fixture
def prepared_capture(dispatch_case, tmp_path):
    packet = freeze_repository_world_packet(dispatch_case[2])
    _, receipt = invoke(dispatch_case)
    directory = tmp_path / "world-invocations"
    identity = store.prepare_repository_world_artifact_directory(directory, workspace=dispatch_case[0])
    options = dict(directory=directory, expected_identity=identity, packet_artifact=packet,
                   expected_packet_sha256=sha(packet), invocation_receipt=receipt)
    return directory, receipt, options


def test_exclusive_artifact_write_never_replaces_existing_bytes(prepared_capture):
    directory, receipt, options = prepared_capture
    reference = store.store_repository_world_invocation_artifact(**options)
    path = directory / (receipt["invocation_id"] + ".json")
    retained = path.read_bytes()
    assert reference["artifact_sha256"] == sha(retained)
    with pytest.raises((ValueError, OSError)):
        store.store_repository_world_invocation_artifact(**options)
    assert path.read_bytes() == retained


def test_directory_replacement_refuses_before_artifact_write(prepared_capture):
    directory, _receipt, options = prepared_capture
    directory.rename(directory.with_name("old-capture"))
    directory.mkdir(mode=0o700)
    with pytest.raises((ValueError, OSError)):
        store.store_repository_world_invocation_artifact(**options)
    assert not list(directory.iterdir())


@pytest.mark.parametrize("kind", ["symlink", "fifo", "hardlink"])
def test_artifact_collision_never_follows_or_overwrites(prepared_capture, kind):
    directory, receipt, options = prepared_capture
    path = directory / (receipt["invocation_id"] + ".json")
    original = directory.parent / "untouched"
    original.write_bytes(b"untouched")
    if kind == "symlink":
        path.symlink_to(original)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        os.link(original, path)
    with pytest.raises((ValueError, OSError)):
        store.store_repository_world_invocation_artifact(**options)
    assert original.read_bytes() == b"untouched"


def test_invalid_packet_pin_writes_no_artifact(prepared_capture):
    directory, _receipt, options = prepared_capture
    with pytest.raises(ValueError):
        store.store_repository_world_invocation_artifact(**dict(options, expected_packet_sha256="0"*64))
    assert not list(directory.iterdir())


@pytest.mark.parametrize("partial_bytes", [0, 17])
def test_failed_write_keeps_partial_but_allows_later_capture(prepared_capture, monkeypatch, partial_bytes):
    directory, receipt, options = prepared_capture
    original_write = os.write
    calls = 0

    def fail_write(descriptor, pending):
        nonlocal calls
        calls += 1
        if partial_bytes and calls == 1:
            return original_write(descriptor, pending[:partial_bytes])
        raise OSError(errno.ENOSPC, "authored full storage")

    with monkeypatch.context() as fault:
        fault.setattr(store.os, "write", fail_write)
        with pytest.raises(OSError) as failure:
            store.store_repository_world_invocation_artifact(**options)
        assert failure.value.errno == errno.ENOSPC
    abandoned = directory / (receipt["invocation_id"] + ".json")
    retained = abandoned.read_bytes()
    assert len(retained) == partial_bytes
    assert stat.S_IMODE(abandoned.stat().st_mode) == 0o600
    later = {**receipt, "invocation_id": "f" * 32}
    assert later["invocation_id"] != receipt["invocation_id"]
    reference = store.store_repository_world_invocation_artifact(**{
        **options, "invocation_receipt": later})
    _, restored = read_capture(directory, {**later, "world_packet_replay_artifact": reference})
    assert restored.receipt()["historical_input_reconstructed"] is True
    assert abandoned.read_bytes() == retained
    assert len(list(directory.iterdir())) == 2


@pytest.mark.parametrize("limit", ["count", "bytes"])
def test_store_capacity_refuses_without_discarding_previous_inputs(prepared_capture, monkeypatch, limit):
    directory, receipt, options = prepared_capture
    if limit == "count":
        for number in range(store.MAX_ARTIFACTS):
            path = directory / (format(number, "032x") + ".json")
            path.write_bytes(b"historical input")
            path.chmod(0o600)
    else:
        monkeypatch.setattr(store, "MAX_STORE_BYTES", 1)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(ValueError):
        store.store_repository_world_invocation_artifact(**options)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_contended_store_lock_has_a_deadline(prepared_capture, monkeypatch):
    directory, _receipt, options = prepared_capture
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(store, "LOCK_TIMEOUT_SECONDS", .01)
        with pytest.raises(TimeoutError):
            store.store_repository_world_invocation_artifact(**options)
        assert not list(directory.iterdir())
    finally:
        os.close(descriptor)
