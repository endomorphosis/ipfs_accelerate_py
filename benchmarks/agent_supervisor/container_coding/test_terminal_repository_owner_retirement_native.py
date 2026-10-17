"""Admitted lifecycle -> sealed retirement -> copied offline observation.

Native reservations, source index, sockets, retirement and file custody are
real. Container, peer, task and invocation identities are authored. One case
injects a single release exception; no provider or different-UID deployment
is performed, and scheduler capacity reclamation is not inferred.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_owner_observation as observation
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_native_observation import (
    _row, _wait_for_response_accounting)
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_binding as bindings
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_monitor as monitor
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import _protected_directory
from test.api.test_repository_source_owner_lifecycle import native_owner
from test.api.test_repository_source_parent_admission import native_parent
from test.api.test_repository_source_owner_broker import initialize, tool


@pytest.fixture
def short_service_directory():
    # A short real protected owner directory satisfies both ancestor custody
    # and AF_UNIX's bounded pathname; no shared workspace permissions change.
    with tempfile.TemporaryDirectory(prefix=".rr-", dir=Path.home()) as directory:
        yield Path(directory)


@pytest.mark.parametrize("release_exception", [False, True])
def test_native_retirement_records_own_release_and_preserves_borrowed_parent(
        native_owner, native_parent, monkeypatch, release_exception, request):
    owner = native_owner
    service = owner._broker
    # Restore actual directory validation after the existing fixture's
    # explicitly authored container/worktree/peer setup.
    monkeypatch.setattr(broker.guard, "_protected_directory", _protected_directory)
    monkeypatch.setattr(bindings.protocol, "load_descriptor", lambda *args, **kwargs: copy.deepcopy(service.descriptor))
    binding = bindings.create_binding(workspace=service.workspace,
        requested_policy="repository-source-raw@1", descriptor_path=service.descriptor_path,
        descriptor_sha256=service.descriptor_sha256)
    binding["usage"] = bindings.unavailable_usage()
    row = _row(binding)
    report = dict(repository_source_policy="repository-source-raw@1", provider_invocations=[row],
        worker_cleanup_returncode=0, usage={"total_tokens": None})
    report["public_instruction"] = {key: row["public_instruction"][key] for key in (
        "artifact", "context_cid", "manifest_cid", "task_cid", "source_path", "source_sha256", "source_bytes",
        "completion_authority", "scope_expansion_authority")}
    report["public_instruction"]["sha256"] = row["public_instruction"]["artifact_sha256"]
    scope = dict(task_cid="authored-task-cid", task_alias="authored-task", baseline_commit="d" * 40,
        outputs=[dict(path="calc.py", effect="modify", baseline_sha256="e" * 64)])
    report_before = copy.deepcopy(report)
    initialize(service)
    generation = tool(service, "repository_status")["generation"]
    assert "value + 1" in tool(service, "repository_read", generation=generation, path="calc.py")["content"]
    _wait_for_response_accounting(service, 4)
    calls = []
    if release_exception:
        original_release = owner._lease.release
        def fail_once():
            calls.append(True)
            if len(calls) == 1:
                raise OSError("authored transient release failure")
            return original_release()
        monkeypatch.setattr(owner._lease, "release", fail_once)
    receipt = monitor.retire_repository_source_owner(owner)
    assert receipt["cleanup_complete"] and receipt["native_parent_released"]
    assert receipt["renewal_unproven"] is release_exception
    assert calls == ([True, True] if release_exception else [])
    assert owner._lease.released and service._native.released
    assert not native_parent.released and not native_parent.cancelled
    retained = receipt["retained_lifecycle_receipt"]
    assert retained["status"] == "retained", retained
    retirement = service.audit_path.parent / retained["name"]
    raw = retirement.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == retained["sha256"]
    envelope = json.loads(raw)
    assert envelope["lifecycle"]["native_parent"]["parent_lease_id"] == native_parent.lease_id
    assert envelope["lifecycle"]["broker_audit"]["native_child"]["parent_lease_id"] == owner._lease.lease_id
    # A repeated observation is idempotent and performs no new release/admission.
    identity = retirement.stat().st_ino, retirement.stat().st_mtime_ns
    again = monitor.retire_repository_source_owner(owner)
    assert again["retained_lifecycle_receipt"] == retained
    assert (retirement.stat().st_ino, retirement.stat().st_mtime_ns) == identity
    assert calls == ([True, True] if release_exception else [])
    paths = [service.audit_path, retirement,
        service.audit_path.parent / receipt["broker_audit"]["retained_mcp_audit"]["name"]]
    originals = {path: path.read_bytes() for path in paths}
    with tempfile.TemporaryDirectory(prefix=".rr-copy-", dir=Path.home()) as directory:
        parent = Path(directory) / "observations"
        parent.mkdir(mode=0o750)
        for path, content in originals.items():
            copied = parent / path.name
            copied.write_bytes(content)
            copied.chmod(0o400)
        result = observation.collect(parent=parent, scope=scope, report=report)
        assert result["status"] == "observed", result
        assert result["prepared_mcp_counts"]["successful_reads"] == 1
        correlation = result["operation_correlation"]
        assert correlation["status"] == "observed", correlation
        assert correlation["counts"]["total_operations"] == 3
        assert correlation["counts"]["completed_socket_send_operations"] == 3
        assert correlation["counts"]["completed_socket_send_successful_reads"] == 1
        assert correlation["counts"]["unconfirmed_operations"] == 0
        assert correlation["worker_receipt_verified"] is False
        lifecycle = result["lifecycle_observation"]
        assert lifecycle["status"] == "observed", lifecycle
        assert lifecycle["own_service_reservation_release_reported"] is True
        assert lifecycle["known_session_count"] == 1 and lifecycle["unknown_session_count"] == 0
        assert lifecycle["independent_scheduler_capacity_verified"] is False
        assert lifecycle["borrowed_caller_release"] is None
        assert lifecycle["worker_reaping_verified"] is False
        assert lifecycle["sessions"][0]["renewal_unproven"] is release_exception
        assert result["native_parent_release_verified"] is False
        assert result["delivered_reads"] is result["model_consumed_tokens"] is None
        request.node.user_properties.append(("owner_retirement_observation", json.dumps(result, sort_keys=True)))
        request.node.user_properties.append(("evidence_boundary",
            "New admitted native owner fixture; authored isolation/task/router identities; independent capacity reclamation unverified"))
        # Missing optional lifecycle evidence cannot erase valid session counts.
        (parent / retirement.name).unlink()
        missing = observation.collect(parent=parent, scope=scope, report=report)
        assert missing["status"] == "observed" and missing["prepared_mcp_counts"] == result["prepared_mcp_counts"]
        assert missing["lifecycle_observation"]["own_service_reservation_release_reported"] is None
    assert {path: path.read_bytes() for path in paths} == originals
    assert report == report_before
    assert not native_parent.released and not native_parent.cancelled
