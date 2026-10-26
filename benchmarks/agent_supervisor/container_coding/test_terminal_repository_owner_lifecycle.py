"""Authored retirement evidence joins; no admission, providers or live state."""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from benchmarks.agent_supervisor.container_coding import terminal_repository_owner_lifecycle as lifecycle
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_audit import authored_owner_evidence


def authored_lifecycle_envelope(evidence, *, borrowed=True, renewal_unproven=False):
    audit = json.loads(evidence["audit_raw"])
    parent = dict(lease_id=audit["native_child"]["parent_lease_id"],
        owner_pid=audit["session"]["owner_pid"], parent_lease_id="4" * 32 if borrowed else None,
        cpu_slots=1, memory_mb=512, child_process_slots=1)
    receipt = dict(schema="repository-source-owner-lifecycle@1", state="closed", ready=False,
        started=True, worker_spawn_registered=False, cleanup_complete=True, cleanup_pending=False,
        native_parent_acquired=True, native_parent_released=True, native_parent=parent,
        admission_uncertain=False, observation_pending=False,
        admission_request_id="repository-source-lifecycle:" + "5" * 32,
        broker_cleanup_complete=True, broker_audit=audit, renewal_unproven=renewal_unproven,
        retained_custody=False, error=None, cleanup_error=None, borrowed_parent_present=borrowed,
        resource_scope="repository_source_service_only", worker_reaping_performed=False,
        worker_cleanup_verified=False, automatic_terminal_wiring=False,
        cross_uid_deployment_qualified=False, provider_calls=0, token_savings_measured=False,
        proof_authority=False, execution_authority=False, completion_authority=False)
    return dict(schema=lifecycle.RETIREMENT_SCHEMA, session_id=audit["session"]["session_id"],
        descriptor_sha256=audit["descriptor_sha256"],
        session_audit_sha256=hashlib.sha256(evidence["audit_raw"]).hexdigest(),
        session_audit_bytes=len(evidence["audit_raw"]), lifecycle=receipt)


def _summarize(envelope, evidence):
    return lifecycle.summarize_owner_lifecycle(retirement_raw=protocol.canonical_json(envelope),
                                              audit_raw=evidence["audit_raw"])


@pytest.mark.parametrize("borrowed", [True, False])
@pytest.mark.parametrize("renewal_unproven", [True, False])
def test_exact_chain_reports_own_release_without_other_authority(borrowed, renewal_unproven, monkeypatch):
    evidence = authored_owner_evidence()
    value = authored_lifecycle_envelope(evidence, borrowed=borrowed, renewal_unproven=renewal_unproven)
    before = deepcopy((value, evidence))
    def no_io(*args, **kwargs):
        raise AssertionError("historical evidence joins must not perform I/O")
    monkeypatch.setattr("builtins.open", no_io)
    result = _summarize(value, evidence)
    assert (value, evidence) == before
    assert result["status"] == "joined_complete"
    assert result["own_service_reservation_release_reported"] is True
    assert result["borrowed_parent_present"] is borrowed
    assert result["renewal_unproven"] is renewal_unproven
    assert result["borrowed_caller_release"] is None
    assert result["independent_scheduler_capacity_verified"] is False
    assert result["worker_reaping_verified"] is result["worker_cleanup_independently_verified"] is False
    assert result["provider_ingestion_verified"] is result["token_savings_measured"] is False
    raw = protocol.canonical_json(result)
    assert b"/authored" not in raw and b"owner_pid" not in raw
    assert value["lifecycle"]["admission_request_id"].encode() not in raw
    assert value["lifecycle"]["native_parent"]["lease_id"].encode() not in raw


@pytest.mark.parametrize("path,bad", [
    ("schema", "unknown"), ("session_id", "0" * 32), ("descriptor_sha256", "0" * 64),
    ("session_audit_sha256", "0" * 64), ("session_audit_bytes", True),
    ("session_audit_bytes", 0), ("lifecycle.schema", "unknown"),
    ("lifecycle.state", "cleanup_pending"), ("lifecycle.ready", True),
    ("lifecycle.started", False), ("lifecycle.cleanup_complete", False),
    ("lifecycle.cleanup_pending", True), ("lifecycle.native_parent_acquired", False),
    ("lifecycle.native_parent_released", False), ("lifecycle.native_parent_released", 1),
    ("lifecycle.admission_uncertain", True), ("lifecycle.observation_pending", True),
    ("lifecycle.retained_custody", True), ("lifecycle.broker_cleanup_complete", False),
    ("lifecycle.borrowed_parent_present", False), ("lifecycle.borrowed_parent_present", 1),
    ("lifecycle.worker_spawn_registered", 0), ("lifecycle.renewal_unproven", 0),
    ("lifecycle.admission_request_id", "other-owner"),
    ("lifecycle.cleanup_error", "native_parent_release_failed"),
    ("lifecycle.error", "/private/path"), ("lifecycle.resource_scope", "all_workers"),
    ("lifecycle.worker_reaping_performed", True), ("lifecycle.worker_cleanup_verified", True),
    ("lifecycle.automatic_terminal_wiring", True), ("lifecycle.cross_uid_deployment_qualified", True),
    ("lifecycle.provider_calls", False), ("lifecycle.provider_calls", 1),
    ("lifecycle.token_savings_measured", True), ("lifecycle.proof_authority", True),
    ("lifecycle.execution_authority", True), ("lifecycle.completion_authority", True),
    ("lifecycle.native_parent.lease_id", "different-parent"),
    ("lifecycle.native_parent.owner_pid", 124), ("lifecycle.native_parent.owner_pid", True),
    ("lifecycle.native_parent.cpu_slots", 2), ("lifecycle.native_parent.memory_mb", 1024),
    ("lifecycle.native_parent.child_process_slots", True),
    ("lifecycle.native_parent.parent_lease_id", None),
    ("lifecycle.native_parent.parent_lease_id", "3" * 32),
    ("lifecycle.native_parent.parent_lease_id", "2" * 32),
    ("lifecycle.native_parent.parent_lease_id", ""),
    ("lifecycle.broker_audit.cleanup_complete", False),
    ("lifecycle.broker_audit.native_child.parent_lease_id", "other-parent"),
])
def test_mismatched_or_overclaiming_retirement_is_refused(path, bad):
    evidence = authored_owner_evidence()
    value = authored_lifecycle_envelope(evidence)
    target = value
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = bad
    with pytest.raises(ValueError):
        _summarize(value, evidence)


@pytest.mark.parametrize("where", ["envelope", "lifecycle", "parent"])
@pytest.mark.parametrize("change", ["extra", "missing"])
def test_closed_shapes_reject_unrecognized_or_missing_fields(where, change):
    evidence = authored_owner_evidence()
    value = authored_lifecycle_envelope(evidence)
    target = value if where == "envelope" else value["lifecycle"]
    if where == "parent":
        target = target["native_parent"]
    if change == "extra":
        target["other"] = False
    else:
        target.pop(next(iter(target)))
    with pytest.raises(ValueError):
        _summarize(value, evidence)


@pytest.mark.parametrize("bad", [None, "{}", bytearray(b"{}"), b"", b"{}", b"\xff",
    b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}',
    b'{"a":"\\ud800"}', b"[" * 1024 + b"]" * 1024])
def test_untrusted_retirement_parsing_is_bounded_and_strict(bad):
    with pytest.raises(ValueError):
        lifecycle.summarize_owner_lifecycle(retirement_raw=bad, audit_raw=authored_owner_evidence()["audit_raw"])


def test_retirement_byte_limit_and_canonical_wire_are_required():
    evidence = authored_owner_evidence()
    raw = protocol.canonical_json(authored_lifecycle_envelope(evidence))
    for bad in (raw + b"\n", b" " * (lifecycle.MAX_RETIREMENT_BYTES + 1)):
        with pytest.raises(ValueError):
            lifecycle.summarize_owner_lifecycle(retirement_raw=bad, audit_raw=evidence["audit_raw"])


def test_changed_retained_session_cannot_join_even_if_semantically_equivalent():
    evidence = authored_owner_evidence()
    value = authored_lifecycle_envelope(evidence)
    evidence["audit_raw"] += b"\n"
    with pytest.raises(ValueError):
        _summarize(value, evidence)
    # Updating the envelope hash still cannot turn a different byte snapshot
    # into the canonical embedded session audit retained by the owner.
    value["session_audit_sha256"] = hashlib.sha256(evidence["audit_raw"]).hexdigest()
    value["session_audit_bytes"] = len(evidence["audit_raw"])
    with pytest.raises(ValueError):
        _summarize(value, evidence)


@pytest.mark.parametrize("key,bad", [("canonical_source_read_only", False),
    ("cross_uid_deployment_qualified", True), ("automatic_terminal_wiring", True),
    ("token_savings_measured", True), ("proof_authority", True),
    ("execution_authority", True), ("completion_authority", True),
    ("provider_calls", False), ("provider_calls", 1), ("accepted_requests", 3),
    ("accounting_complete", False), ("thread_launch_attempted", False),
    ("native_child_released", False), ("service_stopped", False)])
def test_matching_envelope_cannot_endorse_contradictory_session_evidence(key, bad):
    evidence = authored_owner_evidence()
    audit = json.loads(evidence["audit_raw"])
    audit[key] = bad
    evidence["audit_raw"] = protocol.canonical_json(audit)
    with pytest.raises(ValueError):
        _summarize(authored_lifecycle_envelope(evidence), evidence)


def test_partial_population_has_unknown_aggregate_disposition():
    evidence = authored_owner_evidence()
    known = _summarize(authored_lifecycle_envelope(evidence), evidence)
    result = lifecycle.aggregate([known, lifecycle.unavailable()])
    assert result["status"] == "unavailable"
    assert result["known_session_count"] == result["unknown_session_count"] == 1
    assert result["own_service_reservation_release_reported"] is None
    assert result["sessions"][1]["own_service_reservation_release_reported"] is None
    complete = lifecycle.aggregate([known])
    assert complete["own_service_reservation_release_reported"] is True
    assert complete["independent_scheduler_capacity_verified"] is False
