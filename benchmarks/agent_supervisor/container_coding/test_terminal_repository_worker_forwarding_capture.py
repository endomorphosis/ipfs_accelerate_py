"""Authored row handoffs; sealed-file collector checks, no provider execution."""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_binding as binding_api
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from benchmarks.agent_supervisor.container_coding import terminal_repository_worker_forwarding as forwarding
from benchmarks.agent_supervisor.container_coding import terminal_repository_owner_observation as collector
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_correlation import authored_correlated_owner_evidence
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_correlation_population import _authored_v2, _lifecycle
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_observation import (
    evidence, _authored, _collect)
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_worker_forwarding import authored_worker_forwarding_receipt


def _sha(value):
    return hashlib.sha256(protocol.canonical_json(value)).hexdigest()


def authored_captured_worker_evidence(value, *, receipts=None):
    """Explicit fixture metadata, not a recovered invocation or actual outbox."""
    value = deepcopy(value)
    row = value["row"]
    binding = row["repository_source_policy"]
    target = dict(schema="repository-source-worker-outbox@1",
        directory="/authored/outbox/" + row["invocation_id"], device=1, inode=2,
        uid=binding["descriptor"]["worker_uid"], session_id=binding["session_id"],
        descriptor_sha256=binding["descriptor_sha256"], invocation_id=row["invocation_id"])
    row["repository_source_policy"] = binding_api.with_worker_forwarding(binding, target)
    row["usage"]["codex_repository_source_config_sha256"] = row["repository_source_policy"]["config_sha256"]
    if receipts is None:
        receipts = [authored_worker_forwarding_receipt(value)]
    entries = [dict(slot=index, receipt=receipt, sha256=_sha(receipt),
                    bytes=len(protocol.canonical_json(receipt))) for index, receipt in enumerate(receipts)]
    row["repository_source_worker_forwarding"] = dict(
        schema="repository-source-worker-forwarding-capture@1", status="captured", reason=None,
        session_id=binding["session_id"], descriptor_sha256=binding["descriptor_sha256"],
        invocation_id=row["invocation_id"], target_sha256=_sha(target), receipts=entries,
        receipt_count=len(entries), receipt_bytes=sum(entry["bytes"] for entry in entries),
        slot_scan_complete=True, cleanup_complete=True, worker_reported=True,
        independently_verified=False, provider_ingestion_verified=False, token_savings_measured=False,
        proof_authority=False, execution_authority=False, completion_authority=False,
        raw_content_exported=False, observer_provider_calls=0)
    return value


def _capture(value):
    return value["row"]["repository_source_worker_forwarding"]


def _summarize(value):
    return forwarding.summarize_captured_worker_forwarding(**value)


def _save(evidence, value):
    evidence["write"](value)
    evidence["report"]["provider_invocations"][0] = value["row"]
    return _collect(evidence)


def test_canonical_capture_hashes_join_without_io_or_authority(monkeypatch):
    value = authored_captured_worker_evidence(authored_correlated_owner_evidence())
    before = deepcopy(value)
    def no_io(*args, **kwargs):
        raise AssertionError("offline handoff validation cannot reopen outboxes or state")
    monkeypatch.setattr("builtins.open", no_io)
    result = _summarize(value)
    assert value == before
    assert result["status"] == "observed" and result["reason"] is None
    assert result["counts"]["reported_completed_stdout_successful_reads"] == 1
    assert result["capture_sha256"] == _sha(_capture(value))
    assert result["target_sha256"] == _sha(value["row"]["repository_source_policy"]["worker_forwarding"])
    assert result["forwarding_receipt_sha256s"] == [entry["sha256"] for entry in _capture(value)["receipts"]]
    assert result["worker_reported_only"] is True
    assert all(result[key] is False for key in ("independent_worker_receipt_verified",
        "provider_ingestion_verified", "output_payloads_independently_verified", "token_savings_measured",
        "proof_authority", "execution_authority", "completion_authority", "production_route_qualified",
        "worker_invocation_population_verified"))
    public = json.dumps(result)
    assert "/authored" not in public and value["row"]["invocation_id"] not in public
    assert "stdout_sha256" not in public and "request_sha256" not in public


@pytest.mark.parametrize("field,bad", [("schema", "unknown"), ("target_sha256", "0" * 64),
    ("invocation_id", "0" * 32), ("session_id", "0" * 32), ("descriptor_sha256", "0" * 64),
    ("receipt_count", True), ("receipt_count", 0), ("receipt_bytes", 0),
    ("slot_scan_complete", False), ("cleanup_complete", False), ("independently_verified", True),
    ("raw_content_exported", True), ("observer_provider_calls", False), ("extra_field", "/never-exported")])
def test_malformed_capture_is_unknown_without_counts(field, bad):
    value = authored_captured_worker_evidence(authored_correlated_owner_evidence())
    _capture(value)[field] = bad
    result = _summarize(value)
    assert result["status"] == "unavailable" and result["counts"] is None
    assert result["reason"] == "invalid_worker_capture"
    assert result["capture_sha256"] is None and "/never-exported" not in json.dumps(result)


@pytest.mark.parametrize("change", ["payload", "hash", "bytes", "duplicate_slot", "slot_bound"])
def test_retained_receipt_entries_require_exact_canonical_hashes_and_unique_slots(change):
    value = authored_captured_worker_evidence(authored_correlated_owner_evidence())
    capture = _capture(value)
    entry = capture["receipts"][0]
    if change == "payload":
        entry["receipt"]["events"][0]["stdout_sha256"] = "0" * 64
    elif change == "hash":
        entry["sha256"] = "0" * 64
    elif change == "bytes":
        entry["bytes"] += 1
    elif change == "duplicate_slot":
        capture["receipts"].append(deepcopy(entry))
        capture["receipt_count"] += 1
        capture["receipt_bytes"] *= 2
    else:
        entry["slot"] = 16
    result = _summarize(value)
    assert result["status"] == "unavailable" and result["reason"] == "invalid_worker_capture"
    assert result["counts"] is None


def test_absent_capture_and_empty_scan_remain_unknown():
    legacy = authored_correlated_owner_evidence()
    assert _summarize(legacy)["reason"] == "worker_capture_unavailable"
    value = authored_captured_worker_evidence(legacy, receipts=[])
    result = _summarize(value)
    # A claimed captured population with no retained receipt is contradictory.
    assert result["status"] == "unavailable" and result["reason"] == "invalid_worker_capture"
    assert result["counts"] is None and result["capture_sha256"] is None
    # The producer reports an empty completed scan as unavailable, not zero.
    _capture(value).update(status="unavailable", reason="capture_unavailable",
                           receipt_count=None, receipt_bytes=None)
    result = _summarize(value)
    assert result["status"] == "unavailable" and result["reason"] == "worker_capture_unavailable"
    assert result["counts"] is None and result["capture_sha256"] is None
    value["row"].pop("repository_source_worker_forwarding")
    assert _summarize(value)["reason"] == "worker_capture_unavailable"


def test_unavailable_scan_diagnostics_cannot_become_a_qualified_subtotal():
    value = authored_captured_worker_evidence(authored_correlated_owner_evidence())
    _capture(value).update(status="unavailable", reason="capture_failed", receipt_count=None,
        receipt_bytes=None, slot_scan_complete=False, cleanup_complete=False)
    result = _summarize(value)
    assert result["status"] == "unavailable" and result["reason"] == "worker_capture_unavailable"
    assert result["counts"] is None


def test_complete_slot_scan_still_requires_every_owner_response():
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence, ordinals={1})
    value = authored_captured_worker_evidence(evidence, receipts=[receipt])
    result = _summarize(value)
    assert result["status"] == "unavailable" and result["reason"] == "incomplete_owner_population"
    assert result["counts"] is None


def test_captured_target_cannot_be_replayed_under_another_invocation():
    value = authored_captured_worker_evidence(authored_correlated_owner_evidence())
    value["row"]["invocation_id"] = "a" * 32
    result = _summarize(value)
    assert result["status"] == "unavailable" and result["reason"] == "invalid_worker_capture"
    assert result["counts"] is None and result["invocation_sha256"] is not None


def test_valid_capture_preserves_known_failed_stdout_report():
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence, returncode=2)
    receipt["events"][-1]["forwarding_status"] = "unconfirmed"
    value = authored_captured_worker_evidence(evidence, receipts=[receipt])
    result = _summarize(value)
    assert result["status"] == "observed"
    assert result["counts"]["reported_received_operations"] == 2
    assert result["counts"]["reported_completed_stdout_operations"] == 1
    assert result["counts"]["unconfirmed_stdout_operations"] == 1


def test_owner_join_cannot_be_rescued_by_valid_capture():
    value = authored_captured_worker_evidence(authored_correlated_owner_evidence())
    value["operations_raw"] += b"changed"
    result = _summarize(value)
    assert result["reason"] == "owner_evidence_unavailable" and result["counts"] is None


def test_historical_owner_audit_has_unknown_forwarding_despite_capture():
    value = authored_captured_worker_evidence(_authored())
    result = _summarize(value)
    assert result["reason"] == "owner_correlation_unavailable" and result["counts"] is None


def test_collector_retains_separate_reported_population(evidence):
    value = authored_captured_worker_evidence(_authored_v2(evidence["first"]))
    _lifecycle(evidence, value)
    result = _save(evidence, value)
    original_report = deepcopy(evidence["report"])
    assert result["status"] == "observed" and result["prepared_mcp_counts"]["successful_reads"] == 1
    population = result["worker_forwarding_observation"]
    assert population["status"] == "observed" and population["reason"] is None
    assert population["session_count"] == population["known_session_count"] == 1
    assert population["unknown_session_count"] == 0
    assert population["counts"]["reported_completed_stdout_successful_reads"] == 1
    assert population["sessions"][0]["invocation_sha256"] == result["sessions"][0]["invocation_sha256"]
    assert result["delivered_reads"] is result["model_consumed_tokens"] is None
    assert result["native_parent_release_verified"] is result["worker_cleanup_independently_verified"] is False
    assert result["lifecycle_observation"]["own_service_reservation_release_reported"] is True
    assert population["worker_invocation_population_verified"] is False
    assert _collect(evidence) == result and evidence["report"] == original_report


@pytest.mark.parametrize("missing_second", [False, True])
def test_selected_sessions_sum_only_when_every_forwarding_population_is_known(evidence, missing_second):
    first = authored_captured_worker_evidence(_authored_v2(evidence["first"]))
    second = authored_captured_worker_evidence(_authored_v2(_authored("3" * 32, "4" * 32)))
    if missing_second:
        second["row"].pop("repository_source_worker_forwarding")
    evidence["write"](second)
    evidence["report"]["provider_invocations"].append(second["row"])
    result = _save(evidence, first)
    assert result["status"] == "observed" and result["prepared_mcp_counts"]["successful_reads"] == 2
    population = result["worker_forwarding_observation"]
    assert population["session_count"] == 2
    assert population["known_session_count"] == (1 if missing_second else 2)
    assert population["unknown_session_count"] == (1 if missing_second else 0)
    if missing_second:
        assert population["status"] == "unavailable" and population["counts"] is None
    else:
        assert population["status"] == "observed"
        assert population["counts"]["reported_completed_stdout_successful_reads"] == 2
        for key in forwarding.COUNTERS:
            assert population["counts"][key] == sum(item["counts"][key] for item in population["sessions"])


@pytest.mark.parametrize("change", ["bad_hash", "missing_receipt", "malformed_receipt", "projection_error"])
def test_optional_capture_failure_preserves_owner_and_lifecycle_evidence(evidence, monkeypatch, change):
    value = authored_captured_worker_evidence(_authored_v2(evidence["first"]))
    capture = _capture(value)
    if change == "bad_hash":
        capture["receipts"][0]["sha256"] = "0" * 64
    elif change == "missing_receipt":
        value["row"].pop("repository_source_worker_forwarding")
    elif change == "malformed_receipt":
        capture["receipts"][0]["receipt"]["private"] = "/never-exported"
    else:
        def failed_projection(**kwargs):
            raise ValueError("/never-exported")
        monkeypatch.setattr(forwarding, "summarize_captured_worker_forwarding", failed_projection)
    _lifecycle(evidence, value)
    result = _save(evidence, value)
    assert result["status"] == "observed" and result["prepared_mcp_counts"]["successful_reads"] == 1
    assert result["operation_correlation"]["status"] == "observed"
    assert result["lifecycle_observation"]["own_service_reservation_release_reported"] is True
    population = result["worker_forwarding_observation"]
    assert population["status"] == "unavailable" and population["counts"] is None
    assert population["known_session_count"] == 0 and population["unknown_session_count"] == 1
    assert "/never-exported" not in json.dumps(result)


def test_empty_selected_forwarding_population_cannot_become_known_zero():
    result = forwarding.aggregate([])
    assert result["status"] == "unavailable" and result["counts"] is None
    assert result["session_count"] == result["known_session_count"] == result["unknown_session_count"] == 0


def test_unavailable_and_legacy_collector_shapes_are_unchanged():
    assert "worker_forwarding_observation" not in collector.unavailable()
    result = collector.collect(parent=None, scope=None, report={"repository_source_policy": "legacy"})
    assert result["status"] == "not_selected" and "worker_forwarding_observation" not in result
