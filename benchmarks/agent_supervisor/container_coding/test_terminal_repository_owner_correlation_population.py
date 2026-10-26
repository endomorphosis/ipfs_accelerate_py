"""Authored correlation populations; real sealed-file checks, no live providers."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_owner_observation as collector
from benchmarks.agent_supervisor.container_coding import terminal_repository_owner_audit as audit_api
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_observation import (
    evidence, _authored, _collect, _write_sealed)
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_lifecycle import authored_lifecycle_envelope
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol


def _authored_v2(value):
    """Explicitly author the two fixture associations; never recover old ones."""
    value = deepcopy(value)
    audit = json.loads(value["audit_raw"])
    lines = value["operations_raw"].splitlines(keepends=True)
    assert len(lines) == len(audit["requests"]) == 2
    audit["schema"] = "repository-source-owner-session@2"
    for index in (0, 1):
        audit["requests"][index]["operation_capture"] = dict(status="complete", operations=[dict(
            sequence=index + 1, sha256=hashlib.sha256(lines[index]).hexdigest(), bytes=len(lines[index]))])
    value["audit_raw"] = protocol.canonical_json(audit)
    return value


def _save(evidence, value):
    evidence["write"](value)
    return _collect(evidence)


def _lifecycle(evidence, value):
    envelope = authored_lifecycle_envelope(value)
    session = value["row"]["repository_source_policy"]["session_id"]
    _write_sealed(evidence["parent"] / ("repository-mcp-" + session + "-lifecycle.json"),
        protocol.canonical_json(envelope))


def test_complete_population_counts_socket_results_without_ingestion(evidence):
    first = _authored_v2(evidence["first"])
    second = _authored_v2(_authored("3" * 32, "4" * 32))
    evidence["write"](first)
    evidence["write"](second)
    evidence["report"]["provider_invocations"].append(second["row"])
    before = deepcopy(evidence["report"])
    result = _collect(evidence)
    assert result["status"] == "observed"
    population = result["operation_correlation"]
    assert population["status"] == "observed" and population["reason"] is None
    assert population["session_count"] == population["known_session_count"] == 2
    assert population["unknown_session_count"] == 0
    assert population["counts"]["total_operations"] == 4
    assert population["counts"]["completed_socket_send_successful_reads"] == 2
    assert population["counts"]["completed_socket_send_native_payload_bytes"] == 240
    for key in audit_api.OPERATION_COUNTERS:
        assert population["counts"][key] == sum(row["operation_correlation"]["counts"][key]
            for row in result["sessions"])
    assert result["unknown_reason"] == "missing_worker_receipt_and_model_ingestion_evidence"
    for key in ("worker_receipt_verified", "provider_ingestion_verified", "response_payloads_independently_verified",
                "token_savings_measured", "proof_authority", "execution_authority", "completion_authority"):
        assert population[key] is False
    assert result["delivered_reads"] is result["model_consumed_tokens"] is None
    assert evidence["report"] == before


@pytest.mark.parametrize("legacy_first", [True, False])
def test_mixed_legacy_population_keeps_full_correlation_unknown(evidence, legacy_first):
    first = evidence["first"] if legacy_first else _authored_v2(evidence["first"])
    second = _authored("3" * 32, "4" * 32)
    if legacy_first:
        second = _authored_v2(second)
    evidence["write"](first)
    evidence["write"](second)
    evidence["report"]["provider_invocations"].append(second["row"])
    result = _collect(evidence)
    assert result["status"] == "observed" and result["prepared_mcp_counts"]["successful_reads"] == 2
    population = result["operation_correlation"]
    assert population["status"] == "unavailable" and population["counts"] is None
    assert population["known_session_count"] == population["unknown_session_count"] == 1
    assert population["session_count"] == 2


@pytest.mark.parametrize("change", ["incomplete", "missing", "duplicate", "hash", "bytes", "bool_sequence", "extra_field"])
def test_bad_capture_preserves_prepared_counts_and_independent_lifecycle(evidence, change):
    value = _authored_v2(evidence["first"])
    audit = json.loads(value["audit_raw"])
    capture = audit["requests"][1]["operation_capture"]
    if change == "incomplete": capture["status"] = "incomplete"
    elif change == "missing": capture["operations"] = []
    elif change == "duplicate": capture["operations"] = deepcopy(audit["requests"][0]["operation_capture"]["operations"])
    elif change == "hash": capture["operations"][0]["sha256"] = "f" * 64
    elif change == "bytes": capture["operations"][0]["bytes"] += 1
    elif change == "bool_sequence": capture["operations"][0]["sequence"] = True
    elif change == "extra_field": capture["private_path"] = "/never-exported"
    value["audit_raw"] = protocol.canonical_json(audit)
    _lifecycle(evidence, value)
    result = _save(evidence, value)
    assert result["status"] == "observed" and result["prepared_mcp_counts"]["successful_reads"] == 1
    assert result["operation_correlation"]["status"] == "unavailable"
    assert result["operation_correlation"]["counts"] is None
    assert result["lifecycle_observation"]["own_service_reservation_release_reported"] is True
    assert "/never-exported" not in json.dumps(result)


@pytest.mark.parametrize("status", ["transport_error", "cancelled", "stale_context"])
def test_unconfirmed_send_preserves_operation_count_without_delivery_claim(evidence, status):
    value = _authored_v2(evidence["first"])
    audit = json.loads(value["audit_raw"])
    audit["requests"][1].update(status=status, delivered=False)
    value["audit_raw"] = protocol.canonical_json(audit)
    result = _save(evidence, value)
    assert result["status"] == "observed"
    counts = result["operation_correlation"]["counts"]
    assert counts["total_operations"] == 2 and counts["unconfirmed_operations"] == 1
    assert counts["completed_socket_send_operations"] == 1
    assert counts["completed_socket_send_successful_reads"] == 0
    assert result["prepared_mcp_counts"]["successful_reads"] == 1
    assert result["delivered_reads"] is None


def test_missing_v2_session_cannot_be_aggregated_as_zero(evidence):
    value = _authored_v2(evidence["first"])
    evidence["write"](value)
    second = _authored_v2(_authored("3" * 32, "4" * 32))
    evidence["report"]["provider_invocations"].append(second["row"])
    result = _collect(evidence)
    assert result == collector.unavailable("retained_evidence_unavailable")


def test_empty_population_cannot_report_known_zero():
    result = collector._operation_population([])
    assert result["status"] == "unavailable" and result["counts"] is None


def test_v2_lifecycle_and_correlation_have_separate_evidence(evidence):
    value = _authored_v2(evidence["first"])
    result = _save(evidence, value)
    assert result["operation_correlation"]["status"] == "observed"
    assert result["lifecycle_observation"]["own_service_reservation_release_reported"] is None
    _lifecycle(evidence, value)
    joined = _collect(evidence)
    assert joined["operation_correlation"] == result["operation_correlation"]
    assert joined["lifecycle_observation"]["own_service_reservation_release_reported"] is True
