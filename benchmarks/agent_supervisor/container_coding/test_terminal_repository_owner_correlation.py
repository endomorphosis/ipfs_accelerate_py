"""Authored explicit operation correlation; no native or provider execution."""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from benchmarks.agent_supervisor.container_coding import terminal_repository_owner_audit as audit
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_audit import authored_owner_evidence


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _reference(line):
    return dict(sequence=json.loads(line)["sequence"], sha256=_sha(line), bytes=len(line))


def _store(evidence, value):
    evidence["audit_raw"] = protocol.canonical_json(value)


def authored_correlated_owner_evidence(**kwargs):
    """Version 2 authored records with exact, LF-inclusive operation references."""
    evidence = authored_owner_evidence(**kwargs)
    value = json.loads(evidence["audit_raw"])
    value["schema"] = "repository-source-owner-session@2"
    for record, line in zip(value["requests"], evidence["operations_raw"].splitlines(keepends=True)):
        record["operation_capture"] = dict(status="complete", operations=[_reference(line)])
    _store(evidence, value)
    return evidence


def _rewrite_operations(evidence, rows, *, insertion_order=False):
    raw = b"".join((json.dumps(row, separators=(",", ":")).encode()
        if insertion_order else protocol.canonical_json(row)) + b"\n" for row in rows)
    value = json.loads(evidence["audit_raw"])
    value["retained_mcp_audit"].update(sha256=_sha(raw), bytes=len(raw))
    value["prepared_mcp_usage"] = {**mcp.summarize_repository_audit_bytes(raw),
        "store_cleanup_complete": True, "artifacts_retained_after_invocation": False}
    for record, line in zip(value["requests"], raw.splitlines(keepends=True)):
        record["operation_capture"] = dict(status="complete", operations=[_reference(line)])
    evidence["operations_raw"] = raw
    _store(evidence, value)


def _result(evidence):
    return audit.summarize_owner_session(**evidence)


def _unknown(evidence, reason):
    result = _result(evidence)
    assert result["status"] == "joined_complete"
    assert result["operation_correlation"] == audit.operation_correlation_unavailable(reason)
    assert result["prepared_mcp_usage"]["successful_reads"] == 1
    assert result["consumption"]["delivered_reads"] is None
    return result


def test_exact_operation_join_counts_completed_sends_without_consumption_claims():
    evidence = authored_correlated_owner_evidence()
    before = deepcopy(evidence)
    result = _result(evidence)
    assert evidence == before
    correlation = result["operation_correlation"]
    assert correlation["status"] == "observed" and correlation["reason"] is None
    assert correlation["counts"] == dict(total_operations=2, completed_socket_send_operations=2,
        completed_socket_send_tool_results=1, completed_socket_send_successful_tool_results=1,
        completed_socket_send_successful_reads=1, completed_socket_send_failed_tool_results=0,
        completed_socket_send_native_payload_bytes=120, unconfirmed_operations=0)
    assert all(correlation[key] is False for key in ("worker_receipt_verified", "provider_ingestion_verified",
        "response_payloads_independently_verified", "token_savings_measured",
        "proof_authority", "execution_authority", "completion_authority"))
    assert result["consumption"]["delivered_tool_calls"] is None
    assert result["consumption"]["provider_tokens"] is None
    assert result["consumption"]["delivery_reason"] == "missing_worker_receipt_evidence"
    raw = protocol.canonical_json(correlation)
    assert b"/authored" not in raw and b"request_sha256" not in raw


def test_historical_owner_audit_keeps_unknown_correlation():
    evidence = authored_owner_evidence()
    result = _unknown(evidence, "legacy_owner_audit")
    assert result["transport"]["completed_frame_sends"] == 2
    assert result["consumption"]["delivery_reason"] == "missing_per_operation_correlation"


@pytest.mark.parametrize("capture", [None, [], "complete", {},
    {"status": "complete", "operations": [], "extra": False},
    {"status": "other", "operations": []}, {"status": 1, "operations": []},
    {"status": "complete", "operations": None}, {"status": "complete", "operations": {}},
    {"status": "complete", "operations": [None]},
    {"status": "complete", "operations": [{}, {}]}])
def test_malformed_inner_capture_preserves_prepared_evidence(capture):
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    value["requests"][0]["operation_capture"] = capture
    _store(evidence, value)
    _unknown(evidence, "invalid_operation_capture")


@pytest.mark.parametrize("field,bad", [("sequence", True), ("sequence", 0), ("sequence", 513),
    ("sequence", "1"), ("sha256", None), ("sha256", "x" * 64),
    ("bytes", False), ("bytes", 0), ("bytes", mcp.MAX_AUDIT_BYTES + 1), ("bytes", "120")])
def test_reference_requires_exact_bounded_types(field, bad):
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    value["requests"][0]["operation_capture"]["operations"][0][field] = bad
    _store(evidence, value)
    _unknown(evidence, "invalid_operation_capture")


@pytest.mark.parametrize("change", ["extra", "missing", "not_entered_with_reference"])
def test_reference_shapes_and_unentered_state_are_closed(change):
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    capture = value["requests"][0]["operation_capture"]
    if change == "extra": capture["operations"][0]["source"] = "unsafe"
    elif change == "missing": capture["operations"][0].pop("bytes")
    else: capture["status"] = "not_entered"
    _store(evidence, value)
    _unknown(evidence, "invalid_operation_capture")


@pytest.mark.parametrize("change", ["sha", "bytes", "without_lf", "missing", "duplicate", "reordered"])
def test_exact_full_ordered_bijection_is_required(change):
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    first, second = [record["operation_capture"] for record in value["requests"]]
    reference = first["operations"][0]
    if change == "sha": reference["sha256"] = "0" * 64
    elif change == "bytes": reference["bytes"] += 1
    elif change == "without_lf":
        line = evidence["operations_raw"].splitlines()[0]
        reference.update(sha256=_sha(line), bytes=len(line))
    elif change == "missing": first["operations"] = []
    elif change == "duplicate": second["operations"] = deepcopy(first["operations"])
    else: first["operations"], second["operations"] = second["operations"], first["operations"]
    _store(evidence, value)
    _unknown(evidence, "operation_reference_mismatch")


@pytest.mark.parametrize("retain_reference", [True, False])
def test_explicit_incomplete_capture_never_produces_partial_totals(retain_reference):
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    capture = value["requests"][1]["operation_capture"]
    capture["status"] = "incomplete"
    if not retain_reference: capture["operations"] = []
    _store(evidence, value)
    _unknown(evidence, "incomplete_operation_capture")


@pytest.mark.parametrize("status", ["complete", "incomplete"])
def test_entered_capture_requires_a_validated_request_hash(status):
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    record = value["requests"][0]
    record.update(status="peer_refused", request_sha256=None, prepared_response_sha256=None,
        prepared_response_bytes=None, response_sha256=None, response_bytes=0, delivered=False,
        operation_capture=dict(status=status, operations=[]))
    # Keep the original two retained operations inside the valid-request bound
    # using an additional no-operation request; order is deliberately explicit.
    extra = deepcopy(value["requests"][1])
    extra.update(ordinal=3, operation_capture=dict(status="complete", operations=[]))
    value["requests"].append(extra)
    value["accepted_requests"] = 3
    value["response_bytes"] = sum(item["response_bytes"] for item in value["requests"])
    _store(evidence, value)
    _unknown(evidence, "invalid_operation_capture")


@pytest.mark.parametrize("outcome", ["post_handle_refusal", "socket_send_failed", "budget_replacement",
    "prepared_but_not_reserved"])
def test_complete_capture_does_not_claim_unconfirmed_results_were_sent(outcome):
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    record = value["requests"][1]
    record["status"] = "retrieval_refused"
    record["delivered"] = False
    if outcome == "post_handle_refusal":
        record.update(prepared_response_sha256=None, prepared_response_bytes=None,
                      response_sha256=None, response_bytes=0)
    elif outcome == "prepared_but_not_reserved":
        record.update(response_sha256=None, response_bytes=0)
    elif outcome == "budget_replacement":
        record.update(status="budget_exhausted", delivered=True,
            prepared_response_bytes=value["session"]["max_total_response_bytes"],
            response_sha256="a" * 64, response_bytes=120)
    value["response_bytes"] = sum(item["response_bytes"] for item in value["requests"])
    _store(evidence, value)
    result = _result(evidence)
    counts = result["operation_correlation"]["counts"]
    assert result["operation_correlation"]["status"] == "observed"
    assert counts["total_operations"] == 2
    assert counts["completed_socket_send_operations"] == counts["unconfirmed_operations"] == 1
    assert counts["completed_socket_send_tool_results"] == counts["completed_socket_send_successful_reads"] == 0
    assert counts["completed_socket_send_native_payload_bytes"] == 0
    assert result["prepared_mcp_usage"]["successful_reads"] == 1
    if outcome == "budget_replacement":
        assert result["transport"]["completed_frame_sends"] == 2


def test_outer_ok_can_carry_failed_tool_result_without_successful_read():
    evidence = authored_correlated_owner_evidence()
    rows = [json.loads(line) for line in evidence["operations_raw"].splitlines()]
    rows[1].update(success=False, compact=False, returned_bytes=24)
    _rewrite_operations(evidence, rows)
    counts = _result(evidence)["operation_correlation"]["counts"]
    assert counts["completed_socket_send_operations"] == 2
    assert counts["completed_socket_send_tool_results"] == counts["completed_socket_send_failed_tool_results"] == 1
    assert counts["completed_socket_send_successful_tool_results"] == counts["completed_socket_send_successful_reads"] == 0
    assert counts["completed_socket_send_native_payload_bytes"] == 24


def test_initialize_outcome_is_not_a_successful_tool_result():
    evidence = authored_correlated_owner_evidence()
    rows = [json.loads(evidence["operations_raw"].splitlines()[0])]
    rows[0]["success"] = False
    _rewrite_operations(evidence, rows)
    value = json.loads(evidence["audit_raw"])
    value["requests"][1]["operation_capture"] = dict(status="complete", operations=[])
    _store(evidence, value)
    counts = _result(evidence)["operation_correlation"]["counts"]
    assert counts["completed_socket_send_operations"] == 1
    assert counts["completed_socket_send_tool_results"] == 0


def test_exact_native_insertion_order_is_hashed_without_canonical_rewrite():
    evidence = authored_correlated_owner_evidence()
    rows = [dict(reversed(list(json.loads(line).items())))
            for line in evidence["operations_raw"].splitlines()]
    _rewrite_operations(evidence, rows, insertion_order=True)
    assert _result(evidence)["operation_correlation"]["status"] == "observed"
    value = json.loads(evidence["audit_raw"])
    canonical = protocol.canonical_json(rows[0]) + b"\n"
    assert canonical != evidence["operations_raw"].splitlines(keepends=True)[0]
    value["requests"][0]["operation_capture"]["operations"][0].update(sha256=_sha(canonical), bytes=len(canonical))
    _store(evidence, value)
    _unknown(evidence, "operation_reference_mismatch")


@pytest.mark.parametrize("capture_status", ["complete", "not_entered"])
def test_retained_empty_operations_and_complete_empty_captures_establish_zero(capture_status):
    evidence = authored_correlated_owner_evidence()
    _rewrite_operations(evidence, [])
    value = json.loads(evidence["audit_raw"])
    for record in value["requests"]:
        record["operation_capture"] = dict(status=capture_status, operations=[])
        if capture_status == "not_entered":
            record["status"] = "invalid_request"
    _store(evidence, value)
    result = _result(evidence)
    assert result["operation_correlation"]["status"] == "observed"
    assert result["operation_correlation"]["counts"] == dict.fromkeys(audit.OPERATION_COUNTERS, 0)


def test_control_requests_do_not_require_artificial_operation_rows():
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    control = deepcopy(value["requests"][0])
    control["operation_capture"] = dict(status="complete", operations=[])
    value["requests"].insert(1, control)
    for ordinal, record in enumerate(value["requests"], 1):
        record["ordinal"] = ordinal
    value["accepted_requests"] = 3
    value["response_bytes"] = sum(item["response_bytes"] for item in value["requests"])
    _store(evidence, value)
    result = _result(evidence)
    assert result["transport"]["completed_frame_sends"] == 3
    assert result["operation_correlation"]["counts"]["completed_socket_send_operations"] == 2


@pytest.mark.parametrize("change", ["missing_capture", "extra_record_field", "legacy_with_capture", "future_schema"])
def test_record_and_schema_version_boundaries_remain_closed(change):
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    if change == "missing_capture": value["requests"][0].pop("operation_capture")
    elif change == "extra_record_field": value["requests"][0]["other"] = None
    elif change == "legacy_with_capture": value["schema"] = "repository-source-owner-session@1"
    else: value["schema"] = "repository-source-owner-session@3"
    _store(evidence, value)
    with pytest.raises(ValueError):
        _result(evidence)
