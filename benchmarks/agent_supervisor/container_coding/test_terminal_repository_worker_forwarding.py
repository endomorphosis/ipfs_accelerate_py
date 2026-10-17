"""Authored worker reports remain reports after exact owner-evidence joins."""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from benchmarks.agent_supervisor.container_coding import terminal_repository_worker_forwarding as forwarding
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_audit import authored_owner_evidence
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_correlation import authored_correlated_owner_evidence


def authored_worker_forwarding_receipt(evidence, *, bridge_id="f" * 32, ordinals=None, returncode=None):
    value = json.loads(evidence["audit_raw"])
    events = []
    for record in value["requests"]:
        if ordinals is not None and record["ordinal"] not in ordinals:
            continue
        ok = record["status"] not in {"invalid_request", "budget_exhausted"}
        events.append(dict(exchange_ordinal=len(events) + 1, request_sha256=record["request_sha256"],
            owner_response_sha256=record["response_sha256"], owner_response_bytes=record["response_bytes"],
            owner_ok=ok, owner_reason_code=None if ok else record["status"],
            output_kind="owner_message" if ok else "local_error",
            stdout_sha256=hashlib.sha256(("authored-stdout-" + str(record["ordinal"])).encode()).hexdigest(),
            stdout_bytes=100, forwarding_status="write_completed"))
    if returncode is None:
        returncode = 2 if any(event["owner_ok"] is False for event in events) else 0
    return dict(schema=forwarding.WORKER_SCHEMA, bridge_id=bridge_id, status="observed", complete=True,
        reason=None, session_id=value["session"]["session_id"], descriptor_sha256=value["descriptor_sha256"],
        returncode=returncode, owner_exchange_attempts=len(events), owner_exchange_unconfirmed=0,
        observed_responses=len(events), events=events, worker_reported=True, independently_verified=False,
        provider_ingestion_verified=False, token_savings_measured=False, proof_authority=False,
        execution_authority=False, completion_authority=False, raw_content_exported=False,
        observer_provider_calls=0)


def _result(evidence, *receipts):
    return forwarding.summarize_worker_forwarding(**evidence,
        forwarding_receipts=[protocol.canonical_json(value) for value in receipts])


def _unknown(result, reason):
    assert result["status"] == "unavailable" and result["reason"] == reason
    assert result["counts"] is None and result["forwarding_receipt_count"] is None
    assert result["independent_worker_receipt_verified"] is result["provider_ingestion_verified"] is False
    return result


def _store(evidence, value):
    evidence["audit_raw"] = protocol.canonical_json(value)


def test_exact_worker_claims_are_scoped_to_reported_pipe_writes(monkeypatch):
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence)
    before = deepcopy((evidence, receipt))
    def no_io(*args, **kwargs):
        raise AssertionError("pure joins must not open state or execute providers")
    monkeypatch.setattr("builtins.open", no_io)
    result = _result(evidence, receipt)
    assert (evidence, receipt) == before
    assert result["status"] == "observed" and result["reason"] is None
    assert result["counts"] == dict(reported_received_operations=2, reported_received_tool_results=1,
        reported_received_successful_reads=1, reported_completed_stdout_operations=2,
        reported_completed_stdout_tool_results=1, reported_completed_stdout_successful_reads=1,
        unconfirmed_stdout_operations=0)
    assert result["worker_reported_only"] is True
    assert all(result[key] is False for key in ("independent_worker_receipt_verified",
        "provider_ingestion_verified", "output_payloads_independently_verified", "token_savings_measured",
        "proof_authority", "execution_authority", "completion_authority", "production_route_qualified"))
    assert result["forwarding_receipt_sha256s"] == [hashlib.sha256(protocol.canonical_json(receipt)).hexdigest()]
    public = protocol.canonical_json(result)
    assert b"/authored" not in public and receipt["bridge_id"].encode() not in public
    assert b"stdout_sha256" not in public and b"request_sha256" not in public


@pytest.mark.parametrize("receipts", [None, (), {}, [], [b"{}"] * 17])
def test_missing_or_unbounded_receipt_population_never_becomes_zero(receipts):
    result = forwarding.summarize_worker_forwarding(**authored_correlated_owner_evidence(), forwarding_receipts=receipts)
    _unknown(result, "worker_evidence_unavailable")


@pytest.mark.parametrize("raw", [None, "{}", b"", b"{}", b"\xff", b'{"x":1,"x":2}',
    b'{"x":NaN}', b'{"x":"\\ud800"}', b"[" * 1024 + b"]" * 1024,
    b" " * (forwarding.MAX_RECEIPT_BYTES + 1)])
def test_malformed_worker_receipts_are_bounded_unknown_evidence(raw):
    result = forwarding.summarize_worker_forwarding(**authored_correlated_owner_evidence(), forwarding_receipts=[raw])
    _unknown(result, "invalid_worker_receipt")


@pytest.mark.parametrize("path,bad", [("schema", "unknown"), ("bridge_id", "bad"),
    ("worker_reported", False), ("independently_verified", True), ("provider_ingestion_verified", True),
    ("token_savings_measured", True), ("proof_authority", True), ("execution_authority", True),
    ("completion_authority", True), ("raw_content_exported", True), ("observer_provider_calls", False),
    ("observer_provider_calls", 1), ("owner_exchange_attempts", True), ("owner_exchange_attempts", 65),
    ("owner_exchange_unconfirmed", False), ("observed_responses", True), ("events", {}),
    ("event.exchange_ordinal", True), ("event.exchange_ordinal", 2), ("event.request_sha256", "x" * 64),
    ("event.owner_response_sha256", None), ("event.owner_response_bytes", False),
    ("event.owner_response_bytes", 0), ("event.owner_ok", 1), ("event.owner_reason_code", "unknown"),
    ("event.output_kind", "source"), ("event.stdout_sha256", None), ("event.stdout_bytes", True),
    ("event.stdout_bytes", 0), ("event.stdout_bytes", protocol.MAX_MESSAGE_RESPONSE_BYTES + 2),
    ("event.forwarding_status", "pending"), ("event.forwarding_status", "no_output")])
def test_closed_worker_fields_reject_overclaims_and_invalid_shapes(path, bad):
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence)
    if path.startswith("event."):
        receipt["events"][0][path.split(".")[1]] = bad
    else:
        receipt[path] = bad
    _unknown(_result(evidence, receipt), "invalid_worker_receipt")


@pytest.mark.parametrize("field", ["session_id", "descriptor_sha256"])
def test_worker_identity_must_match_the_selected_owner_session(field):
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence)
    receipt[field] = "0" * len(receipt[field])
    _unknown(_result(evidence, receipt), "worker_identity_mismatch")


@pytest.mark.parametrize("field,bad", [("status", "not_started"), ("status", "incomplete"),
    ("complete", False), ("reason", "observation_failed"), ("returncode", None),
    ("returncode", True), ("returncode", 1), ("owner_exchange_unconfirmed", 1),
    ("owner_exchange_attempts", 3), ("observed_responses", 1)])
def test_incomplete_observation_or_unknown_exchange_prevents_totals(field, bad):
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence)
    receipt[field] = bad
    _unknown(_result(evidence, receipt), "incomplete_worker_receipt")


@pytest.mark.parametrize("where", ["receipt", "event"])
@pytest.mark.parametrize("change", ["extra", "missing"])
def test_receipt_and_event_schemas_are_closed(where, change):
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence)
    target = receipt if where == "receipt" else receipt["events"][0]
    if change == "extra": target["other"] = False
    else: target.pop(next(iter(target)))
    _unknown(_result(evidence, receipt), "invalid_worker_receipt")


def test_multiple_bridge_receipt_order_is_not_global_exchange_order():
    evidence = authored_correlated_owner_evidence()
    first = authored_worker_forwarding_receipt(evidence, bridge_id="1" * 32, ordinals={1})
    second = authored_worker_forwarding_receipt(evidence, bridge_id="2" * 32, ordinals={2})
    result = _result(evidence, second, first)
    assert result["status"] == "observed" and result["forwarding_receipt_count"] == 2
    assert result["counts"]["reported_completed_stdout_operations"] == 2


@pytest.mark.parametrize("kind", ["bridge", "response", "within_bridge_order"])
def test_duplicate_or_reordered_worker_evidence_is_not_attributed(kind):
    evidence = authored_correlated_owner_evidence()
    first = authored_worker_forwarding_receipt(evidence, bridge_id="1" * 32, ordinals={1})
    second = authored_worker_forwarding_receipt(evidence, bridge_id="2" * 32, ordinals={2})
    if kind == "bridge": second["bridge_id"] = first["bridge_id"]
    elif kind == "response": second["events"] = deepcopy(first["events"])
    else:
        first = authored_worker_forwarding_receipt(evidence)
        first["events"].reverse()
        for ordinal, event in enumerate(first["events"], 1): event["exchange_ordinal"] = ordinal
        _unknown(_result(evidence, first), "ambiguous_worker_population")
        return
    _unknown(_result(evidence, first, second), "ambiguous_worker_population")


def test_incomplete_owner_population_is_not_a_qualified_subtotal():
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence, ordinals={1})
    _unknown(_result(evidence, receipt), "incomplete_owner_population")


@pytest.mark.parametrize("field", ["request_sha256", "owner_response_sha256", "owner_response_bytes"])
def test_exact_owner_request_and_final_response_are_required(field):
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence)
    receipt["events"][1][field] = 241 if field.endswith("bytes") else "0" * 64
    _unknown(_result(evidence, receipt), "owner_response_mismatch")


def test_validated_client_reply_can_precede_owner_final_checkpoint_failure():
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    value["requests"][1].update(delivered=False, status="transport_error")
    _store(evidence, value)
    result = _result(evidence, authored_worker_forwarding_receipt(evidence))
    assert result["status"] == "observed"
    assert result["counts"]["reported_received_successful_reads"] == 1
    assert result["counts"]["reported_completed_stdout_successful_reads"] == 1
    assert result["independent_worker_receipt_verified"] is False


@pytest.mark.parametrize("status", ["unconfirmed", "budget_refused"])
def test_partial_or_budget_refused_output_preserves_known_received_results(status):
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence, returncode=2)
    receipt["events"][1]["forwarding_status"] = status
    result = _result(evidence, receipt)
    assert result["status"] == "observed"
    assert result["counts"]["reported_received_operations"] == 2
    assert result["counts"]["reported_received_successful_reads"] == 1
    assert result["counts"]["reported_completed_stdout_operations"] == 1
    assert result["counts"]["reported_completed_stdout_successful_reads"] == 0
    assert result["counts"]["unconfirmed_stdout_operations"] == 1


def test_budget_replaced_owner_result_is_not_received_or_forwarded_operation_payload():
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    value["requests"][1].update(status="budget_exhausted",
        prepared_response_bytes=value["session"]["max_total_response_bytes"],
        response_sha256="a" * 64, response_bytes=120)
    value["response_bytes"] = sum(record["response_bytes"] for record in value["requests"])
    _store(evidence, value)
    receipt = authored_worker_forwarding_receipt(evidence, returncode=2)
    result = _result(evidence, receipt)
    assert result["status"] == "observed"
    assert result["counts"]["reported_received_operations"] == 1
    assert result["counts"]["reported_received_successful_reads"] == 0
    assert result["counts"]["reported_completed_stdout_operations"] == 1
    assert result["counts"]["unconfirmed_stdout_operations"] == 1


@pytest.mark.parametrize("replaced", [False, True])
def test_audited_operation_cannot_claim_notification_no_output_even_when_replaced(replaced):
    evidence = authored_correlated_owner_evidence()
    if replaced:
        value = json.loads(evidence["audit_raw"])
        value["requests"][1].update(status="budget_exhausted",
            prepared_response_bytes=value["session"]["max_total_response_bytes"],
            response_sha256="a" * 64, response_bytes=120)
        value["response_bytes"] = sum(record["response_bytes"] for record in value["requests"])
        _store(evidence, value)
    receipt = authored_worker_forwarding_receipt(evidence)
    receipt["events"][1].update(output_kind="no_output", stdout_sha256=None,
                              stdout_bytes=0, forwarding_status="no_output")
    _unknown(_result(evidence, receipt), "owner_response_mismatch")


def test_complete_no_output_event_cannot_report_unconfirmed_write():
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence, returncode=2)
    receipt["events"][1].update(output_kind="no_output", stdout_sha256=None,
                              stdout_bytes=0, forwarding_status="unconfirmed")
    _unknown(_result(evidence, receipt), "invalid_worker_receipt")


@pytest.mark.parametrize("terminal", ["owner_refusal", "budget_refused", "unconfirmed"])
@pytest.mark.parametrize("contradiction", ["zero_returncode", "later_response"])
def test_stop_events_are_final_and_require_failure_returncode(terminal, contradiction):
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence, returncode=2)
    event = receipt["events"][-1 if contradiction == "zero_returncode" else 0]
    if terminal == "owner_refusal":
        event.update(owner_ok=False, owner_reason_code="invalid_request", output_kind="local_error")
    else:
        event["forwarding_status"] = terminal
    if contradiction == "zero_returncode":
        receipt["returncode"] = 0
    _unknown(_result(evidence, receipt), "invalid_worker_receipt")


def test_notification_response_does_not_create_a_forwarded_operation():
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    control = deepcopy(value["requests"][0])
    control.update(ordinal=3, request_sha256="c" * 64, response_sha256="d" * 64,
        prepared_response_sha256="d" * 64, operation_capture=dict(status="complete", operations=[]))
    value["requests"].append(control)
    value["accepted_requests"] = 3
    value["response_bytes"] += control["response_bytes"]
    _store(evidence, value)
    receipt = authored_worker_forwarding_receipt(evidence)
    receipt["events"][2].update(output_kind="no_output", stdout_sha256=None,
                              stdout_bytes=0, forwarding_status="no_output")
    result = _result(evidence, receipt)
    assert result["status"] == "observed"
    assert result["counts"]["reported_received_operations"] == 2
    assert result["counts"]["reported_completed_stdout_operations"] == 2


def test_request_replay_requires_unique_composite_response_identity():
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    replay = deepcopy(value["requests"][0])
    replay.update(ordinal=3, status="invalid_request", prepared_response_sha256="c" * 64,
        response_sha256="c" * 64, operation_capture=dict(status="not_entered", operations=[]))
    value["requests"].append(replay)
    value["accepted_requests"] = 3
    value["response_bytes"] += replay["response_bytes"]
    _store(evidence, value)
    result = _result(evidence, authored_worker_forwarding_receipt(evidence, returncode=2))
    assert result["status"] == "observed" and result["counts"]["reported_completed_stdout_operations"] == 2


def test_matching_untrusted_stdout_digest_cannot_gain_authentication():
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence)
    receipt["events"][1]["stdout_sha256"] = "0" * 64
    result = _result(evidence, receipt)
    assert result["status"] == "observed"
    assert result["counts"]["reported_completed_stdout_successful_reads"] == 1
    assert result["worker_reported_only"] is True
    assert result["output_payloads_independently_verified"] is result["independent_worker_receipt_verified"] is False


def test_legacy_owner_evidence_has_no_worker_operation_attribution():
    evidence = authored_owner_evidence()
    _unknown(_result(evidence, authored_worker_forwarding_receipt(evidence)), "owner_correlation_unavailable")


def test_incomplete_owner_operation_capture_keeps_worker_counts_unknown():
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    value["requests"][0]["operation_capture"]["status"] = "incomplete"
    _store(evidence, value)
    _unknown(_result(evidence, authored_worker_forwarding_receipt(evidence)), "owner_correlation_unavailable")


def test_invalid_original_owner_evidence_is_never_rescued_by_worker_claims():
    evidence = authored_correlated_owner_evidence()
    receipt = authored_worker_forwarding_receipt(evidence)
    evidence["operations_raw"] += b"changed"
    _unknown(_result(evidence, receipt), "owner_evidence_unavailable")


def test_zero_is_reserved_for_retained_empty_owner_and_closed_bridge_population():
    evidence = authored_correlated_owner_evidence()
    value = json.loads(evidence["audit_raw"])
    value.update(requests=[], accepted_requests=0, response_bytes=0,
        prepared_mcp_usage={**mcp.summarize_repository_audit_bytes(b""),
            "store_cleanup_complete": True, "artifacts_retained_after_invocation": False})
    value["retained_mcp_audit"].update(sha256=hashlib.sha256(b"").hexdigest(), bytes=0)
    evidence["operations_raw"] = b""
    _store(evidence, value)
    result = _result(evidence, authored_worker_forwarding_receipt(evidence))
    assert result["status"] == "observed" and result["counts"] == dict.fromkeys(forwarding.COUNTERS, 0)
