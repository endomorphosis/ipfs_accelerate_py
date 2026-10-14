"""Terminal accounting joins retain unavailable populations and separate scopes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_delivery_observation as adapter
from ipfs_accelerate_py.agent_supervisor.runtime import context_delivery_accounting as core


def invocation(identity="router-1", **changes):
    row = dict(schema="router-implementation-invocation@1", invocation_id=identity,
        provider="codex_cli", model="gpt-6.1-sol", phase="coding", purpose="coding",
        status="provider_returned", model_prompt_sha256="a" * 64, model_prompt_bytes=123,
        native_rollout_usage=None)
    row.update(changes)
    return row


def source_session(identity="router-1"):
    counts = dict(accepted_requests=3, delivered_responses=2, prepared_pages=2,
        delivered_pages=1, prepared_source_bytes=70, delivered_source_bytes=30,
        response_bytes=100, delivered_response_bytes=60, request_hash_count=3, response_hash_count=3)
    return dict(session_id_sha256="b" * 64, invocation_id_sha256=core.invocation_id_sha256(identity),
        audit_sha256="c" * 64, descriptor_sha256="d" * 64, profile="python-comments@1",
        semantic_sha256="e" * 64, source_manifest_sha256="f" * 64, counts=counts,
        candidate_context_counts=dict(prepared_pages=1, delivered_pages=1, prepared_bytes=5, delivered_bytes=5),
        discovery_counts=dict(prepared_pages=0, delivered_pages=0, prepared_rows=0, delivered_rows=0))


def source_report(*sessions, status="observed"):
    return dict(schema="terminal-source-view-observation@1", status=status, sessions=list(sessions),
        observation_only=True, proof_authority=False, execution_authority=False,
        completion_authority=False, settlement_authority=False, retry_authority=False)


def report_with_source():
    row = invocation(source_view_retrieval={"selected": True})
    return dict(provider_invocations=[row], unreceipted_provider_attempt=False,
                source_view_retrieval_observation=source_report(source_session()))


def mcp_binding(status="observed"):
    counters = dict(initializations=1, tool_calls=1, successful_reads=1, compact_reads=1,
                    canonical_edits=0, uncertain_edits=0, refreshes=0, errors=0, returned_bytes=100)
    if status == "unavailable":
        counters = dict.fromkeys(counters)
    elif status == "not_observed":
        counters = dict.fromkeys(counters, 0)
    return dict(schema="supervisor-repository-source-policy@1", requested_policy="repository-source-compact@1",
        native_tool_interception=False, helper_adoption_guaranteed=False, token_savings_measured=False,
        proof_authority=False, completion_authority=False,
        usage=dict(status=status, **counters, store_cleanup_complete=True,
                   artifacts_retained_after_invocation=False))


def test_complete_owner_join_keeps_delivery_and_native_usage_separate():
    report = report_with_source()
    original = deepcopy(report)
    result = adapter.collect(report=report)
    source = result["source_view_retrieval"]
    assert source["status"] == "observed" and source["matched_session_count"] == 1
    assert source["counts"]["prepared_source_bytes"] == 70
    assert source["counts"]["delivered_source_bytes"] == 30
    assert source["counts"]["response_bytes"] == 100
    assert source["candidate_context_counts"]["prepared_bytes"] == 5
    assert result["join_coverage"]["source_owner_response_to_native_output_digest_matches"] is None
    assert result["join_coverage"]["per_tool_provider_tokens"] is None
    assert result["provider_ingestion_observed"] is None
    assert result["token_savings_established"] is result["completion_authority"] is False
    assert result["status"] == "observed" and result["raw_content_exported"] is False
    assert result["provider_ingestion_verified"] is result["total_token_savings_measured"] is False
    assert report == original


def test_phase_fallback_does_not_replace_existing_purpose(monkeypatch):
    seen = []
    original = core.observe_invocation
    def observe(value):
        seen.append(dict(value))
        return original(value)
    monkeypatch.setattr(core, "observe_invocation", observe)
    first = invocation(); first.pop("purpose")
    second = invocation("router-2", phase="coding", purpose="planning")
    adapter.collect(report=dict(provider_invocations=[first, second]))
    assert seen[0]["purpose"] == "coding" and seen[1]["purpose"] == "planning"


@pytest.mark.parametrize("marker,expected", [
    ({"phase": "planning", "usage": None, "reason": "private failure detail"}, True),
    ({}, True), (True, True), (False, False), (None, None), (0, None), ("false", None),
])
def test_unreceipted_attempt_marker_is_conservative(marker, expected):
    result = adapter.collect(report=dict(provider_invocations=[invocation()], unreceipted_provider_attempt=marker))
    assert result["unreceipted_attempt_marker"] is expected
    assert "private failure detail" not in json.dumps(result)


def test_unmatched_source_session_is_retained_without_zero_or_subtotal():
    report = report_with_source()
    report["source_view_retrieval_observation"]["sessions"][0]["invocation_id_sha256"] = "0" * 64
    source = adapter.collect(report=report)["source_view_retrieval"]
    assert source["status"] == "partial" and source["counts"] is None
    assert source["declared_session_count"] == 1 and source["matched_session_count"] == 0
    assert source["missing_selected_invocation_count"] == 1
    assert source["sessions"][0]["status"] == "unmatched"


@pytest.mark.parametrize("change", ["duplicate_invocation", "duplicate_session"])
def test_duplicate_source_bindings_are_ambiguous(change):
    report = report_with_source()
    if change == "duplicate_invocation":
        report["provider_invocations"].append(deepcopy(report["provider_invocations"][0]))
    else:
        report["source_view_retrieval_observation"]["sessions"].append(source_session())
    source = adapter.collect(report=report)["source_view_retrieval"]
    assert source["status"] == "partial" and source["counts"] is None
    assert source["matched_session_count"] == 0


@pytest.mark.parametrize("change", ["bool_counter", "negative_counter", "delivered_exceeds_prepared", "foreign_profile", "bad_hash"])
def test_malformed_owner_projection_cannot_claim_delivered_source(change):
    report = report_with_source()
    value = report["source_view_retrieval_observation"]["sessions"][0]
    if change == "bool_counter": value["counts"]["prepared_source_bytes"] = True
    elif change == "negative_counter": value["counts"]["response_bytes"] = -1
    elif change == "delivered_exceeds_prepared": value["counts"]["delivered_source_bytes"] = 71
    elif change == "foreign_profile": value["profile"] = "foreign"
    else: value["audit_sha256"] = "bad"
    source = adapter.collect(report=report)["source_view_retrieval"]
    assert source["status"] == "partial" and source["counts"] is None
    assert source["sessions"][0]["status"] == "invalid"


@pytest.mark.parametrize("status", ["unavailable", "not_selected", "not_applicable"])
def test_owner_unavailable_or_inapplicable_is_not_zero_usage(status):
    report = report_with_source()
    report["source_view_retrieval_observation"] = source_report(status=status)
    result = adapter.collect(report=report)["source_view_retrieval"]
    assert result["status"] == status and result["counts"] is None


def test_missing_selected_owner_evidence_remains_unknown():
    report = report_with_source(); report.pop("source_view_retrieval_observation")
    result = adapter.collect(report=report)
    assert result["source_view_retrieval"]["status"] == "not_observed"
    assert result["join_coverage"]["selected_source_invocations_missing_session"] == 1


def test_authoritative_owner_observation_is_refused():
    report = report_with_source()
    report["source_view_retrieval_observation"]["completion_authority"] = True
    result = adapter.collect(report=report)["source_view_retrieval"]
    assert result["status"] == "unavailable" and result["counts"] is None


@pytest.mark.parametrize("field,value", [("status", {}), ("status", []), ("completion_authority", {}),
                                         ("observation_only", 1)])
def test_corrupt_owner_fields_do_not_drop_unrelated_native_summary(field, value):
    report = report_with_source()
    report["source_view_retrieval_observation"][field] = value
    result = adapter.collect(report=report)
    assert result["supplied_invocation_count"] == 1
    assert result["source_view_retrieval"]["status"] == "unavailable"
    assert result["source_view_retrieval"]["counts"] is None


def test_repository_mcp_counts_are_prepared_only():
    row = invocation(repository_source_policy=mcp_binding())
    result = adapter.collect(report=dict(provider_invocations=[row]))["repository_source_mcp"]
    assert result["status"] == "observed"
    assert result["observations"][0]["prepared_counts"]["returned_bytes"] == 100
    assert result["delivery_acknowledgements"] is result["displayed_source_bytes"] is None
    assert result["provider_ingestion_observed"] is result["tool_token_attribution"] is None


@pytest.mark.parametrize("status", ["unavailable", "not_observed"])
def test_repository_mcp_unknown_and_explicit_zero_remain_distinct(status):
    row = invocation(repository_source_policy=mcp_binding(status))
    result = adapter.collect(report=dict(provider_invocations=[row]))["repository_source_mcp"]["observations"][0]
    assert result["status"] == status
    assert result["prepared_counts"]["returned_bytes"] is None if status == "unavailable" else result["prepared_counts"]["returned_bytes"] == 0


def test_impossible_mcp_counts_and_authority_are_not_observations():
    binding = mcp_binding(); binding["usage"]["successful_reads"] = 2
    row = invocation(repository_source_policy=binding)
    first = adapter.collect(report=dict(provider_invocations=[row]))["repository_source_mcp"]["observations"][0]
    assert first["status"] == "unavailable" and first["prepared_counts"]["returned_bytes"] is None
    binding = mcp_binding(); binding["helper_adoption_guaranteed"] = True
    row["repository_source_policy"] = binding
    second = adapter.collect(report=dict(provider_invocations=[row]))["repository_source_mcp"]["observations"][0]
    assert second["status"] == "unavailable"


@pytest.mark.parametrize("value", [{}, [], True])
def test_corrupt_mcp_policy_profile_preserves_native_summary(value):
    binding = mcp_binding(); binding["requested_policy"] = value
    result = adapter.collect(report=dict(provider_invocations=[invocation(repository_source_policy=binding)]))
    assert result["supplied_invocation_count"] == 1
    assert result["repository_source_mcp"]["observations"][0]["status"] == "unavailable"


def test_projection_never_exports_private_receipt_or_report_fields(monkeypatch):
    report = report_with_source()
    report["private_report"] = "private-report-text"
    report["provider_invocations"][0].update(prompt="private-prompt-text", source="private-source-text",
        output="private-output-text", api_key="private-credential-text")
    report["source_view_retrieval_observation"]["sessions"][0]["query"] = "private-query-text"
    def refuse(*args, **kwargs):
        pytest.fail("count composition attempted private artifact I/O")
    monkeypatch.setattr(Path, "open", refuse)
    monkeypatch.setattr(Path, "read_bytes", refuse)
    encoded = json.dumps(adapter.collect(report=report))
    assert "private-" not in encoded


def test_invocation_identity_uses_exact_case_sensitive_hash():
    identity = "router-Coding@1"
    report = dict(provider_invocations=[invocation(identity, source_view_retrieval={})],
        source_view_retrieval_observation=source_report(source_session(identity)))
    result = adapter.collect(report=report)["source_view_retrieval"]
    assert result["status"] == "observed"
    assert result["sessions"][0]["invocation_id_sha256"] == hashlib.sha256(identity.encode()).hexdigest()


def test_unsupported_invocation_identity_is_not_normalized():
    with pytest.raises(ValueError):
        adapter.collect(report=dict(provider_invocations=[invocation("router-§-🙂")]))


@pytest.mark.parametrize("value", [None, {}, [None], [invocation()] * (adapter.MAX_INVOCATIONS + 1)])
def test_invalid_or_unbounded_invocation_population_refuses(value):
    with pytest.raises(ValueError):
        adapter.collect(report=dict(provider_invocations=value))


def test_unbounded_source_session_population_is_unavailable():
    report = report_with_source()
    report["source_view_retrieval_observation"]["sessions"] *= adapter.MAX_SOURCE_SESSIONS + 1
    result = adapter.collect(report=report)["source_view_retrieval"]
    assert result["status"] == "unavailable" and result["counts"] is None


@pytest.mark.parametrize("counter_group", ["counts", "candidate_context_counts", "discovery_counts"])
def test_source_counter_overflow_retains_sessions_and_native_summary(counter_group):
    first, second = source_session(), source_session("router-2")
    second["session_id_sha256"] = "7" * 64
    if counter_group == "counts":
        for key in ("prepared_source_bytes", "response_bytes", "delivered_source_bytes", "delivered_response_bytes"):
            first["counts"][key] = adapter.MAX_COUNTER
    elif counter_group == "candidate_context_counts":
        first[counter_group].update(prepared_bytes=adapter.MAX_COUNTER, delivered_bytes=adapter.MAX_COUNTER)
    else:
        first[counter_group].update(prepared_pages=1, delivered_pages=1,
            prepared_rows=adapter.MAX_COUNTER, delivered_rows=adapter.MAX_COUNTER)
        second[counter_group].update(prepared_pages=1, delivered_pages=1, prepared_rows=1, delivered_rows=1)
    report = dict(provider_invocations=[invocation(source_view_retrieval={}),
        invocation("router-2", source_view_retrieval={})],
        source_view_retrieval_observation=source_report(first, second))
    result = adapter.collect(report=report)
    source = result["source_view_retrieval"]
    assert source["status"] == "partial" and source["reason"] == "source_counter_aggregate_overflow"
    assert source["matched_session_count"] == source["declared_session_count"] == 2
    assert len(source["sessions"]) == result["supplied_invocation_count"] == 2
    assert all(item["status"] == "matched" for item in source["sessions"])
    assert all(source[key] is None for key in ("counts", "candidate_context_counts", "discovery_counts"))
    assert result["native_invocations"] is not None and result["status"] == "observed"


def test_output_budget_refuses_without_truncation(monkeypatch):
    monkeypatch.setattr(adapter, "MAX_OBSERVATION_BYTES", 1)
    with pytest.raises(ValueError, match="output bound"):
        adapter.collect(report={})
