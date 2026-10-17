"""Prepared-read observations from authored metadata, without native workloads."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_delivery_observation as collector
from benchmarks.agent_supervisor.container_coding import terminal_repository_source_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_read_observation as reads


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _wire(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_row(sequence, *, generation=1, representation="compact", content="value = 'Ω'\n",
              legacy=False, success=True):
    arguments = {"generation": generation, "path": "private_source.py", "representation": representation}
    payload = {"generation": generation, "path": "private_source.py", "source_sha256": _sha("original"),
               "representation": representation, "encoding": "utf-8", "content": content,
               "proof_authority": False} if success else {"error": "stale_generation"}
    row = {"schema": "repository-source-mcp-audit@1" if legacy else "repository-source-mcp-audit@3",
           "sequence": sequence, "event": "tool", "tool": "repository_read", "success": success,
           "returned_bytes": len(_wire(payload)), "compact": success and representation == "compact",
           "canonical_changed": False, "canonical_uncertain": False, "refreshed": False}
    if not legacy:
        row["read_observation"] = reads.make_read_event(arguments, payload, success)
    return row


def _usage(*rows):
    usage = dict(status="observed", initializations=1, tool_calls=len(rows),
        successful_reads=sum(row["success"] for row in rows),
        compact_reads=sum(row["compact"] for row in rows), canonical_edits=0,
        uncertain_edits=0, refreshes=0, errors=sum(not row["success"] for row in rows),
        returned_bytes=sum(row["returned_bytes"] for row in rows),
        store_cleanup_complete=True, artifacts_retained_after_invocation=False)
    summary = reads.summarize_read_observation(list(rows))
    if summary is not None:
        usage["read_observation"] = summary
    return usage


def _binding(usage):
    return dict(schema="supervisor-repository-source-policy@1", requested_policy="repository-source-compact@1",
        native_tool_interception=False, helper_adoption_guaranteed=False, token_savings_measured=False,
        proof_authority=False, completion_authority=False, usage=usage)


def _collect(usage, *, binding_change=None):
    binding = _binding(usage)
    binding.update(binding_change or {})
    invocation = dict(schema="router-implementation-invocation@1", invocation_id="authored-read-call",
        provider="codex_cli", model="gpt-6.1-sol", phase="coding", purpose="coding",
        status="provider_returned", model_prompt_sha256=_sha("prompt"), model_prompt_bytes=6,
        repository_source_policy=binding, native_rollout_usage={
            "schema": "native-codex-cumulative-usage@1", "thread_id": "native-session-123456789",
            "rollout_sha256": _sha("snapshot"), "observed_token_count_records": 1,
            "usage_available": True, "task_complete_observed": True, "cache_included_in_input": True,
            "billing_total_verified": False,
            "usage": {"input_tokens": 100, "cached_input_tokens": 60,
                      "output_tokens": 20, "total_tokens": 120}})
    result = collector.collect(report={"provider_invocations": [invocation],
                                       "unreceipted_provider_attempt": False})
    return result, result["repository_source_mcp"]["observations"][0]


def test_valid_prepared_read_summary_survives_policy_and_collector_without_token_charge():
    usage = _usage(_read_row(1), _read_row(2), _read_row(3, generation=2))
    expected = deepcopy(usage["read_observation"])
    assert expected["known_first_identity_reads"] == 1
    assert expected["known_same_generation_repeat_reads"] == 1
    assert expected["known_cross_generation_repeat_reads"] == 1
    assert expected["known_content_bytes"] == 3 * len("value = 'Ω'\n".encode("utf-8"))
    projected = policy._usage_observation(usage)
    assert projected["server_audit_valid"] is True
    assert projected["read_observation_valid"] is True
    assert projected["read_observation"] == expected
    result, observed = _collect(usage)
    assert observed["read_observation_valid"] is True
    assert observed["read_observation"] == expected
    assert observed["prepared_counts"]["successful_reads"] == 3
    assert observed["prepared_counts"]["returned_bytes"] == usage["returned_bytes"]
    assert result["native_invocations"]["complete_native_total_tokens"] == 120
    assert result["native_invocations"]["delivery_bytes_added_to_native_tokens"] is False
    assert result["repository_source_mcp"]["provider_ingestion_observed"] is None
    assert result["repository_source_mcp"]["tool_token_attribution"] is None
    assert result["join_coverage"]["source_owner_response_to_native_output_digest_matches"] is None
    assert "private_source.py" not in json.dumps(result)
    assert "value =" not in json.dumps(result)


def test_absent_legacy_read_summary_preserves_previous_projection_shape():
    usage = _usage(_read_row(1, legacy=True))
    assert "read_observation" not in usage
    projected = policy._usage_observation(usage)
    _, observed = _collect(usage)
    for value in (projected, observed):
        assert "read_observation" not in value
        assert "read_observation_valid" not in value
    assert observed["prepared_counts"]["successful_reads"] == 1


@pytest.mark.parametrize("value", [None, {}, [], True, {"raw_source": "PRIVATE-READ-MARKER"}])
def test_malformed_optional_summary_does_not_invalidate_base_counts_or_native_usage(value):
    usage = _usage(_read_row(1))
    usage["read_observation"] = value
    projected = policy._usage_observation(usage)
    assert projected["server_audit_valid"] is True
    assert projected["successful_reads"] == 1
    assert projected["read_observation_valid"] is False
    assert projected["read_observation"] is None
    result, observed = _collect(usage)
    assert observed["status"] == "observed"
    assert observed["prepared_counts"]["successful_reads"] == 1
    assert observed["read_observation_valid"] is False
    assert observed["read_observation"] is None
    assert result["native_invocations"]["complete_native_total_tokens"] == 120
    assert "PRIVATE-READ-MARKER" not in json.dumps(result)


@pytest.mark.parametrize("counter,value", [
    ("successful_reads", 0), ("compact_reads", 0), ("returned_bytes", 2),
])
def test_read_summary_must_match_independently_valid_base_population(counter, value):
    usage = _usage(_read_row(1))
    if counter == "successful_reads":
        usage["compact_reads"] = 0
    usage[counter] = value
    projected = policy._usage_observation(usage)
    assert projected["server_audit_valid"] is True
    assert projected["read_observation_valid"] is False
    assert projected["read_observation"] is None
    result, observed = _collect(usage)
    assert observed["prepared_counts"][counter] == value
    assert observed["read_observation"] is None
    assert result["native_invocations"]["complete_native_total_tokens"] == 120


def test_valid_but_incomplete_mixed_legacy_summary_is_preserved():
    usage = _usage(_read_row(1, legacy=True), _read_row(2))
    expected = deepcopy(usage["read_observation"])
    assert expected["complete"] is False
    projected = policy._usage_observation(usage)
    assert projected["read_observation_valid"] is True
    assert projected["read_observation"] == expected
    result, observed = _collect(usage)
    assert observed["read_observation_valid"] is True
    assert observed["read_observation"]["complete"] is False
    assert observed["read_observation"] == expected
    assert observed["prepared_counts"]["successful_reads"] == 2
    assert result["native_invocations"]["complete_native_total_tokens"] == 120


def test_summary_is_detached_from_mutable_input_and_other_observations():
    usage = _usage(_read_row(1))
    expected = deepcopy(usage["read_observation"])
    projected = policy._usage_observation(usage)
    _, observed = _collect(usage)
    usage["read_observation"]["groups"].clear()
    assert projected["read_observation"] == expected
    assert observed["read_observation"] == expected
    projected["read_observation"]["groups"].clear()
    assert observed["read_observation"] == expected


@pytest.mark.parametrize("binding_change", [
    {"proof_authority": True}, {"requested_policy": "foreign-policy@1"},
])
def test_valid_read_summary_cannot_override_invalid_repository_binding(binding_change):
    result, observed = _collect(_usage(_read_row(1)), binding_change=binding_change)
    assert observed["status"] == "unavailable"
    assert all(value is None for value in observed["prepared_counts"].values())
    assert observed["read_observation_valid"] is False
    assert observed["read_observation"] is None
    assert result["native_invocations"]["complete_native_total_tokens"] == 120


def test_valid_read_summary_cannot_rescue_invalid_base_counters():
    usage = _usage(_read_row(1))
    usage["tool_calls"] = 0
    projected = policy._usage_observation(usage)
    assert projected["server_audit_valid"] is False
    assert projected["read_observation_valid"] is False
    result, observed = _collect(usage)
    assert observed["status"] == "unavailable"
    assert observed["read_observation"] is None
    assert result["native_invocations"]["complete_native_total_tokens"] == 120


def test_read_errors_cannot_exceed_valid_base_error_population():
    usage = _usage(_read_row(1), _read_row(2, success=False))
    assert usage["read_observation"]["read_error_count"] == 1
    usage["errors"] = 0
    projected = policy._usage_observation(usage)
    assert projected["server_audit_valid"] is True
    assert projected["errors"] == 0
    assert projected["read_observation_valid"] is False
    assert projected["read_observation"] is None
    result, observed = _collect(usage)
    assert observed["prepared_counts"]["errors"] == 0
    assert observed["read_observation_valid"] is False
    assert result["native_invocations"]["complete_native_total_tokens"] == 120


def _authored_probe_with_read_summary():
    # Reuse the existing deployment validator fixture, never run its probe or
    # container fixtures. The audit population below is explicitly authored.
    from benchmarks.agent_supervisor.container_coding.test_terminal_repository_source_deployment import authored_probe_receipt

    receipt = authored_probe_receipt()
    rows = [{"schema": "repository-source-mcp-audit@1", "sequence": 1,
             "event": "initialize", "tool": None, "success": True, "returned_bytes": 0,
             "compact": False, "canonical_changed": False, "canonical_uncertain": False,
             "refreshed": False}]
    rows.extend(_read_row(index + 2, representation="compact" if index < 2 else "original")
                for index in range(5))
    other_tools = ["repository_status", "repository_cache_inventory", "repository_symbols",
                   "repository_context_query", "repository_capsule_expand",
                   "repository_replace_symbol", "repository_replace_symbol",
                   "repository_status", "repository_status"]
    for index, tool in enumerate(other_tools):
        success = index < 7
        payload = {"ok": True} if success else {"error": "authored_refusal"}
        rows.append({"schema": "repository-source-mcp-audit@1", "sequence": len(rows) + 1,
            "event": "tool", "tool": tool, "success": success, "returned_bytes": len(_wire(payload)),
            "compact": False, "canonical_changed": index in {5, 6}, "canonical_uncertain": False,
            "refreshed": index in {5, 6}})
    receipt["usage"]["returned_bytes"] = sum(row["returned_bytes"] for row in rows)
    receipt["usage"]["read_observation"] = reads.summarize_read_observation(rows)
    assert sum(row["event"] == "tool" for row in rows) == 14
    assert sum(row["tool"] == "repository_read" and row["success"] for row in rows) == 5
    assert sum(row["compact"] for row in rows) == 2
    return receipt


def test_probe_receipt_accepts_current_read_summary_without_executing_probe():
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_probe import validate_probe

    receipt = _authored_probe_with_read_summary()
    before = deepcopy(receipt)
    assert validate_probe(receipt) == before
    assert receipt == before
    assert receipt["usage"]["read_observation"]["complete"] is True
    assert receipt["provider_calls"] == 0
    assert receipt["live_model_adoption_qualified"] is False


def test_probe_receipt_without_optional_read_summary_still_validates():
    from benchmarks.agent_supervisor.container_coding.test_terminal_repository_source_deployment import authored_probe_receipt
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_probe import validate_probe

    receipt = authored_probe_receipt()
    assert "read_observation" not in receipt["usage"]
    assert validate_probe(receipt) == receipt
    assert "read_observation" not in receipt["usage"]


@pytest.mark.parametrize("mutation", ["null", "read_count", "content_bytes", "payload_bytes"])
def test_probe_receipt_rejects_present_forged_or_null_read_summary(mutation):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_probe import validate_probe

    receipt = _authored_probe_with_read_summary()
    summary = receipt["usage"]["read_observation"]
    if mutation == "null":
        receipt["usage"]["read_observation"] = None
    elif mutation == "read_count":
        summary["successful_reads"] += 1
    elif mutation == "content_bytes":
        summary["known_content_bytes"] += 1
    else:
        summary["other_operation_payload_bytes"] += 1
    with pytest.raises(ValueError):
        validate_probe(receipt)
