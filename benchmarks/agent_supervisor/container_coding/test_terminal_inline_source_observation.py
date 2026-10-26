"""Authored initial-prompt/read joins; no model, native index or audit replay."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_delivery_observation as collector
from ipfs_accelerate_py.agent_supervisor.runtime.inline_source_observation import observe_inline_sources
from ipfs_accelerate_py.agent_supervisor.runtime.rendered_input_census import census_rendered_input
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_read_observation import make_read_event, summarize_read_observation


SOURCE = "# PRIVATE-SOURCE-MARKER\nanswer = 'Ω'\n"
PATH = "private-source.py"
INVOCATION = "authored-inline-invocation"


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read(sequence, *, path=PATH, source=SOURCE, generation=1, representation="original",
          encoding="utf-8", symbol=False, unbound=False, legacy=False):
    arguments = dict(path=path, generation=generation, representation=representation)
    content = source if encoding == "utf-8" else base64.b64encode(source.encode()).decode("ascii")
    payload = dict(path=path, generation=generation, source_sha256=_sha(source),
                   representation=representation, encoding=encoding, content=content, proof_authority=False)
    if symbol:
        arguments["symbol_id"] = "cid:authored-symbol"
        payload.update(symbol_id=arguments["symbol_id"], scope="python_definition")
    row = dict(schema="repository-source-mcp-audit@1" if legacy else "repository-source-mcp-audit@3",
        sequence=sequence, event="tool", tool="repository_read", success=True,
        returned_bytes=len(_json(payload).encode()), compact=representation == "compact",
        canonical_changed=False, canonical_uncertain=False, refreshed=False)
    if not legacy:
        row["read_observation"] = None if unbound else make_read_event(arguments, payload, True)
        assert unbound or row["read_observation"] is not None
    return row


def _report(*read_rows, inline_sources=None):
    if inline_sources is None:
        inline_sources = {PATH: SOURCE}
    model = _json({"schema": "supervisor-semantic-router-input@1", "translated_semantic": {
        "raw_sources": inline_sources, "manifest": {path: {"sha256": _sha(source)}
        for path, source in inline_sources.items()}}})
    census = census_rendered_input(model_prompt=model, workspace_advisory="", invocation_id=INVOCATION)
    census["inline_sources"] = observe_inline_sources(model_prompt=model, workspace_advisory="", invocation_id=INVOCATION)
    usage = dict(status="observed", initializations=1, tool_calls=len(read_rows),
        successful_reads=len(read_rows), compact_reads=sum(row["compact"] for row in read_rows),
        canonical_edits=0, uncertain_edits=0, refreshes=0, errors=0,
        returned_bytes=sum(row["returned_bytes"] for row in read_rows),
        store_cleanup_complete=True, artifacts_retained_after_invocation=False)
    summary = summarize_read_observation(list(read_rows))
    if summary is not None:
        usage["read_observation"] = summary
    binding = dict(schema="supervisor-repository-source-policy@1", requested_policy="repository-source-raw@1",
        native_tool_interception=False, helper_adoption_guaranteed=False, token_savings_measured=False,
        proof_authority=False, completion_authority=False, usage=usage)
    invocation = dict(schema="router-implementation-invocation@1", invocation_id=INVOCATION,
        provider="codex_cli", model="gpt-6.1-sol", phase="coding", purpose="coding", status="provider_returned",
        model_prompt_sha256=_sha(model), model_prompt_bytes=len(model.encode()), repository_source_policy=binding,
        native_rollout_usage=dict(schema="native-codex-cumulative-usage@1", thread_id="native-session-123456789",
            rollout_sha256=_sha("snapshot"), observed_token_count_records=1, usage_available=True,
            task_complete_observed=True, cache_included_in_input=True, billing_total_verified=False,
            usage=dict(input_tokens=100, cached_input_tokens=60, output_tokens=20, total_tokens=120)))
    audit = dict(schema="terminal-final-context-audit@1", status="verified_all_observed_model_inputs",
        any_model_input_verified=True, all_observed_coding_inputs_verified=True,
        proof_authority=False, execution_authority=False, completion_authority=False,
        raw_prompts_exported=False, provider_calls=0,
        invocations=[dict(invocation_id=INVOCATION, status="verified_exact_model_input",
                          model_input_verified=True, native_input_verified=True)],
        matches=[dict(invocation_id=INVOCATION, model_input_verified=True,
                      model_prompt_sha256=_sha(model), model_prompt_bytes=len(model.encode()),
                      model_input_checks=dict(model_prompt_sha256=True, model_prompt_bytes=True))],
        rendered_input_census=dict(schema="terminal-rendered-input-census@1",
            status="complete_for_verified_inputs", observations=[census], unavailable=[],
            verified_invocation_count=1, observed_invocation_count=1,
            raw_content_exported=False, token_savings_established=False, provider_calls=0))
    return dict(provider_invocations=[invocation], unreceipted_provider_attempt=False,
                context_input_audit=audit)


def _collect(report):
    result = collector.collect(report=report)
    row = result["repository_source_mcp"]["observations"][0]
    assert row["prepared_counts"]["successful_reads"] == report["provider_invocations"][0]["repository_source_policy"]["usage"]["successful_reads"]
    assert result["native_invocations"]["complete_native_total_tokens"] == 120
    return result, row["inline_source_overlap"]


def test_exact_inline_whole_file_matches_each_prepared_original_read():
    report = _report(_read(1), _read(2), _read(3, generation=2))
    before = deepcopy(report)
    result, overlap = _collect(report)
    assert overlap["status"] == "observed"
    assert overlap["comparison_complete"] is True
    assert overlap["matched_reads"] == overlap["known_matched_reads"] == 3
    assert overlap["matched_logical_bytes"] == overlap["known_matched_logical_bytes"] == 3 * len(SOURCE.encode())
    assert overlap["comparable_reads"] == 3 and overlap["unclassified_reads"] == 0
    assert overlap["retained_input_audit_joined"] is True
    assert overlap["journal_closure_verified"] is False
    assert overlap["inline_overlap_bytes"] is overlap["provider_tokens"] is overlap["token_savings"] is None
    assert overlap["full_prompt_source_coverage"] is overlap["provider_ingestion_observed"] is None
    assert result["repository_source_mcp"]["observations"][0]["read_observation"]["inline_overlap_bytes"] is None
    assert result["native_invocations"]["delivery_bytes_added_to_native_tokens"] is False
    assert report == before
    assert PATH not in _json(result) and "PRIVATE-SOURCE-MARKER" not in _json(result)


@pytest.mark.parametrize("change", ["different_path", "changed_source", "empty_inventory"])
def test_observed_zero_is_scoped_to_valid_whole_file_inventory(change):
    row = _read(1, path="other.py" if change == "different_path" else PATH,
                source=SOURCE + "# changed\n" if change == "changed_source" else SOURCE)
    _, overlap = _collect(_report(row, inline_sources={} if change == "empty_inventory" else None))
    assert overlap["status"] == "observed" and overlap["comparison_complete"] is True
    assert overlap["matched_reads"] == overlap["matched_logical_bytes"] == 0
    assert overlap["full_prompt_source_coverage"] is None


@pytest.mark.parametrize("other", ["compact", "symbol", "base64", "unbound", "legacy"])
def test_noncomparable_or_unbound_reads_preserve_known_matches_but_not_full_totals(other):
    options = {"representation": "compact"} if other == "compact" else {"symbol": True} if other == "symbol" else {
        "encoding": "base64"} if other == "base64" else {other: True}
    _, overlap = _collect(_report(_read(1), _read(2, **options)))
    assert overlap["status"] == "partial" and overlap["comparison_complete"] is False
    assert overlap["known_matched_reads"] == 1
    assert overlap["known_matched_logical_bytes"] == len(SOURCE.encode())
    assert overlap["comparable_reads"] == overlap["unclassified_reads"] == 1
    assert overlap["matched_reads"] is overlap["matched_logical_bytes"] is None


@pytest.mark.parametrize("missing", ["audit", "census", "inline", "read_summary"])
def test_missing_historical_observation_is_not_zero(missing):
    report = _report(_read(1))
    if missing == "audit":
        report.pop("context_input_audit")
    elif missing == "census":
        report["context_input_audit"].pop("rendered_input_census")
    elif missing == "inline":
        report["context_input_audit"]["rendered_input_census"]["observations"][0].pop("inline_sources")
    else:
        report["provider_invocations"][0]["repository_source_policy"]["usage"].pop("read_observation")
    _, overlap = _collect(report)
    assert overlap["status"] == "not_observed"
    assert overlap["known_matched_reads"] is overlap["matched_reads"] is None
    assert overlap["known_matched_logical_bytes"] is overlap["matched_logical_bytes"] is None


@pytest.mark.parametrize("population", ["audit_invocations", "matches", "census"])
def test_even_equal_duplicate_join_records_are_ambiguous(population):
    report = _report(_read(1))
    audit = report["context_input_audit"]
    rows = audit["invocations"] if population == "audit_invocations" else audit["matches"] if population == "matches" else audit["rendered_input_census"]["observations"]
    rows.append(deepcopy(rows[0]))
    if population == "census":
        audit["rendered_input_census"].update(observed_invocation_count=2, verified_invocation_count=2)
    _, overlap = _collect(report)
    assert overlap["status"] == "unavailable" and overlap["reason"] == "ambiguous_exact_input_join"
    assert overlap["known_matched_reads"] is overlap["matched_reads"] is None


@pytest.mark.parametrize("purpose", ["coding", "planning"])
def test_duplicate_supplied_invocation_id_is_not_resolved_by_context_filtering(purpose):
    report = _report(_read(1))
    second = deepcopy(report["provider_invocations"][0])
    second.update(purpose=purpose, phase=purpose)
    report["provider_invocations"].append(second)
    result = collector.collect(report=report)
    for row in result["repository_source_mcp"]["observations"]:
        overlap = row["inline_source_overlap"]
        assert overlap["status"] == "unavailable"
        assert overlap["reason"] == "ambiguous_invocation_population"
        assert overlap["matched_reads"] is None


@pytest.mark.parametrize("target,field,value", [
    ("match", "invocation_id", "different-invocation"),
    ("census", "invocation_id_sha256", "f" * 64),
    ("match", "model_prompt_sha256", "f" * 64),
    ("census", "model_prompt_sha256", "f" * 64),
    ("match", "model_prompt_bytes", True),
    ("census", "model_prompt_bytes", 1),
    ("inline", "model_prompt_sha256", "f" * 64),
    ("inline", "invocation_id_sha256", "f" * 64),
])
def test_foreign_or_mismatched_input_binding_never_authorizes_overlap(target, field, value):
    report = _report(_read(1))
    audit = report["context_input_audit"]
    census = audit["rendered_input_census"]["observations"][0]
    row = audit["matches"][0] if target == "match" else census if target == "census" else census["inline_sources"]
    row[field] = value
    _, overlap = _collect(report)
    assert overlap["status"] == "unavailable"
    assert overlap["known_matched_reads"] is overlap["matched_reads"] is None


@pytest.mark.parametrize("mutation", ["unverified_invocation", "false_check", "empty_checks", "audit_status", "census_status", "inline_null", "extra_private_entry"])
def test_invalid_optional_join_keeps_base_and_native_measurements(mutation):
    report = _report(_read(1))
    audit = report["context_input_audit"]
    if mutation == "unverified_invocation":
        audit["invocations"][0]["model_input_verified"] = False
    elif mutation == "false_check":
        audit["matches"][0]["model_input_checks"]["model_prompt_sha256"] = False
    elif mutation == "empty_checks":
        audit["matches"][0]["model_input_checks"] = {}
    elif mutation == "audit_status":
        audit["status"] = []
    elif mutation == "census_status":
        audit["rendered_input_census"]["status"] = []
    elif mutation == "inline_null":
        audit["rendered_input_census"]["observations"][0]["inline_sources"] = None
    else:
        audit["rendered_input_census"]["observations"][0]["inline_sources"]["entries"][0]["path"] = "PRIVATE-ENTRY-MARKER"
    result, overlap = _collect(report)
    assert overlap["status"] == "unavailable"
    assert overlap["matched_reads"] is overlap["known_matched_reads"] is None
    assert "PRIVATE-ENTRY-MARKER" not in _json(result)


def test_same_prompt_hash_in_another_invocation_does_not_supply_missing_history():
    report = _report(_read(1))
    second = deepcopy(report["provider_invocations"][0])
    second["invocation_id"] = "second-invocation"
    second["native_rollout_usage"]["thread_id"] = "second-session-123456789"
    report["provider_invocations"].append(second)
    result = collector.collect(report=report)
    first, second = result["repository_source_mcp"]["observations"]
    assert first["inline_source_overlap"]["matched_reads"] == 1
    assert second["inline_source_overlap"]["matched_reads"] is None
    assert second["inline_source_overlap"]["reason"] == "missing_exact_input_join"
    assert "matched_reads" not in result["repository_source_mcp"]
    assert result["native_invocations"]["complete_native_total_tokens"] == 240


@pytest.mark.parametrize("target,field,value", [
    ("audit", "proof_authority", True), ("audit", "execution_authority", True),
    ("audit", "completion_authority", True), ("audit", "raw_prompts_exported", True),
    ("audit", "provider_calls", False), ("audit", "provider_calls", 1),
    ("census", "provider_calls", False), ("census", "provider_calls", 1),
])
def test_contradictory_retained_audit_claims_do_not_authorize_overlap(target, field, value):
    report = _report(_read(1))
    audit = report["context_input_audit"]
    (audit if target == "audit" else audit["rendered_input_census"])[field] = value
    _, overlap = _collect(report)
    assert overlap["status"] == "unavailable"
    assert overlap["matched_reads"] is overlap["known_matched_reads"] is None


def test_optional_validator_failure_preserves_native_counts_and_hides_exception(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import inline_source_observation as inline
    report = _report(_read(1))
    def broken(*args, **kwargs):
        raise RuntimeError("PRIVATE-EXCEPTION-MARKER")
    monkeypatch.setattr(inline, "validate_inline_sources", broken)
    result, overlap = _collect(report)
    assert overlap["status"] == "unavailable"
    assert overlap["reason"] == "inline_source_diagnostic_failure"
    assert overlap["matched_reads"] is None
    assert "PRIVATE-EXCEPTION-MARKER" not in _json(result)


def test_oversized_optional_population_preserves_native_and_base_counts():
    report = _report(_read(1))
    report["context_input_audit"]["matches"] *= 513
    _, overlap = _collect(report)
    assert overlap["status"] == "unavailable"
    assert overlap["reason"] == "invalid_input_join_population"
    assert overlap["known_matched_reads"] is None


def test_optional_overlap_yields_output_budget_before_base_measurements(monkeypatch):
    report = _report(_read(1))
    enriched = collector.collect(report=report)
    base = deepcopy(enriched)
    base["repository_source_mcp"]["observations"][0].pop("inline_source_overlap")
    monkeypatch.setattr(collector, "MAX_OBSERVATION_BYTES", len(_json(base).encode()))
    result = collector.collect(report=report)
    assert result == base
    assert result["native_invocations"]["complete_native_total_tokens"] == 120
    assert result["repository_source_mcp"]["observations"][0]["prepared_counts"]["successful_reads"] == 1


@pytest.mark.parametrize("mutation", ["also_unavailable", "false_complete", "malformed_unavailable"])
def test_contradictory_census_coverage_keeps_overlap_unknown(mutation):
    report = _report(_read(1))
    census = report["context_input_audit"]["rendered_input_census"]
    if mutation == "also_unavailable":
        census["unavailable"].append(dict(invocation_id_sha256=_sha(INVOCATION),
                                         reason="audit_result_budget_unavailable"))
    elif mutation == "false_complete":
        census["verified_invocation_count"] = 2
    else:
        census["unavailable"] = None
    _, overlap = _collect(report)
    assert overlap["status"] == "unavailable"
    assert overlap["matched_reads"] is overlap["known_matched_reads"] is None


def test_real_mcp_file_reads_join_authored_input_audit_after_private_store_cleanup(tmp_path):
    """Real source/stdio/audit cleanup; authored index, input audit and native usage.

    This qualifies the source identity bridge only. It makes no provider or
    native index execution claim, and does not replay a final-input receipt.
    """
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
    from test.api.test_repository_source_read_transport import authored_index_client, configuration
    config, _, _ = configuration(tmp_path)
    workspace = Path(config["workspace"])
    source = (workspace / "calc.py").read_text()
    report = _report(inline_sources={"calc.py": source})
    binding = policy.create_binding(workspace=workspace, requested_policy=policy.POLICIES[1])
    identity = policy.create_runtime_store(binding)
    try:
        with authored_index_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
            for _ in range(2):
                payload, failed = client.call("repository_read", generation=1, path="calc.py",
                                              representation="original")
                assert not failed and payload["content"] == source
    finally:
        usage = policy.finish_runtime_store(binding, expected_identity=identity)
    assert usage["store_cleanup_complete"] is True
    assert not Path(binding["invocation_dir"]).exists()
    assert usage["read_observation"]["identified_reads"] == 2
    report["provider_invocations"][0]["repository_source_policy"]["usage"] = usage
    result, overlap = _collect(report)
    assert overlap["status"] == "observed" and overlap["comparison_complete"] is True
    assert overlap["matched_reads"] == overlap["known_matched_reads"] == 2
    assert overlap["matched_logical_bytes"] == 2 * len(source.encode())
    assert overlap["journal_closure_verified"] is False
    assert overlap["provider_ingestion_observed"] is overlap["provider_tokens"] is None
    assert "calc.py" not in _json(result) and source not in _json(result)
