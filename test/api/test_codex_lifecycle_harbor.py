"""Authored JSONL exercises the real parser and existing native usage consumers.

These fixtures run no provider, model, task verifier, or worker. Native facts
remain authored observations; parser compatibility establishes no billing or
campaign authority.
"""
from copy import deepcopy
import hashlib
import json

import pytest
from harbor.models.agent.context import AgentContext

from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from benchmarks.agent_supervisor.container_coding import native_usage_accounting as scopes
from benchmarks.agent_supervisor.container_coding import terminal_decision_usage_projection as terminal
from ipfs_accelerate_py.agent_supervisor.runtime import context_delivery_accounting as context_core
from ipfs_accelerate_py.agent_supervisor.runtime import cumulative_usage_reconciliation as cumulative
from ipfs_accelerate_py.agent_supervisor.runtime.codex_usage_receipt import recover_codex_usage


MODEL = "authored-parser-model"
THREAD = "authored-parser-shared-123456789"
PRIVATE = "PRIVATE_AUTHORED_LIFECYCLE_BODY"


def _event(kind, **fields):
    return dict(type="event_msg", payload=dict(type=kind, **fields))


def _counts(inputs=100, cached=50, outputs=20):
    return dict(input_tokens=inputs, cached_input_tokens=cached, output_tokens=outputs,
                reasoning_output_tokens=outputs // 2, total_tokens=inputs + outputs)


def _token(value):
    return _event("token_count", info=dict(total_token_usage=value, private_prompt=PRIVATE))


def _recover(tmp_path, records, *, thread=THREAD):
    home, workspace = tmp_path / "authored-codex-home", tmp_path / "authored-workspace"
    folder = home / "sessions/2026/10/16"
    folder.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(exist_ok=True)
    rows = [dict(type="session_meta", payload=dict(id=thread, cwd=str(workspace)))] + records
    raw = "".join((row if type(row) is str else json.dumps(row)) + "\n" for row in rows).encode()
    path = folder / ("rollout-" + thread + ".jsonl")
    path.write_bytes(raw)
    native = recover_codex_usage(home=home, thread_id=thread, workspace=workspace)
    assert native is not None
    assert native["rollout_sha256"] == hashlib.sha256(raw).hexdigest()
    assert native["completion_basis"] == cumulative.COMPLETION_BASIS
    assert native["billing_total_verified"] is False
    assert PRIVATE not in json.dumps(native) and str(workspace) not in json.dumps(native)
    return native, workspace


def _router(native, workspace, identity, *, failed=False, phase="coding"):
    prompt = ("authored " + identity + " prompt").encode()
    return dict(schema=cumulative.INVOCATION_SCHEMA, invocation_id=identity, provider="codex_cli",
        model=MODEL, workspace=str(workspace), purpose=phase, phase=phase,
        status="failed" if failed else "provider_returned",
        model_prompt_sha256=hashlib.sha256(prompt).hexdigest(), model_prompt_bytes=len(prompt),
        usage=dict(thread_id=native["thread_id"], model_id=MODEL), native_rollout_usage=native)


def _surfaces(rows, **report_fields):
    report = dict(provider_invocations=rows, **report_fields)
    before = deepcopy(report)
    reconciled = cumulative.reconcile_router_cumulative_usage(rows)
    terminal.finalize_usage_accounting(report)
    assert {key: value for key, value in report.items() if key != "provider_usage_accounting"} == before
    final = report["provider_usage_accounting"]
    measured = adapter.measured_usage(report)
    context = AgentContext()
    context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens = 999, 999, 999
    adapter.populate_harbor_usage(context, measured)
    for value in (reconciled, final, measured):
        assert value["campaign_usage_complete"] is None
        assert value["billing_total_verified"] is False
        assert PRIVATE not in json.dumps(value)
    return reconciled, final, measured, context


def _core(rows):
    # The core's explicit supplied-population marker has a different API from
    # Terminal's missing-attempt-key contract. It never proves campaign coverage.
    return context_core.summarize_invocations([context_core.observe_invocation(row) for row in rows],
        unreceipted_provider_attempt=False)


def _collection(*, complete):
    counts = dict.fromkeys(scopes._COLLECTION_COUNTERS, 0)
    counts.update(entries_examined=1, log_files_seen=1, log_files_read=1,
                  bytes_read=100, lines_examined=1, provider_observations=1)
    if not complete:
        counts.update(entries_examined=2, log_files_seen=2, oversized_logs=1)
    return dict(schema="terminal-provider-receipt-collection@1", status="observed" if complete else "incomplete",
        complete=complete, counts=counts, reason_codes=[] if complete else ["oversized_log"],
        scope="retained_state_logs_only", attempt_population_complete=None, provider_call_count=None,
        campaign_call_population_complete=None, observation_only=True,
        **dict.fromkeys(scopes._COLLECTION_FALSE_FIELDS, False))


def test_real_parser_continuation_and_completed_failed_retry_reconcile_once(tmp_path, record_property):
    first, latest = _counts(), _counts(160, 80, 40)
    native_first, workspace = _recover(tmp_path, [_token(first), _event("task_complete")])
    native_latest, _ = _recover(tmp_path, [_token(first), _event("task_complete"),
        _event("turn_started"), _token(latest), _event("task_complete")])
    native_retry, _ = _recover(tmp_path, [_token(_counts(70, 35, 10)), _event("task_complete")],
                               thread="authored-parser-fresh-retry-123456789")
    rows = [_router(native_first, workspace, "planning", phase="planning"),
            _router(native_latest, workspace, "coding"),
            _router(native_retry, workspace, "failed-postprocessing-retry", failed=True)]
    assert all(row["native_rollout_usage"]["usage_lifecycle_complete"] is True for row in rows)
    assert [row["native_rollout_usage"]["observed_token_count_records"] for row in rows] == [1, 2, 1]
    reconciled, final, measured, context = _surfaces(rows)
    selected = [session["provider_total_tokens"] for session in reconciled["sessions"]]
    assert selected == [200, 80] and reconciled["provider_total_tokens"] == 280
    assert final["report_provider_total_tokens"] == measured["total_tokens"] == 280
    assert measured["input_tokens"] == 230 and measured["cached_input_tokens"] == 115
    assert measured["output_tokens"] == 50 and measured["provider_calls"] == 3
    assert measured["harbor_usage_exportable"] is True
    assert (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) == (230, 115, 50)
    assert rows[-1]["status"] == "failed" and native_retry["usage_lifecycle_complete"] is True
    assert _core(rows)["complete_native_total_tokens"] == 280
    raw_total = sum(row["native_rollout_usage"]["usage"]["total_tokens"] for row in rows)
    assert raw_total == 400
    observation = dict(schema="codex-lifecycle-harbor-observation@1",
        case="continuation_and_completed_failed_fresh_retry",
        scope="authored_native_jsonl_to_existing_usage_consumers", authored_native_fixture=True,
        real_parser_exercised=True, retained_router_receipts=3, native_sessions_observed=reconciled["native_sessions_observed"],
        validated_snapshot_records_by_receipt=[native_first["observed_token_count_records"],
            native_latest["observed_token_count_records"], native_retry["observed_token_count_records"]],
        final_rollout_validated_snapshot_records=native_latest["observed_token_count_records"] + native_retry["observed_token_count_records"],
        raw_wrapper_total_tokens=raw_total, selected_session_total_tokens=selected, provider_total_tokens=measured["total_tokens"],
        input_tokens=measured["input_tokens"], cached_input_tokens=measured["cached_input_tokens"], output_tokens=measured["output_tokens"],
        harbor_input_tokens=context.n_input_tokens, harbor_cache_tokens=context.n_cache_tokens, harbor_output_tokens=context.n_output_tokens,
        completed_failed_retry_retained=True, completion_basis=cumulative.COMPLETION_BASIS,
        campaign_usage_complete=None, provider_receipts_authenticated=False, live_provider_execution=False,
        billing_total_verified=False, token_savings_measured=False, completion_authority=False)
    encoded = json.dumps(observation, sort_keys=True, separators=(",", ":"))
    assert len(encoded.encode()) <= 4096
    record_property("codex_lifecycle_harbor_observation", encoded)


def test_real_parser_unfinished_fresh_retry_retains_charge_without_complete_export(tmp_path):
    first, workspace = _recover(tmp_path, [_token(_counts(160, 80, 40)), _event("task_complete")])
    retry, _ = _recover(tmp_path, [_token(_counts(70, 35, 10)), _event("task_aborted")],
                        thread="authored-parser-unfinished-retry-123456789")
    rows = [_router(first, workspace, "complete"), _router(retry, workspace, "unfinished", failed=True)]
    assert retry["usage_lifecycle_complete"] is False
    reconciled, final, measured, context = _surfaces(rows)
    assert reconciled["known_provider_token_subtotal"] == measured["known_provider_token_subtotal"] == 280
    assert reconciled["native_sessions_observed"] == 2 and measured["provider_calls"] == 2
    assert final["report_provider_total_tokens"] is measured["total_tokens"] is None
    assert measured["harbor_usage_exportable"] is False
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None
    assert _core(rows)["complete_native_total_tokens"] is None


@pytest.mark.parametrize("case", ["resumed", "malformed", "invalid_latest", "regression", "no_usage"])
def test_real_parser_incomplete_boundaries_never_export_stale_or_zero_totals(tmp_path, case):
    records, expected = [_token(_counts())], 120
    if case == "resumed":
        records += [_event("task_complete"), _event("turn_started")]
    elif case == "malformed":
        records += ["{private-malformed-json", _event("task_complete")]
    elif case == "invalid_latest":
        records += [_token(dict(_counts(), input_tokens=True)), _event("task_complete")]
    elif case == "regression":
        records += [_token(_counts(90, 45, 18)), _event("task_complete")]
        expected = 108
    else:
        records, expected = [_event("task_complete")], None
    native, workspace = _recover(tmp_path, records)
    assert native["usage_lifecycle_complete"] is False
    if case == "no_usage":
        assert native["task_complete_observed"] is True and native["usage_available"] is False
        assert native["usage"] == {} and native["observed_token_count_records"] == 0
    else:
        assert native["task_complete_observed"] is False
    rows = [_router(native, workspace, case)]
    reconciled, final, measured, context = _surfaces(rows)
    assert reconciled["known_provider_token_subtotal"] == measured["known_provider_token_subtotal"] == expected
    assert final["report_provider_total_tokens"] is measured["total_tokens"] is None
    assert measured["harbor_usage_exportable"] is False
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None
    assert _core(rows)["complete_native_total_tokens"] is None


@pytest.mark.parametrize("latest_input,regression", [(55, True), (70, False)])
def test_real_parser_sparse_observation_cannot_hide_regression_and_monotone_recovery_is_supported(tmp_path, latest_input, regression):
    final_counts = _counts(latest_input, 35, 7)
    native, workspace = _recover(tmp_path, [_token(_counts(50, 30, 5)),
        _token({"input_tokens": 60}), _token(final_counts), _event("task_complete")])
    assert native["observed_token_count_records"] == 2
    assert native["usage_growth"]["observed_token_events"] == 3
    assert native["counter_regression_observed"] is regression
    assert native["task_complete_observed"] is (not regression)
    assert native["usage_lifecycle_complete"] is (not regression)
    assert scopes.native_usage_contract({"provider": "codex_cli"}, native) == (not regression, "native_session_cumulative", True)
    rows = [_router(native, workspace, "sparse-recovery")]
    reconciled, final, measured, context = _surfaces(rows)
    assert reconciled["known_provider_token_subtotal"] == final_counts["total_tokens"]
    if regression:
        assert final["report_provider_total_tokens"] is measured["total_tokens"] is None
        assert _core(rows)["complete_native_total_tokens"] is None
        assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None
    else:
        assert final["report_provider_total_tokens"] == measured["total_tokens"] == final_counts["total_tokens"]
        assert _core(rows)["complete_native_total_tokens"] == final_counts["total_tokens"]
        assert context.n_input_tokens == latest_input and context.n_output_tokens == 7


def test_real_parser_receipt_without_lifecycle_contract_remains_legacy_unknown(tmp_path):
    native, workspace = _recover(tmp_path, [_token(_counts()), _event("task_complete")])
    for field in ("completion_basis", "lifecycle_malformed_records", "usage_lifecycle_complete"):
        native.pop(field)
    reconciled, final, measured, context = _surfaces([_router(native, workspace, "historical")])
    assert native["task_complete_observed"] is True
    assert reconciled["known_provider_token_subtotal"] == 120
    assert "legacy_final_usage_completeness_unknown" in reconciled["sessions"][0]["reasons"]
    assert final["report_provider_total_tokens"] is measured["total_tokens"] is None
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None


@pytest.mark.parametrize("legacy_first", [False, True])
def test_same_digest_old_and_new_lifecycle_metadata_stays_conservative_conflict(tmp_path, legacy_first):
    native, workspace = _recover(tmp_path, [_token(_counts()), _event("task_complete")])
    historical = deepcopy(native)
    for field in ("completion_basis", "lifecycle_malformed_records", "usage_lifecycle_complete"):
        historical.pop(field)
    rows = [_router(native, workspace, "current"), _router(historical, workspace, "historical")]
    if legacy_first:
        rows.reverse()
    reconciled, final, measured, context = _surfaces(rows)
    assert reconciled["sessions"][0]["status"] == "conflict"
    assert "rollout_digest_alias_conflict" in reconciled["sessions"][0]["reasons"]
    assert reconciled["known_provider_token_subtotal"] is None
    assert final["report_provider_total_tokens"] is measured["total_tokens"] is None
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None


@pytest.mark.parametrize("gap", ["marker_none", "marker_empty", "marker_false", "scan_incomplete", "scan_invalid"])
def test_real_parser_complete_receipt_cannot_repair_missing_attempt_or_collection_coverage(tmp_path, gap):
    native, workspace = _recover(tmp_path, [_token(_counts()), _event("task_complete")])
    fields = {}
    if gap.startswith("marker"):
        fields["unreceipted_provider_attempt"] = {"marker_none": None, "marker_empty": {}, "marker_false": False}[gap]
    else:
        collection = _collection(complete=gap == "scan_invalid")
        if gap == "scan_invalid":
            collection["provider_usage_authenticity_verified"] = True
        fields["provider_receipt_collection"] = collection
    reconciled, final, measured, context = _surfaces([_router(native, workspace, "complete-subset")], **fields)
    assert reconciled["provider_total_tokens"] == 120
    assert final["reconciliation"]["known_provider_token_subtotal"] == measured["known_provider_token_subtotal"] == 120
    assert final["report_provider_total_tokens"] is measured["total_tokens"] is None
    assert measured["provider_calls"] is None and measured["observed_provider_invocations"] == 1
    assert measured["harbor_usage_exportable"] is False
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None
