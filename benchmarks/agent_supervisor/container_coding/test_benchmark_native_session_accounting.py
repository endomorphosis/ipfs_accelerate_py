"""Authored native JSONL and MCP journals exercise the retained comparison route.

These fixtures call the existing receipt producers and comparator. They do not
call a provider, run a task verifier, or authenticate the authored native facts.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as supervisor
from benchmarks.agent_supervisor.container_coding import native_codex_baseline as baseline
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from ipfs_accelerate_py.agent_supervisor.runtime.codex_usage_receipt import recover_codex_usage


SOURCE = "line1\nline2\nline3\nline4\n"
SOURCE_SHA = hashlib.sha256(SOURCE.encode()).hexdigest()
MODEL = baseline.MODEL
IDENTITY = dict(task=baseline.TASK, model=MODEL, reasoning_effort=baseline.REASONING,
                cli_version=baseline.CLI_VERSION)


def _event(name, **fields):
    return dict(type="event_msg", payload=dict(type=name, **fields))


def _counts(inputs, cached, outputs):
    return dict(input_tokens=inputs, cached_input_tokens=cached,
                output_tokens=outputs, reasoning_output_tokens=outputs // 2,
                total_tokens=inputs + outputs)


def _snapshot(home, workspace, thread, snapshots, *, complete=True):
    folder = home / "sessions/2026/10/15"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [dict(type="session_meta", payload=dict(id=thread, cwd=str(workspace)))]
    for value in snapshots:
        rows.append(_event("token_count", info=dict(total_token_usage=value)))
        rows.append(_event("task_complete"))
    if not complete:
        rows.append(_event("task_aborted"))
    path = folder / ("rollout-" + thread + ".jsonl")
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    observed = recover_codex_usage(home=home, thread_id=thread, workspace=workspace)
    assert observed is not None
    assert observed["rollout_sha256"] == hashlib.sha256(raw).hexdigest()
    assert observed["observed_token_count_records"] == len(snapshots)
    assert observed["task_complete_observed"] is complete
    assert observed["billing_total_verified"] is False
    assert observed["usage"] == snapshots[-1]
    return observed


def _read_binding(tmp_path, name, *, line_range, stale=False):
    """Use the actual content-free producer, then remove its private journal."""
    private = tmp_path / name
    private.mkdir(mode=0o700)
    path = private / "operations.jsonl"
    audit = mcp._Audit(path)
    audit.record("initialize")
    request = dict(generation=1, path="module.txt", representation="original")
    value = dict(generation=1, path="module.txt", source_sha256=SOURCE_SHA,
                 representation="original", encoding="utf-8", content=SOURCE,
                 proof_authority=False)
    if line_range:
        request.update(start_line=2, end_line=3)
        value.update(scope="line_range", start_line=2, end_line=3,
                     total_lines=4, omitted_lines=2, content="line2\nline3\n")
    audit.record("tool", tool="repository_read", returned_bytes=len(mcp._json(value)),
                 read=mcp._read_event(request, value, True))
    if stale:
        error = {"error": "stale_generation"}
        audit.record("tool", tool="repository_read", success=False,
                     returned_bytes=len(mcp._json(error)),
                     read=mcp._read_event(request, error, False))
    audit.close()
    usage = mcp.summarize_repository_audit(path)
    assert usage["status"] == "observed"
    assert usage["read_observation"]["complete"] is True
    path.unlink()
    private.rmdir()
    usage.update(store_cleanup_complete=True, artifacts_retained_after_invocation=False)
    return dict(schema="supervisor-repository-source-policy@1",
        requested_policy="repository-source-raw@1", usage=usage,
        native_tool_interception=False, helper_adoption_guaranteed=False,
        token_savings_measured=False, proof_authority=False, completion_authority=False)


def _invocation(identity, phase, workspace, native, *, failed=False, binding=None):
    prompt = ("authored " + phase + " prompt").encode()
    result = dict(schema="router-implementation-invocation@1", invocation_id=identity,
        provider="codex_cli", model=MODEL, purpose=phase, phase=phase,
        status="failed" if failed else "provider_returned", workspace=str(workspace),
        model_prompt_sha256=hashlib.sha256(prompt).hexdigest(), model_prompt_bytes=len(prompt),
        native_rollout_usage=native)
    if failed:
        result["error_type"] = "AuthoredPostprocessingError"
    if binding is not None:
        result["repository_source_policy"] = binding
    return result


def _declared(config):
    hashes = {"instruction.md": hashlib.sha256(b"authored task bytes").hexdigest()}
    declaration = controls.build_controls(config, task_input_sha256=hashes, **IDENTITY)
    return controls.observe_controls({"comparison_controls": declaration}, config,
                                     current_task_hashes=hashes)


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return path


@pytest.fixture
def campaign(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "authored-codex-home"
    early = _counts(100, 60, 10)
    later = _counts(150, 90, 20)
    thread = "authored-shared-thread-20261015"
    planning = _snapshot(home, workspace, thread, [early])
    coding = _snapshot(home, workspace, thread, [early, later])
    retry = _snapshot(home, workspace, "authored-retry-thread-20261015", [_counts(70, 40, 7)])
    native_baseline = _snapshot(home, workspace, "authored-baseline-thread-20261015",
                                [_counts(80, 40, 8)])
    calls = [
        _invocation("planning", "planning", workspace, planning),
        _invocation("coding", "coding", workspace, coding,
                    binding=_read_binding(tmp_path, "coding-audit", line_range=True, stale=True)),
        _invocation("failed-retry", "coding", workspace, retry, failed=True,
                    binding=_read_binding(tmp_path, "retry-audit", line_range=False)),
    ]
    report = dict(task_completed=True, provider_invocations=calls, unreceipted_provider_attempt=False)
    trial = dict(trial="authored-supervisor", reward={"reward": 1}, supervisor=report,
                 native_result_sha256=hashlib.sha256(b"authored native result.json").hexdigest())
    common = dict(**IDENTITY, original_task_inputs_unchanged=True, trial_count=1)
    base_config = baseline.config_for(tmp_path / "dataset", tmp_path / "baseline")
    full_config = supervisor.config_for(tmp_path / "dataset", tmp_path / "full",
        tmp_path / "archive", "full", repository_source_policy="repository-source-raw@1")
    baseline_value = dict(common, schema=comparison.BASELINE, comparison_controls=_declared(base_config),
        trials=[dict(trial="authored-baseline", reward={"reward": 1}, raw_usage=dict(
            sessions=[dict(session_id=native_baseline["thread_id"], task_complete_observed=True)],
            usage=native_baseline["usage"]))])
    full_value = dict(common, schema=comparison.SUPERVISOR, arm="full",
                      comparison_controls=_declared(full_config), trials=[trial])
    base_path = _write(tmp_path / "baseline.json", baseline_value)
    full_path = _write(tmp_path / "supervisor.json", full_value)
    return dict(root=tmp_path, workspace=workspace, home=home, baseline=base_path,
                full=full_path, value=full_value, report=report, early=early, later=later)


def _collect(campaign):
    _write(campaign["full"], campaign["value"])
    result = comparison.collect(baseline=campaign["baseline"], supervisors=[campaign["full"]])
    row = next(value for value in result["rows"] if value["arm"] == "full")
    return result, row


def test_authored_rollouts_reconcile_planning_coding_and_failed_retry_once(campaign, record_property):
    result, row = _collect(campaign)
    expected = dict(input_tokens=220, cached_input_tokens=130, output_tokens=27, total_tokens=247)
    assert row["tokens"] == row["known_token_subtotals"] == expected
    assert row["native_token_accounting_basis"] == "native_session_reconciled"
    assert row["input_token_scope"] == "native_input_including_cache"
    assert row["native_session_reconciliation_available"] is True
    assert row["shared_native_session_count"] == 1
    assert row["provider_calls"] == row["observed_provider_invocations"] == 3
    assert row["cache_included_in_input"] is True
    assert row["official_reward"] == 1 and row["native_task_completed"] is True
    assert [value["status"] for value in row["invocations"]] == ["provider_returned", "provider_returned", "failed"]
    summary = row["native_session_accounting"]
    assert summary["supplied_invocation_count"] == 3 and summary["native_session_count"] == 2
    assert summary["failed_invocation_count"] == 1
    assert summary["complete_supplied_native_usage"] is True
    assert [value["selected_observation_index"] for value in summary["sessions"]] == [1, 2]
    selected = [value["reported_counts"]["total_tokens"] for value in summary["sessions"]]
    assert selected == [170, 77] and sum(selected) == row["tokens"]["total_tokens"]
    raw_total = sum(call["native_rollout_usage"]["usage"]["total_tokens"]
                    for call in campaign["report"]["provider_invocations"])
    assert raw_total == 357 and raw_total != row["tokens"]["total_tokens"]
    assert summary["complete_native_counts"]["fresh_input_tokens"] is None
    assert summary["complete_native_counts"]["reasoning_output_tokens"] == 13
    for key in ("cumulative_snapshots_are_additive", "delivery_bytes_added_to_native_tokens",
                "local_tokenizer_counts_used_as_native_usage", "input_observation_authenticity_verified"):
        assert summary[key] is False
    delivery = row["context_delivery_observation"]
    assert delivery["native_invocations"] == summary
    repository = delivery["repository_source_mcp"]
    assert repository["selected_invocation_count"] == 2
    assert repository["read_scope_status"] == "observed" and repository["read_scope_complete"] is True
    read = repository["read_scope_counts"]
    assert (read["read_calls"], read["successful_reads"], read["failed_reads"]) == (3, 2, 1)
    assert (read["line_range_reads"], read["full_file_reads"], read["definition_reads"]) == (1, 1, 0)
    assert (read["line_range_selected_lines"], read["line_range_total_lines"], read["line_range_omitted_lines"]) == (2, 4, 2)
    assert read["exposed_content_utf8_bytes"] == 36
    assert repository["provider_ingestion_observed"] is repository["tool_token_attribution"] is None
    assert delivery["join_coverage"]["source_owner_response_to_native_output_digest_matches"] is None
    assert row["reported_cost_usd"] is None and row["whole_interaction_qualified"] is False
    assert result["all_declared_controls_match"] is True
    assert result["retained_trial_count"] == 2
    assert result["arm_totals"]["full"]["observed_efficiency"]["total_tokens_per_official_pass"] == 247
    assert result["causal_claim"] is result["runtime_control_enforcement_verified"] is False
    assert result["arm_totals"]["native-codex"]["tokens"]["total_tokens"] == 88
    observation = dict(schema="native-session-comparison-observation@1",
        measurement_scope="authored_native_jsonl_and_authored_mcp_audits", authored_provider_facts=True,
        supplied_invocations=3, native_sessions=2, shared_native_sessions=1, failed_invocations=1,
        planning_invocations=1, coding_invocations=2, selected_observation_indices=[1, 2],
        raw_cumulative_wrapper_total_tokens=raw_total, selected_native_session_total_tokens=selected,
        tokens=expected, baseline_total_tokens=88, retained_trial_count=2, supervisor_official_passes=1,
        read_scope_counts=read, provider_receipts_authenticity_verified=False, live_provider_execution=False,
        provider_ingestion_verified=False, source_displacement_verified=False,
        tool_token_attribution_available=False, token_savings_measured=False, causal_claim=False,
        runtime_control_enforcement_verified=False, whole_interaction_qualified=False,
        billing_total_verified=False, local_tokenizer_counts_added_to_native_usage=False,
        delivery_bytes_added_to_native_tokens=False)
    record_property("native_session_comparison_observation", json.dumps(observation, sort_keys=True))


def test_interrupted_fresh_retry_retains_observed_subtotal_without_complete_total(campaign):
    call = campaign["report"]["provider_invocations"][-1]
    call["native_rollout_usage"] = _snapshot(campaign["home"], campaign["workspace"],
        "authored-retry-thread-20261015", [_counts(70, 40, 7)], complete=False)
    call["error_type"] = "AuthoredInterruption"
    result, row = _collect(campaign)
    assert row["tokens"] == dict.fromkeys(comparison.TOKENS)
    assert row["known_token_subtotals"] == dict(input_tokens=220, cached_input_tokens=130,
                                               output_tokens=27, total_tokens=247)
    assert row["native_session_accounting"]["failed_invocation_count"] == 1
    assert row["native_session_accounting"]["complete_supplied_native_usage"] is False
    assert row["context_delivery_observation"]["repository_source_mcp"]["read_scope_counts"]["line_range_reads"] == 1
    assert row["official_reward"] == 1  # An authored reward never repairs incomplete usage.
    assert result["arm_totals"]["full"]["observed_efficiency"]["total_tokens_per_official_pass"] is None


def test_comparison_recomputes_observations_and_never_charges_prompt_or_reply_bytes(campaign):
    before, row_before = _collect(campaign)
    campaign["report"].update(context_delivery_observation={"status": "forged", "total_tokens": 999999},
        native_session_accounting={"complete_native_total_tokens": 999999})
    for call in campaign["report"]["provider_invocations"]:
        call["model_prompt_bytes"] += 1000
        call["local_tokenizer_count"] = 999999
    _, after = _collect(campaign)
    assert after["tokens"] == row_before["tokens"]
    assert after["known_token_subtotals"] == row_before["known_token_subtotals"]
    assert after["native_session_accounting"]["complete_native_total_tokens"] == 247
    assert after["context_delivery_observation"]["status"] == "observed"
    assert after["context_delivery_observation"]["repository_source_mcp"] == row_before["context_delivery_observation"]["repository_source_mcp"]
    assert after["native_result_sha256"] == campaign["value"]["trials"][0]["native_result_sha256"]
    assert before["causal_claim"] is False


def test_same_native_session_in_distinct_retained_trials_withholds_arm_total(campaign):
    second = deepcopy(campaign["value"]["trials"][0])
    second.update(trial="authored-second-trial", reward={"reward": 0})
    for call in second["supervisor"]["provider_invocations"]:
        call["invocation_id"] += "-second"
    campaign["value"]["trials"].append(second)
    campaign["value"]["trial_count"] = 2
    result, row = _collect(campaign)
    rows = [value for value in result["rows"] if value["arm"] == "full"]
    assert [value["tokens"]["total_tokens"] for value in rows] == [247, 247]
    totals = result["arm_totals"]["full"]
    assert totals["trials"] == 2 and totals["observed_official_passes"] == 1
    assert totals["tokens"] == dict.fromkeys(comparison.TOKENS)
    assert totals["observed_efficiency"]["total_tokens_per_official_pass"] is None
    assert result["unsuccessful_attempt_costs"]["full"]["trials"] == 1
    assert row["native_session_accounting"]["complete_native_total_tokens"] == 247


def test_malformed_optional_read_observation_preserves_complete_native_usage(campaign):
    usage = campaign["report"]["provider_invocations"][1]["repository_source_policy"]["usage"]
    usage["read_observation"]["line_range_reads"] = True
    result, row = _collect(campaign)
    assert row["tokens"]["total_tokens"] == 247
    assert row["native_session_accounting"]["complete_supplied_native_usage"] is True
    repository = row["context_delivery_observation"]["repository_source_mcp"]
    assert repository["read_scope_status"] == "partial" and repository["read_scope_counts"] is None
    assert repository["observations"][0]["read_observation_valid"] is False
    assert repository["observations"][0]["prepared_counts"]["successful_reads"] == 1
    assert result["causal_claim"] is False
