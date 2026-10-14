import copy
import hashlib
import json

import pytest
from benchmarks.agent_supervisor.container_coding.benchmark_comparison import (
    BASELINE,
    SUPERVISOR,
    collect,
    markdown,
)


def _receipt(tmp_path, name, schema, trials, arm=None):
    value = {
        "schema": schema,
        "task": "fix-code-vulnerability",
        "model": "pinned-model",
        "reasoning_effort": "high",
        "cli_version": "pinned-cli",
        "original_task_inputs_unchanged": True,
        "trial_count": len(trials),
        "trials": trials,
    }
    if arm:
        value["arm"] = arm
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def _baseline(tmp_path):
    return _receipt(
        tmp_path,
        "baseline.json",
        BASELINE,
        [
            {
                "trial": "baseline",
                "reward": {"reward": 1},
                "seconds": {"agent_execution": 12},
                "raw_usage": {
                    "sessions": [{"session_id": "session", "task_complete_observed": True}],
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 80,
                        "output_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            }
        ],
    )


def _invocation(identity="call"):
    return {
        "invocation_id": identity,
        "provider": "codex_cli",
        "phase": "planning",
        "status": "provider_returned",
        "native_rollout_usage": {
            "schema": "native-codex-cumulative-usage@1",
            "cache_included_in_input": True,
            "task_complete_observed": True,
            "usage": {
                "input_tokens": 20,
                "cached_input_tokens": 15,
                "output_tokens": 2,
                "total_tokens": 22,
            }
        },
        "raw_model_text": "MUST NOT APPEAR",
        "prompt": "MUST NOT APPEAR",
    }


def _grok_invocation(identity="grok-call", complete=None):
    return {
        "invocation_id": identity, "phase": "coding", "provider": "grok_cli", "status": "provider_returned",
        "native_rollout_usage": {
            "schema": "native-grok-final-usage@1", "usage_available": True,
            "usage_complete_observed": complete, "task_complete_observed": True,
            "totals_scope": "native_final_envelope", "cache_included_in_input": False,
            "billing_total_verified": False, "dollar_cost": None,
            "usage": {"input_tokens": 10, "cached_input_tokens": 30, "cache_write_input_tokens": 0,
                      "output_tokens": 2, "total_tokens": 42},
        },
    }


def test_observed_efficiency_includes_failed_attempts_and_deduplicates_receipts(tmp_path):
    trials = []
    for index, (reward, cost) in enumerate(((1, 0.5), (0, 0.25), (1, 0.75))):
        invocation = _invocation(f"call-{index}")
        invocation["native_rollout_usage"]["dollar_cost"] = cost
        if reward == 0:
            invocation.update(status="failed", error_type="ResponseDecodeError")
        trials.append({"trial": f"trial-{index}", "reward": {"reward": reward},
            "supervisor": {"task_completed": True, "provider_invocations": [invocation, invocation]}})
    receipt = _receipt(tmp_path, "efficiency.json", SUPERVISOR, trials, "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt, receipt])
    totals = result["arm_totals"]["full"]
    efficiency = totals["observed_efficiency"]
    assert totals["trials"] == 3 and totals["observed_official_passes"] == 2
    assert totals["tokens"]["total_tokens"] == 66
    assert totals["reported_cost_usd"] == 1.5
    assert efficiency["total_tokens_per_official_pass"] == 33
    assert efficiency["reported_cost_usd_per_official_pass"] == 0.75
    assert efficiency["token_ratio_unavailable_reasons"] == []
    assert efficiency["cost_ratio_unavailable_reasons"] == []
    assert efficiency["numerator_scope"] == "all_retained_attempts_in_aggregate"
    assert efficiency["cache_included_in_input"] is True
    assert efficiency["causal_claim"] is False and efficiency["billing_total_verified"] is False
    assert result["unsuccessful_attempt_costs"]["full"]["tokens"]["total_tokens"] == 22
    assert "| Supervisor: full indexed context | 3 | 2 | 0 | 0 | 33.000 | 0.750 |" in markdown(result)
    assert "do not establish causal token savings or verified billing" in markdown(result)


@pytest.mark.parametrize("gap,token_reason,cost_reason", [
    ("unknown_reward", "unknown_official_reward", "unknown_official_reward"),
    ("zero_passes", "no_official_passes", "no_official_passes"),
    ("partial_reward", "no_official_passes", "no_official_passes"),
    ("incomplete_session", "incomplete_native_token_usage", "incomplete_reported_provider_cost"),
    ("missing_total", "incomplete_native_token_usage", None),
    ("unreceipted_attempt", "incomplete_native_token_usage", "incomplete_reported_provider_cost"),
    ("missing_cost", None, "incomplete_reported_provider_cost"),
])
def test_observed_efficiency_refuses_unknown_denominators_and_incomplete_numerators(
        tmp_path, gap, token_reason, cost_reason):
    trials = []
    for index in range(2):
        invocation = _invocation(f"call-{index}")
        invocation["native_rollout_usage"]["dollar_cost"] = 0.25
        trials.append({"trial": f"trial-{index}", "reward": {"reward": 1},
            "supervisor": {"task_completed": True, "provider_invocations": [invocation]}})
    changed = trials[1]
    native = changed["supervisor"]["provider_invocations"][0]["native_rollout_usage"]
    if gap == "unknown_reward":
        changed.pop("reward")
    elif gap in {"zero_passes", "partial_reward"}:
        for trial in trials:
            trial["reward"]["reward"] = 0.5 if gap == "partial_reward" else 0
    elif gap == "incomplete_session":
        native["task_complete_observed"] = False
    elif gap == "missing_total":
        native["usage"].pop("total_tokens")
    elif gap == "unreceipted_attempt":
        changed["supervisor"]["unreceipted_provider_attempt"] = {"phase": "coding"}
    elif gap == "missing_cost":
        native.pop("dollar_cost")
    receipt = _receipt(tmp_path, "efficiency-gap.json", SUPERVISOR, trials, "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    totals = result["arm_totals"]["full"]
    efficiency = totals["observed_efficiency"]
    assert efficiency["token_ratio_unavailable_reasons"] == ([token_reason] if token_reason else [])
    assert efficiency["cost_ratio_unavailable_reasons"] == ([cost_reason] if cost_reason else [])
    assert efficiency["total_tokens_per_official_pass"] == (None if token_reason else 22)
    assert efficiency["reported_cost_usd_per_official_pass"] == (None if cost_reason else 0.25)
    assert totals["known_token_subtotals"]["total_tokens"] == (22 if gap == "missing_total" else 44)
    assert "Total tokens / official pass" in markdown(result)


def test_setup_abort_does_not_turn_unknown_cost_into_zero_per_pass(tmp_path):
    receipt = _receipt(tmp_path, "setup-abort-efficiency.json", SUPERVISOR, [
        {"trial": "passed", "reward": {"reward": 1},
         "supervisor": {"provider_invocations": [_invocation()]}},
        {"trial": "setup-abort", "exception_type": "SetupError", "supervisor": None},
    ], "no-index")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    totals = result["arm_totals"]["no-index"]
    efficiency = totals["observed_efficiency"]
    assert totals["observed_official_passes"] == 1 and totals["unknown_reward_trials"] == 1
    assert totals["known_token_subtotals"]["total_tokens"] == 22
    assert efficiency["total_tokens_per_official_pass"] is None
    assert efficiency["token_ratio_unavailable_reasons"] == [
        "unknown_official_reward", "incomplete_native_token_usage"]
    assert efficiency["reported_cost_usd_per_official_pass"] is None


def test_observed_efficiency_preserves_mixed_provider_total_and_cache_semantics(tmp_path):
    codex, grok = _invocation(), _grok_invocation(complete=True)
    codex["native_rollout_usage"]["dollar_cost"] = 0
    grok["native_rollout_usage"]["dollar_cost"] = 0
    receipt = _receipt(tmp_path, "mixed-provider-efficiency.json", SUPERVISOR, [{
        "trial": "mixed-provider", "reward": {"reward": 1},
        "supervisor": {"provider_invocations": [codex, grok]},
    }], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    efficiency = result["arm_totals"]["full"]["observed_efficiency"]
    assert row["tokens"] == dict(input_tokens=30, cached_input_tokens=45, output_tokens=4, total_tokens=64)
    assert efficiency["total_tokens_per_official_pass"] == 64  # Native totals, without adding cache twice.
    assert efficiency["reported_cost_usd_per_official_pass"] == 0  # Explicit zero is a known cost.
    assert efficiency["cache_included_in_input"] is None
    assert [call["cache_included_in_input"] for call in row["invocations"]] == [True, False]
    assert [call["token_count_scope"] for call in row["invocations"]] == [
        "native_session_cumulative", "native_final_envelope"]


def test_native_baseline_efficiency_uses_all_failed_session_costs(tmp_path):
    baseline = _baseline(tmp_path)
    receipt = json.loads(baseline.read_text())
    failed = copy.deepcopy(receipt["trials"][0])
    failed.update(trial="failed-native", reward={"reward": 0})
    receipt["trials"].append(failed)
    receipt["trial_count"] = 2
    for trial in receipt["trials"]:
        trial["raw_usage"]["reported_cost_usd"] = 0.125
    baseline.write_text(json.dumps(receipt))
    result = collect(baseline=baseline, supervisors=[])
    efficiency = result["arm_totals"]["native-codex"]["observed_efficiency"]
    assert efficiency["total_tokens_per_official_pass"] == 220
    assert efficiency["reported_cost_usd_per_official_pass"] == 0.25
    assert efficiency["cache_included_in_input"] is True
    assert "| Native Codex harness | 2 | 1 | 0 | 0 | 220.000 | 0.250 |" in markdown(result)


@pytest.mark.parametrize("complete", [True, False, None])
def test_grok_session_completion_does_not_replace_usage_completeness(tmp_path, complete):
    invocation = _grok_invocation(complete=complete)
    receipt = _receipt(tmp_path, "grok.json", SUPERVISOR, [{
        "trial": "grok", "reward": {"reward": 1}, "supervisor": {
            "task_completed": True, "provider_invocations": [invocation, invocation]}}], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    call, = row["invocations"]
    assert row["official_reward"] == 1 and row["native_task_completed"] is True
    assert call["native_task_complete_observed"] is True
    assert call["native_usage_complete_observed"] is complete
    assert row["native_usage_complete_observed"] is complete
    assert row["tokens"]["total_tokens"] == (42 if complete is True else None)
    assert result["arm_totals"]["full"]["tokens"]["total_tokens"] == (42 if complete is True else None)
    assert row["known_token_subtotals"]["total_tokens"] == 42
    assert row["provider_calls"] == 1
    assert call["token_count_scope"] == "native_final_envelope"
    assert row["cache_included_in_input"] is False
    assert result["cached_tokens_are_already_in_input"] is None  # Mixed Codex/Grok semantics.
    assert row["reported_cost_usd"] is None
    efficiency = result["arm_totals"]["full"]["observed_efficiency"]
    assert efficiency["total_tokens_per_official_pass"] == (42 if complete is True else None)
    if complete is not True:
        assert "42 (observed)" in markdown(result)
        assert "≥ 42" not in markdown(result)


@pytest.mark.parametrize("mutation", ["missing_flag", "string_flag", "integer_flag", "wrong_schema",
    "wrong_scope", "wrong_provider", "missing_write", "missing_total", "inconsistent_total", "cache_semantics",
    "usage_unavailable"])
def test_grok_counter_or_completeness_contract_gaps_do_not_create_totals(tmp_path, mutation):
    invocation = _grok_invocation(complete=True)
    native = invocation["native_rollout_usage"]
    if mutation == "missing_flag": native.pop("usage_complete_observed")
    elif mutation == "string_flag": native["usage_complete_observed"] = "true"
    elif mutation == "integer_flag": native["usage_complete_observed"] = 1
    elif mutation == "wrong_schema": native["schema"] = "native-codex-cumulative-usage@1"
    elif mutation == "wrong_scope": native["totals_scope"] = "arbitrary_scope"
    elif mutation == "wrong_provider": invocation["provider"] = "codex_cli"
    elif mutation == "missing_write": native["usage"].pop("cache_write_input_tokens")
    elif mutation == "missing_total": native["usage"].pop("total_tokens")
    elif mutation == "inconsistent_total": native["usage"]["total_tokens"] += 1
    elif mutation == "cache_semantics": native["cache_included_in_input"] = True
    elif mutation == "usage_unavailable": native["usage_available"] = False
    receipt = _receipt(tmp_path, "grok-gap.json", SUPERVISOR, [{
        "trial": "grok-gap", "reward": {"reward": 1}, "supervisor": {
            "task_completed": True, "provider_invocations": [invocation]}}], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    assert all(value is None for value in row["tokens"].values())
    assert row["native_usage_complete_observed"] is None
    assert row["known_token_subtotals"]["input_tokens"] == 10
    assert all(f"≥ {value} (observed)" not in markdown(result) for value in (10, 30, 2, 42, 43))


def test_grok_incomplete_call_prevents_pair_total_and_unreceipted_attempt_remains_unknown(tmp_path):
    planning = _grok_invocation("planning", True)
    planning["phase"] = "planning"
    coding = _grok_invocation("coding", False)
    report = {"provider_invocations": [planning, coding]}
    receipt = _receipt(tmp_path, "mixed-completeness.json", SUPERVISOR, [{
        "trial": "mixed-completeness", "supervisor": report}], "full")
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    assert row["native_usage_complete_observed"] is False
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 84
    coding["native_rollout_usage"]["usage_complete_observed"] = True
    report["unreceipted_provider_attempt"] = {"phase": "coding"}
    receipt = _receipt(tmp_path, "unreceipted-grok.json", SUPERVISOR, [{
        "trial": "unreceipted-grok", "supervisor": report}], "full")
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    assert row["native_usage_complete_observed"] is None and row["tokens"]["total_tokens"] is None


def test_unidentified_legacy_receipt_does_not_inherit_codex_contract(tmp_path):
    invocation = _invocation()
    invocation.pop("provider")
    invocation["native_rollout_usage"].pop("schema")
    receipt = _receipt(tmp_path, "unknown-provider.json", SUPERVISOR, [{
        "trial": "unknown-provider", "supervisor": {"provider_invocations": [invocation]}}], "full")
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    assert row["native_usage_complete_observed"] is None
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 22


def test_stream_progress_survives_comparison_without_becoming_usage_or_completion():
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import GrokStreamParser
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _invocations
    parser = GrokStreamParser()
    parser.feed(b'{"type":"tool_call","toolName":"read_file","args":{"path":"private"}}\n',
                elapsed_seconds=0.1)
    progress = parser.snapshot()
    row = {"invocation_id": "call", "phase": "coding", "status": "failed",
           "native_grok_progress": progress}
    observed, = _invocations({"provider_invocations": [row]})
    assert observed["native_grok_progress"] == progress
    assert observed["native_task_complete_observed"] is None
    assert all(value is None for value in observed["tokens"].values())
    assert "private" not in json.dumps(observed)
    row["native_grok_progress"] = {**progress, "raw_text": "private"}
    assert _invocations({"provider_invocations": [row]})[0]["native_grok_progress"] is None


def test_bounded_stream_usage_survives_comparison_without_replacing_final_accounting():
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import GrokStreamParser
    from ipfs_accelerate_py.cli_runtime.grok_stream_usage import MAX_USAGE_EVENTS
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _invocations
    parser = GrokStreamParser()
    for index in range(MAX_USAGE_EVENTS + 2):
        parser.feed((json.dumps({"type": "usage", "messageId": "private-message",
            "signature": "private-signature", "stopReason": "tool_use",
            "usage": {"input_tokens": 10000 + index, "cache_read_input_tokens": 9000,
                      "output_tokens": 10, "private": "private-content"}}) + "\n").encode(),
            elapsed_seconds=index / 10)
    progress = parser.snapshot()
    invocation = _grok_invocation("streamed", True)
    invocation["native_grok_progress"] = progress
    observed, = _invocations({"provider_invocations": [invocation]})
    events = observed["native_grok_progress"]["usage_observations"]
    assert events["retained_events"] == MAX_USAGE_EVENTS and events["omitted_events"] == 2
    assert events["rows"][0]["usage"]["input_tokens"] == 10000
    assert events["rows"][-1]["event_ordinal"] == MAX_USAGE_EVENTS
    assert events["summation_permitted"] is False
    assert observed["tokens"]["total_tokens"] == 42  # Only the separate final-envelope contract.
    assert "private" not in json.dumps(observed)
    # Existing receipts lack event history; accepting them must not invent it.
    legacy = dict(progress, schema="native-grok-stream-progress@1")
    legacy.pop("usage_observations")
    invocation["native_grok_progress"] = legacy
    historical, = _invocations({"provider_invocations": [invocation]})
    assert historical["native_grok_progress"] == legacy
    assert "usage_observations" not in historical["native_grok_progress"]


def test_symbolic_coding_does_not_invent_provider_calls_or_learned_proof(tmp_path):
    receipt = _receipt(tmp_path, "symbolic.json", SUPERVISOR, [{
        "trial": "symbolic", "reward": {"reward": 1}, "supervisor": {
            "task_completed": True, "provider_invocations": [_invocation()],
            "implementation_route": "doctor_contract_candidate",
            "doctor_dispatch": {"task_cid": "task", "provider_calls": 0,
                "contract_workflow": {"proof": {"status": "proved_local_contract"}}},
            "doctor_invocations": [{"schema": "native-doctor-contract-candidate-materialization@1",
                "task_cid": "task", "status": "candidate_materialized"}],
            "initial_context": {"codebase_autoencoder": {
                "sample_count": 3, "epochs_completed": 24, "training_elapsed_seconds": 1.2,
                "metrics": {"native_kernel_calls": 24, "before_reconstruction_loss": 0.2,
                    "after_reconstruction_loss": 0.1, "holdout_evaluated": False}}},
            "context": {"new_autoencoder_training_steps": 0},
            "post_publication_proof_index": {"status": "observed", "active_receipt_ids": [],
                "invalidated_receipt_ids": ["old-proof"]}}}], "full")
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    assert row["provider_calls"] == 1 and row["tokens"]["total_tokens"] == 22
    assert row["doctor_observation"]["candidate_materializations_observed"] == 1
    assert row["doctor_observation"]["local_contract_proof_status"] == "proved_local_contract"
    assert row["code_learning_observation"]["new_training_steps_after_admission"] == 0
    assert row["code_learning_observation"]["formal_translation_authority"] is False
    assert row["code_learning_observation"]["holdout_evaluated"] is False
    assert row["post_publication_proof_index"]["invalidated_receipts"] == 1
    assert row["post_publication_proof_index"]["active_receipts"] == 0


def test_retains_failed_planning_and_setup_without_cache_double_count(tmp_path):
    failure = _receipt(
        tmp_path,
        "failed.json",
        SUPERVISOR,
        [
            {
                "trial": "failed-plan",
                "reward": {"reward": 0},
                "durations_seconds": {"agent_execution": 10},
                "supervisor": {
                    "task_completed": False,
                    "provider_invocations": [_invocation()],
                    "error": {"type": "ContractError", "message": "MUST NOT APPEAR"},
                },
            },
            {"trial": "setup-abort", "exception_type": "SetupError", "supervisor": None},
        ],
        "no-index",
    )
    result = collect(baseline=_baseline(tmp_path), supervisors=[failure])
    assert result["retained_trial_count"] == 3
    baseline, failed, setup = result["rows"]
    assert baseline["tokens"]["total_tokens"] == 110
    assert baseline["native_task_completed"] is None
    assert failed["tokens"]["total_tokens"] == 22
    assert failed["native_task_completed"] is False
    assert setup["tokens"]["total_tokens"] is None
    assert setup["provider_calls"] is None
    costs = result["unsuccessful_attempt_costs"]["no-index"]
    assert costs["tokens"]["total_tokens"] is None
    assert costs["known_token_subtotals"]["total_tokens"] == 22
    assert costs["unknown_token_trials"]["total_tokens"] == 1
    assert "MUST NOT APPEAR" not in json.dumps(result)
    assert "setup-abort" in markdown(result)
    assert not result["benchmark_advantage_claimed"]


def test_missing_invocation_preserves_known_cost_but_not_complete_total(tmp_path):
    receipt = _receipt(
        tmp_path,
        "partial.json",
        SUPERVISOR,
        [
            {
                "trial": "partial",
                "reward": {"reward": 0},
                "supervisor": {
                    "provider_invocations": [_invocation(), _invocation()],
                    "unreceipted_provider_attempt": {"phase": "coding"},
                },
            }
        ],
        "full",
    )
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    assert row["observed_provider_invocations"] == 1
    assert row["provider_calls"] is None
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 22


@pytest.mark.parametrize("completion", [False, None])
def test_interrupted_or_legacy_usage_is_only_an_observed_subtotal(tmp_path, completion):
    coding = _invocation("interrupted-coding")
    coding.update(phase="coding", status="failed", error_type="TimeoutExpired", timeout_seconds=45)
    coding["native_rollout_usage"]["usage"]["total_tokens"] = 244688
    coding["native_rollout_usage"]["dollar_cost"] = 0.75
    if completion is None:
        coding["native_rollout_usage"].pop("task_complete_observed")
    else:
        coding["native_rollout_usage"]["task_complete_observed"] = completion
    planning = _invocation("planning")
    planning["native_rollout_usage"]["usage"]["total_tokens"] = 21244
    planning["native_rollout_usage"]["dollar_cost"] = 0.25
    receipt = _receipt(tmp_path, "interrupted.json", SUPERVISOR, [{
        "trial": "interrupted", "reward": {"reward": 0},
        "agent_context": {"metadata": {"runtime_archive_sha256": "a" * 64,
            "private_notes": "MUST NOT APPEAR"}},
        "supervisor": {"task_completed": False, "provider_invocations": [planning, coding]},
    }], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    assert row["provider_calls"] == row["observed_provider_invocations"] == 2
    assert row["native_usage_complete_observed"] is completion
    assert all(value is None for value in row["tokens"].values())
    assert row["known_token_subtotals"]["total_tokens"] == 265932
    call = row["invocations"][1]
    assert call["native_task_complete_observed"] is completion
    assert call["tokens"]["total_tokens"] is None
    assert call["known_token_subtotals"]["total_tokens"] == 244688
    assert call["error_type"] == "TimeoutExpired" and call["timeout_seconds"] == 45
    assert row["reported_cost_usd"] is None
    assert row["known_reported_cost_subtotal_usd"] == 1
    costs = result["unsuccessful_attempt_costs"]["full"]
    assert costs["tokens"]["total_tokens"] is None
    assert costs["known_token_subtotals"]["total_tokens"] == 265932
    assert costs["unknown_token_trials"]["total_tokens"] == 1
    assert costs["reported_cost_usd"] is None and costs["known_reported_cost_subtotal_usd"] == 1
    assert result["arm_totals"]["full"]["tokens"]["total_tokens"] is None
    assert row["runtime_archive_sha256"] == "a" * 64
    assert len(result["comparison_source_sha256"]) == 64
    rendered = markdown(result)
    assert "≥ 265932 (observed)" in rendered and "≥ 244688 (observed)" in rendered
    assert "complete totals remain unknown" in rendered
    assert "TimeoutExpired | 45" in rendered and "a" * 64 in rendered
    assert "MUST NOT APPEAR" not in json.dumps(result)


def test_legacy_baseline_without_session_completion_keeps_only_observed_usage(tmp_path):
    baseline = _baseline(tmp_path)
    value = json.loads(baseline.read_text())
    value["trials"][0]["raw_usage"]["sessions"][0].pop("task_complete_observed")
    baseline.write_text(json.dumps(value))
    result = collect(baseline=baseline, supervisors=[])
    row = result["rows"][0]
    assert row["official_reward"] == 1
    assert row["native_usage_complete_observed"] is None
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 110
    assert result["arm_totals"]["native-codex"]["unknown_token_trials"]["total_tokens"] == 1
    assert "≥ 110 (observed)" in markdown(result)


def test_completed_native_session_can_retain_failed_postprocessing_cost(tmp_path):
    invocation = {**_invocation(), "status": "failed", "error_type": "ResponseDecodeError"}
    receipt = _receipt(tmp_path, "decode-error.json", SUPERVISOR, [{"trial": "decode-error",
        "reward": {"reward": 0}, "supervisor": {"provider_invocations": [invocation]}}], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    assert row["native_usage_complete_observed"] is True
    assert row["tokens"]["total_tokens"] == row["known_token_subtotals"]["total_tokens"] == 22
    assert row["invocations"][0]["error_type"] == "ResponseDecodeError"
    assert result["unsuccessful_attempt_costs"]["full"]["tokens"]["total_tokens"] == 22


def test_conflicting_same_invocation_is_rejected(tmp_path):
    changed = _invocation()
    changed["status"] = "failed"
    receipt = _receipt(
        tmp_path,
        "conflict.json",
        SUPERVISOR,
        [
            {
                "trial": "conflict",
                "supervisor": {"provider_invocations": [_invocation(), changed]},
            }
        ],
        "full",
    )
    with pytest.raises(ValueError, match="conflicting provider"):
        collect(baseline=_baseline(tmp_path), supervisors=[receipt])


def test_duplicate_retained_trial_counted_once_and_empty_receipt_preserved(tmp_path):
    receipt = _receipt(tmp_path, "empty.json", SUPERVISOR, [], "full")
    duplicate = _receipt(tmp_path, "one.json", SUPERVISOR, [{"trial": "one"}], "no-index")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt, duplicate, duplicate])
    assert result["retained_trial_count"] == 2
    assert len(result["receipts_without_trials"]) == 1
    assert len(result["rows"][1]["source_receipts"]) == 2
    assert result["rows"][1]["tokens"]["total_tokens"] is None


def test_context_preparation_is_not_worker_consumption_or_repair_evidence(tmp_path):
    receipt = _receipt(tmp_path, "indexed.json", SUPERVISOR, [{
        "trial": "indexed",
        "supervisor": {
            "phases": {"context_seconds": 101.0},
            "context": {"learned_embeddings": True, "native_fact_rows_replayed": 426,
                        "full_capsules": 531, "worker_capsules": 2,
                        "worker_semantic_bytes": 28388,
                        "doctor_repair": {"status": "abstained", "raw": "MUST NOT APPEAR"}},
        },
    }], "full")
    comparison = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = comparison["rows"][1]
    assert row["context_seconds"] == 101
    assert row["indexed_context_observation"]["doctor_repair_status"] == "abstained"
    assert row["indexed_context_observation"]["context_preparation_is_dispatch_evidence"] is False
    assert row["official_reward"] is None
    assert row["native_task_completed"] is None
    assert "MUST NOT APPEAR" not in json.dumps(comparison)
    assert "do not by themselves prove worker consumption" in markdown(comparison)


def test_full_lifecycle_costs_and_actual_input_observations_are_separate(tmp_path):
    invocation = {**_invocation("coding-call"), "phase": "coding",
        "native_prompt_bytes": 12000, "router_prompt_bytes": 9400, "model_prompt_bytes": 9900,
        "workspace_advisory_bytes": 500, "native_prompt_sha256": "a" * 64,
        "router_prompt_sha256": "b" * 64, "model_prompt_sha256": "c" * 64,
        "semantic_translation": {"schema": "supervisor-semantic-router-encoding@1",
            "translation_cid": "cid:translation", "native_prompt_bytes": 12000,
            "provider_prompt_bytes": 9000, "identifier_mappings": 12, "identifier_occurrences": 24,
            "freshness_checked": True, "raw_prompt": "MUST NOT APPEAR"},
        "doctor_residual_context": {"schema": "doctor-residual-router-context@1",
            "context_cid": "cid:residual", "native_capsule_id": "cid:capsule",
            "advisory_sha256": "d" * 64, "advisory_bytes": 400, "extra_provider_calls": 0,
            "source_freshness_verified": True, "candidate_only": True,
            "derived_runtime_admitted": False, "raw_advisory": "MUST NOT APPEAR"}}
    phases = {"prepare_seconds": 1, "initial_context_seconds": 12, "planning_seconds": 8,
        "context_seconds": 3, "doctor_seconds": 2, "post_publication_refresh_seconds": 4}
    receipt = _receipt(tmp_path, "full.json", SUPERVISOR, [{"trial": "full", "reward": {"reward": 0},
        "durations_seconds": {"agent_execution": 40}, "supervisor": {
            "seconds": 38, "phases": phases, "task_completed": True,
            "initial_context": {"seconds": 11, "nonoverlapping_seconds": {"vectors": 10},
                "indexed_symbols": 10, "full_capsules": 11, "world_task_count": 0,
                "provider_calls": 0, "canonical_tasks_created": False, "learned_embeddings": True},
            "planning": {"elapsed_seconds": 7, "model_request_sha256": "e" * 64,
                "initial_indexed_context": {"supplied_to_router": True, "summary_sha256": "f" * 64}},
            "context": {"learned_embeddings": True, "initial_indexes_reused": True,
                "new_embedding_calls": 0, "timings": {"nested_helper_seconds": {"cold": 9999}}},
            "implementation_route": "model_router", "doctor_dispatch": {"status": "residual",
                "task_cid": "cid:task", "provider_calls": 0, "residual_successors": 1,
                "residual_work_proposals": 0, "residual_context": {"artifact": "MUST NOT APPEAR"}},
            "provider_invocations": [invocation],
            "context_input_audit": {"status": "verified_all_observed_model_inputs",
                "any_native_input_verified": True, "any_model_input_verified": True,
                "all_observed_coding_inputs_verified": True, "coding_receipts_seen": 1,
                "usable_coding_receipts": 1, "raw_prompt": "MUST NOT APPEAR",
                "invocations": [{"invocation_id": "coding-call", "native_input_verified": True,
                    "model_input_verified": True, "status": "verified_exact_model_input"}]},
            "post_publication_context": {"status": "refreshed", "refresh_seconds": 4,
                "budget_seconds": 20, "embedding_calls": 2, "embedding_accounting": {
                    "all_refreshes_receipted": True,
                    "totals": {"local_embedding_calls": 2, "local_embedding_texts": 6,
                        "remote_embedding_calls": 0, "text_generation_calls": 0},
                    "known_subtotals": {"local_embedding_calls": 2, "local_embedding_texts": 6,
                        "remote_embedding_calls": 0, "text_generation_calls": 0}}},
        }}], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    assert row["supervisor_phases_seconds"] == phases
    assert row["supervisor_seconds"] == 38 and row["durations_seconds"]["agent_execution"] == 40
    assert row["initial_indexed_context_observation"]["supplied_to_planning_router"] is True
    assert row["indexed_context_observation"]["initial_indexes_reused"] is True
    assert row["doctor_observation"]["status"] == "residual"
    assert row["doctor_observation"]["candidate_materializations_observed"] is None
    assert row["context_input_audit"]["all_observed_coding_inputs_verified"] is True
    wire = row["invocations"][0]["input_observation"]
    assert wire["native_prompt_bytes"] == 12000 and wire["router_prompt_bytes"] == 9400
    assert wire["semantic_translation"]["provider_prompt_bytes"] == 9000
    assert wire["doctor_residual_context"]["advisory_bytes"] == 400
    assert wire["measurements_are_token_savings"] is False
    refresh = row["post_publication_context"]
    assert refresh["embedding_accounting"]["totals"]["local_embedding_calls"] == 2
    assert refresh["embedding_accounting"]["totals"]["text_generation_calls"] == 0
    assert row["official_reward"] == 0 and row["native_task_completed"] is True
    assert row["tokens"]["total_tokens"] == 22  # neither byte counts nor embeddings are text tokens
    assert not result["benchmark_advantage_claimed"]
    serialized = json.dumps(result)
    assert "MUST NOT APPEAR" not in serialized and "9999" not in serialized
    rendered = markdown(result)
    assert "Cold initial context s" in rendered and "| full | 1 | 12 | 8 | 3 | 2 | 4 |" in rendered
    assert "not token savings" in rendered


def test_failed_cold_stage_is_retained_without_preparation_or_dispatch_claim(tmp_path):
    receipt = _receipt(tmp_path, "cold-failure.json", SUPERVISOR, [{"trial": "cold-failure",
        "supervisor": {"phases": {"prepare_seconds": 1, "initial_context_seconds": 55},
            "error": {"type": "TimeoutError"}}}], "full")
    comparison = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = comparison["rows"][1]
    assert row["supervisor_phases_seconds"]["initial_context_seconds"] == 55
    assert row["supervisor_phases_seconds"]["planning_seconds"] is None
    assert row["initial_indexed_context_observation"]["prepared"] is False
    assert row["initial_indexed_context_observation"]["supplied_to_planning_router"] is None
    assert row["context_input_audit"] is None and row["post_publication_context"] is None
    assert row["tokens"]["total_tokens"] is None
    assert "| cold-failure | 1 | 55 | unknown | unknown | unknown | unknown |" in markdown(comparison)


def test_declared_output_bytes_survive_comparison_without_correctness_or_token_claims(tmp_path):
    observation = {"schema": "terminal-declared-output-observation@1", "status": "observed",
        "task_cid_sha256": "a" * 64, "attempt_id_sha256": "b" * 64, "invocation_id_sha256": "c" * 64,
        "baseline_commit": "d" * 40, "private_prompt": "MUST NOT APPEAR", "task_correctness_established": True,
        "outputs": [{"path": "eigen.py", "effect": "modify", "baseline_sha256": "e" * 64,
            "candidate": {"status": "observed", "exists": True, "size_bytes": 100, "sha256": "f" * 64,
                "equals_admitted_baseline": False, "raw_source": "MUST NOT APPEAR"},
            "canonical": {"status": "observed", "exists": True, "size_bytes": 80, "sha256": "e" * 64,
                "equals_admitted_baseline": True}}]}
    receipt = _receipt(tmp_path, "output.json", SUPERVISOR, [{"trial": "timeout", "reward": {"reward": 0},
        "supervisor": {"task_completed": False, "declared_output_observation": observation}}], "no-index")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    retained = row["declared_output_observation"]
    assert retained["status"] == "observed" and retained["invocation_id_sha256"] == "c" * 64
    assert retained["outputs"][0]["candidate"]["equals_admitted_baseline"] is False
    assert retained["task_correctness_established"] is retained["patch_application_established"] is False
    assert row["official_reward"] == 0 and row["native_task_completed"] is False
    assert row["tokens"]["total_tokens"] is None and "MUST NOT APPEAR" not in json.dumps(result)


@pytest.mark.parametrize("malformed", ["missing", "extra_rows", "unsafe_path", "bad_digest", "status_list"])
def test_absent_or_malformed_byte_observation_remains_unknown(malformed):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _declared_output_observation
    raw = {"schema": "terminal-declared-output-observation@1", "status": "observed", "outputs": [
        {"path": "eigen.py", "effect": "modify", "candidate": {"status": "observed", "exists": True,
            "size_bytes": 9, "sha256": "bad"}, "canonical": {"status": "unavailable"}}]}
    if malformed == "extra_rows": raw["outputs"] *= 33
    elif malformed == "unsafe_path": raw["outputs"][0]["path"] = "../private"
    elif malformed == "status_list": raw["status"] = []
    result = _declared_output_observation({} if malformed == "missing" else {"declared_output_observation": raw})
    assert result["status"] in {"partial", "unavailable"}
    assert result["task_correctness_established"] is False
    assert "private" not in json.dumps(result) and "bad" not in json.dumps(result)


@pytest.mark.parametrize("stage", [
    "admitted_scope", "settlement_and_cleanup", "native_directory", "attempt_directory",
    "attempt_binding", "owner_event_log", "allocation_event", "allocation_root",
    "invocation_join", "declared_output_files",
])
def test_closed_output_failure_stage_survives_comparison(tmp_path, stage):
    observation = {"schema": "terminal-declared-output-observation@1", "status": "unavailable",
        "reason": "binding_or_file_evidence_unavailable", "outputs": [], "evidence_stage": stage,
        "exception": "PRIVATE-EXCEPTION-CANARY", "completion_authority": True}
    receipt = _receipt(tmp_path, "output-stage.json", SUPERVISOR, [{"trial": "output-stage",
        "supervisor": {"task_completed": True, "declared_output_observation": observation}}], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    retained = row["declared_output_observation"]
    assert retained["status"] == "unavailable"
    assert retained["evidence_stage"] == stage
    assert retained["outputs"] == [] and retained["completion_authority"] is False
    assert row["native_task_completed"] is True and row["official_reward"] is None
    assert row["tokens"]["total_tokens"] is None
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("value", [None, "", "PRIVATE-EXCEPTION-CANARY", "owner_event_log\nPRIVATE",
    " owner_event_log", "owner_event_log ", [], {}, True, 1])
def test_unknown_output_stage_is_omitted_without_losing_byte_observation(value):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _declared_output_observation
    raw = {"schema": "terminal-declared-output-observation@1", "status": "unavailable",
        "reason": "binding_or_file_evidence_unavailable", "outputs": [], "evidence_stage": value}
    result = _declared_output_observation({"declared_output_observation": raw})
    assert result["status"] == "unavailable" and result["outputs"] == []
    assert "evidence_stage" not in result
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("change", [
    {"status": "observed"}, {"status": "partial"}, {"reason": "PRIVATE-EXCEPTION-CANARY"},
    {"schema": "unknown"}, {"outputs": None},
])
def test_stage_is_not_retained_from_inconsistent_or_malformed_observation(change):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _declared_output_observation
    raw = {"schema": "terminal-declared-output-observation@1", "status": "unavailable",
        "reason": "binding_or_file_evidence_unavailable", "outputs": [], "evidence_stage": "owner_event_log", **change}
    result = _declared_output_observation({"declared_output_observation": raw})
    assert "evidence_stage" not in result
    assert "PRIVATE" not in json.dumps(result)


def test_historical_output_observation_does_not_gain_a_failure_stage():
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _declared_output_observation
    raw = {"schema": "terminal-declared-output-observation@1", "status": "unavailable",
        "reason": "binding_or_file_evidence_unavailable", "outputs": []}
    result = _declared_output_observation({"declared_output_observation": raw})
    assert result["status"] == "unavailable"
    assert result["reason"] == "reported_declared_byte_observation"
    assert "evidence_stage" not in result


def test_doctor_selection_is_distinct_from_matching_worker_materialization(tmp_path):
    receipt = _receipt(tmp_path, "doctor.json", SUPERVISOR, [{"trial": "doctor",
        "supervisor": {"implementation_route": "doctor_candidate",
            "doctor_dispatch": {"status": "candidate_ready", "task_cid": "cid:current", "provider_calls": 0},
            "doctor_invocations": [
                {"schema": "native-doctor-candidate-materialization@1", "task_cid": "cid:foreign",
                    "status": "candidate_materialized"},
                {"schema": "native-doctor-candidate-materialization@1", "task_cid": "cid:current",
                    "status": "candidate_materialized", "source": "MUST NOT APPEAR"}],
        }}], "full")
    comparison = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = comparison["rows"][1]
    assert row["doctor_observation"]["candidate_materializations_observed"] == 1
    assert row["doctor_observation"]["observed_worker_receipts"] == 1
    assert row["doctor_observation"]["selection_is_publication_or_completion_evidence"] is False
    assert row["native_task_completed"] is None and row["official_reward"] is None
    assert "MUST NOT APPEAR" not in json.dumps(comparison)


@pytest.mark.parametrize("refresh", [
    {"status": "deferred", "reason": "post_stop_refresh_budget_unavailable", "embedding_calls": None},
    {"status": "unavailable", "error_type": "_PostStopRefreshExpired", "embedding_calls": None},
    {"status": "incomplete", "embedding_calls": 0, "embedding_accounting": {
        "all_refreshes_receipted": False, "totals": {"local_embedding_calls": 0},
        "known_subtotals": {"local_embedding_calls": 1, "local_embedding_texts": 3}}},
])
def test_refresh_unknown_totals_preserve_observed_failed_costs(tmp_path, refresh):
    receipt = _receipt(tmp_path, "refresh.json", SUPERVISOR, [{"trial": "refresh",
        "supervisor": {"task_completed": True, "post_publication_context": refresh}}], "full")
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    observed = row["post_publication_context"]
    assert observed["embedding_calls"] is None
    assert all(value is None for value in observed["embedding_accounting"]["totals"].values())
    assert observed["embedding_accounting"]["known_subtotals"]["local_embedding_calls"] == (
        1 if refresh["status"] == "incomplete" else None)
    assert row["native_task_completed"] is True
    assert observed["task_completion_is_context_refresh_evidence"] is False


def test_reported_wire_measurements_do_not_imply_successful_input_audit(tmp_path):
    invocation = {**_invocation(), "native_prompt_bytes": 12000, "router_prompt_bytes": 9000,
        "semantic_translation": {"identifier_mappings": 10, "freshness_checked": True}}
    receipt = _receipt(tmp_path, "unaudited.json", SUPERVISOR, [{"trial": "unaudited",
        "supervisor": {"provider_invocations": [invocation], "context_input_audit": {
            "status": "verified_native_input_only", "any_native_input_verified": True,
            "any_model_input_verified": False, "all_observed_coding_inputs_verified": False}}}], "full")
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    assert row["invocations"][0]["input_observation"]["semantic_translation"]["identifier_mappings"] == 10
    assert row["context_input_audit"]["any_model_input_verified"] is False
    assert row["context_input_audit"]["all_observed_coding_inputs_verified"] is False


def _metadata_selection(schema="supervisor-semantic-metadata-selection@2", qualification="provider_tokenizer"):
    value = {"schema": schema, "view_receipt_sha256": "a" * 64,
        "selected_mode": "common-bindings@1", "fallback_reason": None,
        "token_proxy": "utf8-bytes-ceil-div4@1", "candidate_only": True,
        "native_complete_sha256": "b" * 64, "native_complete_bytes": 1000,
        "native_complete_proxy_tokens": 250, "candidate_complete_sha256": "c" * 64,
        "candidate_complete_bytes": 800, "candidate_complete_proxy_tokens": 200,
        "selected_complete_sha256": "c" * 64, "selected_complete_bytes": 800,
        "selected_complete_proxy_tokens": 200}
    if schema.endswith("@2"):
        identity = {"schema": "supervisor-semantic-metadata-token-counter@1",
            "provider": "codex_cli", "model": "gpt-6.1-sol", "tokenizer": "declared-tokenizer",
            "tokenizer_revision": "revision-1", "qualification": qualification,
            "measurement_scope": "complete-rendered-prompt-text@1",
            "identity_caller_supplied": True, "provider_usage_verified": False}
        value.update(selection_basis="token_counter", token_counter_status="counted",
            token_counter_identity=identity,
            token_counter_identity_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
            native_complete_tokens=120, candidate_complete_tokens=90, selected_complete_tokens=90)
    return value


def test_absent_metadata_preserves_existing_input_observation_shape():
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _input_observation
    observed = _input_observation({})
    assert set(observed) == {"native_prompt_bytes", "router_prompt_bytes", "model_prompt_bytes",
        "workspace_advisory_bytes", "native_prompt_sha256", "router_prompt_sha256", "model_prompt_sha256",
        "workspace_advisory_sha256", "semantic_translation", "doctor_residual_context",
        "measurements_are_token_savings", "input_reconstruction_verified_by_receipt_presence"}
    assert "semantic_metadata_view" not in observed


@pytest.mark.parametrize("schema, qualification", [
    ("supervisor-semantic-metadata-selection@1", "provider_tokenizer"),
    ("supervisor-semantic-metadata-selection@2", "provider_tokenizer"),
    ("supervisor-semantic-metadata-selection@2", "estimate"),
])
def test_metadata_measurements_survive_comparison_without_charging_native_usage(tmp_path, schema, qualification):
    before = _invocation()
    metadata = _metadata_selection(schema, qualification)
    metadata.update(prompt="MUST NOT APPEAR", raw_source="MUST NOT APPEAR",
        counter_error="MUST NOT APPEAR", provider_usage_reported=True,
        complete_run_token_savings_claimed=True)
    if schema.endswith("@2"):
        metadata["token_counter_identity"]["source_body"] = "MUST NOT APPEAR"
    after = {**before, "semantic_metadata_view": metadata}
    reports = []
    for label, invocation in (("without", before), ("with", after)):
        path = _receipt(tmp_path, label + ".json", SUPERVISOR, [{"trial": label,
            "supervisor": {"provider_invocations": [invocation, copy.deepcopy(invocation)]}}], "full")
        reports.append(collect(baseline=_baseline(tmp_path), supervisors=[path]))
    old, new = (report["rows"][1] for report in reports)
    observed = new["invocations"][0]["input_observation"]["semantic_metadata_view"]
    assert observed["status"] == "reported"
    assert observed["selected_complete_bytes"] == 800
    assert observed["selected_complete_proxy_tokens"] == 200
    assert observed["provider_usage_reported"] is False
    assert observed["tokenizer_compatibility_verified"] is False
    assert observed["complete_run_token_savings_claimed"] is False
    if schema.endswith("@1"):
        assert observed["selection_basis"] == "byte_proxy"
        assert "selected_complete_tokens" not in observed
    else:
        assert observed["selection_basis"] == "token_counter"
        assert observed["selected_complete_tokens"] == 90
        assert observed["token_counter_identity"]["qualification"] == qualification
    for field in ("tokens", "known_token_subtotals", "provider_calls", "reported_cost_usd",
                  "native_usage_complete_observed", "cache_included_in_input"):
        assert new[field] == old[field]
    assert new["tokens"]["input_tokens"] == 20  # Includes its 15 cached tokens already.
    assert new["tokens"]["total_tokens"] == 22 and new["provider_calls"] == 1
    assert "MUST NOT APPEAR" not in json.dumps(reports[1])


@pytest.mark.parametrize("mutation", ["unknown_schema", "list_schema", "list_status", "unknown_qualification",
    "list_qualification", "unverified_scope", "identity_digest", "boolean_count", "negative_count",
    "overflow_count", "selected_count", "proxy_count", "caller_flag", "unsafe_identity"])
def test_malformed_metadata_measurements_remain_unavailable(mutation):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _input_observation
    metadata = _metadata_selection()
    identity = metadata["token_counter_identity"]
    if mutation == "unknown_schema": metadata["schema"] = "unknown-MUST NOT APPEAR"
    elif mutation == "list_schema": metadata["schema"] = ["MUST NOT APPEAR"]
    elif mutation == "list_status": metadata["token_counter_status"] = ["MUST NOT APPEAR"]
    elif mutation == "unknown_qualification": identity["qualification"] = "MUST NOT APPEAR"
    elif mutation == "list_qualification": identity["qualification"] = ["MUST NOT APPEAR"]
    elif mutation == "unverified_scope": identity["measurement_scope"] = "MUST NOT APPEAR"
    elif mutation == "identity_digest": metadata["token_counter_identity_sha256"] = "MUST NOT APPEAR"
    elif mutation == "boolean_count": metadata["native_complete_tokens"] = True
    elif mutation == "negative_count": metadata["native_complete_tokens"] = -1
    elif mutation == "overflow_count": metadata["native_complete_tokens"] = 2**63
    elif mutation == "selected_count": metadata["selected_complete_tokens"] = 91
    elif mutation == "proxy_count": metadata["candidate_complete_proxy_tokens"] += 1
    elif mutation == "caller_flag": identity["identity_caller_supplied"] = 1
    elif mutation == "unsafe_identity": identity["tokenizer"] = "MUST NOT APPEAR\nsource()"
    observed = _input_observation({"semantic_metadata_view": metadata})["semantic_metadata_view"]
    assert observed == {"status": "unavailable", "provider_usage_reported": False,
        "tokenizer_compatibility_verified": False, "complete_run_token_savings_claimed": False}
    assert "MUST NOT APPEAR" not in json.dumps(observed)


def test_counter_failure_retains_fallback_without_exception_text_or_invented_counts():
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _input_observation
    metadata = _metadata_selection()
    metadata.update(token_counter_status="failed", selected_mode="legacy",
        fallback_reason="complete_input_token_counter_failed", native_complete_tokens=None,
        candidate_complete_tokens=None, selected_complete_tokens=None, counter_error="MUST NOT APPEAR")
    for field in ("sha256", "bytes", "proxy_tokens"):
        metadata["selected_complete_" + field] = metadata["native_complete_" + field]
    observed = _input_observation({"semantic_metadata_view": metadata})["semantic_metadata_view"]
    assert observed["status"] == "reported" and observed["token_counter_status"] == "failed"
    assert observed["selected_mode"] == "legacy"
    assert observed["selected_complete_tokens"] is None
    assert "MUST NOT APPEAR" not in json.dumps(observed)


@pytest.mark.parametrize("schema", ["supervisor-semantic-metadata-selection@1", "supervisor-semantic-metadata-selection@2"])
@pytest.mark.parametrize("fallback", [False, True])
def test_instruction_metadata_mode_retains_requested_treatment_including_fallback(schema, fallback):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _input_observation
    metadata = _metadata_selection(schema)
    metadata.update(requested_mode="common-bindings-instruction@1",
                    selected_mode="legacy" if fallback else "common-bindings-instruction@1")
    if fallback:
        metadata["fallback_reason"] = ("complete_input_not_smaller_under_bytes_and_proxy"
            if schema.endswith("@1") else "complete_input_not_smaller_under_bytes_and_token_counter")
        for field in ("sha256", "bytes", "proxy_tokens", "tokens"):
            key = "native_complete_" + field
            if key in metadata:
                metadata["selected_complete_" + field] = metadata[key]
    observed = _input_observation({"semantic_metadata_view": metadata})["semantic_metadata_view"]
    assert observed["status"] == "reported"
    assert observed["requested_mode"] == "common-bindings-instruction@1"
    assert observed["selected_mode"] == metadata["selected_mode"]
    assert observed["selected_complete_sha256"] == metadata["selected_complete_sha256"]
    assert observed["complete_run_token_savings_claimed"] is False


@pytest.mark.parametrize("requested", [None, [], "foreign-MUST NOT APPEAR", "common-bindings@1", "legacy"])
def test_instruction_metadata_mode_refuses_missing_foreign_or_conflicting_requested_mode(requested):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _input_observation
    metadata = _metadata_selection()
    metadata["selected_mode"] = "common-bindings-instruction@1"
    if requested is not None:
        metadata["requested_mode"] = requested
    observed = _input_observation({"semantic_metadata_view": metadata})["semantic_metadata_view"]
    assert observed["status"] == "unavailable"
    assert "requested_mode" not in observed and "MUST NOT APPEAR" not in json.dumps(observed)
