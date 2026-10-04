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
        "phase": "planning",
        "status": "provider_returned",
        "native_rollout_usage": {
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
