"""Authored accounting controls; no provider, tokenizer compatibility or success claims."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.context_projection_comparison import (
    MAX_TASKS,
    compare_context_projection,
    measure_context_candidates,
)
from ipfs_accelerate_py.agent_supervisor.runtime.cumulative_usage_reconciliation import (
    COMPLETION_BASIS,
)
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    PromptTokenCounter,
    render_model_prompt,
)


def task(name="task-1"):
    return {"task_id": name, "provider": "codex_cli", "model": "fixture-model",
            "tree_id": "tree:current-with-dirty-overlay", "acceptance_binding": "acceptance:v1"}


def receipt(name="base", *, inputs=100, outputs=20, count=1, complete=True, thread=None):
    thread = thread or "native-thread-" + name + "-123456789"
    return {
        "schema": "router-implementation-invocation@1", "invocation_id": name,
        "provider": "codex_cli", "model": "fixture-model", "workspace": "/isolated/task",
        "status": "provider_returned", "phase": "coding", "router_calls": 1,
        "usage": {"thread_id": thread, "model_id": "fixture-model"},
        "native_rollout_usage": {
            "schema": "native-codex-cumulative-usage@1", "thread_id": thread,
            "rollout_sha256": sha256(f"{thread}:{count}:{inputs}:{outputs}".encode()).hexdigest(),
            "usage": {"input_tokens": inputs, "output_tokens": outputs,
                      "cached_input_tokens": inputs // 2, "reasoning_output_tokens": outputs // 2,
                      "total_tokens": inputs + outputs},
            "usage_available": True, "observed_token_count_records": count,
            "task_complete_observed": complete, "usage_lifecycle_complete": complete,
            "completion_basis": COMPLETION_BASIS, "lifecycle_malformed_records": 0,
            "billing_total_verified": False, "cache_included_in_input": True,
        }, "completion_authority": False,
    }


def observed(binding=None, *, outcome="accepted", rows=None):
    binding = task() if binding is None else binding
    rows = [("coding", receipt())] if rows is None else rows
    return {**binding, "accepted": outcome == "accepted", "outcome": outcome,
            "attempts": [{"attempt_id": f"{binding['task_id']}-attempt-{index}", "kind": kind, "receipt": row}
                         for index, (kind, row) in enumerate(rows)]}


def paired(*, left=None, right=None, assigned=None, **kwargs):
    return compare_context_projection(
        assigned_tasks=[task()] if assigned is None else assigned,
        baseline=[observed()] if left is None else left,
        candidate=[observed(rows=[("coding", receipt("candidate", inputs=60))])] if right is None else right,
        **kwargs)


def counter(callback=None, **bindings):
    return PromptTokenCounter(**{
        "provider": "codex_cli", "model": "fixture-model", "tokenizer_id": "authored:utf8",
        "tokenizer_revision": "fixture-v1", "special_token_policy": "fixture-only", **bindings,
    }, count_tokens=callback or (lambda text: len(text.encode())))


def measured(**kwargs):
    return measure_context_candidates(**{
        "native_prompt": "N" * 1000, "projected_prompt": "P" * 800,
        "alias_prompt": "A" * 50, "suffix": "\nLiteral acceptance and /app/out.txt",
        "purpose": "coding", "workspace": Path("/isolated/task"), "provider": "codex_cli",
        "model": "fixture-model", "token_counter": counter(), "measurement_basis": "authored_fixture",
        **kwargs,
    })


def test_measures_all_complete_wires_without_storing_text_or_claiming_provider_tokens():
    seen = []
    result = measured(token_counter=counter(lambda text: seen.append(text) or len(text.encode())))
    assert len(seen) == 3
    assert all(text.count("Literal acceptance") == 1 for text in seen)
    assert result["candidates"]["native"]["input_tokens"] == len(seen[0].encode())
    assert result["candidates"]["projected"]["input_tokens"] == len(seen[1].encode())
    assert result["candidates"]["alias"]["input_tokens"] == len(seen[2].encode())
    assert result["measurement_basis"] == "authored_fixture"
    assert result["provider_compatibility_verified"] is False
    assert result["whole_session_token_savings"] is None
    assert "N" * 1000 not in str(result)


def test_alias_transport_overhead_can_reverse_reduction_and_keeps_projected_native():
    result = measured(projected_prompt="P" * 100, alias_prompt="A" * 99)
    assert result["projected_alias_selection"] == "native"
    native, _ = render_model_prompt(prompt="P" * 100 + "\nLiteral acceptance and /app/out.txt",
                                   purpose="coding", workspace=Path("/isolated/task"))
    assert result["candidates"]["projected"]["input_tokens"] == len(native.encode())
    assert result["candidates"]["alias"]["input_tokens"] > len(native.encode())


def test_no_counter_retains_native_and_unknown_content_counts():
    result = measured(token_counter=None)
    assert all(row["input_tokens"] is None for row in result["candidates"].values())
    assert result["projected_alias_selection"] == "native"
    assert result["comparisons"]["projected"]["rendered_user_content_token_delta"] is None


def test_failed_counter_never_turns_partial_counts_into_savings():
    result = measured(token_counter=counter(lambda _: False))
    assert result["measurement_status"] == {"native": "failed", "projected_and_alias": "failed"}
    assert result["comparisons"]["alias"]["rendered_user_content_token_delta"] is None


@pytest.mark.parametrize("binding", [{"provider": "other"}, {"model": "other"}])
def test_foreign_counter_binding_refuses(binding):
    with pytest.raises(ValueError, match="pinned provider/model"):
        measured(token_counter=counter(**binding))


def test_strict_counter_budget_and_complete_candidate_ceiling_refuse():
    with pytest.raises(ValueError, match="working pinned"):
        measured(token_counter=None, max_input_tokens=5000)
    with pytest.raises(ValueError, match="byte bound"):
        measured(max_prompt_bytes=10)


def test_supplied_totals_and_exact_denominators_require_explicit_population_declaration():
    unknown = paired()
    assert unknown["baseline"]["usage"]["provider_total_tokens"] == 120
    assert unknown["candidate"]["usage"]["provider_total_tokens"] == 80
    assert unknown["paired_metrics"]["provider_token_delta"] is None
    result = paired(population_complete=True)
    assert result["paired_metrics"]["provider_token_delta"] == 40
    assert result["paired_metrics"]["candidate_provider_tokens_per_assigned_task"] == 80
    assert result["paired_metrics"]["baseline_provider_tokens_per_accepted_task"] == 120
    assert result["campaign_usage_complete"] is None
    assert result["campaign_population_verified"] is False
    assert result["task_acceptance_verified"] is False


def test_initial_local_reduction_is_erased_by_fresh_retry_and_expansion_sessions():
    local = measured()
    assert local["comparisons"]["projected"]["rendered_user_content_token_delta"] > 0
    rows = [("coding", receipt("compact", inputs=40)),
            ("retry", receipt("retry", inputs=60)),
            ("expansion", receipt("expand", inputs=20))]
    result = paired(right=[observed(rows=rows)], population_complete=True)
    assert result["candidate"]["usage"]["provider_total_tokens"] == 180
    assert result["paired_metrics"]["provider_token_delta"] == -60
    assert result["paired_metrics"]["positive_provider_token_reduction_observed"] is False
    assert [row["kind"] for row in result["candidate"]["tasks"][0]["attempts"]] == ["coding", "retry", "expansion"]


def test_authored_receipts_never_claim_actual_provider_or_campaign_savings():
    result = paired(population_complete=True, usage_evidence_basis="authored_fixture")
    assert result["candidate"]["usage"]["provider_total_tokens"] == 80
    assert result["paired_metrics"]["provider_token_delta"] is None
    assert result["paired_metrics"]["positive_provider_token_reduction_observed"] is False
    assert result["campaign_usage_complete"] is None
    assert result["provider_usage_authentication_verified"] is False


def test_failed_task_and_all_its_sessions_stay_in_cost_denominator():
    first, second = task(), task("task-failed")
    baseline = [observed(first), observed(second, outcome="failed", rows=[("coding", receipt("failed-base", inputs=80))])]
    candidate = [observed(first, rows=[("coding", receipt("ok", inputs=30))]),
                 observed(second, outcome="failed", rows=[("retry", receipt("failed-candidate", inputs=200))])]
    result = paired(assigned=[first, second], left=baseline, right=candidate, population_complete=True)
    assert result["assigned_task_count"] == 2
    assert result["baseline"]["accepted_task_count"] == 1
    assert result["baseline"]["usage"]["provider_total_tokens"] == 220
    assert result["candidate"]["usage"]["provider_total_tokens"] == 270
    assert result["paired_metrics"]["candidate_provider_tokens_per_assigned_task"] == 135
    assert result["paired_metrics"]["candidate_provider_tokens_per_accepted_task"] == 270


def test_changed_task_outcome_refuses_metric_but_keeps_failed_usage():
    result = paired(right=[observed(outcome="failed", rows=[("retry", receipt("fail", inputs=130))])], population_complete=True)
    assert "per_task_outcomes_differ" in result["ineligible_reasons"]
    assert result["candidate"]["usage"]["provider_total_tokens"] == 150
    assert result["paired_metrics"]["provider_token_delta"] is None


def test_interrupted_attempt_preserves_lower_bound_without_fabricating_complete_savings():
    result = paired(right=[observed(rows=[("retry", receipt("interrupted", inputs=70, complete=False))])], population_complete=True)
    assert result["candidate"]["usage"]["known_provider_token_subtotal"] == 90
    assert result["candidate"]["usage"]["provider_total_tokens"] is None
    assert result["paired_metrics"]["provider_token_delta"] is None


def test_repeated_receipts_and_cumulative_updates_charge_one_session_once():
    early = receipt("early", inputs=40, complete=False)
    late = receipt("late", thread=early["native_rollout_usage"]["thread_id"], inputs=80, count=2)
    result = paired(right=[observed(rows=[("coding", early), ("retry", late), ("expansion", deepcopy(late))])], population_complete=True)
    assert result["candidate"]["usage"]["provider_total_tokens"] == 100
    assert result["candidate"]["usage"]["native_sessions_observed"] == 1
    assert result["candidate"]["usage"]["repeated_receipt_observations"] == 1
    assert result["candidate"]["native_session_task_binding_issues"] == []
    assert result["paired_metrics_eligible"] is True


def test_nonmonotonic_session_is_quarantined_without_choosing_largest_snapshot():
    early = receipt("early", inputs=150, complete=False)
    late = receipt("late", thread=early["native_rollout_usage"]["thread_id"], inputs=80, count=2)
    result = paired(right=[observed(rows=[("coding", early), ("retry", late)])], population_complete=True)
    assert result["candidate"]["usage"]["sessions"][0]["status"] == "conflict"
    assert result["candidate"]["usage"]["known_provider_token_subtotal"] is None
    assert result["paired_metrics"]["provider_token_delta"] is None


def test_cache_reasoning_local_prompt_and_tool_counts_are_not_added_to_native_usage():
    row = receipt("candidate", inputs=100, outputs=20)
    row.update(tool_tokens=5000, prompt_selection={"selected_content_tokens": 5000})
    result = paired(right=[observed(rows=[("coding", row)])], population_complete=True)
    assert result["candidate"]["usage"]["provider_total_tokens"] == 120
    assert result["candidate"]["usage"]["known_counter_subtotals"]["cached_input_tokens"] == 50
    assert result["candidate"]["usage"]["known_counter_subtotals"]["reasoning_output_tokens"] == 10
    assert result["paired_metrics"]["provider_token_delta"] == 0


@pytest.mark.parametrize("field,value", [("tree_id", "tree:changed"), ("acceptance_binding", "acceptance:changed"),
                                         ("provider", "other"), ("model", "other")])
def test_changed_pair_binding_refuses(field, value):
    right = observed()
    right[field] = value
    with pytest.raises(ValueError, match="bindings differ"):
        paired(right=[right])


def test_receipt_model_mismatch_refuses_metric_without_dropping_charge():
    row = receipt("candidate")
    row["model"] = row["usage"]["model_id"] = "other-model"
    result = paired(right=[observed(rows=[("coding", row)])], population_complete=True)
    assert result["candidate"]["usage"]["provider_total_tokens"] == 120
    assert "candidate_receipt_bindings_differ" in result["ineligible_reasons"]
    assert result["paired_metrics"]["provider_token_delta"] is None


def test_shared_session_between_arms_cannot_establish_independent_reduction():
    result = paired(right=[observed()], population_complete=True)
    assert "native_session_shared_across_comparison_arms" in result["ineligible_reasons"]
    assert result["paired_metrics"]["provider_token_delta"] is None


def test_empty_assigned_population_or_no_accepted_tasks_cannot_claim_positive():
    empty = paired(assigned=[], left=[], right=[], population_complete=True)
    assert empty["paired_metrics"]["provider_token_delta"] is None
    failed = paired(left=[observed(outcome="failed")],
                    right=[observed(outcome="failed", rows=[("coding", receipt("candidate", inputs=10))])],
                    population_complete=True)
    assert failed["candidate"]["usage"]["provider_total_tokens"] == 30
    assert failed["paired_metrics"]["provider_token_delta"] is None


@pytest.mark.parametrize("mutation", ["missing_task", "duplicate_task", "unlisted_task", "accepted_int", "accepted_inconsistent", "extra_field", "bad_kind", "duplicate_attempt", "empty_attempts"])
def test_closed_population_and_dispositions_cannot_be_relaxed(mutation):
    right = [observed(rows=[("coding", receipt("candidate"))])]
    if mutation == "missing_task":
        right = []
    elif mutation == "duplicate_task":
        right.append(deepcopy(right[0]))
    elif mutation == "unlisted_task":
        right[0]["task_id"] = "not-assigned"
    elif mutation == "accepted_int":
        right[0]["accepted"] = 1
    elif mutation == "accepted_inconsistent":
        right[0]["outcome"] = "failed"
    elif mutation == "extra_field":
        right[0]["authority"] = True
    elif mutation == "bad_kind":
        right[0]["attempts"][0]["kind"] = "arbitrary"
    elif mutation == "duplicate_attempt":
        right[0]["attempts"].append(deepcopy(right[0]["attempts"][0]))
    else:
        right[0]["attempts"] = []
        result = paired(right=right, population_complete=True)
        assert "candidate_task_has_no_supplied_attempt_observation" in result["ineligible_reasons"]
        return
    with pytest.raises(ValueError):
        paired(right=right)


def test_bounds_and_unknown_native_usage_do_not_fabricate_zero():
    with pytest.raises(ValueError, match="assigned task"):
        paired(assigned=[task(str(i)) for i in range(MAX_TASKS + 1)])
    row = receipt("unavailable")
    row["native_rollout_usage"] = None
    result = paired(right=[observed(rows=[("coding", row)])], population_complete=True)
    assert result["candidate"]["usage"]["known_provider_token_subtotal"] is None
    assert result["paired_metrics"]["provider_token_delta"] is None


@pytest.mark.parametrize("arm", ["baseline", "candidate"])
@pytest.mark.parametrize("sharing", ["repeated_receipt", "cumulative_session"])
def test_native_session_shared_across_tasks_preserves_charge_but_withholds_paired_metrics(arm, sharing):
    first, second = task(), task("task-2")
    arms = {
        name: [observed(first, rows=[("coding", receipt(name + "-first"))]),
               observed(second, rows=[("coding", receipt(name + "-second"))])]
        for name in ("baseline", "candidate")
    }
    early = receipt(arm + "-shared", inputs=40, complete=sharing == "repeated_receipt")
    late = (deepcopy(early) if sharing == "repeated_receipt" else
            receipt(arm + "-late", inputs=80, count=2,
                    thread=early["native_rollout_usage"]["thread_id"]))
    arms[arm] = [observed(first, rows=[("coding", early)]),
                 observed(second, rows=[("retry", late)])]
    before = deepcopy(arms)
    result = paired(assigned=[first, second], left=arms["baseline"], right=arms["candidate"],
                    population_complete=True)
    assert arms == before
    charged = result[arm]
    assert charged["usage"]["provider_total_tokens"] == (60 if sharing == "repeated_receipt" else 100)
    assert charged["usage"]["native_sessions_observed"] == 1
    assert charged["usage"]["repeated_receipt_observations"] == (1 if sharing == "repeated_receipt" else 0)
    assert charged["accepted_task_count"] == result["assigned_task_count"] == 2
    assert charged["attempt_observation_count"] == 2
    assert charged["native_session_task_binding_issues"] == [{
        "thread_id": early["native_rollout_usage"]["thread_id"],
        "task_ids": [first["task_id"], second["task_id"]],
        "reason": "native_session_assigned_to_multiple_tasks",
    }]
    assert arm + "_native_session_shared_across_tasks" in result["ineligible_reasons"]
    assert result["paired_metrics_eligible"] is False
    assert result["paired_metrics"]["positive_provider_token_reduction_observed"] is False
    assert all(value is None for key, value in result["paired_metrics"].items()
               if key != "positive_provider_token_reduction_observed")
