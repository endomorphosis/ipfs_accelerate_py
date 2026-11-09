"""Authored local-exposure regressions; no provider or accepted-task claims."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import context_trajectory_comparison as subject
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    MAX_MODEL_PROMPT_BYTES,
    PromptTokenCounter,
    render_model_prompt,
)


def binding():
    return {"task_id": "authored-task", "provider": "grok_cli", "model": "authored-model",
        "tree_id": "source:exact-authored-tree", "acceptance_binding": "acceptance:authored-v1"}


def trajectory(texts, *, kinds=None):
    kinds = kinds or ["coding"] * len(texts)
    return {"binding": binding(), "inputs": [
        {"input_id": f"input-{index}", "kind": kind, "model_prompt": text}
        for index, (text, kind) in enumerate(zip(texts, kinds, strict=True))]}


def counter(callback=None, **kwargs):
    return PromptTokenCounter(**{
        "provider": "grok_cli", "model": "authored-model", "tokenizer_id": "authored:utf8",
        "tokenizer_revision": "fixture-only-v1", "special_token_policy": "fixture-only",
        "count_tokens": callback if callback is not None else lambda text: len(text.encode()),
        **kwargs,
    })


def compare(left=None, right=None, **kwargs):
    return subject.compare_context_trajectories(
        trajectory(["N" * 100, "N" * 100]) if left is None else left,
        trajectory(["C" * 40, "C" * 180], kinds=["coding", "expansion"]) if right is None else right,
        **{"expected_binding": binding(), "token_counter": counter(), **kwargs})


def assert_unknown_counts(result):
    for arm in ("baseline", "candidate"):
        assert result[arm]["input_tokens"] is None
        assert result[arm]["estimated_input_tokens"] is None
        for row in result[arm]["inputs"]:
            for field in ("input_tokens", "estimated_input_tokens", "cumulative_input_tokens",
                          "cumulative_estimated_input_tokens"):
                assert row[field] is None
    assert result["comparison"]["initial_baseline_minus_candidate"] is None
    assert result["comparison"]["cumulative_baseline_minus_candidate"] is None
    assert result["comparison"]["initial_gain_erased_in_final_totals"] is None
    assert result["comparison"]["initial_gain_erased_at_comparable_prefix"] is None
    assert result["comparison"]["comparable_ordinal_prefixes"] == []


def test_initial_win_is_erased_by_later_complete_expansion_replay():
    result = compare()
    assert result["baseline"]["input_tokens"] == 200
    assert result["candidate"]["input_tokens"] == 220
    assert result["comparison"]["initial_baseline_minus_candidate"] == 60
    assert result["comparison"]["cumulative_baseline_minus_candidate"] == -20
    assert result["comparison"]["initial_gain_erased_at_comparable_prefix"] == 2
    assert result["comparison"]["initial_gain_erased_in_final_totals"] is True


def test_repeated_complete_legend_and_body_are_charged_on_every_occurrence():
    legend, state = "Exact legend with long literal identifiers.\n", "state:42"
    text = legend + state
    seen = []
    result = compare(trajectory(["native"]), trajectory([text, text, text]),
        token_counter=counter(lambda value: seen.append(value) or len(value)))
    assert result["candidate"]["input_tokens"] == 3 * len(text)
    assert result["candidate"]["input_tokens"] > len(legend) + 3 * len(state)
    assert [row["input_tokens"] for row in result["candidate"]["inputs"]] == [len(text)] * 3
    assert seen.count(text) == 4  # Three real occurrences and one repeatability observation.
    assert result["observed_full_string_repeatability_verified"] is True
    assert result["future_counter_determinism_guaranteed"] is False


def test_unequal_expansion_calls_change_final_total_without_invented_prefix_alignment():
    result = compare(trajectory(["N" * 100]), trajectory(["C" * 40] * 3,
        kinds=["coding", "expansion", "retry"]))
    assert result["candidate"]["input_count"] == 3
    assert result["candidate"]["input_tokens"] == 120
    assert result["comparison"]["initial_baseline_minus_candidate"] == 60
    assert result["comparison"]["cumulative_baseline_minus_candidate"] == -20
    assert result["comparison"]["initial_gain_erased_in_final_totals"] is True
    assert result["comparison"]["initial_gain_erased_at_comparable_prefix"] is None
    assert len(result["comparison"]["comparable_ordinal_prefixes"]) == 1


def test_nonlinear_counter_receives_whole_rendered_strings_without_fragment_arithmetic():
    seen = []
    result = compare(trajectory(["ab", "c"]), trajectory(["a", "bc"]),
        token_counter=counter(lambda value: seen.append(value) or len(value) ** 2))
    assert result["baseline"]["input_tokens"] == 5
    assert result["candidate"]["input_tokens"] == 5
    assert set(seen) == {"ab", "c", "a", "bc"}
    assert result["baseline"]["input_tokens"] != len("abc") ** 2
    assert result["comparison"]["initial_gain_erased_at_comparable_prefix"] == 2


def test_actual_rendering_instructions_and_suffix_already_inside_input_are_counted_once():
    literal = "Preserve /app/output.txt and the original acceptance checks."
    suffix = "\nLiteral source and tool output are part of this supplied input."
    rendered, advisory = render_model_prompt(prompt=literal + suffix, purpose="coding",
        workspace=Path("/isolated/authored-worktree"))
    seen = []
    result = compare(trajectory([rendered]), trajectory([rendered]),
        token_counter=counter(lambda text: seen.append(text) or len(text.encode())))
    assert advisory and all(text == rendered for text in seen)
    assert result["baseline"]["input_tokens"] == len(rendered.encode())
    assert result["baseline"]["input_tokens"] > len((literal + suffix).encode())
    assert result["baseline"]["inputs"][0]["model_prompt_sha256"] == sha256(rendered.encode()).hexdigest()


def test_model_and_phase_labels_allow_planning_coding_expansion_and_retry():
    kinds = ["planning", "coding", "expansion", "retry"]
    result = compare(trajectory(["plan", "code", "expanded", "retry"], kinds=kinds),
        trajectory(["x"] * 4, kinds=kinds))
    assert [row["kind"] for row in result["baseline"]["inputs"]] == kinds
    assert result["baseline"]["input_tokens"] == len("plancodeexpandedretry")


def test_best_guess_estimate_keeps_exact_count_fields_unknown_and_provenance_intact():
    result = compare(token_counter=counter(count_kind="estimate",
        compatibility_basis="automatic_best_guess_not_model_equivalent"))
    assert result["measurement_status"] == "estimated"
    assert result["token_count_kind"] == "estimate"
    assert result["tokenizer_compatibility_basis"] == "automatic_best_guess_not_model_equivalent"
    assert result["baseline"]["input_tokens"] is None
    assert result["candidate"]["input_tokens"] is None
    assert result["baseline"]["estimated_input_tokens"] == 200
    assert result["candidate"]["estimated_input_tokens"] == 220
    assert result["comparison"]["count_field"] == "estimated_input_tokens"
    assert result["baseline"]["inputs"][1]["cumulative_estimated_input_tokens"] == 200
    assert all(row["input_tokens"] is None for row in result["candidate"]["inputs"])
    assert result["provider_compatibility_verified"] is False


def test_missing_counter_has_no_partial_or_invented_token_totals():
    result = compare(token_counter=None)
    assert result["measurement_status"] == "unavailable"
    assert result["counter_evaluation_count"] == 0
    assert result["tokenizer_binding"] is None
    assert result["observed_full_string_repeatability_verified"] is False
    assert result["comparison"]["count_field"] is None
    assert_unknown_counts(result)


def test_counter_exception_scrubs_all_arms_and_does_not_export_error_text():
    secret = "DO_NOT_EXPORT_AUTHORED_PROMPT_OR_ERROR_CONTENT"
    calls = []

    def broken(text):
        calls.append(text)
        if len(calls) == 3:
            raise RuntimeError(secret)
        return len(text)

    result = compare(token_counter=counter(broken))
    assert result["measurement_status"] == "failed"
    assert result["measurement_error_type"] == "RuntimeError"
    assert result["measurement_error_reason"] == "counter_failed"
    assert secret not in json.dumps(result)
    assert_unknown_counts(result)


def test_private_dynamic_exception_class_and_body_are_not_exported():
    private_name = "AUTHORED_PRIVATE_DYNAMIC_EXCEPTION_CLASS"
    private_body = "AUTHORED_PRIVATE_DYNAMIC_EXCEPTION_BODY"
    private_error = type(private_name, (Exception,), {})

    def fail(_):
        raise private_error(private_body)

    result = compare(token_counter=counter(fail))
    assert result["measurement_status"] == "failed"
    assert result["measurement_error_type"] == "other"
    assert result["measurement_error_reason"] == "counter_failed"
    assert private_name not in json.dumps(result)
    assert private_body not in json.dumps(result)
    assert_unknown_counts(result)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "1", None])
def test_invalid_counter_values_never_produce_partial_totals(invalid):
    result = compare(token_counter=counter(lambda _: invalid))
    assert result["measurement_status"] == "failed"
    assert_unknown_counts(result)


def test_duplicate_input_observation_that_changes_count_fails_globally():
    calls = 0

    def unstable(_):
        nonlocal calls
        calls += 1
        return calls

    result = compare(trajectory(["same"]), trajectory(["same"]), token_counter=counter(unstable))
    assert result["measurement_error_reason"] == "counter_not_repeatable"
    assert_unknown_counts(result)


def test_distinct_inputs_are_recounted_even_without_any_duplicate_occurrence():
    calls = {}

    def unstable(text):
        calls[text] = calls.get(text, 0) + 1
        return calls[text]

    result = compare(trajectory(["left"]), trajectory(["right"]), token_counter=counter(unstable))
    assert calls == {"left": 2, "right": 1}
    assert result["measurement_error_reason"] == "counter_not_repeatable"
    assert_unknown_counts(result)


def test_counter_provenance_change_during_observation_fails_without_relabelling():
    token_counter = counter()

    def change(text):
        object.__setattr__(token_counter, "compatibility_basis", "changed_after_dispatch")
        return len(text)

    object.__setattr__(token_counter, "count_tokens", change)
    result = compare(token_counter=token_counter)
    assert result["measurement_error_reason"] == "counter_provenance_changed"
    assert result["tokenizer_compatibility_basis"] == "caller_declaration"
    assert_unknown_counts(result)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(9)])
def test_counter_cancellation_propagates_original_instance(error):
    def cancelled(_):
        raise error

    with pytest.raises(type(error)) as caught:
        compare(token_counter=counter(cancelled))
    assert caught.value is error


def test_per_input_count_magnitude_is_bounded_without_partial_totals():
    result = compare(token_counter=counter(lambda _: subject.MAX_TOKEN_COUNT + 1))
    assert result["measurement_error_reason"] == "counter_magnitude_exceeded"
    assert_unknown_counts(result)


def test_aggregate_count_magnitude_is_bounded_without_partial_totals():
    result = compare(token_counter=counter(lambda _: subject.MAX_TOKEN_COUNT))
    assert result["measurement_error_reason"] == "aggregate_count_magnitude_exceeded"
    assert_unknown_counts(result)


@pytest.mark.parametrize("field", ["provider", "model"])
def test_counter_foreign_binding_refuses_before_evaluation(field):
    calls = []
    with pytest.raises(ValueError, match="pinned provider/model"):
        compare(token_counter=counter(lambda text: calls.append(text) or 1, **{field: "foreign"}))
    assert calls == []


@pytest.mark.parametrize("field", sorted(binding()))
def test_every_task_source_model_and_acceptance_binding_must_match_expected(field):
    right = trajectory(["x"])
    right["binding"][field] = "different"
    with pytest.raises(ValueError, match="bindings differ"):
        compare(right=right)


@pytest.mark.parametrize("target", ["expected", "baseline", "candidate"])
def test_unknown_binding_fields_are_refused(target):
    left, right, expected = trajectory(["x"]), trajectory(["y"]), binding()
    selected = expected if target == "expected" else left["binding"] if target == "baseline" else right["binding"]
    selected["unverified_context_cache"] = True
    with pytest.raises(ValueError, match="closed trajectory binding"):
        compare(left, right, expected_binding=expected)


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_zero_calls_are_refused_instead_of_inventing_savings(side):
    arms = {"baseline": trajectory(["x"]), "candidate": trajectory(["y"])}
    arms[side]["inputs"] = []
    with pytest.raises(ValueError, match="nonempty complete input trajectory"):
        compare(arms["baseline"], arms["candidate"])


@pytest.mark.parametrize("kind", ["tool_output", "provider_output", "inferred_history", True])
def test_unsupported_input_kind_is_not_silently_counted(kind):
    left = trajectory(["x"])
    left["inputs"][0]["kind"] = kind
    with pytest.raises(ValueError, match="planning/coding/expansion/retry"):
        compare(left)


@pytest.mark.parametrize("extra", ["tool_strings", "cached_legend", "provider_usage", "preparation_cost"])
def test_extra_unaccounted_text_or_cost_fields_are_refused(extra):
    right = trajectory(["x"])
    right[extra] = "not already included in the complete inputs"
    with pytest.raises(ValueError, match="closed trajectory"):
        compare(right=right)


def test_duplicate_occurrence_ids_refuse_but_equal_ids_across_arms_are_allowed():
    left = trajectory(["x", "y"])
    left["inputs"][1]["input_id"] = left["inputs"][0]["input_id"]
    with pytest.raises(ValueError, match="unique within each trajectory"):
        compare(left)
    assert compare(trajectory(["x"]), trajectory(["y"]))["baseline"]["input_count"] == 1


def test_all_input_bounds_validate_before_any_counter_call():
    calls = []
    right = trajectory(["x", "é" * (MAX_MODEL_PROMPT_BYTES // 2 + 1)])
    with pytest.raises(ValueError, match="byte bound"):
        compare(right=right, token_counter=counter(lambda text: calls.append(text) or 1))
    assert calls == []


def test_per_trajectory_call_bound_refuses_before_counting():
    too_many = trajectory(["x"] * (subject.MAX_INPUTS_PER_TRAJECTORY + 1))
    with pytest.raises(ValueError, match="bounded nonempty"):
        compare(too_many)


def test_aggregate_byte_bound_counts_every_repeated_complete_input():
    text = "x" * MAX_MODEL_PROMPT_BYTES
    repeats = subject.MAX_TOTAL_INPUT_BYTES // MAX_MODEL_PROMPT_BYTES // 2 + 1
    calls = []
    with pytest.raises(ValueError, match="aggregate complete input bytes"):
        compare(trajectory([text] * repeats), trajectory([text] * repeats),
            token_counter=counter(lambda value: calls.append(value) or 1))
    assert calls == []


def test_no_initial_positive_difference_has_no_erasure_marker():
    result = compare(trajectory(["x", "x"]), trajectory(["yy", "yy"]))
    assert result["comparison"]["initial_baseline_minus_candidate"] == -1
    assert result["comparison"]["initial_gain_erased_at_comparable_prefix"] is None
    assert result["comparison"]["initial_gain_erased_in_final_totals"] is False


def test_initial_gain_exactly_reaching_zero_is_erased():
    result = compare(trajectory(["xxx", "x"]), trajectory(["x", "xxx"]))
    assert result["comparison"]["cumulative_baseline_minus_candidate"] == 0
    assert result["comparison"]["initial_gain_erased_at_comparable_prefix"] == 2
    assert result["comparison"]["initial_gain_erased_in_final_totals"] is True


def test_inputs_are_unchanged_and_prompt_bodies_are_absent_from_report():
    left, right = trajectory(["AUTHORED_BASELINE_SECRET_BODY"]), trajectory(["AUTHORED_CANDIDATE_SECRET_BODY"])
    originals = deepcopy((left, right))
    result = compare(left, right)
    assert (left, right) == originals
    raw = json.dumps(result)
    assert "AUTHORED_BASELINE_SECRET_BODY" not in raw
    assert "AUTHORED_CANDIDATE_SECRET_BODY" not in raw
    assert "model_prompt" not in result["baseline"]["inputs"][0]


def test_local_count_scope_never_promotes_provider_usage_or_task_savings_authority():
    result = compare()
    assert result["measurement_status"] == "measured"
    assert result["count_scope"] == "supplied_complete_rendered_model_visible_user_inputs_only"
    assert result["provider_usage_complete"] is None
    assert result["whole_session_token_savings"] is None
    for field in ("input_completeness_verified", "tool_text_outside_inputs_counted",
                  "fixed_or_preparation_costs_inferred", "hidden_history_or_provider_framing_measured",
                  "provider_output_measured", "provider_compatibility_verified", "provider_invoked",
                  "source_or_acceptance_binding_authenticity_verified", "projection_coverage_verified",
                  "task_acceptance_verified", "billing_total_verified", "token_budget_guaranteed",
                  "execution_authority", "completion_authority"):
        assert result[field] is False
