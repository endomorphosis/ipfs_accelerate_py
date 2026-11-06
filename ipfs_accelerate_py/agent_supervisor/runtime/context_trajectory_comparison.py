"""Pure comparison of supplied complete model-visible input trajectories.

Every input must already contain its complete rendered user message, including
any required instructions, legend, prior deliveries or tool text. Count repeated
inputs every time they occur. No fragment costs, hidden history, native framing,
provider output, preparation cost or provider charges are inferred or added.
The caller declares task/source/acceptance identities; this module verifies their
equality, not their authenticity or semantic equivalence. It invokes no provider
and retains no prompt bodies. A local count or estimate grants no authority.
"""
from __future__ import annotations

from hashlib import sha256
import json

from .router_implementation_runner import (
    MAX_MODEL_PROMPT_BYTES,
    RUNNER_FAILURE_TYPES,
    PromptTokenCounter,
)

SCHEMA = "supervisor-context-trajectory-comparison@1"
MAX_INPUTS_PER_TRAJECTORY = 128
MAX_TOTAL_INPUT_BYTES = 8 * 1024 * 1024
MAX_TOKEN_COUNT = (1 << 63) - 1
MAX_REPORT_BYTES = 1024 * 1024
_BINDING_FIELDS = frozenset({"task_id", "provider", "model", "tree_id", "acceptance_binding"})
_INPUT_FIELDS = frozenset({"input_id", "kind", "model_prompt"})
_KINDS = frozenset({"planning", "coding", "expansion", "retry"})
_COUNTER_REASONS = frozenset({"counter_magnitude_exceeded", "counter_not_repeatable",
    "counter_provenance_changed", "aggregate_count_magnitude_exceeded"})


def _text(value, name, maximum=256):
    if (type(value) is not str or not value or value != value.strip()
            or "\x00" in value or len(value.encode("utf-8")) > maximum):
        raise ValueError(f"bounded explicit {name} required")


def _closed(value, fields, name):
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"closed {name} fields required")


def _binding(value):
    _closed(value, _BINDING_FIELDS, "trajectory binding")
    for name in _BINDING_FIELDS:
        _text(value[name], name, 4096 if name in {"tree_id", "acceptance_binding"} else 256)
    return dict(value)


def _trajectory(value, binding):
    _closed(value, {"binding", "inputs"}, "trajectory")
    if _binding(value["binding"]) != binding:
        raise ValueError("trajectory task/provider/model/source/acceptance bindings differ")
    inputs = value["inputs"]
    if type(inputs) is not list or not 1 <= len(inputs) <= MAX_INPUTS_PER_TRAJECTORY:
        raise ValueError("bounded nonempty complete input trajectory required")
    seen, texts, rows = set(), [], []
    for item in inputs:
        _closed(item, _INPUT_FIELDS, "complete input")
        _text(item["input_id"], "input_id")
        if item["input_id"] in seen:
            raise ValueError("input identities must be unique within each trajectory")
        if type(item["kind"]) is not str or item["kind"] not in _KINDS:
            raise ValueError("explicit planning/coding/expansion/retry kind required")
        text = item["model_prompt"]
        if type(text) is not str:
            raise ValueError("complete rendered input must be a literal string")
        raw = text.encode("utf-8")
        if not 1 <= len(raw) <= MAX_MODEL_PROMPT_BYTES:
            raise ValueError("complete rendered input exceeds its nonempty byte bound")
        seen.add(item["input_id"])
        texts.append(text)
        rows.append({"input_id": item["input_id"], "kind": item["kind"],
            "model_prompt_sha256": sha256(raw).hexdigest(), "model_prompt_bytes": len(raw),
            "input_tokens": None, "estimated_input_tokens": None,
            "cumulative_input_tokens": None, "cumulative_estimated_input_tokens": None})
    return texts, {"input_count": len(rows), "model_prompt_bytes": sum(
        row["model_prompt_bytes"] for row in rows), "inputs": rows,
        "input_tokens": None, "estimated_input_tokens": None}


class _CounterObservationError(ValueError):
    pass


def compare_context_trajectories(baseline: dict, candidate: dict, *,
                                 expected_binding: dict,
                                 token_counter: PromptTokenCounter | None) -> dict:
    """Compare complete local input exposure for two explicit ordered traces.

    Unequal input counts are allowed and every supplied extra input contributes
    to final totals. The earliest-erasure diagnostic compares only their common
    ordinal prefixes: it does not align elapsed time, task progress or semantic
    state. If extra calls erase the initial gain beyond that common prefix, the
    separate final-total diagnostic still records it.

    Each occurrence is counted; then each distinct full string is counted again
    to check repeatability in this observation. This is not a future determinism
    guarantee. A missing, failed, changing or out-of-bounds counter exposes no
    partial token totals for either trace. Cancellation BaseExceptions propagate.
    """
    binding = _binding(expected_binding)
    if token_counter is not None and (
            type(token_counter) is not PromptTokenCounter
            or token_counter.provider != binding["provider"]
            or token_counter.model != binding["model"]):
        raise ValueError("token counter differs from pinned provider/model")
    baseline_texts, baseline_report = _trajectory(baseline, binding)
    candidate_texts, candidate_report = _trajectory(candidate, binding)
    if baseline_report["model_prompt_bytes"] + candidate_report["model_prompt_bytes"] > MAX_TOTAL_INPUT_BYTES:
        raise ValueError("aggregate complete input bytes exceed the trajectory bound")
    counter_binding = token_counter.binding() if token_counter is not None else None
    count_kind = token_counter.count_kind if token_counter is not None else None
    compatibility = token_counter.compatibility_basis if token_counter is not None else None
    status = "unavailable" if token_counter is None else "estimated" if count_kind == "estimate" else "measured"
    evaluations, error_type, error_reason = 0, None, None
    measured = None
    if token_counter is not None:
        observations = {}

        def count(text):
            nonlocal evaluations
            evaluations += 1
            amount = token_counter.count(text)
            if amount > MAX_TOKEN_COUNT:
                raise _CounterObservationError("counter_magnitude_exceeded")
            if text in observations and observations[text] != amount:
                raise _CounterObservationError("counter_not_repeatable")
            observations[text] = amount
            return amount

        try:
            measured = [[count(text) for text in texts]
                for texts in (baseline_texts, candidate_texts)]
            for text in tuple(observations):
                count(text)
            if (token_counter.binding() != counter_binding
                    or token_counter.count_kind != count_kind
                    or token_counter.compatibility_basis != compatibility):
                raise _CounterObservationError("counter_provenance_changed")
            if any(sum(amounts) > MAX_TOKEN_COUNT for amounts in measured):
                raise _CounterObservationError("aggregate_count_magnitude_exceeded")
        except Exception as error:
            status, measured = "failed", None
            error_type = ("ValueError" if type(error) is _CounterObservationError
                else type(error).__name__ if type(error).__name__ in RUNNER_FAILURE_TYPES else "other")
            error_reason = (str(error) if type(error) is _CounterObservationError
                and str(error) in _COUNTER_REASONS else "counter_failed")

    field = "estimated_input_tokens" if count_kind == "estimate" else "input_tokens"
    initial, final, first_erasure = None, None, None
    prefixes = []
    if measured is not None:
        for report, amounts in zip((baseline_report, candidate_report), measured, strict=True):
            running = 0
            for row, amount in zip(report["inputs"], amounts, strict=True):
                running += amount
                row[field] = amount
                row["cumulative_" + field] = running
            report[field] = running
        initial = measured[0][0] - measured[1][0]
        final = baseline_report[field] - candidate_report[field]
        for index in range(min(len(measured[0]), len(measured[1]))):
            left = baseline_report["inputs"][index]["cumulative_" + field]
            right = candidate_report["inputs"][index]["cumulative_" + field]
            delta = left - right
            prefixes.append({"input_ordinal": index + 1,
                "baseline_cumulative": left, "candidate_cumulative": right,
                "baseline_minus_candidate": delta})
            if initial > 0 and delta <= 0 and first_erasure is None:
                first_erasure = index + 1
    report = {
        "schema": SCHEMA, "binding": binding,
        "count_scope": "supplied_complete_rendered_model_visible_user_inputs_only",
        "input_completeness_verified": False,
        "measurement_status": status, "measurement_error_type": error_type,
        "measurement_error_reason": error_reason,
        "tokenizer_binding": counter_binding, "token_count_kind": count_kind,
        "tokenizer_compatibility_basis": compatibility,
        "counter_evaluation_count": evaluations,
        "observed_full_string_repeatability_verified": measured is not None,
        "future_counter_determinism_guaranteed": False,
        "baseline": baseline_report, "candidate": candidate_report,
        "comparison": {
            "count_field": field if token_counter is not None else None,
            "initial_baseline_minus_candidate": initial,
            "cumulative_baseline_minus_candidate": final,
            "initial_gain_erased_at_comparable_prefix": first_erasure,
            "initial_gain_erased_in_final_totals": (
                initial > 0 and final <= 0 if measured is not None else None),
            "comparable_ordinal_prefixes": prefixes,
            "prefix_scope": "common_call_ordinals_only_not_time_task_progress_or_semantic_equivalence",
            "extra_input_calls_in_final_total_definition": True,
        },
        "tool_text_outside_inputs_counted": False,
        "fixed_or_preparation_costs_inferred": False,
        "hidden_history_or_provider_framing_measured": False,
        "provider_output_measured": False, "provider_usage_complete": None,
        "provider_compatibility_verified": False, "provider_invoked": False,
        "source_or_acceptance_binding_authenticity_verified": False,
        "projection_coverage_verified": False, "task_acceptance_verified": False,
        "whole_session_token_savings": None, "billing_total_verified": False,
        "token_budget_guaranteed": False, "execution_authority": False,
        "completion_authority": False,
    }
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("trajectory comparison report exceeds its byte bound")
    return report
