"""Compare supplied context and usage observations without dispatch or admission.

Local complete rendered user-message counts stay separate from native cumulative
provider input plus output. This module wraps the existing reconciler; it owns
no token ledger, provider, source authority, or task acceptance decision.
"""
from __future__ import annotations

import json
from pathlib import Path

from .cumulative_usage_reconciliation import MAX_RECEIPTS, reconcile_router_cumulative_usage
from .router_implementation_runner import (
    MAX_MODEL_PROMPT_BYTES,
    PromptTokenCounter,
    select_model_prompt,
)

PROMPT_SCHEMA = "supervisor-context-candidate-comparison@1"
PAIR_SCHEMA = "supervisor-context-projection-comparison@1"
MAX_TASKS = 512
MAX_REPORT_BYTES = 8 * 1024 * 1024
_BINDINGS = {"task_id", "provider", "model", "tree_id", "acceptance_binding"}
_OUTCOMES = {"accepted", "failed", "interrupted", "rejected"}
_KINDS = {"planning", "coding", "retry", "expansion"}
_BASES = {"retained_native_receipts", "authored_fixture"}


def _text(value, name, maximum=256):
    if (type(value) is not str or not value or value != value.strip()
            or "\x00" in value or len(value.encode("utf-8")) > maximum):
        raise ValueError(f"bounded explicit {name} required")
    return value


def _closed(value, fields, name):
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"closed {name} fields required")


def _bounded_report(report):
    # All records here are derived projections; arbitrary receipt fields never
    # enter the report. The existing reconciler also bounds retained snapshots.
    raw = json.dumps(report, ensure_ascii=True, separators=(",", ":"),
                     sort_keys=True, allow_nan=False).encode()
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("comparison report exceeds its byte bound")
    return report


def measure_context_candidates(*, native_prompt: str, projected_prompt: str,
                               alias_prompt: str | None = None, suffix: str,
                               purpose: str, workspace: Path, provider: str, model: str,
                               token_counter: PromptTokenCounter | None,
                               measurement_basis: str = "owner_declared_local_tokenizer",
                               max_input_tokens: int | None = None,
                               max_prompt_bytes: int = MAX_MODEL_PROMPT_BYTES) -> dict:
    """Measure three fully rendered inputs; do not establish projection validity.

    The caller supplies already verified projected/alias content and all required
    suffixes. Native and projected text use normal rendering; aliases use the
    existing semantic transport advisory. Fixture callbacks qualify accounting
    behavior only. No prompt body is retained by this report.
    """
    if type(measurement_basis) is not str or measurement_basis not in {
            "owner_declared_local_tokenizer", "authored_fixture", "automatic_offline_tokenizer"}:
        raise ValueError("explicit local measurement basis required")
    _text(provider, "provider")
    _text(model, "model")
    if not isinstance(workspace, Path) or not workspace.is_absolute():
        raise ValueError("explicit absolute workspace required")
    for name, value in (("native", native_prompt), ("projected", projected_prompt),
                        ("alias", alias_prompt), ("suffix", suffix)):
        if value is None and name == "alias":
            continue
        if type(value) is not str or len(value.encode()) > MAX_MODEL_PROMPT_BYTES:
            raise ValueError("bounded literal candidate and suffix strings required")
    arguments = dict(suffix=suffix, purpose=purpose, workspace=workspace,
                     provider=provider, model=model, token_counter=token_counter,
                     max_input_tokens=max_input_tokens, max_prompt_bytes=max_prompt_bytes)
    full = select_model_prompt(native_prompt=native_prompt, semantic_prompt=None,
                               **arguments)[3]
    compact = select_model_prompt(native_prompt=projected_prompt, semantic_prompt=alias_prompt,
                                  **arguments)[3]
    candidates = {"native": full["candidates"]["native"],
                  "projected": compact["candidates"]["native"]}
    if alias_prompt is not None:
        candidates["alias"] = compact["candidates"]["semantic"]
    count_field = token_counter.count_field if token_counter is not None else "input_tokens"
    for row in candidates.values():
        if row["model_prompt_bytes"] > max_prompt_bytes:
            raise ValueError("complete candidate exceeds unchanged prompt byte ceiling")
        if (max_input_tokens is not None and row.get(count_field) is not None
                and row[count_field] > max_input_tokens):
            raise ValueError("complete candidate exceeds unchanged prompt token ceiling")
    native = candidates["native"]
    comparisons = {}
    for name in ("projected", "alias"):
        if name not in candidates:
            continue
        row = candidates[name]
        known = native["input_tokens"] is not None and row["input_tokens"] is not None
        estimated = (native.get("estimated_input_tokens") is not None
                     and row.get("estimated_input_tokens") is not None)
        comparisons[name] = {
            "rendered_user_byte_delta": native["model_prompt_bytes"] - row["model_prompt_bytes"],
            "rendered_user_content_token_delta": native["input_tokens"] - row["input_tokens"] if known else None,
            "rendered_user_estimated_content_token_delta": (
                native["estimated_input_tokens"] - row["estimated_input_tokens"] if estimated else None),
        }
    return _bounded_report({
        "schema": PROMPT_SCHEMA, "provider": provider, "model": model,
        "tokenizer_binding": full["tokenizer_binding"], "measurement_basis": measurement_basis,
        "token_count_kind": full["token_count_kind"],
        "tokenizer_compatibility_basis": full["tokenizer_compatibility_basis"],
        "token_budget_basis": full["token_budget_basis"], "token_budget_guaranteed": False,
        "count_scope": "rendered_user_prompt_only", "candidates": candidates,
        "comparisons": comparisons,
        "measurement_status": {"native": full["measurement_status"], "projected_and_alias": compact["measurement_status"]},
        "projected_alias_selection": compact["selected_representation"],
        "projected_alias_selection_reason": compact["selection_reason"],
        "provider_compatibility_verified": False, "projection_coverage_verified": False,
        "alias_restoration_verified": False, "provider_invoked": False,
        "whole_session_token_savings": None, "campaign_usage_complete": None,
        "execution_authority": False, "completion_authority": False,
    })


def _assigned(records):
    if type(records) is not list or len(records) > MAX_TASKS:
        raise ValueError("bounded assigned task list required")
    result = {}
    for row in records:
        _closed(row, _BINDINGS, "assigned task")
        for field in _BINDINGS:
            _text(row[field], field, 4096 if field in {"tree_id", "acceptance_binding"} else 256)
        if row["task_id"] in result:
            raise ValueError("duplicate assigned task")
        result[row["task_id"]] = dict(row)
    return result


def _arm(records, assigned):
    if type(records) is not list or len(records) > MAX_TASKS:
        raise ValueError("bounded observed task list required")
    tasks, receipts, receipt_tasks, seen_attempts, mismatches = {}, [], [], set(), []
    for row in records:
        _closed(row, _BINDINGS | {"outcome", "accepted", "attempts"}, "observed task")
        task = row["task_id"]
        if type(task) is not str or task not in assigned or task in tasks:
            raise ValueError("exact unique assigned task observation required")
        if {field: row[field] for field in _BINDINGS} != assigned[task]:
            raise ValueError("paired task source/model/acceptance bindings differ")
        if (type(row["accepted"]) is not bool or type(row["outcome"]) is not str or row["outcome"] not in _OUTCOMES
                or row["accepted"] is not (row["outcome"] == "accepted")):
            raise ValueError("exact caller outcome and accepted flag required")
        if type(row["attempts"]) is not list or len(row["attempts"]) > MAX_RECEIPTS:
            raise ValueError("bounded ordered attempt observations required")
        attempts = []
        for attempt in row["attempts"]:
            _closed(attempt, {"attempt_id", "kind", "receipt"}, "attempt")
            attempt_id = _text(attempt["attempt_id"], "attempt_id")
            if attempt_id in seen_attempts or type(attempt["kind"]) is not str or attempt["kind"] not in _KINDS:
                raise ValueError("unique named planning/coding/retry/expansion attempt required")
            seen_attempts.add(attempt_id)
            if len(receipts) >= MAX_RECEIPTS:
                raise ValueError("aggregate receipt population exceeds reconciler bound")
            receipt = attempt["receipt"]
            if type(receipt) is not dict:
                raise ValueError("supplied router receipt mapping required")
            if receipt.get("provider") != row["provider"] or receipt.get("model") != row["model"]:
                mismatches.append({"task_id": task, "attempt_id": attempt_id,
                                   "reason": "receipt_provider_or_model_differs_from_task"})
            receipts.append(receipt)
            receipt_tasks.append(task)
            attempts.append({"attempt_id": attempt_id, "kind": attempt["kind"],
                             "receipt_observation_ordinal": len(receipts) - 1})
        tasks[task] = {"task_id": task, "outcome": row["outcome"], "accepted": row["accepted"],
                       "attempts": attempts, "attempt_count": len(attempts)}
    if set(tasks) != set(assigned):
        raise ValueError("all assigned tasks including failed/interrupted observations required")
    usage = reconcile_router_cumulative_usage(receipts)
    # Reconciliation deduplicates cumulative sessions across the whole arm.
    # Keep ownership from every observation, including exact repeats, so one
    # retained session cannot silently stand in for several assigned tasks.
    session_tasks = {row["thread_id"]: set() for row in usage["sessions"]}
    for task, receipt in zip(receipt_tasks, receipts, strict=True):
        native = receipt.get("native_rollout_usage")
        thread = native.get("thread_id") if type(native) is dict else None
        if type(thread) is str and thread in session_tasks:
            session_tasks[thread].add(task)
    ownership_issues = [
        {"thread_id": row["thread_id"],
         "task_ids": [task for task in assigned if task in session_tasks[row["thread_id"]]],
         "reason": "native_session_assigned_to_multiple_tasks"}
        for row in usage["sessions"] if len(session_tasks[row["thread_id"]]) > 1
    ]
    return {"tasks": [tasks[task] for task in assigned],
            "accepted_task_count": sum(row["accepted"] for row in tasks.values()),
            "attempt_observation_count": len(receipts), "receipt_binding_mismatches": mismatches,
            "native_session_task_binding_issues": ownership_issues,
            "usage": usage}


def compare_context_projection(*, assigned_tasks: list[dict], baseline: list[dict],
                               candidate: list[dict], population_complete: bool = False,
                               usage_evidence_basis: str = "retained_native_receipts") -> dict:
    """Pair caller observations; charge every supplied session in each arm once.

    Complete campaign coverage is a caller declaration, not proven by these
    receipts. Unmatched outcomes, missing/quarantined usage, empty populations,
    no accepted tasks, sessions shared across arms or tasks and authored fixtures never
    yield a positive savings claim. Acceptance is observed, never certified.
    """
    if (type(population_complete) is not bool or type(usage_evidence_basis) is not str
            or usage_evidence_basis not in _BASES):
        raise ValueError("exact population declaration and usage evidence basis required")
    assigned = _assigned(assigned_tasks)
    left, right = _arm(baseline, assigned), _arm(candidate, assigned)
    outcomes_match = all((a["outcome"], a["accepted"]) == (b["outcome"], b["accepted"])
                         for a, b in zip(left["tasks"], right["tasks"], strict=True))
    shared = {row["thread_id"] for row in left["usage"]["sessions"]} & {
        row["thread_id"] for row in right["usage"]["sessions"]}
    reasons = []
    if not assigned:
        reasons.append("zero_assigned_tasks")
    if left["accepted_task_count"] == 0 or right["accepted_task_count"] == 0:
        reasons.append("no_accepted_task_denominator")
    if not outcomes_match:
        reasons.append("per_task_outcomes_differ")
    if not population_complete:
        reasons.append("campaign_population_not_declared_complete")
    if usage_evidence_basis == "authored_fixture":
        reasons.append("authored_usage_fixture_not_provider_observation")
    if shared:
        reasons.append("native_session_shared_across_comparison_arms")
    for name, arm in (("baseline", left), ("candidate", right)):
        if arm["receipt_binding_mismatches"]:
            reasons.append(name + "_receipt_bindings_differ")
        if arm["native_session_task_binding_issues"]:
            reasons.append(name + "_native_session_shared_across_tasks")
        if not arm["usage"]["supplied_receipt_usage_complete"]:
            reasons.append(name + "_usage_incomplete_or_quarantined")
        if any(not row["attempts"] for row in arm["tasks"]):
            reasons.append(name + "_task_has_no_supplied_attempt_observation")
    eligible = not reasons
    total_left, total_right = left["usage"]["provider_total_tokens"], right["usage"]["provider_total_tokens"]
    savings = total_left - total_right if eligible else None
    count, accepted = len(assigned), left["accepted_task_count"]
    metrics = {
        "provider_token_delta": savings,
        "provider_token_reduction_fraction": savings / total_left if eligible and total_left > 0 else None,
        "baseline_provider_tokens_per_assigned_task": total_left / count if eligible else None,
        "candidate_provider_tokens_per_assigned_task": total_right / count if eligible else None,
        "baseline_provider_tokens_per_accepted_task": total_left / accepted if eligible else None,
        "candidate_provider_tokens_per_accepted_task": total_right / accepted if eligible else None,
        "positive_provider_token_reduction_observed": eligible and savings > 0,
    }
    return _bounded_report({
        "schema": PAIR_SCHEMA, "assigned_tasks": list(assigned.values()), "assigned_task_count": count,
        "baseline": left, "candidate": right, "per_task_outcomes_match": outcomes_match,
        "population_complete_declared": population_complete,
        "campaign_usage_complete": None,
        "campaign_population_verified": False,
        "declared_campaign_supplied_usage_complete": eligible,
        "campaign_completeness_basis": "caller_population_declaration_and_supplied_receipts_only" if population_complete else None,
        "usage_evidence_basis": usage_evidence_basis, "paired_metrics_eligible": eligible,
        "ineligible_reasons": reasons, "paired_metrics": metrics,
        "provider_token_definition": "input_tokens + output_tokens",
        "cached_and_reasoning_tokens_are_subsets": True, "local_prompt_counts_added": False,
        "tool_tokens_added": False, "acceptance_basis": "caller_observed_outcomes_only",
        "task_acceptance_verified": False, "billing_total_verified": False,
        "provider_invoked": False, "execution_authority": False, "completion_authority": False,
        "provider_usage_authentication_verified": False,
    })
