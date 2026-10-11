"""Closed observation of numerical summaries in an actual planner request.

This reads only the bounded public planner request and numerical receipt. It
does not export their contents, replay a model, authenticate arbitrary supplied
receipts, or establish source proofs. The owner calls it after its independently
checked planner dispatch, before coding, and retains the result in its report.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .terminal_numerical_policy import validate_numerical_profile

SCHEMA = "terminal-numerical-planner-observation@1"
SUMMARY_SCHEMA = "terminal-numerical-formalization-planning-summary@1"
MAX_REQUEST_BYTES = 262_144
_FALSE = dict(proof_authority=False, completion_authority=False,
    execution_authority=False, source_semantics_verified=False,
    worker_delivery_verified=False, raw_prompts_exported=False,
    model_summary_equality_verified=False,
    source_obligations_discharged=0, observer_provider_calls=0)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class _Refusal(ValueError):
    def __init__(self, reason, *, unavailable=False):
        self.reason, self.unavailable = reason, unavailable


def _require(condition, reason, *, unavailable=False):
    if not condition:
        raise _Refusal(reason, unavailable=unavailable)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_json_field")
        result[key] = value
    return result


def _json(raw):
    def reject_constant(_):
        raise _Refusal("nonfinite_json_value")
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=reject_constant)


def _all_numerical_summaries(payload):
    found, pending, visited = [], [(payload, 0)], 0
    while pending:
        value, depth = pending.pop()
        visited += 1
        _require(visited <= 32768 and depth <= 64, "planner_request_structure_exceeds_bound")
        if type(value) is dict:
            if value.get("schema") == SUMMARY_SCHEMA:
                found.append(value)
            pending.extend((child, depth + 1) for child in value.values())
        elif type(value) is list:
            pending.extend((child, depth + 1) for child in value)
    return found


def observe_numerical_planning(*, state, report, profile):
    """Return verified, mismatch, or unavailable metadata, never provider text.

    A verified full observation establishes receipt/request/dispatch joins, not
    that the model used the supplied constraints correctly. A verified no-index
    observation establishes absence in that actual request. Missing evidence is
    unavailable; callers must not reinterpret it as a successful delivery.
    """
    profile = validate_numerical_profile(profile)
    if profile is None:
        return None
    arm = report.get("arm")
    result = dict(schema=SCHEMA, profile=profile, arm=arm if arm in {"full", "no-index"} else None,
        status="unavailable", reason="evidence_not_checked", request_checked=False,
        planner_request_sha256=None, planner_request_bytes=None,
        model_prompt_sha256=None,
        request_hash_join_verified=False, planning_invocation_join_verified=False,
        planning_invocation_id=None, planning_execution_receipt_sha256=None,
        numerical_summary_count=None, numerical_summary_sha256=None,
        numerical_receipt_sha256=None, numerical_source_tree_id=None,
        exact_summary_match=False, live_planner_delivery_verified=False,
        no_index_absence_verified=False, provider_free=False,
        delivery_boundary="native_planner_request_submitted_to_router",
        observation_scope="initial_planning_only_not_post_publication_currentness", **_FALSE)
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_advisor import _read
        from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA
        from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import INTENT_PLAN_PROVIDER_REQUEST_SCHEMA

        state = Path(state)
        _require(state.is_absolute() and state.resolve(strict=True) == state, "noncanonical_state")
        _require(arm in {"full", "no-index"}, "unsupported_arm")
        raw = _read(state / "planner-provider-request.json", MAX_REQUEST_BYTES)
        payload = _json(raw)
        _require(type(payload) is dict, "planner_request_shape")
        graph = payload.get("graph_request") if payload.get("schema") == INTENT_PLAN_PROVIDER_REQUEST_SCHEMA else payload
        _require(type(graph) is dict and graph.get("schema") == PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA,
                 "planner_request_schema")
        constraints = graph.get("constraints")
        _require(type(constraints) is dict and type(constraints.get("constraint_summaries")) is list,
                 "planner_summary_slot_missing")
        summaries = [row for row in constraints["constraint_summaries"]
                     if type(row) is dict and row.get("schema") == SUMMARY_SCHEMA]
        all_summaries = _all_numerical_summaries(payload)
        result.update(request_checked=True, planner_request_sha256=_sha(raw),
                      planner_request_bytes=len(raw), numerical_summary_count=len(all_summaries))
        _require(len(summaries) == len(all_summaries), "numerical_summary_outside_constraint_slot")

        planning = report.get("planning")
        _require(type(planning) is dict, "planning_result_missing", unavailable=True)
        receipt = planning.get("execution_receipt")
        _require(type(receipt) is dict, "planning_execution_receipt_missing", unavailable=True)
        _require(all(_digest(value) for value in (planning.get("model_request_sha256"),
                 receipt.get("prompt_sha256"), receipt.get("native_prompt_sha256"),
                 receipt.get("model_prompt_sha256"))), "planning_prompt_hashes_missing", unavailable=True)
        _require(planning["model_request_sha256"] == receipt["prompt_sha256"] ==
                 receipt["native_prompt_sha256"] == _sha(raw)
                 and type(receipt.get("prompt_bytes")) is int and receipt["prompt_bytes"] == len(raw)
                 and type(receipt.get("native_prompt_bytes")) is int and receipt["native_prompt_bytes"] == len(raw),
                 "planning_request_hash_mismatch")
        result.update(request_hash_join_verified=True,
                      model_prompt_sha256=receipt["model_prompt_sha256"])
        _require(type(receipt.get("model_prompt_bytes")) is int and
                 0 < receipt["model_prompt_bytes"] <= 256_000, "model_input_size_missing", unavailable=True)
        _require(planning.get("qualified") is True and type(planning.get("provider_calls")) is int
                 and planning["provider_calls"] == 1 and receipt.get("schema") == "router-implementation-invocation@1"
                 and receipt.get("purpose") == "planning" and receipt.get("status") == "provider_returned"
                 and type(receipt.get("router_calls")) is int and receipt["router_calls"] == 1
                 and receipt.get("completion_authority") is False, "successful_planner_dispatch_not_observed", unavailable=True)
        selected = report.get("provider_profile")
        _require(type(selected) is dict and all(receipt.get(key) == planning.get(key) == selected.get(key)
                 and type(receipt.get(key)) is str and receipt[key] for key in ("provider", "model", "reasoning_effort")),
                 "planning_provider_selection_mismatch")
        identity = receipt.get("invocation_id")
        _require(type(identity) is str and re.fullmatch(r"[0-9a-f]{32}", identity) is not None,
                 "planning_invocation_identity_missing", unavailable=True)
        invocations = report.get("provider_invocations")
        _require(type(invocations) is list and len(invocations) <= 256, "planning_invocations_missing", unavailable=True)
        observed = [row for row in invocations if type(row) is dict and row.get("phase") == "planning"]
        _require(bool(observed), "planning_invocation_unobserved", unavailable=True)
        _require(all(_wire({k: v for k, v in row.items() if k != "phase"}) == _wire(receipt)
                     for row in observed), "planning_invocation_receipts_differ")
        result.update(planning_invocation_join_verified=True, planning_invocation_id=identity,
                      planning_execution_receipt_sha256=_sha(_wire(receipt)))

        initial = report.get("initial_context") or {}
        _require(type(initial) is dict, "initial_context_shape")
        numerical = initial.get("numerical_formalization")
        numerical_path = state / "numerical-formalization" / "receipt.json"
        selection_path = state / "numerical-formalization-selection.json"
        receipt_raw = None
        if arm == "full":
            _require(type(numerical) is dict and type(numerical.get("summary")) is dict,
                     "numerical_receipt_missing", unavailable=True)
            summary = numerical["summary"]
            _require(numerical.get("profile") == profile and summary.get("profile") == profile
                     and summary.get("schema") == SUMMARY_SCHEMA and len(_wire(summary)) <= 8192,
                     "numerical_summary_profile_mismatch")
            _require(type(summary.get("source_tree_id")) is str
                     and 0 < len(summary["source_tree_id"].encode()) <= 256,
                     "numerical_source_identity_missing")
            _require(all(summary.get(key) is False for key in ("proof_authority", "completion_authority",
                     "execution_authority", "source_semantics_verified", "whole_program_semantics_verified"))
                     and type(summary.get("source_obligations_discharged")) is int
                     and summary["source_obligations_discharged"] == 0, "numerical_summary_claims_authority")
            _require(len(summaries) == 1 and _wire(summaries[0]) == _wire(summary), "numerical_summary_missing_or_different")
            receipt_raw = _read(numerical_path, 1_048_576)
            _require(_digest(numerical.get("receipt_sha256")) and _sha(receipt_raw) == numerical["receipt_sha256"]
                     and _wire(_json(receipt_raw)) == _wire({k: v for k, v in numerical.items() if k != "receipt_sha256"}),
                     "numerical_receipt_file_mismatch")
            _require(_json(_read(selection_path, 4096)) == {"profile": profile}, "numerical_selection_mismatch")
            indexed = planning.get("initial_indexed_context")
            _require(type(indexed) is dict and indexed.get("supplied_to_router") is True,
                     "initial_index_delivery_missing", unavailable=True)
            result.update(exact_summary_match=True, numerical_summary_sha256=_sha(_wire(summary)),
                numerical_receipt_sha256=numerical["receipt_sha256"],
                numerical_source_tree_id=summary.get("source_tree_id"))
        else:
            _require(not summaries and numerical is None and planning.get("initial_indexed_context") is None
                     and not numerical_path.exists() and not numerical_path.is_symlink()
                     and not selection_path.exists() and not selection_path.is_symlink(),
                     "no_index_numerical_context_present")
        _require(_read(state / "planner-provider-request.json", MAX_REQUEST_BYTES) == raw,
                 "planner_request_changed_during_observation")
        if receipt_raw is not None:
            _require(_read(numerical_path, 1_048_576) == receipt_raw, "numerical_receipt_changed_during_observation")
        result.update(status="verified", reason="exact_delivery_join" if arm == "full" else "checked_no_index_absence",
            live_planner_delivery_verified=arm == "full", no_index_absence_verified=arm == "no-index")
    except _Refusal as error:
        result.update(status="unavailable" if error.unavailable else "mismatch", reason=error.reason)
    except (OSError, ValueError, TypeError, KeyError, RecursionError, UnicodeError):
        result.update(status="unavailable", reason="bounded_evidence_unreadable")
    return result


__all__ = ["SCHEMA", "observe_numerical_planning"]
