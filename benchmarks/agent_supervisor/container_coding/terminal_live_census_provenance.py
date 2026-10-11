"""Join live census planning receipts to independently observed worker custody.

This is a pure metadata validator. The trusted launcher must independently read
and protect the current deployment observation; this module cannot authenticate
caller-supplied mappings or replace native admission, source, or worker checks.
No provider text, files, databases, or token-count event streams are inspected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re


SCHEMA = "terminal-live-census-planning-provenance@1"
USAGE_SCHEMA = "terminal-live-census-planning-usage@1"
TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                "output_tokens", "reasoning_output_tokens", "total_tokens")
_ROUTER_FIELDS = frozenset((
    "schema", "invocation_id", "provider", "model", "allocation_catalog", "reasoning_effort", "purpose",
    "prompt_sha256", "prompt_bytes", "native_prompt_sha256", "native_prompt_bytes",
    "model_prompt_sha256", "model_prompt_bytes", "workspace_advisory_sha256", "workspace_advisory_bytes",
    "workspace", "router_prompt_sha256", "router_prompt_bytes", "semantic_translation",
    "doctor_residual_context", "public_instruction", "timeout_seconds", "requested_output_tokens",
    "provider_output_token_cap_enforced", "router_calls", "usage", "completion_authority",
    "external_container_boundary", "provider_invocation_policy", "status", "native_rollout_usage",
    "native_provider_outcome", "seconds", "error_type", "failure_phase", "provider_failure",
    "semantic_response_failure",
))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _raw(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise ValueError("bounded JSON metadata required") from None
    _require(len(raw) <= 1_048_576, "metadata exceeds its byte bound")
    return raw


def _digest(value):
    return hashlib.sha256(_raw(value)).hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _count(value):
    return type(value) is int and value >= 0


def _invocations(invocations):
    _require(type(invocations) is list and len(invocations) <= 256, "bounded planning receipt list required")
    unique = {}
    for decorated in invocations:
        _require(type(decorated) is dict, "planning router receipt required")
        row = {key: value for key, value in decorated.items() if key != "phase"}
        _raw(row)
        _require(set(row) <= _ROUTER_FIELDS and row.get("schema") == "router-implementation-invocation@1"
                 and row.get("purpose") == "planning" and decorated.get("phase", "planning") == "planning",
                 "closed planning router invocation required")
        identity = row.get("invocation_id")
        _require(type(identity) is str and re.fullmatch(r"[0-9a-f]{32}", identity) is not None,
                 "exact planning invocation identity required")
        if identity in unique:
            _require(unique[identity] == row, "conflicting planning invocation receipts")
        unique[identity] = row
    return list(unique.values())


def audit_planning_usage(planning, invocations):
    """Preserve final native categories once per invocation, including failures.

    Native Codex receipts already contain the latest cumulative token_count;
    their event count is descriptive and is never multiplied into usage.
    Missing or incomplete categories remain null with separate known subtotals.
    """
    _require(type(planning) is dict, "planning metadata required")
    rows = _invocations(invocations)
    attempted = planning.get("provider_calls")
    _require(attempted is None or _count(attempted), "invalid planner callable attempt count")
    _require(attempted is None or attempted >= len(rows), "receipt count exceeds planner callable attempts")
    sessions, seen_threads = [], set()
    for row in rows:
        native = row.get("native_rollout_usage")
        _require(native is None or type(native) is dict, "invalid native usage receipt")
        native = native or {}
        provider = row.get("provider")
        _require(provider in {"codex_cli", "grok_cli"}, "unsupported native usage provider")
        expected = "native-codex-cumulative-usage@1" if provider == "codex_cli" else "native-grok-final-usage@1"
        _require(not native or native.get("schema") == expected, "native usage provider or schema differs")
        usage = native.get("usage", {})
        _require(type(usage) is dict and set(usage) <= set(TOKEN_FIELDS)
                 and all(_count(value) for value in usage.values()), "invalid native token categories")
        _require(native.get("usage_available") is not True or bool(usage), "available native usage has no categories")
        thread = native.get("thread_id")
        if thread is not None:
            _require(type(thread) is str and 16 <= len(thread) <= 80, "invalid native thread identity")
            _require(thread not in seen_threads, "multiple invocations reuse one cumulative native thread")
            seen_threads.add(thread)
        complete = (native.get("usage_available") is True and native.get("task_complete_observed") is True
                    and (provider != "grok_cli" or native.get("usage_complete_observed") is True))
        observed = row.get("usage") or {}
        _require(type(observed) is dict, "invalid observed CLI metadata")
        cost = observed.get("total_cost_usd")
        _require(cost is None or type(cost) in (int, float) and cost >= 0 and cost < float("inf"),
                 "invalid reported provider cost")
        sessions.append({"usage": usage, "complete": complete, "cost": cost,
                         "cache_included_in_input": native.get("cache_included_in_input"),
                         "router_calls": row.get("router_calls")})
    missing = None if attempted is None else attempted - len(rows)
    all_receipted = attempted is not None and missing == 0
    complete = bool(rows) and all_receipted and all(row["complete"] for row in sessions)
    totals, subtotals = {}, {}
    for key in TOKEN_FIELDS:
        known = [row["usage"][key] for row in sessions if key in row["usage"]]
        subtotals[key] = sum(known) if known else None
        totals[key] = sum(known) if complete and len(known) == len(sessions) else None
    costs = [row["cost"] for row in sessions if row["cost"] is not None]
    caches = [row["cache_included_in_input"] for row in sessions]
    return {"schema": USAGE_SCHEMA, "observation_only": True,
        "planner_callable_attempts": attempted, "emitted_router_receipts": len(invocations),
        "unique_router_invocations": len(rows), "unreceipted_callable_attempts": missing,
        "observed_router_dispatches": (sum(row["router_calls"] for row in sessions)
            if all(_count(row["router_calls"]) for row in sessions) else None),
        "all_callable_attempts_receipted": all_receipted,
        "observed_complete_sessions": complete,
        "all_six_token_categories_complete": all(value is not None for value in totals.values()),
        "tokens": totals, "known_token_subtotals": subtotals,
        "native_token_count_events_are_call_counts": False,
        "cumulative_records_summed": False, "native_totals_counted_once_per_invocation": True,
        "cache_included_in_input": (caches[0] if caches and type(caches[0]) is bool
            and all(value is caches[0] for value in caches) else None),
        "reasoning_output_added_to_output": False, "cached_input_added_to_input": False,
        "reported_cost_usd": sum(costs) if complete and len(costs) == len(sessions) else None,
        "known_reported_cost_subtotal_usd": sum(costs) if costs else None,
        "billing_total_verified": False, "completion_authority": False}


def _worker_boundary(deployment, invocation, provider):
    _require(type(deployment) is dict and deployment.get("schema") == "container-worker-deployment@1"
             and deployment.get("qualified") is True and type(deployment.get("provider_calls")) is int
             and deployment["provider_calls"] == 0,
             "independent qualified worker deployment observation required")
    _raw(deployment)
    boundary, qualification = deployment.get("boundary"), deployment.get("qualification")
    _require(type(boundary) is dict and type(qualification) is dict,
             "structured deployment and qualification observations required")
    raw = (json.dumps(boundary, sort_keys=True, indent=2) + "\n").encode()
    manifest_sha = deployment.get("boundary_sha256")
    _require(_sha(manifest_sha) and hashlib.sha256(raw).hexdigest() == manifest_sha,
             "deployment boundary manifest digest differs")
    namespaces = boundary.get("namespaces")
    owner, worker = boundary.get("owner_uid"), boundary.get("worker_uid")
    _require(boundary.get("schema") == "supervisor-container-worker-boundary@1"
             and boundary.get("provider") == provider and _sha(boundary.get("container_id"))
             and type(boundary.get("image_id")) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", boundary["image_id"])
             and type(owner) is int and type(worker) is int and owner > 0 and worker > 0 and owner != worker
             and type(namespaces) is dict and set(namespaces) == {"pid", "mnt", "net"}
             and all(type(value) is str and re.fullmatch(key + r":\[[0-9]+\]", value)
                     for key, value in namespaces.items()), "exact deployed container identity required")
    expected = {key: boundary[key] for key in
                ("schema", "container_id", "image_id", "namespaces", "owner_uid", "worker_uid")}
    expected.update(purpose="coding", manifest_sha256=manifest_sha, completion_authority=False)
    _require(invocation.get("external_container_boundary") == expected,
             "planning invocation differs from independently deployed worker boundary")
    evidence = qualification.get("permission_evidence")
    _require(qualification.get("qualified") is True and qualification.get("provider_calls") == 0
             and type(qualification.get("provider_calls")) is int
             and qualification.get("completion_authority") is False
             and qualification.get("remaining_worker_pids") == [] and type(evidence) is dict
             and evidence.get("schema") == "container-worker-preflight@1"
             and evidence.get("uid") == worker and evidence.get("provider_calls") == 0
             and type(evidence.get("uid")) is int and type(evidence.get("provider_calls")) is int
             and evidence.get("candidate_write") is True and evidence.get("boundary") == expected,
             "independent native worker permission and termination qualification required")
    root = deployment.get("worker_worktree_root")
    _require(type(root) is str and boundary.get("allowed_worktree_roots") == [root],
             "exact deployed worktree scope required")
    worktrees = PurePosixPath(root)
    _require(worktrees.is_absolute() and ".." not in worktrees.parts, "invalid deployed worktree scope")
    for workspace in (invocation.get("workspace"), evidence.get("workspace")):
        _require(type(workspace) is str and PurePosixPath(workspace) != worktrees
                 and PurePosixPath(workspace).is_relative_to(worktrees) and ".." not in PurePosixPath(workspace).parts,
                 "planning or qualification workspace escapes deployed scope")
    state = worktrees.parent / "state"
    denials = {str(state / name) + ":" + mode for name in ("preflight-key", "preflight-owner.duckdb")
               for mode in ("rb", "ab")} | {".git:write", ".git:unlink"}
    _require(type(evidence.get("private_denials")) is list and len(evidence["private_denials"]) == 6
             and set(evidence["private_denials"]) == denials, "native private-authority denials differ")
    return expected


def validate_live_planning_provenance(planning, invocations, selected_provider, *, deployment_boundary):
    """Validate retained joins; the caller authenticates the deployment input.

    Exactly one unique successful planning invocation is required. Identical
    repeated observations are deduplicated; failed attempts remain available
    through audit_planning_usage and cannot qualify this success gate.
    """
    from .benchmark_provider_profile import resolve_provider_profile

    _require(type(planning) is dict and type(selected_provider) is dict,
             "planning metadata and resolved provider pin required")
    _raw(planning)
    _require(selected_provider == resolve_provider_profile(selected_provider.get("id")),
             "selected provider differs from immutable benchmark profile")
    rows = _invocations(invocations)
    _require(len(rows) == 1, "exactly one unique planning invocation required")
    row = rows[0]
    _require(planning.get("schema") == "terminal-indexed-planner-result@1"
             and planning.get("qualified") is True and planning.get("provider_calls") == 1
             and type(planning.get("provider_calls")) is int
             and planning.get("planning_strategy") in {"direct", "intent_coverage"}
             and row.get("status") == "provider_returned" and row.get("router_calls") == 1
             and type(row.get("router_calls")) is int
             and row.get("completion_authority") is False
             and planning.get("execution_receipt") == row,
             "actual successful planning execution receipt required")
    provider_receipt = planning.get("provider_receipt") or {}
    native_provider = provider_receipt.get("provider") or {}
    parsed = provider_receipt.get("parse") or {}
    _require(provider_receipt.get("outcome") == "provider"
             and provider_receipt.get("fallback", {}).get("used") is False
             and native_provider.get("attempted") is True and native_provider.get("status") == "succeeded"
             and parsed.get("attempted") is True and parsed.get("status") == "succeeded"
             and parsed.get("plan_root_cid") == provider_receipt.get("plan_root_cid")
             and type(provider_receipt.get("plan_root_cid")) is str and bool(provider_receipt["plan_root_cid"]),
             "strict provider proposal and parser provenance required")
    for key in ("provider", "model", "reasoning_effort"):
        _require(row.get(key) == planning.get(key) == selected_provider[key], "selected planning route differs")
    _require(native_provider.get("provider_id") == selected_provider["provider"]
             and native_provider.get("model_id") == selected_provider["model"], "native planner route differs")
    prompt_sha = planning.get("model_request_sha256")
    _require(_sha(prompt_sha) and row.get("prompt_sha256") == row.get("native_prompt_sha256") == prompt_sha
             and _count(row.get("prompt_bytes")) and row["prompt_bytes"] > 0
             and row["prompt_bytes"] <= 256_000
             and row.get("native_prompt_bytes") == row["prompt_bytes"]
             and _sha(row.get("model_prompt_sha256")) and _count(row.get("model_prompt_bytes"))
             and 0 < row["model_prompt_bytes"] <= 256_000,
             "actual planning stdin and complete model input binding required")
    boundary = _worker_boundary(deployment_boundary, row, selected_provider["provider"])
    structured = (row.get("provider_invocation_policy") or {}).get("structured_output") or {}
    projection = structured.get("native_schema_projection") or {}
    observation = row.get("usage") or {}
    _require(selected_provider["provider"] == "codex_cli", "observed native planning schema custody unavailable for this provider")
    _require(structured.get("schema") == "codex-native-planning-json-schema@1"
             and structured.get("native_schema_requested") is True
             and structured.get("response_schema_validated") is True
             and projection.get("projection_id") == "canonical-prompt-goal-codex-generation-schema@1"
             and projection.get("canonical_validation_preserved") is True
             and all(projection.get(key) is True for key in
                     ("top_level_id_omitted", "bounds_annotation_omitted", "definitions_relocated", "shape_only"))
             and _sha(projection.get("canonical_schema_sha256"))
             and _sha(projection.get("native_wire_schema_sha256"))
             and _count(projection.get("canonical_schema_bytes")) and projection["canonical_schema_bytes"] > 0
             and _count(projection.get("native_wire_schema_bytes")) and projection["native_wire_schema_bytes"] > 0
             and observation.get("exit_code") == 0 and type(observation.get("exit_code")) is int
             and observation.get("codex_output_schema_sha256") == projection["native_wire_schema_sha256"]
             and observation.get("codex_output_schema_bytes") == projection["native_wire_schema_bytes"],
             "requested and observed native planning schema custody differs")
    if observation.get("model_id") is not None:
        _require(observation["model_id"] == selected_provider["model"], "observed CLI model differs")
    native = row.get("native_rollout_usage")
    if native is not None:
        _require(type(native) is dict and native.get("schema") == "native-codex-cumulative-usage@1"
                 and native.get("thread_id") == (observation.get("thread_id") or observation.get("session_id"))
                 and _sha(native.get("rollout_sha256")) and _count(native.get("observed_token_count_records"))
                 and native.get("cache_included_in_input") is True
                 and native.get("billing_total_verified") is False,
                 "native cumulative usage custody differs from observed CLI thread")
    usage = audit_planning_usage(planning, invocations)
    return {"schema": SCHEMA, "observation_only": True, "provider": selected_provider["provider"],
        "model": selected_provider["model"], "reasoning_effort": selected_provider["reasoning_effort"],
        "cli_version_pin": selected_provider["cli_version"], "profile_id": selected_provider["id"],
        "planning_execution_receipt_sha256": _digest(row), "planning_metadata_sha256": _digest(planning),
        "deployment_observation_sha256": _digest(deployment_boundary),
        "worker_boundary_manifest_sha256": boundary["manifest_sha256"],
        "native_planning_prompt_sha256": prompt_sha, "native_planning_prompt_bytes": row["prompt_bytes"],
        "complete_model_prompt_sha256": row["model_prompt_sha256"],
        "complete_model_prompt_bytes": row["model_prompt_bytes"],
        "canonical_planning_schema_sha256": projection["canonical_schema_sha256"],
        "canonical_planning_schema_bytes": projection["canonical_schema_bytes"],
        "native_generation_schema_sha256": projection["native_wire_schema_sha256"],
        "native_generation_schema_bytes": projection["native_wire_schema_bytes"],
        "actual_native_schema_observation_matched": True, "external_worker_custody_join_verified": True,
        "deployment_observation_authentication_performed_here": False,
        "admission_verified_here": False, "source_freshness_verified_here": False,
        "raw_provider_data_exported": False, "usage": usage,
        "proof_authority": False, "execution_authority": False, "completion_authority": False,
        "publication_authority": False, "settlement_authority": False}
