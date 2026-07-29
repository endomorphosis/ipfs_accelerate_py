"""CLI bridge for reusable objective-graph backlog generation.

This package-level entry point scans an objective heap, appends missing-evidence
tasks, persists AST records, writes parallel bundle shards, and optionally
submits those bundles to a local task queue.
"""

from __future__ import annotations

import argparse
from hashlib import sha1, sha256
import json
import logging
import os
import re
import shlex
import tempfile
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from .goal_completion import CompletionEvidence
from ..core.external_completion import (
    ExternalCompletionAuthority,
    load_external_completion_authority,
)
from .objective_graph import (
    DEFAULT_DISCOVERY_OUTPUT_PATH,
    DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    DEFAULT_TASK_PREFIX,
    ObjectiveFinding,
    ObjectiveWorkKind,
    ObjectiveWorkProposal,
    external_authority_goal_fence,
    generate_objective_todos,
    parse_goal_heap,
    repo_relative_path,
    resolve_scan_exclude_paths,
    scan_exclude_path_metadata,
    scan_objective_gaps,
    source_protected_scan_policy,
    submit_bundle_tasks,
)
from .objective_tracker import (
    DEFAULT_GOAL_PREFIX,
    DEFAULT_ROOT_GOAL_TITLE,
    DEFAULT_TRACKING_DOCUMENT_TITLE,
    DEFAULT_ULTIMATE_GOAL,
    append_interoperability_goals,
    append_launch_readiness_goals,
    append_refinement_goals,
    completion_tree_identity,
    deduplicate_interoperability_goals,
    directly_open_goal_ids_from_todo_board,
    ensure_objective_tracking_document,
    open_goal_ids_from_todo_board,
    parse_root_evidence,
    reconcile_objective_goal_completion,
    write_objective_graph_artifact,
)

logger = logging.getLogger(__name__)

OBJECTIVE_COMPLETION_GATE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.objective_daemon.completion_gate.v1"
)
OBJECTIVE_COMPLETION_EVIDENCE_ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.objective_daemon.completion_evidence.v1"
)
OBJECTIVE_GENERATION_ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.objective_daemon.generation.v1"
)
OBJECTIVE_ADMISSION_RECORD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/objective-admission-record@1"
)


def _plan_value_dict(value: Any) -> dict[str, Any]:
    """Return a JSON-safe mapping for a structured planning value."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return {str(key): item for key, item in payload.items()}
    if is_dataclass(value):
        payload = asdict(value)
        if isinstance(payload, dict):
            return payload
    raise TypeError(f"structured plan value must be mapping-like, got {type(value).__name__}")


def objective_record_plan_context(record: Any) -> dict[str, Any]:
    """Build the schema router input for one generated objective subgoal."""

    finding = record.finding
    validation = str(getattr(finding, "validation", "") or "").strip()
    validation_commands = [item.strip() for item in validation.split(";") if item.strip()]
    predicted_files = list(
        dict.fromkeys(
            str(item)
            for item in [
                *(getattr(finding, "predicted_files", ()) or ()),
                *(getattr(finding, "outputs", ()) or ()),
            ]
            if str(item).strip()
        )
    )
    predicted_symbols = list(
        dict.fromkeys(
            str(item)
            for item in (getattr(finding, "ast_symbols", ()) or ())
            if str(item).strip()
        )
    )
    return {
        "task_id": str(record.task_id),
        "goal_id": str(getattr(finding, "goal_id", "") or ""),
        "title": str(getattr(finding, "title", "") or ""),
        "summary": str(getattr(finding, "summary", "") or ""),
        "goal": str(getattr(finding, "goal", "") or getattr(finding, "title", "") or ""),
        "priority": str(getattr(finding, "priority", "") or ""),
        "track": str(getattr(finding, "track", "") or ""),
        "missing_evidence": [str(item) for item in getattr(finding, "missing_evidence", ()) or ()],
        "predicted_files": predicted_files,
        "predicted_symbols": predicted_symbols,
        "dependencies": [str(item) for item in getattr(finding, "parent_goal_ids", ()) or ()],
        "validation_commands": validation_commands,
        "validation_proof": validation_commands,
    }


def _evaluated_branch_dict(value: Any, evaluation: Any = None) -> dict[str, Any]:
    branch = getattr(value, "branch", None)
    if branch is None and isinstance(value, Mapping):
        branch = value.get("branch", value)
    if branch is None:
        # PlanEvaluation exposes selected/rejected PlanBranch values while
        # some consumers wrap them as EvaluatedPlanBranch. Support both
        # representations at this daemon boundary.
        branch = value
    branch_payload = _plan_value_dict(branch)
    branch_id = str(branch_payload.get("branch_id") or "")
    scores = getattr(evaluation, "scores", {}) if evaluation is not None else {}
    rationales = getattr(evaluation, "rationales", {}) if evaluation is not None else {}
    wrapped_score = (
        getattr(value, "score_millionths", None)
        if not isinstance(value, Mapping)
        else value.get("score_millionths")
    )
    wrapped_rationale = (
        getattr(value, "rationale", None)
        if not isinstance(value, Mapping)
        else value.get("rationale")
    )
    payload = {
        "branch": branch_payload,
        "score_millionths": int(wrapped_score if wrapped_score is not None else scores.get(branch_id, 0)),
        "rationale": list(
            wrapped_rationale if wrapped_rationale is not None else rationales.get(branch_id, ()) or ()
        ),
    }
    return payload


def plan_objective_records(
    records: Sequence[Any],
    *,
    branch_count: int = 3,
    router: Callable[..., Any] | None = None,
    fallback_planner: Callable[..., Any] | None = None,
    router_config: Any = None,
    evaluator: Callable[[Sequence[Any]], Any] | None = None,
    use_llm_router: bool = True,
) -> list[dict[str, Any]]:
    """Generate and deterministically select plan branches per ready record.

    Router failures are isolated to their record and use the deterministic
    planner.  Consequently a failed provider cannot prevent later ready work
    from receiving a selected implementation plan.
    """

    from ..planning.plan_evaluator import evaluate_plan_branches
    from ..planning.task_proposal_router import deterministic_plan_branches, generate_structured_plan_branches

    evaluate = evaluator or evaluate_plan_branches
    count = max(1, int(branch_count))
    decisions: list[dict[str, Any]] = []
    for record in records:
        context = objective_record_plan_context(record)
        route_error = ""
        used_fallback = False
        try:
            if use_llm_router:
                routed = generate_structured_plan_branches(
                    context,
                    router=router,
                    fallback_planner=fallback_planner,
                    config=router_config,
                    branch_count=count,
                )
                branches = tuple(routed.branches)
                used_fallback = bool(routed.used_fallback)
                route_error = str(routed.router_error or "")
            else:
                planner = fallback_planner or deterministic_plan_branches
                branches = tuple(planner(context, 1))
            evaluation = evaluate(branches)
            selected = _evaluated_branch_dict(evaluation.selected, evaluation)
            rejected = [_evaluated_branch_dict(item, evaluation) for item in evaluation.rejected]
            decisions.append(
                {
                    "task_id": context["task_id"],
                    "goal_id": context["goal_id"],
                    "source": (
                        "deterministic_fallback"
                        if used_fallback
                        else ("llm_router" if use_llm_router else "deterministic_planner")
                    ),
                    "used_fallback": used_fallback,
                    "analysis_inconclusive": bool(used_fallback or route_error),
                    "router_error": route_error or None,
                    "evaluator_version": str(evaluation.evaluator_version),
                    "selected": selected,
                    "rejected": rejected,
                    "selection_rationale": selected["rationale"],
                }
            )
        except Exception as exc:  # one provider/record must never stall later ready work
            logger.warning("Plan routing failed for %s; using deterministic fallback: %s", context["task_id"], exc)
            try:
                branches = tuple(deterministic_plan_branches(context, branch_count=1))
                evaluation = evaluate(branches)
                selected = _evaluated_branch_dict(evaluation.selected, evaluation)
                decisions.append(
                    {
                        "task_id": context["task_id"],
                        "goal_id": context["goal_id"],
                        "source": "deterministic_fallback",
                        "used_fallback": True,
                        "analysis_inconclusive": True,
                        "router_error": str(exc),
                        "evaluator_version": str(evaluation.evaluator_version),
                        "selected": selected,
                        "rejected": [_evaluated_branch_dict(item, evaluation) for item in evaluation.rejected],
                        "selection_rationale": selected["rationale"],
                    }
                )
            except Exception as fallback_exc:
                # Preserve the failure as evidence and continue planning the
                # remaining records. A malformed injected fallback must not
                # turn one subgoal into a daemon-wide outage.
                logger.error("Deterministic plan fallback failed for %s: %s", context["task_id"], fallback_exc)
                decisions.append(
                    {
                        "task_id": context["task_id"],
                        "goal_id": context["goal_id"],
                        "source": "planning_error",
                        "used_fallback": True,
                        "analysis_inconclusive": True,
                        "router_error": str(exc),
                        "fallback_error": str(fallback_exc),
                        "evaluator_version": "",
                        "selected": None,
                        "rejected": [],
                        "selection_rationale": ["deterministic fallback could not produce a valid branch"],
                    }
                )
    return decisions


def persist_objective_plan_evaluations(
    path: Path,
    decisions: Sequence[Mapping[str, Any]],
    *,
    bundle_index_path: Path | None = None,
) -> None:
    """Persist decisions and project them into scheduler-visible bundle tasks."""

    retained: dict[str, dict[str, Any]] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous_items = previous.get("evaluations", []) if isinstance(previous, Mapping) else []
            if isinstance(previous_items, list):
                retained.update(
                    {
                        str(item.get("task_id") or ""): dict(item)
                        for item in previous_items
                        if isinstance(item, Mapping) and str(item.get("task_id") or "")
                    }
                )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not retain prior plan decisions from %s: %s", path, exc)
    retained.update(
        {
            str(item.get("task_id") or ""): dict(item)
            for item in decisions
            if str(item.get("task_id") or "")
        }
    )
    ordered = [retained[task_id] for task_id in sorted(retained)]
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.objective_plan_evaluations@1",
        "evaluation_count": len(ordered),
        "evaluations": ordered,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if bundle_index_path is None or not bundle_index_path.exists():
        return
    try:
        bundle_payload = json.loads(bundle_index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not project plan decisions into %s: %s", bundle_index_path, exc)
        return
    if not isinstance(bundle_payload, dict):
        return
    by_task = {str(item.get("task_id") or ""): item for item in ordered}
    bundles = bundle_payload.get("bundles")
    if isinstance(bundles, Mapping):
        for info in bundles.values():
            if not isinstance(info, dict) or not isinstance(info.get("tasks"), list):
                continue
            for task in info["tasks"]:
                if not isinstance(task, dict):
                    continue
                decision = by_task.get(str(task.get("task_id") or ""))
                if decision is None:
                    continue
                task["plan_evaluation"] = decision
                selected = decision.get("selected")
                task["selected_plan_evaluation"] = selected
                task["selected_plan_branch"] = (
                    selected.get("branch") if isinstance(selected, Mapping) else None
                )
                task["rejected_plan_branches"] = decision.get("rejected", [])
                task["plan_selection_rationale"] = decision.get("selection_rationale", [])
    bundle_payload["plan_evaluation_path"] = str(path)
    bundle_payload["plan_evaluation_count"] = len(ordered)
    from ..runtime.artifact_store import write_bundle_index_artifact

    write_bundle_index_artifact(bundle_index_path, bundle_payload)


def objective_terms_for_analysis(
    objective_path: Path,
    records: Sequence[Any] = (),
) -> tuple[str, ...]:
    """Return deterministic uncovered terms for goal-directed escalation."""

    from .objective_graph import parse_goal_heap

    terms: list[str] = []
    if objective_path.exists():
        for goal in parse_goal_heap(objective_path.read_text(encoding="utf-8", errors="replace")):
            # Provisional and inconclusive goals remain schedulable so their
            # unproven criteria can feed bounded evidence generation.
            if goal.lifecycle_state_value in {
                "active", "reopened", "provisionally_complete", "analysis_inconclusive"
            }:
                terms.extend(goal.required_evidence)
    for record in records:
        finding = getattr(record, "finding", None)
        terms.extend(str(item) for item in getattr(finding, "missing_evidence", ()) or ())
    return tuple(dict.fromkeys(" ".join(item.strip().split()) for item in terms if item.strip()))


def persist_analysis_escalation(path: Path, result: Any) -> dict[str, Any]:
    """Persist one complete escalation artifact for daemon/status consumers."""

    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_objective_generation_work(path: Path | None) -> tuple[dict[str, Any], ...]:
    """Load durable generated work used to deduplicate later daemon cycles.

    A malformed artifact fails closed instead of silently forgetting identity
    history and regenerating equivalent tasks.  Legacy artifacts which stored
    work under ``accepted`` are accepted to keep upgrades idempotent.
    """

    if path is None or not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("objective generation artifact must be a JSON object")
    schema = str(payload.get("schema") or "")
    if schema and schema != OBJECTIVE_GENERATION_ARTIFACT_SCHEMA:
        raise ValueError(f"unsupported objective generation artifact schema: {schema}")
    raw = payload.get("generated_work", payload.get("accepted", ()))
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise ValueError("objective generation artifact work must be an array of objects")
    from .objective_graph import ObjectiveWorkProposal

    # Revalidate canonical identity when loading untrusted persisted state.
    validated = [ObjectiveWorkProposal.from_dict(item).to_dict() for item in raw]
    by_id: dict[str, dict[str, Any]] = {}
    for item in validated:
        canonical_id = str(item["canonical_id"])
        prior = by_id.get(canonical_id)
        if prior is not None and prior != item:
            raise ValueError(f"conflicting generated work identity {canonical_id}")
        by_id[canonical_id] = item
    return tuple(by_id[key] for key in sorted(by_id))


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON artifact without exposing partial bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _load_generation_payload(path: Path) -> dict[str, Any]:
    """Load the complete generation artifact while validating its identity ledger."""

    if not path.exists():
        return {
            "schema": OBJECTIVE_GENERATION_ARTIFACT_SCHEMA,
            "cycle_count": 0,
            "generated_work_count": 0,
            "generated_work": [],
            "admission_records": {},
            "gap_family_states": {},
        }
    # Validate every work identity before retaining any admission state.
    load_objective_generation_work(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("objective generation artifact must be a JSON object")
    result = dict(payload)
    records = result.get("admission_records", {})
    if not isinstance(records, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, Mapping)
        for key, value in records.items()
    ):
        raise ValueError("objective generation admission_records must be an object")
    result["admission_records"] = {
        str(key): dict(value) for key, value in sorted(records.items())
    }
    family_states = result.get("gap_family_states", {})
    if not isinstance(family_states, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, Mapping)
        for key, value in family_states.items()
    ):
        raise ValueError("objective generation gap_family_states must be an object")
    result["gap_family_states"] = {
        str(key): dict(value) for key, value in sorted(family_states.items())
    }
    return result


def load_objective_admission_records(
    path: Path | None,
) -> dict[str, dict[str, Any]]:
    """Return durable review/admission state keyed by canonical proposal ID."""

    if path is None or not path.exists():
        return {}
    payload = _load_generation_payload(path)
    return {
        str(key): dict(value)
        for key, value in (payload.get("admission_records") or {}).items()
    }


def _persist_objective_admission_records(
    path: Path,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge reviewable/admitted records without discarding generation history."""

    payload = _load_generation_payload(path)
    retained = {
        str(key): dict(value)
        for key, value in (payload.get("admission_records") or {}).items()
    }
    for value in records:
        canonical_id = str(value.get("canonical_id") or "").strip()
        if not canonical_id:
            raise ValueError("objective admission records require canonical_id")
        prior = retained.get(canonical_id)
        next_record = dict(value)
        if prior is not None:
            for field_name in ("preview", "tracker_transaction"):
                if (
                    next_record.get(field_name) is None
                    and prior.get(field_name) is not None
                ):
                    next_record[field_name] = prior[field_name]
            attempts = [
                *(
                    prior.get("attempts", ())
                    if isinstance(prior.get("attempts"), list)
                    else ()
                ),
                *(
                    next_record.get("attempts", ())
                    if isinstance(next_record.get("attempts"), list)
                    else ()
                ),
            ]
            if attempts:
                unique_attempts: dict[str, dict[str, Any]] = {}
                for attempt in attempts:
                    if not isinstance(attempt, Mapping):
                        continue
                    key = json.dumps(
                        dict(attempt), sort_keys=True, separators=(",", ":"), default=str
                    )
                    unique_attempts[key] = dict(attempt)
                next_record["attempts"] = [
                    unique_attempts[key] for key in sorted(unique_attempts)
                ]
        retained[canonical_id] = next_record
    payload["admission_records"] = {
        key: retained[key] for key in sorted(retained)
    }
    _atomic_json_write(path, payload)
    return payload


def persist_objective_generation(
    path: Path,
    result: Any,
    *,
    existing_work: Iterable[Mapping[str, Any]] = (),
    evaluation: Any = None,
    gap_family_states: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Persist a bounded generation cycle and its cross-cycle identity ledger."""

    cycle = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    raw_accepted = cycle.get("accepted", cycle.get("generated_work", ()))
    if not isinstance(raw_accepted, list):
        raise ValueError("objective generation result accepted work must be an array")
    from .objective_graph import ObjectiveWorkProposal

    merged: dict[str, dict[str, Any]] = {}
    for value in [*existing_work, *raw_accepted]:
        item = ObjectiveWorkProposal.from_dict(value).to_dict()
        canonical_id = str(item["canonical_id"])
        prior = merged.get(canonical_id)
        if prior is not None and prior != item:
            raise ValueError(f"conflicting generated work identity {canonical_id}")
        merged[canonical_id] = item
    prior_cycle_count = 0
    if path.exists():
        try:
            prior_payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(prior_payload, Mapping):
                prior_cycle_count = max(0, int(prior_payload.get("cycle_count", 0)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Loading the identity ledger is separately fail-closed.  This
            # best-effort counter is observability only and cannot admit work.
            prior_cycle_count = 0
    retained_admission_records: dict[str, Any] = {}
    retained_gap_family_states: dict[str, Any] = {}
    if path.exists():
        try:
            previous_payload = _load_generation_payload(path)
            retained_admission_records = dict(
                previous_payload.get("admission_records") or {}
            )
            retained_gap_family_states = dict(
                previous_payload.get("gap_family_states") or {}
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # The identity load above remains fail-closed.  This branch is only
            # reachable for observability fields in a legacy/corrupt artifact.
            retained_admission_records = {}
            retained_gap_family_states = {}
    if gap_family_states is not None:
        retained_gap_family_states = {
            str(key): dict(value)
            for key, value in gap_family_states.items()
        }
    payload = {
        "schema": OBJECTIVE_GENERATION_ARTIFACT_SCHEMA,
        "cycle_count": prior_cycle_count + 1,
        "generated_work_count": len(merged),
        "generated_work": [merged[key] for key in sorted(merged)],
        "last_cycle": cycle,
        "last_evaluation": (
            evaluation.to_dict() if hasattr(evaluation, "to_dict") else evaluation
        ),
        "admission_records": retained_admission_records,
        "gap_family_states": {
            key: retained_gap_family_states[key]
            for key in sorted(retained_gap_family_states)
        },
    }
    _atomic_json_write(path, payload)
    return payload


def materialize_objective_generation_cycle(
    proposals: Iterable[Any],
    *,
    artifact_path: Path,
    limits: Any = None,
    current_open_work: int | None = None,
    evaluation_policy: Any = None,
    objective_terms: Sequence[str] = (),
    active_family_keys: Iterable[str] | None = None,
    terminal_family_counts: Mapping[str, int] | None = None,
    blocked_family_counts: Mapping[str, int] | None = None,
    observed_gap_goal_ids: Iterable[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Apply finite graph limits and persist canonical history for one cycle.

    Typed completion gaps advance through a bounded repair lifecycle.  A
    completed board task authorizes one retry, up to ``max_retries``; persistent
    failure then produces one review task and finally a durable blocked-review
    state. A blocked task bypasses retries and produces the review directly.
    Board task counts, rather than receipt hashes, drive this lifecycle.
    """

    from .objective_graph import (
        ObjectiveGenerationLimits,
        ObjectiveWorkProposal,
        materialize_bounded_objective_work,
    )
    from ..planning.plan_evaluator import evaluate_objective_work_proposals

    existing = load_objective_generation_work(artifact_path)
    proposal_values = tuple(proposals)
    active_keys = (
        None
        if active_family_keys is None
        else {
            str(item).strip()
            for item in active_family_keys
            if str(item).strip()
        }
    )
    completed_counts = {
        str(key).strip(): max(0, int(value))
        for key, value in (terminal_family_counts or {}).items()
        if str(key).strip()
    }
    blocked_counts = {
        str(key).strip(): max(0, int(value))
        for key, value in (blocked_family_counts or {}).items()
        if str(key).strip()
    }
    observed_goal_ids = (
        None
        if observed_gap_goal_ids is None
        else {
            str(item).strip()
            for item in observed_gap_goal_ids
            if str(item).strip()
        }
    )
    if limits is None:
        max_retries = ObjectiveGenerationLimits().max_retries
    elif isinstance(limits, Mapping):
        max_retries = int(
            limits.get("max_retries", ObjectiveGenerationLimits().max_retries)
        )
    else:
        max_retries = int(
            getattr(limits, "max_retries", ObjectiveGenerationLimits().max_retries)
        )
    prior_payload = _load_generation_payload(artifact_path)
    prior_family_states = {
        str(key): dict(value)
        for key, value in (prior_payload.get("gap_family_states") or {}).items()
    }
    typed_candidates: dict[str, ObjectiveWorkProposal] = {}
    for proposal in proposal_values:
        family_key = str(getattr(proposal, "family_key", "") or "").strip()
        if not family_key and isinstance(proposal, Mapping):
            family_key = str(proposal.get("family_key") or "").strip()
        if not family_key:
            continue
        normalized = (
            proposal
            if isinstance(proposal, ObjectiveWorkProposal)
            else ObjectiveWorkProposal.from_dict(proposal)
        )
        prior = typed_candidates.get(family_key)
        if prior is not None:
            if prior.instance_key != normalized.instance_key:
                raise ValueError(
                    f"completion gap family {family_key} has conflicting instances"
                )
        typed_candidates[family_key] = normalized

    lifecycle_actions: dict[str, str] = {}
    prepared_typed: dict[str, ObjectiveWorkProposal] = {}

    def retry_proposal(
        proposal: ObjectiveWorkProposal,
        *,
        retry_ordinal: int,
        note: str,
    ) -> ObjectiveWorkProposal:
        return replace(
            proposal,
            title=f"Retry {retry_ordinal}: {proposal.title}",
            expected_evidence_delta=tuple(
                dict.fromkeys([*proposal.expected_evidence_delta, note])
            ),
            retry_count=retry_ordinal,
            source="completion_gate_gap_retry",
            source_id=f"{proposal.instance_key}:retry:{retry_ordinal}",
            rationale="; ".join(
                dict.fromkeys([proposal.rationale, note])
            ),
            canonical_id="",
        )

    def review_proposal(
        proposal: ObjectiveWorkProposal,
        *,
        retry_count: int,
        note: str,
    ) -> ObjectiveWorkProposal:
        return replace(
            proposal,
            title=f"Review persistent gap: {proposal.title}",
            expected_evidence_delta=tuple(
                dict.fromkeys([*proposal.expected_evidence_delta, note])
            ),
            retry_count=retry_count,
            source="completion_gate_gap_review",
            source_id=f"{proposal.instance_key}:review",
            rationale="; ".join(
                dict.fromkeys([proposal.rationale, note])
            ),
            canonical_id="",
        )

    for family_key, proposal in typed_candidates.items():
        prior = prior_family_states.get(family_key, {})
        family_unresolved = bool(prior) and prior.get("resolved") is not True
        same_instance = (
            family_unresolved
            and str(prior.get("instance_key") or "") == proposal.instance_key
        )
        completed_count = completed_counts.get(family_key, 0)
        blocked_count = blocked_counts.get(family_key, 0)
        prior_completed_count = max(
            0,
            int(
                prior.get(
                    "completed_task_count",
                    prior.get("terminal_task_count", 0),
                )
                or 0
            ),
        )
        prior_blocked_count = max(
            0, int(prior.get("blocked_task_count", 0) or 0)
        )
        attempt_count = max(1, int(prior.get("attempt_count", 1) or 1))
        review_emitted = bool(prior.get("review_emitted", False))
        action = "stable"
        prepared = proposal

        # A family already represented by active board work cannot admit a
        # second task, even when fresher diagnostics arrive mid-attempt.
        if active_keys is not None and family_key in active_keys:
            action = "active"
        elif (
            family_unresolved
            and str(prior.get("outcome") or "") == "blocked_review"
        ):
            action = "blocked_review"
        elif family_unresolved and blocked_count > prior_blocked_count:
            if review_emitted:
                action = "blocked_review"
            else:
                review_note = (
                    "A completion-evidence alignment task was blocked; "
                    "perform one manual review and record a durable block or "
                    "a concrete remediation. The block does not consume the "
                    "automated retry budget."
                )
                prepared = review_proposal(
                    proposal,
                    retry_count=max(0, attempt_count - 1),
                    note=review_note,
                )
                action = "review"
        elif family_unresolved and completed_count > prior_completed_count:
            if review_emitted:
                action = "blocked_review"
            elif attempt_count <= max_retries:
                retry_ordinal = attempt_count
                retry_note = (
                    "The completion-evidence gap persists after completed "
                    f"attempt {retry_ordinal}; produce fresh aligned evidence."
                )
                prepared = retry_proposal(
                    proposal,
                    retry_ordinal=retry_ordinal,
                    note=retry_note,
                )
                action = "retry"
            else:
                review_note = (
                    f"Retry budget exhausted after {max_retries} retries; "
                    "review the persistent completion-evidence gap and "
                    "record a durable block or an actionable remediation."
                )
                prepared = review_proposal(
                    proposal,
                    retry_count=max_retries,
                    note=review_note,
                )
                action = "review"
        elif family_unresolved and not same_instance:
            # Fresher diagnostics update observability only. A board task must
            # complete or become blocked before another task or review can be
            # emitted and before either lifecycle budget advances.
            action = "diagnostics_updated"
        elif not family_unresolved:
            action = "fresh"

        lifecycle_actions[family_key] = action
        prepared_typed[family_key] = prepared

    materialization_values: list[Any] = []
    emitted_families: set[str] = set()
    for proposal in proposal_values:
        family_key = str(getattr(proposal, "family_key", "") or "").strip()
        if not family_key and isinstance(proposal, Mapping):
            family_key = str(proposal.get("family_key") or "").strip()
        if not family_key:
            materialization_values.append(proposal)
            continue
        if family_key in emitted_families:
            continue
        emitted_families.add(family_key)
        # Stable unresolved work, refreshed diagnostics, durable blocks, and
        # active tasks are lifecycle state updates, not generation candidates.
        if lifecycle_actions[family_key] in {
            "active",
            "blocked_review",
            "diagnostics_updated",
            "stable",
        }:
            continue
        materialization_values.append(prepared_typed[family_key])
    proposal_values = tuple(materialization_values)

    dedupe_existing = list(existing)
    if active_keys is not None:
        typed_goal_ids = {
            value.parent_goal_id
            for value in typed_candidates.values()
        }
        dedupe_existing = []
        for raw in existing:
            item = ObjectiveWorkProposal.from_dict(raw)
            family_key = item.family_key
            if not family_key:
                # Completed generic completion-gate tasks are historical
                # provenance, not authority to suppress a newly typed gap.
                if (
                    item.source == "completion_gate"
                    and item.parent_goal_id in typed_goal_ids
                    and item.semantic_key not in active_keys
                ):
                    continue
                dedupe_existing.append(raw)
                continue
            current = typed_candidates.get(family_key)
            if current is None:
                dedupe_existing.append(raw)
                continue
            # Fresh occurrences, terminal-driven retries, and the one review
            # task intentionally supersede historical canonical records in the
            # same semantic family. All other states remain deduplicated.
            if lifecycle_actions.get(family_key) not in {
                "fresh",
                "retry",
                "review",
            }:
                dedupe_existing.append(raw)

    evaluation = None
    if evaluation_policy is not None:
        evaluation = evaluate_objective_work_proposals(
            proposal_values,
            policy=evaluation_policy,
            objective_terms=objective_terms,
            known_canonical_ids=(
                str(item.get("canonical_id") or "") for item in dedupe_existing
            ),
            known_semantic_keys=(
                str(item.get("semantic_key") or "") for item in dedupe_existing
            ),
        )
        proposal_values = evaluation.accepted_proposals
    result = materialize_bounded_objective_work(
        proposal_values,
        existing_work=dedupe_existing,
        limits=limits,
        current_open_work=current_open_work,
    )
    accepted_by_family = {
        item.family_key: item
        for item in result.accepted
        if item.family_key
    }
    next_family_states = {
        key: dict(value) for key, value in prior_family_states.items()
    }
    current_families = set(typed_candidates)
    for family_key, state in next_family_states.items():
        goal_was_observed = (
            observed_goal_ids is None
            or str(state.get("goal_id") or "") in observed_goal_ids
        )
        if family_key not in current_families and goal_was_observed:
            state["resolved"] = True
            state["active"] = False
            state["outcome"] = "resolved"
    for family_key, proposal in typed_candidates.items():
        prior = prior_family_states.get(family_key, {})
        action = lifecycle_actions[family_key]
        completed_count = completed_counts.get(family_key, 0)
        blocked_count = blocked_counts.get(family_key, 0)
        is_active = bool(
            active_keys is not None and family_key in active_keys
        )
        accepted = accepted_by_family.get(family_key)

        if action == "blocked_review":
            if prior:
                blocked_state = dict(prior)
                blocked_state.update(
                    {
                        "resolved": False,
                        "active": False,
                        "outcome": "blocked_review",
                        "review_emitted": True,
                        "completed_task_count": completed_count,
                        "blocked_task_count": blocked_count,
                        "latest_instance_key": proposal.instance_key,
                        "latest_diagnostic_canonical_id": proposal.canonical_id,
                    }
                )
                next_family_states[family_key] = blocked_state
            continue
        if is_active:
            active_state = dict(prior) if prior else {
                "family_key": family_key,
                "instance_key": proposal.instance_key,
                "canonical_id": proposal.canonical_id,
                "goal_id": proposal.parent_goal_id,
                "occurrence": 1,
                "attempt_count": 1,
                "completed_task_count": completed_count,
                "blocked_task_count": blocked_count,
                "review_emitted": (
                    proposal.source == "completion_gate_gap_manual_review"
                ),
                "outcome": (
                    "review_required"
                    if proposal.source == "completion_gate_gap_manual_review"
                    else "actionable"
                ),
            }
            active_state.update(
                {
                    "resolved": False,
                    "active": True,
                    "latest_instance_key": proposal.instance_key,
                    "latest_diagnostic_canonical_id": proposal.canonical_id,
                }
            )
            next_family_states[family_key] = active_state
            continue
        if accepted is None:
            if prior:
                retained_state = dict(prior)
                retained_state["resolved"] = False
                retained_state["active"] = False
                retained_state["latest_instance_key"] = proposal.instance_key
                retained_state[
                    "latest_diagnostic_canonical_id"
                ] = proposal.canonical_id
                next_family_states[family_key] = retained_state
            continue

        is_fresh = action == "fresh"
        prior_occurrence = max(0, int(prior.get("occurrence", 0) or 0))
        occurrence = (
            prior_occurrence + 1
            if prior
            else 1
        ) if is_fresh else max(1, prior_occurrence)
        prior_attempt_count = max(
            1, int(prior.get("attempt_count", 1) or 1)
        )
        attempt_count = (
            1
            if is_fresh
            else prior_attempt_count + (1 if action == "retry" else 0)
        )
        manual_review_accepted = (
            accepted.source == "completion_gate_gap_manual_review"
        )
        outcome = {
            "fresh": "actionable",
            "retry": "retry",
            "review": "review_required",
            "stable": "actionable",
        }.get(action, "actionable")
        if manual_review_accepted:
            outcome = "review_required"
        next_family_states[family_key] = {
            "family_key": family_key,
            "instance_key": proposal.instance_key,
            "canonical_id": accepted.canonical_id,
            "goal_id": proposal.parent_goal_id,
            "occurrence": occurrence,
            "attempt_count": attempt_count,
            "completed_task_count": completed_count,
            "blocked_task_count": blocked_count,
            "review_emitted": (
                action == "review"
                or manual_review_accepted
                or (not is_fresh and bool(prior.get("review_emitted", False)))
            ),
            "outcome": outcome,
            "resolved": False,
            "active": False,
            "latest_instance_key": proposal.instance_key,
            "latest_diagnostic_canonical_id": proposal.canonical_id,
        }
    return result, persist_objective_generation(
        artifact_path,
        result,
        existing_work=existing,
        evaluation=evaluation,
        gap_family_states=next_family_states,
    )


@dataclass(frozen=True)
class ObjectiveGenerationAdmissionResult:
    """Review/admission outcome for generated GOAL and SUBGOAL records."""

    mode: str
    status: str
    transaction_id: str
    preview: Any
    reason_codes: tuple[str, ...] = ()
    materialized_goal_ids: tuple[str, ...] = ()
    review_persisted: bool = False
    resumable: bool = False
    tracker_transaction: Mapping[str, Any] | None = None

    @property
    def admitted(self) -> bool:
        return self.status in {"committed", "already_committed", "resumed"}

    def to_dict(self) -> dict[str, Any]:
        preview_payload = (
            self.preview.to_dict()
            if hasattr(self.preview, "to_dict")
            else dict(self.preview or {})
        )
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/objective-generation-admission@1",
            "mode": self.mode,
            "status": self.status,
            "transaction_id": self.transaction_id,
            "admitted": self.admitted,
            "reason_codes": list(self.reason_codes),
            "materialized_goal_ids": list(self.materialized_goal_ids),
            "review_persisted": self.review_persisted,
            "resumable": self.resumable,
            "preview": preview_payload,
            "tracker_transaction": (
                dict(self.tracker_transaction)
                if isinstance(self.tracker_transaction, Mapping)
                else self.tracker_transaction
            ),
        }


def _contract_value(value: Any, contract_type: type[Any]) -> Any:
    if isinstance(value, contract_type):
        return value
    if isinstance(value, Mapping):
        return contract_type.from_dict(value)
    raise TypeError(f"expected {contract_type.__name__}")


def _verified_authority_receipt_ids(
    verification: Any,
    supplied_receipts: Iterable[Any] | Mapping[str, Any] = (),
) -> set[str]:
    """Resolve actual verified authority records instead of trusting claimed IDs."""

    resolved: set[str] = set()
    if verification is not None and bool(getattr(verification, "verified", False)):
        content_id = str(getattr(verification, "content_id", "") or "").strip()
        if content_id:
            resolved.add(content_id)
        rounds = tuple(getattr(verification, "rounds", ()) or ())
        final_round = rounds[-1] if rounds else None
        authoritative_attempt_ids: set[str] = set()
        for result in tuple(getattr(final_round, "portfolio_results", ()) or ()):
            verdict = str(
                getattr(getattr(result, "verdict", ""), "value", getattr(result, "verdict", ""))
            ).lower()
            if verdict != "proved":
                continue
            result_id = str(
                getattr(result, "result_id", "")
                or getattr(result, "content_id", "")
                or ""
            ).strip()
            if result_id:
                resolved.add(result_id)
            authoritative_attempt_ids.update(
                str(item)
                for item in (getattr(result, "authority_attempt_ids", ()) or ())
                if str(item).strip()
            )
        for retained in tuple(getattr(verification, "all_attempts", ()) or ()):
            attempt = getattr(retained, "attempt", retained)
            attempt_id = str(
                getattr(attempt, "attempt_id", "")
                or getattr(attempt, "content_id", "")
                or ""
            ).strip()
            outcome = str(
                getattr(
                    getattr(attempt, "effective_outcome", ""),
                    "value",
                    getattr(attempt, "effective_outcome", ""),
                )
            ).lower()
            if (
                attempt_id in authoritative_attempt_ids
                and bool(getattr(attempt, "authoritative", False))
                and outcome == "verified"
            ):
                resolved.add(attempt_id)
                for name in ("capability_receipt_id", "conformance_gate_id"):
                    identifier = str(getattr(attempt, name, "") or "").strip()
                    if identifier:
                        resolved.add(identifier)

    if isinstance(supplied_receipts, Mapping):
        values = supplied_receipts.items()
    else:
        values = (("", value) for value in supplied_receipts)
    for supplied_id, receipt in values:
        authoritative_verdict = getattr(receipt, "authoritative_verdict", "")
        verdict_value = str(
            getattr(authoritative_verdict, "value", authoritative_verdict) or ""
        ).lower()
        authoritative = bool(getattr(receipt, "authoritative", False)) or (
            verdict_value in {"proved", "verified"}
        )
        verified = bool(
            getattr(receipt, "verified", False)
            or getattr(receipt, "proved", False)
            or verdict_value in {"proved", "verified"}
        )
        if isinstance(receipt, Mapping):
            status = str(
                receipt.get("status")
                or receipt.get("verdict")
                or receipt.get("decision")
                or ""
            ).lower()
            authoritative = receipt.get("authoritative") is True
            verified = receipt.get("verified") is True or status in {
                "verified",
                "proved",
            }
        # Deterministic validation or supervisor acceptance is not proof
        # authority.  External records must explicitly carry both properties;
        # otherwise only receipts derived from the verified refinement
        # portfolio above may authorize admission.
        if not authoritative or not verified:
            continue
        identifiers = {
            str(supplied_id or "").strip(),
            str(
                getattr(receipt, "receipt_id", "")
                or getattr(receipt, "result_id", "")
                or getattr(receipt, "content_id", "")
                or ""
            ).strip(),
        }
        if isinstance(receipt, Mapping):
            identifiers.update(
                str(receipt.get(name) or "").strip()
                for name in ("receipt_id", "result_id", "content_id", "attempt_id")
            )
        resolved.update(item for item in identifiers if item)
    return resolved


def _objective_admission_transaction_id(
    preview: Any,
    *,
    mode: str,
    proposal_receipt_id: str = "",
    admission_receipt_id: str = "",
) -> str:
    material = {
        "schema": OBJECTIVE_ADMISSION_RECORD_SCHEMA,
        "mode": mode,
        "base_heap_content_id": str(preview.base_heap_content_id),
        "candidate_heap_content_id": str(preview.candidate_heap_content_id),
        "root_goal_id": str(preview.root_goal_id),
        "root_content_id": str(preview.root_content_id),
        "proposal_ids": list(preview.admitted_proposal_ids),
        "proposal_receipt_id": proposal_receipt_id,
        "admission_receipt_id": admission_receipt_id,
    }
    return "objective-admission:" + sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def materialize_admitted_objective_work(
    proposals: Iterable[Any],
    *,
    repo_root: Path,
    objective_path: Path,
    generation_path: Path,
    mode: Any,
    limits: Any = None,
    root_goal_id: str = "",
    expected_root_content_id: str = "",
    expected_repository_tree_id: str = "",
    current_repository_tree_id: str = "",
    lifecycle_owner: str = "objective_daemon",
    draft: Any = None,
    proposal_receipt: Any = None,
    admission_receipt: Any = None,
    refinement_verification: Any = None,
    required_authoritative_receipt_ids: Sequence[str] = (),
    authoritative_receipts: Iterable[Any] | Mapping[str, Any] = (),
    proposal_bindings: Mapping[str, str] | None = None,
    new_assumption_ids: Sequence[str] = (),
    unsupported_semantics: Sequence[str] = (),
    hard_policy_gates: Mapping[str, bool] | None = None,
    journal_path: Path | None = None,
    lease_guard: Callable[..., Any] | None = None,
    expected_lease_token: str | None = None,
    preview_only: bool = False,
) -> ObjectiveGenerationAdmissionResult:
    """Preview and transactionally admit generated goals under strict authority.

    Shadow is pure and assist writes only review state to the generation
    ledger.  Auto-safe is the sole mode which may call the objective tracker,
    and only after validating the frozen draft, both receipts, independent
    refinement verification, repository/root freshness, configured authority
    records, and every graph-policy gate.
    """

    from .goal_development_contracts import (
        GoalDevelopmentAdmissionReceipt,
        GoalDevelopmentMode,
        GoalDevelopmentProposalReceipt,
        GoalDecompositionDraft,
        GoalAdmissionDecision,
        GoalProposalDecision,
    )
    from .goal_refinement_verification import RefinementVerificationResult
    from .objective_graph import (
        ObjectiveGoalMaterializationPolicy,
        ObjectiveWorkKind,
        ObjectiveWorkProposal,
        parse_goal_heap,
        preview_objective_goal_materialization,
    )
    from .objective_tracker import (
        commit_objective_goal_materialization,
        objective_materialization_tree_identity,
    )

    normalized_mode = (
        mode if isinstance(mode, GoalDevelopmentMode) else GoalDevelopmentMode(str(mode))
    )
    objective_text = (
        objective_path.read_text(encoding="utf-8", errors="strict")
        if objective_path.exists()
        else ""
    )
    goal_work: list[ObjectiveWorkProposal] = []
    for raw in proposals:
        item = (
            raw
            if isinstance(raw, ObjectiveWorkProposal)
            else ObjectiveWorkProposal.from_dict(raw)
        )
        if item.kind in {ObjectiveWorkKind.GOAL, ObjectiveWorkKind.SUBGOAL}:
            goal_work.append(item)

    policy = ObjectiveGoalMaterializationPolicy(
        limits=(limits or {}),
        root_goal_id=root_goal_id,
        expected_root_content_id=expected_root_content_id,
        lifecycle_owner=lifecycle_owner,
        atomic=True,
    )
    current_goals = {
        goal.goal_id: goal for goal in parse_goal_heap(objective_text)
    }
    selected_root_id = str(root_goal_id or "").strip()
    if not selected_root_id:
        roots = [
            goal.goal_id
            for goal in current_goals.values()
            if not goal.parent_goal_ids
        ]
        if len(roots) == 1:
            selected_root_id = roots[0]

    def is_exact_materialization(item: ObjectiveWorkProposal) -> bool:
        """Recognize only a lossless prior commit of this exact proposal.

        This makes the daemon resumable when the heap replacement completed
        but final generation-ledger persistence was interrupted.  A merely
        colliding canonical ID is deliberately not treated as a replay.
        """

        goal = current_goals.get(item.canonical_id)
        if goal is None:
            return False
        parent_id = item.parent_goal_id
        if item.kind is ObjectiveWorkKind.GOAL and (
            not parent_id or parent_id in policy.root_parent_aliases
        ):
            parent_id = selected_root_id
        expected_parents = [parent_id] if parent_id else []
        try:
            graph_depth = int(goal.fields.get("graph_depth") or -1)
        except (TypeError, ValueError):
            return False
        return (
            goal.title == item.title
            and goal.parent_goal_ids == expected_parents
            and tuple(goal.dependencies) == item.dependencies
            and tuple(goal.required_evidence) == item.expected_evidence_delta
            and tuple(goal.predicted_files) == item.predicted_files
            and tuple(goal.predicted_symbols) == item.predicted_symbols
            and tuple(goal.validation_commands) == item.validation_commands
            and graph_depth == item.depth
            and goal.canonical_proposal_id == item.canonical_id
            and goal.semantic_key == item.semantic_key
            and goal.lifecycle_owner == lifecycle_owner
            and goal.fields.get("proposal_kind") == item.kind.value
            and goal.fields.get("proposal_source") == item.source
            and goal.fields.get("proposal_source_id", "") == item.source_id
        )

    already_materialized_ids = tuple(
        item.canonical_id
        for item in sorted(
            goal_work, key=lambda value: (value.depth, value.canonical_id)
        )
        if is_exact_materialization(item)
    )
    already_materialized = set(already_materialized_ids)
    pending_goal_work = [
        item for item in goal_work if item.canonical_id not in already_materialized
    ]
    preview = preview_objective_goal_materialization(
        objective_text,
        # Keep an informative duplicate preview on a full replay.  For a
        # partially finalized transaction, preview only the remaining suffix
        # so it can be committed without duplicating the durable prefix.
        pending_goal_work or goal_work,
        policy=policy,
    )
    reason_codes = list(preview.fatal_reasons)
    reason_codes.extend(
        item.reason
        for item in preview.rejected
        if not (
            not pending_goal_work
            and item.reason == "canonical_duplicate"
            and item.canonical_id in already_materialized
        )
    )
    if not goal_work:
        reason_codes.append("no_goal_or_subgoal_proposals")

    parsed_proposal_receipt = None
    parsed_admission_receipt = None
    parsed_draft = None
    parsed_verification = None
    proposal_receipt_id = ""
    admission_receipt_id = ""
    if proposal_receipt is not None:
        try:
            parsed_proposal_receipt = _contract_value(
                proposal_receipt, GoalDevelopmentProposalReceipt
            )
            proposal_receipt_id = parsed_proposal_receipt.receipt_id
        except (TypeError, ValueError) as exc:
            reason_codes.append(
                f"invalid_proposal_receipt:{type(exc).__name__}"
            )
    if admission_receipt is not None:
        try:
            parsed_admission_receipt = _contract_value(
                admission_receipt, GoalDevelopmentAdmissionReceipt
            )
            admission_receipt_id = parsed_admission_receipt.receipt_id
        except (TypeError, ValueError) as exc:
            reason_codes.append(
                f"invalid_admission_receipt:{type(exc).__name__}"
            )
    transaction_id = _objective_admission_transaction_id(
        preview,
        mode=normalized_mode.value,
        proposal_receipt_id=proposal_receipt_id,
        admission_receipt_id=admission_receipt_id,
    )
    effective_journal_path = journal_path or generation_path.with_name(
        f"{generation_path.stem}.objective-admission.json"
    )

    if normalized_mode is GoalDevelopmentMode.SHADOW:
        return ObjectiveGenerationAdmissionResult(
            mode=normalized_mode.value,
            status="shadow",
            transaction_id=transaction_id,
            preview=preview,
            reason_codes=tuple(sorted(set(reason_codes or ["shadow_mode"]))),
        )

    def admission_records(
        status: str,
        reasons: Sequence[str],
        tracker_transaction: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        by_id = {item.proposal.canonical_id: item for item in preview.materialized}
        return [
            {
                "schema": OBJECTIVE_ADMISSION_RECORD_SCHEMA,
                "canonical_id": item.canonical_id,
                "semantic_key": item.semantic_key,
                "status": status,
                "mode": normalized_mode.value,
                "transaction_id": transaction_id,
                "proposal": item.to_dict(),
                "preview": (
                    by_id[item.canonical_id].to_dict()
                    if item.canonical_id in by_id
                    else None
                ),
                "proposal_receipt_id": proposal_receipt_id,
                "admission_receipt_id": admission_receipt_id,
                "authoritative_receipt_ids": list(
                    getattr(parsed_admission_receipt, "authoritative_receipt_ids", ())
                    or ()
                ),
                "reason_codes": sorted(set(str(reason) for reason in reasons if reason)),
                "lifecycle_owner": lifecycle_owner,
                "tracker_transaction": (
                    dict(tracker_transaction)
                    if tracker_transaction is not None
                    else None
                ),
                "attempts": [
                    {
                        "transaction_id": transaction_id,
                        "status": status,
                        "reason_codes": sorted(
                            set(str(reason) for reason in reasons if reason)
                        ),
                    }
                ],
            }
            for item in sorted(goal_work, key=lambda value: value.canonical_id)
        ]

    if normalized_mode is GoalDevelopmentMode.ASSIST:
        reasons = tuple(sorted(set(reason_codes or ["review_required"])))
        _persist_objective_admission_records(
            generation_path, admission_records("review_required", reasons)
        )
        return ObjectiveGenerationAdmissionResult(
            mode=normalized_mode.value,
            status="review_required",
            transaction_id=transaction_id,
            preview=preview,
            reason_codes=reasons,
            review_persisted=True,
            resumable=True,
        )

    if normalized_mode is not GoalDevelopmentMode.AUTO_SAFE:
        reasons = tuple(
            sorted(set(reason_codes or [f"{normalized_mode.value}_mode_not_admissible"]))
        )
        return ObjectiveGenerationAdmissionResult(
            mode=normalized_mode.value,
            status="not_admitted",
            transaction_id=transaction_id,
            preview=preview,
            reason_codes=reasons,
        )

    # AUTO_SAFE trust boundary.
    try:
        parsed_draft = _contract_value(draft, GoalDecompositionDraft)
    except (TypeError, ValueError) as exc:
        reason_codes.append(f"invalid_draft:{type(exc).__name__}")
    try:
        if parsed_proposal_receipt is None:
            raise TypeError("proposal receipt is required")
        if parsed_draft is None:
            raise TypeError("draft is required")
        parsed_proposal_receipt.validate_draft(parsed_draft)
        if parsed_proposal_receipt.decision is not GoalProposalDecision.ACCEPTED:
            reason_codes.append("proposal_receipt_not_accepted")
    except (TypeError, ValueError) as exc:
        reason_codes.append(f"proposal_receipt_gate:{type(exc).__name__}")
    try:
        if parsed_admission_receipt is None:
            raise TypeError("admission receipt is required")
        if parsed_proposal_receipt is None:
            raise TypeError("proposal receipt is required")
        parsed_admission_receipt.validate_proposal_receipt(parsed_proposal_receipt)
        if (
            parsed_admission_receipt.mode is not GoalDevelopmentMode.AUTO_SAFE
            or parsed_admission_receipt.decision is not GoalAdmissionDecision.ADMITTED
        ):
            reason_codes.append("admission_receipt_not_admitted")
    except (TypeError, ValueError) as exc:
        reason_codes.append(f"admission_receipt_gate:{type(exc).__name__}")

    try:
        parsed_verification = _contract_value(
            refinement_verification, RefinementVerificationResult
        )
        if not parsed_verification.verified:
            reason_codes.append("refinement_not_verified")
    except (TypeError, ValueError) as exc:
        reason_codes.append(f"refinement_verification_gate:{type(exc).__name__}")

    if parsed_draft is not None:
        request = parsed_draft.request
        if request.root_goal_id != preview.root_goal_id:
            reason_codes.append("changed_root_goal")
        if parsed_verification is not None:
            frozen = parsed_verification.frozen_context
            if (
                frozen.root_goal_id != request.root_goal_id
                or frozen.root_goal_content_id != request.root_goal_content_id
            ):
                reason_codes.append("changed_frozen_root")
            if tuple(frozen.assumption_ids) != tuple(request.assumption_ids):
                reason_codes.append("changed_assumptions")
        bindings = {
            str(key): str(value)
            for key, value in (proposal_bindings or {}).items()
            if str(key).strip() and str(value).strip()
        }
        admitted_ids = set(
            getattr(parsed_admission_receipt, "proposal_ids", ()) or ()
        )
        for item in goal_work:
            source_proposal_id = (
                bindings.get(item.canonical_id)
                or item.source_id
                or item.canonical_id
            )
            if source_proposal_id not in admitted_ids:
                reason_codes.append(
                    f"unbound_proposal:{item.canonical_id}"
                )

    if new_assumption_ids:
        reason_codes.append("new_assumptions")
    if unsupported_semantics:
        reason_codes.append("unsupported_semantics")
    for gate_name, passed in sorted((hard_policy_gates or {}).items()):
        if passed is not True:
            reason_codes.append(f"hard_policy_gate:{gate_name}")

    current_tree = str(current_repository_tree_id or "").strip()
    if not current_tree:
        current_tree = objective_materialization_tree_identity(
            repo_root,
            objective_path=objective_path,
            journal_path=effective_journal_path,
            control_paths=(generation_path,),
        ).tree_id
    frozen_tree = str(expected_repository_tree_id or "").strip()
    if parsed_draft is not None:
        request_tree = str(parsed_draft.request.repository_tree_id)
        if frozen_tree and frozen_tree != request_tree:
            reason_codes.append("repository_tree_binding_mismatch")
        frozen_tree = request_tree
    if frozen_tree and current_tree != frozen_tree:
        reason_codes.append("stale_repository_tree")

    resolved_authorities = _verified_authority_receipt_ids(
        parsed_verification, authoritative_receipts
    )
    claimed_authorities = set(
        getattr(parsed_admission_receipt, "authoritative_receipt_ids", ()) or ()
    )
    configured_authorities = {
        str(item).strip()
        for item in required_authoritative_receipt_ids
        if str(item).strip()
    }
    if configured_authorities and not configured_authorities.issubset(
        claimed_authorities
    ):
        reason_codes.append("missing_configured_authoritative_receipts")
    required_resolution = claimed_authorities | configured_authorities
    if not required_resolution or not required_resolution.issubset(
        resolved_authorities
    ):
        reason_codes.append("unresolved_authoritative_receipts")

    reason_codes = sorted(set(reason_codes))
    graph_ready = bool(goal_work) and (
        len(already_materialized) == len(goal_work) or preview.ready
    )
    if reason_codes or not graph_ready:
        _persist_objective_admission_records(
            generation_path, admission_records("rejected", reason_codes)
        )
        return ObjectiveGenerationAdmissionResult(
            mode=normalized_mode.value,
            status="rejected",
            transaction_id=transaction_id,
            preview=preview,
            reason_codes=tuple(reason_codes),
            review_persisted=True,
            resumable=True,
        )
    if len(already_materialized) == len(goal_work):
        _persist_objective_admission_records(
            generation_path, admission_records("admitted", ())
        )
        return ObjectiveGenerationAdmissionResult(
            mode=normalized_mode.value,
            status="already_committed",
            transaction_id=transaction_id,
            preview=preview,
            materialized_goal_ids=already_materialized_ids,
            review_persisted=True,
            resumable=False,
        )
    if preview_only:
        return ObjectiveGenerationAdmissionResult(
            mode=normalized_mode.value,
            status="prepared",
            transaction_id=transaction_id,
            preview=preview,
            resumable=True,
        )

    _persist_objective_admission_records(
        generation_path, admission_records("prepared", ())
    )
    try:
        tracker_result = commit_objective_goal_materialization(
            repo_root=repo_root,
            objective_path=objective_path,
            journal_path=effective_journal_path,
            preview=preview,
            expected_repository_tree_id=frozen_tree,
            lease_guard=lease_guard,
            expected_lease_token=expected_lease_token,
            control_paths=(generation_path,),
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        failure_reasons = (f"partial_write:{type(exc).__name__}",)
        _persist_objective_admission_records(
            generation_path, admission_records("prepared", failure_reasons)
        )
        return ObjectiveGenerationAdmissionResult(
            mode=normalized_mode.value,
            status="prepared",
            transaction_id=transaction_id,
            preview=preview,
            reason_codes=failure_reasons,
            materialized_goal_ids=already_materialized_ids,
            review_persisted=True,
            resumable=True,
        )
    tracker_payload = (
        tracker_result.to_dict()
        if hasattr(tracker_result, "to_dict")
        else dict(tracker_result)
    )
    tracker_status = str(
        getattr(
            getattr(tracker_result, "state", ""),
            "value",
            getattr(tracker_result, "state", ""),
        )
        or tracker_payload.get("state")
        or ""
    )
    committed = bool(
        getattr(tracker_result, "committed", False)
        or tracker_status in {"committed", "already_committed", "resumed"}
    )
    final_status = tracker_status or ("committed" if committed else "failed")
    final_reasons = tuple(
        sorted(
            set(
                str(item)
                for item in (
                    tracker_payload.get("reason_codes")
                    or tracker_payload.get("reasons")
                    or ()
                )
                if str(item)
            )
        )
    )
    _persist_objective_admission_records(
        generation_path,
        admission_records(
            "admitted" if committed else "prepared",
            final_reasons,
            tracker_payload,
        ),
    )
    return ObjectiveGenerationAdmissionResult(
        mode=normalized_mode.value,
        status=final_status,
        transaction_id=transaction_id,
        preview=preview,
        reason_codes=final_reasons,
        materialized_goal_ids=(
            (
                already_materialized_ids
                + tuple(preview.admitted_proposal_ids)
            )
            if committed
            else already_materialized_ids
        ),
        review_persisted=True,
        resumable=not committed,
        tracker_transaction=tracker_payload,
    )


# Discoverable aliases used by callers which phrase admission around goals or a cycle.
materialize_objective_generation_admission = materialize_admitted_objective_work
admit_objective_generation_cycle = materialize_admitted_objective_work


def completion_gate_work_terms(decision: Mapping[str, Any]) -> tuple[str, ...]:
    """Project verbose completion diagnostics onto stable repair dimensions."""

    diagnostics = decision.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    terms: list[str] = []
    if diagnostics.get("uncovered_criteria"):
        terms.append("completion criterion coverage")
    if diagnostics.get("stale_evidence"):
        terms.append("completion evidence freshness")
    analyzer = diagnostics.get("analyzer_health")
    if isinstance(analyzer, Mapping) and analyzer.get("passed") is not True:
        terms.append("completion analyzer health")
    quorum = diagnostics.get("exhaustion_quorum")
    if isinstance(quorum, Mapping) and quorum.get("satisfied") is not True:
        terms.append("completion exhaustion quorum")
    if diagnostics.get("reopen_reasons"):
        terms.append("completion contradiction repair")

    reason_codes = tuple(
        dict.fromkeys(
            str(item).strip().casefold().replace("-", "_").replace(" ", "_")
            for item in decision.get("reason_codes", ())
            if str(item).strip()
        )
    )
    if any("task" in reason for reason in reason_codes):
        terms.append("completion task closure")
    if any("child" in reason or "descendant" in reason for reason in reason_codes):
        terms.append("completion child-goal verification")
    if not terms:
        terms.extend(f"completion gate {reason}" for reason in reason_codes)
    return tuple(dict.fromkeys(terms))


def _stable_completion_gap_key(prefix: str, material: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(material),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return f"{prefix}/" + sha256(canonical.encode("utf-8")).hexdigest()


def _normalized_completion_gap_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _completion_gap_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        normalized = " ".join(value.split())
        return (normalized,) if normalized else ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    values: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            rendered = json.dumps(
                dict(item),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        else:
            rendered = " ".join(str(item or "").split())
        if rendered and rendered not in values:
            values.append(rendered)
    return tuple(values)


_VOLATILE_COMPLETION_DIAGNOSTIC_FIELDS = frozenset(
    {
        "cid",
        "generated_at",
        "observed_at",
        "provenance_cid",
        "receipt_cid",
        "receipt_sha256",
        "refreshed_at",
        "repository_tree",
        "sha256",
        "timestamp",
        "tree_id",
    }
)


def _stable_completion_gap_diagnostic(value: Any) -> Any:
    """Remove refresh-only provenance from an actionable diagnostic value."""

    if isinstance(value, Mapping):
        return {
            str(key): _stable_completion_gap_diagnostic(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).strip().casefold()
            not in _VOLATILE_COMPLETION_DIAGNOSTIC_FIELDS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_stable_completion_gap_diagnostic(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def _completion_gap_alignment_diagnostics(
    *surfaces: Mapping[str, Any],
) -> tuple[str, ...]:
    labels = {
        "probe_outcome": "Probe outcome",
        "documentation_alignment": "Documentation alignment",
        "debt_path": "Documentation debt path",
    }
    diagnostics: list[str] = []
    for surface in surfaces:
        for field_name, label in labels.items():
            raw = surface.get(field_name)
            if raw in (None, "", (), [], {}):
                continue
            stable = _stable_completion_gap_diagnostic(raw)
            if stable in (None, "", (), [], {}):
                continue
            rendered = (
                stable
                if isinstance(stable, str)
                else json.dumps(
                    stable,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            )
            diagnostic = f"{label}: {rendered}"
            if diagnostic not in diagnostics:
                diagnostics.append(diagnostic)
    return tuple(diagnostics)


def _completion_gap_paths(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Iterable[Any] = re.split(r"[,\n]+", value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        return ()
    paths: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        path = item.strip().replace("\\", "/")
        if path and path not in paths:
            paths.append(path)
    return tuple(paths)


_COMPLETION_GAP_EDIT_TARGET_FIELDS = (
    "implementation_paths",
    "analyzer_implementation_paths",
    "affected_document_paths",
    "validator_source_paths",
    "validator_source_path",
)


def _completion_gap_explicit_paths(
    *surfaces: Mapping[str, Any],
) -> tuple[str, ...]:
    paths: list[str] = []
    for surface in surfaces:
        for field_name in _COMPLETION_GAP_EDIT_TARGET_FIELDS:
            for path in _completion_gap_paths(surface.get(field_name)):
                if path not in paths:
                    paths.append(path)
    return tuple(paths)


def _completion_gap_evidence_path_diagnostics(
    *surfaces: Mapping[str, Any],
) -> tuple[str, ...]:
    diagnostics: list[str] = []
    for surface in surfaces:
        for path in _completion_gap_paths(surface.get("evidence_paths")):
            diagnostic = f"Evidence path (read-only): {path}"
            if diagnostic not in diagnostics:
                diagnostics.append(diagnostic)
        receipt_path = str(surface.get("path") or "").strip()
        if receipt_path:
            diagnostic = f"Receipt/report path (read-only): {receipt_path}"
            if diagnostic not in diagnostics:
                diagnostics.append(diagnostic)
    return tuple(diagnostics)


def _completion_gap_precise_files(
    values: Iterable[str],
    *,
    repo_root: Path,
) -> tuple[str, ...]:
    """Retain explicit file targets which resolve inside ``repo_root``."""

    paths: list[str] = []
    extensionless_files = {
        "changelog",
        "contributing",
        "license",
        "makefile",
        "readme",
    }
    resolved_root = repo_root.resolve()
    for value in values:
        raw_path = str(value or "").strip()
        if (
            not raw_path
            or "\x00" in raw_path
            or raw_path.endswith(("/", "\\"))
        ):
            continue
        normalized = raw_path.replace("\\", "/")
        if (
            "://" in normalized
            or re.match(r"^[A-Za-z]:", normalized)
        ):
            continue
        relative_path = PurePosixPath(normalized)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            continue
        normalized = relative_path.as_posix()
        if normalized in {"", "."}:
            continue
        name = relative_path.name
        if "." not in name and name.casefold() not in extensionless_files:
            continue
        try:
            resolved_target = (
                resolved_root.joinpath(*relative_path.parts).resolve(strict=False)
            )
            resolved_target.relative_to(resolved_root)
            if resolved_target.exists() and not resolved_target.is_file():
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        if normalized not in paths:
            paths.append(normalized)
    return tuple(paths)


def _completion_gap_validation_commands(
    *,
    row: Mapping[str, Any],
    receipt: Mapping[str, Any] | None,
    goal_validation: Sequence[str],
    default_validation: Sequence[str],
) -> tuple[str, ...]:
    values: list[str] = []
    tiers: list[list[tuple[str, Any]]] = [
        [
            ("validation_commands", row.get("validation_commands")),
            ("validation_command", row.get("validation_command")),
        ],
    ]
    if receipt is not None:
        tiers.append(
            [
                ("validation_commands", receipt.get("validation_commands")),
                ("validation_command", receipt.get("validation_command")),
                ("command", receipt.get("command")),
            ]
        )
    for sources in tiers:
        tier_values: list[str] = []
        for field_name, raw in sources:
            if not raw:
                continue
            if field_name in {"validation_command", "command"} and isinstance(
                raw, Sequence
            ) and not isinstance(raw, (str, bytes)):
                command = shlex.join(
                    str(item) for item in raw if str(item).strip()
                )
                if command and command not in tier_values:
                    tier_values.append(command)
                continue
            for command in _completion_gap_strings(raw):
                if command not in tier_values:
                    tier_values.append(command)
        if tier_values:
            values.extend(tier_values)
            break
    if not values:
        values.extend(str(item) for item in goal_validation if str(item).strip())
    for command in default_validation:
        normalized = str(command).strip()
        if normalized and normalized not in values:
            values.append(normalized)
    return tuple(values)


def _rejected_receipt_channel(
    receipt: Mapping[str, Any],
    *,
    goal_id: str,
    missing_channels: Sequence[str],
) -> str:
    explicit = str(
        receipt.get("producer_channel")
        or receipt.get("required_producer_channel")
        or receipt.get("evidence_channel")
        or ""
    ).strip()
    if explicit:
        return explicit
    repository_channel = f"repository-validator:{goal_id}"
    if repository_channel in missing_channels:
        return repository_channel
    return str(missing_channels[0]) if len(missing_channels) == 1 else repository_channel


def _documentation_completion_gap_proposals(
    *,
    repo_root: Path,
    objective_path: Path,
    goal_id: str,
    goal: Any,
    record: Mapping[str, Any],
    default_validation: Sequence[str],
) -> tuple[ObjectiveWorkProposal, ...]:
    """Adapt documentation-style unverified rows into direct goal tasks.

    This is intentionally a generation adapter, not a completion-evidence
    adapter. Rejected receipts contribute bounded repair diagnostics only and
    are never returned as evidence records.
    """

    coverage = record.get("coverage")
    if not isinstance(coverage, Mapping):
        coverage = {}
    raw_rows = coverage.get("criteria", ())
    rows = [
        dict(item)
        for item in raw_rows
        if isinstance(item, Mapping)
        and (
            item.get("verified") is False
            or (
                item.get("verified") is not True
                and str(item.get("status") or "").strip().lower()
                not in {
                    "complete",
                    "completed",
                    "passed",
                    "satisfied",
                    "verified",
                }
            )
        )
    ] if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)) else []
    for row in rows:
        if str(row.get("required_producer_channel") or "").strip():
            continue
        criterion = " ".join(str(row.get("criterion") or "").split())
        criterion_id = str(row.get("criterion_id") or "").strip()
        if not criterion_id:
            criterion_id = _stable_completion_gap_key(
                "criterion/v1",
                {"criterion": _normalized_completion_gap_text(criterion)},
            )
            row["criterion_id"] = criterion_id
        row["required_producer_channel"] = (
            f"completion-gate-criterion:{criterion_id}"
        )
    missing_channels = sorted(
        {
            str(item).strip()
            for item in record.get("missing_producer_channels", ())
            if str(item).strip()
        }
    ) if isinstance(record.get("missing_producer_channels", ()), Sequence) and not isinstance(record.get("missing_producer_channels", ()), (str, bytes)) else []
    row_channels = {
        str(row.get("required_producer_channel") or "").strip()
        for row in rows
        if str(row.get("required_producer_channel") or "").strip()
    }
    channels = sorted(set(missing_channels) | row_channels)

    rejected_values = record.get("rejected_receipts", ())
    rejected = [
        dict(item)
        for item in rejected_values
        if isinstance(item, Mapping)
    ] if isinstance(rejected_values, Sequence) and not isinstance(rejected_values, (str, bytes)) else []
    receipts_by_channel: dict[str, list[dict[str, Any]]] = {}
    for receipt in rejected:
        channel = _rejected_receipt_channel(
            receipt,
            goal_id=goal_id,
            missing_channels=channels,
        )
        receipts_by_channel.setdefault(channel, []).append(receipt)
    channels = sorted(set(channels) | set(receipts_by_channel))
    if not channels:
        return ()

    binding: dict[str, Any] = {}
    for candidate in (record.get("binding"), coverage.get("binding")):
        if isinstance(candidate, Mapping):
            for key, value in candidate.items():
                binding.setdefault(str(key), value)
    proof_revisions = {
        key: str(binding.get(key) or "")
        for key in (
            "objective_revision",
            "analyzer_version",
            "configuration_revision",
        )
    }
    namespace = _stable_completion_gap_key(
        "objective-namespace/v1",
        {"objective_path": objective_path.resolve().as_posix()},
    )
    fields = goal.fields
    goal_symbols = tuple(goal.required_evidence)
    goal_validation = tuple(split_csv([str(fields.get("validation") or "")]))
    proposals: list[ObjectiveWorkProposal] = []

    work_rows: list[dict[str, Any]] = list(rows)
    for channel in channels:
        if channel not in row_channels:
            missing_channel = channel in missing_channels
            work_rows.append(
                {
                    "criterion": (
                        f"Align documentation with evidence from {channel}."
                        if missing_channel
                        else (
                            "Align documentation evidence with the rejected "
                            f"receipt from {channel}."
                        )
                    ),
                    "criterion_id": f"channel:{channel}",
                    "status": "unverified",
                    "verified": False,
                    "reason_codes": [
                        (
                            f"missing_producer_channel:{channel}"
                            if missing_channel
                            else f"rejected_receipt:{channel}"
                        ),
                    ],
                    "required_producer_channel": channel,
                }
            )

    for row in work_rows:
        channel = str(row.get("required_producer_channel") or "").strip()
        criterion = " ".join(str(row.get("criterion") or "").split())
        if not channel or not criterion:
            continue
        criterion_id = str(row.get("criterion_id") or "").strip()
        if not criterion_id:
            criterion_id = _stable_completion_gap_key(
                "criterion/v1",
                {"criterion": _normalized_completion_gap_text(criterion)},
            )
        family_key = _stable_completion_gap_key(
            "objective-family/v1",
            {
                "namespace": namespace,
                "goal_id": goal_id,
                "criterion_id": criterion_id,
                "channel": channel,
            },
        )
        channel_receipts = receipts_by_channel.get(channel, [])
        receipt = channel_receipts[0] if channel_receipts else None
        row_reasons = sorted(
            set(_completion_gap_strings(row.get("reason_codes", ())))
        )
        receipt_reasons: list[str] = []
        receipt_diagnostics: list[str] = []
        receipt_instance: dict[str, Any] = {}
        for rejected_receipt in channel_receipts:
            receipt_reasons.extend(
                _completion_gap_strings(rejected_receipt.get("reason_codes", ()))
            )
            for field_name in ("errors", "failed_checks"):
                receipt_diagnostics.extend(
                    _completion_gap_strings(rejected_receipt.get(field_name, ()))
                )
            path = str(rejected_receipt.get("path") or "").strip()
            receipt_instance.setdefault("path", path)
            receipt_instance.setdefault(
                "status",
                str(rejected_receipt.get("status") or ""),
            )
            receipt_instance.setdefault(
                "validation_returncode",
                rejected_receipt.get("validation_returncode"),
            )
        receipt_reasons = sorted(set(receipt_reasons))
        receipt_diagnostics = sorted(set(receipt_diagnostics))
        alignment_diagnostics = _completion_gap_alignment_diagnostics(
            row,
            *channel_receipts,
            coverage,
            record,
        )
        evidence_path_diagnostics = _completion_gap_evidence_path_diagnostics(
            row,
            *channel_receipts,
        )
        predicted_files = _completion_gap_precise_files(
            _completion_gap_explicit_paths(
                row,
                *channel_receipts,
            ),
            repo_root=repo_root,
        )
        manual_review_only = not predicted_files
        instance_key = _stable_completion_gap_key(
            "objective-instance/v1",
            {
                "family_key": family_key,
                "proof_revisions": proof_revisions,
                "row_reason_codes": row_reasons,
                "receipt": receipt_instance,
                "receipt_reason_codes": receipt_reasons,
                "failed_checks": receipt_diagnostics,
                "alignment_diagnostics": alignment_diagnostics,
                "affected_paths": predicted_files,
                "evidence_path_diagnostics": evidence_path_diagnostics,
            },
        )
        delta = tuple(
            dict.fromkeys(
                [
                    (
                        "Align documentation claims with "
                        f"{channel} evidence for: {criterion}"
                    ),
                    *(
                        f"Reconcile documentation evidence for gate diagnostic: {item}"
                        for item in row_reasons
                    ),
                    *(
                        f"Reconcile documentation evidence for rejected-receipt "
                        f"diagnostic: {item}"
                        for item in receipt_reasons
                    ),
                    *receipt_diagnostics,
                    *alignment_diagnostics,
                    *evidence_path_diagnostics,
                    (
                        "Treat failed product probes as current-state observations; "
                        "change product code only when the parent acceptance "
                        "criterion explicitly requires product repair."
                    ),
                    *(
                        (
                            "Manual review required: no precise implementation, "
                            "affected-document, or validator-source file was "
                            "authorized as an edit target.",
                        )
                        if manual_review_only
                        else ()
                    ),
                ]
            )
        )
        predicted_symbols = tuple(
            dict.fromkeys([*goal_symbols, criterion_id, channel])
        )
        validation = _completion_gap_validation_commands(
            row=row,
            receipt=receipt,
            goal_validation=goal_validation,
            default_validation=default_validation,
        )
        if not predicted_symbols or not validation:
            continue
        parent_terms = tuple(
            dict.fromkeys(
                [
                    *goal.required_evidence,
                    criterion,
                    channel,
                    "documentation-evidence alignment",
                ]
            )
        )
        criterion_label = (
            criterion
            if len(criterion) <= 96
            else criterion[:93].rstrip() + "..."
        )
        proposals.append(
            ObjectiveWorkProposal(
                kind=ObjectiveWorkKind.TASK,
                title=(
                    (
                        "Review documentation evidence for "
                        if manual_review_only
                        else "Align documentation evidence for "
                    )
                    + f"{criterion_label} [{channel}]: "
                    f"{goal.title}"
                ),
                parent_goal_id=goal_id,
                parent_objective_terms=parent_terms,
                expected_evidence_delta=delta,
                dependencies=tuple(goal.parent_goal_ids),
                predicted_files=predicted_files,
                predicted_symbols=predicted_symbols,
                validation_commands=validation,
                confidence=1.0,
                estimated_cost=max(1.0, float(len(delta))),
                novelty=1.0,
                depth=1,
                estimated_tokens=max(128, 64 * len(delta)),
                source=(
                    "completion_gate_gap_manual_review"
                    if manual_review_only
                    else "completion_gate_gap"
                ),
                source_id=instance_key,
                rationale="; ".join(delta),
                family_key=family_key,
                instance_key=instance_key,
            )
        )
    return tuple(proposals)


def _completion_decision_gap_proposal(
    *,
    repo_root: Path,
    objective_path: Path,
    goal_id: str,
    goal: Any,
    record: Mapping[str, Any],
    decision: Mapping[str, Any],
    reasons: Sequence[str],
    default_validation: Sequence[str],
) -> ObjectiveWorkProposal | None:
    """Adapt a fail-closed completion decision into one bounded gap family."""

    coverage = record.get("coverage")
    if not isinstance(coverage, Mapping):
        coverage = {}
    analysis_result = record.get("analysis_result")
    if not isinstance(analysis_result, Mapping):
        analysis_result = {}
    predicted_files = _completion_gap_precise_files(
        _completion_gap_explicit_paths(
            decision,
            record,
            coverage,
            analysis_result,
        ),
        repo_root=repo_root,
    )
    manual_review_only = not predicted_files
    namespace = _stable_completion_gap_key(
        "objective-namespace/v1",
        {"objective_path": objective_path.resolve().as_posix()},
    )
    family_key = _stable_completion_gap_key(
        "objective-family/v1",
        {
            "namespace": namespace,
            "goal_id": goal_id,
            "criterion_id": "completion-reconciliation",
            "channel": "completion-gate-decision",
        },
    )
    binding: dict[str, Any] = {}
    for surface in (decision, record, coverage):
        candidate = surface.get("binding")
        if isinstance(candidate, Mapping):
            for key, value in candidate.items():
                binding.setdefault(str(key), value)
    proof_revisions = {
        key: str(binding.get(key) or "")
        for key in (
            "objective_revision",
            "analyzer_version",
            "configuration_revision",
        )
    }
    stable_decision = _stable_completion_gap_diagnostic(
        {
            "state": decision.get("state"),
            "next_state": decision.get("next_state"),
            "reason_codes": list(reasons),
            "completion_gate": decision.get("completion_gate"),
            "validation_results": decision.get("validation_results"),
        }
    )
    instance_key = _stable_completion_gap_key(
        "objective-instance/v1",
        {
            "family_key": family_key,
            "proof_revisions": proof_revisions,
            "decision": stable_decision,
            "affected_paths": predicted_files,
        },
    )
    delta = tuple(
        dict.fromkeys(
            [
                "Reconcile the unverified completion decision with current "
                f"evidence for: {goal.title}",
                *(str(item).strip() for item in reasons if str(item).strip()),
                *(
                    (
                        "Manual review required: no precise implementation, "
                        "affected-document, or validator-source file was "
                        "authorized as an edit target.",
                    )
                    if manual_review_only
                    else ()
                ),
            ]
        )
    )
    goal_validation = tuple(
        split_csv([str(goal.fields.get("validation") or "")])
    )
    validation = _completion_gap_validation_commands(
        row=decision,
        receipt=None,
        goal_validation=goal_validation,
        default_validation=default_validation,
    )
    predicted_symbols = tuple(
        dict.fromkeys([*goal.required_evidence, "completion-reconciliation"])
    )
    if not validation or not predicted_symbols:
        return None
    parent_terms = tuple(
        dict.fromkeys(
            [
                *goal.required_evidence,
                "completion reconciliation",
                "completion-evidence alignment",
            ]
        )
    )
    return ObjectiveWorkProposal(
        kind=ObjectiveWorkKind.TASK,
        title=(
            (
                "Review completion-evidence alignment for "
                if manual_review_only
                else "Align completion evidence for decision: "
            )
            + goal.title
        ),
        parent_goal_id=goal_id,
        parent_objective_terms=parent_terms,
        expected_evidence_delta=delta,
        dependencies=tuple(goal.parent_goal_ids),
        predicted_files=predicted_files,
        predicted_symbols=predicted_symbols,
        validation_commands=validation,
        confidence=1.0,
        estimated_cost=max(1.0, float(len(delta))),
        novelty=1.0,
        depth=1,
        estimated_tokens=max(128, 64 * len(delta)),
        source=(
            "completion_gate_gap_manual_review"
            if manual_review_only
            else "completion_gate_gap"
        ),
        source_id=instance_key,
        rationale="; ".join(delta),
        family_key=family_key,
        instance_key=instance_key,
    )


def _objective_generation_board_state(
    todo_text: str,
    *,
    task_prefix: str,
) -> tuple[set[str], dict[str, int], dict[str, int]]:
    """Return active, completed, and blocked family counts from a todo board."""

    prefix = str(task_prefix or DEFAULT_TASK_PREFIX).strip()
    if prefix.startswith("## "):
        prefix = prefix[3:].strip()
    blocks = re.split(r"(?=^##\s+)", str(todo_text or ""), flags=re.MULTILINE)
    active: set[str] = set()
    completed_counts: dict[str, int] = {}
    blocked_counts: dict[str, int] = {}
    for block in blocks:
        header = block.splitlines()[0] if block.splitlines() else ""
        if not header.startswith(f"## {prefix}"):
            continue
        merge_match = re.search(
            r"^- Merge key:\s*(.+?)\s*$",
            block,
            flags=re.MULTILINE,
        )
        if not merge_match or not merge_match.group(1).strip():
            continue
        merge_key = merge_match.group(1).strip()
        status_match = re.search(r"^- Status:\s*(.+?)\s*$", block, flags=re.MULTILINE)
        status = (
            " ".join(status_match.group(1).strip().lower().split())
            if status_match
            else ""
        )
        if status == "completed":
            completed_counts[merge_key] = completed_counts.get(merge_key, 0) + 1
            continue
        if status == "blocked":
            blocked_counts[merge_key] = blocked_counts.get(merge_key, 0) + 1
            continue
        active.add(merge_key)
    return active, completed_counts, blocked_counts


def _active_objective_generation_keys(
    todo_text: str,
    *,
    task_prefix: str,
) -> set[str]:
    active, _completed_counts, _blocked_counts = _objective_generation_board_state(
        todo_text,
        task_prefix=task_prefix,
    )
    return active


def active_objective_generation_work(
    todo_text: str,
    work_items: Iterable[Mapping[str, Any]],
    *,
    task_prefix: str = DEFAULT_TASK_PREFIX,
) -> list[dict[str, Any]]:
    """Return generated task records which still have an active board task."""

    active_keys = _active_objective_generation_keys(
        todo_text,
        task_prefix=task_prefix,
    )
    active: list[dict[str, Any]] = []
    for raw in work_items:
        item = dict(raw)
        identity = str(item.get("family_key") or item.get("semantic_key") or "")
        if identity in active_keys:
            active.append(item)
    return active


def blocked_review_objective_generation_families(
    gap_family_states: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return unresolved families occupying durable manual-review capacity."""

    return tuple(
        sorted(
            str(family_key)
            for family_key, state in gap_family_states.items()
            if state.get("resolved") is not True
            and str(state.get("outcome") or "") == "blocked_review"
        )
    )


def objective_generation_proposals(
    *,
    objective_path: Path,
    repo_root: Path | None = None,
    completion_gate_records: Mapping[str, Mapping[str, Any]] | None = None,
    completion_decisions: Mapping[str, Mapping[str, Any]] | None = None,
    analysis_escalation: Mapping[str, Any] | None = None,
    default_validation: Sequence[str] = (),
    estimated_router_tokens: int = 0,
    router_retry_count: int = 0,
    objective_terms: Sequence[str] = (),
    trust_recorded_external_completion: bool = True,
) -> tuple[Any, ...]:
    """Collect deterministic coverage and routed-analysis work candidates."""

    from .objective_graph import ObjectiveWorkProposal, parse_goal_heap
    from ..planning.plan_evaluator import AnalysisProposal
    from ..planning.task_proposal_router import analysis_proposals_to_objective_work

    resolved_repo_root = (
        repo_root.resolve()
        if repo_root is not None
        else objective_path.resolve().parent
    )
    goals = (
        parse_goal_heap(objective_path.read_text(encoding="utf-8", errors="replace"))
        if objective_path.exists()
        else []
    )
    goals_by_id = {str(goal.goal_id): goal for goal in goals}
    _external_goal_ids, external_blocked_goal_ids = (
        external_authority_goal_fence(
            goals,
            trust_recorded_completion=trust_recorded_external_completion,
        )
    )
    default_parent = next(
        (
            str(goal.goal_id)
            for goal in goals
            if getattr(goal, "is_schedulable", False)
            and str(goal.goal_id) not in external_blocked_goal_ids
        ),
        "objective-analysis",
    )
    proposals: list[Any] = []
    typed_gap_goal_ids: set[str] = set()
    gates = completion_gate_records or {}
    for goal_id in sorted(str(item) for item in gates):
        if goal_id in external_blocked_goal_ids:
            continue
        record = gates.get(goal_id) or {}
        coverage = record.get("coverage")
        if not isinstance(coverage, Mapping):
            coverage = {}
        contradictions = record.get("contradictions", record.get("contradiction_receipts", ()))
        if not isinstance(contradictions, Sequence) or isinstance(contradictions, (str, bytes)):
            contradictions = ()
        goal = goals_by_id.get(goal_id)
        typed: tuple[ObjectiveWorkProposal, ...] = ()
        if goal is not None:
            typed = _documentation_completion_gap_proposals(
                repo_root=resolved_repo_root,
                objective_path=objective_path,
                goal_id=goal_id,
                goal=goal,
                record=record,
                default_validation=default_validation,
            )
            if typed:
                typed_gap_goal_ids.add(goal_id)
                proposals.extend(typed)
        if (
            not typed
            and coverage
            and coverage.get("verified") is not True
            and goal is not None
        ):
            coverage_reasons: list[str] = []
            coverage_reasons.extend(
                _completion_gap_strings(record.get("reason_codes", ()))
            )
            coverage_reasons.extend(
                _completion_gap_strings(coverage.get("reason_codes", ()))
            )
            for contradiction in contradictions:
                if isinstance(contradiction, Mapping):
                    coverage_reasons.extend(
                        _completion_gap_strings(
                            contradiction.get("reason_codes", ())
                        )
                    )
            reasons = tuple(
                dict.fromkeys(
                    coverage_reasons
                    or ("completion_gate_coverage_unverified",)
                )
            )
            fallback = _completion_decision_gap_proposal(
                repo_root=resolved_repo_root,
                objective_path=objective_path,
                goal_id=goal_id,
                goal=goal,
                record=record,
                decision=coverage,
                reasons=reasons,
                default_validation=default_validation,
            )
            if fallback is not None:
                typed_gap_goal_ids.add(goal_id)
                proposals.append(fallback)

    # Completion reconciliation remains fail-closed when a rich coverage map
    # is unavailable. Its actionable reasons use the same stable family
    # lifecycle and explicit edit-target boundary as typed coverage gaps.
    for goal_id in sorted(str(item) for item in (completion_decisions or {})):
        if goal_id in typed_gap_goal_ids:
            continue
        decision = (completion_decisions or {}).get(goal_id) or {}
        if decision.get("verified") is True:
            continue
        goal = goals_by_id.get(goal_id)
        if goal is None or goal_id in external_blocked_goal_ids:
            continue
        reasons = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in (
                    decision.get("actionable_reasons")
                    or decision.get("reason_codes")
                    or ()
                )
                if str(item).strip()
            )
        )
        if not reasons:
            continue
        fallback = _completion_decision_gap_proposal(
            repo_root=resolved_repo_root,
            objective_path=objective_path,
            goal_id=goal_id,
            goal=goal,
            record=gates.get(goal_id) or {},
            decision=decision,
            reasons=reasons,
            default_validation=default_validation,
        )
        if fallback is not None:
            proposals.append(fallback)

    escalation = dict(analysis_escalation or {})
    raw_analysis = escalation.get("proposals", ())
    if not isinstance(raw_analysis, Sequence) or isinstance(raw_analysis, (str, bytes)):
        raw_analysis = ()
    routed: list[AnalysisProposal] = []
    direct: list[Mapping[str, Any]] = []
    for raw in raw_analysis:
        if not isinstance(raw, Mapping):
            continue
        if isinstance(raw.get("branch"), Mapping):
            try:
                routed.append(AnalysisProposal.from_dict(raw))
            except (TypeError, ValueError):
                logger.warning("Ignoring malformed routed objective proposal")
        else:
            direct.append(raw)
    per_routed_tokens = (
        max(1, int(estimated_router_tokens) // len(routed)) if routed and estimated_router_tokens else 0
    )
    if routed:
        proposals.extend(
            analysis_proposals_to_objective_work(
                routed,
                parent_goal_id=default_parent,
                depth=1,
                estimated_tokens=per_routed_tokens,
                retry_count=max(0, int(router_retry_count)),
            )
        )
    # Static/AST escalation findings use the same durable record even though
    # they do not pass through the AnalysisProposal provider schema.
    for raw in direct:
        title = str(raw.get("summary") or raw.get("title") or "").strip()
        path = str(raw.get("root_relative_path") or raw.get("path") or "").strip()
        validation = str(raw.get("validation") or "").strip()
        if not title or not path or not validation:
            continue
        parent_goal_id = str(raw.get("goal_id") or default_parent).strip()
        dependencies = tuple(
            str(item) for item in raw.get("dependencies", ()) if str(item).strip()
        )
        if (
            parent_goal_id in external_blocked_goal_ids
            or external_blocked_goal_ids.intersection(dependencies)
        ):
            continue
        proposals.append(
            ObjectiveWorkProposal(
                kind="task",
                title=title,
                parent_goal_id=parent_goal_id,
                parent_objective_terms=tuple(
                    str(item)
                    for item in (
                        escalation.get("objective_terms", ()) or objective_terms
                    )
                    if str(item).strip()
                ) or (title,),
                expected_evidence_delta=(title,),
                dependencies=dependencies,
                predicted_files=(path,),
                predicted_symbols=(str(raw.get("kind") or "codebase_finding"),),
                validation_commands=(validation,),
                confidence=1.0,
                estimated_cost=1.0,
                novelty=1.0,
                depth=1,
                source="deterministic_analysis",
                source_id=str(raw.get("fingerprint") or ""),
                rationale=str(raw.get("snippet") or "static analysis finding"),
            )
        )
    return tuple(
        proposal
        for proposal in proposals
        if str(getattr(proposal, "parent_goal_id", "") or "")
        not in external_blocked_goal_ids
        and not external_blocked_goal_ids.intersection(
            str(item)
            for item in (getattr(proposal, "dependencies", ()) or ())
            if str(item).strip()
        )
    )


def objective_generation_task_findings(
    work_items: Iterable[Mapping[str, Any]],
    *,
    repo_root: Path,
    objective_path: Path,
    generation_path: Path,
    seen_fingerprints: Iterable[str] = (),
    open_goal_ids: Iterable[str] = (),
    gap_family_states: Mapping[str, Mapping[str, Any]] | None = None,
    trust_recorded_external_completion: bool = True,
) -> tuple[ObjectiveFinding, ...]:
    """Convert independent bounded task proposals into taskboard findings.

    Goal/subgoal proposals and tasks whose parent is another generated work
    item stay in the durable generation ledger until their hierarchy can be
    resolved.  A task attached directly to an actionable objective goal can
    use the ordinary discovery and bundle machinery immediately.
    """

    goals = (
        parse_goal_heap(objective_path.read_text(encoding="utf-8", errors="replace"))
        if objective_path.exists()
        else []
    )
    goals_by_id = {goal.goal_id: goal for goal in goals}
    _external_goal_ids, external_blocked_goal_ids = (
        external_authority_goal_fence(
            goals,
            trust_recorded_completion=trust_recorded_external_completion,
        )
    )
    seen = {str(item) for item in seen_fingerprints if str(item).strip()}
    open_goals = {str(item) for item in open_goal_ids if str(item).strip()}
    objective_relative = repo_relative_path(repo_root, objective_path)
    generation_relative = repo_relative_path(repo_root, generation_path)
    findings: list[ObjectiveFinding] = []
    family_states = gap_family_states or {}

    for raw in work_items:
        try:
            proposal = ObjectiveWorkProposal.from_dict(raw)
        except (TypeError, ValueError):
            logger.warning("Ignoring malformed persisted objective work proposal")
            continue
        if proposal.kind is not ObjectiveWorkKind.TASK:
            continue
        goal = goals_by_id.get(proposal.parent_goal_id)
        if (
            goal is None
            or not goal.is_schedulable
            or goal.goal_id in open_goals
            or goal.goal_id in external_blocked_goal_ids
            or external_blocked_goal_ids.intersection(proposal.dependencies)
        ):
            continue
        if proposal.family_key:
            state = family_states.get(proposal.family_key, {})
            if (
                state.get("resolved") is True
                or str(state.get("outcome") or "") == "blocked_review"
                or str(state.get("instance_key") or "") != proposal.instance_key
                or str(state.get("canonical_id") or "") != proposal.canonical_id
            ):
                continue
            occurrence = max(1, int(state.get("occurrence", 1) or 1))
            attempt_count = max(1, int(state.get("attempt_count", 1) or 1))
            outcome = str(state.get("outcome") or "actionable")
            fingerprint_material = (
                f"{proposal.family_key}\0{proposal.instance_key}\0"
                f"{occurrence}\0{attempt_count}\0{outcome}"
            )
        else:
            fingerprint_material = proposal.semantic_key
        fingerprint = sha1(fingerprint_material.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            continue

        missing_evidence = list(
            proposal.expected_evidence_delta or proposal.parent_objective_terms
        )
        outputs = list(proposal.predicted_files)
        validation = "; ".join(proposal.validation_commands) or "git diff --check"
        bundle_key = goal.bundle_key(missing_evidence)
        dependency_note = ""
        if proposal.dependencies:
            dependency_note = (
                " Preserve the generated planning dependencies in the implementation plan: "
                + ", ".join(proposal.dependencies)
                + "."
            )
        findings.append(
            ObjectiveFinding(
                fingerprint=fingerprint,
                goal_id=goal.goal_id,
                title=goal.title,
                summary=proposal.title,
                priority=str(goal.fields.get("priority") or "P1").strip().upper(),
                track=str(goal.fields.get("track") or "objective").strip().lower(),
                missing_evidence=missing_evidence,
                present_evidence={},
                evidence_methods=[
                    "bounded_objective_generation",
                    str(proposal.source or "deterministic"),
                ],
                objective_path=objective_relative,
                outputs=outputs,
                validation=validation,
                goal=str(goal.fields.get("goal") or goal.title),
                refinement=(
                    "Keep the parent goal actionable until fresh proof receipts "
                    "satisfy its completion gate."
                ),
                gap_task=(proposal.rationale or proposal.title) + dependency_note,
                parent_goal_ids=list(goal.parent_goal_ids),
                graph_depth=max(0, int(proposal.depth)),
                bundle_key=bundle_key,
                parallel_lane=bundle_key,
                bundle_strategy="bounded_objective_generation",
                embedding_query=" ".join(proposal.parent_objective_terms),
                ast_query=", ".join(proposal.predicted_symbols),
                candidate_kind="generated_task",
                surplus_group=goal.goal_id,
                merge_key=proposal.family_key or proposal.semantic_key,
                merge_family=goal.goal_id,
                merge_role=str(proposal.source or "generated_task"),
                work_item_count=max(1, len(missing_evidence)),
                work_scope="bounded_objective_generation",
                todo_vector_key=proposal.semantic_key.rsplit("/", 1)[-1][:16],
                predicted_files=outputs,
                ast_symbols=list(proposal.predicted_symbols),
                generated_artifacts=[generation_relative],
                dedupe_key=proposal.semantic_key,
            )
        )
        seen.add(fingerprint)
    return tuple(findings)


def run_objective_analysis_escalation(
    *,
    repo_root: Path,
    objective_path: Path,
    healthy_backlog_count: int,
    objective_terms: Sequence[str] = (),
    artifact_path: Path | None = None,
    policy: Any = None,
    analysis_pipeline: Any = None,
    analysis_cache_path: Path | None = None,
    analysis_provider: Any = None,
    **kwargs: Any,
) -> Any:
    """Production bridge from objective state to the read-only analysis policy."""

    from ..analysis.audit_scanner import run_low_backlog_analysis

    if analysis_pipeline is None and (
        analysis_cache_path is not None or artifact_path is not None
    ):
        from ..analysis.analysis_cache import AnalysisCache
        from ..analysis.analysis_pipeline import AnalysisPipeline, make_analysis_stage_receipt

        cache_path = Path(
            analysis_cache_path
            or (Path(artifact_path).parent / "analysis_cache")
        )

        def objective_pipeline_analyzer(context: Any) -> Any:
            return make_analysis_stage_receipt(
                context.request,
                successful=True,
                reason_code="bounded_objective_analysis_complete",
            )

        analysis_pipeline = AnalysisPipeline(
            AnalysisCache(cache_path),
            objective_pipeline_analyzer,
            provider=analysis_provider,
        )
    terms = tuple(objective_terms) or objective_terms_for_analysis(objective_path)
    result = run_low_backlog_analysis(
        repo_root,
        objective_path=objective_path,
        healthy_backlog_count=healthy_backlog_count,
        objective_terms=terms,
        policy=policy,
        analysis_pipeline=analysis_pipeline,
        **kwargs,
    )
    if artifact_path is not None:
        persist_analysis_escalation(artifact_path, result)
    return result


def default_repo_root() -> Path:
    return Path.cwd()


def default_objective_path(repo_root: Path) -> Path:
    return repo_root / "implementation_plan" / "docs" / "23-virtual-ai-os-objective-goal-heap.md"


def default_todo_path(repo_root: Path) -> Path:
    return repo_root / "docs" / "AGENT_OBJECTIVE_TODO.md"


def default_state_root(repo_root: Path) -> Path:
    return repo_root / "data" / "agent_supervisor"


def split_csv(values: Iterable[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        for raw in value.split(","):
            item = " ".join(raw.strip().split())
            if item:
                items.append(item)
    return items


def parse_goal_completion_todo_boards(
    specs: Iterable[str],
    *,
    repo_root: Path,
    default_task_prefix: str,
) -> list[tuple[Path, str]]:
    """Parse extra objective-completion board specs as ``path::task-prefix``."""

    boards: list[tuple[Path, str]] = []
    for raw_spec in specs:
        spec = str(raw_spec or "").strip()
        if not spec:
            continue
        if "::" in spec:
            raw_path, raw_prefix = spec.split("::", 1)
            prefix = raw_prefix.strip() or default_task_prefix
        else:
            raw_path = spec
            prefix = default_task_prefix
        path = Path(raw_path.strip())
        if not path.is_absolute():
            path = repo_root / path
        boards.append((path.resolve(), prefix))
    return boards


def discovery_fingerprints(discovery_dir: Path) -> set[str]:
    """Return previously filed objective-gap fingerprints from discovery files."""

    if not discovery_dir.exists():
        return set()
    fingerprints: set[str] = set()
    pattern = re.compile(r"^Fingerprint:\s*(\S+)\s*$", flags=re.MULTILINE)
    for path in discovery_dir.rglob("*objective-gap*.md"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fingerprints.update(match.group(1) for match in pattern.finditer(text))
    return fingerprints


def _validate_completion_gate_edit_targets(
    value: Any,
    *,
    goal_id: str,
    repo_root: Path | None,
    location: str = "record",
) -> None:
    """Fail closed on unsafe scoped task fields in one gate record."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            nested_location = f"{location}.{key}"
            if key == "validation_commands":
                if not isinstance(nested, Sequence) or isinstance(
                    nested, (str, bytes)
                ):
                    raise ValueError(
                        "goal completion gate record "
                        f"{goal_id!r} field {nested_location!r} must be a "
                        "sequence of strings"
                    )
                if len(nested) > 8:
                    raise ValueError(
                        "goal completion gate record "
                        f"{goal_id!r} field {nested_location!r} contains too "
                        "many validation commands"
                    )
                for command in nested:
                    if (
                        not isinstance(command, str)
                        or not command.strip()
                        or command != command.strip()
                        or len(command) > 512
                        or any(marker in command for marker in ("\x00", "\r", "\n"))
                    ):
                        raise ValueError(
                            "goal completion gate record "
                            f"{goal_id!r} field {nested_location!r} contains "
                            "an invalid validation command"
                        )
                continue
            if key in _COMPLETION_GAP_EDIT_TARGET_FIELDS:
                if nested in (None, "", (), []):
                    continue
                if repo_root is None:
                    raise ValueError(
                        "repo_root is required to validate goal completion "
                        f"gate edit target field {nested_location!r} for "
                        f"{goal_id!r}"
                    )
                if isinstance(nested, str):
                    raw_targets: Sequence[Any] = (nested,)
                elif isinstance(nested, Sequence) and not isinstance(
                    nested, (str, bytes)
                ):
                    raw_targets = nested
                else:
                    raise ValueError(
                        "goal completion gate record "
                        f"{goal_id!r} field {nested_location!r} must be a "
                        "string or a sequence of strings"
                    )
                if any(not isinstance(item, str) for item in raw_targets):
                    raise ValueError(
                        "goal completion gate record "
                        f"{goal_id!r} field {nested_location!r} must contain "
                        "only strings"
                    )
                targets = _completion_gap_paths(nested)
                for target in targets:
                    if not _completion_gap_precise_files(
                        (target,),
                        repo_root=repo_root,
                    ):
                        raise ValueError(
                            "goal completion gate record "
                            f"{goal_id!r} field {nested_location!r} contains "
                            f"an unsafe or imprecise edit target: {target!r}"
                        )
                continue
            _validate_completion_gate_edit_targets(
                nested,
                goal_id=goal_id,
                repo_root=repo_root,
                location=nested_location,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _validate_completion_gate_edit_targets(
                nested,
                goal_id=goal_id,
                repo_root=repo_root,
                location=f"{location}[{index}]",
            )


def load_goal_completion_gate_records(
    path: Path | None,
    *,
    repo_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load gate records; edit-target-bearing records require ``repo_root``."""

    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"goal completion gate artifact does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("goal completion gate artifact must be a JSON object")
    binding = payload.get("binding")
    if binding is not None and not isinstance(binding, Mapping):
        raise ValueError("goal completion gate artifact 'binding' must be an object")
    if binding is not None and not isinstance(payload.get("goals"), Mapping):
        raise ValueError(
            "goal completion gate artifact with a binding must contain a 'goals' object"
        )
    raw = payload.get("goals", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("goal completion gate artifact 'goals' must be an object")
    validation_root = repo_root.resolve() if repo_root is not None else None
    records: dict[str, dict[str, Any]] = {}
    for goal_id, record in raw.items():
        normalized_goal_id = str(goal_id).strip()
        if not normalized_goal_id:
            raise ValueError("goal completion gate artifact contains an empty goal id")
        if not isinstance(record, Mapping):
            raise ValueError(
                f"goal completion gate record for {normalized_goal_id!r} must be an object"
            )
        normalized = dict(record)
        _validate_completion_gate_edit_targets(
            normalized,
            goal_id=normalized_goal_id,
            repo_root=validation_root,
        )
        supplied_binding = normalized.get("binding")
        if supplied_binding is not None and not isinstance(supplied_binding, Mapping):
            raise ValueError(
                f"goal completion gate record {normalized_goal_id!r} field "
                "'binding' must be an object"
            )
        quorum_value = normalized.get("exhaustion_quorum")
        quorum_binding = (
            quorum_value.get("binding")
            if isinstance(quorum_value, Mapping)
            and isinstance(quorum_value.get("binding"), Mapping)
            else None
        )
        # Prefer the binding nearest the goal.  A combined multi-goal bundle
        # cannot truthfully put one per-goal objective revision in its envelope,
        # while the quorum is necessarily scoped to exactly one goal.
        effective_binding = supplied_binding or quorum_binding or binding
        if isinstance(effective_binding, Mapping):
            normalized.setdefault("binding", dict(effective_binding))
            for field_name in ("coverage", "analyzer_health"):
                surface = normalized.get(field_name)
                if isinstance(surface, Mapping):
                    surface = dict(surface)
                    surface.setdefault("binding", dict(effective_binding))
                    normalized[field_name] = surface
            quorum = normalized.get("exhaustion_quorum")
            if isinstance(quorum, Mapping):
                quorum = dict(quorum)
                quorum.setdefault("binding", dict(effective_binding))
                normalized["exhaustion_quorum"] = quorum
        for field_name in ("coverage", "analyzer_health", "exhaustion_quorum", "analysis_result"):
            value = normalized.get(field_name)
            if value is not None and not isinstance(value, Mapping):
                raise ValueError(
                    f"goal completion gate record {normalized_goal_id!r} field "
                    f"{field_name!r} must be an object"
                )
        child_goals = normalized.get("child_goals")
        if child_goals is not None and (
            not isinstance(child_goals, list)
            or any(not isinstance(item, Mapping) for item in child_goals)
        ):
            raise ValueError(
                f"goal completion gate record {normalized_goal_id!r} field "
                "'child_goals' must be a list of objects"
            )
        required_child_goal_ids = normalized.get("required_child_goal_ids")
        if required_child_goal_ids is not None:
            if (
                not isinstance(required_child_goal_ids, list)
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in required_child_goal_ids
                )
            ):
                raise ValueError(
                    f"goal completion gate record {normalized_goal_id!r} field "
                    "'required_child_goal_ids' must be a list of non-empty strings"
                )
            normalized_child_goal_ids = [
                item.strip() for item in required_child_goal_ids
            ]
            if len(set(normalized_child_goal_ids)) != len(
                normalized_child_goal_ids
            ):
                raise ValueError(
                    f"goal completion gate record {normalized_goal_id!r} field "
                    "'required_child_goal_ids' must contain unique strings"
                )
            normalized["required_child_goal_ids"] = normalized_child_goal_ids
        if "analysis_inconclusive" in normalized and not isinstance(
            normalized["analysis_inconclusive"], bool
        ):
            raise ValueError(
                f"goal completion gate record {normalized_goal_id!r} field "
                "'analysis_inconclusive' must be a boolean"
            )
        records[normalized_goal_id] = normalized
    return records


def load_goal_completion_evidence_records(
    path: Path | None,
) -> dict[str, list[CompletionEvidence]]:
    """Load canonical external evidence records indexed by objective goal.

    Repository/tree identity may be shared by the envelope.  Goal-specific
    objective, analyzer, and configuration revisions may live in a per-goal
    binding so one artifact can safely carry evidence for several goals.
    Individual records may repeat those fields, but an explicit mismatch is
    retained and rejected by reconciliation rather than silently overwritten.
    """

    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"goal completion evidence artifact does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("goal completion evidence artifact must be a JSON object")
    schema = str(payload.get("schema") or "")
    if schema and schema != OBJECTIVE_COMPLETION_EVIDENCE_ARTIFACT_SCHEMA:
        raise ValueError(f"unsupported goal completion evidence artifact schema: {schema}")
    binding = payload.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("goal completion evidence artifact 'binding' must be an object")
    missing_envelope_binding = [
        field_name
        for field_name in ("repository_id", "tree_id")
        if not str(binding.get(field_name) or "").strip()
    ]
    if missing_envelope_binding:
        raise ValueError(
            "goal completion evidence artifact binding is missing: "
            + ", ".join(missing_envelope_binding)
        )
    raw = payload.get("goals")
    if not isinstance(raw, Mapping):
        raise ValueError("goal completion evidence artifact 'goals' must be an object")
    records: dict[str, list[CompletionEvidence]] = {}
    for goal_id, raw_goal_value in raw.items():
        normalized_goal_id = str(goal_id).strip()
        if not normalized_goal_id:
            raise ValueError("goal completion evidence artifact contains an empty goal id")
        goal_binding: Mapping[str, Any] = {}
        if isinstance(raw_goal_value, Mapping):
            supplied_goal_binding = raw_goal_value.get("binding")
            if not isinstance(supplied_goal_binding, Mapping):
                raise ValueError(
                    f"goal completion evidence for {normalized_goal_id!r} "
                    "must contain a binding object"
                )
            goal_binding = supplied_goal_binding
            for field_name in (
                "repository_id",
                "tree_id",
                "objective_revision",
                "analyzer_version",
                "configuration_revision",
            ):
                envelope_value = binding.get(
                    field_name,
                    binding.get("configuration_id")
                    if field_name == "configuration_revision"
                    else "",
                )
                goal_value = goal_binding.get(
                    field_name,
                    goal_binding.get("configuration_id")
                    if field_name == "configuration_revision"
                    else "",
                )
                if (
                    str(envelope_value or "").strip()
                    and str(goal_value or "").strip()
                    and str(envelope_value) != str(goal_value)
                ):
                    raise ValueError(
                        f"goal completion evidence for {normalized_goal_id!r} "
                        f"has conflicting {field_name} bindings"
                    )
            has_canonical_records = "completion_evidence_records" in raw_goal_value
            has_short_records = "records" in raw_goal_value
            if has_canonical_records == has_short_records:
                raise ValueError(
                    f"goal completion evidence for {normalized_goal_id!r} must contain "
                    "exactly one of 'completion_evidence_records' or 'records'"
                )
            raw_records = raw_goal_value.get(
                "completion_evidence_records"
                if has_canonical_records
                else "records"
            )
        else:
            raw_records = raw_goal_value
        if not isinstance(raw_records, list):
            raise ValueError(
                f"goal completion evidence for {normalized_goal_id!r} records must be a list"
            )
        typed_records: list[CompletionEvidence] = []
        for index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, Mapping):
                raise ValueError(
                    f"goal completion evidence {normalized_goal_id!r}[{index}] "
                    "must be an object"
                )
            record = dict(raw_record)
            bound_tree = goal_binding.get("tree_id", binding.get("tree_id"))
            record.setdefault(
                "repository_id",
                goal_binding.get("repository_id", binding.get("repository_id")),
            )
            record.setdefault(
                "repository_tree",
                record.get("tree_id", bound_tree),
            )
            record.setdefault(
                "tree_id",
                record.get("repository_tree", bound_tree),
            )
            record.setdefault(
                "objective_revision",
                record.get(
                    "objective_id",
                    goal_binding.get(
                        "objective_revision",
                        binding.get("objective_revision"),
                    ),
                ),
            )
            record.setdefault(
                "analyzer_version",
                record.get(
                    "analyzer_revision",
                    goal_binding.get(
                        "analyzer_version",
                        binding.get("analyzer_version"),
                    ),
                ),
            )
            record.setdefault(
                "configuration_revision",
                record.get(
                    "configuration_id",
                    goal_binding.get(
                        "configuration_revision",
                        goal_binding.get(
                            "configuration_id",
                            binding.get(
                                "configuration_revision",
                                binding.get("configuration_id"),
                            ),
                        ),
                    ),
                ),
            )
            if (
                str(record.get("repository_tree") or "").strip()
                != str(record.get("tree_id") or "").strip()
            ):
                raise ValueError(
                    f"goal completion evidence {normalized_goal_id!r}[{index}] "
                    "has conflicting repository_tree and tree_id values"
                )
            required_record_binding = {
                "repository_id": record.get("repository_id"),
                "tree_id": record.get(
                    "repository_tree",
                    record.get("tree_id"),
                ),
                "objective_revision": record.get("objective_revision"),
                "analyzer_version": record.get("analyzer_version"),
                "configuration_revision": record.get("configuration_revision"),
            }
            missing_record_binding = [
                field_name
                for field_name, value in required_record_binding.items()
                if not str(value or "").strip()
            ]
            if missing_record_binding:
                raise ValueError(
                    f"goal completion evidence {normalized_goal_id!r}[{index}] "
                    "binding is missing: "
                    + ", ".join(missing_record_binding)
                )
            try:
                typed_records.append(CompletionEvidence.from_dict(record))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"goal completion evidence {normalized_goal_id!r}[{index}] "
                    f"is malformed: {exc}"
                ) from exc
        records[normalized_goal_id] = typed_records
    return records


def completion_evidence_records_from_gate_records(
    gate_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[CompletionEvidence]]:
    """Extract typed evidence embedded in a combined completion bundle."""

    extracted: dict[str, list[CompletionEvidence]] = {}
    for goal_id, gate_record in gate_records.items():
        if "completion_evidence_records" not in gate_record:
            continue
        raw_records = gate_record.get("completion_evidence_records")
        if not isinstance(raw_records, list):
            raise ValueError(
                f"goal completion gate record {goal_id!r} field "
                "'completion_evidence_records' must be a list"
            )
        binding = gate_record.get("binding")
        binding = binding if isinstance(binding, Mapping) else {}
        typed_records: list[CompletionEvidence] = []
        for index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, Mapping):
                raise ValueError(
                    f"goal completion gate evidence {goal_id!r}[{index}] "
                    "must be an object"
                )
            record = dict(raw_record)
            bound_tree = binding.get("tree_id")
            record.setdefault("repository_id", binding.get("repository_id"))
            record.setdefault(
                "repository_tree",
                record.get("tree_id", bound_tree),
            )
            record.setdefault(
                "tree_id",
                record.get("repository_tree", bound_tree),
            )
            record.setdefault(
                "objective_revision",
                record.get(
                    "objective_id",
                    gate_record.get(
                        "objective_revision",
                        binding.get("objective_revision"),
                    ),
                ),
            )
            record.setdefault(
                "analyzer_version",
                record.get(
                    "analyzer_revision",
                    gate_record.get(
                        "analyzer_version",
                        gate_record.get(
                            "analyzer_revision",
                            binding.get("analyzer_version"),
                        ),
                    ),
                ),
            )
            record.setdefault(
                "configuration_revision",
                record.get(
                    "configuration_id",
                    gate_record.get(
                        "configuration_revision",
                        binding.get(
                            "configuration_revision",
                            binding.get("configuration_id"),
                        ),
                    ),
                ),
            )
            if (
                str(record.get("repository_tree") or "").strip()
                != str(record.get("tree_id") or "").strip()
            ):
                raise ValueError(
                    f"goal completion gate evidence {goal_id!r}[{index}] "
                    "has conflicting repository_tree and tree_id values"
                )
            try:
                typed_records.append(CompletionEvidence.from_dict(record))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"goal completion gate evidence {goal_id!r}[{index}] "
                    f"is malformed: {exc}"
                ) from exc
        extracted[str(goal_id)] = typed_records
    return extracted


def completion_gate_receipts_from_decisions(
    decisions: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Project completion decisions into exact, machine-readable receipts.

    This deliberately rechecks the serialized shape. A producer cannot make a
    malformed or internally contradictory record pass merely by setting one
    top-level boolean.
    """

    receipts: dict[str, dict[str, Any]] = {}
    for goal_id in sorted(str(item) for item in decisions):
        raw_decision = decisions.get(goal_id)
        decision = dict(raw_decision) if isinstance(raw_decision, Mapping) else {}
        raw_gate = decision.get("completion_gate")
        gate = dict(raw_gate) if isinstance(raw_gate, Mapping) else {}
        raw_checks = gate.get("checks")
        checks = (
            [dict(item) for item in raw_checks if isinstance(item, Mapping)]
            if isinstance(raw_checks, list)
            else []
        )
        reasons = [
            str(item)
            for item in [
                *(decision.get("reason_codes") or ()),
                *(gate.get("reason_codes") or ()),
            ]
            if str(item)
        ]
        if not decision:
            reasons.append("completion_decision_malformed")
        if not gate:
            reasons.append("completion_gate_missing")
        elif not checks:
            reasons.append("completion_gate_checks_missing")
        elif any(check.get("passed") is not True for check in checks):
            reasons.append("completion_gate_check_failed")
        evaluated_evidence = gate.get("evaluated_evidence")
        if not isinstance(evaluated_evidence, Mapping):
            reasons.append("completion_gate_evidence_missing")
            evaluated_evidence = {}
        state = str(decision.get("state") or decision.get("next_state") or "")
        gate_reason_codes = [str(item) for item in gate.get("reason_codes", ()) if str(item)]
        passed = bool(
            decision.get("verified") is True
            and state == "verified_complete"
            and gate.get("passed") is True
            and checks
            and all(check.get("passed") is True for check in checks)
            and evaluated_evidence
            and not gate_reason_codes
            and not decision.get("reason_codes")
        )
        if not passed and not reasons:
            reasons.append("completion_gate_failed")
        receipts[goal_id] = {
            "schema": OBJECTIVE_COMPLETION_GATE_RECEIPT_SCHEMA,
            "goal_id": goal_id,
            "passed": passed,
            "state": state,
            "reason_codes": list(dict.fromkeys(reasons)),
            "actionable_reasons": list(
                dict.fromkeys(
                    str(item)
                    for item in [
                        *(decision.get("actionable_reasons") or ()),
                        *(gate.get("actionable_reasons") or ()),
                    ]
                    if str(item)
                )
            ),
            "checks": checks,
            "evaluated_evidence": dict(evaluated_evidence),
        }
    return receipts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate autonomous-agent todos from an objective goal heap")
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--objective-path", type=Path, default=None)
    parser.add_argument("--todo-path", type=Path, default=None)
    parser.add_argument("--discovery-dir", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--graph-path", type=Path, default=None)
    parser.add_argument(
        "--scan-exclude-path",
        action="append",
        type=Path,
        default=[],
        help=(
            "Repo-relative or absolute subtree inside --repo-root that the "
            "objective evidence and AST scanners must not read. Repeat for "
            "multiple sensitive roots."
        ),
    )
    parser.add_argument("--task-prefix", default=DEFAULT_TASK_PREFIX)
    parser.add_argument("--objective-summary-prefix", default=DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX)
    parser.add_argument("--discovery-output-path", default=DEFAULT_DISCOVERY_OUTPUT_PATH)
    parser.add_argument("--depends-on", action="append", default=[])
    parser.add_argument("--seen-fingerprint", action="append", default=[])
    parser.add_argument(
        "--force-goal-id",
        action="append",
        default=[],
        help="Objective goal id to rescan even when an existing discovery fingerprint would suppress it.",
    )
    parser.add_argument("--repeat-existing", action="store_true", help="Do not suppress fingerprints already in discovery files")
    parser.add_argument("--max-findings", type=int, default=10)
    parser.add_argument(
        "--surplus-findings-per-goal",
        type=int,
        default=DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
        help=(
            "Generate up to this many structured candidate todos per missing goal. "
            "The first candidate is the aggregate gap; additional candidates form multi-evidence batches "
            "so the vector index can bundle or merge related surplus work without creating tiny tasks."
        ),
    )
    parser.add_argument(
        "--surplus-min-terms-per-todo",
        type=int,
        default=DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
        help="Minimum missing-evidence terms for non-aggregate surplus todos when enough terms are available.",
    )
    parser.add_argument("--ensure-tracking-document", action="store_true")
    parser.add_argument("--ultimate-goal", default=DEFAULT_ULTIMATE_GOAL)
    parser.add_argument("--root-evidence", action="append", default=[])
    parser.add_argument("--goal-prefix", default=None)
    parser.add_argument("--root-goal-id", default=None)
    parser.add_argument("--root-goal-title", default=DEFAULT_ROOT_GOAL_TITLE)
    parser.add_argument("--tracking-document-title", default=DEFAULT_TRACKING_DOCUMENT_TITLE)
    parser.add_argument("--refine-objective-heap", action="store_true")
    parser.add_argument("--max-refinement-children", type=int, default=3)
    parser.add_argument("--max-refinement-depth", type=int, default=4)
    parser.add_argument(
        "--no-reconcile-goal-completion",
        action="store_true",
        help="Skip marking active goals completed when all required evidence is already present.",
    )
    parser.add_argument(
        "--objective-goal-completion-todo-board",
        action="append",
        default=[],
        help=(
            "Extra todo board that can keep objective goals open while referenced work is pending. "
            "Use 'path::TASK-' or 'path::## TASK-' and repeat for shared cross-track boards."
        ),
    )
    parser.add_argument(
        "--objective-goal-completion-gate-path",
        type=Path,
        default=None,
        help="JSON artifact containing coverage, analyzer health, exhaustion quorum, and child proof per goal.",
    )
    parser.add_argument(
        "--objective-goal-completion-evidence-path",
        type=Path,
        default=None,
        help=(
            "Canonical JSON artifact containing tree- and policy-bound "
            "CompletionEvidence records per goal."
        ),
    )
    parser.add_argument(
        "--objective-external-completion-receipt-path",
        type=Path,
        default=None,
        help=(
            "Explicit identity-only JSON authority for externally executed "
            "operational goals. The file is validated against the current "
            "clean commit, tree, and recursive gitlinks."
        ),
    )
    parser.add_argument(
        "--seed-interoperability-goals",
        action="store_true",
        help="Seed objective subgoals for cross-submodule interoperability and integration tests.",
    )
    parser.add_argument(
        "--interoperability-focus",
        action="append",
        default=[],
        help=(
            "Submodule path to pair with other submodules when seeding interoperability goals. "
            "If omitted, all submodule pairs are eligible."
        ),
    )
    parser.add_argument(
        "--interoperability-component-path",
        action="append",
        default=[],
        help="Repo-relative component path to include when seeding interoperability goals.",
    )
    parser.add_argument("--max-interoperability-goals", type=int, default=12)
    parser.add_argument(
        "--seed-launch-readiness-goals",
        action="store_true",
        help="Seed high-value launch-readiness goals for Swissknife, Hallucinate App, MCP servers, and Meta glasses.",
    )
    parser.add_argument("--max-launch-readiness-goals", type=int, default=8)
    parser.add_argument("--no-persist-ast-dataset", action="store_true")
    parser.add_argument(
        "--no-todo-vector-index",
        action="store_true",
        help="Skip writing the todo vector/AST index artifact.",
    )
    parser.add_argument(
        "--todo-vector-index-path",
        type=Path,
        default=None,
        help="Path for the todo vector/AST index artifact. Defaults to <bundle-dir>/todo_vector_index.json.",
    )
    parser.add_argument(
        "--generate-plan-branches",
        action="store_true",
        help=(
            "Generate multiple schema-validated branches for each new objective subgoal through llm_router. "
            "Without this flag the same scheduler artifact is populated by the deterministic planner."
        ),
    )
    parser.add_argument(
        "--plan-branch-count",
        type=int,
        default=3,
        help="Requested number of LLM plan alternatives per eligible subgoal (default: 3).",
    )
    parser.add_argument(
        "--plan-evaluation-path",
        type=Path,
        default=None,
        help="Selected and rejected plan artifact. Defaults to <state-root>/plan_evaluations.json.",
    )
    parser.add_argument(
        "--plan-router-provider",
        default=os.environ.get("IPFS_DATASETS_PY_LLM_PROVIDER", ""),
        help="Optional llm_router provider for structured planning.",
    )
    parser.add_argument(
        "--plan-router-model",
        default=os.environ.get("IPFS_DATASETS_PY_LLM_MODEL", "gpt-5.3-codex-spark"),
        help="llm_router model for structured planning.",
    )
    parser.add_argument("--plan-router-max-new-tokens", type=int, default=4096)
    parser.add_argument("--plan-router-timeout", type=int, default=300)
    parser.add_argument("--plan-router-temperature", type=float, default=0.1)
    parser.add_argument(
        "--plan-router-allow-local-fallback",
        action="store_true",
        help="Allow llm_router's local provider fallback before deterministic planning.",
    )
    parser.add_argument(
        "--escalate-low-backlog-analysis",
        action="store_true",
        help="Run bounded static, exhaustive AST, and llm_router analysis when the healthy backlog is below target.",
    )
    parser.add_argument("--analysis-backlog-count", type=int, default=-1)
    parser.add_argument("--analysis-backlog-target", type=int, default=5)
    parser.add_argument("--analysis-max-router-calls", type=int, default=2)
    parser.add_argument("--analysis-router-rate", type=int, default=4)
    parser.add_argument("--analysis-max-router-tokens", type=int, default=8192)
    parser.add_argument("--analysis-max-router-retries", type=int, default=1)
    parser.add_argument("--analysis-max-novel-proposals", type=int, default=5)
    parser.add_argument("--analysis-min-confidence", type=float, default=0.65)
    parser.add_argument("--analysis-min-novelty", type=float, default=0.35)
    parser.add_argument(
        "--analysis-escalation-path",
        type=Path,
        default=None,
        help="Escalation evidence artifact. Defaults to <state-root>/analysis_escalation.json.",
    )
    parser.add_argument(
        "--no-generate-bounded-work",
        action="store_true",
        help="Disable durable bounded work generation from coverage and analysis evidence.",
    )
    parser.add_argument("--objective-generation-path", type=Path, default=None)
    parser.add_argument("--objective-generation-max-depth", type=int, default=3)
    parser.add_argument("--objective-generation-max-breadth", type=int, default=4)
    parser.add_argument("--objective-generation-max-new-work", type=int, default=12)
    parser.add_argument("--objective-generation-max-open-work", type=int, default=48)
    parser.add_argument("--objective-generation-token-budget", type=int, default=8192)
    parser.add_argument("--objective-generation-max-retries", type=int, default=2)
    parser.add_argument("--objective-generation-semantic-threshold", type=float, default=0.82)
    parser.add_argument("--objective-generation-min-confidence", type=float, default=0.0)
    parser.add_argument("--objective-generation-min-novelty", type=float, default=0.0)
    parser.add_argument("--objective-generation-max-cost", type=float, default=1000000.0)
    parser.add_argument(
        "--objective-generation-current-open-work",
        type=int,
        default=-1,
        help="Override scheduler open-work count; negative derives it from active goals and generated tasks.",
    )
    parser.add_argument("--submit-bundles", action="store_true", help="Submit generated bundle shards to the local task queue")
    parser.add_argument("--queue-path", default=None)
    parser.add_argument("--queue-task-type", default="codex.todo_bundle")
    parser.add_argument("--queue-model-name", default="codex")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def run_objective_daemon(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    scan_exclude_paths = resolve_scan_exclude_paths(
        repo_root,
        getattr(args, "scan_exclude_path", ()) or (),
    )
    scan_exclude_metadata = scan_exclude_path_metadata(
        repo_root,
        scan_exclude_paths,
    )
    objective_path = (args.objective_path or default_objective_path(repo_root)).resolve()
    todo_path = (args.todo_path or default_todo_path(repo_root)).resolve()
    state_root = default_state_root(repo_root)
    discovery_dir = (args.discovery_dir or state_root / "discovery").resolve()
    bundle_dir = (args.bundle_dir or state_root / "objective_bundles").resolve()
    dataset_dir = (args.dataset_dir or state_root / "objective_datasets").resolve()
    graph_path = (getattr(args, "graph_path", None) or state_root / "objective_graph.json").resolve()
    external_completion_path = getattr(
        args,
        "objective_external_completion_receipt_path",
        None,
    )
    if external_completion_path is not None and not external_completion_path.is_absolute():
        external_completion_path = (repo_root / external_completion_path).resolve()
    external_completion_authority: ExternalCompletionAuthority | None = None
    if external_completion_path is not None:
        external_completion_authority = load_external_completion_authority(
            external_completion_path
        )
    completion_reconciliation_enabled = not bool(
        getattr(args, "no_reconcile_goal_completion", False)
    )

    seen_fingerprints = set(split_csv(args.seen_fingerprint))
    if not args.repeat_existing:
        seen_fingerprints.update(discovery_fingerprints(discovery_dir))

    tracking_created = False
    ensured_goal_ids: list[str] = []
    if getattr(args, "ensure_tracking_document", False):
        tracking = ensure_objective_tracking_document(
            objective_path,
            ultimate_goal=getattr(args, "ultimate_goal", DEFAULT_ULTIMATE_GOAL),
            root_evidence=parse_root_evidence(getattr(args, "root_evidence", [])),
            root_goal_id=getattr(args, "root_goal_id", None),
            goal_prefix=getattr(args, "goal_prefix", None) or DEFAULT_GOAL_PREFIX,
            root_goal_title=getattr(args, "root_goal_title", DEFAULT_ROOT_GOAL_TITLE),
            document_title=getattr(args, "tracking_document_title", DEFAULT_TRACKING_DOCUMENT_TITLE),
        )
        tracking_created = tracking.created
        ensured_goal_ids = tracking.appended_goal_ids

    deduplicated_interoperability_goal_ids: list[str] = []
    if objective_path.exists():
        deduplicated_interoperability_goal_ids = deduplicate_interoperability_goals(objective_path)

    seeded_interoperability_goal_ids: list[str] = []
    if getattr(args, "seed_interoperability_goals", False) and objective_path.exists():
        interoperability = append_interoperability_goals(
            objective_path,
            repo_root=repo_root,
            focus=getattr(args, "interoperability_focus", []) or (),
            component_paths=getattr(args, "interoperability_component_path", []) or (),
            max_goals=getattr(args, "max_interoperability_goals", 12),
            goal_prefix=getattr(args, "goal_prefix", None),
        )
        seeded_interoperability_goal_ids = interoperability.appended_goal_ids

    seeded_launch_readiness_goal_ids: list[str] = []
    if getattr(args, "seed_launch_readiness_goals", False) and objective_path.exists():
        launch_readiness = append_launch_readiness_goals(
            objective_path,
            repo_root=repo_root,
            max_goals=getattr(args, "max_launch_readiness_goals", 8),
            goal_prefix=getattr(args, "goal_prefix", None),
        )
        seeded_launch_readiness_goal_ids = launch_readiness.appended_goal_ids

    completed_goal_ids: list[str] = []
    objective_completed_goal_count = 0
    objective_completion_validation_results: dict[str, Any] = {}
    objective_completion_decisions: dict[str, Any] = {}
    evidence_repository_tree = ""
    external_completion_results: dict[str, Any] = {}
    completion_gate_path = getattr(args, "objective_goal_completion_gate_path", None)
    if completion_gate_path is not None and not completion_gate_path.is_absolute():
        completion_gate_path = (repo_root / completion_gate_path).resolve()
    completion_gate_records = load_goal_completion_gate_records(
        completion_gate_path,
        repo_root=repo_root,
    )
    completion_evidence_path = getattr(
        args,
        "objective_goal_completion_evidence_path",
        None,
    )
    if completion_evidence_path is not None and not completion_evidence_path.is_absolute():
        completion_evidence_path = (repo_root / completion_evidence_path).resolve()
    embedded_completion_evidence_records = (
        completion_evidence_records_from_gate_records(completion_gate_records)
    )
    completion_evidence_records = load_goal_completion_evidence_records(
        completion_evidence_path
    )
    duplicate_evidence_goal_ids = sorted(
        set(embedded_completion_evidence_records) & set(completion_evidence_records)
    )
    if duplicate_evidence_goal_ids:
        raise ValueError(
            "completion evidence is supplied by both gate and evidence artifacts "
            "for goals: " + ", ".join(duplicate_evidence_goal_ids)
        )
    completion_evidence_records = {
        **embedded_completion_evidence_records,
        **completion_evidence_records,
    }
    completion_control_paths = [
        path
        for path in (completion_gate_path, completion_evidence_path)
        if path is not None
    ]
    require_artifact_binding = bool(completion_control_paths)
    goal_completion_todo_boards = parse_goal_completion_todo_boards(
        getattr(args, "objective_goal_completion_todo_board", []) or [],
        repo_root=repo_root,
        default_task_prefix=args.task_prefix,
    )
    if completion_reconciliation_enabled and objective_path.exists():
        completion = reconcile_objective_goal_completion(
            repo_root=repo_root,
            objective_path=objective_path,
            todo_path=todo_path,
            task_header_prefix=args.task_prefix,
            todo_boards=goal_completion_todo_boards,
            completion_evidence_records=completion_evidence_records,
            completion_gate_records=completion_gate_records,
            completion_control_paths=completion_control_paths,
            require_artifact_binding=require_artifact_binding,
            external_completion_authority=external_completion_authority,
            scan_exclude_paths=scan_exclude_paths,
        )
        completed_goal_ids = completion.completed_goal_ids
        objective_completed_goal_count = completion.completed_goal_count
        objective_completion_validation_results = completion.validation_results
        objective_completion_decisions = completion.decisions
        evidence_repository_tree = completion_tree_identity(
            repo_root,
            objective_path=objective_path,
            scan_exclude_paths=scan_exclude_paths,
        ).tree_id
        external_completion_results = completion.external_completion

    refined_goal_ids: list[str] = []
    if getattr(args, "refine_objective_heap", False) and objective_path.exists():
        forced_refinement_goal_ids = split_csv(
            getattr(args, "force_goal_id", []) or []
        )
        refinement_findings = scan_objective_gaps(
            repo_root,
            objective_path=objective_path,
            max_findings=args.max_findings,
            seen_fingerprints=seen_fingerprints,
            force_goal_ids=forced_refinement_goal_ids,
            scope_goal_ids=forced_refinement_goal_ids,
            evidence_repository_tree=evidence_repository_tree,
            scan_exclude_paths=scan_exclude_paths,
            trust_recorded_external_completion=(
                completion_reconciliation_enabled
            ),
        )
        refinement = append_refinement_goals(
            objective_path,
            refinement_findings,
            max_children_per_finding=getattr(args, "max_refinement_children", 3),
            max_depth=getattr(args, "max_refinement_depth", 4),
            goal_prefix=getattr(args, "goal_prefix", None),
        )
        refined_goal_ids = refinement.appended_goal_ids
        if refined_goal_ids:
            seen_fingerprints.update(finding.fingerprint for finding in refinement_findings)

    records = generate_objective_todos(
        repo_root=repo_root,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        dataset_dir=dataset_dir,
        task_prefix=args.task_prefix,
        depends_on=split_csv(args.depends_on),
        max_findings=args.max_findings,
        seen_fingerprints=seen_fingerprints,
        force_goal_ids=[
            *ensured_goal_ids,
            *seeded_interoperability_goal_ids,
            *seeded_launch_readiness_goal_ids,
            *refined_goal_ids,
            *split_csv(getattr(args, "force_goal_id", []) or []),
        ],
        persist_ast_dataset=not args.no_persist_ast_dataset,
        write_todo_vector_index=not getattr(args, "no_todo_vector_index", False),
        todo_vector_index_path=getattr(args, "todo_vector_index_path", None),
        surplus_findings_per_goal=getattr(args, "surplus_findings_per_goal", DEFAULT_SURPLUS_FINDINGS_PER_GOAL),
        surplus_min_terms_per_todo=getattr(args, "surplus_min_terms_per_todo", DEFAULT_SURPLUS_MIN_TERMS_PER_TODO),
        summary_prefix=getattr(args, "objective_summary_prefix", DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX),
        discovery_output_path=getattr(args, "discovery_output_path", DEFAULT_DISCOVERY_OUTPUT_PATH),
        evidence_repository_tree=evidence_repository_tree,
        scan_exclude_paths=scan_exclude_paths,
        trust_recorded_external_completion=completion_reconciliation_enabled,
    )
    plan_evaluation_path = (
        getattr(args, "plan_evaluation_path", None) or state_root / "plan_evaluations.json"
    ).resolve()
    router_config = None
    if bool(getattr(args, "generate_plan_branches", False)):
        from ..planning.task_proposal_router import StructuredPlanRouterConfig

        router_config = StructuredPlanRouterConfig(
            repo_root=repo_root,
            provider=str(getattr(args, "plan_router_provider", "") or "") or None,
            model=str(getattr(args, "plan_router_model", "gpt-5.3-codex-spark")),
            branch_count=max(1, int(getattr(args, "plan_branch_count", 3))),
            max_new_tokens=int(getattr(args, "plan_router_max_new_tokens", 4096)),
            timeout_seconds=int(getattr(args, "plan_router_timeout", 300)),
            allow_local_fallback=bool(getattr(args, "plan_router_allow_local_fallback", False)),
            temperature=float(getattr(args, "plan_router_temperature", 0.1)),
        )
    plan_decisions = plan_objective_records(
        records,
        branch_count=max(1, int(getattr(args, "plan_branch_count", 3))),
        router_config=router_config,
        use_llm_router=bool(getattr(args, "generate_plan_branches", False)),
    )
    persist_objective_plan_evaluations(
        plan_evaluation_path,
        plan_decisions,
        bundle_index_path=bundle_dir / "index.json",
    )
    analysis_escalation_path = (
        getattr(args, "analysis_escalation_path", None)
        or state_root / "analysis_escalation.json"
    ).resolve()
    analysis_escalation_payload: dict[str, Any] | None = None
    if bool(getattr(args, "escalate_low_backlog_analysis", False)):
        from ..analysis.analyzer_health import AnalysisEscalationPolicy
        from ..planning.task_proposal_router import StructuredPlanRouterConfig

        analysis_policy = AnalysisEscalationPolicy(
            backlog_target=int(getattr(args, "analysis_backlog_target", 5)),
            max_router_calls=int(getattr(args, "analysis_max_router_calls", 2)),
            router_calls_per_window=int(getattr(args, "analysis_router_rate", 4)),
            max_router_tokens=int(getattr(args, "analysis_max_router_tokens", 8192)),
            max_router_retries=int(getattr(args, "analysis_max_router_retries", 1)),
            max_novel_proposals=int(getattr(args, "analysis_max_novel_proposals", 5)),
            min_confidence=float(getattr(args, "analysis_min_confidence", 0.65)),
            min_novelty=float(getattr(args, "analysis_min_novelty", 0.35)),
        )
        configured_backlog = int(getattr(args, "analysis_backlog_count", -1))
        healthy_backlog_count = len(records) if configured_backlog < 0 else configured_backlog
        prior_router_call_timestamps: list[float] = []
        if analysis_escalation_path.exists():
            try:
                previous_escalation = json.loads(
                    analysis_escalation_path.read_text(encoding="utf-8")
                )
                for stage_record in previous_escalation.get("records", []):
                    if isinstance(stage_record, Mapping) and stage_record.get("stage") == "llm_router":
                        cost = stage_record.get("cost")
                        if isinstance(cost, Mapping):
                            prior_router_call_timestamps.extend(
                                float(item)
                                for item in cost.get("router_call_timestamps", [])
                            )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                prior_router_call_timestamps = []
        escalation_router_config = StructuredPlanRouterConfig(
            repo_root=repo_root,
            provider=str(getattr(args, "plan_router_provider", "") or "") or None,
            model=str(getattr(args, "plan_router_model", "gpt-5.3-codex-spark")),
            branch_count=max(1, min(
                int(getattr(args, "plan_branch_count", 3)),
                max(1, analysis_policy.max_novel_proposals),
            )),
            max_new_tokens=min(
                int(getattr(args, "plan_router_max_new_tokens", 4096)),
                analysis_policy.max_router_tokens,
            ),
            timeout_seconds=int(getattr(args, "plan_router_timeout", 300)),
            allow_local_fallback=bool(getattr(args, "plan_router_allow_local_fallback", False)),
            temperature=float(getattr(args, "plan_router_temperature", 0.1)),
        )
        escalation_result = run_objective_analysis_escalation(
            repo_root=repo_root,
            objective_path=objective_path,
            healthy_backlog_count=healthy_backlog_count,
            objective_terms=objective_terms_for_analysis(objective_path, records),
            artifact_path=analysis_escalation_path,
            analysis_cache_path=state_root / "analysis_cache",
            policy=analysis_policy,
            seen_fingerprints=seen_fingerprints,
            router_config=escalation_router_config,
            router_calls_in_window=prior_router_call_timestamps,
        )
        analysis_escalation_payload = escalation_result.to_dict()

    objective_generation_path = Path(
        getattr(args, "objective_generation_path", None)
        or state_root / "objective_generation.json"
    )
    if not objective_generation_path.is_absolute():
        objective_generation_path = repo_root / objective_generation_path
    objective_generation_path = objective_generation_path.resolve()
    objective_generation_payload: dict[str, Any] | None = None
    objective_generation_error = ""
    objective_generation_materialized_records: list[Any] = []
    if not bool(getattr(args, "no_generate_bounded_work", False)):
        from .objective_graph import ObjectiveGenerationLimits
        from ..planning.plan_evaluator import ObjectiveWorkEvaluationPolicy

        generation_terms = objective_terms_for_analysis(objective_path, records)
        reserved_router_tokens = 0
        router_retry_count = 0
        if analysis_escalation_payload:
            for stage in analysis_escalation_payload.get("records", ()):
                if isinstance(stage, Mapping) and isinstance(stage.get("cost"), Mapping):
                    reserved_router_tokens += max(
                        0, int(stage["cost"].get("reserved_tokens", 0) or 0)
                    )
                    router_retry_count = max(
                        router_retry_count,
                        int(stage["cost"].get("router_retries", 0) or 0),
                    )
        generation_candidates = objective_generation_proposals(
            objective_path=objective_path,
            repo_root=repo_root,
            completion_gate_records=completion_gate_records,
            completion_decisions=objective_completion_decisions,
            analysis_escalation=analysis_escalation_payload,
            default_validation=("git diff --check",),
            estimated_router_tokens=reserved_router_tokens,
            router_retry_count=router_retry_count,
            objective_terms=generation_terms,
            trust_recorded_external_completion=(
                completion_reconciliation_enabled
            ),
        )
        generation_limits = ObjectiveGenerationLimits(
            max_depth=int(getattr(args, "objective_generation_max_depth", 3)),
            max_breadth_per_parent=int(
                getattr(args, "objective_generation_max_breadth", 4)
            ),
            max_new_work=int(getattr(args, "objective_generation_max_new_work", 12)),
            max_open_work=int(getattr(args, "objective_generation_max_open_work", 48)),
            token_budget=int(getattr(args, "objective_generation_token_budget", 8192)),
            max_retries=int(getattr(args, "objective_generation_max_retries", 2)),
            semantic_similarity_threshold=float(
                getattr(args, "objective_generation_semantic_threshold", 0.82)
            ),
        )
        configured_open_work = int(
            getattr(args, "objective_generation_current_open_work", -1)
        )
        todo_board_text = (
            todo_path.read_text(encoding="utf-8", errors="replace")
            if todo_path.exists()
            else ""
        )
        (
            active_generation_keys,
            completed_generation_counts,
            blocked_generation_counts,
        ) = _objective_generation_board_state(
            todo_board_text,
            task_prefix=args.task_prefix,
        )
        if configured_open_work < 0:
            active_goal_count = 0
            if objective_path.exists():
                open_work_goals = parse_goal_heap(
                    objective_path.read_text(encoding="utf-8", errors="replace")
                )
                _external_goal_ids, externally_blocked_goal_ids = (
                    external_authority_goal_fence(
                        open_work_goals,
                        trust_recorded_completion=(
                            completion_reconciliation_enabled
                        ),
                    )
                )
                active_goal_count = sum(
                    1
                    for goal in open_work_goals
                    if (
                        goal.is_schedulable
                        and goal.goal_id not in externally_blocked_goal_ids
                    )
                )
            try:
                persisted_generation_payload = _load_generation_payload(
                    objective_generation_path
                )
                persisted_work = persisted_generation_payload.get(
                    "generated_work",
                    (),
                )
                persisted_generated_count = len(
                    active_objective_generation_work(
                        todo_path.read_text(encoding="utf-8", errors="replace")
                        if todo_path.exists()
                        else "",
                        persisted_work,
                        task_prefix=args.task_prefix,
                    )
                )
                persisted_blocked_review_count = len(
                    blocked_review_objective_generation_families(
                        persisted_generation_payload.get(
                            "gap_family_states",
                            {},
                        )
                    )
                )
            except (OSError, TypeError, ValueError):
                # The materialization call below reports the corrupt ledger
                # and admits no work; this count must not mask that failure.
                persisted_generated_count = 0
                persisted_blocked_review_count = 0
            configured_open_work = (
                active_goal_count
                + len(records)
                + persisted_generated_count
                + persisted_blocked_review_count
            )
        evaluation_policy = ObjectiveWorkEvaluationPolicy(
            min_confidence=float(
                getattr(args, "objective_generation_min_confidence", 0.0)
            ),
            min_novelty=float(getattr(args, "objective_generation_min_novelty", 0.0)),
            max_proposals=generation_limits.max_new_work,
            max_total_cost=float(
                getattr(args, "objective_generation_max_cost", 1000000.0)
            ),
            max_open_work=generation_limits.max_open_work,
            current_open_work=configured_open_work,
            remaining_token_budget=generation_limits.token_budget,
        )
        try:
            _, objective_generation_payload = materialize_objective_generation_cycle(
                generation_candidates,
                artifact_path=objective_generation_path,
                limits=generation_limits,
                current_open_work=configured_open_work,
                evaluation_policy=evaluation_policy,
                objective_terms=generation_terms,
                active_family_keys=active_generation_keys,
                terminal_family_counts=completed_generation_counts,
                blocked_family_counts=blocked_generation_counts,
                observed_gap_goal_ids=completion_gate_records,
            )
        except (OSError, TypeError, ValueError) as exc:
            # A corrupt identity ledger or malformed proposal must fail closed
            # for generated work without suppressing the daemon's ordinary
            # deterministic backlog scan.
            objective_generation_error = f"{type(exc).__name__}: {exc}"
            logger.error("Bounded objective generation failed closed: %s", exc)
        if objective_generation_payload is not None:
            work_items = objective_generation_payload.get("generated_work", ())
            if isinstance(work_items, list):
                generated_findings = objective_generation_task_findings(
                    work_items,
                    repo_root=repo_root,
                    objective_path=objective_path,
                    generation_path=objective_generation_path,
                    seen_fingerprints=discovery_fingerprints(discovery_dir),
                    open_goal_ids=(
                        directly_open_goal_ids_from_todo_board(
                            todo_path,
                            args.task_prefix,
                        )
                        if seeded_interoperability_goal_ids
                        else open_goal_ids_from_todo_board(
                            todo_path,
                            args.task_prefix,
                        )
                    ),
                    gap_family_states=objective_generation_payload.get(
                        "gap_family_states",
                        {},
                    ),
                    trust_recorded_external_completion=(
                        completion_reconciliation_enabled
                    ),
                )
                remaining_findings = max(0, int(args.max_findings) - len(records))
                generated_findings = generated_findings[:remaining_findings]
                if generated_findings:
                    objective_generation_materialized_records = generate_objective_todos(
                        repo_root=repo_root,
                        objective_path=objective_path,
                        todo_path=todo_path,
                        discovery_dir=discovery_dir,
                        bundle_dir=bundle_dir,
                        dataset_dir=dataset_dir,
                        task_prefix=args.task_prefix,
                        depends_on=split_csv(args.depends_on),
                        persist_ast_dataset=not args.no_persist_ast_dataset,
                        write_todo_vector_index=not getattr(args, "no_todo_vector_index", False),
                        todo_vector_index_path=getattr(args, "todo_vector_index_path", None),
                        discovery_output_path=getattr(
                            args,
                            "discovery_output_path",
                            DEFAULT_DISCOVERY_OUTPUT_PATH,
                        ),
                        precomputed_findings=generated_findings,
                        trust_recorded_external_completion=(
                            completion_reconciliation_enabled
                        ),
                    )
                    records.extend(objective_generation_materialized_records)
                    generated_plan_decisions = plan_objective_records(
                        objective_generation_materialized_records,
                        branch_count=max(1, int(getattr(args, "plan_branch_count", 3))),
                        router_config=router_config,
                        use_llm_router=bool(
                            getattr(args, "generate_plan_branches", False)
                        ),
                    )
                    plan_decisions.extend(generated_plan_decisions)
                    persist_objective_plan_evaluations(
                        plan_evaluation_path,
                        plan_decisions,
                        bundle_index_path=bundle_dir / "index.json",
                    )
    blocked_review_family_keys = blocked_review_objective_generation_families(
        (objective_generation_payload or {}).get("gap_family_states", {})
    )
    graph_payload = write_objective_graph_artifact(objective_path=objective_path, graph_path=graph_path)

    bundle_index_path = bundle_dir / "index.json"
    submitted_bundle_task_ids: list[str] = []
    if args.submit_bundles and bundle_index_path.exists():
        submitted_bundle_task_ids = submit_bundle_tasks(
            bundle_index_path,
            queue_path=args.queue_path,
            task_type=args.queue_task_type,
            model_name=args.queue_model_name,
        )

    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.objective_daemon",
        "repo_root": str(repo_root),
        "objective_path": repo_relative_path(repo_root, objective_path),
        "todo_path": repo_relative_path(repo_root, todo_path),
        "discovery_dir": repo_relative_path(repo_root, discovery_dir),
        "bundle_index_path": repo_relative_path(repo_root, bundle_index_path),
        "todo_vector_index_path": repo_relative_path(
            repo_root,
            (getattr(args, "todo_vector_index_path", None) or bundle_dir / "todo_vector_index.json").resolve(),
        ),
        "dataset_dir": repo_relative_path(repo_root, dataset_dir),
        "graph_path": repo_relative_path(repo_root, graph_path),
        "scan_exclude_paths": scan_exclude_metadata,
        "scan_exclude_path_count": len(scan_exclude_metadata),
        "source_protected_scan_policy": source_protected_scan_policy(),
        "objective_completion_reconciliation_enabled": (
            completion_reconciliation_enabled
        ),
        "recorded_external_completion_trusted_for_generation": (
            completion_reconciliation_enabled
        ),
        "objective_external_authority_declared_goal_ids": sorted(
            goal.goal_id
            for goal in parse_goal_heap(
                objective_path.read_text(encoding="utf-8")
            )
            if goal.requires_external_completion
        )
        if objective_path.exists()
        else [],
        "plan_evaluation_path": repo_relative_path(repo_root, plan_evaluation_path),
        "plan_evaluation_count": len(plan_decisions),
        "plan_router_branch_count": max(1, int(getattr(args, "plan_branch_count", 3))),
        "plan_router_enabled": bool(getattr(args, "generate_plan_branches", False)),
        "plan_router_fallback_count": sum(1 for item in plan_decisions if item.get("used_fallback")),
        "plan_router_error_count": sum(1 for item in plan_decisions if item.get("router_error")),
        "analysis_escalation_enabled": bool(getattr(args, "escalate_low_backlog_analysis", False)),
        "analysis_escalation_path": repo_relative_path(repo_root, analysis_escalation_path),
        "analysis_escalation": analysis_escalation_payload,
        "analysis_inconclusive": bool(
            analysis_escalation_payload and analysis_escalation_payload.get("analysis_inconclusive")
        ),
        "objective_generation_enabled": not bool(
            getattr(args, "no_generate_bounded_work", False)
        ),
        "objective_generation_path": repo_relative_path(
            repo_root, objective_generation_path
        ),
        "objective_generation": objective_generation_payload,
        "objective_generation_error": objective_generation_error or None,
        "objective_generated_work_count": int(
            (objective_generation_payload or {}).get("generated_work_count", 0)
        ),
        "objective_generation_materialized_count": len(
            objective_generation_materialized_records
        ),
        "objective_generation_materialized_task_ids": [
            record.task_id for record in objective_generation_materialized_records
        ],
        "objective_generation_cycle_accepted_count": len(
            ((objective_generation_payload or {}).get("last_cycle") or {}).get("accepted", ())
        ),
        "objective_generation_blocked_review_count": len(
            blocked_review_family_keys
        ),
        "objective_generation_blocked_review_family_keys": list(
            blocked_review_family_keys
        ),
        "tracking_document_created": tracking_created,
        "ensured_goal_ids": ensured_goal_ids,
        "deduplicated_interoperability_goal_ids": deduplicated_interoperability_goal_ids,
        "seeded_interoperability_goal_ids": seeded_interoperability_goal_ids,
        "seeded_launch_readiness_goal_ids": seeded_launch_readiness_goal_ids,
        "completed_goal_ids": completed_goal_ids,
        "goal_completion_todo_boards": [
            {
                "todo_path": repo_relative_path(repo_root, path),
                "task_prefix": prefix,
            }
            for path, prefix in goal_completion_todo_boards
        ],
        "objective_completion_validation_results": objective_completion_validation_results,
        "objective_completion_decisions": objective_completion_decisions,
        "objective_completion_gate_inputs": completion_gate_records,
        "objective_completion_evidence_inputs": {
            goal_id: [record.to_dict() for record in records]
            for goal_id, records in completion_evidence_records.items()
        },
        "objective_completion_gate_receipts": completion_gate_receipts_from_decisions(
            objective_completion_decisions
        ),
        "objective_external_completion": external_completion_results,
        "objective_external_completion_authority_cid": (
            external_completion_authority.authority_cid
            if external_completion_authority is not None
            else ""
        ),
        "objective_external_completion_governed_goal_ids": (
            list(external_completion_authority.governed_goal_ids)
            if external_completion_authority is not None
            else []
        ),
        "objective_goal_completion_gate_path": (
            repo_relative_path(repo_root, completion_gate_path) if completion_gate_path else ""
        ),
        "objective_goal_completion_evidence_path": (
            repo_relative_path(repo_root, completion_evidence_path)
            if completion_evidence_path
            else ""
        ),
        "refined_goal_ids": refined_goal_ids,
        "objective_goal_count": graph_payload["goal_count"],
        "objective_active_goal_count": graph_payload["active_goal_count"],
        "objective_completed_goal_count": graph_payload.get("completed_goal_count", objective_completed_goal_count),
        "objective_heap_schedule_count": len(graph_payload.get("heap_schedule") or []),
        "generated_count": len(records),
        "surplus_findings_per_goal": getattr(args, "surplus_findings_per_goal", DEFAULT_SURPLUS_FINDINGS_PER_GOAL),
        "surplus_min_terms_per_todo": getattr(args, "surplus_min_terms_per_todo", DEFAULT_SURPLUS_MIN_TERMS_PER_TODO),
        "task_ids": [record.task_id for record in records],
        "discovery_paths": [repo_relative_path(repo_root, record.discovery_path) for record in records],
        "bundle_keys": sorted({record.finding.bundle_key for record in records}),
        "submitted_bundle_task_ids": submitted_bundle_task_ids,
    }
    logger.info("Objective daemon generated %s tasks", len(records))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    payload = run_objective_daemon(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
