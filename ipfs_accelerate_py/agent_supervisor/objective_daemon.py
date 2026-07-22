"""CLI bridge for reusable objective-graph backlog generation.

This package-level entry point scans an objective heap, appends missing-evidence
tasks, persists AST records, writes parallel bundle shards, and optionally
submits those bundles to a local task queue.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .objective_graph import (
    DEFAULT_DISCOVERY_OUTPUT_PATH,
    DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    DEFAULT_TASK_PREFIX,
    generate_objective_todos,
    repo_relative_path,
    scan_objective_gaps,
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
    deduplicate_interoperability_goals,
    ensure_objective_tracking_document,
    parse_root_evidence,
    reconcile_objective_goal_completion,
    write_objective_graph_artifact,
)

logger = logging.getLogger(__name__)

OBJECTIVE_COMPLETION_GATE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.objective_daemon.completion_gate.v1"
)
OBJECTIVE_GENERATION_ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.objective_daemon.generation.v1"
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

    from .plan_evaluator import evaluate_plan_branches
    from .task_proposal_router import deterministic_plan_branches, generate_structured_plan_branches

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
    from .artifact_store import write_bundle_index_artifact

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
            # Provisional and inconclusive goals are not claimable execution
            # work, but their unproven criteria are precisely the input to
            # bounded evidence-generation.
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


def persist_objective_generation(
    path: Path,
    result: Any,
    *,
    existing_work: Iterable[Mapping[str, Any]] = (),
    evaluation: Any = None,
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
    payload = {
        "schema": OBJECTIVE_GENERATION_ARTIFACT_SCHEMA,
        "cycle_count": prior_cycle_count + 1,
        "generated_work_count": len(merged),
        "generated_work": [merged[key] for key in sorted(merged)],
        "last_cycle": cycle,
        "last_evaluation": (
            evaluation.to_dict() if hasattr(evaluation, "to_dict") else evaluation
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return payload


def materialize_objective_generation_cycle(
    proposals: Iterable[Any],
    *,
    artifact_path: Path,
    limits: Any = None,
    current_open_work: int | None = None,
    evaluation_policy: Any = None,
    objective_terms: Sequence[str] = (),
) -> tuple[Any, dict[str, Any]]:
    """Apply finite graph limits and persist canonical history for one cycle."""

    from .objective_graph import materialize_bounded_objective_work
    from .plan_evaluator import evaluate_objective_work_proposals

    existing = load_objective_generation_work(artifact_path)
    proposal_values = tuple(proposals)
    evaluation = None
    if evaluation_policy is not None:
        evaluation = evaluate_objective_work_proposals(
            proposal_values,
            policy=evaluation_policy,
            objective_terms=objective_terms,
            known_canonical_ids=(str(item.get("canonical_id") or "") for item in existing),
            known_semantic_keys=(str(item.get("semantic_key") or "") for item in existing),
        )
        proposal_values = evaluation.accepted_proposals
    result = materialize_bounded_objective_work(
        proposal_values,
        existing_work=existing,
        limits=limits,
        current_open_work=current_open_work,
    )
    return result, persist_objective_generation(
        artifact_path,
        result,
        existing_work=existing,
        evaluation=evaluation,
    )


def objective_generation_proposals(
    *,
    objective_path: Path,
    completion_gate_records: Mapping[str, Mapping[str, Any]] | None = None,
    completion_decisions: Mapping[str, Mapping[str, Any]] | None = None,
    analysis_escalation: Mapping[str, Any] | None = None,
    default_validation: Sequence[str] = (),
    estimated_router_tokens: int = 0,
    router_retry_count: int = 0,
    objective_terms: Sequence[str] = (),
) -> tuple[Any, ...]:
    """Collect deterministic coverage and routed-analysis work candidates."""

    from .goal_coverage import goal_coverage_work_seeds
    from .objective_graph import ObjectiveWorkProposal, parse_goal_heap
    from .plan_evaluator import AnalysisProposal
    from .task_proposal_router import analysis_proposals_to_objective_work

    goals = (
        parse_goal_heap(objective_path.read_text(encoding="utf-8", errors="replace"))
        if objective_path.exists()
        else []
    )
    goals_by_id = {str(goal.goal_id): goal for goal in goals}
    default_parent = next(
        (str(goal.goal_id) for goal in goals if getattr(goal, "is_schedulable", False)),
        next((str(goal.goal_id) for goal in goals), "objective-analysis"),
    )
    proposals: list[Any] = []
    gates = completion_gate_records or {}
    for goal_id in sorted(str(item) for item in gates):
        record = gates.get(goal_id) or {}
        coverage = record.get("coverage")
        if not isinstance(coverage, Mapping):
            continue
        contradictions = record.get("contradictions", record.get("contradiction_receipts", ()))
        if not isinstance(contradictions, Sequence) or isinstance(contradictions, (str, bytes)):
            contradictions = ()
        goal = goals_by_id.get(goal_id)
        proposals.extend(
            goal_coverage_work_seeds(
                coverage,
                goals=([goal] if goal is not None else ()),
                contradictions=contradictions,
                default_validation=default_validation,
            )
        )

    # Completion reconciliation remains fail-closed when a rich coverage map
    # is unavailable.  Its actionable reasons still become a fully linked
    # evidence task instead of allowing provisional/inconclusive work to
    # disappear from the scheduler.
    for goal_id in sorted(str(item) for item in (completion_decisions or {})):
        decision = (completion_decisions or {}).get(goal_id) or {}
        if decision.get("verified") is True:
            continue
        goal = goals_by_id.get(goal_id)
        if goal is None:
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
        fields = goal.fields
        predicted_files = tuple(split_csv([str(fields.get("outputs") or "")]))
        predicted_symbols = tuple(goal.required_evidence)
        validation = tuple(split_csv([str(fields.get("validation") or "")]))
        if not predicted_files or not predicted_symbols or not validation:
            continue
        proposals.append(
            ObjectiveWorkProposal(
                kind="task",
                title=f"Produce completion evidence for {goal.title}",
                parent_goal_id=goal_id,
                parent_objective_terms=tuple(goal.required_evidence),
                expected_evidence_delta=reasons,
                dependencies=tuple(goal.parent_goal_ids),
                predicted_files=predicted_files,
                predicted_symbols=predicted_symbols,
                validation_commands=validation,
                confidence=1.0,
                estimated_cost=max(1.0, float(len(reasons))),
                novelty=1.0,
                depth=1,
                estimated_tokens=128,
                source="completion_gate",
                source_id=f"{goal_id}:{decision.get('state') or decision.get('next_state') or 'unverified'}",
                rationale="; ".join(reasons),
            )
        )

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
                dependencies=tuple(str(item) for item in raw.get("dependencies", ()) if str(item).strip()),
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
    return tuple(proposals)


def run_objective_analysis_escalation(
    *,
    repo_root: Path,
    objective_path: Path,
    healthy_backlog_count: int,
    objective_terms: Sequence[str] = (),
    artifact_path: Path | None = None,
    policy: Any = None,
    **kwargs: Any,
) -> Any:
    """Production bridge from objective state to the read-only analysis policy."""

    from .audit_scanner import run_low_backlog_analysis

    terms = tuple(objective_terms) or objective_terms_for_analysis(objective_path)
    result = run_low_backlog_analysis(
        repo_root,
        objective_path=objective_path,
        healthy_backlog_count=healthy_backlog_count,
        objective_terms=terms,
        policy=policy,
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


def load_goal_completion_gate_records(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load a persisted per-goal gate artifact, failing closed on bad shapes."""

    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("goal completion gate artifact must be a JSON object")
    raw = payload.get("goals", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("goal completion gate artifact 'goals' must be an object")
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
        if "analysis_inconclusive" in normalized and not isinstance(
            normalized["analysis_inconclusive"], bool
        ):
            raise ValueError(
                f"goal completion gate record {normalized_goal_id!r} field "
                "'analysis_inconclusive' must be a boolean"
            )
        records[normalized_goal_id] = normalized
    return records


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
    objective_path = (args.objective_path or default_objective_path(repo_root)).resolve()
    todo_path = (args.todo_path or default_todo_path(repo_root)).resolve()
    state_root = default_state_root(repo_root)
    discovery_dir = (args.discovery_dir or state_root / "discovery").resolve()
    bundle_dir = (args.bundle_dir or state_root / "objective_bundles").resolve()
    dataset_dir = (args.dataset_dir or state_root / "objective_datasets").resolve()
    graph_path = (getattr(args, "graph_path", None) or state_root / "objective_graph.json").resolve()

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
    completion_gate_path = getattr(args, "objective_goal_completion_gate_path", None)
    if completion_gate_path is not None and not completion_gate_path.is_absolute():
        completion_gate_path = (repo_root / completion_gate_path).resolve()
    completion_gate_records = load_goal_completion_gate_records(completion_gate_path)
    goal_completion_todo_boards = parse_goal_completion_todo_boards(
        getattr(args, "objective_goal_completion_todo_board", []) or [],
        repo_root=repo_root,
        default_task_prefix=args.task_prefix,
    )
    if not getattr(args, "no_reconcile_goal_completion", False) and objective_path.exists():
        completion = reconcile_objective_goal_completion(
            repo_root=repo_root,
            objective_path=objective_path,
            todo_path=todo_path,
            task_header_prefix=args.task_prefix,
            todo_boards=goal_completion_todo_boards,
            completion_gate_records=completion_gate_records,
        )
        completed_goal_ids = completion.completed_goal_ids
        objective_completed_goal_count = completion.completed_goal_count
        objective_completion_validation_results = completion.validation_results
        objective_completion_decisions = completion.decisions

    refined_goal_ids: list[str] = []
    if getattr(args, "refine_objective_heap", False) and objective_path.exists():
        refinement_findings = scan_objective_gaps(
            repo_root,
            objective_path=objective_path,
            max_findings=args.max_findings,
            seen_fingerprints=seen_fingerprints,
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
    )
    plan_evaluation_path = (
        getattr(args, "plan_evaluation_path", None) or state_root / "plan_evaluations.json"
    ).resolve()
    router_config = None
    if bool(getattr(args, "generate_plan_branches", False)):
        from .task_proposal_router import StructuredPlanRouterConfig

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
        from .analyzer_health import AnalysisEscalationPolicy
        from .task_proposal_router import StructuredPlanRouterConfig

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
    if not bool(getattr(args, "no_generate_bounded_work", False)):
        from .objective_graph import ObjectiveGenerationLimits, parse_goal_heap
        from .plan_evaluator import ObjectiveWorkEvaluationPolicy

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
            completion_gate_records=completion_gate_records,
            completion_decisions=objective_completion_decisions,
            analysis_escalation=analysis_escalation_payload,
            default_validation=("git diff --check",),
            estimated_router_tokens=reserved_router_tokens,
            router_retry_count=router_retry_count,
            objective_terms=generation_terms,
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
        if configured_open_work < 0:
            active_goal_count = 0
            if objective_path.exists():
                active_goal_count = sum(
                    1
                    for goal in parse_goal_heap(
                        objective_path.read_text(encoding="utf-8", errors="replace")
                    )
                    if goal.is_schedulable
                )
            try:
                persisted_generated_count = len(
                    load_objective_generation_work(objective_generation_path)
                )
            except (OSError, TypeError, ValueError):
                # The materialization call below reports the corrupt ledger
                # and admits no work; this count must not mask that failure.
                persisted_generated_count = 0
            configured_open_work = (
                active_goal_count + len(records) + persisted_generated_count
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
            )
        except (OSError, TypeError, ValueError) as exc:
            # A corrupt identity ledger or malformed proposal must fail closed
            # for generated work without suppressing the daemon's ordinary
            # deterministic backlog scan.
            objective_generation_error = f"{type(exc).__name__}: {exc}"
            logger.error("Bounded objective generation failed closed: %s", exc)
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
        "objective_generation_cycle_accepted_count": len(
            ((objective_generation_payload or {}).get("last_cycle") or {}).get("accepted", ())
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
        "objective_completion_gate_receipts": completion_gate_receipts_from_decisions(
            objective_completion_decisions
        ),
        "objective_goal_completion_gate_path": (
            repo_relative_path(repo_root, completion_gate_path) if completion_gate_path else ""
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
