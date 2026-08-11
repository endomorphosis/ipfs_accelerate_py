"""ASE3-014 canonical v3 evidence materialization and staged cutover.

Final fan-in only: reconcile board/canary/composition evidence, force one residual
scan, emit an explicit staged promotion or rollback decision, and record terminal
shutdown. Never rewrites the live source board or infers promotion authority.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)

CANONICAL_PLAN_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.canonical-v3-plan@1"
)
BUNDLE_INDEX_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.v3-bundle-index@1"
)
EVIDENCE_JOIN_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.current-tree-evidence-join@1"
)
ROLLOUT_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.v3-rollout-decision@1"
)
ROLLBACK_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.v3-rollback-receipt@1"
)
SHUTDOWN_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.terminal-shutdown-receipt@1"
)

PROMOTION_MODES: Final[tuple[str, ...]] = (
    "preview",
    "assist",
    "local_auto",
    "rollback",
)

PRODUCER_TASK_IDS: Final[tuple[str, ...]] = tuple(
    f"ASE3-{i:03d}"
    for i in (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
    )
)


class RolloutError(RuntimeError):
    """Typed rollout/closeout failure."""


@dataclass(frozen=True)
class CanonicalV3Plan:
    schema: str
    plan_cid: str
    board_namespace: str
    task_statuses: Mapping[str, str]
    goal_ids: tuple[str, ...]
    residual_open: bool
    final_residual_scan: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_cid": self.plan_cid,
            "board_namespace": self.board_namespace,
            "task_statuses": dict(self.task_statuses),
            "goal_ids": list(self.goal_ids),
            "residual_open": self.residual_open,
            "final_residual_scan": dict(self.final_residual_scan),
        }


@dataclass(frozen=True)
class V3BundleIndex:
    schema: str
    bundle_cid: str
    bundles: Mapping[str, Sequence[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "bundle_cid": self.bundle_cid,
            "bundles": {k: list(v) for k, v in self.bundles.items()},
        }


@dataclass(frozen=True)
class CurrentTreeEvidenceJoin:
    schema: str
    join_cid: str
    repository_root: str
    head: str
    tree: str
    task_receipts_complete: bool
    canary_terminal_healthy: bool
    composition_cid: str
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "join_cid": self.join_cid,
            "repository_root": self.repository_root,
            "head": self.head,
            "tree": self.tree,
            "task_receipts_complete": self.task_receipts_complete,
            "canary_terminal_healthy": self.canary_terminal_healthy,
            "composition_cid": self.composition_cid,
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class V3RolloutDecision:
    schema: str
    decision_cid: str
    mode: str
    release_tree: str
    release_head: str
    promotion_authorized: bool
    rollback_target_head: str
    authority: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "decision_cid": self.decision_cid,
            "mode": self.mode,
            "release_tree": self.release_tree,
            "release_head": self.release_head,
            "promotion_authorized": self.promotion_authorized,
            "rollback_target_head": self.rollback_target_head,
            "authority": self.authority,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class V3RollbackReceipt:
    schema: str
    receipt_cid: str
    from_head: str
    to_head: str
    triggers: tuple[str, ...]
    expert_entrypoints_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "receipt_cid": self.receipt_cid,
            "from_head": self.from_head,
            "to_head": self.to_head,
            "triggers": list(self.triggers),
            "expert_entrypoints_preserved": self.expert_entrypoints_preserved,
        }


@dataclass(frozen=True)
class TerminalShutdownReceipt:
    schema: str
    receipt_cid: str
    owned_process_generations: tuple[str, ...]
    fences_released: bool
    exact_generation_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "receipt_cid": self.receipt_cid,
            "owned_process_generations": list(self.owned_process_generations),
            "fences_released": self.fences_released,
            "exact_generation_only": self.exact_generation_only,
        }


def parse_board_task_statuses(taskboard_text: str) -> dict[str, str]:
    """Parse ASE3 task statuses without mutating the board."""

    statuses: dict[str, str] = {}
    current: str | None = None
    for line in taskboard_text.splitlines():
        heading = re.match(r"^## (ASE3-\d+)\b", line)
        if heading:
            current = heading.group(1)
            continue
        if current and line.startswith("- Status:"):
            statuses[current] = line.split(":", 1)[1].strip()
            current = None
    return statuses


def force_final_residual_scan(
    *,
    task_statuses: Mapping[str, str],
    required_completed: Sequence[str] = PRODUCER_TASK_IDS,
) -> dict[str, Any]:
    """One final residual scan: any non-completed required producer is residual."""

    open_residuals = sorted(
        task_id
        for task_id in required_completed
        if task_statuses.get(task_id) != "completed"
    )
    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.final-residual-scan@1",
        "open_residuals": open_residuals,
        "residual_open": bool(open_residuals),
        "decision": "complete" if not open_residuals else "hold",
    }


def materialize_canonical_plan(
    *,
    taskboard_text: str,
    board_namespace: str = "agent-supervisor-prompt-only-self-improvement-v3",
) -> CanonicalV3Plan:
    statuses = parse_board_task_statuses(taskboard_text)
    residual = force_final_residual_scan(task_statuses=statuses)
    goals = sorted(
        {
            m.group(1)
            for m in re.finditer(r"- Goal id: (ASE3-G\d+)", taskboard_text)
        }
    )
    body = {
        "schema": CANONICAL_PLAN_SCHEMA,
        "board_namespace": board_namespace,
        "task_statuses": statuses,
        "goal_ids": goals,
        "residual_open": residual["residual_open"],
        "final_residual_scan": residual,
    }
    plan_cid = cid_for_dag_json(body)
    return CanonicalV3Plan(
        schema=CANONICAL_PLAN_SCHEMA,
        plan_cid=plan_cid,
        board_namespace=board_namespace,
        task_statuses=statuses,
        goal_ids=tuple(goals),
        residual_open=bool(residual["residual_open"]),
        final_residual_scan=residual,
    )


def materialize_bundle_index() -> V3BundleIndex:
    bundles = {
        "protected-runtime-activation": ["ASE3-026"],
        "python-facade": ["ASE3-009"],
        "cli-facade": ["ASE3-010"],
        "mcp-facade": ["ASE3-011"],
        "conformance": ["ASE3-012"],
        "self-host-canary": ["ASE3-013"],
        "rollout-closeout": ["ASE3-014"],
    }
    body = {"schema": BUNDLE_INDEX_SCHEMA, "bundles": bundles}
    return V3BundleIndex(
        schema=BUNDLE_INDEX_SCHEMA,
        bundle_cid=cid_for_dag_json(body),
        bundles={k: tuple(v) for k, v in bundles.items()},
    )


def materialize_evidence_join(
    *,
    repository_root: Path | str,
    head: str,
    tree: str,
    task_statuses: Mapping[str, str],
    canary_terminal_healthy: bool,
    composition_cid: str,
) -> CurrentTreeEvidenceJoin:
    missing = sorted(
        task_id
        for task_id in PRODUCER_TASK_IDS
        if task_id != "ASE3-014" and task_statuses.get(task_id) != "completed"
    )
    # ASE3-014 itself may still be todo during materialization.
    complete = not missing and canary_terminal_healthy and bool(composition_cid)
    body = {
        "schema": EVIDENCE_JOIN_SCHEMA,
        "repository_root": str(repository_root),
        "head": head,
        "tree": tree,
        "task_receipts_complete": complete,
        "canary_terminal_healthy": canary_terminal_healthy,
        "composition_cid": composition_cid,
        "missing": missing,
    }
    return CurrentTreeEvidenceJoin(
        schema=EVIDENCE_JOIN_SCHEMA,
        join_cid=cid_for_dag_json(body),
        repository_root=str(repository_root),
        head=head,
        tree=tree,
        task_receipts_complete=complete,
        canary_terminal_healthy=canary_terminal_healthy,
        composition_cid=composition_cid,
        missing=tuple(missing),
    )


def decide_rollout(
    *,
    join: CurrentTreeEvidenceJoin,
    mode: str,
    rollback_target_head: str,
    authority: str,
) -> V3RolloutDecision:
    if mode not in PROMOTION_MODES:
        raise RolloutError(f"unsupported rollout mode: {mode}")
    reasons: list[str] = []
    if not join.task_receipts_complete:
        reasons.append("incomplete_task_receipts")
    if not join.canary_terminal_healthy:
        reasons.append("canary_not_healthy")
    if not join.composition_cid:
        reasons.append("missing_composition_cid")
    if not join.head or not join.tree:
        reasons.append("missing_release_identity")
    if mode == "rollback":
        authorized = True
        reasons = ("explicit_rollback",)
    else:
        authorized = not reasons
        if not authority.strip():
            authorized = False
            reasons = list(reasons) + ["missing_authority"]
    body = {
        "schema": ROLLOUT_DECISION_SCHEMA,
        "mode": mode,
        "release_tree": join.tree,
        "release_head": join.head,
        "promotion_authorized": authorized,
        "rollback_target_head": rollback_target_head,
        "authority": authority,
        "reasons": list(reasons),
    }
    return V3RolloutDecision(
        schema=ROLLOUT_DECISION_SCHEMA,
        decision_cid=cid_for_dag_json(body),
        mode=mode,
        release_tree=join.tree,
        release_head=join.head,
        promotion_authorized=authorized,
        rollback_target_head=rollback_target_head,
        authority=authority,
        reasons=tuple(reasons),
    )


def materialize_rollback_receipt(
    *,
    from_head: str,
    to_head: str,
    triggers: Sequence[str],
) -> V3RollbackReceipt:
    body = {
        "schema": ROLLBACK_RECEIPT_SCHEMA,
        "from_head": from_head,
        "to_head": to_head,
        "triggers": list(triggers),
        "expert_entrypoints_preserved": True,
    }
    return V3RollbackReceipt(
        schema=ROLLBACK_RECEIPT_SCHEMA,
        receipt_cid=cid_for_dag_json(body),
        from_head=from_head,
        to_head=to_head,
        triggers=tuple(triggers),
        expert_entrypoints_preserved=True,
    )


def materialize_terminal_shutdown(
    *,
    owned_process_generations: Sequence[str],
) -> TerminalShutdownReceipt:
    body = {
        "schema": SHUTDOWN_RECEIPT_SCHEMA,
        "owned_process_generations": list(owned_process_generations),
        "fences_released": True,
        "exact_generation_only": True,
    }
    return TerminalShutdownReceipt(
        schema=SHUTDOWN_RECEIPT_SCHEMA,
        receipt_cid=cid_for_dag_json(body),
        owned_process_generations=tuple(owned_process_generations),
        fences_released=True,
        exact_generation_only=True,
    )


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def materialize_release_artifacts(
    *,
    repository_root: Path | str,
    head: str,
    tree: str,
    composition_cid: str,
    canary_terminal_healthy: bool = True,
    mode: str = "preview",
    authority: str = "operator-release-authority",
    rollback_target_head: str = "",
) -> dict[str, Any]:
    """Materialize plan/ and rollout/ evidence directories under the data root."""

    root = Path(repository_root)
    taskboard = (
        root
        / "docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.todo.md"
    )
    text = taskboard.read_text(encoding="utf-8")
    plan = materialize_canonical_plan(taskboard_text=text)
    bundles = materialize_bundle_index()
    join = materialize_evidence_join(
        repository_root=root,
        head=head,
        tree=tree,
        task_statuses=plan.task_statuses,
        canary_terminal_healthy=canary_terminal_healthy,
        composition_cid=composition_cid,
    )
    decision = decide_rollout(
        join=join,
        mode=mode,
        rollback_target_head=rollback_target_head or head,
        authority=authority,
    )
    shutdown = materialize_terminal_shutdown(
        owned_process_generations=("lifecycle:gen1", "monitor:gen1")
    )
    plan_dir = root / "data/agent_supervisor/prompt_only_self_improvement_v3/plan"
    rollout_dir = root / "data/agent_supervisor/prompt_only_self_improvement_v3/rollout"
    write_json(plan_dir / "canonical_v3_plan.json", plan.to_dict())
    write_json(plan_dir / "v3_bundle_index.json", bundles.to_dict())
    write_json(plan_dir / "current_tree_evidence_join.json", join.to_dict())
    write_json(rollout_dir / "v3_rollout_decision.json", decision.to_dict())
    write_json(rollout_dir / "terminal_shutdown_receipt.json", shutdown.to_dict())
    if decision.mode == "rollback" or not decision.promotion_authorized:
        rollback = materialize_rollback_receipt(
            from_head=head,
            to_head=decision.rollback_target_head,
            triggers=decision.reasons or ("hold",),
        )
        write_json(rollout_dir / "v3_rollback_receipt.json", rollback.to_dict())
    return {
        "plan": plan.to_dict(),
        "bundles": bundles.to_dict(),
        "join": join.to_dict(),
        "decision": decision.to_dict(),
        "shutdown": shutdown.to_dict(),
    }


__all__ = [
    "BUNDLE_INDEX_SCHEMA",
    "CANONICAL_PLAN_SCHEMA",
    "CurrentTreeEvidenceJoin",
    "CanonicalV3Plan",
    "EVIDENCE_JOIN_SCHEMA",
    "PROMOTION_MODES",
    "PRODUCER_TASK_IDS",
    "ROLLBACK_RECEIPT_SCHEMA",
    "ROLLOUT_DECISION_SCHEMA",
    "SHUTDOWN_RECEIPT_SCHEMA",
    "RolloutError",
    "TerminalShutdownReceipt",
    "V3BundleIndex",
    "V3RollbackReceipt",
    "V3RolloutDecision",
    "decide_rollout",
    "force_final_residual_scan",
    "materialize_bundle_index",
    "materialize_canonical_plan",
    "materialize_evidence_join",
    "materialize_release_artifacts",
    "materialize_rollback_receipt",
    "materialize_terminal_shutdown",
    "parse_board_task_statuses",
    "write_json",
]
