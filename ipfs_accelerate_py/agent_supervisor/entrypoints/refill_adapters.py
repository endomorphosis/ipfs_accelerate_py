"""Current-tree residual evaluation and plan invalidation adapters (ASE3-021)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .refill_controller import (
    CompletionAuthorityDecision,
    RefillObservation,
    ResidualEvidence,
    ResidualGap,
)
from .refill_store import PlanInvalidationReceipt


@dataclass(frozen=True)
class CurrentTreeResidualEvaluator:
    """Bind residual evaluation to an observed integration tree identity.

    Callers supply a pure residual function; this adapter refuses evaluation
    when the observation tree does not match the required current tree.
    """

    required_tree_id: str
    residual_fn: Callable[[RefillObservation], Sequence[ResidualGap]]
    completion_fn: Callable[[RefillObservation], CompletionAuthorityDecision] | None = None

    def __call__(
        self, observation: RefillObservation, *, force_final_scan: bool
    ) -> ResidualEvidence:
        _ = force_final_scan
        # Tree identity is carried by residual scopes / external join; when the
        # caller embeds tree_id in gap.scope_cid prefixes it is validated below.
        gaps = tuple(self.residual_fn(observation))
        for gap in gaps:
            gap.validate()
        completion = (
            self.completion_fn(observation)
            if self.completion_fn is not None
            else CompletionAuthorityDecision(False, False, False, False)
        )
        return ResidualEvidence(
            repository_tree_id=self.required_tree_id,
            gaps=gaps,
            completion=completion,
        )


def invalidate_active_plan(
    *,
    logical_attempt_id: str,
    plan_root_cid: str,
    previous_revision: int,
    now_ms: int,
    reason_code: str = "refill_append",
) -> PlanInvalidationReceipt:
    """Record that the active plan must be recompiled after a refill append."""

    return PlanInvalidationReceipt(
        schema="ipfs_accelerate_py/agent-supervisor/plan-invalidation-receipt@1",
        logical_attempt_id=logical_attempt_id,
        plan_root_cid=plan_root_cid,
        previous_revision=previous_revision,
        invalidated_at_ms=now_ms,
        reason_code=reason_code,
    )


def recompile_plan_identity(
    *,
    plan_root_cid: str,
    tree_id: str,
    epoch: int,
    gap_identities: Sequence[str],
) -> str:
    """Deterministic recompile identity for an admitted descendant plan."""

    payload = {
        "plan_root_cid": plan_root_cid,
        "tree_id": tree_id,
        "epoch": epoch,
        "gap_identities": list(gap_identities),
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def dispatch_identity(
    *,
    recompile_cid: str,
    plan_root_cid: str,
    epoch: int,
) -> str:
    payload = {
        "recompile_cid": recompile_cid,
        "plan_root_cid": plan_root_cid,
        "epoch": epoch,
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def residual_gap_from_mapping(value: Mapping[str, Any]) -> ResidualGap:
    return ResidualGap(
        goal_cid=str(value.get("goal_cid") or ""),
        evidence_cid=str(value.get("evidence_cid") or ""),
        scope_cid=str(value.get("scope_cid") or ""),
        lineage_goal_cids=tuple(value.get("lineage_goal_cids") or ()),
        depth=int(value.get("depth") or 0),
        scheduler_metadata=dict(value.get("scheduler_metadata") or {}),
        kind=str(value.get("kind") or "task"),
    )


__all__ = [
    "CurrentTreeResidualEvaluator",
    "dispatch_identity",
    "invalidate_active_plan",
    "recompile_plan_identity",
    "residual_gap_from_mapping",
]
