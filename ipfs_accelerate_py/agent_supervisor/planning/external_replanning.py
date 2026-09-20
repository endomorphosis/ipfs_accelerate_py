"""Typed bounded replanning triggers (EAAEF-102)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final


REPLAN_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/external-replanning@1"
TRIGGERS: Final[frozenset[str]] = frozenset(
    {
        "changed_assumption",
        "invalidation",
        "failed_tests",
        "failed_proofs",
        "counterexample",
        "stale_history",
        "conflict",
        "outage",
        "resource_exhaustion",
        "task_resize",
        "evidence_gap",
        "no_progress",
    }
)


class ReplanError(ValueError):
    """Unknown or untyped replanning trigger."""


@dataclass(frozen=True)
class ReplanTrigger:
    kind: str
    plan_id: str
    evidence_id: str

    def __post_init__(self) -> None:
        if str(self.kind) not in TRIGGERS:
            raise ReplanError(f"unknown replan trigger: {self.kind}")
        if not str(self.plan_id).strip() or not str(self.evidence_id).strip():
            raise ReplanError("plan_id and evidence_id are required")

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": REPLAN_SCHEMA,
                "kind": self.kind,
                "plan_id": self.plan_id,
                "evidence_id": self.evidence_id,
            }
        )


def compile_trigger(kind: str, *, plan_id: str, evidence_id: str) -> ReplanTrigger:
    trigger = ReplanTrigger(kind=kind, plan_id=plan_id, evidence_id=evidence_id)
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        mirror_work_record(
            catalog_kind="metadata",
            record_kind="replan_trigger",
            record_ref=str(plan_id or evidence_id or "replan-trigger"),
            subject_kind="record_cid",
            subject_ref=str(evidence_id or plan_id or "replan-trigger"),
        )
    except Exception:
        pass
    return trigger
