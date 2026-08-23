"""Immutable plan-revision operations (EAAEF-103).

Add, supersede, split, coalesce, rewire, reprioritize, block, unblock, cancel
future work.  Claimed or accepted history is not edited.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final


REVISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-plan-revision@1"
)
OPS: Final[frozenset[str]] = frozenset(
    {
        "add",
        "supersede",
        "split",
        "coalesce",
        "rewire",
        "reprioritize",
        "block",
        "unblock",
        "cancel_future",
        "add_proof_task",
        "add_repair_task",
        "add_review_task",
    }
)


class PlanRevisionError(ValueError):
    """Plan revision would mutate claimed history."""


@dataclass(frozen=True)
class PlanRevisionOp:
    op: str
    target_id: str
    claimed: bool = False
    accepted: bool = False

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise PlanRevisionError(f"unknown plan op: {self.op}")
        if not str(self.target_id).strip():
            raise PlanRevisionError("target_id is required")
        if self.claimed or self.accepted:
            raise PlanRevisionError("claimed or accepted history cannot be edited")

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": REVISION_SCHEMA,
                "op": self.op,
                "target_id": self.target_id,
                "claimed": False,
                "accepted": False,
            }
        )


def apply_ops(ops: Sequence[Mapping[str, Any] | PlanRevisionOp]) -> tuple[PlanRevisionOp, ...]:
    compiled = []
    for item in ops:
        if isinstance(item, PlanRevisionOp):
            compiled.append(item)
        else:
            compiled.append(
                PlanRevisionOp(
                    op=str(item.get("op") or ""),
                    target_id=str(item.get("target_id") or ""),
                    claimed=bool(item.get("claimed")),
                    accepted=bool(item.get("accepted")),
                )
            )
    return tuple(compiled)
