"""SAWM-037 required dispatch receipts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


REQUIRED_DISPATCH_RECEIPTS: tuple[str, ...] = (
    "pre_root",
    "current_state",
    "goal_task_revision",
    "context",
    "reuse",
    "prediction_consideration",
)


@dataclass(frozen=True, slots=True)
class RequiredTaskWorldBinding:
    task_id: str
    receipts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RequiredModeReceiptSet:
    present: tuple[str, ...]
    missing: tuple[str, ...]


class ProgramWorldRequiredDispatchGate:
    def admit_required_program_world_dispatch(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        present = tuple(request.get("receipts") or ())
        missing = tuple(
            name for name in REQUIRED_DISPATCH_RECEIPTS if name not in present
        )
        return {
            "admitted": not missing,
            "missing": missing,
            "binding": RequiredTaskWorldBinding(
                task_id=str(request.get("task_id") or ""),
                receipts=present,
            ),
            "completion_authority": False,
        }


def admit_required_program_world_dispatch(request: Mapping[str, Any]) -> dict[str, Any]:
    return ProgramWorldRequiredDispatchGate().admit_required_program_world_dispatch(request)
