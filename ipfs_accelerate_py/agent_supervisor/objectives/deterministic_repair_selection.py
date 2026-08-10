"""DCR-081: selection and refill consume typed repair dispositions only.

Interfaces
----------
* ``DeterministicRepairSelection@1`` — choose the next repair task from
  typed dispositions without provider or prose ranking.

Predicted symbols: :func:`select_deterministic_repair_task`,
:func:`project_repair_disposition`, :func:`refill_from_dispositions`.

Normative rules (fail-closed)
-----------------------------
* Selection keys only closed disposition enums and content-addressed evidence.
* ``proved_valid`` / ``completed`` never re-enter the ready queue.
* ``abstain_review`` / ``defer_capability`` / ``refuted_repairable`` project to
  typed residual or deferred queues — never free LLM retry.
* Runtime model calls remain 0.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    ContractValidationError,
    content_identity,
)


DETERMINISTIC_REPAIR_SELECTION_INTERFACE: Final[str] = (
    "DeterministicRepairSelection@1"
)
DCR_SELECTION_EVIDENCE: Final[str] = "dcr/selection-refill@1"
DCR_SELECTION_VERSION: Final[int] = 1
SELECTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-selection-receipt@1"
)
SELECTION_CATALOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-selection-refill-catalog@1"
)
DEFAULT_SELECTION_REFILL_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/selection-refill.json"
)

# Public DCR disposition vocabulary (aligns with DeterministicRepairDisposition).
_CLOSED_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "proved_valid",
        "refuted_repairable",
        "repaired_pending_validation",
        "abstain_review",
        "defer_capability",
        "rejected",
        "completed",
        "ready",
        "blocked",
        "todo",
    }
)

# Dispositions that may enter the selectable ready set.
_SELECTABLE: Final[frozenset[str]] = frozenset({"ready", "todo", "refuted_repairable"})

# Dispositions that must never re-enter selection.
_TERMINAL: Final[frozenset[str]] = frozenset(
    {"proved_valid", "completed", "rejected"}
)

# Residual / deferred queues (no free provider retry).
_RESIDUAL: Final[frozenset[str]] = frozenset({"abstain_review"})
_DEFERRED: Final[frozenset[str]] = frozenset({"defer_capability"})


class SelectionDisposition(str, Enum):  # noqa: UP042
    SELECTED = "selected"
    EMPTY = "empty"
    ABSTAIN = "abstain"
    DEFERRED = "deferred"


class DeterministicRepairSelectionError(ContractValidationError):
    """Malformed selection input or closed-boundary violation."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeterministicRepairSelectionError(f"{name} must be a non-empty string")
    return value.strip()


def _disposition(value: Any) -> str:
    text = _text(value, "disposition").lower()
    if text not in _CLOSED_DISPOSITIONS:
        raise DeterministicRepairSelectionError(
            f"disposition {text!r} is not in the closed DCR vocabulary"
        )
    return text


def project_repair_disposition(
    task: Mapping[str, Any] | str,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> str:
    """Project a task row / status into one closed repair disposition.

    Prefer explicit ``disposition`` / ``repair_disposition`` fields, then board
    ``status``, then residual evidence.  Never invent provider retries.
    """

    if isinstance(task, str):
        status = task.strip().lower()
        if status in _CLOSED_DISPOSITIONS:
            return status
        if status in {"ready", "todo", "open"}:
            return "ready" if status != "todo" else "todo"
        raise DeterministicRepairSelectionError(f"unknown status {task!r}")

    if not isinstance(task, Mapping):
        raise DeterministicRepairSelectionError("task must be a mapping or string")

    for key in ("repair_disposition", "disposition", "public_disposition"):
        if key in task and task[key] not in (None, ""):
            return _disposition(task[key])

    evidence = evidence or {}
    # Typed residual evidence outranks a bare board status of todo/ready.
    if evidence.get("proved_valid") is True:
        return "proved_valid"
    if evidence.get("defer_capability") is True or evidence.get("capability_missing"):
        return "defer_capability"
    if evidence.get("abstain") is True or evidence.get("residual"):
        return "abstain_review"
    if evidence.get("refuted") is True or evidence.get("repairable") is True:
        return "refuted_repairable"

    status = str(task.get("status") or task.get("task_status") or "").strip().lower()
    if status in _CLOSED_DISPOSITIONS:
        return status
    if status in {"ready", "open"}:
        return "ready"
    if status in {"todo", "pending"}:
        return "todo"
    if status in {"blocked", "waiting"}:
        return "blocked"
    if status in {"completed", "done"}:
        return "completed"

    raise DeterministicRepairSelectionError(
        "cannot project a closed disposition without status or evidence"
    )


@dataclass(frozen=True)
class SelectionCandidate:
    task_id: str
    disposition: str
    priority: int = 0
    evidence_cid: str = ""
    shard_index: int | None = None
    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/dcr-selection-candidate@1"
    )
    INTERFACE: ClassVar[str] = DETERMINISTIC_REPAIR_SELECTION_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "disposition", _disposition(self.disposition))
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "evidence_cid", str(self.evidence_cid or ""))
        if self.shard_index is not None:
            object.__setattr__(self, "shard_index", int(self.shard_index))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "task_id": self.task_id,
            "disposition": self.disposition,
            "priority": self.priority,
            "evidence_cid": self.evidence_cid,
            "shard_index": self.shard_index,
        }


@dataclass(frozen=True)
class SelectionReceipt:
    disposition: SelectionDisposition
    selected_task_id: str = ""
    selectable: tuple[str, ...] = ()
    residual: tuple[str, ...] = ()
    deferred: tuple[str, ...] = ()
    terminal: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    runtime_model_calls: int = 0
    grants_provider_dispatch: bool = False
    SCHEMA: ClassVar[str] = SELECTION_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = DETERMINISTIC_REPAIR_SELECTION_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, SelectionDisposition):
            raise DeterministicRepairSelectionError("invalid selection disposition")
        object.__setattr__(self, "selected_task_id", str(self.selected_task_id or ""))
        object.__setattr__(
            self, "selectable", tuple(str(item) for item in self.selectable)
        )
        object.__setattr__(self, "residual", tuple(str(item) for item in self.residual))
        object.__setattr__(self, "deferred", tuple(str(item) for item in self.deferred))
        object.__setattr__(self, "terminal", tuple(str(item) for item in self.terminal))
        object.__setattr__(
            self, "reason_codes", tuple(str(item) for item in self.reason_codes)
        )
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "grants_provider_dispatch", False)

    @property
    def ok(self) -> bool:
        return self.disposition is SelectionDisposition.SELECTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "disposition": self.disposition.value,
            "selected_task_id": self.selected_task_id,
            "selectable": list(self.selectable),
            "residual": list(self.residual),
            "deferred": list(self.deferred),
            "terminal": list(self.terminal),
            "reason_codes": list(self.reason_codes),
            "runtime_model_calls": 0,
            "grants_provider_dispatch": False,
            "content_id": content_identity(
                {
                    "disposition": self.disposition.value,
                    "selected_task_id": self.selected_task_id,
                    "selectable": list(self.selectable),
                    "residual": list(self.residual),
                    "deferred": list(self.deferred),
                    "terminal": list(self.terminal),
                    "reason_codes": list(self.reason_codes),
                }
            ),
        }


def _as_candidates(
    tasks: Sequence[Mapping[str, Any] | SelectionCandidate | str],
    *,
    evidence_by_task: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[SelectionCandidate]:
    evidence_by_task = evidence_by_task or {}
    out: list[SelectionCandidate] = []
    for item in tasks:
        if isinstance(item, SelectionCandidate):
            out.append(item)
            continue
        if isinstance(item, str):
            out.append(
                SelectionCandidate(
                    task_id=item,
                    disposition=project_repair_disposition(
                        item, evidence=evidence_by_task.get(item)
                    ),
                )
            )
            continue
        if not isinstance(item, Mapping):
            raise DeterministicRepairSelectionError("invalid candidate")
        task_id = _text(item.get("task_id") or item.get("id"), "task_id")
        disp = project_repair_disposition(
            item, evidence=evidence_by_task.get(task_id) or item.get("evidence")
        )
        out.append(
            SelectionCandidate(
                task_id=task_id,
                disposition=disp,
                priority=int(item.get("priority") or item.get("rank") or 0),
                evidence_cid=str(item.get("evidence_cid") or ""),
                shard_index=(
                    None
                    if item.get("shard_index") is None
                    else int(item.get("shard_index"))
                ),
            )
        )
    return out


def select_deterministic_repair_task(
    tasks: Sequence[Mapping[str, Any] | SelectionCandidate | str],
    *,
    evidence_by_task: Mapping[str, Mapping[str, Any]] | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> SelectionReceipt:
    """Select one ready repair task from typed dispositions only.

    Ordering is deterministic: lower priority number first, then task_id.
    Shard filters use explicit shard_index when provided.
    """

    candidates = _as_candidates(tasks, evidence_by_task=evidence_by_task)
    selectable: list[SelectionCandidate] = []
    residual: list[str] = []
    deferred: list[str] = []
    terminal: list[str] = []

    for candidate in candidates:
        disp = candidate.disposition
        if disp in _TERMINAL:
            terminal.append(candidate.task_id)
            continue
        if disp in _RESIDUAL:
            residual.append(candidate.task_id)
            continue
        if disp in _DEFERRED:
            deferred.append(candidate.task_id)
            continue
        if disp not in _SELECTABLE:
            # blocked / repaired_pending_validation stay non-selectable
            continue
        if shard_index is not None and shard_count is not None and shard_count > 0:
            assigned = (
                candidate.shard_index
                if candidate.shard_index is not None
                else (
                    int(content_identity({"task_id": candidate.task_id})[-8:], 16)
                    % int(shard_count)
                )
            )
            if int(assigned) != int(shard_index):
                continue
        selectable.append(candidate)

    selectable_sorted = sorted(
        selectable, key=lambda item: (item.priority, item.task_id)
    )
    selectable_ids = tuple(item.task_id for item in selectable_sorted)

    if not selectable_sorted:
        if residual and not deferred:
            return SelectionReceipt(
                disposition=SelectionDisposition.ABSTAIN,
                selectable=(),
                residual=tuple(sorted(set(residual))),
                deferred=tuple(sorted(set(deferred))),
                terminal=tuple(sorted(set(terminal))),
                reason_codes=("no_selectable", "residual_only", "no_provider_retry"),
            )
        if deferred:
            return SelectionReceipt(
                disposition=SelectionDisposition.DEFERRED,
                selectable=(),
                residual=tuple(sorted(set(residual))),
                deferred=tuple(sorted(set(deferred))),
                terminal=tuple(sorted(set(terminal))),
                reason_codes=("no_selectable", "capability_deferred", "no_provider_retry"),
            )
        return SelectionReceipt(
            disposition=SelectionDisposition.EMPTY,
            selectable=(),
            residual=tuple(sorted(set(residual))),
            deferred=tuple(sorted(set(deferred))),
            terminal=tuple(sorted(set(terminal))),
            reason_codes=("no_selectable", "empty_queue"),
        )

    chosen = selectable_sorted[0]
    return SelectionReceipt(
        disposition=SelectionDisposition.SELECTED,
        selected_task_id=chosen.task_id,
        selectable=selectable_ids,
        residual=tuple(sorted(set(residual))),
        deferred=tuple(sorted(set(deferred))),
        terminal=tuple(sorted(set(terminal))),
        reason_codes=("typed_disposition_order", "deterministic_priority", "no_provider"),
    )


def refill_from_dispositions(
    tasks: Sequence[Mapping[str, Any] | SelectionCandidate | str],
    *,
    evidence_by_task: Mapping[str, Mapping[str, Any]] | None = None,
    max_refill: int = 8,
) -> dict[str, Any]:
    """Project residual/deferred/selectable queues for backlog refill.

    Does not mint provider work items.  Residual tasks stay review-bound;
    deferred tasks stay capability-bound.
    """

    if max_refill < 0:
        raise DeterministicRepairSelectionError("max_refill must be non-negative")
    candidates = _as_candidates(tasks, evidence_by_task=evidence_by_task)
    ready: list[str] = []
    residual: list[str] = []
    deferred: list[str] = []
    closed: list[str] = []
    for candidate in sorted(candidates, key=lambda item: (item.priority, item.task_id)):
        if candidate.disposition in _TERMINAL:
            closed.append(candidate.task_id)
        elif candidate.disposition in _RESIDUAL:
            residual.append(candidate.task_id)
        elif candidate.disposition in _DEFERRED:
            deferred.append(candidate.task_id)
        elif candidate.disposition in _SELECTABLE:
            if len(ready) < max_refill:
                ready.append(candidate.task_id)
    return {
        "schema": SELECTION_CATALOG_SCHEMA,
        "interface": DETERMINISTIC_REPAIR_SELECTION_INTERFACE,
        "evidence_id": DCR_SELECTION_EVIDENCE,
        "version": DCR_SELECTION_VERSION,
        "ready_task_ids": ready,
        "residual_task_ids": residual,
        "deferred_task_ids": deferred,
        "terminal_task_ids": closed,
        "runtime_model_calls": 0,
        "grants_provider_dispatch": False,
        "reason_codes": [
            "typed_disposition_refill",
            "no_provider_retry",
            "bounded_ready",
        ],
    }


def materialize_selection_refill(
    *,
    tasks: Sequence[Mapping[str, Any] | SelectionCandidate | str] | None = None,
    destination: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize selection-refill.json evidence for DCR-081."""

    if tasks is None:
        tasks = (
            {"task_id": "DCR-081", "status": "ready", "priority": 0},
            {"task_id": "DCR-083", "status": "ready", "priority": 1},
            {
                "task_id": "DCR-X-residual",
                "disposition": "abstain_review",
                "priority": 0,
            },
            {
                "task_id": "DCR-X-defer",
                "disposition": "defer_capability",
                "priority": 0,
            },
            {"task_id": "DCR-X-done", "status": "completed", "priority": 0},
        )
    selection = select_deterministic_repair_task(tasks)
    refill = refill_from_dispositions(tasks)
    payload = {
        "schema": SELECTION_CATALOG_SCHEMA,
        "interface": DETERMINISTIC_REPAIR_SELECTION_INTERFACE,
        "evidence_id": DCR_SELECTION_EVIDENCE,
        "version": DCR_SELECTION_VERSION,
        "selection": selection.to_dict(),
        "refill": refill,
        "runtime_model_calls": 0,
        "grants_provider_dispatch": False,
    }
    base = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    path = (
        Path(destination)
        if destination is not None
        else base.joinpath(*PurePosixPath(DEFAULT_SELECTION_REFILL_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DCR_SELECTION_EVIDENCE",
    "DCR_SELECTION_VERSION",
    "DEFAULT_SELECTION_REFILL_PATH",
    "DETERMINISTIC_REPAIR_SELECTION_INTERFACE",
    "DeterministicRepairSelectionError",
    "SelectionCandidate",
    "SelectionDisposition",
    "SelectionReceipt",
    "materialize_selection_refill",
    "project_repair_disposition",
    "refill_from_dispositions",
    "select_deterministic_repair_task",
]
