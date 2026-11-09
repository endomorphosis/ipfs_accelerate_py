"""Spawn a bounded repair subagent when a task board is blocked or stalled.

The subagent may record successor subgoals or insert a missing task
status. It does not mark a task completed, drop a dependency, renew a
clock, add a ready task, restart the supervisor, or grant completion
authority. A blocked frontier is repaired by a proposal. A live
supervisor is left running.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


BOARD_STALL_SCHEMA = "ipfs_accelerate_py/agent-supervisor/board-stall-watch@1"
BOARD_STALL_PACKET_SCHEMA = "ipfs_accelerate_py/agent-supervisor/board-stall-packet@1"
BOARD_STALL_RECEIPT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/board-stall-receipt@1"

FORBIDDEN_EFFECTS: tuple[str, ...] = (
    "mark_parent_completed",
    "grant_completion_authority",
    "drop_dependency",
    "renew_exhausted_clock",
    "restart_supervisor",
    "add_ready_task",
    "invent_owner_evidence",
)

_COMPLETED = frozenset({"completed", "complete", "done"})
_OPEN = frozenset({"ready", "pending", "in_progress", "todo"})
_KNOWN_STATUS = _COMPLETED | _OPEN | frozenset({"blocked"})
_EMPTY_DEPENDENCY = frozenset({"none", "n/a", "na", "-", "—"})
_SPAWN_KINDS = frozenset({
    "successor_subgoals",
    "successor_tasks",
    "repair_board",
    "record_only",
    "attention",
    "none",
})
_SUCCESSOR_CHAIN = ("successor_subgoals", "successor_tasks")
_UNSAFE_COMMAND = set(";|&$`<>\n")
_ATTEMPT_CAP = 3


@dataclass(frozen=True)
class BoardTask:
    """One task heading read from a markdown board."""

    task_id: str
    status: str
    depends_on: tuple[str, ...]
    goal_id: str
    has_status: bool


@dataclass(frozen=True)
class BoardParse:
    """Structural reading of a board. It does not decide completion."""

    tasks: tuple[BoardTask, ...]
    duplicate_ids: tuple[str, ...]
    missing_status_ids: tuple[str, ...]
    unknown_status_ids: tuple[str, ...]
    cycles: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Process facts supplied by the caller. This module does not signal a pid."""

    supervisor_alive: bool
    phase: str = ""
    active_task_id: str = ""
    heartbeat_age_s: float | None = None
    idle_for_s: float = 0.0
    work_complete: bool = False
    completion_authority: bool = False


@dataclass(frozen=True)
class StallVerdict:
    """What the watchdog may do about one observation."""

    stalled: bool
    reason: str
    repair_kind: str
    task_ids: tuple[str, ...] = ()
    work_complete: bool = False
    completion_authority: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BOARD_STALL_SCHEMA,
            "stalled": self.stalled,
            "reason": self.reason,
            "repair_kind": self.repair_kind,
            "task_ids": list(self.task_ids),
            "work_complete": False,
            "completion_authority": False,
        }


def _ids(raw: str) -> tuple[str, ...]:
    """Split a dependency field. ``none`` is an empty frontier, not a task id."""

    return tuple(
        item.strip()
        for item in raw.split(",")
        if item.strip() and item.strip().casefold() not in _EMPTY_DEPENDENCY
    )


def _cycles(tasks: Sequence[BoardTask]) -> tuple[tuple[str, ...], ...]:
    by_id = {task.task_id: task for task in tasks}
    found: list[tuple[str, ...]] = []
    visiting: list[str] = []
    state: dict[str, int] = {}

    def walk(task_id: str) -> None:
        state[task_id] = 1
        visiting.append(task_id)
        for dep in by_id[task_id].depends_on:
            if dep not in by_id:
                continue
            mark = state.get(dep, 0)
            if mark == 1:
                start = visiting.index(dep)
                cycle = tuple(visiting[start:] + [dep])
                if cycle not in found:
                    found.append(cycle)
            elif mark == 0:
                walk(dep)
        visiting.pop()
        state[task_id] = 2

    for task in tasks:
        if state.get(task.task_id, 0) == 0:
            walk(task.task_id)
    return tuple(found)


def parse_task_board(text: str, prefix: str) -> BoardParse:
    """Read task headings. The first status line in a section wins."""

    if not prefix.startswith("## "):
        raise ValueError("task prefix must be a markdown heading")
    tasks: list[BoardTask] = []
    duplicate_ids: list[str] = []
    seen: set[str] = set()
    current_id = ""
    current: dict[str, str] | None = None
    sections: list[tuple[str, dict[str, str]]] = []

    def close() -> None:
        nonlocal current, current_id
        if current is None:
            return
        sections.append((current_id, current))
        current = None
        current_id = ""

    for line in text.splitlines():
        if line.startswith("## "):
            close()
            if not line.startswith(prefix):
                continue
            current_id = line[3:].split(None, 1)[0]
            current = {}
            continue
        if current is not None and line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            lowered = key.strip().lower()
            if lowered not in current:
                current[lowered] = value.strip()
    close()

    for task_id, fields in sections:
        if task_id in seen:
            duplicate_ids.append(task_id)
            continue
        seen.add(task_id)
        status = fields.get("status", "")
        tasks.append(BoardTask(
            task_id=task_id,
            status=status,
            depends_on=_ids(fields.get("depends on", "")),
            goal_id=fields.get("goal id", "") or fields.get("parent goal", ""),
            has_status="status" in fields,
        ))
    missing = tuple(task.task_id for task in tasks if not task.has_status)
    unknown = tuple(
        task.task_id for task in tasks
        if task.has_status and task.status not in _KNOWN_STATUS
    )
    return BoardParse(
        tasks=tuple(tasks),
        duplicate_ids=tuple(duplicate_ids),
        missing_status_ids=missing,
        unknown_status_ids=unknown,
        cycles=_cycles(tasks),
    )


def selectable_task_ids(tasks: Sequence[BoardTask]) -> tuple[str, ...]:
    """Tasks whose status is open and whose dependencies are all completed."""

    done = {task.task_id for task in tasks if task.status in _COMPLETED}
    return tuple(
        task.task_id for task in tasks
        if task.status in _OPEN and all(dep in done for dep in task.depends_on)
    )


def classify_board_stall(
    parsed: BoardParse,
    runtime: RuntimeSnapshot,
    *,
    idle_stall_seconds: float = 300.0,
    heartbeat_stale_seconds: float = 600.0,
) -> StallVerdict:
    """Classify one observation. Completion flags on the snapshot are ignored."""

    del heartbeat_stale_seconds  # quiet heartbeats do not become a restart
    if parsed.duplicate_ids:
        return StallVerdict(True, "duplicate_task_heading", "record_only", parsed.duplicate_ids)
    if parsed.cycles:
        involved = tuple(dict.fromkeys(task_id for cycle in parsed.cycles for task_id in cycle))
        return StallVerdict(True, "dependency_cycle", "record_only", involved)
    if parsed.unknown_status_ids:
        return StallVerdict(True, "unknown_task_status", "record_only", parsed.unknown_status_ids)
    if parsed.missing_status_ids:
        return StallVerdict(True, "missing_task_status", "repair_board", parsed.missing_status_ids)
    if not runtime.supervisor_alive:
        return StallVerdict(True, "supervisor_dead", "attention")
    selectable = selectable_task_ids(parsed.tasks)
    if runtime.active_task_id:
        return StallVerdict(False, "active_task", "none", (runtime.active_task_id,))
    incomplete = tuple(
        task.task_id for task in parsed.tasks
        if task.status not in _COMPLETED
    )
    if selectable:
        if runtime.idle_for_s >= idle_stall_seconds:
            return StallVerdict(True, "idle_incomplete", "attention", selectable)
        return StallVerdict(False, "selectable_grace", "none", selectable)
    if incomplete:
        return StallVerdict(True, "idle_incomplete", "successor_tasks", incomplete)
    return StallVerdict(False, "frontier_clear", "none")


def stall_signature(parsed: BoardParse, verdict: StallVerdict) -> str:
    """Stable id for one frontier. A repeated id does not spawn again."""

    body = {
        "reason": verdict.reason,
        "repair_kind": verdict.repair_kind,
        "tasks": [
            {
                "id": task.task_id,
                "status": task.status,
                "depends_on": list(task.depends_on),
            }
            for task in sorted(parsed.tasks, key=lambda item: item.task_id)
            if task.status not in _COMPLETED
        ],
        "duplicates": list(parsed.duplicate_ids),
        "missing": list(parsed.missing_status_ids),
        "unknown": list(parsed.unknown_status_ids),
        "cycles": [list(cycle) for cycle in parsed.cycles],
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def repair_missing_task_status(text: str, prefix: str) -> tuple[str, tuple[str, ...]]:
    """Insert ``- Status: blocked`` where a task heading has fields and no status.

    Existing status lines and dependency lines are copied unchanged.
    A heading with no field lines is left alone. Nothing is marked completed.
    """

    if not prefix.startswith("## "):
        raise ValueError("task prefix must be a markdown heading")
    lines = text.splitlines()
    output: list[str] = []
    inserted: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith(prefix):
            output.append(line)
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not lines[end].startswith("## "):
            end += 1
        block = lines[index:end]
        task_id = block[0][3:].split(None, 1)[0]
        has_status = any(item.startswith("- Status:") for item in block)
        has_field = any(item.startswith("- ") and ":" in item for item in block)
        output.append(block[0])
        cursor = 1
        if not has_status and has_field:
            if cursor < len(block) and block[cursor] == "":
                output.append("")
                cursor += 1
            output.append("- Status: blocked")
            inserted.append(task_id)
        output.extend(block[cursor:])
        index = end
    updated = "\n".join(output)
    if text.endswith("\n") or not text:
        updated += "\n"
    return updated, tuple(inserted)


def _flag_true(value: Any) -> bool:
    return value is True or value == "true"


def packet_refusal(packet: Mapping[str, Any] | None) -> str:
    """Return a refusal code, or an empty string when the packet is legal."""

    if not isinstance(packet, Mapping):
        return "packet_not_object"
    if _flag_true(packet.get("work_complete")) or _flag_true(packet.get("completion_authority")):
        return "completion_refused"
    effects = packet.get("requested_effects") or []
    if not isinstance(effects, list):
        return "effects_not_list"
    if any(item in FORBIDDEN_EFFECTS for item in effects):
        return "forbidden_effect"
    kind = str(packet.get("repair_kind") or "")
    if kind not in _SPAWN_KINDS:
        return "unknown_repair_kind"
    if kind == "repair_board" and not str(packet.get("prefix") or "").startswith("## "):
        return "repair_prefix_missing"
    return ""


def _completed_kinds(
    state: Mapping[str, Any],
    signature: str,
    verdict: StallVerdict,
) -> list[str]:
    if state.get("repair_signature") == signature:
        raw = state.get("completed_kinds") or []
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, str)]
    if state.get("last_success_signature") == signature and verdict.repair_kind == "successor_tasks":
        return ["successor_subgoals"]
    return []


def decide_spawn(
    state: Mapping[str, Any],
    verdict: StallVerdict,
    signature: str,
    *,
    child_alive: bool,
    attempt_cap: int = _ATTEMPT_CAP,
) -> dict[str, Any]:
    """Choose one watchdog action. The same signature spawns at most once after success."""

    base = {
        "schema": BOARD_STALL_SCHEMA,
        "spawn": False,
        "repair_kind": "none",
        "signature": signature,
        "work_complete": False,
        "completion_authority": False,
        "reason": verdict.reason,
    }
    if not verdict.stalled or verdict.repair_kind == "none":
        return {**base, "action": "observe"}
    if child_alive:
        return {**base, "action": "subagent_running", "repair_kind": verdict.repair_kind}
    spawn_kind = "record_only" if verdict.repair_kind == "attention" else verdict.repair_kind
    if verdict.repair_kind == "successor_tasks":
        done = _completed_kinds(state, signature, verdict)
        remaining = [kind for kind in _SUCCESSOR_CHAIN if kind not in done]
        if not remaining:
            return {**base, "action": "already_repaired", "repair_kind": "successor_tasks"}
        spawn_kind = remaining[0]
    elif state.get("last_success_signature") == signature:
        return {**base, "action": "already_repaired", "repair_kind": verdict.repair_kind}
    attempts = int(state.get("attempts") or 0)
    if state.get("attempt_signature") != signature:
        attempts = 0
    if attempts >= attempt_cap:
        return {**base, "action": "repair_attempt_cap", "repair_kind": spawn_kind}
    return {
        **base,
        "action": "subagent_started",
        "spawn": True,
        "repair_kind": spawn_kind,
        "attempts": attempts + 1,
        "attempt_signature": signature,
        "task_ids": list(verdict.task_ids),
    }


def accept_receipt(receipt: Mapping[str, Any] | None, signature: str) -> dict[str, Any]:
    """Accept a child receipt only when it refuses completion and matches the frontier."""

    if not isinstance(receipt, Mapping):
        return {"accepted": False, "reason": "receipt_missing", "work_complete": False,
                "completion_authority": False}
    if _flag_true(receipt.get("work_complete")) or _flag_true(receipt.get("completion_authority")):
        return {"accepted": False, "reason": "completion_refused", "work_complete": False,
                "completion_authority": False}
    if receipt.get("signature") != signature:
        return {"accepted": False, "reason": "signature_mismatch", "work_complete": False,
                "completion_authority": False}
    reason = str(receipt.get("reason") or "")
    if reason.startswith("forbidden") or reason == "completion_refused":
        return {"accepted": False, "reason": reason or "forbidden_effect", "work_complete": False,
                "completion_authority": False}
    return {"accepted": True, "reason": "receipt_accepted", "work_complete": False,
            "completion_authority": False}


def _proposal_refusal(proposals: Sequence[Any] | None) -> str:
    if proposals is None:
        return ""
    if isinstance(proposals, (str, bytes)) or not isinstance(proposals, Sequence):
        return "proposals_not_a_list"
    for item in proposals:
        if not isinstance(item, Mapping):
            return "proposal_not_object"
        if item.get("status") != "blocked":
            return "add_ready_task"
        command = str(item.get("command") or "")
        validation = str(item.get("validation") or "")
        if not command or not validation or any(char in command + validation for char in _UNSAFE_COMMAND):
            return "unsafe_task_command"
        acceptance = str(item.get("acceptance") or "")
        if not acceptance.strip():
            return "acceptance_missing"
        lowered = acceptance.casefold()
        if "completion authority" in lowered or "mark the parent" in lowered or "mark parent" in lowered:
            return "completion_refused"
        depends_on = item.get("depends_on")
        if not isinstance(depends_on, (list, tuple)) or not [dep for dep in depends_on if str(dep).strip()]:
            return "dependency_missing"
        if not str(item.get("task_id") or "").strip() or not str(item.get("goal_id") or "").strip():
            return "proposal_identity_missing"
        if not str(item.get("predicted_files") or "").strip():
            return "predicted_files_missing"
    return ""


def append_blocked_successor_tasks(
    text: str,
    prefix: str,
    proposals: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    """Append blocked successor tasks. Existing status and dependency lines stay."""

    if not prefix.startswith("## "):
        raise ValueError("task prefix must be a markdown heading")
    token = prefix[3:]
    present = {
        line[3:].split(None, 1)[0]
        for line in text.splitlines()
        if line.startswith(prefix)
    }
    sections: list[str] = []
    inserted: list[str] = []
    for item in proposals:
        if item.get("status") != "blocked":
            raise ValueError("a successor task must stay blocked")
        task_id = str(item.get("task_id") or "")
        if not task_id.startswith(token) or task_id in present or any(char.isspace() for char in task_id):
            continue
        depends = ", ".join(str(dep).strip() for dep in item.get("depends_on") or [] if str(dep).strip())
        title = " ".join(str(item.get("title") or "Successor task").split())[:180]
        acceptance = " ".join(str(item.get("acceptance") or "").split())
        sections.append(
            f"\n## {task_id} {title}\n\n"
            "- Status: blocked\n"
            f"- Depends on: {depends}\n"
            f"- Goal id: {item.get('goal_id')}\n"
            f"- Deterministic command: {item.get('command')}\n"
            f"- Validation: {item.get('validation')}\n"
            f"- Predicted files: {item.get('predicted_files')}\n"
            f"- Acceptance: {acceptance}\n"
        )
        present.add(task_id)
        inserted.append(task_id)
    if not sections:
        if text and not text.endswith("\n"):
            return text + "\n", ()
        return text, ()
    base = text if text.endswith("\n") or not text else text + "\n"
    return base + "".join(sections), tuple(inserted)


def apply_repair_packet(
    packet: Mapping[str, Any],
    *,
    board_text: str,
    successor: Callable[[], Mapping[str, Any]] | None = None,
    proposals: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply one packet. The returned board text keeps every existing status and dependency."""

    refusal = packet_refusal(packet)
    result: dict[str, Any] = {
        "schema": BOARD_STALL_RECEIPT_SCHEMA,
        "signature": str(packet.get("signature") or "") if isinstance(packet, Mapping) else "",
        "applied": False,
        "board_text": board_text,
        "work_complete": False,
        "completion_authority": False,
    }
    if refusal:
        result["reason"] = refusal
        return result
    kind = str(packet.get("repair_kind") or "")
    if kind == "repair_board":
        updated, inserted = repair_missing_task_status(board_text, str(packet.get("prefix") or ""))
        result.update(applied=True, reason="missing_status_inserted", board_text=updated,
                      inserted_status_ids=list(inserted))
        return result
    if kind == "successor_subgoals":
        if successor is None:
            result["reason"] = "successor_unavailable"
            return result
        planned = dict(successor())
        if _flag_true(planned.get("work_complete")) or _flag_true(planned.get("completion_authority")):
            result["reason"] = "completion_refused"
            return result
        result.update(
            applied=True,
            reason="successor_subgoals_recorded",
            repair_kind=kind,
            admitted=bool(planned.get("admitted")),
            content_id=str(planned.get("content_id") or ""),
            subgoal_count=int(planned.get("subgoal_count") or 0),
            task_ids=list(planned.get("task_ids") or []),
        )
        return result
    if kind == "successor_tasks":
        proposal_refusal = _proposal_refusal(proposals)
        if proposal_refusal:
            result["reason"] = proposal_refusal
            result["repair_kind"] = kind
            return result
        updated, inserted = append_blocked_successor_tasks(board_text, str(packet.get("prefix") or "## "), list(proposals or []))
        result.update(
            applied=True,
            reason="successor_tasks_recorded",
            repair_kind=kind,
            board_text=updated,
            inserted_task_ids=list(inserted),
        )
        return result
    result["reason"] = str(packet.get("reason") or "record_only")
    result["repair_kind"] = kind
    return result
