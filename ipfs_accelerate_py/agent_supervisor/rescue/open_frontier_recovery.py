"""Recover an open task frontier without marking work complete.

A structural board fault stays with the board-stall watchdog. Selectable
work and no live worker is idle. A task left ``in_progress`` by a dead
worker is still that idle frontier. This module names the first selectable
task and asks ``llm_router`` which provider may run. It does not launch a
process, edit a board, or grant completion.

The running watchdog stores a repair latch as the reason text and does not
reload its script. A later selectable task with the same reason must not
stay behind that latch.
"""

from __future__ import annotations

from typing import Any, Callable

from ipfs_accelerate_py.agent_supervisor.rescue.board_stall_watchdog import (
    BoardParse,
    selectable_task_ids,
)

SCHEMA = "ipfs_accelerate_py/agent-supervisor/open-frontier-recovery@1"
_COMPLETED = frozenset({"completed", "complete", "done"})
_ROUTED = frozenset({"grok", "codex"})


def classify_open_work(parsed: BoardParse, *, worker_alive: bool) -> dict[str, Any]:
    """Classify one parsed board. Structural defects are not decided here."""

    selectable = selectable_task_ids(parsed.tasks)
    active = tuple(task.task_id for task in parsed.tasks if task.status == "in_progress")
    incomplete = tuple(
        task.task_id for task in parsed.tasks if task.status not in _COMPLETED
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "stalled": False,
        "reason": "frontier_clear",
        "repair": "none",
        "task_id": "",
        "task_ids": [],
        "work_complete": False,
        "completion_authority": False,
    }
    if worker_alive:
        payload["reason"] = "worker_active"
        payload["task_id"] = active[0] if active else ""
        payload["task_ids"] = list(active)
        return payload
    if selectable:
        payload.update(
            stalled=True,
            reason="idle_with_open_work",
            repair="route_model",
            task_id=selectable[0],
            task_ids=list(selectable),
        )
        return payload
    if incomplete:
        payload.update(
            stalled=True,
            reason="idle_blocked_frontier",
            repair="successor_tasks",
            task_id=incomplete[0],
            task_ids=list(incomplete),
        )
        return payload
    return payload


def frontier_latch_action(
    latch_text: str,
    *,
    reason: str,
    task_id: str,
    recorded_task_id: str,
) -> dict[str, Any]:
    """Say whether a reason-only repair latch still blocks this frontier.

    The watchdog compares the latch file to the stall reason. The same
    reason on a later task is not the same frontier. The first time a
    reason-only latch is seen, the current task is recorded and the latch
    stays. A different recorded task releases it. No provider is ranked.
    """

    held = str(latch_text or "").strip()
    current = str(task_id or "").strip()
    recorded = str(recorded_task_id or "").strip()
    stall_reason = str(reason or "").strip()
    record = current if stall_reason == "idle_with_open_work" and current else recorded
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "release": False,
        "hold": False,
        "record_task_id": record,
        "work_complete": False,
        "completion_authority": False,
    }
    if held != "idle_with_open_work" or stall_reason != "idle_with_open_work" or not current:
        return payload
    if recorded and recorded != current:
        payload["release"] = True
        return payload
    payload["hold"] = True
    return payload


def select_route(
    grok_ready: bool,
    codex_ready: bool,
    *,
    decider: Callable[[bool, bool], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ask llm_router which provider may run. Observations are not re-ranked."""

    decision = (
        decider(grok_ready, codex_ready)
        if decider is not None
        else _router_decision(grok_ready, codex_ready)
    )
    provider = str(decision.get("selected_provider") or "")
    authorized = bool(decision.get("authorized")) and provider in _ROUTED
    return {
        "schema": SCHEMA,
        "authorized": authorized,
        "selected_provider": provider if authorized else "",
        "reason_codes": list(decision.get("reason_codes") or []),
        "work_complete": False,
        "completion_authority": False,
    }


def _router_decision(grok_ready: bool, codex_ready: bool) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_provider_auto import (
        probe_llm_router_backends,
        select_implementation_provider,
    )

    observations = probe_llm_router_backends(
        grok_binary=bool(grok_ready),
        grok_authenticated=bool(grok_ready),
        codex_binary=bool(codex_ready),
        codex_authenticated=bool(codex_ready),
    )
    selection = select_implementation_provider(observations)
    provider = str(selection.selected_provider or "")
    return {
        "selected_provider": provider,
        "authorized": selection.decision.value in _ROUTED and provider in _ROUTED,
        "reason_codes": list(selection.reason_codes),
    }
