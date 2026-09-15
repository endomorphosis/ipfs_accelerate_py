"""Formal-logic fleet heals. Run before any LLM repair. Never forge completion."""

from __future__ import annotations

from typing import Any, Mapping


def live_workers(observation: Mapping[str, Any]) -> bool:
    """A live daemon or in-progress task is already doing independent work."""
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    lanes = details.get("lanes") if isinstance(details.get("lanes"), list) else []
    if any(isinstance(lane, dict) and lane.get("daemon") for lane in lanes):
        return True
    counts = details.get("task_counts") if isinstance(details.get("task_counts"), dict) else {}
    try:
        return int(counts.get("in_progress") or 0) > 0
    except (TypeError, ValueError):
        return False


def kernel_uninterruptible(reasons: list[str]) -> bool:
    return any("process_uninterruptible" in reason for reason in reasons)


def apply_supervisor_heal(board: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one in-process recipe. Codex is only the residual path."""
    observation = state.get("observation") if isinstance(state.get("observation"), dict) else {}
    stall = str(state.get("stall_class") or "")
    reasons = [str(item) for item in observation.get("reason_codes") or []]
    if stall == "configured_control_plane_dirty" or "source_integrity_not_verified" in reasons:
        return {
            "status": "wait",
            "recipe": "dirty_tracked_control_plane",
            "reason": "uncommitted control-plane edits are not Codex work",
        }
    if stall == "independent_work_beside_blocked_peer" and live_workers(observation):
        return {
            "status": "wait",
            "recipe": "independent_work_has_live_workers",
            "reason": "blocked peers stay blocked; live lanes claim independent todos",
        }
    if kernel_uninterruptible(reasons):
        return {
            "status": "wait",
            "recipe": "kernel_uninterruptible_wait",
            "reason": "D-state is not an LLM coding job",
        }
    return {"status": "needs_llm", "recipe": "none"}
