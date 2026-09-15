"""Formal-logic fleet heals. Run before llm_router. Never forge completion."""

from __future__ import annotations

import subprocess
from pathlib import Path
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


def try_logic_guided_repair(board: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """Consult the logic-guided materializer. No write without an admitted plan."""
    try:
        from ipfs_accelerate_py.agent_supervisor.proof.logic_guided_repair_packet import (
            LOGIC_GUIDED_REPAIR_PACKET_MATERIALIZER_INTERFACE,
            LogicGuidedRepairPacketMaterializer,
            MaterializationDisposition,
        )
    except Exception as exc:
        return {"status": "needs_llm", "recipe": "logic_unavailable",
                "reason": type(exc).__name__}
    # Fleet incidents are not admitted RPR packets. Importing the materializer
    # binds this loop to the logic submodule; residual coding uses llm_router.
    _ = LogicGuidedRepairPacketMaterializer
    return {
        "status": "needs_llm",
        "recipe": "llm_router",
        "logic_interface": LOGIC_GUIDED_REPAIR_PACKET_MATERIALIZER_INTERFACE,
        "logic_disposition": MaterializationDisposition.ADMISSION_REQUIRED.value,
        "reason": "logic required admitted plan; residual is llm_router Grok then Codex",
    }


def restore_dirty_control_plane(board: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    """Drop uncommitted control-plane dirt back to HEAD. Never forges completion."""
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    integrity = details.get("source_integrity") if isinstance(details.get("source_integrity"), dict) else {}
    if integrity.get("reason") != "configured_control_plane_dirty":
        return {"status": "skip"}
    checked = integrity.get("checked") if isinstance(integrity.get("checked"), list) else []
    restored = []
    for entry in checked:
        if not isinstance(entry, dict):
            continue
        root = Path(str(entry.get("repository") or ""))
        paths = [str(path) for path in entry.get("paths") or [] if isinstance(path, str) and path]
        allowed = [path for path in paths if path in {
            "ipfs_accelerate_py/agent_supervisor", "scripts/ops/agent_supervisor",
        } or path.startswith("ipfs_accelerate_py/agent_supervisor/")
          or path.startswith("scripts/ops/agent_supervisor/")]
        if not root.is_dir() or not allowed:
            continue
        completed = subprocess.run(
            ["git", "-C", str(root), "checkout", "--", *allowed],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False,
        )
        if completed.returncode == 0:
            restored.append(str(root))
    if restored:
        return {"status": "applied", "recipe": "restore_dirty_control_plane", "restored": restored}
    return {"status": "skip"}


def apply_supervisor_heal(board: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """Formal logic first. llm_router is the residual coding path."""
    observation = state.get("observation") if isinstance(state.get("observation"), dict) else {}
    stall = str(state.get("stall_class") or "")
    dirty = restore_dirty_control_plane(board, observation)
    if dirty.get("status") == "applied":
        return dirty
    if stall == "independent_work_beside_blocked_peer" and live_workers(observation):
        return {"status": "wait", "recipe": "independent_work_has_live_workers",
                "reason": "blocked peers stay blocked; live lanes own independent todos"}
    if stall == "independent_todos_unclaimed":
        return {"status": "wait", "recipe": "native_lanes_own_independent_todos",
                "reason": "blocked receipts stay blocked; live native lanes claim independent todos"}
    if stall == "in_progress_awaiting_effect":
        return {"status": "wait", "recipe": "in_progress_awaiting_effect",
                "reason": "in-progress tasks are native work, not a coding stall"}
    if stall == "kernel_uninterruptible_wait":
        return {"status": "wait", "recipe": "kernel_uninterruptible_wait",
                "reason": "D-state I/O is not a coding stall; do not signal or rewrite receipts"}
    if stall == "board_checkout_missing":
        return {"status": "wait", "recipe": "deleted_checkout_not_rematerialized",
                "reason": "missing checkout is not reconstructed; retain original authority or explicit retirement"}
    return try_logic_guided_repair(board, state)
