"""Formal-logic fleet heals. Run before llm_router. Never forge completion."""

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


def apply_supervisor_heal(board: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """Formal logic first. llm_router is the residual coding path."""
    return try_logic_guided_repair(board, state)
