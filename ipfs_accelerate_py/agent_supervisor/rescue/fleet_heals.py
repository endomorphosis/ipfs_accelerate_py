"""Formal-logic fleet heals. Run before llm_router. Never forge completion."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

_RECEIPT_SEARCH_ROOTS = (
    "artifacts",
    "external/ipfs_accelerate/artifacts",
    "external/ipfs_datasets/artifacts",
)
_MAX_RECEIPT_BYTES = 1_000_000
_LOCAL_VALIDATION_TIMEOUT = 90
_ALLOWED_VALIDATORS = {"python3", "pytest", "/usr/bin/python3", "/usr/bin/pytest"}


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


def _blocked_task_ids(observation: Mapping[str, Any]) -> list[str]:
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    values = details.get("blocked_task_ids")
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str) and value.strip()][:8]


def _receipt_path(cwd: Path, task_id: str) -> Path | None:
    for relative in _RECEIPT_SEARCH_ROOTS:
        root = cwd / relative
        if not root.is_dir():
            continue
        for receipts in root.rglob("receipts"):
            if not receipts.is_dir():
                continue
            path = receipts / f"{task_id}.json"
            if path.is_file():
                return path
    return None


def _validation_command(payload: Mapping[str, Any]) -> tuple[list[str], str] | None:
    validation = payload.get("validation")
    if isinstance(validation, dict):
        commands = validation.get("commands")
        if isinstance(commands, list):
            for command in commands:
                if not isinstance(command, dict):
                    continue
                argv = command.get("argv")
                if (
                    isinstance(argv, list) and argv
                    and all(isinstance(item, str) and item for item in argv)
                    and argv[0] in _ALLOWED_VALIDATORS
                ):
                    rel = command.get("cwd")
                    return argv, rel if isinstance(rel, str) else ""
    profile = payload.get("validation_profile")
    if isinstance(profile, str) and profile:
        return None
    return None


def local_validation_already_recorded(state: Mapping[str, Any]) -> bool:
    """Do not re-run pytest every watchdog cycle after a recorded local pass."""
    observation = state.get("observation") if isinstance(state.get("observation"), dict) else {}
    result = state.get("last_action_result") if isinstance(state.get("last_action_result"), dict) else {}
    if result.get("recipe") != "local_validation_pending_native_admission":
        return False
    current = _blocked_task_ids(observation)
    if not current:
        return False
    prior = {
        item.get("task_id"): item.get("status")
        for item in result.get("results") or []
        if isinstance(item, dict)
    }
    return all(
        prior.get(task_id) in {"passed", "receipt_missing", "validation_unspecified"}
        for task_id in current
    )


def run_local_blocked_candidate_validation(
    board: Mapping[str, Any], observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Run declared local checks for blocked candidates. Never admits completion.

    Billing-locked CI is not an excuse to skip deterministic tests. Passing
    local checks do not flip native admission or rewrite 044/063 receipts.
    """
    cwd = Path(str(board.get("cwd") or ""))
    task_ids = _blocked_task_ids(observation)
    if not cwd.is_dir() or not task_ids:
        return {"status": "skip"}
    results = []
    for task_id in task_ids:
        path = _receipt_path(cwd, task_id)
        if path is None:
            results.append({"task_id": task_id, "status": "receipt_missing"})
            continue
        try:
            raw = path.read_bytes()
            if len(raw) > _MAX_RECEIPT_BYTES:
                results.append({"task_id": task_id, "status": "receipt_too_large"})
                continue
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            results.append({"task_id": task_id, "status": "receipt_unreadable"})
            continue
        if not isinstance(payload, dict):
            results.append({"task_id": task_id, "status": "receipt_unreadable"})
            continue
        command = _validation_command(payload)
        if command is None:
            results.append({"task_id": task_id, "status": "validation_unspecified",
                            "completion_authoritative": payload.get("completion_authoritative") is True})
            continue
        argv, relative = command
        workdir = (cwd / relative).resolve() if relative else cwd.resolve()
        root = cwd.resolve()
        if not workdir.is_dir() or (root not in workdir.parents and workdir != root):
            results.append({"task_id": task_id, "status": "validation_cwd_rejected"})
            continue
        try:
            completed = subprocess.run(
                argv, cwd=str(workdir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=_LOCAL_VALIDATION_TIMEOUT, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            results.append({"task_id": task_id, "status": "validation_unavailable"})
            continue
        results.append({
            "task_id": task_id,
            "status": "passed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "completion_authoritative": False,
        })
    if not results:
        return {"status": "skip"}
    passed = [item for item in results if item.get("status") == "passed"]
    failed = [item for item in results if item.get("status") == "failed"]
    ran = passed or failed
    if not ran:
        return {"status": "wait", "recipe": "todos_waiting_on_blocked_dependencies",
                "reason": "remaining todos depend on blocked peers; do not rewrite those receipts",
                "results": results}
    return {
        "status": "applied" if passed and not failed else "wait",
        "recipe": "local_validation_pending_native_admission",
        "completion_authoritative": False,
        "results": results,
        "reason": (
            "local checks passed; native fenced admission still required"
            if passed and not failed else
            "local checks did not pass; do not rewrite blocked receipts"
        ),
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
    if stall == "blocked_without_independent_work":
        if local_validation_already_recorded(state):
            return {"status": "wait", "recipe": "local_validation_pending_native_admission",
                    "completion_authoritative": False,
                    "reason": "local checks already recorded; native fenced admission still required"}
        local = run_local_blocked_candidate_validation(board, observation)
        if local.get("status") != "skip":
            return local
        return {"status": "wait", "recipe": "todos_waiting_on_blocked_dependencies",
                "reason": "remaining todos depend on blocked peers; do not rewrite those receipts"}
    if stall == "closeout_waiting_on_unsettled_goals":
        return {"status": "wait", "recipe": "native_goals_still_active",
                "reason": "all-tasks-complete is not goal closeout; native owner still has active goals"}
    if stall == "native_status_unavailable_with_live_workers":
        return {"status": "wait", "recipe": "native_status_retry_with_live_workers",
                "reason": "nonzero native status is not a coding stall while lanes are live"}
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
