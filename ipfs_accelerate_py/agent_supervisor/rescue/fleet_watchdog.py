"""Persistent, bounded supervision of independent configured DuckDB boards.

The watchdog uses each board's existing operator API. It never opens a live
DuckDB file, completes tasks, clears holds, or signals arbitrary processes.
Commands are trusted operator configuration, never taken from probe output.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .live_board_probe import COMPLETED, read_json_object

SCHEMA = "agent-supervisor/fleet-watchdog@1"
HEALTH = {"healthy", "degraded", "blocked", "stalled", "stopped", "unknown", "complete"}
WAIT_STALLS = {
    "in_progress_awaiting_effect",
    "independent_work_beside_blocked_peer",
    "independent_todos_unclaimed",
    "blocked_without_independent_work",
    "missing_independent_clause_evidence",
    "closeout_requires_native_authority",
    "closeout_waiting_on_unsettled_goals",
    "operator_hold",
    "board_checkout_missing",
    "kernel_uninterruptible_wait",
    "native_status_unavailable_with_live_workers",
    "complete",
}
FLEET_HEALTH_SCHEMA = "ipfs_accelerate_py/agent-supervisor/ducklake-fleet-health@1"
# Publisher-logic holds. Retry publish after the supervisor tree heals;
# an LLM cannot mint gitlinks, native authority, or GitHub review state.
PUBLICATION_AUTOHEAL_STALLS = {
    "nested_leaf_gitlinks_travel_with_source",
    "nested_source_head_from_parent_gitlink",
    "bootstrap_mode_after_native_authority",
    "publication_awaiting_github_review",
    "publication_lock_busy",
    "publication_command_timeout",
    "publication_integration_diverged_from_accepted_source",
    "publication_github_actions_billing_locked",
}
# Reserved for holds that must not retry publish or enqueue LLM.
PUBLICATION_STOP_STALLS = set()


def missing_board_checkout(board: dict[str, Any]) -> dict[str, Any] | None:
    """Configured checkout paths that must exist before a probe subprocess can run.

    A deleted worktree must not become probe_failed. The probe adapter cannot
    start when cwd is gone, and FileNotFoundError is not a coding stall.
    """
    for key in ("cwd", "state_root", "owner_status_path"):
        value = board.get(key)
        if not isinstance(value, str) or not value:
            continue
        if not Path(value).exists():
            return {"missing": key, "path": value}
    return None


def checkout_missing_observation(board_id: str, missing: dict[str, Any]) -> dict[str, Any]:
    return {
        "board_id": board_id,
        "health": "unknown",
        "complete": False,
        "busy": False,
        "progress_token": "",
        "reason_codes": ["board_checkout_missing"],
        "details": dict(missing),
    }


def _live_daemons(details: dict[str, Any]) -> bool:
    lanes = details.get("lanes") if isinstance(details.get("lanes"), list) else []
    return any(isinstance(lane, dict) and lane.get("daemon") for lane in lanes)


def _uninterruptible_lanes(details: dict[str, Any]) -> bool:
    lanes = details.get("lanes") if isinstance(details.get("lanes"), list) else []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        if lane.get("stalled_without_active_worker") is True:
            continue
        if not (lane.get("claimed") or lane.get("task")):
            continue
        for role in ("daemon", "supervisor"):
            identity = lane.get(role) if isinstance(lane.get(role), dict) else {}
            if identity.get("process_state") == "D":
                return True
    return False


def _stale_in_progress_without_workers(details: dict[str, Any]) -> bool:
    """True when in-progress counts have no live claimed worker."""
    lanes = details.get("lanes") if isinstance(details.get("lanes"), list) else []
    named = [lane for lane in lanes if isinstance(lane, dict)]
    if not named:
        return False
    return all(
        lane.get("stalled_without_active_worker") is True
        or not (lane.get("claimed") or lane.get("task"))
        for lane in named
    )


def _native_admission_work(details: dict[str, Any], counts: dict[str, Any],
                           reasons: set[str]) -> bool:
    """Native extra-gate still has work to admit. Do not treat that as recursion."""
    try:
        if int(counts.get("in_progress") or 0) > 0 and not _stale_in_progress_without_workers(details):
            return True
        if int(counts.get("retrying") or 0) > 0:
            return True
        if int(counts.get("blocked") or 0) > 0:
            return True
        if int(counts.get("todo") or 0) > 0 and _live_daemons(details):
            return True
    except (TypeError, ValueError):
        return False
    if _uninterruptible_lanes(details):
        return True
    return any("process_uninterruptible" in str(reason) for reason in reasons)


def _todos_are_ready_beside_blocked(details: dict[str, Any], counts: dict[str, Any],
                                    reasons: set[str]) -> bool:
    """Remaining todos are independent only when native readiness says they are claimable."""
    if int(counts.get("in_progress") or 0) > 0:
        return True
    if "no_ready_independent_tasks" in reasons:
        return False
    if details.get("selection_idle_reason") == "no_ready_tasks":
        return False
    ready = details.get("ready_count")
    if type(ready) is int and not isinstance(ready, bool):
        return ready > 0
    eligible = details.get("eligible_ready_count")
    if type(eligible) is int and not isinstance(eligible, bool):
        return eligible > 0
    return int(counts.get("todo") or 0) > 0


def _locally_validated_blocked_ids(
    observation: dict[str, Any], previous: dict[str, Any] | None,
) -> bool:
    """True when every blocked id already passed current-tree pytest."""
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    blocked = [str(item) for item in details.get("blocked_task_ids") or [] if item]
    if not blocked:
        return False
    prior = previous if isinstance(previous, dict) else {}
    result = prior.get("last_action_result") if isinstance(prior.get("last_action_result"), dict) else {}
    passed = {
        str(item.get("task_id"))
        for item in result.get("results") or []
        if isinstance(item, dict) and item.get("status") == "passed" and item.get("task_id")
    }
    return set(blocked) <= passed and bool(passed)


def classify_stall(observation: dict[str, Any], previous: dict[str, Any] | None = None) -> str:
    """Map probe evidence to a bounded stall class. Never infers completion."""
    health = observation.get("health")
    reasons = {str(x) for x in observation.get("reason_codes") or []}
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    counts = details.get("task_counts") if isinstance(details.get("task_counts"), dict) else {}
    if "board_checkout_missing" in reasons:
        return "board_checkout_missing"
    if health == "operator_hold":
        return "operator_hold"
    if health == "complete" or observation.get("complete") is True:
        return "complete"
    if "source_integrity_not_verified" in reasons:
        # Supervisor-path dirt is restored separately. Remaining board-doc
        # dirt must not outrank live independent work beside blocked peers.
        if "board_has_blocked_or_quarantined_tasks" in reasons:
            if int(counts.get("in_progress") or 0) > 0:
                return "independent_work_beside_blocked_peer"
            if not _todos_are_ready_beside_blocked(details, counts, reasons):
                return "blocked_without_independent_work"
            if int(counts.get("todo") or 0) > 0 and _live_daemons(details):
                return "independent_todos_unclaimed"
            if int(counts.get("todo") or 0) > 0:
                return "independent_work_beside_blocked_peer"
        return "configured_control_plane_dirty"
    if "board_has_unsettled_goals" in reasons:
        # All-tasks-complete is not goal closeout. Native owner/cron still
        # owns the remaining goals; an LLM cannot mint that admission.
        return "closeout_waiting_on_unsettled_goals"
    if (
        observation.get("completion_candidate") is True
        and observation.get("complete") is not True
        and health != "complete"
    ):
        # All-tasks-complete is a separate closeout review, not native
        # completion_authority. SPAR bootstrap closeout is this class.
        board_id = str(observation.get("board_id") or "").lower()
        if board_id in {"spar", ""} and details.get("native_completion_authority") is False:
            return "missing_independent_clause_evidence"
        return "closeout_requires_native_authority"
    if health == "stopped" and not details.get("owner_ready"):
        if (
            "owner_not_ready" in reasons
            or "owner_status_identity_missing" in reasons
            or bool(
                (details.get("owner") if isinstance(details.get("owner"), dict) else {}).get("pid")
            )
        ):
            return "owner_live_status_unreadable"
        return "owner_missing"
    owner = details.get("owner") if isinstance(details.get("owner"), dict) else {}
    if (
        "owner_not_ready" in reasons
        and (
            bool(owner.get("pid"))
            or "owner_status_identity_missing" in reasons
        )
    ):
        # Native extra-gate is live but not ready. Wait; do not ensure a second owner.
        return "owner_live_status_unreadable"
    if int(counts.get("in_progress") or 0) > 0 and _stale_in_progress_without_workers(details):
        # Stale in-progress rows without a claimed worker do not stall SAWM-like boards.
        if int(counts.get("todo") or 0) > 0:
            return "independent_todos_unclaimed"
        return "stalled_no_progress"
    if "extra_gate_recursion_sealed_package" in reasons:
        # Sealed extra-gate is the native owner. Native admission first.
        if not _native_admission_work(details, counts, reasons):
            return "extra_gate_recursion"
    if "board_has_blocked_or_quarantined_tasks" in reasons:
        if _locally_validated_blocked_ids(observation, previous):
            # Current-tree tests passed. DuckDB blocked→retrying is not the
            # remaining requirement for those tasks.
            if int(counts.get("in_progress") or 0) > 0:
                return "in_progress_awaiting_effect"
            if int(counts.get("todo") or 0) > 0 and _live_daemons(details):
                return "independent_todos_unclaimed"
            if int(counts.get("todo") or 0) > 0:
                return "independent_work_beside_blocked_peer"
        if int(counts.get("in_progress") or 0) > 0:
            return "independent_work_beside_blocked_peer"
        if not _todos_are_ready_beside_blocked(details, counts, reasons):
            return "blocked_without_independent_work"
        if int(counts.get("todo") or 0) > 0 and _live_daemons(details):
            return "independent_todos_unclaimed"
        if int(counts.get("todo") or 0) > 0:
            return "independent_work_beside_blocked_peer"
        return "blocked_without_independent_work"
    if int(counts.get("in_progress") or 0) > 0 or int(counts.get("retrying") or 0) > 0:
        # Native in-progress or just-rearmed retrying work, including D-state
        # I/O, is not a coding stall. An LLM cannot unstick __flush_work or
        # rewrite live receipts.
        return "in_progress_awaiting_effect"
    if int(counts.get("todo") or 0) > 0 and _live_daemons(details):
        return "independent_todos_unclaimed"
    if any("process_uninterruptible" in reason for reason in reasons) or _uninterruptible_lanes(details):
        return "kernel_uninterruptible_wait"
    if "native_status_nonzero" in reasons and _live_daemons(details):
        # A nonzero native status with live lanes is not a coding stall.
        # Task counts may be missing for one sample; workers still own the board.
        return "native_status_unavailable_with_live_workers"
    if "native_operator_reports_unhealthy" in reasons and (
            _live_daemons(details) or details.get("owner_ready") is True):
        # Extra-gate unhealthy with a live owner is a native projection gap,
        # not llm_router. Empty task_counts often ride along with D-state I/O.
        return "native_status_unavailable_with_live_workers"
    if "no_task_progress" in reasons or health == "stalled":
        return "stalled_no_progress"
    if "probe_failed" in reasons:
        return "probe_failed"
    if health in {"degraded", "blocked", "unknown"}:
        if _live_daemons(details) or details.get("owner_ready") is True:
            return "native_status_unavailable_with_live_workers"
        return str(health)
    return "none"


def classify_publication_hold(receipt: dict[str, Any]) -> str:
    """Map a publisher receipt to a stall class. Never infers completion."""
    reason = str(receipt.get("reason") or "")
    if "changed gitlinks lack declared repository dependencies" in reason:
        return "nested_leaf_gitlinks_travel_with_source"
    if "source_ref must match its clean integration checkout HEAD" in reason:
        return "nested_source_head_from_parent_gitlink"
    if "current_rollout_mode_is_not_required" in reason:
        return "bootstrap_mode_after_native_authority"
    if "billing issue" in reason:
        return "publication_github_actions_billing_locked"
    if "local required checks failed" in reason:
        return "publication_local_required_checks_failed"
    if any(token in reason for token in (
            "publication pull request created",
            "required GitHub checks",
            "publication PR is not ready",
    )):
        return "publication_awaiting_github_review"
    if "another publisher holds this board's lock" in reason:
        return "publication_lock_busy"
    if "timed out after" in reason:
        return "publication_command_timeout"
    if "publication validation failed" in reason and any(
            token in reason for token in ("ImportError", "AttributeError", "AssertionError")
    ):
        return "publication_integration_diverged_from_accepted_source"
    if "publication validation failed" in reason:
        return "publication_validation_failed"
    return "publication_held"


def write_ducklake_fleet_health(root: Path, report: dict[str, Any]) -> None:
    """Observational DuckLake fleet snapshot. Not completion authority."""
    boards = {}
    for board_id, state in (report.get("boards") or {}).items():
        if not isinstance(state, dict):
            continue
        observation = state.get("observation") if isinstance(state.get("observation"), dict) else {}
        details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
        boards[str(board_id)] = {
            "health": state.get("health"),
            "stall_class": state.get("stall_class") or (
                classify_stall(observation) if observation else "watchdog_error"
            ),
            "reason_codes": list(observation.get("reason_codes") or []),
            "complete": bool(observation.get("complete")),
            "owner_ready": bool(details.get("owner_ready")),
            "planned_action": state.get("planned_action") or "",
        }
    write_json(root / "ducklake_fleet_health.json", {
        "schema": FLEET_HEALTH_SCHEMA,
        "completion_authority": False,
        "observed_at": report.get("observed_at"),
        "apply": bool(report.get("apply")),
        "boards": boards,
    })


def hold_paths(board: dict[str, Any]) -> list[str]:
    """A configured stop marker remains present even if its symlink is dangling."""
    return [str(path) for path in board.get("hold_files", []) if os.path.lexists(path)]


def repair_hold_paths(board: dict[str, Any]) -> list[str]:
    """Keep full stops while permitting repair under explicit launch custody.

    Scope comes from operator configuration, never from interpreting marker
    text. The marker stays in place and still forbids watchdog ensure/start.
    Reserved full-stop markers and symlinks cannot become launch-only holds.
    """
    holds = hold_paths(board)
    scoped = board.get("launch_only_hold_files", [])
    if not isinstance(scoped, list) or any(not isinstance(p, str) for p in scoped):
        return holds
    return [p for p in holds if p not in scoped or Path(p).is_symlink()
            or Path(p).name in {"HOLD", "OPERATOR_STOP", "watchdog.disabled"}]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return read_json_object(path)
    except FileNotFoundError:
        return {}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def supervisor_pythonpath(existing: str = "") -> str:
    """Keep fleet probes on this supervisor tree, not a stale sealed release."""
    root = str(Path(__file__).resolve().parents[3])
    prior = [part for part in str(existing or "").split(":") if part and part != root]
    return ":".join([root, *prior])


def command(spec: dict[str, Any], *, cwd: str, timeout: float = 60) -> dict[str, Any]:
    """Bound time and memory, including descendants holding stdout open.

stdout/stderr go to temporary files, avoiding PIPE hangs from detached board
launchers. On timeout only this invocation's newly-created process group is
terminated; existing owners are never selected or killed.
    """
    argv = spec.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
        raise ValueError("command argv must be a nonempty string list")
    timeout_seconds = float(spec.get("timeout_seconds", timeout))
    termination_grace = float(spec.get("termination_grace_seconds", 5))
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("command timeout must be positive and finite")
    if not math.isfinite(termination_grace) or not 0 <= termination_grace <= 60:
        raise ValueError("termination grace must be finite and between 0 and 60 seconds")
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in spec.get("env", {}).items()})
    env["PYTHONPATH"] = supervisor_pythonpath(env.get("PYTHONPATH", ""))
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        started = time.monotonic()
        proc = subprocess.Popen(argv, cwd=spec.get("cwd", cwd), env=env,
                                stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            # The launcher can exit on TERM while a grandchild ignores it.
            # Retain the unreaped leader PID until its entire original group
            # has been fenced, so the group ID cannot be reused during grace.
            deadline = time.monotonic() + termination_grace
            while time.monotonic() < deadline:
                try:
                    os.killpg(proc.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        def tail(handle: Any) -> str:
            size = handle.tell()
            handle.seek(max(0, size - 65536))
            return handle.read(65536).decode("utf-8", "replace")
        return {"returncode": proc.returncode, "timed_out": timed_out,
                "duration_seconds": round(time.monotonic() - started, 3),
                "stdout": tail(out), "stderr": tail(err)}


def normalize_probe(board_id: str, result: dict[str, Any]) -> dict[str, Any]:
    try:
        if result["returncode"] != 0 or result.get("timed_out"):
            raise ValueError("probe failed or timed out")
        observation = json.loads(result["stdout"])
        if not isinstance(observation, dict) or observation.get("board_id") != board_id:
            raise ValueError("probe returned wrong board identity")
        if observation.get("health") not in HEALTH:
            raise ValueError("probe returned unknown health state")
        if not isinstance(observation.get("reason_codes", []), list):
            raise ValueError("probe reasons must be a list")
        return observation
    except (ValueError, KeyError, TypeError) as exc:
        return {"board_id": board_id, "health": "unknown", "complete": False,
                "reason_codes": ["probe_failed"], "details": {"error": str(exc),
                "returncode": result.get("returncode"), "timed_out": result.get("timed_out")}}


def _accepted_progress_counts(observation: dict[str, Any]) -> dict[str, int]:
    details = observation.get("details", {})
    if not isinstance(details, dict) or details.get("authenticated_task_observation") is not True:
        return {}
    counts = details.get("task_counts")
    completed = None
    if isinstance(counts, dict):
        totals = [counts.get(status, 0) for status in COMPLETED]
        if all(type(value) is int and value >= 0 for value in totals):
            completed = sum(totals)
    values = {"completed": completed,
              "receipts": details.get("completion_receipt_count"),
              "unsettled_goals": details.get("unsettled_goal_count")}
    return {key: value for key, value in values.items() if type(value) is int and value >= 0}


def _ready_owner_epoch(details: Any) -> list[Any] | None:
    """A fresh authenticated owner may establish a new bounded launch budget.

    Work can remain blocked throughout a healthy owner's lifetime. Its repair
    queue activity must not consume the attempts to restart a later stopped
    owner. This observation supplies no task progress or completion authority.
    """
    if not isinstance(details, dict):
        return None
    owner = details.get("owner")
    custody = details.get("owner_writer_custody")
    if (details.get("authenticated_task_observation") is not True
            or details.get("owner_ready") is not True
            or not isinstance(owner, dict) or not isinstance(custody, dict)
            or custody.get("configured") is not True
            or custody.get("verified") is not True or custody.get("held") is not True):
        return None
    pid, birth, boot = (owner.get(key) for key in ("pid", "start_time_ticks", "boot_id"))
    if (type(pid) is not int or pid <= 1 or type(birth) is not int or birth <= 0
            or not isinstance(boot, str) or not boot):
        return None
    return [pid, birth, boot]


def _carry_observational_task_counts(
    observation: dict[str, Any], previous: dict[str, Any],
) -> dict[str, Any]:
    """Keep last native counts when extra-gate is live but this probe is empty.

    Never treats carried counts as completion_authority.
    """
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    counts = details.get("task_counts") if isinstance(details.get("task_counts"), dict) else {}
    if counts:
        return observation
    if details.get("owner_ready") is not True and not _live_daemons(details):
        return observation
    prior_obs = previous.get("observation") if isinstance(previous.get("observation"), dict) else {}
    prior = prior_obs.get("details") if isinstance(prior_obs.get("details"), dict) else {}
    prior_counts = prior.get("task_counts") if isinstance(prior.get("task_counts"), dict) else {}
    if not prior_counts:
        return observation
    details = dict(details)
    details["task_counts"] = dict(prior_counts)
    prior_blocked = prior.get("blocked_task_ids")
    if not details.get("blocked_task_ids") and isinstance(prior_blocked, list):
        details["blocked_task_ids"] = [item for item in prior_blocked if isinstance(item, str)]
    details["task_counts_source"] = "carried_last_native_projection"
    observation = dict(observation, details=details)
    blocked_ids = details.get("blocked_task_ids")
    if isinstance(blocked_ids, list) and blocked_ids:
        reasons = [str(item) for item in observation.get("reason_codes") or []]
        if "board_has_blocked_or_quarantined_tasks" not in reasons:
            observation["reason_codes"] = [*reasons, "board_has_blocked_or_quarantined_tasks"]
    return observation


def assess(observation: dict[str, Any], previous: dict[str, Any], board: dict[str, Any], now: float) -> dict[str, Any]:
    state = dict(previous)
    observation = _carry_observational_task_counts(observation, previous)
    state.update(board_id=board["id"], observed_at=now, observation=observation)
    token = observation.get("progress_token")
    details = observation.get("details", {})
    previous_details = previous.get("observation", {}).get("details", {})
    native_probe = any(isinstance(value, dict) and "authenticated_task_observation" in value
                       for value in (details, previous_details))
    if "task_progress_counts" in previous or native_probe:
        # Status/retry cycling and temporarily missing authority are not new
        # accepted work. Persist monotone evidence across either kind of churn.
        prior = dict(previous.get("task_progress_counts",
                     _accepted_progress_counts(previous.get("observation", {}))))
        progressed = False
        for key, value in _accepted_progress_counts(observation).items():
            if key not in prior:
                prior[key] = value
            elif (value < prior[key] if key == "unsettled_goals" else value > prior[key]):
                prior[key] = value
                progressed = True
        state["task_progress_counts"] = prior
        if progressed:
            state["last_progress_at"] = now
    elif token and token != previous.get("progress_token"):
        state["last_progress_at"] = now
    if token:
        state["progress_token"] = token
    state.setdefault("last_progress_at", now)
    # A newly healthy heartbeat is not work progress. Stale busy flags are the
    # adapter's responsibility; only an explicit live activity probe sets busy.
    health = observation["health"]
    if (health == "healthy" and observation.get("completion_candidate") is not True
            and observation.get("busy") is not True
            and now - state["last_progress_at"] >= board.get("stall_seconds", 900)):
        health = "stalled"
        observation = dict(observation, health=health,
                           reason_codes=[*observation.get("reason_codes", []), "no_task_progress"])
        state["observation"] = observation
    state["stall_class"] = classify_stall(observation, previous)
    reasons = sorted(str(x) for x in observation.get("reason_codes", []))
    signature = hashlib.sha256(json.dumps([health, reasons]).encode()).hexdigest()[:20]
    if previous.get("incident_signature") != signature:
        state["incident_signature"] = signature
        # Diagnosis can vary while the same outage persists. Keep its grace
        # origin and retry budget, or alternating reasons can either postpone
        # recovery forever or bypass an in-flight action's durable cooldown.
        if previous.get("health") in {None, "healthy", "complete", "operator_hold"}:
            state["incident_since"] = now
    state.setdefault("incident_since", now)
    state.setdefault("attempts", 0)
    # Preserve a legacy unknown launch budget until positive readiness is seen.
    # Merely upgrading the watchdog cannot replay an interrupted launch.
    state.setdefault("ensure_attempts", previous.get("attempts", 0))
    epoch = _ready_owner_epoch(details)
    if epoch is not None and epoch != previous.get("ensure_owner_epoch"):
        state.update(ensure_attempts=0, ensure_owner_epoch=epoch)
    state.setdefault("next_action_at", 0)
    state["health"] = health
    if health == "healthy" and observation.get("completion_candidate") is not True:
        state.update(incident_since=now, attempts=0, ensure_attempts=0)
        state.pop("pending_action", None)
    return state


def _ensure_unit_name(board: dict[str, Any]) -> str:
    spec = board.get("ensure") if isinstance(board.get("ensure"), dict) else {}
    argv = spec.get("argv") if isinstance(spec.get("argv"), list) else []
    if (
        len(argv) < 4
        or argv[0] != "systemctl"
        or "start" not in argv
        or not str(argv[-1]).endswith(".service")
    ):
        return ""
    return str(argv[-1])


def _ensure_unit_already_active(board: dict[str, Any]) -> bool:
    """True when the configured exclusive-owner unit is already running."""
    unit = _ensure_unit_name(board)
    if not unit:
        return False
    try:
        completed = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and str(completed.stdout or "").strip() == "active"


def reset_failed_ensure_unit(board: dict[str, Any]) -> bool:
    """Clear systemd start-limit so a dirty-copy loop can start again."""
    unit = _ensure_unit_name(board)
    if not unit:
        return False
    try:
        completed = subprocess.run(
            ["systemctl", "--user", "reset-failed", unit],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def select_action(state: dict[str, Any], board: dict[str, Any], now: float) -> str:
    health = state["health"]
    if now < state.get("next_action_at", 0):
        stall = state.get("stall_class") or classify_stall(state.get("observation") or {})
        observation = state.get("observation") if isinstance(state.get("observation"), dict) else {}
        details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
        # A ready exclusive owner after ensure is not a 1h launch backoff.
        # Receipt materialization and unstall retry on cooldown.
        if not (
            (
                stall == "blocked_without_independent_work"
                and details.get("owner_ready") is True
                and state.get("last_action") == "ensure"
            )
            or stall in {
                "owner_missing",
                "independent_todos_unclaimed",
                "stalled_no_progress",
            }
        ):
            return ""
    if health == "complete" or state["observation"].get("complete") is True:
        stall = (state.get("publication_failure") or {}).get("stall_class")
        if stall in PUBLICATION_STOP_STALLS:
            return ""
        return "publish" if board.get("publication") else "completion_review"
    stall = state.get("stall_class") or classify_stall(state.get("observation") or {})
    if stall == "closeout_requires_native_authority":
        # Observational completion_candidate is not publication authority.
        return ""
    if stall == "closeout_waiting_on_unsettled_goals":
        reasons = {str(x) for x in (state.get("observation") or {}).get("reason_codes") or []}
        if "goal_closeout_disabled_on_launch" in reasons:
            from .fleet_heals import provisional_goal_closeout_already_recorded
            if provisional_goal_closeout_already_recorded(state):
                return ""
            return "supervisor_heal"
        return ""
    if stall == "missing_independent_clause_evidence":
        # Datasets producer still lacks current-bound clause records. An LLM
        # cannot mint those records or flip completion_authority.
        return ""
    if health == "healthy":
        return ""
    grace = board.get("failure_grace_seconds", 60) if health in {"stopped", "unknown"} else board.get("blocked_grace_seconds", 300)
    if now - state.get("incident_since", now) < grace:
        return ""
    # An inconclusive probe cannot authorize relaunching an already-live owner.
    recovery = state["observation"].get("recovery_action")
    if stall == "in_progress_awaiting_effect":
        from .fleet_heals import _observation_uninterruptible
        observation = state.get("observation") if isinstance(state.get("observation"), dict) else {}
        reasons = {str(x) for x in observation.get("reason_codes") or []}
        if _observation_uninterruptible(observation):
            # One D-state lane is not a board-wide freeze. Rearm false-terminal
            # blocks on other lanes; do not CAS the D-state worker's claim.
            return "supervisor_heal"
        if "extra_gate_recursion_sealed_package" in reasons:
            return "supervisor_heal"
        if "no_task_progress" in reasons:
            return "supervisor_heal"
        return ""
    if stall == "kernel_uninterruptible_wait":
        return "supervisor_heal"
    if stall == "stalled_no_progress":
        # Remaining SAWM/DOEP todos are not llm_router work.
        return "supervisor_heal"
    if stall == "independent_work_beside_blocked_peer":
        return "supervisor_heal"
    if stall == "independent_todos_unclaimed":
        return "supervisor_heal"
    if stall == "blocked_without_independent_work":
        # Remaining todos wait on blocked peers. Rearm false-terminal blocks,
        # then run declared local checks once. Never rewrite those receipts.
        # Keep selecting heal after a recorded local pass so unstall can still
        # rearm false-terminal blocks; pytest is skipped inside the recipe.
        return "supervisor_heal"
    if stall == "native_status_unavailable_with_live_workers":
        # Empty native counts with live extra-gate still need false-terminal
        # unstall (PCTDD-035/038). Wait after skip; never llm_router.
        return "supervisor_heal"
    if stall == "extra_gate_recursion":
        # Bind overlay heals to the live exclusive owner. Never ensure a
        # competing extra-gate unit while that owner is live.
        return "supervisor_heal"
    if stall in WAIT_STALLS:
        return ""
    if stall == "configured_control_plane_dirty":
        return "supervisor_heal"
    if stall == "owner_live_status_unreadable":
        # A live exclusive owner with a torn/failed status projection is not
        # owner-missing. Ensure would start a competing owner.
        return ""
    if stall == "owner_missing":
        # A missing exclusive owner is not an llm_router job. Clear overlay
        # copies if the last ensure failed dirty, then start the owner even
        # after max_ensure_attempts.
        result = state.get("last_action_result") if isinstance(state.get("last_action_result"), dict) else {}
        nested = result.get("supervisor_heal") if isinstance(result.get("supervisor_heal"), dict) else {}
        recipe = str(result.get("recipe") or nested.get("recipe") or "")
        if (
            state.get("last_action") == "ensure"
            and recipe != "clear_overlay_copies_for_owner_start"
        ):
            return "supervisor_heal"
        if board.get("ensure") and not hold_paths(board):
            if _ensure_unit_already_active(board):
                return ""
            return "ensure"
        return "supervisor_heal"
    if (board.get("ensure") and not hold_paths(board)
            and state.get("ensure_attempts", state.get("attempts", 0)) < board.get("max_ensure_attempts", 2)
            and recovery == "ensure"):
        return "ensure"
    return "repair"


def tick_board(board: dict[str, Any], state_root: Path, *, apply: bool = False,
               runner=command, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    board_id = board["id"]
    directory = state_root / board_id
    with lock(directory / "watchdog.lock") as acquired:
        if not acquired:
            return {"board_id": board_id, "health": "owned_by_another_watchdog"}
        path = directory / "state.json"
        previous = read_json(path)
        missing = missing_board_checkout(board)
        if missing:
            observation = checkout_missing_observation(board_id, missing)
        else:
            try:
                result = runner(board["probe"], cwd=board["cwd"], timeout=60)
            except Exception as exc:
                result = {"returncode": None, "stdout": "", "timed_out": False,
                          "stderr": f"{type(exc).__name__}: {exc}"}
            observation = normalize_probe(board_id, result)
        if "storage_checks" in board:
            from .storage_diagnostics import observe_storage
            diagnostics = observe_storage(board["storage_checks"])
            observation = dict(observation, storage_diagnostics=diagnostics)
            prior_diagnostics = previous.get("observation", {}).get("storage_diagnostics", {})
            if diagnostics["reason_codes"] or prior_diagnostics.get("reason_codes"):
                # Separate evidence only: storage conditions cannot change native
                # task health, completion authority, holds or recovery budgets.
                write_json(directory / "storage-incident.json", {
                    "schema": "agent-supervisor/storage-incident@1", "board_id": board_id,
                    "observed_at": now, "status": diagnostics["status"],
                    "action": "diagnostic_only", "diagnostics": diagnostics,
                })
        state = assess(observation, previous, board, now)
        if apply and state.get("stall_class") in WAIT_STALLS:
            job = read_json(state_root / "repairs" / board_id / "job.json")
            if job.get("status") == "running":
                try:
                    command({"argv": ["systemctl", "--user", "stop",
                                      "ipfs-taskboard-repair-job.service"]},
                            cwd="/", timeout=45)
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    pass
        # Holds fence mutations, not observation. A board assigned to another
        # owner still needs fresh health and progress evidence during a hold.
        state["observed_health"] = state["health"]
        holds = hold_paths(board)
        if repair_hold_paths(board):
            # A deletion hold still fences ensure/start. Keep a typed missing
            # checkout visible so the fleet does not hide it as probe_failed
            # or rematerialize the original authority.
            stall = state.get("stall_class") or classify_stall(state.get("observation") or {})
            state.update(
                health="operator_hold",
                stall_class="board_checkout_missing" if stall == "board_checkout_missing" else "operator_hold",
                holds=holds,
            )
            if apply:
                from .fleet_holds import review_board_holds
                result = review_board_holds(board, state.get("observation") or {})
                state.update(last_action="hold_review", last_action_at=now,
                             last_action_result=result, planned_action="")
                holds = hold_paths(board)
                state["holds"] = holds
                if not repair_hold_paths(board):
                    state["health"] = str(state.get("observed_health") or "healthy")
                    state["stall_class"] = classify_stall(state.get("observation") or {})
            else:
                state["planned_action"] = "hold_review"
            write_json(path, state)
            return state
        state.pop("holds", None)
        state["launch_only_holds"] = holds
        action = select_action(state, board, now)
        state["planned_action"] = action
        write_json(path, state)
        if not action:
            return state
        incident = {"schema": SCHEMA, "board_id": board_id, "observed_at": now,
                    "signature": state["incident_signature"], "observation": state["observation"],
                    "action": action, "attempts": state.get("attempts", 0),
                    "cwd": board["cwd"], "config": board.get("config", "")}
        incident_path = directory / "incident.json"
        write_json(incident_path, incident)
        if not apply:
            return state
        # A slow status command must not race an operator's newly placed hold.
        holds = hold_paths(board)
        if repair_hold_paths(board):
            stall = state.get("stall_class") or classify_stall(state.get("observation") or {})
            state.update(
                health="operator_hold",
                stall_class="board_checkout_missing" if stall == "board_checkout_missing" else "operator_hold",
                holds=holds, planned_action="",
            )
            write_json(path, state)
            return state
        if holds and action == "ensure":
            # A launch-custody marker may have arrived after action selection.
            action = "repair"
            state["planned_action"] = action
            state["launch_only_holds"] = holds
            incident["action"] = action
            write_json(incident_path, incident)
        attempts = state.get("attempts", 0) + 1
        # Record intent and backoff BEFORE side effects: interrupted watchdogs
        # must not repeatedly relaunch work or burn provider retry budgets.
        state.update(attempts=attempts, pending_action=action,
                     last_action_at=now, next_action_at=now + min(
                         board.get("max_backoff_seconds", 3600),
                         board.get("cooldown_seconds", 180) * 2 ** min(attempts - 1, 8)))
        if action == "ensure":
            state["ensure_attempts"] = state.get("ensure_attempts", 0) + 1
        write_json(path, state)
        if action == "supervisor_heal":
            from .fleet_heals import apply_supervisor_heal
            action_result = apply_supervisor_heal(board, state)
            reason = str((action_result or {}).get("reason") or "")
            recipe = str((action_result or {}).get("recipe") or "")
            if (
                reason.startswith("owner_cas_failed:")
                or reason == "quack_attach_token_absent"
                or recipe in {
                    "unstall_stale_native_work",
                    "local_validation_pending_native_admission",
                    "native_goals_still_active",
                    "provisionally_complete_terminal_goals",
                    "independent_work_has_live_workers",
                    "native_lanes_own_independent_todos",
                    "in_progress_awaiting_effect",
                    "todos_waiting_on_blocked_dependencies",
                    "kernel_uninterruptible_wait",
                    "native_status_retry_with_live_workers",
                    "collapse_extra_gate_recursion",
                    "unstall_stale_native_work",
                    "overlay_first_native_admission",
                    "rearm_locally_validated_blocked_tasks",
                    "clear_overlay_copies_for_owner_start",
                    "local_validation_satisfies_current_tree_requirements",
                    "successors_may_run_on_current_tree_evidence",
                    "stale_in_progress_does_not_stall_remaining_todos",
                    "overlay_current_tree_smoke",
                }
            ):
                # Unstall/false-terminal rearm must retry on cooldown, not 1h
                # max backoff, while native lanes own independent work.
                state["next_action_at"] = now + float(board.get("cooldown_seconds", 180))
            if recipe == "clear_overlay_copies_for_owner_start":
                state["ensure_attempts"] = 0
                state["next_action_at"] = now
        elif action == "publish":
            from .fleet_completion import publish_completed_board
            try:
                action_result = publish_completed_board(board["publication"], directory / "publication")
                if not isinstance(action_result, dict):
                    raise ValueError("publisher did not return a receipt object")
            except Exception as exc:
                action_result = {"status": "failed", "reason": f"publisher raised {type(exc).__name__}"}
            if action_result.get("status") != "published":
                # Merge conflicts and failed validation need implementation
                # repair. Publisher-logic holds retry publish after the
                # supervisor tree heals; an LLM cannot mint those receipts.
                reason_code = "publication_held" if action_result.get("status") == "held" else "publication_failed"
                stall = classify_publication_hold(action_result)
                failure = {
                    "reason_code": reason_code,
                    "stall_class": stall,
                    "status": action_result.get("status", "invalid_receipt"),
                    "reason": action_result.get("reason", "publication did not succeed"),
                    "receipt_path": str(directory / "publication/publication.json"),
                    "repositories": action_result.get("repositories", []),
                }
                autoheal = stall in PUBLICATION_AUTOHEAL_STALLS
                stop = stall in PUBLICATION_STOP_STALLS
                incident.update(
                    recovery_action="publish" if autoheal else ("none" if stop else "repair"),
                    publication_failure=failure,
                )
                incident["observation"] = dict(
                    state["observation"],
                    reason_codes=[*state["observation"].get("reason_codes", []), reason_code, stall],
                )
                write_json(incident_path, incident)
                state["publication_failure"] = failure
                if autoheal:
                    state["next_action_at"] = now + float(board.get("cooldown_seconds", 180))
                # Persist the diagnosis before enqueueing, so a watchdog crash
                # retains both the publication failure and the existing backoff.
                write_json(path, state)
                if autoheal or stop:
                    action_result["repair_result"] = {
                        "status": "autoheal_retry" if autoheal else "supervisor_heal_required",
                        "stall_class": stall,
                        "incident": str(incident_path),
                    }
                elif board.get("repair"):
                    spec = dict(board["repair"])
                    spec["argv"] = [*spec["argv"], "--incident", str(incident_path)]
                    repair_result = runner(spec, cwd=board["cwd"], timeout=120)
                    action_result["repair_result"] = {
                        k: v for k, v in repair_result.items() if k not in {"stdout", "stderr"}
                    }
                else:
                    action_result["repair_result"] = {"status": "repair_required", "incident": str(incident_path)}
            else:
                state.pop("publication_failure", None)
        elif action in {"repair", "completion_review"}:
            from .fleet_heals import apply_supervisor_heal
            heal = apply_supervisor_heal(board, state)
            if heal.get("status") in {"applied", "wait"}:
                action_result = {"status": heal["status"], "supervisor_heal": heal,
                                 "incident": str(incident_path)}
            elif not board.get("repair"):
                action_result = {"status": "repair_required", "incident": str(incident_path),
                                 "supervisor_heal": heal}
            else:
                spec = dict(board["repair"])
                spec["argv"] = [*spec["argv"], "--incident", str(incident_path)]
                action_result = runner(spec, cwd=board["cwd"], timeout=120)
                if isinstance(action_result, dict):
                    action_result = dict(action_result, supervisor_heal=heal)
        else:
            from .fleet_heals import clear_overlay_copies_blocking_owner_start
            cleared = clear_overlay_copies_blocking_owner_start(board)
            reset_failed_ensure_unit(board)
            action_result = runner(board["ensure"], cwd=board["cwd"], timeout=180)
            if isinstance(action_result, dict) and isinstance(cleared, dict):
                action_result = dict(action_result, overlay_copies=cleared)
        # Keep command output in private action evidence, not the shared status.
        write_json(directory / "last_action.json", {"at": now, "action": action, "result": action_result})
        state.pop("pending_action", None)
        state.update(last_action=action, last_action_result={
            k: v for k, v in action_result.items() if k not in {"stdout", "stderr"}})
        write_json(path, state)
        return state


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    boards = config.get("boards")
    if not isinstance(boards, list) or not boards:
        raise ValueError("at least one board is required")
    identifiers = set()
    for board in boards:
        identifier = board.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in identifiers:
            raise ValueError("board identifiers must be unique safe directory names")
        identifiers.add(identifier)
        if not Path(board["cwd"]).is_absolute() or not board.get("probe"):
            raise ValueError("each board needs an absolute cwd and a probe command")
        if "storage_checks" in board:
            from .storage_diagnostics import validate_storage_checks
            validate_storage_checks(board["storage_checks"])
    return config


def run_cycle(config: dict[str, Any], *, apply: bool) -> dict[str, Any]:
    root = Path(config["state_dir"])
    observations = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(config["boards"]))) as pool:
        work = {pool.submit(tick_board, board, root, apply=apply): board["id"] for board in config["boards"]}
        for future in concurrent.futures.as_completed(work):
            identifier = work[future]
            try:
                observations[identifier] = future.result()
            except Exception as exc:
                observations[identifier] = {"board_id": identifier, "health": "watchdog_error",
                                            "error": f"{type(exc).__name__}: {exc}"}
    report = {"schema": SCHEMA, "observed_at": time.time(), "apply": apply, "boards": observations}
    write_json(root / "status.json", report)
    write_ducklake_fleet_health(root, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    with lock(Path(config["state_dir"]) / "fleet.lock") as acquired:
        if not acquired:
            return 0
        while not stop.is_set():
            report = run_cycle(config, apply=args.apply)
            print(json.dumps({"observed_at": report["observed_at"], "boards": {
                key: value["health"] for key, value in report["boards"].items()}}), flush=True)
            if args.once:
                break
            stop.wait(max(5, float(config.get("poll_seconds", 60))))
            config = load_config(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
