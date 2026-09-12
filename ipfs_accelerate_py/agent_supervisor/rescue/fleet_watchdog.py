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


def assess(observation: dict[str, Any], previous: dict[str, Any], board: dict[str, Any], now: float) -> dict[str, Any]:
    state = dict(previous)
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


def select_action(state: dict[str, Any], board: dict[str, Any], now: float) -> str:
    health = state["health"]
    if now < state.get("next_action_at", 0):
        return ""
    if health == "complete" or state["observation"].get("completion_candidate") is True:
        return "publish" if board.get("publication") else "completion_review"
    if health == "healthy":
        return ""
    grace = board.get("failure_grace_seconds", 60) if health in {"stopped", "unknown"} else board.get("blocked_grace_seconds", 300)
    if now - state.get("incident_since", now) < grace:
        return ""
    # An inconclusive probe cannot authorize relaunching an already-live owner.
    recovery = state["observation"].get("recovery_action")
    if (recovery == "ensure" and board.get("ensure") and not hold_paths(board)
            and state.get("ensure_attempts", state.get("attempts", 0)) < board.get("max_ensure_attempts", 2)):
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
        try:
            result = runner(board["probe"], cwd=board["cwd"], timeout=60)
        except Exception as exc:
            result = {"returncode": None, "stdout": "", "timed_out": False,
                      "stderr": f"{type(exc).__name__}: {exc}"}
        observation = normalize_probe(board_id, result)
        state = assess(observation, previous, board, now)
        # Holds fence mutations, not observation. A board assigned to another
        # owner still needs fresh health and progress evidence during a hold.
        state["observed_health"] = state["health"]
        holds = hold_paths(board)
        if repair_hold_paths(board):
            state.update(health="operator_hold", holds=holds, planned_action="")
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
            state.update(health="operator_hold", holds=holds, planned_action="")
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
        if action == "publish":
            from .fleet_completion import publish_completed_board
            try:
                action_result = publish_completed_board(board["publication"], directory / "publication")
                if not isinstance(action_result, dict):
                    raise ValueError("publisher did not return a receipt object")
            except Exception as exc:
                action_result = {"status": "failed", "reason": f"publisher raised {type(exc).__name__}"}
            if action_result.get("status") != "published":
                # A merge conflict, stale completion gate, or failed validation
                # needs an implementation repair, not endless identical pushes.
                # Queue through the same bounded operator-owned repair command;
                # it deduplicates existing jobs and retains this exact evidence.
                reason_code = "publication_held" if action_result.get("status") == "held" else "publication_failed"
                failure = {
                    "reason_code": reason_code,
                    "status": action_result.get("status", "invalid_receipt"),
                    "reason": action_result.get("reason", "publication did not succeed"),
                    "receipt_path": str(directory / "publication/publication.json"),
                    "repositories": action_result.get("repositories", []),
                }
                incident.update(recovery_action="repair", publication_failure=failure)
                incident["observation"] = dict(
                    state["observation"],
                    reason_codes=[*state["observation"].get("reason_codes", []), reason_code],
                )
                write_json(incident_path, incident)
                state["publication_failure"] = failure
                # Persist the diagnosis before enqueueing, so a watchdog crash
                # retains both the publication failure and the existing backoff.
                write_json(path, state)
                if board.get("repair"):
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
            if not board.get("repair"):
                action_result = {"status": "repair_required", "incident": str(incident_path)}
            else:
                spec = dict(board["repair"])
                spec["argv"] = [*spec["argv"], "--incident", str(incident_path)]
                action_result = runner(spec, cwd=board["cwd"], timeout=120)
        else:
            action_result = runner(board["ensure"], cwd=board["cwd"], timeout=180)
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
