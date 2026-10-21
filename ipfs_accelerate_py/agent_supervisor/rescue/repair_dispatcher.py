"""Persistent, evidence-driven dispatch of a bounded supervisor repair worker.

The dispatcher never changes task completion, signals a supervisor, or assumes
that a successful repair process proves recovery.  All worker commands are
explicit argv lists; the caller supplies the worker's instructions on stdin.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Callable, Iterator
import uuid

from .command_processes import CommandProcessTree
from .gate_recovery import evidence_file_fingerprint
from .watchdog_guardian import discover_watchdogs


SCHEMA = "ipfs_accelerate_py.agent_supervisor.repair-dispatcher@1"
FLAGS = {"work_complete": False, "completion_authority": False}
_HELD_STATES = {"paused", "draining", "stopping"}
_COMPLETED = {"completed", "complete", "done"}


@dataclass
class Config:
    manifest_path: Path
    repo_root: Path
    state_dir: Path
    prompt_path: Path | None = None
    worker_argv: list[str] = field(default_factory=list)
    owner_pid_path: Path | None = None
    repair_lock_path: Path | None = None
    hold_paths: tuple[Path, ...] = ()
    evidence_paths: tuple[Path, ...] = ()
    interval: float = 30.0
    heartbeat_seconds: float = 120.0
    report_seconds: float = 600.0
    quiet_seconds: float = 900.0
    startup_grace_seconds: float = 180.0
    worker_timeout: float = 1800.0
    cooldown_seconds: float = 300.0
    max_cooldown_seconds: float = 3600.0
    max_attempts: int = 3


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _age(path: Path, now: float) -> float | None:
    try:
        return max(0.0, now - path.stat().st_mtime)
    except OSError:
        return None


def _pid(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return 0


def probe_process(pid: int) -> dict[str, Any]:
    """Read /proc without signalling the process; zombies are not live owners."""
    result: dict[str, Any] = {"pid": pid, "alive": False}
    if pid <= 0:
        return result
    try:
        proc = Path("/proc") / str(pid)
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        result["process_state"] = stat[0]
        result["start_ticks"] = int(stat[19])
        result["pgid"] = int(stat[2])
        result["session_id"] = int(stat[3])
        result["alive"] = stat[0] not in {"Z", "X"}
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        result["age_seconds"] = max(0.0, uptime - int(stat[19]) / os.sysconf("SC_CLK_TCK"))
        result["argv"] = [x.decode(errors="replace") for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
        result["cwd"] = str((proc / "cwd").resolve())
    except (OSError, ValueError, IndexError):
        result["alive"] = False
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _tasks(path: Path, prefix: str) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            current = None
            if line.startswith(prefix):
                task_id = line[3:].split()[0]
                current = {"status": "unknown", "depends": []}
                tasks[task_id] = current
        elif current is not None:
            match = re.match(r"- (Status|Depends on|Deterministic command|Validation):\s*(.*)", line)
            if match:
                key, value = match.groups()
                if key == "Depends on":
                    current["depends"] = re.findall(r"[A-Za-z][A-Za-z0-9]*-[A-Za-z0-9-]+", value)
                else:
                    current[key.lower()] = value.strip().lower() if key == "Status" else value.strip()
    return tasks


def _configured_holds(config: Config, manifest: dict[str, Any]) -> list[Path]:
    paths = list(config.hold_paths)
    paths.extend((config.state_dir / "hold", config.manifest_path.parent / "repair-dispatcher.hold",
                  config.manifest_path.parent / "operator.hold"))
    paths.extend(directory / name for directory in (config.repo_root, config.manifest_path.parent)
                 for name in ("HOLD", "OPERATOR_STOP", "watchdog.disabled"))
    for item in [manifest, *manifest.get("lanes", [])]:
        values = [item.get("hold_path"), *item.get("hold_paths", [])]
        paths.extend(_path(config.repo_root, p) for p in values if p)
    return paths


def _verify_owner(owner: dict[str, Any], config: Config) -> dict[str, Any]:
    if not owner["alive"]:
        return owner
    argv = owner.get("argv", [])

    def option(name: str) -> str:
        for i, value in enumerate(argv):
            if value == name and i + 1 < len(argv):
                return argv[i + 1]
            if value.startswith(name + "="):
                return value.split("=", 1)[1]
        return ""

    module = "ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog"
    cwd = Path(owner.get("cwd") or config.repo_root)
    manifest = option("--manifest-path")
    root = option("--repo-root")
    verified = (option("-m") == module and bool(manifest)
                and _path(cwd, manifest).resolve() == config.manifest_path.resolve()
                and (_path(cwd, root).resolve() if root else cwd.resolve()) == config.repo_root.resolve())
    return {**owner, "alive": verified, "identity_verified": verified,
            "identity_reason": "module_manifest_repo_match" if verified else "owner_identity_mismatch"}


def _public_process(process: dict[str, Any]) -> dict[str, Any]:
    # An unrelated reused PID may carry unrelated private argv; do not persist it.
    return {k: v for k, v in process.items() if k not in {"argv", "command", "cwd"}}


def _find_owner(
    config: Config, owner_path: Path, report: dict[str, Any],
    probe: Callable[[int], dict[str, Any]], now: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Reconcile live identities before trusting an advisory PID file."""
    candidates = {_pid(owner_path)}
    try:
        candidates.add(int(report.get("status", {}).get("pid") or 0))
    except (ValueError, TypeError):
        pass
    guardian_path = config.manifest_path.parent / "watchdog-guardian-status.json"
    guardian_age = _age(guardian_path, now)
    if guardian_age is not None and guardian_age <= config.report_seconds:
        try:
            guardian = _json(guardian_path)
            if (Path(guardian.get("manifest_path", "")).resolve() == config.manifest_path.resolve()
                    and Path(guardian.get("repo_root", "")).resolve() == config.repo_root.resolve()):
                candidates.add(int(guardian.get("owner_pid") or 0))
                candidates.update(int(item.get("pid") or 0) for item in guardian.get("owners", []))
        except (OSError, ValueError, TypeError):
            pass
    verified = {pid: _verify_owner(probe(pid), config) for pid in candidates if pid > 0}
    verified = {pid: process for pid, process in verified.items() if process["alive"]}
    errors: list[str] = []
    if not verified:
        discovered, errors = discover_watchdogs(config.manifest_path.resolve(), config.repo_root.resolve())
        for candidate in discovered:
            pid = int(candidate["pid"])
            process = _verify_owner(probe(pid), config)
            if process["alive"]:
                verified[pid] = process
    owners = [verified[pid] for pid in sorted(verified)]
    owner = owners[0] if len(owners) == 1 else _verify_owner(probe(_pid(owner_path)), config)
    if len(owners) > 1:
        errors.append("multiple_verified_watchdog_owners")
    return owner, [_public_process(process) for process in owners], errors


def _exhausted_wait_proofs(
    config: Config, lane: dict[str, Any], ready: list[str],
) -> list[dict[str, Any]]:
    """Read current invocation receipts; a cooldown label alone is not failure.

    A producer can succeed while its behavioral validator fails. Conversely,
    a failing producer needs its own nonzero exit receipt. Both cases require
    three spent attempts, a declared deadline, and unchanged source contracts.
    """
    if lane["phase"] != "attempt_retry_wait":
        return []
    try:
        ledger_path = Path(lane["status_path"]).with_name("lane_resume_attempts.json")
        ledger = _json(ledger_path)
        from .lane_resume import _command_contract, parse_deterministic_command
        from .supervisor_watchdog import parse_todo_task_fields
        manifest = _json(config.manifest_path)
        board = _path(config.repo_root, manifest["todo_path"])
        fields_by_id = dict(parse_todo_task_fields(
            board.read_text(), task_header_prefix=manifest.get("task_header_prefix", "## TASK-"),
        ))
    except (OSError, ValueError, KeyError, TypeError):
        return []
    proofs = []
    for reason in lane["backpressure_reasons"]:
        if not isinstance(reason, str):
            continue
        task_id, separator, suffix = reason.partition(":")
        if not separator or suffix != "attempts_exhausted" or task_id not in ready:
            continue
        count = ledger.get(task_id)
        retry = ledger.get("_retry_after_ms", {})
        receipt = ledger.get("_command_evidence", {})
        fixed = ledger.get("_validation_fixed_points", {})
        if not all(isinstance(value, dict) for value in (retry, receipt, fixed)):
            continue
        retry = retry.get(task_id)
        receipt = receipt.get(task_id)
        fixed = fixed.get(task_id, {})
        if (not isinstance(count, int) or isinstance(count, bool) or count < 3
                or not isinstance(retry, (int, float)) or isinstance(retry, bool)
                or not math.isfinite(retry) or retry <= 0 or not isinstance(receipt, dict)):
            continue
        before, after, code = receipt.get("before"), receipt.get("after"), receipt.get("exit_code")
        contract = receipt.get("contract")
        if (not isinstance(before, dict) or not isinstance(after, dict) or not after
                or not isinstance(code, int) or isinstance(code, bool) or code == 75
                or not isinstance(contract, str) or not re.fullmatch(r"[0-9a-f]{64}", contract)):
            continue
        failures = fixed.get("failures") if isinstance(fixed, dict) else None
        if code == 0 and (not isinstance(failures, int) or isinstance(failures, bool) or failures < 3
                          or fixed.get("admitted")
                          or not re.fullmatch(r"[0-9a-f]{64}", str(fixed.get("digest") or ""))):
            continue
        fields = fields_by_id.get(task_id) or {}
        command = parse_deterministic_command(
            fields.get("deterministic_command") or fields.get("deterministic_repair") or "",
            repo_root=config.repo_root,
        )
        validation = parse_deterministic_command(fields.get("validation") or "", repo_root=config.repo_root)
        if not command or (fields.get("validation") and not validation):
            continue
        if contract != _command_contract(fields, command, validation, repo_root=config.repo_root):
            continue
        current = {}
        valid = True
        for relative, digest in after.items():
            if not isinstance(relative, str) or not (digest is None or isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)):
                valid = False
                break
            try:
                filename = (config.repo_root / relative).resolve()
                filename.relative_to(config.repo_root.resolve())
                body = filename.read_bytes() if filename.is_file() else None
            except (OSError, ValueError):
                valid = False
                break
            # JSON clocks can change independently of source. Non-JSON bytes
            # must still match the invocation that spent this attempt budget.
            if filename.suffix.casefold() != ".json" and (hashlib.sha256(body).hexdigest() if body is not None else None) != digest:
                valid = False
                break
            current[relative] = evidence_file_fingerprint(filename)
        if valid:
            proofs.append({"task_id": task_id, "attempts": count,
                           "failure_stage": "validation" if code == 0 else "command",
                           "command_exit_code": code, "retry_deadline": float(retry) / 1000,
                           "attempt_path": str(ledger_path), "contract": contract,
                           "semantic_evidence": _digest(current)})
    return proofs


def observe(
    config: Config,
    state: dict[str, Any],
    *,
    now: float | None = None,
    probe: Callable[[int], dict[str, Any]] = probe_process,
) -> dict[str, Any]:
    """Observe current evidence and update only persisted gate-observation clocks."""
    now = time.time() if now is None else now
    manifest = _json(config.manifest_path)
    lanes = manifest.get("lanes")
    if not isinstance(lanes, list) or not lanes or not all(isinstance(x, dict) for x in lanes):
        raise ValueError("manifest must contain non-empty lanes")
    report_path = _path(config.repo_root, manifest.get("report_path") or config.manifest_path.parent / "watchdog-report.json")
    report_age = _age(report_path, now)
    try:
        report = _json(report_path)
        report_error = ""
    except (OSError, ValueError) as exc:
        report, report_error = {}, str(exc)
    report_lanes = {r.get("bundle_key"): r for r in report.get("reports", []) if isinstance(r, dict)}
    owner_path = config.owner_pid_path or config.manifest_path.parent / "watchdog.pid"
    owner, owner_candidates, owner_errors = _find_owner(config, owner_path, report, probe, now)
    owner_uncertain = bool(owner_errors)
    holds = [str(p) for p in _configured_holds(config, manifest) if p.exists()]
    tasks: dict[str, dict[str, Any]] = {}
    if manifest.get("todo_path"):
        tasks = _tasks(_path(config.repo_root, manifest["todo_path"]), manifest.get("task_header_prefix", "## TASK-"))
        if not tasks:
            raise ValueError("configured taskboard has no recognized task headers")
    pending = sorted(k for k, v in tasks.items() if v["status"] in {"pending", "ready", "blocked", "failed", "in_progress", "in-progress"})
    ready = sorted(k for k in pending if all(tasks.get(dep, {}).get("status") in _COMPLETED for dep in tasks[k]["depends"]))
    board_digest = _digest(tasks)
    sources = config.evidence_paths or (Path(__file__).with_name("lane_resume.py"), Path(__file__).with_name("supervisor_watchdog.py"))
    source_digest = _digest({str(p): evidence_file_fingerprint(p) for p in sources})
    incidents: list[dict[str, Any]] = []
    snapshots = []

    def incident(kind: str, lane: str = "", task: str = "", **evidence: Any) -> None:
        incidents.append({"kind": kind, "lane": lane, "task": task, "evidence": evidence})

    for lane in lanes:
        key = str(lane["bundle_key"])
        directory = _path(config.repo_root, lane["state_dir"])
        prefix = lane["state_prefix"]
        status_path = directory / f"{prefix}_status.json"
        try:
            status = _json(status_path)
            status_error = ""
        except (OSError, ValueError) as exc:
            status, status_error = {}, str(exc)
        pid_path = directory / f"{prefix}_bundle_supervisor.pid"
        if not pid_path.exists():
            pid_path = directory / f"{prefix}_supervisor.pid"
        process = probe(_pid(pid_path) or int(status.get("pid") or 0))
        heartbeat = status.get("heartbeat_at_ms") or status.get("updated_at_ms")
        heartbeat_age = max(0.0, now - float(heartbeat) / 1000) if heartbeat else None
        transition = status.get("last_task_transition_at_ms")
        transition_age = max(0.0, now - float(transition) / 1000) if transition else None
        log_path = _path(config.repo_root, lane.get("log_path") or directory / "supervisor.log")
        lane_holds = [str(p) for p in (directory / "hold", directory / "operator.hold") if p.exists()]
        intentional = status.get("state") in _HELD_STATES or status.get("terminal_reason") == "resume_stopped"
        snapshots.append({"lane": key, "status_path": str(status_path), "process": _public_process(process),
                          "state": status.get("state", ""), "phase": status.get("phase", ""),
                          "active_task_id": status.get("active_task_id", ""),
                          "terminal_reason": status.get("terminal_reason", ""),
                          "backpressure_reasons": status.get("backpressure_reasons", []),
                          "gate_evidence_fingerprint": status.get("gate_evidence_fingerprint", ""),
                          "repair_packet": status.get("repair_packet", ""),
                          "next_retry_at_ms": status.get("next_retry_at_ms"),
                          "last_task_transition_at_ms": transition,
                          "gate_recovery_needed": status.get("gate_recovery_needed") is True,
                          "heartbeat_age_seconds": heartbeat_age, "transition_age_seconds": transition_age,
                          "log_age_seconds": _age(log_path, now), "status_error": status_error,
                          "held": bool(intentional or lane_holds), "hold_paths": lane_holds})

    eligible = [lane for lane in snapshots if not lane["held"]]
    has_work = bool(pending or any(lane["active_task_id"] for lane in eligible))
    if eligible and has_work and not owner["alive"] and not owner_uncertain:
        incident("owner_dead", pending_task_ids=pending, pid=owner["pid"])
    startup = owner["alive"] and owner.get("age_seconds", config.startup_grace_seconds) < config.startup_grace_seconds
    fresh_active = any(lane["process"]["alive"] and lane["state"] in {"healthy", "starting"}
                       and lane["heartbeat_age_seconds"] is not None
                       and lane["heartbeat_age_seconds"] <= config.heartbeat_seconds for lane in eligible)
    if eligible and has_work and not startup and (report_error or report_age is None or report_age > config.report_seconds):
        incident("owner_report_stale", report_path=str(report_path), error=report_error,
                 age_seconds=report_age)
    gates = state.setdefault("gate_observations", {})
    current_gates: set[str] = set()
    exhausted_waits = state.setdefault("exhausted_wait_observations", {})
    current_exhausted: set[str] = set()
    for lane in eligible:
        key, task, phase = lane["lane"], lane["active_task_id"], lane["phase"]
        alive = lane["process"]["alive"]
        lane_ready = ready or ([task] if task and not tasks else [])
        if not alive and lane_ready:
            incident("lane_dead_pending", key, task, pending_task_ids=lane_ready,
                     terminal_reason=lane["terminal_reason"], pid=lane["process"]["pid"])
        elif alive:
            heartbeat_age = lane["heartbeat_age_seconds"]
            if heartbeat_age is not None and heartbeat_age > config.heartbeat_seconds:
                incident("lane_heartbeat_stale", key, task, age_seconds=heartbeat_age)
            if lane["state"] in {"blocked", "failed"} and (pending or task):
                incident("lane_blocked", key, task, phase=phase, reasons=lane["backpressure_reasons"])
            if lane["status_error"] and not startup and not fresh_active:
                incident("lane_status_unreadable", key, error=lane["status_error"])
            if not holds:
                for proof in _exhausted_wait_proofs(config, lane, ready):
                    wait_key = f"{key}:{proof['task_id']}"
                    signature = _digest([board_digest, source_digest, proof["contract"],
                                         proof["semantic_evidence"], proof["failure_stage"],
                                         lane["last_task_transition_at_ms"]])
                    previous = exhausted_waits.get(wait_key, {})
                    same = isinstance(previous, dict) and previous.get("signature") == signature
                    first = previous["first_seen"] if same else now
                    deadline = previous["retry_deadline"] if same else proof["retry_deadline"]
                    exhausted_waits[wait_key] = {"signature": signature, "first_seen": first,
                                                "retry_deadline": deadline, "last_seen": now}
                    current_exhausted.add(wait_key)
                    # Honor the first bounded wait. A recurring failure cannot
                    # postpone escalation by moving its retry deadline forever.
                    if now > first and (now - first >= config.quiet_seconds or now >= deadline):
                        incident("exhausted_hard_failure", key, proof["task_id"],
                                 failure_stage=proof["failure_stage"], attempts=proof["attempts"],
                                 command_exit_code=proof["command_exit_code"],
                                 first_seen=first, retry_deadline=deadline,
                                 attempt_path=proof["attempt_path"],
                                 failure_evidence_fingerprint=signature)
            if task and phase not in {"gate_withheld", "gate_retry_wait", "gate_recovery_needed", "attempt_retry_wait"}:
                log_age, transition_age = lane["log_age_seconds"], lane["transition_age_seconds"]
                if log_age is not None and transition_age is not None and min(log_age, transition_age) >= config.quiet_seconds:
                    incident("lane_no_progress", key, task, log_age_seconds=log_age,
                             transition_age_seconds=transition_age)
            if task and (phase == "gate_recovery_needed" or lane["gate_recovery_needed"]):
                incident("gate_fixed_point", key, task,
                         gate_evidence_fingerprint=lane["gate_evidence_fingerprint"],
                         repair_packet=lane["repair_packet"],
                         next_retry_at_ms=lane["next_retry_at_ms"],
                         dependencies=tasks.get(task, {}).get("depends", []))
            elif phase in {"gate_withheld", "gate_retry_wait"} and task in ready and ready == [task] and not holds:
                others = [x for x in eligible if x["lane"] != key and x["process"]["alive"]
                          and x["active_task_id"] in ready]
                if not others and all(tasks.get(d, {}).get("status") in _COMPLETED for d in tasks[task]["depends"]):
                    signature = _digest([board_digest, source_digest, key, task, phase])
                    previous = gates.get(key, {})
                    first = previous.get("first_seen", now) if previous.get("signature") == signature else now
                    gates[key] = {"signature": signature, "first_seen": first, "last_seen": now}
                    current_gates.add(key)
                    if now - first >= config.quiet_seconds:
                        incident("gate_fixed_point", key, task, first_seen=first, board_digest=board_digest,
                                 dependencies=tasks[task]["depends"], sole_selectable_task=True)
        prior = report_lanes.get(key, {})
        # An older report may describe a failure already replaced by a healthy lane.
        if not fresh_active or not (alive and lane["state"] in {"healthy", "starting"}):
            action = str(prior.get("action", ""))
            if action.startswith("repeated_") and action.endswith("_repair"):
                incident("repeated_repair", key, task, action=action, reason=prior.get("reason", ""))
    for key in list(gates):
        if key not in current_gates:
            gates.pop(key)
    for key in list(exhausted_waits):
        if key not in current_exhausted:
            exhausted_waits.pop(key)
    # Volatile ages and PIDs are evidence, not a new incident on each poll.
    identities = [{"kind": i["kind"], "lane": i["lane"], "task": i["task"],
                   "phase": i["evidence"].get("phase", ""),
                   "reason": i["evidence"].get("reason", ""),
                   "reasons": i["evidence"].get("reasons", []),
                   **({"failure_evidence_fingerprint": i["evidence"]["failure_evidence_fingerprint"]}
                      if i["kind"] == "exhausted_hard_failure" else {})} for i in incidents]
    signature = _digest([identities, board_digest, source_digest]) if incidents else ""
    return {"schema": SCHEMA, "observed_at": now, "manifest_path": str(config.manifest_path),
            "owner": _public_process(owner), "owner_pid_path": str(owner_path), "owner_startup_grace": startup,
            "owner_candidates": owner_candidates, "owner_discovery_errors": owner_errors,
            "owner_discovery_uncertain": owner_uncertain,
            "report_path": str(report_path), "report_age_seconds": report_age,
            "pending_task_ids": pending, "selectable_task_ids": ready,
            "lanes": snapshots, "incidents": incidents, "signature": signature,
            "held": bool(holds) or owner_uncertain or (bool(snapshots) and not eligible), "hold_paths": holds,
            **FLAGS}


class AlreadyRunning(RuntimeError):
    pass


@contextlib.contextmanager
def singleton_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AlreadyRunning(str(path)) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _remove_owned_marker(path: Path, inode: int, token: str) -> None:
    try:
        if path.stat().st_ino == inode and _json(path).get("token") == token:
            path.unlink()
    except (OSError, ValueError):
        pass


@contextlib.contextmanager
def repair_lock(path: Path | None) -> Iterator[None]:
    """Share a presence marker with older repair agents without taking theirs."""
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    _atomic_json(temporary, {"schema": SCHEMA, "pid": os.getpid(),
                            "start_ticks": probe_process(os.getpid()).get("start_ticks"),
                            "token": token, "created_at": time.time(), **FLAGS})
    inode = temporary.stat().st_ino
    try:
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            # Legacy markers have no trustworthy ownership and remain held.
            try:
                prior_inode = path.stat().st_ino
                prior = _json(path)
                owner = probe_process(int(prior.get("pid") or 0))
                if prior.get("schema") == SCHEMA and prior.get("token") and (
                    not owner["alive"] or owner.get("start_ticks") != prior.get("start_ticks")
                ):
                    _remove_owned_marker(path, prior_inode, prior["token"])
                    os.link(temporary, path)
                else:
                    raise AlreadyRunning(str(path)) from exc
            except (OSError, ValueError, TypeError) as error:
                raise AlreadyRunning(str(path)) from error
        temporary.unlink()
        yield
    finally:
        temporary.unlink(missing_ok=True)
        _remove_owned_marker(path, inode, token)


def _stop_owned_worker(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    except ProcessLookupError:
        process.wait()


def stop_adopted_worker(identity: dict[str, Any]) -> bool:
    """Stop only a recorded session leader after rechecking its birth identity."""
    pid = int(identity.get("pid") or 0)

    def verified() -> bool:
        current = probe_process(pid)
        return bool(current["alive"] and current.get("start_ticks") == identity.get("start_ticks")
                    and current.get("pgid") == pid and current.get("session_id") == pid
                    and identity.get("owned_session") is True)

    if not verified():
        return not probe_process(pid)["alive"]
    tree = CommandProcessTree(pid)
    try:
        tree.snapshot()
        tree.send_signal(signal.SIGTERM)
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + 5
        while verified() and time.monotonic() < deadline:
            tree.snapshot()
            time.sleep(0.05)
        if verified():
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        tree.send_signal(signal.SIGKILL)
        tree.close()
    return True


def run_worker(config: Config, log_path: Path, on_start: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    if not config.worker_argv or config.prompt_path is None:
        return {"outcome": "configuration_error", "error": "worker argv and prompt path are required"}
    process = None
    tree = None
    try:
        with config.prompt_path.open("rb") as prompt, log_path.open("ab") as output:
            environment = dict(os.environ, REPAIR_DISPATCHER_STATUS_PATH=str(config.state_dir / "status.json"))
            process = subprocess.Popen(config.worker_argv, stdin=prompt, stdout=output, stderr=subprocess.STDOUT,
                                       cwd=config.repo_root, env=environment, start_new_session=True)
            tree = CommandProcessTree(process.pid)
            tree.snapshot()
            on_start(probe_process(process.pid))
            deadline = time.monotonic() + config.worker_timeout
            while True:
                tree.snapshot()
                try:
                    code = process.wait(timeout=min(0.2, max(0.001, deadline - time.monotonic())))
                    return {"outcome": "awaiting_observation" if code == 0 else "failed", "exit_code": code}
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        _stop_owned_worker(process)
                        return {"outcome": "timed_out", "exit_code": process.returncode}
    except (OSError, ValueError) as exc:
        return {"outcome": "failed", "error": str(exc)}
    finally:
        if process is not None and process.poll() is None:
            _stop_owned_worker(process)
        if tree is not None:
            tree.snapshot()
            tree.send_signal(signal.SIGTERM)
            time.sleep(0.1)
            tree.send_signal(signal.SIGKILL)
            tree.close()


def _save(config: Config, state: dict[str, Any], telemetry: dict[str, Any]) -> None:
    _atomic_json(config.state_dir / "state.json", state)
    _atomic_json(config.state_dir / "status.json", telemetry)
    with (config.state_dir / "events.jsonl").open("a") as events:
        events.write(json.dumps(telemetry, sort_keys=True) + "\n")
        events.flush()


def tick(
    config: Config, state: dict[str, Any], *, apply: bool = False, now: float | None = None,
    probe: Callable[[int], dict[str, Any]] = probe_process,
    worker: Callable[..., dict[str, Any]] = run_worker,
) -> dict[str, Any]:
    use_wall_clock = now is None
    now = time.time() if now is None else now
    state.update({"schema": SCHEMA, **FLAGS})
    try:
        observation = observe(config, state, now=now, probe=probe)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        telemetry = {"schema": SCHEMA, "observed_at": now, "action": "observation_error", "error": str(exc), **FLAGS}
        _save(config, state, telemetry)
        return telemetry
    telemetry = {**observation, "action": "observational", "apply": apply}
    active = state.get("active_worker", {})
    if active:
        live = probe(int(active.get("pid") or 0))
        if live["alive"] and live.get("start_ticks") == active.get("start_ticks"):
            deadline = active.get("deadline")
            if deadline is None:
                telemetry["action"] = "attention_required"
                telemetry["reason"] = "adopted_worker_missing_deadline"
            elif now >= float(deadline):
                if observation["held"]:
                    telemetry["action"] = "held"
                elif not apply:
                    telemetry["action"] = "would_timeout_adopted_worker"
                else:
                    stopped = stop_adopted_worker(active)
                    telemetry["action"] = "adopted_worker_timed_out" if stopped else "attention_required"
                    if stopped:
                        state.pop("active_worker", None)
                        state["last_worker"] = {"outcome": "timed_out", "finished_at": now, **FLAGS}
                        state["next_dispatch_at"] = now + config.cooldown_seconds
            else:
                telemetry["action"] = "worker_already_running"
            _save(config, state, telemetry)
            return telemetry
        state.pop("active_worker", None)
        state["last_worker"] = {"outcome": "interrupted", "finished_at": now, **FLAGS}
    if state.get("spawn_intent"):
        telemetry.update({"action": "attention_required", "reason": "spawn_uncertain_after_dispatcher_interruption"})
        _save(config, state, telemetry)
        return telemetry
    signature = observation["signature"]
    if observation["held"]:
        telemetry["action"] = "owner_uncertain" if observation["owner_discovery_uncertain"] else "held"
    elif not signature:
        telemetry["action"] = "healthy"
        if state.get("unresolved_signature"):
            telemetry["action"] = "recovery_observed"
            state["recovery_observed_at"] = now
            state["attempts"] = {}
            state["next_dispatch_at"] = 0
        state["unresolved_signature"] = ""
    else:
        state["unresolved_signature"] = signature
        record = state.setdefault("attempts", {}).setdefault(signature, {"count": 0, "next_dispatch_at": 0})
        telemetry["attempt_count"] = record["count"]
        telemetry["max_attempts"] = config.max_attempts
        telemetry["next_dispatch_at"] = max(record["next_dispatch_at"], state.get("next_dispatch_at", 0))
        if record["count"] >= config.max_attempts:
            telemetry["action"] = "attention_required"
        elif not apply:
            telemetry["action"] = "would_dispatch"
        elif now < telemetry["next_dispatch_at"]:
            telemetry["action"] = "cooldown"
        else:
            try:
                with repair_lock(config.repair_lock_path):
                    record["count"] += 1
                    delay = min(config.max_cooldown_seconds, config.cooldown_seconds * 2 ** min(record["count"] - 1, 30))
                    record["next_dispatch_at"] = now + delay
                    state["next_dispatch_at"] = now + delay
                    telemetry.update({"action": "dispatching", "attempt_count": record["count"], "next_dispatch_at": now + delay})
                    state["spawn_intent"] = {"signature": signature, "started_at": now, "argv_digest": _digest(config.worker_argv)}
                    _save(config, state, telemetry)

                    def started(identity: dict[str, Any]) -> None:
                        state["active_worker"] = {**_public_process(identity), "owned_session": True,
                                                  "started_at": now, "deadline": now + config.worker_timeout}
                        state.pop("spawn_intent", None)
                        _atomic_json(config.state_dir / "state.json", state)

                    log_path = config.state_dir / f"worker-{signature[:16]}-{record['count']}.log"
                    result = worker(config, log_path, started)
                    state.pop("spawn_intent", None)
                    state.pop("active_worker", None)
                    finished = time.time() if use_wall_clock else now
                    record["next_dispatch_at"] = finished + delay
                    state["next_dispatch_at"] = finished + delay
                    state["last_worker"] = {**result, "finished_at": finished, "signature": signature, "log_path": str(log_path), **FLAGS}
                    telemetry.update({"action": "worker_finished", "worker": state["last_worker"], "next_dispatch_at": finished + delay})
                    # No recovery is inferred from exit 0.  The next tick reads live evidence again.
            except AlreadyRunning:
                telemetry.update({"action": "repair_already_running", "repair_lock_path": str(config.repair_lock_path)})
    _save(config, state, telemetry)
    return telemetry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--owner-pid-path", type=Path)
    parser.add_argument("--repair-lock-path", type=Path, help="shared presence marker excluding other repair agents")
    parser.add_argument("--prompt-path", type=Path)
    parser.add_argument("--worker-argv-json", help="JSON array of argv strings; shell interpretation is never used")
    parser.add_argument("--hold-path", type=Path, action="append", default=[])
    parser.add_argument("--evidence-path", type=Path, action="append", default=[])
    parser.add_argument("--once", action="store_true", help="observe one cycle; dispatch requires --apply")
    parser.add_argument("--apply", action="store_true", help="enable the explicitly configured repair worker")
    for option, default in (("interval", 30), ("heartbeat-seconds", 120), ("report-seconds", 600),
                            ("quiet-seconds", 900), ("startup-grace-seconds", 180), ("worker-timeout", 1800),
                            ("cooldown-seconds", 300), ("max-cooldown-seconds", 3600)):
        parser.add_argument(f"--{option}", type=float, default=default)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        worker_argv = json.loads(args.worker_argv_json) if args.worker_argv_json else []
    except ValueError:
        parser.error("--worker-argv-json must be a JSON array")
    if not isinstance(worker_argv, list) or any(not isinstance(s, str) or not s for s in worker_argv):
        parser.error("worker argv must be an array of non-empty strings")
    if args.apply and (not worker_argv or not args.prompt_path):
        parser.error("--apply requires --worker-argv-json and --prompt-path")
    for name in ("interval", "heartbeat_seconds", "report_seconds", "quiet_seconds", "startup_grace_seconds",
                 "worker_timeout", "cooldown_seconds", "max_cooldown_seconds", "max_attempts"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    root = args.repo_root.resolve()
    manifest_path = _path(root, args.manifest_path).resolve()
    config = Config(manifest_path=manifest_path, repo_root=root,
                    state_dir=_path(root, args.state_dir) if args.state_dir else manifest_path.parent / "repair-dispatcher",
                    prompt_path=_path(root, args.prompt_path) if args.prompt_path else None, worker_argv=worker_argv,
                    owner_pid_path=_path(root, args.owner_pid_path) if args.owner_pid_path else None,
                    repair_lock_path=_path(root, args.repair_lock_path) if args.repair_lock_path else None,
                    hold_paths=tuple(_path(root, p) for p in args.hold_path),
                    evidence_paths=tuple(_path(root, p) for p in args.evidence_path),
                    **{key: getattr(args, key) for key in ("interval", "heartbeat_seconds", "report_seconds", "quiet_seconds",
                       "startup_grace_seconds", "worker_timeout", "cooldown_seconds", "max_cooldown_seconds", "max_attempts")})
    try:
        def stop(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        with singleton_lock(config.state_dir / "dispatcher.lock"):
            try:
                state = _json(config.state_dir / "state.json")
            except FileNotFoundError:
                state = {}
            except (OSError, ValueError) as exc:
                print(json.dumps({"action": "state_error", "error": str(exc), **FLAGS}), flush=True)
                return 2
            while True:
                result = tick(config, state, apply=args.apply)
                print(json.dumps({k: result.get(k) for k in ("action", "signature", "observed_at", "attempt_count", "error")}), flush=True)
                if args.once:
                    return 2 if result["action"] == "observation_error" else 0
                time.sleep(config.interval)
    except AlreadyRunning as exc:
        print(json.dumps({"action": "already_running", "lock": str(exc), **FLAGS}), flush=True)
        return 3
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
