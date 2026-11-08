"""Persistent, evidence-driven dispatch of a bounded supervisor repair worker.

The dispatcher never changes task completion, signals a supervisor, or assumes
that a successful repair process proves recovery.  All worker commands are
explicit argv lists; the caller supplies the worker's instructions on stdin.

A configured Codex repair command is only the Codex candidate.  Before launch
the dispatcher asks ``llm_router.decide_router_owned_implementation_provider``
which source is usable.  Grok is preferred.  Codex is launched only after Grok
hard-quota evidence.  One source's quota does not park a source the router
still considers good.  The decision builds no model call.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import dataclass, field
from datetime import datetime
import fcntl
import hashlib
import json
import math
import marshal
import os
from pathlib import Path
import re
import select
import signal
import socket
import stat
import subprocess
import time
from typing import Any, Callable, Iterator
from types import CodeType
import uuid

from .command_processes import CommandProcessTree
from .gate_recovery import evidence_file_fingerprint
from .watchdog_guardian import (discover_watchdogs, _declared_storage_paths, _minimum_storage_bytes,
                                _storage_status, _lane_storage_identity,
                                lane_storage_emergency, watchdog_storage_emergency,
                                _option, _python_entrypoint)
from .storage_pressure import (check_storage_capacity, clear_emergency_status,
                               is_storage_pressure, publish_emergency_status)
from .planned_work_watchdog import observe_planned_work, collect_gate_blockers
from .queue_hold_lease import (QUEUE_HOLD_LEASE_VERSION, review_queue_hold, expire_queue_hold,
                               register_queue_hold, publisher_queue_hold_proof, expired_queue_hold_matches)
from .previous_worker_context import build_previous_worker_context, context_environment


SCHEMA = "ipfs_accelerate_py.agent_supervisor.repair-dispatcher@1"
SERVICE_WATCHDOG_VERSION = 1
LANE_ENVIRONMENT_GUARD_VERSION = 1
WORKER_STORAGE_ADMISSION_VERSION = 1
FLAGS = {"work_complete": False, "completion_authority": False}
_QUOTA_RETRY_RE = re.compile(
    r"try again at ([A-Z][a-z]{2,8} \d{1,2}(?:st|nd|rd|th)?, \d{4} \d{1,2}:\d{2} [AP]M)"
)
_RELATIVE_RETRY_RE = re.compile(r"try again in (\d+)\s*(second|minute|hour|day)s?", re.IGNORECASE)
_RETRY_UNITS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
_TRANSIENT_PREWORK_RE = re.compile(
    r"hit your usage limit|usage limit|quota exceeded|rate limit|too many requests|"
    r"overloaded|try again (?:at|in)|\b429\b|temporarily unavailable|server is busy",
    re.IGNORECASE,
)
_AUTH_STICK_RE = re.compile(
    r"incorrect api key|invalid api key|unauthorized|payment required|billing hard limit|"
    r"authentication failed",
    re.IGNORECASE,
)
_TRANSIENT_PREWORK_TOKENS = (
    b"usage limit", b"rate limit", b"quota exceeded", b"too many requests",
    b"overloaded", b"try again", b"429", b"temporarily unavailable", b"server is busy",
)
_QUOTA_NEUTRAL_TYPES = {"thread.started", "turn.started"}
_QUOTA_LOG_LIMIT = 65536
_QUOTA_RETRY_HORIZON_SECONDS = 14 * 24 * 60 * 60
_HELD_STATES = {"paused", "draining", "stopping"}
_COMPLETED = {"completed", "complete", "done"}
_DESCENDANT_CLEANUP_GRACE_SECONDS = 5.0


@dataclass
class Config:
    manifest_path: Path
    repo_root: Path
    state_dir: Path
    prompt_path: Path | None = None
    worker_argv: list[str] = field(default_factory=list)
    # Argv chosen for the launch in progress.  Empty means worker_argv.
    launch_argv: list[str] = field(default_factory=list)
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
    required_lane_env: tuple[str, ...] = ()
    worker_minimum_free_bytes: int = 64 * 1024 * 1024
    _storage_retry_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _storage_retry_path: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if (type(self.worker_minimum_free_bytes) is not int
                or not 1 <= self.worker_minimum_free_bytes <= 2**63 - 1):
            raise ValueError("worker_minimum_free_bytes must be a positive signed64-bit integer")


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _json(path: Path) -> dict[str, Any]:
    raw = path.read_text()
    try:
        value = json.loads(raw)
    except RecursionError as exc:
        raise ValueError('dispatcher metadata JSON nesting exceeds decoder limit') from exc
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


def _verify_lane(
    process: dict[str, Any], config: Config, manifest: dict[str, Any],
    lane: dict[str, Any], status_path: Path, pid_path: Path,
) -> dict[str, Any]:
    """A recycled advisory PID is not evidence that this lane is running."""
    if not process["alive"]:
        return process
    argv = process.get("argv", [])
    mode, module, index = _python_entrypoint(argv)
    verified = bool(argv) and bool(re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", Path(argv[0]).name))
    verified = (verified and mode == "module"
                and module == "ipfs_accelerate_py.agent_supervisor.rescue.lane_resume"
                and argv[index] == "-m"
                and all(flag in {"-u", "-B", "-E", "-s", "-S", "-I", "-O", "-OO"}
                        for flag in argv[1:index]))
    if verified:
        application_args = argv[index + 2:]
        cwd = Path(process.get("cwd") or config.repo_root)
        paths = {"--repo-root": config.repo_root, "--status-path": status_path, "--pid-path": pid_path}
        if manifest.get("todo_path"):
            paths["--todo-path"] = _path(config.repo_root, manifest["todo_path"])
        verified = _option(application_args, "--bundle-key") == lane["bundle_key"]
        for name, expected in paths.items():
            value = _option(application_args, name)
            if not value or _path(cwd, value).resolve() != expected.resolve():
                verified = False
                break
    return {**process, "alive": verified, "identity_verified": verified,
            "identity_reason": "module_lane_repo_match" if verified else "lane_identity_mismatch"}


def _public_process(process: dict[str, Any]) -> dict[str, Any]:
    # An unrelated reused PID may carry unrelated private argv; do not persist it.
    return {k: v for k, v in process.items() if k not in {"argv", "command", "cwd"}}


def _lane_environment(process: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Compare explicitly allowed policy variables without publishing their values."""
    if not keys:
        return {"enabled": False}
    if any(not isinstance(key, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*_BUDGET_POLICY", key)
           for key in keys):
        return {"enabled": True, "verified": False, "reason": "invalid_required_key", "required_keys": []}
    keys = tuple(sorted(set(keys)))
    result = {"enabled": True, "verified": False, "pid": process.get("pid"),
              "start_ticks": process.get("start_ticks"), "required_keys": list(keys)}
    expected = {key: os.environ.get(key) for key in keys}
    missing = [key for key in keys if not expected[key]]
    if missing:
        return {**result, "reason": "expected_environment_missing", "affected_keys": missing}
    if not process.get("alive") or not process.get("identity_verified"):
        return {**result, "reason": "lane_identity_unverified"}
    pid = int(process["pid"])
    expected_start = process.get("start_ticks")
    try:
        before = probe_process(pid)
        uid_before = (Path("/proc") / str(pid)).stat().st_uid
        if (not before["alive"] or before.get("start_ticks") != expected_start
                or uid_before != os.getuid() or before.get("argv") != process.get("argv")
                or before.get("cwd") != process.get("cwd")):
            return {**result, "reason": "lane_identity_changed"}
        raw = (Path("/proc") / str(pid) / "environ").read_bytes()
        after = probe_process(pid)
        uid_after = (Path("/proc") / str(pid)).stat().st_uid
    except OSError:
        # A process exit during /proc reads is ordinary identity uncertainty,
        # not proof that a live lane has an unreadable policy environment.
        try:
            current = probe_process(pid)
            current_uid = (Path("/proc") / str(pid)).stat().st_uid
        except OSError:
            return {**result, "reason": "lane_identity_changed"}
        if (not current["alive"] or current.get("start_ticks") != expected_start
                or current_uid != os.getuid() or current.get("argv") != process.get("argv")
                or current.get("cwd") != process.get("cwd")):
            return {**result, "reason": "lane_identity_changed"}
        return {**result, "reason": "environment_unreadable"}
    if (not after["alive"] or after.get("start_ticks") != expected_start
            or uid_after != uid_before or uid_after != os.getuid()
            or after.get("argv") != before.get("argv") or after.get("cwd") != before.get("cwd")):
        return {**result, "reason": "lane_identity_changed"}
    values: dict[bytes, list[bytes]] = {}
    wanted = {key.encode() for key in keys}
    for entry in raw.split(b"\0"):
        name, separator, value = entry.partition(b"=")
        if separator and name in wanted:
            values.setdefault(name, []).append(value)
    missing = [key for key in keys if key.encode() not in values]
    if missing:
        return {**result, "reason": "required_environment_missing", "affected_keys": missing}
    mismatched = [key for key in keys if values[key.encode()] != [expected[key].encode()]]
    if mismatched:
        return {**result, "reason": "required_environment_mismatch", "affected_keys": mismatched}
    return {**result, "verified": True, "reason": "required_environment_matches"}


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


def _native_rotation_grace(config: Config, state: dict[str, Any], lane: dict[str, Any],
                           owner: dict[str, Any], report: dict[str, Any], *, now: float,
                           report_age: float | None, owner_uncertain: bool,
                           storage_pressure: bool) -> dict[str, Any] | None:
    """Give a verified native watchdog one bounded ordinary lease rotation."""
    pid = lane["process"].get("pid")
    stamp = lane.get("terminal_heartbeat_at_ms")
    if (lane["state"] != "stopped" or lane["phase"] != "lane_resume_exited"
            or lane["terminal_reason"] != "resume_duration_elapsed"
            or type(pid) is not int or pid <= 0 or lane.get("status_pid") != pid
            or lane["process"].get("process_state") not in (None, "Z", "X")
            or type(stamp) not in (int, float) or not math.isfinite(stamp)
            or not 0 < stamp / 1000 <= now):
        return None
    interval = _option(owner.get("argv", []), "--check-interval")
    if interval is None and owner.get("alive"):
        try:
            directory = Path("/proc") / str(owner["pid"])
            if directory.stat().st_uid != os.getuid():
                return None
            # Read only the native watchdog's nonsecret poll setting. Never
            # persist its environment or use the dispatcher's own setting.
            environment = (directory / "environ").read_bytes()
            interval = next((entry.split(b"=", 1)[1].decode() for entry in environment.split(b"\0")
                             if entry.startswith(b"WATCHDOG_CHECK_INTERVAL_SECONDS=")), None)
        except (OSError, UnicodeError):
            return None
    try:
        poll = float(interval) if interval is not None else 120.0
        if not math.isfinite(poll) or poll <= 0:
            return None
    except (TypeError, ValueError):
        return None
    grace = min(600.0, max(config.startup_grace_seconds, poll + config.interval + 15.0))
    records = state.setdefault("lease_rotation_observations", {})
    if not isinstance(records, dict):
        records = state["lease_rotation_observations"] = {}
    previous = records.get(lane["lane"], {})
    if isinstance(previous, dict) and previous.get("expired_pid") == pid:
        anchor, deadline = previous.get("anchor"), previous.get("deadline")
        if (type(anchor) not in (int, float) or type(deadline) not in (int, float)
                or not math.isfinite(anchor) or not math.isfinite(deadline)
                or not 0 < anchor <= deadline <= anchor + 600):
            return None
    else:
        anchor, deadline = stamp / 1000, stamp / 1000 + grace
    observation = {"lane": lane["lane"], "expired_pid": pid, "anchor": anchor,
                   "deadline": deadline, "waiting": False,
                   "watchdog_pid": owner.get("pid"), "watchdog_start_ticks": owner.get("start_ticks")}
    records[lane["lane"]] = observation
    status = report.get("status") if isinstance(report.get("status"), dict) else {}
    if (now >= deadline or owner_uncertain or storage_pressure or not owner.get("alive")
            or not owner.get("identity_verified") or report_age is None
            or report_age > config.report_seconds or status.get("pid") != owner.get("pid")
            or status.get("state") not in {"healthy", "starting", "degraded"}):
        return observation
    for prior in report.get("reports", []):
        if not isinstance(prior, dict) or prior.get("bundle_key") != lane["lane"]:
            continue
        if (prior.get("action") in {"restart_failed", "fenced_stop_required"}
                or str(prior.get("action", "")).startswith("repeated_")):
            return observation
    # Independently reread the real role/UID/start tick; an advisory PID or a
    # report belonging to an earlier process cannot earn this grace.
    actual, errors = discover_watchdogs(config.manifest_path.resolve(), config.repo_root.resolve())
    if (errors or len(actual) != 1 or actual[0]["pid"] != owner["pid"]
            or str(actual[0]["start_ticks"]) != str(owner.get("start_ticks"))
            or report_age > float(owner.get("age_seconds", 0)) + 1):
        return observation
    observation["waiting"] = True
    return observation


def _runtime_policy_provenance() -> dict[str, Any]:
    """Compare loaded contract functions with current disk code, without exec."""
    from . import lane_resume
    def normalized(code: CodeType) -> CodeType:
        return code.replace(co_consts=tuple(normalized(c) if isinstance(c, CodeType) else c
                                          for c in code.co_consts),
                            co_filename="", co_firstlineno=0, co_linetable=b"")
    path = Path(lane_resume.__file__).resolve()
    try:
        raw = path.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("runtime policy source exceeds bound")
        compiled = {code.co_name: code for code in compile(raw, str(path), "exec", dont_inherit=True).co_consts
                    if isinstance(code, CodeType)}
        names = ("_command_contract", "_runtime_contract_code", "_run_argv", "_declared_evidence",
                 "_execution_input_contract", "parse_deterministic_command")
        changed = [name for name in names if name in compiled and
                   (not hasattr(getattr(lane_resume, name, None), "__code__") or
                    marshal.dumps(normalized(getattr(lane_resume, name).__code__)) !=
                    marshal.dumps(normalized(compiled[name])))]
        return {"policy_source": str(path), "disk_source_sha256": hashlib.sha256(raw).hexdigest(),
                "loaded_policy_matches_disk": not changed, "mismatched_functions": changed}
    except (OSError, ValueError, SyntaxError) as exc:
        return {"policy_source": str(path), "loaded_policy_matches_disk": False,
                "provenance_error": str(exc)}


def _exhausted_wait_proofs(
    config: Config, lane: dict[str, Any], ready: list[str], *, reviews: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Read current invocation receipts; a cooldown label alone is not failure.

    A producer can succeed while its behavioral validator fails. Conversely,
    a failing producer needs its own nonzero exit receipt. Both cases require
    three spent attempts, a declared deadline, and unchanged source contracts.
    """
    if lane["phase"] not in {"attempt_retry_wait", "semantic_failure_wait"}:
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
        semantic = suffix == "unchanged_semantic_failure_budget_exhausted"
        if not separator or suffix not in {"attempts_exhausted", "unchanged_semantic_failure_budget_exhausted"} or task_id not in ready:
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
        progress = fixed.get("execution_progress", {}) if isinstance(fixed, dict) else {}
        if semantic:
            if (not isinstance(progress, dict) or progress.get("schema") != "agent-supervisor/execution-progress@1"
                    or type(progress.get("semantic_failures")) is not int or progress["semantic_failures"] < 6
                    or progress.get("state") == "validated"):
                continue
            if retry is None:
                retry = progress.get("finished_at_ms")
        if (not isinstance(count, int) or isinstance(count, bool) or count < 3
                or not isinstance(retry, (int, float)) or isinstance(retry, bool)
                or not math.isfinite(retry) or retry <= 0 or not isinstance(receipt, dict)):
            continue
        before, after, code = receipt.get("before"), receipt.get("after"), receipt.get("exit_code")
        contract = receipt.get("contract")
        # Producers and validators may legitimately rewrite declared source.
        # Their invocation-start contract remains proof provenance; the durable
        # post-invocation contract identifies which code spent this retry budget.
        # A present malformed retry contract must never downgrade to old proof.
        retry_contract = receipt.get("retry_contract") if "retry_contract" in receipt else contract
        if (not isinstance(before, dict) or not isinstance(after, dict) or not after
                or not isinstance(code, int) or isinstance(code, bool) or code == 75
                or not isinstance(contract, str) or not re.fullmatch(r"[0-9a-f]{64}", contract)
                or not isinstance(retry_contract, str) or not re.fullmatch(r"[0-9a-f]{64}", retry_contract)):
            continue
        failures = fixed.get("failures") if isinstance(fixed, dict) else None
        semantic_validation_failure = (semantic and progress.get("state") == "failed"
            and progress.get("last_stage") == "validation" and type(progress.get("last_exit_code")) is int
            and progress["last_exit_code"] not in (0, 75))
        if code == 0 and not semantic_validation_failure and (not isinstance(failures, int) or isinstance(failures, bool) or failures < 3
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
        current_contract = _command_contract(fields, command, validation, repo_root=config.repo_root)
        if retry_contract != current_contract:
            if reviews is not None:
                reviews.append({"task_id": task_id, "reason": "recorded_retry_contract_not_current",
                    "recorded_retry_contract": retry_contract, "loaded_current_contract": current_contract,
                    "current_proof_accepted": False, **_runtime_policy_provenance()})
            continue
        if semantic and progress.get("contract") != current_contract:
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
                           "retry_contract": retry_contract,
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
    owner_emergency = watchdog_storage_emergency(report_path, owner, config.manifest_path.resolve(),
                                                  config.repo_root.resolve()) if owner["alive"] else None
    if owner_emergency:
        report = owner_emergency["payload"]
        report_error = ""
        report_age = max(0, now - owner_emergency["observed_at_ms"] / 1000)
        report_lanes = {r.get("bundle_key"): r for r in report.get("reports", []) if isinstance(r, dict)}
    report_status = report.get("status") if isinstance(report.get("status"), dict) else {}
    owner_pressure = bool(owner_emergency) or bool(owner["alive"] and report_status.get("pid") == owner["pid"]
        and report_age is not None and report_age <= config.report_seconds
        and (_storage_status(report) or _storage_status(report_status)
             or any(item.get("action") == "storage_wait" for item in report_lanes.values())))
    declared_storage = _declared_storage_paths(manifest, config.repo_root)
    minimum_free_bytes = _minimum_storage_bytes(manifest)
    owner_uncertain = bool(owner_errors)
    queue_lease = review_queue_hold(config, now)
    holds = [str(p) for p in _configured_holds(config, manifest) if p.exists()
             and not expired_queue_hold_matches(config, p, now, queue_lease)]
    tasks: dict[str, dict[str, Any]] = {}
    if manifest.get("todo_path"):
        tasks = _tasks(_path(config.repo_root, manifest["todo_path"]), manifest.get("task_header_prefix", "## TASK-"))
        if not tasks:
            raise ValueError("configured taskboard has no recognized task headers")
    pending = sorted(k for k, v in tasks.items() if v["status"] in {"todo", "pending", "ready", "blocked", "failed", "in_progress", "in-progress"})
    ready = sorted(k for k in pending if all(tasks.get(dep, {}).get("status") in _COMPLETED for dep in tasks[k]["depends"]))
    board_digest = _digest(tasks)
    sources = config.evidence_paths or (Path(__file__).with_name("lane_resume.py"), Path(__file__).with_name("supervisor_watchdog.py"))
    source_digest = _digest({str(p): evidence_file_fingerprint(p) for p in sources})
    incidents: list[dict[str, Any]] = []
    snapshots = []
    environment_reviews: list[dict[str, Any]] = []

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
        legacy_pid_path = directory / f"{prefix}_supervisor.pid"
        if not pid_path.exists() and legacy_pid_path.exists():
            pid_path = legacy_pid_path
        process = _verify_lane(probe(_pid(pid_path) or int(status.get("pid") or 0)),
                               config, manifest, lane, status_path, pid_path)
        environment = _lane_environment(process, config.required_lane_env)
        if environment.get("enabled") and not environment.get("verified"):
            review = {"lane": key, **environment}
            environment_reviews.append(review)
            # An absent/unverified actor follows the existing bounded native
            # recovery path. Environment uncertainty cannot bypass its normal
            # startup/lease-rotation grace or authorize an extra repair worker.
            if environment["reason"] not in {"lane_identity_unverified", "lane_identity_changed"}:
                incident("lane_environment_review_needed", key,
                         **{name: value for name, value in environment.items() if name not in {"enabled", "verified"}})
        emergency = lane_storage_emergency(status_path, process["pid"], config.repo_root.resolve())
        if emergency:
            status = {**status, **emergency["payload"], "pid": process["pid"]}
            status_error = ""
        heartbeat = status.get("heartbeat_at_ms") or status.get("updated_at_ms")
        heartbeat_age = max(0.0, now - float(heartbeat) / 1000) if heartbeat else None
        storage_pressure = bool(emergency) or bool(_storage_status(status) and process["alive"]
            and status.get("pid") == process["pid"] and heartbeat_age is not None
            and heartbeat_age <= config.heartbeat_seconds
            and _lane_storage_identity(process["pid"], config.repo_root.resolve(), status_path))
        transition = status.get("last_task_transition_at_ms")
        transition_age = max(0.0, now - float(transition) / 1000) if transition else None
        meaningful = status.get("meaningful_progress_at_ms", status.get("last_verified_progress_at_ms"))
        meaningful_age = max(0.0, now - float(meaningful) / 1000) if meaningful else None
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
                          "status_pid": status.get("pid"), "terminal_heartbeat_at_ms": heartbeat,
                          "meaningful_progress_age_seconds": meaningful_age,
                          "storage_pressure": storage_pressure, "emergency_identity_verified": bool(emergency),
                          "log_age_seconds": _age(log_path, now), "status_error": status_error,
                          "held": bool(intentional or lane_holds), "hold_paths": lane_holds})
        if environment.get("enabled"):
            snapshots[-1]["required_environment"] = environment

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
    retry_waits: list[dict[str, Any]] = []
    provenance_reviews: list[dict[str, Any]] = []
    rotation_waits: list[dict[str, Any]] = []
    current_rotations: set[str] = set()
    for lane in eligible:
        key, task, phase = lane["lane"], lane["active_task_id"], lane["phase"]
        alive = lane["process"]["alive"]
        lane_ready = ready or ([task] if task and not tasks else [])
        if not alive and lane_ready:
            rotation = _native_rotation_grace(config, state, lane, owner, report, now=now,
                report_age=report_age, owner_uncertain=owner_uncertain,
                storage_pressure=owner_pressure or lane["storage_pressure"])
            if rotation is not None:
                current_rotations.add(key)
            if rotation and rotation["waiting"]:
                rotation_waits.append(rotation)
            else:
                incident("lane_dead_pending", key, task, pending_task_ids=lane_ready,
                         terminal_reason=lane["terminal_reason"], pid=lane["process"]["pid"])
        elif alive:
            if lane["storage_pressure"]:
                continue
            heartbeat_age = lane["heartbeat_age_seconds"]
            if (pending or task) and heartbeat_age is not None and heartbeat_age > config.heartbeat_seconds:
                incident("lane_heartbeat_stale", key, task, age_seconds=heartbeat_age)
            if lane["state"] in {"blocked", "failed"} and (pending or task):
                incident("lane_blocked", key, task, phase=phase, reasons=lane["backpressure_reasons"])
            if lane["status_error"] and not startup and not fresh_active:
                incident("lane_status_unreadable", key, error=lane["status_error"])
            if not holds:
                reviews: list[dict[str, Any]] = []
                for proof in _exhausted_wait_proofs(config, lane, ready, reviews=reviews):
                    wait_key = f"{key}:{proof['task_id']}"
                    signature = _digest([proof["retry_contract"],
                                         proof["semantic_evidence"], proof["failure_stage"],
                                         lane["last_task_transition_at_ms"]])
                    previous = exhausted_waits.get(wait_key, {})
                    same = isinstance(previous, dict) and previous.get("signature") == signature
                    first = previous["first_seen"] if same else now
                    deadline = previous["retry_deadline"] if same else proof["retry_deadline"]
                    exhausted_waits[wait_key] = {"signature": signature, "first_seen": first,
                                                "retry_deadline": deadline, "last_seen": now}
                    current_exhausted.add(wait_key)
                    retry_waits.append({"lane": key, "task_id": proof["task_id"],
                                        "first_seen": first, "retry_deadline": deadline,
                                        "failure_stage": proof["failure_stage"]})
                    # Honor the first bounded wait. A recurring failure cannot
                    # postpone escalation by moving its retry deadline forever.
                    if now > first and (now - first >= config.quiet_seconds or now >= deadline):
                        incident("exhausted_hard_failure", key, proof["task_id"],
                                 failure_stage=proof["failure_stage"], attempts=proof["attempts"],
                                 command_exit_code=proof["command_exit_code"],
                                 first_seen=first, retry_deadline=deadline,
                                 attempt_path=proof["attempt_path"],
                                 failure_evidence_fingerprint=signature)
                provenance_reviews.extend({"lane": key, **review} for review in reviews)
                for review in reviews:
                    if review.get("loaded_policy_matches_disk") is False:
                        incident("source_change_review_needed", key, review["task_id"], **{
                            k: v for k, v in review.items() if k != "task_id"})
            if task and phase not in {"gate_withheld", "gate_retry_wait", "gate_recovery_needed", "attempt_retry_wait", "semantic_failure_wait"}:
                log_age, transition_age = lane["log_age_seconds"], lane["transition_age_seconds"]
                meaningful_age = lane["meaningful_progress_age_seconds"]
                stalled = (meaningful_age >= config.quiet_seconds if meaningful_age is not None else
                           log_age is not None and transition_age is not None and min(log_age, transition_age) >= config.quiet_seconds)
                if stalled:
                    incident("lane_no_progress", key, task, log_age_seconds=log_age,
                             transition_age_seconds=transition_age, meaningful_progress_age_seconds=meaningful_age,
                             evidence_basis="verified_progress_clock" if meaningful_age is not None else "legacy_log_and_transition")
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
            lane_key, _, task_id = key.rpartition(":")
            running_retry = any(lane["lane"] == lane_key and not lane["held"] and
                                lane["phase"] in {"deterministic_command", "validation", "lane_resume"}
                                for lane in eligible)
            if holds or task_id not in ready or not running_retry:
                exhausted_waits.pop(key)
    if not isinstance(state.get("lease_rotation_observations", {}), dict):
        state["lease_rotation_observations"] = {}
    for key in list(state.get("lease_rotation_observations", {})):
        if key not in current_rotations:
            state["lease_rotation_observations"].pop(key)
    policy = manifest.get("planned_work_policy")
    frontier_gates = (collect_gate_blockers(repo_root=config.repo_root, manifest=manifest,
                      lanes=eligible, ready=ready, now=now)
                      if isinstance(policy, dict) and policy.get("enabled") is True and not holds
                      else {"proofs": [], "reviews": []})
    if frontier_gates["reviews"]:
        provenance = _runtime_policy_provenance()
        if provenance.get("loaded_policy_matches_disk") is False:
            for review in frontier_gates["reviews"]:
                incident("source_change_review_needed", review["lane"], review["task_id"],
                         **{key: value for key, value in review.items() if key not in {"lane", "task_id"}},
                         **provenance)
    planned_work = observe_planned_work(policy=policy,
        repo_root=config.repo_root, tasks=tasks, lanes=snapshots, state=state, now=now,
        held=bool(holds) or owner_uncertain or bool(snapshots) and not eligible
             or owner_pressure or any(lane["storage_pressure"] for lane in snapshots),
        gate_blockers=frontier_gates["proofs"])
    for entry in planned_work["incidents"]:
        incident(entry["kind"], **entry["evidence"])
    if planned_work.get("enabled"):
        source_digest = _digest([source_digest, planned_work["contract_fingerprint"]])
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
            "storage_pressure": owner_pressure or any(lane["storage_pressure"] for lane in snapshots),
            "storage_paths": [str(path) for path in declared_storage], "storage_minimum_free_bytes": minimum_free_bytes,
            "report_path": str(report_path), "report_age_seconds": report_age,
            "pending_task_ids": pending, "selectable_task_ids": ready,
            "retry_waits": retry_waits, "source_provenance_reviews": provenance_reviews,
            "lane_environment_reviews": environment_reviews,
            "native_rotation_grace": rotation_waits,
            "retry_failure_observation_pending": bool(exhausted_waits),
            "native_retry_wait_observed": bool(ready) and any(
                lane["phase"] in {"attempt_retry_wait", "semantic_failure_wait"} for lane in eligible),
            "planned_work": planned_work,
            "frontier_gate_reviews": frontier_gates["reviews"],
            "lanes": snapshots, "incidents": incidents, "signature": signature,
            "held": bool(holds) or owner_uncertain or (bool(snapshots) and not eligible), "hold_paths": holds,
            **({"queue_hold_lease": queue_lease} if queue_lease["status"] != "absent" else {}),
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
        try:
            # The descriptor/flock owns the dispatcher, not its optional PID
            # text. Raw writes avoid retrying a failed buffered flush at close.
            os.ftruncate(handle.fileno(), 0)
            os.write(handle.fileno(), (str(os.getpid()) + "\n").encode())
        except OSError as exc:
            if not is_storage_pressure(exc):
                raise
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


def _repair_marker_matches(lease: dict[str, Any]) -> bool:
    try:
        path = Path(lease["path"])
        info = path.lstat()
        value = _json(path)
        return (stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1
                and (info.st_dev, info.st_ino) == (lease["device"], lease["inode"])
                and value.get("schema") == SCHEMA and value.get("token") == lease["token"]
                and value.get("pid") == lease["pid"]
                and value.get("start_ticks") == lease["start_ticks"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _release_repair_marker(lease: dict[str, Any] | None) -> bool:
    if lease is None or not _repair_marker_matches(lease):
        return False
    try:
        path = Path(lease["path"])
        current = path.lstat()
        if (current.st_dev, current.st_ino) != (lease["device"], lease["inode"]):
            return False
        path.unlink()
        return True
    except OSError:
        return False


def _transfer_repair_marker(lease: dict[str, Any] | None,
                            identity: dict[str, Any]) -> dict[str, Any] | None:
    if lease is None or not identity.get("alive"):
        return None
    pid = identity.get("pid")
    actual = probe_process(int(pid or 0))
    try:
        directory = Path("/proc") / str(pid)
        parent = int((directory / "stat").read_text().rsplit(")", 1)[1].split()[1])
        owned = (directory.stat().st_uid == os.getuid() and parent == os.getpid()
                 and actual.get("alive") and actual.get("start_ticks") == identity.get("start_ticks")
                 and actual.get("pgid") == pid and actual.get("session_id") == pid)
    except (OSError, ValueError, IndexError):
        owned = False
    if not owned or not _repair_marker_matches(lease):
        raise AlreadyRunning("repair presence ownership changed before worker handoff")
    path = Path(lease["path"])
    value = _json(path)
    value.update(pid=pid, start_ticks=actual["start_ticks"], owner_kind="repair_worker")
    temporary = path.with_name(f".{path.name}.{lease['token']}.{uuid.uuid4().hex}.transfer")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            replacement = os.fstat(handle.fileno())
        if not _repair_marker_matches(lease):
            raise AlreadyRunning("repair presence changed during worker handoff")
        os.replace(temporary, path)
        lease.update(device=replacement.st_dev, inode=replacement.st_ino,
                     pid=pid, start_ticks=actual["start_ticks"])
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return dict(lease)
    finally:
        temporary.unlink(missing_ok=True)


def _saved_repair_marker(config: Config, active: dict[str, Any]) -> dict[str, Any] | None:
    marker = active.get("repair_marker")
    if not isinstance(marker, dict) or config.repair_lock_path is None:
        return None
    if (marker.get("path") != str(config.repair_lock_path.absolute())
            or marker.get("pid") != active.get("pid")
            or marker.get("start_ticks") != active.get("start_ticks")):
        return None
    return marker


@contextlib.contextmanager
def repair_lock(path: Path | None) -> Iterator[dict[str, Any] | None]:
    """Share a presence marker with older repair agents without taking theirs."""
    if path is None:
        yield
        return
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    start_ticks = probe_process(os.getpid()).get("start_ticks")
    _atomic_json(temporary, {"schema": SCHEMA, "pid": os.getpid(),
                            "start_ticks": start_ticks,
                            "token": token, "created_at": time.time(), **FLAGS})
    information = temporary.stat()
    inode = information.st_ino
    lease = {"path": str(path), "device": information.st_dev, "inode": inode,
             "token": token, "pid": os.getpid(), "start_ticks": start_ticks}
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
        yield lease
    finally:
        temporary.unlink(missing_ok=True)
        _release_repair_marker(lease)


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


def _finish_worker_tree(tree: CommandProcessTree, *, term_already_sent: bool = False) -> None:
    """Let observed descendants finish cleanup within one fixed grace period.

    The registry contains pidfds opened while those exact births were owned.
    Poll their exit readiness rather than trusting reusable PID numbers. Never
    inspect or clear a hold: only its actual owner may release it in finally.
    """
    def live_handles() -> bool:
        handles = [fd for _start, fd in tree._owned.values()]
        if not handles:
            return False
        poller = select.poll()
        for fd in handles:
            poller.register(fd, select.POLLIN)
        return len(poller.poll(0)) < len(handles)

    try:
        tree.snapshot()
        if not term_already_sent:
            tree.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + _DESCENDANT_CLEANUP_GRACE_SECONDS
        while True:
            tree.snapshot()
            if not live_handles():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))
    finally:
        try:
            tree.snapshot()
            tree.send_signal(signal.SIGKILL)
        finally:
            tree.close()


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
        _finish_worker_tree(tree, term_already_sent=True)
    return True


def run_worker(config: Config, log_path: Path, on_start: Callable[[dict[str, Any]], None], *,
               on_poll: Callable[[dict[str, Any]], None] | None = None,
               previous_worker_context_json: str | None = None) -> dict[str, Any]:
    argv = [str(part) for part in (config.launch_argv or config.worker_argv)]
    if not argv or config.prompt_path is None:
        return {"outcome": "configuration_error", "error": "worker argv and prompt path are required"}
    process = None
    tree = None
    storage_result = None
    try:
        with config.prompt_path.open("rb") as prompt, log_path.open("ab") as output:
            environment = dict(os.environ, REPAIR_DISPATCHER_STATUS_PATH=str(config.state_dir / "status.json"))
            environment = context_environment(environment, previous_worker_context_json)
            process = subprocess.Popen(argv, stdin=prompt, stdout=output, stderr=subprocess.STDOUT,
                                       cwd=config.repo_root, env=environment, start_new_session=True)
            tree = CommandProcessTree(process.pid)
            tree.snapshot()
            on_start(probe_process(process.pid))
            deadline = time.monotonic() + config.worker_timeout
            next_control_poll = time.monotonic() + min(30.0, config.interval)
            while True:
                tree.snapshot()
                try:
                    code = process.wait(timeout=min(0.2, max(0.001, deadline - time.monotonic())))
                    return {"outcome": "awaiting_observation" if code == 0 else "failed", "exit_code": code}
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        _stop_owned_worker(process)
                        return {"outcome": "timed_out", "exit_code": process.returncode}
                    # A real bounded wait and the original deadline/owned tree
                    # check must complete before any control-plane heartbeat.
                    tree.snapshot()
                    if on_poll is not None and time.monotonic() >= next_control_poll:
                        identity = probe_process(process.pid)
                        if identity["alive"] and process.poll() is None:
                            on_poll(identity)
                        next_control_poll = time.monotonic() + min(30.0, config.interval)
    except (OSError, ValueError) as exc:
        if is_storage_pressure(exc):
            storage_result = {"outcome": "storage_blocked", "error": str(exc),
                              "error_errno": exc.errno, "spawned": process is not None,
                              "spawned_pid": process.pid if process is not None else None}
            return storage_result
        return {"outcome": "failed", "error": str(exc), "error_errno": getattr(exc, "errno", None)}
    finally:
        if process is not None and process.poll() is None:
            _stop_owned_worker(process)
        if tree is not None:
            _finish_worker_tree(tree)
        if storage_result is not None:
            storage_result["owned_child_stopped"] = process is None or process.poll() is not None


def _storage_blocked(config: Config, telemetry: dict[str, Any], *, error: OSError | None = None,
                     capacity: dict[str, Any] | None = None) -> dict[str, Any]:
    telemetry.update(action="storage_blocked", storage_pressure=True, degraded=True,
                     retry_after_seconds=config.interval, **FLAGS)
    if error is not None:
        telemetry["storage_error_errno"] = error.errno
    if capacity is not None:
        telemetry["storage_capacity"] = capacity
    publish_emergency_status(config.state_dir / "status.json", telemetry, error=error)
    return telemetry


def _remember_storage_payload(config: Config, payload: dict[str, Any]) -> None:
    previous = config._storage_retry_payload
    if previous is None or len(json.dumps(payload)) > len(json.dumps(previous)):
        config._storage_retry_payload = copy.deepcopy(payload)


def _retry_storage_payload(config: Config) -> OSError | None:
    """A small free-space probe cannot prove a previously rejected JSON fits.

    Retry its full allocation in an explicitly non-authoritative, private
    temporary file. Never replay a reaped worker identity into state.json.
    """
    if config._storage_retry_payload is None:
        return None
    if config._storage_retry_path is None:
        config._storage_retry_path = config.state_dir / f".storage-admission-{os.getpid()}-{uuid.uuid4().hex}.json"
    path = config._storage_retry_path
    error = None
    try:
        _atomic_json(path, {"schema": SCHEMA + "/storage-admission-probe", "storage_probe_only": True,
                            "rejected_payload": config._storage_retry_payload, **FLAGS})
    except OSError as exc:
        if not is_storage_pressure(exc):
            raise
        error = exc
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            if not is_storage_pressure(exc):
                raise
            error = exc
    if error is None:
        config._storage_retry_payload = None
    return error


def _save(config: Config, state: dict[str, Any], telemetry: dict[str, Any]) -> bool:
    telemetry.update(control_heartbeat_at=time.time(), control_pid=os.getpid(),
                     queue_hold_lease_version=QUEUE_HOLD_LEASE_VERSION,
                     control_start_ticks=probe_process(os.getpid()).get("start_ticks"))
    pending = state
    try:
        _atomic_json(config.state_dir / "state.json", state)
        pending = telemetry
        _atomic_json(config.state_dir / "status.json", telemetry)
        with (config.state_dir / "events.jsonl").open("a") as events:
            events.write(json.dumps(telemetry, sort_keys=True) + "\n")
            events.flush()
        if telemetry.get("action") != "storage_blocked":
            clear_emergency_status(config.state_dir / "status.json")
        return True
    except OSError as exc:
        if not is_storage_pressure(exc):
            raise
        _remember_storage_payload(config, pending)
        _storage_blocked(config, telemetry, error=exc)
        return False


def _transient_prework_message(message: str) -> bool:
    """True for a provider refusal that ends before repair work starts.

    Authentication and billing failures stay ordinary attempts.  Waiting does
    not clear a bad credential, and those must still reach attention_required.
    """
    if _AUTH_STICK_RE.search(message):
        return False
    return _TRANSIENT_PREWORK_RE.search(message) is not None


def provider_quota_backoff(log_path: Path, *, now: float, fallback_seconds: float) -> dict[str, Any] | None:
    """Return a local retry time when a worker log is only a transient provider refusal.

    A refusal that exits before any repair work must not spend the signature's
    attempt budget.  Other log shapes, including a limit message mixed with
    real events, stay ordinary failures.
    """
    try:
        raw = log_path.read_bytes()
        log_mtime = log_path.stat().st_mtime
    except OSError:
        return None
    if not raw or len(raw) > _QUOTA_LOG_LIMIT or not any(token in raw.lower() for token in _TRANSIENT_PREWORK_TOKENS):
        return None
    messages: list[str] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(item, dict):
            return None
        kind = item.get("type")
        if kind in _QUOTA_NEUTRAL_TYPES:
            continue
        if kind == "error" and isinstance(item.get("message"), str):
            messages.append(item["message"])
            continue
        if kind == "turn.failed" and isinstance(item.get("error"), dict):
            message = item["error"].get("message")
            if isinstance(message, str):
                messages.append(message)
                continue
        return None
    if not messages or any(not _transient_prework_message(message) for message in messages):
        return None
    stated_retry_at = _parse_quota_retry_at(messages[-1], log_mtime=log_mtime)
    retry_at = stated_retry_at
    if (retry_at is None or retry_at < now or retry_at > now + _QUOTA_RETRY_HORIZON_SECONDS):
        retry_at = now + fallback_seconds
    return {"reason": "provider_usage_limit", "retry_at": retry_at,
            "stated_retry_at": stated_retry_at, "log_mtime": log_mtime}


def _parse_quota_retry_at(message: str, *, log_mtime: float) -> float | None:
    match = _QUOTA_RETRY_RE.search(message)
    if match is not None:
        stamp = re.sub(r"(\d)(?:st|nd|rd|th)", r"\1", match.group(1))
        try:
            parsed = datetime.strptime(stamp, "%b %d, %Y %I:%M %p")
        except ValueError:
            parsed = None
        if parsed is not None:
            try:
                return parsed.replace(tzinfo=datetime.now().astimezone().tzinfo).timestamp()
            except (OSError, OverflowError, ValueError):
                return None
    relative = _RELATIVE_RETRY_RE.search(message)
    if relative is None:
        return None
    try:
        amount = int(relative.group(1))
    except ValueError:
        return None
    return log_mtime + amount * _RETRY_UNITS[relative.group(2).lower()]


def _quota_attempt_logs(config: Config, signature: str) -> list[Path]:
    prefix = f"worker-{signature[:16]}-"
    try:
        return sorted(path for path in config.state_dir.glob(prefix + "*.log")
                      if path.is_file() and path.name.removeprefix(prefix).removesuffix(".log").isdigit())
    except OSError:
        return []


def _latest_stated_quota_retry(config: Config, now: float) -> float | None:
    """Use the newest worker log's stated provider time, across repair signatures.

    A signature change must not launch another worker while the provider has
    already named a future retry.  A stamp that is already due does not block:
    the next real attempt is what discovers whether the limit has lifted.
    """
    try:
        logs = [path for path in config.state_dir.glob("worker-*.log") if path.is_file()]
    except OSError:
        return None
    if not logs:
        return None
    try:
        latest = max(logs, key=lambda path: path.stat().st_mtime)
    except OSError:
        return None
    quota = provider_quota_backoff(latest, now=now, fallback_seconds=config.max_cooldown_seconds)
    if quota is None:
        return None
    stated = quota.get("stated_retry_at")
    if not isinstance(stated, (int, float)):
        # Anchor an unstamped refusal to the log time.  Re-reading the same
        # file must not keep sliding the wait forward.
        stated = float(quota.get("log_mtime") or now) + config.max_cooldown_seconds
    if now < float(stated) <= now + _QUOTA_RETRY_HORIZON_SECONDS:
        return float(stated)
    return None


def _recover_quota_exhaustion(config: Config, signature: str, record: dict[str, Any], now: float) -> float | None:
    """Drop a spent budget when every persisted attempt log is only a usage limit."""
    logs = _quota_attempt_logs(config, signature)
    if not logs:
        return None
    latest = None
    for path in logs:
        quota = provider_quota_backoff(path, now=now, fallback_seconds=config.max_cooldown_seconds)
        if quota is None:
            return None
        latest = quota
    assert latest is not None
    record["count"] = 0
    record["provider_quota"] = {"reason": "provider_usage_limit", "retry_at": latest["retry_at"]}
    return float(latest["retry_at"])


def _repair_lock_owner_alive(config: Config) -> bool:
    """True when the repair marker is missing a proof that its owner has exited."""
    path = config.repair_lock_path
    if path is None:
        return False
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if not stat.S_ISREG(info.st_mode):
        return True
    try:
        prior = _json(path)
        owner = probe_process(int(prior.get("pid") or 0))
    except (OSError, ValueError, TypeError):
        return True
    if prior.get("schema") != SCHEMA or not prior.get("token"):
        return True
    return bool(owner.get("alive") and owner.get("start_ticks") == prior.get("start_ticks"))


def _worker_argv_still_running(config: Config, intent: dict[str, Any]) -> bool:
    """True unless every live process fails to match the recorded worker argv.

    An empty argv cannot be distinguished from an unrelated process, so the
    interrupted launch stays uncertain.  A router-selected command is matched
    from the spawn record, not from the configured Codex candidate.
    """
    recorded = intent.get("argv")
    if (isinstance(recorded, list) and recorded
            and all(isinstance(part, str) and part for part in recorded)):
        expected = list(recorded)
    else:
        expected = [str(part) for part in config.worker_argv]
    if not expected:
        return True
    digest = intent.get("argv_digest")
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return True
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        command = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if command[:len(expected)] != expected:
            continue
        if isinstance(digest, str) and digest and _digest(command[:len(expected)]) != digest:
            continue
        return True
    return False


def _spawn_intent_still_uncertain(config: Config, intent: Any, now: float) -> bool:
    """Keep a single-launch guard until the worker deadline passes with no survivor.

    A restart between the spawn record and the worker handoff must not start a
    second repair.  After the original deadline, a dead lock and no matching
    process means that launch did not survive, so the dispatcher may continue.
    """
    if not isinstance(intent, dict):
        return True
    try:
        started = float(intent.get("started_at"))
    except (TypeError, ValueError):
        return True
    if now < started + config.worker_timeout:
        return True
    if _repair_lock_owner_alive(config):
        return True
    return _worker_argv_still_running(config, intent)


def _remember_provider_quota(config: Config, state: dict[str, Any], record: dict[str, Any],
                             result: dict[str, Any], log_path: Path, finished: float) -> bool:
    if result.get("outcome") != "failed":
        record.pop("provider_quota", None)
        return False
    quota = provider_quota_backoff(log_path, now=finished, fallback_seconds=config.max_cooldown_seconds)
    if quota is None:
        record.pop("provider_quota", None)
        return False
    record["count"] = max(0, int(record.get("count") or 0) - 1)
    provider = _provider_log_index(config).get(log_path.name, "")
    record["provider_quota"] = {"reason": "provider_usage_limit", "retry_at": quota["retry_at"],
                                **({"provider": provider} if provider else {})}
    record["next_dispatch_at"] = quota["retry_at"]
    state["next_dispatch_at"] = quota["retry_at"]
    return True


def _argv_provider(argv: list[str]) -> str:
    """Return the implementation source named by an argv, or an empty string."""
    if not argv:
        return ""
    name = Path(str(argv[0])).name
    if name == "codex":
        return "codex"
    if name in {"grok", "grok.exe"}:
        return "grok"
    return ""


def _grok_binary(_config: Config) -> str:
    """Locate the Grok CLI without starting it or calling a model."""
    from ipfs_accelerate_py.llm_router import find_grok_cli

    found = str(find_grok_cli() or "").strip()
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "grok"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return ""


def _codex_binary(config: Config) -> str:
    """Return the configured Codex executable when that file is executable."""
    argv = config.worker_argv
    if _argv_provider([str(part) for part in argv]) != "codex":
        return ""
    candidate = Path(argv[0])
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return ""


def _provider_log_index(config: Config) -> dict[str, str]:
    path = config.state_dir / "provider-log-index.json"
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {str(name): provider for name, provider in value.items() if provider in {"grok", "codex"}}


def _bind_log_provider(config: Config, log_path: Path, provider: str) -> None:
    if provider not in {"grok", "codex"}:
        return
    index = _provider_log_index(config)
    index[log_path.name] = provider
    _atomic_json(config.state_dir / "provider-log-index.json", index)


def _plain_transient_backoff(raw: bytes, *, log_mtime: float, now: float,
                             fallback_seconds: float) -> dict[str, Any] | None:
    """A one-line pre-work refusal, used for a Grok log that is not Codex JSON."""
    if not raw or len(raw) > _QUOTA_LOG_LIMIT or b"\n" in raw.strip():
        return None
    text = raw.decode("utf-8", "replace").strip()
    if not text or text.startswith("{") or not _transient_prework_message(text):
        return None
    stated_retry_at = _parse_quota_retry_at(text, log_mtime=log_mtime)
    retry_at = stated_retry_at
    if retry_at is None or retry_at < now or retry_at > now + _QUOTA_RETRY_HORIZON_SECONDS:
        retry_at = now + fallback_seconds
    return {"reason": "provider_usage_limit", "retry_at": retry_at,
            "stated_retry_at": stated_retry_at, "log_mtime": log_mtime}


def _classify_worker_log(config: Config, log_path: Path, now: float) -> tuple[str, dict[str, Any]] | None:
    """Attribute a pure pre-work refusal to the provider that produced it."""
    hint = _provider_log_index(config).get(log_path.name, "")
    quota = provider_quota_backoff(log_path, now=now, fallback_seconds=config.max_cooldown_seconds)
    if quota is not None:
        return (hint or "codex"), quota
    if hint != "grok":
        return None
    try:
        raw = log_path.read_bytes()
        log_mtime = log_path.stat().st_mtime
    except OSError:
        return None
    plain = _plain_transient_backoff(raw, log_mtime=log_mtime, now=now,
                                     fallback_seconds=config.max_cooldown_seconds)
    if plain is None:
        return None
    return "grok", plain


def _log_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _provider_quota_deadlines(config: Config, now: float) -> dict[str, float]:
    """Future refusal time for each provider, from that provider's newest log.

    A Codex usage-limit log does not become a deadline for Grok.  A newer log
    from the same provider that is real work clears that provider's deadline.
    """
    try:
        logs = [path for path in config.state_dir.glob("worker-*.log") if path.is_file()]
    except OSError:
        return {}
    index = _provider_log_index(config)
    newest: dict[str, Path] = {}
    for path in sorted(logs, key=_log_mtime):
        provider = index.get(path.name, "")
        if not provider and provider_quota_backoff(
                path, now=now, fallback_seconds=config.max_cooldown_seconds) is not None:
            provider = "codex"
        if provider in {"grok", "codex"}:
            newest[provider] = path
    deadlines: dict[str, float] = {}
    for provider, path in newest.items():
        classified = _classify_worker_log(config, path, now)
        if classified is None or classified[0] != provider:
            continue
        retry_at = classified[1].get("retry_at")
        if isinstance(retry_at, (int, float)) and now < float(retry_at) <= now + _QUOTA_RETRY_HORIZON_SECONDS:
            deadlines[provider] = float(retry_at)
    return deadlines


def _repair_source_observations(config: Config, now: float):
    """Non-charging observations for the two repair sources."""
    from ipfs_accelerate_py.llm_router import RouterOwnedProviderObservation

    deadlines = _provider_quota_deadlines(config, now)
    binaries = {"grok": _grok_binary(config), "codex": _codex_binary(config)}
    observations = []
    for provider in ("grok", "codex"):
        binary = binaries[provider]
        retry_at = deadlines.get(provider)
        exhausted = retry_at is not None
        observations.append(RouterOwnedProviderObservation(
            provider_id=provider,
            ready=bool(binary) and not exhausted,
            authenticated=bool(binary),
            binary_available=bool(binary),
            hard_quota_exhausted=exhausted,
            retry_at="" if retry_at is None else str(retry_at),
            source="repair_dispatcher",
        ))
    return tuple(observations), deadlines


def _grok_repair_argv(config: Config) -> list[str]:
    """Build the Grok repair command with the router's own argv helper."""
    from ipfs_accelerate_py.llm_router import build_grok_cli_command

    if config.prompt_path is None:
        raise ValueError("repair prompt is required")
    return build_grok_cli_command(
        mode="agent",
        workspace=config.repo_root,
        model_name="grok-4.7",
        grok_bin=_grok_binary(config),
        prompt_file=config.prompt_path,
        permission_mode="bypassPermissions",
    )


def _choose_repair_source(config: Config, now: float) -> dict[str, Any]:
    """Ask the router which repair source is usable.  This makes no model call."""
    from ipfs_accelerate_py.llm_router import (
        LLMRouterError, ROUTER_OWNED_COMPATIBILITY_CANONICAL,
        decide_router_owned_implementation_provider,
    )

    observations, deadlines = _repair_source_observations(config, now)
    decision = decide_router_owned_implementation_provider(
        observations,
        preferred_provider="grok",
        fallback_provider="codex",
        secondary_providers=("codex",),
        allow_secondary_without_preferred_quota=False,
        compatibility_mode=ROUTER_OWNED_COMPATIBILITY_CANONICAL,
    )
    argv: list[str] = []
    if decision.authorized and decision.selected_provider == "grok":
        try:
            argv = _grok_repair_argv(config)
        except (OSError, ValueError, LLMRouterError):
            argv = []
    elif decision.authorized and decision.selected_provider == "codex":
        argv = [str(part) for part in config.worker_argv]
    authorized = bool(decision.authorized and argv)
    retry_at = None
    if not authorized:
        try:
            parsed = float(decision.retry_at) if decision.retry_at else None
        except ValueError:
            parsed = None
        if parsed is not None and now < parsed <= now + _QUOTA_RETRY_HORIZON_SECONDS:
            retry_at = parsed
        elif deadlines:
            future = [stamp for stamp in deadlines.values() if stamp > now]
            retry_at = min(future) if future else None
    if not authorized and deadlines:
        reason = "provider_usage_limit"
    elif decision.reason_codes:
        reason = str(decision.reason_codes[0])
    else:
        reason = "no_eligible_implementation_provider"
    return {"authorized": authorized, "decision": decision.decision if authorized else decision.decision,
            "selected_provider": decision.selected_provider if authorized else "",
            "reason_codes": list(decision.reason_codes), "reason": reason, "retry_at": retry_at,
            "argv": argv, "deadlines": deadlines}


_EXTERNAL_GATE_INCIDENTS = frozenset({"gate_fixed_point", "planned_work_blocked_frontier"})
# Grace and deferred are the same proofs while the lane cursor or a gate
# revalidation heartbeat is outside the retry-wait set. They must not dispatch.
_EXTERNAL_FRONTIER_STATES = frozenset({
    "plan_replan_required", "blocked_frontier_deferred", "blocked_frontier_grace",
})


def _unchanged_exit75_frontier(observation: dict[str, Any], state: dict[str, Any]) -> bool:
    """True when a clean repair already saw this same exit-75 frontier.

    The lane cursor rotates the active task across the same withheld proofs.
    A gate revalidation heartbeat also drops the frontier out of
    plan_replan_required for one observation. Neither event opens a new
    repair attempt. A later proof change resets the frontier clock and
    repair can run again. Tasks stay pending.
    """
    planned = observation.get("planned_work")
    if not isinstance(planned, dict) or planned.get("state") not in _EXTERNAL_FRONTIER_STATES:
        return False
    blocked = planned.get("blocked_task_ids")
    selectable = observation.get("selectable_task_ids")
    proofs = planned.get("blocked_frontier_proofs")
    incidents = observation.get("incidents")
    fingerprint = planned.get("blockage_fingerprint")
    if (not isinstance(blocked, list) or not blocked
            or not isinstance(selectable, list) or not set(selectable) <= set(blocked)
            or not isinstance(proofs, list) or not proofs
            or any(not isinstance(proof, dict) or proof.get("last_exit_code") != 75 for proof in proofs)
            or not isinstance(incidents, list) or not incidents
            or any(not isinstance(item, dict) or item.get("kind") not in _EXTERNAL_GATE_INCIDENTS for item in incidents)
            or not isinstance(fingerprint, str) or len(fingerprint) != 64):
        return False
    def _clean(record: Any) -> bool:
        return (isinstance(record, dict) and record.get("exit_code") == 0
                and record.get("work_complete") is False
                and record.get("outcome") == "awaiting_observation")

    frontier = state.get("blocked_frontier_observation")
    last = state.get("last_worker")
    settled = (state.get("settled_exit75_frontiers") or {}).get(fingerprint) if isinstance(
        state.get("settled_exit75_frontiers"), dict) else None
    clock_ok = False
    if isinstance(frontier, dict) and isinstance(last, dict):
        first = frontier.get("first_seen")
        finished = last.get("finished_at")
        clock_ok = (frontier.get("signature") == fingerprint and _clean(last)
                    and type(first) in (int, float) and not isinstance(first, bool)
                    and type(finished) in (int, float) and not isinstance(finished, bool)
                    and first <= float(finished))
    if not clock_ok and not _clean(settled):
        return False
    if not _clean(settled):
        remembered = state.setdefault("settled_exit75_frontiers", {})
        if isinstance(remembered, dict):
            remembered[fingerprint] = {"exit_code": 0, "work_complete": False,
                                       "outcome": "awaiting_observation",
                                       "finished_at": last.get("finished_at")}
    return True


def _matches_foreign_deadline(stamp: Any, foreign: list[float]) -> bool:
    return isinstance(stamp, (int, float)) and any(abs(float(stamp) - item) < 1.0 for item in foreign)


def _drop_foreign_provider_deadline(state: dict[str, Any], record: dict[str, Any], selected: str,
                                    deadlines: dict[str, float]) -> None:
    """A deadline recorded for another source must not cool down the selected one.

    Older dispatchers copied a Codex refusal onto the attempt record without a
    provider name.  That untagged stamp is still a foreign deadline when it
    matches another source's refusal time.
    """
    foreign = [stamp for provider, stamp in deadlines.items() if provider != selected]
    if _matches_foreign_deadline(state.get("next_dispatch_at"), foreign):
        state["next_dispatch_at"] = 0
    quota = record.get("provider_quota") if isinstance(record.get("provider_quota"), dict) else {}
    tagged = str(quota.get("provider") or "")
    foreign_quota = bool(tagged and tagged != selected)
    untagged_foreign = bool(not tagged and quota and (
        _matches_foreign_deadline(quota.get("retry_at"), foreign)
        or _matches_foreign_deadline(record.get("next_dispatch_at"), foreign)))
    if foreign_quota or untagged_foreign or _matches_foreign_deadline(record.get("next_dispatch_at"), foreign):
        if foreign_quota or untagged_foreign:
            record.pop("provider_quota", None)
        if _matches_foreign_deadline(record.get("next_dispatch_at"), foreign):
            record["next_dispatch_at"] = 0


def tick(
    config: Config, state: dict[str, Any], *, apply: bool = False, now: float | None = None,
    probe: Callable[[int], dict[str, Any]] = probe_process,
    worker: Callable[..., dict[str, Any]] = run_worker,
) -> dict[str, Any]:
    use_wall_clock = now is None
    now = time.time() if now is None else now
    state.update({"schema": SCHEMA, **FLAGS})
    queue_retirement = expire_queue_hold(config, state, now, apply=apply)
    try:
        observation = observe(config, state, now=now, probe=probe)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        telemetry = {"schema": SCHEMA, "observed_at": now,
                     "last_full_observation_at": state.get("last_full_observation_at"),
                     "action": "observation_error", "error": str(exc), **FLAGS}
        _save(config, state, telemetry)
        return telemetry
    state["last_full_observation_at"] = observation["observed_at"]
    telemetry = {**observation, "last_full_observation_at": observation["observed_at"],
                 "action": "observational", "apply": apply,
                 **({"queue_hold_retirement": queue_retirement} if "retirement" in queue_retirement else {})}
    active = state.get("active_worker", {})
    if active:
        live = probe(int(active.get("pid") or 0))
        if live["alive"] and live.get("start_ticks") == active.get("start_ticks"):
            marker = _saved_repair_marker(config, active)
            marker_mismatch = "repair_marker" in active and (marker is None or not _repair_marker_matches(marker))
            deadline = active.get("deadline")
            if deadline is None:
                telemetry["action"] = "attention_required"
                telemetry["reason"] = "adopted_worker_missing_deadline"
            elif now >= float(deadline):
                if observation["held"]:
                    # Preserve holds while making the expired original deadline explicit.
                    telemetry.update(action="held", reason="adopted_worker_deadline_exceeded_while_held",
                                     worker_deadline=float(deadline),
                                     worker_overdue_seconds=max(0.0, now - float(deadline)), degraded=True)
                elif not apply:
                    telemetry["action"] = "would_timeout_adopted_worker"
                else:
                    stopped = stop_adopted_worker(active)
                    telemetry["action"] = "adopted_worker_timed_out" if stopped else "attention_required"
                    if stopped:
                        _release_repair_marker(_saved_repair_marker(config, active))
                        state.pop("active_worker", None)
                        state["last_worker"] = {"outcome": "timed_out", "finished_at": now, **FLAGS}
                        state["next_dispatch_at"] = now + config.cooldown_seconds
            elif marker_mismatch:
                telemetry.update(action="attention_required", reason="adopted_worker_presence_mismatch")
            else:
                telemetry["action"] = "worker_already_running"
            _save(config, state, telemetry)
            return telemetry
        _release_repair_marker(_saved_repair_marker(config, active))
        state.pop("active_worker", None)
        state["last_worker"] = {"outcome": "interrupted", "finished_at": now, **FLAGS}
    if state.get("spawn_intent") and _spawn_intent_still_uncertain(config, state.get("spawn_intent"), now):
        telemetry.update({"action": "attention_required", "reason": "spawn_uncertain_after_dispatcher_interruption"})
        _save(config, state, telemetry)
        return telemetry
    state.pop("spawn_intent", None)
    if any(review["reason"] in {"invalid_required_key", "expected_environment_missing"}
           for review in observation["lane_environment_reviews"]):
        telemetry["action"] = "lane_environment_configuration_error"
        _save(config, state, telemetry)
        return telemetry
    if observation.get("storage_pressure"):
        _storage_blocked(config, telemetry)
        _save(config, state, telemetry)
        return telemetry
    capacity = check_storage_capacity([config.state_dir, config.manifest_path.parent, config.repo_root,
                                       *([config.repair_lock_path.parent] if config.repair_lock_path else []),
                                       *(Path(path) for path in observation["storage_paths"])],
                                      minimum_free_bytes=max(observation["storage_minimum_free_bytes"],
                                                             config.worker_minimum_free_bytes))
    if not capacity["ok"]:
        _storage_blocked(config, telemetry, capacity=capacity)
        _save(config, state, telemetry)
        return telemetry
    storage_error = _retry_storage_payload(config)
    if storage_error is not None:
        _storage_blocked(config, telemetry, error=storage_error)
        _save(config, state, telemetry)
        return telemetry
    signature = observation["signature"]
    if observation["held"]:
        telemetry["action"] = "owner_uncertain" if observation["owner_discovery_uncertain"] else "held"
    elif not signature:
        planned_state = observation["planned_work"].get("state")
        planned_wait = (observation["planned_work"].get("enabled") and planned_state not in
                        {"disabled", "complete", "execution_available", "held", "materialization_grace"})
        environment_identity_wait = any(review["reason"] == "lane_identity_changed"
                                        for review in observation["lane_environment_reviews"])
        unresolved = (observation["retry_failure_observation_pending"] or
                      observation["native_retry_wait_observed"] or
                      bool(observation["native_rotation_grace"]) or
                      environment_identity_wait or
                      planned_wait or bool(observation["frontier_gate_reviews"]) or
                      bool(observation["source_provenance_reviews"]))
        telemetry["action"] = ("source_change_review_needed" if observation["source_provenance_reviews"]
                               else "gate_evidence_review_needed" if observation["frontier_gate_reviews"]
                               else "native_lease_rotation_wait" if observation["native_rotation_grace"]
                               else "lane_environment_identity_wait" if environment_identity_wait
                               else "replan_deferred_active_work" if planned_state == "blocked_frontier_deferred"
                               else "blocked_frontier_wait" if planned_state == "blocked_frontier_grace"
                               else "waiting_external_gate" if planned_state == "waiting_external_gate"
                               else "plan_dependency_wait" if planned_wait
                               else "retry_wait" if unresolved else "healthy")
        if state.get("unresolved_signature") and not unresolved:
            telemetry["action"] = "recovery_observed"
            state["recovery_observed_at"] = now
            state["attempts"] = {}
            state["next_dispatch_at"] = 0
        if not unresolved:
            state["unresolved_signature"] = ""
    else:
        state["unresolved_signature"] = signature
        record = state.setdefault("attempts", {}).setdefault(signature, {"count": 0, "next_dispatch_at": 0})
        telemetry["attempt_count"] = record["count"]
        telemetry["max_attempts"] = config.max_attempts
        selected_argv = [str(part) for part in config.worker_argv]
        selected_provider = ""
        routed = _argv_provider(selected_argv) == "codex"
        router_blocked = False
        if routed:
            choice = _choose_repair_source(config, now)
            telemetry["router_decision"] = choice["decision"]
            telemetry["router_provider"] = choice["selected_provider"]
            telemetry["router_reasons"] = choice["reason_codes"]
            if choice["authorized"]:
                selected_provider = str(choice["selected_provider"])
                selected_argv = list(choice["argv"])
                _drop_foreign_provider_deadline(state, record, selected_provider, choice["deadlines"])
            else:
                router_blocked = True
                retry_at = choice["retry_at"]
                if not isinstance(retry_at, (int, float)):
                    retry_at = now + config.cooldown_seconds
                record["next_dispatch_at"] = retry_at
                state["next_dispatch_at"] = retry_at
                telemetry["action"] = "provider_backoff"
                telemetry["reason"] = choice["reason"]
                telemetry["next_dispatch_at"] = float(retry_at)
                telemetry["attempt_count"] = record["count"]
        if not router_blocked:
            telemetry["next_dispatch_at"] = max(record["next_dispatch_at"], state.get("next_dispatch_at", 0))
        provider_quota = record.get("provider_quota") if isinstance(record.get("provider_quota"), dict) else {}
        provider_retry_at = None if routed else provider_quota.get("retry_at")
        stated_retry_at = None if routed else _latest_stated_quota_retry(config, now)
        if stated_retry_at is not None:
            provider_retry_at = stated_retry_at
            record["provider_quota"] = {"reason": "provider_usage_limit", "retry_at": stated_retry_at}
            record["next_dispatch_at"] = stated_retry_at
            state["next_dispatch_at"] = stated_retry_at
        if router_blocked:
            pass
        elif record["count"] >= config.max_attempts:
            recovered_retry_at = _recover_quota_exhaustion(config, signature, record, now)
            telemetry["attempt_count"] = record["count"]
            if recovered_retry_at is None:
                telemetry["action"] = "attention_required"
            else:
                telemetry["action"] = "provider_backoff"
                telemetry["reason"] = "provider_usage_limit"
                telemetry["next_dispatch_at"] = recovered_retry_at
                record["next_dispatch_at"] = recovered_retry_at
                state["next_dispatch_at"] = recovered_retry_at
        elif _unchanged_exit75_frontier(observation, state):
            # The proofs did not change after the last clean repair. Launching
            # another worker only because the lane cursor moved, or because a
            # lease restart cleared the frontier clock, spends the attempt
            # budget without a new gate result.
            telemetry["action"] = "waiting_external_gate"
            telemetry["reason"] = "unchanged_exit75_frontier"
            telemetry["attempt_count"] = record["count"]
        elif isinstance(provider_retry_at, (int, float)) and now < float(provider_retry_at):
            telemetry["action"] = "provider_backoff"
            telemetry["reason"] = "provider_usage_limit"
            telemetry["next_dispatch_at"] = float(provider_retry_at)
        elif not apply:
            telemetry["action"] = "would_dispatch"
        elif now < telemetry["next_dispatch_at"]:
            telemetry["action"] = "cooldown"
        else:
            before_dispatch = copy.deepcopy(state)
            worker_started = False
            try:
                with repair_lock(config.repair_lock_path) as presence:
                    record["count"] += 1
                    delay = min(config.max_cooldown_seconds, config.cooldown_seconds * 2 ** min(record["count"] - 1, 30))
                    record["next_dispatch_at"] = now + delay
                    state["next_dispatch_at"] = now + delay
                    telemetry.update({"action": "dispatching", "attempt_count": record["count"], "next_dispatch_at": now + delay,
                                      **({"router_provider": selected_provider} if selected_provider else {})})
                    state["spawn_intent"] = {"signature": signature, "started_at": now, "argv": list(selected_argv),
                                             "argv_digest": _digest(selected_argv),
                                             **({"provider": selected_provider} if selected_provider else {})}
                    if not _save(config, state, telemetry):
                        state.clear()
                        state.update(before_dispatch)
                        telemetry.update(attempt_count=state.get("attempts", {}).get(signature, {}).get("count", 0),
                                         next_dispatch_at=state.get("next_dispatch_at", 0))
                        _storage_blocked(config, telemetry)
                        return telemetry

                    def started(identity: dict[str, Any]) -> None:
                        # Handoff before state commit: if the dispatcher dies
                        # here, live worker presence still excludes contenders
                        # and the persisted spawn_intent refuses another launch.
                        marker = _transfer_repair_marker(presence, identity)
                        state["active_worker"] = {**_public_process(identity), "owned_session": True,
                                                  "started_at": now, "deadline": now + config.worker_timeout}
                        if marker is not None:
                            state["active_worker"]["repair_marker"] = marker
                        state.pop("spawn_intent", None)
                        try:
                            _atomic_json(config.state_dir / "state.json", state)
                        except OSError as exc:
                            if is_storage_pressure(exc):
                                _remember_storage_payload(config, state)
                            raise

                    log_path = config.state_dir / f"worker-{signature[:16]}-{record['count']}.log"
                    config.launch_argv = list(selected_argv)
                    try:
                        if selected_provider in {"grok", "codex"}:
                            _bind_log_provider(config, log_path, selected_provider)
                        worker_started = True
                        result = worker(config, log_path, started)
                    finally:
                        config.launch_argv = []
                    if result.get("outcome") == "storage_blocked":
                        # run_worker has already stopped and reaped only its
                        # owned child when storage admission/persistence fails.
                        if result.get("spawned") is True:
                            # Popen is a real attempt: even a quickly stopped
                            # child may have produced side effects already.
                            state.pop("spawn_intent", None)
                            state.pop("active_worker", None)
                            finished = time.time() if use_wall_clock else now
                            record["next_dispatch_at"] = finished + delay
                            state["next_dispatch_at"] = finished + delay
                            state["last_worker"] = {**result, "outcome": "storage_interrupted",
                                                    "finished_at": finished, "signature": signature,
                                                    "log_path": str(log_path), **FLAGS}
                        else:
                            state.clear()
                            state.update(before_dispatch)
                        telemetry.update(attempt_count=state.get("attempts", {}).get(signature, {}).get("count", 0),
                                         next_dispatch_at=state.get("next_dispatch_at", 0))
                        _storage_blocked(config, telemetry)
                    else:
                        state.pop("spawn_intent", None)
                        state.pop("active_worker", None)
                        finished = time.time() if use_wall_clock else now
                        record["next_dispatch_at"] = finished + delay
                        state["next_dispatch_at"] = finished + delay
                        state["last_worker"] = {**result, "finished_at": finished, "signature": signature, "log_path": str(log_path), **FLAGS}
                        if _remember_provider_quota(config, state, record, result, log_path, finished):
                            telemetry.update({"action": "provider_backoff", "reason": "provider_usage_limit",
                                              "worker": state["last_worker"], "attempt_count": record["count"],
                                              "next_dispatch_at": record["next_dispatch_at"]})
                        else:
                            telemetry.update({"action": "worker_finished", "worker": state["last_worker"], "next_dispatch_at": finished + delay})
                    # No recovery is inferred from exit 0.  The next tick reads live evidence again.
            except AlreadyRunning:
                telemetry.update({"action": "repair_already_running", "repair_lock_path": str(config.repair_lock_path)})
            except OSError as exc:
                if not is_storage_pressure(exc):
                    raise
                if not worker_started:
                    state.clear()
                    state.update(before_dispatch)
                telemetry.update(attempt_count=state.get("attempts", {}).get(signature, {}).get("count", 0),
                                 next_dispatch_at=state.get("next_dispatch_at", 0))
                _storage_blocked(config, telemetry, error=exc)
    _save(config, state, telemetry)
    return telemetry


def _notify_control_watchdog() -> bool:
    """Notify systemd only when its watchdog explicitly targets this PID."""
    try:
        address = os.environ.get("NOTIFY_SOCKET", "")
        if (not address or int(os.environ.get("WATCHDOG_PID", "0")) != os.getpid()
                or int(os.environ.get("WATCHDOG_USEC", "0")) <= 0
                or not address.startswith(("/", "@"))):
            return False
        if address.startswith("@"):
            address = "\0" + address[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.settimeout(0.05)
            channel.sendto(b"WATCHDOG=1", address)
        return True
    except (OSError, ValueError, TypeError):
        # An optional notification failure cannot stop the CLI or its worker.
        return False


def _control_worker(config: Config, state: dict[str, Any], log_path: Path,
                    on_start: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Keep the synchronous owned worker observable after genuine poll steps."""
    last_observation = _json(config.state_dir / "status.json")

    def polled(identity: dict[str, Any]) -> None:
        active = state.get("active_worker", {})
        if (identity.get("pid") != active.get("pid")
                or identity.get("start_ticks") != active.get("start_ticks")):
            return
        telemetry = {**last_observation, "action": "worker_running",
                     "control_phase": "worker_poll", "active_worker": dict(active)}
        # Preserve observed_at/last_full_observation_at: this is control
        # liveness, never a new lane observation, repair, or task progress.
        _save(config, state, telemetry)
        _notify_control_watchdog()

    return run_worker(config, log_path, on_start, on_poll=polled,
                      previous_worker_context_json=build_previous_worker_context(
                          config.state_dir, state.get("last_worker")))


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
    parser.add_argument("--required-lane-env", action="append", default=[],
                        help="require a nonsecret uppercase *_BUDGET_POLICY variable to match this dispatcher's environment")
    parser.add_argument("--once", action="store_true", help="observe one cycle; dispatch requires --apply")
    parser.add_argument("--apply", action="store_true", help="enable the explicitly configured repair worker")
    for option, default in (("interval", 30), ("heartbeat-seconds", 120), ("report-seconds", 600),
                            ("quiet-seconds", 900), ("startup-grace-seconds", 180), ("worker-timeout", 1800),
                            ("cooldown-seconds", 300), ("max-cooldown-seconds", 3600)):
        parser.add_argument(f"--{option}", type=float, default=default)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--worker-min-free-bytes", dest="worker_minimum_free_bytes", type=int,
                        default=64 * 1024 * 1024, help="minimum free bytes before admitting a new repair worker")
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r"[A-Z][A-Z0-9_]*_BUDGET_POLICY", key) for key in args.required_lane_env):
        parser.error("--required-lane-env must name an uppercase *_BUDGET_POLICY key")
    try:
        worker_argv = json.loads(args.worker_argv_json) if args.worker_argv_json else []
    except ValueError:
        parser.error("--worker-argv-json must be a JSON array")
    if not isinstance(worker_argv, list) or any(not isinstance(s, str) or not s for s in worker_argv):
        parser.error("worker argv must be an array of non-empty strings")
    if args.apply and (not worker_argv or not args.prompt_path):
        parser.error("--apply requires --worker-argv-json and --prompt-path")
    for name in ("interval", "heartbeat_seconds", "report_seconds", "quiet_seconds", "startup_grace_seconds",
                 "worker_timeout", "cooldown_seconds", "max_cooldown_seconds", "max_attempts", "worker_minimum_free_bytes"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.worker_minimum_free_bytes > 2**63 - 1:
        parser.error("worker_minimum_free_bytes exceeds signed64-bit bound")
    root = args.repo_root.resolve()
    manifest_path = _path(root, args.manifest_path).resolve()
    config = Config(manifest_path=manifest_path, repo_root=root,
                    state_dir=_path(root, args.state_dir) if args.state_dir else manifest_path.parent / "repair-dispatcher",
                    prompt_path=_path(root, args.prompt_path) if args.prompt_path else None, worker_argv=worker_argv,
                    owner_pid_path=_path(root, args.owner_pid_path) if args.owner_pid_path else None,
                    repair_lock_path=_path(root, args.repair_lock_path) if args.repair_lock_path else None,
                    hold_paths=tuple(_path(root, p) for p in args.hold_path),
                    evidence_paths=tuple(_path(root, p) for p in args.evidence_path),
                    required_lane_env=tuple(sorted(set(args.required_lane_env))),
                    worker_minimum_free_bytes=args.worker_minimum_free_bytes,
                    **{key: getattr(args, key) for key in ("interval", "heartbeat_seconds", "report_seconds", "quiet_seconds",
                       "startup_grace_seconds", "worker_timeout", "cooldown_seconds", "max_cooldown_seconds", "max_attempts")})
    try:
        def stop(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        with contextlib.ExitStack() as ownership:
            while True:
                try:
                    ownership.enter_context(singleton_lock(config.state_dir / "dispatcher.lock"))
                    break
                except OSError as exc:
                    if not is_storage_pressure(exc):
                        raise
                    result = _storage_blocked(config, {"schema": SCHEMA, "observed_at": time.time(), **FLAGS}, error=exc)
                    print(json.dumps(result), flush=True)
                    if args.once:
                        return 2
                    time.sleep(config.interval)
            try:
                state = _json(config.state_dir / "state.json")
            except FileNotFoundError:
                state = {}
            except (OSError, ValueError) as exc:
                print(json.dumps({"action": "state_error", "error": str(exc), **FLAGS}), flush=True)
                return 2
            while True:
                result = tick(config, state, apply=args.apply,
                              worker=lambda cfg, log, started: _control_worker(cfg, state, log, started))
                print(json.dumps({k: result.get(k) for k in ("action", "signature", "observed_at", "control_heartbeat_at", "attempt_count", "error")}), flush=True)
                _notify_control_watchdog()
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
