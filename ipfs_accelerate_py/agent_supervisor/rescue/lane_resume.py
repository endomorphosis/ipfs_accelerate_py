"""Fenced deterministic resume for a stopped lane that still has pending work.

The watchdog calls :func:`resume_stopped_lane` once. It starts this module as
the lane process. The process heartbeats, runs repo-local commands declared on
selectable tasks, and refuses provider CLIs. It never grants completion
authority to the watchdog.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import marshal
import os
import re
import select
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import CodeType
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from .command_processes import CommandProcessTree
from .storage_pressure import MINIMUM_FREE_BYTES, StorageGuard, is_storage_pressure, persist_text, publish_emergency_status, open_command_log, check_storage_capacity
import tempfile
from .gate_recovery import GATE_REPEAT_LIMIT, GateRecovery, gate_evidence, gate_source_contract, _is_source_path
from .vdg_pre_command_recovery import RecoveryPolicyError, recovery_commands, recovery_contract, recovery_environment
from . import semantic_recovery

_PROVIDER_BASENAMES = frozenset(
    {"grok", "codex", "claude", "muse", "gemini", "mistral"}
)
_ALLOWED_BASENAMES = frozenset({"node", "python3", "python", "test"})
_SHELL_META = set(";|&$`<>\n")
_COMPLETABLE_STATUSES = frozenset({"pending", "ready", "in_progress"})
_ATTEMPT_LIMIT = 3
_GATE_WITHHELD_EXIT = 75
_RESOURCE_RETRY_LIMIT = 6
_RESOURCE_RETRY_BASE_MS = 30_000
_EXHAUSTED_RETRY_MS = 900_000
_RETRY_AFTER_KEY = "_retry_after_ms"
_FIXED_POINT_KEY = "_validation_fixed_points"
_COMMAND_EVIDENCE_KEY = "_command_evidence"
_SCHEMA = "ipfs_accelerate_py/agent-supervisor/lifecycle-status@1"
_COMMAND_POLL_SECONDS = 5.0
_COMMAND_STOP_GRACE_SECONDS = 5.0
_COMMAND_TIMEOUT_EXIT = 124
_NODE_PRELOAD_RESOLVE_SECONDS = 5.0
_RESUME_READY_SECONDS = 15.0


class _LaneResumeInterrupted(Exception):
    """Unwind an in-flight command before publishing the lane's stopped state."""


def _header_prefix(value: str) -> str:
    prefix = str(value or "").strip()
    if prefix and not prefix.startswith("## "):
        prefix = f"## {prefix}"
    return prefix


def _order_explicit_task_priorities(
    task_ids: Sequence[str], blocks: Mapping[str, Mapping[str, str]],
) -> list[str]:
    """Honor P0/P1/... among selectable tasks, stably within each tier.

    Unspecified or differently encoded priorities keep their document slots;
    a numeric/Fibonacci value must not acquire invented urgency semantics.
    Dependency selection, refusal checks and retry cooldowns stay separate.
    """

    ordered = list(task_ids)
    annotated: list[tuple[int, str, int]] = []
    for index, task_id in enumerate(ordered):
        match = re.fullmatch(r"[Pp]([0-9]+)", str((blocks.get(task_id) or {}).get("priority") or "").strip())
        if match:
            try:
                annotated.append((index, task_id, int(match.group(1))))
            except ValueError:
                # Extremely long numeric input has no established priority.
                continue
    for (index, _task_id, _rank), (_old_index, task_id, _priority) in zip(
        annotated, sorted(annotated, key=lambda item: item[2]),
    ):
        ordered[index] = task_id
    return ordered


def parse_deterministic_command(raw: str, *, repo_root: Path) -> list[str] | None:
    """Return a safe argv, or None when the command is a shell or provider CLI."""

    text = str(raw or "").strip()
    if not text or any(char in text for char in _SHELL_META):
        return None
    try:
        argv = shlex.split(text)
    except ValueError:
        return None
    if not argv:
        return None
    base = Path(argv[0]).name
    if base in _PROVIDER_BASENAMES:
        return None
    if base in _ALLOWED_BASENAMES:
        return argv
    candidate = Path(argv[0])
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        candidate.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return [str(candidate), *argv[1:]]


def mark_task_completed(
    todo_text: str,
    task_id: str,
    *,
    task_header_prefix: str,
    note: str,
) -> str:
    """Mark one pending task completed. Leave every other block unchanged."""

    prefix = _header_prefix(task_header_prefix)
    if not prefix or not task_id:
        raise ValueError("task identity is required")
    lines = todo_text.splitlines(keepends=True)
    rewritten: list[str] = []
    in_block = False
    replaced = False
    for line in lines:
        if line.startswith(prefix):
            header = line[3:].strip().split(" ", 1)[0]
            in_block = header == task_id
        stripped = line.strip()
        if in_block and stripped.lower().startswith("- status:"):
            status = stripped.split(":", 1)[1].strip().lower()
            if status not in _COMPLETABLE_STATUSES:
                raise ValueError(f"{task_id} status {status} is not completable")
            newline = "\n" if line.endswith("\n") else ""
            rewritten.append(f"- Status: completed{newline}")
            rewritten.append(f"- Completion: {note}{newline}")
            replaced = True
            continue
        rewritten.append(line)
    if not replaced:
        raise ValueError(f"{task_id} has no status field")
    return "".join(rewritten)


def _atomic_write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write(path: Path, text: str) -> None:
    persist_text(path, text, _atomic_write_once)


def _load_attempt_state(
    path: Path,
) -> tuple[
    dict[str, int], dict[str, int], dict[str, dict[str, Any]], dict[str, dict[str, Any]],
]:
    """Return failure counts, retry times, fixed points, and command evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}, {}, {}
    if not isinstance(payload, dict):
        return {}, {}, {}, {}
    attempts: dict[str, int] = {}
    retry_after: dict[str, int] = {}
    fixed_points: dict[str, dict[str, Any]] = {}
    command_evidence: dict[str, dict[str, Any]] = {}
    raw_retry = payload.get(_RETRY_AFTER_KEY)
    if isinstance(raw_retry, dict):
        for key, value in raw_retry.items():
            try:
                retry_after[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    raw_fixed = payload.get(_FIXED_POINT_KEY)
    if isinstance(raw_fixed, dict):
        for key, value in raw_fixed.items():
            if isinstance(value, dict):
                fixed_points[str(key)] = dict(value)
    raw_evidence = payload.get(_COMMAND_EVIDENCE_KEY)
    if isinstance(raw_evidence, dict):
        for key, value in raw_evidence.items():
            if isinstance(value, dict):
                command_evidence[str(key)] = dict(value)
    for key, value in payload.items():
        if key in {_RETRY_AFTER_KEY, _FIXED_POINT_KEY, _COMMAND_EVIDENCE_KEY}:
            continue
        try:
            attempts[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return attempts, retry_after, fixed_points, command_evidence


def _store_attempt_state(
    path: Path,
    attempts: Mapping[str, int],
    retry_after: Mapping[str, int],
    fixed_points: Mapping[str, Mapping[str, Any]] | None = None,
    command_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    payload: dict[str, Any] = {str(key): int(value) for key, value in attempts.items()}
    if retry_after:
        payload[_RETRY_AFTER_KEY] = {
            str(key): int(value) for key, value in retry_after.items()
        }
    if fixed_points:
        payload[_FIXED_POINT_KEY] = {
            str(key): dict(value) for key, value in fixed_points.items()
        }
    if command_evidence:
        payload[_COMMAND_EVIDENCE_KEY] = {
            str(key): dict(value) for key, value in command_evidence.items()
        }
    _atomic_write(path, json.dumps(payload, indent=2) + "\n")


def _unresolved_execution_failure(
    task: Mapping[str, Any], previous_evidence: Mapping[str, Any] | None = None,
) -> bool:
    """Keep task health separate from the current contract's retry budget.

    Historical records are newest last. Only a later semantic validation can
    resolve an earlier failure; changing source bytes or starting a PID cannot.
    This also migrates journals written before the explicit health flag.
    """
    current = task.get("execution_progress") or {}
    history = task.get("execution_progress_history") or []
    records = [current, *reversed(history[-8:] if isinstance(history, list) else [])]
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if record.get("state") == "validated":
            return False
        if (record.get("unresolved_prior_failure") is True
                or int(record.get("semantic_failures") or 0) > 0):
            return True
    evidence = previous_evidence or {}
    return type(evidence.get("exit_code")) is int and evidence["exit_code"] not in (0, _GATE_WITHHELD_EXIT)


def _begin_execution_progress(
    fixed_points: dict[str, dict[str, Any]], task_id: str, contract: str, *,
    now_ms: int, previous_evidence: Mapping[str, Any], previous_attempts: int,
) -> None:
    """Retain semantic failure history before an invocation can be interrupted."""
    task = fixed_points.setdefault(task_id, {})
    unresolved = _unresolved_execution_failure(task, previous_evidence)
    prior = task.get("execution_progress") or {}
    if prior.get("contract") != contract:
        if prior:
            history = list(task.get("execution_progress_history") or [])
            task["execution_progress_history"] = [*history[-7:], prior]
        qualified_legacy = (previous_evidence.get("retry_contract", previous_evidence.get("contract")) == contract
                            and type(previous_evidence.get("exit_code")) is int
                            and previous_evidence["exit_code"] not in (0, _GATE_WITHHELD_EXIT))
        prior = {"schema": "agent-supervisor/execution-progress@1", "contract": contract,
                 "first_started_at_ms": now_ms, "attempts_started": 0,
                 "semantic_failures": max(0, previous_attempts) if qualified_legacy else 0,
                 "abandoned_attempts": 0, "gate_refusals": 0}
        if qualified_legacy:
            prior["legacy_failures_qualified"] = True
    else:
        prior = dict(prior)
        if prior.get("state") in {"running", "produced"}:
            prior["abandoned_attempts"] = int(prior.get("abandoned_attempts") or 0) + 1
    prior.update(state="running", started_at_ms=now_ms,
                 attempts_started=int(prior.get("attempts_started") or 0) + 1,
                 unresolved_prior_failure=unresolved, completion_authority=False)
    if unresolved:
        prior["unresolved_prior_failure_reason"] = "prior_semantic_failure_not_validated"
    else:
        prior.pop("unresolved_prior_failure_reason", None)
    task["execution_progress"] = prior


def _finish_execution_progress(
    fixed_points: dict[str, dict[str, Any]], task_id: str, *, code: int, stage: str,
    contract: str, now_ms: int, validated: bool = False, resource_refusal: str = "",
) -> None:
    task = fixed_points.setdefault(task_id, {})
    progress = dict(task.get("execution_progress") or {})
    if not progress:
        return
    progress.update(contract=contract, last_exit_code=code, last_stage=stage,
                    finished_at_ms=now_ms, completion_authority=False)
    if validated:
        progress.update(state="validated", semantic_failures=0, abandoned_attempts=0,
                        gate_refusals=0, validated_at_ms=now_ms, unresolved_prior_failure=False)
        progress.pop("escalated", None)
        progress.pop("unresolved_prior_failure_reason", None)
    elif code == _GATE_WITHHELD_EXIT and resource_refusal:
        progress.update(state="resource_wait", resource_refusal=resource_refusal,
                        resource_refusals=int(progress.get("resource_refusals") or 0) + 1)
    elif code == _GATE_WITHHELD_EXIT:
        progress.update(state="gate_withheld", gate_refusals=int(progress.get("gate_refusals") or 0) + 1)
    elif code:
        progress.update(state="failed", semantic_failures=int(progress.get("semantic_failures") or 0) + 1,
                        unresolved_prior_failure=True,
                        unresolved_prior_failure_reason="prior_semantic_failure_not_validated")
        progress.setdefault("first_failure_at_ms", now_ms)
    else:
        progress["state"] = "produced"
    task["execution_progress"] = progress


def execution_progress_snapshot(path: Path, task_id: str) -> dict[str, Any]:
    """Read bounded, body-free invocation history; clocks never count as work."""
    _attempts, _retry, fixed_points, _evidence = _load_attempt_state(path)
    task = fixed_points.get(task_id) or {}
    record = dict(task.get("execution_progress") or {})
    if record:
        record["unresolved_prior_failure"] = _unresolved_execution_failure(task)
        if record["unresolved_prior_failure"]:
            record["unresolved_prior_failure_reason"] = "prior_semantic_failure_not_validated"
        else:
            record.pop("unresolved_prior_failure_reason", None)
    keys = ("schema", "contract", "state", "first_started_at_ms", "started_at_ms", "finished_at_ms",
            "attempts_started", "semantic_failures", "abandoned_attempts", "gate_refusals", "last_stage",
            "last_exit_code", "first_failure_at_ms", "validated_at_ms", "escalated",
            "unresolved_prior_failure", "unresolved_prior_failure_reason", "resource_refusal", "resource_refusals")
    result = {key: record[key] for key in keys if key in record}
    if "resource_refusals_total" in task:
        result["resource_refusals_total"] = task["resource_refusals_total"]
    try:
        recovery = semantic_recovery.summary(path, task_id)
        if recovery.get("spent"):
            result["semantic_recovery"] = recovery
    except semantic_recovery.SemanticRecoveryError:
        result["semantic_recovery"] = {"state": "blocked", "reason": "recovery_journal_invalid", "completion_authority": False}
    return result


def lane_execution_health(
    *, repo_root: Path, todo_path: Path, task_header_prefix: str, attempt_path: Path,
) -> dict[str, Any]:
    """Pending failures belong to the lane even while another task runs well.

    Keep this separate from the active task's execution_progress: its own new
    attempt has not inherited another task's failure count or retry budget.
    Historical completed tasks do not degrade the current pending frontier.
    """
    from .supervisor_watchdog import parse_todo_task_fields

    base = {"schema": "agent-supervisor/lane-execution-health@1", "observation_only": True,
            "work_complete": False, "completion_authority": False}
    try:
        fields_by_id = dict(parse_todo_task_fields(todo_path.read_text(encoding="utf-8"),
                                                 task_header_prefix=task_header_prefix))
        if attempt_path.exists():
            payload = json.loads(attempt_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("invalid attempt state")
        attempts, _retry, fixed, evidence = _load_attempt_state(attempt_path)
        failures = []
        for task_id, fields in fields_by_id.items():
            if str(fields.get("status") or "").strip().casefold() in {"completed", "complete", "done"}:
                continue
            record, proof = fixed.get(task_id) or {}, evidence.get(task_id) or {}
            progress = record.get("execution_progress") or {}
            unresolved = _unresolved_execution_failure(record, proof)
            unqualified = bool(attempts.get(task_id, 0) > 0 and not progress and not proof)
            if not unresolved and not unqualified:
                continue
            command = parse_deterministic_command(fields.get("deterministic_command") or fields.get("deterministic_repair") or "", repo_root=repo_root)
            validation = parse_deterministic_command(fields.get("validation") or "", repo_root=repo_root)
            contract = _command_contract(fields, command, validation, repo_root=repo_root) if command else ""
            qualified = bool(contract and progress.get("contract") == contract)
            count = int(progress.get("semantic_failures") or 0)
            failures.append({"task_id": task_id, "semantic_failures": count,
                             "current_contract_qualified": qualified,
                             "unqualified_prior_attempts": unqualified,
                             "exhausted": qualified and count >= semantic_failure_limit(fields)})
        return {**base, "health": "degraded" if failures else "operational",
                "pending_failure_count": len(failures), "pending_failure_task_ids": [item["task_id"] for item in failures[:32]],
                "exhausted_failure_count": sum(item["exhausted"] for item in failures),
                "pending_failures": failures[:32], "unresolved_prior_failure": bool(failures)}
    except (OSError, ValueError, TypeError, AttributeError):
        return {**base, "health": "degraded", "reason": "lane_failure_history_unavailable",
                "pending_failure_count": 0, "pending_failure_task_ids": [], "pending_failures": [],
                "exhausted_failure_count": 0, "unresolved_prior_failure": False}


def _load_attempts(path: Path) -> dict[str, int]:
    attempts, _retry_after, _fixed, _evidence = _load_attempt_state(path)
    return attempts


def _refill_module(repo_root: Path) -> Any | None:
    """Load the packaged admission rules independently of workspace siblings."""

    del repo_root
    from . import board_gap_refill

    return board_gap_refill


def _artifact_text(fields: Mapping[str, str], *, repo_root: Path) -> tuple[str, str]:
    raw = str(fields.get("predicted_files") or "")
    digest = hashlib.sha256()
    first = ""
    for item in raw.split(","):
        relative = item.strip()
        if not relative:
            continue
        path = repo_root / relative
        digest.update(relative.encode("utf-8"))
        if path.is_file():
            body = path.read_bytes()[:1_000_000]
            digest.update(body)
            if not first:
                first = body.decode("utf-8", errors="replace")
        else:
            digest.update(b"<missing>")
    return digest.hexdigest(), first


def _file_digest(path: Path) -> str | None:
    """Hash complete file contents; timestamps never establish production."""

    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1_048_576), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _declared_evidence(
    fields: Mapping[str, str], *, repo_root: Path, excluded: Sequence[Path],
) -> dict[str, str | None]:
    """Snapshot declared files inside the repo, excluding the lane's own state."""

    root = repo_root.resolve()
    ignored = {path.resolve() for path in excluded}
    evidence: dict[str, str | None] = {}
    for field in ("outputs", "predicted_files"):
        for value in str(fields.get(field) or "").split(","):
            if not value.strip():
                continue
            path = root / value.strip()
            try:
                path = path.resolve()
                relative = path.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            if path in ignored or path.suffix.casefold() == ".log":
                continue
            evidence[relative] = _file_digest(path)
    return evidence


def _runtime_contract_code(code: CodeType) -> CodeType:
    """Ignore source locations when hashing the loaded runner implementation."""

    return code.replace(
        co_consts=tuple(_runtime_contract_code(value) if isinstance(value, CodeType) else value
                        for value in code.co_consts),
        co_filename="", co_firstlineno=0, co_linetable=b"",
    )


def _execution_input_contract(raw: str, *, repo_root: Path) -> dict[str, dict[str, str | None]]:
    """Fingerprint explicitly declared read-only inputs without expanding edits.

    Missing in-repository files remain bound as None. Invalid/outside paths
    have deterministic markers and never reach the content reader, including
    symlinks whose resolved target escapes this repository.
    """

    root = repo_root.resolve()
    inputs: dict[str, dict[str, str | None]] = {}
    for name in (part.strip() for part in raw.split(",")):
        if not name:
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            inputs[name] = {"sha256": None, "invalid": "not_repo_relative"}
            continue
        try:
            resolved = (root / relative).resolve()
            canonical = resolved.relative_to(root).as_posix()
            if resolved.exists() and not resolved.is_file():
                inputs[name] = {"sha256": None, "invalid": "not_a_file"}
                continue
        except (OSError, ValueError, RuntimeError):
            inputs[name] = {"sha256": None, "invalid": "unresolvable_or_outside_repo"}
            continue
        inputs[name] = {"path": canonical, "sha256": _file_digest(resolved)}
    return inputs


def _command_contract(
    fields: Mapping[str, str], command: Sequence[str], validation: Sequence[str] | None,
    *, repo_root: Path,
) -> str:
    """Bind evidence to the declared contract and its directly named scripts."""

    names = (
        "deterministic_command", "deterministic_repair", "validation", "depends_on",
        "outputs", "predicted_files", "acceptance", "conflict_policy", "satisfies",
        "resource_class", "implementation_timeout_seconds",
    )
    scripts: dict[str, str | None] = {}
    root = repo_root.resolve()
    suffixes = {".py", ".js", ".cjs", ".mjs", ".ts", ".tsx", ".jsx", ".sh"}
    for argv in (command, validation or []):
        for index, arg in enumerate(argv):
            if not arg or arg.startswith("-") or (index and Path(arg).suffix.casefold() not in suffixes):
                continue
            try:
                path = (root / arg).resolve()
                relative = path.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            # A bare interpreter name is resolved on PATH, not under the repo.
            if index == 0 and not path.is_file():
                continue
            scripts[relative] = _file_digest(path)
    payload = {"fields": {name: str(fields.get(name) or "") for name in names},
               "command": list(command), "validation": list(validation or []), "scripts": scripts,
               "declared_sources": {
                   name: digest for name, digest in _declared_evidence(fields, repo_root=root, excluded=[]).items()
                   if _is_source_path(Path(name)) or semantic_recovery.is_authored_scenario_source(Path(name))
               }}
    if str(fields.get("retry_on_changed_evidence") or "").strip():
        # Preserve every existing lane's contract and attempt ledger.
        payload["retry_on_changed_evidence"] = str(fields["retry_on_changed_evidence"])
    if str(fields.get("semantic_failure_limit") or "").strip():
        payload["semantic_failure_limit"] = semantic_failure_limit(fields)
    execution_inputs = str(fields.get("execution_inputs") or "").strip()
    if execution_inputs:
        # Absent/blank declarations must retain legacy fingerprints and retry
        # budgets. These read-only bindings are never writable predicted files.
        payload["execution_inputs"] = _execution_input_contract(execution_inputs, repo_root=root)
    if any(Path(argument).name == "run-vdg-all-families.mjs" for argument in command):
        recovery = recovery_contract(command, repo_root=root)
        if recovery is not None:
            payload["pre_command_recovery"] = recovery
    for argv in (command, validation or []):
        if not argv or Path(argv[0]).name != "node":
            continue
        for index, argument in enumerate(argv[1:], 1):
            specifier = argv[index + 1] if argument == "--import" and index + 1 < len(argv) else (
                argument.partition("=")[2] if argument.startswith("--import=") else "")
            if specifier and not specifier.startswith((".", "/", "\\")) and ":" not in specifier:
                # Bind attempts to the actual loaded normalizer. A repaired
                # runner gets one fresh contract budget after a restart.
                implementation = (
                    *(_runtime_contract_code(function.__code__) for function in (
                        _normalize_node_preloads, _resolve_node_preload, _package_name, _declared_package_root,
                    )),
                    _NODE_PRELOAD_RESOLVE_SECONDS,
                )
                payload["node_preload_normalization"] = hashlib.sha256(marshal.dumps(implementation)).hexdigest()
                break
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _produced_evidence_matches(
    record: Mapping[str, Any], fields: Mapping[str, str], command: Sequence[str],
    validation: Sequence[str] | None, *, repo_root: Path, excluded: Sequence[Path],
) -> bool:
    """Only reuse changed evidence from the last invocation of this contract."""

    before, after = record.get("before"), record.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict) or not after:
        return False
    if not any(digest is not None and digest != before.get(path) for path, digest in after.items()):
        return False
    return bool(
        record.get("contract") == _command_contract(fields, command, validation, repo_root=repo_root)
        and after == _declared_evidence(fields, repo_root=repo_root, excluded=excluded)
    )


def _admit_validation_gap(
    todo_text: str,
    *,
    repo_root: Path,
    task_header_prefix: str,
    task_id: str,
    fields: Mapping[str, str],
    artifact_text: str,
) -> dict[str, Any]:
    module = _refill_module(repo_root)
    if module is None:
        return {
            "admitted": False,
            "reason": "refill_module_missing",
            "todo_text": todo_text,
            "work_complete": False,
            "completion_authority": False,
        }
    return module.admit_validation_gap(
        todo_text,
        task_header_prefix=task_header_prefix,
        task_id=task_id,
        fields=fields,
        artifact_text=artifact_text,
    )


def _attempt_backoff_ids(
    skipped: Sequence[str],
    retry_after: Mapping[str, int],
    *,
    now_ms: int,
) -> list[str]:
    """Task ids whose only skip is a retry that is still in the future.

    Any other skip, including a blank command, is not a backoff. An empty
    result means the caller should keep the unavailable block.
    """

    if not skipped:
        return []
    waiting: list[str] = []
    for item in skipped:
        text = str(item)
        task_id, separator, reason = text.partition(":")
        if not separator or reason != "attempts_exhausted" or not task_id:
            return []
        try:
            due = int(retry_after.get(task_id, 0) or 0)
        except (TypeError, ValueError):
            return []
        if due <= now_ms:
            return []
        if task_id not in waiting:
            waiting.append(task_id)
    return waiting


def _task_timeout(fields: Mapping[str, str]) -> float:
    raw = fields.get("implementation_timeout_seconds") or ""
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        timeout = 1800.0
    resource = str(fields.get("resource_class") or "")
    cap = 50400.0 if resource == "cpu-browser" else 1800.0
    return max(1.0, min(timeout, cap))


def _validation_timeout(fields: Mapping[str, str]) -> float:
    """Honor an explicitly registered budget; preserve the legacy 300s default."""
    try:
        declared = float(fields.get("implementation_timeout_seconds") or "")
        explicit = 0 < declared < float("inf")
    except (TypeError, ValueError):
        explicit = False
    return min(_task_timeout(fields), float("inf") if explicit else 300.0)


def _dependency_validation_timeout(fields: Mapping[str, str]) -> float:
    """Honor an explicit bounded validator budget; keep legacy quick defaults."""
    try:
        declared = float(fields.get("implementation_timeout_seconds") or "")
        explicit = 0 < declared < float("inf")
    except (TypeError, ValueError):
        explicit = False
    return min(_task_timeout(fields), float("inf") if explicit else 30.0)


def _stop_command_group(process: subprocess.Popen[Any]) -> None:
    """Stop the owned session and reap its leader, including on early leader exit."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=_COMMAND_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # A cooperative leader can exit while a browser descendant ignores
        # SIGTERM. Its exit must not skip escalation for the remaining group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=_COMMAND_STOP_GRACE_SECONDS)


def _resolve_node_preload(
    executable: str, specifier: str, *, cwd: Path, repo_root: Path, timeout: float,
    allow_external: bool = False,
) -> str | None:
    """Resolve locally without importing the package or invoking an installer."""

    try:
        result = subprocess.run(
            [executable, "--input-type=module", "--eval",
             "process.stdout.write(import.meta.resolve(process.argv[1]))", specifier],
            cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, timeout=timeout, check=False,
        )
        if result.returncode != 0:
            return None
        resolved = result.stdout.strip()
        parsed = urlsplit(resolved)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            return None
        target = Path(unquote(parsed.path)).resolve()
        if not allow_external:
            target.relative_to(repo_root.resolve())
        if not target.is_file():
            return None
        return target.as_uri()
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _normalize_node_preloads(
    argv: Sequence[str], *, repo_root: Path, cwd: Path, timeout: float,
) -> list[str]:
    """Use a script's declared package for a bare preload unavailable in cwd.

    A nested project can declare and install tsx without making it resolvable
    from the workspace root. Only preload tokens change; cwd and every script
    or test argument retain their original meaning.
    """

    normalized = list(argv)
    if not normalized or Path(normalized[0]).name != "node":
        return normalized
    imports: list[tuple[int, str, bool]] = []
    files: list[str] = []
    value_options = {"-e", "--eval", "-p", "--print", "-r", "--require", "--loader", "--experimental-loader",
                     "--test-name-pattern", "--test-skip-pattern", "--test-reporter", "--test-reporter-destination",
                     "--conditions", "-C", "--env-file", "--env-file-if-exists"}
    index = 1
    options = True
    test_mode = False
    while index < len(normalized):
        argument = normalized[index]
        if argument == "--":
            break
        if options and (argument == "--import" or argument.startswith("--import=")):
            separate = argument == "--import"
            if separate and index + 1 >= len(normalized):
                return normalized
            specifier = normalized[index + 1] if separate else argument.partition("=")[2]
            if specifier and not specifier.startswith((".", "/", "\\")) and ":" not in specifier:
                imports.append((index + 1 if separate else index, specifier, separate))
            index += 2 if separate else 1
            continue
        if options and argument in value_options:
            index += 2
            continue
        if options and argument == "--test":
            test_mode = True
        if argument and not argument.startswith("-") and Path(argument).suffix in {".js", ".cjs", ".mjs", ".ts", ".tsx", ".jsx"}:
            candidate = Path(argument)
            if not candidate.is_absolute():
                candidate = cwd / candidate
            try:
                candidate = candidate.resolve()
                candidate.relative_to(repo_root.resolve())
            except (OSError, ValueError):
                index += 1
                continue
            if candidate.is_file():
                files.append(str(candidate))
                options = False
                if not test_mode:
                    break
        index += 1
    if not imports or not files:
        return normalized
    deadline = time.monotonic() + max(0.0, min(float(timeout), _NODE_PRELOAD_RESOLVE_SECONDS))
    for index, specifier, separate in imports:
        package_name = _package_name(specifier)
        roots = {
            package_root for filename in files
            if (package_root := _declared_package_root(
                [normalized[0], filename], repo_root=repo_root, package_name=package_name,
            )) is not None
        }
        if len(roots) != 1:
            continue
        package_root = roots.pop()
        if package_root.resolve() == cwd.resolve():
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if _resolve_node_preload(normalized[0], specifier, cwd=cwd, repo_root=repo_root,
                                 timeout=remaining, allow_external=True):
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        resolved = _resolve_node_preload(
            normalized[0], specifier, cwd=package_root, repo_root=repo_root, timeout=remaining,
        )
        if resolved:
            normalized[index] = resolved if separate else f"--import={resolved}"
    return normalized


def _run_argv(
    argv: Sequence[str],
    *,
    repo_root: Path,
    timeout: float,
    log_path: Path,
    heartbeat: Callable[[], None],
    cwd: Path | None = None,
    cleanup_descendants_on_success: bool = False,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run a command, containing failures and optionally its successful children.

    Commands may intentionally launch services, so successful descendants stay
    alive by default. Browser tasks opt into cleanup of their temporary servers.
    Errors, cancellation, and timeouts always clean the owned command tree.
    """

    workdir = cwd or repo_root
    command = _normalize_node_preloads(argv, repo_root=repo_root, cwd=workdir, timeout=timeout)
    with open_command_log(log_path, f"$ {' '.join(command)}  # cwd={workdir}\n") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(workdir),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            **({"env": dict(env)} if env is not None else {}),
        )
        descendants = CommandProcessTree(process.pid)
        deadline = time.monotonic() + timeout
        completed = False
        try:
            while True:
                descendants.snapshot()
                heartbeat()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # A SIGTERM handler exiting zero is still a timed-out task.
                    return _COMMAND_TIMEOUT_EXIT
                try:
                    result = int(process.wait(timeout=min(_COMMAND_POLL_SECONDS, remaining)))
                except subprocess.TimeoutExpired:
                    continue
                # Stop signals or the lane deadline may arrive while wait()
                # returns a successful exit. Check before granting task success.
                heartbeat()
                if time.monotonic() >= deadline:
                    return _COMMAND_TIMEOUT_EXIT
                completed = result == 0
                return result
        finally:
            cleanup_required = not completed or cleanup_descendants_on_success
            try:
                if cleanup_required:
                    descendants.snapshot()
                    descendants.send_signal(signal.SIGTERM)
                    _stop_command_group(process)
            finally:
                # Observed descendants may detach into another session, or
                # outlive even a successful leader. Use their identity handles.
                try:
                    if cleanup_required:
                        descendants.send_signal(signal.SIGKILL)
                finally:
                    descendants.close()


_MISSING_MODULE = re.compile(r"Cannot find module '([^']+)'")
_MISSING_EXPORT = re.compile(
    r"The requested module '([^']+)' does not provide an export named '([^']+)'"
)
_SOURCE_SUFFIXES = (".ts", ".tsx", ".js", ".mjs")


def _package_name(module_id: str) -> str:
    text = str(module_id or "").strip()
    if not text or text.startswith(".") or text.startswith("node:"):
        return ""
    if text.startswith("@"):
        scope, _sep, rest = text[1:].partition("/")
        name, _sep, _sub = rest.partition("/")
        return f"@{scope}/{name}" if scope and name else ""
    return text.split("/", 1)[0]


def _declared_package_root(argv: Sequence[str], *, repo_root: Path, package_name: str) -> Path | None:
    if not package_name:
        return None
    root = repo_root.resolve()
    for arg in argv[1:]:
        candidate = Path(arg)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if not candidate.is_file():
            continue
        current = candidate.resolve().parent
        while True:
            try:
                current.relative_to(root)
            except ValueError:
                break
            manifest = current / "package.json"
            if manifest.is_file():
                try:
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
                declared: dict[str, Any] = {}
                for key in ("dependencies", "devDependencies", "optionalDependencies"):
                    raw = payload.get(key) or {}
                    if isinstance(raw, dict):
                        declared.update(raw)
                if package_name in declared:
                    return current
                return None
            if current == root:
                break
            parent = current.parent
            if parent == current:
                break
            current = parent
    return None


def _repair_marker(attempt_path: Path, package_root: Path) -> Path:
    safe = "".join(char if char.isalnum() else "-" for char in str(package_root))[:140]
    return attempt_path.with_name(f"module-repair-{safe or 'package'}.json")


def missing_module_repair_pending(
    argv: Sequence[str],
    *,
    repo_root: Path,
    log_path: Path,
    attempt_path: Path,
) -> bool:
    """True when the log shows one declared Node module and no install was tried."""

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    found = _MISSING_MODULE.findall(log_text)
    package_name = _package_name(found[-1]) if found else ""
    package_root = _declared_package_root(argv, repo_root=repo_root, package_name=package_name)
    if package_root is None:
        return False
    return not _repair_marker(attempt_path, package_root).exists()


def repair_missing_module(
    argv: Sequence[str],
    *,
    repo_root: Path,
    log_path: Path,
    attempt_path: Path,
    timeout: float,
    heartbeat: Callable[..., None],
) -> bool:
    """Install a missing declared package once. Never uses a shell or a provider."""

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    found = _MISSING_MODULE.findall(log_text)
    package_name = _package_name(found[-1]) if found else ""
    package_root = _declared_package_root(argv, repo_root=repo_root, package_name=package_name)
    if package_root is None:
        return False
    marker = _repair_marker(attempt_path, package_root)
    if marker.exists():
        return False
    install = (
        ["npm", "ci", "--ignore-scripts"]
        if (package_root / "package-lock.json").is_file()
        else ["npm", "install", "--ignore-scripts"]
    )
    heartbeat(state="starting", phase="module_repair")
    exit_code = _run_argv(
        install,
        repo_root=repo_root,
        cwd=package_root,
        timeout=max(1.0, min(timeout, 1800.0)),
        log_path=log_path,
        heartbeat=lambda: heartbeat(state="healthy", phase="module_repair"),
    )
    _atomic_write(
        marker,
        json.dumps(
            {
                "package_root": str(package_root),
                "package_name": package_name,
                "ok": exit_code == 0,
                "work_complete": False,
                "completion_authority": False,
            },
            indent=2,
        )
        + "\n",
    )
    return exit_code == 0


def _command_source_files(argv: Sequence[str], *, repo_root: Path) -> list[Path]:
    root = repo_root.resolve()
    found: list[Path] = []
    for arg in argv[1:]:
        if not arg or arg.startswith("-") or Path(arg).suffix not in _SOURCE_SUFFIXES:
            continue
        candidate = Path(arg)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except ValueError:
            continue
        found.append(resolved)
    return found


def _resolve_requested_module(spec: str, anchor: Path, *, repo_root: Path) -> Path | None:
    text = str(spec or "").strip()
    if not text or text.startswith("node:"):
        return None
    root = repo_root.resolve()
    if text.startswith("/"):
        options = [Path(text)]
    elif text.startswith("."):
        raw = anchor.parent / text
        if raw.suffix in _SOURCE_SUFFIXES:
            options = [raw]
        elif raw.suffix:
            return None
        else:
            options = [raw.with_suffix(suffix) for suffix in _SOURCE_SUFFIXES]
    else:
        return None
    fallback: Path | None = None
    for option in options:
        try:
            resolved = option.resolve()
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.suffix not in _SOURCE_SUFFIXES:
            continue
        if resolved.is_file():
            return resolved
        if fallback is None:
            fallback = resolved
    return fallback


def _missing_export_target(argv: Sequence[str], spec: str, *, repo_root: Path) -> Path | None:
    fallback: Path | None = None
    for anchor in _command_source_files(argv, repo_root=repo_root):
        resolved = _resolve_requested_module(spec, anchor, repo_root=repo_root)
        if resolved is None:
            continue
        if resolved.is_file():
            return resolved
        if fallback is None:
            fallback = resolved
    return fallback


def _export_repair_marker(attempt_path: Path, source: Path, export_name: str) -> Path:
    key = f"{source}:{export_name}"
    safe = "".join(char if char.isalnum() else "-" for char in key)[:140]
    return attempt_path.with_name(f"export-repair-{safe or 'export'}.json")


def _newest_missing_export(log_path: Path, log_start: int) -> tuple[str, str]:
    matches = list(_MISSING_EXPORT.finditer(_log_slice(log_path, log_start)))
    if not matches:
        return "", ""
    found = matches[-1]
    return str(found.group(1) or ""), str(found.group(2) or "")


def repair_missing_named_export(
    argv: Sequence[str],
    *,
    repo_root: Path,
    log_path: Path,
    attempt_path: Path,
    timeout: float,
    heartbeat: Callable[..., None],
    allow_bounded_implement: bool = False,
    log_start: int = 0,
) -> bool:
    """Add one missing runtime export. The same file and name are not tried twice.

    The slice starts at the offset taken before this command, so a previous
    task's log line cannot select the export. Grok is not spawned unless
    bounded implement is already allowed, and it may edit only the resolved file.
    """

    if not allow_bounded_implement:
        return False
    spec, export_name = _newest_missing_export(log_path, log_start)
    if not spec or not export_name:
        return False
    source = _missing_export_target(argv, spec, repo_root=repo_root)
    if source is None:
        return False
    marker = _export_repair_marker(attempt_path, source, export_name)
    if marker.exists():
        return False
    grok = shutil.which("grok")
    if not grok:
        return False
    try:
        relative = source.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return False
    prompt_path = marker.with_suffix(".prompt.txt")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(
            [
                f"Add a runtime export of {export_name} used by the importer.",
                "Keep every existing export.",
                "Do not edit other files.",
                "No shell.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    grok_argv = [
        grok,
        "--cwd",
        str(repo_root.resolve()),
        "--prompt-file",
        str(prompt_path),
        "--no-subagents",
        "--no-plan",
        "--disable-web-search",
        "--max-turns",
        "16",
        "--permission-mode",
        "dontAsk",
        "--output-format",
        "plain",
        "--disallowed-tools",
        "spawn_subagent,run_terminal_command,workflow,scheduler_create",
        "--allow",
        f"Edit({relative})",
    ]
    heartbeat(state="starting", phase="export_repair")
    exit_code = _run_argv(
        grok_argv,
        repo_root=repo_root,
        timeout=max(1.0, min(float(timeout), 1800.0)),
        log_path=log_path,
        heartbeat=lambda: heartbeat(state="healthy", phase="export_repair"),
    )
    try:
        present = export_name in source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        present = False
    ok = exit_code == 0 and present
    _atomic_write(
        marker,
        json.dumps(
            {
                "path": relative,
                "export_name": export_name,
                "ok": ok,
                "work_complete": False,
                "completion_authority": False,
            },
            indent=2,
        )
        + "\n",
    )
    return ok


def _repo_file_args(argv: Sequence[str], *, repo_root: Path) -> list[Path]:
    root = repo_root.resolve()
    found: list[Path] = []
    for arg in argv[1:]:
        if not arg or arg.startswith("-"):
            continue
        candidate = Path(arg)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.suffix or "/" in arg:
            found.append(resolved)
    return found


def _predicted_relative_paths(raw: str, *, repo_root: Path) -> list[str]:
    root = repo_root.resolve()
    paths: list[str] = []
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text or text.startswith("/") or ".." in Path(text).parts:
            continue
        try:
            (root / text).resolve().relative_to(root)
        except ValueError:
            continue
        if text not in paths:
            paths.append(text)
    return paths


def _bounded_marker(attempt_path: Path, task_id: str) -> Path:
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in task_id)[:80]
    return attempt_path.with_name(f"bounded-implement-{safe or 'task'}.json")


_BOUNDED_ATTEMPT_LIMIT = 2


def _bounded_attempts(marker: Path) -> int:
    """How many bounded implement attempts are already spent.

    A successful marker is closed. A missing or unreadable marker does not
    grant extra attempts. A failed marker records its attempt count so the
    lane can try once more, then stop.
    """

    if not marker.is_file():
        return 0
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _BOUNDED_ATTEMPT_LIMIT
    if not isinstance(payload, dict) or payload.get("ok") is True:
        return _BOUNDED_ATTEMPT_LIMIT
    if "attempts" not in payload:
        return _BOUNDED_ATTEMPT_LIMIT
    try:
        return max(0, int(payload.get("attempts")))
    except (TypeError, ValueError):
        return _BOUNDED_ATTEMPT_LIMIT


def _bounded_marker_payload(marker: Path) -> dict[str, Any]:
    if not marker.is_file():
        return {}
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _bounded_max_turns(spent: int, escalation: int) -> int:
    """16 then 32. One turn-exhaustion escalation uses 64 then 128."""

    if int(escalation) <= 0:
        return 16 if int(spent) <= 0 else 32
    return 64 if int(spent) <= 0 else 128


def _log_slice(log_path: Path, start: int = 0) -> str:
    try:
        data = log_path.read_bytes()
    except OSError:
        return ""
    return data[max(0, int(start)) :].decode("utf-8", errors="replace")


def _slice_turn_exhausted(log_path: Path, start: int) -> bool:
    return "max turns reached" in _log_slice(log_path, start).casefold()


def _legacy_turn_exhausted(log_path: Path, task_id: str) -> bool:
    """Markers written before turn_exhausted existed still name the task in the log."""

    folded = _log_slice(log_path, 0).casefold()
    phrase = "max turns reached"
    needle = str(task_id or "").casefold()
    if phrase not in folded or not needle:
        return False
    idx = folded.rfind(needle)
    return idx >= 0 and phrase in folded[idx:]


def _write_bounded_marker(
    marker: Path,
    *,
    task_id: str,
    attempts: int,
    ok: bool,
    escalation: int,
    turn_exhausted: bool,
) -> None:
    _atomic_write(
        marker,
        json.dumps(
            {
                "task_id": task_id,
                "attempts": int(attempts),
                "ok": bool(ok),
                "escalation": int(escalation),
                "turn_exhausted": bool(turn_exhausted),
                "work_complete": False,
                "completion_authority": False,
            },
            indent=2,
        )
        + "\n",
    )


def prepare_bounded_implement(
    task_id: str,
    fields: Mapping[str, str],
    *,
    repo_root: Path,
    attempt_path: Path,
    max_turns: int = 16,
) -> list[str] | None:
    """Build one edit-scoped grok argv. A todo command cannot ask for this.

    The argv is returned only when a deterministic validation names a file
    inside the repo and the task lists predicted files. The process may edit
    those files and cannot start a shell, a subagent, or another supervisor.
    """

    validation = parse_deterministic_command(
        fields.get("validation") or "",
        repo_root=repo_root,
    )
    predicted = _predicted_relative_paths(
        fields.get("predicted_files") or "",
        repo_root=repo_root,
    )
    if not validation or not _repo_file_args(validation, repo_root=repo_root) or not predicted:
        return None
    grok = shutil.which("grok")
    if not grok:
        return None
    prompt_path = _bounded_marker(attempt_path, task_id).with_suffix(".prompt.txt")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        "\n".join(
            [
                f"Implement task {task_id} only.",
                "Edit only the allowed predicted files.",
                "Do not edit the task board, supervisor state, or git metadata.",
                "Do not start a supervisor, a provider CLI, or a subagent.",
                "Prefer a deterministic implementation.",
                "The first tool call must be Edit on a predicted file.",
                "Read at most one existing file, and only to copy an exact Edit string.",
                "A run that reaches max turns without an Edit is a failed attempt.",
                f"Acceptance: {fields.get('acceptance') or ''}",
                f"Conflict policy: {fields.get('conflict_policy') or ''}",
                f"Outputs: {fields.get('outputs') or ''}",
                f"Validation run after you exit: {fields.get('validation') or ''}",
                "Predicted files:",
                *predicted,
            ]
        )
        + (
            f"\nThe previous attempt exhausted its turn budget. This attempt allows {int(max_turns)} turns. Write the file now.\n"
            if int(max_turns) >= 64
            else "\n"
        ),
        encoding="utf-8",
    )
    argv = [
        grok,
        "--cwd",
        str(repo_root.resolve()),
        "--prompt-file",
        str(prompt_path),
        "--no-subagents",
        "--no-plan",
        "--disable-web-search",
        "--max-turns",
        str(max(1, int(max_turns))),
        "--permission-mode",
        "dontAsk",
        "--output-format",
        "plain",
        "--disallowed-tools",
        "spawn_subagent,run_terminal_command,workflow,scheduler_create",
    ]
    for relative in predicted:
        argv.extend(["--allow", f"Edit({relative})"])
    return argv


def _bounded_outputs_material(validation: str, *, repo_root: Path) -> bool:
    argv = parse_deterministic_command(validation, repo_root=repo_root)
    if not argv:
        return False
    files = _repo_file_args(argv, repo_root=repo_root)
    return bool(files) and all(path.is_file() and path.stat().st_size >= 128 for path in files)


def _recovery_source_snapshot(fields: Mapping[str, str], *, repo_root: Path) -> dict[str, str | None]:
    """Production source only: receipts, feedback and runtime churn cannot rearm."""
    names = [part.strip() for key in ("predicted_files", "execution_inputs")
             for part in str(fields.get(key) or "").split(",") if part.strip()]
    selected = []
    for name in names:
        item = Path(name)
        if ({"data", "test-results", "evidence", "logs", "model-inputs", "model-invocations"}.intersection(item.parts)
                or not (_is_source_path(item) or semantic_recovery.is_authored_scenario_source(item))):
            continue
        selected.append(name)
    contract = _execution_input_contract(",".join(selected), repo_root=repo_root)
    if any(value.get("invalid") for value in contract.values()):
        raise semantic_recovery.SemanticRecoveryError("recovery_source_path_invalid")
    return {**{name: value["sha256"] for name, value in contract.items()},
            **semantic_recovery.fixture_inputs(repo_root, fields)}


def semantic_failure_limit(fields: Mapping[str, str]) -> int:
    """A canary may use a smaller real failure budget; ordinary tasks keep six."""
    value = fields.get("semantic_failure_limit")
    raw = "" if value is None else str(value).strip()
    if not raw:
        return 6
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError("semantic_failure_limit_invalid") from exc
    if not 1 <= limit <= 6:
        raise ValueError("semantic_failure_limit_invalid")
    return limit


def _recovery_protected_scripts(command: Sequence[str], validation: Sequence[str] | None,
                                recovery: Sequence[str], *, repo_root: Path) -> dict[str, str | None]:
    result = {}
    for argv in (command, validation or [], recovery):
        for name in argv:
            if Path(name).suffix.lower() not in {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}:
                continue
            relative = Path(name)
            if relative.is_absolute():
                try:
                    relative = relative.relative_to(repo_root)
                except ValueError:
                    continue
            binding = _execution_input_contract(str(relative), repo_root=repo_root)
            result[str(relative)] = binding[str(relative)].get("sha256")
    return result


def _recovery_storage_wait(task_id: str, *, repo_root: Path, attempt_path: Path,
                           log_path: Path) -> dict[str, Any] | None:
    """Fresh durable admission before reserving recovery or starting its work.

    Do not write a pressure refusal to the full filesystem or refund an earlier
    admission. Existing journals and all repair/phase budgets remain untouched.
    """
    capacity = check_storage_capacity((repo_root, attempt_path.parent, log_path.parent))
    if capacity["ok"]:
        return None
    return {"state": "degraded", "phase": "resource_retry_wait", "active_task_id": "",
            "resource_task_id": task_id, "resource_code": "storage_pressure", "storage_pressure": True,
            "terminal_reason": "", "backpressure_reasons": [f"{task_id}:storage_pressure"],
            "storage_reason": capacity["reason"],
            "next_retry_at_ms": int(time.time() * 1000) + _RESOURCE_RETRY_BASE_MS,
            "busy": False, "work_complete": False, "completion_authority": False}


def _run_semantic_recovery(
    task_id: str, fields: Mapping[str, str], command: Sequence[str], validation: Sequence[str] | None, *,
    repo_root: Path, todo_path: Path, attempt_path: Path, log_path: Path,
    progress: Mapping[str, Any], heartbeat: Callable[..., None], hold_result: Callable[[], dict[str, Any] | None],
) -> dict[str, Any] | None:
    """One finite, task-declared repair; never reset attempts or complete work."""
    policy = semantic_recovery.policy(fields, root=repo_root, task_id=task_id,
                                      parse_command=parse_deterministic_command, task_timeout=_task_timeout(fields))
    if policy is None:
        return None
    held = hold_result()
    if held:
        return held
    storage_wait = _recovery_storage_wait(task_id, repo_root=repo_root, attempt_path=attempt_path,
                                          log_path=log_path)
    if storage_wait:
        return storage_wait
    contract = _command_contract(fields, command, validation, repo_root=repo_root)
    ledger = semantic_recovery.Journal(semantic_recovery.journal_path(attempt_path), _atomic_write)
    ledger.reconcile(task_id, contract, int(time.time() * 1000))
    before = _recovery_source_snapshot(fields, repo_root=repo_root)
    if not before:
        raise semantic_recovery.SemanticRecoveryError("recovery_has_no_declared_production_source")
    fingerprint = semantic_recovery.digest({"task_id": task_id, "contract": contract, "source": before,
                                             "policy": policy["policy_fingerprint"], "feedback": policy["qualification_inputs"]})
    spent = ledger.summary(task_id)["spent"]
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", task_id)[:128]
    packet = attempt_path.with_name(f"semantic-recovery-{safe_id}-{spent + 1}.context.json")
    try:
        packet_relative = packet.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise semantic_recovery.SemanticRecoveryError("recovery_runtime_outside_repo") from exc
    row = ledger.admit(task_id, contract=contract, fingerprint=fingerprint, limit=policy["limit"],
                       context_path=packet_relative, now_ms=int(time.time() * 1000))
    row["timeout_seconds"] = policy["timeout"]
    row["recovery_limit"] = policy["limit"]
    row["source_before"] = before
    ledger.save()
    context = {"schema": "agent-supervisor/semantic-recovery-context@1", "task_id": task_id,
               "producer_contract": contract, "qualification_fingerprint": fingerprint,
               "admission_index": spent + 1, "recovery_limit": policy["limit"], "feedback_inputs": policy["inputs"],
               "source_before": before, "failure_summary": {key: progress[key] for key in (
                   "semantic_failures", "attempts_started", "last_stage", "last_exit_code") if key in progress},
               "requires_inherited_lease_validation": True, "observation_only": True,
               "work_complete": False, "completion_authority": False}
    _atomic_write(packet, json.dumps(context, sort_keys=True, indent=2) + "\n")
    protected = _recovery_protected_scripts(command, validation, policy["argv"], repo_root=repo_root)
    board_before = _file_digest(todo_path)
    env = dict(os.environ)
    env[semantic_recovery.CONTEXT_ENV] = str(packet.resolve())

    class RecoveryHeld(Exception):
        pass

    def beat() -> None:
        if hold_result():
            raise RecoveryHeld
        heartbeat(active_task_id=task_id, state="degraded", phase="semantic_recovery")

    code = 1
    error = None
    held_after = None
    try:
        code = _run_argv(policy["argv"], repo_root=repo_root, timeout=policy["timeout"], log_path=log_path,
                         heartbeat=beat, cleanup_descendants_on_success=True, env=env)
        held_after = hold_result()
    except Exception as exc:
        error = exc
    after_contract = _command_contract(fields, command, validation, repo_root=repo_root)
    after = _recovery_source_snapshot(fields, repo_root=repo_root)
    protected_after = _recovery_protected_scripts(command, validation, policy["argv"], repo_root=repo_root)
    changed = [name for name in before if before[name] != after.get(name)]
    forbidden = protected != protected_after or _file_digest(todo_path) != board_before
    if forbidden:
        state, reason = "blocked", "recovery_changed_controller_validator_or_taskboard"
    elif error is not None or held_after:
        state, reason = "interrupted", "recovery_operator_hold" if held_after or isinstance(error, RecoveryHeld) else "recovery_owner_interrupted"
    elif code:
        state, reason = "failed", "recovery_command_failed"
    elif not changed or after_contract == contract:
        state, reason = "noop", "recovery_no_material_production_source_change"
    elif any(after.get(name) is None for name in changed):
        state, reason = "blocked", "recovery_removed_production_source"
    else:
        state, reason = "applied", "recovery_production_source_changed_requires_original_validation"
    row.update(state=state, reason=reason, exit_code=code, finished_at_ms=int(time.time() * 1000),
               source_before=before, source_after=after)
    row["approved_contract" if state == "applied" else "blocked_contract"] = after_contract
    ledger.save()
    with open_command_log(log_path, f"semantic recovery {task_id}: {state} ({reason}); spent={spent + 1}/{policy['limit']}\n"):
        pass
    if isinstance(error, _LaneResumeInterrupted):
        raise error
    if held_after or isinstance(error, RecoveryHeld):
        return hold_result() or {"state": "degraded", "phase": "operator_hold", "active_task_id": "", "busy": False,
                                "work_complete": False, "completion_authority": False}
    return {"state": "degraded", "phase": "semantic_recovery_" + state, "active_task_id": task_id,
            "terminal_reason": "", "backpressure_reasons": [f"{task_id}:{reason}"],
            "busy": False, "work_complete": False, "completion_authority": False}


def _transient_resource_refusal(log_path: Path, start: int, task_id: str) -> str:
    """Typed refusal, or fresh parent measurement when full storage lost stdout.

    The caller restricts this to exit 75 from the current producer. A readable
    current-task diagnostic remains authoritative; unrelated failures never
    become storage errors merely because a message contains pressure wording.
    """
    def measured_pressure() -> str:
        capacity = check_storage_capacity((log_path,))
        return "storage_pressure" if not capacity["ok"] else ""

    try:
        with log_path.open("rb") as stream:
            stream.seek(start)
            data = stream.read(131073)
        if len(data) > 131072:
            return measured_pressure()
        for line in reversed(data.decode("utf-8", errors="replace").splitlines()):
            try:
                value = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(value, dict) or value.get("task_id") != task_id:
                continue
            error = value.get("error")
            code = error.get("code") if isinstance(error, dict) else value.get("code")
            if code == "browser_contention":
                return code
            if (code == "storage_pressure" and isinstance(error, dict)
                    and value.get("retryable") is True
                    and value.get("work_complete") is False
                    and value.get("completion_authority") is False
                    and "errno" in error):
                reason, number = error.get("reason"), error["errno"]
                if ((reason == "insufficient_free_bytes" and number is None)
                        or (reason == "ENOSPC" and type(number) is int and number == 28)
                        or (reason == "EDQUOT" and type(number) is int and number == 122)):
                    return code
            return ""
    except OSError as error:
        if not is_storage_pressure(error) and not isinstance(error, FileNotFoundError):
            return ""
    # The child's write may fail even though the parent successfully opened its
    # log earlier. This reports our actual measurement, not a fabricated errno.
    return measured_pressure()


def run_resume_cycle(
    *,
    repo_root: Path,
    todo_path: Path,
    task_header_prefix: str,
    attempt_path: Path,
    log_path: Path,
    heartbeat: Callable[..., None],
    allow_bounded_implement: bool = False,
) -> dict[str, Any]:
    """Run one safe task action.

    A deterministic command or repair wins. A blank task may use one
    edit-scoped implementer when that opt-in is set. A provider command
    written on the task is still refused. Completion authority stays false.
    """

    from .supervisor_watchdog import (
        parse_todo_task_fields,
        selectable_pending_task_ids,
    )

    try:
        todo_text = todo_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "state": "blocked",
            "phase": "todo_unreadable",
            "active_task_id": "",
            "terminal_reason": f"todo_unreadable: {exc}",
            "backpressure_reasons": ["todo_unreadable"],
            "busy": False,
            "work_complete": False,
            "completion_authority": False,
        }
    selected = selectable_pending_task_ids(
        todo_text, task_header_prefix=task_header_prefix
    )
    blocks = {
        task_id: fields
        for task_id, fields in parse_todo_task_fields(
            todo_text, task_header_prefix=task_header_prefix
        )
    }
    selected = _order_explicit_task_priorities(selected, blocks)
    attempts, retry_after, fixed_points, command_evidence = _load_attempt_state(attempt_path)
    gate_recovery = GateRecovery(attempt_path)

    def save_attempts() -> None:
        _store_attempt_state(attempt_path, attempts, retry_after, fixed_points, command_evidence)

    evidence_exclusions = (todo_path, attempt_path, log_path, gate_recovery.path)

    def archive_resource_window(task: dict[str, Any], record: Mapping[str, Any], reason: str) -> None:
        history = list(task.get("resource_wait_history") or [])
        task["resource_wait_history"] = [*history[-15:], {**record, "ended_at_ms": int(time.time() * 1000),
                                                        "end_reason": reason}]

    def run_producer(
        task_id: str, fields: Mapping[str, str], command: Sequence[str],
        validation: Sequence[str] | None, **options: Any,
    ) -> int:
        # Invalidate first: an interrupted or later failed invocation cannot
        # inherit evidence produced by an earlier invocation.
        previous_evidence = command_evidence.pop(task_id, None) or {}
        before = _declared_evidence(fields, repo_root=repo_root, excluded=evidence_exclusions)
        contract = _command_contract(fields, command, validation, repo_root=repo_root)
        _begin_execution_progress(fixed_points, task_id, contract, now_ms=int(time.time() * 1000),
                                  previous_evidence=previous_evidence, previous_attempts=attempts.get(task_id, 0))
        save_attempts()
        timeout = float(options["timeout"])
        code = 0
        preflight_attempted = False
        producer_invoked = False
        producer_log_start = log_path.stat().st_size if log_path.is_file() else 0
        failed_stage = "pre_command_recovery"
        try:
            hooks = recovery_commands(task_id, command, repo_root=repo_root)
            hook_env = recovery_environment() if hooks else None
        except RecoveryPolicyError as exc:
            preflight_attempted = True
            with open_command_log(log_path, f"pre-command recovery refused {task_id}: {exc}\n"):
                pass
            hooks = []
            code = 1
        deadline = time.monotonic() + timeout if hooks else 0.0
        for hook, limit in hooks:
            preflight_attempted = True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                code = _COMMAND_TIMEOUT_EXIT
                break
            held = hold_result()
            if held:
                code = 1
                break
            try:
                if recovery_commands(task_id, command, repo_root=repo_root) != hooks:
                    raise RecoveryPolicyError("VDG pre-command recovery policy changed during invocation")
            except RecoveryPolicyError as exc:
                with open_command_log(log_path, f"pre-command recovery refused {task_id}: {exc}\n"):
                    pass
                code = 1
                break
            hook_options = {**options, "timeout": min(limit, remaining), "env": hook_env}
            hook_options["heartbeat"] = lambda: heartbeat(
                active_task_id=task_id, state="healthy", phase="pre_command_recovery",
            )
            hook_options["heartbeat"]()
            code = _run_argv(hook, **hook_options)
            if code:
                # A repair preflight refusal is not an acceptance gate. It
                # spends a genuine ordinary attempt and remains recoverable.
                if code == _GATE_WITHHELD_EXIT:
                    code = 1
                break
        if code == 0:
            if not hooks:
                # Preserve the ordinary runner path when no policy applies.
                producer_invoked = True
                failed_stage = "command"
                producer_log_start = log_path.stat().st_size if log_path.is_file() else 0
                code = _run_argv(command, **options)
            else:
                held = hold_result()
                if held:
                    code = 1
                else:
                    try:
                        if recovery_commands(task_id, command, repo_root=repo_root) != hooks:
                            raise RecoveryPolicyError("VDG pre-command recovery policy changed before producer")
                    except RecoveryPolicyError as exc:
                        with open_command_log(log_path, f"pre-command recovery refused {task_id}: {exc}\n"):
                            pass
                        code = 1
                    if code == 0:
                        remaining = deadline - time.monotonic()
                        if remaining > 0:
                            producer_invoked = True
                            failed_stage = "command"
                            producer_log_start = log_path.stat().st_size if log_path.is_file() else 0
                            code = _run_argv(command, **{**options, "timeout": remaining})
                        else:
                            code = _COMMAND_TIMEOUT_EXIT
        if code == 0 and hooks:
            # A global test deadline can leave a full NO_GO report that the
            # acquisition-only checker accepts. Never commit that producer as
            # success: fail the whole contract ordinarily, so its next attempt
            # repairs the prefix before any cached acquisition can bypass it.
            failed_stage = "post_command_recovery_guard"
            try:
                if recovery_commands(task_id, command, repo_root=repo_root) != hooks:
                    raise RecoveryPolicyError("VDG recovery policy changed after producer")
                guards = recovery_commands(task_id, command, repo_root=repo_root, apply=False)
            except RecoveryPolicyError as exc:
                with open_command_log(log_path, f"post-command recovery guard refused {task_id}: {exc}\n"):
                    pass
                guards = []
                code = 1
            for guard, limit in guards:
                held = hold_result()
                if held:
                    code = 1
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    code = _COMMAND_TIMEOUT_EXIT
                    break
                try:
                    if recovery_commands(task_id, command, repo_root=repo_root, apply=False) != guards:
                        raise RecoveryPolicyError("VDG recovery policy changed during post-command guard")
                except RecoveryPolicyError as exc:
                    with open_command_log(log_path, f"post-command recovery guard refused {task_id}: {exc}\n"):
                        pass
                    code = 1
                    break
                guard_heartbeat = lambda: heartbeat(
                    active_task_id=task_id, state="healthy", phase="post_command_recovery_guard",
                )
                guard_heartbeat()
                code = _run_argv(guard, **{**options, "timeout": min(limit, remaining),
                    "env": hook_env, "heartbeat": guard_heartbeat})
                if code:
                    if code == _GATE_WITHHELD_EXIT:
                        code = 1
                    break
            if code == 0:
                failed_stage = "command"
        if code != _GATE_WITHHELD_EXIT:
            command_evidence[task_id] = {
                # Proof remains bound to the code that started this invocation.
                "contract": contract, "before": before,
                "after": _declared_evidence(fields, repo_root=repo_root, excluded=evidence_exclusions),
                "exit_code": code,
                # Own generated source must not appear to be an external repair.
                "retry_contract": _command_contract(fields, command, validation, repo_root=repo_root),
            }
            if preflight_attempted:
                command_evidence[task_id].update(
                    producer_invoked=producer_invoked,
                    failed_stage=failed_stage,
                )
        resource_code = (_transient_resource_refusal(log_path, producer_log_start, task_id)
                         if code == _GATE_WITHHELD_EXIT and producer_invoked and failed_stage == "command" else "")
        finished_at = int(time.time() * 1000)
        if resource_code:
            task_record = fixed_points.setdefault(task_id, {})
            old = task_record.get("resource_wait") or {}
            if old.get("contract") != contract:
                old = {}
            refusals = int(old.get("refusals") or 0) + 1
            first = int(old.get("first_at_ms") or finished_at)
            task_record["resource_refusals_total"] = int(task_record.get("resource_refusals_total") or 0) + 1
            task_record["resource_wait"] = {"contract": contract, "code": resource_code,
                "refusals": refusals, "first_at_ms": first,
                "deadline_at_ms": int(old.get("deadline_at_ms") or first + min(timeout, 1200) * 1000),
                "next_retry_at_ms": finished_at + min(300_000, _RESOURCE_RETRY_BASE_MS * 2 ** min(refusals - 1, 4))}
            waiting = task_record["resource_wait"]
            if refusals >= _RESOURCE_RETRY_LIMIT or finished_at >= waiting["deadline_at_ms"]:
                waiting.update(window_closed=True, next_retry_at_ms=finished_at + 300_000)
        else:
            task_record = fixed_points.setdefault(task_id, {})
            old = task_record.pop("resource_wait", None)
            if old:
                archive_resource_window(task_record, old, "producer_reacquired" if code == 0 else "producer_exit")
        _finish_execution_progress(fixed_points, task_id, code=code, stage=failed_stage,
                                   contract=_command_contract(fields, command, validation, repo_root=repo_root),
                                   now_ms=finished_at, resource_refusal=resource_code)
        save_attempts()
        return code

    def record_retry_contract(
        task_id: str, fields: Mapping[str, str], command: Sequence[str],
        validation: Sequence[str] | None,
    ) -> None:
        # Validators may regenerate source too. Refresh only the retry budget;
        # invocation-start proof provenance remains unchanged.
        record = command_evidence.get(task_id)
        if not isinstance(record, dict):
            return
        current = _command_contract(fields, command, validation, repo_root=repo_root)
        if record.get("retry_contract") != current:
            record["retry_contract"] = current
            progress = (fixed_points.get(task_id) or {}).get("execution_progress")
            if isinstance(progress, dict):
                progress["contract"] = current
            save_attempts()

    validation_commands = {
        task_id: parse_deterministic_command((blocks.get(task_id) or {}).get("validation") or "",
                                             repo_root=repo_root)
        for task_id in selected
    }
    invalid_validators = {
        task_id for task_id in selected
        if str((blocks.get(task_id) or {}).get("validation") or "").strip()
        and validation_commands[task_id] is None
    }
    skipped: list[str] = []
    gate_waits: list[dict[str, Any]] = []
    semantic_waits: list[str] = []
    timeout_revalidation_ids: list[str] = []
    now_ms = int(time.time() * 1000)

    def resource_result(task_id: str) -> dict[str, Any] | None:
        record = (fixed_points.get(task_id) or {}).get("resource_wait") or {}
        if not record:
            return None
        exhausted = bool(record.get("window_closed"))
        return {"state": "degraded", "phase": "resource_retry_wait",
                "active_task_id": "", "resource_task_id": task_id,
                "terminal_reason": "",
                "backpressure_reasons": [f"{task_id}:{record['code']}"],
                "next_retry_at_ms": record["next_retry_at_ms"], "resource_window_exhausted": exhausted,
                "resource_refusals": record["refusals"], "resource_retry_limit": _RESOURCE_RETRY_LIMIT,
                "resource_refusals_total": (fixed_points.get(task_id) or {}).get("resource_refusals_total", 0),
                "busy": False, "work_complete": False, "completion_authority": False}

    def withheld_result(task_id: str, fields: Mapping[str, str], command: Sequence[str],
                        validation: Sequence[str] | None = None) -> dict[str, Any]:
        return resource_result(task_id) or gate_result(task_id, fields, command, validation)

    def gate_result(
        task_id: str, fields: Mapping[str, str], command: Sequence[str],
        validation_command: Sequence[str] | None = None,
        *, probe_only: bool = False, validation_exit_code: int = _GATE_WITHHELD_EXIT,
    ) -> dict[str, Any]:
        held = hold_result()
        if held:
            return held
        storage_wait = _recovery_storage_wait(task_id, repo_root=repo_root, attempt_path=attempt_path,
                                              log_path=log_path)
        if storage_wait:
            return storage_wait
        # Exit 75 is a valid refusal, not permission to complete the task or
        # erase the evidence that this same refusal has repeated for days.
        attempts.pop(task_id, None)
        retry_after.pop(task_id, None)
        save_attempts()
        fingerprint, evidence = gate_evidence(
            task_id, blocks, repo_root=repo_root, todo_path=todo_path, log_path=log_path,
        )
        source_contract = gate_source_contract(
            evidence, _command_contract(fields, command, validation_commands.get(task_id), repo_root=repo_root),
        )
        previous = gate_recovery.records.get(task_id) or {}
        record = gate_recovery.observe(task_id, fingerprint, now_ms=int(time.time() * 1000), source_contract=source_contract)
        if str(fields.get("retry_on_changed_evidence") or "").casefold() == "true":
            # A probe can discover new evidence without executing the producer.
            # Keep the last producer's refusal fingerprint across that probe,
            # including restarts; only a real producer consumes the new guard.
            producer_fingerprint = previous.get("producer_evidence_fingerprint", previous.get("fingerprint"))
            record["producer_evidence_fingerprint"] = (producer_fingerprint
                if probe_only and isinstance(producer_fingerprint, str)
                and re.fullmatch(r"[0-9a-f]{64}", producer_fingerprint) else fingerprint)
        record["withheld_stage"] = "validation" if validation_command else "command"
        record["last_probe"] = "validation_only" if probe_only else "producer"
        if validation_command:
            record["last_validation_exit_code"] = validation_exit_code
        gate_recovery.store(task_id, record)
        repeated = max(record["unchanged_probes"], int(record.get("source_refusals") or 0)) >= GATE_REPEAT_LIMIT
        if repeated and not record.get("dependencies_revalidated"):
            # Persist the fence before executing a validator: a restart cannot
            # convert an interrupted repair into an unlimited retry budget.
            probe_started = time.monotonic()
            probe_budget = _task_timeout(fields)
            probe_deadline = probe_started + probe_budget
            record["dependencies_revalidated"] = True
            gate_recovery.store(task_id, record)
            validations: list[dict[str, Any]] = []
            for dependency in evidence["dependencies"]:
                held = hold_result()
                if held:
                    return held
                source = blocks.get(dependency) or {}
                raw = str(source.get("validation") or "")
                argv = parse_deterministic_command(raw, repo_root=repo_root)
                remaining = max(0.0, probe_deadline - time.monotonic())
                timeout = min(_dependency_validation_timeout(source), remaining)
                executed = argv is not None and remaining > 0
                code = _run_argv(
                    argv, repo_root=repo_root,
                    cleanup_descendants_on_success=source.get("resource_class") == "cpu-browser",
                    timeout=timeout, log_path=log_path,
                    heartbeat=lambda: heartbeat(
                        active_task_id=task_id, state="degraded", phase="gate_dependency_revalidation",
                    ),
                ) if executed else None
                validations.append({
                    "task_id": dependency, "validation": raw, "exit_code": code,
                    "executed": executed, "timeout_seconds": timeout if executed else None,
                    "not_run_reason": None if executed else "dependency_probe_budget_exhausted"
                    if remaining <= 0 else "validation_command_refused",
                    "probe_budget_seconds": probe_budget,
                    "probe_elapsed_seconds": max(0.0, time.monotonic() - probe_started),
                    "validation_strength": "existence_only" if argv and Path(argv[0]).name == "test"
                    and len(argv) > 1 and argv[1] in {"-f", "-e", "-d", "-s", "-r", "-w", "-x", "-L", "-h"}
                    else "declared",
                    "resolved_gate": False,
                })
            record["dependency_validations"] = validations
            gate_recovery.store(task_id, record)
        repair = parse_deterministic_command(fields.get("deterministic_repair") or "", repo_root=repo_root)
        if repeated and repair and list(repair) != list(command) and not record.get("repair_attempted"):
            held = hold_result()
            if held:
                return held
            # Dependency probes may have run for minutes since gate admission.
            storage_wait = _recovery_storage_wait(task_id, repo_root=repo_root, attempt_path=attempt_path,
                                                  log_path=log_path)
            if storage_wait:
                return storage_wait
            record["repair_attempted"] = True
            gate_recovery.store(task_id, record)
            code = _run_argv(
                repair, repo_root=repo_root,
                cleanup_descendants_on_success=fields.get("resource_class") == "cpu-browser",
                timeout=_task_timeout(fields), log_path=log_path,
                heartbeat=lambda: heartbeat(active_task_id=task_id, state="degraded", phase="gate_deterministic_repair"),
            )
            record["deterministic_repair"] = {"command": list(repair), "exit_code": code}
            gate_recovery.store(task_id, record)
            if code == 0:
                held = hold_result()
                if held:
                    return held
                gate_code = _run_argv(
                    command, repo_root=repo_root,
                    cleanup_descendants_on_success=fields.get("resource_class") == "cpu-browser",
                    timeout=_task_timeout(fields), log_path=log_path,
                    heartbeat=lambda: heartbeat(active_task_id=task_id, state="healthy", phase="gate_revalidation"),
                )
                if gate_code == 0 and validation_command:
                    held = hold_result()
                    if held:
                        return held
                    gate_code = _run_argv(
                        validation_command, repo_root=repo_root,
                        cleanup_descendants_on_success=fields.get("resource_class") == "cpu-browser",
                        timeout=_validation_timeout(fields), log_path=log_path,
                        heartbeat=lambda: heartbeat(active_task_id=task_id, state="healthy", phase="gate_revalidation"),
                    )
                record["deterministic_repair"]["gate_exit_code"] = gate_code
                gate_recovery.store(task_id, record)
                if gate_code == 0:
                    gate_recovery.clear(task_id)
                    return {
                        "state": "healthy", "phase": "gate_repaired", "active_task_id": task_id,
                        "terminal_reason": "", "backpressure_reasons": [], "busy": False,
                        "evidence_changed": True, "work_complete": False, "completion_authority": False,
                    }
        if repeated:
            # Dependency probes and a repair may take longer than the initial
            # retry delay. The cooldown starts when those actions finish.
            delay_ms = max(0, int(record["next_retry_at_ms"]) - int(record["last_seen_ms"]))
            record["next_retry_at_ms"] = int(time.time() * 1000) + delay_ms
            gate_recovery.store(task_id, record)
        packet = str(gate_recovery.packet(task_id, record, evidence)) if repeated else ""
        return {
            "state": "degraded" if repeated else "healthy",
            "phase": "gate_recovery_needed" if repeated else "gate_withheld",
            "active_task_id": task_id, "terminal_reason": "",
            "backpressure_reasons": [f"{task_id}:gate_fixed_point" if repeated else f"{task_id}:gate_withheld"],
            "gate_evidence_fingerprint": fingerprint, "repair_packet": packet,
            "gate_recovery_needed": repeated,
            "next_retry_at_ms": record["next_retry_at_ms"],
            "evidence_changed": record["unchanged_probes"] == 1,
            "busy": False, "work_complete": False, "completion_authority": False,
        }

    hold_directories = {repo_root.resolve(), attempt_path.parent.resolve()}
    for directory in attempt_path.parent.resolve().parents:
        try:
            directory.relative_to(repo_root.resolve())
        except ValueError:
            continue
        hold_directories.add(directory)
    hold_candidates = [directory / name for directory in sorted(hold_directories)
                       for name in ("HOLD", "OPERATOR_STOP", "watchdog.disabled", "hold", "operator.hold")]
    def hold_result() -> dict[str, Any] | None:
        holds = [path for path in hold_candidates if path.exists()]
        return {
            "state": "paused", "phase": "operator_hold", "active_task_id": "",
            "terminal_reason": "operator_hold", "backpressure_reasons": [str(path) for path in holds],
            "busy": False, "work_complete": False, "completion_authority": False,
        } if holds else None

    held = hold_result()
    if held:
        return held
    completed_statuses = {"completed", "complete", "done"}
    for task_id, fields in blocks.items():
        if str(fields.get("status") or "").strip().lower() not in completed_statuses:
            continue
        record = dict(fixed_points.get(task_id) or {})
        if record.get("admitted"):
            continue
        _digest, artifact_text = _artifact_text(fields, repo_root=repo_root)
        admission = _admit_validation_gap(
            todo_text,
            repo_root=repo_root,
            task_header_prefix=task_header_prefix,
            task_id=task_id,
            fields=fields,
            artifact_text=artifact_text,
        )
        if admission.get("admitted") and admission.get("todo_text"):
            _atomic_write(todo_path, str(admission["todo_text"]))
            record["admitted"] = str(admission.get("derived_task_id") or "")
            record["reason"] = str(admission.get("reason") or "")
            fixed_points[task_id] = record
            save_attempts()
            return {
                "state": "degraded",
                "phase": "gap_refilled",
                "active_task_id": record["admitted"],
                "terminal_reason": "",
                "backpressure_reasons": [f"{task_id}:validation_fixed_point"],
                "refill_state": "admitted",
                "busy": False,
                "work_complete": False,
                "completion_authority": False,
            }
        if admission.get("reason") == "duplicate_refill" and admission.get("derived_task_id"):
            record["admitted"] = str(admission["derived_task_id"])
            fixed_points[task_id] = record
            save_attempts()
    for task_id in selected:
        held = hold_result()
        if held:
            return held
        fields = blocks.get(task_id) or {}
        try:
            failure_limit = semantic_failure_limit(fields)
            semantic_recovery.fixture_inputs(repo_root, fields)
        except (ValueError, semantic_recovery.SemanticRecoveryError) as exc:
            semantic_waits.append(task_id)
            skipped.append(f"{task_id}:{exc}")
            continue
        if task_id in invalid_validators:
            skipped.append(f"{task_id}:validation_provider_or_shell_refused")
            continue
        validation = validation_commands[task_id]
        command_text = str(fields.get("deterministic_command") or "")
        repair_text = str(fields.get("deterministic_repair") or "")
        command = parse_deterministic_command(command_text, repo_root=repo_root)
        if command is None and not command_text.strip():
            command = parse_deterministic_command(repair_text, repo_root=repo_root)
        if command is None:
            reason = (
                "provider_or_shell_refused"
                if command_text.strip() or repair_text.strip()
                else "no_deterministic_command"
            )
            skipped.append(f"{task_id}:{reason}")
            continue
        resource_wait = (fixed_points.get(task_id) or {}).get("resource_wait") or {}
        if resource_wait.get("contract") == _command_contract(fields, command, validation, repo_root=repo_root):
            if not resource_wait.get("window_closed") and resource_wait["deadline_at_ms"] <= now_ms:
                resource_wait.update(window_closed=True,
                                     next_retry_at_ms=max(resource_wait["next_retry_at_ms"], now_ms + 300_000))
                save_attempts()
            result = resource_result(task_id)
            if result and resource_wait["next_retry_at_ms"] > now_ms:
                gate_waits.append(result)
                continue
            if resource_wait.get("window_closed"):
                archive_resource_window(fixed_points[task_id], resource_wait, "bounded_window_cooldown_finished")
                fixed_points[task_id]["resource_wait"] = {"contract": resource_wait["contract"],
                    "code": resource_wait["code"], "refusals": 0, "first_at_ms": now_ms,
                    "deadline_at_ms": now_ms + min(_task_timeout(fields), 1200) * 1000,
                    "next_retry_at_ms": now_ms}
                save_attempts()
        recovery_blocked = False
        if semantic_recovery.journal_path(attempt_path).exists():
            try:
                recovery_journal = semantic_recovery.Journal(semantic_recovery.journal_path(attempt_path), _atomic_write)
                current_contract = _command_contract(fields, command, validation, repo_root=repo_root)
                source_now = _recovery_source_snapshot(fields, repo_root=repo_root)
                recovery_journal.reconcile(task_id, current_contract, now_ms, source_now)
                recovery_summary = recovery_journal.summary(task_id)
                recovery_blocked = bool(recovery_summary.get("blocked_contract") and (
                    recovery_summary["blocked_contract"] == current_contract
                    or recovery_summary.get("source_after") == source_now))
            except semantic_recovery.SemanticRecoveryError as exc:
                semantic_waits.append(task_id)
                skipped.append(f"{task_id}:{exc}")
                continue
        if str((fixed_points.get(task_id) or {}).get("admitted") or ""):
            skipped.append(f"{task_id}:validation_fixed_point")
            continue
        fingerprint = ""
        source_contract = ""
        if task_id in gate_recovery.records:
            fingerprint, _evidence = gate_evidence(
                task_id, blocks, repo_root=repo_root, todo_path=todo_path, log_path=log_path,
            )
            source_contract = gate_source_contract(
                _evidence, _command_contract(fields, command, validation, repo_root=repo_root),
            )
        previous_gate = gate_recovery.records.get(task_id) or {}
        same_source = bool(source_contract and isinstance(previous_gate, dict)
                           and previous_gate.get("source_contract") == source_contract)
        producer_fingerprint = previous_gate.get("producer_evidence_fingerprint", previous_gate.get("fingerprint"))
        retry_changed_evidence = (
            str(fields.get("retry_on_changed_evidence") or "").casefold() == "true"
            and isinstance(producer_fingerprint, str)
            and re.fullmatch(r"[0-9a-f]{64}", producer_fingerprint) is not None
            and producer_fingerprint != fingerprint
        )
        gate_record = gate_recovery.current(task_id, fingerprint) if same_source else {}
        if int(gate_record.get("next_retry_at_ms") or 0) > now_ms and not retry_changed_evidence:
            packet = gate_recovery.path.with_name(f"gate-recovery-{task_id}.json")
            gate_waits.append({
                "state": "degraded", "phase": "gate_retry_wait", "active_task_id": task_id,
                "terminal_reason": "", "backpressure_reasons": [f"{task_id}:gate_fixed_point"],
                "gate_evidence_fingerprint": fingerprint,
                "gate_recovery_needed": True,
                "repair_packet": str(packet) if packet.is_file() else "",
                "next_retry_at_ms": int(gate_record["next_retry_at_ms"]),
                "busy": False, "work_complete": False, "completion_authority": False,
            })
            continue
        # Some explicitly registered tasks expose a cheap external resource
        # guard as declared evidence. A changed guard must admit the producer
        # again: its validator cannot manufacture the missing execution.
        # Default lanes keep their existing validation-only retry policy.
        if same_source and validation is not None and not retry_changed_evidence:
            # A producer that already refused has evidence to inspect. Re-run
            # its acceptance check cheaply before another expensive capture,
            # even when the command explicitly bypasses its own cache. A green
            # probe still has to pass the original producer and validation.
            validation_code = _run_argv(
                validation, repo_root=repo_root,
                cleanup_descendants_on_success=fields.get("resource_class") == "cpu-browser",
                timeout=_validation_timeout(fields), log_path=log_path,
                heartbeat=lambda: heartbeat(
                    active_task_id=task_id, state="degraded", phase="gate_evidence_revalidation",
                    gate_evidence_fingerprint=fingerprint,
                    gate_recovery_needed=int(previous_gate.get("source_refusals") or 0) >= GATE_REPEAT_LIMIT,
                    repair_packet=str(gate_recovery.path.with_name(f"gate-recovery-{task_id}.json"))
                    if int(previous_gate.get("source_refusals") or 0) >= GATE_REPEAT_LIMIT else "",
                    next_retry_at_ms=int(previous_gate.get("next_retry_at_ms") or 0),
                ),
            )
            held = hold_result()
            if held:
                return held
            if validation_code != 0:
                gate_waits.append(gate_result(
                    task_id, fields, command, validation,
                    probe_only=True, validation_exit_code=validation_code,
                ))
                continue
        exhausted = attempts.get(task_id, 0) >= _ATTEMPT_LIMIT
        if exhausted and not recovery_blocked:
            prior = command_evidence.get(task_id) or {}
            prior_contract = prior.get("retry_contract", prior.get("contract"))
            proof_contract = prior.get("contract")
            if (
                isinstance(proof_contract, str)
                and re.fullmatch(r"[0-9a-f]{64}", proof_contract)
                and isinstance(prior_contract, str)
                and re.fullmatch(r"[0-9a-f]{64}", prior_contract)
                and isinstance(prior.get("before"), dict)
                and isinstance(prior.get("after"), dict)
                and isinstance(prior.get("exit_code"), int)
                and prior_contract != _command_contract(fields, command, validation, repo_root=repo_root)
            ):
                # A repaired contract has not spent the old contract's budget.
                # The producer replaces this receipt before it starts, so an
                # unchanged repair cannot keep resetting its own cooldown.
                attempts[task_id] = 0
                retry_after.pop(task_id, None)
                exhausted = False
                save_attempts()
                with open_command_log(log_path, f"retry rearmed {task_id}: command contract changed\n"):
                    pass
        progress = (fixed_points.get(task_id) or {}).get("execution_progress") or {}
        if (recovery_blocked or (progress.get("contract") == _command_contract(fields, command, validation, repo_root=repo_root)
                and int(progress.get("semantic_failures") or 0) >= failure_limit)):
            # A successful producer must not run again merely because the
            # old implicit 300s validator ceiling cut off its acceptance. Admit
            # one durable validation-only review under the explicit task cap.
            # Preserve the original contract, attempt/failure counts and raw
            # outputs; changed or unsuccessful producer evidence cannot enter.
            recovery_key = hashlib.sha256(json.dumps({
                "contract": progress.get("contract"), "timeout": _validation_timeout(fields),
                "workflow": "explicit-validator-budget-v1",
            }, sort_keys=True).encode()).hexdigest()
            old_review = (fixed_points.get(task_id) or {}).get("validation_timeout_recovery") or {}
            producer_record = command_evidence.get(task_id) or {}
            if (not recovery_blocked and validation is not None
                    and progress.get("last_stage") == "validation"
                    and progress.get("last_exit_code") == _COMMAND_TIMEOUT_EXIT
                    and _validation_timeout(fields) > 300.0
                    and producer_record.get("exit_code") == 0
                    and old_review.get("key") != recovery_key
                    and _produced_evidence_matches(producer_record, fields, command, validation,
                        repo_root=repo_root, excluded=evidence_exclusions)):
                timeout_revalidation_ids.append(task_id)
                continue
            try:
                recovery_result = _run_semantic_recovery(
                    task_id, fields, command, validation, repo_root=repo_root, todo_path=todo_path,
                    attempt_path=attempt_path, log_path=log_path, progress=progress,
                    heartbeat=heartbeat, hold_result=hold_result,
                )
                if recovery_result:
                    return recovery_result
            except semantic_recovery.SemanticRecoveryError as exc:
                skipped.append(f"{task_id}:{exc}")
            semantic_waits.append(task_id)
            skipped.append(f"{task_id}:unchanged_semantic_failure_budget_exhausted")
            if not progress.get("escalated"):
                progress["escalated"] = True
                save_attempts()
                with open_command_log(log_path, f"semantic failure budget exhausted {task_id}: awaiting changed execution inputs\n"):
                    pass
            continue
        if (
            exhausted
            and retry_after.get(task_id, 0) > now_ms
            and not missing_module_repair_pending(
                command,
                repo_root=repo_root,
                log_path=log_path,
                attempt_path=attempt_path,
            )
        ):
            skipped.append(f"{task_id}:attempts_exhausted")
            continue
        timeout = _task_timeout(fields)
        if exhausted or missing_module_repair_pending(
            command,
            repo_root=repo_root,
            log_path=log_path,
            attempt_path=attempt_path,
        ):
            repaired = repair_missing_module(
                command,
                repo_root=repo_root,
                log_path=log_path,
                attempt_path=attempt_path,
                timeout=timeout,
                heartbeat=lambda **kwargs: heartbeat(active_task_id=task_id, **kwargs),
            )
            if repaired:
                attempts[task_id] = 0
                retry_after.pop(task_id, None)
                save_attempts()
        log_start = log_path.stat().st_size if log_path.is_file() else 0
        heartbeat(active_task_id=task_id, state="starting", phase="deterministic_command")
        exit_code = run_producer(
            task_id, fields, command, validation,
            repo_root=repo_root,
            cleanup_descendants_on_success=fields.get("resource_class") == "cpu-browser",
            timeout=timeout,
            log_path=log_path,
            heartbeat=lambda: heartbeat(
                active_task_id=task_id,
                state="healthy",
                phase="deterministic_command",
            ),
        )
        held = hold_result()
        if held:
            return held
        if exit_code == _GATE_WITHHELD_EXIT:
            gate_waits.append(withheld_result(task_id, fields, command))
            continue
        if exit_code != 0:
            gate_recovery.clear(task_id)
        attempts[task_id] = attempts.get(task_id, 0) + 1
        if exit_code != 0 and attempts[task_id] >= _ATTEMPT_LIMIT:
            retry_after[task_id] = int(time.time() * 1000) + _EXHAUSTED_RETRY_MS
        save_attempts()
        if exit_code != 0 and repair_missing_module(
            command,
            repo_root=repo_root,
            log_path=log_path,
            attempt_path=attempt_path,
            timeout=timeout,
            heartbeat=lambda **kwargs: heartbeat(active_task_id=task_id, **kwargs),
        ):
            attempts[task_id] = 0
            exit_code = run_producer(
                task_id, fields, command, validation,
                repo_root=repo_root,
                cleanup_descendants_on_success=fields.get("resource_class") == "cpu-browser",
                timeout=timeout,
                log_path=log_path,
                heartbeat=lambda: heartbeat(
                    active_task_id=task_id,
                    state="healthy",
                    phase="deterministic_command",
                ),
            )
            held = hold_result()
            if held:
                return held
            if exit_code == _GATE_WITHHELD_EXIT:
                gate_waits.append(withheld_result(task_id, fields, command))
                continue
            attempts[task_id] = 1 if exit_code == 0 else attempts.get(task_id, 0) + 1
            if exit_code != 0 and attempts[task_id] >= _ATTEMPT_LIMIT:
                retry_after[task_id] = int(time.time() * 1000) + _EXHAUSTED_RETRY_MS
            save_attempts()
        if exit_code != 0 and repair_missing_named_export(
            command,
            repo_root=repo_root,
            log_path=log_path,
            log_start=log_start,
            attempt_path=attempt_path,
            timeout=timeout,
            allow_bounded_implement=allow_bounded_implement,
            heartbeat=lambda **kwargs: heartbeat(active_task_id=task_id, **kwargs),
        ):
            attempts[task_id] = 0
            exit_code = run_producer(
                task_id, fields, command, validation,
                repo_root=repo_root,
                cleanup_descendants_on_success=fields.get("resource_class") == "cpu-browser",
                timeout=timeout,
                log_path=log_path,
                heartbeat=lambda: heartbeat(
                    active_task_id=task_id,
                    state="healthy",
                    phase="deterministic_command",
                ),
            )
            if exit_code == _GATE_WITHHELD_EXIT:
                gate_waits.append(withheld_result(task_id, fields, command))
                continue
            attempts[task_id] = 1 if exit_code == 0 else attempts.get(task_id, 0) + 1
            if exit_code != 0 and attempts[task_id] >= _ATTEMPT_LIMIT:
                retry_after[task_id] = int(time.time() * 1000) + _EXHAUSTED_RETRY_MS
            save_attempts()
        if exit_code != 0:
            return {
                "state": "degraded",
                "phase": "deterministic_command_failed",
                "active_task_id": task_id,
                "terminal_reason": "",
                "backpressure_reasons": [f"{task_id}:exit_{exit_code}"],
                "busy": False,
                "work_complete": False,
                "completion_authority": False,
            }
        held = hold_result()
        if held:
            return held
        if validation is not None:
            validation_code = _run_argv(
                validation,
                repo_root=repo_root,
                cleanup_descendants_on_success=fields.get("resource_class") == "cpu-browser",
                timeout=_validation_timeout(fields),
                log_path=log_path,
                heartbeat=lambda: heartbeat(
                    active_task_id=task_id,
                    state="healthy",
                    phase="validation",
                ),
            )
            held = hold_result()
            if held:
                return held
            record_retry_contract(task_id, fields, command, validation)
            _finish_execution_progress(fixed_points, task_id, code=validation_code, stage="validation",
                                       contract=_command_contract(fields, command, validation, repo_root=repo_root),
                                       now_ms=int(time.time() * 1000))
            save_attempts()
            if validation_code == _GATE_WITHHELD_EXIT:
                gate_waits.append(gate_result(task_id, fields, command, validation))
                continue
            if validation_code != 0:
                digest, artifact_text = _artifact_text(fields, repo_root=repo_root)
                record = dict(fixed_points.get(task_id) or {})
                record["digest"] = digest
                record["failures"] = int(record.get("failures") or 0) + 1
                fixed_points[task_id] = record
                admitted_id = ""
                if attempts.get(task_id, 0) >= _ATTEMPT_LIMIT and not record.get("admitted"):
                    admission = _admit_validation_gap(
                        todo_path.read_text(encoding="utf-8"),
                        repo_root=repo_root,
                        task_header_prefix=task_header_prefix,
                        task_id=task_id,
                        fields=fields,
                        artifact_text=artifact_text,
                    )
                    if admission.get("admitted") and admission.get("todo_text"):
                        _atomic_write(todo_path, str(admission["todo_text"]))
                        admitted_id = str(admission.get("derived_task_id") or "")
                        record["admitted"] = admitted_id
                        fixed_points[task_id] = record
                    elif admission.get("reason") == "duplicate_refill" and admission.get(
                        "derived_task_id"
                    ):
                        admitted_id = str(admission["derived_task_id"])
                        record["admitted"] = admitted_id
                        fixed_points[task_id] = record
                if not admitted_id and attempts.get(task_id, 0) >= _ATTEMPT_LIMIT:
                    retry_after[task_id] = int(time.time() * 1000) + _EXHAUSTED_RETRY_MS
                save_attempts()
                if admitted_id:
                    return {
                        "state": "degraded",
                        "phase": "gap_refilled",
                        "active_task_id": admitted_id,
                        "terminal_reason": "",
                        "backpressure_reasons": [f"{task_id}:validation_fixed_point"],
                        "refill_state": "admitted",
                        "busy": False,
                        "work_complete": False,
                        "completion_authority": False,
                    }
                return {
                    "state": "degraded",
                    "phase": "validation_failed",
                    "active_task_id": task_id,
                    "terminal_reason": "",
                    "backpressure_reasons": [f"{task_id}:validation_exit_{validation_code}"],
                    "refill_state": "idle",
                    "busy": False,
                    "work_complete": False,
                    "completion_authority": False,
                }
        held = hold_result()
        if held:
            return held
        gate_recovery.clear(task_id)
        note = f"lane_resume deterministic command exit 0 at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
        try:
            updated = mark_task_completed(
                todo_path.read_text(encoding="utf-8"),
                task_id,
                task_header_prefix=task_header_prefix,
                note=note,
            )
        except (OSError, ValueError) as exc:
            return {
                "state": "blocked",
                "phase": "completion_write_failed",
                "active_task_id": task_id,
                "terminal_reason": str(exc),
                "backpressure_reasons": [f"{task_id}:completion_write_failed"],
                "busy": False,
                "work_complete": False,
                "completion_authority": False,
            }
        satisfies = str(fields.get("satisfies") or "").strip()
        if satisfies and satisfies in blocks:
            source_validation = parse_deterministic_command(
                (blocks.get(satisfies) or {}).get("validation") or "",
                repo_root=repo_root,
            )
            source_code = (
                1 if str((blocks.get(satisfies) or {}).get("validation") or "").strip() else 0
            ) if source_validation is None else _run_argv(
                source_validation,
                repo_root=repo_root,
                cleanup_descendants_on_success=(blocks.get(satisfies) or {}).get("resource_class") == "cpu-browser",
                timeout=_validation_timeout(blocks.get(satisfies) or {}),
                log_path=log_path,
                heartbeat=lambda: heartbeat(
                    active_task_id=satisfies,
                    state="healthy",
                    phase="validation",
                ),
            )
            if source_code == 0:
                try:
                    updated = mark_task_completed(
                        updated,
                        satisfies,
                        task_header_prefix=task_header_prefix,
                        note=(
                            "satisfied by "
                            + task_id
                            + " at "
                            + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        ),
                    )
                except ValueError:
                    pass
                else:
                    attempts.pop(satisfies, None)
                    retry_after.pop(satisfies, None)
                    fixed_points.pop(satisfies, None)
                    save_attempts()
        _atomic_write(todo_path, updated if updated.endswith("\n") else updated + "\n")
        _finish_execution_progress(fixed_points, task_id, code=0, stage="validation",
                                   contract=_command_contract(fields, command, validation, repo_root=repo_root),
                                   now_ms=int(time.time() * 1000), validated=True)
        save_attempts()
        return {
            "state": "healthy",
            "phase": "task_command_passed",
            "active_task_id": task_id,
            "terminal_reason": "",
            "backpressure_reasons": [],
            "refill_state": "satisfied" if satisfies else "idle",
            "busy": False,
            "advanced_task_id": task_id,
            "work_complete": False,
            "completion_authority": False,
        }
    if gate_waits and len(gate_waits) == len(selected):
        return gate_waits[0]
    if allow_bounded_implement:
        for task_id in selected:
            if task_id in invalid_validators:
                continue
            fields = blocks.get(task_id) or {}
            if str(fields.get("deterministic_command") or "").strip():
                continue
            if str(fields.get("deterministic_repair") or "").strip():
                continue
            marker = _bounded_marker(attempt_path, task_id)
            payload = _bounded_marker_payload(marker)
            try:
                escalation = max(0, int(payload.get("escalation") or 0))
            except (TypeError, ValueError):
                escalation = 0
            spent = _bounded_attempts(marker)
            if spent >= _BOUNDED_ATTEMPT_LIMIT:
                legacy = (
                    "turn_exhausted" not in payload
                    and _legacy_turn_exhausted(log_path, task_id)
                )
                if (payload.get("turn_exhausted") is True or legacy) and escalation < 1:
                    escalation = 1
                    spent = 0
                    _write_bounded_marker(
                        marker,
                        task_id=task_id,
                        attempts=0,
                        ok=False,
                        escalation=escalation,
                        turn_exhausted=False,
                    )
                else:
                    skipped.append(f"{task_id}:bounded_implement_already_used")
                    continue
            argv = prepare_bounded_implement(
                task_id,
                fields,
                repo_root=repo_root,
                attempt_path=attempt_path,
                max_turns=_bounded_max_turns(spent, escalation),
            )
            if not argv:
                skipped.append(f"{task_id}:bounded_implement_unavailable")
                continue
            timeout = _task_timeout(fields)
            heartbeat(active_task_id=task_id, state="starting", phase="bounded_implement")
            log_start = log_path.stat().st_size if log_path.is_file() else 0
            exit_code = _run_argv(
                argv,
                repo_root=repo_root,
                cleanup_descendants_on_success=fields.get("resource_class") == "cpu-browser",
                timeout=timeout,
                log_path=log_path,
                heartbeat=lambda: heartbeat(
                    active_task_id=task_id,
                    state="healthy",
                    phase="bounded_implement",
                ),
            )
            turn_exhausted = _slice_turn_exhausted(log_path, log_start)
            _write_bounded_marker(
                marker,
                task_id=task_id,
                attempts=spent + 1,
                ok=False,
                escalation=escalation,
                turn_exhausted=turn_exhausted,
            )
            validation = parse_deterministic_command(
                fields.get("validation") or "",
                repo_root=repo_root,
            )
            validation_code = 1
            if exit_code == 0 and validation is not None:
                validation_code = _run_argv(
                    validation,
                    repo_root=repo_root,
                    cleanup_descendants_on_success=fields.get("resource_class") == "cpu-browser",
                    timeout=_validation_timeout(fields),
                    log_path=log_path,
                    heartbeat=lambda: heartbeat(
                        active_task_id=task_id,
                        state="healthy",
                        phase="validation",
                    ),
                )
            material = _bounded_outputs_material(
                fields.get("validation") or "",
                repo_root=repo_root,
            )
            if exit_code != 0 or validation_code != 0 or not material:
                _write_bounded_marker(
                    marker,
                    task_id=task_id,
                    attempts=spent + 1,
                    ok=False,
                    escalation=escalation,
                    turn_exhausted=turn_exhausted,
                )
                return {
                    "state": "degraded",
                    "phase": "bounded_implement_failed",
                    "active_task_id": task_id,
                    "terminal_reason": "",
                    "backpressure_reasons": [f"{task_id}:bounded_implement_failed"],
                    "busy": False,
                    "work_complete": False,
                    "completion_authority": False,
                }
            _write_bounded_marker(
                marker,
                task_id=task_id,
                attempts=spent + 1,
                ok=True,
                escalation=escalation,
                turn_exhausted=False,
            )
            note = (
                "lane_resume bounded implement validation exit 0 at "
                + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
            try:
                updated = mark_task_completed(
                    todo_path.read_text(encoding="utf-8"),
                    task_id,
                    task_header_prefix=task_header_prefix,
                    note=note,
                )
            except (OSError, ValueError) as exc:
                return {
                    "state": "blocked",
                    "phase": "completion_write_failed",
                    "active_task_id": task_id,
                    "terminal_reason": str(exc),
                    "backpressure_reasons": [f"{task_id}:completion_write_failed"],
                    "busy": False,
                    "work_complete": False,
                    "completion_authority": False,
                }
            _atomic_write(todo_path, updated if updated.endswith("\n") else updated + "\n")
            return {
                "state": "healthy",
                "phase": "task_command_passed",
                "active_task_id": task_id,
                "terminal_reason": "",
                "backpressure_reasons": [],
                "busy": False,
                "advanced_task_id": task_id,
                "work_complete": False,
                "completion_authority": False,
            }
    if not selected:
        return {
            "state": "healthy",
            "phase": "no_selectable_pending",
            "active_task_id": "",
            "terminal_reason": "no_selectable_pending",
            "backpressure_reasons": [],
            "busy": False,
            "work_complete": False,
            "completion_authority": False,
        }
    # An exhausted command has a future retry time. Staying in the process
    # until that time is a wait, not a dead lane the watchdog should resume.
    # Any other skip, including a blank command, keeps the unavailable exit.
    # active_task_id stays empty so a quiet log during the wait is not a
    # progress stall. The log line is the evidence the wait is intentional.
    waiting_ids = _attempt_backoff_ids(skipped, retry_after, now_ms=now_ms)
    waiting_ids = list(dict.fromkeys([*timeout_revalidation_ids, *waiting_ids]))
    if waiting_ids:
        # A failing command can still produce valid evidence. Reuse only a
        # content change observed during its last invocation of this contract.
        for task_id in waiting_ids:
            fields = blocks.get(task_id) or {}
            validation = validation_commands[task_id]
            command = parse_deterministic_command(
                fields.get("deterministic_command") or fields.get("deterministic_repair") or "",
                repo_root=repo_root,
            )
            record = command_evidence.get(task_id) or {}
            if validation is None or command is None or not _produced_evidence_matches(
                record, fields, command, validation,
                repo_root=repo_root, excluded=evidence_exclusions,
            ):
                continue
            held = hold_result()
            if held:
                return held
            if task_id in timeout_revalidation_ids:
                progress = (fixed_points.get(task_id) or {}).get("execution_progress") or {}
                key = hashlib.sha256(json.dumps({
                    "contract": progress.get("contract"), "timeout": _validation_timeout(fields),
                    "workflow": "explicit-validator-budget-v1",
                }, sort_keys=True).encode()).hexdigest()
                fixed_points[task_id]["validation_timeout_recovery"] = {
                    "key": key, "timeout_seconds": _validation_timeout(fields),
                    "reserved_at_ms": int(time.time()*1000),
                    "prior_execution_progress": dict(progress), "producer_reexecuted": False,
                    "completion_authority": False,
                }
                # Reserve before launching, including across interrupts/restarts.
                save_attempts()
            validation_code = _run_argv(
                validation,
                repo_root=repo_root,
                cleanup_descendants_on_success=fields.get("resource_class") == "cpu-browser",
                timeout=_validation_timeout(fields),
                log_path=log_path,
                heartbeat=lambda **kwargs: heartbeat(active_task_id=task_id, **kwargs),
            )
            held = hold_result()
            if held:
                return held
            record_retry_contract(task_id, fields, command, validation)
            if validation_code != 0:
                _finish_execution_progress(fixed_points, task_id, code=validation_code, stage="validation",
                                           contract=_command_contract(fields, command, validation, repo_root=repo_root),
                                           now_ms=int(time.time() * 1000))
                save_attempts()
                continue
            held = hold_result()
            if held:
                return held
            # The validator and other workers may change the board or evidence.
            # Recheck both before committing validation-only completion.
            current_todo = todo_path.read_text(encoding="utf-8")
            current_fields = dict(parse_todo_task_fields(
                current_todo, task_header_prefix=task_header_prefix,
            )).get(task_id) or {}
            if not _produced_evidence_matches(
                record, current_fields, command, validation,
                repo_root=repo_root, excluded=evidence_exclusions,
            ):
                continue
            note = (
                "lane_resume validation already passed during attempt backoff at "
                + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
            try:
                updated = mark_task_completed(
                    current_todo,
                    task_id,
                    task_header_prefix=task_header_prefix,
                    note=note,
                )
            except (OSError, ValueError) as exc:
                return {
                    "state": "blocked",
                    "phase": "completion_write_failed",
                    "active_task_id": task_id,
                    "terminal_reason": str(exc),
                    "backpressure_reasons": [f"{task_id}:completion_write_failed"],
                    "busy": False,
                    "work_complete": False,
                    "completion_authority": False,
                }
            retry_after.pop(task_id, None)
            save_attempts()
            _atomic_write(todo_path, updated if updated.endswith("\n") else updated + "\n")
            _finish_execution_progress(fixed_points, task_id, code=0, stage="validation",
                                       contract=_command_contract(fields, command, validation, repo_root=repo_root),
                                       now_ms=int(time.time() * 1000), validated=True)
            save_attempts()
            return {
                "state": "healthy",
                "phase": "task_command_passed",
                "active_task_id": task_id,
                "terminal_reason": "",
                "backpressure_reasons": [],
                "busy": False,
                "advanced_task_id": task_id,
                "work_complete": False,
                "completion_authority": False,
            }
        with open_command_log(log_path, "retry waiting " + ",".join(waiting_ids) + "\n"):
            pass
        return {
            "state": "degraded",
            "phase": "semantic_failure_wait" if semantic_waits else "attempt_retry_wait",
            "active_task_id": semantic_waits[0] if semantic_waits else "",
            "terminal_reason": "",
            "backpressure_reasons": skipped[:32],
            "next_retry_at_ms": min(retry_after.get(identity, now_ms + _EXHAUSTED_RETRY_MS) for identity in waiting_ids),
            "busy": False,
            "work_complete": False,
            "completion_authority": False,
        }
    if semantic_waits:
        return {"state": "degraded", "phase": "semantic_failure_wait", "active_task_id": semantic_waits[0],
                "terminal_reason": "unchanged_semantic_failure_budget_exhausted", "backpressure_reasons": skipped[:32],
                "busy": False, "work_complete": False, "completion_authority": False}
    if gate_waits:
        return gate_waits[0]
    return {
        "state": "blocked",
        "phase": "deterministic_resume_unavailable",
        "active_task_id": "",
        "terminal_reason": "deterministic_resume_unavailable",
        "backpressure_reasons": skipped[:32],
        "busy": False,
        "work_complete": False,
        "completion_authority": False,
    }


def write_lane_status(
    path: Path,
    *,
    target_id: str,
    state: str,
    phase: str,
    active_task_id: str = "",
    terminal_reason: str = "",
    backpressure_reasons: Sequence[str] = (),
    transition_at_ms: int | None = None,
    refill_state: str = "idle",
    gate_evidence_fingerprint: str = "",
    repair_packet: str = "",
    next_retry_at_ms: int = 0,
    gate_recovery_needed: bool = False,
    execution_progress: Mapping[str, Any] | None = None,
    lane_health: Mapping[str, Any] | None = None,
) -> None:
    now_ms = int(time.time() * 1000)
    payload = {
        "schema": _SCHEMA,
        "target_id": target_id,
        "state": state,
        "phase": phase,
        "heartbeat_at_ms": now_ms,
        "pid": os.getpid(),
        "active_task_id": active_task_id,
        "last_task_transition_at_ms": transition_at_ms if transition_at_ms is not None else now_ms,
        "active_leases": [],
        "active_lease_count": 0,
        "refill_state": refill_state or "idle",
        "backpressure": bool(backpressure_reasons) or state in {"blocked", "degraded"},
        "backpressure_reasons": list(backpressure_reasons)[:32],
        "terminal_reason": terminal_reason,
        "transition_id": "",
        "generation": 1,
        "fencing_epoch": 0,
        "updated_at_ms": now_ms,
        "work_complete": False,
        "completion_authority": False,
    }
    if gate_evidence_fingerprint:
        payload.update(
            gate_evidence_fingerprint=gate_evidence_fingerprint,
            repair_packet=repair_packet,
            next_retry_at_ms=int(next_retry_at_ms),
            gate_recovery_needed=bool(gate_recovery_needed),
        )
    if execution_progress:
        payload["execution_progress"] = dict(execution_progress)
        payload["unresolved_prior_failure"] = execution_progress.get("unresolved_prior_failure") is True
        if payload["unresolved_prior_failure"]:
            payload["unresolved_prior_failure_reason"] = "prior_semantic_failure_not_validated"
    if lane_health:
        payload["lane_execution_health"] = dict(lane_health)
        payload["lane_unresolved_prior_failure"] = lane_health.get("unresolved_prior_failure") is True
    if next_retry_at_ms:
        payload["next_retry_at_ms"] = int(next_retry_at_ms)
    _atomic_write(path, json.dumps(payload, indent=2) + "\n")


def _pid_alive(pid: int) -> bool:
    from .supervisor_watchdog import pid_alive

    return bool(pid_alive(pid))


def _master_alive(master_pid_path: str, repo_root: Path) -> bool:
    if not master_pid_path:
        return False
    path = Path(master_pid_path)
    if not path.is_absolute():
        path = repo_root / path
    try:
        pid = int(path.read_text(encoding="utf-8").strip().split()[0])
    except (OSError, IndexError, TypeError, ValueError):
        return False
    return pid > 0 and _pid_alive(pid)


def _resume_lock_path(status_path: Path, bundle_key: str) -> Path:
    return status_path.with_name(f"{bundle_key.replace('/', '_')}.resume.lock")


def _validate_resume_lock_fd(descriptor: int, lock_path: Path) -> None:
    """Validate and reassert the inherited open-file description's fence.

    The caller must close its copy normally after handoff, never LOCK_UN: an
    explicit unlock would also release the child's shared flock description.
    """
    if type(descriptor) is not int or descriptor < 3:
        raise ValueError("invalid resume lock descriptor")
    actual = os.fstat(descriptor)
    expected = lock_path.stat(follow_symlinks=False)
    flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    if (not stat.S_ISREG(actual.st_mode) or not stat.S_ISREG(expected.st_mode)
            or (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)
            or (flags & os.O_ACCMODE) not in {os.O_WRONLY, os.O_RDWR}
            or flags & getattr(os, "O_PATH", 0)):
        raise ValueError("resume lock descriptor does not match writable lifecycle fence")
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _await_resume_ready(
    process: subprocess.Popen[Any], ready_fd: int, *, status_path: Path,
    pid_path: Path, bundle_key: str, observe: Callable[[], None] = lambda: None,
) -> str:
    """A spawned pid alone is not proof that the lane acquired its fence."""
    deadline = time.monotonic() + _RESUME_READY_SECONDS
    received = b""
    while time.monotonic() < deadline:
        observe()
        readable, _, _ = select.select([ready_fd], [], [], min(0.05, max(0.0, deadline - time.monotonic())))
        if readable:
            chunk = os.read(ready_fd, 4096)
            received += chunk
            if len(received) > 4096:
                return "resume_readiness_invalid"
            if b"\n" in received:
                try:
                    ready = json.loads(received)
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    published_pid = int(pid_path.read_text(encoding="utf-8").strip())
                    if (ready.get("pid") == process.pid == published_pid == status.get("pid")
                            and ready.get("bundle_key") == bundle_key == status.get("target_id")
                            and 0 < int(ready["heartbeat_at_ms"]) <= int(status["heartbeat_at_ms"])
                            and status.get("state") != "stopped" and process.poll() is None):
                        return ""
                except (OSError, ValueError, TypeError, KeyError, AttributeError):
                    pass
                return "resume_readiness_invalid"
            if not chunk:
                return "resume_exited_before_ready"
        if process.poll() is not None:
            return "resume_exited_before_ready"
    return "resume_readiness_timeout"


def resume_stopped_lane(
    lane_info: Mapping[str, Any],
    *,
    repo_root: Path,
    manifest: Mapping[str, Any] | None = None,
    resume_lock_fd: int | None = None,
) -> dict[str, Any]:
    """Start one fenced deterministic lane process.

    A live lane pid or a live master refuses the spawn. Provider campaign
    modules are not a resume command. The result does not grant completion.
    """

    root = Path(repo_root)
    manifest = manifest or {}
    pid_path = Path(str(lane_info.get("pid_path") or ""))
    if pid_path and not pid_path.is_absolute():
        pid_path = root / pid_path
    if pid_path:
        try:
            existing = int(pid_path.read_text(encoding="utf-8").strip().split()[0])
        except (OSError, IndexError, TypeError, ValueError):
            existing = 0
        if existing > 0 and _pid_alive(existing):
            return {
                "restarted": False,
                "reason": "live_lane_pid",
                "control_recovery_required": True,
                "work_complete": False,
                "completion_authority": False,
            }
    master_raw = str(
        lane_info.get("master_pid_path") or manifest.get("master_pid_path") or ""
    )
    if _master_alive(master_raw, root):
        return {
            "restarted": False,
            "reason": "live_master_pid",
            "control_recovery_required": True,
            "work_complete": False,
            "completion_authority": False,
        }
    state_dir = Path(str(lane_info.get("state_dir") or pid_path.parent))
    if not state_dir.is_absolute():
        state_dir = root / state_dir
    todo_raw = str(lane_info.get("todo_path") or manifest.get("todo_path") or "")
    prefix = str(
        lane_info.get("task_header_prefix") or manifest.get("task_header_prefix") or ""
    )
    if not todo_raw or not prefix:
        return {
            "restarted": False,
            "reason": "resume_board_unbound",
            "control_recovery_required": True,
            "work_complete": False,
            "completion_authority": False,
        }
    status_prefix = str(lane_info.get("state_prefix") or "lane")
    status_path = state_dir / f"{status_prefix}_status.json"
    bundle_key = str(lane_info.get("bundle_key") or "lane")
    if resume_lock_fd is not None:
        try:
            _validate_resume_lock_fd(resume_lock_fd, _resume_lock_path(status_path, bundle_key))
        except (OSError, ValueError):
            return {"restarted": False, "reason": "resume_lock_invalid", "readiness_verified": False,
                    "work_complete": False, "completion_authority": False}
    log_raw = str(lane_info.get("log_path") or "")
    log_path = Path(log_raw) if log_raw else state_dir / "supervisor.log"
    if not log_path.is_absolute():
        log_path = root / log_path
    if not pid_path:
        pid_path = state_dir / f"{status_prefix}_bundle_supervisor.pid"
    command = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.rescue.lane_resume",
        "--repo-root",
        str(root),
        "--todo-path",
        todo_raw,
        "--task-header-prefix",
        prefix,
        "--status-path",
        str(status_path),
        "--pid-path",
        str(pid_path),
        "--log-path",
        str(log_path),
        "--bundle-key",
        bundle_key,
        "--max-seconds",
        str(int(lane_info.get("max_resume_seconds") or manifest.get("max_resume_seconds") or 28800)),
    ]
    if manifest.get("allow_bounded_task_implement") is True:
        command.append("--allow-bounded-implement")
    minimum_free = lane_info.get("minimum_free_bytes", manifest.get("storage_minimum_free_bytes"))
    storage_paths = lane_info.get("storage_paths", manifest.get("storage_paths", []))
    if (minimum_free is not None and (type(minimum_free) is not int or minimum_free < 1)
            or not isinstance(storage_paths, list) or any(not isinstance(path, str) or not path for path in storage_paths)):
        return {"restarted": False, "reason": "invalid_storage_policy", "readiness_verified": False,
                "work_complete": False, "completion_authority": False}
    if minimum_free is not None:
        command.extend(["--minimum-free-bytes", str(minimum_free)])
    for raw_path in storage_paths:
        path = Path(raw_path)
        command.extend(["--storage-path", str(path if path.is_absolute() else root / path)])
    if any(part in _PROVIDER_BASENAMES or "implementation_supervisor" in part or "multi_supervisor_runner" in part for part in command):
        return {
            "restarted": False,
            "reason": "provider_campaign_refused",
            "work_complete": False,
            "completion_authority": False,
        }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    ready_read, ready_write = os.pipe()
    command.extend(["--ready-fd", str(ready_write)])
    if resume_lock_fd is not None:
        command.extend(["--resume-lock-fd", str(resume_lock_fd)])
    inherited = (ready_write,) if resume_lock_fd is None else (ready_write, resume_lock_fd)
    handle = None
    process = None
    descendants = None
    reason = "resume_startup_interrupted"
    try:
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=inherited,
        )
        os.close(ready_write)
        ready_write = -1
        descendants = CommandProcessTree(process.pid)
        reason = _await_resume_ready(
            process, ready_read, status_path=status_path, pid_path=pid_path,
            bundle_key=bundle_key, observe=descendants.snapshot,
        )
        if reason:
            return {"restarted": False, "reason": reason, "readiness_verified": False,
                    "work_complete": False, "completion_authority": False}
    except OSError as exc:
        return {
            "restarted": False,
            "reason": f"resume_spawn_failed: {exc}",
            "readiness_verified": False,
            "work_complete": False,
            "completion_authority": False,
        }
    finally:
        if handle is not None:
            handle.close()
        os.close(ready_read)
        if ready_write >= 0:
            os.close(ready_write)
        if descendants is not None:
            try:
                if reason:
                    descendants.snapshot()
                    descendants.send_signal(signal.SIGTERM)
                    if process.poll() is None:
                        _stop_command_group(process)
                    descendants.send_signal(signal.SIGKILL)
            finally:
                descendants.close()
    return {
        "restarted": True,
        "new_pid": int(process.pid),
        "reason": "resume_stopped_lane",
        "pid_persisted": True,
        "readiness_verified": True,
        "lock_handoff": resume_lock_fd is not None,
        "work_complete": False,
        "completion_authority": False,
    }


def serve(argv: Sequence[str] | None = None, *, between_cycles: Callable[[], Mapping[str, Any] | None] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic stopped-lane resume")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--todo-path", required=True)
    parser.add_argument("--task-header-prefix", required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--pid-path", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--bundle-key", required=True)
    parser.add_argument("--max-seconds", type=float, default=28800)
    parser.add_argument("--allow-bounded-implement", action="store_true")
    parser.add_argument("--resume-lock-fd", type=int)
    parser.add_argument("--ready-fd", type=int)
    parser.add_argument("--minimum-free-bytes", type=int, default=MINIMUM_FREE_BYTES)
    parser.add_argument("--storage-path", type=Path, action="append", default=[])
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo_root = args.repo_root.resolve()
    todo_path = Path(args.todo_path)
    if not todo_path.is_absolute():
        todo_path = repo_root / todo_path
    status_path = args.status_path
    pid_path = args.pid_path
    log_path = args.log_path
    lock_path = _resume_lock_path(status_path, args.bundle_key)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.resume_lock_fd is not None:
            _validate_resume_lock_fd(args.resume_lock_fd, lock_path)
            os.set_inheritable(args.resume_lock_fd, False)
            lock_handle = os.fdopen(args.resume_lock_fd, "a", encoding="utf-8")
        else:
            lock_handle = lock_path.open("a", encoding="utf-8")
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.ready_fd is not None:
            ready_flags = fcntl.fcntl(args.ready_fd, fcntl.F_GETFL)
            if (args.ready_fd < 3 or not stat.S_ISFIFO(os.fstat(args.ready_fd).st_mode)
                    or (ready_flags & os.O_ACCMODE) != os.O_WRONLY):
                return 2
            os.set_inheritable(args.ready_fd, False)
    except (OSError, ValueError):
        return 0
    if pid_path.exists():
        try:
            existing = int(pid_path.read_text(encoding="utf-8").strip().split()[0])
        except (OSError, IndexError, TypeError, ValueError):
            existing = 0
        if existing > 0 and existing != os.getpid() and _pid_alive(existing):
            return 0
    if args.minimum_free_bytes < 1:
        parser.error("--minimum-free-bytes must be positive")
    running = True

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    deadline = time.monotonic() + max(1.0, float(args.max_seconds))
    transition_at_ms = int(time.time() * 1000)
    try:
        previous_status = json.loads(status_path.read_text(encoding="utf-8"))
        previous_transition = int(previous_status.get("last_task_transition_at_ms") or 0)
        if previous_status.get("target_id") == str(args.bundle_key) and 0 < previous_transition <= transition_at_ms:
            transition_at_ms = previous_transition
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    attempt_path = status_path.with_name("lane_resume_attempts.json")

    def check_continue() -> None:
        if not running or time.monotonic() >= deadline:
            raise _LaneResumeInterrupted

    storage_paths = [repo_root, todo_path, status_path, log_path, Path(tempfile.gettempdir()),
                     *(path if path.is_absolute() else repo_root / path for path in args.storage_path)]
    storage_guard = StorageGuard(status_path, storage_paths, check_continue=check_continue,
                                 minimum_free_bytes=args.minimum_free_bytes)
    storage_context = storage_guard.activate()
    storage_context.__enter__()

    def heartbeat(
        active_task_id: str = "",
        state: str = "healthy",
        phase: str = "lane_resume",
        terminal_reason: str = "",
        backpressure_reasons: Sequence[str] = (),
        refill_state: str = "idle",
        gate_evidence_fingerprint: str = "",
        repair_packet: str = "",
        next_retry_at_ms: int = 0,
        gate_recovery_needed: bool = False,
    ) -> None:
        if not running or time.monotonic() >= deadline:
            raise _LaneResumeInterrupted
        progress = execution_progress_snapshot(attempt_path, active_task_id) if active_task_id else {}
        if progress and state in {"healthy", "starting"} and (
                progress.get("unresolved_prior_failure") is True
                or int(progress.get("semantic_failures") or 0) > 0
                or int(progress.get("abandoned_attempts") or 0) >= _ATTEMPT_LIMIT):
            state = "degraded"
            backpressure_reasons = (*backpressure_reasons, active_task_id + ":prior_semantic_failure")
        health = lane_execution_health(repo_root=repo_root, todo_path=todo_path,
                                       task_header_prefix=str(args.task_header_prefix), attempt_path=attempt_path)
        if health["health"] == "degraded" and state in {"healthy", "starting"}:
            state = "degraded"
            backpressure_reasons = (*backpressure_reasons, "lane:" + (
                "pending_semantic_failures" if health["pending_failure_count"] else "failure_history_unavailable"))
        write_lane_status(
            status_path,
            target_id=str(args.bundle_key),
            state=state,
            phase=phase,
            active_task_id=active_task_id,
            terminal_reason=terminal_reason,
            backpressure_reasons=backpressure_reasons,
            transition_at_ms=transition_at_ms,
            refill_state=refill_state,
            gate_evidence_fingerprint=gate_evidence_fingerprint,
            repair_packet=repair_packet,
            next_retry_at_ms=next_retry_at_ms,
            gate_recovery_needed=gate_recovery_needed,
            execution_progress=progress,
            lane_health=health,
        )

    try:
        _atomic_write(pid_path, f"{os.getpid()}\n")
        heartbeat(state="starting", phase="lane_resume_starting")
        if args.ready_fd is not None:
            ready_status = json.loads(status_path.read_text(encoding="utf-8"))
            try:
                os.write(args.ready_fd, (json.dumps({
                    "pid": os.getpid(), "bundle_key": args.bundle_key,
                    "heartbeat_at_ms": ready_status["heartbeat_at_ms"],
                }) + "\n").encode("utf-8"))
            finally:
                os.close(args.ready_fd)
        while running and time.monotonic() < deadline:
            capacity = storage_guard.admission()
            if not capacity["ok"]:
                heartbeat(state="degraded", phase="storage_wait",
                          backpressure_reasons=("storage_pressure:" + capacity["reason"],))
                time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
                continue
            if between_cycles is not None:
                try:
                    boundary = between_cycles() or {}
                    if boundary.get("state") == "refused":
                        raise ValueError(str(boundary.get("reason") or "callback_refused"))
                except Exception as error:
                    heartbeat(state="degraded", phase="between_cycles_refused",
                              backpressure_reasons=("between_cycles:" + type(error).__name__ + ":" + " ".join(str(error).split())[:240],))
                    time.sleep(min(5.0, max(0.0, deadline-time.monotonic())))
                    continue
            started = time.monotonic()
            cycle = run_resume_cycle(
                repo_root=repo_root,
                todo_path=todo_path,
                task_header_prefix=str(args.task_header_prefix),
                attempt_path=attempt_path,
                log_path=log_path,
                heartbeat=heartbeat,
                allow_bounded_implement=bool(args.allow_bounded_implement),
            )
            if cycle.get("advanced_task_id") or cycle.get("evidence_changed") or cycle.get("phase") == "gap_refilled":
                transition_at_ms = int(time.time() * 1000)
            heartbeat(
                active_task_id=str(cycle.get("active_task_id") or ""),
                state=str(cycle.get("state") or "blocked"),
                phase=str(cycle.get("phase") or "lane_resume"),
                terminal_reason=str(cycle.get("terminal_reason") or ""),
                backpressure_reasons=tuple(cycle.get("backpressure_reasons") or ()),
                refill_state=str(cycle.get("refill_state") or "idle"),
                gate_evidence_fingerprint=str(cycle.get("gate_evidence_fingerprint") or ""),
                repair_packet=str(cycle.get("repair_packet") or ""),
                next_retry_at_ms=int(cycle.get("next_retry_at_ms") or 0),
                gate_recovery_needed=bool(cycle.get("gate_recovery_needed")),
            )
            # A live heartbeat in this phase hides a block from the dead-lane
            # resume. Exit and leave the blocked status so the watchdog can see
            # a dead lane with pending work. gate_withheld stays in the loop.
            if str(cycle.get("phase") or "") == "deterministic_resume_unavailable":
                return 0
            if not running:
                break
            remaining = 30.0 - (time.monotonic() - started)
            while remaining > 0 and running and time.monotonic() < deadline:
                time.sleep(min(5.0, remaining, max(0.0, deadline - time.monotonic())))
                remaining = 30.0 - (time.monotonic() - started)
                heartbeat(
                    active_task_id=str(cycle.get("active_task_id") or ""),
                    state=str(cycle.get("state") or "blocked"),
                    phase=str(cycle.get("phase") or "lane_resume"),
                    terminal_reason=str(cycle.get("terminal_reason") or ""),
                    backpressure_reasons=tuple(cycle.get("backpressure_reasons") or ()),
                    refill_state=str(cycle.get("refill_state") or "idle"),
                    gate_evidence_fingerprint=str(cycle.get("gate_evidence_fingerprint") or ""),
                    repair_packet=str(cycle.get("repair_packet") or ""),
                    next_retry_at_ms=int(cycle.get("next_retry_at_ms") or 0),
                    gate_recovery_needed=bool(cycle.get("gate_recovery_needed")),
                )
    except _LaneResumeInterrupted:
        pass
    finally:
        storage_context.__exit__(None, None, None)
    try:
        write_lane_status(
            status_path,
            target_id=str(args.bundle_key),
            state="stopped",
            phase="lane_resume_exited",
            terminal_reason="resume_duration_elapsed" if running else "resume_stopped",
            transition_at_ms=transition_at_ms,
        )
    except OSError as error:
        if not is_storage_pressure(error):
            raise
        publish_emergency_status(status_path, {"pid": os.getpid(), "state": "stopped", "phase": "lane_resume_exited",
            "terminal_reason": "resume_duration_elapsed" if running else "resume_stopped"}, error=error)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return serve(argv)


if __name__ == "__main__":
    raise SystemExit(main())
