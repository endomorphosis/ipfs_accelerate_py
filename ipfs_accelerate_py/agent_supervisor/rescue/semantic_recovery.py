"""Finite, foreground-owned repair admission for exhausted producer contracts.

This journal grants neither task completion nor arbitrary edit authority. A
declared repository worker must enforce its inherited checkout lease and edit
scope. Only a subsequent original producer/validator can establish completion.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .gate_recovery import evidence_file_fingerprint

MAX_RECOVERIES = 2
MAX_JOURNAL_BYTES = 4 * 1024 * 1024
CONTEXT_ENV = "AGENT_SUPERVISOR_RECOVERY_CONTEXT_PATH"


def is_authored_scenario_source(path: Path) -> bool:
    """Declared e2e workflow JSON is source, not generated proof evidence.

    Classification only binds hashes; the worker's exact edit allowlist and
    independent original validator still determine whether an edit is allowed.
    Missing declared scenarios retain a None hash until genuinely produced.
    """
    if path.suffix.lower() != ".json" or path.is_absolute() or ".." in path.parts:
        return False
    if {"data", "test-results", "evidence", "logs", "reports", "model-inputs", "model-invocations", ".git"}.intersection(path.parts):
        return False
    if path.name.lower() in {"receipt.json", "report.json", "status.json", "feedback.json"} or re.search(r"\.(?:progress|receipt|report|feedback)\.json$", path.name, re.I):
        return False
    return any(path.parts[index:index + 3] in (("test", "e2e", "functional-workflows"),
                                              ("tests", "e2e", "functional-workflows"))
               for index in range(max(0, len(path.parts) - 3)))


class SemanticRecoveryError(ValueError):
    """A repair cannot safely be admitted; the failure is actionable."""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def journal_path(attempt_path: Path) -> Path:
    return attempt_path.with_name(attempt_path.stem + ".semantic-recovery.json")


def _regular(root: Path, raw: str, *, required: bool = True) -> Path:
    name = Path(raw)
    if not raw or name.is_absolute() or ".." in name.parts:
        raise SemanticRecoveryError("recovery_path_not_repo_relative")
    candidate = root / name
    try:
        candidate.resolve().relative_to(root.resolve())
        for item in [candidate, *candidate.parents]:
            if item == root:
                break
            if item.is_symlink():
                raise SemanticRecoveryError("recovery_symlink_refused")
        if candidate.exists():
            if not stat.S_ISREG(candidate.stat().st_mode):
                raise SemanticRecoveryError("recovery_input_not_regular")
        elif required:
            raise SemanticRecoveryError("recovery_input_missing")
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, SemanticRecoveryError):
            raise
        raise SemanticRecoveryError("recovery_path_unavailable") from exc
    return candidate


def file_inputs(root: Path, raw: str) -> dict[str, str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names or len(names) > 32 or len(set(names)) != len(names):
        raise SemanticRecoveryError("recovery_inputs_missing_or_unbounded")
    result = {}
    total = 0
    for name in names:
        path = _regular(root, name)
        size = path.stat().st_size
        total += size
        if size > 4 * 1024 * 1024 or total > 16 * 1024 * 1024:
            raise SemanticRecoveryError("recovery_inputs_too_large")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def fixture_inputs(root: Path, fields: Mapping[str, str]) -> dict[str, str]:
    """Explicit production fixtures are distinct from retained proof artifacts."""
    raw = str(fields.get("semantic_recovery_fixture_inputs") or "").strip()
    if not raw:
        return {}
    predicted = {part.strip() for part in str(fields.get("predicted_files") or "").split(",") if part.strip()}
    execution = {part.strip() for part in str(fields.get("execution_inputs") or "").split(",") if part.strip()}
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if len(names) > 8 or len(names) != len(set(names)) or not set(names) <= predicted & execution:
        raise SemanticRecoveryError("recovery_fixture_must_be_predicted_execution_input")
    result = {}
    for name in names:
        item = Path(name)
        if (item.suffix.lower() != ".json" or "fixtures" not in item.parts
                or {"logs", "test-results", "evidence", "model-inputs", "model-invocations"}.intersection(item.parts)
                or re.search(r"(?:progress|receipt|feedback|rejected|log)", item.name, re.I)):
            raise SemanticRecoveryError("recovery_fixture_is_proof_or_not_json_fixture")
        path = _regular(root, name)
        if path.stat().st_size > 65536:
            raise SemanticRecoveryError("recovery_fixture_too_large")
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError
        except (OSError, ValueError) as exc:
            raise SemanticRecoveryError("recovery_fixture_invalid_json_object") from exc
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def policy(fields: Mapping[str, str], *, root: Path, task_id: str,
           parse_command: Callable[..., list[str] | None], task_timeout: float) -> dict[str, Any] | None:
    raw = str(fields.get("semantic_recovery_command") or "").strip()
    if not raw:
        return None
    argv = parse_command(raw, repo_root=root)
    if (not argv or len(argv) < 3 or Path(argv[0]).name not in {"python", "python3", "node"}
            or Path(argv[1]).suffix not in {".py", ".js", ".mjs", ".cjs"} or task_id not in argv):
        raise SemanticRecoveryError("recovery_requires_task_scoped_repo_script")
    script = _regular(root, argv[1])
    try:
        limit = int(str(fields.get("semantic_recovery_limit") or "1"))
        timeout = float(str(fields.get("semantic_recovery_timeout_seconds") or min(task_timeout, 300)))
    except (TypeError, ValueError) as exc:
        raise SemanticRecoveryError("recovery_bounds_invalid") from exc
    if not 1 <= limit <= MAX_RECOVERIES or not math.isfinite(timeout) or not 0 < timeout <= min(task_timeout, 1200):
        raise SemanticRecoveryError("recovery_bounds_invalid")
    inputs = file_inputs(root, str(fields.get("semantic_recovery_inputs") or ""))
    return {"argv": argv, "limit": limit, "timeout": timeout, "inputs": inputs,
            "qualification_inputs": {name: evidence_file_fingerprint(root / name) for name in inputs},
            "policy_fingerprint": digest({"argv": argv, "limit": limit, "timeout": timeout,
                                           "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()})}


class Journal:
    def __init__(self, path: Path, persist: Callable[[Path, str], None]):
        self.path, self.persist = path, persist
        self.data: dict[str, Any] = {"schema": "agent-supervisor/semantic-recovery-journal@1", "tasks": {}}
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JOURNAL_BYTES:
            raise SemanticRecoveryError("recovery_journal_invalid")
        try:
            value = json.loads(path.read_text())
            if (not isinstance(value, dict) or value.get("schema") != self.data["schema"]
                    or not isinstance(value.get("tasks"), dict) or len(value["tasks"]) > 4096):
                raise ValueError
            for item in value["tasks"].values():
                if (not isinstance(item, dict) or type(item.get("spent")) is not int
                        or not 0 <= item["spent"] <= MAX_RECOVERIES
                        or not isinstance(item.get("history"), list) or len(item["history"]) != item["spent"]):
                    raise ValueError
                if any(not isinstance(row, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("qualification_fingerprint") or ""))
                       for row in item["history"]):
                    raise ValueError
            self.data = value
        except (OSError, ValueError, TypeError) as exc:
            raise SemanticRecoveryError("recovery_journal_invalid") from exc

    def save(self) -> None:
        text = json.dumps(self.data, sort_keys=True, indent=2) + "\n"
        if len(text.encode()) > MAX_JOURNAL_BYTES:
            raise SemanticRecoveryError("recovery_journal_too_large")
        self.persist(self.path, text)

    def summary(self, task_id: str) -> dict[str, Any]:
        task = self.data["tasks"].get(task_id) or {}
        last = (task.get("history") or [{}])[-1]
        return {"spent": task.get("spent", 0), **{key: last[key] for key in (
            "state", "reason", "producer_contract", "blocked_contract", "approved_contract",
            "qualification_fingerprint", "context_path", "exit_code", "started_at_ms", "finished_at_ms", "timeout_seconds", "recovery_limit", "source_after",
        ) if key in last}, "completion_authority": False}

    def reconcile(self, task_id: str, contract: str, now_ms: int, source: Mapping[str, Any] | None = None) -> None:
        task = self.data["tasks"].get(task_id) or {}
        last = (task.get("history") or [{}])[-1]
        if last.get("state") == "running":
            last.update(state="interrupted", reason="recovery_owner_interrupted", blocked_contract=contract, finished_at_ms=now_ms)
            if source is not None:
                last["source_after"] = dict(source)
            self.save()

    def admit(self, task_id: str, *, contract: str, fingerprint: str, limit: int,
              context_path: str, now_ms: int) -> dict[str, Any]:
        task = self.data["tasks"].setdefault(task_id, {"spent": 0, "history": []})
        if task["spent"] >= limit:
            raise SemanticRecoveryError("recovery_budget_exhausted")
        if any(row["qualification_fingerprint"] == fingerprint for row in task["history"]):
            raise SemanticRecoveryError("recovery_qualification_unchanged")
        row = {"state": "running", "producer_contract": contract, "qualification_fingerprint": fingerprint,
               "context_path": context_path, "started_at_ms": now_ms, "completion_authority": False}
        task["spent"] += 1
        task["history"].append(row)
        self.save()  # Spend before starting a process, including on interruption.
        return row


def summary(attempt_path: Path, task_id: str) -> dict[str, Any]:
    return Journal(journal_path(attempt_path), lambda *_: None).summary(task_id)
