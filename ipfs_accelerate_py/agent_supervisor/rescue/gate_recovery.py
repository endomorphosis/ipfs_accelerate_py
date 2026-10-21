"""Durable semantic evidence for a repeatedly withheld deterministic gate.

This ledger diagnoses missing work; it never changes a task's status or a
gate's decision. Its repair packet is consumed by the fenced repair dispatcher.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any, Mapping

GATE_REPEAT_LIMIT = 3
GATE_RETRY_BASE_MS = 60_000
GATE_RETRY_MAX_MS = 900_000
_SOURCE_SUFFIXES = {".py", ".js", ".cjs", ".mjs", ".ts", ".tsx", ".jsx"}
_VOLATILE_KEYS = {
    "generated_at", "generated_at_ms", "generated_at_utc", "updated_at",
    "updated_at_ms", "created_at", "created_at_ms", "timestamp", "timestamp_ms",
    "heartbeat_at_ms", "last_seen", "last_seen_ms", "checked_at", "checked_at_ms",
    "duration_ms", "duration_seconds", "elapsed_ms", "elapsed_seconds", "pid",
    "log_path", "log_file", "stdout", "stderr",
    "logs", "log", "log_bytes", "log_size", "log_size_bytes", "mtime", "mtime_ns",
    "last_checked_at", "last_checked_at_ms", "observed_at", "observed_at_ms",
}
_VOLATILE_NORMALIZED = {re.sub(r"[^a-z0-9]", "", key.casefold()) for key in _VOLATILE_KEYS}
_FIELDS = (
    "status", "depends_on", "deterministic_command", "deterministic_repair",
    "validation", "outputs", "predicted_files", "acceptance", "conflict_policy",
)


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable(item) for key, item in value.items()
                if re.sub(r"[^a-z0-9]", "", str(key).casefold()) not in _VOLATILE_NORMALIZED}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def evidence_file_fingerprint(path: Path) -> str:
    """Fingerprint file meaning without treating regenerated JSON clocks as work."""
    try:
        body = path.read_bytes()
    except OSError:
        return "absent"
    if path.suffix.casefold() == ".json":
        try:
            body = _json(_stable(json.loads(body))).encode("utf-8")
        except (ValueError, UnicodeError):
            pass
    return hashlib.sha256(body).hexdigest()


def _artifact_summary(value: Any) -> dict[str, Any]:
    """Keep gate diagnostics without copying arbitrary artifact rows or bodies."""

    result: dict[str, Any] = {"semantic_sha256": hashlib.sha256(_json(value).encode("utf-8")).hexdigest()}
    if not isinstance(value, dict):
        return result
    for key in ("schema", "decision", "passed", "missing", "sha256"):
        item = value.get(key)
        if isinstance(item, (str, bool, int, float)):
            result[key] = item[:240] if isinstance(item, str) else item
    reasons = value.get("reasons")
    if isinstance(reasons, list):
        result["reason_count"] = len(reasons)
        result["reason_summaries"] = [reason[:240] for reason in reasons[:20] if isinstance(reason, str)]
    for key, item in value.items():
        if (key.endswith("_count") or key.endswith("_counts")) and isinstance(item, (int, float)) and not isinstance(item, bool):
            result[key] = item
        elif key.endswith("_counts") and isinstance(item, dict):
            result[key] = {str(name): count for name, count in list(item.items())[:50]
                           if isinstance(count, (int, float)) and not isinstance(count, bool)}
    return result


def dependency_ids(fields: Mapping[str, str]) -> list[str]:
    return [part for part in re.split(r"[,\s]+", str(fields.get("depends_on") or ""))
            if part and part.casefold() not in {"none", "n/a", "-"}]


def dependency_closure(task_id: str, blocks: Mapping[str, Mapping[str, str]]) -> list[str]:
    """Return direct dependencies before ancestors, tolerating a malformed cycle."""

    pending = dependency_ids(blocks.get(task_id) or {})
    result: list[str] = []
    seen = {task_id}
    while pending and len(result) < 256:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        result.append(current)
        pending.extend(dependency_ids(blocks.get(current) or {}))
    return result


def _referenced_paths(fields: Mapping[str, str], repo_root: Path) -> list[Path]:
    values: list[str] = []
    for key in ("outputs", "predicted_files"):
        values.extend(str(fields.get(key) or "").split(","))
    for key in ("deterministic_command", "deterministic_repair", "validation"):
        command = str(fields.get(key) or "")
        try:
            args = shlex.split(command)
        except ValueError:
            args = []
        values.extend(arg for arg in args[1:] if not any(c.isspace() for c in arg))
        # Python -c validators often name their evidence inside a string literal.
        for arg in args[1:]:
            values.extend(match[1] for match in re.findall(r"(['\"])([^'\"\n]+)\1", arg))
        if "-c" in args and Path(args[0]).name.startswith("python"):
            try:
                code = ast.parse(args[args.index("-c") + 1])
            except (IndexError, SyntaxError):
                pass
            else:
                values.extend(node.value for node in ast.walk(code)
                              if isinstance(node, ast.Constant) and isinstance(node.value, str))
    root = repo_root.resolve()
    result: list[Path] = []
    for value in values:
        value = value.strip()
        if not value or value.startswith("-") or any(c in value for c in "\n\x00"):
            continue
        candidate = Path(value)
        if not candidate.suffix:
            continue
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            candidate = candidate.resolve()
            candidate.relative_to(root)
        except (OSError, ValueError):
            continue
        if candidate not in result:
            result.append(candidate)
    return result


def gate_evidence(
    task_id: str,
    blocks: Mapping[str, Mapping[str, str]],
    *,
    repo_root: Path,
    todo_path: Path,
    log_path: Path,
) -> tuple[str, dict[str, Any]]:
    """Hash decisions and source content, excluding clocks, logs, and board prose."""

    root = repo_root.resolve()
    dependencies = dependency_closure(task_id, blocks)
    tasks = {key: {field: str((blocks.get(key) or {}).get(field) or "")
                   for field in _FIELDS} for key in [task_id, *dependencies]}
    artifacts: dict[str, Any] = {}
    scope: set[str] = set()
    excluded = {todo_path.resolve(), log_path.resolve()}
    for key in tasks:
        for path in _referenced_paths(blocks.get(key) or {}, root):
            if path in excluded or path.suffix.casefold() == ".log":
                continue
            relative = path.relative_to(root).as_posix()
            if key != task_id and path.suffix.casefold() in _SOURCE_SUFFIXES:
                scope.add(relative)
            if relative in artifacts:
                continue
            try:
                body = path.read_bytes()
            except OSError:
                artifacts[relative] = {"missing": True}
                continue
            if path.suffix.casefold() == ".json":
                try:
                    artifacts[relative] = _stable(json.loads(body))
                    continue
                except (ValueError, UnicodeError):
                    pass
            artifacts[relative] = {"sha256": hashlib.sha256(body).hexdigest()}
    evidence = {"task_id": task_id, "tasks": tasks, "artifacts": artifacts,
                "dependencies": dependencies, "allowed_source_paths": sorted(scope)}
    return hashlib.sha256(_json(evidence).encode("utf-8")).hexdigest(), evidence


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class GateRecovery:
    """A restart-safe ledger, separate from hard command-failure budgets."""

    def __init__(self, attempt_path: Path):
        self.path = attempt_path.with_name(f"{attempt_path.stem}.gates.json")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            value = {}
        self.records: dict[str, Any] = value if isinstance(value, dict) else {}

    def current(self, task_id: str, fingerprint: str) -> dict[str, Any]:
        record = self.records.get(task_id)
        return dict(record) if isinstance(record, dict) and record.get("fingerprint") == fingerprint else {}

    def store(self, task_id: str, record: Mapping[str, Any]) -> None:
        self.records[task_id] = dict(record)
        _write(self.path, self.records)

    def clear(self, task_id: str) -> None:
        if self.records.pop(task_id, None) is not None:
            _write(self.path, self.records)

    def observe(self, task_id: str, fingerprint: str, *, now_ms: int) -> dict[str, Any]:
        record = self.current(task_id, fingerprint)
        count = max(0, int(record.get("unchanged_probes") or 0)) + 1
        record.update(fingerprint=fingerprint, unchanged_probes=count,
                      first_seen_ms=int(record.get("first_seen_ms") or now_ms), last_seen_ms=now_ms)
        delay = 0 if count < GATE_REPEAT_LIMIT else min(
            GATE_RETRY_MAX_MS, GATE_RETRY_BASE_MS * 2 ** min(count - GATE_REPEAT_LIMIT, 8))
        record["next_retry_at_ms"] = now_ms + delay
        self.store(task_id, record)
        return record

    def packet(
        self, task_id: str, record: Mapping[str, Any], evidence: Mapping[str, Any],
    ) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "-", task_id)[:80] or "task"
        path = self.path.with_name(f"gate-recovery-{safe}.json")
        validations = record.get("dependency_validations") or []
        packet = {
            "schema": "ipfs_accelerate_py/agent-supervisor/gate-recovery@1",
            "reason": "unchanged_gate_withheld", "task_id": task_id,
            "gate_evidence_fingerprint": record.get("fingerprint"),
            "unchanged_probes": record.get("unchanged_probes"),
            "next_retry_at_ms": record.get("next_retry_at_ms"),
            **{key: value for key, value in evidence.items() if key != "artifacts"},
            "artifacts": {path: _artifact_summary(value) for path, value in (evidence.get("artifacts") or {}).items()},
            "body_free": True, "withheld_stage": record.get("withheld_stage", "command"),
            "dependency_validations": validations,
            "deterministic_repair": record.get("deterministic_repair"),
            "required_actions": [
                "Inspect failed or presence-only upstream validation and implement missing behavior.",
                "Use the upstream source scope; preserve the gate and operator holds.",
                "Regenerate evidence and run the original gate before completing any task.",
            ],
            "work_complete": False, "completion_authority": False,
        }
        _write(path, packet)
        return path


def admit_gate_repair_task(
    todo_text: str,
    *,
    repo_root: Path,
    task_id: str,
    repair_task_id: str,
    command: str,
    validation: str,
    predicted_files: list[str],
    acceptance: str,
    depends_on: list[str] | None = None,
    resource_class: str = "cpu-small",
    implementation_timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Append an explicit repair task and add it as a gate dependency.

    The caller supplies a concrete repair contract. Original task statuses and
    acceptance remain intact; the normal lane must execute and validate repair.
    Repeating admission is idempotent, and dependency cycles are refused.
    """

    from .lane_resume import parse_deterministic_command
    from .supervisor_watchdog import parse_todo_task_fields

    prefix = task_id.partition("-")[0] + "-"
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*-[A-Za-z0-9-]+", repair_task_id) or not repair_task_id.startswith(prefix):
        raise ValueError("repair task identity must use the gate's board namespace")
    blocks = dict(parse_todo_task_fields(todo_text, task_header_prefix=prefix))
    if task_id not in blocks or task_id == repair_task_id:
        raise ValueError("a distinct existing gate task is required")
    argv = parse_deterministic_command(command, repo_root=repo_root)
    validator = parse_deterministic_command(validation, repo_root=repo_root)
    if not argv or not validator or (Path(validator[0]).name == "test" and len(validator) > 1
                                     and validator[1] in {"-f", "-e", "-d", "-s", "-r", "-w", "-x", "-L", "-h"}):
        raise ValueError("repair requires a deterministic command and behavioral validation")
    if not predicted_files or not acceptance.strip():
        raise ValueError("repair source scope and acceptance are required")
    if resource_class not in {"cpu-small", "cpu-browser"}:
        raise ValueError("repair resource class must be cpu-small or cpu-browser")
    timeout_limit = 7200 if resource_class == "cpu-browser" else 1800
    if not 1 <= implementation_timeout_seconds <= timeout_limit:
        raise ValueError("repair timeout exceeds its resource bound")
    for relative in predicted_files:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not relative.strip() or relative.endswith(".todo.md"):
            raise ValueError("repair scope must stay inside the repo and exclude task boards")
        try:
            (repo_root / path).resolve().relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError("repair scope escapes the repo") from exc
    dependencies = list(dict.fromkeys(depends_on or []))
    if any(dependency not in blocks or dependency == task_id or task_id in dependency_closure(dependency, blocks)
           for dependency in dependencies):
        raise ValueError("repair dependencies must exist and must not depend on the gate")
    existing = blocks.get(repair_task_id)
    if existing and (
        existing.get("deterministic_command") != command or existing.get("validation") != validation
        or existing.get("acceptance") != acceptance
        or [part.strip() for part in str(existing.get("predicted_files") or "").split(",")] != predicted_files
        or set(dependency_ids(existing)) != set(dependencies)
        or existing.get("resource_class", "cpu-small") != resource_class
        or int(existing.get("implementation_timeout_seconds") or 1800) != implementation_timeout_seconds
    ):
        raise ValueError("existing repair task has a different contract")
    if existing and task_id in dependency_closure(repair_task_id, blocks):
        raise ValueError("existing repair task depends on the gate")
    for value in [command, validation, acceptance, *predicted_files, *dependencies]:
        if "\n" in value or "\r" in value:
            raise ValueError("repair metadata must use single-line fields")
    # Rewire only the gate's dependency field. No status or evidence is edited.
    lines = todo_text.splitlines(keepends=True)
    in_gate = False
    found_dependencies = False
    updated: list[str] = []
    original_dependencies = dependency_ids(blocks[task_id])
    gate_dependencies = list(dict.fromkeys([*original_dependencies, repair_task_id]))
    for line in lines:
        if line.startswith("## "):
            if in_gate and not found_dependencies:
                updated.append("- Depends on: " + ", ".join(gate_dependencies) + "\n\n")
            in_gate = line[3:].strip().split(" ", 1)[0] == task_id
        if in_gate and re.match(r"- Depends on:\s*", line):
            updated.append("- Depends on: " + ", ".join(gate_dependencies) + "\n")
            found_dependencies = True
        else:
            updated.append(line)
    if in_gate and not found_dependencies:
        updated.append("\n- Depends on: " + ", ".join(gate_dependencies) + "\n")
    text = "".join(updated)
    if not existing:
        text = text.rstrip() + "\n\n" + "\n".join([
            f"## {repair_task_id} Repair upstream evidence for {task_id}",
            "", "- Status: pending", "- Priority: P0", "- Track: gate-recovery",
            "- Depends on: " + ", ".join(dependencies), f"- Recovery for: {task_id}",
            f"- Deterministic command: {command}", f"- Validation: {validation}",
            f"- Resource class: {resource_class}",
            f"- Implementation timeout seconds: {implementation_timeout_seconds}",
            "- Predicted files: " + ", ".join(predicted_files),
            "- Conflict policy: Preserve operator holds, original acceptance, and gate decisions.",
            f"- Acceptance: {acceptance}", "",
        ])
    return {"admitted": not bool(existing), "reason": "duplicate_refill" if existing else "gate_repair_admitted",
            "derived_task_id": repair_task_id, "todo_text": text,
            "work_complete": False, "completion_authority": False}
