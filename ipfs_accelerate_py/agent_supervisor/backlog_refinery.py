"""Backlog-refinery helpers for autonomous agent supervisors.

This module ports the repo-local supervisor feed logic into ``ipfs_accelerate_py``
without depending on the ``ipfs_datasets_py`` implementation package.  It keeps
the reusable pieces close to the accelerator daemon runtime:

* refill low todo queues from an objective heap,
* scan tracked code for small bug/improvement findings,
* turn repeated validation or merge failures into evidence-backed follow-up
  tasks instead of allowing indefinite retry loops.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .objective_graph import generate_objective_todos, repo_relative_path, safe_bundle_key
from .todo_daemon.implementation_daemon import parse_task_file


DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS = int(os.environ.get("IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_MIN_OPEN_TASKS", "5"))
DEFAULT_CODEBASE_SCAN_MAX_FINDINGS = int(os.environ.get("IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_MAX_FINDINGS", "5"))
DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_COOLDOWN_SECONDS", "21600")
)
DEFAULT_OBJECTIVE_SCAN_MIN_OPEN_TASKS = int(os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SCAN_MIN_OPEN_TASKS", "5"))
DEFAULT_OBJECTIVE_SCAN_MAX_FINDINGS = int(os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SCAN_MAX_FINDINGS", "5"))
DEFAULT_OBJECTIVE_SCAN_COOLDOWN_SECONDS = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SCAN_COOLDOWN_SECONDS", "21600")
)
DEFAULT_VALIDATION_RETRY_BUDGET = int(os.environ.get("IPFS_ACCELERATE_AGENT_VALIDATION_RETRY_BUDGET", "3"))
DEFAULT_MERGE_RETRY_BUDGET = int(os.environ.get("IPFS_ACCELERATE_AGENT_MERGE_RETRY_BUDGET", "3"))
DEFAULT_TASK_ID_PREFIX = "AUTO-"
DEFAULT_TASK_HEADER_PREFIX = "## AUTO-"
CODEBASE_SCAN_MAX_FILE_BYTES = int(os.environ.get("IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_MAX_FILE_BYTES", "262144"))
CODEBASE_SCAN_SUFFIXES = {
    ".cjs",
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
CODEBASE_SCAN_SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "playwright-report",
    "test-results",
}
CODEBASE_SCAN_SKIP_PREFIXES = (
    "data/agent_supervisor/discovery/",
    "data/agent_supervisor/objective_bundles/",
    "data/agent_supervisor/objective_datasets/",
    "data/agent_supervisor/state/",
    "data/agent_supervisor/worktrees/",
)


@dataclass(frozen=True)
class CodebaseFinding:
    """One static codebase finding that can be converted to a todo task."""

    fingerprint: str
    kind: str
    priority: str
    track: str
    root_relative_path: str
    line_number: int
    snippet: str
    summary: str
    validation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def task_id_prefix(value: str) -> str:
    value = str(value or DEFAULT_TASK_ID_PREFIX).strip()
    if value.startswith("## "):
        value = value[3:].strip()
    return value or DEFAULT_TASK_ID_PREFIX


def task_header_prefix(value: str) -> str:
    value = str(value or DEFAULT_TASK_HEADER_PREFIX).strip()
    if value.startswith("## "):
        return value
    return f"## {value}"


def split_csv(values: Iterable[str] | str) -> list[str]:
    raw_values = [values] if isinstance(values, str) else list(values)
    items: list[str] = []
    for value in raw_values:
        for raw in str(value).split(","):
            item = " ".join(raw.strip().split())
            if item and item.lower() not in {"none", "n/a"}:
                items.append(item)
    return items


def task_ids_from_todo_text(todo_text: str, *, task_prefix: str = DEFAULT_TASK_ID_PREFIX) -> list[str]:
    prefix = task_id_prefix(task_prefix)
    ids: list[str] = []
    for line in todo_text.splitlines():
        if not line.startswith(f"## {prefix}"):
            continue
        parts = line[3:].strip().split(" ", 1)
        if parts:
            ids.append(parts[0])
    return ids


def next_task_id(todo_text: str, *, task_prefix: str = DEFAULT_TASK_ID_PREFIX) -> str:
    prefix = task_id_prefix(task_prefix)
    highest = 0
    for current in task_ids_from_todo_text(todo_text, task_prefix=prefix):
        try:
            highest = max(highest, int(current.rsplit("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return f"{prefix}{highest + 1:03d}"


def task_statuses_from_todo_text(todo_text: str, *, task_prefix: str = DEFAULT_TASK_ID_PREFIX) -> dict[str, str]:
    prefix = task_id_prefix(task_prefix)
    statuses: dict[str, str] = {}
    current_task_id = ""
    for line in todo_text.splitlines():
        if line.startswith(f"## {prefix}"):
            parts = line[3:].strip().split(" ", 1)
            current_task_id = parts[0] if parts else ""
            continue
        if current_task_id and line.startswith("- Status:"):
            statuses[current_task_id] = line.split(":", 1)[1].strip().lower()
            current_task_id = ""
    return statuses


def open_task_count(todo_text: str, *, task_prefix: str = DEFAULT_TASK_ID_PREFIX) -> int:
    statuses = task_statuses_from_todo_text(todo_text, task_prefix=task_prefix)
    return sum(1 for status in statuses.values() if status not in {"completed", "blocked"})


def load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_strategy(path: Path) -> dict[str, Any]:
    strategy = load_json_dict(path)
    if not strategy:
        strategy = {"blocked_tasks": []}
    blocked = strategy.get("blocked_tasks")
    strategy["blocked_tasks"] = [str(item) for item in blocked] if isinstance(blocked, list) else []
    return strategy


def effective_open_task_count(
    todo_text: str,
    *,
    state_path: Path | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
) -> int:
    if state_path is None or not state_path.exists():
        return open_task_count(todo_text, task_prefix=task_prefix)
    payload = load_json_dict(state_path)
    statuses = payload.get("task_statuses")
    if not isinstance(statuses, dict):
        return open_task_count(todo_text, task_prefix=task_prefix)
    task_ids = set(task_ids_from_todo_text(todo_text, task_prefix=task_prefix))
    normalized = {str(task_id): str(status).lower() for task_id, status in statuses.items()}
    if set(normalized) != task_ids:
        return open_task_count(todo_text, task_prefix=task_prefix)
    try:
        state_task_count = int(payload.get("task_count") or 0)
    except (TypeError, ValueError):
        return open_task_count(todo_text, task_prefix=task_prefix)
    if state_task_count != len(task_ids):
        return open_task_count(todo_text, task_prefix=task_prefix)
    return sum(1 for status in normalized.values() if status not in {"completed", "blocked"})


def parse_iso_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def should_refill_backlog(
    *,
    todo_text: str,
    state_path: Path | None,
    strategy: Mapping[str, Any],
    last_scan_key: str,
    last_drained_scan_task_count_key: str,
    task_prefix: str,
    min_open_tasks: int,
    cooldown_seconds: int,
    force: bool = False,
) -> tuple[bool, str, int, int]:
    current_open = effective_open_task_count(todo_text, state_path=state_path, task_prefix=task_prefix)
    task_count = len(task_ids_from_todo_text(todo_text, task_prefix=task_prefix))
    if force:
        return True, "force", current_open, task_count
    if current_open > min_open_tasks:
        return False, "open_task_threshold", current_open, task_count
    drained = current_open == 0
    try:
        last_drained_count = int(strategy.get(last_drained_scan_task_count_key) or -1)
    except (TypeError, ValueError):
        last_drained_count = -1
    if drained and last_drained_count != task_count:
        return True, "drained_exhaustive", current_open, task_count
    last_scan_at = parse_iso_timestamp(str(strategy.get(last_scan_key) or ""))
    if last_scan_at is None:
        return True, "low_backlog", current_open, task_count
    elapsed = (datetime.now(timezone.utc) - last_scan_at).total_seconds()
    if elapsed >= cooldown_seconds:
        return True, "low_backlog", current_open, task_count
    return False, "cooldown", current_open, task_count


def git_toplevel_for_path(cwd: Path) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve()


def repo_relative_path_safe(relative: str) -> bool:
    if not relative or relative.startswith("/") or "\0" in relative:
        return False
    return ".." not in Path(relative).parts


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def codebase_scan_path_skipped(
    path: Path,
    *,
    repo_root: Path,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
) -> bool:
    try:
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        relative = path.as_posix()
    if any(relative == prefix.rstrip("/") or relative.startswith(prefix) for prefix in skip_prefixes):
        return True
    return any(part in CODEBASE_SCAN_SKIP_PARTS for part in path.parts)


def discover_git_worktrees(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add_if_worktree(candidate: Path) -> None:
        top = git_toplevel_for_path(candidate)
        if top is None:
            return
        resolved = top.resolve()
        if not path_is_under(resolved, repo_root):
            return
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)

    add_if_worktree(repo_root)
    for current, dirnames, _filenames in os.walk(repo_root):
        current_path = Path(current)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in CODEBASE_SCAN_SKIP_PARTS
            and not codebase_scan_path_skipped(current_path / dirname, repo_root=repo_root)
        ]
        if current_path != repo_root and (current_path / ".git").exists():
            add_if_worktree(current_path)
            dirnames[:] = []
    return roots


def tracked_files(repo: Path) -> list[Path]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=repo, capture_output=True, check=False)
    if result.returncode != 0:
        return []
    files: list[Path] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        if not repo_relative_path_safe(relative):
            continue
        path = repo / relative
        if path.is_file():
            files.append(path)
    return files


def root_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def file_is_scan_candidate(path: Path, *, repo_root: Path) -> bool:
    if codebase_scan_path_skipped(path, repo_root=repo_root):
        return False
    if "-codebase-scan-" in path.name or "retry-budget" in path.name:
        return False
    if path.name == "todo.md" or path.name.endswith(".todo.md"):
        return False
    if path.suffix.lower() not in CODEBASE_SCAN_SUFFIXES:
        return False
    try:
        if path.stat().st_size > CODEBASE_SCAN_MAX_FILE_BYTES:
            return False
    except OSError:
        return False
    return True


def scan_fingerprint(*, kind: str, root_relative_path: str, line_number: int, snippet: str) -> str:
    normalized = " ".join(snippet.strip().split())
    payload = f"{kind}\0{root_relative_path}\0{line_number}\0{normalized}"
    return sha1(payload.encode("utf-8")).hexdigest()


def scan_track_for_path(path: str) -> str:
    lowered = path.lower()
    if "/test/" in lowered or lowered.startswith("tests/") or "test_" in Path(lowered).name:
        return "quality"
    if "ui" in Path(lowered).parts or "frontend" in Path(lowered).parts:
        return "ui"
    if lowered.endswith((".md", ".rst")):
        return "docs"
    if lowered.endswith((".py", ".rs", ".sh")):
        return "runtime"
    return "ops"


def scan_validation_for_path(root_relative: str) -> str:
    quoted = shlex.quote(root_relative)
    suffix = Path(root_relative).suffix.lower()
    if suffix == ".py":
        return f"python3 -m py_compile {quoted}"
    if suffix == ".json":
        return f"python3 -m json.tool {quoted} >/dev/null"
    if suffix in {".yaml", ".yml"}:
        source = 'import pathlib, sys; p=pathlib.Path(sys.argv[1]); assert p.read_text(encoding="utf-8").strip()'
        return f"python3 -c {shlex.quote(source)} {quoted}"
    return f"test -f {quoted}"


def scan_findings_in_file(path: Path, *, repo_root: Path) -> list[CodebaseFinding]:
    root_relative = root_relative_path(repo_root, path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    findings: list[CodebaseFinding] = []
    in_fenced_block = False
    scan_fences = path.suffix.lower() in {".md", ".rst"}
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if scan_fences and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fenced_block = not in_fenced_block
            continue
        if in_fenced_block or not stripped:
            continue
        lowered = stripped.lower()
        kind = ""
        priority = "P2"
        summary = ""
        if re.search(r"\b(todo|fixme|hack|xxx)\b", stripped, flags=re.IGNORECASE):
            kind = "annotated_followup"
            priority = "P2" if re.search(r"\b(fixme|hack|xxx)\b", stripped, flags=re.IGNORECASE) else "P3"
            summary = f"Resolve code annotation in {root_relative}:{index}"
        elif re.search(r"\bexcept\s*:\s*$", stripped) or re.search(r"\bexcept\s+Exception\b", stripped):
            window = "\n".join(lines[index : min(len(lines), index + 3)]).lower()
            if "pass" in window or "return none" in window:
                kind = "swallowed_exception"
                priority = "P1"
                summary = f"Review swallowed exception path in {root_relative}:{index}"
        elif "assert false" in lowered or "raise notimplementederror" in lowered:
            kind = "placeholder_runtime_path"
            priority = "P1"
            summary = f"Replace placeholder runtime path in {root_relative}:{index}"
        if not kind:
            continue
        fingerprint = scan_fingerprint(
            kind=kind,
            root_relative_path=root_relative,
            line_number=index,
            snippet=stripped,
        )
        findings.append(
            CodebaseFinding(
                fingerprint=fingerprint,
                kind=kind,
                priority=priority,
                track=scan_track_for_path(root_relative),
                root_relative_path=root_relative,
                line_number=index,
                snippet=stripped[:240],
                summary=summary,
                validation=scan_validation_for_path(root_relative),
            )
        )
    return findings


def scan_codebase_findings(
    repo_root: Path,
    *,
    max_findings: int,
    seen_fingerprints: Iterable[str] = (),
    exhaustive: bool = False,
) -> list[CodebaseFinding]:
    findings: list[CodebaseFinding] = []
    seen = {str(item) for item in seen_fingerprints if str(item).strip()}
    for git_root in discover_git_worktrees(repo_root):
        for path in tracked_files(git_root):
            if not file_is_scan_candidate(path, repo_root=repo_root):
                continue
            for finding in scan_findings_in_file(path, repo_root=repo_root):
                if finding.fingerprint in seen:
                    continue
                if len(findings) < max_findings:
                    findings.append(finding)
                    seen.add(finding.fingerprint)
                    continue
                if not exhaustive:
                    return findings
    return findings


def write_codebase_scan_discovery(
    *,
    discovery_dir: Path,
    task_id: str,
    finding: CodebaseFinding,
) -> Path:
    date = datetime.now(timezone.utc).date().isoformat()
    path = discovery_dir / f"{date}-{task_id.lower()}-codebase-scan-{finding.fingerprint[:12]}.md"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    content = f"""# {task_id} Codebase Scan Finding

Date: {date}
Fingerprint: {finding.fingerprint}
Kind: {finding.kind}
Source: {finding.root_relative_path}:{finding.line_number}
Priority: {finding.priority}
Track: {finding.track}

## Evidence

```text
{finding.snippet}
```

## Suggested Handling

Review the finding in context, decide whether it represents a bug, missing test,
maintenance risk, or false positive, and land a small fix with validation. If the
finding is a false positive, document why in the changed code or discovery notes
so the supervisor does not keep re-adding the same work.
"""
    path.write_text(content, encoding="utf-8")
    return path


def codebase_scan_task_block(
    *,
    task_id: str,
    finding: CodebaseFinding,
    discovery_path: Path,
    depends_on: Sequence[str] = (),
) -> str:
    outputs = ["data/agent_supervisor/discovery", finding.root_relative_path]
    return f"""## {task_id} {finding.summary}

- Status: todo
- Completion: manual
- Priority: {finding.priority}
- Track: {finding.track}
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(outputs)}
- Validation: {finding.validation}
- Acceptance: Codebase scan filed this finding from {finding.root_relative_path}:{finding.line_number}. Use evidence in {discovery_path}, fix the bug or improvement, add or update focused validation when appropriate, and keep the supervisor-fed backlog parseable.
"""


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def event_merge_result(event: Mapping[str, Any]) -> dict[str, Any]:
    merge_result = event.get("merge_result") or {}
    return dict(merge_result) if isinstance(merge_result, Mapping) else {}


def consecutive_validation_failures(events: Sequence[Mapping[str, Any]], task_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for event in reversed(events):
        if str(event.get("type") or "") != "implementation_finished":
            continue
        if str(event.get("task_id") or "") != task_id:
            continue
        validation = event.get("validation_result") or {}
        if not isinstance(validation, Mapping) or not validation.get("attempted"):
            break
        if validation.get("passed", False):
            break
        failures.append(dict(event))
    failures.reverse()
    return failures


def consecutive_merge_failures(events: Sequence[Mapping[str, Any]], task_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for event in reversed(events):
        event_type = str(event.get("type") or "")
        if event_type not in {"implementation_finished", "merge_reconciled"}:
            continue
        if str(event.get("task_id") or "") != task_id:
            continue

        merge_result = event_merge_result(event)
        if event_type == "merge_reconciled" and event.get("resolved", False):
            break
        if event_type == "implementation_finished":
            validation = event.get("validation_result") or {}
            if isinstance(validation, Mapping) and validation.get("attempted") and not validation.get("passed", False):
                break
            if merge_result.get("merged", False):
                break

        if not merge_result.get("attempted", False):
            continue
        if merge_result.get("merged", False):
            break
        if str(merge_result.get("reason") or "") == "not_attempted":
            continue
        failures.append(dict(event))

    failures.reverse()
    return failures


def write_retry_budget_discovery(
    *,
    discovery_dir: Path,
    task_id: str,
    source_task_id: str,
    failed_command: str,
    failures: Sequence[Mapping[str, Any]],
    retry_budget: int,
    failure_kind: str = "validation",
) -> Path:
    date = datetime.now(timezone.utc).date().isoformat()
    suffix = "merge-retry-budget" if failure_kind == "merge" else "retry-budget"
    path = discovery_dir / f"{date}-{task_id.lower()}-{source_task_id.lower()}-{suffix}.md"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    log_paths = [str(event.get("log_path") or "") for event in failures if event.get("log_path")]
    attempt_numbers = [str(event.get("attempt") or "") for event in failures if event.get("attempt")]
    content = f"""# {task_id} {failure_kind.title()} Retry-Budget Finding: {source_task_id}

Date: {date}
Source task: {source_task_id}
Follow-up task: {task_id}
Retry budget: {retry_budget}
Observed consecutive {failure_kind} failures: {len(failures)}

## Evidence

- Failed command: `{failed_command}`
- Attempts: {", ".join(attempt_numbers) or "not recorded"}
- Logs: {", ".join(log_paths) or "not recorded"}

## Guardrail Result

The accelerator backlog refinery classified this as backlog work instead of
allowing another implementation attempt to loop on the same failure. The source
task is added to the strategy `blocked_tasks` list and the follow-up task below
is appended for normal daemon parsing.
"""
    path.write_text(content, encoding="utf-8")
    return path


def validation_retry_task_block(
    *,
    task_id: str,
    source_task: Any,
    failed_command: str,
    discovery_path: Path,
    depends_on: Sequence[str] = (),
) -> str:
    outputs = list(getattr(source_task, "outputs", []) or [])
    if "data/agent_supervisor/discovery" not in outputs:
        outputs.append("data/agent_supervisor/discovery")
    return f"""## {task_id} Resolve validation retry-budget failure for {source_task.task_id}

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(outputs)}
- Validation: {failed_command}
- Acceptance: Retry-budget guardrail filed this from repeated validation failures in {source_task.task_id}. Use evidence in {discovery_path} to fix the validation blocker, then remove {source_task.task_id} from the strategy blocked_tasks list so the original backlog item can continue without an indefinite retry loop.
"""


def merge_command_label(merge_result: Mapping[str, Any]) -> str:
    command = merge_result.get("command")
    if isinstance(command, list) and command:
        return shlex.join(str(part) for part in command)
    if command:
        return str(command)
    return f"git merge ({str(merge_result.get('reason') or 'merge_failed')})"


def merge_retry_task_block(
    *,
    task_id: str,
    source_task: Any,
    discovery_path: Path,
    strategy_path: Path,
    depends_on: Sequence[str] = (),
) -> str:
    outputs = list(getattr(source_task, "outputs", []) or [])
    if "data/agent_supervisor/discovery" not in outputs:
        outputs.append("data/agent_supervisor/discovery")
    validation_source = "\n".join(
        [
            "import json, pathlib",
            f"strategy = json.loads(pathlib.Path({str(strategy_path)!r}).read_text(encoding='utf-8'))",
            f"assert {source_task.task_id!r} not in strategy.get('blocked_tasks', [])",
        ]
    )
    validation_command = f"python3 -c {shlex.quote(f'exec({validation_source!r})')}"
    return f"""## {task_id} Resolve merge retry-budget failure for {source_task.task_id}

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(outputs)}
- Validation: {validation_command}
- Acceptance: Merge retry-budget guardrail filed this from repeated merge failures in {source_task.task_id}. Use evidence in {discovery_path} to fix the merge blocker, verify the intended implementation changes are committed in their owning repository or submodule, run `ipfs-accelerate-agent-merge-resolver --events-path ... --apply` when the conflict is semantic, then remove {source_task.task_id} from the strategy blocked_tasks list so the original backlog item can continue without an indefinite retry loop.
"""


def record_retry_budget_findings(
    *,
    todo_path: Path,
    events_path: Path,
    strategy_path: Path,
    discovery_dir: Path,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    validation_retry_budget: int = DEFAULT_VALIDATION_RETRY_BUDGET,
    merge_retry_budget: int = DEFAULT_MERGE_RETRY_BUDGET,
) -> list[dict[str, Any]]:
    """Append follow-up tasks for repeated validation or merge failures."""

    if not todo_path.exists():
        return []
    tasks = parse_task_file(todo_path, task_header_prefix(task_header_prefix_value))
    if not tasks:
        return []

    todo_text = todo_path.read_text(encoding="utf-8")
    task_ids = set(task_ids_from_todo_text(todo_text, task_prefix=task_prefix))
    completed_task_ids = {task.task_id for task in tasks if task.status == "completed"}
    events = iter_jsonl(events_path)
    strategy = load_strategy(strategy_path)
    blocked_tasks = [str(item) for item in strategy.get("blocked_tasks", []) if str(item).strip()]
    findings: list[dict[str, Any]] = []

    if validation_retry_budget > 0:
        for task in tasks:
            if task.task_id in completed_task_ids:
                continue
            marker = f"retry-budget failure for {task.task_id}"
            if marker in todo_text:
                continue
            failures = consecutive_validation_failures(events, task.task_id)
            if len(failures) < validation_retry_budget:
                continue
            latest_validation = failures[-1].get("validation_result") or {}
            failed_command = str(latest_validation.get("failed_command") or "")
            if not failed_command:
                continue
            follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
            discovery_path = write_retry_budget_discovery(
                discovery_dir=discovery_dir,
                task_id=follow_up_task_id,
                source_task_id=task.task_id,
                failed_command=failed_command,
                failures=failures,
                retry_budget=validation_retry_budget,
            )
            depends_on = list(task.depends_on)
            task_block = validation_retry_task_block(
                task_id=follow_up_task_id,
                source_task=task,
                failed_command=failed_command,
                discovery_path=discovery_path,
                depends_on=depends_on,
            )
            todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
            task_ids.add(follow_up_task_id)
            if task.task_id not in blocked_tasks:
                blocked_tasks.append(task.task_id)
            findings.append(
                {
                    "source_task_id": task.task_id,
                    "follow_up_task_id": follow_up_task_id,
                    "failure_count": len(failures),
                    "failed_command": failed_command,
                    "discovery_path": str(discovery_path),
                    "failure_kind": "validation",
                }
            )

    if merge_retry_budget > 0:
        for task in tasks:
            if task.task_id in completed_task_ids:
                continue
            marker = f"merge retry-budget failure for {task.task_id}"
            if marker in todo_text:
                continue
            failures = consecutive_merge_failures(events, task.task_id)
            if len(failures) < merge_retry_budget:
                continue
            latest_merge_result = event_merge_result(failures[-1])
            if not latest_merge_result:
                continue
            follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
            failed_command = merge_command_label(latest_merge_result)
            discovery_path = write_retry_budget_discovery(
                discovery_dir=discovery_dir,
                task_id=follow_up_task_id,
                source_task_id=task.task_id,
                failed_command=failed_command,
                failures=failures,
                retry_budget=merge_retry_budget,
                failure_kind="merge",
            )
            task_block = merge_retry_task_block(
                task_id=follow_up_task_id,
                source_task=task,
                discovery_path=discovery_path,
                strategy_path=strategy_path,
                depends_on=task.depends_on,
            )
            todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
            task_ids.add(follow_up_task_id)
            if task.task_id not in blocked_tasks:
                blocked_tasks.append(task.task_id)
            findings.append(
                {
                    "source_task_id": task.task_id,
                    "follow_up_task_id": follow_up_task_id,
                    "failure_count": len(failures),
                    "failed_command": failed_command,
                    "discovery_path": str(discovery_path),
                    "failure_kind": "merge",
                }
            )

    if not findings:
        return []

    todo_path.write_text(todo_text, encoding="utf-8")
    strategy["blocked_tasks"] = blocked_tasks
    strategy["last_retry_budget_guardrail_at"] = utc_now()
    strategy["retry_budget_findings"] = findings
    write_json(strategy_path, strategy)
    return findings


def record_codebase_scan_findings(
    *,
    todo_path: Path,
    state_path: Path | None,
    strategy_path: Path,
    discovery_dir: Path,
    repo_root: Path,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    depends_on: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_CODEBASE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Feed a todo board with static codebase findings when backlog runs low."""

    if max_findings <= 0 or not todo_path.exists():
        return []
    todo_text = todo_path.read_text(encoding="utf-8")
    strategy = load_strategy(strategy_path)
    should_scan, mode, current_open, task_count = should_refill_backlog(
        todo_text=todo_text,
        state_path=state_path,
        strategy=strategy,
        last_scan_key="last_codebase_scan_at",
        last_drained_scan_task_count_key="last_drained_codebase_scan_task_count",
        task_prefix=task_prefix,
        min_open_tasks=min_open_tasks,
        cooldown_seconds=cooldown_seconds,
        force=force,
    )
    if not should_scan:
        return []

    seen = {str(item) for item in strategy.get("codebase_scan_seen_fingerprints", []) if str(item).strip()}
    findings = scan_codebase_findings(
        repo_root,
        max_findings=max_findings,
        seen_fingerprints=seen,
        exhaustive=mode == "drained_exhaustive",
    )
    strategy["last_codebase_scan_at"] = utc_now()
    strategy["last_codebase_scan_mode"] = mode
    if current_open == 0:
        strategy["last_drained_codebase_scan_task_count"] = task_count
    strategy["codebase_scan_seen_fingerprints"] = sorted(seen | {finding.fingerprint for finding in findings})
    if not findings:
        strategy["last_codebase_scan_findings"] = []
        write_json(strategy_path, strategy)
        return []

    appended: list[dict[str, Any]] = []
    for finding in findings:
        follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
        discovery_path = write_codebase_scan_discovery(
            discovery_dir=discovery_dir,
            task_id=follow_up_task_id,
            finding=finding,
        )
        task_block = codebase_scan_task_block(
            task_id=follow_up_task_id,
            finding=finding,
            discovery_path=discovery_path,
            depends_on=depends_on,
        )
        todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
        appended.append(
            {
                "follow_up_task_id": follow_up_task_id,
                "fingerprint": finding.fingerprint,
                "kind": finding.kind,
                "source": f"{finding.root_relative_path}:{finding.line_number}",
                "discovery_path": str(discovery_path),
            }
        )

    todo_path.write_text(todo_text, encoding="utf-8")
    strategy["last_codebase_scan_findings"] = appended
    write_json(strategy_path, strategy)
    return appended


def record_objective_backlog_findings(
    *,
    repo_root: Path,
    objective_path: Path,
    todo_path: Path,
    discovery_dir: Path,
    bundle_dir: Path,
    strategy_path: Path,
    state_path: Path | None = None,
    dataset_dir: Path | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    depends_on: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_OBJECTIVE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_OBJECTIVE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_OBJECTIVE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    persist_ast_dataset: bool = True,
) -> list[dict[str, Any]]:
    """Feed a todo board from objective-heap gaps when backlog runs low."""

    if max_findings <= 0 or not todo_path.exists() or not objective_path.exists():
        return []
    todo_text = todo_path.read_text(encoding="utf-8")
    strategy = load_strategy(strategy_path)
    should_scan, mode, current_open, task_count = should_refill_backlog(
        todo_text=todo_text,
        state_path=state_path,
        strategy=strategy,
        last_scan_key="last_objective_goal_scan_at",
        last_drained_scan_task_count_key="last_drained_objective_goal_scan_task_count",
        task_prefix=task_prefix,
        min_open_tasks=min_open_tasks,
        cooldown_seconds=cooldown_seconds,
        force=force,
    )
    if not should_scan:
        return []

    seen = {str(item) for item in strategy.get("objective_goal_seen_fingerprints", []) if str(item).strip()}
    records = generate_objective_todos(
        repo_root=repo_root,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        dataset_dir=dataset_dir,
        task_prefix=task_prefix,
        depends_on=depends_on,
        max_findings=max_findings,
        seen_fingerprints=seen,
        persist_ast_dataset=persist_ast_dataset,
    )
    strategy["last_objective_goal_scan_at"] = utc_now()
    strategy["last_objective_goal_scan_mode"] = mode
    if current_open == 0:
        strategy["last_drained_objective_goal_scan_task_count"] = task_count
    strategy["objective_goal_seen_fingerprints"] = sorted(
        seen | {record.finding.fingerprint for record in records}
    )

    appended = [
        {
            "follow_up_task_id": record.task_id,
            "fingerprint": record.finding.fingerprint,
            "kind": "objective_goal_gap",
            "goal_id": record.finding.goal_id,
            "missing_evidence": record.finding.missing_evidence,
            "bundle_key": record.finding.bundle_key,
            "bundle_shard": repo_relative_path(repo_root, bundle_dir / f"{safe_bundle_key(record.finding.bundle_key)}.todo.md"),
            "bundle_strategy": record.finding.bundle_strategy,
            "graph_depth": record.finding.graph_depth,
            "parent_goal_ids": record.finding.parent_goal_ids,
            "discovery_path": str(record.discovery_path),
        }
        for record in records
    ]
    strategy["last_objective_goal_scan_findings"] = appended
    write_json(strategy_path, strategy)
    return appended


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refill and guard an accelerator todo backlog")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--todo-path", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--strategy-path", type=Path, default=None)
    parser.add_argument("--events-path", type=Path, default=None)
    parser.add_argument("--discovery-dir", type=Path, default=None)
    parser.add_argument("--objective-path", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--task-prefix", default=DEFAULT_TASK_ID_PREFIX)
    parser.add_argument("--task-header-prefix", default=DEFAULT_TASK_HEADER_PREFIX)
    parser.add_argument("--depends-on", action="append", default=[])
    parser.add_argument("--objective-scan", action="store_true")
    parser.add_argument("--codebase-scan", action="store_true")
    parser.add_argument("--retry-budget", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-open-tasks", type=int, default=DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS)
    parser.add_argument("--max-findings", type=int, default=DEFAULT_CODEBASE_SCAN_MAX_FINDINGS)
    parser.add_argument("--cooldown-seconds", type=int, default=DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS)
    parser.add_argument("--validation-retry-budget", type=int, default=DEFAULT_VALIDATION_RETRY_BUDGET)
    parser.add_argument("--merge-retry-budget", type=int, default=DEFAULT_MERGE_RETRY_BUDGET)
    parser.add_argument("--no-persist-ast-dataset", action="store_true")
    return parser


def run_backlog_refinery(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    state_root = repo_root / "data" / "agent_supervisor"
    strategy_path = (args.strategy_path or state_root / "strategy.json").resolve()
    discovery_dir = (args.discovery_dir or state_root / "discovery").resolve()
    bundle_dir = (args.bundle_dir or state_root / "objective_bundles").resolve()
    state_path = args.state_path.resolve() if args.state_path else None
    events_path = args.events_path.resolve() if args.events_path else state_root / "events.jsonl"
    depends_on = split_csv(args.depends_on)

    run_all = not (args.objective_scan or args.codebase_scan or args.retry_budget)
    objective_findings: list[dict[str, Any]] = []
    codebase_findings: list[dict[str, Any]] = []
    retry_findings: list[dict[str, Any]] = []

    if (args.objective_scan or run_all) and args.objective_path:
        objective_findings = record_objective_backlog_findings(
            repo_root=repo_root,
            objective_path=args.objective_path.resolve(),
            todo_path=args.todo_path.resolve(),
            discovery_dir=discovery_dir,
            bundle_dir=bundle_dir,
            dataset_dir=args.dataset_dir.resolve() if args.dataset_dir else None,
            strategy_path=strategy_path,
            state_path=state_path,
            task_prefix=args.task_prefix,
            depends_on=depends_on,
            min_open_tasks=args.min_open_tasks,
            max_findings=args.max_findings,
            cooldown_seconds=args.cooldown_seconds,
            force=args.force,
            persist_ast_dataset=not args.no_persist_ast_dataset,
        )
    if args.codebase_scan or run_all:
        codebase_findings = record_codebase_scan_findings(
            todo_path=args.todo_path.resolve(),
            state_path=state_path,
            strategy_path=strategy_path,
            discovery_dir=discovery_dir,
            repo_root=repo_root,
            task_prefix=args.task_prefix,
            depends_on=depends_on,
            min_open_tasks=args.min_open_tasks,
            max_findings=args.max_findings,
            cooldown_seconds=args.cooldown_seconds,
            force=args.force,
        )
    if args.retry_budget or run_all:
        retry_findings = record_retry_budget_findings(
            todo_path=args.todo_path.resolve(),
            events_path=events_path,
            strategy_path=strategy_path,
            discovery_dir=discovery_dir,
            task_header_prefix_value=args.task_header_prefix,
            task_prefix=args.task_prefix,
            validation_retry_budget=args.validation_retry_budget,
            merge_retry_budget=args.merge_retry_budget,
        )

    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.backlog_refinery",
        "repo_root": str(repo_root),
        "todo_path": str(args.todo_path.resolve()),
        "strategy_path": str(strategy_path),
        "objective_generated_count": len(objective_findings),
        "codebase_generated_count": len(codebase_findings),
        "retry_budget_generated_count": len(retry_findings),
        "objective_findings": objective_findings,
        "codebase_findings": codebase_findings,
        "retry_budget_findings": retry_findings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_backlog_refinery(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
