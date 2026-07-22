"""Objective tracking document helpers for autonomous agent supervisors."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .goal_completion import (
    DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
    CompletionEvidence,
    GoalState,
    evaluate_goal_completion,
    normalize_goal_state,
)
from .objective_graph import (
    DEFAULT_EMBEDDING_MIN_SCORE,
    ObjectiveFinding,
    ObjectiveGoal,
    canonical_interoperability_component,
    evidence_index,
    goal_graph,
    normalize_field_key,
    objective_heap_schedule,
    parse_goal_heap,
    safe_bundle_key,
    split_terms,
    utc_now,
)
from .validation_commands import split_validation_commands
from .scan_receipts import RepositoryTreeIdentity, scan_identity
from .task_identity import canonical_content_cid


DEFAULT_ULTIMATE_GOAL = (
    "Make this repository satisfy its stated objective with verifiable code, tests, docs, and runtime evidence."
)
DEFAULT_ROOT_EVIDENCE = (
    "objective goal graph",
    "bundle-local todo shards",
    "AST evidence dataset",
    "embedding evidence scan",
    "LLM merge conflict resolver",
)
DEFAULT_GOAL_PREFIX = os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_GOAL_PREFIX", "OBJ-G")
DEFAULT_TRACKING_DOCUMENT_TITLE = os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_DOCUMENT_TITLE", "Objective Heap")
DEFAULT_ROOT_GOAL_TITLE = os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_ROOT_TITLE", "Objective outcome")
OPEN_TASK_STATUSES_FOR_GOAL_COMPLETION = {"todo", "ready", "in_progress"}
TASK_GOAL_METADATA_KEYS = (
    "goal id",
    "goal ids",
    "goal packet goals",
    "graph parents",
)


@dataclass(frozen=True)
class ObjectiveTrackingResult:
    """Summary of objective tracking document mutations."""

    objective_path: Path
    created: bool
    appended_goal_ids: list[str]
    graph_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objective_path"] = str(self.objective_path)
        if self.graph_path is not None:
            payload["graph_path"] = str(self.graph_path)
        return payload


@dataclass(frozen=True)
class ObjectiveCompletionResult:
    """Summary of objective goals reconciled from repository evidence."""

    objective_path: Path
    completed_goal_ids: list[str]
    active_goal_count: int
    completed_goal_count: int
    completion_evidence: dict[str, dict[str, list[str]]]
    validation_results: dict[str, dict[str, Any]]
    provisional_goal_ids: list[str] = field(default_factory=list)
    verified_goal_ids: list[str] = field(default_factory=list)
    reopened_goal_ids: list[str] = field(default_factory=list)
    analysis_inconclusive_goal_ids: list[str] = field(default_factory=list)
    blocked_goal_ids: list[str] = field(default_factory=list)
    state_counts: dict[str, int] = field(default_factory=dict)
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objective_path"] = str(self.objective_path)
        return payload


@dataclass(frozen=True)
class RepositoryComponent:
    """A repository component that can participate in interoperability goals."""

    path: str
    sources: list[str] = field(default_factory=list)
    exists: bool = False
    is_gitlink: bool = False
    is_gitmodule: bool = False
    manifests: list[str] = field(default_factory=list)
    interface_descriptors: list[str] = field(default_factory=list)
    mcp_descriptors: list[str] = field(default_factory=list)
    python_import_roots: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fibonacci_number(index: int) -> int:
    """Return a one-indexed Fibonacci-ish priority bucket."""

    if index <= 1:
        return 1
    left, right = 1, 1
    for _ in range(2, index + 1):
        left, right = right, left + right
    return right


def fibonacci_priority(depth: int, sibling_index: int = 0) -> int:
    """Return a stable integer priority for a goal depth and sibling order."""

    return fibonacci_number(max(1, depth + 2)) * 1000 + max(0, sibling_index)


def infer_goal_prefix(goals: Sequence[ObjectiveGoal], *, fallback: str = DEFAULT_GOAL_PREFIX) -> str:
    """Infer a numeric goal-id prefix from existing goals."""

    prefixes: dict[str, int] = {}
    for goal in goals:
        match = re.match(r"^(.+?)(\d+)$", goal.goal_id)
        if not match:
            continue
        prefixes[match.group(1)] = prefixes.get(match.group(1), 0) + 1
    if not prefixes:
        return fallback
    return sorted(prefixes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def next_goal_id(goals: Sequence[ObjectiveGoal], *, prefix: str | None = None) -> str:
    prefix = prefix or infer_goal_prefix(goals)
    highest = -1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for goal in goals:
        match = pattern.match(goal.goal_id)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{prefix}{highest + 1:03d}"


def render_goal_block(*, goal_id: str, title: str, fields: dict[str, str]) -> str:
    rows = [f"## {goal_id} {title.strip()}", ""]
    for key, value in fields.items():
        rows.append(f"- {key}: {value}")
    return "\n".join(rows).rstrip() + "\n"


def rewrite_goal_fields(text: str, updates: Mapping[str, Mapping[str, str]]) -> str:
    """Rewrite selected markdown goal fields without reparsing the whole file."""

    if not updates:
        return text

    lines = text.splitlines()
    rewritten: list[str] = []
    block: list[str] = []
    current_goal_id = ""
    header_pattern = re.compile(r"^##\s+(\S+)\s+.+?\s*$")

    def flush() -> None:
        nonlocal block, current_goal_id
        if not block:
            return
        goal_updates = updates.get(current_goal_id)
        if not goal_updates:
            rewritten.extend(block)
            block = []
            return

        normalized_updates = {normalize_field_key(key): (key, value) for key, value in goal_updates.items()}
        seen_keys: set[str] = set()
        output: list[str] = []
        last_field_index = 0
        for line in block:
            if line.startswith("- ") and ":" in line:
                key, _value = line[2:].split(":", 1)
                normalized = normalize_field_key(key)
                if normalized in normalized_updates:
                    output.append(f"- {normalized_updates[normalized][0]}: {normalized_updates[normalized][1]}")
                    seen_keys.add(normalized)
                else:
                    output.append(line)
                last_field_index = len(output)
            else:
                output.append(line)

        missing_lines = [
            f"- {key}: {value}"
            for normalized, (key, value) in normalized_updates.items()
            if normalized not in seen_keys
        ]
        if missing_lines:
            insert_at = max(1, last_field_index)
            output[insert_at:insert_at] = missing_lines
        rewritten.extend(output)
        block = []

    for line in lines:
        header = header_pattern.match(line)
        if header:
            flush()
            current_goal_id = header.group(1)
            block = [line]
            continue
        if block:
            block.append(line)
        else:
            rewritten.append(line)
    flush()
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(rewritten) + suffix


def completion_evidence_summary(evidence: Mapping[str, Sequence[str]]) -> str:
    parts: list[str] = []
    for term, paths in evidence.items():
        compact_paths = ", ".join(str(path) for path in list(paths)[:3])
        if compact_paths:
            parts.append(f"{term} => {compact_paths}")
    return "; ".join(parts)


def open_goal_ids_from_todo_board(todo_path: Path, task_header_prefix: str = "") -> set[str]:
    """Return objective goal ids with open tasks in the current task board."""

    if not todo_path.exists():
        return set()

    from .todo_daemon.implementation_daemon import TASK_HEADER_PREFIX, parse_task_file

    open_goal_ids: set[str] = set()
    for task in parse_task_file(todo_path, task_header_prefix or TASK_HEADER_PREFIX):
        if task.status not in OPEN_TASK_STATUSES_FOR_GOAL_COMPLETION:
            continue
        for key in TASK_GOAL_METADATA_KEYS:
            open_goal_ids.update(split_terms(task.metadata.get(key, "")))
    return {goal_id for goal_id in open_goal_ids if goal_id}


def open_goal_ids_from_todo_boards(
    todo_boards: Sequence[tuple[Path, str]],
) -> dict[str, list[str]]:
    """Return open objective goal ids mapped to the board paths that still reference them."""

    open_goal_ids: dict[str, list[str]] = {}
    seen_boards: set[tuple[str, str]] = set()
    for todo_path, task_header_prefix in todo_boards:
        board_key = (str(todo_path), str(task_header_prefix))
        if board_key in seen_boards:
            continue
        seen_boards.add(board_key)
        for goal_id in open_goal_ids_from_todo_board(todo_path, task_header_prefix):
            open_goal_ids.setdefault(goal_id, []).append(str(todo_path))
    return open_goal_ids


def run_goal_validation(
    *,
    repo_root: Path,
    goal: ObjectiveGoal,
    timeout_seconds: float = 300.0,
    repository_identity: RepositoryTreeIdentity | None = None,
) -> dict[str, Any]:
    """Run goal validation and return a tree-bound, content-addressed receipt."""

    commands = split_validation_commands(str(goal.fields.get("validation") or ""))
    started_at = utc_now()
    if not commands:
        payload = {
            "schema": "ipfs_accelerate_py.agent_supervisor.goal-validation@1",
            "goal_id": goal.goal_id,
            "attempted": False,
            "passed": False,
            "returncode": 1,
            "results": [],
            "reason": "missing_validation_commands",
            "started_at": started_at,
            "finished_at": utc_now(),
        }
        identity = repository_identity or scan_identity(repo_root)
        payload.update(identity.to_dict())
        payload["receipt_cid"] = canonical_content_cid(payload)
        return payload
    results: list[dict[str, Any]] = []
    failure: dict[str, Any] = {}
    for command in commands:
        command_started_at = utc_now()
        try:
            completed = subprocess.run(
                ["/bin/bash", "-lc", command],
                cwd=repo_root,
                text=True,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            result = {
                "command": command,
                "started_at": command_started_at,
                "finished_at": utc_now(),
                "returncode": 124,
                "timed_out": True,
                "stdout": str(exc.stdout or "")[-4000:],
                "stderr": str(exc.stderr or "")[-4000:],
            }
            results.append(result)
            failure = {"returncode": 124, "failed_command": command, "error": "timeout"}
            break
        result = {
            "command": command,
            "started_at": command_started_at,
            "finished_at": utc_now(),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
        results.append(result)
        if completed.returncode != 0:
            failure = {"returncode": completed.returncode, "failed_command": command}
            break
    identity = repository_identity or scan_identity(repo_root)
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.goal-validation@1",
        "goal_id": goal.goal_id,
        "attempted": True,
        "passed": not failure,
        "returncode": int(failure.get("returncode", 0)),
        "results": results,
        "started_at": started_at,
        "finished_at": utc_now(),
        **identity.to_dict(),
    }
    payload.update({key: value for key, value in failure.items() if key != "returncode"})
    payload["receipt_cid"] = canonical_content_cid(payload)
    return payload


def _git_output(repo_root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    """Best-effort bounded git query used for completion tree identity."""

    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            text=not binary,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return b"" if binary else ""
    if completed.returncode != 0:
        return b"" if binary else ""
    return completed.stdout if binary else str(completed.stdout).strip()


def completion_tree_identity(repo_root: Path, *, objective_path: Path) -> RepositoryTreeIdentity:
    """Return source-tree identity without self-invalidating tracker writes.

    The objective document is mutable supervisor state.  Persisting a
    lifecycle transition must not immediately make otherwise-current evidence
    stale.  This calculation is byte-for-byte compatible with ``scan_identity``
    while excluding only that control document; every code, test, task-board,
    configuration, tracked, and untracked change remains in the digest.
    """

    root = repo_root.resolve()
    top_text = str(_git_output(root, "rev-parse", "--show-toplevel") or "")
    if not top_text:
        return scan_identity(root)
    top = Path(top_text).resolve()
    try:
        relative = objective_path.resolve().relative_to(top).as_posix()
    except ValueError:
        return scan_identity(root)
    base = scan_identity(root)
    head_tree = str(_git_output(root, "rev-parse", "HEAD^{tree}") or "")
    if not head_tree:
        return base
    exclude = f":(exclude){relative}"
    pathspec = ("--", ".", exclude)
    status = _git_output(
        top,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        *pathspec,
        binary=True,
    )
    assert isinstance(status, bytes)
    if not status:
        return RepositoryTreeIdentity(repository_id=base.repository_id, tree_id=head_tree)
    digest = sha256()
    digest.update(head_tree.encode("ascii", errors="replace"))
    digest.update(b"\0status\0")
    digest.update(status)
    digest.update(b"\0diff\0")
    diff = _git_output(
        top,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        *pathspec,
        binary=True,
    )
    assert isinstance(diff, bytes)
    digest.update(diff)
    untracked = _git_output(
        top,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        *pathspec,
        binary=True,
    )
    assert isinstance(untracked, bytes)
    for raw_relative in sorted(path for path in untracked.split(b"\0") if path):
        digest.update(b"\0untracked\0")
        digest.update(raw_relative)
        try:
            candidate = top / raw_relative.decode("utf-8", errors="surrogateescape")
            if candidate.is_symlink():
                digest.update(candidate.readlink().as_posix().encode("utf-8"))
            elif candidate.is_file():
                with candidate.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        except OSError:
            continue
    return RepositoryTreeIdentity(
        repository_id=base.repository_id,
        tree_id=f"sha256:{digest.hexdigest()}",
    )


def reconcile_objective_goal_completion(
    *,
    repo_root: Path,
    objective_path: Path,
    todo_path: Path | None = None,
    task_header_prefix: str = "",
    todo_boards: Sequence[tuple[Path, str]] | None = None,
    embedding_min_score: float = DEFAULT_EMBEDDING_MIN_SCORE,
    completion_evidence_records: Mapping[
        str, Sequence[CompletionEvidence | Mapping[str, Any]]
    ] | None = None,
    completion_gate_records: Mapping[str, Mapping[str, Any]] | None = None,
    now: str | None = None,
    evidence_freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
) -> ObjectiveCompletionResult:
    """Reconcile objective goals through the evidence-backed lifecycle.

    Closed tasks advance an active goal to ``provisionally_complete``.  A
    separate reconciliation can advance that provisional goal to
    ``verified_complete`` only when every acceptance criterion has a fresh,
    tree-bound evidence record and the goal's validation command passes.
    """

    if not objective_path.exists():
        return ObjectiveCompletionResult(
            objective_path=objective_path,
            completed_goal_ids=[],
            active_goal_count=0,
            completed_goal_count=0,
            completion_evidence={},
            validation_results={},
        )

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    supplied_records = completion_evidence_records or {}
    supplied_gate_records = completion_gate_records or {}
    candidate_goals = []
    persisted_records: dict[str, list[CompletionEvidence]] = {}
    for goal in goals:
        records: list[CompletionEvidence] = []
        for item in supplied_records.get(goal.goal_id, ()):
            records.append(item if isinstance(item, CompletionEvidence) else CompletionEvidence.from_dict(item))
        if not records:
            raw_records = str(
                goal.fields.get("completion_evidence_records")
                or goal.fields.get("completion_evidence_json")
                or ""
            ).strip()
            if raw_records:
                try:
                    decoded = json.loads(raw_records)
                except (TypeError, ValueError, json.JSONDecodeError):
                    decoded = []
                if isinstance(decoded, Mapping):
                    decoded = [decoded]
                if isinstance(decoded, list):
                    records.extend(
                        CompletionEvidence.from_dict(item)
                        for item in decoded
                        if isinstance(item, Mapping)
                    )
        persisted_records[goal.goal_id] = records
        state = normalize_goal_state(goal.status)
        if state in {
            GoalState.ACTIVE,
            GoalState.REOPENED,
            GoalState.PROVISIONALLY_COMPLETE,
            GoalState.ANALYSIS_INCONCLUSIVE,
        } or (goal.status == GoalState.VERIFIED_COMPLETE.value and records):
            candidate_goals.append(goal)

    terms: list[str] = []
    for goal in candidate_goals:
        terms.extend(goal.required_evidence)
    discovered_evidence = evidence_index(
        repo_root,
        objective_path=objective_path,
        terms=terms,
        embedding_min_score=embedding_min_score,
    )

    updates: dict[str, dict[str, str]] = {}
    completed_goal_ids: list[str] = []
    provisional_goal_ids: list[str] = []
    reopened_goal_ids: list[str] = []
    analysis_inconclusive_goal_ids: list[str] = []
    blocked_goal_ids: list[str] = []
    completion_evidence: dict[str, dict[str, list[str]]] = {}
    validation_results: dict[str, dict[str, Any]] = {}
    decisions: dict[str, dict[str, Any]] = {}
    transitioned_at = now or utc_now()
    completion_boards: list[tuple[Path, str]] = []
    if todo_path is not None:
        completion_boards.append((todo_path, task_header_prefix))
    completion_boards.extend(todo_boards or ())
    open_goal_ids = open_goal_ids_from_todo_boards(completion_boards)
    repository_identity = completion_tree_identity(repo_root, objective_path=objective_path)
    hierarchy = goal_graph(goals)
    goals_by_id = {item.goal_id: item for item in goals if item.goal_id}

    def descendant_states(goal_id: str) -> list[dict[str, Any]]:
        pending = list(hierarchy.get("children", {}).get(goal_id, ()))
        seen: set[str] = set()
        descendants: list[dict[str, Any]] = []
        while pending:
            child_id = str(pending.pop(0))
            if not child_id or child_id in seen:
                continue
            seen.add(child_id)
            child = goals_by_id.get(child_id)
            if child is not None:
                state = child.lifecycle_state_value
                descendants.append({
                    "goal_id": child_id,
                    "state": state,
                    "verified": state == GoalState.VERIFIED_COMPLETE.value,
                })
            pending.extend(hierarchy.get("children", {}).get(child_id, ()))
        return descendants

    for goal in candidate_goals:
        current_state = normalize_goal_state(goal.status)
        tasks_complete = goal.goal_id not in open_goal_ids
        records = persisted_records.get(goal.goal_id, [])
        criteria_text = str(
            goal.fields.get("acceptance_criteria")
            or goal.fields.get("acceptance")
            or ""
        ).strip()
        criteria: Sequence[str] | str = criteria_text or goal.required_evidence

        if not tasks_complete:
            validation_results[goal.goal_id] = {
                "attempted": False,
                "passed": False,
                "returncode": 1,
                "reason": "open_todo_tasks",
                "todo_boards": open_goal_ids[goal.goal_id],
            }
        elif current_state in {GoalState.PROVISIONALLY_COMPLETE, GoalState.VERIFIED_COMPLETE} and records:
            # Receipts supplied by another process remain provenance inputs,
            # but verification also requires the repository's current
            # validation command to pass at reconciliation time.
            validation_results[goal.goal_id] = run_goal_validation(
                repo_root=repo_root,
                goal=goal,
                repository_identity=repository_identity,
            )
            records = [
                CompletionEvidence.from_dict(
                    {
                        **record.to_dict(),
                        "validation_receipt": validation_results[goal.goal_id],
                        "validation_passed": bool(
                            validation_results[goal.goal_id].get("passed", False)
                        ),
                    }
                )
                for record in records
            ]

        decision = evaluate_goal_completion(
            current_state=current_state,
            acceptance_criteria=criteria,
            evidence=records,
            tasks_complete=tasks_complete,
            repository_tree=repository_identity.tree_id,
            repository_id=repository_identity.repository_id,
            now=now,
            freshness_seconds=evidence_freshness_seconds,
            coverage=supplied_gate_records.get(goal.goal_id, {}).get("coverage"),
            analyzer_health=supplied_gate_records.get(goal.goal_id, {}).get("analyzer_health"),
            exhaustion_quorum=supplied_gate_records.get(goal.goal_id, {}).get("exhaustion_quorum"),
            child_goals=[
                *descendant_states(goal.goal_id),
                *supplied_gate_records.get(goal.goal_id, {}).get("child_goals", ()),
            ],
            analysis_result=supplied_gate_records.get(goal.goal_id, {}).get("analysis_result"),
            analysis_inconclusive=bool(
                supplied_gate_records.get(goal.goal_id, {}).get("analysis_inconclusive", False)
            ),
        )
        decisions[goal.goal_id] = decision.to_dict()
        goal_evidence = {
            term: list(discovered_evidence.get(term, []))
            for term in goal.required_evidence
        }
        if any(goal_evidence.values()):
            completion_evidence[goal.goal_id] = goal_evidence

        if decision.state is current_state:
            continue
        reason = "; ".join(decision.actionable_reasons) or "all completion evidence gates passed"
        goal_updates = {
            "Status": decision.state.value,
            "State transitioned at": transitioned_at,
            "State transition reason": reason,
        }
        if records:
            goal_updates["Completion evidence records"] = json.dumps(
                [record.to_dict() for record in records],
                sort_keys=True,
                separators=(",", ":"),
            )
        if decision.state is GoalState.PROVISIONALLY_COMPLETE:
            provisional_goal_ids.append(goal.goal_id)
            goal_updates["Provisional at"] = transitioned_at
        elif decision.state is GoalState.VERIFIED_COMPLETE:
            completed_goal_ids.append(goal.goal_id)
            goal_updates["Completed at"] = transitioned_at
            goal_updates["Completion evidence"] = completion_evidence_summary(goal_evidence)
            goal_updates["Completion validation"] = str(
                validation_results.get(goal.goal_id, {}).get("receipt_cid", "")
            )
        elif decision.state is GoalState.REOPENED:
            reopened_goal_ids.append(goal.goal_id)
        elif decision.state is GoalState.ANALYSIS_INCONCLUSIVE:
            analysis_inconclusive_goal_ids.append(goal.goal_id)
        elif decision.state is GoalState.BLOCKED:
            blocked_goal_ids.append(goal.goal_id)
        updates[goal.goal_id] = goal_updates

    if updates:
        objective_path.write_text(rewrite_goal_fields(text, updates), encoding="utf-8")
        goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))

    state_counts = {state.value: 0 for state in GoalState}
    for goal in goals:
        state_counts[normalize_goal_state(goal.status).value] += 1

    return ObjectiveCompletionResult(
        objective_path=objective_path,
        completed_goal_ids=completed_goal_ids,
        active_goal_count=state_counts[GoalState.ACTIVE.value] + state_counts[GoalState.REOPENED.value],
        completed_goal_count=state_counts[GoalState.VERIFIED_COMPLETE.value],
        completion_evidence=completion_evidence,
        validation_results=validation_results,
        provisional_goal_ids=provisional_goal_ids,
        verified_goal_ids=list(completed_goal_ids),
        reopened_goal_ids=reopened_goal_ids,
        analysis_inconclusive_goal_ids=analysis_inconclusive_goal_ids,
        blocked_goal_ids=blocked_goal_ids,
        state_counts=state_counts,
        decisions=decisions,
    )


def ensure_objective_tracking_document(
    objective_path: Path,
    *,
    ultimate_goal: str = DEFAULT_ULTIMATE_GOAL,
    root_evidence: Sequence[str] = DEFAULT_ROOT_EVIDENCE,
    root_goal_id: str | None = None,
    goal_prefix: str = DEFAULT_GOAL_PREFIX,
    document_title: str = DEFAULT_TRACKING_DOCUMENT_TITLE,
    root_goal_title: str = DEFAULT_ROOT_GOAL_TITLE,
) -> ObjectiveTrackingResult:
    """Create the objective tracking document if it does not exist."""

    if objective_path.exists():
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    root_goal_id = root_goal_id or f"{goal_prefix}000"
    objective_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"# {document_title}",
            "",
            "This document is the supervisor planning state. It is intentionally separate from markdown todo",
            "boards: todos represent executable work, while this heap represents the objective graph used to",
            "decide what work should exist.",
            "",
            render_goal_block(
                goal_id=root_goal_id,
                title=root_goal_title,
                fields={
                    "Status": "active",
                    "Parent": "",
                    "Fib priority": str(fibonacci_priority(0)),
                    "Track": "ops",
                    "Priority": "P0",
                    "Bundle": "objective/ops/root",
                    "Goal": ultimate_goal,
                    "Evidence": ", ".join(root_evidence),
                    "Outputs": "ipfs_accelerate_py/agent_supervisor, docs",
                    "Validation": f"test -f {objective_path.as_posix()}",
                    "Refinement depth": "0",
                    "Conflict policy": "prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts",
                    "Gap task": "Refine this root objective into concrete child goals with code, tests, docs, and runtime evidence.",
                },
            ).rstrip(),
            "",
        ]
    )
    objective_path.write_text(text, encoding="utf-8")
    return ObjectiveTrackingResult(objective_path=objective_path, created=True, appended_goal_ids=[root_goal_id])


COMPONENT_SCAN_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
COMPONENT_MANIFEST_NAMES = {
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "Cargo.toml",
    "go.mod",
}
INTERFACE_DESCRIPTOR_SUFFIXES = {".idl", ".proto", ".thrift", ".graphql", ".graphqls"}


def _unique_paths(paths: Iterable[str]) -> list[str]:
    unique: list[str] = []
    for raw in paths:
        path = str(raw).strip().strip("/")
        if not path or "\0" in path or ".." in Path(path).parts:
            continue
        if path not in unique:
            unique.append(path)
    return unique


def discover_gitmodule_paths(repo_root: Path) -> list[str]:
    """Return repo-relative Git submodule paths declared in .gitmodules."""

    gitmodules = repo_root / ".gitmodules"
    if not gitmodules.exists():
        return []
    paths: list[str] = []
    for line in gitmodules.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("path") or "=" not in stripped:
            continue
        _key, value = stripped.split("=", 1)
        path = value.strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def discover_gitlink_paths(repo_root: Path) -> list[str]:
    """Return repo-relative gitlink paths from the Git index.

    Gitlinks are the authoritative submodule entries in the index.  Some repos
    can have stale or incomplete .gitmodules mappings, so interoperability
    planning must not rely on .gitmodules alone.
    """

    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--stage"],
            text=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []

    paths: list[str] = []
    for line in completed.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) != 4 or parts[0] != "160000":
            continue
        path = parts[3].strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def discover_submodule_paths(repo_root: Path) -> list[str]:
    """Return repo-relative component paths from .gitmodules and Git gitlinks."""

    return _unique_paths([*discover_gitmodule_paths(repo_root), *discover_gitlink_paths(repo_root)])


def _component_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        try:
            return path.relative_to(repo_root).as_posix()
        except ValueError:
            return path.as_posix()


def _component_scan_filename_allowed(filename: str) -> bool:
    if filename in COMPONENT_SCAN_SKIP_DIRS:
        return False
    return True


def _component_scan_dirname_allowed(dirname: str) -> bool:
    if dirname in COMPONENT_SCAN_SKIP_DIRS:
        return False
    return True


def _component_scan_path_usable(path: Path) -> bool:
    try:
        path.resolve()
    except (OSError, RuntimeError):
        return False
    return True


def _component_walk_error(_error: OSError) -> None:
    return None


def _component_path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except (OSError, RuntimeError):
        return False


def _component_path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except (OSError, RuntimeError):
        return False


def _component_path_is_symlink(path: Path) -> bool:
    try:
        return path.is_symlink()
    except (OSError, RuntimeError):
        return False


def _component_path_safe_for_file_scan(path: Path) -> bool:
    if _component_path_is_symlink(path) and not _component_scan_path_usable(path):
        return False
    return True


def _component_path_safe_for_component(path: Path) -> bool:
    if not _component_path_exists(path):
        return False
    if not _component_path_is_dir(path):
        return False
    if _component_path_is_symlink(path) and not _component_scan_path_usable(path):
        return False
    return True


def _scan_component_metadata(repo_root: Path, component_path: str, *, max_files: int = 256) -> dict[str, list[str]]:
    root = repo_root / component_path
    metadata = {
        "manifests": [],
        "interface_descriptors": [],
        "mcp_descriptors": [],
        "python_import_roots": [],
    }
    if not _component_path_safe_for_component(root):
        return metadata

    import_roots: set[str] = set()
    scanned = 0
    for current_root, dirnames, filenames in os.walk(root, onerror=_component_walk_error):
        dirnames[:] = [
            name
            for name in dirnames
            if _component_scan_dirname_allowed(name) and _component_scan_path_usable(Path(current_root) / name)
        ]
        current = Path(current_root)
        try:
            depth = len(current.relative_to(root).parts)
        except ValueError:
            depth = 0
        if depth > 4:
            dirnames[:] = []
            continue
        for filename in sorted(filenames):
            if not _component_scan_filename_allowed(filename):
                continue
            path = current / filename
            if not _component_path_safe_for_file_scan(path):
                continue
            relative = _component_relative_path(repo_root, path)
            lowered = filename.lower()
            suffix = path.suffix.lower()
            if filename in COMPONENT_MANIFEST_NAMES:
                metadata["manifests"].append(relative)
            if suffix in INTERFACE_DESCRIPTOR_SUFFIXES or any(
                token in lowered for token in ("interface", "descriptor", "contract", "schema")
            ):
                metadata["interface_descriptors"].append(relative)
            if "mcp" in lowered or "orb" in lowered:
                metadata["mcp_descriptors"].append(relative)
            if suffix == ".py" and scanned < max_files:
                scanned += 1
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for match in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_\.]*)", text, flags=re.MULTILINE):
                    import_roots.add(match.group(1).split(".", 1)[0])

    metadata["manifests"] = sorted(dict.fromkeys(metadata["manifests"]))[:40]
    metadata["interface_descriptors"] = sorted(dict.fromkeys(metadata["interface_descriptors"]))[:80]
    metadata["mcp_descriptors"] = sorted(dict.fromkeys(metadata["mcp_descriptors"]))[:80]
    metadata["python_import_roots"] = sorted(import_roots)[:80]
    return metadata


def discover_repository_components(
    repo_root: Path,
    *,
    component_paths: Sequence[str] = (),
) -> list[RepositoryComponent]:
    """Discover repository components for interoperability planning.

    Callers may provide explicit component paths, but the function also uses
    .gitmodules and Git gitlinks so it works in repos with incomplete
    .gitmodules metadata.
    """

    gitmodule_path_list = discover_gitmodule_paths(repo_root)
    gitlink_path_list = discover_gitlink_paths(repo_root)
    gitmodule_paths = set(gitmodule_path_list)
    gitlink_paths = set(gitlink_path_list)
    configured_paths = set(_unique_paths(component_paths))
    paths = _unique_paths([*component_paths, *gitmodule_path_list, *gitlink_path_list])
    components: list[RepositoryComponent] = []
    for path in paths:
        sources: list[str] = []
        if path in configured_paths:
            sources.append("configured")
        if path in gitmodule_paths:
            sources.append("gitmodules")
        if path in gitlink_paths:
            sources.append("gitlink")
        metadata = _scan_component_metadata(repo_root, path)
        components.append(
            RepositoryComponent(
                path=path,
                sources=sources,
                exists=_component_path_exists(repo_root / path),
                is_gitlink=path in gitlink_paths,
                is_gitmodule=path in gitmodule_paths,
                manifests=metadata["manifests"],
                interface_descriptors=metadata["interface_descriptors"],
                mcp_descriptors=metadata["mcp_descriptors"],
                python_import_roots=metadata["python_import_roots"],
            )
        )
    return components


def interoperability_pairs(submodules: Sequence[str], *, focus: Sequence[str] = ()) -> list[tuple[str, str]]:
    paths = [path for path in dict.fromkeys(str(item).strip() for item in submodules) if path]
    focus_paths = [path for path in dict.fromkeys(str(item).strip() for item in focus) if path]
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    if focus_paths:
        for left in focus_paths:
            if left not in paths:
                continue
            for right in paths:
                if right == left:
                    continue
                pair_key = tuple(sorted((left, right)))
                if pair_key in seen:
                    continue
                pairs.append((left, right))
                seen.add(pair_key)
        return pairs
    for left_index, left in enumerate(paths):
        for right in paths[left_index + 1 :]:
            pairs.append((left, right))
    return pairs


def interoperability_pair_key(value: str | Sequence[str]) -> str:
    """Return a stable key for an unordered interoperability component pair."""

    if isinstance(value, str):
        terms = split_terms(value)
    else:
        terms = [str(item).strip() for item in value if str(item).strip()]
    canonical_terms = [
        key
        for term in terms
        for key in [canonical_interoperability_component(term)]
        if key
    ]
    return "\0".join(sorted(canonical_terms))


def deduplicate_interoperability_goals(objective_path: Path) -> list[str]:
    """Remove duplicate interoperability goal blocks from an objective heap."""

    if not objective_path.exists():
        return []

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    goal_by_id = {goal.goal_id: goal for goal in goals}
    header_matches = list(re.finditer(r"^##\s+(\S+)\s+.+$", text, flags=re.MULTILINE))
    if not header_matches:
        return []

    preamble = text[: header_matches[0].start()].rstrip()
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(header_matches):
        start = match.start()
        end = header_matches[index + 1].start() if index + 1 < len(header_matches) else len(text)
        blocks.append((match.group(1), text[start:end].strip()))

    winner_by_pair: dict[str, str] = {}
    duplicate_goal_ids: set[str] = set()
    for goal_id, _block in blocks:
        goal = goal_by_id.get(goal_id)
        if goal is None:
            continue
        pair_key = interoperability_pair_key(str(goal.fields.get("interoperability_pair") or ""))
        if not pair_key:
            continue
        if pair_key in winner_by_pair:
            duplicate_goal_ids.add(goal_id)
            continue
        winner_by_pair[pair_key] = goal_id

    if not duplicate_goal_ids:
        return []

    retained_blocks = [block for goal_id, block in blocks if goal_id not in duplicate_goal_ids]
    rewritten = "\n\n".join(part for part in [preamble, *retained_blocks] if part).rstrip() + "\n"
    objective_path.write_text(rewritten, encoding="utf-8")
    return sorted(duplicate_goal_ids)


def _component_pair_metadata(
    left: RepositoryComponent | None,
    right: RepositoryComponent | None,
) -> dict[str, Any]:
    components = [component for component in (left, right) if component is not None]
    manifests = sorted({path for component in components for path in component.manifests})
    interface_descriptors = sorted({path for component in components for path in component.interface_descriptors})
    mcp_descriptors = sorted({path for component in components for path in component.mcp_descriptors})
    python_import_roots = sorted({root for component in components for root in component.python_import_roots})
    sources = sorted({source for component in components for source in component.sources})
    score = 1
    score += len(components)
    score += min(6, len(manifests) * 2)
    score += min(9, len(interface_descriptors) * 3)
    score += min(9, len(mcp_descriptors) * 3)
    score += min(6, len(python_import_roots))
    score += 2 if any(component.is_gitlink for component in components) else 0
    return {
        "score": score,
        "manifests": manifests[:12],
        "interface_descriptors": interface_descriptors[:16],
        "mcp_descriptors": mcp_descriptors[:16],
        "python_import_roots": python_import_roots[:24],
        "sources": sources,
    }


def _join_or_none(values: Sequence[str]) -> str:
    return ", ".join(str(value) for value in values if str(value))


def append_interoperability_goals(
    objective_path: Path,
    *,
    repo_root: Path,
    focus: Sequence[str] = (),
    component_paths: Sequence[str] = (),
    max_goals: int = 12,
    goal_prefix: str | None = None,
) -> ObjectiveTrackingResult:
    """Seed graph goals for cross-submodule integration and interoperability tests."""

    if not objective_path.exists() or max_goals <= 0:
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    components = discover_repository_components(repo_root, component_paths=component_paths)
    component_by_path = {component.path: component for component in components}
    submodules = [component.path for component in components]
    pairs = interoperability_pairs(submodules, focus=focus)
    if not pairs:
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    existing_pairs = {
        interoperability_pair_key(str(goal.fields.get("interoperability_pair") or ""))
        for goal in goals
        if str(goal.fields.get("interoperability_pair") or "").strip()
    }
    graph = goal_graph(goals)
    root_goal_id = sorted(graph.get("roots") or [goals[0].goal_id if goals else ""])[0]
    goal_prefix = goal_prefix or infer_goal_prefix(goals)
    next_id = next_goal_id(goals, prefix=goal_prefix)
    appended_blocks: list[str] = []
    appended_goal_ids: list[str] = []

    def allocate_goal_id() -> str:
        nonlocal next_id
        current = next_id
        number = int(current[len(goal_prefix) :]) + 1
        next_id = f"{goal_prefix}{number:03d}"
        return current

    for left, right in pairs:
        pair_key = interoperability_pair_key((left, right))
        if pair_key in existing_pairs:
            continue
        goal_id = allocate_goal_id()
        metadata = _component_pair_metadata(component_by_path.get(left), component_by_path.get(right))
        safe_left = safe_bundle_key(left).replace("-", "_")
        safe_right = safe_bundle_key(right).replace("-", "_")
        test_path = f"tests/integration/test_{safe_left}_{safe_right}_interop.py"
        doc_path = f"docs/integration/{safe_left}-{safe_right}.md"
        descriptor_terms = [
            *metadata["interface_descriptors"],
            *metadata["mcp_descriptors"],
        ]
        evidence_terms = [
            test_path,
            doc_path,
            f"interface contract {left} {right}",
            *descriptor_terms[:6],
        ]
        fields = {
            "Status": "active",
            "Parent": root_goal_id,
            "Fib priority": str(fibonacci_priority(1, len(appended_goal_ids))),
            "Track": "interoperability",
            "Priority": "P1",
            "Bundle": f"objective/interoperability/{safe_left}-{safe_right}",
            "Goal kind": "interoperability",
            "Interoperability pair": f"{left}, {right}",
            "Submodules": f"{left}, {right}",
            "Interoperability score": str(metadata["score"]),
            "Discovery sources": _join_or_none(metadata["sources"]),
            "Package manifests": _join_or_none(metadata["manifests"]),
            "Interface descriptors": _join_or_none(metadata["interface_descriptors"]),
            "MCP descriptors": _join_or_none(metadata["mcp_descriptors"]),
            "Python import roots": _join_or_none(metadata["python_import_roots"]),
            "Goal": (
                f"Prove `{left}` interoperates with `{right}` through importable contracts, "
                "interface descriptors, runtime handoff behavior, and integration tests."
            ),
            "Evidence": ", ".join(evidence_terms),
            "Outputs": ", ".join([test_path, doc_path, left, right, *descriptor_terms[:4]]),
            "Validation": "python -m pytest tests/integration -q",
            "Refinement depth": "1",
            "Embedding query": (
                f"{left} {right} interoperability integration test interface descriptor "
                f"{' '.join(metadata['python_import_roots'][:12])}"
            ),
            "AST query": ", ".join(
                [
                    left,
                    right,
                    "interface contract",
                    "integration test",
                    *metadata["python_import_roots"][:12],
                ]
            ),
            "Parallel lane": f"objective/interoperability/{safe_left}-{safe_right}",
            "Conflict policy": "keep pair-specific integration edits isolated; use the LLM merge resolver for conflicts",
            "Gap task": (
                f"Create one larger integration work item proving `{left}` and `{right}` can be used together, "
                "including a test, a contract note, and any adapter code needed by the objective."
            ),
        }
        appended_blocks.append(render_goal_block(goal_id=goal_id, title=f"Interoperate {left} with {right}", fields=fields))
        appended_goal_ids.append(goal_id)
        existing_pairs.add(pair_key)
        if len(appended_goal_ids) >= max_goals:
            break

    if appended_blocks:
        objective_path.write_text(text.rstrip() + "\n\n" + "\n\n".join(block.strip() for block in appended_blocks) + "\n", encoding="utf-8")
    return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=appended_goal_ids)


LAUNCH_READINESS_GOAL_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "key": "hallucinate-mcp-dashboard-capability-catalog",
        "title": "Hallucinate App MCP dashboard capability catalog",
        "track": "launch",
        "priority": "P0",
        "submodules": "hallucinate_app, swissknife, external/ipfs_accelerate, external/ipfs_datasets, external/ipfs_kit",
        "goal": (
            "Hallucinate App menus and dashboards expose ipfs_accelerate_py, ipfs_datasets_py, "
            "and ipfs_kit_py MCP server dashboards, daemon health, tools/list, and tools/call "
            "so Swissknife can test backend interoperability from the UI."
        ),
        "evidence": (
            "hallucinate_app menus, Hallucinate App MCP dashboard, dashboard capability catalog, "
            "daemon health, tools/list, tools/call, ipfs_accelerate_py MCP server, "
            "ipfs_datasets_py MCP server, ipfs_kit_py MCP server, Swissknife applications, "
            "Playwright MCP dashboard interoperability, launch Playwright validation gate"
        ),
        "outputs": (
            "hallucinate_app, swissknife, external/ipfs_accelerate, external/ipfs_datasets, "
            "external/ipfs_kit, hallucinate_app/test/e2e/mcp-feature-exposure.spec.ts, "
            "hallucinate_app/test/e2e/mcp-dashboard-interoperability.spec.ts"
        ),
        "validation": (
            "npm --prefix hallucinate_app run test:e2e -- "
            "mcp-feature-exposure.spec.ts mcp-dashboard-interoperability.spec.ts"
        ),
        "embedding_query": (
            "Hallucinate App MCP dashboard dashboard capability catalog tools/list tools/call "
            "ipfs_accelerate_py ipfs_datasets_py ipfs_kit_py Swissknife Playwright"
        ),
        "ast_query": (
            "hallucinate_app, swissknife, ipfs_accelerate_py, ipfs_datasets_py, ipfs_kit_py, "
            "tools/list, tools/call, daemon health, MCP dashboard"
        ),
        "gap_task": (
            "Create or repair the Hallucinate App dashboard/menu integration so it lists and "
            "calls each backend MCP server tool, then cover it with Playwright."
        ),
    },
    {
        "key": "swissknife-mcp-plus-plus-server-dashboard-interop",
        "title": "Swissknife MCP++ server dashboard interoperability",
        "track": "launch",
        "priority": "P0",
        "submodules": "swissknife, hallucinate_app, Mcp-Plus-Plus, external/ipfs_accelerate, external/ipfs_datasets, external/ipfs_kit",
        "goal": (
            "Swissknife applications consume MCP++ compatible control-plane contracts from "
            "Hallucinate App and the ipfs_accelerate_py, ipfs_datasets_py, and ipfs_kit_py MCP servers."
        ),
        "evidence": (
            "Swissknife applications, Mcp-Plus-Plus, MCP++ compatibility, MCP server dashboard, "
            "dashboard capability catalog, ipfs_accelerate_py, ipfs_datasets_py, ipfs_kit_py, "
            "tools/list, tools/call, control plane"
        ),
        "outputs": (
            "swissknife, Mcp-Plus-Plus, hallucinate_app, external/ipfs_accelerate, "
            "external/ipfs_datasets, external/ipfs_kit, swissknife/test/e2e/mcp-dashboard.spec.ts"
        ),
        "validation": "npm --prefix swissknife run test:e2e:mcp",
        "embedding_query": (
            "Swissknife MCP++ MCP server dashboard control plane tools/list tools/call "
            "ipfs_accelerate_py ipfs_datasets_py ipfs_kit_py"
        ),
        "ast_query": (
            "swissknife, Mcp-Plus-Plus, MCP++, MCP server, tools/list, tools/call, control plane"
        ),
        "gap_task": (
            "Implement the Swissknife-facing MCP++ dashboard contract and tests that prove "
            "Swissknife can enumerate and invoke backend services through the control plane."
        ),
    },
    {
        "key": "cross-device-virtual-desktop-offload-replay",
        "title": "Cross-device virtual desktop offload launch replay",
        "track": "launch",
        "priority": "P0",
        "submodules": "mobile, swissknife, hallucinate_app, external/ipfs_accelerate, external/ipfs_datasets, external/ipfs_kit",
        "goal": (
            "A phone-hosted Swissknife virtual desktop can discover a desktop peer, offload "
            "compute, route Hallucinate App mediation through IPFS/libp2p/MCP++, and produce "
            "a launch readiness receipt."
        ),
        "evidence": (
            "phone-hosted Swissknife virtual desktop, desktop peer offload, mobile phone, "
            "Hallucinate App mediation, IPFS, libp2p, MCP++, launch readiness receipt, "
            "cross-device e2e validation, Playwright launch replay"
        ),
        "outputs": (
            "mobile, swissknife, hallucinate_app, external/ipfs_accelerate, external/ipfs_datasets, "
            "external/ipfs_kit, tests/test_virtual_ai_os_launch_readiness_gate.py"
        ),
        "validation": (
            "PYTHONPATH=external/ipfs_accelerate:external/ipfs_datasets "
            "pytest tests/test_virtual_ai_os_launch_readiness_gate.py -q"
        ),
        "embedding_query": (
            "phone hosted Swissknife virtual desktop desktop peer offload mobile IPFS libp2p "
            "MCP++ launch readiness receipt Playwright"
        ),
        "ast_query": (
            "mobile, swissknife, hallucinate_app, desktop peer offload, launch readiness, Playwright"
        ),
        "gap_task": (
            "Build the deterministic cross-device launch replay and receipt path that proves "
            "phone-to-desktop offload works through the planned control plane."
        ),
    },
    {
        "key": "meta-glasses-control-plane-input-routing",
        "title": "Meta glasses control-plane input routing",
        "track": "launch",
        "priority": "P0",
        "submodules": "external/meta-wearables-dat-android, external/meta-wearables-dat-ios, mobile, swissknife, hallucinate_app",
        "goal": (
            "Meta glasses camera, microphone, headphones, Neural Band, and captouch inputs "
            "route through the mobile phone into Swissknife applications and the control plane "
            "using Bluetooth, Wi-Fi, IPFS/libp2p, and MCP++ compatible envelopes where possible."
        ),
        "evidence": (
            "Meta glasses interface, Meta Wearables DAT, camera, microphone, headphones, "
            "Neural Band, captouch, Bluetooth transport, Wi-Fi transport, mobile phone, "
            "Swissknife applications, IPFS, libp2p, MCP++, control plane"
        ),
        "outputs": (
            "external/meta-wearables-dat-android, external/meta-wearables-dat-ios, mobile, "
            "swissknife, hallucinate_app, tests/test_hallucinate_multimodal_control_todo_queue.py, "
            "tests/test_virtual_ai_os_launch_readiness_gate.py"
        ),
        "validation": (
            "PYTHONPATH=external/ipfs_accelerate:external/ipfs_datasets "
            "pytest tests/test_hallucinate_multimodal_control_todo_queue.py "
            "tests/test_virtual_ai_os_launch_readiness_gate.py -q"
        ),
        "embedding_query": (
            "Meta glasses camera microphone headphones Neural Band captouch Bluetooth Wi-Fi "
            "Swissknife control plane IPFS libp2p MCP++"
        ),
        "ast_query": (
            "Meta Wearables DAT, camera, microphone, headphones, Neural Band, captouch, "
            "Bluetooth, Wi-Fi, Swissknife, control plane"
        ),
        "gap_task": (
            "Research and codify the Meta glasses input contracts, mocks, and transport tests "
            "that let Swissknife applications consume those interaction methods."
        ),
    },
    {
        "key": "hallucinate-daemon-launch-orchestration",
        "title": "Hallucinate App daemon launch orchestration",
        "track": "launch",
        "priority": "P0",
        "submodules": "hallucinate_app, swissknife, external/ipfs_accelerate, external/ipfs_datasets, external/ipfs_kit",
        "goal": (
            "Hallucinate App launches and monitors the ipfs_accelerate_py, ipfs_datasets_py, "
            "and ipfs_kit_py MCP daemons, exposes their health in dashboards, and hands "
            "capability records to Swissknife."
        ),
        "evidence": (
            "Hallucinate App daemon health, daemon launcher, MCP server, MCP dashboard, "
            "ipfs_accelerate_py, ipfs_datasets_py, ipfs_kit_py, dashboard capability catalog, "
            "Swissknife applications, launch Playwright validation gate"
        ),
        "outputs": (
            "hallucinate_app, swissknife, external/ipfs_accelerate, external/ipfs_datasets, "
            "external/ipfs_kit, hallucinate_app/test/e2e/daemon-launch-health.spec.ts"
        ),
        "validation": (
            "PYTHONPATH=external/ipfs_accelerate:external/ipfs_datasets "
            "pytest tests/test_hallucinate_multimodal_control_todo_queue.py -q && "
            "(test ! -f hallucinate_app/package.json || npm --prefix hallucinate_app run test:e2e -- daemon-launch-health.spec.ts) && "
            "(test ! -f swissknife/package.json || npm --prefix swissknife run test:e2e:meta-glasses) && "
            "(test ! -f hallucinate_app/package.json || npm --prefix hallucinate_app run test:e2e -- multimodal-control-surface.spec.ts)"
        ),
        "embedding_query": (
            "Hallucinate App daemon launch health MCP server dashboard ipfs_accelerate_py "
            "ipfs_datasets_py ipfs_kit_py Swissknife"
        ),
        "ast_query": (
            "hallucinate_app, daemon health, MCP server, MCP dashboard, ipfs_accelerate_py, "
            "ipfs_datasets_py, ipfs_kit_py"
        ),
        "gap_task": (
            "Make Hallucinate App own daemon launch and health reporting for the backend MCP "
            "servers, with UI and integration tests that Swissknife can exercise."
        ),
    },
    {
        "key": "objective-heap-autosteer-validation-repair",
        "title": "Objective heap active steering and validation repair",
        "track": "launch",
        "priority": "P0",
        "submodules": "external/ipfs_accelerate, hallucinate_app, swissknife, mobile",
        "goal": (
            "The supervisor actively manages the objective heap and todo boards by adding, "
            "deprioritizing, and repairing goals, subgoals, tasks, and subtasks from validation "
            "results including Playwright launch replays."
        ),
        "evidence": (
            "objective heap, fibonacci priority, supervisor active management, failed validation "
            "repair, Playwright launch replay, HAO task board, MGW task board, VAI task board, "
            "production readiness, launch Playwright validation gate"
        ),
        "outputs": (
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor, "
            "tests/test_supervisor_objective_task_janitor.py, "
            "tests/test_reconciliation_guardrail_refresh.py"
        ),
        "validation": (
            "PYTHONPATH=external/ipfs_accelerate "
            "pytest tests/test_supervisor_objective_task_janitor.py "
            "tests/test_reconciliation_guardrail_refresh.py -q"
        ),
        "embedding_query": (
            "objective heap fibonacci priority supervisor active management failed validation "
            "repair Playwright VAI MGW HAO production readiness"
        ),
        "ast_query": (
            "objective heap, supervisor, validation repair, Playwright, VAI, MGW, HAO"
        ),
        "gap_task": (
            "Extend the supervisor loop so failed validation and stale idle lanes generate "
            "mission-aligned follow-up tasks and subgoals instead of generic reconciliation churn."
        ),
    },
)


def normalize_launch_key(value: str) -> str:
    return safe_bundle_key(value).strip("-")


def append_launch_readiness_goals(
    objective_path: Path,
    *,
    repo_root: Path,
    max_goals: int = 8,
    goal_prefix: str | None = None,
) -> ObjectiveTrackingResult:
    """Seed high-value launch-readiness goals for the Swissknife virtual desktop plan."""

    _ = repo_root
    if not objective_path.exists() or max_goals <= 0:
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    existing_keys = {
        normalize_launch_key(str(goal.fields.get("launch_key") or ""))
        for goal in goals
        if str(goal.fields.get("launch_key") or "").strip()
    }
    existing_bundles = {
        normalize_launch_key(str(goal.fields.get("bundle") or "").removeprefix("objective/launch/"))
        for goal in goals
        if str(goal.fields.get("bundle") or "").startswith("objective/launch/")
    }
    existing_keys.update(existing_bundles)
    graph = goal_graph(goals)
    root_goal_id = sorted(graph.get("roots") or [goals[0].goal_id if goals else ""])[0]
    goal_prefix = goal_prefix or infer_goal_prefix(goals)
    next_id = next_goal_id(goals, prefix=goal_prefix)
    appended_blocks: list[str] = []
    appended_goal_ids: list[str] = []

    def allocate_goal_id() -> str:
        nonlocal next_id
        current = next_id
        number = int(current[len(goal_prefix) :]) + 1
        next_id = f"{goal_prefix}{number:03d}"
        return current

    for template in LAUNCH_READINESS_GOAL_TEMPLATES:
        launch_key = normalize_launch_key(str(template["key"]))
        if launch_key in existing_keys:
            continue
        goal_id = allocate_goal_id()
        fields = {
            "Status": "active",
            "Parent": root_goal_id,
            "Fib priority": str(fibonacci_priority(1, len(appended_goal_ids))),
            "Track": str(template["track"]),
            "Priority": str(template["priority"]),
            "Bundle": f"objective/launch/{launch_key}",
            "Goal kind": "launch_readiness",
            "Launch key": launch_key,
            "Submodules": str(template["submodules"]),
            "Mission terms": str(template["evidence"]),
            "Goal": str(template["goal"]),
            "Evidence": str(template["evidence"]),
            "Outputs": str(template["outputs"]),
            "Validation": str(template["validation"]),
            "Refinement depth": "1",
            "Embedding query": str(template["embedding_query"]),
            "AST query": str(template["ast_query"]),
            "Parallel lane": f"objective/launch/{launch_key}",
            "Conflict policy": (
                "prefer launch-critical integration evidence; use the LLM merge resolver "
                "when dashboard, daemon, or mobile control-plane edits conflict"
            ),
            "Gap task": str(template["gap_task"]),
        }
        appended_blocks.append(render_goal_block(goal_id=goal_id, title=str(template["title"]), fields=fields))
        appended_goal_ids.append(goal_id)
        existing_keys.add(launch_key)
        if len(appended_goal_ids) >= max_goals:
            break

    if appended_blocks:
        objective_path.write_text(
            text.rstrip() + "\n\n" + "\n\n".join(block.strip() for block in appended_blocks) + "\n",
            encoding="utf-8",
        )
    return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=appended_goal_ids)


def existing_refinement_keys(goals: Sequence[ObjectiveGoal]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for goal in goals:
        for parent in goal.parent_goal_ids:
            for evidence in goal.required_evidence:
                keys.add((parent, normalize_evidence_key(evidence)))
    return keys


def normalize_evidence_key(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def refinement_title(parent_title: str, evidence: str) -> str:
    compact = " ".join(evidence.strip().split())
    if len(compact) > 72:
        compact = compact[:69].rstrip() + "..."
    return f"Prove {compact} for {parent_title}"


def refinement_fields(finding: ObjectiveFinding, *, evidence: str, depth: int, sibling_index: int) -> dict[str, str]:
    outputs = ", ".join(finding.outputs) if finding.outputs else "ipfs_accelerate_py/agent_supervisor, docs, tests"
    return {
        "Status": "active",
        "Parent": finding.goal_id,
        "Fib priority": str(fibonacci_priority(depth, sibling_index)),
        "Track": finding.track,
        "Priority": finding.priority,
        "Bundle": finding.bundle_key,
        "Goal": f"Create concrete implementation, tests, docs, or interface descriptors proving `{evidence}`.",
        "Evidence": evidence,
        "Outputs": outputs,
        "Validation": finding.validation,
        "Refinement depth": str(depth),
        "Embedding query": evidence,
        "AST query": evidence,
        "Parallel lane": finding.parallel_lane,
        "Conflict policy": finding.conflict_policy,
        "Gap task": f"Close the missing objective evidence `{evidence}` with a narrow, verifiable change.",
    }


def append_refinement_goals(
    objective_path: Path,
    findings: Sequence[ObjectiveFinding],
    *,
    max_children_per_finding: int = 3,
    max_depth: int = 4,
    goal_prefix: str | None = None,
) -> ObjectiveTrackingResult:
    """Append child goals for missing evidence terms that are still too broad."""

    if not objective_path.exists() or max_children_per_finding <= 0:
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    graph = goal_graph(goals)
    refinement_keys = existing_refinement_keys(goals)
    appended_blocks: list[str] = []
    appended_goal_ids: list[str] = []
    goal_prefix = goal_prefix or infer_goal_prefix(goals)
    next_id = next_goal_id(goals, prefix=goal_prefix)

    def allocate_goal_id() -> str:
        nonlocal next_id
        current = next_id
        number = int(current[len(goal_prefix) :]) + 1
        next_id = f"{goal_prefix}{number:03d}"
        return current

    for finding in findings:
        parent_depth = int(graph.get("depths", {}).get(finding.goal_id, finding.graph_depth))
        child_depth = parent_depth + 1
        if child_depth > max_depth:
            continue
        created_for_parent = 0
        for evidence in finding.missing_evidence:
            key = (finding.goal_id, normalize_evidence_key(evidence))
            if key in refinement_keys:
                continue
            goal_id = allocate_goal_id()
            fields = refinement_fields(
                finding,
                evidence=evidence,
                depth=child_depth,
                sibling_index=created_for_parent,
            )
            appended_blocks.append(
                render_goal_block(
                    goal_id=goal_id,
                    title=refinement_title(finding.title, evidence),
                    fields=fields,
                )
            )
            appended_goal_ids.append(goal_id)
            refinement_keys.add(key)
            created_for_parent += 1
            if created_for_parent >= max_children_per_finding:
                break

    if appended_blocks:
        objective_path.write_text(text.rstrip() + "\n\n" + "\n\n".join(block.strip() for block in appended_blocks) + "\n", encoding="utf-8")
    return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=appended_goal_ids)


def thought_node_id(kind: str, *parts: str) -> str:
    seed = "\0".join(str(part) for part in parts)
    digest = sha1(seed.encode("utf-8")).hexdigest()[:12]
    safe_kind = re.sub(r"[^a-z0-9_]+", "_", kind.lower()).strip("_") or "thought"
    return f"{safe_kind}:{digest}"


def build_objective_thought_graph(goals: Sequence[ObjectiveGoal]) -> dict[str, Any]:
    """Build typed thought nodes from objective goals for integration planning."""

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    def add_node(node_id: str, **payload: Any) -> None:
        nodes.setdefault(node_id, {"id": node_id, **payload})

    def add_edge(source: str, target: str, kind: str) -> None:
        edges.append({"from": source, "to": target, "kind": kind})

    for goal in goals:
        goal_node = f"goal:{goal.goal_id}"
        outputs = split_terms(str(goal.fields.get("outputs") or ""))
        validation = str(goal.fields.get("validation") or "").strip()
        interop_pair = split_terms(str(goal.fields.get("interoperability_pair") or ""))
        submodules = split_terms(str(goal.fields.get("submodules") or ""))
        package_manifests = split_terms(str(goal.fields.get("package_manifests") or ""))
        interface_descriptors = split_terms(str(goal.fields.get("interface_descriptors") or ""))
        mcp_descriptors = split_terms(str(goal.fields.get("mcp_descriptors") or ""))
        add_node(
            goal_node,
            kind="goal",
            goal_id=goal.goal_id,
            title=goal.title,
            status=goal.status,
            track=str(goal.fields.get("track") or "ops"),
            priority=str(goal.fields.get("priority") or "P2"),
            thought=(
                str(goal.fields.get("goal") or goal.title)
                or "Decide what implementation evidence proves this objective."
            ),
        )
        for term in goal.required_evidence:
            evidence_node = thought_node_id("evidence", goal.goal_id, term)
            add_node(
                evidence_node,
                kind="evidence_requirement",
                goal_id=goal.goal_id,
                term=term,
                thought=f"Find or create repository evidence proving `{term}`.",
            )
            add_edge(goal_node, evidence_node, "requires_evidence")
        if validation:
            validation_node = thought_node_id("validation", goal.goal_id, validation)
            add_node(
                validation_node,
                kind="validation_strategy",
                goal_id=goal.goal_id,
                command=validation,
                thought="Run this validation before closing the goal.",
            )
            add_edge(goal_node, validation_node, "validated_by")
        for output in outputs:
            surface_node = thought_node_id("code_surface", goal.goal_id, output)
            add_node(
                surface_node,
                kind="code_surface",
                goal_id=goal.goal_id,
                path=output,
                thought=f"Inspect or modify `{output}` as part of this objective.",
            )
            add_edge(goal_node, surface_node, "touches_surface")
        if interop_pair or submodules:
            pair_values = interop_pair or submodules
            interop_node = thought_node_id("interoperability_pair", goal.goal_id, ",".join(pair_values))
            add_node(
                interop_node,
                kind="interoperability_pair",
                goal_id=goal.goal_id,
                submodules=pair_values,
                thought="Prove these components interoperate through contracts and tests.",
            )
            add_edge(goal_node, interop_node, "targets_interoperability")
            test_node = thought_node_id("test_strategy", goal.goal_id, ",".join(pair_values))
            add_node(
                test_node,
                kind="test_strategy",
                goal_id=goal.goal_id,
                submodules=pair_values,
                thought="Write or update integration tests that exercise the shared runtime boundary.",
            )
            add_edge(interop_node, test_node, "needs_test_strategy")
            for manifest in package_manifests:
                manifest_node = thought_node_id("package_manifest", goal.goal_id, manifest)
                add_node(
                    manifest_node,
                    kind="package_manifest",
                    goal_id=goal.goal_id,
                    path=manifest,
                    thought=f"Use `{manifest}` to identify package entrypoints and dependency surfaces.",
                )
                add_edge(interop_node, manifest_node, "uses_package_manifest")
            for descriptor in interface_descriptors:
                descriptor_node = thought_node_id("interface_descriptor", goal.goal_id, descriptor)
                add_node(
                    descriptor_node,
                    kind="interface_descriptor",
                    goal_id=goal.goal_id,
                    path=descriptor,
                    thought=f"Map `{descriptor}` into the interoperability contract.",
                )
                add_edge(interop_node, descriptor_node, "uses_interface_descriptor")
            for descriptor in mcp_descriptors:
                mcp_node = thought_node_id("mcp_descriptor", goal.goal_id, descriptor)
                add_node(
                    mcp_node,
                    kind="mcp_descriptor",
                    goal_id=goal.goal_id,
                    path=descriptor,
                    thought=f"Use `{descriptor}` as an MCP or ORB capability boundary.",
                )
                add_edge(interop_node, mcp_node, "uses_mcp_descriptor")

    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.objective_thought_graph",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [nodes[node_id] for node_id in sorted(nodes)],
        "edges": edges,
    }


def write_objective_graph_artifact(
    *,
    objective_path: Path,
    graph_path: Path,
) -> dict[str, Any]:
    """Write a JSON graph artifact for the current objective heap."""

    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8")) if objective_path.exists() else []
    graph = goal_graph(goals)
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.objective_graph",
        "generated_at": utc_now(),
        "objective_path": str(objective_path),
        "goal_count": len(goals),
        "active_goal_count": sum(1 for goal in goals if goal.status in {"active", "todo", "open"}),
        "completed_goal_count": sum(1 for goal in goals if goal.status == "completed"),
        "heap_schedule": [record.to_dict() for record in objective_heap_schedule(goals)],
        "thought_graph": build_objective_thought_graph(goals),
        "goals": [
            {
                "goal_id": goal.goal_id,
                "title": goal.title,
                "status": goal.status,
                "fib_priority": goal.priority[0],
                "parents": goal.parent_goal_ids,
                "evidence": goal.required_evidence,
                "track": goal.fields.get("track", "ops"),
                "bundle": goal.fields.get("bundle", ""),
                "refinement_depth": goal.fields.get("refinement_depth", str(graph["depths"].get(goal.goal_id, 0))),
            }
            for goal in sorted(goals, key=lambda item: item.priority)
        ],
        "graph": graph,
    }
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_root_evidence(values: Iterable[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        terms.extend(split_terms(value))
    return terms or list(DEFAULT_ROOT_EVIDENCE)
