#!/usr/bin/env python3
"""Materialize and verify the sealed APMC board in its DuckDB authority.

Initial materialization is allowed only while no Quack owner is serving the
new database.  Once Quack owns the file, verification must use ``--endpoint``
so this script never becomes a second file owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import shlex
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (  # noqa: E402
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (  # noqa: E402
    MAX_QUERY_LIMIT,
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (  # noqa: E402
    is_quack_transport_target,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (  # noqa: E402
    COMPLETION_EVIDENCE_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    DatabaseImplementationDaemon,
    parse_task_text,
)

PROGRAM_ID = "agent-supervisor-autonomous-meta-controller-v1"
ROOT_OBJECTIVE = "APMC-G000"
BRANCH = "codex/agent-supervisor-autonomous-meta-controller-v1"
TODO_PATH = REPO_ROOT / "docs/architecture/agent_supervisor_autonomous_meta_controller.todo.md"
OBJECTIVES_PATH = (
    REPO_ROOT / "docs/architecture/agent_supervisor_autonomous_meta_controller.objectives.md"
)
PLAN_PATH = REPO_ROOT / "docs/architecture/AGENT_SUPERVISOR_AUTONOMOUS_META_CONTROLLER_PLAN.md"
VALIDATOR_PATH = REPO_ROOT / "scripts/validate_agent_supervisor_autonomous_meta_controller_board.py"
EXPECTED_TASK_IDS = tuple(f"APMC-{index:03d}" for index in range(21))
EXPECTED_GOAL_IDS = tuple(["APMC-G000", *(f"APMC-G{index:03d}" for index in range(10, 111, 10))])
EXPECTED_MANUAL_REVIEW_TASK_IDS = ("APMC-019", "APMC-020")
BASELINE_TASK_ID = "APMC-000"
BASELINE_COMPLETION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/baseline-current-tree-completion@1"
)
BASELINE_VALIDATION_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/baseline-current-tree-validation-evidence@1"
)
BASELINE_VALIDATION_EVIDENCE_KIND = "apmc_baseline_current_tree_validation"
BASELINE_VALIDATION_SET_EVIDENCE_KIND = "apmc_baseline_validation_set"
PRELAUNCH_QUALIFIED_TASK_IDS = (
    "APMC-000",
    "APMC-001",
    "APMC-018",
    "APMC-002",
    "APMC-003",
    "APMC-004",
    "APMC-005",
)
PRELAUNCH_COMPLETED_TASK_IDS = tuple(f"APMC-{index:03d}" for index in range(6)) + ("APMC-018",)
PRELAUNCH_READY_TASK_IDS = ("APMC-006", "APMC-012", "APMC-014")
PRELAUNCH_COMPLETION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/prelaunch-current-tree-completion@1"
)
PRELAUNCH_VALIDATION_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/prelaunch-current-tree-validation-evidence@1"
)
PRELAUNCH_VALIDATION_EVIDENCE_KIND = "apmc_prelaunch_current_tree_validation"
PRELAUNCH_VALIDATION_SET_EVIDENCE_KIND = "apmc_prelaunch_validation_set"
MAX_VALIDATION_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_BASELINE_VALIDATION_ENTRY_BYTES = 64 * 1024
MAX_BASELINE_COMPLETION_RECEIPT_BYTES = 512 * 1024
TRUSTED_PYTHON = "/usr/bin/python3"
TRUSTED_GIT = "/usr/bin/git"
TRUSTED_VALIDATION_PATH = "/usr/bin:/bin"
ValidationRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[bytes]]
SourceIdentityReader = Callable[[], tuple[str, str]]
VALIDATION_ENVIRONMENT_NAMES = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "TEMP",
    "TMP",
    "TMPDIR",
    "TZ",
)
_TASK_RELATION_FIELDS = frozenset(
    {
        "task_cid",
        "task_id",
        "task_alias",
        "cid",
        "goal_cid",
        "goal_id",
        "depends_on",
        "dependencies",
        "effects",
        "outputs",
        "acceptance_criteria",
        "acceptance",
        "validation_commands",
        "validations",
        "status",
        "priority",
        "ordinal",
        "plan_cid",
        "objective_id",
    }
)


class MaterializationError(RuntimeError):
    """The board cannot be safely materialized or its projection changed."""


def _git(*args: str) -> str:
    result = subprocess.run(
        [TRUSTED_GIT, *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise MaterializationError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sealed_validation_environment() -> dict[str, str]:
    return {
        **{
            name: os.environ[name]
            for name in VALIDATION_ENVIRONMENT_NAMES
            if name in os.environ and name != "PATH"
        },
        "PATH": TRUSTED_VALIDATION_PATH,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }


def _json_size(payload: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[,;]", value) if item.strip())


def _require_clean_committed_source() -> tuple[str, str]:
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise MaterializationError(
            "refusing to bind APMC state to a dirty worktree; commit the exact source first"
        )
    branch = _git("branch", "--show-current")
    if branch != BRANCH:
        raise MaterializationError(f"expected branch {BRANCH!r}, observed {branch!r}")
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    return head, tree


def _safe_new_database(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise MaterializationError("database must be inside this repository worktree") from exc
    if not relative.parts or relative.parts[0] not in {"data", "state"}:
        raise MaterializationError("database must be under ignored data/ or state/")
    if lexical.suffix.lower() not in {".duckdb", ".ddb"}:
        raise MaterializationError("database must have a .duckdb or .ddb suffix")
    current = REPO_ROOT
    for component in relative.parts[:-1]:
        current /= component
        if current.is_symlink():
            raise MaterializationError("database parent may not traverse a symlink")
        if current.exists() and not current.is_dir():
            raise MaterializationError("database parent must be a directory")
    if lexical.is_symlink():
        raise MaterializationError("database target may not be a symlink")
    return lexical


def _runtime_output_relative(value: str | Path, *, noun: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise MaterializationError(f"{noun} must remain inside the repository") from exc
    if not relative.parts or relative.parts[0] not in {"data", "state"}:
        raise MaterializationError(f"{noun} must be under ignored data/ or state/")
    if relative.suffix.lower() != ".json":
        raise MaterializationError(f"{noun} must have a .json suffix")
    return relative


def _write_runtime_receipt(value: str | Path, payload: Mapping[str, Any]) -> Path:
    """Publish a new receipt through no-follow directory descriptors."""

    relative = _runtime_output_relative(value, noun="receipt")
    directory_fd = os.open(REPO_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    temporary_name = f".{relative.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        for component in relative.parts[:-1]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise MaterializationError(
                    "receipt parent may not traverse a symlink or non-directory"
                ) from exc
            os.close(directory_fd)
            directory_fd = next_fd
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(
                    temporary_name,
                    relative.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise MaterializationError("receipt target already exists") from exc
            os.unlink(temporary_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
    finally:
        os.close(directory_fd)
    return REPO_ROOT / relative


def _validated_inputs() -> tuple[list[Any], list[Any]]:
    result = subprocess.run(
        [TRUSTED_PYTHON, str(VALIDATOR_PATH), "--check-all"],
        cwd=REPO_ROOT,
        env=_sealed_validation_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise MaterializationError(
            "APMC validator refused materialization: " + (result.stdout or result.stderr).strip()
        )
    tasks = parse_task_text(
        TODO_PATH.read_text(encoding="utf-8"),
        path=TODO_PATH,
        task_header_prefix="## APMC-",
    )
    goals = parse_goal_heap(OBJECTIVES_PATH.read_text(encoding="utf-8"))
    if tuple(task.task_id for task in tasks) != EXPECTED_TASK_IDS:
        raise MaterializationError("task population changed after validation")
    if tuple(goal.goal_id for goal in goals) != EXPECTED_GOAL_IDS:
        raise MaterializationError("goal population changed after validation")
    return tasks, goals


def build_population(*, source_head: str, source_tree: str) -> dict[str, Any]:
    tasks, goals = _validated_inputs()
    source_bindings = {
        "source_head": source_head,
        "repository_tree_id": source_tree,
        "plan_sha256": _sha256(PLAN_PATH),
        "objectives_sha256": _sha256(OBJECTIVES_PATH),
        "taskboard_sha256": _sha256(TODO_PATH),
    }
    plan_root = content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/accepted-plan-root@1",
            "program_id": PROGRAM_ID,
            **source_bindings,
        }
    )
    goal_cids = {
        goal.goal_id: content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/goal@1",
                "program_id": PROGRAM_ID,
                "goal_id": goal.goal_id,
                "title": goal.title,
                "source_tree": source_tree,
            }
        )
        for goal in goals
    }
    objective_rows: list[dict[str, Any]] = []
    for ordinal, goal in enumerate(goals, start=1):
        parent = str(goal.fields.get("parent") or "").strip()
        objective_rows.append(
            {
                "goal_cid": goal_cids[goal.goal_id],
                "goal_id": goal.goal_id,
                "goal_alias": goal.goal_id,
                "title": goal.title,
                "objective_id": ROOT_OBJECTIVE if goal.goal_id == ROOT_OBJECTIVE else "",
                "objective_alias": ROOT_OBJECTIVE,
                "parent_goal_cid": goal_cids[parent] if parent else "",
                "ordinal": ordinal,
                "status": "open",
                "priority": str(goal.fields.get("priority") or "P2"),
                "program_id": PROGRAM_ID,
                "source_tree": source_tree,
                "fields": dict(goal.fields),
            }
        )
    goal_edges: list[dict[str, str]] = []
    for goal in goals:
        parent = str(goal.fields.get("parent") or "").strip()
        if parent:
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[parent],
                    "child_goal_cid": goal_cids[goal.goal_id],
                    "edge_kind": "goal_parent",
                }
            )
        for dependency in _csv(str(goal.fields.get("depends_on") or "")):
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[dependency],
                    "child_goal_cid": goal_cids[goal.goal_id],
                    "edge_kind": "goal_dependency",
                }
            )
    task_cids_by_alias = {task.task_id: task.canonical_task_cid for task in tasks}
    task_rows: list[dict[str, Any]] = []
    for ordinal, task in enumerate(tasks, start=1):
        validation_rows = [
            {
                "argv": shlex.split(command),
                "current_tree_required": True,
                "shell_interpolation_permitted": False,
            }
            for command in task.validation
        ]
        task_rows.append(
            {
                "task_cid": task.canonical_task_cid,
                "task_id": task.task_id,
                "task_alias": task.task_id,
                "task_key": task.canonical_task_key,
                "goal_cid": goal_cids[str(task.metadata["goal id"])],
                "goal_id": str(task.metadata["goal id"]),
                "plan_cid": plan_root,
                "objective_id": ROOT_OBJECTIVE,
                "ordinal": ordinal,
                "status": "todo",
                "priority": task.priority,
                "title": task.title,
                "objective": task.title,
                "track": task.track,
                "completion": task.completion,
                "review_only": str(task.metadata["review only"]).strip().casefold()
                in {"1", "true", "yes"},
                "board_namespace": PROGRAM_ID,
                "repository_tree_id": source_tree,
                "source_head": source_head,
                "source_line": task.source_line,
                "metadata": dict(task.metadata),
                "provenance": {
                    "board_namespace": PROGRAM_ID,
                    "source_path": TODO_PATH.relative_to(REPO_ROOT).as_posix(),
                    "acceptance": task.acceptance,
                },
                # Bind aliases before insertion so a dependency on a later
                # task ordinal cannot remain an unresolved display alias.
                "dependencies": [task_cids_by_alias[dependency] for dependency in task.depends_on],
                "outputs": [
                    {
                        "path": path,
                        "effect_class": "bounded_repository_path",
                        "effect_id": content_identity({"task_id": task.task_id, "path": path}),
                    }
                    for path in task.outputs
                ],
                "acceptance": [
                    {
                        "criterion": task.acceptance,
                        "current_tree_required": True,
                        "declared_validation_required": True,
                        "markdown_status_is_authority": False,
                        **(
                            {
                                "evidence_kind": (
                                    BASELINE_VALIDATION_SET_EVIDENCE_KIND
                                    if task.task_id == BASELINE_TASK_ID
                                    else PRELAUNCH_VALIDATION_SET_EVIDENCE_KIND
                                )
                            }
                            if task.task_id in PRELAUNCH_COMPLETED_TASK_IDS
                            else {}
                        ),
                    }
                ],
                "validations": validation_rows,
                # The attempt-local Portal projection also understands this
                # singular compatibility field.  It remains a declaration,
                # never completion evidence.
                "validation": " ; ".join(task.validation),
            }
        )
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/board-population@1",
        "program_id": PROGRAM_ID,
        "repository_tree_id": source_tree,
        "source_head": source_head,
        "plan_root_cid": plan_root,
        "objectives": objective_rows,
        "goal_edges": goal_edges,
        "plans": [
            {
                "plan_cid": plan_root,
                "plan_alias": "APMC-PLAN-R1",
                "goal_cid": goal_cids[ROOT_OBJECTIVE],
                "status": "active",
                "program_id": PROGRAM_ID,
                **source_bindings,
            }
        ],
        "tasks": task_rows,
        "task_cids_by_alias": task_cids_by_alias,
        "goal_cids_by_alias": goal_cids,
    }


def _run_validation(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    """Run fixed argv with a hard combined-output and wall-time bound."""

    normalized = list(_validation_execution_argv(argv))
    process = subprocess.Popen(
        normalized,
        cwd=REPO_ROOT,
        env=_sealed_validation_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None:  # pragma: no cover
        raise MaterializationError("validation output pipes are unavailable")
    stdout_fd = process.stdout.fileno()
    stderr_fd = process.stderr.fileno()
    streams = {stdout_fd: bytearray(), stderr_fd: bytearray()}
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)
    deadline = time.monotonic() + 1_800

    def terminate() -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminate()
                raise subprocess.TimeoutExpired(
                    normalized,
                    1_800,
                    output=bytes(streams[stdout_fd]),
                    stderr=bytes(streams[stderr_fd]),
                )
            for key, _mask in selector.select(timeout=min(remaining, 1.0)):
                total = sum(len(item) for item in streams.values())
                chunk = os.read(
                    key.fd,
                    min(65_536, MAX_VALIDATION_OUTPUT_BYTES - total + 1),
                )
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                streams[key.fd].extend(chunk)
                if sum(len(item) for item in streams.values()) > MAX_VALIDATION_OUTPUT_BYTES:
                    terminate()
                    raise MaterializationError("APMC-000 validation output exceeds its bound")
        returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            terminate()
    return subprocess.CompletedProcess(
        args=normalized,
        returncode=returncode,
        stdout=bytes(streams[stdout_fd]),
        stderr=bytes(streams[stderr_fd]),
    )


def _validation_execution_argv(argv: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(item) for item in argv)
    if not normalized:
        raise MaterializationError("validation argv must not be empty")
    if normalized[0] != "python3":
        raise MaterializationError(
            f"validation executable {normalized[0]!r} is outside the closed bootstrap set"
        )
    if not Path(TRUSTED_PYTHON).is_file():
        raise MaterializationError("trusted system Python is unavailable")
    return (TRUSTED_PYTHON, *normalized[1:])


def _require_population_identity(
    population: Mapping[str, Any],
    identity_reader: SourceIdentityReader,
) -> None:
    head, tree = identity_reader()
    if head != str(population["source_head"]) or tree != str(population["repository_tree_id"]):
        raise MaterializationError(
            "current clean commit/tree changed during APMC-000 qualification"
        )


def _task_population_contract(
    population: Mapping[str, Any],
    task_alias: str,
) -> tuple[Mapping[str, Any], tuple[dict[str, Any], ...], str, str]:
    expected = next(
        (item for item in population["tasks"] if str(item["task_alias"]) == task_alias),
        None,
    )
    if not isinstance(expected, Mapping):
        raise MaterializationError(f"{task_alias} is absent from the sealed population")
    normalized_declarations: list[dict[str, Any]] = []
    for declaration in expected["validations"]:
        if not isinstance(declaration, Mapping):
            raise MaterializationError(f"{task_alias} validation declaration is malformed")
        argv = declaration.get("argv")
        if (
            not isinstance(argv, Sequence)
            or isinstance(argv, (str, bytes, bytearray, memoryview))
            or not argv
            or declaration.get("current_tree_required") is not True
            or declaration.get("shell_interpolation_permitted") is not False
        ):
            raise MaterializationError(
                f"{task_alias} validation must be nonempty current-tree shell-free argv"
            )
        normalized_declarations.append(
            {
                "argv": [str(item) for item in argv],
                "current_tree_required": True,
                "shell_interpolation_permitted": False,
            }
        )
    if not normalized_declarations:
        raise MaterializationError(f"{task_alias} declares no current-tree validation")
    population_cid = content_identity(population)
    declaration_set_cid = content_identity(
        {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/autonomy/"
                "current-tree-validation-declaration-set@1"
            ),
            "program_id": PROGRAM_ID,
            "task_alias": task_alias,
            "task_cid": expected["task_cid"],
            "validations": normalized_declarations,
        }
    )
    return (
        expected,
        tuple(normalized_declarations),
        population_cid,
        declaration_set_cid,
    )


def _expected_task_projection(expected: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the exact task relations persisted by DatabaseTaskSource."""

    outputs = [
        {
            "ordinal": ordinal,
            "path": str(item["path"]),
            "effect": dict(item),
        }
        for ordinal, item in enumerate(expected["outputs"])
    ]
    acceptance = [
        {
            "ordinal": ordinal,
            "criterion": str(item["criterion"]),
            "evidence_policy": dict(item),
        }
        for ordinal, item in enumerate(expected["acceptance"])
    ]
    validations = [
        {
            "ordinal": ordinal,
            "argv": list(item["argv"]),
            "policy": {
                key: value
                for key, value in item.items()
                if key not in {"argv", "validation_commands", "command"}
            },
        }
        for ordinal, item in enumerate(expected["validations"])
    ]
    body = {key: value for key, value in expected.items() if key not in _TASK_RELATION_FIELDS}
    return {
        "goal_cid": str(expected["goal_cid"]),
        "plan_cid": str(expected["plan_cid"]),
        "objective_id": str(expected["objective_id"]),
        "ordinal": int(expected["ordinal"]),
        "priority": str(expected["priority"]),
        "body": body,
        "outputs": outputs,
        "acceptance": acceptance,
        "validations": validations,
    }


def _collect_current_tree_qualification(
    source: DatabaseTaskSource,
    population: Mapping[str, Any],
    task_alias: str,
    *,
    completion_schema: str,
    validation_schema: str,
    validation_evidence_kind: str,
    validation_set_evidence_kind: str,
    validation_runner: ValidationRunner = _run_validation,
    source_identity_reader: SourceIdentityReader = _require_clean_committed_source,
) -> dict[str, Any]:
    """Collect one allowlisted task's exact current-tree validation evidence.

    Collection is deliberately pure with respect to task/evidence state.  The
    caller validates every closed-allowlist task first and only then persists
    the batch in dependency order.
    """

    (
        expected,
        normalized_declarations,
        population_cid,
        declaration_set_cid,
    ) = _task_population_contract(population, task_alias)
    task = source.get_task(task_alias)
    if (
        task is None
        or task.task_cid != expected["task_cid"]
        or task.status != "todo"
        or task.revision != 1
    ):
        raise MaterializationError(
            f"{task_alias} qualification requires the exact untouched revision-1 task"
        )
    acceptance = tuple(task.acceptance)
    policy = (
        acceptance[0].get("evidence_policy")
        if len(acceptance) == 1 and isinstance(acceptance[0], Mapping)
        else None
    )
    if (
        not isinstance(policy, Mapping)
        or policy.get("evidence_kind") != validation_set_evidence_kind
        or policy.get("current_tree_required") is not True
        or policy.get("declared_validation_required") is not True
        or policy.get("markdown_status_is_authority") is not False
    ):
        raise MaterializationError(
            f"{task_alias} acceptance does not require its dedicated validation set"
        )

    acceptance_declaration_cid = content_identity(
        {"task_cid": task.task_cid, "acceptance": expected["acceptance"]}
    )
    output_declaration_cid = content_identity(
        {"task_cid": task.task_cid, "outputs": expected["outputs"]}
    )

    _require_population_identity(population, source_identity_reader)
    evidence_digests: list[str] = []
    validation_entries: list[dict[str, Any]] = []
    for ordinal, declaration in enumerate(normalized_declarations, start=1):
        normalized_argv = tuple(declaration["argv"])
        execution_argv = _validation_execution_argv(normalized_argv)
        result = validation_runner(normalized_argv)
        _require_population_identity(population, source_identity_reader)
        if not isinstance(result.stdout, (bytes, bytearray)) or not isinstance(
            result.stderr, (bytes, bytearray)
        ):
            raise MaterializationError(f"{task_alias} validation output must be bytes")
        stdout = bytes(result.stdout)
        stderr = bytes(result.stderr)
        if len(stdout) + len(stderr) > MAX_VALIDATION_OUTPUT_BYTES:
            raise MaterializationError(
                f"{task_alias} validation {ordinal} output exceeds its bound"
            )
        entry: dict[str, Any] = {
            "schema": validation_schema,
            "program_id": PROGRAM_ID,
            "task_alias": task_alias,
            "task_cid": task.task_cid,
            "expected_task_revision": task.revision,
            "source_head": population["source_head"],
            "repository_tree_id": population["repository_tree_id"],
            "plan_root_cid": population["plan_root_cid"],
            "population_cid": population_cid,
            "validation_declaration_set_cid": declaration_set_cid,
            "acceptance_declaration_cid": acceptance_declaration_cid,
            "output_declaration_cid": output_declaration_cid,
            "ordinal": ordinal,
            "argv": list(normalized_argv),
            "execution_argv": list(execution_argv),
            "declaration_id": content_identity(declaration),
            "returncode": int(result.returncode),
            "stdout_sha256": _sha256_bytes(stdout),
            "stderr_sha256": _sha256_bytes(stderr),
        }
        entry["evidence_digest"] = content_identity(entry)
        if _json_size(entry) > MAX_BASELINE_VALIDATION_ENTRY_BYTES:
            raise MaterializationError(
                f"{task_alias} validation evidence {ordinal} exceeds its bound"
            )
        if result.returncode != 0:
            raise MaterializationError(
                f"{task_alias} current-tree validation {ordinal} failed "
                f"with return code {result.returncode}"
            )
        evidence_digests.append(str(entry["evidence_digest"]))
        validation_entries.append(entry)

    _require_population_identity(population, source_identity_reader)
    bundle: dict[str, Any] = {
        "schema": completion_schema,
        "program_id": PROGRAM_ID,
        "status": "validated",
        "task_alias": task_alias,
        "task_cid": task.task_cid,
        "expected_task_revision": task.revision,
        "completion_task_revision": task.revision + 1,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "population_cid": population_cid,
        "validation_declaration_set_cid": declaration_set_cid,
        "acceptance_declaration_cid": acceptance_declaration_cid,
        "output_declaration_cid": output_declaration_cid,
        "evidence_kind": validation_set_evidence_kind,
        "validation_evidence_kind": validation_evidence_kind,
        "validation_evidence_digests": evidence_digests,
        "validation_entries": validation_entries,
        "validation_count": len(validation_entries),
        "markdown_status_authoritative": False,
        "model_result_authoritative": False,
    }
    if task_alias == "APMC-018":
        bundle["benchmark_measurement_status"] = "not_run"
        bundle["promotion_eligible"] = False
    bundle["receipt_id"] = content_identity(bundle)
    if _json_size(bundle) > MAX_BASELINE_COMPLETION_RECEIPT_BYTES:
        raise MaterializationError(f"{task_alias} completion receipt exceeds its bound")
    return bundle


def _persist_current_tree_qualification(
    source: DatabaseTaskSource,
    population: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    source_identity_reader: SourceIdentityReader,
) -> Mapping[str, Any]:
    task_alias = str(bundle["task_alias"])
    task = source.get_task(task_alias)
    if (
        task is None
        or task.task_cid != bundle["task_cid"]
        or task.status != "todo"
        or task.revision != 1
    ):
        raise MaterializationError(
            f"{task_alias} changed before qualification evidence persistence"
        )
    for dependency_cid in task.dependencies:
        dependency = source.get_task(dependency_cid)
        if dependency is None or dependency.status != "completed":
            raise MaterializationError(f"{task_alias} qualification dependency is not completed")

    entries = tuple(bundle["validation_entries"])
    evidence_digests = tuple(str(item) for item in bundle["validation_evidence_digests"])
    for entry in entries:
        execution_argv = tuple(str(item) for item in entry["execution_argv"])
        source.record_validation_result(
            task_cid=task.task_cid,
            outcome="passed",
            evidence_digest=str(entry["evidence_digest"]),
            argv=execution_argv,
            body={
                "schema": entry["schema"],
                "source_head": entry["source_head"],
                "repository_tree_id": entry["repository_tree_id"],
                "plan_root_cid": entry["plan_root_cid"],
                "population_cid": entry["population_cid"],
                "validation_declaration_set_cid": entry["validation_declaration_set_cid"],
                "acceptance_declaration_cid": entry["acceptance_declaration_cid"],
                "output_declaration_cid": entry["output_declaration_cid"],
                "ordinal": entry["ordinal"],
                "stdout_sha256": entry["stdout_sha256"],
                "stderr_sha256": entry["stderr_sha256"],
            },
        )
        source.record_evidence(
            task_cid=task.task_cid,
            evidence_kind=str(bundle["validation_evidence_kind"]),
            digest=str(entry["evidence_digest"]),
            body=entry,
        )
    source.record_evidence(
        task_cid=task.task_cid,
        evidence_kind=str(bundle["evidence_kind"]),
        digest=str(bundle["receipt_id"]),
        body=bundle,
    )
    _require_population_identity(population, source_identity_reader)
    completed = source.compare_and_set_status(
        task.task_cid,
        expected_revision=task.revision,
        status="completed",
        receipt=bundle,
        evidence_digests=[str(bundle["receipt_id"]), *evidence_digests],
    )
    if (
        completed.changed is not True
        or completed.task.status != "completed"
        or completed.task.revision != 2
    ):
        raise MaterializationError(f"{task_alias} completion CAS was not exact")
    _require_population_identity(population, source_identity_reader)
    return bundle


def _qualify_prelaunch_current_tree(
    source: DatabaseTaskSource,
    population: Mapping[str, Any],
    *,
    validation_runner: ValidationRunner,
    source_identity_reader: SourceIdentityReader,
) -> tuple[Mapping[str, Any], ...]:
    """Qualify only the closed, already-implemented prelaunch task set."""

    collected: list[Mapping[str, Any]] = []
    for task_alias in PRELAUNCH_QUALIFIED_TASK_IDS:
        baseline = task_alias == BASELINE_TASK_ID
        collected.append(
            _collect_current_tree_qualification(
                source,
                population,
                task_alias,
                completion_schema=(
                    BASELINE_COMPLETION_RECEIPT_SCHEMA
                    if baseline
                    else PRELAUNCH_COMPLETION_RECEIPT_SCHEMA
                ),
                validation_schema=(
                    BASELINE_VALIDATION_EVIDENCE_SCHEMA
                    if baseline
                    else PRELAUNCH_VALIDATION_EVIDENCE_SCHEMA
                ),
                validation_evidence_kind=(
                    BASELINE_VALIDATION_EVIDENCE_KIND
                    if baseline
                    else PRELAUNCH_VALIDATION_EVIDENCE_KIND
                ),
                validation_set_evidence_kind=(
                    BASELINE_VALIDATION_SET_EVIDENCE_KIND
                    if baseline
                    else PRELAUNCH_VALIDATION_SET_EVIDENCE_KIND
                ),
                validation_runner=validation_runner,
                source_identity_reader=source_identity_reader,
            )
        )
    _require_population_identity(population, source_identity_reader)
    persisted = tuple(
        _persist_current_tree_qualification(
            source,
            population,
            bundle,
            source_identity_reader=source_identity_reader,
        )
        for bundle in collected
    )
    _require_population_identity(population, source_identity_reader)
    return persisted


def _verify_current_tree_evidence(
    source: DatabaseTaskSource,
    population: Mapping[str, Any],
    task: Any,
    authority: Mapping[str, Any],
    errors: list[str],
    *,
    completion_schema: str,
    validation_schema: str,
    validation_evidence_kind: str,
    validation_set_evidence_kind: str,
) -> Mapping[str, Any]:
    task_alias = str(task.task_alias)
    (
        expected,
        declarations,
        population_cid,
        declaration_set_cid,
    ) = _task_population_contract(population, task_alias)
    candidate = task.body.get("completion_receipt")
    receipt = dict(candidate) if isinstance(candidate, Mapping) else {}
    raw_entries = receipt.get("validation_entries")
    raw_digests = receipt.get("validation_evidence_digests")
    entries = (
        list(raw_entries)
        if isinstance(raw_entries, Sequence)
        and not isinstance(raw_entries, (str, bytes, bytearray, memoryview))
        else []
    )
    digests = (
        [str(item) for item in raw_digests]
        if isinstance(raw_digests, Sequence)
        and not isinstance(raw_digests, (str, bytes, bytearray, memoryview))
        else []
    )
    unsigned = dict(receipt)
    receipt_id = str(unsigned.pop("receipt_id", "") or "")
    expected_core = {
        "schema": completion_schema,
        "program_id": PROGRAM_ID,
        "status": "validated",
        "task_alias": task_alias,
        "task_cid": task.task_cid,
        "expected_task_revision": 1,
        "completion_task_revision": 2,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "population_cid": population_cid,
        "validation_declaration_set_cid": declaration_set_cid,
        "acceptance_declaration_cid": content_identity(
            {"task_cid": task.task_cid, "acceptance": expected["acceptance"]}
        ),
        "output_declaration_cid": content_identity(
            {"task_cid": task.task_cid, "outputs": expected["outputs"]}
        ),
        "evidence_kind": validation_set_evidence_kind,
        "validation_evidence_kind": validation_evidence_kind,
        "validation_count": len(declarations),
        "markdown_status_authoritative": False,
        "model_result_authoritative": False,
    }
    if task_alias == "APMC-018":
        expected_core["benchmark_measurement_status"] = "not_run"
        expected_core["promotion_eligible"] = False
    expected_receipt_keys = {
        *expected_core,
        "validation_evidence_digests",
        "validation_entries",
        "receipt_id",
    }
    if (
        task.task_cid != expected["task_cid"]
        or task.revision != 2
        or set(receipt) != expected_receipt_keys
        or _json_size(receipt) > MAX_BASELINE_COMPLETION_RECEIPT_BYTES
        or any(receipt.get(key) != value for key, value in expected_core.items())
        or receipt_id != content_identity(unsigned)
        or len(entries) != len(declarations)
        or len(digests) != len(declarations)
        or len(set(digests)) != len(digests)
    ):
        errors.append(f"{task_alias} completion evidence is absent, malformed, or stale")

    current = source.current_evidence_for_task(task.task_cid)
    if len(current) != (2 * len(declarations)) + 1:
        errors.append(f"{task_alias} current evidence population is not exact")
    aggregate_kind_nodes = [
        item for item in current if item.get("evidence_kind") == validation_set_evidence_kind
    ]
    expected_aggregate_evidence_id = content_identity(
        {
            "task_cid": task.task_cid,
            "evidence_kind": validation_set_evidence_kind,
            "digest": receipt_id,
            "body": receipt,
        }
    )
    aggregate_nodes = [
        item
        for item in aggregate_kind_nodes
        if item.get("digest") == receipt_id
        and dict(item.get("body") or {}) == receipt
        and item.get("evidence_id") == expected_aggregate_evidence_id
        and item.get("parent_evidence_id") == ""
        and item.get("task_cid") == task.task_cid
    ]
    if len(aggregate_kind_nodes) != 1 or len(aggregate_nodes) != 1:
        errors.append(f"{task_alias} validation-set evidence is not current and exact")

    individual_nodes = [
        item for item in current if item.get("evidence_kind") == validation_evidence_kind
    ]
    validation_nodes = [item for item in current if item.get("evidence_kind") == "validation"]
    if len(individual_nodes) != len(declarations) or len(validation_nodes) != len(declarations):
        errors.append(f"{task_alias} individual validation evidence is incomplete")
    authority_runs = tuple(authority.get("validation_runs") or ())
    authority_results = tuple(authority.get("validation_results") or ())
    authority_completions = tuple(authority.get("completion_receipts") or ())
    if len(authority_runs) != len(declarations) or len(authority_results) != len(declarations):
        errors.append(f"{task_alias} canonical validation authority is incomplete")

    for ordinal, declaration in enumerate(declarations, start=1):
        if ordinal > len(entries) or ordinal > len(digests):
            break
        entry = entries[ordinal - 1]
        if not isinstance(entry, Mapping):
            errors.append(f"{task_alias} validation evidence {ordinal} is malformed")
            continue
        entry_map = dict(entry)
        entry_digest = str(entry_map.pop("evidence_digest", "") or "")
        expected_entry = {
            "schema": validation_schema,
            "program_id": PROGRAM_ID,
            "task_alias": task_alias,
            "task_cid": task.task_cid,
            "expected_task_revision": 1,
            "source_head": population["source_head"],
            "repository_tree_id": population["repository_tree_id"],
            "plan_root_cid": population["plan_root_cid"],
            "population_cid": population_cid,
            "validation_declaration_set_cid": declaration_set_cid,
            "acceptance_declaration_cid": expected_core["acceptance_declaration_cid"],
            "output_declaration_cid": expected_core["output_declaration_cid"],
            "ordinal": ordinal,
            "argv": list(declaration["argv"]),
            "execution_argv": list(_validation_execution_argv(declaration["argv"])),
            "declaration_id": content_identity(declaration),
            "returncode": 0,
        }
        expected_entry_keys = {
            *expected_entry,
            "stdout_sha256",
            "stderr_sha256",
            "evidence_digest",
        }
        if (
            set(entry) != expected_entry_keys
            or _json_size(entry) > MAX_BASELINE_VALIDATION_ENTRY_BYTES
            or any(entry.get(key) != value for key, value in expected_entry.items())
            or entry_digest != digests[ordinal - 1]
            or entry_digest != content_identity(entry_map)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(entry.get("stdout_sha256") or "")) is None
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(entry.get("stderr_sha256") or "")) is None
        ):
            errors.append(f"{task_alias} validation evidence {ordinal} changed")
            continue
        exact_nodes = [
            item
            for item in individual_nodes
            if item.get("digest") == entry_digest
            and dict(item.get("body") or {}) == dict(entry)
            and item.get("evidence_id")
            == content_identity(
                {
                    "task_cid": task.task_cid,
                    "evidence_kind": validation_evidence_kind,
                    "digest": entry_digest,
                    "body": dict(entry),
                }
            )
            and item.get("parent_evidence_id") == ""
            and item.get("task_cid") == task.task_cid
        ]
        exact_validation_nodes = [
            item
            for item in validation_nodes
            if item.get("digest") == entry_digest
            and item.get("parent_evidence_id") == ""
            and item.get("task_cid") == task.task_cid
            and isinstance(item.get("body"), Mapping)
            and set(item["body"]) == {"run_id", "result_id", "argv", "outcome"}
            and list(item["body"].get("argv") or ()) == list(entry["execution_argv"])
            and item["body"].get("outcome") == "passed"
            and item["body"].get("result_id")
            == content_identity(
                {
                    "run_id": item["body"].get("run_id"),
                    "outcome": "passed",
                    "evidence_digest": entry_digest,
                }
            )
            and item.get("evidence_id")
            == content_identity(
                {
                    "task_cid": task.task_cid,
                    "evidence_kind": "validation",
                    "digest": entry_digest,
                    "run_id": item["body"].get("run_id"),
                }
            )
        ]
        if len(exact_nodes) != 1 or len(exact_validation_nodes) != 1:
            errors.append(f"{task_alias} validation evidence {ordinal} is not current")
            continue
        validation_node_body = dict(exact_validation_nodes[0]["body"])
        run_id = str(validation_node_body["run_id"])
        result_id = str(validation_node_body["result_id"])
        execution_argv = list(entry["execution_argv"])
        authority_body = {
            "schema": entry["schema"],
            "source_head": entry["source_head"],
            "repository_tree_id": entry["repository_tree_id"],
            "plan_root_cid": entry["plan_root_cid"],
            "population_cid": entry["population_cid"],
            "validation_declaration_set_cid": entry["validation_declaration_set_cid"],
            "acceptance_declaration_cid": entry["acceptance_declaration_cid"],
            "output_declaration_cid": entry["output_declaration_cid"],
            "ordinal": entry["ordinal"],
            "stdout_sha256": entry["stdout_sha256"],
            "stderr_sha256": entry["stderr_sha256"],
        }
        exact_runs = []
        for row in authority_runs:
            row_map = dict(row)
            started_at = str(row_map.get("started_at") or "")
            if (
                row_map.get("run_id") == run_id
                and row_map.get("task_cid") == task.task_cid
                and row_map.get("attempt_id") == ""
                and started_at
                and row_map.get("finished_at") == started_at
                and row_map.get("status") == "passed"
                and row_map.get("command_digest") == content_identity({"argv": execution_argv})
                and dict(row_map.get("body") or {}) == {"argv": execution_argv, **authority_body}
                and run_id
                == content_identity(
                    {
                        "task_cid": task.task_cid,
                        "attempt_id": "",
                        "argv": execution_argv,
                        "recorded_at": started_at,
                    }
                )
            ):
                exact_runs.append(row)
        exact_results = [
            row
            for row in authority_results
            if row.get("result_id") == result_id
            and row.get("run_id") == run_id
            and row.get("task_cid") == task.task_cid
            and row.get("ordinal") == 0
            and row.get("outcome") == "passed"
            and row.get("evidence_digest") == entry_digest
            and dict(row.get("body") or {}) == authority_body
            and result_id
            == content_identity(
                {
                    "run_id": run_id,
                    "outcome": "passed",
                    "evidence_digest": entry_digest,
                }
            )
        ]
        if len(exact_runs) != 1 or len(exact_results) != 1:
            errors.append(f"{task_alias} validation authority {ordinal} is not exact")

    completion_evidence_digests = [receipt_id, *digests]
    completion_evidence_digest = content_identity(
        {
            "task_cid": task.task_cid,
            "revision": 2,
            "receipt": receipt,
            "evidence_digests": completion_evidence_digests,
        }
    )
    completion_receipt_cid = content_identity(
        {
            "namespace": "completion-receipt",
            "task_cid": task.task_cid,
            "revision": 2,
            "evidence_digest": completion_evidence_digest,
        }
    )
    exact_completions = [
        row
        for row in authority_completions
        if row.get("receipt_cid") == completion_receipt_cid
        and row.get("task_cid") == task.task_cid
        and row.get("goal_cid") == task.goal_cid
        and row.get("attempt_id") == ""
        and row.get("claim_cid") == ""
        and row.get("fencing_token") == 0
        and bool(row.get("completed_at"))
        and row.get("validation_run_id") == ""
        and row.get("evidence_digest") == completion_evidence_digest
        and dict(row.get("body") or {})
        == {
            "schema": COMPLETION_EVIDENCE_SCHEMA,
            "receipt": receipt,
            "evidence_digests": completion_evidence_digests,
            "revision": 2,
        }
    ]
    if len(authority_completions) != 1 or len(exact_completions) != 1:
        errors.append(f"{task_alias} canonical completion authority is not exact")
    return receipt


def _verify_source(source: DatabaseTaskSource, population: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = source.snapshot()
    page = source.list_tasks(limit=100)
    ready = source.ready_tasks(limit=100)
    tasks = tuple(page.tasks)
    aliases = tuple(item.task_alias for item in tasks)
    expected_tasks = tuple(population["tasks"])
    by_alias = {item.task_alias: item for item in tasks}
    qualification_authorities: dict[str, Mapping[str, Any]] = {}
    observed_goal_edges = tuple(
        (
            str(item["parent_goal_cid"]),
            str(item["child_goal_cid"]),
            str(item["edge_kind"]),
        )
        for item in source.list_goal_edges(limit=MAX_QUERY_LIMIT)
    )
    expected_goal_edges = tuple(
        sorted(
            (
                str(item["parent_goal_cid"]),
                str(item["child_goal_cid"]),
                str(item["edge_kind"]),
            )
            for item in population["goal_edges"]
        )
    )
    errors: list[str] = []
    if aliases != EXPECTED_TASK_IDS:
        errors.append("task aliases/order changed")
    expected_objective_count = sum(
        bool(str(item.get("objective_id") or "")) for item in population["objectives"]
    )
    if (
        snapshot.task_count != len(EXPECTED_TASK_IDS)
        or snapshot.goal_count != len(EXPECTED_GOAL_IDS)
        or snapshot.objective_count != expected_objective_count
        or snapshot.plan_count != len(population["plans"])
    ):
        errors.append("objective/goal/plan/task population count changed")
    if snapshot.plan_root_cid != population["plan_root_cid"]:
        errors.append("plan root changed")
    if snapshot.repository_tree_id != population["repository_tree_id"]:
        errors.append("repository tree changed")
    if observed_goal_edges != expected_goal_edges:
        errors.append("goal parent/dependency edges changed")
    for expected_goal in population["objectives"]:
        observed_goal = source.get_goal(str(expected_goal["goal_cid"]))
        expected_goal_body = {
            key: value
            for key, value in expected_goal.items()
            if key
            not in {
                "goal_cid",
                "goal_id",
                "goal_alias",
                "title",
                "status",
                "ordinal",
                "objective_id",
            }
        }
        expected_goal_projection = {
            "goal_cid": str(expected_goal["goal_cid"]),
            "goal_alias": str(expected_goal["goal_alias"]),
            "objective_id": str(expected_goal["objective_id"]),
            "parent_goal_cid": str(expected_goal["parent_goal_cid"]),
            "ordinal": int(expected_goal["ordinal"]),
            "title": str(expected_goal["title"]),
            "status": str(expected_goal["status"]),
            "revision": 1,
            "body": expected_goal_body,
        }
        if observed_goal is None or any(
            observed_goal.get(key) != value for key, value in expected_goal_projection.items()
        ):
            errors.append(f"{expected_goal['goal_id']} goal projection changed")
        objective_id = str(expected_goal["objective_id"])
        if objective_id:
            observed_objective = source.get_objective(objective_id)
            expected_objective_body = {
                key: value
                for key, value in expected_goal.items()
                if key
                not in {
                    "objective_id",
                    "objective_alias",
                    "title",
                    "status",
                    "priority",
                }
            }
            expected_objective_projection = {
                "objective_id": objective_id,
                "objective_alias": str(expected_goal["objective_alias"]),
                "parent_objective_id": "",
                "title": str(expected_goal["title"]),
                "status": str(expected_goal["status"]),
                "priority": str(expected_goal["priority"]),
                "revision": 1,
                "body": expected_objective_body,
            }
            if observed_objective is None or any(
                observed_objective.get(key) != value
                for key, value in expected_objective_projection.items()
            ):
                errors.append(f"{objective_id} objective projection changed")
    for expected_plan in population["plans"]:
        observed_plan = source.get_plan(str(expected_plan["plan_cid"]))
        expected_plan_projection = {
            "plan_cid": str(expected_plan["plan_cid"]),
            "goal_cid": str(expected_plan["goal_cid"]),
            "plan_alias": str(expected_plan["plan_alias"]),
            "status": str(expected_plan["status"]),
            "revision": 1,
            "body": dict(expected_plan),
        }
        if observed_plan is None or any(
            observed_plan.get(key) != value for key, value in expected_plan_projection.items()
        ):
            errors.append(f"{expected_plan['plan_alias']} plan projection changed")
    baseline = by_alias.get(BASELINE_TASK_ID)
    baseline_qualified = baseline is not None and baseline.status == "completed"
    completed_aliases = tuple(item.task_alias for item in tasks if item.status == "completed")
    prelaunch_qualified = completed_aliases == PRELAUNCH_COMPLETED_TASK_IDS
    if not prelaunch_qualified:
        errors.append("closed prelaunch task set is not evidence-qualified for production runtime")
    for expected in expected_tasks:
        observed = by_alias.get(str(expected["task_alias"]))
        if observed is None:
            continue
        expected_projection = _expected_task_projection(expected)
        authority = source.qualification_authority_for_task(observed.task_cid)
        qualification_authorities[observed.task_alias] = authority
        expected_dependencies = tuple(
            sorted(
                population["task_cids_by_alias"].get(item, item)
                for item in expected["dependencies"]
            )
        )
        if (
            observed.task_cid != expected["task_cid"]
            or tuple(observed.dependencies) != expected_dependencies
        ):
            errors.append(f"{observed.task_alias} identity/dependencies changed")
        if (
            observed.goal_cid != expected_projection["goal_cid"]
            or observed.plan_cid != expected_projection["plan_cid"]
            or observed.objective_id != expected_projection["objective_id"]
            or observed.ordinal != expected_projection["ordinal"]
            or observed.priority != expected_projection["priority"]
        ):
            errors.append(f"{observed.task_alias} task projection fields changed")
        observed_body = dict(observed.body)
        if observed.task_alias in PRELAUNCH_COMPLETED_TASK_IDS:
            observed_body.pop("completion_receipt", None)
        if observed_body != expected_projection["body"]:
            errors.append(f"{observed.task_alias} task body changed")
        expected_identity = {
            "task_cid": str(expected["task_cid"]),
            "task_alias": str(expected["task_alias"]),
            "repository_tree_id": str(population["repository_tree_id"]),
        }
        if (
            dict(authority.get("identity") or {}) != expected_identity
            or authority.get("extension_schema") != ""
            or dict(authority.get("extension") or {}) != {}
        ):
            errors.append(f"{observed.task_alias} identity/extension authority changed")
        if [dict(item) for item in observed.outputs] != expected_projection["outputs"]:
            errors.append(f"{observed.task_alias} output declarations changed")
        if [dict(item) for item in observed.acceptance] != expected_projection["acceptance"]:
            errors.append(f"{observed.task_alias} acceptance declarations changed")
        if [dict(item) for item in observed.validations] != expected_projection["validations"]:
            errors.append(f"{observed.task_alias} validation declarations changed")
        if (
            observed.body.get("review_only") is not expected["review_only"]
            or str(observed.body.get("completion") or "") != expected["completion"]
        ):
            errors.append(f"{observed.task_alias} completion/review gate changed")
        expected_auto_claim_forbidden = (
            expected["review_only"] is True or str(expected["completion"]).casefold() == "manual"
        )
        if (
            DatabaseImplementationDaemon._automatic_claim_forbidden(observed)  # noqa: SLF001
            is not expected_auto_claim_forbidden
        ):
            errors.append(f"{observed.task_alias} automatic-claim gate changed")
        expected_status = (
            "completed" if observed.task_alias in PRELAUNCH_COMPLETED_TASK_IDS else "todo"
        )
        if observed.status != expected_status:
            errors.append(f"{observed.task_alias} status changed")
        expected_revision = 2 if observed.task_alias in PRELAUNCH_COMPLETED_TASK_IDS else 1
        if observed.revision != expected_revision:
            errors.append(f"{observed.task_alias} revision changed")
        if observed.task_alias not in PRELAUNCH_COMPLETED_TASK_IDS and any(
            authority.get(key)
            for key in ("validation_runs", "validation_results", "completion_receipts")
        ):
            errors.append(f"{observed.task_alias} has undeclared qualification authority rows")
    ready_aliases = tuple(item.task_alias for item in ready.tasks)
    expected_ready_aliases = PRELAUNCH_READY_TASK_IDS
    if ready_aliases != expected_ready_aliases:
        errors.append(f"initial ready set changed: {ready_aliases!r}")
    completion_receipts: dict[str, Mapping[str, Any]] = {}
    for task_alias in PRELAUNCH_COMPLETED_TASK_IDS:
        observed = by_alias.get(task_alias)
        if observed is None or observed.status != "completed":
            continue
        is_baseline = task_alias == BASELINE_TASK_ID
        completion_receipts[task_alias] = _verify_current_tree_evidence(
            source,
            population,
            observed,
            qualification_authorities.get(task_alias, {}),
            errors,
            completion_schema=(
                BASELINE_COMPLETION_RECEIPT_SCHEMA
                if is_baseline
                else PRELAUNCH_COMPLETION_RECEIPT_SCHEMA
            ),
            validation_schema=(
                BASELINE_VALIDATION_EVIDENCE_SCHEMA
                if is_baseline
                else PRELAUNCH_VALIDATION_EVIDENCE_SCHEMA
            ),
            validation_evidence_kind=(
                BASELINE_VALIDATION_EVIDENCE_KIND
                if is_baseline
                else PRELAUNCH_VALIDATION_EVIDENCE_KIND
            ),
            validation_set_evidence_kind=(
                BASELINE_VALIDATION_SET_EVIDENCE_KIND
                if is_baseline
                else PRELAUNCH_VALIDATION_SET_EVIDENCE_KIND
            ),
        )
    review_only_aliases = tuple(
        str(item["task_alias"]) for item in expected_tasks if item["review_only"] is True
    )
    manual_completion_aliases = tuple(
        str(item["task_alias"])
        for item in expected_tasks
        if str(item["completion"]).casefold() == "manual"
    )
    if review_only_aliases != EXPECTED_MANUAL_REVIEW_TASK_IDS:
        errors.append(f"review-only task set changed: {review_only_aliases!r}")
    if manual_completion_aliases != EXPECTED_MANUAL_REVIEW_TASK_IDS:
        errors.append(f"manual-completion task set changed: {manual_completion_aliases!r}")
    if errors:
        raise MaterializationError("; ".join(errors))
    receipt: dict[str, Any] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/board-materialization-receipt@1",
        "program_id": PROGRAM_ID,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "projection_cid": snapshot.projection_cid,
        "source_identity": snapshot.source_identity,
        "task_count": snapshot.task_count,
        "goal_count": snapshot.goal_count,
        "goal_edge_count": len(observed_goal_edges),
        "goal_edges_cid": content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/goal-edge-set@1",
                "edges": observed_goal_edges,
            }
        ),
        "dependency_count": snapshot.dependency_count,
        "ready_task_aliases": list(ready_aliases),
        "baseline_qualified": baseline_qualified,
        "prelaunch_qualified": prelaunch_qualified,
        "prelaunch_completed_task_aliases": list(completed_aliases),
        "completion_receipt_ids_by_alias": {
            task_alias: str(receipt.get("receipt_id") or "")
            for task_alias, receipt in sorted(completion_receipts.items())
        },
        "baseline_completion_receipt_id": str(
            completion_receipts.get(BASELINE_TASK_ID, {}).get("receipt_id") or ""
        ),
        "task_cids_by_alias": dict(population["task_cids_by_alias"]),
        "review_only_task_aliases": list(review_only_aliases),
        "manual_completion_task_aliases": list(manual_completion_aliases),
        "duckdb_authoritative": True,
        "quack_required_after_materialization": True,
        "ducklake_authoritative": False,
        "ducklake_projection_status": "disabled_activation_held",
        "markdown_status_authoritative": False,
    }
    receipt["receipt_id"] = content_identity(receipt)
    return receipt


def materialize(
    database: Path,
    population: Mapping[str, Any],
    *,
    validation_runner: ValidationRunner = _run_validation,
    source_identity_reader: SourceIdentityReader = _require_clean_committed_source,
) -> dict[str, Any]:
    if database.exists():
        raise MaterializationError(
            "database already exists; refuse a second direct owner "
            "(use --verify-only --endpoint after Quack starts)"
        )
    database.parent.mkdir(parents=True, exist_ok=True)
    source = DatabaseTaskSource(
        database,
        owner_id="apmc-materializer:exclusive-bootstrap",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        # These receipts are immutable and exact-tree bound, so age alone is
        # not an invalidation signal.  Source identity is checked before,
        # throughout, and after qualification instead.
        evidence_freshness_seconds=0,
        install_schema=True,
    )
    try:
        source.materialize(
            population,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
        _qualify_prelaunch_current_tree(
            source,
            population,
            validation_runner=validation_runner,
            source_identity_reader=source_identity_reader,
        )
        receipt = _verify_source(source, population)
        _require_population_identity(population, source_identity_reader)
        return receipt
    finally:
        source.close()


def verify(endpoint_or_database: str, population: Mapping[str, Any]) -> dict[str, Any]:
    if not is_quack_transport_target(endpoint_or_database):
        raise MaterializationError("post-bootstrap verification requires a loopback Quack endpoint")
    source = DatabaseTaskSource(
        endpoint_or_database,
        owner_id="apmc-materializer:read-only-verifier",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        evidence_freshness_seconds=0,
        install_schema=False,
    )
    try:
        _require_population_identity(population, _require_clean_committed_source)
        receipt = _verify_source(source, population)
        _require_population_identity(population, _require_clean_committed_source)
        return receipt
    finally:
        source.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, help="New repository-local DuckDB target.")
    parser.add_argument(
        "--endpoint",
        default="",
        help="Loopback quack: endpoint for verification after the owner starts.",
    )
    parser.add_argument("--verify-only", action="store_true", help="Perform no population writes.")
    parser.add_argument(
        "--receipt",
        default="",
        help="Optional repository-local runtime receipt path under data/ or state/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        head, tree = _require_clean_committed_source()
        population = build_population(source_head=head, source_tree=tree)
        database = _safe_new_database(args.database)
        if args.verify_only:
            if not args.endpoint:
                raise MaterializationError(
                    "--verify-only requires --endpoint; direct DuckDB verification "
                    "would create a second owner"
                )
            receipt = verify(str(args.endpoint), population)
        else:
            if args.endpoint:
                raise MaterializationError("--endpoint is valid only with --verify-only")
            receipt = materialize(
                database,
                population,
            )
        _require_population_identity(population, _require_clean_committed_source)
        if args.receipt:
            _write_runtime_receipt(args.receipt, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        report = {
            "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/board-materialization-error@1",
            "program_id": PROGRAM_ID,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
