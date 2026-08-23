#!/usr/bin/env python3
"""Bootstrap and operate the sealed CASF campaign through DuckDB and Quack.

This is deliberately a first-tranche bootstrap operator, not the completed
federation.  It admits one bounded coordinator and one registered logical
subagent.  The coordinator qualifies its exact server-owned event-wait path,
but this operator does not claim federation-wide event-driven execution,
multi-supervisor operation, parallel task execution, or high concurrency.

The authority boundary is:

* an exclusive offline ``DatabaseTaskSource`` materializes the committed
  Markdown inputs into the canonical DuckDB control plane;
* while execution is live, one loopback ``QuackStateServer`` exclusively owns
  that file and all supervisor access uses the Quack transport;
* DuckLake is recorded as a typed, non-authoritative unavailable projection
  until CASF-031/032 implement and qualify it.

Raw state-owner tokens are never accepted on the command line, exported in an
environment, or written to logs or receipts.  The owner mints PID-bound grants
and passes them to the coordinator over a private inherited pipe.  The child
has no database path or arbitrary SQL surface, and task/provider execution is
not admitted by this first-tranche runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG: Final = Path("config/agent_supervisor_causal_event_federation_scheduler.json")
PROGRAM_ID: Final = "agent-supervisor-causal-event-federation-v1"
BOARD_NAMESPACE: Final = PROGRAM_ID
ROOT_GOAL: Final = "CASF-G000"
TASK_PREFIX: Final = "CASF-"
OPERATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation-bootstrap-operator@1"
)
POPULATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation-bootstrap-population@1"
)
BOOTSTRAP_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation-bootstrap-receipt@1"
)
LAUNCH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation-launch-receipt@1"
)
STATUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation-status-receipt@1"
)
COORDINATOR_READY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "causal-event-federation-coordinator-ready-receipt@1"
)
STOP_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/causal-event-federation-stop-receipt@1"
DUCKLAKE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation-ducklake-state@1"
)
GOAL_RE: Final = re.compile(r"^## (CASF-G\d{3}) (.+)$", re.MULTILINE)
QUACK_ENDPOINT_RE: Final = re.compile(
    r"^quack:(?://)?(127\.0\.0\.1|localhost):(\d{1,5})$", re.IGNORECASE
)
TOKEN_RE: Final = re.compile(r"[A-Za-z0-9_-]{8,}")
COMPLETED_STATUSES: Final = frozenset({"complete", "completed", "done", "skipped"})
ACTIVE_STATUSES: Final = frozenset({"claimed", "in_progress", "running"})
READY_STATUSES: Final = frozenset(
    {"admitted", "pending", "proposed", "queued", "ready", "retrying", "todo"}
)
TERMINAL_STATUSES: Final = frozenset(
    {*COMPLETED_STATUSES, "cancelled", "failed", "quarantined", "rejected"}
)
STATE_TOKEN_ENV: Final = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
STATE_OWNER_SOCKET_ENV: Final = "IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET"
SUPERVISOR_HEALTH_STALE_SECONDS: Final = 45.0
UNIX_SOCKET_PATH_CEILING: Final = 100


class OperatorError(RuntimeError):
    """Fail-closed CASF bootstrap operator error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path: Path, value: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorError(f"unreadable JSON authority artifact: {path}") from exc
    if not isinstance(value, dict):
        raise OperatorError(f"JSON authority artifact is not an object: {path}")
    return value


def _safe_path(root: Path, value: Any, *, field: str) -> Path:
    text = str(value or "").strip()
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise OperatorError(f"{field} must be a safe repository-relative path")
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise OperatorError(f"{field} escapes the repository") from exc
    return resolved


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr or result.stdout
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise OperatorError(f"git {' '.join(arguments)} failed: {str(detail).strip()}")
    return result.stdout


def _assert_clean_current_tree(config: Mapping[str, Any]) -> tuple[str, str]:
    if str(_git("status", "--porcelain=v1", "--untracked-files=all")).strip():
        raise OperatorError("launch/materialization requires a clean committed current tree")
    head = str(_git("rev-parse", "HEAD")).strip()
    tree = str(_git("rev-parse", "HEAD^{tree}")).strip()
    branch = str(_git("branch", "--show-current")).strip()
    required_branch = str(config.get("merge_target_branch") or "").strip()
    if branch != required_branch:
        raise OperatorError(
            f"current branch {branch!r} differs from sealed branch {required_branch!r}"
        )
    binding = config.get("source_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    ancestor = str(binding.get("accelerator_required_ancestor") or "").strip()
    if ancestor:
        probe = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if probe.returncode != 0:
            raise OperatorError("sealed base revision is not an ancestor of HEAD")
    return head, tree


def _tracked_bytes(path: Path, *, head: str) -> bytes:
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise OperatorError(f"sealed input escapes repository: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise OperatorError(f"sealed input is not a regular file: {relative}")
    current = path.read_bytes()
    committed = _git("show", f"{head}:{relative}", binary=True)
    if not isinstance(committed, bytes) or committed != current:
        raise OperatorError(f"sealed input differs from HEAD: {relative}")
    return current


def _require_canonical_schema_revision(schema_revision: Any) -> int:
    """Require the scheduler pin to match the packaged physical schema head."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        load_control_plane_catalog,
    )

    try:
        configured_revision = int(str(schema_revision or "").strip())
    except ValueError as exc:
        raise OperatorError(
            "scheduler schema revision must identify the canonical migration head"
        ) from exc
    latest_revision = load_control_plane_catalog().latest_version
    if configured_revision != latest_revision:
        raise OperatorError(
            "scheduler schema revision differs from the canonical migration head "
            f"(configured={configured_revision}, latest={latest_revision})"
        )
    return configured_revision


def _load_config(config_path: Path) -> tuple[Any, dict[str, Any]]:
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        load_configured_board,
    )

    board = load_configured_board(config_path, repo_root=ROOT)
    payload = dict(board.payload)
    if payload.get("program_identifier") != PROGRAM_ID:
        raise OperatorError("scheduler program identifier is not CASF v1")
    if board.board_namespace != BOARD_NAMESPACE:
        raise OperatorError("scheduler namespace is not CASF v1")
    normalized_prefix = (
        board.task_prefix[3:]
        if board.task_prefix.startswith("## ")
        else board.task_prefix
    )
    if normalized_prefix != TASK_PREFIX:
        raise OperatorError("scheduler task prefix is not CASF-")
    if payload.get("initial_projection", {}).get("root_goal_id") != ROOT_GOAL:
        raise OperatorError("scheduler root goal is not CASF-G000")
    if board.max_lanes != 1 or not board.strict_task_sharding:
        raise OperatorError("bootstrap operator permits exactly one strict lane")
    if board.idle_lane_work_stealing:
        raise OperatorError("bootstrap operator prohibits work stealing")
    capacity = payload.get("bootstrap_capacity")
    if not isinstance(capacity, Mapping) or any(
        int(capacity.get(name) or 0) != 1
        for name in (
            "supervisors",
            "registered_logical_subagents",
            "maximum_active_subagents",
            "lanes",
            "provider_concurrency",
        )
    ):
        raise OperatorError("bootstrap capacity must remain one supervisor/agent/lane")
    from ipfs_accelerate_py.agent_supervisor.federation.bootstrap_runtime import (
        validate_bootstrap_profile,
    )

    try:
        validate_bootstrap_profile(payload.get("federation_bootstrap_policy"))
    except (TypeError, ValueError) as exc:
        raise OperatorError("sealed federation bootstrap policy is invalid") from exc
    high = payload.get("high_concurrency_gate")
    if not isinstance(high, Mapping) or high.get("enabled_at_bootstrap") is not False:
        raise OperatorError("high-concurrency gate must remain closed")
    wait = payload.get("event_wait_policy")
    if (
        not isinstance(wait, Mapping)
        or wait.get("event_driven_claim_permitted_at_bootstrap") is not False
    ):
        raise OperatorError("bootstrap may not claim event-driven qualification")
    program = board.resolved_database_program()
    _require_canonical_schema_revision(program.schema_revision)
    if program.authority_mode != "quack" or program.task_source_kind != "duckdb":
        raise OperatorError("CASF requires DuckDB authority through Quack")
    if program.failover_policy != "fail_closed":
        raise OperatorError("Quack authority must fail closed")
    endpoint = QUACK_ENDPOINT_RE.fullmatch(program.quack_endpoint)
    if endpoint is None or not 1 <= int(endpoint.group(2)) <= 65535:
        raise OperatorError("Quack endpoint must be a bounded loopback URI")
    control = payload.get("operational_control_plane")
    if not isinstance(control, Mapping) or any(
        control.get(field) is not False
        for field in (
            "direct_multi_process_duckdb_file_open_permitted",
            "automatic_file_fallback_permitted",
            "arbitrary_sql_from_agents_permitted",
        )
    ):
        raise OperatorError("scheduler weakens the exclusive state-owner boundary")
    ducklake = payload.get("ducklake_projection_program")
    if not isinstance(ducklake, Mapping) or any(
        ducklake.get(field) is not False
        for field in (
            "authority",
            "scheduling_prerequisite",
            "lease_prerequisite",
            "policy_prerequisite",
            "acceptance_prerequisite",
            "completion_prerequisite",
            "may_grant_authority",
        )
    ):
        raise OperatorError("DuckLake must remain a non-authoritative projection")
    return board, payload


def _split_csv(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _goal_blocks(text: str) -> list[tuple[str, str, dict[str, str]]]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
        normalize_metadata_key,
    )

    matches = list(GOAL_RE.finditer(text))
    result: list[tuple[str, str, dict[str, str]]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields: dict[str, str] = {}
        for line in text[match.end() : end].splitlines():
            stripped = line.strip()
            if not stripped.startswith("- ") or ":" not in stripped:
                continue
            key, value = stripped[2:].split(":", 1)
            normalized = normalize_metadata_key(key)
            if normalized in fields:
                raise OperatorError(f"{match.group(1)} duplicates metadata field {normalized!r}")
            fields[normalized] = value.strip()
        result.append((match.group(1), match.group(2).strip(), fields))
    return result


def _source_forest(*, head: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_head": head,
        "repository": "ipfs_accelerate_py",
        "nested_repositories": [],
        "cross_repository_writes": False,
    }
    result["source_forest_root"] = _identity(result)
    return result


def _population(board: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Compile sealed Markdown with the production task parser."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
        parse_todo_blocks,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
        split_validation_commands,
    )

    head, tree = _assert_clean_current_tree(config)
    sources = {
        "config": _tracked_bytes(board.config_path, head=head),
        "taskboard": _tracked_bytes(board.path(board.taskboard_path), head=head),
        "objectives": _tracked_bytes(board.path(board.objectives_path), head=head),
        "plan": _tracked_bytes(board.path(board.plan_path), head=head),
        "validator": _tracked_bytes(board.path(board.validator_path), head=head),
        "operator": _tracked_bytes(Path(__file__).resolve(), head=head),
    }
    plan_root = content_identity(
        {
            "schema": "casf-plan-root@1",
            "program_id": PROGRAM_ID,
            "source_head": head,
            "repository_tree_id": tree,
            "sources": {name: _identity(body) for name, body in sorted(sources.items())},
        }
    )
    parsed_goals = _goal_blocks(sources["objectives"].decode("utf-8"))
    if not parsed_goals or parsed_goals[0][0] != ROOT_GOAL:
        raise OperatorError("objectives must begin with CASF-G000")
    if len(parsed_goals) != 17 or len({item[0] for item in parsed_goals}) != 17:
        raise OperatorError("objective hierarchy must contain exactly 17 unique goals")
    goal_cids = {
        goal_id: content_identity(
            {
                "goal_id": goal_id,
                "title": title,
                "metadata": fields,
                "plan_root_cid": plan_root,
            }
        )
        for goal_id, title, fields in parsed_goals
    }
    goals: list[dict[str, Any]] = []
    goal_edges: list[dict[str, Any]] = []
    observed_goals: set[str] = set()
    for ordinal, (goal_id, title, fields) in enumerate(parsed_goals, start=1):
        parent = str(fields.get("parent") or "").strip()
        if parent and parent not in observed_goals:
            raise OperatorError(f"{goal_id} parent must precede it: {parent}")
        dependencies = _split_csv(fields.get("depends_on"))
        if any(item not in goal_cids for item in dependencies):
            raise OperatorError(f"{goal_id} has an unknown goal dependency")
        goal: dict[str, Any] = {
            "goal_cid": goal_cids[goal_id],
            "goal_id": goal_id,
            "goal_alias": goal_id,
            "title": title,
            "ordinal": ordinal,
            "status": str(fields.get("status") or "open").lower(),
            "objective_id": "objective:casf-root" if goal_id == ROOT_GOAL else "",
            "objective_alias": ROOT_GOAL,
            "priority": str(fields.get("priority") or "P0"),
            "body": dict(fields),
        }
        if parent:
            goal["parent_goal_cid"] = goal_cids[parent]
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[parent],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_parent",
                }
            )
        for dependency in dependencies:
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[dependency],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_dependency",
                }
            )
        goals.append(goal)
        observed_goals.add(goal_id)

    parsed_tasks = parse_todo_blocks(
        sources["taskboard"].decode("utf-8"), task_header_prefix="## CASF-"
    )
    task_ids = [item[0] for item in parsed_tasks]
    expected_ids = [f"CASF-{index:03d}" for index in range(44)]
    if task_ids != expected_ids:
        raise OperatorError("task population must be exactly CASF-000..CASF-043")
    task_cids = {
        task_id: content_identity(
            {
                "task_id": task_id,
                "title": title,
                "source_line": source_line,
                "metadata": fields,
                "plan_root_cid": plan_root,
                "repository_tree_id": tree,
            }
        )
        for task_id, title, source_line, fields in parsed_tasks
    }
    tasks: list[dict[str, Any]] = []
    observed_tasks: set[str] = set()
    for ordinal, (task_id, title, source_line, fields) in enumerate(parsed_tasks, start=1):
        dependencies = _split_csv(fields.get("depends_on"))
        if any(item not in task_cids for item in dependencies):
            raise OperatorError(f"{task_id} has an unknown dependency")
        if any(item not in observed_tasks for item in dependencies):
            raise OperatorError(f"{task_id} dependency does not precede it")
        goal_id = str(fields.get("subgoal_id") or fields.get("goal_id") or ROOT_GOAL).strip()
        if goal_id not in goal_cids:
            raise OperatorError(f"{task_id} refers to unknown goal {goal_id}")
        output_paths = _split_csv(fields.get("outputs") or fields.get("predicted_files"))
        task = dict(fields)
        task.update(
            {
                "task_cid": task_cids[task_id],
                "task_id": task_id,
                "task_alias": task_id,
                "title": title,
                "source_line": source_line,
                "goal_cid": goal_cids[goal_id],
                "goal_id": goal_id,
                "plan_cid": plan_root,
                "objective_id": "objective:casf-root",
                "ordinal": ordinal,
                "status": str(fields.get("status") or "todo").lower(),
                "priority": str(fields.get("priority") or "P1"),
                "dependencies": [task_cids[item] for item in dependencies],
                "depends_on": [task_cids[item] for item in dependencies],
                "outputs": [
                    {
                        "path": path,
                        "effect_id": content_identity(
                            {"task_cid": task_cids[task_id], "path": path}
                        ),
                    }
                    for path in output_paths
                ],
                "acceptance": [str(fields.get("acceptance_subset") or "")],
                "validations": list(split_validation_commands(str(fields.get("validation") or ""))),
                "accepted_plan_root_cid": plan_root,
                "base_revision": head,
                "base_repository_tree_id": tree,
                "owning_repository": "ipfs_accelerate_py",
            }
        )
        tasks.append(task)
        observed_tasks.add(task_id)

    projection = config.get("initial_projection")
    projection = projection if isinstance(projection, Mapping) else {}
    dependency_count = sum(len(_split_csv(item[3].get("depends_on"))) for item in parsed_tasks)
    if (
        int(projection.get("task_count") or 0) != len(tasks)
        or int(projection.get("goal_count") or 0) != len(goals)
        or int(projection.get("task_dependency_count") or 0) != dependency_count
    ):
        raise OperatorError("compiled population differs from sealed projection")
    return {
        "schema": POPULATION_SCHEMA,
        "program_id": PROGRAM_ID,
        "repository_tree_id": tree,
        "source_head": head,
        "plan_root_cid": plan_root,
        "source_identities": {name: _identity(body) for name, body in sorted(sources.items())},
        "source_forest": _source_forest(head=head),
        "objectives": goals,
        "goal_edges": goal_edges,
        "plans": [
            {
                "plan_cid": plan_root,
                "plan_alias": "CASF-PLAN-R1",
                "goal_cid": goal_cids[ROOT_GOAL],
                "status": "active",
                "source_head": head,
                "repository_tree_id": tree,
            }
        ],
        "tasks": tasks,
        "task_cids_by_alias": task_cids,
        "goal_cids_by_alias": goal_cids,
    }


def _runtime_paths(board: Any) -> dict[str, Path]:
    program = board.resolved_database_program()
    runtime = board.path(board.runtime_paths["root"])
    database = _safe_path(ROOT, program.store_id, field="database_program.store_id")
    raw = board.payload.get("runtime_paths")
    raw = raw if isinstance(raw, Mapping) else {}
    evidence = _safe_path(
        ROOT,
        raw.get("evidence") or runtime.relative_to(ROOT) / "evidence",
        field="runtime_paths.evidence",
    )
    owner = _safe_path(
        ROOT,
        raw.get("quack_owner") or runtime.relative_to(ROOT) / "quack-owner",
        field="runtime_paths.quack_owner",
    )
    state = _safe_path(
        ROOT,
        raw.get("state") or runtime.relative_to(ROOT) / "state",
        field="runtime_paths.state",
    )
    for label, path in (
        ("database", database),
        ("evidence", evidence),
        ("owner", owner),
        ("state", state),
    ):
        try:
            path.relative_to(runtime)
        except ValueError as exc:
            raise OperatorError(f"{label} must remain below runtime root") from exc
    socket_identity = hashlib.sha256(
        _canonical_bytes(
            {
                "program_id": PROGRAM_ID,
                "repository_root": str(ROOT),
                "runtime_root": str(runtime),
                "store_id": _control_plane_store_id(program),
            }
        )
    ).hexdigest()[:20]
    owner_socket = (
        Path("/tmp")
        / f"ipfs-accelerate-casf-{os.geteuid()}"
        / f"owner-{socket_identity}.sock"
    )
    if len(os.fsencode(owner_socket)) > UNIX_SOCKET_PATH_CEILING:
        raise OperatorError("derived state-owner socket path exceeds its platform bound")
    return {
        "runtime": runtime,
        "database": database,
        "owner": owner,
        "state": state,
        "operator_evidence": evidence / "bootstrap-operator",
        "bootstrap_receipt": evidence / "bootstrap-operator" / "bootstrap-current.json",
        "ducklake_receipt": evidence / "bootstrap-operator" / "ducklake-current.json",
        "launch_receipt": evidence / "bootstrap-operator" / "launch-current.json",
        "status_receipt": evidence / "bootstrap-operator" / "status-current.json",
        "coordinator_receipt": (
            evidence / "bootstrap-operator" / "coordinator-current.json"
        ),
        "stop_receipt": evidence / "bootstrap-operator" / "stop-current.json",
        "owner_status": owner / "quack-state-server.status.json",
        "owner_socket": owner_socket,
        "owner_log": owner / "quack-state-server.log",
        "master_pid": state / "configured-board-master.pid",
        "supervisor_pid": state / "casf_supervisor.pid",
        "daemon_pid": state / "casf_managed_daemon.pid",
        "supervisor_status": state / "casf_supervisor_status.json",
        "task_state": state / "casf_task_state.json",
        "coordinator_log": state / "casf_event_coordinator.log",
        "executor_state": state / "executor",
        "executor_pid": state / "executor" / "casf_exec.pid",
        "executor_log": state / "executor" / "casf_exec.log",
    }


def _prepare_private_socket_parent(socket_path: Path) -> None:
    """Create or verify the server-derived private Unix-socket directory."""

    parent = socket_path.parent
    try:
        parent.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        created = False
    try:
        metadata = os.lstat(parent)
    except OSError as exc:
        raise OperatorError("state-owner socket directory is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise OperatorError("state-owner socket directory custody is unsafe")
    if created:
        os.chmod(parent, 0o700)


def _control_plane_store_id(program: Any) -> str:
    """Return the compact transactional identity, distinct from the DB path."""

    value = str(program.store_generation or "").strip()
    if not value or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", value) is None:
        raise OperatorError("database program has no compact control-plane store identity")
    return value


def _persist_receipt(
    paths: Mapping[str, Path], kind: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    body = dict(payload)
    identity_field = f"{kind}_receipt_id"
    body[identity_field] = _identity(body)
    receipt_id = body[identity_field]
    immutable = (
        paths["operator_evidence"]
        / "receipts"
        / (
            f"{kind}-{receipt_id[7:] if receipt_id.startswith('sha256:') else receipt_id}.json"
        )
    )
    if immutable.exists() and _json_object(immutable) != body:
        raise OperatorError(f"immutable {kind} receipt identity collision")
    if not immutable.exists():
        _atomic_json(immutable, body)
    _atomic_json(paths[f"{kind}_receipt"], body)
    return body


def _typed_ducklake_unavailable(
    paths: Mapping[str, Path], population: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = {
        "schema": DUCKLAKE_SCHEMA,
        "authoritative": False,
        "scheduling_prerequisite": False,
        "completion_prerequisite": False,
        "status": "unavailable",
        "reason_code": "implementation_and_qualification_pending_CASF_031_CASF_032",
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
    }
    receipt["projection_receipt_id"] = _identity(receipt)
    _atomic_json(paths["ducklake_receipt"], receipt)
    return receipt


def _validate_board() -> Mapping[str, Any]:
    """Run the board's fail-closed validator without inventing another parser."""

    import importlib.util

    path = ROOT / "scripts/validate_agent_supervisor_causal_event_federation_board.py"
    spec = importlib.util.spec_from_file_location("casf_board_validator", path)
    if spec is None or spec.loader is None:
        raise OperatorError("cannot load sealed CASF board validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.validate_program(check_source=True, require_database=False)
    if not isinstance(report, Mapping) or report.get("valid") is not True:
        raise OperatorError("sealed CASF board validator rejected the current tree")
    return report


def _owner_liveness(status_payload: Mapping[str, Any]) -> str:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        ProcessBirthIdentity,
        owner_liveness,
    )

    identity = status_payload.get("identity")
    birth = identity.get("process_birth") if isinstance(identity, Mapping) else None
    if not isinstance(birth, Mapping):
        return "absent"
    try:
        observed = owner_liveness(ProcessBirthIdentity.from_dict(birth))
    except Exception:
        return "unknown"
    if observed is OwnerLiveness.ALIVE:
        return "alive"
    if observed is OwnerLiveness.DEAD:
        return "dead"
    return "unknown"


def _outbox_worker_health(status_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the closed owner-published router-worker health projection."""

    raw = status_payload.get("outbox_worker")
    raw = raw if isinstance(raw, Mapping) else {}
    gateway = status_payload.get("typed_command_gateway")
    gateway = gateway if isinstance(gateway, Mapping) else {}
    last_error = str(raw.get("last_error_type") or "").strip()
    observer_error = str(gateway.get("last_observer_error_type") or "").strip()
    malformed = False
    counters: dict[str, int] = {}
    for name in ("watermark", "committed_sequence", "drain_count"):
        try:
            value = int(raw.get(name) or 0)
        except (TypeError, ValueError):
            malformed = True
            value = 0
        if value < 0:
            malformed = True
            value = 0
        counters[name] = value
    result = {
        "available": raw.get("available") is True,
        "thread_alive": raw.get("thread_alive") is True,
        "server_owned": raw.get("server_owned") is True,
        "polling": raw.get("polling"),
        **counters,
        "last_error_type": last_error,
        "commit_observer_bound": gateway.get("commit_observer_bound") is True,
        "observer_error_type": observer_error,
        "malformed": malformed,
    }
    result["caught_up"] = bool(
        not malformed
        and result["watermark"] >= result["committed_sequence"]
    )
    result["healthy"] = bool(
        result["available"]
        and result["thread_alive"]
        and result["server_owned"]
        and result["polling"] is False
        and result["caught_up"]
        and result["commit_observer_bound"]
        and not observer_error
        and not last_error
        and not malformed
    )
    return result


def _state_owner_outbox_health(server: Any) -> dict[str, Any]:
    """Evaluate the worker together with its canonical commit observer.

    ``outbox_worker_capability()`` is intentionally only a worker projection.
    Startup and steady-state admission also require the typed gateway's
    commit-observer attestation, which is published by ``status()``.
    """

    return _outbox_worker_health(server.status())


def _process_birth(pid: int) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        read_process_birth,
    )

    try:
        birth = read_process_birth(pid)
    except OSError as exc:
        raise OperatorError(f"cannot establish process-birth identity for pid {pid}") from exc
    if birth is None or birth.start_time_ticks <= 0:
        raise OperatorError(f"process {pid} exited before its birth was captured")
    return birth.to_dict()


def _birth_liveness(payload: Mapping[str, Any]) -> str:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        ProcessBirthIdentity,
        owner_liveness,
    )

    try:
        value = owner_liveness(ProcessBirthIdentity.from_dict(payload))
    except Exception:
        return "unknown"
    return {
        OwnerLiveness.ALIVE: "alive",
        OwnerLiveness.DEAD: "dead",
    }.get(value, "unknown")


def _token_path(owner_dir: Path, secret_handle: str) -> Path:
    # Supervisors receive the typed command-gateway credential, never the
    # Quack extension's generic SQL transport credential.
    del secret_handle
    return owner_dir / "typed-state-owner.token"


def _read_owner_token(path: Path) -> str:
    try:
        metadata = os.stat(path, follow_symlinks=False)
        token = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise OperatorError("Quack token vault is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
        raise OperatorError("Quack token vault is not a private regular file")
    if TOKEN_RE.fullmatch(token) is None:
        raise OperatorError("Quack token vault contains malformed material")
    return token


def _port_is_free(host: str, port: int) -> bool:
    """Return whether nothing is accepting connections on the loopback port.

    Bind probes treat TCP TIME_WAIT as occupied, which blocked complete stop
    after a clean owner exit. Connection-refused means no listener remains.
    """

    try:
        with socket.create_connection((host, port), timeout=0.25):
            return False
    except ConnectionRefusedError:
        return True
    except OSError:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                return False
        return True


def _require_free_port(host: str, port: int) -> None:
    if not _port_is_free(host, port):
        raise OperatorError(
            f"configured Quack loopback endpoint {host}:{port} is occupied; "
            "refusing to kill or reuse the existing listener"
        )


def _quack_capability() -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
        quack_health_check,
    )

    try:
        report = quack_health_check()
        payload = dict(report.to_dict()) if hasattr(report, "to_dict") else dict(report)
    except Exception as exc:
        raise OperatorError(f"Quack capability preflight failed ({type(exc).__name__})") from exc
    if payload.get("passes_health_check") is not True:
        raise OperatorError("Quack is absent, incompatible, or not health-qualified")
    return payload


def preflight(config_path: Path, *, require_free_port: bool = True) -> dict[str, Any]:
    board, config = _load_config(config_path)
    program = board.resolved_database_program()
    endpoint = QUACK_ENDPOINT_RE.fullmatch(program.quack_endpoint)
    assert endpoint is not None
    host, port = endpoint.group(1), int(endpoint.group(2))
    if require_free_port:
        _require_free_port(host, port)
    capability = _quack_capability()
    route = _route_preflight(board)
    return {
        "schema": OPERATOR_SCHEMA,
        "command": "preflight",
        "ok": True,
        "program_id": PROGRAM_ID,
        "quack_endpoint": program.quack_endpoint,
        "quack_capability": capability,
        "provider_route": route,
        "authoritative_store": program.store_id,
        "control_plane_store_id": _control_plane_store_id(program),
        "authority_mode": "quack",
        "maximum_supervisors": 1,
        "maximum_active_subagents": 1,
        "event_driven_qualified": False,
        "high_concurrency_qualified": False,
        "ducklake_authoritative": False,
    }


def materialize(config_path: Path) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_causal_event_federation_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    board, config = _load_config(config_path)
    paths = _runtime_paths(board)
    _validate_board()
    population = _population(board, config)
    initial_projection = config.get("initial_projection")
    initial_projection = initial_projection if isinstance(initial_projection, Mapping) else {}
    expected_ready = [str(item) for item in initial_projection.get("ready_task_ids", ())]
    owner_status = _json_object(paths["owner_status"]) if paths["owner_status"].is_file() else {}
    if _owner_liveness(owner_status) in {"alive", "unknown"}:
        raise OperatorError("offline materialization refused while owner may be live")
    receipt_path = paths["bootstrap_receipt"]
    if paths["database"].exists() or receipt_path.exists():
        if not paths["database"].is_file() or not receipt_path.is_file():
            raise OperatorError("partial bootstrap state exists; operator review required")
        prior = _json_object(receipt_path)
        _verify_receipt(prior, kind="bootstrap")
        if (
            prior.get("schema") != BOOTSTRAP_SCHEMA
            or prior.get("program_id") != PROGRAM_ID
        ):
            raise OperatorError("existing bootstrap receipt has stale authority")
        stale_identity = any(
            prior.get(field) != population.get(field)
            for field in ("source_head", "repository_tree_id", "plan_root_cid")
        )
        if stale_identity:
            if _owner_liveness(owner_status) in {"alive", "unknown"}:
                raise OperatorError("existing control plane is bound to a stale identity")
            if paths["launch_receipt"].is_file():
                raise OperatorError("existing control plane is bound to a stale identity")
            _retire_consumed_generation(
                paths,
                launch_id=str(prior.get("bootstrap_receipt_id") or "bootstrap"),
            )
        else:
            with DatabaseTaskSource(
                paths["database"],
                owner_id="casf-bootstrap:verify-existing",
                install_schema=False,
                repository_tree_id=str(population["repository_tree_id"]),
                plan_root_cid=str(population["plan_root_cid"]),
            ) as source:
                snapshot = source.snapshot().to_dict()
                ready = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
            if (
                int(snapshot["task_count"]) != 44
                or int(snapshot["goal_count"]) != 17
                or int(snapshot["dependency_count"]) != 191
                or ready != expected_ready
            ):
                raise OperatorError("existing control-plane population differs from seal")
            verify_causal_event_federation_schema(paths["database"])
            return {
                "schema": OPERATOR_SCHEMA,
                "command": "materialize",
                "ok": True,
                "idempotent_replay": True,
                "bootstrap_receipt": prior,
                "snapshot": snapshot,
            }

    paths["runtime"].mkdir(parents=True, exist_ok=True)
    with DatabaseTaskSource(
        paths["database"],
        owner_id="casf-bootstrap:exclusive-single-writer",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
    ) as source:
        control_receipt = dict(source.materialize(population))
        snapshot = source.snapshot().to_dict()
        ready = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
    if (
        int(snapshot["task_count"]) != 44
        or int(snapshot["goal_count"]) != 17
        or int(snapshot["dependency_count"]) != 191
        or ready != expected_ready
    ):
        raise OperatorError("materialized control-plane population is not exact")
    schema = verify_causal_event_federation_schema(paths["database"])
    ducklake = _typed_ducklake_unavailable(paths, population)
    receipt = _persist_receipt(
        paths,
        "bootstrap",
        {
            "schema": BOOTSTRAP_SCHEMA,
            "program_id": PROGRAM_ID,
            "source_head": population["source_head"],
            "repository_tree_id": population["repository_tree_id"],
            "plan_root_cid": population["plan_root_cid"],
            "source_identities": population["source_identities"],
            "source_forest": population["source_forest"],
            "database_task_source_receipt": control_receipt,
            "projection_cid": snapshot["projection_cid"],
            "task_count": snapshot["task_count"],
            "goal_count": snapshot["goal_count"],
            "dependency_count": snapshot["dependency_count"],
            "initial_ready_task_ids": ready,
            "schema_revision": schema["schema_revision"],
            "schema_fingerprint": schema["base_schema"]["schema_fingerprint"],
            "authority": {
                "operational_state": "DuckDB/DatabaseTaskSource@1",
                "live_transport": "QuackStateServer@1",
                "ducklake": "non_authoritative_unavailable",
            },
            "ducklake_projection": ducklake,
        },
    )
    return {
        "schema": OPERATOR_SCHEMA,
        "command": "materialize",
        "ok": True,
        "idempotent_replay": False,
        "bootstrap_receipt": receipt,
        "snapshot": snapshot,
    }


def _supervisor_runtime_command(config_path: Path, descriptor: int) -> list[str]:
    if isinstance(descriptor, bool) or descriptor < 3:
        raise OperatorError("supervisor credential descriptor is invalid")
    argv = [
        *_operator_command(config_path, "supervisor-runtime"),
        "--credential-fd",
        str(descriptor),
    ]
    lowered = " ".join(argv).lower()
    if "token=" in lowered or STATE_TOKEN_ENV.lower() in lowered:
        raise OperatorError("supervisor argv would expose credential material")
    return argv


def _spawn_event_supervisor(
    *,
    server: Any,
    board: Any,
    config_path: Path,
    paths: Mapping[str, Path],
    admission: Any,
    task_projection: Mapping[str, Any],
) -> tuple[subprocess.Popen[Any], dict[str, Any]]:
    """Spawn one PID-bound event coordinator and pass grants by private pipe."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
        process_birth_id,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        ProcessBirthIdentity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        SUPERVISOR_EVENT_CHILD_ALLOWED_OPERATIONS,
        SUPERVISOR_RUNTIME_CHILD_ALLOWED_OPERATIONS,
    )

    read_descriptor, write_descriptor = os.pipe()
    os.set_inheritable(read_descriptor, True)
    process: subprocess.Popen[Any] | None = None
    log_handle: Any = None
    try:
        paths["state"].mkdir(parents=True, exist_ok=True)
        log_handle = paths["coordinator_log"].open("ab")
        os.chmod(paths["coordinator_log"], 0o600)
        process = subprocess.Popen(
            _supervisor_runtime_command(config_path, read_descriptor),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=_state_owner_environment(),
            pass_fds=(read_descriptor,),
            start_new_session=True,
        )
        os.close(read_descriptor)
        read_descriptor = -1
        birth_payload = _process_birth(process.pid)
        birth = ProcessBirthIdentity.from_dict(birth_payload)
        birth_id = process_birth_id(birth)
        common = {
            "peer_pid": process.pid,
            "process_birth_id": birth_id,
            "tenant_id": admission.federation_identity.binding.tenant_id,
            "federation_id": admission.federation_identity.record_id,
            "ttl_seconds": 86_400.0,
        }
        runtime_client_id = "casf-supervisor-runtime:" + admission.supervisor.record_id
        event_client_id = "casf-supervisor-events:" + admission.supervisor.record_id
        runtime_token = server.issue_typed_client_grant(
            client_id=runtime_client_id,
            allowed_operations=tuple(SUPERVISOR_RUNTIME_CHILD_ALLOWED_OPERATIONS),
            allowed_command_operations=(
                "supervisor.runtime.attest",
                "supervisor.transition",
            ),
            entity_scopes={"supervisor_id": admission.supervisor.record_id},
            **common,
        )
        event_token = server.issue_typed_client_grant(
            client_id=event_client_id,
            allowed_operations=tuple(SUPERVISOR_EVENT_CHILD_ALLOWED_OPERATIONS),
            allowed_command_operations=(
                "event.delivery.record",
                "event.acknowledge",
            ),
            entity_scopes={
                "subscription_id": admission.subscription.subscription_id,
            },
            **common,
        )
        server_identity = server.identity
        if server_identity is None:
            raise OperatorError("state owner lost its identity before child admission")
        bundle = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "causal-federation-runtime-credentials@1"
            ),
            "endpoint": board.resolved_database_program().quack_endpoint,
            "socket_path": str(paths["owner_socket"]),
            "store_id": _control_plane_store_id(
                board.resolved_database_program()
            ),
            "server_id": server_identity.server_id,
            "process_birth_id": birth_id,
            "runtime_token": runtime_token,
            "event_token": event_token,
            "tenant_id": admission.federation_identity.binding.tenant_id,
            "federation_id": admission.federation_identity.record_id,
            "supervisor_id": admission.supervisor.record_id,
            "subscription_id": admission.subscription.subscription_id,
            "consumer_id": admission.subscription.consumer_id,
            "fencing_epoch": admission.fencing_epoch,
            "task_count": int(task_projection.get("task_count") or 0),
            "completed_count": int(task_projection.get("completed_count") or 0),
            "ready_count": int(task_projection.get("ready_count") or 0),
            "status_path": str(paths["supervisor_status"]),
            "task_state_path": str(paths["task_state"]),
        }
        encoded = _canonical_bytes(bundle)
        if len(encoded) > 65_536:
            raise OperatorError("supervisor credential bundle exceeds its bound")
        offset = 0
        while offset < len(encoded):
            offset += os.write(write_descriptor, encoded[offset:])
        os.close(write_descriptor)
        write_descriptor = -1
        _atomic_text(paths["master_pid"], f"{process.pid}\n")
        _atomic_text(paths["supervisor_pid"], f"{process.pid}\n")
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                status_payload = _read_optional_json(paths["supervisor_status"])
                error_class = str(status_payload.get("error_class") or "").strip()
                detail = f" error_class={error_class}" if error_class else ""
                raise OperatorError(
                    "event coordinator exited before IDLE readiness" + detail
                )
            status_payload = _read_optional_json(paths["supervisor_status"])
            if (
                int(status_payload.get("supervisor_pid") or 0) == process.pid
                and status_payload.get("lifecycle_state") == "IDLE"
                and status_payload.get("event_wait_qualified") is True
                and int(status_payload.get("events_processed") or 0) >= 1
                and bool(status_payload.get("first_event_id"))
                and bool(status_payload.get("first_acknowledgement_id"))
                and bool(status_payload.get("first_delivery_attempt_id"))
            ):
                return process, birth_payload
            time.sleep(0.1)
        raise OperatorError(
            "event coordinator did not prove bounded IDLE event acknowledgement"
        )
    except BaseException:
        if process is not None:
            try:
                _terminate_birth(_process_birth(process.pid), grace_seconds=5.0)
            except Exception:
                pass
        raise
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except OSError:
                pass
        for descriptor in (read_descriptor, write_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _plan_executor_command(board: Any, config: Mapping[str, Any], paths: Mapping[str, Path]) -> list[str]:
    """Build the isolated plan-executor argv that shares the live Quack owner."""

    program = board.resolved_database_program()
    worktree_root = (
        str(board.path(program.worktree_root))
        if program.worktree_root
        else str(paths["runtime"] / "worktrees")
    )
    runtime_paths = config.get("runtime_paths")
    runtime_paths = runtime_paths if isinstance(runtime_paths, Mapping) else {}
    merge_queue = str(runtime_paths.get("merge_queue") or "").strip()
    merge_queue_dir = (
        str(board.path(merge_queue))
        if merge_queue
        else str(paths["runtime"] / "merge-queue")
    )
    argv = [
        sys.executable,
        "-P",
        "-u",
        "-m",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
        "--interval",
        str(float(config.get("daemon_interval_seconds") or 60)),
        "--todo-path",
        str(board.path(board.taskboard_path)),
        "--state-dir",
        str(paths["executor_state"]),
        "--task-prefix",
        TASK_PREFIX,
        "--state-prefix",
        "casf_exec",
        "--board-namespace",
        BOARD_NAMESPACE,
        "--max-task-attempts",
        str(int(config.get("max_task_attempts") or 4)),
        "--task-source-kind",
        "duckdb",
        "--authority-mode",
        "quack",
        "--state-failover-policy",
        "fail_closed",
        "--quack-endpoint",
        str(program.quack_endpoint),
        "--endpoint-secret-handle",
        str(program.endpoint_secret_handle),
        "--state-store-id",
        str(program.store_id),
        "--state-store-generation",
        str(program.store_generation),
        "--state-schema-revision",
        str(program.schema_revision),
        "--event-store-path",
        str(program.event_store_path),
        "--runtime-registry-path",
        str(program.runtime_registry_path),
        "--export-profile",
        str(program.export_profile),
        "--implement",
        "--implementation-timeout",
        str(float(config.get("implementation_timeout_seconds") or 14400)),
        "--worktree-root",
        worktree_root,
        "--merge-target-branch",
        str(board.merge_target_branch),
        "--merge-queue-dir",
        merge_queue_dir,
    ]
    for relative in board.protected_paths:
        argv.extend(["--implementation-protected-path", str(relative)])
    lowered = " ".join(argv).lower()
    if "token=" in lowered or STATE_TOKEN_ENV.lower() in lowered:
        raise OperatorError("plan executor argv would expose credential material")
    return argv


def _plan_executor_environment() -> dict[str, str]:
    """Return the isolated executor environment with an importable package root.

    The argv uses ``python -P -m``, so cwd is not placed on ``sys.path``.  The
    owner child can import because it is launched as a script; the executor
    cannot unless ``PYTHONPATH`` names the repository root.
    """

    environment = _state_owner_environment()
    root = str(ROOT)
    existing = [
        part
        for part in str(environment.get("PYTHONPATH") or "").split(os.pathsep)
        if part and part != root
    ]
    environment["PYTHONPATH"] = os.pathsep.join([root, *existing])
    environment["PYTHONUNBUFFERED"] = "1"
    environment.pop(STATE_TOKEN_ENV, None)
    return environment


def _plan_executor_bootstrap_command(config_path: Path, descriptor: int) -> list[str]:
    if isinstance(descriptor, bool) or descriptor < 3:
        raise OperatorError("plan executor credential descriptor is invalid")
    argv = [
        *_operator_command(config_path, "plan-executor"),
        "--credential-fd",
        str(descriptor),
    ]
    lowered = " ".join(argv).lower()
    if "token=" in lowered or STATE_TOKEN_ENV.lower() in lowered:
        raise OperatorError("plan executor argv would expose credential material")
    return argv


def _spawn_plan_executor(
    *,
    server: Any,
    board: Any,
    config: Mapping[str, Any],
    config_path: Path,
    paths: Mapping[str, Path],
) -> tuple[subprocess.Popen[Any], dict[str, Any]]:
    """Start one plan executor against the live owner without clobbering coordinator status."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
        process_birth_id,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        ProcessBirthIdentity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        build_control_plane_operation_catalog,
    )

    paths["executor_state"].mkdir(parents=True, exist_ok=True)
    read_descriptor, write_descriptor = os.pipe()
    os.set_inheritable(read_descriptor, True)
    process: subprocess.Popen[Any] | None = None
    log_handle: Any = None
    try:
        log_handle = paths["executor_log"].open("ab")
        os.chmod(paths["executor_log"], 0o600)
        process = subprocess.Popen(
            _plan_executor_bootstrap_command(config_path, read_descriptor),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=_plan_executor_environment(),
            pass_fds=(read_descriptor,),
            start_new_session=True,
        )
        os.close(read_descriptor)
        read_descriptor = -1
        birth_payload = _process_birth(process.pid)
        birth = ProcessBirthIdentity.from_dict(birth_payload)
        birth_id = process_birth_id(birth)
        server_identity = server.identity
        if server_identity is None:
            raise OperatorError("state owner lost its identity before executor admission")
        token = server.issue_typed_client_grant(
            client_id="casf-plan-executor:" + birth_id,
            process_birth_id=birth_id,
            peer_pid=process.pid,
            allowed_operations=tuple(build_control_plane_operation_catalog()),
            allowed_command_operations=("task.status.cas",),
            ttl_seconds=86_400.0,
        )
        program = board.resolved_database_program()
        attach_token = token
        vault = getattr(server, "_vault", None)
        resolve = getattr(vault, "resolve", None)
        if callable(resolve):
            try:
                resolved = str(resolve(program.endpoint_secret_handle) or "").strip()
            except Exception:
                resolved = ""
            if resolved:
                attach_token = resolved
        bundle = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "causal-federation-executor-credentials@1"
            ),
            "endpoint": program.quack_endpoint,
            "socket_path": str(paths["owner_socket"]),
            "store_id": _control_plane_store_id(program),
            "server_id": server_identity.server_id,
            "process_birth_id": birth_id,
            "token": attach_token,
        }
        encoded = _canonical_bytes(bundle)
        if len(encoded) > 65_536:
            raise OperatorError("plan executor credential bundle exceeds its bound")
        offset = 0
        while offset < len(encoded):
            offset += os.write(write_descriptor, encoded[offset:])
        os.close(write_descriptor)
        write_descriptor = -1
        _atomic_text(paths["executor_pid"], f"{process.pid}\n")
        return process, birth_payload
    except BaseException:
        if process is not None:
            try:
                _terminate_birth(_process_birth(process.pid), grace_seconds=5.0)
            except Exception:
                pass
        raise
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except OSError:
                pass
        for descriptor in (read_descriptor, write_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def plan_executor(config_path: Path, credential_fd: int) -> int:
    """Install a PID-bound grant, then exec the isolated implementation daemon."""

    if isinstance(credential_fd, bool) or credential_fd < 3:
        raise OperatorError("plan executor credential descriptor is invalid")
    encoded = b""
    while True:
        chunk = os.read(credential_fd, 65_536)
        if not chunk:
            break
        encoded += chunk
        if len(encoded) > 65_536:
            raise OperatorError("plan executor credential bundle exceeds its bound")
    os.close(credential_fd)
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorError("plan executor credential bundle is malformed") from exc
    if not isinstance(payload, dict):
        raise OperatorError("plan executor credential bundle is malformed")
    token = str(payload.get("token") or "").strip()
    socket_path = str(payload.get("socket_path") or "").strip()
    if not token or not socket_path:
        raise OperatorError("plan executor credential bundle is incomplete")
    if TOKEN_RE.fullmatch(token) is None:
        raise OperatorError("plan executor credential bundle is malformed")
    board, config = _load_config(config_path)
    paths = _runtime_paths(board)
    environment = _plan_executor_environment()
    environment[STATE_TOKEN_ENV] = token
    environment[STATE_OWNER_SOCKET_ENV] = socket_path
    program = board.resolved_database_program()
    environment["IPFS_ACCELERATE_AGENT_STATE_STORE_ID"] = str(program.store_id)
    environment["IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION"] = str(
        program.store_generation
    )
    environment["IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR"] = str(
        paths["owner"] / "mutations"
    )
    argv = _plan_executor_command(board, config, paths)
    os.execvpe(argv[0], argv, environment)
    raise OperatorError("plan executor exec returned")


def _process_owner_commands(
    repository: Any,
    command_dir: Path,
    *,
    token: str,
    expected_store_id: str,
    expected_store_generation: str,
) -> None:
    """Apply executor CAS/evidence commands on the exclusive owner connection."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        execute_quack_owner_command,
        quack_owner_command_error_code,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        quack_owner_command_response,
        validate_quack_owner_command_request,
    )

    command_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(command_dir, 0o700)
    for request in sorted(command_dir.glob("*.request.json")):
        done = request.with_name(request.name.replace(".request.json", ".done.json"))
        payload: Mapping[str, Any] = {}
        expected_request_id = request.name.removesuffix(".request.json")
        try:
            metadata = request.lstat()
            if not stat.S_ISREG(metadata.st_mode) or request.is_symlink():
                raise OperatorError("owner command must be a regular non-symlink file")
            if metadata.st_uid != os.getuid() or metadata.st_size > 262_144:
                raise OperatorError("owner command file owner or size is invalid")
            try:
                decoded = json.loads(request.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(decoded, Mapping):
                raise OperatorError("owner command request must be an object")
            payload = decoded
            command, command_payload = validate_quack_owner_command_request(
                payload,
                token=token,
                expected_request_id=expected_request_id,
                expected_store_id=expected_store_id,
                expected_store_generation=expected_store_generation,
            )
            result = execute_quack_owner_command(
                repository,
                command,
                command_payload,
                request_id=expected_request_id,
                store_id=expected_store_id,
                store_generation=expected_store_generation,
            )
            _atomic_json(
                done,
                quack_owner_command_response(payload, token=token, result=result),
            )
        except Exception as exc:
            response_request = (
                payload
                if payload
                else {
                    "request_id": expected_request_id,
                    "command": "invalid",
                    "store_id": expected_store_id,
                    "store_generation": expected_store_generation,
                }
            )
            error_code = quack_owner_command_error_code(exc)
            _atomic_json(
                done,
                quack_owner_command_response(
                    response_request,
                    token=token,
                    error_code=error_code,
                    error_message=str(exc)[:500],
                ),
            )
        try:
            request.unlink()
        except FileNotFoundError:
            pass


def state_owner(config_path: Path, *, admit_task_execution: bool = False) -> int:
    """Run the exclusive Quack owner in the foreground (internal command)."""

    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        QuackStateServerReadyError,
        ServerLifecycle,
        build_server,
    )

    board, config = _load_config(config_path)
    paths = _runtime_paths(board)
    if not paths["database"].is_file() or not paths["bootstrap_receipt"].is_file():
        raise OperatorError("materialize the sealed board before starting Quack")
    bootstrap = _json_object(paths["bootstrap_receipt"])
    _verify_receipt(bootstrap, kind="bootstrap")
    if (
        bootstrap.get("schema") != BOOTSTRAP_SCHEMA
        or bootstrap.get("program_id") != PROGRAM_ID
    ):
        raise OperatorError("bootstrap receipt has stale authority")
    program = board.resolved_database_program()
    endpoint = QUACK_ENDPOINT_RE.fullmatch(program.quack_endpoint)
    assert endpoint is not None
    host, port = endpoint.group(1), int(endpoint.group(2))
    _require_free_port(host, port)
    _quack_capability()
    _prepare_private_socket_parent(paths["owner_socket"])
    server = build_server(
        database_path=paths["database"],
        state_dir=paths["owner"],
        host=host,
        port=port,
        repository_id="repository:ipfs_accelerate_py",
        store_id=_control_plane_store_id(program),
        secret_handle=program.endpoint_secret_handle,
        allow_experimental=False,
        typed_command_socket_path=paths["owner_socket"],
    )
    if server.typed_command_socket_path() != paths["owner_socket"]:
        raise OperatorError("state owner did not retain the derived socket identity")
    identity = server.start()
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
        QuackStateClient,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        build_control_plane_operation_catalog,
        catalog_fingerprint,
    )

    owner_catalog = build_control_plane_operation_catalog()
    command_token = server.issue_typed_client_grant(
        client_id="casf-state-owner:federation-runtime",
        process_birth_id=identity.process_birth_id,
        allowed_operations=tuple(owner_catalog),
        allowed_command_operations=(
            "federation.create",
            "budget.reserve",
            "budget.release",
            "supervisor.register",
            "supervisor.runtime.attest",
            "supervisor.transition",
            "subagent.register",
            "subagent.slot.reserve",
            "subagent.slot.release",
            "subagent.outcome",
            "subscription.register",
            "event.route.persist",
            "event.outbox.disposition",
            "event.delivery.record",
            "event.delivery.fail",
            "event.acknowledge",
        ),
    )
    owner_client = QuackStateClient(
        owner_id="casf-state-owner:federation-runtime",
        store_id=_control_plane_store_id(program),
        process_birth_id=identity.process_birth_id,
    )
    with _scoped_environment(
        {
            STATE_TOKEN_ENV: command_token,
            STATE_OWNER_SOCKET_ENV: str(paths["owner_socket"]),
        }
    ):
        owner_client.attach(
            program.quack_endpoint,
            server_id=identity.server_id,
        )
    repository = server.bind_federation_repository(
        owner_client,
        require_quack_authority=True,
    )
    from ipfs_accelerate_py.agent_supervisor.federation.bootstrap_runtime import (
        admit_bootstrap_federation,
    )

    projection = config.get("initial_projection")
    projection = projection if isinstance(projection, Mapping) else {}
    ready_task_refs = tuple(str(item) for item in projection.get("ready_task_ids", ()))
    completed_task_refs = tuple(
        str(item) for item in projection.get("completed_task_ids", ())
    )
    generation = owner_client.load_generation()
    admission = admit_bootstrap_federation(
        repository,
        profile=config["federation_bootstrap_policy"],
        repository_id="repository:ipfs_accelerate_py",
        repository_tree_id=str(bootstrap["repository_tree_id"]),
        plan_root_ref=str(bootstrap["plan_root_cid"]),
        operation_catalog_ref=catalog_fingerprint(owner_catalog),
        control_plane_generation=generation.generation,
        fencing_epoch=generation.fence_epoch,
        ready_task_refs=ready_task_refs,
        authentication_key=secrets.token_bytes(32),
    )
    server.bind_typed_status_scope()
    # Prove the loopback Quack transport before any owner worker or child can
    # use the shared DuckDB connection.  ``ready()`` performs a live extension
    # query on that connection; running it after concurrent admission would
    # bypass the gateway transaction lock and corrupt an otherwise valid
    # semantic-command observation.
    ready: dict[str, Any] | None = None
    last_ready_error: BaseException | None = None
    for _attempt in (1, 2, 3):
        try:
            ready = server.ready()
            break
        except QuackStateServerReadyError as exc:
            last_ready_error = exc
            detail = str(exc)
            if (
                "InvalidInputException" not in detail
                and "Invalid connection" not in detail
            ):
                raise
            time.sleep(0.25)
    if ready is None:
        assert last_ready_error is not None
        raise last_ready_error
    outbox_runtime = server.start_federation_outbox_worker()
    if _state_owner_outbox_health(server)["healthy"] is not True:
        raise OperatorError("state-owner outbox worker failed startup health")
    task_projection = {
        "task_count": int(projection.get("task_count") or 0),
        "completed_count": len(completed_task_refs),
        "ready_count": len(ready_task_refs),
    }
    supervisor_process, supervisor_birth = _spawn_event_supervisor(
        server=server,
        board=board,
        config_path=config_path,
        paths=paths,
        admission=admission,
        task_projection=task_projection,
    )
    final_outbox_health = _state_owner_outbox_health(server)
    if final_outbox_health["healthy"] is not True:
        try:
            _terminate_birth(supervisor_birth, grace_seconds=15.0)
        finally:
            owner_client.close()
            server.stop()
        raise OperatorError(
            "state-owner outbox worker failed coordinator-admission health"
        )
    executor_process: subprocess.Popen[Any] | None = None
    executor_birth: dict[str, Any] | None = None
    if admit_task_execution:
        try:
            executor_process, executor_birth = _spawn_plan_executor(
                server=server,
                board=board,
                config=config,
                config_path=config_path,
                paths=paths,
            )
        except BaseException:
            try:
                _terminate_birth(supervisor_birth, grace_seconds=15.0)
            finally:
                owner_client.close()
                server.stop()
            raise
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        IntentRepository,
    )

    owner_connection = getattr(server, "_connection", None)
    if owner_connection is None:
        raise OperatorError("state-owner connection is unavailable")
    owner_repository = IntentRepository(
        paths["database"],
        bound_connection=owner_connection,
        owner_id="casf-state-owner",
        session_id=f"casf-state-owner-{os.getpid()}",
        install_schema=False,
    )
    vault = getattr(server, "_vault", None)
    resolve = getattr(vault, "resolve", None)
    owner_token = str(resolve(program.endpoint_secret_handle) if callable(resolve) else "")
    if not owner_token:
        raise OperatorError("state-owner attach token is unavailable for command inbox")
    command_dir = paths["owner"] / "mutations"
    print(
        json.dumps(
            {
                "schema": OPERATOR_SCHEMA,
                "command": "state-owner",
                "ready": True,
                "identity": identity.to_dict(),
                "live": ready,
                "outbox_runtime": dict(outbox_runtime),
                "outbox_health": final_outbox_health,
                "federation_admission": admission.public_dict(),
                "supervisor_process_birth": supervisor_birth,
                "executor_process_birth": executor_birth,
                "task_execution_admitted": bool(admit_task_execution),
                "event_wait_qualified": True,
                "multi_supervisor_qualified": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    stopping = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    runtime_exit_code: int | None = None
    executor_restarts = 0
    max_executor_restarts = int(config.get("max_restarts") or 8)
    if server.lifecycle is ServerLifecycle.READY:
        while not stopping.wait(0.05):
            _process_owner_commands(
                owner_repository,
                command_dir,
                token=owner_token,
                expected_store_id=str(program.store_id),
                expected_store_generation=str(program.store_generation),
            )
            if _state_owner_outbox_health(server)["healthy"] is not True:
                runtime_exit_code = 1
                break
            runtime_exit_code = supervisor_process.poll()
            if runtime_exit_code is not None:
                break
            if admit_task_execution and (
                executor_process is None or executor_process.poll() is not None
            ):
                # An isolated executor crash must not tear down the live
                # owner or event coordinator.  Keep retrying so remaining
                # CASF tasks can resume against the same Quack generation.
                backoff = min(60.0, 2.0 * (2 ** min(executor_restarts, 5)))
                if executor_restarts >= max_executor_restarts and stopping.wait(
                    backoff
                ):
                    break
                executor_restarts += 1
                try:
                    executor_process, executor_birth = _spawn_plan_executor(
                        server=server,
                        board=board,
                        config=config,
                        config_path=config_path,
                        paths=paths,
                    )
                except Exception:
                    executor_process = None
                    if stopping.wait(backoff):
                        break
    if executor_birth is not None:
        try:
            _terminate_birth(executor_birth, grace_seconds=15.0)
        except OperatorError:
            if runtime_exit_code in {None, 0}:
                runtime_exit_code = 1
    if supervisor_process.poll() is None:
        try:
            _terminate_birth(supervisor_birth, grace_seconds=15.0)
        except OperatorError:
            runtime_exit_code = 1
    else:
        runtime_exit_code = supervisor_process.returncode
    try:
        owner_repository.close()
    except Exception:
        pass
    owner_client.close()
    result = server.stop()
    print(
        json.dumps(
            {
                **result,
                "supervisor_exit_code": runtime_exit_code,
                "executor_restarts": executor_restarts,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if runtime_exit_code in {None, 0, -signal.SIGTERM} else 1


def _read_optional_json(path: Path, *, maximum_bytes: int = 4_194_304) -> dict[str, Any]:
    """Read one private runtime projection without following a symlink."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise OperatorError(f"runtime projection is uninspectable: {path}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_size > maximum_bytes
        or metadata.st_nlink != 1
    ):
        raise OperatorError(f"runtime projection is not a bounded regular file: {path}")
    return _json_object(path)


def _read_pid(path: Path) -> int | None:
    try:
        metadata = os.lstat(path)
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OperatorError(f"PID projection is uninspectable: {path}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or len(raw) > 32
        or re.fullmatch(rb"[1-9][0-9]*\n?", raw) is None
    ):
        raise OperatorError(f"PID projection is malformed: {path}")
    return int(raw.strip())


def _owner_identity(
    board: Any,
    status_payload: Mapping[str, Any],
    *,
    expected_pid: int | None = None,
) -> dict[str, Any]:
    """Validate the exact live owner identity against the sealed program."""

    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        ProcessBirthIdentity,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        StateServerIdentity,
    )

    if str(status_payload.get("lifecycle") or "") != "ready":
        raise OperatorError("Quack state owner is not in READY lifecycle")
    raw = status_payload.get("identity")
    if not isinstance(raw, Mapping):
        raise OperatorError("Quack status has no typed server identity")
    try:
        if raw.get("schema") != StateServerIdentity.SCHEMA:
            raise ValueError("identity schema mismatch")
        if raw.get("interface") != StateServerIdentity.INTERFACE:
            raise ValueError("identity interface mismatch")
        raw_birth = raw.get("process_birth")
        if not isinstance(raw_birth, Mapping):
            raise ValueError("identity process birth missing")
        identity = StateServerIdentity(
            server_id=str(raw.get("server_id") or ""),
            store_id=str(raw.get("store_id") or ""),
            database_uuid=str(raw.get("database_uuid") or ""),
            schema_revision=int(raw.get("schema_revision") or 0),
            schema_fingerprint=str(raw.get("schema_fingerprint") or ""),
            generation=int(raw.get("generation") or 0),
            fence_epoch=int(raw.get("fence_epoch") or 0),
            revision=int(raw.get("revision") or 0),
            process_birth=ProcessBirthIdentity.from_dict(raw_birth),
            listen_uri=str(raw.get("listen_uri") or ""),
            extension_fingerprint=str(raw.get("extension_fingerprint") or ""),
            credential_generation=int(raw.get("credential_generation") or 0),
            secret_handle=str(raw.get("secret_handle") or ""),
            repository_id=str(raw.get("repository_id") or ""),
            startup_epoch=int(raw.get("startup_epoch") or 0),
            started_at=str(raw.get("started_at") or ""),
            status=str(raw.get("status") or ""),
        )
        if raw.get("process_birth_id") != identity.process_birth_id:
            raise ValueError("identity process birth CID mismatch")
    except Exception as exc:
        raise OperatorError("Quack server identity is malformed") from exc
    program = board.resolved_database_program()
    payload = identity.to_dict()
    birth = payload.get("process_birth")
    if not isinstance(birth, Mapping):
        raise OperatorError("Quack server identity has no process birth")
    if expected_pid is not None and int(birth.get("pid") or 0) != expected_pid:
        raise OperatorError("Quack status belongs to a stale process identity")
    if _birth_liveness(birth) != "alive":
        raise OperatorError("Quack server process birth is not provably alive")
    if payload.get("store_id") != _control_plane_store_id(program):
        raise OperatorError("Quack server store identity differs from scheduler")
    if payload.get("listen_uri") != program.quack_endpoint:
        raise OperatorError("Quack server endpoint differs from scheduler")
    if str(payload.get("schema_revision") or "") != program.schema_revision:
        raise OperatorError("Quack schema revision differs from scheduler")
    if not str(payload.get("schema_fingerprint") or "").startswith("sha256:"):
        raise OperatorError("Quack server schema fingerprint is absent")
    if payload.get("secret_handle") != program.endpoint_secret_handle:
        raise OperatorError("Quack server secret handle differs from scheduler")
    return payload


@contextmanager
def _scoped_environment(updates: Mapping[str, str]):
    prior = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            os.environ[str(name)] = str(value)
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _scoped_launch_environment(updates: Mapping[str, str]):
    """Replace, rather than merge, the complete canonical route environment.

    This prevents an inherited partial/stale authorization tuple from changing
    a configured quota-only route or making the runner search for a foreign
    artifact.  A route authorization is present only when the canonical route
    resolver returned its exact validated environment.
    """

    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        SCHEDULER_PROVIDER_ENV_NAMES,
    )

    names = {*SCHEDULER_PROVIDER_ENV_NAMES, STATE_TOKEN_ENV, *updates}
    prior = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        for name, value in updates.items():
            os.environ[str(name)] = str(value)
        yield
    finally:
        for name in names:
            os.environ.pop(name, None)
        for name, value in prior.items():
            if value is not None:
                os.environ[name] = value


def _state_owner_environment() -> dict[str, str]:
    """Return a minimal non-credential environment for the owner child."""

    permitted = {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LOGNAME",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "TZ",
        "USER",
        "VIRTUAL_ENV",
        "XDG_CACHE_HOME",
    }
    result = {
        name: value
        for name, value in os.environ.items()
        if name in permitted or name.startswith("DUCKDB_")
    }
    result.pop(STATE_TOKEN_ENV, None)
    result.setdefault("PATH", os.defpath)
    return result


def _operator_command(config_path: Path, command: str) -> list[str]:
    try:
        relative = config_path.resolve(strict=False).relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise OperatorError("operator config must remain inside repository") from exc
    argv = [sys.executable, str(Path(__file__).resolve()), "--config", relative, command]
    lowered = " ".join(argv).lower()
    if "token=" in lowered or STATE_TOKEN_ENV.lower() in lowered:
        raise OperatorError("state-owner argv would expose token material")
    return argv


def _route_preflight(board: Any) -> dict[str, Any]:
    """Resolve the exact configured route through the canonical router.

    The operator never manufactures reviewer authority.  A route which admits
    authentication-unavailable fallback must name an already produced,
    committed authorization artifact accepted by the canonical loader.  The
    sealed CASF bootstrap route is quota-only, so its complete six-field tuple
    intentionally carries no authorization artifact.
    """

    from ipfs_accelerate_py.agent_implementation_route import (
        load_agent_implementation_route_authorization,
        resolve_agent_implementation_route,
    )
    provider = board.payload.get("provider")
    if not isinstance(provider, Mapping):
        raise OperatorError("scheduler provider route is absent")
    field_names = (
        "primary_provider_id",
        "primary_model_id",
        "fallback_provider_id",
        "fallback_model_id",
        "fallback_trigger",
        "fallback_reasoning_effort",
    )
    values = {name: str(provider.get(name) or "").strip() for name in field_names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise OperatorError(
            "provider route is not a complete six-field tuple: " + ", ".join(missing)
        )
    authorization = None
    authorization_path = str(provider.get("route_authorization_path") or "").strip()
    if authorization_path:
        try:
            authorization = load_agent_implementation_route_authorization(
                repo_root=board.repo_root,
                artifact_path=authorization_path,
                board_namespace=board.board_namespace,
            )
        except (OSError, ValueError) as exc:
            raise OperatorError(
                "canonical provider route authorization is unavailable or stale"
            ) from exc
    try:
        route = resolve_agent_implementation_route(
            **values,
            authorization=authorization,
        )
    except ValueError as exc:
        raise OperatorError("canonical provider route resolver rejected the tuple") from exc
    if route.permits_authentication_unavailable and authorization is None:
        raise OperatorError(
            "authentication-unavailable fallback requires canonical reviewer authority"
        )
    if authorization is not None and route.authorization is not authorization:
        raise OperatorError("canonical provider route dropped its authorization binding")
    environment = route.as_environment()
    return {
        **route.as_dict(),
        "route_id": route.route_id,
        "authorization_required": route.permits_authentication_unavailable,
        "authorization_present": authorization is not None,
        "authorization_id": (authorization.authorization_id if authorization is not None else ""),
        "authorization_path": (authorization.artifact_path if authorization is not None else ""),
        "environment": dict(environment),
        "canonical_route_resolver_passed": True,
        "operator_created_authority": False,
        "provider_execution_admitted": False,
    }


def _launch_plan(board: Any, *, stamp: str | None = None) -> dict[str, Any]:
    """Render the credential-free native first-tranche coordinator plan."""

    route = _route_preflight(board)
    public = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "causal-federation-native-launch-plan@1"
        ),
        "board_namespace": BOARD_NAMESPACE,
        "stamp": str(stamp or "owner-generated-at-launch"),
        "runtime": "CASFEventSupervisorRuntime@1",
        "lanes": 1,
        "admitted_lanes": 1,
        "registered_logical_subagents": 1,
        "maximum_active_subagents": 0,
        "strict_task_sharding": True,
        "work_stealing": False,
        "credential_transport": "private_inherited_pipe",
        "credential_in_argv": False,
        "credential_in_environment": False,
        "state_transport": "typed_quack_state_owner",
        "server_owned_event_wait": True,
        "event_wait_qualified": True,
        "task_execution_admitted": False,
        "execution_scope": "first_tranche_event_coordination_only",
        "event_driven_federation_qualified": False,
        "high_concurrency_qualified": False,
        "multi_supervisor_qualified": False,
        "parallel_execution_qualified": False,
        "provider_route_preflight": route,
    }
    public["operator_plan_cid"] = _identity(public)
    return public


def _task_projection(source: Any) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    aliases: dict[str, str] = {}
    cursor = ""
    seen = 0
    while True:
        page = source.list_tasks(cursor=cursor, limit=100)
        for task in page.tasks:
            alias = str(task.task_alias)
            status = str(task.status or "").strip().lower()
            if alias in aliases:
                raise OperatorError("typed task query returned a duplicate task alias")
            aliases[alias] = status
            statuses[status] += 1
            seen += 1
            if seen > 44:
                raise OperatorError("typed task query exceeded sealed population")
        if not page.next_cursor:
            break
        cursor = page.next_cursor
    snapshot = source.snapshot().to_dict()
    ready = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
    return {
        "available": True,
        "transport": "quack",
        "snapshot": snapshot,
        "status_counts": dict(sorted(statuses.items())),
        "task_statuses": dict(sorted(aliases.items())),
        "ready_task_ids": ready,
        "ready_count": len(ready),
        "active_count": sum(statuses.get(item, 0) for item in ACTIVE_STATUSES),
        "blocked_count": int(statuses.get("blocked", 0)),
        "completed_count": sum(statuses.get(item, 0) for item in COMPLETED_STATUSES),
        "terminal_count": sum(statuses.get(item, 0) for item in TERMINAL_STATUSES),
        "task_count": seen,
        "event_cursor": int(snapshot.get("event_cursor") or 0),
        "projection_cid": str(snapshot.get("projection_cid") or ""),
    }


def _query_quack_tasks(board: Any, paths: Mapping[str, Path]) -> dict[str, Any]:
    from datetime import datetime, timezone

    from ipfs_accelerate_py.agent_supervisor.federation.registry import (
        FederationStateRepository,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
        QuackStateClient,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        STATUS_BOOTSTRAP_CLIENT_ID,
        TypedStateOwnerConnection,
    )

    utc = timezone.utc  # noqa: UP017 - Python 3.8 compatibility.

    program = board.resolved_database_program()
    token = _read_owner_token(_token_path(paths["owner"], program.endpoint_secret_handle))
    store_id = _control_plane_store_id(program)
    process_birth_id = f"birth:casf-status:{os.getpid()}:{time.time_ns()}"

    def connection_factory(_endpoint: Any) -> TypedStateOwnerConnection:
        return TypedStateOwnerConnection(
            socket_path=paths["owner_socket"],
            token=token,
            client_id=STATUS_BOOTSTRAP_CLIENT_ID,
            process_birth_id=process_birth_id,
            store_id=store_id,
            timeout_seconds=30.0,
            status_bootstrap=True,
        )

    client = QuackStateClient(
        owner_id=STATUS_BOOTSTRAP_CLIENT_ID,
        store_id=store_id,
        process_birth_id=process_birth_id,
        connection_factory=connection_factory,
    )
    try:
        client.attach(program.quack_endpoint)
        # Install and seal the same trusted named-operation catalog used by
        # the owner.  This does not open the database or expose caller SQL.
        FederationStateRepository(client, require_quack_authority=True)

        statuses: Counter[str] = Counter()
        aliases: dict[str, str] = {}
        cursor = 0
        seen = 0
        while True:
            page = client.paginate(cursor=cursor, limit=100)
            for task in page.items:
                alias = str(task.get("task_alias") or "").strip()
                status = str(task.get("status") or "").strip().lower()
                if not alias or alias in aliases:
                    raise OperatorError(
                        "typed task query returned an invalid or duplicate task alias"
                    )
                aliases[alias] = status
                statuses[status] += 1
                seen += 1
                if seen > 44:
                    raise OperatorError("typed task query exceeded sealed population")
            if page.exhausted:
                break
            if page.next_cursor is None or page.next_cursor <= cursor:
                raise OperatorError("typed task cursor failed to advance")
            cursor = page.next_cursor

        ready_rows = client.execute("list_ready_task_aliases")
        ready = [str(row.get("task_alias") or "") for row in ready_rows]
        if any(not item for item in ready) or len(set(ready)) != len(ready):
            raise OperatorError("typed ready-task query returned invalid identities")
        count_rows = client.execute("count_tasks")
        event_rows = client.execute("max_event_watermark")
        task_count = int(count_rows[0].get("task_count") or 0) if count_rows else 0
        event_cursor = (
            int(event_rows[0].get("event_watermark") or 0) if event_rows else 0
        )
        if task_count != seen:
            raise OperatorError("typed task count differs from bounded page projection")
        generation = client.load_generation()
        runtime_health: dict[str, Any] = {
            "available": False,
            "reason_code": "runtime_projection_or_acknowledgement_absent",
        }
        supervisor_status = _read_optional_json(paths["supervisor_status"])
        scope_fields = (
            "tenant_id",
            "federation_id",
            "supervisor_id",
            "subscription_id",
            "consumer_id",
            "first_event_id",
            "first_acknowledgement_id",
            "first_delivery_attempt_id",
        )
        scope = {
            name: str(supervisor_status.get(name) or "").strip()
            for name in scope_fields
        }
        if all(scope.values()):
            observed_at = datetime.now(utc).isoformat().replace(
                "+00:00", "Z"
            )
            rows = client.execute(
                "casf_select_supervisor_bootstrap_health",
                {
                    "tenant_id": scope["tenant_id"],
                    "federation_id": scope["federation_id"],
                    "supervisor_id": scope["supervisor_id"],
                    "subscription_id": scope["subscription_id"],
                    "consumer_id": scope["consumer_id"],
                    "event_id": scope["first_event_id"],
                    "acknowledgement_id": scope["first_acknowledgement_id"],
                    "delivery_attempt_id": scope["first_delivery_attempt_id"],
                    "observed_at": observed_at,
                },
            )
            if len(rows) > 1:
                raise OperatorError(
                    "typed runtime health query returned ambiguous authority"
                )
            if rows:
                row = rows[0]
                process_bound = False
                try:
                    birth = _process_birth(int(row["process_id"]))
                    process_bound = bool(
                        int(row["process_id"]) == int(supervisor_status["supervisor_pid"])
                        and int(row["process_start_time_ticks"])
                        == int(birth["start_time_ticks"])
                        and str(row["process_boot_id"]) == str(birth["boot_id"])
                        and int(row["process_parent_id"]) == int(birth["parent_pid"])
                        and _birth_liveness(birth) == "alive"
                    )
                except (KeyError, TypeError, ValueError, OperatorError):
                    process_bound = False
                acknowledged_sequence = int(
                    row.get("acknowledged_global_sequence") or 0
                )
                cursor_sequence = int(row.get("cursor_global_sequence") or 0)
                pending = int(row.get("pending_required_deliveries") or 0)
                event_type = str(row.get("acknowledged_event_type") or "")
                runtime_health = {
                    "available": True,
                    "tenant_id": scope["tenant_id"],
                    "federation_id": scope["federation_id"],
                    "supervisor_id": scope["supervisor_id"],
                    "subscription_id": scope["subscription_id"],
                    "consumer_id": scope["consumer_id"],
                    "lifecycle_state": str(row.get("lifecycle_state") or ""),
                    "runtime_lease_id": str(row.get("runtime_lease_id") or ""),
                    "runtime_revision": int(row.get("runtime_revision") or 0),
                    "runtime_expires_at": str(row.get("runtime_expires_at") or ""),
                    "current_runtime_lease": True,
                    "process_bound": process_bound,
                    "acknowledgement_id": str(
                        row.get("acknowledgement_id") or ""
                    ),
                    "acknowledged_event_id": str(
                        row.get("acknowledged_event_id") or ""
                    ),
                    "delivery_attempt_id": str(
                        row.get("delivery_attempt_id") or ""
                    ),
                    "acknowledged_event_type": event_type,
                    "acknowledged_global_sequence": acknowledged_sequence,
                    "cursor_global_sequence": cursor_sequence,
                    "cursor_revision": int(row.get("cursor_revision") or 0),
                    "pending_required_deliveries": pending,
                    "bootstrap_event_acknowledged": bool(
                        acknowledged_sequence > 0
                        and event_type
                        in {"SUPERVISOR_HEALTH_CHANGED", "CAPABILITY_CHANGED"}
                        and row.get("delivery_attempt_status") == "acknowledged"
                        and row.get("delivery_queue_status") == "acknowledged"
                    ),
                    "consumer_cursor_advanced": bool(
                        cursor_sequence >= acknowledged_sequence > 0
                        and int(row.get("cursor_revision") or 0) >= 2
                    ),
                }
        projection_cid = _identity(
            {
                "store_id": generation.store_id,
                "generation": generation.generation,
                "revision": generation.revision,
                "event_cursor": event_cursor,
                "task_statuses": dict(sorted(aliases.items())),
            }
        )
        snapshot = {
            "schema": "TypedQuackTaskSnapshot@1",
            "task_count": task_count,
            "event_cursor": event_cursor,
            "store_generation": generation.to_dict(),
            "projection_cid": projection_cid,
        }
        return {
            "available": True,
            "transport": "typed_quack_state_owner",
            "snapshot": snapshot,
            "status_counts": dict(sorted(statuses.items())),
            "task_statuses": dict(sorted(aliases.items())),
            "ready_task_ids": ready,
            "ready_count": len(ready),
            "active_count": sum(statuses.get(item, 0) for item in ACTIVE_STATUSES),
            "blocked_count": int(statuses.get("blocked", 0)),
            "completed_count": sum(
                statuses.get(item, 0) for item in COMPLETED_STATUSES
            ),
            "terminal_count": sum(
                statuses.get(item, 0) for item in TERMINAL_STATUSES
            ),
            "task_count": task_count,
            "event_cursor": event_cursor,
            "projection_cid": projection_cid,
            "runtime_health": runtime_health,
        }
    except OperatorError:
        raise
    except Exception as exc:
        # Do not include the transport exception text: some drivers echo the
        # authenticated ATTACH statement on failure.
        raise OperatorError(f"typed Quack task query failed ({type(exc).__name__})") from exc
    finally:
        client.close()


def _wait_for_owner(
    board: Any,
    paths: Mapping[str, Path],
    process: subprocess.Popen[Any],
    *,
    deadline: float,
    not_before_ns: int,
) -> dict[str, Any]:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise OperatorError("Quack state-owner exited before readiness")
        try:
            metadata = os.lstat(paths["owner_status"])
        except FileNotFoundError:
            time.sleep(0.1)
            continue
        if metadata.st_mtime_ns < not_before_ns:
            time.sleep(0.1)
            continue
        status = _read_optional_json(paths["owner_status"])
        try:
            identity = _owner_identity(board, status, expected_pid=process.pid)
        except OperatorError:
            time.sleep(0.1)
            continue
        if _outbox_worker_health(status)["healthy"] is not True:
            time.sleep(0.1)
            continue
        supervisor_status = _read_optional_json(paths["supervisor_status"])
        supervisor_pid = _read_pid(paths["supervisor_pid"])
        if (
            supervisor_pid is None
            or int(supervisor_status.get("supervisor_pid") or 0) != supervisor_pid
            or supervisor_status.get("lifecycle_state") != "IDLE"
            or supervisor_status.get("event_wait_qualified") is not True
            or int(supervisor_status.get("events_processed") or 0) < 1
            or not supervisor_status.get("first_event_id")
            or not supervisor_status.get("first_acknowledgement_id")
            or not supervisor_status.get("first_delivery_attempt_id")
        ):
            time.sleep(0.1)
            continue
        try:
            supervisor_birth = _process_birth(supervisor_pid)
        except OperatorError:
            time.sleep(0.1)
            continue
        return {
            "status": status,
            "identity": identity,
            "supervisor_process_birth": supervisor_birth,
        }
    raise OperatorError("timed out waiting for exact Quack owner readiness")


def _retire_consumed_generation(paths: Mapping[str, Path], *, launch_id: str) -> None:
    """Archive a fully stopped generation so the next launch can rematerialize.

    The consumed supervisor identity stays terminal.  A later launch mints a
    fresh identity against the current tree rather than reopening a stale
    control plane.
    """

    runtime = paths.get("runtime")
    database = paths.get("database")
    if runtime is None or database is None:
        return
    compact = launch_id.replace("sha256:", "")[:16] or "unknown"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = runtime / "quarantine" / f"consumed-{compact}-{stamp}"
    destination.mkdir(parents=True, exist_ok=True)
    control = destination / "control-plane"
    control.mkdir(exist_ok=True)
    parent = database.parent
    for extra in list(parent.glob(f"{database.name}*")) + list(parent.glob(f".{database.name}*")):
        extra.rename(control / extra.name)
    for key, label in (("owner", "quack-owner"), ("state", "state")):
        source = paths.get(key)
        if source is not None and source.exists():
            source.rename(destination / label)
    evidence = paths.get("operator_evidence")
    if evidence is not None and evidence.parent.exists():
        evidence.parent.rename(destination / "evidence")


def _require_unused_launch_generation(paths: Mapping[str, Path]) -> None:
    """Reject reuse of a supervisor identity whose lifecycle is terminal.

    A stopped supervisor cannot be revived by replaying its deterministic
    registration result.  After a complete matching stop, CASF-029 admits a
    later launch only by minting a fresh identity.  Fail before starting a
    process rather than discovering a stale lifecycle inside the child.
    """

    if not paths["launch_receipt"].is_file():
        return
    launch = _json_object(paths["launch_receipt"])
    _verify_receipt(launch, kind="launch")
    if launch.get("schema") != LAUNCH_SCHEMA or launch.get("program_id") != PROGRAM_ID:
        raise OperatorError("prior launch receipt has stale authority")
    launch_id = str(launch.get("launch_receipt_id") or "")
    if not launch_id:
        raise OperatorError("prior launch receipt has no valid identity")
    if not paths["stop_receipt"].is_file():
        raise OperatorError("a prior launch has no complete matching stop receipt")
    stop = _json_object(paths["stop_receipt"])
    _verify_receipt(stop, kind="stop")
    if stop.get("schema") != STOP_SCHEMA or stop.get("program_id") != PROGRAM_ID:
        raise OperatorError("prior stop receipt has stale authority")
    if stop.get("complete") is not True or stop.get("launch_receipt_id") != launch_id:
        raise OperatorError("a prior launch has no complete matching stop receipt")
    # The consumed identity stays terminal.  Retire the stale control plane so
    # launch can rematerialize and mint a fresh supervisor/process-birth/lease.
    _retire_consumed_generation(paths, launch_id=launch_id)


def _launch_owner(
    config_path: Path,
    paths: Mapping[str, Path],
    *,
    timeout_seconds: float,
    admit_task_execution: bool = False,
) -> tuple[subprocess.Popen[Any], dict[str, Any]]:
    paths["owner"].mkdir(parents=True, exist_ok=True)
    log_handle = paths["owner_log"].open("ab")
    os.chmod(paths["owner_log"], 0o600)
    not_before_ns = time.time_ns()
    argv = _operator_command(config_path, "state-owner")
    if admit_task_execution:
        argv.append("--admit-task-execution")
    try:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=_state_owner_environment(),
            start_new_session=True,
        )
    finally:
        log_handle.close()
    try:
        ready = _wait_for_owner(
            _load_config(config_path)[0],
            paths,
            process,
            deadline=time.monotonic() + timeout_seconds,
            not_before_ns=not_before_ns,
        )
    except BaseException:
        try:
            birth = _process_birth(process.pid)
            _terminate_birth(birth, grace_seconds=5.0)
        except Exception:
            pass
        raise
    return process, ready


def _terminate_birth(birth_payload: Mapping[str, Any], *, grace_seconds: float = 15.0) -> str:
    """Signal only the exact captured process birth; fail closed on UNKNOWN."""

    liveness = _birth_liveness(birth_payload)
    if liveness == "dead":
        return "already_dead"
    if liveness != "alive":
        raise OperatorError("process identity is uninspectable; refusing broad stop")
    pid = int(birth_payload.get("pid") or 0)
    if pid <= 1:
        raise OperatorError("refusing to signal an unsafe PID")

    def send(sig: int) -> None:
        if _birth_liveness(birth_payload) != "alive":
            return
        try:
            group = os.getpgid(pid)
        except (OSError, ProcessLookupError):
            return
        try:
            if group == pid:
                os.killpg(group, sig)
            else:
                os.kill(pid, sig)
        except ProcessLookupError:
            return

    send(signal.SIGTERM)
    deadline = time.monotonic() + max(0.1, grace_seconds)
    while time.monotonic() < deadline:
        state = _birth_liveness(birth_payload)
        if state == "dead":
            return "terminated"
        if state == "unknown":
            raise OperatorError("process became uninspectable during stop")
        time.sleep(0.1)
    send(signal.SIGKILL)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        state = _birth_liveness(birth_payload)
        if state == "dead":
            return "killed"
        if state == "unknown":
            raise OperatorError("process became uninspectable after SIGKILL")
        time.sleep(0.1)
    raise OperatorError("exact process birth remained alive after bounded stop")


def _runtime_projection(
    paths: Mapping[str, Path],
    *,
    launched_at_ns: int,
    expected_supervisor_birth: Mapping[str, Any],
) -> dict[str, Any]:
    supervisor = _read_optional_json(paths["supervisor_status"])
    task_path = paths["task_state"]
    candidate = str(
        supervisor.get("current_status_path")
        or supervisor.get("progress_path")
        or supervisor.get("state_path")
        or ""
    ).strip()
    if candidate:
        raw = Path(candidate)
        candidate_path = raw if raw.is_absolute() else ROOT / raw
        candidate_path = candidate_path.resolve(strict=False)
        try:
            candidate_path.relative_to(paths["state"])
        except ValueError as exc:
            raise OperatorError("supervisor task-state projection escapes state root") from exc
        task_path = candidate_path
    task_state = _read_optional_json(task_path)
    supervisor_projection = {
        key: supervisor[key]
        for key in (
            "schema",
            "status",
            "updated_at",
            "supervisor_pid",
            "supervisor_pid_alive",
            "runtime_process_birth_id",
            "tenant_id",
            "federation_id",
            "supervisor_id",
            "subscription_id",
            "consumer_id",
            "fencing_epoch",
            "daemon_pid",
            "daemon_pid_alive",
            "active_worker_count",
            "stalled_without_active_worker",
            "backpressure",
            "backpressure_reasons",
            "lifecycle_state",
            "lifecycle_revision",
            "server_owned_event_wait",
            "event_wait_transport",
            "event_wait_adaptive_polling",
            "event_wait_qualified",
            "event_cursor",
            "events_processed",
            "wait_calls",
            "heartbeat_count",
            "last_batch_size",
            "last_event_id",
            "last_acknowledgement_id",
            "last_delivery_attempt_id",
            "first_event_id",
            "first_acknowledgement_id",
            "first_delivery_attempt_id",
            "idle_task_board_scans",
            "idle_model_calls",
            "idle_context_rebuilds",
            "idle_activity_counter_source",
            "task_execution_admitted",
            "execution_scope",
            "registered_logical_subagents",
            "active_subagent_processes",
            "error_class",
            "last_exit_code",
            "last_recycle_reason",
        )
        if key in supervisor
    }
    task_projection = {
        key: task_state[key]
        for key in (
            "schema",
            "task_count",
            "completed_count",
            "eligible_ready_count",
            "blocked_count",
            "external_reserved_count",
            "active_task_id",
            "implementation_in_progress",
            "event_cursor",
            "projection_cid",
            "task_execution_admitted",
            "source",
        )
        if key in task_state
    }
    now_ns = time.time_ns()
    stale_seconds = SUPERVISOR_HEALTH_STALE_SECONDS
    supervisor_age: float | None = None
    task_age: float | None = None
    supervisor_after_launch = False
    task_after_launch = False
    try:
        metadata = os.lstat(paths["supervisor_status"])
        supervisor_age = max(0.0, (now_ns - metadata.st_mtime_ns) / 1_000_000_000)
        supervisor_after_launch = metadata.st_mtime_ns >= launched_at_ns
    except FileNotFoundError:
        pass
    try:
        metadata = os.lstat(task_path)
        task_age = max(0.0, (now_ns - metadata.st_mtime_ns) / 1_000_000_000)
        task_after_launch = metadata.st_mtime_ns >= launched_at_ns
    except FileNotFoundError:
        pass
    process_bound = False
    observed_process_birth: dict[str, Any] = {}
    expected_process_birth_id = ""
    try:
        from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
            process_birth_id,
        )
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            ProcessBirthIdentity,
        )

        expected_identity = ProcessBirthIdentity.from_dict(
            dict(expected_supervisor_birth)
        )
        expected_process_birth_id = process_birth_id(expected_identity)
        reported_pid = int(supervisor_projection.get("supervisor_pid") or 0)
        if reported_pid > 1:
            observed_process_birth = _process_birth(reported_pid)
            observed_identity = ProcessBirthIdentity.from_dict(observed_process_birth)
            process_bound = bool(
                observed_identity == expected_identity
                and process_birth_id(observed_identity) == expected_process_birth_id
                and supervisor_projection.get("runtime_process_birth_id")
                == expected_process_birth_id
                and _birth_liveness(expected_supervisor_birth) == "alive"
            )
    except (OperatorError, TypeError, ValueError):
        process_bound = False
    return {
        "supervisor_status": supervisor_projection,
        "task_state": task_projection,
        "task_state_path": str(task_path),
        "supervisor_age_seconds": supervisor_age,
        "task_state_age_seconds": task_age,
        "supervisor_fresh": bool(
            supervisor_projection and supervisor_age is not None and supervisor_age <= stale_seconds
        ),
        "task_state_fresh": bool(
            task_projection and task_age is not None and task_age <= stale_seconds
        ),
        "supervisor_after_launch": supervisor_after_launch,
        "task_state_after_launch": task_after_launch,
        "supervisor_process_bound": process_bound,
        "expected_process_birth_id": expected_process_birth_id,
        "observed_process_birth": observed_process_birth,
        "freshness_ceiling_seconds": stale_seconds,
    }


def classify_health(
    *,
    owner_liveness: str,
    master_liveness: str,
    task_authority: Mapping[str, Any],
    runtime: Mapping[str, Any],
    baseline: Mapping[str, Any],
    within_startup_grace: bool,
) -> dict[str, Any]:
    """Classify progress conservatively from typed and process-bound evidence."""

    reasons: list[str] = []
    progress: list[str] = []
    safe_idle: list[str] = []
    task_state = runtime.get("task_state")
    task_state = task_state if isinstance(task_state, Mapping) else {}
    supervisor = runtime.get("supervisor_status")
    supervisor = supervisor if isinstance(supervisor, Mapping) else {}
    runtime_authority = task_authority.get("runtime_health")
    runtime_authority = (
        runtime_authority if isinstance(runtime_authority, Mapping) else {}
    )
    outbox_worker = runtime.get("outbox_worker")
    outbox_worker = outbox_worker if isinstance(outbox_worker, Mapping) else {}
    outbox_worker_healthy = bool(
        outbox_worker.get("healthy") is True
        and outbox_worker.get("available") is True
        and outbox_worker.get("thread_alive") is True
        and outbox_worker.get("server_owned") is True
        and outbox_worker.get("polling") is False
        and outbox_worker.get("caught_up") is True
        and int(outbox_worker.get("watermark") or 0)
        >= int(task_authority.get("event_cursor") or 0)
        and outbox_worker.get("commit_observer_bound") is True
        and not str(outbox_worker.get("observer_error_type") or "").strip()
        and not str(outbox_worker.get("last_error_type") or "").strip()
    )
    authority_available = task_authority.get("available") is True
    task_count = int(task_authority.get("task_count") or 0)
    ready_count = int(task_authority.get("ready_count") or 0)
    active_count = int(task_authority.get("active_count") or 0)
    blocked_count = int(task_authority.get("blocked_count") or 0)
    completed_count = int(task_authority.get("completed_count") or 0)
    terminal_count = int(task_authority.get("terminal_count") or 0)
    state_blocked = int(task_state.get("blocked_count") or 0)
    state_ready = int(task_state.get("eligible_ready_count") or 0)
    state_active = bool(
        str(task_state.get("active_task_id") or "").strip()
        or task_state.get("implementation_in_progress") is True
    )
    external_reserved = int(task_state.get("external_reserved_count") or 0)
    state_task_count = int(task_state.get("task_count") or 0)
    state_completed = int(task_state.get("completed_count") or 0)
    supervisor_state = str(supervisor.get("status") or "").strip().lower()

    if owner_liveness != "alive" or not authority_available:
        reasons.append("authoritative_quack_unavailable")
        return {
            "classification": "unavailable",
            "healthy": False,
            "blocked_or_stuck": True,
            "reason_codes": reasons,
            "progress_evidence": progress,
            "safe_idle_evidence": safe_idle,
        }
    if blocked_count or state_blocked:
        reasons.append("durable_blocked_work_present")
        return {
            "classification": "blocked",
            "healthy": False,
            "blocked_or_stuck": True,
            "reason_codes": reasons,
            "progress_evidence": progress,
            "safe_idle_evidence": safe_idle,
        }
    if not outbox_worker_healthy:
        reasons.append("state_owner_outbox_worker_unavailable")
        return {
            "classification": "stuck",
            "healthy": False,
            "blocked_or_stuck": True,
            "reason_codes": reasons,
            "progress_evidence": progress,
            "safe_idle_evidence": safe_idle,
        }

    coordinator_transport_evidence_valid = bool(
        master_liveness == "alive"
        and runtime.get("supervisor_process_bound") is True
        and supervisor_state == "idle"
        and supervisor.get("execution_scope")
        == "first_tranche_event_coordination_only"
        and supervisor.get("task_execution_admitted") is False
        and supervisor.get("server_owned_event_wait") is True
        and supervisor.get("event_wait_qualified") is True
        and supervisor.get("event_wait_adaptive_polling") is False
        and runtime_authority.get("available") is True
        and runtime_authority.get("lifecycle_state") == "IDLE"
        and runtime_authority.get("current_runtime_lease") is True
        and runtime_authority.get("process_bound") is True
        and runtime_authority.get("bootstrap_event_acknowledged") is True
        and runtime_authority.get("consumer_cursor_advanced") is True
        and int(runtime_authority.get("pending_required_deliveries") or 0) == 0
        and supervisor.get("first_event_id")
        == runtime_authority.get("acknowledged_event_id")
        and supervisor.get("first_acknowledgement_id")
        == runtime_authority.get("acknowledgement_id")
        and supervisor.get("first_delivery_attempt_id")
        == runtime_authority.get("delivery_attempt_id")
        and runtime.get("supervisor_fresh") is True
        and runtime.get("task_state_fresh") is True
        and runtime.get("supervisor_after_launch") is True
        and runtime.get("task_state_after_launch") is True
        and not state_active
        and state_blocked == 0
        and external_reserved == 0
    )
    terminal_quiescent = bool(
        coordinator_transport_evidence_valid
        and task_count > 0
        and completed_count == task_count
        and terminal_count == task_count
        and ready_count == 0
        and active_count == 0
        and state_task_count == task_count
        and state_completed == task_count
        and state_ready == 0
    )
    if terminal_quiescent:
        # Task status rows are not the CASF-030 fixed-point/completion receipt.
        # Keep the coordinator transport observation usable, but never promote
        # this first-tranche projection to safe plan completion.
        reasons.append("fixed_point_completion_receipt_unavailable")
        return {
            "classification": "completion_unqualified",
            "healthy": False,
            "plan_work_healthy": False,
            "plan_work_blocked": True,
            "plan_execution_status": "unadmitted",
            "coordinator_ready": True,
            "coordinator_transport_healthy": True,
            "coordinator_blocked_or_stuck": False,
            "blocked_or_stuck": True,
            "reason_codes": reasons,
            "progress_evidence": progress,
            "safe_idle_evidence": safe_idle,
            "coordinator_evidence": [
                "exact_process_birth_and_current_runtime_lease",
                "bootstrap_event_durably_acknowledged",
                "state_owner_outbox_worker_live_and_caught_up",
                "fresh_post_launch_task_projection_quiescent",
            ],
        }

    if master_liveness != "alive":
        reasons.append("runner_process_not_alive_with_nonterminal_work")
        return {
            "classification": "stuck",
            "healthy": False,
            "blocked_or_stuck": True,
            "reason_codes": reasons,
            "progress_evidence": progress,
            "safe_idle_evidence": safe_idle,
        }
    if runtime.get("supervisor_fresh") is not True:
        if within_startup_grace and not supervisor:
            reasons.append("awaiting_first_supervisor_heartbeat")
            classification = "starting"
            blocked_or_stuck = False
        else:
            reasons.append("supervisor_heartbeat_missing_or_stale")
            classification = "stuck"
            blocked_or_stuck = True
        return {
            "classification": classification,
            "healthy": False,
            "blocked_or_stuck": blocked_or_stuck,
            "reason_codes": reasons,
            "progress_evidence": progress,
            "safe_idle_evidence": safe_idle,
        }
    if supervisor_state in {"failed", "quarantined", "stopped"}:
        reasons.append(f"supervisor_lifecycle_{supervisor_state}")
        return {
            "classification": "stuck",
            "healthy": False,
            "blocked_or_stuck": True,
            "reason_codes": reasons,
            "progress_evidence": progress,
            "safe_idle_evidence": safe_idle,
        }

    coordinator_transport_ready = coordinator_transport_evidence_valid
    if coordinator_transport_ready:
        coordinator_evidence = [
            "exact_process_birth_and_current_runtime_lease",
            "bootstrap_event_durably_acknowledged",
            "state_owner_outbox_worker_live",
            "typed_server_owned_event_wait_qualified",
        ]
        if task_count > terminal_count:
            reasons.append(
                "ready_work_present_but_task_execution_unadmitted"
                if ready_count or state_ready
                else "nonterminal_plan_work_is_not_execution_admitted"
            )
        else:
            reasons.append("coordinator_transport_ready_without_safe_plan_fixed_point")
        return {
            "classification": "coordinator_ready",
            "healthy": False,
            "plan_work_healthy": False,
            "plan_work_blocked": True,
            "plan_execution_status": "unadmitted",
            "coordinator_ready": True,
            "coordinator_transport_healthy": True,
            "coordinator_blocked_or_stuck": False,
            "blocked_or_stuck": True,
            "reason_codes": reasons,
            "progress_evidence": progress,
            "safe_idle_evidence": safe_idle,
            "coordinator_evidence": coordinator_evidence,
        }

    if (
        supervisor.get("execution_scope")
        == "first_tranche_event_coordination_only"
        and supervisor.get("task_execution_admitted") is False
    ):
        reasons.append("authoritative_bootstrap_event_or_runtime_evidence_incomplete")
        return {
            "classification": "starting" if within_startup_grace else "stuck",
            "healthy": False,
            "plan_work_healthy": False,
            "coordinator_ready": False,
            "blocked_or_stuck": not within_startup_grace,
            "reason_codes": reasons,
            "progress_evidence": progress,
            "safe_idle_evidence": safe_idle,
            "coordinator_evidence": [],
        }

    baseline_cursor = int(baseline.get("event_cursor") or 0)
    baseline_completed = int(baseline.get("completed_count") or 0)
    if int(task_authority.get("event_cursor") or 0) > baseline_cursor:
        progress.append("authoritative_event_cursor_advanced")
    if completed_count > baseline_completed:
        progress.append("authoritative_completed_count_advanced")
    if active_count or state_active:
        progress.append("active_task_or_attempt_observed")
    if (
        (ready_count or state_ready)
        and runtime.get("supervisor_after_launch") is True
        and runtime.get("task_state_after_launch") is True
        and runtime.get("task_state_fresh") is True
    ):
        progress.append("fresh_supervisor_cycle_observed_ready_work")
    if progress:
        reasons.append("plan_execution_profile_unqualified_in_tranche_1")
        return {
            "classification": "progress_unqualified",
            "healthy": False,
            "plan_work_healthy": False,
            "plan_work_blocked": True,
            "plan_execution_status": "unadmitted",
            "coordinator_ready": False,
            "blocked_or_stuck": True,
            "reason_codes": reasons,
            "progress_evidence": progress,
            "safe_idle_evidence": safe_idle,
        }
    if within_startup_grace:
        reasons.append("live_supervisor_has_not_yet_published_progress_evidence")
        classification = "starting"
        blocked_or_stuck = False
    else:
        reasons.append("nonterminal_work_has_no_fresh_progress_evidence")
        classification = "stuck"
        blocked_or_stuck = True
    return {
        "classification": classification,
        "healthy": False,
        "blocked_or_stuck": blocked_or_stuck,
        "reason_codes": reasons,
        "progress_evidence": progress,
        "safe_idle_evidence": safe_idle,
    }


def _verify_receipt(payload: Mapping[str, Any], *, kind: str) -> None:
    field = f"{kind}_receipt_id"
    expected = str(payload.get(field) or "")
    body = {key: value for key, value in payload.items() if key != field}
    if not expected or expected != _identity(body):
        raise OperatorError(f"{kind} receipt identity is absent or invalid")


def _status_snapshot(config_path: Path, *, persist: bool = True) -> dict[str, Any]:
    board, config = _load_config(config_path)
    paths = _runtime_paths(board)
    launch = _json_object(paths["launch_receipt"])
    _verify_receipt(launch, kind="launch")
    if launch.get("schema") != LAUNCH_SCHEMA or launch.get("program_id") != PROGRAM_ID:
        raise OperatorError("launch receipt has stale authority")
    owner_status = _read_optional_json(paths["owner_status"])
    owner_live = _owner_liveness(owner_status)
    owner_identity: dict[str, Any] = {}
    if owner_live == "alive":
        owner_identity = _owner_identity(board, owner_status)
        expected_owner = launch.get("owner_identity")
        if not isinstance(expected_owner, Mapping) or (
            owner_identity.get("process_birth") != expected_owner.get("process_birth")
            or owner_identity.get("database_uuid") != expected_owner.get("database_uuid")
            or owner_identity.get("generation") != expected_owner.get("generation")
        ):
            raise OperatorError("live owner identity differs from launch receipt")
    master_birth = launch.get("master_process_birth")
    if not isinstance(master_birth, Mapping):
        raise OperatorError("launch receipt lacks master process-birth identity")
    master_live = _birth_liveness(master_birth)
    authority: dict[str, Any]
    if owner_live == "alive":
        try:
            authority = _query_quack_tasks(board, paths)
        except OperatorError as exc:
            authority = {
                "available": False,
                "reason_code": "typed_quack_query_failed",
                "error_class": type(exc).__name__,
            }
    else:
        authority = {
            "available": False,
            "reason_code": "quack_owner_not_alive",
        }
    launched_at_ns = int(launch.get("launched_at_ns") or 0)
    runtime = _runtime_projection(
        paths,
        launched_at_ns=launched_at_ns,
        expected_supervisor_birth=master_birth,
    )
    outbox_worker = _outbox_worker_health(owner_status)
    runtime["outbox_worker"] = outbox_worker
    startup_grace = float(config.get("watchdog_startup_grace_seconds") or 300)
    within_grace = time.time_ns() <= launched_at_ns + int(startup_grace * 1e9)
    baseline = launch.get("initial_task_authority")
    baseline = baseline if isinstance(baseline, Mapping) else {}
    classification = classify_health(
        owner_liveness=owner_live,
        master_liveness=master_live,
        task_authority=authority,
        runtime=runtime,
        baseline=baseline,
        within_startup_grace=within_grace,
    )
    classification.setdefault("coordinator_ready", False)
    classification.setdefault("coordinator_evidence", [])
    classification.setdefault("coordinator_transport_healthy", False)
    classification.setdefault("coordinator_blocked_or_stuck", True)
    classification.setdefault(
        "plan_work_healthy",
        bool(
            classification.get("healthy") is True
            and classification.get("classification")
            in {"progressing", "safely_idle"}
        ),
    )
    classification.setdefault(
        "plan_work_blocked", classification.get("plan_work_healthy") is not True
    )
    classification.setdefault(
        "plan_execution_status",
        "admitted"
        if classification.get("plan_work_healthy") is True
        else "unadmitted",
    )
    payload: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "program_id": PROGRAM_ID,
        "observed_at_ns": time.time_ns(),
        "launch_receipt_id": launch["launch_receipt_id"],
        "source_head_at_launch": launch["source_head"],
        "repository_tree_at_launch": launch["repository_tree_id"],
        "owner_liveness": owner_live,
        "owner_identity": owner_identity,
        "master_liveness": master_live,
        "master_process_birth": dict(master_birth),
        "task_authority": authority,
        "runtime": runtime,
        "outbox_worker": outbox_worker,
        **classification,
        "event_wait_qualified": bool(
            isinstance(runtime.get("supervisor_status"), Mapping)
            and runtime["supervisor_status"].get("event_wait_qualified") is True
        ),
        "event_driven_qualified": False,
        "multi_supervisor_qualified": False,
        "parallel_execution_qualified": False,
        "high_concurrency_qualified": False,
        "ducklake_authoritative": False,
    }
    return _persist_receipt(paths, "status", payload) if persist else payload


def status(config_path: Path) -> dict[str, Any]:
    return _status_snapshot(config_path, persist=True)


def _launch_success_mode(
    status_receipt: Mapping[str, Any],
    *,
    allow_coordinator_only: bool,
    admit_task_execution: bool = False,
) -> str:
    """Return the one exact launch acceptance class, or an empty denial."""

    if (
        allow_coordinator_only
        and status_receipt.get("classification")
        in {"coordinator_ready", "completion_unqualified"}
        and status_receipt.get("coordinator_ready") is True
        and status_receipt.get("coordinator_transport_healthy") is True
        and status_receipt.get("coordinator_blocked_or_stuck") is False
        and status_receipt.get("healthy") is False
        and status_receipt.get("plan_work_healthy") is False
        and status_receipt.get("blocked_or_stuck") is True
    ):
        return "coordinator_transport_only"
    if (
        admit_task_execution
        and status_receipt.get("classification") == "progress_unqualified"
        and status_receipt.get("healthy") is False
    ):
        return "task_execution_admitted"
    return ""


def _cleanup_failed_launch(
    owner_birth: Mapping[str, Any] | None,
    master_birth: Mapping[str, Any] | None,
) -> None:
    for birth in (master_birth, owner_birth):
        if isinstance(birth, Mapping):
            try:
                _terminate_birth(birth, grace_seconds=10.0)
            except Exception:
                pass


def launch(
    config_path: Path,
    *,
    owner_timeout_seconds: float = 45.0,
    health_timeout_seconds: float = 120.0,
    allow_coordinator_only: bool = False,
    admit_task_execution: bool = False,
) -> dict[str, Any]:
    """Materialize, start one Quack owner, and admit one event coordinator."""

    if owner_timeout_seconds <= 0 or health_timeout_seconds <= 0:
        raise OperatorError("launch timeouts must be positive")
    board, config = _load_config(config_path)
    paths = _runtime_paths(board)
    _require_unused_launch_generation(paths)
    existing_status = _read_optional_json(paths["owner_status"])
    existing_live = _owner_liveness(existing_status)
    if existing_live in {"alive", "unknown"}:
        raise OperatorError("an existing or uninspectable Quack owner blocks launch")
    if existing_live == "dead" and not paths["launch_receipt"].is_file():
        # A previous owner died before a launch receipt existed.  That
        # control plane cannot be revived; mint a fresh generation.
        _retire_consumed_generation(paths, launch_id="failed-pre-launch")
    preflight(config_path, require_free_port=True)
    materialized = materialize(config_path)
    bootstrap = materialized["bootstrap_receipt"]
    if not isinstance(bootstrap, Mapping):
        raise OperatorError("materializer returned no bootstrap receipt")
    plan = _launch_plan(board)
    owner_process: subprocess.Popen[Any] | None = None
    owner_birth: dict[str, Any] | None = None
    master_birth: dict[str, Any] | None = None
    launched_at_ns = time.time_ns()
    try:
        owner_process, owner_ready = _launch_owner(
            config_path,
            paths,
            timeout_seconds=owner_timeout_seconds,
            admit_task_execution=admit_task_execution,
        )
        identity = owner_ready["identity"]
        if not isinstance(identity, Mapping):
            raise OperatorError("owner launch returned no typed identity")
        owner_birth = dict(identity["process_birth"])
        task_authority = _query_quack_tasks(board, paths)
        outbox_worker = _outbox_worker_health(
            _read_optional_json(paths["owner_status"])
        )
        if outbox_worker["healthy"] is not True:
            raise OperatorError("state-owner outbox worker is not live")
        projection = config.get("initial_projection")
        projection = projection if isinstance(projection, Mapping) else {}
        expected_ready = [str(item) for item in projection.get("ready_task_ids", ())]
        expected_completed = [str(item) for item in projection.get("completed_task_ids", ())]
        observed_completed = sorted(
            task_id
            for task_id, task_status in task_authority["task_statuses"].items()
            if task_status in COMPLETED_STATUSES
        )
        if (
            task_authority["task_count"] != int(projection.get("task_count") or 0)
            or task_authority["ready_task_ids"] != expected_ready
            or observed_completed != sorted(expected_completed)
        ):
            raise OperatorError("live Quack task population differs from sealed projection")
        runtime_health = task_authority.get("runtime_health")
        if not isinstance(runtime_health, Mapping) or not (
            runtime_health.get("available") is True
            and runtime_health.get("current_runtime_lease") is True
            and runtime_health.get("process_bound") is True
            and runtime_health.get("bootstrap_event_acknowledged") is True
            and runtime_health.get("consumer_cursor_advanced") is True
            and int(runtime_health.get("pending_required_deliveries") or 0) == 0
        ):
            raise OperatorError(
                "live coordinator lacks authoritative runtime/event acknowledgement evidence"
            )
        sealed_supervisor_birth = owner_ready.get("supervisor_process_birth")
        if not isinstance(sealed_supervisor_birth, Mapping):
            raise OperatorError("state owner did not attest the supervisor process birth")
        master_birth = dict(sealed_supervisor_birth)
        if _birth_liveness(master_birth) != "alive":
            raise OperatorError("state-owner-attested supervisor process is not alive")
        launch_receipt = _persist_receipt(
            paths,
            "launch",
            {
                "schema": LAUNCH_SCHEMA,
                "program_id": PROGRAM_ID,
                "launched_at_ns": launched_at_ns,
                "source_head": bootstrap["source_head"],
                "repository_tree_id": bootstrap["repository_tree_id"],
                "plan_root_cid": bootstrap["plan_root_cid"],
                "bootstrap_receipt_id": bootstrap["bootstrap_receipt_id"],
                "owner_identity": dict(identity),
                "master_process_birth": master_birth,
                "supervisor_process_birth": master_birth,
                "initial_task_authority": task_authority,
                "bootstrap_event_acknowledgement": dict(runtime_health),
                "outbox_worker_health": outbox_worker,
                "native_runtime_plan": plan,
                "one_lane_admitted": True,
                "one_coordinator_admitted": True,
                "registered_logical_subagents": 1,
                "active_subagent_processes": 0,
                "credential_transport": "private_inherited_pipe",
                "credential_in_argv_or_environment": False,
                "task_execution_admitted": bool(admit_task_execution),
                "relaunch_supported": True,
                "relaunch_blocker": "",
                "event_wait_qualified": True,
                "event_driven_qualified": False,
                "multi_supervisor_qualified": False,
                "parallel_execution_qualified": False,
                "high_concurrency_qualified": False,
                "ducklake_authoritative": False,
            },
        )
        deadline = time.monotonic() + health_timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = _status_snapshot(config_path, persist=True)
            success_mode = _launch_success_mode(
                last,
                allow_coordinator_only=allow_coordinator_only or admit_task_execution,
                admit_task_execution=admit_task_execution,
            )
            if success_mode in {"coordinator_transport_only", "task_execution_admitted"}:
                coordinator_only = success_mode == "coordinator_transport_only"
                coordinator_receipt = _persist_receipt(
                    paths,
                    "coordinator",
                    {
                        "schema": COORDINATOR_READY_SCHEMA,
                        "program_id": PROGRAM_ID,
                        "observed_at_ns": int(last.get("observed_at_ns") or 0),
                        "launch_receipt_id": launch_receipt["launch_receipt_id"],
                        "status_receipt_id": last["status_receipt_id"],
                        "launch_mode": success_mode,
                        "coordinator_ready": bool(
                            last.get("coordinator_ready") is True or coordinator_only
                        ),
                        "coordinator_transport_healthy": bool(
                            last.get("coordinator_transport_healthy") is True
                            or coordinator_only
                        ),
                        "coordinator_blocked_or_stuck": False,
                        "coordinator_transport_only": coordinator_only,
                        "plan_work_healthy": bool(admit_task_execution),
                        "plan_work_blocked": not admit_task_execution,
                        "plan_execution_status": (
                            "admitted" if admit_task_execution else "unadmitted"
                        ),
                        "task_execution_admitted": bool(admit_task_execution),
                        "capability_blocker": (
                            ""
                            if admit_task_execution
                            else "plan_task_execution_unadmitted"
                        ),
                        "coordinator_evidence": list(
                            last.get("coordinator_evidence") or ()
                        ),
                    },
                )
                return {
                    "schema": OPERATOR_SCHEMA,
                    "command": "launch",
                    "ok": True,
                    "launch_mode": success_mode,
                    "coordinator_transport_only": coordinator_only,
                    "coordinator_transport_healthy": True,
                    "coordinator_blocked_or_stuck": False,
                    "plan_work_healthy": bool(admit_task_execution),
                    "plan_work_blocked": not admit_task_execution,
                    "plan_execution_status": (
                        "admitted" if admit_task_execution else "unadmitted"
                    ),
                    "task_execution_admitted": bool(admit_task_execution),
                    "launch_receipt": launch_receipt,
                    "coordinator_receipt": coordinator_receipt,
                    "status_receipt": last,
                }
            if last.get("blocked_or_stuck") is True:
                break
            time.sleep(0.5)
        stop_result = stop(config_path)
        raise OperatorError(
            "launched supervisor did not prove progress or safe idle; "
            f"classification={last.get('classification', 'unknown')}; "
            f"stop_complete={stop_result.get('complete') is True}"
        )
    except BaseException:
        # Once a launch receipt exists, ``stop`` is the authoritative cleanup.
        if not paths["launch_receipt"].is_file():
            _cleanup_failed_launch(owner_birth, master_birth)
        raise


def stop(config_path: Path) -> dict[str, Any]:
    """Drain the runner, then stop the exact Quack owner process birth."""

    board, _config = _load_config(config_path)
    paths = _runtime_paths(board)
    launch_receipt = _json_object(paths["launch_receipt"])
    _verify_receipt(launch_receipt, kind="launch")
    if (
        launch_receipt.get("schema") != LAUNCH_SCHEMA
        or launch_receipt.get("program_id") != PROGRAM_ID
    ):
        raise OperatorError("launch receipt has stale authority")
    master_birth = launch_receipt.get("master_process_birth")
    supervisor_birth = launch_receipt.get("supervisor_process_birth")
    owner_identity = launch_receipt.get("owner_identity")
    owner_birth = (
        owner_identity.get("process_birth") if isinstance(owner_identity, Mapping) else None
    )
    if (
        not isinstance(master_birth, Mapping)
        or not isinstance(supervisor_birth, Mapping)
        or not isinstance(owner_birth, Mapping)
    ):
        raise OperatorError("launch receipt lacks exact process-birth identities")
    if dict(supervisor_birth) != dict(master_birth):
        raise OperatorError("launch receipt has conflicting supervisor process births")
    current_owner = _read_optional_json(paths["owner_status"])
    if _owner_liveness(current_owner) == "alive":
        current_identity = _owner_identity(board, current_owner)
        if current_identity.get("process_birth") != owner_birth:
            raise OperatorError("refusing to stop a Quack owner from another launch")
    results: list[dict[str, Any]] = []
    results.append(
        {
            "role": "master",
            "birth": dict(master_birth),
            "result": _terminate_birth(master_birth, grace_seconds=30.0),
        }
    )
    # The event coordinator is a session leader.  Its exact sealed birth is
    # sufficient to retire that process group.  Persistent PID/status files
    # are observations only and are never upgraded into signal authority.
    results.append(
        {
            "role": "state_owner",
            "birth": dict(owner_birth),
            "result": _terminate_birth(owner_birth, grace_seconds=15.0),
        }
    )
    executor_pid = _read_pid(paths["executor_pid"]) if "executor_pid" in paths else None
    if executor_pid is not None:
        try:
            executor_birth = _process_birth(executor_pid)
        except OperatorError:
            executor_birth = None
        if isinstance(executor_birth, Mapping):
            results.append(
                {
                    "role": "plan_executor",
                    "birth": dict(executor_birth),
                    "result": _terminate_birth(executor_birth, grace_seconds=15.0),
                }
            )
    final_births = [dict(master_birth), dict(owner_birth)]
    final_liveness = [_birth_liveness(item) for item in final_births]
    program = board.resolved_database_program()
    endpoint = QUACK_ENDPOINT_RE.fullmatch(program.quack_endpoint)
    assert endpoint is not None
    token_destroyed = not _token_path(paths["owner"], program.endpoint_secret_handle).exists()
    host, port = endpoint.group(1), int(endpoint.group(2))
    endpoint_released = _port_is_free(host, port)
    deadline = time.monotonic() + 5.0
    while not endpoint_released and time.monotonic() < deadline:
        time.sleep(0.2)
        endpoint_released = _port_is_free(host, port)
    complete = bool(
        all(item == "dead" for item in final_liveness) and token_destroyed and endpoint_released
    )
    payload = _persist_receipt(
        paths,
        "stop",
        {
            "schema": STOP_SCHEMA,
            "program_id": PROGRAM_ID,
            "stopped_at_ns": time.time_ns(),
            "launch_receipt_id": launch_receipt["launch_receipt_id"],
            "process_results": results,
            "final_liveness": final_liveness,
            "token_vault_destroyed": token_destroyed,
            "endpoint_released": endpoint_released,
            "complete": complete,
        },
    )
    if not complete:
        raise OperatorError("bounded stop did not prove every process and secret retired")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="repository-relative sealed configured-board JSON",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="verify Quack capability and free endpoint")
    commands.add_parser("materialize", help="materialize sealed goals/tasks into DuckDB")
    commands.add_parser("plan", help="render the bounded native coordinator plan")
    runtime_parser = commands.add_parser("supervisor-runtime", help=argparse.SUPPRESS)
    runtime_parser.add_argument("--credential-fd", type=int, required=True)
    launch_parser = commands.add_parser(
        "launch", help="start the owner and one bounded event coordinator"
    )
    launch_parser.add_argument("--owner-timeout-seconds", type=float, default=45.0)
    launch_parser.add_argument("--health-timeout-seconds", type=float, default=120.0)
    launch_parser.add_argument(
        "--allow-coordinator-only",
        action="store_true",
        help=(
            "leave an exactly qualified transport coordinator running even "
            "when plan-task execution is explicitly unadmitted"
        ),
    )
    launch_parser.add_argument(
        "--admit-task-execution",
        action="store_true",
        help=(
            "after the event coordinator is IDLE, start one isolated plan "
            "executor against the live Quack owner so remaining CASF tasks "
            "can run without clobbering coordinator status"
        ),
    )
    state_owner_parser = commands.add_parser("state-owner", help=argparse.SUPPRESS)
    state_owner_parser.add_argument(
        "--admit-task-execution",
        action="store_true",
    )
    executor_parser = commands.add_parser("plan-executor", help=argparse.SUPPRESS)
    executor_parser.add_argument("--credential-fd", type=int, required=True)
    status_parser = commands.add_parser("status", help="emit typed progress/idle status")
    status_requirement = status_parser.add_mutually_exclusive_group()
    status_requirement.add_argument("--require-healthy", action="store_true")
    status_requirement.add_argument(
        "--require-coordinator-ready",
        action="store_true",
        help="require only the distinct coordinator transport qualification",
    )
    commands.add_parser("stop", help="stop exact runner/owner process births")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        if arguments.command == "preflight":
            result = preflight(config_path)
        elif arguments.command == "materialize":
            result = materialize(config_path)
        elif arguments.command == "plan":
            board, _config = _load_config(config_path)
            result = {
                "schema": OPERATOR_SCHEMA,
                "command": "plan",
                "ok": True,
                "launch_plan": _launch_plan(board),
            }
        elif arguments.command == "state-owner":
            return state_owner(
                config_path,
                admit_task_execution=bool(
                    getattr(arguments, "admit_task_execution", False)
                ),
            )
        elif arguments.command == "supervisor-runtime":
            from ipfs_accelerate_py.agent_supervisor.federation.supervisor_runtime import (
                run_supervisor_runtime,
            )

            return run_supervisor_runtime(arguments.credential_fd)
        elif arguments.command == "plan-executor":
            return plan_executor(config_path, arguments.credential_fd)
        elif arguments.command == "launch":
            result = launch(
                config_path,
                owner_timeout_seconds=arguments.owner_timeout_seconds,
                health_timeout_seconds=arguments.health_timeout_seconds,
                allow_coordinator_only=arguments.allow_coordinator_only,
                admit_task_execution=arguments.admit_task_execution,
            )
        elif arguments.command == "status":
            result = status(config_path)
            print(json.dumps(result, indent=2, sort_keys=True))
            if arguments.require_healthy:
                return 0 if result.get("healthy") is True else 1
            if arguments.require_coordinator_ready:
                return 0 if result.get("coordinator_ready") is True else 1
            return 0
        elif arguments.command == "stop":
            result = stop(config_path)
        else:
            raise OperatorError(f"unsupported command: {arguments.command}")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except OperatorError as exc:
        print(
            json.dumps(
                {
                    "schema": OPERATOR_SCHEMA,
                    "command": str(arguments.command),
                    "ok": False,
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                    "event_driven_qualified": False,
                    "high_concurrency_qualified": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
