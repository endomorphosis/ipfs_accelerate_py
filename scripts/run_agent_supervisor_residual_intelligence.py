#!/usr/bin/env python3
"""Bootstrap and operate the VRIF DuckDB + Quack control plane.

The authority split is deliberately narrow:

* ``DatabaseTaskSource@1`` over DuckDB is transactional task/goal authority.
* one fenced loopback Quack process exclusively owns the DuckDB file while
  supervisors are running;
* DuckLake is an optional, rebuildable history projection and is never read by
  readiness, completion, promotion, or release gates.

The Markdown plan, objectives, and task board are immutable bootstrap inputs.
This operator never mutates their status fields and never publishes the raw
Quack authentication token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG: Final = Path("config/agent_supervisor_residual_intelligence_scheduler.json")
RUNTIME_RELATIVE: Final = Path("data/agent_supervisor/residual_intelligence_foundry")
BOOTSTRAP_RECEIPT_NAME: Final = "bootstrap-materialization.json"
DUCKLAKE_RECEIPT_NAME: Final = "ducklake-history-projection.json"
OPERATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/verified-residual-intelligence-foundry-operator@1"
)
POPULATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/verified-residual-intelligence-foundry-population@1"
)
BOOTSTRAP_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/verified-residual-intelligence-foundry-bootstrap@1"
)
DUCKLAKE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-residual-intelligence-foundry-ducklake-projection@1"
)
OWNER_RESTART_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-residual-intelligence-foundry-owner-restart-admission@1"
)
OWNER_RESTART_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-residual-intelligence-foundry-owner-restart-receipt@1"
)
OWNER_DATABASE_VERIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-residual-intelligence-foundry-owner-database-verification@1"
)
SUPERVISOR_LAUNCH_ACK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-residual-intelligence-foundry-supervisor-launch-ack@1"
)
DATABASE_TASK_SOURCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/database-task-source@1"
)
OWNER_RESTART_ALLOWED_SOURCE_FIELDS: Final = frozenset(
    {
        "accelerator_required_ancestor",
        "accelerator_planning_revision",
        "accelerator_planning_tree",
    }
)
GOAL_RE: Final = re.compile(r"^## (VRIF-G\d{3}) (.+)$", re.MULTILINE)
QUACK_ENDPOINT_RE: Final = re.compile(
    r"^quack:(?://)?(127(?:\.\d{1,3}){3}|localhost):(\d{1,5})$",
    re.IGNORECASE,
)
READY_STATUSES: Final = (
    "proposed",
    "admitted",
    "pending",
    "ready",
    "todo",
    "queued",
    "retrying",
)
COMPLETED_STATUSES: Final = ("completed", "skipped", "complete", "done")
ACTIVE_STATUSES: Final = ("claimed", "in_progress", "running")
TERMINAL_STATUSES: Final = (
    *COMPLETED_STATUSES,
    "cancelled",
    "failed",
    "quarantined",
    "rejected",
)
OWNER_COMMAND_ENVELOPE_MAX_BYTES: Final = 1_048_576
TYPED_DEFERRAL_PROVIDER_CANARY_MAX_BYTES: Final = 8 * 1024 * 1024
TYPED_DEFERRAL_PROVIDER_CANARY_TIMEOUT_SECONDS: Final = 600
TYPED_DEFERRAL_RECOVERY_PRODUCTION_PATHS: Final = frozenset(
    {
        "ipfs_accelerate_py/agent_implementation_route.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
        (
            "ipfs_accelerate_py/agent_supervisor/task_sources/"
            "database_task_source.py"
        ),
        (
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "implementation_daemon.py"
        ),
        (
            "ipfs_accelerate_py/agent_supervisor/validation/"
            "project_dependency_preflight.py"
        ),
    }
)


class OperatorError(RuntimeError):
    """Fail-closed VRIF operator error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
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


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise OperatorError(f"JSON root must be an object: {path}")
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
        raise OperatorError(f"{field} escapes repository") from exc
    return resolved


def _git(*arguments: str, check: bool = True, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if check and completed.returncode != 0:
        error = completed.stderr or completed.stdout
        if isinstance(error, bytes):
            error = error.decode("utf-8", errors="replace")
        raise OperatorError(f"git {' '.join(arguments)} failed: {str(error).strip()}")
    return completed.stdout


def _assert_clean_current_tree(config: Mapping[str, Any]) -> tuple[str, str]:
    status_output = str(_git("status", "--porcelain=v1", "--untracked-files=all")).strip()
    if status_output:
        raise OperatorError(
            "refusing to materialize from a dirty worktree; commit the exact "
            "plan, board, configuration, validator, and operator first"
        )
    head = str(_git("rev-parse", "HEAD")).strip()
    tree = str(_git("rev-parse", "HEAD^{tree}")).strip()
    branch = str(_git("branch", "--show-current")).strip()
    required_branch = str(config.get("merge_target_branch") or "").strip()
    if required_branch and branch != required_branch:
        raise OperatorError(
            f"execution branch {branch!r} differs from configured branch {required_branch!r}"
        )
    binding = config.get("source_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    ancestor = str(binding.get("accelerator_required_ancestor") or "").strip()
    if ancestor:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise OperatorError("configured accelerator base is not an ancestor")
    return head, tree


def _tracked_bytes(path: Path, *, head: str) -> bytes:
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise OperatorError(f"authority input escapes repository: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise OperatorError(f"authority input is not a regular file: {relative}")
    working = path.read_bytes()
    recorded = _git("show", f"{head}:{relative}", binary=True)
    if not isinstance(recorded, bytes) or working != recorded:
        raise OperatorError(f"authority input differs from current HEAD: {relative}")
    return working


def _git_commit_tree(commit: Any, *, field: str) -> str:
    revision = str(commit or "").strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise OperatorError(f"{field} must be an exact Git commit")
    try:
        object_type = str(_git("cat-file", "-t", revision)).strip()
        tree = str(_git("rev-parse", f"{revision}^{{tree}}")).strip()
    except OperatorError as exc:
        raise OperatorError(f"{field} is not available in the repository") from exc
    if object_type != "commit" or re.fullmatch(r"[0-9a-f]{40}", tree) is None:
        raise OperatorError(f"{field} is not an exact Git commit")
    return tree


def _git_is_ancestor(ancestor: str, descendant: str, *, field: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return
    if completed.returncode == 1:
        raise OperatorError(f"{field} is not monotonic")
    raise OperatorError(f"cannot verify {field}")


def _git_blob_at(*, head: str, path: Path, field: str) -> bytes:
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise OperatorError(f"{field} escapes repository") from exc
    try:
        value = _git("show", f"{head}:{relative}", binary=True)
    except OperatorError as exc:
        raise OperatorError(f"{field} is absent from the bootstrap source") from exc
    if not isinstance(value, bytes):
        raise OperatorError(f"{field} could not be read as bytes")
    return value


def _json_mapping_bytes(value: bytes, *, field: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorError(f"{field} must be a JSON object") from exc
    if not isinstance(decoded, dict):
        raise OperatorError(f"{field} must be a JSON object")
    return decoded


def _restart_source_binding(
    config: Mapping[str, Any],
    *,
    label: str,
    source_head: str,
) -> dict[str, str]:
    raw = config.get("source_binding")
    if not isinstance(raw, Mapping):
        raise OperatorError(f"{label} source_binding must be an object")
    values = {
        field: str(raw.get(field) or "").strip()
        for field in OWNER_RESTART_ALLOWED_SOURCE_FIELDS
    }
    planning_revision = values["accelerator_planning_revision"]
    required_ancestor = values["accelerator_required_ancestor"]
    planning_tree = values["accelerator_planning_tree"]
    if required_ancestor != planning_revision:
        raise OperatorError(
            f"{label} planning revision and required ancestor must be exact"
        )
    observed_tree = _git_commit_tree(
        planning_revision,
        field=f"{label}.source_binding.accelerator_planning_revision",
    )
    if planning_tree != observed_tree:
        raise OperatorError(f"{label} planning tree does not match its commit")
    _git_is_ancestor(
        planning_revision,
        source_head,
        field=f"{label} planning revision ancestry",
    )
    return values


def _restart_static_config(config: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_bytes(config))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperatorError(f"{label} config is not canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise OperatorError(f"{label} config must be an object")
    source_binding = normalized.get("source_binding")
    if not isinstance(source_binding, dict):
        raise OperatorError(f"{label} source_binding must be an object")
    for field in OWNER_RESTART_ALLOWED_SOURCE_FIELDS:
        source_binding.pop(field, None)
    return normalized


def _owner_restart_admission(
    board: Any,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Admit only the sealed bootstrap tree or a bounded descendant restart."""

    current_head, current_tree = _assert_clean_current_tree(config)
    bootstrap = _json_object(paths["bootstrap_receipt"])
    if bootstrap.get("schema") != BOOTSTRAP_SCHEMA:
        raise OperatorError("owner restart bootstrap schema is not admitted")
    bootstrap_receipt_id = str(bootstrap.get("bootstrap_receipt_id") or "").strip()
    bootstrap_body = dict(bootstrap)
    bootstrap_body.pop("bootstrap_receipt_id", None)
    if (
        re.fullmatch(r"sha256:[0-9a-f]{64}", bootstrap_receipt_id) is None
        or _identity(bootstrap_body) != bootstrap_receipt_id
    ):
        raise OperatorError("owner restart bootstrap receipt identity is invalid")

    bootstrap_head = str(bootstrap.get("source_head") or "").strip()
    bootstrap_tree = str(bootstrap.get("repository_tree_id") or "").strip()
    observed_bootstrap_tree = _git_commit_tree(
        bootstrap_head,
        field="bootstrap source_head",
    )
    if bootstrap_tree != observed_bootstrap_tree:
        raise OperatorError("bootstrap source tree does not match its commit")
    _git_is_ancestor(
        bootstrap_head,
        current_head,
        field="bootstrap-to-current source ancestry",
    )

    plan_root_cid = str(bootstrap.get("plan_root_cid") or "").strip()
    database_receipt = bootstrap.get("database_task_source_receipt")
    if not isinstance(database_receipt, Mapping):
        raise OperatorError("bootstrap database task-source receipt is absent")
    if str(database_receipt.get("schema") or "") != DATABASE_TASK_SOURCE_SCHEMA:
        raise OperatorError("bootstrap database task-source schema is not admitted")
    if (
        str(database_receipt.get("repository_tree_id") or "") != bootstrap_tree
        or str(database_receipt.get("plan_root_cid") or "") != plan_root_cid
    ):
        raise OperatorError("bootstrap database authority roots are inconsistent")
    task_cids_raw = database_receipt.get("task_cids")
    if (
        not isinstance(task_cids_raw, Sequence)
        or isinstance(task_cids_raw, (str, bytes, bytearray))
    ):
        raise OperatorError("bootstrap database task identities are absent")
    task_cids = tuple(str(item or "").strip() for item in task_cids_raw)
    if not task_cids or any(not item for item in task_cids):
        raise OperatorError("bootstrap database task identities are invalid")
    if len(set(task_cids)) != len(task_cids):
        raise OperatorError("bootstrap database task identities are not unique")
    try:
        task_count = int(database_receipt.get("task_count"))
        goal_count = int(database_receipt.get("goal_count"))
        plan_count = int(database_receipt.get("plan_count"))
    except (TypeError, ValueError) as exc:
        raise OperatorError("bootstrap database authority counts are invalid") from exc
    if task_count != len(task_cids) or goal_count < 1 or plan_count < 1:
        raise OperatorError("bootstrap database authority counts are inconsistent")

    source_identities = bootstrap.get("source_identities")
    if not isinstance(source_identities, Mapping):
        raise OperatorError("bootstrap source identities are absent")
    source_paths = {
        "config": board.config_path,
        "taskboard": board.path(board.taskboard_path),
        "objectives": board.path(board.objectives_path),
        "plan": board.path(board.plan_path),
        "validator": board.path(board.validator_path),
    }
    if set(source_identities) != set(source_paths):
        raise OperatorError("bootstrap source identity key set is not exact")
    bootstrap_sources: dict[str, bytes] = {}
    current_sources: dict[str, bytes] = {}
    for name, path in source_paths.items():
        expected = str(source_identities.get(name) or "").strip()
        if re.fullmatch(r"sha256:[0-9a-f]{64}", expected) is None:
            raise OperatorError(f"bootstrap {name} source identity is invalid")
        bootstrap_bytes = _git_blob_at(
            head=bootstrap_head,
            path=path,
            field=f"bootstrap {name}",
        )
        if _identity(bootstrap_bytes) != expected:
            raise OperatorError(f"bootstrap {name} bytes differ from their seal")
        current_bytes = _tracked_bytes(path, head=current_head)
        if name != "config" and _identity(current_bytes) != expected:
            raise OperatorError(f"current {name} bytes differ from bootstrap")
        bootstrap_sources[name] = bootstrap_bytes
        current_sources[name] = current_bytes

    bootstrap_config = _json_mapping_bytes(
        bootstrap_sources["config"],
        field="bootstrap config",
    )
    current_config = _json_mapping_bytes(
        current_sources["config"],
        field="current config",
    )
    if _canonical_bytes(current_config) != _canonical_bytes(config):
        raise OperatorError("loaded config differs from tracked current config")
    bootstrap_binding = _restart_source_binding(
        bootstrap_config,
        label="bootstrap",
        source_head=bootstrap_head,
    )
    current_binding = _restart_source_binding(
        current_config,
        label="current",
        source_head=current_head,
    )
    _git_is_ancestor(
        bootstrap_binding["accelerator_planning_revision"],
        current_binding["accelerator_planning_revision"],
        field="planning revision lineage",
    )
    bootstrap_static = _restart_static_config(
        bootstrap_config,
        label="bootstrap",
    )
    current_static = _restart_static_config(current_config, label="current")
    if _canonical_bytes(bootstrap_static) != _canonical_bytes(current_static):
        raise OperatorError(
            "current config changes fields outside the admitted source-binding lineage"
        )

    mode = (
        "exact_bootstrap"
        if current_head == bootstrap_head and current_tree == bootstrap_tree
        else "verified_descendant"
    )
    admission: dict[str, Any] = {
        "schema": OWNER_RESTART_ADMISSION_SCHEMA,
        "mode": mode,
        "bootstrap_receipt_id": bootstrap_receipt_id,
        "bootstrap_source_head": bootstrap_head,
        "bootstrap_source_tree": bootstrap_tree,
        "current_source_head": current_head,
        "current_source_tree": current_tree,
        "plan_root_cid": plan_root_cid,
        "bootstrap_config_identity": _identity(bootstrap_sources["config"]),
        "current_config_identity": _identity(current_sources["config"]),
        "static_config_identity": _identity(bootstrap_static),
        "source_identities": {
            name: str(source_identities[name]) for name in sorted(source_paths)
        },
        "planning_lineage": {
            "bootstrap_revision": bootstrap_binding[
                "accelerator_planning_revision"
            ],
            "bootstrap_tree": bootstrap_binding["accelerator_planning_tree"],
            "current_revision": current_binding["accelerator_planning_revision"],
            "current_tree": current_binding["accelerator_planning_tree"],
        },
        "authority_config_identity": _identity(
            {
                "database_program": current_config.get("database_program"),
                "runtime_paths": current_config.get("runtime_paths"),
            }
        ),
        "database_authority": {
            "receipt_identity": _identity(database_receipt),
            "schema": DATABASE_TASK_SOURCE_SCHEMA,
            "repository_tree_id": bootstrap_tree,
            "source_head": bootstrap_head,
            "plan_root_cid": plan_root_cid,
            "projection_cid": str(database_receipt.get("projection_cid") or ""),
            "task_cids": sorted(task_cids),
            "task_count": task_count,
            "goal_count": goal_count,
            "plan_count": plan_count,
        },
    }
    admission["admission_id"] = _identity(admission)
    return admission


def _owner_restart_prior_status(path: Path) -> dict[str, Any]:
    """Admit only an absent, stopped, or provably dead prior owner."""

    try:
        observed = path.lstat()
    except FileNotFoundError:
        return {
            "state": "absent",
            "status_identity": "",
            "server_id": "",
            "database_uuid": "",
            "store_id": "",
            "schema_revision": 0,
            "schema_fingerprint": "",
            "generation": 0,
            "fence_epoch": 0,
            "process_birth_id": "",
        }
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.getuid()
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        raise OperatorError("prior state-owner status is not a private regular file")
    payload = _json_object(path)
    if (
        payload.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/quack-state-server@1"
        or payload.get("interface") != "QuackStateServer@1"
    ):
        raise OperatorError("prior state-owner status contract is not admitted")
    lifecycle = str(payload.get("lifecycle") or "").strip()
    liveness = _owner_liveness(payload)
    if lifecycle == "ready" and liveness in {"alive", "unknown"}:
        raise OperatorError(
            f"prior ready state owner has {liveness} process-birth liveness"
        )
    if lifecycle != "stopped" and liveness != "dead":
        raise OperatorError("prior state-owner status is neither stopped nor dead")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        if lifecycle != "stopped":
            raise OperatorError("prior dead state owner has no exact identity")
        identity = {}
    try:
        generation = int(identity.get("generation") or 0)
        fence_epoch = int(identity.get("fence_epoch") or 0)
        schema_revision = int(identity.get("schema_revision") or 0)
    except (TypeError, ValueError) as exc:
        raise OperatorError("prior state-owner identity counters are invalid") from exc
    if identity and (
        not str(identity.get("server_id") or "").strip()
        or not str(identity.get("database_uuid") or "").strip()
        or not str(identity.get("store_id") or "").strip()
        or generation < 1
        or fence_epoch < 1
        or schema_revision < 1
    ):
        raise OperatorError("prior state-owner identity is incomplete")
    return {
        "state": "stopped" if lifecycle == "stopped" else "dead",
        "lifecycle": lifecycle,
        "liveness": liveness,
        "status_identity": _identity(payload),
        "server_id": str(identity.get("server_id") or ""),
        "database_uuid": str(identity.get("database_uuid") or ""),
        "store_id": str(identity.get("store_id") or ""),
        "schema_revision": schema_revision,
        "schema_fingerprint": str(identity.get("schema_fingerprint") or ""),
        "generation": generation,
        "fence_epoch": fence_epoch,
        "process_birth_id": str(identity.get("process_birth_id") or ""),
        "identity": dict(identity),
    }


def _rows(connection: Any, sql: str, parameters: Sequence[Any] = ()) -> list[Any]:
    try:
        return list(connection.execute(sql, list(parameters)).fetchall())
    except Exception as exc:
        raise OperatorError("cannot verify bound state-owner database authority") from exc


def _row_item(row: Any, index: int, key: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    try:
        return row[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise OperatorError("bound database returned a malformed authority row") from exc


def _owner_database_verification(
    connection: Any,
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Reproduce the sealed immutable population through the owner connection."""

    authority = admission.get("database_authority")
    if not isinstance(authority, Mapping):
        raise OperatorError("owner restart admission has no database authority")
    expected_tasks_raw = authority.get("task_cids")
    if not isinstance(expected_tasks_raw, Sequence) or isinstance(
        expected_tasks_raw, (str, bytes, bytearray)
    ):
        raise OperatorError("owner restart task authority is malformed")
    expected_tasks = sorted(str(item) for item in expected_tasks_raw)
    task_rows = _rows(
        connection,
        "SELECT task_cid, identity_json FROM tasks ORDER BY task_cid",
    )
    actual_tasks: list[str] = []
    expected_tree = str(authority.get("repository_tree_id") or "")
    for row in task_rows:
        task_cid = str(_row_item(row, 0, "task_cid") or "")
        identity_json = str(_row_item(row, 1, "identity_json") or "")
        try:
            task_identity = json.loads(identity_json)
        except json.JSONDecodeError as exc:
            raise OperatorError("bound database task identity is not JSON") from exc
        if (
            not isinstance(task_identity, Mapping)
            or str(task_identity.get("task_cid") or "") != task_cid
            or str(task_identity.get("repository_tree_id") or "") != expected_tree
        ):
            raise OperatorError("bound database task identity differs from bootstrap")
        actual_tasks.append(task_cid)
    if actual_tasks != expected_tasks or len(actual_tasks) != int(
        authority.get("task_count") or 0
    ):
        raise OperatorError("bound database task population differs from bootstrap")

    goal_rows = _rows(connection, "SELECT goal_cid FROM goals ORDER BY goal_cid")
    if len(goal_rows) != int(authority.get("goal_count") or 0):
        raise OperatorError("bound database goal population differs from bootstrap")
    plan_rows = _rows(connection, "SELECT plan_cid, body_json FROM plans ORDER BY plan_cid")
    if len(plan_rows) != int(authority.get("plan_count") or 0):
        raise OperatorError("bound database plan population differs from bootstrap")
    expected_plan = str(authority.get("plan_root_cid") or "")
    plan_body: Mapping[str, Any] | None = None
    for row in plan_rows:
        if str(_row_item(row, 0, "plan_cid") or "") != expected_plan:
            continue
        raw_body = str(_row_item(row, 1, "body_json") or "")
        try:
            decoded = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise OperatorError("bound database plan body is not JSON") from exc
        if not isinstance(decoded, Mapping):
            raise OperatorError("bound database plan body is not an object")
        plan_body = decoded
        break
    if plan_body is None:
        raise OperatorError("bound database plan root differs from bootstrap")
    if (
        str(plan_body.get("repository_tree_id") or "") != expected_tree
        or str(plan_body.get("source_head") or "")
        != str(authority.get("source_head") or "")
    ):
        raise OperatorError("bound database plan lineage differs from bootstrap")
    projection = {
        "schema": OWNER_DATABASE_VERIFICATION_SCHEMA,
        "bootstrap_database_receipt_identity": str(
            authority.get("receipt_identity") or ""
        ),
        "repository_tree_id": expected_tree,
        "source_head": str(authority.get("source_head") or ""),
        "plan_root_cid": expected_plan,
        "task_cids": actual_tasks,
        "task_count": len(actual_tasks),
        "goal_count": len(goal_rows),
        "plan_count": len(plan_rows),
    }
    projection["verification_id"] = _identity(projection)
    return projection


def _owner_restart_receipt(
    admission: Mapping[str, Any],
    identity: Any,
    *,
    expected_store_id: str,
    prior_owner: Mapping[str, Any],
    database_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an admitted source transition to the newly live server identity."""

    store_id = str(getattr(identity, "store_id", "") or "")
    database_uuid = str(getattr(identity, "database_uuid", "") or "")
    try:
        generation = int(getattr(identity, "generation", 0) or 0)
        fence_epoch = int(getattr(identity, "fence_epoch", 0) or 0)
        schema_revision = int(getattr(identity, "schema_revision", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise OperatorError("new state-owner identity counters are invalid") from exc
    if (
        store_id != expected_store_id
        or not str(getattr(identity, "server_id", "") or "")
        or not database_uuid
        or generation < 1
        or fence_epoch < 1
        or schema_revision < 1
        or not str(getattr(identity, "schema_fingerprint", "") or "")
        or not str(getattr(identity, "process_birth_id", "") or "")
    ):
        raise OperatorError("new state-owner database identity is invalid")
    verification_id = str(database_verification.get("verification_id") or "")
    verification_body = dict(database_verification)
    verification_body.pop("verification_id", None)
    if (
        database_verification.get("schema") != OWNER_DATABASE_VERIFICATION_SCHEMA
        or re.fullmatch(r"sha256:[0-9a-f]{64}", verification_id) is None
        or _identity(verification_body) != verification_id
        or str(database_verification.get("plan_root_cid") or "")
        != str(admission.get("plan_root_cid") or "")
        or str(database_verification.get("repository_tree_id") or "")
        != str(admission.get("bootstrap_source_tree") or "")
    ):
        raise OperatorError("bound database verification is invalid")
    prior_generation = int(prior_owner.get("generation") or 0)
    prior_fence = int(prior_owner.get("fence_epoch") or 0)
    prior_server_id = str(prior_owner.get("server_id") or "")
    if prior_generation:
        if generation <= prior_generation or fence_epoch <= prior_fence:
            raise OperatorError("new state-owner generation does not advance prior owner")
        if prior_server_id == str(getattr(identity, "server_id", "") or ""):
            raise OperatorError("new state-owner server identity was reused")
        if str(prior_owner.get("store_id") or "") != store_id:
            raise OperatorError("new state-owner store differs from prior owner")
        if str(prior_owner.get("database_uuid") or "") != database_uuid:
            raise OperatorError("new state-owner database differs from prior owner")
        if int(prior_owner.get("schema_revision") or 0) != schema_revision:
            raise OperatorError("new state-owner schema differs from prior owner")
        if str(prior_owner.get("schema_fingerprint") or "") != str(
            getattr(identity, "schema_fingerprint", "") or ""
        ):
            raise OperatorError("new state-owner schema fingerprint differs from prior owner")
    elif str(admission.get("mode") or "") == "verified_descendant" and generation <= 1:
        raise OperatorError("descendant owner restart did not advance store generation")
    receipt: dict[str, Any] = {
        "schema": OWNER_RESTART_RECEIPT_SCHEMA,
        "admission_id": str(admission.get("admission_id") or ""),
        "mode": str(admission.get("mode") or ""),
        "bootstrap_receipt_id": str(admission.get("bootstrap_receipt_id") or ""),
        "bootstrap_source_head": str(
            admission.get("bootstrap_source_head") or ""
        ),
        "bootstrap_source_tree": str(
            admission.get("bootstrap_source_tree") or ""
        ),
        "current_source_head": str(admission.get("current_source_head") or ""),
        "current_source_tree": str(admission.get("current_source_tree") or ""),
        "plan_root_cid": str(admission.get("plan_root_cid") or ""),
        "authority_config_identity": str(
            admission.get("authority_config_identity") or ""
        ),
        "prior_state_owner": dict(prior_owner),
        "database_verification": dict(database_verification),
        "state_owner": {
            "server_id": str(getattr(identity, "server_id", "") or ""),
            "store_id": store_id,
            "database_uuid": database_uuid,
            "schema_revision": schema_revision,
            "schema_fingerprint": str(
                getattr(identity, "schema_fingerprint", "") or ""
            ),
            "generation": generation,
            "fence_epoch": fence_epoch,
            "process_birth_id": str(
                getattr(identity, "process_birth_id", "") or ""
            ),
        },
    }
    receipt["receipt_id"] = _identity(receipt)
    return receipt


def _source_forest(config: Mapping[str, Any], *, head: str) -> dict[str, Any]:
    """Verify configured sibling gitlinks without granting write authority."""

    binding = config.get("source_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    nested: list[dict[str, str]] = []
    configured_repositories = (
        (
            "ipfs_datasets",
            ("ipfs_datasets_submodule_path", "datasets_submodule_path"),
            ("ipfs_datasets_planning_revision", "datasets_planning_revision"),
        ),
        (
            "ipfs_kit",
            ("ipfs_kit_submodule_path", "kit_submodule_path"),
            ("ipfs_kit_planning_revision", "kit_planning_revision"),
        ),
        (
            "mcp_plus_plus",
            ("mcp_plus_plus_submodule_path",),
            ("mcp_plus_plus_planning_revision",),
        ),
    )
    for prefix, path_fields, revision_fields in configured_repositories:
        raw_path = next(
            (binding.get(field) for field in path_fields if binding.get(field)),
            None,
        )
        raw_revision = next(
            (binding.get(field) for field in revision_fields if binding.get(field)),
            None,
        )
        if raw_path in (None, "") and raw_revision in (None, ""):
            continue
        if raw_path in (None, "") or raw_revision in (None, ""):
            raise OperatorError(f"{prefix} source binding is incomplete")
        nested_path = _safe_path(
            ROOT,
            raw_path,
            field=f"source_binding.{prefix}_submodule_path",
        )
        if not nested_path.is_dir():
            raise OperatorError(f"{prefix} submodule is not initialized")
        nested_status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=nested_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if nested_status.returncode != 0 or nested_status.stdout.strip():
            raise OperatorError(f"{prefix} nested worktree is not clean")
        nested_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=nested_path,
            text=True,
            capture_output=True,
            check=False,
        )
        nested_tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=nested_path,
            text=True,
            capture_output=True,
            check=False,
        )
        revision = nested_head.stdout.strip()
        tree = nested_tree.stdout.strip()
        if (
            nested_head.returncode != 0
            or nested_tree.returncode != 0
            or revision != str(raw_revision)
            or not tree
        ):
            raise OperatorError(f"{prefix} nested revision differs from its seal")
        relative = nested_path.relative_to(ROOT).as_posix()
        tree_row = str(_git("ls-tree", head, "--", relative)).strip().split()
        if (
            len(tree_row) < 3
            or tree_row[0] != "160000"
            or tree_row[1] != "commit"
            or tree_row[2] != revision
        ):
            raise OperatorError(f"{prefix} gitlink differs from its nested HEAD")
        nested.append(
            {
                "repository": prefix,
                "path": relative,
                "head": revision,
                "tree": tree,
                "access": "read_only_contract_audit",
            }
        )
    result: dict[str, Any] = {
        "source_head": head,
        "nested_repositories": nested,
        "cross_repository_writes": False,
    }
    result["source_forest_root"] = _identity(result)
    return result


def _load_config(config_path: Path) -> tuple[Any, dict[str, Any]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        load_configured_board,
    )

    board = load_configured_board(config_path, repo_root=ROOT)
    payload = dict(board.payload)
    if board.task_prefix.removeprefix("## ") != "VRIF-":
        raise OperatorError("VRIF operator requires task_prefix='VRIF-'")
    if board.board_namespace != "agent-supervisor-verified-residual-intelligence-foundry-v1":
        raise OperatorError("scheduler board_namespace is not the VRIF v1 namespace")
    program = board.resolved_database_program()
    if program.authority_mode != "quack" or program.task_source_kind != "duckdb":
        raise OperatorError("VRIF requires DuckDB task authority served through Quack")
    if program.failover_policy != "fail_closed":
        raise OperatorError("VRIF Quack authority must fail closed")
    if QUACK_ENDPOINT_RE.fullmatch(program.quack_endpoint) is None:
        raise OperatorError("VRIF Quack endpoint must be a bounded loopback URI")
    projection = payload.get("initial_projection")
    projection = projection if isinstance(projection, Mapping) else {}
    if int(projection.get("task_count") or -1) != 33:
        raise OperatorError("VRIF v1 requires exactly 33 configured tasks")
    if int(projection.get("goal_count") or -1) != 9:
        raise OperatorError("VRIF v1 requires exactly 9 configured goals")
    return board, payload


def _split_csv(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _metadata_value(value: Any) -> str:
    """Accept plain PCAR fields and the board validator's bold field form."""

    text = str(value or "").strip()
    if text.startswith("**"):
        text = text[2:].lstrip()
    return text


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
                raise OperatorError(
                    f"{match.group(1)} contains duplicate metadata field {normalized}"
                )
            fields[normalized] = _metadata_value(value)
        result.append((match.group(1), match.group(2).strip(), fields))
    return result


def _population(board: Any, config: Mapping[str, Any]) -> dict[str, Any]:
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
    source_forest = _source_forest(config, head=head)
    sources = {
        "config": _tracked_bytes(board.config_path, head=head),
        "taskboard": _tracked_bytes(board.path(board.taskboard_path), head=head),
        "objectives": _tracked_bytes(board.path(board.objectives_path), head=head),
        "plan": _tracked_bytes(board.path(board.plan_path), head=head),
        "validator": _tracked_bytes(board.path(board.validator_path), head=head),
    }
    plan_root = content_identity(
        {
            "schema": "vrif-plan-root@1",
            "source_head": head,
            "repository_tree_id": tree,
            "sources": {name: _identity(value) for name, value in sorted(sources.items())},
        }
    )

    objective_text = sources["objectives"].decode("utf-8")
    parsed_goals = _goal_blocks(objective_text)
    if not parsed_goals or parsed_goals[0][0] != "VRIF-G000":
        raise OperatorError("objectives must begin with root VRIF-G000")
    if len({item[0] for item in parsed_goals}) != len(parsed_goals):
        raise OperatorError("objectives contain duplicate goal IDs")
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
        unknown = [item for item in dependencies if item not in goal_cids]
        if unknown:
            raise OperatorError(f"{goal_id} has unknown goal dependencies: {unknown}")
        goal = {
            "goal_cid": goal_cids[goal_id],
            "goal_id": goal_id,
            "goal_alias": goal_id,
            "title": title,
            "ordinal": ordinal,
            "status": str(fields.get("status") or "open").lower(),
            "objective_id": "objective:vrif-root" if goal_id == "VRIF-G000" else "",
            "objective_alias": "VRIF-G000",
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

    task_text = sources["taskboard"].decode("utf-8")
    parsed_tasks = parse_todo_blocks(task_text, task_header_prefix="## VRIF-")
    parsed_tasks = [
        (
            task_id,
            title,
            source_line,
            {key: _metadata_value(value) for key, value in fields.items()},
        )
        for task_id, title, source_line, fields in parsed_tasks
    ]
    if not parsed_tasks:
        raise OperatorError("task board contains no VRIF tasks")
    task_ids = [item[0] for item in parsed_tasks]
    if len(task_ids) != len(set(task_ids)):
        raise OperatorError("task board contains duplicate VRIF task IDs")
    expected_task_ids = [f"VRIF-{ordinal:03d}" for ordinal in range(33)]
    if task_ids != expected_task_ids:
        raise OperatorError("task board must contain ordered VRIF-000 through VRIF-032")
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
        unknown = [item for item in dependencies if item not in task_cids]
        if unknown:
            raise OperatorError(f"{task_id} has unknown dependencies: {unknown}")
        future = [item for item in dependencies if item not in observed_tasks]
        if future:
            raise OperatorError(
                f"{task_id} dependencies must precede it for atomic ingestion: {future}"
            )
        goal_id = str(
            fields.get("subgoal_id") or fields.get("goal_id") or fields.get("goal") or "VRIF-G000"
        ).strip()
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
                "objective_id": "objective:vrif-root",
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
                "acceptance": [
                    str(fields.get("acceptance") or fields.get("acceptance_subset") or "")
                ],
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
    expected_tasks = projection.get("task_count")
    expected_goals = projection.get("goal_count")
    expected_dependencies = projection.get("task_dependency_count")
    if expected_tasks is not None and int(expected_tasks) != len(tasks):
        raise OperatorError("task count differs from configured initial projection")
    if expected_goals is not None and int(expected_goals) != len(goals):
        raise OperatorError("goal count differs from configured initial projection")
    dependency_count = sum(len(_split_csv(item[3].get("depends_on"))) for item in parsed_tasks)
    if expected_dependencies is not None and int(expected_dependencies) != dependency_count:
        raise OperatorError("task dependency count differs from configured initial projection")
    return {
        "schema": POPULATION_SCHEMA,
        "repository_tree_id": tree,
        "source_head": head,
        "plan_root_cid": plan_root,
        "source_identities": {name: _identity(value) for name, value in sorted(sources.items())},
        "source_forest": source_forest,
        "objectives": goals,
        "goal_edges": goal_edges,
        "plans": [
            {
                "plan_cid": plan_root,
                "plan_alias": "VRIF-PLAN-V1",
                "goal_cid": goal_cids["VRIF-G000"],
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
    database = _safe_path(ROOT, program.store_id, field="database_program.store_id")
    runtime = board.path(board.runtime_paths["root"])
    try:
        database.relative_to(runtime)
    except ValueError as exc:
        raise OperatorError("DuckDB authority store must be below runtime_paths.root") from exc
    raw_runtime = board.payload.get("runtime_paths")
    raw_runtime = raw_runtime if isinstance(raw_runtime, Mapping) else {}
    evidence = _safe_path(
        ROOT,
        raw_runtime.get("evidence") or runtime.relative_to(ROOT) / "evidence",
        field="runtime_paths.evidence",
    )
    owner = _safe_path(
        ROOT,
        raw_runtime.get("quack_owner") or runtime.relative_to(ROOT) / "quack-owner",
        field="runtime_paths.quack_owner",
    )
    raw_ducklake = board.payload.get("ducklake_projection_program")
    raw_ducklake = raw_ducklake if isinstance(raw_ducklake, Mapping) else {}
    ducklake_catalog = _safe_path(
        ROOT,
        raw_ducklake.get("catalog_path")
        or runtime.relative_to(ROOT) / "ducklake" / "catalog.duckdb",
        field="ducklake_projection_program.catalog_path",
    )
    ducklake_data = _safe_path(
        ROOT,
        raw_ducklake.get("data_path") or runtime.relative_to(ROOT) / "ducklake" / "data",
        field="ducklake_projection_program.data_path",
    )
    for label, path in (
        ("evidence", evidence),
        ("quack_owner", owner),
        ("ducklake_catalog", ducklake_catalog),
        ("ducklake_data", ducklake_data),
    ):
        try:
            path.relative_to(runtime)
        except ValueError as exc:
            raise OperatorError(f"{label} must be below runtime_paths.root") from exc
    return {
        "runtime": runtime,
        "database": database,
        "owner": owner,
        "bootstrap_receipt": evidence / "bootstrap" / BOOTSTRAP_RECEIPT_NAME,
        "ducklake_receipt": evidence / "bootstrap" / DUCKLAKE_RECEIPT_NAME,
        "ducklake_catalog": ducklake_catalog,
        "ducklake_data": ducklake_data,
    }


def _ducklake_projection(
    *,
    paths: Mapping[str, Path],
    population: Mapping[str, Any],
    control_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one non-authoritative bootstrap observation to DuckLake."""

    projection: dict[str, Any] = {
        "schema": DUCKLAKE_SCHEMA,
        "authoritative": False,
        "scheduler_gate": False,
        "completion_gate": False,
        "status": "unavailable",
        "reason_code": "ducklake_projection_unavailable",
        "source_head": str(population["source_head"]),
        "repository_tree_id": str(population["repository_tree_id"]),
        "plan_root_cid": str(population["plan_root_cid"]),
    }
    try:
        import duckdb

        catalog = paths["ducklake_catalog"]
        data_path = paths["ducklake_data"]
        catalog.parent.mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)
        memory = duckdb.connect(":memory:")
        try:
            memory.execute("LOAD ducklake")
            catalog_sql = str(catalog).replace("'", "''")
            data_sql = str(data_path).replace("'", "''")
            memory.execute(
                f"ATTACH 'ducklake:{catalog_sql}' AS vrif_history (DATA_PATH '{data_sql}')"
            )
            memory.execute(
                """
                CREATE TABLE IF NOT EXISTS vrif_history.bootstrap_history (
                    event_id VARCHAR,
                    observed_at_epoch DOUBLE,
                    source_head VARCHAR,
                    repository_tree_id VARCHAR,
                    plan_root_cid VARCHAR,
                    projection_cid VARCHAR,
                    task_count BIGINT,
                    goal_count BIGINT,
                    body_json VARCHAR
                )
                """
            )
            event_id = _identity(
                {
                    "source_head": population["source_head"],
                    "plan_root_cid": population["plan_root_cid"],
                    "projection_cid": control_receipt.get("projection_cid"),
                }
            )
            existing = memory.execute(
                "SELECT COUNT(*) FROM vrif_history.bootstrap_history WHERE event_id = ?",
                [event_id],
            ).fetchone()
            if existing is None or int(existing[0]) == 0:
                memory.execute(
                    """
                    INSERT INTO vrif_history.bootstrap_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        event_id,
                        time.time(),
                        population["source_head"],
                        population["repository_tree_id"],
                        population["plan_root_cid"],
                        str(control_receipt.get("projection_cid") or ""),
                        int(control_receipt.get("task_count") or 0),
                        int(control_receipt.get("goal_count") or 0),
                        json.dumps(
                            {
                                "authority": "DuckDB/DatabaseTaskSource@1",
                                "transport": "QuackStateServer@1",
                                "projection": "DuckLake/non-authoritative",
                            },
                            sort_keys=True,
                        ),
                    ],
                )
            row_count = int(
                memory.execute("SELECT COUNT(*) FROM vrif_history.bootstrap_history").fetchone()[0]
            )
            memory.execute("DETACH vrif_history")
        finally:
            memory.close()
        projection.update(
            {
                "status": "available",
                "reason_code": "",
                "event_id": event_id,
                "row_count": row_count,
                "catalog_path": str(catalog.relative_to(ROOT)),
                "data_path": str(data_path.relative_to(ROOT)),
            }
        )
    except Exception as exc:
        # This projection is optional by contract. Preserve a typed absence and
        # never use it to reject a valid DuckDB materialization.
        projection["error_class"] = type(exc).__name__
    projection["projection_receipt_id"] = _identity(projection)
    _atomic_json(paths["ducklake_receipt"], projection)
    return projection


def _run_bootstrap_validation(*, board: Any, population: Mapping[str, Any]) -> dict[str, Any]:
    """Run the fixed hermetic qualification used for bootstrap completions."""

    commands = (
        (
            sys.executable,
            str(board.path(board.validator_path)),
            "--check-all",
            "--json",
        ),
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "test/api/residual_intelligence",
        ),
    )
    observations: list[dict[str, Any]] = []
    for argv in commands:
        completed = subprocess.run(
            argv,
            cwd=ROOT,
            capture_output=True,
            text=False,
            timeout=600,
            check=False,
        )
        observation = {
            "argv": list(argv),
            "returncode": int(completed.returncode),
            "stdout_digest": _identity(completed.stdout),
            "stderr_digest": _identity(completed.stderr),
        }
        observations.append(observation)
        if completed.returncode != 0:
            raise OperatorError("sealed bootstrap validation failed")
    receipt: dict[str, Any] = {
        "schema": "vrif-bootstrap-validation@1",
        "source_head": str(population["source_head"]),
        "repository_tree_id": str(population["repository_tree_id"]),
        "plan_root_cid": str(population["plan_root_cid"]),
        "commands": observations,
        "hermetic": True,
        "training_performed": False,
    }
    receipt["validation_digest"] = _identity(receipt)
    return receipt


def _seal_completed_tasks(
    source: Any,
    *,
    completed_aliases: Sequence[str],
    validation_receipt: Mapping[str, Any],
    population: Mapping[str, Any],
) -> list[str]:
    """Complete bootstrap tasks through evidence-gated CAS, never board status."""

    completion_receipt_cids: list[str] = []
    for alias in completed_aliases:
        task = source.get_task(alias)
        if task is None:
            raise OperatorError(f"bootstrap completion task is absent: {alias}")
        evidence_digest = _identity(
            {
                "task_cid": task.task_cid,
                "task_alias": alias,
                "validation_digest": validation_receipt["validation_digest"],
                "source_head": population["source_head"],
                "repository_tree_id": population["repository_tree_id"],
            }
        )
        source.intent.record_validation_result(
            task_cid=task.task_cid,
            outcome="passed",
            evidence_digest=evidence_digest,
            argv=["sealed-vrif-bootstrap-validation"],
            body={
                "producer": "vrif-bootstrap-operator@1",
                "validation_receipt": dict(validation_receipt),
                "task_alias": alias,
            },
        )
        completion = source.intent.cas_task_status(
            task_cid=task.task_cid,
            expected_revision=task.revision,
            new_status="completed",
            evidence_digests=[evidence_digest],
            receipt={
                "schema": "vrif-bootstrap-completion@1",
                "producer": "vrif-bootstrap-operator@1",
                "candidate_only": False,
                "model_created": False,
                "source_head": population["source_head"],
                "repository_tree_id": population["repository_tree_id"],
                "validation_digest": validation_receipt["validation_digest"],
                "task_alias": alias,
            },
        )
        receipt_cid = str(completion.details.get("completion_receipt_cid") or "")
        if not receipt_cid:
            raise OperatorError(f"completion receipt missing for {alias}")
        completion_receipt_cids.append(receipt_cid)
    return completion_receipt_cids


def materialize(config_path: Path) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    board, config = _load_config(config_path)
    paths = _runtime_paths(board)
    population = _population(board, config)
    receipt_path = paths["bootstrap_receipt"]
    if paths["database"].exists() or receipt_path.exists():
        if not paths["database"].is_file() or not receipt_path.is_file():
            raise OperatorError("partial bootstrap state exists; operator review required")
        prior = _json_object(receipt_path)
        exact = all(
            prior.get(key) == population.get(key)
            for key in ("source_head", "repository_tree_id", "plan_root_cid")
        )
        if not exact:
            raise OperatorError(
                "existing DuckDB authority is bound to a different source tree or plan"
            )
        with DatabaseTaskSource(
            paths["database"],
            owner_id="vrif-bootstrap:verify-existing",
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        ) as source:
            snapshot = source.snapshot().to_dict()
            ready_ids = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
            completion_projection = dict(source.intent.completion_evidence_projection())
        if int(snapshot["task_count"]) != len(population["tasks"]):
            raise OperatorError("existing DuckDB task population differs from sealed board")
        if int(snapshot["goal_count"]) != len(population["objectives"]):
            raise OperatorError("existing DuckDB goal population differs from sealed board")
        projection = config.get("initial_projection")
        projection = projection if isinstance(projection, Mapping) else {}
        expected_ready = [str(item) for item in projection.get("ready_task_ids", ())]
        if ready_ids != expected_ready:
            raise OperatorError("existing DuckDB readiness frontier differs from seal")
        expected_completed = {
            population["task_cids_by_alias"][str(alias)]
            for alias in projection.get("completed_task_ids", ())
        }
        receipt_tasks = {
            str(item["task_cid"]) for item in completion_projection["completion_receipts"]
        }
        if receipt_tasks != expected_completed:
            raise OperatorError("existing bootstrap completion receipts are incomplete")
        return {
            "schema": OPERATOR_SCHEMA,
            "command": "materialize",
            "idempotent_replay": True,
            "materialized": True,
            "bootstrap_receipt": prior,
            "snapshot": snapshot,
            "completion_evidence_projection_cid": completion_projection["projection_cid"],
        }

    projection = config.get("initial_projection")
    projection = projection if isinstance(projection, Mapping) else {}
    completed_aliases = [str(item) for item in projection.get("completed_task_ids", ())]
    if completed_aliases != [f"VRIF-{ordinal:03d}" for ordinal in range(9)]:
        raise OperatorError("bootstrap completion set must be exactly VRIF-000..008")
    validation_receipt = _run_bootstrap_validation(
        board=board,
        population=population,
    )
    ingestion_population = dict(population)
    ingestion_population["tasks"] = [
        {
            **task,
            "status": "todo" if task["task_alias"] in completed_aliases else task["status"],
        }
        for task in population["tasks"]
    ]
    paths["runtime"].mkdir(parents=True, exist_ok=True)
    with DatabaseTaskSource(
        paths["database"],
        owner_id="vrif-bootstrap:single-writer",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
    ) as source:
        control_receipt = dict(source.materialize(ingestion_population))
        completion_receipt_cids = _seal_completed_tasks(
            source,
            completed_aliases=completed_aliases,
            validation_receipt=validation_receipt,
            population=population,
        )
        snapshot = source.snapshot().to_dict()
        completion_projection = dict(source.intent.completion_evidence_projection())
        ready_ids = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
    if int(snapshot["task_count"]) != len(population["tasks"]):
        raise OperatorError("DuckDB materialization task count is not exact")
    if int(snapshot["goal_count"]) != len(population["objectives"]):
        raise OperatorError("DuckDB materialization goal count is not exact")
    expected_ready_ids = [str(item) for item in projection.get("ready_task_ids", ())]
    if ready_ids != expected_ready_ids:
        raise OperatorError("initial DuckDB readiness frontier differs from the sealed projection")
    ducklake = _ducklake_projection(
        paths=paths,
        population=population,
        control_receipt=control_receipt,
    )
    receipt = {
        "schema": BOOTSTRAP_SCHEMA,
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
        "initial_ready_task_ids": ready_ids,
        "bootstrap_validation": validation_receipt,
        "completion_receipt_cids": completion_receipt_cids,
        "completion_evidence_projection_cid": completion_projection["projection_cid"],
        "authority": {
            "semantic_state": "DuckDB/DatabaseTaskSource@1",
            "state_owner_transport": "QuackStateServer@1",
            "ducklake": "optional_non_authoritative_history_projection",
        },
        "ducklake_projection": ducklake,
    }
    receipt["bootstrap_receipt_id"] = _identity(receipt)
    _atomic_json(receipt_path, receipt)
    return {
        "schema": OPERATOR_SCHEMA,
        "command": "materialize",
        "idempotent_replay": False,
        "materialized": True,
        "bootstrap_receipt": receipt,
        "snapshot": snapshot,
    }


class _LiveQuackTransport:
    """Real loopback Quack transport with an identity-complete live probe."""

    def __init__(self) -> None:
        self._listen_uri = ""

    def start(
        self,
        connection: Any,
        *,
        host: str,
        port: int,
        token: str,
        identity: Any,
    ) -> Mapping[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
            listen_uri,
        )

        uri = listen_uri(host, port)
        connection.execute(
            "SELECT * FROM quack_serve(?, token := ?, "
            "allow_other_hostname := false, disable_ssl := true)",
            [uri, token],
        )
        self._listen_uri = uri
        return MappingProxyType(
            {
                "server_id": identity.server_id,
                "store_id": identity.store_id,
                "database_uuid": identity.database_uuid,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "generation": identity.generation,
                "process_birth_id": identity.process_birth_id,
                "listen_uri": uri,
            }
        )

    def live_query(
        self,
        connection: Any,
        *,
        identity: Any,
        token: str,
    ) -> Mapping[str, Any]:
        del token
        row = connection.execute("SELECT 1").fetchone()
        if row is None:
            raise OperatorError("Quack owner connection failed its live query")
        return MappingProxyType(
            {
                "server_id": identity.server_id,
                "store_id": identity.store_id,
                "database_uuid": identity.database_uuid,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "generation": identity.generation,
                "process_birth_id": identity.process_birth_id,
                "listen_uri": self._listen_uri,
            }
        )

    def stop(self, connection: Any | None = None) -> None:
        if connection is None:
            return
        try:
            connection.execute("SELECT quack_stop()")
        except Exception:
            pass


def _verify_control_plane(path: Path) -> Any:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
        MigrationRunReport,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        CONTROL_PLANE_MIGRATION_VERSION,
        load_control_plane_catalog,
        verify_installed_schema,
    )

    # VRIF uses the canonical full control-plane schema revision ``1``.  The
    # smaller datasets-authoritative operational profile is deliberately not
    # selected: the generic multi-supervisor rejects that profile for live
    # Quack operation, and the VRIF board needs the full proof/evidence tables.
    verification = verify_installed_schema(path)
    fingerprint = str(verification.get("schema_fingerprint") or "")
    if not fingerprint:
        raise OperatorError("existing full control plane has no schema fingerprint")
    return MigrationRunReport(
        from_version=CONTROL_PLANE_MIGRATION_VERSION,
        to_version=CONTROL_PLANE_MIGRATION_VERSION,
        receipts=(),
        schema_fingerprint=fingerprint,
        catalog_fingerprint=load_control_plane_catalog().fingerprint(),
        changed=False,
    )


def _drop_fragile_task_status_indexes(connection: Any) -> None:
    """Drop ART indexes that fatally fail when ``tasks.status`` is updated.

    DuckDB 1.5 can abort the exclusive owner with
    ``Failed to delete all rows from index`` while CAS-ing ``blocked`` to
    ``retrying``. The VRIF board is 33 rows; status scans without these
    indexes remain exact.
    """

    for name in ("tasks_status_idx", "tasks_goal_idx"):
        try:
            connection.execute(f"DROP INDEX IF EXISTS {name}")
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "schema": OPERATOR_SCHEMA,
                        "event": "task_status_index_drop_failed",
                        "index": name,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:300],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def _owner_connection(path: Path) -> Any:
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnection,
    )

    connection = duckdb.connect(str(path))
    try:
        connection.execute("LOAD quack")
    except BaseException:
        connection.close()
        raise
    return DuckDBConnection.wrap(connection)


def _typed_deferral_repair_context(
    config: Mapping[str, Any],
    *,
    task_cid: str,
    task_revision: int,
    task_body: Mapping[str, Any],
    repair_head: str,
    repair_tree: str,
) -> dict[str, Any]:
    """Bind one blocked task generation to the exact clean repair HEAD."""

    source_head = str(task_body.get("base_revision") or "")
    source_tree = str(task_body.get("base_repository_tree_id") or "")
    repair_head_text = str(repair_head or "")
    repair_tree_text = str(repair_tree or "")
    if (
        not str(task_cid or "")
        or isinstance(task_revision, bool)
        or not isinstance(task_revision, int)
        or task_revision < 1
        or re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
        or re.fullmatch(r"[0-9a-f]{40}", repair_head_text) is None
        or re.fullmatch(r"[0-9a-f]{40}", repair_tree_text) is None
        or repair_head_text == source_head
    ):
        raise OperatorError("typed-deferral repair generation is invalid")
    current_head, current_tree = _assert_clean_current_tree(config)
    if current_head != repair_head_text or current_tree != repair_tree_text:
        raise OperatorError(
            "typed-deferral recovery requires the exact clean current HEAD/tree"
        )
    if _git_commit_tree(source_head, field="typed-deferral source") != source_tree:
        raise OperatorError("typed-deferral source tree does not match its commit")
    if (
        _git_commit_tree(repair_head_text, field="typed-deferral repair")
        != repair_tree_text
    ):
        raise OperatorError("typed-deferral repair tree does not match its commit")
    _git_is_ancestor(
        source_head,
        repair_head_text,
        field="typed-deferral source-to-repair ancestry",
    )
    changed_raw = _git(
        "diff",
        "--name-only",
        "--diff-filter=ACMRT",
        "--no-renames",
        source_head,
        repair_head_text,
        "--",
        *sorted(TYPED_DEFERRAL_RECOVERY_PRODUCTION_PATHS),
    )
    changed_paths = {
        line.strip()
        for line in str(changed_raw).splitlines()
        if line.strip()
    }
    if not changed_paths or not changed_paths.issubset(
        TYPED_DEFERRAL_RECOVERY_PRODUCTION_PATHS
    ):
        raise OperatorError(
            "typed-deferral repair changes no admitted production recovery path"
        )
    after_head, after_tree = _assert_clean_current_tree(config)
    if (after_head, after_tree) != (current_head, current_tree):
        raise OperatorError("typed-deferral repair generation changed during admission")
    return {
        "task_cid": str(task_cid),
        "task_revision": int(task_revision),
        "source_head": source_head,
        "source_tree": source_tree,
        "repair_head": repair_head_text,
        "repair_tree": repair_tree_text,
        "changed_production_paths": sorted(changed_paths),
    }


def _terminate_provider_canary(process: subprocess.Popen[bytes]) -> None:
    """Boundedly terminate one detached canary process group."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 15
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _run_typed_deferral_provider_canary(
    *, database_program: Any
) -> dict[str, Mapping[str, Any]]:
    """Run the real quota/high route in an inert disposable Git workspace."""

    from ipfs_accelerate_py.agent_implementation_route import (
        resolve_agent_implementation_route,
        valid_agent_implementation_failure_receipt,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner import (
        build_grok_quota_routed_agent_command,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        provider_subprocess_environment,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy import (
        extract_grok_failure_receipts,
        extract_grok_route_outcomes,
        valid_grok_route_outcome,
    )

    route = resolve_agent_implementation_route(
        primary_provider_id="grok_cli",
        primary_model_id="grok-4.6",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_trigger="primary_quota_exhausted",
        fallback_reasoning_effort="high",
    )
    nonce = secrets.token_hex(32)
    provider_env = provider_subprocess_environment(
        os.environ,
        program=database_program,
    )
    grok = shutil.which("grok", path=provider_env.get("PATH")) or ""
    codex = shutil.which("codex", path=provider_env.get("PATH")) or ""
    if not grok or not codex:
        raise OperatorError("typed-deferral canary requires trusted Grok and Codex CLIs")
    git_env = {
        key: value
        for key, value in provider_env.items()
        if not key.startswith("GIT_")
    }
    git_env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    with tempfile.TemporaryDirectory(
        prefix="vrif-provider-route-canary-"
    ) as raw_workspace:
        workspace = Path(raw_workspace).resolve()
        subprocess.run(
            ["git", "init", "-q", "-b", "main", str(workspace)],
            env=git_env,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(workspace),
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=VRIF Provider Canary",
                "-c",
                "user.email=vrif-canary.invalid",
                "commit",
                "--allow-empty",
                "--no-verify",
                "-q",
                "-m",
                "provider-route canary baseline",
            ],
            env=git_env,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        command = build_grok_quota_routed_agent_command(
            workspace=workspace,
            python_executable=sys.executable,
            grok_bin=grok,
            codex_bin=codex,
            fallback_reasoning_effort="high",
            accepted_runner_path=(
                ROOT
                / "ipfs_accelerate_py"
                / "agent_supervisor"
                / "runtime"
                / "grok_cli_runner.py"
            ),
        )
        command.extend(
            [
                "--grok-failure-receipt-nonce",
                nonce,
                "--agent-implementation-route-json",
                json.dumps(
                    route.as_binding_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
        prompt = (
            "Provider-route recovery canary only. Do not inspect or modify "
            "files and do not invoke tools. Reply with exactly CANARY_OK.\n"
        ).encode("utf-8")
        with tempfile.TemporaryFile() as control:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=git_env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=control,
                    start_new_session=True,
                )
            except OSError as exc:
                raise OperatorError("typed-deferral provider canary did not start") from exc
            try:
                assert process.stdin is not None
                process.stdin.write(prompt)
                process.stdin.close()
                deadline = (
                    time.monotonic()
                    + TYPED_DEFERRAL_PROVIDER_CANARY_TIMEOUT_SECONDS
                )
                while process.poll() is None:
                    if os.fstat(control.fileno()).st_size > (
                        TYPED_DEFERRAL_PROVIDER_CANARY_MAX_BYTES
                    ):
                        raise OperatorError(
                            "typed-deferral provider canary exceeded its log bound"
                        )
                    if time.monotonic() >= deadline:
                        raise OperatorError("typed-deferral provider canary timed out")
                    time.sleep(0.1)
            except BaseException:
                _terminate_provider_canary(process)
                raise
            control.flush()
            if os.fstat(control.fileno()).st_size > (
                TYPED_DEFERRAL_PROVIDER_CANARY_MAX_BYTES
            ):
                raise OperatorError(
                    "typed-deferral provider canary exceeded its log bound"
                )
            control.seek(0)
            control_text = control.read().decode("utf-8", errors="replace")

        receipts = extract_grok_failure_receipts(control_text)
        outcomes = extract_grok_route_outcomes(control_text)
        if len(receipts) != 1 or len(outcomes) != 1:
            raise OperatorError(
                "typed-deferral provider canary returned an ambiguous receipt chain"
            )
        receipt = receipts[0]
        outcome = outcomes[0]
        probe_returncode = receipt.get("probe_returncode")
        observed_now_ms = int(time.time() * 1000)
        valid = bool(
            process.returncode == 0
            and isinstance(probe_returncode, int)
            and not isinstance(probe_returncode, bool)
            and valid_agent_implementation_failure_receipt(
                receipt,
                nonce=nonce,
                model=route.primary_model_id,
                probe_returncode=probe_returncode,
                now_ms=observed_now_ms,
                max_age_ms=60_000,
            )
            and receipt.get("failure_class") == "hard_quota_exhausted"
            and valid_grok_route_outcome(
                outcome,
                receipt=receipt,
                route_plan=route.as_outcome_dict(),
                runner_returncode=process.returncode,
            )
            and outcome.get("decision") == "fallback_succeeded"
            and outcome.get("verifier_status") == "confirmed_quota"
            and outcome.get("fallback_dispatched") is True
            and outcome.get("fallback_returncode") == 0
            and bool(outcome.get("quota_evidence_id"))
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(workspace),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            env=git_env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if not valid or status:
            raise OperatorError(
                "typed-deferral provider canary did not produce the exact "
                "fresh quota/high success chain"
            )
        return {
            "quota_probe_receipt": receipt,
            "route_outcome": outcome,
        }


def _owner_typed_deferral_provider_evidence(
    config: Mapping[str, Any],
    *,
    database_program: Any,
    task_cid: str,
    task_revision: int,
    task_body: Mapping[str, Any],
    repair_head: str,
    repair_tree: str,
) -> dict[str, Mapping[str, Any]]:
    """Validate Git on both sides of the owner-executed provider canary."""

    before = _typed_deferral_repair_context(
        config,
        task_cid=task_cid,
        task_revision=task_revision,
        task_body=task_body,
        repair_head=repair_head,
        repair_tree=repair_tree,
    )
    evidence = _run_typed_deferral_provider_canary(
        database_program=database_program,
    )
    after = _typed_deferral_repair_context(
        config,
        task_cid=task_cid,
        task_revision=task_revision,
        task_body=task_body,
        repair_head=repair_head,
        repair_tree=repair_tree,
    )
    if before != after:
        raise OperatorError("typed-deferral repair context changed during canary")
    return evidence


def _process_owner_commands(
    repository: Any,
    command_dir: Path,
    *,
    token: str,
    expected_store_id: str,
    expected_store_generation: str,
    typed_deferral_provider_evidence_factory: Any = None,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        execute_quack_owner_command,
        quack_owner_command_error_code,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET,
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
            if (
                metadata.st_uid != os.getuid()
                or metadata.st_size > OWNER_COMMAND_ENVELOPE_MAX_BYTES
            ):
                raise OperatorError("owner command file owner or size is invalid")
            try:
                decoded = json.loads(request.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # The client creates a tiny same-filesystem request. A partial
                # read is retried rather than converted into a false failure.
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
            owner_bindings: dict[str, Any] = {
                "request_id": expected_request_id,
                "store_id": expected_store_id,
                "store_generation": expected_store_generation,
            }
            if command == QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET:
                if not callable(typed_deferral_provider_evidence_factory):
                    raise OperatorError(
                        "typed-deferral recovery provider boundary is unavailable"
                    )
                owner_bindings["typed_deferral_provider_evidence_factory"] = (
                    typed_deferral_provider_evidence_factory
                )
            result = execute_quack_owner_command(
                repository,
                command,
                command_payload,
                **owner_bindings,
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
            if error_code == "owner_error":
                print(
                    json.dumps(
                        {
                            "schema": OPERATOR_SCHEMA,
                            "event": "owner_command_error",
                            "command": str(
                                (payload or {}).get("command") or "invalid"
                            ),
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            _atomic_json(
                done,
                quack_owner_command_response(
                    response_request,
                    token=token,
                    error_code=error_code,
                    error_message=(
                        str(exc)
                        if error_code != "owner_error"
                        else (
                            "typed owner command rejected "
                            f"({type(exc).__name__}: {str(exc)[:240]})"
                        )
                    ),
                ),
            )
        try:
            request.unlink()
        except FileNotFoundError:
            pass


def state_owner(config_path: Path) -> int:
    from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
        harden_state_authority_process,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        ServerLifecycle,
        build_server,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        IntentRepository,
    )

    board, config = _load_config(config_path)
    paths = _runtime_paths(board)
    if not paths["database"].is_file() or not paths["bootstrap_receipt"].is_file():
        raise OperatorError("materialize the sealed VRIF board before starting Quack")
    prior_owner = _owner_restart_prior_status(
        paths["owner"] / "quack-state-server.status.json"
    )
    restart_admission = _owner_restart_admission(board, config, paths)
    program = board.resolved_database_program()
    endpoint = QUACK_ENDPOINT_RE.fullmatch(program.quack_endpoint)
    if endpoint is None:
        raise OperatorError("configured Quack endpoint is not loopback")
    host = endpoint.group(1)
    port = int(endpoint.group(2))
    if not 1 <= port <= 65535:
        raise OperatorError("configured Quack port is out of range")
    server = build_server(
        database_path=paths["database"],
        state_dir=paths["owner"],
        host=host,
        port=port,
        repository_id="repository:ipfs_accelerate_py",
        store_id=program.store_id,
        secret_handle=program.endpoint_secret_handle,
        # The installed Quack extension is a live-qualified beta build. This
        # permits that real transport; it never permits simulated inference.
        allow_experimental=True,
        migrate=_verify_control_plane,
        connection_factory=_owner_connection,
        transport=_LiveQuackTransport(),
    )
    identity = server.start()
    try:
        ready = server.ready()
        after_head, after_tree = _assert_clean_current_tree(config)
        if (
            after_head != restart_admission["current_source_head"]
            or after_tree != restart_admission["current_source_tree"]
        ):
            raise OperatorError("owner restart source changed during admission")
        owner_connection = getattr(server, "_connection", None)
        if owner_connection is None:
            raise OperatorError("state-owner connection is unavailable")
        _drop_fragile_task_status_indexes(owner_connection)
        database_verification = _owner_database_verification(
            owner_connection,
            restart_admission,
        )
        restart_receipt = _owner_restart_receipt(
            restart_admission,
            identity,
            expected_store_id=program.store_id,
            prior_owner=prior_owner,
            database_verification=database_verification,
        )
        restart_receipt_path = (
            paths["bootstrap_receipt"].parent
            / "owner-restarts"
            / (
                f"{int(identity.generation):020d}-"
                f"{restart_receipt['receipt_id'].removeprefix('sha256:')}.json"
            )
        )
        _atomic_json(restart_receipt_path, restart_receipt)
        owner_token = _read_owner_token(
            _token_path(paths["owner"], program.endpoint_secret_handle)
        )
        # The state owner retains the raw transport/command credential in memory.
        # Same-UID provider processes must not be able to recover it through procfs.
        os.environ["IPFS_ACCELERATE_AGENT_QUACK_TOKEN"] = owner_token
        harden_state_authority_process()
        owner_repository = IntentRepository(
            paths["database"],
            bound_connection=owner_connection,
            owner_id="vrif-quack-owner",
            session_id=f"vrif-quack-owner-{os.getpid()}",
            install_schema=False,
        )
    except BaseException:
        server.stop()
        raise
    print(
        json.dumps(
            {
                "schema": OPERATOR_SCHEMA,
                "command": "state-owner",
                "ready": True,
                "identity": identity.to_dict(),
                "live": ready,
                "owner_restart_receipt": restart_receipt,
                "owner_restart_receipt_path": str(
                    restart_receipt_path.relative_to(ROOT)
                ),
                "owner_command_dir": str(
                    (paths["owner"] / "mutations").relative_to(ROOT)
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    stopped = {"value": False}

    def request_stop(_signum: int, _frame: Any) -> None:
        stopped["value"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    command_dir = paths["owner"] / "mutations"
    control_path = server.stop_control_path()
    while server.lifecycle is ServerLifecycle.READY and not stopped["value"]:
        if control_path.is_file():
            break
        _process_owner_commands(
            owner_repository,
            command_dir,
            token=owner_token,
            expected_store_id=program.store_id,
            expected_store_generation=program.store_generation,
            typed_deferral_provider_evidence_factory=(
                lambda **context: _owner_typed_deferral_provider_evidence(
                    config,
                    database_program=program,
                    **context,
                )
            ),
        )
        time.sleep(0.05)
    owner_repository.close()
    result = server.stop()
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _unlink_token_vault(path: Path) -> None:
    """Remove one validated token file after trusted processes inherit it."""

    observed = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.getuid()
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_nlink != 1
    ):
        raise OperatorError("refusing to unlink an unsafe Quack token vault")
    path.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _launch_with_one_use_owner_token(
    launch: Any,
    launch_args: Sequence[str],
    *,
    token_path: Path,
) -> int:
    """Consume the one-use vault only after the trusted launcher succeeds."""

    result = int(launch(list(launch_args)))
    if result == 0:
        _unlink_token_vault(token_path)
    return result


def launch_supervisor(
    config_path: Path,
    *,
    dry_run: bool = False,
    duration_seconds: float = float("inf"),
) -> int:
    """Launch the configured parallel supervisor without a credential file.

    Preflight runs while the owner token is still recoverable.  For a real
    launch, this process becomes non-dumpable, starts the trusted launcher,
    and unlinks the single validated token file only after that launcher
    returns success.  Failed launches retain the vault.  Provider subprocesses
    use the canonical scrubbed environment.
    """

    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        main as configured_board_main,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
        harden_state_authority_process,
    )

    board, config = _load_config(config_path)
    paths = _runtime_paths(board)
    _assert_clean_current_tree(config)
    owner_status_path = paths["owner"] / "quack-state-server.status.json"
    if not owner_status_path.is_file():
        raise OperatorError("Quack state owner has no current status")
    owner_status = _json_object(owner_status_path)
    if (
        str(owner_status.get("lifecycle") or "") != "ready"
        or _owner_liveness(owner_status) != "alive"
    ):
        raise OperatorError("Quack state owner is not live-ready")
    program = board.resolved_database_program()
    token_path = _token_path(paths["owner"], program.endpoint_secret_handle)
    token = _read_owner_token(token_path)
    os.environ["IPFS_ACCELERATE_AGENT_QUACK_TOKEN"] = token
    # Do not ATTACH here.  A launch-time probe shares Quack's tiny listen
    # backlog with the four lanes and can make the live token fail closed
    # before any daemon claims work.  Owner liveness is already required.
    common = [
        "--repo-root",
        str(ROOT),
        "--config",
        str(config_path),
    ]
    preflight_result = configured_board_main([*common, "preflight"])
    if preflight_result != 0:
        return int(preflight_result)
    launch_args = [*common, "launch", "--implement"]
    if dry_run:
        launch_args.append("--dry-run")
        return int(configured_board_main(launch_args))
    if duration_seconds != float("inf"):
        if duration_seconds <= 0:
            raise OperatorError("supervisor duration must be positive")
        launch_args.extend(["--duration-seconds", str(duration_seconds)])
    harden_state_authority_process()
    return _launch_with_one_use_owner_token(
        configured_board_main,
        launch_args,
        token_path=token_path,
    )


def _owner_liveness(status_payload: Mapping[str, Any]) -> str:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        ProcessBirthIdentity,
        owner_liveness,
    )

    identity = status_payload.get("identity")
    if not isinstance(identity, Mapping):
        return "absent"
    birth_payload = identity.get("process_birth")
    if not isinstance(birth_payload, Mapping):
        return "unknown"
    try:
        observed = owner_liveness(ProcessBirthIdentity.from_dict(birth_payload))
    except Exception:
        return "unknown"
    if observed is OwnerLiveness.ALIVE:
        return "alive"
    if observed is OwnerLiveness.DEAD:
        return "dead"
    return "unknown"


def _token_path(owner_dir: Path, secret_handle: str) -> Path:
    safe = secret_handle.replace(":", "_").replace("/", "_")
    return owner_dir / f"{safe}.quack-token"


def _read_owner_token(path: Path) -> str:
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
        raise OperatorError("Quack token vault file is not a private regular file")
    token = path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,}", token):
        raise OperatorError("Quack token vault material is malformed")
    return token


def _probe_quack_attach(endpoint: str, token: str) -> None:
    """Fail closed if the vault token cannot attach to the live owner."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_quack_transport_connection,
    )

    try:
        connection = open_quack_transport_connection(endpoint, token=token)
    except Exception as exc:
        raise OperatorError(
            "Quack vault token does not authenticate to the live state-owner"
        ) from exc
    try:
        connection.close()
    except Exception:
        pass


def _task_status(connection: Any) -> dict[str, Any]:
    rows = connection.execute(
        "SELECT status, COUNT(*) FROM tasks GROUP BY status ORDER BY status"
    ).fetchall()
    counts = {str(row[0]): int(row[1]) for row in rows}
    # The current Quack table transport supports simple scans but can reject a
    # correlated NOT EXISTS plan as unimplemented.  Read the three canonical
    # relations separately and calculate this read-only projection locally;
    # task/dependency/block rows remain authoritative in DuckDB.
    task_rows = connection.execute(
        "SELECT task_cid, task_alias, ordinal, status FROM tasks ORDER BY ordinal, task_alias"
    ).fetchall()
    dependency_rows = connection.execute(
        "SELECT task_cid, dependency_task_cid FROM task_dependencies"
    ).fetchall()
    blocked_rows = connection.execute(
        "SELECT task_cid FROM task_blocks WHERE state = 'active'"
    ).fetchall()
    status_by_cid = {str(row[0]): str(row[3]) for row in task_rows}
    dependencies_by_cid: dict[str, list[str]] = {}
    for row in dependency_rows:
        dependencies_by_cid.setdefault(str(row[0]), []).append(str(row[1]))
    actively_blocked = {str(row[0]) for row in blocked_rows}
    ready_ids = [
        str(row[1])
        for row in task_rows
        if str(row[3]) in READY_STATUSES
        and str(row[0]) not in actively_blocked
        and all(
            status_by_cid.get(dependency) in COMPLETED_STATUSES
            for dependency in dependencies_by_cid.get(str(row[0]), ())
        )
    ][:100]
    active_rows = connection.execute(
        "SELECT task_alias FROM tasks WHERE status IN (?, ?, ?) "
        "ORDER BY ordinal, task_alias LIMIT 100",
        list(ACTIVE_STATUSES),
    ).fetchall()
    return {
        "status_counts": counts,
        "dependency_ready_task_ids": ready_ids,
        "active_task_ids": [str(row[0]) for row in active_rows],
        "blocked_count": int(counts.get("blocked", 0)),
        "terminal_count": sum(counts.get(item, 0) for item in TERMINAL_STATUSES),
        "task_count": sum(counts.values()),
    }


def status(config_path: Path) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_quack_transport_connection,
    )

    board, _config = _load_config(config_path)
    paths = _runtime_paths(board)
    program = board.resolved_database_program()
    state_status_path = paths["owner"] / "quack-state-server.status.json"
    owner_status: dict[str, Any] = {}
    if state_status_path.is_file():
        try:
            owner_status = _json_object(state_status_path)
        except OperatorError:
            owner_status = {"lifecycle": "malformed"}
    liveness = _owner_liveness(owner_status)
    lifecycle = str(owner_status.get("lifecycle") or "absent")
    live_ready = lifecycle == "ready" and liveness == "alive"
    task_projection: dict[str, Any] = {
        "available": False,
        "reason_code": "control_plane_unavailable",
    }
    connection = None
    try:
        if live_ready:
            token = _read_owner_token(_token_path(paths["owner"], program.endpoint_secret_handle))
            connection = open_quack_transport_connection(
                program.quack_endpoint,
                token=token,
            )
            task_projection = {
                "available": True,
                "transport": "quack",
                **_task_status(connection),
            }
        elif paths["database"].is_file():
            task_projection = {
                "available": False,
                "reason_code": "quack_authority_unavailable",
            }
    except Exception as exc:
        task_projection = {
            "available": False,
            "reason_code": "control_plane_probe_failed",
            "error_class": type(exc).__name__,
        }
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
    ducklake: dict[str, Any] = {
        "status": "absent",
        "authoritative": False,
        "scheduler_gate": False,
    }
    if paths["ducklake_receipt"].is_file():
        try:
            observed = _json_object(paths["ducklake_receipt"])
            ducklake = {
                "status": str(observed.get("status") or "unknown"),
                "authoritative": False,
                "scheduler_gate": False,
                "projection_receipt_id": str(observed.get("projection_receipt_id") or ""),
            }
        except OperatorError:
            ducklake["status"] = "malformed"
    return {
        "schema": OPERATOR_SCHEMA,
        "command": "status",
        "materialized": paths["database"].is_file() and paths["bootstrap_receipt"].is_file(),
        "state_owner": {
            "ready": live_ready,
            "lifecycle": lifecycle,
            "liveness": liveness,
            "identity": owner_status.get("identity"),
        },
        "task_authority": task_projection,
        "ducklake_projection": ducklake,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="repository-relative or absolute configured-board JSON",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "materialize",
        help="seal the committed Markdown bootstrap into DuckDB and DuckLake",
    )
    commands.add_parser(
        "state-owner",
        help="serve the materialized DuckDB authority through fenced loopback Quack",
    )
    status_parser = commands.add_parser(
        "status",
        help="report owner liveness and durable task readiness without exposing tokens",
    )
    status_parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit nonzero unless Quack is live and task authority is queryable",
    )
    launch_parser = commands.add_parser(
        "launch-supervisor",
        help="preflight and launch the credential-isolated parallel supervisor",
    )
    launch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render the launch without unlinking the state credential or starting workers",
    )
    launch_parser.add_argument(
        "--duration-seconds",
        type=float,
        default=float("inf"),
        help="optional positive supervisor runtime bound",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        if arguments.command == "materialize":
            result = materialize(config_path)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if arguments.command == "state-owner":
            return state_owner(config_path)
        if arguments.command == "status":
            result = status(config_path)
            print(json.dumps(result, indent=2, sort_keys=True))
            if arguments.require_ready and not (
                result["state_owner"]["ready"] and result["task_authority"].get("available") is True
            ):
                return 1
            return 0
        if arguments.command == "launch-supervisor":
            return launch_supervisor(
                config_path,
                dry_run=bool(arguments.dry_run),
                duration_seconds=float(arguments.duration_seconds),
            )
        raise OperatorError(f"unsupported command: {arguments.command}")
    except OperatorError as exc:
        print(
            json.dumps(
                {
                    "schema": OPERATOR_SCHEMA,
                    "command": str(arguments.command),
                    "ok": False,
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        # Third-party transport exception text is not a trusted secret-
        # redaction surface, so unexpected failures publish only their class.
        print(
            json.dumps(
                {
                    "schema": OPERATOR_SCHEMA,
                    "command": str(arguments.command),
                    "ok": False,
                    "error_class": type(exc).__name__,
                    "error": "operation failed closed",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
