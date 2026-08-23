"""Fresh, typed lifecycle for EAAEF reconciliation.

The historical EAAEF recovery utilities were tied to one numbered run and
mixed owner lifecycle with direct DuckDB and transport-token access.  This
module is the replacement public orchestration surface.  It deliberately does
not open DuckDB, read a Quack token, accept SQL, or infer completion from an
older generation.  Database effects are delegated to one exact typed owner
adapter; Plan R2 is applied through :class:`ExternalAgentStateRepository`.

The current accelerator tree does not yet provide the portfolio bootstrap
effect required by ``EAAEFTypedReconciliationOwner@1``.  Its statically named
facade reports the missing production bindings and refuses every effect.  The
final CASF merge must replace that blocker facade with a narrow adapter over
its exclusive typed owner.  Until it does, the public commands fail closed
after producing a useful preflight and a fresh authority request.  Stale EAAEF
receipts are never rebound.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from ..planning.external_agent_plan_r2 import (
    ExternalAgentPlanR2Error,
    prepare_plan_r2_transition_authorization,
)
from ..task_sources.external_agent_state_repository import (
    AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE,
    ExternalAgentStateRepository,
)
from ..validation.plan_r2_remote_owner_admission import (
    MAX_PLAN_R2_REMOTE_REQUEST_BYTES,
    PLAN_R2_REMOTE_CLIENT_GATEWAY_INTERFACE,
    PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
    PlanR2RemoteOwnerAdmissionError,
    VerifiedPlanR2RemoteOwnerAdmission,
    verify_plan_r2_remote_owner_admission,
)
from .plan_r2_remote_owner import (
    PLAN_R2_REMOTE_RUNTIME_PRODUCTION_BLOCKERS,
    PLAN_R2_REMOTE_RUNTIME_QUALIFICATION_STATUS,
    PlanR2ProcessRemoteOwnerGateway,
)

EAAEF_BOARD_NAMESPACE: Final = "external-agent-autonomous-execution-fabric-v1"
EAAEF_PLAN_R1_ALIAS: Final = "EAAEF-PLAN-R1"
EAAEF_PLAN_R2_ALIAS: Final = "EAAEF-PLAN-R2"
EAAEF_TASK_COUNT: Final = 116
EAAEF_BOOTSTRAP_TASK_COUNT: Final = 22
EAAEF_PLAN_R2_TASK_COUNT: Final = 94
EAAEF_GOAL_COUNT: Final = 20
EAAEF_GOAL_EDGE_COUNT: Final = 18
EAAEF_RECONCILIATION_OWNER_INTERFACE: Final = "EAAEFTypedReconciliationOwner@1"
EAAEF_RECONCILIATION_LIFECYCLE_INTERFACE: Final = "EAAEFReconciliationLifecycle@1"
EAAEF_RECONCILIATION_ROOT: Final = (
    "data/agent_supervisor/external_agent_autonomous_execution_fabric/reconciliation-v2"
)
EAAEF_BOARD_PATH: Final = (
    "docs/architecture/external_agent_autonomous_execution_fabric/task_board.json"
)
EAAEF_CONFIG_PATH: Final = "config/external_agent_autonomous_execution_fabric_scheduler.json"
EAAEF_ADMISSION_BUNDLE_PATH: Final = (
    "docs/architecture/external_agent_autonomous_execution_fabric/"
    "receipts/host_admission/admission_bundle.json"
)
EAAEF_FRESH_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-fresh-reconciliation-authority@1"
)
EAAEF_FRESH_TRUST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-fresh-reconciliation-trust-roots@1"
)
EAAEF_FOREST_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/eaaef-repository-forest-binding@1"
EAAEF_BOARD_SOURCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-board-git-source-binding@1"
)
EAAEF_POPULATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-reconciliation-population@1"
)
EAAEF_EXECUTION_CONTRACT_POPULATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-execution-contract-population@1"
)
EAAEF_OWNER_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-typed-owner-materialization-request@1"
)
EAAEF_OWNER_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-typed-owner-materialization-receipt@1"
)
EAAEF_OFFLINE_POPULATION_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-offline-population-request@1"
)
EAAEF_OFFLINE_POPULATION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-offline-population-receipt@1"
)
EAAEF_OWNER_QUALIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-typed-owner-qualification@1"
)
EAAEF_TYPED_TASK_SOURCE_INTERFACE: Final = "TypedDatabaseTaskSource@1"
EAAEF_LAUNCH_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-reconciliation-launch-request@1"
)
EAAEF_LAUNCH_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-reconciliation-launch-receipt@1"
)
EAAEF_OWNER_STATUS_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-owner-status-request@1"
)
EAAEF_OWNER_STATUS_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-owner-status-receipt@1"
)
EAAEF_OWNER_STOP_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-owner-stop-tracks-request@1"
)
EAAEF_OWNER_STOP_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-owner-stop-tracks-receipt@1"
)
EAAEF_BOOTSTRAP_SNAPSHOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-fresh-bootstrap-owner-snapshot@1"
)
EAAEF_UNSIGNED_AUTHORITY_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-unsigned-authority-request@1"
)
EAAEF_STATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-reconciliation-generation-state@1"
)
EAAEF_CURSOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-reconciliation-generation-cursor@1"
)

_GIT_OID_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_GENERATION_ID_RE: Final = re.compile(r"^eaaef-[a-z0-9][a-z0-9.-]{7,95}$")
_TASK_ALIAS_RE: Final = re.compile(r"^EAAEF-[0-9]{3}$")
_SHA256_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_DID_RE: Final = re.compile(r"^did:key:z[A-Za-z0-9]{8,511}$")
_TERMINAL_STATUSES: Final = frozenset(
    {"accepted", "cancelled", "complete", "completed", "done", "failed", "quarantined"}
)
_TASK_STATUS_VOCABULARY: Final = frozenset(
    {
        "admitted",
        "blocked",
        "cancelled",
        "claimed",
        "complete",
        "completed",
        "done",
        "failed",
        "in_progress",
        "pending",
        "proposed",
        "quarantined",
        "queued",
        "ready",
        "rejected",
        "retrying",
        "running",
        "skipped",
        "todo",
    }
)
_PLAN_R2_REMOTE_REQUEST_OVERHEAD_RESERVE: Final = 128 * 1024
_SUBMODULES: Final[tuple[tuple[str, str], ...]] = (
    ("ipfs_datasets_py", "ipfs_datasets_py"),
    ("ipfs_kit_py", "ipfs_kit_py"),
    ("mcpplusplus", "ipfs_accelerate_py/mcplusplus"),
)
_FORBIDDEN_BOUNDARY_KEYS: Final = frozenset(
    {
        "attach",
        "credential",
        "credential_path",
        "database",
        "database_path",
        "duckdb_path",
        "password",
        "secret",
        "secret_handle",
        "sql",
        "store_path",
        "token",
    }
)
_RUNTIME_ARTIFACT_NAMES: Final = frozenset(
    {
        "owner.sock",
        "supervisor.sock",
        "owner.pid.json",
        "supervisor.pid.json",
        "stop.request",
    }
)
_FRESH_AUTHORITY_FIELDS: Final = frozenset(
    {
        "schema",
        "authorization",
        "plan_r2_operational_capability",
        "plan_r2_remote_owner_capability",
        "authority_bundle_cid",
    }
)
_FRESH_TRUST_FIELDS: Final = frozenset(
    {
        "schema",
        "remote_reviewer_dids",
        "plan_r2_capability_reviewer_dids",
        "operator_dids",
        "security_reviewer_dids",
        "trust_bundle_cid",
    }
)
_OWNER_QUALIFICATION_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "source_forest_root",
        "bootstrap_materialization_mode",
        "bootstrap_materialization_before_owner_start",
        "offline_population_includes_execution_contracts",
        "direct_database_mutation_after_owner_start",
        "typed_task_source_interface",
        "plan_r2_repository_interface",
        "plan_r2_remote_gateway_interface",
        "plan_r2_wire_channel_interface",
        "plan_r2_remote_runtime_qualification_status",
        "plan_r2_remote_runtime_blockers",
        "status_operation",
        "stop_tracks_operation",
        "launch_modes",
        "database_authority_crossing_allowed",
        "filesystem_path_authority_crossing_allowed",
        "transport_token_authority_crossing_allowed",
        "sql_crossing_allowed",
        "provider_launch_allowed",
        "qualification_cid",
    }
)
_BOOTSTRAP_SNAPSHOT_FIELDS: Final = frozenset(
    {
        "schema",
        "source_head",
        "source_tree",
        "source_forest_root",
        "board_cid",
        "reconciliation_population_cid",
        "bootstrap_population_cid",
        "bootstrap_task_count",
        "held_task_count",
        "terminal_statuses_imported",
        "bootstrap_materialization_mode",
        "bootstrap_owner_absent_during_materialization",
        "owner_started_after_bootstrap",
        "direct_database_mutation_after_owner_start",
        "bootstrap_admission_cid",
        "r1_launch_capsule_cid",
        "quack_owner_qualification_cid",
        "quack_command_fabric_qualification_cid",
        "owner_principal_did",
        "shard_id",
        "store_id",
        "owner_generation",
        "expected_epoch",
        "fencing_token",
        "lease_id",
        "expected_version",
        "expected_active_plan_cid",
        "expected_active_plan_root_cid",
        "expected_active_plan_revision",
        "expected_event_cursor",
        "expected_semantic_root_cid",
        "request_id",
        "idempotency_key",
        "deadline_ms",
        "issued_at_ms",
        "expires_at_ms",
        "one_use_nonce",
        "snapshot_cid",
    }
)
_TASK_EXECUTION_BODY_FIELDS: Final = (
    "schema",
    "title",
    "objective",
    "completion_contract",
    "external_effect_scope",
    "context_artifacts",
    "owning_repository",
    "track",
    "model_route",
    "priority",
    "task_spec_cid",
    "subgoal_id",
    "epic",
)
_BOARD_SOURCE_FIELDS: Final = frozenset(
    {
        "schema",
        "relative_path",
        "source_head",
        "source_tree",
        "git_mode",
        "object_type",
        "blob_oid",
        "byte_count",
        "bytes_cid",
        "canonical_json_cid",
        "declared_board_cid",
    }
)


class EAAEFReconciliationError(RuntimeError):
    """Base failure for the fresh EAAEF reconciliation lifecycle."""


class EAAEFReconciliationBlocked(EAAEFReconciliationError):
    """A missing or stale authority prevents a safe effect."""


class EAAEFReconciliationIdentityError(EAAEFReconciliationError):
    """A source, population, process, or receipt identity differs."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise EAAEFReconciliationIdentityError("value is not canonical JSON") from exc


def _cid(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _eaaef_source_cid(value: Any) -> str:
    """Return the UTF-8 canonical CID used by the existing EAAEF board schema."""

    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise EAAEFReconciliationIdentityError(
            "EAAEF source value is not canonical JSON"
        ) from exc
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _decode_json_object(raw: bytes, *, noun: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except EAAEFReconciliationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EAAEFReconciliationIdentityError(f"{noun} is unavailable or corrupt") from exc
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise EAAEFReconciliationIdentityError(f"{noun} is not a JSON object")
    return payload


def _json_object_with_metadata(
    path: Path,
    *,
    noun: str,
    maximum_bytes: int = 32 * 1024 * 1024,
) -> tuple[dict[str, Any], os.stat_result]:
    """Read one bounded regular JSON file through a stable, no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EAAEFReconciliationIdentityError(f"{noun} is unavailable or corrupt") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise EAAEFReconciliationIdentityError(f"{noun} is not a regular file")
        if metadata.st_size <= 0 or metadata.st_size > maximum_bytes:
            raise EAAEFReconciliationIdentityError(f"{noun} size is outside its bound")
        chunks: list[bytes] = []
        observed_size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - observed_size))
            if not chunk:
                break
            chunks.append(chunk)
            observed_size += len(chunk)
            if observed_size > maximum_bytes:
                raise EAAEFReconciliationIdentityError(f"{noun} size is outside its bound")
        final_metadata = os.fstat(descriptor)
    except EAAEFReconciliationError:
        raise
    except OSError as exc:
        raise EAAEFReconciliationIdentityError(f"{noun} is unavailable or corrupt") from exc
    finally:
        os.close(descriptor)

    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        observed_size != metadata.st_size
        or any(getattr(metadata, field) != getattr(final_metadata, field) for field in stable_fields)
    ):
        raise EAAEFReconciliationIdentityError(f"{noun} changed while it was read")
    try:
        path_metadata = os.lstat(path)
    except OSError as exc:
        raise EAAEFReconciliationIdentityError(f"{noun} changed while it was read") from exc
    if (
        stat.S_ISLNK(path_metadata.st_mode)
        or path_metadata.st_dev != metadata.st_dev
        or path_metadata.st_ino != metadata.st_ino
    ):
        raise EAAEFReconciliationIdentityError(f"{noun} changed while it was read")
    return _decode_json_object(b"".join(chunks), noun=noun), metadata


def _json_object(
    path: Path,
    *,
    noun: str,
    maximum_bytes: int = 32 * 1024 * 1024,
) -> dict[str, Any]:
    payload, _metadata = _json_object_with_metadata(
        path,
        noun=noun,
        maximum_bytes=maximum_bytes,
    )
    return payload


def _private_json_object(path: Path, *, noun: str) -> dict[str, Any]:
    payload, metadata = _json_object_with_metadata(path, noun=noun)
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise EAAEFReconciliationIdentityError(f"{noun} is not private to the current owner")
    return payload


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EAAEFReconciliationIdentityError("git identity probe failed") from exc
    if check and result.returncode != 0:
        raise EAAEFReconciliationIdentityError(
            result.stderr.strip() or result.stdout.strip() or "git identity probe failed"
        )
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_blob(repo_root: Path, blob_oid: str, *, maximum_bytes: int) -> bytes:
    if not _GIT_OID_RE.fullmatch(blob_oid):
        raise EAAEFReconciliationIdentityError("Git blob identity is malformed")
    raw_size = _git(repo_root, "cat-file", "-s", blob_oid)
    try:
        size = int(raw_size)
    except ValueError as exc:
        raise EAAEFReconciliationIdentityError("Git blob size is malformed") from exc
    if size <= 0 or size > maximum_bytes:
        raise EAAEFReconciliationIdentityError("Git blob size is outside its bound")
    try:
        result = subprocess.run(
            ["git", "cat-file", "blob", blob_oid],
            cwd=repo_root,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EAAEFReconciliationIdentityError("Git blob read failed") from exc
    if result.returncode != 0 or len(result.stdout) != size:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise EAAEFReconciliationIdentityError(detail or "Git blob read failed")
    return result.stdout


def _board_source_binding(
    raw_board: bytes,
    *,
    source_head: str,
    source_tree: str,
    git_mode: str,
    blob_oid: str,
) -> dict[str, Any]:
    """Bind exact committed board bytes and semantics to their Git locator."""

    if (
        not _GIT_OID_RE.fullmatch(source_head)
        or not _GIT_OID_RE.fullmatch(source_tree)
        or git_mode != "100644"
        or not _GIT_OID_RE.fullmatch(blob_oid)
    ):
        raise EAAEFReconciliationIdentityError("EAAEF board Git source is malformed")
    expected_blob_oid = hashlib.sha1(
        b"blob " + str(len(raw_board)).encode("ascii") + b"\0" + raw_board,
        usedforsecurity=False,
    ).hexdigest()
    if blob_oid != expected_blob_oid:
        raise EAAEFReconciliationIdentityError("EAAEF board Git blob identity differs")
    board = _decode_json_object(raw_board, noun="committed EAAEF task board")
    return {
        "schema": EAAEF_BOARD_SOURCE_SCHEMA,
        "relative_path": EAAEF_BOARD_PATH,
        "source_head": source_head,
        "source_tree": source_tree,
        "git_mode": git_mode,
        "object_type": "blob",
        "blob_oid": blob_oid,
        "byte_count": len(raw_board),
        "bytes_cid": _cid(raw_board),
        "canonical_json_cid": _eaaef_source_cid(board),
        "declared_board_cid": str(board.get("board_cid") or ""),
    }


def _git_board_source(repo_root: Path, *, source_head: str, source_tree: str) -> dict[str, Any]:
    line = _git(repo_root, "ls-tree", source_tree, "--", EAAEF_BOARD_PATH)
    if "\t" not in line:
        raise EAAEFReconciliationIdentityError("EAAEF board is absent from the sealed Git tree")
    identity, observed_path = line.split("\t", 1)
    fields = identity.split()
    if len(fields) != 3:
        raise EAAEFReconciliationIdentityError("EAAEF board Git source is malformed")
    git_mode, object_type, blob_oid = fields
    if observed_path != EAAEF_BOARD_PATH or object_type != "blob":
        raise EAAEFReconciliationIdentityError("EAAEF board Git source differs")
    raw_board = _git_blob(repo_root, blob_oid, maximum_bytes=32 * 1024 * 1024)
    return _board_source_binding(
        raw_board,
        source_head=source_head,
        source_tree=source_tree,
        git_mode=git_mode,
        blob_oid=blob_oid,
    )


def _gitlink_commit(repo_root: Path, source_tree: str, relative_path: str) -> str:
    line = _git(repo_root, "ls-tree", source_tree, "--", relative_path)
    prefix = "160000 commit "
    if not line.startswith(prefix) or "\t" not in line:
        raise EAAEFReconciliationIdentityError(f"{relative_path} is not one exact gitlink")
    identity, observed_path = line.split("\t", 1)
    commit = identity.removeprefix(prefix)
    if observed_path != relative_path or not _GIT_OID_RE.fullmatch(commit):
        raise EAAEFReconciliationIdentityError(f"{relative_path} gitlink identity is malformed")
    return commit


def inspect_current_repository_forest(repo_root: str | Path) -> dict[str, Any]:
    """Inspect and seal the exact current accelerator and required gitlinks.

    A gitlink commit is not silently treated as a checked-out repository.  The
    nested checkout must exist, be clean, and match the superproject entry so
    its tree can be sealed.  This is intentionally stricter than merely
    hashing ``.gitmodules``.
    """

    root = Path(repo_root).resolve(strict=True)
    blockers: list[str] = []
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", f"{head}^{{tree}}")
    if not _GIT_OID_RE.fullmatch(head) or not _GIT_OID_RE.fullmatch(tree):
        raise EAAEFReconciliationIdentityError("accelerator HEAD/tree is malformed")
    board_source = _git_board_source(root, source_head=head, source_tree=tree)
    root_status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if root_status:
        blockers.append("accelerator_worktree_not_clean")
    repositories: list[dict[str, Any]] = [
        {
            "name": "ipfs_accelerate_py",
            "relative_path": ".",
            "commit": head,
            "tree": tree,
            "gitlink": False,
            "initialized": True,
            "clean": not bool(root_status),
        }
    ]
    for name, relative_path in _SUBMODULES:
        gitlink = _gitlink_commit(root, tree, relative_path)
        nested = root / relative_path
        try:
            nested_metadata = os.lstat(nested)
        except FileNotFoundError:
            nested_metadata = None
        safe_directory = bool(
            nested_metadata is not None
            and stat.S_ISDIR(nested_metadata.st_mode)
            and not stat.S_ISLNK(nested_metadata.st_mode)
        )
        nested_top = (
            _git(nested, "rev-parse", "--show-toplevel", check=False) if safe_directory else ""
        )
        exact_repository = bool(
            nested_top and Path(nested_top).resolve(strict=True) == nested.resolve(strict=True)
        )
        nested_head = _git(nested, "rev-parse", "HEAD", check=False) if exact_repository else ""
        nested_tree = (
            _git(nested, "rev-parse", f"{nested_head}^{{tree}}", check=False)
            if nested_head
            else ""
        )
        nested_status = (
            _git(nested, "status", "--porcelain=v1", "--untracked-files=all", check=False)
            if nested_head
            else ""
        )
        initialized = bool(
            _GIT_OID_RE.fullmatch(nested_head) and _GIT_OID_RE.fullmatch(nested_tree)
        )
        if not initialized:
            blockers.append(f"required_gitlink_uninitialized:{relative_path}")
        elif nested_head != gitlink:
            blockers.append(f"gitlink_checkout_mismatch:{relative_path}")
        if initialized and nested_status:
            blockers.append(f"required_gitlink_dirty:{relative_path}")
        repositories.append(
            {
                "name": name,
                "relative_path": relative_path,
                "commit": gitlink,
                "tree": nested_tree if initialized else "",
                "gitlink": True,
                "initialized": initialized,
                "clean": initialized and not bool(nested_status),
            }
        )
    final_head = _git(root, "rev-parse", "HEAD")
    final_status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if final_head != head:
        blockers.append("accelerator_head_changed_during_inspection")
    if final_status != root_status:
        blockers.append("accelerator_worktree_changed_during_inspection")
    if final_status and "accelerator_worktree_not_clean" not in blockers:
        blockers.append("accelerator_worktree_not_clean")
    repositories[0]["clean"] = not bool(root_status or final_status) and final_head == head
    identity = {
        "schema": EAAEF_FOREST_SCHEMA,
        "repositories": repositories,
        "board_source": board_source,
    }
    complete = not blockers
    forest_root = _cid(identity) if complete else ""
    return {
        **identity,
        "valid": complete,
        "blockers": blockers,
        "source_head": head,
        "source_tree": tree,
        "source_forest_root": forest_root,
        "source_generation_cid": forest_root,
        "binding_cid": _cid({**identity, "source_forest_root": forest_root}) if complete else "",
    }


def _require_sealed_forest(forest: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(forest)
    if (
        value.get("schema") != EAAEF_FOREST_SCHEMA
        or value.get("valid") is not True
        or not _GIT_OID_RE.fullmatch(str(value.get("source_head") or ""))
        or not _GIT_OID_RE.fullmatch(str(value.get("source_tree") or ""))
        or not _SHA256_RE.fullmatch(str(value.get("source_forest_root") or ""))
        or value.get("source_generation_cid") != value.get("source_forest_root")
    ):
        raise EAAEFReconciliationIdentityError("repository forest is not one complete current seal")
    repositories = value.get("repositories")
    if not isinstance(repositories, list) or len(repositories) != 4:
        raise EAAEFReconciliationIdentityError("repository forest population differs")
    expected_repositories = (
        ("ipfs_accelerate_py", ".", False),
        ("ipfs_datasets_py", "ipfs_datasets_py", True),
        ("ipfs_kit_py", "ipfs_kit_py", True),
        ("mcpplusplus", "ipfs_accelerate_py/mcplusplus", True),
    )
    for observed, (name, relative_path, gitlink) in zip(
        repositories,
        expected_repositories,
        strict=True,
    ):
        if (
            not isinstance(observed, Mapping)
            or set(observed)
            != {"name", "relative_path", "commit", "tree", "gitlink", "initialized", "clean"}
            or observed.get("name") != name
            or observed.get("relative_path") != relative_path
            or observed.get("gitlink") is not gitlink
            or observed.get("initialized") is not True
            or observed.get("clean") is not True
            or not _GIT_OID_RE.fullmatch(str(observed.get("commit") or ""))
            or not _GIT_OID_RE.fullmatch(str(observed.get("tree") or ""))
        ):
            raise EAAEFReconciliationIdentityError("repository forest member differs")
    if (
        repositories[0]["commit"] != value["source_head"]
        or repositories[0]["tree"] != value["source_tree"]
    ):
        raise EAAEFReconciliationIdentityError("accelerator forest member differs")
    raw_board_source = value.get("board_source")
    if not isinstance(raw_board_source, Mapping):
        raise EAAEFReconciliationIdentityError("repository forest board source is absent")
    board_source = dict(raw_board_source)
    if (
        set(board_source) != _BOARD_SOURCE_FIELDS
        or board_source.get("schema") != EAAEF_BOARD_SOURCE_SCHEMA
        or board_source.get("relative_path") != EAAEF_BOARD_PATH
        or board_source.get("source_head") != value["source_head"]
        or board_source.get("source_tree") != value["source_tree"]
        or board_source.get("git_mode") != "100644"
        or board_source.get("object_type") != "blob"
        or not _GIT_OID_RE.fullmatch(str(board_source.get("blob_oid") or ""))
        or not isinstance(board_source.get("byte_count"), int)
        or isinstance(board_source.get("byte_count"), bool)
        or not 0 < int(board_source["byte_count"]) <= 32 * 1024 * 1024
        or not _SHA256_RE.fullmatch(str(board_source.get("bytes_cid") or ""))
        or not _SHA256_RE.fullmatch(str(board_source.get("canonical_json_cid") or ""))
        or not _SHA256_RE.fullmatch(str(board_source.get("declared_board_cid") or ""))
    ):
        raise EAAEFReconciliationIdentityError("repository forest board source differs")
    identity = {
        "schema": EAAEF_FOREST_SCHEMA,
        "repositories": repositories,
        "board_source": board_source,
    }
    expected_root = _cid(identity)
    if (
        value.get("source_forest_root") != expected_root
        or value.get("binding_cid") != _cid({**identity, "source_forest_root": expected_root})
        or value.get("blockers") != []
    ):
        raise EAAEFReconciliationIdentityError("repository forest self-address differs")
    return value


@dataclass(frozen=True)
class CompiledEAAEFPopulation:
    """Fresh R1 bootstrap and R2-held populations bound to one forest."""

    board_cid: str
    source_head: str
    source_tree: str
    source_forest_root: str
    plan_r1_cid: str
    goals: tuple[Mapping[str, Any], ...]
    goal_edges: tuple[Mapping[str, str], ...]
    plan_r1: Mapping[str, Any]
    bootstrap_tasks: tuple[Mapping[str, Any], ...]
    plan_r2_tasks: tuple[Mapping[str, Any], ...]
    dependencies: tuple[Mapping[str, str], ...]
    goal_population_cid: str
    execution_contract_population_cid: str
    bootstrap_population_cid: str
    plan_r2_population_cid: str
    population_cid: str

    @property
    def task_count(self) -> int:
        return len(self.bootstrap_tasks) + len(self.plan_r2_tasks)

    @property
    def execution_contract_counts(self) -> dict[str, int]:
        """Return exact auxiliary-row counts required by offline materialization."""

        tasks = (*self.bootstrap_tasks, *self.plan_r2_tasks)
        return {
            "task_dependencies": len(self.dependencies),
            "task_outputs": sum(len(item["body"]["outputs"]) for item in tasks),
            "task_validations": sum(len(item["body"]["validations"]) for item in tasks),
            "task_acceptance": sum(len(item["body"]["acceptance"]) for item in tasks),
        }

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema": EAAEF_POPULATION_SCHEMA,
            "board_cid": self.board_cid,
            "source_head": self.source_head,
            "source_tree": self.source_tree,
            "source_forest_root": self.source_forest_root,
            "plan_r1_cid": self.plan_r1_cid,
            "goal_population_cid": self.goal_population_cid,
            "execution_contract_population_cid": self.execution_contract_population_cid,
            "goal_count": len(self.goals),
            "goal_edge_count": len(self.goal_edges),
            "plan_count": 1,
            "bootstrap_task_count": len(self.bootstrap_tasks),
            "plan_r2_task_count": len(self.plan_r2_tasks),
            "task_count": self.task_count,
            "dependency_count": len(self.dependencies),
            "bootstrap_population_cid": self.bootstrap_population_cid,
            "plan_r2_population_cid": self.plan_r2_population_cid,
            "population_cid": self.population_cid,
            "terminal_statuses_imported": 0,
        }

    def plan_r2_transition_tasks(self, *, plan_cid: str) -> tuple[dict[str, Any], ...]:
        """Return the exact fresh target rows admitted by a signed Plan R2."""

        if not _SHA256_RE.fullmatch(plan_cid):
            raise EAAEFReconciliationIdentityError("Plan-R2 plan CID is malformed")
        rows: list[dict[str, Any]] = []
        for source in (*self.bootstrap_tasks, *self.plan_r2_tasks):
            row = json.loads(_canonical_bytes(dict(source)))
            row.pop("execution_contract_cid", None)
            row["plan_cid"] = plan_cid
            row["revision"] = int(row["revision"]) + 1
            row["status"] = "todo"
            body = dict(row["body"])
            body["plan_revision"] = EAAEF_PLAN_R2_ALIAS
            body["accepted_plan_cid"] = plan_cid
            body["population_state"] = "plan_r2_accepted"
            body["is_schedulable"] = True
            body["blocked_reason"] = ""
            row["body"] = body
            rows.append(row)
        return tuple(sorted(rows, key=lambda item: str(item["task_cid"])))


def _require_current_board_provenance(
    board: Mapping[str, Any],
    *,
    sealed_forest: Mapping[str, Any],
) -> None:
    board_source = sealed_forest["board_source"]
    if (
        board_source["canonical_json_cid"] != _eaaef_source_cid(board)
        or board_source["declared_board_cid"] != str(board.get("board_cid") or "")
    ):
        raise EAAEFReconciliationIdentityError(
            "EAAEF current board differs from the sealed Git board source"
        )


def compile_fresh_eaaef_population(
    board: Mapping[str, Any],
    *,
    forest: Mapping[str, Any],
) -> CompiledEAAEFPopulation:
    """Compile exactly 22 R1 bootstrap plus 94 held R2 tasks.

    Declared completion is deliberately ignored: a fresh generation imports no
    terminal status.  The signed Plan-R2 transition may move the held tasks to
    ``todo`` only after the owner verifies its complete authorization.
    """

    sealed = _require_sealed_forest(forest)
    try:
        board = json.loads(_canonical_bytes(dict(board)))
    except (TypeError, ValueError) as exc:
        raise EAAEFReconciliationIdentityError("EAAEF board snapshot is malformed") from exc
    if not isinstance(board, dict):
        raise EAAEFReconciliationIdentityError("EAAEF board snapshot is malformed")
    declared_board_cid = str(board.get("board_cid") or "")
    board_cid_projection = dict(board)
    board_cid_projection.pop("board_cid", None)
    if declared_board_cid != _eaaef_source_cid(board_cid_projection):
        raise EAAEFReconciliationIdentityError("EAAEF declared board CID differs")
    if (
        board.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or board.get("plan_revision") != EAAEF_PLAN_R1_ALIAS
    ):
        raise EAAEFReconciliationIdentityError("EAAEF board identity differs")
    raw_tasks = board.get("tasks")
    raw_initial = board.get("initial_population_task_ids")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != EAAEF_TASK_COUNT:
        raise EAAEFReconciliationIdentityError("EAAEF board must contain exactly 116 tasks")
    if not isinstance(raw_initial, list) or len(raw_initial) != EAAEF_BOOTSTRAP_TASK_COUNT:
        raise EAAEFReconciliationIdentityError(
            "EAAEF bootstrap population must contain exactly 22 tasks"
        )
    initial = tuple(str(item) for item in raw_initial)
    if len(set(initial)) != len(initial) or any(
        not _TASK_ALIAS_RE.fullmatch(item) for item in initial
    ):
        raise EAAEFReconciliationIdentityError("EAAEF bootstrap identities are malformed")
    by_alias: dict[str, Mapping[str, Any]] = {}
    for task in raw_tasks:
        if not isinstance(task, Mapping):
            raise EAAEFReconciliationIdentityError("EAAEF task is not an object")
        alias = str(task.get("stable_task_id") or "")
        if not _TASK_ALIAS_RE.fullmatch(alias) or alias in by_alias:
            raise EAAEFReconciliationIdentityError("EAAEF task identity is malformed or duplicated")
        if task.get("board_namespace") != EAAEF_BOARD_NAMESPACE:
            raise EAAEFReconciliationIdentityError(f"{alias} crosses the board namespace")
        if not _SHA256_RE.fullmatch(str(task.get("task_spec_cid") or "")):
            raise EAAEFReconciliationIdentityError(f"{alias} task spec CID is malformed")
        task_spec_projection = dict(task)
        task_spec_cid = str(task_spec_projection.pop("task_spec_cid", ""))
        if task_spec_cid != _eaaef_source_cid(task_spec_projection):
            raise EAAEFReconciliationIdentityError(f"{alias} task spec CID differs")
        by_alias[alias] = task
    if set(initial) - set(by_alias):
        raise EAAEFReconciliationIdentityError("EAAEF bootstrap task is absent")
    held = tuple(alias for alias in by_alias if alias not in set(initial))
    if len(held) != EAAEF_PLAN_R2_TASK_COUNT:
        raise EAAEFReconciliationIdentityError("EAAEF Plan-R2 population must contain 94 tasks")

    raw_goals = board.get("goals")
    if not isinstance(raw_goals, list) or len(raw_goals) != EAAEF_GOAL_COUNT:
        raise EAAEFReconciliationIdentityError("EAAEF goal population is malformed")
    goal_specs: dict[str, Mapping[str, Any]] = {}
    for goal in raw_goals:
        if not isinstance(goal, Mapping):
            raise EAAEFReconciliationIdentityError("EAAEF goal is not an object")
        alias = str(goal.get("goal_id") or "")
        if not alias or alias in goal_specs:
            raise EAAEFReconciliationIdentityError("EAAEF goal identity is malformed or duplicated")
        goal_specs[alias] = goal

    _require_current_board_provenance(board, sealed_forest=sealed)

    board_identity = {
        "schema": "EAAEFFreshBoardIdentity@1",
        "declared_board_cid": declared_board_cid,
        "source_forest_root": sealed["source_forest_root"],
        "tasks": [
            {
                "task_alias": alias,
                "task_spec_cid": str(by_alias[alias].get("task_spec_cid") or ""),
            }
            for alias in sorted(by_alias)
        ],
    }
    board_cid = _cid(board_identity)
    plan_r1_cid = _cid(
        {
            "schema": "EAAEFFreshPlanIdentity@1",
            "plan_alias": EAAEF_PLAN_R1_ALIAS,
            "board_cid": board_cid,
            "source_forest_root": sealed["source_forest_root"],
        }
    )
    goal_cids = {
        alias: _cid(
            {
                "schema": "EAAEFFreshGoalIdentity@1",
                "goal_alias": alias,
                "goal_spec": goal,
                "board_cid": board_cid,
                "source_forest_root": sealed["source_forest_root"],
            }
        )
        for alias, goal in goal_specs.items()
    }
    goal_records: list[Mapping[str, Any]] = []
    goal_edges: list[Mapping[str, str]] = []
    for ordinal, (alias, goal) in enumerate(goal_specs.items(), start=1):
        parent_alias = str(goal.get("parent_goal_id") or "")
        if parent_alias and parent_alias not in goal_cids:
            raise EAAEFReconciliationIdentityError(f"{alias} has an unknown parent goal")
        raw_goal_dependencies = goal.get("dependencies")
        if not isinstance(raw_goal_dependencies, list) or any(
            not isinstance(item, str) or item not in goal_cids
            for item in raw_goal_dependencies
        ):
            raise EAAEFReconciliationIdentityError(f"{alias} has malformed goal dependencies")
        dependency_goal_cids = [goal_cids[item] for item in raw_goal_dependencies]
        for dependency_alias, dependency_cid in zip(
            raw_goal_dependencies,
            dependency_goal_cids,
            strict=True,
        ):
            goal_edges.append(
                {
                    "parent_goal_cid": dependency_cid,
                    "child_goal_cid": goal_cids[alias],
                    "edge_kind": "requires",
                    "parent_goal_alias": dependency_alias,
                    "child_goal_alias": alias,
                }
            )
        goal_records.append(
            {
                "goal_cid": goal_cids[alias],
                "goal_alias": alias,
                "title": str(goal.get("title") or alias),
                "objective_id": "objective:eaaef-root",
                "parent_goal_cid": goal_cids.get(parent_alias, ""),
                "ordinal": ordinal,
                "status": "open",
                "identity": {
                    "schema": "EAAEFFreshGoalIdentity@1",
                    "goal_cid": goal_cids[alias],
                    "goal_alias": alias,
                    "board_cid": board_cid,
                    "source_forest_root": sealed["source_forest_root"],
                },
                "body": {
                    "schema": "EAAEFFreshGoalBody@1",
                    "goal_spec": json.loads(_canonical_bytes(goal)),
                    "dependency_goal_cids": dependency_goal_cids,
                    "board_cid": board_cid,
                    "source_forest_root": sealed["source_forest_root"],
                    "fresh_generation": True,
                    "historical_status_imported": False,
                },
            }
        )
    root_goals = [item for item in goal_records if not item["parent_goal_cid"]]
    if len(root_goals) != 1 or root_goals[0]["goal_alias"] != "EAAEF-G000":
        raise EAAEFReconciliationIdentityError("EAAEF root goal identity differs")
    if len(goal_edges) != EAAEF_GOAL_EDGE_COUNT:
        raise EAAEFReconciliationIdentityError("EAAEF goal dependency population differs")
    plan_r1 = {
        "plan_cid": plan_r1_cid,
        "plan_alias": EAAEF_PLAN_R1_ALIAS,
        "goal_cid": root_goals[0]["goal_cid"],
        "status": "active",
        "revision": 1,
        "semantic_root_cid": sealed["source_forest_root"],
        "body": {
            "schema": "EAAEFFreshPlanR1Body@1",
            "board_cid": board_cid,
            "source_head": sealed["source_head"],
            "source_tree": sealed["source_tree"],
            "source_forest_root": sealed["source_forest_root"],
            "bootstrap_task_count": EAAEF_BOOTSTRAP_TASK_COUNT,
            "held_task_count": EAAEF_PLAN_R2_TASK_COUNT,
            "terminal_statuses_imported": 0,
            "fresh_generation": True,
        },
    }
    goal_population_cid = _cid(
        {
            "schema": "EAAEFFreshGoalPopulation@1",
            "goals": goal_records,
            "goal_edges": goal_edges,
            "plan_r1": plan_r1,
            "source_forest_root": sealed["source_forest_root"],
        }
    )
    task_cids = {
        alias: _cid(
            {
                "schema": "EAAEFFreshTaskIdentity@1",
                "task_alias": alias,
                "task_spec_cid": str(task.get("task_spec_cid") or ""),
                "board_cid": board_cid,
                "source_forest_root": sealed["source_forest_root"],
            }
        )
        for alias, task in by_alias.items()
    }

    dependencies: list[Mapping[str, str]] = []
    records: dict[str, Mapping[str, Any]] = {}
    for ordinal, alias in enumerate([*initial, *held], start=1):
        task = by_alias[alias]
        goal_alias = str(task.get("subgoal_id") or "")
        if goal_alias not in goal_cids:
            raise EAAEFReconciliationIdentityError(f"{alias} has an unknown goal")
        dependency_aliases = tuple(str(item) for item in task.get("dependencies") or ())
        if any(item not in task_cids for item in dependency_aliases):
            raise EAAEFReconciliationIdentityError(f"{alias} has an unknown dependency")
        for dependency_alias in dependency_aliases:
            dependencies.append(
                {
                    "task_cid": task_cids[alias],
                    "dependency_task_cid": task_cids[dependency_alias],
                    "kind": "requires",
                }
            )
        body = {
            field_name: json.loads(_canonical_bytes(task[field_name]))
            for field_name in _TASK_EXECUTION_BODY_FIELDS
            if field_name in task
        }
        raw_writes = task.get("execution_owned_files")
        if (
            not isinstance(raw_writes, list)
            or not raw_writes
            or any(not isinstance(item, str) or not item for item in raw_writes)
        ):
            raise EAAEFReconciliationIdentityError(f"{alias} has no exact write scope")
        write_scope = list(raw_writes)
        raw_reads = task.get("read_scope")
        if raw_reads is not None and (
            not isinstance(raw_reads, list)
            or not raw_reads
            or any(not isinstance(item, str) or not item for item in raw_reads)
        ):
            raise EAAEFReconciliationIdentityError(f"{alias} has a malformed read scope")
        raw_validations = task.get("execution_validation")
        if not isinstance(raw_validations, list) or not raw_validations:
            raise EAAEFReconciliationIdentityError(f"{alias} has no exact validation scope")
        body.update(
            {
                "task_alias": alias,
                "task_id": alias,
                "task_spec_cid": str(task.get("task_spec_cid") or ""),
                "board_cid": board_cid,
                "accepted_source_forest_root": sealed["source_forest_root"],
                "dependency_task_cids": [task_cids[item] for item in dependency_aliases],
                "read_scope": list(raw_reads or ["declared dependency receipts"]),
                "write_scope": write_scope,
                "effect_scope": write_scope,
                "completion": str(task.get("completion_mode") or "manual"),
                "review_only": False,
                "predicted_files": write_scope,
                "outputs": write_scope,
                "depends_on": list(dependency_aliases),
                "validations": list(raw_validations),
                "acceptance": [str(task.get("acceptance") or "")],
                "fresh_generation": True,
                "historical_status_imported": False,
            }
        )
        in_bootstrap = alias in set(initial)
        body["plan_revision"] = EAAEF_PLAN_R1_ALIAS if in_bootstrap else EAAEF_PLAN_R2_ALIAS
        body["population_state"] = (
            "materialized_bootstrap" if in_bootstrap else "held_until_signed_plan_r2"
        )
        body["is_schedulable"] = in_bootstrap
        body["blocked_reason"] = "" if in_bootstrap else "awaiting_fresh_signed_plan_r2"
        record = {
            "task_cid": task_cids[alias],
            "task_alias": alias,
            "goal_cid": goal_cids[goal_alias],
            "plan_cid": plan_r1_cid,
            "objective_id": "objective:eaaef-root",
            "ordinal": ordinal,
            "status": "todo" if in_bootstrap else "blocked",
            "revision": 1,
            "priority": str(task.get("priority") or "P0"),
            "identity": {
                "schema": "EAAEFFreshTaskIdentity@1",
                "task_cid": task_cids[alias],
                "task_alias": alias,
                "task_spec_cid": str(task.get("task_spec_cid") or ""),
                "board_cid": board_cid,
                "source_forest_root": sealed["source_forest_root"],
            },
            "body": body,
        }
        record["execution_contract_cid"] = _cid(
            {
                "schema": "EAAEFTaskExecutionContract@1",
                "task": record,
                "source_forest_root": sealed["source_forest_root"],
            }
        )
        records[alias] = record
    bootstrap_records = tuple(records[alias] for alias in initial)
    held_records = tuple(records[alias] for alias in held)
    bootstrap_cid = _cid(
        {
            "schema": "EAAEFBootstrapPopulation@1",
            "tasks": bootstrap_records,
            "dependencies": dependencies,
            "source_forest_root": sealed["source_forest_root"],
        }
    )
    plan_r2_cid = _cid(
        {
            "schema": "EAAEFPlanR2HeldPopulation@1",
            "tasks": held_records,
            "dependencies": dependencies,
            "source_forest_root": sealed["source_forest_root"],
        }
    )
    execution_contract_population_cid = _cid(
        {
            "schema": EAAEF_EXECUTION_CONTRACT_POPULATION_SCHEMA,
            "contracts": [
                {
                    "task_cid": record["task_cid"],
                    "execution_contract_cid": record["execution_contract_cid"],
                }
                for record in (*bootstrap_records, *held_records)
            ],
            "source_forest_root": sealed["source_forest_root"],
        }
    )
    population_cid = _cid(
        {
            "schema": EAAEF_POPULATION_SCHEMA,
            "board_cid": board_cid,
            "bootstrap_population_cid": bootstrap_cid,
            "plan_r2_population_cid": plan_r2_cid,
            "goal_population_cid": goal_population_cid,
            "execution_contract_population_cid": execution_contract_population_cid,
            "source_forest_root": sealed["source_forest_root"],
            "task_count": EAAEF_TASK_COUNT,
        }
    )
    return CompiledEAAEFPopulation(
        board_cid=board_cid,
        source_head=str(sealed["source_head"]),
        source_tree=str(sealed["source_tree"]),
        source_forest_root=str(sealed["source_forest_root"]),
        plan_r1_cid=plan_r1_cid,
        goals=tuple(goal_records),
        goal_edges=tuple(goal_edges),
        plan_r1=plan_r1,
        bootstrap_tasks=bootstrap_records,
        plan_r2_tasks=held_records,
        dependencies=tuple(dependencies),
        goal_population_cid=goal_population_cid,
        execution_contract_population_cid=execution_contract_population_cid,
        bootstrap_population_cid=bootstrap_cid,
        plan_r2_population_cid=plan_r2_cid,
        population_cid=population_cid,
    )


def verify_compiled_eaaef_population_commitments(
    population: CompiledEAAEFPopulation,
    *,
    current_board: Mapping[str, Any],
    current_forest: Mapping[str, Any],
) -> CompiledEAAEFPopulation:
    """Rebuild every population commitment from the current sealed inputs.

    ``CompiledEAAEFPopulation`` is frozen, but its nested JSON records are
    ordinary Python values.  A retained top-level population CID therefore is
    not evidence that a caller did not mutate one nested row.  This verifier
    recomputes the complete commitment chain and then compares the entire
    population with an independent compilation of the self-addressed board.
    """

    if type(population) is not CompiledEAAEFPopulation:
        raise EAAEFReconciliationIdentityError("EAAEF population was not freshly compiled")
    sealed = _require_sealed_forest(current_forest)
    source_mismatches = sorted(
        field_name
        for field_name in ("source_head", "source_tree", "source_forest_root")
        if getattr(population, field_name) != sealed[field_name]
    )
    if source_mismatches:
        raise EAAEFReconciliationIdentityError(
            "compiled EAAEF population belongs to a stale repository forest: "
            + ", ".join(source_mismatches)
        )

    if (
        len(population.bootstrap_tasks) != EAAEF_BOOTSTRAP_TASK_COUNT
        or len(population.plan_r2_tasks) != EAAEF_PLAN_R2_TASK_COUNT
    ):
        raise EAAEFReconciliationIdentityError("compiled EAAEF 22+94 partitions differ")
    tasks = (*population.bootstrap_tasks, *population.plan_r2_tasks)
    contract_bindings: list[dict[str, str]] = []
    seen_task_cids: set[str] = set()
    for raw in tasks:
        if not isinstance(raw, Mapping):
            raise EAAEFReconciliationIdentityError("compiled EAAEF task is not an object")
        record = dict(raw)
        body = record.get("body")
        identity = record.get("identity")
        task_alias = str(record.get("task_alias") or "")
        if not isinstance(body, Mapping) or not isinstance(identity, Mapping):
            raise EAAEFReconciliationIdentityError(
                f"compiled EAAEF task {task_alias or 'unknown'} contract is absent"
            )
        task_spec_cid = str(body.get("task_spec_cid") or "")
        expected_task_cid = _cid(
            {
                "schema": "EAAEFFreshTaskIdentity@1",
                "task_alias": task_alias,
                "task_spec_cid": task_spec_cid,
                "board_cid": population.board_cid,
                "source_forest_root": population.source_forest_root,
            }
        )
        expected_identity = {
            "schema": "EAAEFFreshTaskIdentity@1",
            "task_cid": expected_task_cid,
            "task_alias": task_alias,
            "task_spec_cid": task_spec_cid,
            "board_cid": population.board_cid,
            "source_forest_root": population.source_forest_root,
        }
        if record.get("task_cid") != expected_task_cid or dict(identity) != expected_identity:
            raise EAAEFReconciliationIdentityError(
                f"compiled EAAEF task {task_alias or 'unknown'} CID differs"
            )
        if expected_task_cid in seen_task_cids:
            raise EAAEFReconciliationIdentityError("compiled EAAEF task CID is duplicated")
        seen_task_cids.add(expected_task_cid)
        observed_contract_cid = str(record.pop("execution_contract_cid", ""))
        expected_contract_cid = _cid(
            {
                "schema": "EAAEFTaskExecutionContract@1",
                "task": record,
                "source_forest_root": population.source_forest_root,
            }
        )
        if observed_contract_cid != expected_contract_cid:
            raise EAAEFReconciliationIdentityError(
                f"compiled EAAEF task {task_alias} execution-contract CID differs"
            )
        contract_bindings.append(
            {
                "task_cid": expected_task_cid,
                "execution_contract_cid": expected_contract_cid,
            }
        )

    expected_contract_population_cid = _cid(
        {
            "schema": EAAEF_EXECUTION_CONTRACT_POPULATION_SCHEMA,
            "contracts": contract_bindings,
            "source_forest_root": population.source_forest_root,
        }
    )
    if population.execution_contract_population_cid != expected_contract_population_cid:
        raise EAAEFReconciliationIdentityError(
            "compiled EAAEF execution-contract population CID differs"
        )
    expected_goal_population_cid = _cid(
        {
            "schema": "EAAEFFreshGoalPopulation@1",
            "goals": list(population.goals),
            "goal_edges": list(population.goal_edges),
            "plan_r1": population.plan_r1,
            "source_forest_root": population.source_forest_root,
        }
    )
    if population.goal_population_cid != expected_goal_population_cid:
        raise EAAEFReconciliationIdentityError("compiled EAAEF goal population CID differs")
    expected_bootstrap_cid = _cid(
        {
            "schema": "EAAEFBootstrapPopulation@1",
            "tasks": population.bootstrap_tasks,
            "dependencies": population.dependencies,
            "source_forest_root": population.source_forest_root,
        }
    )
    if population.bootstrap_population_cid != expected_bootstrap_cid:
        raise EAAEFReconciliationIdentityError("compiled EAAEF bootstrap population CID differs")
    expected_plan_r2_population_cid = _cid(
        {
            "schema": "EAAEFPlanR2HeldPopulation@1",
            "tasks": population.plan_r2_tasks,
            "dependencies": population.dependencies,
            "source_forest_root": population.source_forest_root,
        }
    )
    if population.plan_r2_population_cid != expected_plan_r2_population_cid:
        raise EAAEFReconciliationIdentityError("compiled EAAEF held-R2 population CID differs")
    expected_population_cid = _cid(
        {
            "schema": EAAEF_POPULATION_SCHEMA,
            "board_cid": population.board_cid,
            "bootstrap_population_cid": expected_bootstrap_cid,
            "plan_r2_population_cid": expected_plan_r2_population_cid,
            "goal_population_cid": expected_goal_population_cid,
            "execution_contract_population_cid": expected_contract_population_cid,
            "source_forest_root": population.source_forest_root,
            "task_count": EAAEF_TASK_COUNT,
        }
    )
    if population.population_cid != expected_population_cid:
        raise EAAEFReconciliationIdentityError("compiled EAAEF overall population CID differs")
    try:
        observed_contract_counts = population.execution_contract_counts
    except (KeyError, TypeError) as exc:
        raise EAAEFReconciliationIdentityError(
            "compiled EAAEF execution-contract counts are malformed"
        ) from exc
    if observed_contract_counts != {
        "task_dependencies": 270,
        "task_outputs": 415,
        "task_validations": 117,
        "task_acceptance": 116,
    }:
        raise EAAEFReconciliationIdentityError(
            "compiled EAAEF execution-contract counts differ"
        )

    canonical = compile_fresh_eaaef_population(current_board, forest=sealed)
    if _canonical_bytes(vars(population)) != _canonical_bytes(vars(canonical)):
        raise EAAEFReconciliationIdentityError(
            "compiled EAAEF population differs from the current sealed board"
        )
    return canonical


def _assert_no_boundary_authority(value: Any, *, path: str = "request") -> None:
    """Reject database authority, raw SQL, and credentials at the owner seam."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            suffix_forbidden = any(
                normalized.endswith("_" + suffix) for suffix in ("password", "secret", "sql")
            ) or (normalized.endswith("_token") and normalized != "fencing_token")
            if normalized in _FORBIDDEN_BOUNDARY_KEYS or suffix_forbidden:
                raise EAAEFReconciliationIdentityError(
                    f"{path} exposes forbidden boundary field {key!r}"
                )
            _assert_no_boundary_authority(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_no_boundary_authority(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.casefold()
        if "control.duckdb" in lowered or "attach " in lowered or "select " in lowered:
            raise EAAEFReconciliationIdentityError(f"{path} exposes a database path or SQL text")


def _verify_self_addressed_object(
    value: Mapping[str, Any],
    *,
    fields: frozenset[str],
    cid_field: str,
    noun: str,
) -> dict[str, Any]:
    payload = dict(value)
    if set(payload) != fields:
        raise EAAEFReconciliationIdentityError(f"{noun} shape is not exact")
    claimed = str(payload.pop(cid_field, ""))
    if claimed != _cid(payload):
        raise EAAEFReconciliationIdentityError(f"{noun} self-address differs")
    payload[cid_field] = claimed
    return payload


def assemble_fresh_authority_bundle(
    *,
    authorization: Mapping[str, Any],
    plan_r2_operational_capability: Mapping[str, Any],
    plan_r2_remote_owner_capability: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble signed artifacts without selecting their trust roots."""

    value = {
        "schema": EAAEF_FRESH_AUTHORITY_SCHEMA,
        "authorization": dict(authorization),
        "plan_r2_operational_capability": dict(plan_r2_operational_capability),
        "plan_r2_remote_owner_capability": dict(plan_r2_remote_owner_capability),
    }
    value["authority_bundle_cid"] = _cid(value)
    return value


def load_fresh_authority_artifacts(
    *,
    authorization_path: str | Path,
    plan_r2_operational_capability_path: str | Path,
    plan_r2_remote_owner_capability_path: str | Path,
) -> dict[str, Any]:
    """Read only the three public signed artifacts and assemble their CID."""

    return assemble_fresh_authority_bundle(
        authorization=_json_object(
            Path(authorization_path).resolve(strict=True),
            noun="fresh signed Plan-R2 authorization",
        ),
        plan_r2_operational_capability=_json_object(
            Path(plan_r2_operational_capability_path).resolve(strict=True),
            noun="fresh signed Plan-R2 operational capability",
        ),
        plan_r2_remote_owner_capability=_json_object(
            Path(plan_r2_remote_owner_capability_path).resolve(strict=True),
            noun="fresh signed Plan-R2 remote-owner capability",
        ),
    )


def load_fresh_trust_roots(path: str | Path) -> dict[str, Any]:
    """Load independently configured verification roots, never signing keys."""

    payload = _json_object(Path(path).resolve(strict=True), noun="fresh EAAEF trust roots")
    if payload.get("schema") != EAAEF_FRESH_TRUST_SCHEMA:
        raise EAAEFReconciliationIdentityError("fresh EAAEF trust schema differs")
    return _verify_self_addressed_object(
        payload,
        fields=_FRESH_TRUST_FIELDS,
        cid_field="trust_bundle_cid",
        noun="fresh EAAEF trust roots",
    )


@dataclass(frozen=True)
class VerifiedFreshEAAEFAuthority:
    """Exact signed authority joined to independent trust and one population."""

    authority_bundle_cid: str
    trust_bundle_cid: str
    signed_population_cid: str
    authorization_cid: str
    operational_capability_cid: str
    remote_capability_cid: str
    admission: VerifiedPlanR2RemoteOwnerAdmission = field(repr=False)
    _bundle: Mapping[str, Any] = field(repr=False)

    def signed_bundle(self) -> dict[str, Any]:
        return json.loads(_canonical_bytes(dict(self._bundle)))


def _verified_trust_lists(trust_roots: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    trust = _verify_self_addressed_object(
        trust_roots,
        fields=_FRESH_TRUST_FIELDS,
        cid_field="trust_bundle_cid",
        noun="fresh EAAEF trust roots",
    )
    result: dict[str, tuple[str, ...]] = {}
    all_dids: list[str] = []
    for field_name in (
        "remote_reviewer_dids",
        "plan_r2_capability_reviewer_dids",
        "operator_dids",
        "security_reviewer_dids",
    ):
        raw = trust.get(field_name)
        if (
            not isinstance(raw, list)
            or not raw
            or any(not isinstance(item, str) or not _DID_RE.fullmatch(item) for item in raw)
            or raw != sorted(set(raw))
        ):
            raise EAAEFReconciliationIdentityError(
                f"fresh EAAEF trust roots {field_name} are not exact"
            )
        result[field_name] = tuple(raw)
        all_dids.extend(raw)
    if len(all_dids) != len(set(all_dids)):
        raise EAAEFReconciliationIdentityError(
            "fresh EAAEF trust roots reuse an identity across independent roles"
        )
    return result


def _fresh_plan_r2(population: CompiledEAAEFPopulation) -> dict[str, Any]:
    body = {
        "transition": "fresh_reconciliation_population",
        "predecessor_plan_cid": population.plan_r1_cid,
        "board_cid": population.board_cid,
        "reconciliation_population_cid": population.population_cid,
        "execution_contract_population_cid": population.execution_contract_population_cid,
        "bootstrap_population_cid": population.bootstrap_population_cid,
        "held_plan_r2_population_cid": population.plan_r2_population_cid,
        "source_forest_root": population.source_forest_root,
        "task_count": EAAEF_TASK_COUNT,
        "bootstrap_task_count": EAAEF_BOOTSTRAP_TASK_COUNT,
        "held_task_count": EAAEF_PLAN_R2_TASK_COUNT,
        "terminal_statuses_imported": 0,
    }
    plan_root_cid = _cid(
        {
            "schema": "EAAEFFreshPlanR2Root@1",
            "semantic_root_cid": population.source_forest_root,
            "body": body,
            "revision": 2,
        }
    )
    plan_cid = _cid(
        {
            "schema": "EAAEFFreshPlanR2Identity@1",
            "plan_alias": EAAEF_PLAN_R2_ALIAS,
            "plan_root_cid": plan_root_cid,
        }
    )
    return {
        "plan_cid": plan_cid,
        "plan_alias": EAAEF_PLAN_R2_ALIAS,
        "plan_root_cid": plan_root_cid,
        "semantic_root_cid": population.source_forest_root,
        "status": "active",
        "revision": 2,
        "body": body,
    }


def load_fresh_bootstrap_snapshot(path: str | Path) -> dict[str, Any]:
    payload = _json_object(Path(path).resolve(strict=True), noun="fresh bootstrap owner snapshot")
    if payload.get("schema") != EAAEF_BOOTSTRAP_SNAPSHOT_SCHEMA:
        raise EAAEFReconciliationIdentityError("fresh bootstrap owner snapshot schema differs")
    return _verify_self_addressed_object(
        payload,
        fields=_BOOTSTRAP_SNAPSHOT_FIELDS,
        cid_field="snapshot_cid",
        noun="fresh bootstrap owner snapshot",
    )


def _fresh_plan_r2_frontier(tasks: Sequence[Mapping[str, Any]]) -> list[str]:
    candidates = [
        item
        for item in tasks
        if item.get("status") == "todo"
        and isinstance(item.get("body"), Mapping)
        and not list(item["body"].get("dependency_task_cids") or ())
    ]
    chosen: list[Mapping[str, Any]] = []
    for candidate in candidates:
        body = candidate["body"]
        reads = set(body.get("read_scope") or ())
        writes = set(body.get("write_scope") or ())
        effects = set(body.get("effect_scope") or ())
        conflicts = False
        for existing in chosen:
            other = existing["body"]
            other_reads = set(other.get("read_scope") or ())
            other_writes = set(other.get("write_scope") or ())
            other_effects = set(other.get("effect_scope") or ())
            if (
                writes & (other_reads | other_writes)
                or other_writes & reads
                or effects & other_effects
            ):
                conflicts = True
                break
        if not conflicts:
            chosen.append(candidate)
        if len(chosen) == 5:
            break
    frontier = sorted(str(item["task_cid"]) for item in chosen)
    if not frontier:
        raise EAAEFReconciliationIdentityError(
            "fresh EAAEF population has no conflict-free Plan-R2 frontier"
        )
    return frontier


def build_unsigned_fresh_plan_r2_statement(
    *,
    population: CompiledEAAEFPopulation,
    bootstrap_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical unsigned statement for an external signing ceremony."""

    snapshot = _verify_self_addressed_object(
        bootstrap_snapshot,
        fields=_BOOTSTRAP_SNAPSHOT_FIELDS,
        cid_field="snapshot_cid",
        noun="fresh bootstrap owner snapshot",
    )
    expected = {
        "schema": EAAEF_BOOTSTRAP_SNAPSHOT_SCHEMA,
        "source_head": population.source_head,
        "source_tree": population.source_tree,
        "source_forest_root": population.source_forest_root,
        "board_cid": population.board_cid,
        "reconciliation_population_cid": population.population_cid,
        "bootstrap_population_cid": population.bootstrap_population_cid,
        "bootstrap_task_count": EAAEF_BOOTSTRAP_TASK_COUNT,
        "held_task_count": EAAEF_PLAN_R2_TASK_COUNT,
        "terminal_statuses_imported": 0,
        "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
        "bootstrap_owner_absent_during_materialization": True,
        "owner_started_after_bootstrap": True,
        "direct_database_mutation_after_owner_start": False,
        "expected_active_plan_cid": population.plan_r1_cid,
        "expected_active_plan_root_cid": population.plan_r1_cid,
        "expected_active_plan_revision": 1,
        "expected_semantic_root_cid": population.source_forest_root,
    }
    mismatched = sorted(
        field_name
        for field_name, expected_value in expected.items()
        if snapshot.get(field_name) != expected_value
    )
    if mismatched:
        raise EAAEFReconciliationIdentityError(
            "fresh bootstrap owner snapshot differs: " + ", ".join(mismatched)
        )
    plan = _fresh_plan_r2(population)
    tasks = list(population.plan_r2_transition_tasks(plan_cid=str(plan["plan_cid"])))
    dependencies = sorted(
        (dict(item) for item in population.dependencies),
        key=lambda item: (
            str(item["task_cid"]),
            str(item["dependency_task_cid"]),
            str(item["kind"]),
        ),
    )
    frontier = _fresh_plan_r2_frontier(tasks)
    try:
        statement = prepare_plan_r2_transition_authorization(
            board_namespace=EAAEF_BOARD_NAMESPACE,
            source_head=population.source_head,
            source_tree=population.source_tree,
            source_generation_cid=population.source_forest_root,
            bootstrap_admission_cid=str(snapshot["bootstrap_admission_cid"]),
            r1_launch_capsule_cid=str(snapshot["r1_launch_capsule_cid"]),
            quack_owner_qualification_cid=str(snapshot["quack_owner_qualification_cid"]),
            quack_command_fabric_qualification_cid=str(
                snapshot["quack_command_fabric_qualification_cid"]
            ),
            owner_principal_did=str(snapshot["owner_principal_did"]),
            shard_id=str(snapshot["shard_id"]),
            store_id=str(snapshot["store_id"]),
            owner_generation=int(snapshot["owner_generation"]),
            expected_epoch=int(snapshot["expected_epoch"]),
            fencing_token=int(snapshot["fencing_token"]),
            lease_id=str(snapshot["lease_id"]),
            expected_version=int(snapshot["expected_version"]),
            expected_active_plan_cid=population.plan_r1_cid,
            expected_active_plan_root_cid=population.plan_r1_cid,
            expected_active_plan_revision=1,
            expected_event_cursor=str(snapshot["expected_event_cursor"]),
            expected_semantic_root_cid=population.source_forest_root,
            new_plan=plan,
            tasks=tasks,
            dependencies=dependencies,
            protected_tasks=[],
            frontier_task_cids=frontier,
            delta_cid=_cid(
                {
                    "schema": "EAAEFFreshPlanR2Delta@1",
                    "before_plan_cid": population.plan_r1_cid,
                    "after_plan_cid": plan["plan_cid"],
                    "population_cid": population.population_cid,
                    "frontier_task_cids": frontier,
                }
            ),
            request_id=str(snapshot["request_id"]),
            idempotency_key=str(snapshot["idempotency_key"]),
            deadline_ms=int(snapshot["deadline_ms"]),
            issued_at_ms=int(snapshot["issued_at_ms"]),
            expires_at_ms=int(snapshot["expires_at_ms"]),
            one_use_nonce=str(snapshot["one_use_nonce"]),
        )
    except (ExternalAgentPlanR2Error, KeyError, TypeError, ValueError) as exc:
        raise EAAEFReconciliationIdentityError(
            "fresh bootstrap snapshot cannot produce a Plan-R2 statement"
        ) from exc
    encoded = _canonical_bytes(statement)
    if len(encoded) > (MAX_PLAN_R2_REMOTE_REQUEST_BYTES - _PLAN_R2_REMOTE_REQUEST_OVERHEAD_RESERVE):
        raise EAAEFReconciliationBlocked(
            "fresh Plan-R2 statement leaves insufficient canonical remote-envelope headroom"
        )
    return statement


def build_unsigned_fresh_authority_request(
    *,
    population: CompiledEAAEFPopulation,
    bootstrap_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe or materialize the unsigned inputs; never read a signing key."""

    statement = (
        build_unsigned_fresh_plan_r2_statement(
            population=population,
            bootstrap_snapshot=bootstrap_snapshot,
        )
        if bootstrap_snapshot is not None
        else None
    )
    value = {
        "schema": EAAEF_UNSIGNED_AUTHORITY_REQUEST_SCHEMA,
        "board_namespace": EAAEF_BOARD_NAMESPACE,
        "source_head": population.source_head,
        "source_tree": population.source_tree,
        "source_forest_root": population.source_forest_root,
        "board_cid": population.board_cid,
        "reconciliation_population_cid": population.population_cid,
        "bootstrap_population_cid": population.bootstrap_population_cid,
        "held_plan_r2_population_cid": population.plan_r2_population_cid,
        "task_count": EAAEF_TASK_COUNT,
        "bootstrap_task_count": EAAEF_BOOTSTRAP_TASK_COUNT,
        "held_task_count": EAAEF_PLAN_R2_TASK_COUNT,
        "bootstrap_snapshot_schema": EAAEF_BOOTSTRAP_SNAPSHOT_SCHEMA,
        "bootstrap_snapshot_required_fields": sorted(_BOOTSTRAP_SNAPSHOT_FIELDS),
        "unsigned_plan_r2_statement": statement,
        "required_external_signatures": [
            "independent_operator",
            "independent_security_reviewer",
            "independent_plan_r2_capability_reviewer",
            "independent_plan_r2_remote_transport_reviewer",
        ],
        "signing_key_read": False,
        "authority_mutated": False,
        "provider_process_started": False,
    }
    value["request_cid"] = _cid(value)
    return value


def verify_fresh_authority_bundle(
    authority: Mapping[str, Any],
    *,
    population: CompiledEAAEFPopulation,
    trust_roots: Mapping[str, Any],
    now_ms: int | None = None,
) -> VerifiedFreshEAAEFAuthority:
    """Verify fresh, independently signed Plan-R2 and remote-owner authority."""

    if authority.get("schema") != EAAEF_FRESH_AUTHORITY_SCHEMA:
        raise EAAEFReconciliationIdentityError("fresh EAAEF authority schema differs")
    bundle = _verify_self_addressed_object(
        authority,
        fields=_FRESH_AUTHORITY_FIELDS,
        cid_field="authority_bundle_cid",
        noun="fresh EAAEF authority bundle",
    )
    trust = _verified_trust_lists(trust_roots)
    authorization = bundle.get("authorization")
    capability = bundle.get("plan_r2_operational_capability")
    remote = bundle.get("plan_r2_remote_owner_capability")
    if not all(isinstance(item, Mapping) for item in (authorization, capability, remote)):
        raise EAAEFReconciliationIdentityError("fresh EAAEF authority is incomplete")
    new_plan = authorization.get("new_plan")
    if not isinstance(new_plan, Mapping):
        raise EAAEFReconciliationIdentityError("fresh Plan-R2 authorization has no exact plan")
    expected_plan = _fresh_plan_r2(population)
    plan_cid = str(expected_plan["plan_cid"])
    if (
        authorization.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or authorization.get("source_head") != population.source_head
        or authorization.get("source_tree") != population.source_tree
        or authorization.get("source_generation_cid") != population.source_forest_root
        or authorization.get("expected_active_plan_cid") != population.plan_r1_cid
        or authorization.get("expected_active_plan_root_cid") != population.plan_r1_cid
        or authorization.get("expected_active_plan_revision") != 1
        or new_plan != expected_plan
    ):
        raise EAAEFReconciliationIdentityError(
            "Plan-R2 authorization is stale or belongs to another source"
        )
    raw_tasks = authorization.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != EAAEF_TASK_COUNT:
        raise EAAEFReconciliationIdentityError(
            "Plan-R2 authorization must bind all 116 fresh tasks"
        )
    expected_tasks = list(population.plan_r2_transition_tasks(plan_cid=plan_cid))
    if raw_tasks != expected_tasks:
        raise EAAEFReconciliationIdentityError(
            "Plan-R2 authorization task population differs from the board"
        )
    if any(str(item.get("status") or "").casefold() in _TERMINAL_STATUSES for item in raw_tasks):
        raise EAAEFReconciliationIdentityError(
            "fresh Plan-R2 authority imports a terminal task status"
        )
    expected_dependencies = sorted(
        (dict(item) for item in population.dependencies),
        key=lambda item: (
            str(item["task_cid"]),
            str(item["dependency_task_cid"]),
            str(item["kind"]),
        ),
    )
    if authorization.get("dependencies") != expected_dependencies:
        raise EAAEFReconciliationIdentityError(
            "Plan-R2 authorization dependency population differs from the board"
        )
    if authorization.get("protected_tasks") != []:
        raise EAAEFReconciliationIdentityError(
            "fresh Plan-R2 authorization imports protected historical rows"
        )
    try:
        admitted = verify_plan_r2_remote_owner_admission(
            remote,
            plan_r2_operational_capability=capability,
            authorization=authorization,
            trusted_remote_reviewer_dids=trust["remote_reviewer_dids"],
            trusted_plan_r2_capability_reviewer_dids=trust["plan_r2_capability_reviewer_dids"],
            trusted_operator_dids=trust["operator_dids"],
            trusted_security_reviewer_dids=trust["security_reviewer_dids"],
            now_ms=int(now_ms if now_ms is not None else time.time_ns() // 1_000_000),
        )
    except PlanR2RemoteOwnerAdmissionError as exc:
        raise EAAEFReconciliationIdentityError(
            "fresh EAAEF signed authority chain was rejected"
        ) from exc
    return VerifiedFreshEAAEFAuthority(
        authority_bundle_cid=str(bundle["authority_bundle_cid"]),
        trust_bundle_cid=str(trust_roots["trust_bundle_cid"]),
        signed_population_cid=str(authorization.get("population_cid") or ""),
        authorization_cid=str(authorization.get("authorization_cid") or ""),
        operational_capability_cid=str(capability.get("capability_cid") or ""),
        remote_capability_cid=admitted.capability_cid,
        admission=admitted,
        _bundle=MappingProxyType(json.loads(_canonical_bytes(bundle))),
    )


def build_typed_owner_materialization_request(
    *,
    generation_id: str,
    population: CompiledEAAEFPopulation,
    authority: VerifiedFreshEAAEFAuthority,
    offline_population_receipt_cid: str,
) -> dict[str, Any]:
    """Build the closed CID-only request that may cross into the owner."""

    if not _GENERATION_ID_RE.fullmatch(generation_id):
        raise EAAEFReconciliationIdentityError("fresh generation id is invalid")
    if type(authority) is not VerifiedFreshEAAEFAuthority:
        raise EAAEFReconciliationIdentityError("fresh EAAEF authority was not verified")
    request = {
        "schema": EAAEF_OWNER_REQUEST_SCHEMA,
        "interface": EAAEF_RECONCILIATION_OWNER_INTERFACE,
        "operation": "apply_signed_plan_r2_to_offline_population",
        "generation_id": generation_id,
        "board_namespace": EAAEF_BOARD_NAMESPACE,
        "source_head": population.source_head,
        "source_tree": population.source_tree,
        "source_forest_root": population.source_forest_root,
        "board_cid": population.board_cid,
        "population_cid": population.population_cid,
        "goal_population_cid": population.goal_population_cid,
        "execution_contract_population_cid": population.execution_contract_population_cid,
        "bootstrap_population_cid": population.bootstrap_population_cid,
        "plan_r2_population_cid": population.plan_r2_population_cid,
        "plan_r1_cid": population.plan_r1_cid,
        "expected_goal_count": EAAEF_GOAL_COUNT,
        "expected_goal_edge_count": EAAEF_GOAL_EDGE_COUNT,
        "expected_plan_count": 1,
        "bootstrap_task_count": EAAEF_BOOTSTRAP_TASK_COUNT,
        "plan_r2_task_count": EAAEF_PLAN_R2_TASK_COUNT,
        "expected_task_count": EAAEF_TASK_COUNT,
        "expected_execution_contract_counts": population.execution_contract_counts,
        "signed_plan_r2_population_cid": authority.signed_population_cid,
        "plan_r2_authorization_cid": authority.authorization_cid,
        "plan_r2_operational_capability_cid": authority.operational_capability_cid,
        "plan_r2_remote_owner_capability_cid": authority.remote_capability_cid,
        "authority_bundle_cid": authority.authority_bundle_cid,
        "trust_bundle_cid": authority.trust_bundle_cid,
        "offline_population_receipt_cid": offline_population_receipt_cid,
        "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
        "direct_database_mutation_after_owner_start_allowed": False,
        "historical_generation_import_allowed": False,
        "historical_status_overlay_allowed": False,
        "provider_launch_allowed": False,
    }
    if any(
        not _SHA256_RE.fullmatch(str(request[field] or ""))
        for field in (
            "source_forest_root",
            "board_cid",
            "population_cid",
            "goal_population_cid",
            "execution_contract_population_cid",
            "bootstrap_population_cid",
            "plan_r2_population_cid",
            "plan_r1_cid",
            "signed_plan_r2_population_cid",
            "plan_r2_authorization_cid",
            "plan_r2_operational_capability_cid",
            "plan_r2_remote_owner_capability_cid",
            "authority_bundle_cid",
            "trust_bundle_cid",
            "offline_population_receipt_cid",
        )
    ):
        raise EAAEFReconciliationIdentityError("typed owner request has an incomplete CID binding")
    _assert_no_boundary_authority(request)
    request["request_cid"] = _cid(request)
    return request


def apply_plan_r2_through_existing_repository(
    repository: ExternalAgentStateRepository,
    authority: VerifiedFreshEAAEFAuthority,
) -> dict[str, Any]:
    """Apply prepare/apply/observe through the existing exact repository seam."""

    if type(repository) is not ExternalAgentStateRepository:
        raise EAAEFReconciliationIdentityError(
            "Plan R2 requires exact ExternalAgentStateRepository"
        )
    if type(authority) is not VerifiedFreshEAAEFAuthority:
        raise EAAEFReconciliationIdentityError("Plan-R2 authority is not verified")
    gateway = repository.owner_gateway
    if (
        type(gateway) is not PlanR2ProcessRemoteOwnerGateway
        or gateway.remote_capability_cid != authority.remote_capability_cid
        or gateway.production_capability_cid != authority.operational_capability_cid
        or repository.capability_cid != authority.operational_capability_cid
        or repository.board_namespace != EAAEF_BOARD_NAMESPACE
        or repository.shard_id != authority.admission["shard_id"]
        or repository.store_id != authority.admission["store_id"]
        or repository.owner_generation != authority.admission["owner_generation"]
        or repository.owner_epoch != authority.admission["epoch"]
        or repository.fence_epoch != authority.admission["fence"]
    ):
        raise EAAEFReconciliationIdentityError(
            "Plan-R2 repository is not the exact qualified process-remote seam"
        )
    authorization = authority.signed_bundle()["authorization"]
    repository.attach()
    try:
        prepared = repository.prepare_authorized_plan_r2_transition(authorization)
        receipt = repository.apply_authorized_plan_r2_transition(authorization, prepared)
        observation = repository.observe_authorized_plan_r2_transition(authorization, receipt)
    finally:
        repository.close()
    return {
        "prepared": dict(prepared),
        "receipt": dict(receipt),
        "observation": dict(observation),
    }


@runtime_checkable
class EAAEFTypedReconciliationOwner(Protocol):
    """Exact final-CASF adapter; no database authority crosses this protocol."""

    INTERFACE: str

    def reconciliation_qualification(self) -> Mapping[str, Any]:
        """Return exact bootstrap/Plan-R2/status/stop boundary evidence."""

    def materialize_offline_population(
        self,
        request: Mapping[str, Any],
        *,
        population: CompiledEAAEFPopulation,
    ) -> Mapping[str, Any]:
        """Materialize 22+94 while owner-absent, then start the owner."""

    def apply_signed_plan_r2(
        self,
        request: Mapping[str, Any],
        *,
        population: CompiledEAAEFPopulation,
        authority: VerifiedFreshEAAEFAuthority,
    ) -> Mapping[str, Any]:
        """Apply signed R2 through the live remote repository seam."""

    def launch_reconciliation_supervisor(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Launch the selected paused or full Plan-R2 supervisor mode."""

    def reconciliation_status_snapshot(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the authoritative owner status for one exact generation."""

    def stop_reconciliation_tracks(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Stop only owner-tracked births and return a durable typed receipt."""


def require_typed_reconciliation_owner(
    owner: object | None,
    *,
    source_forest_root: str = "",
) -> EAAEFTypedReconciliationOwner:
    if (
        owner is None
        or not isinstance(owner, EAAEFTypedReconciliationOwner)
        or getattr(owner, "INTERFACE", "") != EAAEF_RECONCILIATION_OWNER_INTERFACE
    ):
        raise EAAEFReconciliationBlocked(
            "typed_portfolio_materialization_owner_unavailable: final CASF must provide "
            "EAAEFTypedReconciliationOwner@1 over its exclusive typed owner"
        )
    qualification = owner.reconciliation_qualification()
    if not isinstance(qualification, Mapping):
        raise EAAEFReconciliationBlocked("typed reconciliation owner has no qualification")
    value = dict(qualification)
    body = dict(value)
    qualification_cid = str(body.pop("qualification_cid", ""))
    required = {
        "schema": EAAEF_OWNER_QUALIFICATION_SCHEMA,
        "interface": EAAEF_RECONCILIATION_OWNER_INTERFACE,
        "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
        "bootstrap_materialization_before_owner_start": True,
        "offline_population_includes_execution_contracts": True,
        "direct_database_mutation_after_owner_start": False,
        "typed_task_source_interface": EAAEF_TYPED_TASK_SOURCE_INTERFACE,
        "plan_r2_repository_interface": AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE,
        "plan_r2_remote_gateway_interface": PLAN_R2_REMOTE_CLIENT_GATEWAY_INTERFACE,
        "plan_r2_wire_channel_interface": PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
        "plan_r2_remote_runtime_qualification_status": "production_qualified",
        "plan_r2_remote_runtime_blockers": [],
        "status_operation": "status.snapshot",
        "stop_tracks_operation": "stop_tracks",
        "launch_modes": ["paused", "plan_r2"],
        "database_authority_crossing_allowed": False,
        "filesystem_path_authority_crossing_allowed": False,
        "transport_token_authority_crossing_allowed": False,
        "sql_crossing_allowed": False,
        "provider_launch_allowed": True,
    }
    mismatched = sorted(
        field_name for field_name, expected in required.items() if value.get(field_name) != expected
    )
    observed_forest = str(value.get("source_forest_root") or "")
    if not _SHA256_RE.fullmatch(observed_forest) or (
        source_forest_root and observed_forest != source_forest_root
    ):
        mismatched.append("source_forest_root")
    if set(value) != _OWNER_QUALIFICATION_FIELDS or qualification_cid != _cid(body) or mismatched:
        raise EAAEFReconciliationBlocked(
            "typed reconciliation owner qualification differs: "
            + ", ".join(sorted(set(mismatched or ["shape_or_cid"])))
        )
    _assert_no_boundary_authority(value)
    return owner


def resolve_production_reconciliation_owner(
    repo_root: str | Path,
) -> EAAEFTypedReconciliationOwner:
    """Resolve the one statically named final-CASF adapter.

    No module, class, database, or credential name is accepted from argv or a
    task payload.  The EAAEF-only branch provides only a statically named
    blocker facade; the final CASF integration must bind the exact qualified
    effect adapter behind this opener.
    """

    module_name = (
        "ipfs_accelerate_py.agent_supervisor.task_sources.typed_eaaef_reconciliation_owner"
    )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise EAAEFReconciliationBlocked(
                "typed_portfolio_materialization_owner_unavailable: missing "
                "typed_eaaef_reconciliation_owner final-CASF adapter"
            ) from exc
        raise
    opener = getattr(module, "open_eaaef_typed_reconciliation_owner", None)
    if not callable(opener):
        raise EAAEFReconciliationBlocked(
            "typed_portfolio_materialization_owner_unavailable: exact opener is absent"
        )
    return require_typed_reconciliation_owner(
        opener(repo_root=Path(repo_root).resolve(strict=True))
    )


@dataclass(frozen=True)
class ProcessBirth:
    pid: int
    start_time_ticks: int
    parent_pid: int
    boot_id: str
    argv_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "start_time_ticks": self.start_time_ticks,
            "parent_pid": self.parent_pid,
            "boot_id": self.boot_id,
            "argv_sha256": self.argv_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProcessBirth:
        try:
            birth = cls(
                pid=int(value["pid"]),
                start_time_ticks=int(value["start_time_ticks"]),
                parent_pid=int(value["parent_pid"]),
                boot_id=str(value["boot_id"]),
                argv_sha256=str(value["argv_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EAAEFReconciliationIdentityError("process birth receipt is malformed") from exc
        if (
            birth.pid <= 1
            or birth.start_time_ticks <= 0
            or birth.parent_pid <= 0
            or not birth.boot_id
            or not _SHA256_RE.fullmatch(birth.argv_sha256)
        ):
            raise EAAEFReconciliationIdentityError("process birth receipt is invalid")
        return birth


def inspect_process_birth(pid: int) -> ProcessBirth | None:
    """Return the exact live Linux birth tuple, or ``None`` when absent."""

    if isinstance(pid, bool) or int(pid) <= 1:
        return None
    process_id = int(pid)
    proc = Path("/proc") / str(process_id)
    try:
        raw_stat = (proc / "stat").read_text(encoding="utf-8")
        close = raw_stat.rindex(")")
        fields = raw_stat[close + 1 :].strip().split()
        parent_pid = int(fields[1])
        start_ticks = int(fields[19])
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        argv = (proc / "cmdline").read_bytes()
    except (IndexError, OSError, UnicodeError, ValueError):
        return None
    if not argv or not argv.endswith(b"\0"):
        return None
    return ProcessBirth(
        pid=process_id,
        start_time_ticks=start_ticks,
        parent_pid=parent_pid,
        boot_id=boot_id,
        argv_sha256="sha256:" + hashlib.sha256(argv).hexdigest(),
    )


class ReconciliationStateStore:
    """Private, symlink-safe local registry; never a task-status authority."""

    def __init__(self, root: str | Path) -> None:
        raw = Path(root).expanduser()
        try:
            if stat.S_ISLNK(os.lstat(raw).st_mode):
                raise EAAEFReconciliationIdentityError(
                    "reconciliation state root must not be a symlink"
                )
        except FileNotFoundError:
            pass
        selected = raw.resolve(strict=False)
        self.root = selected
        self.cursor_path = selected / "active-generation.json"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = os.lstat(self.root)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise EAAEFReconciliationIdentityError("reconciliation state root is not a directory")
        if metadata.st_uid != os.geteuid():
            raise EAAEFReconciliationIdentityError("reconciliation state root owner differs")
        os.chmod(self.root, 0o700)

    def generation_dir(self, generation_id: str) -> Path:
        if not _GENERATION_ID_RE.fullmatch(generation_id):
            raise EAAEFReconciliationIdentityError("fresh generation id is invalid")
        return self.root / "generations" / generation_id

    def state_path(self, generation_id: str) -> Path:
        return self.generation_dir(generation_id) / "state.json"

    def create_generation(self, generation_id: str, state: Mapping[str, Any]) -> None:
        self.initialize()
        directory = self.generation_dir(generation_id)
        try:
            directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        except FileExistsError as exc:
            raise EAAEFReconciliationBlocked("fresh generation id already exists") from exc
        os.chmod(directory, 0o700)
        self.write_state(generation_id, state)

    def _write_private_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        parent_metadata = os.lstat(path.parent)
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or stat.S_ISLNK(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.geteuid()
        ):
            raise EAAEFReconciliationIdentityError("private state parent is unsafe")
        os.chmod(path.parent, 0o700)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temporary_path = Path(temporary)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(dict(payload), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def write_state(self, generation_id: str, state: Mapping[str, Any]) -> None:
        body = dict(state)
        state_cid = str(body.pop("state_cid", ""))
        if (
            state.get("schema") != EAAEF_STATE_SCHEMA
            or state.get("generation_id") != generation_id
            or state_cid != _cid(body)
        ):
            raise EAAEFReconciliationIdentityError("generation state identity differs")
        self._write_private_json(self.state_path(generation_id), state)

    def read_state(self, generation_id: str) -> dict[str, Any]:
        state = _private_json_object(
            self.state_path(generation_id),
            noun="EAAEF generation state",
        )
        body = dict(state)
        state_cid = str(body.pop("state_cid", ""))
        if (
            state.get("schema") != EAAEF_STATE_SCHEMA
            or state.get("generation_id") != generation_id
            or state_cid != _cid(body)
        ):
            raise EAAEFReconciliationIdentityError("EAAEF generation state CID differs")
        return state

    def activate(self, generation_id: str, *, state_cid: str) -> None:
        if not _SHA256_RE.fullmatch(state_cid):
            raise EAAEFReconciliationIdentityError("EAAEF state CID is malformed")
        payload = {
            "schema": EAAEF_CURSOR_SCHEMA,
            "generation_id": generation_id,
            "state_cid": state_cid,
        }
        payload["cursor_cid"] = _cid(payload)
        self._write_private_json(self.cursor_path, payload)

    def active_generation(self) -> str:
        try:
            os.lstat(self.cursor_path)
        except FileNotFoundError:
            return ""
        cursor = _private_json_object(self.cursor_path, noun="EAAEF generation cursor")
        body = dict(cursor)
        cursor_cid = str(body.pop("cursor_cid", ""))
        if cursor.get("schema") != EAAEF_CURSOR_SCHEMA or cursor_cid != _cid(body):
            raise EAAEFReconciliationIdentityError("EAAEF generation cursor schema differs")
        generation_id = str(cursor.get("generation_id") or "")
        if not _GENERATION_ID_RE.fullmatch(generation_id):
            raise EAAEFReconciliationIdentityError("EAAEF generation cursor is malformed")
        state = self.read_state(generation_id)
        if cursor.get("state_cid") != state.get("state_cid"):
            raise EAAEFReconciliationIdentityError("EAAEF generation cursor state CID differs")
        return generation_id

    def deactivate(self, generation_id: str) -> None:
        try:
            os.lstat(self.cursor_path)
        except FileNotFoundError:
            return
        if self.active_generation() != generation_id:
            raise EAAEFReconciliationIdentityError("active generation changed during stop")
        self.cursor_path.unlink()

    def cleanup_runtime_artifacts(self, generation_id: str) -> list[str]:
        directory = self.generation_dir(generation_id)
        removed: list[str] = []
        for name in sorted(_RUNTIME_ARTIFACT_NAMES):
            path = directory / name
            try:
                metadata = os.lstat(path)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise EAAEFReconciliationIdentityError(
                    f"runtime cleanup target {name} has an unsafe type"
                )
            path.unlink()
            removed.append(name)
        return removed


def _new_generation_id(source_head: str) -> str:
    nonce = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    return f"eaaef-{source_head[:12]}-{nonce}"


def _build_offline_population_request(
    *,
    generation_id: str,
    population: CompiledEAAEFPopulation,
) -> dict[str, Any]:
    if not _GENERATION_ID_RE.fullmatch(generation_id):
        raise EAAEFReconciliationIdentityError("fresh generation id is invalid")
    request = {
        "schema": EAAEF_OFFLINE_POPULATION_REQUEST_SCHEMA,
        "interface": EAAEF_RECONCILIATION_OWNER_INTERFACE,
        "operation": "materialize_offline_22_plus_94_then_start_owner",
        "generation_id": generation_id,
        "board_namespace": EAAEF_BOARD_NAMESPACE,
        "source_head": population.source_head,
        "source_tree": population.source_tree,
        "source_forest_root": population.source_forest_root,
        "board_cid": population.board_cid,
        "population_cid": population.population_cid,
        "goal_population_cid": population.goal_population_cid,
        "execution_contract_population_cid": population.execution_contract_population_cid,
        "bootstrap_population_cid": population.bootstrap_population_cid,
        "held_plan_r2_population_cid": population.plan_r2_population_cid,
        "plan_r1_cid": population.plan_r1_cid,
        "expected_goal_count": EAAEF_GOAL_COUNT,
        "expected_goal_edge_count": EAAEF_GOAL_EDGE_COUNT,
        "expected_plan_count": 1,
        "bootstrap_task_count": EAAEF_BOOTSTRAP_TASK_COUNT,
        "held_task_count": EAAEF_PLAN_R2_TASK_COUNT,
        "expected_task_count": EAAEF_TASK_COUNT,
        "expected_execution_contract_counts": population.execution_contract_counts,
        "owner_must_be_absent_during_population_write": True,
        "owner_start_allowed_only_after_population_commit": True,
        "direct_database_mutation_after_owner_start_allowed": False,
        "provider_launch_allowed": False,
    }
    _assert_no_boundary_authority(request)
    request["request_cid"] = _cid(request)
    return request


def _validate_offline_population_receipt(
    receipt: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    population: CompiledEAAEFPopulation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = dict(receipt)
    required = {
        "schema": EAAEF_OFFLINE_POPULATION_RECEIPT_SCHEMA,
        "interface": EAAEF_RECONCILIATION_OWNER_INTERFACE,
        "request_cid": request["request_cid"],
        "generation_id": request["generation_id"],
        "source_forest_root": population.source_forest_root,
        "population_cid": population.population_cid,
        "goal_population_cid": population.goal_population_cid,
        "execution_contract_population_cid": population.execution_contract_population_cid,
        "bootstrap_population_cid": population.bootstrap_population_cid,
        "held_plan_r2_population_cid": population.plan_r2_population_cid,
        "plan_r1_cid": population.plan_r1_cid,
        "goal_count": EAAEF_GOAL_COUNT,
        "goal_edge_count": EAAEF_GOAL_EDGE_COUNT,
        "plan_count": 1,
        "task_count": EAAEF_TASK_COUNT,
        "bootstrap_task_count": EAAEF_BOOTSTRAP_TASK_COUNT,
        "held_task_count": EAAEF_PLAN_R2_TASK_COUNT,
        "task_status_counts": {
            "blocked": EAAEF_PLAN_R2_TASK_COUNT,
            "todo": EAAEF_BOOTSTRAP_TASK_COUNT,
        },
        "execution_contract_counts": population.execution_contract_counts,
        "execution_contracts_materialized": True,
        "terminal_statuses_imported": 0,
        "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
        "bootstrap_owner_absent_during_materialization": True,
        "owner_started_after_bootstrap": True,
        "direct_database_mutation_after_owner_start": False,
        "provider_process_started": False,
    }
    mismatched = sorted(
        field_name for field_name, expected in required.items() if value.get(field_name) != expected
    )
    raw_snapshot = value.get("bootstrap_snapshot")
    if set(value) != {*required, "bootstrap_snapshot", "receipt_cid"}:
        mismatched.append("shape")
    if not isinstance(raw_snapshot, Mapping):
        mismatched.append("bootstrap_snapshot")
    receipt_cid = str(value.get("receipt_cid") or "")
    body = {key: item for key, item in value.items() if key != "receipt_cid"}
    if receipt_cid != _cid(body):
        mismatched.append("receipt_cid")
    if mismatched:
        raise EAAEFReconciliationIdentityError(
            "offline population receipt differs: " + ", ".join(sorted(set(mismatched)))
        )
    _assert_no_boundary_authority(value, path="offline_population_receipt")
    snapshot = _verify_self_addressed_object(
        raw_snapshot,
        fields=_BOOTSTRAP_SNAPSHOT_FIELDS,
        cid_field="snapshot_cid",
        noun="fresh bootstrap owner snapshot",
    )
    build_unsigned_fresh_plan_r2_statement(
        population=population,
        bootstrap_snapshot=snapshot,
    )
    return value, snapshot


def _validate_owner_receipt(
    receipt: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(receipt)
    required = {
        "schema": EAAEF_OWNER_RECEIPT_SCHEMA,
        "interface": EAAEF_RECONCILIATION_OWNER_INTERFACE,
        "request_cid": request["request_cid"],
        "generation_id": request["generation_id"],
        "source_forest_root": request["source_forest_root"],
        "population_cid": request["population_cid"],
        "goal_population_cid": request["goal_population_cid"],
        "execution_contract_population_cid": request[
            "execution_contract_population_cid"
        ],
        "plan_r1_cid": request["plan_r1_cid"],
        "signed_plan_r2_population_cid": request["signed_plan_r2_population_cid"],
        "offline_population_receipt_cid": request["offline_population_receipt_cid"],
        "bootstrap_task_count": EAAEF_BOOTSTRAP_TASK_COUNT,
        "plan_r2_task_count": EAAEF_PLAN_R2_TASK_COUNT,
        "task_count": EAAEF_TASK_COUNT,
        "goal_count": EAAEF_GOAL_COUNT,
        "goal_edge_count": EAAEF_GOAL_EDGE_COUNT,
        "plan_count": 2,
        "plan_alias": EAAEF_PLAN_R2_ALIAS,
        "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
        "bootstrap_owner_absent_during_materialization": True,
        "owner_started_after_bootstrap": True,
        "direct_database_mutation_after_owner_start": False,
        "plan_r2_repository_interface": AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE,
        "plan_r2_remote_gateway_interface": PLAN_R2_REMOTE_CLIENT_GATEWAY_INTERFACE,
        "plan_r2_wire_channel_interface": PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
        "typed_task_source_interface": EAAEF_TYPED_TASK_SOURCE_INTERFACE,
        "plan_r2_prepare_apply_observe_complete": True,
        "task_status_counts": {"todo": EAAEF_TASK_COUNT},
        "execution_contract_counts": request["expected_execution_contract_counts"],
        "execution_contracts_materialized": True,
        "historical_statuses_imported": 0,
        "completed_task_count": 0,
        "provider_process_started": False,
    }
    mismatched = sorted(key for key, expected in required.items() if value.get(key) != expected)
    if set(value) != {*required, "receipt_cid"}:
        mismatched.append("shape")
    if mismatched:
        raise EAAEFReconciliationIdentityError(
            "typed owner receipt differs: " + ", ".join(mismatched)
        )
    receipt_cid = str(value.get("receipt_cid") or "")
    unsigned = {key: item for key, item in value.items() if key != "receipt_cid"}
    if receipt_cid != _cid(unsigned):
        raise EAAEFReconciliationIdentityError("typed owner receipt CID differs")
    _assert_no_boundary_authority(value, path="plan_r2_owner_receipt")
    return value


def prepare_fresh_generation(
    *,
    repo_root: str | Path,
    state_root: str | Path,
    owner: EAAEFTypedReconciliationOwner,
    generation_id: str = "",
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Materialize 22+94 offline, start the owner, and stop before signing."""

    root = Path(repo_root).resolve(strict=True)
    sealed = _require_sealed_forest(inspect_current_repository_forest(root))
    board = _json_object(root / EAAEF_BOARD_PATH, noun="EAAEF task board")
    population = compile_fresh_eaaef_population(board, forest=sealed)
    typed_owner = require_typed_reconciliation_owner(
        owner,
        source_forest_root=population.source_forest_root,
    )
    selected_generation = generation_id or _new_generation_id(population.source_head)
    request = _build_offline_population_request(
        generation_id=selected_generation,
        population=population,
    )
    store = ReconciliationStateStore(state_root)
    if store.active_generation():
        raise EAAEFReconciliationBlocked(
            "an active EAAEF reconciliation generation already exists; stop it first"
        )
    state = {
        "schema": EAAEF_STATE_SCHEMA,
        "interface": EAAEF_RECONCILIATION_LIFECYCLE_INTERFACE,
        "generation_id": selected_generation,
        "phase": "offline_population_materializing",
        "source_head": population.source_head,
        "source_tree": population.source_tree,
        "source_forest_root": population.source_forest_root,
        "population": population.public_dict(),
        "offline_population_request_cid": request["request_cid"],
        "supervisor_birth": None,
        "provider_process_started": False,
        "updated_at_ms": int(now_ms if now_ms is not None else time.time_ns() // 1_000_000),
    }
    state["state_cid"] = _cid(state)
    store.create_generation(selected_generation, state)
    try:
        population = verify_compiled_eaaef_population_commitments(
            population,
            current_board=board,
            current_forest=sealed,
        )
        receipt, snapshot = _validate_offline_population_receipt(
            typed_owner.materialize_offline_population(
                request,
                population=population,
            ),
            request=request,
            population=population,
        )
        unsigned_request = build_unsigned_fresh_authority_request(
            population=population,
            bootstrap_snapshot=snapshot,
        )
    except BaseException as exc:
        failed = {
            **state,
            "phase": "offline_population_materialization_failed",
            "failure_type": type(exc).__name__,
            "updated_at_ms": time.time_ns() // 1_000_000,
        }
        failed.pop("state_cid", None)
        failed["state_cid"] = _cid(failed)
        store.write_state(selected_generation, failed)
        raise
    awaiting = {
        **state,
        "phase": "awaiting_external_authority",
        "offline_population_receipt": receipt,
        "bootstrap_snapshot": snapshot,
        "unsigned_authority_request": unsigned_request,
        "updated_at_ms": time.time_ns() // 1_000_000,
    }
    awaiting.pop("state_cid", None)
    awaiting["state_cid"] = _cid(awaiting)
    store.write_state(selected_generation, awaiting)
    store.activate(selected_generation, state_cid=awaiting["state_cid"])
    return awaiting


def materialize_fresh_generation(
    *,
    repo_root: str | Path,
    state_root: str | Path,
    authority: Mapping[str, Any],
    trust_roots: Mapping[str, Any],
    owner: EAAEFTypedReconciliationOwner,
    generation_id: str = "",
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Resume a prepared generation and apply externally signed Plan R2."""

    root = Path(repo_root).resolve(strict=True)
    selected_forest = inspect_current_repository_forest(root)
    sealed = _require_sealed_forest(selected_forest)
    board = _json_object(root / EAAEF_BOARD_PATH, noun="EAAEF task board")
    population = compile_fresh_eaaef_population(board, forest=sealed)
    admission = verify_fresh_authority_bundle(
        authority,
        population=population,
        trust_roots=trust_roots,
        now_ms=now_ms,
    )
    typed_owner = require_typed_reconciliation_owner(
        owner,
        source_forest_root=population.source_forest_root,
    )
    store = ReconciliationStateStore(state_root)
    active_generation = store.active_generation()
    selected_generation = generation_id or active_generation
    if not selected_generation or selected_generation != active_generation:
        raise EAAEFReconciliationBlocked(
            "signed Plan R2 requires the exact active prepared generation"
        )
    state = store.read_state(selected_generation)
    if state.get("phase") != "awaiting_external_authority":
        raise EAAEFReconciliationBlocked(
            "EAAEF generation is not awaiting an external signing ceremony"
        )
    expected_state = {
        "source_head": population.source_head,
        "source_tree": population.source_tree,
        "source_forest_root": population.source_forest_root,
    }
    state_population = state.get("population")
    if not isinstance(state_population, Mapping):
        raise EAAEFReconciliationIdentityError("prepared EAAEF population state is absent")
    mismatched = [
        field_name
        for field_name, expected in expected_state.items()
        if state.get(field_name) != expected
    ]
    if state_population.get("population_cid") != population.population_cid:
        mismatched.append("population_cid")
    raw_offline_receipt = state.get("offline_population_receipt")
    if not isinstance(raw_offline_receipt, Mapping):
        mismatched.append("offline_population_receipt")
    if mismatched:
        raise EAAEFReconciliationIdentityError(
            "prepared EAAEF generation is stale: " + ", ".join(sorted(mismatched))
        )
    offline_receipt_cid = str(raw_offline_receipt.get("receipt_cid") or "")
    if not _SHA256_RE.fullmatch(offline_receipt_cid):
        raise EAAEFReconciliationIdentityError("offline population receipt CID is absent")
    request = build_typed_owner_materialization_request(
        generation_id=selected_generation,
        population=population,
        authority=admission,
        offline_population_receipt_cid=offline_receipt_cid,
    )
    applying = {
        **state,
        "phase": "materializing",
        "owner_request_cid": request["request_cid"],
        "authority_bundle_cid": admission.authority_bundle_cid,
        "trust_bundle_cid": admission.trust_bundle_cid,
        "plan_r2_remote_admission_cid": admission.remote_capability_cid,
        "updated_at_ms": int(now_ms if now_ms is not None else time.time_ns() // 1_000_000),
    }
    applying.pop("state_cid", None)
    applying["state_cid"] = _cid(applying)
    store.write_state(selected_generation, applying)
    try:
        population = verify_compiled_eaaef_population_commitments(
            population,
            current_board=board,
            current_forest=sealed,
        )
        receipt = _validate_owner_receipt(
            typed_owner.apply_signed_plan_r2(
                request,
                population=population,
                authority=admission,
            ),
            request=request,
        )
    except BaseException as exc:
        failed = {
            **applying,
            "phase": "materialization_failed",
            "failure_type": type(exc).__name__,
            "updated_at_ms": time.time_ns() // 1_000_000,
        }
        failed.pop("state_cid", None)
        failed["state_cid"] = _cid(failed)
        store.write_state(selected_generation, failed)
        raise
    materialized = {
        **applying,
        "phase": "materialized",
        "owner_receipt": receipt,
        "updated_at_ms": time.time_ns() // 1_000_000,
    }
    materialized.pop("state_cid", None)
    materialized["state_cid"] = _cid(materialized)
    store.write_state(selected_generation, materialized)
    store.activate(selected_generation, state_cid=materialized["state_cid"])
    return materialized


def launch_reconciliation_supervisor(
    *,
    state_root: str | Path,
    owner: EAAEFTypedReconciliationOwner,
    mode: str = "paused",
    process_probe: Callable[[int], ProcessBirth | None] = inspect_process_birth,
) -> dict[str, Any]:
    """Launch a paused inspection phase or the explicit full Plan-R2 phase."""

    if mode not in {"paused", "plan_r2"}:
        raise EAAEFReconciliationIdentityError("EAAEF launch mode is not exact")
    store = ReconciliationStateStore(state_root)
    generation_id = store.active_generation()
    if not generation_id:
        raise EAAEFReconciliationBlocked("no materialized EAAEF generation is active")
    state = store.read_state(generation_id)
    if state.get("phase") != "materialized":
        raise EAAEFReconciliationBlocked("EAAEF generation is not ready for launch")
    typed_owner = require_typed_reconciliation_owner(
        owner,
        source_forest_root=str(state.get("source_forest_root") or ""),
    )
    execute_plan_r2 = mode == "plan_r2"
    request = {
        "schema": EAAEF_LAUNCH_REQUEST_SCHEMA,
        "interface": EAAEF_RECONCILIATION_OWNER_INTERFACE,
        "generation_id": generation_id,
        "source_forest_root": state["source_forest_root"],
        "population_cid": state["population"]["population_cid"],
        "owner_receipt_cid": state["owner_receipt"]["receipt_cid"],
        "expected_task_count": EAAEF_TASK_COUNT,
        "launch_mode": mode,
        "provider_launch_allowed": execute_plan_r2,
        "implementation_enabled": execute_plan_r2,
        "typed_task_source_interface": EAAEF_TYPED_TASK_SOURCE_INTERFACE,
    }
    _assert_no_boundary_authority(request)
    request["request_cid"] = _cid(request)
    receipt = dict(typed_owner.launch_reconciliation_supervisor(request))
    receipt_body = dict(receipt)
    receipt_cid = str(receipt_body.pop("receipt_cid", ""))
    expected_receipt_fields = {
        "schema",
        "interface",
        "request_cid",
        "generation_id",
        "source_forest_root",
        "population_cid",
        "launch_mode",
        "implementation_enabled",
        "provider_process_started",
        "typed_task_source_interface",
        "process_birth",
        "receipt_cid",
    }
    if (
        receipt.get("schema") != EAAEF_LAUNCH_RECEIPT_SCHEMA
        or receipt.get("interface") != EAAEF_RECONCILIATION_OWNER_INTERFACE
        or receipt.get("generation_id") != generation_id
        or receipt.get("request_cid") != request["request_cid"]
        or receipt.get("source_forest_root") != request["source_forest_root"]
        or receipt.get("population_cid") != request["population_cid"]
        or receipt.get("launch_mode") != mode
        or receipt.get("implementation_enabled") is not execute_plan_r2
        or receipt.get("typed_task_source_interface") != EAAEF_TYPED_TASK_SOURCE_INTERFACE
        or not isinstance(receipt.get("provider_process_started"), bool)
        or (not execute_plan_r2 and receipt.get("provider_process_started") is not False)
        or set(receipt) != expected_receipt_fields
        or receipt_cid != _cid(receipt_body)
    ):
        raise EAAEFReconciliationIdentityError("supervisor launch receipt differs")
    _assert_no_boundary_authority(receipt, path="supervisor_launch_receipt")
    birth = ProcessBirth.from_mapping(receipt.get("process_birth") or {})
    live = process_probe(birth.pid)
    if live != birth:
        raise EAAEFReconciliationIdentityError(
            "supervisor launch receipt differs from the live process birth"
        )
    launched = {
        **state,
        "phase": "launched_plan_r2" if execute_plan_r2 else "launched_paused",
        "launch_mode": mode,
        "provider_process_started": bool(receipt["provider_process_started"]),
        "supervisor_birth": birth.to_dict(),
        "launch_receipt": receipt,
        "updated_at_ms": time.time_ns() // 1_000_000,
    }
    launched.pop("state_cid", None)
    launched["state_cid"] = _cid(launched)
    store.write_state(generation_id, launched)
    store.activate(generation_id, state_cid=launched["state_cid"])
    return launched


def reconciliation_status(
    *,
    state_root: str | Path,
    owner: EAAEFTypedReconciliationOwner,
    process_probe: Callable[[int], ProcessBirth | None] = inspect_process_birth,
) -> dict[str, Any]:
    store = ReconciliationStateStore(state_root)
    generation_id = store.active_generation()
    state = store.read_state(generation_id) if generation_id else {}
    typed_owner = require_typed_reconciliation_owner(
        owner,
        source_forest_root=str(state.get("source_forest_root") or ""),
    )
    request = {
        "schema": EAAEF_OWNER_STATUS_REQUEST_SCHEMA,
        "interface": EAAEF_RECONCILIATION_OWNER_INTERFACE,
        "operation": "status.snapshot",
        "generation_id": generation_id,
        "local_state_cid": str(state.get("state_cid") or ""),
        "expected_task_count": EAAEF_TASK_COUNT,
    }
    request["request_cid"] = _cid(request)
    _assert_no_boundary_authority(request)
    receipt = dict(typed_owner.reconciliation_status_snapshot(request))
    body = dict(receipt)
    receipt_cid = str(body.pop("receipt_cid", ""))
    expected_receipt_fields = {
        "schema",
        "interface",
        "request_cid",
        "active",
        "generation_id",
        "phase",
        "source_head",
        "source_forest_root",
        "task_count",
        "task_status_counts",
        "supervisor_birth",
        "provider_process_started",
        "receipt_cid",
    }
    active = receipt.get("active")
    owner_generation = str(receipt.get("generation_id") or "")
    if (
        receipt.get("schema") != EAAEF_OWNER_STATUS_RECEIPT_SCHEMA
        or receipt.get("interface") != EAAEF_RECONCILIATION_OWNER_INTERFACE
        or receipt.get("request_cid") != request["request_cid"]
        or type(active) is not bool
        or type(receipt.get("provider_process_started")) is not bool
        or set(receipt) != expected_receipt_fields
        or receipt_cid != _cid(body)
        or (active and not _GENERATION_ID_RE.fullmatch(owner_generation))
        or (not active and owner_generation)
        or (generation_id and active and owner_generation != generation_id)
    ):
        raise EAAEFReconciliationIdentityError("typed owner status receipt differs")
    _assert_no_boundary_authority(receipt, path="owner_status_receipt")
    if not active:
        if (
            receipt.get("phase") != "absent"
            or receipt.get("source_head")
            or receipt.get("source_forest_root")
            or receipt.get("task_count") != 0
            or receipt.get("task_status_counts") != {}
            or receipt.get("supervisor_birth") is not None
            or receipt.get("provider_process_started") is not False
        ):
            raise EAAEFReconciliationIdentityError("inactive owner status receipt differs")
        return {
            "schema": EAAEF_STATE_SCHEMA,
            "active": False,
            "phase": "absent",
            "owner_status_receipt_cid": receipt_cid,
            "local_registry_generation_id": generation_id,
            "provider_process_started": False,
        }
    if receipt.get("task_count") != EAAEF_TASK_COUNT:
        raise EAAEFReconciliationIdentityError("typed owner status task population differs")
    if receipt.get("phase") not in {
        "awaiting_external_authority",
        "materialized",
        "launched_paused",
        "launched_plan_r2",
        "stopping",
    }:
        raise EAAEFReconciliationIdentityError("typed owner status phase differs")
    raw_counts = receipt.get("task_status_counts")
    if (
        not isinstance(raw_counts, Mapping)
        or not raw_counts
        or any(not isinstance(key, str) or key not in _TASK_STATUS_VOCABULARY for key in raw_counts)
        or any(type(value) is not int or value < 0 for value in raw_counts.values())
        or sum(raw_counts.values()) != EAAEF_TASK_COUNT
    ):
        raise EAAEFReconciliationIdentityError("typed owner status counts are malformed")
    if (
        not _GIT_OID_RE.fullmatch(str(receipt.get("source_head") or ""))
        or not _SHA256_RE.fullmatch(str(receipt.get("source_forest_root") or ""))
        or (
            state
            and (
                receipt.get("source_head") != state.get("source_head")
                or receipt.get("source_forest_root") != state.get("source_forest_root")
            )
        )
    ):
        raise EAAEFReconciliationIdentityError("typed owner status source binding differs")
    raw_birth = receipt.get("supervisor_birth")
    birth_status = "not_launched"
    alive = False
    corroborated = raw_birth is None
    if isinstance(raw_birth, Mapping):
        birth = ProcessBirth.from_mapping(raw_birth)
        live = process_probe(birth.pid)
        alive = live == birth
        corroborated = alive
        birth_status = "corroborated_alive" if alive else "not_locally_corroborated"
    elif raw_birth is not None:
        raise EAAEFReconciliationIdentityError("typed owner status birth is malformed")
    return {
        "schema": EAAEF_STATE_SCHEMA,
        "active": True,
        "generation_id": owner_generation,
        "phase": receipt.get("phase"),
        "source_head": receipt.get("source_head"),
        "source_forest_root": receipt.get("source_forest_root"),
        "task_count": EAAEF_TASK_COUNT,
        "task_status_counts": dict(raw_counts),
        "supervisor_birth_status": birth_status,
        "owner_supervisor_birth": dict(raw_birth) if isinstance(raw_birth, Mapping) else None,
        "supervisor_alive": alive,
        "local_birth_corroborated": corroborated,
        "provider_process_started": bool(receipt.get("provider_process_started")),
        "owner_status_receipt_cid": receipt_cid,
        "local_registry_generation_id": generation_id,
        "local_state_cid": state.get("state_cid"),
    }


def stop_reconciliation_generation(
    *,
    state_root: str | Path,
    owner: EAAEFTypedReconciliationOwner,
    process_probe: Callable[[int], ProcessBirth | None] = inspect_process_birth,
) -> dict[str, Any]:
    """Ask the owner to stop exact tracked births, then corroborate and clean."""

    store = ReconciliationStateStore(state_root)
    local_generation = store.active_generation()
    state = store.read_state(local_generation) if local_generation else {}
    typed_owner = require_typed_reconciliation_owner(
        owner,
        source_forest_root=str(state.get("source_forest_root") or ""),
    )
    status = reconciliation_status(
        state_root=state_root,
        owner=typed_owner,
        process_probe=process_probe,
    )
    if not status["active"]:
        return {
            "schema": EAAEF_STATE_SCHEMA,
            "stopped": False,
            "status": "not_running",
            "removed_runtime_artifacts": [],
        }
    generation_id = str(status["generation_id"])
    request = {
        "schema": EAAEF_OWNER_STOP_REQUEST_SCHEMA,
        "interface": EAAEF_RECONCILIATION_OWNER_INTERFACE,
        "operation": "stop_tracks",
        "generation_id": generation_id,
        "owner_status_receipt_cid": status["owner_status_receipt_cid"],
        "expected_task_count": EAAEF_TASK_COUNT,
        "cleanup_scope": "exact_owner_tracked_births_only",
    }
    request["request_cid"] = _cid(request)
    _assert_no_boundary_authority(request)
    receipt = dict(typed_owner.stop_reconciliation_tracks(request))
    body = dict(receipt)
    receipt_cid = str(body.pop("receipt_cid", ""))
    raw_births = receipt.get("stopped_process_births")
    expected_receipt_fields = {
        "schema",
        "interface",
        "request_cid",
        "generation_id",
        "stopped",
        "remaining_track_count",
        "stopped_process_births",
        "provider_processes_stopped",
        "task_state_mutated",
        "receipt_cid",
    }
    if (
        receipt.get("schema") != EAAEF_OWNER_STOP_RECEIPT_SCHEMA
        or receipt.get("interface") != EAAEF_RECONCILIATION_OWNER_INTERFACE
        or receipt.get("request_cid") != request["request_cid"]
        or receipt.get("generation_id") != generation_id
        or receipt.get("stopped") is not True
        or receipt.get("remaining_track_count") != 0
        or receipt.get("provider_processes_stopped") is not True
        or receipt.get("task_state_mutated") is not False
        or set(receipt) != expected_receipt_fields
        or receipt_cid != _cid(body)
        or not isinstance(raw_births, list)
    ):
        raise EAAEFReconciliationIdentityError("typed owner stop receipt differs")
    _assert_no_boundary_authority(receipt, path="owner_stop_receipt")
    stopped_births = [ProcessBirth.from_mapping(item) for item in raw_births]
    if len(set(stopped_births)) != len(stopped_births):
        raise EAAEFReconciliationIdentityError("typed owner stop receipt repeats a process birth")
    raw_status_birth = status.get("owner_supervisor_birth")
    if isinstance(raw_status_birth, Mapping):
        status_birth = ProcessBirth.from_mapping(raw_status_birth)
        if status_birth not in stopped_births:
            raise EAAEFReconciliationIdentityError(
                "typed owner stop receipt omits the status-bound supervisor birth"
            )
    still_live = [birth for birth in stopped_births if process_probe(birth.pid) == birth]
    if still_live:
        raise EAAEFReconciliationBlocked(
            "owner stop receipt is not corroborated for every exact tracked birth"
        )
    removed = (
        store.cleanup_runtime_artifacts(generation_id) if local_generation == generation_id else []
    )
    stopped_state_cid = ""
    if local_generation == generation_id:
        stopped = {
            **state,
            "phase": "stopped",
            "supervisor_birth": None,
            "provider_process_started": False,
            "owner_stop_receipt_cid": receipt_cid,
            "removed_runtime_artifacts": removed,
            "updated_at_ms": time.time_ns() // 1_000_000,
        }
        stopped.pop("state_cid", None)
        stopped["state_cid"] = _cid(stopped)
        store.deactivate(generation_id)
        store.write_state(generation_id, stopped)
        stopped_state_cid = str(stopped["state_cid"])
    return {
        "schema": EAAEF_STATE_SCHEMA,
        "stopped": True,
        "status": "stopped",
        "generation_id": generation_id,
        "stopped_process_count": len(stopped_births),
        "removed_runtime_artifacts": removed,
        "owner_stop_receipt_cid": receipt_cid,
        "state_cid": stopped_state_cid,
    }


def preflight_reconciliation(
    repo_root: str | Path,
    *,
    authority: Mapping[str, Any] | None = None,
    trust_roots: Mapping[str, Any] | None = None,
    bootstrap_snapshot: Mapping[str, Any] | None = None,
    owner: object | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Return current forest/population and exact production blockers."""

    root = Path(repo_root).resolve(strict=True)
    selected_forest = inspect_current_repository_forest(root)
    blockers = list(selected_forest.get("blockers") or ())
    population: CompiledEAAEFPopulation | None = None
    unsigned_request: dict[str, Any] | None = None
    try:
        sealed = _require_sealed_forest(selected_forest)
        board = _json_object(root / EAAEF_BOARD_PATH, noun="EAAEF task board")
        population = compile_fresh_eaaef_population(board, forest=sealed)
        unsigned_request = build_unsigned_fresh_authority_request(
            population=population,
            bootstrap_snapshot=bootstrap_snapshot,
        )
    except EAAEFReconciliationError as exc:
        blockers.append(str(exc))
    stale_bindings: list[str] = []
    if population is not None:
        try:
            config = _json_object(root / EAAEF_CONFIG_PATH, noun="EAAEF scheduler config")
        except EAAEFReconciliationError as exc:
            blockers.append(str(exc))
        else:
            binding = config.get("source_binding")
            if not isinstance(binding, Mapping) or (
                binding.get("source_forest_root") != population.source_forest_root
                or binding.get("ipfs_accelerate_planning_revision") != population.source_head
                or binding.get("ipfs_accelerate_planning_tree") != population.source_tree
            ):
                stale_bindings.append("scheduler_source_binding")
        try:
            old_admission = _json_object(
                root / EAAEF_ADMISSION_BUNDLE_PATH,
                noun="historical EAAEF admission bundle",
            )
        except EAAEFReconciliationError:
            old_admission = {}
        if old_admission and (
            old_admission.get("source_head") != population.source_head
            or old_admission.get("source_tree") != population.source_tree
        ):
            stale_bindings.append("historical_host_admission")
        if authority is None:
            blockers.append("fresh_independently_signed_plan_r2_authority_absent")
        elif trust_roots is None:
            blockers.append("independently_configured_fresh_trust_roots_absent")
        else:
            try:
                verify_fresh_authority_bundle(
                    authority,
                    population=population,
                    trust_roots=trust_roots,
                    now_ms=now_ms,
                )
            except Exception as exc:
                blockers.append(f"fresh_authority_rejected:{type(exc).__name__}:{exc}")
    if stale_bindings:
        blockers.append("stale_bindings_rejected:" + ",".join(sorted(stale_bindings)))
    try:
        require_typed_reconciliation_owner(
            owner,
            source_forest_root=(population.source_forest_root if population else ""),
        )
    except EAAEFReconciliationBlocked:
        blockers.append(
            "typed_portfolio_materialization_owner_unavailable_until_final_casf_adapter"
        )
    blockers = list(dict.fromkeys(str(item) for item in blockers if str(item)))
    return {
        "schema": ("ipfs_accelerate_py/agent-supervisor/eaaef-reconciliation-preflight@1"),
        "valid": not blockers,
        "launch_allowed": False,
        "provider_launch_allowed": False,
        "board_namespace": EAAEF_BOARD_NAMESPACE,
        "forest": selected_forest,
        "population": population.public_dict() if population is not None else None,
        "unsigned_authority_request": unsigned_request,
        "stale_bindings": sorted(stale_bindings),
        "blockers": blockers,
        "required_owner_interface": EAAEF_RECONCILIATION_OWNER_INTERFACE,
        "required_typed_task_source_interface": EAAEF_TYPED_TASK_SOURCE_INTERFACE,
        "required_plan_r2_repository_interface": AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE,
        "source_only_plan_r2_remote_qualification_status": (
            PLAN_R2_REMOTE_RUNTIME_QUALIFICATION_STATUS
        ),
        "source_only_plan_r2_remote_production_blockers": list(
            PLAN_R2_REMOTE_RUNTIME_PRODUCTION_BLOCKERS
        ),
        "historical_status_overlay_allowed": False,
        "raw_database_authority_crossing_allowed": False,
    }


def _state_root_from_args(repo_root: Path, value: str) -> Path:
    selected = Path(value).expanduser()
    return (
        selected.resolve(strict=False)
        if selected.is_absolute()
        else (repo_root / selected).resolve(strict=False)
    )


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), indent=2, sort_keys=True))


def _add_authority_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--authorization", default="")
    parser.add_argument("--plan-r2-operational-capability", default="")
    parser.add_argument("--plan-r2-remote-owner-capability", default="")
    parser.add_argument("--trust-roots", default="")


def _authority_from_args(
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    paths = (
        str(getattr(args, "authorization", "") or ""),
        str(getattr(args, "plan_r2_operational_capability", "") or ""),
        str(getattr(args, "plan_r2_remote_owner_capability", "") or ""),
    )
    trust_path = str(getattr(args, "trust_roots", "") or "")
    if not any(paths) and not trust_path:
        return None, None
    if not all(paths) or not trust_path:
        raise EAAEFReconciliationIdentityError(
            "authorization, both capability artifacts, and independent trust roots "
            "must be supplied together"
        )
    return (
        load_fresh_authority_artifacts(
            authorization_path=paths[0],
            plan_r2_operational_capability_path=paths[1],
            plan_r2_remote_owner_capability_path=paths[2],
        ),
        load_fresh_trust_roots(trust_path),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh, typed EAAEF reconciliation lifecycle. No command accepts a "
            "DuckDB path, SQL, token, credential, or historical run identifier."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--state-root", default=EAAEF_RECONCILIATION_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser(
        "preflight", help="Seal the current forest and report exact blockers."
    )
    _add_authority_artifact_arguments(preflight)
    preflight.add_argument(
        "--bootstrap-snapshot",
        default="",
        help="Typed fresh-owner snapshot used to emit the actual unsigned Plan-R2 statement.",
    )
    materialize = commands.add_parser(
        "materialize",
        help=(
            "With no signed artifacts, materialize 22+94 offline and emit an unsigned "
            "statement. With all four artifacts, apply signed Plan R2."
        ),
    )
    _add_authority_artifact_arguments(materialize)
    materialize.add_argument("--generation-id", default="")
    launch = commands.add_parser(
        "launch", help="Launch paused, or explicitly enter the full Plan-R2 execution phase."
    )
    launch_mode = launch.add_mutually_exclusive_group()
    launch_mode.add_argument("--paused", action="store_true")
    launch_mode.add_argument("--plan-r2", action="store_true")
    commands.add_parser("status", help="Report the active generation and exact birth liveness.")
    commands.add_parser(
        "stop", help="Ask the typed owner to stop exact tracked births, then clean fixed artifacts."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public ``preflight|materialize|launch|status|stop`` commands."""

    args = _argument_parser().parse_args(argv)
    try:
        repo_root = Path(args.repo_root).resolve(strict=True)
        state_root = _state_root_from_args(repo_root, str(args.state_root))
        if args.command == "preflight":
            authority, trust_roots = _authority_from_args(args)
            bootstrap_snapshot = (
                load_fresh_bootstrap_snapshot(args.bootstrap_snapshot)
                if args.bootstrap_snapshot
                else None
            )
            try:
                owner: EAAEFTypedReconciliationOwner | None = (
                    resolve_production_reconciliation_owner(repo_root)
                )
            except EAAEFReconciliationBlocked:
                owner = None
            result = preflight_reconciliation(
                repo_root,
                authority=authority,
                trust_roots=trust_roots,
                bootstrap_snapshot=bootstrap_snapshot,
                owner=owner,
            )
            _print_json(result)
            return 0 if result["valid"] else 2
        owner = resolve_production_reconciliation_owner(repo_root)
        if args.command == "status":
            _print_json(reconciliation_status(state_root=state_root, owner=owner))
            return 0
        if args.command == "stop":
            _print_json(
                stop_reconciliation_generation(
                    state_root=state_root,
                    owner=owner,
                )
            )
            return 0
        if args.command == "materialize":
            authority, trust_roots = _authority_from_args(args)
            result = (
                prepare_fresh_generation(
                    repo_root=repo_root,
                    state_root=state_root,
                    owner=owner,
                    generation_id=args.generation_id,
                )
                if authority is None
                else materialize_fresh_generation(
                    repo_root=repo_root,
                    state_root=state_root,
                    authority=authority,
                    trust_roots=trust_roots or {},
                    owner=owner,
                    generation_id=args.generation_id,
                )
            )
            _print_json(result)
            return 0
        if args.command == "launch":
            _print_json(
                launch_reconciliation_supervisor(
                    state_root=state_root,
                    owner=owner,
                    mode="plan_r2" if args.plan_r2 else "paused",
                )
            )
            return 0
    except EAAEFReconciliationError as exc:
        _print_json(
            {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/eaaef-reconciliation-command-error@1"
                ),
                "valid": False,
                "command": str(getattr(args, "command", "") or ""),
                "error_code": type(exc).__name__,
                "error": str(exc),
                "authority_mutated": False,
                "provider_process_started": False,
            }
        )
        return 2
    return 2


__all__ = [
    "CompiledEAAEFPopulation",
    "EAAEFReconciliationBlocked",
    "EAAEFReconciliationError",
    "EAAEFReconciliationIdentityError",
    "EAAEFTypedReconciliationOwner",
    "EAAEF_BOARD_NAMESPACE",
    "EAAEF_BOOTSTRAP_TASK_COUNT",
    "EAAEF_BOARD_SOURCE_SCHEMA",
    "EAAEF_FRESH_AUTHORITY_SCHEMA",
    "EAAEF_FRESH_TRUST_SCHEMA",
    "EAAEF_EXECUTION_CONTRACT_POPULATION_SCHEMA",
    "EAAEF_FOREST_SCHEMA",
    "EAAEF_GOAL_EDGE_COUNT",
    "EAAEF_GOAL_COUNT",
    "EAAEF_PLAN_R2_TASK_COUNT",
    "EAAEF_RECONCILIATION_OWNER_INTERFACE",
    "EAAEF_TASK_COUNT",
    "ProcessBirth",
    "ReconciliationStateStore",
    "VerifiedFreshEAAEFAuthority",
    "apply_plan_r2_through_existing_repository",
    "assemble_fresh_authority_bundle",
    "build_unsigned_fresh_authority_request",
    "build_unsigned_fresh_plan_r2_statement",
    "build_typed_owner_materialization_request",
    "compile_fresh_eaaef_population",
    "inspect_current_repository_forest",
    "inspect_process_birth",
    "launch_reconciliation_supervisor",
    "load_fresh_authority_artifacts",
    "load_fresh_bootstrap_snapshot",
    "load_fresh_trust_roots",
    "materialize_fresh_generation",
    "preflight_reconciliation",
    "prepare_fresh_generation",
    "reconciliation_status",
    "require_typed_reconciliation_owner",
    "resolve_production_reconciliation_owner",
    "stop_reconciliation_generation",
    "verify_fresh_authority_bundle",
    "verify_compiled_eaaef_population_commitments",
]


if __name__ == "__main__":
    raise SystemExit(main())
