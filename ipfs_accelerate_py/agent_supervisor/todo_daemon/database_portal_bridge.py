"""Attempt-local Portal execution for database-authoritative task claims.

``DatabaseImplementationDaemon`` owns the durable claim and completion state.
``PortalImplementationDaemon`` owns the already-landed implementation pipeline
(provider routing, isolated worktrees, validation, proof gates, and merge
reconciliation).  This module joins those authorities without allowing the
Portal daemon to mutate the canonical task board: each database attempt gets a
single-task Markdown *projection* below its private state directory.

The projection is deliberately disposable and non-authoritative.  Its
immutable fields are sealed before provider execution; only its status line
may change.  A database phase may consume the result only after the projected
task has a matching durable Portal completion event.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final

from ..merge.protected_recovery_fence import (
    FENCE_CONTENTION_BACKOFF_SECONDS,
    is_protected_recovery_fence_contention,
)
from ..runtime.event_log import append_jsonl_event
from ..validation.validation_commands import validation_command_repository_root

DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE: Final[str] = "DatabasePortalExecutionBridge@1"
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@1"
)
DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@1"
)
DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-validation-retry@1"
)
DATABASE_PORTAL_VALIDATION_RETRY_SEED_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-validation-retry-seed@1"
)
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "complete", "done"}
)
_MUTABLE_PROJECTION_LINE = re.compile(r"(?mi)^-\s*status\s*:\s*.*$")
_OPERATIONAL_PROJECTION_LINE = re.compile(
    r"(?mi)^-\s*completion\s+receipt\s*:\s*.*$"
)
_HEADER = re.compile(r"(?m)^##\s+([^\s]+)(?:\s+.*)?$")
_ROOT_REPOSITORY_AUTHORITY: Final[str] = "ipfs_accelerate_py"
_MAX_REPOSITORY_PATH_BYTES: Final[int] = 1024
_MAX_TASK_IDENTITY_BYTES: Final[int] = 4096
_MAX_DATABASE_PORTAL_BACKOFF_SECONDS: Final[int] = 86_400
_MAX_DATABASE_PORTAL_TASK_ATTEMPTS: Final[int] = 10_000
_MAX_DATABASE_PORTAL_EVENT_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_DATABASE_PORTAL_EVENTS: Final[int] = 4096


class DatabasePortalBridgeError(RuntimeError):
    """A database claim could not obtain trustworthy Portal evidence."""


class DatabasePortalBridgeDeferred(DatabasePortalBridgeError):
    """Portal execution made bounded progress but is not yet acceptable."""

    def __init__(self, reason: str, *, backoff_seconds: int = 300) -> None:
        if (
            isinstance(backoff_seconds, bool)
            or not isinstance(backoff_seconds, int)
            or backoff_seconds < 0
            or backoff_seconds > _MAX_DATABASE_PORTAL_BACKOFF_SECONDS
        ):
            raise ValueError(
                "backoff_seconds must be an integer in "
                f"[0, {_MAX_DATABASE_PORTAL_BACKOFF_SECONDS}]"
            )
        reason_text = str(reason or "portal_execution_deferred").strip()
        super().__init__(reason_text or "portal_execution_deferred")
        self.reason = reason_text or "portal_execution_deferred"
        self.backoff_seconds = int(backoff_seconds)
        # A typed deferral occurs before the provider is admitted.  These
        # fields deliberately mirror the Portal result contract so the outer
        # database authority need not infer retry semantics from prose.
        self.attempt_consumed = False
        self.provider_dispatched = False


class DatabasePortalValidationRetry(DatabasePortalBridgeError):
    """A dispatched candidate failed only its authoritative validation.

    This is deliberately distinct from a pre-dispatch deferral.  It consumes
    an ordinary provider attempt and carries independently reproducible,
    identity-bound evidence.  Callers must not infer this disposition from a
    provider error string.
    """

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        value = dict(receipt)
        if (
            value.get("schema") != DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA
            or value.get("disposition") != "retry"
            or value.get("attempt_consumed") is not True
            or value.get("provider_dispatched") is not True
            or value.get("reason") != "declared_validation_failed"
        ):
            raise ValueError("validation retry receipt has an invalid disposition")
        backoff_seconds = value.get("backoff_seconds")
        if (
            isinstance(backoff_seconds, bool)
            or not isinstance(backoff_seconds, int)
            or backoff_seconds < 0
            or backoff_seconds > _MAX_DATABASE_PORTAL_BACKOFF_SECONDS
        ):
            raise ValueError("validation retry receipt has an invalid backoff")
        super().__init__("declared_validation_failed")
        self.reason = "declared_validation_failed"
        self.backoff_seconds = int(backoff_seconds)
        self.attempt_consumed = True
        self.provider_dispatched = True
        self.retry_receipt = value


@dataclass(frozen=True)
class DatabasePortalAttemptPaths:
    """Private, non-authoritative paths for one database task attempt."""

    root: Path
    task_projection: Path
    binding: Path
    state: Path
    strategy: Path
    events: Path
    implementation_logs: Path


PortalDaemonFactory = Callable[[DatabasePortalAttemptPaths, str], Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise DatabasePortalBridgeError(
            f"could not read Portal attempt artifact {path.name!r}"
        ) from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _line_value(value: Any) -> str:
    if isinstance(value, str):
        selected = value
    elif isinstance(value, Mapping):
        selected = _canonical_json(dict(value)).decode("utf-8")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        selected = ", ".join(_line_value(item) for item in value)
    else:
        selected = str(value or "")
    return " ".join(selected.replace("\x00", "").splitlines()).strip()


def _mapping_path(value: Mapping[str, Any]) -> str:
    candidates = tuple(
        value[key]
        for key in ("path", "output", "artifact_id", "fluent_id")
        if key in value and value[key] not in (None, "")
    )
    if not candidates:
        raise DatabasePortalBridgeError("task output mapping has no path identity")
    if any(type(candidate) is not str for candidate in candidates):
        raise DatabasePortalBridgeError("task output path identity is not a string")
    if len(set(candidates)) != 1:
        raise DatabasePortalBridgeError("task output mapping has ambiguous path identities")
    return candidates[0]


def _output_values(record: Any, body: Mapping[str, Any]) -> list[str]:
    raw = getattr(record, "outputs", ()) or body.get("outputs") or ()
    if isinstance(raw, (str, Mapping)):
        raw = (raw,)
    selected: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            value = _mapping_path(item)
        elif type(item) is str:
            value = item
        else:
            raise DatabasePortalBridgeError("task output path identity is not a string")
        if value and value not in selected:
            selected.append(value)
    return selected


def _safe_output_path(value: Any) -> str:
    """Return one lossless repository-relative output path or fail closed."""

    if type(value) is not str:
        raise DatabasePortalBridgeError("task output path identity is not a string")
    path = PurePosixPath(value or ".")
    if (
        not value
        or value != value.strip()
        or len(value.encode("utf-8", errors="surrogatepass"))
        > _MAX_REPOSITORY_PATH_BYTES
        or "\\" in value
        or "," in value
        or path.is_absolute()
        or bool(PureWindowsPath(value).drive)
        or path == PurePosixPath(".")
        or path.as_posix() != value
        or ".." in path.parts
        or any(ord(character) < 32 for character in value)
    ):
        raise DatabasePortalBridgeError("task output path identity is unsafe or ambiguous")
    return path.as_posix()


def _validation_values(record: Any, body: Mapping[str, Any]) -> list[str]:
    raw = (
        getattr(record, "validations", ())
        or body.get("validations")
        or body.get("validation_commands")
        or body.get("validation")
        or ()
    )
    if isinstance(raw, (str, Mapping)):
        raw = (raw,)
    selected: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            argv = item.get("argv")
            if isinstance(argv, Sequence) and not isinstance(
                argv, (str, bytes, bytearray, memoryview)
            ):
                if not argv or any(type(part) is not str for part in argv):
                    raise DatabasePortalBridgeError(
                        "validation argv must contain exact nonempty strings"
                    )
                parts = tuple(argv)
                if any(not part or part != _line_value(part) for part in parts):
                    raise DatabasePortalBridgeError(
                        "validation argv contains noncanonical command text"
                    )
                # Database task sources preserve a Markdown shell command as
                # one argv item.  Re-joining that singleton would quote the
                # entire command and make the shell treat it as one executable
                # name.  Multi-item argv records remain losslessly joined.
                value = parts[0] if len(parts) == 1 else shlex.join(parts)
            else:
                if argv is not None:
                    raise DatabasePortalBridgeError(
                        "validation argv must be a sequence of exact strings"
                    )
                raw_value = item.get("command") or item.get("value")
                if (
                    type(raw_value) is not str
                    or not raw_value
                    or raw_value != _line_value(raw_value)
                ):
                    raise DatabasePortalBridgeError(
                        "validation command text is absent or noncanonical"
                    )
                value = raw_value
        else:
            if type(item) is not str or not item or item != _line_value(item):
                raise DatabasePortalBridgeError(
                    "validation command text is absent or noncanonical"
                )
            value = item
        if value and value not in selected:
            selected.append(value)
    return selected


def _safe_repository_path(value: Any) -> str:
    """Return one canonical relative repository path or fail closed."""

    if type(value) is not str:
        raise DatabasePortalBridgeError("owning repository metadata is not a string")
    selected = value.strip()
    path = PurePosixPath(selected or ".")
    if (
        not selected
        or selected != value
        or len(selected.encode("utf-8", errors="surrogatepass"))
        > _MAX_REPOSITORY_PATH_BYTES
        or "\\" in selected
        or path.is_absolute()
        or path.as_posix() != selected
        or ".." in path.parts
        or any(ord(character) < 32 for character in selected)
    ):
        raise DatabasePortalBridgeError("owning repository metadata is unsafe")
    return path.as_posix()


def _owning_repository(body: Mapping[str, Any]) -> str:
    """Read the owning-repository authority from consistent sealed fields."""

    raw_values: list[Any] = []
    for key in ("owning_repository", "owning repository"):
        if key in body and body[key] not in (None, ""):
            raw_values.append(body[key])
    markdown_metadata = body.get("markdown_metadata")
    if isinstance(markdown_metadata, Mapping):
        for key in ("owning_repository", "owning repository"):
            if key in markdown_metadata and markdown_metadata[key] not in (None, ""):
                raw_values.append(markdown_metadata[key])
    if not raw_values:
        return ""
    values = tuple(_safe_repository_path(value) for value in raw_values)
    if len(set(values)) != 1:
        raise DatabasePortalBridgeError("owning repository metadata is inconsistent")
    return values[0]


def _canonical_projection_identity(
    record: Any,
    body: Mapping[str, Any],
) -> tuple[str, str]:
    """Project the database task identity without creating new authority."""

    task_cid = getattr(record, "task_cid", None)
    if (
        type(task_cid) is not str
        or not task_cid
        or task_cid != _line_value(task_cid)
        or len(task_cid.encode("utf-8", errors="surrogatepass"))
        > _MAX_TASK_IDENTITY_BYTES
    ):
        raise DatabasePortalBridgeError(
            "database task CID is absent or noncanonical"
        )
    declared_cids = tuple(
        body[key]
        for key in ("canonical_task_cid", "canonical task cid")
        if key in body and body[key] not in (None, "")
    )
    if any(type(value) is not str or value != task_cid for value in declared_cids):
        raise DatabasePortalBridgeError(
            "database task body conflicts with its canonical CID"
        )

    declared_keys = tuple(
        body[key]
        for key in ("canonical_task_key", "canonical task key", "task_key")
        if key in body and body[key] not in (None, "")
    )
    if declared_keys:
        if (
            any(type(value) is not str for value in declared_keys)
            or len(set(declared_keys)) != 1
        ):
            raise DatabasePortalBridgeError(
                "database task body has an ambiguous canonical key"
            )
        task_key = declared_keys[0]
    else:
        # DuckDB task records currently persist the canonical CID but do not
        # require the historical semantic-key projection.  Derive only a
        # stable lookup key; the database CID remains the task authority.
        task_key = "task/v1/" + hashlib.sha256(task_cid.encode("utf-8")).hexdigest()
    if (
        not task_key
        or task_key != _line_value(task_key)
        or len(task_key.encode("utf-8", errors="surrogatepass"))
        > _MAX_TASK_IDENTITY_BYTES
    ):
        raise DatabasePortalBridgeError(
            "database task canonical key is absent or noncanonical"
        )
    return task_key, task_cid


def _acceptance_value(record: Any, body: Mapping[str, Any]) -> str:
    raw = (
        getattr(record, "acceptance", ())
        or body.get("acceptance")
        or body.get("completion_contract")
        or body.get("completion rule")
        or body.get("completion_rule")
        or ()
    )
    if isinstance(raw, (str, Mapping)):
        raw = (raw,)
    values: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            value = _line_value(
                item.get("criterion") or item.get("statement") or item.get("value") or item
            )
        else:
            value = _line_value(item)
        if value:
            values.append(value)
    return " ; ".join(values)


def _projection_immutable_digest(text: str) -> str:
    normalized = _MUTABLE_PROJECTION_LINE.sub("- Status: <mutable>", text)
    return _sha256_bytes(normalized.encode("utf-8"))


def _projection_recovery_digest(text: str) -> str:
    """Ignore only mutable status and legacy operational receipt projection.

    Older bridge revisions projected the accelerator-owned status receipt into
    provider context.  Its replacement by the terminal blocked receipt must
    not invalidate semantic recovery, while every other projected task byte
    remains bound.
    """

    normalized = _MUTABLE_PROJECTION_LINE.sub("- Status: <mutable>", text)
    normalized = _OPERATIONAL_PROJECTION_LINE.sub("", normalized)
    normalized = "\n".join(
        line for line in normalized.splitlines() if line.strip()
    )
    return _sha256_bytes((normalized + "\n").encode("utf-8"))


def _projection_status(text: str) -> str:
    match = re.search(r"(?mi)^-\s*status\s*:\s*([^\r\n]+)$", text)
    return str(match.group(1) if match else "").strip().lower().replace("-", "_")


def _bounded_portal_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep control evidence while excluding raw provider/model payloads."""

    summary: dict[str, Any] = {}
    for key in (
        "task_count",
        "completed_count",
        "ready_count",
        "blocked_count",
        "active_task_id",
        "selection_idle_reason",
        "unchanged",
        "write_count",
        "blocked",
        "reason",
    ):
        if key in result:
            summary[key] = result[key]
    implementation = result.get("implementation_result")
    if isinstance(implementation, Mapping):
        summary["implementation"] = {
            key: implementation[key]
            for key in (
                "task_id",
                "attempt",
                "returncode",
                "reason",
                "deferred",
                "attempt_consumed",
                "provider_dispatched",
                "backoff_seconds",
                "skipped",
                "implementation_commit",
                "branch",
                "merge_queued",
            )
            if key in implementation
        }
    reconciliation = result.get("merge_reconciliation")
    if isinstance(reconciliation, Sequence) and not isinstance(
        reconciliation, (str, bytes, bytearray, memoryview)
    ):
        summary["merge_reconciliation"] = [
            {
                key: item[key]
                for key in (
                    "task_id",
                    "returncode",
                    "reason",
                    "status",
                    "implementation_commit",
                    "merge_commit",
                    "resolved",
                )
                if key in item
            }
            for item in reconciliation[-8:]
            if isinstance(item, Mapping)
        ]
    return summary


class DatabasePortalExecutionBridge:
    """Run one database claim through a private Portal execution projection."""

    INTERFACE = DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE
    RECEIPT_SCHEMA = DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA

    def __init__(
        self,
        *,
        task_source: Any,
        attempt_root: Path | str,
        portal_factory: PortalDaemonFactory,
        repository_root: Path | str | None = None,
        worktree_submodule_paths: Sequence[str] = (),
        task_header_prefix: str = "## ",
        max_passes: int = 4,
        max_task_attempts: int = 0,
    ) -> None:
        if not callable(portal_factory):
            raise TypeError("portal_factory must be callable")
        if isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer")
        if (
            isinstance(max_task_attempts, bool)
            or not isinstance(max_task_attempts, int)
            or max_task_attempts < 0
            or max_task_attempts > _MAX_DATABASE_PORTAL_TASK_ATTEMPTS
        ):
            raise ValueError(
                "max_task_attempts must be an integer in "
                f"[0, {_MAX_DATABASE_PORTAL_TASK_ATTEMPTS}]"
            )
        self.task_source = task_source
        self.attempt_root = Path(attempt_root).absolute()
        self.portal_factory = portal_factory
        self.repository_root = (
            Path(repository_root).absolute() if repository_root is not None else None
        )
        self.worktree_submodule_paths = tuple(
            _safe_repository_path(path) for path in worktree_submodule_paths
        )
        if len(set(self.worktree_submodule_paths)) != len(
            self.worktree_submodule_paths
        ):
            raise ValueError("worktree_submodule_paths must be unique")
        self.task_header_prefix = str(task_header_prefix or "## ")
        self.max_passes = max_passes
        self.max_task_attempts = int(max_task_attempts)

    def _validation_repository_scope(self, body: Mapping[str, Any]) -> str:
        """Return the checked nested repository namespace for this task.

        Git mutation authority remains rooted at the accelerator checkout.
        Owner-relative outputs are projected into that root under this
        namespace, while validations enter the same configured repository.
        """

        owner = _owning_repository(body)
        if not owner or owner == _ROOT_REPOSITORY_AUTHORITY:
            return ""
        if owner not in self.worktree_submodule_paths:
            raise DatabasePortalBridgeError(
                f"owning repository {owner!r} is not a configured worktree submodule"
            )
        if self.repository_root is None:
            raise DatabasePortalBridgeError(
                "nested owning repository cannot be verified without repository_root"
            )
        try:
            root = self.repository_root.resolve(strict=True)
            candidate = (root / owner).resolve(strict=True)
            candidate.relative_to(root)
        except (OSError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                f"owning repository {owner!r} is unavailable or outside repository_root"
            ) from exc
        if not candidate.is_dir() or not (candidate / ".git").exists():
            raise DatabasePortalBridgeError(
                f"owning repository {owner!r} is not an initialized nested Git repository"
            )
        return owner

    @staticmethod
    def _scope_outputs(outputs: Sequence[str], repository: str) -> list[str]:
        """Project owner-relative paths into the superproject namespace.

        The owner is prepended exactly once.  A datasets-local package path
        such as ``ipfs_datasets_py/logic/api.py`` therefore intentionally
        becomes ``ipfs_datasets_py/ipfs_datasets_py/logic/api.py`` in the
        accelerator worktree.
        """

        scoped: list[str] = []
        for output in outputs:
            relative = _safe_output_path(output)
            projected = f"{repository}/{relative}" if repository else relative
            projected = _safe_output_path(projected)
            if projected not in scoped:
                scoped.append(projected)
        return scoped

    @staticmethod
    def _scope_validations(validations: Sequence[str], repository: str) -> list[str]:
        if not repository:
            return list(validations)
        if not validations:
            return []
        unscoped: list[str] = []
        for command in validations:
            command_root = validation_command_repository_root(command)
            if command_root is None:
                raise DatabasePortalBridgeError(
                    "nested-repository validation command has unsafe shell structure"
                )
            if command_root == "":
                unscoped.append(command)
            elif command_root == repository:
                if len(validations) != 1:
                    raise DatabasePortalBridgeError(
                        "multiple nested-repository validations must be unscoped"
                    )
                return [command]
            else:
                raise DatabasePortalBridgeError(
                    "validation command repository root conflicts with owning repository"
                )
        # The Markdown projection has one Validation field.  Emit exactly one
        # leading repository transition and fail fast across multiple typed
        # validation records; independently prefixing each record would make
        # the second ``cd`` relative to the already-entered nested repository.
        value = (
            f"cd {shlex.quote(repository)} && "
            + " && ".join(dict.fromkeys(unscoped))
        )
        if validation_command_repository_root(value) != repository:
            raise DatabasePortalBridgeError(
                "scoped validation command does not preserve repository authority"
            )
        return [value]

    def _paths(self, attempt: Any) -> DatabasePortalAttemptPaths:
        attempt_key = hashlib.sha256(str(attempt.attempt_id).encode("utf-8")).hexdigest()[:24]
        root = self.attempt_root / attempt_key
        return DatabasePortalAttemptPaths(
            root=root,
            task_projection=root / "task-projection.md",
            binding=root / "database-attempt-binding.json",
            state=root / "portal-task-state.json",
            strategy=root / "portal-strategy.json",
            events=root / "portal-events.jsonl",
            implementation_logs=root / "implementation-logs",
        )

    @staticmethod
    def _record_for_attempt(task_source: Any, attempt: Any) -> Any:
        getter = getattr(task_source, "get_task", None) or getattr(task_source, "get", None)
        if not callable(getter):
            raise DatabasePortalBridgeError("database task source does not expose get_task()")
        record = getter(str(attempt.task_cid))
        if record is None:
            raise DatabasePortalBridgeError(
                f"claimed database task {attempt.task_cid!r} disappeared"
            )
        if str(getattr(record, "task_cid", "")) != str(attempt.task_cid):
            raise DatabasePortalBridgeError("database task identity changed")
        attempt_alias = str(getattr(attempt, "task_alias", "") or "")
        record_alias = str(getattr(record, "task_alias", "") or "")
        if attempt_alias and record_alias and attempt_alias != record_alias:
            raise DatabasePortalBridgeError("database task alias changed")
        return record

    def _binding(self, attempt: Any, record: Any, seed: str) -> dict[str, Any]:
        body = dict(getattr(record, "body", {}) or {})
        payload = {
            "schema": DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
            "interface": self.INTERFACE,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": str(
                getattr(record, "task_alias", "")
                or getattr(attempt, "task_alias", "")
                or attempt.task_cid
            ),
            "goal_cid": str(getattr(record, "goal_cid", "") or ""),
            "plan_cid": str(getattr(record, "plan_cid", "") or ""),
            "task_revision": int(getattr(record, "revision", 0) or 0),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "task_body_digest": _sha256_bytes(_canonical_json(body)),
            "projection_seed_digest": _sha256_bytes(seed.encode("utf-8")),
            "projection_immutable_digest": _projection_immutable_digest(seed),
            "authoritative_task_store": "duckdb",
            "projection_authority": False,
        }
        payload["binding_id"] = _sha256_bytes(_canonical_json(payload))
        return payload

    def _render_projection(self, attempt: Any, record: Any) -> str:
        body = dict(getattr(record, "body", {}) or {})
        canonical_task_key, canonical_task_cid = _canonical_projection_identity(
            record,
            body,
        )
        alias = _line_value(
            getattr(record, "task_alias", "")
            or getattr(attempt, "task_alias", "")
            or attempt.task_cid
        )
        if not alias or any(character.isspace() for character in alias):
            raise DatabasePortalBridgeError("database task alias is not projection-safe")
        title = _line_value(
            body.get("objective") or body.get("title") or body.get("description") or alias
        )
        repository_scope = self._validation_repository_scope(body)
        outputs = self._scope_outputs(_output_values(record, body), repository_scope)
        validations = self._scope_validations(
            _validation_values(record, body),
            repository_scope,
        )
        acceptance = _acceptance_value(record, body)
        priority = _line_value(
            getattr(record, "priority", "") or body.get("priority") or "P2"
        )
        reserved = {
            "status",
            "completion",
            # Operational status receipts are accelerator-owned control-plane
            # state.  They are neither provider context nor semantic task
            # input, and status CASes must not invalidate an immutable Portal
            # attempt projection.
            "completion receipt",
            "completion_receipt",
            "priority",
            "track",
            "depends on",
            "depends_on",
            "outputs",
            "validation",
            "validations",
            "validation_commands",
            "acceptance",
            "canonical task key",
            "canonical_task_key",
            "canonical task cid",
            "canonical_task_cid",
            "task key",
            "task_key",
        }
        lines = [
            "# Database attempt projection (non-authoritative)",
            "",
            f"## {alias} {title}",
            "",
            "- Status: ready",
            f"- Completion: {_line_value(body.get('completion') or 'auto')}",
            f"- Priority: {priority}",
            f"- Track: {_line_value(body.get('track') or 'implementation')}",
            "- Depends on:",
            f"- Outputs: {', '.join(outputs)}",
            f"- Validation: {' ; '.join(validations)}",
            f"- Acceptance: {acceptance}",
            f"- Database task CID: {_line_value(attempt.task_cid)}",
            f"- Database attempt ID: {_line_value(attempt.attempt_id)}",
            f"- Database claim ID: {_line_value(attempt.claim_id)}",
            f"- Database dependency CIDs: {_line_value(getattr(record, 'dependencies', ()))}",
            f"- Canonical task key: {canonical_task_key}",
            f"- Canonical task CID: {canonical_task_cid}",
            "- Projection authority: false",
        ]
        for key in sorted(body):
            normalized = str(key).strip().lower().replace("_", " ")
            if not normalized or normalized in reserved:
                continue
            if "credential" in normalized or "secret" in normalized:
                continue
            value = _line_value(body[key])
            if value:
                label = " ".join(word.capitalize() for word in normalized.split())
                lines.append(f"- {label}: {value}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _read_binding(path: Path) -> Mapping[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding is unreadable"
            ) from exc
        if not isinstance(value, Mapping):
            raise DatabasePortalBridgeError("database Portal attempt binding is malformed")
        return value

    def _ensure_attempt_projection(
        self, attempt: Any, record: Any
    ) -> tuple[DatabasePortalAttemptPaths, Mapping[str, Any]]:
        paths = self._paths(attempt)
        seed = self._render_projection(attempt, record)
        expected = self._binding(attempt, record, seed)
        paths.root.mkdir(parents=True, exist_ok=True)
        if paths.binding.exists():
            observed = self._read_binding(paths.binding)
            if observed != expected:
                raise DatabasePortalBridgeError(
                    "database Portal attempt binding changed across resume"
                )
        else:
            _atomic_write(
                paths.binding,
                json.dumps(expected, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            )
        if not paths.task_projection.exists():
            _atomic_write(paths.task_projection, seed.encode("utf-8"))
        self._verify_projection(paths, expected)
        return paths, expected

    @staticmethod
    def _verify_projection(paths: DatabasePortalAttemptPaths, binding: Mapping[str, Any]) -> str:
        try:
            text = paths.task_projection.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DatabasePortalBridgeError("Portal task projection is unreadable") from exc
        if _projection_immutable_digest(text) != str(
            binding.get("projection_immutable_digest") or ""
        ):
            raise DatabasePortalBridgeError(
                "Portal task projection changed outside its mutable status field"
            )
        headers = _HEADER.findall(text)
        if headers != [str(binding.get("task_alias") or "")]:
            raise DatabasePortalBridgeError(
                "Portal task projection no longer contains exactly the claimed task"
            )
        return text

    @staticmethod
    def _has_completion_event(paths: DatabasePortalAttemptPaths, alias: str) -> bool:
        if not paths.events.is_file():
            return False
        try:
            lines = paths.events.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return False
        for line in reversed(lines[-4096:]):
            try:
                event = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                isinstance(event, Mapping)
                and event.get("type") == "task_completed"
                and str(event.get("task_id") or "") == alias
            ):
                return True
        return False

    @staticmethod
    def _terminal_failure(result: Mapping[str, Any]) -> str:
        if result.get("blocked") is True:
            reason = str(result.get("reason") or "portal_execution_blocked")
            if is_protected_recovery_fence_contention(reason):
                # Peer-owner recovery is a wait, not a task defect. The
                # typed-deferral classifier admits retry; do not CAS blocked.
                return ""
            return reason
        implementation = result.get("implementation_result")
        if not isinstance(implementation, Mapping):
            return ""
        if implementation.get("deferred") is True:
            return str(implementation.get("reason") or "portal_execution_deferred")
        returncode = implementation.get("returncode")
        if isinstance(returncode, int) and not isinstance(returncode, bool) and returncode != 0:
            return str(implementation.get("reason") or "portal_provider_failed")
        if implementation.get("skipped") is True:
            return str(implementation.get("reason") or "portal_execution_skipped")
        return ""

    @staticmethod
    def _typed_deferral(
        result: Mapping[str, Any],
    ) -> tuple[str, int] | None:
        """Return exact Portal deferral data without parsing reason text."""

        if result.get("blocked") is True:
            reason = str(result.get("reason") or "")
            if is_protected_recovery_fence_contention(reason):
                raw_backoff = result.get(
                    "backoff_seconds",
                    FENCE_CONTENTION_BACKOFF_SECONDS,
                )
                if (
                    isinstance(raw_backoff, bool)
                    or not isinstance(raw_backoff, int)
                    or raw_backoff < 0
                    or raw_backoff > _MAX_DATABASE_PORTAL_BACKOFF_SECONDS
                ):
                    raise DatabasePortalBridgeError(
                        "Portal fence deferral returned an invalid "
                        "backoff_seconds value"
                    )
                return (
                    reason or "external_protected_checkout_recovery_required",
                    int(raw_backoff),
                )
        implementation = result.get("implementation_result")
        if not isinstance(implementation, Mapping):
            return None
        # ``attempt_consumed=false``/``provider_dispatched=false`` also
        # describe a successful deterministic zero-provider closure.  Only
        # the explicit closed deferral signal grants retry semantics.
        if implementation.get("deferred") is not True:
            return None
        # Older typed deferrals predate the duration field.  They retain a
        # conservative bounded default instead of silently becoming a
        # zero-delay reconstruction loop.
        raw_backoff = implementation.get("backoff_seconds", 300)
        if (
            isinstance(raw_backoff, bool)
            or not isinstance(raw_backoff, int)
            or raw_backoff < 0
            or raw_backoff > _MAX_DATABASE_PORTAL_BACKOFF_SECONDS
        ):
            raise DatabasePortalBridgeError(
                "Portal deferral returned an invalid backoff_seconds value"
            )
        return (
            str(implementation.get("reason") or "portal_execution_deferred"),
            int(raw_backoff),
        )

    @staticmethod
    def _looks_like_validation_retry(
        implementation: Mapping[str, Any],
    ) -> bool:
        """Select only the closed post-dispatch validation-failure shape."""

        validation = implementation.get("validation_result")
        return bool(
            implementation.get("returncode") not in (None, 0)
            and implementation.get("attempt_consumed") is True
            and implementation.get("provider_dispatched") is True
            and isinstance(validation, Mapping)
            and validation.get("attempted") is True
            and validation.get("passed") is False
            and validation.get("reason") == "declared_validation_failed"
        )

    @staticmethod
    def _verified_event_chain(paths: DatabasePortalAttemptPaths) -> list[dict[str, Any]]:
        """Read one bounded attempt-local event chain without repairing it."""

        try:
            size = paths.events.stat().st_size
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "validation retry has no durable Portal event stream"
            ) from exc
        if size <= 0 or size > _MAX_DATABASE_PORTAL_EVENT_BYTES:
            raise DatabasePortalBridgeError(
                "validation retry Portal event stream exceeds its closed bound"
            )
        try:
            raw_lines = paths.events.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "validation retry Portal event stream is unreadable"
            ) from exc
        if not raw_lines or len(raw_lines) > _MAX_DATABASE_PORTAL_EVENTS:
            raise DatabasePortalBridgeError(
                "validation retry Portal event population is outside its closed bound"
            )

        events: list[dict[str, Any]] = []
        prior_event_id = ""
        stream_id = ""
        snapshot_id = ""
        for ordinal, line in enumerate(raw_lines, start=1):
            try:
                event = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DatabasePortalBridgeError(
                    "validation retry Portal event stream contains invalid JSON"
                ) from exc
            if not isinstance(event, dict):
                raise DatabasePortalBridgeError(
                    "validation retry Portal event stream contains a non-object"
                )
            body = dict(event)
            claimed_event_id = str(body.pop("event_id", "") or "")
            try:
                encoded = json.dumps(
                    body,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError, RecursionError) as exc:
                raise DatabasePortalBridgeError(
                    "validation retry Portal event is not canonical JSON"
                ) from exc
            expected_event_id = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
            current_stream = str(event.get("stream_id") or "")
            current_snapshot = str(event.get("snapshot_id") or "")
            sequence = event.get("sequence")
            if (
                claimed_event_id != expected_event_id
                or not current_stream
                or not current_snapshot
                or isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence != ordinal
                or str(event.get("previous_event_id") or "") != prior_event_id
                or (stream_id and current_stream != stream_id)
                or (snapshot_id and current_snapshot != snapshot_id)
            ):
                raise DatabasePortalBridgeError(
                    "validation retry Portal event chain failed identity verification"
                )
            stream_id = current_stream
            snapshot_id = current_snapshot
            prior_event_id = claimed_event_id
            events.append(event)
        return events

    def _preserved_commit_exists(
        self,
        *,
        commit: str,
        rescue_branch: str,
    ) -> bool:
        """Independently bind the claimed rescue ref to the preserved commit."""

        if self.repository_root is None:
            return False
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            return False
        if (
            not rescue_branch.startswith("rescue/")
            or ".." in rescue_branch
            or "@{" in rescue_branch
            or "\\" in rescue_branch
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", rescue_branch)
        ):
            return False
        try:
            checked = subprocess.run(
                ["git", "check-ref-format", f"refs/heads/{rescue_branch}"],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
            resolved = subprocess.run(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    f"refs/heads/{rescue_branch}^{{commit}}",
                ],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return (
            checked.returncode == 0
            and resolved.returncode == 0
            and resolved.stdout.strip() == commit
        )

    def _validation_retry_receipt(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        implementation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Reproduce the one post-dispatch failure class eligible to retry."""

        if self.max_task_attempts <= 0:
            return None
        attempt_number = getattr(attempt, "attempt_number", 0)
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number < 1
        ):
            return None
        if implementation is not None and not self._looks_like_validation_retry(
            implementation
        ):
            return None

        events = self._verified_event_chain(paths)
        alias = str(binding.get("task_alias") or "")
        task_cid = str(attempt.task_cid)
        matching_finished = [
            (index, event)
            for index, event in enumerate(events)
            if event.get("type") == "implementation_finished"
            and str(event.get("task_id") or "") == alias
            and str(event.get("canonical_task_cid") or event.get("task_cid") or "")
            == task_cid
        ]
        if not matching_finished:
            return None
        finished_index, finished = matching_finished[-1]
        if not self._looks_like_validation_retry(finished):
            return None
        portal_attempt = finished.get("attempt")
        # The outer coordination attempt number is a monotone database fence,
        # not this current-schema retry budget.  Legacy attempts can therefore
        # legitimately make it much larger than max_task_attempts.  Portal's
        # independently replayed per-task attempt counter is the bounded
        # generation and is carried into the next private attempt state.
        if (
            isinstance(portal_attempt, bool)
            or not isinstance(portal_attempt, int)
            or portal_attempt < 1
            or portal_attempt >= self.max_task_attempts
        ):
            return None
        if implementation is not None:
            for field in (
                "attempt",
                "returncode",
                "attempt_consumed",
                "provider_dispatched",
                "implementation_commit",
            ):
                if implementation.get(field) != finished.get(field):
                    return None

        validation = finished.get("validation_result")
        assert isinstance(validation, Mapping)
        proposal_gate = validation.get("proposal_gate")
        review = validation.get("failure_review")
        dag = validation.get("validation_dag_receipt")
        preservation = finished.get("failed_preservation_result")
        if not all(
            isinstance(value, Mapping)
            for value in (proposal_gate, review, dag, preservation)
        ):
            return None
        assert isinstance(proposal_gate, Mapping)
        assert isinstance(review, Mapping)
        assert isinstance(dag, Mapping)
        assert isinstance(preservation, Mapping)
        board_completion = finished.get("board_completion")
        merge_result = finished.get("merge_result")
        preservation_commit = preservation.get("commit_result")
        if not all(
            isinstance(value, Mapping)
            for value in (board_completion, merge_result, preservation_commit)
        ):
            return None
        assert isinstance(board_completion, Mapping)
        assert isinstance(merge_result, Mapping)
        assert isinstance(preservation_commit, Mapping)

        returncode = validation.get("returncode")
        coverage_errors = validation.get("coverage_errors")
        reason_codes = review.get("reason_codes")
        nodes = dag.get("nodes")
        if (
            isinstance(returncode, bool)
            or not isinstance(returncode, int)
            or returncode == 0
            or validation.get("auto_rescue_terminal") is not True
            or validation.get("completion_authoritative") is not False
            or validation.get("merge_eligible") is not False
            or coverage_errors not in ([], ())
            or proposal_gate.get("attempted") is not True
            or proposal_gate.get("accepted") is not True
            or proposal_gate.get("reason_codes") not in ([], ())
            or review.get("decision") != "guide_rescue"
            or set(str(item) for item in (reason_codes or ()))
            != {"validation_command_failed"}
            or review.get("denied_paths") not in ([], ())
            or review.get("out_of_scope_paths") not in ([], ())
            or review.get("contract_gap_paths") not in ([], ())
            or review.get("missing_expected_outputs") not in ([], ())
            or review.get("justified_paths") not in ([], ())
            or dag.get("passed") is not False
            or dag.get("coverage_complete") is not True
            or dag.get("uncovered_impact") is not False
            or not isinstance(nodes, Sequence)
            or isinstance(nodes, (str, bytes, bytearray, memoryview))
            or not any(
                isinstance(node, Mapping)
                and node.get("mandatory") is True
                and node.get("selected") is True
                and node.get("disposition") == "failed"
                and isinstance(node.get("returncode"), int)
                and not isinstance(node.get("returncode"), bool)
                and int(node.get("returncode")) != 0
                and bool(str(node.get("result_digest") or ""))
                for node in nodes
            )
            or finished.get("protected_path_violation") is not None
            or board_completion.get("complete") is True
            or merge_result.get("merged") is True
            or merge_result.get("queued") is True
        ):
            return None
        for container in (finished, validation, proposal_gate, review):
            for field in (
                "denied_effects",
                "forbidden_effects",
                "unauthorized_effects",
            ):
                if container.get(field) not in (None, [], ()):
                    return None

        proposal_id = str(proposal_gate.get("proposal_id") or "")
        proposal_receipt_id = str(proposal_gate.get("receipt_id") or "")
        proposal_policy_id = str(proposal_gate.get("policy_id") or "")
        validation_receipt_id = str(dag.get("receipt_id") or "")
        review_receipt_id = str(review.get("receipt_id") or "")
        commit = str(finished.get("implementation_commit") or "")
        preserved_commit = str(preservation.get("preserved_commit") or "")
        rescue_branch = str(preservation.get("rescue_branch") or "")
        changed_paths = tuple(str(item) for item in (proposal_gate.get("changed_paths") or ()))
        if (
            not all(
                (
                    proposal_id,
                    proposal_receipt_id,
                    proposal_policy_id,
                    validation_receipt_id,
                    review_receipt_id,
                )
            )
            or not changed_paths
            or len(set(changed_paths)) != len(changed_paths)
            or dag.get("proposal_receipt_id") != proposal_receipt_id
            or dag.get("objective_id") != task_cid
            or tuple(dag.get("changed_paths") or ()) != changed_paths
            or preservation.get("preserved") is not True
            or preservation.get("implementation_commit") != commit
            or preserved_commit != commit
            or preservation_commit.get("committed") is not True
            or preservation_commit.get("commit") != commit
            or not self._preserved_commit_exists(
                commit=commit,
                rescue_branch=rescue_branch,
            )
        ):
            return None

        preservation_matches = [
            (index, event)
            for index, event in enumerate(events[:finished_index])
            if event.get("type") == "failed_validation_worktree_preserved"
            and str(event.get("task_id") or "") == alias
            and str(event.get("canonical_task_cid") or "") == task_cid
            and event.get("attempt") == finished.get("attempt")
            and event.get("preserved") is True
            and event.get("implementation_commit") == commit
            and event.get("preserved_commit") == commit
            and event.get("rescue_branch") == rescue_branch
        ]
        if not preservation_matches:
            return None
        preservation_index, preservation_event = preservation_matches[-1]
        proposal_matches = [
            (index, event)
            for index, event in enumerate(events[:preservation_index])
            if event.get("type") == "implementation_proposal_validated"
            and str(event.get("task_id") or "") == alias
            and str(event.get("canonical_task_cid") or "") == task_cid
            and event.get("attempted") is True
            and event.get("accepted") is True
            and event.get("reason_codes") in ([], ())
            and event.get("proposal_id") == proposal_id
            and event.get("receipt_id") == proposal_receipt_id
            and event.get("policy_id") == proposal_policy_id
            and tuple(event.get("changed_paths") or ()) == changed_paths
        ]
        if not proposal_matches:
            return None
        proposal_index, proposal_event = proposal_matches[-1]
        output_matches = [
            (index, event)
            for index, event in enumerate(events[:proposal_index])
            if event.get("type") == "implementation_expected_outputs_checked"
            and str(event.get("task_id") or "") == alias
            and str(event.get("canonical_task_cid") or "") == task_cid
            and event.get("proposal_id") == proposal_id
            and event.get("passed") is True
            and event.get("issues") in ([], ())
            and tuple(event.get("expected_paths") or ()) == changed_paths
            and tuple(event.get("staged_paths") or ()) == changed_paths
            and event.get("force_staged_paths") in ([], ())
        ]
        if not output_matches:
            return None
        _output_index, output_event = output_matches[-1]

        receipt = {
            "schema": DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA,
            "disposition": "retry",
            "reason": "declared_validation_failed",
            "task_cid": task_cid,
            "task_alias": alias,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "attempt_number": int(attempt_number),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "portal_attempt": int(portal_attempt),
            "typed_retry_generation": int(portal_attempt),
            "retry_budget_basis": "portal_attempt",
            "legacy_database_attempts_excluded": True,
            "max_task_attempts": int(self.max_task_attempts),
            "remaining_task_attempts": int(
                self.max_task_attempts - portal_attempt
            ),
            "attempt_consumed": True,
            "provider_dispatched": True,
            "backoff_seconds": 0,
            "implementation_commit": commit,
            "rescue_branch": rescue_branch,
            "binding_id": str(binding.get("binding_id") or ""),
            "events_digest": _sha256_file(paths.events),
            "event_stream_id": str(finished.get("stream_id") or ""),
            "expected_output_event_id": str(output_event.get("event_id") or ""),
            "proposal_event_id": str(proposal_event.get("event_id") or ""),
            "preservation_event_id": str(preservation_event.get("event_id") or ""),
            "implementation_event_id": str(finished.get("event_id") or ""),
            "proposal_id": proposal_id,
            "proposal_receipt_id": proposal_receipt_id,
            "proposal_policy_id": proposal_policy_id,
            "validation_receipt_id": validation_receipt_id,
            "failure_review_receipt_id": review_receipt_id,
            "changed_paths": list(changed_paths),
            "authoritative_validation_executed": True,
            "proposal_policy_accepted": True,
            "output_policy_passed": True,
            "denial_findings": [],
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def recover_validation_retry(self, attempt: Any) -> Mapping[str, Any]:
        """Reproduce retry evidence for a previously terminalized attempt.

        This reads only the attempt-local projection and immutable event
        evidence.  It does not update a task, claim, queue, or execution row.
        """

        record = self._record_for_attempt(self.task_source, attempt)
        paths = self._paths(attempt)
        if not (
            paths.binding.is_file()
            and paths.task_projection.is_file()
            and paths.events.is_file()
        ):
            raise DatabasePortalBridgeError(
                "validation retry recovery artifacts are incomplete"
            )
        seed = self._render_projection(attempt, record)
        expected_binding = self._binding(attempt, record, seed)
        observed_binding = self._read_binding(paths.binding)
        observed_body = dict(observed_binding)
        observed_binding_id = str(observed_body.pop("binding_id", "") or "")
        observed_revision = observed_body.get("task_revision")
        current_revision = int(getattr(record, "revision", 0) or 0)
        # Control status transitions legitimately advance the task revision
        # after this immutable attempt binding was written.  All semantic
        # body and claim fields remain exact; only a positive historical task
        # revision not newer than the current control record is accepted.
        stable_expected = {
            key: value
            for key, value in expected_binding.items()
            if key
            not in {
                "binding_id",
                "task_revision",
                "task_body_digest",
                "projection_seed_digest",
                "projection_immutable_digest",
            }
        }
        stable_observed = {
            key: value
            for key, value in observed_binding.items()
            if key
            not in {
                "binding_id",
                "task_revision",
                "task_body_digest",
                "projection_seed_digest",
                "projection_immutable_digest",
            }
        }
        observed_projection = self._verify_projection(paths, observed_binding)
        if (
            observed_binding_id
            != _sha256_bytes(_canonical_json(observed_body))
            or isinstance(observed_revision, bool)
            or not isinstance(observed_revision, int)
            or observed_revision < 1
            or current_revision < observed_revision
            or stable_observed != stable_expected
            or _projection_recovery_digest(observed_projection)
            != _projection_recovery_digest(seed)
        ):
            raise DatabasePortalBridgeError(
                "validation retry recovery binding does not match the claim"
            )
        receipt = self._validation_retry_receipt(
            attempt=attempt,
            paths=paths,
            binding=observed_binding,
        )
        if receipt is None:
            raise DatabasePortalBridgeError(
                "attempt is not eligible for typed validation retry recovery"
            )
        return receipt

    def _validation_retry_seed_from_record(
        self,
        *,
        attempt: Any,
        record: Any,
    ) -> dict[str, Any] | None:
        """Recover the prior retry receipt carried through the claim CAS."""

        body = dict(getattr(record, "body", {}) or {})
        status_receipt = body.get("completion_receipt")
        if not isinstance(status_receipt, Mapping):
            return None
        seed = status_receipt.get("validation_retry_seed")
        if seed is None:
            return None
        if (
            status_receipt.get("operation") != "database_claim"
            or status_receipt.get("attempt_id") != str(attempt.attempt_id)
            or status_receipt.get("claim_id") != str(attempt.claim_id)
            or status_receipt.get("attempt_number")
            != int(attempt.attempt_number)
            or status_receipt.get("fencing_token")
            != int(attempt.fencing_token)
            or status_receipt.get("fence_epoch") != int(attempt.fence_epoch)
            or status_receipt.get("lease_id")
            != str(getattr(attempt, "lease_id", "") or "")
            or not isinstance(seed, Mapping)
        ):
            raise DatabasePortalBridgeError(
                "database claim carries a malformed validation retry seed"
            )
        value = dict(seed)
        receipt_id = value.pop("receipt_id", None)
        changed_paths = seed.get("changed_paths")
        source_attempt_number = seed.get("attempt_number")
        target_attempt_number = getattr(attempt, "attempt_number", 0)
        source_portal_attempt = seed.get("portal_attempt")
        scoped_outputs = self._scope_outputs(
            _output_values(record, body),
            self._validation_repository_scope(body),
        )
        if (
            seed.get("schema") != DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA
            or seed.get("disposition") != "retry"
            or seed.get("reason") != "declared_validation_failed"
            or seed.get("task_cid") != str(attempt.task_cid)
            or seed.get("task_alias")
            != str(getattr(attempt, "task_alias", "") or "")
            or seed.get("attempt_consumed") is not True
            or seed.get("provider_dispatched") is not True
            or seed.get("proposal_policy_accepted") is not True
            or seed.get("output_policy_passed") is not True
            or seed.get("denial_findings") != []
            or seed.get("max_task_attempts") != self.max_task_attempts
            or isinstance(source_attempt_number, bool)
            or not isinstance(source_attempt_number, int)
            or isinstance(target_attempt_number, bool)
            or not isinstance(target_attempt_number, int)
            or target_attempt_number < 1
            or target_attempt_number <= source_attempt_number
            or str(seed.get("attempt_id") or "")
            == str(attempt.attempt_id)
            or status_receipt.get("validation_retry_source_attempt_id")
            != str(seed.get("attempt_id") or "")
            or isinstance(source_portal_attempt, bool)
            or not isinstance(source_portal_attempt, int)
            or source_portal_attempt < 1
            or source_portal_attempt >= self.max_task_attempts
            or seed.get("typed_retry_generation") != source_portal_attempt
            or seed.get("retry_budget_basis") != "portal_attempt"
            or seed.get("legacy_database_attempts_excluded") is not True
            or seed.get("remaining_task_attempts")
            != self.max_task_attempts - source_portal_attempt
            or not isinstance(changed_paths, list)
            or changed_paths != scoped_outputs
            or receipt_id != _sha256_bytes(_canonical_json(value))
            or not self._preserved_commit_exists(
                commit=str(seed.get("implementation_commit") or ""),
                rescue_branch=str(seed.get("rescue_branch") or ""),
            )
        ):
            raise DatabasePortalBridgeError(
                "database claim validation retry seed failed verification"
            )
        return dict(seed)

    def _initialize_validation_retry_seed(
        self,
        *,
        attempt: Any,
        record: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Project a checked prior candidate into the new private Portal state."""

        seed = self._validation_retry_seed_from_record(
            attempt=attempt,
            record=record,
        )
        if seed is None:
            return None
        alias = str(binding.get("task_alias") or "")
        task_cid = str(attempt.task_cid)
        _canonical_task_key, canonical_task_cid = _canonical_projection_identity(
            record,
            dict(getattr(record, "body", {}) or {}),
        )
        if canonical_task_cid != task_cid:
            raise DatabasePortalBridgeError(
                "validation retry seed task identity changed"
            )
        source_portal_attempt = seed.get("portal_attempt")
        if (
            isinstance(source_portal_attempt, bool)
            or not isinstance(source_portal_attempt, int)
            or source_portal_attempt < 1
        ):
            raise DatabasePortalBridgeError(
                "validation retry seed has no exact Portal attempt"
            )
        seed_body = {
            "schema": DATABASE_PORTAL_VALIDATION_RETRY_SEED_SCHEMA,
            "task_id": alias,
            "canonical_task_key": _canonical_task_key,
            "canonical_task_cid": task_cid,
            "source_database_attempt_id": str(seed.get("attempt_id") or ""),
            "target_database_attempt_id": str(attempt.attempt_id),
            "target_claim_id": str(attempt.claim_id),
            "source_retry_receipt_id": str(seed.get("receipt_id") or ""),
            "implementation_commit": str(seed.get("implementation_commit") or ""),
            "rescue_branch": str(seed.get("rescue_branch") or ""),
            "changed_paths": list(seed.get("changed_paths") or ()),
            "validation_retry_receipt": dict(seed),
            "completion_authoritative": False,
        }
        seed_body["seed_id"] = _sha256_bytes(_canonical_json(seed_body))

        existing_seed_event: Mapping[str, Any] | None = None
        if paths.events.exists():
            for event in self._verified_event_chain(paths):
                if (
                    event.get("type")
                    == "database_portal_validation_retry_seeded"
                    and event.get("seed_id") == seed_body["seed_id"]
                ):
                    existing_seed_event = event
                    break
            if existing_seed_event is None:
                raise DatabasePortalBridgeError(
                    "Portal attempt event stream predates its required retry seed"
                )
        else:
            existing_seed_event = append_jsonl_event(
                paths.events,
                "database_portal_validation_retry_seeded",
                seed_body,
            )

        state_seed = {
            "implementation_attempts": {alias: source_portal_attempt},
            "implementation_attempts_by_cid": {
                task_cid: source_portal_attempt,
            },
            "last_implementation_task_id": alias,
            "last_implementation_task_key": _canonical_task_key,
            "last_implementation_task_cid": task_cid,
            "last_implementation_returncode": 1,
            "last_implementation_branch": str(seed.get("rescue_branch") or ""),
            "last_implementation_commit": str(
                seed.get("implementation_commit") or ""
            ),
        }
        if paths.state.exists():
            try:
                current_state = json.loads(paths.state.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatabasePortalBridgeError(
                    "Portal retry seed state is unreadable"
                ) from exc
            if not isinstance(current_state, Mapping) or any(
                current_state.get(key) != value
                for key, value in state_seed.items()
            ):
                raise DatabasePortalBridgeError(
                    "Portal retry seed state conflicts with its source receipt"
                )
        else:
            _atomic_write(
                paths.state,
                json.dumps(state_seed, indent=2, sort_keys=True).encode("utf-8")
                + b"\n",
            )
        return {
            "seed_id": str(seed_body["seed_id"]),
            "seed_event_id": str(existing_seed_event.get("event_id") or ""),
            "source_retry_receipt_id": str(seed.get("receipt_id") or ""),
            "implementation_commit": str(seed.get("implementation_commit") or ""),
            "rescue_branch": str(seed.get("rescue_branch") or ""),
            "portal_attempt": source_portal_attempt,
        }

    def _acceptance_receipt(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        summaries: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        alias = str(binding.get("task_alias") or "")
        projection_text = self._verify_projection(paths, binding)
        if _projection_status(projection_text) not in _TERMINAL_STATUSES:
            raise DatabasePortalBridgeDeferred("Portal task projection is not complete")
        if not self._has_completion_event(paths, alias):
            raise DatabasePortalBridgeError(
                "Portal completion lacks a matching durable task_completed event"
            )
        evidence = {
            "binding_id": str(binding.get("binding_id") or ""),
            "task_cid": str(attempt.task_cid),
            "task_alias": alias,
            "attempt_id": str(attempt.attempt_id),
            "projection_digest": _sha256_bytes(projection_text.encode("utf-8")),
            "projection_immutable_digest": str(binding.get("projection_immutable_digest") or ""),
            "state_digest": _sha256_file(paths.state) if paths.state.is_file() else "",
            "events_digest": _sha256_file(paths.events),
            "portal_passes": [dict(item) for item in summaries],
        }
        evidence_digest = _sha256_bytes(_canonical_json(evidence))
        receipt = {
            "schema": self.RECEIPT_SCHEMA,
            "interface": self.INTERFACE,
            "status": "succeeded",
            "provider": "PortalImplementationDaemon",
            "execution_mode": "database-authoritative-portal-bridge",
            "accepted": True,
            "completion_authority": "DatabaseImplementationDaemon",
            "task_cid": str(attempt.task_cid),
            "task_alias": alias,
            "attempt_id": str(attempt.attempt_id),
            "binding_id": str(binding.get("binding_id") or ""),
            "evidence_digest": evidence_digest,
            "portal_evidence": evidence,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def run_provider(self, attempt: Any) -> Mapping[str, Any]:
        """Run bounded real Portal passes and return only accepted evidence."""

        record = self._record_for_attempt(self.task_source, attempt)
        paths, binding = self._ensure_attempt_projection(attempt, record)
        self._initialize_validation_retry_seed(
            attempt=attempt,
            record=record,
            paths=paths,
            binding=binding,
        )
        summaries: list[Mapping[str, Any]] = []
        daemon = self.portal_factory(
            paths,
            str(binding.get("task_alias") or attempt.task_cid),
        )
        if daemon is None or not callable(getattr(daemon, "run_once", None)):
            raise DatabasePortalBridgeError(
                "portal_factory did not return a Portal-compatible daemon"
            )
        try:
            for _pass_index in range(self.max_passes):
                projection = self._verify_projection(paths, binding)
                if _projection_status(
                    projection
                ) in _TERMINAL_STATUSES and self._has_completion_event(
                    paths, str(binding.get("task_alias") or "")
                ):
                    return self._acceptance_receipt(
                        attempt=attempt,
                        paths=paths,
                        binding=binding,
                        summaries=summaries,
                    )
                raw_result = daemon.run_once()
                if not isinstance(raw_result, Mapping):
                    raise DatabasePortalBridgeError("Portal daemon returned a non-object result")
                summary = _bounded_portal_result(raw_result)
                summaries.append(summary)
                self._verify_projection(paths, binding)
                deferral = self._typed_deferral(raw_result)
                if deferral is not None:
                    reason, backoff_seconds = deferral
                    raise DatabasePortalBridgeDeferred(
                        reason,
                        backoff_seconds=backoff_seconds,
                    )
                implementation = raw_result.get("implementation_result")
                if (
                    isinstance(implementation, Mapping)
                    and self._looks_like_validation_retry(implementation)
                ):
                    retry_receipt = self._validation_retry_receipt(
                        attempt=attempt,
                        paths=paths,
                        binding=binding,
                        implementation=implementation,
                    )
                    if retry_receipt is not None:
                        raise DatabasePortalValidationRetry(retry_receipt)
                failure = self._terminal_failure(raw_result)
                if failure:
                    raise DatabasePortalBridgeError(failure)
            return self._acceptance_receipt(
                attempt=attempt,
                paths=paths,
                binding=binding,
                summaries=summaries,
            )
        finally:
            close = getattr(daemon, "close_event_runtime", None) or getattr(daemon, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _require_accepted_provider(attempt: Any, provider_result: Mapping[str, Any]) -> str:
        if (
            provider_result.get("schema") != DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA
            or provider_result.get("interface") != DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE
            or provider_result.get("accepted") is not True
            or provider_result.get("status") != "succeeded"
            or provider_result.get("provider") != "PortalImplementationDaemon"
            or str(provider_result.get("task_cid") or "") != str(attempt.task_cid)
            or str(provider_result.get("attempt_id") or "") != str(attempt.attempt_id)
        ):
            raise DatabasePortalBridgeError(
                "database effect rejected unaccepted Portal provider evidence"
            )
        digest = str(provider_result.get("evidence_digest") or "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise DatabasePortalBridgeError(
                "database effect rejected malformed Portal evidence identity"
            )
        return digest

    def apply_effect(self, attempt: Any, provider_result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Bind the already-applied Portal effect to the database phase."""

        digest = self._require_accepted_provider(attempt, provider_result)
        return {
            "status": "applied",
            "effect": "portal-supervised-accepted-effect",
            "effect_key": f"portal:{attempt.task_cid}:{attempt.attempt_id}",
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "portal_receipt_id": str(provider_result.get("receipt_id") or ""),
            "evidence_digest": digest,
        }

    def validate_effect(self, attempt: Any, effect_result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Admit only an exact effect derived from accepted Portal evidence."""

        digest = str(effect_result.get("evidence_digest") or "")
        if (
            effect_result.get("status") != "applied"
            or effect_result.get("effect") != "portal-supervised-accepted-effect"
            or str(effect_result.get("task_cid") or "") != str(attempt.task_cid)
            or str(effect_result.get("attempt_id") or "") != str(attempt.attempt_id)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        ):
            raise DatabasePortalBridgeError(
                "database validation rejected unbound Portal effect evidence"
            )
        return {
            "outcome": "passed",
            "evidence_digest": digest,
            "argv": ["portal-supervisor-gates"],
            "validator": self.INTERFACE,
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "portal_receipt_id": str(effect_result.get("portal_receipt_id") or ""),
        }


__all__ = (
    "DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA",
    "DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE",
    "DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA",
    "DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA",
    "DATABASE_PORTAL_VALIDATION_RETRY_SEED_SCHEMA",
    "DatabasePortalAttemptPaths",
    "DatabasePortalBridgeDeferred",
    "DatabasePortalBridgeError",
    "DatabasePortalExecutionBridge",
    "DatabasePortalValidationRetry",
    "PortalDaemonFactory",
)
