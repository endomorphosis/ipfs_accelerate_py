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
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final

from ..validation.validation_commands import validation_command_repository_root

DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE: Final[str] = "DatabasePortalExecutionBridge@1"
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@1"
)
DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@1"
)
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "complete", "done"}
)
_MUTABLE_PROJECTION_LINE = re.compile(r"(?mi)^-\s*status\s*:\s*.*$")
_HEADER = re.compile(r"(?m)^##\s+([^\s]+)(?:\s+.*)?$")
_ROOT_REPOSITORY_AUTHORITY: Final[str] = "ipfs_accelerate_py"
_MAX_REPOSITORY_PATH_BYTES: Final[int] = 1024
_MAX_TASK_IDENTITY_BYTES: Final[int] = 4096
_MAX_DATABASE_PORTAL_BACKOFF_SECONDS: Final[int] = 86_400


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
    ) -> None:
        if not callable(portal_factory):
            raise TypeError("portal_factory must be callable")
        if isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer")
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
            return str(result.get("reason") or "portal_execution_blocked")
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
    "DatabasePortalAttemptPaths",
    "DatabasePortalBridgeDeferred",
    "DatabasePortalBridgeError",
    "DatabasePortalExecutionBridge",
    "PortalDaemonFactory",
)
