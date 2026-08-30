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
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..merge.checkout_lock import checkout_repository_id
from ..task_sources.task_identity import canonical_task_identity

DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE: Final[str] = "DatabasePortalExecutionBridge@1"
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@1"
)
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@2"
)
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@3"
)
DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@1"
)
DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@2"
)
DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@3"
)
DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@1"
)
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "complete", "done"}
)
_MUTABLE_PROJECTION_LINE = re.compile(r"(?mi)^-\s*status\s*:\s*.*$")
_HEADER = re.compile(r"(?m)^##\s+([^\s]+)(?:\s+.*)?$")
_OUTPUT_PATH_FIELDS: Final[tuple[str, ...]] = (
    "path",
    "output",
    "artifact_id",
    "fluent_id",
)
_DECLARED_OUTPUT_EFFECT_FIELDS: Final[frozenset[str]] = frozenset(
    {"effect_id", "declared_path", "effect"}
)
_MAX_ACCEPTED_SOURCE_EVENT_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_ACCEPTED_SOURCE_EVENT_LINES: Final[int] = 65_536


class DatabasePortalBridgeError(RuntimeError):
    """A database claim could not obtain trustworthy Portal evidence."""


class DatabasePortalBridgeDeferred(DatabasePortalBridgeError):
    """Portal execution made bounded progress but is not yet acceptable."""


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


def _canonical_transition_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DatabasePortalBridgeError(
            "Portal accepted-source transition is not canonical JSON"
        ) from exc


def _reject_duplicate_event_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatabasePortalBridgeError(
                "Portal accepted-source event repeats a JSON key"
            )
        result[key] = value
    return result


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise DatabasePortalBridgeError(
            f"could not read Portal attempt artifact {path.name!r}"
        ) from exc


def _accepted_source_events(
    path: Path,
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    """Read a bounded regular event log without following a link."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "Portal accepted-source events are unreadable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > _MAX_ACCEPTED_SOURCE_EVENT_BYTES
        ):
            raise DatabasePortalBridgeError(
                "Portal accepted-source event log is not a bounded regular file"
            )
        payload = bytearray()
        while len(payload) <= _MAX_ACCEPTED_SOURCE_EVENT_BYTES:
            block = os.read(
                descriptor,
                min(
                    65_536,
                    _MAX_ACCEPTED_SOURCE_EVENT_BYTES + 1 - len(payload),
                ),
            )
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "Portal accepted-source events are unreadable"
        ) from exc
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or len(payload) != before.st_size
        or len(payload) > _MAX_ACCEPTED_SOURCE_EVENT_BYTES
    ):
        raise DatabasePortalBridgeError(
            "Portal accepted-source event log changed while read"
        )
    try:
        text = bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatabasePortalBridgeError(
            "Portal accepted-source events are not UTF-8"
        ) from exc
    lines = text.splitlines()
    if len(lines) > _MAX_ACCEPTED_SOURCE_EVENT_LINES:
        raise DatabasePortalBridgeError(
            "Portal accepted-source event log exceeds its line bound"
        )
    records: list[Mapping[str, Any]] = []
    for line in lines:
        if not line:
            continue
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_event_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant: {value}")
                ),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "Portal accepted-source event log contains invalid JSON"
            ) from exc
        if not isinstance(record, Mapping):
            raise DatabasePortalBridgeError(
                "Portal accepted-source event is not an object"
            )
        records.append(record)
    return tuple(records), _sha256_bytes(bytes(payload))


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


def _canonical_declared_output_path(value: Any) -> str:
    """Return one exact repository-relative path or fail closed.

    This stricter profile applies only to the typed declared-output envelope.
    Legacy projected output strings retain their existing normalization in
    :func:`_mapping_path`.
    """

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value.splitlines() != [value]
        or value.lower() in {"none", "n/a"}
    ):
        raise DatabasePortalBridgeError(
            "database task declared-output path is malformed"
        )
    if "," in value or "\\" in value or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise DatabasePortalBridgeError(
            "database task declared-output path is malformed"
        )
    candidate = PurePosixPath(value)
    if (
        value.startswith("/")
        or candidate.is_absolute()
        or not candidate.parts
        or "." in candidate.parts
        or ".." in candidate.parts
        or candidate.as_posix() != value
        or (candidate.parts and candidate.parts[0].endswith(":"))
    ):
        raise DatabasePortalBridgeError(
            "database task declared-output path is not canonical and repo-relative"
        )
    return value


def _nested_declared_output_path(value: Mapping[str, Any]) -> str | None:
    """Resolve the closed IntentRepository declared-output envelope.

    IntentRepository deliberately stores the effect identity in the outer
    ``path`` column and the exact repository path in the canonical nested
    effect.  The nested path may replace that storage identity in a disposable
    Portal projection only when the whole typed declaration is exact and the
    two identity copies agree.
    """

    if (
        "declared_path" in value
        or value.get("effect") == "declared_output"
    ):
        raise DatabasePortalBridgeError(
            "database task declared-output declaration must be nested"
        )

    effect = value.get("effect")
    if not isinstance(effect, Mapping):
        return None
    is_declared_output = (
        "declared_path" in effect
        or effect.get("effect") == "declared_output"
    )
    if not is_declared_output:
        return None
    if set(effect) != _DECLARED_OUTPUT_EFFECT_FIELDS:
        raise DatabasePortalBridgeError(
            "database task declared-output effect is not a closed record"
        )
    if effect.get("effect") != "declared_output":
        raise DatabasePortalBridgeError(
            "database task declared-output effect kind is invalid"
        )
    effect_id = effect.get("effect_id")
    if (
        not isinstance(effect_id, str)
        or not effect_id
        or effect_id != effect_id.strip()
        or any(character.isspace() for character in effect_id)
    ):
        raise DatabasePortalBridgeError(
            "database task declared-output effect identity is malformed"
        )

    outer_identities: list[str] = []
    for field in _OUTPUT_PATH_FIELDS:
        if field not in value:
            continue
        identity = value[field]
        if (
            not isinstance(identity, str)
            or not identity
            or identity != identity.strip()
            or any(character.isspace() for character in identity)
        ):
            raise DatabasePortalBridgeError(
                "database task declared-output outer identity is malformed"
            )
        outer_identities.append(identity)
    if not outer_identities:
        raise DatabasePortalBridgeError(
            "database task declared-output outer identity is missing"
        )
    if len(set(outer_identities)) != 1:
        raise DatabasePortalBridgeError(
            "database task declared-output outer identities conflict"
        )
    if outer_identities[0] != effect_id:
        raise DatabasePortalBridgeError(
            "database task declared-output effect identity conflicts with its outer identity"
        )
    return _canonical_declared_output_path(effect.get("declared_path"))


def _mapping_path(value: Mapping[str, Any]) -> str:
    declared_path = _nested_declared_output_path(value)
    if declared_path is not None:
        return declared_path

    selected = [
        _line_value(value[field])
        for field in _OUTPUT_PATH_FIELDS
        if value.get(field)
    ]
    selected = [item for item in selected if item]
    if len(set(selected)) > 1:
        raise DatabasePortalBridgeError(
            "database task output path declarations conflict"
        )
    return selected[0] if selected else _line_value(value)


def _output_values(record: Any, body: Mapping[str, Any]) -> list[str]:
    raw = getattr(record, "outputs", ()) or body.get("outputs") or ()
    if isinstance(raw, (str, Mapping)):
        raw = (raw,)
    return list(
        dict.fromkeys(
            selected
            for item in raw
            if (
                selected := (
                    _mapping_path(item) if isinstance(item, Mapping) else _line_value(item)
                )
            )
        )
    )


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
                value = shlex.join(str(part) for part in argv)
            else:
                value = _line_value(item.get("command") or item.get("value") or item)
        else:
            value = _line_value(item)
        if value and value not in selected:
            selected.append(value)
    return selected


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
        repo_root: Path | str | None = None,
        board_namespace: str = "",
        configured_board_admission_cid: str = "",
        merge_target_branch: str = "",
        task_header_prefix: str = "## ",
        max_passes: int = 4,
    ) -> None:
        if not callable(portal_factory):
            raise TypeError("portal_factory must be callable")
        if isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer")
        self.task_source = task_source
        self.attempt_root = Path(attempt_root).absolute()
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self.board_namespace = str(board_namespace or "").strip()
        self.configured_board_admission_cid = str(
            configured_board_admission_cid or ""
        ).strip()
        self.merge_target_branch = str(merge_target_branch or "").strip()
        self.portal_factory = portal_factory
        self.task_header_prefix = str(task_header_prefix or "## ")
        self.max_passes = max_passes

    def _paths(self, attempt: Any) -> DatabasePortalAttemptPaths:
        attempt_key = hashlib.sha256(str(attempt.attempt_id).encode("utf-8")).hexdigest()[:24]
        root = self.attempt_root / attempt_key
        return DatabasePortalAttemptPaths(
            root=root,
            task_projection=root / "task-projection.runtime.todo.md",
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
        outputs = _output_values(record, body)
        validations = _validation_values(record, body)
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

    def _accepted_source_transition(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        task_alias: str,
        task_cid: str,
        merge_request_loader: Callable[[str], Any] | None = None,
    ) -> dict[str, Any] | None:
        """Reconstruct one exact landed transition before Quack completion.

        The Portal event file is only an observation.  This method resolves its
        bounded commit claims against Git and emits a content-addressed packet;
        the database completion CAS later makes that packet authoritative for
        the exact task revision.
        """

        if not paths.events.is_file():
            return None
        event_records, event_log_sha256 = _accepted_source_events(paths.events)
        direct_candidates = [
            (index, event)
            for index, event in enumerate(event_records)
            if (
                event.get("type") == "implementation_finished"
                and str(event.get("task_id") or "") == task_alias
                and event.get("returncode") == 0
                and event.get("board_completion")
                == {
                    "complete": True,
                    "pending_merge": False,
                    "reason": "merged_into_target",
                }
            )
        ]
        queued_candidates = [
            (index, event)
            for index, event in enumerate(event_records)
            if (
                event.get("type") == "implementation_finished"
                and str(event.get("task_id") or "") == task_alias
                and event.get("returncode") == 0
                and event.get("board_completion")
                == {
                    "complete": False,
                    "pending_merge": True,
                    "reason": "merge_queued_awaiting_integration",
                }
                and isinstance(event.get("merge_result"), Mapping)
                and event["merge_result"].get("queued") is True
            )
        ]
        reconciled_pairs: list[
            tuple[Mapping[str, Any], Mapping[str, Any]]
        ] = []
        for queued_index, queued_event in queued_candidates:
            implementation_commit = str(
                queued_event.get("implementation_commit") or ""
            )
            portal_attempt_number = queued_event.get("attempt")
            canonical_task_cid = str(
                queued_event.get("canonical_task_cid") or ""
            )
            canonical_task_key = str(
                queued_event.get("canonical_task_key") or ""
            )
            for reconciliation_index, reconciliation in enumerate(
                event_records
            ):
                if reconciliation_index <= queued_index:
                    continue
                if (
                    reconciliation.get("type") == "merge_reconciled"
                    and reconciliation.get("resolved") is True
                    and str(reconciliation.get("task_id") or "")
                    == task_alias
                    and reconciliation.get("attempt")
                    == portal_attempt_number
                    and str(
                        reconciliation.get("implementation_commit") or ""
                    )
                    == implementation_commit
                    and str(
                        reconciliation.get("canonical_task_cid") or ""
                    )
                    == canonical_task_cid
                    and str(
                        reconciliation.get("canonical_task_key") or ""
                    )
                    == canonical_task_key
                ):
                    reconciled_pairs.append(
                        (queued_event, reconciliation)
                    )
        candidate_count = len(direct_candidates) + len(reconciled_pairs)
        if candidate_count == 0:
            return None
        if candidate_count != 1:
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition is not unique"
            )
        normalized_binding = dict(binding)
        binding_id = str(normalized_binding.pop("binding_id", "") or "")
        expected_binding_fields = {
            "schema",
            "interface",
            "attempt_id",
            "claim_id",
            "task_cid",
            "task_alias",
            "goal_cid",
            "plan_cid",
            "task_revision",
            "fencing_token",
            "fence_epoch",
            "lease_id",
            "task_body_digest",
            "projection_seed_digest",
            "projection_immutable_digest",
            "authoritative_task_store",
            "projection_authority",
        }
        if (
            set(normalized_binding) != expected_binding_fields
            or binding.get("schema") != DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA
            or binding.get("interface") != self.INTERFACE
            or binding.get("attempt_id") != str(attempt.attempt_id)
            or binding.get("claim_id") != str(attempt.claim_id)
            or binding.get("task_cid") != task_cid
            or binding.get("task_alias") != task_alias
            or binding.get("fencing_token") != int(attempt.fencing_token)
            or binding.get("fence_epoch") != int(attempt.fence_epoch)
            or binding.get("authoritative_task_store") != "duckdb"
            or binding.get("projection_authority") is not False
            or binding_id != _sha256_bytes(_canonical_json(normalized_binding))
        ):
            raise DatabasePortalBridgeError(
                "Portal accepted-source database binding is inconsistent"
            )
        projection_text = self._verify_projection(paths, binding)
        try:
            # Imported lazily because implementation_daemon owns the parser and
            # imports this bridge.  Invocation happens only after both modules
            # are fully initialized.
            from .implementation_daemon import (
                parse_task_text,
                task_declared_output_paths,
            )

            parsed_tasks = parse_task_text(
                projection_text,
                path=paths.task_projection,
                task_header_prefix=self.task_header_prefix,
            )
            if len(parsed_tasks) != 1 or parsed_tasks[0].task_id != task_alias:
                raise DatabasePortalBridgeError(
                    "Portal accepted-source projection identity is ambiguous"
                )
            parsed_task = parsed_tasks[0]
            identity_metadata = dict(parsed_task.metadata)
            if parsed_task.canonical_task_key:
                identity_metadata["canonical task key"] = (
                    parsed_task.canonical_task_key
                )
            if parsed_task.canonical_task_cid:
                identity_metadata["canonical task cid"] = (
                    parsed_task.canonical_task_cid
                )
            canonical_identity = canonical_task_identity(
                {
                    "task_id": parsed_task.task_id,
                    "title": parsed_task.title,
                    "outputs": task_declared_output_paths(parsed_task),
                    "acceptance": parsed_task.acceptance,
                    "metadata": identity_metadata,
                },
                board_namespace=(
                    parsed_task.board_namespace
                    or self.board_namespace
                    or paths.task_projection.name
                ),
                source_path=paths.task_projection,
            )
        except DatabasePortalBridgeError:
            raise
        except Exception as exc:
            raise DatabasePortalBridgeError(
                "Portal accepted-source projection identity is invalid"
            ) from exc
        if self.repo_root is None:
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition has no repository authority"
            )
        reconciliation: Mapping[str, Any] | None = None
        if direct_candidates:
            event = direct_candidates[0][1]
            transition_schema = (
                DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA
            )
        else:
            event, reconciliation = reconciled_pairs[0]
            transition_schema = (
                DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA
            )
        merge = (
            reconciliation.get("merge_result")
            if reconciliation is not None
            else event.get("merge_result")
        )
        if not isinstance(merge, Mapping):
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition has no merge result"
            )
        queued_merge = event.get("merge_result")
        if not isinstance(queued_merge, Mapping):
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition has no queue binding"
            )
        baseline = str(event.get("baseline_ref") or "")
        implementation = str(event.get("implementation_commit") or "")
        proof = (
            reconciliation.get("integration_commit_proof")
            if reconciliation is not None
            else merge.get("integration_commit_proof")
        )
        invariant = (
            reconciliation.get("post_merge_declared_output_invariant")
            if reconciliation is not None
            else merge.get("post_merge_declared_output_invariant")
        )
        merge_commit = str(
            merge.get("merge_commit")
            or (
                proof.get("integration_commit")
                if isinstance(proof, Mapping)
                else ""
            )
            or (
                reconciliation.get("merge_commit")
                if reconciliation is not None
                else ""
            )
            or ""
        )
        target_branch = str(
            merge.get("target_branch")
            or (
                proof.get("target_branch")
                if isinstance(proof, Mapping)
                else ""
            )
            or ""
        )
        canonical_task_cid = str(
            event.get("canonical_task_cid") or ""
        )
        canonical_task_key = str(
            event.get("canonical_task_key") or ""
        )
        request_id = str(queued_merge.get("request_id") or "")
        portal_attempt_number = event.get("attempt")
        event_target_repository_id = str(
            event.get("target_repository_id") or ""
        )
        merge_target_repository_id = str(
            queued_merge.get("target_repository_id")
            or merge.get("target_repository_id")
            or ""
        )
        target_repository_id = (
            event_target_repository_id or merge_target_repository_id
        )
        expected_repository_id = checkout_repository_id(self.repo_root)
        if (
            any(re.fullmatch(r"[0-9a-f]{40}", item) is None for item in (
                baseline,
                implementation,
                merge_commit,
            ))
            or not target_branch
            or not request_id
            or isinstance(portal_attempt_number, bool)
            or not isinstance(portal_attempt_number, int)
            or portal_attempt_number < 1
            or str(event.get("board_namespace") or "") != self.board_namespace
            or target_branch != self.merge_target_branch
            or (
                event_target_repository_id
                and merge_target_repository_id
                and event_target_repository_id != merge_target_repository_id
            )
            or target_repository_id != expected_repository_id
            or merge.get("merged") is not True
            or (
                reconciliation is None
                and merge.get("returncode") != 0
            )
            or (
                reconciliation is None
                and str(merge.get("implementation_commit") or "")
                != implementation
            )
            or not isinstance(proof, Mapping)
            or proof.get("passed") is not True
            or proof.get("implementation_commit") != implementation
            or proof.get("integration_commit") != merge_commit
            or proof.get("integration_ref") != merge_commit
            or proof.get("target_branch") != target_branch
            or not isinstance(invariant, Mapping)
            or invariant.get("passed") is not True
            or invariant.get("repository_ref") != merge_commit
            or canonical_task_cid != canonical_identity.canonical_task_cid
            or canonical_task_key != canonical_identity.canonical_task_key
            or event.get("canonical_task_cid") != canonical_task_cid
            or event.get("canonical_task_key") != canonical_task_key
            or queued_merge.get("canonical_task_cid") != canonical_task_cid
            or queued_merge.get("canonical_task_key") != canonical_task_key
        ):
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition is inconsistent"
            )
        if reconciliation is not None:
            completion_persistence = reconciliation.get(
                "completion_persistence"
            )
            runtime_binding = (
                completion_persistence.get("runtime_taskboard_binding")
                if isinstance(completion_persistence, Mapping)
                else None
            )
            taskboard_snapshot = (
                completion_persistence.get("fsynced_taskboard_snapshot")
                if isinstance(completion_persistence, Mapping)
                else None
            )
            expected_completion = {task_alias: canonical_task_cid}
            try:
                runtime_projection_path = Path(
                    str(runtime_binding.get("path") or "")
                ).resolve()
                snapshot_projection_path = Path(
                    str(taskboard_snapshot.get("path") or "")
                ).resolve()
                expected_projection_path = paths.task_projection.resolve()
            except (AttributeError, OSError, TypeError, ValueError) as exc:
                raise DatabasePortalBridgeError(
                    "Portal reconciled-source completion is inconsistent"
                ) from exc
            if (
                reconciliation.get("reason")
                not in {
                    "merge_retried",
                    "completion_persistence_recovered_from_landed_rewrite",
                }
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(event.get("event_id") or ""),
                )
                is None
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(reconciliation.get("event_id") or ""),
                )
                is None
                or reconciliation.get("completion_task_cids")
                != expected_completion
                or not isinstance(completion_persistence, Mapping)
                or completion_persistence.get("passed") is not True
                or completion_persistence.get("reason")
                != "completion_persisted"
                or completion_persistence.get("durable_update") is not True
                or completion_persistence.get("status_persisted") is not True
                or completion_persistence.get("expected_task_ids")
                != [task_alias]
                or completion_persistence.get("completed_task_ids")
                != [task_alias]
                or completion_persistence.get("missing_task_ids") != []
                or completion_persistence.get("receipt_mismatches") != {}
                or not isinstance(runtime_binding, Mapping)
                or runtime_binding.get("passed") is not True
                or runtime_binding.get("authoritative") is not True
                or runtime_binding.get("ignored") is not True
                or runtime_binding.get("runtime_projection") is not True
                or runtime_projection_path != expected_projection_path
                or not isinstance(taskboard_snapshot, Mapping)
                or taskboard_snapshot.get("passed") is not True
                or taskboard_snapshot.get("reason")
                != "fsynced_taskboard_completion_proven"
                or taskboard_snapshot.get("runtime_projection") is not True
                or taskboard_snapshot.get("runtime_binding")
                != runtime_binding
                or taskboard_snapshot.get("expected_task_ids")
                != [task_alias]
                or taskboard_snapshot.get("observed_statuses")
                != {task_alias: "completed"}
                or taskboard_snapshot.get("observed_task_cids")
                != expected_completion
                or taskboard_snapshot.get("missing_task_ids") != []
                or taskboard_snapshot.get("ambiguous_task_ids") != []
                or taskboard_snapshot.get("status_mismatches") != {}
                or taskboard_snapshot.get("task_cid_mismatches") != {}
                or snapshot_projection_path != expected_projection_path
            ):
                raise DatabasePortalBridgeError(
                    "Portal reconciled-source completion is inconsistent"
                )
        if not callable(merge_request_loader):
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition has no merge-queue authority"
            )
        try:
            request = merge_request_loader(request_id)
            converter = getattr(request, "to_dict", None)
            request_record = (
                dict(converter())
                if callable(converter)
                else dict(request)
                if isinstance(request, Mapping)
                else None
            )
        except Exception as exc:
            raise DatabasePortalBridgeError(
                "Portal accepted-source merge request is unavailable"
            ) from exc
        if not isinstance(request_record, dict):
            raise DatabasePortalBridgeError(
                "Portal accepted-source merge request is unavailable"
            )
        request_metadata = request_record.get("metadata")
        request_task = (
            request_metadata.get("task")
            if isinstance(request_metadata, Mapping)
            else None
        )
        request_task_metadata = (
            request_task.get("metadata")
            if isinstance(request_task, Mapping)
            else None
        )
        completion_task_cids = (
            request_metadata.get("completion_task_cids")
            if isinstance(request_metadata, Mapping)
            else None
        )
        request_dedupe_key = str(request_record.get("dedupe_key") or "")
        request_status = str(request_record.get("status") or "")
        request_attempt = request_record.get("attempt")
        cancellation = (
            request_metadata.get("cancellation")
            if isinstance(request_metadata, Mapping)
            else None
        )
        direct_queue_terminal = bool(
            reconciliation is None
            and request_status == "completed"
            and request_attempt == portal_attempt_number
        )
        reconciled_queue_terminal = bool(
            reconciliation is not None
            and request_status == "cancelled"
            and isinstance(request_attempt, int)
            and not isinstance(request_attempt, bool)
            and request_attempt >= portal_attempt_number
            and isinstance(cancellation, Mapping)
            and set(cancellation) == {"at", "reason"}
            and isinstance(cancellation.get("at"), (int, float))
            and not isinstance(cancellation.get("at"), bool)
            and float(cancellation["at"]) >= 0.0
            and cancellation.get("reason") == "stale_quarantined_merge"
            and request_record.get("branch_name")
            == str(event.get("branch") or "")
        )
        if (
            request_record.get("request_id") != request_id
            or not (direct_queue_terminal or reconciled_queue_terminal)
            or request_record.get("task_id") != task_alias
            or request_record.get("commit_sha") != implementation
            or request_record.get("canonical_task_id") != canonical_task_cid
            or request_record.get("canonical_task_key") != canonical_task_key
            or not re.fullmatch(r"[0-9a-f]{64}", request_dedupe_key)
            or not isinstance(request_metadata, Mapping)
            or request_metadata.get("baseline_ref") != baseline
            or request_metadata.get("implementation_commit") != implementation
            or request_metadata.get("target_binding_schema")
            != "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
            or request_metadata.get("target_repository_id")
            != target_repository_id
            or request_metadata.get("target_branch") != target_branch
            or request_metadata.get("repo_root") != str(self.repo_root)
            or not isinstance(completion_task_cids, Mapping)
            or completion_task_cids.get(task_alias) != canonical_task_cid
            or not isinstance(request_task, Mapping)
            or request_task.get("task_id") != task_alias
            or request_task.get("board_namespace") != self.board_namespace
            or request_task.get("canonical_task_cid") != canonical_task_cid
            or request_task.get("canonical_task_key") != canonical_task_key
            or not isinstance(request_task_metadata, Mapping)
            or request_task_metadata.get("database attempt id")
            != str(attempt.attempt_id)
            or request_task_metadata.get("database claim id")
            != str(attempt.claim_id)
            or request_task_metadata.get("database task cid") != task_cid
        ):
            raise DatabasePortalBridgeError(
                "Portal accepted-source merge request is inconsistent"
            )
        merge_request_digest = _sha256_bytes(_canonical_json(request_record))
        git_environment = {
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }

        def git(*arguments: str) -> bytes:
            try:
                completed = subprocess.run(
                    ["/usr/bin/git", "--no-replace-objects", *arguments],
                    cwd=self.repo_root,
                    env=git_environment,
                    capture_output=True,
                    check=False,
                    timeout=10.0,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DatabasePortalBridgeError(
                    "Portal accepted-source Git proof is unavailable"
                ) from exc
            if completed.returncode != 0:
                raise DatabasePortalBridgeError(
                    "Portal accepted-source Git proof failed"
                )
            return completed.stdout

        parents = git("rev-list", "--parents", "-n", "1", merge_commit)
        parent_fields = parents.decode("ascii").strip().split()
        if (
            len(parent_fields) != 3
            or parent_fields[0] != merge_commit
            or parent_fields[2] != implementation
        ):
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition is not the exact Git merge"
            )
        integration_base_commit = parent_fields[1]
        target_advanced = integration_base_commit != baseline
        transition_proof = dict(proof)
        if target_advanced:
            # A task's dispatch baseline and the target's first parent at
            # integration are different identities when another fenced lane
            # lands first.  The completed merge request still binds the
            # immutable implementation parent; Git supplies the exact target
            # parent.  Both histories must descend from the dispatch baseline.
            try:
                git(
                    "merge-base",
                    "--is-ancestor",
                    baseline,
                    integration_base_commit,
                )
                git(
                    "merge-base",
                    "--is-ancestor",
                    baseline,
                    implementation,
                )
            except DatabasePortalBridgeError as exc:
                raise DatabasePortalBridgeError(
                    "Portal accepted-source transition is not the exact Git merge"
                ) from exc
            claimed_candidate_baselines = {
                str(value)
                for value in (
                    merge.get("candidate_baseline_ref"),
                    proof.get("candidate_baseline_ref"),
                )
                if value is not None and str(value)
            }
            claimed_integration_bases = {
                str(value)
                for value in (
                    merge.get("integration_base_commit"),
                    proof.get("integration_base_commit"),
                )
                if value is not None and str(value)
            }
            claimed_exact_topology = proof.get("exact_two_parent_merge")
            if (
                (
                    claimed_candidate_baselines
                    and claimed_candidate_baselines != {baseline}
                )
                or (
                    claimed_integration_bases
                    and claimed_integration_bases
                    != {integration_base_commit}
                )
                or claimed_exact_topology not in {None, True}
            ):
                raise DatabasePortalBridgeError(
                    "Portal accepted-source transition is not the exact Git merge"
                )
            transition_schema = (
                DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA
            )
            transition_proof.update(
                {
                    "candidate_baseline_ref": baseline,
                    "integration_base_commit": integration_base_commit,
                    "exact_two_parent_merge": True,
                }
            )
        implementation_tree = git(
            "rev-parse", f"{implementation}^{{tree}}"
        ).decode("ascii").strip()
        merge_tree = git("rev-parse", f"{merge_commit}^{{tree}}").decode(
            "ascii"
        ).strip()
        changed_path_diff_sha256 = _sha256_bytes(
            git(
                "diff-tree",
                "--no-commit-id",
                "--name-status",
                "-r",
                "-z",
                integration_base_commit,
                merge_commit,
            )
        )
        transition: dict[str, Any] = {
            "schema": transition_schema,
            "board_namespace": self.board_namespace,
            "configured_board_admission_cid": self.configured_board_admission_cid,
            "task_alias": task_alias,
            "database_task_cid": task_cid,
            "attempt_id": str(attempt.attempt_id),
            "attempt_number": int(attempt.attempt_number),
            "portal_attempt_number": portal_attempt_number,
            "claim_id": str(attempt.claim_id),
            "fencing_token": int(attempt.fencing_token),
            "database_attempt_binding": dict(binding),
            "canonical_task_cid": canonical_task_cid,
            "canonical_task_key": canonical_task_key,
            "request_id": request_id,
            "merge_request_digest": merge_request_digest,
            "merge_request_dedupe_key": request_dedupe_key,
            "target_repository_id": target_repository_id,
            "implementation_commit": implementation,
            "implementation_tree": implementation_tree,
            "merge_commit": merge_commit,
            "merge_tree": merge_tree,
            "target_branch": target_branch,
            "changed_path_diff_sha256": changed_path_diff_sha256,
            "integration_commit_proof": transition_proof,
            "declared_output_invariant": dict(invariant),
            "portal_event_log_sha256": event_log_sha256,
            "authority": "database_completion_cas_after_portal_and_git_verification",
            "task_completion_authority": False,
            "worker_self_approval": False,
        }
        if target_advanced:
            transition.update(
                {
                    "candidate_baseline_ref": baseline,
                    "integration_base_commit": integration_base_commit,
                }
            )
        else:
            # Preserve the accepted-source-transition@1/@2 byte vocabulary.
            transition["baseline_ref"] = baseline
        if reconciliation is not None:
            transition.update(
                {
                    "source_event_mode": "queued_merge_reconciliation",
                    "queued_implementation_event_id": str(
                        event.get("event_id") or ""
                    ),
                    "reconciliation_event_id": str(
                        reconciliation.get("event_id") or ""
                    ),
                    "merge_queue_terminal_status": request_status,
                    "merge_queue_attempt": request_attempt,
                    "merge_queue_cancellation_reason": str(
                        cancellation.get("reason") or ""
                    ),
                    "completion_persistence": dict(
                        reconciliation["completion_persistence"]
                    ),
                }
            )
        transition["transition_cid"] = _sha256_bytes(
            _canonical_transition_json(transition)
        )
        return transition

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
    def _is_external_protected_recovery_deferral(
        result: Mapping[str, Any],
    ) -> bool:
        """Recognize only the daemon's exact no-write owner deferral."""

        recovery = result.get("protected_checkout_recovery")
        write_count = result.get("write_count")
        return bool(
            result.get("blocked") is True
            and result.get("unchanged") is True
            and isinstance(write_count, int)
            and not isinstance(write_count, bool)
            and write_count == 0
            and result.get("implementation_result") is None
            and result.get("reason")
            == "external_protected_checkout_recovery_required"
            and isinstance(recovery, Mapping)
            and recovery.get("required") is True
            and recovery.get("adopted") is False
            and recovery.get("blocked") is True
            and recovery.get("recovered") is False
            and recovery.get("reason")
            == "external_protected_checkout_recovery_required"
            and recovery.get("protected_recovery_owner")
            == "implementation_supervisor"
            and bool(str(recovery.get("lock_path") or "").strip())
            and result.get("projection_delta") == {}
            and result.get("merge_reconciliation") == []
        )

    def _acceptance_receipt(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        summaries: Sequence[Mapping[str, Any]],
        merge_request_loader: Callable[[str], Any] | None = None,
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
        accepted_source_transition = self._accepted_source_transition(
            attempt=attempt,
            paths=paths,
            binding=binding,
            task_alias=alias,
            task_cid=str(attempt.task_cid),
            merge_request_loader=merge_request_loader,
        )
        if accepted_source_transition is not None:
            evidence["accepted_source_transition"] = accepted_source_transition
        evidence_digest = _sha256_bytes(_canonical_json(evidence))
        transition_schema = (
            accepted_source_transition.get("schema")
            if isinstance(accepted_source_transition, Mapping)
            else None
        )
        receipt = {
            "schema": (
                DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1
                if accepted_source_transition is None
                else self.RECEIPT_SCHEMA
                if transition_schema
                == DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA
                else DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2
            ),
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
        if accepted_source_transition is not None:
            receipt["accepted_source_transition"] = accepted_source_transition
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
        merge_queue = getattr(daemon, "merge_queue", None)
        merge_request_loader = getattr(merge_queue, "get", None)
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
                        merge_request_loader=merge_request_loader,
                    )
                raw_result = daemon.run_once()
                if not isinstance(raw_result, Mapping):
                    raise DatabasePortalBridgeError("Portal daemon returned a non-object result")
                summary = _bounded_portal_result(raw_result)
                summaries.append(summary)
                self._verify_projection(paths, binding)
                implementation = raw_result.get("implementation_result")
                if (
                    isinstance(implementation, Mapping)
                    and implementation.get("deferred") is True
                ):
                    raise DatabasePortalBridgeDeferred(
                        str(
                            implementation.get("reason")
                            or "portal_execution_deferred"
                        )
                    )
                if self._is_external_protected_recovery_deferral(raw_result):
                    raise DatabasePortalBridgeDeferred(
                        "external_protected_checkout_recovery_required"
                    )
                failure = self._terminal_failure(raw_result)
                if failure:
                    if (
                        "deferred" in failure
                        or "backoff" in failure
                        or "capacity" in failure
                        or "resource_claim" in failure
                        or failure
                        in {
                            "inflight_process",
                            "inflight_process_missing",
                            "worktree_lifecycle_claim_exists",
                        }
                    ):
                        raise DatabasePortalBridgeDeferred(failure)
                    raise DatabasePortalBridgeError(failure)
            return self._acceptance_receipt(
                attempt=attempt,
                paths=paths,
                binding=binding,
                summaries=summaries,
                merge_request_loader=merge_request_loader,
            )
        finally:
            close = getattr(daemon, "close_event_runtime", None) or getattr(daemon, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _require_accepted_provider(attempt: Any, provider_result: Mapping[str, Any]) -> str:
        schema = provider_result.get("schema")
        if (
            schema
            not in {
                DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1,
                DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2,
                DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA,
            }
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
        evidence = provider_result.get("portal_evidence")
        normalized_receipt = dict(provider_result)
        receipt_id = str(normalized_receipt.pop("receipt_id", "") or "")
        if (
            not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            or not isinstance(evidence, Mapping)
            or digest != _sha256_bytes(_canonical_json(evidence))
            or receipt_id != _sha256_bytes(_canonical_json(normalized_receipt))
        ):
            raise DatabasePortalBridgeError(
                "database effect rejected malformed Portal evidence identity"
            )
        transition = provider_result.get("accepted_source_transition")
        if schema == DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1 and transition is not None:
            raise DatabasePortalBridgeError(
                "database effect rejected a legacy receipt with a source transition"
            )
        if schema != DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1 and not isinstance(
            transition, Mapping
        ):
            raise DatabasePortalBridgeError(
                "database effect rejected a source-transition receipt without its transition"
            )
        if transition is not None:
            if not isinstance(transition, Mapping):
                raise DatabasePortalBridgeError(
                    "database effect rejected malformed source transition"
                )
            normalized = dict(transition)
            transition_cid = str(normalized.pop("transition_cid", "") or "")
            if (
                transition.get("schema")
                not in {
                    DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA,
                    DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA,
                    DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA,
                }
                or (
                    schema == DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2
                    and transition.get("schema")
                    == DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA
                )
                or (
                    schema == DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA
                    and transition.get("schema")
                    != DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA
                )
                or transition.get("database_task_cid") != str(attempt.task_cid)
                or transition.get("worker_self_approval") is not False
                or (
                    transition.get("schema")
                    == DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA
                    and transition.get("source_event_mode")
                    != "queued_merge_reconciliation"
                )
                or transition_cid
                != _sha256_bytes(_canonical_transition_json(normalized))
            ):
                raise DatabasePortalBridgeError(
                    "database effect rejected unbound source transition"
                )
        return digest

    def apply_effect(self, attempt: Any, provider_result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Bind the already-applied Portal effect to the database phase."""

        digest = self._require_accepted_provider(attempt, provider_result)
        result = {
            "status": "applied",
            "effect": "portal-supervised-accepted-effect",
            "effect_key": f"portal:{attempt.task_cid}:{attempt.attempt_id}",
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "portal_receipt_id": str(provider_result.get("receipt_id") or ""),
            "evidence_digest": digest,
        }
        if provider_result.get("accepted_source_transition") is not None:
            result["accepted_source_transition"] = dict(
                provider_result["accepted_source_transition"]
            )
        return result

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
        result = {
            "outcome": "passed",
            "evidence_digest": digest,
            "argv": ["portal-supervisor-gates"],
            "validator": self.INTERFACE,
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "portal_receipt_id": str(effect_result.get("portal_receipt_id") or ""),
        }
        if effect_result.get("accepted_source_transition") is not None:
            result["accepted_source_transition"] = dict(
                effect_result["accepted_source_transition"]
            )
        return result


__all__ = (
    "DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA",
    "DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE",
    "DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA",
    "DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1",
    "DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2",
    "DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA",
    "DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA",
    "DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA",
    "DatabasePortalAttemptPaths",
    "DatabasePortalBridgeDeferred",
    "DatabasePortalBridgeError",
    "DatabasePortalExecutionBridge",
    "PortalDaemonFactory",
)
