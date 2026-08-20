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
from pathlib import Path
from typing import Any, Final

DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE: Final[str] = "DatabasePortalExecutionBridge@1"
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@1"
)
DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@1"
)
DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-consumed-no-progress@1"
)
DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/portal-retry-deferral@1"
)
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset({"completed", "complete", "done"})
_MUTABLE_PROJECTION_LINE = re.compile(r"(?mi)^-\s*status\s*:\s*.*$")
_HEADER = re.compile(r"(?m)^##\s+([^\s]+)(?:\s+.*)?$")
_SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_DIAGNOSTIC_RECEIPT_BYTES: Final[int] = 256 * 1024
_MAX_CONTEXT_RECEIPT_BYTES: Final[int] = 256 * 1024
_MAX_FAILURE_LOG_BYTES: Final[int] = 128 * 1024
_MAX_CONSUMED_FAILURE_EVIDENCE_BYTES: Final[int] = 24 * 1024
_TASK_CONTRACT_MUTABLE_FIELDS: Final[frozenset[str]] = frozenset(
    {"completion_receipt", "status"}
)


class DatabasePortalBridgeError(RuntimeError):
    """A database claim could not obtain trustworthy Portal evidence."""


class DatabasePortalBridgeDeferred(DatabasePortalBridgeError):
    """Portal execution made bounded progress but is not yet acceptable."""


class DatabasePortalBridgeConsumedNoProgressError(DatabasePortalBridgeError):
    """One Portal attempt was consumed without an implementation candidate.

    The provider-effect state is deliberately unknown.  This exception seals
    only the durable no-progress outcome; it does not classify provider text
    or claim that a model call did or did not occur.
    """

    def __init__(
        self,
        message: str,
        *,
        failure_evidence: Mapping[str, Any],
    ) -> None:
        evidence = dict(failure_evidence)
        allowed = {
            "schema",
            "failure_kind",
            "failure_fingerprint",
            "diagnostic_failure_id",
            "diagnostic_receipt_id",
            "diagnostic_receipt_digest",
            "diagnostic_receipt_size",
            "context_receipt_id",
            "context_receipt_digest",
            "context_receipt_size",
            "log_digest",
            "log_size",
            "repository_id",
            "tree_id",
            "control_repository_tree_id",
            "task_cid",
            "task_contract_digest",
            "database_binding_id",
            "database_attempt_id",
            "database_claim_id",
            "database_lease_id",
            "database_fencing_token",
            "database_fence_epoch",
            "portal_task_id",
            "portal_attempt_number",
            "returncode",
            "attempt_consumed",
            "portal_provider_dispatched",
            "provider_effect_state",
            "implementation_commit_present",
            "implementation_candidate_present",
            "validation_state",
        }
        text_fields = (
            "failure_fingerprint",
            "diagnostic_failure_id",
            "diagnostic_receipt_id",
            "diagnostic_receipt_digest",
            "context_receipt_id",
            "context_receipt_digest",
            "log_digest",
            "repository_id",
            "tree_id",
            "control_repository_tree_id",
            "task_cid",
            "task_contract_digest",
            "database_binding_id",
            "database_attempt_id",
            "database_claim_id",
            "database_lease_id",
            "portal_task_id",
            "provider_effect_state",
            "validation_state",
        )
        if (
            set(evidence) != allowed
            or evidence.get("schema")
            != DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA
            or evidence.get("failure_kind") != "consumed_no_progress"
            or evidence.get("provider_effect_state")
            != "unknown_may_have_started"
            or evidence.get("attempt_consumed") is not True
            or type(evidence.get("portal_provider_dispatched")) is not bool
            or evidence.get("implementation_commit_present") is not False
            or evidence.get("implementation_candidate_present") is not False
            or evidence.get("validation_state") != "not_run"
            or isinstance(evidence.get("portal_attempt_number"), bool)
            or not isinstance(evidence.get("portal_attempt_number"), int)
            or int(evidence.get("portal_attempt_number") or 0) < 1
            or type(evidence.get("returncode")) is not int
            or evidence.get("returncode") == 0
            or not -(2**31) <= evidence.get("returncode") < 2**31
            or type(evidence.get("database_fencing_token")) is not int
            or evidence.get("database_fencing_token") < 1
            or type(evidence.get("database_fence_epoch")) is not int
            or evidence.get("database_fence_epoch") < 1
            or type(evidence.get("diagnostic_receipt_size")) is not int
            or not 1
            <= evidence.get("diagnostic_receipt_size")
            <= _MAX_DIAGNOSTIC_RECEIPT_BYTES
            or type(evidence.get("context_receipt_size")) is not int
            or not 1
            <= evidence.get("context_receipt_size")
            <= _MAX_CONTEXT_RECEIPT_BYTES
            or type(evidence.get("log_size")) is not int
            or not 0 <= evidence.get("log_size") <= _MAX_FAILURE_LOG_BYTES
            or any(
                not isinstance(evidence.get(key), str)
                or not str(evidence.get(key) or "")
                or len(str(evidence.get(key)).encode("utf-8")) > 1024
                or "\x00" in str(evidence.get(key))
                or "\n" in str(evidence.get(key))
                or "\r" in str(evidence.get(key))
                for key in text_fields
            )
            or not _SHA256_ID.fullmatch(
                str(evidence.get("failure_fingerprint") or "")
            )
            or not _SHA256_ID.fullmatch(
                str(evidence.get("diagnostic_receipt_digest") or "")
            )
            or not _SHA256_ID.fullmatch(
                str(evidence.get("context_receipt_digest") or "")
            )
            or not _SHA256_ID.fullmatch(str(evidence.get("log_digest") or ""))
            or not _SHA256_ID.fullmatch(
                str(evidence.get("task_contract_digest") or "")
            )
            or not _SHA256_ID.fullmatch(
                str(evidence.get("database_binding_id") or "")
            )
            or evidence.get("failure_fingerprint")
            != database_portal_consumed_no_progress_fingerprint(evidence)
            or len(_canonical_json(evidence))
            > _MAX_CONSUMED_FAILURE_EVIDENCE_BYTES
        ):
            raise ValueError("Portal consumed-no-progress evidence is invalid")
        super().__init__(message)
        self.failure_evidence = evidence


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


def database_portal_task_contract_digest(record: Any) -> str:
    """Commit to the complete status-independent Portal task contract.

    ``TaskRecord.body`` is only one part of the execution contract.  The
    Portal projection also consumes the task's graph identity, ordering,
    declared outputs, acceptance policy, and validation commands.  Keep the
    mutable lifecycle fields (status, revision, and completion receipts) out
    of this digest so the same contract remains verifiable after quarantine.
    """

    def value(name: str, default: Any = None) -> Any:
        if isinstance(record, Mapping):
            return record.get(name, default)
        return getattr(record, name, default)

    raw_body = value("body", {})
    body = dict(raw_body) if isinstance(raw_body, Mapping) else {}
    contract_body = {
        str(key): item
        for key, item in body.items()
        if str(key) not in _TASK_CONTRACT_MUTABLE_FIELDS
    }

    raw_dependencies = value("dependencies", ()) or ()
    dependencies = [str(item) for item in raw_dependencies]

    def mappings(name: str) -> list[dict[str, Any]]:
        raw_items = value(name, ()) or ()
        return [dict(item) for item in raw_items if isinstance(item, Mapping)]

    contract = {
        "task_cid": str(value("task_cid", "") or ""),
        "task_alias": str(value("task_alias", "") or ""),
        "goal_cid": str(value("goal_cid", "") or ""),
        "plan_cid": str(value("plan_cid", "") or ""),
        "objective_id": str(value("objective_id", "") or ""),
        "priority": str(value("priority", "") or ""),
        "ordinal": int(value("ordinal", 0) or 0),
        "dependencies": dependencies,
        "outputs": mappings("outputs"),
        "acceptance": mappings("acceptance"),
        "validations": mappings("validations"),
        "body": contract_body,
    }
    if not contract["task_cid"]:
        raise DatabasePortalBridgeError("task contract has no canonical CID")
    return _sha256_bytes(_canonical_json(contract))


def database_portal_authoritative_repository_tree_id(
    task_source: Any,
    task_cid: str,
) -> str:
    """Resolve the persisted task tree, rejecting a divergent live view.

    ``DatabaseTaskSource.repository_tree_id`` is populated while materializing
    but is not itself persisted by that adapter.  The exact tree is persisted
    in the task identity, so cold-restart validation must prefer that identity
    while still rejecting a conflicting non-empty snapshot value.
    """

    snapshot = task_source.snapshot()
    snapshot_tree = str(
        getattr(snapshot, "repository_tree_id", "")
        or (
            snapshot.get("repository_tree_id", "")
            if isinstance(snapshot, Mapping)
            else ""
        )
    ).strip()
    identity_tree = ""
    intent = getattr(task_source, "intent", None)
    get_task = getattr(intent, "get_task", None)
    if callable(get_task):
        persisted = get_task(str(task_cid))
        identity = (
            persisted.get("identity")
            if isinstance(persisted, Mapping)
            and isinstance(persisted.get("identity"), Mapping)
            else {}
        )
        identity_tree = str(identity.get("repository_tree_id") or "").strip()
    if identity_tree and snapshot_tree and identity_tree != snapshot_tree:
        raise DatabasePortalBridgeError(
            "database task repository tree conflicts with persisted identity"
        )
    repository_tree_id = identity_tree or snapshot_tree
    if not repository_tree_id:
        raise DatabasePortalBridgeError(
            "database task source has no authoritative repository tree"
        )
    return repository_tree_id


def database_portal_consumed_no_progress_fingerprint(
    evidence: Mapping[str, Any],
) -> str:
    """Return the neutral circuit-breaker key for one sealed outcome."""

    material = {
        "schema": DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA,
        "failure_kind": str(evidence.get("failure_kind") or ""),
        "repository_id": str(evidence.get("repository_id") or ""),
        "tree_id": str(evidence.get("tree_id") or ""),
        "control_repository_tree_id": str(
            evidence.get("control_repository_tree_id") or ""
        ),
        "task_cid": str(evidence.get("task_cid") or ""),
        "task_contract_digest": str(
            evidence.get("task_contract_digest") or ""
        ),
        "diagnostic_failure_id": str(
            evidence.get("diagnostic_failure_id") or ""
        ),
        "diagnostic_receipt_id": str(
            evidence.get("diagnostic_receipt_id") or ""
        ),
        "context_receipt_id": str(evidence.get("context_receipt_id") or ""),
        "log_digest": str(evidence.get("log_digest") or ""),
        "returncode": evidence.get("returncode"),
        "provider_effect_state": str(
            evidence.get("provider_effect_state") or ""
        ),
    }
    return _sha256_bytes(_canonical_json(material))


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise DatabasePortalBridgeError(
            f"could not read Portal attempt artifact {path.name!r}"
        ) from exc


def _bounded_file(path: Path, *, limit: int) -> bytes:
    """Read one bounded regular artifact without accepting truncation."""

    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("artifact is not a regular non-symlink file")
        size = path.stat().st_size
        if size > limit:
            raise OSError("artifact exceeds its byte limit")
        with path.open("rb") as handle:
            payload = handle.read(limit + 1)
        if len(payload) != size:
            raise OSError("artifact changed while read")
        return payload
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
    return _line_value(
        value.get("path")
        or value.get("output")
        or value.get("artifact_id")
        or value.get("fluent_id")
        or value
    )


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


def _projection_task_identity(
    record: Any,
    body: Mapping[str, Any],
) -> tuple[str, str]:
    """Return the exact database identity admitted into a Portal projection.

    Portal's Markdown parser recognizes a canonical identity only when both
    the key and CID are present.  A database task CID rendered merely as
    descriptive metadata would otherwise be re-derived from the disposable
    projection and create a second identity for the same claimed task.
    """

    task_cid = str(getattr(record, "task_cid", "") or "").strip()
    if not task_cid or len(task_cid) > 1024 or any(character.isspace() for character in task_cid):
        raise DatabasePortalBridgeError("database task CID is not projection-safe")

    def claimed_values(*names: str) -> set[str]:
        selected = set(names)
        return {
            str(value).strip()
            for key, value in body.items()
            if str(key).strip().lower().replace("_", " ") in selected and value not in (None, "")
        }

    claimed_cids = claimed_values("task cid", "canonical task cid")
    if any(value != task_cid for value in claimed_cids):
        raise DatabasePortalBridgeError("database task body contradicts its authoritative task CID")

    claimed_keys = claimed_values("task key", "canonical task key")
    if len(claimed_keys) > 1:
        raise DatabasePortalBridgeError(
            "database task body contains contradictory canonical task keys"
        )
    task_key = next(iter(claimed_keys), task_cid)
    if not task_key or len(task_key) > 1024 or any(character.isspace() for character in task_key):
        raise DatabasePortalBridgeError("database task key is not projection-safe")
    return task_key, task_cid


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
        self.task_header_prefix = str(task_header_prefix or "## ")
        self.max_passes = max_passes

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
        canonical_task_key, canonical_task_cid = _projection_task_identity(
            record,
            body,
        )
        repository_tree_id = database_portal_authoritative_repository_tree_id(
            self.task_source,
            canonical_task_cid,
        )
        payload = {
            "schema": DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
            "interface": self.INTERFACE,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": canonical_task_cid,
            "canonical_task_key": canonical_task_key,
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
            "task_contract_digest": database_portal_task_contract_digest(record),
            "repository_tree_id": repository_tree_id,
            "projection_seed_digest": _sha256_bytes(seed.encode("utf-8")),
            "projection_immutable_digest": _projection_immutable_digest(seed),
            "authoritative_task_store": "duckdb",
            "projection_authority": False,
        }
        payload["binding_id"] = _sha256_bytes(_canonical_json(payload))
        return payload

    def _render_projection(self, attempt: Any, record: Any) -> str:
        body = dict(getattr(record, "body", {}) or {})
        canonical_task_key, canonical_task_cid = _projection_task_identity(
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
        outputs = _output_values(record, body)
        validations = _validation_values(record, body)
        acceptance = _acceptance_value(record, body)
        priority = _line_value(getattr(record, "priority", "") or body.get("priority") or "P2")
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
            "task key",
            "task cid",
            "canonical task key",
            "canonical task cid",
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
            f"- Canonical Task Key: {canonical_task_key}",
            f"- Canonical Task CID: {canonical_task_cid}",
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
    def _has_completion_event(
        paths: DatabasePortalAttemptPaths,
        alias: str,
        canonical_task_key: str,
        canonical_task_cid: str,
    ) -> bool:
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
                and str(event.get("canonical_task_key") or "") == canonical_task_key
                and str(event.get("canonical_task_cid") or "") == canonical_task_cid
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
    def _explicit_retryable_deferral(
        implementation: Mapping[str, Any],
    ) -> bool:
        """Admit only a closed, structured non-consuming deferral."""

        if (
            implementation.get("deferral_schema")
            != DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA
            or implementation.get("deferred") is not True
            or implementation.get("retryable") is not True
            or implementation.get("attempt_consumed") is not False
        ):
            return False
        kind = str(implementation.get("failure_kind") or "")
        provider_dispatched = implementation.get("provider_dispatched")
        if kind in {"lifecycle_setup", "lifecycle_race"}:
            return (
                provider_dispatched is False
                and implementation.get("provider_call_allowed")
                in (None, False)
            )
        if kind == "provider_capacity_backoff":
            retry_at = str(implementation.get("retry_at") or "")
            retry_after = implementation.get("retry_after_seconds")
            return bool(
                provider_dispatched is False
                and retry_at
                and type(retry_after) in {int, float}
                and not isinstance(retry_after, bool)
                and retry_after >= 0
            )
        if kind != "provider_capacity":
            return False
        returncode = implementation.get("returncode")
        retry_at = str(implementation.get("retry_at") or "")
        failure_class = str(implementation.get("failure_class") or "")
        providers = implementation.get("providers")
        return bool(
            type(provider_dispatched) is bool
            and type(returncode) is int
            and returncode != 0
            and retry_at
            and failure_class in {"transient_capacity", "hard_quota_exhausted"}
            and isinstance(providers, Sequence)
            and not isinstance(providers, (str, bytes, bytearray, memoryview))
            and any(str(item or "").strip() for item in providers)
        )

    @staticmethod
    def _consumed_no_progress_failure(
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        implementation: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Seal one consumed, validation-not-run, no-candidate outcome.

        Raw runner/provider text is never interpreted as a root cause.  The
        canonical context and diagnostic receipts establish the repository,
        tree, task and no-progress boundary; provider-effect state remains
        explicitly unknown.
        """

        returncode = implementation.get("returncode")
        validation = implementation.get("validation_result")
        commit_result = implementation.get("commit_result")
        merge_result = implementation.get("merge_result")
        board_completion = implementation.get("board_completion")
        if (
            type(returncode) is not int
            or returncode == 0
            or not -(2**31) <= returncode < 2**31
            or type(implementation.get("provider_dispatched")) is not bool
            or implementation.get("attempt_consumed") is not True
            or str(implementation.get("implementation_commit") or "")
            or str(implementation.get("completion_tree_id") or "")
            or not isinstance(commit_result, Mapping)
            or commit_result.get("committed") is not False
            or not isinstance(merge_result, Mapping)
            or merge_result.get("merged") is not False
            or not isinstance(board_completion, Mapping)
            or board_completion.get("complete") is not False
            or board_completion.get("pending_merge") is not False
            or not isinstance(validation, Mapping)
            or validation.get("attempted") is not False
            or validation.get("passed") is not True
            or str(validation.get("reason") or "") != "not_run"
            or type(validation.get("returncode")) is not int
            or validation.get("returncode") != 0
            or not isinstance(validation.get("results"), Sequence)
            or isinstance(
                validation.get("results"),
                (str, bytes, bytearray, memoryview),
            )
            or len(validation.get("results")) != 0
        ):
            return None

        task_id = str(implementation.get("task_id") or "").strip()
        canonical_task_cid = str(
            implementation.get("canonical_task_cid")
            or implementation.get("task_cid")
            or ""
        ).strip()
        portal_attempt = implementation.get("attempt")
        log_value = str(implementation.get("log_path") or "").strip()
        context_value = str(
            implementation.get("context_receipt_path") or ""
        ).strip()
        diagnostic_id = str(
            implementation.get("diagnostic_receipt_id") or ""
        ).strip()
        baseline = str(implementation.get("baseline_ref") or "").strip()
        if (
            task_id != str(binding.get("task_alias") or "")
            or canonical_task_cid != str(binding.get("task_cid") or "")
            or type(portal_attempt) is not int
            or portal_attempt < 1
            or not log_value
            or not context_value
            or not diagnostic_id
            or not baseline
        ):
            return None
        try:
            log_path = Path(log_value).expanduser().resolve(strict=True)
            context_path = Path(context_value).expanduser().resolve(strict=True)
            log_root = paths.implementation_logs.resolve(strict=True)
        except OSError:
            return None
        if log_root not in log_path.parents or log_root not in context_path.parents:
            return None
        try:
            raw_log = _bounded_file(
                log_path,
                limit=_MAX_FAILURE_LOG_BYTES,
            )
            raw_context = _bounded_file(
                context_path,
                limit=_MAX_CONTEXT_RECEIPT_BYTES,
            )
        except DatabasePortalBridgeError:
            return None

        try:
            diagnostic_paths = tuple(
                paths.implementation_logs.glob("*-diagnostic-receipt.json")
            )
        except OSError:
            return None
        if len(diagnostic_paths) != 1:
            return None
        diagnostic_path = diagnostic_paths[0]
        try:
            raw_diagnostic = _bounded_file(
                diagnostic_path,
                limit=_MAX_DIAGNOSTIC_RECEIPT_BYTES,
            )
            def reject_duplicate_keys(
                pairs: Sequence[tuple[str, Any]],
            ) -> dict[str, Any]:
                parsed: dict[str, Any] = {}
                for key, value in pairs:
                    if key in parsed:
                        raise ValueError(
                            "implementation diagnostic receipt has duplicate keys"
                        )
                    parsed[key] = value
                return parsed

            diagnostic_payload = json.loads(
                raw_diagnostic.decode("utf-8"),
                object_pairs_hook=reject_duplicate_keys,
            )
            if not isinstance(diagnostic_payload, Mapping):
                return None
            from .implementation_daemon import ImplementationDiagnosticReceipt

            diagnostic = ImplementationDiagnosticReceipt.from_dict(
                diagnostic_payload
            )
            context_payload = json.loads(
                raw_context.decode("utf-8"),
                object_pairs_hook=reject_duplicate_keys,
            )
            if not isinstance(context_payload, Mapping):
                return None
            from ..context.context_compiler import ContextCompilationReceipt

            context = ContextCompilationReceipt.from_dict(context_payload)
        except (DatabasePortalBridgeError, OSError, UnicodeDecodeError, ValueError):
            return None
        payload_receipt_id = diagnostic_payload.get("receipt_id")
        payload_failure_id = diagnostic_payload.get("failure_id")
        if (
            not isinstance(payload_receipt_id, str)
            or payload_receipt_id != diagnostic.receipt_id
            or payload_receipt_id != diagnostic_id
            or not isinstance(payload_failure_id, str)
            or payload_failure_id != diagnostic.failure_id
            or diagnostic.prior_decision_id != context.receipt_id
            or diagnostic.repository_id != context.repository_id
            or diagnostic.tree_id != context.tree_id
            or diagnostic.tree_id != baseline
            or context.objective_id != task_id
            or context.stage != "implementation"
            or isinstance(diagnostic.failure.get("returncode"), bool)
            or not isinstance(diagnostic.failure.get("returncode"), int)
            or diagnostic.failure.get("returncode") != returncode
            or diagnostic.changed_files
        ):
            return None
        projected_validation = {
            key: validation[key]
            for key in (
                "passed",
                "returncode",
                "reason",
                "reason_codes",
                "failed_commands",
                "failure_review",
            )
            if validation.get(key) not in (None, "", (), [], {})
        }
        if (
            diagnostic.failure.get("kind") != "implementation_failure"
            or diagnostic.failure.get("validation") != projected_validation
        ):
            return None

        signature_material = {
            "schema": DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA,
            "failure_kind": "consumed_no_progress",
            "repository_id": diagnostic.repository_id,
            "tree_id": baseline,
            "control_repository_tree_id": str(
                binding.get("repository_tree_id") or ""
            ),
            "task_cid": canonical_task_cid,
            "task_contract_digest": str(
                binding.get("task_contract_digest") or ""
            ),
            "diagnostic_failure_id": diagnostic.failure_id,
            "provider_effect_state": "unknown_may_have_started",
        }
        evidence = {
            **signature_material,
            "diagnostic_receipt_id": diagnostic_id,
            "diagnostic_receipt_digest": _sha256_bytes(raw_diagnostic),
            "diagnostic_receipt_size": len(raw_diagnostic),
            "context_receipt_id": context.receipt_id,
            "context_receipt_digest": _sha256_bytes(raw_context),
            "context_receipt_size": len(raw_context),
            "log_digest": _sha256_bytes(raw_log),
            "log_size": len(raw_log),
            "database_binding_id": str(binding.get("binding_id") or ""),
            "database_attempt_id": str(binding.get("attempt_id") or ""),
            "database_claim_id": str(binding.get("claim_id") or ""),
            "database_lease_id": str(binding.get("lease_id") or ""),
            "database_fencing_token": int(binding.get("fencing_token") or 0),
            "database_fence_epoch": int(binding.get("fence_epoch") or 0),
            "portal_task_id": task_id,
            "portal_attempt_number": portal_attempt,
            "returncode": returncode,
            "attempt_consumed": True,
            "portal_provider_dispatched": implementation.get(
                "provider_dispatched"
            ),
            "implementation_commit_present": False,
            "implementation_candidate_present": False,
            "validation_state": "not_run",
        }
        evidence["failure_fingerprint"] = (
            database_portal_consumed_no_progress_fingerprint(evidence)
        )
        try:
            return DatabasePortalBridgeConsumedNoProgressError(
                "portal_consumed_no_progress",
                failure_evidence=evidence,
            ).failure_evidence
        except ValueError:
            return None

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
        if not self._has_completion_event(
            paths,
            alias,
            str(binding.get("canonical_task_key") or ""),
            str(binding.get("task_cid") or ""),
        ):
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
                    paths,
                    str(binding.get("task_alias") or ""),
                    str(binding.get("canonical_task_key") or ""),
                    str(binding.get("task_cid") or ""),
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
                failure = self._terminal_failure(raw_result)
                if failure:
                    implementation = raw_result.get("implementation_result")
                    if isinstance(
                        implementation, Mapping
                    ) and self._explicit_retryable_deferral(implementation):
                        raise DatabasePortalBridgeDeferred(failure)
                    if isinstance(implementation, Mapping):
                        consumed_no_progress = (
                            self._consumed_no_progress_failure(
                                paths,
                                binding,
                                implementation,
                            )
                        )
                        if consumed_no_progress is not None:
                            raise DatabasePortalBridgeConsumedNoProgressError(
                                "portal_consumed_no_progress",
                                failure_evidence=consumed_no_progress,
                            )
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
    "DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA",
    "DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE",
    "DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA",
    "DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA",
    "DatabasePortalAttemptPaths",
    "DatabasePortalBridgeDeferred",
    "DatabasePortalBridgeConsumedNoProgressError",
    "DatabasePortalBridgeError",
    "DatabasePortalExecutionBridge",
    "PortalDaemonFactory",
    "database_portal_authoritative_repository_tree_id",
    "database_portal_consumed_no_progress_fingerprint",
    "database_portal_task_contract_digest",
)
