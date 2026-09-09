"""Authenticated atomic mutation bundles for the exclusive Quack state owner.

The Quack read transport is deliberately non-authoritative. A client buffers
only the closed SQL templates in this module, then submits the complete
transaction to the exclusive DuckDB writer through a local owner-only inbox.
The writer authenticates the exact store generation, validates the domain
transition, applies every step atomically, refreshes the read-only replica,
and returns a signed terminal receipt. Importing this module performs no I/O.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import json
import os
import re
import stat
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .control_plane_contracts import canonical_json_bytes, content_identity

QUACK_OWNER_MUTATION_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-mutation-request@2"
)
QUACK_OWNER_MUTATION_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-mutation-result@2"
)
QUACK_OWNER_MUTATION_SEMANTIC_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-mutation-semantic@1"
)
QUACK_OWNER_MUTATION_PROTOCOL_REVISION: Final[int] = 2
QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES: Final[int] = 1_048_576
QUACK_OWNER_MUTATION_MAX_PARAMETER_BYTES: Final[int] = 262_144
QUACK_OWNER_MUTATION_MAX_PARAMETERS: Final[int] = 17
QUACK_OWNER_MUTATION_MAX_STEPS: Final[int] = 5
QUACK_OWNER_MUTATION_REQUEST_TTL_MS: Final[int] = 30_000
QUACK_OWNER_MUTATION_SETTLEMENT_MS: Final[int] = 30_000
QUACK_OWNER_MUTATION_MAX_CLOCK_SKEW_MS: Final[int] = 5_000
MUTATION_MAX_DIRECTORY_ENTRIES: Final[int] = 4_096
MUTATION_MAX_PER_PASS: Final[int] = 32

QUACK_MUTATION_TASK_STATUS_CAS: Final = "task_status_cas@1"
QUACK_MUTATION_TASK_REVISION_INSERT: Final = "task_revision_insert@1"
QUACK_MUTATION_COMPLETION_RECEIPT_INSERT: Final = "completion_receipt_insert@1"
QUACK_MUTATION_DOMAIN_EVENT_INSERT: Final = "domain_event_insert@1"
QUACK_MUTATION_VALIDATION_RUN_INSERT: Final = "validation_run_insert@1"
QUACK_MUTATION_VALIDATION_RESULT_INSERT: Final = "validation_result_insert@1"
QUACK_MUTATION_EVIDENCE_DELETE: Final = "evidence_delete@1"
QUACK_MUTATION_EVIDENCE_INSERT: Final = "evidence_insert@1"
QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT: Final = "lease_queue_backoff_insert@1"
QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE: Final = "lease_queue_backoff_update@1"

QUACK_MUTATION_TASK_STATUS_TRANSITION: Final = "task_status_transition@1"
QUACK_MUTATION_VALIDATION_RECORD: Final = "validation_record@1"
QUACK_MUTATION_EVIDENCE_RECORD: Final = "evidence_record@1"
QUACK_MUTATION_QUEUE_BACKOFF: Final = "queue_backoff@1"

_TOKEN_RE: Final = re.compile(r"^[A-Za-z0-9_-]{8,}$")
_REQUEST_NAME_RE: Final = re.compile(
    r"^(?P<request_id>b[a-z2-7]{40,127})\.request\.json$"
)
_PROCESSING_NAME_RE: Final = re.compile(
    r"^(?P<request_id>b[a-z2-7]{40,127})\.processing\.json$"
)
_DIGEST_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_FIELDS: Final = frozenset(
    {
        "schema", "protocol_revision", "request_id", "issued_at_ms",
        "expires_at_ms", "operation", "binding", "steps", "request_cid",
        "auth_mac",
    }
)
_RESULT_FIELDS: Final = frozenset(
    {
        "schema", "protocol_revision", "request_id", "request_cid",
        "issued_at_ms", "expires_at_ms", "operation", "binding", "ok",
        "error_code", "rowcounts", "observed", "result_cid", "result_mac",
    }
)
_BINDING_FIELDS: Final = frozenset(
    {
        "server_id", "store_id", "database_uuid", "schema_revision",
        "schema_fingerprint", "generation", "process_birth_id", "listen_uri",
        "extension_fingerprint",
    }
)


class QuackOwnerMutationError(RuntimeError):
    """A mutation request, effect, or signed result was not admissible."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = str(code or "mutation_error")
        super().__init__(message or self.code)


class QuackOwnerMutationConflictError(QuackOwnerMutationError):
    """The owner observed a stale compare-and-set or event head."""


class QuackOwnerMutationUnknownOutcomeError(QuackOwnerMutationError):
    """The caller cannot prove whether an external effect settled."""


def normalize_mutation_sql(sql: str) -> str:
    normalized = " ".join(str(sql).strip().upper().split())
    normalized = re.sub(r"\(\s+", "(", normalized)
    return re.sub(r"\s+\)", ")", normalized)


MUTATION_SQL_TEMPLATES: Final[Mapping[str, str]] = MappingProxyType(
    {
        QUACK_MUTATION_TASK_STATUS_CAS: (
            "UPDATE tasks SET status = ?, revision = ?, updated_at = ?, "
            "body_json = ? WHERE task_cid = ? AND revision = ?"
        ),
        QUACK_MUTATION_TASK_REVISION_INSERT: (
            "INSERT INTO task_revisions (task_cid, revision, status, body_json, "
            "recorded_at) VALUES (?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_COMPLETION_RECEIPT_INSERT: (
            "INSERT INTO completion_receipts (receipt_cid, task_cid, goal_cid, "
            "attempt_id, claim_cid, fencing_token, completed_at, "
            "validation_run_id, evidence_digest, body_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_DOMAIN_EVENT_INSERT: (
            "INSERT INTO domain_events (event_id, stream_id, sequence, "
            "global_sequence, event_type, task_cid, attempt_id, session_id, "
            "recorded_at, body_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_VALIDATION_RUN_INSERT: (
            "INSERT INTO validation_runs (run_id, task_cid, attempt_id, "
            "started_at, finished_at, status, command_digest, body_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_VALIDATION_RESULT_INSERT: (
            "INSERT INTO validation_results (result_id, run_id, task_cid, "
            "ordinal, outcome, evidence_digest, body_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_EVIDENCE_DELETE: (
            "DELETE FROM evidence_nodes WHERE evidence_id = ?"
        ),
        QUACK_MUTATION_EVIDENCE_INSERT: (
            "INSERT INTO evidence_nodes (evidence_id, parent_evidence_id, "
            "task_cid, evidence_kind, digest, created_at, body_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT: (
            "INSERT INTO leases (task_cid, claim_cid, resolution_cid, "
            "claimant_did, logical_epoch, fencing_token, expires_at_ms, "
            "attempt, state, started_at_ms, release_reason, retry_not_before_ms, "
            "owner_session_id, fence_epoch, revision, extension_schema, "
            "extension_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE: (
            "UPDATE leases SET attempt = ?, retry_not_before_ms = ?, "
            "release_reason = ?, state = 'released', extension_schema = ?, "
            "extension_json = ?, revision = revision + 1 WHERE task_cid = ?"
        ),
    }
)
MUTATION_SQL_TO_TEMPLATE: Final[Mapping[str, str]] = MappingProxyType(
    {
        normalize_mutation_sql(sql): template_id
        for template_id, sql in MUTATION_SQL_TEMPLATES.items()
    }
)

_TASK_SHAPES: Final = frozenset(
    {
        (QUACK_MUTATION_TASK_STATUS_CAS, QUACK_MUTATION_TASK_REVISION_INSERT,
         QUACK_MUTATION_DOMAIN_EVENT_INSERT),
        (QUACK_MUTATION_TASK_STATUS_CAS, QUACK_MUTATION_TASK_REVISION_INSERT,
         QUACK_MUTATION_COMPLETION_RECEIPT_INSERT,
         QUACK_MUTATION_DOMAIN_EVENT_INSERT),
    }
)
_VALIDATION_SHAPES: Final = frozenset(
    {
        (QUACK_MUTATION_VALIDATION_RUN_INSERT,
         QUACK_MUTATION_VALIDATION_RESULT_INSERT,
         QUACK_MUTATION_DOMAIN_EVENT_INSERT),
        (QUACK_MUTATION_VALIDATION_RUN_INSERT,
         QUACK_MUTATION_VALIDATION_RESULT_INSERT,
         QUACK_MUTATION_EVIDENCE_DELETE, QUACK_MUTATION_EVIDENCE_INSERT,
         QUACK_MUTATION_DOMAIN_EVENT_INSERT),
    }
)
_EVIDENCE_SHAPES: Final = frozenset(
    {(QUACK_MUTATION_EVIDENCE_DELETE, QUACK_MUTATION_EVIDENCE_INSERT,
      QUACK_MUTATION_DOMAIN_EVENT_INSERT)}
)
_QUEUE_SHAPES: Final = frozenset(
    {
        (QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT,
         QUACK_MUTATION_DOMAIN_EVENT_INSERT),
        (QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE,
         QUACK_MUTATION_DOMAIN_EVENT_INSERT),
    }
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return canonical_json_bytes(dict(value))
    except Exception as exc:
        raise QuackOwnerMutationError(
            "canonical_json_invalid", "mutation payload is not canonical JSON"
        ) from exc


def _token(value: str) -> str:
    secret = str(value or "").strip()
    if not _TOKEN_RE.fullmatch(secret):
        raise QuackOwnerMutationError(
            "authentication_unavailable", "mutation token is unavailable"
        )
    return secret


def mutation_content_id(value: Mapping[str, Any]) -> str:
    try:
        return content_identity(dict(value))
    except Exception as exc:
        raise QuackOwnerMutationError("identity_invalid") from exc


def mutation_mac(value: Mapping[str, Any], token: str) -> str:
    return hmac.new(
        _token(token).encode("ascii"), _canonical_bytes(value), hashlib.sha256
    ).hexdigest()


def validate_mutation_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BINDING_FIELDS:
        raise QuackOwnerMutationError("binding_invalid")
    result = {str(key): member for key, member in value.items()}
    for name in _BINDING_FIELDS - {"schema_revision", "generation"}:
        member = result[name]
        if (
            not isinstance(member, str)
            or not member
            or "\x00" in member
            or len(member.encode("utf-8")) > 1024
        ):
            raise QuackOwnerMutationError("binding_invalid")
    for name in ("schema_revision", "generation"):
        if type(result[name]) is not int or result[name] < 1:
            raise QuackOwnerMutationError("binding_invalid")
    return result


def mutation_binding_from_identity(identity: Any) -> dict[str, Any]:
    return validate_mutation_binding(
        {
            "server_id": str(identity.server_id),
            "store_id": str(identity.store_id),
            "database_uuid": str(identity.database_uuid),
            "schema_revision": int(identity.schema_revision),
            "schema_fingerprint": str(identity.schema_fingerprint),
            "generation": int(identity.generation),
            "process_birth_id": str(identity.process_birth_id),
            "listen_uri": str(identity.listen_uri),
            "extension_fingerprint": str(identity.extension_fingerprint or "none"),
        }
    )


def validate_mutation_parameters(parameters: Sequence[Any]) -> list[Any]:
    if not isinstance(parameters, (list, tuple)):
        raise QuackOwnerMutationError("parameters_invalid")
    if len(parameters) > QUACK_OWNER_MUTATION_MAX_PARAMETERS:
        raise QuackOwnerMutationError("parameters_invalid")
    result: list[Any] = []
    for value in parameters:
        if value is None:
            result.append(None)
        elif type(value) is int and -(2**63) <= value < 2**63:
            result.append(value)
        elif (
            isinstance(value, str)
            and len(value.encode("utf-8")) <= QUACK_OWNER_MUTATION_MAX_PARAMETER_BYTES
        ):
            result.append(value)
        else:
            raise QuackOwnerMutationError("parameters_invalid")
    return result


def mutation_step(sql: str, parameters: Sequence[Any]) -> dict[str, Any]:
    template_id = MUTATION_SQL_TO_TEMPLATE.get(normalize_mutation_sql(sql))
    if template_id is None:
        raise QuackOwnerMutationError("template_not_allowlisted")
    return {
        "template_id": template_id,
        "parameters": validate_mutation_parameters(parameters),
    }


def mutation_operation(steps: Sequence[Mapping[str, Any]]) -> str:
    if (
        not isinstance(steps, Sequence)
        or not 1 <= len(steps) <= QUACK_OWNER_MUTATION_MAX_STEPS
    ):
        raise QuackOwnerMutationError("operation_shape_invalid")
    # A closed read request uses the same authenticated owner transport. It is
    # never admitted into a SQL mutation bundle and carries a fresh challenge.
    if (
        len(steps) == 1
        and isinstance(steps[0], Mapping)
        and steps[0].get("template_id") == "owner_snapshot@1"
    ):
        step = steps[0]
        parameters = step.get("parameters")
        if (
            set(step) != {"template_id", "parameters"}
            or not isinstance(parameters, list)
            or len(parameters) != 1
            or not isinstance(parameters[0], str)
            or re.fullmatch(r"[0-9a-f]{32}", parameters[0]) is None
        ):
            raise QuackOwnerMutationError("operation_shape_invalid")
        return "owner_snapshot@1"
    templates: list[str] = []
    for step in steps:
        if not isinstance(step, Mapping) or set(step) != {"template_id", "parameters"}:
            raise QuackOwnerMutationError("operation_shape_invalid")
        template_id = str(step.get("template_id") or "")
        if template_id not in MUTATION_SQL_TEMPLATES:
            raise QuackOwnerMutationError("template_not_allowlisted")
        validate_mutation_parameters(step.get("parameters"))
        templates.append(template_id)
    shape = tuple(templates)
    if shape in _TASK_SHAPES:
        return QUACK_MUTATION_TASK_STATUS_TRANSITION
    if shape in _VALIDATION_SHAPES:
        return QUACK_MUTATION_VALIDATION_RECORD
    if shape in _EVIDENCE_SHAPES:
        return QUACK_MUTATION_EVIDENCE_RECORD
    if shape in _QUEUE_SHAPES:
        return QUACK_MUTATION_QUEUE_BACKOFF
    raise QuackOwnerMutationError("operation_shape_invalid")


def build_mutation_request(
    *,
    steps: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
    token: str,
    issued_at_ms: int | None = None,
) -> dict[str, Any]:
    normalized_steps = [
        {
            "template_id": str(step["template_id"]),
            "parameters": validate_mutation_parameters(step["parameters"]),
        }
        for step in steps
    ]
    operation = mutation_operation(normalized_steps)
    admitted_binding = validate_mutation_binding(binding)
    semantic = {
        "schema": QUACK_OWNER_MUTATION_SEMANTIC_SCHEMA,
        "protocol_revision": QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
        "operation": operation,
        "binding": admitted_binding,
        "steps": normalized_steps,
    }
    request_id = mutation_content_id(semantic)
    issued = int(time.time() * 1000) if issued_at_ms is None else issued_at_ms
    if type(issued) is not int or issued < 1:
        raise QuackOwnerMutationError("request_time_invalid")
    unsigned = {
        "schema": QUACK_OWNER_MUTATION_REQUEST_SCHEMA,
        "protocol_revision": QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
        "request_id": request_id,
        "issued_at_ms": issued,
        "expires_at_ms": issued + QUACK_OWNER_MUTATION_REQUEST_TTL_MS,
        "operation": operation,
        "binding": admitted_binding,
        "steps": normalized_steps,
    }
    request_cid = mutation_content_id(unsigned)
    authenticated = {**unsigned, "request_cid": request_cid}
    return {**authenticated, "auth_mac": mutation_mac(authenticated, token)}


def validate_mutation_request(
    payload: Mapping[str, Any],
    *,
    request_id: str,
    binding: Mapping[str, Any],
    token: str,
    now_ms: int | None = None,
    allow_expired: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _REQUEST_FIELDS:
        raise QuackOwnerMutationError("request_schema_invalid")
    if (
        payload.get("schema") != QUACK_OWNER_MUTATION_REQUEST_SCHEMA
        or payload.get("protocol_revision") != QUACK_OWNER_MUTATION_PROTOCOL_REVISION
        or payload.get("request_id") != request_id
    ):
        raise QuackOwnerMutationError("request_schema_invalid")
    expected_binding = validate_mutation_binding(binding)
    if payload.get("binding") != expected_binding:
        raise QuackOwnerMutationError("request_binding_invalid")
    steps = payload.get("steps")
    if not isinstance(steps, list) or mutation_operation(steps) != payload.get("operation"):
        raise QuackOwnerMutationError("operation_shape_invalid")
    issued = payload.get("issued_at_ms")
    expires = payload.get("expires_at_ms")
    if (
        type(issued) is not int
        or type(expires) is not int
        or expires - issued != QUACK_OWNER_MUTATION_REQUEST_TTL_MS
    ):
        raise QuackOwnerMutationError("request_time_invalid")
    observed = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if issued > observed + QUACK_OWNER_MUTATION_MAX_CLOCK_SKEW_MS:
        raise QuackOwnerMutationError("request_from_future")
    if not allow_expired and expires < observed:
        raise QuackOwnerMutationError("request_expired")
    semantic = {
        "schema": QUACK_OWNER_MUTATION_SEMANTIC_SCHEMA,
        "protocol_revision": QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
        "operation": payload["operation"],
        "binding": expected_binding,
        "steps": steps,
    }
    if mutation_content_id(semantic) != request_id:
        raise QuackOwnerMutationError("request_identity_invalid")
    unsigned = dict(payload)
    auth_mac = unsigned.pop("auth_mac", None)
    request_cid = unsigned.pop("request_cid", None)
    if not isinstance(request_cid, str) or request_cid != mutation_content_id(unsigned):
        raise QuackOwnerMutationError("request_cid_invalid")
    authenticated = {**unsigned, "request_cid": request_cid}
    if (
        not isinstance(auth_mac, str)
        or not _DIGEST_RE.fullmatch(auth_mac)
        or not hmac.compare_digest(auth_mac, mutation_mac(authenticated, token))
    ):
        raise QuackOwnerMutationError("request_mac_invalid")
    return dict(payload)


def build_mutation_result(
    request: Mapping[str, Any],
    *,
    ok: bool,
    token: str,
    error_code: str = "",
    rowcounts: Sequence[int] = (),
    observed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if type(ok) is not bool:
        raise QuackOwnerMutationError("result_invalid")
    counts = [int(value) for value in rowcounts]
    if len(counts) > QUACK_OWNER_MUTATION_MAX_STEPS:
        raise QuackOwnerMutationError("result_invalid")
    unsigned = {
        "schema": QUACK_OWNER_MUTATION_RESULT_SCHEMA,
        "protocol_revision": QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
        "request_id": request["request_id"],
        "request_cid": request["request_cid"],
        "issued_at_ms": request["issued_at_ms"],
        "expires_at_ms": request["expires_at_ms"],
        "operation": request["operation"],
        "binding": dict(request["binding"]),
        "ok": ok,
        "error_code": "" if ok else str(error_code or "owner_transaction_failed")[:128],
        "rowcounts": counts,
        "observed": dict(observed or {}),
    }
    result_cid = mutation_content_id(unsigned)
    authenticated = {**unsigned, "result_cid": result_cid}
    return {**authenticated, "result_mac": mutation_mac(authenticated, token)}


def validate_mutation_result(
    payload: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    token: str,
) -> int:
    if not isinstance(payload, Mapping) or set(payload) != _RESULT_FIELDS:
        raise QuackOwnerMutationError("result_schema_invalid")
    result_cid = payload.get("result_cid")
    result_mac = payload.get("result_mac")
    unsigned = dict(payload)
    unsigned.pop("result_cid", None)
    unsigned.pop("result_mac", None)
    authenticated = {**unsigned, "result_cid": result_cid}
    if (
        payload.get("schema") != QUACK_OWNER_MUTATION_RESULT_SCHEMA
        or payload.get("protocol_revision") != QUACK_OWNER_MUTATION_PROTOCOL_REVISION
        or any(
            payload.get(name) != request.get(name)
            for name in (
                "request_id", "request_cid", "issued_at_ms", "expires_at_ms",
                "operation", "binding",
            )
        )
        or not isinstance(result_cid, str)
        or result_cid != mutation_content_id(unsigned)
        or not isinstance(result_mac, str)
        or not _DIGEST_RE.fullmatch(result_mac)
        or not hmac.compare_digest(result_mac, mutation_mac(authenticated, token))
    ):
        raise QuackOwnerMutationError("result_authentication_invalid")
    if payload.get("ok") is not True:
        code = str(payload.get("error_code") or "owner_transaction_failed")
        if code in {"cas_conflict", "event_head_conflict", "lease_conflict"}:
            raise QuackOwnerMutationConflictError(code)
        if code in {"unknown_external_outcome", "read_replica_refresh_unknown_outcome"}:
            raise QuackOwnerMutationUnknownOutcomeError(code)
        raise QuackOwnerMutationError(code)
    counts = payload.get("rowcounts")
    if not isinstance(counts, list) or any(type(value) is not int for value in counts):
        raise QuackOwnerMutationError("result_rowcounts_invalid")
    return counts[0] if counts else -1


# ---------------------------------------------------------------------------
# Owner-only inbox I/O
# ---------------------------------------------------------------------------

_AT_FDCWD: Final[int] = -100
_RENAME_NOREPLACE: Final[int] = 1
try:
    _LIBC = ctypes.CDLL(None, use_errno=True)
    _RENAMEAT2 = getattr(_LIBC, "renameat2", None)
except OSError:  # pragma: no cover - fail-closed platform boundary
    _RENAMEAT2 = None
if _RENAMEAT2 is not None:
    _RENAMEAT2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    _RENAMEAT2.restype = ctypes.c_int


def _entry_name(value: str | os.PathLike[str]) -> str:
    name = os.fspath(value)
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "\x00" in name
        or os.path.basename(name) != name
    ):
        raise QuackOwnerMutationError("inbox_filename_invalid")
    return name


def _rename_noreplace_at(directory_fd: int, source: str, target: str) -> None:
    if _RENAMEAT2 is None:
        raise QuackOwnerMutationError("atomic_publication_unavailable")
    ctypes.set_errno(0)
    result = _RENAMEAT2(
        int(directory_fd), os.fsencode(_entry_name(source)), int(directory_fd),
        os.fsencode(_entry_name(target)), _RENAME_NOREPLACE,
    )
    if result != 0:
        number = ctypes.get_errno() or errno.EIO
        raise OSError(number, os.strerror(number), target)


def open_mutation_inbox_directory(path: Path) -> int:
    """Create and pin one absolute owner-only directory without following links."""

    target = Path(path).expanduser()
    if not target.is_absolute() or not target.name or "\x00" in os.fspath(target):
        raise QuackOwnerMutationError("unsafe_inbox")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise QuackOwnerMutationError("unsafe_inbox")
    flags = (
        os.O_RDONLY | nofollow | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(target.anchor, flags)
    try:
        for component in target.parts[1:]:
            child = -1
            try:
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    child = os.open(component, flags, dir_fd=descriptor)
                opened = os.fstat(child)
                named = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or not stat.S_ISDIR(named.st_mode)
                    or (opened.st_dev, opened.st_ino, opened.st_uid)
                    != (named.st_dev, named.st_ino, named.st_uid)
                ):
                    raise QuackOwnerMutationError("unsafe_inbox")
            except BaseException:
                if child >= 0:
                    os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        os.fchmod(descriptor, 0o700)
        admitted = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(admitted.st_mode)
            or admitted.st_uid != os.geteuid()
            or stat.S_IMODE(admitted.st_mode) != 0o700
        ):
            raise QuackOwnerMutationError("unsafe_inbox")
        result = descriptor
        descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def mutation_envelope_exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(_entry_name(name), dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def unlink_mutation_envelope_at(
    directory_fd: int, name: str, *, missing_ok: bool = False
) -> None:
    try:
        os.unlink(_entry_name(name), dir_fd=directory_fd)
    except FileNotFoundError:
        if not missing_ok:
            raise


def rename_mutation_envelope_noreplace_at(
    directory_fd: int, source: str, target: str
) -> None:
    _rename_noreplace_at(directory_fd, source, target)
    os.fsync(directory_fd)


def write_envelope_atomic_at(
    directory_fd: int,
    name: str,
    payload: Mapping[str, Any],
    *,
    replace: bool = False,
) -> None:
    target = _entry_name(name)
    encoded = _canonical_bytes(payload) + b"\n"
    if len(encoded) > QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES:
        raise QuackOwnerMutationError("envelope_too_large")
    temporary = f".{target}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o600)
            os.fsync(stream.fileno())
        if replace:
            os.replace(
                temporary, target, src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        else:
            _rename_noreplace_at(directory_fd, temporary, target)
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_envelope_at(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int = QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES,
) -> dict[str, Any]:
    filename = _entry_name(name)
    descriptor = os.open(
        filename,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) & 0o077
            or before.st_size < 2
            or before.st_size > max_bytes
        ):
            raise QuackOwnerMutationError("unsafe_envelope")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        def identity(item: os.stat_result) -> tuple[int, ...]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_uid,
                item.st_nlink,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
        if (
            len(encoded) != before.st_size
            or identity(before) != identity(after)
            or identity(after) != identity(named)
        ):
            raise QuackOwnerMutationError("unstable_envelope")
        payload = json.loads(
            encoded.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON number: {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, QuackOwnerMutationError):
            raise
        raise QuackOwnerMutationError("malformed_envelope") from exc
    finally:
        os.close(descriptor)
    if not isinstance(payload, dict):
        raise QuackOwnerMutationError("malformed_envelope")
    if _canonical_bytes(payload) + b"\n" != encoded:
        raise QuackOwnerMutationError("noncanonical_envelope")
    return payload


def _request_for_retained_result(
    result: Mapping[str, Any],
    *,
    steps: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
    token: str,
) -> dict[str, Any]:
    issued = result.get("issued_at_ms")
    if type(issued) is not int:
        raise QuackOwnerMutationError("result_schema_invalid")
    request = build_mutation_request(
        steps=steps, binding=binding, token=token, issued_at_ms=issued
    )
    if request["request_id"] != result.get("request_id"):
        raise QuackOwnerMutationError("result_identity_invalid")
    return request


def execute_mutation_bundle(
    steps: Sequence[Mapping[str, Any]],
    *,
    binding: Mapping[str, Any],
    token: str,
    inbox: Path,
) -> int:
    """Publish a semantic transaction and return its authenticated row count."""
    request = build_mutation_request(steps=steps, binding=binding, token=token)
    result = execute_owner_request(
        request, binding=binding, token=token, inbox=inbox
    )
    counts = result["rowcounts"]
    return counts[0] if counts else -1


def execute_owner_request(
    request: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    token: str,
    inbox: Path,
) -> dict[str, Any]:
    """Publish one semantic transaction and admit only a signed terminal result."""

    request = validate_mutation_request(
        request, request_id=request["request_id"], binding=binding, token=token
    )
    steps = request["steps"]
    request_id = request["request_id"]
    request_name = f"{request_id}.request.json"
    processing_name = f"{request_id}.processing.json"
    done_name = f"{request_id}.done.json"
    cancelled_name = f"{request_id}.cancelled.json"
    descriptor = open_mutation_inbox_directory(Path(inbox).resolve())
    try:
        if mutation_envelope_exists_at(descriptor, done_name):
            retained = read_envelope_at(descriptor, done_name)
            retained_request = _request_for_retained_result(
                retained, steps=steps, binding=binding, token=token
            )
            validate_mutation_result(
                retained, request=retained_request, token=token
            )
            return dict(retained)
        try:
            write_envelope_atomic_at(descriptor, request_name, request)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            existing = read_envelope_at(descriptor, request_name)
            validate_mutation_request(
                existing,
                request_id=request_id,
                binding=binding,
                token=token,
                allow_expired=True,
            )
        deadline = time.monotonic() + (
            QUACK_OWNER_MUTATION_REQUEST_TTL_MS / 1000.0
        )
        while time.monotonic() < deadline:
            if mutation_envelope_exists_at(descriptor, done_name):
                result = read_envelope_at(descriptor, done_name)
                validate_mutation_result(
                    result, request=request, token=token
                )
                unlink_mutation_envelope_at(
                    descriptor, request_name, missing_ok=True
                )
                return dict(result)
            time.sleep(0.025)
        try:
            rename_mutation_envelope_noreplace_at(
                descriptor, request_name, cancelled_name
            )
        except OSError as exc:
            if exc.errno not in {errno.ENOENT, errno.EEXIST}:
                raise
        else:
            unlink_mutation_envelope_at(
                descriptor, cancelled_name, missing_ok=True
            )
            raise QuackOwnerMutationError(
                "request_cancelled_before_owner_claim"
            )
        settlement = time.monotonic() + (
            QUACK_OWNER_MUTATION_SETTLEMENT_MS / 1000.0
        )
        while time.monotonic() < settlement:
            if mutation_envelope_exists_at(descriptor, done_name):
                result = read_envelope_at(descriptor, done_name)
                validate_mutation_result(
                    result, request=request, token=token
                )
                return dict(result)
            if (
                not mutation_envelope_exists_at(descriptor, processing_name)
                and mutation_envelope_exists_at(descriptor, request_name)
            ):
                time.sleep(0.025)
                continue
            time.sleep(0.025)
        raise QuackOwnerMutationUnknownOutcomeError(
            "unknown_external_outcome"
        )
    finally:
        os.close(descriptor)


# ---------------------------------------------------------------------------
# Exclusive-writer semantic validation and atomic execution
# ---------------------------------------------------------------------------

_READY_FROM: Final = frozenset(
    {"todo", "ready", "open", "pending", "queued", "proposed", "admitted"}
)
_ALLOWED_STATUS_TRANSITIONS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        **{status: frozenset({"in_progress"}) for status in _READY_FROM},
        "retrying": frozenset({"in_progress", "blocked"}),
        "claimed": frozenset({"in_progress", "ready", "blocked"}),
        "running": frozenset({"ready", "completed", "blocked", "retrying"}),
        "in_progress": frozenset({"ready", "completed", "blocked", "retrying"}),
        "blocked": frozenset({"retrying", "ready"}),
    }
)


def _parameters(step: Mapping[str, Any], count: int) -> list[Any]:
    values = validate_mutation_parameters(step.get("parameters"))
    if len(values) != count:
        raise QuackOwnerMutationError("parameter_count_invalid")
    return values


def _json_object(value: Any, *, code: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise QuackOwnerMutationError(code)
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda member: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON number: {member}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QuackOwnerMutationError(code) from exc
    if not isinstance(payload, dict) or _canonical_bytes(payload).decode("utf-8") != value:
        raise QuackOwnerMutationError(code)
    return payload


def _row_tuple(row: Any) -> tuple[Any, ...]:
    values = getattr(row, "_values", None)
    if values is not None:
        return tuple(values)
    if isinstance(row, Mapping):
        return tuple(row.values())
    return tuple(row)


def _validate_domain_event(
    connection: Any,
    parameters: list[Any],
    *,
    replay: bool = False,
) -> dict[str, Any]:
    event_body = _json_object(parameters[9], code="event_body_invalid")
    if set(event_body) != {
        "schema", "event_type", "subject_id", "body", "recorded_at", "owner_id"
    }:
        raise QuackOwnerMutationError("event_body_invalid")
    if (
        event_body.get("event_type") != parameters[4]
        or event_body.get("recorded_at") != parameters[8]
        or not isinstance(event_body.get("body"), dict)
    ):
        raise QuackOwnerMutationError("event_body_invalid")
    expected_id = content_identity(
        {
            "stream_id": parameters[1],
            "sequence": parameters[2],
            "global_sequence": parameters[3],
            "event_type": parameters[4],
            "body": event_body,
        }
    )
    if parameters[0] != expected_id:
        raise QuackOwnerMutationError("event_identity_invalid")
    if (
        type(parameters[2]) is not int
        or type(parameters[3]) is not int
        or parameters[2] < 1
        or parameters[3] < 1
    ):
        raise QuackOwnerMutationConflictError("event_head_conflict")
    # Exact replay has already resolved this immutable event row byte-for-byte.
    # Requiring it to be the *next* event would reject every committed replay;
    # operation-level validators below still bind the event to every other
    # effect before idempotent success is admitted.
    if replay:
        return event_body
    row = connection.execute(
        "SELECT COALESCE(MAX(sequence), 0), "
        "(SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events) "
        "FROM domain_events WHERE stream_id = ?",
        [parameters[1]],
    ).fetchone()
    if (
        row is None
        or parameters[2] != int(row[0]) + 1
        or parameters[3] != int(row[1]) + 1
    ):
        raise QuackOwnerMutationConflictError("event_head_conflict")
    return event_body


def _parse_iso_ms(value: str) -> int:
    text = str(value or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return 0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def _completion_evidence_missing(
    connection: Any,
    task_cid: str,
    *,
    evidence_digests: Sequence[str],
    now_ms: int,
) -> tuple[str, ...]:
    acceptance_rows = connection.execute(
        "SELECT ordinal, criterion, evidence_policy_json FROM task_acceptance "
        "WHERE task_cid = ? ORDER BY ordinal",
        [task_cid],
    ).fetchall()
    evidence_rows = connection.execute(
        "SELECT evidence_kind, digest, created_at FROM evidence_nodes "
        "WHERE task_cid = ?",
        [task_cid],
    ).fetchall()
    current: list[tuple[str, str]] = []
    for row in evidence_rows:
        created = _parse_iso_ms(str(row[2] or ""))
        if created and now_ms - created > 3_600_000:
            continue
        current.append((str(row[0] or ""), str(row[1] or "")))
    kinds = {kind for kind, _digest in current}
    digests = {digest for _kind, digest in current}
    missing = [
        f"digest:{digest}"
        for digest in evidence_digests
        if not isinstance(digest, str) or digest not in digests
    ]
    if not acceptance_rows and not kinds.intersection({"validation", "test", "acceptance"}):
        missing.append("required:current_validation_evidence")
    for row in acceptance_rows:
        policy = _json_object(str(row[2] or "{}"), code="acceptance_policy_invalid")
        required_digest = str(
            policy.get("required_digest")
            or policy.get("evidence_digest")
            or policy.get("digest")
            or ""
        )
        required_kind = str(policy.get("evidence_kind") or policy.get("kind") or "")
        if required_digest and required_digest not in digests:
            missing.append(f"digest:{required_digest}")
        elif required_kind and required_kind not in kinds:
            missing.append(f"kind:{required_kind}")
        elif not required_digest and not required_kind and not current:
            missing.append(f"criterion:{row[1] or row[0]}")
    return tuple(sorted(set(missing)))


def _validate_task_transition(
    connection: Any,
    steps: Sequence[Mapping[str, Any]],
    *,
    now_ms: int,
    replay: bool = False,
) -> dict[str, Any]:
    update = _parameters(steps[0], 6)
    revision = _parameters(steps[1], 5)
    completing = len(steps) == 4
    event = _parameters(steps[-1], 10)
    event_body = _validate_domain_event(connection, event, replay=replay)
    inner = event_body["body"]
    task = connection.execute(
        "SELECT task_alias, goal_cid, status, revision, body_json "
        "FROM tasks WHERE task_cid = ?",
        [update[4]],
    ).fetchone()
    if task is None:
        raise QuackOwnerMutationError("task_missing")
    if replay:
        if str(task[2]) != str(update[0]) or int(task[3]) != int(update[1]):
            raise QuackOwnerMutationError("replay_integrity_failure")
        previous = connection.execute(
            "SELECT status FROM task_revisions "
            "WHERE task_cid = ? AND revision = ?",
            [update[4], update[5]],
        ).fetchone()
        if previous is None:
            raise QuackOwnerMutationError("replay_integrity_failure")
        old_status, old_revision = str(previous[0]), int(update[5])
    else:
        old_status, old_revision = str(task[2]), int(task[3])
        if old_revision != update[5]:
            raise QuackOwnerMutationConflictError("cas_conflict")
    if (
        update[0] not in _ALLOWED_STATUS_TRANSITIONS.get(old_status, frozenset())
        or update[1] != update[5] + 1
    ):
        raise QuackOwnerMutationError("transition_invalid")
    _json_object(update[3], code="task_body_invalid")
    if revision != [update[4], update[1], update[0], update[3], update[2]]:
        raise QuackOwnerMutationError("revision_binding_invalid")
    expected_event_type = (
        "intent.completion_recorded" if update[0] == "completed"
        else "intent.task_status_changed"
    )
    if completing != (update[0] == "completed"):
        raise QuackOwnerMutationError("completion_shape_invalid")
    if (
        not isinstance(inner, dict)
        or event[1] != "stream:intent"
        or event[4] != expected_event_type
        or event[5] != update[4]
        or event_body["subject_id"] != update[4]
        or inner.get("task_cid") != update[4]
        or inner.get("task_alias") != str(task[0])
        or inner.get("goal_cid") != str(task[1])
        or inner.get("previous_status") != old_status
        or inner.get("status") != update[0]
        or inner.get("revision") != update[1]
        or inner.get("recorded_at") != update[2]
    ):
        raise QuackOwnerMutationError("event_binding_invalid")
    if completing:
        completion = _parameters(steps[2], 10)
        body = _json_object(completion[9], code="completion_receipt_invalid")
        evidence_digests = body.get("evidence_digests")
        receipt = body.get("receipt")
        if (
            set(body) != {"schema", "receipt", "evidence_digests", "revision"}
            or not isinstance(receipt, dict)
            or not isinstance(evidence_digests, list)
            or any(not isinstance(item, str) for item in evidence_digests)
            or completion[1] != update[4]
            or completion[2] != str(task[1])
            or completion[6] != update[2]
            or body.get("revision") != update[1]
            or inner.get("completion_receipt_cid") != completion[0]
            or inner.get("evidence_digest") != completion[8]
        ):
            raise QuackOwnerMutationError("completion_receipt_invalid")
        expected_evidence = content_identity(
            {
                "task_cid": update[4], "revision": update[1],
                "receipt": receipt, "evidence_digests": evidence_digests,
            }
        )
        expected_receipt = content_identity(
            {
                "namespace": "completion-receipt", "task_cid": update[4],
                "revision": update[1], "evidence_digest": expected_evidence,
            }
        )
        if completion[8] != expected_evidence or completion[0] != expected_receipt:
            raise QuackOwnerMutationError("completion_receipt_invalid")
        if not replay and _completion_evidence_missing(
            connection, update[4], evidence_digests=evidence_digests,
            now_ms=now_ms,
        ):
            raise QuackOwnerMutationError("completion_evidence_stale")
    return {
        "task_cid": update[4], "old_status": old_status,
        "old_revision": old_revision, "new_status": update[0],
        "new_revision": update[1], "event_id": event[0],
    }


def _validate_validation_record(
    connection: Any,
    steps: Sequence[Mapping[str, Any]],
    *,
    replay: bool = False,
) -> dict[str, Any]:
    run = _parameters(steps[0], 8)
    result = _parameters(steps[1], 7)
    passed = len(steps) == 5
    event = _parameters(steps[-1], 10)
    run_body = _json_object(run[7], code="validation_body_invalid")
    _json_object(result[6], code="validation_body_invalid")
    event_body = _validate_domain_event(connection, event, replay=replay)
    inner = event_body["body"]
    argv = run_body.get("argv")
    if (
        not isinstance(argv, list)
        or any(not isinstance(item, str) for item in argv)
        or run[4] != run[3]
        or run[5] not in {"passed", "failed", "error", "skipped"}
        or result[1] != run[0]
        or result[2] != run[1]
        or result[3] != 0
        or result[4] != run[5]
        or passed != (run[5] == "passed")
        or run[6] != content_identity({"argv": argv})
        or run[0] != content_identity(
            {
                "task_cid": run[1], "attempt_id": run[2], "argv": argv,
                "recorded_at": run[3],
            }
        )
        or result[0] != content_identity(
            {"run_id": run[0], "outcome": result[4], "evidence_digest": result[5]}
        )
    ):
        raise QuackOwnerMutationError("validation_binding_invalid")
    if connection.execute("SELECT 1 FROM tasks WHERE task_cid = ?", [run[1]]).fetchone() is None:
        raise QuackOwnerMutationError("task_missing")
    if (
        not isinstance(inner, dict)
        or event[1] != "stream:intent"
        or event[4] != "intent.validation_recorded"
        or event[5] != run[1]
        or event[6] != run[2]
        or event_body["subject_id"] != result[0]
        or inner.get("result_id") != result[0]
        or inner.get("run_id") != run[0]
        or inner.get("task_cid") != run[1]
        or inner.get("outcome") != run[5]
        or inner.get("evidence_digest") != result[5]
    ):
        raise QuackOwnerMutationError("event_binding_invalid")
    if passed:
        delete = _parameters(steps[2], 1)
        evidence = _parameters(steps[3], 7)
        evidence_body = _json_object(evidence[6], code="evidence_binding_invalid")
        expected_id = content_identity(
            {
                "task_cid": run[1], "evidence_kind": "validation",
                "digest": result[5], "run_id": run[0],
            }
        )
        if (
            delete[0] != expected_id
            or evidence[:5] != [expected_id, "", run[1], "validation", result[5]]
            or evidence_body.get("run_id") != run[0]
            or evidence_body.get("result_id") != result[0]
            or evidence_body.get("outcome") != "passed"
        ):
            raise QuackOwnerMutationError("evidence_binding_invalid")
    return {
        "task_cid": run[1], "run_id": run[0], "result_id": result[0],
        "event_id": event[0], "outcome": run[5],
    }


def _validate_evidence_record(
    connection: Any,
    steps: Sequence[Mapping[str, Any]],
    *,
    replay: bool = False,
) -> dict[str, Any]:
    delete = _parameters(steps[0], 1)
    evidence = _parameters(steps[1], 7)
    event = _parameters(steps[2], 10)
    body = _json_object(evidence[6], code="evidence_binding_invalid")
    event_body = _validate_domain_event(connection, event, replay=replay)
    inner = event_body["body"]
    expected_id = content_identity(
        {
            "task_cid": evidence[2], "evidence_kind": evidence[3],
            "digest": evidence[4], "body": body,
        }
    )
    if connection.execute("SELECT 1 FROM tasks WHERE task_cid = ?", [evidence[2]]).fetchone() is None:
        raise QuackOwnerMutationError("task_missing")
    if (
        delete[0] != expected_id
        or evidence[0] != expected_id
        or event[1] != "stream:intent"
        or event[4] != "intent.evidence_recorded"
        or event[5] != evidence[2]
        or event_body["subject_id"] != expected_id
        or not isinstance(inner, dict)
        or inner.get("evidence_id") != expected_id
        or inner.get("parent_evidence_id") != evidence[1]
        or inner.get("task_cid") != evidence[2]
        or inner.get("evidence_kind") != evidence[3]
        or inner.get("digest") != evidence[4]
        or inner.get("body") != body
        or inner.get("created_at") != evidence[5]
    ):
        raise QuackOwnerMutationError("evidence_binding_invalid")
    return {"task_cid": evidence[2], "evidence_id": expected_id, "event_id": event[0]}


def _validate_queue_backoff(
    connection: Any,
    steps: Sequence[Mapping[str, Any]],
    *,
    replay: bool = False,
) -> dict[str, Any]:
    inserting = steps[0].get("template_id") == QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT
    lease = _parameters(steps[0], 17 if inserting else 6)
    event = _parameters(steps[-1], 10)
    event_body = _validate_domain_event(connection, event, replay=replay)
    inner = event_body["body"]
    task_cid = str(lease[0] if inserting else lease[5])
    extension = _json_object(
        lease[16] if inserting else lease[4],
        code="lease_binding_invalid",
    )
    attempt = lease[7] if inserting else lease[0]
    retry_not_before_ms = lease[11] if inserting else lease[1]
    reason = lease[10] if inserting else lease[2]
    extension_schema = lease[15] if inserting else lease[3]
    if not task_cid or connection.execute(
        "SELECT 1 FROM tasks WHERE task_cid = ?", [task_cid]
    ).fetchone() is None:
        raise QuackOwnerMutationError("task_missing")
    if (
        type(attempt) is not int
        or attempt < 1
        or type(retry_not_before_ms) is not int
        or retry_not_before_ms < 0
        or not isinstance(reason, str)
        or not isinstance(extension_schema, str)
        or not extension_schema
        or set(extension) != {
            "selection_penalty", "consecutive_failures", "reason"
        }
        or extension.get("consecutive_failures") != attempt
        or extension.get("reason") != reason
        or event[1] != "stream:intent"
        or event[4] != "intent.queue_backoff"
        or event[5] != task_cid
        or event_body["subject_id"] != task_cid
        or not isinstance(inner, dict)
        or set(inner) != {
            "task_cid", "attempt", "retry_not_before_ms", "delay_ms",
            "selection_penalty", "reason", "revision",
        }
        or inner.get("task_cid") != task_cid
        or inner.get("attempt") != attempt
        or inner.get("retry_not_before_ms") != retry_not_before_ms
        or inner.get("selection_penalty") != extension.get("selection_penalty")
        or inner.get("reason") != reason
        or inner.get("revision") != attempt
        or type(inner.get("delay_ms")) is not int
        or inner["delay_ms"] < 0
    ):
        raise QuackOwnerMutationError("lease_binding_invalid")
    if inserting and (
        lease[8] != "released"
        or lease[14] != attempt
        or event[7] != lease[12]
        or event_body["owner_id"] != lease[3]
        or type(lease[9]) is not int
        or retry_not_before_ms - lease[9] != inner["delay_ms"]
    ):
        raise QuackOwnerMutationError("lease_binding_invalid")
    existing = connection.execute(
        "SELECT 1 FROM leases WHERE task_cid = ?", [task_cid]
    ).fetchone()
    if replay:
        if existing is None:
            raise QuackOwnerMutationError("replay_integrity_failure")
    else:
        if inserting and existing is not None:
            raise QuackOwnerMutationConflictError("lease_conflict")
        if not inserting and existing is None:
            raise QuackOwnerMutationError("lease_missing")
    return {"task_cid": task_cid, "event_id": event[0], "inserted": inserting}


def _validate_operation(
    connection: Any,
    request: Mapping[str, Any],
    *,
    now_ms: int,
    replay: bool = False,
) -> dict[str, Any]:
    operation = request["operation"]
    steps = request["steps"]
    if operation == QUACK_MUTATION_TASK_STATUS_TRANSITION:
        return _validate_task_transition(
            connection, steps, now_ms=now_ms, replay=replay
        )
    if operation == QUACK_MUTATION_VALIDATION_RECORD:
        return _validate_validation_record(connection, steps, replay=replay)
    if operation == QUACK_MUTATION_EVIDENCE_RECORD:
        return _validate_evidence_record(connection, steps, replay=replay)
    if operation == QUACK_MUTATION_QUEUE_BACKOFF:
        return _validate_queue_backoff(connection, steps, replay=replay)
    raise QuackOwnerMutationError("operation_not_allowlisted")


def _record_matches(
    connection: Any, table: str, columns: Sequence[str], key: str, values: Sequence[Any]
) -> bool:
    selected = ", ".join(columns)
    row = connection.execute(
        f"SELECT {selected} FROM {table} WHERE {columns[0]} = ?", [key]
    ).fetchone()
    return row is not None and _row_tuple(row) == tuple(values)


def mutation_effects_present(
    connection: Any, request: Mapping[str, Any]
) -> bool:
    if request["operation"] == "owner_snapshot@1":
        # An interrupted read is not a durable effect. Its caller needs a new
        # challenge, rather than an invented observation of past owner state.
        return False
    steps = request["steps"]
    event = _parameters(steps[-1], 10)
    event_row = connection.execute(
        "SELECT event_id, stream_id, sequence, global_sequence, event_type, "
        "task_cid, attempt_id, session_id, recorded_at, body_json "
        "FROM domain_events WHERE event_id = ?",
        [event[0]],
    ).fetchone()
    if event_row is None:
        return False
    if _row_tuple(event_row) != tuple(event):
        raise QuackOwnerMutationError("replay_integrity_failure")
    for step in steps[:-1]:
        template = step["template_id"]
        values = _parameters(step, len(step["parameters"]))
        exact = True
        if template == QUACK_MUTATION_TASK_STATUS_CAS:
            row = connection.execute(
                "SELECT status, revision, updated_at, body_json FROM tasks "
                "WHERE task_cid = ?", [values[4]]
            ).fetchone()
            exact = row is not None and _row_tuple(row) == tuple(values[:4])
        elif template == QUACK_MUTATION_TASK_REVISION_INSERT:
            row = connection.execute(
                "SELECT task_cid, revision, status, body_json, recorded_at "
                "FROM task_revisions WHERE task_cid = ? AND revision = ?",
                values[:2],
            ).fetchone()
            exact = row is not None and _row_tuple(row) == tuple(values)
        elif template == QUACK_MUTATION_COMPLETION_RECEIPT_INSERT:
            exact = _record_matches(
                connection, "completion_receipts",
                ("receipt_cid", "task_cid", "goal_cid", "attempt_id", "claim_cid",
                 "fencing_token", "completed_at", "validation_run_id",
                 "evidence_digest", "body_json"), values[0], values,
            )
        elif template == QUACK_MUTATION_VALIDATION_RUN_INSERT:
            exact = _record_matches(
                connection, "validation_runs",
                ("run_id", "task_cid", "attempt_id", "started_at", "finished_at",
                 "status", "command_digest", "body_json"), values[0], values,
            )
        elif template == QUACK_MUTATION_VALIDATION_RESULT_INSERT:
            exact = _record_matches(
                connection, "validation_results",
                ("result_id", "run_id", "task_cid", "ordinal", "outcome",
                 "evidence_digest", "body_json"), values[0], values,
            )
        elif template == QUACK_MUTATION_EVIDENCE_INSERT:
            exact = _record_matches(
                connection, "evidence_nodes",
                ("evidence_id", "parent_evidence_id", "task_cid", "evidence_kind",
                 "digest", "created_at", "body_json"), values[0], values,
            )
        elif template == QUACK_MUTATION_EVIDENCE_DELETE:
            # Every admitted delete is paired with an exact insert of the same key.
            continue
        elif template == QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT:
            exact = _record_matches(
                connection, "leases",
                ("task_cid", "claim_cid", "resolution_cid", "claimant_did",
                 "logical_epoch", "fencing_token", "expires_at_ms", "attempt",
                 "state", "started_at_ms", "release_reason", "retry_not_before_ms",
                 "owner_session_id", "fence_epoch", "revision", "extension_schema",
                 "extension_json"), values[0], values,
            )
        elif template == QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE:
            row = connection.execute(
                "SELECT attempt, retry_not_before_ms, release_reason, state, "
                "extension_schema, extension_json FROM leases WHERE task_cid = ?",
                [values[5]],
            ).fetchone()
            exact = row is not None and _row_tuple(row) == (
                values[0], values[1], values[2], "released", values[3], values[4]
            )
        if not exact:
            raise QuackOwnerMutationError("replay_integrity_failure")
    return True


def execute_owner_mutation(
    connection: Any,
    request: Mapping[str, Any],
    *,
    refresh_replica: Any,
    now_ms: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Validate and commit one admitted bundle on the exclusive writer."""

    if request["operation"] == "owner_snapshot@1":
        from .quack_owner_snapshot import execute_owner_snapshot

        return [0], execute_owner_snapshot(connection, request)
    observed_now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if mutation_effects_present(connection, request):
        # Independent row equality is necessary but not sufficient: an
        # authenticated caller could otherwise compose already-present rows
        # from unrelated admitted operations. Re-run all immutable semantic
        # bindings in replay mode before accepting those effects as one bundle.
        observed = _validate_operation(
            connection, request, now_ms=observed_now, replay=True
        )
        observed.update(
            {
                "idempotent_replay": True,
                "request_id": request["request_id"],
            }
        )
        observed["read_replica"] = dict(refresh_replica())
        return [1] * len(request["steps"]), observed
    connection.execute("BEGIN TRANSACTION")
    committed = False
    try:
        observed = _validate_operation(connection, request, now_ms=observed_now)
        rowcounts: list[int] = []
        for index, step in enumerate(request["steps"]):
            cursor = connection.execute(
                MUTATION_SQL_TEMPLATES[step["template_id"]], step["parameters"]
            )
            rowcount = int(getattr(cursor, "rowcount", -1))
            if (
                index == 0
                and request["operation"] == QUACK_MUTATION_TASK_STATUS_TRANSITION
                and rowcount != 1
            ):
                raise QuackOwnerMutationConflictError("cas_conflict")
            rowcounts.append(rowcount)
        connection.execute("COMMIT")
        committed = True
    except BaseException:
        if not committed:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
        raise
    try:
        observed["read_replica"] = dict(refresh_replica())
    except BaseException as exc:
        raise QuackOwnerMutationUnknownOutcomeError(
            "read_replica_refresh_unknown_outcome"
        ) from exc
    return rowcounts, observed


def _write_owner_result(
    descriptor: int,
    done_name: str,
    request: Mapping[str, Any],
    *,
    token: str,
    ok: bool,
    error_code: str = "",
    rowcounts: Sequence[int] = (),
    observed: Mapping[str, Any] | None = None,
) -> None:
    result = build_mutation_result(
        request, ok=ok, token=token, error_code=error_code,
        rowcounts=rowcounts, observed=observed,
    )
    write_envelope_atomic_at(descriptor, done_name, result)


def service_mutation_inbox(
    connection: Any,
    *,
    inbox: Path,
    binding: Mapping[str, Any],
    token: str,
    refresh_replica: Any,
    max_requests: int = MUTATION_MAX_PER_PASS,
) -> int:
    """Claim and execute bounded authenticated bundles on the exclusive writer."""

    if type(max_requests) is not int or not 1 <= max_requests <= MUTATION_MAX_PER_PASS:
        raise ValueError("max_requests is outside the closed service bound")
    admitted_binding = validate_mutation_binding(binding)
    descriptor = open_mutation_inbox_directory(Path(inbox).resolve())
    serviced = 0
    try:
        entries = tuple(os.listdir(descriptor))
        if len(entries) > MUTATION_MAX_DIRECTORY_ENTRIES:
            raise QuackOwnerMutationError("inbox_population_exceeded")

        # Reconcile a claim retained across an owner interruption before new work.
        for name in sorted(entries):
            match = _PROCESSING_NAME_RE.fullmatch(name)
            if match is None:
                continue
            request_id = match.group("request_id")
            done_name = f"{request_id}.done.json"
            try:
                request = validate_mutation_request(
                    read_envelope_at(descriptor, name),
                    request_id=request_id,
                    binding=admitted_binding,
                    token=token,
                    allow_expired=True,
                )
                if mutation_envelope_exists_at(descriptor, done_name):
                    retained = read_envelope_at(descriptor, done_name)
                    validate_mutation_result(retained, request=request, token=token)
                elif mutation_effects_present(connection, request):
                    observed = _validate_operation(
                        connection,
                        request,
                        now_ms=int(time.time() * 1000),
                        replay=True,
                    )
                    observed.update({
                        "reconciled_after_interruption": True,
                        "read_replica": dict(refresh_replica()),
                    })
                    _write_owner_result(
                        descriptor, done_name, request, token=token, ok=True,
                        rowcounts=[1] * len(request["steps"]), observed=observed,
                    )
                else:
                    _write_owner_result(
                        descriptor, done_name, request, token=token, ok=False,
                        error_code="owner_interrupted_no_effect",
                        observed={"reconciled_after_interruption": True},
                    )
            except QuackOwnerMutationError:
                pass
            finally:
                unlink_mutation_envelope_at(descriptor, name, missing_ok=True)

        for request_name in sorted(os.listdir(descriptor)):
            if serviced >= max_requests:
                break
            match = _REQUEST_NAME_RE.fullmatch(request_name)
            if match is None:
                continue
            request_id = match.group("request_id")
            processing_name = f"{request_id}.processing.json"
            done_name = f"{request_id}.done.json"
            claimed = False
            try:
                try:
                    rename_mutation_envelope_noreplace_at(
                        descriptor, request_name, processing_name
                    )
                    claimed = True
                except OSError as exc:
                    if exc.errno in {errno.ENOENT, errno.EEXIST}:
                        continue
                    raise
                request = validate_mutation_request(
                    read_envelope_at(descriptor, processing_name),
                    request_id=request_id,
                    binding=admitted_binding,
                    token=token,
                )
                if mutation_envelope_exists_at(descriptor, done_name):
                    retained = read_envelope_at(descriptor, done_name)
                    validate_mutation_result(retained, request=request, token=token)
                else:
                    try:
                        rowcounts, observed = execute_owner_mutation(
                            connection, request, refresh_replica=refresh_replica
                        )
                    except QuackOwnerMutationError as exc:
                        _write_owner_result(
                            descriptor, done_name, request, token=token,
                            ok=False, error_code=exc.code,
                        )
                    except Exception:
                        _write_owner_result(
                            descriptor, done_name, request, token=token,
                            ok=False, error_code="owner_transaction_failed",
                        )
                    else:
                        _write_owner_result(
                            descriptor, done_name, request, token=token, ok=True,
                            rowcounts=rowcounts, observed=observed,
                        )
                serviced += 1
            except (OSError, ValueError, QuackOwnerMutationError):
                # Malformed or unauthenticated input receives no signed oracle.
                serviced += 1
            finally:
                if claimed:
                    unlink_mutation_envelope_at(
                        descriptor, processing_name, missing_ok=True
                    )
        return serviced
    finally:
        os.close(descriptor)


__all__ = (
    "MUTATION_SQL_TEMPLATES",
    "MUTATION_SQL_TO_TEMPLATE",
    "QUACK_OWNER_MUTATION_PROTOCOL_REVISION",
    "QUACK_OWNER_MUTATION_REQUEST_SCHEMA",
    "QUACK_OWNER_MUTATION_RESULT_SCHEMA",
    "QUACK_OWNER_MUTATION_MAX_STEPS",
    "QuackOwnerMutationConflictError",
    "QuackOwnerMutationError",
    "QuackOwnerMutationUnknownOutcomeError",
    "build_mutation_request",
    "execute_mutation_bundle",
    "execute_owner_mutation",
    "execute_owner_request",
    "mutation_binding_from_identity",
    "mutation_content_id",
    "mutation_envelope_exists_at",
    "mutation_mac",
    "mutation_operation",
    "mutation_step",
    "open_mutation_inbox_directory",
    "read_envelope_at",
    "rename_mutation_envelope_noreplace_at",
    "service_mutation_inbox",
    "unlink_mutation_envelope_at",
    "validate_mutation_binding",
    "validate_mutation_request",
    "validate_mutation_result",
    "write_envelope_atomic_at",
)
