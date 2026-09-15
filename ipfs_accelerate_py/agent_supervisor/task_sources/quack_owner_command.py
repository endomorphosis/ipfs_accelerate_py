"""Closed Quack owner-command vocabulary and client.

DatabaseTaskSource and fleet heals mutate through this API. They never open
the exclusive DuckDB file. The exclusive owner maps each command name back
to a canonical IntentRepository method.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .control_plane_contracts import content_identity
from .duckdb_state import (
    DuckDBConnectionPolicyError,
    exclusive_file_lock,
    quack_owner_mutation_dir,
)

STALE_IN_PROGRESS_UNSTALL_SECONDS = 16_200

FALSE_TERMINAL_BLOCKED_REASON_MARKERS = (
    "isolate_merge_queue_to_task_projection",
    "typed_portal_deferral_budget_exhausted",
    "inflight_process_deferral_budget_unstall",
    "implementation_protected_path_mutated",
    "identity_changed",
    "ProcessLookupError",
    "claim_not_accepted_outputs_missing",
    "Portal task projection is not complete",
    "quack_transport_unavailable",
    "grok_quota_exhausted",
    "callback_authority_incomplete_blocked",
    "database_unknown_outcome_blocked",
    "database_task_state_projection_incomplete",
)

QUACK_OWNER_COMMAND_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-command-request@1"
)
QUACK_OWNER_COMMAND_RESPONSE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-command-response@1"
)
QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS = "compare_and_set_status"
QUACK_OWNER_COMMAND_COMPARE_AND_SET_GOAL_STATUS = "compare_and_set_goal_status"
QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK = "rearm_blocked_task"
QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET = (
    "recover_typed_deferral_budget"
)
QUACK_OWNER_COMMAND_RECOVER_LEFTOVER_WAIT_DEFERRAL_BUDGET = (
    "recover_leftover_wait_deferral_budget"
)
QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF = "record_queue_backoff"
QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS = (
    "record_queue_backoff_and_cas_status"
)
QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY = "record_queue_retry"
QUACK_OWNER_COMMAND_RECORD_EVIDENCE = "record_evidence"
QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT = "record_validation_result"
QUACK_OWNER_COMMANDS = frozenset(
    {
        QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
        QUACK_OWNER_COMMAND_COMPARE_AND_SET_GOAL_STATUS,
        QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK,
        QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET,
        QUACK_OWNER_COMMAND_RECOVER_LEFTOVER_WAIT_DEFERRAL_BUDGET,
        QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF,
        QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS,
        QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY,
        QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
        QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT,
    }
)
QUACK_OWNER_COMMAND_MAX_BYTES = 262_144
QUACK_OWNER_COMMAND_MAX_ENVELOPE_BYTES = QUACK_OWNER_COMMAND_MAX_BYTES + 32_768
QUACK_OWNER_COMMAND_MAX_AGE_MS = 300_000
QUACK_OWNER_COMMAND_TIMEOUT_SECONDS = 60.0
_QUACK_OWNER_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_QUACK_OWNER_WRITER_RE = re.compile(r"^supervisor-process:[1-9][0-9]{0,19}$")
_QUACK_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{8,}$")
_QUACK_ATTACH_TOKEN_ENV = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
_GRANT_BROKER_SOCKET_ENV = "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET"
_GRANT_BROKER_SECRET_FD_ENV = "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD"

_QUACK_OWNER_COMMAND_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS: (
        frozenset({"task_cid_or_alias", "expected_revision", "status"}),
        frozenset({"receipt", "expected_control_receipt", "evidence_digests"}),
    ),
    QUACK_OWNER_COMMAND_COMPARE_AND_SET_GOAL_STATUS: (
        frozenset({"goal_cid_or_alias", "expected_revision", "status"}),
        frozenset({"receipt"}),
    ),
    QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK: (
        frozenset({"task_cid_or_alias"}),
        frozenset({"receipt"}),
    ),
    QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET: (
        frozenset({"task_cid_or_alias", "repair_head", "repair_tree"}),
        frozenset(),
    ),
    QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF: (
        frozenset({"task_cid", "delay_ms"}),
        frozenset({"reason", "selection_penalty"}),
    ),
    QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS: (
        frozenset(
            {
                "task_cid",
                "expected_revision",
                "expected_control_receipt",
                "status",
                "receipt",
                "delay_ms",
                "reason",
            }
        ),
        frozenset({"selection_penalty", "exact_retry_not_before_ms"}),
    ),
    QUACK_OWNER_COMMAND_RECOVER_LEFTOVER_WAIT_DEFERRAL_BUDGET: (
        frozenset(
            {
                "task_cid",
                "expected_revision",
                "expected_control_receipt",
                "status",
                "receipt",
                "delay_ms",
                "reason",
            }
        ),
        frozenset({"selection_penalty", "exact_retry_not_before_ms"}),
    ),
    QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY: (
        frozenset({"task_cid"}),
        frozenset(),
    ),
    QUACK_OWNER_COMMAND_RECORD_EVIDENCE: (
        frozenset({"task_cid", "evidence_kind", "digest"}),
        frozenset({"body"}),
    ),
    QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT: (
        frozenset({"task_cid", "outcome", "evidence_digest"}),
        frozenset({"argv", "attempt_id", "body"}),
    ),
}


class QuackOwnerCommandRemoteError(DuckDBConnectionPolicyError):
    """A typed owner command was rejected by the exclusive state owner."""

    def __init__(self, code: str, message: str, *, request_id: str = "") -> None:
        self.code = str(code or "owner_error")
        self.message = str(message or "quack owner command rejected")
        self.request_id = str(request_id or "")
        super().__init__(self.message)


def _owner_command_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise DuckDBConnectionPolicyError(
            f"quack owner command field {field!r} must be a non-empty string"
        )
    if len(value.encode("utf-8")) > 16_384:
        raise DuckDBConnectionPolicyError(
            f"quack owner command field {field!r} exceeds byte bound"
        )
    return value


def validate_quack_owner_command(
    command: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and copy one command from the closed owner vocabulary."""

    if type(command) is not str or command not in QUACK_OWNER_COMMANDS:
        raise DuckDBConnectionPolicyError(
            f"unsupported quack owner command: {command!r}"
        )
    if not isinstance(payload, Mapping):
        raise DuckDBConnectionPolicyError(
            "quack owner command payload must be a mapping"
        )
    required, optional = _QUACK_OWNER_COMMAND_FIELDS[command]
    fields = frozenset(payload)
    if not required.issubset(fields) or not fields.issubset(required | optional):
        raise DuckDBConnectionPolicyError(
            f"quack owner command {command!r} fields do not match its closed schema"
        )
    copied = dict(payload)
    text_fields = {
        "task_cid_or_alias",
        "goal_cid_or_alias",
        "task_cid",
        "status",
        "reason",
        "evidence_kind",
        "digest",
        "outcome",
        "evidence_digest",
        "repair_head",
        "repair_tree",
    }
    for field in text_fields & fields:
        _owner_command_text(copied[field], field=field)
    if "attempt_id" in fields:
        attempt_id = copied["attempt_id"]
        if type(attempt_id) is not str or "\x00" in attempt_id:
            raise DuckDBConnectionPolicyError(
                "quack owner command field 'attempt_id' must be a string"
            )
    for field in {
        "expected_revision",
        "delay_ms",
        "selection_penalty",
        "exact_retry_not_before_ms",
    } & fields:
        value = copied[field]
        if type(value) is not int or value < 0:
            raise DuckDBConnectionPolicyError(
                f"quack owner command field {field!r} must be a non-negative integer"
            )
    for field in {"receipt", "expected_control_receipt", "body"} & fields:
        value = copied[field]
        if value is not None and not isinstance(value, Mapping):
            raise DuckDBConnectionPolicyError(
                f"quack owner command field {field!r} must be a mapping or null"
            )
        if isinstance(value, Mapping):
            copied[field] = dict(value)
    for field in {"argv", "evidence_digests"} & fields:
        value = copied[field]
        if value is None:
            continue
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(
            value, Sequence
        ):
            raise DuckDBConnectionPolicyError(
                f"quack owner command field {field!r} must be a sequence or null"
            )
        items = list(value)
        if len(items) > 4_096 or any(type(item) is not str for item in items):
            raise DuckDBConnectionPolicyError(
                f"quack owner command field {field!r} has invalid items"
            )
        copied[field] = items
    try:
        encoded = json.dumps(
            copied,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DuckDBConnectionPolicyError(
            "quack owner command payload is not canonical JSON"
        ) from exc
    if len(encoded) > QUACK_OWNER_COMMAND_MAX_BYTES:
        raise DuckDBConnectionPolicyError(
            "quack owner command payload exceeds byte bound"
        )
    return copied


def quack_owner_command_signature(
    payload: Mapping[str, Any],
    token: str,
) -> str:
    """Return the HMAC for one exact typed owner-command envelope."""

    secret = str(token or "").strip()
    if not _QUACK_TOKEN_RE.fullmatch(secret):
        raise DuckDBConnectionPolicyError(
            "quack owner command requires the admitted opaque transport token"
        )
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    try:
        encoded = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DuckDBConnectionPolicyError(
            "quack owner command envelope is not canonical JSON"
        ) from exc
    return hmac.new(secret.encode("utf-8"), encoded, hashlib.sha256).hexdigest()


def validate_quack_owner_command_request(
    request: Mapping[str, Any],
    *,
    token: str,
    expected_request_id: str,
    expected_store_id: str,
    expected_store_generation: str,
    now_ms: int | None = None,
    max_age_ms: int = QUACK_OWNER_COMMAND_MAX_AGE_MS,
    allow_expired: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Authenticate and bind one owner-side request before dispatch."""

    if not isinstance(request, Mapping):
        raise DuckDBConnectionPolicyError(
            "quack owner command request must be a mapping"
        )
    expected_fields = {
        "schema",
        "request_id",
        "issued_at_ms",
        "writer_identity",
        "store_id",
        "store_generation",
        "command",
        "payload",
        "signature",
    }
    if set(request) != expected_fields:
        raise DuckDBConnectionPolicyError(
            "quack owner command request fields do not match the closed envelope"
        )
    request_id = str(request.get("request_id") or "")
    if (
        request.get("schema") != QUACK_OWNER_COMMAND_REQUEST_SCHEMA
        or _QUACK_OWNER_REQUEST_ID_RE.fullmatch(request_id) is None
        or request_id != expected_request_id
    ):
        raise DuckDBConnectionPolicyError(
            "quack owner command request identity is invalid"
        )
    issued_at_ms = request.get("issued_at_ms")
    if type(issued_at_ms) is not int:
        raise DuckDBConnectionPolicyError(
            "quack owner command issued_at_ms must be an integer"
        )
    if type(max_age_ms) is not int or not 1 <= max_age_ms <= 300_000:
        raise DuckDBConnectionPolicyError(
            "quack owner command freshness bound is invalid"
        )
    if type(allow_expired) is not bool:
        raise DuckDBConnectionPolicyError(
            "quack owner command allow_expired flag is invalid"
        )
    observed_now = int(time.time() * 1000) if now_ms is None else now_ms
    if type(observed_now) is not int:
        raise DuckDBConnectionPolicyError(
            "quack owner command current time must be an integer"
        )
    age_ms = observed_now - issued_at_ms
    if age_ms < -5_000 or (not allow_expired and age_ms > max_age_ms):
        raise DuckDBConnectionPolicyError(
            "quack owner command request is stale or future-dated"
        )
    if _QUACK_OWNER_WRITER_RE.fullmatch(str(request.get("writer_identity") or "")) is None:
        raise DuckDBConnectionPolicyError(
            "quack owner command writer identity is invalid"
        )
    if (
        request.get("store_id") != expected_store_id
        or request.get("store_generation") != expected_store_generation
        or not expected_store_id
        or not expected_store_generation
    ):
        raise DuckDBConnectionPolicyError(
            "quack owner command store or generation binding is stale"
        )
    observed_signature = str(request.get("signature") or "")
    expected_signature = quack_owner_command_signature(request, token)
    if not hmac.compare_digest(observed_signature, expected_signature):
        raise DuckDBConnectionPolicyError(
            "quack owner command authorization is invalid"
        )
    command = str(request.get("command") or "")
    payload = request.get("payload")
    if not isinstance(payload, Mapping):
        raise DuckDBConnectionPolicyError(
            "quack owner command payload must be a mapping"
        )
    return command, validate_quack_owner_command(command, payload)


def quack_owner_command_response(
    request: Mapping[str, Any],
    *,
    token: str,
    result: Mapping[str, Any] | None = None,
    error_code: str = "",
    error_message: str = "",
) -> dict[str, Any]:
    """Build and owner-sign the exact response consumed by command clients."""

    command = str(request.get("command") or "")
    payload = request.get("payload")
    command_id = content_identity(
        {
            "command": command,
            "payload": dict(payload) if isinstance(payload, Mapping) else {},
        }
    )
    common: dict[str, Any] = {
        "schema": QUACK_OWNER_COMMAND_RESPONSE_SCHEMA,
        "request_id": str(request.get("request_id") or ""),
        "request_signature": str(request.get("signature") or ""),
        "command": command,
        "command_id": command_id,
        "store_id": str(request.get("store_id") or ""),
        "store_generation": str(request.get("store_generation") or ""),
    }
    if result is not None and not error_code and not error_message:
        response = {**common, "ok": True, "result": dict(result)}
    else:
        code = str(error_code or "owner_error")
        message = str(error_message or "typed owner command rejected")
        response = {
            **common,
            "ok": False,
            "error_code": code,
            "error_message": message,
        }
    response["signature"] = quack_owner_command_signature(response, token)
    return response


def quack_owner_command_dir(store_id: object = "") -> Path | None:
    """Return the exclusive owner's local typed-command inbox, if configured."""

    return quack_owner_mutation_dir(store_id)


def quack_owner_mutation_write_lock_path(store_id: object = "") -> Path | None:
    inbox = quack_owner_mutation_dir(store_id)
    return None if inbox is None else inbox.parent / "write-transaction.lock"


def resolve_quack_attach_token(
    token: str = "",
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve the current owner attach token from argument or process env."""

    source = os.environ if environment is None else environment
    explicit = str(token or "").strip()
    if explicit:
        if not _QUACK_TOKEN_RE.fullmatch(explicit):
            raise DuckDBConnectionPolicyError(
                "quack attach token must be an opaque url-safe secret"
            )
        return explicit
    secret = str(source.get(_QUACK_ATTACH_TOKEN_ENV, "") or "").strip()
    if not secret or not _QUACK_TOKEN_RE.fullmatch(secret):
        raise DuckDBConnectionPolicyError(
            "quack attach token must be an opaque url-safe secret"
        )
    return secret


def reset_quack_transport_cache(uri: object = "", *, store_id: object = "") -> None:
    """Evict cached Quack attachments after a successful owner command.

    This overlay does not keep a client ATTACH cache. The function remains
    because native broker clients must still call it after publication.
    """

    endpoint = str(uri or "").strip()
    store = str(store_id or "").strip()
    if endpoint and store:
        raise DuckDBConnectionPolicyError(
            "Quack cache eviction requires one endpoint or store selector"
        )


def _read_quack_owner_command_response(path: Path) -> Mapping[str, Any]:
    """Read one same-UID, bounded, regular response without following links."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise DuckDBConnectionPolicyError(
            "this platform cannot safely open Quack owner command responses"
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | nofollow)
    except OSError as exc:
        raise DuckDBConnectionPolicyError(
            "quack owner command response is not a safe regular file"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_size <= 0
            or metadata.st_size > QUACK_OWNER_COMMAND_MAX_ENVELOPE_BYTES
        ):
            raise DuckDBConnectionPolicyError(
                "quack owner command response owner, type, or size is invalid"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            encoded = handle.read(QUACK_OWNER_COMMAND_MAX_ENVELOPE_BYTES + 1)
        if len(encoded) != metadata.st_size:
            raise DuckDBConnectionPolicyError(
                "quack owner command response changed while being read"
            )
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DuckDBConnectionPolicyError(
                "quack owner command response is not valid JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise DuckDBConnectionPolicyError(
                "quack owner command response must be a mapping"
            )
        return payload
    finally:
        os.close(descriptor)


def submit_quack_owner_command(
    command: str,
    payload: Mapping[str, Any],
    *,
    timeout_seconds: float = QUACK_OWNER_COMMAND_TIMEOUT_SECONDS,
    request_id: str = "",
) -> Mapping[str, Any]:
    """Submit one typed command and return its typed result mapping.

    A configured owner broker is authoritative and issues one peer-bound
    socket session. Legacy launchers without broker bindings retain the
    signed filesystem rendezvous; native denial never selects that fallback.
    """

    command_name = str(command or "")
    command_payload = validate_quack_owner_command(command_name, payload)
    maximum_timeout = (
        660.0
        if command_name == QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET
        else 60.0
    )
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not 0 < float(timeout_seconds) <= maximum_timeout
    ):
        raise DuckDBConnectionPolicyError(
            "quack owner command timeout exceeds its closed command bound"
        )
    requested_id = str(request_id or "").strip()
    if requested_id and _QUACK_OWNER_REQUEST_ID_RE.fullmatch(requested_id) is None:
        raise DuckDBConnectionPolicyError(
            "quack owner command request_id must be 32 lowercase hexadecimal characters"
        )
    request_id = requested_id or uuid.uuid4().hex
    broker_socket = str(os.environ.get(_GRANT_BROKER_SOCKET_ENV, "") or "").strip()
    broker_descriptor = str(
        os.environ.get(_GRANT_BROKER_SECRET_FD_ENV, "") or ""
    ).strip()
    if broker_socket or broker_descriptor:
        if not broker_socket or not broker_descriptor:
            raise DuckDBConnectionPolicyError(
                "Quack command credential broker binding is incomplete"
            )
        store_id = str(
            os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "") or ""
        ).strip()
        if not store_id:
            raise DuckDBConnectionPolicyError(
                "Quack command credential broker lacks an exact store binding"
            )
        from .typed_state_owner import (
            TypedStateOwnerConnection,
            TypedStateOwnerDatabaseTaskOutcomeUnknownError,
            TypedStateOwnerError,
            TypedStateOwnerRemoteError,
            kernel_process_birth_id,
            request_database_task_command_credential,
            typed_owner_socket_path,
        )

        write_lock = quack_owner_mutation_write_lock_path(store_id)
        if write_lock is None:
            raise DuckDBConnectionPolicyError(
                "typed owner command has no accepted-root replica lock path"
            )
        with exclusive_file_lock(write_lock, timeout_seconds=float(timeout_seconds)):
            client_id = f"database-task-source:{os.getpid()}"
            process_birth_id = kernel_process_birth_id()
            connection = None
            try:
                grant = request_database_task_command_credential(
                    store_id=store_id,
                    client_id=client_id,
                    process_birth_id=process_birth_id,
                    timeout_seconds=min(float(timeout_seconds), 30.0),
                )
                connection = TypedStateOwnerConnection(
                    socket_path=typed_owner_socket_path(store_id),
                    token=grant,
                    client_id=client_id,
                    process_birth_id=process_birth_id,
                    store_id=store_id,
                    timeout_seconds=float(timeout_seconds),
                )
                try:
                    result = connection.execute_database_task_command(
                        command_name,
                        command_payload,
                        command_request_id=request_id,
                    )
                finally:
                    connection.close()
            except TypedStateOwnerRemoteError as exc:
                raise QuackOwnerCommandRemoteError(
                    exc.error_code,
                    "typed owner command rejected",
                    request_id=request_id,
                ) from exc
            except TypedStateOwnerDatabaseTaskOutcomeUnknownError as exc:
                raise QuackOwnerCommandRemoteError(
                    "unknown_external_outcome",
                    "typed owner command outcome requires reconciliation",
                    request_id=request_id,
                ) from exc
            except (OSError, TypedStateOwnerError) as exc:
                raise DuckDBConnectionPolicyError(
                    "typed owner command transport failed closed"
                ) from exc
            if not isinstance(result, Mapping):
                raise QuackOwnerCommandRemoteError(
                    "unknown_external_outcome",
                    "typed owner command returned no admissible result",
                    request_id=request_id,
                )
            from . import duckdb_state as state

            state.reset_quack_transport_cache(store_id=store_id)
            return dict(result)
    from . import duckdb_state as state

    target = state.quack_owner_command_dir()
    if target is None:
        raise DuckDBConnectionPolicyError(
            "quack ATTACH cannot mutate remote base tables; set "
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR or "
            "IPFS_ACCELERATE_AGENT_STATE_STORE_ID so the state-owner can "
            "apply a typed command"
        )
    target.mkdir(parents=True, exist_ok=True)
    os.chmod(target, 0o700)
    store_id = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "") or ""
    ).strip()
    store_generation = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "") or ""
    ).strip()
    if not store_id or not store_generation:
        raise DuckDBConnectionPolicyError(
            "quack owner command requires exact store and generation bindings"
        )
    request_path = target / f"{request_id}.request.json"
    done_path = target / f"{request_id}.done.json"
    token = state.resolve_quack_attach_token()
    request_payload: dict[str, Any] = {
        "schema": QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
        "request_id": request_id,
        "issued_at_ms": int(time.time() * 1000),
        "writer_identity": f"supervisor-process:{os.getpid()}",
        "store_id": store_id,
        "store_generation": store_generation,
        "command": command_name,
        "payload": command_payload,
    }
    request_payload["signature"] = quack_owner_command_signature(
        request_payload,
        token,
    )
    expected_command_id = content_identity(
        {"command": command_name, "payload": command_payload}
    )
    encoded_request = (
        json.dumps(
            request_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded_request) > QUACK_OWNER_COMMAND_MAX_ENVELOPE_BYTES:
        raise DuckDBConnectionPolicyError(
            "quack owner command request envelope exceeds its byte bound"
        )
    temporary_path = target / f".{request_id}.{uuid.uuid4().hex}.request.tmp"
    descriptor = os.open(
        temporary_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded_request)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, request_path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    deadline = time.monotonic() + float(timeout_seconds)
    last_response_error: DuckDBConnectionPolicyError | None = None
    while time.monotonic() < deadline:
        if done_path.is_file():
            try:
                response = _read_quack_owner_command_response(done_path)
                common = {
                    "schema",
                    "request_id",
                    "request_signature",
                    "command",
                    "command_id",
                    "store_id",
                    "store_generation",
                    "ok",
                    "signature",
                }
                variant = (
                    {"result"}
                    if response.get("ok") is True
                    else {"error_code", "error_message"}
                )
                if (
                    set(response) != common | variant
                    or response.get("schema") != QUACK_OWNER_COMMAND_RESPONSE_SCHEMA
                    or response.get("request_id") != request_id
                    or response.get("command") != command_name
                    or response.get("command_id") != expected_command_id
                    or response.get("store_id") != request_payload["store_id"]
                    or response.get("store_generation")
                    != request_payload["store_generation"]
                    or not hmac.compare_digest(
                        str(response.get("request_signature") or ""),
                        str(request_payload["signature"]),
                    )
                    or type(response.get("ok")) is not bool
                ):
                    raise DuckDBConnectionPolicyError(
                        "quack owner command response authorization binding is invalid"
                    )
                observed_signature = str(response.get("signature") or "")
                expected_signature = quack_owner_command_signature(response, token)
                if not hmac.compare_digest(observed_signature, expected_signature):
                    raise DuckDBConnectionPolicyError(
                        "quack owner command response authorization is invalid"
                    )
                try:
                    request_path.unlink(missing_ok=True)
                    done_path.unlink(missing_ok=True)
                except OSError:
                    pass
                if response["ok"] is not True:
                    raise QuackOwnerCommandRemoteError(
                        str(response.get("error_code") or "owner_error"),
                        str(response.get("error_message") or "owner command rejected"),
                        request_id=request_id,
                    )
                result = response.get("result")
                if not isinstance(result, Mapping):
                    raise DuckDBConnectionPolicyError(
                        "quack owner command result must be a mapping"
                    )
                return dict(result)
            except QuackOwnerCommandRemoteError:
                raise
            except DuckDBConnectionPolicyError as exc:
                last_response_error = exc
        time.sleep(0.05)
    unknown_outcome = QuackOwnerCommandRemoteError(
        "command_timeout_unknown_outcome",
        "timed out waiting for quack state-owner to reconcile typed command",
        request_id=request_id,
    )
    if last_response_error is not None:
        raise unknown_outcome from last_response_error
    raise unknown_outcome
