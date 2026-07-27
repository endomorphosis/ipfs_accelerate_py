"""Provider-free canonical contracts for prompt-driven supervisor workflows.

This module is intentionally a leaf module.  It defines immutable data,
validation, canonical serialization, and content identities; it does not scan
a checkout, import a model/provider, open DuckDB, construct a graph runtime, or
start a supervisor.  Live integrations consume these contracts lazily.

Prompt bodies are transient inputs.  A :class:`PromptSource` serializes only a
CID, byte count, source kind, redacted metadata, and (where applicable) a
bounded path or opaque artifact handle.  Consequently prompt text and secrets
cannot accidentally become durable workflow receipts.
"""

from __future__ import annotations

import base64
import hashlib
import json
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final


PROMPT_WORKFLOW_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = PROMPT_WORKFLOW_CONTRACT_VERSION
SCHEMA_VERSION: Final[int] = PROMPT_WORKFLOW_CONTRACT_VERSION

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
PROMPT_SOURCE_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/prompt-source@1"
WORKFLOW_BUDGET_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/prompt-workflow-budget@1"
SCAN_POLICY_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/directory-scan-policy@1"
PLANNING_POLICY_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/prompt-planning-policy@1"
OUTPUT_POLICY_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/prompt-output-policy@1"
PROMPT_WORKFLOW_REQUEST_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/prompt-workflow-request@1"
)
PROMPT_EVIDENCE_RECORD_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/prompt-evidence-record@1"
)
PROMPT_ACCEPTANCE_RECORD_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/prompt-acceptance-record@1"
)
PROMPT_VALIDATION_RECORD_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/prompt-validation-record@1"
)
PROMPT_OUTPUT_RECORD_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/prompt-output-record@1"
)
PROMPT_GOAL_RECORD_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/prompt-goal-record@1"
PROMPT_TASK_RECORD_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/prompt-task-record@1"
DIRECTORY_SCAN_RECEIPT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/directory-scan-receipt@1"
)
PROMPT_GOAL_GRAPH_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/prompt-goal-graph@1"
MATERIALIZATION_REFERENCE_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/prompt-materialization-reference@1"
)
SUPERVISOR_RUN_REFERENCE_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/supervisor-run-reference@1"
)
SUPERVISOR_INCIDENT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/supervisor-incident@1"
)
RECOVERY_ATTEMPT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/recovery-attempt@1"
PROGRAMMATIC_RECOVERY_EXHAUSTION_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/programmatic-recovery-exhaustion-receipt@1"
)
RESCUE_ACTION_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/rescue-action@1"
RESCUE_PLAN_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/rescue-plan@1"
PROMPT_WORKFLOW_PREVIEW_RECEIPT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/prompt-workflow-preview-receipt@1"
)
PROMPT_WORKFLOW_RESULT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/prompt-workflow-result@1"
)

ABSOLUTE_MAX_CONTRACT_BYTES: Final[int] = 1_048_576
ABSOLUTE_MAX_PROMPT_BYTES: Final[int] = 4 * 1024 * 1024
ABSOLUTE_MAX_SCAN_BYTES: Final[int] = 512 * 1024 * 1024
ABSOLUTE_MAX_TEXT_BYTES: Final[int] = 65_536
ABSOLUTE_MAX_ITEMS: Final[int] = 4_096
ABSOLUTE_MAX_DEPTH: Final[int] = 32
ABSOLUTE_MAX_FILES: Final[int] = 100_000
ABSOLUTE_MAX_GOALS: Final[int] = 1_024
ABSOLUTE_MAX_TASKS: Final[int] = 4_096
ABSOLUTE_MAX_EVIDENCE: Final[int] = 4_096
ABSOLUTE_MAX_RESCUE_ACTIONS: Final[int] = 32
ABSOLUTE_MAX_LATENCY_MS: Final[int] = 86_400_000

_CID_PREFIX = b"\x01\xa9\x02\x12\x20"  # CIDv1, dag-json, sha2-256
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CID_RE = re.compile(r"^b[a-z2-7]+$")
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
)
_SECRET_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "prompt",
        "prompt_body",
        "prompt_text",
        "raw_log",
        "refresh_token",
        "secret",
        "session_token",
        "source_body",
        "source_text",
        "token",
    }
)
_VOLATILE_FIELDS = frozenset(
    {
        "status",
        "created_at_ms",
        "updated_at_ms",
        "observed_at_ms",
        "started_at_ms",
        "finished_at_ms",
    }
)


class PromptWorkflowContractError(ValueError):
    """Base error for malformed prompt-workflow contracts."""


class PromptWorkflowBoundsError(PromptWorkflowContractError):
    """A count, byte, depth, or time limit exceeds a hard bound."""


class PromptWorkflowIdentityError(PromptWorkflowContractError):
    """A claimed content identity does not match canonical identity bytes."""


class PromptWorkflowPathError(PromptWorkflowContractError):
    """A root or repository-relative path is non-canonical or escapes scope."""


class PromptSourceError(PromptWorkflowContractError):
    """A prompt source is ambiguous, unsafe, or incomplete."""


class PromptSecretError(PromptWorkflowContractError):
    """A durable contract contains inline secret-bearing material."""


class PromptGraphError(PromptWorkflowContractError):
    """A goal/task graph is disconnected, cyclic, or inconsistent."""


class RescuePlanError(PromptWorkflowContractError):
    """A rescue plan is open-ended, unbound, or otherwise unsafe."""


class NonCanonicalPromptWorkflowError(PromptWorkflowContractError):
    """Serialized input is not the exact canonical representation."""


class PromptSourceKind(str, Enum):
    INLINE = "inline"
    FILE = "file"
    STDIN = "stdin"
    ARTIFACT = "artifact"


class OutputMode(str, Enum):
    MARKDOWN = "markdown"
    DUCKDB = "duckdb"
    BOTH = "both"


class LocalFallbackPolicy(str, Enum):
    REQUIRED = "required"
    ALLOWED = "allowed"
    DISABLED = "disabled"


class RecordStatus(str, Enum):
    PROPOSED = "proposed"
    ADMITTED = "admitted"
    REJECTED = "rejected"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class EvidenceAuthority(str, Enum):
    PROMPT = "prompt"
    SCAN_ADVISORY = "scan_advisory"
    VERIFIED = "verified"
    AUTHORITATIVE = "authoritative"


class WorkflowOutcome(str, Enum):
    PREVIEWED = "previewed"
    MATERIALIZED = "materialized"
    STARTED = "started"
    PARTIAL = "partial"
    REJECTED = "rejected"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class RecoveryAttemptOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INAPPLICABLE = "inapplicable"
    DENIED = "denied"
    TIMED_OUT = "timed_out"


class IncidentKind(str, Enum):
    STALE_PROJECTION = "stale_projection"
    STALE_LIFECYCLE = "stale_lifecycle"
    STALE_HEARTBEAT = "stale_heartbeat"
    STALE_LEASE = "stale_lease"
    ORPHANED_LOCK = "orphaned_lock"
    CONSUMED_ATTEMPT = "consumed_attempt"
    LANE_FAILURE = "lane_failure"
    DIRTY_WORKTREE = "dirty_worktree"
    VALIDATION_FAILURE = "validation_failure"
    MERGE_FAILURE = "merge_failure"
    CORRUPT_TASK_SOURCE = "corrupt_task_source"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    SPLIT_BRAIN = "split_brain"
    UNKNOWN = "unknown"


class RescueOperation(str, Enum):
    """Closed operation vocabulary selectable by a rescue proposal."""

    STATUS = "status"
    HEALTH = "health"
    EVENTS = "events"
    RECONCILE_PROJECTION = "reconcile_projection"
    REPAIR_LIFECYCLE_STATE = "repair_lifecycle_state"
    REPAIR_EXPIRED_LEASE = "repair_expired_lease"
    REPAIR_ORPHANED_LOCK = "repair_orphaned_lock"
    RETRY = "retry"
    RESTART_LANE = "restart_lane"
    RESTART = "restart"
    VALIDATION_REPLAY = "validation_replay"
    RESCUE_DIRTY_WORK = "rescue_dirty_work"
    RECONCILE_WORKTREE = "reconcile_worktree"
    QUARANTINE = "quarantine"
    REASSIGN_INDEPENDENT_WORK = "reassign_independent_work"
    OBJECTIVE_RECONCILE = "objective_reconcile"
    BACKLOG_REFILL = "backlog_refill"
    PAUSE = "pause"
    RESUME = "resume"
    DRAIN = "drain"
    STOP = "stop"


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted({str(member.value) for member in kind}))
        raise PromptWorkflowContractError(
            f"{name} must be one of: {allowed}"
        ) from exc


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = ABSOLUTE_MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise PromptWorkflowContractError(f"{name} must be a string")
    if value != value.strip():
        raise NonCanonicalPromptWorkflowError(
            f"{name} has leading or trailing whitespace"
        )
    if required and not value:
        raise PromptWorkflowContractError(f"{name} must not be empty")
    if "\x00" in value:
        raise PromptWorkflowContractError(f"{name} must not contain NUL")
    if len(value.encode("utf-8")) > maximum:
        raise PromptWorkflowBoundsError(f"{name} exceeds {maximum} UTF-8 bytes")
    if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
        raise PromptSecretError(f"{name} contains inline secret material")
    return value


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PromptWorkflowContractError(f"{name} must be a finite integer")
    if value < minimum:
        raise PromptWorkflowContractError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise PromptWorkflowBoundsError(f"{name} exceeds its absolute limit")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PromptWorkflowContractError(f"{name} must be boolean")
    return value


def _absolute_path(value: Any, name: str) -> str:
    result = _text(value, name)
    if "\\" in result or not result.startswith("/"):
        raise PromptWorkflowPathError(f"{name} must be a canonical absolute path")
    normalized = posixpath.normpath(result)
    if normalized == "/":
        raise PromptWorkflowPathError(f"{name} must not be the filesystem root")
    if normalized != result or ".." in PurePosixPath(result).parts:
        raise NonCanonicalPromptWorkflowError(f"{name} is not canonical")
    return result


def _relative_path(value: Any, name: str, *, allow_empty: bool = False) -> str:
    result = _text(value, name, required=not allow_empty)
    if allow_empty and not result:
        return ""
    candidate = PurePosixPath(result)
    if (
        "\\" in result
        or candidate.is_absolute()
        or ".." in candidate.parts
        or (candidate.parts and candidate.parts[0].endswith(":"))
    ):
        raise PromptWorkflowPathError(f"{name} must be repository-relative")
    normalized = candidate.as_posix()
    if normalized in ("", ".") or normalized != result:
        raise NonCanonicalPromptWorkflowError(f"{name} is not canonical")
    return result


def _is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _content_digest_bytes(value: bytes) -> str:
    raw = _CID_PREFIX + hashlib.sha256(value).digest()
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def canonical_prompt_workflow_bytes(value: Any) -> bytes:
    """Encode finite JSON using the one workflow canonical representation."""

    try:
        encoded = json.dumps(
            _wire_value(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PromptWorkflowContractError(
            "value is not canonical prompt-workflow JSON"
        ) from exc
    if len(encoded) > ABSOLUTE_MAX_CONTRACT_BYTES:
        raise PromptWorkflowBoundsError("canonical contract exceeds byte bound")
    return encoded


def prompt_workflow_cid(value: Any) -> str:
    return _content_digest_bytes(canonical_prompt_workflow_bytes(value))


def _validate_cid(value: Any, name: str, *, required: bool = True) -> str:
    result = _text(value, name, required=required)
    if not result and not required:
        return ""
    if not _CID_RE.fullmatch(result):
        raise PromptWorkflowIdentityError(f"{name} must be a canonical CIDv1")
    try:
        padding = "=" * ((8 - (len(result) - 1) % 8) % 8)
        decoded = base64.b32decode((result[1:].upper() + padding).encode("ascii"))
    except (ValueError, UnicodeError) as exc:
        raise PromptWorkflowIdentityError(f"{name} is malformed") from exc
    if len(decoded) != len(_CID_PREFIX) + 32 or not decoded.startswith(_CID_PREFIX):
        raise PromptWorkflowIdentityError(
            f"{name} must use CIDv1 dag-json with sha2-256"
        )
    canonical = "b" + base64.b32encode(decoded).decode("ascii").rstrip("=").lower()
    if canonical != result:
        raise PromptWorkflowIdentityError(f"{name} is not canonical")
    return result


def _identity(value: Any, name: str, *, required: bool = True) -> str:
    result = _text(value, name, required=required)
    if not result and not required:
        return ""
    if result.startswith("b"):
        return _validate_cid(result, name)
    if _DIGEST_RE.fullmatch(result):
        return result
    raise PromptWorkflowIdentityError(
        f"{name} must be a CIDv1 or sha256:<64 lowercase hex>"
    )


def _strings(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = ABSOLUTE_MAX_ITEMS,
    sort: bool = True,
    paths: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PromptWorkflowContractError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise PromptWorkflowBoundsError(f"{name} exceeds its count bound")
    result = tuple(
        _relative_path(item, name) if paths else _text(item, name)
        for item in values
    )
    if required and not result:
        raise PromptWorkflowContractError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise PromptWorkflowContractError(f"{name} contains duplicates")
    return tuple(sorted(result)) if sort else result


def _secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SECRET_KEYS or any(
        marker in normalized
        for marker in ("password", "private_key", "access_token", "api_key")
    )


def _freeze_json(
    value: Any,
    name: str,
    *,
    reject_secrets: bool = True,
    max_items: int = ABSOLUTE_MAX_ITEMS,
    max_depth: int = ABSOLUTE_MAX_DEPTH,
) -> Any:
    seen = 0

    def visit(item: Any, depth: int, key_name: str = "") -> Any:
        nonlocal seen
        seen += 1
        if seen > max_items:
            raise PromptWorkflowBoundsError(f"{name} exceeds item-count bound")
        if depth > max_depth:
            raise PromptWorkflowBoundsError(f"{name} exceeds depth bound")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            raise PromptWorkflowContractError(f"{name} must not contain floats")
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, str):
            result = _text(item, name, required=False)
            if reject_secrets and any(pattern.search(result) for pattern in _SECRET_VALUE_PATTERNS):
                raise PromptSecretError(f"{name} contains inline secret material")
            return result
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key in sorted(item):
                normalized = _text(key, f"{name} key")
                if reject_secrets and _secret_key(normalized):
                    raise PromptSecretError(
                        f"{name} contains forbidden secret-bearing field"
                    )
                result[normalized] = visit(item[key], depth + 1, normalized)
            return MappingProxyType(result)
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return tuple(visit(member, depth + 1, key_name) for member in item)
        raise PromptWorkflowContractError(
            f"{name} contains unsupported type {type(item).__name__}"
        )

    return visit(value, 0)


def _wire_value(value: Any) -> Any:
    if isinstance(value, _WorkflowContract):
        return value.to_record()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _wire_value(member) for key, member in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_wire_value(member) for member in value]
    return value


def _decode_json_object(payload: str, noun: str) -> Mapping[str, Any]:
    if not isinstance(payload, str):
        raise PromptWorkflowContractError(f"{noun} JSON must be text")
    if len(payload.encode("utf-8")) > ABSOLUTE_MAX_CONTRACT_BYTES:
        raise PromptWorkflowBoundsError(
            f"{noun} JSON exceeds the serialized byte bound"
        )

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise NonCanonicalPromptWorkflowError(
                    f"{noun} JSON contains duplicate keys"
                )
            result[key] = value
        return result

    try:
        result = json.loads(payload, object_pairs_hook=pairs_hook)
    except NonCanonicalPromptWorkflowError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise PromptWorkflowContractError(f"{noun} JSON is malformed") from exc
    if not isinstance(result, Mapping):
        raise PromptWorkflowContractError(f"{noun} JSON must contain an object")
    if canonical_prompt_workflow_bytes(result) != payload.encode("utf-8"):
        raise NonCanonicalPromptWorkflowError(
            f"{noun} JSON changes during canonical round trip"
        )
    return result


class _WorkflowContract:
    SCHEMA: ClassVar[str] = ""
    FIELDS: ClassVar[tuple[str, ...]] = ()
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = frozenset()
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {}
    NESTED_FIELDS: ClassVar[Mapping[str, type["_WorkflowContract"]]] = {}
    NESTED_SEQUENCES: ClassVar[Mapping[str, type["_WorkflowContract"]]] = {}

    @property
    def schema(self) -> str:
        return self.SCHEMA

    @property
    def contract_version(self) -> int:
        return PROMPT_WORKFLOW_CONTRACT_VERSION

    def _payload(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.FIELDS}

    def to_dict(self) -> dict[str, Any]:
        return _wire_value(
            {
                "schema": self.SCHEMA,
                "contract_version": PROMPT_WORKFLOW_CONTRACT_VERSION,
                **self._payload(),
            }
        )

    def _identity_payload(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in self.to_dict().items()
            if key not in self.IDENTITY_EXCLUDED
        }

        # Lifecycle fields can occur inside embedded goal/task/evidence,
        # materialization, and run records.  Strip them recursively so a plan
        # root or saga identity cannot drift merely because a nested record was
        # observed later or moved through a mutable status.
        def semantic(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    key: semantic(member)
                    for key, member in value.items()
                    if key not in _VOLATILE_FIELDS
                }
            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray, memoryview)
            ):
                return [semantic(member) for member in value]
            return value

        return semantic(payload)

    @property
    def content_id(self) -> str:
        return prompt_workflow_cid(self._identity_payload())

    @property
    def cid(self) -> str:
        return self.content_id

    @property
    def canonical_id(self) -> str:
        return self.content_id

    @property
    def identity(self) -> str:
        return self.content_id

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    def canonical_bytes(self) -> bytes:
        return canonical_prompt_workflow_bytes(self.to_dict())

    def to_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    canonical_json = to_json

    @classmethod
    def _decode_field(cls, name: str, value: Any) -> Any:
        if name in cls.ENUM_FIELDS:
            return _enum(value, cls.ENUM_FIELDS[name], name)
        nested = cls.NESTED_FIELDS.get(name)
        if nested is not None:
            if not isinstance(value, Mapping):
                raise PromptWorkflowContractError(f"{name} must be an object")
            return nested.from_dict(value)
        nested_item = cls.NESTED_SEQUENCES.get(name)
        if nested_item is not None:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise PromptWorkflowContractError(f"{name} must be a sequence")
            return tuple(nested_item.from_dict(member) for member in value)
        return value

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_WorkflowContract":
        if not isinstance(payload, Mapping):
            raise PromptWorkflowContractError(f"{cls.__name__} must be an object")
        allowed = {"schema", "contract_version", "content_id", *cls.FIELDS}
        unknown = set(payload).difference(allowed)
        if unknown:
            raise PromptWorkflowContractError(
                f"{cls.__name__} contains unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        if payload.get("schema") != cls.SCHEMA:
            raise PromptWorkflowContractError(
                f"{cls.__name__} requires exact schema {cls.SCHEMA!r}"
            )
        if payload.get("contract_version") != PROMPT_WORKFLOW_CONTRACT_VERSION:
            raise PromptWorkflowContractError(
                f"{cls.__name__} requires contract_version "
                f"{PROMPT_WORKFLOW_CONTRACT_VERSION}"
            )
        missing = [name for name in cls.FIELDS if name not in payload]
        if missing:
            raise PromptWorkflowContractError(
                f"{cls.__name__} is missing required field {missing[0]}"
            )
        result = cls(
            **{
                name: cls._decode_field(name, payload[name])
                for name in cls.FIELDS
            }
        )
        claimed = payload.get("content_id")
        if claimed is not None and claimed != result.content_id:
            raise PromptWorkflowIdentityError(
                f"{cls.__name__} content identity does not match"
            )
        return result

    @classmethod
    def from_json(cls, payload: str) -> "_WorkflowContract":
        result = cls.from_dict(_decode_json_object(payload, cls.__name__))
        if result.to_json() != payload:
            raise NonCanonicalPromptWorkflowError(
                f"{cls.__name__} changed during canonical round trip"
            )
        return result


@dataclass(frozen=True)
class PromptSource(_WorkflowContract):
    """Body-free, unambiguous descriptor for exactly one prompt source."""

    SCHEMA: ClassVar[str] = PROMPT_SOURCE_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "kind",
        "prompt_cid",
        "byte_count",
        "redacted_metadata",
        "source_path",
        "artifact_handle",
    )
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {"kind": PromptSourceKind}

    kind: PromptSourceKind
    prompt_cid: str
    byte_count: int
    redacted_metadata: Mapping[str, Any] = field(default_factory=dict)
    source_path: str = ""
    artifact_handle: str = ""
    _transient_body: bytes | None = field(
        default=None, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, PromptSourceKind, "kind"))
        object.__setattr__(
            self, "prompt_cid", _validate_cid(self.prompt_cid, "prompt_cid")
        )
        object.__setattr__(
            self,
            "byte_count",
            _integer(
                self.byte_count,
                "byte_count",
                minimum=1,
                maximum=ABSOLUTE_MAX_PROMPT_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "redacted_metadata",
            _freeze_json(self.redacted_metadata, "redacted_metadata"),
        )
        object.__setattr__(
            self,
            "source_path",
            _relative_path(self.source_path, "source_path", allow_empty=True),
        )
        object.__setattr__(
            self,
            "artifact_handle",
            _text(self.artifact_handle, "artifact_handle", required=False),
        )
        if self.kind is PromptSourceKind.FILE:
            if not self.source_path or self.artifact_handle:
                raise PromptSourceError(
                    "file prompt source requires only source_path"
                )
        elif self.kind is PromptSourceKind.ARTIFACT:
            if not self.artifact_handle or self.source_path:
                raise PromptSourceError(
                    "artifact prompt source requires only artifact_handle"
                )
        elif self.source_path or self.artifact_handle:
            raise PromptSourceError(
                "inline/stdin prompt source cannot carry path or artifact handle"
            )
        if self._transient_body is not None:
            if not isinstance(self._transient_body, bytes):
                raise PromptSourceError("transient prompt body must be bytes")
            if len(self._transient_body) != self.byte_count:
                raise PromptSourceError("prompt byte count does not match body")
            if _content_digest_bytes(
                canonical_prompt_workflow_bytes(
                    {"media_type": "text/plain; charset=utf-8", "body_sha256": hashlib.sha256(self._transient_body).hexdigest()}
                )
            ) != self.prompt_cid:
                raise PromptSourceError("prompt CID does not match body")
            try:
                prompt_text = self._transient_body.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PromptSourceError("prompt body must be UTF-8 text") from exc

            def metadata_strings(value: Any) -> tuple[str, ...]:
                if isinstance(value, str):
                    return (value,)
                if isinstance(value, Mapping):
                    return tuple(
                        member
                        for item in value.values()
                        for member in metadata_strings(item)
                    )
                if isinstance(value, Sequence) and not isinstance(
                    value, (str, bytes, bytearray, memoryview)
                ):
                    return tuple(
                        member
                        for item in value
                        for member in metadata_strings(item)
                    )
                return ()

            if prompt_text in metadata_strings(self.redacted_metadata):
                raise PromptSourceError(
                    "redacted_metadata must not reproduce the raw prompt"
                )

    @staticmethod
    def _cid_for_body(body: bytes) -> str:
        return _content_digest_bytes(
            canonical_prompt_workflow_bytes(
                {
                    "media_type": "text/plain; charset=utf-8",
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                }
            )
        )

    @classmethod
    def inline(
        cls, text: str, *, redacted_metadata: Mapping[str, Any] | None = None
    ) -> "PromptSource":
        if not isinstance(text, str) or not text:
            raise PromptSourceError("inline prompt text must be non-empty text")
        body = text.encode("utf-8")
        return cls(
            kind=PromptSourceKind.INLINE,
            prompt_cid=cls._cid_for_body(body),
            byte_count=len(body),
            redacted_metadata=redacted_metadata or {},
            _transient_body=body,
        )

    @classmethod
    def stdin(
        cls, text: str, *, redacted_metadata: Mapping[str, Any] | None = None
    ) -> "PromptSource":
        source = cls.inline(text, redacted_metadata=redacted_metadata)
        return cls(
            kind=PromptSourceKind.STDIN,
            prompt_cid=source.prompt_cid,
            byte_count=source.byte_count,
            redacted_metadata=source.redacted_metadata,
            _transient_body=source._transient_body,
        )

    @classmethod
    def file(
        cls,
        path: str,
        *,
        text: str | None = None,
        prompt_cid: str = "",
        byte_count: int | None = None,
        redacted_metadata: Mapping[str, Any] | None = None,
    ) -> "PromptSource":
        body = None if text is None else text.encode("utf-8")
        if body is not None:
            prompt_cid = cls._cid_for_body(body)
            byte_count = len(body)
        if byte_count is None:
            raise PromptSourceError("file source requires text or byte_count")
        return cls(
            kind=PromptSourceKind.FILE,
            prompt_cid=prompt_cid,
            byte_count=byte_count,
            redacted_metadata=redacted_metadata or {},
            source_path=path,
            _transient_body=body,
        )

    @classmethod
    def artifact(
        cls,
        handle: str,
        *,
        prompt_cid: str,
        byte_count: int,
        redacted_metadata: Mapping[str, Any] | None = None,
    ) -> "PromptSource":
        return cls(
            kind=PromptSourceKind.ARTIFACT,
            prompt_cid=prompt_cid,
            byte_count=byte_count,
            redacted_metadata=redacted_metadata or {},
            artifact_handle=handle,
        )

    @property
    def transient_body(self) -> bytes | None:
        """Return process-local prompt bytes; never part of a record or identity."""

        return self._transient_body


@dataclass(frozen=True)
class PromptWorkflowBudget(_WorkflowContract):
    SCHEMA: ClassVar[str] = WORKFLOW_BUDGET_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "max_files",
        "max_scan_bytes",
        "max_file_bytes",
        "max_symbols",
        "max_prompt_tokens",
        "max_provider_tokens",
        "max_latency_ms",
        "max_goals",
        "max_tasks",
        "max_evidence",
        "max_graph_depth",
        "max_serialized_bytes",
        "max_rescue_actions",
    )

    max_files: int = 10_000
    max_scan_bytes: int = 64 * 1024 * 1024
    max_file_bytes: int = 2 * 1024 * 1024
    max_symbols: int = 50_000
    max_prompt_tokens: int = 16_384
    max_provider_tokens: int = 32_768
    max_latency_ms: int = 300_000
    max_goals: int = 128
    max_tasks: int = 512
    max_evidence: int = 1_024
    max_graph_depth: int = 16
    max_serialized_bytes: int = 512 * 1024
    max_rescue_actions: int = 8

    def __post_init__(self) -> None:
        limits = {
            "max_files": ABSOLUTE_MAX_FILES,
            "max_scan_bytes": ABSOLUTE_MAX_SCAN_BYTES,
            "max_file_bytes": ABSOLUTE_MAX_PROMPT_BYTES,
            "max_symbols": ABSOLUTE_MAX_FILES * 10,
            "max_prompt_tokens": 1_000_000,
            "max_provider_tokens": 1_000_000,
            "max_latency_ms": ABSOLUTE_MAX_LATENCY_MS,
            "max_goals": ABSOLUTE_MAX_GOALS,
            "max_tasks": ABSOLUTE_MAX_TASKS,
            "max_evidence": ABSOLUTE_MAX_EVIDENCE,
            "max_graph_depth": ABSOLUTE_MAX_DEPTH,
            "max_serialized_bytes": ABSOLUTE_MAX_CONTRACT_BYTES,
            "max_rescue_actions": ABSOLUTE_MAX_RESCUE_ACTIONS,
        }
        for name, maximum in limits.items():
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=1, maximum=maximum),
            )
        if self.max_file_bytes > self.max_scan_bytes:
            raise PromptWorkflowBoundsError(
                "max_file_bytes cannot exceed max_scan_bytes"
            )


@dataclass(frozen=True)
class DirectoryScanPolicy(_WorkflowContract):
    SCHEMA: ClassVar[str] = SCAN_POLICY_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "policy_id",
        "scanner_version",
        "include_patterns",
        "exclude_patterns",
        "include_untracked",
        "reject_symlinks",
        "reject_nested_repositories",
        "exclude_credentials",
        "exclude_generated_state",
    )

    policy_id: str
    scanner_version: str
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    include_untracked: bool = True
    reject_symlinks: bool = True
    reject_nested_repositories: bool = True
    exclude_credentials: bool = True
    exclude_generated_state: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "scanner_version", _text(self.scanner_version, "scanner_version")
        )
        for name in ("include_patterns", "exclude_patterns"):
            object.__setattr__(
                self, name, _strings(getattr(self, name), name, maximum=1_024)
            )
        for name in (
            "include_untracked",
            "reject_symlinks",
            "reject_nested_repositories",
            "exclude_credentials",
            "exclude_generated_state",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))
        if not self.exclude_credentials:
            raise PromptWorkflowContractError(
                "scan policy must exclude credentials"
            )


@dataclass(frozen=True)
class PromptPlanningPolicy(_WorkflowContract):
    SCHEMA: ClassVar[str] = PLANNING_POLICY_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "policy_id",
        "provider_preferences",
        "model_preferences",
        "candidate_count",
        "allow_model",
        "fallback_policy",
        "require_acceptance",
        "require_validation",
        "reject_unknown_applicability",
    )
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {
        "fallback_policy": LocalFallbackPolicy
    }

    policy_id: str
    provider_preferences: tuple[str, ...] = ()
    model_preferences: tuple[str, ...] = ()
    candidate_count: int = 1
    allow_model: bool = True
    fallback_policy: LocalFallbackPolicy = LocalFallbackPolicy.REQUIRED
    require_acceptance: bool = True
    require_validation: bool = True
    reject_unknown_applicability: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        for name in ("provider_preferences", "model_preferences"):
            object.__setattr__(
                self, name, _strings(getattr(self, name), name, maximum=64)
            )
        object.__setattr__(
            self,
            "candidate_count",
            _integer(self.candidate_count, "candidate_count", minimum=1, maximum=32),
        )
        object.__setattr__(self, "allow_model", _boolean(self.allow_model, "allow_model"))
        object.__setattr__(
            self,
            "fallback_policy",
            _enum(self.fallback_policy, LocalFallbackPolicy, "fallback_policy"),
        )
        for name in (
            "require_acceptance",
            "require_validation",
            "reject_unknown_applicability",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))


@dataclass(frozen=True)
class PromptOutputPolicy(_WorkflowContract):
    SCHEMA: ClassVar[str] = OUTPUT_POLICY_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "policy_id",
        "mode",
        "output_root",
        "allowed_output_roots",
        "markdown_path",
        "duckdb_path",
        "board_namespace",
        "task_prefix",
    )
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {"mode": OutputMode}

    policy_id: str
    mode: OutputMode
    output_root: str
    allowed_output_roots: tuple[str, ...]
    markdown_path: str = ""
    duckdb_path: str = ""
    board_namespace: str = "prompt-workflow"
    task_prefix: str = "TASK"

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(self, "mode", _enum(self.mode, OutputMode, "mode"))
        object.__setattr__(
            self, "output_root", _absolute_path(self.output_root, "output_root")
        )
        if isinstance(self.allowed_output_roots, str) or not isinstance(
            self.allowed_output_roots, Sequence
        ):
            raise PromptWorkflowContractError(
                "allowed_output_roots must be a sequence"
            )
        roots = tuple(
            sorted(
                {
                    _absolute_path(root, "allowed_output_roots")
                    for root in self.allowed_output_roots
                }
            )
        )
        if not roots:
            raise PromptWorkflowContractError(
                "allowed_output_roots must not be empty"
            )
        if not any(_is_within(self.output_root, root) for root in roots):
            raise PromptWorkflowPathError(
                "output_root is outside allowed_output_roots"
            )
        object.__setattr__(self, "allowed_output_roots", roots)
        for name in ("markdown_path", "duckdb_path"):
            object.__setattr__(
                self,
                name,
                _relative_path(getattr(self, name), name, allow_empty=True),
            )
        if self.mode in (OutputMode.MARKDOWN, OutputMode.BOTH) and not self.markdown_path:
            raise PromptWorkflowContractError(
                "markdown output mode requires markdown_path"
            )
        if self.mode in (OutputMode.DUCKDB, OutputMode.BOTH) and not self.duckdb_path:
            raise PromptWorkflowContractError(
                "duckdb output mode requires duckdb_path"
            )
        if self.mode is OutputMode.MARKDOWN and self.duckdb_path:
            raise PromptWorkflowContractError(
                "markdown output mode cannot define duckdb_path"
            )
        if self.mode is OutputMode.DUCKDB and self.markdown_path:
            raise PromptWorkflowContractError(
                "duckdb output mode cannot define markdown_path"
            )
        object.__setattr__(
            self, "board_namespace", _text(self.board_namespace, "board_namespace")
        )
        object.__setattr__(self, "task_prefix", _text(self.task_prefix, "task_prefix"))


@dataclass(frozen=True)
class PromptWorkflowRequest(_WorkflowContract):
    """Canonical pre-resolution workflow request with every semantic root pinned."""

    SCHEMA: ClassVar[str] = PROMPT_WORKFLOW_REQUEST_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "prompt_source",
        "repository_root",
        "directory",
        "repository_root_cid",
        "allowlist_cid",
        "scan_policy",
        "planning_policy",
        "output_policy",
        "budget",
        "caller",
        "program_root",
        "intent_ir_root",
        "legal_ir_root",
        "security_ir_root",
        "policy_root",
        "dry_run",
        "materialize",
        "start_after_materialize",
        "supervisor_profile",
        "state_root",
        "authority_cid",
        "idempotency_key",
        "lease_id",
        "fencing_epoch",
    )
    NESTED_FIELDS: ClassVar[Mapping[str, type[_WorkflowContract]]] = {
        "prompt_source": PromptSource,
        "scan_policy": DirectoryScanPolicy,
        "planning_policy": PromptPlanningPolicy,
        "output_policy": PromptOutputPolicy,
        "budget": PromptWorkflowBudget,
    }

    prompt_source: PromptSource
    repository_root: str
    directory: str
    repository_root_cid: str
    allowlist_cid: str
    scan_policy: DirectoryScanPolicy
    planning_policy: PromptPlanningPolicy
    output_policy: PromptOutputPolicy
    budget: PromptWorkflowBudget
    caller: str
    program_root: str
    intent_ir_root: str
    legal_ir_root: str
    security_ir_root: str
    policy_root: str
    dry_run: bool = True
    materialize: bool = False
    start_after_materialize: bool = False
    supervisor_profile: str = ""
    state_root: str = ""
    authority_cid: str = ""
    idempotency_key: str = ""
    lease_id: str = ""
    fencing_epoch: int | None = None

    def __post_init__(self) -> None:
        for name, kind in self.NESTED_FIELDS.items():
            value = getattr(self, name)
            if not isinstance(value, kind):
                raise PromptWorkflowContractError(f"{name} must be {kind.__name__}")
        repository_root = _absolute_path(self.repository_root, "repository_root")
        directory = _absolute_path(self.directory, "directory")
        if not _is_within(directory, repository_root):
            raise PromptWorkflowPathError(
                "directory must resolve within repository_root"
            )
        object.__setattr__(self, "repository_root", repository_root)
        object.__setattr__(self, "directory", directory)
        object.__setattr__(
            self,
            "repository_root_cid",
            _identity(self.repository_root_cid, "repository_root_cid"),
        )
        object.__setattr__(
            self, "allowlist_cid", _identity(self.allowlist_cid, "allowlist_cid")
        )
        object.__setattr__(self, "caller", _text(self.caller, "caller"))
        for name in (
            "program_root",
            "intent_ir_root",
            "legal_ir_root",
            "security_ir_root",
            "policy_root",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        for name in ("dry_run", "materialize", "start_after_materialize"):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))
        if self.dry_run and (self.materialize or self.start_after_materialize):
            raise PromptWorkflowContractError(
                "dry_run cannot request materialization or start"
            )
        if self.start_after_materialize and not self.materialize:
            raise PromptWorkflowContractError(
                "start_after_materialize requires materialize"
            )
        object.__setattr__(
            self,
            "supervisor_profile",
            _text(self.supervisor_profile, "supervisor_profile", required=False),
        )
        state_root = (
            _absolute_path(self.state_root, "state_root") if self.state_root else ""
        )
        object.__setattr__(self, "state_root", state_root)
        if self.start_after_materialize and (
            not self.supervisor_profile or not self.state_root
        ):
            raise PromptWorkflowContractError(
                "start requires supervisor_profile and state_root"
            )
        for name in ("authority_cid",):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name, required=False),
            )
        for name in ("idempotency_key", "lease_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if self.fencing_epoch is not None:
            object.__setattr__(
                self,
                "fencing_epoch",
                _integer(self.fencing_epoch, "fencing_epoch", minimum=1),
            )
        mutation = self.materialize or self.start_after_materialize
        bindings = (
            bool(self.authority_cid),
            bool(self.idempotency_key),
            bool(self.lease_id),
            self.fencing_epoch is not None,
        )
        if mutation and not all(bindings):
            raise PromptWorkflowContractError(
                "mutation requires authority, idempotency, lease, and fence"
            )
        if not mutation and any(bindings) and not all(bindings):
            raise PromptWorkflowContractError(
                "mutation bindings must be supplied together"
            )
        if len(self.canonical_bytes()) > self.budget.max_serialized_bytes:
            raise PromptWorkflowBoundsError(
                "request exceeds max_serialized_bytes"
            )

    @property
    def request_cid(self) -> str:
        return self.content_id

    @property
    def prompt_cid(self) -> str:
        return self.prompt_source.prompt_cid

    @property
    def output_root(self) -> str:
        return self.output_policy.output_root


@dataclass(frozen=True)
class PromptEvidenceRecord(_WorkflowContract):
    SCHEMA: ClassVar[str] = PROMPT_EVIDENCE_RECORD_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "evidence_key",
        "source_kind",
        "artifact_cid",
        "summary",
        "repository_paths",
        "claim_keys",
        "authority",
        "provenance",
        "status",
        "created_at_ms",
        "updated_at_ms",
    )
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = _VOLATILE_FIELDS
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {
        "authority": EvidenceAuthority,
        "status": RecordStatus,
    }

    evidence_key: str
    source_kind: str
    artifact_cid: str
    summary: str
    repository_paths: tuple[str, ...] = ()
    claim_keys: tuple[str, ...] = ()
    authority: EvidenceAuthority = EvidenceAuthority.SCAN_ADVISORY
    provenance: Mapping[str, Any] = field(default_factory=dict)
    status: RecordStatus = RecordStatus.ADMITTED
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in ("evidence_key", "source_kind"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "artifact_cid", _identity(self.artifact_cid, "artifact_cid")
        )
        object.__setattr__(self, "summary", _text(self.summary, "summary"))
        object.__setattr__(
            self,
            "repository_paths",
            _strings(self.repository_paths, "repository_paths", paths=True),
        )
        object.__setattr__(
            self, "claim_keys", _strings(self.claim_keys, "claim_keys")
        )
        object.__setattr__(
            self, "authority", _enum(self.authority, EvidenceAuthority, "authority")
        )
        object.__setattr__(
            self, "provenance", _freeze_json(self.provenance, "provenance")
        )
        object.__setattr__(self, "status", _enum(self.status, RecordStatus, "status"))
        for name in ("created_at_ms", "updated_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def evidence_cid(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class PromptAcceptanceRecord(_WorkflowContract):
    SCHEMA: ClassVar[str] = PROMPT_ACCEPTANCE_RECORD_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "criterion_key",
        "criterion",
        "evidence_cids",
        "validation_keys",
    )

    criterion_key: str
    criterion: str
    evidence_cids: tuple[str, ...] = ()
    validation_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "criterion_key", _text(self.criterion_key, "criterion_key")
        )
        object.__setattr__(self, "criterion", _text(self.criterion, "criterion"))
        object.__setattr__(
            self,
            "evidence_cids",
            tuple(
                sorted(
                    _identity(item, "evidence_cids")
                    for item in _strings(self.evidence_cids, "evidence_cids")
                )
            ),
        )
        object.__setattr__(
            self,
            "validation_keys",
            _strings(self.validation_keys, "validation_keys"),
        )


@dataclass(frozen=True)
class PromptValidationRecord(_WorkflowContract):
    SCHEMA: ClassVar[str] = PROMPT_VALIDATION_RECORD_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "validation_key",
        "argv",
        "cwd",
        "expected_exit_codes",
        "policy_cid",
    )

    validation_key: str
    argv: tuple[str, ...]
    cwd: str = "."
    expected_exit_codes: tuple[int, ...] = (0,)
    policy_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "validation_key", _text(self.validation_key, "validation_key")
        )
        object.__setattr__(
            self,
            "argv",
            _strings(self.argv, "argv", required=True, maximum=256, sort=False),
        )
        if any("\n" in item or "\r" in item or "\x00" in item for item in self.argv):
            raise PromptWorkflowContractError("argv contains unsafe control characters")
        if self.cwd == ".":
            object.__setattr__(self, "cwd", ".")
        else:
            object.__setattr__(self, "cwd", _relative_path(self.cwd, "cwd"))
        if isinstance(self.expected_exit_codes, (str, bytes)) or not isinstance(
            self.expected_exit_codes, Sequence
        ):
            raise PromptWorkflowContractError(
                "expected_exit_codes must be a sequence"
            )
        codes = tuple(
            sorted(
                {
                    _integer(code, "expected_exit_codes", maximum=255)
                    for code in self.expected_exit_codes
                }
            )
        )
        if not codes:
            raise PromptWorkflowContractError(
                "expected_exit_codes must not be empty"
            )
        object.__setattr__(self, "expected_exit_codes", codes)
        object.__setattr__(
            self,
            "policy_cid",
            _identity(self.policy_cid, "policy_cid", required=False),
        )


@dataclass(frozen=True)
class PromptOutputRecord(_WorkflowContract):
    SCHEMA: ClassVar[str] = PROMPT_OUTPUT_RECORD_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = ("path", "effect", "media_type")

    path: str
    effect: str = "write"
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, "path"))
        if self.effect not in {"create", "write", "modify", "delete"}:
            raise PromptWorkflowContractError(
                "effect must be create, write, modify, or delete"
            )
        object.__setattr__(self, "media_type", _text(self.media_type, "media_type"))


@dataclass(frozen=True)
class PromptGoalRecord(_WorkflowContract):
    SCHEMA: ClassVar[str] = PROMPT_GOAL_RECORD_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "goal_key",
        "parent_goal_cid",
        "dependency_goal_cids",
        "title",
        "objective",
        "rationale",
        "scope_paths",
        "acceptance",
        "evidence_cids",
        "risks",
        "assumptions",
        "provenance",
        "status",
        "created_at_ms",
        "updated_at_ms",
    )
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = _VOLATILE_FIELDS
    NESTED_SEQUENCES: ClassVar[Mapping[str, type[_WorkflowContract]]] = {
        "acceptance": PromptAcceptanceRecord
    }
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {"status": RecordStatus}

    goal_key: str
    parent_goal_cid: str
    dependency_goal_cids: tuple[str, ...]
    title: str
    objective: str
    rationale: str
    scope_paths: tuple[str, ...]
    acceptance: tuple[PromptAcceptanceRecord, ...]
    evidence_cids: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    status: RecordStatus = RecordStatus.PROPOSED
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_key", _text(self.goal_key, "goal_key"))
        object.__setattr__(
            self,
            "parent_goal_cid",
            _validate_cid(self.parent_goal_cid, "parent_goal_cid", required=False),
        )
        for name in ("dependency_goal_cids", "evidence_cids"):
            object.__setattr__(
                self,
                name,
                tuple(
                    sorted(
                        _validate_cid(item, name)
                        for item in _strings(getattr(self, name), name)
                    )
                ),
            )
        for name in ("title", "objective", "rationale"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "scope_paths", _strings(self.scope_paths, "scope_paths", paths=True)
        )
        if not self.acceptance or not all(
            isinstance(item, PromptAcceptanceRecord) for item in self.acceptance
        ):
            raise PromptGraphError("goal acceptance must not be empty")
        acceptance = tuple(sorted(self.acceptance, key=lambda item: item.content_id))
        if len({item.criterion_key for item in acceptance}) != len(acceptance):
            raise PromptGraphError("goal acceptance keys must be unique")
        object.__setattr__(self, "acceptance", acceptance)
        for name in ("risks", "assumptions"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        object.__setattr__(
            self, "provenance", _freeze_json(self.provenance, "provenance")
        )
        object.__setattr__(self, "status", _enum(self.status, RecordStatus, "status"))
        for name in ("created_at_ms", "updated_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def goal_cid(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class PromptTaskRecord(_WorkflowContract):
    """Immutable task specification; status and timestamps do not affect its CID."""

    SCHEMA: ClassVar[str] = PROMPT_TASK_RECORD_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "task_key",
        "goal_cid",
        "dependency_task_cids",
        "objective",
        "rationale",
        "scope_paths",
        "outputs",
        "validations",
        "acceptance",
        "evidence_cids",
        "policy_roots",
        "priority",
        "track",
        "bundle",
        "parallel_lane",
        "resource_class",
        "predicted_files",
        "risks",
        "assumptions",
        "fallback_behavior",
        "provenance",
        "status",
        "created_at_ms",
        "updated_at_ms",
    )
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = _VOLATILE_FIELDS
    NESTED_SEQUENCES: ClassVar[Mapping[str, type[_WorkflowContract]]] = {
        "outputs": PromptOutputRecord,
        "validations": PromptValidationRecord,
        "acceptance": PromptAcceptanceRecord,
    }
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {"status": RecordStatus}

    task_key: str
    goal_cid: str
    dependency_task_cids: tuple[str, ...]
    objective: str
    rationale: str
    scope_paths: tuple[str, ...]
    outputs: tuple[PromptOutputRecord, ...]
    validations: tuple[PromptValidationRecord, ...]
    acceptance: tuple[PromptAcceptanceRecord, ...]
    evidence_cids: tuple[str, ...]
    policy_roots: tuple[str, ...]
    priority: str = "P1"
    track: str = "prompt-workflow"
    bundle: str = ""
    parallel_lane: str = ""
    resource_class: str = "cpu-medium"
    predicted_files: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    fallback_behavior: str = "fail_closed"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    status: RecordStatus = RecordStatus.PROPOSED
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_key", _text(self.task_key, "task_key"))
        object.__setattr__(self, "goal_cid", _validate_cid(self.goal_cid, "goal_cid"))
        for name in ("dependency_task_cids", "evidence_cids"):
            object.__setattr__(
                self,
                name,
                tuple(
                    sorted(
                        _validate_cid(item, name)
                        for item in _strings(getattr(self, name), name)
                    )
                ),
            )
        object.__setattr__(self, "objective", _text(self.objective, "objective"))
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale"))
        object.__setattr__(
            self, "scope_paths", _strings(self.scope_paths, "scope_paths", paths=True)
        )
        for name, item_type in (
            ("outputs", PromptOutputRecord),
            ("validations", PromptValidationRecord),
            ("acceptance", PromptAcceptanceRecord),
        ):
            values = getattr(self, name)
            if not values or not all(isinstance(item, item_type) for item in values):
                raise PromptGraphError(f"task {name} must not be empty")
            canonical = tuple(sorted(values, key=lambda item: item.content_id))
            object.__setattr__(self, name, canonical)
        if len({item.path for item in self.outputs}) != len(self.outputs):
            raise PromptGraphError("task output paths must be unique")
        if len({item.validation_key for item in self.validations}) != len(
            self.validations
        ):
            raise PromptGraphError("task validation keys must be unique")
        if len({item.criterion_key for item in self.acceptance}) != len(
            self.acceptance
        ):
            raise PromptGraphError("task acceptance keys must be unique")
        validation_keys = {item.validation_key for item in self.validations}
        for criterion in self.acceptance:
            if not set(criterion.validation_keys).issubset(validation_keys):
                raise PromptGraphError(
                    "acceptance references an unknown validation key"
                )
        object.__setattr__(
            self,
            "policy_roots",
            tuple(
                sorted(
                    _identity(item, "policy_roots")
                    for item in _strings(
                        self.policy_roots, "policy_roots", required=True
                    )
                )
            ),
        )
        for name in (
            "priority",
            "track",
            "bundle",
            "parallel_lane",
            "resource_class",
            "fallback_behavior",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    name,
                    required=name not in {"bundle", "parallel_lane"},
                ),
            )
        object.__setattr__(
            self,
            "predicted_files",
            _strings(self.predicted_files, "predicted_files", paths=True),
        )
        for name in ("risks", "assumptions"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        object.__setattr__(
            self, "provenance", _freeze_json(self.provenance, "provenance")
        )
        object.__setattr__(self, "status", _enum(self.status, RecordStatus, "status"))
        for name in ("created_at_ms", "updated_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def task_cid(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class DirectoryScanReceipt(_WorkflowContract):
    SCHEMA: ClassVar[str] = DIRECTORY_SCAN_RECEIPT_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "request_cid",
        "repository_root",
        "directory",
        "repository_root_cid",
        "dirty_worktree_root",
        "scanner_policy_cid",
        "program_root",
        "ast_root",
        "index_root",
        "budget",
        "evidence",
        "counts",
        "exclusions",
        "truncations",
        "truncated",
        "started_at_ms",
        "finished_at_ms",
    )
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = frozenset(
        {"started_at_ms", "finished_at_ms"}
    )
    NESTED_FIELDS: ClassVar[Mapping[str, type[_WorkflowContract]]] = {
        "budget": PromptWorkflowBudget
    }
    NESTED_SEQUENCES: ClassVar[Mapping[str, type[_WorkflowContract]]] = {
        "evidence": PromptEvidenceRecord
    }

    request_cid: str
    repository_root: str
    directory: str
    repository_root_cid: str
    dirty_worktree_root: str
    scanner_policy_cid: str
    program_root: str
    ast_root: str
    index_root: str
    budget: PromptWorkflowBudget
    evidence: tuple[PromptEvidenceRecord, ...]
    counts: Mapping[str, int]
    exclusions: tuple[str, ...] = ()
    truncations: tuple[str, ...] = ()
    truncated: bool = False
    started_at_ms: int = 0
    finished_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in (
            "request_cid",
            "repository_root_cid",
            "dirty_worktree_root",
            "scanner_policy_cid",
            "program_root",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        for name in ("ast_root", "index_root"):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name, required=False)
            )
        repository_root = _absolute_path(self.repository_root, "repository_root")
        directory = _absolute_path(self.directory, "directory")
        if not _is_within(directory, repository_root):
            raise PromptWorkflowPathError(
                "scan directory must be within repository_root"
            )
        object.__setattr__(self, "repository_root", repository_root)
        object.__setattr__(self, "directory", directory)
        if not isinstance(self.budget, PromptWorkflowBudget):
            raise PromptWorkflowContractError(
                "budget must be PromptWorkflowBudget"
            )
        if len(self.evidence) > self.budget.max_evidence:
            raise PromptWorkflowBoundsError("scan evidence exceeds budget")
        if not all(isinstance(item, PromptEvidenceRecord) for item in self.evidence):
            raise PromptWorkflowContractError(
                "evidence must contain PromptEvidenceRecord values"
            )
        evidence = tuple(sorted(self.evidence, key=lambda item: item.content_id))
        if len({item.content_id for item in evidence}) != len(evidence):
            raise PromptWorkflowContractError("scan evidence contains duplicates")
        object.__setattr__(self, "evidence", evidence)
        frozen_counts = _freeze_json(self.counts, "counts")
        if not isinstance(frozen_counts, Mapping) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in frozen_counts.values()
        ):
            raise PromptWorkflowContractError(
                "scan counts must be non-negative integers"
            )
        object.__setattr__(self, "counts", frozen_counts)
        count_limits = {
            "files": self.budget.max_files,
            "file_count": self.budget.max_files,
            "scan_bytes": self.budget.max_scan_bytes,
            "scanned_bytes": self.budget.max_scan_bytes,
            "byte_count": self.budget.max_scan_bytes,
            "symbols": self.budget.max_symbols,
            "symbol_count": self.budget.max_symbols,
        }
        for key, maximum in count_limits.items():
            if key in frozen_counts and frozen_counts[key] > maximum:
                raise PromptWorkflowBoundsError(
                    f"scan count {key} exceeds declared budget"
                )
        for name in ("exclusions", "truncations"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        object.__setattr__(self, "truncated", _boolean(self.truncated, "truncated"))
        if bool(self.truncations) != self.truncated:
            raise PromptWorkflowContractError(
                "truncated must exactly reflect truncation reasons"
            )
        for name in ("started_at_ms", "finished_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.finished_at_ms and self.finished_at_ms < self.started_at_ms:
            raise PromptWorkflowContractError(
                "finished_at_ms cannot precede started_at_ms"
            )
        if len(self.canonical_bytes()) > self.budget.max_serialized_bytes:
            raise PromptWorkflowBoundsError("scan receipt exceeds byte budget")

    @property
    def scan_cid(self) -> str:
        return self.content_id


def _assert_acyclic(nodes: Mapping[str, Sequence[str]], noun: str) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise PromptGraphError(f"{noun} graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for dependency in nodes[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(nodes):
        visit(node)


def _graph_depth(nodes: Mapping[str, Sequence[str]]) -> int:
    depths: dict[str, int] = {}

    def depth(node: str) -> int:
        if node not in depths:
            dependencies = nodes[node]
            depths[node] = (
                1
                if not dependencies
                else 1 + max(depth(dependency) for dependency in dependencies)
            )
        return depths[node]

    return max((depth(node) for node in nodes), default=0)


@dataclass(frozen=True)
class PromptGoalGraph(_WorkflowContract):
    SCHEMA: ClassVar[str] = PROMPT_GOAL_GRAPH_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "request_cid",
        "scan_cid",
        "program_root",
        "policy_roots",
        "goals",
        "tasks",
        "evidence",
        "unresolved_questions",
        "uncertainty_debt",
        "status",
        "created_at_ms",
        "updated_at_ms",
    )
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = _VOLATILE_FIELDS
    NESTED_SEQUENCES: ClassVar[Mapping[str, type[_WorkflowContract]]] = {
        "goals": PromptGoalRecord,
        "tasks": PromptTaskRecord,
        "evidence": PromptEvidenceRecord,
    }
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {"status": RecordStatus}

    request_cid: str
    scan_cid: str
    program_root: str
    policy_roots: tuple[str, ...]
    goals: tuple[PromptGoalRecord, ...]
    tasks: tuple[PromptTaskRecord, ...]
    evidence: tuple[PromptEvidenceRecord, ...]
    unresolved_questions: tuple[str, ...] = ()
    uncertainty_debt: tuple[str, ...] = ()
    status: RecordStatus = RecordStatus.PROPOSED
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in ("request_cid", "scan_cid", "program_root"):
            object.__setattr__(self, name, _validate_cid(getattr(self, name), name))
        object.__setattr__(
            self,
            "policy_roots",
            tuple(
                sorted(
                    _identity(item, "policy_roots")
                    for item in _strings(
                        self.policy_roots, "policy_roots", required=True
                    )
                )
            ),
        )
        for name, item_type, maximum in (
            ("goals", PromptGoalRecord, ABSOLUTE_MAX_GOALS),
            ("tasks", PromptTaskRecord, ABSOLUTE_MAX_TASKS),
            ("evidence", PromptEvidenceRecord, ABSOLUTE_MAX_EVIDENCE),
        ):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise PromptGraphError(f"{name} must be a sequence")
            if not values and name in {"goals", "tasks"}:
                raise PromptGraphError(f"{name} must not be empty")
            if len(values) > maximum or not all(
                isinstance(item, item_type) for item in values
            ):
                raise PromptGraphError(f"{name} is malformed or exceeds its bound")
            canonical = tuple(sorted(values, key=lambda item: item.content_id))
            if len({item.content_id for item in canonical}) != len(canonical):
                raise PromptGraphError(f"{name} contains duplicate identities")
            keys = [
                getattr(item, f"{name[:-1]}_key", None)
                if name != "evidence"
                else item.evidence_key
                for item in canonical
            ]
            if len(keys) != len(set(keys)):
                raise PromptGraphError(f"{name} contains duplicate local keys")
            object.__setattr__(self, name, canonical)
        goals = {item.goal_cid: item for item in self.goals}
        roots = [item for item in self.goals if not item.parent_goal_cid]
        if len(roots) != 1:
            raise PromptGraphError("goal graph requires exactly one root goal")
        goal_edges: dict[str, tuple[str, ...]] = {}
        for goal in self.goals:
            references = tuple(
                item
                for item in (goal.parent_goal_cid, *goal.dependency_goal_cids)
                if item
            )
            if any(item not in goals for item in references):
                raise PromptGraphError("goal references unknown goal CID")
            if goal.goal_cid in references:
                raise PromptGraphError("goal cannot depend on itself")
            goal_edges[goal.goal_cid] = references
        _assert_acyclic(goal_edges, "goal")
        if _graph_depth(goal_edges) > ABSOLUTE_MAX_DEPTH:
            raise PromptWorkflowBoundsError("goal graph exceeds depth bound")
        root_cid = roots[0].goal_cid
        for goal in self.goals:
            cursor = goal
            seen: set[str] = set()
            while cursor.parent_goal_cid:
                if cursor.goal_cid in seen:
                    raise PromptGraphError("goal parent chain contains a cycle")
                seen.add(cursor.goal_cid)
                cursor = goals[cursor.parent_goal_cid]
            if cursor.goal_cid != root_cid:
                raise PromptGraphError("goal graph is disconnected")
        tasks = {item.task_cid: item for item in self.tasks}
        task_edges: dict[str, tuple[str, ...]] = {}
        for task in self.tasks:
            if task.goal_cid not in goals:
                raise PromptGraphError("task references unknown goal CID")
            if any(dep not in tasks for dep in task.dependency_task_cids):
                raise PromptGraphError("task references unknown task dependency")
            if task.task_cid in task.dependency_task_cids:
                raise PromptGraphError("task cannot depend on itself")
            task_edges[task.task_cid] = task.dependency_task_cids
            if not set(task.policy_roots).issubset(set(self.policy_roots)):
                raise PromptGraphError("task policy root is outside graph policy roots")
        _assert_acyclic(task_edges, "task")
        if _graph_depth(task_edges) > ABSOLUTE_MAX_DEPTH:
            raise PromptWorkflowBoundsError("task graph exceeds depth bound")
        evidence_cids = {item.evidence_cid for item in self.evidence}
        referenced = {
            cid for goal in self.goals for cid in goal.evidence_cids
        } | {cid for task in self.tasks for cid in task.evidence_cids}
        acceptance_referenced = {
            cid
            for goal in self.goals
            for criterion in goal.acceptance
            for cid in criterion.evidence_cids
        } | {
            cid
            for task in self.tasks
            for criterion in task.acceptance
            for cid in criterion.evidence_cids
        }
        if not referenced.issubset(evidence_cids):
            raise PromptGraphError("goal or task references unknown evidence CID")
        if not acceptance_referenced.issubset(evidence_cids):
            raise PromptGraphError(
                "acceptance references unknown evidence CID"
            )
        for name in ("unresolved_questions", "uncertainty_debt"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        object.__setattr__(self, "status", _enum(self.status, RecordStatus, "status"))
        for name in ("created_at_ms", "updated_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def root_goal(self) -> PromptGoalRecord:
        return next(goal for goal in self.goals if not goal.parent_goal_cid)

    @property
    def plan_root_cid(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class MaterializationReference(_WorkflowContract):
    SCHEMA: ClassVar[str] = MATERIALIZATION_REFERENCE_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "request_cid",
        "preview_receipt_cid",
        "plan_root_cid",
        "repository_root",
        "output_root",
        "mode",
        "projection_cids",
        "revision",
        "status",
        "created_at_ms",
        "updated_at_ms",
    )
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = _VOLATILE_FIELDS
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {
        "mode": OutputMode,
        "status": RecordStatus,
    }

    request_cid: str
    preview_receipt_cid: str
    plan_root_cid: str
    repository_root: str
    output_root: str
    mode: OutputMode
    projection_cids: tuple[str, ...]
    revision: int
    status: RecordStatus = RecordStatus.READY
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in ("request_cid", "preview_receipt_cid", "plan_root_cid"):
            object.__setattr__(self, name, _validate_cid(getattr(self, name), name))
        for name in ("repository_root", "output_root"):
            object.__setattr__(
                self, name, _absolute_path(getattr(self, name), name)
            )
        object.__setattr__(self, "mode", _enum(self.mode, OutputMode, "mode"))
        expected = 2 if self.mode is OutputMode.BOTH else 1
        projection_cids = tuple(
            sorted(
                _validate_cid(item, "projection_cids")
                for item in _strings(
                    self.projection_cids, "projection_cids", required=True
                )
            )
        )
        if len(projection_cids) != expected:
            raise PromptWorkflowContractError(
                "projection count does not match output mode"
            )
        object.__setattr__(self, "projection_cids", projection_cids)
        object.__setattr__(
            self, "revision", _integer(self.revision, "revision", minimum=1)
        )
        object.__setattr__(self, "status", _enum(self.status, RecordStatus, "status"))
        for name in ("created_at_ms", "updated_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def receipt_cid(self) -> str:
        return self.content_id

    @property
    def materialization_cid(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class SupervisorRunReference(_WorkflowContract):
    SCHEMA: ClassVar[str] = SUPERVISOR_RUN_REFERENCE_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "materialization_cid",
        "plan_root_cid",
        "repository_root",
        "state_root",
        "supervisor_profile",
        "lifecycle_request_cid",
        "process_identity_cid",
        "status",
        "started_at_ms",
        "updated_at_ms",
    )
    # The run is named before a process exists.  Process identity is an
    # observed-effect binding on the reference, not an input to the stable run
    # CID (plan/profile/state/lifecycle request are the inputs).
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = (
        _VOLATILE_FIELDS | frozenset({"process_identity_cid"})
    )
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {"status": RecordStatus}

    materialization_cid: str
    plan_root_cid: str
    repository_root: str
    state_root: str
    supervisor_profile: str
    lifecycle_request_cid: str
    process_identity_cid: str = ""
    status: RecordStatus = RecordStatus.READY
    started_at_ms: int = 0
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in (
            "materialization_cid",
            "plan_root_cid",
            "lifecycle_request_cid",
        ):
            object.__setattr__(self, name, _validate_cid(getattr(self, name), name))
        object.__setattr__(
            self,
            "process_identity_cid",
            _validate_cid(
                self.process_identity_cid, "process_identity_cid", required=False
            ),
        )
        for name in ("repository_root", "state_root"):
            object.__setattr__(
                self, name, _absolute_path(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "supervisor_profile",
            _text(self.supervisor_profile, "supervisor_profile"),
        )
        object.__setattr__(self, "status", _enum(self.status, RecordStatus, "status"))
        for name in ("started_at_ms", "updated_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def run_cid(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class SupervisorIncident(_WorkflowContract):
    SCHEMA: ClassVar[str] = SUPERVISOR_INCIDENT_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "repository_root",
        "state_root",
        "repository_root_cid",
        "policy_root",
        "run_cid",
        "kind",
        "failure_fingerprint",
        "target_ids",
        "evidence_cids",
        "health",
        "prior_recovery_cids",
        "cooldown_key",
        "status",
        "observed_at_ms",
        "updated_at_ms",
    )
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = _VOLATILE_FIELDS
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {
        "kind": IncidentKind,
        "status": RecordStatus,
    }

    repository_root: str
    state_root: str
    repository_root_cid: str
    policy_root: str
    run_cid: str
    kind: IncidentKind
    failure_fingerprint: str
    target_ids: tuple[str, ...]
    evidence_cids: tuple[str, ...]
    health: Mapping[str, Any]
    prior_recovery_cids: tuple[str, ...] = ()
    cooldown_key: str = ""
    status: RecordStatus = RecordStatus.FAILED
    observed_at_ms: int = 0
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in ("repository_root", "state_root"):
            object.__setattr__(
                self, name, _absolute_path(getattr(self, name), name)
            )
        for name in ("repository_root_cid", "policy_root", "run_cid"):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        object.__setattr__(self, "kind", _enum(self.kind, IncidentKind, "kind"))
        fingerprint = _text(
            self.failure_fingerprint, "failure_fingerprint", maximum=512
        )
        if not _DIGEST_RE.fullmatch(fingerprint):
            raise PromptWorkflowIdentityError(
                "failure_fingerprint must be sha256:<64 lowercase hex>"
            )
        object.__setattr__(self, "failure_fingerprint", fingerprint)
        object.__setattr__(
            self,
            "target_ids",
            _strings(self.target_ids, "target_ids", required=True, maximum=256),
        )
        for name in ("evidence_cids", "prior_recovery_cids"):
            object.__setattr__(
                self,
                name,
                tuple(
                    sorted(
                        _validate_cid(item, name)
                        for item in _strings(getattr(self, name), name)
                    )
                ),
            )
        object.__setattr__(self, "health", _freeze_json(self.health, "health"))
        object.__setattr__(
            self,
            "cooldown_key",
            _text(self.cooldown_key, "cooldown_key", required=False),
        )
        object.__setattr__(self, "status", _enum(self.status, RecordStatus, "status"))
        for name in ("observed_at_ms", "updated_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def incident_cid(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class RecoveryAttempt(_WorkflowContract):
    SCHEMA: ClassVar[str] = RECOVERY_ATTEMPT_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "operation",
        "target_id",
        "attempt",
        "outcome",
        "receipt_cid",
        "failure_fingerprint",
        "started_at_ms",
        "finished_at_ms",
    )
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = frozenset(
        {"started_at_ms", "finished_at_ms"}
    )
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {
        "operation": RescueOperation,
        "outcome": RecoveryAttemptOutcome,
    }

    operation: RescueOperation
    target_id: str
    attempt: int
    outcome: RecoveryAttemptOutcome
    receipt_cid: str = ""
    failure_fingerprint: str = ""
    started_at_ms: int = 0
    finished_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation", _enum(self.operation, RescueOperation, "operation")
        )
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        object.__setattr__(
            self, "attempt", _integer(self.attempt, "attempt", minimum=1)
        )
        object.__setattr__(
            self, "outcome", _enum(self.outcome, RecoveryAttemptOutcome, "outcome")
        )
        object.__setattr__(
            self,
            "receipt_cid",
            _validate_cid(self.receipt_cid, "receipt_cid", required=False),
        )
        fingerprint = _text(
            self.failure_fingerprint, "failure_fingerprint", required=False
        )
        if fingerprint and not _DIGEST_RE.fullmatch(fingerprint):
            raise PromptWorkflowIdentityError(
                "failure_fingerprint must be a sha256 digest"
            )
        object.__setattr__(self, "failure_fingerprint", fingerprint)
        for name in ("started_at_ms", "finished_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))


@dataclass(frozen=True)
class ProgrammaticRecoveryExhaustionReceipt(_WorkflowContract):
    SCHEMA: ClassVar[str] = PROGRAMMATIC_RECOVERY_EXHAUSTION_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "incident_cid",
        "repository_root_cid",
        "policy_root",
        "run_cid",
        "attempts",
        "inapplicable_operations",
        "exhaustion_reason",
        "budget",
        "circuit_open",
        "status",
        "created_at_ms",
        "updated_at_ms",
    )
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = _VOLATILE_FIELDS
    NESTED_FIELDS: ClassVar[Mapping[str, type[_WorkflowContract]]] = {
        "budget": PromptWorkflowBudget
    }
    NESTED_SEQUENCES: ClassVar[Mapping[str, type[_WorkflowContract]]] = {
        "attempts": RecoveryAttempt
    }
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {"status": RecordStatus}

    incident_cid: str
    repository_root_cid: str
    policy_root: str
    run_cid: str
    attempts: tuple[RecoveryAttempt, ...]
    inapplicable_operations: tuple[RescueOperation, ...]
    exhaustion_reason: str
    budget: PromptWorkflowBudget
    circuit_open: bool = False
    status: RecordStatus = RecordStatus.QUARANTINED
    created_at_ms: int = 0
    updated_at_ms: int = 0

    @classmethod
    def _decode_field(cls, name: str, value: Any) -> Any:
        if name == "inapplicable_operations":
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise PromptWorkflowContractError(
                    "inapplicable_operations must be a sequence"
                )
            return tuple(
                _enum(item, RescueOperation, "inapplicable_operations")
                for item in value
            )
        return super()._decode_field(name, value)

    def __post_init__(self) -> None:
        for name in (
            "incident_cid",
            "repository_root_cid",
            "policy_root",
            "run_cid",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        if not isinstance(self.budget, PromptWorkflowBudget):
            raise PromptWorkflowContractError(
                "budget must be PromptWorkflowBudget"
            )
        if len(self.attempts) > ABSOLUTE_MAX_ITEMS or not all(
            isinstance(item, RecoveryAttempt) for item in self.attempts
        ):
            raise PromptWorkflowBoundsError("recovery attempts exceed bound")
        attempts = tuple(
            sorted(
                self.attempts,
                key=lambda item: (
                    item.operation.value,
                    item.target_id,
                    item.attempt,
                    item.content_id,
                ),
            )
        )
        if len({item.content_id for item in attempts}) != len(attempts):
            raise PromptWorkflowContractError("recovery attempts contain duplicates")
        object.__setattr__(self, "attempts", attempts)
        operations = tuple(
            sorted(
                {
                    _enum(item, RescueOperation, "inapplicable_operations")
                    for item in self.inapplicable_operations
                },
                key=lambda item: item.value,
            )
        )
        if not attempts and not operations:
            raise PromptWorkflowContractError(
                "exhaustion requires attempted or inapplicable recovery actions"
            )
        object.__setattr__(self, "inapplicable_operations", operations)
        object.__setattr__(
            self,
            "exhaustion_reason",
            _text(self.exhaustion_reason, "exhaustion_reason"),
        )
        object.__setattr__(
            self, "circuit_open", _boolean(self.circuit_open, "circuit_open")
        )
        object.__setattr__(self, "status", _enum(self.status, RecordStatus, "status"))
        for name in ("created_at_ms", "updated_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def receipt_cid(self) -> str:
        return self.content_id

    @property
    def exhaustion_receipt_cid(self) -> str:
        return self.content_id


_FORBIDDEN_RESCUE_PARAMETER_KEYS = frozenset(
    {
        "argv",
        "code",
        "command",
        "completion",
        "credential",
        "new_path",
        "path",
        "paths",
        "patch",
        "policy",
        "script",
        "shell",
        "source_path",
        "destination_path",
        "output_path",
        "taskboard",
    }
)


@dataclass(frozen=True)
class RescueAction(_WorkflowContract):
    SCHEMA: ClassVar[str] = RESCUE_ACTION_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "operation",
        "target_id",
        "parameters",
        "precondition_cids",
        "expected_effects",
        "success_test",
        "stop_condition",
        "rollback_operation",
    )
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {
        "operation": RescueOperation
    }

    operation: RescueOperation
    target_id: str
    parameters: Mapping[str, Any]
    precondition_cids: tuple[str, ...]
    expected_effects: tuple[str, ...]
    success_test: str
    stop_condition: str
    rollback_operation: RescueOperation | None = None

    @classmethod
    def _decode_field(cls, name: str, value: Any) -> Any:
        if name == "rollback_operation":
            return (
                None
                if value is None
                else _enum(value, RescueOperation, "rollback_operation")
            )
        return super()._decode_field(name, value)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation", _enum(self.operation, RescueOperation, "operation")
        )
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        parameters = _freeze_json(self.parameters, "parameters")
        if not isinstance(parameters, Mapping):
            raise RescuePlanError("parameters must be an object")
        for key in parameters:
            normalized = key.lower().replace("-", "_")
            if normalized in _FORBIDDEN_RESCUE_PARAMETER_KEYS or any(
                marker in normalized
                for marker in (
                    "command",
                    "shell",
                    "patch",
                    "credential",
                    "policy",
                    "path",
                )
            ):
                raise RescuePlanError(
                    "rescue parameters contain a forbidden open-ended field"
                )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(
            self,
            "precondition_cids",
            tuple(
                sorted(
                    _validate_cid(item, "precondition_cids")
                    for item in _strings(
                        self.precondition_cids,
                        "precondition_cids",
                        required=True,
                    )
                )
            ),
        )
        object.__setattr__(
            self,
            "expected_effects",
            _strings(
                self.expected_effects, "expected_effects", required=True, maximum=64
            ),
        )
        object.__setattr__(
            self, "success_test", _text(self.success_test, "success_test")
        )
        object.__setattr__(
            self, "stop_condition", _text(self.stop_condition, "stop_condition")
        )
        if self.rollback_operation is not None:
            object.__setattr__(
                self,
                "rollback_operation",
                _enum(
                    self.rollback_operation,
                    RescueOperation,
                    "rollback_operation",
                ),
            )


@dataclass(frozen=True)
class RescuePlan(_WorkflowContract):
    SCHEMA: ClassVar[str] = RESCUE_PLAN_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "incident_cid",
        "exhaustion_receipt_cid",
        "repository_root_cid",
        "run_cid",
        "policy_root",
        "actions",
        "rationale_reference_cids",
        "unresolved_risks",
        "max_actions",
        "status",
        "created_at_ms",
        "updated_at_ms",
    )
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = _VOLATILE_FIELDS
    NESTED_SEQUENCES: ClassVar[Mapping[str, type[_WorkflowContract]]] = {
        "actions": RescueAction
    }
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {"status": RecordStatus}

    incident_cid: str
    exhaustion_receipt_cid: str
    repository_root_cid: str
    run_cid: str
    policy_root: str
    actions: tuple[RescueAction, ...]
    rationale_reference_cids: tuple[str, ...]
    unresolved_risks: tuple[str, ...]
    max_actions: int
    status: RecordStatus = RecordStatus.PROPOSED
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in (
            "incident_cid",
            "exhaustion_receipt_cid",
            "repository_root_cid",
            "run_cid",
            "policy_root",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        object.__setattr__(
            self,
            "max_actions",
            _integer(
                self.max_actions,
                "max_actions",
                minimum=1,
                maximum=ABSOLUTE_MAX_RESCUE_ACTIONS,
            ),
        )
        if (
            not self.actions
            or len(self.actions) > self.max_actions
            or not all(isinstance(item, RescueAction) for item in self.actions)
        ):
            raise RescuePlanError("actions are empty, malformed, or over budget")
        # Action order is semantic and is deliberately retained.
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(
            self,
            "rationale_reference_cids",
            tuple(
                sorted(
                    _validate_cid(item, "rationale_reference_cids")
                    for item in _strings(
                        self.rationale_reference_cids,
                        "rationale_reference_cids",
                        required=True,
                    )
                )
            ),
        )
        object.__setattr__(
            self,
            "unresolved_risks",
            _strings(self.unresolved_risks, "unresolved_risks"),
        )
        object.__setattr__(self, "status", _enum(self.status, RecordStatus, "status"))
        for name in ("created_at_ms", "updated_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def rescue_plan_cid(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class PromptWorkflowPreviewReceipt(_WorkflowContract):
    SCHEMA: ClassVar[str] = PROMPT_WORKFLOW_PREVIEW_RECEIPT_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "request_cid",
        "scan_cid",
        "plan_root_cid",
        "repository_root_cid",
        "program_root",
        "policy_roots",
        "admitted_goal_cids",
        "admitted_task_cids",
        "rejected_branch_cids",
        "rejection_reasons",
        "provider_receipt_cid",
        "deterministic_fallback",
        "expected_materialization_effects",
        "budget",
        "status",
        "created_at_ms",
        "updated_at_ms",
    )
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = _VOLATILE_FIELDS
    NESTED_FIELDS: ClassVar[Mapping[str, type[_WorkflowContract]]] = {
        "budget": PromptWorkflowBudget
    }
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {"status": RecordStatus}

    request_cid: str
    scan_cid: str
    plan_root_cid: str
    repository_root_cid: str
    program_root: str
    policy_roots: tuple[str, ...]
    admitted_goal_cids: tuple[str, ...]
    admitted_task_cids: tuple[str, ...]
    rejected_branch_cids: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    provider_receipt_cid: str = ""
    deterministic_fallback: bool = False
    expected_materialization_effects: tuple[str, ...] = ()
    budget: PromptWorkflowBudget = field(default_factory=PromptWorkflowBudget)
    status: RecordStatus = RecordStatus.ADMITTED
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in (
            "request_cid",
            "scan_cid",
            "plan_root_cid",
            "repository_root_cid",
            "program_root",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        for name in (
            "policy_roots",
            "admitted_goal_cids",
            "admitted_task_cids",
            "rejected_branch_cids",
        ):
            object.__setattr__(
                self,
                name,
                tuple(
                    sorted(
                        _identity(item, name)
                        for item in _strings(
                            getattr(self, name),
                            name,
                            required=name
                            in {
                                "policy_roots",
                                "admitted_goal_cids",
                                "admitted_task_cids",
                            },
                        )
                    )
                ),
            )
        if bool(self.rejected_branch_cids) != bool(self.rejection_reasons):
            raise PromptWorkflowContractError(
                "rejected branches and rejection reasons must occur together"
            )
        object.__setattr__(
            self,
            "rejection_reasons",
            _strings(self.rejection_reasons, "rejection_reasons"),
        )
        object.__setattr__(
            self,
            "provider_receipt_cid",
            _validate_cid(
                self.provider_receipt_cid,
                "provider_receipt_cid",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "deterministic_fallback",
            _boolean(self.deterministic_fallback, "deterministic_fallback"),
        )
        object.__setattr__(
            self,
            "expected_materialization_effects",
            _strings(
                self.expected_materialization_effects,
                "expected_materialization_effects",
            ),
        )
        if not isinstance(self.budget, PromptWorkflowBudget):
            raise PromptWorkflowContractError(
                "budget must be PromptWorkflowBudget"
            )
        if len(self.admitted_goal_cids) > self.budget.max_goals:
            raise PromptWorkflowBoundsError(
                "admitted goals exceed the declared workflow budget"
            )
        if len(self.admitted_task_cids) > self.budget.max_tasks:
            raise PromptWorkflowBoundsError(
                "admitted tasks exceed the declared workflow budget"
            )
        object.__setattr__(self, "status", _enum(self.status, RecordStatus, "status"))
        for name in ("created_at_ms", "updated_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if len(self.canonical_bytes()) > self.budget.max_serialized_bytes:
            raise PromptWorkflowBoundsError(
                "preview receipt exceeds max_serialized_bytes"
            )

    @property
    def receipt_cid(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class PromptWorkflowResult(_WorkflowContract):
    """Receipt-linked result for preview/materialize/start saga progress."""

    SCHEMA: ClassVar[str] = PROMPT_WORKFLOW_RESULT_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "request_cid",
        "outcome",
        "preview_receipt_cid",
        "materialization",
        "run",
        "completed_stage_cids",
        "failure_codes",
        "safe_continuation",
        "status",
        "created_at_ms",
        "updated_at_ms",
    )
    IDENTITY_EXCLUDED: ClassVar[frozenset[str]] = _VOLATILE_FIELDS
    ENUM_FIELDS: ClassVar[Mapping[str, type[Enum]]] = {
        "outcome": WorkflowOutcome,
        "status": RecordStatus,
    }

    request_cid: str
    outcome: WorkflowOutcome
    preview_receipt_cid: str
    materialization: MaterializationReference | None = None
    run: SupervisorRunReference | None = None
    completed_stage_cids: tuple[str, ...] = ()
    failure_codes: tuple[str, ...] = ()
    safe_continuation: str = ""
    status: RecordStatus = RecordStatus.READY
    created_at_ms: int = 0
    updated_at_ms: int = 0

    @classmethod
    def _decode_field(cls, name: str, value: Any) -> Any:
        if name == "materialization":
            return None if value is None else MaterializationReference.from_dict(value)
        if name == "run":
            return None if value is None else SupervisorRunReference.from_dict(value)
        return super()._decode_field(name, value)

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_cid", _validate_cid(self.request_cid, "request_cid"))
        object.__setattr__(
            self, "outcome", _enum(self.outcome, WorkflowOutcome, "outcome")
        )
        object.__setattr__(
            self,
            "preview_receipt_cid",
            _validate_cid(self.preview_receipt_cid, "preview_receipt_cid"),
        )
        if self.materialization is not None and not isinstance(
            self.materialization, MaterializationReference
        ):
            raise PromptWorkflowContractError(
                "materialization must be MaterializationReference"
            )
        if self.run is not None and not isinstance(self.run, SupervisorRunReference):
            raise PromptWorkflowContractError("run must be SupervisorRunReference")
        if self.run is not None and self.materialization is None:
            raise PromptWorkflowContractError(
                "run reference requires materialization reference"
            )
        if (
            self.materialization is not None
            and self.materialization.request_cid != self.request_cid
        ):
            raise PromptWorkflowIdentityError(
                "materialization is bound to another workflow request"
            )
        if (
            self.materialization is not None
            and self.materialization.preview_receipt_cid
            != self.preview_receipt_cid
        ):
            raise PromptWorkflowIdentityError(
                "materialization is bound to another preview receipt"
            )
        if (
            self.run is not None
            and self.run.materialization_cid != self.materialization.materialization_cid
        ):
            raise PromptWorkflowIdentityError(
                "run is bound to another materialization"
            )
        if (
            self.run is not None
            and self.run.plan_root_cid != self.materialization.plan_root_cid
        ):
            raise PromptWorkflowIdentityError(
                "run is bound to another plan root"
            )
        if (
            self.run is not None
            and self.run.repository_root != self.materialization.repository_root
        ):
            raise PromptWorkflowIdentityError(
                "run is bound to another repository root"
            )
        if self.outcome is WorkflowOutcome.PREVIEWED and (
            self.materialization is not None or self.run is not None
        ):
            raise PromptWorkflowContractError(
                "previewed outcome cannot include materialization or run"
            )
        if self.outcome is WorkflowOutcome.MATERIALIZED and (
            self.materialization is None or self.run is not None
        ):
            raise PromptWorkflowContractError(
                "materialized outcome requires only materialization"
            )
        if self.outcome is WorkflowOutcome.STARTED and (
            self.materialization is None or self.run is None
        ):
            raise PromptWorkflowContractError(
                "started outcome requires materialization and run"
            )
        object.__setattr__(
            self,
            "completed_stage_cids",
            tuple(
                sorted(
                    _validate_cid(item, "completed_stage_cids")
                    for item in _strings(
                        self.completed_stage_cids, "completed_stage_cids"
                    )
                )
            ),
        )
        object.__setattr__(
            self, "failure_codes", _strings(self.failure_codes, "failure_codes")
        )
        object.__setattr__(
            self,
            "safe_continuation",
            _text(self.safe_continuation, "safe_continuation", required=False),
        )
        if self.outcome in {
            WorkflowOutcome.PARTIAL,
            WorkflowOutcome.REJECTED,
            WorkflowOutcome.FAILED,
            WorkflowOutcome.QUARANTINED,
        } and not self.failure_codes:
            raise PromptWorkflowContractError(
                "non-success outcome requires failure_codes"
            )
        object.__setattr__(self, "status", _enum(self.status, RecordStatus, "status"))
        for name in ("created_at_ms", "updated_at_ms"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def receipt_cid(self) -> str:
        return self.content_id


def decode_prompt_workflow_request(
    payload: Mapping[str, Any],
) -> PromptWorkflowRequest:
    """Decode the strict provider-free workflow boundary."""

    return PromptWorkflowRequest.from_dict(payload)


# Stable compatibility spellings used by projection, lifecycle, and transport
# implementations.  Aliases do not alter schemas or identities.
WorkflowBudget = PromptWorkflowBudget
ScanPolicy = DirectoryScanPolicy
PlanningPolicy = PromptPlanningPolicy
OutputPolicy = PromptOutputPolicy
EvidenceRecord = PromptEvidenceRecord
GoalRecord = PromptGoalRecord
TaskRecord = PromptTaskRecord
AcceptanceRecord = PromptAcceptanceRecord
ValidationRecord = PromptValidationRecord
TaskOutputRecord = PromptOutputRecord
PromptMaterializationRef = MaterializationReference
PromptMaterializationReference = MaterializationReference
MaterializationRef = MaterializationReference
SupervisorRunRef = SupervisorRunReference
PromptRunReference = SupervisorRunReference
RunReference = SupervisorRunReference
RecoveryAttemptRecord = RecoveryAttempt
PromptWorkflowReceipt = PromptWorkflowResult
WorkflowResult = PromptWorkflowResult
WorkflowPreviewReceipt = PromptWorkflowPreviewReceipt


__all__ = [
    "ABSOLUTE_MAX_CONTRACT_BYTES",
    "ABSOLUTE_MAX_DEPTH",
    "ABSOLUTE_MAX_EVIDENCE",
    "ABSOLUTE_MAX_FILES",
    "ABSOLUTE_MAX_GOALS",
    "ABSOLUTE_MAX_PROMPT_BYTES",
    "ABSOLUTE_MAX_RESCUE_ACTIONS",
    "ABSOLUTE_MAX_TASKS",
    "AcceptanceRecord",
    "CONTRACT_VERSION",
    "DirectoryScanPolicy",
    "DirectoryScanReceipt",
    "EvidenceAuthority",
    "EvidenceRecord",
    "GoalRecord",
    "IncidentKind",
    "LocalFallbackPolicy",
    "MaterializationRef",
    "MaterializationReference",
    "NonCanonicalPromptWorkflowError",
    "OutputMode",
    "OutputPolicy",
    "PlanningPolicy",
    "ProgrammaticRecoveryExhaustionReceipt",
    "PROMPT_WORKFLOW_CONTRACT_VERSION",
    "PromptAcceptanceRecord",
    "PromptEvidenceRecord",
    "PromptGoalGraph",
    "PromptGoalRecord",
    "PromptMaterializationRef",
    "PromptMaterializationReference",
    "PromptOutputPolicy",
    "PromptOutputRecord",
    "PromptPlanningPolicy",
    "PromptRunReference",
    "PromptSecretError",
    "PromptSource",
    "PromptSourceError",
    "PromptSourceKind",
    "PromptTaskRecord",
    "PromptValidationRecord",
    "PromptWorkflowBoundsError",
    "PromptWorkflowBudget",
    "PromptWorkflowContractError",
    "PromptWorkflowIdentityError",
    "PromptWorkflowPathError",
    "PromptWorkflowPreviewReceipt",
    "PromptWorkflowReceipt",
    "PromptWorkflowRequest",
    "PromptWorkflowResult",
    "PromptGraphError",
    "RecordStatus",
    "RecoveryAttempt",
    "RecoveryAttemptOutcome",
    "RecoveryAttemptRecord",
    "RescueAction",
    "RescueOperation",
    "RescuePlan",
    "RescuePlanError",
    "RunReference",
    "SCHEMA_VERSION",
    "ScanPolicy",
    "SupervisorIncident",
    "SupervisorRunRef",
    "SupervisorRunReference",
    "TaskOutputRecord",
    "TaskRecord",
    "ValidationRecord",
    "WorkflowBudget",
    "WorkflowOutcome",
    "WorkflowPreviewReceipt",
    "WorkflowResult",
    "canonical_prompt_workflow_bytes",
    "decode_prompt_workflow_request",
    "prompt_workflow_cid",
]
