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

import argparse
import base64
import hashlib
import json
import posixpath
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Final, MutableMapping, Optional, TextIO


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


class PromptWorkflowServiceError(RuntimeError):
    """Base error raised by the provider-lazy workflow orchestrator."""


class PromptWorkflowStaleRootError(PromptWorkflowServiceError):
    """A request, scan, plan, policy, catalog, or output root is no longer current."""


class PromptWorkflowAuthorizationError(PromptWorkflowServiceError):
    """A mutation stage lacks its own current authority boundary."""


class PromptWorkflowReceiptError(PromptWorkflowServiceError):
    """A receipt is missing, foreign, corrupt, or bound to another stage."""


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
        "scan_cid",
        "program_root",
        "policy_roots",
        "catalog_root",
        "output_policy_cid",
        "task_source_identities",
        "expected_effects",
        "observed_effects",
        "event_cursors",
        "control_receipt_cid",
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
    scan_cid: str = ""
    program_root: str = ""
    policy_roots: tuple[str, ...] = ()
    catalog_root: str = ""
    output_policy_cid: str = ""
    task_source_identities: tuple[Mapping[str, Any], ...] = ()
    expected_effects: tuple[str, ...] = ()
    observed_effects: tuple[str, ...] = ()
    event_cursors: Mapping[str, Any] = field(default_factory=dict)
    control_receipt_cid: str = ""
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
        for name in (
            "scan_cid",
            "program_root",
            "catalog_root",
            "output_policy_cid",
            "control_receipt_cid",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self,
            "policy_roots",
            tuple(
                sorted(
                    _identity(item, "policy_roots")
                    for item in _strings(self.policy_roots, "policy_roots")
                )
            ),
        )
        if isinstance(self.task_source_identities, (str, bytes)) or not isinstance(
            self.task_source_identities, Sequence
        ):
            raise PromptWorkflowContractError(
                "task_source_identities must be a sequence"
            )
        identities = tuple(
            sorted(
                (
                    _freeze_json(item, "task_source_identities")
                    for item in self.task_source_identities
                ),
                key=lambda item: canonical_prompt_workflow_bytes(item),
            )
        )
        if any(not isinstance(item, Mapping) for item in identities):
            raise PromptWorkflowContractError(
                "task_source_identities must contain objects"
            )
        if identities and len(identities) != expected:
            raise PromptWorkflowContractError(
                "task-source identity count does not match output mode"
            )
        object.__setattr__(self, "task_source_identities", identities)
        for name in ("expected_effects", "observed_effects"):
            object.__setattr__(
                self, name, _strings(getattr(self, name), name)
            )
        if not set(self.observed_effects).issubset(set(self.expected_effects)):
            raise PromptWorkflowContractError(
                "observed effects must be declared expected effects"
            )
        cursors = _freeze_json(self.event_cursors, "event_cursors")
        if not isinstance(cursors, Mapping):
            raise PromptWorkflowContractError("event_cursors must be an object")
        object.__setattr__(self, "event_cursors", cursors)
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
        "intent_ir_root",
        "legal_ir_root",
        "security_ir_root",
        "output_policy_cid",
        "catalog_root",
        "planner_receipt_cid",
        "admission_receipt_cid",
        "artifact_refs",
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
    intent_ir_root: str = ""
    legal_ir_root: str = ""
    security_ir_root: str = ""
    output_policy_cid: str = ""
    catalog_root: str = ""
    planner_receipt_cid: str = ""
    admission_receipt_cid: str = ""
    artifact_refs: tuple[str, ...] = ()
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
        for name in (
            "intent_ir_root",
            "legal_ir_root",
            "security_ir_root",
            "output_policy_cid",
            "catalog_root",
            "planner_receipt_cid",
            "admission_receipt_cid",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self,
            "artifact_refs",
            tuple(
                sorted(
                    _identity(item, "artifact_refs")
                    for item in _strings(self.artifact_refs, "artifact_refs")
                )
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
        if self.status is RecordStatus.ADMITTED and (
            not self.admitted_goal_cids or not self.admitted_task_cids
        ):
            raise PromptWorkflowContractError(
                "admitted preview requires admitted goal and task branches"
            )
        if self.status is RecordStatus.REJECTED and (
            self.admitted_goal_cids or self.admitted_task_cids
        ):
            raise PromptWorkflowContractError(
                "rejected preview cannot publish admitted branches"
            )
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
        "expected_effects",
        "observed_effects",
        "task_source_identities",
        "event_cursors",
        "control_receipt_cids",
        "rollback_receipt_cids",
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
    expected_effects: tuple[str, ...] = ()
    observed_effects: tuple[str, ...] = ()
    task_source_identities: tuple[Mapping[str, Any], ...] = ()
    event_cursors: Mapping[str, Any] = field(default_factory=dict)
    control_receipt_cids: tuple[str, ...] = ()
    rollback_receipt_cids: tuple[str, ...] = ()
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
        for name in ("expected_effects", "observed_effects"):
            object.__setattr__(
                self, name, _strings(getattr(self, name), name)
            )
        if not set(self.observed_effects).issubset(set(self.expected_effects)):
            raise PromptWorkflowContractError(
                "observed effects must be declared expected effects"
            )
        if isinstance(self.task_source_identities, (str, bytes)) or not isinstance(
            self.task_source_identities, Sequence
        ):
            raise PromptWorkflowContractError(
                "task_source_identities must be a sequence"
            )
        identities = tuple(
            sorted(
                (
                    _freeze_json(item, "task_source_identities")
                    for item in self.task_source_identities
                ),
                key=lambda item: canonical_prompt_workflow_bytes(item),
            )
        )
        if any(not isinstance(item, Mapping) for item in identities):
            raise PromptWorkflowContractError(
                "task_source_identities must contain objects"
            )
        object.__setattr__(self, "task_source_identities", identities)
        cursors = _freeze_json(self.event_cursors, "event_cursors")
        if not isinstance(cursors, Mapping):
            raise PromptWorkflowContractError("event_cursors must be an object")
        object.__setattr__(self, "event_cursors", cursors)
        for name in ("control_receipt_cids", "rollback_receipt_cids"):
            object.__setattr__(
                self,
                name,
                tuple(
                    sorted(
                        _identity(item, name)
                        for item in _strings(getattr(self, name), name)
                    )
                ),
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


@dataclass
class _PromptPreviewState:
    """Process-local bodies needed to continue a body-free preview receipt."""

    request: PromptWorkflowRequest
    receipt: PromptWorkflowPreviewReceipt
    scan: DirectoryScanReceipt
    graph: PromptGoalGraph
    admission: Any


@dataclass
class _PromptMaterializationState:
    preview: _PromptPreviewState
    reference: MaterializationReference
    result: PromptWorkflowResult


def _reference_cid(value: Any, namespace: str) -> str:
    """Normalize a bounded external receipt/identity into this contract's CID."""

    if isinstance(value, str) and _CID_RE.fullmatch(value):
        return value
    if isinstance(value, _WorkflowContract):
        return value.content_id
    if isinstance(value, Mapping):
        return prompt_workflow_cid(
            {"namespace": namespace, "record": _wire_value(value)}
        )
    text = str(value or "").strip()
    if not text:
        return ""
    return prompt_workflow_cid({"namespace": namespace, "identity": text})


def _component_call(
    component: Any,
    method: str,
    variants: Sequence[tuple[tuple[Any, ...], Mapping[str, Any]]],
) -> Any:
    """Invoke one injected component without masking a component TypeError."""

    target = getattr(component, method, None)
    if target is None:
        target = component if callable(component) else None
    if target is None:
        raise TypeError(f"workflow component does not implement {method}")
    try:
        import inspect

        signature = inspect.signature(target)
    except (TypeError, ValueError):
        args, kwargs = variants[0]
        return target(*args, **dict(kwargs))
    for args, kwargs in variants:
        try:
            signature.bind(*args, **dict(kwargs))
        except TypeError:
            continue
        return target(*args, **dict(kwargs))
    raise TypeError(f"workflow component {method} has an unsupported signature")


class PromptSupervisorService:
    """Canonical provider-lazy prompt-to-supervisor workflow orchestrator.

    The service deliberately keeps the three authority and idempotency
    boundaries separate:

    * :meth:`preview` scans, plans, and admits but never mutates;
    * :meth:`materialize` requires a fresh mutation grant and delegates each
      projection to its own journaled/transactional task source;
    * :meth:`start` uses the existing shared control service and a different
      lifecycle request.

    Durable records contain only canonical roots and bounded artifact
    references.  The admitted graph is retained process-locally solely to
    continue the saga; callers that require cross-process continuation can
    supply a receipt mapping and restore the service with their own artifact
    loader rather than placing prompt or model bodies in a receipt.
    """

    def __init__(
        self,
        *,
        control_service: Any | None = None,
        repository_allowlist: Sequence[str | Path] = (),
        scanner: Any | None = None,
        planner: Any | None = None,
        admission: Any | None = None,
        admission_request_factory: Any | None = None,
        markdown_materializer: Any | None = None,
        duckdb_materializer: Any | None = None,
        artifact_store: Any | None = None,
        optional_analysis: Any | None = None,
        receipt_store: MutableMapping[str, Mapping[str, Any]] | Any | None = None,
        root_observer: Callable[..., Mapping[str, Any]] | None = None,
        catalog_root: str = "",
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.control_service = control_service
        self.repository_allowlist = tuple(str(Path(item)) for item in repository_allowlist)
        self.scanner = scanner
        self.planner = planner
        self.admission = admission
        self.admission_request_factory = admission_request_factory
        self.markdown_materializer = markdown_materializer
        self.duckdb_materializer = duckdb_materializer
        self.artifact_store = artifact_store
        self.optional_analysis = optional_analysis
        self.receipt_store = receipt_store
        self.root_observer = root_observer
        self._catalog_root = (
            _identity(catalog_root, "catalog_root") if catalog_root else ""
        )
        self._clock_ms = clock_ms or (lambda: 0)
        self._preview_by_request: dict[str, _PromptPreviewState] = {}
        self._preview_by_receipt: dict[str, _PromptPreviewState] = {}
        self._materialization_by_key: dict[str, _PromptMaterializationState] = {}
        self._materialization_by_cid: dict[str, _PromptMaterializationState] = {}
        self._idempotency_fingerprints: dict[tuple[str, str], str] = {}
        self._start_results: dict[str, PromptWorkflowResult] = {}
        self._lock = threading.RLock()

    @property
    def catalog_root(self) -> str:
        if self._catalog_root:
            return self._catalog_root
        if self.control_service is not None:
            catalog = getattr(self.control_service, "_catalog", None)
            value = getattr(catalog, "catalog_id", "") or getattr(
                catalog, "content_id", ""
            )
            if value:
                self._catalog_root = _reference_cid(value, "control-catalog")
                return self._catalog_root
        # Keep provider and process modules out of import/discovery.  The
        # contract catalog itself is a provider-free dependency loaded only
        # when a service instance is actually used.
        from .control_contracts import OPERATION_CATALOG_V2

        self._catalog_root = _reference_cid(
            OPERATION_CATALOG_V2.catalog_id, "control-catalog"
        )
        return self._catalog_root

    def _persist(self, value: _WorkflowContract) -> None:
        if self.receipt_store is None:
            return
        record = value.to_record()
        if isinstance(self.receipt_store, MutableMapping):
            existing = self.receipt_store.get(value.content_id)
            if existing is not None and dict(existing) != record:
                raise PromptWorkflowReceiptError(
                    "receipt store contains a conflicting canonical receipt"
                )
            self.receipt_store[value.content_id] = record
            return
        put = getattr(self.receipt_store, "put", None) or getattr(
            self.receipt_store, "store", None
        )
        if not callable(put):
            raise PromptWorkflowReceiptError(
                "receipt_store must be a mutable mapping or implement put"
            )
        put(value.content_id, record)

    def _scanner(self, request: PromptWorkflowRequest) -> Any:
        if self.scanner is not None:
            return self.scanner
        from .prompt_directory_scanner import PromptDirectoryScanner

        roots = self.repository_allowlist or (request.repository_root,)
        self.scanner = PromptDirectoryScanner(
            roots,
            artifact_store=self.artifact_store,
            optional_analysis=self.optional_analysis,
        )
        return self.scanner

    def _scan(self, request: PromptWorkflowRequest) -> DirectoryScanReceipt:
        scan = _component_call(
            self._scanner(request),
            "scan",
            (
                ((request,), {"clock_ms": self._clock_ms}),
                ((request,), {}),
            ),
        )
        if not isinstance(scan, DirectoryScanReceipt):
            scan = DirectoryScanReceipt.from_dict(scan)
        self._validate_scan(request, scan)
        return scan

    @staticmethod
    def _validate_scan(
        request: PromptWorkflowRequest, scan: DirectoryScanReceipt
    ) -> None:
        exact = (
            (scan.request_cid, request.request_cid, "request"),
            (scan.repository_root, request.repository_root, "repository path"),
            (scan.directory, request.directory, "directory"),
            (
                scan.repository_root_cid,
                request.repository_root_cid,
                "repository root",
            ),
            (
                scan.scanner_policy_cid,
                request.scan_policy.content_id,
                "scan policy",
            ),
            (scan.program_root, request.program_root, "program root"),
        )
        for observed, expected, noun in exact:
            if observed != expected:
                raise PromptWorkflowStaleRootError(
                    f"scan is bound to a different {noun}"
                )

    @staticmethod
    def _validate_graph(
        request: PromptWorkflowRequest,
        scan: DirectoryScanReceipt,
        graph: PromptGoalGraph,
    ) -> None:
        if graph.request_cid != request.request_cid:
            raise PromptWorkflowStaleRootError(
                "plan is bound to a different request root"
            )
        if graph.scan_cid != scan.scan_cid:
            raise PromptWorkflowStaleRootError(
                "plan is bound to a different scan root"
            )
        if graph.program_root != request.program_root:
            raise PromptWorkflowStaleRootError(
                "plan is bound to a different program root"
            )
        roots = {
            request.policy_root,
            request.intent_ir_root,
            request.legal_ir_root,
            request.security_ir_root,
        }
        if set(graph.policy_roots) != roots:
            raise PromptWorkflowStaleRootError(
                "plan policy/IR roots differ from the request"
            )

    def _plan(
        self,
        request: PromptWorkflowRequest,
        scan: DirectoryScanReceipt,
        *,
        capabilities: Mapping[str, Any] | None,
        constraint_summaries: Mapping[str, Any] | None,
    ) -> tuple[PromptGoalGraph, Any, str, bool]:
        if self.planner is None:
            from .prompt_goal_planner import generate_prompt_goal_graph

            result = generate_prompt_goal_graph(
                request,
                scan,
                capabilities=capabilities,
                constraint_summaries=constraint_summaries,
            )
        else:
            result = _component_call(
                self.planner,
                "plan",
                (
                    (
                        (request, scan),
                        {
                            "capabilities": capabilities,
                            "constraint_summaries": constraint_summaries,
                        },
                    ),
                    ((request, scan), {}),
                ),
            )
        graph = getattr(result, "graph", result)
        if not isinstance(graph, PromptGoalGraph):
            graph = PromptGoalGraph.from_dict(graph)
        self._validate_graph(request, scan, graph)
        planning_receipt = getattr(result, "receipt", None)
        planning_ref = _reference_cid(
            planning_receipt.to_dict()
            if hasattr(planning_receipt, "to_dict")
            else planning_receipt or {"plan_root_cid": graph.plan_root_cid},
            "prompt-planning-receipt",
        )
        used_fallback = bool(
            getattr(result, "used_fallback", False)
            or getattr(getattr(planning_receipt, "fallback", None), "used", False)
        )
        return graph, result, planning_ref, used_fallback

    def _admit(
        self,
        request: PromptWorkflowRequest,
        scan: DirectoryScanReceipt,
        graph: PromptGoalGraph,
        planning_result: Any,
    ) -> Any:
        if self.admission is not None:
            result = _component_call(
                self.admission,
                "admit",
                (
                    ((request, scan, graph, planning_result), {}),
                    ((request, scan, graph), {}),
                    ((graph,), {}),
                ),
            )
        else:
            from .prompt_plan_admission import (
                PromptPlanAdmissionRequest,
                admit_prompt_plan,
            )

            ir_request = None
            compound = None
            if self.admission_request_factory is not None:
                built = _component_call(
                    self.admission_request_factory,
                    "build",
                    (
                        ((request, scan, graph), {}),
                        ((graph, request, scan), {}),
                        ((graph,), {}),
                    ),
                )
                if isinstance(built, PromptPlanAdmissionRequest):
                    compound = built
                else:
                    ir_request = built
            result = (
                admit_prompt_plan(compound)
                if compound is not None
                else admit_prompt_plan(
                    graph,
                    repository_tree_id=scan.dirty_worktree_root,
                    ir_request=ir_request,
                    workflow_request=request,
                    scan_receipt=scan,
                )
            )
        admitted_graph = getattr(result, "admitted_graph", None)
        if admitted_graph is not None:
            if not isinstance(admitted_graph, PromptGoalGraph):
                raise PromptWorkflowReceiptError(
                    "admission returned a non-canonical admitted graph"
                )
            self._validate_graph(request, scan, admitted_graph)
            if admitted_graph.plan_root_cid != graph.plan_root_cid:
                raise PromptWorkflowReceiptError(
                    "admission changed the candidate graph identity"
                )
        return result

    @staticmethod
    def _admission_receipt_ref(result: Any) -> str:
        receipt = getattr(result, "receipt", result)
        payload = receipt.to_dict() if hasattr(receipt, "to_dict") else receipt
        return _reference_cid(payload, "prompt-admission-receipt")

    @staticmethod
    def _expected_materialization_effects(
        request: PromptWorkflowRequest,
    ) -> tuple[str, ...]:
        effects: list[str] = []
        if request.output_policy.mode in {OutputMode.MARKDOWN, OutputMode.BOTH}:
            effects.append(
                f"write_markdown:{request.output_policy.markdown_path}"
            )
        if request.output_policy.mode in {OutputMode.DUCKDB, OutputMode.BOTH}:
            effects.append(f"write_duckdb:{request.output_policy.duckdb_path}")
        return tuple(sorted(effects))

    @staticmethod
    def _rejections(admission: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
        receipt = getattr(admission, "receipt", admission)
        findings = tuple(getattr(receipt, "findings", ()) or ())
        branch_refs: list[str] = []
        reasons: list[str] = []
        for finding in findings:
            ref = getattr(finding, "finding_id", "") or _reference_cid(
                finding.to_dict() if hasattr(finding, "to_dict") else str(type(finding)),
                "prompt-admission-finding",
            )
            branch_refs.append(_reference_cid(ref, "prompt-admission-finding"))
            code = str(getattr(finding, "code", "") or "admission_rejected")
            reasons.append(code)
        if not branch_refs and not bool(getattr(admission, "admitted", False)):
            codes = tuple(getattr(admission, "reason_codes", ()) or ())
            reasons = [str(item) for item in codes] or ["admission_rejected"]
            branch_refs = [
                prompt_workflow_cid(
                    {"namespace": "prompt-admission-rejection", "code": code}
                )
                for code in reasons
            ]
        return tuple(sorted(set(branch_refs))), tuple(sorted(set(reasons)))

    def _validate_preview_control(
        self, control_request: Any | None
    ) -> str:
        if control_request is None:
            return ""
        if self.control_service is None:
            raise PromptWorkflowAuthorizationError(
                "preview control request requires a control service"
            )
        from .control_contracts import Operation

        if getattr(control_request, "operation", None) is not Operation.WORKFLOW_PREVIEW:
            raise PromptWorkflowAuthorizationError(
                "preview control request uses the wrong operation"
            )
        result = self.control_service.workflow_preview(control_request)
        if not bool(getattr(result, "succeeded", False)):
            raise PromptWorkflowAuthorizationError(
                "workflow preview was rejected by the shared control service"
            )
        return _reference_cid(
            getattr(result, "audit_receipt_id", "")
            or getattr(result, "result_id", ""),
            "workflow-preview-control-receipt",
        )

    def preview(
        self,
        request: PromptWorkflowRequest,
        *,
        control_request: Any | None = None,
        capabilities: Mapping[str, Any] | None = None,
        constraint_summaries: Mapping[str, Any] | None = None,
    ) -> PromptWorkflowPreviewReceipt:
        """Scan, plan, and admit one request without applying an effect."""

        if not isinstance(request, PromptWorkflowRequest):
            raise TypeError("request must be PromptWorkflowRequest")
        with self._lock:
            existing = self._preview_by_request.get(request.request_cid)
            if existing is not None:
                self._verify_current(existing)
                return existing.receipt

            control_ref = self._validate_preview_control(control_request)
            scan = self._scan(request)
            graph, planning, planner_ref, fallback = self._plan(
                request,
                scan,
                capabilities=capabilities,
                constraint_summaries=constraint_summaries,
            )
            admission = self._admit(request, scan, graph, planning)
            admitted = bool(getattr(admission, "admitted", False))
            admitted_graph = getattr(admission, "admitted_graph", None)
            if admitted and not isinstance(admitted_graph, PromptGoalGraph):
                raise PromptWorkflowReceiptError(
                    "admitted result does not carry the exact admitted graph"
                )
            admission_ref = self._admission_receipt_ref(admission)
            rejected_refs, rejection_reasons = self._rejections(admission)
            selected_graph = admitted_graph if admitted else graph
            plan_root = (
                str(getattr(admission, "plan_root_cid", "") or "")
                if admitted
                else graph.plan_root_cid
            )
            if not plan_root:
                raise PromptWorkflowReceiptError(
                    "admission did not publish a plan root"
                )
            artifact_refs = tuple(
                sorted(
                    {
                        scan.scan_cid,
                        graph.plan_root_cid,
                        planner_ref,
                        admission_ref,
                        *(
                            (control_ref,)
                            if control_ref
                            else ()
                        ),
                    }
                )
            )
            now = int(self._clock_ms())
            receipt = PromptWorkflowPreviewReceipt(
                request_cid=request.request_cid,
                scan_cid=scan.scan_cid,
                plan_root_cid=plan_root,
                repository_root_cid=request.repository_root_cid,
                program_root=request.program_root,
                policy_roots=selected_graph.policy_roots,
                admitted_goal_cids=(
                    tuple(goal.goal_cid for goal in selected_graph.goals)
                    if admitted
                    else ()
                ),
                admitted_task_cids=(
                    tuple(task.task_cid for task in selected_graph.tasks)
                    if admitted
                    else ()
                ),
                rejected_branch_cids=rejected_refs,
                rejection_reasons=rejection_reasons,
                provider_receipt_cid=planner_ref,
                deterministic_fallback=fallback,
                expected_materialization_effects=(
                    self._expected_materialization_effects(request)
                ),
                budget=request.budget,
                intent_ir_root=request.intent_ir_root,
                legal_ir_root=request.legal_ir_root,
                security_ir_root=request.security_ir_root,
                output_policy_cid=request.output_policy.content_id,
                catalog_root=self.catalog_root,
                planner_receipt_cid=planner_ref,
                admission_receipt_cid=admission_ref,
                artifact_refs=artifact_refs,
                status=(
                    RecordStatus.ADMITTED if admitted else RecordStatus.REJECTED
                ),
                created_at_ms=now,
                updated_at_ms=now,
            )
            self._persist(receipt)
            state = _PromptPreviewState(
                request=request,
                receipt=receipt,
                scan=scan,
                graph=selected_graph,
                admission=admission,
            )
            self._preview_by_request[request.request_cid] = state
            self._preview_by_receipt[receipt.receipt_cid] = state
            return receipt

    def _preview_state(
        self, preview_ref: str | PromptWorkflowPreviewReceipt
    ) -> _PromptPreviewState:
        ref = (
            preview_ref.receipt_cid
            if isinstance(preview_ref, PromptWorkflowPreviewReceipt)
            else str(preview_ref or "")
        )
        state = self._preview_by_receipt.get(ref)
        if state is None:
            raise PromptWorkflowReceiptError(
                "preview receipt is unavailable for safe continuation"
            )
        return state

    def _verify_current(self, state: _PromptPreviewState) -> None:
        expected = {
            "request_cid": state.request.request_cid,
            "repository_root_cid": state.request.repository_root_cid,
            "scan_cid": state.scan.scan_cid,
            "dirty_worktree_root": state.scan.dirty_worktree_root,
            "plan_root_cid": state.receipt.plan_root_cid,
            "program_root": state.request.program_root,
            "policy_root": state.request.policy_root,
            "intent_ir_root": state.request.intent_ir_root,
            "legal_ir_root": state.request.legal_ir_root,
            "security_ir_root": state.request.security_ir_root,
            "output_policy_cid": state.request.output_policy.content_id,
            "catalog_root": state.receipt.catalog_root,
            "output_mode": state.request.output_policy.mode.value,
            "markdown_path": state.request.output_policy.markdown_path,
            "duckdb_path": state.request.output_policy.duckdb_path,
        }
        if self.catalog_root != state.receipt.catalog_root:
            raise PromptWorkflowStaleRootError(
                "current catalog_root differs from the preview receipt"
            )
        if self.root_observer is not None:
            observed = _component_call(
                self.root_observer,
                "observe",
                (
                    ((state.request, state.scan, state.receipt), {}),
                    ((state.request,), {}),
                    ((), {}),
                ),
            )
            if not isinstance(observed, Mapping):
                raise PromptWorkflowStaleRootError(
                    "root observer did not return a root mapping"
                )
            missing = set(expected).difference(observed)
            if missing:
                raise PromptWorkflowStaleRootError(
                    "root observer omitted required current roots: "
                    + ", ".join(sorted(missing))
                )
            unknown = set(observed).difference(expected)
            if unknown:
                raise PromptWorkflowStaleRootError(
                    "root observer returned unknown roots: "
                    + ", ".join(sorted(str(item) for item in unknown))
                )
            for key, value in expected.items():
                if observed[key] != value:
                    raise PromptWorkflowStaleRootError(
                        f"current {key} differs from the preview receipt"
                    )
            return
        fresh = self._scan(state.request)
        if (
            fresh.scan_cid != state.scan.scan_cid
            or fresh.dirty_worktree_root != state.scan.dirty_worktree_root
        ):
            raise PromptWorkflowStaleRootError(
                "repository scan root changed after preview"
            )

    @staticmethod
    def _failure_result(
        state: _PromptPreviewState,
        *,
        code: str,
        outcome: WorkflowOutcome = WorkflowOutcome.FAILED,
        materialization: MaterializationReference | None = None,
        completed: Sequence[str] = (),
        continuation: str = "",
        expected: Sequence[str] = (),
        observed: Sequence[str] = (),
        identities: Sequence[Mapping[str, Any]] = (),
        cursors: Mapping[str, Any] | None = None,
        control_receipts: Sequence[str] = (),
    ) -> PromptWorkflowResult:
        status = (
            RecordStatus.BLOCKED
            if outcome is WorkflowOutcome.PARTIAL
            else RecordStatus.REJECTED
            if outcome is WorkflowOutcome.REJECTED
            else RecordStatus.FAILED
        )
        return PromptWorkflowResult(
            request_cid=state.request.request_cid,
            outcome=outcome,
            preview_receipt_cid=state.receipt.receipt_cid,
            materialization=materialization,
            completed_stage_cids=tuple(completed),
            failure_codes=(code,),
            safe_continuation=continuation,
            expected_effects=tuple(expected),
            observed_effects=tuple(observed),
            task_source_identities=tuple(identities),
            event_cursors=cursors or {},
            control_receipt_cids=tuple(control_receipts),
            status=status,
        )

    def _validate_materialize_control(
        self, state: _PromptPreviewState, control_request: Any
    ) -> tuple[Any, str]:
        from .control_contracts import Operation

        if getattr(control_request, "operation", None) is not Operation.WORKFLOW_MATERIALIZE:
            raise PromptWorkflowAuthorizationError(
                "materialization control request uses the wrong operation"
            )
        parameters = control_request.parameters
        exact = {
            "preview_ref": state.receipt.receipt_cid,
            "preview_root": state.receipt.plan_root_cid,
            "output_mode": state.request.output_policy.mode.value,
            "markdown_path": state.request.output_policy.markdown_path,
            "duckdb_path": state.request.output_policy.duckdb_path,
            "catalog_root": state.receipt.catalog_root,
        }
        for key, expected in exact.items():
            observed = parameters.get(key)
            if observed not in (None, "") and observed != expected:
                raise PromptWorkflowStaleRootError(
                    f"materialization {key} differs from preview"
                )
        if str(control_request.repository_root) != state.request.repository_root:
            raise PromptWorkflowStaleRootError(
                "materialization repository root differs from preview"
            )
        result = self.control_service.workflow_materialize(control_request)
        receipt = _reference_cid(
            getattr(result, "audit_receipt_id", "")
            or getattr(result, "result_id", ""),
            "workflow-materialize-control-receipt",
        )
        return result, receipt

    @staticmethod
    def _authorization_binding(
        state: _PromptPreviewState,
        *,
        control_request: Any | None,
        authorization: Any | None,
        idempotency_key: str,
        lease_id: str,
        fencing_epoch: int | None,
    ) -> tuple[str, str, str, int]:
        if control_request is not None:
            return (
                _reference_cid(control_request.request_id, "materialize-request"),
                str(control_request.idempotency_key),
                str(control_request.lease_id),
                int(control_request.fencing_epoch),
            )
        selected_authorization = authorization or state.request.authority_cid
        selected_key = idempotency_key or state.request.idempotency_key
        selected_lease = lease_id or state.request.lease_id
        selected_fence = (
            fencing_epoch
            if fencing_epoch is not None
            else state.request.fencing_epoch
        )
        if hasattr(selected_authorization, "permitted") and not bool(
            selected_authorization.permitted
        ):
            raise PromptWorkflowAuthorizationError(
                "materialization authorization is not permitted"
            )
        authority_ref = _reference_cid(
            getattr(selected_authorization, "content_id", "")
            or selected_authorization,
            "workflow-materialization-authority",
        )
        if (
            not authority_ref
            or not selected_key
            or not selected_lease
            or selected_fence is None
        ):
            raise PromptWorkflowAuthorizationError(
                "materialization requires separate authority, idempotency, lease, and fence"
            )
        return authority_ref, str(selected_key), str(selected_lease), int(selected_fence)

    @staticmethod
    def _cursor_reference(value: Any, namespace: str) -> str:
        if value in (None, ""):
            return ""
        if hasattr(value, "to_dict"):
            return _reference_cid(value.to_dict(), namespace)
        if isinstance(value, Mapping):
            return _reference_cid(value, namespace)
        return str(value)

    def _normalize_projection(
        self,
        kind: str,
        raw: Any,
        state: _PromptPreviewState,
        path: str,
    ) -> Mapping[str, Any]:
        effect = f"write_{kind}:{path}"
        if isinstance(raw, Mapping):
            committed = bool(raw.get("committed", True))
            if not committed:
                raise PromptWorkflowServiceError(
                    f"{kind} projection did not commit"
                )
            projection_identity = (
                raw.get("projection_cid")
                or raw.get("projection_id")
                or raw.get("source_id")
                or raw.get("receipt_cid")
            )
            revision = raw.get("revision", 1)
            changed = bool(raw.get("changed", False))
            replayed = bool(raw.get("replayed", not changed))
            cursor = raw.get("event_cursor", raw.get("cursor", ""))
            source_schema = str(
                raw.get("source_schema")
                or raw.get("schema")
                or f"prompt-{kind}-task-source"
            )
            task_cids = tuple(
                raw.get("task_cids")
                or (task.task_cid for task in state.graph.tasks)
            )
            root_id = str(
                raw.get("plan_root_cid")
                or raw.get("plan_root")
                or state.receipt.plan_root_cid
            )
            source_identity = raw.get("task_source_identity")
        else:
            committed = bool(getattr(raw, "committed", True))
            if not committed:
                raise PromptWorkflowServiceError(
                    f"{kind} projection did not commit"
                )
            projection = getattr(raw, "projection", None)
            snapshot = getattr(raw, "snapshot", None)
            projection_identity = (
                getattr(projection, "projection_id", "")
                or getattr(snapshot, "projection_cid", "")
                or getattr(snapshot, "projection_id", "")
                or getattr(raw, "projection_cid", "")
            )
            revision = (
                getattr(snapshot, "revision", None)
                or getattr(projection, "revision", None)
                or 1
            )
            changed = bool(getattr(raw, "changed", False))
            replayed = bool(
                getattr(raw, "no_op", False)
                or getattr(raw, "replayed", not changed)
            )
            cursor = getattr(raw, "event_cursor", "")
            source_schema = str(
                getattr(snapshot, "source_schema", "")
                or getattr(snapshot, "projection_schema", "")
                or getattr(projection, "schema", "")
                or f"prompt-{kind}-task-source"
            )
            task_cids = tuple(
                getattr(snapshot, "task_cids", ())
                or getattr(projection, "task_cids", ())
                or tuple(task.task_cid for task in state.graph.tasks)
            )
            root_id = str(
                getattr(snapshot, "plan_root_cid", "")
                or getattr(snapshot, "plan_root", "")
                or getattr(projection, "plan_root", "")
                or state.receipt.plan_root_cid
            )
            source_identity = getattr(raw, "task_source_identity", None)
        try:
            selected_revision = int(revision)
        except (TypeError, ValueError) as exc:
            raise PromptWorkflowReceiptError(
                f"{kind} projection revision is malformed"
            ) from exc
        if selected_revision < 1:
            raise PromptWorkflowReceiptError(
                f"{kind} projection revision is malformed"
            )
        projection_cid = _reference_cid(
            projection_identity,
            f"{kind}-task-source-projection",
        )
        if not projection_cid:
            raise PromptWorkflowReceiptError(
                f"{kind} projection did not publish an identity"
            )
        if root_id != state.receipt.plan_root_cid:
            raise PromptWorkflowStaleRootError(
                f"{kind} task source published a foreign plan root"
            )
        expected_tasks = tuple(
            sorted(task.task_cid for task in state.graph.tasks)
        )
        if tuple(sorted(str(item) for item in task_cids)) != expected_tasks:
            raise PromptWorkflowReceiptError(
                f"{kind} task source population differs from admitted tasks"
            )
        if isinstance(source_identity, Mapping):
            identity = dict(source_identity)
        else:
            identity = {
                "kind": kind,
                "source_schema": source_schema,
                "source_id": str(projection_identity),
                "root_id": root_id,
                "repository_root_cid": state.request.repository_root_cid,
                "scan_cid": state.scan.scan_cid,
                "revision": selected_revision,
                "path": path,
                "task_cids": list(expected_tasks),
            }
        identity.setdefault("kind", kind)
        identity.setdefault("source_id", str(projection_identity))
        identity.setdefault("root_id", root_id)
        identity.setdefault("revision", selected_revision)
        cursor_ref = self._cursor_reference(
            cursor, f"{kind}-task-source-event-cursor"
        )
        return MappingProxyType(
            {
                "kind": kind,
                "projection_cid": projection_cid,
                "revision": selected_revision,
                "identity": identity,
                "cursor": cursor_ref,
                "effect": effect,
                "changed": changed,
                "replayed": replayed,
            }
        )

    def _materialize_markdown(
        self,
        state: _PromptPreviewState,
        *,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        path = state.request.output_policy.markdown_path
        if self.markdown_materializer is not None:
            raw = _component_call(
                self.markdown_materializer,
                "materialize",
                (
                    (
                        (state.admission,),
                        {
                            "request": state.request,
                            "preview": state.receipt,
                            "idempotency_key": idempotency_key,
                        },
                    ),
                    ((state.admission, state.request, state.receipt), {}),
                    ((state.admission,), {}),
                ),
            )
        else:
            from .markdown_task_source import MarkdownTaskSource

            absolute = (
                Path(state.request.output_policy.output_root) / path
            )
            backend = MarkdownTaskSource(
                absolute,
                root=state.request.output_policy.output_root,
                task_prefix=state.request.output_policy.task_prefix,
                board_namespace=state.request.output_policy.board_namespace,
                max_bytes=state.request.budget.max_serialized_bytes,
                max_tasks=state.request.budget.max_tasks,
            )
            raw = backend.materialize(
                state.admission,
                revision=1,
                epoch_id=idempotency_key,
            )
            cursor = backend.store.event_cursor()
            # The native result intentionally omits the event stream because
            # it lives beside the taskboard.  Attach only its bounded cursor.
            raw = {
                "committed": raw.committed,
                "changed": raw.changed,
                "replayed": raw.no_op,
                "projection_id": raw.projection.projection_id,
                "source_schema": raw.projection.schema,
                "plan_root": raw.projection.plan_root,
                "revision": raw.projection.revision,
                "task_cids": raw.projection.task_cids,
                "event_cursor": cursor,
            }
        return self._normalize_projection("markdown", raw, state, path)

    def _materialize_duckdb(
        self,
        state: _PromptPreviewState,
        *,
        authority_ref: str,
        idempotency_key: str,
        fencing_epoch: int,
    ) -> Mapping[str, Any]:
        path = state.request.output_policy.duckdb_path
        if self.duckdb_materializer is not None:
            raw = _component_call(
                self.duckdb_materializer,
                "materialize",
                (
                    (
                        (state.admission,),
                        {
                            "request": state.request,
                            "preview": state.receipt,
                            "authority_ref": authority_ref,
                            "idempotency_key": idempotency_key,
                            "fencing_epoch": fencing_epoch,
                        },
                    ),
                    ((state.admission, state.request, state.receipt), {}),
                    ((state.graph,), {}),
                ),
            )
        else:
            from .duckdb_task_source import DuckDBTaskSource
            from .formal_plan_compiler import prompt_goal_graph_to_formal_input

            if not DuckDBTaskSource.available():
                raise PromptWorkflowServiceError(
                    "optional DuckDB capability is unavailable"
                )
            absolute = (
                Path(state.request.output_policy.output_root) / path
            )
            backend = DuckDBTaskSource(
                absolute,
                expected_plan_root_cid=state.receipt.plan_root_cid,
                expected_repository_tree_id=state.scan.dirty_worktree_root,
                writer_id=authority_ref,
                fencing_token=fencing_epoch,
            )
            # Use the compiler projection with the admitted root as its source
            # root.  This gives DuckDB and Markdown the same post-admission
            # plan identity while retaining independent recompilation.
            formal = prompt_goal_graph_to_formal_input(
                state.graph,
                repository_tree_id=state.scan.dirty_worktree_root,
            )
            formal["plan_root_cid"] = state.receipt.plan_root_cid
            raw = backend.materialize(
                formal,
                repository_tree_id=state.scan.dirty_worktree_root,
                plan_root_cid=state.receipt.plan_root_cid,
                receipt={
                    "preview_receipt_cid": state.receipt.receipt_cid,
                    "request_cid": state.request.request_cid,
                    "scan_cid": state.scan.scan_cid,
                    "catalog_root": state.receipt.catalog_root,
                    "idempotency_ref": _reference_cid(
                        idempotency_key, "workflow-materialization-idempotency"
                    ),
                },
                writer_id=authority_ref,
                fencing_token=fencing_epoch,
            )
            snapshot = backend.snapshot()
            raw = {
                **dict(raw),
                "source_schema": snapshot.source_schema,
                "plan_root_cid": snapshot.plan_root_cid,
                "revision": snapshot.revision,
                "task_cids": tuple(
                    record.task_cid for record in backend.list_tasks(limit=state.request.budget.max_tasks)
                ),
                "event_cursor": snapshot.event_cursor,
            }
        return self._normalize_projection("duckdb", raw, state, path)

    def materialize(
        self,
        preview_ref: str | PromptWorkflowPreviewReceipt,
        *,
        control_request: Any | None = None,
        authorization: Any | None = None,
        idempotency_key: str = "",
        lease_id: str = "",
        fencing_epoch: int | None = None,
    ) -> PromptWorkflowResult:
        """Apply an admitted preview to its exact pinned task-source paths."""

        with self._lock:
            state = self._preview_state(preview_ref)
            expected = state.receipt.expected_materialization_effects
            if state.receipt.status is not RecordStatus.ADMITTED:
                result = self._failure_result(
                    state,
                    code="preview_rejected",
                    outcome=WorkflowOutcome.REJECTED,
                    completed=(state.receipt.receipt_cid,),
                    expected=expected,
                )
                self._persist(result)
                return result
            try:
                self._verify_current(state)
                if self.control_service is not None and control_request is None:
                    raise PromptWorkflowAuthorizationError(
                        "configured control service requires a separate materialization request"
                    )
                authority_ref, selected_key, selected_lease, selected_fence = (
                    self._authorization_binding(
                        state,
                        control_request=control_request,
                        authorization=authorization,
                        idempotency_key=idempotency_key,
                        lease_id=lease_id,
                        fencing_epoch=fencing_epoch,
                    )
                )
            except (PromptWorkflowServiceError, ValueError, TypeError) as exc:
                code = (
                    "stale_roots"
                    if isinstance(exc, PromptWorkflowStaleRootError)
                    else "missing_authority"
                    if isinstance(exc, PromptWorkflowAuthorizationError)
                    else "invalid_materialization_request"
                )
                result = self._failure_result(
                    state,
                    code=code,
                    completed=(state.receipt.receipt_cid,),
                    expected=expected,
                    continuation=(
                        f"materialize:{state.receipt.receipt_cid}"
                        if code != "stale_roots"
                        else ""
                    ),
                )
                self._persist(result)
                return result

            fingerprint = prompt_workflow_cid(
                {
                    "schema": "prompt-materialization-stage@1",
                    "preview_receipt_cid": state.receipt.receipt_cid,
                    "authority_ref": authority_ref,
                    "idempotency_key": selected_key,
                    "lease_id": selected_lease,
                    "fencing_epoch": selected_fence,
                    "output_policy_cid": state.request.output_policy.content_id,
                }
            )
            scope = ("materialize", selected_key)
            prior_fingerprint = self._idempotency_fingerprints.get(scope)
            if prior_fingerprint is not None and prior_fingerprint != fingerprint:
                result = self._failure_result(
                    state,
                    code="idempotency_conflict",
                    completed=(state.receipt.receipt_cid,),
                    expected=expected,
                )
                self._persist(result)
                return result
            cached = self._materialization_by_key.get(fingerprint)
            if cached is not None:
                return cached.result
            self._idempotency_fingerprints[scope] = fingerprint

            control_receipts: tuple[str, ...] = ()
            if control_request is not None:
                try:
                    control_result, control_ref = self._validate_materialize_control(
                        state, control_request
                    )
                except (PromptWorkflowServiceError, ValueError, TypeError):
                    result = self._failure_result(
                        state,
                        code="materialization_control_rejected",
                        completed=(state.receipt.receipt_cid,),
                        expected=expected,
                        continuation=f"materialize:{state.receipt.receipt_cid}",
                    )
                    self._persist(result)
                    return result
                if not bool(getattr(control_result, "succeeded", False)):
                    result = self._failure_result(
                        state,
                        code="materialization_control_rejected",
                        completed=(state.receipt.receipt_cid,),
                        expected=expected,
                        continuation=f"materialize:{state.receipt.receipt_cid}",
                        control_receipts=(control_ref,) if control_ref else (),
                    )
                    self._persist(result)
                    return result
                control_receipts = (control_ref,) if control_ref else ()

            kinds = (
                ("markdown",)
                if state.request.output_policy.mode is OutputMode.MARKDOWN
                else ("duckdb",)
                if state.request.output_policy.mode is OutputMode.DUCKDB
                else ("markdown", "duckdb")
            )
            completed_projections: list[Mapping[str, Any]] = []
            failures: list[str] = []
            for kind in kinds:
                try:
                    projection = (
                        self._materialize_markdown(
                            state, idempotency_key=selected_key
                        )
                        if kind == "markdown"
                        else self._materialize_duckdb(
                            state,
                            authority_ref=authority_ref,
                            idempotency_key=selected_key,
                            fencing_epoch=selected_fence,
                        )
                    )
                except Exception as exc:
                    unavailable = (
                        "unavailable" in str(exc).lower()
                        or isinstance(exc, (ImportError, ModuleNotFoundError))
                    )
                    failures.append(
                        f"{kind}_capability_unavailable"
                        if unavailable
                        else f"{kind}_projection_failed"
                    )
                    continue
                completed_projections.append(projection)

            observed = tuple(
                sorted(str(item["effect"]) for item in completed_projections)
            )
            identities = tuple(
                item["identity"] for item in completed_projections
            )
            cursors = {
                str(item["kind"]): str(item["cursor"])
                for item in completed_projections
                if item["cursor"]
            }
            completed = (
                state.receipt.receipt_cid,
                *(
                    item["projection_cid"]
                    for item in completed_projections
                ),
            )
            if failures:
                outcome = (
                    WorkflowOutcome.PARTIAL
                    if completed_projections
                    else WorkflowOutcome.FAILED
                )
                result = self._failure_result(
                    state,
                    code="+".join(sorted(failures)),
                    outcome=outcome,
                    completed=completed,
                    continuation=f"materialize:{state.receipt.receipt_cid}",
                    expected=expected,
                    observed=observed,
                    identities=identities,
                    cursors=cursors,
                    control_receipts=control_receipts,
                )
                self._persist(result)
                return result

            reference = MaterializationReference(
                request_cid=state.request.request_cid,
                preview_receipt_cid=state.receipt.receipt_cid,
                plan_root_cid=state.receipt.plan_root_cid,
                repository_root=state.request.repository_root,
                output_root=state.request.output_policy.output_root,
                mode=state.request.output_policy.mode,
                projection_cids=tuple(
                    item["projection_cid"] for item in completed_projections
                ),
                revision=max(
                    int(item["revision"]) for item in completed_projections
                ),
                scan_cid=state.scan.scan_cid,
                program_root=state.request.program_root,
                policy_roots=state.receipt.policy_roots,
                catalog_root=state.receipt.catalog_root,
                output_policy_cid=state.request.output_policy.content_id,
                task_source_identities=identities,
                expected_effects=expected,
                observed_effects=observed,
                event_cursors=cursors,
                control_receipt_cid=(
                    control_receipts[0] if control_receipts else ""
                ),
                status=RecordStatus.READY,
                created_at_ms=int(self._clock_ms()),
                updated_at_ms=int(self._clock_ms()),
            )
            result = PromptWorkflowResult(
                request_cid=state.request.request_cid,
                outcome=WorkflowOutcome.MATERIALIZED,
                preview_receipt_cid=state.receipt.receipt_cid,
                materialization=reference,
                completed_stage_cids=(
                    state.receipt.receipt_cid,
                    reference.materialization_cid,
                    *(item["projection_cid"] for item in completed_projections),
                ),
                expected_effects=expected,
                observed_effects=observed,
                task_source_identities=identities,
                event_cursors=cursors,
                control_receipt_cids=control_receipts,
                status=RecordStatus.READY,
                created_at_ms=int(self._clock_ms()),
                updated_at_ms=int(self._clock_ms()),
            )
            try:
                self._persist(reference)
                self._persist(result)
            except PromptWorkflowReceiptError:
                return self._failure_result(
                    state,
                    code="receipt_projection_failed",
                    outcome=WorkflowOutcome.PARTIAL,
                    materialization=reference,
                    completed=result.completed_stage_cids,
                    continuation=f"persist:{reference.materialization_cid}",
                    expected=expected,
                    observed=observed,
                    identities=identities,
                    cursors=cursors,
                    control_receipts=control_receipts,
                )
            materialized = _PromptMaterializationState(
                preview=state,
                reference=reference,
                result=result,
            )
            self._materialization_by_key[fingerprint] = materialized
            self._materialization_by_cid[reference.materialization_cid] = materialized
            return result

    def _materialization_state(
        self, materialization_ref: str | MaterializationReference | PromptWorkflowResult
    ) -> _PromptMaterializationState:
        if isinstance(materialization_ref, PromptWorkflowResult):
            reference = materialization_ref.materialization
            ref = reference.materialization_cid if reference is not None else ""
        elif isinstance(materialization_ref, MaterializationReference):
            ref = materialization_ref.materialization_cid
        else:
            ref = str(materialization_ref or "")
        state = self._materialization_by_cid.get(ref)
        if state is None:
            raise PromptWorkflowReceiptError(
                "materialization receipt is unavailable for safe continuation"
            )
        return state

    def start(
        self,
        materialization_ref: str | MaterializationReference | PromptWorkflowResult,
        *,
        control_request: Any,
        supervisor_profile: str = "",
    ) -> PromptWorkflowResult:
        """Start a materialized task source through the existing control service."""

        with self._lock:
            materialized = self._materialization_state(materialization_ref)
            state = materialized.preview
            reference = materialized.reference
            expected_materialization = tuple(reference.expected_effects)
            observed_materialization = tuple(reference.observed_effects)
            if self.control_service is None or control_request is None:
                result = self._failure_result(
                    state,
                    code="missing_start_authority",
                    outcome=WorkflowOutcome.PARTIAL,
                    materialization=reference,
                    completed=materialized.result.completed_stage_cids,
                    continuation=f"start:{reference.materialization_cid}",
                    expected=expected_materialization,
                    observed=observed_materialization,
                    identities=reference.task_source_identities,
                    cursors=reference.event_cursors,
                    control_receipts=materialized.result.control_receipt_cids,
                )
                self._persist(result)
                return result
            try:
                from .control_contracts import Operation

                if getattr(control_request, "operation", None) is not Operation.START:
                    raise PromptWorkflowAuthorizationError(
                        "start control request uses the wrong operation"
                    )
                if str(control_request.repository_root) != state.request.repository_root:
                    raise PromptWorkflowStaleRootError(
                        "start repository root differs from materialization"
                    )
                if control_request.state_root != state.request.state_root:
                    raise PromptWorkflowStaleRootError(
                        "start state root differs from the workflow request"
                    )
                parameters = control_request.parameters
                exact_parameters = {
                    "materialization_ref": reference.materialization_cid,
                    "plan_root_cid": reference.plan_root_cid,
                    "task_source_root": prompt_workflow_cid(
                        {
                            "task_source_identities": _wire_value(
                                reference.task_source_identities
                            )
                        }
                    ),
                }
                for key, expected_value in exact_parameters.items():
                    observed_value = parameters.get(key)
                    if (
                        observed_value not in (None, "")
                        and observed_value != expected_value
                    ):
                        raise PromptWorkflowStaleRootError(
                            f"start {key} differs from materialization"
                        )
                profile = (
                    supervisor_profile
                    or str(parameters.get("supervisor_profile") or "")
                    or state.request.supervisor_profile
                )
                if not profile:
                    raise PromptWorkflowAuthorizationError(
                        "start requires a supervisor profile"
                    )
                self._verify_current(state)
            except (PromptWorkflowServiceError, ValueError, TypeError):
                result = self._failure_result(
                    state,
                    code="invalid_start_binding",
                    outcome=WorkflowOutcome.PARTIAL,
                    materialization=reference,
                    completed=materialized.result.completed_stage_cids,
                    continuation=f"start:{reference.materialization_cid}",
                    expected=expected_materialization,
                    observed=observed_materialization,
                    identities=reference.task_source_identities,
                    cursors=reference.event_cursors,
                    control_receipts=materialized.result.control_receipt_cids,
                )
                self._persist(result)
                return result

            key = str(getattr(control_request, "idempotency_key", "") or "")
            if not key:
                result = self._failure_result(
                    state,
                    code="missing_start_idempotency",
                    outcome=WorkflowOutcome.PARTIAL,
                    materialization=reference,
                    completed=materialized.result.completed_stage_cids,
                    continuation=f"start:{reference.materialization_cid}",
                    expected=expected_materialization,
                    observed=observed_materialization,
                    identities=reference.task_source_identities,
                    cursors=reference.event_cursors,
                    control_receipts=materialized.result.control_receipt_cids,
                )
                self._persist(result)
                return result
            fingerprint = prompt_workflow_cid(
                {
                    "schema": "prompt-start-stage@1",
                    "materialization_cid": reference.materialization_cid,
                    "lifecycle_request_cid": _reference_cid(
                        control_request.request_id, "lifecycle-request"
                    ),
                    "supervisor_profile": profile,
                    "state_root": control_request.state_root,
                }
            )
            scope = ("start", key)
            prior_fingerprint = self._idempotency_fingerprints.get(scope)
            if prior_fingerprint is not None and prior_fingerprint != fingerprint:
                result = self._failure_result(
                    state,
                    code="start_idempotency_conflict",
                    outcome=WorkflowOutcome.PARTIAL,
                    materialization=reference,
                    completed=materialized.result.completed_stage_cids,
                    continuation=f"start:{reference.materialization_cid}",
                    expected=expected_materialization,
                    observed=observed_materialization,
                    identities=reference.task_source_identities,
                    cursors=reference.event_cursors,
                    control_receipts=materialized.result.control_receipt_cids,
                )
                self._persist(result)
                return result
            cached = self._start_results.get(fingerprint)
            if cached is not None:
                return cached
            self._idempotency_fingerprints[scope] = fingerprint

            control_result = self.control_service.start(control_request)
            control_ref = _reference_cid(
                getattr(control_result, "audit_receipt_id", "")
                or getattr(control_result, "result_id", ""),
                "workflow-start-control-receipt",
            )
            start_expected = tuple(
                sorted(
                    f"start:{item.effect_id}"
                    for item in getattr(control_request, "expected_effects", ())
                )
            )
            start_observed = tuple(
                sorted(
                    f"start:{item.effect_id}"
                    for item in getattr(control_result, "effects", ())
                    if bool(getattr(item, "applied", False))
                )
            )
            all_expected = tuple(
                sorted({*expected_materialization, *start_expected})
            )
            all_observed = tuple(
                sorted({*observed_materialization, *start_observed})
            )
            control_receipts = tuple(
                sorted(
                    {
                        *materialized.result.control_receipt_cids,
                        *((control_ref,) if control_ref else ()),
                    }
                )
            )
            if (
                not bool(getattr(control_result, "succeeded", False))
                or set(start_observed) != set(start_expected)
            ):
                result = self._failure_result(
                    state,
                    code="partial_start",
                    outcome=WorkflowOutcome.PARTIAL,
                    materialization=reference,
                    completed=(
                        *materialized.result.completed_stage_cids,
                        *((control_ref,) if control_ref else ()),
                    ),
                    continuation=f"start:{reference.materialization_cid}",
                    expected=all_expected,
                    observed=all_observed,
                    identities=reference.task_source_identities,
                    cursors=reference.event_cursors,
                    control_receipts=control_receipts,
                )
                self._persist(result)
                return result

            data = getattr(control_result, "data", {}) or {}
            process_value = (
                data.get("process_identity_cid")
                or data.get("process_identity")
                or data.get("process_id")
                or data.get("pid")
                or ""
            )
            process_cid = _reference_cid(
                process_value, "supervisor-process-identity"
            )
            run = SupervisorRunReference(
                materialization_cid=reference.materialization_cid,
                plan_root_cid=reference.plan_root_cid,
                repository_root=reference.repository_root,
                state_root=control_request.state_root,
                supervisor_profile=profile,
                lifecycle_request_cid=_reference_cid(
                    control_request.request_id, "lifecycle-request"
                ),
                process_identity_cid=process_cid,
                status=RecordStatus.RUNNING,
                started_at_ms=int(self._clock_ms()),
                updated_at_ms=int(self._clock_ms()),
            )
            cursors = dict(reference.event_cursors)
            event_cursor = data.get("event_cursor")
            if event_cursor not in (None, ""):
                cursors["supervisor"] = self._cursor_reference(
                    event_cursor, "supervisor-event-cursor"
                )
            result = PromptWorkflowResult(
                request_cid=state.request.request_cid,
                outcome=WorkflowOutcome.STARTED,
                preview_receipt_cid=state.receipt.receipt_cid,
                materialization=reference,
                run=run,
                completed_stage_cids=(
                    *materialized.result.completed_stage_cids,
                    run.run_cid,
                    *((control_ref,) if control_ref else ()),
                ),
                expected_effects=all_expected,
                observed_effects=all_observed,
                task_source_identities=reference.task_source_identities,
                event_cursors=cursors,
                control_receipt_cids=control_receipts,
                status=RecordStatus.RUNNING,
                created_at_ms=int(self._clock_ms()),
                updated_at_ms=int(self._clock_ms()),
            )
            self._persist(run)
            self._persist(result)
            self._start_results[fingerprint] = result
            return result

    def bootstrap(
        self,
        request: PromptWorkflowRequest,
        *,
        preview_control_request: Any | None = None,
        materialize_control_request: Any | None = None,
        start_control_request: Any | None = None,
        authorization: Any | None = None,
        idempotency_key: str = "",
        lease_id: str = "",
        fencing_epoch: int | None = None,
        capabilities: Mapping[str, Any] | None = None,
        constraint_summaries: Mapping[str, Any] | None = None,
    ) -> PromptWorkflowResult:
        """Compose the independently receipted preview/materialize/start saga."""

        preview = self.preview(
            request,
            control_request=preview_control_request,
            capabilities=capabilities,
            constraint_summaries=constraint_summaries,
        )
        state = self._preview_state(preview)
        if preview.status is RecordStatus.REJECTED:
            result = self._failure_result(
                state,
                code="preview_rejected",
                outcome=WorkflowOutcome.REJECTED,
                completed=(preview.receipt_cid,),
                expected=preview.expected_materialization_effects,
            )
            self._persist(result)
            return result
        should_materialize = bool(
            request.materialize
            or materialize_control_request is not None
            or authorization is not None
            or idempotency_key
        )
        if not should_materialize:
            result = PromptWorkflowResult(
                request_cid=request.request_cid,
                outcome=WorkflowOutcome.PREVIEWED,
                preview_receipt_cid=preview.receipt_cid,
                completed_stage_cids=(preview.receipt_cid,),
                expected_effects=preview.expected_materialization_effects,
                status=RecordStatus.ADMITTED,
                created_at_ms=int(self._clock_ms()),
                updated_at_ms=int(self._clock_ms()),
            )
            self._persist(result)
            return result
        materialized = self.materialize(
            preview,
            control_request=materialize_control_request,
            authorization=authorization,
            idempotency_key=idempotency_key,
            lease_id=lease_id,
            fencing_epoch=fencing_epoch,
        )
        if materialized.outcome is not WorkflowOutcome.MATERIALIZED:
            return materialized
        should_start = bool(
            request.start_after_materialize or start_control_request is not None
        )
        if not should_start:
            return materialized
        if start_control_request is None:
            result = self._failure_result(
                state,
                code="missing_start_authority",
                outcome=WorkflowOutcome.PARTIAL,
                materialization=materialized.materialization,
                completed=materialized.completed_stage_cids,
                continuation=(
                    f"start:{materialized.materialization.materialization_cid}"
                    if materialized.materialization is not None
                    else ""
                ),
                expected=materialized.expected_effects,
                observed=materialized.observed_effects,
                identities=materialized.task_source_identities,
                cursors=materialized.event_cursors,
                control_receipts=materialized.control_receipt_cids,
            )
            self._persist(result)
            return result
        assert materialized.materialization is not None
        return self.start(
            materialized.materialization,
            control_request=start_control_request,
            supervisor_profile=request.supervisor_profile,
        )


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


PROMPT_WORKFLOW_CLI_EXIT_SUCCESS = 0
PROMPT_WORKFLOW_CLI_EXIT_FAILED = 1
PROMPT_WORKFLOW_CLI_EXIT_INVALID = 2
PROMPT_WORKFLOW_CLI_COMMANDS: Final[tuple[str, ...]] = (
    "workflow-preview",
    "workflow-create",
    "restart",
    "rescue-preview",
    "rescue",
)
_PROMPT_WORKFLOW_COMMAND_TO_OPERATION: Final[Mapping[str, str]] = {
    "workflow-preview": "workflow_preview",
    "workflow-create": "workflow_materialize",
    "restart": "restart",
    "rescue-preview": "rescue_preview",
    "rescue": "rescue",
}


class PromptWorkflowCLIError(ValueError):
    """A safe, user-correctable module-entry CLI error."""


def _optional_json_object(raw: str | None, *, noun: str) -> Mapping[str, Any]:
    if raw is None or raw == "":
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PromptWorkflowCLIError(f"{noun} must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise PromptWorkflowCLIError(f"{noun} must be a JSON object")
    return dict(value)


def _read_exactly_one_prompt(
    *,
    prompt: str | None,
    prompt_file: Path | None,
    stdin_stream: TextIO,
    stdin_flag: bool,
) -> PromptSource:
    """Resolve exactly one prompt source without logging the body."""

    provided = sum(
        1
        for item in (prompt is not None, prompt_file is not None, stdin_flag)
        if item
    )
    if provided != 1:
        raise PromptWorkflowCLIError(
            "provide exactly one of --prompt, --prompt-file, or --stdin "
            "(prefer --prompt-file/--stdin so sensitive text is not listed in "
            "process arguments)"
        )
    if prompt is not None:
        return PromptSource.inline(prompt)
    if prompt_file is not None:
        path = Path(prompt_file)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PromptWorkflowCLIError(
                f"unable to read prompt file: {path}"
            ) from exc
        if not text:
            raise PromptWorkflowCLIError("prompt file must be non-empty UTF-8 text")
        return PromptSource.file(path.name, text=text)
    if stdin_stream is None or getattr(stdin_stream, "isatty", lambda: False)():
        raise PromptWorkflowCLIError(
            "stdin prompt source requires piped non-empty UTF-8 text"
        )
    text = stdin_stream.read()
    if not text:
        raise PromptWorkflowCLIError("stdin prompt source must be non-empty")
    return PromptSource.stdin(text)


def build_prompt_workflow_arg_parser() -> argparse.ArgumentParser:
    """Build the provider-free ``python -m`` entry parser.

    Discovery and ``--help`` are side-effect free: no repository scan, provider,
    DuckDB connection, or supervisor process is started.
    """

    parser = argparse.ArgumentParser(
        prog="python -m ipfs_accelerate_py.agent_supervisor.prompt_workflow",
        description=(
            "Thin prompt-workflow entry over the shared agent control catalog. "
            "Prefer --prompt-file or --stdin for sensitive prompts. This entry "
            "does not import providers or mutate policy."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=PROMPT_WORKFLOW_CLI_COMMANDS,
        help="Catalog-aligned workflow or rescue command.",
    )
    parser.add_argument(
        "--request-json",
        help="Complete canonical OperationRequest JSON object.",
    )
    parser.add_argument(
        "--request-file",
        type=Path,
        help="File containing a complete canonical OperationRequest.",
    )
    parser.add_argument(
        "--parameters-json",
        help="Operation parameters as a JSON object (default: {}).",
    )
    parser.add_argument("--directory", help="Directory parameter for workflow ops.")
    parser.add_argument("--repository-root", help="Absolute allowlisted repository root.")
    parser.add_argument("--state-root", help="Absolute allowlisted supervisor state root.")
    parser.add_argument("--repository-id", help="Canonical repository identity.")
    parser.add_argument("--tree-id", help="Current repository tree identity.")
    parser.add_argument("--objective-id", help="Objective identity.")
    parser.add_argument("--objective-revision", help="Objective revision identity.")
    parser.add_argument("--policy-id", help="Control policy identity.")
    parser.add_argument("--policy-revision", help="Control policy revision.")
    parser.add_argument("--caller", help="Authenticated caller identity.")
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument(
        "--prompt",
        help=(
            "Inline prompt text. Prefer --prompt-file or --stdin so sensitive "
            "text is not visible in process listings."
        ),
    )
    prompt_source.add_argument(
        "--prompt-file",
        type=Path,
        help="Read the sole prompt body from a UTF-8 file.",
    )
    prompt_source.add_argument(
        "--stdin",
        action="store_true",
        help="Read the sole prompt body from stdin.",
    )
    parser.add_argument(
        "--output-mode",
        choices=tuple(item.value for item in OutputMode),
        help="Materialization projection mode: markdown, duckdb, or both.",
    )
    parser.add_argument(
        "--markdown-path",
        help="Root-relative Markdown task projection path.",
    )
    parser.add_argument(
        "--duckdb-path",
        help="Root-relative DuckDB task projection path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Proposal/preview path only; never mutates.",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Request start_after_materialize for workflow-create.",
    )
    parser.add_argument(
        "--authorization-json",
        help="Canonical AuthorizationDecision JSON object.",
    )
    parser.add_argument(
        "--authorization-file",
        type=Path,
        help="File containing a canonical AuthorizationDecision.",
    )
    parser.add_argument("--idempotency-key", help="Caller-chosen replay key.")
    parser.add_argument("--lease-id", help="Lease identity.")
    parser.add_argument(
        "--fencing-epoch",
        type=int,
        help="Non-negative fencing epoch.",
    )
    parser.add_argument(
        "--expected-effects-json",
        help="ExpectedEffect records as a JSON array.",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Emit a concise human summary instead of compact JSON.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Emit compact canonical JSON (default).",
    )
    return parser


def _merge_prompt_parameters(
    args: argparse.Namespace,
    *,
    stdin_stream: TextIO,
    parameters: MutableMapping[str, Any],
) -> None:
    if args.directory:
        if "directory" in parameters:
            raise PromptWorkflowCLIError(
                "directory was supplied both directly and in --parameters-json"
            )
        parameters["directory"] = str(args.directory)
    if args.output_mode:
        if "output_mode" in parameters:
            raise PromptWorkflowCLIError(
                "output_mode was supplied both directly and in --parameters-json"
            )
        parameters["output_mode"] = str(args.output_mode)
    if args.markdown_path:
        if "markdown_path" in parameters:
            raise PromptWorkflowCLIError(
                "markdown_path was supplied both directly and in --parameters-json"
            )
        parameters["markdown_path"] = str(args.markdown_path)
    if args.duckdb_path:
        if "duckdb_path" in parameters:
            raise PromptWorkflowCLIError(
                "duckdb_path was supplied both directly and in --parameters-json"
            )
        parameters["duckdb_path"] = str(args.duckdb_path)
    if args.start:
        parameters.setdefault("start_after_materialize", True)

    wants_prompt = bool(args.prompt is not None or args.prompt_file is not None or args.stdin)
    if not wants_prompt:
        return
    if "prompt_source" in parameters:
        raise PromptWorkflowCLIError(
            "prompt source was supplied both directly and in --parameters-json"
        )
    source = _read_exactly_one_prompt(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        stdin_stream=stdin_stream,
        stdin_flag=bool(args.stdin),
    )
    # Durable parameters carry only the body-free descriptor.
    parameters["prompt_source"] = {
        key: value
        for key, value in source.to_record().items()
        if key not in {"schema", "contract_version", "content_id"}
    }


def _render_human_summary(result: Mapping[str, Any]) -> str:
    status = result.get("status") or result.get("outcome") or "unknown"
    operation = result.get("operation") or ""
    request_id = result.get("request_id") or result.get("request_cid") or ""
    error = result.get("error") or {}
    code = ""
    if isinstance(error, Mapping):
        code = str(error.get("code") or error.get("error_code") or "")
    lines = [
        f"status={status}",
        f"operation={operation}" if operation else "",
        f"request_id={request_id}" if request_id else "",
        f"error={code}" if code else "",
    ]
    return "\n".join(line for line in lines if line)


def run_prompt_workflow_cli(
    argv: Optional[Sequence[str]] = None,
    *,
    stdin_stream: TextIO | None = None,
    stdout_stream: TextIO | None = None,
    stderr_stream: TextIO | None = None,
    control_service: Any | None = None,
) -> int:
    """Execute the thin module CLI and return a stable process exit code."""

    import io

    parser = build_prompt_workflow_arg_parser()
    out = stdout_stream or sys.stdout
    err = stderr_stream or sys.stderr
    stdin = stdin_stream or sys.stdin
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exited:
        # argparse writes usage to its own streams; map exit codes stably.
        code = exited.code
        if code in (None, 0):
            return PROMPT_WORKFLOW_CLI_EXIT_SUCCESS
        message = str(getattr(exited, "message", "") or "")
        if message:
            print(message, file=err)
        return PROMPT_WORKFLOW_CLI_EXIT_INVALID

    if args.command is None:
        parser.print_help(out)
        return PROMPT_WORKFLOW_CLI_EXIT_INVALID

    # Lazy import keeps ``import prompt_workflow`` free of control/process work.
    from .control_cli import AgentCLIError, run_agent_cli
    from .control_contracts import Operation

    try:
        operation_name = _PROMPT_WORKFLOW_COMMAND_TO_OPERATION[args.command]
        operation = Operation(operation_name)
        has_complete_request = bool(args.request_json or args.request_file)
        parameters_json: str | None = None
        if has_complete_request:
            convenience = any(
                (
                    args.prompt is not None,
                    args.prompt_file is not None,
                    bool(args.stdin),
                    args.directory,
                    args.output_mode,
                    args.markdown_path,
                    args.duckdb_path,
                    bool(args.start),
                    args.parameters_json,
                )
            )
            if convenience:
                raise PromptWorkflowCLIError(
                    "--request-json/--request-file cannot be combined with "
                    "prompt/parameter convenience flags"
                )
        else:
            parameters = dict(
                _optional_json_object(
                    args.parameters_json, noun="--parameters-json"
                )
            )
            _merge_prompt_parameters(
                args, stdin_stream=stdin, parameters=parameters
            )
            parameters_json = json.dumps(parameters, sort_keys=True)

        # Build a control_cli-compatible namespace so both entry points share one
        # request decoder, exit-code map, and allowlist factory.
        dry_run = bool(args.dry_run)
        if not has_complete_request and operation in {
            Operation.WORKFLOW_PREVIEW,
            Operation.RESCUE_PREVIEW,
        }:
            dry_run = True
        namespace = argparse.Namespace(
            agent_command=args.command,
            agent_operation=operation_name,
            request_json=args.request_json,
            request_file=args.request_file,
            parameters_json=parameters_json,
            repository_root=args.repository_root,
            state_root=args.state_root,
            repository_id=args.repository_id,
            tree_id=args.tree_id,
            objective_id=args.objective_id,
            objective_revision=args.objective_revision,
            policy_id=args.policy_id,
            policy_revision=args.policy_revision,
            caller=args.caller,
            path=None,
            limit=None,
            offset=None,
            cursor=None,
            event_cursor=None,
            task_header_prefix=None,
            target_id=None,
            service_id=None,
            task_id=None,
            bundle_id=None,
            lane_id=None,
            stream_id=None,
            receipt_id=None,
            cache_namespace=None,
            artifact_id=None,
            validation_id=None,
            reason=None,
            requested_state=None,
            expected_effects_json=(
                None if has_complete_request else args.expected_effects_json
            ),
            idempotency_key=(
                None if has_complete_request else args.idempotency_key
            ),
            authorization_json=(
                None if has_complete_request else args.authorization_json
            ),
            authorization_file=(
                None if has_complete_request else args.authorization_file
            ),
            lease_id=None if has_complete_request else (args.lease_id or None),
            fencing_epoch=(
                None if has_complete_request else args.fencing_epoch
            ),
            dry_run=False if has_complete_request else dry_run,
            max_items=None,
            max_bytes=None,
            max_text_bytes=None,
            timeout_ms=None,
            watch_count=1,
            watch_interval_ms=0,
            output_json=True,
        )
        capture = io.StringIO() if (args.human and not args.output_json) else None
        code = run_agent_cli(
            namespace,
            service=control_service,
            stdout=capture if capture is not None else out,
            stderr=err,
        )
        if capture is not None:
            raw = capture.getvalue().strip()
            if raw:
                try:
                    record = json.loads(raw.splitlines()[-1])
                except json.JSONDecodeError:
                    out.write(raw + "\n")
                else:
                    if isinstance(record, Mapping):
                        out.write(_render_human_summary(record) + "\n")
                    else:
                        out.write(raw + "\n")
        return int(code)
    except PromptWorkflowCLIError as exc:
        print(str(exc), file=err)
        return PROMPT_WORKFLOW_CLI_EXIT_INVALID
    except AgentCLIError as exc:
        print(str(exc), file=err)
        return PROMPT_WORKFLOW_CLI_EXIT_INVALID
    except Exception as exc:  # noqa: BLE001 - boundary exit mapping
        print(str(exc), file=err)
        return PROMPT_WORKFLOW_CLI_EXIT_FAILED


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m ipfs_accelerate_py.agent_supervisor.prompt_workflow`` entry."""

    return run_prompt_workflow_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())


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
    "PromptSupervisorService",
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
    "PromptWorkflowReceiptError",
    "PromptWorkflowRequest",
    "PromptWorkflowResult",
    "PromptWorkflowServiceError",
    "PromptWorkflowStaleRootError",
    "PromptWorkflowAuthorizationError",
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
    "PROMPT_WORKFLOW_CLI_COMMANDS",
    "PROMPT_WORKFLOW_CLI_EXIT_FAILED",
    "PROMPT_WORKFLOW_CLI_EXIT_INVALID",
    "PROMPT_WORKFLOW_CLI_EXIT_SUCCESS",
    "PromptWorkflowCLIError",
    "build_prompt_workflow_arg_parser",
    "canonical_prompt_workflow_bytes",
    "decode_prompt_workflow_request",
    "main",
    "prompt_workflow_cid",
    "run_prompt_workflow_cli",
]
