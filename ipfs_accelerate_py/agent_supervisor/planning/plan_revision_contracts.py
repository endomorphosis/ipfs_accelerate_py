"""Provider-free create/steer, plan-delta, and plan-revision contracts (PDR-020).

This module is a leaf serialization boundary.  It defines immutable,
content-addressed records for:

* :class:`PlanCreateRequest` and :class:`PlanSteerRequest` (proposal inputs);
* :class:`PlanDelta` with a closed delta operation vocabulary;
* :class:`PlanRevision` with immutable ancestry and population digests; and
* lifecycle-safe population helpers that refuse to edit or delete completed,
  accepted, or claimed history.

Records reject unknown fields, inline secrets, floats, non-canonical paths,
malformed CIDs/digests, stale root mismatches, and identity tampering at
construction.  No scanner, model provider, task-source, or supervisor runtime
is imported.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final, Iterable, Mapping as TypingMapping

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


# ---------------------------------------------------------------------------
# Version, schemas, bounds
# ---------------------------------------------------------------------------

PLAN_REVISION_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = PLAN_REVISION_CONTRACT_VERSION
SCHEMA_VERSION: Final[int] = PLAN_REVISION_CONTRACT_VERSION
PLAN_REVISION_CONTRACTS_INTERFACE: Final[str] = "PlanRevisionContracts@1"

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
PLAN_AUTHORITY_ROOTS_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/plan-authority-roots@1"
)
PLAN_CREATE_REQUEST_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/plan-create-request@1"
PLAN_STEER_REQUEST_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/plan-steer-request@1"
PLAN_DELTA_ITEM_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/plan-delta-item@1"
PLAN_DELTA_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/plan-delta@1"
PLAN_REVISION_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/plan-revision@1"
PLAN_POPULATION_DIGEST_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/plan-population-digest@1"
)
PLAN_BUDGET_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/plan-request-budget@1"
PLAN_RESOURCE_CONTRACT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/plan-resource-contract@1"
)
PLAN_LEASE_CONTRACT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/plan-lease-contract@1"
PLAN_WORKTREE_CONTRACT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/plan-worktree-contract@1"
)
PLAN_MERGE_STRATEGY_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/plan-merge-strategy@1"
PLAN_VALIDATION_NODE_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/plan-validation-dag-node@1"
)
PLAN_CONFLICT_CONTRACT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/plan-conflict-contract@1"
)
PLAN_PROVIDER_CONTRACT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/plan-provider-contract@1"
)
PLAN_RETRY_CONTRACT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/plan-retry-contract@1"
PLAN_COMPLETION_RULE_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/plan-completion-rule@1"
)

MAX_RECORD_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_PATH_BYTES: Final[int] = 1_024
MAX_REFERENCE_COUNT: Final[int] = 4_096
MAX_DELTA_ITEMS: Final[int] = 4_096
MAX_GOALS: Final[int] = 1_024
MAX_TASKS: Final[int] = 4_096
MAX_DEPTH: Final[int] = 32
MAX_INT: Final[int] = 2**63 - 1

_CID_PREFIX = b"\x01\xa9\x02\x12\x20"
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
_BODY_MARKERS = frozenset(
    {
        "body",
        "prompt",
        "prompt_body",
        "prompt_text",
        "raw_log",
        "source_body",
        "source_text",
        "transcript",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PlanRevisionContractError(ContractValidationError):
    """Base error for malformed plan-revision contracts."""


class PlanRevisionBoundsError(PlanRevisionContractError):
    """A count, byte, depth, or integer limit exceeds a hard bound."""


class PlanRevisionIdentityError(PlanRevisionContractError):
    """A claimed content identity does not match canonical identity bytes."""


class PlanRevisionPathError(PlanRevisionContractError):
    """A root or repository-relative path is non-canonical or escapes scope."""


class PlanRevisionSecretError(PlanRevisionContractError):
    """A durable contract contains inline secret-bearing material."""


class PlanRevisionStaleRootError(PlanRevisionContractError):
    """A bound root no longer matches the expected authority snapshot."""


class PlanRevisionLifecycleError(PlanRevisionContractError):
    """A delta would edit or delete completed, accepted, or claimed history."""


class PlanRevisionAuthorityError(PlanRevisionContractError):
    """Authority widening or identity tampering was attempted."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class PlanOrigin(str, Enum):
    """Whether a revision was produced by create or steer."""

    CREATE = "create"
    STEER = "steer"


class TaskSourceKind(str, Enum):
    MARKDOWN = "markdown"
    DUCKDB = "duckdb"
    BOTH = "both"


class DirtyTreePolicy(str, Enum):
    REQUIRE_CLEAN = "require_clean"
    ALLOW_DIRTY = "allow_dirty"
    OBSERVE_AND_BIND = "observe_and_bind"


class FallbackPolicy(str, Enum):
    FAIL_CLOSED = "fail_closed"
    DETERMINISTIC_SUBSTITUTE = "deterministic_substitute"
    DEGRADE_WITH_DEBT = "degrade_with_debt"


class PlanDeltaOperation(str, Enum):
    """Closed delta language.  History-erasing ops are intentionally absent."""

    ADD_GOAL = "add_goal"
    SUPERSEDE_GOAL = "supersede_goal"
    AMEND_UNSTARTED_GOAL = "amend_unstarted_goal"
    ADD_TASK = "add_task"
    SUPERSEDE_UNSTARTED_TASK = "supersede_unstarted_task"
    SPLIT_UNSTARTED_TASK = "split_unstarted_task"
    COALESCE_UNSTARTED_TASKS = "coalesce_unstarted_tasks"
    REWIRE_UNSTARTED_DEPENDENCY = "rewire_unstarted_dependency"
    BLOCK_UNSTARTED_TASK = "block_unstarted_task"
    UNBLOCK_TASK = "unblock_task"
    REPRIORITIZE_UNSTARTED_TASK = "reprioritize_unstarted_task"
    ASSIGN_PARALLEL_CONTRACT = "assign_parallel_contract"
    ATTACH_EVIDENCE = "attach_evidence"
    RECORD_UNCERTAINTY = "record_uncertainty"
    REQUEST_LIFECYCLE_ACTION = "request_lifecycle_action"


class DeltaEffectClass(str, Enum):
    MATERIALIZABLE_NOW = "materializable_now"
    DEFERRED = "deferred"
    LIFECYCLE_REQUEST = "lifecycle_request"
    EVIDENCE_ONLY = "evidence_only"


class PopulationKind(str, Enum):
    ADDED = "added"
    SUPERSEDED = "superseded"
    RETAINED = "retained"
    DEFERRED = "deferred"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    ACCEPTED = "accepted"
    BLOCKED = "blocked"
    FAILED = "failed"
    UNSTARTED = "unstarted"
    RUNNING = "running"
    SETTLING = "settling"


class LifecycleState(str, Enum):
    """Task/goal lifecycle states relevant to immutability gates."""

    PROPOSED = "proposed"
    ADMITTED = "admitted"
    READY = "ready"
    UNSTARTED = "unstarted"
    BLOCKED = "blocked"
    CLAIMED = "claimed"
    RUNNING = "running"
    SETTLING = "settling"
    COMPLETED = "completed"
    ACCEPTED = "accepted"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"
    QUARANTINED = "quarantined"


# States whose exact specification and history cannot be edited or deleted.
_IMMUTABLE_HISTORY_STATES: Final[frozenset[LifecycleState]] = frozenset(
    {
        LifecycleState.CLAIMED,
        LifecycleState.RUNNING,
        LifecycleState.SETTLING,
        LifecycleState.COMPLETED,
        LifecycleState.ACCEPTED,
    }
)

# Delta ops that may target an already-claimed/running task only as a
# non-mutating successor or lifecycle *request*, never an in-place edit.
_CLAIMED_SAFE_OPERATIONS: Final[frozenset[PlanDeltaOperation]] = frozenset(
    {
        PlanDeltaOperation.ATTACH_EVIDENCE,
        PlanDeltaOperation.RECORD_UNCERTAINTY,
        PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION,
        PlanDeltaOperation.ADD_TASK,
        PlanDeltaOperation.ADD_GOAL,
        PlanDeltaOperation.BLOCK_UNSTARTED_TASK,
        PlanDeltaOperation.UNBLOCK_TASK,
    }
)

# Ops that rewrite an existing target record in place (forbidden on history).
_MUTATING_TARGET_OPERATIONS: Final[frozenset[PlanDeltaOperation]] = frozenset(
    {
        PlanDeltaOperation.SUPERSEDE_GOAL,
        PlanDeltaOperation.AMEND_UNSTARTED_GOAL,
        PlanDeltaOperation.SUPERSEDE_UNSTARTED_TASK,
        PlanDeltaOperation.SPLIT_UNSTARTED_TASK,
        PlanDeltaOperation.COALESCE_UNSTARTED_TASKS,
        PlanDeltaOperation.REWIRE_UNSTARTED_DEPENDENCY,
        PlanDeltaOperation.BLOCK_UNSTARTED_TASK,
        PlanDeltaOperation.REPRIORITIZE_UNSTARTED_TASK,
        PlanDeltaOperation.ASSIGN_PARALLEL_CONTRACT,
    }
)


class MergeStrategyKind(str, Enum):
    SERIAL = "serial"
    MERGE_TRAIN = "merge_train"
    REBASE_THEN_MERGE = "rebase_then_merge"
    FAST_FORWARD_ONLY = "fast_forward_only"
    REVIEW_ONLY = "review_only"


class OutputEffect(str, Enum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    WRITE = "write"


class DependencyEdgeKind(str, Enum):
    DATA = "data"
    CODE = "code"
    POLICY = "policy"
    PROOF = "proof"
    VALIDATION = "validation"
    MERGE = "merge"
    LIFECYCLE = "lifecycle"


class CompletionAuthority(str, Enum):
    SUPERVISOR = "supervisor"
    HUMAN = "human"
    PROOF_GATE = "proof_gate"
    VALIDATION_GATE = "validation_gate"
    FORBIDDEN = "forbidden"


# ---------------------------------------------------------------------------
# Primitive validators
# ---------------------------------------------------------------------------


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise PlanRevisionContractError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise PlanRevisionContractError(f"{field_name} must be a string")
    if value != value.strip():
        raise PlanRevisionContractError(
            f"{field_name} has leading or trailing whitespace"
        )
    if required and not value:
        raise PlanRevisionContractError(f"{field_name} must not be empty")
    if "\x00" in value:
        raise PlanRevisionContractError(f"{field_name} must not contain NUL")
    if len(value.encode("utf-8")) > limit:
        raise PlanRevisionBoundsError(f"{field_name} exceeds its byte bound")
    if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
        raise PlanRevisionSecretError(
            f"{field_name} contains inline secret material"
        )
    return value


def _identifier(value: Any, field_name: str, *, required: bool = True) -> str:
    value = _text(value, field_name, required=required)
    if not value and not required:
        return ""
    if any(char.isspace() for char in value):
        raise PlanRevisionContractError(
            f"{field_name} must be an opaque compact identifier"
        )
    return value


def _bounded_int(
    value: Any,
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_INT,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanRevisionContractError(f"{field_name} must be a finite integer")
    if value < minimum or value > maximum:
        raise PlanRevisionBoundsError(
            f"{field_name} is outside the supported bound"
        )
    return value


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise PlanRevisionContractError(f"{field_name} must be boolean")
    return value


def _absolute_path(value: Any, field_name: str) -> str:
    result = _text(value, field_name)
    if "\\" in result or not result.startswith("/"):
        raise PlanRevisionPathError(
            f"{field_name} must be a canonical absolute path"
        )
    import posixpath

    normalized = posixpath.normpath(result)
    if normalized == "/":
        raise PlanRevisionPathError(
            f"{field_name} must not be the filesystem root"
        )
    if normalized != result or ".." in PurePosixPath(result).parts:
        raise PlanRevisionPathError(f"{field_name} is not canonical")
    return result


def _relative_path(value: Any, field_name: str) -> str:
    path = _text(value, field_name, limit=MAX_PATH_BYTES)
    candidate = PurePosixPath(path)
    if (
        "\\" in path
        or candidate.is_absolute()
        or ".." in candidate.parts
        or path in {".", ""}
    ):
        raise PlanRevisionPathError(
            f"{field_name} must be a relative repository path"
        )
    if candidate.as_posix() != path:
        raise PlanRevisionPathError(f"{field_name} is not canonical")
    return path


def _validate_cid(value: Any, field_name: str, *, required: bool = True) -> str:
    result = _text(value, field_name, required=required)
    if not result and not required:
        return ""
    if not _CID_RE.fullmatch(result):
        raise PlanRevisionIdentityError(
            f"{field_name} must be a canonical CIDv1"
        )
    try:
        padding = "=" * ((8 - (len(result) - 1) % 8) % 8)
        decoded = base64.b32decode(
            (result[1:].upper() + padding).encode("ascii")
        )
    except (ValueError, UnicodeError) as exc:
        raise PlanRevisionIdentityError(f"{field_name} is malformed") from exc
    if len(decoded) != len(_CID_PREFIX) + 32 or not decoded.startswith(
        _CID_PREFIX
    ):
        raise PlanRevisionIdentityError(
            f"{field_name} must use CIDv1 dag-json with sha2-256"
        )
    canonical = (
        "b" + base64.b32encode(decoded).decode("ascii").rstrip("=").lower()
    )
    if canonical != result:
        raise PlanRevisionIdentityError(f"{field_name} is not canonical")
    return result


def _identity(value: Any, field_name: str, *, required: bool = True) -> str:
    result = _text(value, field_name, required=required)
    if not result and not required:
        return ""
    # Only treat as CID when the full canonical CIDv1 pattern matches.
    # A bare "b" or short tokens must not enter CID validation.
    if _CID_RE.fullmatch(result):
        return _validate_cid(result, field_name)
    if _DIGEST_RE.fullmatch(result):
        return result
    # Compact identifiers: namespaced keys, operation names, enum-like tokens.
    if not any(char.isspace() for char in result) and all(
        char.isalnum() or char in "._:-@/" for char in result
    ):
        return result
    raise PlanRevisionIdentityError(
        f"{field_name} must be a CIDv1, sha256 digest, or compact identifier"
    )


def _ids(
    values: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCE_COUNT,
    as_cid: bool = False,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PlanRevisionContractError(
            f"{field_name} must be a sequence of identifiers"
        )
    else:
        raw = values
    if len(raw) > limit:
        raise PlanRevisionBoundsError(f"{field_name} exceeds its item bound")
    result_list: list[str] = []
    seen: set[str] = set()
    for value in raw:
        item = (
            _validate_cid(value, field_name)
            if as_cid
            else _identity(value, field_name)
        )
        if item not in seen:
            seen.add(item)
            result_list.append(item)
    result = tuple(result_list if preserve_order else sorted(result_list))
    if required and not result:
        raise PlanRevisionContractError(f"{field_name} must not be empty")
    return result


def _paths(
    values: Any, field_name: str, *, limit: int = MAX_REFERENCE_COUNT
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PlanRevisionContractError(
            f"{field_name} must be a sequence of paths"
        )
    else:
        raw = values
    if len(raw) > limit:
        raise PlanRevisionBoundsError(f"{field_name} exceeds its item bound")
    return tuple(sorted({_relative_path(value, field_name) for value in raw}))


def _secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SECRET_KEYS or any(
        marker in normalized
        for marker in ("password", "private_key", "access_token", "api_key")
    )


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    if isinstance(value, float):
        raise PlanRevisionContractError(
            f"{field_name} may not contain floating-point values"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PlanRevisionContractError(
                    f"{field_name} has a non-string key"
                )
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS or _secret_key(normalized):
                raise PlanRevisionSecretError(
                    f"{field_name} may not contain secrets or source bodies"
                )
            _assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise PlanRevisionContractError(
            f"{field_name} may not contain binary bodies"
        )


def _freeze_mapping(
    value: Any,
    field_name: str,
    *,
    max_items: int = MAX_REFERENCE_COUNT,
    max_depth: int = MAX_DEPTH,
) -> TypingMapping[str, Any]:
    seen = 0

    def visit(item: Any, depth: int) -> Any:
        nonlocal seen
        seen += 1
        if seen > max_items:
            raise PlanRevisionBoundsError(
                f"{field_name} exceeds item-count bound"
            )
        if depth > max_depth:
            raise PlanRevisionBoundsError(f"{field_name} exceeds depth bound")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            raise PlanRevisionContractError(
                f"{field_name} must not contain floats"
            )
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, str):
            return _text(item, field_name, required=False)
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key in sorted(item):
                normalized = _text(str(key), f"{field_name} key")
                if _secret_key(normalized) or normalized.lower().replace(
                    "-", "_"
                ) in _BODY_MARKERS:
                    raise PlanRevisionSecretError(
                        f"{field_name} contains forbidden secret-bearing field"
                    )
                result[normalized] = visit(item[key], depth + 1)
            return MappingProxyType(result)
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return tuple(visit(member, depth + 1) for member in item)
        raise PlanRevisionContractError(
            f"{field_name} contains unsupported type {type(item).__name__}"
        )

    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise PlanRevisionContractError(f"{field_name} must be a mapping")
    frozen = visit(value, 0)
    assert isinstance(frozen, Mapping)
    return frozen


def _bounded(record: CanonicalContract, name: str) -> None:
    payload = record.to_dict()
    _assert_body_free(payload, name)
    if len(canonical_json_bytes(payload)) > MAX_RECORD_BYTES:
        raise PlanRevisionBoundsError(
            f"{name} exceeds its serialized byte bound"
        )


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    supplied = payload.get("content_id", payload.get("cid", ""))
    if supplied not in (None, ""):
        if not isinstance(supplied, str) or supplied != record.content_id:
            raise PlanRevisionIdentityError(
                "stored content identity does not match the canonical record"
            )


def _decode_fields(
    payload: Mapping[str, Any],
    schema: str,
    fields: Sequence[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise PlanRevisionContractError(f"{name} has an unsupported schema")
    if payload.get("contract_version") not in (
        None,
        PLAN_REVISION_CONTRACT_VERSION,
    ):
        raise PlanRevisionContractError(
            f"{name} has an unsupported contract version"
        )
    allowed = set(fields) | {"schema", "contract_version", "content_id", "cid"}
    unknown = set(payload).difference(allowed)
    if unknown:
        raise PlanRevisionContractError(
            f"{name} contains unsupported fields: "
            + ", ".join(sorted(unknown))
        )
    _assert_body_free(payload, name)
    return {
        field_name: payload[field_name]
        for field_name in fields
        if field_name in payload
    }


def _decode_nested(
    value: Any,
    cls: type[CanonicalContract],
    field_name: str,
) -> CanonicalContract:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        if "schema" in value:
            return cls.from_dict(value)  # type: ignore[attr-defined, return-value]
        return cls(**value)  # type: ignore[arg-type, call-arg, return-value]
    raise PlanRevisionContractError(f"{field_name} must be {cls.__name__}")


def _decode_sequence(
    values: Any,
    cls: type[CanonicalContract],
    field_name: str,
    *,
    limit: int,
    required: bool = False,
) -> tuple[CanonicalContract, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        raw = values
    else:
        raise PlanRevisionContractError(f"{field_name} must be a sequence")
    if len(raw) > limit:
        raise PlanRevisionBoundsError(f"{field_name} exceeds its item bound")
    items: list[CanonicalContract] = []
    seen: set[str] = set()
    for item in raw:
        decoded = _decode_nested(item, cls, field_name)
        if decoded.content_id not in seen:
            seen.add(decoded.content_id)
            items.append(decoded)
    result = tuple(sorted(items, key=lambda record: record.content_id))
    if required and not result:
        raise PlanRevisionContractError(f"{field_name} must not be empty")
    return result


def plan_revision_cid(value: Any) -> str:
    """Return a CIDv1 identity for an arbitrary JSON-compatible value."""

    return content_identity(value)


# ---------------------------------------------------------------------------
# Nested contract records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanAuthorityRoots(CanonicalContract):
    """Exact roots whose drift invalidates a create/steer or revision record."""

    SCHEMA: ClassVar[str] = PLAN_AUTHORITY_ROOTS_SCHEMA

    repository_id: str
    repository_root_cid: str
    dirty_worktree_root: str
    task_source_id: str
    task_source_revision: str
    policy_root: str
    intent_ir_root: str
    legal_ir_root: str
    security_ir_root: str
    program_root: str
    capability_catalog_root: str
    provider_catalog_root: str
    usage_policy_root: str
    configuration_root: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        for name in (
            "repository_root_cid",
            "dirty_worktree_root",
            "task_source_id",
            "task_source_revision",
            "policy_root",
            "intent_ir_root",
            "legal_ir_root",
            "security_ir_root",
            "program_root",
            "capability_catalog_root",
            "provider_catalog_root",
            "usage_policy_root",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "configuration_root",
            _identity(self.configuration_root, "configuration_root", required=False),
        )
        _bounded(self, "plan authority roots")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "repository_root_cid": self.repository_root_cid,
            "dirty_worktree_root": self.dirty_worktree_root,
            "task_source_id": self.task_source_id,
            "task_source_revision": self.task_source_revision,
            "policy_root": self.policy_root,
            "intent_ir_root": self.intent_ir_root,
            "legal_ir_root": self.legal_ir_root,
            "security_ir_root": self.security_ir_root,
            "program_root": self.program_root,
            "capability_catalog_root": self.capability_catalog_root,
            "provider_catalog_root": self.provider_catalog_root,
            "usage_policy_root": self.usage_policy_root,
            "configuration_root": self.configuration_root,
        }

    def matches(self, other: "PlanAuthorityRoots") -> bool:
        return self.content_id == other.content_id

    def require_current(self, expected: "PlanAuthorityRoots") -> None:
        if not self.matches(expected):
            raise PlanRevisionStaleRootError(
                "authority roots are stale relative to the expected snapshot"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanAuthorityRoots":
        names = (
            "repository_id",
            "repository_root_cid",
            "dirty_worktree_root",
            "task_source_id",
            "task_source_revision",
            "policy_root",
            "intent_ir_root",
            "legal_ir_root",
            "security_ir_root",
            "program_root",
            "capability_catalog_root",
            "provider_catalog_root",
            "usage_policy_root",
            "configuration_root",
        )
        value = cls(**_decode_fields(payload, cls.SCHEMA, names, "plan authority roots"))
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanRequestBudget(CanonicalContract):
    """Integer-unit budgets for scan/analysis/model/repair stages."""

    SCHEMA: ClassVar[str] = PLAN_BUDGET_SCHEMA

    max_goals: int = 64
    max_tasks: int = 256
    max_graph_depth: int = 16
    max_output_paths: int = 1_024
    max_ready_width: int = 1
    max_repair_rounds: int = 2
    max_scan_bytes: int = 64 * 1024 * 1024
    max_analysis_operations: int = 64
    max_evidence_items: int = 512
    max_logic_families: int = 16
    max_model_calls: int = 2
    max_latency_ms: int = 600_000
    max_provider_tokens: int = 32_768
    max_cost_micros: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name == "SCHEMA":
                continue
            object.__setattr__(
                self,
                name,
                _bounded_int(getattr(self, name), name, minimum=0),
            )
        if self.max_ready_width < 1:
            raise PlanRevisionContractError(
                "max_ready_width must be at least 1 (serial default)"
            )
        _bounded(self, "plan request budget")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "SCHEMA"
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanRequestBudget":
        names = tuple(
            name for name in cls.__dataclass_fields__ if name != "SCHEMA"
        )
        value = cls(
            **_decode_fields(payload, cls.SCHEMA, names, "plan request budget")
        )
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanPopulationDigest(CanonicalContract):
    """Content-addressed population of task/goal CIDs for one lifecycle slice."""

    SCHEMA: ClassVar[str] = PLAN_POPULATION_DIGEST_SCHEMA

    kind: PopulationKind
    member_cids: tuple[str, ...] = ()
    digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, PopulationKind, "kind")
        )
        members = _ids(
            self.member_cids, "member_cids", as_cid=False, limit=MAX_TASKS
        )
        object.__setattr__(self, "member_cids", members)
        computed = plan_revision_cid(
            {"kind": self.kind.value, "member_cids": list(members)}
        )
        if self.digest:
            claimed = _identity(self.digest, "digest")
            if claimed != computed:
                raise PlanRevisionIdentityError(
                    "population digest does not match member set"
                )
            object.__setattr__(self, "digest", claimed)
        else:
            object.__setattr__(self, "digest", computed)
        _bounded(self, "plan population digest")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "kind": self.kind.value,
            "member_cids": list(self.member_cids),
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanPopulationDigest":
        values = _decode_fields(
            payload,
            cls.SCHEMA,
            ("kind", "member_cids", "digest"),
            "plan population digest",
        )
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanResourceContract(CanonicalContract):
    """Projected resource requirements; zero means conservative unknown."""

    SCHEMA: ClassVar[str] = PLAN_RESOURCE_CONTRACT_SCHEMA

    resource_class: str = "cpu-small"
    resource_stage: str = "analysis"
    cpu_slots: int = 1
    process_slots: int = 1
    memory_bytes: int = 0
    gpu_memory_bytes: int = 0
    disk_bytes: int = 0
    wall_time_ms: int = 0
    child_process_limit: int = 0
    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_class",
            _identifier(self.resource_class, "resource_class"),
        )
        object.__setattr__(
            self,
            "resource_stage",
            _identifier(self.resource_stage, "resource_stage"),
        )
        for name in (
            "cpu_slots",
            "process_slots",
            "memory_bytes",
            "gpu_memory_bytes",
            "disk_bytes",
            "wall_time_ms",
            "child_process_limit",
        ):
            object.__setattr__(
                self, name, _bounded_int(getattr(self, name), name)
            )
        if self.cpu_slots < 1 or self.process_slots < 1:
            raise PlanRevisionContractError(
                "cpu_slots and process_slots must be at least 1"
            )
        object.__setattr__(
            self,
            "required_capabilities",
            _ids(self.required_capabilities, "required_capabilities"),
        )
        _bounded(self, "plan resource contract")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "resource_class": self.resource_class,
            "resource_stage": self.resource_stage,
            "cpu_slots": self.cpu_slots,
            "process_slots": self.process_slots,
            "memory_bytes": self.memory_bytes,
            "gpu_memory_bytes": self.gpu_memory_bytes,
            "disk_bytes": self.disk_bytes,
            "wall_time_ms": self.wall_time_ms,
            "child_process_limit": self.child_process_limit,
            "required_capabilities": list(self.required_capabilities),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanResourceContract":
        names = (
            "resource_class",
            "resource_stage",
            "cpu_slots",
            "process_slots",
            "memory_bytes",
            "gpu_memory_bytes",
            "disk_bytes",
            "wall_time_ms",
            "child_process_limit",
            "required_capabilities",
        )
        value = cls(
            **_decode_fields(payload, cls.SCHEMA, names, "plan resource contract")
        )
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanProviderContract(CanonicalContract):
    """Provider envelope; empty requirements mean no remote capacity."""

    SCHEMA: ClassVar[str] = PLAN_PROVIDER_CONTRACT_SCHEMA

    provider_requirement: str = ""
    endpoint_policy_class: str = "none"
    context_tokens: int = 0
    output_token_budget: int = 0
    quota_units: int = 0
    cost_limit_micros: int = 0
    max_provider_latency_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_requirement",
            _text(
                self.provider_requirement,
                "provider_requirement",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "endpoint_policy_class",
            _identifier(self.endpoint_policy_class, "endpoint_policy_class"),
        )
        for name in (
            "context_tokens",
            "output_token_budget",
            "quota_units",
            "cost_limit_micros",
            "max_provider_latency_ms",
        ):
            object.__setattr__(
                self, name, _bounded_int(getattr(self, name), name)
            )
        _bounded(self, "plan provider contract")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "provider_requirement": self.provider_requirement,
            "endpoint_policy_class": self.endpoint_policy_class,
            "context_tokens": self.context_tokens,
            "output_token_budget": self.output_token_budget,
            "quota_units": self.quota_units,
            "cost_limit_micros": self.cost_limit_micros,
            "max_provider_latency_ms": self.max_provider_latency_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanProviderContract":
        names = (
            "provider_requirement",
            "endpoint_policy_class",
            "context_tokens",
            "output_token_budget",
            "quota_units",
            "cost_limit_micros",
            "max_provider_latency_ms",
        )
        value = cls(
            **_decode_fields(payload, cls.SCHEMA, names, "plan provider contract")
        )
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanLeaseContract(CanonicalContract):
    """Lease scope, duration, fence, and owner identity rules."""

    SCHEMA: ClassVar[str] = PLAN_LEASE_CONTRACT_SCHEMA

    lease_scope: str = "task"
    lease_duration_ms: int = 0
    fencing_epoch: int = 0
    owner_identity_rule: str = "lane-owner"
    heartbeat_interval_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "lease_scope", _identifier(self.lease_scope, "lease_scope")
        )
        object.__setattr__(
            self,
            "owner_identity_rule",
            _identifier(self.owner_identity_rule, "owner_identity_rule"),
        )
        for name in (
            "lease_duration_ms",
            "fencing_epoch",
            "heartbeat_interval_ms",
        ):
            object.__setattr__(
                self, name, _bounded_int(getattr(self, name), name)
            )
        _bounded(self, "plan lease contract")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "lease_scope": self.lease_scope,
            "lease_duration_ms": self.lease_duration_ms,
            "fencing_epoch": self.fencing_epoch,
            "owner_identity_rule": self.owner_identity_rule,
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanLeaseContract":
        names = (
            "lease_scope",
            "lease_duration_ms",
            "fencing_epoch",
            "owner_identity_rule",
            "heartbeat_interval_ms",
        )
        value = cls(
            **_decode_fields(payload, cls.SCHEMA, names, "plan lease contract")
        )
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanRetryContract(CanonicalContract):
    """Retry counts, backoff, and compensation policy."""

    SCHEMA: ClassVar[str] = PLAN_RETRY_CONTRACT_SCHEMA

    max_retries: int = 0
    retryable_classes: tuple[str, ...] = ()
    backoff_ms: int = 0
    circuit_breaker_threshold: int = 0
    compensation_policy: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_retries", _bounded_int(self.max_retries, "max_retries")
        )
        object.__setattr__(
            self,
            "retryable_classes",
            _ids(self.retryable_classes, "retryable_classes"),
        )
        object.__setattr__(
            self, "backoff_ms", _bounded_int(self.backoff_ms, "backoff_ms")
        )
        object.__setattr__(
            self,
            "circuit_breaker_threshold",
            _bounded_int(
                self.circuit_breaker_threshold, "circuit_breaker_threshold"
            ),
        )
        object.__setattr__(
            self,
            "compensation_policy",
            _identifier(self.compensation_policy, "compensation_policy"),
        )
        _bounded(self, "plan retry contract")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "max_retries": self.max_retries,
            "retryable_classes": list(self.retryable_classes),
            "backoff_ms": self.backoff_ms,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "compensation_policy": self.compensation_policy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanRetryContract":
        names = (
            "max_retries",
            "retryable_classes",
            "backoff_ms",
            "circuit_breaker_threshold",
            "compensation_policy",
        )
        value = cls(
            **_decode_fields(payload, cls.SCHEMA, names, "plan retry contract")
        )
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanWorktreeContract(CanonicalContract):
    """Worktree policy and expected base/merge-target revisions."""

    SCHEMA: ClassVar[str] = PLAN_WORKTREE_CONTRACT_SCHEMA

    policy: str = "none"
    expected_base_revision: str = ""
    expected_merge_target: str = ""
    isolation_required: bool = False
    max_worktree_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", _identifier(self.policy, "policy"))
        object.__setattr__(
            self,
            "expected_base_revision",
            _identity(
                self.expected_base_revision,
                "expected_base_revision",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "expected_merge_target",
            _identity(
                self.expected_merge_target,
                "expected_merge_target",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "isolation_required",
            _boolean(self.isolation_required, "isolation_required"),
        )
        object.__setattr__(
            self,
            "max_worktree_bytes",
            _bounded_int(self.max_worktree_bytes, "max_worktree_bytes"),
        )
        _bounded(self, "plan worktree contract")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "policy": self.policy,
            "expected_base_revision": self.expected_base_revision,
            "expected_merge_target": self.expected_merge_target,
            "isolation_required": self.isolation_required,
            "max_worktree_bytes": self.max_worktree_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanWorktreeContract":
        names = (
            "policy",
            "expected_base_revision",
            "expected_merge_target",
            "isolation_required",
            "max_worktree_bytes",
        )
        value = cls(
            **_decode_fields(
                payload, cls.SCHEMA, names, "plan worktree contract"
            )
        )
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanMergeStrategy(CanonicalContract):
    """Merge train/group, ordering, and post-merge validation policy."""

    SCHEMA: ClassVar[str] = PLAN_MERGE_STRATEGY_SCHEMA

    kind: MergeStrategyKind = MergeStrategyKind.SERIAL
    merge_group: str = ""
    merge_train_id: str = ""
    ordering_constraints: tuple[str, ...] = ()
    post_merge_validation_cids: tuple[str, ...] = ()
    conflict_repair_policy: str = "fail_closed"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, MergeStrategyKind, "kind")
        )
        object.__setattr__(
            self,
            "merge_group",
            _text(self.merge_group, "merge_group", required=False),
        )
        object.__setattr__(
            self,
            "merge_train_id",
            _text(self.merge_train_id, "merge_train_id", required=False),
        )
        object.__setattr__(
            self,
            "ordering_constraints",
            _ids(self.ordering_constraints, "ordering_constraints"),
        )
        object.__setattr__(
            self,
            "post_merge_validation_cids",
            _ids(self.post_merge_validation_cids, "post_merge_validation_cids"),
        )
        object.__setattr__(
            self,
            "conflict_repair_policy",
            _identifier(
                self.conflict_repair_policy, "conflict_repair_policy"
            ),
        )
        _bounded(self, "plan merge strategy")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "kind": self.kind.value,
            "merge_group": self.merge_group,
            "merge_train_id": self.merge_train_id,
            "ordering_constraints": list(self.ordering_constraints),
            "post_merge_validation_cids": list(self.post_merge_validation_cids),
            "conflict_repair_policy": self.conflict_repair_policy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanMergeStrategy":
        names = (
            "kind",
            "merge_group",
            "merge_train_id",
            "ordering_constraints",
            "post_merge_validation_cids",
            "conflict_repair_policy",
        )
        value = cls(
            **_decode_fields(payload, cls.SCHEMA, names, "plan merge strategy")
        )
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanConflictContract(CanonicalContract):
    """Conflict surface and concurrency policy for a task or plan wave."""

    SCHEMA: ClassVar[str] = PLAN_CONFLICT_CONTRACT_SCHEMA

    predicted_files: tuple[str, ...] = ()
    predicted_directories: tuple[str, ...] = ()
    predicted_symbols: tuple[str, ...] = ()
    read_only_paths: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()
    exclusive_paths: tuple[str, ...] = ()
    allow_concurrent_with: tuple[str, ...] = ()
    exclusive_group: str = ""
    shard_key: str = ""
    affinity_key: str = ""
    anti_affinity_key: str = ""
    max_files: int = 0
    max_bytes: int = 0
    conflict_surface_cid: str = ""

    def __post_init__(self) -> None:
        for name in (
            "predicted_files",
            "predicted_directories",
            "read_only_paths",
            "protected_paths",
            "exclusive_paths",
        ):
            object.__setattr__(self, name, _paths(getattr(self, name), name))
        object.__setattr__(
            self,
            "predicted_symbols",
            _ids(self.predicted_symbols, "predicted_symbols"),
        )
        object.__setattr__(
            self,
            "allow_concurrent_with",
            _ids(self.allow_concurrent_with, "allow_concurrent_with"),
        )
        for name in (
            "exclusive_group",
            "shard_key",
            "affinity_key",
            "anti_affinity_key",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self, "max_files", _bounded_int(self.max_files, "max_files")
        )
        object.__setattr__(
            self, "max_bytes", _bounded_int(self.max_bytes, "max_bytes")
        )
        object.__setattr__(
            self,
            "conflict_surface_cid",
            _identity(
                self.conflict_surface_cid,
                "conflict_surface_cid",
                required=False,
            ),
        )
        _bounded(self, "plan conflict contract")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "predicted_files": list(self.predicted_files),
            "predicted_directories": list(self.predicted_directories),
            "predicted_symbols": list(self.predicted_symbols),
            "read_only_paths": list(self.read_only_paths),
            "protected_paths": list(self.protected_paths),
            "exclusive_paths": list(self.exclusive_paths),
            "allow_concurrent_with": list(self.allow_concurrent_with),
            "exclusive_group": self.exclusive_group,
            "shard_key": self.shard_key,
            "affinity_key": self.affinity_key,
            "anti_affinity_key": self.anti_affinity_key,
            "max_files": self.max_files,
            "max_bytes": self.max_bytes,
            "conflict_surface_cid": self.conflict_surface_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanConflictContract":
        names = (
            "predicted_files",
            "predicted_directories",
            "predicted_symbols",
            "read_only_paths",
            "protected_paths",
            "exclusive_paths",
            "allow_concurrent_with",
            "exclusive_group",
            "shard_key",
            "affinity_key",
            "anti_affinity_key",
            "max_files",
            "max_bytes",
            "conflict_surface_cid",
        )
        value = cls(
            **_decode_fields(
                payload, cls.SCHEMA, names, "plan conflict contract"
            )
        )
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanValidationNode(CanonicalContract):
    """One node in a validation DAG (not an unordered command list)."""

    SCHEMA: ClassVar[str] = PLAN_VALIDATION_NODE_SCHEMA

    validation_key: str
    dependency_keys: tuple[str, ...] = ()
    argv: tuple[str, ...] = ()
    cwd: str = "."
    expected_exit_codes: tuple[int, ...] = (0,)
    policy_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validation_key",
            _identifier(self.validation_key, "validation_key"),
        )
        object.__setattr__(
            self,
            "dependency_keys",
            _ids(self.dependency_keys, "dependency_keys", preserve_order=True),
        )
        if not isinstance(self.argv, Sequence) or isinstance(
            self.argv, (str, bytes)
        ):
            raise PlanRevisionContractError("argv must be a sequence")
        argv = tuple(
            _text(item, "argv", required=True) for item in self.argv
        )
        if not argv:
            raise PlanRevisionContractError("argv must not be empty")
        if any("\n" in item or "\r" in item or "\x00" in item for item in argv):
            raise PlanRevisionContractError(
                "argv contains unsafe control characters"
            )
        object.__setattr__(self, "argv", argv)
        if self.cwd == ".":
            object.__setattr__(self, "cwd", ".")
        else:
            object.__setattr__(self, "cwd", _relative_path(self.cwd, "cwd"))
        if not isinstance(self.expected_exit_codes, Sequence) or isinstance(
            self.expected_exit_codes, (str, bytes)
        ):
            raise PlanRevisionContractError(
                "expected_exit_codes must be a sequence"
            )
        codes = tuple(
            sorted(
                {
                    _bounded_int(code, "expected_exit_codes", maximum=255)
                    for code in self.expected_exit_codes
                }
            )
        )
        if not codes:
            raise PlanRevisionContractError(
                "expected_exit_codes must not be empty"
            )
        object.__setattr__(self, "expected_exit_codes", codes)
        object.__setattr__(
            self,
            "policy_cid",
            _identity(self.policy_cid, "policy_cid", required=False),
        )
        _bounded(self, "plan validation node")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "validation_key": self.validation_key,
            "dependency_keys": list(self.dependency_keys),
            "argv": list(self.argv),
            "cwd": self.cwd,
            "expected_exit_codes": list(self.expected_exit_codes),
            "policy_cid": self.policy_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanValidationNode":
        names = (
            "validation_key",
            "dependency_keys",
            "argv",
            "cwd",
            "expected_exit_codes",
            "policy_cid",
        )
        value = cls(
            **_decode_fields(
                payload, cls.SCHEMA, names, "plan validation node"
            )
        )
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanCompletionRule(CanonicalContract):
    """Completion authority and forbidden self-completion policy."""

    SCHEMA: ClassVar[str] = PLAN_COMPLETION_RULE_SCHEMA

    authority: CompletionAuthority = CompletionAuthority.VALIDATION_GATE
    forbidden_authorities: tuple[str, ...] = ("model", "provider", "task")
    required_evidence_kinds: tuple[str, ...] = ()
    required_proof_obligations: tuple[str, ...] = ()
    require_current_tree: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, CompletionAuthority, "authority"),
        )
        object.__setattr__(
            self,
            "forbidden_authorities",
            _ids(self.forbidden_authorities, "forbidden_authorities"),
        )
        object.__setattr__(
            self,
            "required_evidence_kinds",
            _ids(self.required_evidence_kinds, "required_evidence_kinds"),
        )
        object.__setattr__(
            self,
            "required_proof_obligations",
            _ids(
                self.required_proof_obligations, "required_proof_obligations"
            ),
        )
        object.__setattr__(
            self,
            "require_current_tree",
            _boolean(self.require_current_tree, "require_current_tree"),
        )
        if self.authority is CompletionAuthority.FORBIDDEN:
            raise PlanRevisionContractError(
                "completion authority cannot be forbidden for an executable rule"
            )
        _bounded(self, "plan completion rule")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "authority": self.authority.value,
            "forbidden_authorities": list(self.forbidden_authorities),
            "required_evidence_kinds": list(self.required_evidence_kinds),
            "required_proof_obligations": list(
                self.required_proof_obligations
            ),
            "require_current_tree": self.require_current_tree,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanCompletionRule":
        names = (
            "authority",
            "forbidden_authorities",
            "required_evidence_kinds",
            "required_proof_obligations",
            "require_current_tree",
        )
        value = cls(
            **_decode_fields(
                payload, cls.SCHEMA, names, "plan completion rule"
            )
        )
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Create / steer requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanCreateRequest(CanonicalContract):
    """Canonical create-plan request (proposal tier)."""

    SCHEMA: ClassVar[str] = PLAN_CREATE_REQUEST_SCHEMA

    prompt_source_cid: str
    repository_id: str
    repository_root: str
    scope_paths: tuple[str, ...]
    dirty_tree_policy: DirtyTreePolicy
    task_source_kind: TaskSourceKind
    board_namespace: str
    alias_prefix: str
    roots: PlanAuthorityRoots
    budget: PlanRequestBudget
    required_analysis_operations: tuple[str, ...]
    optional_analysis_operations: tuple[str, ...]
    required_logic_families: tuple[str, ...]
    optional_logic_families: tuple[str, ...]
    fallback_policy: FallbackPolicy = FallbackPolicy.FAIL_CLOSED
    supervisor_profile: str = "implementation-daemon"
    observe_roots: bool = False
    redacted_source_metadata: Mapping[str, Any] = field(default_factory=dict)
    caller: str = "principal:unknown"
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prompt_source_cid",
            _validate_cid(self.prompt_source_cid, "prompt_source_cid"),
        )
        object.__setattr__(
            self,
            "repository_id",
            _identifier(self.repository_id, "repository_id"),
        )
        object.__setattr__(
            self,
            "repository_root",
            _absolute_path(self.repository_root, "repository_root"),
        )
        object.__setattr__(
            self, "scope_paths", _paths(self.scope_paths, "scope_paths")
        )
        object.__setattr__(
            self,
            "dirty_tree_policy",
            _enum(self.dirty_tree_policy, DirtyTreePolicy, "dirty_tree_policy"),
        )
        object.__setattr__(
            self,
            "task_source_kind",
            _enum(self.task_source_kind, TaskSourceKind, "task_source_kind"),
        )
        object.__setattr__(
            self,
            "board_namespace",
            _identifier(self.board_namespace, "board_namespace"),
        )
        object.__setattr__(
            self, "alias_prefix", _identifier(self.alias_prefix, "alias_prefix")
        )
        object.__setattr__(
            self,
            "roots",
            _decode_nested(self.roots, PlanAuthorityRoots, "roots"),
        )
        if self.roots.repository_id != self.repository_id:
            raise PlanRevisionAuthorityError(
                "repository_id must match authority roots"
            )
        object.__setattr__(
            self,
            "budget",
            _decode_nested(self.budget, PlanRequestBudget, "budget"),
        )
        for name in (
            "required_analysis_operations",
            "optional_analysis_operations",
            "required_logic_families",
            "optional_logic_families",
        ):
            object.__setattr__(
                self, name, _ids(getattr(self, name), name, preserve_order=True)
            )
        object.__setattr__(
            self,
            "fallback_policy",
            _enum(self.fallback_policy, FallbackPolicy, "fallback_policy"),
        )
        object.__setattr__(
            self,
            "supervisor_profile",
            _identifier(self.supervisor_profile, "supervisor_profile"),
        )
        object.__setattr__(
            self, "observe_roots", _boolean(self.observe_roots, "observe_roots")
        )
        object.__setattr__(
            self,
            "redacted_source_metadata",
            _freeze_mapping(
                self.redacted_source_metadata, "redacted_source_metadata"
            ),
        )
        object.__setattr__(self, "caller", _identifier(self.caller, "caller"))
        object.__setattr__(
            self,
            "idempotency_key",
            _text(self.idempotency_key, "idempotency_key", required=False),
        )
        _bounded(self, "plan create request")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "prompt_source_cid": self.prompt_source_cid,
            "repository_id": self.repository_id,
            "repository_root": self.repository_root,
            "scope_paths": list(self.scope_paths),
            "dirty_tree_policy": self.dirty_tree_policy.value,
            "task_source_kind": self.task_source_kind.value,
            "board_namespace": self.board_namespace,
            "alias_prefix": self.alias_prefix,
            "roots": self.roots.to_dict(),
            "budget": self.budget.to_dict(),
            "required_analysis_operations": list(
                self.required_analysis_operations
            ),
            "optional_analysis_operations": list(
                self.optional_analysis_operations
            ),
            "required_logic_families": list(self.required_logic_families),
            "optional_logic_families": list(self.optional_logic_families),
            "fallback_policy": self.fallback_policy.value,
            "supervisor_profile": self.supervisor_profile,
            "observe_roots": self.observe_roots,
            "redacted_source_metadata": dict(self.redacted_source_metadata),
            "caller": self.caller,
            "idempotency_key": self.idempotency_key,
        }

    @property
    def request_cid(self) -> str:
        return self.content_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanCreateRequest":
        names = (
            "prompt_source_cid",
            "repository_id",
            "repository_root",
            "scope_paths",
            "dirty_tree_policy",
            "task_source_kind",
            "board_namespace",
            "alias_prefix",
            "roots",
            "budget",
            "required_analysis_operations",
            "optional_analysis_operations",
            "required_logic_families",
            "optional_logic_families",
            "fallback_policy",
            "supervisor_profile",
            "observe_roots",
            "redacted_source_metadata",
            "caller",
            "idempotency_key",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, names, "plan create request"
        )
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanSteerRequest(CanonicalContract):
    """Canonical steer-plan request bound to an exact base revision."""

    SCHEMA: ClassVar[str] = PLAN_STEER_REQUEST_SCHEMA

    directive_cid: str
    base_admitted_plan_root: str
    base_materialized_plan_root: str
    plan_revision: int
    parent_revision: int
    roots: PlanAuthorityRoots
    event_cursor: str
    status_population: PlanPopulationDigest
    claimed_population: PlanPopulationDigest
    accepted_population: PlanPopulationDigest
    accepted_evidence_root: str
    completion_revision: str
    allowed_delta_operations: tuple[str, ...]
    budget: PlanRequestBudget
    max_affected_goals: int = 64
    max_affected_tasks: int = 256
    max_affected_paths: int = 1_024
    may_request_lifecycle_action: bool = False
    supervisor_run_id: str = ""
    supervisor_state_revision: str = ""
    lease_id: str = ""
    fencing_epoch: int = 0
    fallback_policy: FallbackPolicy = FallbackPolicy.FAIL_CLOSED
    redacted_directive_metadata: Mapping[str, Any] = field(default_factory=dict)
    caller: str = "principal:unknown"
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "directive_cid",
            _validate_cid(self.directive_cid, "directive_cid"),
        )
        object.__setattr__(
            self,
            "base_admitted_plan_root",
            _identity(self.base_admitted_plan_root, "base_admitted_plan_root"),
        )
        object.__setattr__(
            self,
            "base_materialized_plan_root",
            _identity(
                self.base_materialized_plan_root, "base_materialized_plan_root"
            ),
        )
        object.__setattr__(
            self,
            "plan_revision",
            _bounded_int(self.plan_revision, "plan_revision", minimum=1),
        )
        object.__setattr__(
            self,
            "parent_revision",
            _bounded_int(self.parent_revision, "parent_revision", minimum=0),
        )
        if self.parent_revision >= self.plan_revision:
            raise PlanRevisionContractError(
                "parent_revision must be strictly less than plan_revision"
            )
        object.__setattr__(
            self,
            "roots",
            _decode_nested(self.roots, PlanAuthorityRoots, "roots"),
        )
        object.__setattr__(
            self, "event_cursor", _identity(self.event_cursor, "event_cursor")
        )
        object.__setattr__(
            self,
            "status_population",
            _decode_nested(
                self.status_population, PlanPopulationDigest, "status_population"
            ),
        )
        object.__setattr__(
            self,
            "claimed_population",
            _decode_nested(
                self.claimed_population,
                PlanPopulationDigest,
                "claimed_population",
            ),
        )
        if self.claimed_population.kind not in (
            PopulationKind.CLAIMED,
            PopulationKind.RUNNING,
            PopulationKind.SETTLING,
        ):
            raise PlanRevisionContractError(
                "claimed_population.kind must be claimed, running, or settling"
            )
        object.__setattr__(
            self,
            "accepted_population",
            _decode_nested(
                self.accepted_population,
                PlanPopulationDigest,
                "accepted_population",
            ),
        )
        if self.accepted_population.kind not in (
            PopulationKind.ACCEPTED,
            PopulationKind.COMPLETED,
        ):
            raise PlanRevisionContractError(
                "accepted_population.kind must be accepted or completed"
            )
        object.__setattr__(
            self,
            "accepted_evidence_root",
            _identity(self.accepted_evidence_root, "accepted_evidence_root"),
        )
        object.__setattr__(
            self,
            "completion_revision",
            _identity(self.completion_revision, "completion_revision"),
        )
        ops = _ids(
            self.allowed_delta_operations,
            "allowed_delta_operations",
            required=True,
            preserve_order=True,
        )
        for op in ops:
            try:
                PlanDeltaOperation(op)
            except ValueError as exc:
                raise PlanRevisionContractError(
                    f"allowed_delta_operations contains unknown op {op!r}"
                ) from exc
        object.__setattr__(self, "allowed_delta_operations", ops)
        object.__setattr__(
            self,
            "budget",
            _decode_nested(self.budget, PlanRequestBudget, "budget"),
        )
        for name in (
            "max_affected_goals",
            "max_affected_tasks",
            "max_affected_paths",
            "fencing_epoch",
        ):
            object.__setattr__(
                self, name, _bounded_int(getattr(self, name), name, minimum=0)
            )
        object.__setattr__(
            self,
            "may_request_lifecycle_action",
            _boolean(
                self.may_request_lifecycle_action,
                "may_request_lifecycle_action",
            ),
        )
        for name in (
            "supervisor_run_id",
            "supervisor_state_revision",
            "lease_id",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self,
            "fallback_policy",
            _enum(self.fallback_policy, FallbackPolicy, "fallback_policy"),
        )
        object.__setattr__(
            self,
            "redacted_directive_metadata",
            _freeze_mapping(
                self.redacted_directive_metadata, "redacted_directive_metadata"
            ),
        )
        object.__setattr__(self, "caller", _identifier(self.caller, "caller"))
        object.__setattr__(
            self,
            "idempotency_key",
            _text(self.idempotency_key, "idempotency_key", required=False),
        )
        _bounded(self, "plan steer request")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "directive_cid": self.directive_cid,
            "base_admitted_plan_root": self.base_admitted_plan_root,
            "base_materialized_plan_root": self.base_materialized_plan_root,
            "plan_revision": self.plan_revision,
            "parent_revision": self.parent_revision,
            "roots": self.roots.to_dict(),
            "event_cursor": self.event_cursor,
            "status_population": self.status_population.to_dict(),
            "claimed_population": self.claimed_population.to_dict(),
            "accepted_population": self.accepted_population.to_dict(),
            "accepted_evidence_root": self.accepted_evidence_root,
            "completion_revision": self.completion_revision,
            "allowed_delta_operations": list(self.allowed_delta_operations),
            "budget": self.budget.to_dict(),
            "max_affected_goals": self.max_affected_goals,
            "max_affected_tasks": self.max_affected_tasks,
            "max_affected_paths": self.max_affected_paths,
            "may_request_lifecycle_action": self.may_request_lifecycle_action,
            "supervisor_run_id": self.supervisor_run_id,
            "supervisor_state_revision": self.supervisor_state_revision,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "fallback_policy": self.fallback_policy.value,
            "redacted_directive_metadata": dict(
                self.redacted_directive_metadata
            ),
            "caller": self.caller,
            "idempotency_key": self.idempotency_key,
        }

    @property
    def request_cid(self) -> str:
        return self.content_id

    def require_fresh(
        self,
        *,
        roots: PlanAuthorityRoots,
        plan_revision: int,
        event_cursor: str,
        claimed_digest: str,
        accepted_evidence_root: str,
    ) -> None:
        """Fail closed when any bound root or population is stale."""

        self.roots.require_current(roots)
        if plan_revision != self.plan_revision:
            raise PlanRevisionStaleRootError("plan revision is stale")
        if event_cursor != self.event_cursor:
            raise PlanRevisionStaleRootError("event cursor is stale")
        if claimed_digest != self.claimed_population.digest:
            raise PlanRevisionStaleRootError("claimed population is stale")
        if accepted_evidence_root != self.accepted_evidence_root:
            raise PlanRevisionStaleRootError("accepted evidence root is stale")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanSteerRequest":
        names = (
            "directive_cid",
            "base_admitted_plan_root",
            "base_materialized_plan_root",
            "plan_revision",
            "parent_revision",
            "roots",
            "event_cursor",
            "status_population",
            "claimed_population",
            "accepted_population",
            "accepted_evidence_root",
            "completion_revision",
            "allowed_delta_operations",
            "budget",
            "max_affected_goals",
            "max_affected_tasks",
            "max_affected_paths",
            "may_request_lifecycle_action",
            "supervisor_run_id",
            "supervisor_state_revision",
            "lease_id",
            "fencing_epoch",
            "fallback_policy",
            "redacted_directive_metadata",
            "caller",
            "idempotency_key",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, names, "plan steer request"
        )
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Plan delta / revision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanDeltaItem(CanonicalContract):
    """One closed delta operation against a base plan population."""

    SCHEMA: ClassVar[str] = PLAN_DELTA_ITEM_SCHEMA

    item_key: str
    operation: PlanDeltaOperation
    target_cid: str
    expected_target_lifecycle: LifecycleState
    expected_target_spec_revision: str
    before_digest: str
    after_record_cid: str
    effect_class: DeltaEffectClass
    rationale: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    preconditions: tuple[str, ...] = ()
    expected_effects: tuple[str, ...] = ()
    rollback_refs: tuple[str, ...] = ()
    affected_goal_cids: tuple[str, ...] = ()
    affected_task_cids: tuple[str, ...] = ()
    affected_paths: tuple[str, ...] = ()
    dependency_impact: tuple[str, ...] = ()
    conflict_impact: tuple[str, ...] = ()
    resource_impact: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "item_key", _identifier(self.item_key, "item_key")
        )
        object.__setattr__(
            self,
            "operation",
            _enum(self.operation, PlanDeltaOperation, "operation"),
        )
        object.__setattr__(
            self,
            "target_cid",
            _identity(self.target_cid, "target_cid", required=False),
        )
        object.__setattr__(
            self,
            "expected_target_lifecycle",
            _enum(
                self.expected_target_lifecycle,
                LifecycleState,
                "expected_target_lifecycle",
            ),
        )
        object.__setattr__(
            self,
            "expected_target_spec_revision",
            _identity(
                self.expected_target_spec_revision,
                "expected_target_spec_revision",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "before_digest",
            _identity(self.before_digest, "before_digest", required=False),
        )
        object.__setattr__(
            self,
            "after_record_cid",
            _identity(
                self.after_record_cid, "after_record_cid", required=False
            ),
        )
        object.__setattr__(
            self,
            "effect_class",
            _enum(self.effect_class, DeltaEffectClass, "effect_class"),
        )
        object.__setattr__(
            self, "rationale", _text(self.rationale, "rationale")
        )
        object.__setattr__(
            self, "provenance", _freeze_mapping(self.provenance, "provenance")
        )
        for name in (
            "preconditions",
            "expected_effects",
            "rollback_refs",
            "affected_goal_cids",
            "affected_task_cids",
            "dependency_impact",
            "conflict_impact",
            "resource_impact",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self, "affected_paths", _paths(self.affected_paths, "affected_paths")
        )
        self._assert_lifecycle_safe()
        _bounded(self, "plan delta item")

    def _assert_lifecycle_safe(self) -> None:
        state = self.expected_target_lifecycle
        op = self.operation
        if state in _IMMUTABLE_HISTORY_STATES:
            if op in _MUTATING_TARGET_OPERATIONS:
                raise PlanRevisionLifecycleError(
                    f"operation {op.value} cannot mutate "
                    f"{state.value} history"
                )
            if op not in _CLAIMED_SAFE_OPERATIONS and state in (
                LifecycleState.COMPLETED,
                LifecycleState.ACCEPTED,
            ):
                raise PlanRevisionLifecycleError(
                    f"operation {op.value} cannot target "
                    f"{state.value} history"
                )
            if op is PlanDeltaOperation.SUPERSEDE_GOAL and state in (
                LifecycleState.COMPLETED,
                LifecycleState.ACCEPTED,
                LifecycleState.CLAIMED,
                LifecycleState.RUNNING,
                LifecycleState.SETTLING,
            ):
                raise PlanRevisionLifecycleError(
                    "completed/accepted/claimed goals cannot be superseded "
                    "in place"
                )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "item_key": self.item_key,
            "operation": self.operation.value,
            "target_cid": self.target_cid,
            "expected_target_lifecycle": self.expected_target_lifecycle.value,
            "expected_target_spec_revision": (
                self.expected_target_spec_revision
            ),
            "before_digest": self.before_digest,
            "after_record_cid": self.after_record_cid,
            "effect_class": self.effect_class.value,
            "rationale": self.rationale,
            "provenance": dict(self.provenance),
            "preconditions": list(self.preconditions),
            "expected_effects": list(self.expected_effects),
            "rollback_refs": list(self.rollback_refs),
            "affected_goal_cids": list(self.affected_goal_cids),
            "affected_task_cids": list(self.affected_task_cids),
            "affected_paths": list(self.affected_paths),
            "dependency_impact": list(self.dependency_impact),
            "conflict_impact": list(self.conflict_impact),
            "resource_impact": list(self.resource_impact),
        }

    @property
    def item_cid(self) -> str:
        return self.content_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanDeltaItem":
        names = (
            "item_key",
            "operation",
            "target_cid",
            "expected_target_lifecycle",
            "expected_target_spec_revision",
            "before_digest",
            "after_record_cid",
            "effect_class",
            "rationale",
            "provenance",
            "preconditions",
            "expected_effects",
            "rollback_refs",
            "affected_goal_cids",
            "affected_task_cids",
            "affected_paths",
            "dependency_impact",
            "conflict_impact",
            "resource_impact",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, names, "plan delta item"
        )
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanDelta(CanonicalContract):
    """Closed, history-preserving delta against an exact base plan root."""

    SCHEMA: ClassVar[str] = PLAN_DELTA_SCHEMA

    base_plan_root: str
    base_plan_revision: int
    request_cid: str
    roots: PlanAuthorityRoots
    items: tuple[PlanDeltaItem, ...]
    expected_effects: tuple[str, ...] = ()
    deferred_item_keys: tuple[str, ...] = ()
    claimed_population_digest: str = ""
    accepted_population_digest: str = ""
    scan_receipt_cid: str = ""
    evidence_bundle_cid: str = ""
    admission_receipt_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_plan_root",
            _identity(self.base_plan_root, "base_plan_root"),
        )
        object.__setattr__(
            self,
            "base_plan_revision",
            _bounded_int(
                self.base_plan_revision, "base_plan_revision", minimum=1
            ),
        )
        object.__setattr__(
            self, "request_cid", _identity(self.request_cid, "request_cid")
        )
        object.__setattr__(
            self,
            "roots",
            _decode_nested(self.roots, PlanAuthorityRoots, "roots"),
        )
        if self.items is None:
            raw_items: Sequence[Any] = ()
        elif isinstance(self.items, Sequence) and not isinstance(
            self.items, (str, bytes, bytearray)
        ):
            raw_items = self.items
        else:
            raise PlanRevisionContractError("items must be a sequence")
        if len(raw_items) > MAX_DELTA_ITEMS:
            raise PlanRevisionBoundsError("items exceeds its item bound")
        if not raw_items:
            raise PlanRevisionContractError("items must not be empty")
        decoded_items: list[PlanDeltaItem] = []
        seen_keys: set[str] = set()
        for raw in raw_items:
            item = _decode_nested(raw, PlanDeltaItem, "items")
            assert isinstance(item, PlanDeltaItem)
            if item.item_key in seen_keys:
                raise PlanRevisionContractError("delta item keys must be unique")
            seen_keys.add(item.item_key)
            decoded_items.append(item)
        # Stable order by content identity for canonical serialization.
        items = tuple(sorted(decoded_items, key=lambda record: record.content_id))
        object.__setattr__(self, "items", items)
        object.__setattr__(
            self,
            "expected_effects",
            _ids(self.expected_effects, "expected_effects"),
        )
        object.__setattr__(
            self,
            "deferred_item_keys",
            _ids(self.deferred_item_keys, "deferred_item_keys"),
        )
        deferred = set(self.deferred_item_keys)
        item_keys = {item.item_key for item in self.items}  # type: ignore[attr-defined]
        if not deferred.issubset(item_keys):
            raise PlanRevisionContractError(
                "deferred_item_keys references unknown items"
            )
        for name in (
            "claimed_population_digest",
            "accepted_population_digest",
            "scan_receipt_cid",
            "evidence_bundle_cid",
            "admission_receipt_cid",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name, required=False),
            )
        assert_delta_preserves_history(self)
        _bounded(self, "plan delta")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "base_plan_root": self.base_plan_root,
            "base_plan_revision": self.base_plan_revision,
            "request_cid": self.request_cid,
            "roots": self.roots.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "expected_effects": list(self.expected_effects),
            "deferred_item_keys": list(self.deferred_item_keys),
            "claimed_population_digest": self.claimed_population_digest,
            "accepted_population_digest": self.accepted_population_digest,
            "scan_receipt_cid": self.scan_receipt_cid,
            "evidence_bundle_cid": self.evidence_bundle_cid,
            "admission_receipt_cid": self.admission_receipt_cid,
        }

    @property
    def delta_cid(self) -> str:
        return self.content_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanDelta":
        names = (
            "base_plan_root",
            "base_plan_revision",
            "request_cid",
            "roots",
            "items",
            "expected_effects",
            "deferred_item_keys",
            "claimed_population_digest",
            "accepted_population_digest",
            "scan_receipt_cid",
            "evidence_bundle_cid",
            "admission_receipt_cid",
        )
        values = _decode_fields(payload, cls.SCHEMA, names, "plan delta")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PlanRevision(CanonicalContract):
    """Immutable plan revision with ancestry and population digests."""

    SCHEMA: ClassVar[str] = PLAN_REVISION_SCHEMA

    plan_root_cid: str
    semantic_revision: int
    parent_plan_root: str
    origin: PlanOrigin
    roots: PlanAuthorityRoots
    request_cid: str
    delta_cid: str
    scan_receipt_cid: str
    query_plan_cid: str
    evidence_bundle_cid: str
    admission_receipt_cid: str
    execution_plan_cid: str
    goal_population: PlanPopulationDigest
    task_population: PlanPopulationDigest
    added_population: PlanPopulationDigest
    superseded_population: PlanPopulationDigest
    retained_population: PlanPopulationDigest
    deferred_population: PlanPopulationDigest
    claimed_population: PlanPopulationDigest
    completed_population: PlanPopulationDigest
    blocked_population: PlanPopulationDigest
    resource_contract: PlanResourceContract
    provider_contract: PlanProviderContract
    lease_contract: PlanLeaseContract
    retry_contract: PlanRetryContract
    worktree_contract: PlanWorktreeContract
    merge_strategy: PlanMergeStrategy
    conflict_contract: PlanConflictContract
    completion_rule: PlanCompletionRule
    validation_dag: tuple[PlanValidationNode, ...] = ()
    materialization_transaction_cid: str = ""
    rollback_ref: str = ""
    event_cursor: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_root_cid",
            _identity(self.plan_root_cid, "plan_root_cid"),
        )
        object.__setattr__(
            self,
            "semantic_revision",
            _bounded_int(
                self.semantic_revision, "semantic_revision", minimum=1
            ),
        )
        object.__setattr__(
            self,
            "parent_plan_root",
            _identity(
                self.parent_plan_root, "parent_plan_root", required=False
            ),
        )
        object.__setattr__(
            self, "origin", _enum(self.origin, PlanOrigin, "origin")
        )
        if self.origin is PlanOrigin.CREATE:
            if self.semantic_revision != 1:
                raise PlanRevisionContractError(
                    "create origin requires semantic_revision == 1"
                )
            if self.parent_plan_root:
                raise PlanRevisionContractError(
                    "create origin must not set parent_plan_root"
                )
        else:
            if self.semantic_revision < 2:
                raise PlanRevisionContractError(
                    "steer origin requires semantic_revision >= 2"
                )
            if not self.parent_plan_root:
                raise PlanRevisionContractError(
                    "steer origin requires parent_plan_root"
                )
            if self.parent_plan_root == self.plan_root_cid:
                raise PlanRevisionIdentityError(
                    "parent_plan_root must differ from plan_root_cid"
                )
        object.__setattr__(
            self,
            "roots",
            _decode_nested(self.roots, PlanAuthorityRoots, "roots"),
        )
        for name in (
            "request_cid",
            "delta_cid",
            "scan_receipt_cid",
            "query_plan_cid",
            "evidence_bundle_cid",
            "admission_receipt_cid",
            "execution_plan_cid",
            "materialization_transaction_cid",
            "rollback_ref",
            "event_cursor",
        ):
            required = name in {
                "request_cid",
                "scan_receipt_cid",
                "admission_receipt_cid",
            }
            if name == "delta_cid" and self.origin is PlanOrigin.STEER:
                required = True
            if name == "delta_cid" and self.origin is PlanOrigin.CREATE:
                required = False
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name, required=required),
            )
        for name, expected_kind in (
            ("goal_population", None),
            ("task_population", None),
            ("added_population", PopulationKind.ADDED),
            ("superseded_population", PopulationKind.SUPERSEDED),
            ("retained_population", PopulationKind.RETAINED),
            ("deferred_population", PopulationKind.DEFERRED),
            ("claimed_population", PopulationKind.CLAIMED),
            ("completed_population", PopulationKind.COMPLETED),
            ("blocked_population", PopulationKind.BLOCKED),
        ):
            pop = _decode_nested(
                getattr(self, name), PlanPopulationDigest, name
            )
            if expected_kind is not None and pop.kind is not expected_kind:
                # Allow completed to also appear as accepted digest for create.
                if not (
                    expected_kind is PopulationKind.COMPLETED
                    and pop.kind is PopulationKind.ACCEPTED
                ) and not (
                    expected_kind is PopulationKind.CLAIMED
                    and pop.kind
                    in (
                        PopulationKind.CLAIMED,
                        PopulationKind.RUNNING,
                        PopulationKind.SETTLING,
                    )
                ):
                    raise PlanRevisionContractError(
                        f"{name}.kind must be {expected_kind.value}"
                    )
            object.__setattr__(self, name, pop)
        object.__setattr__(
            self,
            "resource_contract",
            _decode_nested(
                self.resource_contract, PlanResourceContract, "resource_contract"
            ),
        )
        object.__setattr__(
            self,
            "provider_contract",
            _decode_nested(
                self.provider_contract, PlanProviderContract, "provider_contract"
            ),
        )
        object.__setattr__(
            self,
            "lease_contract",
            _decode_nested(
                self.lease_contract, PlanLeaseContract, "lease_contract"
            ),
        )
        object.__setattr__(
            self,
            "retry_contract",
            _decode_nested(
                self.retry_contract, PlanRetryContract, "retry_contract"
            ),
        )
        object.__setattr__(
            self,
            "worktree_contract",
            _decode_nested(
                self.worktree_contract, PlanWorktreeContract, "worktree_contract"
            ),
        )
        object.__setattr__(
            self,
            "merge_strategy",
            _decode_nested(
                self.merge_strategy, PlanMergeStrategy, "merge_strategy"
            ),
        )
        object.__setattr__(
            self,
            "conflict_contract",
            _decode_nested(
                self.conflict_contract, PlanConflictContract, "conflict_contract"
            ),
        )
        object.__setattr__(
            self,
            "completion_rule",
            _decode_nested(
                self.completion_rule, PlanCompletionRule, "completion_rule"
            ),
        )
        object.__setattr__(
            self,
            "validation_dag",
            _decode_sequence(
                self.validation_dag,
                PlanValidationNode,
                "validation_dag",
                limit=MAX_REFERENCE_COUNT,
            ),
        )
        _assert_validation_dag_acyclic(self.validation_dag)  # type: ignore[arg-type]
        _bounded(self, "plan revision")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_REVISION_CONTRACT_VERSION,
            "plan_root_cid": self.plan_root_cid,
            "semantic_revision": self.semantic_revision,
            "parent_plan_root": self.parent_plan_root,
            "origin": self.origin.value,
            "roots": self.roots.to_dict(),
            "request_cid": self.request_cid,
            "delta_cid": self.delta_cid,
            "scan_receipt_cid": self.scan_receipt_cid,
            "query_plan_cid": self.query_plan_cid,
            "evidence_bundle_cid": self.evidence_bundle_cid,
            "admission_receipt_cid": self.admission_receipt_cid,
            "execution_plan_cid": self.execution_plan_cid,
            "goal_population": self.goal_population.to_dict(),
            "task_population": self.task_population.to_dict(),
            "added_population": self.added_population.to_dict(),
            "superseded_population": self.superseded_population.to_dict(),
            "retained_population": self.retained_population.to_dict(),
            "deferred_population": self.deferred_population.to_dict(),
            "claimed_population": self.claimed_population.to_dict(),
            "completed_population": self.completed_population.to_dict(),
            "blocked_population": self.blocked_population.to_dict(),
            "resource_contract": self.resource_contract.to_dict(),
            "provider_contract": self.provider_contract.to_dict(),
            "lease_contract": self.lease_contract.to_dict(),
            "retry_contract": self.retry_contract.to_dict(),
            "worktree_contract": self.worktree_contract.to_dict(),
            "merge_strategy": self.merge_strategy.to_dict(),
            "conflict_contract": self.conflict_contract.to_dict(),
            "completion_rule": self.completion_rule.to_dict(),
            "validation_dag": [node.to_dict() for node in self.validation_dag],
            "materialization_transaction_cid": (
                self.materialization_transaction_cid
            ),
            "rollback_ref": self.rollback_ref,
            "event_cursor": self.event_cursor,
        }

    @property
    def revision_cid(self) -> str:
        return self.content_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanRevision":
        names = (
            "plan_root_cid",
            "semantic_revision",
            "parent_plan_root",
            "origin",
            "roots",
            "request_cid",
            "delta_cid",
            "scan_receipt_cid",
            "query_plan_cid",
            "evidence_bundle_cid",
            "admission_receipt_cid",
            "execution_plan_cid",
            "goal_population",
            "task_population",
            "added_population",
            "superseded_population",
            "retained_population",
            "deferred_population",
            "claimed_population",
            "completed_population",
            "blocked_population",
            "resource_contract",
            "provider_contract",
            "lease_contract",
            "retry_contract",
            "worktree_contract",
            "merge_strategy",
            "conflict_contract",
            "completion_rule",
            "validation_dag",
            "materialization_transaction_cid",
            "rollback_ref",
            "event_cursor",
        )
        values = _decode_fields(payload, cls.SCHEMA, names, "plan revision")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Lifecycle / population helpers
# ---------------------------------------------------------------------------


def _assert_validation_dag_acyclic(
    nodes: Sequence[PlanValidationNode],
) -> None:
    if not nodes:
        return
    keys = {node.validation_key for node in nodes}
    if len(keys) != len(nodes):
        raise PlanRevisionContractError(
            "validation DAG keys must be unique"
        )
    graph = {node.validation_key: tuple(node.dependency_keys) for node in nodes}
    for key, deps in graph.items():
        for dep in deps:
            if dep not in keys:
                raise PlanRevisionContractError(
                    f"validation DAG references unknown dependency {dep!r}"
                )
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise PlanRevisionContractError("validation DAG contains a cycle")
        visiting.add(node)
        for dep in graph[node]:
            walk(dep)
        visiting.remove(node)
        visited.add(node)

    for key in graph:
        walk(key)


def assert_delta_preserves_history(delta: PlanDelta) -> None:
    """Reject deltas that would edit or delete immutable history."""

    claimed = set()
    if delta.claimed_population_digest:
        # Digest alone is not expandable; rely on per-item lifecycle fields.
        pass
    for item in delta.items:
        state = item.expected_target_lifecycle
        if state in _IMMUTABLE_HISTORY_STATES:
            if item.operation in _MUTATING_TARGET_OPERATIONS:
                raise PlanRevisionLifecycleError(
                    f"delta item {item.item_key!r} mutates "
                    f"{state.value} history"
                )
            if item.operation is PlanDeltaOperation.SUPERSEDE_UNSTARTED_TASK:
                raise PlanRevisionLifecycleError(
                    f"delta item {item.item_key!r} cannot supersede "
                    f"{state.value} task"
                )
        # Explicit prohibition of delete-like language via closed enum;
        # double-check no after-empty mutation of completed targets.
        if state in (
            LifecycleState.COMPLETED,
            LifecycleState.ACCEPTED,
        ) and item.operation not in (
            PlanDeltaOperation.ATTACH_EVIDENCE,
            PlanDeltaOperation.RECORD_UNCERTAINTY,
            PlanDeltaOperation.ADD_TASK,
            PlanDeltaOperation.ADD_GOAL,
            PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION,
        ):
            raise PlanRevisionLifecycleError(
                f"delta item {item.item_key!r} cannot alter "
                f"{state.value} history"
            )


def assert_population_history_intact(
    *,
    prior_completed: Iterable[str],
    prior_accepted: Iterable[str],
    prior_claimed: Iterable[str],
    next_completed: Iterable[str],
    next_accepted: Iterable[str],
    next_claimed: Iterable[str],
    deleted_cids: Iterable[str] = (),
) -> None:
    """Completed/accepted/claimed history cannot shrink or be deleted."""

    prior_c = set(prior_completed)
    prior_a = set(prior_accepted)
    prior_k = set(prior_claimed)
    next_c = set(next_completed)
    next_a = set(next_accepted)
    next_k = set(next_claimed)
    deleted = set(deleted_cids)

    protected = prior_c | prior_a | prior_k
    if deleted & protected:
        raise PlanRevisionLifecycleError(
            "completed/accepted/claimed history cannot be deleted"
        )
    if not prior_c.issubset(next_c | next_a):
        raise PlanRevisionLifecycleError(
            "completed history cannot shrink across revisions"
        )
    if not prior_a.issubset(next_a):
        raise PlanRevisionLifecycleError(
            "accepted history cannot shrink across revisions"
        )
    # Claimed may transition to completed/accepted, but must not vanish.
    if not prior_k.issubset(next_k | next_c | next_a):
        raise PlanRevisionLifecycleError(
            "claimed population cannot disappear without terminal transition"
        )


def closed_delta_operations() -> tuple[str, ...]:
    """Return the closed PlanDelta operation vocabulary in stable order."""

    return tuple(op.value for op in PlanDeltaOperation)


def is_history_immutable(state: LifecycleState | str) -> bool:
    state = _enum(state, LifecycleState, "state")  # type: ignore[assignment]
    return state in _IMMUTABLE_HISTORY_STATES  # type: ignore[operator]


__all__ = [
    "CONTRACT_VERSION",
    "CompletionAuthority",
    "DeltaEffectClass",
    "DependencyEdgeKind",
    "DirtyTreePolicy",
    "FallbackPolicy",
    "LifecycleState",
    "MAX_RECORD_BYTES",
    "MergeStrategyKind",
    "OutputEffect",
    "PLAN_REVISION_CONTRACTS_INTERFACE",
    "PLAN_REVISION_CONTRACT_VERSION",
    "PlanAuthorityRoots",
    "PlanCompletionRule",
    "PlanConflictContract",
    "PlanCreateRequest",
    "PlanDelta",
    "PlanDeltaItem",
    "PlanDeltaOperation",
    "PlanLeaseContract",
    "PlanMergeStrategy",
    "PlanOrigin",
    "PlanPopulationDigest",
    "PlanProviderContract",
    "PlanRequestBudget",
    "PlanResourceContract",
    "PlanRetryContract",
    "PlanRevision",
    "PlanRevisionAuthorityError",
    "PlanRevisionBoundsError",
    "PlanRevisionContractError",
    "PlanRevisionIdentityError",
    "PlanRevisionLifecycleError",
    "PlanRevisionPathError",
    "PlanRevisionSecretError",
    "PlanRevisionStaleRootError",
    "PlanSteerRequest",
    "PlanValidationNode",
    "PlanWorktreeContract",
    "PopulationKind",
    "SCHEMA_VERSION",
    "TaskSourceKind",
    "assert_delta_preserves_history",
    "assert_population_history_intact",
    "closed_delta_operations",
    "is_history_immutable",
    "plan_revision_cid",
]
