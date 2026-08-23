"""Closed, bounded contracts for proof-carrying procedures.

This module is deliberately a data boundary, not an execution boundary.  It
contains no callbacks, executable source, shell fragments, or policy logic.
All durable identities reuse the supervisor's canonical DAG-JSON profile and
``CanonicalContract`` implementation.
"""

# ruff: noqa: UP042 -- the supervisor package still supports Python 3.8.

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final, TypeVar

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)

PROCEDURE_CONTRACT_VERSION: Final[int] = 1
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_IDENTIFIER_BYTES: Final[int] = 512
MAX_ITEMS: Final[int] = 128
MAX_MAPPING_ITEMS: Final[int] = 64
MAX_NESTING: Final[int] = 6
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_STRUCTURED_INTEGER: Final[int] = 2**63 - 1
MAX_STEPS: Final[int] = 128
MAX_BRANCHES: Final[int] = 32
MAX_LOOPS: Final[int] = 16
MAX_HOLES: Final[int] = 32
MAX_SCOPE_PATHS: Final[int] = 128


class ProcedureContractError(ContractValidationError):
    """A procedure compiler artifact is malformed or unsafe."""


class ProcedureBoundsError(ProcedureContractError):
    """A bounded artifact exceeded a structural or serialized limit."""


class ProcedureIdentityError(ProcedureContractError):
    """A supplied stored identity disagrees with canonical content."""


class ProcedureSafetyError(ProcedureContractError):
    """A contract contains an unsafe path or executable-shaped value."""


class ArtifactState(str, Enum):
    CANDIDATE = "candidate"
    DEVELOPMENT = "development"
    SHADOW = "shadow"
    VERIFIED = "verified"
    PROMOTED = "promoted"
    DEGRADED = "degraded"
    STALE = "stale"
    REVOKED = "revoked"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class RiskClass(str, Enum):
    OBSERVATION_ONLY = "observation_only"
    REVERSIBLE_LOCAL = "reversible_local"
    REPOSITORY_WRITE = "repository_write"
    PUBLIC_CONTRACT = "public_contract"
    AUTHORITY_OR_SECURITY = "authority_or_security"


class StepOperation(str, Enum):
    READ_STATE = "READ_STATE"
    QUERY_AST_INDEX = "QUERY_AST_INDEX"
    QUERY_DEPENDENCY_GRAPH = "QUERY_DEPENDENCY_GRAPH"
    QUERY_SEMANTIC_INDEX = "QUERY_SEMANTIC_INDEX"
    QUERY_RECEIPT_CACHE = "QUERY_RECEIPT_CACHE"
    SELECT_EVIDENCE = "SELECT_EVIDENCE"
    EXPAND_CONTEXT_REFERENCE = "EXPAND_CONTEXT_REFERENCE"
    CHECK_CAPABILITY = "CHECK_CAPABILITY"
    CHECK_POLICY = "CHECK_POLICY"
    CHECK_AUTHORITY = "CHECK_AUTHORITY"
    CREATE_ISOLATED_WORKTREE = "CREATE_ISOLATED_WORKTREE"
    APPLY_APPROVED_PATCH_TEMPLATE = "APPLY_APPROVED_PATCH_TEMPLATE"
    REQUEST_TYPED_MODEL_HOLE = "REQUEST_TYPED_MODEL_HOLE"
    RUN_STATIC_ANALYSIS = "RUN_STATIC_ANALYSIS"
    RUN_TYPE_CHECK = "RUN_TYPE_CHECK"
    RUN_SELECTED_TESTS = "RUN_SELECTED_TESTS"
    RUN_FULL_TEST_FALLBACK = "RUN_FULL_TEST_FALLBACK"
    RUN_PROOF = "RUN_PROOF"
    RUN_ADVERSARIAL_ASSURANCE = "RUN_ADVERSARIAL_ASSURANCE"
    CHECK_DIFF = "CHECK_DIFF"
    CHECK_SCOPE = "CHECK_SCOPE"
    CHECK_POSTCONDITION = "CHECK_POSTCONDITION"
    PREPARE_MERGE = "PREPARE_MERGE"
    MERGE_IN_ISOLATED_TRAIN = "MERGE_IN_ISOLATED_TRAIN"
    VERIFY_MERGED_TREE = "VERIFY_MERGED_TREE"
    PERSIST_ARTIFACT = "PERSIST_ARTIFACT"
    EMIT_RECEIPT = "EMIT_RECEIPT"
    ROLLBACK = "ROLLBACK"
    ESCALATE = "ESCALATE"


ALLOWED_STEP_OPERATIONS: Final[frozenset[str]] = frozenset(
    operation.value for operation in StepOperation
)
FORBIDDEN_STEP_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "ARBITRARY_SHELL",
        "ARBITRARY_PYTHON",
        "ARBITRARY_NETWORK_REQUEST",
        "ARBITRARY_FILESYSTEM_PATH",
        "DISABLE_VALIDATION",
        "MODIFY_AUTHORITY_POLICY",
        "MODIFY_TRUSTED_KEYS",
        "CLAIM_COMPLETION",
    }
)


class EffectClass(str, Enum):
    OBSERVE = "observe"
    WORKTREE_CREATE = "worktree_create"
    REPOSITORY_WRITE = "repository_write"
    MODEL_REQUEST = "model_request"
    VALIDATION = "validation"
    PROOF = "proof"
    MERGE_PREPARE = "merge_prepare"
    MERGE = "merge"
    ARTIFACT_PERSIST = "artifact_persist"
    RECEIPT_EMIT = "receipt_emit"
    ROLLBACK = "rollback"
    ESCALATION = "escalation"


class IdempotencyClass(str, Enum):
    PURE = "pure"
    IDEMPOTENT = "idempotent"
    IDEMPOTENCY_KEY_REQUIRED = "idempotency_key_required"
    NEVER_REPLAY_UNKNOWN = "never_replay_unknown"


class FailureTransition(str, Enum):
    ABORT = "abort"
    RETRY = "retry"
    ROLLBACK = "rollback"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    QUARANTINE = "quarantine"


class ConditionOperator(str, Enum):
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN_CLOSED_SET = "in_closed_set"
    SUBSET_OF = "subset_of"
    CID_EQUALS = "cid_equals"
    CURRENT = "current"
    ADMITTED = "admitted"


class ValueType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"
    CID = "cid"
    RELATIVE_PATH = "relative_path"
    ENUM = "enum"
    STRING_SEQUENCE = "string_sequence"
    CID_SEQUENCE = "cid_sequence"
    STRUCTURED = "structured"


class HoleType(str, Enum):
    SELECT_ONE_OF_ALLOWED_SYMBOLS = "SELECT_ONE_OF_ALLOWED_SYMBOLS"
    GENERATE_DOCSTRING = "GENERATE_DOCSTRING"
    PROPOSE_BOUNDED_PATCH = "PROPOSE_BOUNDED_PATCH"
    CLASSIFY_FAILURE = "CLASSIFY_FAILURE"
    CHOOSE_APPROVED_REPAIR_TEMPLATE = "CHOOSE_APPROVED_REPAIR_TEMPLATE"
    SUGGEST_MISSING_TEST_CASE = "SUGGEST_MISSING_TEST_CASE"
    SUGGEST_LEMMA = "SUGGEST_LEMMA"


FORBIDDEN_HOLE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "AUTHORITY_DECISION",
        "POLICY_DECISION",
        "CONFIRMATION",
        "TRUSTED_KEY_SELECTION",
        "TEST_OMISSION",
        "PROOF_ACCEPTANCE",
        "RELEASE_PROMOTION",
        "TASK_COMPLETION",
        "UNBOUNDED_SHELL_COMMAND",
    }
)


class ProviderClass(str, Enum):
    EXACT_CACHE = "exact_cache"
    DECLARATIVE_RULE = "declarative_rule"
    DETERMINISTIC_CLASSIFIER = "deterministic_classifier"
    LOCAL_SMALL_MODEL = "local_small_model"
    REMOTE_STANDARD_MODEL = "remote_standard_model"
    REMOTE_STRONG_MODEL = "remote_strong_model"
    HUMAN = "human"


class ProcedureOutcomeStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    ROLLED_BACK = "rolled_back"
    ESCALATED = "escalated"
    QUARANTINED = "quarantined"
    CANCELLED = "cancelled"
    REFUSED = "refused"


class TrajectoryTerminalStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED_RECOVERED = "failed_recovered"
    ROLLED_BACK = "rolled_back"
    INCOMPLETE = "incomplete"


class EpisodeKind(str, Enum):
    ACCEPTED_TASK_RECEIPT = "accepted_task_receipt"
    CURRENT_TREE_POST_MERGE_RECEIPT = "current_tree_post_merge_receipt"
    VERIFIED_PROOF_RECEIPT = "verified_proof_receipt"
    ADMITTED_TEST_RECEIPT = "admitted_test_receipt"
    SUCCESSFUL_ROLLBACK_RECEIPT = "successful_rollback_receipt"
    AUTHORIZED_HUMAN_DECISION_RECEIPT = "authorized_human_decision_receipt"
    REJECTED_TASK_RECORD = "rejected_task_record"
    FAILED_RECOVERED_EXECUTION = "failed_recovered_execution"


class FamilyMembershipClass(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOUNDARY = "boundary"
    UNKNOWN = "unknown"


class TraceEventStatus(str, Enum):
    STARTED = "started"
    OBSERVED = "observed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    ROLLED_BACK = "rolled_back"


class TraceState(str, Enum):
    OPEN = "open"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    RECOVERED = "recovered"
    FAILED = "failed"


_E = TypeVar("_E", bound=Enum)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/+\-]*$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_SECRET_MARKERS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
    }
)
_EXECUTABLE_MARKERS = frozenset(
    {
        "callback",
        "callable",
        "command",
        "code_body",
        "executable",
        "policy_code",
        "python_source",
        "shell_command",
        "source_code",
    }
)


def _enum(value: Any, enum_type: type[_E], field_name: str) -> _E:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, Enum) or type(value) is not str:
        raise ProcedureContractError(
            f"{field_name} must be one of: "
            + ", ".join(sorted(str(item.value) for item in enum_type))
        )
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ProcedureContractError(
            f"{field_name} must be one of: "
            + ", ".join(sorted(str(item.value) for item in enum_type))
        ) from exc


def _text(value: Any, field_name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ProcedureContractError(f"{field_name} must be a string")
    normalized = value.strip()
    if required and not normalized:
        raise ProcedureContractError(f"{field_name} is required")
    if len(normalized.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ProcedureBoundsError(f"{field_name} exceeds its byte bound")
    if "\x00" in normalized:
        raise ProcedureSafetyError(f"{field_name} contains a NUL byte")
    return normalized


def _identifier(value: Any, field_name: str, *, required: bool = True) -> str:
    normalized = _text(value, field_name, required=required)
    if not normalized:
        return normalized
    if len(normalized.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
        raise ProcedureBoundsError(f"{field_name} exceeds its identifier bound")
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ProcedureContractError(f"{field_name} must be a compact identifier")
    return normalized


def _nonnegative_int(value: Any, field_name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProcedureContractError(f"{field_name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ProcedureBoundsError(f"{field_name} exceeds its numeric bound")
    return value


def _positive_int(value: Any, field_name: str, *, maximum: int | None = None) -> int:
    result = _nonnegative_int(value, field_name, maximum=maximum)
    if result == 0:
        raise ProcedureContractError(f"{field_name} must be positive")
    return result


def _relative_path(value: Any, field_name: str) -> str:
    normalized = _text(value, field_name)
    if (
        normalized.startswith("/")
        or normalized.startswith("\\")
        or _WINDOWS_ABSOLUTE_RE.match(normalized)
        or "\\" in normalized
        or "//" in normalized
        or (normalized != "." and (normalized.startswith("./") or normalized.endswith("/")))
    ):
        raise ProcedureSafetyError(f"{field_name} must be repository-relative")
    path = PurePosixPath(normalized)
    if any(part in {"..", ""} for part in path.parts):
        raise ProcedureSafetyError(f"{field_name} may not escape repository scope")
    if normalized != "." and path.as_posix() != normalized:
        raise ProcedureSafetyError(f"{field_name} must use a canonical repository-relative path")
    return normalized


def _strings(
    values: Any,
    field_name: str,
    *,
    limit: int = MAX_ITEMS,
    required: bool = False,
    identifiers: bool = False,
    paths: bool = False,
    preserve_order: bool = True,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raw = values
    else:
        raise ProcedureContractError(f"{field_name} must be a sequence")
    if len(raw) > limit:
        raise ProcedureBoundsError(f"{field_name} exceeds its item bound")
    items: list[str] = []
    for item in raw:
        if paths:
            normalized = _relative_path(item, field_name)
        elif identifiers:
            normalized = _identifier(item, field_name)
        else:
            normalized = _text(item, field_name)
        if normalized not in items:
            items.append(normalized)
    if required and not items:
        raise ProcedureContractError(f"{field_name} must not be empty")
    return tuple(items if preserve_order else sorted(items))


def _unsafe_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(marker in normalized for marker in _SECRET_MARKERS | _EXECUTABLE_MARKERS)


def _freeze(value: Any, field_name: str, *, depth: int = 0) -> Any:
    if depth > MAX_NESTING:
        raise ProcedureBoundsError(f"{field_name} exceeds its nesting bound")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if abs(value) > MAX_STRUCTURED_INTEGER:
            raise ProcedureBoundsError(f"{field_name} contains an unbounded integer")
        return value
    if type(value) is str:
        _text(value, field_name, required=False)
        if (
            value.startswith(("/", "\\"))
            or _WINDOWS_ABSOLUTE_RE.match(value)
            or "\\" in value
            or ".." in value.split("/")
        ):
            raise ProcedureSafetyError(f"{field_name} contains an unsafe filesystem path")
        return value
    if isinstance(value, float):
        raise ProcedureContractError(f"{field_name} cannot contain floating point values")
    if isinstance(value, Enum):
        return _freeze(value.value, field_name, depth=depth)
    if isinstance(value, CanonicalContract):
        return _freeze(value.to_dict(), field_name, depth=depth + 1)
    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_ITEMS:
            raise ProcedureBoundsError(f"{field_name} exceeds its mapping bound")
        result: dict[str, Any] = {}
        for raw_key in sorted(value):
            if not isinstance(raw_key, str):
                raise ProcedureContractError(f"{field_name} keys must be strings")
            key = _text(raw_key, field_name)
            if _unsafe_key(key):
                raise ProcedureSafetyError(
                    f"{field_name} contains a forbidden secret or executable field"
                )
            if key in result:
                raise ProcedureContractError(f"{field_name} contains duplicate normalized keys")
            result[key] = _freeze(value[raw_key], field_name, depth=depth + 1)
        return MappingProxyType(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        if len(value) > MAX_ITEMS:
            raise ProcedureBoundsError(f"{field_name} exceeds its item bound")
        return tuple(_freeze(item, field_name, depth=depth + 1) for item in value)
    raise ProcedureSafetyError(
        f"{field_name} contains unsupported value type {type(value).__name__}"
    )


def _binding_map(value: Any, field_name: str) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ProcedureContractError(f"{field_name} must be a mapping")
    if len(value) > MAX_MAPPING_ITEMS:
        raise ProcedureBoundsError(f"{field_name} exceeds its mapping bound")
    result: dict[str, str] = {}
    for raw_key in sorted(value):
        key = _identifier(raw_key, field_name)
        source = _text(value[raw_key], field_name)
        if source.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(source):
            raise ProcedureSafetyError(f"{field_name} contains an absolute path binding")
        if key in result:
            raise ProcedureContractError(f"{field_name} contains duplicate normalized keys")
        result[key] = source
    return MappingProxyType(result)


def _enums(
    values: Any,
    enum_type: type[_E],
    field_name: str,
    *,
    limit: int,
    required: bool = False,
) -> tuple[_E, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray, memoryview)):
        raise ProcedureContractError(f"{field_name} must be a sequence")
    if len(values) > limit:
        raise ProcedureBoundsError(f"{field_name} exceeds its item bound")
    result: list[_E] = []
    for value in values:
        item = _enum(value, enum_type, field_name)
        if item not in result:
            result.append(item)
    if required and not result:
        raise ProcedureContractError(f"{field_name} must not be empty")
    return tuple(result)


def _schema_name(name: str) -> str:
    slug = re.sub(r"(?<!^)(?=[A-Z])", "-", name).replace("_", "-").lower()
    return f"ipfs_accelerate_py/agent-supervisor/procedure-compiler/{slug}@1"


def _decode_fields(
    payload: Mapping[str, Any],
    schema: str,
    fields: Sequence[str],
    artifact_name: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ProcedureContractError(f"{artifact_name} has an unsupported schema")
    version = payload.get("contract_version", PROCEDURE_CONTRACT_VERSION)
    if type(version) is not int or version != PROCEDURE_CONTRACT_VERSION:
        raise ProcedureContractError(f"{artifact_name} has an unsupported version")
    allowed = set(fields) | {"schema", "contract_version", "content_id", "cid"}
    if set(payload).difference(allowed):
        raise ProcedureContractError(
            f"{artifact_name} contains unsupported fields; rebuild its canonical payload"
        )
    return {name: payload[name] for name in fields if name in payload}


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    supplied = payload.get("content_id", payload.get("cid", ""))
    if supplied not in (None, "") and supplied != record.content_id:
        raise ProcedureIdentityError(
            "stored content identity does not match canonical procedure artifact"
        )


def _bounded(record: CanonicalContract, field_name: str) -> None:
    if len(canonical_json_bytes(record.to_dict())) > MAX_RECORD_BYTES:
        raise ProcedureBoundsError(f"{field_name} exceeds its serialized byte bound")


def _nested(value: Any, cls: type[Any], field_name: str) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        if "schema" in value:
            return cls.from_dict(value)
        return cls(**value)
    raise ProcedureContractError(f"{field_name} must be {cls.__name__}")


def _records(
    values: Any,
    cls: type[Any],
    field_name: str,
    *,
    limit: int = MAX_ITEMS,
    required: bool = False,
) -> tuple[Any, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raw = values
    else:
        raise ProcedureContractError(f"{field_name} must be a sequence")
    if len(raw) > limit:
        raise ProcedureBoundsError(f"{field_name} exceeds its item bound")
    result = tuple(_nested(item, cls, field_name) for item in raw)
    if required and not result:
        raise ProcedureContractError(f"{field_name} must not be empty")
    return result


@dataclass(frozen=True)
class ArtifactBindings(CanonicalContract):
    """Exact authority roots that prevent cross-tree artifact reuse."""

    SCHEMA: ClassVar[str] = _schema_name("ArtifactBindings")

    repository_id: str
    repository_commit: str
    tree_id: str
    objective_id: str
    task_id: str
    contract_revision: str
    policy_revision: str
    environment_id: str

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "repository_commit",
            "tree_id",
            "objective_id",
            "task_id",
            "contract_revision",
            "policy_revision",
            "environment_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        _bounded(self, "ArtifactBindings")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "repository_commit": self.repository_commit,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "task_id": self.task_id,
            "contract_revision": self.contract_revision,
            "policy_revision": self.policy_revision,
            "environment_id": self.environment_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactBindings:
        fields = (
            "repository_id",
            "repository_commit",
            "tree_id",
            "objective_id",
            "task_id",
            "contract_revision",
            "policy_revision",
            "environment_id",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureVersion(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureVersion")

    major: int
    minor: int = 0
    patch: int = 0
    predecessor_cid: str = ""

    def __post_init__(self) -> None:
        for name in ("major", "minor", "patch"):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name, maximum=65_535)
            )
        object.__setattr__(
            self,
            "predecessor_cid",
            _identifier(self.predecessor_cid, "predecessor_cid", required=False),
        )

    @property
    def semantic_version(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "predecessor_cid": self.predecessor_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureVersion:
        fields = ("major", "minor", "patch", "predecessor_cid")
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureParameter(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureParameter")

    name: str
    value_type: ValueType
    required: bool = True
    allowed_values: tuple[Any, ...] = ()
    default_value: Any = None
    path_scoped: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "name"))
        object.__setattr__(self, "value_type", _enum(self.value_type, ValueType, "value_type"))
        if type(self.required) is not bool or type(self.path_scoped) is not bool:
            raise ProcedureContractError("required and path_scoped must be booleans")
        frozen = _freeze(self.allowed_values, "allowed_values")
        if not isinstance(frozen, tuple):
            raise ProcedureContractError("allowed_values must be a sequence")
        object.__setattr__(self, "allowed_values", frozen)
        object.__setattr__(self, "default_value", _freeze(self.default_value, "default_value"))
        if self.required and self.default_value is not None:
            raise ProcedureContractError("required parameters cannot carry a default")
        if self.value_type is ValueType.ENUM and not self.allowed_values:
            raise ProcedureContractError("enum parameters require a closed allowed_values set")
        if self.path_scoped and self.value_type is not ValueType.RELATIVE_PATH:
            raise ProcedureContractError("path_scoped is valid only for relative_path parameters")
        if self.value_type is ValueType.RELATIVE_PATH:
            for value in self.allowed_values:
                _relative_path(value, "allowed_values")
            if self.default_value is not None:
                _relative_path(self.default_value, "default_value")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "name": self.name,
            "value_type": self.value_type.value,
            "required": self.required,
            "allowed_values": self.allowed_values,
            "default_value": self.default_value,
            "path_scoped": self.path_scoped,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureParameter:
        fields = (
            "name",
            "value_type",
            "required",
            "allowed_values",
            "default_value",
            "path_scoped",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureLocal(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureLocal")

    name: str
    value_type: ValueType

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "name"))
        object.__setattr__(self, "value_type", _enum(self.value_type, ValueType, "value_type"))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "name": self.name,
            "value_type": self.value_type.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureLocal:
        fields = ("name", "value_type")
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureEffect(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureEffect")

    effect_id: str
    effect_class: EffectClass
    targets: tuple[str, ...] = ()
    description: str = ""
    reversible: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect_id", _identifier(self.effect_id, "effect_id"))
        object.__setattr__(
            self, "effect_class", _enum(self.effect_class, EffectClass, "effect_class")
        )
        object.__setattr__(self, "targets", _strings(self.targets, "targets", paths=True))
        object.__setattr__(
            self, "description", _text(self.description, "description", required=False)
        )
        if type(self.reversible) is not bool:
            raise ProcedureContractError("reversible must be a boolean")
        if self.effect_class is EffectClass.REPOSITORY_WRITE and not self.targets:
            raise ProcedureContractError("repository writes require explicit target paths")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "effect_id": self.effect_id,
            "effect_class": self.effect_class.value,
            "targets": self.targets,
            "description": self.description,
            "reversible": self.reversible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureEffect:
        fields = ("effect_id", "effect_class", "targets", "description", "reversible")
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class _Condition(CanonicalContract):
    condition_id: str
    binding: str
    operator: ConditionOperator
    operand: Any = None
    evidence_producer: str = ""
    evidence_type: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", _identifier(self.condition_id, "condition_id"))
        object.__setattr__(self, "binding", _text(self.binding, "binding"))
        object.__setattr__(self, "operator", _enum(self.operator, ConditionOperator, "operator"))
        object.__setattr__(self, "operand", _freeze(self.operand, "operand"))
        object.__setattr__(
            self,
            "evidence_producer",
            _identifier(self.evidence_producer, "evidence_producer", required=False),
        )
        object.__setattr__(
            self, "evidence_type", _identifier(self.evidence_type, "evidence_type", required=False)
        )
        if type(self.required) is not bool:
            raise ProcedureContractError("required must be a boolean")
        if self.required and (not self.evidence_producer or not self.evidence_type):
            raise ProcedureContractError(
                "required conditions need an independently admitted evidence producer and type"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "condition_id": self.condition_id,
            "binding": self.binding,
            "operator": self.operator.value,
            "operand": self.operand,
            "evidence_producer": self.evidence_producer,
            "evidence_type": self.evidence_type,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> _Condition:
        fields = (
            "condition_id",
            "binding",
            "operator",
            "operand",
            "evidence_producer",
            "evidence_type",
            "required",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedurePrecondition(_Condition):
    SCHEMA: ClassVar[str] = _schema_name("ProcedurePrecondition")


@dataclass(frozen=True)
class ProcedureInvariant(_Condition):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureInvariant")


@dataclass(frozen=True)
class ProcedurePostcondition(_Condition):
    SCHEMA: ClassVar[str] = _schema_name("ProcedurePostcondition")


@dataclass(frozen=True)
class ProcedureObservation(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureObservation")

    observation_id: str
    producer_contract: str
    output_binding: str
    operator: ConditionOperator
    operand: Any = None
    evidence_type: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", _identifier(self.observation_id, "observation_id")
        )
        object.__setattr__(
            self, "producer_contract", _identifier(self.producer_contract, "producer_contract")
        )
        object.__setattr__(self, "output_binding", _text(self.output_binding, "output_binding"))
        object.__setattr__(self, "operator", _enum(self.operator, ConditionOperator, "operator"))
        object.__setattr__(self, "operand", _freeze(self.operand, "operand"))
        object.__setattr__(self, "evidence_type", _identifier(self.evidence_type, "evidence_type"))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "observation_id": self.observation_id,
            "producer_contract": self.producer_contract,
            "output_binding": self.output_binding,
            "operator": self.operator.value,
            "operand": self.operand,
            "evidence_type": self.evidence_type,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureObservation:
        fields = (
            "observation_id",
            "producer_contract",
            "output_binding",
            "operator",
            "operand",
            "evidence_type",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class RetryPolicy(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("RetryPolicy")

    max_attempts: int = 1
    retryable_failure_codes: tuple[str, ...] = ()
    backoff_ms: int = 0
    requires_new_evidence: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_attempts", _positive_int(self.max_attempts, "max_attempts", maximum=8)
        )
        object.__setattr__(
            self,
            "retryable_failure_codes",
            _strings(
                self.retryable_failure_codes, "retryable_failure_codes", limit=32, identifiers=True
            ),
        )
        object.__setattr__(
            self, "backoff_ms", _nonnegative_int(self.backoff_ms, "backoff_ms", maximum=60_000)
        )
        if type(self.requires_new_evidence) is not bool:
            raise ProcedureContractError("requires_new_evidence must be a boolean")
        if self.max_attempts > 1 and not self.retryable_failure_codes:
            raise ProcedureContractError("retries require a closed retryable failure set")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "max_attempts": self.max_attempts,
            "retryable_failure_codes": self.retryable_failure_codes,
            "backoff_ms": self.backoff_ms,
            "requires_new_evidence": self.requires_new_evidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RetryPolicy:
        fields = ("max_attempts", "retryable_failure_codes", "backoff_ms", "requires_new_evidence")
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureStep(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureStep")

    step_id: str
    operation: StepOperation
    operation_contract: str
    input_bindings: Mapping[str, str] = field(default_factory=dict)
    output_bindings: Mapping[str, str] = field(default_factory=dict)
    declared_effect_ids: tuple[str, ...] = ()
    required_authority_ids: tuple[str, ...] = ()
    timeout_ms: int = 30_000
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    idempotency: IdempotencyClass = IdempotencyClass.IDEMPOTENT
    failure_transition: FailureTransition = FailureTransition.ABORT
    failure_target: str = ""
    evidence_outputs: tuple[str, ...] = ()
    next_step_id: str = ""
    hole_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(self, "operation", _enum(self.operation, StepOperation, "operation"))
        object.__setattr__(
            self, "operation_contract", _identifier(self.operation_contract, "operation_contract")
        )
        object.__setattr__(
            self, "input_bindings", _binding_map(self.input_bindings, "input_bindings")
        )
        object.__setattr__(
            self, "output_bindings", _binding_map(self.output_bindings, "output_bindings")
        )
        object.__setattr__(
            self,
            "declared_effect_ids",
            _strings(self.declared_effect_ids, "declared_effect_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "required_authority_ids",
            _strings(self.required_authority_ids, "required_authority_ids", identifiers=True),
        )
        object.__setattr__(
            self, "timeout_ms", _positive_int(self.timeout_ms, "timeout_ms", maximum=86_400_000)
        )
        object.__setattr__(
            self, "retry_policy", _nested(self.retry_policy, RetryPolicy, "retry_policy")
        )
        object.__setattr__(
            self, "idempotency", _enum(self.idempotency, IdempotencyClass, "idempotency")
        )
        object.__setattr__(
            self,
            "failure_transition",
            _enum(self.failure_transition, FailureTransition, "failure_transition"),
        )
        object.__setattr__(
            self,
            "failure_target",
            _identifier(self.failure_target, "failure_target", required=False),
        )
        object.__setattr__(
            self,
            "evidence_outputs",
            _strings(self.evidence_outputs, "evidence_outputs", identifiers=True),
        )
        object.__setattr__(
            self, "next_step_id", _identifier(self.next_step_id, "next_step_id", required=False)
        )
        object.__setattr__(self, "hole_id", _identifier(self.hole_id, "hole_id", required=False))
        if (
            self.failure_transition
            in {
                FailureTransition.ROLLBACK,
                FailureTransition.FALLBACK,
            }
            and not self.failure_target
        ):
            raise ProcedureContractError("rollback and fallback transitions require a target")
        if self.operation is StepOperation.REQUEST_TYPED_MODEL_HOLE and not self.hole_id:
            raise ProcedureContractError("REQUEST_TYPED_MODEL_HOLE requires a typed hole_id")
        if self.operation is not StepOperation.REQUEST_TYPED_MODEL_HOLE and self.hole_id:
            raise ProcedureContractError("hole_id is valid only on REQUEST_TYPED_MODEL_HOLE")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "step_id": self.step_id,
            "operation": self.operation.value,
            "operation_contract": self.operation_contract,
            "input_bindings": self.input_bindings,
            "output_bindings": self.output_bindings,
            "declared_effect_ids": self.declared_effect_ids,
            "required_authority_ids": self.required_authority_ids,
            "timeout_ms": self.timeout_ms,
            "retry_policy": self.retry_policy,
            "idempotency": self.idempotency.value,
            "failure_transition": self.failure_transition.value,
            "failure_target": self.failure_target,
            "evidence_outputs": self.evidence_outputs,
            "next_step_id": self.next_step_id,
            "hole_id": self.hole_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureStep:
        fields = (
            "step_id",
            "operation",
            "operation_contract",
            "input_bindings",
            "output_bindings",
            "declared_effect_ids",
            "required_authority_ids",
            "timeout_ms",
            "retry_policy",
            "idempotency",
            "failure_transition",
            "failure_target",
            "evidence_outputs",
            "next_step_id",
            "hole_id",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "retry_policy" in values:
            values["retry_policy"] = _nested(values["retry_policy"], RetryPolicy, "retry_policy")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureBranch(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureBranch")

    branch_id: str
    observation_id: str
    true_step_id: str
    false_step_id: str

    def __post_init__(self) -> None:
        for name in ("branch_id", "observation_id", "true_step_id", "false_step_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.true_step_id == self.false_step_id:
            raise ProcedureContractError("a branch must have distinct outcomes")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            **{
                name: getattr(self, name)
                for name in ("branch_id", "observation_id", "true_step_id", "false_step_id")
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureBranch:
        fields = ("branch_id", "observation_id", "true_step_id", "false_step_id")
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureLoop(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureLoop")

    loop_id: str
    condition_observation_id: str
    body_step_id: str
    exit_step_id: str
    max_iterations: int

    def __post_init__(self) -> None:
        for name in ("loop_id", "condition_observation_id", "body_step_id", "exit_step_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "max_iterations", _positive_int(self.max_iterations, "max_iterations", maximum=64)
        )
        if self.body_step_id == self.exit_step_id:
            raise ProcedureContractError("a loop body and exit must be distinct")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "loop_id": self.loop_id,
            "condition_observation_id": self.condition_observation_id,
            "body_step_id": self.body_step_id,
            "exit_step_id": self.exit_step_id,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureLoop:
        fields = (
            "loop_id",
            "condition_observation_id",
            "body_step_id",
            "exit_step_id",
            "max_iterations",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureHole(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureHole")

    hole_id: str
    hole_type: HoleType
    input_schema_ref: str
    output_schema_ref: str
    allowed_provider_classes: tuple[ProviderClass, ...]
    context_budget_bytes: int
    authority_requirement_ids: tuple[str, ...]
    effect_classes: tuple[EffectClass, ...]
    validation_observation_ids: tuple[str, ...]
    fallback_step_id: str
    maximum_attempts: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "hole_id", _identifier(self.hole_id, "hole_id"))
        object.__setattr__(self, "hole_type", _enum(self.hole_type, HoleType, "hole_type"))
        object.__setattr__(
            self, "input_schema_ref", _identifier(self.input_schema_ref, "input_schema_ref")
        )
        object.__setattr__(
            self, "output_schema_ref", _identifier(self.output_schema_ref, "output_schema_ref")
        )
        providers = _enums(
            self.allowed_provider_classes,
            ProviderClass,
            "allowed_provider_classes",
            limit=8,
            required=True,
        )
        object.__setattr__(self, "allowed_provider_classes", providers)
        object.__setattr__(
            self,
            "context_budget_bytes",
            _positive_int(self.context_budget_bytes, "context_budget_bytes", maximum=1_048_576),
        )
        object.__setattr__(
            self,
            "authority_requirement_ids",
            _strings(self.authority_requirement_ids, "authority_requirement_ids", identifiers=True),
        )
        effect_classes = _enums(self.effect_classes, EffectClass, "effect_classes", limit=8)
        object.__setattr__(self, "effect_classes", effect_classes)
        object.__setattr__(
            self,
            "validation_observation_ids",
            _strings(
                self.validation_observation_ids,
                "validation_observation_ids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self, "fallback_step_id", _identifier(self.fallback_step_id, "fallback_step_id")
        )
        object.__setattr__(
            self,
            "maximum_attempts",
            _positive_int(self.maximum_attempts, "maximum_attempts", maximum=4),
        )
        if (
            any(
                provider in {ProviderClass.REMOTE_STANDARD_MODEL, ProviderClass.REMOTE_STRONG_MODEL}
                for provider in providers
            )
            and self.context_budget_bytes == 0
        ):
            raise ProcedureContractError("remote holes require a nonzero context budget")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "hole_id": self.hole_id,
            "hole_type": self.hole_type.value,
            "input_schema_ref": self.input_schema_ref,
            "output_schema_ref": self.output_schema_ref,
            "allowed_provider_classes": tuple(item.value for item in self.allowed_provider_classes),
            "context_budget_bytes": self.context_budget_bytes,
            "authority_requirement_ids": self.authority_requirement_ids,
            "effect_classes": tuple(item.value for item in self.effect_classes),
            "validation_observation_ids": self.validation_observation_ids,
            "fallback_step_id": self.fallback_step_id,
            "maximum_attempts": self.maximum_attempts,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureHole:
        fields = (
            "hole_id",
            "hole_type",
            "input_schema_ref",
            "output_schema_ref",
            "allowed_provider_classes",
            "context_budget_bytes",
            "authority_requirement_ids",
            "effect_classes",
            "validation_observation_ids",
            "fallback_step_id",
            "maximum_attempts",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureResourceEnvelope(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureResourceEnvelope")

    wall_time_ms: int
    cpu_time_ms: int
    memory_bytes: int
    disk_bytes: int
    model_token_limit: int
    model_call_limit: int
    subprocess_limit: int = 0
    network_request_limit: int = 0

    def __post_init__(self) -> None:
        maxima = {
            "wall_time_ms": 86_400_000,
            "cpu_time_ms": 86_400_000,
            "memory_bytes": 1 << 50,
            "disk_bytes": 1 << 50,
            "model_token_limit": 10_000_000,
            "model_call_limit": 1_024,
            "subprocess_limit": 1_024,
            "network_request_limit": 1_024,
        }
        for name, maximum in maxima.items():
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name, maximum=maximum)
            )
        if self.wall_time_ms == 0:
            raise ProcedureContractError("wall_time_ms must be positive")

    def _payload(self) -> dict[str, Any]:
        names = (
            "wall_time_ms",
            "cpu_time_ms",
            "memory_bytes",
            "disk_bytes",
            "model_token_limit",
            "model_call_limit",
            "subprocess_limit",
            "network_request_limit",
        )
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            **{name: getattr(self, name) for name in names},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureResourceEnvelope:
        fields = (
            "wall_time_ms",
            "cpu_time_ms",
            "memory_bytes",
            "disk_bytes",
            "model_token_limit",
            "model_call_limit",
            "subprocess_limit",
            "network_request_limit",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureAuthorityEnvelope(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureAuthorityEnvelope")

    authority_policy_revision: str
    requirement_ids: tuple[str, ...]
    required_capability_ids: tuple[str, ...]
    allowed_operations: tuple[StepOperation, ...]
    risk_ceiling: RiskClass
    confirmation_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_policy_revision",
            _identifier(self.authority_policy_revision, "authority_policy_revision"),
        )
        object.__setattr__(
            self,
            "requirement_ids",
            _strings(self.requirement_ids, "requirement_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "required_capability_ids",
            _strings(self.required_capability_ids, "required_capability_ids", identifiers=True),
        )
        operations = _enums(
            self.allowed_operations,
            StepOperation,
            "allowed_operations",
            limit=len(StepOperation),
            required=True,
        )
        object.__setattr__(self, "allowed_operations", operations)
        object.__setattr__(
            self, "risk_ceiling", _enum(self.risk_ceiling, RiskClass, "risk_ceiling")
        )
        if type(self.confirmation_required) is not bool:
            raise ProcedureContractError("confirmation_required must be a boolean")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "authority_policy_revision": self.authority_policy_revision,
            "requirement_ids": self.requirement_ids,
            "required_capability_ids": self.required_capability_ids,
            "allowed_operations": tuple(item.value for item in self.allowed_operations),
            "risk_ceiling": self.risk_ceiling.value,
            "confirmation_required": self.confirmation_required,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureAuthorityEnvelope:
        fields = (
            "authority_policy_revision",
            "requirement_ids",
            "required_capability_ids",
            "allowed_operations",
            "risk_ceiling",
            "confirmation_required",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureValidationPlan(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureValidationPlan")

    required_step_ids: tuple[str, ...]
    required_observation_ids: tuple[str, ...]
    required_test_contracts: tuple[str, ...] = ()
    required_proof_contracts: tuple[str, ...] = ()
    post_merge_validation_contracts: tuple[str, ...] = ()
    full_test_fallback_contract: str = ""

    def __post_init__(self) -> None:
        for name in (
            "required_step_ids",
            "required_observation_ids",
            "required_test_contracts",
            "required_proof_contracts",
            "post_merge_validation_contracts",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    name,
                    identifiers=True,
                    required=name in {"required_step_ids", "required_observation_ids"},
                ),
            )
        object.__setattr__(
            self,
            "full_test_fallback_contract",
            _identifier(
                self.full_test_fallback_contract, "full_test_fallback_contract", required=False
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "required_step_ids": self.required_step_ids,
            "required_observation_ids": self.required_observation_ids,
            "required_test_contracts": self.required_test_contracts,
            "required_proof_contracts": self.required_proof_contracts,
            "post_merge_validation_contracts": self.post_merge_validation_contracts,
            "full_test_fallback_contract": self.full_test_fallback_contract,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureValidationPlan:
        fields = (
            "required_step_ids",
            "required_observation_ids",
            "required_test_contracts",
            "required_proof_contracts",
            "post_merge_validation_contracts",
            "full_test_fallback_contract",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureRollback(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureRollback")

    rollback_id: str
    trigger_effect_ids: tuple[str, ...]
    step_ids: tuple[str, ...]
    verification_observation_ids: tuple[str, ...]
    exact_target_cid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "rollback_id", _identifier(self.rollback_id, "rollback_id"))
        for name in ("trigger_effect_ids", "step_ids", "verification_observation_ids"):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True, required=True),
            )
        object.__setattr__(
            self, "exact_target_cid", _identifier(self.exact_target_cid, "exact_target_cid")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "rollback_id": self.rollback_id,
            "trigger_effect_ids": self.trigger_effect_ids,
            "step_ids": self.step_ids,
            "verification_observation_ids": self.verification_observation_ids,
            "exact_target_cid": self.exact_target_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureRollback:
        fields = (
            "rollback_id",
            "trigger_effect_ids",
            "step_ids",
            "verification_observation_ids",
            "exact_target_cid",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureFallback(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureFallback")

    fallback_id: str
    trigger_failure_codes: tuple[str, ...]
    entry_step_id: str
    maximum_uses: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "fallback_id", _identifier(self.fallback_id, "fallback_id"))
        object.__setattr__(
            self,
            "trigger_failure_codes",
            _strings(
                self.trigger_failure_codes, "trigger_failure_codes", identifiers=True, required=True
            ),
        )
        object.__setattr__(self, "entry_step_id", _identifier(self.entry_step_id, "entry_step_id"))
        object.__setattr__(
            self, "maximum_uses", _positive_int(self.maximum_uses, "maximum_uses", maximum=4)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "fallback_id": self.fallback_id,
            "trigger_failure_codes": self.trigger_failure_codes,
            "entry_step_id": self.entry_step_id,
            "maximum_uses": self.maximum_uses,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureFallback:
        fields = ("fallback_id", "trigger_failure_codes", "entry_step_id", "maximum_uses")
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureSpec(CanonicalContract):
    """A bounded, non-executable ProcedureIR document."""

    SCHEMA: ClassVar[str] = _schema_name("ProcedureSpec")

    bindings: ArtifactBindings
    name: str
    version: ProcedureVersion
    task_family_id: str
    entry_step_id: str
    parameters: tuple[ProcedureParameter, ...] = ()
    locals: tuple[ProcedureLocal, ...] = ()
    preconditions: tuple[ProcedurePrecondition, ...] = ()
    declared_reads: tuple[str, ...] = ()
    declared_effects: tuple[ProcedureEffect, ...] = ()
    steps: tuple[ProcedureStep, ...] = ()
    branches: tuple[ProcedureBranch, ...] = ()
    loops: tuple[ProcedureLoop, ...] = ()
    holes: tuple[ProcedureHole, ...] = ()
    invariants: tuple[ProcedureInvariant, ...] = ()
    postconditions: tuple[ProcedurePostcondition, ...] = ()
    observations: tuple[ProcedureObservation, ...] = ()
    validation: ProcedureValidationPlan | None = None
    rollback: tuple[ProcedureRollback, ...] = ()
    fallback: tuple[ProcedureFallback, ...] = ()
    authority: ProcedureAuthorityEnvelope | None = None
    resources: ProcedureResourceEnvelope | None = None
    terminal_step_ids: tuple[str, ...] = ()
    scope_paths: tuple[str, ...] = ()
    provenance_cids: tuple[str, ...] = ()
    state: ArtifactState = ArtifactState.CANDIDATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        object.__setattr__(self, "name", _identifier(self.name, "name"))
        object.__setattr__(self, "version", _nested(self.version, ProcedureVersion, "version"))
        object.__setattr__(
            self, "task_family_id", _identifier(self.task_family_id, "task_family_id")
        )
        object.__setattr__(self, "entry_step_id", _identifier(self.entry_step_id, "entry_step_id"))
        record_fields: tuple[tuple[str, type[Any], int, bool], ...] = (
            ("parameters", ProcedureParameter, 64, False),
            ("locals", ProcedureLocal, 128, False),
            ("preconditions", ProcedurePrecondition, 64, True),
            ("declared_effects", ProcedureEffect, 128, False),
            ("steps", ProcedureStep, MAX_STEPS, True),
            ("branches", ProcedureBranch, MAX_BRANCHES, False),
            ("loops", ProcedureLoop, MAX_LOOPS, False),
            ("holes", ProcedureHole, MAX_HOLES, False),
            ("invariants", ProcedureInvariant, 64, False),
            ("postconditions", ProcedurePostcondition, 64, True),
            ("observations", ProcedureObservation, 128, True),
            ("rollback", ProcedureRollback, 32, False),
            ("fallback", ProcedureFallback, 32, False),
        )
        for field_name, cls, limit, required in record_fields:
            object.__setattr__(
                self,
                field_name,
                _records(
                    getattr(self, field_name), cls, field_name, limit=limit, required=required
                ),
            )
        object.__setattr__(
            self,
            "declared_reads",
            _strings(self.declared_reads, "declared_reads", paths=True, limit=MAX_SCOPE_PATHS),
        )
        if self.validation is None or self.authority is None or self.resources is None:
            raise ProcedureContractError(
                "validation, authority, and resources are required procedure envelopes"
            )
        object.__setattr__(
            self, "validation", _nested(self.validation, ProcedureValidationPlan, "validation")
        )
        object.__setattr__(
            self, "authority", _nested(self.authority, ProcedureAuthorityEnvelope, "authority")
        )
        object.__setattr__(
            self, "resources", _nested(self.resources, ProcedureResourceEnvelope, "resources")
        )
        object.__setattr__(
            self,
            "terminal_step_ids",
            _strings(self.terminal_step_ids, "terminal_step_ids", identifiers=True, required=True),
        )
        object.__setattr__(
            self,
            "scope_paths",
            _strings(
                self.scope_paths, "scope_paths", paths=True, limit=MAX_SCOPE_PATHS, required=True
            ),
        )
        object.__setattr__(
            self,
            "provenance_cids",
            _strings(self.provenance_cids, "provenance_cids", identifiers=True, required=True),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state in {ArtifactState.PROMOTED, ArtifactState.VERIFIED}:
            raise ProcedureContractError(
                "a ProcedureSpec cannot assert verification or promotion for itself"
            )
        _bounded(self, "ProcedureSpec")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "name": self.name,
            "version": self.version,
            "task_family_id": self.task_family_id,
            "entry_step_id": self.entry_step_id,
            "parameters": self.parameters,
            "locals": self.locals,
            "preconditions": self.preconditions,
            "declared_reads": self.declared_reads,
            "declared_effects": self.declared_effects,
            "steps": self.steps,
            "branches": self.branches,
            "loops": self.loops,
            "holes": self.holes,
            "invariants": self.invariants,
            "postconditions": self.postconditions,
            "observations": self.observations,
            "validation": self.validation,
            "rollback": self.rollback,
            "fallback": self.fallback,
            "authority": self.authority,
            "resources": self.resources,
            "terminal_step_ids": self.terminal_step_ids,
            "scope_paths": self.scope_paths,
            "provenance_cids": self.provenance_cids,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureSpec:
        fields = (
            "bindings",
            "name",
            "version",
            "task_family_id",
            "entry_step_id",
            "parameters",
            "locals",
            "preconditions",
            "declared_reads",
            "declared_effects",
            "steps",
            "branches",
            "loops",
            "holes",
            "invariants",
            "postconditions",
            "observations",
            "validation",
            "rollback",
            "fallback",
            "authority",
            "resources",
            "terminal_step_ids",
            "scope_paths",
            "provenance_cids",
            "state",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        nested_one = {
            "bindings": ArtifactBindings,
            "version": ProcedureVersion,
            "validation": ProcedureValidationPlan,
            "authority": ProcedureAuthorityEnvelope,
            "resources": ProcedureResourceEnvelope,
        }
        for name, nested_cls in nested_one.items():
            if name in values:
                values[name] = _nested(values[name], nested_cls, name)
        nested_many = {
            "parameters": ProcedureParameter,
            "locals": ProcedureLocal,
            "preconditions": ProcedurePrecondition,
            "declared_effects": ProcedureEffect,
            "steps": ProcedureStep,
            "branches": ProcedureBranch,
            "loops": ProcedureLoop,
            "holes": ProcedureHole,
            "invariants": ProcedureInvariant,
            "postconditions": ProcedurePostcondition,
            "observations": ProcedureObservation,
            "rollback": ProcedureRollback,
            "fallback": ProcedureFallback,
        }
        for name, nested_cls in nested_many.items():
            if name in values:
                values[name] = _records(values[name], nested_cls, name)
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureCandidate(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureCandidate")

    bindings: ArtifactBindings
    procedure: ProcedureSpec
    synthesis_plan_cid: str
    source_episode_cids: tuple[str, ...]
    counterexample_set_cid: str
    state: ArtifactState = ArtifactState.CANDIDATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        object.__setattr__(self, "procedure", _nested(self.procedure, ProcedureSpec, "procedure"))
        object.__setattr__(
            self, "synthesis_plan_cid", _identifier(self.synthesis_plan_cid, "synthesis_plan_cid")
        )
        object.__setattr__(
            self,
            "source_episode_cids",
            _strings(
                self.source_episode_cids, "source_episode_cids", identifiers=True, required=True
            ),
        )
        object.__setattr__(
            self,
            "counterexample_set_cid",
            _identifier(self.counterexample_set_cid, "counterexample_set_cid"),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state not in {
            ArtifactState.CANDIDATE,
            ArtifactState.DEVELOPMENT,
            ArtifactState.REJECTED,
        }:
            raise ProcedureContractError("a candidate cannot assert verified or promoted status")
        if self.bindings != self.procedure.bindings:
            raise ProcedureContractError("candidate and procedure exact bindings differ")
        _bounded(self, "ProcedureCandidate")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "procedure": self.procedure,
            "synthesis_plan_cid": self.synthesis_plan_cid,
            "source_episode_cids": self.source_episode_cids,
            "counterexample_set_cid": self.counterexample_set_cid,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureCandidate:
        fields = (
            "bindings",
            "procedure",
            "synthesis_plan_cid",
            "source_episode_cids",
            "counterexample_set_cid",
            "state",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        if "procedure" in values:
            values["procedure"] = _nested(values["procedure"], ProcedureSpec, "procedure")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureCertificate(CanonicalContract):
    """External verification evidence; identity alone grants no authority."""

    SCHEMA: ClassVar[str] = _schema_name("ProcedureCertificate")

    bindings: ArtifactBindings
    procedure_cid: str
    procedure_version: ProcedureVersion
    task_family_cid: str
    source_episode_cids: tuple[str, ...]
    specification_cids: tuple[str, ...]
    counterexample_set_cid: str
    operation_catalog_revision: str
    effect_policy_revision: str
    authority_policy_revision: str
    verification_policy_revision: str
    repository_families: tuple[str, ...]
    supported_language_classes: tuple[str, ...]
    supported_framework_classes: tuple[str, ...]
    risk_ceiling: RiskClass
    proof_receipt_cids: tuple[str, ...]
    test_receipt_cids: tuple[str, ...]
    adversarial_assurance_cids: tuple[str, ...]
    held_out_evaluation_cid: str
    shadow_evaluation_cid: str
    known_limitations: tuple[str, ...]
    issuer: str
    signature: str
    issued_at_ms: int
    expires_at_ms: int
    state: ArtifactState = ArtifactState.VERIFIED

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        object.__setattr__(self, "procedure_cid", _identifier(self.procedure_cid, "procedure_cid"))
        object.__setattr__(
            self,
            "procedure_version",
            _nested(self.procedure_version, ProcedureVersion, "procedure_version"),
        )
        for name in (
            "task_family_cid",
            "counterexample_set_cid",
            "operation_catalog_revision",
            "effect_policy_revision",
            "authority_policy_revision",
            "verification_policy_revision",
            "held_out_evaluation_cid",
            "shadow_evaluation_cid",
            "issuer",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "source_episode_cids",
            "specification_cids",
            "repository_families",
            "supported_language_classes",
            "supported_framework_classes",
            "proof_receipt_cids",
            "test_receipt_cids",
            "adversarial_assurance_cids",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True, required=True),
            )
        object.__setattr__(
            self,
            "known_limitations",
            _strings(self.known_limitations, "known_limitations", limit=64),
        )
        object.__setattr__(
            self, "risk_ceiling", _enum(self.risk_ceiling, RiskClass, "risk_ceiling")
        )
        object.__setattr__(self, "signature", _text(self.signature, "signature"))
        object.__setattr__(
            self, "issued_at_ms", _nonnegative_int(self.issued_at_ms, "issued_at_ms")
        )
        object.__setattr__(
            self, "expires_at_ms", _positive_int(self.expires_at_ms, "expires_at_ms")
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.expires_at_ms <= self.issued_at_ms:
            raise ProcedureContractError("certificate expiry must follow issuance")
        if self.authority_policy_revision != self.bindings.policy_revision:
            raise ProcedureContractError(
                "certificate authority policy is not exact-binding current"
            )
        if self.state not in {
            ArtifactState.VERIFIED,
            ArtifactState.PROMOTED,
            ArtifactState.STALE,
            ArtifactState.REVOKED,
            ArtifactState.SUPERSEDED,
            ArtifactState.REJECTED,
        }:
            raise ProcedureContractError("certificate state is not a certificate-tier state")
        _bounded(self, "ProcedureCertificate")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "procedure_cid": self.procedure_cid,
            "procedure_version": self.procedure_version,
            "task_family_cid": self.task_family_cid,
            "source_episode_cids": self.source_episode_cids,
            "specification_cids": self.specification_cids,
            "counterexample_set_cid": self.counterexample_set_cid,
            "operation_catalog_revision": self.operation_catalog_revision,
            "effect_policy_revision": self.effect_policy_revision,
            "authority_policy_revision": self.authority_policy_revision,
            "verification_policy_revision": self.verification_policy_revision,
            "repository_families": self.repository_families,
            "supported_language_classes": self.supported_language_classes,
            "supported_framework_classes": self.supported_framework_classes,
            "risk_ceiling": self.risk_ceiling.value,
            "proof_receipt_cids": self.proof_receipt_cids,
            "test_receipt_cids": self.test_receipt_cids,
            "adversarial_assurance_cids": self.adversarial_assurance_cids,
            "held_out_evaluation_cid": self.held_out_evaluation_cid,
            "shadow_evaluation_cid": self.shadow_evaluation_cid,
            "known_limitations": self.known_limitations,
            "issuer": self.issuer,
            "signature": self.signature,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureCertificate:
        fields = (
            "bindings",
            "procedure_cid",
            "procedure_version",
            "task_family_cid",
            "source_episode_cids",
            "specification_cids",
            "counterexample_set_cid",
            "operation_catalog_revision",
            "effect_policy_revision",
            "authority_policy_revision",
            "verification_policy_revision",
            "repository_families",
            "supported_language_classes",
            "supported_framework_classes",
            "risk_ceiling",
            "proof_receipt_cids",
            "test_receipt_cids",
            "adversarial_assurance_cids",
            "held_out_evaluation_cid",
            "shadow_evaluation_cid",
            "known_limitations",
            "issuer",
            "signature",
            "issued_at_ms",
            "expires_at_ms",
            "state",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        if "procedure_version" in values:
            values["procedure_version"] = _nested(
                values["procedure_version"], ProcedureVersion, "procedure_version"
            )
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureInvocation(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureInvocation")

    bindings: ArtifactBindings
    procedure_cid: str
    certificate_cid: str
    registry_revision: str
    parameters: Mapping[str, Any]
    requested_scope: tuple[str, ...]
    authority_receipt_cids: tuple[str, ...]
    idempotency_key: str
    dry_run: bool
    requested_at_ms: int
    lease_id: str = ""
    fencing_token: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        for name in ("procedure_cid", "certificate_cid", "registry_revision", "idempotency_key"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        parameters = _freeze(self.parameters, "parameters")
        if not isinstance(parameters, Mapping):
            raise ProcedureContractError("parameters must be a mapping")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(
            self,
            "requested_scope",
            _strings(self.requested_scope, "requested_scope", paths=True, required=True),
        )
        object.__setattr__(
            self,
            "authority_receipt_cids",
            _strings(self.authority_receipt_cids, "authority_receipt_cids", identifiers=True),
        )
        if type(self.dry_run) is not bool:
            raise ProcedureContractError("dry_run must be a boolean")
        object.__setattr__(
            self, "requested_at_ms", _nonnegative_int(self.requested_at_ms, "requested_at_ms")
        )
        object.__setattr__(self, "lease_id", _identifier(self.lease_id, "lease_id", required=False))
        object.__setattr__(
            self, "fencing_token", _nonnegative_int(self.fencing_token, "fencing_token")
        )
        if bool(self.lease_id) != bool(self.fencing_token):
            raise ProcedureContractError(
                "lease_id and nonzero fencing_token must be bound together"
            )
        _bounded(self, "ProcedureInvocation")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "procedure_cid": self.procedure_cid,
            "certificate_cid": self.certificate_cid,
            "registry_revision": self.registry_revision,
            "parameters": self.parameters,
            "requested_scope": self.requested_scope,
            "authority_receipt_cids": self.authority_receipt_cids,
            "idempotency_key": self.idempotency_key,
            "dry_run": self.dry_run,
            "requested_at_ms": self.requested_at_ms,
            "lease_id": self.lease_id,
            "fencing_token": self.fencing_token,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureInvocation:
        fields = (
            "bindings",
            "procedure_cid",
            "certificate_cid",
            "registry_revision",
            "parameters",
            "requested_scope",
            "authority_receipt_cids",
            "idempotency_key",
            "dry_run",
            "requested_at_ms",
            "lease_id",
            "fencing_token",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureTraceEntry(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureTraceEntry")

    sequence: int
    step_id: str
    operation: StepOperation
    status: TraceEventStatus
    attempt: int
    started_at_ms: int
    ended_at_ms: int
    input_digest: str
    output_digest: str = ""
    observed_effect_ids: tuple[str, ...] = ()
    evidence_cids: tuple[str, ...] = ()
    failure_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _nonnegative_int(self.sequence, "sequence"))
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(self, "operation", _enum(self.operation, StepOperation, "operation"))
        object.__setattr__(self, "status", _enum(self.status, TraceEventStatus, "status"))
        object.__setattr__(self, "attempt", _positive_int(self.attempt, "attempt", maximum=64))
        object.__setattr__(
            self, "started_at_ms", _nonnegative_int(self.started_at_ms, "started_at_ms")
        )
        object.__setattr__(self, "ended_at_ms", _nonnegative_int(self.ended_at_ms, "ended_at_ms"))
        object.__setattr__(self, "input_digest", _identifier(self.input_digest, "input_digest"))
        object.__setattr__(
            self, "output_digest", _identifier(self.output_digest, "output_digest", required=False)
        )
        object.__setattr__(
            self,
            "observed_effect_ids",
            _strings(self.observed_effect_ids, "observed_effect_ids", identifiers=True),
        )
        object.__setattr__(
            self, "evidence_cids", _strings(self.evidence_cids, "evidence_cids", identifiers=True)
        )
        object.__setattr__(
            self, "failure_code", _identifier(self.failure_code, "failure_code", required=False)
        )
        if self.ended_at_ms and self.ended_at_ms < self.started_at_ms:
            raise ProcedureContractError("trace entry ended before it started")
        if self.status is TraceEventStatus.FAILED and not self.failure_code:
            raise ProcedureContractError("failed trace entries require a typed failure code")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "sequence": self.sequence,
            "step_id": self.step_id,
            "operation": self.operation.value,
            "status": self.status.value,
            "attempt": self.attempt,
            "started_at_ms": self.started_at_ms,
            "ended_at_ms": self.ended_at_ms,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "observed_effect_ids": self.observed_effect_ids,
            "evidence_cids": self.evidence_cids,
            "failure_code": self.failure_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureTraceEntry:
        fields = (
            "sequence",
            "step_id",
            "operation",
            "status",
            "attempt",
            "started_at_ms",
            "ended_at_ms",
            "input_digest",
            "output_digest",
            "observed_effect_ids",
            "evidence_cids",
            "failure_code",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureExecutionTrace(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureExecutionTrace")

    bindings: ArtifactBindings
    invocation_cid: str
    procedure_cid: str
    entries: tuple[ProcedureTraceEntry, ...]
    checkpoint_cids: tuple[str, ...]
    state: TraceState

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        object.__setattr__(
            self, "invocation_cid", _identifier(self.invocation_cid, "invocation_cid")
        )
        object.__setattr__(self, "procedure_cid", _identifier(self.procedure_cid, "procedure_cid"))
        object.__setattr__(
            self, "entries", _records(self.entries, ProcedureTraceEntry, "entries", limit=1_024)
        )
        object.__setattr__(
            self,
            "checkpoint_cids",
            _strings(self.checkpoint_cids, "checkpoint_cids", identifiers=True),
        )
        object.__setattr__(self, "state", _enum(self.state, TraceState, "state"))
        sequences = tuple(entry.sequence for entry in self.entries)
        if sequences != tuple(range(len(sequences))):
            raise ProcedureContractError("trace entry sequences must be contiguous from zero")
        _bounded(self, "ProcedureExecutionTrace")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "invocation_cid": self.invocation_cid,
            "procedure_cid": self.procedure_cid,
            "entries": self.entries,
            "checkpoint_cids": self.checkpoint_cids,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureExecutionTrace:
        fields = (
            "bindings",
            "invocation_cid",
            "procedure_cid",
            "entries",
            "checkpoint_cids",
            "state",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        if "entries" in values:
            values["entries"] = _records(
                values["entries"], ProcedureTraceEntry, "entries", limit=1_024
            )
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureOutcome(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureOutcome")

    bindings: ArtifactBindings
    invocation_cid: str
    procedure_cid: str
    status: ProcedureOutcomeStatus
    observed_effect_ids: tuple[str, ...]
    validation_receipt_cids: tuple[str, ...]
    satisfied_postcondition_ids: tuple[str, ...]
    rollback_receipt_cids: tuple[str, ...]
    trace_cid: str
    terminal_at_ms: int
    failure_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        for name in ("invocation_cid", "procedure_cid", "trace_cid"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "status", _enum(self.status, ProcedureOutcomeStatus, "status"))
        for name in (
            "observed_effect_ids",
            "validation_receipt_cids",
            "satisfied_postcondition_ids",
            "rollback_receipt_cids",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name, identifiers=True))
        object.__setattr__(
            self, "failure_cid", _identifier(self.failure_cid, "failure_cid", required=False)
        )
        object.__setattr__(
            self, "terminal_at_ms", _nonnegative_int(self.terminal_at_ms, "terminal_at_ms")
        )
        if self.status is ProcedureOutcomeStatus.SUCCEEDED:
            if (
                self.failure_cid
                or not self.validation_receipt_cids
                or not self.satisfied_postcondition_ids
            ):
                raise ProcedureContractError(
                    "successful outcomes require validation and postconditions "
                    "and cannot bind failure"
                )
        elif (
            self.status
            not in {
                ProcedureOutcomeStatus.CANCELLED,
                ProcedureOutcomeStatus.REFUSED,
            }
            and not self.failure_cid
        ):
            raise ProcedureContractError("non-success outcomes require a typed failure artifact")
        _bounded(self, "ProcedureOutcome")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "invocation_cid": self.invocation_cid,
            "procedure_cid": self.procedure_cid,
            "status": self.status.value,
            "observed_effect_ids": self.observed_effect_ids,
            "validation_receipt_cids": self.validation_receipt_cids,
            "satisfied_postcondition_ids": self.satisfied_postcondition_ids,
            "rollback_receipt_cids": self.rollback_receipt_cids,
            "trace_cid": self.trace_cid,
            "terminal_at_ms": self.terminal_at_ms,
            "failure_cid": self.failure_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureOutcome:
        fields = (
            "bindings",
            "invocation_cid",
            "procedure_cid",
            "status",
            "observed_effect_ids",
            "validation_receipt_cids",
            "satisfied_postcondition_ids",
            "rollback_receipt_cids",
            "trace_cid",
            "terminal_at_ms",
            "failure_cid",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureInvocationReceipt(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureInvocationReceipt")

    bindings: ArtifactBindings
    invocation_cid: str
    procedure_cid: str
    certificate_cid: str
    trace_cid: str
    outcome_cid: str
    admitted_evidence_cids: tuple[str, ...]
    emitted_at_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        for name in (
            "invocation_cid",
            "procedure_cid",
            "certificate_cid",
            "trace_cid",
            "outcome_cid",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "admitted_evidence_cids",
            _strings(
                self.admitted_evidence_cids,
                "admitted_evidence_cids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self, "emitted_at_ms", _nonnegative_int(self.emitted_at_ms, "emitted_at_ms")
        )
        _bounded(self, "ProcedureInvocationReceipt")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "invocation_cid": self.invocation_cid,
            "procedure_cid": self.procedure_cid,
            "certificate_cid": self.certificate_cid,
            "trace_cid": self.trace_cid,
            "outcome_cid": self.outcome_cid,
            "admitted_evidence_cids": self.admitted_evidence_cids,
            "emitted_at_ms": self.emitted_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureInvocationReceipt:
        fields = (
            "bindings",
            "invocation_cid",
            "procedure_cid",
            "certificate_cid",
            "trace_cid",
            "outcome_cid",
            "admitted_evidence_cids",
            "emitted_at_ms",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureFailure(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureFailure")

    bindings: ArtifactBindings
    invocation_cid: str
    procedure_cid: str
    step_id: str
    failure_code: str
    retryable: bool
    diagnostic_cids: tuple[str, ...]
    observed_at_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        for name in ("invocation_cid", "procedure_cid", "step_id", "failure_code"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if type(self.retryable) is not bool:
            raise ProcedureContractError("retryable must be a boolean")
        object.__setattr__(
            self,
            "diagnostic_cids",
            _strings(self.diagnostic_cids, "diagnostic_cids", identifiers=True),
        )
        object.__setattr__(
            self, "observed_at_ms", _nonnegative_int(self.observed_at_ms, "observed_at_ms")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "invocation_cid": self.invocation_cid,
            "procedure_cid": self.procedure_cid,
            "step_id": self.step_id,
            "failure_code": self.failure_code,
            "retryable": self.retryable,
            "diagnostic_cids": self.diagnostic_cids,
            "observed_at_ms": self.observed_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureFailure:
        fields = (
            "bindings",
            "invocation_cid",
            "procedure_cid",
            "step_id",
            "failure_code",
            "retryable",
            "diagnostic_cids",
            "observed_at_ms",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureRecoveryPlan(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ProcedureRecoveryPlan")

    bindings: ArtifactBindings
    failure_cid: str
    recovery_step_ids: tuple[str, ...]
    maximum_attempts: int
    requires_new_evidence: bool
    escalation_target: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        object.__setattr__(self, "failure_cid", _identifier(self.failure_cid, "failure_cid"))
        object.__setattr__(
            self,
            "recovery_step_ids",
            _strings(self.recovery_step_ids, "recovery_step_ids", identifiers=True, required=True),
        )
        object.__setattr__(
            self,
            "maximum_attempts",
            _positive_int(self.maximum_attempts, "maximum_attempts", maximum=8),
        )
        if type(self.requires_new_evidence) is not bool:
            raise ProcedureContractError("requires_new_evidence must be a boolean")
        object.__setattr__(
            self, "escalation_target", _identifier(self.escalation_target, "escalation_target")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "failure_cid": self.failure_cid,
            "recovery_step_ids": self.recovery_step_ids,
            "maximum_attempts": self.maximum_attempts,
            "requires_new_evidence": self.requires_new_evidence,
            "escalation_target": self.escalation_target,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureRecoveryPlan:
        fields = (
            "bindings",
            "failure_cid",
            "recovery_step_ids",
            "maximum_attempts",
            "requires_new_evidence",
            "escalation_target",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class TrajectoryStep(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("TrajectoryStep")

    sequence: int
    operation: StepOperation
    operation_contract: str
    initial_state_cid: str
    terminal_state_cid: str
    observation_cids: tuple[str, ...]
    effect_ids: tuple[str, ...]
    validation_receipt_cids: tuple[str, ...]
    hole_type: str = ""
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    human_interventions: int = 0
    status: TraceEventStatus = TraceEventStatus.SUCCEEDED

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _nonnegative_int(self.sequence, "sequence"))
        object.__setattr__(self, "operation", _enum(self.operation, StepOperation, "operation"))
        for name in ("operation_contract", "initial_state_cid", "terminal_state_cid"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in ("observation_cids", "effect_ids", "validation_receipt_cids"):
            object.__setattr__(self, name, _strings(getattr(self, name), name, identifiers=True))
        object.__setattr__(
            self, "hole_type", _identifier(self.hole_type, "hole_type", required=False)
        )
        for name in (
            "model_calls",
            "input_tokens",
            "output_tokens",
            "latency_ms",
            "human_interventions",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        object.__setattr__(self, "status", _enum(self.status, TraceEventStatus, "status"))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "sequence": self.sequence,
            "operation": self.operation.value,
            "operation_contract": self.operation_contract,
            "initial_state_cid": self.initial_state_cid,
            "terminal_state_cid": self.terminal_state_cid,
            "observation_cids": self.observation_cids,
            "effect_ids": self.effect_ids,
            "validation_receipt_cids": self.validation_receipt_cids,
            "hole_type": self.hole_type,
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "human_interventions": self.human_interventions,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrajectoryStep:
        fields = (
            "sequence",
            "operation",
            "operation_contract",
            "initial_state_cid",
            "terminal_state_cid",
            "observation_cids",
            "effect_ids",
            "validation_receipt_cids",
            "hole_type",
            "model_calls",
            "input_tokens",
            "output_tokens",
            "latency_ms",
            "human_interventions",
            "status",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class TrajectoryOutcome(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("TrajectoryOutcome")

    status: TrajectoryTerminalStatus
    accepted_criterion_ids: tuple[str, ...]
    validation_receipt_cids: tuple[str, ...]
    proof_receipt_cids: tuple[str, ...]
    rejection_reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(self.status, TrajectoryTerminalStatus, "status"))
        for name in ("accepted_criterion_ids", "validation_receipt_cids", "proof_receipt_cids"):
            object.__setattr__(self, name, _strings(getattr(self, name), name, identifiers=True))
        object.__setattr__(
            self,
            "rejection_reason_code",
            _identifier(self.rejection_reason_code, "rejection_reason_code", required=False),
        )
        if self.status is TrajectoryTerminalStatus.ACCEPTED:
            if not self.accepted_criterion_ids or not self.validation_receipt_cids:
                raise ProcedureContractError(
                    "accepted trajectories require criteria and validation"
                )
            if self.rejection_reason_code:
                raise ProcedureContractError(
                    "accepted trajectories cannot carry a rejection reason"
                )
        elif not self.rejection_reason_code and self.status is TrajectoryTerminalStatus.REJECTED:
            raise ProcedureContractError("rejected trajectories require a typed reason")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "status": self.status.value,
            "accepted_criterion_ids": self.accepted_criterion_ids,
            "validation_receipt_cids": self.validation_receipt_cids,
            "proof_receipt_cids": self.proof_receipt_cids,
            "rejection_reason_code": self.rejection_reason_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrajectoryOutcome:
        fields = (
            "status",
            "accepted_criterion_ids",
            "validation_receipt_cids",
            "proof_receipt_cids",
            "rejection_reason_code",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ExecutionTrajectory(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("ExecutionTrajectory")

    bindings: ArtifactBindings
    source_episode_cid: str
    source_episode_kind: EpisodeKind
    initial_abstract_state_cid: str
    terminal_abstract_state_cid: str
    objective_criterion_ids: tuple[str, ...]
    task_family_hint: str
    steps: tuple[TrajectoryStep, ...]
    outcome: TrajectoryOutcome
    total_cost_units: int
    total_tokens: int
    total_latency_ms: int
    human_interventions: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        for name in (
            "source_episode_cid",
            "initial_abstract_state_cid",
            "terminal_abstract_state_cid",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "source_episode_kind",
            _enum(self.source_episode_kind, EpisodeKind, "source_episode_kind"),
        )
        object.__setattr__(
            self,
            "objective_criterion_ids",
            _strings(
                self.objective_criterion_ids,
                "objective_criterion_ids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "task_family_hint",
            _identifier(self.task_family_hint, "task_family_hint", required=False),
        )
        object.__setattr__(
            self,
            "steps",
            _records(self.steps, TrajectoryStep, "steps", limit=MAX_STEPS, required=True),
        )
        object.__setattr__(self, "outcome", _nested(self.outcome, TrajectoryOutcome, "outcome"))
        for name in ("total_cost_units", "total_tokens", "total_latency_ms", "human_interventions"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        sequences = tuple(step.sequence for step in self.steps)
        if sequences != tuple(range(len(sequences))):
            raise ProcedureContractError("trajectory sequences must be contiguous from zero")
        if self.outcome.status is TrajectoryTerminalStatus.ACCEPTED:
            if self.source_episode_kind not in {
                EpisodeKind.ACCEPTED_TASK_RECEIPT,
                EpisodeKind.CURRENT_TREE_POST_MERGE_RECEIPT,
                EpisodeKind.VERIFIED_PROOF_RECEIPT,
                EpisodeKind.ADMITTED_TEST_RECEIPT,
                EpisodeKind.SUCCESSFUL_ROLLBACK_RECEIPT,
                EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT,
            }:
                raise ProcedureContractError(
                    "source episode kind cannot demonstrate accepted success"
                )
        _bounded(self, "ExecutionTrajectory")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "source_episode_cid": self.source_episode_cid,
            "source_episode_kind": self.source_episode_kind.value,
            "initial_abstract_state_cid": self.initial_abstract_state_cid,
            "terminal_abstract_state_cid": self.terminal_abstract_state_cid,
            "objective_criterion_ids": self.objective_criterion_ids,
            "task_family_hint": self.task_family_hint,
            "steps": self.steps,
            "outcome": self.outcome,
            "total_cost_units": self.total_cost_units,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "human_interventions": self.human_interventions,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionTrajectory:
        fields = (
            "bindings",
            "source_episode_cid",
            "source_episode_kind",
            "initial_abstract_state_cid",
            "terminal_abstract_state_cid",
            "objective_criterion_ids",
            "task_family_hint",
            "steps",
            "outcome",
            "total_cost_units",
            "total_tokens",
            "total_latency_ms",
            "human_interventions",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        if "steps" in values:
            values["steps"] = _records(values["steps"], TrajectoryStep, "steps", limit=MAX_STEPS)
        if "outcome" in values:
            values["outcome"] = _nested(values["outcome"], TrajectoryOutcome, "outcome")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class TrajectoryNormalizationReceipt(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("TrajectoryNormalizationReceipt")

    bindings: ArtifactBindings
    source_episode_cid: str
    trajectory_cid: str
    admitted_evidence_cids: tuple[str, ...]
    removed_field_classes: tuple[str, ...]
    normalizer_revision: str
    emitted_at_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        for name in ("source_episode_cid", "trajectory_cid", "normalizer_revision"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "admitted_evidence_cids",
            _strings(
                self.admitted_evidence_cids,
                "admitted_evidence_cids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "removed_field_classes",
            _strings(self.removed_field_classes, "removed_field_classes", identifiers=True),
        )
        object.__setattr__(
            self, "emitted_at_ms", _nonnegative_int(self.emitted_at_ms, "emitted_at_ms")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "source_episode_cid": self.source_episode_cid,
            "trajectory_cid": self.trajectory_cid,
            "admitted_evidence_cids": self.admitted_evidence_cids,
            "removed_field_classes": self.removed_field_classes,
            "normalizer_revision": self.normalizer_revision,
            "emitted_at_ms": self.emitted_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrajectoryNormalizationReceipt:
        fields = (
            "bindings",
            "source_episode_cid",
            "trajectory_cid",
            "admitted_evidence_cids",
            "removed_field_classes",
            "normalizer_revision",
            "emitted_at_ms",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class TaskFamilyBoundary(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("TaskFamilyBoundary")

    positive_member_cids: tuple[str, ...]
    negative_example_cids: tuple[str, ...]
    boundary_example_cids: tuple[str, ...]
    unknown_case_cids: tuple[str, ...]
    risk_ceiling: RiskClass
    permitted_repositories: tuple[str, ...]
    permitted_languages: tuple[str, ...]
    permitted_frameworks: tuple[str, ...]
    permitted_effect_classes: tuple[EffectClass, ...]

    def __post_init__(self) -> None:
        for name in (
            "positive_member_cids",
            "negative_example_cids",
            "boundary_example_cids",
            "unknown_case_cids",
            "permitted_repositories",
            "permitted_languages",
            "permitted_frameworks",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    name,
                    identifiers=True,
                    required=name
                    in {
                        "positive_member_cids",
                        "negative_example_cids",
                        "boundary_example_cids",
                        "permitted_repositories",
                        "permitted_languages",
                    },
                ),
            )
        object.__setattr__(
            self, "risk_ceiling", _enum(self.risk_ceiling, RiskClass, "risk_ceiling")
        )
        effects = _enums(
            self.permitted_effect_classes,
            EffectClass,
            "permitted_effect_classes",
            limit=len(EffectClass),
            required=True,
        )
        object.__setattr__(self, "permitted_effect_classes", effects)
        sets = (
            set(self.positive_member_cids),
            set(self.negative_example_cids),
            set(self.boundary_example_cids),
            set(self.unknown_case_cids),
        )
        if any(
            left.intersection(right)
            for index, left in enumerate(sets)
            for right in sets[index + 1 :]
        ):
            raise ProcedureContractError("family boundary example classes must be disjoint")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "positive_member_cids": self.positive_member_cids,
            "negative_example_cids": self.negative_example_cids,
            "boundary_example_cids": self.boundary_example_cids,
            "unknown_case_cids": self.unknown_case_cids,
            "risk_ceiling": self.risk_ceiling.value,
            "permitted_repositories": self.permitted_repositories,
            "permitted_languages": self.permitted_languages,
            "permitted_frameworks": self.permitted_frameworks,
            "permitted_effect_classes": tuple(item.value for item in self.permitted_effect_classes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskFamilyBoundary:
        fields = (
            "positive_member_cids",
            "negative_example_cids",
            "boundary_example_cids",
            "unknown_case_cids",
            "risk_ceiling",
            "permitted_repositories",
            "permitted_languages",
            "permitted_frameworks",
            "permitted_effect_classes",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class TaskFamily(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("TaskFamily")

    bindings: ArtifactBindings
    name: str
    goal_semantics: tuple[str, ...]
    precondition_shape: tuple[str, ...]
    affected_artifact_classes: tuple[str, ...]
    effect_classes: tuple[EffectClass, ...]
    required_operation_contracts: tuple[str, ...]
    validation_structure: tuple[str, ...]
    failure_signatures: tuple[str, ...]
    postcondition_shape: tuple[str, ...]
    rollback_structure: tuple[str, ...]
    boundary: TaskFamilyBoundary
    state: ArtifactState = ArtifactState.CANDIDATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        object.__setattr__(self, "name", _identifier(self.name, "name"))
        for name in (
            "goal_semantics",
            "precondition_shape",
            "affected_artifact_classes",
            "required_operation_contracts",
            "validation_structure",
            "failure_signatures",
            "postcondition_shape",
            "rollback_structure",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True, required=True),
            )
        effects = _enums(
            self.effect_classes,
            EffectClass,
            "effect_classes",
            limit=len(EffectClass),
            required=True,
        )
        object.__setattr__(self, "effect_classes", effects)
        object.__setattr__(self, "boundary", _nested(self.boundary, TaskFamilyBoundary, "boundary"))
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if not set(effects).issubset(set(self.boundary.permitted_effect_classes)):
            raise ProcedureContractError("family effects exceed its declared boundary")
        _bounded(self, "TaskFamily")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "name": self.name,
            "goal_semantics": self.goal_semantics,
            "precondition_shape": self.precondition_shape,
            "affected_artifact_classes": self.affected_artifact_classes,
            "effect_classes": tuple(item.value for item in self.effect_classes),
            "required_operation_contracts": self.required_operation_contracts,
            "validation_structure": self.validation_structure,
            "failure_signatures": self.failure_signatures,
            "postcondition_shape": self.postcondition_shape,
            "rollback_structure": self.rollback_structure,
            "boundary": self.boundary,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskFamily:
        fields = (
            "bindings",
            "name",
            "goal_semantics",
            "precondition_shape",
            "affected_artifact_classes",
            "effect_classes",
            "required_operation_contracts",
            "validation_structure",
            "failure_signatures",
            "postcondition_shape",
            "rollback_structure",
            "boundary",
            "state",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        if "boundary" in values:
            values["boundary"] = _nested(values["boundary"], TaskFamilyBoundary, "boundary")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class TaskFamilyMembership(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("TaskFamilyMembership")

    bindings: ArtifactBindings
    task_family_cid: str
    trajectory_cid: str
    membership: FamilyMembershipClass
    evidence_cids: tuple[str, ...]
    classifier_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        for name in ("task_family_cid", "trajectory_cid", "classifier_revision"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "membership", _enum(self.membership, FamilyMembershipClass, "membership")
        )
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True, required=True),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "task_family_cid": self.task_family_cid,
            "trajectory_cid": self.trajectory_cid,
            "membership": self.membership.value,
            "evidence_cids": self.evidence_cids,
            "classifier_revision": self.classifier_revision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskFamilyMembership:
        fields = (
            "bindings",
            "task_family_cid",
            "trajectory_cid",
            "membership",
            "evidence_cids",
            "classifier_revision",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class TaskFamilyCounterexample(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("TaskFamilyCounterexample")

    bindings: ArtifactBindings
    task_family_cid: str
    example_cid: str
    violation_class: str
    conflicting_authority_classes: tuple[str, ...] = ()
    conflicting_effect_classes: tuple[EffectClass, ...] = ()
    conflicting_validation_classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        for name in ("task_family_cid", "example_cid", "violation_class"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "conflicting_authority_classes",
            _strings(
                self.conflicting_authority_classes,
                "conflicting_authority_classes",
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "conflicting_validation_classes",
            _strings(
                self.conflicting_validation_classes,
                "conflicting_validation_classes",
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "conflicting_effect_classes",
            _enums(
                self.conflicting_effect_classes,
                EffectClass,
                "conflicting_effect_classes",
                limit=len(EffectClass),
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "task_family_cid": self.task_family_cid,
            "example_cid": self.example_cid,
            "violation_class": self.violation_class,
            "conflicting_authority_classes": self.conflicting_authority_classes,
            "conflicting_effect_classes": tuple(
                item.value for item in self.conflicting_effect_classes
            ),
            "conflicting_validation_classes": self.conflicting_validation_classes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskFamilyCounterexample:
        fields = (
            "bindings",
            "task_family_cid",
            "example_cid",
            "violation_class",
            "conflicting_authority_classes",
            "conflicting_effect_classes",
            "conflicting_validation_classes",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class BoundedArtifact(CanonicalContract):
    """Closed generic envelope for later-tranche artifacts.

    The generic envelope intentionally carries references and bounded facts,
    never large bodies.  A later tranche may replace a named subclass with a
    richer closed schema without weakening this P0 storage boundary.
    """

    bindings: ArtifactBindings
    artifact_version: int = 1
    state: ArtifactState = ArtifactState.CANDIDATE
    subject_cid: str = ""
    reference_cids: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        object.__setattr__(
            self,
            "artifact_version",
            _positive_int(self.artifact_version, "artifact_version", maximum=65_535),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        object.__setattr__(
            self, "subject_cid", _identifier(self.subject_cid, "subject_cid", required=False)
        )
        object.__setattr__(
            self,
            "reference_cids",
            _strings(self.reference_cids, "reference_cids", identifiers=True),
        )
        object.__setattr__(self, "labels", _strings(self.labels, "labels", identifiers=True))
        facts = _freeze(self.facts, "facts")
        if not isinstance(facts, Mapping):
            raise ProcedureContractError("facts must be a mapping")
        object.__setattr__(self, "facts", facts)
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        _bounded(self, self.__class__.__name__)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "artifact_version": self.artifact_version,
            "state": self.state.value,
            "subject_cid": self.subject_cid,
            "reference_cids": self.reference_cids,
            "labels": self.labels,
            "facts": self.facts,
            "created_at_ms": self.created_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BoundedArtifact:
        fields = (
            "bindings",
            "artifact_version",
            "state",
            "subject_cid",
            "reference_cids",
            "labels",
            "facts",
            "created_at_ms",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


_GENERIC_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    "ProcedureSynthesisPlan",
    "ProcedureSynthesisCounterexample",
    "ProcedureVerificationResult",
    "SpecificationCandidate",
    "SpecificationEvidence",
    "SpecificationCounterexample",
    "SpecificationMiningReceipt",
    "InvariantCandidate",
    "InvariantValidationReceipt",
    "NonVacuityReceipt",
    "AntiUnificationPattern",
    "GeneralizationBoundary",
    "GeneralizationCounterexample",
    "ProcedureRegistry",
    "ProcedureRegistryRevision",
    "ProcedurePromotionReceipt",
    "ProcedureRollbackReceipt",
    "ProcedureDeprecationReceipt",
    "ProcedureDriftReport",
    "HoleRequest",
    "HoleCandidate",
    "HoleResolution",
    "HoleValidationReceipt",
    "DistillationCorpus",
    "DistillationExample",
    "DistillationEvaluation",
    "LocalDecisionModelArtifact",
    "GeneratedToolSpec",
    "GeneratedToolCandidate",
    "GeneratedToolCertificate",
    "GeneratedToolInvocationReceipt",
    "ExperimentPlan",
    "ExperimentObservation",
    "ExperimentEvaluation",
    "ProcedureCompilerRunReceipt",
    "ProcedureCompilerReleaseReceipt",
)


for _artifact_name in _GENERIC_ARTIFACT_NAMES:
    globals()[_artifact_name] = type(
        _artifact_name,
        (BoundedArtifact,),
        {
            "__module__": __name__,
            "__doc__": f"Bounded P0 contract for {_artifact_name}.",
            "SCHEMA": _schema_name(_artifact_name),
        },
    )


ARTIFACT_TYPES_BY_SCHEMA: dict[str, type[CanonicalContract]] = {
    cls.SCHEMA: cls
    for cls in (
        ArtifactBindings,
        ProcedureVersion,
        ProcedureParameter,
        ProcedureLocal,
        ProcedureEffect,
        ProcedurePrecondition,
        ProcedureInvariant,
        ProcedurePostcondition,
        ProcedureObservation,
        RetryPolicy,
        ProcedureStep,
        ProcedureBranch,
        ProcedureLoop,
        ProcedureHole,
        ProcedureResourceEnvelope,
        ProcedureAuthorityEnvelope,
        ProcedureValidationPlan,
        ProcedureRollback,
        ProcedureFallback,
        ProcedureSpec,
        ProcedureCandidate,
        ProcedureCertificate,
        ProcedureInvocation,
        ProcedureTraceEntry,
        ProcedureExecutionTrace,
        ProcedureOutcome,
        ProcedureInvocationReceipt,
        ProcedureFailure,
        ProcedureRecoveryPlan,
        TrajectoryStep,
        TrajectoryOutcome,
        ExecutionTrajectory,
        TrajectoryNormalizationReceipt,
        TaskFamilyBoundary,
        TaskFamily,
        TaskFamilyMembership,
        TaskFamilyCounterexample,
        *(globals()[name] for name in _GENERIC_ARTIFACT_NAMES),
    )
}


def parse_procedure_artifact(payload: Mapping[str, Any]) -> CanonicalContract:
    """Decode one artifact through its exact closed schema."""

    if not isinstance(payload, Mapping):
        raise ProcedureContractError("artifact payload must be a mapping")
    schema = payload.get("schema")
    if type(schema) is not str or schema not in ARTIFACT_TYPES_BY_SCHEMA:
        raise ProcedureContractError("artifact has an unknown procedure compiler schema")
    return ARTIFACT_TYPES_BY_SCHEMA[schema].from_dict(payload)  # type: ignore[attr-defined]


__all__ = [
    "ALLOWED_STEP_OPERATIONS",
    "ARTIFACT_TYPES_BY_SCHEMA",
    "ArtifactBindings",
    "ArtifactState",
    "BoundedArtifact",
    "ConditionOperator",
    "EffectClass",
    "EpisodeKind",
    "ExecutionTrajectory",
    "FORBIDDEN_HOLE_TYPES",
    "FORBIDDEN_STEP_OPERATIONS",
    "FailureTransition",
    "FamilyMembershipClass",
    "HoleType",
    "IdempotencyClass",
    "ProcedureAuthorityEnvelope",
    "ProcedureBoundsError",
    "ProcedureBranch",
    "ProcedureCandidate",
    "ProcedureCertificate",
    "ProcedureContractError",
    "ProcedureEffect",
    "ProcedureExecutionTrace",
    "ProcedureFailure",
    "ProcedureFallback",
    "ProcedureHole",
    "ProcedureIdentityError",
    "ProcedureInvariant",
    "ProcedureInvocation",
    "ProcedureInvocationReceipt",
    "ProcedureLocal",
    "ProcedureLoop",
    "ProcedureObservation",
    "ProcedureOutcome",
    "ProcedureOutcomeStatus",
    "ProcedureParameter",
    "ProcedurePostcondition",
    "ProcedurePrecondition",
    "ProcedureRecoveryPlan",
    "ProcedureResourceEnvelope",
    "ProcedureRollback",
    "ProcedureSafetyError",
    "ProcedureSpec",
    "ProcedureStep",
    "ProcedureTraceEntry",
    "ProcedureValidationPlan",
    "ProcedureVersion",
    "ProviderClass",
    "RetryPolicy",
    "RiskClass",
    "StepOperation",
    "TaskFamily",
    "TaskFamilyBoundary",
    "TaskFamilyCounterexample",
    "TaskFamilyMembership",
    "TraceEventStatus",
    "TraceState",
    "TrajectoryNormalizationReceipt",
    "TrajectoryOutcome",
    "TrajectoryStep",
    "TrajectoryTerminalStatus",
    "ValueType",
    "canonical_json_bytes",
    "content_identity",
    "parse_procedure_artifact",
    *_GENERIC_ARTIFACT_NAMES,
]
