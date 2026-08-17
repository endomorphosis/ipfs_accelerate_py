"""Provider-free neutral contracts for the agent-supervisor control plane.

The records in this module are the durable boundary between target inference
and runtime construction.  They deliberately contain references to prompt
bodies, credentials, and authority proofs rather than those values themselves.
All records are closed, bounded, canonical DAG-JSON objects identified by
CIDv1/base32/dag-json/sha2-256 content addresses.

DuckDB coordination and Parquet/IPLD replication are separate contracts:
DuckDB owns mutable claims and fences for one shard, while Parquet and IPLD
carry immutable checkpoint history and never grant a lease or effect.
"""

# ruff: noqa: UP042 - this package supports Python 3.8; StrEnum requires 3.11.

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final, TypeVar

from ..core.multiformats_identity import (
    MultiformatsIdentityError,
    canonical_dag_json_bytes,
    cid_for_bytes,
    cid_for_dag_json,
    validate_cid,
)

SCHEMA_PREFIX: Final = "ipfs_accelerate_py/agent-supervisor/entrypoints"
INVOCATION_BUDGET_SCHEMA: Final = f"{SCHEMA_PREFIX}/invocation-budget@1"
INVOCATION_REQUEST_SCHEMA: Final = f"{SCHEMA_PREFIX}/invocation-request@1"
TARGET_CANDIDATE_SCHEMA: Final = f"{SCHEMA_PREFIX}/target-candidate@1"
TARGET_DECISION_SCHEMA: Final = f"{SCHEMA_PREFIX}/target-inference-decision@1"
PROVIDER_ROUTE_SCHEMA: Final = f"{SCHEMA_PREFIX}/provider-route-provenance@1"
RESOURCE_BUDGET_SCHEMA: Final = f"{SCHEMA_PREFIX}/resource-budget@1"
COORDINATION_SHARD_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/duckdb-coordination-shard@1"
)
REPLICATION_BINDING_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/parquet-ipld-replication-binding@1"
)
TARGET_RESOLUTION_SCHEMA: Final = f"{SCHEMA_PREFIX}/target-resolution-receipt@1"
RESOLVED_PROFILE_SCHEMA: Final = f"{SCHEMA_PREFIX}/resolved-profile@1"
LAUNCH_PLAN_SCHEMA: Final = f"{SCHEMA_PREFIX}/launch-plan@1"
RUN_HANDLE_SCHEMA: Final = f"{SCHEMA_PREFIX}/run-handle@1"
INVOCATION_RESULT_SCHEMA: Final = f"{SCHEMA_PREFIX}/invocation-result@1"

MAX_TEXT_BYTES: Final = 4_096
MAX_REFERENCE_BYTES: Final = 2_048
MAX_PATH_BYTES: Final = 4_096
MAX_ARG_BYTES: Final = 8_192
MAX_ARGV_ITEMS: Final = 256
MAX_DECISIONS: Final = 64
MAX_CANDIDATES: Final = 64
MAX_REASON_CODES: Final = 64
MAX_QUESTIONS: Final = 3
MAX_RECORD_BYTES: Final = 2 * 1024 * 1024
MAX_PROMPT_BYTES: Final = 1024 * 1024
MAX_LANES: Final = 256
MAX_SHARDS: Final = 1024
MAX_TIMEOUT_MS: Final = 7 * 24 * 60 * 60 * 1000

DEFAULT_PARQUET_PARTITIONS: Final[tuple[str, ...]] = (
    "repository_id",
    "run_id",
    "event_date",
    "shard_id",
)

# Every inferred/defaulted/denied field has one decision.  Explicit overrides
# still produce a decision with the explicit-override source.
REQUIRED_TARGET_DECISION_FIELDS: Final[tuple[str, ...]] = (
    "repository_root",
    "state_root",
    "repository_id",
    "checkout_id",
    "scope",
    "tree_id",
    "dirty_overlay",
    "submodules",
    "nested_repositories",
    "run_namespace",
    "objective",
    "plan",
    "task_source",
    "policy",
    "principal",
    "authority_source",
    "effect_ceiling",
    "output",
    "provider",
    "resources",
    "lane_ceiling",
    "merge_target",
    "worktree_strategy",
    "validation",
    "coordination",
    "replication",
)
AUTHORITY_DECISION_FIELDS: Final[frozenset[str]] = frozenset(
    {"policy", "principal", "authority_source", "effect_ceiling"}
)

_REFERENCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$")
_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")
_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_REASON_RE = re.compile(r"^[a-z][a-z0-9_:-]*$")
_JWT_RE = re.compile(
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?:api[_-]?key|authorization|credential|password|passwd|private[_-]?key|"
    r"secret|token|ucan)\s*[:=]\s*\S+"
)
_KNOWN_SECRET_TOKEN_RE = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{12,}|gh[pousr]_[a-z0-9]{20,}|"
    r"github_pat_[a-z0-9_]{20,}|AKIA[0-9A-Z]{16}|"
    r"xox[baprs]-[a-z0-9-]{10,})"
)
_SECRET_TEXT_MARKERS: Final[tuple[str, ...]] = (
    "-----begin private key-----",
    "-----begin rsa private key-----",
    "-----begin ec private key-----",
    "-----begin openssh private key-----",
    "bearer ",
    "basic ",
)
_FORBIDDEN_ARG_MARKERS: Final[tuple[str, ...]] = (
    "--prompt",
    "--authorization",
    "--api-key",
    "--apikey",
    "--password",
    "--private-key",
    "--secret",
    "--token",
    "--ucan",
)


class EntrypointContractError(ValueError):
    """A prompt-entrypoint contract is malformed or unsafe."""


class UnknownContractFieldError(EntrypointContractError):
    """A closed contract received an unknown field."""


class ContractBoundsError(EntrypointContractError):
    """A contract exceeds a frozen item, byte, resource, or time bound."""


class ContractIdentityError(EntrypointContractError):
    """A contract carries a malformed or mismatched content identity."""


class SecretBearingRecordError(EntrypointContractError):
    """A durable record appears to contain a secret or prompt body."""


class InvocationMode(str, Enum):
    PREVIEW = "preview"
    WORKTREE = "worktree"
    CI_WORKER = "ci_worker"
    DISTRIBUTED_WORKER = "distributed_worker"


class OutputMode(str, Enum):
    MARKDOWN = "markdown"
    DUCKDB = "duckdb"
    BOTH = "both"


class ResolutionSource(str, Enum):
    CANONICAL_REQUEST = "canonical_request"
    EXPLICIT_OVERRIDE = "explicit_override"
    EXISTING_RUN = "existing_run"
    AUTHENTICATED_TRANSPORT = "authenticated_transport"
    SIGNED_PROFILE = "signed_profile"
    REPOSITORY_HINT = "repository_hint"
    DISCOVERY = "discovery"
    BUILTIN_DEFAULT = "builtin_default"


TRUSTED_AUTHORITY_SOURCES: Final[frozenset[ResolutionSource]] = frozenset(
    {
        ResolutionSource.EXISTING_RUN,
        ResolutionSource.AUTHENTICATED_TRANSPORT,
        ResolutionSource.SIGNED_PROFILE,
    }
)


class ResolutionDisposition(str, Enum):
    UNIQUE = "unique"
    DEFAULTED = "defaulted"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"
    DENIED = "denied"


class DecisionEffect(str, Enum):
    IDENTITY_ONLY = "identity_only"
    CONFIGURATION = "configuration"
    REQUIRES_AUTHORITY = "requires_authority"


class RevalidationRule(str, Enum):
    IMMUTABLE = "immutable"
    BEFORE_PREVIEW = "before_preview"
    BEFORE_MUTATION = "before_mutation"
    ON_EXPIRY = "on_expiry"


class ProviderSelection(str, Enum):
    GROK = "grok"
    CODEX = "codex"
    UNAVAILABLE = "unavailable"


class ProviderFallbackReason(str, Enum):
    NONE = "none"
    PREFERRED_UNAVAILABLE = "preferred_provider_unavailable"
    PREFERRED_QUOTA_EXHAUSTED = "preferred_provider_quota_exhausted"
    PREFERRED_CAPACITY_UNAVAILABLE = (
        "preferred_provider_capacity_unavailable"
    )
    PREFERRED_PRE_EFFECT_FAILURE = "preferred_provider_pre_effect_failure"


class WorktreeStrategy(str, Enum):
    NONE = "none"
    ISOLATED = "isolated"
    CURRENT_CHECKOUT = "current_checkout"


class TaskSourceKind(str, Enum):
    MARKDOWN = "markdown"
    DUCKDB = "duckdb"
    DUAL = "dual"


class ReplicationMode(str, Enum):
    PARQUET_IPLD = "parquet_ipld"
    PARQUET_IPLD_IPFS = "parquet_ipld_ipfs"


class ExpectedEffect(str, Enum):
    INSPECT_REPOSITORY = "inspect_repository"
    WRITE_SUPERVISOR_STATE = "write_supervisor_state"
    CREATE_ISOLATED_WORKTREE = "create_isolated_worktree"
    EDIT_ISOLATED_WORKTREE = "edit_isolated_worktree"
    RUN_VALIDATION = "run_validation"
    LAUNCH_LOCAL_PROCESS = "launch_local_process"
    MERGE = "merge"
    PUSH = "push"
    DEPLOY = "deploy"
    DESTRUCTIVE_CLEANUP = "destructive_cleanup"


class RunState(str, Enum):
    RECEIVED = "received"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    PREVIEWING = "previewing"
    ADMITTED = "admitted"
    NEEDS_INPUT = "needs_input"
    REJECTED = "rejected"
    AUTHORIZING = "authorizing"
    MATERIALIZING = "materializing"
    STARTING = "starting"
    ADOPTING = "adopting"
    RUNNING = "running"
    DRAINED = "drained"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    QUARANTINED = "quarantined"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RunHealth(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    TERMINAL = "terminal"


class ContinuationAction(str, Enum):
    NONE = "none"
    RESOLVE = "resolve"
    PREVIEW = "preview"
    AUTHORIZE = "authorize"
    MATERIALIZE = "materialize"
    START = "start"
    ADOPT = "adopt"
    MONITOR = "monitor"
    ASK_INPUT = "ask_input"
    RESUME = "resume"
    RETRY = "retry"


class InvocationStatus(str, Enum):
    PREVIEW = "preview"
    STARTED = "started"
    ADOPTED = "adopted"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    COMPLETED = "completed"


E = TypeVar("E", bound=Enum)
C = TypeVar("C", bound="_CanonicalContract")


def _enum(value: Any, enum_type: type[E], name: str) -> E:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise EntrypointContractError(
            f"{name} must be one of {[item.value for item in enum_type]}"
        ) from exc


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EntrypointContractError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        ceiling = f" and at most {maximum}" if maximum is not None else ""
        raise ContractBoundsError(
            f"{name} must be at least {minimum}{ceiling}"
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise EntrypointContractError(f"{name} must be a boolean")
    return value


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
    reject_secrets: bool = True,
) -> str:
    if not isinstance(value, str):
        raise EntrypointContractError(f"{name} must be text")
    if "\x00" in value:
        raise EntrypointContractError(f"{name} contains a NUL byte")
    if required and not value:
        raise EntrypointContractError(f"{name} must not be empty")
    if len(value.encode("utf-8")) > maximum:
        raise ContractBoundsError(f"{name} exceeds {maximum} UTF-8 bytes")
    if reject_secrets and value:
        lowered = value.casefold()
        if (
            any(marker in lowered for marker in _SECRET_TEXT_MARKERS)
            or _JWT_RE.search(value)
            or _SECRET_ASSIGNMENT_RE.search(value)
            or _KNOWN_SECRET_TOKEN_RE.search(value)
        ):
            raise SecretBearingRecordError(
                f"{name} contains secret-bearing material; store a reference"
            )
    return value


def _token(value: Any, name: str, *, required: bool = True) -> str:
    result = _text(value, name, required=required, maximum=256)
    if result and not _TOKEN_RE.fullmatch(result):
        raise EntrypointContractError(f"{name} is not a canonical token")
    return result


def _field_name(value: Any, name: str = "field_name") -> str:
    result = _text(value, name, maximum=128)
    if not _FIELD_RE.fullmatch(result):
        raise EntrypointContractError(f"{name} is not a canonical field name")
    return result


def _reason(value: Any, name: str) -> str:
    result = _text(value, name, maximum=256)
    if not _REASON_RE.fullmatch(result):
        raise EntrypointContractError(f"{name} is not a typed reason code")
    return result


def _reference(value: Any, name: str, *, required: bool = True) -> str:
    result = _text(
        value,
        name,
        required=required,
        maximum=MAX_REFERENCE_BYTES,
    )
    if result and (
        not _REFERENCE_RE.fullmatch(result)
        or "=" in result
        or _JWT_RE.search(result)
    ):
        raise SecretBearingRecordError(
            f"{name} must be an opaque handle, DID, or content reference"
        )
    return result


def _cid(
    value: Any,
    name: str,
    *,
    required: bool = True,
    codecs: tuple[str, ...] = ("dag-json",),
) -> str:
    if not value and not required:
        return ""
    result = _text(value, name, maximum=256)
    try:
        return validate_cid(result, codecs=codecs)
    except (MultiformatsIdentityError, TypeError, ValueError) as exc:
        raise ContractIdentityError(f"{name} must be a canonical CIDv1") from exc


def _prompt_cid(value: Any, name: str = "prompt_cid") -> str:
    return _cid(value, name, codecs=("raw",))


def _absolute_path(value: Any, name: str, *, required: bool = True) -> str:
    result = _text(
        value,
        name,
        required=required,
        maximum=MAX_PATH_BYTES,
    )
    if not result:
        return ""
    if not result.startswith("/") or "\\" in result:
        raise EntrypointContractError(f"{name} must be an absolute POSIX path")
    normalized = posixpath.normpath(result)
    if normalized != result or any(part == ".." for part in result.split("/")):
        raise EntrypointContractError(f"{name} must be lexically normalized")
    return result


def _relative_path(value: Any, name: str, *, required: bool = True) -> str:
    result = _text(
        value,
        name,
        required=required,
        maximum=MAX_PATH_BYTES,
    )
    if not result:
        return ""
    if result.startswith("/") or "\\" in result:
        raise EntrypointContractError(f"{name} must be a relative POSIX path")
    normalized = posixpath.normpath(result)
    if (
        normalized != result
        or result == ".."
        or result.startswith("../")
        or "/../" in result
    ):
        raise EntrypointContractError(f"{name} escapes its selected root")
    return result


def _is_contained_path(path: str, root: str) -> bool:
    if not path or not root:
        return False
    try:
        return posixpath.commonpath((path, root)) == root
    except ValueError:
        return False


def _require_contained_path(path: str, root: str, name: str) -> None:
    if path and not _is_contained_path(path, root):
        raise EntrypointContractError(f"{name} must be contained by its selected root")


def _decision_value(value: Any, name: str = "value") -> str:
    """Accept a path or opaque identifier, never durable free-form prose."""

    result = _text(value, name)
    if result.startswith("/"):
        return _absolute_path(result, name)
    return _reference(result, name)


def _reject_embedded_prompt(
    value: str,
    *,
    prompt_body: bytes,
    name: str,
) -> None:
    try:
        prompt_text = prompt_body.decode("utf-8")
    except UnicodeDecodeError:
        return
    if prompt_text and prompt_text in value:
        raise SecretBearingRecordError(
            f"{name} must reference prompt content, never persist its body"
        )


def _text_tuple(
    value: Any,
    name: str,
    *,
    maximum_items: int = MAX_REASON_CODES,
    item_kind: str = "text",
    unique: bool = True,
    sorted_items: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EntrypointContractError(f"{name} must be a sequence")
    if len(value) > maximum_items:
        raise ContractBoundsError(f"{name} exceeds {maximum_items} items")
    if item_kind == "reason":
        items = tuple(_reason(item, f"{name}[]") for item in value)
    elif item_kind == "reference":
        items = tuple(_reference(item, f"{name}[]") for item in value)
    elif item_kind == "cid":
        items = tuple(_cid(item, f"{name}[]") for item in value)
    elif item_kind == "path":
        items = tuple(_absolute_path(item, f"{name}[]") for item in value)
    else:
        items = tuple(_text(item, f"{name}[]") for item in value)
    if unique and len(items) != len(set(items)):
        raise EntrypointContractError(f"{name} contains duplicates")
    return tuple(sorted(items)) if sorted_items else items


def _argv(value: Any, name: str) -> tuple[str, ...]:
    items = _text_tuple(
        value,
        name,
        maximum_items=MAX_ARGV_ITEMS,
        unique=False,
    )
    for item in items:
        if len(item.encode("utf-8")) > MAX_ARG_BYTES:
            raise ContractBoundsError(f"{name} argument exceeds {MAX_ARG_BYTES} bytes")
        lowered = item.casefold()
        if any(
            lowered == marker or lowered.startswith(marker + "=")
            for marker in _FORBIDDEN_ARG_MARKERS
        ):
            raise SecretBearingRecordError(
                f"{name} must not carry prompt, credential, UCAN, or authority values"
            )
    return items


def _environment_names(value: Any) -> tuple[str, ...]:
    names = _text_tuple(
        value,
        "environment_names",
        maximum_items=128,
        sorted_items=True,
    )
    if any(not _ENV_NAME_RE.fullmatch(name) for name in names):
        raise EntrypointContractError(
            "environment_names may contain names only, never NAME=value"
        )
    return names


def _enum_tuple(
    value: Any,
    enum_type: type[E],
    name: str,
    *,
    maximum_items: int = 64,
    sorted_items: bool = False,
) -> tuple[E, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EntrypointContractError(f"{name} must be a sequence")
    if len(value) > maximum_items:
        raise ContractBoundsError(f"{name} exceeds {maximum_items} items")
    items = tuple(_enum(item, enum_type, f"{name}[]") for item in value)
    if len(items) != len(set(items)):
        raise EntrypointContractError(f"{name} contains duplicates")
    if sorted_items:
        return tuple(sorted(items, key=lambda item: str(item.value)))
    return items


def _closed(
    value: Mapping[str, Any],
    *,
    schema: str,
    fields: Sequence[str],
) -> None:
    if not isinstance(value, Mapping):
        raise EntrypointContractError("contract record must be an object")
    if any(not isinstance(key, str) for key in value):
        raise UnknownContractFieldError(
            f"{schema} contains a non-text field name"
        )
    allowed = {"schema", "content_id", *fields}
    unknown = set(value).difference(allowed)
    if unknown:
        raise UnknownContractFieldError(
            f"{schema} contains unknown fields: {sorted(unknown)}"
        )
    missing = set(fields).difference(value)
    if missing:
        raise EntrypointContractError(
            f"{schema} is missing fields: {sorted(missing)}"
        )
    if value.get("schema") != schema:
        raise EntrypointContractError(f"record must use schema {schema!r}")


def _json_load_canonical(payload: str, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, str):
        raise EntrypointContractError(f"{name} JSON must be text")
    if len(payload.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ContractBoundsError(f"{name} JSON exceeds the record byte bound")

    def reject_constant(constant: str) -> None:
        raise EntrypointContractError(
            f"{name} JSON contains forbidden constant {constant}"
        )

    try:
        value = json.loads(payload, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise EntrypointContractError(f"{name} JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise EntrypointContractError(f"{name} JSON must contain an object")
    return value


class _CanonicalContract:
    SCHEMA: ClassVar[str]

    def _payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def _identity_payload(self) -> dict[str, Any]:
        return self._payload()

    @property
    def content_id(self) -> str:
        identity = self._identity_payload()
        content_id = cid_for_dag_json(identity)
        record = self._payload()
        record["content_id"] = content_id
        if len(canonical_dag_json_bytes(record)) > MAX_RECORD_BYTES:
            raise ContractBoundsError("canonical contract exceeds record byte bound")
        return content_id

    def to_dict(self) -> dict[str, Any]:
        record = self._payload()
        record["content_id"] = self.content_id
        if len(canonical_dag_json_bytes(record)) > MAX_RECORD_BYTES:
            raise ContractBoundsError("canonical contract exceeds record byte bound")
        return record

    to_record = to_dict

    def canonical_bytes(self) -> bytes:
        return canonical_dag_json_bytes(self.to_dict())

    def to_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    @classmethod
    def from_json(cls: type[C], payload: str) -> C:
        record = _json_load_canonical(payload, cls.__name__)
        result = cls.from_dict(record)  # type: ignore[attr-defined]
        if result.to_json() != payload:
            raise EntrypointContractError(
                f"{cls.__name__} JSON is not the exact canonical encoding"
            )
        return result

    @classmethod
    def _verify_claimed(cls, value: Mapping[str, Any], result: C) -> C:
        claimed = value.get("content_id")
        actual = result.content_id
        if claimed is not None and claimed != actual:
            raise ContractIdentityError(
                f"{cls.__name__} content_id does not match its canonical payload"
            )
        return result


@dataclass(frozen=True)
class InvocationBudget(_CanonicalContract):
    """Finite work and response bounds supplied with one invocation."""

    SCHEMA: ClassVar[str] = INVOCATION_BUDGET_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "max_prompt_bytes",
        "max_actions",
        "max_lanes",
        "timeout_ms",
        "max_result_bytes",
    )

    max_prompt_bytes: int = 256 * 1024
    max_actions: int = 64
    max_lanes: int = 4
    timeout_ms: int = 3_600_000
    max_result_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_prompt_bytes",
            _integer(
                self.max_prompt_bytes,
                "max_prompt_bytes",
                minimum=1,
                maximum=MAX_PROMPT_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "max_actions",
            _integer(self.max_actions, "max_actions", minimum=1, maximum=4096),
        )
        object.__setattr__(
            self,
            "max_lanes",
            _integer(self.max_lanes, "max_lanes", minimum=1, maximum=MAX_LANES),
        )
        object.__setattr__(
            self,
            "timeout_ms",
            _integer(
                self.timeout_ms,
                "timeout_ms",
                minimum=1,
                maximum=MAX_TIMEOUT_MS,
            ),
        )
        object.__setattr__(
            self,
            "max_result_bytes",
            _integer(
                self.max_result_bytes,
                "max_result_bytes",
                minimum=1024,
                maximum=MAX_RECORD_BYTES,
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "max_prompt_bytes": self.max_prompt_bytes,
            "max_actions": self.max_actions,
            "max_lanes": self.max_lanes,
            "timeout_ms": self.timeout_ms,
            "max_result_bytes": self.max_result_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InvocationBudget:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(**{name: value[name] for name in cls.FIELDS})
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class SupervisorInvocationRequest(_CanonicalContract):
    """A body-free durable invocation descriptor.

    ``transient_prompt_body`` may be supplied in process memory.  It is checked
    against ``prompt_cid`` and never appears in equality, repr, canonical bytes,
    JSON, or the invocation CID.  ``prompt_ref`` is the capability-protected
    broker/artifact handle used to retrieve that body when needed.
    """

    SCHEMA: ClassVar[str] = INVOCATION_REQUEST_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "prompt_cid",
        "prompt_ref",
        "mode",
        "budget",
        "repository_hint",
        "scope_hint",
        "run_hint",
        "objective_hint",
        "task_source_hint",
        "profile_hint",
        "provider_hint",
        "output_mode_hint",
        "lane_ceiling_hint",
        "expected_target_cid",
        "resolution_receipt_hint",
        "canonical_request_cid",
    )

    prompt_cid: str
    prompt_ref: str
    mode: InvocationMode = InvocationMode.WORKTREE
    budget: InvocationBudget = field(default_factory=InvocationBudget)
    repository_hint: str = ""
    scope_hint: str = ""
    run_hint: str = ""
    objective_hint: str = ""
    task_source_hint: str = ""
    profile_hint: str = ""
    provider_hint: str = ""
    output_mode_hint: str = ""
    lane_ceiling_hint: int = 0
    expected_target_cid: str = ""
    resolution_receipt_hint: str = ""
    canonical_request_cid: str = ""
    transient_prompt_body: bytes | None = field(
        default=None,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "prompt_cid", _prompt_cid(self.prompt_cid, "prompt_cid")
        )
        object.__setattr__(
            self, "prompt_ref", _reference(self.prompt_ref, "prompt_ref")
        )
        object.__setattr__(self, "mode", _enum(self.mode, InvocationMode, "mode"))
        if not isinstance(self.budget, InvocationBudget):
            if isinstance(self.budget, Mapping):
                object.__setattr__(
                    self, "budget", InvocationBudget.from_dict(self.budget)
                )
            else:
                raise EntrypointContractError(
                    "budget must be an InvocationBudget"
                )
        for name in (
            "prompt_ref",
            "repository_hint",
            "scope_hint",
            "run_hint",
            "objective_hint",
            "task_source_hint",
            "profile_hint",
            "provider_hint",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=False),
            )
        output = self.output_mode_hint
        if output:
            output = _enum(output, OutputMode, "output_mode_hint").value
        object.__setattr__(self, "output_mode_hint", output)
        object.__setattr__(
            self,
            "lane_ceiling_hint",
            _integer(
                self.lane_ceiling_hint,
                "lane_ceiling_hint",
                maximum=MAX_LANES,
            ),
        )
        for name in (
            "expected_target_cid",
            "resolution_receipt_hint",
            "canonical_request_cid",
        ):
            object.__setattr__(
                self,
                name,
                _cid(getattr(self, name), name, required=False),
            )
        for name in (
            "repository_hint",
            "scope_hint",
            "run_hint",
            "objective_hint",
            "task_source_hint",
            "profile_hint",
            "provider_hint",
            "output_mode_hint",
        ):
            durable_text = getattr(self, name)
            if durable_text and cid_for_bytes(
                durable_text.encode("utf-8"), codec="raw"
            ) == self.prompt_cid:
                raise SecretBearingRecordError(
                    f"{name} must reference prompt content, never persist its body"
                )
        body = self.transient_prompt_body
        if body is not None:
            if type(body) is not bytes:
                raise EntrypointContractError(
                    "transient_prompt_body must be exact bytes"
                )
            if not body:
                raise EntrypointContractError(
                    "transient_prompt_body must not be empty"
                )
            if len(body) > self.budget.max_prompt_bytes:
                raise ContractBoundsError(
                    "transient_prompt_body exceeds invocation budget"
                )
            if cid_for_bytes(body, codec="raw") != self.prompt_cid:
                raise ContractIdentityError(
                    "transient_prompt_body does not match prompt_cid"
                )
            for name in (
                "prompt_ref",
                "repository_hint",
                "scope_hint",
                "run_hint",
                "objective_hint",
                "task_source_hint",
                "profile_hint",
                "provider_hint",
                "output_mode_hint",
            ):
                _reject_embedded_prompt(
                    getattr(self, name),
                    prompt_body=body,
                    name=name,
                )

    @classmethod
    def from_prompt(
        cls,
        prompt: str | bytes,
        *,
        prompt_ref: str,
        **values: Any,
    ) -> SupervisorInvocationRequest:
        if isinstance(prompt, str):
            body = prompt.encode("utf-8")
        elif type(prompt) is bytes:
            body = prompt
        else:
            raise EntrypointContractError("prompt must be text or exact bytes")
        return cls(
            prompt_cid=cid_for_bytes(body, codec="raw"),
            prompt_ref=prompt_ref,
            transient_prompt_body=body,
            **values,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "prompt_cid": self.prompt_cid,
            "prompt_ref": self.prompt_ref,
            "mode": self.mode.value,
            "budget": self.budget.to_dict(),
            "repository_hint": self.repository_hint,
            "scope_hint": self.scope_hint,
            "run_hint": self.run_hint,
            "objective_hint": self.objective_hint,
            "task_source_hint": self.task_source_hint,
            "profile_hint": self.profile_hint,
            "provider_hint": self.provider_hint,
            "output_mode_hint": self.output_mode_hint,
            "lane_ceiling_hint": self.lane_ceiling_hint,
            "expected_target_cid": self.expected_target_cid,
            "resolution_receipt_hint": self.resolution_receipt_hint,
            "canonical_request_cid": self.canonical_request_cid,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> SupervisorInvocationRequest:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(
            prompt_cid=value["prompt_cid"],
            prompt_ref=value["prompt_ref"],
            mode=value["mode"],
            budget=InvocationBudget.from_dict(value["budget"]),
            repository_hint=value["repository_hint"],
            scope_hint=value["scope_hint"],
            run_hint=value["run_hint"],
            objective_hint=value["objective_hint"],
            task_source_hint=value["task_source_hint"],
            profile_hint=value["profile_hint"],
            provider_hint=value["provider_hint"],
            output_mode_hint=value["output_mode_hint"],
            lane_ceiling_hint=value["lane_ceiling_hint"],
            expected_target_cid=value["expected_target_cid"],
            resolution_receipt_hint=value["resolution_receipt_hint"],
            canonical_request_cid=value["canonical_request_cid"],
        )
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class TargetCandidate(_CanonicalContract):
    """One bounded candidate considered for an inferred field."""

    SCHEMA: ClassVar[str] = TARGET_CANDIDATE_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "field_name",
        "value",
        "source",
        "source_precedence",
        "evidence_cid",
        "confidence_ppm",
        "rejection_reason",
    )

    field_name: str
    value: str
    source: ResolutionSource
    source_precedence: int
    evidence_cid: str
    confidence_ppm: int = 1_000_000
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_name", _field_name(self.field_name))
        object.__setattr__(self, "value", _decision_value(self.value, "value"))
        object.__setattr__(
            self, "source", _enum(self.source, ResolutionSource, "source")
        )
        object.__setattr__(
            self,
            "source_precedence",
            _integer(
                self.source_precedence,
                "source_precedence",
                maximum=10_000,
            ),
        )
        object.__setattr__(
            self, "evidence_cid", _cid(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(
            self,
            "confidence_ppm",
            _integer(
                self.confidence_ppm,
                "confidence_ppm",
                maximum=1_000_000,
            ),
        )
        reason = self.rejection_reason
        if reason:
            reason = _reason(reason, "rejection_reason")
        object.__setattr__(self, "rejection_reason", reason)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "field_name": self.field_name,
            "value": self.value,
            "source": self.source.value,
            "source_precedence": self.source_precedence,
            "evidence_cid": self.evidence_cid,
            "confidence_ppm": self.confidence_ppm,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TargetCandidate:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(**{name: value[name] for name in cls.FIELDS})
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class TargetInferenceDecision(_CanonicalContract):
    """A complete, replayable inference decision for one target field."""

    SCHEMA: ClassVar[str] = TARGET_DECISION_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "field_name",
        "disposition",
        "selected_value",
        "selected_source",
        "source_precedence",
        "evidence_cid",
        "candidates",
        "reason_codes",
        "effect",
        "override_accepted",
        "fresh_until_ms",
        "revalidation_rule",
    )

    field_name: str
    disposition: ResolutionDisposition
    selected_value: str
    selected_source: ResolutionSource
    source_precedence: int
    evidence_cid: str
    candidates: tuple[TargetCandidate, ...]
    reason_codes: tuple[str, ...]
    effect: DecisionEffect
    override_accepted: bool
    fresh_until_ms: int
    revalidation_rule: RevalidationRule

    def __post_init__(self) -> None:
        field_name = _field_name(self.field_name)
        object.__setattr__(self, "field_name", field_name)
        disposition = _enum(
            self.disposition, ResolutionDisposition, "disposition"
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "selected_value",
            _text(self.selected_value, "selected_value", required=False),
        )
        object.__setattr__(
            self,
            "selected_source",
            _enum(self.selected_source, ResolutionSource, "selected_source"),
        )
        object.__setattr__(
            self,
            "source_precedence",
            _integer(
                self.source_precedence,
                "source_precedence",
                maximum=10_000,
            ),
        )
        object.__setattr__(
            self, "evidence_cid", _cid(self.evidence_cid, "evidence_cid")
        )
        if isinstance(self.candidates, (str, bytes)) or not isinstance(
            self.candidates, Sequence
        ):
            raise EntrypointContractError("candidates must be a sequence")
        if not 0 <= len(self.candidates) <= MAX_CANDIDATES:
            raise ContractBoundsError(
                f"candidates exceeds {MAX_CANDIDATES} items"
            )
        candidates = tuple(
            item
            if isinstance(item, TargetCandidate)
            else TargetCandidate.from_dict(item)
            for item in self.candidates
        )
        if any(item.field_name != field_name for item in candidates):
            raise EntrypointContractError(
                "all candidates must belong to the decision field"
            )
        if len({item.content_id for item in candidates}) != len(candidates):
            raise EntrypointContractError("candidates contain duplicates")
        candidates = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.source_precedence,
                    item.source.value,
                    item.value,
                    item.evidence_cid,
                    item.confidence_ppm,
                    item.rejection_reason,
                ),
            )
        )
        object.__setattr__(self, "candidates", candidates)
        reasons = _text_tuple(
            self.reason_codes,
            "reason_codes",
            item_kind="reason",
            sorted_items=True,
        )
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self, "effect", _enum(self.effect, DecisionEffect, "effect")
        )
        object.__setattr__(
            self,
            "override_accepted",
            _boolean(self.override_accepted, "override_accepted"),
        )
        object.__setattr__(
            self,
            "fresh_until_ms",
            _integer(self.fresh_until_ms, "fresh_until_ms"),
        )
        rule = _enum(
            self.revalidation_rule, RevalidationRule, "revalidation_rule"
        )
        object.__setattr__(self, "revalidation_rule", rule)

        selected = disposition in {
            ResolutionDisposition.UNIQUE,
            ResolutionDisposition.DEFAULTED,
        }
        if selected:
            if not self.selected_value or not candidates:
                raise EntrypointContractError(
                    "selected decisions require a value and candidate"
                )
            matches = [
                item
                for item in candidates
                if item.value == self.selected_value
                and item.source is self.selected_source
            ]
            if len(matches) != 1:
                raise EntrypointContractError(
                    "selected value/source must identify exactly one candidate"
                )
            if matches[0].rejection_reason:
                raise EntrypointContractError(
                    "the selected candidate cannot carry a rejection reason"
                )
            if any(
                not item.rejection_reason for item in candidates if item not in matches
            ):
                raise EntrypointContractError(
                    "non-selected alternatives require rejection reasons"
                )
        else:
            if self.selected_value:
                raise EntrypointContractError(
                    "unresolved decisions cannot select a value"
                )
            if not reasons:
                raise EntrypointContractError(
                    "unresolved decisions require typed reason codes"
                )
            if disposition is ResolutionDisposition.AMBIGUOUS and len(candidates) < 2:
                raise EntrypointContractError(
                    "ambiguous decisions require at least two candidates"
                )
        if self.override_accepted and self.selected_source is not ResolutionSource.EXPLICIT_OVERRIDE:
            raise EntrypointContractError(
                "override_accepted requires the explicit_override source"
            )
        authority_field = field_name in AUTHORITY_DECISION_FIELDS
        if authority_field and self.effect is not DecisionEffect.REQUIRES_AUTHORITY:
            raise EntrypointContractError(
                f"{field_name} must be marked requires_authority"
            )
        if selected and authority_field and self.selected_source not in TRUSTED_AUTHORITY_SOURCES:
            raise EntrypointContractError(
                f"{field_name} requires authenticated transport, signed profile, "
                "or existing-run authority evidence"
            )
        if authority_field and self.override_accepted:
            raise EntrypointContractError(
                f"{field_name} cannot accept an untrusted explicit override"
            )
        if rule is RevalidationRule.IMMUTABLE and self.fresh_until_ms != 0:
            raise EntrypointContractError(
                "immutable decisions must use fresh_until_ms=0"
            )

    @property
    def unresolved(self) -> bool:
        return self.disposition in {
            ResolutionDisposition.AMBIGUOUS,
            ResolutionDisposition.UNAVAILABLE,
            ResolutionDisposition.DENIED,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "field_name": self.field_name,
            "disposition": self.disposition.value,
            "selected_value": self.selected_value,
            "selected_source": self.selected_source.value,
            "source_precedence": self.source_precedence,
            "evidence_cid": self.evidence_cid,
            "candidates": [item.to_dict() for item in self.candidates],
            "reason_codes": list(self.reason_codes),
            "effect": self.effect.value,
            "override_accepted": self.override_accepted,
            "fresh_until_ms": self.fresh_until_ms,
            "revalidation_rule": self.revalidation_rule.value,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> TargetInferenceDecision:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(
            field_name=value["field_name"],
            disposition=value["disposition"],
            selected_value=value["selected_value"],
            selected_source=value["selected_source"],
            source_precedence=value["source_precedence"],
            evidence_cid=value["evidence_cid"],
            candidates=value["candidates"],
            reason_codes=value["reason_codes"],
            effect=value["effect"],
            override_accepted=value["override_accepted"],
            fresh_until_ms=value["fresh_until_ms"],
            revalidation_rule=value["revalidation_rule"],
        )
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class ProviderRouteProvenance(_CanonicalContract):
    """Typed Grok-preferred, Codex-fallback provider route evidence."""

    SCHEMA: ClassVar[str] = PROVIDER_ROUTE_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "preferred_provider",
        "fallback_provider",
        "selected_provider",
        "fallback_reason",
        "fallback_receipt_cid",
        "observed_capability_cid",
        "usage_evidence_cid",
        "budget_cid",
        "task_revision_cid",
        "attempt_cid",
        "worktree_cid",
        "authenticated_profile_override_cid",
        "maximum_fallback_dispatches",
        "independent_review_required",
    )

    preferred_provider: str
    fallback_provider: str
    selected_provider: ProviderSelection
    fallback_reason: ProviderFallbackReason
    fallback_receipt_cid: str
    observed_capability_cid: str
    usage_evidence_cid: str
    budget_cid: str
    task_revision_cid: str
    attempt_cid: str
    worktree_cid: str
    authenticated_profile_override_cid: str = ""
    maximum_fallback_dispatches: int = 1
    independent_review_required: bool = True

    def __post_init__(self) -> None:
        preferred = _token(self.preferred_provider, "preferred_provider")
        fallback = _token(self.fallback_provider, "fallback_provider")
        if preferred != "grok" or fallback != "codex":
            raise EntrypointContractError(
                "the built-in provider route must be Grok then Codex"
            )
        object.__setattr__(self, "preferred_provider", preferred)
        object.__setattr__(self, "fallback_provider", fallback)
        selected = _enum(
            self.selected_provider, ProviderSelection, "selected_provider"
        )
        reason = _enum(
            self.fallback_reason, ProviderFallbackReason, "fallback_reason"
        )
        object.__setattr__(self, "selected_provider", selected)
        object.__setattr__(self, "fallback_reason", reason)
        for name in (
            "fallback_receipt_cid",
            "attempt_cid",
            "worktree_cid",
            "authenticated_profile_override_cid",
        ):
            object.__setattr__(
                self,
                name,
                _cid(getattr(self, name), name, required=False),
            )
        for name in (
            "observed_capability_cid",
            "usage_evidence_cid",
            "budget_cid",
            "task_revision_cid",
        ):
            object.__setattr__(self, name, _cid(getattr(self, name), name))
        object.__setattr__(
            self,
            "maximum_fallback_dispatches",
            _integer(
                self.maximum_fallback_dispatches,
                "maximum_fallback_dispatches",
                minimum=1,
                maximum=1,
            ),
        )
        object.__setattr__(
            self,
            "independent_review_required",
            _boolean(
                self.independent_review_required,
                "independent_review_required",
            ),
        )
        override = bool(self.authenticated_profile_override_cid)
        if selected is ProviderSelection.GROK:
            if reason is not ProviderFallbackReason.NONE or self.fallback_receipt_cid:
                raise EntrypointContractError(
                    "Grok selection cannot claim a fallback"
                )
        elif selected is ProviderSelection.CODEX:
            if override:
                if reason is not ProviderFallbackReason.NONE:
                    raise EntrypointContractError(
                        "profile override is not a fallback failure"
                    )
            elif reason is not ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED:
                raise EntrypointContractError(
                    "Codex fallback requires confirmed Grok quota exhaustion"
                )
            elif not self.fallback_receipt_cid:
                raise EntrypointContractError(
                    "Codex fallback requires a typed committed fallback receipt"
                )
            if not self.independent_review_required:
                raise EntrypointContractError(
                    "Codex implementation requires an independent reviewer"
                )
        else:
            if reason is ProviderFallbackReason.NONE:
                raise EntrypointContractError(
                    "an unavailable route requires a typed provider failure reason"
                )
            if self.fallback_receipt_cid:
                raise EntrypointContractError(
                    "an unavailable route cannot claim a Codex fallback receipt"
                )
            if self.attempt_cid or self.worktree_cid:
                raise EntrypointContractError(
                    "an unavailable route cannot claim an implementation attempt"
                )
        if selected is not ProviderSelection.UNAVAILABLE and (
            not self.attempt_cid or not self.worktree_cid
        ):
            raise EntrypointContractError(
                "a selected provider requires attempt and worktree identities"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "preferred_provider": self.preferred_provider,
            "fallback_provider": self.fallback_provider,
            "selected_provider": self.selected_provider.value,
            "fallback_reason": self.fallback_reason.value,
            "fallback_receipt_cid": self.fallback_receipt_cid,
            "observed_capability_cid": self.observed_capability_cid,
            "usage_evidence_cid": self.usage_evidence_cid,
            "budget_cid": self.budget_cid,
            "task_revision_cid": self.task_revision_cid,
            "attempt_cid": self.attempt_cid,
            "worktree_cid": self.worktree_cid,
            "authenticated_profile_override_cid": (
                self.authenticated_profile_override_cid
            ),
            "maximum_fallback_dispatches": self.maximum_fallback_dispatches,
            "independent_review_required": self.independent_review_required,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> ProviderRouteProvenance:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(**{name: value[name] for name in cls.FIELDS})
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class ResourceBudget(_CanonicalContract):
    """Resolved host/provider resource ceilings."""

    SCHEMA: ClassVar[str] = RESOURCE_BUDGET_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "max_lanes",
        "max_processes",
        "max_validation_workers",
        "cpu_millis",
        "memory_bytes",
        "provider_request_limit",
        "deadline_ms",
    )

    max_lanes: int
    max_processes: int
    max_validation_workers: int
    cpu_millis: int
    memory_bytes: int
    provider_request_limit: int
    deadline_ms: int

    def __post_init__(self) -> None:
        limits = {
            "max_lanes": (1, MAX_LANES),
            "max_processes": (1, 4096),
            "max_validation_workers": (1, 4096),
            "cpu_millis": (1, 10**9),
            "memory_bytes": (1024 * 1024, 2**60),
            "provider_request_limit": (1, 10**9),
            "deadline_ms": (1, MAX_TIMEOUT_MS),
        }
        for name, (minimum, maximum) in limits.items():
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name,
                    minimum=minimum,
                    maximum=maximum,
                ),
            )
        if self.max_lanes > self.max_processes:
            raise EntrypointContractError(
                "max_lanes cannot exceed max_processes"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            **{name: getattr(self, name) for name in self.FIELDS},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResourceBudget:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(**{name: value[name] for name in cls.FIELDS})
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class CoordinationShardBinding(_CanonicalContract):
    """One mutable DuckDB coordination shard and its elected writer."""

    SCHEMA: ClassVar[str] = COORDINATION_SHARD_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "backend",
        "database_path",
        "shard_id",
        "shard_count",
        "shard_index",
        "owner_principal_ref",
        "coordinator_cid",
        "lease_namespace",
        "fencing_generation",
        "writable",
        "write_model",
        "remote_access",
    )

    backend: str
    database_path: str
    shard_id: str
    shard_count: int
    shard_index: int
    owner_principal_ref: str
    coordinator_cid: str
    lease_namespace: str
    fencing_generation: int
    writable: bool
    write_model: str = "single_writer_transactional_cas"
    remote_access: str = "owner_rpc"

    def __post_init__(self) -> None:
        if self.backend != "duckdb":
            raise EntrypointContractError(
                "mutable coordination backend must be duckdb"
            )
        object.__setattr__(
            self,
            "database_path",
            _absolute_path(self.database_path, "database_path"),
        )
        object.__setattr__(self, "shard_id", _token(self.shard_id, "shard_id"))
        count = _integer(
            self.shard_count,
            "shard_count",
            minimum=1,
            maximum=MAX_SHARDS,
        )
        index = _integer(
            self.shard_index,
            "shard_index",
            maximum=MAX_SHARDS - 1,
        )
        if index >= count:
            raise EntrypointContractError("shard_index must be below shard_count")
        object.__setattr__(self, "shard_count", count)
        object.__setattr__(self, "shard_index", index)
        object.__setattr__(
            self,
            "owner_principal_ref",
            _reference(self.owner_principal_ref, "owner_principal_ref"),
        )
        object.__setattr__(
            self, "coordinator_cid", _cid(self.coordinator_cid, "coordinator_cid")
        )
        object.__setattr__(
            self,
            "lease_namespace",
            _token(self.lease_namespace, "lease_namespace"),
        )
        object.__setattr__(
            self,
            "fencing_generation",
            _integer(self.fencing_generation, "fencing_generation"),
        )
        object.__setattr__(self, "writable", _boolean(self.writable, "writable"))
        if self.write_model != "single_writer_transactional_cas":
            raise EntrypointContractError(
                "DuckDB shard write_model must be single-writer transactional CAS"
            )
        if self.remote_access != "owner_rpc":
            raise EntrypointContractError(
                "remote workers must call the shard owner, not share the DB file"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            **{name: getattr(self, name) for name in self.FIELDS},
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> CoordinationShardBinding:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(**{name: value[name] for name in cls.FIELDS})
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class ReplicationBinding(_CanonicalContract):
    """Immutable Parquet/IPLD checkpoint and optional IPFS publication policy."""

    SCHEMA: ClassVar[str] = REPLICATION_BINDING_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "mode",
        "parquet_dataset_path",
        "parquet_schema_cid",
        "partition_keys",
        "ipld_manifest_schema_cid",
        "ipld_codec",
        "cid_profile",
        "links_must_be_verified",
        "car_export",
        "ipfs_publish",
        "ipfs_backend_handle",
        "pin",
        "max_events_per_epoch",
    )

    mode: ReplicationMode
    parquet_dataset_path: str
    parquet_schema_cid: str
    partition_keys: tuple[str, ...]
    ipld_manifest_schema_cid: str
    ipld_codec: str = "dag-json"
    cid_profile: str = "cidv1-base32-sha2-256"
    links_must_be_verified: bool = True
    car_export: bool = True
    ipfs_publish: bool = False
    ipfs_backend_handle: str = ""
    pin: bool = False
    max_events_per_epoch: int = 10_000

    def __post_init__(self) -> None:
        mode = _enum(self.mode, ReplicationMode, "mode")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "parquet_dataset_path",
            _absolute_path(self.parquet_dataset_path, "parquet_dataset_path"),
        )
        object.__setattr__(
            self,
            "parquet_schema_cid",
            _cid(self.parquet_schema_cid, "parquet_schema_cid"),
        )
        partitions = _text_tuple(
            self.partition_keys,
            "partition_keys",
            maximum_items=16,
        )
        if partitions != DEFAULT_PARQUET_PARTITIONS:
            raise EntrypointContractError(
                "Parquet partitions must bind repository, run, date, and shard"
            )
        object.__setattr__(self, "partition_keys", partitions)
        object.__setattr__(
            self,
            "ipld_manifest_schema_cid",
            _cid(
                self.ipld_manifest_schema_cid,
                "ipld_manifest_schema_cid",
            ),
        )
        if self.ipld_codec != "dag-json":
            raise EntrypointContractError("IPLD manifest codec must be dag-json")
        if self.cid_profile != "cidv1-base32-sha2-256":
            raise EntrypointContractError(
                "IPLD links must use canonical CIDv1/base32/sha2-256"
            )
        for name in (
            "links_must_be_verified",
            "car_export",
            "ipfs_publish",
            "pin",
        ):
            object.__setattr__(
                self, name, _boolean(getattr(self, name), name)
            )
        if not self.links_must_be_verified:
            raise EntrypointContractError(
                "IPLD links must be verified before admission"
            )
        object.__setattr__(
            self,
            "ipfs_backend_handle",
            _reference(
                self.ipfs_backend_handle,
                "ipfs_backend_handle",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "max_events_per_epoch",
            _integer(
                self.max_events_per_epoch,
                "max_events_per_epoch",
                minimum=1,
                maximum=10**7,
            ),
        )
        if mode is ReplicationMode.PARQUET_IPLD_IPFS:
            if not self.ipfs_publish or not self.ipfs_backend_handle:
                raise EntrypointContractError(
                    "IPFS replication requires publication and a backend handle"
                )
        elif self.ipfs_publish or self.pin:
            raise EntrypointContractError(
                "local Parquet/IPLD mode cannot claim IPFS publication or pinning"
            )
        if self.pin and not self.ipfs_publish:
            raise EntrypointContractError("pin requires IPFS publication")

    @property
    def grants_authority(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "mode": self.mode.value,
            "parquet_dataset_path": self.parquet_dataset_path,
            "parquet_schema_cid": self.parquet_schema_cid,
            "partition_keys": list(self.partition_keys),
            "ipld_manifest_schema_cid": self.ipld_manifest_schema_cid,
            "ipld_codec": self.ipld_codec,
            "cid_profile": self.cid_profile,
            "links_must_be_verified": self.links_must_be_verified,
            "car_export": self.car_export,
            "ipfs_publish": self.ipfs_publish,
            "ipfs_backend_handle": self.ipfs_backend_handle,
            "pin": self.pin,
            "max_events_per_epoch": self.max_events_per_epoch,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReplicationBinding:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(**{name: value[name] for name in cls.FIELDS})
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class TargetResolutionReceipt(_CanonicalContract):
    """Complete target/configuration resolution evidence, never authority."""

    SCHEMA: ClassVar[str] = TARGET_RESOLUTION_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "invocation_cid",
        "prompt_cid",
        "repository_root",
        "repository_id",
        "checkout_id",
        "scope_path",
        "head_tree_cid",
        "dirty_overlay_cid",
        "submodule_population_cid",
        "nested_repository_population_cid",
        "state_root",
        "run_namespace",
        "objective_cid",
        "objective_revision_cid",
        "plan_cid",
        "task_source_cid",
        "task_source_revision_cid",
        "task_source_kind",
        "policy_cid",
        "principal_ref",
        "authority_source_ref",
        "effect_ceiling_cid",
        "output_mode",
        "markdown_path",
        "duckdb_path",
        "provider_route",
        "capability_report_cid",
        "resource_budget_cid",
        "lane_ceiling",
        "merge_target",
        "worktree_strategy",
        "validation_profile_cid",
        "coordination_shard",
        "replication",
        "configuration_root_cid",
        "capability_catalog_cid",
        "decisions",
        "unresolved_fields",
        "resolved_at_ms",
        "fresh_until_ms",
        "is_authorization",
    )

    invocation_cid: str
    prompt_cid: str
    repository_root: str
    repository_id: str
    checkout_id: str
    scope_path: str
    head_tree_cid: str
    dirty_overlay_cid: str
    submodule_population_cid: str
    nested_repository_population_cid: str
    state_root: str
    run_namespace: str
    objective_cid: str
    objective_revision_cid: str
    plan_cid: str
    task_source_cid: str
    task_source_revision_cid: str
    task_source_kind: TaskSourceKind
    policy_cid: str
    principal_ref: str
    authority_source_ref: str
    effect_ceiling_cid: str
    output_mode: OutputMode
    markdown_path: str
    duckdb_path: str
    provider_route: ProviderRouteProvenance
    capability_report_cid: str
    resource_budget_cid: str
    lane_ceiling: int
    merge_target: str
    worktree_strategy: WorktreeStrategy
    validation_profile_cid: str
    coordination_shard: CoordinationShardBinding
    replication: ReplicationBinding
    configuration_root_cid: str
    capability_catalog_cid: str
    decisions: tuple[TargetInferenceDecision, ...]
    unresolved_fields: tuple[str, ...]
    resolved_at_ms: int
    fresh_until_ms: int
    is_authorization: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "invocation_cid", _cid(self.invocation_cid, "invocation_cid")
        )
        object.__setattr__(
            self, "prompt_cid", _prompt_cid(self.prompt_cid, "prompt_cid")
        )
        for name in ("repository_root", "scope_path", "state_root"):
            object.__setattr__(
                self,
                name,
                _absolute_path(getattr(self, name), name, required=False),
            )
        for name in ("repository_id", "checkout_id"):
            object.__setattr__(
                self,
                name,
                _reference(getattr(self, name), name, required=False),
            )
        for name in (
            "head_tree_cid",
            "dirty_overlay_cid",
            "submodule_population_cid",
            "nested_repository_population_cid",
            "objective_cid",
            "objective_revision_cid",
            "plan_cid",
            "task_source_cid",
            "task_source_revision_cid",
            "policy_cid",
            "effect_ceiling_cid",
            "capability_report_cid",
            "resource_budget_cid",
            "validation_profile_cid",
            "configuration_root_cid",
            "capability_catalog_cid",
        ):
            object.__setattr__(
                self,
                name,
                _cid(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self,
            "run_namespace",
            _token(self.run_namespace, "run_namespace", required=False),
        )
        object.__setattr__(
            self,
            "task_source_kind",
            _enum(self.task_source_kind, TaskSourceKind, "task_source_kind"),
        )
        object.__setattr__(
            self,
            "principal_ref",
            _reference(self.principal_ref, "principal_ref", required=False),
        )
        object.__setattr__(
            self,
            "authority_source_ref",
            _reference(
                self.authority_source_ref,
                "authority_source_ref",
                required=False,
            ),
        )
        object.__setattr__(
            self, "output_mode", _enum(self.output_mode, OutputMode, "output_mode")
        )
        object.__setattr__(
            self,
            "markdown_path",
            _absolute_path(self.markdown_path, "markdown_path", required=False),
        )
        object.__setattr__(
            self,
            "duckdb_path",
            _absolute_path(self.duckdb_path, "duckdb_path", required=False),
        )
        if not isinstance(self.provider_route, ProviderRouteProvenance):
            object.__setattr__(
                self,
                "provider_route",
                ProviderRouteProvenance.from_dict(self.provider_route),
            )
        object.__setattr__(
            self,
            "lane_ceiling",
            _integer(
                self.lane_ceiling,
                "lane_ceiling",
                minimum=1,
                maximum=MAX_LANES,
            ),
        )
        object.__setattr__(
            self,
            "merge_target",
            _text(self.merge_target, "merge_target", required=False, maximum=512),
        )
        object.__setattr__(
            self,
            "worktree_strategy",
            _enum(
                self.worktree_strategy,
                WorktreeStrategy,
                "worktree_strategy",
            ),
        )
        if not isinstance(self.coordination_shard, CoordinationShardBinding):
            object.__setattr__(
                self,
                "coordination_shard",
                CoordinationShardBinding.from_dict(self.coordination_shard),
            )
        if not isinstance(self.replication, ReplicationBinding):
            object.__setattr__(
                self,
                "replication",
                ReplicationBinding.from_dict(self.replication),
            )
        if isinstance(self.decisions, (str, bytes)) or not isinstance(
            self.decisions, Sequence
        ):
            raise EntrypointContractError("decisions must be a sequence")
        if len(self.decisions) > MAX_DECISIONS:
            raise ContractBoundsError(
                f"decisions exceeds {MAX_DECISIONS} items"
            )
        decisions = tuple(
            item
            if isinstance(item, TargetInferenceDecision)
            else TargetInferenceDecision.from_dict(item)
            for item in self.decisions
        )
        names = tuple(item.field_name for item in decisions)
        if len(names) != len(set(names)):
            raise EntrypointContractError("decisions contain duplicate fields")
        if set(names) != set(REQUIRED_TARGET_DECISION_FIELDS):
            missing = set(REQUIRED_TARGET_DECISION_FIELDS).difference(names)
            extra = set(names).difference(REQUIRED_TARGET_DECISION_FIELDS)
            raise EntrypointContractError(
                f"resolution decisions have missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )
        decisions = tuple(
            sorted(decisions, key=lambda item: item.field_name)
        )
        object.__setattr__(self, "decisions", decisions)
        unresolved = _text_tuple(
            self.unresolved_fields,
            "unresolved_fields",
            maximum_items=MAX_DECISIONS,
            sorted_items=True,
        )
        expected_unresolved = tuple(
            sorted(item.field_name for item in decisions if item.unresolved)
        )
        if unresolved != expected_unresolved:
            raise EntrypointContractError(
                "unresolved_fields must exactly match unresolved decisions"
            )
        object.__setattr__(self, "unresolved_fields", unresolved)
        projections = {
            "repository_root": self.repository_root,
            "state_root": self.state_root,
            "repository_id": self.repository_id,
            "checkout_id": self.checkout_id,
            "scope": self.scope_path,
            "tree_id": self.head_tree_cid,
            "dirty_overlay": self.dirty_overlay_cid,
            "submodules": self.submodule_population_cid,
            "nested_repositories": self.nested_repository_population_cid,
            "run_namespace": self.run_namespace,
            "objective": self.objective_cid,
            "plan": self.plan_cid,
            "task_source": self.task_source_cid,
            "policy": self.policy_cid,
            "principal": self.principal_ref,
            "authority_source": self.authority_source_ref,
            "effect_ceiling": self.effect_ceiling_cid,
            "output": self.output_mode.value,
            "provider": self.provider_route.content_id,
            "resources": self.resource_budget_cid,
            "lane_ceiling": str(self.lane_ceiling),
            "merge_target": self.merge_target,
            "worktree_strategy": self.worktree_strategy.value,
            "validation": self.validation_profile_cid,
            "coordination": self.coordination_shard.content_id,
            "replication": self.replication.content_id,
        }
        decisions_by_name = {item.field_name: item for item in decisions}
        for name, projected in projections.items():
            decision = decisions_by_name[name]
            if not decision.unresolved and decision.selected_value != projected:
                raise EntrypointContractError(
                    f"{name} decision does not match its resolved receipt field"
                )
        neutral_string_fields = {
            "repository_root",
            "state_root",
            "repository_id",
            "checkout_id",
            "scope",
            "tree_id",
            "dirty_overlay",
            "submodules",
            "nested_repositories",
            "run_namespace",
            "objective",
            "plan",
            "task_source",
            "policy",
            "principal",
            "authority_source",
            "effect_ceiling",
            "merge_target",
            "validation",
        }
        for name in unresolved:
            if name in neutral_string_fields and projections[name]:
                raise EntrypointContractError(
                    f"unresolved {name} must use an empty non-authoritative projection"
                )
        resolved_at = _integer(self.resolved_at_ms, "resolved_at_ms")
        fresh_until = _integer(self.fresh_until_ms, "fresh_until_ms")
        if fresh_until and fresh_until < resolved_at:
            raise EntrypointContractError(
                "fresh_until_ms cannot precede resolved_at_ms"
            )
        object.__setattr__(self, "resolved_at_ms", resolved_at)
        object.__setattr__(self, "fresh_until_ms", fresh_until)
        if _boolean(self.is_authorization, "is_authorization"):
            raise EntrypointContractError(
                "a target resolution receipt is evidence, not authorization"
            )
        if self.output_mode in {OutputMode.MARKDOWN, OutputMode.BOTH} and not self.markdown_path:
            raise EntrypointContractError(
                "selected output mode requires markdown_path"
            )
        if self.output_mode in {OutputMode.DUCKDB, OutputMode.BOTH} and not self.duckdb_path:
            raise EntrypointContractError(
                "selected output mode requires duckdb_path"
            )
        if self.repository_root == "/" or self.state_root == "/":
            raise EntrypointContractError(
                "repository_root and state_root cannot be the filesystem root"
            )
        if self.scope_path:
            _require_contained_path(
                self.scope_path, self.repository_root, "scope_path"
            )
        for name in ("markdown_path", "duckdb_path"):
            _require_contained_path(getattr(self, name), self.state_root, name)
        _require_contained_path(
            self.coordination_shard.database_path,
            self.state_root,
            "coordination database_path",
        )
        _require_contained_path(
            self.replication.parquet_dataset_path,
            self.state_root,
            "replication parquet_dataset_path",
        )
        if self.coordination_shard.writable and (
            not self.principal_ref
            or self.principal_ref != self.coordination_shard.owner_principal_ref
        ):
            raise EntrypointContractError(
                "a writable DuckDB shard requires the authenticated owner principal"
            )
        if unresolved and (
            self.coordination_shard.writable
            or self.replication.ipfs_publish
            or self.replication.pin
            or self.worktree_strategy is not WorktreeStrategy.NONE
        ):
            raise EntrypointContractError(
                "an unresolved preview must disable writable coordination, IPFS "
                "publication, and worktrees"
            )
        _ = self.content_id

    @property
    def receipt_cid(self) -> str:
        return self.content_id

    @property
    def authorizes_effects(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "invocation_cid": self.invocation_cid,
            "prompt_cid": self.prompt_cid,
            "repository_root": self.repository_root,
            "repository_id": self.repository_id,
            "checkout_id": self.checkout_id,
            "scope_path": self.scope_path,
            "head_tree_cid": self.head_tree_cid,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "submodule_population_cid": self.submodule_population_cid,
            "nested_repository_population_cid": (
                self.nested_repository_population_cid
            ),
            "state_root": self.state_root,
            "run_namespace": self.run_namespace,
            "objective_cid": self.objective_cid,
            "objective_revision_cid": self.objective_revision_cid,
            "plan_cid": self.plan_cid,
            "task_source_cid": self.task_source_cid,
            "task_source_revision_cid": self.task_source_revision_cid,
            "task_source_kind": self.task_source_kind.value,
            "policy_cid": self.policy_cid,
            "principal_ref": self.principal_ref,
            "authority_source_ref": self.authority_source_ref,
            "effect_ceiling_cid": self.effect_ceiling_cid,
            "output_mode": self.output_mode.value,
            "markdown_path": self.markdown_path,
            "duckdb_path": self.duckdb_path,
            "provider_route": self.provider_route.to_dict(),
            "capability_report_cid": self.capability_report_cid,
            "resource_budget_cid": self.resource_budget_cid,
            "lane_ceiling": self.lane_ceiling,
            "merge_target": self.merge_target,
            "worktree_strategy": self.worktree_strategy.value,
            "validation_profile_cid": self.validation_profile_cid,
            "coordination_shard": self.coordination_shard.to_dict(),
            "replication": self.replication.to_dict(),
            "configuration_root_cid": self.configuration_root_cid,
            "capability_catalog_cid": self.capability_catalog_cid,
            "decisions": [item.to_dict() for item in self.decisions],
            "unresolved_fields": list(self.unresolved_fields),
            "resolved_at_ms": self.resolved_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
            "is_authorization": self.is_authorization,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> TargetResolutionReceipt:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(**{name: value[name] for name in cls.FIELDS})
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class ResolvedSupervisorProfile(_CanonicalContract):
    """Immutable compilation of verbose expert supervisor configuration."""

    SCHEMA: ClassVar[str] = RESOLVED_PROFILE_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "profile_name",
        "profile_source_cid",
        "target_resolution_receipt_cid",
        "mode",
        "repository_root",
        "state_root",
        "run_namespace",
        "policy_cid",
        "principal_ref",
        "effect_ceiling_cid",
        "task_source_kind",
        "task_source_path",
        "task_source_cid",
        "output_mode",
        "markdown_path",
        "duckdb_path",
        "provider_route",
        "resource_budget",
        "validation_profile_cid",
        "lifecycle_health_contract_cid",
        "coordination_shard",
        "replication",
        "supervisor_argv",
        "daemon_argv",
        "environment_names",
        "credential_handles",
        "expected_effects",
        "worktree_strategy",
        "merge_target",
    )

    profile_name: str
    profile_source_cid: str
    target_resolution_receipt_cid: str
    mode: InvocationMode
    repository_root: str
    state_root: str
    run_namespace: str
    policy_cid: str
    principal_ref: str
    effect_ceiling_cid: str
    task_source_kind: TaskSourceKind
    task_source_path: str
    task_source_cid: str
    output_mode: OutputMode
    markdown_path: str
    duckdb_path: str
    provider_route: ProviderRouteProvenance
    resource_budget: ResourceBudget
    validation_profile_cid: str
    lifecycle_health_contract_cid: str
    coordination_shard: CoordinationShardBinding
    replication: ReplicationBinding
    supervisor_argv: tuple[str, ...]
    daemon_argv: tuple[str, ...]
    environment_names: tuple[str, ...]
    credential_handles: tuple[str, ...]
    expected_effects: tuple[ExpectedEffect, ...]
    worktree_strategy: WorktreeStrategy
    merge_target: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "profile_name", _token(self.profile_name, "profile_name")
        )
        for name in (
            "profile_source_cid",
            "target_resolution_receipt_cid",
            "policy_cid",
            "effect_ceiling_cid",
            "task_source_cid",
            "validation_profile_cid",
            "lifecycle_health_contract_cid",
        ):
            object.__setattr__(self, name, _cid(getattr(self, name), name))
        object.__setattr__(self, "mode", _enum(self.mode, InvocationMode, "mode"))
        for name in ("repository_root", "state_root", "task_source_path"):
            object.__setattr__(
                self, name, _absolute_path(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "run_namespace",
            _token(self.run_namespace, "run_namespace"),
        )
        object.__setattr__(
            self,
            "principal_ref",
            _reference(self.principal_ref, "principal_ref"),
        )
        object.__setattr__(
            self,
            "task_source_kind",
            _enum(self.task_source_kind, TaskSourceKind, "task_source_kind"),
        )
        object.__setattr__(
            self, "output_mode", _enum(self.output_mode, OutputMode, "output_mode")
        )
        for name in ("markdown_path", "duckdb_path"):
            object.__setattr__(
                self,
                name,
                _absolute_path(getattr(self, name), name, required=False),
            )
        if self.output_mode in {OutputMode.MARKDOWN, OutputMode.BOTH} and not self.markdown_path:
            raise EntrypointContractError("output mode requires markdown_path")
        if self.output_mode in {OutputMode.DUCKDB, OutputMode.BOTH} and not self.duckdb_path:
            raise EntrypointContractError("output mode requires duckdb_path")
        if not isinstance(self.provider_route, ProviderRouteProvenance):
            object.__setattr__(
                self,
                "provider_route",
                ProviderRouteProvenance.from_dict(self.provider_route),
            )
        if not isinstance(self.resource_budget, ResourceBudget):
            object.__setattr__(
                self,
                "resource_budget",
                ResourceBudget.from_dict(self.resource_budget),
            )
        if not isinstance(self.coordination_shard, CoordinationShardBinding):
            object.__setattr__(
                self,
                "coordination_shard",
                CoordinationShardBinding.from_dict(self.coordination_shard),
            )
        if not isinstance(self.replication, ReplicationBinding):
            object.__setattr__(
                self,
                "replication",
                ReplicationBinding.from_dict(self.replication),
            )
        object.__setattr__(
            self, "supervisor_argv", _argv(self.supervisor_argv, "supervisor_argv")
        )
        object.__setattr__(
            self, "daemon_argv", _argv(self.daemon_argv, "daemon_argv")
        )
        object.__setattr__(
            self, "environment_names", _environment_names(self.environment_names)
        )
        object.__setattr__(
            self,
            "credential_handles",
            _text_tuple(
                self.credential_handles,
                "credential_handles",
                maximum_items=64,
                item_kind="reference",
                sorted_items=True,
            ),
        )
        object.__setattr__(
            self,
            "expected_effects",
            _enum_tuple(
                self.expected_effects,
                ExpectedEffect,
                "expected_effects",
                sorted_items=True,
            ),
        )
        object.__setattr__(
            self,
            "worktree_strategy",
            _enum(
                self.worktree_strategy,
                WorktreeStrategy,
                "worktree_strategy",
            ),
        )
        object.__setattr__(
            self,
            "merge_target",
            _text(self.merge_target, "merge_target", required=False, maximum=512),
        )
        if self.resource_budget.max_lanes > MAX_LANES:
            raise ContractBoundsError("profile lane ceiling exceeds global bound")
        if self.repository_root == "/" or self.state_root == "/":
            raise EntrypointContractError(
                "repository_root and state_root cannot be the filesystem root"
            )
        for name in ("task_source_path", "markdown_path", "duckdb_path"):
            _require_contained_path(getattr(self, name), self.state_root, name)
        _require_contained_path(
            self.coordination_shard.database_path,
            self.state_root,
            "coordination database_path",
        )
        _require_contained_path(
            self.replication.parquet_dataset_path,
            self.state_root,
            "replication parquet_dataset_path",
        )
        if self.coordination_shard.writable and (
            self.principal_ref != self.coordination_shard.owner_principal_ref
        ):
            raise EntrypointContractError(
                "a writable DuckDB shard requires the profile owner principal"
            )
        _ = self.content_id

    @property
    def profile_cid(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "profile_name": self.profile_name,
            "profile_source_cid": self.profile_source_cid,
            "target_resolution_receipt_cid": (
                self.target_resolution_receipt_cid
            ),
            "mode": self.mode.value,
            "repository_root": self.repository_root,
            "state_root": self.state_root,
            "run_namespace": self.run_namespace,
            "policy_cid": self.policy_cid,
            "principal_ref": self.principal_ref,
            "effect_ceiling_cid": self.effect_ceiling_cid,
            "task_source_kind": self.task_source_kind.value,
            "task_source_path": self.task_source_path,
            "task_source_cid": self.task_source_cid,
            "output_mode": self.output_mode.value,
            "markdown_path": self.markdown_path,
            "duckdb_path": self.duckdb_path,
            "provider_route": self.provider_route.to_dict(),
            "resource_budget": self.resource_budget.to_dict(),
            "validation_profile_cid": self.validation_profile_cid,
            "lifecycle_health_contract_cid": (
                self.lifecycle_health_contract_cid
            ),
            "coordination_shard": self.coordination_shard.to_dict(),
            "replication": self.replication.to_dict(),
            "supervisor_argv": list(self.supervisor_argv),
            "daemon_argv": list(self.daemon_argv),
            "environment_names": list(self.environment_names),
            "credential_handles": list(self.credential_handles),
            "expected_effects": [item.value for item in self.expected_effects],
            "worktree_strategy": self.worktree_strategy.value,
            "merge_target": self.merge_target,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> ResolvedSupervisorProfile:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(**{name: value[name] for name in cls.FIELDS})
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class LaunchPlan(_CanonicalContract):
    """Exact, immutable runtime-construction projection."""

    SCHEMA: ClassVar[str] = LAUNCH_PLAN_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "invocation_cid",
        "target_resolution_receipt_cid",
        "resolved_profile_cid",
        "working_directory",
        "state_path",
        "task_source_path",
        "supervisor_argv",
        "daemon_argv",
        "environment_names",
        "provider_route_cid",
        "resource_budget_cid",
        "validation_profile_cid",
        "lifecycle_profile_cid",
        "coordination_shard",
        "replication",
        "expected_effects",
        "idempotency_key",
        "adoption_key",
        "lease_required",
        "authorization_required",
        "dry_run",
    )

    invocation_cid: str
    target_resolution_receipt_cid: str
    resolved_profile_cid: str
    working_directory: str
    state_path: str
    task_source_path: str
    supervisor_argv: tuple[str, ...]
    daemon_argv: tuple[str, ...]
    environment_names: tuple[str, ...]
    provider_route_cid: str
    resource_budget_cid: str
    validation_profile_cid: str
    lifecycle_profile_cid: str
    coordination_shard: CoordinationShardBinding
    replication: ReplicationBinding
    expected_effects: tuple[ExpectedEffect, ...]
    idempotency_key: str
    adoption_key: str
    lease_required: bool
    authorization_required: bool
    dry_run: bool

    def __post_init__(self) -> None:
        for name in (
            "invocation_cid",
            "target_resolution_receipt_cid",
            "resolved_profile_cid",
            "provider_route_cid",
            "resource_budget_cid",
            "validation_profile_cid",
            "lifecycle_profile_cid",
        ):
            object.__setattr__(self, name, _cid(getattr(self, name), name))
        for name in ("working_directory", "state_path", "task_source_path"):
            object.__setattr__(
                self, name, _absolute_path(getattr(self, name), name)
            )
        object.__setattr__(
            self, "supervisor_argv", _argv(self.supervisor_argv, "supervisor_argv")
        )
        object.__setattr__(
            self, "daemon_argv", _argv(self.daemon_argv, "daemon_argv")
        )
        object.__setattr__(
            self, "environment_names", _environment_names(self.environment_names)
        )
        if not isinstance(self.coordination_shard, CoordinationShardBinding):
            object.__setattr__(
                self,
                "coordination_shard",
                CoordinationShardBinding.from_dict(self.coordination_shard),
            )
        if not isinstance(self.replication, ReplicationBinding):
            object.__setattr__(
                self,
                "replication",
                ReplicationBinding.from_dict(self.replication),
            )
        object.__setattr__(
            self,
            "expected_effects",
            _enum_tuple(
                self.expected_effects,
                ExpectedEffect,
                "expected_effects",
                sorted_items=True,
            ),
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _reference(self.idempotency_key, "idempotency_key"),
        )
        object.__setattr__(
            self, "adoption_key", _reference(self.adoption_key, "adoption_key")
        )
        for name in ("lease_required", "authorization_required", "dry_run"):
            object.__setattr__(
                self, name, _boolean(getattr(self, name), name)
            )
        mutating = any(
            effect
            not in {ExpectedEffect.INSPECT_REPOSITORY}
            for effect in self.expected_effects
        )
        if mutating and not self.dry_run and (
            not self.lease_required or not self.authorization_required
        ):
            raise EntrypointContractError(
                "a mutating launch requires lease and authorization boundaries"
            )
        if mutating and not self.dry_run and (
            not self.coordination_shard.writable
            or self.coordination_shard.fencing_generation < 1
        ):
            raise EntrypointContractError(
                "a mutating launch requires its writable DuckDB owner and fence"
            )
        _ = self.content_id

    @property
    def launch_plan_cid(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "invocation_cid": self.invocation_cid,
            "target_resolution_receipt_cid": (
                self.target_resolution_receipt_cid
            ),
            "resolved_profile_cid": self.resolved_profile_cid,
            "working_directory": self.working_directory,
            "state_path": self.state_path,
            "task_source_path": self.task_source_path,
            "supervisor_argv": list(self.supervisor_argv),
            "daemon_argv": list(self.daemon_argv),
            "environment_names": list(self.environment_names),
            "provider_route_cid": self.provider_route_cid,
            "resource_budget_cid": self.resource_budget_cid,
            "validation_profile_cid": self.validation_profile_cid,
            "lifecycle_profile_cid": self.lifecycle_profile_cid,
            "coordination_shard": self.coordination_shard.to_dict(),
            "replication": self.replication.to_dict(),
            "expected_effects": [item.value for item in self.expected_effects],
            "idempotency_key": self.idempotency_key,
            "adoption_key": self.adoption_key,
            "lease_required": self.lease_required,
            "authorization_required": self.authorization_required,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LaunchPlan:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(**{name: value[name] for name in cls.FIELDS})
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class RunHandle(_CanonicalContract):
    """Durable run identity and continuation cursor.

    Wall-clock fields serialize for diagnostics but are excluded from semantic
    identity.  State/revision/cursor changes still produce a new handle CID.
    """

    SCHEMA: ClassVar[str] = RUN_HANDLE_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "run_id",
        "run_revision",
        "target_resolution_receipt_cid",
        "invocation_cid",
        "prompt_cid",
        "workflow_cid",
        "scan_cid",
        "plan_cid",
        "materialization_cid",
        "task_source_cid",
        "task_source_revision_cid",
        "lifecycle_profile_cid",
        "process_cid",
        "objective_cid",
        "objective_revision_cid",
        "lease_id",
        "fencing_generation",
        "state",
        "health",
        "state_revision_cid",
        "health_revision_cid",
        "event_cursor",
        "continuation_action",
        "pending_approval_cid",
        "ambiguity_cid",
        "created_at_ms",
        "updated_at_ms",
    )

    run_id: str
    run_revision: int
    target_resolution_receipt_cid: str
    invocation_cid: str
    prompt_cid: str
    workflow_cid: str
    scan_cid: str
    plan_cid: str
    materialization_cid: str
    task_source_cid: str
    task_source_revision_cid: str
    lifecycle_profile_cid: str
    process_cid: str
    objective_cid: str
    objective_revision_cid: str
    lease_id: str
    fencing_generation: int
    state: RunState
    health: RunHealth
    state_revision_cid: str
    health_revision_cid: str
    event_cursor: str
    continuation_action: ContinuationAction
    pending_approval_cid: str
    ambiguity_cid: str
    created_at_ms: int
    updated_at_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _cid(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "run_revision",
            _integer(self.run_revision, "run_revision", minimum=1),
        )
        for name in (
            "target_resolution_receipt_cid",
            "invocation_cid",
        ):
            object.__setattr__(self, name, _cid(getattr(self, name), name))
        object.__setattr__(
            self, "prompt_cid", _prompt_cid(self.prompt_cid, "prompt_cid")
        )
        for name in (
            "workflow_cid",
            "scan_cid",
            "plan_cid",
            "materialization_cid",
            "task_source_cid",
            "task_source_revision_cid",
            "lifecycle_profile_cid",
            "process_cid",
            "objective_cid",
            "objective_revision_cid",
            "state_revision_cid",
            "health_revision_cid",
            "pending_approval_cid",
            "ambiguity_cid",
        ):
            object.__setattr__(
                self,
                name,
                _cid(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self, "lease_id", _reference(self.lease_id, "lease_id", required=False)
        )
        object.__setattr__(
            self,
            "fencing_generation",
            _integer(self.fencing_generation, "fencing_generation"),
        )
        state = _enum(self.state, RunState, "state")
        health = _enum(self.health, RunHealth, "health")
        action = _enum(
            self.continuation_action,
            ContinuationAction,
            "continuation_action",
        )
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "health", health)
        object.__setattr__(self, "continuation_action", action)
        object.__setattr__(
            self,
            "event_cursor",
            _reference(self.event_cursor, "event_cursor", required=False),
        )
        created = _integer(self.created_at_ms, "created_at_ms")
        updated = _integer(self.updated_at_ms, "updated_at_ms")
        if updated < created:
            raise EntrypointContractError(
                "updated_at_ms cannot precede created_at_ms"
            )
        object.__setattr__(self, "created_at_ms", created)
        object.__setattr__(self, "updated_at_ms", updated)
        if state is RunState.NEEDS_INPUT and (
            not self.ambiguity_cid or action is not ContinuationAction.ASK_INPUT
        ):
            raise EntrypointContractError(
                "needs_input requires ambiguity evidence and ask_input continuation"
            )
        if state is RunState.RUNNING and (
            not self.process_cid
            or not self.lifecycle_profile_cid
            or not self.state_revision_cid
            or not self.health_revision_cid
            or health is RunHealth.UNKNOWN
            or action is not ContinuationAction.MONITOR
        ):
            raise EntrypointContractError(
                "running handle requires process, lifecycle, revision, observed "
                "health, and monitor continuation"
            )
        if bool(self.lease_id) != bool(self.fencing_generation):
            raise EntrypointContractError(
                "lease_id and fencing_generation must be present together"
            )
        if state is RunState.RUNNING and not self.lease_id:
            raise EntrypointContractError(
                "a running handle requires an active fenced lease"
            )
        terminal_states = {
            RunState.COMPLETED,
            RunState.CANCELLED,
            RunState.FAILED,
        }
        if state in terminal_states and (
            health is not RunHealth.TERMINAL
            or action is not ContinuationAction.NONE
        ):
            raise EntrypointContractError(
                "terminal handles require terminal health and no continuation"
            )
        if state is not RunState.NEEDS_INPUT and action is ContinuationAction.ASK_INPUT:
            raise EntrypointContractError(
                "ask_input continuation is valid only for needs_input"
            )

    @property
    def handle_cid(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "run_id": self.run_id,
            "run_revision": self.run_revision,
            "target_resolution_receipt_cid": (
                self.target_resolution_receipt_cid
            ),
            "invocation_cid": self.invocation_cid,
            "prompt_cid": self.prompt_cid,
            "workflow_cid": self.workflow_cid,
            "scan_cid": self.scan_cid,
            "plan_cid": self.plan_cid,
            "materialization_cid": self.materialization_cid,
            "task_source_cid": self.task_source_cid,
            "task_source_revision_cid": self.task_source_revision_cid,
            "lifecycle_profile_cid": self.lifecycle_profile_cid,
            "process_cid": self.process_cid,
            "objective_cid": self.objective_cid,
            "objective_revision_cid": self.objective_revision_cid,
            "lease_id": self.lease_id,
            "fencing_generation": self.fencing_generation,
            "state": self.state.value,
            "health": self.health.value,
            "state_revision_cid": self.state_revision_cid,
            "health_revision_cid": self.health_revision_cid,
            "event_cursor": self.event_cursor,
            "continuation_action": self.continuation_action.value,
            "pending_approval_cid": self.pending_approval_cid,
            "ambiguity_cid": self.ambiguity_cid,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
        }

    @property
    def semantic_id(self) -> str:
        """Stable semantic revision identity excluding diagnostic wall time."""

        payload = self._payload()
        payload.pop("created_at_ms")
        payload.pop("updated_at_ms")
        return cid_for_dag_json(payload)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunHandle:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(**{name: value[name] for name in cls.FIELDS})
        return cls._verify_claimed(value, result)


@dataclass(frozen=True)
class SupervisorInvocationResult(_CanonicalContract):
    """Transport-neutral result for a prompt-first run invocation."""

    SCHEMA: ClassVar[str] = INVOCATION_RESULT_SCHEMA
    FIELDS: ClassVar[tuple[str, ...]] = (
        "invocation_cid",
        "status",
        "target_resolution_receipt_cid",
        "launch_plan_cid",
        "run_handle",
        "reason_codes",
        "questions",
        "continuation_action",
        "effect_receipt_cids",
        "event_cursor",
        "error_code",
    )

    invocation_cid: str
    status: InvocationStatus
    target_resolution_receipt_cid: str
    launch_plan_cid: str
    run_handle: RunHandle | None
    reason_codes: tuple[str, ...]
    questions: tuple[str, ...]
    continuation_action: ContinuationAction
    effect_receipt_cids: tuple[str, ...]
    event_cursor: str
    error_code: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "invocation_cid", _cid(self.invocation_cid, "invocation_cid")
        )
        status = _enum(self.status, InvocationStatus, "status")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "target_resolution_receipt_cid",
            _cid(
                self.target_resolution_receipt_cid,
                "target_resolution_receipt_cid",
            ),
        )
        object.__setattr__(
            self,
            "launch_plan_cid",
            _cid(self.launch_plan_cid, "launch_plan_cid", required=False),
        )
        handle = self.run_handle
        if handle is not None and not isinstance(handle, RunHandle):
            if isinstance(handle, Mapping):
                handle = RunHandle.from_dict(handle)
            else:
                raise EntrypointContractError(
                    "run_handle must be a RunHandle or null"
                )
        object.__setattr__(self, "run_handle", handle)
        object.__setattr__(
            self,
            "reason_codes",
            _text_tuple(
                self.reason_codes,
                "reason_codes",
                item_kind="reason",
                sorted_items=True,
            ),
        )
        questions = _text_tuple(
            self.questions,
            "questions",
            maximum_items=MAX_QUESTIONS,
            item_kind="reason",
            unique=True,
        )
        object.__setattr__(self, "questions", questions)
        action = _enum(
            self.continuation_action,
            ContinuationAction,
            "continuation_action",
        )
        object.__setattr__(self, "continuation_action", action)
        object.__setattr__(
            self,
            "effect_receipt_cids",
            _text_tuple(
                self.effect_receipt_cids,
                "effect_receipt_cids",
                item_kind="cid",
                sorted_items=True,
            ),
        )
        object.__setattr__(
            self,
            "event_cursor",
            _reference(self.event_cursor, "event_cursor", required=False),
        )
        error = self.error_code
        if error:
            error = _reason(error, "error_code")
        object.__setattr__(self, "error_code", error)
        run_statuses = {
            InvocationStatus.STARTED,
            InvocationStatus.ADOPTED,
            InvocationStatus.RUNNING,
            InvocationStatus.COMPLETED,
        }
        if status in run_statuses and (
            handle is None or not self.launch_plan_cid
        ):
            raise EntrypointContractError(
                "run-producing results require launch plan and run handle"
            )
        if handle is not None and (
            handle.invocation_cid != self.invocation_cid
            or handle.target_resolution_receipt_cid
            != self.target_resolution_receipt_cid
        ):
            raise EntrypointContractError(
                "run_handle links must match the invocation result"
            )
        if handle is not None and self.event_cursor != handle.event_cursor:
            raise EntrypointContractError(
                "result event_cursor must match the embedded run handle"
            )
        expected_states = {
            InvocationStatus.STARTED: {RunState.STARTING, RunState.RUNNING},
            InvocationStatus.ADOPTED: {RunState.ADOPTING, RunState.RUNNING},
            InvocationStatus.RUNNING: {RunState.RUNNING},
            InvocationStatus.COMPLETED: {RunState.COMPLETED},
        }
        if status in expected_states and handle is not None:
            if handle.state not in expected_states[status]:
                raise EntrypointContractError(
                    "invocation status does not match run_handle state"
                )
            if action is not handle.continuation_action:
                raise EntrypointContractError(
                    "result continuation must match the run handle"
                )
        if status is InvocationStatus.NEEDS_INPUT and (
            not questions or action is not ContinuationAction.ASK_INPUT
        ):
            raise EntrypointContractError(
                "needs_input requires a bounded question and ask_input continuation"
            )
        if status in {
            InvocationStatus.DENIED,
            InvocationStatus.UNAVAILABLE,
            InvocationStatus.FAILED,
        } and not error:
            raise EntrypointContractError(
                "denied/unavailable/failed results require error_code"
            )
        if status not in {
            InvocationStatus.DENIED,
            InvocationStatus.UNAVAILABLE,
            InvocationStatus.FAILED,
        } and error:
            raise EntrypointContractError(
                "successful or resumable results cannot carry error_code"
            )
        if status is not InvocationStatus.NEEDS_INPUT and questions:
            raise EntrypointContractError(
                "questions are valid only for needs_input"
            )
        if status in {
            InvocationStatus.PREVIEW,
            InvocationStatus.NEEDS_INPUT,
            InvocationStatus.DENIED,
            InvocationStatus.UNAVAILABLE,
        } and self.effect_receipt_cids:
            raise EntrypointContractError(
                "non-effect invocation states cannot claim effect receipts"
            )
        if status in {InvocationStatus.DENIED, InvocationStatus.UNAVAILABLE} and (
            handle is not None or self.launch_plan_cid
        ):
            raise EntrypointContractError(
                "denied/unavailable results cannot claim a launch or run"
            )

    @property
    def succeeded(self) -> bool:
        return self.status in {
            InvocationStatus.PREVIEW,
            InvocationStatus.STARTED,
            InvocationStatus.ADOPTED,
            InvocationStatus.RUNNING,
            InvocationStatus.COMPLETED,
        }

    @property
    def result_cid(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "invocation_cid": self.invocation_cid,
            "status": self.status.value,
            "target_resolution_receipt_cid": (
                self.target_resolution_receipt_cid
            ),
            "launch_plan_cid": self.launch_plan_cid,
            "run_handle": (
                self.run_handle.to_dict() if self.run_handle is not None else None
            ),
            "reason_codes": list(self.reason_codes),
            "questions": list(self.questions),
            "continuation_action": self.continuation_action.value,
            "effect_receipt_cids": list(self.effect_receipt_cids),
            "event_cursor": self.event_cursor,
            "error_code": self.error_code,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> SupervisorInvocationResult:
        _closed(value, schema=cls.SCHEMA, fields=cls.FIELDS)
        result = cls(**{name: value[name] for name in cls.FIELDS})
        return cls._verify_claimed(value, result)


__all__ = [
    "COORDINATION_SHARD_SCHEMA",
    "ContractBoundsError",
    "ContractIdentityError",
    "ContinuationAction",
    "CoordinationShardBinding",
    "DEFAULT_PARQUET_PARTITIONS",
    "DecisionEffect",
    "EntrypointContractError",
    "ExpectedEffect",
    "INVOCATION_REQUEST_SCHEMA",
    "INVOCATION_RESULT_SCHEMA",
    "InvocationBudget",
    "InvocationMode",
    "InvocationStatus",
    "LaunchPlan",
    "OutputMode",
    "ProviderFallbackReason",
    "ProviderRouteProvenance",
    "ProviderSelection",
    "REPLICATION_BINDING_SCHEMA",
    "REQUIRED_TARGET_DECISION_FIELDS",
    "ReplicationBinding",
    "ReplicationMode",
    "ResolvedSupervisorProfile",
    "ResolutionDisposition",
    "ResolutionSource",
    "ResourceBudget",
    "RevalidationRule",
    "RunHandle",
    "RunHealth",
    "RunState",
    "SecretBearingRecordError",
    "SupervisorInvocationRequest",
    "SupervisorInvocationResult",
    "TARGET_DECISION_SCHEMA",
    "TARGET_RESOLUTION_SCHEMA",
    "TargetCandidate",
    "TargetInferenceDecision",
    "TargetResolutionReceipt",
    "TaskSourceKind",
    "UnknownContractFieldError",
    "WorktreeStrategy",
]
