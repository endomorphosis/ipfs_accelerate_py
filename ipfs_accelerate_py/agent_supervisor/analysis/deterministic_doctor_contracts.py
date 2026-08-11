"""Bounded, content-addressed contracts for the deterministic doctor (LPR-029).

Every doctor stage exchanges immutable, body-free, content-addressed records.
This module is the schema and authority boundary for:

* operations ``inspect`` / ``explain`` / ``plan`` / ``repair`` / ``replay`` /
  ``rollback`` (inspect/explain/plan are read-only; report-only is default);
* authority roots spanning forest/tree/overlay/file/AST/graph/corpus/index/
  model/cache/operator/translator/solver/kernel/toolchain/policy/sandbox/
  environment/lease;
* observed facts versus expected behavior, and nomination versus proof/write
  authority;
* dispositions ``supported``, ``abstain``, ``approval_required``,
  ``rolled_back``, and ``quarantined``; and
* fail-closed rejection of forged CIDs, bodies/secrets, cycles, unbounded
  data, partial plans, open required frontiers, advisory semantic-authority
  promotion, and any LLM or remote model-provider invocation in deterministic
  mode.

This module defines records only.  Admission and mode gates live in
:mod:`ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy`.
It reuses the canonical identity bridge and never invokes a model provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


# ---------------------------------------------------------------------------
# Version, schemas, bounds
# ---------------------------------------------------------------------------

DETERMINISTIC_DOCTOR_VERSION: Final[int] = 1
DETERMINISTIC_DOCTOR_CONTRACTS_INTERFACE: Final[str] = "DeterministicDoctorContracts@1"

MAX_RECORD_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_FINDING_COUNT: Final[int] = 256
MAX_OPERATOR_COUNT: Final[int] = 64
MAX_STEP_COUNT: Final[int] = 256
MAX_CONSUMER_COUNT: Final[int] = 1_024
MAX_FRONTIER_COUNT: Final[int] = 256
MAX_SPAN_OFFSET: Final[int] = 2**63 - 1

DOCTOR_AUTHORITY_ROOTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/authority-roots@1"
)
DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/evidence-snapshot@1"
)
DETERMINISTIC_DOCTOR_FINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/finding@1"
)
DOCTOR_REPAIR_OPERATOR_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/repair-operator-spec@1"
)
DETERMINISTIC_DOCTOR_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/plan@1"
)
DOCTOR_PROOF_CACHE_AUDIT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/proof-cache-audit-receipt@1"
)
DETERMINISTIC_DOCTOR_RUN_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/run-receipt@1"
)
DOCTOR_CONSUMER_DISPOSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/consumer-disposition@1"
)
DOCTOR_PLAN_STEP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/plan-step@1"
)
DOCTOR_EDIT_SITE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/edit-site@1"
)
DOCTOR_RESOURCE_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/resource-bounds@1"
)

# Policy schema identity shared with the scheduler / policy module.
DETERMINISTIC_DOCTOR_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.deterministic_doctor.policy@1"
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class DoctorOperation(str, Enum):
    """Closed doctor surface operations.

    ``inspect``, ``explain``, and ``plan`` never write.  ``repair`` is
    write-capable only after an admitted plan, lease, checkpoint, and
    rollback strategy exist.  ``replay`` and ``rollback`` are non-mutating
    with respect to new semantic repairs (rollback restores a prior
    checkpoint).
    """

    INSPECT = "inspect"
    EXPLAIN = "explain"
    PLAN = "plan"
    REPAIR = "repair"
    REPLAY = "replay"
    ROLLBACK = "rollback"

    @property
    def is_read_only(self) -> bool:
        return self in (
            DoctorOperation.INSPECT,
            DoctorOperation.EXPLAIN,
            DoctorOperation.PLAN,
            DoctorOperation.REPLAY,
        )

    @property
    def may_write(self) -> bool:
        return self is DoctorOperation.REPAIR

    @property
    def is_compensating(self) -> bool:
        return self is DoctorOperation.ROLLBACK


class DoctorMode(str, Enum):
    """Orthogonal no-model automation ladder (report-only default)."""

    REPORT_ONLY = "report_only"
    PLAN = "plan"
    SANDBOX_AUTO = "sandbox_auto"
    NARROW_AUTO = "narrow_auto"

    @property
    def rank(self) -> int:
        return {
            DoctorMode.REPORT_ONLY: 0,
            DoctorMode.PLAN: 1,
            DoctorMode.SANDBOX_AUTO: 2,
            DoctorMode.NARROW_AUTO: 3,
        }[self]

    @property
    def allows_source_write(self) -> bool:
        return self is DoctorMode.NARROW_AUTO

    @property
    def allows_sandbox_write(self) -> bool:
        return self in (DoctorMode.SANDBOX_AUTO, DoctorMode.NARROW_AUTO)

    @property
    def allows_plan_materialization(self) -> bool:
        return self.rank >= DoctorMode.PLAN.rank


class DoctorRepairDisposition(str, Enum):
    """Closed finding / repair outcomes for the deterministic doctor.

    ``supported``, ``abstain``, and ``approval_required`` are normal analytical
    outcomes.  ``rolled_back`` and ``quarantined`` record compensating or
    fail-closed terminal states.
    """

    SUPPORTED = "supported"
    ABSTAIN = "abstain"
    APPROVAL_REQUIRED = "approval_required"
    ROLLED_BACK = "rolled_back"
    QUARANTINED = "quarantined"

    @property
    def grants_write_authority(self) -> bool:
        return self is DoctorRepairDisposition.SUPPORTED

    @property
    def is_terminal_failure(self) -> bool:
        return self in (
            DoctorRepairDisposition.ROLLED_BACK,
            DoctorRepairDisposition.QUARANTINED,
        )


class DoctorEvidenceRole(str, Enum):
    """Authority role of a bound evidence or candidate reference.

    Observed facts and nominations never grant expected-behavior, proof, or
    write authority by themselves.
    """

    OBSERVED_FACT = "observed_fact"
    EXPECTED_BEHAVIOR = "expected_behavior"
    NOMINATION = "nomination"
    PROOF = "proof"
    WRITE = "write"


class DoctorPlanDisposition(str, Enum):
    """Closed outcomes of a complete deterministic-doctor plan."""

    ADMITTED = "admitted"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    APPROVAL_REQUIRED = "approval_required"

    @property
    def grants_write_authority(self) -> bool:
        return self is DoctorPlanDisposition.ADMITTED


class DoctorCacheAuditDisposition(str, Enum):
    """Closed outcomes of a proof-cache federation audit."""

    HIT = "hit"
    MISS = "miss"
    STALE = "stale"
    QUARANTINED = "quarantined"
    RECONSTRUCTED = "reconstructed"


class DoctorOperatorKind(str, Enum):
    """Closed analytical repair operator kinds (initially eligible set)."""

    EXACT_RENAME = "exact_rename"
    ADD_ARGUMENT = "add_argument"
    RENAME_ARGUMENT = "rename_argument"
    REORDER_ARGUMENT = "reorder_argument"
    THREAD_ARGUMENT = "thread_argument"
    ADD_IMPORT = "add_import"
    ADD_EXPORT = "add_export"
    ADD_REGISTRATION = "add_registration"
    ADD_CONSTRUCTOR_ROUTE = "add_constructor_route"
    ADD_FACTORY_ROUTE = "add_factory_route"
    FINITE_ADAPTER = "finite_adapter"
    SCHEMA_PROJECTION = "schema_projection"
    RESTORE_TRACKED_ARTIFACT = "restore_tracked_artifact"


class DoctorApprovalClass(str, Enum):
    """Change classes that remain approval-required under deterministic mode."""

    DOCTOR_TRUSTED_COMPUTING_BASE = "doctor_trusted_computing_base"
    STATEFUL_BEHAVIOR = "stateful_behavior"
    PUBLIC_API_OR_SCHEMA = "public_api_or_schema"
    DYNAMIC_OR_GENERATED_CODE = "dynamic_or_generated_code"
    NATIVE_OR_FFI = "native_or_ffi"
    CROSS_REPOSITORY_EDIT = "cross_repository_edit"
    NEW_EXTERNAL_DEPENDENCY = "new_external_dependency"
    UNSUPPORTED_MEMORY_OR_LIFETIME_CLAIM = "unsupported_memory_or_lifetime_claim"


class DoctorRejectionReason(str, Enum):
    """Stable fail-closed reason codes for contract and policy rejection."""

    FORGED_CID = "forged_cid"
    BODY_OR_SECRET = "body_or_secret"
    CYCLE = "cycle"
    UNBOUNDED_DATA = "unbounded_data"
    PARTIAL_PLAN = "partial_plan"
    OPEN_REQUIRED_FRONTIER = "open_required_frontier"
    SEMANTIC_AUTHORITY_KG = "semantic_authority_knowledge_graph"
    SEMANTIC_AUTHORITY_VECTOR = "semantic_authority_vector"
    SEMANTIC_AUTHORITY_EMBEDDING = "semantic_authority_embedding"
    SEMANTIC_AUTHORITY_TACTICIAN = "semantic_authority_tactician"
    SEMANTIC_AUTHORITY_HAMMER = "semantic_authority_hammer_candidate"
    SEMANTIC_AUTHORITY_CACHE_METADATA = "semantic_authority_cache_metadata"
    LLM_INVOCATION = "llm_invocation"
    REMOTE_MODEL_PROVIDER = "remote_model_provider"
    REPAIR_WITHOUT_ADMITTED_PLAN = "repair_without_admitted_plan"
    REPAIR_WITHOUT_LEASE = "repair_without_lease"
    REPAIR_WITHOUT_CHECKPOINT = "repair_without_checkpoint"
    REPAIR_WITHOUT_ROLLBACK = "repair_without_rollback"
    TCB_PATH = "trusted_computing_base_path"
    APPROVAL_REQUIRED = "approval_required"
    MODE_FORBIDS_OPERATION = "mode_forbids_operation"
    NONZERO_MODEL_INVOCATION = "nonzero_model_invocation"
    TARGET_CODE_IMPORT = "target_code_import"
    NETWORK_ACCESS = "network_access"
    MIXED_AUTHORITY_ROOTS = "mixed_authority_roots"
    STALE_ROOT = "stale_root"
    UNSUPPORTED_OPERATION = "unsupported_operation"


READ_ONLY_OPERATIONS: Final[frozenset[DoctorOperation]] = frozenset(
    {
        DoctorOperation.INSPECT,
        DoctorOperation.EXPLAIN,
        DoctorOperation.PLAN,
        DoctorOperation.REPLAY,
    }
)
WRITE_OPERATIONS: Final[frozenset[DoctorOperation]] = frozenset(
    {DoctorOperation.REPAIR}
)
DEFAULT_DOCTOR_MODE: Final[DoctorMode] = DoctorMode.REPORT_ONLY
DEFAULT_DOCTOR_OPERATION: Final[DoctorOperation] = DoctorOperation.INSPECT

ALLOWED_DOCTOR_MODES: Final[tuple[DoctorMode, ...]] = (
    DoctorMode.REPORT_ONLY,
    DoctorMode.PLAN,
    DoctorMode.SANDBOX_AUTO,
    DoctorMode.NARROW_AUTO,
)

ALL_DOCTOR_OPERATIONS: Final[tuple[DoctorOperation, ...]] = tuple(DoctorOperation)
ALL_REPAIR_DISPOSITIONS: Final[tuple[DoctorRepairDisposition, ...]] = tuple(
    DoctorRepairDisposition
)
ALL_APPROVAL_CLASSES: Final[tuple[DoctorApprovalClass, ...]] = tuple(DoctorApprovalClass)

# Semantic-authority claims that are always rejected in deterministic mode.
FORBIDDEN_SEMANTIC_AUTHORITY_FLAGS: Final[tuple[str, ...]] = (
    "knowledge_graph_semantic_authority",
    "vector_semantic_authority",
    "embedding_semantic_authority",
    "tactician_semantic_authority",
    "hammer_candidate_semantic_authority",
    "proof_cache_metadata_semantic_authority",
)

# Path prefixes / markers treated as doctor trusted computing base.
DOCTOR_TCB_PATH_MARKERS: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_",
    "ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_",
    "ipfs_accelerate_py/agent_supervisor/proof/",
    "ipfs_accelerate_py/agent_supervisor/multiformats_identity.py",
    "ipfs_accelerate_py/agent_supervisor/worktree_lifecycle.py",
    "ipfs_accelerate_py/agent_supervisor/validation/logic_repair_fixed_point.py",
    "ipfs_accelerate_py/agent_supervisor/proof/formal_verification_",
    "ipfs_accelerate_py/agent_supervisor/proof/kernel_verification.py",
    "ipfs_accelerate_py/agent_supervisor/control/",
    "ipfs_accelerate_py/agent_supervisor/merge/",
)

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source",
        "source_body",
        "source_text",
        "contents",
        "content",
        "snippet",
        "code",
        "file_text",
        "raw_ast",
        "ast_body",
        "secret",
        "secrets",
        "password",
        "token",
        "api_key",
        "private_key",
        "credential",
        "authorization_header",
    }
)

_PRIVATE_FIELD_MARKERS: Final[tuple[str, ...]] = (
    "secret",
    "password",
    "token",
    "api_key",
    "private_key",
    "credential",
    "authorization",
    "cookie",
    "session",
)

# Authority root field order for DoctorAuthorityRoots.
AUTHORITY_ROOT_FIELDS: Final[tuple[str, ...]] = (
    "repository_id",
    "forest_id",
    "tree_id",
    "overlay_id",
    "file_root_id",
    "ast_root_id",
    "graph_id",
    "corpus_id",
    "index_id",
    "model_id",
    "cache_id",
    "operator_registry_id",
    "translator_id",
    "solver_id",
    "kernel_id",
    "toolchain_id",
    "policy_id",
    "sandbox_id",
    "environment_id",
    "lease_id",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DeterministicDoctorError(ContractValidationError):
    """Base class for deterministic-doctor schema failures."""


class DeterministicDoctorBoundsError(DeterministicDoctorError):
    """A record attempted to exceed its declared compactness bounds."""


class ForgedDeterministicDoctorIdentityError(DeterministicDoctorError):
    """A stored content identity did not match the canonical preimage."""


class DeterministicDoctorAuthorityError(DeterministicDoctorError):
    """Authority roots, paths, dispositions, or mode bindings failed closed."""


class DeterministicDoctorSafetyError(DeterministicDoctorError):
    """A deterministic-mode safety invariant was violated."""


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise DeterministicDoctorError(f"{field_name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise DeterministicDoctorError(f"{field_name} is required")
    if len(text.encode("utf-8")) > limit:
        raise DeterministicDoctorBoundsError(f"{field_name} exceeds its byte bound")
    if "\0" in text:
        raise DeterministicDoctorError(f"{field_name} must not contain NUL")
    return text


def _identifier(value: Any, field_name: str) -> str:
    value = _text(value, field_name, required=True)
    if any(char.isspace() for char in value):
        raise DeterministicDoctorError(
            f"{field_name} must be an opaque compact identifier"
        )
    return value


def _optional_identifier(value: Any, field_name: str) -> str:
    if value in (None, ""):
        return ""
    return _identifier(value, field_name)


def _bounded_int(
    value: Any,
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_SPAN_OFFSET,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeterministicDoctorError(f"{field_name} must be a finite integer")
    if value < minimum or value > maximum:
        raise DeterministicDoctorBoundsError(
            f"{field_name} is outside the supported bound"
        )
    return value


def _nonneg_int(value: Any, field_name: str) -> int:
    return _bounded_int(value, field_name, minimum=0)


def _positive_int(value: Any, field_name: str, *, maximum: int = MAX_SPAN_OFFSET) -> int:
    return _bounded_int(value, field_name, minimum=1, maximum=maximum)


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise DeterministicDoctorError(f"{field_name} must be a boolean")
    return value


def _path(value: Any, field_name: str) -> str:
    path = _text(value, field_name, required=True, limit=MAX_PATH_BYTES)
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or path in {".", ""}:
        raise DeterministicDoctorAuthorityError(
            f"{field_name} must be a relative repository path"
        )
    return candidate.as_posix()


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise DeterministicDoctorError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _ids(
    values: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCE_COUNT,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str) or not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise DeterministicDoctorError(f"{field_name} must be a sequence of identifiers")
    else:
        raw = values
    if len(raw) > limit:
        raise DeterministicDoctorBoundsError(f"{field_name} exceeds its item bound")
    if preserve_order:
        result: list[str] = []
        seen: set[str] = set()
        for item in raw:
            ident = _identifier(item, field_name)
            if ident not in seen:
                seen.add(ident)
                result.append(ident)
        out = tuple(result)
    else:
        out = tuple(sorted({_identifier(value, field_name) for value in raw}))
    if required and not out:
        raise DeterministicDoctorError(f"{field_name} must not be empty")
    return out


def _paths(
    values: Any,
    field_name: str,
    *,
    limit: int = MAX_REFERENCE_COUNT,
    required: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str) or not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise DeterministicDoctorError(f"{field_name} must be a sequence of paths")
    else:
        raw = values
    if len(raw) > limit:
        raise DeterministicDoctorBoundsError(f"{field_name} exceeds its item bound")
    out = tuple(sorted({_path(value, field_name) for value in raw}))
    if required and not out:
        raise DeterministicDoctorError(f"{field_name} must not be empty")
    return out


def _is_forbidden_payload_key(key: str) -> bool:
    """Return whether a mapping key smuggles bodies or private material."""

    normalized = key.lower().replace("-", "_").strip()
    if normalized in _BODY_MARKERS:
        return True
    # Exact private markers, or suffix form (e.g. ``auth_token``), but not
    # legitimate boolean flags such as ``secrets_inherited``.
    for marker in _PRIVATE_FIELD_MARKERS:
        if normalized == marker or normalized.endswith("_" + marker):
            return True
    return False


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    """Reject source bodies and secrets even when smuggled through a mapping."""

    if isinstance(value, float):
        raise DeterministicDoctorError(
            f"{field_name} may not contain floating-point values"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DeterministicDoctorError(f"{field_name} has a non-string key")
            if _is_forbidden_payload_key(key):
                raise DeterministicDoctorError(
                    f"{field_name} may not contain source bodies or secrets"
                )
            _assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise DeterministicDoctorError(f"{field_name} may not contain binary bodies")


def _bounded(record: CanonicalContract, name: str) -> None:
    _assert_body_free(record.to_dict(), name)
    if len(canonical_json_bytes(record.to_dict())) > MAX_RECORD_BYTES:
        raise DeterministicDoctorBoundsError(f"{name} exceeds its serialized byte bound")


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    supplied = payload.get("content_id", payload.get("cid", ""))
    if supplied not in (None, ""):
        if not isinstance(supplied, str) or supplied != record.content_id:
            raise ForgedDeterministicDoctorIdentityError(
                "stored content identity does not match the canonical record"
            )


def _decode_fields(
    payload: Mapping[str, Any], schema: str, fields: Sequence[str], name: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise DeterministicDoctorError(f"{name} has an unsupported schema")
    if payload.get("contract_version") not in (None, DETERMINISTIC_DOCTOR_VERSION):
        raise DeterministicDoctorError(f"{name} has an unsupported contract version")
    # Body/secret keys fail closed before the generic unsupported-field path so
    # callers observe a stable rejection reason for smuggled material.
    _assert_body_free(payload, name)
    allowed = set(fields) | {"schema", "contract_version", "content_id", "cid"}
    if set(payload).difference(allowed):
        raise DeterministicDoctorError(f"{name} contains unsupported fields")
    try:
        return {field_name: payload[field_name] for field_name in fields if field_name in payload}
    except KeyError as exc:
        raise DeterministicDoctorError(f"{name} omits a required field") from exc


def _decode_nested(value: Any, cls: type, field_name: str) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls.from_dict(value)
    raise DeterministicDoctorError(f"{field_name} must be a {cls.__name__} payload")


def _decode_sequence(
    values: Any,
    cls: type,
    field_name: str,
    *,
    limit: int,
    required: bool = False,
) -> tuple[Any, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str) or not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise DeterministicDoctorError(f"{field_name} must be a sequence")
    else:
        raw = values
    if len(raw) > limit:
        raise DeterministicDoctorBoundsError(f"{field_name} exceeds its item bound")
    out = tuple(_decode_nested(item, cls, field_name) for item in raw)
    if required and not out:
        raise DeterministicDoctorError(f"{field_name} must not be empty")
    return out


def _detect_cycle(nodes: Mapping[str, Sequence[str]]) -> bool:
    """Return True if ``nodes`` (node -> dependency ids) contains a cycle."""

    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dep in nodes.get(node, ()):
            if dep in nodes and dfs(dep):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(dfs(node) for node in nodes)


def is_doctor_tcb_path(path: str) -> bool:
    """Return whether ``path`` falls under the doctor trusted computing base."""

    normalized = PurePosixPath(str(path or "").replace("\\", "/")).as_posix()
    if (
        not normalized
        or normalized.startswith("/")
        or ".." in PurePosixPath(normalized).parts
    ):
        return False
    for marker in DOCTOR_TCB_PATH_MARKERS:
        marker_norm = marker.rstrip("/")
        if normalized == marker_norm or normalized.startswith(marker_norm + "/"):
            return True
        if marker.endswith("_") and normalized.startswith(marker):
            return True
        if marker.endswith(".py") and normalized == marker:
            return True
    return False


def operation_is_read_only(operation: DoctorOperation | str) -> bool:
    """Return whether the doctor operation is read-only / report-only safe."""

    op = _enum(operation, DoctorOperation, "operation")
    assert isinstance(op, DoctorOperation)
    return op.is_read_only


def default_doctor_mode() -> DoctorMode:
    """Return the fail-closed default mode (report-only)."""

    return DEFAULT_DOCTOR_MODE


# ---------------------------------------------------------------------------
# Core records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorAuthorityRoots(CanonicalContract):
    """Exact roots whose drift invalidates every doctor record.

    Binds forest/tree/overlay/file/AST/graph/corpus/index/model/cache/operator/
    translator/solver/kernel/toolchain/policy/sandbox/environment/lease.
    """

    SCHEMA: ClassVar[str] = DOCTOR_AUTHORITY_ROOTS_SCHEMA

    repository_id: str
    forest_id: str
    tree_id: str
    overlay_id: str
    file_root_id: str
    ast_root_id: str
    graph_id: str
    corpus_id: str
    index_id: str
    model_id: str
    cache_id: str
    operator_registry_id: str
    translator_id: str
    solver_id: str
    kernel_id: str
    toolchain_id: str
    policy_id: str
    sandbox_id: str
    environment_id: str
    lease_id: str = ""

    def __post_init__(self) -> None:
        for name in AUTHORITY_ROOT_FIELDS:
            if name == "lease_id":
                object.__setattr__(
                    self, name, _optional_identifier(getattr(self, name), name)
                )
            else:
                object.__setattr__(self, name, _identifier(getattr(self, name), name))
        _bounded(self, "doctor authority roots")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            **{name: getattr(self, name) for name in AUTHORITY_ROOT_FIELDS},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorAuthorityRoots":
        values = _decode_fields(payload, cls.SCHEMA, AUTHORITY_ROOT_FIELDS, "authority roots")
        value = cls(**values)
        _verify_identity(payload, value)
        return value

    def require_lease(self) -> None:
        if not self.lease_id:
            raise DeterministicDoctorAuthorityError(
                "repair requires an existing writer lease root"
            )


def _roots(value: Any) -> DoctorAuthorityRoots:
    if isinstance(value, DoctorAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        if value.get("schema") == DOCTOR_AUTHORITY_ROOTS_SCHEMA:
            return DoctorAuthorityRoots.from_dict(value)
        payload = {
            key: value[key] for key in AUTHORITY_ROOT_FIELDS if key in value
        }
        return DoctorAuthorityRoots(**payload)
    raise DeterministicDoctorError("roots must be DoctorAuthorityRoots or a mapping")


@dataclass(frozen=True)
class DoctorResourceBounds(CanonicalContract):
    """Integer resource bounds bound into plans and run receipts."""

    SCHEMA: ClassVar[str] = DOCTOR_RESOURCE_BOUNDS_SCHEMA

    max_findings: int = 256
    max_candidates_per_finding: int = 64
    max_graph_nodes_per_query: int = 2048
    max_proof_routes_per_goal: int = 32
    max_operators_per_finding: int = 32
    max_plan_steps: int = 256
    max_fixed_point_iterations: int = 8
    max_changed_files: int = 128
    max_changed_bytes: int = 1_048_576
    max_processes: int = 8
    max_wall_time_seconds: int = 3600
    max_cpu_time_seconds: int = 1800
    max_memory_bytes: int = 4_294_967_296

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name == "SCHEMA":
                continue
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name, maximum=2**63 - 1),
            )
        _bounded(self, "doctor resource bounds")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "SCHEMA"
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorResourceBounds":
        fields = tuple(name for name in cls.__dataclass_fields__ if name != "SCHEMA")
        values = _decode_fields(payload, cls.SCHEMA, fields, "resource bounds")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorEvidenceSnapshot(CanonicalContract):
    """Immutable frozen evidence roots for one deterministic-doctor run.

    Binds repository forest/tree/overlay and derived indexes plus
    completeness, unsupported frontiers, and clean-rebuild equivalence.
    Target code is referenced by CID only — never embedded as a body.
    """

    SCHEMA: ClassVar[str] = DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA

    roots: DoctorAuthorityRoots
    snapshot_id: str
    file_blob_cids: tuple[str, ...]
    completeness: str = "complete"
    unsupported_frontiers: tuple[str, ...] = ()
    tombstone_refs: tuple[str, ...] = ()
    dependency_links: tuple[str, ...] = ()
    clean_rebuild_equivalence_receipt_id: str = ""
    parser_id: str = ""
    vector_root_id: str = ""
    embedding_config_id: str = ""
    impact_index_id: str = ""
    value_index_id: str = ""
    evidence_graph_id: str = ""
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "snapshot_id", _identifier(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self,
            "file_blob_cids",
            _ids(self.file_blob_cids, "file_blob_cids", limit=MAX_REFERENCE_COUNT),
        )
        object.__setattr__(
            self,
            "completeness",
            _text(self.completeness, "completeness", required=True, limit=64),
        )
        if self.completeness not in {
            "complete",
            "partial_with_frontier",
            "abstained",
        }:
            raise DeterministicDoctorError(
                "completeness must be complete, partial_with_frontier, or abstained"
            )
        object.__setattr__(
            self,
            "unsupported_frontiers",
            _ids(
                self.unsupported_frontiers,
                "unsupported_frontiers",
                limit=MAX_FRONTIER_COUNT,
            ),
        )
        object.__setattr__(
            self, "tombstone_refs", _ids(self.tombstone_refs, "tombstone_refs")
        )
        object.__setattr__(
            self, "dependency_links", _ids(self.dependency_links, "dependency_links")
        )
        object.__setattr__(
            self,
            "clean_rebuild_equivalence_receipt_id",
            _optional_identifier(
                self.clean_rebuild_equivalence_receipt_id,
                "clean_rebuild_equivalence_receipt_id",
            ),
        )
        for name in (
            "parser_id",
            "vector_root_id",
            "embedding_config_id",
            "impact_index_id",
            "value_index_id",
            "evidence_graph_id",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        if self.completeness == "complete" and self.unsupported_frontiers:
            # Completeness claim is about required frontiers; unsupported
            # optional analyses may remain listed without opening required ones.
            pass
        _bounded(self, "doctor evidence snapshot")

    @property
    def has_open_required_frontier(self) -> bool:
        return self.completeness == "partial_with_frontier" or any(
            frontier.startswith("frontier:required:")
            for frontier in self.unsupported_frontiers
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "roots": self.roots.to_dict(),
            "snapshot_id": self.snapshot_id,
            "file_blob_cids": list(self.file_blob_cids),
            "completeness": self.completeness,
            "unsupported_frontiers": list(self.unsupported_frontiers),
            "tombstone_refs": list(self.tombstone_refs),
            "dependency_links": list(self.dependency_links),
            "clean_rebuild_equivalence_receipt_id": self.clean_rebuild_equivalence_receipt_id,
            "parser_id": self.parser_id,
            "vector_root_id": self.vector_root_id,
            "embedding_config_id": self.embedding_config_id,
            "impact_index_id": self.impact_index_id,
            "value_index_id": self.value_index_id,
            "evidence_graph_id": self.evidence_graph_id,
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorEvidenceSnapshot":
        fields = (
            "roots",
            "snapshot_id",
            "file_blob_cids",
            "completeness",
            "unsupported_frontiers",
            "tombstone_refs",
            "dependency_links",
            "clean_rebuild_equivalence_receipt_id",
            "parser_id",
            "vector_root_id",
            "embedding_config_id",
            "impact_index_id",
            "value_index_id",
            "evidence_graph_id",
            "invalidation_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "doctor evidence snapshot")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class DeterministicDoctorFinding(CanonicalContract):
    """One typed diagnostic finding with observed facts ≠ expected behavior."""

    SCHEMA: ClassVar[str] = DETERMINISTIC_DOCTOR_FINDING_SCHEMA

    roots: DoctorAuthorityRoots
    finding_id: str
    snapshot_id: str
    disposition: DoctorRepairDisposition
    observed_fact_refs: tuple[str, ...]
    expected_behavior_refs: tuple[str, ...]
    evidence_role: DoctorEvidenceRole = DoctorEvidenceRole.OBSERVED_FACT
    diagnostic_ref: str = ""
    trace_ref: str = ""
    change_ref: str = ""
    finding_kind: str = "contract_mismatch"
    reason_codes: tuple[str, ...] = ()
    affected_symbol_refs: tuple[str, ...] = ()
    consumer_refs: tuple[str, ...] = ()
    scc_refs: tuple[str, ...] = ()
    open_frontier_refs: tuple[str, ...] = ()
    goal_refs: tuple[str, ...] = ()
    premise_refs: tuple[str, ...] = ()
    candidate_query_refs: tuple[str, ...] = ()
    approval_classes: tuple[str, ...] = ()
    semantic_authority: bool = False
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "finding_id", _identifier(self.finding_id, "finding_id")
        )
        object.__setattr__(
            self, "snapshot_id", _identifier(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorRepairDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "observed_fact_refs",
            _ids(self.observed_fact_refs, "observed_fact_refs"),
        )
        object.__setattr__(
            self,
            "expected_behavior_refs",
            _ids(self.expected_behavior_refs, "expected_behavior_refs"),
        )
        # Observed facts must never be listed as expected behavior authority.
        overlap = set(self.observed_fact_refs) & set(self.expected_behavior_refs)
        if overlap:
            raise DeterministicDoctorAuthorityError(
                "observed facts must remain separate from expected behavior refs"
            )
        object.__setattr__(
            self,
            "evidence_role",
            _enum(self.evidence_role, DoctorEvidenceRole, "evidence_role"),
        )
        if self.evidence_role is DoctorEvidenceRole.NOMINATION:
            if self.disposition is DoctorRepairDisposition.SUPPORTED:
                raise DeterministicDoctorAuthorityError(
                    "nomination evidence cannot grant supported/write disposition"
                )
        if self.evidence_role is DoctorEvidenceRole.OBSERVED_FACT:
            if self.disposition is DoctorRepairDisposition.SUPPORTED and not self.expected_behavior_refs:
                raise DeterministicDoctorAuthorityError(
                    "supported findings require independent expected-behavior authority"
                )
        for name in ("diagnostic_ref", "trace_ref", "change_ref"):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "finding_kind",
            _text(self.finding_kind, "finding_kind", required=True, limit=128),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self,
            "affected_symbol_refs",
            _ids(self.affected_symbol_refs, "affected_symbol_refs"),
        )
        object.__setattr__(
            self, "consumer_refs", _ids(self.consumer_refs, "consumer_refs")
        )
        object.__setattr__(self, "scc_refs", _ids(self.scc_refs, "scc_refs"))
        object.__setattr__(
            self,
            "open_frontier_refs",
            _ids(self.open_frontier_refs, "open_frontier_refs", limit=MAX_FRONTIER_COUNT),
        )
        object.__setattr__(self, "goal_refs", _ids(self.goal_refs, "goal_refs"))
        object.__setattr__(
            self, "premise_refs", _ids(self.premise_refs, "premise_refs")
        )
        object.__setattr__(
            self,
            "candidate_query_refs",
            _ids(self.candidate_query_refs, "candidate_query_refs"),
        )
        approval = _ids(self.approval_classes, "approval_classes")
        for item in approval:
            try:
                DoctorApprovalClass(item)
            except ValueError as exc:
                raise DeterministicDoctorError(
                    "approval_classes must use closed DoctorApprovalClass values"
                ) from exc
        object.__setattr__(self, "approval_classes", approval)
        if self.semantic_authority is not False:
            raise DeterministicDoctorSafetyError(
                "findings cannot claim semantic_authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        if (
            self.disposition is DoctorRepairDisposition.APPROVAL_REQUIRED
            and not self.approval_classes
        ):
            raise DeterministicDoctorError(
                "approval_required findings must name at least one approval class"
            )
        if (
            self.disposition is DoctorRepairDisposition.SUPPORTED
            and self.open_frontier_refs
        ):
            required_open = [
                ref
                for ref in self.open_frontier_refs
                if ref.startswith("frontier:required:")
            ]
            if required_open:
                raise DeterministicDoctorAuthorityError(
                    "supported findings cannot leave required frontiers open"
                )
        _bounded(self, "deterministic doctor finding")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "roots": self.roots.to_dict(),
            "finding_id": self.finding_id,
            "snapshot_id": self.snapshot_id,
            "disposition": self.disposition.value,
            "observed_fact_refs": list(self.observed_fact_refs),
            "expected_behavior_refs": list(self.expected_behavior_refs),
            "evidence_role": self.evidence_role.value,
            "diagnostic_ref": self.diagnostic_ref,
            "trace_ref": self.trace_ref,
            "change_ref": self.change_ref,
            "finding_kind": self.finding_kind,
            "reason_codes": list(self.reason_codes),
            "affected_symbol_refs": list(self.affected_symbol_refs),
            "consumer_refs": list(self.consumer_refs),
            "scc_refs": list(self.scc_refs),
            "open_frontier_refs": list(self.open_frontier_refs),
            "goal_refs": list(self.goal_refs),
            "premise_refs": list(self.premise_refs),
            "candidate_query_refs": list(self.candidate_query_refs),
            "approval_classes": list(self.approval_classes),
            "semantic_authority": False,
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeterministicDoctorFinding":
        fields = (
            "roots",
            "finding_id",
            "snapshot_id",
            "disposition",
            "observed_fact_refs",
            "expected_behavior_refs",
            "evidence_role",
            "diagnostic_ref",
            "trace_ref",
            "change_ref",
            "finding_kind",
            "reason_codes",
            "affected_symbol_refs",
            "consumer_refs",
            "scc_refs",
            "open_frontier_refs",
            "goal_refs",
            "premise_refs",
            "candidate_query_refs",
            "approval_classes",
            "semantic_authority",
            "invalidation_refs",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "deterministic doctor finding"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorRepairOperatorSpec(CanonicalContract):
    """Closed typed repair operator; nomination of a kind is not write authority."""

    SCHEMA: ClassVar[str] = DOCTOR_REPAIR_OPERATOR_SPEC_SCHEMA

    roots: DoctorAuthorityRoots
    operator_id: str
    kind: DoctorOperatorKind
    supported_languages: tuple[str, ...]
    precondition_refs: tuple[str, ...]
    postcondition_refs: tuple[str, ...]
    frame_condition_refs: tuple[str, ...] = ()
    proof_template_refs: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    value_source_refs: tuple[str, ...] = ()
    placement_constraints: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    renderer_id: str = ""
    idempotent: bool = True
    inverse_or_compensation_ref: str = ""
    resource_bound_ref: str = ""
    approval_exclusions: tuple[str, ...] = ()
    unsupported_frontier_exclusions: tuple[str, ...] = ()
    semantic_authority: bool = False
    grants_write_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "operator_id", _identifier(self.operator_id, "operator_id")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, DoctorOperatorKind, "kind")
        )
        object.__setattr__(
            self,
            "supported_languages",
            _ids(self.supported_languages, "supported_languages", required=True),
        )
        object.__setattr__(
            self,
            "precondition_refs",
            _ids(self.precondition_refs, "precondition_refs", required=True),
        )
        object.__setattr__(
            self,
            "postcondition_refs",
            _ids(self.postcondition_refs, "postcondition_refs", required=True),
        )
        object.__setattr__(
            self,
            "frame_condition_refs",
            _ids(self.frame_condition_refs, "frame_condition_refs"),
        )
        object.__setattr__(
            self,
            "proof_template_refs",
            _ids(self.proof_template_refs, "proof_template_refs"),
        )
        object.__setattr__(self, "read_paths", _paths(self.read_paths, "read_paths"))
        object.__setattr__(self, "write_paths", _paths(self.write_paths, "write_paths"))
        object.__setattr__(
            self, "value_source_refs", _ids(self.value_source_refs, "value_source_refs")
        )
        object.__setattr__(
            self,
            "placement_constraints",
            _ids(self.placement_constraints, "placement_constraints"),
        )
        object.__setattr__(
            self, "forbidden_paths", _paths(self.forbidden_paths, "forbidden_paths")
        )
        # Spec paths are constraints, not granted write authority.
        if self.grants_write_authority:
            raise DeterministicDoctorAuthorityError(
                "operator specs cannot grant write authority; only admitted plans may"
            )
        object.__setattr__(self, "grants_write_authority", False)
        if self.semantic_authority is not False:
            raise DeterministicDoctorSafetyError(
                "operator specs cannot claim semantic_authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self, "renderer_id", _optional_identifier(self.renderer_id, "renderer_id")
        )
        object.__setattr__(self, "idempotent", _bool(self.idempotent, "idempotent"))
        object.__setattr__(
            self,
            "inverse_or_compensation_ref",
            _optional_identifier(
                self.inverse_or_compensation_ref, "inverse_or_compensation_ref"
            ),
        )
        object.__setattr__(
            self,
            "resource_bound_ref",
            _optional_identifier(self.resource_bound_ref, "resource_bound_ref"),
        )
        object.__setattr__(
            self,
            "approval_exclusions",
            _ids(self.approval_exclusions, "approval_exclusions"),
        )
        object.__setattr__(
            self,
            "unsupported_frontier_exclusions",
            _ids(
                self.unsupported_frontier_exclusions,
                "unsupported_frontier_exclusions",
            ),
        )
        for path in self.write_paths:
            if is_doctor_tcb_path(path):
                raise DeterministicDoctorAuthorityError(
                    "operator write_paths cannot target doctor trusted computing base paths"
                )
            if path in set(self.forbidden_paths):
                raise DeterministicDoctorAuthorityError(
                    "operator write_paths cannot include forbidden paths"
                )
        _bounded(self, "doctor repair operator spec")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "roots": self.roots.to_dict(),
            "operator_id": self.operator_id,
            "kind": self.kind.value,
            "supported_languages": list(self.supported_languages),
            "precondition_refs": list(self.precondition_refs),
            "postcondition_refs": list(self.postcondition_refs),
            "frame_condition_refs": list(self.frame_condition_refs),
            "proof_template_refs": list(self.proof_template_refs),
            "read_paths": list(self.read_paths),
            "write_paths": list(self.write_paths),
            "value_source_refs": list(self.value_source_refs),
            "placement_constraints": list(self.placement_constraints),
            "forbidden_paths": list(self.forbidden_paths),
            "renderer_id": self.renderer_id,
            "idempotent": self.idempotent,
            "inverse_or_compensation_ref": self.inverse_or_compensation_ref,
            "resource_bound_ref": self.resource_bound_ref,
            "approval_exclusions": list(self.approval_exclusions),
            "unsupported_frontier_exclusions": list(self.unsupported_frontier_exclusions),
            "semantic_authority": False,
            "grants_write_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorRepairOperatorSpec":
        fields = (
            "roots",
            "operator_id",
            "kind",
            "supported_languages",
            "precondition_refs",
            "postcondition_refs",
            "frame_condition_refs",
            "proof_template_refs",
            "read_paths",
            "write_paths",
            "value_source_refs",
            "placement_constraints",
            "forbidden_paths",
            "renderer_id",
            "idempotent",
            "inverse_or_compensation_ref",
            "resource_bound_ref",
            "approval_exclusions",
            "unsupported_frontier_exclusions",
            "semantic_authority",
            "grants_write_authority",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "doctor repair operator spec"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorConsumerDisposition(CanonicalContract):
    """Exactly one disposition for one resolved consumer under a plan."""

    SCHEMA: ClassVar[str] = DOCTOR_CONSUMER_DISPOSITION_SCHEMA

    roots: DoctorAuthorityRoots
    consumer_id: str
    disposition: DoctorRepairDisposition
    reason_codes: tuple[str, ...] = ()
    obligation_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "consumer_id", _identifier(self.consumer_id, "consumer_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorRepairDisposition, "disposition"),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self,
            "obligation_ref",
            _optional_identifier(self.obligation_ref, "obligation_ref"),
        )
        _bounded(self, "doctor consumer disposition")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "roots": self.roots.to_dict(),
            "consumer_id": self.consumer_id,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "obligation_ref": self.obligation_ref,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorConsumerDisposition":
        fields = (
            "roots",
            "consumer_id",
            "disposition",
            "reason_codes",
            "obligation_ref",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "doctor consumer disposition"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorEditSite(CanonicalContract):
    """Exact edit locus bound by path and before-hash (no body)."""

    SCHEMA: ClassVar[str] = DOCTOR_EDIT_SITE_SCHEMA

    path: str
    before_hash: str
    span_start: int = 0
    span_end: int = 0
    artifact_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, "path"))
        object.__setattr__(
            self, "before_hash", _identifier(self.before_hash, "before_hash")
        )
        start = _nonneg_int(self.span_start, "span_start")
        end = _nonneg_int(self.span_end, "span_end")
        if end < start:
            raise DeterministicDoctorError("span_end must be >= span_start")
        object.__setattr__(self, "span_start", start)
        object.__setattr__(self, "span_end", end)
        object.__setattr__(
            self, "artifact_id", _optional_identifier(self.artifact_id, "artifact_id")
        )
        if is_doctor_tcb_path(self.path):
            raise DeterministicDoctorAuthorityError(
                "edit sites cannot target doctor trusted computing base paths"
            )
        _bounded(self, "doctor edit site")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "path": self.path,
            "before_hash": self.before_hash,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorEditSite":
        fields = ("path", "before_hash", "span_start", "span_end", "artifact_id")
        values = _decode_fields(payload, cls.SCHEMA, fields, "doctor edit site")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorPlanStep(CanonicalContract):
    """One ordered plan step; dependency graphs must be acyclic."""

    SCHEMA: ClassVar[str] = DOCTOR_PLAN_STEP_SCHEMA

    step_id: str
    kind: str
    operator_id: str = ""
    dependency_step_ids: tuple[str, ...] = ()
    consumer_ids: tuple[str, ...] = ()
    edit_site_refs: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(
            self, "kind", _text(self.kind, "kind", required=True, limit=64)
        )
        object.__setattr__(
            self, "operator_id", _optional_identifier(self.operator_id, "operator_id")
        )
        object.__setattr__(
            self,
            "dependency_step_ids",
            _ids(self.dependency_step_ids, "dependency_step_ids", preserve_order=True),
        )
        object.__setattr__(
            self, "consumer_ids", _ids(self.consumer_ids, "consumer_ids")
        )
        object.__setattr__(
            self, "edit_site_refs", _ids(self.edit_site_refs, "edit_site_refs")
        )
        object.__setattr__(
            self, "validation_refs", _ids(self.validation_refs, "validation_refs")
        )
        object.__setattr__(
            self, "write_paths", _paths(self.write_paths, "write_paths")
        )
        if self.step_id in set(self.dependency_step_ids):
            raise DeterministicDoctorError("plan step cannot depend on itself")
        _bounded(self, "doctor plan step")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "step_id": self.step_id,
            "kind": self.kind,
            "operator_id": self.operator_id,
            "dependency_step_ids": list(self.dependency_step_ids),
            "consumer_ids": list(self.consumer_ids),
            "edit_site_refs": list(self.edit_site_refs),
            "validation_refs": list(self.validation_refs),
            "write_paths": list(self.write_paths),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorPlanStep":
        fields = (
            "step_id",
            "kind",
            "operator_id",
            "dependency_step_ids",
            "consumer_ids",
            "edit_site_refs",
            "validation_refs",
            "write_paths",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "doctor plan step")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class DeterministicDoctorPlan(CanonicalContract):
    """Complete uniquely admitted (or abstained) deterministic-doctor plan.

    Partial plans, open required frontiers, cycles, missing rollback/lease/
    checkpoint bindings for admitted write plans, and nonzero model routes
    are rejected at construction.
    """

    SCHEMA: ClassVar[str] = DETERMINISTIC_DOCTOR_PLAN_SCHEMA

    roots: DoctorAuthorityRoots
    plan_id: str
    snapshot_id: str
    finding_ids: tuple[str, ...]
    disposition: DoctorPlanDisposition
    consumer_dispositions: tuple[DoctorConsumerDisposition, ...]
    impact_closure_id: str
    steps: tuple[DoctorPlanStep, ...] = ()
    edit_sites: tuple[DoctorEditSite, ...] = ()
    operator_ids: tuple[str, ...] = ()
    target_ref: str = ""
    value_source_ref: str = ""
    placement_ref: str = ""
    selected_operator_id: str = ""
    premise_refs: tuple[str, ...] = ()
    goal_refs: tuple[str, ...] = ()
    proof_route_refs: tuple[str, ...] = ()
    candidate_refs: tuple[str, ...] = ()
    exclusion_refs: tuple[str, ...] = ()
    open_required_frontiers: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    scc_refs: tuple[str, ...] = ()
    tactician_plan_ref: str = ""
    permitted_read_paths: tuple[str, ...] = ()
    permitted_write_paths: tuple[str, ...] = ()
    lease_id: str = ""
    checkpoint_ref: str = ""
    rollback_ref: str = ""
    proof_refs: tuple[str, ...] = ()
    resource_bounds: DoctorResourceBounds | None = None
    no_model_invariant: bool = True
    llm_router_enabled: bool = False
    model_invocation_count: int = 0
    semantic_authority_flags: Mapping[str, bool] | None = None
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "snapshot_id", _identifier(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self,
            "finding_ids",
            _ids(self.finding_ids, "finding_ids", required=True, limit=MAX_FINDING_COUNT),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorPlanDisposition, "disposition"),
        )
        consumers = _decode_sequence(
            self.consumer_dispositions,
            DoctorConsumerDisposition,
            "consumer_dispositions",
            limit=MAX_CONSUMER_COUNT,
            required=True,
        )
        object.__setattr__(self, "consumer_dispositions", consumers)
        for consumer in self.consumer_dispositions:
            if consumer.roots != self.roots:
                raise DeterministicDoctorAuthorityError(
                    "all consumer dispositions must bind the plan authority roots"
                )
        consumer_ids = [item.consumer_id for item in self.consumer_dispositions]
        if len(set(consumer_ids)) != len(consumer_ids):
            raise DeterministicDoctorError(
                "plan requires exactly one disposition per resolved consumer"
            )
        object.__setattr__(
            self,
            "impact_closure_id",
            _identifier(self.impact_closure_id, "impact_closure_id"),
        )
        steps = _decode_sequence(
            self.steps, DoctorPlanStep, "steps", limit=MAX_STEP_COUNT
        )
        object.__setattr__(self, "steps", steps)
        step_ids = [step.step_id for step in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise DeterministicDoctorError("plan steps must have unique step_ids")
        dep_graph = {
            step.step_id: step.dependency_step_ids for step in self.steps
        }
        for step in self.steps:
            missing = set(step.dependency_step_ids) - set(step_ids)
            if missing:
                raise DeterministicDoctorError(
                    "plan step dependencies must reference known steps"
                )
        if _detect_cycle(dep_graph):
            raise DeterministicDoctorError(
                "plan step dependency graph must be acyclic"
            )
        sites = _decode_sequence(
            self.edit_sites, DoctorEditSite, "edit_sites", limit=MAX_REFERENCE_COUNT
        )
        object.__setattr__(self, "edit_sites", sites)
        object.__setattr__(
            self,
            "operator_ids",
            _ids(self.operator_ids, "operator_ids", limit=MAX_OPERATOR_COUNT),
        )
        for name in (
            "target_ref",
            "value_source_ref",
            "placement_ref",
            "selected_operator_id",
            "tactician_plan_ref",
            "lease_id",
            "checkpoint_ref",
            "rollback_ref",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        for name in (
            "premise_refs",
            "goal_refs",
            "proof_route_refs",
            "candidate_refs",
            "exclusion_refs",
            "open_required_frontiers",
            "validation_refs",
            "scc_refs",
            "proof_refs",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self,
            "permitted_read_paths",
            _paths(self.permitted_read_paths, "permitted_read_paths"),
        )
        object.__setattr__(
            self,
            "permitted_write_paths",
            _paths(self.permitted_write_paths, "permitted_write_paths"),
        )
        for path in self.permitted_write_paths:
            if is_doctor_tcb_path(path):
                raise DeterministicDoctorAuthorityError(
                    "plans cannot grant write authority over doctor trusted-base paths"
                )
        if self.resource_bounds is None:
            object.__setattr__(self, "resource_bounds", DoctorResourceBounds())
        elif isinstance(self.resource_bounds, DoctorResourceBounds):
            pass
        elif isinstance(self.resource_bounds, Mapping):
            object.__setattr__(
                self,
                "resource_bounds",
                DoctorResourceBounds.from_dict(self.resource_bounds)
                if self.resource_bounds.get("schema") == DoctorResourceBounds.SCHEMA
                else DoctorResourceBounds(**{
                    k: v
                    for k, v in self.resource_bounds.items()
                    if k in DoctorResourceBounds.__dataclass_fields__
                    and k != "SCHEMA"
                }),
            )
        else:
            raise DeterministicDoctorError("resource_bounds must be DoctorResourceBounds")
        object.__setattr__(
            self, "no_model_invariant", _bool(self.no_model_invariant, "no_model_invariant")
        )
        object.__setattr__(
            self, "llm_router_enabled", _bool(self.llm_router_enabled, "llm_router_enabled")
        )
        object.__setattr__(
            self,
            "model_invocation_count",
            _nonneg_int(self.model_invocation_count, "model_invocation_count"),
        )
        if not self.no_model_invariant:
            raise DeterministicDoctorSafetyError(
                "deterministic doctor plans require no_model_invariant=true"
            )
        if self.llm_router_enabled:
            raise DeterministicDoctorSafetyError(
                "deterministic doctor plans forbid llm_router_enabled"
            )
        if self.model_invocation_count != 0:
            raise DeterministicDoctorSafetyError(
                "deterministic doctor plans require zero model invocations"
            )
        flags = dict(self.semantic_authority_flags or {})
        for key in FORBIDDEN_SEMANTIC_AUTHORITY_FLAGS:
            if flags.get(key) is True:
                raise DeterministicDoctorSafetyError(
                    f"plan forbids semantic authority flag: {key}"
                )
            flags[key] = False
        object.__setattr__(self, "semantic_authority_flags", flags)
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )

        if self.open_required_frontiers:
            if self.disposition is DoctorPlanDisposition.ADMITTED:
                raise DeterministicDoctorAuthorityError(
                    "admitted plans cannot leave required frontiers open"
                )

        if self.disposition is DoctorPlanDisposition.ADMITTED:
            if not self.steps:
                raise DeterministicDoctorError("admitted plans require steps")
            if not self.selected_operator_id:
                raise DeterministicDoctorError(
                    "admitted plans require a unique selected operator"
                )
            if not self.target_ref or not self.value_source_ref or not self.placement_ref:
                raise DeterministicDoctorError(
                    "admitted plans require unique target/value/placement"
                )
            if not self.permitted_write_paths:
                raise DeterministicDoctorAuthorityError(
                    "admitted plans require exact write path authority"
                )
            if not self.lease_id and not self.roots.lease_id:
                raise DeterministicDoctorAuthorityError(
                    "admitted plans require an existing writer lease"
                )
            if not self.checkpoint_ref:
                raise DeterministicDoctorAuthorityError(
                    "admitted plans require a checkpoint ref"
                )
            if not self.rollback_ref:
                raise DeterministicDoctorAuthorityError(
                    "admitted plans require a rollback strategy ref"
                )
            if not self.proof_refs:
                raise DeterministicDoctorError("admitted plans require proof refs")
            if not self.edit_sites:
                raise DeterministicDoctorError("admitted plans require edit sites")
            step_writes = {
                path for step in self.steps for path in step.write_paths
            }
            if not step_writes.issubset(set(self.permitted_write_paths)):
                raise DeterministicDoctorAuthorityError(
                    "step write paths must be within plan write authority"
                )
            # Every consumer must have a closed disposition (enum-enforced);
            # partial coverage is rejected when consumer set is empty already.
            if any(
                item.disposition is DoctorRepairDisposition.ABSTAIN
                for item in self.consumer_dispositions
            ) and any(step.write_paths for step in self.steps):
                # Allowed only when abstaining consumers are excluded from write steps.
                abstain_ids = {
                    item.consumer_id
                    for item in self.consumer_dispositions
                    if item.disposition is DoctorRepairDisposition.ABSTAIN
                }
                for step in self.steps:
                    if step.write_paths and set(step.consumer_ids) & abstain_ids:
                        raise DeterministicDoctorAuthorityError(
                            "write steps cannot cover abstaining consumers"
                        )
        else:
            if self.permitted_write_paths:
                raise DeterministicDoctorAuthorityError(
                    "non-admitted plans cannot grant write path authority"
                )
            if any(step.write_paths for step in self.steps):
                raise DeterministicDoctorAuthorityError(
                    "non-admitted plans cannot schedule write steps"
                )
            # Explicitly reject "partial plan" markers when disposition claims progress.
            if self.disposition is DoctorPlanDisposition.ADMITTED:
                pass  # unreachable
            if not self.consumer_dispositions:
                raise DeterministicDoctorError("plans cannot be partial: consumers required")

        if len(self.steps) > self.resource_bounds.max_plan_steps:
            raise DeterministicDoctorBoundsError("plan exceeds max_plan_steps bound")
        _bounded(self, "deterministic doctor plan")

    @property
    def is_admitted(self) -> bool:
        return self.disposition is DoctorPlanDisposition.ADMITTED

    def _payload(self) -> dict[str, Any]:
        assert self.resource_bounds is not None
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "roots": self.roots.to_dict(),
            "plan_id": self.plan_id,
            "snapshot_id": self.snapshot_id,
            "finding_ids": list(self.finding_ids),
            "disposition": self.disposition.value,
            "consumer_dispositions": [
                item.to_dict() for item in self.consumer_dispositions
            ],
            "impact_closure_id": self.impact_closure_id,
            "steps": [item.to_dict() for item in self.steps],
            "edit_sites": [item.to_dict() for item in self.edit_sites],
            "operator_ids": list(self.operator_ids),
            "target_ref": self.target_ref,
            "value_source_ref": self.value_source_ref,
            "placement_ref": self.placement_ref,
            "selected_operator_id": self.selected_operator_id,
            "premise_refs": list(self.premise_refs),
            "goal_refs": list(self.goal_refs),
            "proof_route_refs": list(self.proof_route_refs),
            "candidate_refs": list(self.candidate_refs),
            "exclusion_refs": list(self.exclusion_refs),
            "open_required_frontiers": list(self.open_required_frontiers),
            "validation_refs": list(self.validation_refs),
            "scc_refs": list(self.scc_refs),
            "tactician_plan_ref": self.tactician_plan_ref,
            "permitted_read_paths": list(self.permitted_read_paths),
            "permitted_write_paths": list(self.permitted_write_paths),
            "lease_id": self.lease_id,
            "checkpoint_ref": self.checkpoint_ref,
            "rollback_ref": self.rollback_ref,
            "proof_refs": list(self.proof_refs),
            "resource_bounds": self.resource_bounds.to_dict(),
            "no_model_invariant": True,
            "llm_router_enabled": False,
            "model_invocation_count": 0,
            "semantic_authority_flags": {
                key: False for key in FORBIDDEN_SEMANTIC_AUTHORITY_FLAGS
            },
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeterministicDoctorPlan":
        fields = (
            "roots",
            "plan_id",
            "snapshot_id",
            "finding_ids",
            "disposition",
            "consumer_dispositions",
            "impact_closure_id",
            "steps",
            "edit_sites",
            "operator_ids",
            "target_ref",
            "value_source_ref",
            "placement_ref",
            "selected_operator_id",
            "premise_refs",
            "goal_refs",
            "proof_route_refs",
            "candidate_refs",
            "exclusion_refs",
            "open_required_frontiers",
            "validation_refs",
            "scc_refs",
            "tactician_plan_ref",
            "permitted_read_paths",
            "permitted_write_paths",
            "lease_id",
            "checkpoint_ref",
            "rollback_ref",
            "proof_refs",
            "resource_bounds",
            "no_model_invariant",
            "llm_router_enabled",
            "model_invocation_count",
            "semantic_authority_flags",
            "invalidation_refs",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "deterministic doctor plan"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorProofCacheAuditReceipt(CanonicalContract):
    """Audit of one proof-cache federation lookup under current roots.

    Cache metadata never grants semantic or write authority.  Positive hits
    inherit only the authority of reconstructed premises.
    """

    SCHEMA: ClassVar[str] = DOCTOR_PROOF_CACHE_AUDIT_RECEIPT_SCHEMA

    roots: DoctorAuthorityRoots
    audit_id: str
    cache_namespace: str
    cache_key: str
    disposition: DoctorCacheAuditDisposition
    canonical_preimage_id: str = ""
    entry_cid: str = ""
    dependency_dag_id: str = ""
    obligation_ref: str = ""
    premise_refs: tuple[str, ...] = ()
    native_goal_ref: str = ""
    reconstruction_ref: str = ""
    kernel_id: str = ""
    provider_local: bool = False
    authoritative: bool = False
    invalidation_refs: tuple[str, ...] = ()
    tombstone_refs: tuple[str, ...] = ()
    single_flight_ref: str = ""
    replay_evidence_ref: str = ""
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "audit_id", _identifier(self.audit_id, "audit_id"))
        object.__setattr__(
            self,
            "cache_namespace",
            _identifier(self.cache_namespace, "cache_namespace"),
        )
        object.__setattr__(
            self, "cache_key", _identifier(self.cache_key, "cache_key")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorCacheAuditDisposition, "disposition"),
        )
        for name in (
            "canonical_preimage_id",
            "entry_cid",
            "dependency_dag_id",
            "obligation_ref",
            "native_goal_ref",
            "reconstruction_ref",
            "kernel_id",
            "single_flight_ref",
            "replay_evidence_ref",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self, "premise_refs", _ids(self.premise_refs, "premise_refs")
        )
        object.__setattr__(
            self, "provider_local", _bool(self.provider_local, "provider_local")
        )
        object.__setattr__(
            self, "authoritative", _bool(self.authoritative, "authoritative")
        )
        if self.semantic_authority is not False:
            raise DeterministicDoctorSafetyError(
                "proof-cache metadata cannot claim semantic_authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        # Only reconstructed hits with premises may be marked authoritative.
        if self.authoritative:
            if self.disposition not in (
                DoctorCacheAuditDisposition.HIT,
                DoctorCacheAuditDisposition.RECONSTRUCTED,
            ):
                raise DeterministicDoctorAuthorityError(
                    "only hit/reconstructed audits may be authoritative"
                )
            if not self.reconstruction_ref or not self.premise_refs:
                raise DeterministicDoctorAuthorityError(
                    "authoritative cache audits require reconstruction and premises"
                )
        if self.disposition is DoctorCacheAuditDisposition.STALE and self.authoritative:
            raise DeterministicDoctorAuthorityError(
                "stale cache entries cannot be authoritative"
            )
        if self.disposition is DoctorCacheAuditDisposition.QUARANTINED and self.authoritative:
            raise DeterministicDoctorAuthorityError(
                "quarantined cache entries cannot be authoritative"
            )
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        object.__setattr__(
            self, "tombstone_refs", _ids(self.tombstone_refs, "tombstone_refs")
        )
        _bounded(self, "doctor proof cache audit receipt")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "roots": self.roots.to_dict(),
            "audit_id": self.audit_id,
            "cache_namespace": self.cache_namespace,
            "cache_key": self.cache_key,
            "disposition": self.disposition.value,
            "canonical_preimage_id": self.canonical_preimage_id,
            "entry_cid": self.entry_cid,
            "dependency_dag_id": self.dependency_dag_id,
            "obligation_ref": self.obligation_ref,
            "premise_refs": list(self.premise_refs),
            "native_goal_ref": self.native_goal_ref,
            "reconstruction_ref": self.reconstruction_ref,
            "kernel_id": self.kernel_id or self.roots.kernel_id,
            "provider_local": self.provider_local,
            "authoritative": self.authoritative,
            "invalidation_refs": list(self.invalidation_refs),
            "tombstone_refs": list(self.tombstone_refs),
            "single_flight_ref": self.single_flight_ref,
            "replay_evidence_ref": self.replay_evidence_ref,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorProofCacheAuditReceipt":
        fields = (
            "roots",
            "audit_id",
            "cache_namespace",
            "cache_key",
            "disposition",
            "canonical_preimage_id",
            "entry_cid",
            "dependency_dag_id",
            "obligation_ref",
            "premise_refs",
            "native_goal_ref",
            "reconstruction_ref",
            "kernel_id",
            "provider_local",
            "authoritative",
            "invalidation_refs",
            "tombstone_refs",
            "single_flight_ref",
            "replay_evidence_ref",
            "semantic_authority",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "doctor proof cache audit receipt"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class DeterministicDoctorRunReceipt(CanonicalContract):
    """Terminal receipt for one deterministic-doctor operation.

    Provider/model invocation counts must be zero.  Repair receipts require
    an admitted plan plus lease/checkpoint/transaction/rollback evidence.
    """

    SCHEMA: ClassVar[str] = DETERMINISTIC_DOCTOR_RUN_RECEIPT_SCHEMA

    roots: DoctorAuthorityRoots
    receipt_id: str
    operation: DoctorOperation
    mode: DoctorMode
    disposition: DoctorRepairDisposition
    snapshot_id: str
    incident_id: str = ""
    plan_id: str = ""
    candidate_tree_cid: str = ""
    committed_tree_cid: str = ""
    cache_audit_ids: tuple[str, ...] = ()
    reconstruction_refs: tuple[str, ...] = ()
    countermodel_refs: tuple[str, ...] = ()
    residual_refs: tuple[str, ...] = ()
    sandbox_enforcement_ref: str = ""
    process_observation_refs: tuple[str, ...] = ()
    network_denied: bool = True
    secrets_inherited: bool = False
    lease_id: str = ""
    checkpoint_ref: str = ""
    transaction_ref: str = ""
    merge_ref: str = ""
    rollback_ref: str = ""
    reindex_ref: str = ""
    invalidation_refs: tuple[str, ...] = ()
    impact_closure_ref: str = ""
    fixed_point_ref: str = ""
    provider_invocation_count: int = 0
    model_invocation_count: int = 0
    llm_router_invoked: bool = False
    remote_model_provider_invoked: bool = False
    target_code_imported: bool = False
    reason_codes: tuple[str, ...] = ()
    resource_bounds: DoctorResourceBounds | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
        )
        object.__setattr__(
            self, "operation", _enum(self.operation, DoctorOperation, "operation")
        )
        object.__setattr__(self, "mode", _enum(self.mode, DoctorMode, "mode"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorRepairDisposition, "disposition"),
        )
        object.__setattr__(
            self, "snapshot_id", _identifier(self.snapshot_id, "snapshot_id")
        )
        for name in (
            "incident_id",
            "plan_id",
            "candidate_tree_cid",
            "committed_tree_cid",
            "sandbox_enforcement_ref",
            "lease_id",
            "checkpoint_ref",
            "transaction_ref",
            "merge_ref",
            "rollback_ref",
            "reindex_ref",
            "impact_closure_ref",
            "fixed_point_ref",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        for name in (
            "cache_audit_ids",
            "reconstruction_refs",
            "countermodel_refs",
            "residual_refs",
            "process_observation_refs",
            "reason_codes",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self, "network_denied", _bool(self.network_denied, "network_denied")
        )
        object.__setattr__(
            self,
            "secrets_inherited",
            _bool(self.secrets_inherited, "secrets_inherited"),
        )
        object.__setattr__(
            self,
            "provider_invocation_count",
            _nonneg_int(self.provider_invocation_count, "provider_invocation_count"),
        )
        object.__setattr__(
            self,
            "model_invocation_count",
            _nonneg_int(self.model_invocation_count, "model_invocation_count"),
        )
        object.__setattr__(
            self,
            "llm_router_invoked",
            _bool(self.llm_router_invoked, "llm_router_invoked"),
        )
        object.__setattr__(
            self,
            "remote_model_provider_invoked",
            _bool(self.remote_model_provider_invoked, "remote_model_provider_invoked"),
        )
        object.__setattr__(
            self,
            "target_code_imported",
            _bool(self.target_code_imported, "target_code_imported"),
        )
        if self.resource_bounds is None:
            object.__setattr__(self, "resource_bounds", DoctorResourceBounds())
        elif isinstance(self.resource_bounds, DoctorResourceBounds):
            pass
        elif isinstance(self.resource_bounds, Mapping):
            rb = self.resource_bounds
            object.__setattr__(
                self,
                "resource_bounds",
                DoctorResourceBounds.from_dict(rb)
                if rb.get("schema") == DoctorResourceBounds.SCHEMA
                else DoctorResourceBounds(
                    **{
                        k: v
                        for k, v in rb.items()
                        if k in DoctorResourceBounds.__dataclass_fields__
                        and k != "SCHEMA"
                    }
                ),
            )
        else:
            raise DeterministicDoctorError("resource_bounds must be DoctorResourceBounds")

        # Hard zero-LLM / zero remote provider invariant.
        if self.provider_invocation_count != 0 or self.model_invocation_count != 0:
            raise DeterministicDoctorSafetyError(
                "deterministic doctor run receipts require zero provider/model invocations"
            )
        if self.llm_router_invoked or self.remote_model_provider_invoked:
            raise DeterministicDoctorSafetyError(
                "deterministic doctor forbids LLM or remote model-provider invocation"
            )
        if self.target_code_imported:
            raise DeterministicDoctorSafetyError(
                "deterministic doctor forbids importing target code"
            )
        if not self.network_denied:
            raise DeterministicDoctorSafetyError(
                "deterministic doctor requires network denial"
            )
        if self.secrets_inherited:
            raise DeterministicDoctorSafetyError(
                "deterministic doctor sandbox must not inherit secrets"
            )

        op = self.operation
        assert isinstance(op, DoctorOperation)
        if op.is_read_only and self.committed_tree_cid:
            raise DeterministicDoctorAuthorityError(
                "read-only operations cannot commit a tree"
            )
        if op is DoctorOperation.REPAIR:
            if self.disposition is DoctorRepairDisposition.SUPPORTED:
                if not self.plan_id:
                    raise DeterministicDoctorAuthorityError(
                        "supported repair receipts require an admitted plan id"
                    )
                if not (self.lease_id or self.roots.lease_id):
                    raise DeterministicDoctorAuthorityError(
                        "supported repair receipts require a writer lease"
                    )
                if not self.checkpoint_ref:
                    raise DeterministicDoctorAuthorityError(
                        "supported repair receipts require a checkpoint"
                    )
                if not self.rollback_ref:
                    raise DeterministicDoctorAuthorityError(
                        "supported repair receipts require rollback evidence"
                    )
                if self.mode is DoctorMode.REPORT_ONLY:
                    raise DeterministicDoctorAuthorityError(
                        "report_only mode cannot complete a supported repair write"
                    )
                if self.mode is DoctorMode.PLAN:
                    raise DeterministicDoctorAuthorityError(
                        "plan mode cannot complete a supported repair write"
                    )
            if (
                self.disposition is DoctorRepairDisposition.SUPPORTED
                and self.mode is DoctorMode.SANDBOX_AUTO
                and self.committed_tree_cid
            ):
                raise DeterministicDoctorAuthorityError(
                    "sandbox_auto may not commit into the target tree"
                )
        if op is DoctorOperation.ROLLBACK:
            if self.disposition not in (
                DoctorRepairDisposition.ROLLED_BACK,
                DoctorRepairDisposition.ABSTAIN,
                DoctorRepairDisposition.QUARANTINED,
            ):
                raise DeterministicDoctorError(
                    "rollback receipts must use rolled_back, abstain, or quarantined"
                )
            if not self.rollback_ref and not self.checkpoint_ref:
                raise DeterministicDoctorAuthorityError(
                    "rollback receipts require checkpoint or rollback evidence"
                )

        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )
        _bounded(self, "deterministic doctor run receipt")

    def _payload(self) -> dict[str, Any]:
        assert self.resource_bounds is not None
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "roots": self.roots.to_dict(),
            "receipt_id": self.receipt_id,
            "operation": self.operation.value,
            "mode": self.mode.value,
            "disposition": self.disposition.value,
            "snapshot_id": self.snapshot_id,
            "incident_id": self.incident_id,
            "plan_id": self.plan_id,
            "candidate_tree_cid": self.candidate_tree_cid,
            "committed_tree_cid": self.committed_tree_cid,
            "cache_audit_ids": list(self.cache_audit_ids),
            "reconstruction_refs": list(self.reconstruction_refs),
            "countermodel_refs": list(self.countermodel_refs),
            "residual_refs": list(self.residual_refs),
            "sandbox_enforcement_ref": self.sandbox_enforcement_ref,
            "process_observation_refs": list(self.process_observation_refs),
            "network_denied": True,
            "secrets_inherited": False,
            "lease_id": self.lease_id,
            "checkpoint_ref": self.checkpoint_ref,
            "transaction_ref": self.transaction_ref,
            "merge_ref": self.merge_ref,
            "rollback_ref": self.rollback_ref,
            "reindex_ref": self.reindex_ref,
            "invalidation_refs": list(self.invalidation_refs),
            "impact_closure_ref": self.impact_closure_ref,
            "fixed_point_ref": self.fixed_point_ref,
            "provider_invocation_count": 0,
            "model_invocation_count": 0,
            "llm_router_invoked": False,
            "remote_model_provider_invoked": False,
            "target_code_imported": False,
            "reason_codes": list(self.reason_codes),
            "resource_bounds": self.resource_bounds.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeterministicDoctorRunReceipt":
        fields = (
            "roots",
            "receipt_id",
            "operation",
            "mode",
            "disposition",
            "snapshot_id",
            "incident_id",
            "plan_id",
            "candidate_tree_cid",
            "committed_tree_cid",
            "cache_audit_ids",
            "reconstruction_refs",
            "countermodel_refs",
            "residual_refs",
            "sandbox_enforcement_ref",
            "process_observation_refs",
            "network_denied",
            "secrets_inherited",
            "lease_id",
            "checkpoint_ref",
            "transaction_ref",
            "merge_ref",
            "rollback_ref",
            "reindex_ref",
            "invalidation_refs",
            "impact_closure_ref",
            "fixed_point_ref",
            "provider_invocation_count",
            "model_invocation_count",
            "llm_router_invoked",
            "remote_model_provider_invoked",
            "target_code_imported",
            "reason_codes",
            "resource_bounds",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "deterministic doctor run receipt"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


def consumer_disposition_set_identity(
    dispositions: Sequence[DoctorConsumerDisposition],
) -> str:
    """Return a canonical identity for a complete consumer-disposition set."""

    ids = sorted(item.content_id for item in dispositions)
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/consumer-disposition-set@1",
            "disposition_ids": list(ids),
        }
    )


__all__ = [
    "ALLOWED_DOCTOR_MODES",
    "ALL_APPROVAL_CLASSES",
    "ALL_DOCTOR_OPERATIONS",
    "ALL_REPAIR_DISPOSITIONS",
    "AUTHORITY_ROOT_FIELDS",
    "DEFAULT_DOCTOR_MODE",
    "DEFAULT_DOCTOR_OPERATION",
    "DETERMINISTIC_DOCTOR_CONTRACTS_INTERFACE",
    "DETERMINISTIC_DOCTOR_FINDING_SCHEMA",
    "DETERMINISTIC_DOCTOR_PLAN_SCHEMA",
    "DETERMINISTIC_DOCTOR_POLICY_SCHEMA",
    "DETERMINISTIC_DOCTOR_RUN_RECEIPT_SCHEMA",
    "DETERMINISTIC_DOCTOR_VERSION",
    "DOCTOR_AUTHORITY_ROOTS_SCHEMA",
    "DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA",
    "DOCTOR_PROOF_CACHE_AUDIT_RECEIPT_SCHEMA",
    "DOCTOR_REPAIR_OPERATOR_SPEC_SCHEMA",
    "DOCTOR_TCB_PATH_MARKERS",
    "FORBIDDEN_SEMANTIC_AUTHORITY_FLAGS",
    "READ_ONLY_OPERATIONS",
    "WRITE_OPERATIONS",
    "DeterministicDoctorAuthorityError",
    "DeterministicDoctorBoundsError",
    "DeterministicDoctorError",
    "DeterministicDoctorFinding",
    "DeterministicDoctorPlan",
    "DeterministicDoctorRunReceipt",
    "DeterministicDoctorSafetyError",
    "DoctorApprovalClass",
    "DoctorAuthorityRoots",
    "DoctorCacheAuditDisposition",
    "DoctorConsumerDisposition",
    "DoctorEditSite",
    "DoctorEvidenceRole",
    "DoctorEvidenceSnapshot",
    "DoctorMode",
    "DoctorOperation",
    "DoctorOperatorKind",
    "DoctorPlanDisposition",
    "DoctorPlanStep",
    "DoctorProofCacheAuditReceipt",
    "DoctorRejectionReason",
    "DoctorRepairDisposition",
    "DoctorRepairOperatorSpec",
    "DoctorResourceBounds",
    "ForgedDeterministicDoctorIdentityError",
    "consumer_disposition_set_identity",
    "default_doctor_mode",
    "is_doctor_tcb_path",
    "operation_is_read_only",
]
