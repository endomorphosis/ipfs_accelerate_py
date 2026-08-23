"""Closed contracts for the causal event supervisor federation.

The records in this module are provider-free value objects.  Importing the
module performs no filesystem, database, network, process, or provider work.
Every wire decoder rejects unknown fields and every record is frozen.  The
common :class:`FederationBinding` keeps authority-bearing roots explicit
rather than hiding join keys in an extension JSON object.

Program: ``agent-supervisor-causal-event-federation-v1``
Interfaces: ``CausalAbstractionSupervisorFederation@1`` and the named records
below.
"""

# Python 3.8 remains supported by the package, so use the compatible
# ``str, Enum`` spelling instead of Python 3.11's ``StrEnum``.
# ``typing.Callable`` also keeps the evaluated ``Decoder`` alias importable on
# Python 3.8; ``collections.abc.Callable`` was not subscriptable there.
# ruff: noqa: UP017, UP035, UP042

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, ClassVar

from ..task_sources.control_plane_contracts import content_identity

PROGRAM_ID = "agent-supervisor-causal-event-federation-v1"
ROOT_OBJECTIVE = "CASF-G000"
CONTRACT_VERSION = 1
MAX_TEXT_BYTES = 16_384
MAX_COLLECTION_ITEMS = 10_000
MAX_DELEGATION_DEPTH = 16
MAX_SUPERVISORS = 1_024
MAX_SUBAGENTS = 65_536

_SCHEMA_PREFIX = "ipfs_accelerate_py/agent-supervisor/causal-federation"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}$")
_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_EFFECT_CLASSES = frozenset(
    {
        "none",
        "read_only",
        "reversible_state",
        "authoritative_state",
        "lease_or_fence",
        "external_reversible",
        "external_irreversible",
        "security_or_legal",
        "payment",
        "proof_lineage",
    }
)
_FEDERATION_RECEIPT_OUTCOMES = frozenset(
    {
        "accepted",
        "created",
        "failed",
        "rejected",
    }
)
_FEDERATION_FIXED_POINT_OUTCOMES = frozenset({"fixed_point"})
_FEDERATION_COMPLETION_OUTCOMES = frozenset({"completed", "fixed_point"})
_SUBAGENT_OUTCOMES = frozenset({"cancelled", "failed", "succeeded"})
_SUPERVISOR_HEALTH_OUTCOMES = frozenset(
    {"healthy", "degraded", "unhealthy", "unknown"}
)
_SUPERVISOR_CHECKPOINT_OUTCOMES = frozenset({"checkpointed", "failed"})
_SUPERVISOR_RECEIPT_OUTCOMES = frozenset(
    {"accepted", "failed", "quarantined", "stopped"}
)
_SHARD_REBALANCE_OUTCOMES = frozenset({"failed", "rebalanced", "rolled_back"})
_FEDERATION_COMMAND_OUTCOMES = frozenset(
    {"applied", "dry_run", "failed", "rejected"}
)
_INTERVENTION_TEST_OUTCOMES = frozenset({"excluded", "matched", "mismatched"})
_SECRET_KEY_RE = re.compile(
    r"(?:password|passwd|private[_-]?key|access[_-]?token|auth[_-]?token|raw[_-]?credential)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|\bsk-[A-Za-z0-9_-]{12,})",
    re.IGNORECASE,
)


class FederationContractError(ValueError):
    """Base fail-closed contract error."""


class UnknownNormativeFieldError(FederationContractError):
    """A wire record contained a field outside its closed schema."""


class FederationBoundsError(FederationContractError):
    """A bounded scalar or collection exceeded its admitted range."""


class FederationSecretError(FederationContractError):
    """Raw credential-shaped material reached a federation contract."""


class FederationAuthorityError(FederationContractError):
    """A record attempted to manufacture authority or completion."""


class CapabilityBlockerCode(str, Enum):
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"
    STALE = "stale"
    UNQUALIFIED = "unqualified"
    UNAVAILABLE = "unavailable"


class FederationLifecycleState(str, Enum):
    DECLARED = "DECLARED"
    ADMITTED = "ADMITTED"
    STARTING = "STARTING"
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    DRAINING = "DRAINING"
    RECOVERING = "RECOVERING"
    QUARANTINED = "QUARANTINED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class SupervisorRole(str, Enum):
    COORDINATOR = "coordinator"
    REPOSITORY = "repository"
    DOMAIN = "domain"
    VERIFICATION = "verification"
    MERGE = "merge"


class BudgetDimensionName(str, Enum):
    CPU_MILLIS = "cpu_millis"
    MEMORY_BYTES = "memory_bytes"
    GPU_MILLIS = "gpu_millis"
    PROCESSES = "processes"
    TEMPORARY_BYTES = "temporary_bytes"
    DURABLE_BYTES = "durable_bytes"
    MODEL_CALLS = "model_calls"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    PROVIDER_SPEND_MICROS = "provider_spend_micros"
    PROOF_MILLIS = "proof_millis"
    VALIDATION_MILLIS = "validation_millis"
    MERGE_SLOTS = "merge_slots"
    HUMAN_QUESTIONS = "human_questions"
    WALL_MILLIS = "wall_millis"


class FederationOperation(str, Enum):
    CREATE = "federation.create"
    START = "federation.start"
    PAUSE = "federation.pause"
    RESUME = "federation.resume"
    DRAIN = "federation.drain"
    STOP = "federation.stop"
    CANCEL = "federation.cancel"
    REBALANCE = "federation.rebalance"
    SCALE = "federation.scale"
    QUARANTINE = "federation.quarantine"
    REPLAY_EVENTS = "federation.replay_events"
    RETRY_DEAD_LETTER = "federation.retry_dead_letter"
    REBUILD_PROJECTION = "federation.rebuild_projection"


class FederationAuthorizationVerdict(str, Enum):
    """Closed state-owner admission verdicts.

    Denials fail before an authoritative admission transaction and therefore
    are deliberately not persisted in the federation control plane.
    """

    ADMITTED = "admitted"


class FederationAuthorizationReason(str, Enum):
    """Closed reasons for an admitted federation-create authorization."""

    AUTHENTICATED_DELEGATED_POLICY_ADMITTED = (
        "authenticated_delegated_policy_admitted"
    )


class CausalLevel(str, Enum):
    L0_RUNTIME = "L0_RUNTIME"
    L1_CODE_ARTIFACT = "L1_CODE_ARTIFACT"
    L2_WORK = "L2_WORK"
    L3_INTENT = "L3_INTENT"
    L4_FEDERATION = "L4_FEDERATION"


class CausalEdgeKind(str, Enum):
    CAUSES = "CAUSES"
    ENABLES = "ENABLES"
    DISABLES = "DISABLES"
    BLOCKS = "BLOCKS"
    INVALIDATES = "INVALIDATES"
    PRODUCES = "PRODUCES"
    OBSERVES = "OBSERVES"
    CONSUMES = "CONSUMES"
    DEPENDS_ON = "DEPENDS_ON"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    REFINES = "REFINES"
    ABSTRACTS = "ABSTRACTS"
    IMPLEMENTS = "IMPLEMENTS"
    DELEGATES_TO = "DELEGATES_TO"
    COMPENSATES = "COMPENSATES"
    SUPERSEDES = "SUPERSEDES"


class CausalEvidenceKind(str, Enum):
    EXACT_STATIC_DEPENDENCY = "exact_static_dependency"
    DYNAMIC_TRACE = "dynamic_trace"
    CONTRACT_DEPENDENCY = "contract_dependency"
    PROOF_DEPENDENCY = "proof_dependency"
    TEST_DEPENDENCY = "test_dependency"
    EFFECT_OBSERVATION = "effect_observation"
    EVENT_SEQUENCE = "event_sequence"
    COUNTEREXAMPLE = "counterexample"
    DELTA_DEBUGGING = "delta_debugging"
    UNSAT_CORE = "unsat_core"
    HUMAN_REVIEWED_ASSERTION = "human_reviewed_causal_assertion"
    RETRIEVAL_NOMINATION = "retrieval_nomination"


class AbstractionFaithfulness(str, Enum):
    EXACT = "EXACT"
    CONSERVATIVE = "CONSERVATIVE"
    EMPIRICALLY_SUPPORTED = "EMPIRICALLY_SUPPORTED"
    HEURISTIC = "HEURISTIC"
    REFUTED = "REFUTED"
    UNKNOWN = "UNKNOWN"


class FrontierDisposition(str, Enum):
    MUST_WAKE = "must_wake"
    MAY_WAKE = "may_wake"
    DO_NOT_WAKE = "do_not_wake"


def _timestamp(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FederationContractError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise FederationContractError(f"{name} must include a timezone")
    return text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value: Any, name: str, *, required: bool = True, maximum: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise FederationContractError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise FederationContractError(f"{name} must not be empty")
    if "\x00" in text or len(text.encode("utf-8")) > maximum:
        raise FederationBoundsError(f"{name} exceeds its text bound")
    if _SECRET_VALUE_RE.search(text):
        raise FederationSecretError(f"{name} contains raw credential material")
    return text


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, maximum=512)
    if text and not _ID_RE.fullmatch(text):
        raise FederationContractError(f"{name} is not a compact identity")
    return text


def _operation(value: Any, name: str = "operation") -> str:
    text = _text(value, name, maximum=128)
    if not _OPERATION_RE.fullmatch(text):
        raise FederationContractError(f"{name} is not a closed operation spelling")
    return text


def _integer(value: Any, name: str, *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, float) and not math.isfinite(value):
        raise FederationBoundsError(f"{name} must be finite")
    if isinstance(value, bool) or not isinstance(value, int):
        raise FederationContractError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise FederationBoundsError(f"{name} must be in {minimum}..{maximum}")
    return int(value)


def _finite(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FederationContractError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise FederationBoundsError(f"{name} must be finite and >= {minimum}")
    return number


def _strings(
    value: Any,
    name: str,
    *,
    maximum: int = MAX_COLLECTION_ITEMS,
    required: bool = False,
    identities: bool = True,
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise FederationContractError(f"{name} must be an array")
    if len(value) > maximum:
        raise FederationBoundsError(f"{name} exceeds {maximum} items")
    result = tuple(
        _identifier(item, f"{name}[{index}]") if identities else _text(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if required and not result:
        raise FederationContractError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise FederationContractError(f"{name} contains duplicate identities")
    return result


def _repository_refs(value: Any, name: str) -> tuple[str, ...]:
    refs = _strings(value, name, maximum=256, required=True)
    for ref in refs:
        if ref.startswith(("/", "~")) or ".." in ref.split("/"):
            raise FederationAuthorityError(
                f"{name} accepts server-resolved identities, not arbitrary paths"
            )
    return refs


def _assert_no_secrets(value: Any, name: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if _SECRET_KEY_RE.search(key_text):
                raise FederationSecretError(f"{name}.{key_text} is a raw-secret field")
            _assert_no_secrets(item, f"{name}.{key_text}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{name}[{index}]")
    elif isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        raise FederationSecretError(f"{name} contains raw credential material")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FederationBoundsError(f"{name} contains a non-finite value")


def _wire(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, ClosedContract):
        return value.to_dict()
    if is_dataclass(value):
        return {field.name: _wire(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _wire(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire(item) for item in value]
    return value


Decoder = Callable[[Any], Any]


@dataclass(frozen=True)
class ClosedContract:
    """Base for immutable, versioned, unknown-field-rejecting records."""

    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/closed-contract@1"
    VERSION: ClassVar[int] = CONTRACT_VERSION
    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType({})

    def to_dict(self) -> dict[str, Any]:
        payload = {"schema": self.SCHEMA}
        for field in fields(self):
            payload[field.name] = _wire(getattr(self, field.name))
        _assert_no_secrets(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClosedContract:
        if not isinstance(payload, Mapping):
            raise FederationContractError(f"{cls.__name__} payload must be an object")
        allowed = {field.name for field in fields(cls)} | {"schema"}
        unknown = set(payload) - allowed
        if unknown:
            raise UnknownNormativeFieldError(
                f"{cls.__name__} has unknown fields: {sorted(unknown)}"
            )
        if payload.get("schema") != cls.SCHEMA:
            raise FederationContractError(f"{cls.__name__}.schema must equal {cls.SCHEMA!r}")
        values: dict[str, Any] = {}
        for field in fields(cls):
            if field.name not in payload:
                raise FederationContractError(f"{cls.__name__}.{field.name} is required")
            raw = payload[field.name]
            decoder = cls.FIELD_DECODERS.get(field.name)
            values[field.name] = decoder(raw) if decoder else raw
        _assert_no_secrets(values)
        return cls(**values)

    @property
    def cid(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class FederationBinding(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/federation-binding@1"

    tenant_id: str
    repository_ids: tuple[str, ...]
    repository_tree_ids: tuple[str, ...]
    program_id: str
    objective_ref: str
    objective_revision: int
    policy_ref: str
    policy_revision: int
    operation_catalog_ref: str
    control_plane_generation: int
    causal_graph_revision: int
    semantic_state_roots: tuple[str, ...]
    supervisor_population: int
    budget_ref: str
    expires_at: str
    issuer: str
    authorization_evidence_ref: str

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {
            "repository_ids": tuple,
            "repository_tree_ids": tuple,
            "semantic_state_roots": tuple,
        }
    )

    def __post_init__(self) -> None:
        _identifier(self.tenant_id, "tenant_id")
        repos = _repository_refs(self.repository_ids, "repository_ids")
        trees = _strings(
            self.repository_tree_ids, "repository_tree_ids", maximum=256, required=True
        )
        if len(repos) != len(trees):
            raise FederationContractError(
                "repository_ids and repository_tree_ids must have equal cardinality"
            )
        _identifier(self.program_id, "program_id")
        _identifier(self.objective_ref, "objective_ref")
        _integer(self.objective_revision, "objective_revision", minimum=1)
        _identifier(self.policy_ref, "policy_ref")
        _integer(self.policy_revision, "policy_revision", minimum=1)
        _identifier(self.operation_catalog_ref, "operation_catalog_ref")
        _integer(self.control_plane_generation, "control_plane_generation", minimum=1)
        _integer(self.causal_graph_revision, "causal_graph_revision")
        _strings(self.semantic_state_roots, "semantic_state_roots", maximum=256, required=True)
        _integer(self.supervisor_population, "supervisor_population", maximum=MAX_SUPERVISORS)
        _identifier(self.budget_ref, "budget_ref")
        _timestamp(self.expires_at, "expires_at")
        _identifier(self.issuer, "issuer")
        _identifier(self.authorization_evidence_ref, "authorization_evidence_ref")


def _binding(value: Any) -> FederationBinding:
    if isinstance(value, FederationBinding):
        return value
    return FederationBinding.from_dict(value)  # type: ignore[return-value]


@dataclass(frozen=True)
class BoundRecord(ClosedContract):
    record_id: str
    revision: int
    binding: FederationBinding

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType({"binding": _binding})

    def __post_init__(self) -> None:
        _identifier(self.record_id, "record_id")
        _integer(self.revision, "revision", minimum=1)
        if not isinstance(self.binding, FederationBinding):
            raise FederationContractError("binding must be a FederationBinding")


@dataclass(frozen=True)
class BoundStateRecord(BoundRecord):
    state: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _identifier(self.state, "state")


@dataclass(frozen=True)
class BoundDefinition(BoundRecord):
    name: str
    capabilities: tuple[str, ...]
    allowed_operations: tuple[str, ...]
    effect_ceiling: str
    risk_ceiling: str
    resource_budget_ref: str
    token_budget_ref: str

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {"binding": _binding, "capabilities": tuple, "allowed_operations": tuple}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.name, "name", maximum=512)
        _strings(self.capabilities, "capabilities", maximum=256)
        operations = _strings(
            self.allowed_operations, "allowed_operations", maximum=256, identities=False
        )
        for operation in operations:
            _operation(operation)
        _identifier(self.effect_ceiling, "effect_ceiling")
        _identifier(self.risk_ceiling, "risk_ceiling")
        _identifier(self.resource_budget_ref, "resource_budget_ref")
        _identifier(self.token_budget_ref, "token_budget_ref")


@dataclass(frozen=True)
class BoundAssignment(BoundRecord):
    subject_id: str
    repository_ids: tuple[str, ...]
    goal_refs: tuple[str, ...]
    task_refs: tuple[str, ...]
    allowed_task_families: tuple[str, ...]
    fencing_epoch: int

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {
            "binding": _binding,
            "repository_ids": tuple,
            "goal_refs": tuple,
            "task_refs": tuple,
            "allowed_task_families": tuple,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _identifier(self.subject_id, "subject_id")
        _repository_refs(self.repository_ids, "repository_ids")
        _strings(self.goal_refs, "goal_refs", maximum=4_096)
        _strings(self.task_refs, "task_refs", maximum=10_000)
        _strings(self.allowed_task_families, "allowed_task_families", maximum=256)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)


@dataclass(frozen=True)
class BoundReceipt(BoundRecord):
    outcome: str
    evidence_refs: tuple[str, ...]
    recorded_at: str

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {"binding": _binding, "evidence_refs": tuple}
    )
    ALLOWED_OUTCOMES: ClassVar[frozenset[str]] = frozenset()

    def __post_init__(self) -> None:
        super().__post_init__()
        outcome = _identifier(self.outcome, "outcome")
        if not self.ALLOWED_OUTCOMES or outcome not in self.ALLOWED_OUTCOMES:
            raise FederationAuthorityError(
                f"{type(self).__name__} outcome is outside its closed vocabulary"
            )
        _strings(
            self.evidence_refs,
            "evidence_refs",
            maximum=4_096,
            required=True,
        )
        _timestamp(self.recorded_at, "recorded_at")


@dataclass(frozen=True)
class BudgetDimension(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/budget-dimension@1"

    name: BudgetDimensionName
    # All dimensions use an integral canonical unit (milliseconds, bytes,
    # counts, tokens, or spend micros).  Integer-only values preserve the
    # control plane's deterministic DAG-JSON identity contract.
    ceiling: int
    reserved: int
    consumed: int

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {"name": BudgetDimensionName}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, BudgetDimensionName):
            raise FederationContractError("budget dimension name is not closed")
        ceiling = _integer(self.ceiling, "ceiling")
        reserved = _integer(self.reserved, "reserved")
        consumed = _integer(self.consumed, "consumed")
        if reserved > ceiling or consumed > reserved:
            raise FederationBoundsError("budget invariant requires consumed <= reserved <= ceiling")


def _budget_dimensions(value: Any) -> tuple[BudgetDimension, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise FederationContractError("dimensions must be an array")
    if len(value) > len(BudgetDimensionName):
        raise FederationBoundsError("too many budget dimensions")
    result = tuple(
        item if isinstance(item, BudgetDimension) else BudgetDimension.from_dict(item)
        for item in value
    )
    names = [item.name for item in result]
    if len(set(names)) != len(names):
        raise FederationContractError("budget dimensions must be unique")
    return result


@dataclass(frozen=True)
class BoundBudget(BoundRecord):
    parent_budget_id: str
    owner_id: str
    dimensions: tuple[BudgetDimension, ...]
    status: str

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {"binding": _binding, "dimensions": _budget_dimensions}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _identifier(self.parent_budget_id, "parent_budget_id", required=False)
        _identifier(self.owner_id, "owner_id")
        _budget_dimensions(self.dimensions)
        _identifier(self.status, "status")


@dataclass(frozen=True)
class FederationRequest(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/federation-request@1"

    caller_did: str
    delegation_chain: tuple[str, ...]
    audience: str
    program_id: str
    repository_roots: tuple[str, ...]
    objective_ref: str
    requested_supervisor_profile: str
    maximum_supervisors: int
    maximum_subagents: int
    resource_budget: ResourceBudget
    token_budget: TokenBudget
    effect_scope: tuple[str, ...]
    policy_ref: str
    expiry: str
    nonce: str
    idempotency_key: str
    binding: FederationBinding

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {
            "delegation_chain": tuple,
            "repository_roots": tuple,
            "resource_budget": lambda value: (
                value if isinstance(value, ResourceBudget) else ResourceBudget.from_dict(value)
            ),
            "token_budget": lambda value: (
                value if isinstance(value, TokenBudget) else TokenBudget.from_dict(value)
            ),
            "effect_scope": tuple,
            "binding": _binding,
        }
    )

    def __post_init__(self) -> None:
        _identifier(self.caller_did, "caller_did")
        _strings(
            self.delegation_chain,
            "delegation_chain",
            maximum=MAX_DELEGATION_DEPTH,
        )
        _identifier(self.audience, "audience")
        if _identifier(self.program_id, "program_id") != self.binding.program_id:
            raise FederationContractError("program_id differs from authority binding")
        roots = _repository_refs(self.repository_roots, "repository_roots")
        if roots != self.binding.repository_ids:
            raise FederationContractError(
                "repository_roots must equal the bound server-resolved repository identities"
            )
        if _identifier(self.objective_ref, "objective_ref") != self.binding.objective_ref:
            raise FederationContractError("objective_ref differs from authority binding")
        _identifier(self.requested_supervisor_profile, "requested_supervisor_profile")
        _integer(
            self.maximum_supervisors,
            "maximum_supervisors",
            minimum=1,
            maximum=MAX_SUPERVISORS,
        )
        _integer(
            self.maximum_subagents,
            "maximum_subagents",
            minimum=1,
            maximum=MAX_SUBAGENTS,
        )
        if not isinstance(self.resource_budget, ResourceBudget):
            raise FederationContractError("resource_budget must be ResourceBudget")
        if not isinstance(self.token_budget, TokenBudget):
            raise FederationContractError("token_budget must be TokenBudget")
        scopes = _strings(
            self.effect_scope, "effect_scope", maximum=128, required=True, identities=False
        )
        for scope in scopes:
            _operation(scope, "effect_scope")
        if _identifier(self.policy_ref, "policy_ref") != self.binding.policy_ref:
            raise FederationContractError("policy_ref differs from authority binding")
        if _timestamp(self.expiry, "expiry") != self.binding.expires_at:
            raise FederationContractError("expiry differs from authority binding")
        _identifier(self.nonce, "nonce")
        _identifier(self.idempotency_key, "idempotency_key")


@dataclass(frozen=True)
class FederationAuthorizationDecision(ClosedContract):
    """Redacted, content-addressed result of server-side admission.

    This record contains only immutable identities.  In particular, it binds
    the CID of the verified :class:`AuthenticationEvidence` rather than its
    caller-selected evidence label, and never carries a signature, key handle,
    credential, or executable policy outcome supplied by the request.
    """

    SCHEMA: ClassVar[str] = (
        f"{_SCHEMA_PREFIX}/federation-authorization-decision@1"
    )

    request_cid: str
    caller_did: str
    delegation_chain_cid: str
    audience: str
    operation: FederationOperation
    resolved_scope_cid: str
    policy_id: str
    policy_revision: int
    verdict: FederationAuthorizationVerdict
    reason: FederationAuthorizationReason
    authentication_evidence_cid: str
    expires_at: str
    decided_at: str

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {
            "operation": FederationOperation,
            "verdict": FederationAuthorizationVerdict,
            "reason": FederationAuthorizationReason,
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "request_cid",
            "caller_did",
            "delegation_chain_cid",
            "audience",
            "resolved_scope_cid",
            "policy_id",
            "authentication_evidence_cid",
        ):
            _identifier(getattr(self, name), name)
        if self.operation is not FederationOperation.CREATE:
            raise FederationAuthorityError(
                "federation authorization decision must bind federation.create"
            )
        _integer(self.policy_revision, "policy_revision", minimum=1)
        if self.verdict is not FederationAuthorizationVerdict.ADMITTED:
            raise FederationAuthorityError(
                "denied authorization decisions are not authoritative admissions"
            )
        if (
            self.reason
            is not FederationAuthorizationReason.AUTHENTICATED_DELEGATED_POLICY_ADMITTED
        ):
            raise FederationAuthorityError(
                "authorization decision reason is outside the admitted vocabulary"
            )
        expiry = _timestamp(self.expires_at, "expires_at")
        decided = _timestamp(self.decided_at, "decided_at")
        expiry_at = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        decided_at = datetime.fromisoformat(decided.replace("Z", "+00:00"))
        if decided_at > expiry_at:
            raise FederationAuthorityError(
                "authorization decision was made after its authority expired"
            )


class FederationIdentity(BoundRecord):
    SCHEMA = f"{_SCHEMA_PREFIX}/federation-identity@1"


@dataclass(frozen=True)
class FederationPolicy(BoundRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/federation-policy@1"

    allowed_callers: tuple[str, ...]
    allowed_audiences: tuple[str, ...]
    allowed_operations: tuple[str, ...]
    allowed_effects: tuple[str, ...]
    maximum_supervisors: int
    maximum_subagents: int
    maximum_concurrent_subagents: int
    conservative_abstraction_scheduling: bool

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {
            "binding": _binding,
            "allowed_callers": tuple,
            "allowed_audiences": tuple,
            "allowed_operations": tuple,
            "allowed_effects": tuple,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _strings(self.allowed_callers, "allowed_callers", maximum=1_024, required=True)
        _strings(self.allowed_audiences, "allowed_audiences", maximum=128, required=True)
        operations = _strings(
            self.allowed_operations,
            "allowed_operations",
            maximum=256,
            required=True,
            identities=False,
        )
        for operation in operations:
            _operation(operation)
        effects = _strings(
            self.allowed_effects, "allowed_effects", maximum=128, required=True, identities=False
        )
        for effect in effects:
            _operation(effect, "allowed_effects")
        _integer(
            self.maximum_supervisors, "maximum_supervisors", minimum=1, maximum=MAX_SUPERVISORS
        )
        _integer(self.maximum_subagents, "maximum_subagents", minimum=1, maximum=MAX_SUBAGENTS)
        concurrency = _integer(
            self.maximum_concurrent_subagents,
            "maximum_concurrent_subagents",
            minimum=1,
            maximum=MAX_SUBAGENTS,
        )
        if concurrency > self.maximum_subagents:
            raise FederationBoundsError("concurrent subagents exceed registered ceiling")
        if type(self.conservative_abstraction_scheduling) is not bool:
            raise FederationContractError("conservative_abstraction_scheduling must be boolean")


@dataclass(frozen=True)
class FederationPlan(BoundRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/federation-plan@1"

    goal_refs: tuple[str, ...]
    task_refs: tuple[str, ...]
    supervisor_definition_refs: tuple[str, ...]

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {
            "binding": _binding,
            "goal_refs": tuple,
            "task_refs": tuple,
            "supervisor_definition_refs": tuple,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _strings(self.goal_refs, "goal_refs", maximum=4_096, required=True)
        _strings(self.task_refs, "task_refs", maximum=10_000, required=True)
        _strings(
            self.supervisor_definition_refs,
            "supervisor_definition_refs",
            maximum=MAX_SUPERVISORS,
            required=True,
        )


class FederationRevision(BoundRecord):
    SCHEMA = f"{_SCHEMA_PREFIX}/federation-revision@1"


class FederationState(BoundStateRecord):
    SCHEMA = f"{_SCHEMA_PREFIX}/federation-state@1"


class FederationReceipt(BoundReceipt):
    SCHEMA = f"{_SCHEMA_PREFIX}/federation-receipt@1"
    ALLOWED_OUTCOMES = _FEDERATION_RECEIPT_OUTCOMES

    def __post_init__(self) -> None:
        super().__post_init__()


class SupervisorDefinition(BoundDefinition):
    SCHEMA = f"{_SCHEMA_PREFIX}/supervisor-definition@1"


class SupervisorIdentity(BoundRecord):
    SCHEMA = f"{_SCHEMA_PREFIX}/supervisor-identity@1"


@dataclass(frozen=True)
class SupervisorInstance(BoundStateRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/supervisor-instance@1"

    federation_id: str
    parent_supervisor_id: str
    role: SupervisorRole
    lease_id: str
    fencing_epoch: int

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {"binding": _binding, "role": SupervisorRole}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _identifier(self.federation_id, "federation_id")
        _identifier(self.parent_supervisor_id, "parent_supervisor_id", required=False)
        if not isinstance(self.role, SupervisorRole):
            raise FederationContractError("role is not in the closed supervisor vocabulary")
        _identifier(self.lease_id, "lease_id")
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        FederationLifecycleState(self.state)


class SupervisorAssignment(BoundAssignment):
    SCHEMA = f"{_SCHEMA_PREFIX}/supervisor-assignment@1"


class SupervisorSpecialization(BoundDefinition):
    SCHEMA = f"{_SCHEMA_PREFIX}/supervisor-specialization@1"


class SupervisorCapability(BoundDefinition):
    SCHEMA = f"{_SCHEMA_PREFIX}/supervisor-capability@1"


class SupervisorLifecycle(BoundStateRecord):
    SCHEMA = f"{_SCHEMA_PREFIX}/supervisor-lifecycle@1"


class SupervisorHealth(BoundReceipt):
    SCHEMA = f"{_SCHEMA_PREFIX}/supervisor-health@1"
    ALLOWED_OUTCOMES = _SUPERVISOR_HEALTH_OUTCOMES


class SupervisorCheckpoint(BoundReceipt):
    SCHEMA = f"{_SCHEMA_PREFIX}/supervisor-checkpoint@1"
    ALLOWED_OUTCOMES = _SUPERVISOR_CHECKPOINT_OUTCOMES


class SupervisorReceipt(BoundReceipt):
    SCHEMA = f"{_SCHEMA_PREFIX}/supervisor-receipt@1"
    ALLOWED_OUTCOMES = _SUPERVISOR_RECEIPT_OUTCOMES


class SubagentDefinition(BoundDefinition):
    SCHEMA = f"{_SCHEMA_PREFIX}/subagent-definition@1"


class SubagentIdentity(BoundRecord):
    SCHEMA = f"{_SCHEMA_PREFIX}/subagent-identity@1"


@dataclass(frozen=True)
class SubagentInstance(BoundStateRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/subagent-instance@1"

    federation_id: str
    supervisor_id: str
    task_id: str
    lease_id: str
    fencing_epoch: int

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType({"binding": _binding})

    def __post_init__(self) -> None:
        super().__post_init__()
        _identifier(self.federation_id, "federation_id")
        _identifier(self.supervisor_id, "supervisor_id")
        _identifier(self.task_id, "task_id", required=False)
        _identifier(self.lease_id, "lease_id", required=False)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        FederationLifecycleState(self.state)


class SubagentAssignment(BoundAssignment):
    SCHEMA = f"{_SCHEMA_PREFIX}/subagent-assignment@1"


class SubagentCapability(BoundDefinition):
    SCHEMA = f"{_SCHEMA_PREFIX}/subagent-capability@1"


class SubagentBudget(BoundBudget):
    SCHEMA = f"{_SCHEMA_PREFIX}/subagent-budget@1"


@dataclass(frozen=True)
class SubagentOutcome(BoundReceipt):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/subagent-outcome@1"
    ALLOWED_OUTCOMES = _SUBAGENT_OUTCOMES

    federation_id: str
    supervisor_id: str
    subagent_id: str
    task_id: str
    fencing_epoch: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.outcome not in _SUBAGENT_OUTCOMES:
            raise FederationAuthorityError(
                "subagent outcomes are observations and cannot assert acceptance or completion"
            )
        _strings(
            self.evidence_refs,
            "evidence_refs",
            maximum=4_096,
            required=True,
        )
        for name in ("federation_id", "supervisor_id", "subagent_id"):
            _identifier(getattr(self, name), name)
        _identifier(self.task_id, "task_id", required=False)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)


class SupervisorShard(BoundStateRecord):
    SCHEMA = f"{_SCHEMA_PREFIX}/supervisor-shard@1"


@dataclass(frozen=True)
class ShardBoundary(BoundRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/shard-boundary@1"

    repository_ids: tuple[str, ...]
    goal_refs: tuple[str, ...]
    task_refs: tuple[str, ...]
    symbol_refs: tuple[str, ...]
    effect_classes: tuple[str, ...]

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {
            "binding": _binding,
            "repository_ids": tuple,
            "goal_refs": tuple,
            "task_refs": tuple,
            "symbol_refs": tuple,
            "effect_classes": tuple,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _repository_refs(self.repository_ids, "repository_ids")
        _strings(self.goal_refs, "goal_refs", maximum=4_096)
        _strings(self.task_refs, "task_refs", maximum=10_000)
        _strings(self.symbol_refs, "symbol_refs", maximum=10_000)
        effect_classes = _strings(
            self.effect_classes,
            "effect_classes",
            maximum=len(_EFFECT_CLASSES),
            identities=False,
        )
        unknown = set(effect_classes) - _EFFECT_CLASSES
        if unknown:
            raise FederationContractError(
                f"effect_classes contains values outside the closed vocabulary: {sorted(unknown)}"
            )


class ShardAssignment(BoundAssignment):
    SCHEMA = f"{_SCHEMA_PREFIX}/shard-assignment@1"


class ShardRevision(BoundRecord):
    SCHEMA = f"{_SCHEMA_PREFIX}/shard-revision@1"


class ShardRebalancePlan(BoundAssignment):
    SCHEMA = f"{_SCHEMA_PREFIX}/shard-rebalance-plan@1"


class ShardRebalanceReceipt(BoundReceipt):
    SCHEMA = f"{_SCHEMA_PREFIX}/shard-rebalance-receipt@1"
    ALLOWED_OUTCOMES = _SHARD_REBALANCE_OUTCOMES


class FederationBudget(BoundBudget):
    SCHEMA = f"{_SCHEMA_PREFIX}/federation-budget@1"


class SupervisorBudget(BoundBudget):
    SCHEMA = f"{_SCHEMA_PREFIX}/supervisor-budget@1"


class AgentBudget(BoundBudget):
    SCHEMA = f"{_SCHEMA_PREFIX}/agent-budget@1"


class TokenBudget(BoundBudget):
    SCHEMA = f"{_SCHEMA_PREFIX}/token-budget@1"


class ResourceBudget(BoundBudget):
    SCHEMA = f"{_SCHEMA_PREFIX}/resource-budget@1"


@dataclass(frozen=True)
class BudgetReservation(BoundBudget):
    """Typed, expiring admission reservation issued by the budget authority.

    The record is intentionally more than an opaque identifier: the state
    owner can independently prove that it belongs to the exact request,
    policy, budget roots, tenant, federation, and idempotency scope before a
    federation transaction consumes it.
    """

    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/budget-reservation@1"

    request_cid: str
    idempotency_key: str
    policy_ref: str
    policy_revision: int
    resource_budget_ref: str
    token_budget_ref: str
    issued_at: str
    expires_at: str
    authorization_evidence_ref: str

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "request_cid",
            "idempotency_key",
            "policy_ref",
            "resource_budget_ref",
            "token_budget_ref",
            "authorization_evidence_ref",
        ):
            _identifier(getattr(self, name), name)
        _integer(self.policy_revision, "policy_revision", minimum=1)
        issued = _timestamp(self.issued_at, "issued_at")
        expires = _timestamp(self.expires_at, "expires_at")
        issued_at = datetime.fromisoformat(issued.replace("Z", "+00:00"))
        expires_at = datetime.fromisoformat(expires.replace("Z", "+00:00"))
        if expires_at <= issued_at:
            raise FederationContractError(
                "budget reservation expiry must be after issuance"
            )
        if expires != self.binding.expires_at:
            raise FederationAuthorityError(
                "budget reservation expiry differs from its authority binding"
            )


class BudgetLedger(BoundBudget):
    SCHEMA = f"{_SCHEMA_PREFIX}/budget-ledger@1"


@dataclass(frozen=True)
class FederationCommand(BoundRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/federation-command@1"

    operation: FederationOperation
    target_id: str
    expected_generation: int
    expected_revision: int
    expected_fencing_epoch: int
    idempotency_key: str
    dry_run: bool
    expected_effects: tuple[str, ...]

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {
            "binding": _binding,
            "operation": FederationOperation,
            "expected_effects": tuple,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.operation, FederationOperation):
            raise FederationContractError("operation is not in the closed catalog")
        _identifier(self.target_id, "target_id")
        _integer(self.expected_generation, "expected_generation", minimum=1)
        _integer(self.expected_revision, "expected_revision")
        _integer(self.expected_fencing_epoch, "expected_fencing_epoch", minimum=1)
        _identifier(self.idempotency_key, "idempotency_key")
        if type(self.dry_run) is not bool:
            raise FederationContractError("dry_run must be boolean")
        effects = _strings(self.expected_effects, "expected_effects", maximum=256, identities=False)
        for effect in effects:
            _operation(effect, "expected_effects")


class FederationCommandResult(BoundReceipt):
    SCHEMA = f"{_SCHEMA_PREFIX}/federation-command-result@1"
    ALLOWED_OUTCOMES = _FEDERATION_COMMAND_OUTCOMES


@dataclass(frozen=True)
class FederationIdempotencyRecord(BoundRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/federation-idempotency-record@1"

    idempotency_key: str
    command_id: str
    result_ref: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _identifier(self.idempotency_key, "idempotency_key")
        _identifier(self.command_id, "command_id")
        _identifier(self.result_ref, "result_ref")


@dataclass(frozen=True)
class FederationWorldSnapshot(BoundRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/federation-world-snapshot@1"

    event_watermark: int
    task_population_ref: str
    claim_population_ref: str
    merge_state_ref: str
    proof_state_ref: str
    semantic_roots: tuple[str, ...]
    causal_frontier_ref: str

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {"binding": _binding, "semantic_roots": tuple}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _integer(self.event_watermark, "event_watermark")
        for name in (
            "task_population_ref",
            "claim_population_ref",
            "merge_state_ref",
            "proof_state_ref",
            "causal_frontier_ref",
        ):
            _identifier(getattr(self, name), name)
        _strings(self.semantic_roots, "semantic_roots", maximum=256, required=True)


@dataclass(frozen=True)
class FederationFixedPoint(BoundReceipt):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/federation-fixed-point@1"

    world_snapshot_ref: str
    event_watermark: int
    outstanding_required_work: int

    ALLOWED_OUTCOMES: ClassVar[frozenset[str]] = _FEDERATION_FIXED_POINT_OUTCOMES

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.outcome not in self.ALLOWED_OUTCOMES:
            raise FederationAuthorityError(
                "fixed-point receipts require a closed fixed-point or completion outcome"
            )
        _strings(
            self.evidence_refs,
            "evidence_refs",
            maximum=4_096,
            required=True,
        )
        _identifier(self.world_snapshot_ref, "world_snapshot_ref")
        _integer(self.event_watermark, "event_watermark")
        _integer(self.outstanding_required_work, "outstanding_required_work")
        if self.outstanding_required_work != 0:
            raise FederationAuthorityError(
                "fixed-point or completion outcome cannot carry outstanding required work"
            )


class FederationCompletionReceipt(FederationFixedPoint):
    SCHEMA = f"{_SCHEMA_PREFIX}/federation-completion-receipt@1"
    ALLOWED_OUTCOMES = _FEDERATION_COMPLETION_OUTCOMES


@dataclass(frozen=True)
class CapabilityBlocker(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/capability-blocker@1"

    blocker_id: str
    capability: str
    code: CapabilityBlockerCode
    authority: str
    reason: str
    independent_work_may_continue: bool
    observed_at: str
    evidence_refs: tuple[str, ...]

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {"code": CapabilityBlockerCode, "evidence_refs": tuple}
    )

    def __post_init__(self) -> None:
        _identifier(self.blocker_id, "blocker_id")
        _identifier(self.capability, "capability")
        if not isinstance(self.code, CapabilityBlockerCode):
            raise FederationContractError("capability blocker code is not closed")
        _identifier(self.authority, "authority")
        _text(self.reason, "reason", maximum=4_096)
        if type(self.independent_work_may_continue) is not bool:
            raise FederationContractError("independent_work_may_continue must be boolean")
        _timestamp(self.observed_at, "observed_at")
        _strings(self.evidence_refs, "evidence_refs", maximum=256)


@dataclass(frozen=True)
class CausalNode(BoundRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/causal-node@1"

    level: CausalLevel
    node_type: str
    subject_ref: str

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {"binding": _binding, "level": CausalLevel}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.level, CausalLevel):
            raise FederationContractError("causal level is not closed")
        _identifier(self.node_type, "node_type")
        _identifier(self.subject_ref, "subject_ref")


@dataclass(frozen=True)
class CausalEvidence(BoundRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/causal-evidence@1"

    evidence_kind: CausalEvidenceKind
    evidence_ref: str
    authoritative: bool

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {"binding": _binding, "evidence_kind": CausalEvidenceKind}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.evidence_kind, CausalEvidenceKind):
            raise FederationContractError("causal evidence kind is not closed")
        _identifier(self.evidence_ref, "evidence_ref")
        if type(self.authoritative) is not bool:
            raise FederationContractError("authoritative must be boolean")
        if self.evidence_kind is CausalEvidenceKind.RETRIEVAL_NOMINATION and self.authoritative:
            raise FederationAuthorityError(
                "retrieval nomination cannot be authoritative causal evidence"
            )


@dataclass(frozen=True)
class CausalEdge(BoundRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/causal-edge@1"

    source_node_id: str
    target_node_id: str
    edge_kind: CausalEdgeKind
    evidence_refs: tuple[str, ...]
    nomination_only: bool

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {
            "binding": _binding,
            "edge_kind": CausalEdgeKind,
            "evidence_refs": tuple,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _identifier(self.source_node_id, "source_node_id")
        _identifier(self.target_node_id, "target_node_id")
        if self.source_node_id == self.target_node_id:
            raise FederationContractError(
                "causal self-cycles require an explicit fixed-point group"
            )
        if not isinstance(self.edge_kind, CausalEdgeKind):
            raise FederationContractError("causal edge kind is not closed")
        _strings(self.evidence_refs, "evidence_refs", maximum=256, required=True)
        if type(self.nomination_only) is not bool:
            raise FederationContractError("nomination_only must be boolean")


@dataclass(frozen=True)
class CausalAbstractionMap(BoundRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/causal-abstraction-map@1"

    low_level_model_ref: str
    high_level_model_ref: str
    low_level_variables: tuple[str, ...]
    high_level_variables: tuple[str, ...]
    abstraction_function_ref: str
    intervention_mapping_ref: str
    admitted_domain_refs: tuple[str, ...]
    excluded_domain_refs: tuple[str, ...]
    validation_evidence_refs: tuple[str, ...]
    faithfulness_status: AbstractionFaithfulness
    policy_admitted: bool

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {
            "binding": _binding,
            "low_level_variables": tuple,
            "high_level_variables": tuple,
            "admitted_domain_refs": tuple,
            "excluded_domain_refs": tuple,
            "validation_evidence_refs": tuple,
            "faithfulness_status": AbstractionFaithfulness,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "low_level_model_ref",
            "high_level_model_ref",
            "abstraction_function_ref",
            "intervention_mapping_ref",
        ):
            _identifier(getattr(self, name), name)
        _strings(self.low_level_variables, "low_level_variables", maximum=4_096, required=True)
        _strings(self.high_level_variables, "high_level_variables", maximum=4_096, required=True)
        _strings(self.admitted_domain_refs, "admitted_domain_refs", maximum=4_096, required=True)
        _strings(self.excluded_domain_refs, "excluded_domain_refs", maximum=4_096)
        _strings(
            self.validation_evidence_refs,
            "validation_evidence_refs",
            maximum=4_096,
            required=True,
        )
        if not isinstance(self.faithfulness_status, AbstractionFaithfulness):
            raise FederationContractError("faithfulness status is not closed")
        if type(self.policy_admitted) is not bool:
            raise FederationContractError("policy_admitted must be boolean")
        if (
            self.faithfulness_status is AbstractionFaithfulness.CONSERVATIVE
            and not self.policy_admitted
        ):
            # A conservative map can exist without policy admission; it simply
            # cannot drive authority.  Preserve that state explicitly.
            return
        if (
            self.faithfulness_status
            in {
                AbstractionFaithfulness.EMPIRICALLY_SUPPORTED,
                AbstractionFaithfulness.HEURISTIC,
                AbstractionFaithfulness.REFUTED,
                AbstractionFaithfulness.UNKNOWN,
            }
            and self.policy_admitted
        ):
            raise FederationAuthorityError(
                "nomination-only abstraction status cannot be policy-admitted for authority"
            )


@dataclass(frozen=True)
class InterventionTest(BoundReceipt):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/intervention-test@1"
    ALLOWED_OUTCOMES = _INTERVENTION_TEST_OUTCOMES

    abstraction_map_id: str
    low_level_intervention_ref: str
    low_level_outcome_ref: str
    abstracted_outcome_ref: str
    high_level_intervention_ref: str
    high_level_outcome_ref: str
    mismatch_ref: str

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "abstraction_map_id",
            "low_level_intervention_ref",
            "low_level_outcome_ref",
            "abstracted_outcome_ref",
            "high_level_intervention_ref",
            "high_level_outcome_ref",
        ):
            _identifier(getattr(self, name), name)
        _identifier(self.mismatch_ref, "mismatch_ref", required=False)


@dataclass(frozen=True)
class CausalFrontierEntry(BoundRecord):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/causal-frontier-entry@1"

    event_id: str
    supervisor_id: str
    node_id: str
    disposition: FrontierDisposition
    evidence_refs: tuple[str, ...]

    FIELD_DECODERS: ClassVar[Mapping[str, Decoder]] = MappingProxyType(
        {"binding": _binding, "disposition": FrontierDisposition, "evidence_refs": tuple}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _identifier(self.event_id, "event_id")
        _identifier(self.supervisor_id, "supervisor_id")
        _identifier(self.node_id, "node_id")
        if not isinstance(self.disposition, FrontierDisposition):
            raise FederationContractError("frontier disposition is not closed")
        _strings(self.evidence_refs, "evidence_refs", maximum=256, required=True)


CONTRACT_TYPES = (
    FederationRequest,
    FederationAuthorizationDecision,
    FederationIdentity,
    FederationPolicy,
    FederationPlan,
    FederationRevision,
    FederationState,
    FederationReceipt,
    SupervisorDefinition,
    SupervisorIdentity,
    SupervisorInstance,
    SupervisorAssignment,
    SupervisorSpecialization,
    SupervisorCapability,
    SupervisorLifecycle,
    SupervisorHealth,
    SupervisorCheckpoint,
    SupervisorReceipt,
    SubagentDefinition,
    SubagentIdentity,
    SubagentInstance,
    SubagentAssignment,
    SubagentCapability,
    SubagentBudget,
    SubagentOutcome,
    SupervisorShard,
    ShardBoundary,
    ShardAssignment,
    ShardRevision,
    ShardRebalancePlan,
    ShardRebalanceReceipt,
    FederationBudget,
    SupervisorBudget,
    AgentBudget,
    TokenBudget,
    ResourceBudget,
    BudgetReservation,
    BudgetLedger,
    FederationCommand,
    FederationCommandResult,
    FederationIdempotencyRecord,
    FederationWorldSnapshot,
    FederationFixedPoint,
    FederationCompletionReceipt,
    CapabilityBlocker,
    CausalNode,
    CausalEvidence,
    CausalEdge,
    CausalAbstractionMap,
    InterventionTest,
    CausalFrontierEntry,
)


def contract_catalog() -> Mapping[str, type[ClosedContract]]:
    return MappingProxyType({contract.SCHEMA: contract for contract in CONTRACT_TYPES})


__all__ = [
    "AbstractionFaithfulness",
    "AgentBudget",
    "BudgetDimension",
    "BudgetDimensionName",
    "BudgetLedger",
    "BudgetReservation",
    "CapabilityBlocker",
    "CapabilityBlockerCode",
    "CausalAbstractionMap",
    "CausalEdge",
    "CausalEdgeKind",
    "CausalEvidence",
    "CausalEvidenceKind",
    "CausalFrontierEntry",
    "CausalLevel",
    "CausalNode",
    "ClosedContract",
    "FederationAuthorityError",
    "FederationAuthorizationDecision",
    "FederationAuthorizationReason",
    "FederationAuthorizationVerdict",
    "FederationBinding",
    "FederationBoundsError",
    "FederationBudget",
    "FederationCommand",
    "FederationCommandResult",
    "FederationCompletionReceipt",
    "FederationContractError",
    "FederationFixedPoint",
    "FederationIdentity",
    "FederationIdempotencyRecord",
    "FederationLifecycleState",
    "FederationOperation",
    "FederationPlan",
    "FederationPolicy",
    "FederationReceipt",
    "FederationRequest",
    "FederationRevision",
    "FederationSecretError",
    "FederationState",
    "FederationWorldSnapshot",
    "FrontierDisposition",
    "InterventionTest",
    "ResourceBudget",
    "ShardAssignment",
    "ShardBoundary",
    "ShardRebalancePlan",
    "ShardRebalanceReceipt",
    "ShardRevision",
    "SubagentAssignment",
    "SubagentBudget",
    "SubagentCapability",
    "SubagentDefinition",
    "SubagentIdentity",
    "SubagentInstance",
    "SubagentOutcome",
    "SupervisorAssignment",
    "SupervisorBudget",
    "SupervisorCapability",
    "SupervisorCheckpoint",
    "SupervisorDefinition",
    "SupervisorHealth",
    "SupervisorIdentity",
    "SupervisorInstance",
    "SupervisorLifecycle",
    "SupervisorReceipt",
    "SupervisorRole",
    "SupervisorShard",
    "SupervisorSpecialization",
    "TokenBudget",
    "UnknownNormativeFieldError",
    "contract_catalog",
    "utc_now",
]
