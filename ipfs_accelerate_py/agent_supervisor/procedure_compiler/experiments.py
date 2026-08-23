"""Bounded shadow-experiment planner and isolated observation runner.

Shadow experiments are decision aids, not authorities.  A plan may run only
when world uncertainty or a family boundary exposes an explicit question, the
declared hypothesis and counterfactual would change a pending decision, and
every risk, privacy, cost, and isolation bound is already satisfied.  Execution
is confined to fixtures or authorized disposable worktrees.  Results are
observations: they cannot grant authority, proof, promotion, completion, or
validation suppression, and they never mutate production or policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol

from ..proof.formal_verification_contracts import CanonicalContract
from .contracts import (
    PROCEDURE_CONTRACT_VERSION,
    ARTIFACT_TYPES_BY_SCHEMA,
    ArtifactBindings,
    EffectClass,
    ProcedureBoundsError,
    ProcedureContractError,
    ProcedureSafetyError,
    RiskClass,
    TaskFamily,
    _bounded,
    _decode_fields,
    _enum,
    _identifier,
    _nested,
    _nonnegative_int,
    _positive_int,
    _relative_path,
    _schema_name,
    _strings,
    _text,
    _unsafe_key,
    _verify_identity,
    canonical_json_bytes,
)
from .world_model import RepositoryWorldState, WorldProjectionStatus

MAX_EXPERIMENT_REFERENCES: Final[int] = 128
MAX_EXPERIMENT_WALL_TIME_MS: Final[int] = 300_000
MAX_EXPERIMENT_CPU_TIME_MS: Final[int] = 300_000
MAX_EXPERIMENT_MEMORY_BYTES: Final[int] = 1 << 30
MAX_EXPERIMENT_DISK_BYTES: Final[int] = 1 << 30
MAX_EXPERIMENT_TOKENS: Final[int] = 100_000
MAX_EXPERIMENT_STEPS: Final[int] = 16
MAX_EXPERIMENT_OBSERVATIONS: Final[int] = 32
MAX_EXPERIMENT_PATHS: Final[int] = 32
MAX_VALUE_BPS: Final[int] = 10_000
OBSERVATION_PRODUCER_CONTRACT: Final[str] = "shadow-experiment-observation@1"
DEFAULT_MINIMUM_VALUE_BPS: Final[int] = 1

_FORBIDDEN_TARGET_PREFIXES: Final[tuple[str, ...]] = (
    ".git/",
    "config/",
    "docs/architecture/",
    "scripts/validate_agent_supervisor_procedure_compiler_board.py",
    "scripts/materialize_agent_supervisor_procedure_compiler_program.py",
    "scripts/ops/agent_supervisor/procedure_compiler_program.py",
    "scripts/ops/agent_supervisor/pcpc_external_runtime_image_v3.manifest.json",
)
_FORBIDDEN_TARGET_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "authority_policy",
        "trusted_keys",
        "trusted-keys",
        "scheduler.json",
        ".objectives.md",
        ".todo.md",
        "procedure_compiler_inventory",
    }
)
_ALLOWED_RISK: Final[frozenset[RiskClass]] = frozenset(
    {RiskClass.OBSERVATION_ONLY, RiskClass.REVERSIBLE_LOCAL}
)
_RISK_RANK: Final[dict[RiskClass, int]] = {
    RiskClass.OBSERVATION_ONLY: 0,
    RiskClass.REVERSIBLE_LOCAL: 1,
    RiskClass.REPOSITORY_WRITE: 2,
    RiskClass.PUBLIC_CONTRACT: 3,
    RiskClass.AUTHORITY_OR_SECURITY: 4,
}


class ExperimentError(ProcedureContractError):
    """A shadow experiment plan, isolation grant, or observation is invalid."""


class ExperimentBoundsError(ExperimentError, ProcedureBoundsError):
    """A shadow experiment exceeded a declared numeric or structural bound."""


class ExperimentPrivacyError(ExperimentError, ProcedureSafetyError):
    """A shadow experiment named forbidden private or executable data."""


class ExperimentIsolationError(ExperimentError):
    """A shadow experiment was not confined to a fixture or disposable worktree."""


class ExperimentAuthorityError(ExperimentError):
    """A shadow experiment was offered as authority, proof, or promotion."""


class ExperimentPrivacyClass(str, Enum):
    PUBLIC_FIXTURE = "public_fixture"
    REPOSITORY_LOCAL = "repository_local"


class ExperimentIsolationKind(str, Enum):
    FIXTURE = "fixture"
    DISPOSABLE_WORKTREE = "disposable_worktree"


class ExperimentAction(str, Enum):
    RUN = "run"
    SKIP = "skip"
    REFUSE = "refuse"


class ExperimentReasonCode(str, Enum):
    DECISION_RELEVANT = "decision_relevant"
    CANNOT_CHANGE_DECISION = "cannot_change_decision"
    NO_PENDING_DECISION = "no_pending_decision"
    QUESTION_NOT_EXPOSED = "question_not_exposed"
    NO_EXPLICIT_QUESTION = "no_explicit_question"
    HYPOTHESIS_EQUALS_COUNTERFACTUAL = "hypothesis_equals_counterfactual"
    ALREADY_DECIDED = "already_decided"
    MISSING_DECLARATION = "missing_declaration"
    BOUND_EXCEEDED = "bound_exceeded"
    UNBOUNDED = "unbounded"
    PRIVACY_VIOLATION = "privacy_violation"
    RISK_CEILING = "risk_ceiling"
    PRODUCTION_MUTATION = "production_mutation"
    POLICY_MUTATION = "policy_mutation"
    ISOLATION_REQUIRED = "isolation_required"
    UNAUTHORIZED_WORKTREE = "unauthorized_worktree"
    NON_DISPOSABLE_WORKTREE = "non_disposable_worktree"
    NETWORK_FORBIDDEN = "network_forbidden"
    ARBITRARY_EXECUTION_FORBIDDEN = "arbitrary_execution_forbidden"
    FAMILY_BOUNDARY_MISMATCH = "family_boundary_mismatch"
    BINDING_MISMATCH = "binding_mismatch"
    AUTHORITY_CLAIM = "authority_claim"


class UncertaintySource(str, Enum):
    WORLD_UNAVAILABLE_DIMENSION = "world_unavailable_dimension"
    WORLD_PROJECTION_STATUS = "world_projection_status"
    FAMILY_UNKNOWN_CASE = "family_unknown_case"
    FAMILY_BOUNDARY_CASE = "family_boundary_case"


class DecisionRuleKind(str, Enum):
    CHANGES_PENDING_DECISION = "changes_pending_decision"


class ExperimentOutcomeStatus(str, Enum):
    OBSERVED = "observed"
    SKIPPED = "skipped"
    REFUSED = "refused"
    INCOMPLETE = "incomplete"


class ObservedSupport(str, Enum):
    HYPOTHESIS = "hypothesis"
    COUNTERFACTUAL = "counterfactual"
    INCONCLUSIVE = "inconclusive"
    NONE = "none"


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ExperimentError(f"{field_name} must be a boolean")
    return value


def _bps(value: Any, field_name: str) -> int:
    return _nonnegative_int(value, field_name, maximum=MAX_VALUE_BPS)


def _ids(values: Any, field_name: str, *, required: bool = False) -> tuple[str, ...]:
    return _strings(
        values,
        field_name,
        limit=MAX_EXPERIMENT_REFERENCES,
        identifiers=True,
        required=required,
        preserve_order=True,
    )


def _paths(values: Any, field_name: str, *, required: bool = False) -> tuple[str, ...]:
    return _strings(
        values,
        field_name,
        limit=MAX_EXPERIMENT_PATHS,
        paths=True,
        required=required,
        preserve_order=True,
    )


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _privacy_identifier(value: str, field_name: str) -> str:
    if _unsafe_key(value):
        raise ExperimentPrivacyError(
            f"{field_name} names forbidden secret, credential, or executable data"
        )
    return value


def _is_forbidden_target(path: str) -> bool:
    normalized = _relative_path(path, "target_path")
    if any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in _FORBIDDEN_TARGET_PREFIXES
    ):
        return True
    lowered = normalized.lower()
    return any(marker in lowered for marker in _FORBIDDEN_TARGET_MARKERS)


def _reject_forbidden_targets(paths: Sequence[str], field_name: str) -> tuple[str, ...]:
    normalized = _paths(paths, field_name)
    forbidden = tuple(path for path in normalized if _is_forbidden_target(path))
    if forbidden:
        raise ExperimentIsolationError(
            f"{field_name} includes production or policy targets: {forbidden[0]}"
        )
    return normalized


def _contained_path(root: Path, relative: str) -> Path:
    normalized = _relative_path(relative, "isolated_path")
    base = root.resolve()
    candidate = (base / normalized).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ExperimentIsolationError("isolated path escaped its disposable root") from exc
    return candidate


def _payload_fields(cls: type[Any]) -> tuple[str, ...]:
    return tuple(name for name in cls.__dataclass_fields__ if name != "SCHEMA")  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ExperimentCost(CanonicalContract):
    """Integer cost envelope declared before a shadow experiment may run."""

    SCHEMA: ClassVar[str] = _schema_name("ExperimentCost")

    tokens: int = 0
    wall_time_ms: int = 0
    cpu_time_ms: int = 0
    disk_bytes: int = 0
    worktree_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tokens", _nonnegative_int(self.tokens, "tokens", maximum=MAX_EXPERIMENT_TOKENS)
        )
        object.__setattr__(
            self,
            "wall_time_ms",
            _nonnegative_int(self.wall_time_ms, "wall_time_ms", maximum=MAX_EXPERIMENT_WALL_TIME_MS),
        )
        object.__setattr__(
            self,
            "cpu_time_ms",
            _nonnegative_int(self.cpu_time_ms, "cpu_time_ms", maximum=MAX_EXPERIMENT_CPU_TIME_MS),
        )
        object.__setattr__(
            self,
            "disk_bytes",
            _nonnegative_int(self.disk_bytes, "disk_bytes", maximum=MAX_EXPERIMENT_DISK_BYTES),
        )
        object.__setattr__(
            self,
            "worktree_count",
            _nonnegative_int(self.worktree_count, "worktree_count", maximum=MAX_EXPERIMENT_PATHS),
        )
        _bounded(self, "ExperimentCost")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "tokens": self.tokens,
            "wall_time_ms": self.wall_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "disk_bytes": self.disk_bytes,
            "worktree_count": self.worktree_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentCost:
        fields = _payload_fields(cls)
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ExperimentExecutionBounds(CanonicalContract):
    """Fixed, finite execution bounds.  Network and subprocesses are forbidden."""

    SCHEMA: ClassVar[str] = _schema_name("ExperimentExecutionBounds")

    wall_time_ms: int
    cpu_time_ms: int
    memory_bytes: int
    disk_bytes: int
    token_limit: int
    step_limit: int
    observation_limit: int
    path_limit: int
    network_request_limit: int = 0
    subprocess_limit: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wall_time_ms",
            _positive_int(self.wall_time_ms, "wall_time_ms", maximum=MAX_EXPERIMENT_WALL_TIME_MS),
        )
        object.__setattr__(
            self,
            "cpu_time_ms",
            _positive_int(self.cpu_time_ms, "cpu_time_ms", maximum=MAX_EXPERIMENT_CPU_TIME_MS),
        )
        object.__setattr__(
            self,
            "memory_bytes",
            _positive_int(self.memory_bytes, "memory_bytes", maximum=MAX_EXPERIMENT_MEMORY_BYTES),
        )
        object.__setattr__(
            self,
            "disk_bytes",
            _positive_int(self.disk_bytes, "disk_bytes", maximum=MAX_EXPERIMENT_DISK_BYTES),
        )
        object.__setattr__(
            self,
            "token_limit",
            _nonnegative_int(self.token_limit, "token_limit", maximum=MAX_EXPERIMENT_TOKENS),
        )
        object.__setattr__(
            self,
            "step_limit",
            _positive_int(self.step_limit, "step_limit", maximum=MAX_EXPERIMENT_STEPS),
        )
        object.__setattr__(
            self,
            "observation_limit",
            _positive_int(
                self.observation_limit,
                "observation_limit",
                maximum=MAX_EXPERIMENT_OBSERVATIONS,
            ),
        )
        object.__setattr__(
            self,
            "path_limit",
            _positive_int(self.path_limit, "path_limit", maximum=MAX_EXPERIMENT_PATHS),
        )
        object.__setattr__(
            self,
            "network_request_limit",
            _nonnegative_int(self.network_request_limit, "network_request_limit", maximum=0),
        )
        object.__setattr__(
            self,
            "subprocess_limit",
            _nonnegative_int(self.subprocess_limit, "subprocess_limit", maximum=0),
        )
        _bounded(self, "ExperimentExecutionBounds")

    def covers(self, cost: ExperimentCost, *, path_count: int) -> bool:
        return (
            cost.tokens <= self.token_limit
            and cost.wall_time_ms <= self.wall_time_ms
            and cost.cpu_time_ms <= self.cpu_time_ms
            and cost.disk_bytes <= self.disk_bytes
            and path_count <= self.path_limit
            and cost.worktree_count <= 1
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "wall_time_ms": self.wall_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_bytes": self.memory_bytes,
            "disk_bytes": self.disk_bytes,
            "token_limit": self.token_limit,
            "step_limit": self.step_limit,
            "observation_limit": self.observation_limit,
            "path_limit": self.path_limit,
            "network_request_limit": self.network_request_limit,
            "subprocess_limit": self.subprocess_limit,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentExecutionBounds:
        fields = _payload_fields(cls)
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ExperimentDecisionRule(CanonicalContract):
    """Closed rule that names the pending decision an experiment could change."""

    SCHEMA: ClassVar[str] = _schema_name("ExperimentDecisionRule")

    kind: DecisionRuleKind
    pending_decision_id: str
    hypothesis_action: str
    counterfactual_action: str
    expected_decision_change_bps: int = DEFAULT_MINIMUM_VALUE_BPS

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, DecisionRuleKind, "kind"))
        object.__setattr__(
            self,
            "pending_decision_id",
            _identifier(self.pending_decision_id, "pending_decision_id"),
        )
        object.__setattr__(
            self,
            "hypothesis_action",
            _identifier(self.hypothesis_action, "hypothesis_action"),
        )
        object.__setattr__(
            self,
            "counterfactual_action",
            _identifier(self.counterfactual_action, "counterfactual_action"),
        )
        object.__setattr__(
            self,
            "expected_decision_change_bps",
            _bps(self.expected_decision_change_bps, "expected_decision_change_bps"),
        )
        _bounded(self, "ExperimentDecisionRule")

    @property
    def distinguishes_actions(self) -> bool:
        return self.hypothesis_action != self.counterfactual_action

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "kind": self.kind.value,
            "pending_decision_id": self.pending_decision_id,
            "hypothesis_action": self.hypothesis_action,
            "counterfactual_action": self.counterfactual_action,
            "expected_decision_change_bps": self.expected_decision_change_bps,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentDecisionRule:
        fields = _payload_fields(cls)
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class PendingDecision(CanonicalContract):
    """An open supervisor decision that a shadow experiment might change."""

    SCHEMA: ClassVar[str] = _schema_name("PendingDecision")

    decision_id: str
    alternatives: tuple[str, ...]
    default_action: str
    open: bool = True
    admitted_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _identifier(self.decision_id, "decision_id"))
        alternatives = _ids(self.alternatives, "alternatives", required=True)
        if len(alternatives) < 2:
            raise ExperimentError("pending decisions must declare at least two alternatives")
        object.__setattr__(self, "alternatives", alternatives)
        object.__setattr__(
            self, "default_action", _identifier(self.default_action, "default_action")
        )
        if self.default_action not in self.alternatives:
            raise ExperimentError("default_action must be one of the declared alternatives")
        object.__setattr__(self, "open", _bool(self.open, "open"))
        object.__setattr__(
            self,
            "admitted_evidence_ids",
            _ids(self.admitted_evidence_ids, "admitted_evidence_ids"),
        )
        _bounded(self, "PendingDecision")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "decision_id": self.decision_id,
            "alternatives": self.alternatives,
            "default_action": self.default_action,
            "open": self.open,
            "admitted_evidence_ids": self.admitted_evidence_ids,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PendingDecision:
        fields = _payload_fields(cls)
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ExplicitQuestion(CanonicalContract):
    """One uncertainty question exposed by world state or a family boundary."""

    SCHEMA: ClassVar[str] = _schema_name("ExplicitQuestion")

    question_id: str
    source: UncertaintySource
    statement: str
    subject_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _identifier(self.question_id, "question_id"))
        object.__setattr__(self, "source", _enum(self.source, UncertaintySource, "source"))
        object.__setattr__(self, "statement", _text(self.statement, "statement"))
        object.__setattr__(self, "subject_id", _identifier(self.subject_id, "subject_id"))
        _bounded(self, "ExplicitQuestion")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "question_id": self.question_id,
            "source": self.source.value,
            "statement": self.statement,
            "subject_id": self.subject_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExplicitQuestion:
        fields = _payload_fields(cls)
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ExperimentIsolation(CanonicalContract):
    """Externally attested isolation; experiments cannot self-authorize it."""

    SCHEMA: ClassVar[str] = _schema_name("ExperimentIsolation")

    kind: ExperimentIsolationKind
    authorized: bool
    disposable: bool
    production: bool = False
    policy_mutable: bool = False
    scope_paths: tuple[str, ...] = ()
    fixture_id: str = ""
    worktree_id: str = ""
    grant_receipt_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, ExperimentIsolationKind, "kind"))
        object.__setattr__(self, "authorized", _bool(self.authorized, "authorized"))
        object.__setattr__(self, "disposable", _bool(self.disposable, "disposable"))
        object.__setattr__(self, "production", _bool(self.production, "production"))
        object.__setattr__(
            self, "policy_mutable", _bool(self.policy_mutable, "policy_mutable")
        )
        object.__setattr__(self, "scope_paths", _paths(self.scope_paths, "scope_paths"))
        object.__setattr__(
            self, "fixture_id", _identifier(self.fixture_id, "fixture_id", required=False)
        )
        object.__setattr__(
            self,
            "worktree_id",
            _identifier(self.worktree_id, "worktree_id", required=False),
        )
        object.__setattr__(
            self,
            "grant_receipt_cid",
            _identifier(self.grant_receipt_cid, "grant_receipt_cid", required=False),
        )
        if self.kind is ExperimentIsolationKind.FIXTURE and not self.fixture_id:
            raise ExperimentIsolationError("fixture isolation requires fixture_id")
        _bounded(self, "ExperimentIsolation")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "kind": self.kind.value,
            "authorized": self.authorized,
            "disposable": self.disposable,
            "production": self.production,
            "policy_mutable": self.policy_mutable,
            "scope_paths": self.scope_paths,
            "fixture_id": self.fixture_id,
            "worktree_id": self.worktree_id,
            "grant_receipt_cid": self.grant_receipt_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentIsolation:
        fields = _payload_fields(cls)
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ShadowExperimentSpec(CanonicalContract):
    """Complete shadow-experiment declaration required before planning."""

    SCHEMA: ClassVar[str] = _schema_name("ShadowExperimentSpec")

    bindings: ArtifactBindings
    question_id: str
    question: str
    hypothesis: str
    counterfactual: str
    required_data_ids: tuple[str, ...]
    risk_class: RiskClass
    privacy_class: ExperimentPrivacyClass
    cost: ExperimentCost
    decision_rule: ExperimentDecisionRule
    bounds: ExperimentExecutionBounds
    isolation_kind: ExperimentIsolationKind
    scope_paths: tuple[str, ...] = ()
    family_cid: str = ""
    world_state_id: str = ""
    fixture_id: str = ""
    worktree_id: str = ""
    hypothesis_expected: str = ""
    counterfactual_expected: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "question_id", _identifier(self.question_id, "question_id"))
        object.__setattr__(self, "question", _text(self.question, "question"))
        object.__setattr__(self, "hypothesis", _text(self.hypothesis, "hypothesis"))
        object.__setattr__(
            self, "counterfactual", _text(self.counterfactual, "counterfactual")
        )
        data_ids = _ids(self.required_data_ids, "required_data_ids", required=True)
        for item in data_ids:
            _privacy_identifier(item, "required_data_ids")
        object.__setattr__(self, "required_data_ids", data_ids)
        object.__setattr__(self, "risk_class", _enum(self.risk_class, RiskClass, "risk_class"))
        object.__setattr__(
            self,
            "privacy_class",
            _enum(self.privacy_class, ExperimentPrivacyClass, "privacy_class"),
        )
        object.__setattr__(self, "cost", _nested(self.cost, ExperimentCost, "cost"))
        object.__setattr__(
            self,
            "decision_rule",
            _nested(self.decision_rule, ExperimentDecisionRule, "decision_rule"),
        )
        object.__setattr__(
            self, "bounds", _nested(self.bounds, ExperimentExecutionBounds, "bounds")
        )
        object.__setattr__(
            self,
            "isolation_kind",
            _enum(self.isolation_kind, ExperimentIsolationKind, "isolation_kind"),
        )
        object.__setattr__(self, "scope_paths", _paths(self.scope_paths, "scope_paths"))
        if len(self.scope_paths) > self.bounds.path_limit:
            raise ExperimentBoundsError("scope_paths exceed the declared path bound")
        object.__setattr__(
            self, "family_cid", _identifier(self.family_cid, "family_cid", required=False)
        )
        object.__setattr__(
            self,
            "world_state_id",
            _identifier(self.world_state_id, "world_state_id", required=False),
        )
        object.__setattr__(
            self, "fixture_id", _identifier(self.fixture_id, "fixture_id", required=False)
        )
        object.__setattr__(
            self,
            "worktree_id",
            _identifier(self.worktree_id, "worktree_id", required=False),
        )
        object.__setattr__(
            self,
            "hypothesis_expected",
            _text(self.hypothesis_expected, "hypothesis_expected", required=False),
        )
        object.__setattr__(
            self,
            "counterfactual_expected",
            _text(
                self.counterfactual_expected,
                "counterfactual_expected",
                required=False,
            ),
        )
        if self.isolation_kind is ExperimentIsolationKind.FIXTURE and not self.fixture_id:
            raise ExperimentIsolationError("fixture experiments must declare fixture_id")
        if (
            self.isolation_kind is ExperimentIsolationKind.DISPOSABLE_WORKTREE
            and not self.scope_paths
        ):
            raise ExperimentIsolationError(
                "disposable worktree experiments must declare scope_paths"
            )
        if self.cost.worktree_count and (
            self.isolation_kind is not ExperimentIsolationKind.DISPOSABLE_WORKTREE
        ):
            raise ExperimentIsolationError(
                "worktree cost is only valid for disposable worktree isolation"
            )
        _bounded(self, "ShadowExperimentSpec")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "question_id": self.question_id,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "counterfactual": self.counterfactual,
            "required_data_ids": self.required_data_ids,
            "risk_class": self.risk_class.value,
            "privacy_class": self.privacy_class.value,
            "cost": self.cost,
            "decision_rule": self.decision_rule,
            "bounds": self.bounds,
            "isolation_kind": self.isolation_kind.value,
            "scope_paths": self.scope_paths,
            "family_cid": self.family_cid,
            "world_state_id": self.world_state_id,
            "fixture_id": self.fixture_id,
            "worktree_id": self.worktree_id,
            "hypothesis_expected": self.hypothesis_expected,
            "counterfactual_expected": self.counterfactual_expected,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ShadowExperimentSpec:
        fields = _payload_fields(cls)
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if "cost" in values:
            values["cost"] = _nested(values["cost"], ExperimentCost, "cost")
        if "decision_rule" in values:
            values["decision_rule"] = _nested(
                values["decision_rule"], ExperimentDecisionRule, "decision_rule"
            )
        if "bounds" in values:
            values["bounds"] = _nested(values["bounds"], ExperimentExecutionBounds, "bounds")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ExperimentDecision(CanonicalContract):
    """Planner verdict.  Running remains observation-only and non-authoritative."""

    SCHEMA: ClassVar[str] = _schema_name("ExperimentDecision")

    spec_id: str
    action: ExperimentAction
    reason_code: ExperimentReasonCode
    decision_relevant: bool
    matched_question_id: str = ""
    pending_decision_id: str = ""
    expected_decision_change_bps: int = 0
    isolation_kind: ExperimentIsolationKind = ExperimentIsolationKind.FIXTURE
    observation_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec_id", _identifier(self.spec_id, "spec_id"))
        object.__setattr__(self, "action", _enum(self.action, ExperimentAction, "action"))
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, ExperimentReasonCode, "reason_code")
        )
        object.__setattr__(
            self, "decision_relevant", _bool(self.decision_relevant, "decision_relevant")
        )
        object.__setattr__(
            self,
            "matched_question_id",
            _identifier(self.matched_question_id, "matched_question_id", required=False),
        )
        object.__setattr__(
            self,
            "pending_decision_id",
            _identifier(self.pending_decision_id, "pending_decision_id", required=False),
        )
        object.__setattr__(
            self,
            "expected_decision_change_bps",
            _bps(self.expected_decision_change_bps, "expected_decision_change_bps"),
        )
        object.__setattr__(
            self,
            "isolation_kind",
            _enum(self.isolation_kind, ExperimentIsolationKind, "isolation_kind"),
        )
        object.__setattr__(
            self, "observation_only", _bool(self.observation_only, "observation_only")
        )
        if self.observation_only is not True:
            raise ExperimentAuthorityError("experiment decisions must remain observation-only")
        if self.action is ExperimentAction.RUN:
            if not self.decision_relevant:
                raise ExperimentError("a run decision must be decision-relevant")
            if self.reason_code is not ExperimentReasonCode.DECISION_RELEVANT:
                raise ExperimentError("a run decision must use the decision-relevant reason")
            if not self.matched_question_id or not self.pending_decision_id:
                raise ExperimentError("a run decision must bind question and pending decision")
            if self.expected_decision_change_bps <= 0:
                raise ExperimentError("a run decision must have positive decision value")
        elif self.decision_relevant:
            raise ExperimentError("skip and refuse decisions cannot be decision-relevant")
        _bounded(self, "ExperimentDecision")

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_establish_proof(self) -> bool:
        return False

    @property
    def can_establish_postcondition(self) -> bool:
        return False

    @property
    def can_establish_completion(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    @property
    def can_suppress_validation(self) -> bool:
        return False

    @property
    def is_authoritative(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "spec_id": self.spec_id,
            "action": self.action.value,
            "reason_code": self.reason_code.value,
            "decision_relevant": self.decision_relevant,
            "matched_question_id": self.matched_question_id,
            "pending_decision_id": self.pending_decision_id,
            "expected_decision_change_bps": self.expected_decision_change_bps,
            "isolation_kind": self.isolation_kind.value,
            "observation_only": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentDecision:
        fields = _payload_fields(cls)
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ExperimentEffect(CanonicalContract):
    """Declared isolated effect.  Production and policy targets are rejected."""

    SCHEMA: ClassVar[str] = _schema_name("ExperimentEffect")

    effect_id: str
    effect_class: EffectClass
    targets: tuple[str, ...] = ()
    reversible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect_id", _identifier(self.effect_id, "effect_id"))
        object.__setattr__(
            self, "effect_class", _enum(self.effect_class, EffectClass, "effect_class")
        )
        if self.effect_class not in {EffectClass.OBSERVE, EffectClass.WORKTREE_CREATE}:
            raise ExperimentAuthorityError(
                "shadow experiments may only observe or create disposable worktrees"
            )
        object.__setattr__(self, "targets", _reject_forbidden_targets(self.targets, "targets"))
        object.__setattr__(self, "reversible", _bool(self.reversible, "reversible"))
        if self.effect_class is EffectClass.WORKTREE_CREATE and not self.reversible:
            raise ExperimentIsolationError("worktree experiment effects must be reversible")
        _bounded(self, "ExperimentEffect")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "effect_id": self.effect_id,
            "effect_class": self.effect_class.value,
            "targets": self.targets,
            "reversible": self.reversible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentEffect:
        fields = _payload_fields(cls)
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ShadowExperimentObservation(CanonicalContract):
    """Persisted observation.  Confirmation of a hypothesis still cannot authorize."""

    SCHEMA: ClassVar[str] = _schema_name("ShadowExperimentObservation")

    bindings: ArtifactBindings
    experiment_id: str
    decision_id: str
    question_id: str
    producer_contract: str
    observed_values: Mapping[str, str]
    support: ObservedSupport
    isolation_kind: ExperimentIsolationKind
    isolation_root: str = ""
    worktree_id: str = ""
    fixture_id: str = ""
    effects: tuple[ExperimentEffect, ...] = ()
    missing_data_ids: tuple[str, ...] = ()
    observation_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self, "experiment_id", _identifier(self.experiment_id, "experiment_id")
        )
        object.__setattr__(self, "decision_id", _identifier(self.decision_id, "decision_id"))
        object.__setattr__(self, "question_id", _identifier(self.question_id, "question_id"))
        object.__setattr__(
            self,
            "producer_contract",
            _identifier(self.producer_contract, "producer_contract"),
        )
        if self.producer_contract != OBSERVATION_PRODUCER_CONTRACT:
            raise ExperimentError("observation producer contract is not the shadow runner")
        values = self.observed_values
        if values is None:
            frozen: dict[str, str] = {}
        elif isinstance(values, Mapping):
            if len(values) > MAX_EXPERIMENT_REFERENCES:
                raise ExperimentBoundsError("observed_values exceeds its item bound")
            frozen = {}
            for raw_key in values:
                key = _privacy_identifier(_identifier(raw_key, "observed_values"), "observed_values")
                frozen[key] = _text(values[raw_key], "observed_values", required=False)
        else:
            raise ExperimentError("observed_values must be a mapping")
        object.__setattr__(self, "observed_values", MappingProxyType(frozen))
        object.__setattr__(self, "support", _enum(self.support, ObservedSupport, "support"))
        object.__setattr__(
            self,
            "isolation_kind",
            _enum(self.isolation_kind, ExperimentIsolationKind, "isolation_kind"),
        )
        object.__setattr__(
            self,
            "isolation_root",
            _text(self.isolation_root, "isolation_root", required=False),
        )
        if self.isolation_root:
            root = PurePosixPath(self.isolation_root)
            if root.is_absolute() or ".." in root.parts or "\\" in self.isolation_root:
                raise ExperimentIsolationError("isolation_root must stay repository-relative")
        object.__setattr__(
            self, "worktree_id", _identifier(self.worktree_id, "worktree_id", required=False)
        )
        object.__setattr__(
            self, "fixture_id", _identifier(self.fixture_id, "fixture_id", required=False)
        )
        effects: list[ExperimentEffect] = []
        raw_effects = self.effects or ()
        if not isinstance(raw_effects, Sequence) or isinstance(
            raw_effects, (str, bytes, bytearray, memoryview)
        ):
            raise ExperimentError("effects must be a sequence")
        if len(raw_effects) > MAX_EXPERIMENT_OBSERVATIONS:
            raise ExperimentBoundsError("effects exceed their item bound")
        for item in raw_effects:
            effects.append(_nested(item, ExperimentEffect, "effects"))
        object.__setattr__(self, "effects", tuple(effects))
        object.__setattr__(
            self, "missing_data_ids", _ids(self.missing_data_ids, "missing_data_ids")
        )
        object.__setattr__(
            self, "observation_only", _bool(self.observation_only, "observation_only")
        )
        if self.observation_only is not True:
            raise ExperimentAuthorityError("shadow observations must remain observation-only")
        _bounded(self, "ShadowExperimentObservation")

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_establish_proof(self) -> bool:
        return False

    @property
    def can_establish_postcondition(self) -> bool:
        return False

    @property
    def can_establish_completion(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    @property
    def is_authoritative(self) -> bool:
        return False

    def authorize(self) -> None:
        raise ExperimentAuthorityError("shadow experiment observations cannot authorize")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "experiment_id": self.experiment_id,
            "decision_id": self.decision_id,
            "question_id": self.question_id,
            "producer_contract": self.producer_contract,
            "observed_values": dict(self.observed_values),
            "support": self.support.value,
            "isolation_kind": self.isolation_kind.value,
            "isolation_root": self.isolation_root,
            "worktree_id": self.worktree_id,
            "fixture_id": self.fixture_id,
            "effects": self.effects,
            "missing_data_ids": self.missing_data_ids,
            "observation_only": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ShadowExperimentObservation:
        fields = _payload_fields(cls)
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if "effects" in values:
            values["effects"] = tuple(
                _nested(item, ExperimentEffect, "effects") for item in values["effects"]
            )
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ShadowExperimentResult:
    """Runner outcome.  Skipped and observed results stay non-authoritative."""

    status: ExperimentOutcomeStatus
    decision: ExperimentDecision
    observation: ShadowExperimentObservation | None = None
    reason_code: ExperimentReasonCode = ExperimentReasonCode.DECISION_RELEVANT

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum(self.status, ExperimentOutcomeStatus, "status")
        )
        if not isinstance(self.decision, ExperimentDecision):
            raise ExperimentError("result must bind an ExperimentDecision")
        if self.observation is not None and not isinstance(
            self.observation, ShadowExperimentObservation
        ):
            raise ExperimentError("result observation has an unsupported type")
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, ExperimentReasonCode, "reason_code")
        )
        if self.status is ExperimentOutcomeStatus.OBSERVED and self.observation is None:
            raise ExperimentError("observed results must persist an observation")
        if self.status is ExperimentOutcomeStatus.SKIPPED and self.observation is not None:
            raise ExperimentError("skipped experiments cannot persist observations")

    @property
    def observation_only(self) -> bool:
        return True

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    @property
    def is_authoritative(self) -> bool:
        return False

    def authorize(self) -> None:
        raise ExperimentAuthorityError("shadow experiment results cannot authorize")


@dataclass(frozen=True)
class ExperimentFixture:
    """Authorized fixture payload.  Production fixtures are refused at runtime."""

    fixture_id: str
    values: Mapping[str, str] = MappingProxyType({})
    root_path: str = ""
    authorized: bool = True
    production: bool = False
    policy_mutable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixture_id", _identifier(self.fixture_id, "fixture_id"))
        raw = self.values or {}
        if not isinstance(raw, Mapping):
            raise ExperimentError("fixture values must be a mapping")
        if len(raw) > MAX_EXPERIMENT_REFERENCES:
            raise ExperimentBoundsError("fixture values exceed their item bound")
        frozen: dict[str, str] = {}
        for raw_key in raw:
            key = _privacy_identifier(_identifier(raw_key, "fixture_values"), "fixture_values")
            frozen[key] = _text(raw[raw_key], "fixture_values", required=False)
        object.__setattr__(self, "values", MappingProxyType(frozen))
        object.__setattr__(self, "root_path", _text(self.root_path, "root_path", required=False))
        object.__setattr__(self, "authorized", _bool(self.authorized, "authorized"))
        object.__setattr__(self, "production", _bool(self.production, "production"))
        object.__setattr__(
            self, "policy_mutable", _bool(self.policy_mutable, "policy_mutable")
        )


@dataclass(frozen=True)
class DisposableWorktreeGrant:
    """Grant from existing isolated-worktree authority, never self-issued."""

    worktree_id: str
    reservation_id: str
    receipt_cid: str
    scope_paths: tuple[str, ...]
    disposable: bool = True
    authorized: bool = True
    production: bool = False
    policy_mutable: bool = False
    lease_id: str = ""
    fencing_token: int = 0
    root_path: str = ""
    read_only: bool = True

    def __post_init__(self) -> None:
        for name in ("worktree_id", "reservation_id", "receipt_cid"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "scope_paths", _reject_forbidden_targets(self.scope_paths, "scope_paths")
        )
        if not self.scope_paths:
            raise ExperimentIsolationError("disposable worktree grants require scope_paths")
        for name in ("disposable", "authorized", "production", "policy_mutable", "read_only"):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(self, "lease_id", _identifier(self.lease_id, "lease_id", required=False))
        object.__setattr__(
            self,
            "fencing_token",
            _nonnegative_int(self.fencing_token, "fencing_token"),
        )
        object.__setattr__(self, "root_path", _text(self.root_path, "root_path", required=False))
        if not self.read_only and (not self.lease_id or self.fencing_token <= 0):
            raise ExperimentIsolationError(
                "effectful disposable worktrees require lease and fence evidence"
            )


@dataclass(frozen=True)
class DisposableWorktreeRequest:
    experiment_id: str
    repository_id: str
    tree_id: str
    scope_paths: tuple[str, ...]
    read_only: bool

    def __post_init__(self) -> None:
        for name in ("experiment_id", "repository_id", "tree_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "scope_paths", _reject_forbidden_targets(self.scope_paths, "scope_paths")
        )
        object.__setattr__(self, "read_only", _bool(self.read_only, "read_only"))


class DisposableWorktreePort(Protocol):
    """Adapter to existing isolated-worktree authority; this module does not own git."""

    def acquire(self, request: DisposableWorktreeRequest) -> DisposableWorktreeGrant: ...

    def release(self, grant: DisposableWorktreeGrant) -> None: ...


class InMemoryShadowObservationStore:
    """Process-local observation persistence.  Not an admission or authority store."""

    def __init__(self) -> None:
        self._records: dict[str, ShadowExperimentObservation] = {}

    def persist(self, observation: ShadowExperimentObservation) -> ShadowExperimentObservation:
        if not isinstance(observation, ShadowExperimentObservation):
            raise ExperimentError("store can persist only shadow experiment observations")
        if observation.can_grant_authority or not observation.observation_only:
            raise ExperimentAuthorityError("store refuses authoritative experiment records")
        existing = self._records.get(observation.experiment_id)
        if existing is not None and existing.content_id != observation.content_id:
            raise ExperimentError("observation identity for this experiment already differs")
        self._records[observation.experiment_id] = observation
        return observation

    def get(self, experiment_id: str) -> ShadowExperimentObservation | None:
        return self._records.get(_identifier(experiment_id, "experiment_id"))

    def records(self) -> tuple[ShadowExperimentObservation, ...]:
        return tuple(self._records.values())


class InMemoryDisposableWorktreePort:
    """Hermetic disposable-worktree double that never touches production git."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self.acquired: list[DisposableWorktreeGrant] = []
        self.released: list[str] = []

    def acquire(self, request: DisposableWorktreeRequest) -> DisposableWorktreeGrant:
        if not isinstance(request, DisposableWorktreeRequest):
            raise ExperimentIsolationError("worktree acquire requires DisposableWorktreeRequest")
        worktree_id = "wt-" + request.experiment_id
        path = self._root / worktree_id
        path.mkdir(parents=True, exist_ok=True)
        grant = DisposableWorktreeGrant(
            worktree_id=worktree_id,
            reservation_id="res-" + request.experiment_id,
            receipt_cid="receipt-" + request.experiment_id,
            scope_paths=request.scope_paths,
            disposable=True,
            authorized=True,
            production=False,
            policy_mutable=False,
            lease_id="lease-" + request.experiment_id,
            fencing_token=1 if not request.read_only else 0,
            root_path=str(path),
            read_only=request.read_only,
        )
        self.acquired.append(grant)
        return grant

    def release(self, grant: DisposableWorktreeGrant) -> None:
        if not isinstance(grant, DisposableWorktreeGrant):
            raise ExperimentIsolationError("worktree release requires DisposableWorktreeGrant")
        self.released.append(grant.worktree_id)


def questions_from_world(world: RepositoryWorldState) -> tuple[ExplicitQuestion, ...]:
    """Expose unavailable dimensions and non-current projection status as questions."""

    if not isinstance(world, RepositoryWorldState):
        raise ExperimentError("world questions require RepositoryWorldState")
    questions: list[ExplicitQuestion] = []
    for dimension in world.unavailable_dimensions:
        questions.append(
            ExplicitQuestion(
                question_id=f"world.unavailable.{dimension}",
                source=UncertaintySource.WORLD_UNAVAILABLE_DIMENSION,
                statement=f"world dimension {dimension} is unavailable",
                subject_id=dimension,
            )
        )
    if world.projection_status is not WorldProjectionStatus.CURRENT:
        status = world.projection_status.value
        questions.append(
            ExplicitQuestion(
                question_id=f"world.projection.{status}",
                source=UncertaintySource.WORLD_PROJECTION_STATUS,
                statement=f"world projection status is {status}",
                subject_id=status,
            )
        )
    return tuple(questions)


def questions_from_family(family: TaskFamily) -> tuple[ExplicitQuestion, ...]:
    """Expose declared unknown and boundary cases as explicit questions."""

    if not isinstance(family, TaskFamily):
        raise ExperimentError("family questions require TaskFamily")
    questions: list[ExplicitQuestion] = []
    for example_cid in family.boundary.unknown_case_cids:
        questions.append(
            ExplicitQuestion(
                question_id=f"family.unknown.{example_cid}",
                source=UncertaintySource.FAMILY_UNKNOWN_CASE,
                statement=f"task family unknown case {example_cid}",
                subject_id=example_cid,
            )
        )
    for example_cid in family.boundary.boundary_example_cids:
        questions.append(
            ExplicitQuestion(
                question_id=f"family.boundary.{example_cid}",
                source=UncertaintySource.FAMILY_BOUNDARY_CASE,
                statement=f"task family boundary case {example_cid}",
                subject_id=example_cid,
            )
        )
    return tuple(questions)


class ExperimentPlanner:
    """Fail-closed decision-value planner for bounded shadow experiments."""

    def __init__(self, *, minimum_value_bps: int = DEFAULT_MINIMUM_VALUE_BPS) -> None:
        self.minimum_value_bps = _positive_int(
            minimum_value_bps, "minimum_value_bps", maximum=MAX_VALUE_BPS
        )

    def exposed_questions(
        self,
        *,
        world: RepositoryWorldState | None = None,
        family: TaskFamily | None = None,
    ) -> tuple[ExplicitQuestion, ...]:
        questions: list[ExplicitQuestion] = []
        if world is not None:
            questions.extend(questions_from_world(world))
        if family is not None:
            questions.extend(questions_from_family(family))
        return tuple(questions)

    def plan(
        self,
        spec: ShadowExperimentSpec,
        *,
        isolation: ExperimentIsolation,
        world: RepositoryWorldState | None = None,
        family: TaskFamily | None = None,
        pending_decisions: Sequence[PendingDecision] = (),
    ) -> ExperimentDecision:
        if not isinstance(spec, ShadowExperimentSpec):
            raise ExperimentError("planner requires ShadowExperimentSpec")
        if not isinstance(isolation, ExperimentIsolation):
            raise ExperimentError("planner requires externally attested ExperimentIsolation")

        refuse = self._refuse_context(spec, isolation, world, family)
        if refuse is not None:
            return refuse

        questions = self.exposed_questions(world=world, family=family)
        if not questions:
            return self._decision(
                spec,
                ExperimentAction.SKIP,
                ExperimentReasonCode.NO_EXPLICIT_QUESTION,
            )
        matched = next((item for item in questions if item.question_id == spec.question_id), None)
        if matched is None:
            return self._decision(
                spec,
                ExperimentAction.SKIP,
                ExperimentReasonCode.QUESTION_NOT_EXPOSED,
            )

        if spec.hypothesis == spec.counterfactual or not spec.decision_rule.distinguishes_actions:
            return self._decision(
                spec,
                ExperimentAction.SKIP,
                ExperimentReasonCode.HYPOTHESIS_EQUALS_COUNTERFACTUAL,
                matched_question_id=matched.question_id,
            )

        pending = self._pending_decision(spec, pending_decisions)
        if pending is None:
            return self._decision(
                spec,
                ExperimentAction.SKIP,
                ExperimentReasonCode.NO_PENDING_DECISION,
                matched_question_id=matched.question_id,
            )
        if (not pending.open) or pending.admitted_evidence_ids:
            return self._decision(
                spec,
                ExperimentAction.SKIP,
                ExperimentReasonCode.ALREADY_DECIDED,
                matched_question_id=matched.question_id,
                pending_decision_id=pending.decision_id,
            )
        rule = spec.decision_rule
        if (
            rule.hypothesis_action not in pending.alternatives
            or rule.counterfactual_action not in pending.alternatives
        ):
            return self._decision(
                spec,
                ExperimentAction.SKIP,
                ExperimentReasonCode.CANNOT_CHANGE_DECISION,
                matched_question_id=matched.question_id,
                pending_decision_id=pending.decision_id,
            )
        if rule.expected_decision_change_bps < self.minimum_value_bps:
            return self._decision(
                spec,
                ExperimentAction.SKIP,
                ExperimentReasonCode.CANNOT_CHANGE_DECISION,
                matched_question_id=matched.question_id,
                pending_decision_id=pending.decision_id,
            )
        return ExperimentDecision(
            spec_id=spec.content_id,
            action=ExperimentAction.RUN,
            reason_code=ExperimentReasonCode.DECISION_RELEVANT,
            decision_relevant=True,
            matched_question_id=matched.question_id,
            pending_decision_id=pending.decision_id,
            expected_decision_change_bps=rule.expected_decision_change_bps,
            isolation_kind=spec.isolation_kind,
        )

    def _pending_decision(
        self,
        spec: ShadowExperimentSpec,
        pending_decisions: Sequence[PendingDecision],
    ) -> PendingDecision | None:
        wanted = spec.decision_rule.pending_decision_id
        for item in pending_decisions:
            if not isinstance(item, PendingDecision):
                raise ExperimentError("pending decisions must be PendingDecision records")
            if item.decision_id == wanted:
                return item
        return None

    def _refuse_context(
        self,
        spec: ShadowExperimentSpec,
        isolation: ExperimentIsolation,
        world: RepositoryWorldState | None,
        family: TaskFamily | None,
    ) -> ExperimentDecision | None:
        if spec.isolation_kind is not isolation.kind:
            return self._decision(
                spec, ExperimentAction.REFUSE, ExperimentReasonCode.ISOLATION_REQUIRED
            )
        if isolation.production:
            return self._decision(
                spec, ExperimentAction.REFUSE, ExperimentReasonCode.PRODUCTION_MUTATION
            )
        if isolation.policy_mutable:
            return self._decision(
                spec, ExperimentAction.REFUSE, ExperimentReasonCode.POLICY_MUTATION
            )
        if not isolation.authorized:
            return self._decision(
                spec, ExperimentAction.REFUSE, ExperimentReasonCode.UNAUTHORIZED_WORKTREE
            )
        if spec.isolation_kind is ExperimentIsolationKind.DISPOSABLE_WORKTREE and (
            not isolation.disposable
        ):
            return self._decision(
                spec, ExperimentAction.REFUSE, ExperimentReasonCode.NON_DISPOSABLE_WORKTREE
            )
        if spec.isolation_kind is ExperimentIsolationKind.FIXTURE:
            if isolation.fixture_id != spec.fixture_id:
                return self._decision(
                    spec, ExperimentAction.REFUSE, ExperimentReasonCode.ISOLATION_REQUIRED
                )
        elif spec.worktree_id and isolation.worktree_id not in {"", spec.worktree_id}:
            return self._decision(
                spec, ExperimentAction.REFUSE, ExperimentReasonCode.ISOLATION_REQUIRED
            )
        try:
            _reject_forbidden_targets(spec.scope_paths, "scope_paths")
            _reject_forbidden_targets(isolation.scope_paths, "isolation.scope_paths")
        except ExperimentIsolationError:
            return self._decision(
                spec, ExperimentAction.REFUSE, ExperimentReasonCode.POLICY_MUTATION
            )
        if spec.risk_class not in _ALLOWED_RISK:
            return self._decision(spec, ExperimentAction.REFUSE, ExperimentReasonCode.RISK_CEILING)
        if spec.privacy_class not in {
            ExperimentPrivacyClass.PUBLIC_FIXTURE,
            ExperimentPrivacyClass.REPOSITORY_LOCAL,
        }:
            return self._decision(
                spec, ExperimentAction.REFUSE, ExperimentReasonCode.PRIVACY_VIOLATION
            )
        if spec.bounds.network_request_limit != 0:
            return self._decision(
                spec, ExperimentAction.REFUSE, ExperimentReasonCode.NETWORK_FORBIDDEN
            )
        if spec.bounds.subprocess_limit != 0:
            return self._decision(
                spec,
                ExperimentAction.REFUSE,
                ExperimentReasonCode.ARBITRARY_EXECUTION_FORBIDDEN,
            )
        path_count = len(spec.scope_paths) or len(spec.required_data_ids)
        if not spec.bounds.covers(spec.cost, path_count=path_count):
            return self._decision(spec, ExperimentAction.REFUSE, ExperimentReasonCode.BOUND_EXCEEDED)
        if world is not None:
            if spec.bindings != world.bindings:
                return self._decision(
                    spec, ExperimentAction.REFUSE, ExperimentReasonCode.BINDING_MISMATCH
                )
            if spec.world_state_id and spec.world_state_id != world.content_id:
                return self._decision(
                    spec, ExperimentAction.REFUSE, ExperimentReasonCode.BINDING_MISMATCH
                )
        if family is not None:
            if spec.bindings != family.bindings:
                return self._decision(
                    spec, ExperimentAction.REFUSE, ExperimentReasonCode.BINDING_MISMATCH
                )
            if spec.family_cid and spec.family_cid != family.content_id:
                return self._decision(
                    spec,
                    ExperimentAction.REFUSE,
                    ExperimentReasonCode.FAMILY_BOUNDARY_MISMATCH,
                )
            if _RISK_RANK[spec.risk_class] > _RISK_RANK[family.boundary.risk_ceiling]:
                return self._decision(
                    spec, ExperimentAction.REFUSE, ExperimentReasonCode.RISK_CEILING
                )
        if spec.isolation_kind is ExperimentIsolationKind.DISPOSABLE_WORKTREE and not (
            spec.scope_paths or isolation.scope_paths
        ):
            return self._decision(
                spec, ExperimentAction.REFUSE, ExperimentReasonCode.ISOLATION_REQUIRED
            )
        return None

    @staticmethod
    def _decision(
        spec: ShadowExperimentSpec,
        action: ExperimentAction,
        reason_code: ExperimentReasonCode,
        *,
        matched_question_id: str = "",
        pending_decision_id: str = "",
    ) -> ExperimentDecision:
        return ExperimentDecision(
            spec_id=spec.content_id,
            action=action,
            reason_code=reason_code,
            decision_relevant=False,
            matched_question_id=matched_question_id,
            pending_decision_id=pending_decision_id,
            expected_decision_change_bps=0,
            isolation_kind=spec.isolation_kind,
        )


class ShadowExperimentRunner:
    """Execute only planned, isolated experiments and persist observations."""

    def __init__(
        self,
        *,
        store: InMemoryShadowObservationStore | None = None,
        worktree_port: DisposableWorktreePort | None = None,
    ) -> None:
        self._store = store if store is not None else InMemoryShadowObservationStore()
        self._worktree_port = worktree_port

    @property
    def store(self) -> InMemoryShadowObservationStore:
        return self._store

    def run(
        self,
        spec: ShadowExperimentSpec,
        decision: ExperimentDecision,
        *,
        isolation: ExperimentIsolation,
        fixture: ExperimentFixture | None = None,
        worktree: DisposableWorktreeGrant | None = None,
    ) -> ShadowExperimentResult:
        if not isinstance(spec, ShadowExperimentSpec):
            raise ExperimentError("runner requires ShadowExperimentSpec")
        if not isinstance(decision, ExperimentDecision):
            raise ExperimentError("runner requires ExperimentDecision")
        if decision.spec_id != spec.content_id:
            raise ExperimentError("decision does not bind the supplied experiment spec")
        if decision.can_grant_authority or not decision.observation_only:
            raise ExperimentAuthorityError("authoritative experiment decisions cannot run")
        if decision.action is ExperimentAction.REFUSE:
            raise ExperimentIsolationError("refused shadow experiments cannot run")
        if decision.action is ExperimentAction.SKIP:
            return ShadowExperimentResult(
                status=ExperimentOutcomeStatus.SKIPPED,
                decision=decision,
                reason_code=decision.reason_code,
            )
        if decision.action is not ExperimentAction.RUN or not decision.decision_relevant:
            raise ExperimentError("only decision-relevant run plans may execute")

        target, acquired = self._require_target(spec, isolation, fixture, worktree)
        try:
            observed_values, missing = self._observe(spec, fixture, target)
            support = self._support(spec, observed_values)
            effects = self._effects(spec, isolation, acquired, target)
            status = (
                ExperimentOutcomeStatus.INCOMPLETE
                if missing
                else ExperimentOutcomeStatus.OBSERVED
            )
            observation = ShadowExperimentObservation(
                bindings=spec.bindings,
                experiment_id=spec.content_id,
                decision_id=decision.content_id,
                question_id=spec.question_id,
                producer_contract=OBSERVATION_PRODUCER_CONTRACT,
                observed_values=observed_values,
                support=support,
                isolation_kind=spec.isolation_kind,
                isolation_root=self._isolation_root_label(target),
                worktree_id=(
                    target.worktree_id if isinstance(target, DisposableWorktreeGrant) else ""
                ),
                fixture_id=(
                    target.fixture_id if isinstance(target, ExperimentFixture) else ""
                ),
                effects=effects,
                missing_data_ids=missing,
            )
            persisted = self._store.persist(observation)
            self._write_observation(target, persisted)
            return ShadowExperimentResult(
                status=status,
                decision=decision,
                observation=persisted,
                reason_code=decision.reason_code,
            )
        finally:
            if acquired and self._worktree_port is not None and isinstance(
                target, DisposableWorktreeGrant
            ):
                self._worktree_port.release(target)

    def _require_target(
        self,
        spec: ShadowExperimentSpec,
        isolation: ExperimentIsolation,
        fixture: ExperimentFixture | None,
        worktree: DisposableWorktreeGrant | None,
    ) -> tuple[ExperimentFixture | DisposableWorktreeGrant, bool]:
        if isolation.production or (fixture is not None and fixture.production) or (
            worktree is not None and worktree.production
        ):
            raise ExperimentIsolationError("shadow experiments cannot mutate production")
        if isolation.policy_mutable or (fixture is not None and fixture.policy_mutable) or (
            worktree is not None and worktree.policy_mutable
        ):
            raise ExperimentIsolationError("shadow experiments cannot mutate policy")
        if spec.isolation_kind is ExperimentIsolationKind.FIXTURE:
            if fixture is None:
                raise ExperimentIsolationError("fixture experiments require an authorized fixture")
            if not fixture.authorized or not isolation.authorized:
                raise ExperimentIsolationError("fixture isolation was not authorized")
            if fixture.fixture_id != spec.fixture_id:
                raise ExperimentIsolationError("fixture identity does not match the experiment")
            return fixture, False
        if spec.isolation_kind is not ExperimentIsolationKind.DISPOSABLE_WORKTREE:
            raise ExperimentIsolationError("unsupported experiment isolation kind")
        grant = worktree
        acquired = False
        if grant is None:
            if self._worktree_port is None:
                raise ExperimentIsolationError(
                    "disposable worktree experiments require existing worktree authority"
                )
            grant = self._worktree_port.acquire(
                DisposableWorktreeRequest(
                    experiment_id=spec.content_id,
                    repository_id=spec.bindings.repository_id,
                    tree_id=spec.bindings.tree_id,
                    scope_paths=spec.scope_paths or isolation.scope_paths,
                    read_only=spec.risk_class is RiskClass.OBSERVATION_ONLY,
                )
            )
            acquired = True
        if not grant.authorized or not isolation.authorized:
            raise ExperimentIsolationError("disposable worktree was not authorized")
        if not grant.disposable or not isolation.disposable:
            raise ExperimentIsolationError("worktree is not disposable")
        if grant.production or isolation.production:
            raise ExperimentIsolationError("shadow experiments cannot mutate production")
        if grant.policy_mutable or isolation.policy_mutable:
            raise ExperimentIsolationError("shadow experiments cannot mutate policy")
        if spec.worktree_id and grant.worktree_id != spec.worktree_id:
            raise ExperimentIsolationError("worktree identity does not match the experiment")
        _reject_forbidden_targets(grant.scope_paths, "worktree.scope_paths")
        return grant, acquired

    def _observe(
        self,
        spec: ShadowExperimentSpec,
        fixture: ExperimentFixture | None,
        target: ExperimentFixture | DisposableWorktreeGrant,
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        observed: dict[str, str] = {}
        missing: list[str] = []
        values: Mapping[str, str] = fixture.values if fixture is not None else MappingProxyType({})
        root = Path(target.root_path) if target.root_path else None
        for data_id in spec.required_data_ids:
            if data_id in values:
                observed[data_id] = values[data_id]
                continue
            if root is None:
                missing.append(data_id)
                continue
            try:
                path = _contained_path(root, data_id)
            except (ExperimentIsolationError, ProcedureSafetyError, ProcedureContractError):
                missing.append(data_id)
                continue
            if not path.is_file():
                missing.append(data_id)
                continue
            payload = path.read_text(encoding="utf-8")
            observed[data_id] = _text(payload, "observed_values", required=False)
        if len(observed) > spec.bounds.observation_limit:
            raise ExperimentBoundsError("observed values exceed the declared observation bound")
        return observed, tuple(missing)

    @staticmethod
    def _support(spec: ShadowExperimentSpec, observed: Mapping[str, str]) -> ObservedSupport:
        if not spec.hypothesis_expected and not spec.counterfactual_expected:
            return ObservedSupport.NONE if not observed else ObservedSupport.INCONCLUSIVE
        joined = "|".join(observed.get(key, "") for key in spec.required_data_ids)
        hypothesis = spec.hypothesis_expected == joined
        counterfactual = spec.counterfactual_expected == joined
        if hypothesis and not counterfactual:
            return ObservedSupport.HYPOTHESIS
        if counterfactual and not hypothesis:
            return ObservedSupport.COUNTERFACTUAL
        if not observed:
            return ObservedSupport.NONE
        return ObservedSupport.INCONCLUSIVE

    @staticmethod
    def _effects(
        spec: ShadowExperimentSpec,
        isolation: ExperimentIsolation,
        acquired: bool,
        target: ExperimentFixture | DisposableWorktreeGrant,
    ) -> tuple[ExperimentEffect, ...]:
        effects: list[ExperimentEffect] = []
        if acquired:
            effects.append(
                ExperimentEffect(
                    effect_id="worktree-create",
                    effect_class=EffectClass.WORKTREE_CREATE,
                    targets=spec.scope_paths or isolation.scope_paths,
                    reversible=True,
                )
            )
        if spec.scope_paths:
            observe_targets = spec.scope_paths
        elif spec.isolation_kind is ExperimentIsolationKind.FIXTURE:
            observe_targets = ("fixtures/" + spec.fixture_id,)
        elif isinstance(target, DisposableWorktreeGrant):
            observe_targets = target.scope_paths
        else:
            observe_targets = ()
        effects.append(
            ExperimentEffect(
                effect_id="observe",
                effect_class=EffectClass.OBSERVE,
                targets=observe_targets,
                reversible=True,
            )
        )
        return tuple(effects)

    @staticmethod
    def _isolation_root_label(target: ExperimentFixture | DisposableWorktreeGrant) -> str:
        if isinstance(target, ExperimentFixture):
            return "fixtures/" + target.fixture_id
        return "worktrees/" + target.worktree_id

    @staticmethod
    def _write_observation(
        target: ExperimentFixture | DisposableWorktreeGrant,
        observation: ShadowExperimentObservation,
    ) -> None:
        if not target.root_path:
            return
        root = Path(target.root_path)
        if not root.exists():
            return
        path = _contained_path(root, "observations/" + observation.experiment_id + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(observation.to_dict()))


def _register_artifact_types() -> None:
    for cls in (
        ExperimentCost,
        ExperimentExecutionBounds,
        ExperimentDecisionRule,
        PendingDecision,
        ExplicitQuestion,
        ExperimentIsolation,
        ShadowExperimentSpec,
        ExperimentDecision,
        ExperimentEffect,
        ShadowExperimentObservation,
    ):
        ARTIFACT_TYPES_BY_SCHEMA[cls.SCHEMA] = cls


_register_artifact_types()


__all__ = [
    "DecisionRuleKind",
    "DisposableWorktreeGrant",
    "DisposableWorktreePort",
    "DisposableWorktreeRequest",
    "ExperimentAction",
    "ExperimentAuthorityError",
    "ExperimentBoundsError",
    "ExperimentCost",
    "ExperimentDecision",
    "ExperimentDecisionRule",
    "ExperimentEffect",
    "ExperimentError",
    "ExperimentExecutionBounds",
    "ExperimentFixture",
    "ExperimentIsolation",
    "ExperimentIsolationError",
    "ExperimentIsolationKind",
    "ExperimentOutcomeStatus",
    "ExperimentPlanner",
    "ExperimentPrivacyClass",
    "ExperimentPrivacyError",
    "ExperimentReasonCode",
    "ExplicitQuestion",
    "InMemoryDisposableWorktreePort",
    "InMemoryShadowObservationStore",
    "ObservedSupport",
    "PendingDecision",
    "ShadowExperimentObservation",
    "ShadowExperimentResult",
    "ShadowExperimentRunner",
    "ShadowExperimentSpec",
    "UncertaintySource",
    "questions_from_family",
    "questions_from_world",
]
