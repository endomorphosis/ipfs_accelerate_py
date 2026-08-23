"""Bounded shadow-experiment planning and isolated observation.

Shadow experiments are decision instruments, not authorities.  They may run
only when every required declaration field is present, the experiment can
change a pending decision, and isolation is a fixture or an already-authorized
disposable worktree.  Results are candidate observations: they cannot grant
authority, establish proof or postconditions, promote, complete, mutate
production or policy, or suppress validation.

Value of experiment is an integer count of extra reachable decision options.
A zero value skips the experiment; malformed, unbounded, or production-touching
experiments are refused fail-closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract
from .contracts import (
    MAX_ITEMS,
    MAX_MAPPING_ITEMS,
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ArtifactState,
    ExperimentEvaluation as ExperimentEvaluationArtifact,
    ExperimentObservation as ExperimentObservationArtifact,
    ExperimentPlan as ExperimentPlanArtifact,
    ProcedureContractError,
    RiskClass,
    TaskFamily,
    _bounded,
    _decode_fields,
    _enum,
    _enums,
    _freeze,
    _identifier,
    _nested,
    _nonnegative_int,
    _schema_name,
    _strings,
    _text,
    _unsafe_key,
    _verify_identity,
)
from .task_family import REQUIRED_BOUNDARY_DIMENSIONS
from .world_model import RepositoryWorldState


PLANNER_REVISION: Final[str] = "shadow-experiment-planner@1"
RUNNER_REVISION: Final[str] = "shadow-experiment-runner@1"
MAX_EXPERIMENT_COST: Final[int] = 2**31 - 1
MAX_EXPERIMENT_VALUE: Final[int] = MAX_ITEMS
REQUIRED_DECLARATION_FIELDS: Final[tuple[str, ...]] = (
    "question",
    "hypothesis",
    "counterfactual",
    "required_data",
    "risk",
    "privacy",
    "cost",
    "decision_rule",
    "execution_bound",
)
REQUIRED_PRIVACY_CLASSES: Final[tuple[str, ...]] = (
    "no-secrets",
    "no-credentials",
    "no-private-prompts",
    "no-chain-of-thought",
    "no-source-bodies",
    "no-model-transcripts",
)
_PRIVACY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "model_transcript",
        "private_prompt",
        "prompt_body",
        "source_body",
        "transcript",
    }
)
_RISK_RANK: Final[dict[RiskClass, int]] = {
    RiskClass.OBSERVATION_ONLY: 0,
    RiskClass.REVERSIBLE_LOCAL: 1,
    RiskClass.REPOSITORY_WRITE: 2,
    RiskClass.PUBLIC_CONTRACT: 3,
    RiskClass.AUTHORITY_OR_SECURITY: 4,
}
_AUTHORITY_USES: Final[frozenset[str]] = frozenset(
    {
        "authority",
        "policy",
        "promotion",
        "postcondition",
        "proof",
        "completion",
        "validation-suppression",
        "human-review-suppression",
    }
)


class ExperimentError(ProcedureContractError):
    """A shadow experiment declaration, decision, or observation is unsafe."""


class ExperimentDeclarationError(ExperimentError):
    """A required experiment declaration field is missing or malformed."""


class ExperimentIsolationError(ExperimentError):
    """An experiment targeted production, policy, or an unauthorized worktree."""


class ExperimentObservationError(ExperimentError):
    """Observed experiment facts could not be admitted as a bounded observation."""


class ExperimentAction(str, Enum):
    RUN = "run"
    SKIP = "skip"
    REFUSE = "refuse"


class ExperimentReason(str, Enum):
    DECISION_RELEVANT = "decision-relevant"
    CANNOT_CHANGE_DECISION = "cannot-change-decision"
    QUESTION_NOT_OPEN = "question-not-open"
    QUESTION_NOT_RELEVANT = "question-not-relevant"
    NO_PENDING_DECISION = "no-pending-decision"
    ALREADY_ANSWERED = "already-answered"
    COST_EXCEEDS_BUDGET = "cost-exceeds-budget"
    COST_EXCEEDS_BOUND = "cost-exceeds-bound"
    MISSING_DECLARATION = "missing-declaration"
    PRODUCTION_MUTATION = "production-mutation"
    POLICY_MUTATION = "policy-mutation"
    UNAUTHORIZED_WORKTREE = "unauthorized-worktree"
    NON_DISPOSABLE_WORKTREE = "non-disposable-worktree"
    UNSAFE_ISOLATION = "unsafe-isolation"
    PRIVACY_VIOLATION = "privacy-violation"
    RISK_CEILING = "risk-ceiling"
    UNBOUNDED = "unbounded"
    FORBIDDEN_EFFECT = "forbidden-effect"
    BINDING_MISMATCH = "binding-mismatch"


class IsolationKind(str, Enum):
    FIXTURE = "fixture"
    AUTHORIZED_DISPOSABLE_WORKTREE = "authorized-disposable-worktree"


class DecisionRuleClass(str, Enum):
    DISTINGUISH_OUTCOMES = "distinguish-outcomes"
    CLOSED_MEMBERSHIP = "closed-membership"
    INTEGER_THRESHOLD = "integer-threshold"


class ExperimentOutcome(str, Enum):
    HYPOTHESIS = "hypothesis"
    COUNTERFACTUAL = "counterfactual"
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"
    REFUSED = "refused"


class ObservationUse(str, Enum):
    PLANNING_OBSERVATION = "planning-observation"
    COST = "cost"
    PRIORITY = "priority"
    AUTHORITY = "authority"
    POLICY = "policy"
    PROMOTION = "promotion"
    POSTCONDITION = "postcondition"
    PROOF = "proof"
    COMPLETION = "completion"
    VALIDATION_SUPPRESSION = "validation-suppression"
    HUMAN_REVIEW_SUPPRESSION = "human-review-suppression"


class PrivacyClass(str, Enum):
    NO_SECRETS = "no-secrets"
    NO_CREDENTIALS = "no-credentials"
    NO_PRIVATE_PROMPTS = "no-private-prompts"
    NO_CHAIN_OF_THOUGHT = "no-chain-of-thought"
    NO_SOURCE_BODIES = "no-source-bodies"
    NO_MODEL_TRANSCRIPTS = "no-model-transcripts"


class ExperimentEffectClass(str, Enum):
    OBSERVE_FIXTURE = "observe-fixture"
    OBSERVE_DISPOSABLE_WORKTREE = "observe-disposable-worktree"
    MUTATE_PRODUCTION = "mutate-production"
    MUTATE_POLICY = "mutate-policy"
    ARBITRARY_SHELL = "arbitrary-shell"
    ARBITRARY_PYTHON = "arbitrary-python"
    NETWORK_REQUEST = "network-request"


ALLOWED_EXPERIMENT_EFFECTS: Final[frozenset[ExperimentEffectClass]] = frozenset(
    {
        ExperimentEffectClass.OBSERVE_FIXTURE,
        ExperimentEffectClass.OBSERVE_DISPOSABLE_WORKTREE,
    }
)
RISK_CEILING: Final[RiskClass] = RiskClass.REVERSIBLE_LOCAL


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ExperimentError(f"{field_name} must be a boolean")
    return value


def _cost_int(value: Any, field_name: str) -> int:
    return _nonnegative_int(value, field_name, maximum=MAX_EXPERIMENT_COST)


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _privacy_hit(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return _unsafe_key(key) or any(marker in normalized for marker in _PRIVACY_MARKERS)


def _reject_privacy(value: Any, field_name: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or _privacy_hit(raw_key):
                raise ExperimentError(f"{field_name} contains a forbidden privacy field")
            _reject_privacy(item, field_name)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        for item in value:
            _reject_privacy(item, field_name)


def _risk_rank(value: RiskClass) -> int:
    return _RISK_RANK[value]


@dataclass(frozen=True)
class ExperimentCost:
    """Integer resource cost of one shadow experiment."""

    tokens: int = 0
    duration_ms: int = 0
    provider_cost_micros: int = 0
    worktree_count: int = 0

    def __post_init__(self) -> None:
        for name in ("tokens", "duration_ms", "provider_cost_micros", "worktree_count"):
            object.__setattr__(self, name, _cost_int(getattr(self, name), name))

    @property
    def units(self) -> int:
        return (
            self.tokens
            + self.duration_ms
            + self.provider_cost_micros
            + self.worktree_count
        )

    def exceeds(self, other: ExperimentCost) -> bool:
        return (
            self.tokens > other.tokens
            or self.duration_ms > other.duration_ms
            or self.provider_cost_micros > other.provider_cost_micros
            or self.worktree_count > other.worktree_count
        )

    def to_record(self) -> dict[str, int]:
        return {
            "tokens": self.tokens,
            "duration_ms": self.duration_ms,
            "provider_cost_micros": self.provider_cost_micros,
            "worktree_count": self.worktree_count,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | None) -> ExperimentCost:
        if payload is None:
            raise ExperimentDeclarationError("cost is required")
        if not isinstance(payload, Mapping):
            raise ExperimentDeclarationError("cost must be a mapping")
        return cls(
            tokens=payload.get("tokens", 0),
            duration_ms=payload.get("duration_ms", 0),
            provider_cost_micros=payload.get("provider_cost_micros", 0),
            worktree_count=payload.get("worktree_count", 0),
        )


@dataclass(frozen=True)
class ExecutionBound:
    """Fixed integer execution envelope.  Exceeding it is fail-closed."""

    max_tokens: int = 4_096
    max_duration_ms: int = 60_000
    max_provider_cost_micros: int = 0
    max_worktrees: int = 1
    max_output_bytes: int = 65_536
    max_steps: int = 8

    def __post_init__(self) -> None:
        for name in (
            "max_tokens",
            "max_duration_ms",
            "max_provider_cost_micros",
            "max_worktrees",
            "max_output_bytes",
            "max_steps",
        ):
            object.__setattr__(self, name, _cost_int(getattr(self, name), name))
        if self.max_worktrees > 1:
            raise ExperimentDeclarationError("shadow experiments may use at most one worktree")

    def as_cost(self) -> ExperimentCost:
        return ExperimentCost(
            tokens=self.max_tokens,
            duration_ms=self.max_duration_ms,
            provider_cost_micros=self.max_provider_cost_micros,
            worktree_count=self.max_worktrees,
        )

    def admits(self, cost: ExperimentCost) -> bool:
        return not cost.exceeds(self.as_cost())

    def to_record(self) -> dict[str, int]:
        return {
            "max_tokens": self.max_tokens,
            "max_duration_ms": self.max_duration_ms,
            "max_provider_cost_micros": self.max_provider_cost_micros,
            "max_worktrees": self.max_worktrees,
            "max_output_bytes": self.max_output_bytes,
            "max_steps": self.max_steps,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | None) -> ExecutionBound:
        if payload is None:
            raise ExperimentDeclarationError("execution_bound is required")
        if not isinstance(payload, Mapping):
            raise ExperimentDeclarationError("execution_bound must be a mapping")
        return cls(
            max_tokens=payload.get("max_tokens", 4_096),
            max_duration_ms=payload.get("max_duration_ms", 60_000),
            max_provider_cost_micros=payload.get("max_provider_cost_micros", 0),
            max_worktrees=payload.get("max_worktrees", 1),
            max_output_bytes=payload.get("max_output_bytes", 65_536),
            max_steps=payload.get("max_steps", 8),
        )


@dataclass(frozen=True)
class DecisionRule:
    """Closed rule mapping hypothesis/counterfactual outcomes onto a decision."""

    rule_class: DecisionRuleClass
    observation_binding: str
    hypothesis_option_id: str
    counterfactual_option_id: str
    hypothesis_operand: Any = True
    counterfactual_operand: Any = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rule_class",
            _enum(self.rule_class, DecisionRuleClass, "rule_class"),
        )
        object.__setattr__(
            self,
            "observation_binding",
            _identifier(self.observation_binding, "observation_binding"),
        )
        object.__setattr__(
            self,
            "hypothesis_option_id",
            _identifier(self.hypothesis_option_id, "hypothesis_option_id"),
        )
        object.__setattr__(
            self,
            "counterfactual_option_id",
            _identifier(self.counterfactual_option_id, "counterfactual_option_id"),
        )
        object.__setattr__(
            self, "hypothesis_operand", _freeze(self.hypothesis_operand, "hypothesis_operand")
        )
        object.__setattr__(
            self,
            "counterfactual_operand",
            _freeze(self.counterfactual_operand, "counterfactual_operand"),
        )
        _reject_privacy(self.hypothesis_operand, "hypothesis_operand")
        _reject_privacy(self.counterfactual_operand, "counterfactual_operand")

    @property
    def distinguishes(self) -> bool:
        return self.hypothesis_option_id != self.counterfactual_option_id

    def to_record(self) -> dict[str, Any]:
        return {
            "rule_class": self.rule_class.value,
            "observation_binding": self.observation_binding,
            "hypothesis_option_id": self.hypothesis_option_id,
            "counterfactual_option_id": self.counterfactual_option_id,
            "hypothesis_operand": self.hypothesis_operand,
            "counterfactual_operand": self.counterfactual_operand,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | None) -> DecisionRule:
        if payload is None:
            raise ExperimentDeclarationError("decision_rule is required")
        if not isinstance(payload, Mapping):
            raise ExperimentDeclarationError("decision_rule must be a mapping")
        return cls(
            rule_class=payload.get("rule_class", ""),
            observation_binding=payload.get("observation_binding", ""),
            hypothesis_option_id=payload.get("hypothesis_option_id", ""),
            counterfactual_option_id=payload.get("counterfactual_option_id", ""),
            hypothesis_operand=payload.get("hypothesis_operand", True),
            counterfactual_operand=payload.get("counterfactual_operand", False),
        )


@dataclass(frozen=True)
class IsolationTarget:
    """Fixture or authorized disposable worktree.  Production is never a target."""

    kind: IsolationKind
    target_id: str
    repository_id: str
    tree_id: str
    disposable: bool = True
    production: bool = False
    policy_mutable: bool = False
    authorized: bool = False
    admission_receipt_id: str = ""
    lease_id: str = ""
    fencing_token: int = 0
    scope_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, IsolationKind, "kind"))
        for name in ("target_id", "repository_id", "tree_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in ("disposable", "production", "policy_mutable", "authorized"):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self,
            "admission_receipt_id",
            _identifier(self.admission_receipt_id, "admission_receipt_id", required=False),
        )
        object.__setattr__(
            self, "lease_id", _identifier(self.lease_id, "lease_id", required=False)
        )
        object.__setattr__(
            self, "fencing_token", _nonnegative_int(self.fencing_token, "fencing_token")
        )
        object.__setattr__(
            self,
            "scope_paths",
            _strings(self.scope_paths, "scope_paths", paths=True, required=True),
        )

    def refusal_reason(self) -> ExperimentReason | None:
        if self.production:
            return ExperimentReason.PRODUCTION_MUTATION
        if self.policy_mutable:
            return ExperimentReason.POLICY_MUTATION
        if self.kind is IsolationKind.FIXTURE:
            if not self.disposable:
                return ExperimentReason.NON_DISPOSABLE_WORKTREE
            return None
        if self.kind is IsolationKind.AUTHORIZED_DISPOSABLE_WORKTREE:
            if not self.disposable:
                return ExperimentReason.NON_DISPOSABLE_WORKTREE
            if not self.authorized or not self.admission_receipt_id:
                return ExperimentReason.UNAUTHORIZED_WORKTREE
            if not self.lease_id or self.fencing_token <= 0:
                return ExperimentReason.UNAUTHORIZED_WORKTREE
            return None
        return ExperimentReason.UNSAFE_ISOLATION

    @property
    def is_safe(self) -> bool:
        return self.refusal_reason() is None

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target_id": self.target_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "disposable": self.disposable,
            "production": self.production,
            "policy_mutable": self.policy_mutable,
            "authorized": self.authorized,
            "admission_receipt_id": self.admission_receipt_id,
            "lease_id": self.lease_id,
            "fencing_token": self.fencing_token,
            "scope_paths": self.scope_paths,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | None) -> IsolationTarget:
        if payload is None:
            raise ExperimentDeclarationError("isolation is required")
        if not isinstance(payload, Mapping):
            raise ExperimentDeclarationError("isolation must be a mapping")
        return cls(
            kind=payload.get("kind", ""),
            target_id=payload.get("target_id", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            disposable=payload.get("disposable", True),
            production=payload.get("production", False),
            policy_mutable=payload.get("policy_mutable", False),
            authorized=payload.get("authorized", False),
            admission_receipt_id=payload.get("admission_receipt_id", ""),
            lease_id=payload.get("lease_id", ""),
            fencing_token=payload.get("fencing_token", 0),
            scope_paths=payload.get("scope_paths", ()),
        )


@dataclass(frozen=True)
class UncertaintyQuestion:
    """One explicit open question from world uncertainty or a family boundary."""

    question_id: str
    source: str
    evidence_id: str = ""
    dimension: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _identifier(self.question_id, "question_id"))
        object.__setattr__(self, "source", _identifier(self.source, "source"))
        object.__setattr__(
            self, "evidence_id", _identifier(self.evidence_id, "evidence_id", required=False)
        )
        object.__setattr__(
            self, "dimension", _identifier(self.dimension, "dimension", required=False)
        )


@dataclass(frozen=True)
class PendingDecision:
    """The decision an experiment must be able to change, or it is skipped."""

    decision_id: str
    option_ids: tuple[str, ...]
    question_ids: tuple[str, ...]
    committed_option_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _identifier(self.decision_id, "decision_id"))
        object.__setattr__(
            self,
            "option_ids",
            _strings(self.option_ids, "option_ids", identifiers=True, required=True),
        )
        object.__setattr__(
            self,
            "question_ids",
            _strings(self.question_ids, "question_ids", identifiers=True, required=True),
        )
        object.__setattr__(
            self,
            "committed_option_id",
            _identifier(self.committed_option_id, "committed_option_id", required=False),
        )
        if self.committed_option_id and self.committed_option_id not in self.option_ids:
            raise ExperimentError("committed option is not one of the pending decision options")


@dataclass(frozen=True)
class ShadowExperiment:
    """Complete shadow-experiment declaration.  Incomplete declarations fail closed."""

    bindings: ArtifactBindings
    experiment_id: str
    question_id: str
    question: str
    hypothesis_id: str
    hypothesis: str
    counterfactual_id: str
    counterfactual: str
    required_data_ids: tuple[str, ...]
    risk_class: RiskClass
    privacy_classes: tuple[PrivacyClass, ...]
    cost: ExperimentCost
    decision_rule: DecisionRule
    execution_bound: ExecutionBound
    isolation: IsolationTarget
    effects: tuple[ExperimentEffectClass, ...] = (ExperimentEffectClass.OBSERVE_FIXTURE,)
    decision_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in (
            "experiment_id",
            "question_id",
            "hypothesis_id",
            "counterfactual_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in ("question", "hypothesis", "counterfactual"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "required_data_ids",
            _strings(self.required_data_ids, "required_data_ids", identifiers=True, required=True),
        )
        object.__setattr__(self, "risk_class", _enum(self.risk_class, RiskClass, "risk_class"))
        object.__setattr__(
            self,
            "privacy_classes",
            _enums(
                self.privacy_classes,
                PrivacyClass,
                "privacy_classes",
                limit=len(PrivacyClass),
                required=True,
            ),
        )
        object.__setattr__(self, "cost", _nested_record(self.cost, ExperimentCost, "cost"))
        object.__setattr__(
            self,
            "decision_rule",
            _nested_record(self.decision_rule, DecisionRule, "decision_rule"),
        )
        object.__setattr__(
            self,
            "execution_bound",
            _nested_record(self.execution_bound, ExecutionBound, "execution_bound"),
        )
        object.__setattr__(
            self, "isolation", _nested_record(self.isolation, IsolationTarget, "isolation")
        )
        object.__setattr__(
            self,
            "effects",
            _enums(
                self.effects,
                ExperimentEffectClass,
                "effects",
                limit=len(ExperimentEffectClass),
                required=True,
            ),
        )
        object.__setattr__(
            self, "decision_id", _identifier(self.decision_id, "decision_id", required=False)
        )
        if self.decision_rule.observation_binding not in self.required_data_ids:
            raise ExperimentDeclarationError(
                "decision-rule observation binding must be declared as required data"
            )
        if self.isolation.repository_id != self.bindings.repository_id:
            raise ExperimentDeclarationError("isolation repository is not exact-binding current")
        if self.isolation.tree_id != self.bindings.tree_id:
            raise ExperimentDeclarationError("isolation tree is not exact-binding current")
        missing = missing_declaration_fields(self)
        if missing:
            raise ExperimentDeclarationError(
                "shadow experiment is missing required declaration fields: " + ",".join(missing)
            )

    def to_facts(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "question_id": self.question_id,
            "question": self.question,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis": self.hypothesis,
            "counterfactual_id": self.counterfactual_id,
            "counterfactual": self.counterfactual,
            "required_data_ids": self.required_data_ids,
            "risk_class": self.risk_class.value,
            "privacy_classes": tuple(item.value for item in self.privacy_classes),
            "cost": self.cost.to_record(),
            "decision_rule": self.decision_rule.to_record(),
            "execution_bound": self.execution_bound.to_record(),
            "isolation": self.isolation.to_record(),
            "effects": tuple(item.value for item in self.effects),
            "decision_id": self.decision_id,
            "can_authorize": False,
        }


def _nested_record(value: Any, cls: type[Any], field_name: str) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls.from_record(value)
    raise ExperimentDeclarationError(f"{field_name} must be {cls.__name__}")


def missing_declaration_fields(experiment: ShadowExperiment) -> tuple[str, ...]:
    present = {
        "question": experiment.question,
        "hypothesis": experiment.hypothesis,
        "counterfactual": experiment.counterfactual,
        "required_data": experiment.required_data_ids,
        "risk": experiment.risk_class.value,
        "privacy": experiment.privacy_classes,
        "cost": experiment.cost.to_record(),
        "decision_rule": experiment.decision_rule.observation_binding,
        "execution_bound": experiment.execution_bound.to_record(),
    }
    return tuple(name for name in REQUIRED_DECLARATION_FIELDS if not present[name])


def extract_uncertainty_questions(
    *,
    world: RepositoryWorldState | None = None,
    family: TaskFamily | None = None,
    questions: Sequence[UncertaintyQuestion] = (),
) -> tuple[UncertaintyQuestion, ...]:
    """Union explicit questions with world-unavailable and family-unknown cases."""

    extracted: list[UncertaintyQuestion] = []
    seen: set[str] = set()

    def _add(item: UncertaintyQuestion) -> None:
        if item.question_id in seen:
            return
        seen.add(item.question_id)
        extracted.append(item)

    for item in questions:
        _add(item if isinstance(item, UncertaintyQuestion) else UncertaintyQuestion(**item))
    if world is not None:
        evidence = world.content_id
        for dimension in world.unavailable_dimensions:
            _add(
                UncertaintyQuestion(
                    question_id=f"world.unavailable.{dimension}",
                    source="world-unavailable-dimension",
                    evidence_id=evidence,
                    dimension=dimension,
                )
            )
        if world.projection_status.value != "current":
            _add(
                UncertaintyQuestion(
                    question_id=f"world.status.{world.projection_status.value}",
                    source="world-projection-status",
                    evidence_id=evidence,
                    dimension=world.projection_status.value,
                )
            )
    if family is not None:
        evidence = family.content_id
        for cid in family.boundary.unknown_case_cids:
            _add(
                UncertaintyQuestion(
                    question_id=f"family.unknown.{cid}",
                    source="family-unknown-case",
                    evidence_id=evidence,
                    dimension="unknown-case",
                )
            )
        present = {
            "positive_member_cids": family.boundary.positive_member_cids,
            "negative_example_cids": family.boundary.negative_example_cids,
            "boundary_example_cids": family.boundary.boundary_example_cids,
            "unknown_case_cids": family.boundary.unknown_case_cids,
            "risk_ceiling": (family.boundary.risk_ceiling.value,),
            "permitted_repositories": family.boundary.permitted_repositories,
            "permitted_languages": family.boundary.permitted_languages,
            "permitted_frameworks": family.boundary.permitted_frameworks,
            "permitted_effect_classes": family.boundary.permitted_effect_classes,
            "required_operation_contracts": family.required_operation_contracts,
            "validation_structure": family.validation_structure,
            "rollback_structure": family.rollback_structure,
            "postcondition_shape": family.postcondition_shape,
        }
        for name in REQUIRED_BOUNDARY_DIMENSIONS:
            if not present[name]:
                _add(
                    UncertaintyQuestion(
                        question_id=f"family.missing.{name.replace('_', '-')}",
                        source="family-incomplete-boundary",
                        evidence_id=evidence,
                        dimension=name.replace("_", "-"),
                    )
                )
    return tuple(extracted)


def value_of_experiment(
    experiment: ShadowExperiment,
    pending: PendingDecision,
) -> tuple[int, tuple[str, ...]]:
    """Return extra reachable options.  Zero means the experiment cannot change the decision."""

    reachable: list[str] = []
    for option in (
        experiment.decision_rule.hypothesis_option_id,
        experiment.decision_rule.counterfactual_option_id,
    ):
        if option in pending.option_ids and option not in reachable:
            reachable.append(option)
    if pending.committed_option_id:
        return 0, tuple(reachable)
    if not experiment.decision_rule.distinguishes:
        return 0, tuple(reachable)
    extra = max(0, len(reachable) - 1)
    return extra, tuple(reachable)


def observation_may_discharge(use: ObservationUse | str) -> bool:
    normalized = _enum(use, ObservationUse, "use")
    return normalized in {
        ObservationUse.PLANNING_OBSERVATION,
        ObservationUse.COST,
        ObservationUse.PRIORITY,
    }


@dataclass(frozen=True)
class ExperimentDecision(CanonicalContract):
    """Typed run/skip/refuse result.  The decision itself cannot authorize."""

    SCHEMA: ClassVar[str] = _schema_name("ExperimentDecision")

    bindings: ArtifactBindings
    experiment_id: str
    action: ExperimentAction
    reason_code: ExperimentReason
    question_id: str
    decision_id: str
    value_of_experiment: int
    reachable_option_ids: tuple[str, ...]
    isolation: IsolationTarget
    risk_class: RiskClass
    estimated_cost: ExperimentCost
    execution_bound: ExecutionBound
    decision_rule: DecisionRule
    privacy_classes: tuple[PrivacyClass, ...] = ()
    planner_revision: str = PLANNER_REVISION
    plan_cid: str = ""
    can_authorize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self, "experiment_id", _identifier(self.experiment_id, "experiment_id")
        )
        object.__setattr__(self, "action", _enum(self.action, ExperimentAction, "action"))
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, ExperimentReason, "reason_code")
        )
        object.__setattr__(self, "question_id", _identifier(self.question_id, "question_id"))
        object.__setattr__(
            self, "decision_id", _identifier(self.decision_id, "decision_id", required=False)
        )
        object.__setattr__(
            self,
            "value_of_experiment",
            _nonnegative_int(
                self.value_of_experiment,
                "value_of_experiment",
                maximum=MAX_EXPERIMENT_VALUE,
            ),
        )
        object.__setattr__(
            self,
            "reachable_option_ids",
            _strings(self.reachable_option_ids, "reachable_option_ids", identifiers=True),
        )
        object.__setattr__(
            self, "isolation", _nested_record(self.isolation, IsolationTarget, "isolation")
        )
        object.__setattr__(self, "risk_class", _enum(self.risk_class, RiskClass, "risk_class"))
        object.__setattr__(
            self, "estimated_cost", _nested_record(self.estimated_cost, ExperimentCost, "cost")
        )
        object.__setattr__(
            self,
            "execution_bound",
            _nested_record(self.execution_bound, ExecutionBound, "execution_bound"),
        )
        object.__setattr__(
            self,
            "decision_rule",
            _nested_record(self.decision_rule, DecisionRule, "decision_rule"),
        )
        object.__setattr__(
            self,
            "privacy_classes",
            _enums(
                self.privacy_classes,
                PrivacyClass,
                "privacy_classes",
                limit=len(PrivacyClass),
            ),
        )
        object.__setattr__(
            self,
            "planner_revision",
            _identifier(self.planner_revision, "planner_revision"),
        )
        if self.planner_revision != PLANNER_REVISION:
            raise ExperimentError("experiment decision planner revision is not current")
        object.__setattr__(
            self, "plan_cid", _identifier(self.plan_cid, "plan_cid", required=False)
        )
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            raise ExperimentError("experiment decisions cannot authorize")
        if self.action is ExperimentAction.RUN:
            if self.value_of_experiment <= 0:
                raise ExperimentError("a run decision requires positive decision value")
            if self.reason_code is not ExperimentReason.DECISION_RELEVANT:
                raise ExperimentError("a run decision must be labeled decision-relevant")
        elif self.action is ExperimentAction.SKIP and self.value_of_experiment != 0:
            if self.reason_code not in {
                ExperimentReason.COST_EXCEEDS_BUDGET,
                ExperimentReason.QUESTION_NOT_OPEN,
                ExperimentReason.QUESTION_NOT_RELEVANT,
            }:
                raise ExperimentError("a skip decision cannot claim unused decision value")
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

    def allows_use(self, use: ObservationUse | str) -> bool:
        return observation_may_discharge(use)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "planner_revision": PLANNER_REVISION,
            "bindings": self.bindings,
            "experiment_id": self.experiment_id,
            "action": self.action.value,
            "reason_code": self.reason_code.value,
            "question_id": self.question_id,
            "decision_id": self.decision_id,
            "value_of_experiment": self.value_of_experiment,
            "reachable_option_ids": self.reachable_option_ids,
            "isolation": self.isolation.to_record(),
            "risk_class": self.risk_class.value,
            "estimated_cost": self.estimated_cost.to_record(),
            "execution_bound": self.execution_bound.to_record(),
            "decision_rule": self.decision_rule.to_record(),
            "privacy_classes": tuple(item.value for item in self.privacy_classes),
            "planner_revision": self.planner_revision,
            "plan_cid": self.plan_cid,
            "can_authorize": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentDecision:
        fields = (
            "bindings",
            "experiment_id",
            "action",
            "reason_code",
            "question_id",
            "decision_id",
            "value_of_experiment",
            "reachable_option_ids",
            "isolation",
            "risk_class",
            "estimated_cost",
            "execution_bound",
            "decision_rule",
            "privacy_classes",
            "planner_revision",
            "plan_cid",
            "can_authorize",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if values.get("can_authorize"):
            raise ExperimentError("experiment decisions cannot authorize")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ExperimentObservationRecord:
    """Persisted shadow-experiment observation.  Candidate-only; never authority."""

    bindings: ArtifactBindings
    experiment_id: str
    question_id: str
    outcome: ExperimentOutcome
    selected_option_id: str
    observed_facts: Mapping[str, Any]
    isolation: IsolationTarget
    producer_id: str = RUNNER_REVISION
    hypothesis_supported: bool = False
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self, "experiment_id", _identifier(self.experiment_id, "experiment_id")
        )
        object.__setattr__(self, "question_id", _identifier(self.question_id, "question_id"))
        object.__setattr__(self, "outcome", _enum(self.outcome, ExperimentOutcome, "outcome"))
        object.__setattr__(
            self,
            "selected_option_id",
            _identifier(self.selected_option_id, "selected_option_id", required=False),
        )
        facts = _freeze(self.observed_facts, "observed_facts")
        if not isinstance(facts, Mapping):
            raise ExperimentObservationError("observed_facts must be a mapping")
        _reject_privacy(facts, "observed_facts")
        object.__setattr__(self, "observed_facts", facts)
        object.__setattr__(
            self, "isolation", _nested_record(self.isolation, IsolationTarget, "isolation")
        )
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))
        object.__setattr__(
            self, "hypothesis_supported", _bool(self.hypothesis_supported, "hypothesis_supported")
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            raise ExperimentObservationError("experiment observations cannot authorize")
        if self.state not in {ArtifactState.CANDIDATE, ArtifactState.SHADOW, ArtifactState.REJECTED}:
            raise ExperimentObservationError(
                "experiment observations cannot claim verified or promoted state"
            )
        if not self.isolation.is_safe:
            raise ExperimentIsolationError("observations cannot be taken from unsafe isolation")

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

    def allows_use(self, use: ObservationUse | str) -> bool:
        return observation_may_discharge(use)

    def to_facts(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "question_id": self.question_id,
            "outcome": self.outcome.value,
            "selected_option_id": self.selected_option_id,
            "observed_facts": dict(self.observed_facts),
            "isolation": self.isolation.to_record(),
            "producer_id": self.producer_id,
            "hypothesis_supported": self.hypothesis_supported,
            "can_authorize": False,
            "can_grant_authority": False,
            "can_establish_proof": False,
            "can_establish_postcondition": False,
            "can_establish_completion": False,
            "can_promote": False,
        }

    def to_artifact(self, *, emitted_at_ms: int = 0) -> ExperimentObservationArtifact:
        return ExperimentObservationArtifact(
            bindings=self.bindings,
            state=self.state,
            subject_cid=self.experiment_id,
            reference_cids=(self.isolation.target_id, self.isolation.admission_receipt_id)
            if self.isolation.admission_receipt_id
            else (self.isolation.target_id,),
            labels=(self.outcome.value, self.isolation.kind.value, "observation-only"),
            facts=self.to_facts(),
            created_at_ms=emitted_at_ms,
        )


@dataclass(frozen=True)
class PlannedExperiment:
    """Planner output: typed decision plus the generic ExperimentPlan wire artifact."""

    decision: ExperimentDecision
    plan_artifact: ExperimentPlanArtifact
    experiment: ShadowExperiment


@dataclass(frozen=True)
class ExperimentRunResult:
    """Runner output: candidate observation plus generic observation/evaluation artifacts."""

    observation: ExperimentObservationRecord
    observation_artifact: ExperimentObservationArtifact
    evaluation_artifact: ExperimentEvaluationArtifact


def _forbidden_effects(experiment: ShadowExperiment) -> tuple[ExperimentEffectClass, ...]:
    return tuple(item for item in experiment.effects if item not in ALLOWED_EXPERIMENT_EFFECTS)


def _privacy_complete(experiment: ShadowExperiment) -> bool:
    present = {item.value for item in experiment.privacy_classes}
    return set(REQUIRED_PRIVACY_CLASSES).issubset(present)


def _effect_matches_isolation(experiment: ShadowExperiment) -> bool:
    if experiment.isolation.kind is IsolationKind.FIXTURE:
        return experiment.effects == (ExperimentEffectClass.OBSERVE_FIXTURE,)
    return (
        ExperimentEffectClass.OBSERVE_DISPOSABLE_WORKTREE in experiment.effects
        and not _forbidden_effects(experiment)
    )


def _worktree_cost_valid(experiment: ShadowExperiment) -> bool:
    if experiment.isolation.kind is IsolationKind.FIXTURE:
        return experiment.cost.worktree_count == 0 and experiment.execution_bound.max_worktrees == 0
    return experiment.cost.worktree_count <= 1


class ExperimentPlanner:
    """Skip experiments that cannot change a decision; refuse unsafe isolation."""

    revision: ClassVar[str] = PLANNER_REVISION

    def plan(
        self,
        experiment: ShadowExperiment,
        *,
        pending_decision: PendingDecision | None = None,
        questions: Sequence[UncertaintyQuestion] = (),
        world: RepositoryWorldState | None = None,
        family: TaskFamily | None = None,
        remaining_budget: ExperimentCost | None = None,
        emitted_at_ms: int = 0,
    ) -> ExperimentDecision:
        return self.plan_experiment(
            experiment,
            pending_decision=pending_decision,
            questions=questions,
            world=world,
            family=family,
            remaining_budget=remaining_budget,
            emitted_at_ms=emitted_at_ms,
        ).decision

    def plan_experiment(
        self,
        experiment: ShadowExperiment,
        *,
        pending_decision: PendingDecision | None = None,
        questions: Sequence[UncertaintyQuestion] = (),
        world: RepositoryWorldState | None = None,
        family: TaskFamily | None = None,
        remaining_budget: ExperimentCost | None = None,
        emitted_at_ms: int = 0,
    ) -> PlannedExperiment:
        if not isinstance(experiment, ShadowExperiment):
            raise ExperimentDeclarationError("experiment must be ShadowExperiment")
        action = ExperimentAction.REFUSE
        reason = ExperimentReason.MISSING_DECLARATION
        value = 0
        reachable: tuple[str, ...] = ()
        decision_id = experiment.decision_id
        isolation_reason = experiment.isolation.refusal_reason()
        open_questions = extract_uncertainty_questions(
            world=world, family=family, questions=questions
        )
        open_ids = {item.question_id for item in open_questions}

        if world is not None and world.bindings != experiment.bindings:
            reason = ExperimentReason.BINDING_MISMATCH
        elif family is not None and family.bindings != experiment.bindings:
            reason = ExperimentReason.BINDING_MISMATCH
        elif isolation_reason is not None:
            reason = isolation_reason
        elif _forbidden_effects(experiment):
            reason = ExperimentReason.FORBIDDEN_EFFECT
        elif not _effect_matches_isolation(experiment):
            reason = ExperimentReason.FORBIDDEN_EFFECT
        elif _risk_rank(experiment.risk_class) > _risk_rank(RISK_CEILING):
            reason = ExperimentReason.RISK_CEILING
        elif (
            experiment.risk_class is RiskClass.REVERSIBLE_LOCAL
            and experiment.isolation.kind is IsolationKind.FIXTURE
        ):
            reason = ExperimentReason.RISK_CEILING
        elif not _privacy_complete(experiment):
            reason = ExperimentReason.PRIVACY_VIOLATION
        elif not experiment.execution_bound.admits(experiment.cost) or not _worktree_cost_valid(
            experiment
        ):
            reason = (
                ExperimentReason.UNBOUNDED
                if experiment.cost.worktree_count > experiment.execution_bound.max_worktrees
                else ExperimentReason.COST_EXCEEDS_BOUND
            )
        elif pending_decision is None:
            action = ExperimentAction.SKIP
            reason = ExperimentReason.NO_PENDING_DECISION
        else:
            decision_id = pending_decision.decision_id
            if (
                experiment.decision_id
                and experiment.decision_id != pending_decision.decision_id
            ):
                reason = ExperimentReason.BINDING_MISMATCH
            elif pending_decision.committed_option_id:
                action = ExperimentAction.SKIP
                reason = ExperimentReason.ALREADY_ANSWERED
            elif experiment.question_id not in pending_decision.question_ids:
                action = ExperimentAction.SKIP
                reason = ExperimentReason.QUESTION_NOT_RELEVANT
            elif experiment.question_id not in open_ids:
                action = ExperimentAction.SKIP
                reason = ExperimentReason.QUESTION_NOT_OPEN
            else:
                value, reachable = value_of_experiment(experiment, pending_decision)
                if remaining_budget is not None and experiment.cost.exceeds(remaining_budget):
                    action = ExperimentAction.SKIP
                    reason = ExperimentReason.COST_EXCEEDS_BUDGET
                elif value <= 0:
                    action = ExperimentAction.SKIP
                    reason = ExperimentReason.CANNOT_CHANGE_DECISION
                else:
                    action = ExperimentAction.RUN
                    reason = ExperimentReason.DECISION_RELEVANT

        state = ArtifactState.SHADOW
        if action is ExperimentAction.REFUSE:
            state = ArtifactState.REJECTED
        elif action is ExperimentAction.SKIP:
            state = ArtifactState.CANDIDATE
        facts = experiment.to_facts()
        facts["action"] = action.value
        facts["reason_code"] = reason.value
        facts["value_of_experiment"] = value
        facts["reachable_option_ids"] = reachable
        plan_artifact = ExperimentPlanArtifact(
            bindings=experiment.bindings,
            state=state,
            subject_cid=experiment.experiment_id,
            reference_cids=(experiment.question_id, *experiment.required_data_ids),
            labels=(action.value, experiment.isolation.kind.value, experiment.risk_class.value),
            facts=facts,
            created_at_ms=emitted_at_ms,
        )
        decision = ExperimentDecision(
            bindings=experiment.bindings,
            experiment_id=experiment.experiment_id,
            action=action,
            reason_code=reason,
            question_id=experiment.question_id,
            decision_id=decision_id,
            value_of_experiment=value,
            reachable_option_ids=reachable,
            isolation=experiment.isolation,
            risk_class=experiment.risk_class,
            estimated_cost=experiment.cost,
            execution_bound=experiment.execution_bound,
            decision_rule=experiment.decision_rule,
            privacy_classes=experiment.privacy_classes,
            plan_cid=plan_artifact.content_id,
        )
        return PlannedExperiment(
            decision=decision, plan_artifact=plan_artifact, experiment=experiment
        )


def _evaluate_rule(rule: DecisionRule, observed_facts: Mapping[str, Any]) -> tuple[ExperimentOutcome, str, bool]:
    if rule.observation_binding not in observed_facts:
        raise ExperimentObservationError("required observation binding is absent from fixture facts")
    observed = observed_facts[rule.observation_binding]
    if observed == rule.hypothesis_operand:
        return ExperimentOutcome.HYPOTHESIS, rule.hypothesis_option_id, True
    if observed == rule.counterfactual_operand:
        return ExperimentOutcome.COUNTERFACTUAL, rule.counterfactual_option_id, False
    if rule.rule_class is DecisionRuleClass.INTEGER_THRESHOLD:
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise ExperimentObservationError("integer-threshold rules require an integer observation")
        if not isinstance(rule.hypothesis_operand, int) or isinstance(rule.hypothesis_operand, bool):
            raise ExperimentDeclarationError("integer-threshold hypothesis operand must be an integer")
        if observed >= rule.hypothesis_operand:
            return ExperimentOutcome.HYPOTHESIS, rule.hypothesis_option_id, True
        return ExperimentOutcome.COUNTERFACTUAL, rule.counterfactual_option_id, False
    if rule.rule_class is DecisionRuleClass.CLOSED_MEMBERSHIP:
        membership = rule.hypothesis_operand
        if isinstance(membership, (str, bytes, bytearray, memoryview)) or not isinstance(
            membership, Sequence
        ):
            raise ExperimentDeclarationError("closed-membership operand must be a sequence")
        if observed in membership:
            return ExperimentOutcome.HYPOTHESIS, rule.hypothesis_option_id, True
        return ExperimentOutcome.COUNTERFACTUAL, rule.counterfactual_option_id, False
    return ExperimentOutcome.INCONCLUSIVE, "", False


class ShadowExperimentRunner:
    """Execute only planner-admitted experiments on fixtures or disposable worktrees."""

    revision: ClassVar[str] = RUNNER_REVISION

    def run(
        self,
        decision: ExperimentDecision,
        experiment: ShadowExperiment,
        *,
        observed_facts: Mapping[str, Any] | None = None,
        emitted_at_ms: int = 0,
    ) -> ExperimentObservationRecord:
        return self.run_experiment(
            decision,
            experiment,
            observed_facts=observed_facts,
            emitted_at_ms=emitted_at_ms,
        ).observation

    def run_experiment(
        self,
        decision: ExperimentDecision,
        experiment: ShadowExperiment,
        *,
        observed_facts: Mapping[str, Any] | None = None,
        emitted_at_ms: int = 0,
    ) -> ExperimentRunResult:
        if not isinstance(decision, ExperimentDecision):
            raise ExperimentError("runner requires an ExperimentDecision")
        if not isinstance(experiment, ShadowExperiment):
            raise ExperimentDeclarationError("runner requires a ShadowExperiment")
        if decision.experiment_id != experiment.experiment_id:
            raise ExperimentError("decision and experiment identities differ")
        if decision.bindings != experiment.bindings:
            raise ExperimentError("decision and experiment exact bindings differ")
        isolation_reason = experiment.isolation.refusal_reason()
        if isolation_reason is not None:
            raise ExperimentIsolationError(
                f"shadow experiments cannot run on {isolation_reason.value} targets"
            )
        if decision.action is not ExperimentAction.RUN:
            raise ExperimentIsolationError(
                "shadow runner will not execute a skipped or refused experiment"
            )
        if decision.can_authorize or decision.value_of_experiment <= 0:
            raise ExperimentError("only decision-relevant non-authorizing experiments may run")
        facts = _freeze(observed_facts or {}, "observed_facts")
        if not isinstance(facts, Mapping):
            raise ExperimentObservationError("observed_facts must be a mapping")
        if len(facts) > MAX_MAPPING_ITEMS:
            raise ExperimentObservationError("observed_facts exceeds its mapping bound")
        _reject_privacy(facts, "observed_facts")
        missing = [item for item in experiment.required_data_ids if item not in facts]
        if missing:
            raise ExperimentObservationError("required experiment data is missing from the fixture")
        outcome, selected, supported = _evaluate_rule(experiment.decision_rule, facts)
        observation = ExperimentObservationRecord(
            bindings=experiment.bindings,
            experiment_id=experiment.experiment_id,
            question_id=experiment.question_id,
            outcome=outcome,
            selected_option_id=selected,
            observed_facts=facts,
            isolation=experiment.isolation,
            hypothesis_supported=supported,
        )
        observation_artifact = observation.to_artifact(emitted_at_ms=emitted_at_ms)
        evaluation_artifact = ExperimentEvaluationArtifact(
            bindings=experiment.bindings,
            state=ArtifactState.CANDIDATE,
            subject_cid=experiment.experiment_id,
            reference_cids=(observation_artifact.content_id, decision.plan_cid),
            labels=(outcome.value, "observation-only"),
            facts={
                "decision_id": decision.decision_id,
                "action": decision.action.value,
                "outcome": outcome.value,
                "selected_option_id": selected,
                "value_of_experiment": decision.value_of_experiment,
                "changed_pending_decision": bool(selected)
                and selected in decision.reachable_option_ids,
                "can_authorize": False,
                "can_grant_authority": False,
                "can_promote": False,
                "mutated_production": False,
                "mutated_policy": False,
            },
            created_at_ms=emitted_at_ms,
        )
        return ExperimentRunResult(
            observation=observation,
            observation_artifact=observation_artifact,
            evaluation_artifact=evaluation_artifact,
        )


__all__ = [
    "ALLOWED_EXPERIMENT_EFFECTS",
    "PLANNER_REVISION",
    "REQUIRED_DECLARATION_FIELDS",
    "REQUIRED_PRIVACY_CLASSES",
    "RISK_CEILING",
    "RUNNER_REVISION",
    "DecisionRule",
    "DecisionRuleClass",
    "ExecutionBound",
    "ExperimentAction",
    "ExperimentCost",
    "ExperimentDeclarationError",
    "ExperimentDecision",
    "ExperimentEffectClass",
    "ExperimentError",
    "ExperimentEvaluationArtifact",
    "ExperimentIsolationError",
    "ExperimentObservationArtifact",
    "ExperimentObservationError",
    "ExperimentObservationRecord",
    "ExperimentOutcome",
    "ExperimentPlanArtifact",
    "ExperimentPlanner",
    "ExperimentReason",
    "ExperimentRunResult",
    "IsolationKind",
    "IsolationTarget",
    "ObservationUse",
    "PendingDecision",
    "PlannedExperiment",
    "PrivacyClass",
    "ShadowExperiment",
    "ShadowExperimentRunner",
    "UncertaintyQuestion",
    "extract_uncertainty_questions",
    "missing_declaration_fields",
    "observation_may_discharge",
    "value_of_experiment",
]
