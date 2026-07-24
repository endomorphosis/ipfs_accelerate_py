"""Deterministic generation and independent verification of goal refinements.

This module is the trust boundary between an unverified decomposition (which
may have been suggested by Leanstral) and formal-plan admission.  It derives a
closed set of semantic obligations from a canonical :class:`FormalWorkPlan`,
routes each obligation through :mod:`multi_prover_router`, and records every
selected prover attempt.  A model suggestion is never treated as proof.

The generated statements are canonical JSON models, not natural-language
formulae.  Translators and prover runners consume the statement together with
the referenced, reviewed formula and record identifiers.  The generator does
not invent predicates, premises, acceptance criteria, effects, or authority.

Repair is deliberately narrow.  At most two policy-controlled Leanstral
rounds may replace the proposed plan.  A content-addressed receipt proves that
the root goal and the complete assumption set are byte-for-byte unchanged
before a repaired plan is allowed to produce another verification round.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Protocol

from .formal_planning_contracts import (
    FormalWorkPlan,
    Goal,
    Norm,
    RefinementMode,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    _canonical_value,
    canonical_json,
    canonical_json_bytes,
    content_identity,
)
from .multi_prover_router import (
    AttemptOutcome,
    MultiProverRouter,
    PortfolioAttempt,
    PortfolioResult,
    PortfolioRunner,
    PortfolioVerdict,
    PropertyKind,
    PropertyObligation,
)


GOAL_REFINEMENT_VERIFICATION_VERSION = 1
REFINEMENT_OBLIGATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refinement-obligation@1"
)
FROZEN_REFINEMENT_CONTEXT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/frozen-refinement-context@1"
)
REFINEMENT_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refinement-verification-policy@1"
)
REFINEMENT_ATTEMPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refinement-verification-attempt@1"
)
REFINEMENT_COUNTEREXAMPLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refinement-counterexample@1"
)
REFINEMENT_REPAIR_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refinement-repair-immutability@1"
)
REFINEMENT_ROUND_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refinement-verification-round@1"
)
REFINEMENT_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refinement-verification-result@1"
)
MAX_LEANSTRAL_REPAIR_ROUNDS = 2
DEFAULT_MAX_COUNTEREXAMPLE_BYTES = 256 * 1024


class RefinementObligationKind(str, Enum):
    """Closed vocabulary of checks required before refinement admission."""

    CHILD_TO_PARENT = "child_to_parent"
    ACCEPTANCE_CRITERION_COVERAGE = "acceptance_criterion_coverage"
    EVIDENCE_PRODUCTION = "evidence_production"
    TASK_EFFECT_SUFFICIENCY = "task_effect_sufficiency"
    DEPENDENCY_LIVENESS = "dependency_liveness"
    AUTHORITY = "authority"
    RESOURCE_FEASIBILITY = "resource_feasibility"
    DEONTIC_CONSISTENCY = "deontic_consistency"


class RefinementVerificationStatus(str, Enum):
    VERIFIED = "verified"
    DISPROVED = "disproved"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


class RefinementPersistenceError(RuntimeError):
    """The append-only audit journal could not retain a verification event."""


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ContractValidationError(f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise ContractValidationError(f"{name} must not be empty")
    if "\x00" in value:
        raise ContractValidationError(f"{name} must not contain NUL bytes")
    return value


def _strings(
    value: Iterable[Any] | None, name: str, *, required: bool = False
) -> tuple[str, ...]:
    if value is None:
        result: tuple[str, ...] = ()
    elif isinstance(value, (str, bytes, bytearray, memoryview)):
        raise ContractValidationError(f"{name} must be a sequence")
    else:
        result = tuple(sorted({_text(item, name) for item in value}))
    if required and not result:
        raise ContractValidationError(f"{name} must not be empty")
    return result


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ContractValidationError(f"{name} must be an object with string keys")
    result = _canonical_value(dict(value))
    if not isinstance(result, dict):  # pragma: no cover - canonical invariant
        raise ContractValidationError(f"{name} must be an object")
    return result


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{name} is unsupported") from exc


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    if not isinstance(payload, Mapping):
        raise ContractValidationError("contract payload must be an object")
    if payload.get("schema") not in (None, "", expected):
        raise ContractValidationError(f"unsupported schema; expected {expected}")


def _identity(payload: Mapping[str, Any], actual: str, noun: str) -> None:
    if payload.get("content_id") not in (None, "", actual):
        raise ContractValidationError(f"{noun} content identity does not match")


def _nonnegative(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ContractValidationError(f"{name} must not exceed {maximum}")
    return value


_PROPERTY_BY_OBLIGATION: Mapping[RefinementObligationKind, PropertyKind] = {
    RefinementObligationKind.CHILD_TO_PARENT: PropertyKind.FIRST_ORDER_THEOREM,
    RefinementObligationKind.ACCEPTANCE_CRITERION_COVERAGE: (
        PropertyKind.TYPED_PLANNING
    ),
    RefinementObligationKind.EVIDENCE_PRODUCTION: PropertyKind.STATE_MACHINE,
    RefinementObligationKind.TASK_EFFECT_SUFFICIENCY: (
        PropertyKind.FIRST_ORDER_THEOREM
    ),
    RefinementObligationKind.DEPENDENCY_LIVENESS: (
        PropertyKind.TEMPORAL_DEONTIC
    ),
    RefinementObligationKind.AUTHORITY: PropertyKind.FINITE_CONSTRAINT,
    RefinementObligationKind.RESOURCE_FEASIBILITY: PropertyKind.FINITE_CONSTRAINT,
    RefinementObligationKind.DEONTIC_CONSISTENCY: PropertyKind.TEMPORAL_DEONTIC,
}


def property_kind_for_refinement_obligation(
    obligation_kind: RefinementObligationKind | str,
) -> PropertyKind:
    """Return the reviewed property family for a refinement check."""

    kind = _enum(
        obligation_kind, RefinementObligationKind, "refinement_obligation_kind"
    )
    return _PROPERTY_BY_OBLIGATION[kind]


def _required_assurance(property_kind: PropertyKind) -> AssuranceLevel:
    # The default router grants bounded authority only to finite-constraint
    # and state-machine model checkers.  All logical, planning, temporal and
    # deontic claims require independent kernel reconstruction.
    if property_kind in (
        PropertyKind.FINITE_CONSTRAINT,
        PropertyKind.STATE_MACHINE,
    ):
        return AssuranceLevel.SOLVER_CHECKED
    return AssuranceLevel.KERNEL_VERIFIED


@dataclass(frozen=True)
class RefinementObligation(CanonicalContract):
    """A content-addressed refinement check and its prover classification."""

    SCHEMA: ClassVar[str] = REFINEMENT_OBLIGATION_SCHEMA

    obligation_id: str
    kind: RefinementObligationKind
    property_kind: PropertyKind
    subject_ids: tuple[str, ...]
    statement_model: Mapping[str, Any]
    premise_ids: tuple[str, ...]
    required_assurance: AssuranceLevel
    root_goal_id: str
    root_goal_content_id: str
    assumption_ids: tuple[str, ...]
    plan_id: str

    def __post_init__(self) -> None:
        for name in (
            "obligation_id",
            "root_goal_id",
            "root_goal_content_id",
            "plan_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        kind = _enum(self.kind, RefinementObligationKind, "kind")
        property_kind = _enum(self.property_kind, PropertyKind, "property_kind")
        if property_kind is not property_kind_for_refinement_obligation(kind):
            raise ContractValidationError(
                "refinement obligation has the wrong property family"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "property_kind", property_kind)
        object.__setattr__(
            self, "subject_ids", _strings(self.subject_ids, "subject_ids", required=True)
        )
        object.__setattr__(
            self, "premise_ids", _strings(self.premise_ids, "premise_ids")
        )
        object.__setattr__(
            self, "assumption_ids", _strings(self.assumption_ids, "assumption_ids")
        )
        object.__setattr__(
            self, "statement_model", _mapping(self.statement_model, "statement_model")
        )
        object.__setattr__(
            self,
            "required_assurance",
            _enum(self.required_assurance, AssuranceLevel, "required_assurance"),
        )
        if not self.required_assurance.satisfies(_required_assurance(property_kind)):
            raise ContractValidationError(
                "refinement obligation assurance is weaker than its property policy"
            )

    @property
    def statement(self) -> str:
        return canonical_json(self.statement_model)

    def to_property_obligation(self) -> PropertyObligation:
        """Project into the provider-independent multi-prover contract."""

        return PropertyObligation(
            obligation_id=self.obligation_id,
            property_kind=self.property_kind,
            statement=self.statement,
            premise_ids=tuple(sorted({*self.premise_ids, *self.assumption_ids})),
            required_assurance=self.required_assurance,
            metadata={
                "refinement_obligation_id": self.content_id,
                "refinement_kind": self.kind.value,
                "root_goal_id": self.root_goal_id,
                "root_goal_content_id": self.root_goal_content_id,
                "assumption_ids": list(self.assumption_ids),
                "plan_id": self.plan_id,
                "subject_ids": list(self.subject_ids),
                "bounds": self.statement_model.get("bounds", {}),
            },
        )

    as_property_obligation = to_property_obligation

    def _payload(self) -> dict[str, Any]:
        return {
            "verification_version": GOAL_REFINEMENT_VERIFICATION_VERSION,
            "obligation_id": self.obligation_id,
            "kind": self.kind,
            "property_kind": self.property_kind,
            "subject_ids": self.subject_ids,
            "statement_model": self.statement_model,
            "premise_ids": self.premise_ids,
            "required_assurance": self.required_assurance,
            "root_goal_id": self.root_goal_id,
            "root_goal_content_id": self.root_goal_content_id,
            "assumption_ids": self.assumption_ids,
            "plan_id": self.plan_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefinementObligation":
        _schema(payload, cls.SCHEMA)
        result = cls(
            obligation_id=payload.get("obligation_id", ""),
            kind=payload.get("kind", ""),
            property_kind=payload.get("property_kind", ""),
            subject_ids=tuple(payload.get("subject_ids") or ()),
            statement_model=payload.get("statement_model") or {},
            premise_ids=tuple(payload.get("premise_ids") or ()),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.UNVERIFIED
            ),
            root_goal_id=payload.get("root_goal_id", ""),
            root_goal_content_id=payload.get("root_goal_content_id", ""),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            plan_id=payload.get("plan_id", ""),
        )
        _identity(payload, result.content_id, "refinement obligation")
        return result


@dataclass(frozen=True)
class FrozenRefinementContext(CanonicalContract):
    """Semantic values that a model-assisted repair is forbidden to change."""

    SCHEMA: ClassVar[str] = FROZEN_REFINEMENT_CONTEXT_SCHEMA

    root_goal_id: str
    root_goal_content_id: str
    assumption_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "root_goal_id", _text(self.root_goal_id, "root_goal_id")
        )
        object.__setattr__(
            self,
            "root_goal_content_id",
            _text(self.root_goal_content_id, "root_goal_content_id"),
        )
        object.__setattr__(
            self, "assumption_ids", _strings(self.assumption_ids, "assumption_ids")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "verification_version": GOAL_REFINEMENT_VERIFICATION_VERSION,
            "root_goal_id": self.root_goal_id,
            "root_goal_content_id": self.root_goal_content_id,
            "assumption_ids": self.assumption_ids,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenRefinementContext":
        _schema(payload, cls.SCHEMA)
        result = cls(
            root_goal_id=payload.get("root_goal_id", ""),
            root_goal_content_id=payload.get("root_goal_content_id", ""),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
        )
        _identity(payload, result.content_id, "frozen refinement context")
        return result


@dataclass(frozen=True)
class RefinementVerificationPolicy(CanonicalContract):
    """Bounds and effect controls for independent refinement checking."""

    SCHEMA: ClassVar[str] = REFINEMENT_POLICY_SCHEMA

    max_repair_rounds: int = 0
    allow_leanstral_repairs: bool = False
    max_counterexample_bytes: int = DEFAULT_MAX_COUNTEREXAMPLE_BYTES
    stop_on_counterexample: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_repair_rounds",
            _nonnegative(
                self.max_repair_rounds,
                "max_repair_rounds",
                maximum=MAX_LEANSTRAL_REPAIR_ROUNDS,
            ),
        )
        if not isinstance(self.allow_leanstral_repairs, bool):
            raise ContractValidationError("allow_leanstral_repairs must be boolean")
        if self.max_repair_rounds and not self.allow_leanstral_repairs:
            raise ContractValidationError(
                "repair rounds require allow_leanstral_repairs"
            )
        if isinstance(self.max_counterexample_bytes, bool) or not isinstance(
            self.max_counterexample_bytes, int
        ) or self.max_counterexample_bytes < 1:
            raise ContractValidationError(
                "max_counterexample_bytes must be positive"
            )
        if not isinstance(self.stop_on_counterexample, bool):
            raise ContractValidationError("stop_on_counterexample must be boolean")

    def _payload(self) -> dict[str, Any]:
        return {
            "verification_version": GOAL_REFINEMENT_VERIFICATION_VERSION,
            "max_repair_rounds": self.max_repair_rounds,
            "allow_leanstral_repairs": self.allow_leanstral_repairs,
            "max_counterexample_bytes": self.max_counterexample_bytes,
            "stop_on_counterexample": self.stop_on_counterexample,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefinementVerificationPolicy":
        _schema(payload, cls.SCHEMA)
        result = cls(
            max_repair_rounds=payload.get("max_repair_rounds", 0),
            allow_leanstral_repairs=payload.get("allow_leanstral_repairs", False),
            max_counterexample_bytes=payload.get(
                "max_counterexample_bytes", DEFAULT_MAX_COUNTEREXAMPLE_BYTES
            ),
            stop_on_counterexample=payload.get("stop_on_counterexample", True),
        )
        _identity(payload, result.content_id, "refinement verification policy")
        return result


def _root(plan: FormalWorkPlan, root_goal_id: str) -> Goal:
    if not isinstance(plan, FormalWorkPlan):
        raise ContractValidationError("plan must be a FormalWorkPlan")
    if root_goal_id:
        matches = [item for item in plan.goals if item.goal_id == root_goal_id]
    elif len(plan.goals) == 1:
        matches = [plan.goals[0]]
    else:
        raise ContractValidationError(
            "root_goal_id is required for plans with multiple goals"
        )
    if not matches:
        raise ContractValidationError("root_goal_id is not present in the plan")
    return matches[0]


def _resource_ids(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    for key in (
        "resource_requirement_ids",
        "resource_ids",
        "resource_needs",
        "resource_spec",
        "required_resources",
        "resources",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return (value.strip(),) if value.strip() else ()
        if isinstance(value, Sequence) and not isinstance(
            value, (bytes, bytearray, memoryview)
        ):
            return _strings(value, key)
        if isinstance(value, Mapping):
            return _strings(value.keys(), key)
    return ()


class GoalRefinementObligationGenerator:
    """Pure deterministic projection from a canonical plan to obligations."""

    def derive(
        self,
        plan: FormalWorkPlan,
        *,
        root_goal_id: str = "",
        root_goal_content_id: str = "",
        assumption_ids: Iterable[str] = (),
    ) -> tuple[RefinementObligation, ...]:
        root = _root(plan, root_goal_id)
        expected_root_content_id = root.content_id
        supplied_root_content_id = (
            _text(
                root_goal_content_id,
                "root_goal_content_id",
                required=False,
            )
            or expected_root_content_id
        )
        if supplied_root_content_id != expected_root_content_id:
            raise ContractValidationError(
                "frozen root content identity does not match the plan root"
            )
        assumptions = _strings(assumption_ids, "assumption_ids")
        plan_id = plan.content_id
        goals = {item.goal_id: item for item in plan.goals}
        subgoals = {item.subgoal_id: item for item in plan.subgoals}
        tasks_by_subgoal = {
            subgoal_id: tuple(
                task for task in plan.tasks if task.subgoal_id == subgoal_id
            )
            for subgoal_id in subgoals
        }
        effects = {item.effect_id: item for item in plan.effects}
        actors = {item.actor_id: item for item in plan.actors}
        norms_by_action: dict[str, list[Norm]] = {}
        for norm in plan.norms:
            norms_by_action.setdefault(norm.action_id, []).append(norm)

        seeds: list[
            tuple[
                RefinementObligationKind,
                tuple[str, ...],
                Mapping[str, Any],
                tuple[str, ...],
            ]
        ] = []
        bounds = {"trace_bound": plan.trace_bound}

        for subgoal in sorted(plan.subgoals, key=lambda item: item.subgoal_id):
            parent = (
                goals[subgoal.parent_id]
                if subgoal.parent_id in goals
                else subgoals[subgoal.parent_id]
            )
            directions = ["child_implies_parent"]
            if subgoal.refinement_mode is RefinementMode.EQUIVALENT:
                directions.append("parent_implies_child")
            parent_formula = parent.satisfaction_formula_id
            seeds.append(
                (
                    RefinementObligationKind.CHILD_TO_PARENT,
                    (subgoal.subgoal_id, subgoal.parent_id),
                    {
                        "relation": subgoal.refinement_mode.value,
                        "directions": directions,
                        "child_formula_id": subgoal.satisfaction_formula_id,
                        "parent_formula_id": parent_formula,
                        "bounds": bounds,
                    },
                    (subgoal.satisfaction_formula_id, parent_formula),
                )
            )

            parent_requirements = set(parent.evidence_requirement_ids)
            child_requirements = set(subgoal.evidence_requirement_ids)
            for task in tasks_by_subgoal[subgoal.subgoal_id]:
                child_requirements.update(task.evidence_requirement_ids)
            seeds.append(
                (
                    RefinementObligationKind.ACCEPTANCE_CRITERION_COVERAGE,
                    (subgoal.subgoal_id, subgoal.parent_id),
                    {
                        "required_criterion_ids": sorted(parent_requirements),
                        "covered_criterion_ids": sorted(child_requirements),
                        "coverage_relation": "required_subset_of_covered",
                        "bounds": bounds,
                    },
                    tuple(sorted(parent_requirements | child_requirements)),
                )
            )

            seeds.append(
                (
                    RefinementObligationKind.DEPENDENCY_LIVENESS,
                    (subgoal.subgoal_id,),
                    {
                        "subject_id": subgoal.subgoal_id,
                        "dependency_ids": list(subgoal.depends_on),
                        "liveness": "ready_dependencies_eventually_enable_subject",
                        "terminal_states": ["satisfied"],
                        "bounds": bounds,
                    },
                    tuple(subgoal.depends_on),
                )
            )

        for requirement in sorted(
            plan.evidence_requirements, key=lambda item: item.requirement_id
        ):
            producing_tasks = tuple(
                sorted(
                    task.task_id
                    for task in plan.tasks
                    if requirement.requirement_id in task.evidence_requirement_ids
                    or task.task_id in requirement.subject_ids
                )
            )
            seeds.append(
                (
                    RefinementObligationKind.EVIDENCE_PRODUCTION,
                    (requirement.requirement_id, *requirement.subject_ids),
                    {
                        "requirement_id": requirement.requirement_id,
                        "requirement_kind": requirement.kind.value,
                        "subject_ids": list(requirement.subject_ids),
                        "producing_task_ids": producing_tasks,
                        "required_assurance": (
                            requirement.minimum_code_assurance.value
                        ),
                        "freshness_seconds": requirement.freshness_seconds,
                        "fallback_check_ids": list(requirement.fallback_check_ids),
                        "transition": "task_completion_leads_to_fresh_evidence",
                        "bounds": bounds,
                    },
                    tuple(requirement.fallback_check_ids),
                )
            )

        for task in sorted(plan.tasks, key=lambda item: item.task_id):
            target = (
                subgoals[task.subgoal_id].satisfaction_formula_id
                if task.subgoal_id
                else goals[task.goal_id].satisfaction_formula_id
            )
            seeds.append(
                (
                    RefinementObligationKind.TASK_EFFECT_SUFFICIENCY,
                    (task.task_id, task.subgoal_id or task.goal_id),
                    {
                        "task_id": task.task_id,
                        "effect_ids": list(task.effect_ids),
                        "effects": [
                            {
                                "effect_id": effects[effect_id].effect_id,
                                "operation": effects[effect_id].operation.value,
                                "fluent_id": effects[effect_id].fluent_id,
                                "event_id": effects[effect_id].event_id,
                                "value": effects[effect_id].value,
                            }
                            for effect_id in task.effect_ids
                        ],
                        "target_satisfaction_formula_id": target,
                        "relation": "completed_task_effects_imply_target",
                        "bounds": bounds,
                    },
                    tuple(sorted({target, *task.effect_ids})),
                )
            )
            seeds.append(
                (
                    RefinementObligationKind.DEPENDENCY_LIVENESS,
                    (task.task_id,),
                    {
                        "subject_id": task.task_id,
                        "dependency_ids": list(task.depends_on),
                        "liveness": "ready_dependencies_eventually_enable_subject",
                        "terminal_states": list(task.terminal_states),
                        "bounds": bounds,
                    },
                    tuple(task.depends_on),
                )
            )
            assigned = [actors[actor_id] for actor_id in task.actor_ids]
            seeds.append(
                (
                    RefinementObligationKind.AUTHORITY,
                    (task.task_id, *task.actor_ids),
                    {
                        "task_id": task.task_id,
                        "actors": [
                            {
                                "actor_id": actor.actor_id,
                                "principal_id": actor.principal_id,
                                "authority_ids": list(actor.authority_ids),
                                "capabilities": list(actor.capabilities),
                            }
                            for actor in assigned
                        ],
                        "required_authority_ids": list(
                            _strings(
                                task.metadata.get("required_authority_ids") or (),
                                "required_authority_ids",
                            )
                        ),
                        "relation": "assigned_actor_authorized_for_task",
                        "bounds": bounds,
                    },
                    tuple(
                        sorted(
                            {
                                authority_id
                                for actor in assigned
                                for authority_id in actor.authority_ids
                            }
                        )
                    ),
                )
            )
            available_resources = {
                *_resource_ids(plan.metadata),
                *(
                    capability
                    for actor in assigned
                    for capability in actor.capabilities
                ),
            }
            required_resources = set(_resource_ids(task.metadata))
            seeds.append(
                (
                    RefinementObligationKind.RESOURCE_FEASIBILITY,
                    (task.task_id,),
                    {
                        "task_id": task.task_id,
                        "required_resource_ids": sorted(required_resources),
                        "available_resource_ids": sorted(available_resources),
                        "relation": "required_resources_within_available_capacity",
                        "bounds": bounds,
                    },
                    tuple(sorted(required_resources | available_resources)),
                )
            )
            action_norms = sorted(
                (
                    *norms_by_action.get(task.task_id, ()),
                    *(
                        norm
                        for event_id in task.event_ids
                        for norm in norms_by_action.get(event_id, ())
                    ),
                ),
                key=lambda item: item.norm_id,
            )
            seeds.append(
                (
                    RefinementObligationKind.DEONTIC_CONSISTENCY,
                    (task.task_id, *(norm.norm_id for norm in action_norms)),
                    {
                        "task_id": task.task_id,
                        "norms": [
                            {
                                "norm_id": norm.norm_id,
                                "kind": norm.kind.value,
                                "bearer_actor_id": norm.bearer_actor_id,
                                "issuer_actor_id": norm.issuer_actor_id,
                                "activation_formula_id": norm.activation_formula_id,
                                "valid_from": norm.valid_from,
                                "valid_until": norm.valid_until,
                            }
                            for norm in action_norms
                        ],
                        "relation": "no_simultaneous_obligation_prohibition_conflict",
                        "bounds": bounds,
                    },
                    tuple(
                        sorted(
                            {
                                norm.activation_formula_id
                                for norm in action_norms
                                if norm.activation_formula_id
                            }
                        )
                    ),
                )
            )

        obligations: list[RefinementObligation] = []
        for kind, subject_ids, statement_model, premise_ids in seeds:
            property_kind = property_kind_for_refinement_obligation(kind)
            identity_seed = {
                "version": GOAL_REFINEMENT_VERIFICATION_VERSION,
                "kind": kind.value,
                "property_kind": property_kind.value,
                "subject_ids": sorted(set(subject_ids)),
                "statement_model": statement_model,
                "premise_ids": sorted(set(premise_ids)),
                "root_goal_id": root.goal_id,
                "root_goal_content_id": supplied_root_content_id,
                "assumption_ids": assumptions,
                "plan_id": plan_id,
            }
            obligations.append(
                RefinementObligation(
                    obligation_id=(
                        "refinement-obligation:" + content_identity(identity_seed)
                    ),
                    kind=kind,
                    property_kind=property_kind,
                    subject_ids=tuple(subject_ids),
                    statement_model=statement_model,
                    premise_ids=premise_ids,
                    required_assurance=_required_assurance(property_kind),
                    root_goal_id=root.goal_id,
                    root_goal_content_id=supplied_root_content_id,
                    assumption_ids=assumptions,
                    plan_id=plan_id,
                )
            )
        return tuple(
            sorted(
                obligations,
                key=lambda item: (
                    item.kind.value,
                    item.subject_ids,
                    item.obligation_id,
                ),
            )
        )

    generate = derive


def derive_refinement_obligations(
    plan: FormalWorkPlan,
    *,
    root_goal_id: str = "",
    root_goal_content_id: str = "",
    assumption_ids: Iterable[str] = (),
) -> tuple[RefinementObligation, ...]:
    """Convenience entry point for deterministic obligation generation."""

    return GoalRefinementObligationGenerator().derive(
        plan,
        root_goal_id=root_goal_id,
        root_goal_content_id=root_goal_content_id,
        assumption_ids=assumption_ids,
    )


@dataclass(frozen=True)
class RefinementVerificationAttempt(CanonicalContract):
    """One persisted prover lane attempt, bound to its round and obligation."""

    SCHEMA: ClassVar[str] = REFINEMENT_ATTEMPT_SCHEMA

    round_index: int
    obligation_id: str
    obligation_content_id: str
    portfolio_result_id: str
    attempt: PortfolioAttempt

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "round_index", _nonnegative(self.round_index, "round_index")
        )
        for name in (
            "obligation_id",
            "obligation_content_id",
            "portfolio_result_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not isinstance(self.attempt, PortfolioAttempt):
            raise ContractValidationError("attempt must be a PortfolioAttempt")

    @property
    def attempt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "verification_version": GOAL_REFINEMENT_VERIFICATION_VERSION,
            "round_index": self.round_index,
            "obligation_id": self.obligation_id,
            "obligation_content_id": self.obligation_content_id,
            "portfolio_result_id": self.portfolio_result_id,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefinementVerificationAttempt":
        _schema(payload, cls.SCHEMA)
        attempt = payload.get("attempt")
        if not isinstance(attempt, Mapping):
            raise ContractValidationError("verification attempt payload is required")
        result = cls(
            round_index=payload.get("round_index", -1),
            obligation_id=payload.get("obligation_id", ""),
            obligation_content_id=payload.get("obligation_content_id", ""),
            portfolio_result_id=payload.get("portfolio_result_id", ""),
            attempt=PortfolioAttempt.from_dict(attempt),
        )
        _identity(payload, result.content_id, "refinement verification attempt")
        return result


@dataclass(frozen=True)
class RefinementCounterexample(CanonicalContract):
    """A size-bounded, auditable witness that rejects an obligation."""

    SCHEMA: ClassVar[str] = REFINEMENT_COUNTEREXAMPLE_SCHEMA

    round_index: int
    obligation_id: str
    portfolio_attempt_id: str
    prover_id: str
    bounds: Mapping[str, Any]
    witness: Mapping[str, Any]
    maximum_bytes: int = DEFAULT_MAX_COUNTEREXAMPLE_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "round_index", _nonnegative(self.round_index, "round_index")
        )
        for name in ("obligation_id", "portfolio_attempt_id", "prover_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "bounds", _mapping(self.bounds, "bounds"))
        object.__setattr__(self, "witness", _mapping(self.witness, "witness"))
        if not self.bounds:
            raise ContractValidationError(
                "counterexample must retain the finite verification bounds"
            )
        if not self.witness:
            raise ContractValidationError("counterexample witness must not be empty")
        if isinstance(self.maximum_bytes, bool) or not isinstance(
            self.maximum_bytes, int
        ) or self.maximum_bytes < 1:
            raise ContractValidationError("maximum_bytes must be positive")
        measured = len(
            canonical_json_bytes({"bounds": self.bounds, "witness": self.witness})
        )
        if measured > self.maximum_bytes:
            raise ContractValidationError(
                f"counterexample exceeds maximum of {self.maximum_bytes} bytes"
            )

    @property
    def counterexample_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "verification_version": GOAL_REFINEMENT_VERIFICATION_VERSION,
            "round_index": self.round_index,
            "obligation_id": self.obligation_id,
            "portfolio_attempt_id": self.portfolio_attempt_id,
            "prover_id": self.prover_id,
            "bounds": self.bounds,
            "witness": self.witness,
            "maximum_bytes": self.maximum_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefinementCounterexample":
        _schema(payload, cls.SCHEMA)
        result = cls(
            round_index=payload.get("round_index", -1),
            obligation_id=payload.get("obligation_id", ""),
            portfolio_attempt_id=payload.get("portfolio_attempt_id", ""),
            prover_id=payload.get("prover_id", ""),
            bounds=payload.get("bounds") or {},
            witness=payload.get("witness") or {},
            maximum_bytes=payload.get(
                "maximum_bytes", DEFAULT_MAX_COUNTEREXAMPLE_BYTES
            ),
        )
        _identity(payload, result.content_id, "refinement counterexample")
        return result


class RefinementAuditSink(Protocol):
    def append(self, record: CanonicalContract) -> None:
        ...


class InMemoryRefinementAuditStore:
    """Thread-safe audit sink useful to embedders which persist the result."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(self, record: CanonicalContract) -> None:
        if not isinstance(record, CanonicalContract):
            raise ContractValidationError("audit record must be canonical")
        with self._lock:
            self._records.append(record.to_record())

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        with self._lock:
            return tuple(dict(item) for item in self._records)


class JsonlRefinementAuditStore:
    """Append-only JSONL journal with flush+fsync durability per record."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.exists() and not self.path.is_file():
            raise ContractValidationError("audit path must be a file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: CanonicalContract) -> None:
        if not isinstance(record, CanonicalContract):
            raise ContractValidationError("audit record must be canonical")
        encoded = canonical_json(record.to_record()) + "\n"
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
        except OSError as exc:
            raise RefinementPersistenceError(
                f"could not persist refinement audit record: {exc}"
            ) from exc

    def read_records(self) -> tuple[Mapping[str, Any], ...]:
        if not self.path.exists():
            return ()
        result: list[Mapping[str, Any]] = []
        try:
            with self._lock:
                for line_number, line in enumerate(
                    self.path.read_text(encoding="utf-8").splitlines(), 1
                ):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise RefinementPersistenceError(
                            f"audit record {line_number} is not an object"
                        )
                    result.append(value)
        except (OSError, json.JSONDecodeError) as exc:
            raise RefinementPersistenceError(
                f"could not read refinement audit journal: {exc}"
            ) from exc
        return tuple(result)


# Concise compatibility name for callers that describe the journal as a ledger.
RefinementVerificationLedger = JsonlRefinementAuditStore


@dataclass(frozen=True)
class RefinementRepairCandidate:
    """A Leanstral-produced plan candidate carrying its claimed frozen context."""

    plan: FormalWorkPlan
    frozen_context: FrozenRefinementContext
    producer_id: str = "leanstral"

    def __post_init__(self) -> None:
        if not isinstance(self.plan, FormalWorkPlan):
            raise ContractValidationError("repair candidate plan must be FormalWorkPlan")
        if not isinstance(self.frozen_context, FrozenRefinementContext):
            raise ContractValidationError(
                "repair candidate must carry FrozenRefinementContext"
            )
        producer = _text(self.producer_id, "producer_id").casefold()
        if not producer.startswith("leanstral"):
            raise ContractValidationError(
                "refinement repair candidates must come from Leanstral"
            )
        object.__setattr__(self, "producer_id", producer)


@dataclass(frozen=True)
class RefinementRepairRequest:
    round_index: int
    frozen_context: FrozenRefinementContext
    prior_plan_id: str
    failed_obligation_ids: tuple[str, ...]
    counterexamples: tuple[RefinementCounterexample, ...]

    def __post_init__(self) -> None:
        if not 1 <= self.round_index <= MAX_LEANSTRAL_REPAIR_ROUNDS:
            raise ContractValidationError("repair round index is outside policy bounds")
        if not isinstance(self.frozen_context, FrozenRefinementContext):
            raise ContractValidationError(
                "repair request requires a frozen refinement context"
            )
        object.__setattr__(
            self, "prior_plan_id", _text(self.prior_plan_id, "prior_plan_id")
        )
        object.__setattr__(
            self,
            "failed_obligation_ids",
            _strings(self.failed_obligation_ids, "failed_obligation_ids"),
        )
        if any(
            not isinstance(item, RefinementCounterexample)
            for item in self.counterexamples
        ):
            raise ContractValidationError(
                "counterexamples must contain RefinementCounterexample values"
            )


@dataclass(frozen=True)
class RepairImmutabilityReceipt(CanonicalContract):
    """Mechanical proof that a repair retained the frozen semantic inputs."""

    SCHEMA: ClassVar[str] = REFINEMENT_REPAIR_RECEIPT_SCHEMA

    round_index: int
    producer_id: str
    previous_plan_id: str
    candidate_plan_id: str
    frozen_context_id: str
    candidate_context_id: str
    root_unchanged: bool
    assumptions_unchanged: bool
    accepted: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.round_index <= MAX_LEANSTRAL_REPAIR_ROUNDS:
            raise ContractValidationError("repair receipt round is outside bounds")
        for name in (
            "producer_id",
            "previous_plan_id",
            "candidate_plan_id",
            "frozen_context_id",
            "candidate_context_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("root_unchanged", "assumptions_unchanged", "accepted"):
            if not isinstance(getattr(self, name), bool):
                raise ContractValidationError(f"{name} must be boolean")
        expected = self.root_unchanged and self.assumptions_unchanged
        if self.accepted != expected:
            raise ContractValidationError(
                "repair acceptance must be derived from frozen root and assumptions"
            )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "verification_version": GOAL_REFINEMENT_VERIFICATION_VERSION,
            "round_index": self.round_index,
            "producer_id": self.producer_id,
            "previous_plan_id": self.previous_plan_id,
            "candidate_plan_id": self.candidate_plan_id,
            "frozen_context_id": self.frozen_context_id,
            "candidate_context_id": self.candidate_context_id,
            "root_unchanged": self.root_unchanged,
            "assumptions_unchanged": self.assumptions_unchanged,
            "accepted": self.accepted,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairImmutabilityReceipt":
        _schema(payload, cls.SCHEMA)
        result = cls(
            round_index=payload.get("round_index", 0),
            producer_id=payload.get("producer_id", ""),
            previous_plan_id=payload.get("previous_plan_id", ""),
            candidate_plan_id=payload.get("candidate_plan_id", ""),
            frozen_context_id=payload.get("frozen_context_id", ""),
            candidate_context_id=payload.get("candidate_context_id", ""),
            root_unchanged=payload.get("root_unchanged", False),
            assumptions_unchanged=payload.get("assumptions_unchanged", False),
            accepted=payload.get("accepted", False),
            reason=payload.get("reason", ""),
        )
        _identity(payload, result.content_id, "repair immutability receipt")
        return result


@dataclass(frozen=True)
class RefinementVerificationRound(CanonicalContract):
    SCHEMA: ClassVar[str] = REFINEMENT_ROUND_SCHEMA

    round_index: int
    plan_id: str
    obligations: tuple[RefinementObligation, ...]
    portfolio_results: tuple[PortfolioResult, ...]
    attempts: tuple[RefinementVerificationAttempt, ...]
    counterexamples: tuple[RefinementCounterexample, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "round_index", _nonnegative(self.round_index, "round_index")
        )
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        if len(self.obligations) != len(self.portfolio_results):
            raise ContractValidationError(
                "every generated obligation must retain a portfolio result"
            )
        if any(not isinstance(item, RefinementObligation) for item in self.obligations):
            raise ContractValidationError(
                "obligations must contain RefinementObligation values"
            )
        if any(not isinstance(item, PortfolioResult) for item in self.portfolio_results):
            raise ContractValidationError(
                "portfolio_results must contain PortfolioResult values"
            )
        expected = sum(len(item.attempts) for item in self.portfolio_results)
        if len(self.attempts) != expected:
            raise ContractValidationError(
                "round must retain every selected portfolio attempt"
            )

    @property
    def proved(self) -> bool:
        return bool(self.portfolio_results) and all(
            item.verdict is PortfolioVerdict.PROVED
            for item in self.portfolio_results
        )

    @property
    def disproved(self) -> bool:
        return any(
            item.verdict is PortfolioVerdict.DISPROVED
            for item in self.portfolio_results
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "verification_version": GOAL_REFINEMENT_VERIFICATION_VERSION,
            "round_index": self.round_index,
            "plan_id": self.plan_id,
            "obligations": self.obligations,
            "portfolio_results": self.portfolio_results,
            "attempts": self.attempts,
            "counterexamples": self.counterexamples,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefinementVerificationRound":
        _schema(payload, cls.SCHEMA)
        result = cls(
            round_index=payload.get("round_index", -1),
            plan_id=payload.get("plan_id", ""),
            obligations=tuple(
                item
                if isinstance(item, RefinementObligation)
                else RefinementObligation.from_dict(item)
                for item in (payload.get("obligations") or ())
            ),
            portfolio_results=tuple(
                item
                if isinstance(item, PortfolioResult)
                else PortfolioResult.from_dict(item)
                for item in (payload.get("portfolio_results") or ())
            ),
            attempts=tuple(
                item
                if isinstance(item, RefinementVerificationAttempt)
                else RefinementVerificationAttempt.from_dict(item)
                for item in (payload.get("attempts") or ())
            ),
            counterexamples=tuple(
                item
                if isinstance(item, RefinementCounterexample)
                else RefinementCounterexample.from_dict(item)
                for item in (payload.get("counterexamples") or ())
            ),
        )
        _identity(payload, result.content_id, "refinement verification round")
        return result


@dataclass(frozen=True)
class RefinementVerificationResult(CanonicalContract):
    SCHEMA: ClassVar[str] = REFINEMENT_RESULT_SCHEMA

    frozen_context: FrozenRefinementContext
    policy: RefinementVerificationPolicy
    status: RefinementVerificationStatus
    rounds: tuple[RefinementVerificationRound, ...]
    repair_receipts: tuple[RepairImmutabilityReceipt, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_context, FrozenRefinementContext):
            raise ContractValidationError("result requires FrozenRefinementContext")
        if not isinstance(self.policy, RefinementVerificationPolicy):
            raise ContractValidationError(
                "result requires RefinementVerificationPolicy"
            )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, RefinementVerificationStatus, "status"),
        )
        if not self.rounds or any(
            not isinstance(item, RefinementVerificationRound)
            for item in self.rounds
        ):
            raise ContractValidationError("result requires verification rounds")
        if len(self.rounds) > self.policy.max_repair_rounds + 1:
            raise ContractValidationError("result exceeds the repair-round policy")
        accepted_receipts = tuple(
            item for item in self.repair_receipts if item.accepted
        )
        rejected_receipts = tuple(
            item for item in self.repair_receipts if not item.accepted
        )
        if len(accepted_receipts) != len(self.rounds) - 1:
            raise ContractValidationError(
                "every repaired round requires an immutability receipt"
            )
        if len(rejected_receipts) > 1 or (
            rejected_receipts
            and self.repair_receipts[-1] is not rejected_receipts[0]
        ):
            raise ContractValidationError(
                "only the final attempted repair may be rejected"
            )
        if rejected_receipts and self.status is not RefinementVerificationStatus.ERROR:
            raise ContractValidationError("rejected repair must fail closed")
        if self.status is RefinementVerificationStatus.VERIFIED:
            if not self.rounds[-1].proved:
                raise ContractValidationError(
                    "verified status requires all final obligations proved"
                )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )

    @property
    def verified(self) -> bool:
        return self.status is RefinementVerificationStatus.VERIFIED

    @property
    def all_attempts(self) -> tuple[RefinementVerificationAttempt, ...]:
        return tuple(item for round_ in self.rounds for item in round_.attempts)

    @property
    def all_counterexamples(self) -> tuple[RefinementCounterexample, ...]:
        return tuple(item for round_ in self.rounds for item in round_.counterexamples)

    def _payload(self) -> dict[str, Any]:
        return {
            "verification_version": GOAL_REFINEMENT_VERIFICATION_VERSION,
            "frozen_context": self.frozen_context,
            "policy": self.policy,
            "status": self.status,
            "rounds": self.rounds,
            "repair_receipts": self.repair_receipts,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefinementVerificationResult":
        _schema(payload, cls.SCHEMA)
        frozen = payload.get("frozen_context")
        policy = payload.get("policy")
        if not isinstance(frozen, Mapping) or not isinstance(policy, Mapping):
            raise ContractValidationError(
                "verification result requires frozen_context and policy objects"
            )
        result = cls(
            frozen_context=FrozenRefinementContext.from_dict(frozen),
            policy=RefinementVerificationPolicy.from_dict(policy),
            status=payload.get("status", ""),
            rounds=tuple(
                item
                if isinstance(item, RefinementVerificationRound)
                else RefinementVerificationRound.from_dict(item)
                for item in (payload.get("rounds") or ())
            ),
            repair_receipts=tuple(
                item
                if isinstance(item, RepairImmutabilityReceipt)
                else RepairImmutabilityReceipt.from_dict(item)
                for item in (payload.get("repair_receipts") or ())
            ),
            reason=payload.get("reason", ""),
        )
        _identity(payload, result.content_id, "refinement verification result")
        return result


RepairRunner = Callable[[RefinementRepairRequest], RefinementRepairCandidate]


class GoalRefinementVerifier:
    """Run independent portfolios and fail closed on any missing authority."""

    def __init__(
        self,
        *,
        router: MultiProverRouter | None = None,
        policy: RefinementVerificationPolicy | None = None,
        audit_store: RefinementAuditSink | None = None,
        generator: GoalRefinementObligationGenerator | None = None,
    ) -> None:
        self.router = router or MultiProverRouter()
        self.policy = policy or RefinementVerificationPolicy()
        self.audit_store = audit_store or InMemoryRefinementAuditStore()
        self.generator = generator or GoalRefinementObligationGenerator()

    def _persist(self, record: CanonicalContract) -> None:
        try:
            self.audit_store.append(record)
        except RefinementPersistenceError:
            raise
        except BaseException as exc:
            raise RefinementPersistenceError(
                f"refinement audit sink failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _run_round(
        self,
        plan: FormalWorkPlan,
        runner: PortfolioRunner,
        frozen: FrozenRefinementContext,
        round_index: int,
    ) -> RefinementVerificationRound:
        obligations = self.generator.derive(
            plan,
            root_goal_id=frozen.root_goal_id,
            root_goal_content_id=frozen.root_goal_content_id,
            assumption_ids=frozen.assumption_ids,
        )
        results: list[PortfolioResult] = []
        attempts: list[RefinementVerificationAttempt] = []
        counterexamples: list[RefinementCounterexample] = []
        for obligation in obligations:
            result = self.router.execute(
                obligation.to_property_obligation(), runner
            )
            results.append(result)
            for attempt in result.attempts:
                retained = RefinementVerificationAttempt(
                    round_index=round_index,
                    obligation_id=obligation.obligation_id,
                    obligation_content_id=obligation.content_id,
                    portfolio_result_id=result.result_id,
                    attempt=attempt,
                )
                self._persist(retained)
                attempts.append(retained)
                if (
                    attempt.effective_outcome is AttemptOutcome.COUNTEREXAMPLE
                    and attempt.conclusive
                ):
                    counterexample = RefinementCounterexample(
                        round_index=round_index,
                        obligation_id=obligation.obligation_id,
                        portfolio_attempt_id=attempt.attempt_id,
                        prover_id=attempt.prover_id,
                        bounds=obligation.statement_model.get(
                            "bounds", {"trace_bound": plan.trace_bound}
                        ),
                        witness=attempt.evidence,
                        maximum_bytes=self.policy.max_counterexample_bytes,
                    )
                    self._persist(counterexample)
                    counterexamples.append(counterexample)
            if (
                self.policy.stop_on_counterexample
                and result.verdict is PortfolioVerdict.DISPROVED
            ):
                # The remaining obligations still need terminal attempt records.
                # Execute them: portfolio runners are bounded and persistence is
                # part of the contract.  "stop" applies between repair rounds,
                # not by silently dropping generated obligations.
                continue
        return RefinementVerificationRound(
            round_index=round_index,
            plan_id=plan.content_id,
            obligations=obligations,
            portfolio_results=tuple(results),
            attempts=tuple(attempts),
            counterexamples=tuple(counterexamples),
        )

    @staticmethod
    def _repair_receipt(
        request: RefinementRepairRequest,
        candidate: RefinementRepairCandidate,
    ) -> RepairImmutabilityReceipt:
        frozen = request.frozen_context
        claimed = candidate.frozen_context
        try:
            candidate_root = _root(candidate.plan, frozen.root_goal_id)
            plan_root_unchanged = (
                candidate_root.content_id == frozen.root_goal_content_id
            )
        except ContractValidationError:
            plan_root_unchanged = False
        root_unchanged = (
            claimed.root_goal_id == frozen.root_goal_id
            and claimed.root_goal_content_id == frozen.root_goal_content_id
            and plan_root_unchanged
        )
        assumptions_unchanged = claimed.assumption_ids == frozen.assumption_ids
        accepted = root_unchanged and assumptions_unchanged
        reasons = []
        if not root_unchanged:
            reasons.append("repair changed or replaced the frozen root")
        if not assumptions_unchanged:
            reasons.append("repair changed the frozen assumptions")
        return RepairImmutabilityReceipt(
            round_index=request.round_index,
            producer_id=candidate.producer_id,
            previous_plan_id=request.prior_plan_id,
            candidate_plan_id=candidate.plan.content_id,
            frozen_context_id=frozen.content_id,
            candidate_context_id=claimed.content_id,
            root_unchanged=root_unchanged,
            assumptions_unchanged=assumptions_unchanged,
            accepted=accepted,
            reason="; ".join(reasons),
        )

    def verify(
        self,
        plan: FormalWorkPlan,
        runner: PortfolioRunner,
        *,
        root_goal_id: str = "",
        assumption_ids: Iterable[str] = (),
        repairer: RepairRunner | None = None,
    ) -> RefinementVerificationResult:
        if not isinstance(plan, FormalWorkPlan):
            raise ContractValidationError("plan must be a FormalWorkPlan")
        if not callable(runner):
            raise ContractValidationError("runner must be callable")
        root = _root(plan, root_goal_id)
        frozen = FrozenRefinementContext(
            root_goal_id=root.goal_id,
            root_goal_content_id=root.content_id,
            assumption_ids=_strings(assumption_ids, "assumption_ids"),
        )
        if repairer is not None and not self.policy.allow_leanstral_repairs:
            raise ContractValidationError(
                "repairer supplied but Leanstral repairs are disabled by policy"
            )
        if self.policy.max_repair_rounds and repairer is None:
            raise ContractValidationError(
                "repair policy requires a Leanstral repair callback"
            )

        rounds: list[RefinementVerificationRound] = []
        receipts: list[RepairImmutabilityReceipt] = []
        current = plan
        for round_index in range(self.policy.max_repair_rounds + 1):
            round_result = self._run_round(current, runner, frozen, round_index)
            rounds.append(round_result)
            if round_result.proved:
                return RefinementVerificationResult(
                    frozen_context=frozen,
                    policy=self.policy,
                    status=RefinementVerificationStatus.VERIFIED,
                    rounds=tuple(rounds),
                    repair_receipts=tuple(receipts),
                    reason="all refinement obligations independently verified",
                )
            if round_index >= self.policy.max_repair_rounds or repairer is None:
                break
            request = RefinementRepairRequest(
                round_index=round_index + 1,
                frozen_context=frozen,
                prior_plan_id=current.content_id,
                failed_obligation_ids=tuple(
                    obligation.obligation_id
                    for obligation, result in zip(
                        round_result.obligations,
                        round_result.portfolio_results,
                    )
                    if result.verdict is not PortfolioVerdict.PROVED
                ),
                counterexamples=round_result.counterexamples,
            )
            try:
                candidate = repairer(request)
                if not isinstance(candidate, RefinementRepairCandidate):
                    raise ContractValidationError(
                        "repairer must return RefinementRepairCandidate"
                    )
            except BaseException as exc:
                return RefinementVerificationResult(
                    frozen_context=frozen,
                    policy=self.policy,
                    status=RefinementVerificationStatus.ERROR,
                    rounds=tuple(rounds),
                    repair_receipts=tuple(receipts),
                    reason=(
                        "Leanstral repair failed closed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            receipt = self._repair_receipt(request, candidate)
            self._persist(receipt)
            if not receipt.accepted:
                # A rejected repair is intentionally journaled but cannot be
                # represented as a completed repair round in the result.
                return RefinementVerificationResult(
                    frozen_context=frozen,
                    policy=self.policy,
                    status=RefinementVerificationStatus.ERROR,
                    rounds=tuple(rounds),
                    repair_receipts=tuple((*receipts, receipt)),
                    reason=receipt.reason,
                )
            receipts.append(receipt)
            current = candidate.plan

        final = rounds[-1]
        status = (
            RefinementVerificationStatus.DISPROVED
            if final.disproved
            else RefinementVerificationStatus.INCONCLUSIVE
        )
        return RefinementVerificationResult(
            frozen_context=frozen,
            policy=self.policy,
            status=status,
            rounds=tuple(rounds),
            repair_receipts=tuple(receipts),
            reason=(
                "one or more refinement obligations have bounded counterexamples"
                if status is RefinementVerificationStatus.DISPROVED
                else "independent authorities did not prove every refinement obligation"
            ),
        )


def verify_refinement_obligations(
    plan: FormalWorkPlan,
    runner: PortfolioRunner,
    *,
    root_goal_id: str = "",
    assumption_ids: Iterable[str] = (),
    router: MultiProverRouter | None = None,
    policy: RefinementVerificationPolicy | None = None,
    audit_store: RefinementAuditSink | None = None,
    repairer: RepairRunner | None = None,
) -> RefinementVerificationResult:
    """Convenience API for complete generation, routing, and verification."""

    return GoalRefinementVerifier(
        router=router,
        policy=policy,
        audit_store=audit_store,
    ).verify(
        plan,
        runner,
        root_goal_id=root_goal_id,
        assumption_ids=assumption_ids,
        repairer=repairer,
    )


# Compatibility-friendly descriptive aliases.
RefinementObligationGenerator = GoalRefinementObligationGenerator
RefinementVerifier = GoalRefinementVerifier
BoundedRefinementCounterexample = RefinementCounterexample


__all__ = [
    "DEFAULT_MAX_COUNTEREXAMPLE_BYTES",
    "FROZEN_REFINEMENT_CONTEXT_SCHEMA",
    "GOAL_REFINEMENT_VERIFICATION_VERSION",
    "MAX_LEANSTRAL_REPAIR_ROUNDS",
    "REFINEMENT_ATTEMPT_SCHEMA",
    "REFINEMENT_COUNTEREXAMPLE_SCHEMA",
    "REFINEMENT_OBLIGATION_SCHEMA",
    "REFINEMENT_POLICY_SCHEMA",
    "REFINEMENT_REPAIR_RECEIPT_SCHEMA",
    "REFINEMENT_RESULT_SCHEMA",
    "REFINEMENT_ROUND_SCHEMA",
    "BoundedRefinementCounterexample",
    "FrozenRefinementContext",
    "GoalRefinementObligationGenerator",
    "GoalRefinementVerifier",
    "InMemoryRefinementAuditStore",
    "JsonlRefinementAuditStore",
    "RefinementAuditSink",
    "RefinementCounterexample",
    "RefinementObligation",
    "RefinementObligationGenerator",
    "RefinementObligationKind",
    "RefinementPersistenceError",
    "RefinementRepairCandidate",
    "RefinementRepairRequest",
    "RefinementVerificationAttempt",
    "RefinementVerificationLedger",
    "RefinementVerificationPolicy",
    "RefinementVerificationResult",
    "RefinementVerificationRound",
    "RefinementVerificationStatus",
    "RefinementVerifier",
    "RepairImmutabilityReceipt",
    "RepairRunner",
    "derive_refinement_obligations",
    "property_kind_for_refinement_obligation",
    "verify_refinement_obligations",
]
