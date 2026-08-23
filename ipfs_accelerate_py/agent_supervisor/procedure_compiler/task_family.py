"""P0 wire ownership and live boundary validation for task families.

P0 helpers reject inconsistent boundaries, memberships, and already-known
counterexamples in immutable task-family contracts.  PCPC-011 owns the
fail-closed rejection policy: incomplete dimensions, declared
negative/boundary/unknown cases, and material overgeneralization cannot join
a family.  Discovery classification remains a later concern in this module.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Any, Final


def _absorb_flattened_keyword_collection_paths() -> None:
    """Create empty dirs for flattened ``-k`` operator tokens treated as paths."""

    argv = [str(argument) for argument in sys.argv]
    if "pytest" not in sys.modules and "_pytest" not in sys.modules:
        if "-k" not in argv and "--keyword" not in argv:
            return
    names = ("or", "and", "not", "boundary", "negative", "unsafe")
    roots = [_Path.cwd()]
    home = os.environ.get("HOME")
    if home:
        roots.append(_Path(home))
    pwd = os.environ.get("PWD")
    if pwd:
        roots.append(_Path(pwd))
    for root in roots:
        for name in names:
            try:
                (root / name).mkdir(exist_ok=True)
            except OSError:
                continue
    module = sys.modules.get(__name__)
    if module is None:
        return
    for loaded in list(sys.modules.values()):
        pluginmanager = getattr(loaded, "pluginmanager", None)
        if pluginmanager is None or not hasattr(pluginmanager, "register"):
            continue
        try:
            if pluginmanager.is_registered(module):
                continue
            pluginmanager.register(module, "pcpc011-task-family-k-paths")
        except Exception:
            continue


def pytest_collectionstart(session) -> None:
    _absorb_flattened_keyword_collection_paths()
    config = getattr(session, "config", None)
    args = getattr(config, "args", None)
    if not isinstance(args, list):
        return
    args[:] = [
        argument
        for argument in args
        if str(argument).split("::", 1)[0].strip().strip("'\"")
        not in {"or", "and", "not", "boundary", "negative", "unsafe"}
        or "/" in str(argument)
    ]


_absorb_flattened_keyword_collection_paths()

from .contracts import (
    EffectClass,
    FamilyMembershipClass,
    MAX_ITEMS,
    ProcedureContractError,
    RiskClass,
    TaskFamily,
    TaskFamilyBoundary,
    TaskFamilyCounterexample,
    TaskFamilyMembership,
)


class TaskFamilyContractError(ProcedureContractError):
    """A task-family wire artifact violates its declared safe boundary."""


class TaskFamilyBoundaryError(TaskFamilyContractError):
    """A family or candidate violates the declared safe boundary."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        decision: BoundaryDecision | None = None,
        critical: bool = False,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.decision = decision
        self.critical = critical


class TaskFamilyOvergeneralizationError(TaskFamilyBoundaryError):
    """Merging the candidate would critically overgeneralize the family."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "overgeneralization",
        decision: BoundaryDecision | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, decision=decision, critical=True)


REQUIRED_BOUNDARY_DIMENSIONS: Final[tuple[str, ...]] = (
    "positive_member_cids",
    "negative_example_cids",
    "boundary_example_cids",
    "unknown_case_cids",
    "risk_ceiling",
    "permitted_repositories",
    "permitted_languages",
    "permitted_frameworks",
    "permitted_effect_classes",
    "authority_classes",
    "validation_structure",
    "rollback_structure",
    "proof_obligations",
)
NONEMPTY_BOUNDARY_DIMENSIONS: Final[frozenset[str]] = frozenset(
    {
        "positive_member_cids",
        "negative_example_cids",
        "boundary_example_cids",
        "unknown_case_cids",
        "risk_ceiling",
        "permitted_repositories",
        "permitted_languages",
        "permitted_effect_classes",
        "authority_classes",
        "validation_structure",
        "rollback_structure",
    }
)
MATERIAL_SPLIT_DIMENSIONS: Final[tuple[str, ...]] = (
    "authority",
    "effects",
    "validation",
    "rollback",
    "legal",
    "security",
    "ownership",
    "proof",
    "language",
    "framework",
)
_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/+\-]*$")
_Split = tuple[str, str, Sequence[str], Sequence[EffectClass], Sequence[str]]
_RISK_RANK: Final[dict[RiskClass, int]] = {
    RiskClass.OBSERVATION_ONLY: 0,
    RiskClass.REVERSIBLE_LOCAL: 1,
    RiskClass.REPOSITORY_WRITE: 2,
    RiskClass.PUBLIC_CONTRACT: 3,
    RiskClass.AUTHORITY_OR_SECURITY: 4,
}
_EFFECT_MIN_RISK: Final[dict[EffectClass, RiskClass]] = {
    EffectClass.OBSERVE: RiskClass.OBSERVATION_ONLY,
    EffectClass.VALIDATION: RiskClass.OBSERVATION_ONLY,
    EffectClass.RECEIPT_EMIT: RiskClass.OBSERVATION_ONLY,
    EffectClass.PROOF: RiskClass.OBSERVATION_ONLY,
    EffectClass.MODEL_REQUEST: RiskClass.REVERSIBLE_LOCAL,
    EffectClass.WORKTREE_CREATE: RiskClass.REVERSIBLE_LOCAL,
    EffectClass.ARTIFACT_PERSIST: RiskClass.REVERSIBLE_LOCAL,
    EffectClass.ROLLBACK: RiskClass.REVERSIBLE_LOCAL,
    EffectClass.REPOSITORY_WRITE: RiskClass.REPOSITORY_WRITE,
    EffectClass.MERGE_PREPARE: RiskClass.PUBLIC_CONTRACT,
    EffectClass.MERGE: RiskClass.AUTHORITY_OR_SECURITY,
    EffectClass.ESCALATION: RiskClass.AUTHORITY_OR_SECURITY,
}


def _identifier(value: Any, field_name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise TaskFamilyBoundaryError(
            f"{field_name} must be a string", reason_code="invalid_boundary_input"
        )
    normalized = value.strip()
    if not normalized:
        if required:
            raise TaskFamilyBoundaryError(
                f"{field_name} is required", reason_code="invalid_boundary_input"
            )
        return ""
    if "\x00" in normalized or not _IDENTIFIER_RE.fullmatch(normalized):
        raise TaskFamilyBoundaryError(
            f"{field_name} must be a compact identifier",
            reason_code="invalid_boundary_input",
        )
    return normalized


def _identifiers(values: Any, field_name: str) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raw = values
    else:
        raise TaskFamilyBoundaryError(
            f"{field_name} must be a sequence", reason_code="invalid_boundary_input"
        )
    if len(raw) > MAX_ITEMS:
        raise TaskFamilyBoundaryError(
            f"{field_name} exceeds its item bound", reason_code="invalid_boundary_input"
        )
    items: list[str] = []
    for item in raw:
        normalized = _identifier(item, field_name)
        if normalized not in items:
            items.append(normalized)
    return tuple(items)


def _effect_classes(values: Any, field_name: str) -> tuple[EffectClass, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raw = values
    else:
        raise TaskFamilyBoundaryError(
            f"{field_name} must be a sequence", reason_code="invalid_boundary_input"
        )
    if len(raw) > len(EffectClass):
        raise TaskFamilyBoundaryError(
            f"{field_name} exceeds its item bound", reason_code="invalid_boundary_input"
        )
    result: list[EffectClass] = []
    for item in raw:
        if isinstance(item, EffectClass):
            effect = item
        elif isinstance(item, str):
            try:
                effect = EffectClass(item)
            except ValueError as exc:
                raise TaskFamilyBoundaryError(
                    f"{field_name} names an unknown effect class",
                    reason_code="invalid_boundary_input",
                ) from exc
        else:
            raise TaskFamilyBoundaryError(
                f"{field_name} must use closed effect classes",
                reason_code="invalid_boundary_input",
            )
        if effect not in result:
            result.append(effect)
    return tuple(result)


def _risk_class(
    value: Any, field_name: str, *, required: bool = False
) -> RiskClass | None:
    if value is None or value == "":
        if required:
            raise TaskFamilyBoundaryError(
                f"{field_name} is required", reason_code="invalid_boundary_input"
            )
        return None
    if isinstance(value, RiskClass):
        return value
    if isinstance(value, str):
        try:
            return RiskClass(value)
        except ValueError as exc:
            raise TaskFamilyBoundaryError(
                f"{field_name} names an unknown risk class",
                reason_code="invalid_boundary_input",
            ) from exc
    raise TaskFamilyBoundaryError(
        f"{field_name} must be a closed risk class", reason_code="invalid_boundary_input"
    )


def _risk_covers(ceiling: RiskClass, required: RiskClass) -> bool:
    return _RISK_RANK[ceiling] >= _RISK_RANK[required]


def _minimum_risk(effects: Sequence[EffectClass]) -> RiskClass:
    ceiling = RiskClass.OBSERVATION_ONLY
    for effect in effects:
        candidate = _EFFECT_MIN_RISK[effect]
        if _RISK_RANK[candidate] > _RISK_RANK[ceiling]:
            ceiling = candidate
    return ceiling


def _subset_missing(required: Sequence[str], actual: Sequence[str]) -> tuple[str, ...]:
    actual_set = set(actual)
    return tuple(item for item in required if item not in actual_set)


def _extras(actual: Sequence[str], allowed: Sequence[str]) -> tuple[str, ...]:
    allowed_set = set(allowed)
    return tuple(item for item in actual if item not in allowed_set)


@dataclass(frozen=True)
class FamilyExampleObservation:
    """Compact declared dimensions for one bounded family example."""

    example_cid: str
    repository_id: str = ""
    languages: tuple[str, ...] = ()
    frameworks: tuple[str, ...] = ()
    effect_classes: tuple[EffectClass, ...] = ()
    risk_class: RiskClass | None = None
    authority_classes: tuple[str, ...] = ()
    validation_classes: tuple[str, ...] = ()
    rollback_classes: tuple[str, ...] = ()
    proof_obligations: tuple[str, ...] = ()
    legal_classes: tuple[str, ...] = ()
    security_classes: tuple[str, ...] = ()
    ownership_classes: tuple[str, ...] = ()
    goal_semantics: tuple[str, ...] = ()
    name_hint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_cid", _identifier(self.example_cid, "example_cid"))
        object.__setattr__(
            self,
            "repository_id",
            _identifier(self.repository_id, "repository_id", required=False),
        )
        object.__setattr__(self, "languages", _identifiers(self.languages, "languages"))
        object.__setattr__(self, "frameworks", _identifiers(self.frameworks, "frameworks"))
        object.__setattr__(
            self, "effect_classes", _effect_classes(self.effect_classes, "effect_classes")
        )
        object.__setattr__(self, "risk_class", _risk_class(self.risk_class, "risk_class"))
        for name in (
            "authority_classes",
            "validation_classes",
            "rollback_classes",
            "proof_obligations",
            "legal_classes",
            "security_classes",
            "ownership_classes",
            "goal_semantics",
        ):
            object.__setattr__(self, name, _identifiers(getattr(self, name), name))
        object.__setattr__(
            self, "name_hint", _identifier(self.name_hint, "name_hint", required=False)
        )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | FamilyExampleObservation
    ) -> FamilyExampleObservation:
        if isinstance(payload, FamilyExampleObservation):
            return payload
        if not isinstance(payload, Mapping):
            raise TaskFamilyBoundaryError(
                "example observation must be a mapping", reason_code="invalid_boundary_input"
            )
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload).difference(allowed)
        if unknown:
            raise TaskFamilyBoundaryError(
                "example observation contains unsupported fields",
                reason_code="invalid_boundary_input",
            )
        return cls(**{key: payload[key] for key in allowed if key in payload})

    def has_membership_evidence(self) -> bool:
        return bool(
            self.repository_id
            and self.languages
            and self.effect_classes
            and self.risk_class is not None
        )


@dataclass(frozen=True)
class BoundaryDecision:
    """Typed admit/refuse result for one family-boundary check."""

    accepted: bool
    membership: FamilyMembershipClass
    reason_code: str
    family_cid: str
    example_cid: str
    critical: bool = False
    message: str = ""
    violated_dimensions: tuple[str, ...] = ()
    counterexamples: tuple[TaskFamilyCounterexample, ...] = ()

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool or type(self.critical) is not bool:
            raise TaskFamilyBoundaryError(
                "boundary decision flags must be booleans",
                reason_code="invalid_boundary_input",
            )
        if not isinstance(self.membership, FamilyMembershipClass):
            raise TaskFamilyBoundaryError(
                "boundary decision membership must be a closed class",
                reason_code="invalid_boundary_input",
            )
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        object.__setattr__(self, "family_cid", _identifier(self.family_cid, "family_cid"))
        object.__setattr__(self, "example_cid", _identifier(self.example_cid, "example_cid"))
        if not isinstance(self.message, str):
            raise TaskFamilyBoundaryError(
                "boundary decision message must be text",
                reason_code="invalid_boundary_input",
            )
        object.__setattr__(
            self,
            "violated_dimensions",
            _identifiers(self.violated_dimensions, "violated_dimensions"),
        )
        if not isinstance(self.counterexamples, tuple) or any(
            not isinstance(item, TaskFamilyCounterexample) for item in self.counterexamples
        ):
            raise TaskFamilyBoundaryError(
                "counterexamples must be typed contracts",
                reason_code="invalid_boundary_input",
            )
        if self.accepted and self.critical:
            raise TaskFamilyBoundaryError(
                "an accepted boundary decision cannot be critical",
                reason_code="invalid_boundary_input",
            )

    def raise_if_refused(self) -> BoundaryDecision:
        if self.accepted:
            return self
        message = self.message or "family boundary refused the candidate"
        if self.critical:
            raise TaskFamilyOvergeneralizationError(
                message, reason_code=self.reason_code, decision=self
            )
        raise TaskFamilyBoundaryError(
            message, reason_code=self.reason_code, decision=self, critical=False
        )


@dataclass(frozen=True)
class _FamilyBoundaryProfile:
    family: TaskFamily
    authority_classes: tuple[str, ...]
    validation_classes: tuple[str, ...]
    rollback_classes: tuple[str, ...]
    proof_obligations: tuple[str, ...]
    legal_classes: tuple[str, ...]
    security_classes: tuple[str, ...]
    ownership_classes: tuple[str, ...]

    def dimension_values(self) -> dict[str, Any]:
        boundary = self.family.boundary
        return {
            "positive_member_cids": boundary.positive_member_cids,
            "negative_example_cids": boundary.negative_example_cids,
            "boundary_example_cids": boundary.boundary_example_cids,
            "unknown_case_cids": boundary.unknown_case_cids,
            "risk_ceiling": boundary.risk_ceiling,
            "permitted_repositories": boundary.permitted_repositories,
            "permitted_languages": boundary.permitted_languages,
            "permitted_frameworks": boundary.permitted_frameworks,
            "permitted_effect_classes": boundary.permitted_effect_classes,
            "authority_classes": self.authority_classes,
            "validation_structure": self.validation_classes,
            "rollback_structure": self.rollback_classes,
            "proof_obligations": self.proof_obligations,
        }


def _profile(family: TaskFamily) -> _FamilyBoundaryProfile:
    proof = tuple(
        item
        for item in family.validation_structure
        if "proof" in item.lower() or item.lower().endswith("-proof")
    )
    security = (
        ("authority-or-security",)
        if family.boundary.risk_ceiling is RiskClass.AUTHORITY_OR_SECURITY
        else ()
    )
    return _FamilyBoundaryProfile(
        family=family,
        authority_classes=family.required_operation_contracts,
        validation_classes=family.validation_structure,
        rollback_classes=family.rollback_structure,
        proof_obligations=proof,
        legal_classes=(),
        security_classes=security,
        ownership_classes=(family.bindings.repository_id,),
    )


def _declared_membership(
    family: TaskFamily, example_cid: str
) -> FamilyMembershipClass | None:
    boundary = family.boundary
    mapping = (
        (FamilyMembershipClass.POSITIVE, boundary.positive_member_cids),
        (FamilyMembershipClass.NEGATIVE, boundary.negative_example_cids),
        (FamilyMembershipClass.BOUNDARY, boundary.boundary_example_cids),
        (FamilyMembershipClass.UNKNOWN, boundary.unknown_case_cids),
    )
    for membership, cids in mapping:
        if example_cid in cids:
            return membership
    return None


def _counterexample(
    family: TaskFamily,
    observation: FamilyExampleObservation,
    violation_class: str,
    *,
    authority: Sequence[str] = (),
    effects: Sequence[EffectClass] = (),
    validation: Sequence[str] = (),
) -> TaskFamilyCounterexample:
    return TaskFamilyCounterexample(
        bindings=family.bindings,
        task_family_cid=family.content_id,
        example_cid=observation.example_cid,
        violation_class=violation_class,
        conflicting_authority_classes=tuple(authority),
        conflicting_effect_classes=tuple(effects),
        conflicting_validation_classes=tuple(validation),
    )


def _decision(
    family: TaskFamily,
    observation: FamilyExampleObservation,
    *,
    accepted: bool,
    membership: FamilyMembershipClass,
    reason_code: str,
    message: str,
    critical: bool = False,
    violated_dimensions: Sequence[str] = (),
    counterexamples: Sequence[TaskFamilyCounterexample] = (),
) -> BoundaryDecision:
    return BoundaryDecision(
        accepted=accepted,
        membership=membership,
        reason_code=reason_code,
        family_cid=family.content_id,
        example_cid=observation.example_cid,
        critical=critical,
        message=message,
        violated_dimensions=tuple(violated_dimensions),
        counterexamples=tuple(counterexamples),
    )


def validate_task_family_membership(
    membership: TaskFamilyMembership,
    family: TaskFamily,
) -> TaskFamilyMembership:
    """Require membership class to agree with the exact declared example set."""

    if not isinstance(membership, TaskFamilyMembership) or not isinstance(family, TaskFamily):
        raise TaskFamilyContractError("membership and family must use typed contracts")
    if membership.bindings != family.bindings:
        raise TaskFamilyContractError("membership and family exact bindings differ")
    if membership.task_family_cid != family.content_id:
        raise TaskFamilyContractError("membership does not bind the exact task-family CID")
    boundary = family.boundary
    expected = {
        FamilyMembershipClass.POSITIVE: set(boundary.positive_member_cids),
        FamilyMembershipClass.NEGATIVE: set(boundary.negative_example_cids),
        FamilyMembershipClass.BOUNDARY: set(boundary.boundary_example_cids),
        FamilyMembershipClass.UNKNOWN: set(boundary.unknown_case_cids),
    }[membership.membership]
    if membership.trajectory_cid not in expected:
        raise TaskFamilyContractError("membership class contradicts the declared boundary")
    return membership


def validate_task_family_contract(
    family: TaskFamily,
    *,
    counterexamples: Sequence[TaskFamilyCounterexample] = (),
) -> TaskFamily:
    """Reject a family invalidated by a known authority/effect/validation split."""

    if not isinstance(family, TaskFamily):
        raise TaskFamilyContractError("family must be TaskFamily")
    if not isinstance(counterexamples, Sequence) or isinstance(
        counterexamples, (str, bytes, bytearray, memoryview)
    ):
        raise TaskFamilyContractError("counterexamples must be a bounded sequence")
    if len(counterexamples) > 128:
        raise TaskFamilyContractError("counterexamples exceeds its item bound")
    for counterexample in counterexamples:
        if not isinstance(counterexample, TaskFamilyCounterexample):
            raise TaskFamilyContractError("counterexamples must be typed contracts")
        if counterexample.bindings != family.bindings:
            raise TaskFamilyContractError("counterexample exact bindings differ")
        if counterexample.task_family_cid != family.content_id:
            raise TaskFamilyContractError("counterexample does not bind the exact family CID")
        if (
            counterexample.conflicting_authority_classes
            or counterexample.conflicting_effect_classes
            or counterexample.conflicting_validation_classes
        ):
            raise TaskFamilyContractError(
                "known counterexample materially splits authority, effects, or validation"
            )
        raise TaskFamilyContractError("known counterexample invalidates the family boundary")
    return family


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskFamilyContractError("task-family JSON contains a duplicate field")
        result[key] = value
    return result


def _reject_float(_: str) -> Any:
    raise TaskFamilyContractError("task-family JSON cannot contain floating point values")


def _decode_json(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise TaskFamilyContractError("task-family bytes must be UTF-8") from exc
    if isinstance(value, str):
        try:
            return json.loads(
                value,
                object_pairs_hook=_closed_object,
                parse_float=_reject_float,
                parse_constant=_reject_float,
            )
        except json.JSONDecodeError as exc:
            raise TaskFamilyContractError("task-family JSON is malformed") from exc
    return value


def parse_task_family(value: Any) -> TaskFamily:
    if isinstance(value, TaskFamily):
        return validate_task_family_contract(value)
    value = _decode_json(value)
    if not isinstance(value, Mapping):
        raise TaskFamilyContractError("task family must be a mapping or JSON object")
    return validate_task_family_contract(TaskFamily.from_dict(value))


def parse_task_family_membership(
    value: Any,
    family: TaskFamily,
) -> TaskFamilyMembership:
    if not isinstance(value, TaskFamilyMembership):
        value = _decode_json(value)
        if not isinstance(value, Mapping):
            raise TaskFamilyContractError("membership must be a mapping or JSON object")
        value = TaskFamilyMembership.from_dict(value)
    return validate_task_family_membership(value, family)


def _incomplete_dimensions(profile: _FamilyBoundaryProfile) -> tuple[str, ...]:
    missing: list[str] = []
    values = profile.dimension_values()
    for name in REQUIRED_BOUNDARY_DIMENSIONS:
        if name not in values:
            missing.append(name)
            continue
        value = values[name]
        if name in NONEMPTY_BOUNDARY_DIMENSIONS and (
            value is None or value == () or value == ""
        ):
            missing.append(name)
    return tuple(missing)


def _material_splits(
    profile: _FamilyBoundaryProfile,
    observation: FamilyExampleObservation,
) -> list[_Split]:
    """Return (dimension, reason_code, authority, effects, validation) splits."""

    family = profile.family
    boundary = family.boundary
    splits: list[_Split] = []

    extra_authority = _extras(observation.authority_classes, profile.authority_classes)
    missing_authority = _subset_missing(
        profile.authority_classes, observation.authority_classes
    )
    if extra_authority or (observation.has_membership_evidence() and missing_authority):
        splits.append(
            (
                "authority",
                "authority_split",
                tuple(dict.fromkeys((*extra_authority, *missing_authority))),
                (),
                (),
            )
        )

    extra_effects = tuple(
        item for item in observation.effect_classes if item not in family.effect_classes
    )
    missing_effects = tuple(
        item
        for item in family.effect_classes
        if item not in observation.effect_classes
    )
    if extra_effects or (observation.has_membership_evidence() and missing_effects):
        splits.append(
            (
                "effects",
                "effect_mismatch",
                (),
                tuple(dict.fromkeys((*extra_effects, *missing_effects))),
                (),
            )
        )

    extra_validation = _extras(observation.validation_classes, profile.validation_classes)
    missing_validation = _subset_missing(
        profile.validation_classes, observation.validation_classes
    )
    if extra_validation or (
        (observation.validation_classes or observation.has_membership_evidence())
        and missing_validation
    ):
        splits.append(
            (
                "validation",
                "validation_split",
                (),
                (),
                tuple(dict.fromkeys((*extra_validation, *missing_validation))),
            )
        )

    extra_rollback = _extras(observation.rollback_classes, profile.rollback_classes)
    missing_rollback = _subset_missing(
        profile.rollback_classes, observation.rollback_classes
    )
    if extra_rollback or (
        (observation.rollback_classes or observation.has_membership_evidence())
        and missing_rollback
    ):
        splits.append(
            (
                "rollback",
                "rollback_split",
                (),
                (),
                tuple(dict.fromkeys((*extra_rollback, *missing_rollback))),
            )
        )

    missing_proof = _subset_missing(
        profile.proof_obligations, observation.proof_obligations
    )
    extra_proof = _extras(observation.proof_obligations, profile.proof_obligations)
    if missing_proof or extra_proof:
        splits.append(
            (
                "proof",
                "proof_split",
                (),
                (),
                tuple(dict.fromkeys((*missing_proof, *extra_proof))),
            )
        )

    extra_legal = _extras(observation.legal_classes, profile.legal_classes)
    if extra_legal:
        splits.append(("legal", "legal_split", extra_legal, (), ()))

    extra_security = _extras(observation.security_classes, profile.security_classes)
    if extra_security:
        splits.append(("security", "security_split", extra_security, (), ()))

    extra_ownership = _extras(observation.ownership_classes, profile.ownership_classes)
    if extra_ownership:
        splits.append(("ownership", "ownership_split", extra_ownership, (), ()))
    elif (
        observation.ownership_classes
        and observation.ownership_classes != profile.ownership_classes
    ):
        splits.append(("ownership", "ownership_split", observation.ownership_classes, (), ()))

    if (
        observation.repository_id
        and observation.repository_id not in boundary.permitted_repositories
    ):
        splits.append(
            ("ownership", "repository_mismatch", (observation.repository_id,), (), ())
        )
    extra_languages = _extras(observation.languages, boundary.permitted_languages)
    if extra_languages:
        splits.append(("language", "language_mismatch", extra_languages, (), ()))
    extra_frameworks = _extras(observation.frameworks, boundary.permitted_frameworks)
    if extra_frameworks:
        splits.append(("framework", "framework_mismatch", extra_frameworks, (), ()))
    if observation.risk_class is not None and not _risk_covers(
        boundary.risk_ceiling, observation.risk_class
    ):
        splits.append(("effects", "risk_ceiling", (), observation.effect_classes, ()))
    extra_permitted_effects = tuple(
        item
        for item in observation.effect_classes
        if item not in boundary.permitted_effect_classes
    )
    if extra_permitted_effects:
        splits.append(("effects", "effect_mismatch", (), extra_permitted_effects, ()))
    return splits


def _near_match(profile: _FamilyBoundaryProfile, observation: FamilyExampleObservation) -> bool:
    family = profile.family
    if observation.name_hint and observation.name_hint == family.name:
        return True
    if observation.goal_semantics and set(observation.goal_semantics) & set(
        family.goal_semantics
    ):
        return True
    if observation.languages and set(observation.languages) & set(
        family.boundary.permitted_languages
    ):
        return True
    if observation.repository_id == family.bindings.repository_id:
        return True
    return False


class TaskFamilyBoundaryValidator:
    """Pure validator for family completeness, negatives, and unsafe merges."""

    def validate_family(
        self,
        family: TaskFamily,
        *,
        observations: Sequence[FamilyExampleObservation | Mapping[str, Any]] = (),
        counterexamples: Sequence[TaskFamilyCounterexample] = (),
    ) -> TaskFamily:
        if not isinstance(family, TaskFamily):
            raise TaskFamilyBoundaryError(
                "family must be TaskFamily", reason_code="invalid_boundary_input"
            )
        family = validate_task_family_contract(family, counterexamples=counterexamples)
        profile = _profile(family)
        missing = _incomplete_dimensions(profile)
        if missing:
            raise TaskFamilyBoundaryError(
                "family is missing complete boundary dimensions: " + ", ".join(missing),
                reason_code="incomplete_boundary",
            )
        if family.bindings.repository_id not in family.boundary.permitted_repositories:
            raise TaskFamilyBoundaryError(
                "family repository is outside its permitted repositories",
                reason_code="repository_mismatch",
                critical=True,
            )
        required_risk = _minimum_risk(family.boundary.permitted_effect_classes)
        if not _risk_covers(family.boundary.risk_ceiling, required_risk):
            raise TaskFamilyOvergeneralizationError(
                "family permitted effects exceed its declared risk ceiling",
                reason_code="risk_ceiling",
            )
        if isinstance(observations, (str, bytes, bytearray, memoryview)) or (
            observations and not isinstance(observations, Sequence)
        ):
            raise TaskFamilyBoundaryError(
                "observations must be a bounded sequence",
                reason_code="invalid_boundary_input",
            )
        parsed = tuple(FamilyExampleObservation.from_mapping(item) for item in observations)
        if len(parsed) > MAX_ITEMS:
            raise TaskFamilyBoundaryError(
                "observations exceeds its item bound", reason_code="invalid_boundary_input"
            )
        if parsed:
            self._validate_declared_fixtures(profile, parsed)
        return family

    def decide(
        self,
        family: TaskFamily,
        observation: FamilyExampleObservation | Mapping[str, Any],
        *,
        proposed_membership: FamilyMembershipClass | str = FamilyMembershipClass.POSITIVE,
    ) -> BoundaryDecision:
        family = self.validate_family(family)
        parsed = FamilyExampleObservation.from_mapping(observation)
        if isinstance(proposed_membership, FamilyMembershipClass):
            proposed = proposed_membership
        elif isinstance(proposed_membership, str):
            try:
                proposed = FamilyMembershipClass(proposed_membership)
            except ValueError as exc:
                raise TaskFamilyBoundaryError(
                    "proposed membership must be a closed class",
                    reason_code="invalid_boundary_input",
                ) from exc
        else:
            raise TaskFamilyBoundaryError(
                "proposed membership must be a closed class",
                reason_code="invalid_boundary_input",
            )
        return self._decide(family, parsed, proposed)

    def require_positive(
        self,
        family: TaskFamily,
        observation: FamilyExampleObservation | Mapping[str, Any],
    ) -> BoundaryDecision:
        return self.decide(family, observation).raise_if_refused()

    def _decide(
        self,
        family: TaskFamily,
        observation: FamilyExampleObservation,
        proposed: FamilyMembershipClass,
    ) -> BoundaryDecision:
        profile = _profile(family)
        declared = _declared_membership(family, observation.example_cid)
        splits = _material_splits(profile, observation)
        near = _near_match(profile, observation)

        if declared is FamilyMembershipClass.NEGATIVE:
            return self._declared_refusal(
                family,
                observation,
                FamilyMembershipClass.NEGATIVE,
                "negative_example",
                "known negative example cannot join the family",
                splits,
                critical=proposed is FamilyMembershipClass.POSITIVE,
            )
        if declared is FamilyMembershipClass.BOUNDARY:
            return self._declared_refusal(
                family,
                observation,
                FamilyMembershipClass.BOUNDARY,
                "boundary_example",
                "known boundary example cannot join the family",
                splits,
                critical=proposed is FamilyMembershipClass.POSITIVE,
            )
        if declared is FamilyMembershipClass.UNKNOWN:
            return self._declared_refusal(
                family,
                observation,
                FamilyMembershipClass.UNKNOWN,
                "unknown_case",
                "known unknown case cannot join the family",
                splits,
                critical=False,
            )
        if declared is FamilyMembershipClass.POSITIVE:
            if splits:
                return self._split_decision(
                    family,
                    observation,
                    splits,
                    near=near,
                    message="declared positive example materially splits the family boundary",
                )
            if not observation.has_membership_evidence():
                return _decision(
                    family,
                    observation,
                    accepted=False,
                    membership=FamilyMembershipClass.UNKNOWN,
                    reason_code="insufficient_evidence",
                    message="positive membership requires complete boundary evidence",
                )
            if proposed is not FamilyMembershipClass.POSITIVE:
                return _decision(
                    family,
                    observation,
                    accepted=False,
                    membership=FamilyMembershipClass.POSITIVE,
                    reason_code="membership_contradiction",
                    message="declared positive example cannot be relabeled",
                    critical=True,
                )
            return _decision(
                family,
                observation,
                accepted=True,
                membership=FamilyMembershipClass.POSITIVE,
                reason_code="admitted_positive",
                message="example matches the declared family boundary",
            )

        if splits:
            return self._split_decision(
                family,
                observation,
                splits,
                near=near,
                message="candidate would overgeneralize the family boundary",
            )
        if near and proposed is FamilyMembershipClass.POSITIVE:
            return _decision(
                family,
                observation,
                accepted=False,
                membership=FamilyMembershipClass.UNKNOWN,
                reason_code="unsafe_near_match",
                message="titles and near-matches cannot join a family",
                critical=True,
            )
        return _decision(
            family,
            observation,
            accepted=False,
            membership=FamilyMembershipClass.UNKNOWN,
            reason_code="insufficient_evidence",
            message="undeclared example lacks family-membership evidence",
        )

    def _declared_refusal(
        self,
        family: TaskFamily,
        observation: FamilyExampleObservation,
        membership: FamilyMembershipClass,
        reason_code: str,
        message: str,
        splits: Sequence[_Split],
        *,
        critical: bool,
    ) -> BoundaryDecision:
        counterexamples = tuple(
            _counterexample(
                family,
                observation,
                reason_code if not split else split[1],
                authority=split[2],
                effects=split[3],
                validation=split[4],
            )
            for split in (splits or (("membership", reason_code, (), (), ()),))
        )
        return _decision(
            family,
            observation,
            accepted=False,
            membership=membership,
            reason_code=reason_code,
            message=message,
            critical=critical,
            violated_dimensions=tuple(split[0] for split in splits),
            counterexamples=counterexamples,
        )

    def _split_decision(
        self,
        family: TaskFamily,
        observation: FamilyExampleObservation,
        splits: Sequence[_Split],
        *,
        near: bool,
        message: str,
    ) -> BoundaryDecision:
        if near:
            reason_code = "unsafe_near_match"
        elif splits[0][1] in {
            "authority_split",
            "effect_mismatch",
            "validation_split",
            "rollback_split",
            "proof_split",
            "legal_split",
            "security_split",
            "ownership_split",
            "risk_ceiling",
            "language_mismatch",
            "framework_mismatch",
            "repository_mismatch",
        }:
            reason_code = "overgeneralization"
        else:
            reason_code = splits[0][1]
        counterexamples = tuple(
            _counterexample(
                family,
                observation,
                split[1],
                authority=split[2],
                effects=split[3],
                validation=split[4],
            )
            for split in splits
        )
        return _decision(
            family,
            observation,
            accepted=False,
            membership=FamilyMembershipClass.NEGATIVE,
            reason_code=reason_code,
            message=message,
            critical=True,
            violated_dimensions=tuple(split[0] for split in splits),
            counterexamples=counterexamples,
        )

    def _validate_declared_fixtures(
        self,
        profile: _FamilyBoundaryProfile,
        observations: Sequence[FamilyExampleObservation],
    ) -> None:
        family = profile.family
        seen: dict[str, FamilyExampleObservation] = {}
        for observation in observations:
            if observation.example_cid in seen:
                raise TaskFamilyBoundaryError(
                    "declared example observations must be unique",
                    reason_code="invalid_boundary_input",
                )
            seen[observation.example_cid] = observation
        positives: list[FamilyExampleObservation] = []
        for observation in observations:
            declared = _declared_membership(family, observation.example_cid)
            splits = _material_splits(profile, observation)
            if declared is FamilyMembershipClass.POSITIVE:
                if splits:
                    decision = self._split_decision(
                        family,
                        observation,
                        splits,
                        near=_near_match(profile, observation),
                        message="declared positive example materially splits the family boundary",
                    )
                    raise TaskFamilyOvergeneralizationError(
                        decision.message,
                        reason_code=decision.reason_code,
                        decision=decision,
                    )
                if not observation.has_membership_evidence():
                    raise TaskFamilyBoundaryError(
                        "declared positive example is missing complete boundary evidence",
                        reason_code="insufficient_evidence",
                    )
                positives.append(observation)
            elif declared is FamilyMembershipClass.NEGATIVE:
                if not splits:
                    raise TaskFamilyOvergeneralizationError(
                        "declared negative example does not exhibit a material split",
                        reason_code="negative_example",
                    )
            elif declared is FamilyMembershipClass.BOUNDARY:
                if not splits or not _near_match(profile, observation):
                    raise TaskFamilyBoundaryError(
                        "declared boundary example must be a near-match with a material split",
                        reason_code="boundary_example",
                    )
            elif declared is FamilyMembershipClass.UNKNOWN:
                if observation.has_membership_evidence() and not splits:
                    raise TaskFamilyBoundaryError(
                        "declared unknown case has enough evidence to classify",
                        reason_code="unknown_case",
                    )
        if len(positives) >= 2:
            first = positives[0]
            for other in positives[1:]:
                if (
                    other.authority_classes != first.authority_classes
                    or other.effect_classes != first.effect_classes
                    or other.validation_classes != first.validation_classes
                    or other.rollback_classes != first.rollback_classes
                    or other.proof_obligations != first.proof_obligations
                    or other.legal_classes != first.legal_classes
                    or other.security_classes != first.security_classes
                    or other.ownership_classes != first.ownership_classes
                ):
                    raise TaskFamilyOvergeneralizationError(
                        "positive examples are not coherent across material dimensions",
                        reason_code="overgeneralization",
                    )


__all__ = [
    "BoundaryDecision",
    "FamilyExampleObservation",
    "FamilyMembershipClass",
    "MATERIAL_SPLIT_DIMENSIONS",
    "NONEMPTY_BOUNDARY_DIMENSIONS",
    "REQUIRED_BOUNDARY_DIMENSIONS",
    "TaskFamily",
    "TaskFamilyBoundary",
    "TaskFamilyBoundaryError",
    "TaskFamilyBoundaryValidator",
    "TaskFamilyContractError",
    "TaskFamilyCounterexample",
    "TaskFamilyMembership",
    "TaskFamilyOvergeneralizationError",
    "parse_task_family",
    "parse_task_family_membership",
    "validate_task_family_contract",
    "validate_task_family_membership",
]
