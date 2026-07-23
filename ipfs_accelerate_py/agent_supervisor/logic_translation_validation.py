"""Contracts and fail-closed validation for logic translations.

Solver output is meaningful only for the formula that was actually submitted.
This module therefore treats translation as a first-class evidence boundary:
the source, target, semantic inventories, implementation, fixture set,
assumptions, abstractions, and finite bounds are all content addressed.

The validator is deliberately independent of optional logic packages.  A
translator may populate :class:`SemanticInventory` while walking its native
AST, or use :func:`inventory_from_reviewed_formula` for the canonical
supervisor AST.  Inferring semantics from pretty-printed solver text is not a
supported authoritative path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .formal_logic_vocabulary import Formula, FormulaOperator, TermKind, TermSort
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    _canonical_value,
    content_identity,
)


TRANSLATION_VALIDATION_VERSION = 1
TRANSLATION_CONTRACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/logic-translation-contract@1"
)
SEMANTIC_INVENTORY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/logic-semantic-inventory@1"
)
TRANSLATION_ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/logic-translation-artifact@1"
)
TRANSLATION_VALIDATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/logic-translation-validation@1"
)


class LogicForm(str, Enum):
    """Reviewed source and target representation families."""

    AST = "ast"
    DCEC = "dcec"
    TDFOL = "tdfol"
    FOL = "fol"
    TPTP = "tptp"
    SMT_LIB = "smt-lib"
    TLA_PLUS = "tla+"
    PROTOCOL = "protocol"
    HYPERPROPERTY = "hyperproperty"


class TranslationClass(str, Enum):
    """Semantic relationship claimed between source and target."""

    EXACT = "exact"
    EQUISATISFIABLE = "equisatisfiable"
    BOUNDED_ABSTRACTION = "bounded_abstraction"
    CONSERVATIVE_APPROXIMATION = "conservative_approximation"
    HEURISTIC = "heuristic"


# Compatibility name used in early design notes.
TranslationExactness = TranslationClass


class ApproximationDirection(str, Enum):
    """Direction of a conservative approximation."""

    NONE = "none"
    OVER = "over_approximation"
    UNDER = "under_approximation"


class SemanticDimension(str, Enum):
    ACTORS = "actors"
    TIMES = "times"
    QUANTIFIERS = "quantifiers"
    MODAL_OPERATORS = "modal_operators"
    BOUNDS = "bounds"
    PREMISES = "premises"
    PREDICATES = "predicates"
    POLARITIES = "polarities"
    VARIABLES = "variables"


class TranslationIssueCode(str, Enum):
    CONTRACT_MISMATCH = "contract_mismatch"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    TARGET_IDENTITY_MISMATCH = "target_identity_mismatch"
    FIXTURE_SET_MISMATCH = "fixture_set_mismatch"
    MISSING_FINITE_BOUNDS = "missing_finite_bounds"
    UNDECLARED_ABSTRACTION = "undeclared_abstraction"
    DROPPED_ACTOR = "dropped_actor"
    DROPPED_TIME = "dropped_time"
    DROPPED_QUANTIFIER = "dropped_quantifier"
    DROPPED_MODAL_OPERATOR = "dropped_modal_operator"
    DROPPED_BOUND = "dropped_bound"
    DROPPED_PREMISE = "dropped_premise"
    DROPPED_PREDICATE = "dropped_predicate"
    CHANGED_POLARITY = "changed_polarity"
    DROPPED_VARIABLE = "dropped_variable"
    SEMANTIC_CHANGE = "semantic_change"
    MALFORMED_ARTIFACT = "malformed_artifact"


_DROPPED_CODES = {
    SemanticDimension.ACTORS: TranslationIssueCode.DROPPED_ACTOR,
    SemanticDimension.TIMES: TranslationIssueCode.DROPPED_TIME,
    SemanticDimension.QUANTIFIERS: TranslationIssueCode.DROPPED_QUANTIFIER,
    SemanticDimension.MODAL_OPERATORS: TranslationIssueCode.DROPPED_MODAL_OPERATOR,
    SemanticDimension.BOUNDS: TranslationIssueCode.DROPPED_BOUND,
    SemanticDimension.PREMISES: TranslationIssueCode.DROPPED_PREMISE,
    SemanticDimension.PREDICATES: TranslationIssueCode.DROPPED_PREDICATE,
    SemanticDimension.POLARITIES: TranslationIssueCode.CHANGED_POLARITY,
    SemanticDimension.VARIABLES: TranslationIssueCode.DROPPED_VARIABLE,
}

_MAXIMUM_ASSURANCE = {
    TranslationClass.EXACT: AssuranceLevel.SOLVER_CHECKED,
    TranslationClass.EQUISATISFIABLE: AssuranceLevel.SOLVER_CHECKED,
    TranslationClass.BOUNDED_ABSTRACTION: AssuranceLevel.SOLVER_CHECKED,
    TranslationClass.CONSERVATIVE_APPROXIMATION: AssuranceLevel.CANDIDATE,
    TranslationClass.HEURISTIC: AssuranceLevel.UNVERIFIED,
}

_ALL_RESULT_CLASSES = ("satisfiable", "unsatisfiable", "proved", "disproved")
_DEFAULT_RESULTS = {
    TranslationClass.EXACT: _ALL_RESULT_CLASSES,
    TranslationClass.EQUISATISFIABLE: ("satisfiable", "unsatisfiable"),
    TranslationClass.BOUNDED_ABSTRACTION: (
        "bounded_satisfiable",
        "bounded_unsatisfiable",
        "bounded_proved",
        "bounded_disproved",
    ),
    TranslationClass.CONSERVATIVE_APPROXIMATION: ("candidate", "counterexample"),
    TranslationClass.HEURISTIC: ("proposal",),
}


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    try:
        return enum_type(str(getattr(value, "value", value)))
    except ValueError as exc:
        raise ContractValidationError(f"{name} is unsupported") from exc


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ContractValidationError(f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise ContractValidationError(f"{name} must not be empty")
    return value


def _strings(
    value: Iterable[Any] | None,
    name: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if value is None:
        result: tuple[str, ...] = ()
    elif isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"{name} must be a sequence of strings")
    else:
        normalized = []
        for item in value:
            text = _text(item, name)
            if text not in normalized:
                normalized.append(text)
        result = tuple(sorted(normalized))
    if required and not result:
        raise ContractValidationError(f"{name} must not be empty")
    return result


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ContractValidationError(f"{name} must be an object with string keys")
    canonical = _canonical_value(dict(value))
    if not isinstance(canonical, dict):  # pragma: no cover - guarded above
        raise ContractValidationError(f"{name} must be an object")
    return canonical


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ContractValidationError(
            f"unsupported schema {supplied!r}; expected {expected}"
        )


def _claimed_identity(
    payload: Mapping[str, Any], actual: str, noun: str
) -> None:
    claimed = payload.get("content_id") or payload.get("identity")
    if claimed and claimed != actual:
        raise ContractValidationError(f"{noun} content identity does not match payload")


@dataclass(frozen=True)
class SemanticInventory(CanonicalContract):
    """Canonical semantic tokens observed while walking a representation.

    Tokens are representation-independent identities (for example
    ``agent:worker-7`` or ``forall:task``), not snippets copied from target
    text.  Multiplicity is retained because dropping one of two equal-looking
    premises is semantically relevant.
    """

    SCHEMA = SEMANTIC_INVENTORY_SCHEMA

    actors: tuple[str, ...] = ()
    times: tuple[str, ...] = ()
    quantifiers: tuple[str, ...] = ()
    modal_operators: tuple[str, ...] = ()
    bounds: tuple[str, ...] = ()
    premises: tuple[str, ...] = ()
    predicates: tuple[str, ...] = ()
    polarities: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for dimension in SemanticDimension:
            object.__setattr__(
                self,
                dimension.value,
                _strings(getattr(self, dimension.value), dimension.value),
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def _payload(self) -> dict[str, Any]:
        return {
            "validation_version": TRANSLATION_VALIDATION_VERSION,
            **{
                dimension.value: getattr(self, dimension.value)
                for dimension in SemanticDimension
            },
            "metadata": self.metadata,
        }

    def values(self, dimension: SemanticDimension | str) -> tuple[str, ...]:
        selected = _enum(dimension, SemanticDimension, "dimension")
        return getattr(self, selected.value)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticInventory":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("semantic inventory must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            **{
                dimension.value: tuple(payload.get(dimension.value) or ())
                for dimension in SemanticDimension
            },
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "semantic inventory")
        return result


def inventory_from_reviewed_formula(
    formula: Formula,
    *,
    premise_ids: Iterable[str] = (),
) -> SemanticInventory:
    """Extract a semantic inventory from the reviewed canonical AST."""

    if not isinstance(formula, Formula):
        raise ContractValidationError("formula must be a reviewed Formula")
    actors: list[str] = []
    times: list[str] = []
    quantifiers: list[str] = []
    modals: list[str] = []
    bounds: list[str] = []
    predicates: list[str] = []
    polarities: list[str] = []
    variables: list[str] = []

    modal_ops = {
        FormulaOperator.BELIEF,
        FormulaOperator.KNOWLEDGE,
        FormulaOperator.INTENTION,
        FormulaOperator.OBLIGATION,
        FormulaOperator.PERMISSION,
        FormulaOperator.PROHIBITION,
        FormulaOperator.DELEGATION,
        FormulaOperator.EXECUTION_EVENT,
        FormulaOperator.LIVENESS,
        FormulaOperator.SAFETY,
        FormulaOperator.GOAL_SATISFACTION,
    }

    def walk(node: Formula, negative: bool = False, path: str = "root") -> None:
        if node.operator in modal_ops:
            modals.append(f"{path}:{node.operator.value}")
        if node.lower_bound is not None:
            bounds.append(f"{path}:lower={node.lower_bound}")
        if node.upper_bound is not None:
            bounds.append(f"{path}:upper={node.upper_bound}")
        if node.predicate is not None:
            predicates.append(f"{path}:{node.predicate.value}")
            polarities.append(f"{path}:{'negative' if negative else 'positive'}")
        for index, term in enumerate(node.terms):
            token = f"{path}:term[{index}]={term.value}"
            if term.sort is TermSort.ACTOR:
                actors.append(token)
            elif term.sort is TermSort.TIME:
                times.append(token)
            if term.kind is TermKind.VARIABLE:
                variables.append(f"{path}:{term.sort.value}:{term.value}")
                # The reviewed v1 AST has typed variables but no binder node.
                # Record their implicit finite-domain quantification so a
                # target cannot silently turn them into constants.
                quantifiers.append(f"{path}:finite:{term.sort.value}:{term.value}")
        child_negative = negative ^ (node.operator is FormulaOperator.NOT)
        for index, operand in enumerate(node.operands):
            walk(operand, child_negative, f"{path}.{index}")

    walk(formula)
    return SemanticInventory(
        actors=tuple(actors),
        times=tuple(times),
        quantifiers=tuple(quantifiers),
        modal_operators=tuple(modals),
        bounds=tuple(bounds),
        premises=tuple(str(value) for value in premise_ids),
        predicates=tuple(predicates),
        polarities=tuple(polarities),
        variables=tuple(variables),
        metadata={
            "source": "reviewed_formula_ast",
            "formula_id": formula.formula_id,
        },
    )


@dataclass(frozen=True)
class TranslationContract(CanonicalContract):
    """Reviewed claim and assurance ceiling for one translation route."""

    SCHEMA = TRANSLATION_CONTRACT_SCHEMA

    contract_id: str
    source_identity: str
    source_form: LogicForm
    target_form: LogicForm
    translator_id: str
    translator_version: str
    translator_identity: str
    semantic_profile_id: str
    semantic_profile_version: str
    translation_class: TranslationClass
    fixture_set_id: str
    permitted_assurance: AssuranceLevel | None = None
    permitted_results: tuple[str, ...] = ()
    approximation_direction: ApproximationDirection = ApproximationDirection.NONE
    assumptions: tuple[str, ...] = ()
    abstracted_dimensions: tuple[SemanticDimension, ...] = ()
    required_bounds: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "contract_id",
            "source_identity",
            "translator_id",
            "translator_version",
            "translator_identity",
            "semantic_profile_id",
            "semantic_profile_version",
            "fixture_set_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "source_form", _enum(self.source_form, LogicForm, "source_form")
        )
        object.__setattr__(
            self, "target_form", _enum(self.target_form, LogicForm, "target_form")
        )
        object.__setattr__(
            self,
            "translation_class",
            _enum(self.translation_class, TranslationClass, "translation_class"),
        )
        object.__setattr__(
            self,
            "approximation_direction",
            _enum(
                self.approximation_direction,
                ApproximationDirection,
                "approximation_direction",
            ),
        )
        dimensions = tuple(
            sorted(
                {
                    _enum(item, SemanticDimension, "abstracted_dimensions")
                    for item in self.abstracted_dimensions
                },
                key=lambda item: item.value,
            )
        )
        object.__setattr__(self, "abstracted_dimensions", dimensions)
        object.__setattr__(
            self, "assumptions", _strings(self.assumptions, "assumptions")
        )
        object.__setattr__(
            self,
            "required_bounds",
            _strings(self.required_bounds, "required_bounds"),
        )
        maximum = _MAXIMUM_ASSURANCE[self.translation_class]
        selected = (
            maximum
            if self.permitted_assurance is None
            else _enum(
                self.permitted_assurance, AssuranceLevel, "permitted_assurance"
            )
        )
        if selected.rank > maximum.rank:
            raise ContractValidationError(
                f"{self.translation_class.value} translations cannot permit "
                f"{selected.value} assurance"
            )
        object.__setattr__(self, "permitted_assurance", selected)
        results = self.permitted_results or _DEFAULT_RESULTS[self.translation_class]
        object.__setattr__(
            self,
            "permitted_results",
            _strings(results, "permitted_results", required=True),
        )
        if self.translation_class in (
            TranslationClass.EXACT,
            TranslationClass.EQUISATISFIABLE,
        ) and dimensions:
            raise ContractValidationError(
                "exact and equisatisfiable contracts cannot declare abstracted dimensions"
            )
        if (
            self.translation_class is TranslationClass.BOUNDED_ABSTRACTION
            and not self.required_bounds
        ):
            raise ContractValidationError(
                "bounded abstraction contracts require named finite bounds"
            )
        if (
            self.translation_class is TranslationClass.CONSERVATIVE_APPROXIMATION
            and self.approximation_direction is ApproximationDirection.NONE
        ):
            raise ContractValidationError(
                "conservative approximations require an approximation direction"
            )
        if (
            self.translation_class is not TranslationClass.CONSERVATIVE_APPROXIMATION
            and self.approximation_direction is not ApproximationDirection.NONE
        ):
            raise ContractValidationError(
                "approximation_direction is only valid for conservative approximations"
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def maximum_assurance(self) -> AssuranceLevel:
        return self.permitted_assurance  # type: ignore[return-value]

    def permits(
        self,
        assurance: AssuranceLevel | str,
        *,
        result_class: Optional[str] = None,
        bounded: bool = False,
    ) -> bool:
        requested = _enum(assurance, AssuranceLevel, "assurance")
        if requested.rank > self.maximum_assurance.rank:
            return False
        if result_class is not None and result_class not in self.permitted_results:
            return False
        if self.translation_class is TranslationClass.BOUNDED_ABSTRACTION and not bounded:
            return False
        return True

    def _payload(self) -> dict[str, Any]:
        return {
            "validation_version": TRANSLATION_VALIDATION_VERSION,
            "contract_id": self.contract_id,
            "source_identity": self.source_identity,
            "source_form": self.source_form,
            "target_form": self.target_form,
            "translator_id": self.translator_id,
            "translator_version": self.translator_version,
            "translator_identity": self.translator_identity,
            "semantic_profile_id": self.semantic_profile_id,
            "semantic_profile_version": self.semantic_profile_version,
            "translation_class": self.translation_class,
            "fixture_set_id": self.fixture_set_id,
            "permitted_assurance": self.maximum_assurance,
            "permitted_results": self.permitted_results,
            "approximation_direction": self.approximation_direction,
            "assumptions": self.assumptions,
            "abstracted_dimensions": tuple(
                item.value for item in self.abstracted_dimensions
            ),
            "required_bounds": self.required_bounds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TranslationContract":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("translation contract must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            contract_id=payload.get("contract_id", ""),
            source_identity=payload.get("source_identity", ""),
            source_form=payload.get("source_form", ""),
            target_form=payload.get("target_form", ""),
            translator_id=payload.get("translator_id", ""),
            translator_version=payload.get("translator_version", ""),
            translator_identity=payload.get("translator_identity", ""),
            semantic_profile_id=payload.get("semantic_profile_id", ""),
            semantic_profile_version=payload.get("semantic_profile_version", ""),
            translation_class=payload.get("translation_class", ""),
            fixture_set_id=payload.get("fixture_set_id", ""),
            permitted_assurance=payload.get("permitted_assurance"),
            permitted_results=tuple(payload.get("permitted_results") or ()),
            approximation_direction=payload.get(
                "approximation_direction", ApproximationDirection.NONE
            ),
            assumptions=tuple(payload.get("assumptions") or ()),
            abstracted_dimensions=tuple(
                payload.get("abstracted_dimensions") or ()
            ),
            required_bounds=tuple(payload.get("required_bounds") or ()),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "translation contract")
        return result


@dataclass(frozen=True)
class TranslationArtifact(CanonicalContract):
    """One concrete translation and its independently collected inventories."""

    SCHEMA = TRANSLATION_ARTIFACT_SCHEMA

    contract_identity: str
    source_identity: str
    target_text: str
    source_inventory: SemanticInventory
    target_inventory: SemanticInventory
    fixture_set_id: str
    finite_bounds: Mapping[str, int] = field(default_factory=dict)
    abstraction_log: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("contract_identity", "source_identity", "fixture_set_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "target_text", _text(self.target_text, "target_text"))
        for name in ("source_inventory", "target_inventory"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = SemanticInventory.from_dict(value)
            if not isinstance(value, SemanticInventory):
                raise ContractValidationError(f"{name} must be a SemanticInventory")
            object.__setattr__(self, name, value)
        bounds = _mapping(self.finite_bounds, "finite_bounds")
        for name, value in bounds.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractValidationError(
                    f"finite bound {name!r} must be a non-negative integer"
                )
        object.__setattr__(self, "finite_bounds", bounds)
        object.__setattr__(
            self,
            "abstraction_log",
            _strings(self.abstraction_log, "abstraction_log"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def target_identity(self) -> str:
        return _sha256_text(self.target_text)

    def _payload(self) -> dict[str, Any]:
        return {
            "validation_version": TRANSLATION_VALIDATION_VERSION,
            "contract_identity": self.contract_identity,
            "source_identity": self.source_identity,
            "target_text": self.target_text,
            "target_identity": self.target_identity,
            "source_inventory": self.source_inventory,
            "target_inventory": self.target_inventory,
            "fixture_set_id": self.fixture_set_id,
            "finite_bounds": self.finite_bounds,
            "abstraction_log": self.abstraction_log,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TranslationArtifact":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("translation artifact must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            contract_identity=payload.get("contract_identity", ""),
            source_identity=payload.get("source_identity", ""),
            target_text=payload.get("target_text", ""),
            source_inventory=payload.get("source_inventory") or {},
            target_inventory=payload.get("target_inventory") or {},
            fixture_set_id=payload.get("fixture_set_id", ""),
            finite_bounds=payload.get("finite_bounds") or {},
            abstraction_log=tuple(payload.get("abstraction_log") or ()),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("target_identity")
        if claimed and claimed != result.target_identity:
            raise ContractValidationError("target identity does not match target text")
        _claimed_identity(payload, result.content_id, "translation artifact")
        return result


@dataclass(frozen=True)
class TranslationIssue:
    code: TranslationIssueCode | str
    dimension: SemanticDimension | str | None
    missing: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _enum(self.code, TranslationIssueCode, "code"))
        if self.dimension is not None:
            object.__setattr__(
                self,
                "dimension",
                _enum(self.dimension, SemanticDimension, "dimension"),
            )
        object.__setattr__(self, "missing", _strings(self.missing, "missing"))
        object.__setattr__(self, "detail", _text(self.detail, "detail"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "dimension": self.dimension.value if self.dimension else None,
            "missing": list(self.missing),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TranslationIssue":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("translation issue must be an object")
        return cls(
            code=payload.get("code", ""),
            dimension=payload.get("dimension"),
            missing=tuple(payload.get("missing") or ()),
            detail=payload.get("detail", ""),
        )


@dataclass(frozen=True)
class TranslationValidationResult(CanonicalContract):
    """Fail-closed promotion decision for a translation artifact."""

    SCHEMA = TRANSLATION_VALIDATION_SCHEMA

    contract_identity: str
    artifact_identity: str
    conformant: bool
    quarantine_required: bool
    maximum_assurance: AssuranceLevel
    issues: tuple[TranslationIssue, ...] = ()
    bounded: bool = False

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (self.conformant, self.quarantine_required, self.bounded)
        ):
            raise ContractValidationError(
                "conformant, quarantine_required, and bounded must be booleans"
            )
        object.__setattr__(
            self,
            "maximum_assurance",
            _enum(self.maximum_assurance, AssuranceLevel, "maximum_assurance"),
        )
        if self.conformant != (not self.issues):
            raise ContractValidationError(
                "conformant must be true exactly when issues are empty"
            )
        if self.quarantine_required != bool(self.issues):
            raise ContractValidationError(
                "translation issues must require quarantine"
            )
        if self.issues and self.maximum_assurance is not AssuranceLevel.UNVERIFIED:
            raise ContractValidationError(
                "non-conformant translations cannot retain assurance"
            )

    @property
    def promotion_allowed(self) -> bool:
        return self.conformant and self.maximum_assurance.rank > 0

    def permits(
        self, assurance: AssuranceLevel | str, *, bounded: bool = False
    ) -> bool:
        requested = _enum(assurance, AssuranceLevel, "assurance")
        return (
            self.conformant
            and requested.rank <= self.maximum_assurance.rank
            and (not self.bounded or bounded)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "validation_version": TRANSLATION_VALIDATION_VERSION,
            "contract_identity": self.contract_identity,
            "artifact_identity": self.artifact_identity,
            "conformant": self.conformant,
            "quarantine_required": self.quarantine_required,
            "maximum_assurance": self.maximum_assurance,
            "issues": tuple(issue.to_dict() for issue in self.issues),
            "bounded": self.bounded,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "TranslationValidationResult":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("translation validation must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            contract_identity=payload.get("contract_identity", ""),
            artifact_identity=payload.get("artifact_identity", ""),
            conformant=payload.get("conformant", False),
            quarantine_required=payload.get("quarantine_required", False),
            maximum_assurance=payload.get(
                "maximum_assurance", AssuranceLevel.UNVERIFIED
            ),
            issues=tuple(
                item
                if isinstance(item, TranslationIssue)
                else TranslationIssue.from_dict(item)
                for item in (payload.get("issues") or ())
            ),
            bounded=payload.get("bounded", False),
        )
        _claimed_identity(payload, result.content_id, "translation validation")
        return result


def validate_translation(
    contract: TranslationContract,
    artifact: TranslationArtifact,
) -> TranslationValidationResult:
    """Validate identities, bounds, and semantic preservation.

    Any loss is visible even when the contract intentionally abstracts that
    dimension.  Declared loss is not an error only for abstraction classes,
    and those classes retain their lower assurance ceiling.  Undeclared loss,
    or *any* loss on an exact/equisatisfiable route, quarantines the path.
    """

    if not isinstance(contract, TranslationContract):
        raise ContractValidationError("contract must be a TranslationContract")
    if not isinstance(artifact, TranslationArtifact):
        raise ContractValidationError("artifact must be a TranslationArtifact")
    issues: list[TranslationIssue] = []

    def issue(
        code: TranslationIssueCode,
        detail: str,
        *,
        dimension: SemanticDimension | None = None,
        missing: Iterable[str] = (),
    ) -> None:
        issues.append(
            TranslationIssue(
                code=code,
                dimension=dimension,
                missing=tuple(missing),
                detail=detail,
            )
        )

    if artifact.contract_identity != contract.content_id:
        issue(
            TranslationIssueCode.CONTRACT_MISMATCH,
            "artifact is bound to a different translation contract",
        )
    if artifact.source_identity != contract.source_identity:
        issue(
            TranslationIssueCode.SOURCE_IDENTITY_MISMATCH,
            "artifact is bound to a different source formula or fixture",
        )
    if artifact.fixture_set_id != contract.fixture_set_id:
        issue(
            TranslationIssueCode.FIXTURE_SET_MISMATCH,
            "artifact fixture set does not match the reviewed contract",
        )
    missing_bounds = tuple(
        name for name in contract.required_bounds if name not in artifact.finite_bounds
    )
    if missing_bounds:
        issue(
            TranslationIssueCode.MISSING_FINITE_BOUNDS,
            "required finite translation bounds are absent",
            dimension=SemanticDimension.BOUNDS,
            missing=missing_bounds,
        )

    declared = set(contract.abstracted_dimensions)
    for dimension in SemanticDimension:
        source_values = set(artifact.source_inventory.values(dimension))
        target_values = set(artifact.target_inventory.values(dimension))
        missing = tuple(sorted(source_values - target_values))
        abstraction_permitted = (
            dimension in declared
            and contract.translation_class
            in (
                TranslationClass.BOUNDED_ABSTRACTION,
                TranslationClass.CONSERVATIVE_APPROXIMATION,
                TranslationClass.HEURISTIC,
            )
            and bool(artifact.abstraction_log)
        )
        if missing and not abstraction_permitted:
            issue(
                _DROPPED_CODES[dimension],
                f"translation dropped or changed {dimension.value}",
                dimension=dimension,
                missing=missing,
            )
        introduced = tuple(sorted(target_values - source_values))
        if introduced and not abstraction_permitted:
            issue(
                TranslationIssueCode.SEMANTIC_CHANGE,
                f"translation introduced or changed {dimension.value}",
                dimension=dimension,
                missing=introduced,
            )

    # An abstraction log is itself evidence-bearing.  Exact routes and logs
    # that are not backed by declared dimensions fail closed.
    if artifact.abstraction_log and not declared:
        issue(
            TranslationIssueCode.UNDECLARED_ABSTRACTION,
            "translation recorded abstractions not declared by its contract",
        )

    conformant = not issues
    bounded = contract.translation_class is TranslationClass.BOUNDED_ABSTRACTION
    return TranslationValidationResult(
        contract_identity=contract.content_id,
        artifact_identity=artifact.content_id,
        conformant=conformant,
        quarantine_required=not conformant,
        maximum_assurance=(
            contract.maximum_assurance
            if conformant
            else AssuranceLevel.UNVERIFIED
        ),
        issues=tuple(issues),
        bounded=bounded,
    )


def semantic_inventory_identity(inventory: SemanticInventory) -> str:
    """Return the stable identity of an inventory (compatibility helper)."""

    return content_identity(inventory.to_dict())


def canonical_translation_json(value: Any) -> str:
    """Canonical JSON helper for external translator implementations."""

    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


__all__ = [
    "ApproximationDirection",
    "LogicForm",
    "SEMANTIC_INVENTORY_SCHEMA",
    "SemanticDimension",
    "SemanticInventory",
    "TRANSLATION_ARTIFACT_SCHEMA",
    "TRANSLATION_CONTRACT_SCHEMA",
    "TRANSLATION_VALIDATION_SCHEMA",
    "TRANSLATION_VALIDATION_VERSION",
    "TranslationArtifact",
    "TranslationClass",
    "TranslationContract",
    "TranslationExactness",
    "TranslationIssue",
    "TranslationIssueCode",
    "TranslationValidationResult",
    "canonical_translation_json",
    "inventory_from_reviewed_formula",
    "semantic_inventory_identity",
    "validate_translation",
]
