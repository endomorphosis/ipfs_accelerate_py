"""Bounded CEGIS procedure synthesis.

``ProcedureCegis`` generates declarative ProcedureIR candidates in a fixed
priority order and independently checks them against an immutable
counterexample set.  It is a synthesis state machine only: it does not
execute code, certify procedures, or promote anything.

Priority order (plan §11):

1. existing verified procedure
2. built-in template
3. anti-unified pattern
4. enumerative/constraint synthesis over ProcedureIR
5. model-proposed declarative sketch
6. human candidate

Every synthesis plan binds candidate, step, branch, hole, loop, model-call,
token, validation, proof, and wall bounds.  Counterexamples are immutable
and evaluation pairs are deduplicated by candidate identity plus
counterexample-set identity.  Bound exhaustion is a typed incomplete result.
Surviving candidates remain unpromoted.

``AssuranceCounterexampleAdapter`` reuses ``AssuranceCampaignApi@1`` for
bounded adversarial campaigns.  Every assurance failure becomes a persistent
typed counterexample that narrows or rejects the candidate.
``EscapedMutantGate`` records the immutable later-promotion requirement that
critical escaped mutants stay at zero; CEGIS itself never promotes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .anti_unification import AntiUnificationPattern, PatternStatus, StepPresence
from .contracts import (
    FORBIDDEN_STEP_OPERATIONS,
    MAX_BRANCHES,
    MAX_HOLES,
    MAX_ITEMS,
    MAX_LOOPS,
    MAX_STEPS,
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    EffectClass,
    ProcedureCandidate,
    ProcedureContractError,
    ProcedureEffect,
    ProcedureLocal,
    ProcedureObservation,
    ProcedureSpec,
    ProviderClass,
    StepOperation,
    ValueType,
    _enum,
    _freeze,
    _identifier,
    _nested,
    _nonnegative_int,
    _positive_int,
    _strings,
    content_identity,
)
from .contracts import (
    ProcedureSynthesisCounterexample as ProcedureSynthesisCounterexampleArtifact,
)
from .contracts import (
    ProcedureSynthesisPlan as ProcedureSynthesisPlanArtifact,
)
from .invariant_mining import (
    ASSURANCE_CAMPAIGN_API_INTERFACE_PIN,
    AssuranceApiStatus,
)
from .procedure_ir import ProcedureIRValidationError, validate_procedure_spec


CEGIS_REVISION: Final[str] = "ProcedureCegis@1"
COUNTEREXAMPLE_SET_SCHEMA: Final[str] = "procedure-compiler/counterexample-set@1"
EMPTY_COUNTEREXAMPLE_SET_CID: Final[str] = content_identity(
    {"schema": COUNTEREXAMPLE_SET_SCHEMA, "member_ids": ()}
)
MAX_GENERATED_VARIANTS: Final[int] = MAX_ITEMS
MAX_WALL_TIME_MS: Final[int] = 86_400_000
MAX_MODEL_TOKENS: Final[int] = 10_000_000
MAX_MODEL_CALLS: Final[int] = 1_024
MAX_VALIDATION_WORK: Final[int] = MAX_ITEMS
MAX_PROOF_WORK: Final[int] = MAX_ITEMS
ASSURANCE_COUNTEREXAMPLE_ADAPTER_REVISION: Final[str] = "assurance-counterexample-adapter@1"
EXECUTE_MUTATION_CAMPAIGN_COMMAND: Final[str] = "execute_mutation_campaign"
CLASSIFY_MUTATION_OUTCOME_COMMAND: Final[str] = "classify_mutation_outcome"
ZERO_CRITICAL_ESCAPED_MUTANTS: Final[str] = "zero-critical-escaped-mutants"
ASSURANCE_CAMPAIGN_SCHEMA: Final[str] = "procedure-compiler/assurance-campaign@1"

_VALIDATION_INSERTIONS: Final[tuple[tuple[StepOperation, str, EffectClass, str], ...]] = (
    (
        StepOperation.RUN_STATIC_ANALYSIS,
        "static-analysis@1",
        EffectClass.VALIDATION,
        "effect.validation",
    ),
    (
        StepOperation.RUN_TYPE_CHECK,
        "type-check@1",
        EffectClass.VALIDATION,
        "effect.validation",
    ),
    (
        StepOperation.RUN_ADVERSARIAL_ASSURANCE,
        "adversarial-assurance@1",
        EffectClass.VALIDATION,
        "effect.validation",
    ),
    (StepOperation.CHECK_SCOPE, "scope-checker@1", EffectClass.VALIDATION, "effect.validation"),
    (StepOperation.RUN_PROOF, "proof-runner@1", EffectClass.PROOF, "effect.proof"),
)


class CegisError(ProcedureContractError):
    """A synthesis plan, source, or counterexample could not be admitted."""


class SynthesisSourceKind(str, Enum):
    EXISTING_VERIFIED = "existing-verified"
    BUILTIN_TEMPLATE = "builtin-template"
    ANTI_UNIFIED_PATTERN = "anti-unified-pattern"
    ENUMERATIVE = "enumerative"
    MODEL_SKETCH = "model-sketch"
    HUMAN = "human"


GENERATION_PRIORITY: Final[tuple[SynthesisSourceKind, ...]] = (
    SynthesisSourceKind.EXISTING_VERIFIED,
    SynthesisSourceKind.BUILTIN_TEMPLATE,
    SynthesisSourceKind.ANTI_UNIFIED_PATTERN,
    SynthesisSourceKind.ENUMERATIVE,
    SynthesisSourceKind.MODEL_SKETCH,
    SynthesisSourceKind.HUMAN,
)


class SynthesisStatus(str, Enum):
    CONVERGED = "converged"
    INCOMPLETE = "incomplete"


class SynthesisStopReason(str, Enum):
    CONVERGED = "converged"
    NO_ADMISSIBLE_CANDIDATE = "no-admissible-candidate"
    CANDIDATE_BUDGET_EXHAUSTED = "candidate-budget-exhausted"
    STEP_BOUND_EXHAUSTED = "step-bound-exhausted"
    BRANCH_BOUND_EXHAUSTED = "branch-bound-exhausted"
    HOLE_BOUND_EXHAUSTED = "hole-bound-exhausted"
    LOOP_BOUND_EXHAUSTED = "loop-bound-exhausted"
    MODEL_CALL_BUDGET_EXHAUSTED = "model-call-budget-exhausted"
    TOKEN_BUDGET_EXHAUSTED = "token-budget-exhausted"
    VALIDATION_BUDGET_EXHAUSTED = "validation-budget-exhausted"
    PROOF_BUDGET_EXHAUSTED = "proof-budget-exhausted"
    WALL_BUDGET_EXHAUSTED = "wall-budget-exhausted"


class CounterexampleKind(str, Enum):
    REPLAY = "replay"
    ADVERSARIAL = "adversarial"
    STRUCTURAL = "structural"
    SPECIFICATION = "specification"


class AssuranceAttackClass(str, Enum):
    """Closed adversarial-assurance vocabulary consumed by CEGIS."""

    CRITICAL_SEEDED_MUTANT = "critical-seeded-mutant"
    PROMPT_INJECTION = "prompt-injection"
    VALIDATION_WEAKENING = "validation-weakening"
    SCOPE_ESCAPE = "scope-escape"
    SELF_PROMOTION = "self-promotion"
    FORGED_EVIDENCE = "forged-evidence"
    UNSAFE_FAMILY_MERGE = "unsafe-family-merge"


REQUIRED_ASSURANCE_ATTACK_CLASSES: Final[tuple[AssuranceAttackClass, ...]] = (
    AssuranceAttackClass.CRITICAL_SEEDED_MUTANT,
    AssuranceAttackClass.PROMPT_INJECTION,
    AssuranceAttackClass.VALIDATION_WEAKENING,
    AssuranceAttackClass.SCOPE_ESCAPE,
    AssuranceAttackClass.SELF_PROMOTION,
    AssuranceAttackClass.FORGED_EVIDENCE,
    AssuranceAttackClass.UNSAFE_FAMILY_MERGE,
)

_WRITE_OPERATIONS: Final[frozenset[StepOperation]] = frozenset(
    {
        StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
        StepOperation.CREATE_ISOLATED_WORKTREE,
        StepOperation.PREPARE_MERGE,
        StepOperation.MERGE_IN_ISOLATED_TRAIN,
    }
)
_MERGE_OPERATIONS: Final[frozenset[StepOperation]] = frozenset(
    {
        StepOperation.PREPARE_MERGE,
        StepOperation.MERGE_IN_ISOLATED_TRAIN,
    }
)
_REMOTE_PROVIDERS: Final[frozenset[ProviderClass]] = frozenset(
    {
        ProviderClass.LOCAL_SMALL_MODEL,
        ProviderClass.REMOTE_STANDARD_MODEL,
        ProviderClass.REMOTE_STRONG_MODEL,
    }
)
_PROMPT_INJECTION_MARKERS: Final[tuple[str, ...]] = (
    "ignore-previous-instructions",
    "override-policy",
    "prompt-injection",
)
_SELF_PROMOTION_MARKERS: Final[tuple[str, ...]] = (
    "self-promote",
    "self-certificate",
    "self-authorization",
)
_FORGED_EVIDENCE_MARKERS: Final[tuple[str, ...]] = (
    "forged",
    "simulated-as-live",
    "mock-receipt",
)
_VALIDATION_WEAKENING_MARKERS: Final[tuple[str, ...]] = (
    "skip-tests",
    "omit-proof",
    "weaken-validation",
)


class SkipReason(str, Enum):
    REPEATED_PAIR = "repeated-pair"
    NARROWED = "narrowed"
    OUT_OF_BOUNDS = "out-of-bounds"
    MODEL_BUDGET = "model-budget"
    TOKEN_BUDGET = "token-budget"
    PROOF_BUDGET = "proof-budget"


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise CegisError(f"{field_name} must be a boolean")
    return value


def _bound_int(value: Any, field_name: str, *, maximum: int, required: bool = False) -> int:
    result = _nonnegative_int(value, field_name, maximum=maximum)
    if required and result == 0:
        raise CegisError(f"{field_name} must be positive")
    return result


def _unique_kinds(values: Any, field_name: str) -> tuple[SynthesisSourceKind, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray, memoryview)):
        raise CegisError(f"{field_name} must be a sequence")
    if not values:
        raise CegisError(f"{field_name} must not be empty")
    if len(values) > len(SynthesisSourceKind):
        raise CegisError(f"{field_name} exceeds its item bound")
    ordered: list[SynthesisSourceKind] = []
    seen: set[SynthesisSourceKind] = set()
    for item in values:
        kind = _enum(item, SynthesisSourceKind, field_name)
        if kind in seen:
            raise CegisError(f"{field_name} contains duplicate source kinds")
        seen.add(kind)
        ordered.append(kind)
    return tuple(ordered)


def _operations(values: Any, field_name: str) -> tuple[StepOperation, ...]:
    if values in (None, ()):
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray, memoryview)):
        raise CegisError(f"{field_name} must be a sequence")
    if len(values) > len(StepOperation):
        raise CegisError(f"{field_name} exceeds its item bound")
    ordered: list[StepOperation] = []
    for item in values:
        operation = _enum(item, StepOperation, field_name)
        if operation.value in FORBIDDEN_STEP_OPERATIONS:
            raise CegisError("forbidden operations cannot enter synthesis constraints")
        if operation not in ordered:
            ordered.append(operation)
    return tuple(ordered)


def _procedure(value: Any, field_name: str) -> ProcedureSpec:
    if not isinstance(value, ProcedureSpec):
        raise CegisError(f"{field_name} must be a declarative ProcedureSpec")
    if value.state in {ArtifactState.VERIFIED, ArtifactState.PROMOTED}:
        raise CegisError("a ProcedureSpec cannot assert verification or promotion")
    return value


def _spec_operations(procedure: ProcedureSpec) -> frozenset[StepOperation]:
    return frozenset(step.operation for step in procedure.steps)


def _candidate_id(procedure: ProcedureSpec) -> str:
    return procedure.content_id


def _counterexample_set_cid(member_ids: Sequence[str]) -> str:
    return content_identity(
        {
            "schema": COUNTEREXAMPLE_SET_SCHEMA,
            "member_ids": tuple(sorted(_identifier(item, "member_id") for item in member_ids)),
        }
    )


def _as_candidate_spec(procedure: ProcedureSpec, *, name: str | None = None) -> ProcedureSpec:
    changes: dict[str, Any] = {"state": ArtifactState.CANDIDATE}
    if name is not None and name != procedure.name:
        changes["name"] = _identifier(name, "name")
    if procedure.state is ArtifactState.CANDIDATE and name is None:
        return procedure
    return replace(procedure, **changes)


@dataclass(frozen=True)
class NarrowingConstraints:
    """Closed constraints accumulated from immutable counterexamples."""

    required_operations: tuple[StepOperation, ...] = ()
    forbidden_operations: tuple[StepOperation, ...] = ()
    required_postcondition_ids: tuple[str, ...] = ()
    required_validation_step_ids: tuple[str, ...] = ()
    required_invariant_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_operations",
            _operations(self.required_operations, "required_operations"),
        )
        object.__setattr__(
            self,
            "forbidden_operations",
            _operations(self.forbidden_operations, "forbidden_operations"),
        )
        overlap = set(self.required_operations).intersection(self.forbidden_operations)
        if overlap:
            raise CegisError("required and forbidden operations overlap")
        for name in (
            "required_postcondition_ids",
            "required_validation_step_ids",
            "required_invariant_ids",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True),
            )

    def union(self, other: NarrowingConstraints) -> NarrowingConstraints:
        def _merge(left: Sequence[Any], right: Sequence[Any]) -> tuple[Any, ...]:
            merged: list[Any] = []
            for item in (*left, *right):
                if item not in merged:
                    merged.append(item)
            return tuple(merged)

        return NarrowingConstraints(
            required_operations=_merge(self.required_operations, other.required_operations),
            forbidden_operations=_merge(self.forbidden_operations, other.forbidden_operations),
            required_postcondition_ids=_merge(
                self.required_postcondition_ids, other.required_postcondition_ids
            ),
            required_validation_step_ids=_merge(
                self.required_validation_step_ids, other.required_validation_step_ids
            ),
            required_invariant_ids=_merge(self.required_invariant_ids, other.required_invariant_ids),
        )

    def rejects(self, procedure: ProcedureSpec) -> bool:
        operations = _spec_operations(procedure)
        if any(item not in operations for item in self.required_operations):
            return True
        if any(item in operations for item in self.forbidden_operations):
            return True
        post_ids = {item.condition_id for item in procedure.postconditions}
        if any(item not in post_ids for item in self.required_postcondition_ids):
            return True
        validation_ids = set(procedure.validation.required_step_ids)
        if any(item not in validation_ids for item in self.required_validation_step_ids):
            return True
        invariant_ids = {item.condition_id for item in procedure.invariants}
        return any(item not in invariant_ids for item in self.required_invariant_ids)

    def to_facts(self) -> dict[str, Any]:
        return {
            "required_operations": tuple(item.value for item in self.required_operations),
            "forbidden_operations": tuple(item.value for item in self.forbidden_operations),
            "required_postcondition_ids": self.required_postcondition_ids,
            "required_validation_step_ids": self.required_validation_step_ids,
            "required_invariant_ids": self.required_invariant_ids,
        }

    @classmethod
    def from_facts(cls, payload: Mapping[str, Any] | None) -> NarrowingConstraints:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise CegisError("constraints must be a mapping")
        return cls(
            required_operations=payload.get("required_operations", ()),
            forbidden_operations=payload.get("forbidden_operations", ()),
            required_postcondition_ids=payload.get("required_postcondition_ids", ()),
            required_validation_step_ids=payload.get("required_validation_step_ids", ()),
            required_invariant_ids=payload.get("required_invariant_ids", ()),
        )


def narrowing_for_attack(attack_class: AssuranceAttackClass) -> NarrowingConstraints:
    """Closed narrowing produced when an assurance attack class hits."""

    kind = _enum(attack_class, AssuranceAttackClass, "attack_class")
    if kind is AssuranceAttackClass.CRITICAL_SEEDED_MUTANT:
        return NarrowingConstraints(
            required_operations=(StepOperation.RUN_ADVERSARIAL_ASSURANCE,)
        )
    if kind is AssuranceAttackClass.VALIDATION_WEAKENING:
        return NarrowingConstraints(required_operations=(StepOperation.RUN_SELECTED_TESTS,))
    if kind is AssuranceAttackClass.SCOPE_ESCAPE:
        return NarrowingConstraints(required_operations=(StepOperation.CHECK_SCOPE,))
    return NarrowingConstraints()


def _marker_tuple(value: Any, fallback: Sequence[str]) -> tuple[str, ...]:
    if value in (None, (), ""):
        return tuple(fallback)
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, memoryview)):
        return tuple(fallback)
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _procedure_payload_fields(procedure: ProcedureSpec) -> tuple[str, ...]:
    """Attack-bearing strings. Procedure names are labels, not payload."""

    parts = (
        procedure.task_family_id,
        *procedure.provenance_cids,
        *procedure.scope_paths,
        *procedure.declared_reads,
        *(item.producer_contract for item in procedure.observations),
        *(item.evidence_type for item in procedure.observations),
        *(item.operation_contract for item in procedure.steps),
        *(item.step_id for item in procedure.steps),
        *(item.hole_id for item in procedure.holes),
        *(item.input_schema_ref for item in procedure.holes),
        *(item.output_schema_ref for item in procedure.holes),
    )
    return tuple(part for part in parts if part)


def _contains_any_marker(fields: Sequence[str], markers: Sequence[str]) -> bool:
    if not fields or not markers:
        return False
    lowered_markers = tuple(marker.lower() for marker in markers if marker)
    if not lowered_markers:
        return False
    for field in fields:
        text = field.lower()
        if any(marker in text for marker in lowered_markers):
            return True
    return False


@dataclass(frozen=True)
class AssuranceSeed:
    """One bounded hermetic assurance attack against a procedure candidate."""

    seed_id: str
    attack_class: AssuranceAttackClass
    obligation: str = ""
    critical: bool = True
    expected_killed: bool = True
    witness: Mapping[str, Any] = field(default_factory=dict)
    constraints: NarrowingConstraints = field(default_factory=NarrowingConstraints)
    evidence_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed_id", _identifier(self.seed_id, "seed_id"))
        object.__setattr__(
            self, "attack_class", _enum(self.attack_class, AssuranceAttackClass, "attack_class")
        )
        obligation = self.obligation or self.attack_class.value
        object.__setattr__(self, "obligation", _identifier(obligation, "obligation"))
        object.__setattr__(self, "critical", _bool(self.critical, "critical"))
        object.__setattr__(
            self, "expected_killed", _bool(self.expected_killed, "expected_killed")
        )
        witness = _freeze(self.witness, "witness")
        if not isinstance(witness, Mapping):
            raise CegisError("witness must be a mapping")
        object.__setattr__(self, "witness", witness)
        constraints = self.constraints
        if not isinstance(constraints, NarrowingConstraints):
            constraints = NarrowingConstraints.from_facts(constraints)
        if constraints == NarrowingConstraints():
            constraints = narrowing_for_attack(self.attack_class)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True),
        )


def default_assurance_seeds() -> tuple[AssuranceSeed, ...]:
    """Closed required attack classes as persistent typed counterexample seeds."""

    witnesses: dict[AssuranceAttackClass, dict[str, Any]] = {
        AssuranceAttackClass.CRITICAL_SEEDED_MUTANT: {
            "mutant_id": "mutant.critical.seeded",
            "severity": "critical",
        },
        AssuranceAttackClass.PROMPT_INJECTION: {
            "injection_markers": _PROMPT_INJECTION_MARKERS,
        },
        AssuranceAttackClass.VALIDATION_WEAKENING: {
            "weakening_markers": _VALIDATION_WEAKENING_MARKERS,
        },
        AssuranceAttackClass.SCOPE_ESCAPE: {
            "require_scope_check": True,
            "escape_markers": ("scope-escape", "path-escape"),
        },
        AssuranceAttackClass.SELF_PROMOTION: {
            "promotion_markers": _SELF_PROMOTION_MARKERS,
        },
        AssuranceAttackClass.FORGED_EVIDENCE: {
            "forged_markers": _FORGED_EVIDENCE_MARKERS,
        },
        AssuranceAttackClass.UNSAFE_FAMILY_MERGE: {
            "foreign_family_id": "UNSAFE_CROSS_FAMILY",
        },
    }
    return tuple(
        AssuranceSeed(
            seed_id=f"seed.{item.value}",
            attack_class=item,
            obligation=item.value,
            critical=True,
            expected_killed=True,
            witness=witnesses[item],
            evidence_cids=(f"evidence.{item.value}",),
        )
        for item in REQUIRED_ASSURANCE_ATTACK_CLASSES
    )


def inspect_assurance_seed(procedure: ProcedureSpec, seed: AssuranceSeed) -> bool:
    """True when the seeded attack still succeeds against the candidate."""

    if not isinstance(procedure, ProcedureSpec):
        raise CegisError("procedure must be a declarative ProcedureSpec")
    if not isinstance(seed, AssuranceSeed):
        raise CegisError("seed must be AssuranceSeed")
    operations = _spec_operations(procedure)
    payload = _procedure_payload_fields(procedure)
    witness = seed.witness
    if seed.constraints != NarrowingConstraints() and seed.constraints.rejects(procedure):
        return True
    if seed.attack_class is AssuranceAttackClass.CRITICAL_SEEDED_MUTANT:
        return StepOperation.RUN_ADVERSARIAL_ASSURANCE not in operations
    if seed.attack_class is AssuranceAttackClass.PROMPT_INJECTION:
        markers = _marker_tuple(witness.get("injection_markers"), _PROMPT_INJECTION_MARKERS)
        if _contains_any_marker(payload, markers):
            return True
        if procedure.holes and StepOperation.CHECK_POLICY not in operations:
            return True
        if any(
            provider in _REMOTE_PROVIDERS
            for hole in procedure.holes
            for provider in hole.allowed_provider_classes
        ) and StepOperation.CHECK_POLICY not in operations:
            return True
        return (
            StepOperation.REQUEST_TYPED_MODEL_HOLE in operations
            and StepOperation.CHECK_POLICY not in operations
        )
    if seed.attack_class is AssuranceAttackClass.VALIDATION_WEAKENING:
        markers = _marker_tuple(
            witness.get("weakening_markers"), _VALIDATION_WEAKENING_MARKERS
        )
        if _contains_any_marker(payload, markers):
            return True
        if StepOperation.RUN_SELECTED_TESTS not in operations:
            return True
        return not procedure.validation.required_test_contracts and (
            StepOperation.RUN_FULL_TEST_FALLBACK not in operations
        )
    if seed.attack_class is AssuranceAttackClass.SCOPE_ESCAPE:
        if any(".." in path or path.startswith("/") for path in procedure.scope_paths):
            return True
        if any(".." in path or path.startswith("/") for path in procedure.declared_reads):
            return True
        permitted = witness.get("permitted_scope_paths")
        if isinstance(permitted, Sequence) and not isinstance(
            permitted, (str, bytes, bytearray, memoryview)
        ):
            allowed = tuple(str(item) for item in permitted if isinstance(item, str))
            if allowed and any(path not in allowed for path in procedure.scope_paths):
                return True
        require_check = witness.get("require_scope_check") is True
        writes = bool(operations.intersection(_WRITE_OPERATIONS))
        if (require_check or writes) and StepOperation.CHECK_SCOPE not in operations:
            return True
        return False
    if seed.attack_class is AssuranceAttackClass.SELF_PROMOTION:
        if procedure.state in {ArtifactState.VERIFIED, ArtifactState.PROMOTED}:
            return True
        markers = _marker_tuple(witness.get("promotion_markers"), _SELF_PROMOTION_MARKERS)
        if _contains_any_marker(payload, markers):
            return True
        if operations.intersection(_MERGE_OPERATIONS) and (
            StepOperation.CHECK_AUTHORITY not in operations
        ):
            return True
        return False
    if seed.attack_class is AssuranceAttackClass.FORGED_EVIDENCE:
        markers = _marker_tuple(witness.get("forged_markers"), _FORGED_EVIDENCE_MARKERS)
        if _contains_any_marker(payload, markers):
            return True
        return (
            StepOperation.EMIT_RECEIPT in operations
            and StepOperation.CHECK_POSTCONDITION not in operations
        )
    if seed.attack_class is AssuranceAttackClass.UNSAFE_FAMILY_MERGE:
        foreign = witness.get("foreign_family_id")
        if isinstance(foreign, str) and foreign:
            if foreign in procedure.provenance_cids or foreign in procedure.name:
                return True
        return False
    return False


@dataclass(frozen=True)
class SynthesisCounterexample:
    """Immutable counterexample.  Identity is independent of later search state."""

    kind: CounterexampleKind
    obligation: str
    candidate_id: str
    counterexample_set_cid: str
    witness: Mapping[str, Any] = field(default_factory=dict)
    constraints: NarrowingConstraints = field(default_factory=NarrowingConstraints)
    evidence_cids: tuple[str, ...] = ()
    counterexample_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, CounterexampleKind, "kind"))
        object.__setattr__(self, "obligation", _identifier(self.obligation, "obligation"))
        object.__setattr__(self, "candidate_id", _identifier(self.candidate_id, "candidate_id"))
        object.__setattr__(
            self,
            "counterexample_set_cid",
            _identifier(self.counterexample_set_cid, "counterexample_set_cid"),
        )
        witness = _freeze(self.witness, "witness")
        if not isinstance(witness, Mapping):
            raise CegisError("witness must be a mapping")
        object.__setattr__(self, "witness", witness)
        constraints = self.constraints
        if not isinstance(constraints, NarrowingConstraints):
            constraints = NarrowingConstraints.from_facts(constraints)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True),
        )
        digest = content_identity(
            {
                "schema": "procedure-compiler/synthesis-counterexample@1",
                "kind": self.kind.value,
                "obligation": self.obligation,
                "candidate_id": self.candidate_id,
                "counterexample_set_cid": self.counterexample_set_cid,
                "witness": dict(self.witness),
                "constraints": self.constraints.to_facts(),
                "evidence_cids": self.evidence_cids,
            }
        )
        supplied = self.counterexample_id
        if supplied:
            object.__setattr__(
                self, "counterexample_id", _identifier(supplied, "counterexample_id")
            )
            if self.counterexample_id != digest:
                raise CegisError("counterexample identity does not match canonical content")
        else:
            object.__setattr__(self, "counterexample_id", digest)

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.candidate_id, self.counterexample_set_cid)

    def to_facts(self) -> dict[str, Any]:
        return {
            "counterexample_id": self.counterexample_id,
            "kind": self.kind.value,
            "obligation": self.obligation,
            "candidate_id": self.candidate_id,
            "counterexample_set_cid": self.counterexample_set_cid,
            "witness": dict(self.witness),
            "constraints": self.constraints.to_facts(),
            "evidence_cids": self.evidence_cids,
        }

    def to_artifact(
        self,
        bindings: ArtifactBindings,
        *,
        emitted_at_ms: int = 0,
    ) -> ProcedureSynthesisCounterexampleArtifact:
        references = tuple(
            item
            for item in (self.candidate_id, self.counterexample_set_cid, *self.evidence_cids)
            if item
        )
        return ProcedureSynthesisCounterexampleArtifact(
            bindings=bindings,
            state=ArtifactState.REJECTED,
            subject_cid=self.candidate_id,
            reference_cids=references,
            labels=(self.kind.value, self.obligation, CEGIS_REVISION),
            facts=self.to_facts(),
            created_at_ms=emitted_at_ms,
        )

    @classmethod
    def from_artifact(
        cls, artifact: ProcedureSynthesisCounterexampleArtifact
    ) -> SynthesisCounterexample:
        if not isinstance(artifact, ProcedureSynthesisCounterexampleArtifact):
            raise CegisError("artifact must be ProcedureSynthesisCounterexample")
        facts = artifact.facts
        return cls(
            kind=facts["kind"],
            obligation=facts["obligation"],
            candidate_id=facts["candidate_id"],
            counterexample_set_cid=facts["counterexample_set_cid"],
            witness=facts.get("witness", {}),
            constraints=NarrowingConstraints.from_facts(facts.get("constraints")),
            evidence_cids=facts.get("evidence_cids", ()),
            counterexample_id=facts.get("counterexample_id", ""),
        )


@dataclass(frozen=True)
class CounterexampleSet:
    """Immutable, identity-addressed set of synthesis counterexamples."""

    members: tuple[SynthesisCounterexample, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.members, Sequence) or isinstance(
            self.members, (str, bytes, bytearray, memoryview)
        ):
            raise CegisError("members must be a sequence")
        if len(self.members) > MAX_ITEMS:
            raise CegisError("counterexample set exceeds its item bound")
        ordered: list[SynthesisCounterexample] = []
        seen: set[str] = set()
        for item in self.members:
            if not isinstance(item, SynthesisCounterexample):
                raise CegisError("members must be SynthesisCounterexample records")
            if item.counterexample_id in seen:
                continue
            seen.add(item.counterexample_id)
            ordered.append(item)
        object.__setattr__(self, "members", tuple(ordered))

    @property
    def content_id(self) -> str:
        if not self.members:
            return EMPTY_COUNTEREXAMPLE_SET_CID
        return _counterexample_set_cid(tuple(item.counterexample_id for item in self.members))

    @property
    def constraints(self) -> NarrowingConstraints:
        merged = NarrowingConstraints()
        for item in self.members:
            merged = merged.union(item.constraints)
        return merged

    def add(self, item: SynthesisCounterexample) -> CounterexampleSet:
        if not isinstance(item, SynthesisCounterexample):
            raise CegisError("counterexample must be SynthesisCounterexample")
        if item.counterexample_id in {member.counterexample_id for member in self.members}:
            return self
        if len(self.members) >= MAX_ITEMS:
            raise CegisError("counterexample set exceeds its item bound")
        return CounterexampleSet(self.members + (item,))

    def contains_pair(self, candidate_id: str, set_cid: str) -> bool:
        return any(item.pair_key == (candidate_id, set_cid) for item in self.members)


@dataclass(frozen=True)
class ProcedureSynthesisPlan:
    """Fixed search and IR bounds for one CEGIS run.

    This is the synthesizer's plan object.  The wire artifact of the same
    closed name is the generic bounded envelope emitted by ``to_artifact``.
    """

    bindings: ArtifactBindings
    task_family_id: str
    max_candidates: int = 32
    max_steps: int = MAX_STEPS
    max_branches: int = MAX_BRANCHES
    max_holes: int = MAX_HOLES
    max_loops: int = MAX_LOOPS
    max_model_calls: int = 0
    max_tokens: int = 0
    max_validation: int = 32
    max_proof: int = 8
    max_wall_time_ms: int = 60_000
    generation_order: tuple[SynthesisSourceKind, ...] = GENERATION_PRIORITY
    include_builtin_templates: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self, "task_family_id", _identifier(self.task_family_id, "task_family_id")
        )
        object.__setattr__(
            self,
            "max_candidates",
            _bound_int(self.max_candidates, "max_candidates", maximum=MAX_ITEMS, required=True),
        )
        object.__setattr__(
            self,
            "max_steps",
            _bound_int(self.max_steps, "max_steps", maximum=MAX_STEPS, required=True),
        )
        object.__setattr__(
            self,
            "max_branches",
            _bound_int(self.max_branches, "max_branches", maximum=MAX_BRANCHES),
        )
        object.__setattr__(
            self, "max_holes", _bound_int(self.max_holes, "max_holes", maximum=MAX_HOLES)
        )
        object.__setattr__(
            self, "max_loops", _bound_int(self.max_loops, "max_loops", maximum=MAX_LOOPS)
        )
        object.__setattr__(
            self,
            "max_model_calls",
            _bound_int(self.max_model_calls, "max_model_calls", maximum=MAX_MODEL_CALLS),
        )
        object.__setattr__(
            self,
            "max_tokens",
            _bound_int(self.max_tokens, "max_tokens", maximum=MAX_MODEL_TOKENS),
        )
        object.__setattr__(
            self,
            "max_validation",
            _bound_int(self.max_validation, "max_validation", maximum=MAX_VALIDATION_WORK),
        )
        object.__setattr__(
            self,
            "max_proof",
            _bound_int(self.max_proof, "max_proof", maximum=MAX_PROOF_WORK),
        )
        object.__setattr__(
            self,
            "max_wall_time_ms",
            _bound_int(
                self.max_wall_time_ms,
                "max_wall_time_ms",
                maximum=MAX_WALL_TIME_MS,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "generation_order",
            _unique_kinds(self.generation_order, "generation_order"),
        )
        object.__setattr__(
            self,
            "include_builtin_templates",
            _bool(self.include_builtin_templates, "include_builtin_templates"),
        )

    def to_facts(self) -> dict[str, Any]:
        return {
            "task_family_id": self.task_family_id,
            "max_candidates": self.max_candidates,
            "max_steps": self.max_steps,
            "max_branches": self.max_branches,
            "max_holes": self.max_holes,
            "max_loops": self.max_loops,
            "max_model_calls": self.max_model_calls,
            "max_tokens": self.max_tokens,
            "max_validation": self.max_validation,
            "max_proof": self.max_proof,
            "max_wall_time_ms": self.max_wall_time_ms,
            "generation_order": tuple(item.value for item in self.generation_order),
            "include_builtin_templates": self.include_builtin_templates,
            "synthesizer_revision": CEGIS_REVISION,
        }

    def to_artifact(self, *, emitted_at_ms: int = 0) -> ProcedureSynthesisPlanArtifact:
        return ProcedureSynthesisPlanArtifact(
            bindings=self.bindings,
            state=ArtifactState.CANDIDATE,
            subject_cid=self.task_family_id,
            reference_cids=(),
            labels=(CEGIS_REVISION, "unpromoted"),
            facts=self.to_facts(),
            created_at_ms=emitted_at_ms,
        )

    @classmethod
    def from_artifact(cls, artifact: ProcedureSynthesisPlanArtifact) -> ProcedureSynthesisPlan:
        if not isinstance(artifact, ProcedureSynthesisPlanArtifact):
            raise CegisError("artifact must be ProcedureSynthesisPlan")
        facts = artifact.facts
        return cls(
            bindings=artifact.bindings,
            task_family_id=facts["task_family_id"],
            max_candidates=facts["max_candidates"],
            max_steps=facts["max_steps"],
            max_branches=facts["max_branches"],
            max_holes=facts["max_holes"],
            max_loops=facts["max_loops"],
            max_model_calls=facts["max_model_calls"],
            max_tokens=facts["max_tokens"],
            max_validation=facts["max_validation"],
            max_proof=facts["max_proof"],
            max_wall_time_ms=facts["max_wall_time_ms"],
            generation_order=facts["generation_order"],
            include_builtin_templates=facts.get("include_builtin_templates", False),
        )


@dataclass(frozen=True)
class VerifiedProcedureSeed:
    """Existing independently verified procedure.  The seed itself is not re-promoted."""

    procedure: ProcedureSpec
    certificate_cid: str
    source_episode_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "procedure", _procedure(self.procedure, "procedure"))
        object.__setattr__(
            self, "certificate_cid", _identifier(self.certificate_cid, "certificate_cid")
        )
        object.__setattr__(
            self,
            "source_episode_cids",
            _strings(self.source_episode_cids, "source_episode_cids", identifiers=True),
        )


@dataclass(frozen=True)
class ModelSketch:
    """Model-proposed declarative sketch.  Executable payloads are refused."""

    procedure: ProcedureSpec
    token_cost: int
    model_calls: int = 1
    source_episode_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "procedure", _procedure(self.procedure, "procedure"))
        object.__setattr__(
            self,
            "token_cost",
            _nonnegative_int(self.token_cost, "token_cost", maximum=MAX_MODEL_TOKENS),
        )
        object.__setattr__(
            self,
            "model_calls",
            _positive_int(self.model_calls, "model_calls", maximum=MAX_MODEL_CALLS),
        )
        object.__setattr__(
            self,
            "source_episode_cids",
            _strings(self.source_episode_cids, "source_episode_cids", identifiers=True),
        )


@dataclass(frozen=True)
class ValidationFinding:
    """Independent replay or adversarial finding against one candidate/set pair."""

    kind: CounterexampleKind
    obligation: str
    witness: Mapping[str, Any] = field(default_factory=dict)
    constraints: NarrowingConstraints = field(default_factory=NarrowingConstraints)
    evidence_cids: tuple[str, ...] = ()
    validation_cost: int = 0
    proof_cost: int = 0
    attack_class: AssuranceAttackClass | None = None
    additional_findings: tuple[ValidationFinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, CounterexampleKind, "kind"))
        object.__setattr__(self, "obligation", _identifier(self.obligation, "obligation"))
        witness = _freeze(self.witness, "witness")
        if not isinstance(witness, Mapping):
            raise CegisError("witness must be a mapping")
        object.__setattr__(self, "witness", witness)
        constraints = self.constraints
        if not isinstance(constraints, NarrowingConstraints):
            constraints = NarrowingConstraints.from_facts(constraints)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True),
        )
        object.__setattr__(
            self,
            "validation_cost",
            _nonnegative_int(self.validation_cost, "validation_cost", maximum=MAX_VALIDATION_WORK),
        )
        object.__setattr__(
            self,
            "proof_cost",
            _nonnegative_int(self.proof_cost, "proof_cost", maximum=MAX_PROOF_WORK),
        )
        if self.attack_class is not None:
            object.__setattr__(
                self,
                "attack_class",
                _enum(self.attack_class, AssuranceAttackClass, "attack_class"),
            )
        extras = self.additional_findings
        if extras in (None, ()):
            object.__setattr__(self, "additional_findings", ())
            return
        if not isinstance(extras, Sequence) or isinstance(
            extras, (str, bytes, bytearray, memoryview)
        ):
            raise CegisError("additional_findings must be a sequence")
        if len(extras) > MAX_ITEMS:
            raise CegisError("additional_findings exceeds its item bound")
        for item in extras:
            if not isinstance(item, ValidationFinding):
                raise CegisError("additional_findings must contain ValidationFinding records")
        object.__setattr__(self, "additional_findings", tuple(extras))


CandidateValidator = Callable[["SynthesisCandidate", CounterexampleSet], ValidationFinding | None]


@dataclass(frozen=True)
class SynthesisCandidate:
    """One generated declarative candidate before or after independent checking."""

    source_kind: SynthesisSourceKind
    procedure: ProcedureSpec
    source_episode_cids: tuple[str, ...]
    model_calls: int = 0
    token_cost: int = 0
    certificate_cid: str = ""
    variant_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_kind", _enum(self.source_kind, SynthesisSourceKind, "source_kind")
        )
        object.__setattr__(self, "procedure", _procedure(self.procedure, "procedure"))
        episodes = self.source_episode_cids or self.procedure.provenance_cids
        object.__setattr__(
            self,
            "source_episode_cids",
            _strings(episodes, "source_episode_cids", identifiers=True, required=True),
        )
        object.__setattr__(
            self,
            "model_calls",
            _nonnegative_int(self.model_calls, "model_calls", maximum=MAX_MODEL_CALLS),
        )
        object.__setattr__(
            self,
            "token_cost",
            _nonnegative_int(self.token_cost, "token_cost", maximum=MAX_MODEL_TOKENS),
        )
        object.__setattr__(
            self,
            "certificate_cid",
            _identifier(self.certificate_cid, "certificate_cid", required=False),
        )
        object.__setattr__(
            self,
            "variant_label",
            _identifier(self.variant_label, "variant_label", required=False),
        )
        if self.source_kind is not SynthesisSourceKind.MODEL_SKETCH and (
            self.model_calls or self.token_cost
        ):
            raise CegisError("only model sketches may consume model-call or token budget")
        if self.source_kind is SynthesisSourceKind.EXISTING_VERIFIED and not self.certificate_cid:
            raise CegisError("existing verified procedures require a certificate identity")

    @property
    def candidate_id(self) -> str:
        return _candidate_id(self.procedure)

    def to_procedure_candidate(
        self,
        *,
        synthesis_plan_cid: str,
        counterexample_set_cid: str,
        state: ArtifactState,
    ) -> ProcedureCandidate:
        if state not in {ArtifactState.CANDIDATE, ArtifactState.REJECTED, ArtifactState.DEVELOPMENT}:
            raise CegisError("synthesized candidates cannot assert verified or promoted status")
        return ProcedureCandidate(
            bindings=self.procedure.bindings,
            procedure=_as_candidate_spec(self.procedure),
            synthesis_plan_cid=synthesis_plan_cid,
            source_episode_cids=self.source_episode_cids,
            counterexample_set_cid=counterexample_set_cid,
            state=state,
        )


def _flatten_findings(finding: ValidationFinding | None) -> tuple[ValidationFinding, ...]:
    if finding is None:
        return ()
    ordered: list[ValidationFinding] = []
    pending: list[ValidationFinding] = [finding]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        extras = item.additional_findings
        if extras:
            ordered.append(replace(item, additional_findings=()))
            pending = list(extras) + pending
        else:
            ordered.append(item)
    return tuple(ordered)


def _finding_from_seed(
    seed: AssuranceSeed,
    *,
    evidence_cids: Sequence[str] = (),
) -> ValidationFinding:
    witness = dict(seed.witness)
    witness["seed_id"] = seed.seed_id
    witness["attack_class"] = seed.attack_class.value
    witness["critical"] = seed.critical
    receipts = tuple(dict.fromkeys((*seed.evidence_cids, *evidence_cids)))
    return ValidationFinding(
        kind=CounterexampleKind.ADVERSARIAL,
        obligation=seed.obligation,
        witness=witness,
        constraints=seed.constraints,
        evidence_cids=receipts,
        attack_class=seed.attack_class,
    )


@dataclass(frozen=True)
class EscapedMutantGate:
    """Immutable later-promotion requirement: zero critical escaped mutants.

    CEGIS never promotes.  A later verifier must still prove this floor; the
    requirement cannot be waived, claimed complete, or treated as satisfied by
    synthesis survival alone.
    """

    candidate_id: str
    counterexample_set_cid: str
    critical_seed_ids: tuple[str, ...]
    escaped_critical_mutant_ids: tuple[str, ...]
    killed_critical_mutant_ids: tuple[str, ...] = ()
    campaign_confirmed: bool = False
    requirement_id: str = ZERO_CRITICAL_ESCAPED_MUTANTS
    waivable: bool = False
    promotion_permitted: bool = False
    completeness_claimed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _identifier(self.candidate_id, "candidate_id"))
        object.__setattr__(
            self,
            "counterexample_set_cid",
            _identifier(self.counterexample_set_cid, "counterexample_set_cid"),
        )
        object.__setattr__(
            self,
            "critical_seed_ids",
            _strings(self.critical_seed_ids, "critical_seed_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "escaped_critical_mutant_ids",
            _strings(
                self.escaped_critical_mutant_ids,
                "escaped_critical_mutant_ids",
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "killed_critical_mutant_ids",
            _strings(
                self.killed_critical_mutant_ids,
                "killed_critical_mutant_ids",
                identifiers=True,
            ),
        )
        object.__setattr__(
            self, "campaign_confirmed", _bool(self.campaign_confirmed, "campaign_confirmed")
        )
        object.__setattr__(
            self,
            "requirement_id",
            _identifier(self.requirement_id, "requirement_id"),
        )
        if self.requirement_id != ZERO_CRITICAL_ESCAPED_MUTANTS:
            raise CegisError("escaped-mutant gate requirement is immutable")
        object.__setattr__(self, "waivable", _bool(self.waivable, "waivable"))
        if self.waivable:
            raise CegisError("escaped-mutant gate cannot be waived")
        object.__setattr__(
            self,
            "promotion_permitted",
            _bool(self.promotion_permitted, "promotion_permitted"),
        )
        if self.promotion_permitted:
            raise CegisError("CEGIS cannot permit promotion")
        object.__setattr__(
            self,
            "completeness_claimed",
            _bool(self.completeness_claimed, "completeness_claimed"),
        )
        if self.completeness_claimed:
            raise CegisError("escaped-mutant gate cannot claim completeness")

    @property
    def blocked(self) -> bool:
        return bool(self.escaped_critical_mutant_ids)

    @property
    def later_promotion_blocked(self) -> bool:
        return True

    def to_facts(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "counterexample_set_cid": self.counterexample_set_cid,
            "critical_seed_ids": self.critical_seed_ids,
            "escaped_critical_mutant_ids": self.escaped_critical_mutant_ids,
            "killed_critical_mutant_ids": self.killed_critical_mutant_ids,
            "campaign_confirmed": self.campaign_confirmed,
            "requirement_id": self.requirement_id,
            "waivable": False,
            "promotion_permitted": False,
            "completeness_claimed": False,
            "blocked": self.blocked,
        }

    @classmethod
    def from_report(
        cls,
        *,
        candidate_id: str,
        counterexample_set_cid: str,
        critical_seed_ids: Sequence[str],
        escaped_critical_mutant_ids: Sequence[str],
        killed_critical_mutant_ids: Sequence[str] = (),
        campaign_confirmed: bool = False,
    ) -> EscapedMutantGate:
        return cls(
            candidate_id=candidate_id,
            counterexample_set_cid=counterexample_set_cid,
            critical_seed_ids=tuple(critical_seed_ids),
            escaped_critical_mutant_ids=tuple(escaped_critical_mutant_ids),
            killed_critical_mutant_ids=tuple(killed_critical_mutant_ids),
            campaign_confirmed=campaign_confirmed,
        )


@dataclass(frozen=True)
class AssuranceCampaignReport:
    """Bounded campaign observation against one candidate/set pair."""

    candidate_id: str
    counterexample_set_cid: str
    status: AssuranceApiStatus
    interface_id: str
    reason_code: str
    findings: tuple[ValidationFinding, ...]
    escaped_critical_mutant_ids: tuple[str, ...]
    killed_critical_mutant_ids: tuple[str, ...]
    receipt_cids: tuple[str, ...]
    plan_cid: str
    gate: EscapedMutantGate
    campaign_confirmed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _identifier(self.candidate_id, "candidate_id"))
        object.__setattr__(
            self,
            "counterexample_set_cid",
            _identifier(self.counterexample_set_cid, "counterexample_set_cid"),
        )
        object.__setattr__(self, "status", _enum(self.status, AssuranceApiStatus, "status"))
        object.__setattr__(self, "interface_id", _identifier(self.interface_id, "interface_id"))
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        if not isinstance(self.findings, Sequence) or isinstance(
            self.findings, (str, bytes, bytearray, memoryview)
        ):
            raise CegisError("findings must be a sequence")
        if len(self.findings) > MAX_ITEMS:
            raise CegisError("findings exceed their item bound")
        for item in self.findings:
            if not isinstance(item, ValidationFinding):
                raise CegisError("findings must contain ValidationFinding records")
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(
            self,
            "escaped_critical_mutant_ids",
            _strings(
                self.escaped_critical_mutant_ids,
                "escaped_critical_mutant_ids",
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "killed_critical_mutant_ids",
            _strings(
                self.killed_critical_mutant_ids,
                "killed_critical_mutant_ids",
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "receipt_cids",
            _strings(self.receipt_cids, "receipt_cids", identifiers=True),
        )
        object.__setattr__(
            self, "plan_cid", _identifier(self.plan_cid, "plan_cid", required=False)
        )
        if not isinstance(self.gate, EscapedMutantGate):
            raise CegisError("gate must be EscapedMutantGate")
        object.__setattr__(
            self, "campaign_confirmed", _bool(self.campaign_confirmed, "campaign_confirmed")
        )


def _load_campaign_api() -> Any | None:
    try:
        from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.api import (
            create_assurance_campaign_api,
        )
    except Exception:
        return None
    try:
        return create_assurance_campaign_api()
    except Exception:
        return None


def _identifier_safe(value: str) -> bool:
    try:
        _identifier(value, "identifier")
    except Exception:
        return False
    return True


def _assurance_plan_cid(payload: Mapping[str, Any]) -> str:
    return content_identity({"schema": ASSURANCE_CAMPAIGN_SCHEMA, **dict(payload)})


def _report_bucket(status: Any) -> str:
    text = str(status or "").lower()
    if any(token in text for token in ("kill", "killed")):
        return "killed"
    if any(token in text for token in ("surviv", "escaped", "equivalent")):
        return "survivor"
    if any(token in text for token in ("invalid", "uncompilable")):
        return "invalid"
    return "inconclusive"


def _mapping_from_result(raw: Any) -> Mapping[str, Any]:
    if hasattr(raw, "to_dict") and callable(raw.to_dict):
        converted = raw.to_dict()
        if isinstance(converted, Mapping):
            return converted
    if isinstance(raw, Mapping):
        return raw
    payload: dict[str, Any] = {}
    for key in (
        "result_cid",
        "plan_cid",
        "finding_cids",
        "candidate_reports",
        "survivor_count",
        "killed_count",
        "precise_nonclaims",
        "terminal_status",
    ):
        if hasattr(raw, key):
            payload[key] = getattr(raw, key)
    return payload


class AssuranceCounterexampleAdapter:
    """Narrow adapter over existing ``AssuranceCampaignApi@1``.

    This is not an assurance engine.  It probes and invokes the canonical
    campaign API, inspects the closed attack seeds locally, and converts every
    failure into a persistent typed CEGIS counterexample.  Missing or
    incompatible leaves never become simulated success and never waive the
    escaped-mutant promotion floor.
    """

    def __init__(
        self,
        campaign_api: Any | None = None,
        *,
        seeds: Sequence[AssuranceSeed] = (),
        adapter_revision: str = ASSURANCE_COUNTEREXAMPLE_ADAPTER_REVISION,
        emitted_at_ms: int = 0,
    ) -> None:
        self.adapter_revision = _identifier(adapter_revision, "adapter_revision")
        self.emitted_at_ms = _nonnegative_int(emitted_at_ms, "emitted_at_ms")
        self._injected = campaign_api is not None
        self._campaign_api = campaign_api
        if seeds in (None, ()):
            sealed = default_assurance_seeds()
        else:
            if not isinstance(seeds, Sequence) or isinstance(
                seeds, (str, bytes, bytearray, memoryview)
            ):
                raise CegisError("seeds must be a sequence")
            if len(seeds) > MAX_ITEMS:
                raise CegisError("assurance seeds exceed their item bound")
            sealed_list: list[AssuranceSeed] = []
            for item in seeds:
                if not isinstance(item, AssuranceSeed):
                    raise CegisError("seeds must contain AssuranceSeed records")
                sealed_list.append(item)
            sealed = tuple(sealed_list)
        self.seeds = sealed
        self.last_report: AssuranceCampaignReport | None = None

    @property
    def interface_id(self) -> str:
        api = self._campaign_api
        pinned = getattr(api, "interface_id", None) if api is not None else None
        if isinstance(pinned, str) and pinned:
            return pinned
        return ASSURANCE_CAMPAIGN_API_INTERFACE_PIN

    def campaign_api(self) -> Any | None:
        if self._campaign_api is not None or self._injected:
            return self._campaign_api
        self._campaign_api = _load_campaign_api()
        return self._campaign_api

    def probe(self, command: str = EXECUTE_MUTATION_CAMPAIGN_COMMAND) -> Mapping[str, Any]:
        command = _identifier(command, "command")
        api = self.campaign_api()
        if api is None:
            return {
                "command": command,
                "available": False,
                "status": AssuranceApiStatus.TYPED_UNAVAILABLE.value,
                "reason_code": "assurance_api_unavailable",
                "interface_id": self.interface_id,
            }
        probe_fn = getattr(api, "probe_api", None)
        if callable(probe_fn):
            try:
                payload = probe_fn(command)
            except Exception:
                return {
                    "command": command,
                    "available": False,
                    "status": AssuranceApiStatus.INCOMPATIBLE.value,
                    "reason_code": "assurance_api_probe_failed",
                    "interface_id": self.interface_id,
                }
            if isinstance(payload, Mapping):
                available = bool(payload.get("available"))
                status_text = str(payload.get("status") or "")
                if available and status_text in {"", AssuranceApiStatus.AVAILABLE.value}:
                    return {
                        "command": command,
                        "available": True,
                        "status": AssuranceApiStatus.AVAILABLE.value,
                        "reason_code": "assurance_api_available",
                        "interface_id": self.interface_id,
                    }
                reason = str(payload.get("reason_code") or "assurance_api_unavailable")
                if not _identifier_safe(reason):
                    reason = "assurance_api_unavailable"
                return {
                    "command": command,
                    "available": False,
                    "status": (
                        AssuranceApiStatus.INCOMPATIBLE.value
                        if available
                        else str(payload.get("status") or AssuranceApiStatus.TYPED_UNAVAILABLE.value)
                    ),
                    "reason_code": reason,
                    "interface_id": self.interface_id,
                }
        if callable(getattr(api, command, None)):
            return {
                "command": command,
                "available": True,
                "status": AssuranceApiStatus.AVAILABLE.value,
                "reason_code": "assurance_api_available",
                "interface_id": self.interface_id,
            }
        return {
            "command": command,
            "available": False,
            "status": AssuranceApiStatus.TYPED_UNAVAILABLE.value,
            "reason_code": "assurance_api_missing_command",
            "interface_id": self.interface_id,
        }

    def evaluate(
        self,
        candidate: SynthesisCandidate,
        counterexamples: CounterexampleSet,
    ) -> AssuranceCampaignReport:
        if not isinstance(candidate, SynthesisCandidate):
            raise CegisError("candidate must be SynthesisCandidate")
        if not isinstance(counterexamples, CounterexampleSet):
            raise CegisError("counterexamples must be CounterexampleSet")
        set_cid = counterexamples.content_id
        bindings = candidate.procedure.bindings
        seeds = self.seeds
        local_escaped: list[AssuranceSeed] = []
        local_killed: list[AssuranceSeed] = []
        for seed in seeds:
            if inspect_assurance_seed(candidate.procedure, seed):
                if seed.expected_killed:
                    local_escaped.append(seed)
                continue
            local_killed.append(seed)

        plan_payload = {
            "adapter_revision": self.adapter_revision,
            "interface_id": self.interface_id,
            "tree_id": bindings.tree_id,
            "repository_commit": bindings.repository_commit,
            "candidate_id": candidate.candidate_id,
            "counterexample_set_cid": set_cid,
            "seed_ids": [item.seed_id for item in seeds],
            "attack_classes": [item.attack_class.value for item in seeds],
            "completeness_claimed": False,
            "promotion_permitted": False,
        }
        plan_cid = _assurance_plan_cid(plan_payload)
        probe = self.probe(EXECUTE_MUTATION_CAMPAIGN_COMMAND)
        available = bool(probe.get("available"))
        status_text = str(probe.get("status") or AssuranceApiStatus.TYPED_UNAVAILABLE.value)
        try:
            status = _enum(status_text, AssuranceApiStatus, "status")
        except Exception:
            status = (
                AssuranceApiStatus.AVAILABLE
                if available
                else AssuranceApiStatus.TYPED_UNAVAILABLE
            )
        reason = str(probe.get("reason_code") or "assurance_api_unavailable")
        if not _identifier_safe(reason):
            reason = "assurance_api_unavailable"

        receipt_cids: list[str] = []
        api_escaped_ids: set[str] = set()
        campaign_confirmed = False
        if available:
            api = self.campaign_api()
            execute = getattr(api, EXECUTE_MUTATION_CAMPAIGN_COMMAND, None)
            if not callable(execute):
                status = AssuranceApiStatus.TYPED_UNAVAILABLE
                reason = "assurance_api_missing_command"
            else:
                reports = [
                    {
                        "candidate_cid": candidate.candidate_id,
                        "mutant_identity_cid": seed.seed_id,
                        "attack_class": seed.attack_class.value,
                        "critical": seed.critical,
                        "terminal_status": (
                            "survived_full_verification"
                            if seed in local_escaped
                            else "killed_by_policy"
                        ),
                        "report_cid": f"report.{seed.seed_id}",
                    }
                    for seed in seeds
                ]
                try:
                    raw = execute(
                        plan_payload,
                        {
                            "policy_cid": bindings.policy_revision,
                            "verification_policy_cid": bindings.policy_revision,
                            "tree_id": bindings.tree_id,
                        },
                        precomputed_reports=reports,
                        notes="pcpc-cegis-assurance",
                    )
                except Exception:
                    status = AssuranceApiStatus.INCOMPATIBLE
                    reason = "assurance_api_invocation_failed"
                else:
                    payload = _mapping_from_result(raw)
                    campaign_confirmed = True
                    status = AssuranceApiStatus.AVAILABLE
                    reason = "assurance_api_available"
                    for key in ("result_cid", "plan_cid"):
                        value = payload.get(key)
                        if isinstance(value, str) and value and _identifier_safe(value):
                            receipt_cids.append(value)
                    for item in payload.get("finding_cids") or ():
                        if isinstance(item, str) and item and _identifier_safe(item):
                            receipt_cids.append(item)
                    api_reports = payload.get("candidate_reports") or ()
                    if not isinstance(api_reports, Sequence) or isinstance(
                        api_reports, (str, bytes)
                    ):
                        api_reports = ()
                    for item in api_reports:
                        if not isinstance(item, Mapping):
                            continue
                        report_cid = item.get("report_cid") or item.get("result_cid")
                        if isinstance(report_cid, str) and _identifier_safe(report_cid):
                            receipt_cids.append(report_cid)
                        mutant_id = str(
                            item.get("mutant_identity_cid") or item.get("seed_id") or ""
                        )
                        bucket = _report_bucket(
                            item.get("terminal_status") or item.get("outcome_status")
                        )
                        if bucket == "survivor" and mutant_id and _identifier_safe(mutant_id):
                            api_escaped_ids.add(mutant_id)

        escaped_ids: list[str] = []
        killed_ids: list[str] = []
        findings: list[ValidationFinding] = []
        by_id = {item.seed_id: item for item in seeds}
        for seed in local_escaped:
            if seed.seed_id not in escaped_ids:
                escaped_ids.append(seed.seed_id)
            findings.append(_finding_from_seed(seed, evidence_cids=receipt_cids))
        for mutant_id in sorted(api_escaped_ids):
            if mutant_id in escaped_ids:
                continue
            seed = by_id.get(mutant_id)
            if seed is None or not seed.expected_killed:
                continue
            escaped_ids.append(mutant_id)
            findings.append(_finding_from_seed(seed, evidence_cids=receipt_cids))
        for seed in local_killed:
            if seed.seed_id not in escaped_ids:
                killed_ids.append(seed.seed_id)

        critical_ids = tuple(item.seed_id for item in seeds if item.critical)
        escaped_critical = tuple(
            item.seed_id
            for item in seeds
            if item.critical and item.seed_id in set(escaped_ids)
        )
        killed_critical = tuple(
            item.seed_id
            for item in seeds
            if item.critical and item.seed_id in set(killed_ids)
        )
        gate = EscapedMutantGate.from_report(
            candidate_id=candidate.candidate_id,
            counterexample_set_cid=set_cid,
            critical_seed_ids=critical_ids,
            escaped_critical_mutant_ids=escaped_critical,
            killed_critical_mutant_ids=killed_critical,
            campaign_confirmed=campaign_confirmed,
        )
        report = AssuranceCampaignReport(
            candidate_id=candidate.candidate_id,
            counterexample_set_cid=set_cid,
            status=status,
            interface_id=self.interface_id,
            reason_code=reason,
            findings=tuple(findings),
            escaped_critical_mutant_ids=escaped_critical,
            killed_critical_mutant_ids=killed_critical,
            receipt_cids=tuple(dict.fromkeys(receipt_cids)),
            plan_cid=plan_cid,
            gate=gate,
            campaign_confirmed=campaign_confirmed,
        )
        self.last_report = report
        return report

    def __call__(
        self,
        candidate: SynthesisCandidate,
        counterexamples: CounterexampleSet,
    ) -> ValidationFinding | None:
        report = self.evaluate(candidate, counterexamples)
        if not report.findings:
            return None
        head, *tail = report.findings
        if not tail:
            return head
        return replace(head, additional_findings=tuple(tail))


@dataclass(frozen=True)
class SkippedPair:
    """A candidate/set pair that was not re-evaluated."""

    candidate_id: str
    counterexample_set_cid: str
    source_kind: SynthesisSourceKind
    reason: SkipReason

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _identifier(self.candidate_id, "candidate_id"))
        object.__setattr__(
            self,
            "counterexample_set_cid",
            _identifier(self.counterexample_set_cid, "counterexample_set_cid"),
        )
        object.__setattr__(
            self, "source_kind", _enum(self.source_kind, SynthesisSourceKind, "source_kind")
        )
        object.__setattr__(self, "reason", _enum(self.reason, SkipReason, "reason"))

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.candidate_id, self.counterexample_set_cid)


@dataclass(frozen=True)
class SynthesisUsage:
    """Integer resource accounting for one bounded run."""

    candidates_tried: int = 0
    candidates_skipped: int = 0
    steps_peak: int = 0
    branches_peak: int = 0
    holes_peak: int = 0
    loops_peak: int = 0
    model_calls: int = 0
    tokens: int = 0
    validation_work: int = 0
    proof_work: int = 0
    wall_time_ms: int = 0
    unique_pairs_evaluated: int = 0

    def __post_init__(self) -> None:
        for name in (
            "candidates_tried",
            "candidates_skipped",
            "steps_peak",
            "branches_peak",
            "holes_peak",
            "loops_peak",
            "model_calls",
            "tokens",
            "validation_work",
            "proof_work",
            "wall_time_ms",
            "unique_pairs_evaluated",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name, maximum=MAX_STRUCTURED_CEILING[name]),
            )


MAX_STRUCTURED_CEILING: Final[Mapping[str, int]] = MappingProxyType(
    {
        "candidates_tried": MAX_ITEMS,
        "candidates_skipped": MAX_ITEMS * 2,
        "steps_peak": MAX_STEPS,
        "branches_peak": MAX_BRANCHES,
        "holes_peak": MAX_HOLES,
        "loops_peak": MAX_LOOPS,
        "model_calls": MAX_MODEL_CALLS,
        "tokens": MAX_MODEL_TOKENS,
        "validation_work": MAX_VALIDATION_WORK,
        "proof_work": MAX_PROOF_WORK,
        "wall_time_ms": MAX_WALL_TIME_MS,
        "unique_pairs_evaluated": MAX_ITEMS * 2,
    }
)


@dataclass
class _UsageTracker:
    candidates_tried: int = 0
    candidates_skipped: int = 0
    steps_peak: int = 0
    branches_peak: int = 0
    holes_peak: int = 0
    loops_peak: int = 0
    model_calls: int = 0
    tokens: int = 0
    validation_work: int = 0
    proof_work: int = 0
    wall_time_ms: int = 0
    unique_pairs_evaluated: int = 0

    def note_structure(self, procedure: ProcedureSpec) -> None:
        self.steps_peak = max(self.steps_peak, len(procedure.steps))
        self.branches_peak = max(self.branches_peak, len(procedure.branches))
        self.holes_peak = max(self.holes_peak, len(procedure.holes))
        self.loops_peak = max(self.loops_peak, len(procedure.loops))

    def freeze(self) -> SynthesisUsage:
        return SynthesisUsage(
            candidates_tried=self.candidates_tried,
            candidates_skipped=self.candidates_skipped,
            steps_peak=self.steps_peak,
            branches_peak=self.branches_peak,
            holes_peak=self.holes_peak,
            loops_peak=self.loops_peak,
            model_calls=self.model_calls,
            tokens=self.tokens,
            validation_work=self.validation_work,
            proof_work=self.proof_work,
            wall_time_ms=self.wall_time_ms,
            unique_pairs_evaluated=self.unique_pairs_evaluated,
        )


@dataclass(frozen=True)
class SynthesisRequest:
    """Closed inputs for one bounded CEGIS run."""

    plan: ProcedureSynthesisPlan
    verified_procedures: tuple[VerifiedProcedureSeed, ...] = ()
    templates: tuple[ProcedureSpec, ...] = ()
    anti_unified: tuple[ProcedureSpec, ...] = ()
    patterns: tuple[AntiUnificationPattern, ...] = ()
    enumerative_seeds: tuple[ProcedureSpec, ...] = ()
    model_sketches: tuple[ModelSketch, ...] = ()
    human_candidates: tuple[ProcedureSpec, ...] = ()
    initial_counterexamples: tuple[SynthesisCounterexample, ...] = ()
    validator: CandidateValidator | None = None
    source_episode_cids: tuple[str, ...] = ()
    assurance_seeds: tuple[AssuranceSeed, ...] = ()
    assurance_adapter: AssuranceCounterexampleAdapter | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ProcedureSynthesisPlan):
            raise CegisError("plan must be ProcedureSynthesisPlan")
        object.__setattr__(
            self,
            "verified_procedures",
            _typed_sequence(
                self.verified_procedures, VerifiedProcedureSeed, "verified_procedures"
            ),
        )
        object.__setattr__(
            self, "templates", _spec_sequence(self.templates, "templates")
        )
        object.__setattr__(
            self, "anti_unified", _spec_sequence(self.anti_unified, "anti_unified")
        )
        object.__setattr__(
            self,
            "patterns",
            _typed_sequence(self.patterns, AntiUnificationPattern, "patterns"),
        )
        object.__setattr__(
            self,
            "enumerative_seeds",
            _spec_sequence(self.enumerative_seeds, "enumerative_seeds"),
        )
        object.__setattr__(
            self,
            "model_sketches",
            _typed_sequence(self.model_sketches, ModelSketch, "model_sketches"),
        )
        object.__setattr__(
            self,
            "human_candidates",
            _spec_sequence(self.human_candidates, "human_candidates"),
        )
        object.__setattr__(
            self,
            "initial_counterexamples",
            _typed_sequence(
                self.initial_counterexamples,
                SynthesisCounterexample,
                "initial_counterexamples",
            ),
        )
        if self.validator is not None and not callable(self.validator):
            raise CegisError("validator must be callable or None")
        object.__setattr__(
            self,
            "source_episode_cids",
            _strings(self.source_episode_cids, "source_episode_cids", identifiers=True),
        )
        if self.assurance_seeds not in (None, ()):
            object.__setattr__(
                self,
                "assurance_seeds",
                _typed_sequence(self.assurance_seeds, AssuranceSeed, "assurance_seeds"),
            )
        else:
            object.__setattr__(self, "assurance_seeds", ())
        adapter = self.assurance_adapter
        if adapter is not None and not isinstance(adapter, AssuranceCounterexampleAdapter):
            raise CegisError("assurance_adapter must be AssuranceCounterexampleAdapter or None")
        self._assert_bindings()

    def _assert_bindings(self) -> None:
        expected = self.plan.bindings
        family = self.plan.task_family_id
        for seed in self.verified_procedures:
            _assert_source_bindings(seed.procedure, expected, family, "verified_procedures")
        for field_name in (
            "templates",
            "anti_unified",
            "enumerative_seeds",
            "human_candidates",
        ):
            for spec in getattr(self, field_name):
                _assert_source_bindings(spec, expected, family, field_name)
        for sketch in self.model_sketches:
            _assert_source_bindings(sketch.procedure, expected, family, "model_sketches")
        for pattern in self.patterns:
            if pattern.bindings != expected:
                raise CegisError("pattern exact bindings differ")
            if pattern.task_family_id != family:
                raise CegisError("pattern task family differs")


def _typed_sequence(values: Any, cls: type[Any], field_name: str) -> tuple[Any, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray, memoryview)):
        raise CegisError(f"{field_name} must be a sequence")
    if len(values) > MAX_ITEMS:
        raise CegisError(f"{field_name} exceeds its item bound")
    for item in values:
        if not isinstance(item, cls):
            raise CegisError(f"{field_name} must contain {cls.__name__} records")
    return tuple(values)


def _spec_sequence(values: Any, field_name: str) -> tuple[ProcedureSpec, ...]:
    items = _typed_sequence(values, ProcedureSpec, field_name)
    return tuple(_procedure(item, field_name) for item in items)


def _assert_source_bindings(
    spec: ProcedureSpec,
    bindings: ArtifactBindings,
    task_family_id: str,
    field_name: str,
) -> None:
    if spec.bindings != bindings:
        raise CegisError(f"{field_name} exact bindings differ")
    if spec.task_family_id != task_family_id:
        raise CegisError(f"{field_name} task family differs")


@dataclass(frozen=True)
class SynthesisResult:
    """Terminal CEGIS result.  Exhaustion is typed incomplete, never success."""

    bindings: ArtifactBindings
    plan: ProcedureSynthesisPlan
    plan_artifact: ProcedureSynthesisPlanArtifact
    status: SynthesisStatus
    stop_reason: SynthesisStopReason
    surviving_candidates: tuple[SynthesisCandidate, ...]
    rejected_candidates: tuple[SynthesisCandidate, ...]
    skipped_pairs: tuple[SkippedPair, ...]
    counterexamples: tuple[SynthesisCounterexample, ...]
    counterexample_set_cid: str
    candidate_artifacts: tuple[ProcedureCandidate, ...]
    counterexample_artifacts: tuple[ProcedureSynthesisCounterexampleArtifact, ...]
    usage: SynthesisUsage
    generation_order: tuple[SynthesisSourceKind, ...]
    considered_source_kinds: tuple[SynthesisSourceKind, ...]
    completeness_claimed: bool = False
    escaped_mutant_gate: EscapedMutantGate | None = None
    promotion_requirements: tuple[str, ...] = ()
    assurance_receipt_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        if not isinstance(self.plan, ProcedureSynthesisPlan):
            raise CegisError("plan must be ProcedureSynthesisPlan")
        object.__setattr__(self, "status", _enum(self.status, SynthesisStatus, "status"))
        object.__setattr__(
            self, "stop_reason", _enum(self.stop_reason, SynthesisStopReason, "stop_reason")
        )
        if self.status is SynthesisStatus.CONVERGED:
            if self.stop_reason is not SynthesisStopReason.CONVERGED:
                raise CegisError("converged results require the converged stop reason")
            if not self.surviving_candidates:
                raise CegisError("converged results require a surviving candidate")
        else:
            if self.stop_reason is SynthesisStopReason.CONVERGED:
                raise CegisError("incomplete results cannot use the converged stop reason")
            if self.surviving_candidates:
                raise CegisError("incomplete results cannot retain surviving candidates")
        if self.completeness_claimed:
            raise CegisError("CEGIS cannot claim completeness")
        if any(
            item.procedure.state in {ArtifactState.VERIFIED, ArtifactState.PROMOTED}
            for item in (*self.surviving_candidates, *self.rejected_candidates)
        ):
            raise CegisError("synthesized procedures cannot assert verification or promotion")
        if any(
            artifact.state in {ArtifactState.VERIFIED, ArtifactState.PROMOTED}
            for artifact in self.candidate_artifacts
        ):
            raise CegisError("candidate artifacts cannot assert verification or promotion")
        if self.plan_artifact.state in {ArtifactState.VERIFIED, ArtifactState.PROMOTED}:
            raise CegisError("synthesis plans cannot assert verification or promotion")
        gate = self.escaped_mutant_gate
        if gate is not None and not isinstance(gate, EscapedMutantGate):
            raise CegisError("escaped_mutant_gate must be EscapedMutantGate or None")
        object.__setattr__(
            self,
            "promotion_requirements",
            _strings(self.promotion_requirements, "promotion_requirements", identifiers=True),
        )
        object.__setattr__(
            self,
            "assurance_receipt_cids",
            _strings(self.assurance_receipt_cids, "assurance_receipt_cids", identifiers=True),
        )
        if gate is not None:
            if gate.waivable or gate.promotion_permitted:
                raise CegisError("escaped-mutant gate cannot permit or waive promotion")
            if ZERO_CRITICAL_ESCAPED_MUTANTS not in self.promotion_requirements:
                object.__setattr__(
                    self,
                    "promotion_requirements",
                    self.promotion_requirements + (ZERO_CRITICAL_ESCAPED_MUTANTS,),
                )

    @property
    def incomplete(self) -> bool:
        return self.status is SynthesisStatus.INCOMPLETE

    @property
    def converged(self) -> bool:
        return self.status is SynthesisStatus.CONVERGED


def structural_overflow(
    procedure: ProcedureSpec, plan: ProcedureSynthesisPlan
) -> SynthesisStopReason | None:
    """Return the first structural bound a candidate exceeds, if any."""

    if len(procedure.steps) > plan.max_steps:
        return SynthesisStopReason.STEP_BOUND_EXHAUSTED
    if len(procedure.branches) > plan.max_branches:
        return SynthesisStopReason.BRANCH_BOUND_EXHAUSTED
    if len(procedure.holes) > plan.max_holes:
        return SynthesisStopReason.HOLE_BOUND_EXHAUSTED
    if len(procedure.loops) > plan.max_loops:
        return SynthesisStopReason.LOOP_BOUND_EXHAUSTED
    return None


def replay_hits(procedure: ProcedureSpec, counterexample: SynthesisCounterexample) -> bool:
    """True when an existing counterexample still refutes the candidate."""

    return counterexample.constraints.rejects(procedure)


def _scaffold_from_request(request: SynthesisRequest) -> ProcedureSpec | None:
    if request.templates:
        return request.templates[0]
    if request.enumerative_seeds:
        return request.enumerative_seeds[0]
    if request.verified_procedures:
        return request.verified_procedures[0].procedure
    if request.anti_unified:
        return request.anti_unified[0]
    if request.human_candidates:
        return request.human_candidates[0]
    return None


def _episodes(spec: ProcedureSpec, extra: Sequence[str] = ()) -> tuple[str, ...]:
    ordered: list[str] = []
    for item in (*spec.provenance_cids, *extra):
        if item and item not in ordered:
            ordered.append(item)
    return tuple(ordered)


def _ensure_effect(
    spec: ProcedureSpec, effect_id: str, effect_class: EffectClass
) -> ProcedureSpec:
    if any(item.effect_id == effect_id for item in spec.declared_effects):
        return spec
    extra = ProcedureEffect(effect_id=effect_id, effect_class=effect_class)
    return replace(spec, declared_effects=spec.declared_effects + (extra,))


def _allow_operation(spec: ProcedureSpec, operation: StepOperation) -> ProcedureSpec:
    allowed = spec.authority.allowed_operations
    if operation in allowed:
        return spec
    return replace(
        spec, authority=replace(spec.authority, allowed_operations=allowed + (operation,))
    )


def _require_validation_step(spec: ProcedureSpec, step_id: str) -> ProcedureSpec:
    required = spec.validation.required_step_ids
    if step_id in required:
        return spec
    return replace(
        spec, validation=replace(spec.validation, required_step_ids=required + (step_id,))
    )


def _insertion_target(spec: ProcedureSpec) -> str | None:
    for step in spec.steps:
        if step.operation is StepOperation.CHECK_POSTCONDITION:
            return step.step_id
    for step in spec.steps:
        if step.operation in {
            StepOperation.RUN_SELECTED_TESTS,
            StepOperation.RUN_FULL_TEST_FALLBACK,
            StepOperation.RUN_STATIC_ANALYSIS,
            StepOperation.RUN_TYPE_CHECK,
        }:
            return step.step_id
    return spec.entry_step_id if spec.steps else None


def insert_closed_validation_step(
    spec: ProcedureSpec,
    operation: StepOperation,
    operation_contract: str,
    *,
    effect_id: str,
    effect_class: EffectClass,
    step_id: str,
) -> ProcedureSpec:
    """Insert one closed validation/proof step before the postcondition gate."""

    if any(step.operation is operation for step in spec.steps):
        return spec
    target = _insertion_target(spec)
    if target is None:
        raise CegisError("enumerative insertion requires an existing ProcedureIR skeleton")
    predecessor = next((step for step in spec.steps if step.next_step_id == target), None)
    if predecessor is None:
        raise CegisError("enumerative insertion requires a predecessor for the target step")
    local_name = f"{step_id}-result"
    observation_id = f"observation.{step_id}"
    output_bindings = {"result": f"local:{local_name}"}
    evidence_outputs = (observation_id,)
    observations = spec.observations
    locals_ = spec.locals
    if not any(item.name == local_name for item in locals_):
        locals_ = locals_ + (ProcedureLocal(local_name, ValueType.STRUCTURED),)
    if not any(item.observation_id == observation_id for item in observations):
        evidence_type = (
            "proof-receipt@1" if operation is StepOperation.RUN_PROOF else "validation-receipt@1"
        )
        observations = observations + (
            ProcedureObservation(
                observation_id=observation_id,
                producer_contract=operation_contract,
                output_binding=f"local:{local_name}",
                operator=ConditionOperator.ADMITTED,
                evidence_type=evidence_type,
            ),
        )
    input_bindings = dict(predecessor.output_bindings)
    if not input_bindings and predecessor.input_bindings:
        input_bindings = dict(predecessor.input_bindings)
    if "state" in {value.split(":", 1)[-1] for value in predecessor.output_bindings.values()}:
        input_bindings = {"state": "local:state"}
    elif any(local.name == "state" for local in spec.locals):
        input_bindings = {"state": "local:state"}
    new_step = replace(
        predecessor,
        step_id=step_id,
        operation=operation,
        operation_contract=operation_contract,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        declared_effect_ids=(effect_id,),
        required_authority_ids=predecessor.required_authority_ids
        or spec.authority.requirement_ids,
        evidence_outputs=evidence_outputs,
        next_step_id=target,
        hole_id="",
    )
    steps: list[Any] = []
    for step in spec.steps:
        if step.step_id == predecessor.step_id:
            steps.append(replace(step, next_step_id=step_id))
            steps.append(new_step)
        else:
            steps.append(step)
    updated = replace(
        spec,
        name=f"enumerative.{step_id}.{spec.name}",
        steps=tuple(steps),
        observations=observations,
        locals=locals_,
        provenance_cids=_episodes(spec, (f"enumerative.{step_id}",)),
        state=ArtifactState.CANDIDATE,
    )
    updated = _ensure_effect(updated, effect_id, effect_class)
    updated = _allow_operation(updated, operation)
    updated = _require_validation_step(updated, step_id)
    return validate_procedure_spec(updated)


def enumerate_procedure_variants(
    seed: ProcedureSpec,
    plan: ProcedureSynthesisPlan,
    *,
    constraints: NarrowingConstraints | None = None,
) -> tuple[ProcedureSpec, ...]:
    """Bounded enumerative expansion of one seed ProcedureIR."""

    constraints = constraints or NarrowingConstraints()
    variants: list[ProcedureSpec] = []
    seen: set[str] = set()

    def _accept(spec: ProcedureSpec) -> None:
        if spec.content_id in seen:
            return
        if structural_overflow(spec, plan) is not None:
            return
        if constraints.rejects(spec):
            return
        seen.add(spec.content_id)
        variants.append(spec)

    identity = _as_candidate_spec(seed, name=f"enumerative.identity.{seed.name}")
    try:
        identity = validate_procedure_spec(identity)
    except ProcedureIRValidationError:
        identity = seed
    _accept(identity)
    present = _spec_operations(seed)
    for operation, contract, effect_class, effect_id in _VALIDATION_INSERTIONS:
        if len(variants) >= MAX_GENERATED_VARIANTS:
            break
        if operation in present:
            continue
        if operation is StepOperation.RUN_PROOF and plan.max_proof == 0:
            continue
        if len(seed.steps) + 1 > plan.max_steps:
            continue
        try:
            variant = insert_closed_validation_step(
                seed,
                operation,
                contract,
                effect_id=effect_id,
                effect_class=effect_class,
                step_id=operation.value.lower().replace("_", "-"),
            )
        except (CegisError, ProcedureContractError):
            continue
        _accept(variant)
    stacked = seed
    stacked_changed = False
    for operation, contract, effect_class, effect_id in _VALIDATION_INSERTIONS:
        if len(variants) >= MAX_GENERATED_VARIANTS:
            break
        if operation in _spec_operations(stacked):
            continue
        if operation is StepOperation.RUN_PROOF and plan.max_proof == 0:
            continue
        if len(stacked.steps) + 1 > plan.max_steps:
            continue
        try:
            stacked = insert_closed_validation_step(
                stacked,
                operation,
                contract,
                effect_id=effect_id,
                effect_class=effect_class,
                step_id=operation.value.lower().replace("_", "-"),
            )
        except (CegisError, ProcedureContractError):
            break
        stacked_changed = True
        _accept(stacked)
    if stacked_changed:
        _accept(stacked)
    return tuple(variants)


def lower_pattern_to_spec(
    pattern: AntiUnificationPattern, scaffold: ProcedureSpec
) -> ProcedureSpec:
    """Lower a candidate anti-unification pattern onto a validated scaffold."""

    if pattern.status is not PatternStatus.CANDIDATE:
        raise CegisError("rejected patterns cannot be lowered into candidates")
    required = tuple(step for step in pattern.steps if step.presence is StepPresence.REQUIRED)
    if not required:
        raise CegisError("anti-unified patterns require at least one required step")
    by_operation = {step.operation: step for step in scaffold.steps}
    rebuilt: list[Any] = []
    previous_id = ""
    for index, pattern_step in enumerate(required):
        template = by_operation.get(pattern_step.operation)
        step_id = f"pattern-{index}-{pattern_step.operation.value.lower().replace('_', '-')}"
        if template is None:
            raise CegisError("pattern operation is not present on the scaffold")
        next_id = ""
        rebuilt.append(
            replace(
                template,
                step_id=step_id,
                operation=pattern_step.operation,
                operation_contract=pattern_step.operation_contract or template.operation_contract,
                next_step_id=next_id,
                hole_id=(
                    pattern_step.hole_id
                    if template.operation is StepOperation.REQUEST_TYPED_MODEL_HOLE
                    else ""
                ),
            )
        )
        if previous_id:
            rebuilt[-2] = replace(rebuilt[-2], next_step_id=step_id)
        previous_id = step_id
    if rebuilt:
        suffix = [
            step
            for step in scaffold.steps
            if step.operation
            in {
                StepOperation.CHECK_POSTCONDITION,
                StepOperation.EMIT_RECEIPT,
                StepOperation.ROLLBACK,
            }
            and step.operation not in {item.operation for item in rebuilt}
        ]
        for extra in suffix:
            rebuilt[-1] = replace(rebuilt[-1], next_step_id=extra.step_id)
            rebuilt.append(extra)
    terminals = scaffold.terminal_step_ids
    if rebuilt and rebuilt[-1].step_id not in terminals:
        terminals = (rebuilt[-1].step_id,)
    lowered = replace(
        scaffold,
        name=f"anti-unified.{scaffold.name}",
        steps=tuple(rebuilt) if rebuilt else scaffold.steps,
        terminal_step_ids=terminals,
        provenance_cids=_episodes(scaffold, pattern.source_trace_cids),
        state=ArtifactState.CANDIDATE,
    )
    return validate_procedure_spec(lowered)


def builtin_template_from_scaffold(scaffold: ProcedureSpec) -> ProcedureSpec:
    """Rename a validated scaffold into the built-in focused-validation template."""

    template = replace(
        scaffold,
        name="builtin-template.focused-validation",
        provenance_cids=_episodes(scaffold, ("builtin-template.focused-validation",)),
        state=ArtifactState.CANDIDATE,
    )
    return validate_procedure_spec(template)


def _structural_finding(procedure: ProcedureSpec, exc: Exception) -> ValidationFinding:
    return ValidationFinding(
        kind=CounterexampleKind.STRUCTURAL,
        obligation="invalid-procedure-ir",
        witness={"error": type(exc).__name__},
        constraints=NarrowingConstraints(),
    )


class ProcedureCegis:
    """Bounded CEGIS state machine over declarative ProcedureIR candidates."""

    def __init__(
        self,
        *,
        clock_ms: Callable[[], int] | None = None,
        validator: CandidateValidator | None = None,
        assurance_adapter: AssuranceCounterexampleAdapter | None = None,
        emitted_at_ms: int = 0,
    ) -> None:
        if clock_ms is not None and not callable(clock_ms):
            raise CegisError("clock_ms must be callable or None")
        if validator is not None and not callable(validator):
            raise CegisError("validator must be callable or None")
        if assurance_adapter is not None and not isinstance(
            assurance_adapter, AssuranceCounterexampleAdapter
        ):
            raise CegisError("assurance_adapter must be AssuranceCounterexampleAdapter or None")
        self._clock_ms = clock_ms or (lambda: 0)
        self._validator = validator
        self._assurance_adapter = assurance_adapter
        self._emitted_at_ms = _nonnegative_int(emitted_at_ms, "emitted_at_ms")

    def generate_candidates(self, request: SynthesisRequest) -> tuple[SynthesisCandidate, ...]:
        """Materialize the priority-ordered candidate stream without evaluation."""

        if not isinstance(request, SynthesisRequest):
            raise CegisError("request must be SynthesisRequest")
        return tuple(self._iter_candidates(request))

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if not isinstance(request, SynthesisRequest):
            raise CegisError("request must be SynthesisRequest")
        plan = request.plan
        emitted_at = self._emitted_at_ms
        plan_artifact = plan.to_artifact(emitted_at_ms=emitted_at)
        started_ms = self._clock_ms()
        usage = _UsageTracker()
        counterexamples = CounterexampleSet(request.initial_counterexamples)
        seen_pairs: set[tuple[str, str]] = set()
        skipped: list[SkippedPair] = []
        rejected: list[SynthesisCandidate] = []
        artifacts: list[ProcedureCandidate] = []
        considered: list[SynthesisSourceKind] = []
        last_resource_reason: SynthesisStopReason | None = None
        validator = request.validator or self._validator
        assurance = self._resolve_assurance_adapter(request)
        last_gate: EscapedMutantGate | None = None
        assurance_receipts: list[str] = []
        promotion_requirements: list[str] = []
        if assurance is not None:
            promotion_requirements.append(ZERO_CRITICAL_ESCAPED_MUTANTS)

        def _elapsed() -> int:
            now = self._clock_ms()
            return max(0, now - started_ms)

        def _finish(
            status: SynthesisStatus,
            reason: SynthesisStopReason,
            surviving: Sequence[SynthesisCandidate] = (),
        ) -> SynthesisResult:
            usage.wall_time_ms = _elapsed()
            ce_artifacts = tuple(
                item.to_artifact(plan.bindings, emitted_at_ms=emitted_at)
                for item in counterexamples.members
            )
            ordered_artifacts = tuple(artifacts)
            if surviving:
                surviving_ids = {item.candidate_id for item in surviving}
                leading = [
                    item for item in artifacts if item.procedure.content_id in surviving_ids
                ]
                trailing = [
                    item for item in artifacts if item.procedure.content_id not in surviving_ids
                ]
                ordered_artifacts = tuple(leading + trailing)
            return SynthesisResult(
                bindings=plan.bindings,
                plan=plan,
                plan_artifact=plan_artifact,
                status=status,
                stop_reason=reason,
                surviving_candidates=tuple(surviving),
                rejected_candidates=tuple(rejected),
                skipped_pairs=tuple(skipped),
                counterexamples=counterexamples.members,
                counterexample_set_cid=counterexamples.content_id,
                candidate_artifacts=ordered_artifacts,
                counterexample_artifacts=ce_artifacts,
                usage=usage.freeze(),
                generation_order=plan.generation_order,
                considered_source_kinds=tuple(considered),
                escaped_mutant_gate=last_gate,
                promotion_requirements=tuple(dict.fromkeys(promotion_requirements)),
                assurance_receipt_cids=tuple(dict.fromkeys(assurance_receipts)),
            )

        pending = list(self._iter_candidates(request))
        index = 0
        while index < len(pending):
            usage.wall_time_ms = _elapsed()
            if usage.wall_time_ms >= plan.max_wall_time_ms:
                return _finish(SynthesisStatus.INCOMPLETE, SynthesisStopReason.WALL_BUDGET_EXHAUSTED)
            candidate = pending[index]
            index += 1
            considered.append(candidate.source_kind)
            usage.note_structure(candidate.procedure)
            overflow = structural_overflow(candidate.procedure, plan)
            set_cid = counterexamples.content_id
            pair = (candidate.candidate_id, set_cid)
            if overflow is not None:
                usage.candidates_skipped += 1
                skipped.append(
                    SkippedPair(
                        candidate_id=candidate.candidate_id,
                        counterexample_set_cid=set_cid,
                        source_kind=candidate.source_kind,
                        reason=SkipReason.OUT_OF_BOUNDS,
                    )
                )
                last_resource_reason = overflow
                continue
            if candidate.model_calls or candidate.token_cost:
                if usage.model_calls + candidate.model_calls > plan.max_model_calls:
                    usage.candidates_skipped += 1
                    skipped.append(
                        SkippedPair(
                            candidate_id=candidate.candidate_id,
                            counterexample_set_cid=set_cid,
                            source_kind=candidate.source_kind,
                            reason=SkipReason.MODEL_BUDGET,
                        )
                    )
                    last_resource_reason = SynthesisStopReason.MODEL_CALL_BUDGET_EXHAUSTED
                    continue
                if usage.tokens + candidate.token_cost > plan.max_tokens:
                    usage.candidates_skipped += 1
                    skipped.append(
                        SkippedPair(
                            candidate_id=candidate.candidate_id,
                            counterexample_set_cid=set_cid,
                            source_kind=candidate.source_kind,
                            reason=SkipReason.TOKEN_BUDGET,
                        )
                    )
                    last_resource_reason = SynthesisStopReason.TOKEN_BUDGET_EXHAUSTED
                    continue
                usage.model_calls += candidate.model_calls
                usage.tokens += candidate.token_cost
            proof_needed = 1 if StepOperation.RUN_PROOF in _spec_operations(candidate.procedure) else 0
            if proof_needed and usage.proof_work + proof_needed > plan.max_proof:
                usage.candidates_skipped += 1
                skipped.append(
                    SkippedPair(
                        candidate_id=candidate.candidate_id,
                        counterexample_set_cid=set_cid,
                        source_kind=candidate.source_kind,
                        reason=SkipReason.PROOF_BUDGET,
                    )
                )
                last_resource_reason = SynthesisStopReason.PROOF_BUDGET_EXHAUSTED
                continue
            if pair in seen_pairs:
                usage.candidates_skipped += 1
                skipped.append(
                    SkippedPair(
                        candidate_id=candidate.candidate_id,
                        counterexample_set_cid=set_cid,
                        source_kind=candidate.source_kind,
                        reason=SkipReason.REPEATED_PAIR,
                    )
                )
                continue
            if (
                assurance is None
                and counterexamples.constraints.rejects(candidate.procedure)
                and any(replay_hits(candidate.procedure, item) for item in counterexamples.members)
            ):
                if any(item.pair_key == pair for item in counterexamples.members):
                    seen_pairs.add(pair)
                    usage.candidates_skipped += 1
                    skipped.append(
                        SkippedPair(
                            candidate_id=candidate.candidate_id,
                            counterexample_set_cid=set_cid,
                            source_kind=candidate.source_kind,
                            reason=SkipReason.REPEATED_PAIR,
                        )
                    )
                    continue
                seen_pairs.add(pair)
                usage.candidates_skipped += 1
                skipped.append(
                    SkippedPair(
                        candidate_id=candidate.candidate_id,
                        counterexample_set_cid=set_cid,
                        source_kind=candidate.source_kind,
                        reason=SkipReason.NARROWED,
                    )
                )
                rejected.append(candidate)
                artifacts.append(
                    candidate.to_procedure_candidate(
                        synthesis_plan_cid=plan_artifact.content_id,
                        counterexample_set_cid=set_cid,
                        state=ArtifactState.REJECTED,
                    )
                )
                continue
            if usage.candidates_tried >= plan.max_candidates:
                return _finish(
                    SynthesisStatus.INCOMPLETE, SynthesisStopReason.CANDIDATE_BUDGET_EXHAUSTED
                )
            if usage.validation_work + 1 > plan.max_validation:
                return _finish(
                    SynthesisStatus.INCOMPLETE, SynthesisStopReason.VALIDATION_BUDGET_EXHAUSTED
                )
            seen_pairs.add(pair)
            usage.candidates_tried += 1
            usage.unique_pairs_evaluated += 1
            usage.validation_work += 1
            usage.proof_work += proof_needed
            findings, gate, receipts = self._evaluate(
                candidate, counterexamples, validator, assurance
            )
            if gate is not None:
                last_gate = gate
                if ZERO_CRITICAL_ESCAPED_MUTANTS not in promotion_requirements:
                    promotion_requirements.append(ZERO_CRITICAL_ESCAPED_MUTANTS)
            for receipt in receipts:
                if receipt not in assurance_receipts:
                    assurance_receipts.append(receipt)
            if not findings:
                artifacts.append(
                    candidate.to_procedure_candidate(
                        synthesis_plan_cid=plan_artifact.content_id,
                        counterexample_set_cid=set_cid,
                        state=ArtifactState.CANDIDATE,
                    )
                )
                return _finish(
                    SynthesisStatus.CONVERGED,
                    SynthesisStopReason.CONVERGED,
                    surviving=(candidate,),
                )
            extra_validation = sum(item.validation_cost for item in findings)
            extra_proof = sum(item.proof_cost for item in findings)
            if usage.validation_work + extra_validation > plan.max_validation:
                last_resource_reason = SynthesisStopReason.VALIDATION_BUDGET_EXHAUSTED
            if usage.proof_work + extra_proof > plan.max_proof:
                last_resource_reason = SynthesisStopReason.PROOF_BUDGET_EXHAUSTED
            usage.validation_work += extra_validation
            usage.proof_work += extra_proof
            for finding in findings:
                counterexample = SynthesisCounterexample(
                    kind=finding.kind,
                    obligation=finding.obligation,
                    candidate_id=candidate.candidate_id,
                    counterexample_set_cid=set_cid,
                    witness=finding.witness,
                    constraints=finding.constraints,
                    evidence_cids=finding.evidence_cids,
                )
                counterexamples = counterexamples.add(counterexample)
            rejected.append(candidate)
            artifacts.append(
                candidate.to_procedure_candidate(
                    synthesis_plan_cid=plan_artifact.content_id,
                    counterexample_set_cid=set_cid,
                    state=ArtifactState.REJECTED,
                )
            )
            if last_resource_reason in {
                SynthesisStopReason.VALIDATION_BUDGET_EXHAUSTED,
                SynthesisStopReason.PROOF_BUDGET_EXHAUSTED,
            }:
                return _finish(SynthesisStatus.INCOMPLETE, last_resource_reason)

        if last_resource_reason is not None and not rejected:
            return _finish(SynthesisStatus.INCOMPLETE, last_resource_reason)
        if last_resource_reason is not None and usage.candidates_tried == 0:
            return _finish(SynthesisStatus.INCOMPLETE, last_resource_reason)
        return _finish(SynthesisStatus.INCOMPLETE, SynthesisStopReason.NO_ADMISSIBLE_CANDIDATE)

    def _resolve_assurance_adapter(
        self, request: SynthesisRequest
    ) -> AssuranceCounterexampleAdapter | None:
        campaign_api = None
        if request.assurance_adapter is not None:
            if not request.assurance_seeds:
                return request.assurance_adapter
            campaign_api = request.assurance_adapter.campaign_api()
        elif self._assurance_adapter is not None:
            if not request.assurance_seeds:
                return self._assurance_adapter
            campaign_api = self._assurance_adapter.campaign_api()
        elif not request.assurance_seeds:
            return None
        return AssuranceCounterexampleAdapter(
            campaign_api,
            seeds=request.assurance_seeds,
            emitted_at_ms=self._emitted_at_ms,
        )

    def _evaluate(
        self,
        candidate: SynthesisCandidate,
        counterexamples: CounterexampleSet,
        validator: CandidateValidator | None,
        assurance: AssuranceCounterexampleAdapter | None = None,
    ) -> tuple[tuple[ValidationFinding, ...], EscapedMutantGate | None, tuple[str, ...]]:
        try:
            validate_procedure_spec(candidate.procedure)
        except ProcedureIRValidationError as exc:
            return ((_structural_finding(candidate.procedure, exc),), None, ())
        findings: list[ValidationFinding] = []
        for item in counterexamples.members:
            if replay_hits(candidate.procedure, item):
                findings.append(
                    ValidationFinding(
                        kind=CounterexampleKind.REPLAY,
                        obligation=item.obligation,
                        witness=item.witness,
                        constraints=item.constraints,
                        evidence_cids=item.evidence_cids,
                    )
                )
                break
        assurance_is_validator = isinstance(validator, AssuranceCounterexampleAdapter)
        if validator is not None and not findings:
            finding = validator(candidate, counterexamples)
            if finding is not None:
                if not isinstance(finding, ValidationFinding):
                    raise CegisError("validator must return ValidationFinding or None")
                findings.extend(_flatten_findings(finding))
        gate: EscapedMutantGate | None = None
        receipts: tuple[str, ...] = ()
        if assurance is not None and not assurance_is_validator:
            report = assurance.evaluate(candidate, counterexamples)
            gate = report.gate
            receipts = report.receipt_cids
            if report.findings:
                findings.extend(report.findings)
            elif not report.escaped_critical_mutant_ids:
                # A clean independent campaign is authoritative for its seeds.
                # Constraint replay from earlier candidates must not override it.
                seed_obligations = {seed.obligation for seed in assurance.seeds}
                findings = [
                    item
                    for item in findings
                    if not (
                        item.kind is CounterexampleKind.REPLAY
                        and item.obligation in seed_obligations
                    )
                ]
        elif assurance_is_validator:
            report = validator.last_report  # type: ignore[union-attr]
            if report is not None:
                gate = report.gate
                receipts = report.receipt_cids
                if not report.findings and not report.escaped_critical_mutant_ids:
                    seed_obligations = {seed.obligation for seed in validator.seeds}  # type: ignore[union-attr]
                    findings = [
                        item
                        for item in findings
                        if not (
                            item.kind is CounterexampleKind.REPLAY
                            and item.obligation in seed_obligations
                        )
                    ]
        return (tuple(findings), gate, receipts)

    def _iter_candidates(self, request: SynthesisRequest) -> Iterator[SynthesisCandidate]:
        plan = request.plan
        extra = request.source_episode_cids
        constraints = CounterexampleSet(request.initial_counterexamples).constraints
        for kind in plan.generation_order:
            if kind is SynthesisSourceKind.EXISTING_VERIFIED:
                for seed in request.verified_procedures:
                    yield SynthesisCandidate(
                        source_kind=kind,
                        procedure=_as_candidate_spec(seed.procedure),
                        source_episode_cids=_episodes(
                            seed.procedure, (*seed.source_episode_cids, *extra)
                        ),
                        certificate_cid=seed.certificate_cid,
                    )
            elif kind is SynthesisSourceKind.BUILTIN_TEMPLATE:
                templates = list(request.templates)
                if plan.include_builtin_templates:
                    scaffold = _scaffold_from_request(request)
                    if scaffold is not None:
                        try:
                            templates.append(builtin_template_from_scaffold(scaffold))
                        except ProcedureIRValidationError:
                            pass
                for spec in templates:
                    yield SynthesisCandidate(
                        source_kind=kind,
                        procedure=_as_candidate_spec(spec),
                        source_episode_cids=_episodes(spec, extra),
                    )
            elif kind is SynthesisSourceKind.ANTI_UNIFIED_PATTERN:
                for spec in request.anti_unified:
                    yield SynthesisCandidate(
                        source_kind=kind,
                        procedure=_as_candidate_spec(spec),
                        source_episode_cids=_episodes(spec, extra),
                    )
                scaffold = _scaffold_from_request(request)
                if scaffold is not None:
                    for pattern in request.patterns:
                        try:
                            lowered = lower_pattern_to_spec(pattern, scaffold)
                        except (CegisError, ProcedureContractError):
                            continue
                        yield SynthesisCandidate(
                            source_kind=kind,
                            procedure=lowered,
                            source_episode_cids=_episodes(lowered, extra),
                        )
            elif kind is SynthesisSourceKind.ENUMERATIVE:
                generated = 0
                stop_enumerative = False
                for seed in request.enumerative_seeds:
                    if stop_enumerative:
                        break
                    for variant in enumerate_procedure_variants(
                        seed, plan, constraints=constraints
                    ):
                        if generated >= MAX_GENERATED_VARIANTS:
                            stop_enumerative = True
                            break
                        generated += 1
                        yield SynthesisCandidate(
                            source_kind=kind,
                            procedure=variant,
                            source_episode_cids=_episodes(variant, extra),
                            variant_label=variant.name,
                        )
            elif kind is SynthesisSourceKind.MODEL_SKETCH:
                for sketch in request.model_sketches:
                    yield SynthesisCandidate(
                        source_kind=kind,
                        procedure=_as_candidate_spec(sketch.procedure),
                        source_episode_cids=_episodes(
                            sketch.procedure, (*sketch.source_episode_cids, *extra)
                        ),
                        model_calls=sketch.model_calls,
                        token_cost=sketch.token_cost,
                    )
            elif kind is SynthesisSourceKind.HUMAN:
                for spec in request.human_candidates:
                    yield SynthesisCandidate(
                        source_kind=kind,
                        procedure=_as_candidate_spec(spec),
                        source_episode_cids=_episodes(spec, extra),
                    )


def synthesize_procedure(
    request: SynthesisRequest,
    *,
    clock_ms: Callable[[], int] | None = None,
    validator: CandidateValidator | None = None,
    assurance_adapter: AssuranceCounterexampleAdapter | None = None,
    emitted_at_ms: int = 0,
) -> SynthesisResult:
    """Run bounded CEGIS procedure synthesis."""

    return ProcedureCegis(
        clock_ms=clock_ms,
        validator=validator,
        assurance_adapter=assurance_adapter,
        emitted_at_ms=emitted_at_ms,
    ).synthesize(request)


__all__ = [
    "ASSURANCE_CAMPAIGN_API_INTERFACE_PIN",
    "ASSURANCE_COUNTEREXAMPLE_ADAPTER_REVISION",
    "CEGIS_REVISION",
    "CLASSIFY_MUTATION_OUTCOME_COMMAND",
    "EXECUTE_MUTATION_CAMPAIGN_COMMAND",
    "GENERATION_PRIORITY",
    "REQUIRED_ASSURANCE_ATTACK_CLASSES",
    "ZERO_CRITICAL_ESCAPED_MUTANTS",
    "AssuranceAttackClass",
    "AssuranceCampaignReport",
    "AssuranceCounterexampleAdapter",
    "AssuranceSeed",
    "CandidateValidator",
    "CegisError",
    "CounterexampleKind",
    "CounterexampleSet",
    "EscapedMutantGate",
    "ModelSketch",
    "NarrowingConstraints",
    "ProcedureCegis",
    "ProcedureSynthesisPlan",
    "SkipReason",
    "SkippedPair",
    "SynthesisCandidate",
    "SynthesisCounterexample",
    "SynthesisRequest",
    "SynthesisResult",
    "SynthesisSourceKind",
    "SynthesisStatus",
    "SynthesisStopReason",
    "SynthesisUsage",
    "ValidationFinding",
    "VerifiedProcedureSeed",
    "builtin_template_from_scaffold",
    "default_assurance_seeds",
    "enumerate_procedure_variants",
    "insert_closed_validation_step",
    "inspect_assurance_seed",
    "lower_pattern_to_spec",
    "narrowing_for_attack",
    "replay_hits",
    "structural_overflow",
    "synthesize_procedure",
]
