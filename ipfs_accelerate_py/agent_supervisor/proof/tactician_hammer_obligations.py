"""Fail-closed lowering of admitted Tactician plans into exact proof obligations.

LPR-011 / ``TacticianHammerObligations@1``

This module does not decide that a hypothesis is true.  It maps an admitted
Tactician plan gate receipt, exact goals/hypotheses, and a pinned premise
corpus into:

* one :class:`CodeProofObligation` (and optional existing contract-repair /
  change-propagation obligation identities) **per supported facet clause**;
* an independent :class:`ProgramLogicNativeGoalBinding` with exact
  ``GoalSnapshot``, single-goal Lean/Coq/Isabelle theorem source, proof hole,
  kernel, and toolchain identities; and
* typed :class:`LoweringResidual` records for higher-order, dependent,
  dynamic, native, concurrency, or lifetime semantics the admitted translator
  cannot express.

Authority rules (fail-closed):

* No retrieved, vector, GraphRAG, model, or free-form natural-language text
  becomes an axiom.  Claims are opaque identifier graphs only.
* Every premise, source authority, assumption, tree/corpus/translator/
  toolchain/policy/environment identity is bound and must match current roots.
* Facets (input, information, output, totality, error, effect, auth, resource,
  state, schema, placement, supported ownership/lifetime) lower **separately**.
* Wrong theorem, changed assumptions/imports, source drift, omitted facets,
  inconsistent assumptions, and cross-root premises are hard rejections.
* Compilation is deterministic and byte-stable for identical inputs.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..analysis.program_logic_prediction_contracts import (
    GoalDisposition,
    HypothesisDisposition,
    LogicFacetKind,
    LogicFacetRef,
    LogicHypothesis,
    LogicSubgoal,
    NativeGoalDisposition,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProgramLogicNativeGoalBinding,
    SemanticRoundTripReceipt,
    SourceAuthorityClass,
    TacticianSearchPlan,
)
from ..analysis.program_logic_premise_corpus import (
    ProgramLogicPremise,
    ProgramLogicPremiseCorpus,
)
from ..validation.tactician_plan_gate import (
    TacticianPlanGateDisposition,
    TacticianPlanGateReceipt,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    CodeProofObligation,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


# ---------------------------------------------------------------------------
# Schemas / constants
# ---------------------------------------------------------------------------

TACTICIAN_HAMMER_OBLIGATIONS_INTERFACE: Final = "TacticianHammerObligations@1"
TACTICIAN_HAMMER_OBLIGATION_COMPILER_ID: Final = (
    "tactician-hammer-obligation-compiler@1"
)
PROGRAM_LOGIC_NATIVE_GOAL_COMPILER_ID: Final = (
    "program-logic-native-goal-compiler@1"
)

TACTICIAN_HAMMER_IR_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/tactician-hammer-ir-claim@1"
)
TACTICIAN_HAMMER_OBLIGATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/tactician-hammer-obligation@1"
)
TACTICIAN_HAMMER_OBLIGATION_COMPILATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/tactician-hammer-obligation-compilation@1"
)
LOWERING_RESIDUAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lowering-residual@1"
)
NATIVE_THEOREM_SOURCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/native-theorem-source@1"
)
GOAL_SNAPSHOT_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-snapshot-binding@1"
)
TRANSLATOR_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/translator-capability-binding@1"
)

MAX_OBLIGATIONS: Final = 256
MAX_RESIDUALS: Final = 128
MAX_PREMISE_IDS: Final = 256
MAX_HYPOTHESES: Final = 512
MAX_GOALS: Final = 256
MAX_FACETS: Final = 64
MAX_TEXT_BYTES: Final = 4_096
MAX_SOURCE_BYTES: Final = 65_536
MAX_IMPORTS: Final = 64

CONTRACT_VERSION: Final = 1
PRODUCER_ID: Final = TACTICIAN_HAMMER_OBLIGATION_COMPILER_ID

_SUPPORTED_ITPS: Final[frozenset[str]] = frozenset({"lean", "coq", "isabelle"})

_ITP_KERNEL: Final[Mapping[str, str]] = {
    "lean": "kernel:lean4",
    "coq": "kernel:coq",
    "isabelle": "kernel:isabelle",
}

_ITP_DEFAULT_IMPORTS: Final[Mapping[str, tuple[str, ...]]] = {
    "lean": ("import:Init",),
    "coq": ("import:Coq.Init.Prelude",),
    "isabelle": ("import:Main",),
}

_PROOF_HOLE_MARKERS: Final[Mapping[str, str]] = {
    "lean": "sorry",
    "coq": "Admitted.",
    "isabelle": "sorry",
}

# Translator semantics that admit facet lowering when present.
_BASE_SUPPORTED_SEMANTICS: Final[frozenset[str]] = frozenset(
    {
        "ir",
        "logic_ir",
        "fol",
        "fol_core",
        "propositional",
        "first_order",
    }
)

# Semantics that remain typed residuals unless the translator admits them.
_RESIDUAL_SEMANTICS: Final[frozenset[str]] = frozenset(
    {
        "higher_order",
        "dependent",
        "dynamic",
        "native",
        "concurrency",
        "lifetime",
        "reflection",
        "ffi",
        "unsafe",
    }
)

_RETRIEVED_ASSUMPTION_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "invented",
        "model",
        "llm",
        "retrieved",
        "vector",
        "graphrag",
        "similarity",
        "embedding",
        "hypothesis",
        "nomination",
        "candidate_only",
        "prompt",
        "natural_language",
        "nl_axiom",
        "free_form",
    }
)

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source_body",
        "source_text",
        "snippet",
        "prompt_body",
        "theorem_text",
        "proof_script",
        "natural_language",
        "nl_axiom",
        "free_form_axiom",
    }
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TacticianHammerObligationError(ContractValidationError):
    """Evidence is insufficient or inconsistent for a sound lowering."""


class OmittedFacetError(TacticianHammerObligationError):
    """A required facet was not lowered and was not residualized."""


class CrossRootPremiseError(TacticianHammerObligationError):
    """A premise or goal bound a different authority root set."""


class InconsistentAssumptionError(TacticianHammerObligationError):
    """Assumptions conflict or are not reviewed assumption evidence."""


class UnauthorizedAxiomError(TacticianHammerObligationError):
    """A retrieved/model/NL statement attempted to become an axiom."""


class NativeGoalBindingError(TacticianHammerObligationError):
    """Native theorem / GoalSnapshot / round-trip binding failed."""


class SourceDriftError(NativeGoalBindingError):
    """Native source no longer matches the bound LogicIR claim."""


class WrongTheoremError(NativeGoalBindingError):
    """Native theorem identity does not denote the expected claim."""


class ChangedAssumptionsError(NativeGoalBindingError):
    """Native imports/assumptions drifted from the bound set."""


# ---------------------------------------------------------------------------
# Closed enumerations
# ---------------------------------------------------------------------------


class LoweringFacetKind(str, Enum):
    """Closed set of separately lowerable program-logic facet clauses."""

    INPUT = "input"
    INFORMATION = "information"
    OUTPUT = "output"
    TOTALITY = "totality"
    ERROR = "error"
    EFFECT = "effect"
    AUTH = "auth"
    RESOURCE = "resource"
    STATE = "state"
    SCHEMA = "schema"
    PLACEMENT = "placement"
    OWNERSHIP = "ownership"
    LIFETIME = "lifetime"


class ResidualSemanticKind(str, Enum):
    """Closed vocabulary for translator-unsupported semantics."""

    HIGHER_ORDER = "higher_order"
    DEPENDENT = "dependent"
    DYNAMIC = "dynamic"
    NATIVE = "native"
    CONCURRENCY = "concurrency"
    LIFETIME = "lifetime"
    REFLECTION = "reflection"
    FFI = "ffi"
    UNSAFE = "unsafe"
    UNSUPPORTED_FACET = "unsupported_facet"
    OMITTED_TRANSLATION = "omitted_translation"


class LoweringDisposition(str, Enum):
    """Closed outcomes for one compilation."""

    LOWERED = "lowered"
    PARTIAL = "partial"
    RESIDUAL_ONLY = "residual_only"
    REJECTED = "rejected"


class NativeITPKind(str, Enum):
    LEAN = "lean"
    COQ = "coq"
    ISABELLE = "isabelle"


# Populate claim-token map after enum definition.
_CLAIM_TOKEN_CLAUSES = {
    "input": LoweringFacetKind.INPUT,
    "inputs": LoweringFacetKind.INPUT,
    "caller_input": LoweringFacetKind.INPUT,
    "caller_input_acceptance": LoweringFacetKind.INPUT,
    "information": LoweringFacetKind.INFORMATION,
    "info": LoweringFacetKind.INFORMATION,
    "value_sufficiency": LoweringFacetKind.INFORMATION,
    "information_provenance": LoweringFacetKind.INFORMATION,
    "output": LoweringFacetKind.OUTPUT,
    "outputs": LoweringFacetKind.OUTPUT,
    "output_refinement": LoweringFacetKind.OUTPUT,
    "postcondition": LoweringFacetKind.OUTPUT,
    "totality": LoweringFacetKind.TOTALITY,
    "total": LoweringFacetKind.TOTALITY,
    "nullability": LoweringFacetKind.TOTALITY,
    "range": LoweringFacetKind.TOTALITY,
    "error": LoweringFacetKind.ERROR,
    "errors": LoweringFacetKind.ERROR,
    "allowed_errors": LoweringFacetKind.ERROR,
    "effect": LoweringFacetKind.EFFECT,
    "effects": LoweringFacetKind.EFFECT,
    "permitted_effects": LoweringFacetKind.EFFECT,
    "side_effect": LoweringFacetKind.EFFECT,
    "auth": LoweringFacetKind.AUTH,
    "authorization": LoweringFacetKind.AUTH,
    "capability": LoweringFacetKind.AUTH,
    "capabilities": LoweringFacetKind.AUTH,
    "resource": LoweringFacetKind.RESOURCE,
    "resources": LoweringFacetKind.RESOURCE,
    "state": LoweringFacetKind.STATE,
    "lifecycle": LoweringFacetKind.STATE,
    "temporal": LoweringFacetKind.STATE,
    "consistency": LoweringFacetKind.STATE,
    "schema": LoweringFacetKind.SCHEMA,
    "type": LoweringFacetKind.SCHEMA,
    "serialization": LoweringFacetKind.SCHEMA,
    "constructor": LoweringFacetKind.SCHEMA,
    "placement": LoweringFacetKind.PLACEMENT,
    "registration": LoweringFacetKind.PLACEMENT,
    "ownership": LoweringFacetKind.OWNERSHIP,
    "memory": LoweringFacetKind.OWNERSHIP,
    "memory_safety": LoweringFacetKind.OWNERSHIP,
    "lifetime": LoweringFacetKind.LIFETIME,
}

_FACET_KIND_DEFAULT_CLAUSE: Final[Mapping[LogicFacetKind, LoweringFacetKind]] = {
    LogicFacetKind.TYPE: LoweringFacetKind.SCHEMA,
    LogicFacetKind.INFORMATION: LoweringFacetKind.INFORMATION,
    LogicFacetKind.ERROR: LoweringFacetKind.ERROR,
    LogicFacetKind.EFFECT: LoweringFacetKind.EFFECT,
    LogicFacetKind.AUTHORIZATION: LoweringFacetKind.AUTH,
    LogicFacetKind.RESOURCE: LoweringFacetKind.RESOURCE,
    LogicFacetKind.STATE: LoweringFacetKind.STATE,
    LogicFacetKind.SCHEMA: LoweringFacetKind.SCHEMA,
    LogicFacetKind.PLACEMENT: LoweringFacetKind.PLACEMENT,
    LogicFacetKind.MEMORY: LoweringFacetKind.OWNERSHIP,
    LogicFacetKind.LIFETIME: LoweringFacetKind.LIFETIME,
    LogicFacetKind.TEMPORAL: LoweringFacetKind.STATE,
}

_OWNERSHIP_LIFETIME_CLAUSES: Final[frozenset[LoweringFacetKind]] = frozenset(
    {LoweringFacetKind.OWNERSHIP, LoweringFacetKind.LIFETIME}
)

_RESIDUAL_REQUIRING_SEMANTIC: Final[Mapping[LoweringFacetKind, ResidualSemanticKind]] = {
    LoweringFacetKind.LIFETIME: ResidualSemanticKind.LIFETIME,
}

_HYPOTHESIS_ADMITTED: Final[frozenset[HypothesisDisposition]] = frozenset(
    {
        HypothesisDisposition.PLAN_ADMITTED,
        HypothesisDisposition.NOMINATED,
        HypothesisDisposition.PROVED,
    }
)

_GOAL_LOWERABLE: Final[frozenset[GoalDisposition]] = frozenset(
    {
        GoalDisposition.PLANNED,
        GoalDisposition.ADMITTED,
        GoalDisposition.OPEN,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise TacticianHammerObligationError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise TacticianHammerObligationError(f"{name} is required")
    if len(result.encode("utf-8")) > limit:
        raise TacticianHammerObligationError(f"{name} exceeds its byte bound")
    if result and any(character.isspace() for character in result):
        # Allow structured native theorem sources (multi-line) only for explicit
        # source fields; identifiers remain opaque and whitespace-free.
        raise TacticianHammerObligationError(f"{name} must be an opaque identifier")
    return result


def _identifier(value: Any, name: str) -> str:
    return _text(value, name, required=True)


def _ids(
    value: Sequence[str] | None,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_PREMISE_IDS,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if value is None:
        if required:
            raise TacticianHammerObligationError(f"{name} must not be empty")
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TacticianHammerObligationError(f"{name} must be identifiers")
    if len(value) > limit:
        raise TacticianHammerObligationError(f"{name} exceeds its bounded size")
    items = [_identifier(item, name) for item in value]
    if preserve_order:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        result = tuple(ordered)
    else:
        result = tuple(sorted(set(items)))
    if required and not result:
        raise TacticianHammerObligationError(f"{name} must not be empty")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TacticianHammerObligationError(f"{name} must be a boolean")
    return value


def _digest(payload: Mapping[str, Any] | Sequence[Any] | str, *, prefix: str) -> str:
    if isinstance(payload, str):
        material = payload.encode("utf-8")
    else:
        material = canonical_json_bytes(payload)
    digest = hashlib.sha256(material).hexdigest()
    return f"{prefix}:{digest}"


def _roots(value: Any) -> ProgramLogicAuthorityRoots:
    if isinstance(value, ProgramLogicAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            ProgramLogicAuthorityRoots.from_dict(value)
            if "schema" in value
            else ProgramLogicAuthorityRoots(**value)
        )
    raise TacticianHammerObligationError("roots must be ProgramLogicAuthorityRoots")


def _roots_equal(left: ProgramLogicAuthorityRoots, right: ProgramLogicAuthorityRoots) -> bool:
    return left.to_dict() == right.to_dict()


def _reject_body_markers(mapping: Mapping[str, Any], *, context: str) -> None:
    for key in mapping:
        folded = str(key).casefold().replace("-", "_")
        if folded in _BODY_MARKERS or any(marker in folded for marker in _BODY_MARKERS):
            raise UnauthorizedAxiomError(
                f"{context} cannot carry natural-language or body axiom fields ({key})"
            )


def _tokenize_claim_ref(claim_ref: str) -> tuple[str, ...]:
    text = claim_ref.casefold().replace(":", "_").replace("-", "_").replace("/", "_")
    return tuple(part for part in text.split("_") if part)


def _clause_from_claim(claim_ref: str, facet: LogicFacetRef | None) -> LoweringFacetKind:
    tokens = _tokenize_claim_ref(claim_ref)
    # Prefer more specific multi-token matches first via single-token map.
    for token in tokens:
        if token in _CLAIM_TOKEN_CLAUSES:
            return _CLAIM_TOKEN_CLAUSES[token]
    if facet is not None:
        return _FACET_KIND_DEFAULT_CLAUSE[facet.kind]
    return LoweringFacetKind.SCHEMA


def _normalize_itp(value: Any) -> NativeITPKind:
    if isinstance(value, NativeITPKind):
        return value
    text = _identifier(str(value), "native_itp").casefold().replace("-", "")
    if text in {"lean", "lean4"}:
        return NativeITPKind.LEAN
    if text == "coq":
        return NativeITPKind.COQ
    if text in {"isabelle", "isabellehol"}:
        return NativeITPKind.ISABELLE
    raise NativeGoalBindingError("native_itp must be lean, coq, or isabelle")


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssumptionBinding:
    """A reviewed assumption identity; never a free-form solver axiom."""

    assumption_id: str
    kind: str
    evidence_ref: str
    authority: SourceAuthorityClass = SourceAuthorityClass.AUTHORITATIVE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "assumption_id", _identifier(self.assumption_id, "assumption_id")
        )
        object.__setattr__(self, "kind", _identifier(self.kind, "kind"))
        object.__setattr__(
            self, "evidence_ref", _identifier(self.evidence_ref, "evidence_ref")
        )
        if isinstance(self.authority, SourceAuthorityClass):
            authority = self.authority
        else:
            authority = SourceAuthorityClass(self.authority)
        object.__setattr__(self, "authority", authority)
        kind = self.kind.casefold().replace("-", "_")
        if "assumption" not in kind:
            raise UnauthorizedAxiomError(
                "assumptions must be reviewed assumption evidence, never free-form axioms"
            )
        if any(marker in kind for marker in _RETRIEVED_ASSUMPTION_MARKERS):
            raise UnauthorizedAxiomError(
                "retrieved, model, vector, or nomination statements cannot become axioms"
            )
        for field_value in (self.assumption_id, self.evidence_ref):
            folded = field_value.casefold()
            if any(
                marker in folded
                for marker in (
                    "llm:",
                    "model:",
                    "vector:",
                    "retrieved:",
                    "nl:",
                    "prompt:",
                )
            ):
                raise UnauthorizedAxiomError(
                    "retrieved/model identifiers cannot be used as assumption evidence"
                )
        if authority in {
            SourceAuthorityClass.NOMINATING,
            SourceAuthorityClass.DIAGNOSTIC,
            SourceAuthorityClass.NONE,
        }:
            raise UnauthorizedAxiomError(
                "assumptions used for lowering require authoritative or conformance authority"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "assumption_id": self.assumption_id,
            "kind": self.kind,
            "evidence_ref": self.evidence_ref,
            "authority": self.authority.value,
        }


@dataclass(frozen=True)
class TranslatorCapabilityBinding:
    """Exact capability-report fact authorizing LogicIR / native translation."""

    SCHEMA: ClassVar[str] = TRANSLATOR_CAPABILITY_SCHEMA

    capability_id: str
    capability_revision: str
    translator_id: str
    reconstruction_compatible: bool = True
    supported_semantics: tuple[str, ...] = ()
    supported_itps: tuple[str, ...] = ("lean",)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "capability_id", _identifier(self.capability_id, "capability_id")
        )
        object.__setattr__(
            self,
            "capability_revision",
            _identifier(self.capability_revision, "capability_revision"),
        )
        object.__setattr__(
            self, "translator_id", _identifier(self.translator_id, "translator_id")
        )
        object.__setattr__(
            self,
            "reconstruction_compatible",
            _bool(self.reconstruction_compatible, "reconstruction_compatible"),
        )
        if not self.reconstruction_compatible:
            raise TacticianHammerObligationError(
                "LogicIR lowering requires independently reconstructable capability semantics"
            )
        semantics = _ids(self.supported_semantics, "supported_semantics", required=True)
        if not set(semantics) & _BASE_SUPPORTED_SEMANTICS:
            raise TacticianHammerObligationError(
                "capability report must explicitly admit LogicIR / FOL semantics"
            )
        object.__setattr__(self, "supported_semantics", semantics)
        itps = _ids(self.supported_itps, "supported_itps", required=True, limit=8)
        normalized = tuple(sorted({_normalize_itp(item).value for item in itps}))
        object.__setattr__(self, "supported_itps", normalized)

    def admits(self, semantic: str) -> bool:
        return semantic.casefold().replace("-", "_") in {
            item.casefold().replace("-", "_") for item in self.supported_semantics
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "capability_id": self.capability_id,
            "capability_revision": self.capability_revision,
            "translator_id": self.translator_id,
            "reconstruction_compatible": self.reconstruction_compatible,
            "supported_semantics": list(self.supported_semantics),
            "supported_itps": list(self.supported_itps),
        }


@dataclass(frozen=True)
class ExistingObligationLink:
    """Optional identity link into an existing supervisor obligation compilation."""

    interface: str
    compilation_id: str
    obligation_ids: tuple[str, ...] = ()
    kind_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "interface", _identifier(self.interface, "interface"))
        object.__setattr__(
            self, "compilation_id", _identifier(self.compilation_id, "compilation_id")
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _ids(self.obligation_ids, "obligation_ids", required=False),
        )
        object.__setattr__(
            self, "kind_refs", _ids(self.kind_refs, "kind_refs", required=False)
        )
        allowed = {
            "ProofObligation",
            "ContractRepairObligationCompilation",
            "ChangePropagationObligation",
            "ChangePropagationObligationCompilation",
            "CodeProofObligation",
        }
        if self.interface not in allowed and not self.interface.endswith(
            ("Obligation", "ObligationCompilation")
        ):
            raise TacticianHammerObligationError(
                "existing obligation link must name a known supervisor obligation interface"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "compilation_id": self.compilation_id,
            "obligation_ids": list(self.obligation_ids),
            "kind_refs": list(self.kind_refs),
        }


@dataclass(frozen=True)
class ObligationContext:
    """Exact non-semantic bindings shared by every emitted claim."""

    capability: TranslatorCapabilityBinding
    assumptions: tuple[AssumptionBinding, ...]
    existing_obligation_links: tuple[ExistingObligationLink, ...] = ()
    native_itp: NativeITPKind = NativeITPKind.LEAN
    kernel_id: str = ""
    import_ids: tuple[str, ...] = ()
    translation_map_id: str = ""
    hammer_premise_selection_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.capability, TranslatorCapabilityBinding):
            raise TacticianHammerObligationError(
                "obligation context requires a TranslatorCapabilityBinding"
            )
        if not isinstance(self.assumptions, Sequence) or not self.assumptions:
            raise TacticianHammerObligationError(
                "obligation context requires reviewed assumption bindings"
            )
        if not all(isinstance(item, AssumptionBinding) for item in self.assumptions):
            raise TacticianHammerObligationError(
                "assumptions must be AssumptionBinding values"
            )
        assumption_ids = [item.assumption_id for item in self.assumptions]
        if len(assumption_ids) != len(set(assumption_ids)):
            raise InconsistentAssumptionError("duplicate assumption identities")
        object.__setattr__(
            self,
            "assumptions",
            tuple(sorted(self.assumptions, key=lambda item: item.assumption_id)),
        )
        links = self.existing_obligation_links or ()
        if not all(isinstance(item, ExistingObligationLink) for item in links):
            raise TacticianHammerObligationError(
                "existing_obligation_links must be ExistingObligationLink values"
            )
        object.__setattr__(
            self,
            "existing_obligation_links",
            tuple(sorted(links, key=lambda item: (item.interface, item.compilation_id))),
        )
        object.__setattr__(self, "native_itp", _normalize_itp(self.native_itp))
        if self.native_itp.value not in self.capability.supported_itps:
            raise NativeGoalBindingError(
                f"native ITP {self.native_itp.value} is not admitted by the translator capability"
            )
        kernel = self.kernel_id or _ITP_KERNEL[self.native_itp.value]
        object.__setattr__(self, "kernel_id", _identifier(kernel, "kernel_id"))
        if self.kernel_id in {"solver-only", "solver_only"}:
            raise NativeGoalBindingError(
                "native goal bindings require a kernel identity, not solver-only"
            )
        imports = self.import_ids or _ITP_DEFAULT_IMPORTS[self.native_itp.value]
        object.__setattr__(
            self,
            "import_ids",
            _ids(imports, "import_ids", required=True, limit=MAX_IMPORTS),
        )
        object.__setattr__(
            self,
            "translation_map_id",
            _text(self.translation_map_id, "translation_map_id", required=False),
        )
        object.__setattr__(
            self,
            "hammer_premise_selection_id",
            _text(
                self.hammer_premise_selection_id,
                "hammer_premise_selection_id",
                required=False,
            ),
        )

    @property
    def assumption_ids(self) -> tuple[str, ...]:
        return tuple(item.assumption_id for item in self.assumptions)


# ---------------------------------------------------------------------------
# LogicIR claim + obligation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IRClaim(CanonicalContract):
    """Small immutable LogicIR claim with every source of authority named."""

    SCHEMA: ClassVar[str] = TACTICIAN_HAMMER_IR_CLAIM_SCHEMA

    predicate: str
    subject_id: str
    facet_kind: LoweringFacetKind
    premise_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    source_authority: SourceAuthorityClass
    repository_id: str
    tree_id: str
    corpus_id: str
    translator_id: str
    toolchain_id: str
    policy_id: str
    environment_id: str
    capability_id: str
    capability_revision: str
    goal_id: str = ""
    hypothesis_id: str = ""
    subgoal_id: str = ""
    facet_id: str = ""
    plan_id: str = ""
    translation_map_id: str = ""
    counterexample_target_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate", _identifier(self.predicate, "predicate"))
        object.__setattr__(self, "subject_id", _identifier(self.subject_id, "subject_id"))
        object.__setattr__(
            self, "facet_kind", LoweringFacetKind(self.facet_kind)
        )
        for name in (
            "repository_id",
            "tree_id",
            "corpus_id",
            "translator_id",
            "toolchain_id",
            "policy_id",
            "environment_id",
            "capability_id",
            "capability_revision",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "goal_id",
            "hypothesis_id",
            "subgoal_id",
            "facet_id",
            "plan_id",
            "translation_map_id",
            "counterexample_target_ref",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self, "premise_ids", _ids(self.premise_ids, "premise_ids", required=True)
        )
        object.__setattr__(
            self, "source_ids", _ids(self.source_ids, "source_ids", required=True)
        )
        object.__setattr__(
            self,
            "assumption_ids",
            _ids(self.assumption_ids, "assumption_ids", required=True),
        )
        if isinstance(self.source_authority, SourceAuthorityClass):
            authority = self.source_authority
        else:
            authority = SourceAuthorityClass(self.source_authority)
        object.__setattr__(self, "source_authority", authority)
        # Reject natural-language predicate smuggling.
        if any(marker in self.predicate.casefold() for marker in _BODY_MARKERS):
            raise UnauthorizedAxiomError(
                "IRClaim predicate cannot be a natural-language axiom body"
            )
        if " " in self.predicate:
            raise UnauthorizedAxiomError(
                "IRClaim predicate must be an opaque identifier, not prose"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "subject_id": self.subject_id,
            "facet_kind": self.facet_kind.value,
            "premise_ids": list(self.premise_ids),
            "source_ids": list(self.source_ids),
            "assumption_ids": list(self.assumption_ids),
            "source_authority": self.source_authority.value,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "corpus_id": self.corpus_id,
            "translator_id": self.translator_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "environment_id": self.environment_id,
            "capability_id": self.capability_id,
            "capability_revision": self.capability_revision,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "subgoal_id": self.subgoal_id,
            "facet_id": self.facet_id,
            "plan_id": self.plan_id,
            "translation_map_id": self.translation_map_id,
            "counterexample_target_ref": self.counterexample_target_ref,
        }

    def to_logic_ir(self) -> dict[str, Any]:
        """Backend-neutral, fully bound lowering payload."""

        return self.to_dict()

    @property
    def claim_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class TacticianHammerProofObligation(CanonicalContract):
    """One facet-separated claim plus its existing code-proof envelope."""

    SCHEMA: ClassVar[str] = TACTICIAN_HAMMER_OBLIGATION_SCHEMA

    kind: LoweringFacetKind
    goal_id: str
    claim: IRClaim
    code_obligation: CodeProofObligation
    premise_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    hypothesis_id: str = ""
    subgoal_id: str = ""
    facet_id: str = ""
    existing_obligation_refs: tuple[str, ...] = ()
    hammer_premise_selection_id: str = ""
    translation_map_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", LoweringFacetKind(self.kind))
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        if not isinstance(self.claim, IRClaim):
            raise TacticianHammerObligationError("obligation requires a typed IRClaim")
        if not isinstance(self.code_obligation, CodeProofObligation):
            raise TacticianHammerObligationError(
                "obligation requires a typed CodeProofObligation"
            )
        object.__setattr__(
            self, "premise_ids", _ids(self.premise_ids, "premise_ids", required=True)
        )
        object.__setattr__(
            self, "source_ids", _ids(self.source_ids, "source_ids", required=True)
        )
        for name in (
            "hypothesis_id",
            "subgoal_id",
            "facet_id",
            "hammer_premise_selection_id",
            "translation_map_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self,
            "existing_obligation_refs",
            _ids(self.existing_obligation_refs, "existing_obligation_refs"),
        )
        if self.claim.facet_kind is not self.kind:
            raise TacticianHammerObligationError(
                "obligation kind must match the LogicIR claim facet"
            )
        if self.code_obligation.repository_tree_id != self.claim.tree_id:
            raise TacticianHammerObligationError(
                "code obligation tree must match the LogicIR claim"
            )
        if set(self.claim.premise_ids).difference(self.premise_ids):
            raise TacticianHammerObligationError(
                "claim premises must be carried by obligation premise_ids"
            )
        if set(self.claim.source_ids).difference(self.source_ids):
            raise TacticianHammerObligationError(
                "claim sources must be carried by obligation source_ids"
            )

    @property
    def obligation_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "subgoal_id": self.subgoal_id,
            "facet_id": self.facet_id,
            "claim": self.claim.to_dict(),
            "code_obligation": self.code_obligation.to_dict(),
            "premise_ids": list(self.premise_ids),
            "source_ids": list(self.source_ids),
            "existing_obligation_refs": list(self.existing_obligation_refs),
            "hammer_premise_selection_id": self.hammer_premise_selection_id,
            "translation_map_id": self.translation_map_id,
        }


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoweringResidual(CanonicalContract):
    """Typed residual for translator-unsupported or unsupported-facet semantics."""

    SCHEMA: ClassVar[str] = LOWERING_RESIDUAL_SCHEMA

    residual_id: str
    kind: ResidualSemanticKind
    subject_id: str
    reason_ref: str
    goal_id: str = ""
    hypothesis_id: str = ""
    facet_id: str = ""
    claim_ref: str = ""
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "residual_id", _identifier(self.residual_id, "residual_id")
        )
        object.__setattr__(self, "kind", ResidualSemanticKind(self.kind))
        object.__setattr__(self, "subject_id", _identifier(self.subject_id, "subject_id"))
        object.__setattr__(self, "reason_ref", _identifier(self.reason_ref, "reason_ref"))
        for name in ("goal_id", "hypothesis_id", "facet_id", "claim_ref"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if self.semantic_authority is not False:
            raise TacticianHammerObligationError(
                "residuals cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "residual_id": self.residual_id,
            "kind": self.kind.value,
            "subject_id": self.subject_id,
            "reason_ref": self.reason_ref,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "facet_id": self.facet_id,
            "claim_ref": self.claim_ref,
            "semantic_authority": False,
        }


# ---------------------------------------------------------------------------
# Native theorem source + GoalSnapshot binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeTheoremSource(CanonicalContract):
    """Single-goal native theorem source with exactly one proof hole."""

    SCHEMA: ClassVar[str] = NATIVE_THEOREM_SOURCE_SCHEMA

    source_id: str
    itp: NativeITPKind
    theorem_id: str
    claim_id: str
    source_text: str
    proof_hole_marker: str
    import_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    kernel_id: str
    toolchain_id: str
    environment_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(self, "itp", _normalize_itp(self.itp))
        object.__setattr__(
            self, "theorem_id", _identifier(self.theorem_id, "theorem_id")
        )
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        if not isinstance(self.source_text, str) or not self.source_text.strip():
            raise NativeGoalBindingError("source_text is required")
        if len(self.source_text.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise NativeGoalBindingError("source_text exceeds its byte bound")
        # source_text is structured identifier material, not free-form NL axioms.
        object.__setattr__(self, "source_text", self.source_text)
        object.__setattr__(
            self,
            "proof_hole_marker",
            _identifier(self.proof_hole_marker, "proof_hole_marker"),
        )
        expected_marker = _PROOF_HOLE_MARKERS[self.itp.value]
        if self.proof_hole_marker != expected_marker:
            raise NativeGoalBindingError(
                f"proof hole marker must be the exact ITP hole ({expected_marker})"
            )
        if self.source_text.count(self.proof_hole_marker) != 1:
            raise NativeGoalBindingError(
                "native theorem source must contain exactly one proof hole"
            )
        if self.theorem_id not in self.source_text:
            raise WrongTheoremError(
                "native theorem source must embed the bound theorem_id"
            )
        if self.claim_id not in self.source_text:
            raise WrongTheoremError(
                "native theorem source must embed the LogicIR claim_id"
            )
        object.__setattr__(
            self,
            "import_ids",
            _ids(self.import_ids, "import_ids", required=True, limit=MAX_IMPORTS),
        )
        object.__setattr__(
            self,
            "assumption_ids",
            _ids(self.assumption_ids, "assumption_ids", required=True),
        )
        for name in ("kernel_id", "toolchain_id", "environment_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for import_id in self.import_ids:
            token = import_id.split(":", 1)[-1]
            if token and token not in self.source_text and import_id not in self.source_text:
                # Imports must be reflected in source (by id or bare name).
                raise ChangedAssumptionsError(
                    "native theorem imports drifted from the bound import set"
                )

    def _payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "itp": self.itp.value,
            "theorem_id": self.theorem_id,
            "claim_id": self.claim_id,
            "source_text": self.source_text,
            "proof_hole_marker": self.proof_hole_marker,
            "import_ids": list(self.import_ids),
            "assumption_ids": list(self.assumption_ids),
            "kernel_id": self.kernel_id,
            "toolchain_id": self.toolchain_id,
            "environment_id": self.environment_id,
        }

    @property
    def proof_hole_id(self) -> str:
        return _digest(
            {
                "theorem_id": self.theorem_id,
                "claim_id": self.claim_id,
                "marker": self.proof_hole_marker,
            },
            prefix="hole",
        )


@dataclass(frozen=True)
class GoalSnapshotBinding(CanonicalContract):
    """Identity-level GoalSnapshot binding (no unbounded ITP process I/O)."""

    SCHEMA: ClassVar[str] = GOAL_SNAPSHOT_BINDING_SCHEMA

    snapshot_id: str
    itp: NativeITPKind
    theorem_id: str
    claim_id: str
    goal_text_id: str
    hypothesis_ids: tuple[str, ...]
    import_ids: tuple[str, ...]
    kernel_id: str
    toolchain_id: str
    environment_id: str
    source_position_id: str = ""
    native_command_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _identifier(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(self, "itp", _normalize_itp(self.itp))
        for name in (
            "theorem_id",
            "claim_id",
            "goal_text_id",
            "kernel_id",
            "toolchain_id",
            "environment_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "hypothesis_ids",
            _ids(self.hypothesis_ids, "hypothesis_ids", required=True),
        )
        object.__setattr__(
            self,
            "import_ids",
            _ids(self.import_ids, "import_ids", required=True, limit=MAX_IMPORTS),
        )
        object.__setattr__(
            self,
            "source_position_id",
            _text(self.source_position_id, "source_position_id", required=False),
        )
        object.__setattr__(
            self,
            "native_command_id",
            _text(self.native_command_id, "native_command_id", required=False),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "itp": self.itp.value,
            "theorem_id": self.theorem_id,
            "claim_id": self.claim_id,
            "goal_text_id": self.goal_text_id,
            "hypothesis_ids": list(self.hypothesis_ids),
            "import_ids": list(self.import_ids),
            "kernel_id": self.kernel_id,
            "toolchain_id": self.toolchain_id,
            "environment_id": self.environment_id,
            "source_position_id": self.source_position_id,
            "native_command_id": self.native_command_id,
        }


# ---------------------------------------------------------------------------
# Compilation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TacticianHammerObligationCompilation:
    """Complete deterministic collection compiled for one admitted plan."""

    roots: ProgramLogicAuthorityRoots
    plan_id: str
    plan_content_id: str
    gate_receipt_id: str
    corpus_content_id: str
    disposition: LoweringDisposition
    obligations: tuple[TacticianHammerProofObligation, ...]
    native_bindings: tuple[ProgramLogicNativeGoalBinding, ...]
    native_sources: tuple[NativeTheoremSource, ...]
    goal_snapshots: tuple[GoalSnapshotBinding, ...]
    residuals: tuple[LoweringResidual, ...] = ()
    existing_obligation_links: tuple[ExistingObligationLink, ...] = ()
    compiler_id: str = TACTICIAN_HAMMER_OBLIGATION_COMPILER_ID
    native_compiler_id: str = PROGRAM_LOGIC_NATIVE_GOAL_COMPILER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "plan_content_id", _identifier(self.plan_content_id, "plan_content_id")
        )
        object.__setattr__(
            self, "gate_receipt_id", _identifier(self.gate_receipt_id, "gate_receipt_id")
        )
        object.__setattr__(
            self,
            "corpus_content_id",
            _identifier(self.corpus_content_id, "corpus_content_id"),
        )
        object.__setattr__(
            self, "disposition", LoweringDisposition(self.disposition)
        )
        object.__setattr__(
            self, "compiler_id", _identifier(self.compiler_id, "compiler_id")
        )
        object.__setattr__(
            self,
            "native_compiler_id",
            _identifier(self.native_compiler_id, "native_compiler_id"),
        )
        if len(self.obligations) > MAX_OBLIGATIONS:
            raise TacticianHammerObligationError("compilation exceeds obligation bound")
        if len(self.residuals) > MAX_RESIDUALS:
            raise TacticianHammerObligationError("compilation exceeds residual bound")
        if not all(
            isinstance(item, TacticianHammerProofObligation) for item in self.obligations
        ):
            raise TacticianHammerObligationError("obligations must be typed")
        if not all(
            isinstance(item, ProgramLogicNativeGoalBinding)
            for item in self.native_bindings
        ):
            raise TacticianHammerObligationError("native_bindings must be typed")
        if not all(isinstance(item, NativeTheoremSource) for item in self.native_sources):
            raise TacticianHammerObligationError("native_sources must be typed")
        if not all(isinstance(item, GoalSnapshotBinding) for item in self.goal_snapshots):
            raise TacticianHammerObligationError("goal_snapshots must be typed")
        if not all(isinstance(item, LoweringResidual) for item in self.residuals):
            raise TacticianHammerObligationError("residuals must be typed")
        # Deterministic ordering — keep native artifacts aligned with obligations
        # by LogicIR claim identity so consumers can zip streams safely.
        object.__setattr__(
            self,
            "obligations",
            tuple(
                sorted(
                    self.obligations,
                    key=lambda item: (
                        item.goal_id,
                        item.kind.value,
                        item.facet_id,
                        item.claim.claim_id,
                        item.obligation_id,
                    ),
                )
            ),
        )
        claim_order = {
            obligation.claim.claim_id: index
            for index, obligation in enumerate(self.obligations)
        }

        def _claim_rank(claim_id: str) -> tuple[int, str]:
            return (claim_order.get(claim_id, len(claim_order)), claim_id)

        object.__setattr__(
            self,
            "native_bindings",
            tuple(
                sorted(
                    self.native_bindings,
                    key=lambda item: (
                        *_claim_rank(item.logic_ir_obligation_id),
                        item.binding_id,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "native_sources",
            tuple(
                sorted(
                    self.native_sources,
                    key=lambda item: (*_claim_rank(item.claim_id), item.source_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "goal_snapshots",
            tuple(
                sorted(
                    self.goal_snapshots,
                    key=lambda item: (*_claim_rank(item.claim_id), item.snapshot_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "residuals",
            tuple(
                sorted(
                    self.residuals,
                    key=lambda item: (item.kind.value, item.subject_id, item.residual_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "existing_obligation_links",
            tuple(
                sorted(
                    self.existing_obligation_links,
                    key=lambda item: (item.interface, item.compilation_id),
                )
            ),
        )
        if self.disposition is LoweringDisposition.LOWERED and not self.obligations:
            raise TacticianHammerObligationError(
                "lowered compilations require at least one obligation"
            )
        if self.disposition is LoweringDisposition.RESIDUAL_ONLY and self.obligations:
            raise TacticianHammerObligationError(
                "residual_only compilations cannot carry obligations"
            )
        if self.disposition is LoweringDisposition.LOWERED and self.residuals:
            # Pure lowered path has no residuals; partial allows both.
            raise TacticianHammerObligationError(
                "lowered disposition cannot carry residuals; use partial"
            )
        # One native binding per obligation when lowered/partial with obligations.
        if self.obligations:
            if len(self.native_bindings) != len(self.obligations):
                raise TacticianHammerObligationError(
                    "each obligation requires exactly one native goal binding"
                )
            if len(self.native_sources) != len(self.obligations):
                raise TacticianHammerObligationError(
                    "each obligation requires exactly one native theorem source"
                )
            if len(self.goal_snapshots) != len(self.obligations):
                raise TacticianHammerObligationError(
                    "each obligation requires exactly one GoalSnapshot binding"
                )

    @property
    def obligation_ids(self) -> tuple[str, ...]:
        return tuple(item.obligation_id for item in self.obligations)

    @property
    def kinds(self) -> tuple[LoweringFacetKind, ...]:
        return tuple(item.kind for item in self.obligations)

    @property
    def compilation_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": TACTICIAN_HAMMER_OBLIGATIONS_INTERFACE,
            "schema": TACTICIAN_HAMMER_OBLIGATION_COMPILATION_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "compiler_id": self.compiler_id,
            "native_compiler_id": self.native_compiler_id,
            "roots": self.roots.to_dict(),
            "plan_id": self.plan_id,
            "plan_content_id": self.plan_content_id,
            "gate_receipt_id": self.gate_receipt_id,
            "corpus_content_id": self.corpus_content_id,
            "disposition": self.disposition.value,
            "obligation_ids": list(self.obligation_ids),
            "kinds": [item.value for item in self.kinds],
            "obligations": [item.to_dict() for item in self.obligations],
            "native_bindings": [item.to_dict() for item in self.native_bindings],
            "native_sources": [item.to_dict() for item in self.native_sources],
            "goal_snapshots": [item.to_dict() for item in self.goal_snapshots],
            "residuals": [item.to_dict() for item in self.residuals],
            "existing_obligation_links": [
                item.to_dict() for item in self.existing_obligation_links
            ],
        }

    def to_canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


# ---------------------------------------------------------------------------
# ProgramLogicNativeGoalCompiler
# ---------------------------------------------------------------------------


class ProgramLogicNativeGoalCompiler:
    """Compile a single LogicIR claim into an exact native ITP binding."""

    compiler_id: ClassVar[str] = PROGRAM_LOGIC_NATIVE_GOAL_COMPILER_ID

    def compile(
        self,
        claim: IRClaim,
        context: ObligationContext,
        *,
        roots: ProgramLogicAuthorityRoots,
        expected_theorem_id: str | None = None,
        expected_import_ids: Sequence[str] | None = None,
        expected_assumption_ids: Sequence[str] | None = None,
        provided_source_text: str | None = None,
    ) -> tuple[NativeTheoremSource, GoalSnapshotBinding, ProgramLogicNativeGoalBinding]:
        if not isinstance(claim, IRClaim):
            raise NativeGoalBindingError("native compiler requires a typed IRClaim")
        if not isinstance(context, ObligationContext):
            raise NativeGoalBindingError("native compiler requires ObligationContext")
        roots = _roots(roots)
        self._assert_claim_roots(claim, roots, context)

        itp = context.native_itp
        theorem_id = expected_theorem_id or _digest(
            {
                "claim_id": claim.claim_id,
                "facet": claim.facet_kind.value,
                "goal_id": claim.goal_id,
            },
            prefix="thm",
        )
        theorem_id = _identifier(theorem_id, "theorem_id")

        import_ids = tuple(
            _ids(
                expected_import_ids if expected_import_ids is not None else context.import_ids,
                "import_ids",
                required=True,
                limit=MAX_IMPORTS,
            )
        )
        if set(import_ids) != set(context.import_ids):
            raise ChangedAssumptionsError(
                "native import set must match the obligation context imports"
            )
        assumption_ids = tuple(
            _ids(
                expected_assumption_ids
                if expected_assumption_ids is not None
                else context.assumption_ids,
                "assumption_ids",
                required=True,
            )
        )
        if set(assumption_ids) != set(claim.assumption_ids):
            raise ChangedAssumptionsError(
                "native assumption set must match the LogicIR claim assumptions"
            )
        if set(assumption_ids) != set(context.assumption_ids):
            raise ChangedAssumptionsError(
                "native assumption set must match the obligation context assumptions"
            )

        source_text = (
            provided_source_text
            if provided_source_text is not None
            else self._emit_source(
                itp=itp,
                theorem_id=theorem_id,
                claim=claim,
                import_ids=import_ids,
                assumption_ids=assumption_ids,
            )
        )
        # Independent round-trip of the native statement → LogicIR claim identity.
        recovered = self.round_trip_claim_id(source_text, itp=itp)
        if recovered != claim.claim_id:
            raise WrongTheoremError(
                "native statement does not denote the expected LogicIR claim"
            )
        recovered_theorem = self.round_trip_theorem_id(source_text, itp=itp)
        if recovered_theorem != theorem_id:
            raise WrongTheoremError(
                "native statement theorem identity drifted from the bound theorem_id"
            )
        recovered_imports = self.round_trip_imports(source_text, itp=itp)
        if set(recovered_imports) != set(import_ids):
            raise ChangedAssumptionsError(
                "native imports drifted from the bound import set"
            )
        recovered_assumptions = self.round_trip_assumptions(source_text)
        if set(recovered_assumptions) != set(assumption_ids):
            raise ChangedAssumptionsError(
                "native assumptions drifted from the bound assumption set"
            )

        hole = _PROOF_HOLE_MARKERS[itp.value]
        source = NativeTheoremSource(
            source_id=_digest(
                {
                    "theorem_id": theorem_id,
                    "claim_id": claim.claim_id,
                    "source_text": source_text,
                },
                prefix="native-src",
            ),
            itp=itp,
            theorem_id=theorem_id,
            claim_id=claim.claim_id,
            source_text=source_text,
            proof_hole_marker=hole,
            import_ids=import_ids,
            assumption_ids=assumption_ids,
            kernel_id=context.kernel_id,
            toolchain_id=roots.toolchain_id,
            environment_id=roots.environment_id,
        )

        goal_text_id = _digest(
            {"claim_id": claim.claim_id, "facet": claim.facet_kind.value},
            prefix="goal-text",
        )
        snapshot = GoalSnapshotBinding(
            snapshot_id=_digest(
                {
                    "theorem_id": theorem_id,
                    "claim_id": claim.claim_id,
                    "goal_text_id": goal_text_id,
                    "itp": itp.value,
                },
                prefix="goal-snapshot",
            ),
            itp=itp,
            theorem_id=theorem_id,
            claim_id=claim.claim_id,
            goal_text_id=goal_text_id,
            hypothesis_ids=tuple(
                f"hyp:{item}" for item in assumption_ids
            ),
            import_ids=import_ids,
            kernel_id=context.kernel_id,
            toolchain_id=roots.toolchain_id,
            environment_id=roots.environment_id,
            source_position_id=_digest(
                {"theorem_id": theorem_id, "claim_id": claim.claim_id},
                prefix="pos",
            ),
            native_command_id=_digest(
                {
                    "itp": itp.value,
                    "theorem_id": theorem_id,
                    "toolchain_id": roots.toolchain_id,
                },
                prefix="native-cmd",
            ),
        )

        round_trip = SemanticRoundTripReceipt(
            receipt_id=_digest(
                {
                    "claim_id": claim.claim_id,
                    "source_id": source.source_id,
                    "method": "identifier-embedded-round-trip@1",
                },
                prefix="round-trip",
            ),
            logic_ir_claim_id=claim.claim_id,
            native_statement_id=source.source_id,
            equivalence_method="identifier-embedded-round-trip@1",
            disposition=NativeGoalDisposition.ROUND_TRIP_OK,
            assumption_refs=assumption_ids,
        )
        binding = ProgramLogicNativeGoalBinding(
            roots=roots,
            binding_id=_digest(
                {
                    "claim_id": claim.claim_id,
                    "source_id": source.source_id,
                    "snapshot_id": snapshot.snapshot_id,
                },
                prefix="binding",
            ),
            logic_ir_obligation_id=claim.claim_id,
            premise_ids=claim.premise_ids,
            native_itp_id=f"itp:{itp.value}",
            goal_snapshot_id=snapshot.snapshot_id,
            native_theorem_source_id=source.source_id,
            proof_hole_id=source.proof_hole_id,
            kernel_id=context.kernel_id,
            semantic_round_trip=round_trip,
            disposition=NativeGoalDisposition.ROUND_TRIP_OK,
            import_ids=import_ids,
            environment_id=roots.environment_id,
            source_position_id=snapshot.source_position_id,
            invalidation_refs=(
                roots.tree_id,
                roots.toolchain_id,
                roots.translator_id,
                roots.environment_id,
            ),
        )
        return source, snapshot, binding

    def round_trip_claim_id(
        self, source_text: str, *, itp: NativeITPKind | str = NativeITPKind.LEAN
    ) -> str:
        """Independently recover the LogicIR claim id from native source text."""

        _normalize_itp(itp)
        if not isinstance(source_text, str) or not source_text.strip():
            raise SourceDriftError("empty native source cannot round-trip")
        match = re.search(r"claim_id=([A-Za-z0-9_.:\-]+)", source_text)
        if not match:
            # Also accept opaque baguqeera... / sha content ids after marker.
            match = re.search(
                r"LogicIRClaim\[([A-Za-z0-9_.:\-]+)\]", source_text
            )
        if not match:
            raise SourceDriftError(
                "native source does not embed a recoverable LogicIR claim identity"
            )
        return match.group(1)

    def round_trip_theorem_id(
        self, source_text: str, *, itp: NativeITPKind | str = NativeITPKind.LEAN
    ) -> str:
        _normalize_itp(itp)
        match = re.search(r"theorem_id=([A-Za-z0-9_.:\-]+)", source_text)
        if not match:
            match = re.search(
                r"(?:theorem|Lemma|lemma)\s+([A-Za-z0-9_.:\-]+)", source_text
            )
        if not match:
            raise WrongTheoremError(
                "native source does not embed a recoverable theorem identity"
            )
        return match.group(1)

    def round_trip_imports(
        self, source_text: str, *, itp: NativeITPKind | str = NativeITPKind.LEAN
    ) -> tuple[str, ...]:
        _normalize_itp(itp)
        found = re.findall(r"import_id=([A-Za-z0-9_.:\-]+)", source_text)
        if not found:
            raise ChangedAssumptionsError(
                "native source does not embed recoverable import identities"
            )
        return tuple(sorted(set(found)))

    def round_trip_assumptions(self, source_text: str) -> tuple[str, ...]:
        found = re.findall(r"assumption_id=([A-Za-z0-9_.:\-]+)", source_text)
        if not found:
            raise ChangedAssumptionsError(
                "native source does not embed recoverable assumption identities"
            )
        return tuple(sorted(set(found)))

    def _emit_source(
        self,
        *,
        itp: NativeITPKind,
        theorem_id: str,
        claim: IRClaim,
        import_ids: Sequence[str],
        assumption_ids: Sequence[str],
    ) -> str:
        hole = _PROOF_HOLE_MARKERS[itp.value]
        import_lines = [
            f"-- import_id={item}" for item in sorted(import_ids)
        ]
        assumption_lines = [
            f"-- assumption_id={item}" for item in sorted(assumption_ids)
        ]
        header = "\n".join(import_lines + assumption_lines)
        # Structured, identifier-only theorem body — not natural-language axioms.
        if itp is NativeITPKind.LEAN:
            body = (
                f"{header}\n"
                f"-- theorem_id={theorem_id}\n"
                f"-- claim_id={claim.claim_id}\n"
                f"theorem {theorem_id} : LogicIRClaim[{claim.claim_id}] := by\n"
                f"  {hole}\n"
            )
        elif itp is NativeITPKind.COQ:
            body = (
                f"{header}\n"
                f"(* theorem_id={theorem_id} *)\n"
                f"(* claim_id={claim.claim_id} *)\n"
                f"Lemma {theorem_id} : LogicIRClaim[{claim.claim_id}].\n"
                f"Proof.\n"
                f"  {hole}\n"
            )
        else:  # isabelle
            body = (
                f"{header}\n"
                f"(* theorem_id={theorem_id} *)\n"
                f"(* claim_id={claim.claim_id} *)\n"
                f"lemma {theorem_id}: \"LogicIRClaim[{claim.claim_id}]\"\n"
                f"  {hole}\n"
            )
        return body

    @staticmethod
    def _assert_claim_roots(
        claim: IRClaim,
        roots: ProgramLogicAuthorityRoots,
        context: ObligationContext,
    ) -> None:
        if claim.repository_id != roots.repository_id:
            raise CrossRootPremiseError("claim repository_id drifted from roots")
        if claim.tree_id != roots.tree_id:
            raise CrossRootPremiseError("claim tree_id drifted from roots")
        if claim.corpus_id != roots.corpus_id:
            raise CrossRootPremiseError("claim corpus_id drifted from roots")
        if claim.translator_id != roots.translator_id:
            raise CrossRootPremiseError("claim translator_id drifted from roots")
        if claim.toolchain_id != roots.toolchain_id:
            raise CrossRootPremiseError("claim toolchain_id drifted from roots")
        if claim.policy_id != roots.policy_id:
            raise CrossRootPremiseError("claim policy_id drifted from roots")
        if claim.environment_id != roots.environment_id:
            raise CrossRootPremiseError("claim environment_id drifted from roots")
        if claim.translator_id != context.capability.translator_id:
            raise CrossRootPremiseError(
                "claim translator_id must match the capability translator binding"
            )
        if claim.capability_id != context.capability.capability_id:
            raise CrossRootPremiseError("claim capability_id drifted from context")
        if claim.capability_revision != context.capability.capability_revision:
            raise CrossRootPremiseError(
                "claim capability_revision drifted from context"
            )


# ---------------------------------------------------------------------------
# TacticianHammerObligationCompiler
# ---------------------------------------------------------------------------


class TacticianHammerObligationCompiler:
    """Lower admitted tactic plans to exact existing proof obligations."""

    compiler_id: ClassVar[str] = TACTICIAN_HAMMER_OBLIGATION_COMPILER_ID

    def __init__(
        self,
        *,
        native_compiler: ProgramLogicNativeGoalCompiler | None = None,
    ) -> None:
        self._native = native_compiler or ProgramLogicNativeGoalCompiler()

    def compile(
        self,
        gate_receipt: TacticianPlanGateReceipt | Mapping[str, Any],
        plan: TacticianSearchPlan | Mapping[str, Any],
        goals: Sequence[ProgramLogicGoal | Mapping[str, Any]],
        hypotheses: Sequence[LogicHypothesis | Mapping[str, Any]],
        corpus: ProgramLogicPremiseCorpus | Mapping[str, Any],
        context: ObligationContext,
        *,
        current_roots: ProgramLogicAuthorityRoots | Mapping[str, Any] | None = None,
    ) -> TacticianHammerObligationCompilation:
        receipt = self._decode_receipt(gate_receipt)
        typed_plan = self._decode_plan(plan)
        typed_goals = self._decode_goals(goals)
        typed_hypotheses = self._decode_hypotheses(hypotheses)
        typed_corpus = self._decode_corpus(corpus)
        if not isinstance(context, ObligationContext):
            raise TacticianHammerObligationError(
                "compiler requires a typed ObligationContext"
            )

        roots = _roots(current_roots) if current_roots is not None else receipt.roots
        self._validate_gate(receipt, typed_plan, typed_corpus, roots, context)
        premise_by_id = self._index_premises(typed_corpus, roots)
        goal_by_id = {goal.goal_id: goal for goal in typed_goals}

        # Cross-check plan goals are present.
        missing_goals = set(typed_plan.goal_ids) - set(goal_by_id)
        if missing_goals:
            raise TacticianHammerObligationError(
                f"plan goals missing from goal inventory: {sorted(missing_goals)}"
            )

        self._assert_no_cross_root_goals(typed_goals, roots)
        self._assert_no_cross_root_hypotheses(typed_hypotheses, roots)
        self._assert_hypothesis_targets(typed_hypotheses, goal_by_id, typed_plan)
        self._assert_assumption_consistency(context, typed_goals, typed_hypotheses)

        obligations: list[TacticianHammerProofObligation] = []
        residuals: list[LoweringResidual] = []
        native_bindings: list[ProgramLogicNativeGoalBinding] = []
        native_sources: list[NativeTheoremSource] = []
        goal_snapshots: list[GoalSnapshotBinding] = []

        permitted_subgoals = set(receipt.permitted_subgoal_ids)
        subgoal_by_id = {item.subgoal_id: item for item in typed_plan.subgoals}

        # Consistency-only path: only lower the consistency subgoal surface.
        if receipt.disposition is TacticianPlanGateDisposition.CONSISTENCY_ONLY:
            consistency = receipt.consistency_subgoal
            if consistency is None:
                raise TacticianHammerObligationError(
                    "consistency_only receipt requires a consistency subgoal plan"
                )
            goal = goal_by_id.get(consistency.goal_id)
            if goal is None:
                raise TacticianHammerObligationError(
                    "consistency subgoal goal_id is not in the goal inventory"
                )
            facet_clauses = self._clauses_for_goal(goal, claim_ref=consistency.claim_ref)
            for clause, facet in facet_clauses:
                residual_or_ob = self._lower_clause(
                    clause=clause,
                    facet=facet,
                    goal=goal,
                    hypothesis=None,
                    subgoal=None,
                    claim_ref=consistency.claim_ref or f"consistency:{clause.value}",
                    plan=typed_plan,
                    receipt=receipt,
                    context=context,
                    roots=roots,
                    premise_by_id=premise_by_id,
                    premise_ids=tuple(consistency.premise_ids),
                )
                if isinstance(residual_or_ob, LoweringResidual):
                    residuals.append(residual_or_ob)
                else:
                    obligations.append(residual_or_ob)
        else:
            # Full admitted lowering: each permitted subgoal + admitted hypothesis facets.
            work_items = self._collect_work_items(
                typed_plan=typed_plan,
                typed_goals=goal_by_id,
                typed_hypotheses=typed_hypotheses,
                permitted_subgoals=permitted_subgoals,
                subgoal_by_id=subgoal_by_id,
                premise_by_id=premise_by_id,
            )
            covered_required: dict[str, set[str]] = {
                goal_id: set() for goal_id in typed_plan.goal_ids
            }

            for item in work_items:
                residual_or_ob = self._lower_clause(
                    clause=item["clause"],
                    facet=item["facet"],
                    goal=item["goal"],
                    hypothesis=item["hypothesis"],
                    subgoal=item["subgoal"],
                    claim_ref=item["claim_ref"],
                    plan=typed_plan,
                    receipt=receipt,
                    context=context,
                    roots=roots,
                    premise_by_id=premise_by_id,
                    premise_ids=item["premise_ids"],
                )
                if isinstance(residual_or_ob, LoweringResidual):
                    residuals.append(residual_or_ob)
                    # Residuals still discharge the originating required facet
                    # (as an explicit unsupported/residual surface).
                    if item["facet"] is not None:
                        covered_required[item["goal"].goal_id].add(item["facet"].facet_id)
                    elif residual_or_ob.facet_id:
                        covered_required[item["goal"].goal_id].add(
                            residual_or_ob.facet_id
                        )
                else:
                    obligations.append(residual_or_ob)
                    if item["facet"] is not None:
                        covered_required[item["goal"].goal_id].add(item["facet"].facet_id)

            # Omitted required facets are hard failures.
            for goal in typed_goals:
                if goal.goal_id not in typed_plan.goal_ids:
                    continue
                if goal.disposition not in _GOAL_LOWERABLE:
                    continue
                required = {
                    facet.facet_id
                    for facet in goal.required_facets
                    if not facet.unsupported
                }
                unsupported = {facet.facet_id for facet in goal.unsupported_facets}
                covered = covered_required.get(goal.goal_id, set()) | unsupported
                missing = required - covered
                if missing:
                    raise OmittedFacetError(
                        f"required facets omitted from lowering: {sorted(missing)}"
                    )
                # Explicit unsupported facets become residuals when not already covered.
                for facet in goal.unsupported_facets:
                    if facet.facet_id in covered_required.get(goal.goal_id, set()):
                        continue
                    residuals.append(
                        self._residual(
                            kind=ResidualSemanticKind.UNSUPPORTED_FACET,
                            subject_id=facet.subject_symbol_id,
                            reason_ref=f"unsupported-facet:{facet.facet_id}",
                            goal_id=goal.goal_id,
                            facet_id=facet.facet_id,
                            claim_ref=facet.contract_ref or facet.facet_id,
                        )
                    )

        # Emit native bindings for every obligation.
        for obligation in obligations:
            source, snapshot, binding = self._native.compile(
                obligation.claim, context, roots=roots
            )
            # Independent re-check of round-trip identity.
            recovered = self._native.round_trip_claim_id(
                source.source_text, itp=source.itp
            )
            if recovered != obligation.claim.claim_id:
                raise SourceDriftError(
                    "post-compile native round-trip drifted from LogicIR claim"
                )
            native_sources.append(source)
            goal_snapshots.append(snapshot)
            native_bindings.append(binding)

        if obligations and residuals:
            disposition = LoweringDisposition.PARTIAL
        elif obligations:
            disposition = LoweringDisposition.LOWERED
        elif residuals:
            disposition = LoweringDisposition.RESIDUAL_ONLY
        else:
            raise TacticianHammerObligationError(
                "compilation produced neither obligations nor residuals"
            )

        return TacticianHammerObligationCompilation(
            roots=roots,
            plan_id=typed_plan.plan_id,
            plan_content_id=receipt.plan_content_id,
            gate_receipt_id=receipt.content_id,
            corpus_content_id=receipt.corpus_content_id,
            disposition=disposition,
            obligations=tuple(obligations),
            native_bindings=tuple(native_bindings),
            native_sources=tuple(native_sources),
            goal_snapshots=tuple(goal_snapshots),
            residuals=tuple(residuals),
            existing_obligation_links=context.existing_obligation_links,
        )

    # -- work-item collection -------------------------------------------------

    def _collect_work_items(
        self,
        *,
        typed_plan: TacticianSearchPlan,
        typed_goals: Mapping[str, ProgramLogicGoal],
        typed_hypotheses: Sequence[LogicHypothesis],
        permitted_subgoals: set[str],
        subgoal_by_id: Mapping[str, LogicSubgoal],
        premise_by_id: Mapping[str, ProgramLogicPremise],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        hyp_by_goal: dict[str, list[LogicHypothesis]] = {}
        for hypothesis in typed_hypotheses:
            if hypothesis.disposition not in _HYPOTHESIS_ADMITTED:
                continue
            if hypothesis.semantic_authority:
                raise UnauthorizedAxiomError(
                    "hypotheses cannot claim semantic authority during lowering"
                )
            hyp_by_goal.setdefault(hypothesis.target_goal_id, []).append(hypothesis)

        for subgoal in typed_plan.subgoals:
            if permitted_subgoals and subgoal.subgoal_id not in permitted_subgoals:
                continue
            goal = typed_goals[subgoal.goal_id]
            if goal.disposition not in _GOAL_LOWERABLE and goal.disposition not in {
                GoalDisposition.RESIDUAL,
                GoalDisposition.UNSUPPORTED,
            }:
                continue
            matched_facet = self._match_facet(goal, subgoal.claim_ref)
            clause = _clause_from_claim(subgoal.claim_ref, matched_facet)
            premise_ids = self._premise_ids_for(
                plan=typed_plan,
                hypothesis=None,
                subgoal=subgoal,
                premise_by_id=premise_by_id,
            )
            items.append(
                {
                    "clause": clause,
                    "facet": matched_facet,
                    "goal": goal,
                    "hypothesis": None,
                    "subgoal": subgoal,
                    "claim_ref": subgoal.claim_ref,
                    "premise_ids": premise_ids,
                }
            )

        # Hypotheses contribute additional facet clauses when they name
        # consequence / construction / placement / value refs.
        for goal_id, hyps in hyp_by_goal.items():
            goal = typed_goals[goal_id]
            for hypothesis in hyps:
                claim_refs = [
                    hypothesis.claimed_consequence_ref,
                    hypothesis.construction_ref,
                    hypothesis.placement_ref,
                    hypothesis.value_ref,
                ]
                claim_refs = [item for item in claim_refs if item]
                if not claim_refs:
                    claim_refs = [hypothesis.claimed_consequence_ref]
                for claim_ref in claim_refs:
                    matched_facet = self._match_facet(goal, claim_ref)
                    clause = _clause_from_claim(claim_ref, matched_facet)
                    premise_ids = self._premise_ids_for(
                        plan=typed_plan,
                        hypothesis=hypothesis,
                        subgoal=None,
                        premise_by_id=premise_by_id,
                    )
                    items.append(
                        {
                            "clause": clause,
                            "facet": matched_facet,
                            "goal": goal,
                            "hypothesis": hypothesis,
                            "subgoal": None,
                            "claim_ref": claim_ref,
                            "premise_ids": premise_ids,
                        }
                    )

        # Ensure every required (non-unsupported) facet appears at least once.
        covered_facets: dict[str, set[str]] = {
            goal_id: set() for goal_id in typed_plan.goal_ids
        }
        for item in items:
            facet = item["facet"]
            if facet is not None:
                covered_facets[item["goal"].goal_id].add(facet.facet_id)

        for goal_id in typed_plan.goal_ids:
            goal = typed_goals[goal_id]
            for facet in goal.required_facets:
                if facet.unsupported:
                    continue
                if facet.facet_id in covered_facets[goal_id]:
                    continue
                clause = _FACET_KIND_DEFAULT_CLAUSE[facet.kind]
                premise_ids = self._premise_ids_for(
                    plan=typed_plan,
                    hypothesis=None,
                    subgoal=None,
                    premise_by_id=premise_by_id,
                )
                items.append(
                    {
                        "clause": clause,
                        "facet": facet,
                        "goal": goal,
                        "hypothesis": None,
                        "subgoal": None,
                        "claim_ref": facet.contract_ref or facet.facet_id,
                        "premise_ids": premise_ids,
                    }
                )
                covered_facets[goal_id].add(facet.facet_id)

        # Deterministic order.
        items.sort(
            key=lambda item: (
                item["goal"].goal_id,
                item["clause"].value,
                item["facet"].facet_id if item["facet"] is not None else "",
                item["claim_ref"],
                item["hypothesis"].hypothesis_id if item["hypothesis"] is not None else "",
                item["subgoal"].subgoal_id if item["subgoal"] is not None else "",
            )
        )
        # De-duplicate identical (goal, clause, facet, claim) work items.
        deduped: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, ...]] = set()
        for item in items:
            key = (
                item["goal"].goal_id,
                item["clause"].value,
                item["facet"].facet_id if item["facet"] is not None else "",
                item["claim_ref"],
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)
        if len(deduped) > MAX_OBLIGATIONS + MAX_RESIDUALS:
            raise TacticianHammerObligationError("work-item inventory exceeds bounds")
        return deduped

    def _clauses_for_goal(
        self, goal: ProgramLogicGoal, *, claim_ref: str
    ) -> list[tuple[LoweringFacetKind, LogicFacetRef | None]]:
        results: list[tuple[LoweringFacetKind, LogicFacetRef | None]] = []
        if goal.required_facets:
            for facet in goal.required_facets:
                clause = _clause_from_claim(claim_ref or facet.facet_id, facet)
                results.append((clause, facet))
        else:
            results.append((_clause_from_claim(claim_ref, None), None))
        return results

    def _match_facet(
        self, goal: ProgramLogicGoal, claim_ref: str
    ) -> LogicFacetRef | None:
        if not claim_ref:
            return None
        for facet in (*goal.required_facets, *goal.unsupported_facets):
            if facet.facet_id == claim_ref:
                return facet
            if claim_ref == f"facet:{facet.facet_id}":
                return facet
            if facet.contract_ref and facet.contract_ref == claim_ref:
                return facet
            if facet.facet_id in claim_ref or (
                facet.contract_ref and facet.contract_ref in claim_ref
            ):
                return facet
        return None

    def _premise_ids_for(
        self,
        *,
        plan: TacticianSearchPlan,
        hypothesis: LogicHypothesis | None,
        subgoal: LogicSubgoal | None,
        premise_by_id: Mapping[str, ProgramLogicPremise],
    ) -> tuple[str, ...]:
        selected: list[str] = list(plan.selected_premise_ids)
        if hypothesis is not None:
            selected.extend(hypothesis.selected_premise_ids)
        # Subgoals do not carry premises directly; plan selection is authoritative.
        _ = subgoal
        ordered = _ids(selected, "premise_ids", required=True, preserve_order=True)
        unknown = [item for item in ordered if item not in premise_by_id]
        if unknown:
            raise TacticianHammerObligationError(
                f"selected premises missing from corpus: {unknown}"
            )
        return ordered

    # -- single-clause lowering ----------------------------------------------

    def _lower_clause(
        self,
        *,
        clause: LoweringFacetKind,
        facet: LogicFacetRef | None,
        goal: ProgramLogicGoal,
        hypothesis: LogicHypothesis | None,
        subgoal: LogicSubgoal | None,
        claim_ref: str,
        plan: TacticianSearchPlan,
        receipt: TacticianPlanGateReceipt,
        context: ObligationContext,
        roots: ProgramLogicAuthorityRoots,
        premise_by_id: Mapping[str, ProgramLogicPremise],
        premise_ids: Sequence[str],
    ) -> TacticianHammerProofObligation | LoweringResidual:
        # Explicit unsupported facets become residuals.
        if facet is not None and facet.unsupported:
            return self._residual(
                kind=ResidualSemanticKind.UNSUPPORTED_FACET,
                subject_id=facet.subject_symbol_id,
                reason_ref=f"unsupported-facet:{facet.facet_id}",
                goal_id=goal.goal_id,
                hypothesis_id=hypothesis.hypothesis_id if hypothesis else "",
                facet_id=facet.facet_id,
                claim_ref=claim_ref,
            )

        # Lifetime / concurrency / native residualization when translator omits.
        residual_kind = self._residual_for_clause(clause, context, facet, claim_ref)
        if residual_kind is not None:
            return self._residual(
                kind=residual_kind,
                subject_id=(
                    facet.subject_symbol_id
                    if facet is not None
                    else (goal.affected_symbol_ids[0] if goal.affected_symbol_ids else goal.goal_id)
                ),
                reason_ref=f"unsupported-semantic:{residual_kind.value}",
                goal_id=goal.goal_id,
                hypothesis_id=hypothesis.hypothesis_id if hypothesis else "",
                facet_id=facet.facet_id if facet is not None else "",
                claim_ref=claim_ref,
            )

        # Hypothesis unsupported flags may force residualization.
        if hypothesis is not None:
            for flag in hypothesis.unsupported_flags:
                token = flag.casefold().replace("-", "_")
                if token in _RESIDUAL_SEMANTICS and not context.capability.admits(token):
                    return self._residual(
                        kind=ResidualSemanticKind(token)
                        if token in {item.value for item in ResidualSemanticKind}
                        else ResidualSemanticKind.OMITTED_TRANSLATION,
                        subject_id=hypothesis.hypothesis_id,
                        reason_ref=f"hypothesis-unsupported:{flag}",
                        goal_id=goal.goal_id,
                        hypothesis_id=hypothesis.hypothesis_id,
                        facet_id=facet.facet_id if facet is not None else "",
                        claim_ref=claim_ref,
                    )

        premises = tuple(premise_by_id[item] for item in premise_ids)
        source_ids = self._source_ids(premises, goal, hypothesis, subgoal)
        source_authority = self._source_authority(premises, hypothesis, subgoal)

        subject_id = (
            facet.subject_symbol_id
            if facet is not None
            else (
                hypothesis.hypothesis_id
                if hypothesis is not None
                else (subgoal.subgoal_id if subgoal is not None else goal.goal_id)
            )
        )
        translation_map_id = (
            context.translation_map_id
            or (plan.translation_refs[0] if plan.translation_refs else "")
            or _digest(
                {
                    "translator_id": roots.translator_id,
                    "capability_revision": context.capability.capability_revision,
                    "facet": clause.value,
                },
                prefix="translation-map",
            )
        )
        hammer_selection = (
            context.hammer_premise_selection_id
            or _digest(
                {
                    "plan_id": plan.plan_id,
                    "premise_ids": list(premise_ids),
                    "method": "supervisor-explicit-premises@1",
                },
                prefix="hammer-premises",
            )
        )

        claim = IRClaim(
            predicate=f"lower:{clause.value}",
            subject_id=subject_id,
            facet_kind=clause,
            premise_ids=tuple(premise_ids),
            source_ids=source_ids,
            assumption_ids=context.assumption_ids,
            source_authority=source_authority,
            repository_id=roots.repository_id,
            tree_id=roots.tree_id,
            corpus_id=roots.corpus_id,
            translator_id=roots.translator_id,
            toolchain_id=roots.toolchain_id,
            policy_id=roots.policy_id,
            environment_id=roots.environment_id,
            capability_id=context.capability.capability_id,
            capability_revision=context.capability.capability_revision,
            goal_id=goal.goal_id,
            hypothesis_id=hypothesis.hypothesis_id if hypothesis else "",
            subgoal_id=subgoal.subgoal_id if subgoal else "",
            facet_id=facet.facet_id if facet is not None else "",
            plan_id=plan.plan_id,
            translation_map_id=translation_map_id,
            counterexample_target_ref=(
                hypothesis.counterexample_target_ref
                if hypothesis is not None and hypothesis.counterexample_target_ref
                else goal.counterexample_target_ref
            ),
        )

        scopes = self._ast_scopes(goal, hypothesis, facet)
        existing_refs = tuple(
            link.compilation_id for link in context.existing_obligation_links
        )
        envelope = CodeProofObligation(
            repository_id=roots.repository_id,
            repository_tree_id=roots.tree_id,
            ast_scope_ids=scopes,
            statement=f"{clause.value}:{claim.claim_id}",
            premise_ids=claim.premise_ids,
            template_id=f"tactician-hammer/{clause.value}",
            template_version="1",
            template_semantic_hash=claim.claim_id,
            invariant_class="tactician_hammer_logic",
            task_id="LPR-011",
            required_assurance=AssuranceLevel.KERNEL_VERIFIED,
            metadata={
                "claim_id": claim.claim_id,
                "facet_kind": clause.value,
                "goal_id": goal.goal_id,
                "plan_id": plan.plan_id,
                "gate_receipt_id": receipt.content_id,
                "corpus_id": roots.corpus_id,
                "translator_id": roots.translator_id,
                "toolchain_id": roots.toolchain_id,
                "policy_id": roots.policy_id,
                "environment_id": roots.environment_id,
                "capability_id": context.capability.capability_id,
                "capability_revision": context.capability.capability_revision,
                "supported_semantics": list(context.capability.supported_semantics),
                "translation_map_id": translation_map_id,
                "hammer_premise_selection_id": hammer_selection,
                "assumption_ids": list(context.assumption_ids),
                "source_authority": source_authority.value,
                "existing_obligation_refs": list(existing_refs),
                "producer_id": PRODUCER_ID,
            },
        )
        return TacticianHammerProofObligation(
            kind=clause,
            goal_id=goal.goal_id,
            claim=claim,
            code_obligation=envelope,
            premise_ids=tuple(premise_ids),
            source_ids=source_ids,
            hypothesis_id=hypothesis.hypothesis_id if hypothesis else "",
            subgoal_id=subgoal.subgoal_id if subgoal else "",
            facet_id=facet.facet_id if facet is not None else "",
            existing_obligation_refs=existing_refs,
            hammer_premise_selection_id=hammer_selection,
            translation_map_id=translation_map_id,
        )

    def _residual_for_clause(
        self,
        clause: LoweringFacetKind,
        context: ObligationContext,
        facet: LogicFacetRef | None,
        claim_ref: str,
    ) -> ResidualSemanticKind | None:
        surfaces = [claim_ref.casefold().replace("-", "_").replace(":", "_")]
        if facet is not None:
            surfaces.append(facet.facet_id.casefold().replace("-", "_").replace(":", "_"))
            surfaces.append(facet.kind.value.casefold())
            if facet.contract_ref:
                surfaces.append(
                    facet.contract_ref.casefold().replace("-", "_").replace(":", "_")
                )
        joined = " ".join(surfaces)
        tokens = set()
        for surface in surfaces:
            tokens.update(part for part in surface.split("_") if part)

        # Higher-order / dependent / dynamic / native / concurrency tokens.
        # Match both atomic tokens and multi-word surface substrings
        # (e.g. claim:higher_order-callback → higher_order).
        for token, kind in (
            ("higher_order", ResidualSemanticKind.HIGHER_ORDER),
            ("higherorder", ResidualSemanticKind.HIGHER_ORDER),
            ("dependent", ResidualSemanticKind.DEPENDENT),
            ("dynamic", ResidualSemanticKind.DYNAMIC),
            ("native", ResidualSemanticKind.NATIVE),
            ("concurrency", ResidualSemanticKind.CONCURRENCY),
            ("concurrent", ResidualSemanticKind.CONCURRENCY),
            ("reflection", ResidualSemanticKind.REFLECTION),
            ("ffi", ResidualSemanticKind.FFI),
            ("unsafe", ResidualSemanticKind.UNSAFE),
            ("lifetime", ResidualSemanticKind.LIFETIME),
        ):
            present = token in tokens or token in joined or token in surfaces
            # Also accept split "higher"+"order" as higher_order.
            if token == "higher_order" and {"higher", "order"} <= tokens:
                present = True
            if present and not context.capability.admits(kind.value):
                return kind

        if clause is LoweringFacetKind.LIFETIME and not context.capability.admits(
            "lifetime"
        ):
            return ResidualSemanticKind.LIFETIME
        if clause is LoweringFacetKind.OWNERSHIP and not (
            context.capability.admits("ownership")
            or context.capability.admits("memory")
            or context.capability.admits("ir")
        ):
            # Ownership without any memory/ownership/ir support is residual.
            if not (set(context.capability.supported_semantics) & _BASE_SUPPORTED_SEMANTICS):
                return ResidualSemanticKind.UNSUPPORTED_FACET
        return None

    def _residual(
        self,
        *,
        kind: ResidualSemanticKind,
        subject_id: str,
        reason_ref: str,
        goal_id: str = "",
        hypothesis_id: str = "",
        facet_id: str = "",
        claim_ref: str = "",
    ) -> LoweringResidual:
        residual_id = _digest(
            {
                "kind": kind.value,
                "subject_id": subject_id,
                "goal_id": goal_id,
                "facet_id": facet_id,
                "claim_ref": claim_ref,
                "reason_ref": reason_ref,
            },
            prefix="residual",
        )
        return LoweringResidual(
            residual_id=residual_id,
            kind=kind,
            subject_id=subject_id,
            reason_ref=reason_ref,
            goal_id=goal_id,
            hypothesis_id=hypothesis_id,
            facet_id=facet_id,
            claim_ref=claim_ref,
        )

    def _source_ids(
        self,
        premises: Sequence[ProgramLogicPremise],
        goal: ProgramLogicGoal,
        hypothesis: LogicHypothesis | None,
        subgoal: LogicSubgoal | None,
    ) -> tuple[str, ...]:
        ids: list[str] = [item.statement_ref for item in premises]
        ids.extend(goal.source_refs)
        if hypothesis is not None:
            ids.extend(hypothesis.evidence_refs)
        if subgoal is not None:
            ids.append(subgoal.claim_ref)
        return _ids(ids, "source_ids", required=True)

    def _source_authority(
        self,
        premises: Sequence[ProgramLogicPremise],
        hypothesis: LogicHypothesis | None,
        subgoal: LogicSubgoal | None,
    ) -> SourceAuthorityClass:
        if any(item.expectation_authority for item in premises):
            return SourceAuthorityClass.AUTHORITATIVE
        if subgoal is not None and subgoal.source_authority is SourceAuthorityClass.AUTHORITATIVE:
            return SourceAuthorityClass.AUTHORITATIVE
        if hypothesis is not None:
            if hypothesis.source_authority is SourceAuthorityClass.AUTHORITATIVE:
                return SourceAuthorityClass.AUTHORITATIVE
            return hypothesis.source_authority
        if subgoal is not None:
            return subgoal.source_authority
        return SourceAuthorityClass.CONFORMANCE

    def _ast_scopes(
        self,
        goal: ProgramLogicGoal,
        hypothesis: LogicHypothesis | None,
        facet: LogicFacetRef | None,
    ) -> tuple[str, ...]:
        scopes: list[str] = list(goal.affected_symbol_ids)
        if facet is not None:
            scopes.append(facet.subject_symbol_id)
        if hypothesis is not None:
            scopes.append(hypothesis.hypothesis_id)
        if not scopes:
            scopes.append(goal.goal_id)
        return _ids(scopes, "ast_scope_ids", required=True)

    # -- validation -----------------------------------------------------------

    def _validate_gate(
        self,
        receipt: TacticianPlanGateReceipt,
        plan: TacticianSearchPlan,
        corpus: ProgramLogicPremiseCorpus,
        roots: ProgramLogicAuthorityRoots,
        context: ObligationContext,
    ) -> None:
        if not receipt.may_lower_obligations:
            raise TacticianHammerObligationError(
                f"gate disposition {receipt.disposition.value} cannot lower obligations"
            )
        if receipt.semantic_authority:
            raise UnauthorizedAxiomError(
                "gate receipts cannot claim semantic authority"
            )
        if receipt.write_authority:
            raise TacticianHammerObligationError(
                "gate receipts cannot claim write authority during lowering"
            )
        if not _roots_equal(receipt.roots, roots):
            raise CrossRootPremiseError(
                "gate receipt roots must match current authority roots"
            )
        if not _roots_equal(plan.roots, roots):
            raise CrossRootPremiseError("plan roots must match current authority roots")
        if not _roots_equal(corpus.roots, roots):
            raise CrossRootPremiseError("corpus roots must match current authority roots")
        if plan.plan_id != receipt.plan_id:
            raise TacticianHammerObligationError(
                "plan_id must match the gate receipt plan_id"
            )
        if plan.content_id != receipt.plan_content_id:
            raise TacticianHammerObligationError(
                "plan content identity drifted from the gate receipt"
            )
        if corpus.content_id != receipt.corpus_content_id:
            raise TacticianHammerObligationError(
                "corpus content identity drifted from the gate receipt"
            )
        if roots.corpus_id != plan.roots.corpus_id:
            raise CrossRootPremiseError("plan corpus_id must match roots.corpus_id")
        if context.capability.translator_id != roots.translator_id:
            raise CrossRootPremiseError(
                "translator capability must bind roots.translator_id"
            )
        if plan.semantic_authority:
            raise UnauthorizedAxiomError(
                "tactician plans cannot claim semantic authority"
            )

    def _assert_no_cross_root_goals(
        self, goals: Sequence[ProgramLogicGoal], roots: ProgramLogicAuthorityRoots
    ) -> None:
        for goal in goals:
            if not _roots_equal(goal.roots, roots):
                raise CrossRootPremiseError(
                    f"goal {goal.goal_id} binds different authority roots"
                )

    def _assert_no_cross_root_hypotheses(
        self,
        hypotheses: Sequence[LogicHypothesis],
        roots: ProgramLogicAuthorityRoots,
    ) -> None:
        for hypothesis in hypotheses:
            if not _roots_equal(hypothesis.roots, roots):
                raise CrossRootPremiseError(
                    f"hypothesis {hypothesis.hypothesis_id} binds different authority roots"
                )

    def _assert_hypothesis_targets(
        self,
        hypotheses: Sequence[LogicHypothesis],
        goal_by_id: Mapping[str, ProgramLogicGoal],
        plan: TacticianSearchPlan,
    ) -> None:
        for hypothesis in hypotheses:
            if hypothesis.target_goal_id not in goal_by_id:
                raise TacticianHammerObligationError(
                    f"hypothesis target goal missing: {hypothesis.target_goal_id}"
                )
            if (
                hypothesis.disposition in _HYPOTHESIS_ADMITTED
                and hypothesis.target_goal_id not in plan.goal_ids
            ):
                raise TacticianHammerObligationError(
                    "admitted hypothesis target must be listed on the plan"
                )

    def _assert_assumption_consistency(
        self,
        context: ObligationContext,
        goals: Sequence[ProgramLogicGoal],
        hypotheses: Sequence[LogicHypothesis],
    ) -> None:
        context_ids = set(context.assumption_ids)
        for goal in goals:
            for ref in goal.assumption_refs:
                # Goal assumption refs may be a subset; unknown refs that look
                # like axioms are rejected.
                if any(marker in ref.casefold() for marker in _RETRIEVED_ASSUMPTION_MARKERS):
                    raise UnauthorizedAxiomError(
                        f"goal assumption ref is not reviewed evidence: {ref}"
                    )
                if goal.assumption_authority in {
                    SourceAuthorityClass.NOMINATING,
                    SourceAuthorityClass.DIAGNOSTIC,
                } and goal.disposition in {
                    GoalDisposition.ADMITTED,
                    GoalDisposition.DISCHARGED,
                }:
                    raise InconsistentAssumptionError(
                        "admitted goals cannot rest on nominating/diagnostic assumptions"
                    )
        # Context assumptions must be stable across the compilation.
        if not context_ids:
            raise InconsistentAssumptionError("empty assumption set")
        # Detect contradictory assumption pairs (same id, different evidence).
        by_id: dict[str, AssumptionBinding] = {}
        for item in context.assumptions:
            prior = by_id.get(item.assumption_id)
            if prior is not None and prior.evidence_ref != item.evidence_ref:
                raise InconsistentAssumptionError(
                    f"assumption {item.assumption_id} has conflicting evidence"
                )
            by_id[item.assumption_id] = item
        _ = hypotheses  # hypotheses do not author assumptions

    def _index_premises(
        self,
        corpus: ProgramLogicPremiseCorpus,
        roots: ProgramLogicAuthorityRoots,
    ) -> dict[str, ProgramLogicPremise]:
        index: dict[str, ProgramLogicPremise] = {}
        for premise in corpus.premises:
            if not _roots_equal(premise.roots, roots):
                raise CrossRootPremiseError(
                    f"premise {premise.premise_id} binds different authority roots"
                )
            if premise.premise_id in index:
                raise TacticianHammerObligationError(
                    f"duplicate premise identity in corpus: {premise.premise_id}"
                )
            # Nominating-only premises may appear but never as sole authority.
            index[premise.premise_id] = premise
        return index

    # -- decoding -------------------------------------------------------------

    @staticmethod
    def _decode_receipt(
        value: TacticianPlanGateReceipt | Mapping[str, Any],
    ) -> TacticianPlanGateReceipt:
        if isinstance(value, TacticianPlanGateReceipt):
            return value
        if isinstance(value, Mapping):
            return TacticianPlanGateReceipt.from_dict(value)
        raise TacticianHammerObligationError(
            "gate_receipt must be TacticianPlanGateReceipt"
        )

    @staticmethod
    def _decode_plan(
        value: TacticianSearchPlan | Mapping[str, Any],
    ) -> TacticianSearchPlan:
        if isinstance(value, TacticianSearchPlan):
            return value
        if isinstance(value, Mapping):
            return (
                TacticianSearchPlan.from_dict(value)
                if "schema" in value
                else TacticianSearchPlan(**value)
            )
        raise TacticianHammerObligationError("plan must be TacticianSearchPlan")

    @staticmethod
    def _decode_goals(
        values: Sequence[ProgramLogicGoal | Mapping[str, Any]],
    ) -> tuple[ProgramLogicGoal, ...]:
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(
            values, Sequence
        ):
            raise TacticianHammerObligationError("goals must be a sequence")
        if len(values) > MAX_GOALS:
            raise TacticianHammerObligationError("goals exceed bound")
        result: list[ProgramLogicGoal] = []
        for item in values:
            if isinstance(item, ProgramLogicGoal):
                result.append(item)
            elif isinstance(item, Mapping):
                result.append(
                    ProgramLogicGoal.from_dict(item)
                    if "schema" in item
                    else ProgramLogicGoal(**item)
                )
            else:
                raise TacticianHammerObligationError(
                    "goals must contain ProgramLogicGoal values"
                )
        return tuple(result)

    @staticmethod
    def _decode_hypotheses(
        values: Sequence[LogicHypothesis | Mapping[str, Any]],
    ) -> tuple[LogicHypothesis, ...]:
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(
            values, Sequence
        ):
            raise TacticianHammerObligationError("hypotheses must be a sequence")
        if len(values) > MAX_HYPOTHESES:
            raise TacticianHammerObligationError("hypotheses exceed bound")
        result: list[LogicHypothesis] = []
        for item in values:
            if isinstance(item, LogicHypothesis):
                result.append(item)
            elif isinstance(item, Mapping):
                result.append(
                    LogicHypothesis.from_dict(item)
                    if "schema" in item
                    else LogicHypothesis(**item)
                )
            else:
                raise TacticianHammerObligationError(
                    "hypotheses must contain LogicHypothesis values"
                )
        return tuple(result)

    @staticmethod
    def _decode_corpus(
        value: ProgramLogicPremiseCorpus | Mapping[str, Any],
    ) -> ProgramLogicPremiseCorpus:
        if isinstance(value, ProgramLogicPremiseCorpus):
            return value
        if isinstance(value, Mapping):
            return (
                ProgramLogicPremiseCorpus.from_dict(value)
                if "schema" in value
                else ProgramLogicPremiseCorpus(**value)
            )
        raise TacticianHammerObligationError(
            "corpus must be ProgramLogicPremiseCorpus"
        )


# ---------------------------------------------------------------------------
# Public factory helpers
# ---------------------------------------------------------------------------


def lower_tactician_plan(
    gate_receipt: TacticianPlanGateReceipt | Mapping[str, Any],
    plan: TacticianSearchPlan | Mapping[str, Any],
    goals: Sequence[ProgramLogicGoal | Mapping[str, Any]],
    hypotheses: Sequence[LogicHypothesis | Mapping[str, Any]],
    corpus: ProgramLogicPremiseCorpus | Mapping[str, Any],
    context: ObligationContext,
    *,
    current_roots: ProgramLogicAuthorityRoots | Mapping[str, Any] | None = None,
) -> TacticianHammerObligationCompilation:
    """Module-level entry point for admitted-plan obligation lowering."""

    return TacticianHammerObligationCompiler().compile(
        gate_receipt,
        plan,
        goals,
        hypotheses,
        corpus,
        context,
        current_roots=current_roots,
    )


__all__ = [
    "AssumptionBinding",
    "ChangedAssumptionsError",
    "CrossRootPremiseError",
    "ExistingObligationLink",
    "GoalSnapshotBinding",
    "IRClaim",
    "InconsistentAssumptionError",
    "LoweringDisposition",
    "LoweringFacetKind",
    "LoweringResidual",
    "MAX_OBLIGATIONS",
    "NativeGoalBindingError",
    "NativeITPKind",
    "NativeTheoremSource",
    "ObligationContext",
    "OmittedFacetError",
    "PROGRAM_LOGIC_NATIVE_GOAL_COMPILER_ID",
    "ProgramLogicNativeGoalCompiler",
    "ResidualSemanticKind",
    "SourceDriftError",
    "TACTICIAN_HAMMER_OBLIGATION_COMPILER_ID",
    "TACTICIAN_HAMMER_OBLIGATIONS_INTERFACE",
    "TacticianHammerObligationCompilation",
    "TacticianHammerObligationCompiler",
    "TacticianHammerObligationError",
    "TacticianHammerProofObligation",
    "TranslatorCapabilityBinding",
    "UnauthorizedAxiomError",
    "WrongTheoremError",
    "lower_tactician_plan",
]
