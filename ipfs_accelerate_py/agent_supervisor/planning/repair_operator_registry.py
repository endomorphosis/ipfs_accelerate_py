"""Reviewed, capability-declared repair operator registry (PDR-050).

Interfaces:

* ``RepairOperatorRegistry@1``
* ``DoctorRepairOperatorSpec@2``

The registry is an immutable catalogue of *proposal grammars*.  It does not
hold renderer callables, source bodies, proof verdicts, permits, or mutation
handles.  Lookup therefore cannot grant semantic, proof, or write authority.
Actual rendering of the already-supported analytical transforms remains in
``deterministic_doctor_transforms`` behind its existing proof and path gates.

The v2 catalogue covers every current ``TransformKind`` plus exact move,
tracked-artifact restoration, and reviewed semantic-patch/equality-rewrite
hooks.  Resolution is fail-closed: target, value, or placement ambiguity is a
rejection; missing evidence/capability is an abstention; and dynamic,
generated, stateful, native, public-API, or dependency-changing work is
reported as approval-required without treating an approval reference as
proof.

Each reviewed grammar also declares the counterexample, unsat-core, failed-
assumption, and validated-interpolant predicates that may refine bounded
search, plus the effect/security restrictions that every candidate must
obey: no undeclared imports, dependencies, files, authority, effects, or
behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.change_propagation_contracts import TransformKind
from ..analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
    DoctorOperatorKind,
)
from ..proof.formal_verification_contracts import CanonicalContract, content_identity
from .deterministic_doctor_transforms import (
    ANALYTICAL_TRANSFORM_OPERATOR_BINDINGS,
    DoctorRepairOperatorRegistry as LegacyDoctorRepairOperatorRegistry,
    build_default_doctor_operator_registry,
)


REPAIR_OPERATOR_REGISTRY_INTERFACE: Final[str] = "RepairOperatorRegistry@1"
DOCTOR_REPAIR_OPERATOR_SPEC_INTERFACE: Final[str] = "DoctorRepairOperatorSpec@2"
REPAIR_OPERATOR_REGISTRY_VERSION: Final[int] = 1
DOCTOR_REPAIR_OPERATOR_SPEC_VERSION: Final[int] = 2
REPAIR_OPERATOR_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-operator-registry@1"
)
DOCTOR_REPAIR_OPERATOR_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-repair-operator-spec@2"
)
REPAIR_OPERATOR_LOOKUP_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-operator-lookup-request@1"
)
REPAIR_OPERATOR_LOOKUP_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-operator-lookup-result@1"
)
REPAIR_OPERATOR_REGISTRY_PRODUCER: Final[str] = "repair-operator-registry@1"

MAX_OPERATOR_COUNT: Final[int] = 64
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024


class RepairOperatorRegistryError(ValueError):
    """Malformed registry declaration or lookup input."""


class RepairOperatorRegistryAuthorityError(RepairOperatorRegistryError):
    """An operator declaration attempted to carry forbidden authority."""


class UnknownRepairOperatorError(RepairOperatorRegistryError):
    """A caller requested a kind outside the reviewed closed catalogue."""


class RepairOperatorKind(str, Enum):
    """Canonical v2 repair operator kinds.

    Aliases preserve the terminology used by diagnoses while serialization
    always emits the canonical value.
    """

    EXACT_RENAME = "exact_rename"
    EXACT_MOVE = "exact_move"
    ADD_ARGUMENT = "add_argument"
    MISSING_ARGUMENT = "add_argument"
    RENAME_ARGUMENT = "rename_argument"
    REORDER_ARGUMENT = "reorder_argument"
    THREAD_ARGUMENT = "thread_argument"
    VALUE_THREADING = "thread_argument"
    ADD_IMPORT = "add_import"
    EXACT_IMPORT = "add_import"
    ADD_EXPORT = "add_export"
    EXACT_EXPORT = "add_export"
    ADD_REGISTRATION = "add_registration"
    EXACT_REGISTRATION = "add_registration"
    ADD_CONSTRUCTOR_ROUTE = "add_constructor_route"
    CONSTRUCTOR = "add_constructor_route"
    ADD_FACTORY_ROUTE = "add_factory_route"
    FACTORY = "add_factory_route"
    FINITE_ADAPTER = "finite_adapter"
    ADAPTER = "finite_adapter"
    SCHEMA_PROJECTION = "schema_projection"
    SCHEMA = "schema_projection"
    SERIALIZER_UPDATE = "serializer_update"
    SERIALIZER = "serializer_update"
    FIXTURE_UPDATE = "fixture_update"
    FIXTURE = "fixture_update"
    MANIFEST_UPDATE = "manifest_update"
    MANIFEST = "manifest_update"
    RESTORE_TRACKED_ARTIFACT = "restore_tracked_artifact"
    ARTIFACT = "restore_tracked_artifact"
    SEMANTIC_PATCH = "semantic_patch"
    EQUALITY_REWRITE = "equality_rewrite"


class RepairOperatorFamily(str, Enum):
    SYMBOL = "symbol"
    MOVE = "move"
    CALL = "call"
    WIRING = "wiring"
    CONSTRUCTION = "construction"
    DATA_CONTRACT = "data_contract"
    ARTIFACT = "artifact"
    REVIEWED_REWRITE = "reviewed_rewrite"


class OperatorValueRequirement(str, Enum):
    NONE = "none"
    UNIQUE_PROVED = "unique_proved"
    TOTAL_MAPPING = "total_mapping"
    VERIFIED_PREIMAGE = "verified_preimage"
    REVIEWED_RULE = "reviewed_rule"


class ReviewedRepairHook(str, Enum):
    NONE = "none"
    EXACT_MOVE = "exact_move"
    ARTIFACT_RESTORE = "artifact_restore"
    SEMANTIC_PATCH = "semantic_patch"
    EQUALITY_REWRITE = "equality_rewrite"


class RepairOperatorCapability(str, Enum):
    """Capabilities are requirements, never authority tokens."""

    EXACT_TARGET = "exact_target"
    EXACT_PLACEMENT = "exact_placement"
    CLOSED_AST = "closed_ast"
    IDEMPOTENT_RENDER = "idempotent_render"
    SCOPE_BOUND = "scope_bound"
    PROPOSAL_ONLY = "proposal_only"
    SYMBOL_EQUIVALENCE = "symbol_equivalence"
    FILE_MOVE = "file_move"
    UNIQUE_VALUE = "unique_value"
    ROUTE_CLOSURE = "route_closure"
    IMPORT_WIRING = "import_wiring"
    EXPORT_WIRING = "export_wiring"
    REGISTRATION_WIRING = "registration_wiring"
    CONSTRUCTOR_WIRING = "constructor_wiring"
    FACTORY_WIRING = "factory_wiring"
    FINITE_ADAPTER = "finite_adapter"
    TOTAL_FIELD_MAPPING = "total_field_mapping"
    SERIALIZER_MAPPING = "serializer_mapping"
    FIXTURE_MAPPING = "fixture_mapping"
    MANIFEST_MAPPING = "manifest_mapping"
    VERIFIED_ARTIFACT = "verified_artifact"
    REVIEWED_SEMANTIC_PATCH = "reviewed_semantic_patch"
    DECLARED_EQUALITY_THEORY = "declared_equality_theory"


class RepairBehaviorClass(str, Enum):
    PURE_LOCAL = "pure_local"
    UNKNOWN = "unknown"
    DYNAMIC = "dynamic"
    GENERATED = "generated"
    STATEFUL = "stateful"
    NATIVE = "native"
    PUBLIC_API = "public_api"
    DEPENDENCY_CHANGING = "dependency_changing"


class RepairCounterevidenceClass(str, Enum):
    """Evidence families that may refine reviewed-operator search."""

    COUNTEREXAMPLE = "counterexample"
    UNSAT_CORE = "unsat_core"
    FAILED_ASSUMPTION = "failed_assumption"
    VALIDATED_INTERPOLANT = "validated_interpolant"


class RepairEffectRestriction(str, Enum):
    """Frame restrictions every reviewed candidate must preserve."""

    NO_UNDECLARED_IMPORTS = "no_undeclared_imports"
    NO_UNDECLARED_DEPENDENCIES = "no_undeclared_dependencies"
    NO_UNDECLARED_FILES = "no_undeclared_files"
    NO_AUTHORITY = "no_authority"
    NO_UNDECLARED_EFFECTS = "no_undeclared_effects"
    NO_UNDECLARED_BEHAVIOR = "no_undeclared_behavior"


class RepairOperatorLookupDisposition(str, Enum):
    """Resolution outcome; ``PROPOSAL_ELIGIBLE`` is not admission."""

    PROPOSAL_ELIGIBLE = "proposal_eligible"
    APPROVAL_REQUIRED = "approval_required"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class RepairOperatorLookupReason(str, Enum):
    UNKNOWN_OPERATOR = "unknown_operator"
    UNKNOWN_BEHAVIOR = "unknown_behavior"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    UNSUPPORTED_AST_SHAPE = "unsupported_ast_shape"
    TARGET_MISSING = "target_missing"
    TARGET_AMBIGUOUS = "target_ambiguous"
    VALUE_MISSING = "value_missing"
    VALUE_AMBIGUOUS = "value_ambiguous"
    PLACEMENT_MISSING = "placement_missing"
    PLACEMENT_AMBIGUOUS = "placement_ambiguous"
    SCOPE_MISSING = "scope_missing"
    SCOPE_ESCAPE = "scope_escape"
    CAPABILITY_MISSING = "capability_missing"
    PROOF_REFERENCE_MISSING = "proof_reference_missing"
    REVIEW_REFERENCE_MISSING = "review_reference_missing"
    DYNAMIC_APPROVAL = "dynamic_behavior_requires_approval"
    GENERATED_APPROVAL = "generated_behavior_requires_approval"
    STATEFUL_APPROVAL = "stateful_behavior_requires_approval"
    NATIVE_APPROVAL = "native_behavior_requires_approval"
    PUBLIC_API_APPROVAL = "public_api_change_requires_approval"
    DEPENDENCY_APPROVAL = "dependency_change_requires_approval"
    CANDIDATE_ONLY = "candidate_only"
    COUNTEREVIDENCE_MISMATCH = "counterevidence_does_not_select_operator"
    UNDECLARED_EFFECT = "undeclared_effect_or_behavior"
    UNDECLARED_IMPORT = "undeclared_import"
    UNDECLARED_DEPENDENCY = "undeclared_dependency"
    UNDECLARED_FILE = "undeclared_file"
    UNVALIDATED_INTERPOLANT = "interpolant_not_independently_validated"


_APPROVAL_BEHAVIORS: Final[Mapping[str, RepairOperatorLookupReason]] = MappingProxyType(
    {
        RepairBehaviorClass.DYNAMIC.value: RepairOperatorLookupReason.DYNAMIC_APPROVAL,
        RepairBehaviorClass.GENERATED.value: RepairOperatorLookupReason.GENERATED_APPROVAL,
        RepairBehaviorClass.STATEFUL.value: RepairOperatorLookupReason.STATEFUL_APPROVAL,
        RepairBehaviorClass.NATIVE.value: RepairOperatorLookupReason.NATIVE_APPROVAL,
        RepairBehaviorClass.PUBLIC_API.value: RepairOperatorLookupReason.PUBLIC_API_APPROVAL,
        RepairBehaviorClass.DEPENDENCY_CHANGING.value: (
            RepairOperatorLookupReason.DEPENDENCY_APPROVAL
        ),
    }
)

_KIND_ALIASES: Final[Mapping[str, RepairOperatorKind]] = MappingProxyType(
    {
        "rename": RepairOperatorKind.EXACT_RENAME,
        "move": RepairOperatorKind.EXACT_MOVE,
        "missing_argument": RepairOperatorKind.ADD_ARGUMENT,
        "value_threading": RepairOperatorKind.THREAD_ARGUMENT,
        "thread_value": RepairOperatorKind.THREAD_ARGUMENT,
        "exact_import": RepairOperatorKind.ADD_IMPORT,
        "import": RepairOperatorKind.ADD_IMPORT,
        "exact_export": RepairOperatorKind.ADD_EXPORT,
        "export": RepairOperatorKind.ADD_EXPORT,
        "exact_registration": RepairOperatorKind.ADD_REGISTRATION,
        "registration": RepairOperatorKind.ADD_REGISTRATION,
        "constructor": RepairOperatorKind.ADD_CONSTRUCTOR_ROUTE,
        "factory": RepairOperatorKind.ADD_FACTORY_ROUTE,
        "adapter": RepairOperatorKind.FINITE_ADAPTER,
        "schema": RepairOperatorKind.SCHEMA_PROJECTION,
        "serializer": RepairOperatorKind.SERIALIZER_UPDATE,
        "fixture": RepairOperatorKind.FIXTURE_UPDATE,
        "manifest": RepairOperatorKind.MANIFEST_UPDATE,
        "artifact": RepairOperatorKind.RESTORE_TRACKED_ARTIFACT,
        "equality": RepairOperatorKind.EQUALITY_REWRITE,
    }
)

CANONICAL_EFFECT_RESTRICTIONS: Final[tuple[str, ...]] = tuple(
    item.value for item in RepairEffectRestriction
)
CANONICAL_COUNTEREVIDENCE_CLASSES: Final[tuple[str, ...]] = tuple(
    item.value for item in RepairCounterevidenceClass
)

# Repair-class strings match FormalCounterexample.RepairClass values.
_REPAIR_CLASS_ADD_DEPENDENCY: Final[str] = "add_or_correct_dependency"
_REPAIR_CLASS_SPLIT_TASK: Final[str] = "split_non_atomic_task"
_REPAIR_CLASS_TIGHTEN_AUTHORITY: Final[str] = "tighten_authority_or_fencing"
_REPAIR_CLASS_ADD_OBLIGATION: Final[str] = "add_obligation_or_fallback_test"
_REPAIR_CLASS_CONSTRAIN_SCOPE: Final[str] = "constrain_ast_scope_or_model_bound"
_REPAIR_CLASS_ADD_PREMISE: Final[str] = "add_premise_or_evidence_dependency"
_REPAIR_CLASS_ADJUST_RESOURCES: Final[str] = "adjust_portfolio_or_resource_bound"
_REPAIR_CLASS_HUMAN_REVIEW: Final[str] = "request_scoped_human_review"

_OPERATOR_COUNTEREVIDENCE_GRAMMAR: Final[
    Mapping[RepairOperatorKind, tuple[tuple[str, ...], tuple[str, ...]]]
] = MappingProxyType(
    {
        RepairOperatorKind.EXACT_RENAME: (
            (_REPAIR_CLASS_CONSTRAIN_SCOPE,),
            ("rename", "symbol", "identifier", "name"),
        ),
        RepairOperatorKind.EXACT_MOVE: (
            (_REPAIR_CLASS_CONSTRAIN_SCOPE,),
            ("move", "path", "file_move", "relocation"),
        ),
        RepairOperatorKind.ADD_ARGUMENT: (
            (_REPAIR_CLASS_ADD_PREMISE,),
            (
                "missing_argument",
                "arity",
                "argument",
                "call",
                "parameter",
                "add_argument",
            ),
        ),
        RepairOperatorKind.RENAME_ARGUMENT: (
            (_REPAIR_CLASS_ADD_PREMISE, _REPAIR_CLASS_CONSTRAIN_SCOPE),
            ("rename_argument", "keyword", "parameter_name", "argument"),
        ),
        RepairOperatorKind.REORDER_ARGUMENT: (
            (_REPAIR_CLASS_ADD_PREMISE,),
            ("reorder_argument", "argument_order", "positional", "arity"),
        ),
        RepairOperatorKind.THREAD_ARGUMENT: (
            (_REPAIR_CLASS_ADD_PREMISE,),
            ("thread_argument", "value_threading", "missing_context", "parameter"),
        ),
        RepairOperatorKind.ADD_IMPORT: (
            (_REPAIR_CLASS_ADD_DEPENDENCY,),
            ("import", "missing_symbol", "module", "unresolved_name"),
        ),
        RepairOperatorKind.ADD_EXPORT: (
            (_REPAIR_CLASS_ADD_DEPENDENCY,),
            ("export", "dunder_all", "public_name"),
        ),
        RepairOperatorKind.ADD_REGISTRATION: (
            (_REPAIR_CLASS_ADD_DEPENDENCY,),
            ("registration", "registry", "plugin"),
        ),
        RepairOperatorKind.ADD_CONSTRUCTOR_ROUTE: (
            (_REPAIR_CLASS_ADD_PREMISE,),
            ("constructor", "instantiation", "init"),
        ),
        RepairOperatorKind.ADD_FACTORY_ROUTE: (
            (_REPAIR_CLASS_ADD_PREMISE,),
            ("factory", "constructor_route"),
        ),
        RepairOperatorKind.FINITE_ADAPTER: (
            (_REPAIR_CLASS_ADD_PREMISE, _REPAIR_CLASS_CONSTRAIN_SCOPE),
            ("adapter", "wrap", "mapping_adapter"),
        ),
        RepairOperatorKind.SCHEMA_PROJECTION: (
            (_REPAIR_CLASS_CONSTRAIN_SCOPE,),
            ("schema", "field", "projection", "contract_field"),
        ),
        RepairOperatorKind.SERIALIZER_UPDATE: (
            (_REPAIR_CLASS_CONSTRAIN_SCOPE,),
            ("serializer", "codec", "encode"),
        ),
        RepairOperatorKind.FIXTURE_UPDATE: (
            (_REPAIR_CLASS_ADD_OBLIGATION,),
            ("fixture", "test_data", "oracle"),
        ),
        RepairOperatorKind.MANIFEST_UPDATE: (
            (_REPAIR_CLASS_ADJUST_RESOURCES, _REPAIR_CLASS_ADD_OBLIGATION),
            ("manifest", "generated_manifest", "lockfile"),
        ),
        RepairOperatorKind.RESTORE_TRACKED_ARTIFACT: (
            (_REPAIR_CLASS_ADD_DEPENDENCY,),
            ("artifact", "cid", "preimage", "tracked_blob"),
        ),
        RepairOperatorKind.SEMANTIC_PATCH: (
            (_REPAIR_CLASS_SPLIT_TASK, _REPAIR_CLASS_CONSTRAIN_SCOPE),
            ("semantic_patch", "reviewed_patch", "pattern"),
        ),
        RepairOperatorKind.EQUALITY_REWRITE: (
            (_REPAIR_CLASS_ADD_PREMISE,),
            ("equality", "rewrite", "equivalent", "egraph"),
        ),
    }
)


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise RepairOperatorRegistryError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise RepairOperatorRegistryError(f"{name} must not be empty")
    if "\x00" in result:
        raise RepairOperatorRegistryError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > limit:
        raise RepairOperatorRegistryError(f"{name} exceeds its byte bound")
    return result


def _optional_text(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _text(value, name)


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise RepairOperatorRegistryError(f"{name} must be a sequence")
    else:
        raw = values
    result: list[str] = []
    for item in raw:
        value = _text(item, name)
        if value not in result:
            result.append(value)
    if required and not result:
        raise RepairOperatorRegistryError(f"{name} must not be empty")
    if len(result) > MAX_REFERENCE_COUNT:
        raise RepairOperatorRegistryError(f"{name} exceeds its item bound")
    return tuple(result if preserve_order else sorted(result))


def _paths(values: Any, name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in _ids(values, name, preserve_order=True):
        text = raw.replace("\\", "/")
        if len(text.encode("utf-8")) > MAX_PATH_BYTES:
            raise RepairOperatorRegistryError(f"{name} exceeds its path byte bound")
        path = PurePosixPath(text)
        if path.is_absolute() or ".." in path.parts or text in {"", "."}:
            raise RepairOperatorRegistryError(
                f"{name} must contain relative repository paths without escape"
            )
        value = path.as_posix()
        if value not in normalized:
            normalized.append(value)
    return tuple(sorted(normalized))


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RepairOperatorRegistryError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise RepairOperatorRegistryError(f"{name} has an unsupported value") from exc


def normalize_repair_operator_kind(value: Any) -> RepairOperatorKind:
    """Normalize a canonical kind, operator id, enum alias, or reviewed alias."""

    if isinstance(value, RepairOperatorKind):
        return value
    raw = str(getattr(value, "value", value)).strip().lower().replace("-", "_")
    if raw.startswith("repair_operator:"):
        raw = raw.split(":", 1)[1].split("@", 1)[0]
    elif raw.startswith("operator:"):
        raw = raw.split(":", 1)[1]
    if raw in _KIND_ALIASES:
        return _KIND_ALIASES[raw]
    try:
        return RepairOperatorKind(raw)
    except ValueError as exc:
        raise UnknownRepairOperatorError(
            RepairOperatorLookupReason.UNKNOWN_OPERATOR.value
        ) from exc


def default_operator_effect_restrictions(
    kind: RepairOperatorKind | str | None = None,
) -> tuple[str, ...]:
    """Every reviewed operator forbids undeclared imports/files/effects."""

    del kind
    return CANONICAL_EFFECT_RESTRICTIONS


def default_operator_counterevidence_grammar(
    kind: RepairOperatorKind | str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(repair_classes, predicates)`` for one reviewed operator."""

    normalized = normalize_repair_operator_kind(kind)
    return _OPERATOR_COUNTEREVIDENCE_GRAMMAR.get(normalized, ((), ()))


def _tokenize_counterevidence(*values: Any) -> frozenset[str]:
    tokens: set[str] = set()
    for value in values:
        if value in (None, "", (), [], {}):
            continue
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, Mapping):
            for key, item in value.items():
                tokens.update(_tokenize_counterevidence(key, item))
            continue
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray, memoryview)
        ):
            for item in value:
                tokens.update(_tokenize_counterevidence(item))
            continue
        text = str(value).strip().lower().replace("-", "_")
        if not text:
            continue
        tokens.add(text)
        # Keep prefixed identifiers intact so cores/assumptions such as
        # ``clause:missing_argument`` still contribute ``missing_argument``.
        if ":" in text:
            suffix = text.rsplit(":", 1)[-1].strip()
            if suffix:
                tokens.add(suffix)
        for piece in text.replace(":", "_").replace("/", "_").replace(".", "_").split(
            "_"
        ):
            if len(piece) >= 3:
                tokens.add(piece)
    return frozenset(tokens)


def _evidence_matches_predicate(predicate: str, evidence: frozenset[str]) -> bool:
    """Exact predicate match; prefixed ids may contribute their suffix."""

    if not predicate:
        return False
    if predicate in evidence:
        return True
    colon_suffix = ":" + predicate
    return any(token.endswith(colon_suffix) for token in evidence)


def score_operator_counterevidence(
    kind: RepairOperatorKind | str,
    *,
    repair_classes: Sequence[str] = (),
    predicates: Sequence[str] = (),
) -> int:
    """Return a deterministic match score; ``0`` means no positive match.

    Matching is exact on reviewed repair-class names and grammar predicates.
    Short token substrings such as ``add`` or ``model`` must not select an
    unrelated operator.
    """

    try:
        normalized = normalize_repair_operator_kind(kind)
    except UnknownRepairOperatorError:
        return 0
    classes, grammar_predicates = default_operator_counterevidence_grammar(normalized)
    evidence = _tokenize_counterevidence(repair_classes, predicates)
    if not evidence:
        return 0
    score = 0
    for item in classes:
        if item in evidence:
            score += 4
    for predicate in grammar_predicates:
        if _evidence_matches_predicate(predicate, evidence):
            score += 3
    return score


@dataclass(frozen=True)
class DoctorRepairOperatorSpec(CanonicalContract):
    """Canonical v2 proposal grammar for one reviewed repair operator."""

    SCHEMA: ClassVar[str] = DOCTOR_REPAIR_OPERATOR_SPEC_SCHEMA
    INTERFACE: ClassVar[str] = DOCTOR_REPAIR_OPERATOR_SPEC_INTERFACE

    operator_id: str
    kind: RepairOperatorKind
    family: RepairOperatorFamily
    aliases: tuple[str, ...]
    supported_languages: tuple[str, ...]
    supported_ast_shapes: tuple[str, ...]
    capability_refs: tuple[str, ...]
    precondition_refs: tuple[str, ...]
    postcondition_refs: tuple[str, ...]
    frame_condition_refs: tuple[str, ...]
    proof_requirement_refs: tuple[str, ...]
    validation_requirement_refs: tuple[str, ...]
    scope_constraints: tuple[str, ...]
    approval_classes: tuple[str, ...]
    abstain_classes: tuple[str, ...]
    value_requirement: OperatorValueRequirement = OperatorValueRequirement.NONE
    placement_required: bool = True
    analytical_transform_kind: str = ""
    reviewed_hook: ReviewedRepairHook = ReviewedRepairHook.NONE
    review_requirement_refs: tuple[str, ...] = ()
    renderer_id: str = ""
    idempotent: bool = True
    inverse_or_compensation_ref: str = ""
    proposal_only: bool = True
    semantic_authority: bool = False
    grants_proof_authority: bool = False
    grants_write_authority: bool = False
    counterevidence_classes: tuple[str, ...] = ()
    addresses_repair_classes: tuple[str, ...] = ()
    counterevidence_predicates: tuple[str, ...] = ()
    effect_restrictions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        kind = normalize_repair_operator_kind(self.kind)
        object.__setattr__(self, "kind", kind)
        family = _enum(self.family, RepairOperatorFamily, "family")
        object.__setattr__(self, "family", family)
        expected_id = f"repair-operator:{kind.value}@2"
        operator_id = _text(self.operator_id, "operator_id")
        if operator_id != expected_id:
            raise RepairOperatorRegistryError(
                f"operator_id must be canonical for {kind.value}: {expected_id}"
            )
        object.__setattr__(self, "operator_id", operator_id)
        for name, required in (
            ("aliases", False),
            ("supported_languages", True),
            ("supported_ast_shapes", True),
            ("capability_refs", True),
            ("precondition_refs", True),
            ("postcondition_refs", True),
            ("frame_condition_refs", True),
            ("proof_requirement_refs", True),
            ("validation_requirement_refs", True),
            ("scope_constraints", True),
            ("approval_classes", True),
            ("abstain_classes", True),
            ("review_requirement_refs", False),
            ("counterevidence_classes", False),
            ("addresses_repair_classes", False),
            ("counterevidence_predicates", False),
            ("effect_restrictions", False),
        ):
            object.__setattr__(
                self,
                name,
                _ids(getattr(self, name), name, required=required),
            )
        value_requirement = _enum(
            self.value_requirement,
            OperatorValueRequirement,
            "value_requirement",
        )
        object.__setattr__(self, "value_requirement", value_requirement)
        object.__setattr__(
            self,
            "placement_required",
            _bool(self.placement_required, "placement_required"),
        )
        analytical = _optional_text(
            self.analytical_transform_kind,
            "analytical_transform_kind",
        )
        if analytical:
            try:
                TransformKind(analytical)
            except ValueError as exc:
                raise RepairOperatorRegistryError(
                    "analytical_transform_kind is not an existing TransformKind"
                ) from exc
        object.__setattr__(self, "analytical_transform_kind", analytical)
        hook = _enum(self.reviewed_hook, ReviewedRepairHook, "reviewed_hook")
        object.__setattr__(self, "reviewed_hook", hook)
        if analytical and hook is not ReviewedRepairHook.NONE:
            raise RepairOperatorRegistryError(
                "operator cannot combine an analytical renderer with a reviewed hook"
            )
        if not analytical and hook is ReviewedRepairHook.NONE:
            raise RepairOperatorRegistryError(
                "operator must declare an analytical transform or reviewed hook"
            )
        if hook in {
            ReviewedRepairHook.EXACT_MOVE,
            ReviewedRepairHook.SEMANTIC_PATCH,
            ReviewedRepairHook.EQUALITY_REWRITE,
        } and not self.review_requirement_refs:
            raise RepairOperatorRegistryError(
                "reviewed hooks must declare review requirements"
            )
        renderer = _text(self.renderer_id, "renderer_id")
        object.__setattr__(self, "renderer_id", renderer)
        inverse = _text(
            self.inverse_or_compensation_ref,
            "inverse_or_compensation_ref",
        )
        object.__setattr__(self, "inverse_or_compensation_ref", inverse)
        if not _bool(self.idempotent, "idempotent"):
            raise RepairOperatorRegistryError("registered operators must be idempotent")
        object.__setattr__(self, "idempotent", True)
        if not _bool(self.proposal_only, "proposal_only"):
            raise RepairOperatorRegistryAuthorityError(
                "repair operators must remain proposal-only"
            )
        object.__setattr__(self, "proposal_only", True)
        authority_claims = {
            "semantic_authority": self.semantic_authority,
            "grants_proof_authority": self.grants_proof_authority,
            "grants_write_authority": self.grants_write_authority,
        }
        if any(value is not False for value in authority_claims.values()):
            raise RepairOperatorRegistryAuthorityError(
                "operator lookup cannot grant semantic, proof, or write authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "grants_write_authority", False)
        required_base = {
            RepairOperatorCapability.EXACT_TARGET.value,
            RepairOperatorCapability.EXACT_PLACEMENT.value,
            RepairOperatorCapability.CLOSED_AST.value,
            RepairOperatorCapability.IDEMPOTENT_RENDER.value,
            RepairOperatorCapability.SCOPE_BOUND.value,
            RepairOperatorCapability.PROPOSAL_ONLY.value,
        }
        if not required_base.issubset(set(self.capability_refs)):
            raise RepairOperatorRegistryError(
                "operator is missing canonical/scope/idempotency capabilities"
            )
        if not self.counterevidence_classes:
            object.__setattr__(
                self, "counterevidence_classes", CANONICAL_COUNTEREVIDENCE_CLASSES
            )
        else:
            allowed = set(CANONICAL_COUNTEREVIDENCE_CLASSES)
            if set(self.counterevidence_classes) - allowed:
                raise RepairOperatorRegistryError(
                    "counterevidence_classes contains an unknown evidence family"
                )
        grammar_classes, grammar_predicates = default_operator_counterevidence_grammar(
            kind
        )
        if not self.addresses_repair_classes:
            object.__setattr__(self, "addresses_repair_classes", grammar_classes)
        if not self.counterevidence_predicates:
            object.__setattr__(self, "counterevidence_predicates", grammar_predicates)
        if not self.effect_restrictions:
            object.__setattr__(
                self,
                "effect_restrictions",
                default_operator_effect_restrictions(kind),
            )
        else:
            allowed_effects = set(CANONICAL_EFFECT_RESTRICTIONS)
            if set(self.effect_restrictions) - allowed_effects:
                raise RepairOperatorRegistryError(
                    "effect_restrictions contains an unknown restriction"
                )
            missing_effects = allowed_effects - set(self.effect_restrictions)
            if missing_effects:
                raise RepairOperatorRegistryError(
                    "reviewed operators must declare the canonical effect restrictions"
                )

    @property
    def spec_id(self) -> str:
        return self.content_id

    @property
    def requires_value(self) -> bool:
        return self.value_requirement is not OperatorValueRequirement.NONE

    @property
    def is_reviewed_hook(self) -> bool:
        return self.reviewed_hook is not ReviewedRepairHook.NONE

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DOCTOR_REPAIR_OPERATOR_SPEC_VERSION,
            "interface": self.INTERFACE,
            "operator_id": self.operator_id,
            "kind": self.kind.value,
            "family": self.family.value,
            "aliases": list(self.aliases),
            "supported_languages": list(self.supported_languages),
            "supported_ast_shapes": list(self.supported_ast_shapes),
            "capability_refs": list(self.capability_refs),
            "precondition_refs": list(self.precondition_refs),
            "postcondition_refs": list(self.postcondition_refs),
            "frame_condition_refs": list(self.frame_condition_refs),
            "proof_requirement_refs": list(self.proof_requirement_refs),
            "validation_requirement_refs": list(self.validation_requirement_refs),
            "scope_constraints": list(self.scope_constraints),
            "approval_classes": list(self.approval_classes),
            "abstain_classes": list(self.abstain_classes),
            "value_requirement": self.value_requirement.value,
            "placement_required": self.placement_required,
            "analytical_transform_kind": self.analytical_transform_kind,
            "reviewed_hook": self.reviewed_hook.value,
            "review_requirement_refs": list(self.review_requirement_refs),
            "renderer_id": self.renderer_id,
            "idempotent": True,
            "inverse_or_compensation_ref": self.inverse_or_compensation_ref,
            "proposal_only": True,
            "semantic_authority": False,
            "grants_proof_authority": False,
            "grants_write_authority": False,
            "counterevidence_classes": list(self.counterevidence_classes),
            "addresses_repair_classes": list(self.addresses_repair_classes),
            "counterevidence_predicates": list(self.counterevidence_predicates),
            "effect_restrictions": list(self.effect_restrictions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorRepairOperatorSpec":
        if not isinstance(payload, Mapping):
            raise RepairOperatorRegistryError("operator spec must be an object")
        field_names = set(cls.__dataclass_fields__) - {"SCHEMA", "INTERFACE"}
        allowed = {
            "schema",
            "content_id",
            "contract_version",
            "interface",
            *field_names,
        }
        if set(payload) - allowed:
            raise RepairOperatorRegistryError("operator spec contains unknown fields")
        if payload.get("schema", cls.SCHEMA) != cls.SCHEMA:
            raise RepairOperatorRegistryError("unsupported operator spec schema")
        if payload.get(
            "contract_version",
            DOCTOR_REPAIR_OPERATOR_SPEC_VERSION,
        ) != DOCTOR_REPAIR_OPERATOR_SPEC_VERSION:
            raise RepairOperatorRegistryError("unsupported operator spec version")
        if payload.get("interface", cls.INTERFACE) != cls.INTERFACE:
            raise RepairOperatorRegistryError("unsupported operator spec interface")
        values = {
            name: payload[name]
            for name in field_names
            if name in payload
        }
        result = cls(**values)
        supplied = payload.get("content_id")
        if supplied not in (None, "", result.content_id):
            raise RepairOperatorRegistryError("operator spec content_id mismatch")
        return result


@dataclass(frozen=True)
class RepairOperatorLookupRequest(CanonicalContract):
    """Exact, body-free facts used to nominate an operator."""

    SCHEMA: ClassVar[str] = REPAIR_OPERATOR_LOOKUP_REQUEST_SCHEMA

    operator_kind: str
    repository_id: str
    tree_id: str
    target_paths: tuple[str, ...]
    placement_refs: tuple[str, ...]
    value_refs: tuple[str, ...] = ()
    capability_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    review_refs: tuple[str, ...] = ()
    behavior_classes: tuple[str, ...] = (RepairBehaviorClass.PURE_LOCAL.value,)
    dependency_paths: tuple[str, ...] = ()
    requested_write_paths: tuple[str, ...] = ()
    language: str = "python"
    ast_shape: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operator_kind",
            _text(self.operator_kind, "operator_kind"),
        )
        object.__setattr__(
            self,
            "repository_id",
            _optional_text(self.repository_id, "repository_id"),
        )
        object.__setattr__(self, "tree_id", _optional_text(self.tree_id, "tree_id"))
        object.__setattr__(self, "target_paths", _paths(self.target_paths, "target_paths"))
        object.__setattr__(
            self,
            "requested_write_paths",
            _paths(self.requested_write_paths, "requested_write_paths"),
        )
        object.__setattr__(
            self,
            "dependency_paths",
            _paths(self.dependency_paths, "dependency_paths"),
        )
        for name in (
            "placement_refs",
            "value_refs",
            "capability_refs",
            "proof_refs",
            "review_refs",
            "behavior_classes",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(self, "language", _text(self.language, "language"))
        object.__setattr__(self, "ast_shape", _optional_text(self.ast_shape, "ast_shape"))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": REPAIR_OPERATOR_REGISTRY_VERSION,
            "operator_kind": self.operator_kind,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "target_paths": list(self.target_paths),
            "placement_refs": list(self.placement_refs),
            "value_refs": list(self.value_refs),
            "capability_refs": list(self.capability_refs),
            "proof_refs": list(self.proof_refs),
            "review_refs": list(self.review_refs),
            "behavior_classes": list(self.behavior_classes),
            "dependency_paths": list(self.dependency_paths),
            "requested_write_paths": list(self.requested_write_paths),
            "language": self.language,
            "ast_shape": self.ast_shape,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairOperatorLookupRequest":
        if not isinstance(payload, Mapping):
            raise RepairOperatorRegistryError("lookup request must be an object")
        field_names = set(cls.__dataclass_fields__) - {"SCHEMA"}
        allowed = {"schema", "content_id", "contract_version", *field_names}
        if set(payload) - allowed:
            raise RepairOperatorRegistryError("lookup request contains unknown fields")
        if payload.get("schema", cls.SCHEMA) != cls.SCHEMA:
            raise RepairOperatorRegistryError("unsupported lookup request schema")
        if payload.get(
            "contract_version",
            REPAIR_OPERATOR_REGISTRY_VERSION,
        ) != REPAIR_OPERATOR_REGISTRY_VERSION:
            raise RepairOperatorRegistryError("unsupported lookup request version")
        result = cls(
            **{
                name: payload[name]
                for name in field_names
                if name in payload
            }
        )
        supplied = payload.get("content_id")
        if supplied not in (None, "", result.content_id):
            raise RepairOperatorRegistryError("lookup request content_id mismatch")
        return result


@dataclass(frozen=True)
class RepairOperatorLookupResult(CanonicalContract):
    """Body-free nomination result which explicitly carries no authority."""

    SCHEMA: ClassVar[str] = REPAIR_OPERATOR_LOOKUP_RESULT_SCHEMA

    request_id: str
    operator_kind: str
    operator_id: str
    spec_id: str
    disposition: RepairOperatorLookupDisposition
    reason_codes: tuple[str, ...]
    matched_capability_refs: tuple[str, ...] = ()
    proof_verification_required: bool = True
    approval_validation_required: bool = False
    proposal_only: bool = True
    semantic_authority: bool = False
    grants_proof_authority: bool = False
    grants_write_authority: bool = False

    def __post_init__(self) -> None:
        for name in ("request_id", "operator_kind"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("operator_id", "spec_id"):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))
        disposition = _enum(
            self.disposition,
            RepairOperatorLookupDisposition,
            "disposition",
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", required=True),
        )
        object.__setattr__(
            self,
            "matched_capability_refs",
            _ids(self.matched_capability_refs, "matched_capability_refs"),
        )
        object.__setattr__(
            self,
            "proof_verification_required",
            _bool(self.proof_verification_required, "proof_verification_required"),
        )
        object.__setattr__(
            self,
            "approval_validation_required",
            _bool(self.approval_validation_required, "approval_validation_required"),
        )
        if self.proposal_only is not True or any(
            value is not False
            for value in (
                self.semantic_authority,
                self.grants_proof_authority,
                self.grants_write_authority,
            )
        ):
            raise RepairOperatorRegistryAuthorityError(
                "lookup results are proposal-only and carry no authority"
            )
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "grants_write_authority", False)

    @property
    def proposal_eligible(self) -> bool:
        return self.disposition is RepairOperatorLookupDisposition.PROPOSAL_ELIGIBLE

    @property
    def requires_approval(self) -> bool:
        return self.disposition is RepairOperatorLookupDisposition.APPROVAL_REQUIRED

    @property
    def admitted(self) -> bool:
        """Registry nomination is never proof/write admission."""

        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": REPAIR_OPERATOR_REGISTRY_VERSION,
            "request_id": self.request_id,
            "operator_kind": self.operator_kind,
            "operator_id": self.operator_id,
            "spec_id": self.spec_id,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "matched_capability_refs": list(self.matched_capability_refs),
            "proof_verification_required": self.proof_verification_required,
            "approval_validation_required": self.approval_validation_required,
            "proposal_only": True,
            "semantic_authority": False,
            "grants_proof_authority": False,
            "grants_write_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairOperatorLookupResult":
        if not isinstance(payload, Mapping):
            raise RepairOperatorRegistryError("lookup result must be an object")
        field_names = set(cls.__dataclass_fields__) - {"SCHEMA"}
        allowed = {"schema", "content_id", "contract_version", *field_names}
        if set(payload) - allowed:
            raise RepairOperatorRegistryError("lookup result contains unknown fields")
        if payload.get("schema", cls.SCHEMA) != cls.SCHEMA:
            raise RepairOperatorRegistryError("unsupported lookup result schema")
        if payload.get(
            "contract_version",
            REPAIR_OPERATOR_REGISTRY_VERSION,
        ) != REPAIR_OPERATOR_REGISTRY_VERSION:
            raise RepairOperatorRegistryError("unsupported lookup result version")
        result = cls(
            **{
                name: payload[name]
                for name in field_names
                if name in payload
            }
        )
        supplied = payload.get("content_id")
        if supplied not in (None, "", result.content_id):
            raise RepairOperatorRegistryError("lookup result content_id mismatch")
        return result


@dataclass(frozen=True)
class RepairOperatorRegistry(CanonicalContract):
    """Immutable closed catalogue of v2 reviewed repair operators."""

    SCHEMA: ClassVar[str] = REPAIR_OPERATOR_REGISTRY_SCHEMA
    INTERFACE: ClassVar[str] = REPAIR_OPERATOR_REGISTRY_INTERFACE

    operators: tuple[DoctorRepairOperatorSpec, ...]
    registry_id: str = ""
    producer_id: str = REPAIR_OPERATOR_REGISTRY_PRODUCER

    def __post_init__(self) -> None:
        if not self.operators or len(self.operators) > MAX_OPERATOR_COUNT:
            raise RepairOperatorRegistryError("registry operator count is out of bounds")
        if not all(isinstance(item, DoctorRepairOperatorSpec) for item in self.operators):
            raise RepairOperatorRegistryError(
                "operators must contain DoctorRepairOperatorSpec values"
            )
        ordered = tuple(sorted(self.operators, key=lambda item: item.operator_id))
        ids = [item.operator_id for item in ordered]
        kinds = [item.kind for item in ordered]
        if len(ids) != len(set(ids)) or len(kinds) != len(set(kinds)):
            raise RepairOperatorRegistryError("operator ids and kinds must be unique")
        alias_owner: dict[str, RepairOperatorKind] = {}
        for item in ordered:
            for alias in (item.kind.value, item.operator_id, *item.aliases):
                normalized = alias.strip().lower().replace("-", "_")
                owner = alias_owner.get(normalized)
                if owner is not None and owner is not item.kind:
                    raise RepairOperatorRegistryError(
                        "operator aliases must resolve uniquely"
                    )
                alias_owner[normalized] = item.kind
        object.__setattr__(self, "operators", ordered)
        object.__setattr__(self, "producer_id", _text(self.producer_id, "producer_id"))
        calculated = content_identity(self._payload_without_registry_id())
        supplied = _optional_text(self.registry_id, "registry_id")
        if supplied and supplied != calculated:
            raise RepairOperatorRegistryError("registry_id mismatch")
        object.__setattr__(self, "registry_id", calculated)
        self._validate_coverage()

    def _validate_coverage(self) -> None:
        registered = {item.kind.value for item in self.operators}
        analytical = {
            repair_kind
            for repair_kinds in ANALYTICAL_TRANSFORM_OPERATOR_BINDINGS.values()
            for repair_kind in repair_kinds
        }
        missing = analytical - registered
        if missing:
            raise RepairOperatorRegistryError(
                "registry omits analytical transforms: " + ", ".join(sorted(missing))
            )
        required = {
            RepairOperatorKind.EXACT_RENAME.value,
            RepairOperatorKind.EXACT_MOVE.value,
            RepairOperatorKind.RESTORE_TRACKED_ARTIFACT.value,
            RepairOperatorKind.SEMANTIC_PATCH.value,
            RepairOperatorKind.EQUALITY_REWRITE.value,
        }
        if required - registered:
            raise RepairOperatorRegistryError("registry omits required reviewed operators")

    def _payload_without_registry_id(self) -> dict[str, Any]:
        return {
            "contract_version": REPAIR_OPERATOR_REGISTRY_VERSION,
            "interface": self.INTERFACE,
            "operators": [item.to_dict() for item in self.operators],
            "producer_id": self.producer_id,
            "semantic_authority": False,
            "grants_proof_authority": False,
            "grants_write_authority": False,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            **self._payload_without_registry_id(),
            "registry_id": self.registry_id,
        }

    @property
    def descriptors(self) -> tuple[DoctorRepairOperatorSpec, ...]:
        return self.operators

    @property
    def semantic_authority(self) -> bool:
        return False

    @property
    def grants_proof_authority(self) -> bool:
        return False

    @property
    def grants_write_authority(self) -> bool:
        return False

    def kinds(self) -> tuple[RepairOperatorKind, ...]:
        return tuple(item.kind for item in self.operators)

    def specs(self) -> tuple[DoctorRepairOperatorSpec, ...]:
        return self.operators

    def get(self, kind_or_id: Any) -> DoctorRepairOperatorSpec:
        raw = str(getattr(kind_or_id, "value", kind_or_id)).strip()
        normalized = raw.lower().replace("-", "_")
        for item in self.operators:
            if normalized in {
                item.operator_id.lower().replace("-", "_"),
                item.kind.value,
                *(alias.lower().replace("-", "_") for alias in item.aliases),
            }:
                return item
        kind = normalize_repair_operator_kind(kind_or_id)
        for item in self.operators:
            if item.kind is kind:
                return item
        raise UnknownRepairOperatorError(
            RepairOperatorLookupReason.UNKNOWN_OPERATOR.value
        )

    def lookup(self, kind_or_id: Any) -> DoctorRepairOperatorSpec:
        """Return immutable metadata only; no proof or mutation authority."""

        return self.get(kind_or_id)

    def operator_effect_restrictions(
        self, kind_or_id: Any
    ) -> tuple[str, ...]:
        return self.get(kind_or_id).effect_restrictions

    def operator_counterevidence_predicates(
        self, kind_or_id: Any
    ) -> tuple[str, ...]:
        spec = self.get(kind_or_id)
        return spec.counterevidence_predicates

    def refine_from_counterevidence(
        self,
        *,
        operator_kinds: Sequence[str] = (),
        repair_classes: Sequence[str] = (),
        predicates: Sequence[str] = (),
        core_ids: Sequence[str] = (),
        failed_assumption_ids: Sequence[str] = (),
        interpolant_vocabulary: Sequence[str] = (),
        interpolant_predicates: Sequence[str] = (),
        interpolant_validated: bool = False,
        counterexample_kind: str = "",
    ) -> tuple[str, ...]:
        """Narrow reviewed operators using counterevidence; never widen.

        Explicit ``operator_kinds`` bound the grammar. Matching cores,
        failed assumptions, repair classes, and *validated* interpolants
        rank and filter that grammar. An unvalidated interpolant is
        ignored rather than treated as a search hint.
        """

        if operator_kinds:
            candidates: list[RepairOperatorKind] = []
            seen: set[RepairOperatorKind] = set()
            for item in operator_kinds:
                try:
                    kind = normalize_repair_operator_kind(item)
                except UnknownRepairOperatorError:
                    continue
                if kind in seen:
                    continue
                seen.add(kind)
                self.get(kind)
                candidates.append(kind)
        else:
            candidates = list(self.kinds())

        evidence_predicates = (
            *predicates,
            *core_ids,
            *failed_assumption_ids,
            *(
                (*tuple(interpolant_vocabulary), *tuple(interpolant_predicates))
                if interpolant_validated
                else ()
            ),
            counterexample_kind,
        )
        scored: list[tuple[int, str]] = []
        for kind in candidates:
            score = score_operator_counterevidence(
                kind,
                repair_classes=repair_classes,
                predicates=evidence_predicates,
            )
            scored.append((score, kind.value))

        matched = tuple(
            kind
            for score, kind in sorted(scored, key=lambda item: (-item[0], item[1]))
            if score > 0
        )
        if matched:
            return matched
        # Unvalidated interpolants and empty evidence must not drop or
        # reorder the caller-declared reviewed grammar.
        if operator_kinds:
            return tuple(kind.value for kind in candidates)
        return ()

    def resolve(
        self,
        request: RepairOperatorLookupRequest,
    ) -> RepairOperatorLookupResult:
        """Resolve exact lookup facts to a proposal-only disposition."""

        if not isinstance(request, RepairOperatorLookupRequest):
            raise RepairOperatorRegistryError(
                "request must be RepairOperatorLookupRequest"
            )
        try:
            spec = self.get(request.operator_kind)
        except UnknownRepairOperatorError:
            return self._result(
                request,
                None,
                RepairOperatorLookupDisposition.ABSTAINED,
                (RepairOperatorLookupReason.UNKNOWN_OPERATOR,),
            )

        if len(request.target_paths) > 1:
            return self._result(
                request,
                spec,
                RepairOperatorLookupDisposition.REJECTED,
                (RepairOperatorLookupReason.TARGET_AMBIGUOUS,),
            )
        if len(request.value_refs) > 1:
            return self._result(
                request,
                spec,
                RepairOperatorLookupDisposition.REJECTED,
                (RepairOperatorLookupReason.VALUE_AMBIGUOUS,),
            )
        if len(request.placement_refs) > 1:
            return self._result(
                request,
                spec,
                RepairOperatorLookupDisposition.REJECTED,
                (RepairOperatorLookupReason.PLACEMENT_AMBIGUOUS,),
            )

        abstain: list[RepairOperatorLookupReason] = []
        if not request.target_paths:
            abstain.append(RepairOperatorLookupReason.TARGET_MISSING)
        if not request.repository_id or not request.tree_id:
            abstain.append(RepairOperatorLookupReason.SCOPE_MISSING)
        if spec.requires_value and not request.value_refs:
            abstain.append(RepairOperatorLookupReason.VALUE_MISSING)
        if spec.placement_required and not request.placement_refs:
            abstain.append(RepairOperatorLookupReason.PLACEMENT_MISSING)
        if request.language not in spec.supported_languages:
            abstain.append(RepairOperatorLookupReason.UNSUPPORTED_LANGUAGE)
        if request.ast_shape and request.ast_shape not in spec.supported_ast_shapes:
            abstain.append(RepairOperatorLookupReason.UNSUPPORTED_AST_SHAPE)
        missing_capabilities = set(spec.capability_refs) - set(request.capability_refs)
        if missing_capabilities:
            abstain.append(RepairOperatorLookupReason.CAPABILITY_MISSING)
        if not request.proof_refs:
            abstain.append(RepairOperatorLookupReason.PROOF_REFERENCE_MISSING)
        if spec.review_requirement_refs and not request.review_refs:
            abstain.append(RepairOperatorLookupReason.REVIEW_REFERENCE_MISSING)
        if request.requested_write_paths and (
            len(request.requested_write_paths) != 1
            or request.requested_write_paths != request.target_paths
        ):
            abstain.append(RepairOperatorLookupReason.SCOPE_ESCAPE)

        behavior = set(request.behavior_classes)
        known_behavior = {item.value for item in RepairBehaviorClass}
        if (
            not behavior
            or RepairBehaviorClass.UNKNOWN.value in behavior
            or behavior - known_behavior
        ):
            abstain.append(RepairOperatorLookupReason.UNKNOWN_BEHAVIOR)
        if abstain:
            return self._result(
                request,
                spec,
                RepairOperatorLookupDisposition.ABSTAINED,
                tuple(dict.fromkeys(abstain)),
            )

        approval: list[RepairOperatorLookupReason] = []
        for behavior_class, reason in _APPROVAL_BEHAVIORS.items():
            if behavior_class in behavior and behavior_class in spec.approval_classes:
                approval.append(reason)
        if request.dependency_paths:
            approval.append(RepairOperatorLookupReason.DEPENDENCY_APPROVAL)
        if approval:
            return self._result(
                request,
                spec,
                RepairOperatorLookupDisposition.APPROVAL_REQUIRED,
                tuple(dict.fromkeys(approval)),
                approval_validation_required=True,
            )

        return self._result(
            request,
            spec,
            RepairOperatorLookupDisposition.PROPOSAL_ELIGIBLE,
            (RepairOperatorLookupReason.CANDIDATE_ONLY,),
        )

    evaluate_lookup = resolve

    def _result(
        self,
        request: RepairOperatorLookupRequest,
        spec: DoctorRepairOperatorSpec | None,
        disposition: RepairOperatorLookupDisposition,
        reasons: tuple[RepairOperatorLookupReason, ...],
        *,
        approval_validation_required: bool = False,
    ) -> RepairOperatorLookupResult:
        return RepairOperatorLookupResult(
            request_id=request.content_id,
            operator_kind=spec.kind.value if spec is not None else request.operator_kind,
            operator_id=spec.operator_id if spec is not None else "",
            spec_id=spec.spec_id if spec is not None else "",
            disposition=disposition,
            reason_codes=tuple(item.value for item in reasons),
            matched_capability_refs=(
                tuple(
                    sorted(
                        set(request.capability_refs).intersection(spec.capability_refs)
                    )
                )
                if spec is not None
                else ()
            ),
            proof_verification_required=True,
            approval_validation_required=approval_validation_required,
            proposal_only=True,
            semantic_authority=False,
            grants_proof_authority=False,
            grants_write_authority=False,
        )

    def build_legacy_registry(
        self,
        roots: DoctorAuthorityRoots,
    ) -> LegacyDoctorRepairOperatorRegistry:
        """Build the existing root-bound renderer registry.

        This adapter deliberately delegates to the legacy factory instead of
        converting v2 lookup results.  Consequently a v2 lookup cannot smuggle
        a proof verdict, path permit, or write capability into rendering.
        """

        if not isinstance(roots, DoctorAuthorityRoots):
            raise RepairOperatorRegistryError("roots must be DoctorAuthorityRoots")
        return build_default_doctor_operator_registry(roots)

    def legacy_kind(self, kind_or_id: Any) -> DoctorOperatorKind | None:
        """Return a legacy renderer kind when one exists, otherwise ``None``."""

        kind = self.get(kind_or_id).kind
        mapping = {
            RepairOperatorKind.EXACT_RENAME: DoctorOperatorKind.EXACT_RENAME,
            RepairOperatorKind.ADD_ARGUMENT: DoctorOperatorKind.ADD_ARGUMENT,
            RepairOperatorKind.RENAME_ARGUMENT: DoctorOperatorKind.RENAME_ARGUMENT,
            RepairOperatorKind.REORDER_ARGUMENT: DoctorOperatorKind.REORDER_ARGUMENT,
            RepairOperatorKind.THREAD_ARGUMENT: DoctorOperatorKind.THREAD_ARGUMENT,
            RepairOperatorKind.ADD_IMPORT: DoctorOperatorKind.ADD_IMPORT,
            RepairOperatorKind.ADD_EXPORT: DoctorOperatorKind.ADD_EXPORT,
            RepairOperatorKind.ADD_REGISTRATION: DoctorOperatorKind.ADD_REGISTRATION,
            RepairOperatorKind.ADD_CONSTRUCTOR_ROUTE: (
                DoctorOperatorKind.ADD_CONSTRUCTOR_ROUTE
            ),
            RepairOperatorKind.ADD_FACTORY_ROUTE: DoctorOperatorKind.ADD_FACTORY_ROUTE,
            RepairOperatorKind.FINITE_ADAPTER: DoctorOperatorKind.FINITE_ADAPTER,
            RepairOperatorKind.SCHEMA_PROJECTION: DoctorOperatorKind.SCHEMA_PROJECTION,
            RepairOperatorKind.RESTORE_TRACKED_ARTIFACT: (
                DoctorOperatorKind.RESTORE_TRACKED_ARTIFACT
            ),
        }
        return mapping.get(kind)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairOperatorRegistry":
        if not isinstance(payload, Mapping):
            raise RepairOperatorRegistryError("registry must be an object")
        allowed = {
            "schema",
            "content_id",
            "contract_version",
            "interface",
            "operators",
            "registry_id",
            "producer_id",
            "semantic_authority",
            "grants_proof_authority",
            "grants_write_authority",
        }
        if set(payload) - allowed:
            raise RepairOperatorRegistryError("registry contains unknown fields")
        if payload.get("schema", cls.SCHEMA) != cls.SCHEMA:
            raise RepairOperatorRegistryError("unsupported registry schema")
        if payload.get(
            "contract_version",
            REPAIR_OPERATOR_REGISTRY_VERSION,
        ) != REPAIR_OPERATOR_REGISTRY_VERSION:
            raise RepairOperatorRegistryError("unsupported registry version")
        if payload.get("interface", cls.INTERFACE) != cls.INTERFACE:
            raise RepairOperatorRegistryError("unsupported registry interface")
        for authority in (
            "semantic_authority",
            "grants_proof_authority",
            "grants_write_authority",
        ):
            if payload.get(authority, False) is not False:
                raise RepairOperatorRegistryAuthorityError(
                    "serialized registry cannot claim authority"
                )
        raw_operators = payload.get("operators")
        if isinstance(raw_operators, (str, bytes, bytearray)) or not isinstance(
            raw_operators,
            Sequence,
        ):
            raise RepairOperatorRegistryError("registry operators must be a sequence")
        result = cls(
            operators=tuple(
                item
                if isinstance(item, DoctorRepairOperatorSpec)
                else DoctorRepairOperatorSpec.from_dict(item)
                for item in raw_operators
            ),
            registry_id=payload.get("registry_id", ""),
            producer_id=payload.get(
                "producer_id",
                REPAIR_OPERATOR_REGISTRY_PRODUCER,
            ),
        )
        supplied = payload.get("content_id")
        if supplied not in (None, "", result.content_id):
            raise RepairOperatorRegistryError("registry content_id mismatch")
        return result


_BASE_CAPABILITIES: Final[tuple[str, ...]] = tuple(
    item.value
    for item in (
        RepairOperatorCapability.EXACT_TARGET,
        RepairOperatorCapability.EXACT_PLACEMENT,
        RepairOperatorCapability.CLOSED_AST,
        RepairOperatorCapability.IDEMPOTENT_RENDER,
        RepairOperatorCapability.SCOPE_BOUND,
        RepairOperatorCapability.PROPOSAL_ONLY,
    )
)

_COMMON_SCOPE: Final[tuple[str, ...]] = (
    "scope:exact_repository",
    "scope:exact_tree",
    "scope:single_target_path",
    "scope:exact_before_hash",
    "scope:closed_impact",
)


def _spec(
    kind: RepairOperatorKind,
    family: RepairOperatorFamily,
    *,
    aliases: tuple[str, ...],
    shapes: tuple[str, ...],
    capability: RepairOperatorCapability,
    analytical: TransformKind | None = None,
    hook: ReviewedRepairHook = ReviewedRepairHook.NONE,
    value: OperatorValueRequirement = OperatorValueRequirement.NONE,
    review: tuple[str, ...] = (),
    languages: tuple[str, ...] = ("python",),
    pre: tuple[str, ...] = (),
    post: tuple[str, ...] = (),
    inverse: str,
    predicates: tuple[str, ...] = (),
    repair_classes: tuple[str, ...] = (),
) -> DoctorRepairOperatorSpec:
    renderer = (
        "AnalyticalChangeTransformer@1"
        if analytical is not None
        else f"reviewed-repair-hook:{hook.value}@1"
    )
    return DoctorRepairOperatorSpec(
        operator_id=f"repair-operator:{kind.value}@2",
        kind=kind,
        family=family,
        aliases=aliases,
        supported_languages=languages,
        supported_ast_shapes=shapes,
        capability_refs=(*_BASE_CAPABILITIES, capability.value),
        precondition_refs=pre
        or ("pre:unique_target", "pre:exact_placement", "pre:closed_scope"),
        postcondition_refs=post or (f"post:{kind.value}",),
        frame_condition_refs=(
            "frame:non_target_bytes_unchanged",
            "frame:dependencies_unchanged",
            "frame:public_api_unchanged",
        ),
        proof_requirement_refs=(
            f"proof:{kind.value}",
            "proof:current_roots",
            "proof:scope_closed",
        ),
        validation_requirement_refs=(
            "validation:parse",
            "validation:type_or_schema",
            "validation:impact_selected",
            "validation:fixed_point",
        ),
        scope_constraints=_COMMON_SCOPE,
        approval_classes=tuple(sorted(_APPROVAL_BEHAVIORS)),
        abstain_classes=(RepairBehaviorClass.UNKNOWN.value,),
        value_requirement=value,
        placement_required=True,
        analytical_transform_kind=analytical.value if analytical is not None else "",
        reviewed_hook=hook,
        review_requirement_refs=review,
        renderer_id=renderer,
        idempotent=True,
        inverse_or_compensation_ref=inverse,
        proposal_only=True,
        semantic_authority=False,
        grants_proof_authority=False,
        grants_write_authority=False,
        counterevidence_classes=CANONICAL_COUNTEREVIDENCE_CLASSES,
        addresses_repair_classes=repair_classes
        or default_operator_counterevidence_grammar(kind)[0],
        counterevidence_predicates=predicates
        or default_operator_counterevidence_grammar(kind)[1],
        effect_restrictions=CANONICAL_EFFECT_RESTRICTIONS,
    )


def _default_specs() -> tuple[DoctorRepairOperatorSpec, ...]:
    """Build the exhaustive reviewed operator catalogue."""

    return (
        _spec(
            RepairOperatorKind.EXACT_RENAME,
            RepairOperatorFamily.SYMBOL,
            aliases=("rename", "symbol_rename"),
            shapes=("identifier", "name", "attribute", "definition"),
            capability=RepairOperatorCapability.SYMBOL_EQUIVALENCE,
            analytical=TransformKind.RENAME_ARGUMENT,
            pre=("pre:unique_symbol", "pre:referent_equivalence", "pre:closed_callers"),
            post=("post:symbol_renamed", "post:referents_preserved"),
            inverse="compensation:exact_rename_inverse",
        ),
        _spec(
            RepairOperatorKind.EXACT_MOVE,
            RepairOperatorFamily.MOVE,
            aliases=("move", "file_move"),
            shapes=("module", "tracked_file", "package_member"),
            capability=RepairOperatorCapability.FILE_MOVE,
            hook=ReviewedRepairHook.EXACT_MOVE,
            review=("review:exact_move_patch@1",),
            pre=("pre:unique_source", "pre:unique_destination", "pre:closed_importers"),
            post=("post:artifact_moved", "post:importers_retargeted"),
            inverse="compensation:exact_move_inverse",
        ),
        _spec(
            RepairOperatorKind.ADD_ARGUMENT,
            RepairOperatorFamily.CALL,
            aliases=("missing_argument",),
            shapes=("call", "keyword_argument", "positional_argument"),
            capability=RepairOperatorCapability.UNIQUE_VALUE,
            analytical=TransformKind.ADD_ARGUMENT,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            inverse="compensation:remove_argument",
        ),
        _spec(
            RepairOperatorKind.RENAME_ARGUMENT,
            RepairOperatorFamily.CALL,
            aliases=("keyword_rename", "parameter_rename"),
            shapes=("call", "function_parameter", "keyword_argument"),
            capability=RepairOperatorCapability.SYMBOL_EQUIVALENCE,
            analytical=TransformKind.RENAME_ARGUMENT,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            inverse="compensation:rename_argument_inverse",
        ),
        _spec(
            RepairOperatorKind.REORDER_ARGUMENT,
            RepairOperatorFamily.CALL,
            aliases=("argument_reorder",),
            shapes=("call", "keyword_argument_list"),
            capability=RepairOperatorCapability.UNIQUE_VALUE,
            analytical=TransformKind.REORDER_ARGUMENT,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            inverse="compensation:reorder_argument_inverse",
        ),
        _spec(
            RepairOperatorKind.THREAD_ARGUMENT,
            RepairOperatorFamily.CALL,
            aliases=("value_threading", "thread_value"),
            shapes=("call_route", "function_parameter", "call"),
            capability=RepairOperatorCapability.ROUTE_CLOSURE,
            analytical=TransformKind.THREAD_PARAMETER,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            pre=("pre:unique_value", "pre:complete_route", "pre:finite_hops"),
            inverse="compensation:unthread_argument",
        ),
        _spec(
            RepairOperatorKind.ADD_IMPORT,
            RepairOperatorFamily.WIRING,
            aliases=("exact_import", "import"),
            shapes=("module", "import_block", "import_from"),
            capability=RepairOperatorCapability.IMPORT_WIRING,
            analytical=TransformKind.ADD_IMPORT,
            inverse="compensation:remove_import",
        ),
        _spec(
            RepairOperatorKind.ADD_EXPORT,
            RepairOperatorFamily.WIRING,
            aliases=("exact_export", "export"),
            shapes=("module", "dunder_all", "export_list"),
            capability=RepairOperatorCapability.EXPORT_WIRING,
            analytical=TransformKind.ADD_EXPORT,
            inverse="compensation:remove_export",
        ),
        _spec(
            RepairOperatorKind.ADD_REGISTRATION,
            RepairOperatorFamily.WIRING,
            aliases=("exact_registration", "registration"),
            shapes=("module", "registration_call", "registry_literal"),
            capability=RepairOperatorCapability.REGISTRATION_WIRING,
            analytical=TransformKind.ADD_REGISTRATION,
            inverse="compensation:remove_registration",
        ),
        _spec(
            RepairOperatorKind.ADD_CONSTRUCTOR_ROUTE,
            RepairOperatorFamily.CONSTRUCTION,
            aliases=("constructor", "constructor_route"),
            shapes=("constructor_call", "class_instantiation"),
            capability=RepairOperatorCapability.CONSTRUCTOR_WIRING,
            analytical=TransformKind.UPDATE_CONSTRUCTOR,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            inverse="compensation:remove_constructor_route",
        ),
        _spec(
            RepairOperatorKind.ADD_FACTORY_ROUTE,
            RepairOperatorFamily.CONSTRUCTION,
            aliases=("factory", "factory_route"),
            shapes=("factory_call",),
            capability=RepairOperatorCapability.FACTORY_WIRING,
            analytical=TransformKind.UPDATE_CONSTRUCTOR,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            inverse="compensation:remove_factory_route",
        ),
        _spec(
            RepairOperatorKind.FINITE_ADAPTER,
            RepairOperatorFamily.CONSTRUCTION,
            aliases=("adapter", "finite_mapping_adapter"),
            shapes=("adapter_wrap", "simple_expression"),
            capability=RepairOperatorCapability.FINITE_ADAPTER,
            analytical=TransformKind.ADD_ADAPTER,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            inverse="compensation:unwrap_adapter",
        ),
        _spec(
            RepairOperatorKind.SCHEMA_PROJECTION,
            RepairOperatorFamily.DATA_CONTRACT,
            aliases=("schema", "schema_update"),
            shapes=("schema_object", "schema_literal", "json_object"),
            capability=RepairOperatorCapability.TOTAL_FIELD_MAPPING,
            analytical=TransformKind.UPDATE_SCHEMA_FIELD,
            value=OperatorValueRequirement.TOTAL_MAPPING,
            languages=("python", "json", "yaml"),
            inverse="compensation:schema_projection_inverse",
        ),
        _spec(
            RepairOperatorKind.SERIALIZER_UPDATE,
            RepairOperatorFamily.DATA_CONTRACT,
            aliases=("serializer",),
            shapes=("serializer_mapping", "serializer_literal", "json_object"),
            capability=RepairOperatorCapability.SERIALIZER_MAPPING,
            analytical=TransformKind.UPDATE_SERIALIZER,
            value=OperatorValueRequirement.TOTAL_MAPPING,
            languages=("python", "json"),
            inverse="compensation:serializer_update_inverse",
        ),
        _spec(
            RepairOperatorKind.FIXTURE_UPDATE,
            RepairOperatorFamily.DATA_CONTRACT,
            aliases=("fixture",),
            shapes=("fixture_literal", "json_object", "mapping_literal"),
            capability=RepairOperatorCapability.FIXTURE_MAPPING,
            analytical=TransformKind.UPDATE_FIXTURE,
            value=OperatorValueRequirement.TOTAL_MAPPING,
            languages=("python", "json", "yaml"),
            inverse="compensation:fixture_update_inverse",
        ),
        _spec(
            RepairOperatorKind.MANIFEST_UPDATE,
            RepairOperatorFamily.DATA_CONTRACT,
            aliases=("manifest", "generated_manifest"),
            shapes=("manifest_object", "json_object", "mapping_literal"),
            capability=RepairOperatorCapability.MANIFEST_MAPPING,
            analytical=TransformKind.UPDATE_GENERATED_MANIFEST,
            value=OperatorValueRequirement.TOTAL_MAPPING,
            languages=("python", "json", "toml", "yaml"),
            inverse="compensation:manifest_update_inverse",
        ),
        _spec(
            RepairOperatorKind.RESTORE_TRACKED_ARTIFACT,
            RepairOperatorFamily.ARTIFACT,
            aliases=("artifact", "restore_artifact"),
            shapes=("whole_file", "tracked_blob"),
            capability=RepairOperatorCapability.VERIFIED_ARTIFACT,
            hook=ReviewedRepairHook.ARTIFACT_RESTORE,
            value=OperatorValueRequirement.VERIFIED_PREIMAGE,
            languages=("binary", "json", "python", "text"),
            pre=("pre:tracked_path", "pre:verified_cid", "pre:canonical_preimage"),
            post=("post:artifact_restored", "post:cid_matches"),
            inverse="compensation:restore_previous_cid",
        ),
        _spec(
            RepairOperatorKind.SEMANTIC_PATCH,
            RepairOperatorFamily.REVIEWED_REWRITE,
            aliases=("reviewed_semantic_patch",),
            shapes=("reviewed_pattern_match", "closed_ast_capture"),
            capability=RepairOperatorCapability.REVIEWED_SEMANTIC_PATCH,
            hook=ReviewedRepairHook.SEMANTIC_PATCH,
            value=OperatorValueRequirement.REVIEWED_RULE,
            review=("review:semantic_patch_template@1",),
            pre=("pre:reviewed_patch", "pre:unique_capture", "pre:closed_metavariables"),
            post=("post:reviewed_patch_postcondition",),
            inverse="compensation:semantic_patch_inverse",
        ),
        _spec(
            RepairOperatorKind.EQUALITY_REWRITE,
            RepairOperatorFamily.REVIEWED_REWRITE,
            aliases=("equality", "egraph_rewrite"),
            shapes=("expression", "egraph_term", "closed_equation"),
            capability=RepairOperatorCapability.DECLARED_EQUALITY_THEORY,
            hook=ReviewedRepairHook.EQUALITY_REWRITE,
            value=OperatorValueRequirement.REVIEWED_RULE,
            review=("review:equality_theory@1", "review:equality_rewrite@1"),
            pre=("pre:declared_theory", "pre:oriented_rule", "pre:equivalence_proof"),
            post=("post:equivalent_under_declared_theory",),
            inverse="compensation:equality_rewrite_inverse",
        ),
    )


def build_default_repair_operator_registry(
    roots: DoctorAuthorityRoots | None = None,
) -> RepairOperatorRegistry:
    """Return the canonical reviewed v2 registry.

    ``roots`` is accepted for composition compatibility and type-checked, but
    is deliberately excluded from registry identity: descriptors are
    capability declarations, not root-bound authority.  Root binding occurs
    in lookup requests and the legacy rendering registry.
    """

    if roots is not None and not isinstance(roots, DoctorAuthorityRoots):
        raise RepairOperatorRegistryError("roots must be DoctorAuthorityRoots")
    return RepairOperatorRegistry(operators=_default_specs())


def default_repair_operator_registry_id(
    roots: DoctorAuthorityRoots | None = None,
) -> str:
    return build_default_repair_operator_registry(roots).registry_id


def refine_repair_operators(
    registry: RepairOperatorRegistry | None = None,
    *,
    operator_kinds: Sequence[str] = (),
    repair_classes: Sequence[str] = (),
    predicates: Sequence[str] = (),
    core_ids: Sequence[str] = (),
    failed_assumption_ids: Sequence[str] = (),
    interpolant_vocabulary: Sequence[str] = (),
    interpolant_predicates: Sequence[str] = (),
    interpolant_validated: bool = False,
    counterexample_kind: str = "",
) -> tuple[str, ...]:
    """Module-level wrapper around reviewed-operator counterevidence refinement."""

    active = registry or build_default_repair_operator_registry()
    return active.refine_from_counterevidence(
        operator_kinds=operator_kinds,
        repair_classes=repair_classes,
        predicates=predicates,
        core_ids=core_ids,
        failed_assumption_ids=failed_assumption_ids,
        interpolant_vocabulary=interpolant_vocabulary,
        interpolant_predicates=interpolant_predicates,
        interpolant_validated=interpolant_validated,
        counterexample_kind=counterexample_kind,
    )


def candidate_effect_violations(
    *,
    operator_kind: str = "",
    extra_imports: Sequence[str] = (),
    extra_files: Sequence[str] = (),
    extra_paths: Sequence[str] = (),
    new_dependencies: Sequence[str] = (),
    undeclared_effects: Sequence[str] = (),
    behavior_class: str = RepairBehaviorClass.PURE_LOCAL.value,
    write_authority: bool = False,
    semantic_authority: bool = False,
    grants_proof_authority: bool = False,
    grants_write_authority: bool = False,
    replacement: str = "",
    declared_imports: Sequence[str] = (),
    declared_paths: Sequence[str] = (),
    declared_dependencies: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return restriction reason codes for undeclared candidate effects."""

    reasons: list[str] = []
    kind_value = ""
    try:
        if operator_kind:
            kind_value = normalize_repair_operator_kind(operator_kind).value
    except UnknownRepairOperatorError:
        kind_value = str(operator_kind or "")

    extra_import_values = tuple(
        item for item in extra_imports if item and item not in declared_imports
    )
    if extra_import_values:
        reasons.append(RepairOperatorLookupReason.UNDECLARED_IMPORT.value)
    extra_file_values = tuple(
        item
        for item in (*extra_files, *extra_paths)
        if item and item not in declared_paths
    )
    if extra_file_values:
        reasons.append(RepairOperatorLookupReason.UNDECLARED_FILE.value)
    extra_deps = tuple(
        item
        for item in new_dependencies
        if item and item not in declared_dependencies
    )
    if extra_deps:
        reasons.append(RepairOperatorLookupReason.UNDECLARED_DEPENDENCY.value)
    if undeclared_effects:
        reasons.append(RepairOperatorLookupReason.UNDECLARED_EFFECT.value)
    if any(
        (
            write_authority,
            semantic_authority,
            grants_proof_authority,
            grants_write_authority,
        )
    ):
        reasons.append(RepairOperatorLookupReason.UNDECLARED_EFFECT.value)
    behavior = str(behavior_class or RepairBehaviorClass.PURE_LOCAL.value)
    if behavior not in {
        RepairBehaviorClass.PURE_LOCAL.value,
        "",
    }:
        reasons.append(RepairOperatorLookupReason.UNDECLARED_EFFECT.value)
    for line in replacement.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("import ") or stripped.startswith("from "):
            if kind_value != RepairOperatorKind.ADD_IMPORT.value:
                reasons.append(RepairOperatorLookupReason.UNDECLARED_IMPORT.value)
                break
    return tuple(dict.fromkeys(reasons))


__all__ = (
    "DOCTOR_REPAIR_OPERATOR_SPEC_INTERFACE",
    "DOCTOR_REPAIR_OPERATOR_SPEC_SCHEMA",
    "DOCTOR_REPAIR_OPERATOR_SPEC_VERSION",
    "REPAIR_OPERATOR_LOOKUP_REQUEST_SCHEMA",
    "REPAIR_OPERATOR_LOOKUP_RESULT_SCHEMA",
    "REPAIR_OPERATOR_REGISTRY_INTERFACE",
    "REPAIR_OPERATOR_REGISTRY_PRODUCER",
    "REPAIR_OPERATOR_REGISTRY_SCHEMA",
    "REPAIR_OPERATOR_REGISTRY_VERSION",
    "DoctorRepairOperatorSpec",
    "OperatorValueRequirement",
    "CANONICAL_COUNTEREVIDENCE_CLASSES",
    "CANONICAL_EFFECT_RESTRICTIONS",
    "RepairBehaviorClass",
    "RepairCounterevidenceClass",
    "RepairEffectRestriction",
    "RepairOperatorCapability",
    "RepairOperatorFamily",
    "RepairOperatorKind",
    "RepairOperatorLookupDisposition",
    "RepairOperatorLookupReason",
    "RepairOperatorLookupRequest",
    "RepairOperatorLookupResult",
    "RepairOperatorRegistry",
    "RepairOperatorRegistryAuthorityError",
    "RepairOperatorRegistryError",
    "ReviewedRepairHook",
    "UnknownRepairOperatorError",
    "build_default_repair_operator_registry",
    "candidate_effect_violations",
    "default_operator_counterevidence_grammar",
    "default_operator_effect_restrictions",
    "default_repair_operator_registry_id",
    "normalize_repair_operator_kind",
    "refine_repair_operators",
    "score_operator_counterevidence",
)
