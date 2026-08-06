"""Exact before-and-after semantic program-contract deltas.

``ProgramContractDeltaAnalyzer@1`` compares independently extracted
:class:`~ipfs_accelerate_py.agent_supervisor.program_contracts.ExpectedProgramContract`
records bound to a reviewed base/candidate change set.  It returns the
canonical RPR-022 :class:`ProgramContractDelta` / :class:`ContractClauseDelta`
records without redefining them.

Non-semantic churn (formatting, comments, generated outputs, pure moves and
renames) is partitioned separately so it cannot manufacture migration
obligations.  Expected contracts must follow reviewed evidence precedence;
candidate implementation observations never author expectation behavior.
Stale, incomplete, cross-root, and unsupported comparisons fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final

from ..proof.program_contracts import (
    CapabilityMode,
    ContractSourceKind,
    EffectKind,
    EffectPolarity,
    ErrorSpec,
    ExpectedProgramContract,
    Optionality,
    ParameterKind,
    ParameterSpec,
    SemanticAspect,
    SideEffectSpec,
    SupportStatus,
    SymbolIdentity,
    TypeConstructor,
    TypeShape,
    UnsupportedSemantics,
)
from .change_propagation_contracts import (
    MAX_CLAUSE_COUNT,
    ContractClauseDelta,
    DeltaDisposition,
    DeltaKind,
    ProgramChangeSet,
    ProgramContractDelta,
    PropagationAuthorityRoots,
)


PROGRAM_CONTRACT_DELTA_ANALYZER_INTERFACE: Final[str] = "ProgramContractDeltaAnalyzer@1"
PROGRAM_CONTRACT_DELTA_ANALYZER_VERSION: Final[str] = "1"

# Closed consumer-domain vocabulary used by disposition rules.  Callers may
# still pass any compact identifier; unknown domains default to fail-closed
# caller-style rules (breaking when uncertain about compatibility).
DOMAIN_PYTHON_CALLERS: Final[str] = "domain:python-callers"
DOMAIN_SCHEMA_CONSUMERS: Final[str] = "domain:schema-consumers"
DOMAIN_SERIALIZERS: Final[str] = "domain:serializers"
DOMAIN_HTTP_CLIENTS: Final[str] = "domain:http-clients"
DOMAIN_PUBLIC_API: Final[str] = "domain:public-api"
DOMAIN_TESTS: Final[str] = "domain:tests"
DOMAIN_REGISTRATION: Final[str] = "domain:registration"
DOMAIN_MEMORY: Final[str] = "domain:memory-safety"

_KNOWN_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        DOMAIN_PYTHON_CALLERS,
        DOMAIN_SCHEMA_CONSUMERS,
        DOMAIN_SERIALIZERS,
        DOMAIN_HTTP_CLIENTS,
        DOMAIN_PUBLIC_API,
        DOMAIN_TESTS,
        DOMAIN_REGISTRATION,
        DOMAIN_MEMORY,
    }
)

_GENERATED_PATH_MARKERS: Final[tuple[str, ...]] = (
    "/generated/",
    "/_generated/",
    "/gen/",
    ".generated.",
    "_pb2.py",
    "_pb2_grpc.py",
    ".g.dart",
    ".pb.go",
    "/dist/",
    "/build/",
    "/.tox/",
    "/htmlcov/",
)
_COMMENT_ONLY_SUFFIXES: Final[tuple[str, ...]] = (
    ".md",
    ".rst",
    ".txt",
    ".comment",
)
_FORMATTING_SUFFIXES: Final[tuple[str, ...]] = (
    ".editorconfig",
    ".prettierrc",
    ".clang-format",
)


class ProgramContractDeltaError(ValueError):
    """Base error for fail-closed semantic delta analysis."""


class StaleContractDeltaError(ProgramContractDeltaError):
    """Base/candidate evidence is stale relative to bound roots."""


class IncompleteContractDeltaError(ProgramContractDeltaError):
    """Contracts or change records are incomplete for a closed comparison."""


class CrossRootContractDeltaError(ProgramContractDeltaError):
    """Contracts or change set do not bind the same authority roots."""


class UnsupportedContractDeltaError(ProgramContractDeltaError):
    """Comparison requires semantics the analyzer cannot represent."""


class SelfAuthoredExpectationError(ProgramContractDeltaError):
    """Candidate implementation attempted to author expected behavior."""


class NonSemanticChurnKind(str, Enum):
    """Closed categories of non-semantic source churn."""

    FORMATTING = "formatting"
    COMMENT = "comment"
    GENERATED = "generated"
    MOVE = "move"
    RENAME = "rename"
    WHITESPACE = "whitespace"


class StructuralSurfaceKind(str, Enum):
    """Structural intro/remove kinds mapped onto :class:`DeltaKind`."""

    CONSTRUCTOR = "constructor"
    FIELD = "field"
    METHOD = "method"
    CLASS = "class"
    DATA_STRUCTURE = "data_structure"
    INTERFACE = "interface"
    FACTORY = "factory"


_SURFACE_TO_INTRO: Final[Mapping[StructuralSurfaceKind, DeltaKind]] = {
    StructuralSurfaceKind.CONSTRUCTOR: DeltaKind.CONSTRUCTOR_INTRO,
    StructuralSurfaceKind.FIELD: DeltaKind.FIELD_INTRO,
    StructuralSurfaceKind.METHOD: DeltaKind.METHOD_INTRO,
    StructuralSurfaceKind.CLASS: DeltaKind.CLASS_INTRO,
    StructuralSurfaceKind.DATA_STRUCTURE: DeltaKind.DATA_STRUCTURE_INTRO,
    StructuralSurfaceKind.INTERFACE: DeltaKind.INTERFACE_INTRO,
    StructuralSurfaceKind.FACTORY: DeltaKind.FACTORY_INTRO,
}

_SURFACE_TO_REMOVE: Final[Mapping[StructuralSurfaceKind, DeltaKind]] = {
    StructuralSurfaceKind.CONSTRUCTOR: DeltaKind.CONSTRUCTOR_REMOVE,
    StructuralSurfaceKind.FIELD: DeltaKind.FIELD_REMOVE,
    StructuralSurfaceKind.METHOD: DeltaKind.METHOD_REMOVE,
    StructuralSurfaceKind.CLASS: DeltaKind.CLASS_REMOVE,
    StructuralSurfaceKind.DATA_STRUCTURE: DeltaKind.DATA_STRUCTURE_REMOVE,
    StructuralSurfaceKind.INTERFACE: DeltaKind.INTERFACE_REMOVE,
    StructuralSurfaceKind.FACTORY: DeltaKind.FACTORY_REMOVE,
}


@dataclass(frozen=True)
class PathChurnClassification:
    """Explicit producer classification for one changed path."""

    path: str
    kind: NonSemanticChurnKind

    def __post_init__(self) -> None:
        path = PurePosixPath(str(self.path or "").strip())
        if (
            not str(path)
            or path.is_absolute()
            or ".." in path.parts
            or str(path) in {".", ""}
        ):
            raise ProgramContractDeltaError(
                "path churn classification requires a relative repository path"
            )
        object.__setattr__(self, "path", path.as_posix())
        kind = self.kind
        if not isinstance(kind, NonSemanticChurnKind):
            try:
                kind = NonSemanticChurnKind(kind)
            except (TypeError, ValueError) as exc:
                raise ProgramContractDeltaError(
                    f"unknown non-semantic churn kind: {self.kind!r}"
                ) from exc
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True)
class StructuralSurfaceChange:
    """Explicit class/method/field/factory intro or removal for a subject."""

    surface: StructuralSurfaceKind
    introduced: bool
    symbol_id: str
    reason: str = ""

    def __post_init__(self) -> None:
        surface = self.surface
        if not isinstance(surface, StructuralSurfaceKind):
            try:
                surface = StructuralSurfaceKind(surface)
            except (TypeError, ValueError) as exc:
                raise ProgramContractDeltaError(
                    f"unknown structural surface: {self.surface!r}"
                ) from exc
        object.__setattr__(self, "surface", surface)
        if not isinstance(self.introduced, bool):
            raise ProgramContractDeltaError("introduced must be a boolean")
        symbol_id = str(self.symbol_id or "").strip()
        if not symbol_id or any(ch.isspace() for ch in symbol_id):
            raise ProgramContractDeltaError("surface change requires a compact symbol_id")
        object.__setattr__(self, "symbol_id", symbol_id)
        object.__setattr__(self, "reason", str(self.reason or "").strip())


@dataclass(frozen=True)
class MovePair:
    """A pure path move of the same logical subject."""

    before_path: str
    after_path: str

    def __post_init__(self) -> None:
        for name in ("before_path", "after_path"):
            value = PurePosixPath(str(getattr(self, name) or "").strip())
            if not str(value) or value.is_absolute() or ".." in value.parts:
                raise ProgramContractDeltaError(f"{name} must be a relative path")
            object.__setattr__(self, name, value.as_posix())
        if self.before_path == self.after_path:
            raise ProgramContractDeltaError("move pair paths must differ")


@dataclass(frozen=True)
class RenamePair:
    """A pure symbol rename of the same logical subject."""

    before_name: str
    after_name: str

    def __post_init__(self) -> None:
        for name in ("before_name", "after_name"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ProgramContractDeltaError(f"{name} is required")
            object.__setattr__(self, name, value)
        if self.before_name == self.after_name:
            raise ProgramContractDeltaError("rename pair names must differ")


@dataclass(frozen=True)
class NormalizedChangePartition:
    """Separation of non-semantic churn from paths that may yield deltas."""

    semantic_paths: tuple[str, ...]
    non_semantic: tuple[PathChurnClassification, ...]
    move_pairs: tuple[MovePair, ...] = ()
    rename_pairs: tuple[RenamePair, ...] = ()

    @property
    def is_purely_non_semantic(self) -> bool:
        return not self.semantic_paths


@dataclass(frozen=True)
class ProgramContractDeltaRequest:
    """Bound inputs for one before/after semantic contract comparison."""

    roots: PropagationAuthorityRoots
    change_set: ProgramChangeSet
    before: ExpectedProgramContract
    after: ExpectedProgramContract
    consumer_domain: str
    subject_symbol_id: str = ""
    path_churn: tuple[PathChurnClassification, ...] = ()
    move_pairs: tuple[MovePair, ...] = ()
    rename_pairs: tuple[RenamePair, ...] = ()
    surface_changes: tuple[StructuralSurfaceChange, ...] = ()
    memory_facet_before_ref: str = ""
    memory_facet_after_ref: str = ""
    before_stale: bool = False
    after_stale: bool = False
    incomplete: bool = False
    reexport_paths: tuple[str, ...] = ()
    registration_changed: bool | None = None
    cancellation_before: str = ""
    cancellation_after: str = ""
    evidence_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise ProgramContractDeltaError("roots must be PropagationAuthorityRoots")
        if not isinstance(self.change_set, ProgramChangeSet):
            raise ProgramContractDeltaError("change_set must be ProgramChangeSet")
        if not isinstance(self.before, ExpectedProgramContract):
            raise ProgramContractDeltaError("before must be ExpectedProgramContract")
        if not isinstance(self.after, ExpectedProgramContract):
            raise ProgramContractDeltaError("after must be ExpectedProgramContract")
        domain = str(self.consumer_domain or "").strip()
        if not domain or any(ch.isspace() for ch in domain):
            raise ProgramContractDeltaError(
                "consumer_domain is required and must be a compact identifier"
            )
        object.__setattr__(self, "consumer_domain", domain)
        subject = str(self.subject_symbol_id or "").strip()
        if subject and any(ch.isspace() for ch in subject):
            raise ProgramContractDeltaError("subject_symbol_id must be compact")
        object.__setattr__(self, "subject_symbol_id", subject)
        object.__setattr__(
            self,
            "path_churn",
            tuple(self.path_churn or ()),
        )
        object.__setattr__(self, "move_pairs", tuple(self.move_pairs or ()))
        object.__setattr__(self, "rename_pairs", tuple(self.rename_pairs or ()))
        object.__setattr__(
            self, "surface_changes", tuple(self.surface_changes or ())
        )
        for flag_name in ("before_stale", "after_stale", "incomplete"):
            if not isinstance(getattr(self, flag_name), bool):
                raise ProgramContractDeltaError(f"{flag_name} must be a boolean")
        object.__setattr__(
            self,
            "reexport_paths",
            tuple(str(p).strip() for p in (self.reexport_paths or ()) if str(p).strip()),
        )
        if self.registration_changed is not None and not isinstance(
            self.registration_changed, bool
        ):
            raise ProgramContractDeltaError("registration_changed must be a boolean or None")
        object.__setattr__(
            self, "cancellation_before", str(self.cancellation_before or "").strip()
        )
        object.__setattr__(
            self, "cancellation_after", str(self.cancellation_after or "").strip()
        )
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(
                str(item).strip()
                for item in (self.evidence_refs or ())
                if str(item).strip()
            ),
        )
        object.__setattr__(
            self,
            "proof_refs",
            tuple(
                str(item).strip() for item in (self.proof_refs or ()) if str(item).strip()
            ),
        )
        object.__setattr__(
            self,
            "memory_facet_before_ref",
            str(self.memory_facet_before_ref or "").strip(),
        )
        object.__setattr__(
            self,
            "memory_facet_after_ref",
            str(self.memory_facet_after_ref or "").strip(),
        )


@dataclass(frozen=True)
class ProgramContractDeltaAnalysis:
    """Full analyzer result: partition plus zero or more typed deltas."""

    partition: NormalizedChangePartition
    deltas: tuple[ProgramContractDelta, ...]
    consumer_domain: str
    subject_symbol_id: str
    pure_non_semantic: bool = False

    @property
    def primary_delta(self) -> ProgramContractDelta | None:
        return self.deltas[0] if self.deltas else None

    @property
    def has_breaking_clauses(self) -> bool:
        return any(delta.breaking_clauses for delta in self.deltas)

    @property
    def all_clauses(self) -> tuple[ContractClauseDelta, ...]:
        clauses: list[ContractClauseDelta] = []
        for delta in self.deltas:
            clauses.extend(delta.clauses)
        return tuple(clauses)


def _text_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ProgramContractDeltaError(f"{field_name} must be a string")
    result = value.strip()
    if not result or any(ch.isspace() for ch in result):
        raise ProgramContractDeltaError(f"{field_name} must be a compact identifier")
    return result


def _infer_path_churn_kind(path: str) -> NonSemanticChurnKind | None:
    lowered = path.lower()
    for marker in _GENERATED_PATH_MARKERS:
        if marker in lowered:
            return NonSemanticChurnKind.GENERATED
    for suffix in _COMMENT_ONLY_SUFFIXES:
        if lowered.endswith(suffix):
            return NonSemanticChurnKind.COMMENT
    for suffix in _FORMATTING_SUFFIXES:
        if lowered.endswith(suffix):
            return NonSemanticChurnKind.FORMATTING
    basename = PurePosixPath(path).name.lower()
    if basename in {"license", "copying", "authors", "changelog"}:
        return NonSemanticChurnKind.COMMENT
    return None


def normalize_change_partition(
    change_set: ProgramChangeSet,
    *,
    path_churn: Sequence[PathChurnClassification] = (),
    move_pairs: Sequence[MovePair] = (),
    rename_pairs: Sequence[RenamePair] = (),
) -> NormalizedChangePartition:
    """Partition change-set paths into semantic vs non-semantic churn.

    Explicit producer classifications win.  Paths listed in
    ``change_set.generated_manifest_ids`` are treated as generated only when
    they are also changed paths (manifest ids that are not paths are ignored
    for path partitioning).  Move/rename pairs remove both endpoints from the
    semantic set when the remaining edit is pure relocation/identity.
    """

    if not isinstance(change_set, ProgramChangeSet):
        raise ProgramContractDeltaError("change_set must be ProgramChangeSet")

    explicit: dict[str, NonSemanticChurnKind] = {}
    for item in path_churn or ():
        if not isinstance(item, PathChurnClassification):
            raise ProgramContractDeltaError(
                "path_churn entries must be PathChurnClassification"
            )
        if item.path not in change_set.changed_paths and item.path not in change_set.tombstone_paths:
            # Explicit classification for an unrelated path is fail-closed noise.
            raise ProgramContractDeltaError(
                f"path churn classification for unknown path: {item.path}"
            )
        explicit[item.path] = item.kind

    moved_paths: set[str] = set()
    normalized_moves: list[MovePair] = []
    for pair in move_pairs or ():
        if not isinstance(pair, MovePair):
            raise ProgramContractDeltaError("move_pairs entries must be MovePair")
        normalized_moves.append(pair)
        moved_paths.add(pair.before_path)
        moved_paths.add(pair.after_path)

    renamed_names: set[str] = set()
    normalized_renames: list[RenamePair] = []
    for pair in rename_pairs or ():
        if not isinstance(pair, RenamePair):
            raise ProgramContractDeltaError("rename_pairs entries must be RenamePair")
        normalized_renames.append(pair)
        renamed_names.add(pair.before_name)
        renamed_names.add(pair.after_name)

    non_semantic: list[PathChurnClassification] = []
    semantic: list[str] = []
    all_paths = sorted(set(change_set.changed_paths) | set(change_set.tombstone_paths))
    generated_ids = set(change_set.generated_manifest_ids)

    for path in all_paths:
        if path in moved_paths:
            non_semantic.append(
                PathChurnClassification(path=path, kind=NonSemanticChurnKind.MOVE)
            )
            continue
        if path in explicit:
            non_semantic.append(
                PathChurnClassification(path=path, kind=explicit[path])
            )
            continue
        if path in generated_ids:
            non_semantic.append(
                PathChurnClassification(path=path, kind=NonSemanticChurnKind.GENERATED)
            )
            continue
        inferred = _infer_path_churn_kind(path)
        if inferred is not None:
            non_semantic.append(PathChurnClassification(path=path, kind=inferred))
            continue
        semantic.append(path)

    # Rename pairs without path moves are recorded but do not remove paths by
    # name alone; symbol rename is handled at the contract layer.
    return NormalizedChangePartition(
        semantic_paths=tuple(sorted(semantic)),
        non_semantic=tuple(sorted(non_semantic, key=lambda item: (item.path, item.kind.value))),
        move_pairs=tuple(sorted(normalized_moves, key=lambda item: (item.before_path, item.after_path))),
        rename_pairs=tuple(
            sorted(normalized_renames, key=lambda item: (item.before_name, item.after_name))
        ),
    )


def _assert_expectation_authority(contract: ExpectedProgramContract, side: str) -> None:
    if not contract.sources:
        raise SelfAuthoredExpectationError(
            f"{side} contract lacks reviewed expectation sources"
        )
    for source in contract.sources:
        if source.source_kind is ContractSourceKind.IMPLEMENTATION_OBSERVATION:
            raise SelfAuthoredExpectationError(
                f"{side} contract cannot use implementation observation as expectation"
            )
        if not source.source_kind.may_define_expectation:
            raise SelfAuthoredExpectationError(
                f"{side} contract source {source.source_kind.value} cannot define expectations"
            )


def _assert_roots_and_binding(request: ProgramContractDeltaRequest) -> None:
    roots = request.roots
    change_set = request.change_set
    if change_set.roots.content_id != roots.content_id:
        raise CrossRootContractDeltaError(
            "change set roots do not match analysis roots"
        )
    before = request.before
    after = request.after
    if before.symbol.repository_id != roots.repository_id:
        raise CrossRootContractDeltaError(
            "before contract repository_id does not match roots"
        )
    if after.symbol.repository_id != roots.repository_id:
        raise CrossRootContractDeltaError(
            "after contract repository_id does not match roots"
        )
    if before.symbol.tree_id != roots.base_tree_id:
        raise CrossRootContractDeltaError(
            "before contract tree_id must equal base_tree_id"
        )
    if after.symbol.tree_id != roots.candidate_tree_id:
        raise CrossRootContractDeltaError(
            "after contract tree_id must equal candidate_tree_id"
        )
    if before.policy_revision != after.policy_revision:
        # Policy drift is not a silent rewrite; fail closed for exact compare.
        raise CrossRootContractDeltaError(
            "before and after contracts bind different policy revisions"
        )
    if request.before_stale or request.after_stale:
        raise StaleContractDeltaError("stale contract evidence cannot be compared")
    if request.incomplete:
        raise IncompleteContractDeltaError("incomplete comparison inputs")
    if before.has_conflicts or after.has_conflicts:
        raise IncompleteContractDeltaError(
            "contracts with unresolved source conflicts cannot be compared"
        )
    _assert_expectation_authority(before, "before")
    _assert_expectation_authority(after, "after")


def _logical_symbol_id(
    before: SymbolIdentity, after: SymbolIdentity, explicit: str
) -> str:
    if explicit:
        return _text_id(explicit, "subject_symbol_id")
    # Prefer stable qualified name (without tree) so base/candidate differ only
    # by tree binding while the logical subject remains comparable.
    name = after.qualified_name or before.qualified_name
    if not name:
        name = after.symbol_name or before.symbol_name
    if not name:
        raise IncompleteContractDeltaError("subject symbol identity is incomplete")
    # Compact opaque identifier without whitespace.
    compact = name.replace(" ", "_")
    if ":" not in compact:
        compact = f"symbol:{compact}"
    return compact


def _aspect_unsupported(
    contract: ExpectedProgramContract, aspect: SemanticAspect
) -> UnsupportedSemantics | None:
    for item in contract.unsupported:
        if item.aspect is aspect:
            return item
    return None


def _clause(
    *,
    clause_id: str,
    kind: DeltaKind,
    disposition: DeltaDisposition,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    reason: str,
) -> ContractClauseDelta:
    return ContractClauseDelta(
        clause_id=clause_id,
        kind=kind,
        disposition=disposition,
        subject_symbol_id=subject_symbol_id,
        consumer_domain=consumer_domain,
        before_contract_ref=before_ref,
        after_contract_ref=after_ref,
        reason=reason[:4000],
    )


def _param_key(param: ParameterSpec) -> str:
    if param.name:
        return f"name:{param.name}"
    if param.position is not None:
        return f"pos:{param.position}"
    return f"id:{param.parameter_id}"


def _match_parameters(
    before_params: Sequence[ParameterSpec],
    after_params: Sequence[ParameterSpec],
) -> tuple[
    list[tuple[ParameterSpec, ParameterSpec, str]],
    list[ParameterSpec],
    list[ParameterSpec],
]:
    """Return (matched pairs with match_kind, removed, added).

    match_kind is one of: name, position, rename.
    """

    before_by_name = {p.name: p for p in before_params if p.name}
    after_by_name = {p.name: p for p in after_params if p.name}
    before_by_pos = {
        p.position: p for p in before_params if p.position is not None
    }
    after_by_pos = {
        p.position: p for p in after_params if p.position is not None
    }

    matched: list[tuple[ParameterSpec, ParameterSpec, str]] = []
    used_before: set[str] = set()
    used_after: set[str] = set()

    # 1) exact name matches
    for name, bp in before_by_name.items():
        ap = after_by_name.get(name)
        if ap is None:
            continue
        matched.append((bp, ap, "name"))
        used_before.add(_param_key(bp))
        used_after.add(_param_key(ap))

    # 2) same position, different name → rename candidate
    for pos, bp in before_by_pos.items():
        if _param_key(bp) in used_before:
            continue
        ap = after_by_pos.get(pos)
        if ap is None or _param_key(ap) in used_after:
            continue
        if bp.name != ap.name:
            # Prefer rename when types are compatible enough to be the same slot.
            matched.append((bp, ap, "rename"))
            used_before.add(_param_key(bp))
            used_after.add(_param_key(ap))
        else:
            matched.append((bp, ap, "position"))
            used_before.add(_param_key(bp))
            used_after.add(_param_key(ap))

    removed = [p for p in before_params if _param_key(p) not in used_before]
    added = [p for p in after_params if _param_key(p) not in used_after]
    return matched, removed, added


def _type_unsupported(shape: TypeShape | None) -> bool:
    if shape is None:
        return False
    return (
        shape.support is SupportStatus.UNSUPPORTED
        or shape.constructor is TypeConstructor.UNSUPPORTED
        or shape.constructor is TypeConstructor.UNKNOWN
    )


def _input_type_disposition(
    before: TypeShape, after: TypeShape, domain: str
) -> DeltaDisposition:
    """Contravariant input: after must accept everything before accepted."""

    if _type_unsupported(before) or _type_unsupported(after):
        return DeltaDisposition.UNSUPPORTED
    if before.content_id == after.content_id:
        return DeltaDisposition.COMPATIBLE
    # After is wider acceptor if before <: after (before values still accepted).
    if before.is_subtype_of(after) and not after.is_subtype_of(before):
        return DeltaDisposition.COMPATIBLE
    if after.is_subtype_of(before) and not before.is_subtype_of(after):
        return DeltaDisposition.BREAKING
    if before.is_subtype_of(after) and after.is_subtype_of(before):
        return DeltaDisposition.COMPATIBLE
    if domain in {DOMAIN_SCHEMA_CONSUMERS, DOMAIN_SERIALIZERS, DOMAIN_HTTP_CLIENTS}:
        return DeltaDisposition.BREAKING
    return DeltaDisposition.BREAKING


def _output_type_disposition(
    before: TypeShape, after: TypeShape, domain: str
) -> DeltaDisposition:
    """Covariant output: after must be subtype of before for callers."""

    if _type_unsupported(before) or _type_unsupported(after):
        return DeltaDisposition.UNSUPPORTED
    if before.content_id == after.content_id:
        return DeltaDisposition.COMPATIBLE
    if after.is_subtype_of(before) and not before.is_subtype_of(after):
        return DeltaDisposition.COMPATIBLE
    if before.is_subtype_of(after) and not after.is_subtype_of(before):
        # Wider result is often behavioral for schema, breaking for typed callers
        # that relied on the narrower shape? Actually wider result is compatible
        # for callers (they still get a before-type value). Fail-closed: if not
        # mutual subtype and not after <: before, treat as breaking.
        return DeltaDisposition.BREAKING
    if before.is_subtype_of(after) and after.is_subtype_of(before):
        return DeltaDisposition.COMPATIBLE
    return DeltaDisposition.BREAKING


def _default_disposition(before: ParameterSpec, after: ParameterSpec, domain: str) -> DeltaDisposition:
    if before.default_summary == after.default_summary:
        return DeltaDisposition.COMPATIBLE
    if not before.default_summary and after.default_summary:
        # Adding a default to a required param may demote optionality elsewhere;
        # still behavioral for callers that omit the arg only after the change.
        return DeltaDisposition.BEHAVIORAL
    if before.default_summary and not after.default_summary:
        return DeltaDisposition.BREAKING
    # Default expression changed.
    if domain in {DOMAIN_TESTS, DOMAIN_SCHEMA_CONSUMERS}:
        return DeltaDisposition.BEHAVIORAL
    return DeltaDisposition.BEHAVIORAL


def _compare_parameters(
    before: ExpectedProgramContract,
    after: ExpectedProgramContract,
    *,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
) -> None:
    unsup_b = _aspect_unsupported(before, SemanticAspect.INPUTS)
    unsup_a = _aspect_unsupported(after, SemanticAspect.INPUTS)
    if unsup_b is not None or unsup_a is not None:
        reason = (unsup_b or unsup_a).reason  # type: ignore[union-attr]
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:inputs-unsupported-{clause_counter[0]}",
                kind=DeltaKind.PARAMETER_VARIANCE,
                disposition=DeltaDisposition.UNSUPPORTED,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"input domain unsupported: {reason}",
            )
        )
        return

    matched, removed, added = _match_parameters(before.inputs, after.inputs)

    for param in removed:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:param-remove-{param.name or clause_counter[0]}",
                kind=DeltaKind.PARAMETER_REMOVE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"parameter {param.name!r} removed",
            )
        )

    for param in added:
        clause_counter[0] += 1
        if param.optionality is Optionality.REQUIRED and not param.default_summary:
            disposition = DeltaDisposition.BREAKING
            reason = f"required parameter {param.name!r} added without default"
        elif param.optionality is Optionality.OPTIONAL or param.default_summary:
            # Optional/defaulted addition is source-compatible for most callers
            # but schema/serializer domains still see a surface change.
            if consumer_domain in {
                DOMAIN_SCHEMA_CONSUMERS,
                DOMAIN_SERIALIZERS,
                DOMAIN_HTTP_CLIENTS,
            }:
                disposition = DeltaDisposition.BEHAVIORAL
            else:
                disposition = DeltaDisposition.COMPATIBLE
            reason = f"optional or defaulted parameter {param.name!r} added"
        else:
            disposition = DeltaDisposition.UNKNOWN
            reason = f"parameter {param.name!r} added with unknown optionality"
        clauses.append(
            _clause(
                clause_id=f"clause:param-add-{param.name or clause_counter[0]}",
                kind=DeltaKind.PARAMETER_ADD,
                disposition=disposition,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=reason,
            )
        )

    for bp, ap, match_kind in matched:
        # Rename
        if match_kind == "rename" or (bp.name and ap.name and bp.name != ap.name):
            clause_counter[0] += 1
            # Keyword-domain callers break on renames; pure positional may be
            # behavioral if types and position match.
            if bp.kind is ParameterKind.KEYWORD or ap.kind is ParameterKind.KEYWORD:
                disposition = DeltaDisposition.BREAKING
            elif consumer_domain in {DOMAIN_SCHEMA_CONSUMERS, DOMAIN_SERIALIZERS}:
                disposition = DeltaDisposition.BREAKING
            else:
                disposition = DeltaDisposition.BREAKING
            clauses.append(
                _clause(
                    clause_id=f"clause:param-rename-{bp.name}-{ap.name}",
                    kind=DeltaKind.PARAMETER_RENAME,
                    disposition=disposition,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason=f"parameter renamed {bp.name!r} -> {ap.name!r}",
                )
            )

        # Reorder (same name, position changed)
        if (
            bp.name
            and ap.name
            and bp.name == ap.name
            and bp.position is not None
            and ap.position is not None
            and bp.position != ap.position
        ):
            clause_counter[0] += 1
            if bp.kind is ParameterKind.KEYWORD and ap.kind is ParameterKind.KEYWORD:
                disposition = DeltaDisposition.COMPATIBLE
            elif consumer_domain == DOMAIN_PYTHON_CALLERS:
                disposition = DeltaDisposition.BEHAVIORAL
            else:
                disposition = DeltaDisposition.COMPATIBLE
            clauses.append(
                _clause(
                    clause_id=f"clause:param-reorder-{bp.name}",
                    kind=DeltaKind.PARAMETER_REORDER,
                    disposition=disposition,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason=(
                        f"parameter {bp.name!r} reordered "
                        f"{bp.position} -> {ap.position}"
                    ),
                )
            )

        # Default
        if bp.default_summary != ap.default_summary:
            clause_counter[0] += 1
            clauses.append(
                _clause(
                    clause_id=f"clause:param-default-{ap.name or bp.name}",
                    kind=DeltaKind.PARAMETER_DEFAULT,
                    disposition=_default_disposition(bp, ap, consumer_domain),
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason=(
                        f"parameter {ap.name or bp.name!r} default changed "
                        f"{bp.default_summary!r} -> {ap.default_summary!r}"
                    ),
                )
            )

        # Keyword / kind change
        if bp.kind is not ap.kind:
            clause_counter[0] += 1
            disposition = DeltaDisposition.BREAKING
            if (
                bp.kind is ParameterKind.POSITIONAL
                and ap.kind is ParameterKind.KEYWORD
            ):
                # Forcing keyword-only can break positional callers.
                disposition = DeltaDisposition.BREAKING
            elif (
                bp.kind is ParameterKind.KEYWORD
                and ap.kind is ParameterKind.POSITIONAL
            ):
                disposition = DeltaDisposition.COMPATIBLE
            clauses.append(
                _clause(
                    clause_id=f"clause:param-keyword-{ap.name or bp.name}",
                    kind=DeltaKind.PARAMETER_KEYWORD,
                    disposition=disposition,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason=(
                        f"parameter {ap.name or bp.name!r} calling convention "
                        f"{bp.kind.value} -> {ap.kind.value}"
                    ),
                )
            )

        # Optionality change (requiredness)
        if bp.optionality is not ap.optionality:
            clause_counter[0] += 1
            if (
                bp.optionality is not Optionality.REQUIRED
                and ap.optionality is Optionality.REQUIRED
            ):
                disposition = DeltaDisposition.BREAKING
                reason = f"parameter {ap.name!r} became required"
            elif (
                bp.optionality is Optionality.REQUIRED
                and ap.optionality is not Optionality.REQUIRED
            ):
                disposition = DeltaDisposition.COMPATIBLE
                reason = f"parameter {ap.name!r} became optional"
            else:
                disposition = DeltaDisposition.BEHAVIORAL
                reason = (
                    f"parameter {ap.name!r} optionality "
                    f"{bp.optionality.value} -> {ap.optionality.value}"
                )
            clauses.append(
                _clause(
                    clause_id=f"clause:param-optionality-{ap.name or bp.name}",
                    kind=DeltaKind.PARAMETER_VARIANCE,
                    disposition=disposition,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason=reason,
                )
            )

        # Type variance / nullability / generic
        if bp.type_shape.content_id != ap.type_shape.content_id:
            if bp.type_shape.nullable != ap.type_shape.nullable:
                clause_counter[0] += 1
                if ap.type_shape.nullable and not bp.type_shape.nullable:
                    # Input accepts null now → wider → compatible for callers.
                    null_disp = DeltaDisposition.COMPATIBLE
                else:
                    null_disp = DeltaDisposition.BREAKING
                clauses.append(
                    _clause(
                        clause_id=f"clause:param-nullability-{ap.name or bp.name}",
                        kind=DeltaKind.NULLABILITY_CHANGE,
                        disposition=null_disp,
                        subject_symbol_id=subject_symbol_id,
                        consumer_domain=consumer_domain,
                        before_ref=before_ref,
                        after_ref=after_ref,
                        reason=(
                            f"parameter {ap.name or bp.name!r} nullability "
                            f"{bp.type_shape.nullable} -> {ap.type_shape.nullable}"
                        ),
                    )
                )
            generic_constructors = {
                TypeConstructor.ARRAY,
                TypeConstructor.OBJECT,
                TypeConstructor.UNION,
                TypeConstructor.INTERSECTION,
                TypeConstructor.REFERENCE,
            }
            if (
                bp.type_shape.constructor in generic_constructors
                or ap.type_shape.constructor in generic_constructors
            ) and (
                bp.type_shape.constructor != ap.type_shape.constructor
                or (bp.type_shape.item is None) != (ap.type_shape.item is None)
                or (
                    bp.type_shape.item is not None
                    and ap.type_shape.item is not None
                    and bp.type_shape.item.content_id != ap.type_shape.item.content_id
                )
                or bp.type_shape.fields != ap.type_shape.fields
                or bp.type_shape.reference != ap.type_shape.reference
            ):
                clause_counter[0] += 1
                clauses.append(
                    _clause(
                        clause_id=f"clause:param-generic-{ap.name or bp.name}",
                        kind=DeltaKind.GENERIC_CHANGE,
                        disposition=_input_type_disposition(
                            bp.type_shape, ap.type_shape, consumer_domain
                        ),
                        subject_symbol_id=subject_symbol_id,
                        consumer_domain=consumer_domain,
                        before_ref=before_ref,
                        after_ref=after_ref,
                        reason=f"parameter {ap.name or bp.name!r} generic/structure changed",
                    )
                )
            # Always emit variance when type identity differs.
            clause_counter[0] += 1
            clauses.append(
                _clause(
                    clause_id=f"clause:param-variance-{ap.name or bp.name}",
                    kind=DeltaKind.PARAMETER_VARIANCE,
                    disposition=_input_type_disposition(
                        bp.type_shape, ap.type_shape, consumer_domain
                    ),
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason=f"parameter {ap.name or bp.name!r} type variance changed",
                )
            )


def _compare_returns(
    before: ExpectedProgramContract,
    after: ExpectedProgramContract,
    *,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
) -> None:
    unsup_b = _aspect_unsupported(before, SemanticAspect.OUTPUTS)
    unsup_a = _aspect_unsupported(after, SemanticAspect.OUTPUTS)
    if unsup_b is not None or unsup_a is not None:
        reason = (unsup_b or unsup_a).reason  # type: ignore[union-attr]
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:result-unsupported-{clause_counter[0]}",
                kind=DeltaKind.RESULT_CHANGE,
                disposition=DeltaDisposition.UNSUPPORTED,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"result domain unsupported: {reason}",
            )
        )
        return

    br = before.returns
    ar = after.returns
    if br is None and ar is None:
        return
    if br is None and ar is not None:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:result-intro-{clause_counter[0]}",
                kind=DeltaKind.RESULT_CHANGE,
                disposition=DeltaDisposition.BEHAVIORAL,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason="result specification introduced",
            )
        )
        return
    if br is not None and ar is None:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:result-remove-{clause_counter[0]}",
                kind=DeltaKind.RESULT_CHANGE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason="result specification removed",
            )
        )
        return
    assert br is not None and ar is not None
    if br.content_id == ar.content_id:
        return

    if br.optionality is not ar.optionality:
        clause_counter[0] += 1
        if (
            br.optionality is Optionality.REQUIRED
            and ar.optionality is not Optionality.REQUIRED
        ):
            disp = DeltaDisposition.BREAKING
        else:
            disp = DeltaDisposition.COMPATIBLE
        clauses.append(
            _clause(
                clause_id=f"clause:result-optionality-{clause_counter[0]}",
                kind=DeltaKind.NULLABILITY_CHANGE,
                disposition=disp,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=(
                    f"result optionality {br.optionality.value} -> {ar.optionality.value}"
                ),
            )
        )

    if br.type_shape.nullable != ar.type_shape.nullable:
        clause_counter[0] += 1
        # Output became nullable → breaking for callers expecting non-null.
        if ar.type_shape.nullable and not br.type_shape.nullable:
            disp = DeltaDisposition.BREAKING
        else:
            disp = DeltaDisposition.COMPATIBLE
        clauses.append(
            _clause(
                clause_id=f"clause:result-nullability-{clause_counter[0]}",
                kind=DeltaKind.NULLABILITY_CHANGE,
                disposition=disp,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=(
                    f"result nullability {br.type_shape.nullable} -> {ar.type_shape.nullable}"
                ),
            )
        )

    if br.type_shape.content_id != ar.type_shape.content_id:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:result-type-{clause_counter[0]}",
                kind=DeltaKind.RESULT_CHANGE,
                disposition=_output_type_disposition(
                    br.type_shape, ar.type_shape, consumer_domain
                ),
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason="result type changed",
            )
        )
        generic_constructors = {
            TypeConstructor.ARRAY,
            TypeConstructor.OBJECT,
            TypeConstructor.UNION,
            TypeConstructor.INTERSECTION,
            TypeConstructor.REFERENCE,
        }
        if (
            br.type_shape.constructor in generic_constructors
            or ar.type_shape.constructor in generic_constructors
        ):
            clause_counter[0] += 1
            clauses.append(
                _clause(
                    clause_id=f"clause:result-generic-{clause_counter[0]}",
                    kind=DeltaKind.GENERIC_CHANGE,
                    disposition=_output_type_disposition(
                        br.type_shape, ar.type_shape, consumer_domain
                    ),
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason="result generic/structure changed",
                )
            )


def _compare_interface_protocol_schema(
    before: ExpectedProgramContract,
    after: ExpectedProgramContract,
    *,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
) -> None:
    bi = before.interface
    ai = after.interface
    if bi.protocol != ai.protocol:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:protocol-{clause_counter[0]}",
                kind=DeltaKind.PROTOCOL_CHANGE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"protocol changed {bi.protocol!r} -> {ai.protocol!r}",
            )
        )
    if bi.media_type != ai.media_type:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:serialization-{clause_counter[0]}",
                kind=DeltaKind.SERIALIZATION_CHANGE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"media_type/serialization changed {bi.media_type!r} -> {ai.media_type!r}",
            )
        )
    if (
        bi.path_or_uri != ai.path_or_uri
        or bi.version != ai.version
        or bi.method != ai.method
        or bi.surface != ai.surface
        or bi.interface_name != ai.interface_name
    ):
        # Surface identity drift is a schema/protocol consumer concern.
        if not bi.binds_same_surface(ai):
            clause_counter[0] += 1
            clauses.append(
                _clause(
                    clause_id=f"clause:schema-surface-{clause_counter[0]}",
                    kind=DeltaKind.SCHEMA_CHANGE,
                    disposition=DeltaDisposition.BREAKING,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason="interface surface identity changed",
                )
            )
        elif bi.version != ai.version or bi.path_or_uri != ai.path_or_uri:
            clause_counter[0] += 1
            disp = (
                DeltaDisposition.BEHAVIORAL
                if consumer_domain in {DOMAIN_PYTHON_CALLERS, DOMAIN_TESTS}
                else DeltaDisposition.BREAKING
            )
            clauses.append(
                _clause(
                    clause_id=f"clause:schema-version-{clause_counter[0]}",
                    kind=DeltaKind.SCHEMA_CHANGE,
                    disposition=disp,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason=(
                        f"schema path/version changed "
                        f"{bi.path_or_uri!r}@{bi.version!r} -> "
                        f"{ai.path_or_uri!r}@{ai.version!r}"
                    ),
                )
            )

    # Object field schema on returns
    if (
        before.returns is not None
        and after.returns is not None
        and before.returns.type_shape.constructor is TypeConstructor.OBJECT
        and after.returns.type_shape.constructor is TypeConstructor.OBJECT
        and before.returns.type_shape.fields != after.returns.type_shape.fields
    ):
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:schema-fields-{clause_counter[0]}",
                kind=DeltaKind.SCHEMA_CHANGE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason="object schema fields changed",
            )
        )


def _compare_sync_async_cancellation(
    before: ExpectedProgramContract,
    after: ExpectedProgramContract,
    *,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
    cancellation_before: str,
    cancellation_after: str,
) -> None:
    unsup_b = _aspect_unsupported(before, SemanticAspect.SYNC_ASYNC)
    unsup_a = _aspect_unsupported(after, SemanticAspect.SYNC_ASYNC)
    if unsup_b is not None or unsup_a is not None:
        reason = (unsup_b or unsup_a).reason  # type: ignore[union-attr]
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:sync-unsupported-{clause_counter[0]}",
                kind=DeltaKind.SYNC_ASYNC_CHANGE,
                disposition=DeltaDisposition.UNSUPPORTED,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"sync/async unsupported: {reason}",
            )
        )
    else:
        bs = before.sync_async
        as_ = after.sync_async
        if bs is not None and as_ is not None and bs.content_id != as_.content_id:
            clause_counter[0] += 1
            if as_.is_compatible_with(bs) and bs.is_compatible_with(as_):
                disp = DeltaDisposition.COMPATIBLE
            elif as_.is_compatible_with(bs):
                disp = DeltaDisposition.BEHAVIORAL
            else:
                disp = DeltaDisposition.BREAKING
            clauses.append(
                _clause(
                    clause_id=f"clause:sync-async-{clause_counter[0]}",
                    kind=DeltaKind.SYNC_ASYNC_CHANGE,
                    disposition=disp,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason=(
                        f"sync/async changed {bs.mode.value} -> {as_.mode.value}"
                    ),
                )
            )
        elif (bs is None) != (as_ is None):
            clause_counter[0] += 1
            clauses.append(
                _clause(
                    clause_id=f"clause:sync-async-presence-{clause_counter[0]}",
                    kind=DeltaKind.SYNC_ASYNC_CHANGE,
                    disposition=DeltaDisposition.BEHAVIORAL,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason="sync/async specification presence changed",
                )
            )

    if cancellation_before != cancellation_after:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:cancellation-{clause_counter[0]}",
                kind=DeltaKind.CANCELLATION_CHANGE,
                disposition=DeltaDisposition.BREAKING
                if cancellation_before or cancellation_after
                else DeltaDisposition.COMPATIBLE,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=(
                    f"cancellation semantics changed "
                    f"{cancellation_before!r} -> {cancellation_after!r}"
                ),
            )
        )


def _set_key_errors(errors: Sequence[ErrorSpec]) -> set[tuple[str, str]]:
    return {(item.error_name, item.code) for item in errors}


def _compare_errors(
    before: ExpectedProgramContract,
    after: ExpectedProgramContract,
    *,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
) -> None:
    unsup_b = _aspect_unsupported(before, SemanticAspect.ERRORS)
    unsup_a = _aspect_unsupported(after, SemanticAspect.ERRORS)
    if unsup_b is not None or unsup_a is not None:
        reason = (unsup_b or unsup_a).reason  # type: ignore[union-attr]
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:errors-unsupported-{clause_counter[0]}",
                kind=DeltaKind.ERROR_CHANGE,
                disposition=DeltaDisposition.UNSUPPORTED,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"errors unsupported: {reason}",
            )
        )
        return
    before_set = _set_key_errors(before.errors)
    after_set = _set_key_errors(after.errors)
    if before_set == after_set:
        # Still check retriable flag drift for same codes.
        before_map = {(e.error_name, e.code): e for e in before.errors}
        for ae in after.errors:
            be = before_map.get((ae.error_name, ae.code))
            if be is not None and be.retriable != ae.retriable:
                clause_counter[0] += 1
                clauses.append(
                    _clause(
                        clause_id=f"clause:error-retriable-{ae.error_name}",
                        kind=DeltaKind.ERROR_CHANGE,
                        disposition=DeltaDisposition.BEHAVIORAL,
                        subject_symbol_id=subject_symbol_id,
                        consumer_domain=consumer_domain,
                        before_ref=before_ref,
                        after_ref=after_ref,
                        reason=f"error {ae.error_name!r} retriable flag changed",
                    )
                )
        return
    added = after_set - before_set
    removed = before_set - after_set
    if added or removed:
        clause_counter[0] += 1
        # New errors are breaking for callers that must handle them; removed
        # errors are generally compatible for callers.
        if added:
            disp = DeltaDisposition.BREAKING
        else:
            disp = DeltaDisposition.COMPATIBLE
        clauses.append(
            _clause(
                clause_id=f"clause:errors-{clause_counter[0]}",
                kind=DeltaKind.ERROR_CHANGE,
                disposition=disp,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=(
                    f"errors added={sorted(n for n, _ in added)} "
                    f"removed={sorted(n for n, _ in removed)}"
                ),
            )
        )


def _effect_keys(
    effects: Sequence[SideEffectSpec],
) -> set[tuple[EffectKind, EffectPolarity, str]]:
    return {(e.effect_kind, e.polarity, e.target) for e in effects}


def _compare_effects(
    before: ExpectedProgramContract,
    after: ExpectedProgramContract,
    *,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
) -> None:
    unsup_b = _aspect_unsupported(before, SemanticAspect.SIDE_EFFECTS)
    unsup_a = _aspect_unsupported(after, SemanticAspect.SIDE_EFFECTS)
    if unsup_b is not None or unsup_a is not None:
        reason = (unsup_b or unsup_a).reason  # type: ignore[union-attr]
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:effects-unsupported-{clause_counter[0]}",
                kind=DeltaKind.EFFECT_CHANGE,
                disposition=DeltaDisposition.UNSUPPORTED,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"effects unsupported: {reason}",
            )
        )
        return
    if _effect_keys(before.side_effects) == _effect_keys(after.side_effects):
        return
    before_required = {
        e.effect_kind
        for e in before.side_effects
        if e.polarity is EffectPolarity.REQUIRED
    }
    after_required = {
        e.effect_kind
        for e in after.side_effects
        if e.polarity is EffectPolarity.REQUIRED
    }
    before_forbidden = {
        e.effect_kind
        for e in before.side_effects
        if e.polarity is EffectPolarity.FORBIDDEN
    }
    after_allowed = {
        e.effect_kind
        for e in after.side_effects
        if e.polarity in {EffectPolarity.ALLOWED, EffectPolarity.REQUIRED}
    }
    # Newly allowed effect that was forbidden, or new required effect.
    if after_allowed & before_forbidden or (after_required - before_required):
        disp = DeltaDisposition.BREAKING
    else:
        disp = DeltaDisposition.BEHAVIORAL
    clause_counter[0] += 1
    clauses.append(
        _clause(
            clause_id=f"clause:effects-{clause_counter[0]}",
            kind=DeltaKind.EFFECT_CHANGE,
            disposition=disp,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=consumer_domain,
            before_ref=before_ref,
            after_ref=after_ref,
            reason="side-effect policy changed",
        )
    )


def _compare_capabilities(
    before: ExpectedProgramContract,
    after: ExpectedProgramContract,
    *,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
) -> None:
    unsup_b = _aspect_unsupported(before, SemanticAspect.CAPABILITIES)
    unsup_a = _aspect_unsupported(after, SemanticAspect.CAPABILITIES)
    if unsup_b is not None or unsup_a is not None:
        reason = (unsup_b or unsup_a).reason  # type: ignore[union-attr]
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:capabilities-unsupported-{clause_counter[0]}",
                kind=DeltaKind.CAPABILITY_CHANGE,
                disposition=DeltaDisposition.UNSUPPORTED,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"capabilities unsupported: {reason}",
            )
        )
        return
    before_req = {
        c.capability_name
        for c in before.capabilities
        if c.mode is CapabilityMode.REQUIRED
    }
    after_req = {
        c.capability_name
        for c in after.capabilities
        if c.mode is CapabilityMode.REQUIRED
    }
    before_all = {(c.capability_name, c.mode) for c in before.capabilities}
    after_all = {(c.capability_name, c.mode) for c in after.capabilities}
    if before_all == after_all:
        return
    clause_counter[0] += 1
    if after_req - before_req:
        disp = DeltaDisposition.BREAKING
        reason = f"new required capabilities: {sorted(after_req - before_req)}"
    elif before_req - after_req:
        disp = DeltaDisposition.COMPATIBLE
        reason = f"required capabilities removed: {sorted(before_req - after_req)}"
    else:
        disp = DeltaDisposition.BEHAVIORAL
        reason = "capability mode set changed"
    clauses.append(
        _clause(
            clause_id=f"clause:capabilities-{clause_counter[0]}",
            kind=DeltaKind.CAPABILITY_CHANGE,
            disposition=disp,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=consumer_domain,
            before_ref=before_ref,
            after_ref=after_ref,
            reason=reason,
        )
    )


def _compare_authorization(
    before: ExpectedProgramContract,
    after: ExpectedProgramContract,
    *,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
) -> None:
    unsup_b = _aspect_unsupported(before, SemanticAspect.AUTHORIZATION)
    unsup_a = _aspect_unsupported(after, SemanticAspect.AUTHORIZATION)
    if unsup_b is not None or unsup_a is not None:
        reason = (unsup_b or unsup_a).reason  # type: ignore[union-attr]
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:auth-unsupported-{clause_counter[0]}",
                kind=DeltaKind.AUTHORIZATION_CHANGE,
                disposition=DeltaDisposition.UNSUPPORTED,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"authorization unsupported: {reason}",
            )
        )
        return
    ba = before.authorization
    aa = after.authorization
    if ba is None and aa is None:
        return
    if ba is None or aa is None:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:auth-presence-{clause_counter[0]}",
                kind=DeltaKind.AUTHORIZATION_CHANGE,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason="authorization specification presence changed",
            )
        )
        return
    if ba.content_id == aa.content_id:
        return
    clause_counter[0] += 1
    # Stronger auth on after is breaking for callers that must satisfy it.
    if aa.is_refinement_of(ba) and not ba.is_refinement_of(aa):
        disp = DeltaDisposition.BREAKING
        reason = "authorization strengthened"
    elif ba.is_refinement_of(aa) and not aa.is_refinement_of(ba):
        disp = DeltaDisposition.COMPATIBLE
        reason = "authorization weakened"
    else:
        disp = DeltaDisposition.BREAKING
        reason = "authorization policy changed incompatibly"
    clauses.append(
        _clause(
            clause_id=f"clause:auth-{clause_counter[0]}",
            kind=DeltaKind.AUTHORIZATION_CHANGE,
            disposition=disp,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=consumer_domain,
            before_ref=before_ref,
            after_ref=after_ref,
            reason=reason,
        )
    )


def _compare_lifecycle_state_consistency_resource(
    before: ExpectedProgramContract,
    after: ExpectedProgramContract,
    *,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
) -> None:
    # Lifecycle: idempotence + atomicity + fallback
    for label, aspect, kind, bval, aval, refine in (
        (
            "idempotence",
            SemanticAspect.IDEMPOTENCE,
            DeltaKind.LIFECYCLE_CHANGE,
            before.idempotence,
            after.idempotence,
            True,
        ),
        (
            "atomicity",
            SemanticAspect.ATOMICITY,
            DeltaKind.LIFECYCLE_CHANGE,
            before.atomicity,
            after.atomicity,
            True,
        ),
        (
            "fallback",
            SemanticAspect.FALLBACK_DEGRADATION,
            DeltaKind.LIFECYCLE_CHANGE,
            before.fallback,
            after.fallback,
            False,
        ),
    ):
        unsup = _aspect_unsupported(before, aspect) or _aspect_unsupported(after, aspect)
        if unsup is not None:
            clause_counter[0] += 1
            clauses.append(
                _clause(
                    clause_id=f"clause:{label}-unsupported-{clause_counter[0]}",
                    kind=kind,
                    disposition=DeltaDisposition.UNSUPPORTED,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason=f"{label} unsupported: {unsup.reason}",
                )
            )
            continue
        if bval is None and aval is None:
            continue
        if bval is None or aval is None:
            clause_counter[0] += 1
            clauses.append(
                _clause(
                    clause_id=f"clause:{label}-presence-{clause_counter[0]}",
                    kind=kind,
                    disposition=DeltaDisposition.BEHAVIORAL,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason=f"{label} presence changed",
                )
            )
            continue
        if getattr(bval, "content_id", None) == getattr(aval, "content_id", None):
            continue
        clause_counter[0] += 1
        if refine and hasattr(aval, "is_refinement_of"):
            if aval.is_refinement_of(bval) and not bval.is_refinement_of(aval):
                disp = DeltaDisposition.COMPATIBLE
            elif bval.is_refinement_of(aval) and not aval.is_refinement_of(bval):
                disp = DeltaDisposition.BREAKING
            else:
                disp = DeltaDisposition.BREAKING
        else:
            disp = (
                DeltaDisposition.COMPATIBLE
                if bval == aval
                else DeltaDisposition.BREAKING
            )
        clauses.append(
            _clause(
                clause_id=f"clause:{label}-{clause_counter[0]}",
                kind=kind,
                disposition=disp,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"{label} changed",
            )
        )

    # Temporal / ordering / state
    unsup = _aspect_unsupported(before, SemanticAspect.ORDERING) or _aspect_unsupported(
        after, SemanticAspect.ORDERING
    )
    if unsup is not None:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:ordering-unsupported-{clause_counter[0]}",
                kind=DeltaKind.TEMPORAL_STATE_CHANGE,
                disposition=DeltaDisposition.UNSUPPORTED,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"ordering unsupported: {unsup.reason}",
            )
        )
    else:
        bo, ao = before.ordering, after.ordering
        if bo is not None and ao is not None and bo.content_id != ao.content_id:
            clause_counter[0] += 1
            clauses.append(
                _clause(
                    clause_id=f"clause:ordering-{clause_counter[0]}",
                    kind=DeltaKind.TEMPORAL_STATE_CHANGE,
                    disposition=DeltaDisposition.BREAKING,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason="ordering/temporal state changed",
                )
            )
        elif (bo is None) != (ao is None):
            clause_counter[0] += 1
            clauses.append(
                _clause(
                    clause_id=f"clause:ordering-presence-{clause_counter[0]}",
                    kind=DeltaKind.TEMPORAL_STATE_CHANGE,
                    disposition=DeltaDisposition.BEHAVIORAL,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason="ordering presence changed",
                )
            )

    # Consistency
    unsup = _aspect_unsupported(
        before, SemanticAspect.CONSISTENCY
    ) or _aspect_unsupported(after, SemanticAspect.CONSISTENCY)
    if unsup is not None:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:consistency-unsupported-{clause_counter[0]}",
                kind=DeltaKind.CONSISTENCY_CHANGE,
                disposition=DeltaDisposition.UNSUPPORTED,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"consistency unsupported: {unsup.reason}",
            )
        )
    else:
        bc, ac = before.consistency, after.consistency
        if bc is not None and ac is not None and bc.content_id != ac.content_id:
            clause_counter[0] += 1
            if ac.is_refinement_of(bc) and not bc.is_refinement_of(ac):
                disp = DeltaDisposition.COMPATIBLE
            else:
                disp = DeltaDisposition.BREAKING
            clauses.append(
                _clause(
                    clause_id=f"clause:consistency-{clause_counter[0]}",
                    kind=DeltaKind.CONSISTENCY_CHANGE,
                    disposition=disp,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason="consistency model changed",
                )
            )

    # Resources
    unsup = _aspect_unsupported(
        before, SemanticAspect.RESOURCE_BOUNDS
    ) or _aspect_unsupported(after, SemanticAspect.RESOURCE_BOUNDS)
    if unsup is not None:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:resource-unsupported-{clause_counter[0]}",
                kind=DeltaKind.RESOURCE_CHANGE,
                disposition=DeltaDisposition.UNSUPPORTED,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"resource bounds unsupported: {unsup.reason}",
            )
        )
    else:
        br, ar = before.resource_bounds, after.resource_bounds
        if br is not None and ar is not None and br.content_id != ar.content_id:
            clause_counter[0] += 1
            if ar.is_refinement_of(br) and not br.is_refinement_of(ar):
                # Tighter bounds refine → may break callers that relied on looser.
                disp = DeltaDisposition.BREAKING
            elif br.is_refinement_of(ar) and not ar.is_refinement_of(br):
                disp = DeltaDisposition.COMPATIBLE
            else:
                disp = DeltaDisposition.BREAKING
            clauses.append(
                _clause(
                    clause_id=f"clause:resource-{clause_counter[0]}",
                    kind=DeltaKind.RESOURCE_CHANGE,
                    disposition=disp,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason="resource bounds changed",
                )
            )
        elif (br is None) != (ar is None):
            clause_counter[0] += 1
            clauses.append(
                _clause(
                    clause_id=f"clause:resource-presence-{clause_counter[0]}",
                    kind=DeltaKind.RESOURCE_CHANGE,
                    disposition=DeltaDisposition.BEHAVIORAL,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=consumer_domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    reason="resource bounds presence changed",
                )
            )


def _compare_applicability_visibility(
    before: ExpectedProgramContract,
    after: ExpectedProgramContract,
    *,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
) -> None:
    ba = before.applicability
    aa = after.applicability
    if ba is None and aa is None:
        return
    if ba is None or aa is None:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:visibility-presence-{clause_counter[0]}",
                kind=DeltaKind.VISIBILITY_CHANGE,
                disposition=DeltaDisposition.BEHAVIORAL,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason="applicability/visibility presence changed",
            )
        )
        return
    if ba.content_id == aa.content_id:
        return
    # Narrowing surfaces/environments is breaking for excluded consumers.
    before_surfaces = set(ba.surfaces)
    after_surfaces = set(aa.surfaces)
    if after_surfaces and before_surfaces and not before_surfaces.issubset(after_surfaces):
        disp = DeltaDisposition.BREAKING
        reason = "visibility surfaces narrowed"
    elif before_surfaces != after_surfaces:
        disp = DeltaDisposition.BEHAVIORAL
        reason = "visibility surfaces changed"
    else:
        disp = DeltaDisposition.BEHAVIORAL
        reason = "applicability conditions changed"
    clause_counter[0] += 1
    clauses.append(
        _clause(
            clause_id=f"clause:visibility-{clause_counter[0]}",
            kind=DeltaKind.VISIBILITY_CHANGE,
            disposition=disp,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=consumer_domain,
            before_ref=before_ref,
            after_ref=after_ref,
            reason=reason,
        )
    )


def _compare_symbol_identity(
    before: ExpectedProgramContract,
    after: ExpectedProgramContract,
    *,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
    move_pairs: Sequence[MovePair],
    rename_pairs: Sequence[RenamePair],
    reexport_paths: Sequence[str],
    registration_changed: bool | None,
) -> None:
    bs = before.symbol
    as_ = after.symbol
    module_changed = bs.module_path != as_.module_path
    name_changed = bs.symbol_name != as_.symbol_name

    explicit_move = any(
        pair.before_path == bs.module_path and pair.after_path == as_.module_path
        for pair in move_pairs
    )
    explicit_rename = any(
        pair.before_name == bs.symbol_name and pair.after_name == as_.symbol_name
        for pair in rename_pairs
    )

    if module_changed and (explicit_move or not name_changed):
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:symbol-move-{clause_counter[0]}",
                kind=DeltaKind.SYMBOL_MOVE,
                disposition=DeltaDisposition.BEHAVIORAL
                if consumer_domain == DOMAIN_PYTHON_CALLERS
                else DeltaDisposition.BREAKING,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"symbol moved {bs.module_path!r} -> {as_.module_path!r}",
            )
        )
    if name_changed and (explicit_rename or not module_changed):
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:symbol-rename-{clause_counter[0]}",
                kind=DeltaKind.SYMBOL_RENAME,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"symbol renamed {bs.symbol_name!r} -> {as_.symbol_name!r}",
            )
        )
    if reexport_paths:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:symbol-reexport-{clause_counter[0]}",
                kind=DeltaKind.SYMBOL_REEXPORT,
                disposition=DeltaDisposition.BEHAVIORAL,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=f"re-export paths recorded: {list(reexport_paths)}",
            )
        )
    if registration_changed is True:
        clause_counter[0] += 1
        clauses.append(
            _clause(
                clause_id=f"clause:symbol-registration-{clause_counter[0]}",
                kind=DeltaKind.SYMBOL_REGISTRATION,
                disposition=DeltaDisposition.BREAKING
                if consumer_domain in {DOMAIN_REGISTRATION, DOMAIN_PUBLIC_API}
                else DeltaDisposition.BEHAVIORAL,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason="symbol registration changed",
            )
        )


def _compare_memory_facets(
    *,
    before_ref_mem: str,
    after_ref_mem: str,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
) -> None:
    if before_ref_mem == after_ref_mem:
        return
    if not before_ref_mem and not after_ref_mem:
        return
    clause_counter[0] += 1
    clauses.append(
        _clause(
            clause_id=f"clause:memory-facet-{clause_counter[0]}",
            kind=DeltaKind.MEMORY_FACET_CHANGE,
            disposition=DeltaDisposition.BREAKING
            if consumer_domain == DOMAIN_MEMORY
            else DeltaDisposition.BEHAVIORAL,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=consumer_domain,
            before_ref=before_ref,
            after_ref=after_ref,
            reason=(
                f"memory facet refs changed {before_ref_mem!r} -> {after_ref_mem!r}"
            ),
        )
    )


def _compare_surface_changes(
    surface_changes: Sequence[StructuralSurfaceChange],
    *,
    subject_symbol_id: str,
    consumer_domain: str,
    before_ref: str,
    after_ref: str,
    clauses: list[ContractClauseDelta],
    clause_counter: list[int],
) -> None:
    for change in surface_changes:
        kind = (
            _SURFACE_TO_INTRO[change.surface]
            if change.introduced
            else _SURFACE_TO_REMOVE[change.surface]
        )
        clause_counter[0] += 1
        if change.introduced:
            # New optional factory may be compatible; new required method/field
            # is breaking for implementors/public API.
            if change.surface in {
                StructuralSurfaceKind.FACTORY,
                StructuralSurfaceKind.CONSTRUCTOR,
            }:
                disp = DeltaDisposition.BEHAVIORAL
            else:
                disp = DeltaDisposition.BREAKING
            reason = change.reason or f"{change.surface.value} introduced"
        else:
            disp = DeltaDisposition.BREAKING
            reason = change.reason or f"{change.surface.value} removed"
        clauses.append(
            _clause(
                clause_id=f"clause:surface-{change.surface.value}-"
                f"{'intro' if change.introduced else 'remove'}-{clause_counter[0]}",
                kind=kind,
                disposition=disp,
                subject_symbol_id=subject_symbol_id,
                consumer_domain=consumer_domain,
                before_ref=before_ref,
                after_ref=after_ref,
                reason=reason,
            )
        )


def _contracts_semantically_equal(
    before: ExpectedProgramContract, after: ExpectedProgramContract
) -> bool:
    """Equality ignoring tree-bound symbol location and source artifact churn.

    Module path and symbol name are location/identity, not contract body; pure
    moves and renames keep the body equal under this comparison.
    """

    def strip_symbol(contract: ExpectedProgramContract) -> dict[str, Any]:
        payload = contract.to_dict()
        # Drop identity fields that always differ across base/candidate trees.
        payload.pop("content_id", None)
        symbol = dict(payload.get("symbol") or {})
        for key in (
            "tree_id",
            "blob_cid",
            "span_start",
            "span_end",
            "content_id",
            "symbol_id",
            "module_path",
            "symbol_name",
            "qualified_name",
        ):
            symbol.pop(key, None)
        payload["symbol"] = symbol
        # Sources may re-extract with new artifact ids; compare kinds+locators only.
        sources = []
        for src in payload.get("sources") or []:
            if not isinstance(src, Mapping):
                continue
            sources.append(
                {
                    "source_kind": src.get("source_kind"),
                    "role": src.get("role"),
                    "locator": src.get("locator"),
                    "extractor_rule": src.get("extractor_rule"),
                }
            )
        payload["sources"] = sources
        return payload

    return strip_symbol(before) == strip_symbol(after)


def _dedupe_clauses(
    clauses: Sequence[ContractClauseDelta],
) -> tuple[ContractClauseDelta, ...]:
    seen: set[str] = set()
    result: list[ContractClauseDelta] = []
    for clause in clauses:
        if clause.clause_id in seen:
            continue
        seen.add(clause.clause_id)
        result.append(clause)
    if len(result) > MAX_CLAUSE_COUNT:
        raise IncompleteContractDeltaError(
            f"clause count {len(result)} exceeds MAX_CLAUSE_COUNT"
        )
    # Stable order by kind then clause_id for determinism.
    return tuple(sorted(result, key=lambda item: (item.kind.value, item.clause_id)))


class ProgramContractDeltaAnalyzer:
    """Compute exact semantic contract deltas for an explicit consumer domain."""

    INTERFACE: Final[str] = PROGRAM_CONTRACT_DELTA_ANALYZER_INTERFACE
    VERSION: Final[str] = PROGRAM_CONTRACT_DELTA_ANALYZER_VERSION

    def normalize_partition(
        self, request: ProgramContractDeltaRequest
    ) -> NormalizedChangePartition:
        return normalize_change_partition(
            request.change_set,
            path_churn=request.path_churn,
            move_pairs=request.move_pairs,
            rename_pairs=request.rename_pairs,
        )

    def analyze(
        self, request: ProgramContractDeltaRequest
    ) -> ProgramContractDeltaAnalysis:
        if not isinstance(request, ProgramContractDeltaRequest):
            raise ProgramContractDeltaError(
                "request must be ProgramContractDeltaRequest"
            )
        _assert_roots_and_binding(request)
        partition = self.normalize_partition(request)
        subject_symbol_id = _logical_symbol_id(
            request.before.symbol, request.after.symbol, request.subject_symbol_id
        )
        before_ref = request.before.expected_contract_id
        after_ref = request.after.expected_contract_id
        domain = request.consumer_domain

        # Pure non-semantic path churn cannot manufacture parameter/result
        # migration obligations.  Identity-only move/rename clauses may still
        # be recorded when the producer declared move/rename pairs or the
        # symbol location itself shifted without a semantic body change.
        if partition.is_purely_non_semantic:
            identity_clauses: list[ContractClauseDelta] = []
            counter = [0]
            body_equal = _contracts_semantically_equal(request.before, request.after)
            location_shifted = (
                request.before.symbol.module_path != request.after.symbol.module_path
                or request.before.symbol.symbol_name != request.after.symbol.symbol_name
            )
            if (
                request.move_pairs
                or request.rename_pairs
                or (body_equal and location_shifted)
            ):
                _compare_symbol_identity(
                    request.before,
                    request.after,
                    subject_symbol_id=subject_symbol_id,
                    consumer_domain=domain,
                    before_ref=before_ref,
                    after_ref=after_ref,
                    clauses=identity_clauses,
                    clause_counter=counter,
                    move_pairs=request.move_pairs,
                    rename_pairs=request.rename_pairs,
                    reexport_paths=(),
                    registration_changed=None,
                )
            # When contracts still differ in semantic body despite non-semantic
            # path classification, fail closed rather than silently drop clauses.
            if not body_equal and not identity_clauses and not location_shifted:
                raise IncompleteContractDeltaError(
                    "semantic contract body changed without a semantic path; "
                    "refuse to treat pure non-semantic churn as authority"
                )
            identity_clauses_t = _dedupe_clauses(identity_clauses)
            if not identity_clauses_t:
                return ProgramContractDeltaAnalysis(
                    partition=partition,
                    deltas=(),
                    consumer_domain=domain,
                    subject_symbol_id=subject_symbol_id,
                    pure_non_semantic=True,
                )
            delta = ProgramContractDelta(
                roots=request.roots,
                change_set_id=request.change_set.content_id,
                subject_symbol_id=subject_symbol_id,
                before_contract_ref=before_ref,
                after_contract_ref=after_ref,
                clauses=identity_clauses_t,
                evidence_refs=request.evidence_refs,
                proof_refs=request.proof_refs,
            )
            return ProgramContractDeltaAnalysis(
                partition=partition,
                deltas=(delta,),
                consumer_domain=domain,
                subject_symbol_id=subject_symbol_id,
                pure_non_semantic=True,
            )

        clauses: list[ContractClauseDelta] = []
        counter = [0]

        _compare_symbol_identity(
            request.before,
            request.after,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
            move_pairs=request.move_pairs,
            rename_pairs=request.rename_pairs,
            reexport_paths=request.reexport_paths,
            registration_changed=request.registration_changed,
        )
        _compare_surface_changes(
            request.surface_changes,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
        )
        _compare_parameters(
            request.before,
            request.after,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
        )
        _compare_returns(
            request.before,
            request.after,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
        )
        _compare_interface_protocol_schema(
            request.before,
            request.after,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
        )
        _compare_sync_async_cancellation(
            request.before,
            request.after,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
            cancellation_before=request.cancellation_before,
            cancellation_after=request.cancellation_after,
        )
        _compare_errors(
            request.before,
            request.after,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
        )
        _compare_effects(
            request.before,
            request.after,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
        )
        _compare_capabilities(
            request.before,
            request.after,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
        )
        _compare_authorization(
            request.before,
            request.after,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
        )
        _compare_lifecycle_state_consistency_resource(
            request.before,
            request.after,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
        )
        _compare_applicability_visibility(
            request.before,
            request.after,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
        )
        _compare_memory_facets(
            before_ref_mem=request.memory_facet_before_ref,
            after_ref_mem=request.memory_facet_after_ref,
            subject_symbol_id=subject_symbol_id,
            consumer_domain=domain,
            before_ref=before_ref,
            after_ref=after_ref,
            clauses=clauses,
            clause_counter=counter,
        )

        final_clauses = _dedupe_clauses(clauses)
        if not final_clauses:
            # Contracts differ only in non-compared metadata (e.g. description).
            # Fail closed with an explicit behavioral unknown rather than silence.
            if not _contracts_semantically_equal(request.before, request.after):
                final_clauses = (
                    _clause(
                        clause_id="clause:unknown-residual",
                        kind=DeltaKind.PARAMETER_VARIANCE,
                        disposition=DeltaDisposition.UNKNOWN,
                        subject_symbol_id=subject_symbol_id,
                        consumer_domain=domain,
                        before_ref=before_ref,
                        after_ref=after_ref,
                        reason=(
                            "contracts differ without a classified semantic clause; "
                            "fail closed as unknown"
                        ),
                    ),
                )
            else:
                return ProgramContractDeltaAnalysis(
                    partition=partition,
                    deltas=(),
                    consumer_domain=domain,
                    subject_symbol_id=subject_symbol_id,
                    pure_non_semantic=partition.is_purely_non_semantic,
                )

        delta = ProgramContractDelta(
            roots=request.roots,
            change_set_id=request.change_set.content_id,
            subject_symbol_id=subject_symbol_id,
            before_contract_ref=before_ref,
            after_contract_ref=after_ref,
            clauses=final_clauses,
            evidence_refs=request.evidence_refs,
            proof_refs=request.proof_refs,
        )
        return ProgramContractDeltaAnalysis(
            partition=partition,
            deltas=(delta,),
            consumer_domain=domain,
            subject_symbol_id=subject_symbol_id,
            pure_non_semantic=False,
        )

    def compare(
        self,
        *,
        roots: PropagationAuthorityRoots,
        change_set: ProgramChangeSet,
        before: ExpectedProgramContract,
        after: ExpectedProgramContract,
        consumer_domain: str,
        **kwargs: Any,
    ) -> ProgramContractDeltaAnalysis:
        """Convenience wrapper around :meth:`analyze`."""

        return self.analyze(
            ProgramContractDeltaRequest(
                roots=roots,
                change_set=change_set,
                before=before,
                after=after,
                consumer_domain=consumer_domain,
                **kwargs,
            )
        )


__all__ = [
    "PROGRAM_CONTRACT_DELTA_ANALYZER_INTERFACE",
    "PROGRAM_CONTRACT_DELTA_ANALYZER_VERSION",
    "DOMAIN_PYTHON_CALLERS",
    "DOMAIN_SCHEMA_CONSUMERS",
    "DOMAIN_SERIALIZERS",
    "DOMAIN_HTTP_CLIENTS",
    "DOMAIN_PUBLIC_API",
    "DOMAIN_TESTS",
    "DOMAIN_REGISTRATION",
    "DOMAIN_MEMORY",
    "ProgramContractDeltaError",
    "StaleContractDeltaError",
    "IncompleteContractDeltaError",
    "CrossRootContractDeltaError",
    "UnsupportedContractDeltaError",
    "SelfAuthoredExpectationError",
    "NonSemanticChurnKind",
    "StructuralSurfaceKind",
    "PathChurnClassification",
    "StructuralSurfaceChange",
    "MovePair",
    "RenamePair",
    "NormalizedChangePartition",
    "ProgramContractDeltaRequest",
    "ProgramContractDeltaAnalysis",
    "ProgramContractDeltaAnalyzer",
    "normalize_change_partition",
    # Re-export canonical RPR-022 records used by callers/tests.
    "ContractClauseDelta",
    "DeltaDisposition",
    "DeltaKind",
    "ProgramContractDelta",
    "ProgramChangeSet",
    "PropagationAuthorityRoots",
]
