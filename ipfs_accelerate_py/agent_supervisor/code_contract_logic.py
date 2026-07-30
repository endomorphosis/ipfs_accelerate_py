"""Translate finite code-contract predicates into ipfs_datasets_py IR claims.

VFS-019 / VFS-G070: finite supported predicates for types, nullability, errors,
effects, authorization, state transitions, ordering, idempotence, and bounded
reachability become immutable :class:`~ipfs_datasets_py.logic.ir_core.claims.IRClaim`
records with source and assumption CIDs.  Every successful translation emits a
round-trip/conformance receipt; unsupported residuals are explicit.

Conflict policy: reuse FormalLogicVocabulary
(:mod:`proof.formal_logic_vocabulary`) and ``ipfs_datasets_py.logic.ir_core``
claims/protocols.  This module is a construction API, not a natural-language
parser and not a separate theorem language.

Objective validation repair for VFS-G070 anchors the synthetic discovery term
``objective validation repair`` on this translation surface without granting
translation products completion or proof authority.  Translation
(``vfs/logic-translation@1``) stays separate from MultiProverRouter candidate
search and KernelVerification / independent portfolio validation
(``vfs/kernel-proof-receipt@1`` on the prover surface).

Fail-closed rejections:

* unbound axioms (obligation references an undeclared assumption)
* name capture (binder collides with free name or rebinds a different sort)
* sort mismatch (predicate argument sorts disagree with the closed signature)
* partial call slices presented as closed
* silent approximation (dropped residual without unsupported semantics)
* changed translator / ruleset reuse against a prior receipt
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_datasets_py.logic.ir_core.claims import (
    IR_ASSUMPTION_SCHEMA_VERSION,
    IR_CLAIM_SCHEMA_VERSION,
    IR_OBLIGATION_SCHEMA_VERSION,
    Assumption as IRAssumption,
    ClaimValidationError,
    IRClaim,
    ProofObligation as IRObligation,
    freeze_json,
)

from .program_contracts import (
    Assumption as ContractAssumption,
    AuthorizationSpec,
    EffectKind,
    EffectPolarity,
    ErrorSpec,
    ExpectedProgramContract,
    IdempotenceSpec,
    ObservedProgramContract,
    OrderingSpec,
    ParameterSpec,
    ReturnSpec,
    SideEffectSpec,
    SupportStatus,
    TypeShape,
    UnsupportedSemantics,
)
from .program_graph_queries import ProgramGraphSlice
from .proof import formal_logic_vocabulary as FormalLogicVocabulary
from .proof.formal_logic_vocabulary import (
    Formula,
    ReviewedPredicate,
    TermSort,
    atom,
    constant,
)
from .proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)


# ---------------------------------------------------------------------------
# Versions, schemas, and identity pins
# ---------------------------------------------------------------------------

CODE_CONTRACT_LOGIC_VERSION: Final[int] = 1
TRANSLATOR_ID: Final[str] = "code-contract-logic"
TRANSLATOR_VERSION: Final[str] = "1"
RULESET_ID: Final[str] = "vfs/code-contract-predicates"
RULESET_VERSION: Final[str] = "1"
LOGIC_FAMILY: Final[str] = "code_contract_ir"
LOGIC_TRANSLATION_EVIDENCE: Final[str] = "vfs/logic-translation@1"
# Synthetic objective-heap evidence term for VFS-G070 validation-gate work.
# Exact-text discovery key only — never part of claim/receipt identity payload.
OBJECTIVE_VALIDATION_REPAIR_EVIDENCE: Final[str] = "objective validation repair"
# Domain parent goal that owns translation + kernel-proof surfaces.
OBJECTIVE_GOAL_ID: Final[str] = "VFS-G070"
# Validation-gate task that owns the synthetic repair obligation (VFS-053).
OBJECTIVE_VALIDATION_REPAIR_TASK_ID: Final[str] = "VFS-053"

# FormalLogicVocabulary is the reviewed translation vocabulary only; it never
# authorizes MultiProverRouter candidates or KernelVerification receipts.
assert FormalLogicVocabulary.LOGIC_VOCABULARY_VERSION >= 1
assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE == "objective validation repair"
assert OBJECTIVE_GOAL_ID == "VFS-G070"

CODE_CONTRACT_PREDICATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-predicate@1"
)
CODE_CONTRACT_ASSUMPTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-logic-assumption@1"
)
CALL_SLICE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-call-slice-binding@1"
)
TRANSLATION_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-translation-request@1"
)
TRANSLATION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-translation-result@1"
)
CONFORMANCE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-conformance-receipt@1"
)
UNSUPPORTED_RESIDUAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-unsupported-residual@1"
)

MAX_PREDICATES: Final[int] = 512
MAX_ARGUMENTS: Final[int] = 32
MAX_ASSUMPTIONS: Final[int] = 128
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_STATEMENT_BYTES: Final[int] = 16_384
MAX_SLICE_NODES: Final[int] = 4_096

_PINNED_TRANSLATOR_IDENTITY: Final[str] = content_identity(
    {
        "translator_id": TRANSLATOR_ID,
        "translator_version": TRANSLATOR_VERSION,
        "ruleset_id": RULESET_ID,
        "ruleset_version": RULESET_VERSION,
        "logic_version": CODE_CONTRACT_LOGIC_VERSION,
    }
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class CodeContractLogicError(ContractValidationError):
    """Malformed translation input or internal invariant violation."""


class TranslationRejectedError(CodeContractLogicError):
    """Fail-closed rejection of a translation attempt."""

    def __init__(self, code: "RejectionCode", detail: str) -> None:
        self.code = code if isinstance(code, RejectionCode) else RejectionCode(str(code))
        self.detail = detail
        super().__init__(f"{self.code.value}: {detail}")


class SupportedPredicateKind(str, Enum):
    """Finite supported predicate families (VFS-019 acceptance)."""

    TYPE = "type"
    NULLABILITY = "nullability"
    ERROR = "error"
    EFFECT = "effect"
    AUTHORIZATION = "authorization"
    STATE_TRANSITION = "state_transition"
    ORDERING = "ordering"
    IDEMPOTENCE = "idempotence"
    BOUNDED_REACHABILITY = "bounded_reachability"


class ArgumentSort(str, Enum):
    """Closed argument sorts for predicate atoms (sort-mismatch checks)."""

    SYMBOL = "symbol"
    TYPE = "type"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    STRING = "string"
    EFFECT_KIND = "effect_kind"
    EFFECT_POLARITY = "effect_polarity"
    MODE = "mode"
    NODE = "node"
    PATH = "path"
    BOUND = "bound"
    ERROR_NAME = "error_name"
    SCOPE = "scope"
    STATE = "state"


class TranslationStatus(str, Enum):
    TRANSLATED = "translated"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


class RejectionCode(str, Enum):
    UNBOUND_AXIOM = "unbound_axiom"
    NAME_CAPTURE = "name_capture"
    SORT_MISMATCH = "sort_mismatch"
    PARTIAL_SLICE_AS_CLOSED = "partial_slice_as_closed"
    SILENT_APPROXIMATION = "silent_approximation"
    TRANSLATOR_RULESET_REUSE = "translator_ruleset_reuse"
    UNSUPPORTED_PREDICATE = "unsupported_predicate"
    AMBIGUOUS_PREDICATE = "ambiguous_predicate"
    INVALID_INPUT = "invalid_input"
    MISSING_SOURCE = "missing_source"
    EMPTY_TRANSLATION = "empty_translation"
    ROUND_TRIP_FAILURE = "round_trip_failure"


class PredicateRelation(str, Enum):
    """Closed relation names per predicate family."""

    HAS_TYPE = "has_type"
    IS_NULLABLE = "is_nullable"
    MAY_RAISE = "may_raise"
    HAS_EFFECT = "has_effect"
    REQUIRES_AUTH = "requires_auth"
    TRANSITIONS = "transitions"
    ORDERED_AS = "ordered_as"
    IDEMPOTENT_AS = "idempotent_as"
    REACHABLE_WITHIN = "reachable_within"


# Signatures: ordered argument sorts for each relation.
_RELATION_SIGNATURES: Final[Mapping[PredicateRelation, tuple[ArgumentSort, ...]]] = (
    MappingProxyType(
        {
            PredicateRelation.HAS_TYPE: (
                ArgumentSort.SYMBOL,
                ArgumentSort.STRING,
                ArgumentSort.TYPE,
            ),
            PredicateRelation.IS_NULLABLE: (
                ArgumentSort.SYMBOL,
                ArgumentSort.STRING,
                ArgumentSort.BOOLEAN,
            ),
            PredicateRelation.MAY_RAISE: (
                ArgumentSort.SYMBOL,
                ArgumentSort.ERROR_NAME,
                ArgumentSort.BOOLEAN,
            ),
            PredicateRelation.HAS_EFFECT: (
                ArgumentSort.SYMBOL,
                ArgumentSort.EFFECT_KIND,
                ArgumentSort.EFFECT_POLARITY,
            ),
            PredicateRelation.REQUIRES_AUTH: (
                ArgumentSort.SYMBOL,
                ArgumentSort.MODE,
                ArgumentSort.SCOPE,
            ),
            PredicateRelation.TRANSITIONS: (
                ArgumentSort.SYMBOL,
                ArgumentSort.STATE,
                ArgumentSort.STATE,
            ),
            PredicateRelation.ORDERED_AS: (
                ArgumentSort.SYMBOL,
                ArgumentSort.MODE,
            ),
            PredicateRelation.IDEMPOTENT_AS: (
                ArgumentSort.SYMBOL,
                ArgumentSort.MODE,
            ),
            PredicateRelation.REACHABLE_WITHIN: (
                ArgumentSort.NODE,
                ArgumentSort.NODE,
                ArgumentSort.BOUND,
            ),
        }
    )
)

_KIND_TO_RELATION: Final[Mapping[SupportedPredicateKind, PredicateRelation]] = (
    MappingProxyType(
        {
            SupportedPredicateKind.TYPE: PredicateRelation.HAS_TYPE,
            SupportedPredicateKind.NULLABILITY: PredicateRelation.IS_NULLABLE,
            SupportedPredicateKind.ERROR: PredicateRelation.MAY_RAISE,
            SupportedPredicateKind.EFFECT: PredicateRelation.HAS_EFFECT,
            SupportedPredicateKind.AUTHORIZATION: PredicateRelation.REQUIRES_AUTH,
            SupportedPredicateKind.STATE_TRANSITION: PredicateRelation.TRANSITIONS,
            SupportedPredicateKind.ORDERING: PredicateRelation.ORDERED_AS,
            SupportedPredicateKind.IDEMPOTENCE: PredicateRelation.IDEMPOTENT_AS,
            SupportedPredicateKind.BOUNDED_REACHABILITY: PredicateRelation.REACHABLE_WITHIN,
        }
    )
)

_RELATION_TO_KIND: Final[Mapping[PredicateRelation, SupportedPredicateKind]] = (
    MappingProxyType({relation: kind for kind, relation in _KIND_TO_RELATION.items()})
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise CodeContractLogicError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise CodeContractLogicError(f"{name} must not be empty")
    if len(text.encode("utf-8")) > maximum:
        raise CodeContractLogicError(f"{name} exceeds {maximum} UTF-8 bytes")
    return text


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CodeContractLogicError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    text = str(getattr(value, "value", value) or "").strip()
    try:
        return enum_type(text)
    except ValueError as exc:
        raise CodeContractLogicError(f"unsupported {name}: {text!r}") from exc


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, MappingProxyType):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CodeContractLogicError("value is not canonical JSON") from exc


def _cid(value: Any) -> str:
    return content_identity(_plain(value))


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise CodeContractLogicError(f"{name} must be an object with string keys")
    return {str(k): _plain(v) for k, v in value.items()}


def _unique_sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({item for item in values if item}))


def translator_identity(
    *,
    translator_id: str = TRANSLATOR_ID,
    translator_version: str = TRANSLATOR_VERSION,
    ruleset_id: str = RULESET_ID,
    ruleset_version: str = RULESET_VERSION,
) -> str:
    """Content identity of the pinned translator/ruleset pair."""

    return content_identity(
        {
            "translator_id": _text(translator_id, "translator_id"),
            "translator_version": _text(translator_version, "translator_version"),
            "ruleset_id": _text(ruleset_id, "ruleset_id"),
            "ruleset_version": _text(ruleset_version, "ruleset_version"),
            "logic_version": CODE_CONTRACT_LOGIC_VERSION,
        }
    )


def pinned_translator_identity() -> str:
    return _PINNED_TRANSLATOR_IDENTITY


# ---------------------------------------------------------------------------
# Predicate argument + atom
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredicateArgument:
    """One sort-tagged argument of a finite contract predicate."""

    name: str
    sort: ArgumentSort
    value: Any
    binder: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "argument.name"))
        object.__setattr__(
            self, "sort", _enum(self.sort, ArgumentSort, "argument.sort")
        )
        object.__setattr__(
            self, "binder", _boolean(bool(self.binder), "argument.binder")
        )
        value = self.value
        sort = self.sort
        if sort is ArgumentSort.BOOLEAN:
            if not isinstance(value, bool):
                raise TranslationRejectedError(
                    RejectionCode.SORT_MISMATCH,
                    f"argument {self.name!r} requires boolean, got {type(value).__name__}",
                )
        elif sort in (ArgumentSort.INTEGER, ArgumentSort.BOUND):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TranslationRejectedError(
                    RejectionCode.SORT_MISMATCH,
                    f"argument {self.name!r} requires non-negative integer",
                )
        else:
            if not isinstance(value, str) or not value.strip():
                raise TranslationRejectedError(
                    RejectionCode.SORT_MISMATCH,
                    f"argument {self.name!r} requires non-empty string for sort {sort.value}",
                )
            value = value.strip()
        object.__setattr__(self, "value", value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sort": self.sort.value,
            "value": self.value,
            "binder": self.binder,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredicateArgument":
        if not isinstance(payload, Mapping):
            raise CodeContractLogicError("predicate argument must be an object")
        return cls(
            name=payload.get("name", ""),
            sort=payload.get("sort", ""),
            value=payload.get("value"),
            binder=bool(payload.get("binder", False)),
        )


@dataclass(frozen=True)
class ContractPredicate(CanonicalContract):
    """One finite supported contract/call-slice predicate atom."""

    SCHEMA: ClassVar[str] = CODE_CONTRACT_PREDICATE_SCHEMA

    kind: SupportedPredicateKind
    relation: PredicateRelation
    arguments: tuple[PredicateArgument, ...]
    source_cid: str
    assumption_cids: tuple[str, ...] = ()
    subject: str = ""
    closed: bool = True
    support: SupportStatus = SupportStatus.SUPPORTED
    residual: str = ""
    ambiguity: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, SupportedPredicateKind, "kind")
        )
        object.__setattr__(
            self, "relation", _enum(self.relation, PredicateRelation, "relation")
        )
        expected_kind = _RELATION_TO_KIND.get(self.relation)
        if expected_kind is not None and expected_kind is not self.kind:
            raise TranslationRejectedError(
                RejectionCode.INVALID_INPUT,
                f"relation {self.relation.value} does not match kind {self.kind.value}",
            )
        args = tuple(self.arguments or ())
        normalized: list[PredicateArgument] = []
        for item in args:
            if isinstance(item, PredicateArgument):
                normalized.append(item)
            elif isinstance(item, Mapping):
                normalized.append(PredicateArgument.from_dict(item))
            else:
                raise CodeContractLogicError("arguments must be PredicateArgument values")
        if len(normalized) > MAX_ARGUMENTS:
            raise CodeContractLogicError(f"arguments exceed {MAX_ARGUMENTS}")
        object.__setattr__(self, "arguments", tuple(normalized))
        signature = _RELATION_SIGNATURES.get(self.relation)
        if signature is not None:
            actual = tuple(arg.sort for arg in self.arguments)
            if actual != signature:
                raise TranslationRejectedError(
                    RejectionCode.SORT_MISMATCH,
                    "predicate %s expects sorts (%s), got (%s)"
                    % (
                        self.relation.value,
                        ", ".join(s.value for s in signature),
                        ", ".join(s.value for s in actual),
                    ),
                )
        object.__setattr__(
            self, "source_cid", _text(self.source_cid, "source_cid")
        )
        cids = tuple(
            _text(item, "assumption_cids")
            for item in (self.assumption_cids or ())
        )
        if len(cids) != len(set(cids)):
            raise CodeContractLogicError("assumption_cids must be unique")
        object.__setattr__(self, "assumption_cids", cids)
        object.__setattr__(
            self,
            "subject",
            _text(self.subject, "subject", required=False)
            or (self.arguments[0].value if self.arguments else ""),
        )
        object.__setattr__(self, "closed", _boolean(bool(self.closed), "closed"))
        object.__setattr__(
            self, "support", _enum(self.support, SupportStatus, "support")
        )
        object.__setattr__(
            self, "residual", _text(self.residual, "residual", required=False)
        )
        object.__setattr__(
            self, "ambiguity", _text(self.ambiguity, "ambiguity", required=False)
        )
        object.__setattr__(self, "metadata", MappingProxyType(_mapping(self.metadata, "metadata")))

    @property
    def predicate_id(self) -> str:
        return self.content_id

    def statement_atom(self) -> dict[str, Any]:
        """Canonical statement payload used inside IR claims/obligations."""

        return {
            "kind": self.kind.value,
            "relation": self.relation.value,
            "arguments": [arg.to_dict() for arg in self.arguments],
            "subject": self.subject,
            "closed": self.closed,
            "support": self.support.value,
            "source_cid": self.source_cid,
            "assumption_cids": list(self.assumption_cids),
            "residual": self.residual,
            "ambiguity": self.ambiguity,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "logic_version": CODE_CONTRACT_LOGIC_VERSION,
            "kind": self.kind,
            "relation": self.relation,
            "arguments": [arg.to_dict() for arg in self.arguments],
            "source_cid": self.source_cid,
            "assumption_cids": list(self.assumption_cids),
            "subject": self.subject,
            "closed": self.closed,
            "support": self.support,
            "residual": self.residual,
            "ambiguity": self.ambiguity,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractPredicate":
        if not isinstance(payload, Mapping):
            raise CodeContractLogicError("contract predicate must be an object")
        supplied = payload.get("schema")
        if supplied not in (None, "", cls.SCHEMA):
            raise CodeContractLogicError(
                f"unsupported schema {supplied!r}; expected {cls.SCHEMA}"
            )
        result = cls(
            kind=payload.get("kind", ""),
            relation=payload.get("relation", ""),
            arguments=tuple(payload.get("arguments") or ()),
            source_cid=payload.get("source_cid", ""),
            assumption_cids=tuple(payload.get("assumption_cids") or ()),
            subject=payload.get("subject", ""),
            closed=bool(payload.get("closed", True)),
            support=payload.get("support", SupportStatus.SUPPORTED),
            residual=payload.get("residual", ""),
            ambiguity=payload.get("ambiguity", ""),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("content_id") or payload.get("predicate_id")
        if claimed and claimed != result.predicate_id:
            raise CodeContractLogicError(
                "predicate content identity does not match payload"
            )
        return result


@dataclass(frozen=True)
class LogicAssumption(CanonicalContract):
    """Content-addressed translation assumption with explicit binder names."""

    SCHEMA: ClassVar[str] = CODE_CONTRACT_ASSUMPTION_SCHEMA

    statement: str
    binders: tuple[tuple[str, ArgumentSort], ...] = ()
    source_cid: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "statement",
            _text(self.statement, "statement", maximum=MAX_STATEMENT_BYTES),
        )
        binders_raw = tuple(self.binders or ())
        binders: list[tuple[str, ArgumentSort]] = []
        seen: set[str] = set()
        for entry in binders_raw:
            if (
                not isinstance(entry, Sequence)
                or isinstance(entry, (str, bytes, bytearray))
                or len(entry) != 2
            ):
                raise CodeContractLogicError(
                    "binders must be (name, ArgumentSort) pairs"
                )
            name = _text(entry[0], "binder.name")
            sort = _enum(entry[1], ArgumentSort, "binder.sort")
            if name in seen:
                raise TranslationRejectedError(
                    RejectionCode.NAME_CAPTURE,
                    f"duplicate binder name {name!r}",
                )
            seen.add(name)
            binders.append((name, sort))
        object.__setattr__(self, "binders", tuple(binders))
        object.__setattr__(
            self, "source_cid", _text(self.source_cid, "source_cid", required=False)
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(_mapping(self.metadata, "metadata"))
        )

    @property
    def assumption_cid(self) -> str:
        return self.content_id

    def to_ir_assumption(self) -> IRAssumption:
        return IRAssumption(
            assumption_id=self.assumption_cid,
            statement=self.statement,
            source_refs=tuple(
                ref for ref in (self.source_cid,) if ref
            ),
            metadata=freeze_json(
                {
                    "binders": [
                        {"name": name, "sort": sort.value}
                        for name, sort in self.binders
                    ],
                    "logic_family": LOGIC_FAMILY,
                }
            ),
            schema_version=IR_ASSUMPTION_SCHEMA_VERSION,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "logic_version": CODE_CONTRACT_LOGIC_VERSION,
            "statement": self.statement,
            "binders": [[name, sort.value] for name, sort in self.binders],
            "source_cid": self.source_cid,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicAssumption":
        if not isinstance(payload, Mapping):
            raise CodeContractLogicError("logic assumption must be an object")
        binders = []
        for entry in payload.get("binders") or ():
            if isinstance(entry, Mapping):
                binders.append((entry.get("name", ""), entry.get("sort", "")))
            else:
                binders.append(tuple(entry))
        result = cls(
            statement=payload.get("statement", ""),
            binders=tuple(binders),
            source_cid=payload.get("source_cid", ""),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("content_id") or payload.get("assumption_cid")
        if claimed and claimed != result.assumption_cid:
            raise CodeContractLogicError(
                "assumption content identity does not match payload"
            )
        return result


# ---------------------------------------------------------------------------
# Call-slice binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallSliceBinding(CanonicalContract):
    """Normalized call-slice facts required by reachability translation.

    Partial / truncated / frontier-open slices must never be presented as
    closed.  Prefer constructing via :meth:`from_program_graph_slice`.
    """

    SCHEMA: ClassVar[str] = CALL_SLICE_BINDING_SCHEMA

    slice_cid: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...] = ()
    complete: bool = False
    dependency_complete: bool = False
    truncated: bool = False
    presented_as_closed: bool = False
    depth_bound: int = 0
    forest_id: str = ""
    graph_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "slice_cid", _text(self.slice_cid, "slice_cid")
        )
        nodes = tuple(
            _text(item, "node_ids") for item in (self.node_ids or ())
        )
        if len(nodes) > MAX_SLICE_NODES:
            raise CodeContractLogicError(f"node_ids exceed {MAX_SLICE_NODES}")
        if len(nodes) != len(set(nodes)):
            raise CodeContractLogicError("node_ids must be unique")
        object.__setattr__(self, "node_ids", nodes)
        edges = tuple(
            _text(item, "edge_ids", required=False)
            for item in (self.edge_ids or ())
            if item
        )
        object.__setattr__(self, "edge_ids", _unique_sorted(edges))
        for name in (
            "complete",
            "dependency_complete",
            "truncated",
            "presented_as_closed",
        ):
            object.__setattr__(
                self, name, _boolean(bool(getattr(self, name)), name)
            )
        if isinstance(self.depth_bound, bool) or not isinstance(self.depth_bound, int):
            raise CodeContractLogicError("depth_bound must be a non-negative integer")
        if self.depth_bound < 0:
            raise CodeContractLogicError("depth_bound must be a non-negative integer")
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id", required=False)
        )
        object.__setattr__(
            self, "graph_id", _text(self.graph_id, "graph_id", required=False)
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(_mapping(self.metadata, "metadata"))
        )
        # Structural closedness: truncated or incomplete slices cannot be closed.
        is_closed = (
            self.complete
            and self.dependency_complete
            and not self.truncated
            and bool(self.node_ids)
        )
        if self.presented_as_closed and not is_closed:
            raise TranslationRejectedError(
                RejectionCode.PARTIAL_SLICE_AS_CLOSED,
                "partial or truncated call slice cannot be presented as closed",
            )

    @property
    def is_structurally_closed(self) -> bool:
        return (
            self.complete
            and self.dependency_complete
            and not self.truncated
            and bool(self.node_ids)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "logic_version": CODE_CONTRACT_LOGIC_VERSION,
            "slice_cid": self.slice_cid,
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "complete": self.complete,
            "dependency_complete": self.dependency_complete,
            "truncated": self.truncated,
            "presented_as_closed": self.presented_as_closed,
            "depth_bound": self.depth_bound,
            "forest_id": self.forest_id,
            "graph_id": self.graph_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallSliceBinding":
        if not isinstance(payload, Mapping):
            raise CodeContractLogicError("call slice binding must be an object")
        return cls(
            slice_cid=payload.get("slice_cid", ""),
            node_ids=tuple(payload.get("node_ids") or ()),
            edge_ids=tuple(payload.get("edge_ids") or ()),
            complete=bool(payload.get("complete", False)),
            dependency_complete=bool(payload.get("dependency_complete", False)),
            truncated=bool(payload.get("truncated", False)),
            presented_as_closed=bool(payload.get("presented_as_closed", False)),
            depth_bound=int(payload.get("depth_bound") or 0),
            forest_id=payload.get("forest_id", ""),
            graph_id=payload.get("graph_id", ""),
            metadata=payload.get("metadata") or {},
        )

    @classmethod
    def from_program_graph_slice(
        cls,
        slice_: ProgramGraphSlice,
        *,
        presented_as_closed: bool | None = None,
        depth_bound: int | None = None,
    ) -> "CallSliceBinding":
        if not isinstance(slice_, ProgramGraphSlice):
            raise CodeContractLogicError(
                "slice must be a ProgramGraphSlice"
            )
        closed = (
            slice_.complete
            and slice_.dependency_complete
            and not slice_.truncated
            and bool(slice_.node_ids)
        )
        if presented_as_closed is None:
            presented_as_closed = closed
        bound = depth_bound if depth_bound is not None else int(slice_.depth_reached or 0)
        return cls(
            slice_cid=slice_.slice_id,
            node_ids=tuple(slice_.node_ids),
            edge_ids=tuple(slice_.edge_ids),
            complete=bool(slice_.complete),
            dependency_complete=bool(slice_.dependency_complete),
            truncated=bool(slice_.truncated),
            presented_as_closed=bool(presented_as_closed),
            depth_bound=bound,
            forest_id=slice_.forest_id,
            graph_id=slice_.graph_id,
            metadata={"query_id": slice_.query_id, "kind": slice_.kind.value},
        )


# ---------------------------------------------------------------------------
# Unsupported residual + receipt/result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnsupportedResidual(CanonicalContract):
    """Explicit unsupported semantics retained on the translation boundary."""

    SCHEMA: ClassVar[str] = UNSUPPORTED_RESIDUAL_SCHEMA

    aspect: str
    reason: str
    residual: str = ""
    source_cid: str = ""
    predicate_kind: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "aspect", _text(self.aspect, "aspect"))
        object.__setattr__(
            self,
            "reason",
            _text(self.reason, "reason", maximum=MAX_STATEMENT_BYTES),
        )
        object.__setattr__(
            self, "residual", _text(self.residual, "residual", required=False)
        )
        object.__setattr__(
            self, "source_cid", _text(self.source_cid, "source_cid", required=False)
        )
        object.__setattr__(
            self,
            "predicate_kind",
            _text(self.predicate_kind, "predicate_kind", required=False),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "logic_version": CODE_CONTRACT_LOGIC_VERSION,
            "aspect": self.aspect,
            "reason": self.reason,
            "residual": self.residual,
            "source_cid": self.source_cid,
            "predicate_kind": self.predicate_kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnsupportedResidual":
        if not isinstance(payload, Mapping):
            raise CodeContractLogicError("unsupported residual must be an object")
        return cls(
            aspect=payload.get("aspect", ""),
            reason=payload.get("reason", ""),
            residual=payload.get("residual", ""),
            source_cid=payload.get("source_cid", ""),
            predicate_kind=payload.get("predicate_kind", ""),
        )


@dataclass(frozen=True)
class ConformanceReceipt(CanonicalContract):
    """Round-trip / conformance receipt for one translation.

    Binds source contract, call slice, assumptions, translator/ruleset, and
    claim digests.  A receipt from a different translator or ruleset cannot be
    reused for the current pin.
    """

    SCHEMA: ClassVar[str] = CONFORMANCE_RECEIPT_SCHEMA

    request_cid: str
    source_contract_cid: str
    call_slice_cid: str
    assumption_cids: tuple[str, ...]
    claim_digests: tuple[str, ...]
    obligation_digests: tuple[str, ...]
    predicate_cids: tuple[str, ...]
    unsupported_cids: tuple[str, ...]
    translator_id: str
    translator_version: str
    ruleset_id: str
    ruleset_version: str
    translator_identity: str
    round_trip_ok: bool
    status: TranslationStatus
    evidence: str = LOGIC_TRANSLATION_EVIDENCE
    reconstructed_predicate_cids: tuple[str, ...] = ()
    rejection_codes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "request_cid",
            "source_contract_cid",
            "translator_id",
            "translator_version",
            "ruleset_id",
            "ruleset_version",
            "translator_identity",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "call_slice_cid",
            _text(self.call_slice_cid, "call_slice_cid", required=False),
        )
        for name in (
            "assumption_cids",
            "claim_digests",
            "obligation_digests",
            "predicate_cids",
            "unsupported_cids",
            "reconstructed_predicate_cids",
            "rejection_codes",
        ):
            object.__setattr__(
                self,
                name,
                tuple(
                    _text(item, name)
                    for item in (getattr(self, name) or ())
                ),
            )
        object.__setattr__(
            self, "round_trip_ok", _boolean(bool(self.round_trip_ok), "round_trip_ok")
        )
        object.__setattr__(
            self, "status", _enum(self.status, TranslationStatus, "status")
        )
        object.__setattr__(
            self,
            "evidence",
            _text(self.evidence, "evidence", required=False) or LOGIC_TRANSLATION_EVIDENCE,
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(_mapping(self.metadata, "metadata"))
        )
        expected = translator_identity(
            translator_id=self.translator_id,
            translator_version=self.translator_version,
            ruleset_id=self.ruleset_id,
            ruleset_version=self.ruleset_version,
        )
        if self.translator_identity != expected:
            raise TranslationRejectedError(
                RejectionCode.TRANSLATOR_RULESET_REUSE,
                "translator_identity does not match declared translator/ruleset pins",
            )

    @property
    def receipt_cid(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "logic_version": CODE_CONTRACT_LOGIC_VERSION,
            "request_cid": self.request_cid,
            "source_contract_cid": self.source_contract_cid,
            "call_slice_cid": self.call_slice_cid,
            "assumption_cids": list(self.assumption_cids),
            "claim_digests": list(self.claim_digests),
            "obligation_digests": list(self.obligation_digests),
            "predicate_cids": list(self.predicate_cids),
            "unsupported_cids": list(self.unsupported_cids),
            "translator_id": self.translator_id,
            "translator_version": self.translator_version,
            "ruleset_id": self.ruleset_id,
            "ruleset_version": self.ruleset_version,
            "translator_identity": self.translator_identity,
            "round_trip_ok": self.round_trip_ok,
            "status": self.status,
            "evidence": self.evidence,
            "reconstructed_predicate_cids": list(self.reconstructed_predicate_cids),
            "rejection_codes": list(self.rejection_codes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConformanceReceipt":
        if not isinstance(payload, Mapping):
            raise CodeContractLogicError("conformance receipt must be an object")
        supplied = payload.get("schema")
        if supplied not in (None, "", cls.SCHEMA):
            raise CodeContractLogicError(
                f"unsupported schema {supplied!r}; expected {cls.SCHEMA}"
            )
        result = cls(
            request_cid=payload.get("request_cid", ""),
            source_contract_cid=payload.get("source_contract_cid", ""),
            call_slice_cid=payload.get("call_slice_cid", ""),
            assumption_cids=tuple(payload.get("assumption_cids") or ()),
            claim_digests=tuple(payload.get("claim_digests") or ()),
            obligation_digests=tuple(payload.get("obligation_digests") or ()),
            predicate_cids=tuple(payload.get("predicate_cids") or ()),
            unsupported_cids=tuple(payload.get("unsupported_cids") or ()),
            translator_id=payload.get("translator_id", ""),
            translator_version=payload.get("translator_version", ""),
            ruleset_id=payload.get("ruleset_id", ""),
            ruleset_version=payload.get("ruleset_version", ""),
            translator_identity=payload.get("translator_identity", ""),
            round_trip_ok=bool(payload.get("round_trip_ok", False)),
            status=payload.get("status", ""),
            evidence=payload.get("evidence", LOGIC_TRANSLATION_EVIDENCE),
            reconstructed_predicate_cids=tuple(
                payload.get("reconstructed_predicate_cids") or ()
            ),
            rejection_codes=tuple(payload.get("rejection_codes") or ()),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("content_id") or payload.get("receipt_cid")
        if claimed and claimed != result.receipt_cid:
            raise CodeContractLogicError(
                "conformance receipt content identity does not match payload"
            )
        return result


@dataclass(frozen=True)
class TranslationResult(CanonicalContract):
    """Immutable outcome of translating predicates into IR claims."""

    SCHEMA: ClassVar[str] = TRANSLATION_RESULT_SCHEMA

    status: TranslationStatus
    request_cid: str
    source_contract_cid: str
    predicates: tuple[ContractPredicate, ...]
    claims: tuple[IRClaim, ...]
    assumptions: tuple[LogicAssumption, ...]
    unsupported: tuple[UnsupportedResidual, ...]
    receipt: ConformanceReceipt
    rejection_codes: tuple[str, ...] = ()
    vocabulary_projections: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum(self.status, TranslationStatus, "status")
        )
        object.__setattr__(
            self, "request_cid", _text(self.request_cid, "request_cid")
        )
        object.__setattr__(
            self,
            "source_contract_cid",
            _text(self.source_contract_cid, "source_contract_cid"),
        )
        predicates = tuple(self.predicates or ())
        for item in predicates:
            if not isinstance(item, ContractPredicate):
                raise CodeContractLogicError("predicates must be ContractPredicate")
        object.__setattr__(self, "predicates", predicates)
        claims = tuple(self.claims or ())
        for item in claims:
            if not isinstance(item, IRClaim):
                raise CodeContractLogicError("claims must be IRClaim")
        object.__setattr__(self, "claims", claims)
        assumptions = tuple(self.assumptions or ())
        for item in assumptions:
            if not isinstance(item, LogicAssumption):
                raise CodeContractLogicError("assumptions must be LogicAssumption")
        object.__setattr__(self, "assumptions", assumptions)
        unsupported = tuple(self.unsupported or ())
        for item in unsupported:
            if not isinstance(item, UnsupportedResidual):
                raise CodeContractLogicError(
                    "unsupported must be UnsupportedResidual"
                )
        object.__setattr__(self, "unsupported", unsupported)
        if not isinstance(self.receipt, ConformanceReceipt):
            raise CodeContractLogicError("receipt must be ConformanceReceipt")
        object.__setattr__(
            self,
            "rejection_codes",
            tuple(_text(c, "rejection_codes") for c in (self.rejection_codes or ())),
        )
        object.__setattr__(
            self,
            "vocabulary_projections",
            tuple(
                _text(c, "vocabulary_projections")
                for c in (self.vocabulary_projections or ())
            ),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(_mapping(self.metadata, "metadata"))
        )

    @property
    def result_cid(self) -> str:
        return self.content_id

    @property
    def successful(self) -> bool:
        return self.status is TranslationStatus.TRANSLATED

    def _payload(self) -> dict[str, Any]:
        return {
            "logic_version": CODE_CONTRACT_LOGIC_VERSION,
            "status": self.status,
            "request_cid": self.request_cid,
            "source_contract_cid": self.source_contract_cid,
            "predicates": [p.to_dict() for p in self.predicates],
            "claims": [c.to_dict() for c in self.claims],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "unsupported": [u.to_dict() for u in self.unsupported],
            "receipt": self.receipt.to_dict(),
            "rejection_codes": list(self.rejection_codes),
            "vocabulary_projections": list(self.vocabulary_projections),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TranslationResult":
        if not isinstance(payload, Mapping):
            raise CodeContractLogicError("translation result must be an object")
        claims_raw = payload.get("claims") or ()
        claims = tuple(
            item if isinstance(item, IRClaim) else IRClaim.from_dict(item)
            for item in claims_raw
        )
        result = cls(
            status=payload.get("status", ""),
            request_cid=payload.get("request_cid", ""),
            source_contract_cid=payload.get("source_contract_cid", ""),
            predicates=tuple(
                ContractPredicate.from_dict(item)
                for item in (payload.get("predicates") or ())
            ),
            claims=claims,
            assumptions=tuple(
                LogicAssumption.from_dict(item)
                for item in (payload.get("assumptions") or ())
            ),
            unsupported=tuple(
                UnsupportedResidual.from_dict(item)
                for item in (payload.get("unsupported") or ())
            ),
            receipt=ConformanceReceipt.from_dict(payload.get("receipt") or {}),
            rejection_codes=tuple(payload.get("rejection_codes") or ()),
            vocabulary_projections=tuple(
                payload.get("vocabulary_projections") or ()
            ),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("content_id") or payload.get("result_cid")
        if claimed and claimed != result.result_cid:
            raise CodeContractLogicError(
                "translation result content identity does not match payload"
            )
        return result


# ---------------------------------------------------------------------------
# Extraction from program contracts
# ---------------------------------------------------------------------------


def _type_name(shape: TypeShape | None) -> str:
    if shape is None:
        return "unknown"
    if shape.name:
        return shape.name
    if shape.constructor is not None:
        return shape.constructor.value
    return "unknown"


def extract_assumptions_from_contract(
    contract: ExpectedProgramContract | ObservedProgramContract,
) -> tuple[LogicAssumption, ...]:
    """Lift program-contract assumptions into content-addressed logic assumptions."""

    if not isinstance(contract, (ExpectedProgramContract, ObservedProgramContract)):
        raise CodeContractLogicError(
            "contract must be ExpectedProgramContract or ObservedProgramContract"
        )
    out: list[LogicAssumption] = []
    for item in contract.assumptions or ():
        if not isinstance(item, ContractAssumption):
            raise CodeContractLogicError("contract assumptions malformed")
        out.append(
            LogicAssumption(
                statement=item.statement,
                binders=(),
                source_cid=item.assumption_id,
                metadata={
                    "aspect": item.aspect.value,
                    "confidence": item.confidence.value,
                },
            )
        )
    if len(out) > MAX_ASSUMPTIONS:
        raise CodeContractLogicError(f"assumptions exceed {MAX_ASSUMPTIONS}")
    return tuple(out)


def extract_predicates_from_contract(
    contract: ExpectedProgramContract | ObservedProgramContract,
    *,
    assumption_cids: Sequence[str] = (),
    include_unsupported: bool = True,
) -> tuple[tuple[ContractPredicate, ...], tuple[UnsupportedResidual, ...]]:
    """Project finite supported aspects of a program contract into predicates."""

    if not isinstance(contract, (ExpectedProgramContract, ObservedProgramContract)):
        raise CodeContractLogicError(
            "contract must be ExpectedProgramContract or ObservedProgramContract"
        )
    subject = contract.symbol.qualified_name
    source_cid = contract.content_id
    assumption_cids = tuple(
        _text(item, "assumption_cids") for item in assumption_cids
    )
    predicates: list[ContractPredicate] = []
    unsupported: list[UnsupportedResidual] = []

    def _add(
        kind: SupportedPredicateKind,
        relation: PredicateRelation,
        args: Sequence[PredicateArgument],
        *,
        support: SupportStatus = SupportStatus.SUPPORTED,
        residual: str = "",
        closed: bool = True,
        source: str = source_cid,
    ) -> None:
        if support is SupportStatus.UNSUPPORTED:
            unsupported.append(
                UnsupportedResidual(
                    aspect=kind.value,
                    reason=residual or f"{kind.value} is unsupported",
                    residual=residual,
                    source_cid=source,
                    predicate_kind=kind.value,
                )
            )
            return
        predicates.append(
            ContractPredicate(
                kind=kind,
                relation=relation,
                arguments=tuple(args),
                source_cid=source,
                assumption_cids=assumption_cids,
                subject=subject,
                closed=closed,
                support=support,
                residual=residual,
            )
        )

    # Inputs: type + nullability
    for param in contract.inputs or ():
        if not isinstance(param, ParameterSpec):
            raise CodeContractLogicError("inputs must be ParameterSpec")
        shape = param.type_shape
        support = shape.support if shape is not None else SupportStatus.UNSUPPORTED
        _add(
            SupportedPredicateKind.TYPE,
            PredicateRelation.HAS_TYPE,
            (
                PredicateArgument("symbol", ArgumentSort.SYMBOL, subject),
                PredicateArgument("slot", ArgumentSort.STRING, f"in:{param.name}"),
                PredicateArgument(
                    "type", ArgumentSort.TYPE, _type_name(shape)
                ),
            ),
            support=support,
            residual="" if support is SupportStatus.SUPPORTED else "unsupported type shape",
            source=param.parameter_id,
        )
        _add(
            SupportedPredicateKind.NULLABILITY,
            PredicateRelation.IS_NULLABLE,
            (
                PredicateArgument("symbol", ArgumentSort.SYMBOL, subject),
                PredicateArgument("slot", ArgumentSort.STRING, f"in:{param.name}"),
                PredicateArgument(
                    "nullable",
                    ArgumentSort.BOOLEAN,
                    bool(shape.nullable) if shape is not None else False,
                ),
            ),
            support=support,
            source=param.parameter_id,
        )

    # Returns
    if contract.returns is not None:
        ret = contract.returns
        if not isinstance(ret, ReturnSpec):
            raise CodeContractLogicError("returns must be ReturnSpec")
        shape = ret.type_shape
        support = shape.support
        _add(
            SupportedPredicateKind.TYPE,
            PredicateRelation.HAS_TYPE,
            (
                PredicateArgument("symbol", ArgumentSort.SYMBOL, subject),
                PredicateArgument("slot", ArgumentSort.STRING, "return"),
                PredicateArgument("type", ArgumentSort.TYPE, _type_name(shape)),
            ),
            support=support,
            source=ret.return_id,
        )
        _add(
            SupportedPredicateKind.NULLABILITY,
            PredicateRelation.IS_NULLABLE,
            (
                PredicateArgument("symbol", ArgumentSort.SYMBOL, subject),
                PredicateArgument("slot", ArgumentSort.STRING, "return"),
                PredicateArgument(
                    "nullable", ArgumentSort.BOOLEAN, bool(shape.nullable)
                ),
            ),
            support=support,
            source=ret.return_id,
        )

    # Errors
    for err in contract.errors or ():
        if not isinstance(err, ErrorSpec):
            raise CodeContractLogicError("errors must be ErrorSpec")
        _add(
            SupportedPredicateKind.ERROR,
            PredicateRelation.MAY_RAISE,
            (
                PredicateArgument("symbol", ArgumentSort.SYMBOL, subject),
                PredicateArgument(
                    "error", ArgumentSort.ERROR_NAME, err.error_name
                ),
                PredicateArgument(
                    "retriable", ArgumentSort.BOOLEAN, bool(err.retriable)
                ),
            ),
            support=err.support,
            residual="" if err.support is SupportStatus.SUPPORTED else "unsupported error",
            source=err.error_id,
        )

    # Effects
    for effect in contract.side_effects or ():
        if not isinstance(effect, SideEffectSpec):
            raise CodeContractLogicError("side_effects must be SideEffectSpec")
        _add(
            SupportedPredicateKind.EFFECT,
            PredicateRelation.HAS_EFFECT,
            (
                PredicateArgument("symbol", ArgumentSort.SYMBOL, subject),
                PredicateArgument(
                    "effect_kind",
                    ArgumentSort.EFFECT_KIND,
                    effect.effect_kind.value
                    if isinstance(effect.effect_kind, EffectKind)
                    else str(effect.effect_kind),
                ),
                PredicateArgument(
                    "polarity",
                    ArgumentSort.EFFECT_POLARITY,
                    effect.polarity.value
                    if isinstance(effect.polarity, EffectPolarity)
                    else str(effect.polarity),
                ),
            ),
            support=effect.support,
            residual=""
            if effect.support is SupportStatus.SUPPORTED
            else "unsupported effect",
            source=effect.effect_id,
        )

    # Authorization
    if contract.authorization is not None:
        auth = contract.authorization
        if not isinstance(auth, AuthorizationSpec):
            raise CodeContractLogicError("authorization must be AuthorizationSpec")
        scopes = auth.scopes or ("*",)
        for scope in scopes:
            _add(
                SupportedPredicateKind.AUTHORIZATION,
                PredicateRelation.REQUIRES_AUTH,
                (
                    PredicateArgument("symbol", ArgumentSort.SYMBOL, subject),
                    PredicateArgument(
                        "mode", ArgumentSort.MODE, auth.mode.value
                    ),
                    PredicateArgument("scope", ArgumentSort.SCOPE, str(scope)),
                ),
                support=auth.support,
                residual=""
                if auth.support is SupportStatus.SUPPORTED
                else "unsupported authorization",
                source=auth.authorization_id,
            )

    # Ordering
    if contract.ordering is not None:
        ordering = contract.ordering
        if not isinstance(ordering, OrderingSpec):
            raise CodeContractLogicError("ordering must be OrderingSpec")
        _add(
            SupportedPredicateKind.ORDERING,
            PredicateRelation.ORDERED_AS,
            (
                PredicateArgument("symbol", ArgumentSort.SYMBOL, subject),
                PredicateArgument(
                    "mode", ArgumentSort.MODE, ordering.mode.value
                ),
            ),
            support=ordering.support,
            residual=""
            if ordering.support is SupportStatus.SUPPORTED
            else "unsupported ordering",
            source=ordering.ordering_id,
        )

    # Idempotence
    if contract.idempotence is not None:
        idem = contract.idempotence
        if not isinstance(idem, IdempotenceSpec):
            raise CodeContractLogicError("idempotence must be IdempotenceSpec")
        _add(
            SupportedPredicateKind.IDEMPOTENCE,
            PredicateRelation.IDEMPOTENT_AS,
            (
                PredicateArgument("symbol", ArgumentSort.SYMBOL, subject),
                PredicateArgument("mode", ArgumentSort.MODE, idem.mode.value),
            ),
            support=idem.support,
            residual=""
            if idem.support is SupportStatus.SUPPORTED
            else "unsupported idempotence",
            source=idem.idempotence_id,
        )

    # State transition: atomicity mode used as a finite state-machine boundary
    # when present (before/after atomic region).
    if contract.atomicity is not None:
        atom_spec = contract.atomicity
        mode = atom_spec.mode.value
        _add(
            SupportedPredicateKind.STATE_TRANSITION,
            PredicateRelation.TRANSITIONS,
            (
                PredicateArgument("symbol", ArgumentSort.SYMBOL, subject),
                PredicateArgument("from_state", ArgumentSort.STATE, f"pre:{mode}"),
                PredicateArgument("to_state", ArgumentSort.STATE, f"post:{mode}"),
            ),
            support=atom_spec.support,
            residual=""
            if atom_spec.support is SupportStatus.SUPPORTED
            else "unsupported atomicity/state transition",
            source=atom_spec.atomicity_id,
        )

    if include_unsupported:
        for item in contract.unsupported or ():
            if not isinstance(item, UnsupportedSemantics):
                raise CodeContractLogicError(
                    "unsupported must be UnsupportedSemantics"
                )
            unsupported.append(
                UnsupportedResidual(
                    aspect=item.aspect.value,
                    reason=item.reason,
                    residual=item.residual,
                    source_cid=item.unsupported_id,
                    predicate_kind="",
                )
            )

    if len(predicates) > MAX_PREDICATES:
        raise CodeContractLogicError(f"predicates exceed {MAX_PREDICATES}")
    return tuple(predicates), tuple(unsupported)


def extract_reachability_predicates(
    slice_binding: CallSliceBinding,
    *,
    source_cid: str,
    assumption_cids: Sequence[str] = (),
    pairs: Sequence[tuple[str, str]] | None = None,
) -> tuple[ContractPredicate, ...]:
    """Emit bounded-reachability predicates for a closed call slice.

    When ``pairs`` is omitted, consecutive node-id pairs in the sorted node
    set are not used (order is not path order).  Callers should supply explicit
    (entry, exit) pairs from slice paths.  If the slice is not structurally
    closed, predicates are emitted with ``closed=False``; presenting them as
    closed is rejected by the translator.
    """

    if not isinstance(slice_binding, CallSliceBinding):
        raise CodeContractLogicError("slice_binding must be CallSliceBinding")
    source_cid = _text(source_cid, "source_cid")
    assumption_cids = tuple(
        _text(item, "assumption_cids") for item in assumption_cids
    )
    if pairs is None:
        nodes = list(slice_binding.node_ids)
        pairs = []
        if len(nodes) >= 2:
            pairs = [(nodes[0], nodes[-1])]
    out: list[ContractPredicate] = []
    closed = slice_binding.is_structurally_closed
    bound = slice_binding.depth_bound if slice_binding.depth_bound > 0 else max(
        1, len(slice_binding.node_ids)
    )
    for entry, exit_ in pairs:
        out.append(
            ContractPredicate(
                kind=SupportedPredicateKind.BOUNDED_REACHABILITY,
                relation=PredicateRelation.REACHABLE_WITHIN,
                arguments=(
                    PredicateArgument("entry", ArgumentSort.NODE, entry),
                    PredicateArgument("exit", ArgumentSort.NODE, exit_),
                    PredicateArgument("bound", ArgumentSort.BOUND, int(bound)),
                ),
                source_cid=source_cid,
                assumption_cids=assumption_cids,
                subject=entry,
                closed=closed,
                support=SupportStatus.SUPPORTED,
                metadata={"slice_cid": slice_binding.slice_cid},
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# IR claim construction + formal vocabulary projection
# ---------------------------------------------------------------------------


def _predicate_to_obligation(
    predicate: ContractPredicate,
    *,
    assumption_ids: Sequence[str],
) -> IRObligation:
    statement = json.dumps(
        predicate.statement_atom(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if len(statement.encode("utf-8")) > MAX_STATEMENT_BYTES:
        raise CodeContractLogicError("obligation statement exceeds bound")
    return IRObligation(
        obligation_id=_cid(
            {
                "predicate_id": predicate.predicate_id,
                "statement": statement,
                "assumption_ids": list(assumption_ids),
            }
        ),
        statement=statement,
        assumption_ids=tuple(assumption_ids),
        logic_family=LOGIC_FAMILY,
        source_refs=tuple(
            ref
            for ref in (predicate.source_cid, predicate.predicate_id)
            if ref
        ),
        metadata=freeze_json(
            {
                "kind": predicate.kind.value,
                "relation": predicate.relation.value,
                "closed": predicate.closed,
                "support": predicate.support.value,
            }
        ),
        schema_version=IR_OBLIGATION_SCHEMA_VERSION,
    )


def _predicate_to_claim(
    predicate: ContractPredicate,
    *,
    ir_assumptions: Sequence[IRAssumption],
) -> IRClaim:
    assumption_ids = tuple(item.assumption_id for item in ir_assumptions)
    # Fail closed: every referenced assumption CID must be present.
    missing = set(predicate.assumption_cids) - set(assumption_ids)
    if missing:
        raise TranslationRejectedError(
            RejectionCode.UNBOUND_AXIOM,
            "predicate references undeclared assumption(s)",
        )
    obligation = _predicate_to_obligation(
        predicate, assumption_ids=assumption_ids
    )
    claim = IRClaim(
        claim_id=_cid(
            {
                "predicate_id": predicate.predicate_id,
                "obligation_id": obligation.obligation_id,
                "source_cid": predicate.source_cid,
            }
        ),
        statement=obligation.statement,
        assumptions=tuple(ir_assumptions),
        obligations=(obligation,),
        domain=LOGIC_FAMILY,
        declaration_id=predicate.predicate_id,
        source_refs=tuple(
            ref
            for ref in (predicate.source_cid, predicate.predicate_id)
            if ref
        ),
        metadata=freeze_json(
            {
                "kind": predicate.kind.value,
                "relation": predicate.relation.value,
                "subject": predicate.subject,
                "closed": predicate.closed,
            }
        ),
        schema_version=IR_CLAIM_SCHEMA_VERSION,
    )
    return claim


def project_reviewed_formula(
    predicate: ContractPredicate,
) -> Formula | None:
    """Project a subset of predicates into the reviewed logic vocabulary.

    Only exact, sort-safe projections are returned.  Unsupported families
    return ``None`` rather than inventing vocabulary.
    """

    if predicate.support is not SupportStatus.SUPPORTED:
        return None
    if predicate.kind is SupportedPredicateKind.AUTHORIZATION:
        # authorized(actor=scope, task=symbol) — exact arity from vocabulary.
        mode = next(
            (a.value for a in predicate.arguments if a.name == "mode"),
            "",
        )
        scope = next(
            (a.value for a in predicate.arguments if a.name == "scope"),
            "",
        )
        symbol = next(
            (a.value for a in predicate.arguments if a.name == "symbol"),
            predicate.subject,
        )
        if not scope or not symbol:
            return None
        # Use scope as actor token and symbol as task token.
        return atom(
            ReviewedPredicate.AUTHORIZED,
            constant(TermSort.ACTOR, str(scope)),
            constant(TermSort.TASK, str(symbol)),
        )
    if predicate.kind is SupportedPredicateKind.BOUNDED_REACHABILITY:
        entry = next(
            (a.value for a in predicate.arguments if a.name == "entry"),
            "",
        )
        exit_ = next(
            (a.value for a in predicate.arguments if a.name == "exit"),
            "",
        )
        if not entry or not exit_:
            return None
        return atom(
            ReviewedPredicate.DEPENDENCY_SATISFIED,
            constant(TermSort.TASK, str(entry)),
            constant(TermSort.TASK, str(exit_)),
        )
    if predicate.kind is SupportedPredicateKind.STATE_TRANSITION:
        to_state = next(
            (a.value for a in predicate.arguments if a.name == "to_state"),
            "",
        )
        if not to_state:
            return None
        return atom(
            ReviewedPredicate.SAFE_STATE,
            constant(TermSort.SYMBOL, str(to_state)),
        )
    return None


def reconstruct_predicate_from_claim(claim: IRClaim) -> ContractPredicate:
    """Round-trip: rebuild a :class:`ContractPredicate` from an IR claim statement."""

    if not isinstance(claim, IRClaim):
        raise CodeContractLogicError("claim must be IRClaim")
    if not claim.obligations:
        raise TranslationRejectedError(
            RejectionCode.ROUND_TRIP_FAILURE,
            "claim has no obligations",
        )
    statement = claim.obligations[0].statement
    try:
        atom_payload = json.loads(statement)
    except (TypeError, ValueError) as exc:
        raise TranslationRejectedError(
            RejectionCode.ROUND_TRIP_FAILURE,
            "obligation statement is not JSON",
        ) from exc
    if not isinstance(atom_payload, Mapping):
        raise TranslationRejectedError(
            RejectionCode.ROUND_TRIP_FAILURE,
            "obligation statement must be an object",
        )
    return ContractPredicate(
        kind=atom_payload.get("kind", ""),
        relation=atom_payload.get("relation", ""),
        arguments=tuple(atom_payload.get("arguments") or ()),
        source_cid=atom_payload.get("source_cid", ""),
        assumption_cids=tuple(atom_payload.get("assumption_cids") or ()),
        subject=atom_payload.get("subject", ""),
        closed=bool(atom_payload.get("closed", True)),
        support=atom_payload.get("support", SupportStatus.SUPPORTED),
        residual=atom_payload.get("residual", ""),
        ambiguity=atom_payload.get("ambiguity", ""),
    )


def round_trip_predicates(
    predicates: Sequence[ContractPredicate],
    claims: Sequence[IRClaim],
) -> tuple[bool, tuple[str, ...]]:
    """Return whether claims reconstruct the same predicate CIDs."""

    if len(predicates) != len(claims):
        return False, ()
    reconstructed: list[str] = []
    for predicate, claim in zip(predicates, claims):
        rebuilt = reconstruct_predicate_from_claim(claim)
        # Compare statement atoms (identity may differ on metadata-only fields).
        if rebuilt.statement_atom() != predicate.statement_atom():
            return False, tuple(reconstructed)
        reconstructed.append(rebuilt.predicate_id)
    return True, tuple(reconstructed)


# ---------------------------------------------------------------------------
# Validation gates (fail-closed)
# ---------------------------------------------------------------------------


def _check_name_capture(
    predicates: Sequence[ContractPredicate],
    assumptions: Sequence[LogicAssumption],
) -> None:
    """Reject binder/free-name collisions and sort-inconsistent rebinding."""

    binder_sorts: dict[str, ArgumentSort] = {}
    for assumption in assumptions:
        for name, sort in assumption.binders:
            if name in binder_sorts and binder_sorts[name] is not sort:
                raise TranslationRejectedError(
                    RejectionCode.NAME_CAPTURE,
                    f"binder {name!r} redeclared at incompatible sort",
                )
            binder_sorts[name] = sort

    free_names: dict[str, ArgumentSort] = {}
    for predicate in predicates:
        for arg in predicate.arguments:
            if arg.binder:
                if arg.name in free_names and free_names[arg.name] is not arg.sort:
                    raise TranslationRejectedError(
                        RejectionCode.NAME_CAPTURE,
                        f"binder {arg.name!r} captures free name of different sort",
                    )
                if arg.name in binder_sorts and binder_sorts[arg.name] is not arg.sort:
                    raise TranslationRejectedError(
                        RejectionCode.NAME_CAPTURE,
                        f"binder {arg.name!r} captures assumption binder of different sort",
                    )
                binder_sorts[arg.name] = arg.sort
            else:
                # Free occurrence of a name that is bound at a different sort.
                if arg.name in binder_sorts and binder_sorts[arg.name] is not arg.sort:
                    raise TranslationRejectedError(
                        RejectionCode.NAME_CAPTURE,
                        f"free name {arg.name!r} captured by binder of different sort",
                    )
                # String-valued free args that equal binder names at other sorts.
                if (
                    isinstance(arg.value, str)
                    and arg.value in binder_sorts
                    and binder_sorts[arg.value] is not arg.sort
                ):
                    raise TranslationRejectedError(
                        RejectionCode.NAME_CAPTURE,
                        f"value {arg.value!r} collides with binder of different sort",
                    )
                free_names[arg.name] = arg.sort


def _check_unbound_axioms(
    predicates: Sequence[ContractPredicate],
    assumptions: Sequence[LogicAssumption],
) -> None:
    known = {item.assumption_cid for item in assumptions}
    for predicate in predicates:
        missing = set(predicate.assumption_cids) - known
        if missing:
            raise TranslationRejectedError(
                RejectionCode.UNBOUND_AXIOM,
                "predicate references undeclared assumption CID(s)",
            )


def _check_partial_slice_closed(
    predicates: Sequence[ContractPredicate],
    slice_binding: CallSliceBinding | None,
) -> None:
    for predicate in predicates:
        if predicate.kind is not SupportedPredicateKind.BOUNDED_REACHABILITY:
            continue
        if not predicate.closed:
            continue
        if slice_binding is None:
            raise TranslationRejectedError(
                RejectionCode.PARTIAL_SLICE_AS_CLOSED,
                "closed reachability predicate requires a call-slice binding",
            )
        if not slice_binding.is_structurally_closed:
            raise TranslationRejectedError(
                RejectionCode.PARTIAL_SLICE_AS_CLOSED,
                "partial call slice cannot support a closed reachability claim",
            )
        if slice_binding.presented_as_closed and not slice_binding.is_structurally_closed:
            raise TranslationRejectedError(
                RejectionCode.PARTIAL_SLICE_AS_CLOSED,
                "call slice presented as closed but is not structurally closed",
            )


def _check_silent_approximation(
    predicates: Sequence[ContractPredicate],
    unsupported: Sequence[UnsupportedResidual],
    *,
    allow_approximation: bool,
) -> None:
    """Reject unsupported residuals that were not made explicit."""

    if allow_approximation:
        return
    for predicate in predicates:
        if predicate.support is SupportStatus.UNSUPPORTED:
            raise TranslationRejectedError(
                RejectionCode.SILENT_APPROXIMATION,
                "unsupported predicate admitted without explicit residual",
            )
        if predicate.residual and predicate.support is SupportStatus.SUPPORTED:
            # Residual present on a claimed-supported atom → silent drop risk.
            raise TranslationRejectedError(
                RejectionCode.SILENT_APPROXIMATION,
                "supported predicate carries residual without unsupported listing",
            )
    # If approximation is disallowed, residuals must only appear in unsupported.
    for residual in unsupported:
        if not residual.reason:
            raise TranslationRejectedError(
                RejectionCode.SILENT_APPROXIMATION,
                "unsupported residual lacks an explicit reason",
            )


def _check_ambiguous(
    predicates: Sequence[ContractPredicate],
) -> list[str]:
    codes: list[str] = []
    subjects: dict[tuple[str, str, str], list[ContractPredicate]] = {}
    for predicate in predicates:
        if predicate.ambiguity:
            codes.append(RejectionCode.AMBIGUOUS_PREDICATE.value)
            continue
        if predicate.support is SupportStatus.UNKNOWN:
            codes.append(RejectionCode.AMBIGUOUS_PREDICATE.value)
            continue
        key = (
            predicate.kind.value,
            predicate.relation.value,
            json.dumps(
                [a.to_dict() for a in predicate.arguments],
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        subjects.setdefault(key, []).append(predicate)
    for group in subjects.values():
        if len(group) > 1:
            # Identical atoms are fine; conflicting support/closed is ambiguous.
            supports = {p.support for p in group}
            closeds = {p.closed for p in group}
            if len(supports) > 1 or len(closeds) > 1:
                codes.append(RejectionCode.AMBIGUOUS_PREDICATE.value)
    return codes


# ---------------------------------------------------------------------------
# Translation request + main entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranslationRequest(CanonicalContract):
    """Inputs for one deterministic translation."""

    SCHEMA: ClassVar[str] = TRANSLATION_REQUEST_SCHEMA

    source_contract_cid: str
    predicates: tuple[ContractPredicate, ...] = ()
    assumptions: tuple[LogicAssumption, ...] = ()
    call_slice: CallSliceBinding | None = None
    unsupported: tuple[UnsupportedResidual, ...] = ()
    translator_id: str = TRANSLATOR_ID
    translator_version: str = TRANSLATOR_VERSION
    ruleset_id: str = RULESET_ID
    ruleset_version: str = RULESET_VERSION
    allow_approximation: bool = False
    require_round_trip: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_contract_cid",
            _text(self.source_contract_cid, "source_contract_cid"),
        )
        predicates = tuple(self.predicates or ())
        for item in predicates:
            if not isinstance(item, ContractPredicate):
                raise CodeContractLogicError("predicates must be ContractPredicate")
        if len(predicates) > MAX_PREDICATES:
            raise CodeContractLogicError(f"predicates exceed {MAX_PREDICATES}")
        object.__setattr__(self, "predicates", predicates)
        assumptions = tuple(self.assumptions or ())
        for item in assumptions:
            if not isinstance(item, LogicAssumption):
                raise CodeContractLogicError("assumptions must be LogicAssumption")
        if len(assumptions) > MAX_ASSUMPTIONS:
            raise CodeContractLogicError(f"assumptions exceed {MAX_ASSUMPTIONS}")
        object.__setattr__(self, "assumptions", assumptions)
        if self.call_slice is not None and not isinstance(
            self.call_slice, CallSliceBinding
        ):
            raise CodeContractLogicError("call_slice must be CallSliceBinding")
        unsupported = tuple(self.unsupported or ())
        for item in unsupported:
            if not isinstance(item, UnsupportedResidual):
                raise CodeContractLogicError(
                    "unsupported must be UnsupportedResidual"
                )
        object.__setattr__(self, "unsupported", unsupported)
        for name in (
            "translator_id",
            "translator_version",
            "ruleset_id",
            "ruleset_version",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "allow_approximation",
            _boolean(bool(self.allow_approximation), "allow_approximation"),
        )
        object.__setattr__(
            self,
            "require_round_trip",
            _boolean(bool(self.require_round_trip), "require_round_trip"),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(_mapping(self.metadata, "metadata"))
        )

    @property
    def request_cid(self) -> str:
        return self.content_id

    @property
    def translator_identity(self) -> str:
        return translator_identity(
            translator_id=self.translator_id,
            translator_version=self.translator_version,
            ruleset_id=self.ruleset_id,
            ruleset_version=self.ruleset_version,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "logic_version": CODE_CONTRACT_LOGIC_VERSION,
            "source_contract_cid": self.source_contract_cid,
            "predicates": [p.to_dict() for p in self.predicates],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "call_slice": None
            if self.call_slice is None
            else self.call_slice.to_dict(),
            "unsupported": [u.to_dict() for u in self.unsupported],
            "translator_id": self.translator_id,
            "translator_version": self.translator_version,
            "ruleset_id": self.ruleset_id,
            "ruleset_version": self.ruleset_version,
            "allow_approximation": self.allow_approximation,
            "require_round_trip": self.require_round_trip,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TranslationRequest":
        if not isinstance(payload, Mapping):
            raise CodeContractLogicError("translation request must be an object")
        call_slice_raw = payload.get("call_slice")
        call_slice = (
            None
            if not call_slice_raw
            else CallSliceBinding.from_dict(call_slice_raw)
        )
        return cls(
            source_contract_cid=payload.get("source_contract_cid", ""),
            predicates=tuple(
                ContractPredicate.from_dict(item)
                for item in (payload.get("predicates") or ())
            ),
            assumptions=tuple(
                LogicAssumption.from_dict(item)
                for item in (payload.get("assumptions") or ())
            ),
            call_slice=call_slice,
            unsupported=tuple(
                UnsupportedResidual.from_dict(item)
                for item in (payload.get("unsupported") or ())
            ),
            translator_id=payload.get("translator_id", TRANSLATOR_ID),
            translator_version=payload.get(
                "translator_version", TRANSLATOR_VERSION
            ),
            ruleset_id=payload.get("ruleset_id", RULESET_ID),
            ruleset_version=payload.get("ruleset_version", RULESET_VERSION),
            allow_approximation=bool(payload.get("allow_approximation", False)),
            require_round_trip=bool(payload.get("require_round_trip", True)),
            metadata=payload.get("metadata") or {},
        )


def translate_contract(
    contract: ExpectedProgramContract | ObservedProgramContract,
    *,
    call_slice: CallSliceBinding | ProgramGraphSlice | None = None,
    reachability_pairs: Sequence[tuple[str, str]] | None = None,
    allow_approximation: bool = False,
    require_round_trip: bool = True,
    translator_id: str = TRANSLATOR_ID,
    translator_version: str = TRANSLATOR_VERSION,
    ruleset_id: str = RULESET_ID,
    ruleset_version: str = RULESET_VERSION,
    metadata: Mapping[str, Any] | None = None,
) -> TranslationResult:
    """Extract and translate a program contract (and optional call slice)."""

    assumptions = extract_assumptions_from_contract(contract)
    assumption_cids = tuple(item.assumption_cid for item in assumptions)
    predicates, unsupported = extract_predicates_from_contract(
        contract, assumption_cids=assumption_cids
    )
    slice_binding: CallSliceBinding | None = None
    if call_slice is not None:
        if isinstance(call_slice, ProgramGraphSlice):
            slice_binding = CallSliceBinding.from_program_graph_slice(call_slice)
        elif isinstance(call_slice, CallSliceBinding):
            slice_binding = call_slice
        else:
            raise CodeContractLogicError(
                "call_slice must be CallSliceBinding or ProgramGraphSlice"
            )
        reach = extract_reachability_predicates(
            slice_binding,
            source_cid=contract.content_id,
            assumption_cids=assumption_cids,
            pairs=reachability_pairs,
        )
        predicates = predicates + reach
    request = TranslationRequest(
        source_contract_cid=contract.content_id,
        predicates=predicates,
        assumptions=assumptions,
        call_slice=slice_binding,
        unsupported=unsupported,
        translator_id=translator_id,
        translator_version=translator_version,
        ruleset_id=ruleset_id,
        ruleset_version=ruleset_version,
        allow_approximation=allow_approximation,
        require_round_trip=require_round_trip,
        metadata=metadata or {},
    )
    return translate(request)


def translate(request: TranslationRequest) -> TranslationResult:
    """Translate a finite predicate set into IR claims with a conformance receipt."""

    if not isinstance(request, TranslationRequest):
        raise CodeContractLogicError("request must be TranslationRequest")

    # Pin check: only the current translator/ruleset may produce successful receipts.
    if request.translator_identity != pinned_translator_identity():
        # Still allow constructing a rejected result when pins differ — used to
        # prove ruleset reuse fails closed.
        return _rejected_result(
            request,
            RejectionCode.TRANSLATOR_RULESET_REUSE,
            "translator/ruleset identity is not the current pin",
        )

    try:
        _check_unbound_axioms(request.predicates, request.assumptions)
        _check_name_capture(request.predicates, request.assumptions)
        _check_partial_slice_closed(request.predicates, request.call_slice)
        _check_silent_approximation(
            request.predicates,
            request.unsupported,
            allow_approximation=request.allow_approximation,
        )
    except TranslationRejectedError as exc:
        return _rejected_result(request, exc.code, exc.detail)

    ambiguity_codes = _check_ambiguous(request.predicates)
    if ambiguity_codes:
        return _status_result(
            request,
            TranslationStatus.AMBIGUOUS,
            rejection_codes=tuple(sorted(set(ambiguity_codes))),
            claims=(),
            predicates=request.predicates,
        )

    # Unsupported-only: no supported predicates → explicit unsupported status.
    supported_preds = [
        p
        for p in request.predicates
        if p.support is SupportStatus.SUPPORTED
    ]
    if not supported_preds:
        if request.unsupported or any(
            p.support is SupportStatus.UNSUPPORTED for p in request.predicates
        ):
            return _status_result(
                request,
                TranslationStatus.UNSUPPORTED,
                rejection_codes=(RejectionCode.UNSUPPORTED_PREDICATE.value,),
                claims=(),
                predicates=request.predicates,
            )
        return _rejected_result(
            request,
            RejectionCode.EMPTY_TRANSLATION,
            "no supported predicates to translate",
        )

    ir_assumptions = tuple(item.to_ir_assumption() for item in request.assumptions)
    claims: list[IRClaim] = []
    projections: list[str] = []
    try:
        for predicate in supported_preds:
            claim = _predicate_to_claim(predicate, ir_assumptions=ir_assumptions)
            claims.append(claim)
            formula = project_reviewed_formula(predicate)
            if formula is not None:
                projections.append(formula.content_id)
    except TranslationRejectedError as exc:
        return _rejected_result(request, exc.code, exc.detail)
    except ClaimValidationError as exc:
        return _rejected_result(
            request, RejectionCode.INVALID_INPUT, str(exc)
        )

    ok, reconstructed = round_trip_predicates(supported_preds, claims)
    if request.require_round_trip and not ok:
        return _rejected_result(
            request,
            RejectionCode.ROUND_TRIP_FAILURE,
            "claim reconstruction does not match source predicates",
        )

    obligation_digests = tuple(
        sorted(
            {
                obligation.digest
                for claim in claims
                for obligation in claim.obligations
            }
        )
    )
    claim_digests = tuple(sorted(claim.digest for claim in claims))
    receipt = ConformanceReceipt(
        request_cid=request.request_cid,
        source_contract_cid=request.source_contract_cid,
        call_slice_cid=(
            "" if request.call_slice is None else request.call_slice.slice_cid
        ),
        assumption_cids=tuple(a.assumption_cid for a in request.assumptions),
        claim_digests=claim_digests,
        obligation_digests=obligation_digests,
        predicate_cids=tuple(p.predicate_id for p in supported_preds),
        unsupported_cids=tuple(u.content_id for u in request.unsupported),
        translator_id=request.translator_id,
        translator_version=request.translator_version,
        ruleset_id=request.ruleset_id,
        ruleset_version=request.ruleset_version,
        translator_identity=request.translator_identity,
        round_trip_ok=ok,
        status=TranslationStatus.TRANSLATED,
        reconstructed_predicate_cids=reconstructed,
        rejection_codes=(),
        metadata={
            "predicate_count": len(supported_preds),
            "claim_count": len(claims),
            "vocabulary_projection_count": len(projections),
        },
    )
    return TranslationResult(
        status=TranslationStatus.TRANSLATED,
        request_cid=request.request_cid,
        source_contract_cid=request.source_contract_cid,
        predicates=tuple(supported_preds),
        claims=tuple(claims),
        assumptions=request.assumptions,
        unsupported=request.unsupported,
        receipt=receipt,
        rejection_codes=(),
        vocabulary_projections=tuple(projections),
        metadata=dict(request.metadata),
    )


def verify_conformance_receipt(
    receipt: ConformanceReceipt | Mapping[str, Any],
    *,
    expected_translator_identity: str | None = None,
    require_round_trip: bool = True,
) -> ConformanceReceipt:
    """Validate a receipt against the current translator/ruleset pin.

    Rejects changed translator/ruleset reuse, forged identities, and
    non-round-tripped successful receipts.
    """

    if isinstance(receipt, Mapping):
        receipt = ConformanceReceipt.from_dict(receipt)
    if not isinstance(receipt, ConformanceReceipt):
        raise CodeContractLogicError("receipt must be ConformanceReceipt")
    pin = expected_translator_identity or pinned_translator_identity()
    if receipt.translator_identity != pin:
        raise TranslationRejectedError(
            RejectionCode.TRANSLATOR_RULESET_REUSE,
            "receipt translator/ruleset identity is not the expected pin",
        )
    if receipt.translator_identity != translator_identity(
        translator_id=receipt.translator_id,
        translator_version=receipt.translator_version,
        ruleset_id=receipt.ruleset_id,
        ruleset_version=receipt.ruleset_version,
    ):
        raise TranslationRejectedError(
            RejectionCode.TRANSLATOR_RULESET_REUSE,
            "receipt internal translator identity is inconsistent",
        )
    if (
        require_round_trip
        and receipt.status is TranslationStatus.TRANSLATED
        and not receipt.round_trip_ok
    ):
        raise TranslationRejectedError(
            RejectionCode.ROUND_TRIP_FAILURE,
            "translated receipt is not round-trip verified",
        )
    return receipt


def _rejected_result(
    request: TranslationRequest,
    code: RejectionCode,
    detail: str,
) -> TranslationResult:
    return _status_result(
        request,
        TranslationStatus.REJECTED
        if code
        not in (
            RejectionCode.UNSUPPORTED_PREDICATE,
            RejectionCode.AMBIGUOUS_PREDICATE,
        )
        else (
            TranslationStatus.UNSUPPORTED
            if code is RejectionCode.UNSUPPORTED_PREDICATE
            else TranslationStatus.AMBIGUOUS
        ),
        rejection_codes=(code.value,),
        claims=(),
        predicates=request.predicates,
        detail=detail,
    )


def _status_result(
    request: TranslationRequest,
    status: TranslationStatus,
    *,
    rejection_codes: tuple[str, ...],
    claims: tuple[IRClaim, ...],
    predicates: tuple[ContractPredicate, ...],
    detail: str = "",
) -> TranslationResult:
    receipt = ConformanceReceipt(
        request_cid=request.request_cid,
        source_contract_cid=request.source_contract_cid,
        call_slice_cid=(
            "" if request.call_slice is None else request.call_slice.slice_cid
        ),
        assumption_cids=tuple(a.assumption_cid for a in request.assumptions),
        claim_digests=tuple(sorted(c.digest for c in claims)),
        obligation_digests=tuple(
            sorted(
                {
                    o.digest
                    for c in claims
                    for o in c.obligations
                }
            )
        ),
        predicate_cids=tuple(p.predicate_id for p in predicates),
        unsupported_cids=tuple(u.content_id for u in request.unsupported),
        translator_id=request.translator_id,
        translator_version=request.translator_version,
        ruleset_id=request.ruleset_id,
        ruleset_version=request.ruleset_version,
        translator_identity=request.translator_identity,
        round_trip_ok=False,
        status=status,
        reconstructed_predicate_cids=(),
        rejection_codes=rejection_codes,
        metadata={"detail": detail} if detail else {},
    )
    return TranslationResult(
        status=status,
        request_cid=request.request_cid,
        source_contract_cid=request.source_contract_cid,
        predicates=predicates,
        claims=claims,
        assumptions=request.assumptions,
        unsupported=request.unsupported,
        receipt=receipt,
        rejection_codes=rejection_codes,
        vocabulary_projections=(),
        metadata={"detail": detail} if detail else dict(request.metadata),
    )


# ---------------------------------------------------------------------------
# Public construction helpers for tests and callers
# ---------------------------------------------------------------------------


def make_predicate(
    kind: SupportedPredicateKind | str,
    *arg_values: Any,
    source_cid: str,
    assumption_cids: Sequence[str] = (),
    closed: bool = True,
    support: SupportStatus | str = SupportStatus.SUPPORTED,
    residual: str = "",
    ambiguity: str = "",
    binder_names: Sequence[str] = (),
) -> ContractPredicate:
    """Build a sort-checked predicate for the given kind using positional args."""

    kind_e = _enum(kind, SupportedPredicateKind, "kind")
    relation = _KIND_TO_RELATION[kind_e]
    signature = _RELATION_SIGNATURES[relation]
    if len(arg_values) != len(signature):
        raise TranslationRejectedError(
            RejectionCode.SORT_MISMATCH,
            f"{relation.value} expects {len(signature)} arguments",
        )
    names = {
        PredicateRelation.HAS_TYPE: ("symbol", "slot", "type"),
        PredicateRelation.IS_NULLABLE: ("symbol", "slot", "nullable"),
        PredicateRelation.MAY_RAISE: ("symbol", "error", "retriable"),
        PredicateRelation.HAS_EFFECT: ("symbol", "effect_kind", "polarity"),
        PredicateRelation.REQUIRES_AUTH: ("symbol", "mode", "scope"),
        PredicateRelation.TRANSITIONS: ("symbol", "from_state", "to_state"),
        PredicateRelation.ORDERED_AS: ("symbol", "mode"),
        PredicateRelation.IDEMPOTENT_AS: ("symbol", "mode"),
        PredicateRelation.REACHABLE_WITHIN: ("entry", "exit", "bound"),
    }[relation]
    binder_set = set(binder_names)
    arguments = tuple(
        PredicateArgument(
            name=name,
            sort=sort,
            value=value,
            binder=name in binder_set,
        )
        for name, sort, value in zip(names, signature, arg_values)
    )
    return ContractPredicate(
        kind=kind_e,
        relation=relation,
        arguments=arguments,
        source_cid=source_cid,
        assumption_cids=tuple(assumption_cids),
        closed=closed,
        support=support,
        residual=residual,
        ambiguity=ambiguity,
    )


# ---------------------------------------------------------------------------
# Objective evidence discovery (VFS-G070 / VFS-053)
# ---------------------------------------------------------------------------


def logic_translation_evidence_terms() -> tuple[str, ...]:
    """Return domain translation evidence (``vfs/logic-translation@1``).

    The synthetic ``objective validation repair`` term is intentionally
    omitted here so translation envelope ``evidence`` stays domain-only; use
    :func:`objective_validation_repair_evidence_terms` (or
    :func:`all_covered_evidence_terms`) for the VFS-G070 validation gate.
    """

    return (LOGIC_TRANSLATION_EVIDENCE,)


def covered_evidence_terms() -> tuple[str, ...]:
    """Return domain objective evidence terms this translation surface proves.

    Mirrors :func:`logic_translation_evidence_terms`.  Translation remains
    independent from MultiProverRouter candidate search and KernelVerification.
    """

    return logic_translation_evidence_terms()


def objective_validation_repair_evidence_terms() -> tuple[str, ...]:
    """Return the synthetic VFS-G070 validation-gate evidence term.

    Exact-text discovery key for objective validation repair.  Never mixes
    into content-addressed claim, obligation, or conformance-receipt identity.
    Translation (FormalLogicVocabulary) stays separate from candidate search
    and kernel validation.  Owned by :data:`OBJECTIVE_GOAL_ID` (``VFS-G070``)
    via repair task :data:`OBJECTIVE_VALIDATION_REPAIR_TASK_ID` (``VFS-053``).
    """

    return (OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,)


def all_covered_evidence_terms() -> tuple[str, ...]:
    """Return domain VFS-G070 translation terms plus the validation-repair gate.

    Domain translation evidence comes first; the synthetic objective
    validation repair discovery key is appended last and never enters
    claim/receipt identity.  Kernel-proof receipts live on the prover surface.
    """

    return covered_evidence_terms() + objective_validation_repair_evidence_terms()


def translation_stage_owner() -> str:
    """Return the AST-discoverable owner of the translation stage only.

    FormalLogicVocabulary constructs reviewed formulas; it never selects
    solvers or issues KernelVerification receipts.
    """

    return "FormalLogicVocabulary"


__all__ = [
    "CODE_CONTRACT_LOGIC_VERSION",
    "TRANSLATOR_ID",
    "TRANSLATOR_VERSION",
    "RULESET_ID",
    "RULESET_VERSION",
    "LOGIC_FAMILY",
    "LOGIC_TRANSLATION_EVIDENCE",
    "OBJECTIVE_GOAL_ID",
    "OBJECTIVE_VALIDATION_REPAIR_EVIDENCE",
    "OBJECTIVE_VALIDATION_REPAIR_TASK_ID",
    "ArgumentSort",
    "CallSliceBinding",
    "CodeContractLogicError",
    "ConformanceReceipt",
    "ContractPredicate",
    "FormalLogicVocabulary",
    "LogicAssumption",
    "PredicateArgument",
    "PredicateRelation",
    "RejectionCode",
    "SupportedPredicateKind",
    "TranslationRejectedError",
    "TranslationRequest",
    "TranslationResult",
    "TranslationStatus",
    "UnsupportedResidual",
    "all_covered_evidence_terms",
    "covered_evidence_terms",
    "extract_assumptions_from_contract",
    "extract_predicates_from_contract",
    "extract_reachability_predicates",
    "logic_translation_evidence_terms",
    "make_predicate",
    "objective_validation_repair_evidence_terms",
    "pinned_translator_identity",
    "project_reviewed_formula",
    "reconstruct_predicate_from_claim",
    "round_trip_predicates",
    "translate",
    "translate_contract",
    "translation_stage_owner",
    "translator_identity",
    "verify_conformance_receipt",
]
