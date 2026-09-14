"""Sound e-graph normalization and scoped relation promotion (SAWM-019).

Interface: ``ProgramWorldEGraph@1``
Evidence: ``sawm/egraph-normalization@1``

Operational integration of equality saturation for *pure* Python fragments.
Datasets remains the relation-meaning owner; current proof authority admits.
This module does not create a global optimizer, unsound rewrite registry,
second prover, or self-approving promotion path.

Normative constraints:

* Only declared, identity-bearing, sound rewrite rules may saturate.
* Similarity, embeddings, scores, and model output cannot support
  equivalence or promotion.
* Unsupported constructs, effects, concurrency, and opaque calls remain
  outside normalization (typed unsupported / abstention).
* Conflicting or inconsistent rules abstain.  Contradiction never grants
  ex-falso admission and never justifies repair.
* Saturation is bounded.  Nontermination is a typed bound exhaustion.
* Promotion is proposal-only.  Independent admission still required.
* Importing this module performs no I/O and starts no analysis.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.program_relations import (
    ContradictionDisposition,
    ProgramRelationClaim,
    ProgramRelationError,
    RelationAuthorityStatus,
    RelationKind,
    RelationScope,
    RelationValidationVerdict,
)


PROGRAM_WORLD_EGRAPH_INTERFACE: Final[str] = "ProgramWorldEGraph@1"
PROGRAM_EGRAPH_NORMALIZER_INTERFACE: Final[str] = "ProgramEGraphNormalizer@1"
SAWM_EGRAPH_NORMALIZATION_EVIDENCE: Final[str] = "sawm/egraph-normalization@1"

PROGRAM_FRAGMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-fragment@1"
)
REWRITE_RULE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/egraph-rewrite-rule@1"
)
REWRITE_THEORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/egraph-rewrite-theory@1"
)
SATURATION_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/equality-saturation-bounds@1"
)
EQUALITY_SATURATION_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/equality-saturation-plan@1"
)
RELATION_PROMOTION_PROPOSAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/relation-promotion-proposal@1"
)

PRODUCER_ID: Final[str] = "program-egraph@1"
NORMALIZER_VERSION: Final[str] = "1"
BUILTIN_THEORY_ID: Final[str] = "python-pure-int-bool@1"

MAX_TEXT_CHARS: Final[int] = 4_096
MAX_COLLECTION_ITEMS: Final[int] = 4_096
MAX_TERM_NODES: Final[int] = 512
MAX_SAFE_INTEGER: Final[int] = (1 << 53) - 1
MAX_PATTERN_VARS: Final[int] = 32
MAX_THEORY_RULES: Final[int] = 256
DEFAULT_MAX_ITERATIONS: Final[int] = 32
DEFAULT_MAX_ECLASSES: Final[int] = 2_048
DEFAULT_MAX_ENODES: Final[int] = 8_192
DEFAULT_MAX_MATCHES: Final[int] = 256
ADMITTED_LANGUAGES: Final[frozenset[str]] = frozenset({"python"})

FORBIDDEN_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "ann_score",
        "cosine",
        "distance",
        "embedding",
        "embedding_score",
        "embeddings",
        "knn",
        "model",
        "model_cid",
        "nearest",
        "rank_score",
        "score",
        "scores",
        "similarity",
        "tokenizer_cid",
        "vector",
        "vector_cid",
        "vectors",
    }
)
FORBIDDEN_SOUNDNESS: Final[frozenset[str]] = frozenset(
    {
        "heuristic",
        "learned",
        "model",
        "neural",
        "observed",
        "similarity",
        "statistical",
        "unreviewed",
    }
)
UNSOUND_EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "analogical",
        "embedding",
        "model",
        "model_output",
        "nearest_neighbor",
        "neural",
        "similarity",
        "vector",
    }
)
EFFECTFUL_NAME_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "append",
        "asyncio",
        "exec",
        "eval",
        "input",
        "multiprocessing",
        "open",
        "print",
        "subprocess",
        "threading",
        "write",
    }
)
UNSUPPORTED_CALL_FUNCS: Final[frozenset[str]] = frozenset(
    {
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "print",
        "setattr",
        "__import__",
    }
)

_BINOPS: Final[Mapping[type, str]] = MappingProxyType(
    {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "mul",
        ast.FloorDiv: "floordiv",
        ast.Mod: "mod",
        ast.Pow: "pow",
        ast.BitAnd: "bitand",
        ast.BitOr: "bitor",
        ast.BitXor: "bitxor",
        ast.LShift: "lshift",
        ast.RShift: "rshift",
    }
)
_UNARYOPS: Final[Mapping[type, str]] = MappingProxyType(
    {
        ast.Not: "not",
        ast.USub: "neg",
        ast.UAdd: "pos",
        ast.Invert: "invert",
    }
)
_CMPOPS: Final[Mapping[type, str]] = MappingProxyType(
    {
        ast.Eq: "eq",
        ast.NotEq: "not_eq",
        ast.Lt: "lt",
        ast.LtE: "le",
        ast.Gt: "gt",
        ast.GtE: "ge",
    }
)
_INFIX: Final[Mapping[str, str]] = MappingProxyType(
    {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "floordiv": "//",
        "mod": "%",
        "pow": "**",
        "bitand": "&",
        "bitor": "|",
        "bitxor": "^",
        "lshift": "<<",
        "rshift": ">>",
        "eq": "==",
        "not_eq": "!=",
        "lt": "<",
        "le": "<=",
        "gt": ">",
        "ge": ">=",
        "and": "and",
        "or": "or",
    }
)
_FOLDABLE_INT: Final[frozenset[str]] = frozenset(
    {"add", "sub", "mul", "floordiv", "mod", "pow", "bitand", "bitor", "bitxor"}
)


class ProgramEGraphError(ValueError):
    """Frozen e-graph inputs cannot produce a trustworthy plan."""


class ProgramEGraphAuthorityError(ProgramEGraphError):
    """Similarity, model output, or self-admission attempted to mint authority."""


class ProgramEGraphBoundsError(ProgramEGraphError):
    """A saturation or payload bound was exceeded."""


class ProgramEGraphStaleError(ProgramEGraphAuthorityError):
    """Theory or environment bindings are not current."""


class SaturationDisposition(str, Enum):
    """Closed equality-saturation outcomes."""

    NORMALIZED = "normalized"
    EQUIVALENT = "equivalent"
    SATURATED = "saturated"
    INEQUIVALENT_UNPROVED = "inequivalent_unproved"
    UNSUPPORTED = "unsupported"
    BOUND_EXHAUSTED = "bound_exhausted"
    CONFLICT = "conflict"
    ABSTAINED = "abstained"
    STALE = "stale"
    REJECTED = "rejected"


class PromotionDisposition(str, Enum):
    """Closed relation-promotion outcomes.  Never self-admits."""

    PROPOSED = "proposed"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    CONFLICT = "conflict"
    STALE = "stale"
    REFUTED = "refuted"


class RewriteSoundness(str, Enum):
    """Closed soundness vocabulary for declared rewrite rules."""

    ALGEBRAIC_IDENTITY = "algebraic_identity"
    DECLARED_AXIOM = "declared_axiom"
    PROVED = "proved"


class FragmentPurity(str, Enum):
    """Closed purity classification of one program fragment."""

    PURE = "pure"
    EFFECTFUL = "effectful"
    CONCURRENT = "concurrent"
    OPAQUE = "opaque"
    UNSUPPORTED = "unsupported"


class EvidenceBasis(str, Enum):
    """Closed bases that may support a promotion proposal."""

    REFLEXIVITY = "reflexivity"
    NORMALIZATION = "normalization"
    KERNEL_PROOF = "kernel_proof"
    NONE = "none"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise ProgramEGraphBoundsError("e-graph payload exceeds depth bound")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, int) and not isinstance(value, bool):
            if value < -MAX_SAFE_INTEGER or value > MAX_SAFE_INTEGER:
                raise ProgramEGraphError("integer is outside the safe JSON range")
        return value
    if isinstance(value, float):
        raise ProgramEGraphError("e-graph payloads reject floating-point values")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        forbidden = set(value) & FORBIDDEN_FIELD_MARKERS
        if forbidden:
            raise ProgramEGraphAuthorityError(
                "e-graph payloads reject non-semantic fields "
                + ", ".join(sorted(forbidden))
            )
        return {
            str(key): _plain(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ProgramEGraphBoundsError("e-graph payload exceeds collection bound")
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise ProgramEGraphError(
        f"e-graph payload contains unsupported type {type(value).__name__}"
    )


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProgramEGraphError(f"{name} must be a string")
    text = value.strip()
    if not empty and not text:
        raise ProgramEGraphError(f"{name} must be a nonempty string")
    if len(text) > MAX_TEXT_CHARS:
        raise ProgramEGraphBoundsError(f"{name} exceeds text bound")
    return text


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProgramEGraphError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise ProgramEGraphError(f"{name} has unsupported value {value!r}") from exc


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise ProgramEGraphError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        raise ProgramEGraphError(f"{name} must be a positive integer")
    if value > maximum:
        raise ProgramEGraphBoundsError(f"{name} exceeds bound {maximum}")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ProgramEGraphError(f"{name} must be a nonnegative integer")
    return value


def _sorted_unique(values: Iterable[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ProgramEGraphError(f"{name} must be a sequence")
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        text = _text(item, name)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) > MAX_COLLECTION_ITEMS:
            raise ProgramEGraphBoundsError(f"{name} exceeds collection bound")
    return tuple(sorted(result))


def _reason_codes(*groups: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                ordered.append(text)
    return tuple(ordered)


def _identity_cid(payload: Mapping[str, Any]) -> str:
    return cid_for_structured(_plain(payload))


def _safe_int(value: int) -> int | None:
    if value < -MAX_SAFE_INTEGER or value > MAX_SAFE_INTEGER:
        return None
    return value


def _const_payload(value: Any) -> str:
    if value is None:
        return "n"
    if type(value) is bool:
        return "b:true" if value else "b:false"
    if type(value) is int and not isinstance(value, bool):
        if _safe_int(value) is None:
            raise ProgramEGraphError("integer is outside the safe JSON range")
        return f"i:{value}"
    if type(value) is str:
        if len(value) > MAX_TEXT_CHARS:
            raise ProgramEGraphBoundsError("string constant exceeds text bound")
        return "s:" + value
    raise ProgramEGraphError(f"unsupported constant {type(value).__name__}")


def _payload_const(payload: str) -> Any:
    if payload == "n":
        return None
    if payload == "b:true":
        return True
    if payload == "b:false":
        return False
    if payload.startswith("i:"):
        return int(payload[2:])
    if payload.startswith("s:"):
        return payload[2:]
    raise ProgramEGraphError(f"malformed constant payload {payload!r}")


# ---------------------------------------------------------------------------
# Terms and rewrite rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Term:
    """Closed term / pattern node for a pure fragment."""

    op: str
    children: tuple["Term", ...] = ()
    payload: str = ""

    def canonical(self) -> list[Any]:
        return [self.op, self.payload, [child.canonical() for child in self.children]]

    def node_count(self) -> int:
        return 1 + sum(child.node_count() for child in self.children)

    def pattern_vars(self) -> tuple[str, ...]:
        if self.op == "pat":
            return (self.payload,)
        ordered: list[str] = []
        seen: set[str] = set()
        for child in self.children:
            for name in child.pattern_vars():
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)
        return tuple(ordered)

    def alpha_canonical(self) -> "Term":
        mapping: dict[str, str] = {}

        def walk(node: Term) -> Term:
            if node.op == "pat":
                renamed = mapping.setdefault(node.payload, f"${len(mapping)}")
                return Term("pat", (), renamed)
            return Term(node.op, tuple(walk(child) for child in node.children), node.payload)

        return walk(self)

    def render(self) -> str:
        if self.op == "const":
            value = _payload_const(self.payload)
            if type(value) is str:
                return repr(value)
            return str(value)
        if self.op == "var":
            return self.payload
        if self.op == "pat":
            return self.payload
        if self.op == "not" and len(self.children) == 1:
            return f"(not {self.children[0].render()})"
        if self.op == "neg" and len(self.children) == 1:
            return f"(-{self.children[0].render()})"
        if self.op == "pos" and len(self.children) == 1:
            return f"(+{self.children[0].render()})"
        if self.op == "invert" and len(self.children) == 1:
            return f"(~{self.children[0].render()})"
        if self.op == "ite" and len(self.children) == 3:
            cond, then, orelse = self.children
            return f"({then.render()} if {cond.render()} else {orelse.render()})"
        if self.op == "tuple":
            body = ", ".join(child.render() for child in self.children)
            if len(self.children) == 1:
                body += ","
            return f"({body})"
        if self.op == "list":
            body = ", ".join(child.render() for child in self.children)
            return f"[{body}]"
        if self.op in _INFIX and len(self.children) == 2:
            left, right = self.children
            return f"({left.render()} {_INFIX[self.op]} {right.render()})"
        args = ", ".join(child.render() for child in self.children)
        return f"{self.op}({args})"

    @classmethod
    def const(cls, value: Any) -> "Term":
        return cls("const", (), _const_payload(value))

    @classmethod
    def var(cls, name: str) -> "Term":
        return cls("var", (), _text(name, "variable"))

    @classmethod
    def pat(cls, name: str) -> "Term":
        text = _text(name, "pattern variable")
        if not text.startswith("$"):
            text = f"${text}"
        return cls("pat", (), text)

    @classmethod
    def app(cls, op: str, *children: "Term") -> "Term":
        return cls(_text(op, "operator"), children, "")

    @classmethod
    def from_canonical(cls, value: Any) -> "Term":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ProgramEGraphError("term encoding must be a sequence")
        if len(value) != 3:
            raise ProgramEGraphError("term encoding must be [op, payload, children]")
        op = _text(value[0], "term.op")
        payload = value[1]
        if payload is None:
            payload = ""
        payload_text = _text(payload, "term.payload", empty=True)
        children_raw = value[2]
        if not isinstance(children_raw, Sequence) or isinstance(children_raw, (str, bytes)):
            raise ProgramEGraphError("term children must be a sequence")
        children = tuple(cls.from_canonical(item) for item in children_raw)
        return cls(op, children, payload_text)


def _parse_term_spec(value: Any, name: str) -> Term:
    if isinstance(value, Term):
        return value
    if isinstance(value, Mapping):
        if "canonical" in value:
            return Term.from_canonical(value["canonical"])
        op = _text(value.get("op"), f"{name}.op")
        payload = _text(value.get("payload") or "", f"{name}.payload", empty=True)
        children_raw = value.get("children") or ()
        children = tuple(_parse_term_spec(item, name) for item in children_raw)
        return Term(op, children, payload)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return Term.from_canonical(value)
    if isinstance(value, str):
        parsed = _parse_fragment_source(value, declared_pure=())
        if parsed.purity is not FragmentPurity.PURE or parsed.term is None:
            raise ProgramEGraphError(f"{name} is not a pure term")
        return parsed.term
    raise ProgramEGraphError(f"{name} must be a term encoding")


@dataclass(frozen=True, slots=True)
class RewriteRule:
    """Identity-bearing rewrite whose soundness is declared, not inferred."""

    rule_id: str
    lhs: Term
    rhs: Term
    soundness: RewriteSoundness | str
    theory_id: str
    oriented: bool = True
    review_ref: str = ""
    proof_receipt_cid: str | None = None

    SCHEMA: ClassVar[str] = REWRITE_RULE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _text(self.rule_id, "rule_id"))
        if not isinstance(self.lhs, Term) or not isinstance(self.rhs, Term):
            raise ProgramEGraphError("rewrite rule lhs/rhs must be Term values")
        raw_soundness = self.soundness
        if isinstance(raw_soundness, str):
            marker = raw_soundness.strip().lower().replace("-", "_").replace(" ", "_")
            if marker in FORBIDDEN_SOUNDNESS:
                raise ProgramEGraphAuthorityError(
                    "similarity/model/heuristic rules cannot be admitted as sound"
                )
        elif isinstance(raw_soundness, Enum):
            if raw_soundness.value in FORBIDDEN_SOUNDNESS:
                raise ProgramEGraphAuthorityError(
                    "similarity/model/heuristic rules cannot be admitted as sound"
                )
        soundness = _enum(raw_soundness, RewriteSoundness, "soundness")
        object.__setattr__(self, "soundness", soundness)
        object.__setattr__(self, "theory_id", _text(self.theory_id, "theory_id"))
        object.__setattr__(self, "oriented", _bool(self.oriented, "oriented"))
        object.__setattr__(
            self, "review_ref", _text(self.review_ref, "review_ref", empty=True)
        )
        object.__setattr__(
            self,
            "proof_receipt_cid",
            _optional_cid(self.proof_receipt_cid, "proof_receipt_cid"),
        )
        if self.lhs.node_count() > MAX_TERM_NODES or self.rhs.node_count() > MAX_TERM_NODES:
            raise ProgramEGraphBoundsError("rewrite rule exceeds term-node bound")
        if len(self.lhs.pattern_vars()) > MAX_PATTERN_VARS:
            raise ProgramEGraphBoundsError("rewrite rule exceeds pattern-variable bound")
        rhs_vars = set(self.rhs.pattern_vars())
        lhs_vars = set(self.lhs.pattern_vars())
        extra = rhs_vars - lhs_vars
        if extra:
            raise ProgramEGraphError(
                f"rule {self.rule_id} rhs introduces unbound pattern variables"
            )
        if soundness == RewriteSoundness.PROVED.value and self.proof_receipt_cid is None:
            raise ProgramEGraphError("proved rules require proof_receipt_cid")
        if (
            soundness == RewriteSoundness.DECLARED_AXIOM.value
            and not self.review_ref
        ):
            raise ProgramEGraphError("declared axioms require review_ref")
        if (
            soundness == RewriteSoundness.ALGEBRAIC_IDENTITY.value
            and not self.review_ref
        ):
            object.__setattr__(self, "review_ref", SAWM_EGRAPH_NORMALIZATION_EVIDENCE)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "rule_id": self.rule_id,
            "lhs": self.lhs.canonical(),
            "rhs": self.rhs.canonical(),
            "soundness": self.soundness,
            "theory_id": self.theory_id,
            "oriented": self.oriented,
            "review_ref": self.review_ref,
            "proof_receipt_cid": self.proof_receipt_cid,
        }

    @property
    def rule_cid(self) -> str:
        return _identity_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["rule_cid"] = self.rule_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RewriteRule":
        if not isinstance(data, Mapping):
            raise ProgramEGraphError("rewrite rule must be a mapping")
        return cls(
            rule_id=str(data.get("rule_id") or ""),
            lhs=_parse_term_spec(data.get("lhs"), "lhs"),
            rhs=_parse_term_spec(data.get("rhs"), "rhs"),
            soundness=str(data.get("soundness") or ""),
            theory_id=str(data.get("theory_id") or ""),
            oriented=bool(data.get("oriented", True)),
            review_ref=str(data.get("review_ref") or ""),
            proof_receipt_cid=data.get("proof_receipt_cid"),
        )


@dataclass(frozen=True, slots=True)
class RewriteTheory:
    """Closed, identity-bearing theory.  Not a global unsound registry."""

    theory_id: str
    rules: tuple[RewriteRule, ...]
    review_refs: tuple[str, ...] = ()
    language: str = "python"

    SCHEMA: ClassVar[str] = REWRITE_THEORY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "theory_id", _text(self.theory_id, "theory_id"))
        language = _text(self.language, "language")
        object.__setattr__(self, "language", language)
        object.__setattr__(
            self, "review_refs", _sorted_unique(self.review_refs, "review_ref")
        )
        if not self.rules:
            raise ProgramEGraphError("rewrite theory requires at least one rule")
        if len(self.rules) > MAX_THEORY_RULES:
            raise ProgramEGraphBoundsError("rewrite theory exceeds rule bound")
        normalized: list[RewriteRule] = []
        seen_ids: set[str] = set()
        for item in self.rules:
            rule = item if isinstance(item, RewriteRule) else RewriteRule.from_dict(item)
            if rule.theory_id != self.theory_id:
                raise ProgramEGraphError("rule theory_id must match the declared theory")
            if rule.rule_id in seen_ids:
                raise ProgramEGraphError(f"duplicate rule_id {rule.rule_id}")
            seen_ids.add(rule.rule_id)
            normalized.append(rule)
        object.__setattr__(self, "rules", tuple(normalized))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "theory_id": self.theory_id,
            "language": self.language,
            "review_refs": list(self.review_refs),
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @property
    def theory_cid(self) -> str:
        return _identity_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["theory_cid"] = self.theory_cid
        return payload

    def oriented_conflicts(self) -> tuple[tuple[str, str], ...]:
        grouped: dict[str, list[RewriteRule]] = {}
        for rule in self.rules:
            if not rule.oriented:
                continue
            key = _identity_cid(
                {
                    "lhs": rule.lhs.alpha_canonical().canonical(),
                    "oriented": True,
                }
            )
            grouped.setdefault(key, []).append(rule)
        conflicts: list[tuple[str, str]] = []
        for group in grouped.values():
            rhs_keys = {
                _identity_cid({"rhs": rule.rhs.alpha_canonical().canonical()})
                for rule in group
            }
            if len(rhs_keys) > 1:
                for index, left in enumerate(group):
                    for right in group[index + 1 :]:
                        conflicts.append((left.rule_id, right.rule_id))
        return tuple(conflicts)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RewriteTheory":
        if not isinstance(data, Mapping):
            raise ProgramEGraphError("rewrite theory must be a mapping")
        rules_raw = data.get("rules") or ()
        rules = tuple(
            item if isinstance(item, RewriteRule) else RewriteRule.from_dict(item)
            for item in rules_raw
        )
        return cls(
            theory_id=str(data.get("theory_id") or ""),
            rules=rules,
            review_refs=tuple(data.get("review_refs") or ()),
            language=str(data.get("language") or "python"),
        )


@dataclass(frozen=True, slots=True)
class SaturationBounds:
    """Finite equality-saturation resource bounds."""

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_eclasses: int = DEFAULT_MAX_ECLASSES
    max_enodes: int = DEFAULT_MAX_ENODES
    max_matches_per_rule: int = DEFAULT_MAX_MATCHES

    SCHEMA: ClassVar[str] = SATURATION_BOUNDS_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_iterations",
            _positive_int(self.max_iterations, "max_iterations", maximum=10_000),
        )
        object.__setattr__(
            self,
            "max_eclasses",
            _positive_int(self.max_eclasses, "max_eclasses", maximum=100_000),
        )
        object.__setattr__(
            self,
            "max_enodes",
            _positive_int(self.max_enodes, "max_enodes", maximum=200_000),
        )
        object.__setattr__(
            self,
            "max_matches_per_rule",
            _positive_int(
                self.max_matches_per_rule, "max_matches_per_rule", maximum=10_000
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "max_iterations": self.max_iterations,
            "max_eclasses": self.max_eclasses,
            "max_enodes": self.max_enodes,
            "max_matches_per_rule": self.max_matches_per_rule,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "SaturationBounds":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ProgramEGraphError("saturation bounds must be a mapping")
        return cls(
            max_iterations=int(value.get("max_iterations") or DEFAULT_MAX_ITERATIONS),
            max_eclasses=int(value.get("max_eclasses") or DEFAULT_MAX_ECLASSES),
            max_enodes=int(value.get("max_enodes") or DEFAULT_MAX_ENODES),
            max_matches_per_rule=int(
                value.get("max_matches_per_rule") or DEFAULT_MAX_MATCHES
            ),
        )


@dataclass(frozen=True, slots=True)
class ParsedFragment:
    source: str
    language: str
    term: Term | None
    purity: FragmentPurity
    reason_codes: tuple[str, ...]
    fragment_cid: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_FRAGMENT_SCHEMA,
            "language": self.language,
            "source": self.source,
            "term": None if self.term is None else self.term.canonical(),
            "purity": self.purity.value,
        }


@dataclass(frozen=True, slots=True)
class _ENode:
    op: str
    children: tuple[int, ...]
    payload: str = ""

    def key(self, find: Any) -> tuple[Any, ...]:
        return (self.op, self.payload, tuple(find(child) for child in self.children))


class _EGraph:
    """Congruence-closed e-graph with bounded e-matching."""

    def __init__(self, bounds: SaturationBounds) -> None:
        self.bounds = bounds
        self.parent: list[int] = []
        self.rank: list[int] = []
        self.nodes_of: dict[int, list[_ENode]] = {}
        self.hashcons: dict[tuple[Any, ...], int] = {}
        self.next_id = 0
        self.enode_count = 0

    def find(self, class_id: int) -> int:
        parent = self.parent[class_id]
        if parent != class_id:
            root = self.find(parent)
            self.parent[class_id] = root
            return root
        return class_id

    def _new_class(self) -> int:
        if self.next_id >= self.bounds.max_eclasses:
            raise ProgramEGraphBoundsError("e-class budget exhausted")
        class_id = self.next_id
        self.next_id += 1
        self.parent.append(class_id)
        self.rank.append(0)
        self.nodes_of[class_id] = []
        return class_id

    def add_node(self, node: _ENode) -> int:
        key = node.key(self.find)
        existing = self.hashcons.get(key)
        if existing is not None:
            return self.find(existing)
        if self.enode_count >= self.bounds.max_enodes:
            raise ProgramEGraphBoundsError("e-node budget exhausted")
        class_id = self._new_class()
        self.nodes_of[class_id].append(node)
        self.hashcons[key] = class_id
        self.enode_count += 1
        return class_id

    def add_term(self, term: Term) -> int:
        if term.op == "pat":
            raise ProgramEGraphError("cannot insert a pattern variable as a term")
        children = tuple(self.add_term(child) for child in term.children)
        return self.add_node(_ENode(term.op, children, term.payload))

    def union(self, left: int, right: int) -> bool:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        self.nodes_of[a].extend(self.nodes_of.pop(b, []))
        return True

    def restore_congruence(self) -> None:
        changed = True
        while changed:
            changed = False
            rebuilt: dict[tuple[Any, ...], int] = {}
            roots = {self.find(class_id) for class_id in range(self.next_id)}
            packed: dict[int, list[_ENode]] = {root: [] for root in roots}
            for root in roots:
                for node in self.nodes_of.get(root, ()):
                    canon = _ENode(
                        node.op,
                        tuple(self.find(child) for child in node.children),
                        node.payload,
                    )
                    packed[self.find(root)].append(canon)
                    key = canon.key(self.find)
                    prior = rebuilt.get(key)
                    if prior is None:
                        rebuilt[key] = self.find(root)
                    elif self.union(prior, root):
                        changed = True
            live: dict[int, list[_ENode]] = {}
            for class_id, nodes in packed.items():
                live.setdefault(self.find(class_id), []).extend(nodes)
            self.nodes_of = live
            self.hashcons = {
                node.key(self.find): root
                for root, nodes in live.items()
                for node in nodes
            }

    def class_has_const(self, class_id: int, value: Any) -> bool:
        payload = _const_payload(value)
        root = self.find(class_id)
        return any(
            node.op == "const" and node.payload == payload
            for node in self.nodes_of.get(root, ())
        )

    def constants_in(self, class_id: int) -> tuple[Any, ...]:
        root = self.find(class_id)
        values: list[Any] = []
        for node in self.nodes_of.get(root, ()):
            if node.op == "const":
                values.append(_payload_const(node.payload))
        return tuple(values)

    def extract(self, class_id: int) -> Term:
        root = self.find(class_id)
        memo: dict[int, Term] = {}
        visiting: set[int] = set()

        def walk(target: int) -> Term:
            target = self.find(target)
            cached = memo.get(target)
            if cached is not None:
                return cached
            visiting.add(target)
            best: Term | None = None
            best_key: tuple[Any, ...] | None = None
            for node in self.nodes_of.get(target, ()):
                if node.op in {"const", "var"}:
                    candidate = Term(node.op, (), node.payload)
                else:
                    child_ids = tuple(self.find(child) for child in node.children)
                    if any(child_id == target or child_id in visiting for child_id in child_ids):
                        continue
                    children = tuple(walk(child_id) for child_id in child_ids)
                    candidate = Term(node.op, children, node.payload)
                key = (candidate.node_count(), candidate.render())
                if best is None or key < best_key:  # type: ignore[operator]
                    best = candidate
                    best_key = key
            visiting.remove(target)
            if best is None:
                raise ProgramEGraphError("e-class has no extractable node")
            memo[target] = best
            return best

        return walk(root)


def _match_in_class(
    pattern: Term, class_id: int, graph: _EGraph, subst: dict[str, int]
) -> list[dict[str, int]]:
    class_id = graph.find(class_id)
    if pattern.op == "pat":
        bound = subst.get(pattern.payload)
        if bound is None:
            next_subst = dict(subst)
            next_subst[pattern.payload] = class_id
            return [next_subst]
        return [subst] if graph.find(bound) == class_id else []
    matches: list[dict[str, int]] = []
    for node in graph.nodes_of.get(class_id, ()):
        if node.op != pattern.op or node.payload != pattern.payload:
            continue
        if len(node.children) != len(pattern.children):
            continue
        current = [dict(subst)]
        failed = False
        for child_pattern, child_class in zip(pattern.children, node.children):
            next_round: list[dict[str, int]] = []
            for item in current:
                next_round.extend(
                    _match_in_class(child_pattern, child_class, graph, item)
                )
            if not next_round:
                failed = True
                break
            current = next_round
        if not failed:
            matches.extend(current)
    return matches


def _instantiate(term: Term, subst: Mapping[str, int], graph: _EGraph) -> int:
    if term.op == "pat":
        return graph.find(subst[term.payload])
    if term.op in {"const", "var"}:
        return graph.add_node(_ENode(term.op, (), term.payload))
    children = tuple(_instantiate(child, subst, graph) for child in term.children)
    return graph.add_node(_ENode(term.op, children, term.payload))


def _fold_constants(graph: _EGraph) -> bool:
    changed = False
    roots = {graph.find(class_id) for class_id in range(graph.next_id)}
    for root in roots:
        for node in list(graph.nodes_of.get(graph.find(root), ())):
            if node.op in {"const", "var"} or not node.children:
                continue
            child_consts: list[Any] = []
            complete = True
            for child in node.children:
                constants = graph.constants_in(child)
                if len(constants) != 1:
                    complete = False
                    break
                child_consts.append(constants[0])
            if not complete:
                continue
            folded = _eval_op(node.op, child_consts)
            if folded is _UNFOLDABLE:
                continue
            const_id = graph.add_node(_ENode("const", (), _const_payload(folded)))
            if graph.union(graph.find(root), const_id):
                changed = True
    if changed:
        graph.restore_congruence()
    return changed


_UNFOLDABLE: Final[object] = object()


def _eval_op(op: str, args: Sequence[Any]) -> Any:
    try:
        if op == "pos" and len(args) == 1 and type(args[0]) is int:
            return args[0]
        if op == "neg" and len(args) == 1 and type(args[0]) is int:
            return _safe_int(-args[0]) if _safe_int(-args[0]) is not None else _UNFOLDABLE
        if op == "not" and len(args) == 1 and type(args[0]) is bool:
            return not args[0]
        if op == "and" and len(args) == 2:
            return args[0] and args[1]
        if op == "or" and len(args) == 2:
            return args[0] or args[1]
        if op == "eq" and len(args) == 2:
            return args[0] == args[1]
        if op == "not_eq" and len(args) == 2:
            return args[0] != args[1]
        if (
            op in {"lt", "le", "gt", "ge"}
            and len(args) == 2
            and type(args[0]) is int
            and type(args[1]) is int
        ):
            left, right = args
            if op == "lt":
                return left < right
            if op == "le":
                return left <= right
            if op == "gt":
                return left > right
            return left >= right
        if op in _FOLDABLE_INT and len(args) == 2 and all(type(item) is int for item in args):
            left, right = args
            if op == "add":
                result = left + right
            elif op == "sub":
                result = left - right
            elif op == "mul":
                result = left * right
            elif op == "floordiv":
                if right == 0:
                    return _UNFOLDABLE
                result = left // right
            elif op == "mod":
                if right == 0:
                    return _UNFOLDABLE
                result = left % right
            elif op == "pow":
                if right < 0 or right > 64:
                    return _UNFOLDABLE
                result = left ** right
            elif op == "bitand":
                result = left & right
            elif op == "bitor":
                result = left | right
            else:
                result = left ^ right
            folded = _safe_int(result)
            return _UNFOLDABLE if folded is None else folded
        if op == "ite" and len(args) == 3 and type(args[0]) is bool:
            return args[1] if args[0] else args[2]
    except (OverflowError, ValueError, ZeroDivisionError, TypeError):
        return _UNFOLDABLE
    return _UNFOLDABLE


def _inconsistent(graph: _EGraph) -> bool:
    roots = {graph.find(class_id) for class_id in range(graph.next_id)}
    for root in roots:
        consts = graph.constants_in(root)
        distinct = {repr(item) for item in consts}
        if len(distinct) > 1:
            return True
        if True in consts and False in consts:
            return True
    true_classes = [
        root
        for root in roots
        if graph.class_has_const(root, True)
    ]
    false_classes = [
        root
        for root in roots
        if graph.class_has_const(root, False)
    ]
    return bool(set(true_classes) & set(false_classes))


# ---------------------------------------------------------------------------
# Fragment parsing
# ---------------------------------------------------------------------------


@dataclass
class _ParseState:
    declared_pure: frozenset[str]
    reasons: list[str] = field(default_factory=list)
    purity: FragmentPurity = FragmentPurity.PURE


def _mark(state: _ParseState, purity: FragmentPurity, reason: str) -> None:
    if reason not in state.reasons:
        state.reasons.append(reason)
    rank = {
        FragmentPurity.PURE: 0,
        FragmentPurity.OPAQUE: 1,
        FragmentPurity.EFFECTFUL: 2,
        FragmentPurity.CONCURRENT: 3,
        FragmentPurity.UNSUPPORTED: 4,
    }
    if rank[purity] > rank[state.purity]:
        state.purity = purity


def _convert_ast(node: ast.AST, state: _ParseState) -> Term | None:
    if isinstance(node, ast.Expression):
        return _convert_ast(node.body, state)
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, float) or type(value) not in {int, bool, str, type(None)}:
            _mark(state, FragmentPurity.UNSUPPORTED, "unsupported_constant")
            return None
        if type(value) is int and _safe_int(value) is None:
            _mark(state, FragmentPurity.UNSUPPORTED, "integer_out_of_range")
            return None
        return Term.const(value)
    if isinstance(node, ast.Name):
        if node.id in EFFECTFUL_NAME_MARKERS:
            purity = (
                FragmentPurity.CONCURRENT
                if node.id in {"asyncio", "threading", "multiprocessing"}
                else FragmentPurity.EFFECTFUL
            )
            _mark(state, purity, f"effectful_name:{node.id}")
            return None
        return Term.var(node.id)
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            _mark(state, FragmentPurity.UNSUPPORTED, "unsupported_unary")
            return None
        child = _convert_ast(node.operand, state)
        return None if child is None else Term.app(op, child)
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            _mark(state, FragmentPurity.UNSUPPORTED, "unsupported_binop")
            return None
        left = _convert_ast(node.left, state)
        right = _convert_ast(node.right, state)
        if left is None or right is None:
            return None
        return Term.app(op, left, right)
    if isinstance(node, ast.BoolOp):
        op = "and" if isinstance(node.op, ast.And) else "or"
        if len(node.values) < 2:
            _mark(state, FragmentPurity.UNSUPPORTED, "malformed_boolop")
            return None
        converted = [_convert_ast(item, state) for item in node.values]
        if any(item is None for item in converted):
            return None
        acc = converted[0]
        for item in converted[1:]:
            acc = Term.app(op, acc, item)  # type: ignore[arg-type]
        return acc
    if isinstance(node, ast.Compare):
        if any(type(op) not in _CMPOPS for op in node.ops):
            _mark(state, FragmentPurity.UNSUPPORTED, "unsupported_compare")
            return None
        comparators = [node.left, *node.comparators]
        converted = [_convert_ast(item, state) for item in comparators]
        if any(item is None for item in converted):
            return None
        parts: list[Term] = []
        for index, op in enumerate(node.ops):
            parts.append(
                Term.app(_CMPOPS[type(op)], converted[index], converted[index + 1])  # type: ignore[arg-type]
            )
        acc = parts[0]
        for item in parts[1:]:
            acc = Term.app("and", acc, item)
        return acc
    if isinstance(node, ast.IfExp):
        cond = _convert_ast(node.test, state)
        then = _convert_ast(node.body, state)
        orelse = _convert_ast(node.orelse, state)
        if cond is None or then is None or orelse is None:
            return None
        return Term.app("ite", cond, then, orelse)
    if isinstance(node, ast.Tuple):
        if not isinstance(node.ctx, ast.Load):
            _mark(state, FragmentPurity.EFFECTFUL, "tuple_store")
            return None
        children = [_convert_ast(item, state) for item in node.elts]
        if any(item is None for item in children):
            return None
        return Term("tuple", tuple(children), "")  # type: ignore[arg-type]
    if isinstance(node, ast.List):
        if not isinstance(node.ctx, ast.Load):
            _mark(state, FragmentPurity.EFFECTFUL, "list_store")
            return None
        children = [_convert_ast(item, state) for item in node.elts]
        if any(item is None for item in children):
            return None
        return Term("list", tuple(children), "")  # type: ignore[arg-type]
    if isinstance(node, ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            _mark(state, FragmentPurity.OPAQUE, "opaque_attribute_call")
            return None
        else:
            _mark(state, FragmentPurity.UNSUPPORTED, "unsupported_call")
            return None
        if func_name in UNSUPPORTED_CALL_FUNCS or func_name in EFFECTFUL_NAME_MARKERS:
            purity = (
                FragmentPurity.CONCURRENT
                if func_name in {"asyncio", "threading"}
                else FragmentPurity.EFFECTFUL
            )
            _mark(state, purity, f"effectful_call:{func_name}")
            return None
        if func_name not in state.declared_pure:
            _mark(state, FragmentPurity.OPAQUE, f"opaque_call:{func_name}")
            return None
        if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
            _mark(state, FragmentPurity.UNSUPPORTED, "unsupported_call_shape")
            return None
        children = [_convert_ast(arg, state) for arg in node.args]
        if any(item is None for item in children):
            return None
        return Term(f"call:{func_name}", tuple(children), "")  # type: ignore[arg-type]
    if isinstance(node, (ast.Await, ast.Yield, ast.YieldFrom)):
        _mark(state, FragmentPurity.CONCURRENT, "concurrency_construct")
        return None
    if isinstance(node, (ast.Attribute, ast.Subscript, ast.NamedExpr, ast.Starred)):
        reason = {
            ast.Attribute: "attribute_access",
            ast.Subscript: "subscript",
            ast.NamedExpr: "assignment_expression",
            ast.Starred: "starred",
        }[type(node)]
        purity = (
            FragmentPurity.EFFECTFUL
            if type(node) is ast.NamedExpr
            else FragmentPurity.UNSUPPORTED
        )
        _mark(state, purity, reason)
        return None
    if isinstance(
        node,
        (
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Dict,
            ast.Set,
            ast.Lambda,
            ast.JoinedStr,
            ast.FormattedValue,
        ),
    ):
        _mark(state, FragmentPurity.UNSUPPORTED, f"unsupported_construct:{type(node).__name__}")
        return None
    _mark(state, FragmentPurity.UNSUPPORTED, f"unsupported_ast:{type(node).__name__}")
    return None


def _parse_fragment_source(
    source: str, *, declared_pure: Sequence[str]
) -> ParsedFragment:
    text = _text(source, "fragment")
    state = _ParseState(declared_pure=frozenset(declared_pure))
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        payload = {
            "schema": PROGRAM_FRAGMENT_SCHEMA,
            "language": "python",
            "source": text,
            "term": None,
            "purity": FragmentPurity.UNSUPPORTED.value,
        }
        return ParsedFragment(
            source=text,
            language="python",
            term=None,
            purity=FragmentPurity.UNSUPPORTED,
            reason_codes=("unparseable_fragment",),
            fragment_cid=_identity_cid(payload),
        )
    term = _convert_ast(tree, state)
    if term is not None and term.node_count() > MAX_TERM_NODES:
        _mark(state, FragmentPurity.UNSUPPORTED, "fragment_too_large")
        term = None
    if state.purity is not FragmentPurity.PURE:
        term = None
    payload = {
        "schema": PROGRAM_FRAGMENT_SCHEMA,
        "language": "python",
        "source": text,
        "term": None if term is None else term.canonical(),
        "purity": state.purity.value,
    }
    return ParsedFragment(
        source=text,
        language="python",
        term=term,
        purity=state.purity,
        reason_codes=tuple(state.reasons),
        fragment_cid=_identity_cid(payload),
    )


def _coerce_fragment(
    value: Any,
    name: str,
    *,
    declared_pure: Sequence[str],
    language: str,
) -> ParsedFragment:
    if language not in ADMITTED_LANGUAGES:
        source = _text(value if isinstance(value, str) else str(value), name, empty=True)
        payload = {
            "schema": PROGRAM_FRAGMENT_SCHEMA,
            "language": language,
            "source": source,
            "term": None,
            "purity": FragmentPurity.UNSUPPORTED.value,
        }
        return ParsedFragment(
            source=source or language,
            language=language,
            term=None,
            purity=FragmentPurity.UNSUPPORTED,
            reason_codes=(f"language_unavailable:{language}",),
            fragment_cid=_identity_cid(payload),
        )
    if isinstance(value, ParsedFragment):
        return value
    if isinstance(value, Mapping):
        source = _text(value.get("source") or value.get("fragment") or "", name)
        lang = _text(value.get("language") or language, "language")
        return _coerce_fragment(
            source, name, declared_pure=declared_pure, language=lang
        )
    return _parse_fragment_source(_text(value, name), declared_pure=declared_pure)


def _pat(name: str) -> Term:
    return Term.pat(name)


def _const(value: Any) -> Term:
    return Term.const(value)


def _builtin_rule(
    rule_id: str,
    lhs: Term,
    rhs: Term,
    *,
    oriented: bool = True,
) -> RewriteRule:
    return RewriteRule(
        rule_id=rule_id,
        lhs=lhs,
        rhs=rhs,
        soundness=RewriteSoundness.ALGEBRAIC_IDENTITY,
        theory_id=BUILTIN_THEORY_ID,
        oriented=oriented,
        review_ref=SAWM_EGRAPH_NORMALIZATION_EVIDENCE,
    )


def builtin_python_pure_theory() -> RewriteTheory:
    """Sound algebraic identities for pure integer/boolean fragments."""

    x, y, z = _pat("x"), _pat("y"), _pat("z")
    zero, one = _const(0), _const(1)
    true, false = _const(True), _const(False)
    rules = (
        _builtin_rule("add-right-identity", Term.app("add", x, zero), x),
        _builtin_rule("add-left-identity", Term.app("add", zero, x), x),
        _builtin_rule("mul-right-identity", Term.app("mul", x, one), x),
        _builtin_rule("mul-left-identity", Term.app("mul", one, x), x),
        _builtin_rule("mul-right-zero", Term.app("mul", x, zero), zero),
        _builtin_rule("mul-left-zero", Term.app("mul", zero, x), zero),
        _builtin_rule("sub-right-identity", Term.app("sub", x, zero), x),
        _builtin_rule("sub-self", Term.app("sub", x, x), zero),
        _builtin_rule("pos-identity", Term.app("pos", x), x),
        _builtin_rule("double-neg-int", Term.app("neg", Term.app("neg", x)), x),
        _builtin_rule("not-not", Term.app("not", Term.app("not", x)), x),
        _builtin_rule("not-true", Term.app("not", true), false),
        _builtin_rule("not-false", Term.app("not", false), true),
        _builtin_rule("and-true-right", Term.app("and", x, true), x),
        _builtin_rule("and-true-left", Term.app("and", true, x), x),
        _builtin_rule("and-false-right", Term.app("and", x, false), false),
        _builtin_rule("and-false-left", Term.app("and", false, x), false),
        _builtin_rule("or-false-right", Term.app("or", x, false), x),
        _builtin_rule("or-false-left", Term.app("or", false, x), x),
        _builtin_rule("or-true-right", Term.app("or", x, true), true),
        _builtin_rule("or-true-left", Term.app("or", true, x), true),
        _builtin_rule("eq-self", Term.app("eq", x, x), true),
        _builtin_rule("ite-true", Term.app("ite", true, x, y), x),
        _builtin_rule("ite-false", Term.app("ite", false, x, y), y),
        _builtin_rule(
            "add-comm", Term.app("add", x, y), Term.app("add", y, x), oriented=False
        ),
        _builtin_rule(
            "mul-comm", Term.app("mul", x, y), Term.app("mul", y, x), oriented=False
        ),
        _builtin_rule(
            "and-comm", Term.app("and", x, y), Term.app("and", y, x), oriented=False
        ),
        _builtin_rule(
            "or-comm", Term.app("or", x, y), Term.app("or", y, x), oriented=False
        ),
        _builtin_rule(
            "eq-comm", Term.app("eq", x, y), Term.app("eq", y, x), oriented=False
        ),
        _builtin_rule(
            "add-assoc",
            Term.app("add", Term.app("add", x, y), z),
            Term.app("add", x, Term.app("add", y, z)),
            oriented=False,
        ),
        _builtin_rule(
            "mul-assoc",
            Term.app("mul", Term.app("mul", x, y), z),
            Term.app("mul", x, Term.app("mul", y, z)),
            oriented=False,
        ),
        _builtin_rule(
            "and-assoc",
            Term.app("and", Term.app("and", x, y), z),
            Term.app("and", x, Term.app("and", y, z)),
            oriented=False,
        ),
        _builtin_rule(
            "or-assoc",
            Term.app("or", Term.app("or", x, y), z),
            Term.app("or", x, Term.app("or", y, z)),
            oriented=False,
        ),
    )
    return RewriteTheory(
        theory_id=BUILTIN_THEORY_ID,
        rules=rules,
        review_refs=(SAWM_EGRAPH_NORMALIZATION_EVIDENCE,),
        language="python",
    )


def _coerce_theory(value: Any) -> RewriteTheory:
    if value is None:
        return builtin_python_pure_theory()
    if isinstance(value, RewriteTheory):
        return value
    if isinstance(value, Mapping):
        return RewriteTheory.from_dict(value)
    raise ProgramEGraphError("theory must be RewriteTheory or a mapping")


def _apply_matches(
    graph: _EGraph,
    *,
    root: int,
    pattern: Term,
    instantiate: Term,
    applied: int,
) -> tuple[int, bool]:
    changed = False
    seen: set[tuple[tuple[str, int], ...]] = set()
    for subst in _match_in_class(pattern, root, graph, {}):
        key = tuple(
            sorted((name, graph.find(class_id)) for name, class_id in subst.items())
        )
        if key in seen:
            continue
        seen.add(key)
        if applied >= graph.bounds.max_matches_per_rule:
            break
        target = _instantiate(instantiate, subst, graph)
        if graph.union(graph.find(root), target):
            changed = True
        applied += 1
    return applied, changed


def _apply_rule(graph: _EGraph, rule: RewriteRule) -> tuple[int, bool]:
    applied = 0
    changed = False
    roots = list({graph.find(class_id) for class_id in range(graph.next_id)})
    for root in roots:
        applied, lhs_changed = _apply_matches(
            graph,
            root=root,
            pattern=rule.lhs,
            instantiate=rule.rhs,
            applied=applied,
        )
        changed = changed or lhs_changed
        if not rule.oriented:
            applied, rhs_changed = _apply_matches(
                graph,
                root=root,
                pattern=rule.rhs,
                instantiate=rule.lhs,
                applied=applied,
            )
            changed = changed or rhs_changed
        if applied >= graph.bounds.max_matches_per_rule:
            break
    return applied, changed


def _saturate(
    left: Term,
    *,
    right: Term | None,
    theory: RewriteTheory,
    bounds: SaturationBounds,
) -> tuple[_EGraph, int, tuple[str, ...], bool, bool]:
    graph = _EGraph(bounds)
    graph.add_term(left)
    if right is not None:
        graph.add_term(right)
    applied_ids: list[str] = []
    iterations = 0
    bound_hit = False
    try:
        changed = True
        while changed and iterations < bounds.max_iterations:
            changed = False
            iterations += 1
            for rule in theory.rules:
                count, rule_changed = _apply_rule(graph, rule)
                if count and rule.rule_id not in applied_ids:
                    applied_ids.append(rule.rule_id)
                if rule_changed:
                    changed = True
            if _fold_constants(graph):
                changed = True
                if "constant-fold" not in applied_ids:
                    applied_ids.append("constant-fold")
            graph.restore_congruence()
        if changed and iterations >= bounds.max_iterations:
            bound_hit = True
    except ProgramEGraphBoundsError:
        bound_hit = True
    return graph, iterations, tuple(applied_ids), bound_hit, _inconsistent(graph)


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


def _promotion_kind(kind: Any) -> str:
    if type(kind) is str:
        marker = kind.strip().lower().replace("-", "_").replace(" ", "_")
        if marker in UNSOUND_EVIDENCE_KINDS or marker in FORBIDDEN_FIELD_MARKERS:
            raise ProgramEGraphAuthorityError(
                "similarity/nearest-neighbor is not a semantic relation kind"
            )
    return _enum(kind, RelationKind, "relation_kind")


@dataclass(frozen=True, slots=True)
class EqualitySaturationPlan:
    """Deterministic saturation receipt.  Never grants semantic authority."""

    disposition: SaturationDisposition | str
    fragment_cid: str
    theory_cid: str
    environment_binding_cid: str
    language: str
    source: str
    normal_form: str
    normal_form_cid: str
    applied_rule_ids: tuple[str, ...]
    iterations: int
    eclass_count: int
    enode_count: int
    bounds: SaturationBounds
    reason_codes: tuple[str, ...] = ()
    peer_fragment_cid: str | None = None
    peer_source: str = ""
    equivalent: bool = False
    saturation_fixed_point: bool = False
    purity: FragmentPurity | str = FragmentPurity.PURE
    conflict_rule_ids: tuple[str, ...] = ()
    proposal_only: bool = True
    grants_semantic_authority: bool = False
    grants_write_authority: bool = False
    justifies_repair: bool = False
    ex_falso_admission: bool = False

    SCHEMA: ClassVar[str] = EQUALITY_SATURATION_PLAN_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_EGRAPH_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "disposition", _enum(self.disposition, SaturationDisposition, "disposition")
        )
        object.__setattr__(self, "fragment_cid", _cid(self.fragment_cid, "fragment_cid"))
        object.__setattr__(self, "theory_cid", _cid(self.theory_cid, "theory_cid"))
        object.__setattr__(
            self,
            "environment_binding_cid",
            _cid(self.environment_binding_cid, "environment_binding_cid"),
        )
        object.__setattr__(self, "language", _text(self.language, "language"))
        object.__setattr__(self, "source", _text(self.source, "source"))
        object.__setattr__(
            self, "normal_form", _text(self.normal_form, "normal_form", empty=True)
        )
        object.__setattr__(
            self, "normal_form_cid", _cid(self.normal_form_cid, "normal_form_cid")
        )
        object.__setattr__(
            self,
            "applied_rule_ids",
            tuple(_text(item, "applied_rule_id") for item in self.applied_rule_ids),
        )
        object.__setattr__(self, "iterations", _nonneg_int(self.iterations, "iterations"))
        object.__setattr__(
            self, "eclass_count", _nonneg_int(self.eclass_count, "eclass_count")
        )
        object.__setattr__(
            self, "enode_count", _nonneg_int(self.enode_count, "enode_count")
        )
        if not isinstance(self.bounds, SaturationBounds):
            raise ProgramEGraphError("bounds must be SaturationBounds")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(
            self,
            "peer_fragment_cid",
            _optional_cid(self.peer_fragment_cid, "peer_fragment_cid"),
        )
        object.__setattr__(
            self, "peer_source", _text(self.peer_source, "peer_source", empty=True)
        )
        object.__setattr__(self, "equivalent", _bool(self.equivalent, "equivalent"))
        object.__setattr__(
            self,
            "saturation_fixed_point",
            _bool(self.saturation_fixed_point, "saturation_fixed_point"),
        )
        object.__setattr__(self, "purity", _enum(self.purity, FragmentPurity, "purity"))
        object.__setattr__(
            self,
            "conflict_rule_ids",
            tuple(_text(item, "conflict_rule_id") for item in self.conflict_rule_ids),
        )
        if self.proposal_only is not True:
            raise ProgramEGraphAuthorityError("saturation plans must remain proposal-only")
        if self.grants_semantic_authority or self.grants_write_authority:
            raise ProgramEGraphAuthorityError(
                "e-graph normalization cannot grant semantic or write authority"
            )
        if self.ex_falso_admission:
            raise ProgramEGraphAuthorityError(
                "contradictions cannot grant ex falso admission"
            )
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "grants_semantic_authority", False)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "ex_falso_admission", False)
        if self.disposition in {
            SaturationDisposition.CONFLICT.value,
            SaturationDisposition.UNSUPPORTED.value,
            SaturationDisposition.ABSTAINED.value,
            SaturationDisposition.REJECTED.value,
            SaturationDisposition.STALE.value,
            SaturationDisposition.BOUND_EXHAUSTED.value,
            SaturationDisposition.INEQUIVALENT_UNPROVED.value,
        }:
            object.__setattr__(self, "justifies_repair", False)
            object.__setattr__(self, "equivalent", False)
        else:
            object.__setattr__(self, "justifies_repair", _bool(self.justifies_repair, "justifies_repair"))
        if self.equivalent and self.disposition not in {
            SaturationDisposition.EQUIVALENT.value,
            SaturationDisposition.NORMALIZED.value,
        }:
            raise ProgramEGraphError(
                "equivalent plans require normalized or equivalent disposition"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "evidence": SAWM_EGRAPH_NORMALIZATION_EVIDENCE,
            "disposition": self.disposition,
            "fragment_cid": self.fragment_cid,
            "theory_cid": self.theory_cid,
            "environment_binding_cid": self.environment_binding_cid,
            "language": self.language,
            "source": self.source,
            "normal_form": self.normal_form,
            "normal_form_cid": self.normal_form_cid,
            "applied_rule_ids": list(self.applied_rule_ids),
            "iterations": self.iterations,
            "eclass_count": self.eclass_count,
            "enode_count": self.enode_count,
            "bounds": self.bounds.to_dict(),
            "reason_codes": list(self.reason_codes),
            "peer_fragment_cid": self.peer_fragment_cid,
            "peer_source": self.peer_source,
            "equivalent": self.equivalent,
            "saturation_fixed_point": self.saturation_fixed_point,
            "purity": self.purity,
            "conflict_rule_ids": list(self.conflict_rule_ids),
            "proposal_only": True,
            "grants_semantic_authority": False,
            "grants_write_authority": False,
            "justifies_repair": self.justifies_repair,
            "ex_falso_admission": False,
            "producer_id": PRODUCER_ID,
            "version": NORMALIZER_VERSION,
        }

    @property
    def plan_cid(self) -> str:
        return _identity_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["plan_cid"] = self.plan_cid
        return payload

    @property
    def supports_equivalence_promotion(self) -> bool:
        return (
            self.equivalent
            and self.disposition
            in {
                SaturationDisposition.EQUIVALENT.value,
                SaturationDisposition.NORMALIZED.value,
            }
            and not self.ex_falso_admission
            and self.purity == FragmentPurity.PURE.value
        )


@dataclass(frozen=True, slots=True)
class RelationPromotionProposal:
    """Proposal-only scoped equivalence.  Independent admission still required."""

    disposition: PromotionDisposition | str
    claim: ProgramRelationClaim | None
    evidence_basis: EvidenceBasis | str
    saturation_plan_cid: str | None
    proof_receipt_cid: str | None
    recommended_authority_status: str
    reason_codes: tuple[str, ...]
    independently_admitted: bool = False
    proposal_only: bool = True
    grants_semantic_authority: bool = False
    grants_write_authority: bool = False
    may_influence_planning: bool = False
    justifies_repair: bool = False
    contradiction_disposition: ContradictionDisposition | str = (
        ContradictionDisposition.NOT_APPLICABLE
    )
    validation_verdict: RelationValidationVerdict | str = RelationValidationVerdict.UNKNOWN
    left_cid: str = ""
    right_cid: str = ""
    scope_cid: str = ""

    SCHEMA: ClassVar[str] = RELATION_PROMOTION_PROPOSAL_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_EGRAPH_NORMALIZER_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, PromotionDisposition, "disposition"),
        )
        object.__setattr__(
            self, "evidence_basis", _enum(self.evidence_basis, EvidenceBasis, "evidence_basis")
        )
        if self.claim is not None and not isinstance(self.claim, ProgramRelationClaim):
            raise ProgramEGraphError("claim must be a ProgramRelationClaim")
        object.__setattr__(
            self,
            "saturation_plan_cid",
            _optional_cid(self.saturation_plan_cid, "saturation_plan_cid"),
        )
        object.__setattr__(
            self,
            "proof_receipt_cid",
            _optional_cid(self.proof_receipt_cid, "proof_receipt_cid"),
        )
        object.__setattr__(
            self,
            "recommended_authority_status",
            _enum(
                self.recommended_authority_status,
                RelationAuthorityStatus,
                "recommended_authority_status",
            ),
        )
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(
            self,
            "contradiction_disposition",
            _enum(
                self.contradiction_disposition,
                ContradictionDisposition,
                "contradiction_disposition",
            ),
        )
        object.__setattr__(
            self,
            "validation_verdict",
            _enum(
                self.validation_verdict,
                RelationValidationVerdict,
                "validation_verdict",
            ),
        )
        if self.independently_admitted or self.grants_semantic_authority or self.grants_write_authority:
            raise ProgramEGraphAuthorityError(
                "relation promotion cannot self-admit or grant authority"
            )
        if self.proposal_only is not True:
            raise ProgramEGraphAuthorityError("relation promotion must remain proposal-only")
        if self.may_influence_planning:
            raise ProgramEGraphAuthorityError(
                "promotion proposals cannot influence planning before independent admission"
            )
        object.__setattr__(self, "independently_admitted", False)
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "grants_semantic_authority", False)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "may_influence_planning", False)
        if self.disposition != PromotionDisposition.PROPOSED.value:
            object.__setattr__(self, "justifies_repair", False)
        else:
            object.__setattr__(
                self, "justifies_repair", _bool(self.justifies_repair, "justifies_repair")
            )
        if self.claim is not None:
            if self.claim.authority_status not in {
                RelationAuthorityStatus.CANDIDATE.value,
                RelationAuthorityStatus.ASSERTED.value,
            }:
                raise ProgramEGraphAuthorityError(
                    "promotion claims cannot self-promote to validated or proved"
                )
            object.__setattr__(self, "left_cid", self.claim.left_cid)
            object.__setattr__(self, "right_cid", self.claim.right_cid)
            object.__setattr__(self, "scope_cid", self.claim.scope_cid)
        else:
            object.__setattr__(
                self, "left_cid", _text(self.left_cid, "left_cid", empty=True)
            )
            object.__setattr__(
                self, "right_cid", _text(self.right_cid, "right_cid", empty=True)
            )
            object.__setattr__(
                self, "scope_cid", _text(self.scope_cid, "scope_cid", empty=True)
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "evidence": SAWM_EGRAPH_NORMALIZATION_EVIDENCE,
            "disposition": self.disposition,
            "claim": None if self.claim is None else self.claim.to_dict(),
            "evidence_basis": self.evidence_basis,
            "saturation_plan_cid": self.saturation_plan_cid,
            "proof_receipt_cid": self.proof_receipt_cid,
            "recommended_authority_status": self.recommended_authority_status,
            "reason_codes": list(self.reason_codes),
            "independently_admitted": False,
            "proposal_only": True,
            "grants_semantic_authority": False,
            "grants_write_authority": False,
            "may_influence_planning": False,
            "justifies_repair": self.justifies_repair,
            "contradiction_disposition": self.contradiction_disposition,
            "validation_verdict": self.validation_verdict,
            "left_cid": self.left_cid,
            "right_cid": self.right_cid,
            "scope_cid": self.scope_cid,
            "producer_id": PRODUCER_ID,
            "version": NORMALIZER_VERSION,
        }

    @property
    def proposal_cid(self) -> str:
        return _identity_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["proposal_cid"] = self.proposal_cid
        return payload


def _empty_normal_form_cid(*, source: str, language: str, reason: str) -> str:
    return _identity_cid(
        {
            "schema": PROGRAM_FRAGMENT_SCHEMA,
            "language": language,
            "source": source,
            "term": None,
            "purity": FragmentPurity.UNSUPPORTED.value,
            "reason": reason,
        }
    )


def _plan_for_failure(
    *,
    fragment: ParsedFragment,
    theory: RewriteTheory,
    environment_binding_cid: str,
    disposition: SaturationDisposition,
    reason_codes: Sequence[str],
    peer: ParsedFragment | None = None,
    conflict_rule_ids: Sequence[str] = (),
    bounds: SaturationBounds,
) -> EqualitySaturationPlan:
    return EqualitySaturationPlan(
        disposition=disposition,
        fragment_cid=fragment.fragment_cid,
        theory_cid=theory.theory_cid,
        environment_binding_cid=environment_binding_cid,
        language=fragment.language,
        source=fragment.source,
        normal_form="",
        normal_form_cid=_empty_normal_form_cid(
            source=fragment.source,
            language=fragment.language,
            reason=disposition.value,
        ),
        applied_rule_ids=(),
        iterations=0,
        eclass_count=0,
        enode_count=0,
        bounds=bounds,
        reason_codes=reason_codes,
        peer_fragment_cid=None if peer is None else peer.fragment_cid,
        peer_source="" if peer is None else peer.source,
        equivalent=False,
        saturation_fixed_point=False,
        purity=fragment.purity,
        conflict_rule_ids=tuple(conflict_rule_ids),
        justifies_repair=False,
    )


def _inspect_forbidden_evidence(*values: Any) -> None:
    for value in values:
        if value is None or value is False:
            continue
        if value is True:
            raise ProgramEGraphAuthorityError(
                "similarity/model evidence cannot support equivalence promotion"
            )
        if isinstance(value, Mapping):
            _plain(value)
            kinds = {
                str(value.get("kind") or ""),
                str(value.get("evidence_kind") or ""),
                str(value.get("basis") or ""),
            }
            if kinds & UNSOUND_EVIDENCE_KINDS:
                raise ProgramEGraphAuthorityError(
                    "similarity/model evidence cannot support equivalence promotion"
                )
            if value.get("model_authored") is True:
                raise ProgramEGraphAuthorityError(
                    "model output cannot support equivalence promotion"
                )
            continue
        if isinstance(value, (str, int, float)):
            raise ProgramEGraphAuthorityError(
                "similarity/model evidence cannot support equivalence promotion"
            )


def _proof_fields(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    payload = {}
    for name in (
        "disposition",
        "proof_status",
        "reconstruction_id",
        "replay_id",
        "model_authored",
        "admitted",
        "tree_id",
        "admission_id",
        "reason_codes",
    ):
        if hasattr(value, name):
            item = getattr(value, name)
            payload[name] = item.value if isinstance(item, Enum) else item
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        encoded = to_dict()
        if isinstance(encoded, Mapping):
            payload.update(encoded)
    return payload or None


def _admitted_kernel_proof(value: Any) -> tuple[bool, str | None, tuple[str, ...]]:
    fields = _proof_fields(value)
    if fields is None:
        return False, None, ()
    _inspect_forbidden_evidence(fields)
    disposition = str(fields.get("disposition") or "").strip()
    status = str(fields.get("proof_status") or "").strip()
    reconstruction = str(fields.get("reconstruction_id") or "").strip()
    model_authored = fields.get("model_authored") is True
    reasons = tuple(
        str(item) for item in (fields.get("reason_codes") or ()) if str(item).strip()
    )
    if model_authored:
        return False, None, reasons + ("model_authored_not_authority",)
    if disposition == "admitted_refutation" or status == "validated_refuted":
        return False, None, reasons + ("admitted_refutation",)
    admitted = (
        disposition == "admitted_proof"
        and status == "kernel_verified"
        and bool(reconstruction)
        and fields.get("admitted", True) is not False
    )
    proof_cid = None
    if admitted:
        try:
            proof_cid = _identity_cid(_plain(dict(fields)))
        except ProgramEGraphError:
            proof_cid = None
            admitted = False
            reasons = reasons + ("proof_receipt_unaddressable",)
    return admitted, proof_cid, reasons


class ProgramEGraphNormalizer:
    """Bound equality-saturation operator for pure program fragments."""

    INTERFACE: ClassVar[str] = PROGRAM_EGRAPH_NORMALIZER_INTERFACE
    VERSION: ClassVar[str] = NORMALIZER_VERSION

    def __init__(
        self,
        *,
        expected_environment_binding_cid: str | None = None,
        expected_theory_cid: str | None = None,
        theory: RewriteTheory | Mapping[str, Any] | None = None,
        bounds: SaturationBounds | Mapping[str, Any] | None = None,
        declared_pure_functions: Sequence[str] = (),
    ) -> None:
        self.expected_environment_binding_cid = (
            None
            if expected_environment_binding_cid is None
            else _cid(expected_environment_binding_cid, "expected_environment_binding_cid")
        )
        self.expected_theory_cid = (
            None
            if expected_theory_cid is None
            else _cid(expected_theory_cid, "expected_theory_cid")
        )
        self.theory = _coerce_theory(theory)
        self.bounds = SaturationBounds.from_mapping(bounds)
        self.declared_pure_functions = _sorted_unique(
            declared_pure_functions, "declared_pure_function"
        )

    def normalize(
        self,
        fragment: str | Mapping[str, Any] | ParsedFragment,
        *,
        peer_fragment: str | Mapping[str, Any] | ParsedFragment | None = None,
        language: str = "python",
        environment_binding_cid: str,
        theory: RewriteTheory | Mapping[str, Any] | None = None,
        bounds: SaturationBounds | Mapping[str, Any] | None = None,
        declared_pure_functions: Sequence[str] | None = None,
        expected_theory_cid: str | None = None,
        expected_environment_binding_cid: str | None = None,
    ) -> EqualitySaturationPlan:
        return normalize_program_fragment(
            fragment,
            peer_fragment=peer_fragment,
            language=language,
            environment_binding_cid=environment_binding_cid,
            theory=self.theory if theory is None else theory,
            bounds=self.bounds if bounds is None else bounds,
            declared_pure_functions=(
                self.declared_pure_functions
                if declared_pure_functions is None
                else declared_pure_functions
            ),
            expected_theory_cid=(
                self.expected_theory_cid
                if expected_theory_cid is None
                else expected_theory_cid
            ),
            expected_environment_binding_cid=(
                self.expected_environment_binding_cid
                if expected_environment_binding_cid is None
                else expected_environment_binding_cid
            ),
        )

    def propose_promotion(
        self,
        left_fragment: str | Mapping[str, Any],
        right_fragment: str | Mapping[str, Any],
        *,
        scope: RelationScope | Mapping[str, Any],
        relation_kind: RelationKind | str = RelationKind.LOGICAL_EQUIVALENCE,
        **kwargs: Any,
    ) -> RelationPromotionProposal:
        return propose_relation_promotion(
            left_fragment,
            right_fragment,
            scope=scope,
            relation_kind=relation_kind,
            theory=kwargs.pop("theory", self.theory),
            bounds=kwargs.pop("bounds", self.bounds),
            declared_pure_functions=kwargs.pop(
                "declared_pure_functions", self.declared_pure_functions
            ),
            expected_theory_cid=kwargs.pop(
                "expected_theory_cid", self.expected_theory_cid
            ),
            expected_environment_binding_cid=kwargs.pop(
                "expected_environment_binding_cid",
                self.expected_environment_binding_cid,
            ),
            **kwargs,
        )


def normalize_program_fragment(
    fragment: str | Mapping[str, Any] | ParsedFragment,
    *,
    peer_fragment: str | Mapping[str, Any] | ParsedFragment | None = None,
    language: str = "python",
    environment_binding_cid: str,
    theory: RewriteTheory | Mapping[str, Any] | None = None,
    bounds: SaturationBounds | Mapping[str, Any] | None = None,
    declared_pure_functions: Sequence[str] = (),
    expected_theory_cid: str | None = None,
    expected_environment_binding_cid: str | None = None,
) -> EqualitySaturationPlan:
    """Normalize a pure fragment under a declared sound theory."""

    env_cid = _cid(environment_binding_cid, "environment_binding_cid")
    expected_env = (
        None
        if expected_environment_binding_cid is None
        else _cid(expected_environment_binding_cid, "expected_environment_binding_cid")
    )
    bound_set = SaturationBounds.from_mapping(bounds)
    declared = _sorted_unique(declared_pure_functions, "declared_pure_function")
    language_value = _text(language, "language")
    left = _coerce_fragment(
        fragment, "fragment", declared_pure=declared, language=language_value
    )
    peer = (
        None
        if peer_fragment is None
        else _coerce_fragment(
            peer_fragment,
            "peer_fragment",
            declared_pure=declared,
            language=language_value,
        )
    )
    theory_obj = _coerce_theory(theory)
    expected_theory = (
        None
        if expected_theory_cid is None
        else _cid(expected_theory_cid, "expected_theory_cid")
    )
    if expected_env is not None and expected_env != env_cid:
        raise ProgramEGraphStaleError("environment_binding_cid is stale")
    if expected_theory is not None and expected_theory != theory_obj.theory_cid:
        raise ProgramEGraphStaleError("theory_cid is stale")
    if language_value not in ADMITTED_LANGUAGES or left.language not in ADMITTED_LANGUAGES:
        return _plan_for_failure(
            fragment=left,
            theory=theory_obj,
            environment_binding_cid=env_cid,
            disposition=SaturationDisposition.UNSUPPORTED,
            reason_codes=(f"language_unavailable:{language_value}",),
            peer=peer,
            bounds=bound_set,
        )
    conflicts = theory_obj.oriented_conflicts()
    if conflicts:
        conflict_ids = tuple(item for pair in conflicts for item in pair)
        return _plan_for_failure(
            fragment=left,
            theory=theory_obj,
            environment_binding_cid=env_cid,
            disposition=SaturationDisposition.CONFLICT,
            reason_codes=("conflicting_rules",),
            peer=peer,
            conflict_rule_ids=conflict_ids,
            bounds=bound_set,
        )
    if left.term is None or left.purity is not FragmentPurity.PURE:
        disposition = (
            SaturationDisposition.UNSUPPORTED
            if left.purity is not FragmentPurity.PURE
            else SaturationDisposition.ABSTAINED
        )
        return _plan_for_failure(
            fragment=left,
            theory=theory_obj,
            environment_binding_cid=env_cid,
            disposition=disposition,
            reason_codes=left.reason_codes or ("unsupported_construct",),
            peer=peer,
            bounds=bound_set,
        )
    if peer is not None and (peer.term is None or peer.purity is not FragmentPurity.PURE):
        return _plan_for_failure(
            fragment=left,
            theory=theory_obj,
            environment_binding_cid=env_cid,
            disposition=SaturationDisposition.UNSUPPORTED,
            reason_codes=peer.reason_codes or ("unsupported_peer_construct",),
            peer=peer,
            bounds=bound_set,
        )
    graph, iterations, applied, bound_hit, inconsistent = _saturate(
        left.term,
        right=None if peer is None else peer.term,
        theory=theory_obj,
        bounds=bound_set,
    )
    if inconsistent:
        return _plan_for_failure(
            fragment=left,
            theory=theory_obj,
            environment_binding_cid=env_cid,
            disposition=SaturationDisposition.CONFLICT,
            reason_codes=("inconsistent_egraph", "contradiction_abstention"),
            peer=peer,
            conflict_rule_ids=applied,
            bounds=bound_set,
        )
    left_class = graph.add_term(left.term)
    try:
        extracted = graph.extract(left_class)
    except ProgramEGraphError:
        return _plan_for_failure(
            fragment=left,
            theory=theory_obj,
            environment_binding_cid=env_cid,
            disposition=SaturationDisposition.ABSTAINED,
            reason_codes=("extraction_failed",),
            peer=peer,
            bounds=bound_set,
        )
    normal_source = extracted.render()
    normal_payload = {
        "schema": PROGRAM_FRAGMENT_SCHEMA,
        "language": left.language,
        "source": normal_source,
        "term": extracted.canonical(),
        "purity": FragmentPurity.PURE.value,
    }
    equivalent = False
    peer_cid = None
    peer_source = ""
    if peer is not None and peer.term is not None:
        peer_class = graph.add_term(peer.term)
        equivalent = graph.find(left_class) == graph.find(peer_class)
        peer_cid = peer.fragment_cid
        peer_source = peer.source
    else:
        # A fragment is equivalent to the normal form extracted from its class.
        equivalent = True
        peer_cid = _identity_cid(normal_payload)
        peer_source = normal_source
    if bound_hit:
        disposition = SaturationDisposition.BOUND_EXHAUSTED
        equivalent = False
        reasons = ("saturation_bound_exhausted",)
        fixed = False
    elif peer is not None and not equivalent:
        disposition = SaturationDisposition.INEQUIVALENT_UNPROVED
        reasons = ("equivalence_unproved",)
        fixed = True
    elif peer is not None and equivalent:
        disposition = SaturationDisposition.EQUIVALENT
        reasons = ("sound_normalization",)
        fixed = True
    else:
        disposition = SaturationDisposition.NORMALIZED
        reasons = ("sound_normalization", "deterministic_normal_form")
        fixed = True
    eclass_count = len({graph.find(class_id) for class_id in range(graph.next_id)})
    sound = disposition in {
        SaturationDisposition.EQUIVALENT,
        SaturationDisposition.NORMALIZED,
    }
    return EqualitySaturationPlan(
        disposition=disposition,
        fragment_cid=left.fragment_cid,
        theory_cid=theory_obj.theory_cid,
        environment_binding_cid=env_cid,
        language=left.language,
        source=left.source,
        normal_form=normal_source,
        normal_form_cid=_identity_cid(normal_payload),
        applied_rule_ids=applied,
        iterations=iterations,
        eclass_count=eclass_count,
        enode_count=graph.enode_count,
        bounds=bound_set,
        reason_codes=reasons,
        peer_fragment_cid=peer_cid,
        peer_source=peer_source,
        equivalent=equivalent and sound,
        saturation_fixed_point=fixed,
        purity=FragmentPurity.PURE,
        justifies_repair=equivalent and sound,
    )


def _coerce_scope(value: Any) -> RelationScope:
    if isinstance(value, RelationScope):
        return value
    if isinstance(value, Mapping):
        if "scope_cid" in value and "schema" in value:
            return RelationScope.from_dict(value)
        payload = dict(value)
        payload.pop("scope_cid", None)
        payload.pop("schema", None)
        return RelationScope(**payload)
    raise ProgramEGraphError("scope must be a RelationScope")


def _abstained_proposal(
    *,
    disposition: PromotionDisposition,
    reason_codes: Sequence[str],
    left_cid: str = "",
    right_cid: str = "",
    scope_cid: str = "",
    saturation_plan_cid: str | None = None,
    proof_receipt_cid: str | None = None,
    contradiction: ContradictionDisposition = ContradictionDisposition.ABSTENTION,
    verdict: RelationValidationVerdict = RelationValidationVerdict.ABSTAIN,
    evidence_basis: EvidenceBasis = EvidenceBasis.NONE,
) -> RelationPromotionProposal:
    return RelationPromotionProposal(
        disposition=disposition,
        claim=None,
        evidence_basis=evidence_basis,
        saturation_plan_cid=saturation_plan_cid,
        proof_receipt_cid=proof_receipt_cid,
        recommended_authority_status=RelationAuthorityStatus.UNKNOWN,
        reason_codes=reason_codes,
        contradiction_disposition=contradiction,
        validation_verdict=verdict,
        left_cid=left_cid,
        right_cid=right_cid,
        scope_cid=scope_cid,
        justifies_repair=False,
    )


def propose_relation_promotion(
    left_fragment: str | Mapping[str, Any] | ParsedFragment,
    right_fragment: str | Mapping[str, Any] | ParsedFragment,
    *,
    scope: RelationScope | Mapping[str, Any],
    relation_kind: RelationKind | str = RelationKind.LOGICAL_EQUIVALENCE,
    language: str = "python",
    environment_binding_cid: str | None = None,
    theory: RewriteTheory | Mapping[str, Any] | None = None,
    bounds: SaturationBounds | Mapping[str, Any] | None = None,
    declared_pure_functions: Sequence[str] = (),
    saturation: EqualitySaturationPlan | Mapping[str, Any] | None = None,
    proof_admission: Any = None,
    similarity: Any = None,
    model_output: Any = None,
    evidence: Mapping[str, Any] | None = None,
    expected_theory_cid: str | None = None,
    expected_environment_binding_cid: str | None = None,
) -> RelationPromotionProposal:
    """Propose scoped equivalence.  Only sound normalization or proof may support it."""

    _inspect_forbidden_evidence(similarity, model_output, evidence)
    kind = _promotion_kind(relation_kind)
    scope_obj = _coerce_scope(scope)
    env_cid = (
        scope_obj.environment_binding_cid
        if environment_binding_cid is None
        else _cid(environment_binding_cid, "environment_binding_cid")
    )
    if env_cid != scope_obj.environment_binding_cid:
        raise ProgramEGraphStaleError(
            "promotion environment_binding_cid is not bound to the relation scope"
        )
    admitted_proof, proof_cid, proof_reasons = _admitted_kernel_proof(proof_admission)
    if proof_admission is not None and not admitted_proof and "admitted_refutation" in proof_reasons:
        return _abstained_proposal(
            disposition=PromotionDisposition.REFUTED,
            reason_codes=_reason_codes(proof_reasons, ("proof_refutation",)),
            proof_receipt_cid=proof_cid,
            contradiction=ContradictionDisposition.NOT_APPLICABLE,
            verdict=RelationValidationVerdict.REFUTED,
            evidence_basis=EvidenceBasis.KERNEL_PROOF,
        )
    if saturation is None:
        plan = normalize_program_fragment(
            left_fragment,
            peer_fragment=right_fragment,
            language=language,
            environment_binding_cid=env_cid,
            theory=theory,
            bounds=bounds,
            declared_pure_functions=declared_pure_functions,
            expected_theory_cid=expected_theory_cid,
            expected_environment_binding_cid=expected_environment_binding_cid,
        )
    elif isinstance(saturation, EqualitySaturationPlan):
        plan = saturation
    else:
        raise ProgramEGraphError("saturation must be an EqualitySaturationPlan")
    left = _coerce_fragment(
        left_fragment,
        "left_fragment",
        declared_pure=declared_pure_functions,
        language=language,
    )
    right = _coerce_fragment(
        right_fragment,
        "right_fragment",
        declared_pure=declared_pure_functions,
        language=language,
    )
    if plan.environment_binding_cid != env_cid:
        raise ProgramEGraphStaleError("saturation environment binding is stale")
    if plan.disposition == SaturationDisposition.STALE.value:
        return _abstained_proposal(
            disposition=PromotionDisposition.STALE,
            reason_codes=plan.reason_codes or ("stale_saturation",),
            left_cid=left.fragment_cid,
            right_cid=right.fragment_cid,
            scope_cid=scope_obj.scope_cid,
            saturation_plan_cid=plan.plan_cid,
            verdict=RelationValidationVerdict.STALE,
        )
    if plan.disposition == SaturationDisposition.CONFLICT.value:
        return _abstained_proposal(
            disposition=PromotionDisposition.CONFLICT,
            reason_codes=_reason_codes(
                plan.reason_codes, ("conflicting_rules_abstain", "no_ex_falso")
            ),
            left_cid=left.fragment_cid,
            right_cid=right.fragment_cid,
            scope_cid=scope_obj.scope_cid,
            saturation_plan_cid=plan.plan_cid,
            contradiction=ContradictionDisposition.ABSTENTION,
            verdict=RelationValidationVerdict.CONFLICT,
        )
    if plan.disposition in {
        SaturationDisposition.UNSUPPORTED.value,
        SaturationDisposition.ABSTAINED.value,
        SaturationDisposition.REJECTED.value,
        SaturationDisposition.BOUND_EXHAUSTED.value,
        SaturationDisposition.INEQUIVALENT_UNPROVED.value,
    }:
        disposition = (
            PromotionDisposition.UNSUPPORTED
            if plan.disposition
            in {
                SaturationDisposition.UNSUPPORTED.value,
                SaturationDisposition.BOUND_EXHAUSTED.value,
            }
            else PromotionDisposition.ABSTAINED
        )
        if admitted_proof:
            disposition = PromotionDisposition.PROPOSED
        else:
            return _abstained_proposal(
                disposition=disposition,
                reason_codes=_reason_codes(
                    plan.reason_codes, ("promotion_requires_sound_evidence",)
                ),
                left_cid=left.fragment_cid,
                right_cid=right.fragment_cid,
                scope_cid=scope_obj.scope_cid,
                saturation_plan_cid=plan.plan_cid,
                verdict=(
                    RelationValidationVerdict.ABSTAIN
                    if disposition == PromotionDisposition.ABSTAINED
                    else RelationValidationVerdict.UNKNOWN
                ),
            )
    covered = {
        plan.fragment_cid,
        plan.peer_fragment_cid,
        plan.normal_form_cid,
    }
    plan_covers_pair = left.fragment_cid in covered and right.fragment_cid in covered
    basis = EvidenceBasis.NONE
    recommended = RelationAuthorityStatus.CANDIDATE
    evidence_cids: list[str] = []
    if admitted_proof and proof_cid is not None:
        basis = EvidenceBasis.KERNEL_PROOF
        recommended = RelationAuthorityStatus.PROVED
        evidence_cids.append(proof_cid)
    elif plan.supports_equivalence_promotion and plan_covers_pair:
        if left.fragment_cid == right.fragment_cid and not plan.applied_rule_ids:
            basis = EvidenceBasis.REFLEXIVITY
        else:
            basis = EvidenceBasis.NORMALIZATION
        recommended = RelationAuthorityStatus.VALIDATED
        evidence_cids.append(plan.plan_cid)
    else:
        return _abstained_proposal(
            disposition=PromotionDisposition.ABSTAINED,
            reason_codes=("promotion_requires_sound_normalization_or_proof",),
            left_cid=left.fragment_cid,
            right_cid=right.fragment_cid,
            scope_cid=scope_obj.scope_cid,
            saturation_plan_cid=plan.plan_cid,
            proof_receipt_cid=proof_cid,
        )
    try:
        claim = ProgramRelationClaim.from_scope(
            scope_obj,
            relation_kind=kind,
            left_cid=left.fragment_cid,
            right_cid=right.fragment_cid,
            authority_status=RelationAuthorityStatus.CANDIDATE,
            evidence_cids=evidence_cids,
        )
    except ProgramRelationError as exc:
        return _abstained_proposal(
            disposition=PromotionDisposition.ABSTAINED,
            reason_codes=("relation_claim_rejected", str(exc.__class__.__name__)),
            left_cid=left.fragment_cid,
            right_cid=right.fragment_cid,
            scope_cid=scope_obj.scope_cid,
            saturation_plan_cid=plan.plan_cid,
            proof_receipt_cid=proof_cid,
        )
    return RelationPromotionProposal(
        disposition=PromotionDisposition.PROPOSED,
        claim=claim,
        evidence_basis=basis,
        saturation_plan_cid=plan.plan_cid,
        proof_receipt_cid=proof_cid,
        recommended_authority_status=recommended,
        reason_codes=_reason_codes(
            (basis.value, "proposal_only", "independent_admission_required"),
            proof_reasons,
        ),
        contradiction_disposition=ContradictionDisposition.NOT_APPLICABLE,
        validation_verdict=RelationValidationVerdict.UNKNOWN,
        justifies_repair=basis
        in {
            EvidenceBasis.NORMALIZATION,
            EvidenceBasis.KERNEL_PROOF,
            EvidenceBasis.REFLEXIVITY,
        },
    )


__all__ = (
    "BUILTIN_THEORY_ID",
    "PROGRAM_EGRAPH_NORMALIZER_INTERFACE",
    "PROGRAM_WORLD_EGRAPH_INTERFACE",
    "SAWM_EGRAPH_NORMALIZATION_EVIDENCE",
    "EqualitySaturationPlan",
    "EvidenceBasis",
    "FragmentPurity",
    "ParsedFragment",
    "ProgramEGraphAuthorityError",
    "ProgramEGraphBoundsError",
    "ProgramEGraphError",
    "ProgramEGraphNormalizer",
    "ProgramEGraphStaleError",
    "PromotionDisposition",
    "RelationPromotionProposal",
    "RewriteRule",
    "RewriteSoundness",
    "RewriteTheory",
    "SaturationBounds",
    "SaturationDisposition",
    "Term",
    "builtin_python_pure_theory",
    "normalize_program_fragment",
    "propose_relation_promotion",
)
