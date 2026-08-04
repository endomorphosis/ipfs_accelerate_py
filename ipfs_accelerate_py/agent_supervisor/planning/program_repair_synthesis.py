"""Bounded deterministic program-repair synthesis with residual-only hybrid path.

Interface: ``ProgramRepairSynthesizer@1`` (PDR-051)

Orchestrates constraint / e-graph / enumerative / CEGIS search over *reviewed*
repair operators and grammars under exact obligations, bounds, and roots.
Every candidate remains **proposal-only** — no write, semantic, or proof
authority. Deterministic mode proves zero model calls. When deterministic
search leaves only behavior-fixed syntax debt, a separately named hybrid
service may request residual syntax under an exact target/path/semantics/
postconditions/tests packet and may not change authority, dependencies, or
meaning.

This module never imports LLM / model-provider surfaces. Hybrid packets are
emitted for a *separate* residual service; the synthesizer itself does not
invoke models.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.deterministic_doctor_contracts import DoctorAuthorityRoots
from ..proof.counterexample_guided_tactician import (
    CEGIS_LOOP_RESULT_SCHEMA,
    COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE,
    CandidateKind,
    CandidateValidationStatus,
    CegisBudget,
    CegisLoopResult,
    CegisStopReason,
    CegisValidationError,
    RefinementCandidate,
    run_counterexample_guided_loop,
)
from ..proof.formal_counterexamples import FormalCounterexample
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from .deterministic_doctor_synthesis import (
    DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE,
    DeterministicDoctorSynthesizer,
    DoctorSynthesisReceipt,
    DoctorSynthesisRequest,
    create_deterministic_doctor_synthesizer,
)
from .repair_operator_registry import (
    REPAIR_OPERATOR_REGISTRY_INTERFACE,
    RepairOperatorKind,
    RepairOperatorLookupDisposition,
    RepairOperatorLookupRequest,
    RepairOperatorLookupResult,
    RepairOperatorRegistry,
    ReviewedRepairHook,
    UnknownRepairOperatorError,
    build_default_repair_operator_registry,
    normalize_repair_operator_kind,
)

# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

PROGRAM_REPAIR_SYNTHESIZER_INTERFACE: Final[str] = "ProgramRepairSynthesizer@1"
PROGRAM_REPAIR_SYNTHESIZER_VERSION: Final[str] = "1.0.0"
PROGRAM_REPAIR_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-repair-request@1"
)
PROGRAM_REPAIR_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-repair-candidate@1"
)
PROGRAM_REPAIR_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-repair-receipt@1"
)
PROGRAM_REPAIR_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-repair-bounds@1"
)
EQUALITY_THEORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/declared-equality-theory@1"
)
EQUALITY_REWRITE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/equality-rewrite-receipt@1"
)
RESIDUAL_HYBRID_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/residual-hybrid-repair-packet@1"
)
RESIDUAL_HYBRID_SERVICE_INTERFACE: Final[str] = "ResidualHybridRepairService@1"
RESIDUAL_HYBRID_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/residual-hybrid-admission@1"
)
HYBRID_USAGE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/residual-hybrid-usage-receipt@1"
)

PRODUCER_ID: Final[str] = "program-repair-synthesis@1"
CONTRACT_VERSION: Final[int] = 1

MAX_OBLIGATIONS: Final[int] = 256
MAX_OPERATORS: Final[int] = 32
MAX_CANDIDATES: Final[int] = 64
MAX_SEARCH_STATES: Final[int] = 1_024
MAX_REWRITE_STEPS: Final[int] = 128
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_SPAN_BYTES: Final[int] = 65_536
MAX_FILE_BYTES: Final[int] = 1_048_576
MAX_REASON_CODES: Final[int] = 64
MAX_THEORY_RULES: Final[int] = 64
MAX_EGRAPH_NODES: Final[int] = 2_048
DEFAULT_MAX_ENUMERATIVE_CANDIDATES: Final[int] = 8
DEFAULT_MAX_CEGIS_ITERATIONS: Final[int] = 8
DEFAULT_MAX_REWRITE_DEPTH: Final[int] = 16

_FORBIDDEN_PROVIDER_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "llm_router",
        "model_provider",
        "openai",
        "anthropic",
        "provider_router",
        "todo_daemon.change_propagation_provider_router",
    }
)

_SCOPE_WIDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "extra_paths",
        "new_dependencies",
        "dependency_paths",
        "write_paths",
        "requested_write_paths",
        "authority_override",
        "policy_override",
        "completion_claim",
        "semantic_change",
        "meaning_change",
        "import_additions",
        "extra_imports",
        "extra_files",
    }
)

_AUTHORITY_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "write_authority",
        "semantic_authority",
        "proof_authority",
        "completion_authority",
        "grants_write_authority",
        "grants_proof_authority",
        "mutation_permit",
        "admission",
    }
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ProgramRepairMode(str, Enum):
    """Search / materialization modes for one synthesis run."""

    DETERMINISTIC = "deterministic"
    ENUMERATIVE = "enumerative"
    EQUALITY_REWRITE = "equality_rewrite"
    CEGIS = "cegis"
    HYBRID_RESIDUAL = "hybrid_residual"


class ProgramRepairDisposition(str, Enum):
    """Closed outcomes for one program-repair synthesis run."""

    SUPPORTED = "supported"
    ABSTAIN = "abstain"
    RESIDUAL_DEBT = "residual_debt"
    APPROVAL_REQUIRED = "approval_required"
    BUDGET_EXHAUSTED = "budget_exhausted"

    @property
    def grants_write_authority(self) -> bool:
        return False

    @property
    def is_success(self) -> bool:
        return self is ProgramRepairDisposition.SUPPORTED

    @property
    def proposal_only(self) -> bool:
        return True


class ProgramRepairReason(str, Enum):
    """Stable machine-readable repair-synthesis reason codes."""

    RENDERED = "deterministic_candidate_supported"
    NO_ADMISSIBLE_OPERATOR = "no_admissible_reviewed_operator"
    OPERATOR_NOT_REVIEWED = "operator_not_in_reviewed_grammar"
    OBLIGATION_MISMATCH = "obligation_root_or_bound_mismatch"
    ROOT_MISMATCH = "root_mismatch"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    MALFORMED_INPUT = "malformed_input"
    EXTRA_FILE = "extra_file_or_path"
    EXTRA_IMPORT = "extra_import"
    EXTRA_DEPENDENCY = "extra_dependency"
    NON_IDEMPOTENT = "non_idempotent_candidate"
    SCOPE_WIDENING = "scope_widening_output"
    AUTHORITY_CLAIM = "forbidden_authority_claim"
    PROVIDER_OR_MODEL_CALL = "provider_or_model_import_or_call"
    ZERO_MODEL_CALLS = "deterministic_zero_model_calls"
    EQUALITY_UNPROVED = "equality_not_proved_under_declared_theory"
    EQUALITY_PROVED = "equality_proved_under_declared_theory"
    CEGIS_CLOSED = "cegis_closed_on_fresh_receipt"
    CEGIS_OPEN = "cegis_open_or_budget_exhausted"
    CEGIS_INDEPENDENT_REJECT = "cegis_independent_validation_rejected"
    RESIDUAL_PACKET_EMITTED = "behavior_fixed_syntax_debt_residual"
    HYBRID_REJECTED = "hybrid_residual_rejected"
    HYBRID_ADMITTED = "hybrid_residual_syntax_admitted"
    PROPOSAL_ONLY = "candidate_is_proposal_only"
    SEARCH_EMPTY = "enumerative_search_empty"
    UNDECLARED_THEORY = "equality_theory_not_declared"
    MEANING_CHANGE = "meaning_or_dependency_change_forbidden"
    PATH_NOT_BOUNDED = "path_not_bounded"
    NO_PARTIAL_OVERLAY = "no_partial_overlay"


class ResidualHybridDisposition(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    DETERMINISTIC_CLOSED = "deterministic_closed"


class EqualityRewriteStatus(str, Enum):
    PROVED = "proved"
    UNPROVED = "unproved"
    UNSUPPORTED = "unsupported"
    BUDGET_EXHAUSTED = "budget_exhausted"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProgramRepairSynthesisError(ContractValidationError):
    """Malformed program-repair input or closed-boundary violation."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: ProgramRepairReason | str = ProgramRepairReason.MALFORMED_INPUT,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(getattr(reason_code, "value", reason_code))


class ProgramRepairAuthorityError(ProgramRepairSynthesisError):
    """Attempt to invent authority, broaden scope, or invoke a model."""


class ProgramRepairBoundsError(ProgramRepairSynthesisError):
    """A search or packet bound was exceeded."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if value is None:
        if required:
            raise ProgramRepairSynthesisError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise ProgramRepairSynthesisError(f"{name} must be a string")
    text = value.strip() if name.endswith(("_id", "_ref", "_cid")) else value
    if required and not text:
        raise ProgramRepairSynthesisError(f"{name} is required")
    if len(text.encode("utf-8")) > limit:
        raise ProgramRepairBoundsError(
            f"{name} exceeds its byte bound",
            reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
        )
    if "\x00" in text:
        raise ProgramRepairSynthesisError(f"{name} must not contain NUL bytes")
    return text


def _optional_text(value: Any, name: str, *, limit: int = MAX_TEXT_BYTES) -> str:
    return _text(value, name, required=False, limit=limit)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgramRepairSynthesisError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProgramRepairSynthesisError(f"{name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ProgramRepairBoundsError(
            f"{name} exceeds maximum {maximum}",
            reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
        )
    return value


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ProgramRepairSynthesisError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ProgramRepairBoundsError(
            f"{name} exceeds maximum {maximum}",
            reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
        )
    return value


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_OBLIGATIONS,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ProgramRepairSynthesisError(f"{name} must be a sequence of identifiers")
    else:
        raw = values
    if required and not raw:
        raise ProgramRepairSynthesisError(f"{name} is required")
    if len(raw) > limit:
        raise ProgramRepairBoundsError(
            f"{name} exceeds its bound",
            reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
        )
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _text(item, name)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(out)


def _path(value: Any, name: str = "path") -> str:
    text = _text(value, name, required=True, limit=MAX_PATH_BYTES)
    if "\\" in text or text.startswith("/") or ".." in PurePosixPath(text).parts:
        raise ProgramRepairAuthorityError(
            f"{name} must be a bounded relative repository path",
            reason_code=ProgramRepairReason.PATH_NOT_BOUNDED,
        )
    return text.replace("\\", "/")


def _paths(values: Any, name: str, *, limit: int = MAX_OPERATORS) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ProgramRepairSynthesisError(f"{name} must be a sequence of paths")
    if len(values) > limit:
        raise ProgramRepairBoundsError(
            f"{name} exceeds path bound",
            reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
        )
    return tuple(_path(item, name) for item in values)


def _enum(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except (TypeError, ValueError) as exc:
        raise ProgramRepairSynthesisError(
            f"{name} is not a valid {enum_cls.__name__}"
        ) from exc


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _assert_no_provider_markers(*texts: str) -> None:
    for text in texts:
        lowered = text.lower()
        for marker in _FORBIDDEN_PROVIDER_MARKERS:
            if marker in lowered:
                raise ProgramRepairAuthorityError(
                    f"provider/model surface forbidden: {marker}",
                    reason_code=ProgramRepairReason.PROVIDER_OR_MODEL_CALL,
                )


def _walk_forbidden_claims(value: Any, *, path: str = "") -> list[str]:
    reasons: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_s = str(key)
            norm = key_s.casefold().replace("-", "_")
            child = f"{path}.{key_s}" if path else key_s
            if norm in _AUTHORITY_CLAIM_KEYS and item is True:
                reasons.append(f"authority_claim:{child}")
            if norm in _SCOPE_WIDEN_KEYS and item not in (None, (), [], {}, False, ""):
                reasons.append(f"scope_key:{child}")
            reasons.extend(_walk_forbidden_claims(item, path=child))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for index, item in enumerate(value):
            reasons.extend(_walk_forbidden_claims(item, path=f"{path}[{index}]"))
    return reasons


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramRepairBounds:
    """Fixed search budgets for deterministic synthesis / CEGIS / rewrites."""

    SCHEMA: ClassVar[str] = PROGRAM_REPAIR_BOUNDS_SCHEMA

    max_enumerative_candidates: int = DEFAULT_MAX_ENUMERATIVE_CANDIDATES
    max_search_states: int = MAX_SEARCH_STATES
    max_cegis_iterations: int = DEFAULT_MAX_CEGIS_ITERATIONS
    max_candidates_per_iteration: int = 4
    max_identical_failures: int = 3
    max_rewrite_depth: int = DEFAULT_MAX_REWRITE_DEPTH
    max_egraph_nodes: int = MAX_EGRAPH_NODES
    max_model_calls: int = 0  # deterministic hard-zero; hybrid uses separate budget
    max_hybrid_calls: int = 1
    max_hybrid_tokens: int = 2_048

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_enumerative_candidates",
            _positive_int(
                self.max_enumerative_candidates,
                "max_enumerative_candidates",
                maximum=MAX_CANDIDATES,
            ),
        )
        object.__setattr__(
            self,
            "max_search_states",
            _positive_int(
                self.max_search_states, "max_search_states", maximum=MAX_SEARCH_STATES
            ),
        )
        object.__setattr__(
            self,
            "max_cegis_iterations",
            _positive_int(
                self.max_cegis_iterations,
                "max_cegis_iterations",
                maximum=64,
            ),
        )
        object.__setattr__(
            self,
            "max_candidates_per_iteration",
            _positive_int(
                self.max_candidates_per_iteration,
                "max_candidates_per_iteration",
                maximum=32,
            ),
        )
        object.__setattr__(
            self,
            "max_identical_failures",
            _positive_int(
                self.max_identical_failures, "max_identical_failures", maximum=32
            ),
        )
        object.__setattr__(
            self,
            "max_rewrite_depth",
            _positive_int(
                self.max_rewrite_depth, "max_rewrite_depth", maximum=MAX_REWRITE_STEPS
            ),
        )
        object.__setattr__(
            self,
            "max_egraph_nodes",
            _positive_int(
                self.max_egraph_nodes, "max_egraph_nodes", maximum=MAX_EGRAPH_NODES
            ),
        )
        # Deterministic mode hard-zeros model calls on the bounds object itself.
        if self.max_model_calls != 0:
            raise ProgramRepairAuthorityError(
                "program-repair bounds must hard-zero max_model_calls; "
                "hybrid residual uses a separately named budget",
                reason_code=ProgramRepairReason.PROVIDER_OR_MODEL_CALL,
            )
        object.__setattr__(self, "max_model_calls", 0)
        object.__setattr__(
            self,
            "max_hybrid_calls",
            _nonneg_int(self.max_hybrid_calls, "max_hybrid_calls", maximum=8),
        )
        object.__setattr__(
            self,
            "max_hybrid_tokens",
            _nonneg_int(self.max_hybrid_tokens, "max_hybrid_tokens", maximum=100_000),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_REPAIR_BOUNDS_SCHEMA,
            "max_enumerative_candidates": self.max_enumerative_candidates,
            "max_search_states": self.max_search_states,
            "max_cegis_iterations": self.max_cegis_iterations,
            "max_candidates_per_iteration": self.max_candidates_per_iteration,
            "max_identical_failures": self.max_identical_failures,
            "max_rewrite_depth": self.max_rewrite_depth,
            "max_egraph_nodes": self.max_egraph_nodes,
            "max_model_calls": 0,
            "max_hybrid_calls": self.max_hybrid_calls,
            "max_hybrid_tokens": self.max_hybrid_tokens,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ProgramRepairBounds":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ProgramRepairSynthesisError("bounds must be a mapping")
        if payload.get("schema") not in {None, PROGRAM_REPAIR_BOUNDS_SCHEMA}:
            raise ProgramRepairSynthesisError("unsupported program-repair bounds schema")
        fields = {
            name: payload[name]
            for name in cls.__dataclass_fields__
            if name in payload and name != "SCHEMA"
        }
        return cls(**fields)

    def to_cegis_budget(self, *, finite_bounds: Mapping[str, Any] | None = None) -> CegisBudget:
        return CegisBudget(
            max_iterations=self.max_cegis_iterations,
            max_candidates_per_iteration=self.max_candidates_per_iteration,
            max_identical_failures=self.max_identical_failures,
            finite_bounds=dict(finite_bounds or {}),
        )


# ---------------------------------------------------------------------------
# Equality theory / e-graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EqualityRule:
    """One oriented, reviewed equality rewrite under a declared theory."""

    rule_id: str
    lhs: str
    rhs: str
    review_ref: str
    theory_id: str
    oriented: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _text(self.rule_id, "rule_id"))
        object.__setattr__(self, "lhs", _text(self.lhs, "lhs", limit=MAX_SPAN_BYTES))
        object.__setattr__(self, "rhs", _text(self.rhs, "rhs", limit=MAX_SPAN_BYTES))
        object.__setattr__(self, "review_ref", _text(self.review_ref, "review_ref"))
        object.__setattr__(self, "theory_id", _text(self.theory_id, "theory_id"))
        object.__setattr__(self, "oriented", _bool(self.oriented, "oriented"))
        if not self.oriented:
            raise ProgramRepairSynthesisError(
                "equality rules must be oriented under the declared theory"
            )
        if self.lhs == self.rhs:
            raise ProgramRepairSynthesisError("equality rule lhs and rhs must differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "lhs": self.lhs,
            "rhs": self.rhs,
            "review_ref": self.review_ref,
            "theory_id": self.theory_id,
            "oriented": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EqualityRule":
        if not isinstance(payload, Mapping):
            raise ProgramRepairSynthesisError("equality rule must be a mapping")
        return cls(
            rule_id=str(payload.get("rule_id") or ""),
            lhs=str(payload.get("lhs") or ""),
            rhs=str(payload.get("rhs") or ""),
            review_ref=str(payload.get("review_ref") or ""),
            theory_id=str(payload.get("theory_id") or ""),
            oriented=bool(payload.get("oriented", True)),
        )


@dataclass(frozen=True)
class DeclaredEqualityTheory(CanonicalContract):
    """Reviewed equality theory used for e-graph / equality saturation rewrites."""

    SCHEMA: ClassVar[str] = EQUALITY_THEORY_SCHEMA

    theory_id: str
    review_refs: tuple[str, ...]
    rules: tuple[EqualityRule, ...]
    repository_id: str = ""
    tree_id: str = ""
    grants_semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "theory_id", _text(self.theory_id, "theory_id"))
        object.__setattr__(
            self, "review_refs", _ids(self.review_refs, "review_refs", required=True)
        )
        if not self.rules:
            raise ProgramRepairSynthesisError("equality theory requires at least one rule")
        if len(self.rules) > MAX_THEORY_RULES:
            raise ProgramRepairBoundsError(
                "equality theory exceeds rule bound",
                reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
            )
        normalized: list[EqualityRule] = []
        for rule in self.rules:
            if isinstance(rule, EqualityRule):
                item = rule
            elif isinstance(rule, Mapping):
                item = EqualityRule.from_dict(rule)
            else:
                raise ProgramRepairSynthesisError("rules must be EqualityRule or mapping")
            if item.theory_id and item.theory_id != self.theory_id:
                raise ProgramRepairSynthesisError(
                    "rule theory_id must match declared theory"
                )
            # Force theory binding.
            item = EqualityRule(
                rule_id=item.rule_id,
                lhs=item.lhs,
                rhs=item.rhs,
                review_ref=item.review_ref,
                theory_id=self.theory_id,
                oriented=True,
            )
            normalized.append(item)
        object.__setattr__(self, "rules", tuple(normalized))
        object.__setattr__(
            self, "repository_id", _optional_text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _optional_text(self.tree_id, "tree_id"))
        if self.grants_semantic_authority is not False:
            raise ProgramRepairAuthorityError(
                "equality theory cannot grant semantic authority",
                reason_code=ProgramRepairReason.AUTHORITY_CLAIM,
            )
        object.__setattr__(self, "grants_semantic_authority", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "theory_id": self.theory_id,
            "review_refs": list(self.review_refs),
            "rules": [rule.to_dict() for rule in self.rules],
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "grants_semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeclaredEqualityTheory":
        if not isinstance(payload, Mapping):
            raise ProgramRepairSynthesisError("equality theory must be a mapping")
        if payload.get("schema") not in {None, EQUALITY_THEORY_SCHEMA}:
            raise ProgramRepairSynthesisError("unsupported equality theory schema")
        rules_raw = payload.get("rules") or ()
        rules = tuple(
            item if isinstance(item, EqualityRule) else EqualityRule.from_dict(item)
            for item in rules_raw
        )
        return cls(
            theory_id=str(payload.get("theory_id") or ""),
            review_refs=tuple(payload.get("review_refs") or ()),
            rules=rules,
            repository_id=str(payload.get("repository_id") or ""),
            tree_id=str(payload.get("tree_id") or ""),
            grants_semantic_authority=bool(
                payload.get("grants_semantic_authority", False)
            ),
        )


@dataclass(frozen=True)
class EqualityRewriteReceipt(CanonicalContract):
    """Proof that two terms are equivalent under a declared equality theory."""

    SCHEMA: ClassVar[str] = EQUALITY_REWRITE_RECEIPT_SCHEMA

    theory_id: str
    source_term: str
    target_term: str
    status: EqualityRewriteStatus
    applied_rule_ids: tuple[str, ...]
    rewrite_depth: int
    egraph_node_count: int
    reason_code: str = ""
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "theory_id", _text(self.theory_id, "theory_id"))
        object.__setattr__(
            self, "source_term", _text(self.source_term, "source_term", limit=MAX_SPAN_BYTES)
        )
        object.__setattr__(
            self, "target_term", _text(self.target_term, "target_term", limit=MAX_SPAN_BYTES)
        )
        object.__setattr__(
            self, "status", _enum(self.status, EqualityRewriteStatus, "status")
        )
        object.__setattr__(
            self, "applied_rule_ids", _ids(self.applied_rule_ids, "applied_rule_ids")
        )
        object.__setattr__(
            self, "rewrite_depth", _nonneg_int(self.rewrite_depth, "rewrite_depth")
        )
        object.__setattr__(
            self,
            "egraph_node_count",
            _nonneg_int(self.egraph_node_count, "egraph_node_count"),
        )
        object.__setattr__(
            self, "reason_code", _optional_text(self.reason_code, "reason_code")
        )
        if self.proposal_only is not True:
            raise ProgramRepairAuthorityError(
                "equality rewrite receipts must remain proposal-only",
                reason_code=ProgramRepairReason.PROPOSAL_ONLY,
            )
        if self.grants_write_authority or self.grants_semantic_authority:
            raise ProgramRepairAuthorityError(
                "equality rewrite cannot grant write or semantic authority",
                reason_code=ProgramRepairReason.AUTHORITY_CLAIM,
            )
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "grants_semantic_authority", False)

    @property
    def proved(self) -> bool:
        return self.status is EqualityRewriteStatus.PROVED

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "theory_id": self.theory_id,
            "source_term": self.source_term,
            "target_term": self.target_term,
            "status": self.status.value,
            "applied_rule_ids": list(self.applied_rule_ids),
            "rewrite_depth": self.rewrite_depth,
            "egraph_node_count": self.egraph_node_count,
            "reason_code": self.reason_code,
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_semantic_authority": False,
        }


class EqualityEGraph:
    """Minimal e-graph / equality-saturation engine under a declared theory.

    Terms are opaque strings. Congruence is string equality after rewrite.
    This is intentionally small and fail-closed: only declared rules apply.
    """

    def __init__(
        self,
        theory: DeclaredEqualityTheory,
        *,
        max_depth: int = DEFAULT_MAX_REWRITE_DEPTH,
        max_nodes: int = MAX_EGRAPH_NODES,
    ) -> None:
        if not isinstance(theory, DeclaredEqualityTheory):
            raise ProgramRepairSynthesisError("theory must be DeclaredEqualityTheory")
        self.theory = theory
        self.max_depth = _positive_int(max_depth, "max_depth", maximum=MAX_REWRITE_STEPS)
        self.max_nodes = _positive_int(max_nodes, "max_nodes", maximum=MAX_EGRAPH_NODES)
        self._parent: dict[str, str] = {}
        self._applied: list[str] = []
        self._nodes = 0

    def _find(self, term: str) -> str:
        parent = self._parent.get(term, term)
        if parent != term:
            root = self._find(parent)
            self._parent[term] = root
            return root
        return term

    def _union(self, left: str, right: str, *, rule_id: str) -> None:
        a = self._find(left)
        b = self._find(right)
        if a == b:
            return
        # Prefer shorter representative (oriented simplification).
        if len(b) < len(a):
            a, b = b, a
        self._parent[b] = a
        self._applied.append(rule_id)

    def _ensure(self, term: str) -> None:
        if term not in self._parent:
            if self._nodes >= self.max_nodes:
                raise ProgramRepairBoundsError(
                    "e-graph node budget exhausted",
                    reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
                )
            self._parent[term] = term
            self._nodes += 1

    def prove(self, source: str, target: str) -> EqualityRewriteReceipt:
        source_t = _text(source, "source_term", limit=MAX_SPAN_BYTES)
        target_t = _text(target, "target_term", limit=MAX_SPAN_BYTES)
        self._ensure(source_t)
        self._ensure(target_t)
        if source_t == target_t:
            return EqualityRewriteReceipt(
                theory_id=self.theory.theory_id,
                source_term=source_t,
                target_term=target_t,
                status=EqualityRewriteStatus.PROVED,
                applied_rule_ids=(),
                rewrite_depth=0,
                egraph_node_count=self._nodes,
                reason_code=ProgramRepairReason.EQUALITY_PROVED.value,
            )

        depth = 0
        changed = True
        try:
            while changed and depth < self.max_depth:
                changed = False
                depth += 1
                # Snapshot current class members.
                members = list(self._parent.keys())
                for term in members:
                    for rule in self.theory.rules:
                        if rule.lhs in term:
                            rewritten = term.replace(rule.lhs, rule.rhs, 1)
                            if rewritten != term:
                                self._ensure(rewritten)
                                self._union(term, rewritten, rule_id=rule.rule_id)
                                changed = True
                        # Also seed pure lhs/rhs.
                        if term == rule.lhs:
                            self._ensure(rule.rhs)
                            self._union(rule.lhs, rule.rhs, rule_id=rule.rule_id)
                            changed = True
                if self._find(source_t) == self._find(target_t):
                    return EqualityRewriteReceipt(
                        theory_id=self.theory.theory_id,
                        source_term=source_t,
                        target_term=target_t,
                        status=EqualityRewriteStatus.PROVED,
                        applied_rule_ids=tuple(dict.fromkeys(self._applied)),
                        rewrite_depth=depth,
                        egraph_node_count=self._nodes,
                        reason_code=ProgramRepairReason.EQUALITY_PROVED.value,
                    )
        except ProgramRepairBoundsError:
            return EqualityRewriteReceipt(
                theory_id=self.theory.theory_id,
                source_term=source_t,
                target_term=target_t,
                status=EqualityRewriteStatus.BUDGET_EXHAUSTED,
                applied_rule_ids=tuple(dict.fromkeys(self._applied)),
                rewrite_depth=depth,
                egraph_node_count=self._nodes,
                reason_code=ProgramRepairReason.BOUNDS_EXCEEDED.value,
            )

        if depth >= self.max_depth and self._find(source_t) != self._find(target_t):
            status = EqualityRewriteStatus.BUDGET_EXHAUSTED
            reason = ProgramRepairReason.BOUNDS_EXCEEDED.value
        else:
            status = EqualityRewriteStatus.UNPROVED
            reason = ProgramRepairReason.EQUALITY_UNPROVED.value
        return EqualityRewriteReceipt(
            theory_id=self.theory.theory_id,
            source_term=source_t,
            target_term=target_t,
            status=status,
            applied_rule_ids=tuple(dict.fromkeys(self._applied)),
            rewrite_depth=depth,
            egraph_node_count=self._nodes,
            reason_code=reason,
        )


def prove_equality_under_theory(
    theory: DeclaredEqualityTheory | Mapping[str, Any],
    source_term: str,
    target_term: str,
    *,
    max_depth: int = DEFAULT_MAX_REWRITE_DEPTH,
    max_nodes: int = MAX_EGRAPH_NODES,
) -> EqualityRewriteReceipt:
    """Prove source ≡ target under a declared equality theory (proposal-only)."""

    if isinstance(theory, Mapping):
        theory = DeclaredEqualityTheory.from_dict(theory)
    if not isinstance(theory, DeclaredEqualityTheory):
        raise ProgramRepairSynthesisError("theory must be DeclaredEqualityTheory")
    return EqualityEGraph(theory, max_depth=max_depth, max_nodes=max_nodes).prove(
        source_term, target_term
    )


# ---------------------------------------------------------------------------
# Residual hybrid packet / service
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualHybridPacket(CanonicalContract):
    """Exact residual packet for a separately named hybrid syntax service.

    Carries target, path, semantics digest, postconditions, and tests.
    Explicitly forbids authority / dependency / meaning changes.
    """

    SCHEMA: ClassVar[str] = RESIDUAL_HYBRID_PACKET_SCHEMA

    packet_id: str
    target_path: str
    span_start: int
    span_end: int
    semantics_digest: str
    postcondition_refs: tuple[str, ...]
    test_refs: tuple[str, ...]
    obligation_refs: tuple[str, ...]
    repository_id: str
    tree_id: str
    behavior_fixed: bool = True
    may_change_authority: bool = False
    may_change_dependencies: bool = False
    may_change_meaning: bool = False
    may_add_imports: bool = False
    may_add_files: bool = False
    allowed_paths: tuple[str, ...] = ()
    syntax_slot_id: str = ""
    reason_codes: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "packet_id", _text(self.packet_id, "packet_id"))
        object.__setattr__(self, "target_path", _path(self.target_path, "target_path"))
        object.__setattr__(
            self, "span_start", _nonneg_int(self.span_start, "span_start")
        )
        object.__setattr__(self, "span_end", _nonneg_int(self.span_end, "span_end"))
        if self.span_end < self.span_start:
            raise ProgramRepairSynthesisError("span_end must be >= span_start")
        object.__setattr__(
            self, "semantics_digest", _text(self.semantics_digest, "semantics_digest")
        )
        for name in (
            "postcondition_refs",
            "test_refs",
            "obligation_refs",
            "reason_codes",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "behavior_fixed", _bool(self.behavior_fixed, "behavior_fixed")
        )
        if self.behavior_fixed is not True:
            raise ProgramRepairAuthorityError(
                "hybrid residual packets require behavior_fixed=True",
                reason_code=ProgramRepairReason.MEANING_CHANGE,
            )
        for name in (
            "may_change_authority",
            "may_change_dependencies",
            "may_change_meaning",
            "may_add_imports",
            "may_add_files",
        ):
            if getattr(self, name) is not False:
                raise ProgramRepairAuthorityError(
                    f"hybrid residual packet must hard-zero {name}",
                    reason_code=ProgramRepairReason.MEANING_CHANGE,
                )
            object.__setattr__(self, name, False)
        allowed = self.allowed_paths or (self.target_path,)
        object.__setattr__(self, "allowed_paths", _paths(allowed, "allowed_paths"))
        if self.target_path not in self.allowed_paths:
            raise ProgramRepairAuthorityError(
                "target_path must be within allowed_paths",
                reason_code=ProgramRepairReason.SCOPE_WIDENING,
            )
        if len(self.allowed_paths) != 1 or self.allowed_paths[0] != self.target_path:
            raise ProgramRepairAuthorityError(
                "hybrid residual admits exactly one target path",
                reason_code=ProgramRepairReason.SCOPE_WIDENING,
            )
        object.__setattr__(
            self, "syntax_slot_id", _optional_text(self.syntax_slot_id, "syntax_slot_id")
        )
        object.__setattr__(
            self, "producer_id", _text(self.producer_id or PRODUCER_ID, "producer_id")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "packet_id": self.packet_id,
            "target_path": self.target_path,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "semantics_digest": self.semantics_digest,
            "postcondition_refs": list(self.postcondition_refs),
            "test_refs": list(self.test_refs),
            "obligation_refs": list(self.obligation_refs),
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "behavior_fixed": True,
            "may_change_authority": False,
            "may_change_dependencies": False,
            "may_change_meaning": False,
            "may_add_imports": False,
            "may_add_files": False,
            "allowed_paths": list(self.allowed_paths),
            "syntax_slot_id": self.syntax_slot_id,
            "reason_codes": list(self.reason_codes),
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualHybridPacket":
        if not isinstance(payload, Mapping):
            raise ProgramRepairSynthesisError("residual packet must be a mapping")
        if payload.get("schema") not in {None, RESIDUAL_HYBRID_PACKET_SCHEMA}:
            raise ProgramRepairSynthesisError("unsupported residual hybrid packet schema")
        field_names = set(cls.__dataclass_fields__) - {"SCHEMA"}
        values = {name: payload[name] for name in field_names if name in payload}
        return cls(**values)


@dataclass(frozen=True)
class HybridUsageReceipt(CanonicalContract):
    """Usage / admission receipt for the residual hybrid service."""

    SCHEMA: ClassVar[str] = HYBRID_USAGE_RECEIPT_SCHEMA

    packet_id: str
    disposition: ResidualHybridDisposition
    model_call_count: int
    token_count: int
    reason_codes: tuple[str, ...]
    syntax_digest: str = ""
    proposal_only: bool = True
    write_authority: bool = False
    semantic_authority: bool = False
    dependency_change: bool = False
    meaning_change: bool = False
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "packet_id", _text(self.packet_id, "packet_id"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ResidualHybridDisposition, "disposition"),
        )
        object.__setattr__(
            self, "model_call_count", _nonneg_int(self.model_call_count, "model_call_count")
        )
        object.__setattr__(
            self, "token_count", _nonneg_int(self.token_count, "token_count")
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes", required=True)
        )
        object.__setattr__(
            self, "syntax_digest", _optional_text(self.syntax_digest, "syntax_digest")
        )
        if self.proposal_only is not True:
            raise ProgramRepairAuthorityError(
                "hybrid usage must remain proposal-only",
                reason_code=ProgramRepairReason.PROPOSAL_ONLY,
            )
        for name in (
            "write_authority",
            "semantic_authority",
            "dependency_change",
            "meaning_change",
        ):
            if getattr(self, name) is not False:
                raise ProgramRepairAuthorityError(
                    f"hybrid usage must hard-zero {name}",
                    reason_code=ProgramRepairReason.AUTHORITY_CLAIM,
                )
            object.__setattr__(self, name, False)
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(
            self, "producer_id", _text(self.producer_id or PRODUCER_ID, "producer_id")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "packet_id": self.packet_id,
            "disposition": self.disposition.value,
            "model_call_count": self.model_call_count,
            "token_count": self.token_count,
            "reason_codes": list(self.reason_codes),
            "syntax_digest": self.syntax_digest,
            "proposal_only": True,
            "write_authority": False,
            "semantic_authority": False,
            "dependency_change": False,
            "meaning_change": False,
            "producer_id": self.producer_id,
        }


@dataclass(frozen=True)
class ResidualHybridAdmission(CanonicalContract):
    """Fail-closed admission of a hybrid residual syntax proposal."""

    SCHEMA: ClassVar[str] = RESIDUAL_HYBRID_ADMISSION_SCHEMA

    packet_id: str
    disposition: ResidualHybridDisposition
    reason_codes: tuple[str, ...]
    syntax: str = ""
    syntax_digest: str = ""
    proposal_only: bool = True
    usage: HybridUsageReceipt | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "packet_id", _text(self.packet_id, "packet_id"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ResidualHybridDisposition, "disposition"),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes", required=True)
        )
        syntax = self.syntax if isinstance(self.syntax, str) else ""
        if len(syntax.encode("utf-8")) > MAX_SPAN_BYTES:
            raise ProgramRepairBoundsError(
                "hybrid syntax exceeds span bound",
                reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
            )
        object.__setattr__(self, "syntax", syntax)
        digest = self.syntax_digest or (_sha256_text(syntax) if syntax else "")
        object.__setattr__(self, "syntax_digest", _optional_text(digest, "syntax_digest"))
        if self.proposal_only is not True:
            raise ProgramRepairAuthorityError(
                "hybrid admission is proposal-only",
                reason_code=ProgramRepairReason.PROPOSAL_ONLY,
            )
        object.__setattr__(self, "proposal_only", True)
        if self.usage is not None and not isinstance(self.usage, HybridUsageReceipt):
            raise ProgramRepairSynthesisError("usage must be HybridUsageReceipt")

    @property
    def admitted(self) -> bool:
        return self.disposition is ResidualHybridDisposition.ADMITTED

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "packet_id": self.packet_id,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "syntax": self.syntax,
            "syntax_digest": self.syntax_digest,
            "proposal_only": True,
            "usage": self.usage.to_dict() if self.usage is not None else None,
        }


class ResidualHybridRepairService:
    """Separately named residual-only hybrid repair admission surface.

    Does **not** inherit deterministic-Doctor authority. Admits only behavior-
    fixed syntax fill for an exact residual packet. Never changes authority,
    dependencies, or meaning. Does not invoke models itself — callers supply
    candidate syntax after an external residual provider run.
    """

    INTERFACE: ClassVar[str] = RESIDUAL_HYBRID_SERVICE_INTERFACE

    def __init__(self, *, bounds: ProgramRepairBounds | None = None) -> None:
        self.bounds = bounds or ProgramRepairBounds()
        self._calls = 0
        self._tokens = 0

    def admit(
        self,
        packet: ResidualHybridPacket | Mapping[str, Any],
        proposal: Mapping[str, Any] | str,
        *,
        response_tokens: int = 0,
        model_calls: int = 1,
    ) -> ResidualHybridAdmission:
        if isinstance(packet, Mapping):
            packet = ResidualHybridPacket.from_dict(packet)
        if not isinstance(packet, ResidualHybridPacket):
            raise ProgramRepairSynthesisError("packet must be ResidualHybridPacket")

        tokens = _nonneg_int(response_tokens, "response_tokens")
        calls = _nonneg_int(model_calls, "model_calls")
        self._calls += calls
        self._tokens += tokens

        if self._calls > self.bounds.max_hybrid_calls:
            usage = HybridUsageReceipt(
                packet_id=packet.packet_id,
                disposition=ResidualHybridDisposition.REJECTED,
                model_call_count=self._calls,
                token_count=self._tokens,
                reason_codes=(ProgramRepairReason.BOUNDS_EXCEEDED.value,),
            )
            return ResidualHybridAdmission(
                packet_id=packet.packet_id,
                disposition=ResidualHybridDisposition.REJECTED,
                reason_codes=(ProgramRepairReason.BOUNDS_EXCEEDED.value,),
                usage=usage,
            )
        if self._tokens > self.bounds.max_hybrid_tokens:
            usage = HybridUsageReceipt(
                packet_id=packet.packet_id,
                disposition=ResidualHybridDisposition.REJECTED,
                model_call_count=self._calls,
                token_count=self._tokens,
                reason_codes=(ProgramRepairReason.BOUNDS_EXCEEDED.value,),
            )
            return ResidualHybridAdmission(
                packet_id=packet.packet_id,
                disposition=ResidualHybridDisposition.REJECTED,
                reason_codes=(ProgramRepairReason.BOUNDS_EXCEEDED.value,),
                usage=usage,
            )

        if isinstance(proposal, str):
            try:
                import json

                decoded: Any = json.loads(proposal)
            except Exception as exc:
                raise ProgramRepairSynthesisError(
                    "hybrid proposal is not valid JSON",
                    reason_code=ProgramRepairReason.MALFORMED_INPUT,
                ) from exc
        else:
            decoded = proposal
        if not isinstance(decoded, Mapping):
            raise ProgramRepairSynthesisError(
                "hybrid proposal must be a mapping",
                reason_code=ProgramRepairReason.MALFORMED_INPUT,
            )

        reasons = _walk_forbidden_claims(decoded)
        syntax = str(decoded.get("syntax") or decoded.get("replacement") or "")
        path = str(decoded.get("path") or decoded.get("target_path") or packet.target_path)
        if path != packet.target_path:
            reasons.append(ProgramRepairReason.SCOPE_WIDENING.value)
        if decoded.get("extra_paths") not in (None, (), [], ""):
            reasons.append(ProgramRepairReason.EXTRA_FILE.value)
        if decoded.get("extra_imports") not in (None, (), [], ""):
            reasons.append(ProgramRepairReason.EXTRA_IMPORT.value)
        if decoded.get("dependency_paths") not in (None, (), [], ""):
            reasons.append(ProgramRepairReason.EXTRA_DEPENDENCY.value)
        if decoded.get("new_dependencies") not in (None, (), [], ""):
            reasons.append(ProgramRepairReason.EXTRA_DEPENDENCY.value)
        if not syntax:
            reasons.append(ProgramRepairReason.MALFORMED_INPUT.value)
        if not packet.behavior_fixed:
            reasons.append(ProgramRepairReason.MEANING_CHANGE.value)
        # Semantics digest binding: proposal may restate but not change it.
        proposed_semantics = str(
            decoded.get("semantics_digest") or packet.semantics_digest
        )
        if proposed_semantics != packet.semantics_digest:
            reasons.append(ProgramRepairReason.MEANING_CHANGE.value)

        if reasons:
            usage = HybridUsageReceipt(
                packet_id=packet.packet_id,
                disposition=ResidualHybridDisposition.REJECTED,
                model_call_count=self._calls,
                token_count=self._tokens,
                reason_codes=tuple(dict.fromkeys(reasons))[
                    :MAX_REASON_CODES
                ],
            )
            return ResidualHybridAdmission(
                packet_id=packet.packet_id,
                disposition=ResidualHybridDisposition.REJECTED,
                reason_codes=tuple(dict.fromkeys(reasons))[:MAX_REASON_CODES],
                usage=usage,
            )

        usage = HybridUsageReceipt(
            packet_id=packet.packet_id,
            disposition=ResidualHybridDisposition.ADMITTED,
            model_call_count=self._calls,
            token_count=self._tokens,
            reason_codes=(ProgramRepairReason.HYBRID_ADMITTED.value,),
            syntax_digest=_sha256_text(syntax),
        )
        return ResidualHybridAdmission(
            packet_id=packet.packet_id,
            disposition=ResidualHybridDisposition.ADMITTED,
            reason_codes=(ProgramRepairReason.HYBRID_ADMITTED.value,),
            syntax=syntax,
            syntax_digest=_sha256_text(syntax),
            usage=usage,
        )


# ---------------------------------------------------------------------------
# Request / candidate / receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramRepairRequest:
    """Inputs for bounded deterministic program-repair synthesis."""

    roots: DoctorAuthorityRoots
    obligation_refs: tuple[str, ...]
    target_paths: tuple[str, ...]
    operator_kinds: tuple[str, ...] = ()
    placement_refs: tuple[str, ...] = ()
    value_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    review_refs: tuple[str, ...] = ()
    postcondition_refs: tuple[str, ...] = ()
    test_refs: tuple[str, ...] = ()
    mode: ProgramRepairMode = ProgramRepairMode.DETERMINISTIC
    bounds: ProgramRepairBounds = field(default_factory=ProgramRepairBounds)
    equality_theory: DeclaredEqualityTheory | None = None
    source_term: str = ""
    target_term: str = ""
    span_text: str = ""
    expression_text: str = ""
    doctor_request: DoctorSynthesisRequest | None = None
    counterexample: FormalCounterexample | Mapping[str, Any] | None = None
    cegis_verify: Callable[[Mapping[str, Any]], Any] | None = None
    cegis_validate: Callable[[RefinementCandidate, Mapping[str, Any]], Any] | None = None
    cegis_refine: Callable[
        [FormalCounterexample, Mapping[str, Any]], Sequence[Any]
    ] | None = None
    previous_witness_id: str | None = None
    allow_hybrid_residual: bool = True
    behavior_fixed_syntax_debt: bool = False
    syntax_slot_id: str = ""
    language: str = "python"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.roots, DoctorAuthorityRoots):
            raise ProgramRepairSynthesisError("roots must be DoctorAuthorityRoots")
        object.__setattr__(
            self,
            "obligation_refs",
            _ids(self.obligation_refs, "obligation_refs", required=True),
        )
        object.__setattr__(
            self, "target_paths", _paths(self.target_paths, "target_paths")
        )
        if not self.target_paths:
            raise ProgramRepairSynthesisError("target_paths is required")
        if len(self.target_paths) != 1:
            raise ProgramRepairAuthorityError(
                "program repair admits exactly one target path per request",
                reason_code=ProgramRepairReason.SCOPE_WIDENING,
            )
        # Normalize via registry helper when possible.
        normalized_kinds: list[str] = []
        for item in self.operator_kinds or ():
            try:
                normalized_kinds.append(normalize_repair_operator_kind(item).value)
            except (UnknownRepairOperatorError, Exception):
                # Keep raw for fail-closed rejection later.
                normalized_kinds.append(str(item))
        object.__setattr__(self, "operator_kinds", tuple(normalized_kinds))
        for name in (
            "placement_refs",
            "value_refs",
            "proof_refs",
            "review_refs",
            "postcondition_refs",
            "test_refs",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self, "mode", _enum(self.mode, ProgramRepairMode, "mode")
        )
        bounds = self.bounds
        if isinstance(bounds, Mapping):
            bounds = ProgramRepairBounds.from_dict(bounds)
        if not isinstance(bounds, ProgramRepairBounds):
            raise ProgramRepairSynthesisError("bounds must be ProgramRepairBounds")
        object.__setattr__(self, "bounds", bounds)
        if self.equality_theory is not None and not isinstance(
            self.equality_theory, DeclaredEqualityTheory
        ):
            if isinstance(self.equality_theory, Mapping):
                object.__setattr__(
                    self,
                    "equality_theory",
                    DeclaredEqualityTheory.from_dict(self.equality_theory),
                )
            else:
                raise ProgramRepairSynthesisError(
                    "equality_theory must be DeclaredEqualityTheory"
                )
        object.__setattr__(
            self, "source_term", _optional_text(self.source_term, "source_term", limit=MAX_SPAN_BYTES)
        )
        object.__setattr__(
            self, "target_term", _optional_text(self.target_term, "target_term", limit=MAX_SPAN_BYTES)
        )
        object.__setattr__(
            self, "span_text", _optional_text(self.span_text, "span_text", limit=MAX_SPAN_BYTES)
        )
        object.__setattr__(
            self,
            "expression_text",
            _optional_text(self.expression_text, "expression_text", limit=MAX_SPAN_BYTES),
        )
        if self.doctor_request is not None and not isinstance(
            self.doctor_request, DoctorSynthesisRequest
        ):
            raise ProgramRepairSynthesisError(
                "doctor_request must be DoctorSynthesisRequest"
            )
        object.__setattr__(
            self,
            "allow_hybrid_residual",
            _bool(self.allow_hybrid_residual, "allow_hybrid_residual"),
        )
        object.__setattr__(
            self,
            "behavior_fixed_syntax_debt",
            _bool(self.behavior_fixed_syntax_debt, "behavior_fixed_syntax_debt"),
        )
        object.__setattr__(
            self, "syntax_slot_id", _optional_text(self.syntax_slot_id, "syntax_slot_id")
        )
        object.__setattr__(self, "language", _text(self.language, "language"))
        if not isinstance(self.metadata, Mapping):
            raise ProgramRepairSynthesisError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        # Reject forbidden claims in metadata.
        forbidden = _walk_forbidden_claims(dict(self.metadata))
        if forbidden:
            raise ProgramRepairAuthorityError(
                f"request metadata contains forbidden claims: {forbidden[0]}",
                reason_code=ProgramRepairReason.AUTHORITY_CLAIM,
            )


@dataclass(frozen=True)
class ProgramRepairCandidate(CanonicalContract):
    """One proposal-only repair candidate produced by bounded search."""

    SCHEMA: ClassVar[str] = PROGRAM_REPAIR_CANDIDATE_SCHEMA

    candidate_id: str
    operator_kind: str
    operator_id: str
    path: str
    mode: ProgramRepairMode
    proposal_only: bool = True
    overlay_cid: str = ""
    patch_cid: str = ""
    replacement: str = ""
    before_hash: str = ""
    after_hash: str = ""
    obligation_refs: tuple[str, ...] = ()
    postcondition_refs: tuple[str, ...] = ()
    equality_receipt: EqualityRewriteReceipt | None = None
    doctor_receipt_id: str = ""
    reason_codes: tuple[str, ...] = ()
    write_authority: bool = False
    semantic_authority: bool = False
    grants_proof_authority: bool = False
    llm_invocation_count: int = 0
    model_provider_call_count: int = 0
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(
            self, "operator_kind", _optional_text(self.operator_kind, "operator_kind")
        )
        object.__setattr__(
            self, "operator_id", _optional_text(self.operator_id, "operator_id")
        )
        object.__setattr__(self, "path", _path(self.path) if self.path else "")
        object.__setattr__(
            self, "mode", _enum(self.mode, ProgramRepairMode, "mode")
        )
        if self.proposal_only is not True:
            raise ProgramRepairAuthorityError(
                "candidates must remain proposal-only",
                reason_code=ProgramRepairReason.PROPOSAL_ONLY,
            )
        object.__setattr__(self, "proposal_only", True)
        for name in (
            "overlay_cid",
            "patch_cid",
            "before_hash",
            "after_hash",
            "doctor_receipt_id",
            "producer_id",
        ):
            object.__setattr__(
                self, name, _optional_text(getattr(self, name), name)
            )
        replacement = self.replacement if isinstance(self.replacement, str) else ""
        if len(replacement.encode("utf-8")) > MAX_SPAN_BYTES:
            raise ProgramRepairBoundsError(
                "replacement exceeds span bound",
                reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
            )
        object.__setattr__(self, "replacement", replacement)
        object.__setattr__(
            self, "obligation_refs", _ids(self.obligation_refs, "obligation_refs")
        )
        object.__setattr__(
            self,
            "postcondition_refs",
            _ids(self.postcondition_refs, "postcondition_refs"),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        if self.equality_receipt is not None and not isinstance(
            self.equality_receipt, EqualityRewriteReceipt
        ):
            raise ProgramRepairSynthesisError(
                "equality_receipt must be EqualityRewriteReceipt"
            )
        for name in (
            "write_authority",
            "semantic_authority",
            "grants_proof_authority",
        ):
            if getattr(self, name) is not False:
                raise ProgramRepairAuthorityError(
                    f"candidate cannot claim {name}",
                    reason_code=ProgramRepairReason.AUTHORITY_CLAIM,
                )
            object.__setattr__(self, name, False)
        if self.llm_invocation_count != 0 or self.model_provider_call_count != 0:
            raise ProgramRepairAuthorityError(
                "candidate must report zero model calls",
                reason_code=ProgramRepairReason.PROVIDER_OR_MODEL_CALL,
            )
        object.__setattr__(self, "llm_invocation_count", 0)
        object.__setattr__(self, "model_provider_call_count", 0)
        object.__setattr__(
            self, "producer_id", _text(self.producer_id or PRODUCER_ID, "producer_id")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "candidate_id": self.candidate_id,
            "operator_kind": self.operator_kind,
            "operator_id": self.operator_id,
            "path": self.path,
            "mode": self.mode.value,
            "proposal_only": True,
            "overlay_cid": self.overlay_cid,
            "patch_cid": self.patch_cid,
            "replacement": self.replacement,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "obligation_refs": list(self.obligation_refs),
            "postcondition_refs": list(self.postcondition_refs),
            "equality_receipt": (
                self.equality_receipt.to_dict()
                if self.equality_receipt is not None
                else None
            ),
            "doctor_receipt_id": self.doctor_receipt_id,
            "reason_codes": list(self.reason_codes),
            "write_authority": False,
            "semantic_authority": False,
            "grants_proof_authority": False,
            "llm_invocation_count": 0,
            "model_provider_call_count": 0,
            "producer_id": self.producer_id,
        }


@dataclass(frozen=True)
class ProgramRepairReceipt(CanonicalContract):
    """Auditable receipt for one bounded program-repair synthesis run."""

    SCHEMA: ClassVar[str] = PROGRAM_REPAIR_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_REPAIR_SYNTHESIZER_INTERFACE

    disposition: ProgramRepairDisposition
    reason_codes: tuple[str, ...]
    roots: DoctorAuthorityRoots
    mode: ProgramRepairMode
    candidates: tuple[ProgramRepairCandidate, ...] = ()
    selected_candidate: ProgramRepairCandidate | None = None
    doctor_receipt: DoctorSynthesisReceipt | None = None
    equality_receipt: EqualityRewriteReceipt | None = None
    cegis_result: CegisLoopResult | None = None
    residual_packet: ResidualHybridPacket | None = None
    hybrid_admission: ResidualHybridAdmission | None = None
    operator_lookup_ids: tuple[str, ...] = ()
    search_states: int = 0
    llm_invocation_count: int = 0
    model_provider_call_count: int = 0
    proposal_only: bool = True
    write_authority: bool = False
    semantic_authority: bool = False
    write_performed: bool = False
    provider_invoked: bool = False
    deterministic_zero_model_calls: bool = True
    producer_id: str = PRODUCER_ID
    bounds: ProgramRepairBounds = field(default_factory=ProgramRepairBounds)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ProgramRepairDisposition, "disposition"),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes", required=True)
        )
        if not isinstance(self.roots, DoctorAuthorityRoots):
            raise ProgramRepairSynthesisError("roots must be DoctorAuthorityRoots")
        object.__setattr__(
            self, "mode", _enum(self.mode, ProgramRepairMode, "mode")
        )
        candidates = tuple(self.candidates or ())
        for item in candidates:
            if not isinstance(item, ProgramRepairCandidate):
                raise ProgramRepairSynthesisError(
                    "candidates must be ProgramRepairCandidate"
                )
        object.__setattr__(self, "candidates", candidates)
        if self.selected_candidate is not None and not isinstance(
            self.selected_candidate, ProgramRepairCandidate
        ):
            raise ProgramRepairSynthesisError(
                "selected_candidate must be ProgramRepairCandidate"
            )
        if self.doctor_receipt is not None and not isinstance(
            self.doctor_receipt, DoctorSynthesisReceipt
        ):
            raise ProgramRepairSynthesisError(
                "doctor_receipt must be DoctorSynthesisReceipt"
            )
        if self.equality_receipt is not None and not isinstance(
            self.equality_receipt, EqualityRewriteReceipt
        ):
            raise ProgramRepairSynthesisError(
                "equality_receipt must be EqualityRewriteReceipt"
            )
        if self.cegis_result is not None and not isinstance(
            self.cegis_result, CegisLoopResult
        ):
            raise ProgramRepairSynthesisError("cegis_result must be CegisLoopResult")
        if self.residual_packet is not None and not isinstance(
            self.residual_packet, ResidualHybridPacket
        ):
            raise ProgramRepairSynthesisError(
                "residual_packet must be ResidualHybridPacket"
            )
        if self.hybrid_admission is not None and not isinstance(
            self.hybrid_admission, ResidualHybridAdmission
        ):
            raise ProgramRepairSynthesisError(
                "hybrid_admission must be ResidualHybridAdmission"
            )
        object.__setattr__(
            self,
            "operator_lookup_ids",
            _ids(self.operator_lookup_ids, "operator_lookup_ids"),
        )
        object.__setattr__(
            self, "search_states", _nonneg_int(self.search_states, "search_states")
        )
        object.__setattr__(
            self,
            "llm_invocation_count",
            _nonneg_int(self.llm_invocation_count, "llm_invocation_count"),
        )
        object.__setattr__(
            self,
            "model_provider_call_count",
            _nonneg_int(
                self.model_provider_call_count, "model_provider_call_count"
            ),
        )
        # Authority / model invariants.
        if self.proposal_only is not True:
            raise ProgramRepairAuthorityError(
                "receipts must remain proposal-only",
                reason_code=ProgramRepairReason.PROPOSAL_ONLY,
            )
        for name in (
            "write_authority",
            "semantic_authority",
            "write_performed",
            "provider_invoked",
        ):
            if getattr(self, name) is not False:
                raise ProgramRepairAuthorityError(
                    f"receipt cannot claim {name}",
                    reason_code=ProgramRepairReason.AUTHORITY_CLAIM,
                )
            object.__setattr__(self, name, False)
        object.__setattr__(self, "proposal_only", True)
        # Deterministic modes prove zero model calls.
        if self.mode is not ProgramRepairMode.HYBRID_RESIDUAL:
            if self.llm_invocation_count != 0 or self.model_provider_call_count != 0:
                raise ProgramRepairAuthorityError(
                    "deterministic modes must prove zero model calls",
                    reason_code=ProgramRepairReason.PROVIDER_OR_MODEL_CALL,
                )
            if self.deterministic_zero_model_calls is not True:
                raise ProgramRepairAuthorityError(
                    "deterministic_zero_model_calls must be true",
                    reason_code=ProgramRepairReason.PROVIDER_OR_MODEL_CALL,
                )
            object.__setattr__(self, "llm_invocation_count", 0)
            object.__setattr__(self, "model_provider_call_count", 0)
            object.__setattr__(self, "deterministic_zero_model_calls", True)
        if not isinstance(self.bounds, ProgramRepairBounds):
            if isinstance(self.bounds, Mapping):
                object.__setattr__(
                    self, "bounds", ProgramRepairBounds.from_dict(self.bounds)
                )
            else:
                raise ProgramRepairSynthesisError("bounds must be ProgramRepairBounds")
        object.__setattr__(
            self, "producer_id", _text(self.producer_id or PRODUCER_ID, "producer_id")
        )

    @property
    def admitted(self) -> bool:
        return (
            self.disposition is ProgramRepairDisposition.SUPPORTED
            and self.selected_candidate is not None
        )

    @property
    def has_residual_debt(self) -> bool:
        return self.disposition is ProgramRepairDisposition.RESIDUAL_DEBT

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "interface": PROGRAM_REPAIR_SYNTHESIZER_INTERFACE,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "roots": self.roots.to_dict(),
            "mode": self.mode.value,
            "candidates": [item.to_dict() for item in self.candidates],
            "selected_candidate": (
                self.selected_candidate.to_dict()
                if self.selected_candidate is not None
                else None
            ),
            "doctor_receipt": (
                self.doctor_receipt.to_dict()
                if self.doctor_receipt is not None
                else None
            ),
            "equality_receipt": (
                self.equality_receipt.to_dict()
                if self.equality_receipt is not None
                else None
            ),
            "cegis_result": (
                self.cegis_result.to_dict() if self.cegis_result is not None else None
            ),
            "residual_packet": (
                self.residual_packet.to_dict()
                if self.residual_packet is not None
                else None
            ),
            "hybrid_admission": (
                self.hybrid_admission.to_dict()
                if self.hybrid_admission is not None
                else None
            ),
            "operator_lookup_ids": list(self.operator_lookup_ids),
            "search_states": self.search_states,
            "llm_invocation_count": self.llm_invocation_count,
            "model_provider_call_count": self.model_provider_call_count,
            "proposal_only": True,
            "write_authority": False,
            "semantic_authority": False,
            "write_performed": False,
            "provider_invoked": False,
            "deterministic_zero_model_calls": self.deterministic_zero_model_calls,
            "producer_id": self.producer_id,
            "bounds": self.bounds.to_dict(),
            "cegis_interface": COUNTEREXAMPLE_GUIDED_PROOF_DEVELOPMENT_INTERFACE,
            "doctor_interface": DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE,
            "registry_interface": REPAIR_OPERATOR_REGISTRY_INTERFACE,
        }


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class ProgramRepairSynthesizer:
    """Bounded deterministic synthesis / CEGIS with residual-only hybrid path.

    Search is restricted to reviewed operators/grammars under exact obligations,
    bounds, and roots. CEGIS independently validates counterexamples and
    terminates on fixed budgets. E-graph rewrites prove equivalence only under
    a declared theory. Every candidate is proposal-only.
    """

    INTERFACE: ClassVar[str] = PROGRAM_REPAIR_SYNTHESIZER_INTERFACE
    VERSION: ClassVar[str] = PROGRAM_REPAIR_SYNTHESIZER_VERSION

    def __init__(
        self,
        *,
        operator_registry: RepairOperatorRegistry | None = None,
        doctor_synthesizer: DeterministicDoctorSynthesizer | None = None,
        hybrid_service: ResidualHybridRepairService | None = None,
        roots: DoctorAuthorityRoots | None = None,
    ) -> None:
        self._registry = operator_registry or build_default_repair_operator_registry(
            roots
        )
        self._doctor = doctor_synthesizer
        self._hybrid = hybrid_service or ResidualHybridRepairService()
        self._roots = roots

    @property
    def registry(self) -> RepairOperatorRegistry:
        return self._registry

    def synthesize(self, request: ProgramRepairRequest) -> ProgramRepairReceipt:
        """Run one bounded program-repair synthesis under the request mode."""

        if not isinstance(request, ProgramRepairRequest):
            raise ProgramRepairSynthesisError(
                "request must be ProgramRepairRequest",
                reason_code=ProgramRepairReason.MALFORMED_INPUT,
            )
        _assert_no_provider_markers(
            request.span_text,
            request.expression_text,
            request.source_term,
            request.target_term,
            *[str(item) for item in request.operator_kinds],
        )
        mode = request.mode
        if mode is ProgramRepairMode.EQUALITY_REWRITE:
            return self._synthesize_equality(request)
        if mode is ProgramRepairMode.CEGIS:
            return self._synthesize_cegis(request)
        if mode is ProgramRepairMode.HYBRID_RESIDUAL:
            return self._synthesize_hybrid_only(request)
        # DETERMINISTIC and ENUMERATIVE share the operator-grammar search path.
        return self._synthesize_deterministic(request)

    # -- deterministic / enumerative --------------------------------------

    def _synthesize_deterministic(
        self, request: ProgramRepairRequest
    ) -> ProgramRepairReceipt:
        lookup_ids: list[str] = []
        search_states = 0
        candidates: list[ProgramRepairCandidate] = []
        doctor_receipt: DoctorSynthesisReceipt | None = None

        # Prefer a supplied doctor request (proof-admitted analytical path).
        if request.doctor_request is not None:
            doctor = self._resolve_doctor(request.roots)
            doctor_receipt = doctor.synthesize(request.doctor_request)
            search_states += 1
            if doctor_receipt.admitted and doctor_receipt.overlay is not None:
                candidate = self._candidate_from_doctor(
                    doctor_receipt,
                    request=request,
                    mode=request.mode,
                )
                candidates.append(candidate)
                return ProgramRepairReceipt(
                    disposition=ProgramRepairDisposition.SUPPORTED,
                    reason_codes=(
                        ProgramRepairReason.RENDERED.value,
                        ProgramRepairReason.PROPOSAL_ONLY.value,
                        ProgramRepairReason.ZERO_MODEL_CALLS.value,
                    ),
                    roots=request.roots,
                    mode=request.mode,
                    candidates=tuple(candidates),
                    selected_candidate=candidate,
                    doctor_receipt=doctor_receipt,
                    operator_lookup_ids=tuple(lookup_ids),
                    search_states=search_states,
                    bounds=request.bounds,
                )
            # Fall through to grammar search / residual if doctor abstained.

        kinds = request.operator_kinds or tuple(
            spec.kind.value for spec in self._registry.operators
        )
        kinds = kinds[: request.bounds.max_enumerative_candidates]
        for kind in kinds:
            if search_states >= request.bounds.max_search_states:
                break
            search_states += 1
            try:
                lookup = self._lookup_operator(request, kind)
            except (UnknownRepairOperatorError, ProgramRepairSynthesisError):
                continue
            lookup_ids.append(lookup.content_id if hasattr(lookup, "content_id") else lookup.request_id)
            if not lookup.proposal_eligible:
                continue
            if lookup.proposal_only is not True:
                continue
            # Equality rewrites require a declared theory.
            try:
                spec = self._registry.get(kind)
            except UnknownRepairOperatorError:
                continue
            if spec.reviewed_hook is ReviewedRepairHook.EQUALITY_REWRITE:
                if request.equality_theory is None:
                    continue
                eq = prove_equality_under_theory(
                    request.equality_theory,
                    request.source_term or request.span_text,
                    request.target_term or request.expression_text,
                    max_depth=request.bounds.max_rewrite_depth,
                    max_nodes=request.bounds.max_egraph_nodes,
                )
                search_states += eq.rewrite_depth
                if not eq.proved:
                    continue
                candidate = ProgramRepairCandidate(
                    candidate_id=f"candidate:eq:{eq.content_id[-16:]}",
                    operator_kind=spec.kind.value,
                    operator_id=spec.operator_id,
                    path=request.target_paths[0],
                    mode=ProgramRepairMode.EQUALITY_REWRITE,
                    replacement=request.target_term or request.expression_text,
                    obligation_refs=request.obligation_refs,
                    postcondition_refs=request.postcondition_refs
                    or spec.postcondition_refs,
                    equality_receipt=eq,
                    reason_codes=(
                        ProgramRepairReason.EQUALITY_PROVED.value,
                        ProgramRepairReason.PROPOSAL_ONLY.value,
                    ),
                )
                candidates.append(candidate)
                if request.mode is ProgramRepairMode.DETERMINISTIC:
                    break
                if len(candidates) >= request.bounds.max_enumerative_candidates:
                    break
                continue

            # Analytical operators without a doctor request cannot be rendered
            # here (proof/path gates live in DeterministicDoctorSynthesizer).
            # Record a proposal-only nomination candidate for portfolio use.
            candidate = ProgramRepairCandidate(
                candidate_id=f"candidate:op:{lookup.content_id[-16:]}",
                operator_kind=lookup.operator_kind,
                operator_id=lookup.operator_id,
                path=request.target_paths[0],
                mode=request.mode,
                obligation_refs=request.obligation_refs,
                postcondition_refs=request.postcondition_refs
                or tuple(spec.postcondition_refs),
                reason_codes=(
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    "operator_nominated_not_rendered",
                ),
            )
            candidates.append(candidate)
            if request.mode is ProgramRepairMode.DETERMINISTIC:
                # Nomination-only does not count as SUPPORTED without overlay.
                break
            if len(candidates) >= request.bounds.max_enumerative_candidates:
                break

        if candidates and any(
            item.overlay_cid or item.patch_cid or item.equality_receipt is not None
            for item in candidates
        ):
            selected = next(
                item
                for item in candidates
                if item.overlay_cid or item.patch_cid or item.equality_receipt is not None
            )
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.SUPPORTED,
                reason_codes=(
                    ProgramRepairReason.RENDERED.value
                    if selected.overlay_cid or selected.patch_cid
                    else ProgramRepairReason.EQUALITY_PROVED.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    ProgramRepairReason.ZERO_MODEL_CALLS.value,
                ),
                roots=request.roots,
                mode=request.mode,
                candidates=tuple(candidates),
                selected_candidate=selected,
                doctor_receipt=doctor_receipt,
                equality_receipt=selected.equality_receipt,
                operator_lookup_ids=tuple(lookup_ids),
                search_states=search_states,
                bounds=request.bounds,
            )

        # Residual debt path.
        if request.allow_hybrid_residual and (
            request.behavior_fixed_syntax_debt
            or (doctor_receipt is not None and not doctor_receipt.admitted)
        ):
            packet = self._build_residual_packet(request)
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.RESIDUAL_DEBT,
                reason_codes=(
                    ProgramRepairReason.RESIDUAL_PACKET_EMITTED.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    ProgramRepairReason.ZERO_MODEL_CALLS.value,
                ),
                roots=request.roots,
                mode=request.mode,
                candidates=tuple(candidates),
                doctor_receipt=doctor_receipt,
                residual_packet=packet,
                operator_lookup_ids=tuple(lookup_ids),
                search_states=search_states,
                bounds=request.bounds,
            )

        reasons = [
            ProgramRepairReason.NO_ADMISSIBLE_OPERATOR.value
            if not candidates
            else ProgramRepairReason.SEARCH_EMPTY.value,
            ProgramRepairReason.PROPOSAL_ONLY.value,
            ProgramRepairReason.ZERO_MODEL_CALLS.value,
            ProgramRepairReason.NO_PARTIAL_OVERLAY.value,
        ]
        return ProgramRepairReceipt(
            disposition=ProgramRepairDisposition.ABSTAIN,
            reason_codes=tuple(reasons),
            roots=request.roots,
            mode=request.mode,
            candidates=tuple(candidates),
            doctor_receipt=doctor_receipt,
            operator_lookup_ids=tuple(lookup_ids),
            search_states=search_states,
            bounds=request.bounds,
        )

    # -- equality rewrite -------------------------------------------------

    def _synthesize_equality(
        self, request: ProgramRepairRequest
    ) -> ProgramRepairReceipt:
        if request.equality_theory is None:
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.ABSTAIN,
                reason_codes=(
                    ProgramRepairReason.UNDECLARED_THEORY.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    ProgramRepairReason.ZERO_MODEL_CALLS.value,
                ),
                roots=request.roots,
                mode=ProgramRepairMode.EQUALITY_REWRITE,
                bounds=request.bounds,
            )
        # Root binding when theory carries roots.
        theory = request.equality_theory
        if theory.repository_id and theory.repository_id != request.roots.repository_id:
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.ABSTAIN,
                reason_codes=(
                    ProgramRepairReason.ROOT_MISMATCH.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    ProgramRepairReason.ZERO_MODEL_CALLS.value,
                ),
                roots=request.roots,
                mode=ProgramRepairMode.EQUALITY_REWRITE,
                bounds=request.bounds,
            )
        if theory.tree_id and theory.tree_id != request.roots.tree_id:
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.ABSTAIN,
                reason_codes=(
                    ProgramRepairReason.ROOT_MISMATCH.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    ProgramRepairReason.ZERO_MODEL_CALLS.value,
                ),
                roots=request.roots,
                mode=ProgramRepairMode.EQUALITY_REWRITE,
                bounds=request.bounds,
            )
        source = request.source_term or request.span_text
        target = request.target_term or request.expression_text
        if not source or not target:
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.ABSTAIN,
                reason_codes=(
                    ProgramRepairReason.MALFORMED_INPUT.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    ProgramRepairReason.ZERO_MODEL_CALLS.value,
                ),
                roots=request.roots,
                mode=ProgramRepairMode.EQUALITY_REWRITE,
                bounds=request.bounds,
            )
        receipt = prove_equality_under_theory(
            theory,
            source,
            target,
            max_depth=request.bounds.max_rewrite_depth,
            max_nodes=request.bounds.max_egraph_nodes,
        )
        if not receipt.proved:
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.ABSTAIN,
                reason_codes=(
                    receipt.reason_code or ProgramRepairReason.EQUALITY_UNPROVED.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    ProgramRepairReason.ZERO_MODEL_CALLS.value,
                ),
                roots=request.roots,
                mode=ProgramRepairMode.EQUALITY_REWRITE,
                equality_receipt=receipt,
                search_states=receipt.rewrite_depth,
                bounds=request.bounds,
            )
        candidate = ProgramRepairCandidate(
            candidate_id=f"candidate:eq:{receipt.content_id[-16:]}",
            operator_kind=RepairOperatorKind.EQUALITY_REWRITE.value,
            operator_id=f"repair-operator:{RepairOperatorKind.EQUALITY_REWRITE.value}@2",
            path=request.target_paths[0],
            mode=ProgramRepairMode.EQUALITY_REWRITE,
            replacement=target,
            before_hash=_sha256_text(source),
            after_hash=_sha256_text(target),
            obligation_refs=request.obligation_refs,
            postcondition_refs=request.postcondition_refs
            or ("post:equivalent_under_declared_theory",),
            equality_receipt=receipt,
            reason_codes=(
                ProgramRepairReason.EQUALITY_PROVED.value,
                ProgramRepairReason.PROPOSAL_ONLY.value,
            ),
        )
        return ProgramRepairReceipt(
            disposition=ProgramRepairDisposition.SUPPORTED,
            reason_codes=(
                ProgramRepairReason.EQUALITY_PROVED.value,
                ProgramRepairReason.PROPOSAL_ONLY.value,
                ProgramRepairReason.ZERO_MODEL_CALLS.value,
            ),
            roots=request.roots,
            mode=ProgramRepairMode.EQUALITY_REWRITE,
            candidates=(candidate,),
            selected_candidate=candidate,
            equality_receipt=receipt,
            search_states=receipt.rewrite_depth,
            bounds=request.bounds,
        )

    # -- CEGIS ------------------------------------------------------------

    def _synthesize_cegis(self, request: ProgramRepairRequest) -> ProgramRepairReceipt:
        if request.counterexample is None:
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.ABSTAIN,
                reason_codes=(
                    ProgramRepairReason.MALFORMED_INPUT.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    ProgramRepairReason.ZERO_MODEL_CALLS.value,
                ),
                roots=request.roots,
                mode=ProgramRepairMode.CEGIS,
                bounds=request.bounds,
            )
        budget = request.bounds.to_cegis_budget(
            finite_bounds={
                "repository_id": request.roots.repository_id,
                "tree_id": request.roots.tree_id,
                "max_search_states": request.bounds.max_search_states,
            }
        )

        # Independent validation defaults to rejecting candidates that do not
        # address the witness; callers may inject a custom validator.
        validate = request.cegis_validate
        if validate is None:

            def validate(candidate: RefinementCandidate, context: Mapping[str, Any]):
                del context
                if not candidate.addresses_witness:
                    return (
                        CandidateValidationStatus.INVALID,
                        ProgramRepairReason.CEGIS_INDEPENDENT_REJECT.value,
                    )
                if not candidate.repaired_tree_id or not candidate.goal_id:
                    return (
                        CandidateValidationStatus.INVALID,
                        ProgramRepairReason.CEGIS_INDEPENDENT_REJECT.value,
                    )
                return CandidateValidationStatus.VALID, "independent_validation_passed"

        refine = request.cegis_refine
        if refine is None and request.operator_kinds:

            def refine(witness: FormalCounterexample, context: Mapping[str, Any]):
                del witness
                out: list[RefinementCandidate] = []
                for index, kind in enumerate(
                    request.operator_kinds[: budget.max_candidates_per_iteration]
                ):
                    out.append(
                        RefinementCandidate(
                            candidate_id=f"candidate:cegis:{kind}:{index}",
                            kind=CandidateKind.REPAIR,
                            goal_id=str(
                                context.get("goal_id")
                                or request.obligation_refs[0]
                            ),
                            repaired_tree_id=str(
                                context.get("repository_tree_id")
                                or request.roots.tree_id
                            ),
                            repaired_plan_id=str(
                                context.get("repaired_plan_id") or "plan:cegis"
                            ),
                            statement=f"Reviewed operator {kind}",
                            addresses_witness=True,
                            parameters={
                                "operator_kind": kind,
                                "obligation_refs": list(request.obligation_refs),
                            },
                        )
                    )
                return tuple(out)

        try:
            result = run_counterexample_guided_loop(
                request.counterexample,
                refine=refine,
                validate=validate,
                verify=request.cegis_verify,
                budget=budget,
                repository_tree_id=request.roots.tree_id,
                goal_id=request.obligation_refs[0] if request.obligation_refs else "",
                policy_id=request.roots.policy_id,
                previous_witness_id=request.previous_witness_id,
                context={
                    "obligation_refs": list(request.obligation_refs),
                    "target_paths": list(request.target_paths),
                    "operator_kinds": list(request.operator_kinds),
                },
            )
        except (CegisValidationError, ContractValidationError) as exc:
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.ABSTAIN,
                reason_codes=(
                    ProgramRepairReason.MALFORMED_INPUT.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    ProgramRepairReason.ZERO_MODEL_CALLS.value,
                    f"cegis_error:{type(exc).__name__}",
                ),
                roots=request.roots,
                mode=ProgramRepairMode.CEGIS,
                bounds=request.bounds,
            )

        candidates: list[ProgramRepairCandidate] = []
        selected: ProgramRepairCandidate | None = None
        if result.selected_candidate is not None:
            sc = result.selected_candidate
            selected = ProgramRepairCandidate(
                candidate_id=sc.candidate_id,
                operator_kind=str(
                    (sc.parameters or {}).get("operator_kind") or sc.kind.value
                ),
                operator_id="",
                path=request.target_paths[0],
                mode=ProgramRepairMode.CEGIS,
                obligation_refs=request.obligation_refs,
                postcondition_refs=request.postcondition_refs,
                reason_codes=(
                    ProgramRepairReason.CEGIS_CLOSED.value
                    if result.closed
                    else ProgramRepairReason.CEGIS_OPEN.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                ),
            )
            candidates.append(selected)

        if result.closed and selected is not None:
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.SUPPORTED,
                reason_codes=(
                    ProgramRepairReason.CEGIS_CLOSED.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    ProgramRepairReason.ZERO_MODEL_CALLS.value,
                ),
                roots=request.roots,
                mode=ProgramRepairMode.CEGIS,
                candidates=tuple(candidates),
                selected_candidate=selected,
                cegis_result=result,
                search_states=result.iteration_count,
                bounds=request.bounds,
            )

        stop = result.stop_reason
        if stop in {
            CegisStopReason.RETRY_BUDGET_EXHAUSTED,
            CegisStopReason.REFINEMENT_DEPTH_EXHAUSTED,
            CegisStopReason.CANDIDATE_BUDGET_EXHAUSTED,
        }:
            disposition = ProgramRepairDisposition.BUDGET_EXHAUSTED
        else:
            disposition = ProgramRepairDisposition.ABSTAIN

        residual = None
        if request.allow_hybrid_residual and request.behavior_fixed_syntax_debt:
            residual = self._build_residual_packet(request)
            disposition = ProgramRepairDisposition.RESIDUAL_DEBT

        return ProgramRepairReceipt(
            disposition=disposition,
            reason_codes=(
                ProgramRepairReason.CEGIS_OPEN.value,
                ProgramRepairReason.PROPOSAL_ONLY.value,
                ProgramRepairReason.ZERO_MODEL_CALLS.value,
                str(getattr(stop, "value", stop)),
            ),
            roots=request.roots,
            mode=ProgramRepairMode.CEGIS,
            candidates=tuple(candidates),
            cegis_result=result,
            residual_packet=residual,
            search_states=result.iteration_count,
            bounds=request.bounds,
        )

    # -- hybrid residual only ---------------------------------------------

    def _synthesize_hybrid_only(
        self, request: ProgramRepairRequest
    ) -> ProgramRepairReceipt:
        if not request.behavior_fixed_syntax_debt:
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.ABSTAIN,
                reason_codes=(
                    ProgramRepairReason.MEANING_CHANGE.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    ProgramRepairReason.ZERO_MODEL_CALLS.value,
                ),
                roots=request.roots,
                mode=ProgramRepairMode.HYBRID_RESIDUAL,
                bounds=request.bounds,
            )
        packet = self._build_residual_packet(request)
        # Deterministic synthesizer never calls the model; packet only.
        return ProgramRepairReceipt(
            disposition=ProgramRepairDisposition.RESIDUAL_DEBT,
            reason_codes=(
                ProgramRepairReason.RESIDUAL_PACKET_EMITTED.value,
                ProgramRepairReason.PROPOSAL_ONLY.value,
                ProgramRepairReason.ZERO_MODEL_CALLS.value,
            ),
            roots=request.roots,
            mode=ProgramRepairMode.HYBRID_RESIDUAL,
            residual_packet=packet,
            # Hybrid residual mode on the synthesizer still proves zero model
            # calls for the deterministic orchestrator itself.
            llm_invocation_count=0,
            model_provider_call_count=0,
            deterministic_zero_model_calls=True,
            bounds=request.bounds,
        )

    # -- helpers ----------------------------------------------------------

    def _resolve_doctor(
        self, roots: DoctorAuthorityRoots
    ) -> DeterministicDoctorSynthesizer:
        if self._doctor is not None:
            return self._doctor
        return create_deterministic_doctor_synthesizer(roots)

    def _lookup_operator(
        self, request: ProgramRepairRequest, kind: str
    ) -> RepairOperatorLookupResult:
        try:
            norm = normalize_repair_operator_kind(kind)
        except UnknownRepairOperatorError as exc:
            raise ProgramRepairSynthesisError(
                f"operator not reviewed: {kind}",
                reason_code=ProgramRepairReason.OPERATOR_NOT_REVIEWED,
            ) from exc
        spec = self._registry.get(norm)
        lookup_request = RepairOperatorLookupRequest(
            operator_kind=norm.value,
            repository_id=request.roots.repository_id,
            tree_id=request.roots.tree_id,
            target_paths=request.target_paths,
            placement_refs=request.placement_refs or ("placement:exact",),
            value_refs=request.value_refs
            or (("value:unique",) if spec.requires_value else ()),
            capability_refs=spec.capability_refs,
            proof_refs=request.proof_refs or ("proof:nomination",),
            review_refs=request.review_refs
            or (spec.review_requirement_refs if spec.review_requirement_refs else ()),
            language=request.language,
        )
        return self._registry.resolve(lookup_request)

    def _candidate_from_doctor(
        self,
        doctor_receipt: DoctorSynthesisReceipt,
        *,
        request: ProgramRepairRequest,
        mode: ProgramRepairMode,
    ) -> ProgramRepairCandidate:
        overlay = doctor_receipt.overlay
        assert overlay is not None
        return ProgramRepairCandidate(
            candidate_id=f"candidate:doc:{overlay.overlay_id[-16:]}",
            operator_kind=doctor_receipt.operator_kind or overlay.operator_kind,
            operator_id=doctor_receipt.operator_id or overlay.operator_id,
            path=overlay.path,
            mode=mode,
            overlay_cid=overlay.overlay_id,
            patch_cid=overlay.patch_cid or doctor_receipt.patch_cid,
            replacement=overlay.replacement,
            before_hash=overlay.before_hash,
            after_hash=overlay.after_hash,
            obligation_refs=request.obligation_refs or overlay.obligation_refs,
            postcondition_refs=request.postcondition_refs or overlay.postcondition_refs,
            doctor_receipt_id=getattr(doctor_receipt, "content_id", "") or "",
            reason_codes=(
                ProgramRepairReason.RENDERED.value,
                ProgramRepairReason.PROPOSAL_ONLY.value,
                ProgramRepairReason.ZERO_MODEL_CALLS.value,
            ),
        )

    def _build_residual_packet(
        self, request: ProgramRepairRequest
    ) -> ResidualHybridPacket:
        path = request.target_paths[0]
        semantics_source = {
            "path": path,
            "obligation_refs": list(request.obligation_refs),
            "postcondition_refs": list(request.postcondition_refs),
            "test_refs": list(request.test_refs),
            "span_text": request.span_text,
            "expression_text": request.expression_text,
            "repository_id": request.roots.repository_id,
            "tree_id": request.roots.tree_id,
        }
        packet_id = content_identity(
            {"kind": "residual-hybrid-packet", **semantics_source}
        )
        span_start = 0
        span_end = len(request.span_text) if request.span_text else 0
        if request.doctor_request is not None:
            site = request.doctor_request.proposal.edit_site
            span_start = int(getattr(site, "span_start", 0) or 0)
            span_end = int(getattr(site, "span_end", span_end) or span_end)
            path = str(getattr(site, "path", path) or path)
        return ResidualHybridPacket(
            packet_id=f"residual:{packet_id[-24:]}",
            target_path=path,
            span_start=span_start,
            span_end=span_end,
            semantics_digest=content_identity(semantics_source),
            postcondition_refs=request.postcondition_refs,
            test_refs=request.test_refs,
            obligation_refs=request.obligation_refs,
            repository_id=request.roots.repository_id,
            tree_id=request.roots.tree_id,
            behavior_fixed=True,
            allowed_paths=(path,),
            syntax_slot_id=request.syntax_slot_id or f"syntax:{path}",
            reason_codes=(ProgramRepairReason.RESIDUAL_PACKET_EMITTED.value,),
        )


def create_program_repair_synthesizer(
    roots: DoctorAuthorityRoots | None = None,
    *,
    operator_registry: RepairOperatorRegistry | None = None,
    doctor_synthesizer: DeterministicDoctorSynthesizer | None = None,
    hybrid_service: ResidualHybridRepairService | None = None,
) -> ProgramRepairSynthesizer:
    """Factory for a root-bound program-repair synthesizer."""

    return ProgramRepairSynthesizer(
        operator_registry=operator_registry,
        doctor_synthesizer=doctor_synthesizer,
        hybrid_service=hybrid_service,
        roots=roots,
    )


def synthesize_program_repair(
    request: ProgramRepairRequest,
    *,
    synthesizer: ProgramRepairSynthesizer | None = None,
) -> ProgramRepairReceipt:
    """Module-level convenience wrapper around :class:`ProgramRepairSynthesizer`."""

    synth = synthesizer or create_program_repair_synthesizer(request.roots)
    return synth.synthesize(request)


__all__ = (
    "CONTRACT_VERSION",
    "EQUALITY_REWRITE_RECEIPT_SCHEMA",
    "EQUALITY_THEORY_SCHEMA",
    "HYBRID_USAGE_RECEIPT_SCHEMA",
    "PROGRAM_REPAIR_BOUNDS_SCHEMA",
    "PROGRAM_REPAIR_CANDIDATE_SCHEMA",
    "PROGRAM_REPAIR_RECEIPT_SCHEMA",
    "PROGRAM_REPAIR_REQUEST_SCHEMA",
    "PROGRAM_REPAIR_SYNTHESIZER_INTERFACE",
    "PROGRAM_REPAIR_SYNTHESIZER_VERSION",
    "PRODUCER_ID",
    "RESIDUAL_HYBRID_ADMISSION_SCHEMA",
    "RESIDUAL_HYBRID_PACKET_SCHEMA",
    "RESIDUAL_HYBRID_SERVICE_INTERFACE",
    "CEGIS_LOOP_RESULT_SCHEMA",
    "DeclaredEqualityTheory",
    "EqualityEGraph",
    "EqualityRewriteReceipt",
    "EqualityRewriteStatus",
    "EqualityRule",
    "HybridUsageReceipt",
    "ProgramRepairAuthorityError",
    "ProgramRepairBounds",
    "ProgramRepairBoundsError",
    "ProgramRepairCandidate",
    "ProgramRepairDisposition",
    "ProgramRepairMode",
    "ProgramRepairReason",
    "ProgramRepairReceipt",
    "ProgramRepairRequest",
    "ProgramRepairSynthesizer",
    "ProgramRepairSynthesisError",
    "ResidualHybridAdmission",
    "ResidualHybridDisposition",
    "ResidualHybridPacket",
    "ResidualHybridRepairService",
    "create_program_repair_synthesizer",
    "prove_equality_under_theory",
    "synthesize_program_repair",
)
