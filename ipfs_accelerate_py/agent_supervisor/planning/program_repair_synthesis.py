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

Equality mode saturates a typed e-graph under a reviewed theory. E-classes
carry sorts, congruence is restored by rebuild, side conditions and
provenance are checked, extraction is cost-bounded and replayable, and
independent equivalence/effect checks gate a proved receipt. Features that
this path cannot discharge (solver-semantic side conditions, kernel
equivalence, an external egg runtime) are recorded as unavailable.
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
    CEGIS_FORBIDDEN_PARAMETER_KEYS,
    REPAIR_OPERATOR_REGISTRY_INTERFACE,
    RepairOperatorKind,
    RepairOperatorLookupDisposition,
    RepairOperatorLookupRequest,
    RepairOperatorLookupResult,
    RepairOperatorRegistry,
    ReviewedRepairHook,
    UnknownRepairOperatorError,
    build_default_repair_operator_registry,
    cegis_restricted_operator_kinds,
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
MAX_EGRAPH_MATCHES: Final[int] = 4_096
MAX_REPLAY_STEPS: Final[int] = 128
MAX_SIDE_CONDITIONS: Final[int] = 16
DEFAULT_MAX_ENUMERATIVE_CANDIDATES: Final[int] = 8
DEFAULT_MAX_CEGIS_ITERATIONS: Final[int] = 8
DEFAULT_MAX_REWRITE_DEPTH: Final[int] = 16
DEFAULT_ECLASS_SORT: Final[str] = "Term"

EQUALITY_SATURATION_CAPABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/equality-saturation-capability@1"
)
EQUALITY_REWRITE_STEP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/equality-rewrite-step@1"
)

_MANDATORY_SIDE_CONDITIONS: Final[tuple[str, ...]] = (
    "reviewed",
    "oriented",
    "same_sort",
    "closed_vars",
    "no_authority",
    "no_undeclared_effects",
)
_REVIEWED_SIDE_CONDITIONS: Final[frozenset[str]] = frozenset(
    {
        *_MANDATORY_SIDE_CONDITIONS,
        "pure",
        "no_effects",
        "no_new_imports",
        "no_new_files",
        "idempotent",
    }
)
_FORBIDDEN_EFFECT_LABELS: Final[frozenset[str]] = frozenset(
    {
        "authority",
        "write_authority",
        "semantic_authority",
        "proof_authority",
        "new_file",
        "new_import",
        "undeclared_dependency",
    }
)
_KNOWN_EFFECT_LABELS: Final[frozenset[str]] = frozenset(
    {
        "pure",
        "io",
        "import",
        "file_write",
        "network",
        "state",
        *_FORBIDDEN_EFFECT_LABELS,
    }
)
_DEFAULT_OPERATOR_SORTS: Final[Mapping[str, tuple[tuple[str, ...], str]]] = (
    MappingProxyType(
        {
            "+": (("Int", "Int"), "Int"),
            "-": (("Int", "Int"), "Int"),
            "*": (("Int", "Int"), "Int"),
            "/": (("Int", "Int"), "Int"),
            "=": (("Int", "Int"), "Bool"),
            "<": (("Int", "Int"), "Bool"),
            ">": (("Int", "Int"), "Bool"),
            "<=": (("Int", "Int"), "Bool"),
            ">=": (("Int", "Int"), "Bool"),
            "and": (("Bool", "Bool"), "Bool"),
            "or": (("Bool", "Bool"), "Bool"),
            "not": (("Bool",), "Bool"),
        }
    )
)
_AVAILABLE_EQUALITY_FEATURES: Final[tuple[str, ...]] = (
    "typed_eclasses",
    "congruence_rebuild",
    "reviewed_side_conditions",
    "provenance",
    "bounded_saturation",
    "extraction_cost",
    "extraction_replay",
    "independent_equivalence_check",
    "independent_effect_check",
)
_UNAVAILABLE_EQUALITY_FEATURES: Final[tuple[tuple[str, str], ...]] = (
    (
        "smt_semantic_side_conditions",
        "no_solver_in_program_repair_synthesizer",
    ),
    (
        "kernel_checked_equivalence",
        "no_kernel_in_equality_rewrite_path",
    ),
    (
        "external_egg_runtime",
        "external_egg_or_egglog_runtime_not_integrated",
    ),
)

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
    EQUALITY_SIDE_CONDITION = "equality_side_condition_failed"
    EQUALITY_EFFECT_CHANGE = "equality_effect_change_rejected"
    EQUALITY_TYPE_MISMATCH = "equality_sort_mismatch"
    EQUALITY_REPLAY_FAILED = "equality_replay_failed"
    EQUALITY_INDEPENDENT_REJECT = "equality_independent_check_rejected"
    EQUALITY_INVALID_REWRITE = "equality_invalid_rewrite"
    EQUALITY_EXTRACTION = "equality_extraction_completed"
    UNVALIDATED_INTERPOLANT = "interpolant_not_independently_validated"
    UNDECLARED_EFFECT = "undeclared_effect_or_security_change"
    COUNTEREVIDENCE_RESTRICTED = "operator_restricted_by_counterevidence"


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
    INVALID = "invalid"


class EqualityFeatureStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


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


def _is_numeric_atom(text: str) -> bool:
    if text.startswith("-") and len(text) > 1:
        text = text[1:]
    return bool(text) and text.isdigit()


def _is_bool_atom(text: str) -> bool:
    return text in {"true", "false"}


def _is_pattern_var_name(name: str, declared: frozenset[str]) -> bool:
    return name.startswith("?") or name in declared


@dataclass(frozen=True)
class EqualityTerm:
    """Parsed S-expression used as a ground term or a rewrite pattern."""

    op: str
    children: tuple["EqualityTerm", ...] = ()
    is_var: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", _text(self.op, "term_op", limit=MAX_TEXT_BYTES))
        if self.is_var and self.children:
            raise ProgramRepairSynthesisError("pattern variables cannot have children")
        object.__setattr__(self, "is_var", bool(self.is_var))
        object.__setattr__(self, "children", tuple(self.children))

    @property
    def name(self) -> str:
        return self.op

    def render(self) -> str:
        if not self.children:
            return self.op
        return "(" + " ".join((self.op, *(child.render() for child in self.children))) + ")"

    def operators(self) -> frozenset[str]:
        if self.is_var:
            return frozenset()
        found = {self.op}
        for child in self.children:
            found.update(child.operators())
        return frozenset(found)

    def pattern_vars(self) -> frozenset[str]:
        if self.is_var:
            return frozenset({self.op})
        found: set[str] = set()
        for child in self.children:
            found.update(child.pattern_vars())
        return frozenset(found)


def _tokenize_equality_term(text: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []
    for char in text:
        if char.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
        elif char in "()":
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(char)
        elif char == "\x00":
            raise ProgramRepairSynthesisError("equality term must not contain NUL bytes")
        else:
            buf.append(char)
    if buf:
        tokens.append("".join(buf))
    return tokens


def parse_equality_term(
    text: str,
    *,
    name: str = "term",
    pattern: bool = False,
    pattern_vars: Sequence[str] = (),
) -> EqualityTerm:
    """Parse a bounded S-expression. Identifier leaves are symbols unless patterned."""

    raw = _text(text, name, limit=MAX_SPAN_BYTES)
    declared = frozenset(_ids(pattern_vars, "pattern_vars", limit=MAX_THEORY_RULES))
    tokens = _tokenize_equality_term(raw)
    if not tokens:
        raise ProgramRepairSynthesisError(f"{name} is required")

    def parse_at(index: int) -> tuple[EqualityTerm, int]:
        if index >= len(tokens):
            raise ProgramRepairSynthesisError(f"{name} is a malformed S-expression")
        token = tokens[index]
        if token == ")":
            raise ProgramRepairSynthesisError(f"{name} has an unmatched ')'")
        if token != "(":
            is_var = pattern and _is_pattern_var_name(token, declared)
            return EqualityTerm(op=token, is_var=is_var), index + 1
        index += 1
        if index >= len(tokens) or tokens[index] in {"(", ")"}:
            raise ProgramRepairSynthesisError(f"{name} is a malformed S-expression")
        op = tokens[index]
        index += 1
        children: list[EqualityTerm] = []
        while index < len(tokens) and tokens[index] != ")":
            child, index = parse_at(index)
            children.append(child)
        if index >= len(tokens) or tokens[index] != ")":
            raise ProgramRepairSynthesisError(f"{name} is missing a closing ')'")
        return EqualityTerm(op=op, children=tuple(children)), index + 1

    term, next_index = parse_at(0)
    if next_index != len(tokens):
        raise ProgramRepairSynthesisError(f"{name} has trailing tokens")
    return term


def _match_equality_term(
    pattern: EqualityTerm,
    term: EqualityTerm,
    subst: dict[str, EqualityTerm],
) -> bool:
    if pattern.is_var:
        bound = subst.get(pattern.op)
        if bound is None:
            subst[pattern.op] = term
            return True
        return bound == term
    if term.is_var or pattern.op != term.op or len(pattern.children) != len(term.children):
        return False
    return all(
        _match_equality_term(left, right, subst)
        for left, right in zip(pattern.children, term.children)
    )


def _instantiate_equality_term(
    pattern: EqualityTerm, subst: Mapping[str, EqualityTerm]
) -> EqualityTerm:
    if pattern.is_var:
        bound = subst.get(pattern.op)
        if bound is None:
            raise ProgramRepairSynthesisError(
                f"unbound pattern variable {pattern.op}",
                reason_code=ProgramRepairReason.EQUALITY_SIDE_CONDITION,
            )
        return bound
    if not pattern.children:
        return pattern
    return EqualityTerm(
        op=pattern.op,
        children=tuple(_instantiate_equality_term(child, subst) for child in pattern.children),
    )


def _unbound_rhs_pattern_vars(
    lhs: EqualityTerm,
    rhs: EqualityTerm,
    subst: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Return RHS pattern variables that are not bound by the LHS (or ``subst``)."""

    extra = set(rhs.pattern_vars() - lhs.pattern_vars())
    if subst is not None:
        extra.update(name for name in rhs.pattern_vars() if name not in subst)
    return tuple(sorted(extra))


def _rewrite_equality_term_all(
    term: EqualityTerm, lhs: EqualityTerm, rhs: EqualityTerm
) -> tuple[EqualityTerm, int]:
    """Rewrite every non-overlapping occurrence of a ground redex.

    E-graph applications are class-level: one recorded application can stand
    for several identical AST redexes.  Replay must therefore rewrite all
    occurrences in one deterministic pass, rather than arbitrarily choosing
    the first one.
    """

    subst: dict[str, EqualityTerm] = {}
    if _match_equality_term(lhs, term, subst):
        if _unbound_rhs_pattern_vars(lhs, rhs, subst):
            return term, 0
        return _instantiate_equality_term(rhs, subst), 1
    if not term.children:
        return term, 0
    children: list[EqualityTerm] = []
    applications = 0
    for child in term.children:
        rewritten, applied = _rewrite_equality_term_all(child, lhs, rhs)
        children.append(rewritten)
        applications += applied
    if not applications:
        return term, 0
    return EqualityTerm(op=term.op, children=tuple(children)), applications


def _collect_term_effects(
    term: EqualityTerm,
    operator_effects: Mapping[str, tuple[str, ...]],
) -> frozenset[str]:
    found: set[str] = set()
    stack = [term]
    while stack:
        current = stack.pop()
        if current.is_var:
            continue
        labels = operator_effects.get(current.op, ())
        found.update(labels)
        stack.extend(current.children)
    found.discard("pure")
    return frozenset(found)


@dataclass(frozen=True)
class EqualityRule:
    """One oriented, reviewed equality rewrite under a declared theory."""

    rule_id: str
    lhs: str
    rhs: str
    review_ref: str
    theory_id: str
    oriented: bool = True
    sort: str = ""
    side_conditions: tuple[str, ...] = ()
    pattern_vars: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    cost: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _text(self.rule_id, "rule_id"))
        object.__setattr__(self, "lhs", _text(self.lhs, "lhs", limit=MAX_SPAN_BYTES))
        object.__setattr__(self, "rhs", _text(self.rhs, "rhs", limit=MAX_SPAN_BYTES))
        object.__setattr__(self, "review_ref", _text(self.review_ref, "review_ref"))
        object.__setattr__(self, "theory_id", _text(self.theory_id, "theory_id"))
        object.__setattr__(self, "oriented", _bool(self.oriented, "oriented"))
        object.__setattr__(self, "sort", _optional_text(self.sort, "sort"))
        object.__setattr__(
            self,
            "side_conditions",
            _ids(self.side_conditions, "side_conditions", limit=MAX_SIDE_CONDITIONS),
        )
        object.__setattr__(
            self, "pattern_vars", _ids(self.pattern_vars, "pattern_vars", limit=MAX_THEORY_RULES)
        )
        object.__setattr__(
            self, "effects", _ids(self.effects, "effects", limit=MAX_SIDE_CONDITIONS)
        )
        object.__setattr__(self, "cost", _positive_int(self.cost, "cost", maximum=1_000))
        if not self.oriented:
            raise ProgramRepairSynthesisError(
                "equality rules must be oriented under the declared theory"
            )
        if self.lhs == self.rhs:
            raise ProgramRepairSynthesisError("equality rule lhs and rhs must differ")
        unknown = [item for item in self.side_conditions if item not in _REVIEWED_SIDE_CONDITIONS]
        if unknown:
            raise ProgramRepairSynthesisError(
                f"unknown reviewed side condition: {unknown[0]}",
                reason_code=ProgramRepairReason.EQUALITY_SIDE_CONDITION,
            )
        unknown_effects = [item for item in self.effects if item not in _KNOWN_EFFECT_LABELS]
        if unknown_effects:
            raise ProgramRepairSynthesisError(
                f"unknown effect label: {unknown_effects[0]}",
                reason_code=ProgramRepairReason.EQUALITY_EFFECT_CHANGE,
            )
        parse_equality_term(
            self.lhs, name="lhs", pattern=True, pattern_vars=self.pattern_vars
        )
        parse_equality_term(
            self.rhs, name="rhs", pattern=True, pattern_vars=self.pattern_vars
        )

    def parsed_lhs(self) -> EqualityTerm:
        return parse_equality_term(
            self.lhs, name="lhs", pattern=True, pattern_vars=self.pattern_vars
        )

    def parsed_rhs(self) -> EqualityTerm:
        return parse_equality_term(
            self.rhs, name="rhs", pattern=True, pattern_vars=self.pattern_vars
        )

    def effective_side_conditions(self) -> tuple[str, ...]:
        extra = [item for item in self.side_conditions if item not in _MANDATORY_SIDE_CONDITIONS]
        return (*_MANDATORY_SIDE_CONDITIONS, *extra)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "lhs": self.lhs,
            "rhs": self.rhs,
            "review_ref": self.review_ref,
            "theory_id": self.theory_id,
            "oriented": True,
            "sort": self.sort,
            "side_conditions": list(self.side_conditions),
            "pattern_vars": list(self.pattern_vars),
            "effects": list(self.effects),
            "cost": self.cost,
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
            sort=str(payload.get("sort") or ""),
            side_conditions=tuple(payload.get("side_conditions") or ()),
            pattern_vars=tuple(payload.get("pattern_vars") or ()),
            effects=tuple(payload.get("effects") or ()),
            cost=int(payload.get("cost") or 1),
        )


def _normalize_operator_sorts(
    value: Any, name: str = "operator_sorts"
) -> Mapping[str, tuple[tuple[str, ...], str]]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ProgramRepairSynthesisError(f"{name} must be a mapping")
    if len(value) > MAX_THEORY_RULES:
        raise ProgramRepairBoundsError(
            f"{name} exceeds its bound",
            reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
        )
    out: dict[str, tuple[tuple[str, ...], str]] = {}
    for raw_op, spec in value.items():
        op = _text(raw_op, "operator")
        if isinstance(spec, Mapping):
            args_raw = spec.get("args") or spec.get("argument_sorts") or ()
            result = spec.get("result") or spec.get("result_sort") or DEFAULT_ECLASS_SORT
        elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and len(spec) == 2:
            args_raw, result = spec
        else:
            raise ProgramRepairSynthesisError(f"{name} entries must be sort signatures")
        args = tuple(_text(item, "argument_sort") for item in args_raw or ())
        out[op] = (args, _text(result, "result_sort"))
    return MappingProxyType(out)


def _normalize_operator_effects(
    value: Any, name: str = "operator_effects"
) -> Mapping[str, tuple[str, ...]]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ProgramRepairSynthesisError(f"{name} must be a mapping")
    if len(value) > MAX_THEORY_RULES:
        raise ProgramRepairBoundsError(
            f"{name} exceeds its bound",
            reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
        )
    out: dict[str, tuple[str, ...]] = {}
    for raw_op, labels in value.items():
        op = _text(raw_op, "operator")
        normalized = _ids(labels or (), "effects", limit=MAX_SIDE_CONDITIONS)
        unknown = [item for item in normalized if item not in _KNOWN_EFFECT_LABELS]
        if unknown:
            raise ProgramRepairSynthesisError(
                f"unknown effect label: {unknown[0]}",
                reason_code=ProgramRepairReason.EQUALITY_EFFECT_CHANGE,
            )
        out[op] = normalized
    return MappingProxyType(out)


def _normalize_operator_costs(
    value: Any, name: str = "operator_costs"
) -> Mapping[str, int]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ProgramRepairSynthesisError(f"{name} must be a mapping")
    if len(value) > MAX_THEORY_RULES:
        raise ProgramRepairBoundsError(
            f"{name} exceeds its bound",
            reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
        )
    out: dict[str, int] = {}
    for raw_op, cost in value.items():
        out[_text(raw_op, "operator")] = _positive_int(cost, "operator_cost", maximum=1_000)
    return MappingProxyType(out)


def _normalize_leaf_sorts(value: Any, name: str = "leaf_sorts") -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ProgramRepairSynthesisError(f"{name} must be a mapping")
    if len(value) > MAX_THEORY_RULES:
        raise ProgramRepairBoundsError(
            f"{name} exceeds its bound",
            reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
        )
    return MappingProxyType(
        {
            _text(raw_name, "leaf"): _text(sort, "leaf_sort")
            for raw_name, sort in value.items()
        }
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
    operator_sorts: Mapping[str, tuple[tuple[str, ...], str]] = field(default_factory=dict)
    operator_effects: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    operator_costs: Mapping[str, int] = field(default_factory=dict)
    leaf_sorts: Mapping[str, str] = field(default_factory=dict)
    pattern_vars: tuple[str, ...] = ()
    allowed_effects: tuple[str, ...] = ()
    default_sort: str = DEFAULT_ECLASS_SORT

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
        object.__setattr__(
            self, "pattern_vars", _ids(self.pattern_vars, "pattern_vars", limit=MAX_THEORY_RULES)
        )
        object.__setattr__(
            self,
            "allowed_effects",
            _ids(self.allowed_effects, "allowed_effects", limit=MAX_SIDE_CONDITIONS),
        )
        object.__setattr__(
            self,
            "default_sort",
            _text(self.default_sort or DEFAULT_ECLASS_SORT, "default_sort"),
        )
        object.__setattr__(
            self, "operator_sorts", _normalize_operator_sorts(self.operator_sorts)
        )
        object.__setattr__(
            self, "operator_effects", _normalize_operator_effects(self.operator_effects)
        )
        object.__setattr__(
            self, "operator_costs", _normalize_operator_costs(self.operator_costs)
        )
        object.__setattr__(self, "leaf_sorts", _normalize_leaf_sorts(self.leaf_sorts))
        normalized: list[EqualityRule] = []
        seen_rule_ids: set[str] = set()
        for rule in self.rules:
            if isinstance(rule, EqualityRule):
                item = rule
            elif isinstance(rule, Mapping):
                item = EqualityRule.from_dict(rule)
            else:
                raise ProgramRepairSynthesisError("rules must be EqualityRule or mapping")
            if item.rule_id in seen_rule_ids:
                raise ProgramRepairSynthesisError("equality theory rule_ids must be unique")
            seen_rule_ids.add(item.rule_id)
            if item.theory_id and item.theory_id != self.theory_id:
                raise ProgramRepairSynthesisError(
                    "rule theory_id must match declared theory"
                )
            merged_vars = tuple(dict.fromkeys((*item.pattern_vars, *self.pattern_vars)))
            item = EqualityRule(
                rule_id=item.rule_id,
                lhs=item.lhs,
                rhs=item.rhs,
                review_ref=item.review_ref,
                theory_id=self.theory_id,
                oriented=True,
                sort=item.sort,
                side_conditions=item.side_conditions,
                pattern_vars=merged_vars,
                effects=item.effects,
                cost=item.cost,
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

    def rule_map(self) -> Mapping[str, EqualityRule]:
        return MappingProxyType({rule.rule_id: rule for rule in self.rules})

    def operator_signature(self, op: str) -> tuple[tuple[str, ...], str] | None:
        if op in self.operator_sorts:
            return self.operator_sorts[op]
        return _DEFAULT_OPERATOR_SORTS.get(op)

    def operator_cost(self, op: str) -> int:
        return int(self.operator_costs.get(op, 1))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "theory_id": self.theory_id,
            "review_refs": list(self.review_refs),
            "rules": [rule.to_dict() for rule in self.rules],
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "grants_semantic_authority": False,
            "operator_sorts": {
                op: {"args": list(args), "result": result}
                for op, (args, result) in self.operator_sorts.items()
            },
            "operator_effects": {
                op: list(labels) for op, labels in self.operator_effects.items()
            },
            "operator_costs": dict(self.operator_costs),
            "leaf_sorts": dict(self.leaf_sorts),
            "pattern_vars": list(self.pattern_vars),
            "allowed_effects": list(self.allowed_effects),
            "default_sort": self.default_sort,
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
            operator_sorts=payload.get("operator_sorts") or {},
            operator_effects=payload.get("operator_effects") or {},
            operator_costs=payload.get("operator_costs") or {},
            leaf_sorts=payload.get("leaf_sorts") or {},
            pattern_vars=tuple(payload.get("pattern_vars") or ()),
            allowed_effects=tuple(payload.get("allowed_effects") or ()),
            default_sort=str(payload.get("default_sort") or DEFAULT_ECLASS_SORT),
        )


@dataclass(frozen=True)
class EqualitySaturationCapability:
    """Availability record for one equality-saturation feature."""

    feature: str
    status: EqualityFeatureStatus
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature", _text(self.feature, "feature"))
        object.__setattr__(
            self, "status", _enum(self.status, EqualityFeatureStatus, "status")
        )
        object.__setattr__(self, "note", _optional_text(self.note, "note"))

    @property
    def available(self) -> bool:
        return self.status is EqualityFeatureStatus.AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EQUALITY_SATURATION_CAPABILITY_SCHEMA,
            "feature": self.feature,
            "status": self.status.value,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EqualitySaturationCapability":
        if not isinstance(payload, Mapping):
            raise ProgramRepairSynthesisError("capability must be a mapping")
        return cls(
            feature=str(payload.get("feature") or ""),
            status=str(payload.get("status") or EqualityFeatureStatus.UNAVAILABLE.value),
            note=str(payload.get("note") or ""),
        )


def equality_saturation_capabilities() -> tuple[EqualitySaturationCapability, ...]:
    """Return the closed inventory of available and unavailable e-graph features."""

    available = tuple(
        EqualitySaturationCapability(
            feature=feature,
            status=EqualityFeatureStatus.AVAILABLE,
            note="implemented_in_program_repair_synthesizer",
        )
        for feature in _AVAILABLE_EQUALITY_FEATURES
    )
    unavailable = tuple(
        EqualitySaturationCapability(
            feature=feature,
            status=EqualityFeatureStatus.UNAVAILABLE,
            note=note,
        )
        for feature, note in _UNAVAILABLE_EQUALITY_FEATURES
    )
    return (*available, *unavailable)


@dataclass(frozen=True)
class EqualityRewriteStep:
    """One reviewed rewrite application recorded for provenance and replay."""

    rule_id: str
    review_ref: str
    lhs: str
    rhs: str
    substitution: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _text(self.rule_id, "rule_id"))
        object.__setattr__(self, "review_ref", _text(self.review_ref, "review_ref"))
        object.__setattr__(self, "lhs", _text(self.lhs, "lhs", limit=MAX_SPAN_BYTES))
        object.__setattr__(self, "rhs", _text(self.rhs, "rhs", limit=MAX_SPAN_BYTES))
        pairs: list[tuple[str, str]] = []
        seen_vars: set[str] = set()
        for item in self.substitution or ():
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) != 2
            ):
                raise ProgramRepairSynthesisError("substitution entries must be pairs")
            name = _text(item[0], "subst_var")
            if name in seen_vars:
                raise ProgramRepairSynthesisError("substitution variables must be unique")
            seen_vars.add(name)
            term = _text(item[1], "subst_term", limit=MAX_SPAN_BYTES)
            parsed = parse_equality_term(term, name="subst_term")
            if parsed.is_var:
                raise ProgramRepairSynthesisError("substitution terms must be ground")
            pairs.append((name, parsed.render()))
        object.__setattr__(self, "substitution", tuple(pairs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EQUALITY_REWRITE_STEP_SCHEMA,
            "rule_id": self.rule_id,
            "review_ref": self.review_ref,
            "lhs": self.lhs,
            "rhs": self.rhs,
            "substitution": [[name, term] for name, term in self.substitution],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EqualityRewriteStep":
        if not isinstance(payload, Mapping):
            raise ProgramRepairSynthesisError("rewrite step must be a mapping")
        raw_subst = payload.get("substitution") or ()
        pairs: list[tuple[str, str]] = []
        for item in raw_subst:
            if isinstance(item, Mapping):
                pairs.append((str(item.get("var") or ""), str(item.get("term") or "")))
            elif (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and len(item) == 2
            ):
                pairs.append((str(item[0]), str(item[1])))
            else:
                raise ProgramRepairSynthesisError("substitution entries must be pairs")
        return cls(
            rule_id=str(payload.get("rule_id") or ""),
            review_ref=str(payload.get("review_ref") or ""),
            lhs=str(payload.get("lhs") or ""),
            rhs=str(payload.get("rhs") or ""),
            substitution=tuple(pairs),
        )


def _coerce_rewrite_steps(value: Any) -> tuple[EqualityRewriteStep, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProgramRepairSynthesisError("replay_steps must be a sequence")
    if len(value) > MAX_REPLAY_STEPS:
        raise ProgramRepairBoundsError(
            "replay_steps exceeds its bound",
            reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
        )
    steps: list[EqualityRewriteStep] = []
    for item in value:
        if isinstance(item, EqualityRewriteStep):
            steps.append(item)
        elif isinstance(item, Mapping):
            steps.append(EqualityRewriteStep.from_dict(item))
        else:
            raise ProgramRepairSynthesisError("replay step must be EqualityRewriteStep")
    return tuple(steps)


def _coerce_capabilities(value: Any) -> tuple[EqualitySaturationCapability, ...]:
    if value is None:
        return equality_saturation_capabilities()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProgramRepairSynthesisError("capabilities must be a sequence")
    out: list[EqualitySaturationCapability] = []
    for item in value:
        if isinstance(item, EqualitySaturationCapability):
            out.append(item)
        elif isinstance(item, Mapping):
            out.append(EqualitySaturationCapability.from_dict(item))
        else:
            raise ProgramRepairSynthesisError(
                "capability must be EqualitySaturationCapability"
            )
    return tuple(out)


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
    eclass_count: int = 0
    rebuild_count: int = 0
    congruence_merges: int = 0
    extraction_cost: int = 0
    extracted_term: str = ""
    source_sort: str = ""
    target_sort: str = ""
    applied_review_refs: tuple[str, ...] = ()
    replay_steps: tuple[EqualityRewriteStep, ...] = ()
    independent_equivalence: str = ""
    independent_effect: str = ""
    side_condition_results: tuple[str, ...] = ()
    capabilities: tuple[EqualitySaturationCapability, ...] = field(
        default_factory=equality_saturation_capabilities
    )

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
        object.__setattr__(
            self, "eclass_count", _nonneg_int(self.eclass_count, "eclass_count")
        )
        object.__setattr__(
            self, "rebuild_count", _nonneg_int(self.rebuild_count, "rebuild_count")
        )
        object.__setattr__(
            self,
            "congruence_merges",
            _nonneg_int(self.congruence_merges, "congruence_merges"),
        )
        object.__setattr__(
            self,
            "extraction_cost",
            _nonneg_int(self.extraction_cost, "extraction_cost"),
        )
        object.__setattr__(
            self,
            "extracted_term",
            _optional_text(self.extracted_term, "extracted_term", limit=MAX_SPAN_BYTES),
        )
        object.__setattr__(self, "source_sort", _optional_text(self.source_sort, "source_sort"))
        object.__setattr__(self, "target_sort", _optional_text(self.target_sort, "target_sort"))
        object.__setattr__(
            self,
            "applied_review_refs",
            _ids(self.applied_review_refs, "applied_review_refs"),
        )
        object.__setattr__(self, "replay_steps", _coerce_rewrite_steps(self.replay_steps))
        object.__setattr__(
            self,
            "independent_equivalence",
            _optional_text(self.independent_equivalence, "independent_equivalence"),
        )
        object.__setattr__(
            self,
            "independent_effect",
            _optional_text(self.independent_effect, "independent_effect"),
        )
        object.__setattr__(
            self,
            "side_condition_results",
            _ids(self.side_condition_results, "side_condition_results"),
        )
        object.__setattr__(self, "capabilities", _coerce_capabilities(self.capabilities))
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
        expected_capabilities = {
            item.feature: (item.status, item.note)
            for item in equality_saturation_capabilities()
        }
        actual_capabilities = {
            item.feature: (item.status, item.note)
            for item in self.capabilities
        }
        if (
            len(self.capabilities) != len(expected_capabilities)
            or actual_capabilities != expected_capabilities
        ):
            raise ProgramRepairSynthesisError(
                "equality receipt capabilities must match the implementation inventory"
            )
        if self.proved and (
            not self.independent_equivalence.startswith("passed")
            or not self.independent_effect.startswith("passed")
        ):
            raise ProgramRepairSynthesisError(
                "proved equality receipt requires independent equivalence and effect checks",
                reason_code=ProgramRepairReason.EQUALITY_INDEPENDENT_REJECT,
            )

    @property
    def proved(self) -> bool:
        return self.status is EqualityRewriteStatus.PROVED

    def capability_map(self) -> Mapping[str, EqualitySaturationCapability]:
        return MappingProxyType({item.feature: item for item in self.capabilities})

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
            "eclass_count": self.eclass_count,
            "rebuild_count": self.rebuild_count,
            "congruence_merges": self.congruence_merges,
            "extraction_cost": self.extraction_cost,
            "extracted_term": self.extracted_term,
            "source_sort": self.source_sort,
            "target_sort": self.target_sort,
            "applied_review_refs": list(self.applied_review_refs),
            "replay_steps": [step.to_dict() for step in self.replay_steps],
            "independent_equivalence": self.independent_equivalence,
            "independent_effect": self.independent_effect,
            "side_condition_results": list(self.side_condition_results),
            "capabilities": [item.to_dict() for item in self.capabilities],
        }


@dataclass(frozen=True)
class _ENode:
    op: str
    children: tuple[int, ...]


class EqualityEGraph:
    """Typed e-graph with congruence rebuild under a declared reviewed theory.

    E-classes carry sorts. Hashcons plus rebuild restore congruence. Only
    reviewed rules whose side conditions hold may union classes. Extraction
    reports AST cost; independent AST replay and effect comparison gate a
    proved receipt. The engine stays proposal-only and fail-closed.
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
        self._parent: list[int] = []
        self._rank: list[int] = []
        self._sorts: list[str] = []
        self._nodes_of: list[list[int]] = []
        self._enodes: list[_ENode] = []
        self._enode_class: list[int] = []
        self._hashcons: dict[tuple[str, tuple[int, ...]], int] = {}
        self._repr: dict[int, str] = {}
        self._applied: list[str] = []
        self._steps: list[EqualityRewriteStep] = []
        self._side_results: list[str] = []
        self._rebuild_count = 0
        self._congruence_merges = 0
        self._depth_budget_exhausted = False
        self._seen_apps: set[tuple[str, int, tuple[tuple[str, int], ...]]] = set()

    @property
    def _nodes(self) -> int:
        return len(self._enodes)

    def _find(self, cid: int) -> int:
        while self._parent[cid] != cid:
            self._parent[cid] = self._parent[self._parent[cid]]
            cid = self._parent[cid]
        return cid

    def _canonical_classes(self) -> list[int]:
        return [index for index, parent in enumerate(self._parent) if parent == index]

    def _infer_leaf_sort(self, op: str) -> str:
        if op in self.theory.leaf_sorts:
            return self.theory.leaf_sorts[op]
        if _is_bool_atom(op):
            return "Bool"
        if _is_numeric_atom(op):
            return "Int"
        return self.theory.default_sort

    def _enode_sort(self, op: str, child_sorts: Sequence[str]) -> str:
        signature = self.theory.operator_signature(op)
        if signature is None:
            if not child_sorts:
                return self._infer_leaf_sort(op)
            unique = {sort for sort in child_sorts if sort != DEFAULT_ECLASS_SORT}
            if len(unique) == 1:
                return next(iter(unique))
            return self.theory.default_sort
        args, result = signature
        if len(args) != len(child_sorts):
            raise ProgramRepairSynthesisError(
                f"operator arity mismatch for {op}",
                reason_code=ProgramRepairReason.EQUALITY_TYPE_MISMATCH,
            )
        for expected, actual in zip(args, child_sorts):
            if self._compatible_sorts(expected, actual) is None:
                raise ProgramRepairSynthesisError(
                    f"argument sort mismatch for {op}",
                    reason_code=ProgramRepairReason.EQUALITY_TYPE_MISMATCH,
                )
        return result

    def _compatible_sorts(self, left: str, right: str) -> str | None:
        if left == right:
            return left
        if left == DEFAULT_ECLASS_SORT:
            return right
        if right == DEFAULT_ECLASS_SORT:
            return left
        return None

    def _union(
        self,
        left: int,
        right: int,
        *,
        rule_id: str = "",
        review_ref: str = "",
        step: EqualityRewriteStep | None = None,
    ) -> bool:
        root_a = self._find(left)
        root_b = self._find(right)
        if root_a == root_b:
            return False
        merged_sort = self._compatible_sorts(self._sorts[root_a], self._sorts[root_b])
        if merged_sort is None:
            self._side_results.append(
                f"{rule_id or 'congruence'}:same_sort:failed:{self._sorts[root_a]}!={self._sorts[root_b]}"
            )
            return False
        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1
        self._sorts[root_a] = merged_sort
        seen_nids = set(self._nodes_of[root_a])
        for nid in self._nodes_of[root_b]:
            self._enode_class[nid] = root_a
            if nid not in seen_nids:
                self._nodes_of[root_a].append(nid)
                seen_nids.add(nid)
        self._nodes_of[root_b] = []
        other = self._repr.get(root_b)
        if other is not None:
            self._remember_repr(root_a, other)
        if rule_id == "congruence":
            self._congruence_merges += 1
        elif rule_id:
            if len(self._steps) >= MAX_REPLAY_STEPS:
                raise ProgramRepairBoundsError(
                    "equality replay-step budget exhausted",
                    reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
                )
            self._applied.append(rule_id)
            if step is not None:
                self._steps.append(step)
            elif review_ref:
                self._steps.append(
                    EqualityRewriteStep(
                        rule_id=rule_id,
                        review_ref=review_ref,
                        lhs=self._repr.get(root_b, ""),
                        rhs=self._repr.get(root_a, ""),
                    )
                )
        return True

    def _add_enode(self, op: str, children: Sequence[int], *, sort: str) -> int:
        canon = tuple(self._find(child) for child in children)
        key = (op, canon)
        existing = self._hashcons.get(key)
        if existing is not None:
            cid = self._find(self._enode_class[existing])
            merged = self._compatible_sorts(self._sorts[cid], sort)
            if merged is None:
                raise ProgramRepairSynthesisError(
                    f"e-class sort conflict for operator {op}",
                    reason_code=ProgramRepairReason.EQUALITY_TYPE_MISMATCH,
                )
            self._sorts[cid] = merged
            return cid
        if len(self._enodes) >= self.max_nodes:
            raise ProgramRepairBoundsError(
                "e-graph node budget exhausted",
                reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
            )
        cid = len(self._parent)
        self._parent.append(cid)
        self._rank.append(0)
        self._sorts.append(sort)
        self._nodes_of.append([])
        nid = len(self._enodes)
        self._enodes.append(_ENode(op=op, children=canon))
        self._enode_class.append(cid)
        self._nodes_of[cid].append(nid)
        self._hashcons[key] = nid
        if not canon:
            self._repr[cid] = op
        else:
            child_text = [self._repr.get(child, f"#{child}") for child in canon]
            self._repr[cid] = "(" + " ".join((op, *child_text)) + ")"
        return cid

    def add_term(self, term: EqualityTerm | str) -> int:
        parsed = term if isinstance(term, EqualityTerm) else parse_equality_term(term)
        if parsed.is_var:
            raise ProgramRepairSynthesisError("ground terms cannot contain pattern variables")
        child_ids = [self.add_term(child) for child in parsed.children]
        child_sorts = [self._sorts[self._find(cid)] for cid in child_ids]
        signature = self.theory.operator_signature(parsed.op)
        if signature is not None:
            args, result = signature
            if len(args) != len(child_sorts):
                raise ProgramRepairSynthesisError(
                    f"operator arity mismatch for {parsed.op}",
                    reason_code=ProgramRepairReason.EQUALITY_TYPE_MISMATCH,
                )
            for expected, actual, child_id in zip(args, child_sorts, child_ids):
                merged = self._compatible_sorts(expected, actual)
                if merged is None:
                    raise ProgramRepairSynthesisError(
                        f"argument sort mismatch for {parsed.op}",
                        reason_code=ProgramRepairReason.EQUALITY_TYPE_MISMATCH,
                    )
                self._sorts[self._find(child_id)] = merged
            sort = result
        else:
            sort = self._enode_sort(parsed.op, child_sorts)
        cid = self._add_enode(parsed.op, child_ids, sort=sort)
        self._remember_repr(cid, parsed.render())
        return cid

    def _rebuild(self) -> int:
        merges = 0
        progressed = True
        while progressed:
            progressed = False
            self._rebuild_count += 1
            if self._rebuild_count > self.max_depth * 8:
                raise ProgramRepairBoundsError(
                    "e-graph rebuild budget exhausted",
                    reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
                )
            buckets: dict[tuple[str, tuple[int, ...]], list[int]] = {}
            for nid, node in enumerate(self._enodes):
                canon_children = tuple(self._find(child) for child in node.children)
                if canon_children != node.children:
                    node = _ENode(op=node.op, children=canon_children)
                    self._enodes[nid] = node
                cid = self._find(self._enode_class[nid])
                self._enode_class[nid] = cid
                buckets.setdefault((node.op, canon_children), []).append(nid)
            self._hashcons = {key: nids[0] for key, nids in buckets.items()}
            for nids in buckets.values():
                root = self._find(self._enode_class[nids[0]])
                for nid in nids[1:]:
                    other = self._find(self._enode_class[nid])
                    if other != root and self._union(root, other, rule_id="congruence"):
                        merges += 1
                        progressed = True
                        root = self._find(root)
        self._reindex_class_nodes()
        return merges

    def _remember_repr(self, cid: int, text: str) -> None:
        """Keep the shortest deterministic printable witness for an e-class."""

        if not text:
            return
        cid = self._find(cid)
        current = self._repr.get(cid)
        if current is None or (len(text), text) < (len(current), current):
            self._repr[cid] = text

    def _reindex_class_nodes(self) -> None:
        """Rebuild per-class enode indexes from canonical union-find roots."""

        indexed: list[list[int]] = [[] for _ in self._parent]
        for nid, cid in enumerate(self._enode_class):
            root = self._find(cid)
            self._enode_class[nid] = root
            indexed[root].append(nid)
        self._nodes_of = indexed

    def _match_pattern_all(
        self, pattern: EqualityTerm, eclass: int, subst: Mapping[str, int]
    ) -> list[dict[str, int]]:
        """Return every e-match for ``pattern`` in one e-class.

        An e-class may hold several enodes after a union.  Returning only the
        first match loses legal rewrite opportunities and can make saturation
        incomplete.  Each result is canonicalized and deduplicated so the
        caller can safely snapshot applications before mutating the graph.
        """

        eclass = self._find(eclass)
        if pattern.is_var:
            bound = subst.get(pattern.op)
            if bound is None:
                return [{**subst, pattern.op: eclass}]
            return [dict(subst)] if self._find(bound) == eclass else []
        matches: list[dict[str, int]] = []
        for nid in tuple(self._nodes_of[eclass]):
            if self._find(self._enode_class[nid]) != eclass:
                continue
            node = self._enodes[nid]
            if node.op != pattern.op or len(node.children) != len(pattern.children):
                continue
            partials = [dict(subst)]
            for child_pat, child_id in zip(pattern.children, node.children):
                next_partials: list[dict[str, int]] = []
                for partial in partials:
                    next_partials.extend(
                        self._match_pattern_all(child_pat, child_id, partial)
                    )
                    if len(next_partials) > MAX_EGRAPH_MATCHES:
                        raise ProgramRepairBoundsError(
                            "e-graph match budget exhausted",
                            reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
                        )
                partials = next_partials
                if not partials:
                    break
            matches.extend(partials)
            if len(matches) > MAX_EGRAPH_MATCHES:
                raise ProgramRepairBoundsError(
                    "e-graph match budget exhausted",
                    reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
                )
        unique: dict[tuple[tuple[str, int], ...], dict[str, int]] = {}
        for match in matches:
            canonical = {
                name: self._find(cid)
                for name, cid in match.items()
            }
            key = tuple(sorted(canonical.items()))
            unique.setdefault(key, canonical)
        return list(unique.values())

    def _ematch(self, pattern: EqualityTerm) -> list[tuple[int, dict[str, int]]]:
        matches: list[tuple[int, dict[str, int]]] = []
        seen: set[tuple[int, tuple[tuple[str, int], ...]]] = set()
        for eclass in self._canonical_classes():
            for subst in self._match_pattern_all(pattern, eclass, {}):
                key = (
                    eclass,
                    tuple(sorted((name, self._find(cid)) for name, cid in subst.items())),
                )
                if key in seen:
                    continue
                seen.add(key)
                matches.append(
                    (eclass, {name: self._find(cid) for name, cid in subst.items()})
                )
                if len(matches) > MAX_EGRAPH_MATCHES:
                    raise ProgramRepairBoundsError(
                        "e-graph match budget exhausted",
                        reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
                    )
        return matches

    def _instantiate(self, pattern: EqualityTerm, subst: Mapping[str, int]) -> int:
        if pattern.is_var:
            if pattern.op not in subst:
                raise ProgramRepairSynthesisError(
                    f"unbound pattern variable {pattern.op}",
                    reason_code=ProgramRepairReason.EQUALITY_SIDE_CONDITION,
                )
            return self._find(subst[pattern.op])
        children = [self._instantiate(child, subst) for child in pattern.children]
        child_sorts = [self._sorts[self._find(cid)] for cid in children]
        sort = self._enode_sort(pattern.op, child_sorts)
        return self._add_enode(pattern.op, children, sort=sort)

    def _sort_of_term(self, term: EqualityTerm, subst: Mapping[str, int]) -> str:
        if term.is_var:
            bound = subst.get(term.op)
            if bound is None:
                return self.theory.default_sort
            return self._sorts[self._find(bound)]
        child_sorts = [self._sort_of_term(child, subst) for child in term.children]
        return self._enode_sort(term.op, child_sorts)

    def _render_eclass(self, cid: int) -> str:
        cid = self._find(cid)
        text = self._repr.get(cid)
        if text:
            return text
        try:
            extracted, _cost = self.extract_eclass(cid)
        except ProgramRepairSynthesisError:
            return f"#{cid}"
        return extracted.render()

    def _check_side_conditions(
        self,
        rule: EqualityRule,
        lhs_class: int,
        rhs_term: EqualityTerm,
        subst: Mapping[str, int],
    ) -> bool:
        lhs_class = self._find(lhs_class)
        lhs_pat = rule.parsed_lhs()
        rhs_pat = rule.parsed_rhs()
        ok = True
        for condition in rule.effective_side_conditions():
            failed = ""
            if condition == "reviewed":
                if rule.review_ref not in self.theory.review_refs:
                    failed = "review_ref_not_in_theory"
            elif condition == "oriented":
                if not rule.oriented or rule.lhs == rule.rhs:
                    failed = "not_oriented"
            elif condition == "closed_vars":
                extra = _unbound_rhs_pattern_vars(lhs_pat, rhs_pat, subst)
                if extra:
                    failed = "unbound:" + ",".join(extra)
            elif condition == "same_sort":
                expected = self._sorts[lhs_class]
                actual = self._sort_of_term(rhs_term, subst)
                if self._compatible_sorts(expected, actual) is None:
                    failed = f"{expected}!={actual}"
                elif rule.sort and (
                    self._compatible_sorts(rule.sort, expected) is None
                    or self._compatible_sorts(rule.sort, actual) is None
                ):
                    failed = f"declared:{rule.sort}!={expected}/{actual}"
            elif condition == "no_authority":
                labels = set(rule.effects) | set(
                    _collect_term_effects(rhs_term, self.theory.operator_effects)
                )
                if labels & _FORBIDDEN_EFFECT_LABELS:
                    failed = "authority_effect"
            elif condition in {"pure", "no_effects"}:
                introduced = set(rule.effects) | set(
                    _collect_term_effects(rhs_term, self.theory.operator_effects)
                )
                introduced -= {"pure"}
                if introduced:
                    failed = "effect:" + ",".join(sorted(introduced))
            elif condition == "no_undeclared_effects":
                introduced = set(rule.effects) | set(
                    _collect_term_effects(rhs_term, self.theory.operator_effects)
                )
                introduced -= {"pure"}
                introduced -= set(self.theory.allowed_effects)
                if introduced:
                    failed = "effect:" + ",".join(sorted(introduced))
            elif condition == "no_new_imports":
                if "import" in rule.effects or "new_import" in rule.effects:
                    failed = "new_import"
            elif condition == "no_new_files":
                if "file_write" in rule.effects or "new_file" in rule.effects:
                    failed = "new_file"
            elif condition == "idempotent":
                if lhs_pat == rhs_pat:
                    failed = "not_productive"
            else:
                failed = "unknown_side_condition"
            if failed:
                ok = False
                self._side_results.append(f"{rule.rule_id}:{condition}:failed:{failed}")
            else:
                self._side_results.append(f"{rule.rule_id}:{condition}:passed")
        return ok

    def _record_closed_vars_failure(self, rule: EqualityRule, unbound: Sequence[str]) -> None:
        names = ",".join(sorted(set(unbound)))
        self._side_results.append(f"{rule.rule_id}:closed_vars:failed:unbound:{names}")

    def _application_key(
        self, rule: EqualityRule, lhs_class: int, subst: Mapping[str, int]
    ) -> tuple[str, int, tuple[tuple[str, int], ...]]:
        return (
            rule.rule_id,
            self._find(lhs_class),
            tuple(sorted((name, self._find(cid)) for name, cid in subst.items())),
        )

    def _collect_rule_applications(
        self, rule: EqualityRule
    ) -> list[tuple[EqualityRule, int, dict[str, int], EqualityRewriteStep]]:
        """Snapshot e-matches for ``rule`` without mutating the graph."""

        applications: list[tuple[EqualityRule, int, dict[str, int], EqualityRewriteStep]] = []
        lhs = rule.parsed_lhs()
        rhs = rule.parsed_rhs()
        if lhs.is_var:
            self._side_results.append(f"{rule.rule_id}:lhs:failed:unconstrained_variable")
            return applications
        static_unbound = _unbound_rhs_pattern_vars(lhs, rhs)
        if static_unbound:
            self._record_closed_vars_failure(rule, static_unbound)
            return applications
        for lhs_class, subst in self._ematch(lhs):
            key = self._application_key(rule, lhs_class, subst)
            if key in self._seen_apps:
                continue
            missing = _unbound_rhs_pattern_vars(lhs, rhs, subst)
            if missing:
                self._record_closed_vars_failure(rule, missing)
                self._seen_apps.add(key)
                continue
            try:
                rhs_term = _instantiate_equality_term(
                    rhs,
                    {
                        name: parse_equality_term(self._render_eclass(cid))
                        for name, cid in subst.items()
                    },
                )
            except ProgramRepairSynthesisError as exc:
                if exc.reason_code == ProgramRepairReason.EQUALITY_SIDE_CONDITION.value:
                    self._record_closed_vars_failure(
                        rule,
                        _unbound_rhs_pattern_vars(lhs, rhs, subst)
                        or tuple(sorted(rhs.pattern_vars())),
                    )
                else:
                    self._side_results.append(
                        f"{rule.rule_id}:instantiate:failed:{exc.reason_code}"
                    )
                self._seen_apps.add(key)
                continue
            if not self._check_side_conditions(rule, lhs_class, rhs_term, subst):
                self._seen_apps.add(key)
                continue
            step = EqualityRewriteStep(
                rule_id=rule.rule_id,
                review_ref=rule.review_ref,
                lhs=rule.lhs,
                rhs=rule.rhs,
                substitution=tuple(
                    sorted((name, self._render_eclass(cid)) for name, cid in subst.items())
                ),
            )
            applications.append((rule, self._find(lhs_class), dict(subst), step))
            self._seen_apps.add(key)
        return applications

    def _commit_application(
        self,
        rule: EqualityRule,
        lhs_class: int,
        subst: Mapping[str, int],
        step: EqualityRewriteStep,
    ) -> bool:
        rhs = rule.parsed_rhs()
        try:
            rhs_class = self._instantiate(rhs, subst)
        except ProgramRepairBoundsError:
            raise
        except ProgramRepairSynthesisError as exc:
            if exc.reason_code == ProgramRepairReason.EQUALITY_SIDE_CONDITION.value:
                self._record_closed_vars_failure(
                    rule,
                    _unbound_rhs_pattern_vars(rule.parsed_lhs(), rhs, subst)
                    or tuple(sorted(rhs.pattern_vars())),
                )
            else:
                self._side_results.append(
                    f"{rule.rule_id}:instantiate:failed:{exc.reason_code}"
                )
            return False
        return self._union(
            lhs_class,
            rhs_class,
            rule_id=rule.rule_id,
            review_ref=rule.review_ref,
            step=step,
        )

    def saturate(self) -> int:
        """One depth step is a snapshot of matches, their unions, then rebuild."""

        depth = 0
        changed = True
        self._depth_budget_exhausted = False
        while changed and depth < self.max_depth:
            changed = False
            depth += 1
            pending: list[tuple[EqualityRule, int, dict[str, int], EqualityRewriteStep]] = []
            for rule in self.theory.rules:
                pending.extend(self._collect_rule_applications(rule))
            for rule, lhs_class, subst, step in pending:
                if self._commit_application(rule, lhs_class, subst, step):
                    changed = True
            if self._rebuild():
                changed = True
        self._depth_budget_exhausted = changed and depth >= self.max_depth
        return depth

    def extract_eclass(self, eclass: int) -> tuple[EqualityTerm, int]:
        inf = 10**9
        best_cost: dict[int, int] = {}
        best_node: dict[int, int] = {}
        progressed = True
        guard = 0
        while progressed and guard <= len(self._enodes) + 1:
            progressed = False
            guard += 1
            for cid in self._canonical_classes():
                for nid in self._nodes_of[cid]:
                    if self._find(self._enode_class[nid]) != cid:
                        continue
                    node = self._enodes[nid]
                    child_roots = [self._find(child) for child in node.children]
                    if any(child not in best_cost for child in child_roots):
                        if child_roots:
                            continue
                    child_total = sum(best_cost.get(child, 0) for child in child_roots)
                    cost = self.theory.operator_cost(node.op) + child_total
                    current = best_cost.get(cid, inf)
                    if cost < current or (
                        cost == current
                        and nid < best_node.get(cid, nid)
                    ):
                        best_cost[cid] = cost
                        best_node[cid] = nid
                        progressed = True
        root = self._find(eclass)
        if root not in best_node:
            raise ProgramRepairSynthesisError(
                "extraction failed for e-class",
                reason_code=ProgramRepairReason.EQUALITY_UNPROVED,
            )

        def build(cid: int, stack: set[int]) -> EqualityTerm:
            cid = self._find(cid)
            if cid in stack:
                raise ProgramRepairSynthesisError(
                    "extraction encountered a cyclic e-class",
                    reason_code=ProgramRepairReason.EQUALITY_UNPROVED,
                )
            nid = best_node[cid]
            node = self._enodes[nid]
            if not node.children:
                return EqualityTerm(op=node.op)
            nested = set(stack)
            nested.add(cid)
            return EqualityTerm(
                op=node.op,
                children=tuple(build(child, nested) for child in node.children),
            )

        return build(root, set()), best_cost[root]

    def extract(self, term: EqualityTerm | str) -> tuple[str, int]:
        parsed = term if isinstance(term, EqualityTerm) else parse_equality_term(term)
        cid = self.add_term(parsed)
        extracted, cost = self.extract_eclass(cid)
        return extracted.render(), cost

    def equivalent(self, source: str, target: str) -> bool:
        source_id = self.add_term(source)
        target_id = self.add_term(target)
        self.saturate()
        return self._find(source_id) == self._find(target_id)

    def _receipt(
        self,
        *,
        source: str,
        target: str,
        status: EqualityRewriteStatus,
        reason_code: str,
        depth: int,
        extracted_term: str = "",
        extraction_cost: int = 0,
        source_sort: str = "",
        target_sort: str = "",
        independent_equivalence: str = "",
        independent_effect: str = "",
    ) -> EqualityRewriteReceipt:
        return EqualityRewriteReceipt(
            theory_id=self.theory.theory_id,
            source_term=source,
            target_term=target,
            status=status,
            applied_rule_ids=tuple(dict.fromkeys(self._applied)),
            rewrite_depth=depth,
            egraph_node_count=self._nodes,
            reason_code=reason_code,
            eclass_count=len(self._canonical_classes()),
            rebuild_count=self._rebuild_count,
            congruence_merges=self._congruence_merges,
            extraction_cost=extraction_cost,
            extracted_term=extracted_term,
            source_sort=source_sort,
            target_sort=target_sort,
            applied_review_refs=tuple(
                dict.fromkeys(step.review_ref for step in self._steps if step.review_ref)
            ),
            replay_steps=tuple(self._steps[:MAX_REPLAY_STEPS]),
            independent_equivalence=independent_equivalence,
            independent_effect=independent_effect,
            side_condition_results=tuple(dict.fromkeys(self._side_results))[:MAX_REASON_CODES],
            capabilities=equality_saturation_capabilities(),
        )

    def prove(self, source: str, target: str) -> EqualityRewriteReceipt:
        source_t = _text(source, "source_term", limit=MAX_SPAN_BYTES)
        target_t = _text(target, "target_term", limit=MAX_SPAN_BYTES)
        depth = 0
        try:
            source_term = parse_equality_term(source_t, name="source_term")
            target_term = parse_equality_term(target_t, name="target_term")
        except ProgramRepairSynthesisError:
            return self._receipt(
                source=source_t,
                target=target_t,
                status=EqualityRewriteStatus.UNSUPPORTED,
                reason_code=ProgramRepairReason.MALFORMED_INPUT.value,
                depth=0,
            )
        try:
            source_id = self.add_term(source_term)
            target_id = self.add_term(target_term)
            source_sort = self._sorts[self._find(source_id)]
            target_sort = self._sorts[self._find(target_id)]
            if source_term == target_term:
                extracted, cost = self.extract_eclass(source_id)
                effect_status, effect_reason = _independent_effect_check(
                    self.theory, source_term, target_term
                )
                return self._receipt(
                    source=source_t,
                    target=target_t,
                    status=EqualityRewriteStatus.PROVED,
                    reason_code=ProgramRepairReason.EQUALITY_PROVED.value,
                    depth=0,
                    extracted_term=extracted.render(),
                    extraction_cost=cost,
                    source_sort=source_sort,
                    target_sort=target_sort,
                    independent_equivalence="passed:identical",
                    independent_effect=effect_status if effect_status.startswith("passed") else effect_reason,
                )
            if self._compatible_sorts(source_sort, target_sort) is None:
                return self._receipt(
                    source=source_t,
                    target=target_t,
                    status=EqualityRewriteStatus.INVALID,
                    reason_code=ProgramRepairReason.EQUALITY_TYPE_MISMATCH.value,
                    depth=0,
                    source_sort=source_sort,
                    target_sort=target_sort,
                    independent_equivalence="not_applicable",
                    independent_effect="not_applicable",
                )
            effect_status, effect_reason = _independent_effect_check(
                self.theory, source_term, target_term
            )
            if not effect_status.startswith("passed"):
                return self._receipt(
                    source=source_t,
                    target=target_t,
                    status=EqualityRewriteStatus.INVALID,
                    reason_code=ProgramRepairReason.EQUALITY_EFFECT_CHANGE.value,
                    depth=0,
                    source_sort=source_sort,
                    target_sort=target_sort,
                    independent_equivalence="not_applicable",
                    independent_effect=effect_reason,
                )
            depth = self.saturate()
            self._reindex_class_nodes()
            extracted, cost = self.extract_eclass(source_id)
            source_sort = self._sorts[self._find(source_id)]
            target_sort = self._sorts[self._find(target_id)]
            united = self._find(source_id) == self._find(target_id)
        except ProgramRepairBoundsError:
            return self._receipt(
                source=source_t,
                target=target_t,
                status=EqualityRewriteStatus.BUDGET_EXHAUSTED,
                reason_code=ProgramRepairReason.BOUNDS_EXCEEDED.value,
                depth=depth,
            )
        except ProgramRepairSynthesisError as exc:
            status = (
                EqualityRewriteStatus.INVALID
                if exc.reason_code
                in {
                    ProgramRepairReason.EQUALITY_TYPE_MISMATCH.value,
                    ProgramRepairReason.EQUALITY_EFFECT_CHANGE.value,
                    ProgramRepairReason.EQUALITY_INVALID_REWRITE.value,
                }
                else EqualityRewriteStatus.UNSUPPORTED
            )
            return self._receipt(
                source=source_t,
                target=target_t,
                status=status,
                reason_code=exc.reason_code,
                depth=depth,
            )

        if not united:
            if self._depth_budget_exhausted:
                status = EqualityRewriteStatus.BUDGET_EXHAUSTED
                reason = ProgramRepairReason.BOUNDS_EXCEEDED.value
            else:
                status = EqualityRewriteStatus.UNPROVED
                reason = ProgramRepairReason.EQUALITY_UNPROVED.value
            return self._receipt(
                source=source_t,
                target=target_t,
                status=status,
                reason_code=reason,
                depth=depth,
                extracted_term=extracted.render(),
                extraction_cost=cost,
                source_sort=source_sort,
                target_sort=target_sort,
            )

        eq_status, eq_reason = _independent_equivalence_check(
            self.theory,
            source_term,
            target_term,
            steps=tuple(self._steps),
            applied_rule_ids=tuple(dict.fromkeys(self._applied)),
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
        )
        effect_status, effect_reason = _independent_effect_check(
            self.theory, source_term, target_term
        )
        extracted_effect_status, extracted_effect_reason = _independent_effect_check(
            self.theory, source_term, extracted
        )
        if extracted_effect_status.startswith("passed"):
            pass
        else:
            effect_status, effect_reason = extracted_effect_status, extracted_effect_reason
        if not eq_status.startswith("passed"):
            return self._receipt(
                source=source_t,
                target=target_t,
                status=EqualityRewriteStatus.INVALID,
                reason_code=ProgramRepairReason.EQUALITY_INDEPENDENT_REJECT.value,
                depth=depth,
                extracted_term=extracted.render(),
                extraction_cost=cost,
                source_sort=source_sort,
                target_sort=target_sort,
                independent_equivalence=eq_reason,
                independent_effect=effect_status,
            )
        if not effect_status.startswith("passed"):
            return self._receipt(
                source=source_t,
                target=target_t,
                status=EqualityRewriteStatus.INVALID,
                reason_code=ProgramRepairReason.EQUALITY_EFFECT_CHANGE.value,
                depth=depth,
                extracted_term=extracted.render(),
                extraction_cost=cost,
                source_sort=source_sort,
                target_sort=target_sort,
                independent_equivalence=eq_status,
                independent_effect=effect_reason,
            )
        return self._receipt(
            source=source_t,
            target=target_t,
            status=EqualityRewriteStatus.PROVED,
            reason_code=ProgramRepairReason.EQUALITY_PROVED.value,
            depth=depth,
            extracted_term=extracted.render(),
            extraction_cost=cost,
            source_sort=source_sort,
            target_sort=target_sort,
            independent_equivalence=eq_status,
            independent_effect=effect_status,
        )


def _independent_effect_check(
    theory: DeclaredEqualityTheory,
    source: EqualityTerm,
    target: EqualityTerm,
) -> tuple[str, str]:
    source_effects = _collect_term_effects(source, theory.operator_effects)
    target_effects = _collect_term_effects(target, theory.operator_effects)
    # An equality proof may retain effects already present in its context,
    # but it may not introduce one.  ``allowed_effects`` controls theory
    # declaration review; it is not an exemption from semantic preservation.
    extra = set(target_effects) - set(source_effects)
    if extra & _FORBIDDEN_EFFECT_LABELS:
        reason = "forbidden_effect:" + ",".join(sorted(extra & _FORBIDDEN_EFFECT_LABELS))
        return "failed", reason
    if extra:
        reason = "undeclared_effect:" + ",".join(sorted(extra))
        return "failed", reason
    return "passed:effects_contained", "passed:effects_contained"


def _ast_replay(
    source: EqualityTerm,
    steps: Sequence[EqualityRewriteStep],
    theory: DeclaredEqualityTheory,
    *,
    max_depth: int,
) -> EqualityTerm:
    rules = theory.rule_map()
    term = source
    for step in steps:
        rule = rules.get(step.rule_id)
        if rule is None:
            raise ProgramRepairSynthesisError(
                f"replay step references unknown rule {step.rule_id}",
                reason_code=ProgramRepairReason.EQUALITY_REPLAY_FAILED,
            )
        if step.review_ref != rule.review_ref:
            raise ProgramRepairSynthesisError(
                f"replay step review provenance does not match {step.rule_id}",
                reason_code=ProgramRepairReason.EQUALITY_REPLAY_FAILED,
            )
        if rule.review_ref not in theory.review_refs or not rule.oriented:
            raise ProgramRepairSynthesisError(
                f"replay step is not reviewed and oriented for {step.rule_id}",
                reason_code=ProgramRepairReason.EQUALITY_REPLAY_FAILED,
            )
        lhs_pattern = rule.parsed_lhs()
        rhs_pattern = rule.parsed_rhs()
        if step.lhs != rule.lhs or step.rhs != rule.rhs:
            raise ProgramRepairSynthesisError(
                f"replay step terms do not match rule {step.rule_id}",
                reason_code=ProgramRepairReason.EQUALITY_REPLAY_FAILED,
            )
        substitution = {
            name: parse_equality_term(value, name="subst_term")
            for name, value in step.substitution
        }
        expected_vars = lhs_pattern.pattern_vars()
        if set(substitution) != expected_vars:
            raise ProgramRepairSynthesisError(
                f"replay substitution does not bind rule variables for {step.rule_id}",
                reason_code=ProgramRepairReason.EQUALITY_REPLAY_FAILED,
            )
        if _unbound_rhs_pattern_vars(lhs_pattern, rhs_pattern, substitution):
            raise ProgramRepairSynthesisError(
                f"replay step has an unbound RHS variable for {step.rule_id}",
                reason_code=ProgramRepairReason.EQUALITY_REPLAY_FAILED,
            )
        lhs = _instantiate_equality_term(lhs_pattern, substitution)
        rhs = _instantiate_equality_term(rhs_pattern, substitution)
        term, _applications = _rewrite_equality_term_all(term, lhs, rhs)
    return term


def _collect_equality_subterms(*terms: EqualityTerm) -> dict[str, EqualityTerm]:
    """Index every ground subterm by its rendered S-expression."""

    out: dict[str, EqualityTerm] = {}
    stack = list(terms)
    while stack:
        current = stack.pop()
        if current.is_var:
            continue
        key = current.render()
        if key in out:
            continue
        out[key] = current
        stack.extend(current.children)
    return out


class _GroundTermUnion:
    """String-keyed union-find used by the independent congruence checker."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, term: str) -> None:
        self._parent.setdefault(term, term)

    def find(self, term: str) -> str:
        self.add(term)
        parent = self._parent[term]
        if parent != term:
            root = self.find(parent)
            self._parent[term] = root
            return root
        return term

    def union(self, left: str, right: str) -> None:
        root_a = self.find(left)
        root_b = self.find(right)
        if root_a != root_b:
            self._parent[root_b] = root_a

    def equivalent(self, left: str, right: str) -> bool:
        return self.find(left) == self.find(right)


def _close_ground_congruence(
    terms: Mapping[str, EqualityTerm],
    union: _GroundTermUnion,
    *,
    max_passes: int,
) -> None:
    rendered = list(terms)
    for term in rendered:
        union.add(term)
    passes = 0
    progressed = True
    while progressed:
        passes += 1
        if passes > max_passes:
            raise ProgramRepairBoundsError(
                "independent congruence closure budget exhausted",
                reason_code=ProgramRepairReason.BOUNDS_EXCEEDED,
            )
        progressed = False
        for index, left_key in enumerate(rendered):
            left = terms[left_key]
            for right_key in rendered[index + 1 :]:
                right = terms[right_key]
                if left.op != right.op or len(left.children) != len(right.children):
                    continue
                if union.equivalent(left_key, right_key):
                    continue
                if all(
                    union.equivalent(child.render(), other.render())
                    for child, other in zip(left.children, right.children)
                ):
                    union.union(left_key, right_key)
                    progressed = True


def _independent_congruence_holds(
    theory: DeclaredEqualityTheory,
    source: EqualityTerm,
    target: EqualityTerm,
    steps: Sequence[EqualityRewriteStep],
    *,
    max_nodes: int,
) -> bool:
    """Decide source ≡ target from reviewed steps plus congruence, without the e-graph."""

    terms = _collect_equality_subterms(source, target)
    union = _GroundTermUnion()
    for term in terms:
        union.add(term)
    rules = theory.rule_map()
    for step in steps:
        rule = rules.get(step.rule_id)
        if rule is None:
            return False
        substitution = {
            name: parse_equality_term(value, name="subst_term")
            for name, value in step.substitution
        }
        lhs = _instantiate_equality_term(rule.parsed_lhs(), substitution)
        rhs = _instantiate_equality_term(rule.parsed_rhs(), substitution)
        for extra in _collect_equality_subterms(lhs, rhs).values():
            key = extra.render()
            if key not in terms:
                terms[key] = extra
            union.add(key)
        union.union(lhs.render(), rhs.render())
    max_passes = max(8, min(max_nodes, len(terms) * len(terms) + 1))
    _close_ground_congruence(terms, union, max_passes=max_passes)
    return union.equivalent(source.render(), target.render())


def _independent_equivalence_check(
    theory: DeclaredEqualityTheory,
    source: EqualityTerm,
    target: EqualityTerm,
    *,
    steps: Sequence[EqualityRewriteStep],
    applied_rule_ids: Sequence[str],
    max_depth: int,
    max_nodes: int,
) -> tuple[str, str]:
    try:
        replayed = _ast_replay(source, steps, theory, max_depth=max_depth)
    except ProgramRepairSynthesisError as exc:
        return "failed", getattr(exc, "reason_code", ProgramRepairReason.EQUALITY_REPLAY_FAILED.value)
    if replayed == target:
        return "passed:replay", "passed:replay"
    step_rules = {step.rule_id for step in steps}
    if any(rule_id not in step_rules for rule_id in applied_rule_ids):
        return "failed", "applied_rule_missing_from_replay"
    try:
        if _independent_congruence_holds(
            theory, source, target, steps, max_nodes=max_nodes
        ):
            return "passed:congruence_closure", "passed:congruence_closure"
    except ProgramRepairBoundsError as exc:
        return "failed", getattr(
            exc, "reason_code", ProgramRepairReason.BOUNDS_EXCEEDED.value
        )
    except ProgramRepairSynthesisError as exc:
        return "failed", getattr(
            exc, "reason_code", ProgramRepairReason.EQUALITY_REPLAY_FAILED.value
        )
    return "failed", ProgramRepairReason.EQUALITY_REPLAY_FAILED.value


def replay_equality_rewrites(
    source_term: str,
    steps: Sequence[EqualityRewriteStep | Mapping[str, Any]],
    theory: DeclaredEqualityTheory | Mapping[str, Any],
    *,
    max_depth: int = DEFAULT_MAX_REWRITE_DEPTH,
) -> str:
    """Replay recorded reviewed rewrites on an AST, independent of e-graph state."""

    if isinstance(theory, Mapping):
        theory = DeclaredEqualityTheory.from_dict(theory)
    if not isinstance(theory, DeclaredEqualityTheory):
        raise ProgramRepairSynthesisError("theory must be DeclaredEqualityTheory")
    source = parse_equality_term(source_term, name="source_term")
    normalized = _coerce_rewrite_steps(steps)
    replayed = _ast_replay(
        source,
        normalized,
        theory,
        max_depth=_positive_int(max_depth, "max_depth", maximum=MAX_REWRITE_STEPS),
    )
    return replayed.render()


def extract_under_equality_theory(
    theory: DeclaredEqualityTheory | Mapping[str, Any],
    term: str,
    *,
    max_depth: int = DEFAULT_MAX_REWRITE_DEPTH,
    max_nodes: int = MAX_EGRAPH_NODES,
) -> tuple[str, int]:
    """Saturate ``term`` under ``theory`` and extract the cheapest representative."""

    if isinstance(theory, Mapping):
        theory = DeclaredEqualityTheory.from_dict(theory)
    if not isinstance(theory, DeclaredEqualityTheory):
        raise ProgramRepairSynthesisError("theory must be DeclaredEqualityTheory")
    graph = EqualityEGraph(theory, max_depth=max_depth, max_nodes=max_nodes)
    graph.add_term(term)
    graph.saturate()
    return graph.extract(term)


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


_CEGIS_FORBIDDEN_PARAMETER_KEYS: Final[frozenset[str]] = CEGIS_FORBIDDEN_PARAMETER_KEYS


@dataclass(frozen=True)
class ProgramRepairCounterevidence:
    """Independently obtained cores, assumptions, and interpolants for CEGIS."""

    unsat_core_refs: tuple[str, ...] = ()
    failed_assumption_refs: tuple[str, ...] = ()
    interpolant_refs: tuple[str, ...] = ()
    interpolants_independently_validated: bool = False
    effect_refs: tuple[str, ...] = ()
    security_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "unsat_core_refs",
            _ids(self.unsat_core_refs, "unsat_core_refs"),
        )
        object.__setattr__(
            self,
            "failed_assumption_refs",
            _ids(self.failed_assumption_refs, "failed_assumption_refs"),
        )
        object.__setattr__(
            self,
            "interpolant_refs",
            _ids(self.interpolant_refs, "interpolant_refs"),
        )
        object.__setattr__(
            self,
            "effect_refs",
            _ids(self.effect_refs, "effect_refs"),
        )
        object.__setattr__(
            self,
            "security_refs",
            _ids(self.security_refs, "security_refs"),
        )
        object.__setattr__(
            self,
            "interpolants_independently_validated",
            _bool(
                self.interpolants_independently_validated,
                "interpolants_independently_validated",
            ),
        )

    def evidence_tags(self) -> tuple[str, ...]:
        return (
            self.unsat_core_refs
            + self.failed_assumption_refs
            + self.interpolant_refs
        )


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
    counterevidence: ProgramRepairCounterevidence | None = None
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
        if self.counterevidence is not None:
            evidence = self.counterevidence
            if isinstance(evidence, Mapping):
                evidence = ProgramRepairCounterevidence(
                    unsat_core_refs=tuple(evidence.get("unsat_core_refs") or ()),
                    failed_assumption_refs=tuple(
                        evidence.get("failed_assumption_refs") or ()
                    ),
                    interpolant_refs=tuple(evidence.get("interpolant_refs") or ()),
                    interpolants_independently_validated=bool(
                        evidence.get("interpolants_independently_validated")
                    ),
                    effect_refs=tuple(evidence.get("effect_refs") or ()),
                    security_refs=tuple(evidence.get("security_refs") or ()),
                )
            if not isinstance(evidence, ProgramRepairCounterevidence):
                raise ProgramRepairSynthesisError(
                    "counterevidence must be ProgramRepairCounterevidence"
                )
            object.__setattr__(self, "counterevidence", evidence)
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
            if receipt.status is EqualityRewriteStatus.BUDGET_EXHAUSTED:
                disposition = ProgramRepairDisposition.BUDGET_EXHAUSTED
            else:
                disposition = ProgramRepairDisposition.ABSTAIN
            return ProgramRepairReceipt(
                disposition=disposition,
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

    def _admitted_cegis_operator_kinds(
        self, request: ProgramRepairRequest
    ) -> tuple[str, ...] | None:
        requested = tuple(request.operator_kinds)
        evidence = request.counterevidence
        if evidence is None:
            return requested
        if evidence.interpolant_refs and not evidence.interpolants_independently_validated:
            return None
        tags = evidence.evidence_tags()
        if tags:
            matched = tuple(
                kind
                for kind in requested
                if any(kind in tag for tag in tags)
            )
            requested = matched
        restricted = evidence.effect_refs + evidence.security_refs
        if restricted:
            sensitive = cegis_restricted_operator_kinds(self._registry)
            requested = tuple(
                kind
                for kind in requested
                if kind not in sensitive
                or any(kind in tag for tag in restricted)
            )
        return requested

    def _cegis_candidate_is_restricted(
        self,
        candidate: RefinementCandidate,
        admitted_kinds: Sequence[str] = (),
    ) -> str | None:
        parameters = candidate.parameters or {}
        if not isinstance(parameters, Mapping):
            return ProgramRepairReason.MALFORMED_INPUT.value
        operator_kind = str(parameters.get("operator_kind") or "").strip()
        if admitted_kinds and operator_kind and operator_kind not in admitted_kinds:
            return ProgramRepairReason.COUNTEREVIDENCE_RESTRICTED.value
        for key in _CEGIS_FORBIDDEN_PARAMETER_KEYS:
            value = parameters.get(key)
            if value not in (None, False, (), [], ""):
                if key in {"extra_imports"}:
                    return ProgramRepairReason.EXTRA_IMPORT.value
                if key in {"extra_paths", "extra_files", "files_added"}:
                    return ProgramRepairReason.EXTRA_FILE.value
                if key in {"extra_dependencies"}:
                    return ProgramRepairReason.EXTRA_DEPENDENCY.value
                if key in {"write_authority", "authority"}:
                    return ProgramRepairReason.AUTHORITY_CLAIM.value
                return ProgramRepairReason.UNDECLARED_EFFECT.value
        return None

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
        admitted_kinds = self._admitted_cegis_operator_kinds(request)
        if admitted_kinds is None:
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.ABSTAIN,
                reason_codes=(
                    ProgramRepairReason.UNVALIDATED_INTERPOLANT.value,
                    ProgramRepairReason.PROPOSAL_ONLY.value,
                    ProgramRepairReason.ZERO_MODEL_CALLS.value,
                ),
                roots=request.roots,
                mode=ProgramRepairMode.CEGIS,
                bounds=request.bounds,
            )
        if request.operator_kinds and not admitted_kinds:
            return ProgramRepairReceipt(
                disposition=ProgramRepairDisposition.ABSTAIN,
                reason_codes=(
                    ProgramRepairReason.COUNTEREVIDENCE_RESTRICTED.value,
                    ProgramRepairReason.NO_ADMISSIBLE_OPERATOR.value,
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
                restricted = self._cegis_candidate_is_restricted(
                    candidate, admitted_kinds
                )
                if restricted is not None:
                    return CandidateValidationStatus.INVALID, restricted
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
        else:
            inner_validate = validate

            def validate(candidate: RefinementCandidate, context: Mapping[str, Any]):
                restricted = self._cegis_candidate_is_restricted(
                    candidate, admitted_kinds
                )
                if restricted is not None:
                    return CandidateValidationStatus.INVALID, restricted
                return inner_validate(candidate, context)

        refine = request.cegis_refine
        if refine is None and admitted_kinds:

            def refine(witness: FormalCounterexample, context: Mapping[str, Any]):
                del witness
                out: list[RefinementCandidate] = []
                evidence = request.counterevidence
                tags = evidence.evidence_tags() if evidence is not None else ()
                for index, kind in enumerate(
                    admitted_kinds[: budget.max_candidates_per_iteration]
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
                                "counterevidence_tags": list(tags),
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
                    "operator_kinds": list(admitted_kinds),
                    "requested_operator_kinds": list(request.operator_kinds),
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
    "EqualityFeatureStatus",
    "EqualityRewriteReceipt",
    "EqualityRewriteStatus",
    "EqualityRewriteStep",
    "EqualityRule",
    "EqualitySaturationCapability",
    "EqualityTerm",
    "equality_saturation_capabilities",
    "extract_under_equality_theory",
    "parse_equality_term",
    "replay_equality_rewrites",
    "HybridUsageReceipt",
    "ProgramRepairAuthorityError",
    "ProgramRepairBounds",
    "ProgramRepairBoundsError",
    "ProgramRepairCandidate",
    "ProgramRepairCounterevidence",
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
